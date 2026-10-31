use std::collections::HashMap;
use std::marker::PhantomData;

use atomic_systems::interactions::pec::PecScalingType;
use cc_calculations::calculations::bound_states::{
    BoundSearchMethod,
    NodeMonotony,
    NodeRangeTarget,
    RegionParams,
    modify_from_f64,
    modify_to_f64,
};
use cc_calculations::calculations::dependence::DependenceCalc;
use cc_calculations::calculations::levels::EnergyLevelsData;
use cc_calculations::calculations::modifications::{
    FloatCalcMod,
    ModificationAction,
    ModifyParam,
    ModifyRegistry,
    NumberCalcMod,
    ScalarCalcMod,
    ScalarParamMod,
};
use cc_calculations::calculations::{
    Modified,
    SingleCalc,
};
use cc_calculations::{
    InputData,
    modify_recipe,
};
use cc_math_utils::brent_root_method;
use cc_system::hamiltonian_spec::HamiltonianSpec;
use cc_system::orbital_basis::{
    OrbitalBasis,
    OrbitalBasisElements,
    OrbitalRecipe,
};
use cc_system::param_ids;
use cc_system::param_modifications::ParamModifications;
use cc_system::parameters::{
    Parameters,
    TypedParamId,
};
use coupled_chan::cc_propagator::Direction;
use coupled_chan::single_chan::airy::AiryLogDerivative;
use coupled_chan::single_chan::log_derivative::LogDerivative;
use coupled_chan::single_chan::s_matrix::SValue;
use coupled_chan::single_chan::{
    SingleChanLogDerivPropagator,
    WFunction,
};
use coupled_chan::{
    CollisionWFunction,
    DynInteraction,
    Interaction,
};
use hilbert_space::faer::complex::Complex64;
use hilbert_space::space::SpaceBasis;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Number;
use unit_systems::atomic_units;
use unit_systems::quantities::phys_quantities::{
    Energy,
    Length,
    Mass,
};
use unit_systems::quantities::{
    Power,
    Scalar,
};

use crate::problems::Problem;
use atomic_systems::interactions::Interactions;
use atomic_systems::interactions::pec::PecScaling;
use cc_calculations::calculations::DynCalc;
use cc_calculations::calculations::scattering::{
    ConstRefStep,
    LinearRefStep,
};

#[derive(Default)]
pub struct SingleChanProblem {
    pub calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>>,
}

impl SingleChanProblem {
    pub fn new() -> Self {
        let mut calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>> = HashMap::default();
        calculations.extend([
            ("pec scan".into(), pec_values_calc()),
            ("scattering scan".into(), single_chan_scattering_problem()),
            ("bound states scan".into(), single_chan_bound_states_problem()),
        ]);

        Self { calculations }
    }
}

impl Problem for SingleChanProblem {
    type BasisRecipe = SingleChanBasis;
    type Params = SingleChanParams;

    const NAME: &str = "single channel";

    const DESCRIPTION: &str = "Single channel problem";

    fn build(basis_recipe: &Self::BasisRecipe, _params: &Self::Params) -> HamiltonianSpec {
        let mut basis = SpaceBasis::default();

        let l = OrbitalRecipe::Single(basis_recipe.l);
        let l = OrbitalBasis::new(l, &mut basis);

        let elements = basis.get_basis();
        let elements = OrbitalBasisElements::from_orbital(elements, &l);

        HamiltonianSpec::new(elements)
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}

#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SingleChanBasis {
    l: u32,
}

#[derive(Debug, Clone, Deserialize, cc_system::Parameters, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SingleChanParams {
    mass: Scalar<Mass>,
    pec: Interactions,

    #[serde(default)]
    pec_scaling: PecScaling,
}

fn params_calc_mods<C: InputData + 'static>() -> ModifyRegistry<SingleChanProblem, C> {
    let ids = SingleChanParams::ids();

    ModifyRegistry::from([
        ("mass", modify_recipe!(|m| ScalarParamMod::new(m, ids.mass))),
        (
            "l",
            modify_recipe!(|l| NumberCalcMod::new(l, |r: &mut SingleChanBasis, v| r.l = v)),
        ),
        ("pec_scaling", modify_recipe!(|s| ScalingMod::new(s, ids.pec_scaling))),
    ])
}

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
struct ScatteringCalcInput {
    energy: Scalar<Energy>,

    r_min: Scalar<Length>,
    r_max: Scalar<Length>,
    #[serde(default)]
    boundary: SingleBoundary,

    propagator: SingleChanPropagator,
}

fn pec_values_calc() -> Box<dyn DynCalc<SingleChanProblem>> {
    let scattering = PecValues;
    let registry = ModifyRegistry::from([(
        "distance",
        modify_recipe!(|d| ScalarCalcMod::new(d, |r: &mut PecValuesInput, v| r.distance = v)),
    )]);

    Box::new(DependenceCalc::new(scattering, registry))
}

#[derive(Clone, Default, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
#[serde(default)]
struct PecValuesInput {
    distance: Scalar<Length>,
}

struct PecValues;

impl SingleCalc for PecValues {
    type P = SingleChanProblem;
    type CalcInput = PecValuesInput;
    type Data = EnergyLevelsData;

    fn calculate(
        &self,
        modified: &mut Modified<Self::P, PecValuesInput>,
        _problem: &Self::P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        let converter = atomic_units::units_converter();
        let ids = SingleChanParams::ids();
        let registry = modified.system.param_registry();

        let mass = converter.scalar_value(registry.get(ids.mass));
        let w_function = CollisionWFunction::new(
            registry.get(ids.pec).scaled_interactions(*registry.get(ids.pec_scaling)),
            mass,
            0.,
            modified.basis.l,
        );
        let value = w_function.value(converter.scalar_value(&modified.calc_input.distance));

        [Ok(EnergyLevelsData(vec![value / (-2. * mass)]))]
    }
}

fn single_chan_scattering_problem() -> Box<dyn DynCalc<SingleChanProblem>> {
    let scattering = SingleChanScattering;
    let registry = params_calc_mods().extend(ModifyRegistry::from([
        (
            "energy",
            modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut ScatteringCalcInput, v| r.energy = v)),
        ),
        (
            "r_min",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.r_min = v)),
        ),
        (
            "r_max",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.r_max = v)),
        ),
        (
            "step_scaling",
            modify_recipe!(|x| FloatCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.propagator.scale_step(v))),
        ),
    ]));

    Box::new(DependenceCalc::new(scattering, registry))
}

struct SingleChanScattering;

impl SingleCalc for SingleChanScattering {
    type P = SingleChanProblem;
    type CalcInput = ScatteringCalcInput;
    type Data = SValueData;

    fn calculate(
        &self,
        modified: &mut Modified<Self::P, ScatteringCalcInput>,
        _problem: &Self::P,
    ) -> impl IntoIterator<Item = anyhow::Result<SValueData>> {
        let converter = atomic_units::units_converter();
        let ids = SingleChanParams::ids();
        let registry = modified.system.param_registry();

        let w_function = CollisionWFunction::new(
            registry.get(ids.pec).scaled_interactions(*registry.get(ids.pec_scaling)),
            converter.scalar_value(registry.get(ids.mass)),
            converter.scalar_value(&modified.calc_input.energy),
            modified.basis.l,
        );

        let boundary =
            modified
                .calc_input
                .boundary
                .get_boundary(&modified.calc_input.r_min, Direction::Outwards, &w_function);
        let r_max = converter.scalar_value(&modified.calc_input.r_max);

        let mut prop = modified.calc_input.propagator.get_propagator(boundary, &w_function);
        let s_value = SValue::from_log_deriv(prop.propagate_to(r_max), &w_function);

        let s_value = SValueData {
            s_value: s_value.value,
            momentum: s_value.momentum,
            s_length: s_value.scattering_length().re,
            cross_section: s_value.elastic_cross_sect(),
        };

        [Ok(s_value)]
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct SValueData {
    s_value: Complex64,
    momentum: f64,
    s_length: f64,
    cross_section: f64,
}

fn single_chan_bound_states_problem() -> Box<dyn DynCalc<SingleChanProblem>> {
    let ids = SingleChanParams::ids();
    let registry = params_calc_mods().extend(ModifyRegistry::from([
        (
            "energy",
            modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut BoundStateCalcInput, v| r.energy = v)),
        ),
        ("mass", modify_recipe!(|m| ScalarParamMod::new(m, ids.mass))),
        ("pec_scaling", modify_recipe!(|s| ScalingMod::new(s, ids.pec_scaling))),
    ]));

    let bound = BoundStateCalc { registry };
    let registry = params_calc_mods().extend(ModifyRegistry::from([
        (
            "energy",
            modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut BoundStateCalcInput, v| r.energy = v)),
        ),
        (
            "r_min",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut BoundStateCalcInput, v| r.r_min = v)),
        ),
        (
            "r_max",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut BoundStateCalcInput, v| r.r_max = v)),
        ),
        (
            "step_scaling",
            modify_recipe!(|x| FloatCalcMod::new(x, |r: &mut BoundStateCalcInput, v| r.propagator.scale_step(v))),
        ),
    ]));

    Box::new(DependenceCalc::new(bound, registry))
}

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
struct BoundStateCalcInput {
    #[serde(default)]
    energy: Scalar<Energy>,

    dependant: RegionParams,

    #[serde(default)]
    boundaries: (SingleBoundary, SingleBoundary),
    r_min: Scalar<Length>,
    r_match: Scalar<Length>,
    r_max: Scalar<Length>,

    /// Coupled channel propagator to use,
    /// inwards propagation can be overridden by
    /// optional `propagator_in` field.
    propagator: SingleChanPropagator,
    #[serde(default)]
    /// Optional overwrite of coupled channel propagator for inwards propagation
    propagator_in: Option<SingleChanPropagator>,

    #[serde(default)]
    get_wave_function: bool,

    #[serde(default)]
    node_monotony: NodeMonotony,
    node_range: Option<NodeRangeTarget>,

    #[serde(default)]
    search_method: BoundSearchMethod,
}

struct BoundStateCalc {
    registry: ModifyRegistry<SingleChanProblem, BoundStateCalcInput>,
}

impl SingleCalc for BoundStateCalc {
    type P = SingleChanProblem;
    type CalcInput = BoundStateCalcInput;
    type Data = BoundStateData;

    fn validate_input(&self, calc_input: &Self::CalcInput) -> anyhow::Result<()> {
        let max = &calc_input.dependant.max.name;
        let min = &calc_input.dependant.min.name;
        let err = &calc_input.dependant.err.name;

        if max != min || max != err {
            anyhow::bail!(
                "Bound state dependant have incompatible min, max, err bounds: {} {} {}",
                min,
                max,
                err
            );
        }

        let keys: Vec<&Box<str>> = self.registry.0.keys().collect();
        if let None = keys.iter().find(|&x| *x == max) {
            anyhow::bail!("Dependant have unknown dependence type - got: {}, available {:?}", max, keys);
        }

        Ok(())
    }

    fn calculate(
        &self,
        modified: &mut Modified<Self::P, BoundStateCalcInput>,
        _problem: &Self::P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        let converter = atomic_units::units_converter();
        let r_start = converter.scalar_value(&modified.calc_input.r_min);
        let r_match = converter.scalar_value(&modified.calc_input.r_match);
        let r_stop = converter.scalar_value(&modified.calc_input.r_max);
        assert!(r_start < r_match && r_match < r_stop, "expected r_start < r_match < r_stop");

        assert!(
            modified.calc_input.dependant.is_compatible(),
            "incompatible dependant fields in calc input"
        );
        let modifier = &self.registry.0[&modified.calc_input.dependant.min.name];
        let p_start = (modifier.recipe)(modified.calc_input.dependant.min.value.clone());
        let p_end = (modifier.recipe)(modified.calc_input.dependant.max.value.clone());
        let p_start_f64 = modify_to_f64(&*p_start);
        let p_end_f64 = modify_to_f64(&*p_end);

        p_start.modify(modified);
        let w_function = self.get_w_function(modified);
        let mut lower_mismatch = bound_mismatch(&w_function, modified.calc_input, p_start_f64);

        p_end.modify(modified);
        let w_function = self.get_w_function(modified);
        let mut upper_mismatch = bound_mismatch(&w_function, modified.calc_input, p_end_f64);

        if lower_mismatch.nodes > upper_mismatch.nodes {
            std::mem::swap(&mut lower_mismatch, &mut upper_mismatch)
        }

        let mut upper_node = upper_mismatch.nodes;
        let mut lower_node = lower_mismatch.nodes;

        if let Some(nodes_range) = modified.calc_input.node_range {
            match nodes_range {
                NodeRangeTarget::Range(a, b) => {
                    lower_node = lower_node.max(a);
                    upper_node = upper_node.min(b + 1);
                }
                NodeRangeTarget::BottomRange(a) => upper_node = upper_node.min(lower_node + a),
                NodeRangeTarget::TopRange(a) => lower_node = lower_node.max(upper_node.saturating_sub(a)),
            }
        }
        let states_no = (upper_node - lower_node) as usize;

        let mut lower_bounds = vec![None; states_no];
        let mut upper_bounds = vec![None; states_no];
        // if no nodes found skip this part
        if !lower_bounds.is_empty() {
            lower_bounds[0] = Some(lower_mismatch);
            upper_bounds[states_no - 1] = Some(upper_mismatch);
        }

        let nodes: Vec<u64> = match modified.calc_input.node_monotony {
            NodeMonotony::Increasing => (lower_node..upper_node).collect(),
            NodeMonotony::Decreasing => (lower_node..upper_node).rev().collect(),
        };

        let mut p_mod = p_start;
        nodes.into_iter().map(move |target_node| {
            let p = match modified.calc_input.search_method {
                BoundSearchMethod::Brent(max_iter) => self.brent_search(
                    modified,
                    &mut lower_bounds,
                    &mut upper_bounds,
                    lower_node,
                    target_node,
                    max_iter,
                ),
                BoundSearchMethod::Bisection => {
                    Ok(self.bisection_search(modified, &mut lower_bounds, &mut upper_bounds, lower_node, target_node))
                }
            };

            p.map(|p| {
                let mut wave_function = None;

                if modified.calc_input.get_wave_function {
                    modify_from_f64(&mut *p_mod, p);
                    p_mod.modify(modified);
                    let w_function = self.get_w_function(modified);
                    let wave = bound_wave(&w_function, modified.calc_input);

                    if modified.calc_input.get_wave_function {
                        wave_function = Some(wave)
                    }
                }

                BoundStateData {
                    nodes: target_node,
                    parameter: p,

                    wave_function,
                }
            })
        })
    }
}

impl BoundStateCalc {
    fn get_w_function(
        &self,
        modified: &Modified<SingleChanProblem, BoundStateCalcInput>,
    ) -> CollisionWFunction<DynInteraction> {
        let converter = atomic_units::units_converter();
        let ids = SingleChanParams::ids();
        let registry = modified.system.param_registry();

        CollisionWFunction::new(
            registry.get(ids.pec).scaled_interactions(*registry.get(ids.pec_scaling)),
            converter.scalar_value(registry.get(ids.mass)),
            converter.scalar_value(&modified.calc_input.energy),
            modified.basis.l,
        )
    }

    fn brent_search(
        &self,
        modified: &mut Modified<SingleChanProblem, BoundStateCalcInput>,
        lower_bounds: &mut [Option<BoundMismatch>],
        upper_bounds: &mut [Option<BoundMismatch>],
        min_nodes: u64,
        target_nodes: u64,
        max_iter: u32,
    ) -> anyhow::Result<f64> {
        let n = lower_bounds.len();
        let node_index = (target_nodes - min_nodes) as usize;
        let modifier = &self.registry.0[&modified.calc_input.dependant.err.name];
        let mut p_mod = (modifier.recipe)(modified.calc_input.dependant.err.value.clone());
        let p_err = modify_to_f64(&*p_mod);

        let mut lower_bound = *lower_bounds
            .iter()
            .take(node_index + 1)
            .rfind(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap();

        let mut upper_bound = *upper_bounds
            .iter()
            .skip(node_index)
            .find(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap();

        let index = lower_bound.nodes_match as i64 + target_nodes as i64 - lower_bound.nodes as i64;
        let mut lower_mismatch = if index == 0 && lower_bound.mismatch > 0. {
            Some(lower_bound.mismatch)
        } else {
            None
        };

        let index = upper_bound.nodes_match as i64 + target_nodes as i64 - upper_bound.nodes as i64;
        let mut upper_mismatch = if index == 0 && upper_bound.mismatch < 0. {
            Some(upper_bound.mismatch)
        } else {
            None
        };

        let monotony = upper_bound.parameter > lower_bound.parameter;

        while upper_bound.nodes != target_nodes + 1
            || lower_bound.nodes != target_nodes
            || lower_mismatch.is_none()
            || upper_mismatch.is_none()
        {
            let p_mid = (upper_bound.parameter + lower_bound.parameter) / 2.;

            if (upper_bound.parameter - lower_bound.parameter).abs() < p_err.abs() {
                return Ok(p_mid);
            }

            modify_from_f64(&mut *p_mod, p_mid);
            p_mod.modify(modified);
            let w_matrix = self.get_w_function(modified);
            let mid_mismatch = bound_mismatch(&w_matrix, modified.calc_input, p_mid);

            if mid_mismatch.nodes <= min_nodes
                && ((lower_bounds[0].as_ref().unwrap().parameter < mid_mismatch.parameter) ^ !monotony)
            {
                lower_bounds[0] = Some(mid_mismatch.clone())
            } else if mid_mismatch.nodes >= min_nodes + n as u64
                && ((upper_bounds[n - 1].as_ref().unwrap().parameter > mid_mismatch.parameter) ^ !monotony)
            {
                upper_bounds[n - 1] = Some(mid_mismatch.clone())
            } else {
                let index = (mid_mismatch.nodes - min_nodes) as usize;
                if let Some(lower) = &mut lower_bounds[index]
                    && ((lower.parameter < mid_mismatch.parameter) ^ !monotony)
                {
                    *lower = mid_mismatch.clone()
                } else if lower_bounds[index].is_none() {
                    lower_bounds[index] = Some(mid_mismatch.clone())
                }

                if let Some(upper) = &mut upper_bounds[index - 1]
                    && ((upper.parameter > mid_mismatch.parameter) ^ !monotony)
                {
                    *upper = mid_mismatch.clone()
                } else if upper_bounds[index - 1].is_none() {
                    upper_bounds[index - 1] = Some(mid_mismatch.clone())
                }
            }

            if mid_mismatch.nodes <= target_nodes {
                lower_bound = mid_mismatch;

                let index = lower_bound.nodes_match as i64 + target_nodes as i64 - lower_bound.nodes as i64;
                lower_mismatch = if index == 0 && lower_bound.mismatch > 0. {
                    Some(lower_bound.mismatch)
                } else {
                    None
                };
            } else {
                upper_bound = mid_mismatch;

                let index = upper_bound.nodes_match as i64 + target_nodes as i64 - upper_bound.nodes as i64;
                upper_mismatch = if index == 0 && upper_bound.mismatch < 0. {
                    Some(upper_bound.mismatch)
                } else {
                    None
                };
            }
        }

        Ok(brent_root_method(
            [lower_bound.parameter, lower_mismatch.unwrap()],
            [upper_bound.parameter, upper_mismatch.unwrap()],
            |x| {
                modify_from_f64(&mut *p_mod, x);
                p_mod.modify(modified);
                let w_matrix = self.get_w_function(modified);
                let mismatch = bound_mismatch(&w_matrix, modified.calc_input, x);

                mismatch.mismatch
            },
            p_err,
            max_iter,
        )?)
    }

    fn bisection_search(
        &self,
        modified: &mut Modified<SingleChanProblem, BoundStateCalcInput>,
        lower_bounds: &mut [Option<BoundMismatch>],
        upper_bounds: &mut [Option<BoundMismatch>],
        min_nodes: u64,
        target_nodes: u64,
    ) -> f64 {
        let n = lower_bounds.len();
        let node_index = (target_nodes - min_nodes) as usize;
        let modifier = &self.registry.0[&modified.calc_input.dependant.err.name];
        let mut p_mod = (modifier.recipe)(modified.calc_input.dependant.err.value.clone());
        let p_err = modify_to_f64(&*p_mod);

        let mut lower_bound = *lower_bounds
            .iter()
            .take(node_index + 1)
            .rfind(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap();

        let mut upper_bound = *upper_bounds
            .iter()
            .skip(node_index + 1)
            .find(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap();

        let monotony = upper_bound.parameter > lower_bound.parameter;
        while (upper_bound.parameter - lower_bound.parameter).abs() > p_err {
            let field_mid = (upper_bound.parameter + lower_bound.parameter) / 2.;

            modify_from_f64(&mut *p_mod, field_mid);
            p_mod.modify(modified);
            let w_matrix = self.get_w_function(modified);
            let mid_mismatch = bound_mismatch(&w_matrix, modified.calc_input, field_mid);

            if mid_mismatch.nodes <= min_nodes
                && ((lower_bounds[0].as_ref().unwrap().parameter < mid_mismatch.parameter) ^ !monotony)
            {
                lower_bounds[0] = Some(mid_mismatch.clone())
            } else if mid_mismatch.nodes >= min_nodes + n as u64
                && ((upper_bounds[n - 1].as_ref().unwrap().parameter > mid_mismatch.parameter) ^ !monotony)
            {
                upper_bounds[n - 1] = Some(mid_mismatch.clone())
            } else {
                let index = (mid_mismatch.nodes - min_nodes) as usize;
                if let Some(lower) = &mut lower_bounds[index]
                    && ((lower.parameter < mid_mismatch.parameter) ^ !monotony)
                {
                    *lower = mid_mismatch.clone()
                } else if lower_bounds[index].is_none() {
                    lower_bounds[index] = Some(mid_mismatch.clone())
                }

                if let Some(upper) = &mut upper_bounds[index - 1]
                    && ((upper.parameter > mid_mismatch.parameter) ^ !monotony)
                {
                    *upper = mid_mismatch.clone()
                } else if upper_bounds[index - 1].is_none() {
                    upper_bounds[index - 1] = Some(mid_mismatch.clone())
                }
            }

            if mid_mismatch.nodes > target_nodes {
                upper_bound = mid_mismatch;
            } else {
                lower_bound = mid_mismatch;
            }
        }

        (upper_bound.parameter + lower_bound.parameter) / 2.
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundMismatch {
    parameter: f64,
    nodes: u64,
    nodes_match: u64,
    mismatch: f64,
}

fn bound_mismatch(
    w_matrix: &CollisionWFunction<impl Interaction>,
    input: &BoundStateCalcInput,
    parameter: f64,
) -> BoundMismatch {
    let boundary_out = input.boundaries.0.get_boundary(&input.r_min, Direction::Outwards, w_matrix);
    let boundary_in = input.boundaries.0.get_boundary(&input.r_max, Direction::Inwards, w_matrix);

    let converter = atomic_units::units_converter();
    let r_match = converter.scalar_value(&input.r_match);

    let mut propagator_in = if let Some(prop_in) = &input.propagator_in {
        prop_in.get_propagator(boundary_in, w_matrix)
    } else {
        input.propagator.get_propagator(boundary_in, w_matrix)
    };

    let sol_in = propagator_in.propagate_to(r_match);

    let mut propagator_out = input.propagator.get_propagator(boundary_out, w_matrix);
    let sol_out = propagator_out.propagate_to(r_match);

    let mismatch = &sol_out.sol.0 - &sol_in.sol.0;
    let nodes = propagator_in.nodes().0 + propagator_out.nodes().0;

    let nodes_match = if mismatch < 0. { 1 } else { 0 };
    let nodes = nodes + nodes_match;

    BoundMismatch {
        parameter,
        nodes,
        nodes_match,
        mismatch,
    }
}

fn bound_wave(w_matrix: &CollisionWFunction<impl Interaction>, input: &BoundStateCalcInput) -> WaveFunction {
    let boundary_out = input.boundaries.0.get_boundary(&input.r_min, Direction::Outwards, w_matrix);
    let boundary_in = input.boundaries.0.get_boundary(&input.r_max, Direction::Inwards, w_matrix);

    let converter = atomic_units::units_converter();
    let r_match = converter.scalar_value(&input.r_match);

    let mut propagator_in = if let Some(prop_in) = &input.propagator_in {
        prop_in.get_propagator(boundary_in, w_matrix)
    } else {
        input.propagator.get_propagator(boundary_in, w_matrix)
    };
    propagator_in.init_wave_storage();
    propagator_in.propagate_to(r_match);

    let mut propagator_out = input.propagator.get_propagator(boundary_out, w_matrix);
    propagator_out.init_wave_storage();
    propagator_out.propagate_to(r_match);

    let wave_in = propagator_in.get_wave_storage().as_ref().unwrap().reconstruct(1., true);
    let wave_in = WaveFunction {
        distances: wave_in.0,
        values: wave_in.1,
    };

    let wave_out = propagator_out.get_wave_storage().as_ref().unwrap().reconstruct(1., true);
    let mut wave_out = WaveFunction {
        distances: wave_out.0,
        values: wave_out.1,
    };

    wave_out.reverse();
    wave_out.distances.extend(wave_in.distances);
    wave_out.values.extend(wave_in.values);

    wave_out.normalize()
}

#[derive(Clone, Debug, Serialize)]
struct BoundStateData {
    nodes: u64,
    parameter: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    wave_function: Option<WaveFunction>,
}

#[derive(Debug, Clone, Serialize)]
struct WaveFunction {
    distances: Vec<f64>,
    values: Vec<f64>,
}

impl WaveFunction {
    fn reverse(&mut self) {
        self.distances.reverse();
        self.values.reverse();
    }

    fn normalize(mut self) -> Self {
        let normalization: f64 = self
            .distances
            .windows(2)
            .zip(self.values.windows(2))
            .map(|(x, f)| unsafe {
                let f1 = f.get_unchecked(1);
                let f0 = f.get_unchecked(0);
                let x1 = x.get_unchecked(1);
                let x0 = x.get_unchecked(0);

                0.5 * (x1 - x0) * (f1 * f1 + f0 * f0)
            })
            .sum();

        for v in &mut self.values {
            *v /= normalization.sqrt()
        }

        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
enum SingleBoundary {
    #[default]
    VanishingWkb,
    Set {
        value: Scalar<Power<Length, -1, 2>>,
        derivative: Scalar<Power<Length, -3, 2>>,
    },
}

impl SingleBoundary {
    fn get_boundary(
        &self,
        r_start: &Scalar<Length>,
        direction: Direction,
        w_function: &impl WFunction,
    ) -> coupled_chan::cc_propagator::Boundary<f64> {
        let converter = atomic_units::units_converter();
        let r = converter.scalar_value(r_start);

        let (value, derivative) = match &self {
            SingleBoundary::VanishingWkb => {
                let value = w_function.value(r);
                let derivative = match direction {
                    Direction::Inwards => -(value).max(0.).sqrt(),
                    Direction::Outwards => value.max(0.).sqrt(),
                };

                (1., derivative)
            }
            SingleBoundary::Set { value, derivative } => {
                let value = converter.scalar_value(value);
                let derivative = converter.scalar_value(derivative);

                (value, derivative)
            }
        };

        coupled_chan::cc_propagator::Boundary {
            r_start: r,
            direction,
            value,
            derivative,
        }
    }
}

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum SingleChanPropagator {
    JohnsonLogDeriv { step: ConstRefStep },
    AiryLogDeriv { step: LinearRefStep },
}

impl SingleChanPropagator {
    fn get_propagator<'a, W: WFunction>(
        &self,
        boundary: coupled_chan::cc_propagator::Boundary<f64>,
        w_function: &'a W,
    ) -> Box<dyn SingleChanLogDerivPropagator + 'a> {
        match self {
            SingleChanPropagator::JohnsonLogDeriv { step } => {
                Box::new(LogDerivative::new(w_function, step.get_step(), boundary))
            }
            SingleChanPropagator::AiryLogDeriv { step } => {
                Box::new(AiryLogDerivative::new(w_function, step.get_step(), boundary))
            }
        }
    }

    fn scale_step(&mut self, scaling: f64) {
        match self {
            SingleChanPropagator::JohnsonLogDeriv { step } => step.scale_step(scaling),
            SingleChanPropagator::AiryLogDeriv { step } => step.scale_step(scaling),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
struct PecScalingInput {
    scaling: f64,
    #[serde(default)]
    scaling_type: PecScalingType,
}

struct ScalingMod<P: Problem, C> {
    scaling: PecScalingInput,
    id: TypedParamId<PecScaling>,
    phantom: PhantomData<(P, C)>,
}

impl<P: Problem, C> ScalingMod<P, C> {
    fn new(scaling: PecScalingInput, id: TypedParamId<PecScaling>) -> Self {
        Self {
            scaling,
            id,
            phantom: PhantomData,
        }
    }
}

impl<P, C> ModifyParam for ScalingMod<P, C>
where
    P: Problem,
    C: Send + Sync,
{
    type P = P;
    type C = C;

    fn prep_modify(&self, _modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        let id = self.id;
        let scaling = self.scaling;

        ModificationAction::ParamModify(
            ParamModifications::new(move |r| {
                r.get_mut(id).modify(scaling.scaling, scaling.scaling_type);

                param_ids![id.vanish()]
            })
            .into_dyn(),
        )
    }

    fn as_number(&self) -> Number {
        Number::from_f64(self.scaling.scaling).unwrap()
    }

    fn mut_number(&mut self, number: Number) {
        let value = number.as_f64().unwrap();
        self.scaling.scaling = value
    }
}
