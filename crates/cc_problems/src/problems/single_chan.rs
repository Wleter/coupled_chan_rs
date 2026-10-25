use std::collections::HashMap;
use std::marker::PhantomData;

use atomic_systems::interactions::pec::PecScalingType;
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
use cc_calculations::modify_recipe;
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
use coupled_chan::CollisionWFunction;
use coupled_chan::cc_propagator::{
    Direction,
    Propagator,
};
use coupled_chan::single_chan::WFunction;
use coupled_chan::single_chan::airy::AiryLogDerivative;
use coupled_chan::single_chan::log_derivative::LogDerivative;
use coupled_chan::single_chan::s_matrix::SValue;
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
            // ("bound states scan".into(), diatom_bound_states_b_field_scan()),
        ]);

        Self { calculations }
    }
}

impl Problem for SingleChanProblem {
    type BasisRecipe = SingleChanBasis;
    type Params = SingleChanParams;

    const NAME: &str = "single channel";

    const DESCRIPTION: &str = "Single channel problem";

    fn schema(&self) -> serde_json::Value {
        todo!()
    }

    fn build(basis_recipe: &Self::BasisRecipe, _params: &Self::Params) -> HamiltonianSpec {
        let mut basis = SpaceBasis::default();

        let l = OrbitalRecipe::Single(basis_recipe.l);
        let l = OrbitalBasis::new(l, &mut basis);

        let elements = basis.get_basis();
        let elements = OrbitalBasisElements::from_orbital(elements, &l);

        let hamiltonian_spec = HamiltonianSpec::new(elements);

        hamiltonian_spec
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SingleChanBasis {
    pub l: u32,
}

#[derive(Debug, Clone, Deserialize, cc_system::Parameters)]
pub struct SingleChanParams {
    pub mass: Scalar<Mass>,
    pub pec: Interactions,

    #[serde(default)]
    pub pec_scaling: PecScaling,
}

fn params_calc_mods() -> ModifyRegistry<SingleChanProblem, SingleChanScatteringCalcInput> {
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

#[derive(Clone, Debug, Deserialize)]
pub struct SingleChanScatteringCalcInput {
    pub energy: Scalar<Energy>,

    pub r_min: Scalar<Length>,
    pub r_max: Scalar<Length>,
    #[serde(default)]
    pub boundary: SingleBoundary,

    pub propagator: SingleChanPropagator,
}

fn pec_values_calc() -> Box<dyn DynCalc<SingleChanProblem>> {
    let scattering = PecValues;
    let registry = ModifyRegistry::from([(
        "distance",
        modify_recipe!(|d| ScalarCalcMod::new(d, |r: &mut PecValuesInput, v| r.distance = v)),
    )]);

    Box::new(DependenceCalc::new(scattering, registry))
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct PecValuesInput {
    pub distance: Scalar<Length>,
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
            modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut SingleChanScatteringCalcInput, v| r.energy = v)),
        ),
        (
            "r_min",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut SingleChanScatteringCalcInput, v| r.r_min = v)),
        ),
        (
            "r_max",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut SingleChanScatteringCalcInput, v| r.r_max = v)),
        ),
        (
            "step_scaling",
            modify_recipe!(|x| FloatCalcMod::new(x, |r: &mut SingleChanScatteringCalcInput, v| r.propagator.scale_step(v))),
        ),
    ]));

    Box::new(DependenceCalc::new(scattering, registry))
}

struct SingleChanScattering;

impl SingleCalc for SingleChanScattering {
    type P = SingleChanProblem;
    type CalcInput = SingleChanScatteringCalcInput;
    type Data = SValueData;

    fn calculate(
        &self,
        modified: &mut Modified<Self::P, SingleChanScatteringCalcInput>,
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
        let s_value = match &modified.calc_input.propagator {
            SingleChanPropagator::JohnsonLogDeriv { step } => {
                let mut prop = LogDerivative::new(&w_function, step.get_step(), boundary);

                SValue::from_log_deriv(prop.propagate_to(r_max), &w_function)
            }
            SingleChanPropagator::AiryLogDeriv { step } => {
                let mut prop = AiryLogDerivative::new(&w_function, step.get_step(), boundary);

                SValue::from_log_deriv(prop.propagate_to(r_max), &w_function)
            }
        };

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

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum SingleBoundary {
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
                    Direction::Inwards => (-value).max(0.).sqrt(),
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

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum SingleChanPropagator {
    JohnsonLogDeriv { step: ConstRefStep },
    AiryLogDeriv { step: LinearRefStep },
}

impl SingleChanPropagator {
    fn scale_step(&mut self, scaling: f64) {
        match self {
            SingleChanPropagator::JohnsonLogDeriv { step } => step.scale_step(scaling),
            SingleChanPropagator::AiryLogDeriv { step } => step.scale_step(scaling),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct PecScalingInput {
    pub scaling: f64,
    #[serde(default)]
    pub scaling_type: PecScalingType,
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
        Number::from_f64(self.scaling.scaling.into()).unwrap()
    }

    fn mut_number(&mut self, number: Number) {
        let value = number.as_f64().unwrap();
        self.scaling.scaling = value.try_into().unwrap()
    }
}
