use std::collections::HashMap;

use hilbert_space::{
    operator_diag_mel,
    space::{
        SpaceBasis,
        SpaceElement,
    },
};
use serde::Deserialize;
use serde_json::Number;
use spin_algebra::{
    SpinLike,
    SpinMagLike,
    get_spin_pair_magnitudes,
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        Energy,
        Length,
        MagneticField,
        Mass,
    },
};

use crate::{
    OrbitalBasisElements,
    UNITS_CONVERTER,
    atom_basis::WithProjection,
    atom_operators::AtomParams,
    calculations::{
        DynCalc,
        Modified,
        adiabats::{
            AdiabatsCalc,
            AdiabatsInput,
        },
        bound_states::{
            BoundStateCalc,
            BoundStateCalcInput,
        },
        dependence::{
            DependenceCalc,
            ModifyParams,
            scalar_from_value,
        },
        levels::EnergyLevelsCalc,
        scattering::{
            ScatteringCalc,
            ScatteringCalcInput,
        },
    },
    diatom_basis::{
        CoupledSIDiatomBasis,
        DiatomRecipe,
    },
    interactions::{
        PecPolarizationSpec,
        PecPolarizations,
        PecScalings,
        Scaling,
        SpinConfiguration,
    },
    operator_mel::spin_projection_term_coupled,
    param_ids,
    parameters::Parameters,
    problems::Problem,
    system::{
        DynOperatorSpec,
        DynPotentialSpec,
        HamiltonianSpec,
        ParamModifications,
        System,
        new_param_modifications,
    },
};

pub type DiatomInBFieldBasis = WithProjection<DiatomRecipe>;

#[derive(Debug, Clone, Deserialize, cc_derive::Parameters)]
pub struct DiatomInBFieldParams {
    #[serde(default)]
    pub b_field: Scalar<MagneticField>,
    #[parameter(nested)]
    #[serde(default)]
    pub atom_a: AtomParams,
    #[parameter(nested)]
    #[serde(default)]
    pub atom_b: AtomParams,

    pub red_mass: Scalar<Mass>,

    pub pecs: PecPolarizations,
    #[serde(default)]
    pub scalings: PecScalings,
}

pub fn diatom_levels_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let levels_calc = EnergyLevelsCalc::default();

    Box::new(DependenceCalc::<_, ModsEnergyLevels>::new(levels_calc))
}

pub fn diatom_adiabats_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let adiabats_calc = AdiabatsCalc::new(ids.red_mass);

    Box::new(DependenceCalc::<_, ModsAdiabatsScan>::new(adiabats_calc))
}

pub fn diatom_scattering_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);

    Box::new(DependenceCalc::<_, ModsScattering>::new(scattering_calc))
}

pub fn diatom_bound_states_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let bound_state_calc: BoundStateCalc<_, ModsBoundSearch> = BoundStateCalc::new(ids.red_mass);

    Box::new(DependenceCalc::<_, ModsBoundScan>::new(bound_state_calc))
}

#[derive(Default)]
pub struct DiatomInBFieldProblem {
    pub calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>>,
}

impl DiatomInBFieldProblem {
    pub fn new() -> Self {
        let mut calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>> = HashMap::default();
        calculations.extend([
            ("levels scan".into(), diatom_levels_b_field_scan()),
            ("adiabats scan".into(), diatom_adiabats_b_field_scan()),
            ("scattering scan".into(), diatom_scattering_b_field_scan()),
            ("bound states scan".into(), diatom_bound_states_b_field_scan()),
        ]);

        Self { calculations }
    }
}

impl Problem for DiatomInBFieldProblem {
    type BasisRecipe = DiatomInBFieldBasis;
    type Params = DiatomInBFieldParams;

    const NAME: &str = "diatom in B field";

    const DESCRIPTION: &str = "Atom + Atom in external magnetic field";

    fn schema(&self) -> serde_json::Value {
        todo!()
    }

    fn build(basis_recipe: &Self::BasisRecipe, _params: &Self::Params) -> HamiltonianSpec {
        let param_ids = Self::Params::ids();

        let s_a = basis_recipe.recipe.atom_a.s;
        let s_b = basis_recipe.recipe.atom_b.s;
        let polarizations = get_spin_pair_magnitudes([s_a], [s_b]);

        let mut basis = SpaceBasis::default();
        let diatom = CoupledSIDiatomBasis::new(&basis_recipe.recipe, &mut basis);

        let homonuclear_filter = diatom.filter_homo_nuclear_symmetry();
        let homonuclear_filter = |x: SpaceElement| {
            if basis_recipe.recipe.is_homonuclear() {
                homonuclear_filter(x)
            } else {
                true
            }
        };
        let projection_filter = |m, x: SpaceElement| diatom.filter(|(s, i, l)| s.m() + i.m() + l.m() == m)(x);
        let projection_filter = |x: SpaceElement| {
            if let Some(proj) = basis_recipe.projection {
                projection_filter(proj, x)
            } else {
                true
            }
        };

        let elements = basis.get_filtered_basis(|x| homonuclear_filter(x) && projection_filter(x));
        let elements = OrbitalBasisElements::from_orbital(elements, &diatom.l);

        let mut hamiltonian_spec = HamiltonianSpec::new(elements);

        hamiltonian_spec.add_operators(vec![
            ("atom_a.hifi", DynOperatorSpec::new(diatom.hifi_a(param_ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_a(param_ids.b_field, param_ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_a(param_ids.b_field, param_ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.hifi_b(param_ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_b(param_ids.b_field, param_ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_b(param_ids.b_field, param_ids.atom_b.g_n)),
            ),
        ]);

        hamiltonian_spec.add_potentials(polarizations.into_iter().map(|p| {
            (
                format!("{}", p.s()),
                DynPotentialSpec::new(PecPolarizationSpec {
                    s_tot: p.s(),
                    pecs: param_ids.pecs,
                    scalings: param_ids.scalings,
                    masking: move |b| operator_diag_mel!(b, [diatom.s_tot], |[s]| spin_projection_term_coupled(s, p)),
                }),
            )
        }));

        hamiltonian_spec
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModsEnergyLevels {
    MagneticField(Scalar<MagneticField>),
}

impl ModifyParams for ModsEnergyLevels {
    type P = DiatomInBFieldProblem;
    type C = ();

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        let ids = DiatomInBFieldParams::ids();
        match self {
            ModsEnergyLevels::MagneticField(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.b_field, scalar.clone())),
        }
    }

    fn as_number(&self) -> serde_json::Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsEnergyLevels::MagneticField(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
        }
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsEnergyLevels::MagneticField(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModsAdiabatsScan {
    Distance(Scalar<Length>),
}

impl ModifyParams for ModsAdiabatsScan {
    type P = DiatomInBFieldProblem;
    type C = AdiabatsInput;

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        match self {
            ModsAdiabatsScan::Distance(scalar) => modified.calc_input.distance = scalar.clone(),
        }
    }

    fn as_number(&self) -> serde_json::Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsAdiabatsScan::Distance(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
        }
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsAdiabatsScan::Distance(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModsScattering {
    MagneticField(Scalar<MagneticField>),
    Mass(Scalar<Mass>),
    L(u32),
    Energy(Scalar<Energy>),
    PecScaling { configuration: SpinConfiguration, scaling: f64 },
    PecScalingFull(f64),
    RStart(Scalar<Length>),
    RStop(Scalar<Length>),
}

impl ModifyParams for ModsScattering {
    type P = DiatomInBFieldProblem;
    type C = ScatteringCalcInput;

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        let ids = DiatomInBFieldParams::ids();

        match self {
            ModsScattering::MagneticField(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.b_field, scalar.clone())),
            ModsScattering::Mass(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.red_mass, scalar.clone())),
            ModsScattering::Energy(scalar) => modified.calc_input.energy = scalar.clone(),
            ModsScattering::L(l_new) => {
                match &mut modified.basis.recipe.l {
                    crate::OrbitalRecipe::Single(l) => *l = *l_new,
                    crate::OrbitalRecipe::LMax(l) => *l = *l_new,
                    crate::OrbitalRecipe::LMaxProjections(l) => *l = *l_new,
                }

                let spec = Self::P::build(modified.basis, modified.params);
                *modified.system = System::new(spec, modified.params.registry())
            }
            ModsScattering::PecScaling { configuration, scaling } => {
                let scaling = *scaling;
                let configuration = *configuration;
                let modify = ParamModifications::new(move |r| {
                    let scalings = r.get_mut(ids.scalings);
                    if let Some(s) = scalings.0.get(&configuration)
                        && s.0 == scaling
                    {
                        param_ids![]
                    } else {
                        scalings.0.insert(configuration, Scaling(scaling));
                        param_ids![ids.scalings.vanish()]
                    }
                });

                modified.system.modify_params(modify)
            }
            ModsScattering::PecScalingFull(scaling) => {
                let scaling = *scaling;
                let modify = ParamModifications::new(move |r| {
                    let configurations: Vec<SpinConfiguration> = r.get(ids.pecs).0.keys().copied().collect();

                    let mut changed = false;
                    let scalings = r.get_mut(ids.scalings);
                    for configuration in configurations {
                        let overridden = scalings.0.insert(configuration, Scaling(scaling));

                        if let Some(overridden) = overridden
                            && overridden.0 == scaling
                        {
                        } else {
                            changed = true
                        }
                    }

                    if changed {
                        param_ids![ids.scalings.vanish()]
                    } else {
                        param_ids![]
                    }
                });

                modified.system.modify_params(modify)
            }
            ModsScattering::RStart(scalar) => modified.calc_input.r_start = scalar.clone(),
            ModsScattering::RStop(scalar) => modified.calc_input.r_stop = scalar.clone(),
        }
    }

    fn as_number(&self) -> serde_json::Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsScattering::MagneticField(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsScattering::Mass(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsScattering::L(l) => Number::from_u128(*l as u128).unwrap(),
            ModsScattering::Energy(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsScattering::PecScaling {
                configuration: _,
                scaling,
            } => Number::from_f64(*scaling).unwrap(),
            ModsScattering::PecScalingFull(scaling) => Number::from_f64(*scaling).unwrap(),
            ModsScattering::RStart(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsScattering::RStop(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
        }
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsScattering::MagneticField(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsScattering::Mass(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsScattering::L(l) => *l = number.as_u64().unwrap() as u32,
            ModsScattering::Energy(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsScattering::PecScaling {
                configuration: _,
                scaling,
            } => *scaling = number.as_f64().unwrap(),
            ModsScattering::PecScalingFull(scaling) => *scaling = number.as_f64().unwrap(),
            ModsScattering::RStart(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsScattering::RStop(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModsBoundScan {
    MagneticField(Scalar<MagneticField>),
    Mass(Scalar<Mass>),
    L(u32),
    Energy(Scalar<Energy>),
    PecScaling { configuration: SpinConfiguration, scaling: f64 },
    PecScalingFull(f64),
    RStart(Scalar<Length>),
    RStop(Scalar<Length>),
}

impl ModifyParams for ModsBoundScan {
    type P = DiatomInBFieldProblem;
    type C = BoundStateCalcInput<ModsBoundSearch>;

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        let ids = DiatomInBFieldParams::ids();

        match self {
            ModsBoundScan::MagneticField(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.b_field, scalar.clone())),
            ModsBoundScan::Mass(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.red_mass, scalar.clone())),
            ModsBoundScan::Energy(scalar) => modified.calc_input.energy = scalar.clone(),
            ModsBoundScan::L(l_new) => {
                match &mut modified.basis.recipe.l {
                    crate::OrbitalRecipe::Single(l) => *l = *l_new,
                    crate::OrbitalRecipe::LMax(l) => *l = *l_new,
                    crate::OrbitalRecipe::LMaxProjections(l) => *l = *l_new,
                }

                let spec = Self::P::build(modified.basis, modified.params);
                *modified.system = System::new(spec, modified.params.registry())
            }
            ModsBoundScan::PecScaling { configuration, scaling } => {
                let scaling = *scaling;
                let configuration = *configuration;
                let modify = ParamModifications::new(move |r| {
                    let scalings = r.get_mut(ids.scalings);
                    if let Some(s) = scalings.0.get(&configuration)
                        && s.0 == scaling
                    {
                        param_ids![]
                    } else {
                        scalings.0.insert(configuration, Scaling(scaling));
                        param_ids![ids.scalings.vanish()]
                    }
                });

                modified.system.modify_params(modify)
            }
            ModsBoundScan::PecScalingFull(scaling) => {
                let scaling = *scaling;
                let modify = ParamModifications::new(move |r| {
                    let configurations: Vec<SpinConfiguration> = r.get(ids.pecs).0.keys().copied().collect();

                    let mut changed = false;
                    let scalings = r.get_mut(ids.scalings);
                    for configuration in configurations {
                        let overridden = scalings.0.insert(configuration, Scaling(scaling));

                        if let Some(overridden) = overridden
                            && overridden.0 == scaling
                        {
                        } else {
                            changed = true
                        }
                    }

                    if changed {
                        param_ids![ids.scalings.vanish()]
                    } else {
                        param_ids![]
                    }
                });

                modified.system.modify_params(modify)
            }
            ModsBoundScan::RStart(scalar) => modified.calc_input.r_min = scalar.clone(),
            ModsBoundScan::RStop(scalar) => modified.calc_input.r_max = scalar.clone(),
        }
    }

    fn as_number(&self) -> serde_json::Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsBoundScan::MagneticField(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundScan::Mass(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundScan::L(l) => Number::from_u128(*l as u128).unwrap(),
            ModsBoundScan::Energy(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundScan::PecScaling {
                configuration: _,
                scaling,
            } => Number::from_f64(*scaling).unwrap(),
            ModsBoundScan::PecScalingFull(scaling) => Number::from_f64(*scaling).unwrap(),
            ModsBoundScan::RStart(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundScan::RStop(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
        }
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsBoundScan::MagneticField(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundScan::Mass(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundScan::L(l) => *l = number.as_u64().unwrap() as u32,
            ModsBoundScan::Energy(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundScan::PecScaling {
                configuration: _,
                scaling,
            } => *scaling = number.as_f64().unwrap(),
            ModsBoundScan::PecScalingFull(scaling) => *scaling = number.as_f64().unwrap(),
            ModsBoundScan::RStart(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundScan::RStop(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModsBoundSearch {
    MagneticField(Scalar<MagneticField>),
    Mass(Scalar<Mass>),
    Energy(Scalar<Energy>),
    PecScalingFull(f64),
}

impl ModifyParams for ModsBoundSearch {
    type P = DiatomInBFieldProblem;
    type C = BoundStateCalcInput<Self>;

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        let ids = DiatomInBFieldParams::ids();

        match self {
            ModsBoundSearch::MagneticField(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.b_field, scalar.clone())),
            ModsBoundSearch::Mass(scalar) => modified
                .system
                .modify_params(new_param_modifications(ids.red_mass, scalar.clone())),
            ModsBoundSearch::Energy(scalar) => modified.calc_input.energy = scalar.clone(),
            ModsBoundSearch::PecScalingFull(scaling) => {
                let scaling = *scaling;
                let modify = ParamModifications::new(move |r| {
                    let configurations: Vec<SpinConfiguration> = r.get(ids.pecs).0.keys().copied().collect();

                    let mut changed = false;
                    let scalings = r.get_mut(ids.scalings);
                    for configuration in configurations {
                        let overridden = scalings.0.insert(configuration, Scaling(scaling));

                        if let Some(overridden) = overridden
                            && overridden.0 == scaling
                        {
                        } else {
                            changed = true
                        }
                    }

                    if changed {
                        param_ids![ids.scalings.vanish()]
                    } else {
                        param_ids![]
                    }
                });

                modified.system.modify_params(modify)
            }
        }
    }

    fn as_number(&self) -> serde_json::Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsBoundSearch::MagneticField(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundSearch::Mass(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundSearch::Energy(scalar) => Number::from_f64(converter.scalar_value(scalar)).unwrap(),
            ModsBoundSearch::PecScalingFull(scaling) => Number::from_f64(*scaling).unwrap(),
        }
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        match self {
            ModsBoundSearch::MagneticField(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundSearch::Mass(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundSearch::Energy(scalar) => *scalar = scalar_from_value(&converter, number.as_f64().unwrap()),
            ModsBoundSearch::PecScalingFull(scaling) => *scaling = number.as_f64().unwrap(),
        }
    }
}
