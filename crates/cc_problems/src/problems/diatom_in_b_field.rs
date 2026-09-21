use std::collections::HashMap;

use hilbert_space::{
    operator_diag_mel,
    space::{
        SpaceBasis,
        SpaceElement,
    },
};
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::{
    SpinLike,
    SpinMagLike,
    get_spin_pair_magnitudes,
    half_integer::HalfI32,
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        MagneticField,
        Mass,
    },
};

use crate::{
    OrbitalBasisElements, atom_operators::AtomParams, calculations::{
        DynCalc, adiabats::{
            AdiabatsCalc,
            adiabats_calc_mods,
        }, bound_states::{
            BoundStateCalc, bound_states_calc_mods, bound_states_calc_search_mods
        }, dependence::DependenceCalc, levels::EnergyLevelsCalc, modifications::{ModifyRegistry, OrbitalRecipeMod, ScalarParamMod}, resonances::
            ResonancesCalc
        , scattering::{
            ScatteringCalc,
            scattering_calc_mods,
        }
    }, diatom_basis::{
        CoupledSIDiatomBasis,
        DiatomRecipe,
    }, diatom_operators::SpinRotationSpec, interactions::{
        Interactions, PecPolarizationSpec, PecPolarizations, PecScalings, 
    }, modify_recipe, operator_mel::spin_projection_term_coupled, parameters::Parameters, problems::Problem, system::{
        DynOperatorSpec,
        DynPotentialSpec,
        HamiltonianSpec,
    }
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiatomInBFieldBasis {
    #[serde(flatten)]
    pub recipe: DiatomRecipe,
    #[serde(default)]
    pub projection: Option<HalfI32>,
    #[serde(default)]
    pub l_parity: OrbitalParity,
}

pub fn diatom_in_b_field_basis_mods<P, C>() -> ModifyRegistry<P, C> 
where 
    P: Problem<BasisRecipe = DiatomInBFieldBasis> + 'static,
    C: Send + Sync + 'static
{
    ModifyRegistry(HashMap::from([
        ("l".into(), modify_recipe!(|l| OrbitalRecipeMod::new(l, |r: &mut P::BasisRecipe| &mut r.recipe.l))),
    ]))
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum OrbitalParity {
    #[default]
    All,
    Even,
    Odd,
}

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

    #[serde(default)]
    pub spin_orbit: Option<Interactions>
}

pub fn diatom_in_b_field_params_mods<P, C>() -> ModifyRegistry<P, C> 
where 
    P: Problem<Params = DiatomInBFieldParams> + 'static,
    C: Send + Sync + 'static
{
    let ids = DiatomInBFieldParams::ids();
    ModifyRegistry(HashMap::from([
        ("magnetic_field".into(), modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field))),
        ("red_mass".into(), modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass))),
    ]))
}

pub fn diatom_in_b_field_params_search_mods<P, C>() -> ModifyRegistry<P, C> 
where 
    P: Problem<Params = DiatomInBFieldParams> + 'static,
    C: Send + Sync + 'static
{
    let ids = DiatomInBFieldParams::ids();
    ModifyRegistry(HashMap::from([
        ("magnetic_field".into(), modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field))),
        ("red_mass".into(), modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass))),
    ]))
}

pub fn diatom_levels_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let levels_calc = EnergyLevelsCalc::default();

    let ids = DiatomInBFieldParams::ids();
    let registry = ModifyRegistry(HashMap::from([
        ("magnetic_field".into(), modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field))),
    ]));
    Box::new(DependenceCalc::new(levels_calc, registry))
}

pub fn diatom_adiabats_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let adiabats_calc = AdiabatsCalc::new(ids.red_mass);

    let registry = adiabats_calc_mods();
    Box::new(DependenceCalc::new(adiabats_calc, registry))
}

pub fn diatom_scattering_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);

    let registry = diatom_in_b_field_basis_mods()
        .extend(diatom_in_b_field_params_mods())
        .extend(scattering_calc_mods());
    Box::new(DependenceCalc::new(scattering_calc, registry))
}

pub fn diatom_bound_states_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let registry = diatom_in_b_field_params_search_mods()
        .extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let registry = diatom_in_b_field_basis_mods()
        .extend(diatom_in_b_field_params_mods())
        .extend(bound_states_calc_mods());
    Box::new(DependenceCalc::new(bound_state_calc, registry))
}

pub fn diatom_resonances_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);
    let registry = diatom_in_b_field_params_search_mods()
        .extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let resonances_calc = ResonancesCalc::new(scattering_calc, bound_state_calc);

    let registry = diatom_in_b_field_basis_mods()
        .extend(diatom_in_b_field_params_mods());
    Box::new(DependenceCalc::new(resonances_calc, registry))
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
            ("resonances scan".into(), diatom_resonances_scan()),
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
        let l_filter = |x: SpaceElement| match basis_recipe.l_parity {
            OrbitalParity::All => true,
            OrbitalParity::Even => diatom.filter(|(_, _, l)| l.l_value().is_multiple_of(2))(x),
            OrbitalParity::Odd => diatom.filter(|(_, _, l)| !l.l_value().is_multiple_of(2))(x),
        };

        let elements = basis.get_filtered_basis(|x| homonuclear_filter(x) && projection_filter(x) && l_filter(x));
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
        hamiltonian_spec.add_potentials([
            (
                "spin_rot_coupling",
                DynPotentialSpec::new(SpinRotationSpec {
                    so_curve: param_ids.spin_orbit,
                    masking: diatom.second_order_so(),
                }),
            ),
        ]);

        hamiltonian_spec
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}
