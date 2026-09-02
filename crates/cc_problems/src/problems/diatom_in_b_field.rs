use std::collections::HashMap;

use hilbert_space::{
    operator_diag_mel,
    space::{
        SpaceBasis,
        SpaceElement,
    },
};
use serde::Deserialize;
use spin_algebra::{
    SpinLike,
    SpinMagLike,
    get_spin_pair_magnitudes,
    hu32,
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        MagneticField,
        Mass,
    },
};

use crate::{
    OrbitalBasisElements,
    atom_basis::WithProjection,
    atom_operators::AtomParams,
    calculations::{
        DynCalc,
        dependence::{
            CalcModifications,
            DependenceCalc,
            ParametersMod,
        },
        levels::{
            EnergyLevelsCalc,
        },
        scattering::{
            ScatteringCalc, ScatteringCalcInput
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
    },
};

pub type DiatomInBFieldBasis = WithProjection<DiatomRecipe>;

#[derive(Debug, Clone, Deserialize, cc_derive::Parameters)]
pub struct DiatomInBFieldParams {
    #[serde(default)]
    pub b_field: Scalar<MagneticField>,
    #[parameter(nested)]
    pub atom_a: AtomParams,
    #[parameter(nested)]
    pub atom_b: AtomParams,

    pub red_mass: Scalar<Mass>,

    pub pecs: PecPolarizations,
    #[serde(default)]
    pub scalings: PecScalings,
}

impl DiatomInBFieldParams {
    pub fn modifications() -> HashMap<Box<str>, ParametersMod> {
        let ids = <Self as Parameters>::ids();
        let mut modifications: HashMap<Box<str>, ParametersMod> = HashMap::default();
        modifications.extend([
            ("b_field".into(), ParametersMod::from_id(ids.b_field)),
            ("red_mass".into(), ParametersMod::from_id(ids.red_mass)),
        ]);

        // conservative 9/2 spin maximum scaling
        for spin in SpinConfiguration::get_configurations(hu32!(9 / 2)) {
            modifications.insert(
                format!("pec.scalings.{}", spin).into(),
                ParametersMod(Box::new(move |value| {
                    ParamModifications::new(move |r| {
                        let ids = DiatomInBFieldParams::ids();
                        let scalings = r.get_mut(ids.scalings);

                        if scalings.scale(spin, value) {
                            param_ids![ids.scalings.vanish()]
                        } else {
                            param_ids![]
                        }
                    })
                    .into_dyn()
                })),
            );
        }

        modifications.insert(
            "pec.scalings.all".into(),
            ParametersMod(Box::new(move |value| {
                ParamModifications::new(move |r| {
                    let ids = DiatomInBFieldParams::ids();
                    let scalings = r.get_mut(ids.scalings);

                    if scalings.scale_all(value) {
                        param_ids![]
                    } else {
                        param_ids![ids.scalings.vanish()]
                    }
                })
                .into_dyn()
            })),
        );

        modifications
    }
}

pub fn diatom_levels_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let levels_calc = EnergyLevelsCalc::default();

    let modifications: HashMap<Box<str>, ParametersMod> =
        HashMap::from([("b_field".into(), ParametersMod::from_id(ids.b_field))]);
    let modifications = CalcModifications::from_params_modifications(modifications);

    Box::new(DependenceCalc::new(levels_calc, modifications))
}

pub fn diatom_scattering_b_field_scan() -> Box<dyn DynCalc<DiatomInBFieldProblem>> {
    let ids = DiatomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);
    let param_mods = DiatomInBFieldParams::modifications();
    let recipe_mods = ScatteringCalcInput::modifications();

    let modifications = CalcModifications::new(HashMap::default(), param_mods, recipe_mods);

    Box::new(DependenceCalc::new(scattering_calc, modifications))
}

pub struct DiatomInBFieldProblem {
    pub calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>>,
}

impl DiatomInBFieldProblem {
    pub fn new() -> Self {
        let mut calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>> = HashMap::default();
        calculations.extend([
            ("levels scan".into(), diatom_levels_b_field_scan()),
            ("scattering scan".into(), diatom_scattering_b_field_scan()),
        ]);

        Self {
            calculations
        }
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

    fn build(&self, basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> System {
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
            ("atom_b.hifi", DynOperatorSpec::new(diatom.hifi_a(param_ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_a(param_ids.b_field, param_ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_a(param_ids.b_field, param_ids.atom_b.g_n)),
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

        System::new(hamiltonian_spec, params.registry())
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}
