use std::collections::HashMap;

use hilbert_space::{
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
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        Energy, MagneticDipole, MagneticField, Mass
    },
};

use crate::{
    OrbitalBasisElements, atom_operators::{AtomParams, electron_g_e}, calculations::{
        DynCalc, adiabats::{AdiabatsCalc, adiabats_calc_mods}, bound_states::{BoundStateCalc, bound_states_calc_mods, bound_states_calc_search_mods}, dependence::DependenceCalc, hamiltonian_terms::hamiltonian_terms_calc, levels::EnergyLevelsCalc, modifications::{ModifyRegistry, NumberCalcMod, ScalarParamMod}, resonances::ResonancesCalc, scattering::{ScatteringCalc, scattering_calc_mods}
    }, diatom_operators::SpinRotationSpec, interactions::{
        Interactions, SpinConfiguration, pes::{PesPolarizationScalingMod, PesPolarizationScalings, PesPolarizationSpec, PesPolarizations}
    }, modify_recipe, parameters::Parameters, problems::{Problem, diatom_in_b_field::OrbitalParity}, rotor_atom_basis::{TRAMRotorAtomBasis, TRAMRotorAtomBasisRecipe}, system::{
        DynOperatorSpec,
        DynPotentialSpec,
        HamiltonianSpec,
    }
};

#[derive(Debug, Clone, Parameters, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RotorSpinParams {
    pub a_hifi_a: Scalar<Energy>,
    pub a_hifi_b: Scalar<Energy>,
    pub c_hifi_a: Scalar<Energy>,
    pub c_hifi_b: Scalar<Energy>,

    pub g_e: Scalar<MagneticDipole>,
    pub g_n_a: Scalar<MagneticDipole>,
    pub g_n_b: Scalar<MagneticDipole>,
}

impl Default for RotorSpinParams {
    fn default() -> Self {
        Self { 
            a_hifi_a: Default::default(), 
            a_hifi_b: Default::default(), 
            c_hifi_a: Default::default(), 
            c_hifi_b: Default::default(), 
            g_e: electron_g_e(), 
            g_n_a: Default::default(), 
            g_n_b: Default::default() 
        }
    }
}

#[derive(Debug, Clone, Parameters, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorParams {
    pub rot_const: Scalar<Energy>,
    #[serde(default)]
    pub rot_distortion: Scalar<Energy>,

    #[serde(default)]
    pub spin_e_rot: Scalar<Energy>,

    #[serde(default)]
    pub spin_n_a_rot: Scalar<Energy>,

    #[serde(default)]
    pub spin_n_b_rot: Scalar<Energy>,
}

#[derive(Debug, Clone, Deserialize, cc_derive::Parameters)]
#[serde(deny_unknown_fields)]
pub struct RotorAtomInBFieldParams {
    #[serde(default)]
    pub b_field: Scalar<MagneticField>,
    pub red_mass: Scalar<Mass>,

    #[parameter(nested)]
    #[serde(default)]
    pub rotor_spins: RotorSpinParams,

    #[parameter(nested)]
    #[serde(default)]
    pub atom: AtomParams,

    #[parameter(nested)]
    pub rotor: RotorParams,

    pub pes: PesPolarizations,
    #[serde(default)]
    pub scalings: PesPolarizationScalings,

    #[serde(default)]
    pub spin_orbit: Option<Interactions>,
}

pub fn rotor_atom_in_b_field_basis_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<BasisRecipe = TRAMRotorAtomBasisRecipe> + 'static,
    C: Send + Sync + 'static,
{
    ModifyRegistry::from([
        ("n_max", modify_recipe!(|n_max| NumberCalcMod::new(n_max, |r: &mut P::BasisRecipe, n| r.tram.n_max = n))),
        ("l_max", modify_recipe!(|l_max| NumberCalcMod::new(l_max, |r: &mut P::BasisRecipe, l| r.tram.l_max = l))),
        ("n_tot_max", modify_recipe!(|n_max| NumberCalcMod::new(n_max, |r: &mut P::BasisRecipe, n| r.tram.n_tot_max = n))),
    ])
}

pub fn rotor_atom_in_b_field_params_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<Params = RotorAtomInBFieldParams> + 'static,
    C: Send + Sync + 'static,
{
    let ids = RotorAtomInBFieldParams::ids();
    ModifyRegistry::from([
        ("magnetic_field", modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field))),
        ("red_mass", modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass))),
        ("pes_scaling", modify_recipe!(|x| PesPolarizationScalingMod::new(x, ids.pes, ids.scalings))),
    ])
}

pub fn rotor_atom_in_b_field_params_search_mods<P, C>() -> ModifyRegistry<P, C>
where
    P: Problem<Params = RotorAtomInBFieldParams> + 'static,
    C: Send + Sync + 'static,
{
    let ids = RotorAtomInBFieldParams::ids();
    ModifyRegistry::from([
        ("magnetic_field", modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field))),
        ("red_mass", modify_recipe!(|x| ScalarParamMod::new(x, ids.red_mass))),
        ("pes_scaling", modify_recipe!(|x| PesPolarizationScalingMod::new(x, ids.pes, ids.scalings))),
    ])
}

pub fn rotor_atom_levels_b_field_scan() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let levels_calc = EnergyLevelsCalc::default();

    let ids = RotorAtomInBFieldParams::ids();
    let registry = ModifyRegistry::from([("magnetic_field", modify_recipe!(|b| ScalarParamMod::new(b, ids.b_field)))]);
    Box::new(DependenceCalc::new(levels_calc, registry))
}

pub fn rotor_atom_adiabats_b_field_scan() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let ids = RotorAtomInBFieldParams::ids();
    let adiabats_calc = AdiabatsCalc::new(ids.red_mass);

    let registry = adiabats_calc_mods();
    Box::new(DependenceCalc::new(adiabats_calc, registry))
}

pub fn rotor_atom_scattering_b_field_scan() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let ids = RotorAtomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);

    let registry = rotor_atom_in_b_field_basis_mods()
        .extend(rotor_atom_in_b_field_params_mods())
        .extend(scattering_calc_mods());
    Box::new(DependenceCalc::new(scattering_calc, registry))
}

pub fn rotor_atom_bound_states_b_field_scan() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let ids = RotorAtomInBFieldParams::ids();
    let registry = rotor_atom_in_b_field_params_search_mods()
        .extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let registry = rotor_atom_in_b_field_basis_mods()
        .extend(rotor_atom_in_b_field_params_mods())
        .extend(bound_states_calc_mods());
    Box::new(DependenceCalc::new(bound_state_calc, registry))
}

pub fn rotor_atom_resonances_scan() -> Box<dyn DynCalc<RotorAtomInBFieldProblem>> {
    let ids = RotorAtomInBFieldParams::ids();
    let scattering_calc = ScatteringCalc::new(ids.red_mass);
    let registry = rotor_atom_in_b_field_params_search_mods()
        .extend(bound_states_calc_search_mods());
    let bound_state_calc = BoundStateCalc::new(ids.red_mass, registry);

    let resonances_calc = ResonancesCalc::new(scattering_calc, bound_state_calc);

    let registry = rotor_atom_in_b_field_basis_mods()
        .extend(rotor_atom_in_b_field_params_mods());
    Box::new(DependenceCalc::new(resonances_calc, registry))
}

#[derive(Default)]
pub struct RotorAtomInBFieldProblem {
    pub calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>>,
}

impl RotorAtomInBFieldProblem {
    pub fn new() -> Self {
        let mut calculations: HashMap<Box<str>, Box<dyn DynCalc<Self>>> = HashMap::default();
        calculations.extend([
            ("hamiltonian terms".into(), hamiltonian_terms_calc()),
            ("levels scan".into(), rotor_atom_levels_b_field_scan()),
            ("adiabats scan".into(), rotor_atom_adiabats_b_field_scan()),
            ("scattering scan".into(), rotor_atom_scattering_b_field_scan()),
            ("bound states scan".into(), rotor_atom_bound_states_b_field_scan()),
            ("resonances scan".into(), rotor_atom_resonances_scan()),
        ]);

        Self { calculations }
    }
}

impl Problem for RotorAtomInBFieldProblem {
    type BasisRecipe = TRAMRotorAtomBasisRecipe;
    type Params = RotorAtomInBFieldParams;

    const NAME: &str = "rotor + atom in B field";

    const DESCRIPTION: &str = "Rotor + Atom in external magnetic field";

    fn schema(&self) -> serde_json::Value {
        todo!()
    }

    fn build(basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> HamiltonianSpec {
        let param_ids = Self::Params::ids();

        let mut basis = SpaceBasis::default();
        let rotor_atom = TRAMRotorAtomBasis::new(basis_recipe, &mut basis);

        let projection_filter = |m, x: SpaceElement| rotor_atom.filter(|((s_r, i_ra, i_rb), (s_a, i_a), n_tot)| {
            s_r.m() + i_ra.m() + i_rb.m() + s_a.m() + i_a.m() + n_tot.m() == m
        })(x);
        let projection_filter = |x: SpaceElement| {
            if let Some(proj) = basis_recipe.projection {
                projection_filter(proj, x)
            } else {
                true
            }
        };
        let l_filter = |x: SpaceElement| match basis_recipe.l_parity {
            OrbitalParity::All => true,
            OrbitalParity::Even => rotor_atom.filter(|(_, _, n_tot)| n_tot.pair.1.is_multiple_of(2))(x),
            OrbitalParity::Odd => rotor_atom.filter(|(_, _, n_tot)| !n_tot.pair.1.is_multiple_of(2))(x),
        };

        let elements = basis.get_filtered_basis(|x| projection_filter(x) && l_filter(x));
        let elements = OrbitalBasisElements::new(elements, rotor_atom.tram.tram, |n_tot| n_tot.pair.1);

        let mut hamiltonian_spec = HamiltonianSpec::new(elements);

        hamiltonian_spec.add_operators(vec![
            ("atom.hifi", DynOperatorSpec::new(rotor_atom.atom.hifi(param_ids.atom.a_hifi))),
            (
                "atom.zeeman_e",
                DynOperatorSpec::new(rotor_atom.atom.zeeman_e(param_ids.b_field, param_ids.atom.g_e)),
            ),
            (
                "atom.zeeman_n",
                DynOperatorSpec::new(rotor_atom.atom.zeeman_n(param_ids.b_field, param_ids.atom.g_n)),
            ),
            ("rotor_energy", DynOperatorSpec::new(rotor_atom.rot_energy(param_ids.rotor.rot_const))),
            ("rotor_energy_distortion", DynOperatorSpec::new(rotor_atom.rot_energy_distortion(param_ids.rotor.rot_distortion))),
            ("rotor.hifi_a", DynOperatorSpec::new(rotor_atom.rotor_a.hifi(param_ids.rotor_spins.a_hifi_a))),
            ("rotor.hifi_b", DynOperatorSpec::new(rotor_atom.rotor_b.hifi(param_ids.rotor_spins.a_hifi_b))),
            (
                "rotor.zeeman_e",
                DynOperatorSpec::new(rotor_atom.rotor_a.zeeman_e(param_ids.b_field, param_ids.rotor_spins.g_e)),
            ),
            (
                "rotor.zeeman_n_a",
                DynOperatorSpec::new(rotor_atom.rotor_a.zeeman_n(param_ids.b_field, param_ids.rotor_spins.g_n_a)),
            ),
            (
                "rotor.zeeman_n_b",
                DynOperatorSpec::new(rotor_atom.rotor_b.zeeman_n(param_ids.b_field, param_ids.rotor_spins.g_n_b)),
            ),
            ("rotor.aniso_hifi_a", DynOperatorSpec::new(rotor_atom.aniso_hifi_a(param_ids.rotor_spins.c_hifi_a))),
            ("rotor.aniso_hifi_b", DynOperatorSpec::new(rotor_atom.aniso_hifi_b(param_ids.rotor_spins.c_hifi_b))),
            ("rotor.e_rot", DynOperatorSpec::new(rotor_atom.spin_e_rot(param_ids.rotor.spin_e_rot))),
            ("rotor.n_a_rot", DynOperatorSpec::new(rotor_atom.spin_n_a_rot(param_ids.rotor.spin_n_a_rot))),
            ("rotor.n_b_rot", DynOperatorSpec::new(rotor_atom.spin_n_b_rot(param_ids.rotor.spin_n_b_rot))),
        ]);

        let s_r = basis_recipe.rotor.s;
        let s_a = basis_recipe.atom.s;
        let polarizations = get_spin_pair_magnitudes([s_r], [s_a]);
        hamiltonian_spec.add_potentials(polarizations.into_iter().flat_map(|p| {
            let pes = &params.pes.0[&SpinConfiguration::Spin(p.s())];
            let s_tot = p.s();
            
            pes.components().into_iter().map(move |lambda| {
                (
                    format!("pes_s_{s_tot}_lambda_{lambda}"),
                    DynPotentialSpec::new(PesPolarizationSpec {
                        s_tot: s_tot,
                        lambda,
                        pes: param_ids.pes,
                        scalings: param_ids.scalings,
                        masking: rotor_atom.pes_polarization_legendre_component(lambda, s_tot),
                    }),
                )
            })
        }));
        hamiltonian_spec.add_potentials([(
            "spin_rot_coupling",
            DynPotentialSpec::new(SpinRotationSpec {
                so_curve: param_ids.spin_orbit,
                masking: rotor_atom.second_order_so(),
            }),
        )]);

        hamiltonian_spec
    }

    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>> {
        &self.calculations
    }
}
