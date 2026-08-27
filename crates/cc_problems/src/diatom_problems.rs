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
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::MagneticField,
};

use crate::{
    OrbitalBasisElements,
    atom_basis::WithProjection,
    atom_operators::AtomParams,
    diatom_basis::{
        CoupledSIDiatomBasis,
        DiatomRecipe,
    },
    interactions::{
        PecPolarizationSpec,
        PecPolarizations,
        PecScalings,
    },
    operator_mel::spin_projection_term_coupled,
    parameters::Parameters,
    system::{
        DynOperatorSpec,
        DynPotentialSpec,
        HamiltonianSpec,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize, cc_derive::Parameters)]
pub struct DiatomInBFieldParams {
    b_field: Scalar<MagneticField>,
    #[parameter(nested)]
    atom_a: AtomParams,
    #[parameter(nested)]
    atom_b: AtomParams,

    pecs: PecPolarizations,
    scalings: PecScalings,
}

pub fn hamiltonian_diatom_in_b_field(
    recipe: &WithProjection<DiatomRecipe>,
    _params: &DiatomInBFieldParams,
) -> HamiltonianSpec {
    let param_ids = DiatomInBFieldParams::ids();

    let s_a = recipe.recipe.atom_a.s;
    let s_b = recipe.recipe.atom_b.s;
    let polarizations = get_spin_pair_magnitudes([s_a], [s_b]);

    let mut basis = SpaceBasis::default();
    let diatom = CoupledSIDiatomBasis::new(&recipe.recipe, &mut basis);

    let homonuclear_filter = diatom.filter_homo_nuclear_symmetry();
    let homonuclear_filter = |x: SpaceElement| {
        if recipe.recipe.is_homonuclear() {
            homonuclear_filter(x)
        } else {
            true
        }
    };
    let projection_filter = |m, x: SpaceElement| diatom.filter(|(s, i, l)| s.m() + i.m() + l.m() == m)(x);
    let projection_filter = |x: SpaceElement| {
        if let Some(proj) = recipe.projection {
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

    hamiltonian_spec
}
