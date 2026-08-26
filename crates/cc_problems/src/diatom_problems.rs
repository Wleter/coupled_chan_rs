use hilbert_space::space::{
    SpaceBasis,
    SpaceElement,
};
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::SpinLike;
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
    interactions::PecPolarizations,
    parameters::Parameters,
    system::{
        DynOperatorSpec,
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

    pec: PecPolarizations,
}

pub fn hamiltonian_diatom_in_b_field(recipe: &WithProjection<DiatomRecipe>, params: &DiatomInBFieldParams) -> HamiltonianSpec {
    let param_ids = DiatomInBFieldParams::ids();

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

    // hamiltonian_spec.add_potentials(
    //     params.pec.iter().map(|((s_tot, p))| {
    //         let operator =
    //         spin_projection_term_coupled(s, s_tot)
    //     }

    //     )
    // );

    hamiltonian_spec
}
