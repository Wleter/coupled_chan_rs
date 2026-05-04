use hilbert_space::{
    operator_diag_mel,
    operator_mel,
    space::BasisId,
};
use spin_algebra::{
    SpinLike,
    SpinMagLike,
    SpinPair,
    get_spin_basis,
    ops::{
        clebsch_gordan_coef,
        triangle_condition,
    },
};

use crate::hamiltonian::HamiltonianTerm;

pub fn dot_uncoupled_term(s1_id: BasisId<impl SpinLike>, s2_id: BasisId<impl SpinLike>) -> HamiltonianTerm {
    HamiltonianTerm::new(move |e| operator_mel!(e, [s1_id, s2_id], |[s1, s2]| spin_algebra::ops::dot(s1, s2)))
}

pub fn dot_coupled_term(f_id: BasisId<SpinPair<impl SpinMagLike, impl SpinMagLike>>) -> HamiltonianTerm {
    HamiltonianTerm::new(move |e| {
        operator_diag_mel!(e, [f_id], |[f]| {
            (f.spin.squared() - f.pair.0.squared() - f.pair.1.squared()) / 2.
        })
    })
}

pub fn spin_sum_projected_term_uncoupled(
    s1_id: BasisId<impl SpinLike>,
    s2_id: BasisId<impl SpinLike>,
    s_tot_projected: impl SpinMagLike,
) -> HamiltonianTerm {
    let spins = get_spin_basis(s_tot_projected.s());

    HamiltonianTerm::new(move |e| {
        operator_mel!(e, [s1_id, s2_id], |[s1, s2]| {
            if s1.bra.m() + s2.bra.m() != s1.ket.m() + s2.ket.m()
                || !triangle_condition(s1.bra, s2.bra, s_tot_projected)
                || !triangle_condition(s1.ket, s2.ket, s_tot_projected)
            {
                return 0.;
            }

            spins
                .iter()
                .map(|&s| clebsch_gordan_coef(s1.bra, s2.bra, s) * clebsch_gordan_coef(s1.ket, s2.ket, s))
                .sum()
        })
    })
}

pub fn spin_projected_term_coupled(s_sum_id: BasisId<impl SpinLike>, s_projected: impl SpinMagLike) -> HamiltonianTerm {
    let s_projected = s_projected.s();
    HamiltonianTerm::new(move |e| operator_diag_mel!(e, [s_sum_id], |[s]| if s.s() == s_projected { 1. } else { 0. }))
}
