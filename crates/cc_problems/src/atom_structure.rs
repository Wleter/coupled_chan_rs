use coupled_chan::Operator;
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel, operator_mel,
};
use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

pub struct AtomStructure {
    pub s: BasisId,
    pub i: BasisId,
}

impl AtomStructure {
    pub fn new(s: HalfU32, i: HalfU32, space_basis: &mut SpaceBasis) -> Self {
        let s = get_spin_basis(s);
        let i = get_spin_basis(i);

        let s = space_basis.push_subspace(SubspaceBasis::new(s));
        let i = space_basis.push_subspace(SubspaceBasis::new(i));

        Self { s, i }
    }

    pub fn zeeman_prop(&self, gamma_e: f64, gamma_i: f64, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.s, self.i], |[s: Spin, i: Spin]| {
            -gamma_e * s.m.value() - gamma_i * i.m.value()
        })
    }

    pub fn hyperfine(&self, a_hifi: f64, basis: &BasisElementsRef) -> Operator {
        operator_mel!(dyn basis, [self.s, self.i], |[s: Spin, i: Spin]| {
            a_hifi * SpinOps::dot(s, i)
        })
    }
}
