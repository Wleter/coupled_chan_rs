use coupled_chan::Operator;
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel,
};

use crate::AngularMomentum;

pub struct RotorStructure {
    pub n: BasisId,
}

impl RotorStructure {
    pub fn new(n_max: AngularMomentum, space_basis: &mut SpaceBasis) -> Self {
        let n = (0..=n_max.0).map(AngularMomentum).collect();

        let n = space_basis.push_subspace(SubspaceBasis::new(n));

        Self { n }
    }

    pub fn rotational_energy(&self, rot_const: f64, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            rot_const * (n.0 * (n.0 + 1)) as f64
        })
    }

    pub fn distortion(&self, distortion: f64, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            -distortion * (n.0 * (n.0 + 1)).pow(2) as f64
        })
    }
}
