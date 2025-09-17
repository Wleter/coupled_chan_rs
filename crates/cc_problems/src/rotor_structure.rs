use coupled_chan::{
    Interaction, Operator,
    constants::units::{Quantity, atomic_units::AuEnergy},
};
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel,
};

use crate::AngularMomentum;

#[derive(Clone, Debug)]
pub struct RotorStructure {
    pub n: BasisId,
}

impl RotorStructure {
    pub fn new(n_max: u32, space_basis: &mut SpaceBasis) -> Self {
        let n = (0..=n_max).map(AngularMomentum).collect();

        let n = space_basis.push_subspace(SubspaceBasis::new(n));

        Self { n }
    }

    pub fn rotational_energy(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            (n.0 * (n.0 + 1)) as f64
        })
    }

    pub fn distortion(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            -((n.0 * (n.0 + 1)).pow(2) as f64)
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RotorParams {
    pub rot_const: Quantity<AuEnergy>,
    pub distortion: Quantity<AuEnergy>,
}

#[derive(Clone, Debug)]
pub struct Interaction2D<I: Interaction>(pub Vec<(u32, I)>);
