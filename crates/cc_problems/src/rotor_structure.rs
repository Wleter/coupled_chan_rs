use coupled_chan::{
    constants::units::{atomic_units::AuEnergy, Quantity}, coupling::AngularBlocks, Interaction, Operator
};
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel,
};

use crate::{AngularBasisElements, AngularMomentum};

#[derive(Clone, Debug)]
pub struct RotorBasis {
    pub n: BasisId,
}

impl RotorBasis {
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

#[derive(Clone, Debug,)]
pub struct RotationalEnergy {
    pub rot_const: Quantity<AuEnergy>,
    pub operator: AngularBlocks
}

impl RotationalEnergy {
    pub fn new(basis: &AngularBasisElements, rot: &RotorBasis) -> Self {
        
        let operator = basis.get_angular_blocks(|basis| {
            operator_diag_mel!(dyn basis, [rot.n], |[n: AngularMomentum]| {
                (n.0 * (n.0 + 1)) as f64
            })
        });

        Self {
            rot_const: Default::default(),
            operator,
        }
    }
    
    pub fn hamiltonian(&self) -> AngularBlocks {
        self.operator.scale(self.rot_const.value())
    }
}

#[derive(Clone, Debug,)]
pub struct DistortionEnergy {
    pub distortion: Quantity<AuEnergy>,
    pub operator: AngularBlocks
}

impl DistortionEnergy {
    pub fn new(basis: &AngularBasisElements, rot: &RotorBasis) -> Self {
        let operator = basis.get_angular_blocks(|basis| {
            operator_diag_mel!(dyn basis, [rot.n], |[n: AngularMomentum]| {
                -((n.0 * (n.0 + 1)).pow(2) as f64)
            })
        });

        Self {
            distortion: Default::default(),
            operator,
        }
    }
    
    pub fn hamiltonian(&self) -> AngularBlocks {
        self.operator.scale(self.distortion.value())
    }
}

#[derive(Clone, Debug)]
pub struct Interaction2D<I: Interaction>(pub Vec<(u32, I)>);
