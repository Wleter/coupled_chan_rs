use coupled_chan::{
    Interaction, Operator,
    constants::units::{Quantity, atomic_units::AuEnergy},
    coupling::AngularBlocks,
    scaled_interaction::ScaledInteraction,
};
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel,
};
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Debug)]
pub struct RotationalEnergy {
    pub rot_const: Quantity<AuEnergy>,
    pub operator: AngularBlocks,
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

#[derive(Clone, Debug)]
pub struct DistortionEnergy {
    pub distortion: Quantity<AuEnergy>,
    pub operator: AngularBlocks,
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

impl<I: Interaction> Interaction2D<ScaledInteraction<I>> {
    pub fn scale(&mut self, scaling: PESScaling) {
        match scaling {
            PESScaling::Composite(scalings) => {
                for scaling in scalings {
                    self.scale(scaling);
                }
            }
            PESScaling::Scaling(scaling) => {
                for (_, p) in &mut self.0 {
                    p.scale(scaling);
                }
            }
            PESScaling::LegendreScaling(n, scaling) => {
                if let Some((_, p)) = self.0.iter_mut().find(|x| x.0 == n) {
                    p.scale(scaling);
                }
            }
            PESScaling::AnisotropicScaling(scaling) => {
                for (n, p) in &mut self.0 {
                    if *n != 0 {
                        p.scale(scaling);
                    }
                }
            }
            PESScaling::LegendreMorphing(_, _) => todo!("Morphing not yet implemented"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PESScaling {
    /// Perform composite scaling in the given order
    Composite(Vec<PESScaling>),
    /// Perform scaling of V(r, theta) -> lambda * V(r, theta)
    Scaling(f64),
    /// Perform scaling of legendre component V_n(r) -> lambda * V_n(r)
    LegendreScaling(u32, f64),
    /// Perform scaling of anisotropic legendre components V_n(r) -> lambda * V_n(r)
    AnisotropicScaling(f64),
    /// Perform morphing of legendre component V(r, theta) -> (1 + lambda * P_n) V(r, theta)
    LegendreMorphing(u32, f64),
}
