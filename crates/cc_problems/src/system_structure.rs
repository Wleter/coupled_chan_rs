use coupled_chan::constants::units::{
    Quantity,
    atomic_units::{AuEnergy, AuMass},
};
use hilbert_space::dyn_space::{BasisId, SpaceBasis, SubspaceBasis};
use serde::{Deserialize, Serialize};

use crate::AngularMomentum;

#[derive(Clone, Debug)]
pub struct AngularBasis {
    pub l: BasisId,
}

impl AngularBasis {
    pub fn new_single(l: AngularMomentum, space_basis: &mut SpaceBasis) -> Self {
        let l = space_basis.push_subspace(SubspaceBasis::new(vec![l]));

        Self { l }
    }

    pub fn new(l_max: AngularMomentum, space_basis: &mut SpaceBasis) -> Self {
        let l = (0..=l_max.0).map(AngularMomentum).collect();

        let l = space_basis.push_subspace(SubspaceBasis::new(l));

        Self { l }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SystemParams {
    pub mass: Quantity<AuMass>,
    pub energy: Quantity<AuEnergy>,
    pub entrance_channel: usize,
}

impl Default for SystemParams {
    fn default() -> Self {
        Self {
            mass: 1. * AuMass,
            energy: Default::default(),
            entrance_channel: 0,
        }
    }
}

impl SystemParams {
    pub fn red_masses(masses: &[Quantity<AuMass>]) -> Quantity<AuMass> {
        1. / masses.iter().fold(0., |acc, x| acc + 1. / x.value()) * AuMass
    }
}
