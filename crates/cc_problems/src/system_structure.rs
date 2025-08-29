use hilbert_space::dyn_space::{BasisId, SpaceBasis, SubspaceBasis};

use crate::AngularMomentum;

pub struct SystemStructure {
    pub l: BasisId,
}

impl SystemStructure {
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
