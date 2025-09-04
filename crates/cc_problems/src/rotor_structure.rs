use coupled_chan::Operator;
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel,
};

use crate::AngularMomentum;

#[derive(Default, Debug, Clone)]
pub struct RotorStructureRecipe {
    pub n_max: u32,
    pub rot_const: f64,
    pub distortion: f64,
}

#[derive(Clone, Debug)]
pub struct RotorStructure {
    pub n: BasisId,

    pub recipe: RotorStructureRecipe,
}

impl RotorStructure {
    pub fn new(recipe: RotorStructureRecipe, space_basis: &mut SpaceBasis) -> Self {
        let n = (0..=recipe.n_max).map(AngularMomentum).collect();

        let n = space_basis.push_subspace(SubspaceBasis::new(n));

        Self { n, recipe }
    }

    pub fn rotational_energy(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            self.recipe.rot_const * (n.0 * (n.0 + 1)) as f64
        })
    }

    pub fn distortion(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.n], |[n: AngularMomentum]| {
            -self.recipe.distortion * (n.0 * (n.0 + 1)).pow(2) as f64
        })
    }
}
