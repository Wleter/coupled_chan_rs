use coupled_chan::{
    Operator,
    constants::{BOHR_MAG, G_FACTOR},
};
use hilbert_space::{
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel, operator_mel,
};
use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

#[derive(Clone, Debug)]
pub struct AtomStructure {
    pub s: BasisId,
    pub i: BasisId,

    pub recipe: AtomStructureRecipe,
}

#[derive(Clone, Copy, Debug)]
pub struct AtomStructureRecipe {
    pub s: HalfU32,
    pub i: HalfU32,

    pub gamma_e: f64,
    pub gamma_i: f64,
    pub a_hifi: f64,
}

impl Default for AtomStructureRecipe {
    fn default() -> Self {
        Self {
            s: Default::default(),
            i: Default::default(),
            gamma_e: -G_FACTOR * BOHR_MAG,
            gamma_i: Default::default(),
            a_hifi: Default::default(),
        }
    }
}

impl AtomStructure {
    pub fn new(recipe: AtomStructureRecipe, space_basis: &mut SpaceBasis) -> Self {
        let s = get_spin_basis(recipe.s);
        let i = get_spin_basis(recipe.i);

        let s = space_basis.push_subspace(SubspaceBasis::new(s));
        let i = space_basis.push_subspace(SubspaceBasis::new(i));

        Self { s, i, recipe }
    }

    pub fn zeeman_prop(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.s, self.i], |[s: Spin, i: Spin]| {
            -self.recipe.gamma_e * s.m.value() - self.recipe.gamma_i * i.m.value()
        })
    }

    pub fn hyperfine(&self, basis: &BasisElementsRef) -> Operator {
        operator_mel!(dyn basis, [self.s, self.i], |[s: Spin, i: Spin]| {
            self.recipe.a_hifi * SpinOps::dot(s, i)
        })
    }
}
