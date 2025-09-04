use hilbert_space::{
    cast_variant,
    dyn_space::{SpaceBasis, SubspaceElement},
};
use spin_algebra::{Spin, half_integer::HalfI32};

use crate::{
    atom_structure::{AtomStructure, AtomStructureRecipe},
    tram_basis::{TRAMBasis, TRAMBasisRecipe},
};

/// Recipe for atom-rotor problem A + B-C
#[derive(Clone, Debug, Default)]
pub struct AtomRotorTRAMProblemRecipe {
    pub atom_a: AtomStructureRecipe,
    pub atom_b: AtomStructureRecipe,
    pub atom_c: AtomStructureRecipe,
    pub tram: TRAMBasisRecipe,

    pub tot_projection: HalfI32,
}

/// Builder for atom-rotor problem A + B-C
#[derive(Clone, Debug)]
pub struct AtomRotorTRAMProblemBuilder {
    pub atom_a: AtomStructure,
    pub atom_b: AtomStructure,
    pub atom_c: AtomStructure,
    pub tram: TRAMBasis,

    tot_projection: HalfI32,

    basis: SpaceBasis,
}

impl AtomRotorTRAMProblemBuilder {
    pub fn new(recipe: AtomRotorTRAMProblemRecipe) -> Self {
        let mut basis = SpaceBasis::default();
        let atom_a = AtomStructure::new(recipe.atom_a, &mut basis);
        let atom_b = AtomStructure::new(recipe.atom_b, &mut basis);
        let atom_c = AtomStructure::new(recipe.atom_c, &mut basis);
        let tram = TRAMBasis::new(recipe.tram, &mut basis);

        Self {
            atom_a,
            atom_b,
            atom_c,
            tram,
            tot_projection: recipe.tot_projection,
            basis,
        }
    }

    pub fn filter(&self, element: &[&SubspaceElement]) -> bool {
        if !self.tram.filter(element) {
            return false;
        }

        let n_tot = cast_variant!(dyn element[self.tram.n_tot.0 as usize], Spin);
        let s_a = cast_variant!(dyn element[self.atom_a.s.0 as usize], Spin);
        let i_a = cast_variant!(dyn element[self.atom_a.i.0 as usize], Spin);
        let s_b = cast_variant!(dyn element[self.atom_b.s.0 as usize], Spin);
        let i_b = cast_variant!(dyn element[self.atom_b.i.0 as usize], Spin);
        let s_c = cast_variant!(dyn element[self.atom_c.s.0 as usize], Spin);
        let i_c = cast_variant!(dyn element[self.atom_c.i.0 as usize], Spin);

        s_a.m + i_a.m + s_b.m + i_b.m + s_c.m + i_c.m + n_tot.m == self.tot_projection
    }
}
