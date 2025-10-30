use hilbert_space::{
    Parity, cast_variant,
    dyn_space::{BasisId, SpaceBasis, SpaceElement, SubspaceBasis},
};
use spin_algebra::{Spin, get_spin_basis};

use crate::{AngularMomentum, rotor_structure::RotorBasis, system_structure::AngularBasis};

#[derive(Clone, Debug, Default)]
pub struct TRAMBasisRecipe {
    pub l_max: u32,
    pub n_max: u32,
    pub n_tot_max: u32,
    pub parity: Parity,
}

#[derive(Clone, Debug)]
pub struct TRAMBasis {
    pub l: AngularBasis,
    pub n: RotorBasis,
    pub n_tot: BasisId,
    pub parity: Parity,
}

impl TRAMBasis {
    pub fn new(recipe: TRAMBasisRecipe, space_basis: &mut SpaceBasis) -> Self {
        let n = RotorBasis::new(recipe.n_max, space_basis);
        let l = AngularBasis::new(AngularMomentum(recipe.l_max), space_basis);

        let n_tot = (0..=recipe.n_tot_max)
            .flat_map(|n_tot| get_spin_basis(n_tot.into()))
            .collect();

        let n_tot = space_basis.push_subspace(SubspaceBasis::new(n_tot));

        Self {
            n,
            l,
            n_tot,
            parity: recipe.parity,
        }
    }

    pub fn new_single_n_tot(recipe: TRAMBasisRecipe, space_basis: &mut SpaceBasis) -> Self {
        let n = RotorBasis::new(recipe.n_max, space_basis);
        let l = AngularBasis::new(AngularMomentum(recipe.l_max), space_basis);

        let n_tot = get_spin_basis(recipe.n_tot_max.into());
        let n_tot = space_basis.push_subspace(SubspaceBasis::new(n_tot));

        Self {
            n,
            l,
            n_tot,
            parity: recipe.parity,
        }
    }

    pub fn filter(&self, element: SpaceElement) -> bool {
        let l = cast_variant!(dyn element[self.l.l], AngularMomentum);
        let n = cast_variant!(dyn element[self.n.n], AngularMomentum);
        let n_tot_spin = cast_variant!(dyn element[self.n_tot], Spin);
        let n_tot = n_tot_spin.s.double_value() / 2;

        (l.0 + n.0) >= n_tot
            && (l.0 as i32 - n.0 as i32).unsigned_abs() <= n_tot
            && match self.parity {
                Parity::All => true,
                Parity::Even => (l.0 + n.0) & 1 == 0,
                Parity::Odd => (l.0 + n.0) & 1 == 1,
            }
    }
}
