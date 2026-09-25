use hilbert_space::{Parity, space::{BasisId, SpaceBasis, SubspaceBasisOf}};
use serde::{Deserialize, Serialize};
use spin_algebra::{SpinPair, get_spin_pair_basis, get_spin_pair_magnitudes};

pub type AngularCoupled = SpinPair<u32, u32>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TRAMRecipe {
    pub n_max: u32,
    pub l_max: u32,
    pub n_tot_max: u32,
    #[serde(default)]
    pub parity: Parity,
}

#[derive(Debug, Clone, Copy)]
pub struct TRAMBasis {
    pub tram: BasisId<AngularCoupled>
}

impl TRAMBasis {
    /// Adds |(n l) N M_N>
    /// to the basis.
    pub fn new(recipe: &TRAMRecipe, basis: &mut SpaceBasis) -> Self {
        let id = basis.push_subspace(SubspaceBasisOf::new(Self::basis(recipe)));

        Self {
            tram: id,
        }
    }

    pub fn basis(recipe: &TRAMRecipe) -> Vec<AngularCoupled> {
        let mut basis = vec![];

        for l in 0..=recipe.l_max {
            let n_min = l.saturating_sub(recipe.n_max);
            let n_max = (l + recipe.n_max).min(recipe.n_max);
            for n in n_min..=n_max {
                match recipe.parity {
                    Parity::All => (),
                    Parity::Even => if (n + l) & 1 == 1 { continue },
                    Parity::Odd => if (n + l) & 1 == 0 { continue },
                }

                basis.extend(get_spin_pair_basis(get_spin_pair_magnitudes([l], [n])))
            }
        }

        basis
    }
}