use crate::Parity;
use hilbert_space::space::{
    BasisId,
    SpaceBasis,
    SubspaceBasisOf,
};
use serde::Deserialize;
use spin_algebra::{
    SpinPair,
    SpinPairMag,
    get_spin_pair_basis,
    spin,
};

pub type AngularCoupled = SpinPair<u32, u32>;

/// Recipe for the total rotational angular momentum basis
/// for the molecule
#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct TRAMRecipe {
    /// maximal value of molecule rotation
    pub n_max: u32,
    /// maximal value of angular momentum
    pub l_max: u32,
    /// maximal value of total rotational angular momentum
    pub n_tot_max: u32,
    #[serde(default)]
    pub parity: Parity,
}

#[derive(Debug, Clone, Copy)]
pub struct TRAMBasis {
    pub tram: BasisId<AngularCoupled>,
}

impl TRAMBasis {
    /// Adds |(n l) N M_N>
    /// to the basis.
    pub fn new(recipe: &TRAMRecipe, basis: &mut SpaceBasis) -> Self {
        let id = basis.push_subspace(SubspaceBasisOf::new(Self::basis(recipe)));

        Self { tram: id }
    }

    pub fn basis(recipe: &TRAMRecipe) -> Vec<AngularCoupled> {
        let mut basis = vec![];

        for l in 0..=recipe.l_max {
            let n_min = l.saturating_sub(recipe.n_tot_max);
            let n_max = (l + recipe.n_tot_max).min(recipe.n_max);
            for n in n_min..=n_max {
                match recipe.parity {
                    Parity::All => (),
                    Parity::Even => {
                        if (n + l) & 1 == 1 {
                            continue;
                        }
                    }
                    Parity::Odd => {
                        if (n + l) & 1 == 0 {
                            continue;
                        }
                    }
                }
                let b: Vec<SpinPairMag<u32, u32>> =
                    (0..=recipe.n_tot_max).map(|n_tot| spin!((n, l), n_tot.into())).collect();

                basis.extend(get_spin_pair_basis(b))
            }
        }

        basis
    }
}
