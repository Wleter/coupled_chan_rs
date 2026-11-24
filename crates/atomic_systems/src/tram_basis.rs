use crate::Parity;
use hilbert_space::space::{
    BasisId,
    SpaceBasis,
    SubspaceBasisOf,
};
use serde::Deserialize;
use spin_algebra::{
    SpinPair, SpinPairMag, get_spin_pair_basis, spin,
};

pub type AngularCoupled = SpinPair<u32, u32>;

/// Recipe for the total rotational angular momentum basis
/// for the molecule
#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct TRAMRecipe {
    /// maximal value of molecule rotation
    pub n_max: u32,
    /// maximal value of angular momentum,
    /// if None all allowed (by n_tot) values of l are included
    #[serde(default)]
    pub l_max: Option<u32>,
    /// maximal value of total rotational angular momentum
    pub n_tot: TotalRotationRecipe,

    #[serde(default)]
    pub parity: Parity,
}

#[derive(Debug, Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(deny_unknown_fields)]
pub enum TotalRotationRecipe {
    /// Single rotational value |N 0>
    Single(u32),
    /// Single rotational value |N M_N>
    SingleWithProjection(u32, i32),
    /// rotational basis from |0 0> up to |N 0>
    UpTo(u32),
    /// rotational basis from |0 0> up to |N M_N> with all projections
    UpToAllProjections(u32),
}

impl TotalRotationRecipe {
    pub fn max_value(&self) -> u32 {
        match *self {
            TotalRotationRecipe::Single(n) => n,
            TotalRotationRecipe::SingleWithProjection(n, _) => n,
            TotalRotationRecipe::UpTo(n) => n,
            TotalRotationRecipe::UpToAllProjections(n) => n,
        }
    }
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

        let n_tot_max = recipe.n_tot.max_value();
        let l_max = match recipe.l_max {
            Some(l_max) => l_max,
            None => recipe.n_max + n_tot_max,
        };

        for l in 0..=l_max {
            let n_min = l.saturating_sub(n_tot_max);
            let n_max = (l + n_tot_max).min(recipe.n_max);
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
                let n_tot_min = u32::abs_diff(l, n);
                let n_tot_max = (l + n).min(n_tot_max);

                match recipe.n_tot {
                    TotalRotationRecipe::Single(n_tot) => {
                        if (n_tot_min..=n_tot_max).contains(&n_tot) {
                            basis.extend([spin!((n, l), n_tot, 0)])
                        }
                    },
                    TotalRotationRecipe::SingleWithProjection(n_tot, m_n) => {
                        if (n_tot_min..=n_tot_max).contains(&n_tot) {
                            basis.extend([spin!((n, l), n_tot, m_n)])
                        }
                    },
                    TotalRotationRecipe::UpTo(_) => {
                        let b: Vec<SpinPair<u32, u32>> =
                            (n_tot_min..=n_tot_max).map(|n_tot| spin!((n, l), n_tot, 0)).collect();

                        basis.extend(b)
                    },
                    TotalRotationRecipe::UpToAllProjections(_) => {
                        let b: Vec<SpinPairMag<u32, u32>> =
                            (n_tot_min..=n_tot_max).map(|n_tot| spin!((n, l), n_tot.into())).collect();

                        basis.extend(get_spin_pair_basis(b))
                    },
                }

            }
        }

        basis
    }
}
