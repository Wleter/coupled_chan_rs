pub mod atom_basis;
pub mod diatom_basis;
pub mod operator_mel;
pub mod hamiltonian;

pub use cc_qol_utils;
use hilbert_space::space::{
    BasisId,
    SpaceBasis,
    SubspaceBasis,
};
use spin_algebra::{
    Spin, SpinLike, get_spin_basis, half_integer::{HalfI32, HalfU32}
};

#[derive(Clone, Copy, PartialEq, Default, Hash)]
pub struct Angular {
    pub l: HalfU32,
    pub m: HalfI32,
}

impl std::fmt::Debug for Angular {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.l, self.m)
    }
}

impl SpinLike for Angular {
    fn s(&self) -> HalfU32 {
        self.l
    }

    fn m(&self) -> HalfI32 {
        self.m
    }
}

impl From<Spin> for Angular {
    fn from(value: Spin) -> Self {
        assert!(value.s.double_value() & 1 == 0, "Can only convert non-half integers");
        assert!(value.m.double_value() & 1 == 0, "Can only convert non-half integers");

        Self { l: value.s, m: value.m }
    }
}

impl Into<Spin> for Angular {
    fn into(self) -> Spin {
        Spin::new(self.l, self.m)
    }
}

impl Angular {
    pub fn new(l: u32, m: i32) -> Self {
        Self { l: l.into(), m: m.into() }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OrbitalRecipe {
    /// Single l is still inserted into the basis
    /// as |l 0>
    Single(u32),
    /// l = 0..=l_max is inserted into the basis
    /// as |l 0>
    LMax(u32),
    /// l = 0..=l_max, m = -l..=l is inserted into the basis
    /// as |l m>
    LMaxProjections(u32),
}

impl OrbitalRecipe {
    pub fn basis(&self) -> Vec<Angular> {
        match self {
            OrbitalRecipe::Single(ang_l) => vec![Angular::new(*ang_l, 0)],
            OrbitalRecipe::LMax(l_max) => {
                angular_range(*l_max)
            }
            OrbitalRecipe::LMaxProjections(l_max) => {
                get_spin_basis((*l_max).into()).into_iter()
                    .map(|l| l.into())
                    .collect()
            }
        }
    }

    pub fn magnitudes(&self) -> Vec<u32> {
        match self {
            OrbitalRecipe::Single(ang_l) => vec![*ang_l],
            OrbitalRecipe::LMax(l_max) 
            | OrbitalRecipe::LMaxProjections(l_max) => {
                (0..=*l_max).collect()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
// Struct for storing |l m_l> state
pub struct OrbitalBasis {
    pub l: BasisId<Angular>
}

impl OrbitalBasis {
    pub fn new(recipe: OrbitalRecipe, basis: &mut SpaceBasis) -> Self {
        let l = recipe.basis();
        let l = basis.push_subspace(SubspaceBasis::new(l));

        Self { l }
    }
}

pub fn angular_range(l_max: u32) -> Vec<Angular> {
    (0..=l_max).map(|x| Angular::new(x, 0)).collect()
}
