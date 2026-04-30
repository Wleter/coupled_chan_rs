pub mod atom_basis;
pub mod diatom_basis;
pub mod hamiltonian;
pub mod hamiltonian_terms;
pub mod operator_mel;

pub use cc_qol_utils;
use coupled_chan::coupling::AngularBlocks;
use hilbert_space::space::{
    BasisElementIndices,
    BasisElements,
    BasisElementsRef,
    BasisId,
    DynSubspaceElement,
    SpaceBasis,
    SubspaceBasis,
};
use spin_algebra::{
    Spin,
    SpinLike,
    SpinMagLike,
    get_spin_basis,
    half_integer::{
        HalfI32,
        HalfU32,
    },
};

use crate::hamiltonian::Operator;

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

impl SpinMagLike for Angular {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.l
    }
}

impl SpinLike for Angular {
    #[inline]
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
        Self {
            l: l.into(),
            m: m.into(),
        }
    }

    pub fn l_value(&self) -> u32 {
        self.l.double_value() / 2
    }

    pub fn m_value(&self) -> i32 {
        self.m.double_value() / 2
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
            OrbitalRecipe::LMax(l_max) => angular_range(*l_max),
            OrbitalRecipe::LMaxProjections(l_max) => get_spin_basis((*l_max).into()).into_iter().map(|l| l.into()).collect(),
        }
    }

    pub fn magnitudes(&self) -> Vec<u32> {
        match self {
            OrbitalRecipe::Single(ang_l) => vec![*ang_l],
            OrbitalRecipe::LMax(l_max) | OrbitalRecipe::LMaxProjections(l_max) => (0..=*l_max).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
// Struct for storing |l m_l> state
pub struct OrbitalBasis {
    pub l: BasisId<Angular>,
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

#[derive(Clone)]
/// Struct for storing basis elements with sorted and
/// separated by orbital angular momentum number elements.
pub struct OrbitalBasisElements {
    pub full_basis: BasisElements,
    ls: Vec<u32>,
    separated_basis_indices: Vec<Vec<BasisElementIndices>>,
}

impl std::fmt::Debug for OrbitalBasisElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrbitalBasisElements")
            .field("full_basis", &self.full_basis)
            .finish()
    }
}

impl OrbitalBasisElements {
    pub fn from_orbital(full_basis: BasisElements, system: &OrbitalBasis) -> Self {
        Self::new(full_basis, system.l, |&a| a.l.double_value() / 2)
    }

    pub fn new<T: DynSubspaceElement>(
        full_basis: BasisElements,
        l_index: BasisId<T>,
        element_to_l: impl Fn(&T) -> u32,
    ) -> Self {
        let basis = full_basis.basis;

        let mut angular_indices: Vec<(u32, BasisElementIndices)> = full_basis
            .elements_indices
            .into_iter()
            .map(|indices| (element_to_l(indices.index(l_index, &basis)), indices))
            .collect();
        angular_indices.sort_by_key(|(l, _)| *l);
        let ordered_indices = angular_indices.iter().map(|x| x.1.clone()).collect();

        let ordered_basis = BasisElements {
            basis,
            elements_indices: ordered_indices,
        };

        let mut l_prev: Option<u32> = None;
        let mut separated_basis_indices = vec![];
        let mut ls = vec![];
        for (l, index) in angular_indices {
            let l_changed = if let Some(l_prev) = l_prev { l_prev != l } else { true };

            if l_changed {
                ls.push(l);
                separated_basis_indices.push(vec![index])
            } else {
                separated_basis_indices.last_mut().unwrap().push(index)
            }

            l_prev = Some(l)
        }

        Self {
            full_basis: ordered_basis,
            ls,
            separated_basis_indices,
        }
    }

    pub fn new_implicit(full_basis: BasisElements, l: u32) -> Self {
        Self {
            separated_basis_indices: vec![full_basis.elements_indices.clone()],
            full_basis,
            ls: vec![l],
        }
    }

    pub fn angular_iter<'a>(&'a self) -> impl Iterator<Item = (u32, BasisElementsRef<'a>)> {
        self.ls.iter().zip(&self.separated_basis_indices).map(|(&l, indices)| {
            let elements = BasisElementsRef {
                basis: &self.full_basis.basis,
                elements_indices: indices,
            };

            (l, elements)
        })
    }

    pub fn get_angular_blocks(&self, mut f: impl FnMut(u32, BasisElementsRef) -> Operator) -> AngularBlocks {
        let blocks = self.angular_iter().map(|(l, e)| f(l, e).0).collect();

        AngularBlocks {
            l: self.ls.clone(),
            blocks,
        }
    }
}
