use coupled_chan::coupling::AngularBlocks;
use hilbert_space::space::{
    BasisElementIndices,
    BasisElements,
    BasisElementsRef,
    BasisId,
    DynSubspaceElement,
    SpaceBasis,
    SubspaceBasisOf,
};
use serde::Deserialize;
use spin_algebra::angular::{
    Angular,
    angular_range,
};
use spin_algebra::get_spin_basis;

use crate::Operator;

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
    pub fn from_orbital(full_basis: BasisElements, orbital_id: &OrbitalBasis) -> Self {
        Self::new(full_basis, orbital_id.l, |&a| a.l.double_value() / 2)
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

    pub fn zero_angular_block(&self) -> AngularBlocks {
        let blocks = self.angular_iter().map(|(_, e)| Operator::zeros(e.len()).0).collect();

        AngularBlocks {
            l: self.ls.clone(),
            blocks,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrbitalParity {
    #[default]
    All,
    Even,
    Odd,
}

#[derive(Debug, Clone, Copy)]
// Struct for storing |l m_l> state
pub struct OrbitalBasis {
    pub l: BasisId<Angular>,
}

impl OrbitalBasis {
    pub fn new(recipe: OrbitalRecipe, basis: &mut SpaceBasis) -> Self {
        let l = recipe.basis();
        let l = basis.push_subspace(SubspaceBasisOf::new(l));

        Self { l }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl Default for OrbitalRecipe {
    fn default() -> Self {
        Self::Single(0)
    }
}

impl OrbitalRecipe {
    pub fn basis(&self) -> Vec<Angular> {
        match self {
            OrbitalRecipe::Single(ang_l) => vec![Angular::new(*ang_l, 0)],
            OrbitalRecipe::LMax(l_max) => angular_range(*l_max),
            OrbitalRecipe::LMaxProjections(l_max) => (0..=*l_max)
                .flat_map(|l| get_spin_basis(l).into_iter().map(|x| x.into()))
                .collect(),
        }
    }

    pub fn magnitudes(&self) -> Vec<u32> {
        match self {
            OrbitalRecipe::Single(ang_l) => vec![*ang_l],
            OrbitalRecipe::LMax(l_max) | OrbitalRecipe::LMaxProjections(l_max) => (0..=*l_max).collect(),
        }
    }
}
