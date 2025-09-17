pub mod alkali_atom_rotor;
pub mod alkali_diatom;
pub mod alkali_homo_diatom;
pub mod atom_structure;
pub mod operator_mel;
pub mod rotor_structure;
pub mod system_structure;
pub mod tram_basis;
pub use coupled_chan;

use coupled_chan::{Operator, coupling::AngularBlocks};
use hilbert_space::{
    cast_variant,
    dyn_space::{BasisElementIndices, BasisElements, BasisElementsRef, BasisId, DynSubspaceElement},
};

use crate::system_structure::SystemStructure;

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct AngularMomentum(pub u32);

pub struct AngularBasisElements {
    pub full_basis: BasisElements,
    ls: Vec<AngularMomentum>,
    separated_basis_indices: Vec<Vec<BasisElementIndices>>,
}

impl AngularBasisElements {
    pub fn new_system(full_basis: BasisElements, system: &SystemStructure) -> Self {
        Self::new(full_basis, system.l, |&a| a)
    }

    pub fn new<T: DynSubspaceElement>(
        full_basis: BasisElements,
        l_index: BasisId,
        conversion: impl Fn(&T) -> AngularMomentum,
    ) -> Self {
        let basis = full_basis.basis;

        let mut angular_indices: Vec<(AngularMomentum, BasisElementIndices)> = full_basis
            .elements_indices
            .into_iter()
            .map(|indices| (conversion(cast_variant!(dyn indices.index(l_index, &basis), T)), indices))
            .collect();
        angular_indices.sort_by_key(|(l, _)| *l);
        let ordered_indices = angular_indices.iter().map(|x| x.1.clone()).collect();

        let ordered_basis = BasisElements {
            basis,
            elements_indices: ordered_indices,
        };

        let mut l_prev: Option<AngularMomentum> = None;
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

    pub fn angular_iter<'a>(&'a self) -> impl Iterator<Item = BasisElementsRef<'a>> {
        self.separated_basis_indices.iter().map(|indices| BasisElementsRef {
            basis: &self.full_basis.basis,
            elements_indices: indices,
        })
    }

    pub fn get_angular_blocks(&self, mut f: impl FnMut(&BasisElementsRef) -> Operator) -> AngularBlocks {
        let blocks = self.angular_iter().map(|e| f(&e)).collect();

        AngularBlocks {
            l: self.ls.iter().map(|a| a.0).collect(),
            angular_blocks: blocks,
        }
    }
}
