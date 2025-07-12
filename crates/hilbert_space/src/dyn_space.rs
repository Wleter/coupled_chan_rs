use downcast_rs::{DowncastSync, impl_downcast};
use dyn_clone::DynClone;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut, Index},
};

pub trait DynSubspaceElement: DynClone + Debug + DowncastSync {}
impl_downcast!(sync DynSubspaceElement);
dyn_clone::clone_trait_object!(DynSubspaceElement);

impl<T: Clone + Debug + Send + Sync + 'static> DynSubspaceElement for T {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasisId(pub u64);

impl Deref for BasisId {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct SubspaceBasis {
    basis: Vec<Box<dyn DynSubspaceElement>>,
    id: BasisId,
}

impl SubspaceBasis {
    pub fn new<T: DynSubspaceElement>(basis: Vec<T>) -> Self {
        assert!(!basis.is_empty(), "0 size basis is not allowed");

        let basis = basis
            .into_iter()
            .map(|x| Box::new(x) as Box<dyn DynSubspaceElement>)
            .collect();

        Self { basis, id: BasisId(0) }
    }

    pub fn elements(&self) -> &[Box<dyn DynSubspaceElement>] {
        &self.basis
    }

    pub fn size(&self) -> usize {
        self.basis.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpaceBasis(Vec<SubspaceBasis>);

impl SpaceBasis {
    pub fn new_single(space_basis: SubspaceBasis) -> Self {
        Self(vec![space_basis])
    }

    pub fn size(&self) -> usize {
        self.0.iter().fold(1, |acc, s| acc * s.size())
    }

    pub fn push_subspace(&mut self, mut state: SubspaceBasis) -> BasisId {
        let id = BasisId(self.0.len() as u64);
        state.id = id;

        self.0.push(state);
        id
    }
}

impl SpaceBasis {
    pub fn get_filtered_basis(self, f: impl Fn(&[&Box<dyn DynSubspaceElement>]) -> bool) -> BasisElements {
        let iter = BasisElementIter {
            size: self.size(),
            basis_sizes: self.0.iter().map(|x| x.size()).collect(),
            current: BasisElementIndices(vec![0; self.0.len()]),
            current_index: 0,
        };

        let mut subspaces_elements = Vec::with_capacity(self.0.len());
        for (i, b) in iter.current.iter().zip(self.0.iter()) {
            subspaces_elements.push(&b.basis[*i])
        }

        let filtered = iter
            .filter(|indices| {
                subspaces_elements
                    .iter_mut()
                    .zip(indices.iter().zip(self.0.iter()))
                    .for_each(|(s, (i, b))| *s = &b.basis[*i]);

                f(&subspaces_elements)
            })
            .collect();

        BasisElements {
            basis: self.0,
            elements_indices: filtered,
        }
    }

    pub fn get_basis(self) -> BasisElements {
        let iter = BasisElementIter {
            size: self.size(),
            basis_sizes: self.0.iter().map(|x| x.size()).collect(),
            current: BasisElementIndices(vec![0; self.0.len()]),
            current_index: 0,
        };

        BasisElements {
            basis: self.0,
            elements_indices: iter.collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasisElementIndices(Vec<usize>);

impl Deref for BasisElementIndices {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BasisElementIndices {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<BasisId> for BasisElementIndices {
    type Output = usize;

    fn index(&self, index: BasisId) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

struct BasisElementIter {
    basis_sizes: Vec<usize>,
    current_index: usize,
    current: BasisElementIndices,
    size: usize,
}

impl Iterator for BasisElementIter {
    type Item = BasisElementIndices;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            return None;
        }

        let mut current_index = self.current_index;
        for (curr, size) in self.current.iter_mut().zip(self.basis_sizes.iter()) {
            *curr = current_index % size;
            current_index /= size
        }

        self.current_index += 1;
        Some(self.current.clone())
    }
}

#[derive(Clone, Debug)]
pub struct BasisElements {
    pub basis: Vec<SubspaceBasis>,
    pub(crate) elements_indices: Vec<BasisElementIndices>,
}

impl BasisElements {
    pub fn is_empty(&self) -> bool {
        self.elements_indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements_indices.len()
    }
}

impl Index<(usize, BasisId)> for BasisElements {
    type Output = Box<dyn DynSubspaceElement>;

    fn index(&self, index: (usize, BasisId)) -> &Self::Output {
        let basis_subspace = &self.basis[index.1.0 as usize];
        let subspace_index = self.elements_indices[index.0][index.1];

        &basis_subspace.basis[subspace_index]
    }
}

impl Display for BasisElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for indices in &self.elements_indices {
            for (index, b) in indices.iter().zip(self.basis.iter()) {
                write!(f, "|{:?} ⟩ ", b.basis[*index])?
            }
            writeln!(f)?
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ElectronSpin(u32, i32);
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct NuclearSpin(u32, i32);
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Vibrational(i32);

    #[test]
    fn test_dyn_space() {
        let mut basis = SpaceBasis::default();

        let e_basis = SubspaceBasis::new(vec![
            ElectronSpin(2, -2),
            ElectronSpin(2, 0),
            ElectronSpin(2, 2),
            ElectronSpin(0, 0),
        ]);
        let e_id = basis.push_subspace(e_basis);

        let nuclear = SubspaceBasis::new(vec![NuclearSpin(1, -1), NuclearSpin(1, 1)]);
        let n_id = basis.push_subspace(nuclear);

        let vib = SubspaceBasis::new(vec![Vibrational(-1), Vibrational(-2)]);
        let vib_id = basis.push_subspace(vib);

        assert_eq!(basis.size(), 4 * 2 * 2);
        let basis_elements = basis.get_basis();
        assert_eq!(basis_elements.len(), 4 * 2 * 2);

        assert_eq!(
            basis_elements[(0, e_id)].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[(0, n_id)].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(
            basis_elements[(0, vib_id)].downcast_ref::<Vibrational>().unwrap(),
            &Vibrational(-1)
        );

        assert_eq!(
            basis_elements[(1, e_id)].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, 0)
        );
        assert_eq!(
            basis_elements[(1, n_id)].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(
            basis_elements[(1, vib_id)].downcast_ref::<Vibrational>().unwrap(),
            &Vibrational(-1)
        );

        assert_eq!(
            basis_elements[(4, e_id)].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[(4, n_id)].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, 1)
        );
        assert_eq!(
            basis_elements[(4, vib_id)].downcast_ref::<Vibrational>().unwrap(),
            &Vibrational(-1)
        );

        assert_eq!(
            basis_elements[(5, e_id)].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, 0)
        );
        assert_eq!(
            basis_elements[(5, n_id)].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, 1)
        );
        assert_eq!(
            basis_elements[(5, vib_id)].downcast_ref::<Vibrational>().unwrap(),
            &Vibrational(-1)
        );

        assert_eq!(
            basis_elements[(8, e_id)].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[(8, n_id)].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(
            basis_elements[(8, vib_id)].downcast_ref::<Vibrational>().unwrap(),
            &Vibrational(-2)
        );
    }
}
