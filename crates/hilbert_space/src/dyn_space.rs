use downcast_rs::{DowncastSync, impl_downcast};
use dyn_clone::DynClone;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut, Index},
    ptr,
};

pub trait DynSubspaceElement: DynClone + Debug + DowncastSync {}
impl_downcast!(sync DynSubspaceElement);
dyn_clone::clone_trait_object!(DynSubspaceElement);

impl<T: Clone + Debug + Send + Sync + 'static> DynSubspaceElement for T {}

#[derive(Clone, Debug)]
pub struct SubspaceElement(Box<dyn DynSubspaceElement>);

impl Deref for SubspaceElement {
    type Target = Box<dyn DynSubspaceElement>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
    basis: Vec<SubspaceElement>,
    id: BasisId,
}

impl Eq for SubspaceBasis {}

impl PartialEq for SubspaceBasis {
    fn eq(&self, other: &Self) -> bool {
        let elements_same = self.basis.iter().zip(other.basis.iter()).all(|(a, b)| ptr::eq(a, b));

        self.basis.len() == other.basis.len() && elements_same && self.id == other.id
    }
}

impl SubspaceBasis {
    pub fn new<T: DynSubspaceElement>(basis: Vec<T>) -> Self {
        assert!(!basis.is_empty(), "0 size basis is not allowed");

        let basis = basis.into_iter().map(|x| SubspaceElement(Box::new(x))).collect();

        Self { basis, id: BasisId(0) }
    }

    pub fn elements(&self) -> &[SubspaceElement] {
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

    pub fn subspaces_len(&self) -> usize {
        self.0.len()
    }

    pub fn push_subspace(&mut self, mut state: SubspaceBasis) -> BasisId {
        let id = BasisId(self.0.len() as u64);
        state.id = id;

        self.0.push(state);
        id
    }
}

impl SpaceBasis {
    pub fn get_filtered_basis(&self, f: impl Fn(&[&SubspaceElement]) -> bool) -> BasisElements {
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
            basis: self.clone(),
            elements_indices: filtered,
        }
    }

    pub fn get_basis(&self) -> BasisElements {
        let iter = BasisElementIter {
            size: self.size(),
            basis_sizes: self.0.iter().map(|x| x.size()).collect(),
            current: BasisElementIndices(vec![0; self.0.len()]),
            current_index: 0,
        };

        BasisElements {
            basis: self.clone(),
            elements_indices: iter.collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasisElementIndices(Vec<usize>);

impl BasisElementIndices {
    #[allow(clippy::borrowed_box)]
    pub fn index<'a>(&'a self, index: BasisId, basis: &'a SpaceBasis) -> &'a SubspaceElement {
        let basis_subspace = &basis.0[index.0 as usize];
        let subspace_index = self[index];

        &basis_subspace.basis[subspace_index]
    }
}

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

#[derive(Clone)]
pub struct BasisElements {
    pub basis: SpaceBasis,
    pub elements_indices: Vec<BasisElementIndices>,
}

// todo! duplicate code
impl BasisElements {
    pub fn as_ref<'a>(&'a self) -> BasisElementsRef<'a> {
        BasisElementsRef {
            basis: &self.basis,
            elements_indices: &self.elements_indices,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.elements_indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements_indices.len()
    }
}

impl Index<(usize, BasisId)> for BasisElements {
    type Output = SubspaceElement;

    fn index(&self, index: (usize, BasisId)) -> &Self::Output {
        let basis_subspace = &self.basis.0[index.1.0 as usize];
        let subspace_index = self.elements_indices[index.0][index.1];

        &basis_subspace.basis[subspace_index]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BasisElementsRef<'a> {
    pub basis: &'a SpaceBasis,
    pub elements_indices: &'a [BasisElementIndices],
}

impl BasisElementsRef<'_> {
    pub fn as_ref<'a>(&'a self) -> BasisElementsRef<'a> {
        *self
    }

    pub fn is_empty(&self) -> bool {
        self.elements_indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements_indices.len()
    }

    pub fn indices_iter(&self) -> impl Iterator<Item = usize> {
        0..self.len()
    }
}

impl<'a> Index<(usize, BasisId)> for BasisElementsRef<'a> {
    type Output = SubspaceElement;

    fn index(&self, index: (usize, BasisId)) -> &'a Self::Output {
        let basis_subspace = &self.basis.0[index.1.0 as usize];
        let subspace_index = self.elements_indices[index.0][index.1];

        &basis_subspace.basis[subspace_index]
    }
}

impl Display for BasisElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for indices in &self.elements_indices {
            for (index, b) in indices.iter().zip(self.basis.0.iter()) {
                write!(f, "|{:?} ⟩ ", b.basis[*index])?
            }
            writeln!(f)?
        }

        Ok(())
    }
}

impl Debug for BasisElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for indices in &self.elements_indices {
            for (index, b) in indices.iter().zip(self.basis.0.iter()) {
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
