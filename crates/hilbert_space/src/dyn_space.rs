use downcast_rs::{DowncastSync, impl_downcast};
use dyn_clone::DynClone;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    slice::Iter,
};

pub trait DynSubspaceElement: DynClone + Debug + DowncastSync {}
impl_downcast!(sync DynSubspaceElement);
dyn_clone::clone_trait_object!(DynSubspaceElement);

impl<T: Clone + Debug + Send + Sync + 'static> DynSubspaceElement for T {}

#[derive(Clone, Copy, Debug)]
pub struct BasisId(u64);

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
    pub fn iter_elements(&self) -> SpaceBasisIter<'_> {
        SpaceBasisIter {
            basis: self,
            subspace_basis_iter: self.0.iter().map(|s| s.elements().iter()).collect(),
            current: BasisElement(Vec::with_capacity(self.0.len())),
            current_index: 0,
            size: self.size(),
        }
    }

    pub fn get_basis(&self) -> BasisElements<'_> {
        self.iter_elements().collect()
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::borrowed_box)]
pub struct BasisElement<'a>(Vec<&'a Box<dyn DynSubspaceElement>>);

impl<'a> Deref for BasisElement<'a> {
    type Target = [&'a Box<dyn DynSubspaceElement>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for BasisElement<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for BasisElement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in self.iter() {
            write!(f, "|{s:?} ⟩ ")?
        }

        Ok(())
    }
}

pub struct SpaceBasisIter<'a> {
    basis: &'a SpaceBasis,
    subspace_basis_iter: Vec<Iter<'a, Box<dyn DynSubspaceElement>>>,
    current: BasisElement<'a>,
    current_index: usize,
    size: usize,
}

// todo! consider if copy is necessary, could be clone
impl<'a> Iterator for SpaceBasisIter<'a> {
    type Item = BasisElement<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            return None;
        }
        if self.current_index == 0 {
            for s in self.subspace_basis_iter.iter_mut() {
                let s_curr = s.next().unwrap(); // at least 1 element exists

                self.current.0.push(s_curr);
            }
            self.current_index += 1;

            return Some(self.current.clone());
        }

        for ((s_spec, s), s_type) in self
            .current
            .iter_mut()
            .zip(self.subspace_basis_iter.iter_mut())
            .zip(self.basis.0.iter())
        {
            match s.next() {
                Some(s_spec_new) => {
                    *s_spec = s_spec_new;
                    break;
                }
                None => {
                    *s = s_type.elements().iter();
                    let s_curr = s.next().unwrap(); // at least 1 element exists
                    *s_spec = s_curr;
                }
            }
        }
        self.current_index += 1;

        Some(self.current.clone())
    }
}

#[derive(Clone, Debug)]
pub struct BasisElements<'a>(Vec<BasisElement<'a>>);

impl<'a> FromIterator<BasisElement<'a>> for BasisElements<'a> {
    fn from_iter<I: IntoIterator<Item = BasisElement<'a>>>(iter: I) -> Self {
        let mut elements = BasisElements(vec![]);

        for val in iter {
            elements.0.push(val);
        }

        elements
    }
}

impl<'a> IntoIterator for BasisElements<'a> {
    type Item = BasisElement<'a>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> Deref for BasisElements<'a> {
    type Target = Vec<BasisElement<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BasisElements<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for BasisElements<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for state in &self.0 {
            writeln!(f, "{state}")?
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
        basis.push_subspace(e_basis);

        let nuclear = SubspaceBasis::new(vec![NuclearSpin(1, -1), NuclearSpin(1, 1)]);
        basis.push_subspace(nuclear);

        let vib = SubspaceBasis::new(vec![Vibrational(-1), Vibrational(-2)]);
        basis.push_subspace(vib);

        assert_eq!(basis.size(), 4 * 2 * 2);
        let basis_elements = basis.get_basis();
        assert_eq!(basis_elements.len(), 4 * 2 * 2);

        assert_eq!(
            basis_elements[0][0].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[0][1].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(basis_elements[0][2].downcast_ref::<Vibrational>().unwrap(), &Vibrational(-1));

        assert_eq!(
            basis_elements[1][0].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, 0)
        );
        assert_eq!(
            basis_elements[1][1].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(basis_elements[1][2].downcast_ref::<Vibrational>().unwrap(), &Vibrational(-1));

        assert_eq!(
            basis_elements[4][0].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[4][1].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, 1)
        );
        assert_eq!(basis_elements[4][2].downcast_ref::<Vibrational>().unwrap(), &Vibrational(-1));

        assert_eq!(
            basis_elements[5][0].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, 0)
        );
        assert_eq!(
            basis_elements[5][1].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, 1)
        );
        assert_eq!(basis_elements[5][2].downcast_ref::<Vibrational>().unwrap(), &Vibrational(-1));

        assert_eq!(
            basis_elements[8][0].downcast_ref::<ElectronSpin>().unwrap(),
            &ElectronSpin(2, -2)
        );
        assert_eq!(
            basis_elements[8][1].downcast_ref::<NuclearSpin>().unwrap(),
            &NuclearSpin(1, -1)
        );
        assert_eq!(basis_elements[8][2].downcast_ref::<Vibrational>().unwrap(), &Vibrational(-2));
    }
}
