use std::{
    fmt::{Debug, Display},
    mem::{Discriminant, discriminant},
    ops::{Deref, DerefMut},
    slice::Iter,
};

#[derive(Clone, Debug)]
pub struct SubspaceBasis<T> {
    basis: Vec<T>,
    variant: Discriminant<T>,
}

impl<T> SubspaceBasis<T> {
    pub fn new(basis: Vec<T>) -> Self {
        assert!(!basis.is_empty(), "0 size basis is not allowed");

        let variant = discriminant(basis.first().unwrap());
        assert!(
            basis.iter().all(|x| discriminant(x) == variant),
            "some basis elements are for different subspace"
        );

        Self { basis, variant }
    }

    pub fn elements(&self) -> &[T] {
        &self.basis
    }

    pub fn variant(&self) -> &Discriminant<T> {
        &self.variant
    }

    pub fn size(&self) -> usize {
        self.basis.len()
    }
}

#[derive(Clone, Debug)]
pub struct SpaceBasis<T>(Vec<SubspaceBasis<T>>);

impl<T> Default for SpaceBasis<T> {
    fn default() -> Self {
        Self(vec![])
    }
}

impl<T> SpaceBasis<T> {
    pub fn new_single(space_basis: SubspaceBasis<T>) -> Self {
        Self(vec![space_basis])
    }

    pub fn size(&self) -> usize {
        self.0.iter().fold(1, |acc, s| acc * s.size())
    }

    pub fn push_subspace(&mut self, state: SubspaceBasis<T>) -> &mut Self {
        let variant = state.variant();

        if self.0.iter().any(|x| x.variant() == variant) {
            panic!("Subspace basis is already pushed into space basis");
        }

        self.0.push(state);
        self
    }
}

impl<T: Copy> SpaceBasis<T> {
    pub fn iter_elements(&self) -> SpaceBasisIter<'_, T> {
        SpaceBasisIter {
            basis: self,
            subspace_basis_iter: self.0.iter().map(|s| s.elements().iter()).collect(),
            current: BasisElement(Vec::with_capacity(self.0.len())),
            current_index: 0,
            size: self.size(),
        }
    }

    pub fn get_basis(&self) -> BasisElements<T> {
        self.iter_elements().collect()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BasisElement<T>(Vec<T>);

impl<T> Deref for BasisElement<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for BasisElement<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Debug> Display for BasisElement<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in self.iter() {
            write!(f, "|{s:?} ⟩ ")?
        }

        Ok(())
    }
}

pub struct SpaceBasisIter<'a, T> {
    basis: &'a SpaceBasis<T>,
    subspace_basis_iter: Vec<Iter<'a, T>>,
    current: BasisElement<T>,
    current_index: usize,
    size: usize,
}

// todo! consider if copy is necessary, could be just clone
impl<T: Copy> Iterator for SpaceBasisIter<'_, T> {
    type Item = BasisElement<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.size {
            return None;
        }
        if self.current_index == 0 {
            for s in self.subspace_basis_iter.iter_mut() {
                let s_curr = s.next().unwrap(); // at least 1 element exists

                self.current.0.push(*s_curr);
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
                    *s_spec = *s_spec_new;
                    break;
                }
                None => {
                    *s = s_type.elements().iter();
                    let s_curr = s.next().unwrap(); // at least 1 element exists
                    *s_spec = *s_curr;
                }
            }
        }
        self.current_index += 1;

        Some(self.current.clone())
    }
}

#[derive(Debug, Clone)]
pub struct BasisElements<T>(Vec<BasisElement<T>>);

impl<T> FromIterator<BasisElement<T>> for BasisElements<T> {
    fn from_iter<I: IntoIterator<Item = BasisElement<T>>>(iter: I) -> Self {
        let mut elements = BasisElements(vec![]);

        for val in iter {
            elements.0.push(val);
        }

        elements
    }
}

impl<T> IntoIterator for BasisElements<T> {
    type Item = BasisElement<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T> Deref for BasisElements<T> {
    type Target = Vec<BasisElement<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for BasisElements<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Debug> Display for BasisElements<T> {
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

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    enum DiatomCoupledBasis {
        ElectronSpin((u32, i32)),
        NuclearSpin((u32, i32)),
        Vibrational(i32),
    }

    #[test]
    fn test_static_space() {
        let mut basis = SpaceBasis::default();

        let e_basis = SubspaceBasis::new(vec![
            DiatomCoupledBasis::ElectronSpin((2, -2)),
            DiatomCoupledBasis::ElectronSpin((2, 0)),
            DiatomCoupledBasis::ElectronSpin((2, 2)),
            DiatomCoupledBasis::ElectronSpin((0, 0)),
        ]);
        basis.push_subspace(e_basis);

        let nuclear = SubspaceBasis::new(vec![
            DiatomCoupledBasis::NuclearSpin((1, -1)),
            DiatomCoupledBasis::NuclearSpin((1, 1)),
        ]);
        basis.push_subspace(nuclear);

        let vib = SubspaceBasis::new(vec![DiatomCoupledBasis::Vibrational(-1), DiatomCoupledBasis::Vibrational(-2)]);
        basis.push_subspace(vib);

        assert_eq!(basis.size(), 4 * 2 * 2);

        let basis_elements = basis.get_basis();

        assert_eq!(basis_elements.len(), 4 * 2 * 2);

        assert_eq!(
            basis_elements[0],
            BasisElement(vec![
                DiatomCoupledBasis::ElectronSpin((2, -2)),
                DiatomCoupledBasis::NuclearSpin((1, -1)),
                DiatomCoupledBasis::Vibrational(-1)
            ])
        );

        assert_eq!(
            basis_elements[1],
            BasisElement(vec![
                DiatomCoupledBasis::ElectronSpin((2, 0)),
                DiatomCoupledBasis::NuclearSpin((1, -1)),
                DiatomCoupledBasis::Vibrational(-1)
            ])
        );

        assert_eq!(
            basis_elements[4],
            BasisElement(vec![
                DiatomCoupledBasis::ElectronSpin((2, -2)),
                DiatomCoupledBasis::NuclearSpin((1, 1)),
                DiatomCoupledBasis::Vibrational(-1)
            ])
        );

        assert_eq!(
            basis_elements[5],
            BasisElement(vec![
                DiatomCoupledBasis::ElectronSpin((2, 0)),
                DiatomCoupledBasis::NuclearSpin((1, 1)),
                DiatomCoupledBasis::Vibrational(-1)
            ])
        );

        assert_eq!(
            basis_elements[8],
            BasisElement(vec![
                DiatomCoupledBasis::ElectronSpin((2, -2)),
                DiatomCoupledBasis::NuclearSpin((1, -1)),
                DiatomCoupledBasis::Vibrational(-2)
            ])
        );
    }
}
