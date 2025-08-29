pub mod atom_atom_problem;
pub mod atom_structure;
pub mod rotor_structure;
pub mod system_structure;
pub use coupled_chan;

use coupled_chan::{
    Operator,
    coupling::{RedCoupling, VanishingCoupling},
};
use hilbert_space::{
    cast_variant,
    dyn_space::{BasisElementIndices, BasisElements, BasisElementsRef},
};

use crate::system_structure::SystemStructure;

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AngularMomentum(pub u32);

pub struct AngularBasisElements {
    pub full_basis: BasisElements,
    ls: Vec<AngularMomentum>,
    separated_basis_indices: Vec<Vec<BasisElementIndices>>,
}

impl AngularBasisElements {
    pub fn new(full_basis: BasisElements, system: &SystemStructure) -> Self {
        let basis = full_basis.basis;

        let mut angular_indices: Vec<(AngularMomentum, BasisElementIndices)> = full_basis
            .elements_indices
            .into_iter()
            .map(|indices| (*cast_variant!(dyn indices.index(system.l, &basis), AngularMomentum), indices))
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

    pub fn angular_iter<'a>(&'a self) -> impl Iterator<Item = (AngularMomentum, BasisElementsRef<'a>)> {
        self.ls.iter().zip(&self.separated_basis_indices).map(|(l, indices)| {
            (
                *l,
                BasisElementsRef {
                    basis: &self.full_basis.basis,
                    elements_indices: indices,
                },
            )
        })
    }
}

pub struct HamiltonianTerm {
    pub name: String,
    pub scaling: f64,
    hamiltonian: Operator,
}

impl HamiltonianTerm {
    pub fn new(name: &str, hamiltonian: Operator) -> Self {
        Self {
            name: name.to_string(),
            scaling: 1.,
            hamiltonian,
        }
    }

    pub fn as_operator(&self) -> Operator {
        Operator::new(self.scaling * &self.hamiltonian.0)
    }
}

pub trait CoupledProblemBuilder {
    type Dependence;

    fn build(self, dependence: Self::Dependence) -> CoupledProblem<impl VanishingCoupling>;
}

pub struct CoupledProblem<V: VanishingCoupling> {
    pub red_coupling: RedCoupling<V>,
}

impl<V: VanishingCoupling> CoupledProblem<V> {}

// pub struct FieldDependentProblem<F> {

// }
