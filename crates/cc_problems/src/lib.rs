pub mod atom_atom_problem;
pub use coupled_chan;

use coupled_chan::coupling::{RedCoupling, VanishingCoupling};
use hilbert_space::{dyn_space::BasisElements, faer::Mat, operator::Operator};
use spin_algebra::Spin;

pub struct HamiltonianTerm {
    pub name: String,
    pub hamiltonian: Box<dyn Fn(BasisElements) -> Operator<Mat<f64>>>
}

pub trait CoupledProblemBuilder {
    type Dependence;

    fn build(self, dependence: Self::Dependence) -> CoupledProblem<impl VanishingCoupling>;
}

pub struct CoupledProblem<V: VanishingCoupling> {
    pub red_coupling: RedCoupling<V>,
}

impl<V: VanishingCoupling> CoupledProblem<V> {

}

// pub struct FieldDependentProblem<F> {
    
// }


#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct AtomS(Spin);

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct AtomI(Spin);
