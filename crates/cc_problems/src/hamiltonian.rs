use std::sync::Arc;

use hilbert_space::{faer::Mat, operator_diag_mel, operator_mel, space::{BasisElementsRef, BasisId}};
use spin_algebra::{SpinLike, SpinMagLike, SpinPair, ops};

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;

#[derive(Clone)]
pub struct HamiltonianTerm {
    mel: Arc<dyn Fn(BasisElementsRef) -> Operator>
}

impl HamiltonianTerm {
    pub fn new<F: Fn(BasisElementsRef) -> Operator + 'static>(mel: F) -> Self {
        Self {
            mel: Arc::new(mel)
        }
    }

    pub fn mel(&self, elements: BasisElementsRef) -> Operator {
        (self.mel)(elements)
    }
}

pub fn projection_term<S: SpinLike>(s_id: BasisId<S>) -> HamiltonianTerm {
    HamiltonianTerm::new(move |e| {
        operator_diag_mel!(e, [s_id], |[s]| {
            -s.m().value()
        })
    })
}

pub fn dot_term_uncoupled<S1, S2>(s1_id: BasisId<S1>, s2_id: BasisId<S2>) -> HamiltonianTerm 
where 
    S1: SpinLike, 
    S2: SpinLike 
{
    HamiltonianTerm::new(move |e| {
        operator_mel!(e, [s1_id, s2_id], |[s1, s2]| {
            ops::dot(s1, s2)
        })
    })
}

pub fn dot_term_coupled<S, I>(f_id: BasisId<SpinPair<S, I>>) -> HamiltonianTerm 
where 
    S: SpinMagLike,
    I: SpinMagLike,
{
    HamiltonianTerm::new(move |e| {
        operator_diag_mel!(e, [f_id], |[f]| {
            (f.spin.squared() - f.pair.0.squared() - f.pair.1.squared()) / 2.
        })
    })
}
