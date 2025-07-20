use crate::interaction::Interaction;

pub struct FuncPotential<F: Fn(f64) -> f64> {
    func: F,
}

impl<F: Fn(f64) -> f64> FuncPotential<F> {
    /// The potential ought to be asymptotic to 0 value
    /// and without included centrifugal term
    pub fn new(f: F) -> Self {
        assert_eq!((f)(f64::INFINITY), 0., "Interaction should be vanishing");

        Self { func: f }
    }
}

impl<F: Fn(f64) -> f64> Interaction for FuncPotential<F> {
    fn value(&self, r: f64) -> f64 {
        (self.func)(r)
    }
}
