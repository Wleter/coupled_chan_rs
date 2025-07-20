use crate::interaction::Interaction;

pub struct FuncPotential<F: Fn(f64) -> f64> {
    func: F,
}

impl<F: Fn(f64) -> f64> FuncPotential<F> {
    pub fn new(f: F) -> Self {
        Self { func: f }
    }
}

impl<F: Fn(f64) -> f64> Interaction for FuncPotential<F> {
    fn value(&self, r: f64) -> f64 {
        (self.func)(r)
    }

    fn asymptote(&self) -> f64 {
        (self.func)(f64::INFINITY)
    }
}
