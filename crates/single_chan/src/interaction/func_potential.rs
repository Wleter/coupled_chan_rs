use crate::interaction::{
    AsymptoteDep,
    Interaction,
};

pub struct FuncPotential<F: Fn(f64) -> f64> {
    func: F,
    dep: AsymptoteDep,
}

impl<F: Fn(f64) -> f64> FuncPotential<F> {
    pub fn new(f: F, dep: AsymptoteDep) -> Self {
        Self { func: f, dep }
    }

    pub fn new_unknown(f: F) -> Self {
        Self {
            func: f,
            dep: AsymptoteDep::Unknown,
        }
    }
}

impl<F: Fn(f64) -> f64> Interaction for FuncPotential<F> {
    fn value(&self, r: f64) -> f64 {
        (self.func)(r)
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        self.dep
    }
}
