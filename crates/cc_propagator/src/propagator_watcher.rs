use crate::{Repr, Solution};

pub trait PropagatorWatcher<R: Repr> {
    fn init(&mut self, _sol: &Solution<R>) {}
    fn before_step(&mut self, _sol: &Solution<R>) {}
    fn after_step(&mut self, _sol: &Solution<R>) {}
    fn finalize(&mut self, _sol: &Solution<R>) {}
}
