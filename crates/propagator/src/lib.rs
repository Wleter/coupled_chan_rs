pub mod propagator_watcher;

pub trait Repr {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ratio<T>(pub T);
impl<T> Repr for Ratio<T> {}

#[derive(Clone, Copy, Debug, Default)]
pub struct LogDeriv<T>(pub T);
impl<T> Repr for LogDeriv<T> {}

#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Inwards,
    Outwards,
}

#[derive(Clone, Debug)]
pub struct Boundary<T> {
    pub r_start: f64,
    pub direction: Direction,
    pub value: T,
    pub derivative: T,
}

#[derive(Clone, Debug)]
pub struct Solution<R> {
    pub r: f64,
    pub dr: f64,
    pub sol: R,
}

pub trait Propagator<R: Repr> {
    fn step(&mut self) -> &Solution<R>;
    fn propagate_to(&mut self, r: f64) -> &Solution<R>;
}

#[derive(Clone, Copy, Debug)]
pub struct Nodes(pub u64);

pub trait NodeCountPropagator<R: Repr>: Propagator<R> {
    fn nodes(&self) -> Nodes;
}
