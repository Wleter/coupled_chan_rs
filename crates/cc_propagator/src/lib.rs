pub mod propagator_watcher;
pub mod step_strategy;

#[cfg(feature = "multi_channel")]
pub mod multi_channel;
#[cfg(feature = "single_channel")]
pub mod single_channel;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nodes(pub u64);

pub trait WithNodeCount {
    fn nodes(&self) -> Nodes;
}

pub trait WithWaveStorage<T> {
    fn init_wave_storage(&mut self);
    fn get_wave_storage(&self) -> Option<&WaveStorage<T>>;
}

pub struct WaveStorage<T> {
    rs: Vec<f64>,
    connections: Vec<T>,
}

impl<T> Default for WaveStorage<T> {
    fn default() -> Self {
        Self {
            rs: vec![],
            connections: vec![],
        }
    }
}

impl<T: Clone> WaveStorage<T> {
    pub fn push(&mut self, r: f64, connection: &T) {
        self.rs.push(r);
        self.connections.push(connection.clone());
    }
}
