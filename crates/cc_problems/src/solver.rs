use std::marker::PhantomData;

use coupled_chan::cc_propagator::step_strategy::Step;
use serde::{Deserialize, Serialize};

use crate::hamiltonian::Hamiltonian;

pub struct SolverScheme<Data, Recipe> 
where 
    Data: Serialize + Deserialize<'static>,
    Recipe: Fn(Data) -> Hamiltonian,
{
    hamiltonian_recipe: Recipe,
    phantom: PhantomData<Data>,
}

pub trait Calculation<Res: Serialize + Deserialize<'static>> {
    fn calculate(&self, hamiltonian: &Hamiltonian) -> Res;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CoupledChanSolver {
    RatioNumerov,
    JohnsonLogDeriv,
    ManolopoulosLogDeriv,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScatteringCalc<S: Step> {
    pub r_start: f64,
    pub r_stop: f64,
    pub step: S,
    pub solver: CoupledChanSolver
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundStateCalc<S: Step> {
    pub r_start: f64,
    pub r_stop: f64,
    pub r_match: f64,
    pub step: S,
    pub solver: CoupledChanSolver
}