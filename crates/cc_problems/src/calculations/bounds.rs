use serde::{
    Deserialize,
    Serialize,
};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        Energy,
        Length,
    },
};

use crate::calculations::{dependence::Dependant, scattering::{Boundary, CoupledChanSolver, Step}};

#[derive(Clone, Debug, Deserialize)]
pub struct BoundStateCalcInput {
    #[serde(default)]
    pub entrance: usize,
    #[serde(default)]
    pub energy: Scalar<Energy>,

    pub dependant: Dependant,

    #[serde(default)]
    pub boundaries: (Boundary, Boundary),
    pub r_start: Scalar<Length>,
    pub r_match: Scalar<Length>,
    pub r_stop: Scalar<Length>,

    pub step: Step,
    pub solver: CoupledChanSolver,

    #[serde(default)]
    monotony: NodeMonotony,
    node_range: Option<NodeRangeTarget>,

    #[serde(default)]
    search_method: BoundSearchMethod,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum NodeMonotony {
    Decreasing,
    #[default]
    Increasing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundSearchMethod {
    Brent(u32),
    Bisection,
}

impl Default for BoundSearchMethod {
    fn default() -> Self {
        Self::Brent(30)
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum NodeRangeTarget {
    Range(u64, u64),
    BottomRange(u64),
    TopRange(u64),
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundStateData {
    pub nodes: u64,
    pub parameter: f64,

    pub occupations: Option<Vec<f64>>,
    pub wave_function: Option<WaveFunction>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WaveFunction {
    pub distances: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundStatesData(pub Vec<BoundStateData>);

pub struct BoundStateCalc {}
