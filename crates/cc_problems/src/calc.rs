use serde::{
    Serialize, de::DeserializeOwned
};
use serde_json::Value;

use crate::system::System;
use anyhow::Result;

pub trait CalcInput: DeserializeOwned {
    fn from_value(value: &Value) -> Self {
        serde_json::from_value(value.clone())
            .expect("Could not parse calc input")
    }
}

pub trait Calc {
    fn calculate(&self, system: &System, input: &Value, worker: usize, workers: usize) -> Result<()>;
}

pub trait SingleCalc<I: DeserializeOwned, D: Serialize> {
    fn calculate(&self, input: &I, system: &System) -> Result<D>;
}

impl CalcInput for () {}

pub struct EnergyLevelsCalc;

#[derive(Clone, Debug, Serialize)]
pub struct EnergyLevelsData(pub Vec<f64>);

impl SingleCalc<(), EnergyLevelsData> for EnergyLevelsCalc {
    fn calculate(&self, _input: &(), system: &System) -> Result<EnergyLevelsData> {
        Ok(EnergyLevelsData(system.angular_blocks().diagonalized().0.asymptote))
    }
}
