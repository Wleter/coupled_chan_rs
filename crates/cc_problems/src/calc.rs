use serde::{
    Serialize,
    de::DeserializeOwned,
};
use serde_json::Value;

use crate::system::System;
use anyhow::Result;

pub trait CalcInput: DeserializeOwned {
    fn from_value(value: &Value) -> Self {
        serde_json::from_value(value.clone()).expect("Could not parse calc input")
    }
}

pub trait Calc {
    fn calculate(&self, system: &System, input: &Value, worker: usize, workers: usize) -> Result<()>;
}

pub trait SingleCalc: Send + Sync {
    type Input: CalcInput + Send + Sync;
    type Data: Serialize + Send + Sync + 'static;

    fn calculate(&self, input: &Self::Input, system: &System) -> Result<Self::Data>;
}

impl CalcInput for () {}

pub struct EnergyLevelsCalc;

#[derive(Clone, Debug, Serialize)]
pub struct EnergyLevelsData(pub Vec<f64>);

impl SingleCalc for EnergyLevelsCalc {
    type Input = ();
    type Data = EnergyLevelsData;

    fn calculate(&self, _input: &Self::Input, system: &System) -> Result<Self::Data> {
        Ok(EnergyLevelsData(system.angular_blocks().diagonalized().0.asymptote))
    }
}
