use serde::{
    Serialize,
    de::DeserializeOwned,
};

use crate::system::System;
use anyhow::Result;

pub trait Calc<I> {
    fn calculate(&self, system: &System, input: &I, worker: usize, workers: usize) -> Result<()>;
}

pub trait SingleCalc<I: DeserializeOwned, D: Serialize> {
    fn calculate(&self, input: &I, system: &System) -> Result<D>;
}
