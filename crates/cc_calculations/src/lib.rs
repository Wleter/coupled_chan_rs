use schemars::JsonSchema;
use serde::de::DeserializeOwned;

pub mod calculations;
pub mod potential_scalings;
pub mod problems;

pub trait InputData: DeserializeOwned + Clone + JsonSchema + Send + Sync {}
impl<T: DeserializeOwned + Clone + JsonSchema + Send + Sync> InputData for T {}
