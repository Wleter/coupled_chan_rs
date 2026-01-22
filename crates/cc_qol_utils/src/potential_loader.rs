use serde::Deserialize;

#[derive(Deserialize)]
pub struct PotentialLoader<P> {
    pub points: Vec<P>,
    pub values: Vec<f64>,
}
