use std::marker::PhantomData;

use serde::Serialize;

use crate::{
    calculations::{
        Modified,
        SingleCalc,
        dependence::DependenceCalc,
    },
    problems::Problem,
};

pub struct EnergyLevelsCalc<P>(PhantomData<P>);

impl<P> Default for EnergyLevelsCalc<P> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EnergyLevelsData(pub Vec<f64>);

impl<P: Problem> SingleCalc for EnergyLevelsCalc<P> {
    type P = P;
    type CalcInput = ();
    type Data = EnergyLevelsData;

    fn calculate(&self, modified: Modified<P, ()>, _problem: &P) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        [Ok(EnergyLevelsData(
            modified.system.angular_blocks().diagonalized().0.asymptote,
        ))]
    }
}

pub type EnergyLevelsScan<P, D> = DependenceCalc<EnergyLevelsCalc<P>, D>;
