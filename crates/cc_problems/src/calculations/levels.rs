use std::marker::PhantomData;

use serde::Serialize;

use crate::{
    calculations::{
        SingleCalc,
        dependence::DependenceCalc,
    },
    problems::Problem,
    system::System,
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

    fn calculate(
        &self,
        system: &mut System,
        _basis_recipe: &mut P::BasisRecipe,
        _parameters: &mut P::Params,
        _calc_input: &mut Self::CalcInput,
        _problem: &P,
    ) -> anyhow::Result<Self::Data> {
        Ok(EnergyLevelsData(system.angular_blocks().diagonalized().0.asymptote))
    }
}

pub type EnergyLevelsScan<P> = DependenceCalc<EnergyLevelsCalc<P>>;
