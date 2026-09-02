pub mod bounds;
pub mod dependence;
pub mod levels;
pub mod scattering;

use anyhow::Result;
use serde::{
    Serialize,
    de::DeserializeOwned,
};
use serde_json::Value;

use crate::{
    problems::{
        Problem,
        TypedProblemInput,
    },
    system::System,
};

pub trait DynCalc<P: Problem>: Send + Sync {
    fn name(&self) -> String;
    fn input_schema(&self) -> Value;
    fn run(&self, input: TypedProblemInput<P::BasisRecipe, P::Params, Value>, problem: &P) -> Result<()>;
}

impl<C: Calc<P = P>, P: Problem> DynCalc<P> for C {
    fn name(&self) -> String {
        C::name(&self)
    }

    fn input_schema(&self) -> Value {
        C::input_schema(&self)
    }

    fn run(&self, input: TypedProblemInput<P::BasisRecipe, P::Params, Value>, problem: &P) -> Result<()> {
        let input = TypedProblemInput {
            basis_recipe: input.basis_recipe,
            parameters: input.parameters,
            calculation_parameters: serde_json::from_value(input.calculation_parameters)?,

            worker: input.worker,
            workers: input.workers,
        };

        C::run(&self, input, problem)
    }
}

pub trait Calc: Send + Sync {
    type P: Problem;
    type CalcInput: DeserializeOwned + Clone + Send + Sync;

    fn name(&self) -> String;
    fn input_schema(&self) -> Value;
    fn run(
        &self,
        input: TypedProblemInput<<Self::P as Problem>::BasisRecipe, <Self::P as Problem>::Params, Self::CalcInput>,
        problem: &Self::P,
    ) -> Result<()>;
}

pub trait SingleCalc: Send + Sync {
    type P: Problem;
    type CalcInput: DeserializeOwned + Clone + Send + Sync;
    type Data: Serialize + Send + Sync + 'static;

    fn calculate(
        &self,
        system: &mut System,
        basis_recipe: &mut <Self::P as Problem>::BasisRecipe,
        parameters: &mut <Self::P as Problem>::Params,
        calc_input: &mut Self::CalcInput,
        problem: &Self::P,
    ) -> Result<Self::Data>;
}
