pub mod bound_states;
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
        C::name(self)
    }

    fn input_schema(&self) -> Value {
        C::input_schema(self)
    }

    fn run(&self, input: TypedProblemInput<P::BasisRecipe, P::Params, Value>, problem: &P) -> Result<()> {
        let input = TypedProblemInput {
            basis_recipe: input.basis_recipe,
            parameters: input.parameters,
            calc_parameters: serde_json::from_value(input.calc_parameters)?,

            worker: input.worker,
            workers: input.workers,
        };

        C::run(self, input, problem)
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
        modified: Modified<Self::P, Self::CalcInput>,
        problem: &Self::P,
    ) -> impl IntoIterator<Item = Result<Self::Data>>;
}

pub struct Modified<'a, P: Problem, C> {
    pub system: &'a mut System,
    pub basis: &'a mut P::BasisRecipe,
    pub params: &'a mut P::Params,
    pub calc_input: &'a mut C
}