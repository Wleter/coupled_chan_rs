pub mod adiabats;
pub mod bound_states;
pub mod dependence;
pub mod hamiltonian_terms;
pub mod levels;
pub mod modifications;
pub mod resonances;
pub mod scattering;

use anyhow::{Context, Result};
use cc_system::System;
use schemars::{
    Schema,
    SchemaGenerator,
};
use serde::Serialize;
use serde_json::Value;

use crate::InputData;
use crate::problems::{
    Problem,
    TypedProblemInput,
};

pub trait DynCalc<P: Problem>: Send + Sync {
    fn name(&self) -> String;
    fn subschema(&self, generator: &mut SchemaGenerator) -> Schema;
    fn schema(&self) -> Schema;
    fn run(&self, input: TypedProblemInput<P::BasisRecipe, P::Params, Value>, problem: &P) -> Result<()>;
}

impl<C: Calc<P = P>, P: Problem> DynCalc<P> for C {
    fn name(&self) -> String {
        C::name(self)
    }

    fn subschema(&self, generator: &mut SchemaGenerator) -> Schema {
        C::subschema(&self, generator)
    }

    fn schema(&self) -> Schema {
        C::schema(self)
    }

    fn run(&self, input: TypedProblemInput<P::BasisRecipe, P::Params, Value>, problem: &P) -> Result<()> {
        let input = TypedProblemInput {
            basis_recipe: input.basis_recipe,
            parameters: input.parameters,
            calc_parameters: serde_json::from_value(input.calc_parameters)
                .context("calculation_parameters validation error")?,

            worker: input.worker,
            workers: input.workers,
        };

        C::run(self, input, problem)
    }
}

pub trait Calc: Send + Sync {
    type P: Problem;
    type CalcInput: InputData;

    fn name(&self) -> String;

    fn subschema(&self, generator: &mut SchemaGenerator) -> Schema {
        generator.subschema_for::<Self::CalcInput>()
    }

    fn schema(&self) -> Schema {
        schemars::schema_for!(Self::CalcInput)
    }

    fn run(
        &self,
        input: TypedProblemInput<<Self::P as Problem>::BasisRecipe, <Self::P as Problem>::Params, Self::CalcInput>,
        problem: &Self::P,
    ) -> Result<()>;
}

pub trait SingleCalc: Send + Sync {
    type P: Problem;
    type CalcInput: InputData;
    type Data: Serialize + Send + Sync + 'static;

    fn input_schema() -> serde_json::Value {
        schemars::schema_for!(Self::CalcInput).to_value()
    }

    fn calculate(
        &self,
        modified: &mut Modified<Self::P, Self::CalcInput>,
        problem: &Self::P,
    ) -> impl IntoIterator<Item = Result<Self::Data>>;
}

pub struct Modified<'a, P: Problem, C> {
    pub system: &'a mut System,
    pub basis: &'a mut P::BasisRecipe,
    pub params: &'a mut P::Params,
    pub calc_input: &'a mut C,
}
