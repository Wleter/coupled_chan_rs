pub mod diatom_in_b_field;

use std::collections::HashMap;

use anyhow::Result;
use serde::{
    Deserialize,
    de::DeserializeOwned,
};
use serde_json::Value;

use crate::{
    calculations::DynCalc, parameters::Parameters, problems::diatom_in_b_field::DiatomInBFieldProblem, system::System
};

pub fn available_problems() -> AvailableProblems {
    let mut problems = AvailableProblems::default();
    problems.insert(DiatomInBFieldProblem::new());

    problems
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProblemInput {
    pub problem_name: Box<str>,
    pub basis_recipe: Value,
    pub parameters: Value,
    pub calculation: Box<str>,
    pub calculation_parameters: Value,

    pub worker: usize,
    pub workers: usize,
}

#[derive(Clone)]
pub struct TypedProblemInput<B: DeserializeOwned, P: DeserializeOwned, I: DeserializeOwned> {
    pub(crate) basis_recipe: B,
    pub(crate) parameters: P,
    pub(crate) calculation_parameters: I,

    pub(crate) worker: usize,
    pub(crate) workers: usize,
}

#[derive(Default)]
pub struct AvailableProblems(HashMap<Box<str>, Box<dyn DynProblem>>);

impl AvailableProblems {
    pub fn insert(&mut self, problem: impl Problem + 'static) -> &mut Self {
        self.0.insert(problem.name().into(), Box::new(problem));

        self
    }

    pub fn run(&self, problem_input: ProblemInput) -> Result<()> {
        self.0
            .get(&problem_input.problem_name)
            .ok_or(anyhow::anyhow!(
                "problem {} not found in {:?}",
                problem_input.problem_name,
                self.0.keys()
            ))
            .map(|s| s.run(problem_input))
            .flatten()
    }
}

pub trait DynProblem {
    fn name(&self) -> &str;
    fn schema(&self) -> Value;
    fn run(&self, problem_input: ProblemInput) -> Result<()>;
}

impl<P: Problem> DynProblem for P {
    fn name(&self) -> &str {
        P::NAME
    }

    fn schema(&self) -> Value {
        P::schema(&self)
    }

    fn run(&self, problem_input: ProblemInput) -> Result<()> {
        let map = &self.calculations();

        let input = TypedProblemInput {
            basis_recipe: serde_json::from_value(problem_input.basis_recipe)?,
            parameters: serde_json::from_value(problem_input.parameters)?,
            calculation_parameters: problem_input.calculation_parameters,
            worker: problem_input.worker,
            workers: problem_input.workers,
        };

        map.get(&problem_input.calculation)
            .ok_or(anyhow::anyhow!("Did not find {} in {:?}", problem_input.calculation, map.keys()))?
            .run(input, &self)
    }
}

pub trait Problem: Sync + Send {
    type BasisRecipe: DeserializeOwned + Clone + Send + Sync;
    type Params: Parameters + DeserializeOwned + Clone + Send + Sync;

    const NAME: &str;
    const DESCRIPTION: &str;
    fn schema(&self) -> Value;

    fn build(&self, basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> System;
    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>>;
}
