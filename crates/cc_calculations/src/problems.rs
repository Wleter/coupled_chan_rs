use std::collections::HashMap;

use anyhow::Result;
use hilbert_space::faer::MatRef;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::calculations::DynCalc;
use cc_system::hamiltonian_spec::HamiltonianSpec;
use cc_system::parameters::Parameters;

#[derive(Deserialize)]
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
    pub(crate) calc_parameters: I,

    pub(crate) worker: usize,
    pub(crate) workers: usize,
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
        P::schema(self)
    }

    fn run(&self, problem_input: ProblemInput) -> Result<()> {
        let map = &self.calculations();

        let input = TypedProblemInput {
            basis_recipe: serde_json::from_value(problem_input.basis_recipe)?,
            parameters: serde_json::from_value(problem_input.parameters)?,
            calc_parameters: problem_input.calculation_parameters,
            worker: problem_input.worker,
            workers: problem_input.workers,
        };

        map.get(&problem_input.calculation)
            .ok_or(anyhow::anyhow!(
                "Did not find {} in {:?}",
                problem_input.calculation,
                map.keys()
            ))?
            .run(input, self)
    }
}

pub trait Problem: Sync + Send {
    type BasisRecipe: DeserializeOwned + Clone + Send + Sync;
    type Params: Parameters + DeserializeOwned + Clone + Send + Sync;

    const NAME: &str;
    const DESCRIPTION: &str;
    fn schema(&self) -> Value;

    fn build(basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> HamiltonianSpec;
    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>>;
}

pub(crate) fn mat_as_nested_vec<T: Copy>(mat: MatRef<T>) -> Vec<Vec<T>> {
    let mut vec = vec![vec![]; mat.nrows()];
    for (i, row) in mat.row_iter().enumerate() {
        vec[i] = row.iter().copied().collect();
    }
    vec
}
