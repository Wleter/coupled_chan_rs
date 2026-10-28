pub mod diatom_in_b_field;
pub mod rotor_atom_in_b_field;
pub mod single_chan;

use std::collections::HashMap;

use anyhow::Result;
use cc_calculations::problems::{
    DynProblem,
    Problem,
    ProblemInput,
};
use serde_json::Value;

use crate::problems::diatom_in_b_field::DiatomInBFieldProblem;
use crate::problems::rotor_atom_in_b_field::RotorAtomInBFieldProblem;
use crate::problems::single_chan::SingleChanProblem;

pub fn available_problems() -> AvailableProblems {
    let mut problems = AvailableProblems::default();
    problems.insert(DiatomInBFieldProblem::new());
    problems.insert(RotorAtomInBFieldProblem::new());
    problems.insert(SingleChanProblem::new());

    problems
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
            .and_then(|s| s.run(problem_input))
    }

    pub fn problems(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(|x| x.as_ref())
    }

    pub fn scheme_for(&self, problem: &str) -> Result<Value> {
        self.0
            .get(problem)
            .ok_or(anyhow::anyhow!("problem {problem} not found in {:?}", self.0.keys()))
            .and_then(|s| Ok(s.schema().to_value()))
    }
}
