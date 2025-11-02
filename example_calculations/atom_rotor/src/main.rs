mod caf_rb_problem;
pub mod model_problem;
use crate::{caf_rb_problem::CaFRbProblem, model_problem::ModelRotorAtomProblem};

use cc_problems::qol_utils::{
    problem_selector::{ProblemSelector, get_args},
    problems_impl,
};

fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "Rotor + atom problems",
    "CaF + Rb problem" => |args| {
        CaFRbProblem::select(args);
        Ok(())
    },
    "model atom rotor problem" => |args| {
        ModelRotorAtomProblem::select(args);
        Ok(())
    },
);
