mod caf_rb_problem;
use crate::caf_rb_problem::CaFRbProblem;
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
);
