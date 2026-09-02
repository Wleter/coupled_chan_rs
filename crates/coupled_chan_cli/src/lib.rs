use cc_problems::problems::{AvailableProblems, ProblemInput};
use clap::Parser;
use serde::Deserialize;
use serde_json::Value;
use std::{
    path::{
        Path,
        PathBuf,
    },
};
use json_comments::StripComments;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CLI for coupled_chan package",
    long_about = "CLI for coupled_chan package solving coupled channel equation given input and problem source"
)]
pub struct Args {
    /// input file path
    #[arg(short, long)]
    input_defaults: Option<PathBuf>,

    /// input file path
    #[arg(short, long)]
    input: PathBuf,

    /// plugin library file path
    #[arg(short, long)]
    plugin: Option<PathBuf>,

    /// total number of workers to divide in task
    #[arg(short('W'), long, default_value_t = 1)]
    workers: usize,

    /// worker number for task assignment
    #[arg(short('w'), long, default_value_t = 0)]
    worker: usize,
}

impl Args {
    pub fn run_problems(self, problems: AvailableProblems) -> Result<()> {
        let defaults = self.input_defaults.map(|path| parse_input(&path));
        let input = if let Some(defaults) = defaults {
            merge(defaults, parse_input(&self.input))
        } else {
            parse_input(&self.input)
        };

        problems.run(into_problem_input(input, self.worker, self.workers)?)
    }
}

pub fn parse_input(path: impl AsRef<Path>) -> Value {
    let path = path.as_ref();
    if let Some(ext) = path.extension() {
        let read = std::fs::read_to_string(path).unwrap();
        match ext.to_str().unwrap() {
            "toml" => toml::from_str(&read).unwrap(),
            "json" | "jsonc" => serde_json::from_reader(StripComments::new(read.as_bytes())).unwrap(),
            "json5" => json5::from_str(&read).unwrap(),
            _ => panic!("Unknown file extension type"),
        }
    } else {
        panic!("Expected path to contain file extension .json/.jsonc/.json5/.toml")
    }
}

fn merge(mut defaults: Value, overrides: Value) -> Value {
    match (&mut defaults, overrides) {
        (Value::Object(default), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match default.get_mut(&key) {
                    Some(default_value) => merge_inplace(default_value, value),
                    None => {
                        default.insert(key, value);
                    }
                }
            }
        }
        (default, overrides) => {
            *default = overrides;
        }
    }

    defaults
}

fn merge_inplace(default: &mut Value, overrides: Value) {
    match (default, overrides) {
        (Value::Object(default), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match default.get_mut(&key) {
                    Some(default_value) => merge_inplace(default_value, value),
                    None => {
                        default.insert(key, value);
                    }
                }
            }
        }

        (default, overrides) => {
            *default = overrides;
        }
    }
}

#[derive(Deserialize)]
struct ProgramInput {
    problem_name: Box<str>,
    basis_recipe: Value,
    parameters: Value,
    calculation: Box<str>,
    calculation_parameters: Value,
}

pub fn into_problem_input(parsed: Value, worker: usize, workers: usize) -> Result<ProblemInput> {
    let input: ProgramInput = serde_json::from_value(parsed)?;

    Ok(ProblemInput {
        problem_name: input.problem_name,
        basis_recipe: input.basis_recipe,
        parameters: input.parameters,
        calculation: input.calculation,
        calculation_parameters: input.calculation_parameters,
        worker,
        workers,
    })
}
