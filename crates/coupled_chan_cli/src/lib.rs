pub mod problem;
pub mod scheme;

use anyhow::Result;
use cc_problems::problems::AvailableProblems;
use clap::Parser;

use crate::problem::ProblemArgs;
use crate::scheme::SchemeArgs;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CLI for coupled_chan package",
    long_about = "CLI for coupled_chan package solving coupled channel equation given input and problem source"
)]
pub struct CoupledChanCli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    Problem(ProblemArgs),
    AvailableProblems,
    Scheme(SchemeArgs),
}

impl CoupledChanCli {
    pub fn parse() -> Self {
        <Self as Parser>::parse()
    }

    pub fn execute(self, problems: AvailableProblems) -> Result<()> {
        match self.command {
            Commands::Problem(problem_args) => problem_args.run_problems(problems),
            Commands::AvailableProblems => {
                println!("available problems are: {:?}", problems.problems().collect::<Vec<&str>>());

                Ok(())
            }
            Commands::Scheme(scheme_args) => scheme_args.generate_scheme(problems),
        }
    }
}
