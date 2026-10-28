use cc_problems::problems::available_problems;
use clap::Parser;
use coupled_chan_cli::Cli;

fn main() -> anyhow::Result<()> {
    Cli::parse().execute(available_problems())
}
