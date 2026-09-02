use cc_problems::problems::available_problems;
use clap::Parser;
use coupled_chan_cli::Args;

fn main() -> anyhow::Result<()> {
    Args::parse().run_problems(available_problems())
}
