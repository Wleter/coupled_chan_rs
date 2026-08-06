pub mod input;

use std::path::PathBuf;

use clap::Parser;

use crate::input::read_input;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CLI for coupled_chan package",
    long_about = "CLI for coupled_chan package solving coupled channel equation given input and problem source"
)]
struct Args {
    /// input file path
    #[arg(short, long)]
    input: PathBuf,

    /// plugin library file path
    #[arg(short, long)]
    plugin: Option<PathBuf>,

    /// total number of workers to divide in task
    #[arg(short('W'), long, default_value_t = 1)]
    workers: u32,

    /// worker number for task assignment
    #[arg(short('w'), long, default_value_t = 0)]
    worker: u32,
}

fn main() {
    let args = Args::parse();

    let input = read_input(&args.input);
    println!("{input:?}");
}
