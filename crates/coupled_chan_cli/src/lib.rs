use std::{marker::PhantomData, path::{Path, PathBuf}};
use clap::Parser;

use cc_problems::{calc::Calc, parameters::Parameters, system::{HamiltonianSpec, System}};
use json_comments::StripComments;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub mod input;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "CLI for coupled_chan package",
    long_about = "CLI for coupled_chan package solving coupled channel equation given input and problem source"
)]
pub struct Args {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProgramInput<B, P: Parameters, I> {
    basis: B,
    parameters: P,
    calculation: I
}

impl<B: DeserializeOwned, P: Parameters + DeserializeOwned, C: DeserializeOwned> ProgramInput<B, P, C> {
    pub fn parse(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        if let Some(ext) = path.extension() {
            let read = std::fs::read_to_string(path).unwrap();
            match ext.to_str().unwrap() {
                "toml" => toml::from_str(&read).unwrap(),
                "json" | "jsonc" => serde_json::from_reader(StripComments::new(read.as_bytes())).unwrap(),
                _ => panic!("Unknown file extension type"),
            }
        } else {
            panic!("Expected path to contain file extension .json/.jsonc/.toml")
        }
    }
}

pub struct ProgramExecutor<B, P, CalcI, H, C> 
where 
    P: Parameters, 
    H: Fn(&B, &P) -> HamiltonianSpec, 
    C: Calc<CalcI> 
{
    hamiltonian_builder: Option<H>,
    calculation: Option<C>,
    phantom: PhantomData<(B, P, CalcI)>,
}

impl<B, P, CalcI, H, C> Default for ProgramExecutor<B, P, CalcI, H, C>
where 
    P: Parameters, 
    H: Fn(&B, &P) -> HamiltonianSpec, 
    C: Calc<CalcI>
{
    fn default() -> Self {
        Self { 
            hamiltonian_builder: Default::default(), 
            calculation: Default::default(), 
            phantom: Default::default() 
        }
    }
}

impl<B, P, I, H, C> ProgramExecutor<B, P, I, H, C>
where 
    B: DeserializeOwned,
    P: Parameters + DeserializeOwned, 
    I: DeserializeOwned,
    H: Fn(&B, &P) -> HamiltonianSpec, 
    C: Calc<I>
{
    pub fn set_hamiltonian_builder(mut self, f: H) -> Self {
        self.hamiltonian_builder = Some(f);
        self
    }

    pub fn set_calculation(mut self, calc: C) -> Self {
        self.calculation = Some(calc);
        self
    }

    pub fn build(self) {
        let args = Args::parse();
        let input: ProgramInput<B, P, I> = ProgramInput::parse(&args.input);

        let hamiltonian_builder = self.hamiltonian_builder
            .expect("Did not provide hamiltonian builder for the program");

        let system = System::new(hamiltonian_builder(&input.basis, &input.parameters), input.parameters.registry());

        self.calculation
            .expect("Did not provide calculation for the program")
            .calculate(&system, &input.calculation, args.worker, args.workers)
            .expect("Calculation encountered error");
    }
}
