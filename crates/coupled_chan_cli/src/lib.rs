use clap::Parser;
use serde_json::Value;
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{
        Path,
        PathBuf,
    },
};

use cc_problems::{
    calc::Calc,
    parameters::Parameters,
    system::{
        HamiltonianSpec,
        System,
    },
};
use json_comments::StripComments;
use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};

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
pub struct ProgramInput<B, P: Parameters> {
    pub basis: B,
    pub parameters: P,
    pub calculation_name: Box<str>,
    pub calculation: Value,
}

impl<B: DeserializeOwned, P: Parameters + DeserializeOwned> ProgramInput<B, P> {
    pub fn parse(path: impl AsRef<Path>) -> Self {
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
            panic!("Expected path to contain file extension .json/.jsonc/.toml")
        }
    }
}

pub struct CalcSpec<B, P>(pub Box<dyn Fn(&B, &P) -> Box<dyn Calc>>);

impl<B, P> CalcSpec<B, P> {
    pub fn new(f: impl Fn(&B, &P) -> Box<dyn Calc> + 'static) -> Self {
        Self(Box::new(f))
    }
}

pub struct ProgramExecutor<B, P, H>
where
    P: Parameters,
    H: Fn(&B, &P) -> HamiltonianSpec,
{
    hamiltonian_builder: Option<H>,
    calculation_specs: HashMap<Box<str>, CalcSpec<B, P>>,
    phantom: PhantomData<(B, P)>,
}

impl<B, P, H> Default for ProgramExecutor<B, P, H>
where
    P: Parameters,
    H: Fn(&B, &P) -> HamiltonianSpec,
{
    fn default() -> Self {
        Self {
            hamiltonian_builder: Default::default(),
            calculation_specs: Default::default(),
            phantom: Default::default(),
        }
    }
}

impl<B, P, H> ProgramExecutor<B, P, H>
where
    B: DeserializeOwned,
    P: Parameters + DeserializeOwned,
    H: Fn(&B, &P) -> HamiltonianSpec,
{
    pub fn set_hamiltonian_builder(mut self, f: H) -> Self {
        self.hamiltonian_builder = Some(f);
        self
    }

    pub fn set_calculation_specs(mut self, specs: HashMap<Box<str>, CalcSpec<B, P>>) -> Self {
        self.calculation_specs = specs;
        self
    }

    pub fn add_calculation_spec(mut self, calc_name: impl AsRef<str>, calc: CalcSpec<B, P>) -> Self {
        self.calculation_specs.insert(calc_name.as_ref().into(), calc);
        self
    }

    pub fn build(self) {
        let args = Args::parse();
        let input: ProgramInput<B, P> = ProgramInput::parse(&args.input);

        let hamiltonian_builder = self
            .hamiltonian_builder
            .expect("Did not provide hamiltonian builder for the program");

        let calculation_spec = self
            .calculation_specs
            .get(&input.calculation_name)
            .expect("Did not find calculation with given name");

        let system = System::new(
            hamiltonian_builder(&input.basis, &input.parameters),
            input.parameters.registry(),
        );

        calculation_spec.0(&input.basis, &input.parameters)
            .calculate(&system, &input.calculation, args.worker, args.workers)
            .expect("Calculation encountered error");
    }
}
