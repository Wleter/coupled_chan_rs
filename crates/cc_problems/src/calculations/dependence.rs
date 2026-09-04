use std::{
    marker::PhantomData, 
    path::PathBuf
};

use anyhow::Result;
use cc_math_utils::{linspace, logspace};
use cc_qol_utils::{
    saving::{
        DataSaver,
        FileAccess,
        JsonFormat,
    },
};
use rayon::prelude::*;
use serde::{
    Deserialize,
    Serialize, de::DeserializeOwned,
};
use serde_json::{Number, Value};
use unit_systems::quantities::{PhysQuantity, Scalar, UnitsConverter};

use crate::{
    calculations::{
        Calc, Modified, SingleCalc
    },
    parameters::Parameters,
    problems::{
        Problem,
        TypedProblemInput,
    }, system::System,
};

pub trait ModifyParams: Send + Sync + DeserializeOwned {
    type P: Problem;
    type C;

    fn modify(&self, modified: &mut Modified<Self::P, Self::C>);
    fn as_number(&self) -> Number;
    fn mut_number(&mut self, number: Number);
}

/// converts value in target unit system to some scalar with unit.
/// hacky way to get back physical quantities todo!
pub fn scalar_from_value<Q: PhysQuantity>(converter: &UnitsConverter, value: f64) -> Scalar<Q> {
    if value == 0.0 {
        return Scalar::new(0.0, Q::default(), "");
    }

    // first unit encountered 
    let unit = &converter.registry.get::<Q>()[0];
    let value_in_unit = value / Q::to_unit_system_logic(unit.name, &converter.registry, &converter.target_unit_system);

    Scalar::new(value_in_unit, Q::default(), unit.name)
}

pub struct DependenceCalc<C: SingleCalc, D: ModifyParams<P = C::P, C = C::CalcInput>> {
    single_calc: C,
    phantom: PhantomData<D>
}

impl<C: SingleCalc, D: ModifyParams<P = C::P, C = C::CalcInput>> DependenceCalc<C, D> {
    pub fn new(single_calc: C) -> Self {
        Self {
            single_calc,
            phantom: PhantomData
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(bound = "D: DeserializeOwned, C: DeserializeOwned")]
#[serde(deny_unknown_fields)]
pub struct DependenceCalcInput<C, D> {
    pub save_filepath: PathBuf,
    #[serde(default = "default_file_access")]
    pub save_option: FileAccess,

    #[serde(flatten)]
    pub calc: C,
    #[serde(default)]
    pub grid: Option<Grid<D>>,

    #[serde(default)]
    pub parallelism: Parallelism,
}

#[derive(Clone, Debug, Serialize)]
pub struct DependenceData<P, D> {
    pub parameter: P,
    pub data: D,
}

fn default_file_access() -> FileAccess {
    FileAccess::Append
}

impl<C, D> Calc for DependenceCalc<C, D>
where
    C: SingleCalc,
    D: ModifyParams<P = C::P, C = C::CalcInput> + Clone + std::fmt::Debug
{
    type P = C::P;
    type CalcInput = DependenceCalcInput<C::CalcInput, D>;

    fn name(&self) -> String {
        format!("Dependence calc for problem {}", Self::P::NAME)
    }

    fn input_schema(&self) -> Value {
        todo!()
    }

    fn run(
        &self,
        input: TypedProblemInput<<Self::P as Problem>::BasisRecipe, <Self::P as Problem>::Params, Self::CalcInput>,
        problem: &Self::P,
    ) -> Result<()> {
        let hamiltonian_spec = Self::P::build(&input.basis_recipe, &input.parameters);
        let system = System::new(hamiltonian_spec, input.parameters.registry());
        let save_filepath = &input.calc_parameters.save_filepath.to_string_lossy();
        let save_option = input.calc_parameters.save_option;
        let parallel_no = input.calc_parameters.parallelism;

        if let Some(grid) = &input.calc_parameters.grid {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;
            let data = grid.collect();

            let data = if input.workers > 1 {
                data.into_iter().skip(input.worker).step_by(input.workers).collect()
            } else {
                data
            };

            ParallelExecutor::new((system, input))
                .with_parallelism(parallel_no)
                .par_execute(data, |(s, input), d| {
                    let mut modified = Modified {
                        system: s,
                        basis: &mut input.basis_recipe,
                        params: &mut input.parameters,
                        calc_input: &mut input.calc_parameters.calc,
                    };
                    d.modify(&mut modified);

                    for data in self.single_calc.calculate(modified, problem) {
                        let data = data?;
                        saver.send(DependenceData { parameter: d.as_number(), data });
                    }

                    Ok(())
                })?;
        } else {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;

            let mut input = input;
            let mut system = system;
            let calc = self.single_calc.calculate(
                Modified {
                    system: &mut system,
                    basis: &mut input.basis_recipe,
                    params: &mut input.parameters,
                    calc_input: &mut input.calc_parameters.calc,
                },
                problem,
            );

            for data in calc {
                let data = data?;
                saver.send(data);
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Grid<D> {
    Linear {
        start: D,
        end: D,
        n: usize,
    },
    Log {
        start: D,
        end: D,
        n: usize,
    },
    Vec {
        values: Vec<D>,
    },
    Composite {
        ranges: Vec<Grid<D>>,
    },
}

impl<D: ModifyParams + Clone> Grid<D> {
    pub fn collect(&self) -> Vec<D> {
        match self {
            Grid::Linear { start, end, n } => {
                let mut d = start.clone();

                let vec = num_linspace(start.as_number(), end.as_number(), *n);
                vec.into_iter().map(|x| {
                    d.mut_number(x);
                    d.clone()
                }).collect()
            },
            Grid::Log { start, end, n } => {
                let mut d = start.clone();

                let vec = num_logspace(start.as_number(), end.as_number(), *n);
                vec.into_iter().map(|x| {
                    d.mut_number(x);
                    d.clone()
                }).collect()
            }
            Grid::Vec { values } => values.clone(),
            Grid::Composite { ranges } => ranges.iter().flat_map(|x| x.collect()).collect(),
        }
    }
}

pub fn num_linspace(start: Number, end: Number, n: usize) -> Vec<Number> {
    if start.is_f64() && end.is_f64() {
        let start = start.as_f64().unwrap();
        let end = end.as_f64().unwrap();
        
        linspace(start, end, n).into_iter().map(|x| Number::from_f64(x).unwrap()).collect()
    } else if start.is_i64() && end.is_i64() {
        if n == 1 {
            return vec![start];
        }
        let start = start.as_i64().unwrap();
        let end = end.as_i64().unwrap();

        let step = (end - start) / (n as i64 - 1);
        if step.abs() < 1 {
            return (start..=end).map(|x| Number::from_i128(x as i128).unwrap()).collect();
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..(n as i64) {
            let value = start + i * step;
            if value > end {
                break;
            }
            result.push(value);
        }

        result.into_iter().map(|x| Number::from_i128(x as i128).unwrap()).collect()
    } else {
        panic!("linspace grid of numbers of different type")
    }
}

pub fn num_logspace(start: Number, end: Number, n: usize) -> Vec<Number> {
    if start.is_f64() && end.is_f64() {
        let start = start.as_f64().unwrap();
        let end = end.as_f64().unwrap();
        
        logspace(start.log10(), end.log10(), n).into_iter().map(|x| Number::from_f64(x).unwrap()).collect()
    } else if start.is_u64() && end.is_u64() {
        if n == 1 {
            return vec![start];
        }
        let start_num = start.as_u64().unwrap();
        let end_num = end.as_u64().unwrap();

        let start = start_num.ilog10();
        let end = end_num.ilog10();

        let mut result = Vec::with_capacity(n);
        let step = (end - start) / (n as u32 - 1);

        for i in 0..(n as u32) {
            let value = 10u64.pow(start + i * step);
            if value > end_num {
                break;
            }

            result.push(value);
        }

        result.into_iter().map(|x| Number::from_i128(x as i128).unwrap()).collect()
    } else {
        panic!("linspace grid of numbers of different type")
    }
}

use indicatif::{
    ProgressBar,
    ProgressStyle,
};

pub struct ParallelExecutor<T> {
    indicator: Indicator,
    parallelism: Parallelism,
    execute: T,
}

impl<T: Send + Clone> ParallelExecutor<T> {
    pub fn new(execute: T) -> Self {
        Self {
            indicator: Default::default(),
            parallelism: Default::default(),
            execute,
        }
    }

    pub fn with_indicator(mut self, indicator: Indicator) -> Self {
        self.indicator = indicator;

        self
    }

    pub fn with_parallelism(mut self, parallelism: Parallelism) -> Self {
        self.parallelism = parallelism;

        self
    }

    pub fn par_execute<'a, D, F>(mut self, data: Vec<D>, op: F) -> Result<()>
    where
        D: Send + Sync + std::fmt::Debug + 'a + Clone,
        F: Fn(&mut T, D) -> Result<()> + Sync + Send,
    {
        let bar = match self.indicator {
            Indicator::Progress(style) => Some(ProgressBar::new(data.len() as u64).with_style(style)),
            Indicator::None => None,
        };

        match self.parallelism {
            Parallelism::Rayon(n) => {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
                pool.install(|| {
                    data.into_par_iter().for_each_with(self.execute, |p, d| {
                        if let Some(b) = &bar {
                            b.inc(1);
                        }
                        if let Err(e) = op(p, d.clone()) {
                            eprintln!("error {e} encountered on data {d:?}")
                        }
                    });
                });
            }
            Parallelism::Seq => {
                for d in data.into_iter() {
                    if let Some(b) = &bar {
                        b.inc(1);
                    }

                    if let Err(e) = op(&mut self.execute, d.clone()) {
                        eprintln!("error {e} encountered on data {d:?}")
                    }
                }
            }
        }

        if let Some(b) = bar {
            b.finish();
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Parallelism {
    Rayon(usize),
    Seq,
}

impl Default for Parallelism {
    fn default() -> Self {
        Parallelism::Rayon(rayon::current_num_threads())
    }
}

#[derive(Clone)]
pub enum Indicator {
    Progress(ProgressStyle),
    None,
}

impl Default for Indicator {
    fn default() -> Self {
        Self::Progress(default_progress_style())
    }
}

pub fn default_progress_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
        .unwrap()
        .progress_chars("#>-")
}
