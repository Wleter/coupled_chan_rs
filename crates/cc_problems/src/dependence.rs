use cc_math_utils::{
    linspace,
    logspace,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
};

use cc_qol_utils::{
    params::CloneAny,
    saving::{
        DataSaver,
        FileAccess,
        JsonFormat,
    },
};
use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
};
use unit_systems::quantities::{
    PhysQuantity,
    Scalar,
};

use crate::{
    calc::{
        Calc,
        SingleCalc,
    },
    parameters::TypedParamId,
    system::{
        DynParamModifications,
        System,
        new_param_modifications,
    },
};
use anyhow::Result;

#[derive(Default)]
pub struct DependantRegistry(HashMap<Box<str>, Box<dyn Fn(ParameterInput) -> DynParamModifications + Send + Sync>>);

impl DependantRegistry {
    pub fn insert_parameter<T: From<ParameterInput> + PartialEq + CloneAny>(
        mut self,
        name: &str,
        param: TypedParamId<T>,
    ) -> Self {
        let dependence = move |p: ParameterInput| {
            let value: T = p.into();

            new_param_modifications(param, value).into_dyn()
        };
        self.0.insert(name.into(), Box::new(dependence));

        self
    }

    pub fn insert_dependant(
        mut self,
        name: &str,
        dependence: impl Fn(ParameterInput) -> DynParamModifications + 'static + Send + Sync,
    ) -> Self {
        self.0.insert(name.into(), Box::new(dependence));

        self
    }

    pub fn get_modification(
        &self,
        name: impl AsRef<str>,
    ) -> &Box<dyn Fn(ParameterInput) -> DynParamModifications + Send + Sync> {
        &self.0[name.as_ref()]
    }
}

pub struct DependenceCalc<I: DeserializeOwned, D: Serialize, C: SingleCalc<I, D>> {
    single_calc: C,
    dependant_registry: DependantRegistry,
    phantom: PhantomData<(I, D)>,
}

impl<I: DeserializeOwned, D: Serialize, C: SingleCalc<I, D>> DependenceCalc<I, D, C> {
    pub fn new(single_calc: C, dependant_registry: DependantRegistry) -> Self {
        Self {
            single_calc,
            dependant_registry,
            phantom: PhantomData,
        }
    }
}

impl<I, D, C> Calc<DependenceCalcInput<I>> for DependenceCalc<I, D, C> 
where 
    I: DeserializeOwned + Send + Sync, 
    D: Serialize + Send + Sync + 'static, 
    C: SingleCalc<I, D> + Send + Sync
{
    fn calculate(&self, system: &System, input: &DependenceCalcInput<I>, worker: usize, workers: usize) -> Result<()> {
        let saver = DataSaver::new(&input.save_filepath.to_string_lossy(), JsonFormat, FileAccess::Append)?;

        if let Some(dependant) = &input.dependant {
            let modification = self.dependant_registry.get_modification(&dependant.name);
            let data = dependant.range.collect();

            let data = if workers > 1 {
                data.into_iter().skip(worker).step_by(workers).collect()
            } else {
                data
            };

            ParallelExecutor::new(system.clone())
                .with_parallelism(input.parallel_no)
                .par_execute(data, |s, d| {
                    let modification = modification(d);
                    s.modify_params(modification);

                    let data = self.single_calc.calculate(&input.calc, s)?;
                    saver.send(data);

                    Ok(())
                })?;
        } else {
            saver.send(self.single_calc.calculate(&input.calc, system)?);
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependenceCalcInput<CalcInput> {
    save_filepath: PathBuf,
    #[serde(flatten)]
    calc: CalcInput,
    #[serde(default)]
    dependant: Option<Dependant>,

    #[serde(default)]
    parallel_no: Parallelism,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dependant {
    name: Box<str>,
    range: Range,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Range {
    Linear {
        start: ParameterInput,
        end: ParameterInput,
        n: usize,
    },
    Log {
        start: ParameterInput,
        end: ParameterInput,
        n: usize,
    },
    Vec(Vec<ParameterInput>),
    Composite(Vec<Range>),
}

impl Range {
    pub fn collect(&self) -> Vec<ParameterInput> {
        match self {
            Range::Linear { start, end, n } => ParameterInput::linspace(start, end, *n),
            Range::Log { start, end, n } => ParameterInput::logspace(start, end, *n),
            Range::Vec(parameter_inputs) => parameter_inputs.clone(),
            Range::Composite(ranges) => ranges.iter().flat_map(|x| x.collect()).collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParameterInput {
    Float(f64),
    Scalar(f64, Box<str>),
    Num(i64),
}

impl ParameterInput {
    pub fn linspace(start: &Self, end: &Self, n: usize) -> Vec<Self> {
        use ParameterInput::*;
        match (start, end) {
            (Float(start), Float(end)) => linspace(*start, *end, n).into_iter().map(|x| Float(x)).collect(),
            (Num(start), Num(end)) => {
                if n == 1 {
                    return vec![Num(*start)];
                }

                let step = (end - start) / (n as i64 - 1);
                if step.abs() < 1 {
                    return (*start..=*end).map(|x| Num(x)).collect();
                }

                let mut result = Vec::with_capacity(n);
                for i in 0..(n as i64) {
                    let value = start + i * step;
                    if value > *end {
                        break;
                    }
                    result.push(value);
                }

                result.into_iter().map(|x| Num(x)).collect()
            }
            // todo! now this clones unit and requires all units to be the same
            // possibly change DependantRegistry to have dyn Fn(Range) -> Vec<DynModifications>
            (Scalar(start, a), Scalar(end, b)) => {
                assert_eq!(a, b, "Scalar Range should for now have same units");

                linspace(*start, *end, n).into_iter().map(|x| Scalar(x, a.clone())).collect()
            }
            _ => panic!("Incompatible start: {start:?}, end: {end:?} in range."),
        }
    }

    pub fn logspace(start: &Self, end: &Self, n: usize) -> Vec<Self> {
        use ParameterInput::*;
        match (start, end) {
            (Float(start), Float(end)) => logspace(start.log10(), end.log10(), n)
                .into_iter()
                .map(|x| Float(x))
                .collect(),
            (Num(start_num), Num(end_num)) => {
                assert!(*start_num > 0 && *end_num > 0, "Only positive numbers can have log grid");
                if n == 1 {
                    return vec![Num(*start_num)];
                }

                let start = start_num.ilog10();
                let end = end_num.ilog10();

                let mut result = Vec::with_capacity(n);
                let step = (end - start) / (n as u32 - 1);

                for i in 0..(n as u32) {
                    let value = 10u64.pow(start + i * step);
                    if value > (*end_num as u64) {
                        break;
                    }

                    result.push(value);
                }

                result.into_iter().map(|x| Num(x as i64)).collect()
            }
            // todo! now this clones unit and requires all units to be the same
            // possibly change DependantRegistry to have dyn Fn(Range) -> Vec<DynModifications>
            (Scalar(start, a), Scalar(end, b)) => {
                assert_eq!(a, b, "Scalar Range should for now have same units");

                logspace(start.log10(), end.log10(), n)
                    .into_iter()
                    .map(|x| Scalar(x, a.clone()))
                    .collect()
            }
            _ => panic!("Incompatible start: {start:?}, end: {end:?} in range."),
        }
    }
}

impl Into<f64> for ParameterInput {
    fn into(self) -> f64 {
        match self {
            ParameterInput::Float(f) => f,
            _ => panic!("Could not convert {self:?} to f64"),
        }
    }
}

impl<Q: PhysQuantity> Into<Scalar<Q>> for ParameterInput {
    fn into(self) -> Scalar<Q> {
        match self {
            ParameterInput::Scalar(v, u) => Scalar::new(v, Q::default(), u),
            _ => panic!("Could not convert {self:?} to Scalar<{:?}>", Q::default()),
        }
    }
}

macro_rules! parameter_input_impl_into_num {
    ($into:ident) => {
        impl Into<$into> for ParameterInput {
            fn into(self) -> $into {
                match self {
                    ParameterInput::Num(i) => i as $into,
                    _ => panic!("Could not convert {self:?} to {}", stringify!($into))
                }
            }
        }
    };
    ($($into:ident),+) => {
        $(parameter_input_impl_into_num!($into);)+
    };
}

parameter_input_impl_into_num!(usize, u64, u32, u16, u8);
parameter_input_impl_into_num!(isize, i64, i32, i16, i8);

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
