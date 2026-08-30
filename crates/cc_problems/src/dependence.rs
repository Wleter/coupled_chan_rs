use cc_math_utils::{
    linspace,
    logspace,
};
use rayon::prelude::*;
use serde_json::Value;
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
    de::{self, DeserializeOwned}, ser::SerializeTuple,
};
use unit_systems::quantities::{
    PhysQuantity,
    Scalar,
};

use crate::{
    calc::{
        Calc, CalcInput, SingleCalc
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
        &mut self,
        name: &str,
        param: TypedParamId<T>,
    ) -> &mut Self {
        let dependence = move |p: ParameterInput| {
            let value: T = p.into();

            new_param_modifications(param, value).into_dyn()
        };
        self.0.insert(name.into(), Box::new(dependence));

        self
    }

    pub fn insert_dependant(
        &mut self,
        name: &str,
        dependence: impl Fn(ParameterInput) -> DynParamModifications + 'static + Send + Sync,
    ) -> &mut Self {
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

pub struct DependenceCalc<I: CalcInput, D: Serialize, C: SingleCalc<I, D>> {
    single_calc: C,
    dependant_registry: DependantRegistry,
    phantom: PhantomData<(I, D)>,
}

impl<I: CalcInput, D: Serialize, C: SingleCalc<I, D>> DependenceCalc<I, D, C> {
    pub fn new(single_calc: C, dependant_registry: DependantRegistry) -> Self {
        Self {
            single_calc,
            dependant_registry,
            phantom: PhantomData,
        }
    }
}

impl<I, D, C> Calc for DependenceCalc<I, D, C>
where
    I: CalcInput + Send + Sync,
    D: Serialize + Send + Sync + 'static,
    C: SingleCalc<I, D> + Send + Sync,
{
    fn calculate(&self, system: &System, input: &Value, worker: usize, workers: usize) -> Result<()> {
        let input = DependenceCalcInput::<I>::from_value(input);

        if let Some(dependant) = &input.dependant {
            let saver = DataSaver::new(&input.save_filepath.to_string_lossy(), JsonFormat, FileAccess::Append)?;
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
                    let modification = modification(d.clone());
                    s.modify_params(modification);

                    let data = self.single_calc.calculate(&input.calc, s)?;
                    saver.send(DependenceData {
                        parameter: d,
                        data,
                    });

                    Ok(())
                })?;
        } else {
            let saver = DataSaver::new(&input.save_filepath.to_string_lossy(), JsonFormat, FileAccess::Create)?;
            saver.send(self.single_calc.calculate(&input.calc, system)?);
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependenceData<P, D> {
    pub parameter: P,
    pub data: D,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = "C: DeserializeOwned"))]
pub struct DependenceCalcInput<C: CalcInput> {
    pub save_filepath: PathBuf,
    #[serde(flatten)]
    pub calc: C,
    #[serde(default)]
    pub dependant: Option<Dependant>,

    #[serde(default)]
    pub parallel_no: Parallelism,
}

impl<C: CalcInput> CalcInput for DependenceCalcInput<C> {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dependant {
    pub name: Box<str>,
    pub range: Range,
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
    Vec {
       values : Vec<ParameterInput>
    },
    Composite {
       ranges : Vec<Range>
    },
}

impl Range {
    pub fn collect(&self) -> Vec<ParameterInput> {
        match self {
            Range::Linear { start, end, n } => ParameterInput::linspace(start, end, *n),
            Range::Log { start, end, n } => ParameterInput::logspace(start, end, *n),
            Range::Vec { values } => values.clone(),
            Range::Composite { ranges } => ranges.iter().flat_map(|x| x.collect()).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ParameterInput {
    Float(f64),
    Scalar(f64, Box<str>),
    Num(i64),
}

impl Serialize for ParameterInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        match self {
            Self::Float(value) => serializer.serialize_f64(*value),
            Self::Num(value) => serializer.serialize_i64(*value),
            Self::Scalar(value, unit) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(value)?;
                tuple.serialize_element(unit)?;
                tuple.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ParameterInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct ParameterInputVisitor;

        impl<'de> de::Visitor<'de> for ParameterInputVisitor {
            type Value = ParameterInput;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str(
                    "an integer, floating-point number, or [floating-point number, unit string]",
                )
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ParameterInput::Num(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let value = i64::try_from(value)
                    .map_err(|_| E::custom("integer does not fit into i64"))?;

                Ok(ParameterInput::Num(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(ParameterInput::Float(value))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let value: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;

                let unit: Box<str> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                if seq.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(3, &self));
                }

                Ok(ParameterInput::Scalar(value, unit))
            }
        }

        deserializer.deserialize_any(ParameterInputVisitor)
    }
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

impl From<ParameterInput> for f64 {
    fn from(value: ParameterInput) -> Self {
        match value {
            ParameterInput::Float(f) => f,
            _ => panic!("Could not convert {value:?} to f64"),
        }
    }
}

impl<Q: PhysQuantity> From<ParameterInput> for Scalar<Q> {
    fn from(value: ParameterInput) -> Self {
        match value {
            ParameterInput::Scalar(v, u) => Scalar::new(v, Q::default(), u),
            _ => panic!("Could not convert {value:?} to Scalar<{:?}>", Q::default()),
        }
    }
}

macro_rules! parameter_input_impl_from_num {
    ($into:ident) => {
        impl From<ParameterInput> for $into {
            fn from(value: ParameterInput) -> Self {
                match value {
                    ParameterInput::Num(i) => i as $into,
                    _ => panic!("Could not convert {value:?} to {}", stringify!($into))
                }
            }
        }
    };
    ($($into:ident),+) => {
        $(parameter_input_impl_from_num!($into);)+
    };
}

parameter_input_impl_from_num!(usize, u64, u32, u16, u8);
parameter_input_impl_from_num!(isize, i64, i32, i16, i8);

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
