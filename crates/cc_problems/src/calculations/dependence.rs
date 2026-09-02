use std::{
    collections::HashMap,
    path::PathBuf,
};

use anyhow::Result;
use cc_math_utils::{
    linspace,
    logspace,
};
use cc_qol_utils::{
    params::CloneAny,
    saving::{
        DataSaver,
        FileAccess,
        JsonFormat,
    },
};
use rayon::prelude::*;
use serde::{
    Deserialize,
    Serialize,
    de::DeserializeOwned,
    ser::SerializeTuple,
};
use serde_json::Value;
use unit_systems::quantities::{
    PhysQuantity,
    Scalar,
};

use crate::{
    calculations::{
        Calc,
        SingleCalc,
    },
    parameters::TypedParamId,
    problems::{
        Problem,
        TypedProblemInput,
    },
    system::{
        DynParamModifications,
        new_param_modifications,
    },
};

pub struct CalcModifications<P: Problem, CalcInput>(pub HashMap<Box<str>, ModificationType<P, CalcInput>>);

impl<P: Problem, C> CalcModifications<P, C> {
    pub fn new(
        basis_mod: HashMap<Box<str>, Modification<P::BasisRecipe>>,
        params_mod: HashMap<Box<str>, ParametersMod>,
        calc_mod: HashMap<Box<str>, Modification<C>>,
    ) -> Self {
        let mut mods = HashMap::from_iter(basis_mod.into_iter().map(|(a, b)| (a, ModificationType::BasisRecipe(b))));
        mods.extend(params_mod.into_iter().map(|(a, b)| (a, ModificationType::Parameter(b))));
        mods.extend(calc_mod.into_iter().map(|(a, b)| (a, ModificationType::CalcParameter(b))));

        Self(mods)
    }

    pub fn from_basis_modifications(basis_mod: HashMap<Box<str>, Modification<P::BasisRecipe>>) -> Self {
        Self(HashMap::from_iter(
            basis_mod.into_iter().map(|(a, b)| (a, ModificationType::BasisRecipe(b))),
        ))
    }

    pub fn from_params_modifications(params_mod: HashMap<Box<str>, ParametersMod>) -> Self {
        Self(HashMap::from_iter(
            params_mod.into_iter().map(|(a, b)| (a, ModificationType::Parameter(b))),
        ))
    }

    pub fn from_calc_modifications(calc_mod: HashMap<Box<str>, Modification<C>>) -> Self {
        Self(HashMap::from_iter(
            calc_mod.into_iter().map(|(a, b)| (a, ModificationType::CalcParameter(b))),
        ))
    }
}

pub enum ModificationType<P: Problem, CalcInput> {
    BasisRecipe(Modification<P::BasisRecipe>),
    Parameter(ParametersMod),
    CalcParameter(Modification<CalcInput>),
}

pub struct Modification<P>(Box<dyn Fn(&mut P, Value) -> bool + Send + Sync>);

pub fn change_value<T: PartialEq + DeserializeOwned>(value: &mut T, new_value: Value) -> Result<bool> {
    let new_value = serde_json::from_value(new_value)?;

    if value == &new_value {
        Ok(false)
    } else {
        *value = new_value;
        Ok(true)
    }
}

impl<P> Modification<P> {
    pub fn new(modification: impl Fn(&mut P, Value) -> bool + Send + Sync + 'static) -> Self {
        Self(Box::new(modification))
    }
}

pub struct ParametersMod(pub Box<dyn Fn(Value) -> DynParamModifications + Send + Sync>);

impl ParametersMod {
    pub fn from_id<T: DeserializeOwned + PartialEq + CloneAny>(id: TypedParamId<T>) -> Self {
        Self(Box::new(move |v| {
            new_param_modifications(id, serde_json::from_value(v).unwrap()).into_dyn()
        }))
    }
}

pub struct DependenceCalc<C: SingleCalc> {
    single_calc: C,
    modifications: CalcModifications<C::P, C::CalcInput>,
}

impl<C: SingleCalc> DependenceCalc<C> {
    pub fn new(single_calc: C, modifications: CalcModifications<C::P, C::CalcInput>) -> Self {
        Self {
            single_calc,
            modifications,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceCalcInput<C> {
    pub save_filepath: PathBuf,
    #[serde(default = "default_file_access")]
    pub save_option: FileAccess,

    #[serde(flatten)]
    pub calc: C,
    #[serde(default)]
    pub dependant: Option<Dependant>,

    #[serde(default)]
    pub parallel_no: Parallelism,
}

#[derive(Clone, Debug, Serialize)]
pub struct DependenceData<P, D> {
    pub parameter: P,
    pub data: D,
}

fn default_file_access() -> FileAccess {
    FileAccess::Append
}

impl<C> Calc for DependenceCalc<C>
where
    C: SingleCalc,
{
    type P = C::P;
    type CalcInput = DependenceCalcInput<C::CalcInput>;

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
        let system = problem.build(&input.basis_recipe, &input.parameters);
        let save_filepath = &input.calculation_parameters.save_filepath.to_string_lossy();
        let save_option = input.calculation_parameters.save_option;
        let parallel_no = input.calculation_parameters.parallel_no;

        if let Some(dependant) = &input.calculation_parameters.dependant {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;
            let modification = self.modifications.0.get(&dependant.name).ok_or(anyhow::anyhow!(
                "{} is not registered in possible modifications",
                &dependant.name
            ))?;
            let data = dependant.range.collect();

            let data = if input.workers > 1 {
                data.into_iter().skip(input.worker).step_by(input.workers).collect()
            } else {
                data
            };

            ParallelExecutor::new((system, input))
                .with_parallelism(parallel_no)
                .par_execute(data, |(s, input), d| {
                    let b = &mut input.basis_recipe;
                    let p = &mut input.parameters;
                    let c = &mut input.calculation_parameters;

                    // todo! change d to be Value from the start
                    let d = serde_json::to_value(d)?;
                    match modification {
                        ModificationType::BasisRecipe(modification) => {
                            if modification.0(b, d.clone()) {
                                *s = problem.build(b, p)
                            }
                        }
                        ModificationType::Parameter(parameters_mod) => {
                            s.modify_params(parameters_mod.0(d.clone()));
                        }
                        ModificationType::CalcParameter(modification) => {
                            modification.0(&mut c.calc, d.clone());
                        }
                    }

                    let data = self.single_calc.calculate(s, b, p, &mut c.calc, problem)?;
                    saver.send(DependenceData { parameter: d, data });

                    Ok(())
                })?;
        } else {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;

            let mut input = input;
            let mut system = system;
            saver.send(self.single_calc.calculate(
                &mut system,
                &mut input.basis_recipe,
                &mut input.parameters,
                &mut input.calculation_parameters.calc,
                problem,
            )?);
        }

        Ok(())
    }
}

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
        values: Vec<ParameterInput>,
    },
    Composite {
        ranges: Vec<Range>,
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

use serde::de;

impl<'de> Deserialize<'de> for ParameterInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct ParameterInputVisitor;

        impl<'de> de::Visitor<'de> for ParameterInputVisitor {
            type Value = ParameterInput;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an integer, floating-point number, or [floating-point number, unit string]")
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
                let value = i64::try_from(value).map_err(|_| E::custom("integer does not fit into i64"))?;

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
                let value: f64 = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;

                let unit: Box<str> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;

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
