use std::path::PathBuf;

use anyhow::Result;
use cc_qol_utils::saving::{
    DataSaver,
    FileAccess,
    JsonFormat,
};
use rayon::prelude::*;
use serde::{
    Deserialize,
    Deserializer,
    Serialize,
};
use serde_json::{
    Number,
    Value,
};
use smallvec::SmallVec;

use crate::{
    calculations::{
        Calc,
        Modified,
        SingleCalc,
        modifications::{
            ModificationAction,
            ModifyParam,
            ModifyRegistry,
        },
    },
    parameters::Parameters,
    problems::{
        Problem,
        TypedProblemInput,
    },
    system::System,
};

#[derive(Clone, Debug)]
pub struct NamedValue {
    pub name: Box<str>,
    pub value: Value,
}

impl<'de> Deserialize<'de> for NamedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = serde_json::Value::deserialize(deserializer)?;

        let obj = map
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("expected an object"))?;

        if obj.len() != 1 {
            return Err(serde::de::Error::custom("expected exactly one field"));
        }

        let (name, value) = obj.iter().next().unwrap();

        Ok(NamedValue {
            name: name.as_str().into(),
            value: value.to_owned(),
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum GridParams {
    Cartesian { components: Vec<GridParams> },
    Line { components: Vec<GridParams> },
    Linear { start: NamedValue, end: NamedValue, n: usize },
    Log { start: NamedValue, end: NamedValue, n: usize },
    Points { name: Box<str>, values: Vec<Value> },
    Sum { components: Vec<GridParams> },
}

impl GridParams {
    pub fn len(&self) -> usize {
        match self {
            GridParams::Cartesian { components } => {
                assert!(!components.is_empty(), "Empty cartesian grid param components");
                components.iter().map(|x| x.len()).product()
            }
            GridParams::Line { components } => {
                assert!(!components.is_empty(), "Empty line grid param components");
                let mut lens = components.iter().map(|x| x.len());

                let n = lens.next().unwrap();
                for l in lens {
                    assert_eq!(n, l, "Line grid params components have unequal grid points")
                }

                n
            }
            GridParams::Linear { start: _, end: _, n } => *n,
            GridParams::Log { start: _, end: _, n } => *n,
            GridParams::Points { name: _, values } => values.len(),
            GridParams::Sum { components } => components.iter().map(|x| x.len()).sum(),
        }
    }

    pub fn get<P: Problem, C>(
        &self,
        index: usize,
        registry: &ModifyRegistry<P, C>,
    ) -> (SmallVec<[Number; 3]>, SmallVec<[Box<dyn ModifyParam<P = P, C = C>>; 3]>) {
        assert!(index < self.len(), "Index of GridParams larger than its size");
        let mut numbers: SmallVec<[Number; 3]> = SmallVec::new();
        let mut modifiers: SmallVec<[Box<dyn ModifyParam<P = P, C = C>>; 3]> = SmallVec::new();

        match self {
            GridParams::Cartesian { components } => {
                let mut reduced_index = index;
                for (i, c) in components.iter().enumerate().rev() {
                    let c_len = c.len();
                    let index = reduced_index % c_len;
                    reduced_index /= c_len;

                    let (n, m) = components[i].get(index, registry);

                    numbers.insert_many(0, n);
                    modifiers.insert_many(0, m);
                }
            }
            GridParams::Line { components } => {
                let mut numbers: SmallVec<[Number; 3]> = SmallVec::new();
                for c in components {
                    let (n, m) = c.get(index, registry);
                    numbers.extend(n);
                    modifiers.extend(m);
                }
            }
            GridParams::Linear { start, end, n } => {
                assert_eq!(
                    start.name, end.name,
                    "start and end modified param for linspace should be the same"
                );
                let modifier = &registry.0[&start.name];

                let mut modifier_s = (modifier.recipe)(start.value.clone());
                let start = modifier_s.as_number();
                let end = (modifier.recipe)(end.value.clone()).as_number();
                let value = num_linspace(start, end, *n, index);

                numbers = smallvec::smallvec![value.clone()];
                modifier_s.mut_number(value);
                modifiers = smallvec::smallvec![modifier_s];
            }
            GridParams::Log { start, end, n } => {
                assert_eq!(
                    start.name, end.name,
                    "start and end modified param for linspace should be the same"
                );
                let modifier = &registry.0[&start.name];

                let mut modifier_s = (modifier.recipe)(start.value.clone());
                let start = modifier_s.as_number();
                let end = (modifier.recipe)(end.value.clone()).as_number();
                let value = num_logspace(start, end, *n, index);

                numbers = smallvec::smallvec![value.clone()];
                modifier_s.mut_number(value);
                modifiers = smallvec::smallvec![modifier_s];
            }
            GridParams::Points { name, values } => {
                let modifier = &registry.0[name];
                let modifier_s = (modifier.recipe)(values[index].clone());
                numbers = smallvec::smallvec![modifier_s.as_number()];
                modifiers = smallvec::smallvec![modifier_s];
            }
            GridParams::Sum { components } => {
                let mut size_start = 0;

                for (i, s) in components.iter().map(|x| x.len()).enumerate() {
                    if size_start + s <= index {
                        size_start += s;
                        continue;
                    }

                    (numbers, modifiers) = components[i].get(index - size_start, registry);
                    break;
                }
            }
        }

        (numbers, modifiers)
    }
}

pub struct DependenceCalc<C: SingleCalc> {
    single_calc: C,
    registry: ModifyRegistry<C::P, C::CalcInput>,
}

impl<C: SingleCalc> DependenceCalc<C> {
    pub fn new(single_calc: C, registry: ModifyRegistry<C::P, C::CalcInput>) -> Self {
        Self { single_calc, registry }
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
    pub grid: Option<GridParams>,

    #[serde(default)]
    pub parallelism: Parallelism,
}

#[derive(Clone, Debug, Serialize)]
pub struct DependenceData<P, D> {
    pub index: usize,
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
        let hamiltonian_spec = Self::P::build(&input.basis_recipe, &input.parameters);
        let system = System::new(hamiltonian_spec, input.parameters.registry());
        let save_filepath = &input.calc_parameters.save_filepath.to_string_lossy();
        let save_option = input.calc_parameters.save_option;
        let parallel_no = input.calc_parameters.parallelism;
        println!("{:?}", system.basis());

        if let Some(grid) = &input.calc_parameters.grid {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;
            let n = grid.len();
            let indices: Vec<usize> = (0..n).skip(input.worker).step_by(input.workers).collect();

            ParallelExecutor::new(system)
                .with_parallelism(parallel_no)
                .par_execute(indices, |s, i| {
                    let mut input = input.clone();
                    let (numbers, modifiers) = grid.get(i, &self.registry);

                    let mut modified = Modified {
                        system: s,
                        basis: &mut input.basis_recipe,
                        params: &mut input.parameters,
                        calc_input: &mut input.calc_parameters.calc,
                    };

                    let mut prep: Vec<ModificationAction> =
                        modifiers.as_ref().iter().map(|x| x.prep_modify(&mut modified)).collect();

                    prep.sort_by(|a, b| match (a, b) {
                        (ModificationAction::BasisChange, _) => std::cmp::Ordering::Less,
                        (_, ModificationAction::BasisChange) => std::cmp::Ordering::Greater,
                        (ModificationAction::ParamModify(_), _) => std::cmp::Ordering::Less,
                        (_, ModificationAction::ParamModify(_)) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Greater,
                    });

                    let mut last_basis_change = false;
                    for p in prep {
                        if matches!(p, ModificationAction::BasisChange) {
                            last_basis_change = true
                        } else if last_basis_change {
                            let spec = Self::P::build(modified.basis, modified.params);
                            *modified.system = System::new(spec, modified.system.param_registry().to_owned())
                        }
                        if let ModificationAction::ParamModify(modify) = p {
                            modified.system.modify_params(modify);
                        }
                    }

                    let mut result = Ok(());
                    for data in self.single_calc.calculate(&mut modified, problem) {
                        if let Ok(data) = data {
                            saver.send(DependenceData {
                                index: i,
                                parameter: numbers.clone(),
                                data,
                            });
                        } else if let Err(err) = data {
                            eprintln!("{}", err);
                            result = Err(err);
                        }
                    }

                    result
                })?;
        } else {
            let saver = DataSaver::new(save_filepath, JsonFormat, save_option)?;

            let mut input = input;
            let mut system = system;
            let mut modified = &mut Modified {
                system: &mut system,
                basis: &mut input.basis_recipe,
                params: &mut input.parameters,
                calc_input: &mut input.calc_parameters.calc,
            };
            let calc = self.single_calc.calculate(&mut modified, problem);

            let mut result = Ok(());
            for data in calc {
                if let Ok(data) = data {
                    saver.send(data);
                } else if let Err(err) = data {
                    eprintln!("{}", err);
                    result = Err(err);
                }
            }

            result?;
        }

        Ok(())
    }
}

pub fn num_linspace(start: Number, end: Number, n: usize, i: usize) -> Number {
    if n == 1 {
        return start;
    }

    if start.is_f64() && end.is_f64() {
        let start = start.as_f64().unwrap();
        let end = end.as_f64().unwrap();

        let step = (end - start) / (n as f64 - 1.0);

        Number::from_f64(start + i as f64 * step).unwrap()
    } else if start.is_i64() && end.is_i64() {
        let start = start.as_i64().unwrap();
        let end = end.as_i64().unwrap();

        let step = (end - start) / (n as i64 - 1);
        if step.abs() < 1 {
            return Number::from_i128(start as i128 + i as i128).unwrap();
        }

        Number::from_i128((start + i as i64 * step) as i128).unwrap()
    } else {
        panic!("different start and end number types for linspace provided")
    }
}

pub fn num_logspace(start: Number, end: Number, n: usize, i: usize) -> Number {
    if n == 1 {
        return start;
    }

    if start.is_f64() && end.is_f64() {
        let start_num = start.as_f64().expect("Logspace can only be performed on positive numbers");
        let end_num = end.as_f64().expect("Logspace can only be performed on positive numbers");

        let start = start_num.log10();
        let end = end_num.log10();

        let step = (end - start) / (n as f64 - 1.0);

        Number::from_f64((10f64).powf(start + (i as f64) * step)).unwrap()
    } else if start.is_i64() && end.is_i64() {
        let start_num = start.as_u64().expect("Logspace can only be performed on positive numbers");
        let end_num = end.as_u64().expect("Logspace can only be performed on positive numbers");

        let start = start_num.ilog10();
        let end = end_num.ilog10();

        let step = (end - start) / (n as u32 - 1);

        Number::from_i128((10u64.pow(start + i as u32 * step)) as i128).unwrap()
    } else {
        panic!("different start and end number types for logspace provided")
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
