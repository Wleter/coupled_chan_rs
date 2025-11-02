use indicatif::{ProgressBar, ProgressStyle};
use crate::prelude::*;

pub fn default_progress() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta})")
        .unwrap()
        .progress_chars("#>-")
}

pub struct DependenceProblem<T> {
    indicator: Indicator,
    parallelism: Parallelism,
    problem: T
}

impl<T: Send + Clone> DependenceProblem<T> {
    pub fn new(problem: T) -> Self {
        Self {
            indicator: Indicator::Progress(default_progress()),
            parallelism: Parallelism::Rayon(rayon::current_num_threads()),
            problem,
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

    pub fn dependence<D, F>(mut self, data: Vec<D>, op: F) -> Result<()> 
    where 
        D: Send + Sync + std::fmt::Display,
        F: Fn(&mut T, &D) -> Result<()> + Sync + Send,
    {
        let bar = match self.indicator {
            Indicator::Progress(style) => Some(ProgressBar::new(data.len() as u64).with_style(style)),
            Indicator::None => None,
        };

        match self.parallelism {
            Parallelism::Rayon(n) => {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
                pool.install(|| {
                    data.par_iter()
                        .for_each_with(self.problem, |p, d| {
                            if let Some(b) = &bar {
                                b.inc(1);
                            }
                            if let Err(e) = op(p, d) {
                                eprintln!("error {e} encountered on data {d}")
                            }
                        });
                });
            },
            Parallelism::Seq => {
                for d in data.iter() {
                    if let Some(b) = &bar {
                        b.inc(1);
                    }

                    if let Err(e) = op(&mut self.problem, d) {
                        eprintln!("error {e} encountered on data {d}")
                    }
                }
            },
        }

        if let Some(b) = bar {
            b.finish();
        }

        Ok(())
    }
}

pub enum Parallelism {
    Rayon(usize),
    Seq,
}

pub enum Indicator {
    Progress(ProgressStyle),
    None,
}
