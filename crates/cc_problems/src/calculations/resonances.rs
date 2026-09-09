use std::{marker::PhantomData, mem::swap};

use anyhow::bail;
use hilbert_space::faer::complex::Complex64;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    calculations::{
        Modified, SingleCalc, bound_states::{BoundStateCalc, BoundStateCalcInput, BoundStateData, modify_from_f64, modify_to_f64}, dependence::{DependenceCalc, ModifyParams}, scattering::{ScatteringCalc, ScatteringCalcInput}
    },
    problems::Problem,
};

pub struct ResonancesCalc<P: Problem, D: ModifyParams<P = P, C = BoundStateCalcInput<D>>> {
    scattering: ScatteringCalc<P>,
    bound: BoundStateCalc<P, D>,
    phantom: PhantomData<(P, D)>,
}

impl<P: Problem, D: ModifyParams<P = P, C = BoundStateCalcInput<D>>> ResonancesCalc<P, D> {
    pub fn new(scattering: ScatteringCalc<P>, bound: BoundStateCalc<P, D>) -> Self {
        Self {
            scattering,
            bound,
            phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ResonancesData {
    parameter_res: f64,
    a_bg: f64,
    width: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    decay_width: Option<f64>
}

#[derive(Clone, Deserialize)]
#[serde(bound = "D: DeserializeOwned")]
pub struct ResonancesInput<D: ModifyParams> {
    pub scattering_input: ScatteringCalcInput,
    pub bound_input: BoundStateCalcInput<D>,

    #[serde(default = "default_t_min")]
    pub t_min: f64,
    #[serde(default = "default_t_max")]
    pub t_max: f64,

    #[serde(default = "default_eps")]
    eps: f64,

    #[serde(default = "default_max_iter")]
    max_iter: usize,
}

fn default_t_min() -> f64 {
    0.1
}
fn default_t_max() -> f64 {
    1.0
}
fn default_eps() -> f64 {
    1e-3
}
fn default_max_iter() -> usize {
    20
}


impl<P, D> SingleCalc for ResonancesCalc<P, D> 
where
    P: Problem,
    D: ModifyParams<P = P, C = BoundStateCalcInput<D>> + Clone,
{
    type P = P;
    type CalcInput = ResonancesInput<D>;
    type Data = ResonancesData;

    fn calculate(
        &self,
        modified: &mut Modified<P, Self::CalcInput>,
        problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        let mut modified_bound = Self::modified_bound(modified);
        let results: Vec<anyhow::Result<BoundStateData>> = 
            self.bound.calculate(&mut modified_bound, problem).into_iter().collect();

        let t_max = modified.calc_input.t_max;
        let t_min = modified.calc_input.t_min;

        results.into_iter().map(move |res| {
            let res = res?;
            let mut p_mod = modified.calc_input.bound_input.dependant_err.clone();
            let p_err = modify_to_f64(&p_mod);

            let min_bound = res.parameter - 10.0 * p_err;
            let max_bound = res.parameter + 10.0 * p_err;

            let mut p1 = res.parameter;
            let mut p2 = p1 - p_err;
            let mut p3 = p1 + p_err;

            modify_from_f64(&mut p_mod, p1);
            let mut modified_bound = Self::modified_bound(modified);
            p_mod.modify(&mut modified_bound);
            let mut modified_scattering = Self::modified_scattering(modified);
            let mut s1 = self.scattering.calculate(&mut modified_scattering, problem).into_iter().next().unwrap()?.s_length;

            modify_from_f64(&mut p_mod, p2);
            let mut modified_bound = Self::modified_bound(modified);
            p_mod.modify(&mut modified_bound);
            let mut modified_scattering = Self::modified_scattering(modified);
            let mut s2 = self.scattering.calculate(&mut modified_scattering, problem).into_iter().next().unwrap()?.s_length;

            modify_from_f64(&mut p_mod, p3);
            let mut modified_bound = Self::modified_bound(modified);
            p_mod.modify(&mut modified_bound);
            let mut modified_scattering = Self::modified_scattering(modified);
            let mut s3 = self.scattering.calculate(&mut modified_scattering, problem).into_iter().next().unwrap()?.s_length;

            for _ in 0..modified.calc_input.max_iter {
                let (a_bg, p_res, width) = get_resonance([
                    (p1, s1),
                    (p2, s2),
                    (p3, s3)
                ]);

                let mut d1 = (p_res - p1) / width;
                let mut d2 = (p_res - p2) / width;
                let mut d3 = (p_res - p3) / width;

                if d1.abs() > d2.abs() {
                    swap(&mut d1, &mut d2);
                    swap(&mut s1, &mut s2);
                    swap(&mut p1, &mut p2);
                }
                if d2.abs() > d3.abs() {
                    swap(&mut d2, &mut d3);
                    swap(&mut s2, &mut s3);
                    swap(&mut p2, &mut p3);
                }
                if d1.abs() > d2.abs() {
                    swap(&mut d1, &mut d2);
                    swap(&mut s1, &mut s2);
                    swap(&mut p1, &mut p2);
                }

                let mut converged = true;
                if t_max > d3.abs() || 2.0 * t_max < d3.abs() {
                    swap(&mut d3, &mut d2);
                    swap(&mut s3, &mut s2);
                    swap(&mut p3, &mut p2);
                    converged = false
                } else if t_min > d2.abs() || 2.0 * t_min < d2.abs() || d2 * d3 > 0.0 {
                    converged = false
                } else if d1.abs() > modified.calc_input.eps {
                    swap(&mut d1, &mut d2);
                    swap(&mut s1, &mut s2);
                    swap(&mut p1, &mut p2);
                    converged = false
                }

                if converged {
                    if !(min_bound..=max_bound).contains(&p_res) {
                        bail!("Resonance outside of the searched region")
                    }

                    return Ok(ResonancesData {
                        parameter_res: p_res,
                        a_bg,
                        width,
                        decay_width: None,
                    });
                }

                if d3.abs() < t_max {
                    p2 = p_res + d3.signum() * 1.5 * t_max * width
                } else if d1.abs() < t_min {
                    p2 = p_res + d3.signum() * 1.5 * t_min * width
                } else {
                    p2 = p_res
                }
                modify_from_f64(&mut p_mod, p2);
                let mut modified_bound = Self::modified_bound(modified);
                p_mod.modify(&mut modified_bound);
                let mut modified_scattering = Self::modified_scattering(modified);
                s2 = self.scattering.calculate(&mut modified_scattering, problem).into_iter().next().unwrap()?.s_length;
            }

            Err(anyhow::anyhow!("Could not obtain resonance characterization in {} iterations", modified.calc_input.max_iter))
        })
    }
}

impl<P: Problem, D: ModifyParams<P = P, C = BoundStateCalcInput<D>>> ResonancesCalc<P, D> {
    fn modified_bound<'b, 'a>(modified: &'b mut Modified<'a, P, ResonancesInput<D>>) -> Modified<'b, P, BoundStateCalcInput<D>> {
        Modified { 
            system: modified.system, 
            basis: modified.basis, 
            params: modified.params, 
            calc_input: &mut modified.calc_input.bound_input 
        }
    }

    fn modified_scattering<'b, 'a>(modified: &'b mut Modified<'a, P, ResonancesInput<D>>) -> Modified<'b, P, ScatteringCalcInput> {
        Modified { 
            system: modified.system, 
            basis: modified.basis, 
            params: modified.params, 
            calc_input: &mut modified.calc_input.scattering_input
        }
    }
}

fn get_resonance(s: [(f64, Complex64); 3]) -> (f64, f64, f64) {
    let [(p1, s1), (p2, s2), (p3, s3)] = s;
    let a1 = s1.re;
    let a2 = s2.re;
    let a3 = s3.re;

    let rho = (p3 - p1) / (p2 - p1) * (a2 - a1) / (a3 - a1);
    let p_res = (p3 - p2 * rho) / (1.0 - rho);
    let a_bg_width = (p3 - p_res) * (p1 - p_res) * (a3 - a1) / (p3 - p1);
    let a_bg = a1 + a_bg_width / (p1 - p_res);
    let width = a_bg_width / a_bg;

    (a_bg, p_res, width)
}

pub type AdiabatsScan<P, D> = DependenceCalc<ResonancesCalc<P, D>, D>;
