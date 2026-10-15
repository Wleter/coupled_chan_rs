use std::f64::consts::{PI, TAU};
use cc_math_utils::airy::eval_airy;

use crate::{Boundary, Direction, LogDeriv, Nodes, Propagator, Solution, WaveStorage, WithNodeCount, WithWaveStorage, single_channel::WFunction, step_strategy::LinearRefStepStrat};

const ROOT_3: f64 = 1.732050807;

/// based on multichannel-channel version of airy log-derivative
pub struct AiryLogDerivative<'a, W: WFunction> {
    w_function: &'a W,
    step: LinearRefStepStrat,
    nodes: Nodes,
    wave_storage: Option<WaveStorage<f64>>,

    dr_prev: f64,
    w_ref_prev: f64,

    solution: Solution<LogDeriv<f64>>,
}

impl<'a, W: WFunction> AiryLogDerivative<'a, W> {
    pub fn new(w_function: &'a W, step: LinearRefStepStrat, boundary: Boundary<f64>) -> Self {
        let r = boundary.r_start;

        let dr = match boundary.direction {
            Direction::Inwards => -(step.init_step(r).abs()),
            Direction::Outwards => step.init_step(r).abs(),
        };
        let sol = LogDeriv(boundary.derivative / boundary.value);

        let w_function_plus = w_function.value(r + dr / 2. * (-1. + 1. / ROOT_3));
        let w_function_minus = w_function.value(r + dr / 2. * (-1. - 1. / ROOT_3));
        let w_ref = (w_function_plus + w_function_minus) / 2.0;

        Self {
            w_function,
            dr_prev: dr,
            w_ref_prev: w_ref,
            step,
            solution: Solution { r, dr, sol },
            nodes: Nodes(0),
            wave_storage: None,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        if let Some(r) = r && (self.solution.r - r).abs() < self.solution.dr.abs() {
            self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
        }
        let h = self.solution.dr;

        let w_function_plus = self.w_function.value(self.solution.r + h / 2. * (1. + 1. / ROOT_3));
        let w_function_minus = self.w_function.value(self.solution.r + h / 2. * (1. - 1. / ROOT_3));

        let w_ref = (w_function_plus + w_function_minus) / 2.0;
        let w_deriv_ref = (w_function_plus - w_function_minus) * ROOT_3 / h ;

        let imbeddings = get_imbeddings(w_ref, w_deriv_ref, h);

        let inversion_term = 1. / (self.solution.sol.0 + imbeddings.y1);
        let sol_new = imbeddings.y4 - (imbeddings.y23).powi(2) * inversion_term;

        if let Some(wave_storage) = &mut self.wave_storage {
            let connection = imbeddings.y23 * inversion_term;

            wave_storage.push(self.solution.r, &connection)
        }

        self.nodes.0 += if self.solution.dr > 0. {
            (inversion_term < 0.) as u64
        } else {
            (inversion_term >= 0.) as u64
        };

        let dr_mid = (self.dr_prev + h) / 2.;
        let w_dderiv_estimate = 2. / (dr_mid * dr_mid) * (w_ref - self.w_ref_prev + w_deriv_ref * dr_mid).abs();
        // w_deriv_off_estimate = 0 so average k does not matter
        let dr_new = self.step.get_step(self.solution.r, self.dr_prev, 0.0, 0.0, Some(w_dderiv_estimate));
        
        self.w_ref_prev = self.solution.sol.0;
        self.solution.sol.0 = sol_new;
        self.solution.r += self.solution.dr;
        self.dr_prev = self.solution.dr;
        self.solution.dr = dr_new;
    }
}

impl<W: WFunction> Propagator<LogDeriv<f64>> for AiryLogDerivative<'_, W> {
    fn step(&mut self) -> &Solution<LogDeriv<f64>> {
        self.step_r_target(None);
        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<f64>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        &self.solution
    }
}

impl<W: WFunction> WithNodeCount for AiryLogDerivative<'_, W> {
    fn nodes(&self) -> crate::Nodes {
        self.nodes
    }
}

impl<W: WFunction> WithWaveStorage<f64> for AiryLogDerivative<'_, W> {
    fn init_wave_storage(&mut self) {
        self.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<f64>> {
        self.wave_storage.as_ref()
    }
}

// Log-derivative imbedding airy-function approximation based on:
// A renormalized potential-following propagation algorithm for solving the coupled-
// channels equations. doi: 10.1063/1.4891809, Appendix A2 equations.

pub(crate) struct Imbeddings {
    pub y1: f64,
    pub y23: f64,
    pub y4: f64,
}

const AIRY_ACC: f64 = 1e-20;
const AIRY_N_MAX: usize = 100;
const ASYMPTOTIC_CRITERION: f64 = 20.0;

pub(crate) fn get_imbeddings(w_ref: f64, mut w_deriv_ref: f64, dr: f64) -> Imbeddings {
    // for stability, todo! reconsider value to be on dimensionless quantity
    if w_deriv_ref == 0.0 {
        w_deriv_ref = 1e-30
    }

    let alpha = -f64::cbrt(w_deriv_ref);
    let rho = -dr / 2.0;
    let beta = w_ref / w_deriv_ref;
    let x = alpha * (rho + beta);
    let dx = alpha * dr;

    if x.abs() >= ASYMPTOTIC_CRITERION {
        let c2_w = airy_x(x, dx, AIRY_N_MAX, AIRY_ACC);

        let c1_w_neg = airy_y(x, dx, AIRY_N_MAX, AIRY_ACC);
        let c4_w_neg = airy_y(x + dx, -dx, AIRY_N_MAX, AIRY_ACC);

        let w = alpha / PI;

        Imbeddings {
            y1: -alpha * c1_w_neg.0 / c2_w.0 * f64::exp(c1_w_neg.1 - c2_w.1),
            y23: w / c2_w.0 * f64::exp(-c2_w.1),
            y4: -alpha * c4_w_neg.0 / c2_w.0 * f64::exp(c4_w_neg.1 - c2_w.1),
        }
    } else {
        let y = x + dx;

        let airy_x = eval_airy(x);
        let airy_y = eval_airy(y);

        let zeta1 = airy_x.exponent();
        let (m1, theta1) = airy_x.red_moduli_phase();
        let (n1, phi1) = airy_x.red_moduli_phase_deriv();
        let zeta2 = airy_y.exponent();
        let (m2, theta2) = airy_y.red_moduli_phase();
        let (n2, phi2) = airy_y.red_moduli_phase_deriv();

        if x >= 0. && y >= 0. {
            let d = f64::sinh(theta1 - theta2) + f64::tanh(zeta2 - zeta1) * f64::sinh(theta1 + theta2);

            Imbeddings {
                y1: alpha * n1 / m1 * (f64::sinh(theta2 - phi1) - f64::tanh(zeta2 - zeta1) * f64::sinh(theta2 + phi1)) / d,
                y23: alpha / (PI * m1 * m2 * f64::cosh(zeta2 - zeta1) * d),
                y4: alpha * n2 / m2 * (f64::sinh(theta1 - phi2) - f64::tanh(zeta2 - zeta1) * f64::sinh(theta1 + phi2)) / d,
            }
        } else if x < 0. && y < 0. {
            Imbeddings {
                y1: alpha * n1 / m1 * f64::sin(phi1 - theta2) / f64::sin(theta2 - theta1),
                y23: alpha / (PI * m1 * m2 * f64::sin(theta2 - theta1)),
                y4: alpha * n2 / m2 * f64::sin(phi2 - theta1) / f64::sin(theta2 - theta1),
            }
        } else if x >= 0. {
            let d = f64::cosh(theta1) * (
                -f64::cos(theta2) * (1.0 + f64::tanh(zeta1)) 
                + f64::tanh(theta1) * f64::sin(theta2) * (1.0 - f64::tanh(zeta1)) 
            );

            Imbeddings {
                y1: alpha * n1 / m1 * f64::cosh(phi1) * (
                    f64::cos(theta2) * (1.0 + f64::tanh(zeta1)) 
                    - f64::tanh(phi1) * f64::sin(theta2) * (1.0 - f64::tanh(zeta1)) 
                ) / d,
                y23: alpha / (PI * m1 * m2 * f64::cosh(zeta1) * d),
                y4: alpha * n2 / m2 * f64::cosh(theta1) * (
                    -f64::cos(phi2) * (1.0 + f64::tanh(zeta1)) 
                    + f64::tanh(theta1) * f64::sin(phi2) * (1.0 - f64::tanh(zeta1)) 
                ) / d,
            }
        } else {
            let d = f64::cosh(theta2) * (
                f64::cos(theta1) * (1.0 + f64::tanh(zeta2)) 
                - f64::tanh(theta2) * f64::sin(theta1) * (1.0 - f64::tanh(zeta2)) 
            );

            Imbeddings {
                y1: alpha * n1 / m1 * f64::cosh(theta2) * (
                    -f64::cos(phi1) * (1.0 + f64::tanh(zeta2)) 
                    + f64::tanh(theta2) * f64::sin(phi1) * (1.0 - f64::tanh(zeta2)) 
                ) / d,
                y23: alpha / (PI * m1 * m2 * f64::cosh(zeta2) * d),
                y4: alpha * n2 / m2 * f64::cosh(phi2) * (
                    f64::cos(theta1) * (1.0 + f64::tanh(zeta2)) 
                    - f64::tanh(phi2) * f64::sin(theta1) * (1.0 - f64::tanh(zeta2)) 
                ) / d,
            }
        }
    }
}

/// Calculates X = [Ai(x)*Bi(x+delta)-Ai(x+delta)*BI(x)] using asymptotic expansion in x,
/// assumes |x| >> 1 and  |delta / x| << 1
/// returns pair (A, B) of result A * exp(B).
fn airy_x(x: f64, delta: f64, n_max: usize, acc: f64) -> (f64, f64) {
    if x > 0.0 {
        let delta_x = delta / x;
        let zeta = 2. / 3. * x.sqrt() * x;
        let pow_series_1_4 = power_expansion_at_one(delta_x, 0.25, n_max, acc);
        let pow_series_3_2 = power_expansion_at_one(delta_x, 1.5, n_max, acc);
        let zeta_shifted = zeta * (1. + pow_series_3_2);

        let pre_factor = 1. / (TAU * x.sqrt() * (1. + pow_series_1_4));
        let exp = pow_series_3_2 * zeta;

        let series_1 = airy_power_series(-zeta, n_max, acc);
        let series_1_shifted = airy_power_series(zeta_shifted, n_max, acc);

        let exp_vanishing = f64::exp(-2. * exp.abs());
        let series_2 = airy_power_series(zeta, n_max, acc);
        let series_2_shifted = airy_power_series(-zeta_shifted, n_max, acc);

        if exp > 0. {
            let a = pre_factor * (series_1 * series_1_shifted - exp_vanishing * series_2 * series_2_shifted);
            (a, exp)
        } else {
            let a = pre_factor * (exp_vanishing * series_1 * series_1_shifted - series_2 * series_2_shifted);
            (a, -exp)
        }
    } else {
        let delta_x = delta / x;
        let zeta = 2. / 3. * (-x).sqrt() * (-x);
        let pow_series_1_4 = power_expansion_at_one(delta_x, 0.25, n_max, acc);
        let pow_series_3_2 = power_expansion_at_one(delta_x, 1.5, n_max, acc);
        let zeta_shifted = zeta * (1. + pow_series_3_2);

        let pre_factor = 1. / (PI * (-x).sqrt() * (1. + pow_series_1_4));
        let i_exp = pow_series_3_2 * zeta;

        let sin = f64::sin(i_exp);
        let cos = f64::cos(i_exp);
        let series_odd_shifted = airy_power_series_odd(zeta_shifted, n_max, acc);
        let series_odd = airy_power_series_odd(zeta, n_max, acc);
        let series_even_shifted = airy_power_series_even(zeta_shifted, n_max, acc);
        let series_even = airy_power_series_even(zeta, n_max, acc);

        let a = pre_factor * (
            -sin * (series_odd_shifted * series_odd + series_even * series_even_shifted)
            + cos * (series_odd_shifted * series_even - series_odd * series_even_shifted)
        );

        (a, 0.0)
    }
}

/// Calculates Y = [Ai'(x)*Bi(x+delta)-Ai(x+delta)*BI'(x)] using asymptotic expansion in x,
/// assumes |x| >> 1 and  |delta / x| << 1
/// returns pair of (A, B) of result A * exp(B).
fn airy_y(x: f64, delta: f64, n_max: usize, acc: f64) -> (f64, f64) {
    if x > 0.0 {
        let delta_x = delta / x;
        let zeta = 2. / 3. * x.sqrt().powi(3);
        let pow_series_1_4 = power_expansion_at_one(delta_x, 0.25, n_max, acc);
        let pow_series_3_2 = power_expansion_at_one(delta_x, 1.5, n_max, acc);
        let zeta_shifted = zeta * (1. + pow_series_3_2);

        let pre_factor = -1.0 / (TAU * (1. + pow_series_1_4));
        let exp = pow_series_3_2 * zeta;

        let series_1 = airy_deriv_power_series(-zeta, n_max, acc);
        let series_1_shifted = airy_power_series(zeta_shifted, n_max, acc);

        let exp_vanishing = f64::exp(-2. * exp.abs());
        let series_2 = airy_deriv_power_series(zeta, n_max, acc);
        let series_2_shifted = airy_power_series(-zeta_shifted, n_max, acc);

        if exp > 0. {
            let a = pre_factor * (series_1 * series_1_shifted + exp_vanishing * series_2 * series_2_shifted);
            (a, exp)
        } else {
            let a = pre_factor * (exp_vanishing * series_1 * series_1_shifted + series_2 * series_2_shifted);
            (a, -exp)
        }
    } else {
        let delta_x = delta / x;
        let zeta = 2. / 3. * (-x).sqrt().powi(3);
        let pow_series_1_4 = power_expansion_at_one(delta_x, 0.25, n_max, acc);
        let pow_series_3_2 = power_expansion_at_one(delta_x, 1.5, n_max, acc);
        let zeta_shifted = zeta * (1. + pow_series_3_2);

        let pre_factor = -1. / (PI * (1. + pow_series_1_4));
        let i_exp = pow_series_3_2 * zeta;

        let sin = f64::sin(i_exp);
        let cos = f64::cos(i_exp);
        let series_odd_shifted = airy_power_series_odd(zeta_shifted, n_max, acc);
        let series_odd = airy_deriv_power_series_odd(zeta, n_max, acc);
        let series_even_shifted = airy_power_series_even(zeta_shifted, n_max, acc);
        let series_even = airy_deriv_power_series_even(zeta, n_max, acc);

        let a = pre_factor * (
            cos * (series_odd_shifted * series_odd + series_even * series_even_shifted)
            + sin * (series_odd_shifted * series_even - series_odd * series_even_shifted)
        );

        (a, 0.0)
    }
}

/// calculates (1 + x)^k - 1 as a power series with up to `n_max` terms to numerical accuracy `acc`
fn power_expansion_at_one(x: f64, k: f64, n_max: usize, acc: f64) -> f64 {
    let mut series_n = 1.0;
    let rel_acc = (k * x).abs() * acc;

    let series = (1..=n_max)
        .map_while(|n| {
            let n_f = n as f64;
            series_n *= (k + 1. - n_f) / n_f * x;

            if series_n.abs() < rel_acc {
                None
            } else {
                Some(series_n)
            }
        })
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy expansion sum_k c_k zeta^-k, with up to `n_max` terms to numerical accuracy `acc`.
fn airy_power_series(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut term_n = 1.0;

    let series = [1.0]
        .into_iter()
        .chain((1..=n_max)
            .map_while(|n| {
                let n_f = n as f64;
                term_n *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                term_n /= zeta;
                if term_n.abs() < acc {
                    None
                } else {
                    Some(term_n)
                }
            })
        )
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy derivative expansion sum_k d_k zeta^-k, with up to `n_max` terms to numerical accuracy `acc`.
fn airy_deriv_power_series(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut term_n_c = 1.0;

    let series = [1.0]
        .into_iter()
        .chain((1..=n_max)
            .map_while(|n| {
                let n_f = n as f64;
                term_n_c *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                term_n_c /= zeta;
                let term_n = (1. + 6. * n_f) / (1. - 6. * n_f) * term_n_c;
                if term_n.abs() < acc {
                    None
                } else {
                    Some(term_n)
                }
            })
        )
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy expansion sum_k (-1)^k c_(2k+1) zeta^-(2k+1), with up to `n_max` terms to numerical accuracy `acc`.
fn airy_power_series_odd(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut index_2n_1 = 1;
    let mut term_2n_1 = 3.75 / 54. / zeta;
    let rel_acc = term_2n_1.abs() * acc;

    let series = (0..=n_max)
            .map_while(|n| {
                while index_2n_1 < 2 * n + 1 {
                    index_2n_1 += 1;
                    let n_f = index_2n_1 as f64;
                    term_2n_1 *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                    term_2n_1 /= zeta;
                }

                if term_2n_1.abs() < rel_acc {
                    None
                } else {
                    Some((-1i32).pow(n as u32) as f64 * term_2n_1)
                }
            })
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy derivative expansion sum_k (-1)^k d_(2k+1) zeta^-(2k+1), with up to `n_max` terms to numerical accuracy `acc`.
fn airy_deriv_power_series_odd(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut index_2n_1 = 1;
    let mut term_2n_1_c = 3.75 / 54. / zeta;
    let rel_acc = 7. / 5. * term_2n_1_c.abs() * acc;

    let series = (0..=n_max)
            .map_while(|n| {
                while index_2n_1 < 2 * n + 1 {
                    index_2n_1 += 1;
                    let n_f = index_2n_1 as f64;
                    term_2n_1_c *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                    term_2n_1_c /= zeta;
                }
                let n_f = index_2n_1 as f64;
                let term_2n_1 = (1. + 6. * n_f) / (1. - 6. * n_f) * term_2n_1_c;

                if term_2n_1.abs() < rel_acc {
                    None
                } else {
                    Some((-1i32).pow(n as u32) as f64 * term_2n_1)
                }
            })
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy expansion sum_k (-1)^k c_2k zeta^-2k, with up to `n_max` terms to numerical accuracy `acc`.
fn airy_power_series_even(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut index_2n = 0;
    let mut term_2n = 1.;

    let series = (0..=n_max)
            .map_while(|n| {
                while index_2n < 2 * n {
                    index_2n += 1;
                    let n_f = index_2n as f64;
                    term_2n *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                    term_2n /= zeta;
                }

                if term_2n.abs() < acc {
                    None
                } else {
                    Some((-1i32).pow(n as u32) as f64 * term_2n)
                }
            })
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

/// calculates power series of the airy derivative expansion sum_k (-1)^k d_2k zeta^-2k, with up to `n_max` terms to numerical accuracy `acc`.
fn airy_deriv_power_series_even(zeta: f64, n_max: usize, acc: f64) -> f64 {
    let mut index_2n = 0;
    let mut term_2n_c = 1.;

    let series = (0..=n_max)
            .map_while(|n| {
                while index_2n < 2 * n {
                    index_2n += 1;
                    let n_f = index_2n as f64;
                    term_2n_c *= 3. / 4. * (6. - 5. / n_f) * ((6. * n_f - 1.) / 54.);
                    term_2n_c /= zeta;
                }
                let n_f = index_2n as f64;
                let term_2n = (1. + 6. * n_f) / (1. - 6. * n_f) * term_2n_c;

                if term_2n.abs() < acc {
                    None
                } else {
                    Some((-1i32).pow(n as u32) as f64 * term_2n)
                }
            })
        .collect::<Vec<f64>>();

    if series.len() == n_max {
        println!(
            "Airy power series expansion possibly inaccurate, last term in power series value: {}, wanted accuracy: {acc}", 
            series.last().unwrap()
        )
    }

    // for numerical stability sum from smallest to largest values
    series
        .iter()
        .rev()
        .sum()
}

#[cfg(test)]
mod tests {
    use cc_math_utils::assert_approx_eq;
    use super::*;

    #[test]
    pub fn test_airy_x() {
        let x = 1000.;
        let delta = 0.1;
        let airy_x_1 = airy_x(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), 0.118693, 1e-3);

        let x = 100.;
        let delta = -0.1;
        let airy_x_1 = airy_x(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), -0.0374049, 1e-3);

        let x = -200.;
        let delta = -0.05;
        let airy_x_1 = airy_x(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), -0.0146218, 1e-3);

        let x = -2000.;
        let delta = 0.2;
        let airy_x_1 = airy_x(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), 0.00329146, 1e-3);
    }

    #[test]
    pub fn test_airy_y() {
        let x = 1000.;
        let delta = 0.1;
        let airy_y_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_y_1.0 * f64::exp(airy_y_1.1), -3.76690, 1e-6);

        let x = 100.;
        let delta = -0.1;
        let airy_y_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_y_1.0 * f64::exp(airy_y_1.1), -0.491242, 1e-3);

        let x = -200.;
        let delta = -0.05;
        let airy_y_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_y_1.0 * f64::exp(airy_y_1.1), -0.241987, 1e-3);

        let x = -2000.;
        let delta = 0.2;
        let airy_y_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_y_1.0 * f64::exp(airy_y_1.1), 0.282239, 1e-3);
    }
}
