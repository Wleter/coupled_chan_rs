// Log-derivative imbedding airy-function approximation based on:
// A renormalized potential-following propagation algorithm for solving the coupled-
// channels equations. doi: 10.1063/1.4891809, Appendix A2 equations.

use std::f64::consts::{PI, TAU};

/// Calculates X = [Ai(x)*Bi(x+delta)-Ai(x+delta)*BI(x)] using asymptotic expansion in x,
/// assumes |x| >> 1 and  |delta / x| << 1
/// returns pair (A, B) of result A * exp(B).
pub fn airy_x(x: f64, delta: f64, n_max: usize, acc: f64) -> (f64, f64) {
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
pub fn airy_y(x: f64, delta: f64, n_max: usize, acc: f64) -> (f64, f64) {
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
        let airy_x_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), -0.241987, 1e-3);

        let x = -2000.;
        let delta = 0.2;
        let airy_x_1 = airy_y(x, delta, 100, 1e-6);
        assert_approx_eq!(airy_x_1.0 * f64::exp(airy_x_1.1), 0.282239, 1e-3);
    }
}
