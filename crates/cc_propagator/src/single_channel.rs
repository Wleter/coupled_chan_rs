use crate::WaveStorage;

pub mod log_derivative;
pub mod numerov;

/// Trait for representing W function
/// in an equation y''(x) + W(x) y(x) = 0
pub trait WFunction {
    fn value(&self, r: f64) -> f64;
}

#[inline]
pub(self) fn local_wavelength(w_function: f64) -> f64 {
    2. * std::f64::consts::PI / w_function.abs().sqrt()
}

impl WaveStorage<f64> {
    pub fn reconstruct(&self, wave_init: f64, last_first: bool) -> (Vec<f64>, Vec<f64>) {
        let mut values = Vec::with_capacity(self.connections.len());
        let mut wave_recent = wave_init;

        values.push(wave_recent);
        let rs = if last_first {
            for c in self.connections.iter().skip(1).rev() {
                wave_recent = c * wave_recent;
                values.push(wave_recent)
            }

            self.rs.iter().rev().copied().collect()
        } else {
            for c in self.connections.iter().take(self.connections.len() - 1) {
                wave_recent = 1. / c * wave_recent;
                values.push(wave_recent)
            }
            self.rs.clone()
        };

        (rs, values)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use cc_math_utils::assert_approx_eq;

    use crate::{
        Boundary,
        Direction,
        Propagator,
        WithWaveStorage,
        single_channel::{
            log_derivative::LogDerivative,
            numerov::RatioNumerov,
        },
        step_strategy::SingleStep,
    };

    use super::*;

    struct SingleValue(f64);

    impl WFunction for SingleValue {
        fn value(&self, _r: f64) -> f64 {
            self.0
        }
    }

    #[test]
    pub fn test_single_chan_numerov() {
        let k = 5.;

        let w_function = SingleValue(k * k);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: 1.,
            derivative: 0.,
        };

        let mut numerov = RatioNumerov::new(&w_function, SingleStep::new(1e-4), boundary);
        numerov.init_wave_storage();
        let analytic = |x: f64, dx: f64| f64::cos(k * (x)) / f64::cos(k * (x - dx));
        let analytic_val = |x: f64| f64::cos(k * x);

        let sol = numerov.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r, sol.dr), sol.sol.0, 1e-4);
        let (rs, wave) = numerov.get_wave_storage().unwrap().reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-3);

        let sol = numerov.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r, sol.dr), sol.sol.0, 1e-4);
        let (rs, wave) = numerov.get_wave_storage().unwrap().reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-3);
    }

    #[test]
    pub fn test_single_chan_log_derivative() {
        let k = 5.;

        let w_function = SingleValue(k * k);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: 1.,
            derivative: 0.,
        };

        let mut log_deriv = LogDerivative::new(&w_function, SingleStep::new(1e-3), boundary);
        log_deriv.init_wave_storage();
        let analytic = |x: f64| -k * f64::tan(k * x);
        let analytic_val = |x: f64| f64::cos(k * x);

        let sol = log_deriv.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);

        let sol = log_deriv.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);
    }
}
