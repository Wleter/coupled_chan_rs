use crate::{
    LogDeriv,
    Propagator,
    WaveStorage,
    WithConnectionWatcher,
    WithNodeCount,
};

pub mod airy;
pub mod log_derivative;
pub mod numerov;

pub trait SingleChanLogDerivPropagator<'a>:
    Propagator<LogDeriv<f64>> + WithNodeCount + WithConnectionWatcher<'a, f64>
{
}
impl<'a, T: Propagator<LogDeriv<f64>> + WithNodeCount + WithConnectionWatcher<'a, f64>> SingleChanLogDerivPropagator<'a>
    for T
{
}

/// Trait for representing W function
/// in an equation y''(x) = W(x) y(x)
pub trait WFunction {
    fn value(&self, r: f64) -> f64;
}

#[inline]
fn local_wavelength(w_function: f64) -> f64 {
    2. * std::f64::consts::PI / w_function.abs().sqrt()
}

impl WaveStorage<f64> {
    pub fn reconstruct(&self, wave_init: f64, last_first: bool) -> (Vec<f64>, Vec<f64>) {
        let mut values = Vec::with_capacity(self.connections.len());
        let mut wave_recent = wave_init;

        values.push(wave_recent);
        let rs = if last_first {
            for c in self.connections.iter().skip(1).rev() {
                wave_recent *= c;
                values.push(wave_recent)
            }

            self.rs.iter().rev().copied().collect()
        } else {
            for c in self.connections.iter().take(self.connections.len() - 1) {
                wave_recent /= c;
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

    use crate::single_channel::airy::AiryLogDerivative;
    use crate::single_channel::log_derivative::LogDerivative;
    use crate::single_channel::numerov::RatioNumerov;
    use crate::step_strategy::{
        ConstRefStepStrat,
        LinearRefStepStrat,
    };
    use crate::{
        Boundary,
        Direction,
        Propagator,
        WithConnectionWatcher,
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

        let analytic = |x: f64, dx: f64| f64::cos(k * (x)) / f64::cos(k * (x - dx));
        let analytic_val = |x: f64| f64::cos(k * x);
        let mut numerov = RatioNumerov::new(&w_function, ConstRefStepStrat::Fixed { dr: 1e-4 }, boundary.clone());
        let mut storage = WaveStorage::default();
        numerov.insert_watcher(&mut storage);

        let sol = numerov.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r, sol.dr), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-3);

        let mut numerov = RatioNumerov::new(&w_function, ConstRefStepStrat::Fixed { dr: 1e-4 }, boundary);
        let mut storage = WaveStorage::default();
        numerov.insert_watcher(&mut storage);

        let sol = numerov.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r, sol.dr), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
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

        let analytic = |x: f64| -k * f64::tan(k * x);
        let analytic_val = |x: f64| f64::cos(k * x);
        let mut log_deriv = LogDerivative::new(&w_function, ConstRefStepStrat::Fixed { dr: 1e-3 }, boundary.clone());
        let mut storage = WaveStorage::default();
        log_deriv.insert_watcher(&mut storage);

        let sol = log_deriv.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);

        let mut log_deriv = LogDerivative::new(&w_function, ConstRefStepStrat::Fixed { dr: 1e-3 }, boundary);
        let mut storage = WaveStorage::default();
        log_deriv.insert_watcher(&mut storage);
        let sol = log_deriv.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);
    }

    #[test]
    pub fn test_single_chan_airy_log_derivative() {
        let k = 5.;

        let w_function = SingleValue(k * k);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: 1.,
            derivative: 0.,
        };

        let analytic = |x: f64| -k * f64::tan(k * x);
        let analytic_val = |x: f64| f64::cos(k * x);
        let mut log_deriv = AiryLogDerivative::new(&w_function, LinearRefStepStrat::Fixed { dr: 1e-3 }, boundary.clone());
        let mut storage = WaveStorage::default();
        log_deriv.insert_watcher(&mut storage);

        let sol = log_deriv.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);

        let mut log_deriv = LogDerivative::new(&w_function, ConstRefStepStrat::Fixed { dr: 1e-3 }, boundary);
        let mut storage = WaveStorage::default();
        log_deriv.insert_watcher(&mut storage);
        let sol = log_deriv.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = storage.reconstruct(1., false);
        assert_approx_eq!(*wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);
    }
}
