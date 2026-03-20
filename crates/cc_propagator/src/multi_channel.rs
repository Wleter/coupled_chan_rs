pub mod log_derivative;
pub mod numerov;

use std::f64::consts::PI;

use faer::{
    ColRef,
    Mat,
};

use crate::WaveStorage;

// todo! make everything generic on matrix backend
pub type Matrix = Mat<f64>;

/// Trait for representing W matrix
/// in an equation y''(x) + W(x) y(x) = 0
pub trait WMatrix<E> {
    fn size(&self) -> usize;
    fn value_inplace(&self, r: f64, value: &mut Mat<E>);
}

#[inline]
pub(self) fn local_wavelength(red_coupling: &Mat<f64>) -> f64 {
    let max_g_val = red_coupling
        .diagonal()
        .column_vector()
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    2. * PI / max_g_val.abs().sqrt()
}

impl WaveStorage<Mat<f64>> {
    pub fn reconstruct(&self, wave_init: ColRef<f64>, last_first: bool) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut values = Vec::with_capacity(self.connections.len());
        let mut wave_recent = wave_init.cloned();

        let rs = if last_first {
            for c in self.connections.iter().rev() {
                wave_recent = c * &wave_recent;
                values.push(wave_recent.iter().copied().collect())
            }

            self.rs.iter().rev().copied().collect()
        } else {
            for c in self.connections.iter() {
                wave_recent = c * &wave_recent;
                values.push(wave_recent.iter().copied().collect())
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
    use faer::mat;

    use crate::{
        Boundary,
        Direction,
        Propagator,
        WithWaveStorage,
        multi_channel::{
            log_derivative::{
                JohnsonLogDerivative,
                ManolopoulosLogDerivative,
            },
            numerov::RatioNumerov,
        },
        step_strategy::SingleStep,
    };

    use super::*;

    struct SingleValue(Matrix);

    impl WMatrix<f64> for SingleValue {
        fn size(&self) -> usize {
            assert_eq!(
                self.0.nrows(),
                self.0.ncols(),
                "mismatch of number of rows and cols in WMatrix"
            );

            self.0.nrows()
        }

        fn value_inplace(&self, _r: f64, value: &mut Matrix) {
            value.copy_from(&self.0);
        }
    }

    #[test]
    pub fn test_multi_chan_numerov() {
        let k = 5.;

        let w_function = SingleValue(mat![[k * k]]);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1.]],
            derivative: mat![[0.]],
        };

        let mut numerov = RatioNumerov::new(&w_function, SingleStep::new(1e-3), boundary);
        let analytic = |x: f64, dx: f64| mat![[f64::cos(k * (x)) / f64::cos(k * (x - dx))]];

        let sol = numerov.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);

        let sol = numerov.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);
    }

    #[test]
    pub fn test_multi_chan_log_derivative_johnson() {
        let k = 5.;

        let w_function = SingleValue(mat![[k * k]]);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1.]],
            derivative: mat![[0.]],
        };

        let mut log_deriv = JohnsonLogDerivative::new(&w_function, SingleStep::new(1e-3), boundary);
        log_deriv.init_wave_storage();
        let analytic = |x: f64, dx: f64| mat![[f64::cos(k * (x)) / f64::cos(k * (x - dx))]];

        let sol = log_deriv.propagate_to(1. / 3. * PI / k);
        assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);

        let sol = log_deriv.propagate_to(2. / 3. * PI / k);
        assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);

        if let Some(w) = &log_deriv.get_wave_storage() {
            let (rs, values) = w.reconstruct(mat![[1.]].col(0), false);

            println!("{:?} {:?}", rs, values);
        }
    }

    #[test]
    pub fn test_multi_chan_log_derivative_manolopoulos() {
        let k = 5.;

        let w_function = SingleValue(mat![[k * k]]);
        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1.]],
            derivative: mat![[0.]],
        };

        let mut log_deriv = ManolopoulosLogDerivative::new(&w_function, SingleStep::new(1e-3), boundary);
        log_deriv.init_wave_storage();
        let analytic = |x: f64, dx: f64| mat![[f64::cos(k * (x)) / f64::cos(k * (x - dx))]];

        let sol = log_deriv.propagate_to(1. / 3. * PI / k);
        // assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);

        let sol = log_deriv.propagate_to(2. / 3. * PI / k);
        // assert_approx_eq!(mat => analytic(sol.r, sol.dr), sol.sol.0, 1e-4);

        if let Some(w) = &log_deriv.get_wave_storage() {
            let (rs, values) = w.reconstruct(mat![[1.]].col(0), false);

            println!("{:?} {:?}", rs, values);
        }
    }
}
