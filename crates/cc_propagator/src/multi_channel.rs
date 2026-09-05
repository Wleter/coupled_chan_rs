pub mod log_derivative;
pub mod numerov;

use std::f64::consts::PI;

use faer::{
    ColRef,
    Mat,
    linalg::solvers::DenseSolveCore,
};

use crate::WaveStorage;

// todo! make everything generic on matrix backend
pub type Matrix = Mat<f64>;

/// Trait for representing W matrix
/// in an equation y''(x) = W(x) y(x)
pub trait WMatrix<E> {
    fn size(&self) -> usize;
    fn value_inplace(&self, r: f64, value: &mut Mat<E>);
}

#[inline]
fn local_wavelength(red_coupling: &Mat<f64>) -> f64 {
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

        values.push(wave_init.iter().copied().collect());
        let rs = if last_first {
            for c in self.connections.iter().skip(1).rev() {
                wave_recent = c * wave_recent;
                values.push(wave_recent.iter().copied().collect())
            }

            self.rs.iter().rev().copied().collect()
        } else {
            for c in self.connections.iter().take(self.connections.len() - 1) {
                // todo! can make it faster if it is slow
                wave_recent = c.partial_piv_lu().inverse() * wave_recent;
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
    use faer::{
        col,
        mat,
    };

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
        let k1 = 5.;
        let k2 = 9.;

        let k2_mid = (k1 * k1 + k2 * k2) / 2.;
        let k2_diff = (k2 * k2 - k1 * k1) / 2.;
        let w_matrix = SingleValue(mat![[k2_mid, k2_diff], [k2_diff, k2_mid]]);

        let u = mat![[1., 1.], [-1., 1.]] / f64::sqrt(2.);
        let start = &u * mat![[1.], [0.]].col(0);

        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1., 0.], [0., 1.]],
            derivative: mat![[0., 0.], [0., 0.]],
        };

        let analytic = |x: f64, dx: f64| {
            &u * mat![
                [f64::cos(k1 * (x)) / f64::cos(k1 * (x - dx)), 0.],
                [0., f64::cos(k2 * (x)) / f64::cos(k2 * (x - dx))],
            ] * u.transpose()
        };
        let analytic_val = |x: f64| (&u * col![f64::cos(k1 * x), 0.,]).iter().copied().collect::<Vec<f64>>();

        let mut numerov = RatioNumerov::new(&w_matrix, SingleStep::new(1e-4), boundary);
        numerov.init_wave_storage();

        let sol = numerov.propagate_to(1. / 3. * PI / k1);
        assert_approx_eq!(analytic(sol.r, sol.dr)[(0, 0)], sol.sol.0[(0, 0)], 1e-4);
        assert_approx_eq!(analytic(sol.r, sol.dr)[(1, 1)], sol.sol.0[(1, 1)], 1e-4);

        let (rs, wave) = numerov.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-3);

        let sol = numerov.propagate_to(2. / 3. * PI / k1);
        assert_approx_eq!(analytic(sol.r, sol.dr)[(0, 0)], sol.sol.0[(0, 0)], 1e-4);
        assert_approx_eq!(analytic(sol.r, sol.dr)[(1, 1)], sol.sol.0[(1, 1)], 1e-4);
        let (rs, wave) = numerov.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-3);
    }

    #[test]
    pub fn test_multi_chan_log_derivative_johnson() {
        let k1 = 5.;
        let k2 = 9.;

        let k2_mid = (k1 * k1 + k2 * k2) / 2.;
        let k2_diff = (k2 * k2 - k1 * k1) / 2.;
        let w_matrix = SingleValue(mat![[k2_mid, k2_diff], [k2_diff, k2_mid]]);

        let u = mat![[1., 1.], [-1., 1.]] / f64::sqrt(2.);
        let start = &u * mat![[1.], [0.]].col(0);

        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1., 0.], [0., 1.]],
            derivative: mat![[0., 0.], [0., 0.]],
        };

        let analytic = |x: f64| &u * mat![[-k1 * f64::tan(k1 * x), 0.], [0., -k2 * f64::tan(k2 * x)],] * u.transpose();
        let analytic_val = |x: f64| (&u * col![f64::cos(k1 * x), 0.,]).iter().copied().collect::<Vec<f64>>();

        let mut log_deriv = JohnsonLogDerivative::new(&w_matrix, SingleStep::new(1e-3), boundary);
        log_deriv.init_wave_storage();

        let sol = log_deriv.propagate_to(1. / 3. * PI / k1);
        assert_approx_eq!(mat => analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);

        let sol = log_deriv.propagate_to(2. / 3. * PI / k1);
        assert_approx_eq!(mat => analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);
    }

    #[test]
    pub fn test_multi_chan_log_derivative_manolopoulos() {
        let k1 = 5.;
        let k2 = 9.;

        let k2_mid = (k1 * k1 + k2 * k2) / 2.;
        let k2_diff = (k2 * k2 - k1 * k1) / 2.;
        let w_matrix = SingleValue(mat![[k2_mid, k2_diff], [k2_diff, k2_mid]]);

        let u = mat![[1., 1.], [-1., 1.]] / f64::sqrt(2.);
        let start = &u * mat![[1.], [0.]].col(0);

        let boundary = Boundary {
            r_start: 0.,
            direction: Direction::Outwards,
            value: mat![[1., 0.], [0., 1.]],
            derivative: mat![[0., 0.], [0., 0.]],
        };

        let analytic = |x: f64| &u * mat![[-k1 * f64::tan(k1 * x), 0.], [0., -k2 * f64::tan(k2 * x)],] * u.transpose();
        let analytic_val = |x: f64| (&u * col![f64::cos(k1 * x), 0.,]).iter().copied().collect::<Vec<f64>>();

        let mut log_deriv = ManolopoulosLogDerivative::new(&w_matrix, SingleStep::new(1e-3), boundary);
        log_deriv.init_wave_storage();

        let sol = log_deriv.propagate_to(1. / 3. * PI / k1);
        assert_approx_eq!(mat => analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);

        let sol = log_deriv.propagate_to(2. / 3. * PI / k1);
        assert_approx_eq!(mat => analytic(sol.r), sol.sol.0, 1e-4);
        let (rs, wave) = log_deriv.get_wave_storage().unwrap().reconstruct(start.as_ref(), false);
        assert_approx_eq!(iter => wave.last().unwrap(), analytic_val(*rs.last().unwrap()), 1e-4);
    }
}
