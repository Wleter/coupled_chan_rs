use std::f64::consts::PI;

use cc_math_utils::bessel::{
    ratio_riccati_i,
    ratio_riccati_k,
    riccati_i_log_deriv,
    riccati_j,
    riccati_j_deriv,
    riccati_k_log_deriv,
    riccati_n,
    riccati_n_deriv,
};
use cc_propagator::{
    LogDeriv,
    Ratio,
    Solution,
    multi_channel::{
        Matrix,
        WMatrix,
    },
};
use faer::{
    Mat,
    MatRef,
    c64,
    linalg::solvers::DenseSolveCore,
};
use num_complex::Complex64;

use crate::coupling::{
    CollisionWMatrix,
    RCoupling,
};

#[derive(Debug, Clone)]
pub struct SMatrix {
    s_matrix: Mat<c64>,
    momenta: Vec<f64>,
    entrance: usize,
}

impl SMatrix {
    pub fn new(s_matrix: Mat<c64>, momenta: Vec<f64>, entrance: usize) -> Self {
        assert_eq!(s_matrix.nrows(), s_matrix.ncols(), "S-Matrix not a square matrix");
        assert_eq!(
            s_matrix.nrows(),
            momenta.len(),
            "S-Matrix and number of open channel momenta should be the same"
        );

        Self {
            s_matrix,
            momenta,
            entrance,
        }
    }

    pub fn from_log_deriv(sol: &Solution<LogDeriv<Matrix>>, w_matrix: &CollisionWMatrix<impl RCoupling>) -> SMatrix {
        let size = w_matrix.size();
        let r = sol.r;

        let asymptote = &w_matrix.asymptote();
        let levels = asymptote.levels();

        let is_open_channel = levels
            .asymptote
            .iter()
            .map(|&val| val < asymptote.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = levels
            .asymptote
            .iter()
            .map(|&val| (2.0 * asymptote.system_params().mass * (asymptote.energy - val).abs()).sqrt())
            .collect();

        let log_deriv = if let Some(transformation) = asymptote.transformation() {
            crate::transform_rev(&sol.sol.0, transformation)
        } else {
            sol.sol.0.clone()
        };

        let mut j_last = Mat::zeros(size, size);
        let mut j_deriv_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_deriv_last = Mat::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            let l = levels.l[i];
            if is_open_channel[i] {
                let (j_riccati, j_deriv_riccati) = riccati_j_deriv(l, momentum * r);
                let (n_riccati, n_deriv_riccati) = riccati_n_deriv(l, momentum * r);

                j_last[(i, i)] = j_riccati / momentum.sqrt();
                j_deriv_last[(i, i)] = j_deriv_riccati * momentum.sqrt();
                n_last[(i, i)] = n_riccati / momentum.sqrt();
                n_deriv_last[(i, i)] = n_deriv_riccati * momentum.sqrt();
            } else {
                let ratio_i = riccati_i_log_deriv(l, momentum * r);
                let ratio_k = riccati_k_log_deriv(l, momentum * r);

                j_deriv_last[(i, i)] = ratio_i * momentum;
                j_last[(i, i)] = 1.0;
                n_deriv_last[(i, i)] = ratio_k * momentum;
                n_last[(i, i)] = 1.0;
            }
        }

        let denominator = (&log_deriv * n_last - n_deriv_last).partial_piv_lu();
        let denominator = denominator.inverse();

        let k_matrix = -denominator * (&log_deriv * j_last - j_deriv_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while !is_open_channel[i_full] {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while !is_open_channel[j_full] {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = c64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = Mat::<c64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).partial_piv_lu();
        let denominator = denominator.inverse();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == asymptote.system_params().entrance)
            .expect("Closed entrance channel")
            .0;

        SMatrix::new(s_matrix, momenta, entrance)
    }

    pub fn from_ratio(sol: &Solution<Ratio<Matrix>>, w_matrix: &CollisionWMatrix<impl RCoupling>) -> SMatrix {
        let size = w_matrix.size();
        let r_last = sol.r;
        let r_prev_last = sol.r - sol.dr;

        let mut f_last = Matrix::zeros(size, size);
        w_matrix.value_inplace(r_last, &mut f_last);
        f_last *= sol.dr * sol.dr / 12.;
        f_last += w_matrix.id();

        let mut f_prev_last = Matrix::zeros(size, size);
        w_matrix.value_inplace(r_prev_last, &mut f_prev_last);
        f_prev_last *= sol.dr * sol.dr / 12.;
        f_prev_last += w_matrix.id();

        let wave_ratio = f_last.partial_piv_lu().inverse() * sol.sol.0.as_ref() * f_prev_last;

        let asymptote = &w_matrix.asymptote();
        let levels = asymptote.levels();

        let wave_ratio = if let Some(transformation) = asymptote.transformation() {
            crate::transform_rev(&wave_ratio, transformation)
        } else {
            wave_ratio
        };

        let is_open_channel = levels
            .asymptote
            .iter()
            .map(|&val| val < asymptote.energy)
            .collect::<Vec<bool>>();
        let momenta: Vec<f64> = levels
            .asymptote
            .iter()
            .map(|&val| (2.0 * asymptote.system_params().mass * (asymptote.energy - val).abs()).sqrt())
            .collect();

        let mut j_last = Mat::zeros(size, size);
        let mut j_prev_last = Mat::zeros(size, size);
        let mut n_last = Mat::zeros(size, size);
        let mut n_prev_last = Mat::zeros(size, size);

        for i in 0..size {
            let momentum = momenta[i];
            let l = levels.l[i];
            if is_open_channel[i] {
                j_last[(i, i)] = riccati_j(l, momentum * r_last) / momentum.sqrt();
                j_prev_last[(i, i)] = riccati_j(l, momentum * r_prev_last) / momentum.sqrt();
                n_last[(i, i)] = riccati_n(l, momentum * r_last) / momentum.sqrt();
                n_prev_last[(i, i)] = riccati_n(l, momentum * r_prev_last) / momentum.sqrt();
            } else {
                j_last[(i, i)] = ratio_riccati_i(l, momentum * r_last, momentum * r_prev_last);
                j_prev_last[(i, i)] = 1.0;
                n_last[(i, i)] = ratio_riccati_k(l, momentum * r_last, momentum * r_prev_last);
                n_prev_last[(i, i)] = 1.0;
            }
        }

        let denominator = (&wave_ratio * n_prev_last - n_last).partial_piv_lu();
        let denominator = denominator.inverse();

        let k_matrix = -denominator * (wave_ratio * j_prev_last - j_last);

        let open_channel_count = is_open_channel.iter().filter(|val| **val).count();
        let mut red_ik_matrix = Mat::<c64>::zeros(open_channel_count, open_channel_count);

        let mut i_full = 0;
        for i in 0..open_channel_count {
            while !is_open_channel[i_full] {
                i_full += 1
            }

            let mut j_full = 0;
            for j in 0..open_channel_count {
                while !is_open_channel[j_full] {
                    j_full += 1
                }

                red_ik_matrix[(i, j)] = c64::new(0.0, k_matrix[(i_full, j_full)]);
                j_full += 1;
            }
            i_full += 1;
        }
        let id = Mat::<c64>::identity(open_channel_count, open_channel_count);

        let denominator = (&id - &red_ik_matrix).partial_piv_lu();
        let denominator = denominator.inverse();
        let s_matrix = denominator * (id + red_ik_matrix);
        let entrance = is_open_channel
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .find(|(i, _)| *i == asymptote.system_params().entrance)
            .expect("Closed entrance channel")
            .0;

        SMatrix::new(s_matrix, momenta, entrance)
    }

    pub fn s_matrix(&self) -> MatRef<'_, c64> {
        self.s_matrix.as_ref()
    }

    pub fn entrance_number(&self) -> usize {
        self.entrance
    }

    pub fn entrance_momentum(&self) -> f64 {
        self.momenta[self.entrance]
    }

    pub fn momenta(&self) -> &[f64] {
        &self.momenta
    }

    pub fn scattering_length(&self) -> Complex64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        1.0 / Complex64::new(0.0, self.momenta[self.entrance]) * (1.0 - s_element) / (1.0 + s_element)
    }

    pub fn elastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element).norm_sqr()
    }

    pub fn inelastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element.norm_sqr())
    }

    pub fn inelastic_cross_sect_to(&self, channel: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, channel)];

        PI / self.momenta[self.entrance].powi(2) * s_element.norm_sqr()
    }

    pub fn cross_sect(&self, in_chan: usize, out_chan: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(in_chan, out_chan)];

        if in_chan == out_chan {
            PI / self.momenta[in_chan].powi(2) * (1. - s_element).norm_sqr()
        } else {
            PI / self.momenta[in_chan].powi(2) * s_element.norm_sqr()
        }
    }

    pub fn scattering_length_in(&self, channel: usize) -> Complex64 {
        let s_element: Complex64 = self.s_matrix[(channel, channel)];

        1.0 / Complex64::new(0.0, self.momenta[channel]) * (1.0 - s_element) / (1.0 + s_element)
    }

    /// Returns cross sections from entrance channel to every open channel
    pub fn cross_sects_entrance(&self) -> Vec<f64> {
        (0..self.momenta.len()).map(|c| self.cross_sect(self.entrance, c)).collect()
    }

    /// Returns scattering lengths of all open channels.
    pub fn scattering_lengths(&self) -> Vec<Complex64> {
        (0..self.momenta.len()).map(|c| self.scattering_length_in(c)).collect()
    }

    /// Returns matrix o_ij with elements being cross
    /// sections from i-th channel to j-th channel.
    pub fn cross_sects(&self) -> Mat<f64> {
        Mat::from_fn(self.momenta.len(), self.momenta.len(), |c, r| self.cross_sect(c, r))
    }
}
