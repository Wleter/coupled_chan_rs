use faer::{Mat, c64, linalg::solvers::DenseSolveCore as _};
use math_utils::bessel::{ratio_riccati_i_deriv, ratio_riccati_k_deriv, riccati_j_deriv, riccati_n_deriv};
use propagator::{LogDeriv, Solution};

use crate::{Operator, coupling::WMatrix, s_matrix::SMatrix};

pub mod diabatic;

pub fn get_s_matrix<W: WMatrix>(sol: &Solution<LogDeriv<Operator>>, w_matrix: &W) -> SMatrix {
    let size = w_matrix.size();
    let r = sol.r;
    let log_deriv = sol.sol.0.as_ref();

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
        .map(|&val| (2.0 * asymptote.red_mass * (asymptote.energy - val).abs()).sqrt())
        .collect();

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
            let ratio_i = ratio_riccati_i_deriv(l, momentum * r);
            let ratio_k = ratio_riccati_k_deriv(l, momentum * r);

            j_deriv_last[(i, i)] = ratio_i * momentum;
            j_last[(i, i)] = 1.0;
            n_deriv_last[(i, i)] = ratio_k * momentum;
            n_last[(i, i)] = 1.0;
        }
    }

    let denominator = (log_deriv * n_last - n_deriv_last).partial_piv_lu();
    let denominator = denominator.inverse();

    let k_matrix = -denominator * (log_deriv * j_last - j_deriv_last);

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
        .find(|(i, _)| *i == asymptote.entrance_level)
        .expect("Closed entrance channel")
        .0;

    SMatrix::new(s_matrix, momenta[asymptote.entrance_level], entrance)
}

