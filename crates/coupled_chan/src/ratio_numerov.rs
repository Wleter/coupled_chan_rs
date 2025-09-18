use std::{f64::consts::PI, mem::swap};

use faer::{
    Accum, Mat, Par, c64,
    dyn_stack::MemBuffer,
    linalg::{matmul::matmul, solvers::DenseSolveCore},
    unzip, zip,
};
use math_utils::bessel::{ratio_riccati_i, ratio_riccati_k, riccati_j, riccati_n};
use matrix_utils::faer::{get_ldlt_inverse_buffer, inverse_ldlt_inplace};
use propagator::{
    Boundary, Direction, Propagator, Ratio, Solution, propagator_watcher::PropagatorWatcher, step_strategy::StepStrategy,
};

use crate::{Operator, coupling::WMatrix, s_matrix::SMatrix};

// todo! look whether V(r) is evaluated at correct values

/// 10.1063/1.436421
pub struct RatioNumerov<'a, W: WMatrix> {
    w_matrix: &'a W,
    step: StepStrategy,

    solution: Solution<Ratio<Operator>>,

    w_matrix_buffer: Operator,
    watchers: Option<Vec<&'a mut dyn PropagatorWatcher<Ratio<Operator>>>>,

    prev_sol: Ratio<Operator>,

    f: Mat<f64>,
    f_last: Mat<f64>,
    f_prev_last: Mat<f64>,

    buffer1: Operator,
    buffer2: Operator,
    buffer3: Operator,
    inverse_buffer: MemBuffer,
}

impl<'a, W: WMatrix> RatioNumerov<'a, W> {
    pub fn new(w_matrix: &'a W, step: StepStrategy, boundary: Boundary<Operator>) -> Self {
        let size = w_matrix.size();
        let r = boundary.r_start;

        let mut red_coupling_buffer = Operator::zeros(size);

        w_matrix.value_inplace(r, &mut red_coupling_buffer);
        let local_wavelength = get_wavelength(&red_coupling_buffer);

        let dr = match boundary.direction {
            Direction::Inwards => -(step.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step.get_step(r, local_wavelength).abs(),
        };

        let mut f_last = Operator::zeros(size);
        w_matrix.value_inplace(r - dr, &mut f_last);

        let mut f_prev_last = Operator::zeros(size);
        w_matrix.value_inplace(r - 2. * dr, &mut f_prev_last);

        let f_last = w_matrix.id().as_ref() + dr * dr / 12. * f_last.0;
        let f_prev_last = w_matrix.id().as_ref() + dr * dr / 12. * f_prev_last.0;
        let f = w_matrix.id().as_ref() + dr * dr / 12. * &red_coupling_buffer.0;

        let sol = Ratio(Operator::new(
            &f * (&boundary.derivative.0 * dr + &boundary.value.0)
                * boundary.value.0.partial_piv_lu().inverse()
                * f_last.partial_piv_lu().inverse(),
        ));

        let prev_sol = Ratio(Operator::new(
            &f_last
                * &boundary.value.0
                * (&boundary.value.0 - &boundary.derivative.0 * dr).partial_piv_lu().inverse()
                * f_prev_last.partial_piv_lu().inverse(),
        ));

        Self {
            w_matrix,
            step,
            solution: Solution { r, dr, sol },
            f,
            f_last,
            f_prev_last,
            prev_sol,

            w_matrix_buffer: red_coupling_buffer,
            watchers: None,

            buffer1: Operator::zeros(size),
            buffer2: Operator::zeros(size),
            buffer3: Operator::zeros(size),
            inverse_buffer: get_ldlt_inverse_buffer(size),
        }
    }

    pub fn add_watcher(&mut self, watcher: &'a mut impl PropagatorWatcher<Ratio<Operator>>) {
        if let Some(watchers) = &mut self.watchers {
            watchers.push(watcher)
        } else {
            self.watchers = Some(vec![watcher])
        }
    }

    pub fn set_watchers(&mut self, watchers: Vec<&'a mut dyn PropagatorWatcher<Ratio<Operator>>>) {
        self.watchers = Some(watchers)
    }

    pub fn remove_watchers(&mut self) {
        self.watchers = None
    }

    pub fn change_step_strategy(&mut self, step: StepStrategy) {
        self.step = step
    }

    fn halve_the_step(&mut self) {
        self.solution.dr /= 2.0;

        inverse_ldlt_inplace(self.f.as_ref(), self.buffer1.0.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer2.0.as_mut(),
            Accum::Replace,
            self.buffer1.0.as_ref(),
            self.solution.sol.0.0.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            self.solution.sol.0.0.as_mut(),
            Accum::Replace,
            self.buffer2.0.as_ref(),
            self.f_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), self.w_matrix.id().as_ref()).for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        zip!(self.f_last.as_mut(), self.w_matrix.id().as_ref()).for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        matmul(
            self.buffer1.0.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            self.solution.sol.0.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f_last.as_ref(), self.buffer2.0.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.solution.sol.0.0.as_mut(),
            Accum::Replace,
            self.buffer1.0.as_ref(),
            self.buffer2.0.as_ref(),
            1.,
            Par::Seq,
        );

        ///////////////////////////////////////////////////////

        self.w_matrix
            .value_inplace(self.solution.r - self.solution.dr, &mut self.buffer2);
        zip!(
            self.f_prev_last.as_mut(),
            self.w_matrix.id().as_ref(),
            self.buffer2.0.as_ref()
        )
        .for_each(|unzip!(b1, u, c)| *b1 = u + self.solution.dr * self.solution.dr / 12. * c);
        // f_prev_last is (1 - T_n)

        inverse_ldlt_inplace(self.f_prev_last.as_ref(), self.buffer3.0.as_mut(), &mut self.inverse_buffer);
        // buffer3 is (1 - T_n)^-1

        zip!(
            self.buffer1.0.as_mut(),
            self.w_matrix.id().as_ref(),
            self.f_prev_last.as_ref()
        )
        .for_each(|unzip!(b1, u, f)| *b1 = 12. * u - 10. * f);
        // buffer1 is (2 + 10T_n)

        matmul(
            self.buffer2.0.as_mut(),
            Accum::Replace,
            self.buffer1.0.as_ref(),
            self.buffer3.0.as_ref(),
            1.,
            Par::Seq,
        );
        // buffer2 is U_n

        inverse_ldlt_inplace(self.buffer2.0.as_ref(), self.buffer1.0.as_mut(), &mut self.inverse_buffer);
        // buffer1 is U_n^-1

        zip!(
            self.buffer2.0.as_mut(),
            self.solution.sol.0.0.as_ref(),
            self.w_matrix.id().as_ref()
        )
        .for_each(|unzip!(b2, sol, u)| *b2 = sol + u);

        matmul(
            self.prev_sol.0.0.as_mut(),
            Accum::Replace,
            self.buffer1.0.as_ref(),
            self.buffer2.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.prev_sol.0.0.as_ref(), self.buffer1.0.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer2.0.as_mut(),
            Accum::Replace,
            self.solution.sol.0.0.as_ref(),
            self.buffer1.0.as_ref(),
            1.,
            Par::Seq,
        );

        swap(&mut self.solution.sol.0, &mut self.buffer2);
        swap(&mut self.f_prev_last, &mut self.f_last)
    }

    fn double_the_step(&mut self) {
        self.solution.dr *= 2.;

        matmul(
            self.buffer1.0.as_mut(),
            Accum::Replace,
            self.solution.sol.0.0.as_ref(),
            self.prev_sol.0.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f.as_ref(), self.buffer2.0.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer3.0.as_mut(),
            Accum::Replace,
            self.buffer2.0.as_ref(),
            self.buffer1.0.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            self.solution.sol.0.0.as_mut(),
            Accum::Replace,
            self.buffer3.0.as_ref(),
            self.f_prev_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), self.w_matrix.id().as_ref()).for_each(|unzip!(f, u)| *f = 4. * *f - 3. * u);

        zip!(self.f_last.as_mut(), self.w_matrix.id().as_ref(), self.f_prev_last.as_ref())
            .for_each(|unzip!(f, u, f_prev)| *f = 4. * *f_prev - 3. * u);

        matmul(
            self.buffer1.0.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            self.solution.sol.0.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f_last.as_ref(), self.buffer2.0.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.solution.sol.0.0.as_mut(),
            Accum::Replace,
            self.buffer1.0.as_ref(),
            self.buffer2.0.as_ref(),
            1.,
            Par::Seq,
        );
    }

    fn perform_step(&mut self) {
        self.solution.r += self.solution.dr;

        zip!(
            self.buffer1.0.as_mut(),
            self.w_matrix.id().as_ref(),
            self.w_matrix_buffer.0.as_ref()
        )
        .for_each(|unzip!(b1, u, c)| *b1 = u + self.solution.dr * self.solution.dr / 12. * c);
        // buffer1 is (1 - T_n)

        inverse_ldlt_inplace(self.buffer1.0.as_ref(), self.prev_sol.0.0.as_mut(), &mut self.inverse_buffer);
        // prev_sol is (1 - T_n)^-1

        zip!(self.buffer3.0.as_mut(), self.w_matrix.id().as_ref(), self.buffer1.0.as_ref())
            .for_each(|unzip!(b3, u, b1)| *b3 = 12. * u - 10. * *b1);
        // buffer3 is (2 + 10T_n)

        matmul(
            self.buffer2.0.as_mut(),
            Accum::Replace,
            self.buffer3.0.as_ref(),
            self.prev_sol.0.0.as_ref(),
            1.,
            Par::Seq,
        );
        // buffer2 is U_n

        inverse_ldlt_inplace(
            self.solution.sol.0.0.as_ref(),
            self.prev_sol.0.0.as_mut(),
            &mut self.inverse_buffer,
        );

        zip!(self.prev_sol.0.0.as_mut(), self.buffer2.0.as_ref()).for_each(|unzip!(sol, u)| *sol = u - *sol);
        // prev_sol is R_n

        swap(&mut self.prev_sol, &mut self.solution.sol);

        swap(&mut self.f_prev_last, &mut self.f_last);
        swap(&mut self.f_last, &mut self.f);
        swap(&mut self.f, &mut self.buffer1.0);
    }
}

impl<W: WMatrix> Propagator<Ratio<Operator>> for RatioNumerov<'_, W> {
    fn step(&mut self) -> &Solution<Ratio<Operator>> {
        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.before_step(&self.solution);
            }
        }

        self.w_matrix.value_inplace(self.solution.r, &mut self.w_matrix_buffer);
        let wavelength = get_wavelength(&self.w_matrix_buffer);

        let dr = self.step.get_step(self.solution.r, wavelength);

        if dr > 2.0 * self.solution.dr.abs() {
            self.double_the_step()
        }

        while dr < self.solution.dr.abs() {
            self.halve_the_step();
        }

        self.perform_step();

        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.after_step(&self.solution);
            }
        }
        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<Ratio<Operator>> {
        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.init(&self.solution);
            }
        }

        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step();
        }

        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.finalize(&self.solution);
            }
        }
        &self.solution
    }
}

#[inline]
pub fn get_wavelength(red_coupling: &Operator) -> f64 {
    let max_g_val = red_coupling
        .0
        .diagonal()
        .column_vector()
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    2. * PI / max_g_val.abs().sqrt()
}

pub fn get_s_matrix(sol: &Solution<Ratio<Operator>>, w_matrix: &impl WMatrix) -> SMatrix {
    let size = w_matrix.size();
    let r_last = sol.r;
    let r_prev_last = sol.r - sol.dr;

    let mut f_last = Operator::zeros(size);
    w_matrix.value_inplace(r_last, &mut f_last);
    f_last.0 *= sol.dr * sol.dr / 12.;
    f_last.0 += &w_matrix.id().0;

    let mut f_prev_last = Operator::zeros(size);
    w_matrix.value_inplace(r_prev_last, &mut f_prev_last);
    f_prev_last.0 *= sol.dr * sol.dr / 12.;
    f_prev_last.0 += &w_matrix.id().0;

    let wave_ratio = f_last.0.partial_piv_lu().inverse() * sol.sol.0.0.as_ref() * f_prev_last.0;

    let asymptote = &w_matrix.asymptote();
    let levels = asymptote.levels();

    let wave_ratio = if let Some(transformation) = asymptote.transformation() {
        transformation.0.transpose() * wave_ratio * &transformation.0
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
        .map(|&val| (2.0 * asymptote.red_mass * (asymptote.energy - val).abs()).sqrt())
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
        .find(|(i, _)| *i == asymptote.entrance_level)
        .expect("Closed entrance channel")
        .0;

    SMatrix::new(s_matrix, momenta[asymptote.entrance_level], entrance)
}
