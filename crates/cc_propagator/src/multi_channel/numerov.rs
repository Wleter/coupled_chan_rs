use std::mem::swap;

use crate::{
    Boundary,
    Direction,
    Propagator,
    Ratio,
    Solution,
    WaveStorage,
    WithWaveStorage,
    multi_channel::{
        Matrix,
        WMatrix,
        local_wavelength,
    },
    step_strategy::Step,
};
use cc_matrix_utils::faer::{
    get_ldlt_inverse_buffer,
    inverse_ldlt_inplace,
};
use faer::{
    Accum,
    Par,
    dyn_stack::MemBuffer,
    linalg::{
        matmul::matmul,
        solvers::DenseSolveCore,
    },
    unzip,
    zip,
};

// todo! look whether V(r) is evaluated at correct values

/// 10.1063/1.436421
pub struct RatioNumerov<'a, W: WMatrix<f64>, S: Step> {
    w_matrix: &'a W,
    step: S,

    solution: Solution<Ratio<Matrix>>,

    w_matrix_buffer: Matrix,
    prev_sol: Ratio<Matrix>,

    f: Matrix,
    f_last: Matrix,
    f_prev_last: Matrix,

    id: Matrix,
    buffer1: Matrix,
    buffer2: Matrix,
    buffer3: Matrix,
    inverse_buffer: MemBuffer,

    wave_storage: Option<WaveStorage<Matrix>>,
}

impl<'a, W: WMatrix<f64>, S: Step> RatioNumerov<'a, W, S> {
    pub fn new(w_matrix: &'a W, step: S, boundary: Boundary<Matrix>) -> Self {
        let size = w_matrix.size();
        let r = boundary.r_start;

        let mut w_matrix_buffer = Matrix::zeros(size, size);

        w_matrix.value_inplace(r, &mut w_matrix_buffer);
        let local_wavelength = local_wavelength(&w_matrix_buffer);

        let dr = match boundary.direction {
            Direction::Inwards => -(step.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step.get_step(r, local_wavelength).abs(),
        };

        let mut f_last = Matrix::zeros(size, size);
        w_matrix.value_inplace(r - dr, &mut f_last);

        let mut f_prev_last = Matrix::zeros(size, size);
        w_matrix.value_inplace(r - 2. * dr, &mut f_prev_last);

        let id = Matrix::identity(size, size);

        let f_last = &id + dr * dr / 12. * f_last;
        let f_prev_last = &id + dr * dr / 12. * f_prev_last;
        let f = &id + dr * dr / 12. * &w_matrix_buffer;

        let sol = Ratio(
            &f * (&boundary.derivative * dr + &boundary.value)
                * boundary.value.partial_piv_lu().inverse()
                * f_last.partial_piv_lu().inverse(),
        );

        let prev_sol = Ratio(
            &f_last
                * &boundary.value
                * (&boundary.value - &boundary.derivative * dr).partial_piv_lu().inverse()
                * f_prev_last.partial_piv_lu().inverse(),
        );

        Self {
            w_matrix,
            step,
            solution: Solution { r, dr, sol },
            f,
            f_last,
            f_prev_last,
            prev_sol,

            w_matrix_buffer,

            id,
            buffer1: Matrix::zeros(size, size),
            buffer2: Matrix::zeros(size, size),
            buffer3: Matrix::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),

            wave_storage: None,
        }
    }

    // todo! small error here
    fn halve_the_step(&mut self) {
        self.solution.dr /= 2.0;

        inverse_ldlt_inplace(self.f.as_ref(), self.buffer1.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.solution.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            self.solution.sol.0.as_mut(),
            Accum::Replace,
            self.buffer2.as_ref(),
            self.f_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), self.id.as_ref()).for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        zip!(self.f_last.as_mut(), self.id.as_ref()).for_each(|unzip!(f, u)| *f = *f / 4. + 0.75 * u);

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            self.solution.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f_last.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.solution.sol.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );

        ///////////////////////////////////////////////////////

        self.w_matrix
            .value_inplace(self.solution.r - self.solution.dr, &mut self.buffer2);
        zip!(self.f_prev_last.as_mut(), self.id.as_ref(), self.buffer2.as_ref())
            .for_each(|unzip!(b1, u, c)| *b1 = u + self.solution.dr * self.solution.dr / 12. * c);
        // f_prev_last is (1 - T_n)

        zip!(self.buffer1.as_mut(), self.id.as_ref(), self.f_prev_last.as_ref())
            .for_each(|unzip!(b1, u, f)| *b1 = 12. * u - 10. * f);
        // buffer1 is (2 + 10T_n)

        inverse_ldlt_inplace(self.buffer1.as_ref(), self.buffer3.as_mut(), &mut self.inverse_buffer);
        // buffer3 is (2 + 10T_n)^-1

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.f_prev_last.as_ref(),
            self.buffer3.as_ref(),
            1.,
            Par::Seq,
        );
        // buffer1 is U_n^-1

        zip!(self.buffer2.as_mut(), self.solution.sol.0.as_ref(), self.id.as_ref())
            .for_each(|unzip!(b2, sol, u)| *b2 = sol + u);

        matmul(
            self.prev_sol.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.prev_sol.0.as_ref(), self.buffer1.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer2.as_mut(),
            Accum::Replace,
            self.solution.sol.0.as_ref(),
            self.buffer1.as_ref(),
            1.,
            Par::Seq,
        );

        swap(&mut self.solution.sol.0, &mut self.buffer2);
        swap(&mut self.f_prev_last, &mut self.f_last)
    }

    fn double_the_step(&mut self) {
        self.solution.dr *= 2.;

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.solution.sol.0.as_ref(),
            self.prev_sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.buffer3.as_mut(),
            Accum::Replace,
            self.buffer2.as_ref(),
            self.buffer1.as_ref(),
            1.,
            Par::Seq,
        );

        matmul(
            self.solution.sol.0.as_mut(),
            Accum::Replace,
            self.buffer3.as_ref(),
            self.f_prev_last.as_ref(),
            1.,
            Par::Seq,
        );

        zip!(self.f.as_mut(), self.id.as_ref()).for_each(|unzip!(f, u)| *f = 4. * *f - 3. * u);

        zip!(self.f_last.as_mut(), self.id.as_ref(), self.f_prev_last.as_ref())
            .for_each(|unzip!(f, u, f_prev)| *f = 4. * *f_prev - 3. * u);

        matmul(
            self.buffer1.as_mut(),
            Accum::Replace,
            self.f.as_ref(),
            self.solution.sol.0.as_ref(),
            1.,
            Par::Seq,
        );

        inverse_ldlt_inplace(self.f_last.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        matmul(
            self.solution.sol.0.as_mut(),
            Accum::Replace,
            self.buffer1.as_ref(),
            self.buffer2.as_ref(),
            1.,
            Par::Seq,
        );
    }

    fn perform_step(&mut self) {
        self.solution.r += self.solution.dr;

        inverse_ldlt_inplace(self.f.as_ref(), self.prev_sol.0.as_mut(), &mut self.inverse_buffer);
        // prev_sol is (1 - T_n)^-1

        zip!(self.buffer2.as_mut(), self.id.as_ref(), self.prev_sol.0.as_ref())
            .for_each(|unzip!(b3, u, f_inv)| *b3 = 12. * f_inv - 10. * u);
        // buffer2 is U_n

        inverse_ldlt_inplace(
            self.solution.sol.0.as_ref(),
            self.prev_sol.0.as_mut(),
            &mut self.inverse_buffer,
        );

        zip!(self.prev_sol.0.as_mut(), self.buffer2.as_ref()).for_each(|unzip!(sol, u)| *sol = u - *sol);
        // prev_sol is R_n

        swap(&mut self.prev_sol, &mut self.solution.sol);

        swap(&mut self.f_prev_last, &mut self.f_last);
        swap(&mut self.f_last, &mut self.f);

        zip!(self.buffer1.as_mut(), self.id.as_ref(), self.w_matrix_buffer.as_ref())
            .for_each(|unzip!(b1, u, c)| *b1 = u + self.solution.dr * self.solution.dr / 12. * c);
        // buffer1 is (1 - T_{n+1})
        swap(&mut self.f, &mut self.buffer1);

        if let Some(w) = &mut self.wave_storage {
            w.push(self.solution.r, &self.solution.sol.0);
        }
    }
}

impl<W: WMatrix<f64>, S: Step> Propagator<Ratio<Matrix>> for RatioNumerov<'_, W, S> {
    fn step(&mut self) -> &Solution<Ratio<Matrix>> {
        let wavelength = local_wavelength(&self.w_matrix_buffer);
        let dr = self.step.get_step(self.solution.r, wavelength);

        if dr > 2.0 * self.solution.dr.abs() {
            self.double_the_step()
        }

        while dr < self.solution.dr.abs() {
            self.halve_the_step();
        }

        self.w_matrix
            .value_inplace(self.solution.r + self.solution.dr, &mut self.w_matrix_buffer);
        self.perform_step();

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<Ratio<Matrix>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step();
        }

        &self.solution
    }
}

impl<W: WMatrix<f64>, S: Step> WithWaveStorage<Matrix> for RatioNumerov<'_, W, S> {
    fn init_wave_storage(&mut self) {
        self.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.wave_storage.as_ref()
    }
}
