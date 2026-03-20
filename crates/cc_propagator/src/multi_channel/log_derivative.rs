use std::marker::PhantomData;

use crate::{
    Boundary,
    Direction,
    LogDeriv,
    Nodes,
    Propagator,
    Solution,
    WaveStorage,
    WithNodeCount,
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
    inverse_ldlt_inplace_nodes,
};
use faer::{
    Accum::Replace,
    Par::Seq,
    dyn_stack::MemBuffer,
    linalg::{
        matmul::matmul,
        solvers::DenseSolveCore,
    },
    unzip,
    zip,
};

// doi: 10.1063/1.451472
pub trait LogDerivReference {
    fn w_ref(w_c: &Matrix, w_ref: &mut Matrix);

    fn imbedding1(h: f64, w_ref: &Matrix, out: &mut Matrix);
    fn imbedding2(h: f64, w_ref: &Matrix, out: &mut Matrix);
    fn imbedding3(h: f64, w_ref: &Matrix, out: &mut Matrix);
    fn imbedding4(h: f64, w_ref: &Matrix, out: &mut Matrix);
}

pub type JohnsonLogDerivative<'a, W, S> = DiabaticLogDerivative<'a, Johnson, W, S>;
pub type ManolopoulosLogDerivative<'a, W, S> = DiabaticLogDerivative<'a, DiabaticManolopoulos, W, S>;

pub struct Johnson;
impl LogDerivReference for Johnson {
    fn w_ref(_w_c: &Matrix, w_ref: &mut Matrix) {
        w_ref.fill(0.);
    }

    fn imbedding1(h: f64, _w_ref: &Matrix, out: &mut Matrix) {
        out.fill(0.);

        out.diagonal_mut().column_vector_mut().iter_mut().for_each(|y1| *y1 = 1.0 / h);
    }

    fn imbedding2(h: f64, _w_ref: &Matrix, out: &mut Matrix) {
        out.fill(0.);

        out.diagonal_mut().column_vector_mut().iter_mut().for_each(|y2| *y2 = 1.0 / h);
    }

    #[inline]
    fn imbedding3(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        Self::imbedding1(h, w_ref, out);
    }
}

pub struct DiabaticManolopoulos;
impl LogDerivReference for DiabaticManolopoulos {
    fn w_ref(w_c: &Matrix, w_ref: &mut Matrix) {
        w_ref.fill(0.);

        w_ref
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_c.diagonal().column_vector().iter())
            .for_each(|(w_ref, &w_c)| *w_ref = w_c);
    }

    fn imbedding1(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        out.fill(0.);

        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_ref.diagonal().column_vector().iter())
            .for_each(|(y1, &p2)| {
                if p2 < 0.0 {
                    *y1 = (-p2).sqrt() * 1.0 / f64::tanh((-p2).sqrt() * h)
                } else {
                    *y1 = p2.sqrt() * 1.0 / f64::tan(p2.sqrt() * h)
                }
            });
    }

    fn imbedding2(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        out.fill(0.);

        out.diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_ref.diagonal().column_vector().iter())
            .for_each(|(y2, &p2)| {
                if p2 < 0.0 {
                    *y2 = (-p2).sqrt() * 1.0 / f64::sinh((-p2).sqrt() * h)
                } else {
                    *y2 = p2.sqrt() * 1.0 / f64::sin(p2.sqrt() * h)
                }
            });
    }

    #[inline]
    fn imbedding3(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: &Matrix, out: &mut Matrix) {
        Self::imbedding1(h, w_ref, out);
    }
}

pub struct DiabaticLogDerivative<'a, R, W, S>
where
    R: LogDerivReference,
    W: WMatrix<f64>,
    S: Step,
{
    w_matrix: &'a W,
    solution: Solution<LogDeriv<Matrix>>,
    nodes: Nodes,

    step: LogDerivativeStep<R>,
    step_strat: S,
}

impl<'a, R: LogDerivReference, W: WMatrix<f64>, S: Step> DiabaticLogDerivative<'a, R, W, S> {
    pub fn new(w_matrix: &'a W, step_strat: S, boundary: Boundary<Matrix>) -> Self {
        let r = boundary.r_start;

        let mut step = LogDerivativeStep::new(w_matrix.size());

        w_matrix.value_inplace(r, &mut step.w_matrix_buffer);
        let local_wavelength = local_wavelength(&step.w_matrix_buffer);

        let dr = match boundary.direction {
            Direction::Inwards => -(step_strat.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step_strat.get_step(r, local_wavelength).abs(),
        };

        let sol = Solution {
            r,
            dr,
            sol: LogDeriv(boundary.derivative * boundary.value.partial_piv_lu().inverse()),
        };

        Self {
            step,
            solution: sol,
            nodes: Nodes(0),
            step_strat,
            w_matrix,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        let wavelength = local_wavelength(&self.step.w_matrix_buffer);

        let dr_new = self.step_strat.get_step(self.solution.r, wavelength);
        self.solution.dr = dr_new.clamp(0., 2. * self.solution.dr.abs()) * self.solution.dr.signum();

        if let Some(r) = r
            && (self.solution.r - r).abs() < self.solution.dr.abs()
        {
            self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
        }

        self.step.perform_step(&mut self.solution, &mut self.nodes, self.w_matrix);
    }
}

impl<R: LogDerivReference, W: WMatrix<f64>, S: Step> Propagator<LogDeriv<Matrix>> for DiabaticLogDerivative<'_, R, W, S> {
    fn step(&mut self) -> &Solution<LogDeriv<Matrix>> {
        self.step_r_target(None);

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<Matrix>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        &self.solution
    }
}

impl<R: LogDerivReference, W: WMatrix<f64>, S: Step> WithNodeCount for DiabaticLogDerivative<'_, R, W, S> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

impl<R: LogDerivReference, W: WMatrix<f64>, S: Step> WithWaveStorage<Matrix> for DiabaticLogDerivative<'_, R, W, S> {
    fn init_wave_storage(&mut self) {
        self.step.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.step.wave_storage.as_ref()
    }
}

/// https://doi.org/10.1016/0010-4655(94)90200-3
struct LogDerivativeStep<R: LogDerivReference> {
    id: Matrix,
    buffer1: Matrix,
    buffer2: Matrix,
    buffer3: Matrix,
    inverse_buffer: MemBuffer,

    z_matrix: Matrix,
    w_ref: Matrix,

    reference: PhantomData<R>,
    w_matrix_buffer: Matrix,

    wave_storage: Option<WaveStorage<Matrix>>,
}

impl<R: LogDerivReference> LogDerivativeStep<R> {
    pub fn new(size: usize) -> Self {
        Self {
            id: Matrix::identity(size, size),
            buffer1: Matrix::zeros(size, size),
            buffer2: Matrix::zeros(size, size),
            buffer3: Matrix::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),

            z_matrix: Matrix::zeros(size, size),
            w_ref: Matrix::zeros(size, size),

            w_matrix_buffer: Matrix::zeros(size, size),

            reference: PhantomData,
            wave_storage: None,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Matrix>>, nodes: &mut Nodes, w_matrix: &impl WMatrix<f64>) {
        let h = sol.dr / 2.0;

        w_matrix.value_inplace(sol.r + h, &mut self.buffer1);
        R::w_ref(&self.buffer1, &mut self.w_ref);

        zip!(self.buffer1.as_mut(), self.id.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(b, u, w_ref)| {
            *b = u - h * h / 6. * (w_ref - *b)  // sign change because of different convention
        });

        inverse_ldlt_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        zip!(self.buffer2.as_mut(), self.id.as_ref())
        .for_each(|unzip!(b, u)| {
            *b = 6. / (h * h) * (*b - u)
        });
        // buffer2 is a W_tilde(c)

        R::imbedding4(h, &self.w_ref, &mut self.buffer1);

        zip!(self.buffer1.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, w_tilde)| {
            *y4 += 2. * h / 3. * w_tilde
        });
        // buffer1 is a y_4(a, c)

        R::imbedding1(h, &self.w_ref, &mut self.buffer3);

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, w_tilde)| {
            *y4 += 2. * h / 3. * w_tilde
        });
        // buffer3 is a y_1(c, b)

        zip!(self.buffer1.as_mut(), self.buffer3.as_ref())
        .for_each(|unzip!(y4, y1)| {
            *y4 += y1
        });
        inverse_ldlt_inplace(self.buffer1.as_ref(), self.z_matrix.as_mut(), &mut self.inverse_buffer);
        // z_matrix is a z(a, b, c)

        R::imbedding2(h, &self.w_ref, &mut self.buffer1);
        matmul(self.buffer3.as_mut(), Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Seq);
        R::imbedding3(h, &self.w_ref, &mut self.buffer1);
        matmul(self.buffer2.as_mut(), Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Seq);
        // buffer2 is a second term in y_1(a, b)

        R::imbedding1(h, &self.w_ref, &mut self.buffer3);

        zip!(self.buffer3.as_mut(), self.w_matrix_buffer.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(y1, w_a, w_ref)| {
            *y1 += h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_1(a, c)

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y1, b)| {
            *y1 -= b
        });
        // buffer3 is a y_1(a, b)

        zip!(self.buffer3.as_mut(), sol.sol.0.as_ref())
        .for_each(|unzip!(y1, sol)| {
            *y1 += sol
        });

        let mut nodes_new = inverse_ldlt_inplace_nodes(self.buffer3.as_ref(), sol.sol.0.as_mut(), &mut self.inverse_buffer);
        // sol is now (y + y1(a, b))^-1

        R::imbedding2(h, &self.w_ref, &mut self.buffer1);
        matmul(self.buffer3.as_mut(), Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Seq);
        matmul(self.buffer2.as_mut(), Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Seq);

        matmul(self.buffer1.as_mut(), Replace, sol.sol.0.as_ref(), self.buffer2.as_ref(), 1.0, Seq);
        // buffer1 is now (y + y1(a, b))^-1 * y_2(a, b)

        if let Some(wave_storage) = &mut self.wave_storage {
            wave_storage.push(sol.r, &self.buffer1)
        }

        R::imbedding3(h, &self.w_ref, &mut self.buffer2);
        matmul(sol.sol.0.as_mut(), Replace, self.buffer2.as_ref(), self.z_matrix.as_ref(), 1.0, Seq);
        matmul(self.buffer3.as_mut(), Replace, sol.sol.0.as_ref(), self.buffer2.as_ref(), 1.0, Seq);

        matmul(sol.sol.0.as_mut(), Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Seq);
        // sol is now y_3(a, b) * (y + y1(a, b))^-1 * y_2(a, b)

        R::imbedding3(h, &self.w_ref, &mut self.buffer1);
        matmul(self.buffer3.as_mut(), Replace, self.buffer1.as_ref(), self.z_matrix.as_ref(), 1.0, Seq);
        R::imbedding2(h, &self.w_ref, &mut self.buffer1);
        matmul(self.buffer2.as_mut(), Replace, self.buffer3.as_ref(), self.buffer1.as_ref(), 1.0, Seq);
        // buffer2 is a second term in y_4(a, b)

        w_matrix.value_inplace(sol.r + sol.dr, &mut self.w_matrix_buffer);
        R::imbedding4(h, &self.w_ref, &mut self.buffer3);

        zip!(self.buffer3.as_mut(), self.w_matrix_buffer.as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(y4, w_a, w_ref)| {
            *y4 += h / 3. * (w_ref - w_a) // sign change because of different convention
        });
        // buffer3 is a y_4(c, b)

        zip!(self.buffer3.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(y4, b)| {
            *y4 -= b
        });
        // buffer3 is a y_4(a, b)

        zip!(sol.sol.0.as_mut(), self.buffer3.as_ref())
        .for_each(|unzip!(y, y4)| {
            *y = y4 - *y
        });
        // sol is y(b)

        let dim = w_matrix.size();
        if sol.dr < 0. {
            nodes_new = dim as u64 - nodes_new
        }
        nodes.0 += nodes_new;

        sol.r += sol.dr;
    }
}
