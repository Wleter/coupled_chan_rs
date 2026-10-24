use std::marker::PhantomData;

use crate::multi_channel::{
    Matrix,
    WMatrix,
    local_wavelength,
};
use crate::step_strategy::ConstRefStepStrat;
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
};
use cc_matrix_utils::faer::{
    get_ldlt_inverse_buffer,
    inverse_ldlt_inplace,
    inverse_ldlt_inplace_nodes,
};
use faer::Accum::Replace;
use faer::Par::Seq;
use faer::diag::{
    Diag,
    DiagRef,
};
use faer::dyn_stack::MemBuffer;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::DenseSolveCore;
use faer::{
    MatMut,
    Shape,
    make_guard,
    unzip,
    zip,
};

// doi: 10.1063/1.451472
pub trait LogDerivReference {
    fn w_ref(w_c: &Matrix, w_ref: &mut Diag<f64>);

    fn imbedding14(h: f64, w_ref: &Diag<f64>, out: &mut Diag<f64>);
    fn imbedding23(h: f64, w_ref: &Diag<f64>, out: &mut Diag<f64>);
}

pub type JohnsonLogDerivative<'a, W> = DiabaticLogDerivative<'a, Johnson, W>;
pub type ManolopoulosLogDerivative<'a, W> = DiabaticLogDerivative<'a, DiabaticManolopoulos, W>;

pub struct Johnson;
impl LogDerivReference for Johnson {
    fn w_ref(_w_c: &Matrix, w_ref: &mut Diag<f64>) {
        zip!(w_ref.as_mut()).for_each(|unzip!(w)| *w = 0.);
    }

    fn imbedding14(h: f64, _w_ref: &Diag<f64>, out: &mut Diag<f64>) {
        out.column_vector_mut().iter_mut().for_each(|y1| *y1 = 1.0 / h);
    }

    fn imbedding23(h: f64, _w_ref: &Diag<f64>, out: &mut Diag<f64>) {
        out.column_vector_mut().iter_mut().for_each(|y2| *y2 = 1.0 / h);
    }
}

pub struct DiabaticManolopoulos;
impl LogDerivReference for DiabaticManolopoulos {
    fn w_ref(w_c: &Matrix, w_ref: &mut Diag<f64>) {
        zip!(w_ref.as_mut(), w_c.diagonal()).for_each(|unzip!(w, w_c)| *w = *w_c);
    }

    fn imbedding14(h: f64, w_ref: &Diag<f64>, out: &mut Diag<f64>) {
        out.column_vector_mut()
            .iter_mut()
            .zip(w_ref.column_vector().iter())
            .for_each(|(y1, &p2)| {
                if p2 < 0.0 {
                    *y1 = (-p2).sqrt() / f64::tanh((-p2).sqrt() * h)
                } else {
                    *y1 = p2.sqrt() / f64::tan(p2.sqrt() * h)
                }
            });
    }

    fn imbedding23(h: f64, w_ref: &Diag<f64>, out: &mut Diag<f64>) {
        out.column_vector_mut()
            .iter_mut()
            .zip(w_ref.column_vector().iter())
            .for_each(|(y2, &p2)| {
                if p2 < 0.0 {
                    *y2 = (-p2).sqrt() / f64::sinh((-p2).sqrt() * h)
                } else {
                    *y2 = p2.sqrt() / f64::sin(p2.sqrt() * h)
                }
            });
    }
}

pub struct DiabaticLogDerivative<'a, R, W>
where
    R: LogDerivReference,
    W: WMatrix<f64>,
{
    w_matrix: &'a W,
    solution: Solution<LogDeriv<Matrix>>,
    nodes: Nodes,

    step: LogDerivativeStep<R>,
    step_strat: ConstRefStepStrat,
}

impl<'a, R: LogDerivReference, W: WMatrix<f64>> DiabaticLogDerivative<'a, R, W> {
    pub fn new(w_matrix: &'a W, step_strat: ConstRefStepStrat, boundary: Boundary<Matrix>) -> Self {
        let r = boundary.r_start;
        let dr = match boundary.direction {
            Direction::Inwards => -(step_strat.init_step(r).abs()),
            Direction::Outwards => step_strat.init_step(r).abs(),
        };

        let mut step = LogDerivativeStep::new(w_matrix.size());
        w_matrix.value_inplace(r, &mut step.w_matrix_buffer);

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

    pub fn from_log_derivative(
        w_matrix: &'a W,
        step_strat: ConstRefStepStrat,
        solution: Solution<LogDeriv<Matrix>>,
        nodes: Nodes,
    ) -> Self {
        let mut step = LogDerivativeStep::new(w_matrix.size());
        w_matrix.value_inplace(solution.r, &mut step.w_matrix_buffer);

        Self {
            step,
            step_strat,
            solution,
            nodes,
            w_matrix,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        let wavelength = local_wavelength(&self.step.w_matrix_buffer);
        self.solution.dr = self.step_strat.get_step(self.solution.r, self.solution.dr, wavelength);

        if let Some(r) = r
            && (self.solution.r - r).abs() < self.solution.dr.abs()
        {
            self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
        }

        self.step.perform_step(&mut self.solution, &mut self.nodes, self.w_matrix);
    }
}

impl<R: LogDerivReference, W: WMatrix<f64>> Propagator<LogDeriv<Matrix>> for DiabaticLogDerivative<'_, R, W> {
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

impl<R: LogDerivReference, W: WMatrix<f64>> WithNodeCount for DiabaticLogDerivative<'_, R, W> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

impl<R: LogDerivReference, W: WMatrix<f64>> WithWaveStorage<Matrix> for DiabaticLogDerivative<'_, R, W> {
    fn init_wave_storage(&mut self) {
        self.step.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.step.wave_storage.as_ref()
    }
}

/// https://doi.org/10.1016/0010-4655(94)90200-3
struct LogDerivativeStep<R: LogDerivReference> {
    buffer1: Matrix,
    buffer2: Matrix,
    inverse_buffer: MemBuffer,

    z22_23_33: Matrix,
    w_ref: Diag<f64>,
    y14: Diag<f64>,
    y23: Diag<f64>,

    reference: PhantomData<R>,
    w_matrix_buffer: Matrix,

    wave_storage: Option<WaveStorage<Matrix>>,
}

impl<R: LogDerivReference> LogDerivativeStep<R> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer1: Matrix::zeros(size, size),
            buffer2: Matrix::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),

            z22_23_33: Matrix::zeros(size, size),
            w_ref: Diag::zeros(size),
            y14: Diag::zeros(size),
            y23: Diag::zeros(size),

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
        R::imbedding14(h, &self.w_ref, &mut self.y14);
        R::imbedding23(h, &self.w_ref, &mut self.y23);

        // sign change because of different convention
        mat_diag_sub_inplace(self.buffer1.as_mut(), self.w_ref.as_ref());
        self.buffer1 *= h * h / 6.;
        zip!(self.buffer1.diagonal_mut()).for_each(|unzip!(b)| *b += 1.);

        inverse_ldlt_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        zip!(self.buffer2.diagonal_mut()).for_each(|unzip!(b)| *b -= 1.);
        self.buffer2 *= 8. / h;
        // buffer2 is a 4 / 3 * h * W_tilde(c)

        zip!(self.buffer2.diagonal_mut(), self.y14.as_ref()).for_each(|unzip!(b, y14)| {
            *b += 2. * y14
        });
        // buffer2 is a z^-1(a, b, c)

        inverse_ldlt_inplace(self.buffer2.as_ref(), self.z22_23_33.as_mut(), &mut self.inverse_buffer);
        // z22_23_33 is a z(a, b, c)

        diag_mat_mul_inplace(self.y23.as_ref(), self.z22_23_33.as_mut());
        mat_diag_mul_inplace(self.z22_23_33.as_mut(), self.y23.as_ref());
        // z22_23_33 is a y2/3 * z * y2/3, where y2/3 is either y2 or y3

        // sign change because of different convention
        mat_diag_sub_inplace(self.w_matrix_buffer.as_mut(), self.w_ref.as_ref());
        self.w_matrix_buffer *= -h / 3.;
        mat_diag_add_inplace(self.w_matrix_buffer.as_mut(), self.y14.as_ref());
        // w_matrix_buffer is a y_1(a, c)

        zip!(self.w_matrix_buffer.as_mut(), self.z22_23_33.as_ref())
        .for_each(|unzip!(y1, z23)| {
            *y1 -= z23
        });
        // w_matrix_buffer is a y_1(a, b)

        zip!(self.w_matrix_buffer.as_mut(), sol.sol.0.as_ref())
        .for_each(|unzip!(y1, sol)| {
            *y1 += sol
        });
        let mut nodes_new = inverse_ldlt_inplace_nodes(self.w_matrix_buffer.as_ref(), sol.sol.0.as_mut(), &mut self.inverse_buffer);
        // sol is now (y + y1(a, b))^-1

        matmul(self.buffer1.as_mut(), Replace, sol.sol.0.as_ref(), self.z22_23_33.as_ref(), 1.0, Seq);
        // buffer1 is now (y + y1(a, b))^-1 * y_2(a, b)
        if let Some(wave_storage) = &mut self.wave_storage {
            wave_storage.push(sol.r, &self.buffer1)
        }
        matmul(sol.sol.0.as_mut(), Replace, self.z22_23_33.as_ref(), self.buffer1.as_ref(), 1.0, Seq);
        // sol is now y_3(a, b) * (y + y1(a, b))^-1 * y_2(a, b)

        w_matrix.value_inplace(sol.r + sol.dr, &mut self.w_matrix_buffer);

        // sign change because of different convention
        zip!(self.buffer1.as_mut(), self.w_matrix_buffer.as_ref())
        .for_each(|unzip!(y4, w_a)| {
            *y4 = h / 3. * (-w_a)
        });
        zip!(self.buffer1.diagonal_mut(), self.w_ref.as_ref())
        .for_each(|unzip!(y4, w_c)| {
            *y4 += h / 3. * w_c
        });
        mat_diag_add_inplace(self.buffer1.as_mut(), self.y14.as_ref());
        // buffer1 is a y_4(c, b)

        zip!(self.buffer1.as_mut(), self.z22_23_33.as_ref())
        .for_each(|unzip!(y4, z32)| {
            *y4 -= z32
        });
        // buffer1 is a y_4(a, b)

        zip!(sol.sol.0.as_mut(), self.buffer1.as_ref())
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

pub(crate) fn diag_mat_mul_inplace<N: Shape, K: Shape>(a: DiagRef<f64, N>, b: MatMut<f64, N, K>) {
    let a_shape = a.dim();
    let (b_rows, b_cols) = b.shape();
    assert_eq!(a_shape, b_rows);

    make_guard!(n);
    make_guard!(k);
    let n = b_rows.bind(n);
    let k = b_cols.bind(k);

    let a = a.as_shape(n);
    let mut b = b.as_shape_mut(n, k);

    for j in k.indices() {
        for i in n.indices() {
            b[(i, j)] *= a[i]
        }
    }
}

pub(crate) fn mat_diag_mul_inplace<N: Shape, K: Shape>(a: MatMut<f64, N, K>, b: DiagRef<f64, K>) {
    let (a_rows, a_cols) = a.shape();
    let b_shape = b.dim();
    assert_eq!(a_cols, b_shape);

    make_guard!(n);
    make_guard!(k);
    let n = a_rows.bind(n);
    let k = a_cols.bind(k);

    let mut a = a.as_shape_mut(n, k);
    let b = b.as_shape(k);

    for j in k.indices() {
        for i in n.indices() {
            a[(i, j)] *= b[j]
        }
    }
}

pub(crate) fn mat_diag_add_inplace(a: MatMut<f64>, b: DiagRef<f64>) {
    a.diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .zip(b.column_vector().iter())
        .for_each(|(b2, y)| *b2 += y);
}

pub(crate) fn mat_diag_sub_inplace(a: MatMut<f64>, b: DiagRef<f64>) {
    a.diagonal_mut()
        .column_vector_mut()
        .iter_mut()
        .zip(b.column_vector().iter())
        .for_each(|(b2, y)| *b2 -= y);
}
