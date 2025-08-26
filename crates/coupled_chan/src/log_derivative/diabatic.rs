use std::marker::PhantomData;

use faer::{
    Accum::Replace,
    Par::Seq,
    dyn_stack::MemBuffer,
    linalg::{matmul::matmul, solvers::DenseSolveCore},
    unzip, zip,
};
use matrix_utils::faer::{get_ldlt_inverse_buffer, inverse_ldlt_inplace, inverse_ldlt_inplace_nodes};
use propagator::{
    Boundary, Direction, LogDeriv, NodeCountPropagator, Nodes, Propagator, Solution, propagator_watcher::PropagatorWatcher,
    step_strategy::StepStrategy,
};

use crate::{Operator, coupling::WMatrix, ratio_numerov::get_wavelength};

// doi: 10.1063/1.451472
pub trait LogDerivativeReference {
    fn w_ref(w_c: &Operator, w_ref: &mut Operator);

    fn imbedding1(h: f64, w_ref: &Operator, out: &mut Operator);
    fn imbedding2(h: f64, w_ref: &Operator, out: &mut Operator);
    fn imbedding3(h: f64, w_ref: &Operator, out: &mut Operator);
    fn imbedding4(h: f64, w_ref: &Operator, out: &mut Operator);
}

pub type JohnsonLogDerivative<'a, W> = DiabaticLogDerivative<'a, Johnson, W>;
pub type ManolopoulosLogDerivative<'a, W> = DiabaticLogDerivative<'a, DiabaticManolopoulos, W>;

pub struct Johnson;
impl LogDerivativeReference for Johnson {
    fn w_ref(_w_c: &Operator, w_ref: &mut Operator) {
        w_ref.fill(0.);
    }

    fn imbedding1(h: f64, _w_ref: &Operator, out: &mut Operator) {
        out.fill(0.);

        out.diagonal_mut().column_vector_mut().iter_mut().for_each(|y1| *y1 = 1.0 / h);
    }

    fn imbedding2(h: f64, _w_ref: &Operator, out: &mut Operator) {
        out.fill(0.);

        out.diagonal_mut().column_vector_mut().iter_mut().for_each(|y2| *y2 = 1.0 / h);
    }

    #[inline]
    fn imbedding3(h: f64, w_ref: &Operator, out: &mut Operator) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: &Operator, out: &mut Operator) {
        Self::imbedding1(h, w_ref, out);
    }
}

pub struct DiabaticManolopoulos;
impl LogDerivativeReference for DiabaticManolopoulos {
    fn w_ref(w_c: &Operator, w_ref: &mut Operator) {
        w_ref.fill(0.);

        w_ref
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(w_c.diagonal().column_vector().iter())
            .for_each(|(w_ref, &w_c)| *w_ref = w_c);
    }

    fn imbedding1(h: f64, w_ref: &Operator, out: &mut Operator) {
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

    fn imbedding2(h: f64, w_ref: &Operator, out: &mut Operator) {
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
    fn imbedding3(h: f64, w_ref: &Operator, out: &mut Operator) {
        Self::imbedding2(h, w_ref, out);
    }

    #[inline]
    fn imbedding4(h: f64, w_ref: &Operator, out: &mut Operator) {
        Self::imbedding1(h, w_ref, out);
    }
}

pub struct DiabaticLogDerivative<'a, R, W>
where
    R: LogDerivativeReference,
    W: WMatrix,
{
    w_matrix: &'a W,
    solution: Solution<LogDeriv<Operator>>,
    nodes: Nodes,
    watchers: Option<Vec<&'a mut dyn PropagatorWatcher<LogDeriv<Operator>>>>,

    step: LogDerivativeStep<R>,
    step_strat: StepStrategy,
}

impl<'a, R: LogDerivativeReference, W: WMatrix> DiabaticLogDerivative<'a, R, W> {
    pub fn new(w_matrix: &'a W, step_strat: StepStrategy, boundary: Boundary<Operator>) -> Self {
        let r = boundary.r_start;

        let mut step = LogDerivativeStep::new(w_matrix.size());

        w_matrix.value_inplace(r, &mut step.w_matrix_buffer);
        let local_wavelength = get_wavelength(&step.w_matrix_buffer);

        let dr = match boundary.direction {
            Direction::Inwards => -(step_strat.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step_strat.get_step(r, local_wavelength).abs(),
        };

        let sol = Solution {
            r,
            dr,
            sol: LogDeriv(Operator::new(
                boundary.derivative.0 * boundary.value.partial_piv_lu().inverse(),
            )),
        };

        Self {
            step,
            solution: sol,
            nodes: Nodes(0),
            step_strat,
            w_matrix,
            watchers: None,
        }
    }

    pub fn add_watcher(&mut self, watcher: &'a mut impl PropagatorWatcher<LogDeriv<Operator>>) {
        if let Some(watchers) = &mut self.watchers {
            watchers.push(watcher)
        } else {
            self.watchers = Some(vec![watcher])
        }
    }

    pub fn set_watchers(&mut self, watchers: Vec<&'a mut dyn PropagatorWatcher<LogDeriv<Operator>>>) {
        self.watchers = Some(watchers)
    }

    pub fn remove_watchers(&mut self) {
        self.watchers = None
    }

    pub fn change_step_strategy(&mut self, step: StepStrategy) {
        self.step_strat = step
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        let wavelength = get_wavelength(&self.step.w_matrix_buffer);

        let dr_new = self.step_strat.get_step(self.solution.r, wavelength);
        self.solution.dr = dr_new.clamp(0., 2. * self.solution.dr.abs()) * self.solution.dr.signum();

        if let Some(r) = r {
            if (self.solution.r - r).abs() < self.solution.dr.abs() {
                self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
            }
        }

        self.step.perform_step(&mut self.solution, &mut self.nodes, self.w_matrix);
    }
}

impl<R: LogDerivativeReference, W: WMatrix> Propagator<LogDeriv<Operator>> for DiabaticLogDerivative<'_, R, W> {
    fn step(&mut self) -> &Solution<LogDeriv<Operator>> {
        self.step_r_target(None);

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<Operator>> {
        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.init(&self.solution);
            }
        }

        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.finalize(&self.solution);
            }
        }

        &self.solution
    }
}

impl<R: LogDerivativeReference, W: WMatrix> NodeCountPropagator<LogDeriv<Operator>> for DiabaticLogDerivative<'_, R, W> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

/// https://doi.org/10.1016/0010-4655(94)90200-3
struct LogDerivativeStep<R: LogDerivativeReference> {
    buffer1: Operator,
    buffer2: Operator,
    buffer3: Operator,
    inverse_buffer: MemBuffer,

    z_matrix: Operator,
    w_ref: Operator,

    reference: PhantomData<R>,
    w_matrix_buffer: Operator,

    wave_reconstruct_buffer: Option<Operator>,
}

impl<R: LogDerivativeReference> LogDerivativeStep<R> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer1: Operator::zeros(size),
            buffer2: Operator::zeros(size),
            buffer3: Operator::zeros(size),
            inverse_buffer: get_ldlt_inverse_buffer(size),

            z_matrix: Operator::zeros(size),
            w_ref: Operator::zeros(size),

            w_matrix_buffer: Operator::zeros(size),

            reference: PhantomData,
            wave_reconstruct_buffer: None,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Operator>>, nodes: &mut Nodes, w_matrix: &impl WMatrix) {
        let h = sol.dr / 2.0;

        w_matrix.value_inplace(sol.r + h, &mut self.buffer1);
        R::w_ref(&self.buffer1, &mut self.w_ref);

        zip!(self.buffer1.as_mut(), w_matrix.id().as_ref(), self.w_ref.as_ref())
        .for_each(|unzip!(b, u, w_ref)| {
            *b = u - h * h / 6. * (w_ref - *b)  // sign change because of different convention
        });

        inverse_ldlt_inplace(self.buffer1.as_ref(), self.buffer2.as_mut(), &mut self.inverse_buffer);

        zip!(self.buffer2.as_mut(), w_matrix.id().as_ref())
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

        if let Some(wave_reconstruct_buffer) = &mut self.wave_reconstruct_buffer {
            wave_reconstruct_buffer.copy_from(self.buffer1.as_ref());
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
