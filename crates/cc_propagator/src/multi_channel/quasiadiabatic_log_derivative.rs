
use std::f64::consts::PI;

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
        Matrix, WMatrix, diabatic_log_derivative::{diag_mat_mul_inplace, mat_diag_add_inplace, mat_diag_mul_inplace, mat_diag_sub_inplace}
    },
    step_strategy::ConstRefStepStrat,
};
use cc_matrix_utils::faer::{
    get_ldlt_inverse_buffer, 
    get_self_adjoint_evd_buffer, 
    inverse_ldlt_inplace_nodes, 
    self_adjoint_evd_inplace
};
use faer::{
    Accum::Replace, Shape, diag::{
        Diag,
    }, dyn_stack::MemBuffer, linalg::{
        matmul::matmul,
        solvers::DenseSolveCore,
    }, make_guard, matrix_free::LinOp as _, unzip, zip
};

pub struct QuasiAdiabaticLogDerivative<'a, W>
where
    W: WMatrix<f64>,
{
    w_matrix: &'a W,
    solution: Solution<LogDeriv<Matrix>>,
    solution_local: Solution<LogDeriv<Matrix>>,
    nodes: Nodes,

    step: LogDerivativeStep,
    step_strat: ConstRefStepStrat,
}

impl<'a, W: WMatrix<f64>> QuasiAdiabaticLogDerivative<'a, W> {
    pub fn new(w_matrix: &'a W, step_strat: ConstRefStepStrat, boundary: Boundary<Matrix>) -> Self {
        let r = boundary.r_start;
        let dr = match boundary.direction {
            Direction::Inwards => -(step_strat.init_step(r).abs()),
            Direction::Outwards => step_strat.init_step(r).abs(),
        };
        let solution = Solution {
            r,
            dr,
            sol: LogDeriv(boundary.derivative * boundary.value.partial_piv_lu().inverse()),
        };

        let mut step = LogDerivativeStep::new(w_matrix.size());
        w_matrix.value_inplace(r, &mut step.w_matrix_buffer);
        self_adjoint_evd_inplace(step.w_matrix_buffer.as_ref(), step.w_ref.as_mut(), None, &mut step.diagonalisation_buffer);

        Self {
            step,
            // adiabatic_transformation is initialized to Id
            solution_local: solution.clone(),
            solution,
            nodes: Nodes(0),
            step_strat,
            w_matrix,
        }
    }

    pub fn from_log_derivative(w_matrix: &'a W, step_strat: ConstRefStepStrat, solution: Solution<LogDeriv<Matrix>>, nodes: Nodes) -> Self {
        let mut step = LogDerivativeStep::new(w_matrix.size());
        w_matrix.value_inplace(solution.r, &mut step.w_matrix_buffer);
        self_adjoint_evd_inplace(step.w_matrix_buffer.as_ref(), step.w_ref.as_mut(), None, &mut step.diagonalisation_buffer);

        Self {
            step,
            step_strat,
            // adiabatic_transformation is initialized to Id
            solution_local: solution.clone(),
            solution,
            nodes,
            w_matrix,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        let wavelength = {
            let max_g_val = self.step.w_ref
                .column_vector()
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            2. * PI / max_g_val.abs().sqrt()
        };
        self.solution_local.dr = self.step_strat.get_step(self.solution_local.r, self.solution_local.dr, wavelength);

        if let Some(r) = r
            && (self.solution_local.r - r).abs() < self.solution_local.dr.abs()
        {
            self.solution_local.dr *= ((self.solution_local.r - r) / self.solution_local.dr).abs()
        }

        self.step.perform_step(&mut self.solution_local, &mut self.nodes, self.w_matrix);
    }
}

impl<W: WMatrix<f64>> Propagator<LogDeriv<Matrix>> for QuasiAdiabaticLogDerivative<'_, W> {
    fn step(&mut self) -> &Solution<LogDeriv<Matrix>> {
        self.step_r_target(None);

        self.solution.r = self.solution_local.r;
        self.solution.dr = self.solution_local.dr;
        matmul(
            self.step.buffer1.as_mut(),
            faer::Accum::Replace,
            self.step.adiabatic_transf.as_ref().transpose(),
            self.solution_local.sol.0.as_ref(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.solution.sol.0.as_mut(),
            faer::Accum::Replace,
            self.step.buffer1.as_ref(),
            self.step.adiabatic_transf.as_ref(),
            1.0,
            faer::Par::Seq,
        );

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<Matrix>> {
        while (self.solution_local.r - r) * self.solution_local.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        self.solution.r = self.solution_local.r;
        self.solution.dr = self.solution_local.dr;
        matmul(
            self.step.buffer1.as_mut(),
            faer::Accum::Replace,
            self.step.adiabatic_transf.as_ref().transpose(),
            self.solution_local.sol.0.as_ref(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.solution.sol.0.as_mut(),
            faer::Accum::Replace,
            self.step.buffer1.as_ref(),
            self.step.adiabatic_transf.as_ref(),
            1.0,
            faer::Par::Seq,
        );

        &self.solution
    }
}

impl<W: WMatrix<f64>> WithNodeCount for QuasiAdiabaticLogDerivative<'_, W> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

impl<W: WMatrix<f64>> WithWaveStorage<Matrix> for QuasiAdiabaticLogDerivative<'_, W> {
    fn init_wave_storage(&mut self) {
        self.step.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.step.wave_storage.as_ref()
    }
}

/// https://doi.org/10.1016/0010-4655(94)90200-3 
struct LogDerivativeStep {
    buffer1: Matrix,
    buffer2: Matrix,

    inverse_buffer: MemBuffer,
    diagonalisation_buffer: MemBuffer,

    y14: Diag<f64>,
    z22_23_33: Diag<f64>,

    w_matrix_buffer: Matrix,
    w_ref: Diag<f64>,
    adiabatic_transf: Matrix,
    adiabatic_connection: Matrix,

    wave_storage: Option<WaveStorage<Matrix>>,
}

impl LogDerivativeStep {
    pub fn new(size: usize) -> Self {
        Self {
            buffer1: Matrix::zeros(size, size),
            buffer2: Matrix::zeros(size, size),
            inverse_buffer: get_ldlt_inverse_buffer(size),
            diagonalisation_buffer: get_self_adjoint_evd_buffer(size),

            y14: Diag::zeros(size),
            z22_23_33: Diag::zeros(size),

            w_matrix_buffer: Matrix::zeros(size, size),
            w_ref: Diag::zeros(size),
            adiabatic_transf: Matrix::identity(size, size),
            adiabatic_connection: Matrix::zeros(size, size),

            wave_storage: None,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Matrix>>, nodes: &mut Nodes, w_matrix: &impl WMatrix<f64>) {
        let h = sol.dr / 2.0;

        // get w_ref, adiabatic_transf and adiabatic_connection
        w_matrix.value_inplace(sol.r + h, &mut self.buffer1);
        self_adjoint_evd_inplace(
            self.buffer1.as_ref(), 
            self.w_ref.as_mut(), 
            Some(self.buffer2.as_mut()), 
            &mut self.diagonalisation_buffer
        );
        matmul(
            self.adiabatic_connection.as_mut(),
            Replace,
            self.buffer2.transpose(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq,
        );
        self.adiabatic_transf.copy_from(self.buffer2.transpose());

        // calculate imbeddings y14, y23 and z22_23_33 (meaning y2/3 z y2/3, y2 and y3 are the same)
        {
	        make_guard!(n);
            let n = self.w_ref.nrows().bind(n);

            let mut y14 = self.y14.as_shape_mut(n);
            let mut z22_23_33 = self.z22_23_33.as_shape_mut(n);

            let w_ref = self.w_ref.as_shape(n);

            for i in n.indices() {
                let p2 = w_ref[i];
                
                let (y14_i, y23_i) = if p2 < 0.0 {
                    (
                        (-p2).sqrt() / f64::tanh((-p2).sqrt() * h),
                        (-p2).sqrt() / f64::sinh((-p2).sqrt() * h)
                    )
                } else {
                    (
                        p2.sqrt() / f64::tan(p2.sqrt() * h),
                        p2.sqrt() / f64::sin(p2.sqrt() * h)
                    )
                };

                y14[i] = y14_i;
                z22_23_33[i] = y23_i * y23_i / (2. * y14_i);
            }
        }

        // transform Y_{n-1} to local basis
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            sol.sol.0.as_ref(),
            self.adiabatic_connection.transpose(),
            1.0,
            faer::Par::Seq
        );
        matmul(
            sol.sol.0.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_connection.as_ref(),
            self.buffer1.as_ref(),
            1.0,
            faer::Par::Seq
        );

        // transform W(a) to local basis
        matmul(
            self.buffer2.as_mut(),
            faer::Accum::Replace,
            self.w_matrix_buffer.as_ref(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq
        );
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_transf.as_ref(),
            self.buffer2.as_ref(),
            1.0,
            faer::Par::Seq
        );
        // buffer1 has W(a) in local basis

        mat_diag_sub_inplace(self.buffer1.as_mut(), self.w_ref.as_ref());
        // buffer1 is -W^1(a) (sign change because of different convention)
        self.buffer1 *= -h / 3.;
        mat_diag_add_inplace(self.buffer1.as_mut(), self.y14.as_ref());
        mat_diag_sub_inplace(self.buffer1.as_mut(), self.z22_23_33.as_ref());
        // buffer1 is a y_1(a, b)

        zip!(self.buffer1.as_mut(), sol.sol.0.as_ref())
        .for_each(|unzip!(y1, sol)| {
            *y1 += sol
        });
        let mut nodes_new = inverse_ldlt_inplace_nodes(self.buffer1.as_ref(), sol.sol.0.as_mut(), &mut self.inverse_buffer);
        // sol is now (y + y1(a, b))^-1

        mat_diag_mul_inplace(sol.sol.0.as_mut(), self.z22_23_33.as_ref());
        // sol is now (y + y1(a, b))^-1 * y_2(a, b)
        if let Some(wave_storage) = &mut self.wave_storage {
            matmul(
                self.buffer1.as_mut(),
                faer::Accum::Replace,
                self.adiabatic_transf.as_ref().transpose(),
                sol.sol.0.as_ref(),
                1.0,
                faer::Par::Seq
            );
            matmul(
                self.buffer2.as_mut(),
                faer::Accum::Replace,
                self.buffer1.as_ref(),
                self.adiabatic_transf.as_ref(),
                1.0,
                faer::Par::Seq
            );

            wave_storage.push(sol.r, &self.buffer2)
        }
        diag_mat_mul_inplace(self.z22_23_33.as_ref(), sol.sol.0.as_mut());
        // sol is now y_3(a, b) * (y + y1(a, b))^-1 * y_2(a, b)

        w_matrix.value_inplace(sol.r + sol.dr, &mut self.w_matrix_buffer);
        // transform W(b) to local basis
        matmul(
            self.buffer2.as_mut(),
            faer::Accum::Replace,
            self.w_matrix_buffer.as_ref(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq
        );
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_transf.as_ref(),
            self.buffer2.as_ref(),
            1.0,
            faer::Par::Seq
        );
        // buffer1 has W(b) in local basis

        mat_diag_sub_inplace(self.buffer1.as_mut(), self.w_ref.as_ref());
        // buffer1 is -W^1(b) (sign change because of different convention)
        self.buffer1 *= -h / 3.;
        mat_diag_add_inplace(self.buffer1.as_mut(), self.y14.as_ref());
        mat_diag_sub_inplace(self.buffer1.as_mut(), self.z22_23_33.as_ref());
        // buffer1 is a y_4(a, b)

        zip!(sol.sol.0.as_mut(), self.buffer1.as_ref())
        .for_each(|unzip!(y, y4)| {
            *y = y4 - *y
        });
        // sol is y(b) in local basis

        let dim = w_matrix.size();
        if sol.dr < 0. {
            nodes_new = dim as u64 - nodes_new
        }
        nodes.0 += nodes_new;

        sol.r += sol.dr;
    }
}

