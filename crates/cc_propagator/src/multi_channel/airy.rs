use crate::multi_channel::diabatic_log_derivative::{
    diag_mat_mul_inplace,
    mat_diag_add_inplace,
    mat_diag_mul_inplace,
};
use crate::multi_channel::{
    Matrix,
    WMatrix,
};
use crate::single_channel::airy::get_imbeddings;
use crate::step_strategy::LinearRefStepStrat;
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
    get_self_adjoint_evd_buffer,
    inverse_ldlt_inplace_nodes,
    self_adjoint_evd_inplace,
};
use faer::diag::Diag;
use faer::dyn_stack::MemBuffer;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::DenseSolveCore;
use faer::matrix_free::LinOp;
use faer::{
    Shape,
    make_guard,
    unzip,
    zip,
};

const ROOT_3: f64 = 1.732050807;

pub struct AiryLogDerivative<'a, W: WMatrix<f64>> {
    w_matrix: &'a W,
    solution_local: Solution<LogDeriv<Matrix>>,
    solution: Solution<LogDeriv<Matrix>>,
    nodes: Nodes,

    step: AiryLogDerivativeStep,
}

impl<'a, W: WMatrix<f64>> AiryLogDerivative<'a, W> {
    pub fn new(w_matrix: &'a W, step_strat: LinearRefStepStrat, boundary: Boundary<Matrix>) -> Self {
        let r = boundary.r_start;
        let dr = match boundary.direction {
            Direction::Inwards => -(step_strat.init_step(r).abs()),
            Direction::Outwards => step_strat.init_step(r).abs(),
        };

        let mut step = AiryLogDerivativeStep::new(w_matrix.size(), step_strat, dr);
        step.get_init_prev_w_ref(r, dr, w_matrix);

        let mut sol = Solution {
            r,
            dr,
            sol: LogDeriv(boundary.derivative * boundary.value.partial_piv_lu().inverse()),
        };
        sol.sol.0 = &step.adiabatic_transf * sol.sol.0 * step.adiabatic_transf.transpose();

        Self {
            step,
            solution_local: sol.clone(),
            solution: sol,
            nodes: Nodes(0),
            w_matrix,
        }
    }

    pub fn from_log_derivative(
        w_matrix: &'a W,
        step_strat: LinearRefStepStrat,
        solution: Solution<LogDeriv<Matrix>>,
        nodes: Nodes,
    ) -> Self {
        let mut step = AiryLogDerivativeStep::new(w_matrix.size(), step_strat, solution.dr);
        step.get_init_prev_w_ref(solution.r, solution.dr, w_matrix);

        let mut sol_local = solution.clone();
        sol_local.sol.0 = &step.adiabatic_transf * sol_local.sol.0 * step.adiabatic_transf.transpose();

        Self {
            step,
            solution_local: sol_local,
            solution,
            nodes,
            w_matrix,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        if let Some(r) = r
            && (self.solution_local.r - r).abs() < self.step.dr_next.abs()
        {
            self.step.dr_next *= ((self.solution_local.r - r) / self.step.dr_next).abs()
        }

        self.step
            .perform_step(&mut self.solution_local, &mut self.nodes, self.w_matrix);
    }
}

impl<W: WMatrix<f64>> Propagator<LogDeriv<Matrix>> for AiryLogDerivative<'_, W> {
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

impl<W: WMatrix<f64>> WithNodeCount for AiryLogDerivative<'_, W> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

impl<W: WMatrix<f64>> WithWaveStorage<Matrix> for AiryLogDerivative<'_, W> {
    fn init_wave_storage(&mut self) {
        self.step.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.step.wave_storage.as_ref()
    }
}

pub struct AiryLogDerivativeStep {
    step_strat: LinearRefStepStrat,
    dr_next: f64,

    w_ref: Diag<f64>,
    w_deriv_ref: Matrix,

    adiabatic_transf: Matrix,
    adiabatic_connection_buffer: Matrix,

    y1: Diag<f64>,
    y23: Diag<f64>,
    y4: Diag<f64>,
    buffer1: Matrix,
    buffer2: Matrix,

    inverse_buffer: MemBuffer,
    diagonalisation_buffer: MemBuffer,
    wave_storage: Option<WaveStorage<Matrix>>,
}

impl AiryLogDerivativeStep {
    pub fn new(size: usize, step_strat: LinearRefStepStrat, dr_next: f64) -> Self {
        Self {
            step_strat,
            dr_next,
            w_ref: Diag::zeros(size),
            w_deriv_ref: Matrix::zeros(size, size),
            y1: Diag::zeros(size),
            y23: Diag::zeros(size),
            y4: Diag::zeros(size),
            buffer1: Matrix::zeros(size, size),
            buffer2: Matrix::zeros(size, size),

            adiabatic_transf: Matrix::zeros(size, size),
            adiabatic_connection_buffer: Matrix::zeros(size, size),

            inverse_buffer: get_ldlt_inverse_buffer(size),
            diagonalisation_buffer: get_self_adjoint_evd_buffer(size),
            wave_storage: None,
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Matrix>>, nodes: &mut Nodes, w_matrix: &impl WMatrix<f64>) {
        let h = self.dr_next;
        self.get_w_ref_local(sol.r, self.dr_next, sol.dr, w_matrix);

        // calculate imbeddings y1, y23, y4
        {
	        make_guard!(n);
            let n = self.w_ref.nrows().bind(n);

            let mut y1 = self.y1.as_shape_mut(n);
            let mut y23 = self.y23.as_shape_mut(n);
            let mut y4 = self.y4.as_shape_mut(n);

            let w_ref = self.w_ref.as_shape(n);
            let w_deriv_ref = self.w_deriv_ref.diagonal().as_shape(n);

            for i in n.indices() {
                let imbeddings = get_imbeddings(w_ref[i], w_deriv_ref[i], h);

                y1[i] = imbeddings.y1;
                y23[i] = imbeddings.y23;
                y4[i] = imbeddings.y4;
            }
        }

        // transform Y_{n-1} to local basis
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            sol.sol.0.as_ref(),
            self.adiabatic_connection_buffer.transpose(),
            1.0,
            faer::Par::Seq
        );
        matmul(
            sol.sol.0.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_connection_buffer.as_ref(),
            self.buffer1.as_ref(),
            1.0,
            faer::Par::Seq
        );

        mat_diag_add_inplace(sol.sol.0.as_mut(), self.y1.as_ref());
        // sol is not Y + y1

        let mut nodes_new = inverse_ldlt_inplace_nodes(sol.sol.0.as_ref(), self.buffer1.as_mut(), &mut self.inverse_buffer);
        // buffer1 is (Y + y1)^-1

        mat_diag_mul_inplace(self.buffer1.as_mut(), self.y23.as_ref());
        if let Some(wave_storage) = &mut self.wave_storage {
            matmul(
                self.buffer2.as_mut(),
                faer::Accum::Replace,
                self.adiabatic_transf.as_ref().transpose(),
                self.buffer1.as_ref(),
                1.0,
                faer::Par::Seq
            );
            matmul(
                sol.sol.0.as_mut(),
                faer::Accum::Replace,
                self.buffer2.as_ref(),
                self.adiabatic_transf.as_ref(),
                1.0,
                faer::Par::Seq
            );

            wave_storage.push(sol.r, &sol.sol.0)
        }
        diag_mat_mul_inplace(self.y23.as_ref(), self.buffer1.as_mut());

        zip!(sol.sol.0.as_mut(), self.buffer1.as_ref())
        .for_each(|unzip!(w, b)| {
            *w = -*b
        });
        mat_diag_add_inplace(sol.sol.0.as_mut(), self.y4.as_ref());
        // sol is Y_new = y4 - y3 (Y + y1)^-1 y2 in local basis

        let dim = w_matrix.size();
        if h < 0. {
            nodes_new = dim as u64 - nodes_new
        }
        nodes.0 += nodes_new;

        sol.dr = h;
        sol.r += sol.dr;
    }

    /// calculates initial previous step w_ref, w_deriv_ref, adiabatic_transf
    /// and calculates dr_next
    fn get_init_prev_w_ref(&mut self, r: f64, dr: f64, w_matrix: &impl WMatrix<f64>) {
        w_matrix.value_inplace(r + dr / 2. * (-1. + 1. / ROOT_3), &mut self.buffer1);
        w_matrix.value_inplace(r + dr / 2. * (-1. - 1. / ROOT_3), &mut self.buffer2);

        self.w_deriv_ref.copy_from(&self.buffer1);
        zip!(self.w_deriv_ref.as_mut(), self.buffer2.as_ref()).for_each(|unzip!(w, b)| *w = (*w - b) * ROOT_3 / dr);

        zip!(self.buffer1.as_mut(), self.buffer2.as_ref()).for_each(|unzip!(w, b)| *w = (*w + b) / 2.0);

        // get in local basis: w_ref, adiabatic_transf
        self_adjoint_evd_inplace(
            self.buffer1.as_ref(),
            self.w_ref.as_mut(),
            Some(self.buffer2.as_mut()),
            &mut self.diagonalisation_buffer,
        );
        self.adiabatic_transf.copy_from(self.buffer2.transpose());

        // convert w_deriv_ref to local basis
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            self.w_deriv_ref.as_ref(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.w_deriv_ref.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_transf.as_ref(),
            self.buffer1.as_ref(),
            1.0,
            faer::Par::Seq,
        );

        let mut w_deriv_off_estimate = 0.0;
        zip!(self.w_deriv_ref.as_ref()).for_each_triangular_lower(faer::linalg::zip::Diag::Skip, |unzip!(x)| {
            w_deriv_off_estimate = f64::max(x.abs(), w_deriv_off_estimate)
        });

        let mut average_wave_length = 0.0;
        let n = w_matrix.size();
        zip!(self.w_ref.as_ref()).for_each(|unzip!(x)| average_wave_length += f64::sqrt(x.abs()));
        average_wave_length /= n as f64;

        self.dr_next = self
            .step_strat
            .get_step(r, dr, average_wave_length, w_deriv_off_estimate, None)
    }

    /// calculates current step w_ref, w_deriv_ref, adiabatic_transf, adiabatic_connection_buffer
    /// and calculates dr_next
    fn get_w_ref_local(&mut self, r: f64, dr: f64, dr_prev: f64, w_matrix: &impl WMatrix<f64>) {
        self.y1.copy_from(self.w_ref.as_ref());
        // y1 temporarily contains old w_ref values for second derivative estimation

        // calculate w_ref, w_deriv_ref
        w_matrix.value_inplace(r + dr / 2. * (1. + 1. / ROOT_3), &mut self.buffer1);
        w_matrix.value_inplace(r + dr / 2. * (1. - 1. / ROOT_3), &mut self.buffer2);

        self.w_deriv_ref.copy_from(&self.buffer1);
        zip!(self.w_deriv_ref.as_mut(), self.buffer2.as_ref()).for_each(|unzip!(w, b)| *w = (*w - b) * ROOT_3 / dr);

        zip!(self.buffer1.as_mut(), self.buffer2.as_ref()).for_each(|unzip!(w, b)| *w = (*w + b) / 2.0);

        // get in local basis: w_ref, adiabatic_transf, adiabatic_connection_buffer
        self_adjoint_evd_inplace(
            self.buffer1.as_ref(),
            self.w_ref.as_mut(),
            Some(self.buffer2.as_mut()),
            &mut self.diagonalisation_buffer,
        );
        matmul(
            self.adiabatic_connection_buffer.as_mut(),
            faer::Accum::Replace,
            self.buffer2.transpose(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq,
        );
        self.adiabatic_transf.copy_from(self.buffer2.transpose());

        // convert w_deriv_ref to local basis
        matmul(
            self.buffer1.as_mut(),
            faer::Accum::Replace,
            self.w_deriv_ref.as_ref(),
            self.adiabatic_transf.transpose(),
            1.0,
            faer::Par::Seq,
        );
        matmul(
            self.w_deriv_ref.as_mut(),
            faer::Accum::Replace,
            self.adiabatic_transf.as_ref(),
            self.buffer1.as_ref(),
            1.0,
            faer::Par::Seq,
        );

        let mut w_deriv_off_estimate = 0.0;
        zip!(self.w_deriv_ref.as_ref()).for_each_triangular_lower(faer::linalg::zip::Diag::Skip, |unzip!(x)| {
            w_deriv_off_estimate = f64::max(x.abs(), w_deriv_off_estimate)
        });

        let mut average_wave_vector = 0.0;
        let n = w_matrix.size();
        zip!(self.w_ref.as_ref()).for_each(|unzip!(x)| average_wave_vector += f64::sqrt(x.abs()));
        average_wave_vector /= n as f64;

        let mut w_dderiv_estimate = 0.0;
        {
            make_guard!(n);
            let n = self.w_ref.nrows().bind(n);

            let w_ref_prev = self.y1.as_shape_mut(n);
            let w_ref = self.w_ref.as_shape_mut(n);
            let w_deriv_ref = self.w_deriv_ref.diagonal().as_shape(n);

            let dr_mid = (dr + dr_prev) / 2.0;
            for i in n.indices() {
                w_dderiv_estimate += 2. / (dr_mid * dr_mid) * (w_ref[i] - w_ref_prev[i] + w_deriv_ref[i] * dr_mid).abs()
            }
        }
        w_dderiv_estimate /= n as f64;

        self.dr_next =
            self.step_strat
                .get_step(r + dr, dr, average_wave_vector, w_deriv_off_estimate, Some(w_dderiv_estimate))
    }
}
