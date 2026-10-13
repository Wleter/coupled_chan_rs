
use cc_matrix_utils::faer::{get_ldlt_inverse_buffer, get_self_adjoint_evd_buffer, inverse_ldlt_inplace_nodes, self_adjoint_evd_inplace};
use faer::{Shape, diag::Diag, dyn_stack::MemBuffer, linalg::{matmul::matmul, solvers::DenseSolveCore}, make_guard, matrix_free::LinOp, unzip, zip};
use crate::{Boundary, Direction, LogDeriv, Nodes, Propagator, Solution, WaveStorage, WithNodeCount, WithWaveStorage, multi_channel::{Matrix, WMatrix, local_wavelength, log_derivative::{diag_mat_mul_inplace, mat_diag_add_inplace, mat_diag_mul_inplace}}, single_channel::airy::get_imbeddings, step_strategy::Step};

pub struct AiryLogDerivative<'a, W, S>
where
    W: WMatrix<f64>,
    S: Step,
{
    w_matrix: &'a W,
    solution: Solution<LogDeriv<Matrix>>,
    nodes: Nodes,

    step: AiryLogDerivativeStep,
    step_strat: S,
}

impl<'a, W: WMatrix<f64>, S: Step> AiryLogDerivative<'a, W, S> {
    pub fn new(w_matrix: &'a W, step_strat: S, boundary: Boundary<Matrix>, asymptotic_criterion: f64) -> Self {
        let r = boundary.r_start;

        let mut step = AiryLogDerivativeStep::new(w_matrix.size(), asymptotic_criterion);

        w_matrix.value_inplace(r, &mut step.buffer1);
        let local_wavelength = local_wavelength(&step.buffer1);

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
        let wavelength = local_wavelength(&self.step.buffer1);

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

impl<W: WMatrix<f64>, S: Step> Propagator<LogDeriv<Matrix>> for AiryLogDerivative<'_, W, S> {
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

impl<W: WMatrix<f64>, S: Step> WithNodeCount for AiryLogDerivative<'_, W, S> {
    fn nodes(&self) -> Nodes {
        self.nodes
    }
}

impl<W: WMatrix<f64>, S: Step> WithWaveStorage<Matrix> for AiryLogDerivative<'_, W, S> {
    fn init_wave_storage(&mut self) {
        self.step.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<Matrix>> {
        self.step.wave_storage.as_ref()
    }
}

pub struct AiryLogDerivativeStep {
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
    asymptotic_criterion: f64
}

impl AiryLogDerivativeStep {
    pub fn new(size: usize, asymptotic_criterion: f64) -> Self {
        Self {
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
            asymptotic_criterion
        }
    }

    #[rustfmt::skip]
    fn perform_step(&mut self, sol: &mut Solution<LogDeriv<Matrix>>, nodes: &mut Nodes, w_matrix: &impl WMatrix<f64>) {
        let h = sol.dr;

        // calculate w_ref, w_deriv_ref
        let root_3 = f64::sqrt(3.0);
        w_matrix.value_inplace(sol.r + h / 2. * (1. + 1. / root_3), &mut self.buffer1);
        w_matrix.value_inplace(sol.r + h / 2. * (1. - 1. / root_3), &mut self.buffer2);

        self.w_deriv_ref.copy_from(&self.buffer1);
        zip!(self.w_deriv_ref.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(w, b)| {
            *w = (*w - b) * root_3 / h
        });

        zip!(self.buffer1.as_mut(), self.buffer2.as_ref())
        .for_each(|unzip!(w, b)| {
            *w = (*w + b) / 2.0
        });

        // get in local basis: w_ref, adiabatic_transf, adiabatic_connection_buffer
        self_adjoint_evd_inplace(
            self.buffer1.as_ref(),
            self.w_ref.as_mut(), 
            Some(self.buffer2.as_mut()), 
            &mut self.diagonalisation_buffer
        );
        matmul(
            self.adiabatic_connection_buffer.as_mut(), 
            faer::Accum::Replace, 
            self.buffer2.as_ref(), 
            self.adiabatic_transf.transpose(), 
            1.0, 
            faer::Par::Seq
        );
        self.adiabatic_transf.copy_from(&self.buffer2);

        // convert w_deriv_ref to local basis
        matmul(
            self.buffer1.as_mut(), 
            faer::Accum::Replace, 
            self.w_deriv_ref.as_ref(), 
            self.adiabatic_transf.transpose(), 
            1.0, 
            faer::Par::Seq
        );
        matmul(
            self.w_deriv_ref.as_mut(), 
            faer::Accum::Replace, 
            self.adiabatic_transf.as_ref(), 
            self.buffer1.as_ref(), 
            1.0, 
            faer::Par::Seq
        );

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
                let imbeddings = get_imbeddings(w_ref[i], w_deriv_ref[i], sol.r, h, self.asymptotic_criterion);

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
            wave_storage.push(sol.r, &self.buffer1)
        }
        diag_mat_mul_inplace(self.y23.as_ref(), self.buffer1.as_mut());

        zip!(sol.sol.0.as_mut(), self.buffer1.as_ref())
        .for_each(|unzip!(w, b)| {
            *w = -*b
        });
        mat_diag_add_inplace(sol.sol.0.as_mut(), self.y4.as_ref());
        // sol is Y_new = y4 - y3 (Y + y1)^-1 y2 in local basis

        let dim = w_matrix.size();
        if sol.dr < 0. {
            nodes_new = dim as u64 - nodes_new
        }
        nodes.0 += nodes_new;

        sol.r += sol.dr;
    }
}