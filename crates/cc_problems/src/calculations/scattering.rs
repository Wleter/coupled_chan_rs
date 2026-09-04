use std::{
    marker::PhantomData,
};

use coupled_chan::{
    cc_propagator::{
        Direction,
        Propagator,
        step_strategy::{
            DynStep,
            LocalWavelengthStep,
            SingleStep,
            TransitionStep,
        },
    },
    coupling::{
        Asymptote,
        CollisionParams,
        CollisionWMatrix,
        RCoupling,
    },
    multi_channel::{
        Matrix,
        WMatrix,
        log_derivative::{
            JohnsonLogDerivative,
            ManolopoulosLogDerivative,
        },
        numerov::RatioNumerov,
    },
    s_matrix::SMatrix,
};
use hilbert_space::faer::{
    Col,
    complex::Complex64,
};
use serde::{
    Deserialize,
    Serialize,
};
use unit_systems::quantities::{
    Power,
    Scalar,
    phys_quantities::{
        Energy,
        Length,
        Mass,
    },
};

use crate::{
    UNITS_CONVERTER,
    calculations::{
        Modified, SingleCalc, dependence::DependenceCalc
    },
    parameters::TypedParamId,
    problems::Problem,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScatteringCalcInput {
    pub entrance: usize,
    pub energy: Scalar<Energy>,

    #[serde(default)]
    pub boundary: Boundary,
    pub r_start: Scalar<Length>,
    pub r_stop: Scalar<Length>,
    pub step: Step,
    pub solver: CoupledChanSolver,
}

impl ScatteringCalcInput {
    pub fn scattering(&self, w_matrix: &CollisionWMatrix<impl RCoupling>) -> SMatrixData {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r_start = converter.scalar_value(&self.r_start);
        let r_stop = converter.scalar_value(&self.r_stop);

        assert!(r_start < r_stop, "Expected r_start < r_stop");

        let mut mat = Matrix::zeros(w_matrix.size(), w_matrix.size());
        w_matrix.value_inplace(r_start, &mut mat);
        let values = mat
            .self_adjoint_eigenvalues(hilbert_space::faer::Side::Lower)
            .expect("Could not diagonalize w_matrix at r_start");

        for v in values {
            assert!(
                v < 0.0,
                "Locally open channels at the r_start = {:?} of scattering calculation",
                self.r_start
            );
        }

        let step = self.step.get_step();
        let boundary = self.boundary.get_boundary(&self.r_start, Direction::Outwards, w_matrix);

        // todo! simplify, log-derivatives should be together
        match &self.solver {
            CoupledChanSolver::RatioNumerov => {
                let mut numerov = RatioNumerov::new(w_matrix, step, boundary);
                let sol = numerov.propagate_to(r_stop);

                SMatrixData::new(&SMatrix::from_ratio(sol, w_matrix))
            }
            CoupledChanSolver::JohnsonLogDeriv => {
                let mut log_deriv = JohnsonLogDerivative::new(w_matrix, step, boundary);
                let sol = log_deriv.propagate_to(r_stop);

                SMatrixData::new(&SMatrix::from_log_deriv(sol, w_matrix))
            }
            CoupledChanSolver::ManolopoulosLogDeriv => {
                let mut log_deriv = ManolopoulosLogDerivative::new(w_matrix, step, boundary);
                let sol = log_deriv.propagate_to(r_stop);

                SMatrixData::new(&SMatrix::from_log_deriv(sol, w_matrix))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SMatrixData {
    s_matrix_vec: Vec<Vec<Complex64>>,
    momenta: Vec<f64>,
    entrance_nr: usize,
}

impl SMatrixData {
    pub fn new(s_matrix: &SMatrix) -> Self {
        let s_matrix_mat = s_matrix.s_matrix();
        let mut vec = vec![vec![]; s_matrix_mat.nrows()];
        for (i, row) in s_matrix_mat.row_iter().enumerate() {
            vec[i] = row.iter().copied().collect();
        }

        Self {
            s_matrix_vec: vec,
            momenta: s_matrix.momenta().to_vec(),
            entrance_nr: s_matrix.entrance_number(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoupledChanSolver {
    RatioNumerov,
    JohnsonLogDeriv,
    ManolopoulosLogDeriv,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Fixed {
        dr: Scalar<Length>,
    },
    LocalWavelength {
        dr_min: Scalar<Length>,
        dr_max: Scalar<Length>,
        wave_ratio: f64,
    },
    Transitioned {
        transition_point: Scalar<Length>,
        before: Box<Step>,
        after: Box<Step>,
    },
}

impl Step {
    pub fn get_step(&self) -> DynStep {
        match self {
            Step::Fixed { dr } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

                DynStep::new(SingleStep::new(converter.scalar_value(dr)))
            }
            Step::LocalWavelength {
                dr_min,
                dr_max,
                wave_ratio,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let dr_min = converter.scalar_value(dr_min);
                let dr_max = converter.scalar_value(dr_max);

                DynStep::new(LocalWavelengthStep::new(dr_min, dr_max, *wave_ratio))
            }
            Step::Transitioned {
                transition_point,
                before,
                after,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let transition_point = converter.scalar_value(transition_point);

                DynStep::new(TransitionStep {
                    r_switch: transition_point,
                    step_short: before.get_step(),
                    step_long: after.get_step(),
                })
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum Boundary {
    #[default]
    VanishingWkb,
    Set {
        value: Scalar<Power<Length, -1, 2>>,
        derivative: Scalar<Power<Length, -3, 2>>,
    },
}

impl Boundary {
    pub fn get_boundary(
        &self,
        r_start: &Scalar<Length>,
        direction: Direction,
        w_matrix: &CollisionWMatrix<impl RCoupling>,
    ) -> coupled_chan::cc_propagator::Boundary<Matrix> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r = converter.scalar_value(r_start);

        let (value, derivative) = match &self {
            Boundary::VanishingWkb => {
                let mut mat = Matrix::zeros(w_matrix.size(), w_matrix.size());
                w_matrix.value_inplace(r, &mut mat);
                let eigen = mat
                    .self_adjoint_eigen(hilbert_space::faer::Side::Lower)
                    .expect("Could not diagonalize w_matrix at r_start");

                let derivative = match direction {
                    Direction::Inwards => {
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| -x.abs().sqrt()));
                        let diag = diag.as_diagonal();

                        eigen.U() * diag * eigen.U().transpose()
                    }
                    Direction::Outwards => {
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| x.abs().sqrt()));
                        let diag = diag.as_diagonal();

                        eigen.U() * diag * eigen.U().transpose()
                    }
                };

                (w_matrix.id().to_owned(), derivative)
            }
            Boundary::Set { value, derivative } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let value = converter.scalar_value(value);
                let derivative = converter.scalar_value(derivative);

                (value * w_matrix.id(), derivative * w_matrix.id())
            }
        };

        coupled_chan::cc_propagator::Boundary {
            r_start: r,
            direction,
            value,
            derivative,
        }
    }
}

pub struct ScatteringCalc<P> {
    pub mass: TypedParamId<Scalar<Mass>>,
    phantom: PhantomData<P>,
}

impl<P: Problem> ScatteringCalc<P> {
    pub fn new(mass_id: TypedParamId<Scalar<Mass>>) -> Self {
        Self {
            mass: mass_id,
            phantom: PhantomData,
        }
    }
}

impl<P: Problem> SingleCalc for ScatteringCalc<P> {
    type P = P;
    type CalcInput = ScatteringCalcInput;
    type Data = SMatrixData;

    fn calculate(
        &self,
        modified: Modified<P, ScatteringCalcInput>,
        _problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<SMatrixData>> {
        let registry = modified.system.param_registry();
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        let blocks = modified.system.angular_blocks();
        let collision_params = CollisionParams {
            mass: converter.scalar_value(registry.get(self.mass)),
            energy: converter.scalar_value(&modified.calc_input.energy),
            entrance: modified.calc_input.entrance,
        };
        let asymptote = Asymptote::new_angular_blocks(blocks, collision_params);

        let w_matrix = CollisionWMatrix::new(modified.system.coupling(), asymptote);

        [Ok(modified.calc_input.scattering(&w_matrix))]
    }
}

pub type ScatteringScan<P, D> = DependenceCalc<ScatteringCalc<P>, D>;
