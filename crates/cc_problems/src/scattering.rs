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
    calc::SingleCalc,
    dependence::DependenceCalc,
    parameters::TypedParamId,
    system::System,
};
use anyhow::Result;

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
            momenta: s_matrix.momenta().iter().copied().collect(),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Fixed {
        dr: Scalar<Length>,
    },
    LocalWaveLength {
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
            Step::LocalWaveLength {
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    pub fn get_direction(&self) -> (f64, f64, Direction) {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r_start = converter.scalar_value(&self.r_start);
        let r_stop = converter.scalar_value(&self.r_stop);

        if r_start <= r_stop {
            (r_start, r_stop, Direction::Outwards)
        } else {
            (r_stop, r_start, Direction::Inwards)
        }
    }

    pub fn get_boundary(
        &self,
        w_matrix: &CollisionWMatrix<impl RCoupling>,
    ) -> coupled_chan::cc_propagator::Boundary<Matrix> {
        let (r_start, _, direction) = self.get_direction();

        let (value, derivative) = match &self.boundary {
            Boundary::VanishingWkb => {
                let mut mat = Matrix::zeros(w_matrix.size(), w_matrix.size());
                w_matrix.value_inplace(r_start, &mut mat);
                let eigen = mat
                    .self_adjoint_eigen(hilbert_space::faer::Side::Lower)
                    .expect("Couled not diagonalize w_matrix at r_start");

                for v in eigen.S().column_vector().iter() {
                    assert!(
                        *v < 0.0,
                        "Locally open channels at r = {:?}, cannot make boundary prediction based on WKB",
                        self.r_start
                    );
                }

                let derivative = match direction {
                    Direction::Inwards => {
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| -(-x).sqrt()));
                        let diag = diag.as_diagonal();

                        eigen.U() * diag * eigen.U().transpose()
                    }
                    Direction::Outwards => {
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| (-x).sqrt()));
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
            r_start,
            direction,
            value,
            derivative,
        }
    }

    pub fn scattering(&self, w_matrix: &CollisionWMatrix<impl RCoupling>) -> SMatrixData {
        let (r_start, r_stop, direction) = self.get_direction();
        assert!(
            matches!(direction, Direction::Outwards),
            "Scattering calculation should be in outward direction"
        );

        let mut mat = Matrix::zeros(w_matrix.size(), w_matrix.size());
        w_matrix.value_inplace(r_start, &mut mat);
        let values = mat
            .self_adjoint_eigenvalues(hilbert_space::faer::Side::Lower)
            .expect("Couled not diagonalize w_matrix at r_start");

        for v in values {
            assert!(
                v < 0.0,
                "Locally open channels at the r_start = {:?} of scattering calculation",
                self.r_start
            );
        }

        let step = self.step.get_step();
        let boundary = self.get_boundary(w_matrix);

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

pub struct ScatteringCalc {
    pub mass: TypedParamId<Scalar<Mass>>,
}

impl SingleCalc<ScatteringCalcInput, SMatrixData> for ScatteringCalc {
    fn calculate(&self, input: &ScatteringCalcInput, system: &System) -> Result<SMatrixData> {
        let registry = system.param_registry();
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        let blocks = system.angular_blocks();
        let collision_params = CollisionParams {
            mass: converter.scalar_value(registry.get(self.mass)),
            energy: converter.scalar_value(&input.energy),
            entrance: input.entrance,
        };
        let asymptote = Asymptote::new_angular_blocks(blocks, collision_params);

        let w_matrix = CollisionWMatrix::new(system.coupling(), asymptote);

        Ok(input.scattering(&w_matrix))
    }
}

pub type ScatteringScan = DependenceCalc<ScatteringCalcInput, SMatrixData, ScatteringCalc>;
