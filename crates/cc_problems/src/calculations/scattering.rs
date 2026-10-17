use std::marker::PhantomData;

use coupled_chan::{
    cc_propagator::{
        self,
        Direction,
        LogDeriv,
        Nodes,
        Solution,
        step_strategy::{
            ConstRefStepStrat, 
            LinearRefStepStrat, 
        },
    },
    coupling::{
        Asymptote,
        CollisionParams,
        CollisionWMatrix,
        RCoupling,
    },
    multi_channel::{
        LogDerivativePropagator,
        Matrix,
        WMatrix,
        airy::AiryLogDerivative,
        diabatic_log_derivative::{
            JohnsonLogDerivative,
            ManolopoulosLogDerivative,
        }, quasiadiabatic_log_derivative::QuasiAdiabaticLogDerivative,
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
        Modified,
        SingleCalc,
        modifications::{
            FloatCalcMod,
            ModifyRegistry,
            ScalarCalcMod,
        },
    },
    modify_recipe,
    parameters::TypedParamId,
    problems::{
        Problem,
        mat_as_nested_vec,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScatteringCalcInput {
    pub entrance: usize,
    pub energy: Scalar<Energy>,

    #[serde(default)]
    pub boundary: Boundary,
    pub r_min: Scalar<Length>,
    pub r_max: Scalar<Length>,

    #[serde(flatten)]
    pub propagator: TwoSegmentPropagator,
}

pub fn scattering_calc_mods<P: Problem + 'static>() -> ModifyRegistry<P, ScatteringCalcInput> {
    ModifyRegistry::from([
        (
            "energy",
            modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut ScatteringCalcInput, v| r.energy = v)),
        ),
        (
            "r_min",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.r_min = v)),
        ),
        (
            "r_max",
            modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.r_max = v)),
        ),
        (
            "step_scaling",
            modify_recipe!(|x| FloatCalcMod::new(x, |r: &mut ScatteringCalcInput, v| r.propagator.scale_step(v))),
        ),
    ])
}

impl ScatteringCalcInput {
    pub fn scattering(&self, w_matrix: &CollisionWMatrix<impl RCoupling>) -> SMatrixData {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r_max = converter.scalar_value(&self.r_min);
        let r_min = converter.scalar_value(&self.r_max);

        assert!(r_max < r_min, "Expected r_min < r_max");

        let mut mat = Matrix::zeros(w_matrix.size(), w_matrix.size());
        w_matrix.value_inplace(r_max, &mut mat);
        let values = mat
            .self_adjoint_eigenvalues(hilbert_space::faer::Side::Lower)
            .expect("Could not diagonalize w_matrix at r_min");

        for v in values {
            assert!(
                v < 0.0,
                "Locally open channels at the r_min = {:?} of scattering calculation",
                self.r_min
            );
        }

        let boundary = self.boundary.get_boundary(&self.r_min, Direction::Outwards, w_matrix);
        let sol = self.propagator.propagate_to(&self.r_max, boundary, w_matrix);

        SMatrixData::new(&SMatrix::from_log_deriv(&sol, w_matrix))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SMatrixData {
    pub s_matrix_vec: Vec<Vec<Complex64>>,
    pub momenta: Vec<f64>,
    pub entrance_nr: usize,
    pub s_length: Complex64,
}

impl SMatrixData {
    pub fn new(s_matrix: &SMatrix) -> Self {
        let s_matrix_mat = s_matrix.s_matrix();
        let vec = mat_as_nested_vec(s_matrix_mat);

        Self {
            s_matrix_vec: vec,
            momenta: s_matrix.momenta().to_vec(),
            entrance_nr: s_matrix.entrance_number(),
            s_length: s_matrix.scattering_length(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum CoupledChanPropagator {
    JohnsonLogDeriv { step: ConstRefStep },
    ManolopoulosLogDeriv { step: ConstRefStep },
    QuasiAdiabaticLogDeriv { step: ConstRefStep },
    AiryLogDeriv { step: LinearRefStep },
}

impl CoupledChanPropagator {
    pub fn get_propagator<'a, W: WMatrix<f64>>(
        &self,
        boundary: cc_propagator::Boundary<Matrix>,
        w_matrix: &'a W,
    ) -> Box<dyn LogDerivativePropagator + 'a> {
        match self {
            CoupledChanPropagator::JohnsonLogDeriv { step } => {
                Box::new(JohnsonLogDerivative::new(w_matrix, step.get_step(), boundary))
            }
            CoupledChanPropagator::ManolopoulosLogDeriv { step } => {
                Box::new(ManolopoulosLogDerivative::new(w_matrix, step.get_step(), boundary))
            }
            CoupledChanPropagator::AiryLogDeriv { step } => {
                Box::new(AiryLogDerivative::new(w_matrix, step.get_step(), boundary))
            }
            CoupledChanPropagator::QuasiAdiabaticLogDeriv { step } => {
                Box::new(QuasiAdiabaticLogDerivative::new(w_matrix, step.get_step(), boundary))
            },
        }
    }

    pub fn transfer_propagator<'a, W: WMatrix<f64>>(
        &self,
        solution: Solution<LogDeriv<Matrix>>,
        nodes: Nodes,
        w_matrix: &'a W,
    ) -> Box<dyn LogDerivativePropagator + 'a> {
        match self {
            CoupledChanPropagator::JohnsonLogDeriv { step } => Box::new(JohnsonLogDerivative::from_log_derivative(
                w_matrix,
                step.get_step(),
                solution,
                nodes,
            )),
            CoupledChanPropagator::ManolopoulosLogDeriv { step } => Box::new(
                ManolopoulosLogDerivative::from_log_derivative(w_matrix, step.get_step(), solution, nodes),
            ),
            CoupledChanPropagator::AiryLogDeriv { step } => Box::new(AiryLogDerivative::from_log_derivative(
                w_matrix,
                step.get_step(),
                solution,
                nodes,
            )),
            CoupledChanPropagator::QuasiAdiabaticLogDeriv { step } => Box::new(
                QuasiAdiabaticLogDerivative::from_log_derivative(w_matrix, step.get_step(), solution, nodes),
            ),
        }
    }

    pub fn scale_step(&mut self, scaling: f64) {
        match self {
            CoupledChanPropagator::JohnsonLogDeriv { step } => step.scale_step(scaling),
            CoupledChanPropagator::ManolopoulosLogDeriv { step } => step.scale_step(scaling),
            CoupledChanPropagator::AiryLogDeriv { step } => step.scale_step(scaling),
            CoupledChanPropagator::QuasiAdiabaticLogDeriv { step } => step.scale_step(scaling),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub struct TwoSegmentPropagator {
    propagator: CoupledChanPropagator,

    #[serde(default)]
    segment_change: Option<Scalar<Length>>,
    #[serde(default)]
    long_range_propagator: Option<CoupledChanPropagator>,
}

impl TwoSegmentPropagator {
    pub fn propagate_to(
        &self,
        r: &Scalar<Length>,
        boundary: cc_propagator::Boundary<Matrix>,
        w_matrix: &impl WMatrix<f64>,
    ) -> Solution<LogDeriv<Matrix>> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r = converter.scalar_value(r);

        match (&self.segment_change, &self.long_range_propagator) {
            (Some(r_change), Some(lr_propagator)) => {
                let r_change = converter.scalar_value(r_change);

                let mut propagator = self.propagator.get_propagator(boundary, w_matrix);
                let nodes = propagator.nodes();
                let solution = propagator.propagate_to(r_change).to_owned();

                lr_propagator
                    .transfer_propagator(solution.to_owned(), nodes, w_matrix)
                    .propagate_to(r)
                    .to_owned()
            }
            _ => self.propagator.get_propagator(boundary, w_matrix).propagate_to(r).to_owned(),
        }
    }

    pub fn scale_step(&mut self, scaling: f64) {
        self.propagator.scale_step(scaling);
        if let Some(lr) = &mut self.long_range_propagator {
            lr.scale_step(scaling);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ConstRefStep {
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
        before: Box<ConstRefStep>,
        after: Box<ConstRefStep>,
    },
}

impl ConstRefStep {
    pub fn get_step(&self) -> ConstRefStepStrat {
        match self {
            ConstRefStep::Fixed { dr } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

                ConstRefStepStrat::Fixed { dr: converter.scalar_value(dr) }
            }
            ConstRefStep::LocalWavelength {
                dr_min,
                dr_max,
                wave_ratio,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let dr_min = converter.scalar_value(dr_min);
                let dr_max = converter.scalar_value(dr_max);

                ConstRefStepStrat::LocalWavelength { dr_min, dr_max, wave_ratio: *wave_ratio }
            }
            ConstRefStep::Transitioned {
                transition_point,
                before,
                after,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let transition_point = converter.scalar_value(transition_point);

                ConstRefStepStrat::Transitioned { 
                    transition_point,
                    before: Box::new(before.get_step()), 
                    after: Box::new(after.get_step()) 
                }
            }
        }
    }

    pub fn scale_step(&mut self, scaling: f64) {
        match self {
            ConstRefStep::Fixed { dr } => dr.scale(scaling),
            ConstRefStep::LocalWavelength {
                dr_min,
                dr_max,
                wave_ratio,
            } => {
                dr_min.scale(scaling);
                dr_max.scale(scaling);
                *wave_ratio /= scaling
            }
            ConstRefStep::Transitioned {
                transition_point: _,
                before,
                after,
            } => {
                before.scale_step(scaling);
                after.scale_step(scaling);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum LinearRefStep {
    Fixed {
        dr: Scalar<Length>,
    },
    ErrorEstimated {
        dr_min: Scalar<Length>,
        dr_max: Scalar<Length>,
        #[serde(default = "default_tolerance_linear_ref")]
        tolerance: f64,
        #[serde(default = "default_change_power_linear_ref")]
        change_power: f64,
    },
    Transitioned {
        transition_point: Scalar<Length>,
        before: Box<LinearRefStep>,
        after: Box<LinearRefStep>,
    },
}

fn default_tolerance_linear_ref() -> f64 {
    1e-9
}
fn default_change_power_linear_ref() -> f64 {
    3.0
}

impl LinearRefStep {
    pub fn get_step(&self) -> LinearRefStepStrat {
        match self {
            LinearRefStep::Fixed { dr } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

                LinearRefStepStrat::Fixed {
                    dr: converter.scalar_value(dr),
                }
            }
            LinearRefStep::ErrorEstimated {
                dr_min,
                dr_max,
                tolerance,
                change_power,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let dr_min = converter.scalar_value(dr_min);
                let dr_max = converter.scalar_value(dr_max);

                LinearRefStepStrat::ErrorEstimated {
                    tolerance: *tolerance,
                    change_power: *change_power,
                    dr_min,
                    dr_max,
                }
            }
            LinearRefStep::Transitioned {
                transition_point,
                before,
                after,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let transition_point = converter.scalar_value(transition_point);

                LinearRefStepStrat::Transitioned {
                    transition_point,
                    before: Box::new(before.get_step()),
                    after: Box::new(after.get_step()),
                }
            }
        }
    }

    pub fn scale_step(&mut self, scaling: f64) {
        match self {
            Self::Fixed { dr } => dr.scale(scaling),
            Self::ErrorEstimated {
                dr_min,
                dr_max,
                tolerance,
                change_power,
            } => {
                dr_min.scale(scaling);
                dr_max.scale(scaling);

                *tolerance *= scaling.powf(*change_power)
            }
            Self::Transitioned {
                transition_point: _,
                before,
                after,
            } => {
                before.scale_step(scaling);
                after.scale_step(scaling);
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
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| -x.max(0.0).sqrt()));
                        let diag = diag.as_diagonal();

                        eigen.U() * diag * eigen.U().transpose()
                    }
                    Direction::Outwards => {
                        let diag = Col::from_iter(eigen.S().column_vector().iter().map(|x| x.max(0.0).sqrt()));
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
        modified: &mut Modified<P, ScatteringCalcInput>,
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
