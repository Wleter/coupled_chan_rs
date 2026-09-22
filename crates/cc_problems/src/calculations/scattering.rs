use std::{collections::HashMap, marker::PhantomData};

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
use serde_json::Number;
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
    UNITS_CONVERTER, calculations::{
        Modified,
        SingleCalc, modifications::{ModificationAction, ModifyParam, ModifyRegistry, ScalarCalcMod},
    }, modify_recipe, parameters::TypedParamId, problems::Problem
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScatteringCalcInput {
    pub entrance: usize,
    pub energy: Scalar<Energy>,

    #[serde(default)]
    pub boundary: Boundary,
    pub r_min: Scalar<Length>,
    pub r_max: Scalar<Length>,
    pub step: Step,
    pub solver: CoupledChanSolver,
}

pub fn scattering_calc_mods<P: Problem + 'static>() -> ModifyRegistry<P, ScatteringCalcInput> {
    ModifyRegistry(HashMap::from([
        ("energy".into(), modify_recipe!(|e| ScalarCalcMod::new(e, |r: &mut ScatteringCalcInput| &mut r.energy))),
        ("r_min".into(), modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput| &mut r.r_min))),
        ("r_max".into(), modify_recipe!(|x| ScalarCalcMod::new(x, |r: &mut ScatteringCalcInput| &mut r.r_max))),
        ("step_scaling".into(), modify_recipe!(|x| StepScalingMod::new(x, |r: &mut ScatteringCalcInput| &mut r.step))),
    ]))
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

        let step = self.step.get_step();
        let boundary = self.boundary.get_boundary(&self.r_min, Direction::Outwards, w_matrix);

        // todo! simplify, log-derivatives should be together
        match &self.solver {
            CoupledChanSolver::RatioNumerov => {
                let mut numerov = RatioNumerov::new(w_matrix, step, boundary);
                let sol = numerov.propagate_to(r_min);

                SMatrixData::new(&SMatrix::from_ratio(sol, w_matrix))
            }
            CoupledChanSolver::JohnsonLogDeriv => {
                let mut log_deriv = JohnsonLogDerivative::new(w_matrix, step, boundary);
                let sol = log_deriv.propagate_to(r_min);

                SMatrixData::new(&SMatrix::from_log_deriv(sol, w_matrix))
            }
            CoupledChanSolver::ManolopoulosLogDeriv => {
                let mut log_deriv = ManolopoulosLogDerivative::new(w_matrix, step, boundary);
                let sol = log_deriv.propagate_to(r_min);

                SMatrixData::new(&SMatrix::from_log_deriv(sol, w_matrix))
            }
        }
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
        let mut vec = vec![vec![]; s_matrix_mat.nrows()];
        for (i, row) in s_matrix_mat.row_iter().enumerate() {
            vec[i] = row.iter().copied().collect();
        }

        Self {
            s_matrix_vec: vec,
            momenta: s_matrix.momenta().to_vec(),
            entrance_nr: s_matrix.entrance_number(),
            s_length: s_matrix.scattering_length(),
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

    pub fn scale_step(&mut self, scaling: f64) {
        match self {
            Step::Fixed { dr } => dr.scale(scaling),
            Step::LocalWavelength { dr_min, dr_max, wave_ratio } => {
                dr_min.scale(scaling);
                dr_max.scale(scaling);
                *wave_ratio /= scaling
            },
            Step::Transitioned { transition_point: _, before, after } => {
                before.scale_step(scaling);
                after.scale_step(scaling); 
            },
        }
    }
}

pub struct StepScalingMod<P, C, F: Fn(&mut C) -> &mut Step> {
    pub scaling: f64,
    pub conversion: F,
    phantom: PhantomData<(P, C)>
}

impl<P, C, F: Fn(&mut C) -> &mut Step> StepScalingMod<P, C, F> {
    pub fn new(scaling: f64, conversion: F) -> Self {
        Self { 
            scaling,
            conversion, 
            phantom: PhantomData 
        }
    }
}

impl<P: Problem, C: Send + Sync, F: Fn(&mut C) -> &mut Step + Send + Sync> ModifyParam for StepScalingMod<P, C, F> {
    type P = P;
    type C = C;

    fn prep_modify(&self, modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        (self.conversion)(modified.calc_input).scale_step(self.scaling);
        ModificationAction::CalcChange
    }

    fn as_number(&self) -> serde_json::Number {
        Number::from_f64(self.scaling).unwrap()
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        self.scaling = number.as_f64().unwrap()
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
