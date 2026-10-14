pub trait Step {
    fn get_step(&self, r: f64, local_wavelength: f64) -> f64;
}

pub struct DynStep(Box<dyn Step>);

impl DynStep {
    pub fn new(step: impl Step + 'static) -> Self {
        Self(Box::new(step))
    }
}

impl Step for DynStep {
    fn get_step(&self, r: f64, local_wavelength: f64) -> f64 {
        self.0.get_step(r, local_wavelength)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SingleStep {
    pub dr: f64,
}

impl SingleStep {
    pub fn new(dr: f64) -> Self {
        Self { dr }
    }
}

impl Step for SingleStep {
    fn get_step(&self, _: f64, _: f64) -> f64 {
        self.dr
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalWavelengthStep {
    pub dr_min: f64,
    pub dr_max: f64,
    pub wave_fraction: f64,
}

impl Default for LocalWavelengthStep {
    fn default() -> Self {
        Self {
            dr_min: 0.,
            dr_max: f64::INFINITY,
            wave_fraction: 500.,
        }
    }
}

impl LocalWavelengthStep {
    pub fn new(dr_min: f64, dr_max: f64, wave_fraction: f64) -> Self {
        Self {
            dr_min,
            dr_max,
            wave_fraction,
        }
    }
}

impl Step for LocalWavelengthStep {
    fn get_step(&self, _: f64, local_wavelength: f64) -> f64 {
        f64::clamp(local_wavelength / self.wave_fraction, self.dr_min, self.dr_max)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TransitionStep<S1: Step, S2: Step> {
    pub r_switch: f64,
    pub step_short: S1,
    pub step_long: S2,
}

impl<S1: Step, S2: Step> Step for TransitionStep<S1, S2> {
    fn get_step(&self, r: f64, local_wavelength: f64) -> f64 {
        if r <= self.r_switch {
            self.step_short.get_step(r, local_wavelength)
        } else {
            self.step_long.get_step(r, local_wavelength)
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConstRefStepStrat {
    Fixed {
        dr: f64
    },
    LocalWavelength {
        dr_min: f64,
        dr_max: f64,
        wave_length_ratio: f64
    },
    Transitioned {
        transition_point: f64,
        before: Box<ConstRefStepStrat>,
        after: Box<ConstRefStepStrat>
    }
}

#[derive(Debug, Clone)]
pub enum LinearRefStepStrat {
    Fixed {
        dr: f64
    },
    ErrorEstimated {
        tolerance: f64,
        change_power: f64,
        dr_min: f64,
        dr_max: f64,
    },
    Transitioned {
        transition_point: f64,
        before: Box<LinearRefStepStrat>,
        after: Box<LinearRefStepStrat>
    }
}

impl LinearRefStepStrat {
    pub fn init_step(
        &self, 
        r: f64,
    ) -> f64 {
        match self {
            LinearRefStepStrat::Fixed { dr } => *dr,
            LinearRefStepStrat::ErrorEstimated { tolerance: _, change_power: _, dr_min, dr_max: _ } => {
                *dr_min
            },
            LinearRefStepStrat::Transitioned { transition_point, before, after } => {
                if r < *transition_point {
                    before.init_step(r)
                } else {
                    after.init_step(r)
                }
            },
        }
    }

    pub fn get_step(
        &self, 
        r: f64,
        dr_prev: f64, 
        average_wave_length: f64,
        w_deriv_off_estimate: f64,
        w_dderiv_estimate: Option<f64>
    ) -> f64 {
        match self {
            LinearRefStepStrat::Fixed { dr } => *dr,
            LinearRefStepStrat::ErrorEstimated { tolerance, change_power, dr_min, dr_max } => {
                let error_deriv_off = (average_wave_length * w_deriv_off_estimate * dr_prev.powi(3) / 12.).abs();
                let error = if let Some(w_dderiv_estimate) = w_dderiv_estimate {
                    let error_dderiv = (w_dderiv_estimate * dr_prev.powi(3) / 12.).abs();

                    f64::max(error_dderiv, error_deriv_off)
                } else {
                    error_deriv_off
                };

                let mut factor = (tolerance / error).powf(1. / change_power);
                if dr_prev < 0. {
                    factor *= (1. + dr_prev / r).powi(3)
                }
                let factor = f64::min(2., factor);

                f64::clamp(dr_prev * factor, *dr_min, *dr_max)
            },
            LinearRefStepStrat::Transitioned { transition_point, before, after } => {
                if r < *transition_point {
                    before.get_step(r, dr_prev, average_wave_length, w_deriv_off_estimate, w_dderiv_estimate)
                } else {
                    after.get_step(r, dr_prev, average_wave_length, w_deriv_off_estimate, w_dderiv_estimate)
                }
            },
        }
    }
}