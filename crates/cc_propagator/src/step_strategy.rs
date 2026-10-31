#[derive(Debug, Clone)]
pub enum ConstRefStepStrat {
    Fixed {
        dr: f64,
    },
    LocalWavelength {
        dr_min: f64,
        dr_max: f64,
        wave_ratio: f64,
    },
    Transitioned {
        transition_point: f64,
        before: Box<ConstRefStepStrat>,
        after: Box<ConstRefStepStrat>,
    },
}

impl ConstRefStepStrat {
    pub fn init_step(&self, r: f64) -> f64 {
        match self {
            ConstRefStepStrat::Fixed { dr } => *dr,
            ConstRefStepStrat::LocalWavelength {
                dr_min,
                dr_max: _,
                wave_ratio: _,
            } => *dr_min,
            ConstRefStepStrat::Transitioned {
                transition_point,
                before,
                after,
            } => {
                if r < *transition_point {
                    before.init_step(r)
                } else {
                    after.init_step(r)
                }
            }
        }
    }

    pub fn get_step(&self, r: f64, dr_prev: f64, local_wavelength: f64) -> f64 {
        match self {
            ConstRefStepStrat::Fixed { dr } => *dr * dr_prev.signum(),
            ConstRefStepStrat::LocalWavelength {
                dr_min,
                dr_max,
                wave_ratio,
            } => {
                let dr_new = local_wavelength / wave_ratio;
                let dr_new = dr_new.min(2.0 * dr_prev.abs());

                f64::clamp(dr_new, *dr_min, *dr_max) * dr_prev.signum()
            }
            ConstRefStepStrat::Transitioned {
                transition_point,
                before,
                after,
            } => {
                if r < *transition_point {
                    before.get_step(r, dr_prev, local_wavelength)
                } else {
                    after.get_step(r, dr_prev, local_wavelength)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum LinearRefStepStrat {
    Fixed {
        dr: f64,
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
        after: Box<LinearRefStepStrat>,
    },
}

impl LinearRefStepStrat {
    pub fn init_step(&self, r: f64) -> f64 {
        match self {
            LinearRefStepStrat::Fixed { dr } => *dr,
            LinearRefStepStrat::ErrorEstimated {
                tolerance: _,
                change_power: _,
                dr_min,
                dr_max: _,
            } => *dr_min,
            LinearRefStepStrat::Transitioned {
                transition_point,
                before,
                after,
            } => {
                if r < *transition_point {
                    before.init_step(r)
                } else {
                    after.init_step(r)
                }
            }
        }
    }

    pub fn get_step(
        &self,
        r: f64,
        dr_prev: f64,
        average_wave_vector: f64,
        w_deriv_off_estimate: f64,
        w_dderiv_estimate: Option<f64>,
    ) -> f64 {
        match self {
            LinearRefStepStrat::Fixed { dr } => *dr * dr_prev.signum(),
            LinearRefStepStrat::ErrorEstimated {
                tolerance,
                change_power,
                dr_min,
                dr_max,
            } => {
                let error_deriv_off = (average_wave_vector * w_deriv_off_estimate * dr_prev.powi(3) / 12.).abs();
                let error = if let Some(w_dderiv_estimate) = w_dderiv_estimate {
                    let error_dderiv = (w_dderiv_estimate * dr_prev.powi(3) / 12.).abs();

                    f64::max(error_dderiv, error_deriv_off)
                } else {
                    error_deriv_off
                };

                // add average_wave_vector to the factor compared to molscat implementation/airy-log
                // derivative paper, this is to make tolerance dimensionless,
                // error estimates the log-derivative difference from the exact
                // thus we normalize it by the average_wave_vector,
                // this is partially motivated because what we ultimately care
                // is the phase-shift (which is related by the wave-vectors) not the log-derivative
                // matrix itself. todo! possibly check if maximum wave_vector is
                // more appropriate
                let mut factor = (tolerance * average_wave_vector / error).powf(1. / change_power);
                if dr_prev < 0. {
                    factor *= (1. + dr_prev / r).powi(3)
                }
                let factor = f64::min(2., factor);

                f64::clamp(dr_prev.abs() * factor, *dr_min, *dr_max) * dr_prev.signum()
            }
            LinearRefStepStrat::Transitioned {
                transition_point,
                before,
                after,
            } => {
                if r < *transition_point {
                    before.get_step(r, dr_prev, average_wave_vector, w_deriv_off_estimate, w_dderiv_estimate)
                } else {
                    after.get_step(r, dr_prev, average_wave_vector, w_deriv_off_estimate, w_dderiv_estimate)
                }
            }
        }
    }
}
