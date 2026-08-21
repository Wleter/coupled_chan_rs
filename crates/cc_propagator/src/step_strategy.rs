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
