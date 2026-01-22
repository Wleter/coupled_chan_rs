pub trait Step {
    fn get_step(&self, r: f64, local_wavelength: f64) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct SingleStep {
    pub dr: f64,
}

impl Step for SingleStep {
    fn get_step(&self, _: f64, _: f64) -> f64 {
        self.dr
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShortLongRangeStep {
    pub r_switch: f64,
    pub dr_short: f64,
    pub dr_long: f64,
}

impl Step for ShortLongRangeStep {
    fn get_step(&self, r: f64, _: f64) -> f64 {
        if r > self.r_switch { self.dr_long } else { self.dr_short }
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
