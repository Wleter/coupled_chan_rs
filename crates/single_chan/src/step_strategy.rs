pub enum StepStrategy {
    Fixed(f64),
    ShortLongRange(ShortLongRangeStep),
    LocalWaveLength(LocalWavelengthStep),
    Custom(CustomStep),
}

impl StepStrategy {
    pub fn get_step(&self, r: f64, local_wavelength: f64) -> f64 {
        match self {
            StepStrategy::Fixed(x) => *x,
            StepStrategy::ShortLongRange(s) => s.get_step(local_wavelength),
            StepStrategy::LocalWaveLength(s) => s.get_step(r),
            StepStrategy::Custom(custom_step) => (custom_step.step)(r, local_wavelength),
        }
    }
}

pub struct CustomStep {
    step: Box<dyn Fn(f64, f64) -> f64>,
}

impl CustomStep {
    /// Creates custom step from function f(r, local_wavelength)
    pub fn new(step: impl Fn(f64, f64) -> f64 + 'static) -> Self {
        Self { step: Box::new(step) }
    }
}

pub struct ShortLongRangeStep {
    pub r_switch: f64,
    pub dr_short: f64,
    pub dr_long: f64,
}

impl ShortLongRangeStep {
    pub fn get_step(&self, r: f64) -> f64 {
        if r > self.r_switch { self.dr_long } else { self.dr_short }
    }
}

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

    pub fn get_step(&self, local_wavelength: f64) -> f64 {
        f64::clamp(local_wavelength / self.wave_fraction, self.dr_min, self.dr_max)
    }
}
