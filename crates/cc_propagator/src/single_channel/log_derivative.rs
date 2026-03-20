use crate::{
    Boundary,
    Direction,
    LogDeriv,
    Nodes,
    Propagator,
    Solution,
    WaveStorage,
    WithNodeCount,
    WithWaveStorage,
    single_channel::{
        WFunction,
        local_wavelength,
    },
    step_strategy::Step,
};

/// doi: https://doi.org/10.1016/0010-4655(94)00133-M
pub struct LogDerivative<'a, W: WFunction, S: Step> {
    w_function: &'a W,
    step: S,
    nodes: Nodes,
    wave_storage: Option<WaveStorage<f64>>,

    solution: Solution<LogDeriv<f64>>,
    w_function_value: f64,
}

impl<'a, W: WFunction, S: Step> LogDerivative<'a, W, S> {
    pub fn new(w_function: &'a W, step: S, boundary: Boundary<f64>) -> Self {
        let r = boundary.r_start;

        let w_function_value = w_function.value(r);
        let local_wavelength = local_wavelength(w_function_value);

        let dr = match boundary.direction {
            Direction::Inwards => -(step.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step.get_step(r, local_wavelength).abs(),
        };

        let sol = LogDeriv(boundary.derivative / boundary.value);

        Self {
            w_function,
            step,
            solution: Solution { r, dr, sol },
            nodes: Nodes(0),
            wave_storage: None,

            w_function_value,
        }
    }

    fn step_r_target(&mut self, r: Option<f64>) {
        let wavelength = local_wavelength(self.w_function_value);

        let dr_new = self.step.get_step(self.solution.r, wavelength);
        self.solution.dr = dr_new.clamp(0., 2. * self.solution.dr.abs()) * self.solution.dr.signum();

        if let Some(r) = r
            && (self.solution.r - r).abs() < self.solution.dr.abs()
        {
            self.solution.dr *= ((self.solution.r - r) / self.solution.dr).abs()
        }

        let h = self.solution.dr / 2.;

        let w_function_half = self.w_function.value(self.solution.r + h);
        let w_function_new = self.w_function.value(self.solution.r + 2. * h);

        let closed = if w_function_half < 0. { true } else { false };
        let k = w_function_half.abs().sqrt();

        let y14_0 = if closed { k / f64::tanh(k * h) } else { k / f64::tan(k * h) };
        let y23_0 = if closed { k / f64::sinh(k * h) } else { k / f64::sin(k * h) };

        let w1_a = w_function_half - self.w_function_value;
        let w1_b = w_function_half - w_function_new;

        let y1_a = y14_0 + h / 3. * w1_a;
        let y4_b = y14_0 + h / 3. * w1_b;
        let y23_ab_2 = y23_0.powi(2);

        let z = 1. / (2. * y14_0);

        let y1 = y1_a - y23_ab_2 * z;
        let y23 = y23_ab_2 * z;
        let y4 = y4_b - y23_ab_2 * z;

        let inversion_term = 1. / (self.solution.sol.0 + y1);
        let sol_new = y4 - (y23).powi(2) * inversion_term;

        if let Some(wave_storage) = &mut self.wave_storage {
            let connection = y23 * inversion_term;

            wave_storage.push(self.solution.r, &connection)
        }

        self.nodes.0 += if self.solution.dr > 0. {
            (inversion_term < 0.) as u64
        } else {
            (inversion_term >= 0.) as u64
        };

        self.solution.sol.0 = sol_new;
        self.w_function_value = w_function_new;
        self.solution.r += self.solution.dr;
    }
}

impl<W: WFunction, S: Step> Propagator<LogDeriv<f64>> for LogDerivative<'_, W, S> {
    fn step(&mut self) -> &Solution<LogDeriv<f64>> {
        self.step_r_target(None);
        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<LogDeriv<f64>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step_r_target(Some(r));
        }

        &self.solution
    }
}

impl<W: WFunction, S: Step> WithNodeCount for LogDerivative<'_, W, S> {
    fn nodes(&self) -> crate::Nodes {
        self.nodes
    }
}

impl<W: WFunction, S: Step> WithWaveStorage<f64> for LogDerivative<'_, W, S> {
    fn init_wave_storage(&mut self) {
        self.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<f64>> {
        self.wave_storage.as_ref()
    }
}
