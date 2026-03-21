use crate::{
    Boundary,
    Direction,
    Propagator,
    Ratio,
    Solution,
    WaveStorage,
    WithWaveStorage,
    single_channel::{
        WFunction,
        local_wavelength,
    },
    step_strategy::Step,
};

/// doi: 10.1063/1.435384
pub struct RatioNumerov<'a, W: WFunction, S: Step> {
    w_function: &'a W,
    step: S,

    solution: Solution<Ratio<f64>>,
    wave_storage: Option<WaveStorage<f64>>,

    f: f64,
    f_last: f64,
    f_prev_last: f64,
    prev_sol: Ratio<f64>,

    w_function_buffer: f64,
}

impl<'a, W: WFunction, S: Step> RatioNumerov<'a, W, S> {
    pub fn new(w_function: &'a W, step: S, boundary: Boundary<f64>) -> Self {
        let r = boundary.r_start;

        let red_pot = w_function.value(r);
        let local_wavelength = local_wavelength(red_pot);

        let dr = match boundary.direction {
            Direction::Inwards => -(step.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step.get_step(r, local_wavelength).abs(),
        };

        let f_prev_last = 1. + dr * dr / 12. * w_function.value(r - 2. * dr);
        let f_last = 1. + dr * dr / 12. * w_function.value(r - dr);
        let f = 1. + dr * dr / 12. * red_pot;

        let sol = Ratio(f * (boundary.derivative * dr + boundary.value) / boundary.value / f_last);
        let prev_sol = Ratio(f_last * boundary.value / (boundary.value - boundary.derivative * dr) / f_prev_last);

        Self {
            w_function,
            step,
            solution: Solution { r, dr, sol },
            wave_storage: None,
            f,
            f_last,
            f_prev_last,
            prev_sol,

            w_function_buffer: red_pot,
        }
    }

    fn halve_the_step(&mut self) {
        self.solution.dr /= 2.;

        self.solution.sol.0 *= self.f_last / self.f;
        self.f = self.f / 4.0 + 0.75;
        self.f_last = self.f_last / 4.0 + 0.75;
        self.solution.sol.0 *= self.f / self.f_last;

        let f_last =
            1.0 + self.solution.dr * self.solution.dr * self.w_function.value(self.solution.r - self.solution.dr) / 12.0;
        let u = 12.0 / f_last - 10.0;

        let sol_half = (self.solution.sol.0 + 1.) / u;

        self.f_prev_last = self.f_last;
        self.f_last = f_last;

        self.prev_sol.0 = sol_half;
        self.solution.sol.0 /= sol_half;
    }

    fn double_the_step(&mut self) {
        self.solution.dr *= 2.0;
        self.solution.sol.0 *= self.prev_sol.0;

        self.solution.sol.0 *= self.f_prev_last / self.f;

        self.f = 4.0 * self.f_last - 3.0;
        self.f_last = 4.0 * self.f_prev_last - 3.0;

        self.solution.sol.0 *= self.f / self.f_last;
    }

    fn perform_step(&mut self) {
        self.solution.r += self.solution.dr;

        let f_new = 1.0 + self.solution.dr * self.solution.dr * self.w_function_buffer / 12.0;
        let u = 12.0 / self.f - 10.0;
        let sol_new = u - 1. / self.solution.sol.0;

        self.prev_sol = self.solution.sol;
        self.solution.sol.0 = sol_new;

        self.f_prev_last = self.f_last;
        self.f_last = self.f;
        self.f = f_new;

        if let Some(w) = &mut self.wave_storage {
            w.push(self.solution.r, &(f_new / self.f / self.solution.sol.0));
        }
    }
}

impl<W: WFunction, S: Step> Propagator<Ratio<f64>> for RatioNumerov<'_, W, S> {
    fn step(&mut self) -> &Solution<Ratio<f64>> {
        let wavelength = local_wavelength(self.w_function_buffer);
        let dr = self.step.get_step(self.solution.r, wavelength);

        if dr > 2.0 * self.solution.dr.abs() {
            self.double_the_step()
        }

        while dr < self.solution.dr.abs() {
            self.halve_the_step();
        }

        self.w_function_buffer = self.w_function.value(self.solution.r + self.solution.dr);
        self.perform_step();

        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<Ratio<f64>> {
        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
            self.step();
        }

        &self.solution
    }
}

impl<W: WFunction, S: Step> WithWaveStorage<f64> for RatioNumerov<'_, W, S> {
    fn init_wave_storage(&mut self) {
        self.wave_storage = Some(WaveStorage::default())
    }

    fn get_wave_storage(&self) -> Option<&WaveStorage<f64>> {
        self.wave_storage.as_ref()
    }
}
