use std::f64::consts::PI;

use crate::{
    interaction::{Interaction, RedInteraction},
    step_strategy::StepStrategy,
};
use propagator::{Boundary, Direction, Propagator, Ratio, Solution, propagator_watcher::PropagatorWatcher};

/// doi: 10.1063/1.435384
pub struct RatioNumerov<'a, P: Interaction> {
    red_interaction: RedInteraction<'a, P>,
    step: StepStrategy,

    solution: Solution<Ratio<f64>>,

    f: f64,
    f_last: f64,
    f_prev_last: f64,
    prev_sol: Ratio<f64>,

    red_potential_buffer: f64,
    watchers: Option<Vec<&'a mut dyn PropagatorWatcher<Ratio<f64>>>>,
}

impl<'a, P: Interaction> RatioNumerov<'a, P> {
    pub fn new(red_potential: RedInteraction<'a, P>, step: StepStrategy, boundary: Boundary<f64>) -> Self {
        let r = boundary.r_start;

        let red_pot = red_potential.value(r);
        let local_wavelength = get_wavelength(red_pot);

        let dr = match boundary.direction {
            Direction::Inwards => -(step.get_step(r, local_wavelength).abs()),
            Direction::Outwards => step.get_step(r, local_wavelength).abs(),
        };

        let f_prev_last = 1. + dr * dr / 12. * red_potential.value(r - 2. * dr);
        let f_last = 1. + dr * dr / 12. * red_potential.value(r - dr);
        let f = 1. + dr * dr / 12. * red_pot;

        let sol = Ratio(f * (boundary.derivative * dr + boundary.value) / boundary.value / f_last);
        let prev_sol = Ratio(f_last * boundary.value / (boundary.value - boundary.derivative * dr) / f_prev_last);

        Self {
            red_interaction: red_potential,
            step,
            solution: Solution { r, dr, sol },
            f,
            f_last,
            f_prev_last,
            prev_sol,

            red_potential_buffer: red_pot,
            watchers: None,
        }
    }

    pub fn add_watcher(&mut self, watcher: &'a mut impl PropagatorWatcher<Ratio<f64>>) {
        if let Some(watchers) = &mut self.watchers {
            watchers.push(watcher)
        } else {
            self.watchers = Some(vec![watcher])
        }
    }

    pub fn set_watchers(&mut self, watchers: Vec<&'a mut dyn PropagatorWatcher<Ratio<f64>>>) {
        self.watchers = Some(watchers)
    }

    pub fn remove_watchers(&mut self) {
        self.watchers = None
    }

    pub fn change_step_strategy(&mut self, step: StepStrategy) {
        self.step = step
    }

    fn halve_the_step(&mut self) {
        self.solution.dr /= 2.;

        self.solution.sol.0 *= self.f_last / self.f;
        self.f = self.f / 4.0 + 0.75;
        self.f_last = self.f_last / 4.0 + 0.75;
        self.solution.sol.0 *= self.f / self.f_last;

        let f_last =
            1.0 + self.solution.dr * self.solution.dr * self.red_interaction.value(self.solution.r - self.solution.dr);
        let u = (12.0 - 10.0 * f_last) / f_last;

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

        let f = 1.0 + self.solution.dr * self.solution.dr * self.red_potential_buffer / 12.0;
        let u = (12.0 - 10.0 * f) / f;
        let sol_new = u - 1. / self.solution.sol.0;

        self.prev_sol = self.solution.sol;
        self.solution.sol.0 = sol_new;

        self.f_prev_last = self.f_last;
        self.f_last = self.f;
        self.f = f;
    }
}

impl<P: Interaction> Propagator<Ratio<f64>> for RatioNumerov<'_, P> {
    fn step(&mut self) -> &Solution<Ratio<f64>> {
        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.before_step(&self.solution);
            }
        }

        let red_pot = self.red_interaction.value(self.solution.r);
        self.red_potential_buffer = red_pot;
        let wavelength = get_wavelength(red_pot);

        let dr = self.step.get_step(self.solution.r, wavelength);

        if dr > 2.0 * self.solution.dr.abs() {
            self.double_the_step()
        }

        while dr < self.solution.dr.abs() {
            self.halve_the_step();
        }

        self.perform_step();

        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.after_step(&self.solution);
            }
        }
        &self.solution
    }

    fn propagate_to(&mut self, r: f64) -> &Solution<Ratio<f64>> {
        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.init(&self.solution);
            }
        }

        while (self.solution.r - r) * self.solution.dr.signum() <= 0. {
            self.step();
        }

        if let Some(watchers) = &mut self.watchers {
            for w in watchers {
                w.finalize(&self.solution);
            }
        }
        &self.solution
    }
}

#[inline]
fn get_wavelength(red_pot: f64) -> f64 {
    2. * PI / red_pot
}
