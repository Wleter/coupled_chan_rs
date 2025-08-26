use std::f64::consts::PI;

use crate::{
    interaction::{Interaction, RedInteraction},
    s_matrix::SMatrix,
    step_strategy::StepStrategy,
};
use math_utils::bessel::{riccati_j, riccati_n};
use num_complex::Complex64;
use propagator::{Boundary, Direction, Propagator, Ratio, Solution, propagator_watcher::PropagatorWatcher};

/// doi: 10.1063/1.435384
pub struct RatioNumerov<'a, P: Interaction> {
    red_interaction: &'a RedInteraction<'a, P>,
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
    pub fn new(red_potential: &'a RedInteraction<'a, P>, step: StepStrategy, boundary: Boundary<f64>) -> Self {
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

        while (self.solution.r - r) * self.solution.dr.signum() < 0. {
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

pub fn get_s_matrix<P: Interaction>(sol: &Solution<Ratio<f64>>, red_interaction: &RedInteraction<P>) -> SMatrix {
    let r_last = sol.r;
    let r_prev_last = sol.r - sol.dr;

    let f_last = 1. + sol.dr * sol.dr / 12. * red_interaction.value(r_last);
    let f_prev_last = 1. + sol.dr * sol.dr / 12. * red_interaction.value(r_prev_last);

    let wave_ratio = 1. / f_last * sol.sol.0 * f_prev_last;

    let red_asymptote = red_interaction.asymptote();
    let l = red_interaction.l();

    let momentum = red_asymptote.sqrt();
    if momentum.is_nan() {
        panic!("propagated in closed channel");
    }

    let j_last = riccati_j(l, momentum * r_last) / momentum.sqrt();
    let j_prev_last = riccati_j(l, momentum * r_prev_last) / momentum.sqrt();
    let n_last = riccati_n(l, momentum * r_last) / momentum.sqrt();
    let n_prev_last = riccati_n(l, momentum * r_prev_last) / momentum.sqrt();

    let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);

    let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

    SMatrix::new(s_matrix, momentum)
}

#[inline]
fn get_wavelength(red_pot: f64) -> f64 {
    2. * PI / red_pot.abs().sqrt()
}

#[cfg(test)]
mod tests {
    use constants::units::{
        Quantity,
        atomic_units::{AuEnergy, AuMass, Bohr, Kelvin},
    };
    use math_utils::assert_approx_eq;
    use propagator::{Boundary, Direction, Propagator};

    use crate::{
        interaction::{Level, RedInteraction, dispersion::lennard_jones},
        ratio_numerov::{RatioNumerov, get_s_matrix},
        step_strategy::LocalWavelengthStep,
    };

    #[test]
    pub fn ratio_numerov_scattering() {
        let energy = Quantity(1e-7, Kelvin);
        let mass = Quantity(5903.538543342382, AuMass);

        let potential = lennard_jones(Quantity(0.002, AuEnergy), Quantity(9., Bohr));
        let red_interaction =
            RedInteraction::new(&potential, mass, energy.to(AuEnergy), Level::new(0, Quantity(0., AuEnergy)));

        let boundary = Boundary {
            r_start: 6.5,
            direction: Direction::Outwards,
            value: 1e-50,
            derivative: 1.,
        };

        let mut numerov = RatioNumerov::new(&red_interaction, LocalWavelengthStep::default().into(), boundary);

        let solution = numerov.propagate_to(1500.0);
        let s_matrix = get_s_matrix(solution, &red_interaction);

        // values at which the result were correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -15.55074, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, 7.741696e-13, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 3.038868e3, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 0., 1e-6);
    }
}
