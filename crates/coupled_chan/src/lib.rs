pub mod coupling;
pub mod log_derivative;
pub mod ratio_numerov;
pub mod s_matrix;

pub use constants;
use faer::Mat;
pub use propagator;
use propagator::{Boundary, Direction, Propagator, Repr, step_strategy::StepStrategy};
pub use single_chan::interaction::*;

use crate::coupling::WMatrix;

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;

const DERIVATIVE_VANISHING: f64 = 1.;
const VALUE_VANISHING: f64 = 1e-50;

pub fn vanishing_boundary(r_start: f64, direction: Direction, w_matrix: &impl WMatrix) -> Boundary<Operator> {
    let sign = match direction {
        Direction::Inwards => -1.,
        Direction::Outwards => 1.,
    };

    Boundary {
        r_start,
        direction,
        value: Operator::new(VALUE_VANISHING * &w_matrix.id().0),
        derivative: Operator::new(sign * DERIVATIVE_VANISHING * &w_matrix.id().0),
    }
}

pub trait CoupledPropagator<'a, W: WMatrix, R: Repr>: Propagator<R> {
    fn get_propagator(w_matrix: &'a W, step: StepStrategy, boundary: Boundary<Operator>) -> Self;
}

#[cfg(test)]
mod tests {
    use constants::units::atomic_units::{AuEnergy, AuMass, Bohr, Kelvin};
    use faer::mat;
    use math_utils::assert_approx_eq;
    use propagator::{Boundary, Direction, Propagator, step_strategy::LocalWavelengthStep};
    use single_chan::interaction::{dispersion::lennard_jones, func_potential::FuncPotential};

    use crate::{
        Operator,
        coupling::{Asymptote, Levels, RedCoupling, VanishingCoupling, diagonal::Diagonal, masked::Masked, pair::Pair},
        log_derivative::diabatic::{JohnsonLogDerivative, ManolopoulosLogDerivative},
        ratio_numerov::RatioNumerov,
        s_matrix::SMatrixGetter,
    };

    pub fn get_red_coupling() -> RedCoupling<impl VanishingCoupling> {
        let potential_lj1 = lennard_jones(0.002 * AuEnergy, 9. * Bohr);
        let potential_lj2 = lennard_jones(0.0021 * AuEnergy, 8.9 * Bohr);

        let k = (10. * Kelvin).to(AuEnergy).value();
        let x0 = (11. * Bohr).value();
        let sigma = (2. * Bohr).value();

        let coupling = FuncPotential::new(move |x| k * f64::exp(-0.5 * ((x - x0) / sigma).powi(2)));

        let coupling = Masked::new(coupling, Operator::new(mat![[0., 1.], [1., 0.]]));
        let potential = Diagonal::new(vec![potential_lj1, potential_lj2]);

        let coupling = Pair::new(potential, coupling);

        let asymptote = Asymptote::new_diagonal(
            5903.538543342382 * AuMass,
            (1e-7 * Kelvin).to(AuEnergy),
            Levels {
                l: vec![0, 0],
                asymptote: vec![0., (1. * Kelvin).to(AuEnergy).value()],
            },
            0,
        );

        RedCoupling::new(coupling, asymptote)
    }

    // values at which the result were correct.
    const SCATTERING_LENGTH_RE: f64 = -36.99816556437914;
    const ELASTIC_CROSS_SECTION: f64 = 1.720171035e4;

    #[test]
    pub fn ratio_numerov_scattering() {
        let red_coupling = get_red_coupling();
        let boundary = Boundary {
            r_start: 6.5,
            direction: Direction::Outwards,
            value: Operator::new(1e-50 * &red_coupling.id.0),
            derivative: Operator::new(1. * &red_coupling.id.0),
        };

        let mut numerov = RatioNumerov::new(&red_coupling, LocalWavelengthStep::default().into(), boundary);

        let solution = numerov.propagate_to(1500.0);
        let s_matrix = solution.get_s_matrix(&red_coupling);

        assert_approx_eq!(s_matrix.get_scattering_length().re, SCATTERING_LENGTH_RE, 1e-4);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), ELASTIC_CROSS_SECTION, 1e-4);
    }

    #[test]
    pub fn johnson_log_deriv_scattering() {
        let red_coupling = get_red_coupling();
        let boundary = Boundary {
            r_start: 6.5,
            direction: Direction::Outwards,
            value: Operator::new(1e-50 * &red_coupling.id.0),
            derivative: Operator::new(1. * &red_coupling.id.0),
        };

        let mut propagator = JohnsonLogDerivative::new(&red_coupling, LocalWavelengthStep::default().into(), boundary);

        let solution = propagator.propagate_to(1500.0);
        let s_matrix = solution.get_s_matrix(&red_coupling);

        assert_approx_eq!(s_matrix.get_scattering_length().re, SCATTERING_LENGTH_RE, 1e-4);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), ELASTIC_CROSS_SECTION, 1e-4);
    }

    #[test]
    pub fn manolopoulos_log_deriv_scattering() {
        let red_coupling = get_red_coupling();
        let boundary = Boundary {
            r_start: 6.5,
            direction: Direction::Outwards,
            value: Operator::new(1e-50 * &red_coupling.id.0),
            derivative: Operator::new(1. * &red_coupling.id.0),
        };

        let mut propagator = ManolopoulosLogDerivative::new(&red_coupling, LocalWavelengthStep::default().into(), boundary);

        let solution = propagator.propagate_to(1500.0);
        let s_matrix = solution.get_s_matrix(&red_coupling);

        assert_approx_eq!(s_matrix.get_scattering_length().re, SCATTERING_LENGTH_RE, 1e-4);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), ELASTIC_CROSS_SECTION, 1e-4);
    }
}
