pub mod coupling;
pub mod log_derivative;
pub mod ratio_numerov;
pub mod s_matrix;

pub use constants;
use faer::Mat;
pub use propagator;
pub use single_chan::interaction::*;

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;

#[cfg(test)]
mod tests {
    use constants::units::{
        Quantity,
        atomic_units::{AuEnergy, AuMass, Bohr, Kelvin},
    };
    use faer::mat;
    use math_utils::assert_approx_eq;
    use propagator::{Boundary, Direction, Propagator, step_strategy::LocalWavelengthStep};
    use single_chan::interaction::{dispersion::lennard_jones, func_potential::FuncPotential};

    use crate::{
        coupling::{diagonal::Diagonal, masked::Masked, pair::Pair, Asymptote, Levels, RedCoupling, VanishingCoupling}, log_derivative::{self, diabatic::{JohnsonLogDerivative, ManolopoulosLogDerivative}}, ratio_numerov::{self, RatioNumerov}, Operator
    };

    pub fn get_red_coupling() -> RedCoupling<impl VanishingCoupling> {
        let potential_lj1 = lennard_jones(Quantity(0.002, AuEnergy), Quantity(9., Bohr));
        let potential_lj2 = lennard_jones(Quantity(0.0021, AuEnergy), Quantity(8.9, Bohr));

        let k = Quantity(10., Kelvin).to(AuEnergy).value();
        let x0 = Quantity(11., Bohr).value();
        let sigma = Quantity(2., Bohr).value();

        let coupling = FuncPotential::new(move |x| k * f64::exp(-0.5 * ((x - x0) / sigma).powi(2)));

        let coupling = Masked::new(coupling, Operator::new(mat![[0., 1.], [1., 0.]]));
        let potential = Diagonal::new(vec![potential_lj1, potential_lj2]);

        let coupling = Pair::new(potential, coupling);

        let asymptote = Asymptote::new_diagonal(
            Quantity(5903.538543342382, AuMass),
            Quantity(1e-7, Kelvin).to(AuEnergy),
            Levels {
                l: vec![0, 0],
                asymptote: vec![0., Quantity(1., Kelvin).to(AuEnergy).value()],
            },
            0,
        );

        RedCoupling::new(coupling, asymptote)
    }

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
        let s_matrix = ratio_numerov::get_s_matrix(solution, &red_coupling);

        // values at which the result were correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -37.07176, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -1.550004e-12, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 1.7270067e4, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 4.1425318e-23, 1e-6);
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
        let s_matrix = log_derivative::get_s_matrix(solution, &red_coupling);

        // values at which the result were correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -37.07176, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -1.550004e-12, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 1.7270067e4, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 4.1425318e-23, 1e-6);
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
        let s_matrix = log_derivative::get_s_matrix(solution, &red_coupling);

        // values at which the result were correct.
        assert_approx_eq!(s_matrix.get_scattering_length().re, -37.07176, 1e-6);
        assert_approx_eq!(s_matrix.get_scattering_length().im, -1.550004e-12, 1e-6);
        assert_approx_eq!(s_matrix.get_elastic_cross_sect(), 1.7270067e4, 1e-6);
        assert_approx_eq!(s_matrix.get_inelastic_cross_sect(), 4.1425318e-23, 1e-6);
    }
}
