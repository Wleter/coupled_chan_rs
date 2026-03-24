use std::f64::consts::PI;

use cc_math_utils::bessel::{
    riccati_j,
    riccati_j_deriv,
    riccati_n,
    riccati_n_deriv,
};
use cc_propagator::{
    LogDeriv,
    Ratio,
    Solution,
    single_channel::WFunction,
};
use num_complex::Complex64;

use crate::interaction::{
    AsymptoteDep,
    CollisionWFunction,
    Interaction,
};

#[derive(Debug, Clone, Copy)]
pub struct SValue {
    pub value: Complex64,
    pub momentum: f64,
}

impl SValue {
    pub fn from_ratio(sol: &Solution<Ratio<f64>>, w_function: &CollisionWFunction<impl Interaction>) -> Self {
        if let Err(err) = scattering_suitable(w_function.interaction_asymptote_dep()) {
            if let ScatteringSuitable::WrongDependence = err {
                panic!("{err}")
            }
        }

        let r_last = sol.r;
        let r_prev_last = sol.r - sol.dr;

        let f_last = 1. + sol.dr * sol.dr / 12. * w_function.value(r_last);
        let f_prev_last = 1. + sol.dr * sol.dr / 12. * w_function.value(r_prev_last);

        let wave_ratio = 1. / f_last * sol.sol.0 * f_prev_last;

        let asymptote = w_function.value_interaction(r_last);
        let l = w_function.l();

        let momentum = asymptote.sqrt();
        if momentum.is_nan() {
            panic!("propagated in closed channel");
        }

        let j_last = riccati_j(l, momentum * r_last);
        let j_prev_last = riccati_j(l, momentum * r_prev_last);
        let n_last = riccati_n(l, momentum * r_last);
        let n_prev_last = riccati_n(l, momentum * r_prev_last);

        let k_matrix = -(wave_ratio * j_prev_last - j_last) / (wave_ratio * n_prev_last - n_last);
        let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        SValue {
            value: s_matrix,
            momentum,
        }
    }

    pub fn from_log_deriv(sol: &Solution<LogDeriv<f64>>, w_function: &CollisionWFunction<impl Interaction>) -> Self {
        if let Err(err) = scattering_suitable(w_function.interaction_asymptote_dep()) {
            if let ScatteringSuitable::WrongDependence = err {
                panic!("{err}")
            }
        }

        let r = sol.r;
        let log_deriv = sol.sol.0;

        let red_asymptote = w_function.value_interaction(r);
        let l = w_function.l();

        let momentum = red_asymptote.sqrt();
        if momentum.is_nan() {
            panic!("propagated in closed channel");
        }

        let (j, j_deriv) = riccati_j_deriv(l, momentum * r);
        let (n, n_deriv) = riccati_n_deriv(l, momentum * r);

        let k_matrix = -(log_deriv * j - j_deriv) / (log_deriv * n - n_deriv);
        let s_matrix = Complex64::new(1.0, k_matrix) / Complex64::new(1.0, -k_matrix);

        SValue {
            value: s_matrix,
            momentum,
        }
    }

    pub fn scattering_length(&self) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - self.value) / (1.0 + self.value)
    }

    pub fn elastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.value).norm_sqr()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ScatteringSuitable {
    WrongDependence,
    UnknownSuitability,
}

impl std::fmt::Display for ScatteringSuitable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScatteringSuitable::WrongDependence => write!(f, "Interaction is not suitable for scattering calculation"),
            ScatteringSuitable::UnknownSuitability => write!(f, "Could not check for suitability for scattering"),
        }
    }
}

impl std::error::Error for ScatteringSuitable {}

pub fn scattering_suitable(dep: AsymptoteDep) -> Result<(), ScatteringSuitable> {
    match dep {
        AsymptoteDep::Const => Ok(()),
        AsymptoteDep::ExpVanishing => Ok(()),
        AsymptoteDep::PowerLawVanishing(x) if x > 2 => Ok(()),
        AsymptoteDep::Growing => Err(ScatteringSuitable::WrongDependence),
        AsymptoteDep::Unknown => Err(ScatteringSuitable::UnknownSuitability),
        _ => Err(ScatteringSuitable::WrongDependence),
    }
}
