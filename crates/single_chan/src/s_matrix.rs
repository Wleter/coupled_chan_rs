use std::f64::consts::PI;

use num_complex::Complex64;

#[derive(Clone, Debug)]
pub struct SMatrix {
    s_matrix: Complex64,
    momentum: f64,
}

impl SMatrix {
    pub fn new(s_matrix: Complex64, momentum: f64) -> Self {
        Self { s_matrix, momentum }
    }

    pub fn get_scattering_length(&self) -> Complex64 {
        1.0 / Complex64::new(0.0, self.momentum) * (1.0 - self.s_matrix) / (1.0 + self.s_matrix)
    }

    pub fn get_elastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix).norm_sqr()
    }

    pub fn get_inelastic_cross_sect(&self) -> f64 {
        PI / self.momentum.powi(2) * (1.0 - self.s_matrix.norm()).powi(2)
    }
}
