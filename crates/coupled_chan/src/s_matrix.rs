use std::f64::consts::PI;

use faer::{Mat, MatRef, c64};
use num_complex::Complex64;

use crate::coupling::WMatrix;

pub trait SMatrixGetter {
    fn get_s_matrix(&self, w_matrix: &impl WMatrix) -> SMatrix;
}

#[derive(Debug, Clone)]
pub struct SMatrix {
    s_matrix: Mat<c64>,
    momenta: Vec<f64>,
    entrance: usize,
}

impl SMatrix {
    pub fn new(s_matrix: Mat<c64>, momenta: Vec<f64>, entrance: usize) -> Self {
        Self {
            s_matrix,
            momenta,
            entrance,
        }
    }

    pub fn s_matrix(&self) -> MatRef<'_, c64> {
        self.s_matrix.as_ref()
    }

    pub fn entrance_number(&self) -> usize {
        self.entrance
    }

    pub fn entrance_momentum(&self) -> f64 {
        self.momenta[self.entrance]
    }

    pub fn momenta(&self) -> &[f64] {
        &self.momenta
    }

    pub fn get_phase_shift(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];
        0.5 * s_element.arg()
    }

    pub fn get_scattering_length(&self) -> Complex64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        1.0 / Complex64::new(0.0, self.momenta[self.entrance]) * (1.0 - s_element) / (1.0 + s_element)
    }

    pub fn get_elastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element).norm_sqr()
    }

    pub fn get_inelastic_cross_sect(&self) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, self.entrance)];

        PI / self.momenta[self.entrance].powi(2) * (1.0 - s_element.norm_sqr())
    }

    pub fn get_inelastic_cross_sect_to(&self, channel: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(self.entrance, channel)];

        PI / self.momenta[self.entrance].powi(2) * s_element.norm_sqr()
    }

    pub fn get_elastic_cross_sect_from(&self, in_chan: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(in_chan, in_chan)];

        PI / self.momenta[in_chan].powi(2) * (1.0 - s_element).norm_sqr()
    }

    pub fn get_inelastic_cross_sect_from_to(&self, in_chan: usize, out_chan: usize) -> f64 {
        let s_element: Complex64 = self.s_matrix[(in_chan, out_chan)];

        if in_chan == out_chan {
            PI / self.momenta[in_chan].powi(2) * (1. - s_element.norm_sqr())
        } else {
            PI / self.momenta[in_chan].powi(2) * s_element.norm_sqr()
        }
    }

    pub fn get_inelastic_cross_sects(&self) -> Vec<f64> {
        (0..self.s_matrix.nrows())
            .map(|i| self.get_inelastic_cross_sect_to(i))
            .collect()
    }
}
