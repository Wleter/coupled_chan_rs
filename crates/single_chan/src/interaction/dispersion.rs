use constants::units::{
    Quantity64,
    atomic_units::{AuEnergy, AuMass, Bohr},
};

use crate::interaction::{Interaction, composite::Composite};

/// Potential of the form d0 * r^n
#[derive(Debug, Clone)]
pub struct Dispersion {
    pub d0: f64,
    pub n: i32,
}

impl Dispersion {
    pub fn new(d0: f64, n: i32) -> Self {
        Self { d0, n }
    }
}

impl Interaction for Dispersion {
    fn value(&self, r: f64) -> f64 {
        self.d0 * r.powi(self.n)
    }

    fn asymptote(&self) -> f64 {
        if self.n == 0 {
            self.d0
        } else if self.n > 0 {
            f64::INFINITY
        } else {
            0.
        }
    }
}

pub fn lennard_jones(d6: Quantity64<AuEnergy>, r6: Quantity64<Bohr>) -> Composite<Dispersion> {
    let d6 = d6.value();
    let r6 = r6.value();
    let c12 = d6 * r6.powi(12);
    let c6 = -2.0 * d6 * r6.powi(6);

    Composite::new(vec![Dispersion::new(c12, -12), Dispersion::new(c6, -6)])
}

pub fn centrifugal_barrier(l: u32, red_mass: Quantity64<AuMass>) -> Option<Dispersion> {
    if l == 0 {
        return None;
    }

    Some(Dispersion::new((l * (l + 1)) as f64 / (2. * red_mass.value()), -2))
}
