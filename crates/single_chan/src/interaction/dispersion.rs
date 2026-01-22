use cc_constants::{
    Unit,
    units::{
        Quantity,
        atomic_units::{AuEnergy, AuMass, Bohr},
    },
};
use cc_qol_utils::Composite;

use crate::interaction::Interaction;

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
}

pub fn lennard_jones(
    d6: Quantity<impl Unit<Base = AuEnergy>>,
    r6: Quantity<impl Unit<Base = Bohr>>,
) -> Composite<Dispersion> {
    let d6 = d6.to(AuEnergy).value();
    let r6 = r6.to(Bohr).value();
    let c12 = d6 * r6.powi(12);
    let c6 = -2.0 * d6 * r6.powi(6);

    Composite::new(vec![Dispersion::new(c12, -12), Dispersion::new(c6, -6)])
}

pub struct Centrifugal {
    pub l: u32,
    d0: f64,
}

impl Centrifugal {
    pub fn new(l: u32, red_mass: Quantity<impl Unit<Base = AuMass>>) -> Self {
        Self {
            l,
            d0: (l * (l + 1)) as f64 / (2. * red_mass.to(AuMass).value()),
        }
    }

    pub fn value(&self, r: f64) -> f64 {
        self.d0 / (r * r)
    }
}
