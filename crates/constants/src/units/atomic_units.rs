use num_traits::real::Real;

use crate::units::Unit;

#[derive(Clone, Copy, Default, Debug)]
pub struct AuEnergy;

#[derive(Clone, Copy, Default, Debug)]
pub struct Kelvin;

impl<T: Real + From<f64>> Unit<T> for Kelvin {
    type Base = AuEnergy;

    const TO_BASE: f64 = 3.1668115634564e-6;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CmInv;

impl<T: Real + From<f64>> Unit<T> for CmInv {
    type Base = AuEnergy;

    const TO_BASE: f64 = 4.5563352529132e-6;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct GHz;

impl<T: Real + From<f64>> Unit<T> for GHz {
    type Base = AuEnergy;

    const TO_BASE: f64 = 1.51982850071586e-07;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MHz;

impl<T: Real + From<f64>> Unit<T> for MHz {
    type Base = AuEnergy;

    const TO_BASE: f64 = 1.51982850071586e-10;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Bohr;

#[derive(Clone, Copy, Default, Debug)]
pub struct Angstrom;

impl<T: Real + From<f64>> Unit<T> for Angstrom {
    type Base = Bohr;

    const TO_BASE: f64 = 1. / 0.529177210544;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AuMass;

#[derive(Clone, Copy, Default, Debug)]
pub struct Dalton;

impl<T: Real + From<f64>> Unit<T> for Dalton {
    type Base = AuMass;

    const TO_BASE: f64 = 1. / 5.485799090441e-4;
}
