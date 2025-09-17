use crate::units::Unit;

#[derive(Clone, Copy, Default, Debug)]
pub struct AuEnergy;

impl Unit for AuEnergy {
    type Base = AuEnergy;

    const TO_BASE: f64 = 1.;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Kelvin;

impl Unit for Kelvin {
    type Base = AuEnergy;

    const TO_BASE: f64 = 3.1668115634564e-6;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct CmInv;

impl Unit for CmInv {
    type Base = AuEnergy;

    const TO_BASE: f64 = 4.5563352529132e-6;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct GHz;

impl Unit for GHz {
    type Base = AuEnergy;

    const TO_BASE: f64 = 1.51982850071586e-07;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MHz;

impl Unit for MHz {
    type Base = AuEnergy;

    const TO_BASE: f64 = 1.51982850071586e-10;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Bohr;

impl Unit for Bohr {
    type Base = Bohr;

    const TO_BASE: f64 = 1.;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Angstrom;

impl Unit for Angstrom {
    type Base = Bohr;

    const TO_BASE: f64 = 1. / 0.529177210544;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct AuMass;

impl Unit for AuMass {
    type Base = AuMass;

    const TO_BASE: f64 = 1.;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Dalton;

impl Unit for Dalton {
    type Base = AuMass;

    const TO_BASE: f64 = 1. / 5.485799090441e-4;
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Gauss;

impl Unit for Gauss {
    type Base = Gauss;

    const TO_BASE: f64 = 1.;
}
