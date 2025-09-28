use crate::units::*;

pub mod units;

pub const BOHR_MAG: Quantity<Frac<AuEnergy, Gauss>> = Quantity(0.5 / 2.350517567e9, Frac(AuEnergy, Gauss));

pub const NUCLEAR_MAG: Quantity<Frac<AuEnergy, Gauss>> = Quantity(0.5 / 1836.0 / 2.350517567e9, Frac(AuEnergy, Gauss));

pub const G_FACTOR: f64 = 2.00231930436256;
