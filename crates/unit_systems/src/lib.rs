use crate::dimension::{
    Dimension,
    Ratio8,
};

#[cfg(feature = "au")]
pub mod atomic_units;
pub mod dimension;
pub mod quantities;
#[cfg(feature = "au")]
pub use atomic_units::*;

pub fn pow_ratio(value: f64, exp: Ratio8) -> f64 {
    value.powf(*exp.numer() as f64 / *exp.denom() as f64)
}

#[derive(Clone, Copy, Debug)]
/// Unit system definition through conversion
/// table to the SI unit system
pub struct UnitSystemTable {
    /// mass of 1 kilogram
    pub mass_kg: f64,

    /// length of 1 meter
    pub length_m: f64,

    /// time of 1 second
    pub time_s: f64,

    /// Charge of 1 Coulomb
    pub charge_c: f64,

    /// Temperature of 1 Kelvin
    pub temperature_k: f64,

    /// Amount of 1 mol of matter
    pub amount_mol: f64,

    /// Luminous-intensity of 1 Candela.
    pub luminous_intensity_cd: f64,
}

impl UnitSystemTable {
    pub fn as_si(&self, dimension: Dimension) -> f64 {
        pow_ratio(self.mass_kg, dimension.mass)
            * pow_ratio(self.length_m, dimension.length)
            * pow_ratio(self.time_s, dimension.time)
            * pow_ratio(self.charge_c, dimension.charge)
            * pow_ratio(self.temperature_k, dimension.temperature)
            * pow_ratio(self.amount_mol, dimension.amount)
            * pow_ratio(self.luminous_intensity_cd, dimension.luminous_intensity)
    }

    pub fn from_si(&self, dimension: Dimension) -> f64 {
        1. / self.as_si(dimension)
    }
}

impl Default for UnitSystemTable {
    /// Default conversion from SI to SI
    fn default() -> Self {
        Self {
            mass_kg: 1.,
            length_m: 1.,
            time_s: 1.,
            charge_c: 1.,
            temperature_k: 1.,
            amount_mol: 1.,
            luminous_intensity_cd: 1.,
        }
    }
}

pub const SI_TABLE: UnitSystemTable = UnitSystemTable {
    mass_kg: 1.,
    length_m: 1.,
    time_s: 1.,
    charge_c: 1.,
    temperature_k: 1.,
    amount_mol: 1.,
    luminous_intensity_cd: 1.,
};

#[derive(Debug, Clone, Copy)]
pub struct FundamentalConstantsSI {
    pub speed_of_light: f64,
    pub reduced_planck: f64,
    pub elementary_charge: f64,
    pub electron_mass: f64,
    pub electric_vacuum_permittivity: f64,
    pub boltzmann_constant: f64,
}

pub const CODATA_2022: FundamentalConstantsSI = FundamentalConstantsSI {
    speed_of_light: 299_792_458.0,
    reduced_planck: 1.054_571_817e-34,
    elementary_charge: 1.602_176_634e-19,
    electron_mass: 9.109_383_713_9e-31,
    electric_vacuum_permittivity: 8.854_187_818_8e-12,
    boltzmann_constant: 1.380_649e-23,
};
