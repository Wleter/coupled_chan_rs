use std::{any::TypeId, collections::HashMap, f64::consts::{PI, TAU}, ops::{Div, Mul}};

use hilbert_space::{faer::traits::num_traits::Pow};
use num_rational::{Ratio};

pub type Ratio8 = Ratio<i8>;

pub fn pow_ratio(value: f64, exp: Ratio8) -> f64 {
    value.powf(*exp.numer() as f64 / *exp.denom() as f64)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub mass: Ratio8,
    pub length: Ratio8,
    pub time: Ratio8,
    pub charge: Ratio8,
    pub temperature: Ratio8,
    pub amount: Ratio8,
    pub luminous_intensity: Ratio8,
}

impl std::fmt::Debug for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dimension {{[M^{} L^{} t^{} e^{} T^{} N^{} J^{}]}}", self.mass, self.length, self.time, self.charge, self.temperature, self.amount, self.luminous_intensity)
    }
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{[M^{} L^{} t^{} e^{} T^{} N^{} J^{}]}}", self.mass, self.length, self.time, self.charge, self.temperature, self.amount, self.luminous_intensity)
    }
}

impl Dimension {
    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0, 0, 0, 0);

    pub const MASS: Self = Self::new(1, 0, 0, 0, 0, 0, 0);
    pub const LENGTH: Self = Self::new(0, 1, 0, 0, 0, 0, 0);
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);
    pub const CHARGE: Self = Self::new(0, 0, 0, 1, 0, 0, 0);
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 0, 1, 0, 0);
    pub const AMOUNT: Self = Self::new(0, 0, 0, 0, 0, 1, 0);
    pub const LUMINOUS_INTENSITY: Self = Self::new(0, 0, 0, 0, 0, 0, 1);

    pub const VELOCITY: Self = Self::new(0, 1, -1, 0, 0, 0, 0);
    pub const ACCELERATION: Self = Self::new(0, 1, -2, 0, 0, 0, 0);
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);
    pub const ENERGY: Self = Self::new(1, 2, -2, 0, 0, 0, 0);
    pub const SURFACE: Self = Self::new(0, 2, 0, 0, 0, 0, 0);
    pub const VOLUME: Self = Self::new(0, 3, 0, 0, 0, 0, 0);
    pub const DENSITY: Self = Self::new(1, -3, 0, 0, 0, 0, 0);

    pub const CURRENT: Self = Self::new(0, 0, -1, 1, 0, 0, 0);
    pub const VOLTAGE: Self = Self::new(1, 2, -2, -1, 0, 0, 0);
    pub const MAGNETIC_FIELD: Self = Self::new(1, 0, -1, -1, 0, 0, 0);
    pub const ELECTRIC_FIELD: Self = Self::new(1, 1, -2, -1, 0, 0, 0);

    pub const MAGNETIC_DIPOLE: Self = Self::new(0, 2, -1, 1, 0, 0, 0);
    pub const ELECTRIC_DIPOLE: Self = Self::new(0, 1, 0, 1, 0, 0, 0);

    pub const fn new_fractional(
        mass: Ratio8,
        length: Ratio8,
        time: Ratio8,
        charge: Ratio8,
        temperature: Ratio8,
        amount: Ratio8,
        luminous_intensity: Ratio8,
    ) -> Self {
        Self {
            mass,
            length,
            time,
            charge,
            temperature,
            amount,
            luminous_intensity,
        }
    }

    pub const fn new(
        mass: i8,
        length: i8,
        time: i8,
        charge: i8,
        temperature: i8,
        amount: i8,
        luminous_intensity: i8,
    ) -> Self {
        Self {
            mass: Ratio::new_raw(mass, 1),
            length: Ratio::new_raw(length, 1),
            time: Ratio::new_raw(time, 1),
            charge: Ratio::new_raw(charge, 1),
            temperature: Ratio::new_raw(temperature, 1),
            amount: Ratio::new_raw(amount, 1),
            luminous_intensity: Ratio::new_raw(luminous_intensity, 1),
        }
    }
}

impl Mul for Dimension {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new_fractional(
            self.mass + rhs.mass,
            self.length + rhs.length,
            self.time + rhs.time,
            self.charge + rhs.charge,
            self.temperature + rhs.temperature,
            self.amount + rhs.amount,
            self.luminous_intensity + rhs.luminous_intensity,
        )
    }
}

impl Div for Dimension {
    type Output = Self;
    
    fn div(self, rhs: Self) -> Self::Output {
        Self::new_fractional(
            self.mass - rhs.mass,
            self.length - rhs.length,
            self.time - rhs.time,
            self.charge - rhs.charge,
            self.temperature - rhs.temperature,
            self.amount - rhs.amount,
            self.luminous_intensity - rhs.luminous_intensity,
        )
    }
}

impl Pow<Ratio8> for Dimension {
    type Output = Self;

    fn pow(self, rhs: Ratio8) -> Self::Output {
        Self::new_fractional(
            self.mass * rhs,
            self.length * rhs,
            self.time * rhs,
            self.charge * rhs,
            self.temperature * rhs,
            self.amount * rhs,
            self.luminous_intensity * rhs,
        )
    }
}

impl Pow<i8> for Dimension {
    type Output = Self;

    fn pow(self, rhs: i8) -> Self::Output {
        Self::new_fractional(
            self.mass * rhs,
            self.length * rhs,
            self.time * rhs,
            self.charge * rhs,
            self.temperature * rhs,
            self.amount * rhs,
            self.luminous_intensity * rhs,
        )
    }
}

#[derive(Clone, Copy, Debug)]
/// Conversion table to the SI unit system
/// for the unit system
pub struct SIConversion {
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

impl SIConversion {
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

impl Default for SIConversion {
    /// Default conversion from SI to SI
    fn default() -> Self {
        Self { 
            mass_kg: 1., 
            length_m: 1., 
            time_s: 1., 
            charge_c: 1., 
            temperature_k: 1., 
            amount_mol: 1., 
            luminous_intensity_cd: 1. 
        }
    }
}

pub const SI_TABLE: SIConversion = SIConversion { 
    mass_kg: 1., 
    length_m: 1., 
    time_s: 1., 
    charge_c: 1., 
    temperature_k: 1., 
    amount_mol: 1., 
    luminous_intensity_cd: 1. 
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

pub const fn atomic_units_table(constants: &FundamentalConstantsSI) -> SIConversion {
    // defining constants
    let hbar = constants.reduced_planck;
    let e = constants.elementary_charge;
    let m_e = constants.electron_mass;
    let permittivity = 4.0 * PI * constants.electric_vacuum_permittivity;

    let bohr = permittivity * hbar * hbar / (m_e * e * e);
    let hartree_energy = hbar * hbar / (m_e * bohr * bohr);
    let atomic_time = hbar / hartree_energy;

    SIConversion {
        mass_kg: m_e,
        length_m: bohr,
        time_s: atomic_time,
        charge_c: e,
        temperature_k: 1.0,
        amount_mol: 1.0,
        luminous_intensity_cd: 1.0,
    }
}

pub const ATOMIC_UNITS_TABLE: SIConversion = atomic_units_table(&CODATA_2022);

pub trait PhysQuantity: std::fmt::Debug + Default {
    fn dimension() -> Dimension;
}

#[macro_export]
macro_rules! phys_quantity {
    ($name:ident, $dimension:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $crate::units::PhysQuantity for $name {
            fn dimension() -> Dimension {
                $dimension
            }
        }
    };
}

phys_quantity!(Mass, Dimension::MASS);
phys_quantity!(Length, Dimension::LENGTH);
phys_quantity!(Time, Dimension::TIME);
phys_quantity!(Charge, Dimension::CHARGE);
phys_quantity!(Energy, Dimension::ENERGY);
phys_quantity!(MagneticField, Dimension::MAGNETIC_FIELD);
phys_quantity!(ElectricField, Dimension::ELECTRIC_FIELD);

#[derive(Clone, Copy, Debug, Default)]
pub struct Frac<L: PhysQuantity, R: PhysQuantity>(pub L, pub R);

impl<L: PhysQuantity, R: PhysQuantity> PhysQuantity for Frac<L, R> {
    fn dimension() -> Dimension {
        L::dimension() / R::dimension()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Prod<L: PhysQuantity, R: PhysQuantity>(pub L, pub R);

impl<L: PhysQuantity, R: PhysQuantity> PhysQuantity for Prod<L, R> {
    fn dimension() -> Dimension {
        L::dimension() * R::dimension()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Quantity<Q: PhysQuantity>(pub f64, pub Q);

impl<Q: PhysQuantity + 'static> Quantity<Q> {
    pub fn from_unit(
        value: InputQuantity, 
        registry: &UnitRegistry,
        system: &SIConversion,
    ) -> Self {
        let dim = Q::dimension();
        let unit = registry.get_unit::<Q>(&value.1);

        let from_si = system.from_si(dim);

        Self(value.0 * unit.to_si * from_si, Q::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Unit {
    pub name: &'static str,
    pub to_si: f64
}

impl Unit {
    pub const fn new(name: &'static str, to_si: f64) -> Self {
        Self {
            name,
            to_si,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UnitRegistry(HashMap<TypeId, Vec<Unit>>);

impl UnitRegistry {
    pub fn insert<Q: PhysQuantity + 'static>(&mut self, _quantity: Q, unit: Unit) {
        let id = TypeId::of::<Q>();
        if let Some(v) = self.0.get_mut(&id) {
            v.push(unit);
        } else {
            self.0.insert(id, vec![unit]);
        }
    }

    pub fn extend<Q: PhysQuantity + 'static>(&mut self, _quantity: Q, units: &[Unit]) {
        let id = TypeId::of::<Q>();
        if let Some(v) = self.0.get_mut(&id) {
            v.extend_from_slice(units);
        } else {
            self.0.insert(id, units.to_owned());
        }
    }

    pub fn get<Q: PhysQuantity + 'static>(&self) -> &[Unit] {
        self.0.get(&TypeId::of::<Q>()).expect("No units defined for physical quantity")
    }

    pub fn get_unit<Q: PhysQuantity + 'static>(&self, name: &str) -> Unit {
        *self.get::<Q>().iter()
            .find(|&x| x.name.to_lowercase() == name.to_lowercase())
            .expect("Could not find searched unit in UnitRegistry")
    }
}

pub fn energy_units() -> Vec<Unit> {
    let kelvin = CODATA_2022.boltzmann_constant;
    let hz = TAU * CODATA_2022.reduced_planck;
    let cm_inv = 100.0 * hz * CODATA_2022.speed_of_light;
    let hartree = 4.359_744_722_2060e-18;

    vec![
        Unit::new("au", hartree),
        Unit::new("Hartree", hartree),
        Unit::new("Kelvin", kelvin),
        Unit::new("eV", 1.602_176_634e-19),
        Unit::new("cm_inv", cm_inv),
        Unit::new("Hz", hz),
        Unit::new("kHz", 1e3 * hz),
        Unit::new("MHz", 1e6 * hz),
        Unit::new("GHz", 1e9 * hz),
        Unit::new("THz", 1e12 * hz),
    ]
}

pub fn mass_units() -> Vec<Unit> {
    vec![
        Unit::new("m_e", 9.109_383_7139e-31),
        Unit::new("u", 1.660_539_068_92e-27)
    ]
}

pub fn length_units() -> Vec<Unit> {
    vec![
        Unit::new("bohr", 5.291_772_105_44e-11),
        Unit::new("Angstrom", 1e-10)
    ]
}

pub fn b_field_units() -> Vec<Unit> {
    vec![
        Unit::new("Gauss", 1.),
        Unit::new("Tesla", 1.),
    ]
}

pub fn unit_registry() -> UnitRegistry {
    let mut registry = UnitRegistry::default();
    registry.extend(Energy, &energy_units());
    registry.extend(Mass, &mass_units());
    registry.extend(Length, &length_units());
    registry.extend(MagneticField, &b_field_units());

    registry
}

pub struct InputQuantity(pub f64, pub String);

#[cfg(test)]
mod tests {
    use cc_math_utils::assert_approx_eq;

    use super::*;

    #[test]
    pub fn test_dimensions() {
        let velocity = Dimension::LENGTH / Dimension::TIME;
        assert_eq!(velocity, Dimension::VELOCITY);
        let acceleration = velocity / Dimension::TIME;
        assert_eq!(acceleration, Dimension::ACCELERATION);
        let force = Dimension::MASS * acceleration;
        assert_eq!(force, Dimension::FORCE);
        let energy = force * Dimension::LENGTH;
        assert_eq!(energy, Dimension::ENERGY);
        let surface = Dimension::LENGTH.pow(2);
        assert_eq!(surface, Dimension::SURFACE);
        let volume = Dimension::LENGTH.pow(3);
        assert_eq!(volume, Dimension::VOLUME);
        let density = Dimension::MASS / volume;
        assert_eq!(density, Dimension::DENSITY);

        let current = Dimension::CHARGE / Dimension::TIME;
        assert_eq!(current, Dimension::CURRENT);
        let voltage = Dimension::ELECTRIC_FIELD * Dimension::LENGTH;
        assert_eq!(voltage, Dimension::VOLTAGE);

        let electric_field = Dimension::FORCE / Dimension::CHARGE;
        assert_eq!(electric_field, Dimension::ELECTRIC_FIELD);
        let magnetic_field = Dimension::FORCE / Dimension::CHARGE / Dimension::VELOCITY;
        assert_eq!(magnetic_field, Dimension::MAGNETIC_FIELD);

        let electric_dipole = Dimension::ENERGY / Dimension::ELECTRIC_FIELD;
        assert_eq!(electric_dipole, Dimension::ELECTRIC_DIPOLE);
        let magnetic_dipole = Dimension::ENERGY / Dimension::MAGNETIC_FIELD;
        assert_eq!(magnetic_dipole, Dimension::MAGNETIC_DIPOLE);
    }

    #[test]
    pub fn test_units() {
        let val = 10.0;
        let energy_cm_inv = InputQuantity(val, "cm_inv".into());
        let registry = unit_registry();
        let system = &ATOMIC_UNITS_TABLE;

        let energy_hartree: Quantity<Energy> = Quantity::from_unit(energy_cm_inv, &registry, system);
        assert_approx_eq!(energy_hartree.0, val * 4.5563352529132e-6, 1e-5);
    }
}
