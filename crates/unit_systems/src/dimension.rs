use std::ops::{Div, Mul};
use num_rational::Ratio;
use num_traits::Pow;

pub type Ratio8 = Ratio<i8>;

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
    pub const ONE: Self = Self::new(0, 0, 0, 0, 0, 0, 0);

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
    pub const ACTION: Self = Self::new(1, 2, -1, 0, 0, 0, 0);
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

#[cfg(test)]
mod tests {
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
}