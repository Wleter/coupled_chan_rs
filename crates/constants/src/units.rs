pub mod atomic_units;

use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Sub, SubAssign},
};

pub trait Unit: Copy + Default {
    type Base;

    const TO_BASE: f64;
}

#[derive(Clone, Copy)]
pub struct Quantity<U>(pub f64, pub U);

impl<U: Debug> Debug for Quantity<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.0, self.1)
    }
}

impl<U: Debug> Display for Quantity<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {:?}", self.0, self.1)
    }
}

impl< U: Unit> Quantity<U> {
    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }

    pub fn to<V: Unit<Base = U::Base>>(&self, unit: V) -> Quantity<V> {
        Quantity(self.0 * (U::TO_BASE / V::TO_BASE), unit)
    }
}

impl<U: Unit> Add for Quantity<U> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quantity(self.0 + rhs.0, U::default())
    }
}

impl<U: Unit> AddAssign for Quantity<U> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<U: Unit> Sum for Quantity<U> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Quantity(iter.map(|x| x.0).sum(), U::default())
    }
}

impl<U: Unit> Sub for Quantity<U> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Quantity(self.0 - rhs.0, U::default())
    }
}

impl<U: Unit> SubAssign for Quantity<U> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use crate::units::Unit;

    #[test]
    fn test_units() {
        use crate::units::{Quantity, atomic_units::*};

        let one_kelvin = Quantity(1., Kelvin);
        let two_kelvin = Quantity(2., Kelvin);

        let result = one_kelvin + two_kelvin;
        assert_eq!(result.value(), 3.);

        let result = two_kelvin - one_kelvin;
        assert_eq!(result.value(), 1.);

        assert!(&format!("{one_kelvin}") == "1 Kelvin", "Wrong format");

        let result = one_kelvin.to(GHz);
        assert_eq!(result.value(), Kelvin::TO_BASE / GHz::TO_BASE)
    }
}
