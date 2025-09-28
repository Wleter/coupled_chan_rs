pub mod atomic_units;
pub use atomic_units::*;

use std::{
    fmt::{Debug, Display, LowerExp, UpperExp},
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
};

pub trait Unit: Copy + Default {
    type Base: Unit;

    const TO_BASE: f64;
}

/// Implements unit group 
/// # Syntax
/// - `impl_unit_group!($unit: ty, $base_unit: ty, $to_base: f64)`
#[macro_export]
macro_rules! impl_unit_group {
    ($unit: ty, $base: ty, $to_base:expr) => {
        impl $crate::units::Unit for $unit {
            type Base = $base;

            const TO_BASE: f64 = $to_base;
        }

        impl<V: $crate::units::Unit> std::ops::Mul<V> for $unit {
            type Output = $crate::units::Prod<$unit, V>;

            fn mul(self, rhs: V) -> Self::Output {
                $crate::units::Prod(self, rhs)
            }
        }

        impl<V: $crate::units::Unit> std::ops::Div<V> for $unit {
            type Output = $crate::units::Frac<$unit, V>;

            fn div(self, rhs: V) -> Self::Output {
                $crate::units::Frac(self, rhs)
            }
        }
    };
}

#[derive(Clone, Copy, Default)]
pub struct Prod<U: Unit, V: Unit>(pub U, pub V);

impl<U: Unit, V: Unit> Unit for Prod<U, V> {
    type Base = Prod<U::Base, V::Base>;

    const TO_BASE: f64 = U::TO_BASE * V::TO_BASE;
}

impl<U: Unit + Debug, V: Unit + Debug> Debug for Prod<U, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} * {:?}", self.0, self.1)
    }
}

impl<U: Unit, V: Unit, K: Unit> std::ops::Mul<K> for Prod<U, V> {
    type Output = Prod<Prod<U, V>, K>;

    fn mul(self, rhs: K) -> Self::Output {
        Prod(self, rhs)
    }
}

impl<U: Unit, V: Unit, K: Unit> std::ops::Div<K> for Prod<U, V> {
    type Output = Frac<Prod<U, V>, K>;

    fn div(self, rhs: K) -> Self::Output {
        Frac(self, rhs)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Frac<U: Unit, V: Unit>(pub U, pub V);

impl<U: Unit, V: Unit> Unit for Frac<U, V> {
    type Base = Frac<U::Base, V::Base>;

    const TO_BASE: f64 = U::TO_BASE / V::TO_BASE;
}

impl<U: Unit + Debug, V: Unit + Debug> Debug for Frac<U, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} / {:?}", self.0, self.1)
    }
}

impl<U: Unit, V: Unit, K: Unit> std::ops::Mul<K> for Frac<U, V> {
    type Output = Prod<Frac<U, V>, K>;

    fn mul(self, rhs: K) -> Self::Output {
        Prod(self, rhs)
    }
}

impl<U: Unit, V: Unit, K: Unit> std::ops::Div<K> for Frac<U, V> {
    type Output = Frac<Frac<U, V>, K>;

    fn div(self, rhs: K) -> Self::Output {
        Frac(self, rhs)
    }
}

#[derive(Clone, Copy, Default)]
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

impl<U: Debug> LowerExp for Quantity<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerExp::fmt(&self.0, f)?;
        write!(f, " {:?}", self.1)
    }
}

impl<U: Debug> UpperExp for Quantity<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperExp::fmt(&self.0, f)?;
        write!(f, " {:?}", self.1)
    }
}

impl<U: Unit> Quantity<U> {
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

impl<U: Unit> Mul<Quantity<U>> for f64 {
    type Output = Quantity<U>;

    fn mul(self, rhs: Quantity<U>) -> Self::Output {
        Quantity(self * rhs.0, rhs.1)
    }
}

impl<U: Unit, V: Unit> Mul<Quantity<V>> for Quantity<U> {
    type Output = Quantity<Prod<U, V>>;

    fn mul(self, rhs: Quantity<V>) -> Self::Output {
        Quantity(self.0 * rhs.0, Prod(self.1, rhs.1))
    }
}

impl<U: Unit, V: Unit> Div<Quantity<V>> for Quantity<U> {
    type Output = Quantity<Frac<U, V>>;

    fn div(self, rhs: Quantity<V>) -> Self::Output {
        Quantity(self.0 / rhs.0, Frac(self.1, rhs.1))
    }
}

#[cfg(test)]
mod tests {
    use crate::units::{Unit, Quantity, atomic_units::*};

    #[test]
    fn test_units() {

        let one_kelvin = Quantity(1., Kelvin);
        let two_kelvin = Quantity(2., Kelvin);

        let result = one_kelvin + two_kelvin;
        assert_eq!(result.value(), 3.);

        let result = two_kelvin - one_kelvin;
        assert_eq!(result.value(), 1.);

        assert_eq!(&format!("{one_kelvin}"), "1 Kelvin", "Wrong format");

        let result = one_kelvin.to(GHz);
        assert_eq!(result.value(), one_kelvin.value() * Kelvin::TO_BASE / GHz::TO_BASE)
    }

    #[test]
    fn test_complex_units() {
        use crate::units::{Quantity, atomic_units::*};

        let quantity = Quantity(1., GHz * Gauss / Bohr);
        assert_eq!(&format!("{quantity}"), "1 GHz * Gauss / Bohr", "Wrong format");

        let quantity_angstrom = quantity.to(GHz * Gauss / Angstrom);
        assert_eq!(quantity_angstrom.value(), quantity.value() * Angstrom::TO_BASE);
        
        let quantity = quantity_angstrom.to(Kelvin * Gauss / Angstrom);
        assert_eq!(quantity_angstrom.value(), quantity.value() * Kelvin::TO_BASE / GHz::TO_BASE);
    }
}
