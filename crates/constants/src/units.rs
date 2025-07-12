pub mod atomic_units;

use std::{
    fmt::{Debug, Display},
    iter::Sum,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use num_traits::real::Real;

pub trait Unit: Copy + Default {
    type Base;

    const TO_BASE: f64;
}

#[derive(Clone, Copy)]
pub struct Quantity<T, U>(pub T, pub U);
pub type Quantity64<U> = Quantity<f64, U>;
pub type Quantity32<U> = Quantity<f32, U>;

impl<T: Debug, U: Debug> Debug for Quantity<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.0, self.1)
    }
}

impl<T: Display, U: Debug> Display for Quantity<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {:?}", self.0, self.1)
    }
}

impl<T: Real + From<f64>, U: Unit> Quantity<T, U> {
    pub fn value(&self) -> T {
        self.0
    }

    pub fn unit(&self) -> U {
        self.1
    }

    pub fn to<V: Unit<Base = U::Base>>(&self, unit: V) -> Quantity<T, V> {
        Quantity(self.0 * (U::TO_BASE / V::TO_BASE).into(), unit)
    }
}

impl<T: Real + From<f64>, U: Unit> Add for Quantity<T, U> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quantity(self.0 + rhs.0, U::default())
    }
}

impl<T: Real + From<f64> + AddAssign, U: Unit> AddAssign for Quantity<T, U> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<T: Real + From<f64> + Sum, U: Unit> Sum for Quantity<T, U> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Quantity(iter.map(|x| x.0).sum(), U::default())
    }
}

impl<T: Real + From<f64>, U: Unit> Sub for Quantity<T, U> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Quantity(self.0 - rhs.0, U::default())
    }
}

impl<T: Real + From<f64> + SubAssign, U: Unit> SubAssign for Quantity<T, U> {
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
