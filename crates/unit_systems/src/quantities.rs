use std::{any::TypeId, collections::HashMap};

use num_traits::Pow;

use crate::{UnitSystemTable, dimension::{Dimension, Ratio8}};

pub trait PhysQuantity: std::fmt::Debug + Default {
    fn dimension() -> Dimension;

    fn dimension_self(self) -> Dimension {
        Self::dimension()
    }
}

#[macro_export]
macro_rules! phys_quantity {
    ($name:ident, $dimension:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $crate::quantities::PhysQuantity for $name {
            fn dimension() -> Dimension {
                $dimension
            }
        }

        impl<V: $crate::quantities::PhysQuantity> std::ops::Mul<V> for $name {
            type Output = $crate::quantities::Prod<$name, V>;

            fn mul(self, rhs: V) -> Self::Output {
                $crate::quantities::Prod(self, rhs)
            }
        }

        impl<V: $crate::quantities::PhysQuantity> std::ops::Div<V> for $name {
            type Output = $crate::quantities::Frac<$name, V>;

            fn div(self, rhs: V) -> Self::Output {
                $crate::quantities::Frac(self, rhs)
            }
        }
    };
}

pub mod phys_quantities {
    use super::*;

    phys_quantity!(Mass, Dimension::MASS);
    phys_quantity!(Length, Dimension::LENGTH);
    phys_quantity!(Time, Dimension::TIME);
    phys_quantity!(Charge, Dimension::CHARGE);
    phys_quantity!(Energy, Dimension::ENERGY);
    phys_quantity!(Action, Dimension::ACTION);
    phys_quantity!(MagneticField, Dimension::MAGNETIC_FIELD);
    phys_quantity!(ElectricField, Dimension::ELECTRIC_FIELD);
}

#[derive(Clone, Copy, Default)]
pub struct Prod<L: PhysQuantity, R: PhysQuantity>(pub L, pub R);

impl<U: PhysQuantity, V: PhysQuantity> std::fmt::Debug for Prod<U, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}*{:?}", self.0, self.1)
    }
}

impl<L: PhysQuantity, R: PhysQuantity> PhysQuantity for Prod<L, R> {
    fn dimension() -> Dimension {
        L::dimension() * R::dimension()
    }
}

impl<L: PhysQuantity, R: PhysQuantity, V: PhysQuantity> std::ops::Mul<V> for Prod<L, R> {
    type Output = Prod<Self, V>;

    fn mul(self, rhs: V) -> Self::Output {
        Prod(self, rhs)
    }
}

impl<L: PhysQuantity, R: PhysQuantity, V: PhysQuantity> std::ops::Div<V> for Prod<L, R> {
    type Output = Frac<Self, V>;

    fn div(self, rhs: V) -> Self::Output {
        Frac(self, rhs)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Frac<L: PhysQuantity, R: PhysQuantity>(pub L, pub R);

impl<U: PhysQuantity, V: PhysQuantity> std::fmt::Debug for Frac<U, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}/{:?}", self.0, self.1)
    }
}

impl<L: PhysQuantity, R: PhysQuantity> PhysQuantity for Frac<L, R> {
    fn dimension() -> Dimension {
        L::dimension() / R::dimension()
    }
}

impl<L: PhysQuantity, R: PhysQuantity, V: PhysQuantity> std::ops::Mul<V> for Frac<L, R> {
    type Output = Prod<Self, V>;

    fn mul(self, rhs: V) -> Self::Output {
        Prod(self, rhs)
    }
}

impl<L: PhysQuantity, R: PhysQuantity, V: PhysQuantity> std::ops::Div<V> for Frac<L, R> {
    type Output = Frac<Self, V>;

    fn div(self, rhs: V) -> Self::Output {
        Frac(self, rhs)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Power<L: PhysQuantity, const N: i8, const M: i8 = 1>(pub L);

impl<V: PhysQuantity, const N: i8, const M: i8> std::fmt::Debug for Power<V, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if N == 1 {
            write!(f, "{:?}**{:?}", self.0, N)
        } else {
            write!(f, "{:?}**({:?}/{:?})", self.0, N, M)
        }
    }
}

impl<L: PhysQuantity, const N: i8, const M: i8> PhysQuantity for Power<L, N, M> {
    fn dimension() -> Dimension {
    L::dimension().pow(Ratio8::new(N, M))
    }
}

impl<L: PhysQuantity, const N: i8, const M: i8, V: PhysQuantity> std::ops::Mul<V> for Power<L, N, M> {
    type Output = Prod<Self, V>;

    fn mul(self, rhs: V) -> Self::Output {
        Prod(self, rhs)
    }
}

impl<L: PhysQuantity, const N: i8, const M: i8, V: PhysQuantity> std::ops::Div<V> for Power<L, N, M> {
    type Output = Frac<Self, V>;

    fn div(self, rhs: V) -> Self::Output {
        Frac(self, rhs)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Scalar<Q: PhysQuantity>(pub f64, pub Q);

#[derive(Clone, Debug)]
pub struct InputScalar(pub f64, pub String);

impl<Q: PhysQuantity + 'static> Scalar<Q> {
    pub fn from_unit(
        value: InputScalar, 
        registry: &UnitRegistry,
        system: &UnitSystemTable,
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::phys_quantities::*;

    #[test]
    pub fn test_quantities() {
        let k_vector = Power::<_, 1, 2>(Mass * Energy / Power::<_, 2>(Action));
        assert_eq!(k_vector.dimension_self(), Dimension::ONE / Dimension::LENGTH);
    }
}