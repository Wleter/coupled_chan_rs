use std::{
    any::TypeId,
    collections::HashMap,
};

use num_traits::Pow;

use crate::{
    UnitSystemTable,
    dimension::{
        Dimension,
        Ratio8,
    },
};

pub trait PhysQuantity: std::fmt::Debug + Default + 'static {
    fn dimension() -> Dimension;

    fn dimension_self(self) -> Dimension {
        Self::dimension()
    }

    /// Logic for converting units to wanted unit system,
    /// Used for physical quantities, which do not have their own units.
    /// Consider using [`Prod`], [`Frac`], [`Power`] for such cases:
    /// `let c6_quantity: Prod<Energy, Power<Length, 6>> = Energy * Power::<_, 6>(Length)`
    ///
    /// # Examples
    ///
    /// ```
    /// use unit_systems::{
    ///     UnitSystemTable,
    ///     dimension::Dimension,
    ///     phys_quantity,
    ///     phys_quantity_ops,
    ///     quantities::{
    ///         Frac,
    ///         PhysQuantity,
    ///         UnitRegistry,
    ///         phys_quantities,
    ///     },
    /// };
    ///
    /// phys_quantity!(Voltage, Dimension::VOLTAGE);
    ///
    /// #[derive(Clone, Copy, Debug, Default)]
    /// pub struct ElectricField;
    ///
    /// impl PhysQuantity for ElectricField {
    ///     fn dimension() -> Dimension {
    ///         Dimension::ELECTRIC_FIELD
    ///     }
    ///
    ///     fn to_unit_system_logic(
    ///         unit: impl AsRef<str>,
    ///         registry: &UnitRegistry,
    ///         system: &UnitSystemTable,
    ///     ) -> f64 {
    ///         // treat electric field as voltage / length, when converting to unit_system
    ///         Frac::<Voltage, phys_quantities::Length>::to_unit_system_logic(unit, registry, system)
    ///     }
    /// }
    ///
    /// phys_quantity_ops!(ElectricField);
    /// ```
    fn to_unit_system_logic(unit: impl AsRef<str>, registry: &UnitRegistry, system: &UnitSystemTable) -> f64 {
        let dim = Self::dimension();
        let from_si = system.from_si(dim);

        let to_si = registry.get_unit::<Self>(unit.as_ref()).to_si;

        to_si * from_si
    }
}

#[macro_export]
macro_rules! phys_quantity_ops {
    ($name:ident) => {
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

#[macro_export]
macro_rules! phys_quantity {
    ($name:ident, $dimension:expr) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name;

        impl $crate::quantities::PhysQuantity for $name {
            fn dimension() -> Dimension {
                $dimension
            }
        }

        $crate::phys_quantity_ops!($name);
    };
}

pub mod phys_quantities {
    use super::*;

    phys_quantity!(Dimensionless, Dimension::DIMENSIONLESS);
    phys_quantity!(Mass, Dimension::MASS);
    phys_quantity!(Length, Dimension::LENGTH);
    phys_quantity!(Time, Dimension::TIME);
    phys_quantity!(Charge, Dimension::CHARGE);
    phys_quantity!(Energy, Dimension::ENERGY);
    phys_quantity!(Action, Dimension::ACTION);
    phys_quantity!(MagneticField, Dimension::MAGNETIC_FIELD);
    phys_quantity!(ElectricField, Dimension::ELECTRIC_FIELD);
    phys_quantity!(MagneticDipole, Dimension::MAGNETIC_DIPOLE);
    phys_quantity!(ElectricDipole, Dimension::ELECTRIC_DIPOLE);
}

const COMPOUND_ERROR_MSG: &str = "Expected unit of type: \"A * B^n / C^(n/m)\" in order specified by the quantity";

#[derive(Clone, Copy, Default, PartialEq)]
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

    // Relies on associativity rule a * b / c == (a * b) / c
    fn to_unit_system_logic(unit: impl AsRef<str>, registry: &UnitRegistry, system: &UnitSystemTable) -> f64 {
        let unit = unit.as_ref().trim_matches(['(', ')', ' ']);

        if let Some((a, b)) = unit.rsplit_once("*") {
            L::to_unit_system_logic(a, registry, system) * R::to_unit_system_logic(b, registry, system)
        } else {
            panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default())
        }
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

#[derive(Clone, Copy, Default, PartialEq)]
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

    // Relies on associativity rule a * b / c == (a * b) / c
    fn to_unit_system_logic(unit: impl AsRef<str>, registry: &UnitRegistry, system: &UnitSystemTable) -> f64 {
        let unit = unit.as_ref().trim_matches(['(', ')', ' ']);

        if let Some((a, b)) = unit.rsplit_once("/") {
            L::to_unit_system_logic(a, registry, system) / R::to_unit_system_logic(b, registry, system)
        } else {
            panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default())
        }
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

pub type Inv<L> = Power<L, -1>;

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Power<L: PhysQuantity, const N: i8, const M: i8 = 1>(pub L);

impl<V: PhysQuantity, const N: i8, const M: i8> std::fmt::Debug for Power<V, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if M == 1 {
            write!(f, "{:?}^{:?}", self.0, N)
        } else {
            write!(f, "({:?})^({:?}/{:?})", self.0, N, M)
        }
    }
}

impl<L: PhysQuantity, const N: i8, const M: i8> PhysQuantity for Power<L, N, M> {
    fn dimension() -> Dimension {
        L::dimension().pow(Ratio8::new(N, M))
    }

    /// Relies on associativity rule a * b ^ c == (a * b)^c
    fn to_unit_system_logic(unit: impl AsRef<str>, registry: &UnitRegistry, system: &UnitSystemTable) -> f64 {
        let unit = unit.as_ref().trim_matches(['(', ')', ' ']);

        if let Some((a, n)) = unit.rsplit_once("^") {
            if let Some(i) = n.find("/") {
                let (n, m) = n.split_at(i);
                let n = n
                    .parse::<i8>()
                    .unwrap_or_else(|_| panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default()));
                let m = m
                    .parse::<i8>()
                    .unwrap_or_else(|_| panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default()));
                assert_eq!(n, N, "{COMPOUND_ERROR_MSG} {:?}", Self::default());
                assert_eq!(m, M, "{COMPOUND_ERROR_MSG} {:?}", Self::default());
            } else {
                let n = n
                    .parse::<i8>()
                    .unwrap_or_else(|_| panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default()));
                assert_eq!(n, N, "{COMPOUND_ERROR_MSG} {:?}", Self::default());
            }

            L::to_unit_system_logic(a, registry, system).powf(N as f64 / M as f64)
        } else {
            panic!("{COMPOUND_ERROR_MSG} {:?}", Self::default())
        }
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

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scalar<Q: PhysQuantity>(f64, Box<str>, #[cfg_attr(feature = "serde", serde(skip))] Q);

impl<Q: PhysQuantity> Scalar<Q> {
    pub fn new(value: f64, quantity: Q, unit: impl AsRef<str>) -> Self {
        Self(value, unit.as_ref().into(), quantity)
    }

    pub fn in_unit_system(&self, registry: &UnitRegistry, system: &UnitSystemTable) -> f64 {
        if self.0 == 0.0 {
            return 0.0;
        }

        self.0 * Q::to_unit_system_logic(&self.1, registry, system)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Unit {
    pub name: &'static str,
    pub to_si: f64,
}

impl Unit {
    pub const fn new(name: &'static str, to_si: f64) -> Self {
        Self { name, to_si }
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
        self.0
            .get(&TypeId::of::<Q>())
            .expect("No units defined for physical quantity")
    }

    pub fn get_unit<Q: PhysQuantity + 'static>(&self, name: &str) -> Unit {
        *self
            .get::<Q>()
            .iter()
            .find(|&x| x.name.trim().to_lowercase() == name.trim().to_lowercase())
            .unwrap_or_else(|| panic!("Could not find unit {name} in UnitRegistry"))
    }
}

pub struct UnitsConverter {
    pub registry: UnitRegistry,
    pub target_unit_system: UnitSystemTable,
}

impl UnitsConverter {
    pub fn scalar_value<Q: PhysQuantity>(&self, scalar: &Scalar<Q>) -> f64 {
        scalar.in_unit_system(&self.registry, &self.target_unit_system)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        phys_quantities::*,
        *,
    };

    #[test]
    pub fn test_quantities() {
        let k_vector = Power::<_, 1, 2>(Mass * Energy / Power::<_, 2>(Action));
        assert_eq!(k_vector.dimension_self(), Dimension::ONE / Dimension::LENGTH);
    }
}
