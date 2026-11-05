from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from math import isclose
from typing import Iterable

from dimension import Dimension

def pow_ratio(value: float, exp: Fraction) -> float:
    return value ** (exp.numerator / exp.denominator)

@dataclass(frozen=True, slots=True)
class UnitSystemTable:
    """Unit system definition through conversion table to the SI unit system."""

    mass_kg: float = 1.0  # mass of 1 kilogram
    length_m: float = 1.0  # length of 1 meter
    time_s: float = 1.0  # time of 1 second
    charge_c: float = 1.0  # charge of 1 Coulomb
    temperature_k: float = 1.0  # temperature of 1 Kelvin
    amount_mol: float = 1.0  # amount of 1 mol of matter
    luminous_intensity_cd: float = 1.0  # luminous intensity of 1 Candela

    def as_si(self, dimension: Dimension) -> float:
        """Multiplier from this system's unit of `dimension` to the SI unit."""
        return (
            pow_ratio(self.mass_kg, dimension.mass)
            * pow_ratio(self.length_m, dimension.length)
            * pow_ratio(self.time_s, dimension.time)
            * pow_ratio(self.charge_c, dimension.charge)
            * pow_ratio(self.temperature_k, dimension.temperature)
            * pow_ratio(self.amount_mol, dimension.amount)
            * pow_ratio(self.luminous_intensity_cd, dimension.luminous_intensity)
        )

    def from_si(self, dimension: Dimension) -> float:
        """Multiplier from the SI unit of `dimension` to this system's unit."""
        return 1.0 / self.as_si(dimension)


#: Default conversion: SI to SI (Rust: `SI_TABLE` / `Default`).
SI_TABLE = UnitSystemTable()


@dataclass(frozen=True, slots=True)
class FundamentalConstantsSI:
    speed_of_light: float
    reduced_planck: float
    elementary_charge: float
    electron_mass: float
    electric_vacuum_permittivity: float
    boltzmann_constant: float
    fine_constant: float


CODATA_2022 = FundamentalConstantsSI(
    speed_of_light=299_792_458.0,
    reduced_planck=1.054_571_817e-34,
    elementary_charge=1.602_176_634e-19,
    electron_mass=9.109_383_713_9e-31,
    electric_vacuum_permittivity=8.854_187_818_8e-12,
    boltzmann_constant=1.380_649e-23,
    fine_constant=0.007_297_352_564_3,
)


COMPOUND_ERROR_MSG = (
    'Expected unit of type: "A * B^n / C^(n/m)" in order specified by the quantity'
)


class QuantityMeta(ABCMeta):
    """Metaclass giving quantity *types* the `*` and `/` operators —
    the analogue of Rust's `phys_quantity_ops!` macro."""

    def __mul__(cls, rhs: Quantity) -> Prod:
        return Prod(cls, rhs)  # pyright: ignore[reportArgumentType]

    def __truediv__(cls, rhs: Quantity) -> Frac:
        return Frac(cls, rhs)  # pyright: ignore[reportArgumentType]


class PhysQuantity(metaclass=QuantityMeta):
    """A physical quantity. Leaf quantities subclass this and implement
    `dimension()`; compound quantities are built with `*`, `/` and
    `Power(...)`.

    Example (port of the Rust doc example)::

        class Voltage(PhysQuantity):
            @classmethod
            def dimension(cls) -> Dimension:
                return Dimension.VOLTAGE

        class ElectricField(PhysQuantity):
            @classmethod
            def dimension(cls) -> Dimension:
                return Dimension.ELECTRIC_FIELD

            @classmethod
            def to_unit_system_logic(cls, unit, registry, system) -> float:
                # treat electric field as voltage / length
                return Frac(Voltage, Length).to_unit_system_logic(unit, registry, system)
    """

    @classmethod
    @abstractmethod
    def dimension(cls) -> Dimension:
        ...


def dimension_of(quantity: Quantity) -> Dimension:
    """Dimension of a leaf quantity class or a compound instance."""
    return quantity.dimension()


def quantity_repr(quantity: Quantity) -> str:
    """`Debug`-style name of a quantity specifier (Rust: `{:?}`)."""
    if isinstance(quantity, type):
        return quantity.__name__
    return repr(quantity)


def _parse_exponent(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(COMPOUND_ERROR_MSG) from None


@dataclass(frozen=True, slots=True)
class Prod:
    """Product of two quantities (Rust: `Prod<L, R>`)."""

    left: Quantity
    right: Quantity

    def dimension(self) -> Dimension:
        return dimension_of(self.left) * dimension_of(self.right)

    def __mul__(self, rhs: Quantity) -> Prod:
        return Prod(self, rhs)

    def __truediv__(self, rhs: Quantity) -> Frac:
        return Frac(self, rhs)

    def __repr__(self) -> str:
        return f"{quantity_repr(self.left)}*{quantity_repr(self.right)}"


@dataclass(frozen=True, slots=True)
class Frac:
    """Quotient of two quantities (Rust: `Frac<L, R>`)."""

    left: Quantity
    right: Quantity

    def dimension(self) -> Dimension:
        return dimension_of(self.left) / dimension_of(self.right)

    def __mul__(self, rhs: Quantity) -> Prod:
        return Prod(self, rhs)

    def __truediv__(self, rhs: Quantity) -> Frac:
        return Frac(self, rhs)

    def __repr__(self) -> str:
        return f"{quantity_repr(self.left)}/{quantity_repr(self.right)}"


@dataclass(frozen=True, slots=True)
class Power:
    """Quantity raised to a rational power (Rust: `Power<L, N, M>`).

    `Power(Length, 6)` is `Length^6`; `Power(Mass, 1, 2)` is `Mass^(1/2)`.
    """

    quantity: Quantity
    numerator: int
    denominator: int = 1

    def dimension(self) -> Dimension:
        return dimension_of(self.quantity) ** Fraction(self.numerator, self.denominator)

    def __mul__(self, rhs: Quantity) -> Prod:
        return Prod(self, rhs)

    def __truediv__(self, rhs: Quantity) -> Frac:
        return Frac(self, rhs)

    def __repr__(self) -> str:
        if self.denominator == 1:
            return f"{quantity_repr(self.quantity)}^{self.numerator}"
        return f"({quantity_repr(self.quantity)})^({self.numerator}/{self.denominator})"


def Inv(quantity: Quantity) -> Power:
    """Analogue of the Rust type alias `Inv<L> = Power<L, -1>`."""
    return Power(quantity, -1)


#: Any quantity specifier: a leaf class or a compound instance.
Quantity = type[PhysQuantity] | Prod | Frac | Power


def phys_quantity(name: str, dimension: Dimension) -> type[PhysQuantity]:
    """Create a leaf quantity class — the analogue of `phys_quantity!`."""

    def dimension_(cls: type[PhysQuantity]) -> Dimension:
        return dimension

    return QuantityMeta(name, (PhysQuantity,), {"dimension": classmethod(dimension_)})  # pyright: ignore[reportReturnType]


# --- phys_quantities ------------------------------------------------------

Dimensionless = phys_quantity("Dimensionless", Dimension.DIMENSIONLESS)
Mass = phys_quantity("Mass", Dimension.MASS)
Length = phys_quantity("Length", Dimension.LENGTH)
Time = phys_quantity("Time", Dimension.TIME)
Charge = phys_quantity("Charge", Dimension.CHARGE)
Energy = phys_quantity("Energy", Dimension.ENERGY)
Action = phys_quantity("Action", Dimension.ACTION)
MagneticField = phys_quantity("MagneticField", Dimension.MAGNETIC_FIELD)
ElectricField = phys_quantity("ElectricField", Dimension.ELECTRIC_FIELD)
MagneticDipole = phys_quantity("MagneticDipole", Dimension.MAGNETIC_DIPOLE)
ElectricDipole = phys_quantity("ElectricDipole", Dimension.ELECTRIC_DIPOLE)

def test_quantities() -> None:
    k_vector = Power(Mass * Energy / Power(Action, 2), 1, 2)
    assert k_vector.dimension() == Dimension.ONE / Dimension.LENGTH

if __name__ == "__main__":
    test_quantities()
    print("All unit_systems tests passed.")