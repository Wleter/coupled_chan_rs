from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

@dataclass(frozen=True, slots=True)
class Dimension:
    """Dimensional exponents of a physical quantity, expressed over the
    seven SI base dimensions (M, L, t, e, T, N, J)."""

    mass: Fraction
    length: Fraction
    time: Fraction
    charge: Fraction
    temperature: Fraction
    amount: Fraction
    luminous_intensity: Fraction

    # --- constructors -----------------------------------------------------

    @classmethod
    def new_fractional(
        cls,
        mass: Fraction,
        length: Fraction,
        time: Fraction,
        charge: Fraction,
        temperature: Fraction,
        amount: Fraction,
        luminous_intensity: Fraction,
    ) -> Dimension:
        return cls(
            mass=mass,
            length=length,
            time=time,
            charge=charge,
            temperature=temperature,
            amount=amount,
            luminous_intensity=luminous_intensity,
        )

    @classmethod
    def new(
        cls,
        mass: int,
        length: int,
        time: int,
        charge: int,
        temperature: int,
        amount: int,
        luminous_intensity: int,
    ) -> Dimension:
        return cls.new_fractional(
            Fraction(mass),
            Fraction(length),
            Fraction(time),
            Fraction(charge),
            Fraction(temperature),
            Fraction(amount),
            Fraction(luminous_intensity),
        )

    # --- operators -----------------------------------------------------------

    def __mul__(self, rhs: Dimension) -> Dimension:
        return Dimension.new_fractional(
            self.mass + rhs.mass,
            self.length + rhs.length,
            self.time + rhs.time,
            self.charge + rhs.charge,
            self.temperature + rhs.temperature,
            self.amount + rhs.amount,
            self.luminous_intensity + rhs.luminous_intensity,
        )

    def __truediv__(self, rhs: Dimension) -> Dimension:
        return Dimension.new_fractional(
            self.mass - rhs.mass,
            self.length - rhs.length,
            self.time - rhs.time,
            self.charge - rhs.charge,
            self.temperature - rhs.temperature,
            self.amount - rhs.amount,
            self.luminous_intensity - rhs.luminous_intensity,
        )

    def __pow__(self, rhs: Fraction | int) -> Dimension:
        rhs = Fraction(rhs)
        return Dimension.new_fractional(
            self.mass * rhs,
            self.length * rhs,
            self.time * rhs,
            self.charge * rhs,
            self.temperature * rhs,
            self.amount * rhs,
            self.luminous_intensity * rhs,
        )

    # --- dimension constants -------------------------------------------------
    # Declared as ClassVar annotations only (no values): the dataclass never
    # treats ClassVar members as instance fields, so they create no slots and
    # don't collide with `slots=True`. The actual values are assigned after
    # the class body, at the bottom of the file.

    DIMENSIONLESS: ClassVar[Dimension]
    ONE: ClassVar[Dimension]

    MASS: ClassVar[Dimension]
    LENGTH: ClassVar[Dimension]
    TIME: ClassVar[Dimension]
    CHARGE: ClassVar[Dimension]
    TEMPERATURE: ClassVar[Dimension]
    AMOUNT: ClassVar[Dimension]
    LUMINOUS_INTENSITY: ClassVar[Dimension]

    VELOCITY: ClassVar[Dimension]
    ACCELERATION: ClassVar[Dimension]
    FORCE: ClassVar[Dimension]
    ENERGY: ClassVar[Dimension]
    ACTION: ClassVar[Dimension]
    SURFACE: ClassVar[Dimension]
    VOLUME: ClassVar[Dimension]
    DENSITY: ClassVar[Dimension]

    CURRENT: ClassVar[Dimension]
    VOLTAGE: ClassVar[Dimension]
    MAGNETIC_FIELD: ClassVar[Dimension]
    ELECTRIC_FIELD: ClassVar[Dimension]

    MAGNETIC_DIPOLE: ClassVar[Dimension]
    ELECTRIC_DIPOLE: ClassVar[Dimension]

    # --- display ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Dimension {{[M^{self.mass} L^{self.length} t^{self.time} "
            f"e^{self.charge} T^{self.temperature} N^{self.amount} "
            f"J^{self.luminous_intensity}]}}"
        )

    def __str__(self) -> str:
        return (
            f"{{[M^{self.mass} L^{self.length} t^{self.time} "
            f"e^{self.charge} T^{self.temperature} N^{self.amount} "
            f"J^{self.luminous_intensity}]}}"
        )


# --- constant definitions -------------------------------------------------

Dimension.DIMENSIONLESS = Dimension.new(0, 0, 0, 0, 0, 0, 0)
Dimension.ONE = Dimension.new(0, 0, 0, 0, 0, 0, 0)

Dimension.MASS = Dimension.new(1, 0, 0, 0, 0, 0, 0)
Dimension.LENGTH = Dimension.new(0, 1, 0, 0, 0, 0, 0)
Dimension.TIME = Dimension.new(0, 0, 1, 0, 0, 0, 0)
Dimension.CHARGE = Dimension.new(0, 0, 0, 1, 0, 0, 0)
Dimension.TEMPERATURE = Dimension.new(0, 0, 0, 0, 1, 0, 0)
Dimension.AMOUNT = Dimension.new(0, 0, 0, 0, 0, 1, 0)
Dimension.LUMINOUS_INTENSITY = Dimension.new(0, 0, 0, 0, 0, 0, 1)

Dimension.VELOCITY = Dimension.new(0, 1, -1, 0, 0, 0, 0)
Dimension.ACCELERATION = Dimension.new(0, 1, -2, 0, 0, 0, 0)
Dimension.FORCE = Dimension.new(1, 1, -2, 0, 0, 0, 0)
Dimension.ENERGY = Dimension.new(1, 2, -2, 0, 0, 0, 0)
Dimension.ACTION = Dimension.new(1, 2, -1, 0, 0, 0, 0)
Dimension.SURFACE = Dimension.new(0, 2, 0, 0, 0, 0, 0)
Dimension.VOLUME = Dimension.new(0, 3, 0, 0, 0, 0, 0)
Dimension.DENSITY = Dimension.new(1, -3, 0, 0, 0, 0, 0)

Dimension.CURRENT = Dimension.new(0, 0, -1, 1, 0, 0, 0)
Dimension.VOLTAGE = Dimension.new(1, 2, -2, -1, 0, 0, 0)
Dimension.MAGNETIC_FIELD = Dimension.new(1, 0, -1, -1, 0, 0, 0)
Dimension.ELECTRIC_FIELD = Dimension.new(1, 1, -2, -1, 0, 0, 0)

Dimension.MAGNETIC_DIPOLE = Dimension.new(0, 2, -1, 1, 0, 0, 0)
Dimension.ELECTRIC_DIPOLE = Dimension.new(0, 1, 0, 1, 0, 0, 0)

# --- tests ------------------------------------------------------------------


def test_dimensions() -> None:
    velocity = Dimension.LENGTH / Dimension.TIME
    assert velocity == Dimension.VELOCITY
    acceleration = velocity / Dimension.TIME
    assert acceleration == Dimension.ACCELERATION
    force = Dimension.MASS * acceleration
    assert force == Dimension.FORCE
    energy = force * Dimension.LENGTH
    assert energy == Dimension.ENERGY
    surface = Dimension.LENGTH ** 2
    assert surface == Dimension.SURFACE
    volume = Dimension.LENGTH ** 3
    assert volume == Dimension.VOLUME
    density = Dimension.MASS / volume
    assert density == Dimension.DENSITY

    current = Dimension.CHARGE / Dimension.TIME
    assert current == Dimension.CURRENT
    voltage = Dimension.ELECTRIC_FIELD * Dimension.LENGTH
    assert voltage == Dimension.VOLTAGE

    electric_field = Dimension.FORCE / Dimension.CHARGE
    assert electric_field == Dimension.ELECTRIC_FIELD
    magnetic_field = Dimension.FORCE / Dimension.CHARGE / Dimension.VELOCITY
    assert magnetic_field == Dimension.MAGNETIC_FIELD

    electric_dipole = Dimension.ENERGY / Dimension.ELECTRIC_FIELD
    assert electric_dipole == Dimension.ELECTRIC_DIPOLE
    magnetic_dipole = Dimension.ENERGY / Dimension.MAGNETIC_FIELD
    assert magnetic_dipole == Dimension.MAGNETIC_DIPOLE

    # Fractional exponents (not in the Rust test, but supported by Pow<Ratio8>)
    half_sqrt_energy = Dimension.ENERGY ** Fraction(1, 2)
    assert half_sqrt_energy.mass == Fraction(1, 2)
    assert half_sqrt_energy.time == Fraction(-1)


if __name__ == "__main__":
    test_dimensions()
    print("All dimension tests passed.")