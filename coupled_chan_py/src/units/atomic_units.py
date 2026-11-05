from __future__ import annotations

from math import pi, tau
from .quantity import *

def atomic_units_table(constants: FundamentalConstantsSI) -> UnitSystemTable:
    # defining constants
    hbar = constants.reduced_planck
    e = constants.elementary_charge
    m_e = constants.electron_mass
    permittivity = 4.0 * pi * constants.electric_vacuum_permittivity

    bohr = permittivity * hbar * hbar / (m_e * e * e)
    hartree_energy = hbar * hbar / (m_e * bohr * bohr)
    atomic_time = hbar / hartree_energy

    return UnitSystemTable(
        mass_kg=m_e,
        length_m=bohr,
        time_s=atomic_time,
        charge_c=e,
        temperature_k=1.0,
        amount_mol=1.0,
        luminous_intensity_cd=1.0,
    )


ATOMIC_UNITS_TABLE = atomic_units_table(CODATA_2022)

@dataclass(frozen=True)
class Unit:
    quantity: Quantity
    to_si: float

    def __rmul__(self, other):
        return other * ATOMIC_UNITS_TABLE.as_si(self.quantity.dimension()) / self.to_si

    def __rtruediv__(self, other):
        return other * self.to_si / ATOMIC_UNITS_TABLE.as_si(self.quantity.dimension())



Hartree = Unit(Energy, 4.359_744_722_206_0e-18)
Kelvin = Unit(Energy, CODATA_2022.boltzmann_constant)
cm_inv = Unit(Energy, 100.0 * tau * CODATA_2022.reduced_planck * CODATA_2022.speed_of_light)
eV = Unit(Energy, 1.602_176_634e-19)
Hz = Unit(Energy, tau * CODATA_2022.reduced_planck)
kHz = Unit(Energy, 1e3 * tau * CODATA_2022.reduced_planck)
MHz = Unit(Energy, 1e6 * tau * CODATA_2022.reduced_planck)
GHz = Unit(Energy, 1e9 * tau * CODATA_2022.reduced_planck)
THz = Unit(Energy, 1e12 * tau * CODATA_2022.reduced_planck)

m_e = Unit(Mass, 9.109_383_713_9e-31)
u = Unit(Mass, 1.660_539_068_92e-27)

nm = Unit(Length, 1e-9)
pm = Unit(Length, 1e-12)
bohr = 5.291_772_105_44e-11
Angstrom = 1e-10

Gauss = Unit(MagneticField, 1e-4)
Tesla = Unit(MagneticField, 1.0)

mu_bohr = Unit(MagneticDipole, 9.274_010_065_7e-24)
mu_nuclear = Unit(MagneticDipole, 5.050_783_739_3e-27)

V_per_m = Unit(ElectricField, 1.)
V_per_cm = Unit(ElectricField, 1e2)
kV_per_cm = Unit(ElectricField, 1e5)

Debye = Unit(ElectricDipole, 1e-21 / 299_792_458.0)
