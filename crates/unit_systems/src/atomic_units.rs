use std::f64::consts::{
    PI,
    TAU,
};

use crate::{
    CODATA_2022,
    FundamentalConstantsSI,
    UnitSystemTable,
    quantities::{
        Unit,
        UnitRegistry,
        phys_quantities::*,
    },
};

pub const fn atomic_units_table(constants: &FundamentalConstantsSI) -> UnitSystemTable {
    // defining constants
    let hbar = constants.reduced_planck;
    let e = constants.elementary_charge;
    let m_e = constants.electron_mass;
    let permittivity = 4.0 * PI * constants.electric_vacuum_permittivity;

    let bohr = permittivity * hbar * hbar / (m_e * e * e);
    let hartree_energy = hbar * hbar / (m_e * bohr * bohr);
    let atomic_time = hbar / hartree_energy;

    UnitSystemTable {
        mass_kg: m_e,
        length_m: bohr,
        time_s: atomic_time,
        charge_c: e,
        temperature_k: 1.0,
        amount_mol: 1.0,
        luminous_intensity_cd: 1.0,
    }
}

pub const ATOMIC_UNITS_TABLE: UnitSystemTable = atomic_units_table(&CODATA_2022);

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
    vec![Unit::new("m_e", 9.109_383_7139e-31), Unit::new("u", 1.660_539_068_92e-27)]
}

pub fn length_units() -> Vec<Unit> {
    vec![Unit::new("bohr", 5.291_772_105_44e-11), Unit::new("Angstrom", 1e-10)]
}

pub fn b_field_units() -> Vec<Unit> {
    vec![Unit::new("Gauss", 1.), Unit::new("Tesla", 1.)]
}

pub fn unit_registry() -> UnitRegistry {
    let mut registry = UnitRegistry::default();
    registry.extend(Energy, &energy_units());
    registry.extend(Mass, &mass_units());
    registry.extend(Length, &length_units());
    registry.extend(MagneticField, &b_field_units());

    registry
}

#[cfg(test)]
mod tests {
    use cc_math_utils::assert_approx_eq;

    use super::*;
    use crate::quantities::{
        InputScalar,
        Scalar,
    };

    #[test]
    pub fn test_units() {
        let val = 10.0;
        let energy_cm_inv = InputScalar(val, "cm_inv".into());
        let registry = unit_registry();
        let system = &ATOMIC_UNITS_TABLE;

        let energy_hartree: Scalar<Energy> = Scalar::from_unit(energy_cm_inv, &registry, system);
        assert_approx_eq!(energy_hartree.0, val * 4.5563352529132e-6, 1e-5);
    }
}
