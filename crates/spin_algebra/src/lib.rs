pub use clebsch_gordan::*;

use clebsch_gordan::half_integer::{HalfI32, HalfU32};
use hilbert_space::operator::Braket;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinType {
    Fermionic,
    Bosonic,
}

#[derive(Clone, Copy, PartialEq, Default)]
pub struct Spin {
    pub s: HalfU32,
    pub m: HalfI32,
}

impl std::fmt::Debug for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) {}", self.s, self.m)
    }
}

impl Spin {
    pub fn new(s: HalfU32, m: HalfI32) -> Self {
        Self { s, m }
    }

    pub fn zero() -> Self {
        Self {
            s: 0.into(),
            m: 0.into(),
        }
    }

    pub fn spin_type(&self) -> SpinType {
        if self.s.double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }
}

pub fn get_spin_basis(s: HalfU32) -> Vec<Spin> {
    let ds = s.double_value() as i32;

    (-ds..=ds)
        .step_by(2)
        .map(|dms| Spin::new(s, HalfI32::from_doubled(dms)))
        .collect()
}

pub fn get_summed_spin_basis(s1: HalfU32, s2: HalfU32) -> Vec<Spin> {
    let dspin_max = (s1 + s2).double_value();
    let dspin_min = (s1.double_value() as i32 - s2.double_value() as i32).unsigned_abs();

    (dspin_min..=dspin_max)
        .step_by(2)
        .flat_map(|s| {
            let s = HalfU32::from_doubled(s);
            get_spin_basis(s)
        })
        .collect()
}

pub struct SpinOps;

impl SpinOps {
    #[inline]
    pub fn proj_z(spin: Braket<&Spin>) -> f64 {
        if spin.bra == spin.ket { spin.bra.m.value() } else { 0.0 }
    }

    #[inline]
    pub fn ladder_plus(spin: Braket<&Spin>) -> f64 {
        if spin.bra.s == spin.ket.s && spin.bra.m.double_value() == spin.ket.m.double_value() + 2 {
            (spin.ket.s.value() * (spin.ket.s.value() + 1.) - spin.bra.m.value() * spin.ket.m.value()).sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    pub fn ladder_minus(spin: Braket<&Spin>) -> f64 {
        if spin.bra.s == spin.ket.s && spin.bra.m.double_value() + 2 == spin.ket.m.double_value() {
            (spin.ket.s.value() * (spin.ket.s.value() + 1.) - spin.bra.m.value() * spin.ket.m.value()).sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    pub fn dot(spin1: Braket<&Spin>, spin2: Braket<&Spin>) -> f64 {
        let val1 = Self::proj_z(spin1) * Self::proj_z(spin2);
        let val2 = 0.5 * Self::ladder_plus(spin1) * Self::ladder_minus(spin2);
        let val3 = 0.5 * Self::ladder_minus(spin1) * Self::ladder_plus(spin2);

        val1 + val2 + val3
    }

    /// Compute Clebsch-Gordan coefficient <spin1; spin2 | spin3>.
    #[inline]
    pub fn clebsch_gordan(spin1: &Spin, spin2: &Spin, spin3: &Spin) -> f64 {
        clebsch_gordan::clebsch_gordan(spin1.s, spin1.m, spin2.s, spin2.m, spin3.s, spin3.m)
    }
}

#[cfg(test)]
mod tests {
    use clebsch_gordan::{hi32, hu32};
    use hilbert_space::operator::Braket;

    use crate::{Spin, SpinOps};

    #[test]
    fn test_spin_operators() {
        let s1 = Spin::new(hu32!(7 / 2), hi32!(3 / 2));
        let s2 = Spin::new(hu32!(7 / 2), hi32!(5 / 2));
        let s3 = Spin::new(hu32!(5 / 2), hi32!(3 / 2));
        let s4 = Spin::new(hu32!(6), hi32!(4));

        let mel = SpinOps::proj_z(Braket::new(&s1, &s1));
        assert_eq!(mel, 1.5);

        let mel = SpinOps::proj_z(Braket::new(&s1, &s2));
        assert_eq!(mel, 0.);

        let mel = SpinOps::proj_z(Braket::new(&s1, &s3));
        assert_eq!(mel, 0.);

        let mel = SpinOps::ladder_plus(Braket::new(&s2, &s1));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = SpinOps::ladder_plus(Braket::new(&s1, &s2));
        assert_eq!(mel, 0.);

        let mel = SpinOps::ladder_plus(Braket::new(&s2, &s3));
        assert_eq!(mel, 0.);

        let mel = SpinOps::ladder_minus(Braket::new(&s1, &s2));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = SpinOps::ladder_minus(Braket::new(&s2, &s1));
        assert_eq!(mel, 0.);

        let mel = SpinOps::ladder_minus(Braket::new(&s3, &s2));
        assert_eq!(mel, 0.);

        let mel = SpinOps::clebsch_gordan(&s1, &s2, &s4);
        assert_eq!(mel, -f64::sqrt(7. / 11.) / 2.);

        let mel = SpinOps::clebsch_gordan(&s2, &s3, &s4);
        assert_eq!(mel, f64::sqrt(35. / 66.));
    }
}
