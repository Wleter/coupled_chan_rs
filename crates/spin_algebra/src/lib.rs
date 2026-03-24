use clebsch_gordan::half_integer::{
    HalfI32,
    HalfU32,
};
pub use clebsch_gordan::*;
use hilbert_space::operator::Braket;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinType {
    Fermionic,
    Bosonic,
}

pub trait SpinMagLike: PartialEq + Copy {
    fn s(&self) -> HalfU32;

    fn spin_type(&self) -> SpinType {
        if self.s().double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }
}

impl SpinMagLike for HalfU32 {
    #[inline]
    fn s(&self) -> HalfU32 {
        *self
    }
}

#[derive(Clone, Copy, PartialEq, Default, Hash)]
pub struct SpinPairMag<S: SpinMagLike, I: SpinMagLike> {
    pub pair: (S, I),
    pub summed: HalfU32,
}

impl<S: SpinMagLike, I: SpinMagLike> SpinPairMag<S, I> {
    pub fn new(pair: (S, I), summed: HalfU32) -> Self {
        Self { pair, summed }
    }
}

impl<S: SpinMagLike, I: SpinMagLike> SpinMagLike for SpinPairMag<S, I> {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.summed
    }
}

impl<S, I> std::fmt::Debug for SpinPairMag<S, I>
where
    S: SpinMagLike + std::fmt::Debug,
    I: SpinMagLike + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?}) {}", self.pair.0, self.pair.1, self.summed)
    }
}

#[derive(Clone, Copy, PartialEq, Default, Hash)]
pub struct Spin {
    pub s: HalfU32,
    pub m: HalfI32,
}

impl std::fmt::Debug for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.s, self.m)
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
}

pub trait SpinLike: PartialEq + Copy {
    fn s(&self) -> HalfU32;
    fn m(&self) -> HalfI32;

    fn spin_type(&self) -> SpinType {
        if self.s().double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }
}

impl<S: SpinLike> SpinMagLike for S {
    fn s(&self) -> HalfU32 {
        SpinLike::s(self)
    }
}

impl SpinLike for Spin {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.s
    }

    #[inline]
    fn m(&self) -> HalfI32 {
        self.m
    }
}

#[derive(Clone, Copy, PartialEq, Default, Hash)]
pub struct SpinPair<S: SpinMagLike, I: SpinMagLike> {
    pub pair: (S, I),
    pub spin: Spin,
}

impl<S: SpinMagLike, I: SpinMagLike> SpinPair<S, I> {
    pub fn new(pair: (S, I), spin: Spin) -> Self {
        Self { pair, spin }
    }
}

impl<S, I> std::fmt::Debug for SpinPair<S, I>
where
    S: SpinMagLike + std::fmt::Debug,
    I: SpinMagLike + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?}) {:?}", self.pair.0, self.pair.1, self.spin)
    }
}

/// Creates vector containing spin basis |s m_s >
/// for given `s`
pub fn get_spin_basis(s: HalfU32) -> Vec<Spin> {
    let ds = s.double_value() as i32;

    (-ds..=ds)
        .step_by(2)
        .map(|dms| Spin::new(s, HalfI32::from_doubled(dms)))
        .collect()
}

/// Creates vector containing combined spin basis |S M_S >
/// from given `s1` and `s2` spins
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

/// Creates vector containing combined spin magnitude (s1, s2) S
/// from vector of pair of spin magnitudes (s1, s2).
pub fn get_spin_pair_magnitudes<S1, S2>(s1: impl AsRef<[S1]>, s2: impl AsRef<[S2]>) -> Vec<SpinPairMag<S1, S2>>
where
    S1: SpinMagLike,
    S2: SpinMagLike,
{
    s1.as_ref()
        .iter()
        .flat_map(|&s1| {
            s2.as_ref().iter().flat_map(move |&s2| {
                let dspin_max = (s1.s() + s2.s()).double_value();
                let dspin_min = (s1.s().double_value() as i32 - s2.s().double_value() as i32).unsigned_abs();

                (dspin_min..=dspin_max)
                    .step_by(2)
                    .map(move |ds| SpinPairMag::new((s1, s2), HalfU32::from_doubled(ds)))
            })
        })
        .collect()
}

/// Creates vector containing combined spin basis |(s1, s2) S m_S >
/// from vector of pair of spins.
pub fn get_spin_pair_basis<S1, S2>(spins: impl AsRef<[SpinPairMag<S1, S2>]>) -> Vec<SpinPair<S1, S2>>
where
    S1: SpinMagLike,
    S2: SpinMagLike,
{
    spins
        .as_ref()
        .iter()
        .flat_map(|s| {
            let ds = s.summed.double_value() as i32;

            (-ds..=ds)
                .step_by(2)
                .map(move |dms| SpinPair::new(s.pair, Spin::new(s.summed, HalfI32::from_doubled(dms))))
        })
        .collect()
}

pub mod ops {
    use super::*;

    #[inline]
    pub fn s_sqr(spin: Braket<impl SpinMagLike>) -> f64 {
        if spin.bra == spin.ket {
            let s = spin.bra.s().value();
            s * (s + 1.)
        } else {
            0.0
        }
    }

    #[inline]
    pub fn proj_z(spin: Braket<impl SpinLike>) -> f64 {
        if spin.bra == spin.ket { spin.bra.m().value() } else { 0.0 }
    }

    #[inline]
    pub fn ladder_plus(spin: Braket<impl SpinLike>) -> f64 {
        if spin.bra.s() == spin.ket.s() && spin.bra.m().double_value() == spin.ket.m().double_value() + 2 {
            (spin.ket.s().value() * (spin.ket.s().value() + 1.) - spin.bra.m().value() * spin.ket.m().value()).sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    pub fn ladder_minus(spin: Braket<impl SpinLike>) -> f64 {
        if spin.bra.s() == spin.ket.s() && spin.bra.m().double_value() + 2 == spin.ket.m().double_value() {
            (spin.ket.s().value() * (spin.ket.s().value() + 1.) - spin.bra.m().value() * spin.ket.m().value()).sqrt()
        } else {
            0.0
        }
    }

    #[inline]
    pub fn dot(spin1: Braket<impl SpinLike>, spin2: Braket<impl SpinLike>) -> f64 {
        let val1 = proj_z(spin1) * proj_z(spin2);
        let val2 = 0.5 * ladder_plus(spin1) * ladder_minus(spin2);
        let val3 = 0.5 * ladder_minus(spin1) * ladder_plus(spin2);

        val1 + val2 + val3
    }

    /// Compute Clebsch-Gordan coefficient <spin1; spin2 | spin3>.
    #[inline]
    pub fn clebsch_gordan_coef(spin1: impl SpinLike, spin2: impl SpinLike, spin3: impl SpinLike) -> f64 {
        clebsch_gordan::clebsch_gordan(spin1.s(), spin1.m(), spin2.s(), spin2.m(), spin3.s(), spin3.m())
    }
}

#[cfg(test)]
mod tests {
    use clebsch_gordan::{
        hi32,
        hu32,
    };
    use hilbert_space::operator::Braket;

    use crate::{
        Spin,
        get_spin_pair_basis,
        get_spin_pair_magnitudes,
        ops::{
            clebsch_gordan_coef,
            ladder_minus,
            ladder_plus,
            proj_z,
        },
    };

    #[test]
    fn test_spin_operators() {
        let s1 = Spin::new(hu32!(7 / 2), hi32!(3 / 2));
        let s2 = Spin::new(hu32!(7 / 2), hi32!(5 / 2));
        let s3 = Spin::new(hu32!(5 / 2), hi32!(3 / 2));
        let s4 = Spin::new(hu32!(6), hi32!(4));

        let mel = proj_z(Braket::new(s1, s1));
        assert_eq!(mel, 1.5);

        let mel = proj_z(Braket::new(s1, s2));
        assert_eq!(mel, 0.);

        let mel = proj_z(Braket::new(s1, s3));
        assert_eq!(mel, 0.);

        let mel = ladder_plus(Braket::new(s2, s1));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = ladder_plus(Braket::new(s1, s2));
        assert_eq!(mel, 0.);

        let mel = ladder_plus(Braket::new(s2, s3));
        assert_eq!(mel, 0.);

        let mel = ladder_minus(Braket::new(s1, s2));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = ladder_minus(Braket::new(s2, s1));
        assert_eq!(mel, 0.);

        let mel = ladder_minus(Braket::new(s3, s2));
        assert_eq!(mel, 0.);

        let mel = clebsch_gordan_coef(s1, s2, s4);
        assert_eq!(mel, -f64::sqrt(7. / 11.) / 2.);

        let mel = clebsch_gordan_coef(s2, s3, s4);
        assert_eq!(mel, f64::sqrt(35. / 66.));
    }

    #[test]
    #[rustfmt::skip]
    fn test_summed_spins() {
        let s1 = hu32!(1 / 2);
        let i1 = hu32!(3 / 2);
        let s2 = hu32!(0);
        let i2 = hu32!(1);

        let s_tot = get_spin_pair_magnitudes(vec![s1], vec![s2]);
        let i_tot = get_spin_pair_magnitudes(vec![i1], vec![i2]);

        let f_tot = get_spin_pair_magnitudes(s_tot, i_tot);
        let expected = [
            "((1/2, 0) 1/2, (3/2, 1) 1/2) 0",
            "((1/2, 0) 1/2, (3/2, 1) 1/2) 1",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 1",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3",
        ];
        for (f, exp) in f_tot.iter().zip(&expected) {
            assert_eq!(&format!("{f:?}"), exp)
        }

        let f_mf = get_spin_pair_basis(f_tot);
        let expected = [
            "((1/2, 0) 1/2, (3/2, 1) 1/2) 0 0",

            "((1/2, 0) 1/2, (3/2, 1) 1/2) 1 -1",
            "((1/2, 0) 1/2, (3/2, 1) 1/2) 1 0",
            "((1/2, 0) 1/2, (3/2, 1) 1/2) 1 1",

            "((1/2, 0) 1/2, (3/2, 1) 3/2) 1 -1",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 1 0",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 1 1",

            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2 -2",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2 -1",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2 0",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2 1",
            "((1/2, 0) 1/2, (3/2, 1) 3/2) 2 2",

            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2 -2",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2 -1",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2 0",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2 1",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 2 2",
            
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 -3",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 -2",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 -1",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 0",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 1",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 2",
            "((1/2, 0) 1/2, (3/2, 1) 5/2) 3 3",
        ];

        for (f_mf, exp) in f_mf.iter().zip(&expected) {
            assert_eq!(&format!("{f_mf:?}"), exp)
        }
    }
}
