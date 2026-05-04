pub mod ops;
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

pub trait SpinMagLike: PartialEq + Copy + std::fmt::Debug + Send + Sync + 'static {
    fn s(&self) -> HalfU32;

    fn spin_type(&self) -> SpinType {
        if self.s().double_value() & 1 == 1 {
            SpinType::Fermionic
        } else {
            SpinType::Bosonic
        }
    }

    #[inline]
    fn squared(&self) -> f64 {
        let s = self.s().value();

        s * (s + 1.)
    }

    #[inline]
    fn dim(&self) -> u32 {
        self.s().double_value() + 1
    }
}

impl SpinMagLike for HalfU32 {
    #[inline]
    fn s(&self) -> HalfU32 {
        *self
    }
}

impl SpinMagLike for u32 {
    #[inline]
    fn s(&self) -> HalfU32 {
        (*self).into()
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
    pub fn new(s: impl Into<HalfU32>, m: impl Into<HalfI32>) -> Self {
        Self {
            s: s.into(),
            m: m.into(),
        }
    }

    pub fn zero() -> Self {
        Self {
            s: 0.into(),
            m: 0.into(),
        }
    }
}

pub trait SpinLike: SpinMagLike {
    fn m(&self) -> HalfI32;

    /// returns (-1)^(s - m)
    fn phase_factor(&self) -> f64 {
        (-1.0f64).powi((self.s().double_value() as i32 - self.m().double_value()) / 2)
    }
}

impl SpinMagLike for Spin {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.s
    }
}
impl SpinLike for Spin {
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

    pub fn as_spin_pair_mag(&self) -> SpinPairMag<S, I> {
        SpinPairMag::new(self.pair, self.spin.s)
    }
}

impl<S: SpinMagLike, I: SpinMagLike> SpinMagLike for SpinPair<S, I> {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.spin.s
    }
}
impl<S: SpinMagLike, I: SpinMagLike> SpinLike for SpinPair<S, I> {
    #[inline]
    fn m(&self) -> HalfI32 {
        self.spin.m
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

#[macro_export]
macro_rules! spin {
    // 1. Both left and right are parenthesized (they are nested sub-trees)
    ((($($s1:tt)+), ($($s2:tt)+)), $s:expr, $m:expr) => {
        $crate::SpinPair::new((spin!($($s1)+), spin!($($s2)+)), Spin::new($s, $m))
    };
    ((($($s1:tt)+), ($($s2:tt)+)), $s:expr) => {
        $crate::SpinPairMag::new((spin!($($s1)+), spin!($($s2)+)), $s)
    };

    // 2. Only one is parenthesized
    ((($($s1:tt)+), $s2:expr), $s:expr, $m:expr) => {
        $crate::SpinPair::new((spin!($($s1)+), $s2), Spin::new($s, $m))
    };
    ((($($s1:tt)+), $s2:expr), $s:expr) => {
        $crate::SpinPairMag::new((spin!($($s1)+), $s2), $s)
    };
    (($s1:expr, ($($s2:tt)+)), $s:expr, $m:expr) => {
        $crate::SpinPair::new(($s1, spin!($($s2)+)), Spin::new($s, $m))
    };
    (($s1:expr, ($($s2:tt)+)), $s:expr) => {
        $crate::SpinPairMag::new(($s1, spin!($($s2)+)), $s)
    };

    // 4. Base cases: Neither is parenthesized (they are leaf nodes)
    (($s1:expr, $s2:expr), $s:expr, $m:expr) => {
        $crate::SpinPair::new(($s1, $s2), $crate::Spin::new($s, $m))
    };
    (($s1:expr, $s2:expr), $s:expr) => {
        $crate::SpinPairMag::new(($s1, $s2), $s)
    };
    ($s1:expr, $s2:expr) => {
        $crate::Spin::new($s1, $s2)
    };

    // 5. Fallback for single expressions (safeguard for leaf nodes)
    ($e:expr) => {
        $e
    };
}

/// Creates vector containing spin basis |s m_s >
/// for given `s`
pub fn get_spin_basis(s: impl SpinMagLike) -> Vec<Spin> {
    let s = s.s();
    let ds = s.double_value() as i32;

    (-ds..=ds)
        .step_by(2)
        .map(|dms| Spin::new(s, HalfI32::from_doubled(dms)))
        .collect()
}

/// Creates vector containing combined spin basis |S M_S >
/// from given `s1` and `s2` spins
pub fn get_summed_spin_basis(s1: impl SpinMagLike, s2: impl SpinMagLike) -> Vec<Spin> {
    let s1 = s1.s();
    let s2 = s2.s();
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

#[cfg(test)]
mod tests {
    use clebsch_gordan::{
        hi32,
        hu32,
    };

    use crate::{
        Spin,
        get_spin_pair_basis,
        get_spin_pair_magnitudes,
    };

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
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3)),
        ];
        for (f, exp) in f_tot.iter().zip(&expected) {
            assert_eq!(f, exp)
        }

        let f_mf = get_spin_pair_basis(f_tot);
        let expected = [
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(0), hi32!(0)),

            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(1), hi32!(-1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(1), hi32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(1/2))), hu32!(1), hi32!(1)),

            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(1), hi32!(-1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(1), hi32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(1), hi32!(1)),

            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2), hi32!(-2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2), hi32!(-1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2), hi32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2), hi32!(1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(3/2))), hu32!(2), hi32!(2)),

            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2), hi32!(-2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2), hi32!(-1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2), hi32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2), hi32!(1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(2), hi32!(2)),

            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(-3)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(-2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(-1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(0)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(1)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(2)),
            spin!((((hu32!(1/2), hu32!(0)), hu32!(1/2)), ((hu32!(3/2), hu32!(1)), hu32!(5/2))), hu32!(3), hi32!(3)),
        ];

        for (f_mf, exp) in f_mf.iter().zip(&expected) {
            assert_eq!(f_mf, exp)
        }
    }
}
