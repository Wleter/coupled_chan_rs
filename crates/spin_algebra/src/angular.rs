use clebsch_gordan::half_integer::{
    HalfI32,
    HalfU32,
};

use crate::{
    Spin,
    SpinLike,
    SpinMagLike,
};

#[derive(Clone, Copy, PartialEq, Default, Hash)]
pub struct Angular {
    pub l: HalfU32,
    pub m: HalfI32,
}

impl std::fmt::Debug for Angular {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.l, self.m)
    }
}

impl SpinMagLike for Angular {
    #[inline]
    fn s(&self) -> HalfU32 {
        self.l
    }
}

impl SpinLike for Angular {
    #[inline]
    fn m(&self) -> HalfI32 {
        self.m
    }
}

impl From<Spin> for Angular {
    fn from(value: Spin) -> Self {
        assert!(value.s.double_value() & 1 == 0, "Can only convert integers");
        assert!(value.m.double_value() & 1 == 0, "Can only convert integers");

        Self { l: value.s, m: value.m }
    }
}

impl From<Angular> for Spin {
    fn from(val: Angular) -> Spin {
        Spin::new(val.l, val.m)
    }
}

impl Angular {
    pub fn new(l: u32, m: i32) -> Self {
        Self {
            l: l.into(),
            m: m.into(),
        }
    }

    pub fn l_value(&self) -> u32 {
        self.l.double_value() / 2
    }

    pub fn m_value(&self) -> i32 {
        self.m.double_value() / 2
    }
}

pub fn angular_range(l_max: u32) -> Vec<Angular> {
    (0..=l_max).map(|x| Angular::new(x, 0)).collect()
}
