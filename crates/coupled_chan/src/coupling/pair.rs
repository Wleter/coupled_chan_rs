use crate::coupling::VanishingCoupling;


#[derive(Debug, Clone)]
pub struct Pair<P: VanishingCoupling, C: VanishingCoupling> {
    first: P,
    second: C
}

impl<P: VanishingCoupling, C: VanishingCoupling> Pair<P, C> {
    pub fn new(first: P, second: C) -> Self {
        assert_eq!(first.size(), second.size(), "Couplings have different channel number");

        Self { first, second }
    }
}

impl<P: VanishingCoupling, C: VanishingCoupling> VanishingCoupling for Pair<P, C> {
    fn value_inplace(&self, r: f64, channels: &mut crate::Channels) {
        self.first.value_inplace(r, channels);
        self.second.value_inplace_add(r, channels);
    }

    fn value_inplace_add(&self, r: f64, channels: &mut crate::Channels) {
        self.first.value_inplace_add(r, channels);
        self.second.value_inplace_add(r, channels);
    }

    fn size(&self) -> usize {
        self.first.size()
    }
}
