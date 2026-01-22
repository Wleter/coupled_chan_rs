use crate::coupling::VanishingCoupling;
pub use cc_qol_utils::Pair;

impl<P: VanishingCoupling, C: VanishingCoupling> VanishingCoupling for Pair<P, C> {
    fn value_inplace(&self, r: f64, channels: &mut crate::Operator) {
        self.first.value_inplace(r, channels);
        self.second.value_inplace_add(r, channels);
    }

    fn value_inplace_add(&self, r: f64, channels: &mut crate::Operator) {
        self.first.value_inplace_add(r, channels);
        self.second.value_inplace_add(r, channels);
    }

    fn size(&self) -> usize {
        assert_eq!(
            self.first.size(),
            self.second.size(),
            "Couplings in pair have different channel number"
        );

        self.first.size()
    }
}
