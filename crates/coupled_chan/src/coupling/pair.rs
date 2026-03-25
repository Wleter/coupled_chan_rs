use crate::coupling::RCoupling;
use cc_propagator::multi_channel::Matrix;
pub use cc_qol_utils::Pair;

impl<P: RCoupling, C: RCoupling> RCoupling for Pair<P, C> {
    fn value_inplace(&self, r: f64, channels: &mut Matrix) {
        self.first.value_inplace(r, channels);
        self.second.value_inplace_add(r, channels);
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
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

    fn asymptote_dep(&self) -> single_chan::interaction::AsymptoteDep {
        let first = self.first.asymptote_dep();
        let second = self.second.asymptote_dep();

        if first < second { second } else { first }
    }
}
