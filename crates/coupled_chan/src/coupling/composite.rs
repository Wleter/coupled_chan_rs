use crate::coupling::RCoupling;

use cc_propagator::multi_channel::Matrix;
pub use cc_qol_utils::Composite;

impl<P: RCoupling> RCoupling for Composite<P> {
    fn value_inplace(&self, r: f64, channels: &mut Matrix) {
        let mut couplings = self.components.iter();

        if let Some(c) = couplings.next() {
            c.value_inplace(r, channels);
        }

        for c in couplings {
            c.value_inplace_add(r, channels);
        }
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
        for c in self.components.iter() {
            c.value_inplace_add(r, channels);
        }
    }

    fn size(&self) -> usize {
        if let Some(c) = self.components.first() {
            assert!(
                self.components.iter().all(|x| x.size() == c.size()),
                "Not all coupling in Composite have the same channel number"
            );

            c.size()
        } else {
            0
        }
    }

    fn asymptote_dep(&self) -> single_chan::interaction::AsymptoteDep {
        todo!()
    }
}
