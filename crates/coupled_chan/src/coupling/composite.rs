use crate::coupling::VanishingCoupling;

pub use cc_qol_utils::Composite;

impl<P: VanishingCoupling> VanishingCoupling for Composite<P> {
    fn value_inplace(&self, r: f64, channels: &mut crate::Operator) {
        let mut couplings = self.components.iter();

        if let Some(c) = couplings.next() {
            c.value_inplace(r, channels);
        }

        for c in couplings {
            c.value_inplace_add(r, channels);
        }
    }

    fn value_inplace_add(&self, r: f64, channels: &mut crate::Operator) {
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
}
