use single_chan::interaction::Interaction;

use crate::coupling::VanishingCoupling;

#[derive(Debug, Clone)]
pub struct Diagonal<P: Interaction> {
    couplings: Vec<P>,
}

impl<P: Interaction> Default for Diagonal<P> {
    fn default() -> Self {
        Self { couplings: Vec::new() }
    }
}

impl<P: Interaction> Diagonal<P> {
    pub fn new(couplings: Vec<P>) -> Self {
        Self { couplings }
    }
}

impl<P: Interaction> VanishingCoupling for Diagonal<P> {
    fn value_inplace(&self, r: f64, channels: &mut crate::Operator) {
        assert_eq!(channels.size(), self.size(), "Number mismatch between channels and coupling");
        channels.0.fill(0.);

        for (c, a) in channels
            .0
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.couplings.iter())
        {
            *c = a.value(r)
        }
    }

    fn value_inplace_add(&self, r: f64, channels: &mut crate::Operator) {
        assert_eq!(channels.size(), self.size(), "Number mismatch between channels and coupling");

        for (c, a) in channels
            .0
            .diagonal_mut()
            .column_vector_mut()
            .iter_mut()
            .zip(self.couplings.iter())
        {
            *c += a.value(r)
        }
    }

    fn size(&self) -> usize {
        self.couplings.len()
    }
}
