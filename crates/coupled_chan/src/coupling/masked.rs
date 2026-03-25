use cc_propagator::multi_channel::Matrix;
use faer::{
    unzip,
    zip,
};
use single_chan::interaction::{
    AsymptoteDep,
    Interaction,
};

use crate::coupling::RCoupling;

#[derive(Debug, Clone)]
pub struct Masked<P: Interaction> {
    interaction: P,
    masking: Matrix,
}

impl<P: Interaction> Masked<P> {
    pub fn new(interaction: P, masking: Matrix) -> Self {
        Self { interaction, masking }
    }

    pub fn masking(&self) -> &Matrix {
        &self.masking
    }
}

impl<P: Interaction> RCoupling for Masked<P> {
    fn value_inplace(&self, r: f64, channels: &mut Matrix) {
        let value = self.interaction.value(r);

        zip!(channels.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v = value * m;
        });
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
        let value = self.interaction.value(r);

        zip!(channels.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v += value * m;
        });
    }

    fn size(&self) -> usize {
        assert_eq!(self.masking.nrows(), self.masking.ncols(), "Masking is not square");
        self.masking.nrows()
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        self.interaction.asymptote_dep()
    }
}
