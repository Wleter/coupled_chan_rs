use faer::{unzip, zip};
use single_chan::interaction::Interaction;

use crate::{Operator, coupling::VanishingCoupling};

#[derive(Debug, Clone)]
pub struct Masked<P: Interaction> {
    interaction: P,
    masking: Operator,
}

impl<P: Interaction> Masked<P> {
    pub fn new(interaction: P, masking: Operator) -> Self {
        Self { interaction, masking }
    }

    pub fn masking(&self) -> &Operator {
        &self.masking
    }
}

impl<P: Interaction> VanishingCoupling for Masked<P> {
    fn value_inplace(&self, r: f64, channels: &mut Operator) {
        channels.fill(0.);
        let value = self.interaction.value(r);

        zip!(channels.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v = value * m;
        });
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Operator) {
        let value = self.interaction.value(r);

        zip!(channels.as_mut(), self.masking.as_ref()).for_each(|unzip!(v, m)| {
            *v += value * m;
        });
    }

    fn size(&self) -> usize {
        self.masking.size()
    }
}
