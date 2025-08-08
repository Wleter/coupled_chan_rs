use faer::{unzip, zip};
use single_chan::interaction::Interaction;

use crate::{Channels, coupling::VanishingCoupling};

#[derive(Debug, Clone)]
pub struct Masked<P: Interaction> {
    interaction: P,
    masking: Channels,
}

impl<P: Interaction> Masked<P> {
    pub fn new(interaction: P, masking: Channels) -> Self {
        Self { interaction, masking }
    }

    pub fn masking(&self) -> &Channels {
        &self.masking
    }
}

impl<P: Interaction> VanishingCoupling for Masked<P> {
    fn value_inplace(&self, r: f64, channels: &mut Channels) {
        channels.0.fill(0.);
        let value = self.interaction.value(r);

        zip!(channels.0.as_mut(), self.masking.0.as_ref()).for_each(|unzip!(v, m)| {
            *v = value * m;
        });
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Channels) {
        let value = self.interaction.value(r);

        zip!(channels.0.as_mut(), self.masking.0.as_ref()).for_each(|unzip!(v, m)| {
            *v += value * m;
        });
    }

    fn size(&self) -> usize {
        self.masking.size()
    }
}
