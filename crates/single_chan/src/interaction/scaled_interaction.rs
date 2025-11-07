use crate::interaction::Interaction;

#[derive(Debug, Clone)]
pub struct ScaledInteraction<P: Interaction> {
    pub scaling: f64,
    interaction: P,
}

impl<P: Interaction> ScaledInteraction<P> {
    pub fn new(interaction: P) -> Self {
        Self {
            interaction,
            scaling: 1.,
        }
    }

    pub fn scale(&mut self, scaling: f64) {
        self.scaling *= scaling
    }
}

impl<P: Interaction> Interaction for ScaledInteraction<P> {
    fn value(&self, r: f64) -> f64 {
        self.scaling * self.interaction.value(r)
    }
}
