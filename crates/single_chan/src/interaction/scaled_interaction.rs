use crate::interaction::Interaction;

#[derive(Debug, Clone)]
pub struct ScaledInteraction<P: Interaction> {
    pub scaling: f64,
    interaction: P,
}

impl<P: Interaction> ScaledInteraction<P> {
    pub fn new(interaction: P, scaling: f64) -> Self {
        Self { interaction, scaling }
    }
}

impl<P: Interaction> Interaction for ScaledInteraction<P> {
    fn value(&self, r: f64) -> f64 {
        self.scaling * self.interaction.value(r)
    }
}
