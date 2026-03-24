use crate::interaction::Interaction;

#[derive(Debug, Clone)]
pub struct Scaled<I: Interaction> {
    pub scaling: f64,
    pub interaction: I,
}

impl<I: Interaction> Scaled<I> {
    pub fn new(interaction: I) -> Self {
        Self {
            interaction,
            scaling: 1.,
        }
    }

    pub fn scale(&mut self, scaling: f64) {
        self.scaling *= scaling
    }

    pub fn set_scaling(&mut self, scaling: f64) {
        self.scaling = scaling
    }
}

impl<I: Interaction> Interaction for Scaled<I> {
    fn value(&self, r: f64) -> f64 {
        self.scaling * self.interaction.value(r)
    }

    fn asymptote_dep(&self) -> super::AsymptoteDep {
        self.interaction.asymptote_dep()
    }
}
