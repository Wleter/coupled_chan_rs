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
        if self.scaling == 0.0 {
            super::AsymptoteDep::Const
        } else if self.scaling > 0.0 {
            self.interaction.asymptote_dep()
        } else {
            match self.interaction.asymptote_dep() {
                super::AsymptoteDep::Const => super::AsymptoteDep::Const,
                super::AsymptoteDep::ExpVanishing => super::AsymptoteDep::Growing,
                super::AsymptoteDep::PowerLawVanishing(_) => super::AsymptoteDep::Growing,
                super::AsymptoteDep::Growing => super::AsymptoteDep::Unknown,
                super::AsymptoteDep::Other => super::AsymptoteDep::Unknown,
                super::AsymptoteDep::Unknown => super::AsymptoteDep::Unknown,
            }
        }
    }
}
