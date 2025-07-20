use constants::units::{
    Quantity64,
    atomic_units::{AuEnergy, AuMass},
};

pub mod composite;
pub mod dispersion;
pub mod func_potential;
pub mod morse_long_range;

pub trait Interaction {
    fn value(&self, r: f64) -> f64;
    fn asymptote(&self) -> f64;
}

pub struct RedInteraction<'a, P: Interaction> {
    energy: f64,
    mass: f64,
    interaction: &'a P,
}

impl<'a, P: Interaction> RedInteraction<'a, P> {
    pub fn new(interaction: &'a P, mass: Quantity64<AuMass>, energy: Quantity64<AuEnergy>) -> Self {
        Self {
            energy: energy.to_base() - interaction.asymptote(),
            mass: mass.to_base(),
            interaction,
        }
    }

    pub fn value(&self, r: f64) -> f64 {
        2. * self.mass * (self.energy - self.interaction.value(r))
    }
}

pub struct DynInteraction {
    interaction: Box<dyn Interaction>,
}

impl DynInteraction {
    pub fn new<P: Interaction + 'static>(potential: P) -> Self {
        Self {
            interaction: Box::new(potential),
        }
    }
}

impl Interaction for DynInteraction {
    fn value(&self, r: f64) -> f64 {
        self.interaction.value(r)
    }

    fn asymptote(&self) -> f64 {
        self.interaction.asymptote()
    }
}
