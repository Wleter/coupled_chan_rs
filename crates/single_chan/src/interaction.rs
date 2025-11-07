use constants::units::{
    Quantity,
    atomic_units::{AuEnergy, AuMass},
};

use crate::interaction::dispersion::Centrifugal;

pub mod composite_int;
pub mod dispersion;
pub mod func_potential;
pub mod morse_long_range;
pub mod scaled_interaction;

pub trait Interaction {
    fn value(&self, r: f64) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub asymptote: f64,
    pub l: u32,
}

impl Level {
    pub fn new(l: u32, asymptote: Quantity<AuEnergy>) -> Self {
        Self {
            asymptote: asymptote.value(),
            l,
        }
    }
}

pub trait WFunction {
    fn value(&self, r: f64) -> f64;
    fn asymptote(&self) -> f64;
    fn l(&self) -> u32;
}

pub struct RedInteraction<'a, P: Interaction> {
    energy: f64,
    mass: f64,
    interaction: &'a P,
    centrifugal: Centrifugal,
    level: Level,
}

impl<'a, P: Interaction> RedInteraction<'a, P> {
    pub fn new(interaction: &'a P, mass: Quantity<AuMass>, energy: Quantity<AuEnergy>, level: Level) -> Self {
        Self {
            energy: energy.value() - level.asymptote,
            mass: mass.value(),
            interaction,
            centrifugal: Centrifugal::new(level.l, mass),
            level,
        }
    }
}

impl<'a, P: Interaction> WFunction for RedInteraction<'a, P> {
    fn value(&self, r: f64) -> f64 {
        2. * self.mass * (self.energy - self.interaction.value(r) - self.centrifugal.value(r))
    }

    fn asymptote(&self) -> f64 {
        2. * self.mass * (self.energy - self.level.asymptote)
    }

    fn l(&self) -> u32 {
        self.level.l
    }
}

pub struct DynInteraction(Box<dyn Interaction>);

impl DynInteraction {
    pub fn new<P: Interaction + 'static>(potential: P) -> Self {
        Self(Box::new(potential))
    }
}

impl Interaction for DynInteraction {
    fn value(&self, r: f64) -> f64 {
        self.0.value(r)
    }
}
