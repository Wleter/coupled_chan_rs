use std::sync::Arc;

use cc_constants::{
    Unit,
    units::{
        Quantity,
        atomic_units::{
            AuEnergy,
            AuMass,
        },
    },
};

use crate::interaction::dispersion::Centrifugal;

pub mod dispersion;
pub mod func_potential;
pub mod interpolated;
pub mod morse_long_range;
pub mod scaled_interaction;

pub trait Interaction {
    fn value(&self, r: f64) -> f64;
}

pub trait WFunction {
    fn value(&self, r: f64) -> f64;
    fn asymptote(&self) -> f64;
    fn l(&self) -> u32;
}

pub use cc_qol_utils::Pair;
impl<P: Interaction, V: Interaction> Interaction for Pair<P, V> {
    fn value(&self, r: f64) -> f64 {
        self.first.value(r) + self.second.value(r)
    }
}

pub use cc_qol_utils::Composite;
impl<P: Interaction> Interaction for Composite<P> {
    fn value(&self, r: f64) -> f64 {
        self.components.iter().fold(0., |acc, p| acc + p.value(r))
    }
}

pub struct RedInteraction<'a, P: Interaction> {
    energy: f64,
    mass: f64,
    interaction: &'a P,
    centrifugal: Centrifugal,
}

impl<'a, P: Interaction> RedInteraction<'a, P> {
    pub fn new(
        interaction: &'a P,
        mass: Quantity<impl Unit<Base = AuMass>>,
        energy: Quantity<impl Unit<Base = AuEnergy>>,
        l: u32,
    ) -> Self {
        Self {
            energy: energy.to(AuEnergy).value(),
            mass: mass.to(AuMass).value(),
            interaction,
            centrifugal: Centrifugal::new(l, mass),
        }
    }
}

impl<'a, P: Interaction> WFunction for RedInteraction<'a, P> {
    fn value(&self, r: f64) -> f64 {
        2. * self.mass * (self.energy - self.interaction.value(r) - self.centrifugal.value(r))
    }

    fn asymptote(&self) -> f64 {
        2. * self.mass * self.energy
    }

    fn l(&self) -> u32 {
        self.centrifugal.l
    }
}

#[derive(Clone)]
pub struct DynInteraction(Arc<dyn Interaction + Send + Sync>);

impl DynInteraction {
    pub fn new<P: Interaction + 'static + Send + Sync>(potential: P) -> Self {
        Self(Arc::new(potential))
    }
}

impl Interaction for DynInteraction {
    fn value(&self, r: f64) -> f64 {
        self.0.value(r)
    }
}
