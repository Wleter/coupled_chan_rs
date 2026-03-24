use std::sync::Arc;

use cc_propagator::single_channel::WFunction;

pub mod dispersion;
pub mod func_potential;
pub mod interpolated;
pub mod morse_long_range;
pub mod scaled;

/// Asymptotic (r going to infinity) 
/// behavior for the [`Interaction`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AsymptoteDep {
    Const,
    ExpVanishing,
    PowerLawVanishing(u8),
    Growing,
    Other,
    Unknown,
}

impl PartialOrd for AsymptoteDep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let order_self = match self {
            AsymptoteDep::Const => i32::MIN,
            AsymptoteDep::ExpVanishing => i32::MIN + 1,
            AsymptoteDep::PowerLawVanishing(n) => -(*n as i32),
            AsymptoteDep::Growing => 0,
            AsymptoteDep::Other => return None,
            AsymptoteDep::Unknown => return None,
        };

        let order_other = match &other {
            AsymptoteDep::Const => i32::MIN,
            AsymptoteDep::ExpVanishing => i32::MIN + 1,
            AsymptoteDep::PowerLawVanishing(n) => -(*n as i32),
            AsymptoteDep::Growing => 0,
            AsymptoteDep::Other => return None,
            AsymptoteDep::Unknown => return None,
        };

        Some(order_self.cmp(&order_other))
    }
}

/// Trait for modeling interaction V(R)
/// in the Hamiltonian p^2/2m + V(R)
pub trait Interaction {
    fn value(&self, r: f64) -> f64;
    fn asymptote_dep(&self) -> AsymptoteDep;
}

pub use cc_qol_utils::Pair;
impl<P: Interaction, V: Interaction> Interaction for Pair<P, V> {
    fn value(&self, r: f64) -> f64 {
        self.first.value(r) + self.second.value(r)
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        if self.first.asymptote_dep() < self.second.asymptote_dep() {
            self.second.asymptote_dep()
        } else {
            self.first.asymptote_dep()
        }
    }
}

pub use cc_qol_utils::Composite;
impl<P: Interaction> Interaction for Composite<P> {
    fn value(&self, r: f64) -> f64 {
        self.components.iter().fold(0., |acc, p| acc + p.value(r))
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        let mut maximal = AsymptoteDep::Const;
        for c in &self.components {
            let dep = c.asymptote_dep();
            if dep > maximal {
                maximal = dep;
            }
        }

        maximal
    }
}

/// Reduced centrifugal term l(l+1) / R^2
pub struct RedCentrifugal(u32);

impl WFunction for RedCentrifugal {
    fn value(&self, r: f64) -> f64 {
        ((self.0 + 1) * self.0) as f64 / (r * r)
    }
}

/// Struct describing [`WFunction`] in a collision scenerio with
/// given mass, energy, angular momentum and interaction
pub struct CollisionWFunction<I> {
    mass: f64,
    energy: f64,
    interaction: I,
    centrifugal: RedCentrifugal,
}

impl<I: Interaction> CollisionWFunction<I> {
    pub fn new(interaction: I, mass: f64, energy: f64, l: u32) -> Self {
        Self {
            centrifugal: RedCentrifugal(l),
            mass,
            energy,
            interaction,
        }
    }

    pub fn l(&self) -> u32 {
        self.centrifugal.0
    }

    /// value without centrifugal term
    pub fn value_interaction(&self, r: f64) -> f64 {
        2.0 * self.mass * (self.energy - self.interaction.value(r))
    }

    pub fn interaction_asymptote_dep(&self) -> AsymptoteDep {
        self.interaction.asymptote_dep()
    }
}

impl<I: Interaction> WFunction for CollisionWFunction<I> {
    fn value(&self, r: f64) -> f64 {
        self.value_interaction(r) + self.centrifugal.value(r)
    }
}

/// Type erased [`Interaction`] implementation.
#[derive(Clone)]
pub struct DynInteraction(Arc<dyn Interaction + Send + Sync>);

impl std::fmt::Debug for DynInteraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynInteraction").finish()
    }
}

impl DynInteraction {
    pub fn new<P: Interaction + 'static + Send + Sync>(potential: P) -> Self {
        Self(Arc::new(potential))
    }
}

impl Interaction for DynInteraction {
    fn value(&self, r: f64) -> f64 {
        self.0.value(r)
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        self.0.asymptote_dep()
    }
}
