pub mod composite;
pub mod masked;
pub mod pair;
pub mod diagonal;

use constants::units::{
    Quantity64,
    atomic_units::{AuEnergy, AuMass},
};
use faer::{Mat, unzip, zip};

use crate::Channels;

pub trait VanishingCoupling {
    fn value_inplace(&self, r: f64, channels: &mut Channels);
    fn value_inplace_add(&self, r: f64, channels: &mut Channels);
    fn size(&self) -> usize;
}

pub struct Levels {
    pub l: Vec<u32>,
    pub asymptote: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AngularBlocks {
    pub l: Vec<u32>,
    pub angular_blocks: Vec<Channels>,
}

pub struct Asymptote {
    levels: Levels,
    transformation: Option<Channels>,

    pub entrance_channel: usize,
    pub red_mass: f64,
    pub energy: f64,

    asymptote_channels: Channels,
    centrifugal: MultiCentrifugal,
}

impl Asymptote {
    pub fn new_diagonal(mass: Quantity64<AuMass>, energy: Quantity64<AuEnergy>, levels: Levels, entrance_channel: usize) -> Self {
        let asymptote_channels = Mat::from_fn(levels.asymptote.len(), levels.asymptote.len(), |i, j| {
            if i != j {
                return 0.;
            }

            levels.asymptote[i]
        });

        let centrifugal = MultiCentrifugal::new_diagonal(&levels, mass);

        Self {
            red_mass: mass.value(),
            energy: energy.value() - levels.asymptote[entrance_channel],

            entrance_channel,
            levels,

            transformation: None,
            asymptote_channels: Channels(asymptote_channels),
            centrifugal,
        }
    }

    pub fn levels(&self) -> &Levels {
        &self.levels
    }

    pub fn entrance_energy(&self) -> f64 {
        self.levels.asymptote[self.entrance_channel]
    }

    pub fn transformation(&self) -> &Option<Channels> {
        &self.transformation
    }
}

/// Multichannel centrifugal term L^2 / (2 m r^2)
pub struct MultiCentrifugal {
    mask: Channels,
    mass: f64,
}

impl MultiCentrifugal {
    pub fn new_diagonal(levels: &Levels, mass: Quantity64<AuMass>) -> Self {
        let mask = Mat::from_fn(levels.l.len(), levels.l.len(), |i, j| {
            if i != j {
                return 0.;
            }

            (levels.l[i] * (levels.l[i] + 1)) as f64
        });

        Self { mask: Channels(mask), mass: mass.value() }
    }

    pub fn value_inplace_add(&self, r: f64, channels: &mut Channels) {
        zip!(channels.0.as_mut(), self.mask.0.as_ref())
            .for_each(|unzip!(o, m)| *o += m / (2. * self.mass * r * r));
    }
}

pub struct RedCoupling<P: VanishingCoupling> {
    pub coupling: P,
    pub asymptote: Asymptote,
    pub id: Channels,
}

impl<'a, P: VanishingCoupling> RedCoupling<P> {
    pub fn new(coupling: P, asymptote: Asymptote) -> Self {
        assert_eq!(coupling.size(), asymptote.asymptote_channels.size(), "mismatched sizes between asymptote and coupling");

        Self {
            id: Channels(Mat::zeros(coupling.size(), coupling.size())),
            coupling,
            asymptote,
        }
    }

    pub fn value_inplace(&self, r: f64, channels: &mut Channels) {
        self.coupling.value_inplace(r, channels);
        channels.0 += &self.asymptote.asymptote_channels.0;
        self.asymptote.centrifugal.value_inplace_add(r, channels);

        zip!(channels.0.as_mut(), self.id.0.as_ref())
            .for_each(|unzip!(c, i)| *c = 2.0 * self.asymptote.red_mass * (self.asymptote.energy * i - *c));
    }

    pub fn size(&self) -> usize {
        self.coupling.size()
    }
}
