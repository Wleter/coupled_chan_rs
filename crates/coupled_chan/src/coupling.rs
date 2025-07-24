pub mod composite;
pub mod masked;

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
    entrance_channel: usize,

    asymptote_channels: Channels,
    centrifugal: MultiCentrifugal,
}

impl Asymptote {
    pub fn new_diagonal(levels: Levels, entrance_channel: usize) -> Self {
        let asymptote_channels = Mat::from_fn(levels.asymptote.len(), levels.asymptote.len(), |i, j| {
            if i != j {
                return 0.;
            }

            levels.asymptote[i]
        });

        let centrifugal = MultiCentrifugal::new_diagonal(&levels);

        Self {
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

/// Multichannel centrifugal term L^2 / r^2
pub struct MultiCentrifugal {
    mask: Channels,
}

impl MultiCentrifugal {
    pub fn new_diagonal(levels: &Levels) -> Self {
        let mask = Mat::from_fn(levels.l.len(), levels.l.len(), |i, j| {
            if i != j {
                return 0.;
            }

            (levels.l[i] * (levels.l[i] + 1)) as f64
        });

        Self { mask: Channels(mask) }
    }

    pub fn value_inplace_add(&self, r: f64, channels: &mut Channels) {
        zip!(channels.0.as_mut(), self.mask.0.as_ref()).for_each(|unzip!(o, m)| *o += m / (r * r));
    }
}

pub struct RedCoupling<'a, P: VanishingCoupling> {
    coupling: &'a P,
    mass: f64,
    energy: f64,
    asymptote: Asymptote,
    pub(crate) id: Channels,
}

impl<'a, P: VanishingCoupling> RedCoupling<'a, P> {
    pub fn new(coupling: &'a P, mass: Quantity64<AuMass>, energy: Quantity64<AuEnergy>, asymptote: Asymptote) -> Self {
        Self {
            id: Channels(Mat::zeros(coupling.size(), coupling.size())),
            energy: energy.value() - asymptote.entrance_energy(),
            mass: mass.value(),
            coupling,
            asymptote,
        }
    }

    pub fn asymptote(&self) -> &Asymptote {
        &self.asymptote
    }

    pub fn value_inplace(&self, r: f64, channels: &mut Channels) {
        self.coupling.value_inplace(r, channels);
        channels.0 += &self.asymptote.asymptote_channels.0;

        zip!(channels.0.as_mut(), self.id.0.as_ref()).for_each(|unzip!(o, u)| *o = 2.0 * self.mass * (self.energy * u - *o));

        self.asymptote.centrifugal.value_inplace_add(r, channels);
    }
}
