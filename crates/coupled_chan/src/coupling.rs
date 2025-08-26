pub mod composite;
pub mod diagonal;
pub mod masked;
pub mod pair;

use constants::units::{
    Quantity,
    atomic_units::{AuEnergy, AuMass},
};
use faer::{Mat, unzip, zip};
use matrix_utils::faer::diagonalize;

use crate::Channels;

pub trait VanishingCoupling {
    fn value_inplace(&self, r: f64, channels: &mut Channels);
    fn value_inplace_add(&self, r: f64, channels: &mut Channels);
    fn size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct Levels {
    pub l: Vec<u32>,
    pub asymptote: Vec<f64>,
}

impl Levels {
    pub fn as_channels(&self) -> Channels {
        let mut channels = Channels::zeros(self.l.len());

        for (c, &a) in channels.0.diagonal_mut().column_vector_mut().iter_mut().zip(&self.asymptote) {
            *c = a
        }

        channels
    }
}

#[derive(Clone, Debug)]
pub struct AngularBlocks {
    pub l: Vec<u32>,
    pub angular_blocks: Vec<Channels>,
}

impl AngularBlocks {
    pub fn size(&self) -> usize {
        self.angular_blocks.iter().map(|b| b.size()).sum()
    }

    pub fn diagonalize(&self) -> (Levels, Channels) {
        let n = self.size();
        let mut energies = Vec::with_capacity(n);
        let mut ls = Vec::with_capacity(n);
        let mut eigenstates = Channels::zeros(n);

        let mut block_index = 0;
        for (block, l) in self.angular_blocks.iter().zip(&self.l) {
            let n_block = block.size();

            let (energies_block, eigenstates_block) = diagonalize(block.0.as_ref());

            energies.extend(energies_block);
            ls.extend(vec![l; n_block]);
            let sub_matrix = eigenstates.0.submatrix_mut(block_index, block_index, n_block, n_block);
            zip!(sub_matrix, eigenstates_block.as_ref()).for_each(|unzip!(s, &e)| *s = e);

            block_index += n_block;
        }

        let levels = Levels {
            l: ls,
            asymptote: energies,
        };

        (levels, eigenstates)
    }

    pub fn channels(&self) -> Channels {
        let n = self.size();
        let mut channels = Channels::zeros(n);

        let mut block_index = 0;
        for block in &self.angular_blocks {
            let n_block = block.size();

            let sub_matrix = channels.0.submatrix_mut(block_index, block_index, n_block, n_block);
            zip!(sub_matrix, block.0.as_ref()).for_each(|unzip!(s, &e)| *s = e);

            block_index += n_block;
        }

        channels
    }
}

pub struct Asymptote {
    levels: Levels,
    transformation: Option<Channels>,

    pub entrance_level: usize,
    pub red_mass: f64,
    pub energy: f64,

    asymptote_channels: Channels,
    centrifugal: MultiCentrifugal,
}

impl Asymptote {
    pub fn new_diagonal(
        mass: Quantity<AuMass>,
        energy: Quantity<AuEnergy>,
        levels: Levels,
        entrance_level: usize,
    ) -> Self {
        let asymptote_channels = Mat::from_fn(levels.asymptote.len(), levels.asymptote.len(), |i, j| {
            if i != j {
                return 0.;
            }

            levels.asymptote[i]
        });

        let centrifugal = MultiCentrifugal::new_diagonal(&levels, mass);

        Self {
            red_mass: mass.value(),
            energy: levels.asymptote[entrance_level] + energy.value(),

            entrance_level,
            levels,

            transformation: None,
            asymptote_channels: Channels(asymptote_channels),
            centrifugal,
        }
    }

    pub fn new_angular_blocks(
        mass: Quantity<AuMass>,
        energy: Quantity<AuEnergy>,
        angular_blocks: AngularBlocks,
        entrance_level: usize,
    ) -> Self {
        let (levels, transformation) = angular_blocks.diagonalize();
        let centrifugal = MultiCentrifugal::new_diagonal(&levels, mass);

        Self {
            red_mass: mass.value(),
            energy: levels.asymptote[entrance_level] + energy.value(),

            entrance_level,
            levels,

            transformation: Some(transformation),
            asymptote_channels: angular_blocks.channels(),
            centrifugal,
        }
    }

    pub fn new_general(
        mass: Quantity<AuMass>,
        energy: Quantity<AuEnergy>,
        levels: Levels,
        transformation: Channels,
        entrance_level: usize,
    ) -> Self {
        let centrifugal = MultiCentrifugal::new_diagonal(&levels, mass);

        let channels = Channels(&transformation.0 * levels.as_channels().0 * transformation.0.transpose());

        Self {
            red_mass: mass.value(),
            energy: levels.asymptote[entrance_level] + energy.value(),

            entrance_level,
            levels,

            transformation: Some(transformation),
            asymptote_channels: channels,
            centrifugal,
        }
    }

    pub fn levels(&self) -> &Levels {
        &self.levels
    }

    pub fn entrance_energy(&self) -> f64 {
        self.levels.asymptote[self.entrance_level]
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
    pub fn new_diagonal(levels: &Levels, mass: Quantity<AuMass>) -> Self {
        let mask = Mat::from_fn(levels.l.len(), levels.l.len(), |i, j| {
            if i != j {
                return 0.;
            }

            (levels.l[i] * (levels.l[i] + 1)) as f64
        });

        Self {
            mask: Channels(mask),
            mass: mass.value(),
        }
    }

    pub fn value_inplace_add(&self, r: f64, channels: &mut Channels) {
        zip!(channels.0.as_mut(), self.mask.0.as_ref()).for_each(|unzip!(o, m)| *o += m / (2. * self.mass * r * r));
    }
}

pub struct RedCoupling<P: VanishingCoupling> {
    pub coupling: P,
    pub asymptote: Asymptote,
    pub id: Channels,
}

impl<P: VanishingCoupling> RedCoupling<P> {
    pub fn new(coupling: P, asymptote: Asymptote) -> Self {
        assert_eq!(
            coupling.size(),
            asymptote.asymptote_channels.size(),
            "mismatched sizes between asymptote and coupling"
        );

        Self {
            id: Channels(Mat::identity(coupling.size(), coupling.size())),
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
