pub mod composite;
pub mod diagonal;
pub mod masked;
pub mod pair;

use std::{
    iter::Sum,
    ops::{
        Add,
        AddAssign,
    },
    sync::Arc,
};

use cc_matrix_utils::faer::diagonalize;
use cc_propagator::multi_channel::{
    Matrix,
    WMatrix,
};
use faer::{
    Mat,
    unzip,
    zip,
};
use single_chan::interaction::AsymptoteDep;

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemParams {
    pub mass: f64,
    pub energy: f64,
    pub entrance: usize,
}

impl SystemParams {
    pub fn new(mass: f64, energy: f64, entrance: usize) -> Self {
        Self { mass, energy, entrance }
    }
}

pub trait RCoupling {
    fn value_inplace_add(&self, r: f64, channels: &mut Matrix);
    fn size(&self) -> usize;
    fn asymptote_dep(&self) -> AsymptoteDep;

    fn value_inplace(&self, r: f64, channels: &mut Matrix) {
        channels.fill(0.);
        self.value_inplace_add(r, channels);
    }

    fn value(&self, r: f64) -> Matrix {
        let mut operator = Matrix::zeros(self.size(), self.size());
        self.value_inplace(r, &mut operator);

        operator
    }

    fn adiabats(&self, r: f64) -> Vec<f64> {
        let mut operator = Matrix::zeros(self.size(), self.size());
        self.value_inplace(r, &mut operator);

        operator.self_adjoint_eigenvalues(faer::Side::Lower).unwrap()
    }
}

#[derive(Clone)]
pub struct DynRCoupling(Arc<dyn RCoupling + Send + Sync>);

impl DynRCoupling {
    pub fn new<T: RCoupling + Send + Sync + 'static>(coupling: T) -> Self {
        Self(Arc::new(coupling))
    }
}

impl RCoupling for DynRCoupling {
    fn value_inplace(&self, r: f64, channels: &mut Matrix) {
        self.0.value_inplace(r, channels);
    }

    fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
        self.0.value_inplace_add(r, channels);
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn asymptote_dep(&self) -> AsymptoteDep {
        self.0.asymptote_dep()
    }
}

#[derive(Clone, Debug)]
pub struct Levels {
    pub l: Vec<u32>,
    pub asymptote: Vec<f64>,
}

impl Levels {
    pub fn as_matrix(&self) -> Matrix {
        let mut channels = Matrix::zeros(self.l.len(), self.l.len());

        for (c, &a) in channels.diagonal_mut().column_vector_mut().iter_mut().zip(&self.asymptote) {
            *c = a
        }

        channels
    }
}

#[derive(Clone, Debug)]
pub struct AngularBlocks {
    pub l: Vec<u32>,
    pub blocks: Vec<Matrix>,
}

impl AngularBlocks {
    pub fn size(&self) -> usize {
        self.blocks.iter().map(|b| b.nrows()).sum()
    }

    pub fn scale(&self, scaling: f64) -> Self {
        AngularBlocks {
            l: self.l.clone(),
            blocks: self.blocks.iter().map(|x| scaling * x).collect(),
        }
    }

    pub fn transform(&self, transform: &Self) -> Self {
        assert_eq!(self.l, transform.l);

        Self {
            l: self.l.clone(),
            blocks: self
                .blocks
                .iter()
                .zip(&transform.blocks)
                .map(|(x, t)| crate::transform(x, t))
                .collect(),
        }
    }

    pub fn diagonalized(&self) -> (Levels, Matrix) {
        let n = self.size();
        let mut energies = Vec::with_capacity(n);
        let mut ls = Vec::with_capacity(n);
        let mut eigenstates = Matrix::zeros(n, n);

        let mut block_index = 0;
        for (block, l) in self.blocks.iter().zip(&self.l) {
            let n_block = block.nrows();

            let (energies_block, eigenstates_block) = diagonalize(block.as_ref());

            energies.extend(energies_block);
            ls.extend(vec![l; n_block]);
            let sub_matrix = eigenstates.submatrix_mut(block_index, block_index, n_block, n_block);
            zip!(sub_matrix, eigenstates_block.as_ref()).for_each(|unzip!(s, &e)| *s = e);

            block_index += n_block;
        }

        let levels = Levels {
            l: ls,
            asymptote: energies,
        };

        (levels, eigenstates)
    }

    pub fn as_matrix(&self) -> Matrix {
        let n = self.size();
        let mut channels = Matrix::zeros(n, n);

        let mut block_index = 0;
        for block in &self.blocks {
            let n_block = block.nrows();

            let sub_matrix = channels.submatrix_mut(block_index, block_index, n_block, n_block);
            zip!(sub_matrix, block.as_ref()).for_each(|unzip!(s, &e)| *s = e);

            block_index += n_block;
        }

        channels
    }
}

impl Add for AngularBlocks {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.l.iter().zip(&rhs.l).all(|(a, b)| a == b));

        Self {
            l: self.l,
            blocks: self.blocks.into_iter().zip(rhs.blocks).map(|(x, y)| x + y).collect(),
        }
    }
}

impl AddAssign for AngularBlocks {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.l.iter().zip(&rhs.l).all(|(a, b)| a == b));

        for (s, r) in self.blocks.iter_mut().zip(&rhs.blocks) {
            *s += r;
        }
    }
}

impl Sum for AngularBlocks {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut first = iter.next().expect("zero element sum");

        for el in iter {
            first += el;
        }

        first
    }
}

#[derive(Clone, Debug)]
pub struct Asymptote {
    levels: Levels,
    transformation: Option<Matrix>,

    system_params: SystemParams,
    pub energy: f64,

    asymptote_channels: Matrix,
    centrifugal: RedMultiCentrifugal,
}

impl Asymptote {
    pub fn new_diagonal(levels: Levels, system_params: SystemParams) -> Self {
        let asymptote_channels = Mat::from_fn(levels.asymptote.len(), levels.asymptote.len(), |i, j| {
            if i != j {
                return 0.;
            }

            levels.asymptote[i]
        });
        let centrifugal = RedMultiCentrifugal::new_diagonal(&levels);

        Self {
            system_params,
            energy: levels.asymptote[system_params.entrance] + system_params.energy,
            levels,

            transformation: None,
            asymptote_channels,
            centrifugal,
        }
    }

    pub fn new_angular_blocks(angular_blocks: AngularBlocks, system_params: SystemParams) -> Self {
        let (levels, transformation) = angular_blocks.diagonalized();
        let centrifugal = RedMultiCentrifugal::new_diagonal(&levels);

        Self {
            system_params,
            energy: levels.asymptote[system_params.entrance] + system_params.energy,
            levels,
            centrifugal,

            transformation: Some(transformation),
            asymptote_channels: angular_blocks.as_matrix(),
        }
    }

    pub fn new_general(levels: Levels, transformation: Matrix, system_params: SystemParams) -> Self {
        let channels = crate::transform(&levels.as_matrix(), &transformation);
        let centrifugal = RedMultiCentrifugal::new_general(&levels, &transformation);

        Self {
            system_params,
            energy: levels.asymptote[system_params.entrance] + system_params.energy,
            levels,
            centrifugal,

            transformation: Some(transformation),
            asymptote_channels: channels,
        }
    }

    pub fn levels(&self) -> &Levels {
        &self.levels
    }

    pub fn set_energy(&mut self, energy: f64) {
        self.system_params.energy = energy;
        self.energy = self.levels.asymptote[self.system_params.entrance] + energy
    }

    pub fn set_entrance(&mut self, entrance: usize) {
        self.system_params.entrance = entrance;
        self.energy = self.levels.asymptote[self.system_params.entrance] + self.system_params.energy
    }

    pub fn set_mass(&mut self, mass: f64) {
        self.system_params.mass = mass
    }

    pub fn system_params(&self) -> &SystemParams {
        &self.system_params
    }

    pub fn entrance_energy(&self) -> f64 {
        self.levels.asymptote[self.system_params.entrance]
    }

    pub fn transformation(&self) -> &Option<Matrix> {
        &self.transformation
    }
}

/// Multichannel centrifugal term L^2 / (2 m r^2)
#[derive(Clone, Debug)]
pub struct RedMultiCentrifugal {
    mask: Matrix,
}

impl RedMultiCentrifugal {
    pub fn new_diagonal(levels: &Levels) -> Self {
        let mask = Mat::from_fn(levels.l.len(), levels.l.len(), |i, j| {
            if i != j {
                return 0.;
            }

            (levels.l[i] * (levels.l[i] + 1)) as f64
        });

        Self { mask }
    }

    pub fn new_general(levels: &Levels, transformation: &Matrix) -> Self {
        let mask = Mat::from_fn(levels.l.len(), levels.l.len(), |i, j| {
            if i != j {
                return 0.;
            }

            (levels.l[i] * (levels.l[i] + 1)) as f64
        });
        let mask = crate::transform(&mask, transformation);

        Self { mask }
    }

    pub fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
        zip!(channels.as_mut(), self.mask.as_ref()).for_each(|unzip!(o, m)| *o += m / (r * r));
    }
}

#[derive(Clone)]
pub struct CollisionWMatrix<P: RCoupling> {
    pub coupling: P,
    pub asymptote: Asymptote,
    id: Matrix,
}

impl<P: RCoupling> CollisionWMatrix<P> {
    pub fn new(coupling: P, asymptote: Asymptote) -> Self {
        assert_eq!(
            coupling.size(),
            asymptote.asymptote_channels.nrows(),
            "mismatched sizes between asymptote and coupling"
        );

        Self {
            id: Matrix::identity(coupling.size(), coupling.size()),
            coupling,
            asymptote,
        }
    }

    pub fn id(&self) -> &Matrix {
        &self.id
    }

    pub fn asymptote(&self) -> &Asymptote {
        &self.asymptote
    }
}

impl<V: RCoupling> WMatrix<f64> for CollisionWMatrix<V> {
    fn value_inplace(&self, r: f64, value: &mut Matrix) {
        self.coupling.value_inplace(r, value);
        *value += &self.asymptote.asymptote_channels;
        zip!(value.as_mut(), self.id.as_ref())
            .for_each(|unzip!(c, i)| *c = 2.0 * self.asymptote.system_params.mass * (self.asymptote.energy * i - *c));

        self.asymptote.centrifugal.value_inplace_add(r, value);
    }

    fn size(&self) -> usize {
        self.coupling.size()
    }
}
