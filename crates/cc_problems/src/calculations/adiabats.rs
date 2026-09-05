use std::marker::PhantomData;

use coupled_chan::{coupling::{Asymptote, CollisionParams, CollisionWMatrix}, multi_channel::WMatrix};
use serde::Deserialize;
use unit_systems::quantities::{Scalar, phys_quantities::Length};

use crate::{
    UNITS_CONVERTER, calculations::{
        Modified, SingleCalc, dependence::DependenceCalc, levels::EnergyLevelsData
    }, problems::Problem
};

pub struct AdiabatsCalc<P>(PhantomData<P>);

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AdiabatsInput {
    pub distance: Scalar<Length>
}

impl<P> Default for AdiabatsCalc<P> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<P: Problem> SingleCalc for AdiabatsCalc<P> {
    type P = P;
    type CalcInput = AdiabatsInput;
    type Data = EnergyLevelsData;

    fn calculate(
        &self,
        modified: Modified<P, AdiabatsInput>,
        _problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        let blocks = modified.system.angular_blocks();
        let collision_params = CollisionParams {
            mass: 0.5,
            energy: 0.0,
            entrance: 0,
        };
        let mut asymptote = Asymptote::new_angular_blocks(blocks, collision_params);
        asymptote.energy = 0.0;

        let w_matrix = CollisionWMatrix::new(modified.system.coupling(), asymptote);

        let mut blocks = w_matrix.id().clone();
        w_matrix.value_inplace(converter.scalar_value(&modified.calc_input.distance), &mut blocks);
        let values = (-blocks).self_adjoint_eigenvalues(hilbert_space::faer::Side::Lower)
            .expect("Could not diagonalize adiabats");

        [Ok(EnergyLevelsData(values))]
    }
}

pub type AdiabatsScan<P, D> = DependenceCalc<AdiabatsCalc<P>, D>;
