use std::marker::PhantomData;

use coupled_chan::coupling::RCoupling;
use serde::{Deserialize, Serialize};
use unit_systems::quantities::{Scalar, phys_quantities::Length};

use crate::{
    UNITS_CONVERTER, calculations::{
        Modified, SingleCalc, dependence::DependenceCalc, levels::EnergyLevelsData
    }, problems::Problem
};

pub struct AdiabatsCalc<P>(PhantomData<P>);

#[derive(Clone, Deserialize)]
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

        let mut blocks = modified.system.angular_blocks().as_matrix();
        let interaction = modified.system.coupling();
        interaction.value_inplace_add(converter.scalar_value(&modified.calc_input.distance), &mut blocks);
        let values = blocks.self_adjoint_eigenvalues(hilbert_space::faer::Side::Lower)
            .expect("Could not diagonalize adiabats");

        [Ok(EnergyLevelsData(values))]
    }
}

pub type AdiabatsScan<P, D> = DependenceCalc<AdiabatsCalc<P>, D>;
