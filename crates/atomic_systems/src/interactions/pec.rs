use std::collections::HashMap;

use cc_system::hamiltonian_spec::PotentialSpec;
use cc_system::parameters::{
    ParamIds,
    ParameterRegistry,
    TypedParamId,
};
use cc_system::{
    Operator,
    param_ids,
};
use coupled_chan::DynInteraction;
use hilbert_space::space::BasisElementsRef;
use serde::Deserialize;
use spin_algebra::half_integer::HalfU32;

use crate::interactions::{
    Interactions,
    SpinConfiguration,
};

#[derive(Clone, Debug, Deserialize)]
pub struct PecPolarizations(pub HashMap<SpinConfiguration, Interactions>);

pub struct PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator,
{
    pub s_tot: HalfU32,
    pub pecs: TypedParamId<PecPolarizations>,
    pub scalings: TypedParamId<PecPolarizationScalings>,
    pub masking: Mask,
}

impl<Mask> PotentialSpec for PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator + Send + Sync,
{
    fn build_params(&self) -> ParamIds {
        param_ids![]
    }

    fn coupling_masking(&self, elements: BasisElementsRef, _params: &ParameterRegistry) -> Operator {
        (self.masking)(elements)
    }

    fn curve_params(&self) -> ParamIds {
        param_ids![self.pecs.vanish(), self.scalings.vanish()]
    }

    fn curve(&self, params: &ParameterRegistry) -> DynInteraction {
        let interaction = params
            .get(self.pecs)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .unwrap_or_else(|| panic!("input does not have PEC for S_tot = {}", self.s_tot));

        let scaling = params
            .get(self.scalings)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .copied()
            .unwrap_or_default();

        interaction.scaled_interactions(scaling)
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct PecPolarizationScalings(pub HashMap<SpinConfiguration, PecScaling>);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct ScalingValue(pub f64);

impl ScalingValue {
    pub fn one() -> Self {
        Self::default()
    }
}

impl Default for ScalingValue {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Clone, Copy, Default, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PecScalingType {
    #[default]
    Full,
    ShortRange,
    LongRange,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct PecScaling {
    pub short_range: ScalingValue,
    pub long_range: ScalingValue,
    pub full: ScalingValue,
}

impl PecScaling {
    pub fn new(value: f64, scaling_type: PecScalingType) -> Self {
        match scaling_type {
            PecScalingType::Full => Self {
                full: ScalingValue(value),
                ..Default::default()
            },
            PecScalingType::ShortRange => Self {
                short_range: ScalingValue(value),
                ..Default::default()
            },
            PecScalingType::LongRange => Self {
                long_range: ScalingValue(value),
                ..Default::default()
            },
        }
    }

    pub fn modify(&mut self, value: f64, scaling_type: PecScalingType) {
        match scaling_type {
            PecScalingType::Full => self.full.0 = value,
            PecScalingType::ShortRange => self.short_range.0 = value,
            PecScalingType::LongRange => self.long_range.0 = value,
        }
    }

    pub fn prod_mut(&mut self, other: Self) {
        self.full.0 *= other.full.0;
        self.short_range.0 *= other.short_range.0;
        self.long_range.0 *= other.long_range.0;
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct PecPolarizationScaling {
    pub scaling: f64,
    #[serde(default)]
    pub scaling_type: PecScalingType,
    #[serde(default)]
    pub configuration: Option<SpinConfiguration>,
}
