use std::collections::{
    HashMap,
    HashSet,
};
use std::hash::RandomState;

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
use unit_systems::quantities::Scalar;
use unit_systems::quantities::phys_quantities::Length;

use crate::interactions::pec::{
    PecScaling,
    PecScalingType,
};
use crate::interactions::{
    Interactions,
    SpinConfiguration,
    SwitchingRegion,
};

pub struct PesPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator,
{
    pub s_tot: HalfU32,
    pub lambda: u32,
    pub pes: TypedParamId<PesPolarizations>,
    pub scalings: TypedParamId<PesPolarizationScalings>,
    pub masking: Mask,
}

impl<Mask> PotentialSpec for PesPolarizationSpec<Mask>
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
        param_ids![self.pes.vanish(), self.scalings.vanish()]
    }

    fn curve(&self, params: &ParameterRegistry) -> DynInteraction {
        let interaction = params
            .get(self.pes)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .unwrap_or_else(|| panic!("input does not have PES for S_tot = {}", self.s_tot))
            .legendre_component(self.lambda);

        let scaling = params
            .get(self.scalings)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .cloned()
            .unwrap_or_default()
            .legendre_component(self.lambda);

        interaction.scaled_interactions(scaling)
    }
}

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum PesInteractions {
    LegendreComponents(HashMap<u32, Interactions>),
    Transition {
        near: Box<PesInteractions>,
        far: Box<PesInteractions>,
        r_start_switch: Scalar<Length>,
        r_end_switch: Scalar<Length>,
        switching: SwitchingRegion,
    },
}

impl PesInteractions {
    pub fn components(&self) -> Vec<u32> {
        match self {
            PesInteractions::LegendreComponents(hash_map) => hash_map.keys().copied().collect(),
            PesInteractions::Transition {
                near,
                far,
                r_start_switch: _,
                r_end_switch: _,
                switching: _,
            } => {
                let near = near.components();
                let far = far.components();

                HashSet::<u32, RandomState>::from_iter(near.into_iter().chain(far))
                    .into_iter()
                    .collect()
            }
        }
    }

    pub fn legendre_component(&self, lambda: u32) -> Interactions {
        match self {
            PesInteractions::LegendreComponents(hash_map) => hash_map.get(&lambda).unwrap_or(&Interactions::Null).clone(),
            PesInteractions::Transition {
                near,
                far,
                r_start_switch,
                r_end_switch,
                switching,
            } => {
                let near = near.legendre_component(lambda);
                let far = far.legendre_component(lambda);

                Interactions::Transition {
                    near: Box::new(near),
                    far: Box::new(far),
                    r_start_switch: r_start_switch.clone(),
                    r_end_switch: r_end_switch.clone(),
                    switching: *switching,
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub enum LegendreComponents {
    #[default]
    All,
    Isotropic,
    Anisotropic,
    LegendreComponent(u32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
#[serde(default)]
pub struct PesScaling {
    pub scaling: PecScaling,
    pub legendre_components: LegendreComponents,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PesScalings(pub Vec<PesScaling>);

impl PesScalings {
    pub fn legendre_component(&self, lambda: u32) -> PecScaling {
        let mut scaling = PecScaling::default();

        for s in &self.0 {
            if matches!(s.legendre_components, LegendreComponents::All)
                || matches!(s.legendre_components, LegendreComponents::LegendreComponent(x) if x == lambda)
                || (lambda == 0 && matches!(s.legendre_components, LegendreComponents::Isotropic))
                || (lambda != 0 && matches!(s.legendre_components, LegendreComponents::Anisotropic))
            {
                scaling.prod_mut(s.scaling);
            }
        }

        scaling
    }
}

#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PesPolarizations(pub HashMap<SpinConfiguration, PesInteractions>);

#[derive(Clone, Debug, Deserialize, Default, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PesPolarizationScalings(pub HashMap<SpinConfiguration, PesScalings>);

#[derive(Clone, Copy, Debug, Deserialize, schemars::JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct PesPolarizationScaling {
    pub scaling: f64,
    #[serde(default)]
    pub scaling_type: PecScalingType,
    #[serde(default)]
    pub configuration: Option<SpinConfiguration>,
    #[serde(default)]
    pub legendre_components: LegendreComponents,
}

impl PesPolarizationScaling {
    pub fn into_pes_scaling(self) -> PesScaling {
        PesScaling {
            scaling: PecScaling::new(self.scaling, self.scaling_type),
            legendre_components: self.legendre_components,
        }
    }
}
