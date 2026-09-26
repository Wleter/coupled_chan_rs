use std::{
    collections::{
        HashMap,
        HashSet,
    },
    hash::RandomState,
    marker::PhantomData,
};

use coupled_chan::DynInteraction;
use hilbert_space::space::BasisElementsRef;
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::half_integer::HalfU32;
use unit_systems::quantities::{
    Scalar,
    phys_quantities::Length,
};

use crate::{
    Operator,
    calculations::{
        Modified,
        modifications::{
            ModificationAction,
            ModifyParam,
        },
    },
    interactions::{
        Interactions,
        PecScaling,
        PecScalingType,
        SpinConfiguration,
        SwitchingRegion,
    },
    param_ids,
    parameters::{
        ParameterRegistry,
        TypedParamId,
    },
    problems::Problem,
    system::{
        ParamIds,
        ParamModifications,
        PotentialSpec,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum LegendreComponents {
    #[default]
    All,
    Isotropic,
    Anisotropic,
    LegendreComponent(u32),
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PesScaling {
    scaling: PecScaling,
    legendre_components: LegendreComponents,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PesScalings(Vec<PesScaling>);

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PesPolarizations(pub HashMap<SpinConfiguration, PesInteractions>);

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PesPolarizationScalings(pub HashMap<SpinConfiguration, PesScalings>);

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PesPolarizationScaling {
    scaling: f64,
    #[serde(default)]
    scaling_type: PecScalingType,
    #[serde(default)]
    configuration: Option<SpinConfiguration>,
    #[serde(default)]
    legendre_components: LegendreComponents,
}

impl PesPolarizationScaling {
    pub fn into_pes_scaling(self) -> PesScaling {
        PesScaling {
            scaling: PecScaling::new(self.scaling, self.scaling_type),
            legendre_components: self.legendre_components,
        }
    }
}

pub struct PesPolarizationScalingMod<P: Problem, C> {
    scaling: PesPolarizationScaling,
    id_pec: TypedParamId<PesPolarizations>,
    id_scalings: TypedParamId<PesPolarizationScalings>,
    phantom: PhantomData<(P, C)>,
}

impl<P: Problem, C> PesPolarizationScalingMod<P, C> {
    pub fn new(
        scaling: PesPolarizationScaling,
        id_pec: TypedParamId<PesPolarizations>,
        id_scalings: TypedParamId<PesPolarizationScalings>,
    ) -> Self {
        Self {
            scaling,
            id_pec,
            id_scalings,
            phantom: PhantomData,
        }
    }
}

impl<P: Problem, C: Send + Sync> ModifyParam for PesPolarizationScalingMod<P, C> {
    type P = P;
    type C = C;

    fn prep_modify(&self, _modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        let id_pec = self.id_pec;
        let id_scalings = self.id_scalings;
        let scaling = self.scaling;

        ModificationAction::ParamModify(
            ParamModifications::new(move |r| {
                if let Some(c) = scaling.configuration {
                    let scalings = r.get_mut(id_scalings).0.get_mut(&c);
                    if let Some(scalings) = scalings {
                        if let Some(s) = scalings
                            .0
                            .iter_mut()
                            .find(|x| x.legendre_components == scaling.legendre_components)
                        {
                            *s = scaling.into_pes_scaling();
                        } else {
                            scalings.0.push(scaling.into_pes_scaling())
                        }
                    } else {
                        r.get_mut(id_scalings)
                            .0
                            .insert(c, PesScalings(vec![scaling.into_pes_scaling()]));
                    }
                } else {
                    for c in r.get(id_pec).0.keys().copied().collect::<Vec<SpinConfiguration>>() {
                        let scalings = r.get_mut(id_scalings).0.get_mut(&c);
                        if let Some(scalings) = scalings {
                            if let Some(s) = scalings
                                .0
                                .iter_mut()
                                .find(|x| x.legendre_components == scaling.legendre_components)
                            {
                                *s = scaling.into_pes_scaling();
                            } else {
                                scalings.0.push(scaling.into_pes_scaling())
                            }
                        } else {
                            r.get_mut(id_scalings)
                                .0
                                .insert(c, PesScalings(vec![scaling.into_pes_scaling()]));
                        }
                    }
                }

                param_ids![id_scalings.vanish()]
            })
            .into_dyn(),
        )
    }

    fn as_number(&self) -> serde_json::Number {
        serde_json::Number::from_f64(self.scaling.scaling).unwrap()
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        self.scaling.scaling = number.as_f64().unwrap()
    }
}

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
