use std::{
    collections::HashMap,
    marker::PhantomData,
};

use coupled_chan::DynInteraction;
use hilbert_space::space::BasisElementsRef;
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::half_integer::HalfU32;

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
        SpinConfiguration,
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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct PecPolarizationScalings(pub HashMap<SpinConfiguration, PecScaling>);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
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

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PecScalingType {
    #[default]
    Full,
    ShortRange,
    LongRange,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PecScaling {
    pub(super) short_range: ScalingValue,
    pub(super) long_range: ScalingValue,
    pub(super) full: ScalingValue,
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
#[serde(deny_unknown_fields)]
pub struct PecPolarizationScaling {
    scaling: f64,
    #[serde(default)]
    scaling_type: PecScalingType,
    #[serde(default)]
    configuration: Option<SpinConfiguration>,
}

pub struct PecPolarizationScalingMod<P: Problem, C> {
    scaling: PecPolarizationScaling,
    id_pec: TypedParamId<PecPolarizations>,
    id_scalings: TypedParamId<PecPolarizationScalings>,
    phantom: PhantomData<(P, C)>,
}

impl<P: Problem, C> PecPolarizationScalingMod<P, C> {
    pub fn new(
        scaling: PecPolarizationScaling,
        id_pec: TypedParamId<PecPolarizations>,
        id_scalings: TypedParamId<PecPolarizationScalings>,
    ) -> Self {
        Self {
            scaling,
            id_pec,
            id_scalings,
            phantom: PhantomData,
        }
    }
}

impl<P: Problem, C: Send + Sync> ModifyParam for PecPolarizationScalingMod<P, C> {
    type P = P;
    type C = C;

    fn prep_modify(&self, _modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        let id_pec = self.id_pec;
        let id_scalings = self.id_scalings;
        let scaling = self.scaling;

        ModificationAction::ParamModify(
            ParamModifications::new(move |r| {
                if let Some(c) = scaling.configuration {
                    r.get_mut(id_scalings)
                        .0
                        .insert(c, PecScaling::new(scaling.scaling, scaling.scaling_type));
                    let s = r.get_mut(id_scalings).0.get_mut(&c).expect("Nonexistent");
                    s.modify(scaling.scaling, scaling.scaling_type);
                } else {
                    for k in r.get(id_pec).0.keys().copied().collect::<Vec<SpinConfiguration>>() {
                        r.get_mut(id_scalings)
                            .0
                            .insert(k, PecScaling::new(scaling.scaling, scaling.scaling_type));
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
