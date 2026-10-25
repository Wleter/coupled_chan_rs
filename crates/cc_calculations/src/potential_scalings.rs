use std::marker::PhantomData;

use atomic_systems::interactions::SpinConfiguration;
use atomic_systems::interactions::pec::{
    PecPolarizationScaling,
    PecPolarizationScalings,
    PecPolarizations,
    PecScaling,
};
use atomic_systems::interactions::pes::{
    PesPolarizationScaling,
    PesPolarizationScalings,
    PesPolarizations,
    PesScalings,
};
use cc_system::param_ids;
use cc_system::param_modifications::ParamModifications;
use cc_system::parameters::TypedParamId;

use crate::calculations::Modified;
use crate::calculations::modifications::{
    ModificationAction,
    ModifyParam,
};
use crate::problems::Problem;

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
