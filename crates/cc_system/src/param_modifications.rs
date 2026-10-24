use cc_qol_utils::params::CloneAny;

use crate::{
    param_ids,
    parameters::{
        ParamIds,
        ParameterRegistry,
        TypedParamId,
    },
};

pub type DynParamModifications = ParamModifications<Box<dyn FnOnce(&mut ParameterRegistry) -> ParamIds>>;

#[derive(Clone, Debug)]
pub struct ParamModifications<F: FnOnce(&mut ParameterRegistry) -> ParamIds> {
    pub modification: F,
}

pub fn new_param_modifications<T: PartialEq + CloneAny>(
    id: TypedParamId<T>,
    value: T,
) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> {
    let modification = move |registry: &mut ParameterRegistry| {
        if registry.modify(id, value) {
            param_ids![id.vanish()]
        } else {
            param_ids![]
        }
    };

    ParamModifications { modification }
}

impl<F: FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> ParamModifications<F> {
    pub fn new(modification: F) -> Self {
        Self { modification }
    }

    pub fn modify<T: PartialEq + CloneAny>(
        self,
        id: TypedParamId<T>,
        value: T,
    ) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> {
        let modification = move |registry: &mut ParameterRegistry| {
            if registry.modify(id, value) {
                let mut ids = (self.modification)(registry);

                ids.push(id.vanish());
                ids
            } else {
                (self.modification)(registry)
            }
        };

        ParamModifications { modification }
    }

    pub fn into_dyn(self) -> DynParamModifications {
        DynParamModifications {
            modification: Box::new(self.modification),
        }
    }
}
