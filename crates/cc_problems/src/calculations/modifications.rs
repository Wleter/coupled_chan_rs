use std::{
    collections::HashMap,
    marker::PhantomData,
};

use serde_json::{
    Number,
    Value,
};
use unit_systems::quantities::{
    PhysQuantity,
    Scalar,
};

use crate::{
    OrbitalRecipe,
    UNITS_CONVERTER,
    calculations::Modified,
    parameters::TypedParamId,
    problems::Problem,
    system::{
        DynParamModifications,
        System,
        new_param_modifications,
    },
};

pub struct ModifyRegistry<P: Problem, C>(pub HashMap<Box<str>, ModifyParamRecipe<P, C>>);

impl<P: Problem, C> Default for ModifyRegistry<P, C> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<P: Problem, C> ModifyRegistry<P, C> {
    pub fn from<const N: usize>(registry: [(impl AsRef<str>, ModifyParamRecipe<P, C>); N]) -> Self {
        ModifyRegistry(HashMap::from(registry.map(|x| (x.0.as_ref().into(), x.1))))
    }

    pub fn extend(mut self, other: Self) -> Self {
        self.0.extend(other.0);
        self
    }
}

pub struct ModifyParamRecipe<P: Problem, C> {
    pub recipe: Box<dyn Fn(Value) -> Box<dyn ModifyParam<P = P, C = C>> + Send + Sync>,
}

impl<P: Problem, C> ModifyParamRecipe<P, C> {
    pub fn new<M: ModifyParam<P = P, C = C> + 'static>(recipe: impl Fn(Value) -> M + Send + Sync + 'static) -> Self {
        Self {
            recipe: Box::new(move |v| Box::new(recipe(v))),
        }
    }
}

#[macro_export]
macro_rules! modify_recipe {
    (|$value:ident| $body:expr) => {
        $crate::calculations::modifications::ModifyParamRecipe::new(move |v| {
            let $value = serde_json::from_value(v).expect("Could not convert modifying value");
            $body
        })
    };
}

pub trait ModifyParam: Send + Sync {
    type P: Problem;
    type C;

    fn prep_modify(&self, modified: &mut Modified<Self::P, Self::C>) -> ModificationAction;
    fn modify(&self, modified: &mut Modified<Self::P, Self::C>) {
        match self.prep_modify(modified) {
            ModificationAction::BasisChange => {
                let spec = Self::P::build(modified.basis, modified.params);
                *modified.system = System::new(spec, modified.system.param_registry().to_owned());
            }
            ModificationAction::ParamModify(modify) => modified.system.modify_params(modify),
            _ => {}
        }
    }

    fn as_number(&self) -> Number;
    fn mut_number(&mut self, number: Number);
}

pub enum ModificationAction {
    BasisChange,
    ParamModify(DynParamModifications),
    CalcChange,
}

pub struct ScalarParamMod<Q: PhysQuantity, P: Problem, C> {
    value: Scalar<Q>,
    id: TypedParamId<Scalar<Q>>,
    phantom: PhantomData<(P, C)>,
}

impl<Q: PhysQuantity, P: Problem, C> ScalarParamMod<Q, P, C> {
    pub fn new(value: Scalar<Q>, id: TypedParamId<Scalar<Q>>) -> Self {
        Self {
            value,
            id,
            phantom: PhantomData,
        }
    }
}

impl<Q: PhysQuantity + Send + Sync, P: Problem, C: Send + Sync> ModifyParam for ScalarParamMod<Q, P, C> {
    type P = P;
    type C = C;

    fn prep_modify(&self, _modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        ModificationAction::ParamModify(new_param_modifications(self.id, self.value.clone()).into_dyn())
    }

    fn as_number(&self) -> Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        Number::from_f64(converter.scalar_value(&self.value)).unwrap()
    }

    fn mut_number(&mut self, number: Number) {
        let value = number.as_f64().unwrap();
        self.value = scalar_from_f64(value);
    }
}

pub struct ScalarCalcMod<Q: PhysQuantity, P: Problem, C, F: Fn(&mut C, Scalar<Q>)> {
    value: Scalar<Q>,
    conversion: F,
    phantom: PhantomData<(P, C)>,
}

impl<Q: PhysQuantity, P: Problem, C, F: Fn(&mut C, Scalar<Q>)> ScalarCalcMod<Q, P, C, F> {
    pub fn new(value: Scalar<Q>, conversion: F) -> Self {
        Self {
            value,
            conversion,
            phantom: PhantomData,
        }
    }
}

impl<Q, P, C, F> ModifyParam for ScalarCalcMod<Q, P, C, F>
where
    Q: PhysQuantity + Send + Sync,
    P: Problem,
    C: Send + Sync,
    F: Fn(&mut C, Scalar<Q>) + Send + Sync,
{
    type P = P;
    type C = C;

    fn prep_modify(&self, modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        (self.conversion)(modified.calc_input, self.value.clone());

        ModificationAction::CalcChange
    }

    fn as_number(&self) -> Number {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        Number::from_f64(converter.scalar_value(&self.value)).unwrap()
    }

    fn mut_number(&mut self, number: Number) {
        let value = number.as_f64().unwrap();
        self.value = scalar_from_f64(value);
    }
}

pub struct OrbitalRecipeMod<P: Problem, C, F: Fn(&mut P::BasisRecipe) -> &mut OrbitalRecipe> {
    value: OrbitalRecipe,
    conversion: F,
    phantom: PhantomData<(P, C)>,
}

impl<P: Problem, C, F: Fn(&mut P::BasisRecipe) -> &mut OrbitalRecipe> OrbitalRecipeMod<P, C, F> {
    pub fn new(value: OrbitalRecipe, conversion: F) -> Self {
        Self {
            value,
            conversion,
            phantom: PhantomData,
        }
    }
}

impl<P, C, F> ModifyParam for OrbitalRecipeMod<P, C, F>
where
    P: Problem,
    C: Send + Sync,
    F: Fn(&mut P::BasisRecipe) -> &mut OrbitalRecipe + Send + Sync,
{
    type P = P;
    type C = C;

    fn prep_modify(&self, modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        *(self.conversion)(modified.basis) = self.value;
        ModificationAction::BasisChange
    }

    fn as_number(&self) -> Number {
        match self.value {
            OrbitalRecipe::Single(l) => Number::from_u128(l as u128).unwrap(),
            OrbitalRecipe::LMax(l) => Number::from_u128(l as u128).unwrap(),
            OrbitalRecipe::LMaxProjections(l) => Number::from_u128(l as u128).unwrap(),
        }
    }

    fn mut_number(&mut self, number: Number) {
        match &mut self.value {
            OrbitalRecipe::Single(l) => *l = number.as_u64().unwrap() as u32,
            OrbitalRecipe::LMax(l) => *l = number.as_u64().unwrap() as u32,
            OrbitalRecipe::LMaxProjections(l) => *l = number.as_u64().unwrap() as u32,
        }
    }
}

pub fn scalar_from_f64<Q: PhysQuantity>(value: f64) -> Scalar<Q> {
    if value == 0.0 {
        Scalar::new(0.0, Q::default(), "");
    }

    let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
    // first unit encountered
    let unit = &converter.registry.get::<Q>()[0];
    let value_in_unit = value / Q::to_unit_system_logic(unit.name, &converter.registry, &converter.target_unit_system);

    Scalar::new(value_in_unit, Q::default(), unit.name)
}
