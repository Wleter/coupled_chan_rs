use std::{
    any::Any,
    marker::PhantomData,
};

pub use cc_derive::Parameters;

pub trait Parameters {
    type Ids;

    const PARAM_COUNT: usize;

    fn ids() -> Self::Ids {
        Self::ids_at(0)
    }

    #[doc(hidden)]
    fn ids_at(offset: usize) -> Self::Ids;
    fn registry(&self) -> ParameterRegistry;
}

#[derive(Default)]
pub struct ParameterRegistry(Vec<Box<dyn Any>>);

impl ParameterRegistry {
    pub fn push<T: 'static>(&mut self, value: T) {
        self.0.push(Box::new(value));
    }

    pub fn extend(&mut self, other: ParameterRegistry) {
        self.0.extend(other.0);
    }

    pub fn modify<T: 'static + PartialEq>(&mut self, id: TypedParamId<T>, value: T) -> bool {
        let value_old = self.get(id);
        if value_old != &value {
            self.0[id.0] = Box::new(value);
            true
        } else {
            false
        }
    }

    pub fn get<T: 'static>(&self, id: TypedParamId<T>) -> &T {
        self.0[id.0]
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("Could not get TypedParamId from ParameterRegistry: downcast error"))
    }
}

pub struct TypedParamId<T>(pub usize, PhantomData<T>);

impl<T> Eq for TypedParamId<T> {}
impl<T> PartialEq for TypedParamId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for TypedParamId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("TypedParamId").field(&self.0).finish()
    }
}
impl<T> Copy for TypedParamId<T> {}
impl<T> Clone for TypedParamId<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<T> TypedParamId<T> {
    pub fn new(id: usize) -> Self {
        Self(id, PhantomData)
    }

    pub fn vanish(&self) -> ParamId {
        ParamId(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamId(pub usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Parameters, Clone)]
    struct AtomParams {
        name: String,
        g_e: f64,
    }

    #[derive(Parameters, Clone)]
    struct Params {
        b_field: f64,
        #[parameter(nested)]
        atom_a: AtomParams,
        e_field: Option<f64>,
        #[parameter(nested)]
        atom_b: AtomParams,
    }

    #[test]
    pub fn test_parameters_derive() {
        let params = Params {
            b_field: 100.,
            atom_a: AtomParams {
                name: "Li".into(),
                g_e: -2.0,
            },
            e_field: None,
            atom_b: AtomParams {
                name: "Na".into(),
                g_e: -2.002,
            },
        };

        let ids = Params::ids();

        assert_eq!(ids.b_field, TypedParamId::new(0));

        assert_eq!(ids.atom_a.name, TypedParamId::new(1));
        assert_eq!(ids.atom_a.g_e, TypedParamId::new(2));

        assert_eq!(ids.e_field, TypedParamId::new(3));

        assert_eq!(ids.atom_b.name, TypedParamId::new(4));
        assert_eq!(ids.atom_b.g_e, TypedParamId::new(5));

        let registry = params.registry();

        assert_eq!(*registry.get(ids.b_field), 100.0);
        assert_eq!(*registry.get(ids.e_field), None);
        assert_eq!(registry.get(ids.atom_a.name), "Li");
        assert_eq!(registry.get(ids.atom_b.name), "Na");
        assert_eq!(*registry.get(ids.atom_a.g_e), -2.0);
        assert_eq!(*registry.get(ids.atom_b.g_e), -2.002);
    }
}
