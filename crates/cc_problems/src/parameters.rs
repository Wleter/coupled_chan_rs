use std::{
    any::Any,
    marker::PhantomData,
    sync::Arc,
};

pub use cc_derive::Parameters;

use coupled_chan::{
    DynInteraction,
    Interaction,
};
use hilbert_space::space::BasisElementsRef;

use crate::hamiltonian::Operator;

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

    pub fn get<T: 'static>(&self, id: TypedParamId<T>) -> &T {
        self.0[id.0]
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("Could not get TypedParamId from ParameterRegistry: downcast error"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedParamId<T>(pub usize, PhantomData<T>);

impl<T> TypedParamId<T> {
    pub fn new(id: usize) -> Self {
        Self(id, PhantomData)
    }

    pub fn vanish(&self) -> ParamId {
        ParamId(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParamId(pub usize);

pub trait OperatorBuilder<P: Parameters> {
    fn build_matrix(&self, elements: BasisElementsRef, params: &P) -> Operator;
    fn coupling(&self, params: &P) -> f64;

    fn multiply_params(&self) -> Vec<ParamId>;
    fn build_params(&self) -> Vec<ParamId>;
}

pub struct DynOperatorBuilder<P: Parameters>(pub Arc<dyn OperatorBuilder<P>>);

impl<P: Parameters> DynOperatorBuilder<P> {
    pub fn new(builder: impl OperatorBuilder<P> + 'static) -> Self {
        Self(Arc::new(builder))
    }
}

pub struct PotentialBuilder<P: Parameters> {
    pub operator_builder: DynOperatorBuilder<P>,
    pub potential_curve: DynInteraction,
}

impl<P: Parameters> PotentialBuilder<P> {
    pub fn new<O, I>(operator: O, interaction: I) -> Self
    where
        O: OperatorBuilder<P> + 'static,
        I: Interaction + 'static + Sync + Send,
    {
        Self {
            operator_builder: DynOperatorBuilder::new(operator),
            potential_curve: DynInteraction::new(interaction),
        }
    }
}

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
