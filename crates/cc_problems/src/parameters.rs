use std::{marker::PhantomData};

use hilbert_space::space::BasisElementsRef;
use serde::{Serialize, de::DeserializeOwned};

use crate::hamiltonian::Operator;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
/// Metadata of one of the [`Parameters`] field.
pub struct ParametersField {
    pub path: String,
    pub rust_type: &'static str,
}

pub trait Parameters: Serialize + DeserializeOwned {
    fn schema() -> Vec<ParametersField>;

    fn get_param<T>(id: TypedParamId<T>) -> T;

    fn contains(path: &str) -> bool {
        Self::schema().iter().any(|descriptor| descriptor.path == path)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TypedParamId<T>(pub usize, PhantomData<T>);

impl<T> TypedParamId<T> {
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

pub struct System<R, P, F> 
where
    P: Parameters,
    F: Fn(&R, &P) -> Vec<Box<dyn OperatorBuilder<P>>>
{
    operators: F,
    phantom: PhantomData<(R, P)>,
}

