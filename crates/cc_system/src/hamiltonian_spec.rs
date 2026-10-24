use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use coupled_chan::DynInteraction;
use hilbert_space::space::BasisElementsRef;

use crate::Operator;
use crate::orbital_basis::OrbitalBasisElements;
use crate::parameters::{
    ParamIds,
    ParameterRegistry,
};

pub struct HamiltonianSpec {
    pub basis: OrbitalBasisElements,
    pub operators: HashMap<Box<str>, DynOperatorSpec>,
    pub potentials: HashMap<Box<str>, DynPotentialSpec>,
}

impl HamiltonianSpec {
    pub fn new(basis: OrbitalBasisElements) -> Self {
        Self {
            basis,
            operators: Default::default(),
            potentials: Default::default(),
        }
    }

    pub fn add_operators<S: AsRef<str>>(&mut self, operators: impl IntoIterator<Item = (S, DynOperatorSpec)>) {
        self.operators
            .extend(operators.into_iter().map(|x| (x.0.as_ref().into(), x.1)));
    }

    pub fn add_potentials<S: AsRef<str>>(&mut self, potentials: impl IntoIterator<Item = (S, DynPotentialSpec)>) {
        self.potentials
            .extend(potentials.into_iter().map(|x| (x.0.as_ref().into(), x.1)));
    }
}

pub trait OperatorSpec: Send + Sync {
    fn build_params(&self) -> ParamIds;
    fn matrix(&self, elements: BasisElementsRef, params: &ParameterRegistry) -> Operator;

    fn coupling_params(&self) -> ParamIds;
    fn coupling(&self, params: &ParameterRegistry) -> f64;
}

#[derive(Clone)]
pub struct DynOperatorSpec(pub Arc<dyn OperatorSpec>);

impl Deref for DynOperatorSpec {
    type Target = dyn OperatorSpec;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DynOperatorSpec {
    pub fn new(spec: impl OperatorSpec + 'static) -> Self {
        Self(Arc::new(spec))
    }
}

pub trait PotentialSpec: Send + Sync {
    fn build_params(&self) -> ParamIds;
    fn coupling_masking(&self, elements: BasisElementsRef, params: &ParameterRegistry) -> Operator;

    fn curve_params(&self) -> ParamIds;
    fn curve(&self, params: &ParameterRegistry) -> DynInteraction;
}

#[derive(Clone)]
pub struct DynPotentialSpec(pub Arc<dyn PotentialSpec>);

impl Deref for DynPotentialSpec {
    type Target = dyn PotentialSpec;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DynPotentialSpec {
    pub fn new(spec: impl PotentialSpec + 'static) -> Self {
        Self(Arc::new(spec))
    }
}
