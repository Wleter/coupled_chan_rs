use std::{
    collections::HashMap,
    marker::PhantomData,
};

use coupled_chan::Interaction;
use serde::{Deserialize, Serialize};
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        Length,
    },
};

use crate::{
    UNITS_CONVERTER, calculations::{
        DynCalc, Modified, SingleCalc, dependence::DependenceCalc, modifications::{
            ModifyRegistry,
            ScalarCalcMod,
        }
    }, modify_recipe, problems::{Problem, mat_as_nested_vec}
};

pub struct HamiltonianTermsCalc<P> {
    phantom: PhantomData<P>,
}

impl<P> HamiltonianTermsCalc<P> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct HamiltonianTermsInput {
    pub term_name: Option<Box<str>>,
    pub distance: Option<Scalar<Length>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OperatorData {
    matrix: Vec<Vec<f64>>,
    coupling: f64,
}

impl<P: Problem> SingleCalc for HamiltonianTermsCalc<P> {
    type P = P;
    type CalcInput = HamiltonianTermsInput;
    type Data = HashMap<Box<str>, OperatorData>;

    fn calculate(
        &self,
        modified: &mut Modified<P, HamiltonianTermsInput>,
        _problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        if let Some(name) = &modified.calc_input.term_name {
            if let Some(distance) = &modified.calc_input.distance {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

                let potential = &modified.system.potentials[name.as_ref()];
                let matrix = mat_as_nested_vec(potential.masking.as_ref());
                let coupling = potential.interaction.value(converter.scalar_value(distance));

                return [Ok(HashMap::from([(name.to_owned(), OperatorData { matrix, coupling })]))]
            } else {
                let operator = &modified.system.operators[name.as_ref()];
                let matrix = mat_as_nested_vec(operator.1.as_matrix().as_ref());
                let coupling = operator.0;

                return [Ok(HashMap::from([(name.to_owned(), OperatorData { matrix, coupling })]))]
            }
        }

        if let Some(distance) = &modified.calc_input.distance {
            let mut terms = HashMap::with_capacity(modified.system.potentials.vec.len());
            let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

            for name in modified.system.potentials.map.keys() {
                let potential = &modified.system.potentials[name.as_ref()];
                let matrix = mat_as_nested_vec(potential.masking.as_ref());
                let coupling = potential.interaction.value(converter.scalar_value(distance));
                terms.insert(name.to_owned(), OperatorData { matrix, coupling });
            }

            [Ok(terms)]
        } else {
            let mut terms = HashMap::with_capacity(modified.system.potentials.vec.len());

            for name in modified.system.potentials.map.keys() {
                let operator = &modified.system.operators[name.as_ref()];
                let matrix = mat_as_nested_vec(operator.1.as_matrix().as_ref());
                let coupling = operator.0;
                terms.insert(name.to_owned(), OperatorData { matrix, coupling });
            }

            [Ok(terms)]
        }
    }
}

pub fn hamiltonian_terms_calc<P: Problem + 'static>() -> Box<dyn DynCalc<P>> {
    let registry = ModifyRegistry::from([(
        "distance",
        modify_recipe!(|d| ScalarCalcMod::new(d, |r: &mut HamiltonianTermsInput, v| r.distance = Some(v))),
    )]);

    Box::new(DependenceCalc::new(HamiltonianTermsCalc::new(), registry))
}