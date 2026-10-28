use std::collections::HashMap;

use anyhow::{
    Context,
    Result,
};
use hilbert_space::faer::MatRef;
use schemars::Schema;
use serde::Deserialize;
use serde_json::Value;

use crate::InputData;
use crate::calculations::DynCalc;
use cc_system::hamiltonian_spec::HamiltonianSpec;
use cc_system::parameters::Parameters;

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProblemInput {
    pub problem_name: Box<str>,
    pub basis_recipe: Value,
    pub parameters: Value,
    pub calculation: Box<str>,
    pub calculation_parameters: Value,

    pub worker: usize,
    pub workers: usize,
}

#[derive(Clone)]
pub struct TypedProblemInput<B: InputData, P: InputData, I: InputData> {
    pub(crate) basis_recipe: B,
    pub(crate) parameters: P,
    pub(crate) calc_parameters: I,

    pub(crate) worker: usize,
    pub(crate) workers: usize,
}

pub trait DynProblem {
    fn name(&self) -> &str;
    fn schema(&self) -> Schema;
    fn run(&self, problem_input: ProblemInput) -> Result<()>;
}

impl<P: Problem> DynProblem for P {
    fn name(&self) -> &str {
        P::NAME
    }

    fn schema(&self) -> Schema {
        let problem_name = P::NAME;
        let description = P::DESCRIPTION;
        let mut schema_generator = schemars::SchemaGenerator::default();
        let basis = schema_generator.subschema_for::<P::BasisRecipe>();
        let params = schema_generator.subschema_for::<P::Params>();

        let calcs: Vec<(&Box<str>, schemars::Schema)> = self
            .calculations()
            .iter()
            .map(|(key, value)| (key, value.as_ref().subschema(&mut schema_generator)))
            .collect();

        let calc_names: Vec<&str> = calcs.iter().map(|(s, _)| s.as_ref()).collect();
        let calc_if_statements: Vec<Value> = calcs
            .iter()
            .map(|(s, x)| {
                serde_json::json!({
                    "if": {
                        "properties": { "calculation": { "const": s.as_ref() } }
                    },
                    "then": {
                        "properties": {
                            "calculation_parameters": x
                        }
                    }
                })
            })
            .collect();

        let mut schema = schema_generator.into_root_schema_for::<ProblemInput>();
        let object = schema.ensure_object();

        object.insert("title".into(), Value::String(problem_name.into()));
        object.insert("description".into(), Value::String(description.into()));
        object.insert("type".into(), "object".into());
        object.insert(
            "properties".into(),
            serde_json::json!({
                "problem_name": {
                    "description": description,
                    "const": problem_name
                },
                "basis_recipe": basis,
                "parameters": params,
                "calculation": {
                    "type": "string",
                    "enum": calc_names
                },
                "calculation_parameters": {
                    "type": "object"
                },
            }),
        );
        object.insert(
            "required".into(),
            serde_json::json!([
                "problem_name",
                "basis_recipe",
                "parameters",
                "calculation",
                "calculation_parameters"
            ]),
        );
        object.insert("allOf".into(), Value::Array(calc_if_statements));
        object.insert("additionalProperties".into(), Value::Bool(true));

        schema
    }

    fn run(&self, problem_input: ProblemInput) -> Result<()> {
        let map = &self.calculations();

        let basis_schema = schemars::schema_for!(P::BasisRecipe);
        let parameters_schema = schemars::schema_for!(P::Params);

        jsonschema::validate(basis_schema.as_value(), &problem_input.basis_recipe)
            .map_err(|err| err.to_owned())
            .context("failed to validate basis_recipe")?;

        jsonschema::validate(parameters_schema.as_value(), &problem_input.parameters)
            .map_err(|err| err.to_owned())
            .context("failed to validate parameters")?;

        let input = TypedProblemInput {
            basis_recipe: serde_json::from_value(problem_input.basis_recipe)?,
            parameters: serde_json::from_value(problem_input.parameters)?,
            calc_parameters: problem_input.calculation_parameters,
            worker: problem_input.worker,
            workers: problem_input.workers,
        };

        let calculation = map.get(&problem_input.calculation).ok_or(anyhow::anyhow!(
            "Did not find calculation {}, available calculations are {:?}",
            problem_input.calculation,
            map.keys()
        ))?;

        let calc_schema = calculation.schema();
        jsonschema::validate(calc_schema.as_value(), &input.calc_parameters)
            .map_err(|err| err.to_owned())
            .context("failed to validate calculation_parameters")?;

        calculation.run(input, self)
    }
}

pub trait Problem: Sync + Send {
    type BasisRecipe: InputData;
    type Params: Parameters + InputData;

    const NAME: &str;
    const DESCRIPTION: &str;

    fn build(basis_recipe: &Self::BasisRecipe, params: &Self::Params) -> HamiltonianSpec;
    fn calculations(&self) -> &HashMap<Box<str>, Box<dyn DynCalc<Self>>>;
}

pub(crate) fn mat_as_nested_vec<T: Copy>(mat: MatRef<T>) -> Vec<Vec<T>> {
    let mut vec = vec![vec![]; mat.nrows()];
    for (i, row) in mat.row_iter().enumerate() {
        vec[i] = row.iter().copied().collect();
    }
    vec
}
