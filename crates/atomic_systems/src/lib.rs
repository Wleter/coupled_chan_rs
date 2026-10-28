use serde::{
    Deserialize,
    Serialize,
};

pub mod atom_basis;
pub mod atom_operators;
pub mod diatom_basis;
pub mod diatom_operators;
pub mod interactions;
pub mod operator_mel;
pub mod rotor_atom_basis;
pub mod tram_basis;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(deny_unknown_fields)]
pub enum Parity {
    All,
    #[default]
    Even,
    Odd,
}

#[derive(schemars::JsonSchema)]
#[allow(unused)]
pub struct HalfU32Scheme(#[schemars(extend("multipleOf" = 0.5), range(min = 0))] f64);

#[derive(schemars::JsonSchema)]
#[allow(unused)]
pub struct HalfI32Scheme(#[schemars(extend("multipleOf" = 0.5), range(min = 0))] f64);
