
use cc_problems::{
    atom_basis::{
        AtomRecipe,
        RecipeWithProj,
    },
    dependence::DependenceCalcInput,
    diatom_basis::DiatomRecipe,
    scattering::ScatteringCalcInput,
};

use serde::{
    Deserialize,
    Serialize,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SystemBasis {
    Atom(RecipeWithProj<AtomRecipe>),
    Diatom(RecipeWithProj<DiatomRecipe>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum SystemParams {
    Atom(RecipeWithProj<AtomRecipe>),
    Diatom(RecipeWithProj<DiatomRecipe>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Calculation {
    Scattering(DependenceCalcInput<ScatteringCalcInput>),
}
