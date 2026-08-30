use std::collections::HashMap;

use cc_problems::diatom_problems::{DiatomInBFieldParams, DiatomInBFieldRecipe, diatom_levels_b_field_scan, diatom_scattering_b_field_scan};

use serde::{
    Deserialize,
    Serialize,
};

use crate::{CalcSpec, ProgramInput};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemInputs {
    DiatomInBField(ProgramInput<DiatomInBFieldRecipe, DiatomInBFieldParams>),
}

impl ProblemInputs {
    
}

pub fn diatom_problems() -> HashMap<Box<str>, CalcSpec<DiatomInBFieldRecipe, DiatomInBFieldParams>> {
    HashMap::from([
        ("levels".into(), CalcSpec::new(|_, _| Box::new(diatom_levels_b_field_scan()))),
        ("scattering scan".into(), CalcSpec::new(|_, p| Box::new(diatom_scattering_b_field_scan(p)))),
    ])
}