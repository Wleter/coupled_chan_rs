use core::panic;
use std::{
    fs,
    path::Path,
};

use clebsch_gordan::half_integer::HalfU32;
use json_comments::StripComments;
use serde::{
    Deserialize,
    Serialize,
};

pub fn read_input(path: impl AsRef<Path>) -> Input {
    let path = path.as_ref();
    if let Some(ext) = path.extension() {
        let read = fs::read_to_string(path).unwrap();
        match ext.to_str().unwrap() {
            "toml" => toml::from_str(&read).unwrap(),
            "json" | "jsonc" => serde_json::from_reader(StripComments::new(read.as_bytes())).unwrap(),
            _ => panic!("Unknown file extension type"),
        }
    } else {
        panic!("Expected path to contain file extension")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Input {
    system: System,
    calculation: Calculation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum System {
    AtomAtom {
        s_a: HalfU32,
        i_a: HalfU32,
        s_b: HalfU32,
        i_b: HalfU32,
    },
    #[serde(rename = "atom_rotor_TRAM")]
    AtomDiatom {
        s_a: HalfU32,
        i_a: HalfU32,
        s_b: HalfU32,
        i_b: HalfU32,
        s_c: HalfU32,
        i_c: HalfU32,
    },
    Custom {
        basis: Vec<Basis>,
        operators: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Basis {
    Spin(String, HalfU32),
    QuantumNumbers(String, Vec<i32>),
    Grid(String, Vec<f64>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Calculation {
    Levels {
        parameter: String,
        range: Range,
    },
    Custom {
        identifier: String,
    },
    BoundState {
        parameter: String,
        range: Range,
        propagator: Propagator,
        step: Step,
        r_match: f64,
    },
    Scattering {
        propagator: Propagator,
        parameter: String,
        range: Range,
        step: Step,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Range {
    Linear { start: f64, stop: f64, n: u32 },
    Logarithmic { arg_start: f64, arg_stop: f64, n: u32 },
    Points(Vec<f64>),
    Composite(Vec<Range>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Fixed {
        dr: f64,
    },
    LocalWaveLength {
        dr_min: f64,
        dr_max: f64,
        wave_ratio: f64,
    },
    Transitioned {
        transition_point: f64,
        before: Box<Step>,
        after: Box<Step>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Propagator {
    RatioNumerov,
    JohnsonLogDeriv,
    ManolopoulosLogDeriv,
}
