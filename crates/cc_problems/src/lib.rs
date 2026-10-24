pub mod atom_basis;
pub mod atom_operators;
pub mod calculations;
pub mod diatom_basis;
pub mod diatom_operators;
pub mod interactions;
pub mod operator_mel;
pub mod problems;
pub mod rotor_atom_basis;
pub mod tram_basis;

use std::sync::{
    LazyLock,
    RwLock,
};

pub use cc_qol_utils;
pub use hilbert_space;
use hilbert_space::faer::Mat;

pub use smallvec;
pub use spin_algebra;
pub use unit_systems;
use unit_systems::quantities::UnitsConverter;
use unit_systems::{
    ATOMIC_UNITS_TABLE,
    unit_registry,
};

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;

pub static UNITS_CONVERTER: LazyLock<RwLock<UnitsConverter>> = LazyLock::new(|| {
    RwLock::new(UnitsConverter {
        registry: unit_registry(),
        target_unit_system: ATOMIC_UNITS_TABLE,
    })
});
