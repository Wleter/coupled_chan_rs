pub mod matrix_elements;
pub mod operator;
pub mod space;

#[cfg(feature = "faer")]
pub use faer;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

#[cfg(feature = "ndarray")]
pub use ndarray;

use serde::{
    Deserialize,
    Serialize,
};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum Parity {
    All,
    #[default]
    Even,
    Odd,
}
