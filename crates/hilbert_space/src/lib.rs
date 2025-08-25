pub mod dyn_space;
pub mod operator;
pub mod static_space;

#[cfg(feature = "faer")]
pub use faer;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

#[cfg(feature = "ndarray")]
pub use ndarray;

#[derive(Clone, Copy, Debug)]
pub enum Parity {
    All,
    Even,
    Odd,
}