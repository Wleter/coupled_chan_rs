pub mod dyn_space;
pub mod operator;
pub mod static_space;

#[cfg(feature = "faer")]
pub use faer;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

#[cfg(feature = "ndarray")]
pub use ndarray;

#[derive(Clone, Copy, Debug, Default)]
pub enum Parity {
    All,
    #[default]
    Even,
    Odd,
}
