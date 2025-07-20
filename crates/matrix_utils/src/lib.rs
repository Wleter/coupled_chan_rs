#[cfg(feature = "faer")]
pub mod faer;

#[cfg(feature = "ndarray")]
pub mod ndarray;

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

pub trait MatrixCreation<E> {
    fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self;
}

pub trait SquareMatrixOps {
    fn size(&self) -> usize;
}
