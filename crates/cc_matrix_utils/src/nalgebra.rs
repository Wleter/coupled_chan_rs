use nalgebra::{DMatrix, SMatrix, Scalar};

use crate::{MatrixCreation, MatrixLike};
impl<E: Scalar> MatrixLike for DMatrix<E> {}
impl<E: Scalar, const N: usize, const M: usize> MatrixLike for SMatrix<E, N, M> {}

impl<E: Scalar> MatrixCreation<E> for DMatrix<E> {
    fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        DMatrix::from_fn(nrows, ncols, f)
    }
}

impl<E: Scalar, const N: usize, const M: usize> MatrixCreation<E> for SMatrix<E, N, M> {
    fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        assert_eq!(nrows, N, "nrows mismatch between constant static rows number");
        assert_eq!(ncols, M, "ncols mismatch between constant static cols number");

        SMatrix::from_fn(f)
    }
}
