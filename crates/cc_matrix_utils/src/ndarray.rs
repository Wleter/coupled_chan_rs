use ndarray::Array2;

use crate::{MatrixCreation, MatrixLike};
impl<E> MatrixLike for Array2<E> {}

impl<E> MatrixCreation<E> for Array2<E> {
    fn from_fn(nrows: usize, ncols: usize, mut f: impl FnMut(usize, usize) -> E) -> Self {
        Array2::from_shape_fn((nrows, ncols), |(i, j)| f(i, j))
    }
}
