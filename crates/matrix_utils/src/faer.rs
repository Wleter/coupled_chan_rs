use faer::Mat;

use crate::MatrixCreation;

impl<E> MatrixCreation<E> for Mat<E> {
    fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> E) -> Self {
        Mat::from_fn(nrows, ncols, f)
    }
}
