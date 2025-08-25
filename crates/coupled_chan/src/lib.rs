pub mod coupling;
pub mod s_matrix;
pub mod ratio_numerov;

pub use constants;
use faer::Mat;
use hilbert_space::operator::Operator;
pub use propagator;
pub use single_chan::interaction::*;

#[derive(Debug, Clone)]
pub struct Channels(pub Mat<f64>);

impl Channels {
    pub fn size(&self) -> usize {
        assert_eq!(self.0.nrows(), self.0.ncols(), "Mismatched number of columns vs rows");

        self.0.nrows()
    }

    pub fn zeros(size: usize) -> Self {
        Channels(Mat::zeros(size, size))
    }
}

impl From<Operator<Mat<f64>>> for Channels {
    fn from(value: Operator<Mat<f64>>) -> Self {
        Channels(value.backed())
    }
}