pub mod coupling;
pub mod ratio_numerov;
pub mod s_matrix;

pub use constants;
use faer::Mat;
pub use propagator;
pub use single_chan::interaction::*;

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;