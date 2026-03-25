pub mod bound_states;
pub mod coupling;
pub mod s_matrix;

use cc_propagator::multi_channel::Matrix;
pub use cc_propagator::{
    self,
    multi_channel,
};
pub use single_chan::{
    self,
    interaction::*,
};

pub type Operator = hilbert_space::operator::Operator<Matrix>;

/// Perform U O U^T
pub fn transform(m: &Matrix, transformation: &Matrix) -> Matrix {
    transformation * m * transformation.transpose()
}

/// Perform U^T O U
pub fn transform_rev(m: &Matrix, transformation: &Matrix) -> Matrix {
    transformation.transpose() * m * transformation
}
