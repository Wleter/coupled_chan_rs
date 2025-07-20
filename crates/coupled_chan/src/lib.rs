pub mod coupling;
pub mod s_matrix;

pub use constants;
use faer::Mat;
pub use propagator;
pub use single_chan::interaction::*;

#[derive(Debug, Clone)]
pub struct Channels(pub Mat<f64>);

impl Channels {
    pub fn size(&self) -> usize {
        assert_eq!(self.0.nrows(), self.0.ncols(), "Mismatched number of columns vs rows");

        self.0.nrows()
    }
}
