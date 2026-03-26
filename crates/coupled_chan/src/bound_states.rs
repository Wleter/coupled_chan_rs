use cc_propagator::multi_channel::Matrix;

#[derive(Debug, Clone)]
pub struct WaveFunction {
    pub distances: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

impl WaveFunction {
    pub fn reverse(&mut self) {
        self.distances.reverse();
        self.values.reverse();
    }

    pub fn normalize(mut self) -> Self {
        let normalization: f64 = self
            .distances
            .windows(2)
            .zip(self.values.windows(2))
            .map(|(x, f)| unsafe {
                let f1 = f.get_unchecked(1);
                let f0 = f.get_unchecked(0);
                let f1_norm = f1.iter().fold(0., |acc, x| acc + x * x);
                let f0_norm = f0.iter().fold(0., |acc, x| acc + x * x);

                0.5 * (x.get_unchecked(1) - x.get_unchecked(0)) * (f1_norm + f0_norm)
            })
            .sum();

        for v in &mut self.values {
            for p in v {
                *p /= normalization.sqrt()
            }
        }

        self
    }

    pub fn occupations(&self) -> Vec<f64> {
        self.distances
            .windows(2)
            .zip(self.values.windows(2))
            .fold(vec![0.; self.values[0].len()], |mut acc, (d, v)| {
                for (i, acc) in acc.iter_mut().enumerate() {
                    *acc += 0.5 * (d[1] - d[0]) * (v[1][i].powi(2) + v[0][i].powi(2))
                }

                acc
            })
    }
}

#[derive(Clone, Debug)]
pub struct BoundMismatch {
    pub parameter: f64,
    pub nodes: u64,
    pub matching_matrix: Matrix,
    pub matching_eigenvalues: Vec<f64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub enum NodeMonotony {
    Decreasing,
    #[default]
    Increasing,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundMethod {
    Brent(u32),
    Bisection,
}

impl Default for BoundMethod {
    fn default() -> Self {
        Self::Brent(30)
    }
}

#[derive(Clone, Debug, Copy)]
pub enum NodeRangeTarget {
    Range(u64, u64),
    BottomRange(u64),
    TopRange(u64),
}
