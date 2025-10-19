use puruspe::utils::factorial;

fn hyper_geometric_series(m: u32, b: u32, c: u32) -> impl Iterator<Item = f64> {
    let m = m as usize;
    let b = b as usize;
    let c = c as usize;

    (0..=m).map(move |n| {
        (-1f64).powi(n as i32) * factorial(m) / factorial(n) / factorial(m - n) * factorial(b + n - 1)
            / factorial(b - 1)
            / (factorial(c + n - 1) / factorial(c - 1))
    })
}

pub trait ReproducingKernel {
    fn value(&self, x1: f64, x2: f64) -> f64;
}

pub struct ReciprocalPowerKernel {
    pub m: u32,
    pub n: u32,

    series_factors: Vec<f64>,
}

impl ReciprocalPowerKernel {
    pub fn new(m: u32, n: u32) -> Self {
        assert!(n > 0, "n factor should be positive");

        let prefactor = n.pow(2) as f64 * puruspe::beta((m + 1) as f64, n as f64);

        let series_factors = hyper_geometric_series(n - 1, m + 1, n + m + 1)
            .map(|b| prefactor * b)
            .collect();

        Self { m, n, series_factors }
    }
}

impl ReproducingKernel for ReciprocalPowerKernel {
    fn value(&self, x1: f64, x2: f64) -> f64 {
        let x_lower = x1.min(x2);
        let x_upper = x1.max(x2);
        let x_ratio = x_lower / x_upper;

        x_upper.powi(-(self.m as i32) - 1)
            * self
                .series_factors
                .iter()
                .enumerate()
                .map(|(k, &b)| b * x_ratio.powi(k as i32))
                .sum::<f64>()
    }
}
