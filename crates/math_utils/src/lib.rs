use std::{error::Error, fmt::Display};

pub mod bessel;
pub mod legendre;

/// Creates evenly spaced grid of points [start, end] (including) with n points.
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n as f64 - 1.0);

    for i in 0..n {
        result.push(start + (i as f64) * step);
    }

    result
}

/// Creates logarithmically spaced grid of points [10^start, 10^end] (including) with n points.
pub fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![10.0f64.powf(start)];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n as f64 - 1.0);

    for i in 0..n {
        result.push((10.0f64).powf(start + (i as f64) * step));
    }

    result
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RootError {
    NoConvergence,
    RootOutside,
}

impl Display for RootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootError::NoConvergence => writeln!(f, "Root Error: No convergence"),
            RootError::RootOutside => writeln!(f, "Root Error: Root outside of bounds"),
        }
    }
}
impl Error for RootError {}

pub fn brent_root_method(
    lower: [f64; 2],
    upper: [f64; 2],
    f: impl Fn(f64) -> f64,
    err: f64,
    max_iter: u32,
) -> Result<f64, RootError> {
    let (mut a, mut ya, mut b, mut yb) = sort_secant(lower[0], lower[1], upper[0], upper[1]);

    let (mut c, mut yc, mut d) = (a, ya, a);
    let mut bisection_last = true;

    for _ in 0..max_iter {
        if (a - b).abs() < err {
            return Ok(c);
        }

        let yab = ya - yb;
        let yac = ya - yc;
        let ybc = yb - yc;

        let mut s = if (ya != yc) && (yb != yc) {
            a * yb * yc / (yab * yac) + b * ya * yc / (-yab * ybc) + c * ya * yb / (yac * ybc)
        } else {
            b - yb * (b - a) / (-yab)
        };

        let cond1 = (s - b) * (s - (3. * a + b) / 4.) > 0.;
        let cond2 = bisection_last && (s - b).abs() >= (b - c).abs() / 2.;
        let cond3 = !bisection_last && (s - b).abs() >= (c - d).abs() / 2.;
        let cond4 = bisection_last && (b - c).abs() < err;
        let cond5 = !bisection_last && (c - d).abs() < err;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            s = (a + b) / 2.;
            bisection_last = true;
        } else {
            bisection_last = false;
        }

        if (s - b).abs() < err {
            return Ok(s);
        }

        let ys = f(s);
        d = c;
        c = b;
        yc = yb;
        if ya * ys < 0. {
            (a, ya, b, yb) = sort_secant(a, ya, s, ys)
        } else {
            (a, ya, b, yb) = sort_secant(s, ys, b, yb)
        }
    }

    Err(RootError::NoConvergence)
}

fn sort_secant(a: f64, ya: f64, b: f64, yb: f64) -> (f64, f64, f64, f64) {
    if ya.abs() > yb.abs() { (a, ya, b, yb) } else { (b, yb, a, ya) }
}

#[macro_export]
/// Asserts relative error |x - y| <= x * err
///
/// # Syntax
///
/// - `assert_approx_eq!(x, y, err)` for single element
/// - `assert_approx_eq!(iter => x, y, err)` for all elements in slice
/// - `assert_approx_eq!(mat => x, y, err)` for all elements in matrices
macro_rules! assert_approx_eq {
    ($x:expr, $y:expr, $err:expr $(, $message:expr)?) => {
        if $x == $y {
        } else if ($x - $y).abs() > $x.abs() * $err {
            panic!("assertion failed\nleft side: {:e}\nright side: {:e}", $x, $y)
        }
    };
    (iter => $x:expr, $y:expr, $err:expr) => {
        assert_eq!($x.len(), $y.len());

        for (x, y) in $x.iter().zip(&$y) {
            assert_approx_eq!(x, y, $err);
        }
    };
    (mat => $x:expr, $y:expr, $err:expr) => {
        assert_eq!($x.nrows(), $y.nrows());
        assert_eq!($x.ncols(), $y.ncols());

        for i in 0..$x.nrows() {
            for j in 0..$x.ncols() {
                assert_approx_eq!($x[(i, j)], $y[(i, j)], $err);
            }
        }
    };
}

#[macro_export]
/// Check for approximate error |x - y| <= x * err
///
/// # Syntax
///
/// - `approx_eq!(x, y, err)`
macro_rules! approx_eq {
    ($x:expr, $y:expr, $err:expr) => {
        ($x - $y).abs() <= $x.abs() * $err
    };
}

#[cfg(test)]
mod tests {
    use crate::{linspace, logspace};

    #[test]
    fn test_grids() {
        let grid = linspace(1.0, 15.0, 6);
        let expected = vec![1.0, 3.8, 6.6, 9.4, 12.2, 15.0];

        assert_approx_eq!(iter => grid, expected, 1e-6);

        let grid = logspace(-2.0, 3.0, 4);
        let expected = vec![0.01, 0.46415888, 21.5443469, 1000.0];

        assert_approx_eq!(iter => grid, expected, 1e-6);
    }
}
