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
