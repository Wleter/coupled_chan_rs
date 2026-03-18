/// Calculates riccati bessel function of the first kind j_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j(n: u32, x: f64) -> f64 {
    backwards_bessel_recurrence_deriv(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x)).0
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n(n: u32, x: f64) -> f64 {
    forwards_bessel_recurrence_deriv(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x)).0
}

/// Calculates riccati bessel function of the first kind j_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) = backwards_bessel_recurrence_deriv(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x));

    (value, value_deriv + value / x)
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) = forwards_bessel_recurrence_deriv(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x));

    (value, value_deriv + value / x)
}

/// Calculates ratio of the riccati modified spherical bessel function of the first kind
/// (that is $sqrt(x) I_{n+1/2}(x)) at points `x_1`, `x_2`
///
/// "Handbook of Mathematical Functions" - eq. 10.2.2 (written as z * sqrt(pi/2z) I_{n+1/2}(z))
pub fn ratio_riccati_i(n: u32, x_1: f64, x_2: f64) -> f64 {
    let red_i_0 = |x| (1. - f64::exp(-2.0 * x)) / 2.0;
    let red_i_1 = |x| -(1. - f64::exp(-2.0 * x)) / (2.0 * x) + (1. + f64::exp(-2.0 * x)) / 2.0;

    // Calculates riccati I bessel without leading exponent
    let i_1 = backwards_mod_bessel_recurrence(n, x_1, red_i_0(x_1), red_i_1(x_1));
    let i_2 = backwards_mod_bessel_recurrence(n, x_2, red_i_0(x_2), red_i_1(x_2));

    f64::exp(x_1 - x_2) * i_1 / i_2
}

/// Calculates ratio of the riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x)) at points `x_1`, `x_2`
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn ratio_riccati_k(n: u32, x_1: f64, x_2: f64) -> f64 {
    let red_k_0 = |_| 1.0;
    let red_k_1 = |x| 1.0 + 1.0 / x;

    // Calculates riccati $(-1)^(n+1) * K$ bessel without leading exponent
    let k_1 = forwards_mod_bessel_recurrence(n, x_1, -red_k_0(x_1), red_k_1(x_1));
    let k_2 = forwards_mod_bessel_recurrence(n, x_2, -red_k_0(x_2), red_k_1(x_2));

    f64::exp(x_2 - x_1) * k_1 / k_2
}

/// Calculates the log derivative of riccati modified spherical bessel function of the first kind
/// (that is $sqrt(x) I_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.2 (written as z * sqrt(pi/2z) I_{n+1/2}(z))
pub fn riccati_i_log_deriv(n: u32, x: f64) -> f64 {
    let red_i_0 = (1. - f64::exp(-2.0 * x)) / 2.0;
    let red_i_1 = -(1. - f64::exp(-2.0 * x)) / (2.0 * x) + (1. + f64::exp(-2.0 * x)) / 2.0;

    let log_deriv = backwards_mod_bessel_recurrence_log_deriv(n, x, red_i_1 / red_i_0);

    log_deriv + 1. / x
}

/// Calculates the log derivative of riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn riccati_k_log_deriv(n: u32, x: f64) -> f64 {
    let red_k_0 = 1.0;
    let red_k_1 = 1.0 + 1.0 / x;

    let log_deriv = forwards_mod_bessel_recurrence_log_deriv(n, x, -red_k_1 / red_k_0);

    log_deriv + 1. / x
}

/// Calculates f_n(x) and its derivative (g(x), d Bessel(x)/dx) given n, x, f_0(x), f_1(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19 using forward method
fn forwards_bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
    if n == 0 {
        return (f_0, -f_1);
    }
    if n == 1 {
        return (f_1, f_0 - (n + 1) as f64 / x * f_1);
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }

    (f_k, f_k_1 - (n + 1) as f64 / x * f_k)
}

const DELTA: u32 = 30;

/// Calculates f_n(x) and its derivative (g(x), d Bessel(x)/dx) given n, x, f_0(x), f_1(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19 using backwards method
fn backwards_bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
    if n == 0 {
        return (f_0, -f_1);
    }
    if n == 1 {
        return (f_1, f_0 - (n + 1) as f64 / x * f_1);
    }

    let n_max = n + x.ceil() as u32 + DELTA;

    let mut f_k_1 = 0.;
    let mut f_k = 1.;
    let mut f_new;
    for k in (n..=n_max).rev() {
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }
    let f_n = f_k_1;
    let f_n_minus_1 = f_k;
    for k in (1..n).rev() {
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }
    let scale = f_0 / f_k;

    let f_n = f_n * scale;
    let f_n_minus_1 = f_n_minus_1 * scale;

    (f_n, f_n_minus_1 - (n + 1) as f64 / x * f_n)
}

/// Calculates log derivative of f_n(x) given n, x, f_0(x), f_1(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18 using backwards method
fn backwards_mod_bessel_recurrence_log_deriv(n: u32, x: f64, r_0: f64) -> f64 {
    if n == 0 {
        return r_0;
    }

    let n_max = n + x.ceil() as u32 + DELTA;

    let mut r_k = 0.;
    for k in ((n + 1)..=n_max).rev() {
        r_k = 1. / ((2 * k + 1) as f64 / x + r_k);
    }
    let r_n = r_k;
    for k in (1..=n).rev() {
        r_k = 1. / ((2 * k + 1) as f64 / x + r_k);
    }
    let scale = r_0 / r_k;

    let r_n = r_n * scale;

    n as f64 / x + r_n
}

/// Calculates logarithmic derivative of f_n(x) given n, x, f_0(x), f_1(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18 using forwards method
fn forwards_mod_bessel_recurrence_log_deriv(n: u32, x: f64, r_0: f64) -> f64 {
    if n == 0 {
        return r_0;
    }

    let mut r_k = r_0;
    for k in 1..=n {
        r_k = 1. / r_k - (2 * k + 1) as f64 / x;
    }

    n as f64 / x + r_k
}

/// Calculates f_n(x) given n, x, f_0(x), f_1(x) using forwards propagation.
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn forwards_mod_bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
    if n == 0 {
        return f_0;
    }
    if n == 1 {
        return f_1;
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = f_k_1 - (2 * k + 1) as f64 / x * f_k;
        f_k_1 = f_k;
        f_k = f_new;
    }

    f_k
}

/// Calculates f_n(x) given n, x, f_0(x), f_1(x) using backwards propagation.
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn backwards_mod_bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
    if n == 0 {
        return f_0;
    }
    if n == 1 {
        return f_1;
    }
    let n_max = n + x.ceil() as u32 + DELTA;

    let mut f_k_1 = 0.;
    let mut f_k = 1.;
    let mut f_new;
    for k in (n..=n_max).rev() {
        f_new = (2 * k + 1) as f64 / x * f_k + f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }
    let f_n = f_k_1;
    for k in (1..n).rev() {
        f_new = (2 * k + 1) as f64 / x * f_k + f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }
    let scale = f_0 / f_k;

    f_n * scale
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_approx_eq,
        bessel::{
            ratio_riccati_i,
            ratio_riccati_k,
            riccati_i_log_deriv,
            riccati_j_deriv,
            riccati_k_log_deriv,
            riccati_n_deriv,
        },
    };

    #[test]
    fn test_bessel() {
        assert_approx_eq!(riccati_j_deriv(5, 1e-3).0, 9.62001e-23, 1e-5);
        assert_approx_eq!(riccati_j_deriv(5, 1e-3).1, 5.77201e-19, 1e-5);
        assert_approx_eq!(riccati_j_deriv(50, 1e-3).0, 3.63287e-234, 1e-5);
        assert_approx_eq!(riccati_j_deriv(50, 1e-3).1, 1.85276e-229, 1e-5);

        assert_approx_eq!(riccati_j_deriv(5, 1e1).0, -0.555345, 1e-5);
        assert_approx_eq!(riccati_j_deriv(5, 1e1).1, -0.77822, 1e-5);
        assert_approx_eq!(riccati_j_deriv(50, 1e1).0, 2.2307e-30, 1e-5);
        assert_approx_eq!(riccati_j_deriv(50, 1e1).1, 1.11579e-29, 1e-5);

        assert_approx_eq!(riccati_n_deriv(5, 1e-3).0, 9.45e17, 1e-5);
        assert_approx_eq!(riccati_n_deriv(5, 1e-3).1, -4.725e21, 1e-5);
        assert_approx_eq!(riccati_n_deriv(50, 1e-3).0, 2.72539e228, 1e-5);
        assert_approx_eq!(riccati_n_deriv(50, 1e-3).1, -1.36270e233, 1e-5);

        assert_approx_eq!(riccati_n_deriv(5, 1e1).0, -0.938335, 1e-5);
        assert_approx_eq!(riccati_n_deriv(5, 1e1).1, 0.485767, 1e-5);
        assert_approx_eq!(riccati_n_deriv(50, 1e1).0, 4.52823e28, 1e-5);
        assert_approx_eq!(riccati_n_deriv(50, 1e1).1, -2.21789e29, 1e-5);

        assert_approx_eq!(riccati_i_log_deriv(5, 1e-3), 6000., 1e-5);
        assert_approx_eq!(riccati_i_log_deriv(50, 1e-3), 51000., 1e-5);
        assert_approx_eq!(riccati_i_log_deriv(5, 1e1), 1.1531, 1e-5);
        assert_approx_eq!(riccati_i_log_deriv(50, 1e1), 5.1962056, 1e-5);

        assert_approx_eq!(riccati_k_log_deriv(5, 1e-3), -5000., 1e-5);
        assert_approx_eq!(riccati_k_log_deriv(50, 1e-3), -50000., 1e-5);
        assert_approx_eq!(riccati_k_log_deriv(5, 1e1), -1.12973, 1e-5);
        assert_approx_eq!(riccati_k_log_deriv(50, 1e1), -5.1, 1e-5);

        assert_approx_eq!(ratio_riccati_i(5, 5.0, 10.0), 0.00157309, 1e-5);
        assert_approx_eq!(ratio_riccati_i(10, 5.0, 10.0), 0.00011066, 1e-5);
        assert_approx_eq!(ratio_riccati_k(5, 5.0, 10.0), 487.227, 1e-5);
        assert_approx_eq!(ratio_riccati_k(10, 5.0, 10.0), 5633.13, 1e-5);
    }
}
