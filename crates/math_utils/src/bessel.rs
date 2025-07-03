/// Calculates riccati bessel function of the first kind j_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j(n: u32, x: f64) -> f64 {
    bessel_recurrence(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x))
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n(n: u32, x: f64) -> f64 {
    bessel_recurrence(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x))
}

/// Calculates riccati bessel function of the first kind j_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as z j_n(z))
pub fn riccati_j_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) = bessel_recurrence_deriv(n, x, f64::sin(x), f64::sin(x) / x - f64::cos(x));

    (value, value_deriv + value / x)
}

/// Calculates riccati bessel function of the third kind n_n(x) = -y_n(x)
/// and the corresponding derivative
///
/// "Handbook of Mathematical Functions" - eq. 10.3.2 (written as -z y_n(z))
pub fn riccati_n_deriv(n: u32, x: f64) -> (f64, f64) {
    let (value, value_deriv) = bessel_recurrence_deriv(n, x, f64::cos(x), f64::cos(x) / x + f64::sin(x));

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
    let i_1 = modified_bessel_recurrence(n, x_1, red_i_0(x_1), red_i_1(x_1));
    let i_2 = modified_bessel_recurrence(n, x_2, red_i_0(x_2), red_i_1(x_2));

    f64::exp(x_1 - x_2) * i_1 / i_2
}

/// Calculates ratio of the riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x)) at points `x_1`, `x_2`
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn ratio_riccati_k(n: u32, x_1: f64, x_2: f64) -> f64 {
    let red_k_0 = |_| 1.0;
    let red_k_1 = |x| (1.0 + 1.0 / x);

    // Calculates riccati $(-1)^(n+1) * K$ bessel without leading exponent
    let k_1 = modified_bessel_recurrence(n, x_1, -red_k_0(x_1), red_k_1(x_1));
    let k_2 = modified_bessel_recurrence(n, x_2, -red_k_0(x_2), red_k_1(x_2));

    f64::exp(x_2 - x_1) * k_1 / k_2
}

/// Calculates the ratio of derivative and the value of the  
/// riccati modified spherical bessel function of the first kind
/// (that is $sqrt(x) I_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.2 (written as z * sqrt(pi/2z) I_{n+1/2}(z))
pub fn ratio_riccati_i_deriv(n: u32, x: f64) -> f64 {
    let red_i_0 = (1. - f64::exp(-2.0 * x)) / 2.0;
    let red_i_1 = -(1. - f64::exp(-2.0 * x)) / (2.0 * x) + (1. + f64::exp(-2.0 * x)) / 2.0;

    // Calculates riccati I bessel, without leading exponent, and its derivative
    let (i_red, i_red_deriv) = modified_bessel_recurrence_deriv(n, x, red_i_0, red_i_1);

    i_red_deriv / i_red + 1.0 / x
}

/// Calculates the ratio of derivative and the value of the  
/// riccati modified spherical bessel function of the third kind
/// (that is $sqrt(x) K_{n+1/2}(x))
///
/// "Handbook of Mathematical Functions" - eq. 10.2.4 (written as z * sqrt(pi/2z) K_{n+1/2}(z))
pub fn ratio_riccati_k_deriv(n: u32, x: f64) -> f64 {
    let red_k_0 = 1.0;
    let red_k_1 = 1.0 + 1.0 / x;

    // Calculates riccati $(-1)^(n+1) * K$ bessel without leading exponent
    let (k_red, k_red_deriv) = modified_bessel_recurrence_deriv(n, x, -red_k_0, red_k_1);

    k_red_deriv / k_red + 1.0 / x
}

/// Calculates f_n(x) given n, x, f_0(x), f_1(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19
fn bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
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
        f_new = (2 * k + 1) as f64 / x * f_k - f_k_1;
        f_k_1 = f_k;
        f_k = f_new;
    }

    f_k
}

/// Calculates f_n(x) and its derivative (g(x) d Bessel(x)/dx) given n, x, f_0(x), f_1(x)
/// "Handbook of Mathematical Functions" - eq. 10.1.19
fn bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
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

/// Calculates f_n(x) given n, x, f_0(x), f_1(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn modified_bessel_recurrence(n: u32, x: f64, f_0: f64, f_1: f64) -> f64 {
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

/// Calculates f_n(x) and its derivative (g(x) d MBessel(x)/dx) given n, x, f_0(x), f_1(x).
/// "Handbook of Mathematical Functions" - eq. 10.2.18
fn modified_bessel_recurrence_deriv(n: u32, x: f64, f_0: f64, f_1: f64) -> (f64, f64) {
    if n == 0 {
        return (f_0, f_1);
    }
    if n == 1 {
        return (f_1, f_0 - (n + 1) as f64 / x * f_1);
    }

    let mut f_k_1 = f_0;
    let mut f_k = f_1;
    let mut f_new;
    for k in 1..n {
        f_new = f_k_1 - (2 * k + 1) as f64 / x * f_k;
        f_k_1 = f_k;
        f_k = f_new;
    }

    (f_k, f_k_1 - (n + 1) as f64 / x * f_k)
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_approx_eq,
        bessel::{
            ratio_riccati_i, ratio_riccati_i_deriv, ratio_riccati_k, ratio_riccati_k_deriv, riccati_j, riccati_j_deriv,
            riccati_n, riccati_n_deriv,
        },
    };

    #[test]
    fn test_bessel() {
        assert_approx_eq!(riccati_j(5, 10.0), -0.555345, 1e-5);
        assert_approx_eq!(riccati_j(10, 10.0), 0.646052, 1e-5);

        assert_approx_eq!(riccati_n(5, 10.0), -0.938335, 1e-5);
        assert_approx_eq!(riccati_n(10, 10.0), 1.72454, 1e-5);

        assert_approx_eq!(ratio_riccati_i(5, 5.0, 10.0), 0.00157309, 1e-5);
        assert_approx_eq!(ratio_riccati_i(10, 5.0, 10.0), 0.00011066, 1e-5);

        assert_approx_eq!(ratio_riccati_k(5, 5.0, 10.0), 487.227, 1e-5);
        assert_approx_eq!(ratio_riccati_k(10, 5.0, 10.0), 5633.13, 1e-5);

        assert_approx_eq!(riccati_j_deriv(5, 10.0).0, -0.555345, 1e-5);
        assert_approx_eq!(riccati_j_deriv(5, 10.0).1, -0.77822, 1e-5);
        assert_approx_eq!(riccati_j_deriv(10, 10.0).0, 0.646052, 1e-5);
        assert_approx_eq!(riccati_j_deriv(10, 10.0).1, 0.354913, 1e-5);

        assert_approx_eq!(riccati_n_deriv(5, 10.0).0, -0.938335, 1e-5);
        assert_approx_eq!(riccati_n_deriv(5, 10.0).1, 0.485767, 1e-5);
        assert_approx_eq!(riccati_n_deriv(10, 10.0).0, 1.72454, 1e-5);
        assert_approx_eq!(riccati_n_deriv(10, 10.0).1, -0.600479, 1e-5);

        assert_approx_eq!(ratio_riccati_i_deriv(5, 10.0), 1.1531, 1e-5);
        assert_approx_eq!(ratio_riccati_i_deriv(10, 10.0), 1.47691, 1e-5);

        assert_approx_eq!(ratio_riccati_k_deriv(5, 10.0), -1.12973, 1e-5);
        assert_approx_eq!(ratio_riccati_k_deriv(10, 10.0), -1.42441, 1e-5);
    }
}
