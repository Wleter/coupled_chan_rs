use std::f64::consts::PI;

use spline_interpolation::UniformSpline;

use crate::interaction::Interaction;
pub use spline_interpolation;

#[derive(Clone)]
pub struct InterpolatedPotential(pub UniformSpline);

impl Interaction for InterpolatedPotential {
    fn value(&self, r: f64) -> f64 {
        self.0.eval(r)
    }
}

#[derive(Clone, Debug)]
pub struct Transitioned<P, V, F>
where
    P: Interaction,
    V: Interaction,
    F: Fn(f64) -> f64,
{
    pub near: P,
    pub far: V,
    transition: F,
}

impl<P, V, F> Transitioned<P, V, F>
where
    P: Interaction,
    V: Interaction,
    F: Fn(f64) -> f64,
{
    /// Creates new transitioned interaction where transition
    /// is a function r -> [0, 1]. 0 correspond to only near interaction
    /// and 1 correspond to only far interaction
    pub fn new(near: P, far: V, transition: F) -> Self {
        Self {
            near,
            far,
            transition,
        }
    }
}

impl<P, V, F> Interaction for Transitioned<P, V, F>
where
    P: Interaction,
    V: Interaction,
    F: Fn(f64) -> f64,
{
    fn value(&self, r: f64) -> f64 {
        let a = (self.transition)(r);
        assert!((0. ..=1.).contains(&a));

        if a == 0. {
            self.near.value(r)
        } else if a == 1. {
            self.far.value(r)
        } else {
            (1. - a) * self.near.value(r) + a * self.far.value(r)
        }
    }
}

/// Creates transition of the form
/// 1/2 + 1/4 * sin(x) * (3 - sin^2(x))
/// 
/// x is in range [-pi/2, pi/2] for r in [a, b]
/// 
/// for r <= a it is 0 for r >= b it is 1
pub fn sin_transition(a: f64, b: f64) -> impl Fn(f64) -> f64 + Clone {
    assert!(a < b, "0 width sin transition range");

    move |r| {
        if r <= a {
            0.
        } else if r >= b{
            1.
        } else {
            let x = ((r - b) - (a - r)) / (b - a);
            0.5 + 0.25 * f64::sin(PI / 2. * x) * (3. - f64::sin(PI / 2. * x).powi(2))
        }
    }
}
