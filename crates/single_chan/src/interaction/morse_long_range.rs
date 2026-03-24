use crate::interaction::{
    AsymptoteDep, Interaction, dispersion::PowerLaw
};

#[derive(Clone, Debug)]
pub struct MorseLongRangeBuilder {
    d0: f64,
    r_e: f64,
    tail: Vec<PowerLaw>,

    p: Option<i32>,
    q: Option<i32>,

    r_ref: Option<f64>,
    rho: Option<f64>,
    betas: Vec<f64>,
}

impl MorseLongRangeBuilder {
    pub fn new(d0: f64, r_e: f64, tail: Vec<PowerLaw>) -> Self {
        assert!(!tail.is_empty(), "Tail is empty");
        for t in &tail {
            assert!(matches!(t.asymptote_dep(), AsymptoteDep::PowerLawVanishing(_)), "Tail should e power law vanishing")
        }

        Self {
            d0,
            tail,
            p: None,
            q: None,
            r_ref: None,
            r_e,
            rho: None,
            betas: vec![],
        }
    }

    pub fn set_betas(self, betas: Vec<f64>) -> Self {
        Self { betas, ..self }
    }

    pub fn set_params(self, p: i32, q: i32, r_ref: f64, rho: f64) -> Self {
        Self {
            p: Some(p),
            q: Some(q),
            r_ref: Some(r_ref),
            rho: Some(rho),
            ..self
        }
    }

    pub fn build(self) -> MorseLongRange {
        let p = self.p.unwrap();
        let q = self.q.unwrap();
        let r_ref = self.r_ref.unwrap();
        let r_e = self.r_e;
        let rho = self.rho;
        let betas = self.betas;

        let tail_re: f64 = if let Some(rho) = rho {
            self.tail
                .iter()
                .map(|tail| douketis_damping(r_e, rho, -tail.n) * tail.value(r_e))
                .sum()
        } else {
            self.tail.iter().map(|tail| tail.value(r_e)).sum()
        };

        let b_inf = (2.0f64 * self.d0 / tail_re).ln();

        MorseLongRange {
            d0: self.d0,
            tail: self.tail,
            p,
            q,
            r_ref,
            r_e,
            rho,
            betas,
            b_inf,
            tail_re,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MorseLongRange {
    d0: f64,
    tail: Vec<PowerLaw>,
    p: i32,
    q: i32,
    r_ref: f64,
    r_e: f64,
    rho: Option<f64>,
    betas: Vec<f64>,
    b_inf: f64,

    tail_re: f64,
}

impl MorseLongRange {
    fn u_lr(&self, r: f64) -> f64 {
        if let Some(rho) = self.rho {
            self.tail
                .iter()
                .map(|tail| douketis_damping(r, rho, -tail.n) * tail.value(r))
                .sum()
        } else {
            self.tail.iter().map(|tail| tail.value(r)).sum()
        }
    }

    fn beta(&self, r: f64) -> f64 {
        let y_p = y_func(r, self.p, self.r_ref);
        let y_q = y_func(r, self.q, self.r_ref);

        let beta_factor = self
            .betas
            .iter()
            .enumerate()
            .map(|(i, beta)| beta * y_q.powi(i as i32))
            .sum::<f64>();

        self.b_inf * y_p + (1. - y_p) * beta_factor
    }
}

impl Interaction for MorseLongRange {
    fn value(&self, r: f64) -> f64 {
        let exponent = (-self.beta(r) * y_func(r, self.p, self.r_e)).exp();

        self.d0 * (1. - self.u_lr(r) / self.tail_re * exponent).powi(2) - self.d0
    }
    
    fn asymptote_dep(&self) -> super::AsymptoteDep {
        let smallest = self.tail.iter()
            .min_by_key(|x| if let AsymptoteDep::PowerLawVanishing(w) = x.asymptote_dep() {
                w
            } else {
                unreachable!("builder asserts power law vanishing")
            });

        // builder asserts at least 1 element
        smallest.unwrap().asymptote_dep()
    }
}

#[inline]
fn douketis_damping(r: f64, rho: f64, m: i32) -> f64 {
    (1. - (-(3.3 + 0.423 * rho * r) * rho * r / (m as f64)).exp()).powi(m - 1)
}

#[inline]
fn y_func(r: f64, n: i32, x: f64) -> f64 {
    (r.powi(n) - x.powi(n)) / (r.powi(n) + x.powi(n))
}
