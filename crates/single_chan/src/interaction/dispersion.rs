use cc_qol_utils::Composite;

use crate::interaction::{
    AsymptoteDep,
    Interaction,
};

/// Potential of the form d0 * r^n
#[derive(Debug, Clone, Copy)]
pub struct PowerLaw {
    pub d0: f64,
    pub n: i32,
}

impl PowerLaw {
    pub fn new(d0: f64, n: i32) -> Self {
        Self { d0, n }
    }
}

impl Interaction for PowerLaw {
    fn value(&self, r: f64) -> f64 {
        self.d0 * r.powi(self.n)
    }

    fn asymptote_dep(&self) -> super::AsymptoteDep {
        if self.d0 == 0.0 {
            return AsymptoteDep::Const
        }

        match self.n {
            0 => AsymptoteDep::Const,
            -255..0 => AsymptoteDep::PowerLawVanishing((-self.n) as u8),
            i32::MIN..-255 => AsymptoteDep::PowerLawVanishing((-self.n) as u8),
            1..=i32::MAX => AsymptoteDep::Growing,
        }
    }
}

/// Potential of the form d * e^(-exponent * (r - r_offset))
#[derive(Debug, Clone, Copy)]
pub struct ExpLaw {
    pub d: f64,
    pub exponent: f64,
    pub r_offset: f64,
}

impl ExpLaw {
    pub fn new(d: f64, exponent: f64, r_offset: f64) -> Self {
        Self { d, exponent, r_offset }
    }
}

impl Interaction for ExpLaw {
    fn value(&self, r: f64) -> f64 {
        self.d * f64::exp(-(r - self.r_offset))
    }

    fn asymptote_dep(&self) -> super::AsymptoteDep {
        if self.d == 0.0 {
            return AsymptoteDep::Const
        }

        match self.exponent {
            x if x > 0.0 => AsymptoteDep::ExpVanishing,
            x if x == 0.0 => AsymptoteDep::Const,
            x if x < 0.0 => AsymptoteDep::Growing,
            _ => AsymptoteDep::Unknown
        }
    }
}

pub struct AnalyticInteraction {
    pub power_law: PowerLaw,
    pub exponent_law: ExpLaw,
}

impl Interaction for AnalyticInteraction {
    fn value(&self, r: f64) -> f64 {
        self.power_law.value(r) * self.exponent_law.value(r)
    }

    fn asymptote_dep(&self) -> super::AsymptoteDep {
        if let AsymptoteDep::Const = self.exponent_law.asymptote_dep() {
            self.power_law.asymptote_dep()
        } else {
            self.exponent_law.asymptote_dep()
        }
    }
}

/// Creates Lennard-Jones potential of the form
/// d6 ((r6/r)^12 - 2 (r6/r)^6), where
/// `d6`, `r6` are well minimum value and it's distances.
pub fn lennard_jones(d6: f64, r6: f64) -> Composite<PowerLaw> {
    let c12 = d6 * r6.powi(12);
    let c6 = -2.0 * d6 * r6.powi(6);

    Composite::new(vec![PowerLaw::new(c12, -12), PowerLaw::new(c6, -6)])
}
