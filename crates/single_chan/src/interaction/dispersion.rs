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
        match self.n {
            0 => AsymptoteDep::Const,
            -255..0 => AsymptoteDep::PowerLawVanishing((-self.n) as u8),
            i32::MIN..-255 => AsymptoteDep::PowerLawVanishing((-self.n) as u8),
            1..=i32::MAX => AsymptoteDep::Growing,
        }
    }
}

pub fn lennard_jones(d6: f64, r6: f64) -> Composite<PowerLaw> {
    let c12 = d6 * r6.powi(12);
    let c6 = -2.0 * d6 * r6.powi(6);

    Composite::new(vec![PowerLaw::new(c12, -12), PowerLaw::new(c6, -6)])
}
