use std::path::PathBuf;

use cc_derive::Parameters;
use cc_qol_utils::Composite;
use coupled_chan::{DynInteraction, dispersion::{PowerLaw, lennard_jones}, morse_long_range};
use serde::{Deserialize, Serialize};
use unit_systems::quantities::{Inv, Power, Prod, Scalar, phys_quantities::{Energy, Length}};

use crate::UNITS_CONVERTER;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub struct C_N<const N: i8>(pub Scalar<Prod<Energy, Power<Length, N>>>);

impl<const N: i8> C_N<N> {
    pub fn interaction(&self) -> PowerLaw {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        PowerLaw::new(converter.scalar_value(&self.0), -N as i32)
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, Parameters)]
#[serde(default)]
pub struct Tail {
    pub c3: C_N<3>,
    pub c4: C_N<4>,
    pub c5: C_N<3>,
    pub c6: C_N<6>,
    pub c7: C_N<7>,
    pub c8: C_N<8>,
    pub c9: C_N<9>,
    pub c10: C_N<10>,
    pub c11: C_N<10>,
    pub c12: C_N<10>,
}

impl Tail {
    pub fn interaction(&self) -> Composite<PowerLaw> {
        let laws = vec![
            self.c3.interaction(),
            self.c4.interaction(),
            self.c5.interaction(),
            self.c6.interaction(),
            self.c7.interaction(),
            self.c8.interaction(),
            self.c9.interaction(),
            self.c10.interaction(),
            self.c11.interaction(),
            self.c12.interaction(),
        ];
        let filtered = laws.into_iter().filter(|x| x.d0 != 0.0).collect();

        Composite::new(filtered)
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Wall {
    pub c12: C_N<12>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LenardJones {
    pub d_e: Scalar<Energy>,
    pub r_e: Scalar<Length>,
}

impl LenardJones {
    pub fn interaction(&self) -> Composite<PowerLaw> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        lennard_jones(converter.scalar_value(&self.d_e), converter.scalar_value(&self.r_e))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Analytic {
    tail: Tail,
    wall: Wall,
}

impl Analytic {
    pub fn interaction(&self) -> Composite<PowerLaw> {
        let mut tail = self.tail.interaction();
        tail.add_component(self.wall.c12.interaction());

        tail
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorseLongRange {
    d0: Scalar<Energy>,
    r_e: Scalar<Length>,
    tail: Tail,

    p: Option<i32>,
    q: Option<i32>,

    r_ref: Option<Scalar<Length>>,
    rho: Option<Scalar<Inv<Length>>>,
    betas: Vec<f64>,
}

impl MorseLongRange {
    pub fn interaction(&self) -> morse_long_range::MorseLongRange {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        morse_long_range::MorseLongRangeBuilder {
            d0: converter.scalar_value(&self.d0),
            r_e: converter.scalar_value(&self.r_e),
            tail: self.tail.interaction().components,
            p: self.p,
            q: self.q,
            r_ref: self.r_ref.as_ref().map(|r_ref| converter.scalar_value(r_ref)),
            rho: self.rho.as_ref().map(|rho| converter.scalar_value(rho)),
            betas: self.betas.clone(),
        }.build()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spline(PathBuf, Option<usize>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RKHSInterpolation(PathBuf);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Interactions {
    LenardJones(LenardJones),
    Analytic(Analytic),
    MorseLongRange(MorseLongRange),
    Spline(Spline),
    RKHSInterpolation(RKHSInterpolation),
}

impl Interactions {
    pub fn interactions(&self) -> DynInteraction {
        match self {
            Interactions::LenardJones(lenard_jones) => DynInteraction::new(lenard_jones.interaction()),
            Interactions::Analytic(analytic) => DynInteraction::new(analytic.interaction()),
            Interactions::MorseLongRange(morse_long_range) => DynInteraction::new(morse_long_range.interaction()),
            Interactions::Spline(_spline) => todo!(),
            Interactions::RKHSInterpolation(_rkhs_interpolation) => todo!(),
        }
    }
}
