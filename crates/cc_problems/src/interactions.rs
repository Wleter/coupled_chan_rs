use std::{
    collections::HashMap,
    fmt::Display,
    path::PathBuf,
};

use cc_derive::Parameters;
use cc_qol_utils::Composite;
use coupled_chan::{
    DynInteraction,
    coupling::masked::Masked,
    dispersion::{
        PowerLaw,
        lennard_jones,
    },
    interpolated::{
        Transitioned,
        sin_transition,
    },
    morse_long_range,
};
use hilbert_space::space::BasisElementsRef;
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::{
    half_integer::HalfU32,
    hu32,
};
use unit_systems::quantities::{
    Inv,
    Power,
    Prod,
    Scalar,
    phys_quantities::{
        Energy,
        Length,
    },
};

use crate::{
    Operator,
    UNITS_CONVERTER,
    param_ids,
    parameters::{
        ParameterRegistry,
        TypedParamId,
    },
    system::{
        ParamIds,
        PotentialSpec,
    },
};

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
        }
        .build()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spline(PathBuf, Option<usize>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RKHSInterpolation(PathBuf);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SwitchingRegion {
    SinTransition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Interactions {
    LenardJones(LenardJones),
    Analytic(Analytic),
    MorseLongRange(MorseLongRange),
    Spline(Spline),
    RKHSInterpolation(RKHSInterpolation),
    Transition {
        near: Box<Interactions>,
        far: Box<Interactions>,
        r_start_switch: Scalar<Length>,
        r_end_switch: Scalar<Length>,
        switching: SwitchingRegion,
    },
}

impl Interactions {
    pub fn interactions(&self) -> DynInteraction {
        match self {
            Interactions::LenardJones(lenard_jones) => DynInteraction::new(lenard_jones.interaction()),
            Interactions::Analytic(analytic) => DynInteraction::new(analytic.interaction()),
            Interactions::MorseLongRange(morse_long_range) => DynInteraction::new(morse_long_range.interaction()),
            Interactions::Spline(_spline) => todo!(),
            Interactions::RKHSInterpolation(_rkhs_interpolation) => todo!(),
            Interactions::Transition {
                near,
                far,
                r_start_switch,
                r_end_switch,
                switching,
            } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let r_start = converter.scalar_value(r_start_switch);
                let r_end = converter.scalar_value(r_end_switch);

                match switching {
                    SwitchingRegion::SinTransition => DynInteraction::new(Transitioned::new(
                        near.interactions(),
                        far.interactions(),
                        sin_transition(r_start, r_end),
                    )),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Scaling(pub f64);

impl Default for Scaling {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PecPolarizations(pub HashMap<SpinConfiguration, Interactions>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PecScalings(pub HashMap<SpinConfiguration, Scaling>);

pub struct PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator,
{
    pub s_tot: HalfU32,
    pub pecs: TypedParamId<PecPolarizations>,
    pub scalings: TypedParamId<PecScalings>,
    pub masking: Mask,
}

impl<Mask> PotentialSpec for PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator + Send + Sync,
{
    fn build_params(&self) -> ParamIds {
        param_ids![self.pecs.vanish()]
    }

    fn r_coupling(&self, elements: BasisElementsRef, params: &ParameterRegistry) -> Masked<DynInteraction> {
        Masked {
            interaction: params
                .get(self.pecs)
                .0
                .get(&SpinConfiguration::Spin(self.s_tot))
                .unwrap_or_else(|| panic!("input does not have PEC for S_tot = {}", self.s_tot))
                .interactions(),
            masking: (self.masking)(elements).0,
        }
    }

    fn scaling_params(&self) -> ParamIds {
        param_ids![self.scalings.vanish()]
    }

    fn scaling(&self, params: &ParameterRegistry) -> f64 {
        params
            .get(self.scalings)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .copied()
            .unwrap_or_default()
            .0
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpinConfiguration {
    Spin(HalfU32),
    Singlet,
    Doublet,
    Triplet,
    Quartet,
    Quintet,
    Sextet,
    Septet,
    Octet,
    Nonet,
    Decet,
}

impl Display for SpinConfiguration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpinConfiguration::Spin(half_u32) => write!(f, "{half_u32}"),
            SpinConfiguration::Singlet => write!(f, "singlet"),
            SpinConfiguration::Doublet => write!(f, "doublet"),
            SpinConfiguration::Triplet => write!(f, "triplet"),
            SpinConfiguration::Quartet => write!(f, "quartet"),
            SpinConfiguration::Quintet => write!(f, "quintet"),
            SpinConfiguration::Sextet => write!(f, "sextet"),
            SpinConfiguration::Septet => write!(f, "septet"),
            SpinConfiguration::Octet => write!(f, "octet"),
            SpinConfiguration::Nonet => write!(f, "nonet"),
            SpinConfiguration::Decet => write!(f, "decet"),
        }
        
    }
}

impl SpinConfiguration {
    pub fn as_spin(&self) -> HalfU32 {
        match self {
            SpinConfiguration::Spin(half_u32) => *half_u32,
            SpinConfiguration::Singlet => hu32!(0),
            SpinConfiguration::Doublet => hu32!(1 / 2),
            SpinConfiguration::Triplet => hu32!(1),
            SpinConfiguration::Quartet => hu32!(3 / 2),
            SpinConfiguration::Quintet => hu32!(2),
            SpinConfiguration::Sextet => hu32!(5 / 2),
            SpinConfiguration::Septet => hu32!(3),
            SpinConfiguration::Octet => hu32!(7 / 2),
            SpinConfiguration::Nonet => hu32!(4),
            SpinConfiguration::Decet => hu32!(9 / 2),
        }
    }
}

impl PartialEq for SpinConfiguration {
    fn eq(&self, other: &Self) -> bool {
        self.as_spin() == other.as_spin()
    }
}

impl std::hash::Hash for SpinConfiguration {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_spin().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_configuration_collection() {
        let hash = HashMap::from([
            (SpinConfiguration::Singlet, 0.0f64),
            (SpinConfiguration::Spin(hu32!(1)), 1.0f64),
        ]);

        assert_eq!(SpinConfiguration::Doublet, SpinConfiguration::Spin(hu32!(1 / 2)));

        assert_eq!(hash[&SpinConfiguration::Triplet], 1.0);
        assert_eq!(hash[&SpinConfiguration::Spin(hu32!(0))], 0.0);
    }
}
