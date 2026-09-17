use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    path::PathBuf,
};

use cc_derive::Parameters;
use cc_qol_utils::Composite;
use coupled_chan::{
    DynInteraction,
    coupling::masked::Masked,
    dispersion::{
        AnalyticInteraction, ExpLaw, PowerLaw, lennard_jones
    },
    interpolated::{
        InterpolatedPotential,
        Transitioned,
        sin_transition,
        spline_interpolation::SplineBuilder,
    },
    morse_long_range,
};
use hilbert_space::space::BasisElementsRef;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;
use spin_algebra::{
    half_integer::HalfU32,
    hu32,
};
use unit_systems::quantities::{
    Inv,
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
pub struct C_N {
    pub d: Scalar<Energy>,
    pub n: i8,
    pub r_e: Scalar<Length>
}

impl C_N {
    pub fn interaction(&self) -> PowerLaw {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let d = converter.scalar_value(&self.d);
        let r_e = converter.scalar_value(&self.r_e);
        let n = self.n as i32;

        PowerLaw::new(d * r_e.powi(n), -n)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub struct Exp {
    d: Scalar<Energy>,
    exponent: Scalar<Inv<Length>>,
    r_offset: Scalar<Length>
}

impl Exp {
    pub fn interaction(&self) -> ExpLaw {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let d = converter.scalar_value(&self.d);
        let exponent = converter.scalar_value(&self.exponent);
        let offset = converter.scalar_value(&self.r_offset);

        ExpLaw::new(d, exponent, offset)
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, Parameters)]
#[serde(default)]
pub struct Analytic {
    pub power_law: Vec<C_N>,
    pub exp_law: Vec<Exp>
}

impl Analytic {
    pub fn interaction(&self) -> AnalyticInteraction {
        AnalyticInteraction { 
            power_law: Composite::new(self.power_law.iter().map(|x| x.interaction()).collect()), 
            exponent_law: Composite::new(self.exp_law.iter().map(|x| x.interaction()).collect()), 
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LennardJones {
    pub d_e: Scalar<Energy>,
    pub r_e: Scalar<Length>,
}

impl LennardJones {
    pub fn interaction(&self) -> Composite<PowerLaw> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        lennard_jones(converter.scalar_value(&self.d_e), converter.scalar_value(&self.r_e))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorseLongRange {
    pub d0: Scalar<Energy>,
    pub r_e: Scalar<Length>,
    pub tail: Vec<C_N>,

    pub p: Option<i32>,
    pub q: Option<i32>,

    pub r_ref: Option<Scalar<Length>>,
    pub rho: Option<Scalar<Inv<Length>>>,
    pub betas: Vec<f64>,
}

impl MorseLongRange {
    pub fn interaction(&self) -> morse_long_range::MorseLongRange {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        morse_long_range::MorseLongRangeBuilder {
            d0: converter.scalar_value(&self.d0),
            r_e: converter.scalar_value(&self.r_e),
            tail: self.tail.iter().map(|x| x.interaction()).collect(),
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
pub struct Spline(PathBuf, #[serde(default)] Option<u32>);

#[derive(Deserialize)]
pub struct PotentialData {
    distance_units: Box<str>,
    value_units: Box<str>,
    distances: Vec<f64>,
    values: Vec<f64>,
}

impl Spline {
    pub fn interaction(&self) -> InterpolatedPotential {
        let file = File::open(&self.0).expect("Could not open potential data");
        let data: PotentialData = serde_json::from_reader(file).expect("Could not parse potential data");

        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let distance_conversion = converter.scalar_value(&Scalar::new(1.0, Length, &data.distance_units));
        let value_conversion = converter.scalar_value(&Scalar::new(1.0, Energy, &data.value_units));
        let distances: Vec<f64> = data.distances.into_iter().map(|x| x * distance_conversion).collect();
        let values: Vec<f64> = data.values.into_iter().map(|x| x * value_conversion).collect();

        let mut builder = SplineBuilder::new(&distances, &values);
        if let Some(k) = self.1 {
            builder = builder.with_degree(k)
        }

        InterpolatedPotential(builder.build())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RKHSInterpolation(PathBuf);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwitchingRegion {
    SinTransition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interactions {
    LennardJones(LennardJones),
    Analytic(Analytic),
    MorseLongRange(MorseLongRange),
    Spline(Spline),
    RkhsInterpolation(RKHSInterpolation),
    Exp(Exp),
    Transition {
        near: Box<Interactions>,
        far: Box<Interactions>,
        r_start_switch: Scalar<Length>,
        r_end_switch: Scalar<Length>,
        switching: SwitchingRegion,
    },
    Composite(Vec<Interactions>),
}

impl Interactions {
    pub fn interactions(&self) -> DynInteraction {
        match self {
            Interactions::LennardJones(lenard_jones) => DynInteraction::new(lenard_jones.interaction()),
            Interactions::Analytic(analytic) => DynInteraction::new(analytic.interaction()),
            Interactions::Exp(exp) => DynInteraction::new(exp.interaction()),
            Interactions::MorseLongRange(morse_long_range) => DynInteraction::new(morse_long_range.interaction()),
            Interactions::Spline(spline) => DynInteraction::new(spline.interaction()),
            Interactions::RkhsInterpolation(_rkhs_interpolation) => todo!(),
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
            Interactions::Composite(items) => DynInteraction::new(Composite::new(
                items.into_iter().map(|x| x.interactions()).collect())
            ),
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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PecScalings(pub HashMap<SpinConfiguration, Scaling>);

impl PecScalings {
    pub fn scale(&mut self, spin: SpinConfiguration, value: Value) -> bool {
        let value: f64 = serde_json::from_value(value).expect("Expecting pec scaling to be of type f64");
        if let Some(s) = self.0.get_mut(&spin)
            && s.0 == value
        {
            false
        } else {
            self.0.insert(spin, Scaling(value));
            true
        }
    }

    pub fn scale_all(&mut self, value: Value) -> bool {
        let value: f64 = serde_json::from_value(value).expect("Expecting pec scaling to be of type f64");
        let mut changed = false;
        for s in self.0.values_mut() {
            if s.0 != value {
                s.0 = value;
                changed = true;
            }
        }

        changed
    }
}

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

impl SpinConfiguration {
    pub fn get_configurations(max_spin: HalfU32) -> Vec<Self> {
        let n = max_spin.double_value() as usize + 1;

        use SpinConfiguration::*;
        let named = [
            Singlet, Doublet, Triplet, Quartet, Quintet, Sextet, Septet, Octet, Nonet, Decet,
        ];

        if n <= 10 {
            named[0..n].into()
        } else {
            let mut values = Vec::from(named);
            values.extend((10..n).map(|i| Spin(HalfU32::from_doubled(i as u32))));

            values
        }
    }
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
