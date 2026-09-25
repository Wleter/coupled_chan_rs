pub mod pes;

use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    marker::PhantomData,
    path::PathBuf,
};

use cc_derive::Parameters;
use cc_qol_utils::Composite;
use coupled_chan::{
    DynInteraction, NullInteraction, dispersion::{
        AnalyticInteraction,
        ExpLaw,
        PowerLaw,
        lenard_jones,
    }, interpolated::{
        InterpolatedPotential,
        Transitioned,
        sin_transition,
        spline_interpolation::SplineBuilder,
    }, morse_long_range, scaled::Scaled
};
use hilbert_space::space::BasisElementsRef;
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::{
    Number,
};
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
    calculations::{
        Modified,
        modifications::{
            ModificationAction,
            ModifyParam,
        },
    },
    param_ids,
    parameters::{
        ParameterRegistry,
        TypedParamId,
    },
    problems::Problem,
    system::{
        ParamIds,
        ParamModifications,
        PotentialSpec,
    },
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub struct C_N {
    pub d: Scalar<Energy>,
    pub n: i8,
    pub r_e: Scalar<Length>,
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
pub struct Exp {
    d: Scalar<Energy>,
    exponent: Scalar<Inv<Length>>,
    r_offset: Scalar<Length>,
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
    pub exp_law: Vec<Exp>,
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
pub struct LenardJones {
    pub d_e: Scalar<Energy>,
    pub r_e: Scalar<Length>,
}

impl LenardJones {
    pub fn interaction(&self) -> Composite<PowerLaw> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        lenard_jones(converter.scalar_value(&self.d_e), converter.scalar_value(&self.r_e))
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwitchingRegion {
    SinTransition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interactions {
    Null,
    LenardJones(LenardJones),
    Analytic(Analytic),
    MorseLongRange(MorseLongRange),
    Spline(Spline),
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
            Interactions::Null => DynInteraction::new(NullInteraction),
            Interactions::LenardJones(lenard_jones) => DynInteraction::new(lenard_jones.interaction()),
            Interactions::Analytic(analytic) => DynInteraction::new(analytic.interaction()),
            Interactions::MorseLongRange(morse_long_range) => DynInteraction::new(morse_long_range.interaction()),
            Interactions::Spline(spline) => DynInteraction::new(spline.interaction()),
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
            Interactions::Composite(items) => {
                DynInteraction::new(Composite::new(items.iter().map(|x| x.interactions()).collect()))
            }
        }
    }

    pub fn scaled_interactions(&self, scaling: PecScaling) -> DynInteraction {
        match self {
            Interactions::Null => DynInteraction::new(NullInteraction),
            Interactions::LenardJones(lenard_jones) => {
                let mut lenard_jones = lenard_jones.clone();
                lenard_jones.d_e.scale(scaling.short_range.0 * scaling.full.0);
                // make C6 = 2.0 * d_e * r_e^6 independent of short_range
                // and scaling linearly with long_range or full scaling
                lenard_jones.r_e.scale((scaling.long_range.0 / scaling.short_range.0).powf(1. / 6.));

                DynInteraction::new(lenard_jones.interaction())
            },
            Interactions::MorseLongRange(morse_long_range) => {
                let mut mlr = morse_long_range.clone();
                mlr.d0.scale(scaling.short_range.0 * scaling.full.0);
                for t in mlr.tail.iter_mut() {
                    t.d.scale(scaling.long_range.0 * scaling.full.0);
                }
                DynInteraction::new(mlr.interaction())
            },
            Interactions::Transition { near, far, r_start_switch, r_end_switch, switching } => {
                let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
                let r_start = converter.scalar_value(r_start_switch);
                let r_end = converter.scalar_value(r_end_switch);

                match switching {
                    SwitchingRegion::SinTransition => DynInteraction::new(Transitioned::new(
                        near.scaled_interactions(PecScaling::new(scaling.short_range.0 * scaling.full.0, PecScalingType::Full)),
                        far.scaled_interactions(PecScaling::new(scaling.long_range.0 * scaling.full.0, PecScalingType::Full)),
                        sin_transition(r_start, r_end),
                    )),
                }
            },
            Interactions::Composite(items) => {
                DynInteraction::new(Composite::new(items.iter().map(|x| x.scaled_interactions(scaling)).collect()))
            },
            Interactions::Analytic(analytic) => {
                assert_ne!(scaling.long_range, ScalingValue::one(), "Analytic interaction does not support short range scaling");
                assert_ne!(scaling.short_range, ScalingValue::one(), "Analytic interaction does not support long range scaling");

                if scaling.full == ScalingValue::one() {
                    DynInteraction::new(analytic.interaction())
                } else {
                    DynInteraction::new(Scaled { interaction: analytic.interaction(), scaling: scaling.full.0 })
                }
            },
            Interactions::Spline(spline) => {
                assert_ne!(scaling.long_range, ScalingValue::one(), "Spline interpolated interaction does not support short range scaling");
                assert_ne!(scaling.short_range, ScalingValue::one(), "Spline interpolated interaction does not support long range scaling");

                if scaling.full == ScalingValue::one() {
                    DynInteraction::new(spline.interaction())
                } else {
                    DynInteraction::new(Scaled { interaction: spline.interaction(), scaling: scaling.full.0 })
                }
            },
        }
    }
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PecScalingType {
    #[default]
    Full,
    ShortRange,
    LongRange,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PecScaling {
    short_range: ScalingValue,
    long_range: ScalingValue,
    full: ScalingValue
}

impl PecScaling {
    pub fn new(value: f64, scaling_type: PecScalingType) -> Self {
        match scaling_type {
            PecScalingType::Full => Self {
                full: ScalingValue(value),
                ..Default::default()
            },
            PecScalingType::ShortRange => Self {
                short_range: ScalingValue(value),
                ..Default::default()
            },
            PecScalingType::LongRange => Self {
                long_range: ScalingValue(value),
                ..Default::default()
            },
        }
    }

    pub fn modify(&mut self, value: f64, scaling_type: PecScalingType) {
        match scaling_type {
            PecScalingType::Full => self.full.0 = value,
            PecScalingType::ShortRange => self.short_range.0 = value,
            PecScalingType::LongRange => self.long_range.0 = value,
        }
    }

    pub fn prod_mut(&mut self, other: Self) {
        self.full.0 *= other.full.0;
        self.short_range.0 *= other.short_range.0;
        self.long_range.0 *= other.long_range.0;
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScalingValue(pub f64);

impl ScalingValue {
    pub fn one() -> Self {
        Self::default()
    }
}

impl Default for ScalingValue {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PecPolarizations(pub HashMap<SpinConfiguration, Interactions>);

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PecPolarizationScalings(pub HashMap<SpinConfiguration, PecScaling>);

pub struct PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator,
{
    pub s_tot: HalfU32,
    pub pecs: TypedParamId<PecPolarizations>,
    pub scalings: TypedParamId<PecPolarizationScalings>,
    pub masking: Mask,
}

impl<Mask> PotentialSpec for PecPolarizationSpec<Mask>
where
    Mask: Fn(BasisElementsRef) -> Operator + Send + Sync,
{
    fn build_params(&self) -> ParamIds {
        param_ids![]
    }

    fn coupling_masking(&self, elements: BasisElementsRef, _params: &ParameterRegistry) -> Operator {
        (self.masking)(elements)
    }

    fn curve_params(&self) -> ParamIds {
        param_ids![self.pecs.vanish(), self.scalings.vanish()]
    }

    fn curve(&self, params: &ParameterRegistry) -> DynInteraction {
        let interaction = params
            .get(self.pecs)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .unwrap_or_else(|| panic!("input does not have PEC for S_tot = {}", self.s_tot));

        let scaling = params
            .get(self.scalings)
            .0
            .get(&SpinConfiguration::Spin(self.s_tot))
            .copied()
            .unwrap_or_default();

        interaction.scaled_interactions(scaling)
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

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PecPolarizationScaling {
    scaling: f64,
    #[serde(default)]
    scaling_type: PecScalingType,
    #[serde(default)]
    configuration: Option<SpinConfiguration>,
}

pub struct PecPolarizationScalingMod<P: Problem, C> {
    scaling: PecPolarizationScaling,
    id_pec: TypedParamId<PecPolarizations>,
    id_scalings: TypedParamId<PecPolarizationScalings>,
    phantom: PhantomData<(P, C)>,
}

impl<P: Problem, C> PecPolarizationScalingMod<P, C> {
    pub fn new(
        scaling: PecPolarizationScaling,
        id_pec: TypedParamId<PecPolarizations>,
        id_scalings: TypedParamId<PecPolarizationScalings>,
    ) -> Self {
        Self {
            scaling,
            id_pec,
            id_scalings,
            phantom: PhantomData,
        }
    }
}

impl<P: Problem, C: Send + Sync> ModifyParam for PecPolarizationScalingMod<P, C> {
    type P = P;
    type C = C;

    fn prep_modify(&self, _modified: &mut Modified<Self::P, Self::C>) -> ModificationAction {
        let id_pec = self.id_pec;
        let id_scalings = self.id_scalings;
        let scaling = self.scaling;

        ModificationAction::ParamModify(
            ParamModifications::new(move |r| {
                if let Some(c) = scaling.configuration {
                    r.get_mut(id_scalings).0.insert(c, PecScaling::new(scaling.scaling, scaling.scaling_type));
                    let s = r.get_mut(id_scalings).0.get_mut(&c).expect("Nonexistent");
                    s.modify(scaling.scaling, scaling.scaling_type);
                } else {
                    for k in r.get(id_pec).0.keys().copied().collect::<Vec<SpinConfiguration>>() {
                        r.get_mut(id_scalings).0.insert(k, PecScaling::new(scaling.scaling, scaling.scaling_type));
                    }
                }

                param_ids![id_scalings.vanish()]
            })
            .into_dyn(),
        )
    }

    fn as_number(&self) -> serde_json::Number {
        Number::from_f64(self.scaling.scaling).unwrap()
    }

    fn mut_number(&mut self, number: serde_json::Number) {
        self.scaling.scaling = number.as_f64().unwrap()
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
