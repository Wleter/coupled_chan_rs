use std::f64::consts::FRAC_PI_4;

/// Scaled Airy functions, translated from the MOLSCAT SCAIRY routine.
///
/// ## For x >= 0:
///   - Ai(x) = red_ai exp(-zeta)
///   - Bi(x) = red_bi exp(zeta)
///   - Ai'(x) = red_ai_deriv exp(-zeta)
///   - Bi'(x) = red_bi_deriv exp(zeta)
///   - zeta = 2 / 3 x^(3/2)
///
/// ## For x < -5:
///   - Ai(x) = red_ai cos(zeta) + red_bi sin(zeta)
///   - Bi(x) = red_bi cos(zeta) - red_ai sin(zeta)
///   - Ai'(x) = red_ai_deriv cos(zeta) + red_bi_deriv sin(zeta)
///   - Bi'(x) = red_bi_deriv cos(zeta) - red_ai_deriv sin(zeta)
///   - zeta = 2 / 3 x^(3/2) + PI / 4
///
/// ## For -5 <= x < 0:
///   - Ai(x) = red_ai
///   - Bi(x) = red_bi
///   - Ai'(x) = red_ai_deriv
///   - Bi'(x) = red_bi_deriv
///   - zeta = 0
#[derive(Debug, Clone, Copy)]
pub struct AiryData {
    x: f64,
    red_ai: f64,
    red_ai_deriv: f64,
    red_bi: f64,
    red_bi_deriv: f64,

    zeta: f64,
}

impl AiryData {
    pub fn ai(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_ai * f64::exp(-self.zeta),
            x if x < -5. => self.red_ai * f64::cos(self.zeta) + self.red_bi * f64::sin(self.zeta),
            x if x < 0. => self.red_ai,
            _ => unreachable!(),
        }
    }

    pub fn bi(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_bi * f64::exp(self.zeta),
            x if x < -5. => self.red_bi * f64::cos(self.zeta) - self.red_ai * f64::sin(self.zeta),
            x if x < 0. => self.red_bi,
            _ => unreachable!(),
        }
    }

    pub fn ai_deriv(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_ai_deriv * f64::exp(-self.zeta),
            x if x < -5. => self.red_ai_deriv * f64::cos(self.zeta) + self.red_bi_deriv * f64::sin(self.zeta),
            x if x < 0. => self.red_ai_deriv,
            _ => unreachable!(),
        }
    }

    pub fn bi_deriv(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_bi_deriv * f64::exp(self.zeta),
            x if x < -5. => self.red_bi_deriv * f64::cos(self.zeta) - self.red_ai_deriv * f64::sin(self.zeta),
            x if x < 0. => self.red_bi_deriv,
            _ => unreachable!(),
        }
    }

    pub fn exponent(&self) -> f64 {
        self.zeta.abs()
    }

    /// Returns reduced moduli M(x) and phase theta(x) of airy functions defined as:
    ///
    /// ## For x < 0:
    ///   - ai(x) = M(x) cos(theta(x))
    ///   - bi(x) = M(x) sin(theta(x))
    /// ## For x >= 0:
    ///   - ai(x) = M(x) sinh(theta(x))
    ///   - bi(x) = M(x) cosh(theta(x))
    pub fn red_moduli_phase(&self) -> (f64, f64) {
        if self.x >= 0.0 {
            (
                f64::sqrt(self.red_bi * self.red_bi - self.red_ai * self.red_ai),
                f64::atanh(self.red_ai / self.red_bi),
            )
        } else if self.x < -5. {
            (
                f64::sqrt(self.red_ai * self.red_ai + self.red_bi * self.red_bi),
                f64::atan2(self.red_bi, self.red_ai) - self.zeta,
            )
        } else {
            (
                f64::sqrt(self.red_ai * self.red_ai + self.red_bi * self.red_bi),
                f64::atan2(self.red_bi, self.red_ai),
            )
        }
    }
    /// Returns reduced moduli N(x) and phase phi(x) of derivatives of airy functions defined as:
    ///
    /// ## For x < 0:
    ///   - ai(x) = N(x) cos(phi(x))
    ///   - bi(x) = N(x) sin(phi(x))
    /// ## For x >= 0:
    ///   - ai(x) = N(x) sinh(phi(x))
    ///   - bi(x) = N(x) cosh(phi(x))
    pub fn red_moduli_phase_deriv(&self) -> (f64, f64) {
        if self.x >= 0.0 {
            (
                f64::sqrt(self.red_bi_deriv * self.red_bi_deriv - self.red_ai_deriv * self.red_ai_deriv),
                f64::atanh(self.red_ai_deriv / self.red_bi_deriv),
            )
        } else if self.x < -5. {
            (
                f64::sqrt(self.red_ai_deriv * self.red_ai_deriv + self.red_bi_deriv * self.red_bi_deriv),
                f64::atan2(self.red_bi_deriv, self.red_ai_deriv) - self.zeta,
            )
        } else {
            (
                f64::sqrt(self.red_ai_deriv * self.red_ai_deriv + self.red_bi_deriv * self.red_bi_deriv),
                f64::atan2(self.red_bi_deriv, self.red_ai_deriv),
            )
        }
    }
}

const AI0: f64 = 3.550_280_538_878_172e-1;
const AI0_DERIV: f64 = -2.588_194_037_928_068e-1;
const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Returns Scaled Airy functions, translated from the MOLSCAT SCAIRY routine.
/// Returns [`AiryData`].
pub fn eval_airy(x: f64) -> AiryData {
    match x {
        x if x.abs() <= NEAR_THRESHOLD => airy_near(x),
        x if x >= 9.0 => airy_9_expansion(x),
        x if x >= 4.5 => airy_4_5_expansion(x),
        x if x > NEAR_THRESHOLD => airy_0_to_4_5_expansion(x),
        x if x >= -5.0 => airy_0_to_neg_5_expansion(x),
        x if x < -5.0 => airy_neg_5_expansion(x),
        _ => unreachable!(),
    }
}

const NEAR_THRESHOLD: f64 = 0.025;

fn airy_near(x: f64) -> AiryData {
    let x_sqr = x * x;
    let x_cube = x_sqr * x;

    let df = 1. + x_cube / 6. + x_cube * x_cube / 180.;
    let dg = x * (1. + x_cube / 12. + x_cube * x_cube / 504.);

    let red_ai = AI0 * df + AI0_DERIV * dg;
    let red_bi = SQRT_3 * (AI0 * df - AI0_DERIV * dg);

    let df = x_sqr / 2. + x_sqr * x_cube / 30.;
    let dg = 1. + x_cube / 3. + x_cube * x_cube / 72.;
    let red_ai_deriv = AI0 * df + AI0_DERIV * dg;
    let red_bi_deriv = SQRT_3 * (AI0 * df - AI0_DERIV * dg);

    if x > 0.0 {
        let zeta = 2. / 3. * x.sqrt() * x;
        let exp = zeta.exp();

        AiryData {
            x,
            red_ai: red_ai * exp,
            red_ai_deriv: red_ai_deriv * exp,
            red_bi: red_bi / exp,
            red_bi_deriv: red_bi_deriv / exp,
            zeta,
        }
    } else {
        AiryData {
            x,
            red_ai,
            red_ai_deriv,
            red_bi,
            red_bi_deriv,
            zeta: 0.0,
        }
    }
}

fn airy_9_expansion(x: f64) -> AiryData {
    let x_sqrt = x.sqrt();
    let x_root4 = x_sqrt.sqrt();
    let zeta = 2. / 3. * x_sqrt * x;
    let t = 36. / zeta - 1.;

    let red_ai = horner(t, RED_AI_9) / x_root4;
    let red_bi = horner(t, RED_BI_9) / x_root4;
    let red_ai_deriv = horner(t, RED_AI_DERIV_9) * x_root4;
    let red_bi_deriv = horner(t, RED_BI_DERIV_9) * x_root4;

    AiryData {
        x,
        red_ai,
        red_ai_deriv,
        red_bi,
        red_bi_deriv,
        zeta,
    }
}

fn airy_4_5_expansion(x: f64) -> AiryData {
    let zeta = 2. / 3. * x.sqrt() * x;
    let exp1x = (zeta - 2.5 * x).exp();
    let exp2x = (zeta - 2.625 * x).exp();
    let t = 4.0 * x / 9.0 - 3.0;

    let red_ai = horner(t, RED_AI_4_5) * exp1x;
    let red_bi = horner(t, RED_BI_4_5) / exp1x;
    let red_ai_deriv = horner(t, RED_AI_DERIV_4_5) * exp1x;
    let red_bi_deriv = horner(t, RED_BI_DERIV_4_5) / exp2x;

    AiryData {
        x,
        red_ai,
        red_ai_deriv,
        red_bi,
        red_bi_deriv,
        zeta,
    }
}

fn airy_0_to_4_5_expansion(x: f64) -> AiryData {
    let zeta = 2. / 3. * x.sqrt() * x;

    let exp1z = (zeta - 1.5 * x).exp();
    let exp2z = (zeta - 1.375 * x).exp();
    let t = 4.0 * x / 9.0 - 1.0;

    let red_ai = horner(t, RED_AI_POS) * exp1z;
    let red_bi = horner(t, RED_BI_POS) / exp2z;
    let red_ai_deriv = horner(t, RED_AI_DERIV_POS) * exp2z;
    let red_bi_deriv = horner(t, RED_BI_DERIV_POS) / exp1z;

    AiryData {
        x,
        red_ai,
        red_ai_deriv,
        red_bi,
        red_bi_deriv,
        zeta,
    }
}

fn airy_0_to_neg_5_expansion(x: f64) -> AiryData {
    let t = x / 5.0;
    let t = -t * t * t;
    let t = 2.0 * t - 1.0;

    let f = chebyshev_series(t, F_AI_NEG);
    let g = chebyshev_series(t, G_AI_NEG);

    let ai = f - g * x;
    let bi = SQRT_3 * (f + g * x);

    let f = chebyshev_series(t, F_DERIV_NEG);
    let g = chebyshev_series(t, G_DERIV_NEG);

    let ai_deriv = x * x * f - g;
    let bi_deriv = SQRT_3 * (x * x * f + g);

    AiryData {
        x,
        red_ai: ai,
        red_ai_deriv: ai_deriv,
        red_bi: bi,
        red_bi_deriv: bi_deriv,
        zeta: 0.0,
    }
}

fn airy_neg_5_expansion(x: f64) -> AiryData {
    let x_sqrt = (-x).sqrt();
    let x_root4 = -x_sqrt.sqrt();
    let zeta = 2. / 3. * (-x) * x_sqrt;

    let t = -250.0 / (x * x * x) - 1.0;

    let ai = horner(t, RED_AI_NEG) / zeta / x_root4;
    let bi = horner(t, RED_BI_NEG) / x_root4;
    let ai_deriv = horner(t, RED_AI_DERIV_NEG) * x_root4;
    let bi_deriv = horner(t, RED_BI_DERIV_NEG) / zeta * x_root4;

    let zeta = zeta + FRAC_PI_4;

    AiryData {
        x,
        red_ai: ai,
        red_ai_deriv: ai_deriv,
        red_bi: bi,
        red_bi_deriv: bi_deriv,
        zeta,
    }
}

#[inline]
fn horner(x: f64, c: &[f64]) -> f64 {
    c.iter().fold(0.0, |acc, &a| acc * x + a)
}

#[inline]
fn chebyshev_series(x: f64, c: &[f64]) -> f64 {
    debug_assert!(c.len() >= 2);

    let a0 = c[0];
    let b0 = 2. * x * a0 - c[1];
    let (a, b) = c[2..c.len() - 1]
        .iter()
        .fold((a0, b0), |(a, b), &ck| (b, 2.0 * x * b - a + ck));

    x * b - a + c[c.len() - 1]
}

const RED_AI_9: &[f64] = &[
    1.165_377_953_249_792e-15,
    -1.164_141_714_555_724_8e-14,
    1.254_206_555_084_019_2e-13,
    -1.558_604_141_003_406_6e-12,
    2.210_457_761_100_112_7e-11,
    -3.674_728_275_171_940_4e-10,
    7.448_308_653_966_066e-9,
    -1.957_435_593_263_805_9e-7,
    7.446_724_319_698_052e-6,
    -5.286_518_814_099_3e-4,
    2.815_584_895_850_063e-1,
];

const RED_BI_9: &[f64] = &[
    4.501_659_992_545_28e-15,
    1.562_320_183_745_024e-14,
    5.262_407_125_599_181e-14,
    2.978_148_988_566_187_7e-13,
    1.975_776_209_756_256_6e-12,
    1.536_789_441_107_427_2e-11,
    1.454_099_335_374_552_2e-10,
    1.715_473_269_723_800_8e-9,
    2.618_986_171_291_470_7e-8,
    5.494_979_934_918_33e-7,
    1.767_198_043_651_093_3e-5,
    1.122_121_099_358_741e-3,
    5.652_945_575_585_221e-1,
];

const RED_AI_DERIV_9: &[f64] = &[
    1.209_546_389_246_976e-15,
    -1.212_812_185_390_208e-14,
    1.313_037_237_249_642_3e-13,
    -1.641_527_817_545_336_8e-12,
    2.346_721_850_257_094_6e-11,
    -3.945_073_291_221_193_3e-10,
    8.131_250_054_209_103e-9,
    -2.197_363_659_323_565_4e-7,
    8.839_935_152_272_579e-6,
    -7.434_563_399_720_802e-4,
    -2.828_473_163_363_792e-1,
];

const RED_BI_DERIV_9: &[f64] = &[
    -4.591_704_370_294_784e-15,
    -1.598_409_605_121_228_8e-14,
    -5.412_588_633_407_846e-14,
    -3.074_145_895_072_611_7e-13,
    -2.048_666_167_705_226_5e-12,
    -1.603_214_159_156_908_8e-11,
    -1.529_220_738_614_883e-10,
    -1.824_456_394_886_953_3e-9,
    -2.832_508_905_888_065e-8,
    -6.111_303_776_390_126e-7,
    -2.078_421_479_636_785_6e-5,
    -1.563_500_176_638_582_5e-3,
    5.626_462_830_948_43e-1,
];

const RED_AI_4_5: &[f64] = &[
    9.690_819_604_153_946e-11,
    3.244_361_360_509_208e-10,
    -3.574_195_134_306_446_6e-9,
    -3.844_613_208_279_747e-9,
    8.881_166_990_859_492e-8,
    -6.261_051_743_747_175e-8,
    -1.690_510_510_042_981_2e-6,
    3.807_314_163_630_417_6e-6,
    2.438_405_291_130_577_8e-5,
    -9.743_796_326_736_547e-5,
    -2.453_242_544_379_32e-4,
    1.695_179_269_533_127_8e-3,
    1.196_384_335_402_252_2e-3,
    -2.152_555_945_903_574_7e-2,
    9.337_770_735_228_443e-3,
    1.987_161_592_577_968_7e-1,
    -2.540_018_588_820_577_3e-1,
    -1.271_487_751_978_781_7,
    2.520_463_761_683_948,
    5.049_872_714_233_871,
    -1.331_209_785_444_192_8e1,
    -9.349_038_465_503_812,
    3.103_308_129_502_578_4e1,
];

const RED_BI_4_5: &[f64] = &[
    3.792_109_357_445_939_4e-14,
    -4.163_466_350_401_946e-14,
    -3.631_106_818_865_889_3e-13,
    1.389_325_920_294_142e-12,
    -4.004_890_688_108_888_4e-12,
    1.390_195_018_349_517_1e-11,
    -4.508_771_822_372_415e-11,
    1.389_423_098_447_332_5e-10,
    -3.925_034_981_087_100_7e-10,
    1.201_250_051_617_569_2e-9,
    -3.142_345_506_778_255_2e-9,
    1.031_005_873_236_947_6e-8,
    -2.352_400_607_831_267_7e-8,
    8.985_256_709_586_112e-8,
    -1.572_730_111_812_420_4e-7,
    7.776_967_632_897_388e-7,
    -8.402_111_811_881_353e-7,
    6.348_873_613_018_646e-6,
    -2.734_640_232_890_557_6e-6,
    4.546_067_299_251_662e-5,
    2.204_591_550_429_470_8e-6,
    2.588_233_889_575_881e-4,
    7.310_237_683_894_665e-5,
    1.010_138_069_045_963_7e-3,
    2.647_944_163_321_187_4e-4,
    1.974_997_855_537_091_3e-3,
];

const RED_AI_DERIV_4_5: &[f64] = &[
    -4.406_799_184_374_928_7e-10,
    1.309_549_454_493_483e-10,
    1.300_520_793_765_967_5e-8,
    -2.213_158_279_454_370_7e-8,
    -2.568_509_093_806_449_5e-7,
    8.669_608_553_656_983e-7,
    3.756_223_074_997_419e-6,
    -2.153_962_333_611_072_3e-5,
    -3.558_040_946_675_971e-5,
    3.953_178_529_140_377e-4,
    5.033_693_619_869_341e-5,
    -5.546_344_174_034_368_4e-3,
    5.296_581_869_083_728_5e-3,
    5.913_116_235_376_582e-2,
    -1.094_466_645_962_865_6e-1,
    -4.635_894_355_291_942_4e-1,
    1.253_232_698_220_309_7,
    2.501_381_089_594_692_4,
    -9.126_687_741_939_955,
    -8.143_857_320_368_765,
    4.001_340_825_508_33e1,
    1.153_962_029_314_448e1,
    -8.173_783_144_445_504e1,
];

const RED_BI_DERIV_4_5: &[f64] = &[
    -1.129_763_794_814_238_7e-13,
    2.841_632_751_998_73e-13,
    9.213_678_596_181_197e-14,
    -6.474_651_169_330_299e-13,
    5.662_104_421_589_319e-13,
    -3.031_580_424_589_017e-12,
    1.326_402_178_098_764_3e-11,
    -3.035_582_230_416_392_5e-11,
    5.322_904_070_735_659e-11,
    1.675_616_909_055_449_6e-11,
    -3.352_342_763_659_180_3e-10,
    2.928_077_730_200_504e-9,
    -8.769_009_941_274_644e-9,
    4.691_380_293_210_039e-8,
    -1.009_299_179_428_767_8e-7,
    5.404_019_346_486_879e-7,
    -8.199_771_292_584_569e-7,
    5.133_676_514_389_746e-6,
    -4.778_006_177_259_227e-6,
    4.024_153_911_178_971e-5,
    -1.745_711_929_122_744_2e-5,
    2.453_320_916_452_152e-4,
    -2.229_163_830_503_74e-5,
    1.025_359_935_497_379_4e-3,
    5.940_332_876_583_01e-5,
    2.174_206_275_393_456_4e-3,
];

const RED_AI_POS: &[f64] = &[
    4.976_358_549_090_205_5e-12,
    -3.250_241_502_739_169_4e-11,
    -5.157_739_467_230_727_6e-11,
    8.668_028_721_600_177e-10,
    -9.512_926_715_198_03e-10,
    -1.332_681_339_246_771e-8,
    4.370_614_061_441_796e-8,
    1.189_437_140_863_083_6e-7,
    -8.669_804_822_445_893e-7,
    -2.467_680_774_949_055e-8,
    1.106_109_398_304_836_2e-5,
    -1.804_756_635_355_164_7e-5,
    -9.222_135_189_891_923e-5,
    3.157_677_126_654_07e-4,
    4.086_264_194_128_51e-4,
    -3.127_042_699_243_407_5e-3,
    6.278_992_441_186_079e-4,
    1.990_621_424_782_29e-2,
    -2.274_270_582_113_221_3e-2,
    -7.948_696_981_362_782e-2,
    1.542_619_991_582_474_5e-1,
    1.756_184_631_287_307_5e-1,
    -5.052_236_706_541_698e-1,
    -1.496_959_024_160_503_2e-1,
    6.912_904_544_398_29e-1,
];

const RED_BI_POS: &[f64] = &[
    -8.011_446_099_079_122e-11,
    2.675_662_080_802_910_3e-10,
    1.744_169_714_069_715e-10,
    -3.126_421_646_668_001e-9,
    1.221_145_690_595_700_5e-8,
    -2.936_477_302_188_788e-8,
    1.769_519_947_858_308_3e-8,
    2.131_432_669_321_238_4e-7,
    -1.155_696_036_022_673e-6,
    3.343_940_657_529_499e-6,
    -5.201_434_922_532_595e-6,
    -3.219_378_900_298_301_4e-6,
    5.003_605_930_646_434e-5,
    -1.774_494_084_341_949e-4,
    3.863_573_899_671_506e-4,
    -4.533_379_221_656_229_4e-4,
    -2.608_663_787_748_831_5e-4,
    3.013_555_853_500_495e-3,
    -8.390_470_773_091_99e-3,
    1.632_402_676_279_661e-2,
    -1.908_307_270_841_125e-2,
    1.655_926_613_879_591_5e-2,
    1.761_018_030_141_848_7e-2,
    -3.366_520_194_725_265e-2,
    1.238_312_588_869_163_3e-1,
    -6.483_423_303_630_176e-2,
    2.203_105_508_828_077_4e-1,
    -1.038_830_149_573_652_2e-2,
    2.068_576_113_424_603_3e-1,
];

const RED_AI_DERIV_POS: &[f64] = &[
    -2.316_358_258_865_157e-11,
    8.438_401_428_028_706e-11,
    3.680_280_652_712_037_6e-10,
    -2.610_432_328_257_549_3e-9,
    -4.651_108_719_302_158_3e-10,
    4.461_648_423_348_557_4e-8,
    -9.245_994_366_905_797e-8,
    -4.558_098_820_959_313_5e-7,
    2.210_245_018_048_344_5e-6,
    1.502_513_989_525_588e-6,
    -2.918_300_086_572_899e-5,
    3.513_911_009_649_824_7e-5,
    2.379_667_670_020_027_3e-4,
    -7.009_698_702_951_48e-4,
    -9.849_233_587_179_427e-4,
    6.689_353_217_406_018e-3,
    -1.663_982_867_401_120_8e-3,
    -3.836_186_548_653_905e-2,
    4.804_636_150_926_588e-2,
    1.283_597_910_764_664_5e-1,
    -2.802_671_558_467_141e-1,
    -2.060_498_153_580_040_7e-1,
    7.635_228_435_308_785e-1,
    6.476_998_929_778_223e-2,
    -8.329_407_374_096_259e-1,
];

const RED_BI_DERIV_POS: &[f64] = &[
    2.693_306_654_718_301_3e-10,
    -1.253_131_112_179_210_2e-9,
    1.450_575_875_086_194e-9,
    5.828_273_511_345_716e-9,
    -3.960_934_123_143_057e-8,
    1.373_465_213_675_211_4e-7,
    -2.789_275_945_181_213e-7,
    2.965_318_454_206_876_7e-8,
    2.277_349_818_880_440_6e-6,
    -1.022_959_028_885_359_9e-5,
    2.655_152_183_195_239_5e-5,
    -3.864_573_702_063_788e-5,
    -1.522_122_324_762_686_4e-5,
    2.847_652_258_036_906_6e-4,
    -9.657_980_462_529_145e-4,
    2.046_180_655_804_535e-3,
    -2.687_024_221_479_725_3e-3,
    8.368_390_396_100_907e-4,
    6.871_311_614_478_666e-3,
    -2.105_637_411_000_046_6e-2,
    4.132_901_316_225_171e-2,
    -5.033_103_945_117_754e-2,
    5.954_677_958_251_798e-2,
    -1.642_131_012_232_358_3e-2,
    5.025_360_064_770_207e-2,
    5.756_017_876_871_96e-2,
    1.332_200_316_510_760_2e-1,
    7.763_563_578_991_546e-2,
    2.112_133_241_760_491_6e-1,
];
const F_AI_NEG: &[f64] = &[
    1.635_864_920_25e-18,
    -1.149_373_682_830_25e-16,
    7.060_906_358_566_96e-15,
    -3.755_045_810_332_901e-13,
    1.708_749_758_076_624_4e-11,
    -6.562_735_990_132_918e-10,
    2.092_500_233_006_598_7e-8,
    -5.427_803_728_939_972e-7,
    1.116_557_634_724_684_7e-5,
    -1.761_932_150_809_126_6e-4,
    2.037_926_574_031_449_3e-3,
    -1.616_162_609_419_079_6e-2,
    7.873_696_950_590_188e-2,
    -1.880_903_202_189_157_2e-1,
    8.835_933_286_664_34e-2,
    9.463_304_395_658_582e-2,
    7.608_699_941_417_267e-2,
];

const G_AI_NEG: &[f64] = &[
    1.233_406_984_67e-19,
    -9.054_405_467_318e-18,
    5.830_523_483_771_46e-16,
    -3.262_530_732_733_058e-14,
    1.569_118_250_996_656_4e-12,
    -6.403_863_753_934_148e-11,
    2.184_145_572_027_330_4e-9,
    -6.111_278_350_334_019e-8,
    1.370_954_782_252_895_6e-6,
    -2.394_645_953_138_124_4e-5,
    3.133_062_569_752_993e-4,
    -2.909_533_805_902_076_6e-3,
    1.769_729_070_740_922_6e-2,
    -6.170_556_771_641_222_4e-2,
    9.524_728_333_672_139e-2,
    -4.323_816_942_234_849e-2,
    3.768_287_177_015_440_5e-2,
];

const F_DERIV_NEG: &[f64] = &[
    -2.513_084_367_43e-18,
    1.655_433_262_420_34e-16,
    -9.492_371_230_281_424e-15,
    4.687_952_604_557_881e-13,
    -1.969_428_958_427_299_6e-11,
    6.934_937_158_184_92e-10,
    -2.010_769_652_644_762e-8,
    4.696_557_358_962_321e-7,
    -8.595_270_331_212_026e-6,
    1.188_714_962_702_695_3e-4,
    -1.182_440_976_973_327e-3,
    7.876_452_021_481_852e-3,
    -3.141_743_726_723_964_7e-2,
    6.204_646_424_452_958e-2,
    -4.838_242_917_763_517_5e-2,
    2.648_084_601_234_867e-2,
];

const G_DERIV_NEG: &[f64] = &[
    5.893_827_780_694e-18,
    -4.048_118_108_879_71e-16,
    2.426_804_532_876_730_7e-14,
    -1.256_839_101_480_993e-12,
    5.556_077_450_695_673e-11,
    -2.066_833_763_045_770_6e-9,
    6.359_244_256_854_255e-8,
    -1.584_225_273_936_19e-6,
    3.110_071_191_129_936e-5,
    -4.641_894_377_872_714_4e-4,
    5.009_700_254_115_791e-3,
    -3.621_663_427_173_734_5e-2,
    1.531_146_716_419_535e-1,
    -2.692_708_077_406_672_5e-1,
    -9.618_436_611_491_528e-2,
    2.070_993_728_792_977_2e-1,
    9.799_438_878_745_478e-2,
];
const RED_BI_NEG: &[f64] = &[
    -4.500_717_728_088_064e-15,
    1.117_779_334_778_060_8e-14,
    -1.399_595_458_484_838_5e-14,
    4.931_101_878_703_206_7e-14,
    -2.021_933_070_345_907_2e-13,
    7.535_854_526_635_7e-13,
    -3.146_323_659_285_013e-12,
    1.523_514_500_249_529_6e-11,
    -8.758_015_722_335_07e-11,
    6.273_494_135_095_551e-10,
    -6.021_835_265_553_032_5e-9,
    8.700_435_367_882_353e-8,
    -2.329_350_440_509_841e-6,
    1.836_053_373_676_384_2e-4,
    -5.640_035_550_994_134e-1,
];

const RED_AI_NEG: &[f64] = &[
    -4.129_727_590_367_232e-15,
    8.365_124_655_513_6e-15,
    -2.059_450_817_740_8e-16,
    6.237_338_407_903_232e-15,
    -5.813_339_839_598_592e-14,
    1.528_935_660_952_883e-13,
    -4.110_647_880_263_332e-13,
    1.338_208_845_595_386_4e-12,
    -4.742_939_149_217_855e-12,
    1.848_680_212_286_050_4e-11,
    -8.156_867_694_766_732e-11,
    4.193_733_903_761_969_6e-10,
    -2.615_840_844_063_036e-9,
    2.100_214_545_393_647e-8,
    -2.378_477_702_105_093_5e-7,
    4.431_146_369_625_163_4e-6,
    -1.832_413_714_365_790_6e-4,
    3.899_189_768_110_265e-2,
];

const RED_AI_DERIV_NEG: &[f64] = &[
    -4.584_843_902_222_336e-15,
    1.139_692_216_157_388_7e-14,
    -1.431_603_282_500_608e-14,
    5.047_349_785_263_001_6e-14,
    -2.070_559_570_150_814_7e-13,
    7.730_435_206_940_045e-13,
    -3.234_545_819_603_57e-12,
    1.570_435_403_326_602_2e-11,
    -9.060_238_276_799_916e-11,
    6.523_036_139_170_504e-10,
    -6.309_939_987_562_82e-9,
    9.237_114_608_317_032e-8,
    -2.540_302_849_536_391_8e-6,
    2.174_483_857_814_484_2e-4,
    5.644_096_716_803_791e-1,
];

const RED_BI_DERIV_NEG: &[f64] = &[
    4.196_121_979_584_512e-15,
    -8.504_547_085_090_816e-15,
    2.314_213_411_225_6e-16,
    -6.396_831_045_574_656e-15,
    5.925_093_218_330_624e-14,
    -1.560_086_609_838_919_6e-13,
    4.201_068_078_133_319_7e-13,
    -1.369_268_963_397_555_3e-12,
    4.860_008_002_867_629e-12,
    -1.897_800_618_195_706_3e-11,
    8.393_147_019_701_22e-11,
    -4.328_438_148_022_657_4e-10,
    2.711_249_349_914_697_3e-9,
    -2.190_268_887_120_029_7e-8,
    2.505_043_951_960_835_6e-7,
    -4.752_454_343_374_721e-6,
    2.052_527_910_979_407_4e-4,
    -5.464_148_416_073_097_4e-2,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    pub fn test_airy() {
        let airy = eval_airy(0.0);
        assert_eq!(airy.ai(), AI0);
        assert_eq!(airy.ai_deriv(), AI0_DERIV);

        let airy = eval_airy(1e-3);
        assert_approx_eq!(airy.ai(), 0.35476923454317420, 1e-9);
        assert_approx_eq!(airy.bi(), 0.61537491590587969, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.2588192263650529, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.44828866496656955, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(1.0);
        assert_approx_eq!(airy.ai(), 0.13529241631288141, 1e-9);
        assert_approx_eq!(airy.bi(), 1.20742359495287125, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.1591474412967932, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.93243593339277563, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(5.0);
        assert_approx_eq!(airy.ai(), 0.00010834442813607, 1e-9);
        assert_approx_eq!(airy.bi(), 657.792044171171182, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.0002474138908684, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1435.81908021798251, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(10.0);
        assert_approx_eq!(airy.ai(), 1.10475325528986859e-10, 1e-9);
        assert_approx_eq!(airy.bi(), 4.55641153548225140e8, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -3.5206336767389236e-10, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.42923613448286577e9, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(10.0);
        assert_approx_eq!(airy.ai(), 1.10475325528986859e-10, 1e-9);
        assert_approx_eq!(airy.bi(), 4.55641153548225140e8, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -3.5206336767389236e-10, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.42923613448286577e9, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(100.0);
        assert_approx_eq!(airy.ai(), 2.63448215208818448e-291, 1e-9);
        assert_approx_eq!(airy.bi(), 6.04122399667020139e288, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -2.6351403616044099e-290, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 6.03971274531060290e289, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::sinh(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::cosh(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::sinh(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::cosh(phi), 1e-9);

        let airy = eval_airy(-1.0);
        assert_approx_eq!(airy.ai(), 0.53556088329235211, 1e-9);
        assert_approx_eq!(airy.bi(), 0.10399738949694461, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.0101605671166452, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.59237562642279235, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.red_ai, m * f64::cos(theta), 1e-9);
        assert_approx_eq!(airy.red_bi, m * f64::sin(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.red_ai_deriv, n * f64::cos(phi), 1e-9);
        assert_approx_eq!(airy.red_bi_deriv, n * f64::sin(phi), 1e-9);

        let airy = eval_airy(-10.0);
        assert_approx_eq!(airy.ai(), 0.04024123848644319, 1e-9);
        assert_approx_eq!(airy.bi(), -0.3146798296438386, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), 0.99626504413279005, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.11941411339990923, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.ai(), m * f64::cos(theta), 1e-9);
        assert_approx_eq!(airy.bi(), m * f64::sin(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.ai_deriv(), n * f64::cos(phi), 1e-9);
        assert_approx_eq!(airy.bi_deriv(), n * f64::sin(phi), 1e-9);

        let airy = eval_airy(-100.0);
        assert_approx_eq!(airy.ai(), 0.17675339323955287, 1e-9);
        assert_approx_eq!(airy.bi(), 0.02427388768016013, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.2422970316605838, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.76759489323406093, 1e-9);
        let (m, theta) = airy.red_moduli_phase();
        assert_approx_eq!(airy.ai(), m * f64::cos(theta), 1e-9);
        assert_approx_eq!(airy.bi(), m * f64::sin(theta), 1e-9);
        let (n, phi) = airy.red_moduli_phase_deriv();
        assert_approx_eq!(airy.ai_deriv(), n * f64::cos(phi), 1e-9);
        assert_approx_eq!(airy.bi_deriv(), n * f64::sin(phi), 1e-9);
    }
}
