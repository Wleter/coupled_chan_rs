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
/// 
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
            _ => unreachable!()
        }
    }

    pub fn bi(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_bi * f64::exp(self.zeta),
            x if x < -5. => self.red_bi * f64::cos(self.zeta) - self.red_ai * f64::sin(self.zeta),
            x if x < 0. => self.red_bi,
            _ => unreachable!()
        }
    }

    pub fn ai_deriv(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_ai_deriv * f64::exp(-self.zeta),
            x if x < -5. => self.red_ai_deriv * f64::cos(self.zeta) + self.red_bi_deriv * f64::sin(self.zeta),
            x if x < 0. => self.red_ai_deriv,
            _ => unreachable!()
        }
    }

    pub fn bi_deriv(&self) -> f64 {
        match self.x {
            x if x >= 0. => self.red_bi_deriv * f64::exp(self.zeta),
            x if x < -5. => self.red_bi_deriv * f64::cos(self.zeta) - self.red_ai_deriv * f64::sin(self.zeta),
            x if x < 0. => self.red_bi_deriv,
            _ => unreachable!()
        }
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
                f64::atanh(self.red_ai / self.red_bi)
            )
        } else if self.x < -5. {
            (
                f64::sqrt(self.red_ai * self.red_ai + self.red_bi * self.red_bi),
                f64::atan2(self.red_bi, self.red_ai) - self.zeta
            )
        } else {
            (
                f64::sqrt(self.red_ai * self.red_ai + self.red_bi * self.red_bi),
                f64::atan2(self.red_bi, self.red_ai)
            )
        }
    }
    /// Returns reduced moduli N(x) and phase phi(x) of derivatives of airy functions defined as:
    /// 
    /// ## For x < 0:
    ///   - ai(x) = M(x) cos(theta(x))
    ///   - bi(x) = M(x) sin(theta(x))
    /// ## For x >= 0:
    ///   - ai(x) = M(x) sinh(theta(x))
    ///   - bi(x) = M(x) cosh(theta(x))
    pub fn red_moduli_phase_deriv(&self) -> (f64, f64) {
        if self.x >= 0.0 {
            (
                f64::sqrt(self.red_bi_deriv * self.red_bi_deriv - self.red_ai_deriv * self.red_ai_deriv),
                f64::atanh(self.red_ai_deriv / self.red_bi_deriv)
            )
        } else if self.x < -5. {
            (
                f64::sqrt(self.red_ai_deriv * self.red_ai_deriv + self.red_bi_deriv * self.red_bi_deriv),
                f64::atan2(self.red_bi_deriv, self.red_ai_deriv) - self.zeta
            )
        } else {
            (
                f64::sqrt(self.red_ai_deriv * self.red_ai_deriv + self.red_bi_deriv * self.red_bi_deriv),
                f64::atan2(self.red_bi_deriv, self.red_ai_deriv)
            )
        }
    }
}

const AI0: f64 = 3.55028053887817239e-01;
const AI0_DERIV: f64 = -2.58819403792806798e-01;
const SQRT_3: f64 = 1.73205080756887729;

/// Returns Scaled Airy functions, translated from the MOLSCAT SCAIRY routine.
/// Returns [`AiryData`].
pub fn eval_airy(x: f64) -> AiryData {
    match x {
        x if x.abs() <= NEAR_THRESHOLD => airy_near(x),
        x if x >= 9.0 => airy_9_expansion(x),
        x if x >= 4.5 => airy_4_5_expansion(x),
        x if x > -NEAR_THRESHOLD => airy_0_to_4_5_expansion(x),
        x if x >= -5.0 => airy_0_to_neg_5_expansion(x),
        x if x < -5.0 => airy_neg_5_expansion(x),
        _ => unreachable!()
    }
}

const NEAR_THRESHOLD: f64 = 0.025;

pub fn airy_near(x: f64) -> AiryData {
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
            zeta
        }
    } else {
        AiryData { x, red_ai, red_ai_deriv, red_bi, red_bi_deriv, zeta: 0.0}
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

    AiryData { x, red_ai, red_ai_deriv, red_bi, red_bi_deriv, zeta }
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

    AiryData { x, red_ai, red_ai_deriv, red_bi, red_bi_deriv, zeta }
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

    AiryData { x, red_ai, red_ai_deriv, red_bi, red_bi_deriv, zeta }
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

    AiryData { x, red_ai: ai, red_ai_deriv: ai_deriv, red_bi: bi, red_bi_deriv: bi_deriv, zeta: 0.0 }
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

    AiryData { x, red_ai: ai, red_ai_deriv: ai_deriv, red_bi: bi, red_bi_deriv: bi_deriv, zeta }
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
        .fold((a0, b0), |(a, b), &ck| {
            (b, 2.0 * x * b - a + ck)
        });

    x * b - a + c[c.len() - 1]
}

const RED_AI_9: &[f64] = &[
    1.16537795324979200e-15,
    -1.16414171455572480e-14,
    1.25420655508401920e-13,
    -1.55860414100340659e-12,
    2.21045776110011276e-11,
    -3.67472827517194031e-10,
    7.44830865396606612e-09,
    -1.95743559326380581e-07,
    7.44672431969805149e-06,
    -5.28651881409929932e-04,
    2.81558489585006298e-01,
];

const RED_BI_9: &[f64] = &[
    4.50165999254528000e-15,
    1.56232018374502400e-14,
    5.26240712559918080e-14,
    2.97814898856618752e-13,
    1.97577620975625677e-12,
    1.53678944110742706e-11,
    1.45409933537455235e-10,
    1.71547326972380087e-09,
    2.61898617129147064e-08,
    5.49497993491833009e-07,
    1.76719804365109334e-05,
    1.12212109935874117e-03,
    5.65294557558522063e-01,
];

const RED_AI_DERIV_9: &[f64] = &[
    1.20954638924697600e-15,
    -1.21281218539020800e-14,
    1.31303723724964224e-13,
    -1.64152781754533677e-12,
    2.34672185025709461e-11,
    -3.94507329122119338e-10,
    8.13125005420910243e-09,
    -2.19736365932356533e-07,
    8.83993515227257822e-06,
    -7.43456339972080231e-04,
    -2.82847316336379200e-01,
];

const RED_BI_DERIV_9: &[f64] = &[
    -4.59170437029478400e-15,
    -1.59840960512122880e-14,
    -5.41258863340784640e-14,
    -3.07414589507261184e-13,
    -2.04866616770522650e-12,
    -1.60321415915690897e-11,
    -1.52922073861488292e-10,
    -1.82445639488695332e-09,
    -2.83250890588806503e-08,
    -6.11130377639012647e-07,
    -2.07842147963678572e-05,
    -1.56350017663858255e-03,
    5.62646283094843014e-01,
];

const RED_AI_4_5: &[f64] = &[
    9.69081960415394529e-11,
    3.24436136050920784e-10,
    -3.57419513430644674e-09,
    -3.84461320827974687e-09,
    8.88116699085949212e-08,
    -6.26105174374717557e-08,
    -1.69051051004298110e-06,
    3.80731416363041759e-06,
    2.43840529113057777e-05,
    -9.74379632673654766e-05,
    -2.45324254437931970e-04,
    1.69517926953312785e-03,
    1.19638433540225211e-03,
    -2.15255594590357451e-02,
    9.33777073522844198e-03,
    1.98716159257796883e-01,
    -2.54001858882057718e-01,
    -1.27148775197878180e+00,
    2.52046376168394778e+00,
    5.04987271423387057e+00,
    -1.33120978544419281e+01,
    -9.34903846550381088e+00,
    3.10330812950257837e+01,
];

const RED_BI_4_5: &[f64] = &[
    3.79210935744593920e-14,
    -4.16346635040194560e-14,
    -3.63110681886588928e-13,
    1.38932592029414195e-12,
    -4.00489068810888806e-12,
    1.39019501834951721e-11,
    -4.50877182237241508e-11,
    1.38942309844733264e-10,
    -3.92503498108710093e-10,
    1.20125005161756928e-09,
    -3.14234550677825531e-09,
    1.03100587323694771e-08,
    -2.35240060783126760e-08,
    8.98525670958611253e-08,
    -1.57273011181242048e-07,
    7.77696763289738864e-07,
    -8.40211181188135235e-07,
    6.34887361301864569e-06,
    -2.73464023289055762e-06,
    4.54606729925166230e-05,
    2.20459155042947089e-06,
    2.58823388957588056e-04,
    7.31023768389466446e-05,
    1.01013806904596356e-03,
    2.64794416332118755e-04,
    1.97499785553709145e-03,
];

const RED_AI_DERIV_4_5: &[f64] = &[
    -4.40679918437492851e-10,
    1.30954945449348301e-10,
    1.30052079376596751e-08,
    -2.21315827945437064e-08,
    -2.56850909380644963e-07,
    8.66960855365698346e-07,
    3.75622307499741911e-06,
    -2.15396233361107222e-05,
    -3.55804094667597110e-05,
    3.95317852914037711e-04,
    5.03369361986934094e-05,
    -5.54634417403436820e-03,
    5.29658186908372832e-03,
    5.91311623537658225e-02,
    -1.09446664596286554e-01,
    -4.63589435529194219e-01,
    1.25323269822030972e+00,
    2.50138108959469254e+00,
    -9.12668774193995449e+00,
    -8.14385732036876466e+00,
    4.00134082550833019e+01,
    1.15396202931444799e+01,
    -8.17378314444550419e+01,
];

const RED_BI_DERIV_4_5: &[f64] = &[
    -1.12976379481423872e-13,
    2.84163275199873024e-13,
    9.21367859618119680e-14,
    -6.47465116933029888e-13,
    5.66210442158931968e-13,
    -3.03158042458901709e-12,
    1.32640217809876419e-11,
    -3.03558223041639219e-11,
    5.32290407073565901e-11,
    1.67561690905544950e-11,
    -3.35234276365918044e-10,
    2.92807773020050397e-09,
    -8.76900994127464369e-09,
    4.69138029321003869e-08,
    -1.00929917942876779e-07,
    5.40401934648687824e-07,
    -8.19977129258456927e-07,
    5.13367651438974580e-06,
    -4.77800617725922708e-06,
    4.02415391117897098e-05,
    -1.74571192912274417e-05,
    2.45332091645215217e-04,
    -2.22916383050374016e-05,
    1.02535993549737948e-03,
    5.94033287658300975e-05,
    2.17420627539345627e-03,
];

const RED_AI_POS: &[f64] = &[
    4.97635854909020570e-12,
    -3.25024150273916928e-11,
    -5.15773946723072737e-11,
    8.66802872160017711e-10,
    -9.51292671519803048e-10,
    -1.33268133924677102e-08,
    4.37061406144179625e-08,
    1.18943714086308365e-07,
    -8.66980482244589319e-07,
    -2.46768077494905499e-08,
    1.10610939830483627e-05,
    -1.80475663535516462e-05,
    -9.22213518989192294e-05,
    3.15767712665407001e-04,
    4.08626419412850994e-04,
    -3.12704269924340764e-03,
    6.27899244118607949e-04,
    1.99062142478229001e-02,
    -2.27427058211322122e-02,
    -7.94869698136278246e-02,
    1.54261999158247445e-01,
    1.75618463128730757e-01,
    -5.05223670654169859e-01,
    -1.49695902416050331e-01,
    6.91290454439828966e-01,
];

const RED_BI_POS: &[f64] = &[
    -8.01144609907912212e-11,
    2.67566208080291037e-10,
    1.74416971406971503e-10,
    -3.12642164666800066e-09,
    1.22114569059570056e-08,
    -2.93647730218878800e-08,
    1.76951994785830839e-08,
    2.13143266932123830e-07,
    -1.15569603602267288e-06,
    3.34394065752949896e-06,
    -5.20143492253259528e-06,
    -3.21937890029830155e-06,
    5.00360593064643409e-05,
    -1.77449408434194908e-04,
    3.86357389967150628e-04,
    -4.53337922165622921e-04,
    -2.60866378774883161e-04,
    3.01355585350049504e-03,
    -8.39047077309199055e-03,
    1.63240267627966090e-02,
    -1.90830727084112485e-02,
    1.65592661387959142e-02,
    1.76101803014184860e-02,
    -3.36652019472526494e-02,
    1.23831258886916327e-01,
    -6.48342330363017516e-02,
    2.20310550882807725e-01,
    -1.03883014957365224e-02,
    2.06857611342460346e-01,
];

const RED_AI_DERIV_POS: &[f64] = &[
    -2.31635825886515692e-11,
    8.43840142802870600e-11,
    3.68028065271203758e-10,
    -2.61043232825754937e-09,
    -4.65110871930215858e-10,
    4.46164842334855713e-08,
    -9.24599436690579710e-08,
    -4.55809882095931368e-07,
    2.21024501804834447e-06,
    1.50251398952558802e-06,
    -2.91830008657289876e-05,
    3.51391100964982453e-05,
    2.37966767002002741e-04,
    -7.00969870295148024e-04,
    -9.84923358717942729e-04,
    6.68935321740601810e-03,
    -1.66398286740112083e-03,
    -3.83618654865390504e-02,
    4.80463615092658847e-02,
    1.28359791076466449e-01,
    -2.80267155846714091e-01,
    -2.06049815358004057e-01,
    7.63522843530878467e-01,
    6.47699892977822355e-02,
    -8.32940737409625965e-01,
];

const RED_BI_DERIV_POS: &[f64] = &[
    2.69330665471830131e-10,
    -1.25313111217921013e-09,
    1.45057587508619405e-09,
    5.82827351134571594e-09,
    -3.96093412314305685e-08,
    1.37346521367521144e-07,
    -2.78927594518121271e-07,
    2.96531845420687661e-08,
    2.27734981888044076e-06,
    -1.02295902888535994e-05,
    2.65515218319523965e-05,
    -3.86457370206378782e-05,
    -1.52212232476268640e-05,
    2.84765225803690646e-04,
    -9.65798046252914453e-04,
    2.04618065580453522e-03,
    -2.68702422147972510e-03,
    8.36839039610090712e-04,
    6.87131161447866570e-03,
    -2.10563741100004648e-02,
    4.13290131622517073e-02,
    -5.03310394511775398e-02,
    5.95467795825179773e-02,
    -1.64213101223235839e-02,
    5.02536006477020710e-02,
    5.75601787687195966e-02,
    1.33220031651076020e-01,
    7.76356357899154668e-02,
    2.11213324176049168e-01,
];
const F_AI_NEG: &[f64] = &[
    1.63586492025000000e-18,
    -1.14937368283025000e-16,
    7.06090635856696000e-15,
    -3.75504581033290114e-13,
    1.70874975807662448e-11,
    -6.56273599013291800e-10,
    2.09250023300659871e-08,
    -5.42780372893997236e-07,
    1.11655763472468469e-05,
    -1.76193215080912647e-04,
    2.03792657403144947e-03,
    -1.61616260941907957e-02,
    7.87369695059018748e-02,
    -1.88090320218915726e-01,
    8.83593328666433903e-02,
    9.46330439565858235e-02,
    7.60869994141726643e-02,
];

const G_AI_NEG: &[f64] = &[
    1.23340698467000000e-19,
    -9.05440546731800000e-18,
    5.83052348377146000e-16,
    -3.26253073273305810e-14,
    1.56911825099665634e-12,
    -6.40386375393414830e-11,
    2.18414557202733054e-09,
    -6.11127835033401880e-08,
    1.37095478225289560e-06,
    -2.39464595313812449e-05,
    3.13306256975299299e-04,
    -2.90953380590207648e-03,
    1.76972907074092250e-02,
    -6.17055677164122241e-02,
    9.52472833367213949e-02,
    -4.32381694223484894e-02,
    3.76828717701544063e-02,
];

const F_DERIV_NEG: &[f64] = &[
    -2.51308436743000000e-18,
    1.65543326242034000e-16,
    -9.49237123028142500e-15,
    4.68795260455788096e-13,
    -1.96942895842729954e-11,
    6.93493715818491929e-10,
    -2.01076965264476206e-08,
    4.69655735896232104e-07,
    -8.59527033121202608e-06,
    1.18871496270269531e-04,
    -1.18244097697332692e-03,
    7.87645202148185146e-03,
    -3.14174372672396468e-02,
    6.20464642445295805e-02,
    -4.83824291776351778e-02,
    2.64808460123486707e-02,
];

const G_DERIV_NEG: &[f64] = &[
    5.89382778069400000e-18,
    -4.04811810887971000e-16,
    2.42680453287673090e-14,
    -1.25683910148099294e-12,
    5.55607745069567295e-11,
    -2.06683376304577072e-09,
    6.35924425685425485e-08,
    -1.58422527393619013e-06,
    3.11007119112993551e-05,
    -4.64189437787271433e-04,
    5.00970025411579034e-03,
    -3.62166342717373453e-02,
    1.53114671641953510e-01,
    -2.69270807740667256e-01,
    -9.61843661149152853e-02,
    2.07099372879297732e-01,
    9.79943887874547828e-02,
];
const RED_BI_NEG: &[f64] = &[
    -4.50071772808806400e-15,
    1.11777933477806080e-14,
    -1.39959545848483840e-14,
    4.93110187870320640e-14,
    -2.02193307034590720e-13,
    7.53585452663569920e-13,
    -3.14632365928501299e-12,
    1.52351450024952975e-11,
    -8.75801572233507014e-11,
    6.27349413509555121e-10,
    -6.02183526555303242e-09,
    8.70043536788235270e-08,
    -2.32935044050984079e-06,
    1.83605337367638430e-04,
    -5.64003555099413391e-01,
];

const RED_AI_NEG: &[f64] = &[
    -4.12972759036723200e-15,
    8.36512465551360000e-15,
    -2.05945081774080000e-16,
    6.23733840790323200e-15,
    -5.81333983959859200e-14,
    1.52893566095288320e-13,
    -4.11064788026333184e-13,
    1.33820884559538637e-12,
    -4.74293914921785574e-12,
    1.84868021228605050e-11,
    -8.15686769476673166e-11,
    4.19373390376196942e-10,
    -2.61584084406303574e-09,
    2.10021454539364698e-08,
    -2.37847770210509358e-07,
    4.43114636962516363e-06,
    -1.83241371436579068e-04,
    3.89918976811026487e-02,
];

const RED_AI_DERIV_NEG: &[f64] = &[
    -4.58484390222233600e-15,
    1.13969221615738880e-14,
    -1.43160328250060800e-14,
    5.04734978526300160e-14,
    -2.07055957015081472e-13,
    7.73043520694004480e-13,
    -3.23454581960357018e-12,
    1.57043540332660220e-11,
    -9.06023827679991573e-11,
    6.52303613917050367e-10,
    -6.30993998756281944e-09,
    9.23711460831703303e-08,
    -2.54030284953639173e-06,
    2.17448385781448409e-04,
    5.64409671680379110e-01,
];

const RED_BI_DERIV_NEG: &[f64] = &[
    4.19612197958451200e-15,
    -8.50454708509081600e-15,
    2.31421341122560000e-16,
    -6.39683104557465600e-15,
    5.92509321833062400e-14,
    -1.56008660983891968e-13,
    4.20106807813331968e-13,
    -1.36926896339755520e-12,
    4.86000800286762854e-12,
    -1.89780061819570625e-11,
    8.39314701970122041e-11,
    -4.32843814802265754e-10,
    2.71124934991469715e-09,
    -2.19026888712002973e-08,
    2.50504395196083566e-07,
    -4.75245434337472120e-06,
    2.05252791097940732e-04,
    -5.46414841607309762e-02,
];

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use super::*;

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

        let airy = eval_airy(1.0);
        assert_approx_eq!(airy.ai(), 0.13529241631288141, 1e-9);
        assert_approx_eq!(airy.bi(), 1.20742359495287125, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.1591474412967932, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.93243593339277563, 1e-9);

        let airy = eval_airy(5.0);
        assert_approx_eq!(airy.ai(), 0.00010834442813607, 1e-9);
        assert_approx_eq!(airy.bi(), 657.792044171171182, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.0002474138908684, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1435.81908021798251, 1e-9);

        let airy = eval_airy(10.0);
        assert_approx_eq!(airy.ai(), 1.10475325528986859e-10, 1e-9);
        assert_approx_eq!(airy.bi(), 4.55641153548225140e8, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -3.5206336767389236e-10, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.42923613448286577e9, 1e-9);

        let airy = eval_airy(10.0);
        assert_approx_eq!(airy.ai(), 1.10475325528986859e-10, 1e-9);
        assert_approx_eq!(airy.bi(), 4.55641153548225140e8, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -3.5206336767389236e-10, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.42923613448286577e9, 1e-9);

        let airy = eval_airy(100.0);
        assert_approx_eq!(airy.ai(), 2.63448215208818448e-291, 1e-9);
        assert_approx_eq!(airy.bi(), 6.04122399667020139e288, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -2.6351403616044099e-290, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 6.03971274531060290e289, 1e-9);

        let airy = eval_airy(-1.0);
        assert_approx_eq!(airy.ai(), 0.53556088329235211, 1e-9);
        assert_approx_eq!(airy.bi(), 0.10399738949694461, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.0101605671166452, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.59237562642279235, 1e-9);

        let airy = eval_airy(-10.0);
        assert_approx_eq!(airy.ai(), 0.04024123848644319, 1e-9);
        assert_approx_eq!(airy.bi(), -0.3146798296438386, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), 0.99626504413279005, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 0.11941411339990923, 1e-9);

        let airy = eval_airy(-100.0);
        assert_approx_eq!(airy.ai(), 0.17675339323955287, 1e-9);
        assert_approx_eq!(airy.bi(), 0.02427388768016013, 1e-9);
        assert_approx_eq!(airy.ai_deriv(), -0.2422970316605838, 1e-9);
        assert_approx_eq!(airy.bi_deriv(), 1.76759489323406093, 1e-9);
    }


}