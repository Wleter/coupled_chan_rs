use std::f64::consts::PI;

use crate::*;

#[inline]
pub fn s_squared(spin: Braket<impl SpinMagLike>) -> f64 {
    if spin.bra == spin.ket { spin.bra.squared() } else { 0.0 }
}

#[inline]
pub fn proj_z(spin: Braket<impl SpinLike>) -> f64 {
    if spin.bra == spin.ket { spin.bra.m().value() } else { 0.0 }
}

#[inline]
pub fn ladder_plus(spin: Braket<impl SpinLike>) -> f64 {
    if spin.bra.s() == spin.ket.s() && spin.bra.m().double_value() == spin.ket.m().double_value() + 2 {
        (spin.ket.s().value() * (spin.ket.s().value() + 1.) - spin.bra.m().value() * spin.ket.m().value()).sqrt()
    } else {
        0.0
    }
}

#[inline]
pub fn ladder_minus(spin: Braket<impl SpinLike>) -> f64 {
    if spin.bra.s() == spin.ket.s() && spin.bra.m().double_value() + 2 == spin.ket.m().double_value() {
        (spin.ket.s().value() * (spin.ket.s().value() + 1.) - spin.bra.m().value() * spin.ket.m().value()).sqrt()
    } else {
        0.0
    }
}

#[inline]
pub fn dot(spin1: Braket<impl SpinLike>, spin2: Braket<impl SpinLike>) -> f64 {
    let val1 = proj_z(spin1) * proj_z(spin2);
    let val2 = 0.5 * ladder_plus(spin1) * ladder_minus(spin2);
    let val3 = 0.5 * ladder_minus(spin1) * ladder_plus(spin2);

    val1 + val2 + val3
}

/// returns Clebsch-Gordan coefficient <s1; s2 | s3>.
#[inline]
pub fn clebsch_gordan_coef(s1: impl SpinLike, s2: impl SpinLike, s3: impl SpinLike) -> f64 {
    clebsch_gordan::clebsch_gordan(s1.s(), s1.m(), s2.s(), s2.m(), s3.s(), s3.m())
}

/// Checks for triangle inequality of 3 spins
pub fn triangle_condition(s1: impl SpinMagLike, s2: impl SpinMagLike, s3: impl SpinMagLike) -> bool {
    let ds1 = s1.s().double_value();
    let ds2 = s2.s().double_value();
    let ds3 = s3.s().double_value();

    return (ds3 <= ds1 + ds2) && (ds1 <= ds2 + ds3) && (ds2 <= ds3 + ds1) && (ds1 + ds2 + ds3) % 2 == 0;
}

///Returns right hand side of the equation
/// without reduced matrix element
///
/// ```text
///                          (s' - m') ⎧s'  k s⎫
/// <s' m'|T^k_q|s m> = (-1)^          ⎩-m' q m⎭ <s'||T^(k)||s>
/// ```
#[inline]
pub fn wigner_eckart_factor(s: Braket<impl SpinLike>, k: impl SpinLike) -> f64 {
    s.bra.phase_factor() * wigner_3j(s.bra.s(), k.s(), s.ket.s(), -s.bra.m(), k.m(), s.ket.m())
}

/// Returns right hand side of the equation
/// without reduced matrix element.
///
///
/// ```text
///                                          (s1' + s2 + S + k) ⎧S' s2  S⎫
/// <(s1' s2) S'||T^k(s1)||(s1 s2) S> = (-1)^                   ⎩s1 k s1'⎭ ((2S + 1) (2S' + 1)).sqrt() <s1'||T^k(s1)||s1>
/// ```
pub fn red_first_subsystem_mel_factor(
    s: Braket<SpinPairMag<impl SpinMagLike, impl SpinMagLike>>,
    k: impl SpinMagLike,
) -> f64 {
    let s_bra_1 = s.bra.pair.1.s();
    let s_ket_1 = s.ket.pair.1.s();
    if s_bra_1 == s_ket_1 {
        let k = k.s();
        let s_bra_0 = s.bra.pair.0.s();
        let s_ket_0 = s.ket.pair.0.s();
        let s_bra = s.bra.summed;
        let s_ket = s.ket.summed;

        let phase = (-1.0f64).powi((s_bra_0 + s_bra_1 + s_ket + k).double_value() as i32 / 2);
        let factor = red_id_mel(s_bra) * red_id_mel(s_ket);
        let wigner = wigner_6j(s_bra, k, s_ket, s_ket_0, s_ket_1, s_bra_0);

        phase * factor * wigner
    } else {
        0.
    }
}

/// Returns right hand side of the equation
/// without reduced matrix element.
///
///
/// ```text
///                                          (s1 + s2 + S' + k) ⎧S' s1  S⎫
/// <(s1 s2') S'||T^k(s2)||(s1 s2) S> = (-1)^                   ⎩s2 k s2'⎭ ((2S + 1) (2S' + 1)).sqrt() <s2'||T^k(s2)||s2>
/// ```
pub fn red_second_subsystem_mel_factor(
    s: Braket<SpinPairMag<impl SpinMagLike, impl SpinMagLike>>,
    k: impl SpinMagLike,
) -> f64 {
    let s_bra_0 = s.bra.pair.0.s();
    let s_ket_0 = s.ket.pair.0.s();

    if s_bra_0 == s_ket_0 {
        let k = k.s();
        let s_bra_1 = s.bra.pair.1.s();
        let s_ket_1 = s.ket.pair.1.s();
        let s_bra = s.bra.summed;
        let s_ket = s.ket.summed;

        let phase = (-1.0f64).powi((s_ket_0 + s_ket_1 + s_bra + k).double_value() as i32 / 2);
        let factor = red_id_mel(s_bra) * red_id_mel(s_ket);
        let wigner = wigner_6j(s_bra, k, s_ket, s_ket_1, s_ket_0, s_bra_1);

        phase * factor * wigner
    } else {
        0.
    }
}

///Returns <s||1||s> = (2s + 1).sqrt()
#[inline]
pub fn red_id_mel(s: impl SpinMagLike) -> f64 {
    f64::sqrt(s.dim() as f64)
}

#[inline]
///Returns <s||s||s>
pub fn red_spin_mel(s: impl SpinMagLike) -> f64 {
    let s = s.s().value();

    (s * (2. * s + 1.) * (s + 1.)).sqrt()
}

#[inline]
///Returns <l'||Y^(l_sph)||l>
pub fn red_harmonics_mel(l: Braket<impl SpinMagLike>, l_sph: impl SpinMagLike) -> f64 {
    let l_sph = l_sph.s();
    let l_bra = l.bra.s();
    let l_ket = l.ket.s();

    f64::sqrt((l_sph.dim() * l_ket.dim()) as f64 / (4. * PI)) * wigner_3j(l_bra, l_sph, l_ket, hi32!(0), hi32!(0), hi32!(0))
}

#[cfg(test)]
mod tests {
    use clebsch_gordan::{
        hi32,
        hu32,
    };
    use hilbert_space::operator::{
        Braket,
        kron_delta,
    };

    use super::*;
    use crate::Spin;

    #[test]
    fn test_spin_operators() {
        let s1 = Spin::new(hu32!(7 / 2), hi32!(3 / 2));
        let s2 = Spin::new(hu32!(7 / 2), hi32!(5 / 2));
        let s3 = Spin::new(hu32!(5 / 2), hi32!(3 / 2));
        let s4 = Spin::new(hu32!(6), hi32!(4));

        let mel = proj_z(Braket::new(s1, s1));
        assert_eq!(mel, 1.5);

        let mel = proj_z(Braket::new(s1, s2));
        assert_eq!(mel, 0.);

        let mel = proj_z(Braket::new(s1, s3));
        assert_eq!(mel, 0.);

        let mel = ladder_plus(Braket::new(s2, s1));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = ladder_plus(Braket::new(s1, s2));
        assert_eq!(mel, 0.);

        let mel = ladder_plus(Braket::new(s2, s3));
        assert_eq!(mel, 0.);

        let mel = ladder_minus(Braket::new(s1, s2));
        assert_eq!(mel, f64::sqrt(48.) / 2.);

        let mel = ladder_minus(Braket::new(s2, s1));
        assert_eq!(mel, 0.);

        let mel = ladder_minus(Braket::new(s3, s2));
        assert_eq!(mel, 0.);

        let mel = clebsch_gordan_coef(s1, s2, s4);
        assert_eq!(mel, -f64::sqrt(7. / 11.) / 2.);

        let mel = clebsch_gordan_coef(s2, s3, s4);
        assert_eq!(mel, f64::sqrt(35. / 66.));
    }

    #[test]
    fn test_wigner_eckart_operators() {
        let s1 = Spin::new(hu32!(7 / 2), hi32!(3 / 2));
        let s2 = Spin::new(hu32!(7 / 2), hi32!(5 / 2));

        let s_z = wigner_eckart_factor(Braket::new(s1, s1), Spin::new(hu32!(1), hi32!(0))) * red_spin_mel(s1);
        assert_eq!(s_z, s1.m.value());

        let mel = -ladder_plus(Braket::new(s2, s1)) / f64::sqrt(2.);
        let mel_eckart = wigner_eckart_factor(Braket::new(s2, s1), Spin::new(hu32!(1), hi32!(1)))
            * kron_delta([Braket::new(s2.s, s1.s)])
            * red_spin_mel(s2);
        assert_eq!(mel, mel_eckart);

        let s1 = hu32!(1 / 2);
        let s2 = hu32!(1);

        let f1 = SpinPair::new((s1, s2), Spin::new(hu32!(3 / 2), hi32!(1 / 2)));
        let f2 = SpinPair::new((s1, s2), Spin::new(hu32!(3 / 2), hi32!(1 / 2)));
        let braket = Braket::new(f1, f2);

        let mel_eckart_1 = wigner_eckart_factor(braket, Spin::new(hu32!(1), hi32!(0)))
            * red_first_subsystem_mel_factor(braket.map(|x| x.as_spin_pair_mag()), hu32!(1))
            * red_spin_mel(s1);

        let mel: f64 = get_spin_basis(s1)
            .iter()
            .map(|&spin| {
                clebsch_gordan(s1, spin.m, s2, f1.m() - spin.m, f1.s(), f1.m())
                    * clebsch_gordan(s1, spin.m, s2, f1.m() - spin.m, f2.s(), f2.m())
                    * spin.m.value()
            })
            .sum();

        assert_eq!(mel_eckart_1, mel);

        let mel_eckart_2 = wigner_eckart_factor(braket, Spin::new(hu32!(1), hi32!(0)))
            * red_second_subsystem_mel_factor(braket.map(|x| x.as_spin_pair_mag()), hu32!(1))
            * red_spin_mel(s2);

        let mel: f64 = get_spin_basis(s2)
            .iter()
            .map(|&spin| {
                clebsch_gordan(s1, f1.m() - spin.m, s2, spin.m, f1.s(), f1.m())
                    * clebsch_gordan(s1, f1.m() - spin.m, s2, spin.m, f2.s(), f2.m())
                    * spin.m.value()
            })
            .sum();

        assert_eq!(mel_eckart_2, mel);

        assert_eq!(mel_eckart_1 + mel_eckart_2, proj_z(braket))
    }
}
