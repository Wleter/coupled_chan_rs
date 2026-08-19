use hilbert_space::operator::Braket;
use spin_algebra::{
    Spin,
    SpinLike,
    SpinMagLike,
    SpinPair,
    get_spin_basis,
    half_integer::HalfU32,
    hi32,
    hu32,
    ops::{
        clebsch_gordan_coef,
        triangle_condition,
    },
    wigner_3j,
    wigner_6j,
};

use crate::Angular;

pub fn dot_coupled(f: SpinPair<impl SpinMagLike, impl SpinMagLike>) -> f64 {
    (f.spin.squared() - f.pair.0.squared() - f.pair.1.squared()) / 2.
}

pub fn spin_sum_projection_uncoupled(
    s1: Braket<impl SpinLike>,
    s2: Braket<impl SpinLike>,
    s_tot_projected: impl SpinMagLike,
) -> f64 {
    let spins = get_spin_basis(s_tot_projected.s());

    if s1.bra.m() + s2.bra.m() != s1.ket.m() + s2.ket.m()
        || !triangle_condition(s1.bra, s2.bra, s_tot_projected)
        || !triangle_condition(s1.ket, s2.ket, s_tot_projected)
    {
        return 0.;
    }

    spins
        .iter()
        .map(|&s| clebsch_gordan_coef(s1.bra, s2.bra, s) * clebsch_gordan_coef(s1.ket, s2.ket, s))
        .sum()
}

pub fn spin_projection_term_coupled(s: impl SpinLike, s_projected: impl SpinMagLike) -> f64 {
    let s_projected = s_projected.s();
    if s.s() == s_projected { 1. } else { 0. }
}

#[rustfmt::skip]
pub fn percival_coef(lambda: u32, l: Braket<Angular>, n: Braket<Angular>, n_tot: HalfU32) -> f64 {
    let lambda = lambda.into();

    let sign = (-1.0f64).powi(((l.bra.l + l.ket.l).double_value() as i32 - n_tot.double_value() as i32) / 2);
        
    let prefactor = p1_factor(n.bra.l) * p1_factor(n.ket.l)
        * p1_factor(l.bra.l) * p1_factor(l.ket.l);

    let wigners = wigner_3j(l.bra.l, lambda, l.ket.l, hi32!(0), hi32!(0), hi32!(0))
        * wigner_3j(n.bra.l, lambda, n.ket.l, hi32!(0), hi32!(0), hi32!(0))
        * wigner_6j(l.bra.l, lambda, l.ket.l, n.ket.l, n_tot, n.bra.l);

    sign * prefactor * wigners
}

#[rustfmt::skip]
pub fn percival_coef_tram_mel(lambda: u32, l: Braket<Angular>, n: Braket<Angular>, n_tot: Braket<Angular>) -> f64 {
    if n_tot.bra == n_tot.ket {
        let lambda = lambda.into();
            
        let sign = (-1.0f64).powi(((l.bra.l + l.ket.l).double_value() as i32 - n_tot.bra.l.double_value() as i32) / 2);
            
        let prefactor = p1_factor(n.bra.l) * p1_factor(n.ket.l)
            * p1_factor(l.bra.l) * p1_factor(l.ket.l);

        let wigners = wigner_3j(l.bra.l, lambda, l.ket.l, hi32!(0), hi32!(0), hi32!(0))
            * wigner_3j(n.bra.l, lambda, n.ket.l, hi32!(0), hi32!(0), hi32!(0))
            * wigner_6j(l.bra.l, lambda, l.ket.l, n.ket.l, n_tot.bra.l, n.bra.l);

        sign * prefactor * wigners
    } else {
        0.0
    }
}

pub fn singlet_projection_uncoupled(s1: Braket<Spin>, s2: Braket<Spin>) -> f64 {
    let singlet_spin = Spin::zero();
    clebsch_gordan_coef(s1.bra, s2.bra, singlet_spin) * clebsch_gordan_coef(s1.ket, s2.ket, singlet_spin)
}

#[rustfmt::skip]
pub fn triplet_projection_uncoupled(s1: Braket<Spin>, s2: Braket<Spin>) -> f64 {
    let mut value = 0.0;

    for ms in [-hi32!(1), hi32!(0), hi32!(1)] {
        let triplet_spin = Spin::new(hu32!(1), ms);
        value += clebsch_gordan_coef(s1.bra, s2.bra, triplet_spin) 
            * clebsch_gordan_coef(s1.ket, s2.ket, triplet_spin)
    }

    value
}

#[rustfmt::skip]
pub fn spin_rot_tram_mel(l: Braket<Angular>, n: Braket<Angular>, n_tot: Braket<Spin>, s: Braket<Spin>) -> f64 {
    if l.bra == l.ket && n.bra == n.ket && s.bra.s == s.ket.s {
        let l = l.bra.l;
        let n = n.bra.l;

        let factor = p1_factor(n_tot.ket.s) * p1_factor(n_tot.bra.s)
            * p3_factor(n) 
            * p3_factor(s.bra.s);

        let sign = (-1f64).powi(1 + (n + l + n_tot.ket.s).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s.bra);

        let mut wigner_sum = 0.;
        for p in [-1, 0, 1] { 
            wigner_sum += (-1.0f64).powi(p) 
                * wigner_6j(n, hu32!(1), n, n_tot.bra.s, l, n_tot.ket.s)
                * wigner_3j(n_tot.bra.s, hu32!(1), n_tot.ket.s, -n_tot.bra.m, p.into(), n_tot.ket.m)
                * wigner_3j(s.bra.s, hu32!(1), s.bra.s, -s.bra.m, (-p).into(), s.ket.m)
        }

        factor * sign * wigner_sum
    } else {
        0.
    }
}

#[rustfmt::skip]
pub fn aniso_hifi_tram_mel(
    l: Braket<Angular>, 
    n: Braket<Angular>, 
    n_tot: Braket<Spin>, 
    s: Braket<Spin>, 
    i: Braket<Spin>
) -> f64 {
    if l.bra == l.ket && s.bra.s == s.ket.s && i.bra.s == i.ket.s {
        let l: HalfU32 = l.bra.l;
        let n = Braket {
            bra: n.bra.l,
            ket: n.ket.l,
        };

        let factor = p1_factor(n_tot.ket.s) * p1_factor(n_tot.bra.s)
            * p1_factor(n.bra) * p1_factor(n.ket)
            * p3_factor(i.bra.s) 
            * p3_factor(s.bra.s);

        let sign = (-1f64).powi((l + n_tot.ket.s).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s.bra)
            * spin_phase_factor(i.bra);


        let wigner = wigner_6j(n.bra, hu32!(2), n.ket, n_tot.ket.s, l, n_tot.bra.s)
            * wigner_3j(hu32!(1), hu32!(1), hu32!(2), 
                        i.bra.m - i.ket.m, s.bra.m - s.ket.m, n_tot.bra.m - n_tot.ket.m)
            * wigner_3j(n_tot.bra.s, hu32!(2), n_tot.ket.s, 
                        -n_tot.bra.m, n_tot.bra.m - n_tot.ket.m, n_tot.ket.m)
            * wigner_3j(n.bra, hu32!(2), n.ket, hi32!(0), hi32!(0), hi32!(0))
            * wigner_3j(i.bra.s, hu32!(1), i.bra.s, -i.bra.m, i.bra.m - i.ket.m, i.ket.m)
            * wigner_3j(s.bra.s, hu32!(1), s.bra.s, -s.bra.m, s.bra.m - s.ket.m, s.ket.m);

        f64::sqrt(30.) / 3. * sign * factor * wigner
    } else {
        0.
    }
}

#[rustfmt::skip]
pub fn dipole_dipole_tram_mel(
    l: Braket<Angular>, 
    n: Braket<Angular>, 
    n_tot: Braket<Spin>, 
    s_r: Braket<Spin>,
    s_a: Braket<Spin>
) -> f64 {
    if n.bra == n.ket && s_r.bra.s == s_r.ket.s && s_a.bra.s == s_a.ket.s {
        let n = n.bra.l;
        let l = Braket {
            bra: l.bra.l,
            ket: l.ket.l,
        };

        let factor = p1_factor(n_tot.bra.s) * p1_factor(n_tot.ket.s)
            * p1_factor(l.bra) * p1_factor(l.ket)
            * p3_factor(s_r.bra.s) * p3_factor(s_a.bra.s);

        let sign = (-1f64).powi((n_tot.bra.s + l.bra + l.ket + n).double_value() as i32 / 2)
            * spin_phase_factor(n_tot.bra)
            * spin_phase_factor(s_r.bra) 
            * spin_phase_factor(s_a.bra);


        let wigner = wigner_6j(l.bra, hu32!(2), l.ket, n_tot.ket.s, n, n_tot.bra.s)
            * wigner_3j(hu32!(1), hu32!(1), hu32!(2), 
                        s_r.bra.m - s_r.ket.m, s_a.bra.m - s_a.ket.m, n_tot.bra.m - n_tot.ket.m)
            * wigner_3j(l.bra, hu32!(2), l.ket, hi32!(0), hi32!(0), hi32!(0))
            * wigner_3j(n_tot.bra.s, hu32!(2), n_tot.bra.s, -n_tot.bra.m, n_tot.bra.m - n_tot.ket.m, n_tot.ket.m)
            * wigner_3j(s_a.bra.s, hu32!(1), s_a.bra.s, -s_a.bra.m, s_a.bra.m - s_a.ket.m, s_a.ket.m)
            * wigner_3j(s_r.bra.s, hu32!(1), s_r.bra.s, -s_r.bra.m, s_r.bra.m - s_r.ket.m, s_r.ket.m);

        -f64::sqrt(30.) * sign * factor * wigner
    } else {
        0.
    }
}

#[inline]
/// Calculates sqrt(2s + 1)
pub fn p1_factor(s: HalfU32) -> f64 {
    (2. * s.value() + 1.).sqrt()
}

#[inline]
/// Calculates sqrt((2s + 1)s(s + 1))
pub fn p3_factor(s: HalfU32) -> f64 {
    let s = s.value();
    ((2. * s + 1.) * s * (s + 1.)).sqrt()
}

#[inline]
/// Calculates (-1)^(s - ms)
pub fn spin_phase_factor(s: Spin) -> f64 {
    (-1.0f64).powi((s.s.double_value() as i32 - s.m.double_value()) / 2)
}
