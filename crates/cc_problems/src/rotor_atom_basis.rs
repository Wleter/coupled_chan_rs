use hilbert_space::{
    operator::Braket,
    operator_diag_mel,
    operator_mel,
    space::{
        BasisElementsRef,
        BasisId,
        SpaceBasis,
        SpaceElement,
        SubspaceBasisOf,
    },
};
use serde::{
    Deserialize,
    Serialize,
};
use spin_algebra::{
    Spin,
    SpinMagLike,
    get_spin_basis,
    half_integer::{
        HalfI32,
        HalfU32,
    },
    ops::{
        dot_product_separation,
        red_first_subsystem_mel_factor,
        red_reduced_harmonics_mel,
        red_second_subsystem_mel_factor,
        red_spin_mel,
        tensor_product_separation,
        wigner_eckart_dot_product_factor,
        wigner_eckart_factor,
    },
    spin,
};

use crate::{
    Angular,
    Operator,
    atom_basis::{
        AtomRecipe,
        UncoupledAtomBasis,
    },
    atom_operators::{
        CouplingId,
        CouplingSpec,
    },
    operator_mel::spin_sum_projection_uncoupled,
    problems::diatom_in_b_field::OrbitalParity,
    tram_basis::{
        AngularCoupled,
        TRAMBasis,
        TRAMRecipe,
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotorRecipe {
    pub name: Box<str>,
    pub s: HalfU32,
    pub i_a: HalfU32,
    pub i_b: HalfU32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TRAMRotorAtomBasisRecipe {
    pub rotor: RotorRecipe,
    pub atom: AtomRecipe,
    pub tram: TRAMRecipe,

    #[serde(default)]
    pub projection: Option<HalfI32>,
    #[serde(default)]
    pub l_parity: OrbitalParity,
}

/// Struct for storing id of
/// |s_r m_sr>|s_a m_sa>|i_ra m_i_ra>|i_rb m_i_rb>|i_a m_i_a>|(n l) N M_N> state.
#[derive(Debug, Clone, Copy)]
pub struct TRAMRotorAtomBasis {
    pub tram: TRAMBasis,
    pub atom: UncoupledAtomBasis,
    pub s_r: BasisId<Spin>,
    pub rotor_a: UncoupledAtomBasis,
    pub rotor_b: UncoupledAtomBasis,
}

impl TRAMRotorAtomBasis {
    pub fn new(recipe: &TRAMRotorAtomBasisRecipe, basis: &mut SpaceBasis) -> Self {
        let s_r = get_spin_basis(recipe.rotor.s);
        let s_r = basis.push_subspace(SubspaceBasisOf::new(s_r));
        let s_a = get_spin_basis(recipe.atom.s);
        let s_a = basis.push_subspace(SubspaceBasisOf::new(s_a));

        let i_a = get_spin_basis(recipe.atom.i);
        let i_a = basis.push_subspace(SubspaceBasisOf::new(i_a));
        let i_ra = get_spin_basis(recipe.rotor.i_a);
        let i_ra = basis.push_subspace(SubspaceBasisOf::new(i_ra));
        let i_rb = get_spin_basis(recipe.rotor.i_b);
        let i_rb = basis.push_subspace(SubspaceBasisOf::new(i_rb));

        Self {
            tram: TRAMBasis::new(&recipe.tram, basis),
            atom: UncoupledAtomBasis { s: s_a, i: i_a },
            s_r,
            rotor_a: UncoupledAtomBasis { s: s_r, i: i_ra },
            rotor_b: UncoupledAtomBasis { s: s_r, i: i_rb },
        }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn(((Spin, Spin, Spin), (Spin, Spin), AngularCoupled)) -> bool,
    {
        move |x| {
            let s_r = x[self.s_r];
            let i_ra = x[self.rotor_a.i];
            let i_rb = x[self.rotor_b.i];
            let s_a = x[self.atom.s];
            let i_a = x[self.atom.i];
            let n_tot = x[self.tram.tram];

            f(((s_r, i_ra, i_rb), (s_a, i_a), n_tot))
        }
    }

    pub fn rot_energy(&self, rot_const: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let n_tot_id = self.tram.tram;
        CouplingSpec {
            coupling: rot_const,
            operator: move |b| operator_diag_mel!(b, [n_tot_id], |[n_tot]| { Angular::new(n_tot.pair.0, 0).squared() }),
        }
    }

    pub fn rot_energy_distortion(
        &self,
        rot_distortion: CouplingId,
    ) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let n_tot_id = self.tram.tram;
        CouplingSpec {
            coupling: rot_distortion,
            operator: move |b| {
                operator_diag_mel!(b, [n_tot_id], |[n_tot]| { -Angular::new(n_tot.pair.0, 0).squared().powi(2) })
            },
        }
    }

    pub fn spin_e_rot(&self, spin_e_rot: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        CouplingSpec {
            coupling: spin_e_rot,
            operator: spin_rot(self.s_r, self.tram.tram),
        }
    }

    pub fn spin_n_a_rot(&self, spin_n_rot: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        CouplingSpec {
            coupling: spin_n_rot,
            operator: spin_rot(self.rotor_a.i, self.tram.tram),
        }
    }

    pub fn spin_n_b_rot(&self, spin_n_rot: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        CouplingSpec {
            coupling: spin_n_rot,
            operator: spin_rot(self.rotor_b.i, self.tram.tram),
        }
    }

    pub fn aniso_hifi_a(&self, c_aniso: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        CouplingSpec {
            coupling: c_aniso,
            operator: aniso_hifi(self.s_r, self.rotor_a.i, self.tram.tram),
        }
    }

    pub fn aniso_hifi_b(&self, c_aniso: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        CouplingSpec {
            coupling: c_aniso,
            operator: aniso_hifi(self.s_r, self.rotor_b.i, self.tram.tram),
        }
    }

    pub fn pes_polarization_legendre_component(
        &self,
        lambda: u32,
        s_tot: HalfU32,
    ) -> impl Fn(BasisElementsRef) -> Operator + use<> {
        let n_tot_id = self.tram.tram;
        let s_r_id = self.s_r;
        let s_a_id = self.atom.s;

        move |b| {
            operator_mel!(b, [s_r_id, s_a_id, n_tot_id], |[s_r, s_a, n_tot]| {
                spin_sum_projection_uncoupled(s_r, s_a, s_tot)
                    * wigner_eckart_dot_product_factor(n_tot, lambda)
                    * red_reduced_harmonics_mel(n_tot.map(|x| x.pair.0), lambda)
                    * red_reduced_harmonics_mel(n_tot.map(|x| x.pair.1), lambda)
            })
        }
    }

    pub fn second_order_so(&self) -> impl Fn(BasisElementsRef) -> Operator + use<> {
        let n_tot_id = self.tram.tram;
        let s_r_id = self.s_r;
        let s_a_id = self.atom.s;

        move |b| {
            operator_mel!(b, [s_r_id, s_a_id, n_tot_id], |[s_r, s_a, n_tot]| {
                if n_tot.bra.pair.0 == n_tot.ket.pair.0 {
                    let n_tot_mag = n_tot.map(|x| x.as_spin_pair_mag());
                    let si = Braket {
                        bra: spin!(s_r.bra.s + s_a.bra.s, s_r.bra.m + s_a.bra.m),
                        ket: spin!(s_r.ket.s + s_a.ket.s, s_r.ket.m + s_a.ket.m),
                    };

                    f64::sqrt(6.0)
                        * dot_product_separation(
                            si,
                            n_tot,
                            2,
                            |q| {
                                tensor_product_separation(
                                    s_r,
                                    s_a,
                                    1,
                                    1,
                                    spin!(2, q),
                                    |q2| wigner_eckart_factor(s_r, spin!(1, q2)) * red_spin_mel(s_r.bra),
                                    |q2| wigner_eckart_factor(s_a, spin!(1, q2)) * red_spin_mel(s_a.bra),
                                )
                            },
                            |q| {
                                wigner_eckart_factor(n_tot, spin!(2, q))
                                    * red_second_subsystem_mel_factor(n_tot_mag, 2)
                                    * red_reduced_harmonics_mel(n_tot_mag.map(|x| x.pair.1), 2)
                            },
                        )
                } else {
                    0.0
                }
            })
        }
    }
}

fn spin_rot(
    s_id: BasisId<Spin>,
    n_tot_id: BasisId<AngularCoupled>,
) -> impl Fn(BasisElementsRef) -> Operator + 'static + Send + Sync {
    move |b| {
        operator_mel!(b, [s_id, n_tot_id], |[s, n_tot]| {
            if n_tot.bra.pair == n_tot.ket.pair {
                let n_tot_mag = n_tot.map(|x| x.as_spin_pair_mag());

                dot_product_separation(
                    s,
                    n_tot,
                    1,
                    |q| wigner_eckart_factor(s, spin!(1, q)) * red_spin_mel(s.bra),
                    |q| {
                        wigner_eckart_factor(n_tot, spin!(1, q))
                            * red_first_subsystem_mel_factor(n_tot_mag, 1)
                            * red_spin_mel(n_tot_mag.bra.pair.0)
                    },
                )
            } else {
                0.0
            }
        })
    }
}

fn aniso_hifi(
    s_id: BasisId<Spin>,
    i_id: BasisId<Spin>,
    n_tot_id: BasisId<AngularCoupled>,
) -> impl Fn(BasisElementsRef) -> Operator + 'static + Send + Sync {
    move |b| {
        operator_mel!(b, [s_id, i_id, n_tot_id], |[s, i, n_tot]| {
            if n_tot.bra.pair.1 == n_tot.ket.pair.1 {
                let n_tot_mag = n_tot.map(|x| x.as_spin_pair_mag());
                let si = Braket {
                    bra: spin!(s.bra.s + i.bra.s, s.bra.m + i.bra.m),
                    ket: spin!(s.ket.s + i.ket.s, s.ket.m + i.ket.m),
                };

                f64::sqrt(6.0) / 3.0
                    * dot_product_separation(
                        si,
                        n_tot,
                        2,
                        |q| {
                            tensor_product_separation(
                                s,
                                i,
                                1,
                                1,
                                spin!(2, q),
                                |q2| wigner_eckart_factor(s, spin!(1, q2)) * red_spin_mel(s.bra),
                                |q2| wigner_eckart_factor(i, spin!(1, q2)) * red_spin_mel(i.bra),
                            )
                        },
                        |q| {
                            wigner_eckart_factor(n_tot, spin!(2, q))
                                * red_first_subsystem_mel_factor(n_tot_mag, 2)
                                * red_reduced_harmonics_mel(n_tot_mag.map(|x| x.pair.0), 2)
                        },
                    )
            } else {
                0.0
            }
        })
    }
}
