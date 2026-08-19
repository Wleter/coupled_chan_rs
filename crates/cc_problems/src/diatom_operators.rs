use hilbert_space::{
    operator_mel,
    space::BasisElementsRef,
};
use spin_algebra::{
    Spin,
    SpinLike,
    SpinMagLike,
    ops::{
        red_dot_product_factor,
        red_first_subsystem_mel_factor,
        red_second_subsystem_mel_factor,
        red_spin_mel,
        wigner_eckart_dot_product_factor,
        wigner_eckart_factor,
    },
};

use crate::{
    Operator,
    atom_operators::{
        AHifiId,
        BFieldId,
        GFactorId,
        HifiSpec,
        ZeemanSpec,
    },
    diatom_basis::{
        CoupledDiatomBasis,
        CoupledFTotDiatomBasis,
        CoupledSIDiatomBasis,
    },
};

impl CoupledSIDiatomBasis {
    pub fn hifi_a(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_tot_id = self.s_tot;
        let i_tot_id = self.i_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [s_tot_id, i_tot_id], |[s_tot, i_tot]| {
                    let q_s = s_tot.bra.m() - s_tot.ket.m();
                    let q_i = i_tot.bra.m() - i_tot.ket.m();
                    let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());
                    let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                    if q_s == -q_i && q_s.double_value().abs() <= 2 {
                        (-1f64).powi(q_s.double_value() / 2)
                            * wigner_eckart_factor(s_tot, Spin::new(1, q_s))
                            * red_first_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot.bra.pair.0)
                            * wigner_eckart_factor(i_tot, Spin::new(1, q_i))
                            * red_first_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn hifi_b(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_tot_id = self.s_tot;
        let i_tot_id = self.i_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [s_tot_id, i_tot_id], |[s_tot, i_tot]| {
                    let q_s = s_tot.bra.m() - s_tot.ket.m();
                    let q_i = i_tot.bra.m() - i_tot.ket.m();
                    let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());
                    let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                    if q_s == -q_i && q_s.double_value().abs() <= 2 {
                        (-1f64).powi(q_s.double_value() / 2)
                            * (wigner_eckart_factor(s_tot, Spin::new(1, q_s))
                                * red_second_subsystem_mel_factor(s_tot_mag, 1)
                                * red_spin_mel(s_tot.bra.pair.1))
                            * (wigner_eckart_factor(i_tot, Spin::new(1, q_i))
                                * red_second_subsystem_mel_factor(i_tot_mag, 1)
                                * red_spin_mel(i_tot.bra.pair.1))
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_tot_id = self.s_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [s_tot_id], |[s_tot]| {
                    let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());

                    if s_tot.bra.m() == s_tot.ket.m() {
                        -wigner_eckart_factor(s_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_tot_id = self.s_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [s_tot_id], |[s_tot]| {
                    let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());

                    if s_tot.bra.m() == s_tot.ket.m() {
                        -wigner_eckart_factor(s_tot, Spin::new(1, 0))
                            * red_second_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let i_tot_id = self.i_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [i_tot_id], |[i_tot]| {
                    let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                    if i_tot.bra.m() == i_tot.ket.m() {
                        -wigner_eckart_factor(i_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let i_tot_id = self.i_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [i_tot_id], |[i_tot]| {
                    let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                    if i_tot.bra.m() == i_tot.ket.m() {
                        -wigner_eckart_factor(i_tot, Spin::new(1, 0))
                            * red_second_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }
}

impl CoupledFTotDiatomBasis {
    pub fn hifi_a(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    if f_tot.bra.spin == f_tot.ket.spin {
                        wigner_eckart_dot_product_factor(f_tot, 1)
                            * red_first_subsystem_mel_factor(f_tot.map(|x| x.pair.0), 1)
                            * red_spin_mel(f_tot.bra.pair.0.pair.0)
                            * red_first_subsystem_mel_factor(f_tot.map(|x| x.pair.1), 1)
                            * red_spin_mel(f_tot.bra.pair.1.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn hifi_b(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    if f_tot.bra.spin == f_tot.ket.spin {
                        wigner_eckart_dot_product_factor(f_tot, 1)
                            * red_second_subsystem_mel_factor(f_tot.map(|x| x.pair.0), 1)
                            * red_spin_mel(f_tot.bra.pair.0.pair.1)
                            * red_second_subsystem_mel_factor(f_tot.map(|x| x.pair.1), 1)
                            * red_spin_mel(f_tot.bra.pair.1.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    let f_tot_mag = f_tot.map(|x| x.as_spin_pair_mag());
                    let s_tot_mag = f_tot_mag.map(|x| x.pair.0);

                    if f_tot.bra.m() == f_tot.ket.m() {
                        -wigner_eckart_factor(f_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(f_tot_mag, 1)
                            * red_first_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    let f_tot_mag = f_tot.map(|x| x.as_spin_pair_mag());
                    let s_tot_mag = f_tot_mag.map(|x| x.pair.0);

                    if f_tot.bra.m() == f_tot.ket.m() {
                        -wigner_eckart_factor(f_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(f_tot_mag, 1)
                            * red_second_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    let f_tot_mag = f_tot.map(|x| x.as_spin_pair_mag());
                    let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                    if f_tot.bra.m() == f_tot.ket.m() {
                        -wigner_eckart_factor(f_tot, Spin::new(1, 0))
                            * red_second_subsystem_mel_factor(f_tot_mag, 1)
                            * red_first_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_tot_id = self.f_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_tot_id], |[f_tot]| {
                    let f_tot_mag = f_tot.map(|x| x.as_spin_pair_mag());
                    let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                    if f_tot.bra.m() == f_tot.ket.m() {
                        -wigner_eckart_factor(f_tot, Spin::new(1, 0))
                            * red_second_subsystem_mel_factor(f_tot_mag, 1)
                            * red_second_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }
}

impl CoupledDiatomBasis {
    pub fn hifi_a(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    if fl_tot.bra.spin == fl_tot.ket.spin
                        && fl_tot.bra.pair.0.s() == fl_tot.ket.pair.0.s()
                        && fl_tot.bra.pair.1.s() == fl_tot.ket.pair.1.s()
                    {
                        let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                        let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                        let s_tot_mag = f_tot_mag.map(|x| x.pair.0);
                        let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                        wigner_eckart_factor(fl_tot, Spin::new(0, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 0)
                            * red_dot_product_factor(f_tot_mag, 1)
                            * red_first_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.0)
                            * red_first_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn hifi_b(&self, a_hifi: AHifiId) -> HifiSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        HifiSpec {
            a_hifi,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    if fl_tot.bra.spin == fl_tot.ket.spin
                        && fl_tot.bra.pair.0.s() == fl_tot.ket.pair.0.s()
                        && fl_tot.bra.pair.1.s() == fl_tot.ket.pair.1.s()
                    {
                        let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                        let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                        let s_tot_mag = f_tot_mag.map(|x| x.pair.0);
                        let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                        wigner_eckart_factor(fl_tot, Spin::new(0, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 0)
                            * red_dot_product_factor(f_tot_mag, 1)
                            * red_second_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.1)
                            * red_second_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                    let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                    let s_tot_mag = f_tot_mag.map(|x| x.pair.0);

                    if fl_tot.bra.m() == fl_tot.ket.m() {
                        -wigner_eckart_factor(fl_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 1)
                            * red_first_subsystem_mel_factor(f_tot_mag, 1)
                            * red_first_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_e_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                    let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                    let s_tot_mag = f_tot_mag.map(|x| x.pair.0);

                    if fl_tot.bra.m() == fl_tot.ket.m() {
                        -wigner_eckart_factor(fl_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 1)
                            * red_first_subsystem_mel_factor(f_tot_mag, 1)
                            * red_second_subsystem_mel_factor(s_tot_mag, 1)
                            * red_spin_mel(s_tot_mag.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_a(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                    let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                    let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                    if fl_tot.bra.m() == fl_tot.ket.m() {
                        -wigner_eckart_factor(fl_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 1)
                            * red_second_subsystem_mel_factor(f_tot_mag, 1)
                            * red_first_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n_b(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let fl_tot_id = self.fl_tot;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [fl_tot_id], |[fl_tot]| {
                    let fl_tot_mag = fl_tot.map(|x| x.as_spin_pair_mag());
                    let f_tot_mag = fl_tot_mag.map(|x| x.pair.0);
                    let i_tot_mag = f_tot_mag.map(|x| x.pair.1);

                    if fl_tot.bra.m() == fl_tot.ket.m() {
                        -wigner_eckart_factor(fl_tot, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(fl_tot_mag, 1)
                            * red_second_subsystem_mel_factor(f_tot_mag, 1)
                            * red_second_subsystem_mel_factor(i_tot_mag, 1)
                            * red_spin_mel(i_tot_mag.bra.pair.1)
                    } else {
                        0.
                    }
                })
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use cc_math_utils::assert_approx_eq;
    use hilbert_space::space::SpaceBasis;
    use serde::{
        Deserialize,
        Serialize,
    };
    use spin_algebra::{
        hi32,
        hu32,
    };
    use unit_systems::quantities::{
        Scalar,
        phys_quantities::{
            Energy,
            MagneticDipole,
            MagneticField,
        },
    };

    use crate::{
        OrbitalBasisElements,
        OrbitalRecipe,
        atom_basis::AtomRecipe,
        atom_operators::AtomParams,
        diatom_basis::{
            CoupledFDiatomBasis,
            DiatomRecipe,
            UncoupledDiatomBasis,
        },
        parameters::Parameters,
        system::{
            DynOperatorSpec,
            HamiltonianSpec,
            System,
        },
    };

    use super::*;

    #[derive(Clone, cc_derive::Parameters, Serialize, Deserialize)]
    struct Params {
        b_field: Scalar<MagneticField>,
        #[parameter(nested)]
        atom_a: AtomParams,
        #[parameter(nested)]
        atom_b: AtomParams,
    }

    #[test]
    fn test_diatom_hamiltonian_terms() {
        let recipe = DiatomRecipe {
            atom_a: AtomRecipe {
                s: hu32!(1 / 2),
                i: hu32!(1),
            },
            atom_b: AtomRecipe {
                s: hu32!(1 / 2),
                i: hu32!(1 / 2),
            },
            l: OrbitalRecipe::LMaxProjections(1),
        };
        let params = Params {
            b_field: Scalar::new(80.0, MagneticField, "Gauss"),
            atom_a: AtomParams {
                a_hifi: Scalar::new(1., Energy, "GHz"),
                g_e: Scalar::new(2., MagneticDipole, "mu_bohr"),
                g_n: Scalar::new(3., MagneticDipole, "mu_bohr"),
            },
            atom_b: AtomParams {
                a_hifi: Scalar::new(0.5, Energy, "GHz"),
                g_e: Scalar::new(1., MagneticDipole, "mu_bohr"),
                g_n: Scalar::new(1.5, MagneticDipole, "mu_bohr"),
            },
        };
        let ids = Params::ids();

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = UncoupledDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| {
            diatom.filter(|((s1, i1), (s2, i2), l)| s1.m + i1.m + s2.m + i2.m + l.m == hi32!(3 / 2))(x)
        });
        let basis = OrbitalBasisElements::from_orbital(elements, &diatom.l);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("atom_a.hifi", DynOperatorSpec::new(diatom.atom_a.hifi(ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.atom_a.zeeman_e(ids.b_field, ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.atom_a.zeeman_n(ids.b_field, ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.atom_b.hifi(ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.atom_b.zeeman_e(ids.b_field, ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.atom_b.zeeman_n(ids.b_field, ids.atom_b.g_n)),
            ),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_uncoupled = system.angular_blocks().diagonalized().0.asymptote;

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledFDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|(f1, f2, l)| f1.m() + f2.m() + l.m == hi32!(3 / 2))(x));
        let basis = OrbitalBasisElements::from_orbital(elements, &diatom.l);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("atom_a.hifi", DynOperatorSpec::new(diatom.atom_a.hifi(ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.atom_a.zeeman_e(ids.b_field, ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.atom_a.zeeman_n(ids.b_field, ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.atom_b.hifi(ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.atom_b.zeeman_e(ids.b_field, ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.atom_b.zeeman_n(ids.b_field, ids.atom_b.g_n)),
            ),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_coupled_f = system.angular_blocks().diagonalized().0.asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_f, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledSIDiatomBasis::new(recipe, &mut basis);
        let elements =
            basis.get_filtered_basis(|x| diatom.filter(|(s_tot, i_tot, l)| s_tot.m() + i_tot.m() + l.m == hi32!(3 / 2))(x));
        let basis = OrbitalBasisElements::from_orbital(elements, &diatom.l);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("atom_a.hifi", DynOperatorSpec::new(diatom.hifi_a(ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_a(ids.b_field, ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_a(ids.b_field, ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.hifi_b(ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_b(ids.b_field, ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_b(ids.b_field, ids.atom_b.g_n)),
            ),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_coupled_si = system.angular_blocks().diagonalized().0.asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_si, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledFTotDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|(f_tot, l)| f_tot.m() + l.m == hi32!(3 / 2))(x));
        let basis = OrbitalBasisElements::from_orbital(elements, &diatom.l);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("atom_a.hifi", DynOperatorSpec::new(diatom.hifi_a(ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_a(ids.b_field, ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_a(ids.b_field, ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.hifi_b(ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_b(ids.b_field, ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_b(ids.b_field, ids.atom_b.g_n)),
            ),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_coupled_f_tot = system.angular_blocks().diagonalized().0.asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_f_tot, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|fl_tot| fl_tot.m() == hi32!(3 / 2))(x));
        let basis = OrbitalBasisElements::new(elements, diatom.fl_tot, |x| x.pair.1);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("atom_a.hifi", DynOperatorSpec::new(diatom.hifi_a(ids.atom_a.a_hifi))),
            (
                "atom_a.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_a(ids.b_field, ids.atom_a.g_e)),
            ),
            (
                "atom_a.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_a(ids.b_field, ids.atom_a.g_n)),
            ),
            ("atom_b.hifi", DynOperatorSpec::new(diatom.hifi_b(ids.atom_b.a_hifi))),
            (
                "atom_b.zeeman_e",
                DynOperatorSpec::new(diatom.zeeman_e_b(ids.b_field, ids.atom_b.g_e)),
            ),
            (
                "atom_b.zeeman_n",
                DynOperatorSpec::new(diatom.zeeman_n_b(ids.b_field, ids.atom_b.g_n)),
            ),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_coupled_fl_tot = system.angular_blocks().diagonalized().0.asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_fl_tot, 1e-6);
    }
}
