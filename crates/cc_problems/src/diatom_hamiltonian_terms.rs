use hilbert_space::operator_mel;
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
    diatom_basis::{
        CoupledDiatomBasis,
        CoupledFTotDiatomBasis,
        CoupledSIDiatomBasis,
    },
    hamiltonian::{
        HamiltonianConstructor,
        HamiltonianTerm,
        TermRecipe,
    },
};

impl CoupledSIDiatomBasis {
    pub fn add_hyperfine_a(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let s_tot_id = self.s_tot;
        let i_tot_id = self.i_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [s_tot_id, i_tot_id], |[s_tot, i_tot]| {
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
                })
            },
        ));
    }

    pub fn add_hyperfine_b(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let s_tot_id = self.s_tot;
        let i_tot_id = self.i_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [s_tot_id, i_tot_id], |[s_tot, i_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_e_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let s_tot_id = self.s_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [s_tot_id], |[s_tot]| {
                        let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());

                        if s_tot.bra.m() == s_tot.ket.m() {
                            -wigner_eckart_factor(s_tot, Spin::new(1, 0))
                                * red_first_subsystem_mel_factor(s_tot_mag, 1)
                                * red_spin_mel(s_tot.bra.pair.0)
                        } else {
                            0.
                        }
                    })
                })
            },
        ));
    }

    pub fn add_zeeman_e_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let s_tot_id = self.s_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [s_tot_id], |[s_tot]| {
                        let s_tot_mag = s_tot.map(|x| x.as_spin_pair_mag());

                        if s_tot.bra.m() == s_tot.ket.m() {
                            -wigner_eckart_factor(s_tot, Spin::new(1, 0))
                                * red_second_subsystem_mel_factor(s_tot_mag, 1)
                                * red_spin_mel(s_tot.bra.pair.1)
                        } else {
                            0.
                        }
                    })
                })
            },
        ));
    }

    pub fn add_zeeman_n_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let i_tot_id = self.i_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [i_tot_id], |[i_tot]| {
                        let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                        if i_tot.bra.m() == i_tot.ket.m() {
                            -wigner_eckart_factor(i_tot, Spin::new(1, 0))
                                * red_first_subsystem_mel_factor(i_tot_mag, 1)
                                * red_spin_mel(i_tot.bra.pair.0)
                        } else {
                            0.
                        }
                    })
                })
            },
        ));
    }

    pub fn add_zeeman_n_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let i_tot_id = self.i_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [i_tot_id], |[i_tot]| {
                        let i_tot_mag = i_tot.map(|x| x.as_spin_pair_mag());

                        if i_tot.bra.m() == i_tot.ket.m() {
                            -wigner_eckart_factor(i_tot, Spin::new(1, 0))
                                * red_second_subsystem_mel_factor(i_tot_mag, 1)
                                * red_spin_mel(i_tot.bra.pair.1)
                        } else {
                            0.
                        }
                    })
                })
            },
        ));
    }
}

impl CoupledFTotDiatomBasis {
    pub fn add_hyperfine_a(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }

    pub fn add_hyperfine_b(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_e_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_e_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_n_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_n_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let f_tot_id = self.f_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_tot_id], |[f_tot]| {
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
                })
            },
        ));
    }
}

impl CoupledDiatomBasis {
    pub fn add_hyperfine_a(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }

    pub fn add_hyperfine_b(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_e_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_e_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_n_a(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }

    pub fn add_zeeman_n_b(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let fl_tot_id = self.fl_tot;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [fl_tot_id], |[fl_tot]| {
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
                })
            },
        ));
    }
}

#[cfg(test)]
mod tests {
    use cc_math_utils::assert_approx_eq;
    use coupled_chan::coupling::SystemParams;
    use hilbert_space::space::SpaceBasis;
    use spin_algebra::{
        hi32,
        hu32,
    };

    use crate::{
        OrbitalBasisElements,
        OrbitalRecipe,
        atom_basis::AtomRecipe,
        diatom_basis::{
            CoupledFDiatomBasis,
            DiatomRecipe,
            UncoupledDiatomBasis,
        },
    };

    use super::*;

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
        let b = 0.8;
        let atom_a_hifi = 1.;
        let atom_b_hifi = 0.5;
        let atom_a_gamma_e = 2.;
        let atom_b_gamma_e = 1.;
        let atom_a_gamma_n = 3.;
        let atom_b_gamma_n = 1.5;

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = UncoupledDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| {
            diatom.filter(|((s1, i1), (s2, i2), l)| s1.m + i1.m + s2.m + i2.m + l.m == hi32!(3 / 2))(x)
        });

        let mut h = HamiltonianConstructor::new(
            OrbitalBasisElements::from_orbital(elements, &diatom.l),
            SystemParams::default(),
        );
        h.add_param(("B", b));
        diatom.atom_a.add_hyperfine(&mut h, "atom_a", atom_a_hifi);
        diatom.atom_a.add_zeeman_e(&mut h, "B", "atom_a", atom_a_gamma_e);
        diatom.atom_a.add_zeeman_n(&mut h, "B", "atom_a", atom_a_gamma_n);

        diatom.atom_b.add_hyperfine(&mut h, "atom_b", atom_b_hifi);
        diatom.atom_b.add_zeeman_e(&mut h, "B", "atom_b", atom_b_gamma_e);
        diatom.atom_b.add_zeeman_n(&mut h, "B", "atom_b", atom_b_gamma_n);

        let h = h.construct();
        let asymptote_uncoupled = &h.asymptote().levels().asymptote;

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledFDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|(f1, f2, l)| f1.m() + f2.m() + l.m == hi32!(3 / 2))(x));

        let mut h = HamiltonianConstructor::new(
            OrbitalBasisElements::from_orbital(elements, &diatom.l),
            SystemParams::default(),
        );
        h.add_param(("B", b));
        diatom.atom_a.add_hyperfine(&mut h, "atom_a", atom_a_hifi);
        diatom.atom_a.add_zeeman_e(&mut h, "B", "atom_a", atom_a_gamma_e);
        diatom.atom_a.add_zeeman_n(&mut h, "B", "atom_a", atom_a_gamma_n);

        diatom.atom_b.add_hyperfine(&mut h, "atom_b", atom_b_hifi);
        diatom.atom_b.add_zeeman_e(&mut h, "B", "atom_b", atom_b_gamma_e);
        diatom.atom_b.add_zeeman_n(&mut h, "B", "atom_b", atom_b_gamma_n);

        let h = h.construct();
        let asymptote_coupled_f = &h.asymptote().levels().asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_f, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledSIDiatomBasis::new(recipe, &mut basis);
        let elements =
            basis.get_filtered_basis(|x| diatom.filter(|(s_tot, i_tot, l)| s_tot.m() + i_tot.m() + l.m == hi32!(3 / 2))(x));

        let mut h = HamiltonianConstructor::new(
            OrbitalBasisElements::from_orbital(elements, &diatom.l),
            SystemParams::default(),
        );
        h.add_param(("B", b));
        diatom.add_hyperfine_a(&mut h, "atom_a", atom_a_hifi);
        diatom.add_zeeman_e_a(&mut h, "B", "atom_a", atom_a_gamma_e);
        diatom.add_zeeman_n_a(&mut h, "B", "atom_a", atom_a_gamma_n);

        diatom.add_hyperfine_b(&mut h, "atom_b", atom_b_hifi);
        diatom.add_zeeman_e_b(&mut h, "B", "atom_b", atom_b_gamma_e);
        diatom.add_zeeman_n_b(&mut h, "B", "atom_b", atom_b_gamma_n);

        let h = h.construct();
        let asymptote_coupled_si = &h.asymptote().levels().asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_si, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledFTotDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|(f_tot, l)| f_tot.m() + l.m == hi32!(3 / 2))(x));

        let mut h = HamiltonianConstructor::new(
            OrbitalBasisElements::from_orbital(elements, &diatom.l),
            SystemParams::default(),
        );
        h.add_param(("B", b));
        diatom.add_hyperfine_a(&mut h, "atom_a", atom_a_hifi);
        diatom.add_zeeman_e_a(&mut h, "B", "atom_a", atom_a_gamma_e);
        diatom.add_zeeman_n_a(&mut h, "B", "atom_a", atom_a_gamma_n);

        diatom.add_hyperfine_b(&mut h, "atom_b", atom_b_hifi);
        diatom.add_zeeman_e_b(&mut h, "B", "atom_b", atom_b_gamma_e);
        diatom.add_zeeman_n_b(&mut h, "B", "atom_b", atom_b_gamma_n);

        let h = h.construct();
        let asymptote_coupled_f_tot = &h.asymptote().levels().asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_f_tot, 1e-6);

        ///////////////////////

        let mut basis = SpaceBasis::default();
        let diatom = CoupledDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| diatom.filter(|fl_tot| fl_tot.m() == hi32!(3 / 2))(x));

        let mut h = HamiltonianConstructor::new(
            OrbitalBasisElements::new(elements, diatom.fl_tot, |x| x.pair.1),
            SystemParams::default(),
        );
        h.add_param(("B", b));
        diatom.add_hyperfine_a(&mut h, "atom_a", atom_a_hifi);
        diatom.add_zeeman_e_a(&mut h, "B", "atom_a", atom_a_gamma_e);
        diatom.add_zeeman_n_a(&mut h, "B", "atom_a", atom_a_gamma_n);

        diatom.add_hyperfine_b(&mut h, "atom_b", atom_b_hifi);
        diatom.add_zeeman_e_b(&mut h, "B", "atom_b", atom_b_gamma_e);
        diatom.add_zeeman_n_b(&mut h, "B", "atom_b", atom_b_gamma_n);

        let h = h.construct();
        let asymptote_coupled_fl_tot = &h.asymptote().levels().asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled_fl_tot, 1e-6);
    }
}
