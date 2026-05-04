use hilbert_space::{
    operator_diag_mel,
    operator_mel,
};
use spin_algebra::{
    Spin,
    SpinLike,
    ops::{
        red_first_subsystem_mel_factor,
        red_second_subsystem_mel_factor,
        red_spin_mel,
        wigner_eckart_factor,
    },
};

use crate::{
    atom_basis::{
        CoupledAtomBasis,
        UncoupledAtomBasis,
    },
    hamiltonian::{
        HamiltonianConstructor,
        HamiltonianTerm,
        TermRecipe,
    },
    hamiltonian_terms::{
        dot_coupled_term,
        dot_uncoupled_term,
    },
};

impl UncoupledAtomBasis {
    pub fn add_hyperfine(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let s = self.s;
        let i = self.i;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| dot_uncoupled_term(s, i),
        ));
    }

    pub fn add_zeeman_e(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let s_id = self.s;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| HamiltonianTerm::new(move |e| operator_diag_mel!(e, [s_id], |[s]| -s.m().value())),
        ));
    }

    pub fn add_zeeman_n(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let i_id = self.i;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| HamiltonianTerm::new(move |e| operator_diag_mel!(e, [i_id], |[i]| -i.m().value())),
        ));
    }
}

impl CoupledAtomBasis {
    pub fn add_hyperfine(&self, constructor: &mut HamiltonianConstructor, name_prefix: &str, a_hifi: f64) {
        let a_hifi_name: &str = &format!("{name_prefix}.a_hifi");
        constructor.add_param((a_hifi_name, a_hifi));

        let f = self.f;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.hifi"),
            [a_hifi_name],
            [],
            move |_| dot_coupled_term(f),
        ));
    }

    pub fn add_zeeman_e(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_e: f64) {
        let gamma_e_name = &format!("{name_prefix}.gamma_e");
        constructor.add_param((gamma_e_name, gamma_e));

        let f_id = self.f;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_e"),
            [b_field, gamma_e_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_id], |[f]| {
                        let f_mag = f.map(|x| x.as_spin_pair_mag());

                        if f.bra.m() == f.ket.m() {
                            -wigner_eckart_factor(f, Spin::new(1, 0))
                                * red_first_subsystem_mel_factor(f_mag, 1)
                                * red_spin_mel(f.bra.pair.0)
                        } else {
                            0.
                        }
                    })
                })
            },
        ));
    }

    pub fn add_zeeman_n(&self, constructor: &mut HamiltonianConstructor, b_field: &str, name_prefix: &str, gamma_n: f64) {
        let gamma_n_name = &format!("{name_prefix}.gamma_n");
        constructor.add_param((gamma_n_name, gamma_n));

        let f_id = self.f;
        constructor.add_term(TermRecipe::new_sized(
            &format!("{name_prefix}.zeeman_n"),
            [b_field, gamma_n_name],
            [],
            move |_| {
                HamiltonianTerm::new(move |e| {
                    operator_mel!(e, [f_id], |[f]| {
                        let f_mag = f.map(|x| x.as_spin_pair_mag());

                        if f.bra.m() == f.ket.m() {
                            -wigner_eckart_factor(f, Spin::new(1, 0))
                                * red_second_subsystem_mel_factor(f_mag, 1)
                                * red_spin_mel(f.bra.pair.1)
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
        atom_basis::AtomRecipe,
    };

    use super::*;

    #[test]
    fn test_atom_hamiltonian_terms() {
        let recipe = AtomRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
        };
        let b = 0.8;
        let a_hifi = 1.;
        let gamma_e = 2.;
        let gamma_n = 3.;

        let mut basis = SpaceBasis::default();
        let atom = UncoupledAtomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| atom.filter(|(s, i)| s.m + i.m == hi32!(1))(x));

        let mut h = HamiltonianConstructor::new(OrbitalBasisElements::new_implicit(elements, 0), SystemParams::default());
        h.add_param(("B", b));
        atom.add_hyperfine(&mut h, "atom", a_hifi);
        atom.add_zeeman_e(&mut h, "B", "atom", gamma_e);
        atom.add_zeeman_n(&mut h, "B", "atom", gamma_n);

        let h = h.construct();
        let asymptote_uncoupled = &h.asymptote().levels().asymptote;

        let mut basis = SpaceBasis::default();
        let atom = CoupledAtomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| atom.filter(|f| f.m() == hi32!(1))(x));

        let mut h = HamiltonianConstructor::new(OrbitalBasisElements::new_implicit(elements, 0), SystemParams::default());
        h.add_param(("B", b));
        atom.add_hyperfine(&mut h, "atom", a_hifi);
        atom.add_zeeman_e(&mut h, "B", "atom", gamma_e);
        atom.add_zeeman_n(&mut h, "B", "atom", gamma_n);

        let h = h.construct();
        let asymptote_coupled = &h.asymptote().levels().asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled, 1e-6);
    }
}
