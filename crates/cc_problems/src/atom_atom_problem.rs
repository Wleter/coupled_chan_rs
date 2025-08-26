use coupled_chan::{
    Interaction, Operator,
    constants::units::{
        Quantity,
        atomic_units::{AuEnergy, AuMass},
    },
    coupling::{AngularBlocks, Asymptote, RedCoupling, VanishingCoupling, masked::Masked, pair::Pair},
};
use hilbert_space::{
    Parity,
    dyn_space::{SpaceBasis, SubspaceBasis},
    filter_space, operator_diag_mel, operator_mel, operator_transform_mel,
};
use spin_algebra::{
    Spin, SpinOps, get_spin_basis, get_summed_spin_basis,
    half_integer::{HalfI32, HalfU32},
};

use crate::CoupledProblem;

pub struct MagneticField(pub f64);

#[derive(Clone)]
pub struct AtomAtomBuilder<P: Interaction, V: Interaction> {
    pub s1: HalfU32,
    pub s2: HalfU32,

    pub i1: HalfU32,
    pub i2: HalfU32,

    pub projection: HalfI32,
    pub parity: Parity,

    pub gamma_e1: f64,
    pub gamma_i1: f64,
    pub a_hifi1: f64,

    pub gamma_e2: f64,
    pub gamma_i2: f64,
    pub a_hifi2: f64,

    pub singlet: P,
    pub triplet: V,

    pub mass: Quantity<AuMass>,
    pub energy: Quantity<AuEnergy>,

    pub entrance: usize,
}

impl<P: Interaction, V: Interaction> AtomAtomBuilder<P, V> {
    pub fn build(self, magnetic_field: MagneticField) -> CoupledProblem<impl VanishingCoupling> {
        let magnetic_field = magnetic_field.0;

        let mut space = SpaceBasis::default();
        let s1 = space.push_subspace(SubspaceBasis::new(get_spin_basis(self.s1)));
        let i1 = space.push_subspace(SubspaceBasis::new(get_spin_basis(self.i1)));

        let s2 = space.push_subspace(SubspaceBasis::new(get_spin_basis(self.s2)));
        let i2 = space.push_subspace(SubspaceBasis::new(get_spin_basis(self.i2)));

        let basis_sep = filter_space!(dyn space, |[s1: Spin, i1: Spin, s2: Spin, i2: Spin]| {
            s1.m + i1.m + s2.m + i2.m == self.projection
        });

        let mut zeeman: Operator = operator_diag_mel!(dyn &basis_sep, [s1], |[s: Spin]| {
            -self.gamma_e1 * magnetic_field * s.m.value()
        });
        zeeman += operator_diag_mel!(dyn &basis_sep, [s2], |[s: Spin]| {
            -self.gamma_e2 * magnetic_field * s.m.value()
        });
        zeeman += operator_diag_mel!(dyn &basis_sep, [i1], |[i: Spin]| {
            -self.gamma_i1 * magnetic_field * i.m.value()
        });
        zeeman += operator_diag_mel!(dyn &basis_sep, [i2], |[i: Spin]| {
            -self.gamma_i2 * magnetic_field * i.m.value()
        });

        let mut hifi: Operator = operator_mel!(dyn &basis_sep, [s1, i1], |[s: Spin, i: Spin]| {
            self.a_hifi1 * SpinOps::dot(s, i)
        });
        hifi += operator_mel!(dyn &basis_sep, [s2, i2], |[s: Spin, i: Spin]| {
            self.a_hifi2 * SpinOps::dot(s, i)
        });

        let mut space = SpaceBasis::default();
        let s = space.push_subspace(SubspaceBasis::new(get_summed_spin_basis(self.s1, self.s2)));
        let i = space.push_subspace(SubspaceBasis::new(get_summed_spin_basis(self.i1, self.i2)));
        let s_max = self.s1 + self.s2;
        let i_max = self.i1 + self.i2;

        let basis = filter_space!(dyn &space, |[s: Spin, i: Spin]| {
            s.m + i.m == self.projection && match self.parity {
                Parity::All => true,
                Parity::Even => (s_max + i_max).double_value() % 4 != (s.s + i.s).double_value() % 4,
                Parity::Odd => (s_max + i_max).double_value() % 4 == (s.s + i.s).double_value() % 4,
            }
        });

        let transf = operator_transform_mel!(
            dyn &basis_sep, [s1, s2, i1, i2],
            dyn &basis, [s, i],
            |[s1: Spin, s2: Spin, i1: Spin, i2: Spin], [s: Spin, i: Spin]| {
                SpinOps::clebsch_gordan(s1, s2, s) * SpinOps::clebsch_gordan(i1, i2, i)
            }
        );

        let hifi = hifi.transform(&transf);
        let zeeman = zeeman.transform(&transf);

        let singlet_masking = operator_diag_mel!(dyn &basis, [s], |[s: Spin]| {
            if s.s == 0 { 1. } else { 0. }
        });
        let singlet = Masked::new(self.singlet, singlet_masking);

        let triplet_masking = operator_diag_mel!(dyn &basis, [s], |[s: Spin]| {
            if s.s == 0 { 0. } else { 1. }
        });
        let triplet = Masked::new(self.triplet, triplet_masking);

        let coupling = Pair::new(singlet, triplet);

        let angular_blocks = AngularBlocks {
            l: vec![0],
            angular_blocks: vec![hifi + zeeman],
        };
        let asymptote = Asymptote::new_angular_blocks(self.mass, self.energy, angular_blocks, self.entrance);

        let red_coupling = RedCoupling::new(coupling, asymptote);

        CoupledProblem { red_coupling }
    }
}

#[cfg(test)]
mod tests {
    use coupled_chan::{
        Operator,
        composite_int::CompositeInt,
        constants::{
            BOHR_MAG, G_FACTOR,
            units::{
                Quantity,
                atomic_units::{AuEnergy, AuMass, Dalton, Kelvin, MHz},
            },
        },
        dispersion::Dispersion,
        propagator::{Boundary, Direction, Propagator, step_strategy::LocalWavelengthStep},
        ratio_numerov::{RatioNumerov, get_s_matrix},
    };
    use hilbert_space::Parity;
    use spin_algebra::{hi32, hu32};

    use crate::atom_atom_problem::{AtomAtomBuilder, MagneticField};

    #[test]
    pub fn test_li2_feshbach() {
        let singlet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(1.112e7, -12)]);

        let triplet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(2.19348e8, -12)]);

        let builder = AtomAtomBuilder {
            s1: hu32!(1 / 2),
            s2: hu32!(1 / 2),
            i1: hu32!(1),
            i2: hu32!(1),
            projection: hi32!(0),
            parity: Parity::Odd,
            gamma_e1: -G_FACTOR * BOHR_MAG,
            gamma_i1: 0.,
            a_hifi1: Quantity(228.2 / 1.5, MHz).to(AuEnergy).value(),
            gamma_e2: -G_FACTOR * BOHR_MAG,
            gamma_i2: 0.,
            a_hifi2: Quantity(228.2 / 1.5, MHz).to(AuEnergy).value(),
            singlet,
            triplet,
            mass: Quantity(6.015122 / 2., Dalton).to(AuMass),
            energy: Quantity(1e-7, Kelvin).to(AuEnergy),
            entrance: 0,
        };

        let problem = builder.clone().build(MagneticField(0.));
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 500.);
        let boundary = Boundary {
            r_start: 4.,
            direction: Direction::Outwards,
            value: Operator::new(1e-50 * &problem.red_coupling.id.0),
            derivative: Operator::new(1. * &problem.red_coupling.id.0),
        };
        let mut numerov = RatioNumerov::new(&problem.red_coupling, step_strategy.into(), boundary);

        let sol = numerov.propagate_to(1.5e3);
        let s_matrix = get_s_matrix(&sol, &problem.red_coupling);

        println!(
            "{:?}\n{}\n{:?}",
            s_matrix.s_matrix(),
            s_matrix.get_scattering_length(),
            s_matrix.get_inelastic_cross_sect()
        );
    }
}
