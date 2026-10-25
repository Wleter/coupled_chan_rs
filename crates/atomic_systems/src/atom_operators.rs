use cc_system::hamiltonian_spec::OperatorSpec;
use cc_system::parameters::{
    ParamIds,
    ParameterRegistry,
    TypedParamId,
};
use cc_system::{
    Operator,
    param_ids,
};
use hilbert_space::space::BasisElementsRef;
use hilbert_space::{
    operator_diag_mel,
    operator_mel,
};
use serde::Deserialize;
use spin_algebra::ops::{
    red_first_subsystem_mel_factor,
    red_second_subsystem_mel_factor,
    red_spin_mel,
    wigner_eckart_factor,
};
use spin_algebra::{
    Spin,
    SpinLike,
};
use unit_systems::atomic_units;
use unit_systems::quantities::Scalar;
use unit_systems::quantities::phys_quantities::{
    Energy,
    MagneticDipole,
    MagneticField,
};

use crate::atom_basis::{
    CoupledAtomBasis,
    UncoupledAtomBasis,
};
use crate::operator_mel::dot_coupled;

pub(crate) type CouplingId = TypedParamId<Scalar<Energy>>;
pub(crate) type BFieldId = TypedParamId<Scalar<MagneticField>>;
pub(crate) type GFactorId = TypedParamId<Scalar<MagneticDipole>>;

pub const ELECTRON_G_FACTOR: f64 = -2.002_319_304_360_92;

pub fn electron_g_e() -> Scalar<MagneticDipole> {
    Scalar::new(ELECTRON_G_FACTOR, MagneticDipole, "mu_bohr")
}

#[derive(Debug, Clone, cc_system::Parameters, Deserialize)]
#[serde(default)]
pub struct AtomParams {
    pub a_hifi: Scalar<Energy>,
    pub g_e: Scalar<MagneticDipole>,
    pub g_n: Scalar<MagneticDipole>,
}

impl Default for AtomParams {
    fn default() -> Self {
        Self {
            a_hifi: Default::default(),
            g_e: electron_g_e(),
            g_n: Default::default(),
        }
    }
}

pub struct CouplingSpec<F: Fn(BasisElementsRef) -> Operator> {
    pub coupling: CouplingId,
    pub(crate) operator: F,
}

impl<F: Fn(BasisElementsRef) -> Operator> CouplingSpec<F> {
    pub fn new(coupling: CouplingId, operator: F) -> Self {
        Self { coupling, operator }
    }
}

impl<F: Fn(BasisElementsRef) -> Operator + Send + Sync> OperatorSpec for CouplingSpec<F> {
    fn build_params(&self) -> ParamIds {
        param_ids![]
    }

    fn matrix(&self, elements: BasisElementsRef, _params: &ParameterRegistry) -> Operator {
        (self.operator)(elements)
    }

    fn coupling_params(&self) -> ParamIds {
        param_ids![self.coupling.vanish()]
    }

    fn coupling(&self, params: &ParameterRegistry) -> f64 {
        let converter = atomic_units::units_converter();

        converter.scalar_value(params.get(self.coupling))
    }
}

pub struct ZeemanSpec<F: Fn(BasisElementsRef) -> Operator> {
    pub b_field: BFieldId,
    pub g_factor: GFactorId,
    pub(crate) operator: F,
}

impl<F: Fn(BasisElementsRef) -> Operator> ZeemanSpec<F> {
    pub fn new(b_field: BFieldId, g_factor: GFactorId, operator: F) -> Self {
        Self {
            b_field,
            g_factor,
            operator,
        }
    }
}

impl<F: Fn(BasisElementsRef) -> Operator + Send + Sync> OperatorSpec for ZeemanSpec<F> {
    fn build_params(&self) -> ParamIds {
        param_ids![]
    }

    fn matrix(&self, elements: BasisElementsRef, _params: &ParameterRegistry) -> Operator {
        (self.operator)(elements)
    }

    fn coupling_params(&self) -> ParamIds {
        param_ids![self.b_field.vanish(), self.g_factor.vanish()]
    }

    fn coupling(&self, params: &ParameterRegistry) -> f64 {
        let converter = atomic_units::units_converter();

        let b_field = converter.scalar_value(params.get(self.b_field));
        let g_factor = converter.scalar_value(params.get(self.g_factor));

        b_field * g_factor
    }
}

impl UncoupledAtomBasis {
    pub fn hifi(&self, a_hifi: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_id = self.s;
        let i_id = self.i;

        CouplingSpec {
            coupling: a_hifi,
            operator: move |b| operator_mel!(b, [s_id, i_id], |[s, i]| spin_algebra::ops::dot(s, i)),
        }
    }

    pub fn zeeman_e(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let s_id = self.s;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| operator_diag_mel!(b, [s_id], |[s]| -s.m().value()),
        }
    }

    pub fn zeeman_n(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let i_id = self.i;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| operator_diag_mel!(b, [i_id], |[i]| -i.m().value()),
        }
    }
}

impl CoupledAtomBasis {
    pub fn hifi(&self, a_hifi: CouplingId) -> CouplingSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_id = self.f;

        CouplingSpec {
            coupling: a_hifi,
            operator: move |b| operator_diag_mel!(b, [f_id], |[f]| dot_coupled(f)),
        }
    }

    pub fn zeeman_e(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_id = self.f;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_id], |[f]| {
                    let f_mag = f.map(|x| x.as_spin_pair_mag());

                    if f.bra.m() == f.ket.m() {
                        -wigner_eckart_factor(f, Spin::new(1, 0))
                            * red_first_subsystem_mel_factor(f_mag, 1)
                            * red_spin_mel(f.bra.pair.0)
                    } else {
                        0.
                    }
                })
            },
        }
    }

    pub fn zeeman_n(
        &self,
        b_field: BFieldId,
        g_factor: GFactorId,
    ) -> ZeemanSpec<impl Fn(BasisElementsRef) -> Operator + use<>> {
        let f_id = self.f;

        ZeemanSpec {
            b_field,
            g_factor,
            operator: move |b| {
                operator_mel!(b, [f_id], |[f]| {
                    let f_mag = f.map(|x| x.as_spin_pair_mag());

                    if f.bra.m() == f.ket.m() {
                        -wigner_eckart_factor(f, Spin::new(1, 0))
                            * red_second_subsystem_mel_factor(f_mag, 1)
                            * red_spin_mel(f.bra.pair.1)
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
    use cc_system::System;
    use cc_system::hamiltonian_spec::{
        DynOperatorSpec,
        HamiltonianSpec,
    };
    use cc_system::orbital_basis::OrbitalBasisElements;
    use cc_system::parameters::Parameters;
    use hilbert_space::space::SpaceBasis;
    use spin_algebra::{
        hi32,
        hu32,
    };

    use crate::atom_basis::AtomRecipe;

    use super::*;

    #[derive(Clone, Parameters, Default)]
    struct Params {
        b_field: Scalar<MagneticField>,
        #[parameter(nested)]
        atom: AtomParams,
    }

    #[test]
    fn test_atom_hamiltonian_terms() {
        let recipe = AtomRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
            name: "test".into(),
        };
        let params = Params {
            b_field: Scalar::new(80.0, MagneticField, "gauss"),
            atom: AtomParams {
                a_hifi: Scalar::new(1., Energy, "GHz"),
                g_e: Scalar::new(-2., MagneticDipole, "mu_bohr"),
                g_n: Scalar::new(1., MagneticDipole, "mu_nuclear"),
            },
        };
        let ids = Params::ids();

        let mut basis = SpaceBasis::default();
        let atom = UncoupledAtomBasis::new(&recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| atom.filter(|(s, i)| s.m + i.m == hi32!(1))(x));
        let basis = OrbitalBasisElements::new_implicit(elements, 0);

        let hifi = atom.hifi(ids.atom.a_hifi);
        let zeeman_e = atom.zeeman_e(ids.b_field, ids.atom.g_e);
        let zeeman_n = atom.zeeman_n(ids.b_field, ids.atom.g_n);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("hifi", DynOperatorSpec::new(hifi)),
            ("zeeman_e", DynOperatorSpec::new(zeeman_e)),
            ("zeeman_n", DynOperatorSpec::new(zeeman_n)),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_uncoupled = system.angular_blocks().diagonalized().0.asymptote;

        let mut basis = SpaceBasis::default();
        let atom = CoupledAtomBasis::new(&recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| atom.filter(|f| f.m() == hi32!(1))(x));
        let basis = OrbitalBasisElements::new_implicit(elements, 0);

        let hifi = atom.hifi(ids.atom.a_hifi);
        let zeeman_e = atom.zeeman_e(ids.b_field, ids.atom.g_e);
        let zeeman_n = atom.zeeman_n(ids.b_field, ids.atom.g_n);

        let mut hamiltonian_spec = HamiltonianSpec::new(basis);
        hamiltonian_spec.add_operators([
            ("hifi", DynOperatorSpec::new(hifi)),
            ("zeeman_e", DynOperatorSpec::new(zeeman_e)),
            ("zeeman_n", DynOperatorSpec::new(zeeman_n)),
        ]);
        let system = System::new(hamiltonian_spec, params.registry());
        let asymptote_coupled = system.angular_blocks().diagonalized().0.asymptote;

        assert_approx_eq!(iter => asymptote_uncoupled, asymptote_coupled, 1e-6);
    }
}
