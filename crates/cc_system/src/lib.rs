use std::collections::HashSet;

use cc_qol_utils::{
    Composite,
    hash_vec::HashVec,
};
use coupled_chan::{
    DynInteraction,
    coupling::{
        AngularBlocks,
        masked::Masked,
    },
};

use crate::{
    hamiltonian_spec::{
        DynOperatorSpec,
        DynPotentialSpec,
        HamiltonianSpec,
    },
    orbital_basis::OrbitalBasisElements,
    param_modifications::ParamModifications,
    parameters::{
        ParamId,
        ParamIds,
        ParameterRegistry,
    },
};

pub mod hamiltonian_spec;
pub mod orbital_basis;
pub mod param_modifications;
pub mod parameters;

pub use smallvec;
pub use cc_derive::Parameters;
pub type Operator = hilbert_space::operator::Operator<coupled_chan::multi_channel::Matrix>;

pub type CouplingPotential = Masked<DynInteraction>;
pub type Coupling = Composite<CouplingPotential>;

#[derive(Clone)]
pub struct System {
    registry: ParameterRegistry,

    basis: OrbitalBasisElements,
    pub operator_specs: HashVec<Box<str>, DynOperatorSpec>,
    pub potential_specs: HashVec<Box<str>, DynPotentialSpec>,

    pub operators: HashVec<Box<str>, (f64, AngularBlocks)>,
    pub potentials: HashVec<Box<str>, CouplingPotential>,
}

impl System {
    pub fn new(hamiltonian_spec: HamiltonianSpec, registry: ParameterRegistry) -> Self {
        let basis = hamiltonian_spec.basis;
        let operator_specs = HashVec::from_hashmap(hamiltonian_spec.operators);
        let potential_specs = HashVec::from_hashmap(hamiltonian_spec.potentials);

        let operators =operator_specs.mapped(|x| 
            (
                x.coupling(&registry), 
                basis.get_angular_blocks(|_, b| x.matrix(b, &registry)
            )
        ));

        let potentials = potential_specs.mapped(|x| {
            let masking = x.coupling_masking(basis.full_basis.as_ref(), &registry);
            let curve = x.curve(&registry);

            Masked::new(curve, masking.0)
        });

        Self {
            registry,
            basis,
            operator_specs,
            potential_specs,
            operators,
            potentials,
        }
    }

    pub fn param_registry(&self) -> &ParameterRegistry {
        &self.registry
    }

    pub fn modify_params(&mut self, modifications: ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds>) {
        let ids = (modifications.modification)(&mut self.registry);

        self.update_operators(&ids);
        self.update_potentials(&ids);
    }

    pub fn angular_blocks(&self) -> AngularBlocks {
        let mut iterator = self.operators.vec.iter();
        let mut acc = iterator
            .next()
            .map(|(f, m)| m.scale(*f))
            .unwrap_or(self.basis.zero_angular_block());

        for (f, m) in iterator {
            if *f == 0.0 {
                continue;
            }

            acc += m.scale(*f)
        }

        acc
    }

    pub fn coupling(&self) -> Coupling {
        if self.potentials.vec.is_empty() {
            return Composite::new(vec![]);
        }

        let zeros = Operator::zeros(self.potentials.vec[0].masking.nrows()).0;
        let filtered = self.potentials.vec.iter().filter(|x| x.masking != zeros).cloned().collect();

        Composite::new(filtered)
    }

    fn update_operators(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut coupling_queue = HashSet::new();

        let specs = &self.operator_specs.vec;
        let ops = &mut self.operators.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if ops.build_params().iter().find(|x| *x == m).is_some() {
                    rebuild_queue.insert(i);
                }
                if ops.coupling_params().iter().find(|x| *x == m).is_some() {
                    coupling_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            ops[ops_id].1 = self.basis.get_angular_blocks(|_, b| specs[ops_id].matrix(b, &self.registry));
        }

        for ops_id in coupling_queue {
            ops[ops_id].0 = specs[ops_id].coupling(&self.registry);
        }
    }

    fn update_potentials(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut curve_queue = HashSet::new();

        let specs = &self.potential_specs.vec;
        let potentials = &mut self.potentials.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if ops.build_params().iter().find(|x| *x == m).is_some() {
                    rebuild_queue.insert(i);
                }
                if ops.curve_params().iter().find(|x| *x == m).is_some() {
                    curve_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            let rebuilt = specs[ops_id].coupling_masking(self.basis.full_basis.as_ref(), &self.registry);

            potentials[ops_id].masking = rebuilt.0;
        }

        for ops_id in curve_queue {
            potentials[ops_id].interaction = specs[ops_id].curve(&self.registry);
        }
    }

    pub fn basis(&self) -> &OrbitalBasisElements {
        &self.basis
    }
}
