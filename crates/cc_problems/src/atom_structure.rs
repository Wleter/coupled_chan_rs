use coupled_chan::{
    constants::{
        BOHR_MAG, G_FACTOR,
        units::{
            Frac, Quantity,
            atomic_units::{AuEnergy, Gauss},
        },
    },
    coupling::AngularBlocks,
};
use hilbert_space::{
    dyn_space::{BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel, operator_mel,
};
use serde::{Deserialize, Serialize};
use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

use crate::AngularBasisElements;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct AtomBasisRecipe {
    pub s: HalfU32,
    pub i: HalfU32,
}

#[derive(Clone, Debug)]
pub struct AtomBasis {
    pub s: BasisId,
    pub i: BasisId,
}

impl AtomBasis {
    pub fn new(recipe: AtomBasisRecipe, space_basis: &mut SpaceBasis) -> Self {
        let s = get_spin_basis(recipe.s);
        let i = get_spin_basis(recipe.i);

        let s = space_basis.push_subspace(SubspaceBasis::new(s));
        let i = space_basis.push_subspace(SubspaceBasis::new(i));

        Self { s, i }
    }
}

#[derive(Clone, Debug)]
pub struct AtomStructure {
    pub zeeman_e: ZeemanSplitting,
    pub zeeman_n: ZeemanSplitting,
    pub hyperfine: HyperfineStructure,
}

impl AtomStructure {
    pub fn new(elements: &AngularBasisElements, atom_basis: &AtomBasis) -> Self {
        let mut zeeman_e = ZeemanSplitting::new(elements, atom_basis.s);
        zeeman_e.gamma = -G_FACTOR * BOHR_MAG;

        Self {
            zeeman_e,
            zeeman_n: ZeemanSplitting::new(elements, atom_basis.i),
            hyperfine: HyperfineStructure::new(elements, atom_basis),
        }
    }

    pub fn set_b_field(&mut self, b_field: Quantity<Gauss>) {
        self.zeeman_e.b_field = b_field;
        self.zeeman_n.b_field = b_field;
    }

    pub fn hamiltonian(&self) -> AngularBlocks {
        self.hyperfine.hamiltonian() + self.zeeman_e.hamiltonian() + self.zeeman_n.hamiltonian()
    }
}

#[derive(Clone, Debug)]
pub struct HyperfineStructure {
    pub a_hifi: Quantity<AuEnergy>,
    pub operator: AngularBlocks,
}

impl HyperfineStructure {
    pub fn new(elements: &AngularBasisElements, atom: &AtomBasis) -> Self {
        let operator = elements.get_angular_blocks(|basis| {
            operator_mel!(dyn basis, [atom.s, atom.i], |[s: Spin, i: Spin]| {
                SpinOps::dot(s, i)
            })
        });

        Self {
            a_hifi: Default::default(),
            operator,
        }
    }

    pub fn hamiltonian(&self) -> AngularBlocks {
        self.operator.scale(self.a_hifi.value())
    }
}

#[derive(Clone, Debug)]
pub struct ZeemanSplitting {
    pub b_field: Quantity<Gauss>,
    pub gamma: Quantity<Frac<AuEnergy, Gauss>>,
    pub operator: AngularBlocks,
}

impl ZeemanSplitting {
    pub fn new(elements: &AngularBasisElements, s: BasisId) -> Self {
        let operator = elements.get_angular_blocks(|basis| {
            operator_diag_mel!(dyn basis, [s], |[s: Spin]| {
                -s.m.value()
            })
        });

        Self {
            b_field: Default::default(),
            gamma: Default::default(),
            operator,
        }
    }

    pub fn hamiltonian(&self) -> AngularBlocks {
        self.operator.scale(self.gamma.value() * self.b_field.value())
    }
}
