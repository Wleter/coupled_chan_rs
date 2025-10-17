use coupled_chan::{
    Operator,
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
    dyn_space::{BasisElementsRef, BasisId, SpaceBasis, SubspaceBasis},
    operator_diag_mel, operator_mel,
};
use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

use crate::AngularBasisElements;

#[derive(Clone, Copy, Debug, Default)]
pub struct AtomStructureRecipe {
    pub s: HalfU32,
    pub i: HalfU32,
}

#[derive(Clone, Debug)]
pub struct AtomStructureBuilder {
    pub s: BasisId,
    pub i: BasisId,
}

impl AtomStructureBuilder {
    pub fn new(recipe: AtomStructureRecipe, space_basis: &mut SpaceBasis) -> Self {
        let s = get_spin_basis(recipe.s);
        let i = get_spin_basis(recipe.i);

        let s = space_basis.push_subspace(SubspaceBasis::new(s));
        let i = space_basis.push_subspace(SubspaceBasis::new(i));

        Self { s, i }
    }

    pub fn zeeman_n(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.i], |[i: Spin]| {
            -i.m.value()
        })
    }

    pub fn zeeman_e(&self, basis: &BasisElementsRef) -> Operator {
        operator_diag_mel!(dyn basis, [self.s], |[s: Spin]| {
            -s.m.value()
        })
    }

    pub fn hyperfine(&self, basis: &BasisElementsRef) -> Operator {
        operator_mel!(dyn basis, [self.s, self.i], |[s: Spin, i: Spin]| {
            SpinOps::dot(s, i)
        })
    }

    pub fn build(self, full_basis: &AngularBasisElements) -> AtomStructure {
        AtomStructure {
            hifi: full_basis.get_angular_blocks(|e| self.hyperfine(e)),
            zee_e: full_basis.get_angular_blocks(|e| self.zeeman_e(e)),
            zee_n: full_basis.get_angular_blocks(|e| self.zeeman_n(e)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AtomStructure {
    pub hifi: AngularBlocks,
    pub zee_e: AngularBlocks,
    pub zee_n: AngularBlocks,
}

impl AtomStructure {
    pub fn with_params(&self, params: &AtomStructureParams) -> AngularBlocks {
        let hifi = self.hifi.scale(params.a_hifi.value());
        let zee_i = self.zee_e.scale(params.gamma_e.value() * params.b_field.value());
        let zee_e = self.zee_n.scale(params.gamma_n.value() * params.b_field.value());

        hifi + zee_i + zee_e
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AtomStructureParams {
    pub gamma_e: Quantity<Frac<AuEnergy, Gauss>>,
    pub gamma_n: Quantity<Frac<AuEnergy, Gauss>>,
    pub a_hifi: Quantity<AuEnergy>,
    pub b_field: Quantity<Gauss>,
}

impl Default for AtomStructureParams {
    fn default() -> Self {
        Self {
            gamma_e: -G_FACTOR * BOHR_MAG,
            gamma_n: Default::default(),
            a_hifi: Default::default(),
            b_field: Default::default(),
        }
    }
}
