use anyhow::bail;
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
use serde_json::Value;
use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

use crate::{AngularBasisElements, Structure};

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

impl Structure for AtomStructure {
    fn modify_parameter(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        match key {
            "gamma_e" => self.zeeman_e.gamma = serde_json::from_value(value)?,
            "gamma_n" => self.zeeman_n.gamma = serde_json::from_value(value)?,
            "a_hifi" => self.hyperfine.a_hifi = serde_json::from_value(value)?,
            "b_field" => self.set_b_field(serde_json::from_value(value)?),
            _ => bail!("Did not find {key} in AtomStructure"),
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use hilbert_space::dyn_space::SpaceBasis;
    use serde_json::json;
    use spin_algebra::hu32;

    use crate::{AngularMomentum, system_structure::AngularBasis};

    use super::*;

    #[test]
    pub fn test_atom_structure_modifier() {
        let recipe = super::AtomBasisRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1),
        };
        let mut basis = SpaceBasis::default();
        let atom = AtomBasis::new(recipe, &mut basis);
        let angular = AngularBasis::new(AngularMomentum(0), &mut basis);
        let full_basis = basis.get_basis();
        let elements = AngularBasisElements::new_angular(full_basis, &angular);

        let mut structure = AtomStructure::new(&elements, &atom);
        let a_hifi = 1.;
        let b_field = 100.;
        let zeeman_n = 1.;
        let zeeman_e = -2.;

        let data = json!({
            "a_hifi": [a_hifi, "AuEnergy"],
            "b_field": [b_field, "Gauss"],
            "gamma_n": [zeeman_n, "AuEnergy/Gauss"],
            "gamma_e": [zeeman_e, "AuEnergy/Gauss"],
        });

        structure.modify_parameters(data).unwrap();
        assert_eq!(structure.hyperfine.a_hifi, a_hifi * AuEnergy);
        assert_eq!(structure.zeeman_e.b_field, b_field * Gauss);
        assert_eq!(structure.zeeman_n.b_field, b_field * Gauss);
        assert_eq!(structure.zeeman_n.gamma, zeeman_n * (AuEnergy / Gauss));
        assert_eq!(structure.zeeman_e.gamma, zeeman_e * (AuEnergy / Gauss));
    }
}
