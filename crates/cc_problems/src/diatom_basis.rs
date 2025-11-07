use coupled_chan::{
    Interaction, Operator,
    constants::{Gauss, Quantity},
    coupling::{Asymptote, RedCoupling, masked::Masked, pair::Pair},
    scaled_interaction::ScaledInteraction,
};
use hilbert_space::{cast_variant, dyn_space::SpaceBasis, operator_diag_mel, operator_mel};
use serde::{Deserialize, Serialize};
use spin_algebra::{Spin, half_integer::HalfI32, hu32};

use crate::{
    AngularBasisElements, AngularMomentum, Hamiltonian, Structure,
    atom_structure::{AtomBasis, AtomBasisRecipe, AtomStructure},
    operator_mel::{singlet_projection_uncoupled, triplet_projection_uncoupled},
    system_structure::{AngularBasis, SystemParams},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DiatomBasisRecipe {
    pub atom_a: AtomBasisRecipe,
    pub atom_b: AtomBasisRecipe,
    pub l_max: AngularMomentum,

    pub tot_projection: HalfI32,
}

#[derive(Clone, Debug)]
pub struct DiatomBasis {
    pub atom_a: AtomBasis,
    pub atom_b: AtomBasis,
    pub angular: AngularBasis,

    basis: AngularBasisElements,
}

impl DiatomBasis {
    pub fn new(recipe: DiatomBasisRecipe) -> Self {
        let mut basis = SpaceBasis::default();
        let atom_a = AtomBasis::new(recipe.atom_a, &mut basis);
        let atom_b = AtomBasis::new(recipe.atom_b, &mut basis);
        let angular = AngularBasis::new(recipe.l_max, &mut basis);

        let basis = basis.get_filtered_basis(|element| {
            let s_a = cast_variant!(dyn element[atom_a.s], Spin);
            let i_a = cast_variant!(dyn element[atom_a.i], Spin);
            let s_b = cast_variant!(dyn element[atom_b.s], Spin);
            let i_b = cast_variant!(dyn element[atom_b.i], Spin);

            s_a.m + i_a.m + s_b.m + i_b.m == recipe.tot_projection
        });

        let basis = AngularBasisElements::new_angular(basis, &angular);

        Self {
            atom_a,
            atom_b,
            angular,
            basis,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PotentialCurve<P: Interaction> {
    pub potential: ScaledInteraction<P>,
    pub operator: Operator,
}

impl<P: Interaction> PotentialCurve<P> {
    pub fn new_triplet(triplet: P, elements: &AngularBasisElements, atom_a: &AtomBasis, atom_b: &AtomBasis) -> Self {
        let operator = operator_mel!(dyn elements.full_basis, [atom_a.s, atom_b.s], |[s1: Spin, s2: Spin]| {
            triplet_projection_uncoupled(s1, s2)
        });

        Self {
            potential: ScaledInteraction::new(triplet),
            operator,
        }
    }

    pub fn new_singlet(singlet: P, elements: &AngularBasisElements, atom_a: &AtomBasis, atom_b: &AtomBasis) -> Self {
        let operator = operator_mel!(dyn elements.full_basis, [atom_a.s, atom_b.s], |[s1: Spin, s2: Spin]| {
            singlet_projection_uncoupled(s1, s2)
        });

        Self {
            potential: ScaledInteraction::new(singlet),
            operator,
        }
    }

    pub fn new_triplet_coupled(triplet: P, elements: &AngularBasisElements, atom_pair: &AtomBasis) -> Self {
        let operator = operator_diag_mel!(dyn elements.full_basis, [atom_pair.s], |[s: Spin]| {
            if s.s == hu32!(1) { 1. } else { 0. }
        });

        Self {
            potential: ScaledInteraction::new(triplet),
            operator,
        }
    }

    pub fn new_singlet_coupled(singlet: P, elements: &AngularBasisElements, atom_pair: &AtomBasis) -> Self {
        let operator = operator_diag_mel!(dyn elements.full_basis, [atom_pair.s], |[s: Spin]| {
            if s.s == hu32!(0) { 1. } else { 0. }
        });

        Self {
            potential: ScaledInteraction::new(singlet),
            operator,
        }
    }
}

impl<P: Interaction + Clone> PotentialCurve<P> {
    pub fn hamiltonian(&self) -> Masked<ScaledInteraction<P>> {
        Masked::new(self.potential.clone(), self.operator.clone())
    }
}

#[derive(Clone, Debug)]
pub struct AlkaliDiatom<T, S>
where
    T: Interaction,
    S: Interaction,
{
    pub system_params: SystemParams,
    pub atom_a: AtomStructure,
    pub atom_b: AtomStructure,

    pub triplet: PotentialCurve<T>,
    pub singlet: PotentialCurve<S>,

    pub basis: DiatomBasis,
}

impl<T, S> AlkaliDiatom<T, S>
where
    T: Interaction,
    S: Interaction,
{
    pub fn new(triplet: T, singlet: S, recipe: DiatomBasisRecipe) -> Self {
        assert!(recipe.atom_a.s == hu32!(1 / 2), "Expected open shell A atom");
        assert!(recipe.atom_b.s == hu32!(1 / 2), "Expected open shell B atom");

        let basis = DiatomBasis::new(recipe);

        Self {
            system_params: SystemParams::default(),
            atom_a: AtomStructure::new(&basis.basis, &basis.atom_a),
            atom_b: AtomStructure::new(&basis.basis, &basis.atom_b),
            triplet: PotentialCurve::new_triplet(triplet, &basis.basis, &basis.atom_a, &basis.atom_b),
            singlet: PotentialCurve::new_singlet(singlet, &basis.basis, &basis.atom_a, &basis.atom_b),
            basis,
        }
    }

    pub fn set_b_field(&mut self, b_field: Quantity<Gauss>) {
        self.atom_a.set_b_field(b_field);
        self.atom_b.set_b_field(b_field);
    }
}

impl<T, S> Structure for AlkaliDiatom<T, S>
where
    T: Interaction,
    S: Interaction,
{
    fn modify_parameter(&mut self, key: &str, value: serde_json::Value) -> anyhow::Result<()> {
        match key {
            key if key.starts_with("system_params.") => self.system_params.modify_parameter(&key[14..], value)?,
            key if key.starts_with("atom_a.") => self.atom_a.modify_parameter(&key[7..], value)?,
            key if key.starts_with("atom_b.") => self.atom_b.modify_parameter(&key[7..], value)?,
            "b_field" => self.set_b_field(serde_json::from_value(value)?),
            "triplet_scaling" => self.triplet.potential.scale(serde_json::from_value(value)?),
            "singlet_scaling" => self.singlet.potential.scale(serde_json::from_value(value)?),
            _ => anyhow::bail!("Could not find {key} in AlkaliDiatom"),
        }

        Ok(())
    }
}

impl<T, S> Hamiltonian for AlkaliDiatom<T, S>
where
    T: Interaction + Clone,
    S: Interaction + Clone,
{
    type Coupling = Pair<Masked<ScaledInteraction<T>>, Masked<ScaledInteraction<S>>>;
    type WMatrix = RedCoupling<Self::Coupling>;

    fn asymptote(&self) -> Asymptote {
        let angular_blocks = self.atom_a.hamiltonian() + self.atom_b.hamiltonian();

        Asymptote::new_angular_blocks(
            self.system_params.mass,
            self.system_params.energy,
            angular_blocks,
            self.system_params.entrance_channel,
        )
    }

    fn coupling(&self) -> Self::Coupling {
        Pair::new(self.triplet.hamiltonian(), self.singlet.hamiltonian())
    }

    fn w_matrix(&self) -> Self::WMatrix {
        RedCoupling::new(self.coupling(), self.asymptote())
    }
}
