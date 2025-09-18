use coupled_chan::{
    Interaction, Operator,
    coupling::{Asymptote, RedCoupling, WMatrix, masked::Masked, pair::Pair},
};
use hilbert_space::{
    cast_variant,
    dyn_space::{SpaceBasis, SubspaceElement},
    operator_mel,
};
use spin_algebra::{Spin, half_integer::HalfI32};

use crate::{
    AngularBasisElements, AngularMomentum,
    atom_structure::{AtomStructure, AtomStructureBuilder, AtomStructureParams, AtomStructureRecipe},
    operator_mel::{singlet_projection_uncoupled, triplet_projection_uncoupled},
    system_structure::{SystemParams, SystemStructure},
};

#[derive(Clone, Debug, Default)]
pub struct AlkaliDiatomRecipe {
    pub atom_a: AtomStructureRecipe,
    pub atom_b: AtomStructureRecipe,
    pub l_max: AngularMomentum,

    pub tot_projection: HalfI32,
}

#[derive(Clone, Debug)]
pub struct AlkaliDiatomParams<T, S> {
    pub atom_a: AtomStructureParams,
    pub atom_b: AtomStructureParams,
    pub system: SystemParams,

    pub triplet: T,
    pub singlet: S,
}

#[derive(Clone, Debug)]
pub struct AlkaliDiatomBuilder {
    pub atom_a: AtomStructureBuilder,
    pub atom_b: AtomStructureBuilder,
    pub system: SystemStructure,

    tot_projection: HalfI32,

    basis: SpaceBasis,
}

impl AlkaliDiatomBuilder {
    pub fn new(recipe: AlkaliDiatomRecipe) -> Self {
        let mut basis = SpaceBasis::default();
        let atom_a = AtomStructureBuilder::new(recipe.atom_a, &mut basis);
        let atom_b = AtomStructureBuilder::new(recipe.atom_b, &mut basis);
        let system = SystemStructure::new(recipe.l_max, &mut basis);

        Self {
            atom_a,
            atom_b,
            tot_projection: recipe.tot_projection,
            system,
            basis,
        }
    }

    pub fn build(self) -> AlkaliDiatom {
        let basis = self.basis.get_filtered_basis(|x| self.filter(x));
        let basis = AngularBasisElements::new_system(basis, &self.system);

        let triplet = operator_mel!(dyn basis.full_basis, [self.atom_a.s, self.atom_b.s], |[s1: Spin, s2: Spin]| {
            triplet_projection_uncoupled(s1, s2)
        });

        let singlet = operator_mel!(dyn basis.full_basis, [self.atom_a.s, self.atom_b.s], |[s1: Spin, s2: Spin]| {
            singlet_projection_uncoupled(s1, s2)
        });

        AlkaliDiatom {
            atom_a: self.atom_a.build(&basis),
            atom_b: self.atom_b.build(&basis),
            basis,
            triplet,
            singlet,
        }
    }

    pub fn filter(&self, element: &[&SubspaceElement]) -> bool {
        let s_a = cast_variant!(dyn element[self.atom_a.s.0 as usize], Spin);
        let i_a = cast_variant!(dyn element[self.atom_a.i.0 as usize], Spin);
        let s_b = cast_variant!(dyn element[self.atom_b.s.0 as usize], Spin);
        let i_b = cast_variant!(dyn element[self.atom_b.i.0 as usize], Spin);

        s_a.m + i_a.m + s_b.m + i_b.m == self.tot_projection
    }
}

#[derive(Clone, Debug)]
pub struct AlkaliDiatom {
    pub basis: AngularBasisElements,
    atom_a: AtomStructure,
    atom_b: AtomStructure,

    triplet: Operator,
    singlet: Operator,
}

impl AlkaliDiatom {
    pub fn with_params<T, S>(&self, params: &AlkaliDiatomParams<T, S>) -> impl WMatrix
    where
        T: Interaction + Clone,
        S: Interaction + Clone,
    {
        let asymptote = self.atom_a.with_params(&params.atom_a) + self.atom_b.with_params(&params.atom_b);
        let asymptote = Asymptote::new_angular_blocks(
            params.system.mass,
            params.system.energy,
            asymptote,
            params.system.entrance_channel,
        );

        let triplet = Masked::new(params.triplet.clone(), self.triplet.clone());
        let singlet = Masked::new(params.singlet.clone(), self.singlet.clone());

        RedCoupling::new(Pair::new(triplet, singlet), asymptote)
    }
}
