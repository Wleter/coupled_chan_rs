use coupled_chan::{
    Interaction, Operator,
    coupling::{AngularBlocks, Asymptote, RedCoupling, WMatrix, masked::Masked, pair::Pair},
};
use hilbert_space::{
    Parity, cast_variant,
    dyn_space::{BasisElementsRef, SpaceBasis, SubspaceBasis, SubspaceElement},
    operator_diag_mel, operator_transform_mel,
};
use spin_algebra::{
    Spin, get_summed_spin_basis,
    half_integer::{HalfI32, HalfU32},
    hu32, wigner_3j,
};

use crate::{
    AngularBasisElements, AngularMomentum,
    atom_structure::{AtomStructure, AtomStructureBuilder, AtomStructureParams, AtomStructureRecipe},
    system_structure::{SystemParams, SystemStructure},
};

#[derive(Clone, Debug, Default)]
pub struct AlkaliHomoDiatomRecipe {
    pub atom: AtomStructureRecipe,
    pub l_max: AngularMomentum,

    pub tot_projection: HalfI32,
}

#[derive(Clone, Debug)]
pub struct AlkaliHomoDiatomParams<T, S> {
    pub atom: AtomStructureParams,
    pub system: SystemParams,

    pub triplet: T,
    pub singlet: S,
}

#[derive(Clone, Debug)]
pub struct AlkaliHomoDiatomBuilder {
    pub combined_structure: AtomStructureBuilder,
    pub system: SystemStructure,

    s_max: HalfU32,
    i_max: HalfU32,
    parity: Parity,
    tot_projection: HalfI32,

    basis: SpaceBasis,

    atom_a: AtomStructureBuilder,
    atom_b: AtomStructureBuilder,
    system_sep: SystemStructure,
    basis_sep: SpaceBasis,
}

impl AlkaliHomoDiatomBuilder {
    pub fn new(recipe: AlkaliHomoDiatomRecipe) -> Self {
        let mut basis = SpaceBasis::default();

        let s_tot = get_summed_spin_basis(recipe.atom.s, recipe.atom.s);
        let s_tot = basis.push_subspace(SubspaceBasis::new(s_tot));
        let i_tot = get_summed_spin_basis(recipe.atom.i, recipe.atom.i);
        let i_tot = basis.push_subspace(SubspaceBasis::new(i_tot));

        let system = SystemStructure::new(recipe.l_max, &mut basis);

        let mut basis_sep = SpaceBasis::default();
        let atom_a = AtomStructureBuilder::new(recipe.atom, &mut basis_sep);
        let atom_b = AtomStructureBuilder::new(recipe.atom, &mut basis_sep);
        let system_sep = SystemStructure::new(recipe.l_max, &mut basis_sep);

        let parity = if (recipe.atom.s + recipe.atom.i).double_value() % 2 == 0 {
            Parity::Even
        } else {
            Parity::Odd
        };

        Self {
            combined_structure: AtomStructureBuilder { s: s_tot, i: i_tot },
            s_max: recipe.atom.s + recipe.atom.s,
            i_max: recipe.atom.i + recipe.atom.i,

            tot_projection: recipe.tot_projection,
            parity,
            system,
            basis,

            atom_a,
            atom_b,
            system_sep,
            basis_sep,
        }
    }

    pub fn build(self) -> AlkaliHomoDiatom {
        let basis = self.basis.get_filtered_basis(|x| self.filter(x));
        let basis = AngularBasisElements::new_system(basis, &self.system);

        let basis_sep = self.basis_sep.get_filtered_basis(|x| self.filter_sep(x));
        let basis_sep = AngularBasisElements::new_system(basis_sep, &self.system_sep);

        let transformation = |e: BasisElementsRef<'_>, e_sep: BasisElementsRef<'_>| {
            operator_transform_mel!(
                dyn e_sep, [self.system_sep.l, self.atom_a.s, self.atom_a.i, self.atom_b.s, self.atom_b.i],
                dyn e, [self.system.l, self.combined_structure.s, self.combined_structure.i],
                |[l_sep: AngularMomentum, s1: Spin, i1: Spin, s2: Spin, i2: Spin],
                    [l: AngularMomentum, s_tot: Spin, i_tot: Spin]|
                {
                    if l == l_sep {
                        wigner_3j(s_tot.s, s1.s, s2.s, s_tot.m, s1.m, s2.m)
                            * wigner_3j(i_tot.s, i1.s, i2.s, i_tot.m, i1.m, i2.m)
                    } else {
                        0.
                    }
                }
            )
        };

        // todo! allow transformations for maximal projection values
        assert_eq!(
            basis.ls.len(),
            basis_sep.ls.len(),
            "Could not transform separated basis into combined basis, for maximal projection consider using All parity"
        );

        let transformation = AngularBlocks {
            l: basis.ls.iter().map(|a| a.0).collect(),
            angular_blocks: basis
                .angular_iter()
                .zip(basis_sep.angular_iter())
                .map(|(e, e_sep)| transformation(e_sep, e))
                .collect(),
        };

        let triplet = operator_diag_mel!(dyn basis.full_basis, [self.combined_structure.s], |[s: Spin]| {
            if s.s == hu32!(1) { 1. } else { 0. }
        });

        let singlet = operator_diag_mel!(dyn basis.full_basis, [self.combined_structure.s], |[s: Spin]| {
            if s.s == hu32!(1) { 0. } else { 1. }
        });

        let hifi = basis_sep.get_angular_blocks(|e| self.atom_a.hyperfine(e) + self.atom_b.hyperfine(e));
        let hifi = hifi.transform(&transformation);

        let structure = AtomStructure {
            hifi,
            zee_e: basis.get_angular_blocks(|e| self.combined_structure.zeeman_e(e)),
            zee_n: basis.get_angular_blocks(|e| self.combined_structure.zeeman_n(e)),
        };

        AlkaliHomoDiatom {
            atom: structure,
            triplet,
            singlet,
        }
    }

    pub fn filter(&self, element: &[&SubspaceElement]) -> bool {
        let s_tot = cast_variant!(dyn element[self.combined_structure.s.0 as usize], Spin);
        let i_tot = cast_variant!(dyn element[self.combined_structure.i.0 as usize], Spin);
        let l = cast_variant!(dyn element[self.system.l.0 as usize], AngularMomentum);

        let even_spin = even_spin(self.s_max, self.i_max, s_tot.s, i_tot.s);
        let parity = match self.parity {
            Parity::All => true,
            Parity::Even => !(even_spin ^ (l.0 & 1 == 0)),
            Parity::Odd => even_spin ^ (l.0 & 1 == 0),
        };

        s_tot.m + i_tot.m == self.tot_projection && parity
    }

    fn filter_sep(&self, element: &[&SubspaceElement]) -> bool {
        let s_a = cast_variant!(dyn element[self.atom_a.s.0 as usize], Spin);
        let i_a = cast_variant!(dyn element[self.atom_a.i.0 as usize], Spin);
        let s_b = cast_variant!(dyn element[self.atom_b.s.0 as usize], Spin);
        let i_b = cast_variant!(dyn element[self.atom_b.i.0 as usize], Spin);

        s_a.m + i_a.m + s_b.m + i_b.m == self.tot_projection
    }
}

fn even_spin(s_max: HalfU32, i_max: HalfU32, s_tot: HalfU32, i_tot: HalfU32) -> bool {
    (s_max + i_max).double_value() % 4 == (s_tot + i_tot).double_value() % 4
}

#[derive(Clone, Debug)]
pub struct AlkaliHomoDiatom {
    atom: AtomStructure,

    triplet: Operator,
    singlet: Operator,
}

impl AlkaliHomoDiatom {
    pub fn with_params<T, S>(&self, params: &AlkaliHomoDiatomParams<T, S>) -> impl WMatrix
    where
        T: Interaction + Clone,
        S: Interaction + Clone,
    {
        let asymptote = self.atom.with_params(&params.atom);
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
