use coupled_chan::{
    constants::{units::{atomic_units::Gauss, Quantity}, BOHR_MAG, G_FACTOR}, coupling::{pair::Pair, AngularBlocks, Asymptote, RedCoupling, WMatrix}, Interaction
};
use hilbert_space::{
    Parity, cast_variant,
    dyn_space::{BasisElementsRef, SpaceBasis, SubspaceBasis},
    operator_transform_mel,
};
use spin_algebra::{
    Spin, clebsch_gordan, get_summed_spin_basis,
    half_integer::{HalfI32, HalfU32},
};

use crate::{
    atom_structure::{AtomBasis, AtomBasisRecipe, AtomStructure, HyperfineStructure, ZeemanSplitting}, diatom_structure::{Singlet, Triplet}, system_structure::{AngularBasis, SystemParams}, AngularBasisElements, AngularMomentum
};

#[derive(Clone, Debug, Default)]
pub struct HomoDiatomRecipe {
    pub atom: AtomBasisRecipe,
    pub l_max: AngularMomentum,

    pub tot_projection: HalfI32,
}

#[derive(Clone, Debug)]
pub struct HomoDiatomBasis {
    pub combined_atom_basis: AtomBasis,
    pub angular: AngularBasis,

    basis: AngularBasisElements,

    atom_a: AtomBasis,
    atom_b: AtomBasis,
    basis_sep: AngularBasisElements,

    transformation: AngularBlocks
}

impl HomoDiatomBasis {
    pub fn new(recipe: HomoDiatomRecipe) -> Self {
        let mut basis = SpaceBasis::default();

        let s_tot = get_summed_spin_basis(recipe.atom.s, recipe.atom.s);
        let s_tot = basis.push_subspace(SubspaceBasis::new(s_tot));
        let i_tot = get_summed_spin_basis(recipe.atom.i, recipe.atom.i);
        let i_tot = basis.push_subspace(SubspaceBasis::new(i_tot));

        let s_max = recipe.atom.s + recipe.atom.s;
        let i_max = recipe.atom.i + recipe.atom.i;

        let combined_atom_basis = AtomBasis {
            s: s_tot,
            i: i_tot
        };
        let angular = AngularBasis::new(recipe.l_max, &mut basis);

        let parity = if (recipe.atom.s + recipe.atom.i).double_value().is_multiple_of(2) {
            Parity::Even
        } else {
            Parity::Odd
        };

        let basis = basis.get_filtered_basis(|element| {
            let s_tot = cast_variant!(dyn element[combined_atom_basis.s.0 as usize], Spin);
            let i_tot = cast_variant!(dyn element[combined_atom_basis.i.0 as usize], Spin);
            let l = cast_variant!(dyn element[angular.l.0 as usize], AngularMomentum);

            let even_spin = even_spin(s_max, i_max, s_tot.s, i_tot.s);
            let parity = match parity {
                Parity::All => true,
                Parity::Even => !(even_spin ^ (l.0 & 1 == 0)),
                Parity::Odd => even_spin ^ (l.0 & 1 == 0),
            };

            s_tot.m + i_tot.m == recipe.tot_projection && parity
        });
        let basis = AngularBasisElements::new_angular(basis, &angular);

        let mut basis_sep = SpaceBasis::default();
        let atom_a = AtomBasis::new(recipe.atom, &mut basis_sep);
        let atom_b = AtomBasis::new(recipe.atom, &mut basis_sep);
        let angular_sep = AngularBasis::new(recipe.l_max, &mut basis_sep);
        
        let basis_sep = basis_sep.get_filtered_basis(|element| {
            let s_a = cast_variant!(dyn element[atom_a.s.0 as usize], Spin);
            let i_a = cast_variant!(dyn element[atom_a.i.0 as usize], Spin);
            let s_b = cast_variant!(dyn element[atom_b.s.0 as usize], Spin);
            let i_b = cast_variant!(dyn element[atom_b.i.0 as usize], Spin);

            s_a.m + i_a.m + s_b.m + i_b.m == recipe.tot_projection
        });
        let basis_sep = AngularBasisElements::new_angular(basis_sep, &angular_sep);

        let transformation = |e_sep: BasisElementsRef<'_>, e: BasisElementsRef<'_>| {
            operator_transform_mel!(
                dyn e_sep, [angular_sep.l, atom_a.s, atom_a.i, atom_b.s, atom_b.i],
                dyn e, [angular.l, combined_atom_basis.s, combined_atom_basis.i],
                |[l_sep: AngularMomentum, s1: Spin, i1: Spin, s2: Spin, i2: Spin],
                    [l: AngularMomentum, s_tot: Spin, i_tot: Spin]|
                {
                    if l == l_sep {
                        clebsch_gordan(s1.s, s1.m, s2.s, s2.m, s_tot.s, s_tot.m)
                            * clebsch_gordan(i1.s, i1.m, i2.s, i2.m, i_tot.s, i_tot.m)
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
        
        Self {
            combined_atom_basis,
            angular,
            basis,
            atom_a,
            atom_b,
            basis_sep,
            transformation,
        }
    }
}

fn even_spin(s_max: HalfU32, i_max: HalfU32, s_tot: HalfU32, i_tot: HalfU32) -> bool {
    (s_max + i_max).double_value() % 4 == (s_tot + i_tot).double_value() % 4
}

#[derive(Clone, Debug)]
pub struct AlkaliHomoDiatom<T, S>
where 
    T: Interaction,
    S: Interaction
{
    pub system_params: SystemParams,
    pub atom: AtomStructure,

    pub triplet: Triplet<T>,
    pub singlet: Singlet<S>,

    pub basis: HomoDiatomBasis
}

impl<T, S> AlkaliHomoDiatom<T, S>  
where 
    T: Interaction + Clone,
    S: Interaction + Clone
{
    pub fn new(recipe: HomoDiatomRecipe) -> Self {
        let basis = HomoDiatomBasis::new(recipe);

        let hifi_a = HyperfineStructure::new(&basis.basis_sep, &basis.atom_a);
        let hifi_b = HyperfineStructure::new(&basis.basis_sep, &basis.atom_b);

        let operator = (hifi_a.operator + hifi_b.operator).transform(&basis.transformation);
        let hifi = HyperfineStructure {
            operator,
            a_hifi: Default::default(),
        };

        let mut atom = AtomStructure {
            zeeman_e: ZeemanSplitting::new(&basis.basis, basis.combined_atom_basis.s),
            zeeman_n: ZeemanSplitting::new(&basis.basis, basis.combined_atom_basis.i),
            hyperfine: hifi,
        };
        atom.zeeman_e.gamma = -G_FACTOR * BOHR_MAG;

        Self {
            system_params: SystemParams::default(),
            atom,
            triplet: Triplet::new_coupled(&basis.basis, &basis.combined_atom_basis),
            singlet: Singlet::new_coupled(&basis.basis, &basis.combined_atom_basis),
            basis,
        }
    }

    pub fn set_b_field(&mut self, b_field: Quantity<Gauss>) {
        self.atom.set_b_field(b_field);
    }

    pub fn w_matrix(&self) -> impl WMatrix + use<T, S> {
        let angular_blocks = self.atom.hamiltonian();

        let asymptote = Asymptote::new_angular_blocks(
            self.system_params.mass, 
            self.system_params.energy, 
            angular_blocks, 
            self.system_params.entrance_channel
        );

        let potential = Pair::new(self.triplet.hamiltonian(), self.singlet.hamiltonian());
        
        RedCoupling::new(potential, asymptote)
    }
}
