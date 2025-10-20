use coupled_chan::{
    constants::{Gauss, Quantity}, coupling::{masked::Masked, pair::Pair, Asymptote, RedCoupling, WMatrix}, scaled_interaction::ScaledInteraction, Interaction, Operator
};
use hilbert_space::{
    cast_variant, dyn_space::SpaceBasis, operator_diag_mel, operator_mel
};
use spin_algebra::{half_integer::HalfI32, hu32, Spin};

use crate::{
    atom_structure::{AtomBasis, AtomBasisRecipe, AtomStructure}, operator_mel::{singlet_projection_uncoupled, triplet_projection_uncoupled}, system_structure::{AngularBasis, SystemParams}, AngularBasisElements, AngularMomentum
};

#[derive(Clone, Debug, Default)]
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
            let s_a = cast_variant!(dyn element[atom_a.s.0 as usize], Spin);
            let i_a = cast_variant!(dyn element[atom_a.i.0 as usize], Spin);
            let s_b = cast_variant!(dyn element[atom_b.s.0 as usize], Spin);
            let i_b = cast_variant!(dyn element[atom_b.i.0 as usize], Spin);

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
pub struct Triplet<P: Interaction> {
    pub triplet: Option<ScaledInteraction<P>>,
    pub operator: Operator
}

impl<P: Interaction + Clone> Triplet<P> {
    pub fn new(elements: &AngularBasisElements, atom_a: &AtomBasis, atom_b: &AtomBasis) -> Self {
        let operator = operator_mel!(dyn elements.full_basis, [atom_a.s, atom_b.s], |[s1: Spin, s2: Spin]| {
            triplet_projection_uncoupled(s1, s2)
        });

        Self {
            triplet: None,
            operator,
        }
    }

    pub fn new_coupled(elements: &AngularBasisElements, atom_pair: &AtomBasis) -> Self {
        let operator = operator_diag_mel!(dyn elements.full_basis, [atom_pair.s], |[s: Spin]| {
            if s.s == hu32!(1) { 1. } else { 0. }
        });

        Self {
            triplet: None,
            operator,
        }
    }

    pub fn set_triplet(&mut self, triplet: P) {
        self.triplet = Some(ScaledInteraction::new(triplet, 1.))
    }

    pub fn hamiltonian(&self) -> Masked<ScaledInteraction<P>> {
        let triplet = self.triplet.as_ref().expect("Did not set triplet potential");

        Masked::new(triplet.clone(), self.operator.clone())
    }
}

#[derive(Debug, Clone)]
pub struct Singlet<P: Interaction> {
    pub singlet: Option<ScaledInteraction<P>>,
    pub operator: Operator
}

impl<P: Interaction + Clone> Singlet<P> {
    pub fn new(elements: &AngularBasisElements, atom_a: &AtomBasis, atom_b: &AtomBasis) -> Self {
        let operator = operator_mel!(dyn elements.full_basis, [atom_a.s, atom_b.s], |[s1: Spin, s2: Spin]| {
            singlet_projection_uncoupled(s1, s2)
        });

        Self {
            singlet: None,
            operator,
        }
    }

    pub fn new_coupled(elements: &AngularBasisElements, atom_pair: &AtomBasis) -> Self {
        let operator = operator_diag_mel!(dyn elements.full_basis, [atom_pair.s], |[s: Spin]| {
            if s.s == hu32!(0) { 1. } else { 0. }
        });

        Self {
            singlet: None,
            operator,
        }
    }

    pub fn set_singlet(&mut self, singlet: P) {
        self.singlet = Some(ScaledInteraction::new(singlet, 1.))
    }

    pub fn hamiltonian(&self) -> Masked<ScaledInteraction<P>> {
        let singlet = self.singlet.as_ref().expect("Did not set triplet potential");

        Masked::new(singlet.clone(), self.operator.clone())
    }
}

#[derive(Clone, Debug)]
pub struct AlkaliDiatom<T, S> 
where 
    T: Interaction,
    S: Interaction
{
    pub system_params: SystemParams,
    pub atom_a: AtomStructure,
    pub atom_b: AtomStructure,

    pub triplet: Triplet<T>,
    pub singlet: Singlet<S>,

    pub basis: DiatomBasis,
}

impl<T, S> AlkaliDiatom<T, S>  
where 
    T: Interaction + Clone,
    S: Interaction + Clone
{
    pub fn new(recipe: DiatomBasisRecipe) -> Self {
        let basis = DiatomBasis::new(recipe);

        Self {
            system_params: SystemParams::default(),
            atom_a: AtomStructure::new(&basis.basis, &basis.atom_a),
            atom_b: AtomStructure::new(&basis.basis, &basis.atom_b),
            triplet: Triplet::new(&basis.basis, &basis.atom_a, &basis.atom_b),
            singlet: Singlet::new(&basis.basis, &basis.atom_a, &basis.atom_b),
            basis,
        }
    }

    pub fn set_b_field(&mut self, b_field: Quantity<Gauss>) {
        self.atom_a.set_b_field(b_field);
        self.atom_b.set_b_field(b_field);
    }

    pub fn w_matrix(&self) -> impl WMatrix {
        let angular_blocks = self.atom_a.hamiltonian() + self.atom_b.hamiltonian();

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
