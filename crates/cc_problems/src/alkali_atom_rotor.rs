use coupled_chan::{
    Interaction, Operator,
    coupling::{Asymptote, RedCoupling, WMatrix, composite::Composite, masked::Masked, pair::Pair},
};
use hilbert_space::{
    cast_variant,
    dyn_space::{SpaceBasis, SubspaceElement},
    operator_mel,
};
use spin_algebra::{Spin, half_integer::HalfI32, hu32};

use crate::{
    AngularBasisElements, AngularMomentum,
    atom_structure::{AtomStructure, AtomStructureBuilder, AtomStructureParams, AtomStructureRecipe},
    operator_mel::{percival_coef_tram_mel, singlet_projection_uncoupled, triplet_projection_uncoupled},
    rotor_structure::Interaction2D,
    tram_basis::{TRAMBasis, TRAMBasisBuilder, TRAMBasisParams, TRAMBasisRecipe},
};

/// Recipe for alkali atom-rotor problem A + B-C,
/// where A, B are alkali-like
#[derive(Clone, Debug, Default)]
pub struct AlkaliAtomRotorTRAMRecipe {
    pub atom_a: AtomStructureRecipe,
    pub atom_b: AtomStructureRecipe,
    pub atom_c: AtomStructureRecipe,
    pub tram: TRAMBasisRecipe,

    pub tot_projection: HalfI32,
    pub anisotropy_lambda_max: u32,
}

/// Params for alkali atom-rotor problem A + B-C,
/// where A, B are alkali-like
#[derive(Clone, Debug)]
pub struct AlkaliAtomRotorTRAMParams<S: Interaction, T: Interaction> {
    pub atom_a: AtomStructureParams,
    pub atom_b: AtomStructureParams,
    pub atom_c: AtomStructureParams,
    pub tram: TRAMBasisParams,

    pub singlet: Interaction2D<S>,
    pub triplet: Interaction2D<T>,
}

/// Builder for alkali atom-rotor problem A + B-C,
/// where A, B are alkali-like
#[derive(Clone, Debug)]
pub struct AlkaliAtomRotorTRAMBuilder {
    pub atom_a: AtomStructureBuilder,
    pub atom_b: AtomStructureBuilder,
    pub atom_c: AtomStructureBuilder,
    pub tram: TRAMBasisBuilder,

    tot_projection: HalfI32,
    anisotropy_lambda_max: u32,

    basis: SpaceBasis,
}

impl AlkaliAtomRotorTRAMBuilder {
    pub fn new(recipe: AlkaliAtomRotorTRAMRecipe) -> Self {
        assert!(recipe.atom_a.s == hu32!(1 / 2));
        assert!(recipe.atom_b.s == hu32!(1 / 2));
        assert!(recipe.atom_c.s == hu32!(0));

        let mut basis = SpaceBasis::default();
        let atom_a = AtomStructureBuilder::new(recipe.atom_a, &mut basis);
        let atom_b = AtomStructureBuilder::new(recipe.atom_b, &mut basis);
        let atom_c = AtomStructureBuilder::new(recipe.atom_c, &mut basis);
        let tram = TRAMBasisBuilder::new(recipe.tram, &mut basis);

        Self {
            atom_a,
            atom_b,
            atom_c,
            tram,
            tot_projection: recipe.tot_projection,
            basis,
            anisotropy_lambda_max: recipe.anisotropy_lambda_max,
        }
    }

    pub fn build(self) -> AlkaliAtomRotorTRAM {
        let basis = self.basis.get_filtered_basis(|x| self.filter(x) && self.tram.filter(x));
        let basis = AngularBasisElements::new_system(basis, &self.tram.l);

        let zeros = Operator::zeros(basis.full_basis.basis.size());
        let triplet = (0..=self.anisotropy_lambda_max)
            .map(|lambda| {
                (
                    lambda,
                    operator_mel!(
                        dyn basis.full_basis,
                        [self.atom_a.s, self.atom_b.s, self.tram.l.l, self.tram.n.n, self.tram.n_tot],
                        |[s1: Spin, s2: Spin, l: AngularMomentum, n: AngularMomentum, n_tot: Spin]| {
                            percival_coef_tram_mel(lambda, l, n, n_tot) * triplet_projection_uncoupled(s1, s2)
                        }
                    ),
                )
            })
            .filter(|a| a.1.0 == zeros.0)
            .collect();

        let singlet = (0..=self.anisotropy_lambda_max)
            .map(|lambda| {
                (
                    lambda,
                    operator_mel!(
                        dyn basis.full_basis,
                        [self.atom_a.s, self.atom_b.s, self.tram.l.l, self.tram.n.n, self.tram.n_tot],
                        |[s1: Spin, s2: Spin, l: AngularMomentum, n: AngularMomentum, n_tot: Spin]| {
                            percival_coef_tram_mel(lambda, l, n, n_tot) * singlet_projection_uncoupled(s1, s2)
                        }
                    ),
                )
            })
            .filter(|a| a.1.0 == zeros.0)
            .collect();

        AlkaliAtomRotorTRAM {
            atom_a: self.atom_a.build(&basis),
            atom_b: self.atom_b.build(&basis),
            atom_c: self.atom_c.build(&basis),
            tram: self.tram.build(&basis),
            triplet,
            singlet,
        }
    }

    pub fn filter(&self, element: &[&SubspaceElement]) -> bool {
        if !self.tram.filter(element) {
            return false;
        }

        let n_tot = cast_variant!(dyn element[self.tram.n_tot.0 as usize], Spin);
        let s_a = cast_variant!(dyn element[self.atom_a.s.0 as usize], Spin);
        let i_a = cast_variant!(dyn element[self.atom_a.i.0 as usize], Spin);
        let s_b = cast_variant!(dyn element[self.atom_b.s.0 as usize], Spin);
        let i_b = cast_variant!(dyn element[self.atom_b.i.0 as usize], Spin);
        let s_c = cast_variant!(dyn element[self.atom_c.s.0 as usize], Spin);
        let i_c = cast_variant!(dyn element[self.atom_c.i.0 as usize], Spin);

        s_a.m + i_a.m + s_b.m + i_b.m + s_c.m + i_c.m + n_tot.m == self.tot_projection
    }
}

#[derive(Clone, Debug)]
pub struct AlkaliAtomRotorTRAM {
    atom_a: AtomStructure,
    atom_b: AtomStructure,
    atom_c: AtomStructure,
    tram: TRAMBasis,

    triplet: Vec<(u32, Operator)>,
    singlet: Vec<(u32, Operator)>,
}

impl AlkaliAtomRotorTRAM {
    pub fn with_params<S, T>(&self, params: &AlkaliAtomRotorTRAMParams<S, T>) -> impl WMatrix
    where
        S: Interaction + Clone,
        T: Interaction + Clone,
    {
        let asymptote = self.atom_a.with_params(&params.atom_a)
            + self.atom_b.with_params(&params.atom_b)
            + self.atom_c.with_params(&params.atom_c)
            + self.tram.with_params(&params.tram);

        let asymptote = Asymptote::new_angular_blocks(
            params.tram.system.mass,
            params.tram.system.energy,
            asymptote,
            params.tram.system.entrance_channel,
        );

        let mut triplet = Composite::default();
        for t in &self.triplet {
            if let Some((_, p)) = params.triplet.0.iter().find(|&x| x.0 == t.0) {
                triplet.add_coupling(Masked::new(p.clone(), t.1.clone()));
            }
        }

        let mut singlet = Composite::default();
        for t in &self.singlet {
            if let Some((_, p)) = params.singlet.0.iter().find(|&x| x.0 == t.0) {
                singlet.add_coupling(Masked::new(p.clone(), t.1.clone()));
            }
        }

        let coupling = Pair::new(triplet, singlet);

        RedCoupling::new(coupling, asymptote)
    }
}
