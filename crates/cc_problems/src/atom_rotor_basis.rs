use coupled_chan::{
    Interaction, Operator,
    constants::units::{Gauss, Quantity},
    coupling::{Asymptote, RedCoupling, VanishingCoupling, WMatrix, composite::Composite, masked::Masked, pair::Pair},
    scaled_interaction::ScaledInteraction,
};
use hilbert_space::{cast_variant, dyn_space::SpaceBasis, operator_mel};
use spin_algebra::{Spin, half_integer::HalfI32, hu32};

use crate::{
    AngularBasisElements, AngularMomentum,
    atom_structure::{AtomBasis, AtomBasisRecipe, AtomStructure},
    operator_mel::{percival_coef_tram_mel, singlet_projection_uncoupled, triplet_projection_uncoupled},
    rotor_structure::{Interaction2D, RotationalEnergy},
    system_structure::SystemParams,
    tram_basis::{TRAMBasis, TRAMBasisRecipe},
};

/// Recipe for alkali atom-rotor problem A + B-C,
/// where A, B are alkali-like
#[derive(Clone, Debug, Default)]
pub struct AtomRotorTRAMRecipe {
    pub atom_a: AtomBasisRecipe,
    pub atom_b: AtomBasisRecipe,
    pub atom_c: AtomBasisRecipe,
    pub tram: TRAMBasisRecipe,

    pub tot_projection: HalfI32,
    pub anisotropy_lambda_max: u32,
}

/// Builder for alkali atom-rotor problem A + B-C,
/// where A, B are alkali-like
#[derive(Clone, Debug)]
pub struct AtomRotorTRAMBasis {
    pub atom_a: AtomBasis,
    pub atom_b: AtomBasis,
    pub atom_c: AtomBasis,
    pub tram: TRAMBasis,

    basis: AngularBasisElements,
}

impl AtomRotorTRAMBasis {
    pub fn new(recipe: AtomRotorTRAMRecipe) -> Self {
        let mut basis = SpaceBasis::default();
        let atom_a = AtomBasis::new(recipe.atom_a, &mut basis);
        let atom_b = AtomBasis::new(recipe.atom_b, &mut basis);
        let atom_c = AtomBasis::new(recipe.atom_c, &mut basis);
        let tram = TRAMBasis::new(recipe.tram, &mut basis);

        let basis = basis.get_filtered_basis(|element| {
            if !tram.filter(element) {
                return false;
            }

            let n_tot = cast_variant!(dyn element[tram.n_tot.0 as usize], Spin);
            let s_a = cast_variant!(dyn element[atom_a.s.0 as usize], Spin);
            let i_a = cast_variant!(dyn element[atom_a.i.0 as usize], Spin);
            let s_b = cast_variant!(dyn element[atom_b.s.0 as usize], Spin);
            let i_b = cast_variant!(dyn element[atom_b.i.0 as usize], Spin);
            let s_c = cast_variant!(dyn element[atom_c.s.0 as usize], Spin);
            let i_c = cast_variant!(dyn element[atom_c.i.0 as usize], Spin);

            s_a.m + i_a.m + s_b.m + i_b.m + s_c.m + i_c.m + n_tot.m == recipe.tot_projection
        });
        let basis = AngularBasisElements::new_angular(basis, &tram.l);

        Self {
            atom_a,
            atom_b,
            atom_c,
            tram,
            basis,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TripletSurface<P: Interaction> {
    pub triplet: Interaction2D<ScaledInteraction<P>>,
    pub operator: Vec<Operator>,
}

impl<P: Interaction + Clone> TripletSurface<P> {
    pub fn new(
        surface: Interaction2D<P>,
        elements: &AngularBasisElements,
        atom: &AtomBasis,
        rotor: &AtomBasis,
        tram: &TRAMBasis,
    ) -> Self {
        let zeros = Operator::zeros(elements.full_basis.basis.size());

        let operator = surface
            .0
            .iter()
            .map(|x| {
                operator_mel!(
                    dyn elements.full_basis,
                    [atom.s, rotor.s, tram.l.l, tram.n.n, tram.n_tot],
                    |[s1: Spin, s2: Spin, l: AngularMomentum, n: AngularMomentum, n_tot: Spin]| {
                        percival_coef_tram_mel(x.0, l, n, n_tot) * triplet_projection_uncoupled(s1, s2)
                    }
                )
            })
            .filter(|a: &Operator| a.0 != zeros.0)
            .collect();

        let triplet = Interaction2D(
            surface
                .0
                .into_iter()
                .map(|x| (x.0, ScaledInteraction::new(x.1, 1.)))
                .collect(),
        );

        Self { triplet, operator }
    }

    pub fn hamiltonian(&self) -> Composite<Masked<ScaledInteraction<P>>> {
        let composite = self
            .triplet
            .0
            .iter()
            .zip(self.operator.iter())
            .map(|(t, o)| Masked::new(t.1.clone(), o.clone()))
            .collect();

        Composite::new(composite)
    }
}

#[derive(Debug, Clone)]
pub struct SingletSurface<P: Interaction> {
    pub singlet: Interaction2D<ScaledInteraction<P>>,
    pub operator: Vec<Operator>,
}

impl<P: Interaction + Clone> SingletSurface<P> {
    pub fn new(
        surface: Interaction2D<P>,
        elements: &AngularBasisElements,
        atom: &AtomBasis,
        rotor: &AtomBasis,
        tram: &TRAMBasis,
    ) -> Self {
        let zeros = Operator::zeros(elements.full_basis.basis.size());

        let operator = surface
            .0
            .iter()
            .map(|x| {
                operator_mel!(
                    dyn elements.full_basis,
                    [atom.s, rotor.s, tram.l.l, tram.n.n, tram.n_tot],
                    |[s1: Spin, s2: Spin, l: AngularMomentum, n: AngularMomentum, n_tot: Spin]| {
                        percival_coef_tram_mel(x.0, l, n, n_tot) * singlet_projection_uncoupled(s1, s2)
                    }
                )
            })
            .filter(|a: &Operator| a.0 != zeros.0)
            .collect();

        let singlet = Interaction2D(
            surface
                .0
                .into_iter()
                .map(|x| (x.0, ScaledInteraction::new(x.1, 1.)))
                .collect(),
        );

        Self { singlet, operator }
    }

    pub fn hamiltonian(&self) -> Composite<Masked<ScaledInteraction<P>>> {
        let composite = self
            .singlet
            .0
            .iter()
            .zip(self.operator.iter())
            .map(|(s, o)| Masked::new(s.1.clone(), o.clone()))
            .collect();

        Composite::new(composite)
    }
}

#[derive(Clone, Debug)]
pub struct AlkaliAtomRotorTRAM<T, S>
where
    T: Interaction,
    S: Interaction,
{
    pub system_params: SystemParams,
    pub atom_a: AtomStructure,
    pub atom_b: AtomStructure,
    pub atom_c: AtomStructure,

    pub rotational: RotationalEnergy,

    pub triplet: TripletSurface<T>,
    pub singlet: SingletSurface<S>,

    pub basis: AtomRotorTRAMBasis,
}

const ALKALI_SPINS: &str = "Expected alkali problem with s_a = 1/2, s_b = 1/2, s_c = 0";

impl<T, S> AlkaliAtomRotorTRAM<T, S>
where
    T: Interaction + Clone,
    S: Interaction + Clone,
{
    pub fn new(triplet: Interaction2D<T>, singlet: Interaction2D<S>, recipe: AtomRotorTRAMRecipe) -> Self {
        assert!(recipe.atom_a.s == hu32!(1 / 2), "{ALKALI_SPINS}");
        assert!(recipe.atom_b.s == hu32!(1 / 2), "{ALKALI_SPINS}");
        assert!(recipe.atom_c.s == hu32!(0), "{ALKALI_SPINS}");

        let basis = AtomRotorTRAMBasis::new(recipe);

        Self {
            system_params: SystemParams::default(),
            atom_a: AtomStructure::new(&basis.basis, &basis.atom_a),
            atom_b: AtomStructure::new(&basis.basis, &basis.atom_b),
            atom_c: AtomStructure::new(&basis.basis, &basis.atom_c),
            rotational: RotationalEnergy::new(&basis.basis, &basis.tram.n),
            triplet: TripletSurface::new(triplet, &basis.basis, &basis.atom_a, &basis.atom_b, &basis.tram),
            singlet: SingletSurface::new(singlet, &basis.basis, &basis.atom_a, &basis.atom_b, &basis.tram),
            basis,
        }
    }

    pub fn set_b_field(&mut self, b_field: Quantity<Gauss>) {
        self.atom_a.set_b_field(b_field);
        self.atom_b.set_b_field(b_field);
        self.atom_c.set_b_field(b_field);
    }

    pub fn asymptote(&self) -> Asymptote {
        let angular_blocks = self.atom_a.hamiltonian()
            + self.atom_b.hamiltonian()
            + self.atom_c.hamiltonian()
            + self.rotational.hamiltonian();

        Asymptote::new_angular_blocks(
            self.system_params.mass,
            self.system_params.energy,
            angular_blocks,
            self.system_params.entrance_channel,
        )
    }

    pub fn coupling(&self) -> impl VanishingCoupling {
        Pair::new(self.triplet.hamiltonian(), self.singlet.hamiltonian())
    }

    pub fn w_matrix(&self) -> impl WMatrix {
        RedCoupling::new(self.coupling(), self.asymptote())
    }
}
