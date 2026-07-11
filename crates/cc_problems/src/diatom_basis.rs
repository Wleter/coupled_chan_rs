use hilbert_space::space::{
    BasisId,
    SpaceBasis,
    SpaceElement,
    SubspaceBasis,
};
use serde::{Deserialize, Serialize};
use spin_algebra::{
    Spin,
    SpinMagLike,
    SpinPair,
    SpinPairMag,
    get_spin_pair_basis,
    get_spin_pair_magnitudes,
    half_integer::HalfU32,
};

use crate::{
    Angular,
    OrbitalBasis,
    OrbitalRecipe,
    atom_basis::{
        AtomRecipe,
        CoupledAtomBasis,
        TwiceSpin,
        UncoupledAtomBasis,
    },
};

pub type SpinSTot = SpinPairMag<HalfU32, HalfU32>;
pub type SpinITot = SpinPairMag<HalfU32, HalfU32>;
pub type SpinFTot = SpinPairMag<SpinSTot, SpinITot>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiatomRecipe {
    pub atom_a: AtomRecipe,
    pub atom_b: AtomRecipe,
    pub l: OrbitalRecipe,
}

/// Struct for storing id of
/// |s1 m_s1>|i1 m_i1>|s2 m_s2>|i2 m_i2>|l m_l> state
#[derive(Debug, Clone, Copy)]
pub struct UncoupledDiatomBasis {
    pub atom_a: UncoupledAtomBasis,
    pub atom_b: UncoupledAtomBasis,
    pub l: OrbitalBasis,
}

impl UncoupledDiatomBasis {
    /// Adds |s1 m_s1>|i1 m_i1>|s2 m_s2>|i2 m_i2>|l m_l>
    /// to the basis.
    pub fn new(recipe: DiatomRecipe, basis: &mut SpaceBasis) -> Self {
        let atom_a = UncoupledAtomBasis::new(recipe.atom_a, basis);
        let atom_b = UncoupledAtomBasis::new(recipe.atom_b, basis);
        let l = OrbitalBasis::new(recipe.l, basis);

        Self { atom_a, atom_b, l }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn(((Spin, Spin), (Spin, Spin), Angular)) -> bool,
    {
        move |x| {
            let s_a = x[self.atom_a.s];
            let i_a = x[self.atom_a.i];
            let s_b = x[self.atom_b.s];
            let i_b = x[self.atom_b.i];
            let l = x[self.l.l];

            f(((s_a, i_a), (s_b, i_b), l))
        }
    }
}

/// Struct for storing id of
/// |(s1, i1) f1 m_f1>|(s2, i2) f2 m_f2>|l m_l> state.
#[derive(Debug, Clone, Copy)]
pub struct CoupledFDiatomBasis {
    pub atom_a: CoupledAtomBasis,
    pub atom_b: CoupledAtomBasis,
    pub l: OrbitalBasis,
}

impl CoupledFDiatomBasis {
    /// Adds |(s1, i1) f1 m_f1>|(s2, i2) f2 m_f2>|l m_l>
    /// to the basis.
    pub fn new(recipe: DiatomRecipe, basis: &mut SpaceBasis) -> Self {
        let atom_a = CoupledAtomBasis::new(recipe.atom_a, basis);
        let atom_b = CoupledAtomBasis::new(recipe.atom_b, basis);
        let l = OrbitalBasis::new(recipe.l, basis);

        Self { atom_a, atom_b, l }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn((TwiceSpin, TwiceSpin, Angular)) -> bool,
    {
        move |x| f((x[self.atom_a.f], x[self.atom_b.f], x[self.l.l]))
    }
}

/// Struct for storing id of
/// |(s1, s2) S M_S>|(i1, i2) I M_I>|l m_l> state.
#[derive(Debug, Clone, Copy)]
pub struct CoupledSIDiatomBasis {
    pub s_tot: BasisId<TwiceSpin>,
    pub i_tot: BasisId<TwiceSpin>,
    pub l: OrbitalBasis,
}

impl CoupledSIDiatomBasis {
    /// Adds |(s1, s2) S M_S>|(i1, i2) I M_I>|l m_l>
    /// to the basis.
    pub fn new(recipe: DiatomRecipe, basis: &mut SpaceBasis) -> Self {
        let s_tot = get_spin_pair_magnitudes([recipe.atom_a.s], [recipe.atom_b.s]);
        let s_tot = get_spin_pair_basis(s_tot);
        let i_tot = get_spin_pair_magnitudes([recipe.atom_a.i], [recipe.atom_b.i]);
        let i_tot = get_spin_pair_basis(i_tot);

        let s_tot = basis.push_subspace(SubspaceBasis::new(s_tot));
        let i_tot = basis.push_subspace(SubspaceBasis::new(i_tot));
        let l = OrbitalBasis::new(recipe.l, basis);

        Self { s_tot, i_tot, l }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn((TwiceSpin, TwiceSpin, Angular)) -> bool,
    {
        move |x| f((x[self.s_tot], x[self.i_tot], x[self.l.l]))
    }

    pub fn filter_homo_nuclear_symmetry(&self) -> impl Fn(SpaceElement) -> bool {
        move |x| {
            let s_tot = x[self.s_tot].as_spin_pair_mag();
            let i_tot = x[self.i_tot].as_spin_pair_mag();
            let l = x[self.l.l].l_value();

            homo_nuclear_symmetry(s_tot, i_tot, l)
        }
    }
}

/// Struct for storing id of
/// |((s1, s2) S, (i1, i2) I) F M_F>|l m_l> state
#[derive(Debug, Clone, Copy)]
pub struct CoupledFTotDiatomBasis {
    pub f_tot: BasisId<SpinPair<SpinSTot, SpinITot>>,
    pub l: OrbitalBasis,
}

impl CoupledFTotDiatomBasis {
    /// Adds |((s1, s2) S, (i1, i2) I) F M_F>|l m_l>
    /// to the basis.
    pub fn new(recipe: DiatomRecipe, basis: &mut SpaceBasis) -> Self {
        let s_tot = get_spin_pair_magnitudes([recipe.atom_a.s], [recipe.atom_b.s]);
        let i_tot = get_spin_pair_magnitudes([recipe.atom_a.i], [recipe.atom_b.i]);
        let f_tot = get_spin_pair_magnitudes(s_tot, i_tot);
        let f_tot = get_spin_pair_basis(f_tot);

        let f_tot = basis.push_subspace(SubspaceBasis::new(f_tot));
        let l = OrbitalBasis::new(recipe.l, basis);

        Self { f_tot, l }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn((SpinPair<SpinSTot, SpinITot>, Angular)) -> bool,
    {
        move |x| f((x[self.f_tot], x[self.l.l]))
    }

    pub fn filter_homo_nuclear_symmetry(&self) -> impl Fn(SpaceElement) -> bool {
        move |x| {
            let f_tot = x[self.f_tot];
            let s_tot = f_tot.pair.0;
            let i_tot = f_tot.pair.1;
            let l = x[self.l.l].l_value();

            homo_nuclear_symmetry(s_tot, i_tot, l)
        }
    }
}

/// Struct for storing id of |(((s1, s2) S, (i1, i2) I) F, l) Fl M_Fl>
/// state
///
/// Note: With this state all projections of l are alway included.
#[derive(Debug, Clone, Copy)]
pub struct CoupledDiatomBasis {
    pub fl_tot: BasisId<SpinPair<SpinFTot, u32>>,
}

impl CoupledDiatomBasis {
    /// Adds |(((s1, s2) S, (i1, i2) I) F, l) Fl M_Fl>
    /// to the basis.
    ///
    /// Note: With this basis all projections of l are alway included.
    pub fn new(recipe: DiatomRecipe, basis: &mut SpaceBasis) -> Self {
        let s_tot = get_spin_pair_magnitudes([recipe.atom_a.s], [recipe.atom_b.s]);
        let i_tot = get_spin_pair_magnitudes([recipe.atom_a.i], [recipe.atom_b.i]);
        let f_tot = get_spin_pair_magnitudes(s_tot, i_tot);
        let l = recipe.l.magnitudes();
        if let OrbitalRecipe::LMax(_) | OrbitalRecipe::Single(_) = recipe.l {
            println!("warning: All orbital projections used when using fully coupled diatom basis")
        }

        let fl_tot = get_spin_pair_magnitudes(f_tot, l);
        let fl_tot = get_spin_pair_basis(fl_tot);

        let fl_tot = basis.push_subspace(SubspaceBasis::new(fl_tot));

        Self { fl_tot }
    }

    pub fn filter<F>(&self, f: F) -> impl Fn(SpaceElement) -> bool
    where
        F: Fn(SpinPair<SpinFTot, u32>) -> bool,
    {
        move |x| f(x[self.fl_tot])
    }

    pub fn filter_homo_nuclear_symmetry(&self) -> impl Fn(SpaceElement) -> bool {
        move |x| {
            let fl_tot = x[self.fl_tot];
            let s_tot = fl_tot.pair.0.pair.0;
            let i_tot = fl_tot.pair.0.pair.1;
            let l = fl_tot.pair.1;

            homo_nuclear_symmetry(s_tot, i_tot, l)
        }
    }
}

pub fn homo_nuclear_symmetry<S, I>(s_tot: SpinPairMag<S, S>, i_tot: SpinPairMag<I, I>, l: u32) -> bool
where
    S: SpinMagLike,
    I: SpinMagLike,
{
    assert_eq!(s_tot.pair.0, s_tot.pair.1, "Different spins in a homo nuclear system");
    assert_eq!(i_tot.pair.0, i_tot.pair.1, "Different spins in a homo nuclear system");

    let f = s_tot.pair.0.s() + i_tot.pair.0.s();
    let s_max = s_tot.pair.0.s() + s_tot.pair.1.s();
    let i_max = i_tot.pair.0.s() + i_tot.pair.1.s();

    let s_tot = s_tot.s();
    let i_tot = i_tot.s();
    assert!(s_max >= s_tot, "combined S is larger than s1 + s2");
    assert!(i_max >= i_tot, "combined I is larger than i1 + i2");

    let symmetry = (-1i32).pow(l + (s_max + i_max - s_tot - i_tot).double_value() / 2);

    match f.spin_type() {
        spin_algebra::SpinType::Fermionic => symmetry == -1,
        spin_algebra::SpinType::Bosonic => symmetry == 1,
    }
}

#[cfg(test)]
mod tests {
    use spin_algebra::{
        SpinLike,
        hi32,
        hu32,
        spin,
    };

    use super::*;

    fn recipe() -> DiatomRecipe {
        DiatomRecipe {
            atom_a: AtomRecipe {
                s: hu32!(1 / 2),
                i: hu32!(0),
            },
            atom_b: AtomRecipe {
                s: hu32!(1),
                i: hu32!(1 / 2),
            },
            l: OrbitalRecipe::LMax(1),
        }
    }

    #[test]
    fn test_uncoupled_diatom_basis() {
        let mut basis = SpaceBasis::default();
        let atoms = UncoupledDiatomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| {
            atoms.filter(|((s_a, i_a), (s_b, i_b), l)| s_a.m + i_a.m + s_b.m + i_b.m + l.m == hi32!(1))(x)
        });

        println!("{elements:?}");
        assert_eq!(elements.len(), 6);

        let u_12 = hu32!(1 / 2);
        let i_12 = hi32!(1 / 2);
        let u_0 = hu32!(0);
        let i_0 = hi32!(0);
        let u_1 = hu32!(1);

        assert_eq!(elements[(0, atoms.atom_a.s)], spin!(u_12, i_12));
        assert_eq!(elements[(0, atoms.atom_a.i)], spin!(u_0, i_0));
        assert_eq!(elements[(0, atoms.atom_b.s)], spin!(u_1, hi32!(1)));
        assert_eq!(elements[(0, atoms.atom_b.i)], spin!(u_12, hi32!(-1 / 2)));
        assert_eq!(elements[(0, atoms.l.l)], Angular::new(0, 0));

        assert_eq!(elements[(4, atoms.atom_a.s)], spin!(u_12, i_12));
        assert_eq!(elements[(4, atoms.atom_a.i)], spin!(u_0, i_0));
        assert_eq!(elements[(4, atoms.atom_b.s)], spin!(u_1, i_0));
        assert_eq!(elements[(4, atoms.atom_b.i)], spin!(u_12, i_12));
        assert_eq!(elements[(4, atoms.l.l)], Angular::new(1, 0));
    }

    #[test]
    fn test_coupled_f_diatom_basis() {
        let mut basis = SpaceBasis::default();
        let atoms = CoupledFDiatomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atoms.filter(|(f_a, f_b, l)| f_a.m() + f_b.m() + l.m == hi32!(1))(x));

        println!("{elements:?}");
        assert_eq!(elements.len(), 6);

        let u_12 = hu32!(1 / 2);
        let i_12 = hi32!(1 / 2);
        let u_0 = hu32!(0);
        let u_1 = hu32!(1);

        assert_eq!(elements[(0, atoms.atom_a.f)], spin!((u_12, u_0), u_12, i_12));
        assert_eq!(elements[(0, atoms.atom_b.f)], spin!((u_1, u_12), u_12, i_12));
        assert_eq!(elements[(0, atoms.l.l)], Angular::new(0, 0));

        assert_eq!(elements[(4, atoms.atom_a.f)], spin!((u_12, u_0), u_12, i_12));
        assert_eq!(elements[(4, atoms.atom_b.f)], spin!((u_1, u_12), hu32!(3 / 2), i_12));
        assert_eq!(elements[(4, atoms.l.l)], Angular::new(1, 0));
    }

    #[test]
    fn test_coupled_si_diatom_basis() {
        let mut basis = SpaceBasis::default();
        let atoms = CoupledSIDiatomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atoms.filter(|(s, i, l)| s.m() + i.m() + l.m == hi32!(1))(x));

        println!("{elements:?}");
        assert_eq!(elements.len(), 6);

        let u_12 = hu32!(1 / 2);
        let i_12 = hi32!(1 / 2);
        let u_0 = hu32!(0);
        let u_1 = hu32!(1);
        let u_32 = hu32!(3 / 2);
        let i_32 = hi32!(3 / 2);

        assert_eq!(elements[(0, atoms.s_tot)], spin!((u_12, u_1), u_32, i_32));
        assert_eq!(elements[(0, atoms.i_tot)], spin!((u_0, u_12), u_12, -i_12));
        assert_eq!(elements[(0, atoms.l.l)], Angular::new(0, 0));

        assert_eq!(elements[(4, atoms.s_tot)], spin!((u_12, u_1), u_12, i_12));
        assert_eq!(elements[(4, atoms.i_tot)], spin!((u_0, u_12), u_12, i_12));
        assert_eq!(elements[(4, atoms.l.l)], Angular::new(1, 0));
    }

    #[test]
    fn test_coupled_f_tot_diatom_basis() {
        let mut basis = SpaceBasis::default();
        let atoms = CoupledFTotDiatomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atoms.filter(|(f, l)| f.m() + l.m == hi32!(1))(x));

        println!("{elements:?}");
        assert_eq!(elements.len(), 6);

        let u_12 = hu32!(1 / 2);
        let u_0 = hu32!(0);
        let u_1 = hu32!(1);
        let i_1 = hi32!(1);
        let u_32 = hu32!(3 / 2);

        assert_eq!(
            elements[(0, atoms.f_tot)],
            spin!((((u_12, u_1), u_12), ((u_0, u_12), u_12)), u_1, i_1)
        );
        assert_eq!(elements[(0, atoms.l.l)], Angular::new(0, 0));

        assert_eq!(
            elements[(4, atoms.f_tot)],
            spin!((((u_12, u_1), u_32), ((u_0, u_12), u_12)), u_1, i_1)
        );
        assert_eq!(elements[(4, atoms.l.l)], Angular::new(1, 0));
    }

    #[test]
    fn test_coupled_diatom_basis() {
        let mut basis = SpaceBasis::default();
        let atoms = CoupledDiatomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atoms.filter(|fl| fl.m() == hi32!(1))(x));

        println!("{elements:?}");
        assert_eq!(elements.len(), 11);

        let u_12 = hu32!(1 / 2);
        let u_0 = hu32!(0);
        let u_1 = hu32!(1);
        let i_1 = hi32!(1);
        let u_32 = hu32!(3 / 2);

        assert_eq!(
            elements[(0, atoms.fl_tot)],
            spin!((((((u_12, u_1), u_12), ((u_0, u_12), u_12)), u_0), 1), u_1, i_1)
        );
        assert_eq!(
            elements[(4, atoms.fl_tot)],
            spin!((((((u_12, u_1), u_32), ((u_0, u_12), u_12)), u_1), 0), u_1, i_1)
        );
    }

    #[test]
    fn test_homo_nuclear_filter() {
        let atom_recipe = AtomRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1),
        };
        let recipe = DiatomRecipe {
            atom_a: atom_recipe,
            atom_b: atom_recipe,
            l: OrbitalRecipe::LMax(2),
        };

        let mut basis = SpaceBasis::default();
        let atoms = CoupledFTotDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| {
            atoms.filter_homo_nuclear_symmetry()(x) && atoms.filter(|(f, l)| f.m() + l.m() == hi32!(1))(x)
        });
        assert_eq!(elements.len(), 11, "{elements}");

        let mut basis = SpaceBasis::default();
        let atoms = CoupledDiatomBasis::new(recipe, &mut basis);
        let elements =
            basis.get_filtered_basis(|x| atoms.filter_homo_nuclear_symmetry()(x) && atoms.filter(|f| f.m() == hi32!(1))(x));
        assert_eq!(elements.len(), 28, "{elements}");

        let atom_recipe = AtomRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
        };
        let recipe = DiatomRecipe {
            atom_a: atom_recipe,
            atom_b: atom_recipe,
            l: OrbitalRecipe::LMax(2),
        };

        let mut basis = SpaceBasis::default();
        let atoms = CoupledFTotDiatomBasis::new(recipe, &mut basis);
        let elements = basis.get_filtered_basis(|x| {
            atoms.filter_homo_nuclear_symmetry()(x) && atoms.filter(|(f, l)| f.m() + l.m() == hi32!(2))(x)
        });
        assert_eq!(elements.len(), 13, "{elements}");

        let mut basis = SpaceBasis::default();
        let atoms = CoupledDiatomBasis::new(recipe, &mut basis);
        let elements =
            basis.get_filtered_basis(|x| atoms.filter_homo_nuclear_symmetry()(x) && atoms.filter(|f| f.m() == hi32!(2))(x));
        assert_eq!(elements.len(), 38, "{elements}");
    }
}
