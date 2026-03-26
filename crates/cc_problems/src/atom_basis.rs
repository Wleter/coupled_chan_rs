use hilbert_space::space::{
    BasisId,
    SpaceBasis,
    SpaceElement,
    SubspaceBasis,
};
use spin_algebra::{
    Spin,
    SpinPair,
    get_spin_basis,
    get_spin_pair_basis,
    get_spin_pair_magnitudes,
    half_integer::HalfU32,
};

pub type TwiceSpin = SpinPair<HalfU32, HalfU32>;

#[derive(Debug, Clone, Copy)]
pub struct AtomRecipe {
    pub s: HalfU32,
    pub i: HalfU32,
}

#[derive(Debug, Clone, Copy)]
// Struct for storing id of |s m_s>|i m_i> state
pub struct UncoupledAtomBasis {
    pub s: BasisId<Spin>,
    pub i: BasisId<Spin>,
}

impl UncoupledAtomBasis {
    /// Adds |s m_s>|i m_i>
    /// to the basis.
    pub fn new(recipe: AtomRecipe, basis: &mut SpaceBasis) -> Self {
        let s = get_spin_basis(recipe.s);
        let i = get_spin_basis(recipe.i);

        let s = basis.push_subspace(SubspaceBasis::new(s));
        let i = basis.push_subspace(SubspaceBasis::new(i));

        Self { s, i }
    }

    pub fn filter(&self, f: impl Fn((Spin, Spin)) -> bool) -> impl Fn(SpaceElement) -> bool {
        move |x| f((x[self.s], x[self.i]))
    }
}

#[derive(Debug, Clone, Copy)]
// Struct for storing id of |(s i) f m_f> state
pub struct CoupledAtomBasis {
    pub f: BasisId<TwiceSpin>,
}

impl CoupledAtomBasis {
    /// Adds |(s i) f m_f>
    /// to the basis.
    pub fn new(recipe: AtomRecipe, basis: &mut SpaceBasis) -> Self {
        let f = get_spin_pair_magnitudes([recipe.s], [recipe.i]);
        let f = get_spin_pair_basis(f);

        let f = basis.push_subspace(SubspaceBasis::new(f));

        Self { f }
    }

    pub fn filter(&self, f: impl Fn(TwiceSpin) -> bool) -> impl Fn(SpaceElement) -> bool {
        move |x| f(x[self.f])
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

    fn recipe() -> AtomRecipe {
        AtomRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
        }
    }

    #[test]
    fn test_uncoupled_atom_basis() {
        let mut basis = SpaceBasis::default();
        let atom = UncoupledAtomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atom.filter(|(s, i)| s.m + i.m == hi32!(1))(x));
        assert_eq!(elements.len(), 2);
        println!("{elements:?}");

        assert_eq!(elements[(0, atom.s)], spin!(hu32!(1 / 2), hi32!(1 / 2)));
        assert_eq!(elements[(0, atom.i)], spin!(hu32!(3 / 2), hi32!(1 / 2)));

        assert_eq!(elements[(1, atom.s)], spin!(hu32!(1 / 2), hi32!(-1 / 2)));
        assert_eq!(elements[(1, atom.i)], spin!(hu32!(3 / 2), hi32!(3 / 2)));
    }

    #[test]
    fn test_coupled_atom_basis() {
        let mut basis = SpaceBasis::default();
        let atom = CoupledAtomBasis::new(recipe(), &mut basis);

        let elements = basis.get_filtered_basis(|x| atom.filter(|f| f.m() == hi32!(1))(x));
        assert_eq!(elements.len(), 2);
        println!("{elements:?}");

        let element1 = spin!((hu32!(1 / 2), hu32!(3 / 2)), hu32!(1), hi32!(1));
        let element2 = spin!((hu32!(1 / 2), hu32!(3 / 2)), hu32!(2), hi32!(1));

        assert_eq!(elements[(0, atom.f)], element1);
        assert_eq!(elements[(1, atom.f)], element2);
    }
}
