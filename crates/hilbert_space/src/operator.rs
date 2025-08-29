use std::ops::{Add, AddAssign, Deref, DerefMut};

use matrix_utils::MatrixLike;

pub mod dyn_operator;
pub mod static_operator;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Braket<T> {
    pub bra: T,
    pub ket: T,
}

impl<T> Braket<T> {
    pub fn new(bra: T, ket: T) -> Self {
        Self { bra, ket }
    }
}

impl<T: PartialEq> Braket<T> {
    pub fn is_diagonal(&self) -> bool {
        self.ket == self.bra
    }
}

pub fn kron_delta<T: PartialEq, const N: usize>(brakets: [Braket<T>; N]) -> f64 {
    if brakets.iter().all(|x| x.is_diagonal()) { 1.0 } else { 0.0 }
}

pub fn into_variant<V, T>(elements: Vec<V>, variant: fn(V) -> T) -> Vec<T> {
    elements.into_iter().map(variant).collect()
}

#[derive(Debug, Clone)]
pub struct Operator<M: MatrixLike>(pub M);

impl<M: MatrixLike> Deref for Operator<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M: MatrixLike> DerefMut for Operator<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<M: MatrixLike> Operator<M> {
    pub fn new(mat: M) -> Self {
        Self(mat)
    }
}

// todo! it is hard to combine faer, ndarray, nalgebra into single api,
// so for now I specialize in faer
#[cfg(feature = "faer")]
impl Operator<faer::Mat<f64>> {
    pub fn size(&self) -> usize {
        assert_eq!(self.0.nrows(), self.0.ncols(), "Mismatched number of columns vs rows");

        self.0.nrows()
    }

    pub fn zeros(size: usize) -> Self {
        Self(faer::Mat::zeros(size, size))
    }

    pub fn identity(size: usize) -> Self {
        Self(faer::Mat::identity(size, size))
    }

    pub fn transform(&self, transformation: &Self) -> Self {
        Self(&transformation.0 * &self.0 * transformation.0.transpose())
    }
}

impl<M: MatrixLike + AddAssign> AddAssign for Operator<M> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl<M: MatrixLike + Add<Output = M>> Add for Operator<M> {
    type Output = Operator<M>;

    fn add(self, rhs: Self) -> Self::Output {
        Operator(self.0 + rhs.0)
    }
}

/// Macro for casting given value into known variant
/// # Syntax
/// - `cast_variant!($value, $pat)`
/// - `cast_variant!(dyn $value, $type)`
#[macro_export]
macro_rules! cast_variant {
    ($value:expr, $pat:path) => {{
        if let $pat(a) = $value {
            a
        } else {
            unreachable!("Incorrect variant cast")
        }
    }};
    (dyn $value:expr, $type:ty) => {{ $value.downcast_ref::<$type>().expect("Could not downcast value") }};
}

/// Macro for casting multiple values into know variants
/// # Syntax
/// - `cast_variants!(($value, $pat),*)`
/// - `cast_variants!(dyn ($value, $type),*)`
#[macro_export]
macro_rules! cast_variants {
    ($($args:ident: $states:path),* $(,)?) => {
        $(
            let $args = $crate::cast_variant!($args, $states);
        )*
    };
    (dyn $($args:ident, $type:ty),* $(,)?) => {
        $(
            let $args = $crate::cast_variant!(dyn $args, $states);
        )*
    };
}

/// Macro for casting braket into known variants
/// # Syntax
/// - `cast_braket!($value, $pat)`
/// - `cast_braket!(dyn $value, $type)`
#[macro_export]
macro_rules! cast_braket {
    ($value:expr, $pat:path) => {{
        let bra = $crate::cast_variant!($value.bra, $pat);
        let ket = $crate::cast_variant!($value.ket, $pat);

        $crate::operator::Braket { bra, ket }
    }};
    (dyn $value:expr, $type:ty) => {{
        let bra = $crate::cast_variant!(dyn $value.bra, $type);
        let ket = $crate::cast_variant!(dyn $value.ket, $type);

        $crate::operator::Braket { bra, ket }
    }};
}

/// Create basis elements from the space that are filtered by condition
/// # Syntax
/// - `filter_space!(dyn $basis, |[$($basis_id: $subspaces),*]| $body)`
/// - `filter_space!($basis, |[$($args: $subspaces),*]| $body)`
#[macro_export]
macro_rules! filter_space {
    (dyn $basis:expr, |[$($basis_id:ident: $subspaces:ty),*]| $body:expr) => {
        $basis.get_filtered_basis(|x| {
            let mut i: usize = 0;
            $(
                let $basis_id = $crate::cast_variant!(dyn x[$basis_id.0 as usize], $subspaces);
                i += 1;
            )*
            assert_eq!(i, x.len(), "Not whole space for space filtering is defined");

            $body
        })
    };
    ($basis:expr, |[$($args:ident: $subspaces:path),*]| $body:expr) => {
        $basis.iter_elements().filter(|x| {
            let mut i: usize = 0;
            $(
                let $args = $crate::cast_variant!(x[i], $subspaces);
                i += 1;
            )*
            assert_eq!(i, x.len(), "Not whole space for space filtering is defined");

            $body
        }).collect()
    };
}

/// Create operator from matrix elements in given basis
/// # Syntax
/// - `operator_mel!(dyn $basis, $action_elements, |[($arg_braket: $subspace),*]| $body)`
/// - `operator_mel!($basis, |[($arg_braket: $subspace),*]| $body)`
#[macro_export]
macro_rules! operator_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_mel_dyn(
            &($basis.as_ref()),
            $elements,
            |[$($args),*]| {
                $(
                    let $args = $crate::cast_braket!(dyn $args, $subspaces);
                )*

                $body
            }
        )
    };
    ($basis:expr, |[$($args:ident: $subspaces:path),*]| $body:expr) => {
        $crate::operator::Operator::from_mel(
            $basis,
            [$($subspaces(Default::default())),*],
            |[$($args),*]| {
                $(
                    let $args = $crate::cast_braket!($args, $subspaces);
                )*

                $body
            }
        )
    };
}

/// Create diagonal operator from matrix elements in given basis
/// # Syntax
/// - `operator_mel!(dyn $basis, $action_elements, |[($arg: $subspace),*]| $body)`
/// - `operator_mel!($basis, |[($arg: $subspace),*]| $body)`
#[macro_export]
macro_rules! operator_diag_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_diag_mel_dyn(
            &($basis.as_ref()),
            $elements,
            |[$($args),*]| {
                $(
                    let $args = $crate::cast_variant!(dyn $args, $subspaces);
                )*

                $body
            }
        )
    };
    ($basis:expr, |[$($args:ident: $subspaces:path),*]| $body:expr) => {
        $crate::operator::Operator::from_diag_mel(
            $basis,
            [$($subspaces(Default::default())),*],
            |[$($args),*]| {
                $(
                    let $args = $crate::cast_variant!($args, $subspaces);
                )*

                $body
            }
        )
    };
}

/// Create transformation operator from matrix elements in given basis
/// # Syntax
/// - `operator_transform_mel!(dyn $basis, $elements,
///     dyn $basis_transform, $elements_transform,
///     |[($arg: $subspace),*], [($arg_transf: $subspace),*]| $body)`
/// - `operator_transform_mel!($basis, $basis_transform,
///     |[($arg: $subspace),*], [($arg_transf: $subspace),*]| $body)`
#[macro_export]
macro_rules! operator_transform_mel {
    (
        dyn $basis:expr, $elements:expr,
        dyn $basis_transf:expr, $elements_transf:expr,
        |[$($args:ident: $subspaces:ty),*], [$($args_transf:ident: $subspaces_transf:ty),*]|
        $body:expr
    ) => {
        $crate::operator::Operator::from_transform_mel_dyn(
            &($basis.as_ref()),
            $elements,
            &($basis_transf.as_ref()),
            $elements_transf,
            |[$($args),*], [$($args_transf),*]| {
                $(
                    let $args = $crate::cast_variant!(dyn $args, $subspaces);
                )*
                $(
                    let $args_transf = $crate::cast_variant!(dyn $args_transf, $subspaces_transf);
                )*

                $body
            }
        )
    };
    (
        $basis:expr, $basis_transf:expr,
        |[$($args:ident: $subspaces:path),*], [$($args_transf:ident: $subspaces_transf:path),*]|
        $body:expr
    ) => {
        $crate::operator::Operator::from_transform_mel(
            $basis,
            $basis_transf,
            |elements, elements_transf| {
                let mut i: usize = 0;
                $(
                    let $args = $crate::cast_variant!(elements[i], $subspaces);
                    i += 1;
                )*
                assert_eq!(i, elements.len(), "Not whole space for transformation is defined");

                let mut i: usize = 0;
                $(
                    let $args_transf = $crate::cast_variant!(elements_transf[i], $subspaces_transf);
                    i += 1;
                )*
                assert_eq!(i, elements_transf.len(), "Not whole space for transformation is defined");

                $body
            }
        )
    };
}
