use std::ops::{
    Add,
    AddAssign,
    Deref,
    DerefMut,
};

use cc_matrix_utils::MatrixLike;

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

    /// Perform U O U^T
    pub fn transform(&self, transformation: &Self) -> Self {
        Self(&transformation.0 * &self.0 * transformation.0.transpose())
    }

    /// Perform U^T O U
    pub fn transform_rev(&self, transformation: &Self) -> Self {
        Self(transformation.0.transpose() * &self.0 * &transformation.0)
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
/// - `cast_variant!($value, $type)`
#[macro_export]
macro_rules! cast_variant {
    ($value:expr, $type:ty) => {{ $value.downcast_ref::<$type>().expect("Could not downcast value") }};
}

/// Macro for casting multiple values into know variants
/// # Syntax
/// - `cast_variants!(($value, $type),*)`
#[macro_export]
macro_rules! cast_variants {
    ($($args:ident, $type:ty),* $(,)?) => {
        $(
            let $args = $crate::cast_variant!($args, $states);
        )*
    };
}

/// Macro for casting braket into known variants
/// # Syntax
/// - `cast_braket!($value, $type)`
#[macro_export]
macro_rules! cast_braket {
    ($value:expr, $type:ty) => {{
        let bra = $crate::cast_variant!($value.bra, $type);
        let ket = $crate::cast_variant!($value.ket, $type);

        $crate::operator::Braket { bra, ket }
    }};
}

/// Create operator from matrix elements in given basis
/// # Syntax
/// - `operator_mel!($basis, [$($action_elements),*], |[$($arg_braket),*]| $body)`
#[macro_export]
macro_rules! operator_mel {
    ($basis:expr, [$($elements:expr),*], |[$($args:ident),*]| $body:expr) => {
        $crate::operator::Operator::from_mel(
            &($basis.as_ref()),
            [$($elements.0),*],
            |[$($args),*]| {
                $(
                    let $args = $crate::operator::Braket {
                        bra: $elements.cast($args.bra),
                        ket: $elements.cast($args.ket),
                    };
                )*

                $body
            }
        )
    };
}

/// Create diagonal operator from matrix elements in given basis
/// # Syntax
/// - `operator_diag_mel!($basis, [$($action_elements),*], |[$($args),*]| $body)`
#[macro_export]
macro_rules! operator_diag_mel {
    ($basis:expr, [$($elements:expr),*], |[$($args:ident),*]| $body:expr) => {
        $crate::operator::Operator::from_diag_mel(
            &($basis.as_ref()),
            [$($elements.0),*],
            |[$($args),*]| {
                $(
                    let $args = $elements.cast($args);
                )*

                $body
            }
        )
    };
}

/// Create transformation operator from matrix elements in given basis
/// # Syntax
/// - `operator_transform_mel!($basis, [$($elements),*], $basis_transform,
///   [$($elements_transform),*], |[$($arg),*], [$($arg_transf),*]| $body)`
#[macro_export]
macro_rules! operator_transform_mel {
    (
        $basis:expr, [$($elements:expr),*],
        $basis_transf:expr, [$($elements_transf:expr),*],
        |[$($args:ident),*], [$($args_transf:ident),*]|
        $body:expr
    ) => {
        $crate::operator::Operator::from_transform_mel(
            &($basis.as_ref()),
            [$($elements.0),*],
            &($basis_transf.as_ref()),
            [$($elements_transf.0),*],
            |[$($args),*], [$($args_transf),*]| {
                $(
                    let $args = $elements.cast($args);
                )*
                $(
                    let $args_transf = $elements_transf.cast($args_transf);
                )*

                $body
            }
        )
    };
}
