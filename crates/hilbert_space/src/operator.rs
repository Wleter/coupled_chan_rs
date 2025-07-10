pub mod dyn_operator;
pub mod static_operator;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Braket<T> {
    pub bra: T,
    pub ket: T,
}

impl<T: PartialEq> Braket<T> {
    pub fn is_diagonal(&self) -> bool {
        self.ket == self.bra
    }
}

pub fn kron_delta<T: PartialEq, const N: usize>(brakets: [Braket<T>; N]) -> f64 {
    if brakets.iter().all(|x| x.is_diagonal()) { 1.0 } else { 0.0 }
}

#[derive(Debug, Clone)]
pub struct Operator<M> {
    backed: M,
}

impl<M> Operator<M> {
    pub fn new(mat: M) -> Self {
        Self { backed: mat }
    }

    pub fn into_backed(self) -> M {
        self.backed
    }
}

/// Cast the expression `value` to the variant `pat` or panic if it is mismatched.
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

/// Cast the expression `value` to the variant `pat` or panic if it is mismatched.
/// # Syntax
/// - `cast_variant!($value, $pat)`
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

/// Cast the expression `value` braket to the variant `pat` or panic if it is mismatched.
/// # Syntax
/// - `cast_braket!($value, $pat)`
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

#[macro_export]
macro_rules! operator_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_mel_dyn(
            $basis,
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

#[macro_export]
macro_rules! operator_diag_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_diag_mel_dyn(
            $basis,
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

#[macro_export]
macro_rules! operator_transform_mel {
    (
        dyn $basis:expr, $elements:expr,
        dyn $basis_transf:expr, $elements_transf:expr,
        |[$($args:ident: $subspaces:ty),*], [$($args_transf:ident: $subspaces_transf:ty),*]|
        $body:expr
    ) => {
        $crate::operator::Operator::from_transform_mel_dyn(
            $basis,
            $elements,
            $basis_transf,
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