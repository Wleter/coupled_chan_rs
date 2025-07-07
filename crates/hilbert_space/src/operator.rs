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
