#[cfg(any(feature = "faer", feature = "nalgebra", feature = "ndarray"))]
use crate::operator::Operator;
use num_traits::Zero;

use crate::{
    dyn_space::{BasisElements, BasisId, DynSubspaceElement},
    operator::Braket,
};

pub fn get_mel<'a, const N: usize, F, E>(
    elements: &'a BasisElements,
    action_subspaces: [BasisId; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([Braket<&'a Box<dyn DynSubspaceElement>>; N]) -> E + 'a,
    E: Zero,
{
    let action_indices = action_subspaces.map(|x| x.0 as usize);

    let subspaces_len = elements.basis.len();
    for subspace_id in action_indices {
        assert!(subspace_id < subspaces_len, "Action subspace ID is larger than subspace size")
    }

    let diagonal_indices: Vec<usize> = (0..subspaces_len).filter(|x| !action_indices.contains(x)).collect();

    move |i, j| unsafe {
        let indices_i = elements.elements_indices.get_unchecked(i);
        let indices_j = elements.elements_indices.get_unchecked(j);

        for &index in &diagonal_indices {
            if indices_i.get_unchecked(index) != indices_j.get_unchecked(index) {
                return E::zero();
            }
        }

        let brakets = action_subspaces.map(|index| {
            let bra = &elements[(i, index)];
            let ket = &elements[(j, index)];

            Braket { bra, ket }
        });

        mat_element(brakets)
    }
}

#[cfg(feature = "faer")]
use faer::Mat;

#[cfg(feature = "faer")]
impl<E: Zero> Operator<Mat<E>> {
    pub fn from_mel<'a, const N: usize, F>(
        elements: &'a BasisElements,
        action_subspaces: [BasisId; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<&'a Box<dyn DynSubspaceElement>>; N]) -> E + 'a,
    {
        let mel = get_mel(elements, action_subspaces, mat_element);
        let mat = Mat::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, Scalar};

#[cfg(feature = "nalgebra")]
impl<E: Scalar + Zero> Operator<DMatrix<E>> {
    pub fn from_mel<'a, const N: usize, F>(
        elements: &'a BasisElements,
        action_subspaces: [BasisId; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<&'a Box<dyn DynSubspaceElement>>; N]) -> E + 'a,
    {
        let mel = get_mel(elements, action_subspaces, mat_element);
        let mat = DMatrix::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[cfg(feature = "ndarray")]
use ndarray::Array2;

#[cfg(feature = "ndarray")]
impl<E: Scalar + Zero> Operator<Array2<E>> {
    pub fn from_mel<'a, const N: usize, F>(
        elements: &'a BasisElements,
        action_subspaces: [BasisId; N],
        mat_element: F,
    ) -> Self
    where
        F: FnMut([Braket<&'a Box<dyn DynSubspaceElement>>; N]) -> E + 'a,
    {
        let mut mel = get_mel(elements, action_subspaces, mat_element);
        let mat = Array2::from_shape_fn((elements.len(), elements.len()), |(i, j)| mel(i, j));

        Self { backed: mat }
    }
}

#[macro_export]
macro_rules! operator_mel {
    ($backed:ty, dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::<$backed>::from_mel(
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
}

#[cfg(test)]
mod tests {
    use crate::dyn_space::{BasisElements, BasisId, SpaceBasis, SubspaceBasis};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ElectronSpin(u32, i32);
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct NuclearSpin(u32, i32);
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Vibrational(i32);

    fn dyn_basis() -> (BasisElements, [BasisId; 3]) {
        let mut basis = SpaceBasis::default();

        let e_basis = SubspaceBasis::new(vec![ElectronSpin(1, -1), ElectronSpin(1, 1)]);
        let e_id = basis.push_subspace(e_basis);

        let nuclear = SubspaceBasis::new(vec![NuclearSpin(1, -1), NuclearSpin(1, 1)]);
        let n_id = basis.push_subspace(nuclear);

        let vib = SubspaceBasis::new(vec![Vibrational(-1), Vibrational(-2)]);
        let vib_id = basis.push_subspace(vib);

        (basis.get_basis(), [e_id, n_id, vib_id])
    }

    #[test]
    #[cfg(feature = "faer")]
    #[rustfmt::skip]
    fn test_dyn_operator_faer() {
        use faer::{mat, Mat};
        use crate::{cast_braket, operator::Operator};

        let (basis, [e_id, _, vib_id]) = dyn_basis();

        let operator = Operator::<Mat<f64>>::from_mel(
            &basis,
            [e_id, vib_id],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(dyn e_braket, ElectronSpin);
                let vib_braket = cast_braket!(dyn vib_braket, Vibrational);

                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            },
        );

        let expected = mat![
            [0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0],
            [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0],
            [1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
            [1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1],
        ];

        assert_eq!(expected, operator.backed);

        let operator_short = operator_mel!(Mat<f64>,
            dyn &basis,
            [e_id, vib_id],
            |[e_braket: ElectronSpin, vib_braket: Vibrational]| {
                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            }
        );

        assert_eq!(operator_short.backed, operator.backed);
    }

    #[test]
    #[cfg(feature = "nalgebra")]
    #[rustfmt::skip]
    fn test_dyn_operator_nalgebra() {
        use nalgebra::{DMatrix};
        use crate::{cast_braket, operator::Operator};

        let (basis, [e_id, _, vib_id]) = dyn_basis();

        let operator = Operator::<DMatrix<f64>>::from_mel(
            &basis,
            [e_id, vib_id],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(dyn e_braket, ElectronSpin);
                let vib_braket = cast_braket!(dyn vib_braket, Vibrational);

                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            },
        );

        let expected = DMatrix::from_vec(8, 8,
            vec![
                0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0,
                0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0,
                0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0,
                1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1,
                0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1
            ]).transpose();

        assert_eq!(expected, operator.backed);

        let operator_short = operator_mel!(DMatrix<f64>,
            dyn &basis,
            [e_id, vib_id],
            |[e_braket: ElectronSpin, vib_braket: Vibrational]| {
                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            }
        );

        assert_eq!(operator_short.backed, operator.backed);
    }

    #[test]
    #[cfg(feature = "ndarray")]
    #[rustfmt::skip]
    fn test_dyn_operator_ndarray() {
        use ndarray::{Array2};
        use crate::{cast_braket, operator::Operator};

        let (basis, [e_id, _, vib_id]) = dyn_basis();

        let operator = Operator::<Array2<f64>>::from_mel(
            &basis,
            [e_id, vib_id],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(dyn e_braket, ElectronSpin);
                let vib_braket = cast_braket!(dyn vib_braket, Vibrational);

                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            },
        );

        let expected = Array2::from_shape_vec((8, 8),
            vec![
                0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0,
                0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0,
                0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1089.0, 1109.0,
                0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 1091.0, 1111.0,
                1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 1089.0, 1109.0, 0.0, 0.0, 0.1, 0.1,
                0.0, 0.0, 1091.0, 1111.0, 0.0, 0.0, 0.1, 0.1
            ]
        ).unwrap();

        assert_eq!(expected, operator.backed);

        let operator_short = operator_mel!(Array2<f64>,
            dyn &basis,
            [e_id, vib_id],
            |[e_braket: ElectronSpin, vib_braket: Vibrational]| {
                if vib_braket.ket != vib_braket.bra {
                    ((e_braket.ket.0 * 1000 + e_braket.bra.0 * 100) as i32 + e_braket.ket.1 * 10 + e_braket.bra.1)
                        as f64
                } else {
                    0.1
                }
            }
        );

        assert_eq!(operator_short.backed, operator.backed);
    }
}
