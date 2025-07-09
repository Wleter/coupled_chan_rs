use crate::operator::Operator;
use matrix_utils::MatrixCreation;
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

pub fn get_diagonal_mel<'a, const N: usize, F, E>(
    elements: &'a BasisElements,
    action_subspaces: [BasisId; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([&'a Box<dyn DynSubspaceElement>; N]) -> E + 'a,
    E: Zero,
{
    let action_indices = action_subspaces.map(|x| x.0 as usize);

    let subspaces_len = elements.basis.len();
    for subspace_id in action_indices {
        assert!(subspace_id < subspaces_len, "Action subspace ID is larger than subspace size")
    }

    move |i, j| {
        if i != j {
            return E::zero()
        }

        let states = action_subspaces.map(|index| &elements[(i, index)]);
        mat_element(states)
    }
}

pub fn get_transform_mel<'a, const N: usize, const M: usize, F, E>(
    elements: &'a BasisElements,
    elements_transform: &'a BasisElements,
    subspaces: [BasisId; N],
    subspaces_transform: [BasisId; M],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([&'a Box<dyn DynSubspaceElement>; N], [&'a Box<dyn DynSubspaceElement>; M]) -> E + 'a,
    E: Zero,
{
    let indices = subspaces.map(|x| x.0 as usize);
    let indices_transform = subspaces_transform.map(|x| x.0 as usize);

    let subspaces_len = elements.basis.len();
    assert_eq!(subspaces_len, elements.basis.len(), "Subspaces do not cover whole basis");
    for subspace_id in indices {
        assert!(subspace_id < subspaces_len, "Subspace ID is larger than subspace size")
    }

    let subspaces_transform_len = elements_transform.basis.len();
    assert_eq!(subspaces_transform_len, elements_transform.basis.len(), "Transformed subspaces do not cover whole transformed basis");
    for subspace_id in indices_transform {
        assert!(subspace_id < subspaces_transform_len, "Subspace ID is larger than subspace size")
    }

    move |i, j| {
        let states = subspaces.map(|index| &elements[(i, index)]);
        let states_transform = subspaces_transform.map(|index| &elements_transform[(j, index)]);

        mat_element(states, states_transform)
    }
}

impl<M> Operator<M> {
    pub fn from_mel<'a, E, const N: usize, F>(
        elements: &'a BasisElements,
        action_subspaces: [BasisId; N],
        mat_element: F,
    ) -> Self
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut([Braket<&'a Box<dyn DynSubspaceElement>>; N]) -> E + 'a,
    {
        let mel = get_mel(elements, action_subspaces, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diag_mel<'a, E, const N: usize, F>(
        elements: &'a BasisElements,
        action_subspaces: [BasisId; N],
        mat_element: F,
    ) -> Self
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut([&'a Box<dyn DynSubspaceElement>; N]) -> E + 'a,
    {
        let mel = get_diagonal_mel(elements, action_subspaces, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_transform_mel<'a, E, const N: usize, const K: usize, F>(
        elements: &'a BasisElements,
        subspaces: [BasisId; N],
        elements_transform: &'a BasisElements,
        subspaces_transform: [BasisId; K],
        mat_element: F,
    ) -> Self 
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut([&'a Box<dyn DynSubspaceElement>; N], [&'a Box<dyn DynSubspaceElement>; K]) -> E + 'a,
    {
        let mel = get_transform_mel(elements, elements_transform, subspaces, subspaces_transform, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[macro_export]
macro_rules! operator_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_mel(
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

#[macro_export]
macro_rules! operator_diag_mel {
    (dyn $basis:expr, $elements:expr, |[$($args:ident: $subspaces:ty),*]| $body:expr) => {
        $crate::operator::Operator::from_diag_mel(
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
}

#[macro_export]
macro_rules! operator_transform_mel {
    (
        dyn $basis:expr, $elements:expr, 
        dyn $basis_transf:expr, $elements_transf:expr,
        |[$($args:ident: $subspaces:ty),*], [$($args_transf:ident: $subspaces_transf:ty),*]| 
        $body:expr
    ) => {
        $crate::operator::Operator::from_transform_mel(
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

        let operator_short: Operator<Mat<f64>> = operator_mel!(dyn &basis,
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

        let operator_diag: Operator<Mat<f64>> = operator_diag_mel!(dyn &basis,
            [e_id, vib_id],
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * vib.0 as f64
            }
        );
        let expected = mat![
            [8.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 8.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 8.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 10.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.8],
        ];
        assert_eq!(expected, operator_diag.backed);
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

        let operator_short: Operator<DMatrix<f64>> = operator_mel!(dyn &basis,
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

        let operator_diag: Operator<DMatrix<f64>> = operator_diag_mel!(dyn &basis,
            [e_id, vib_id],
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * vib.0 as f64
            }
        );
        let expected = DMatrix::from_vec(8, 8, vec![
            8.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 8.9, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 8.8, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 10.8, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.8, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.8,
        ]);
        assert_eq!(expected, operator_diag.backed);
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

        let operator_short: Operator<Array2<f64>> = operator_mel!(dyn &basis,
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

        let operator_diag: Operator<Array2<f64>> = operator_diag_mel!(dyn &basis,
            [e_id, vib_id],
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * vib.0 as f64
            }
        );
        let expected = Array2::from_shape_vec((8, 8), vec![
            8.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 10.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 8.9, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 10.9, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 8.8, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 10.8, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.8, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.8,
        ]).unwrap();
        assert_eq!(expected, operator_diag.backed);
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct CombinedSpin(u32, i32);

    #[test]
    fn test_dyn_transform_faer() {
        use faer::{mat, Mat};
        use crate::operator::Operator;

        let (basis, [e_id, n_id, vib_id]) = dyn_basis();

        let mut basis_transform = SpaceBasis::default();

        let s_basis = SubspaceBasis::new(vec![CombinedSpin(2, -2), CombinedSpin(2, 0), CombinedSpin(2, 2), CombinedSpin(0, 0)]);
        let s_transf_id = basis_transform.push_subspace(s_basis);

        let vib = SubspaceBasis::new(vec![Vibrational(-1), Vibrational(-2)]);
        let vib_transf_id = basis_transform.push_subspace(vib);
        let basis_transform = basis_transform.get_basis();
        println!("{basis_transform}");

        let transform: Operator<Mat<f64>> = operator_transform_mel!(
            dyn &basis, [e_id, n_id, vib_id],
            dyn &basis_transform, [s_transf_id, vib_transf_id],
            |[e: ElectronSpin, n: NuclearSpin, _vib: Vibrational], [s: CombinedSpin, _vib_t: Vibrational]| {
                if e.1 + n.1 != s.1 {
                    return 0.
                }

                s.0 as f64 + 0.1 * s.1 as f64 + e.0 as f64
            }
        );

        let expected = mat![
            [2.8, 0.0, 0.0, 0.0, 2.8, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 3.2, 0.0, 0.0, 0.0, 3.2, 0.0],
            [2.8, 0.0, 0.0, 0.0, 2.8, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 3.2, 0.0, 0.0, 0.0, 3.2, 0.0],
        ];
        assert_eq!(expected, transform.backed);
    }
}
