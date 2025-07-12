use std::mem::discriminant;

use crate::{operator::Operator, static_space::BasisElement};
use matrix_utils::MatrixCreation;
use num_traits::Zero;

use crate::{operator::Braket, static_space::BasisElements};

pub fn get_mel<'a, const N: usize, T, F, E>(
    elements: &'a BasisElements<T>,
    action_subspaces: [T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([Braket<&'a T>; N]) -> E + 'a,
    E: Zero,
    T: PartialEq,
{
    let first = elements.first().expect("0 sized basis size is not allowed");

    let action_indices = action_subspaces.map(|s| {
        first
            .iter()
            .enumerate()
            .find(|(_, x)| discriminant(*x) == discriminant(&s))
            .map_or_else(|| panic!("Many subspaces with same variant are not allowed"), |x| x.0)
    });

    let diagonal_indices: Vec<usize> = (0..first.len()).filter(|x| !action_indices.contains(x)).collect();

    move |i, j| unsafe {
        let elements_i = elements.get_unchecked(i);
        let elements_j = elements.get_unchecked(j);

        for &index in &diagonal_indices {
            if elements_i.get_unchecked(index) != elements_j.get_unchecked(index) {
                return E::zero();
            }
        }

        let brakets = action_indices.map(|index| {
            let bra = elements_i.get_unchecked(index);
            let ket = elements_j.get_unchecked(index);

            Braket { bra, ket }
        });

        mat_element(brakets)
    }
}

pub fn get_diagonal_mel<'a, const N: usize, T, F, E>(
    elements: &'a BasisElements<T>,
    action_subspaces: [T; N],
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut([&'a T; N]) -> E + 'a,
    E: Zero,
{
    let first = elements.first().expect("0 sized basis size is not allowed");

    let action_indices = action_subspaces.map(|s| {
        first
            .iter()
            .enumerate()
            .find(|(_, x)| discriminant(*x) == discriminant(&s))
            .map_or_else(|| panic!("Many subspaces with same variant are not allowed"), |x| x.0)
    });

    move |i, j| {
        if i != j {
            return E::zero();
        }

        unsafe {
            let elements_i = elements.get_unchecked(i);

            let ket = action_indices.map(|index| elements_i.get_unchecked(index));

            mat_element(ket)
        }
    }
}

pub fn get_transform_mel<'a, T, O, F, E>(
    elements: &'a BasisElements<T>,
    elements_transform: &'a BasisElements<O>,
    mut mat_element: F,
) -> impl FnMut(usize, usize) -> E + 'a
where
    F: FnMut(&BasisElement<T>, &BasisElement<O>) -> E + 'a,
    E: Zero,
{
    move |i, j| unsafe {
        let elements_i = elements_transform.get_unchecked(i);
        let elements_j = elements.get_unchecked(j);

        mat_element(elements_j, elements_i)
    }
}

impl<M> Operator<M> {
    pub fn from_mel<'a, E, const N: usize, T, F>(
        elements: &'a BasisElements<T>,
        action_subspaces: [T; N],
        mat_element: F,
    ) -> Self
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut([Braket<&'a T>; N]) -> E + 'a,
        T: PartialEq,
    {
        let mel = get_mel(elements, action_subspaces, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_diag_mel<'a, E, const N: usize, T, F>(
        elements: &'a BasisElements<T>,
        action_subspaces: [T; N],
        mat_element: F,
    ) -> Self
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut([&'a T; N]) -> E + 'a,
    {
        let mel = get_diagonal_mel(elements, action_subspaces, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }

    pub fn from_transform_mel<'a, E, T1, T2, F>(
        elements: &'a BasisElements<T1>,
        elements_transform: &'a BasisElements<T2>,
        mat_element: F,
    ) -> Self
    where
        E: Zero,
        M: MatrixCreation<E>,
        F: FnMut(&BasisElement<T1>, &BasisElement<T2>) -> E + 'a,
        T1: PartialEq,
        T2: PartialEq,
    {
        let mel = get_transform_mel(elements, elements_transform, mat_element);
        let mat = M::from_fn(elements.len(), elements.len(), mel);

        Self { backed: mat }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        operator_transform_mel,
        static_space::{BasisElements, SpaceBasis, SubspaceBasis},
    };

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Basis {
        ElectronSpin((u32, i32)),
        NuclearSpin((u32, i32)),
        Vibrational(i32),
    }

    use Basis::*;

    fn static_basis() -> BasisElements<Basis> {
        let mut basis = SpaceBasis::default();

        let e_basis = SubspaceBasis::new(vec![ElectronSpin((1, -1)), ElectronSpin((1, 1))]);
        basis.push_subspace(e_basis);

        let nuclear = SubspaceBasis::new(vec![NuclearSpin((1, -1)), NuclearSpin((1, 1))]);
        basis.push_subspace(nuclear);

        let vib = SubspaceBasis::new(vec![Vibrational(-1), Vibrational(-2)]);
        basis.push_subspace(vib);

        basis.get_basis()
    }

    #[test]
    #[cfg(feature = "faer")]
    #[rustfmt::skip]
    fn test_static_operator_faer() {
        use faer::{mat, Mat};
        use crate::{cast_braket, operator::Operator, operator_diag_mel, operator_mel};

        let basis = static_basis();

        let operator = Operator::<Mat<f64>>::from_mel(
            &basis,
            [ElectronSpin((0, 0)), Vibrational(0)],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(e_braket, ElectronSpin);
                let vib_braket = cast_braket!(vib_braket, Vibrational);

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

        let operator_short: Operator<Mat<f64>> = operator_mel!(&basis,
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

        let operator_diag: Operator<Mat<f64>> = operator_diag_mel!(&basis,
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * *vib as f64
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
    fn test_static_operator_nalgebra() {
        use nalgebra::{DMatrix};
        use crate::{cast_braket, operator::Operator, operator_diag_mel, operator_mel};

        let basis = static_basis();

        let operator = Operator::<DMatrix<f64>>::from_mel(
            &basis,
            [ElectronSpin((0, 0)), Vibrational(0)],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(e_braket, ElectronSpin);
                let vib_braket = cast_braket!(vib_braket, Vibrational);

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

        let operator_short: Operator<DMatrix<f64>> = operator_mel!(&basis,
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

        let operator_diag: Operator<DMatrix<f64>> = operator_diag_mel!(&basis,
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * *vib as f64
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
    fn test_static_operator_ndarray() {
        use ndarray::{Array2};
        use crate::{cast_braket, operator::Operator, operator_diag_mel, operator_mel};

        let basis = static_basis();

        let operator = Operator::<Array2<f64>>::from_mel(
            &basis,
            [ElectronSpin((0, 0)), Vibrational(0)],
            |[e_braket, vib_braket]| {
                let e_braket = cast_braket!(e_braket, ElectronSpin);
                let vib_braket = cast_braket!(vib_braket, Vibrational);

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

        let operator_short: Operator<Array2<f64>> = operator_mel!(&basis,
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

        let operator_diag: Operator<Array2<f64>> = operator_diag_mel!(&basis,
            |[e: ElectronSpin, vib: Vibrational]| {
                10. * e.0 as f64 + e.1 as f64 + 0.1 * *vib as f64
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
    enum CoupledBasis {
        CombinedSpin((u32, i32)),
        VibrationalC(i32),
    }

    #[test]
    fn test_static_transform_faer() {
        use crate::operator::Operator;
        use CoupledBasis::*;
        use faer::{Mat, mat};

        let basis = static_basis();

        let mut basis_transform = SpaceBasis::default();

        let s_basis = SubspaceBasis::new(vec![
            CombinedSpin((2, -2)),
            CombinedSpin((2, 0)),
            CombinedSpin((2, 2)),
            CombinedSpin((0, 0)),
        ]);
        basis_transform.push_subspace(s_basis);

        let vib = SubspaceBasis::new(vec![VibrationalC(-1), VibrationalC(-2)]);
        basis_transform.push_subspace(vib);
        let basis_transform = basis_transform.get_basis();

        let transform: Operator<Mat<f64>> = operator_transform_mel!(
            &basis, &basis_transform,
            |[e: ElectronSpin, n: NuclearSpin, _vib: Vibrational], [s: CombinedSpin, _vib_t: VibrationalC]| {
                if e.1 + n.1 != s.1 {
                    return 0.
                }
                let factor = if _vib * _vib_t == 2 {
                    -1.
                } else {
                    1.
                };

                factor * (s.0 as f64 + 0.1 * s.1 as f64 + e.0 as f64)
            }
        );

        let expected = mat![
            [2.8, 0.0, 0.0, 0.0, -2.8, 0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0, 0.0, 0.0, -3.0, -3.0, 0.0],
            [0.0, 0.0, 0.0, 3.2, 0.0, 0.0, 0.0, -3.2],
            [0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0],
            [-2.8, 0.0, 0.0, 0.0, 2.8, 0.0, 0.0, 0.0],
            [0.0, -3.0, -3.0, 0.0, 0.0, 3.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, -3.2, 0.0, 0.0, 0.0, 3.2],
            [0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ];
        assert_eq!(expected, transform.backed);
    }
}
