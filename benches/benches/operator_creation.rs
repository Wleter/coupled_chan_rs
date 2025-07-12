use diol::prelude::*;
use hilbert_space::faer::Mat;
use spin_algebra::{
    get_spin_basis,
    half_integer::{HalfI32, HalfU32},
};

fn main() -> eyre::Result<()> {
    let bench = Bench::new(Config::from_args()?);

    bench.register("static operator", static_way::operator_static, [2, 4, 8, 16]);
    bench.register("dynamic operator", dynamic_way::dynamic_operator, [2, 4, 8, 16]);

    bench.register("manual operator", manual_operator, [2, 4, 8, 16]);

    bench.run()?;
    Ok(())
}

mod static_way {
    use diol::prelude::*;
    use hilbert_space::{
        cast_variant,
        faer::Mat,
        operator::{Operator, into_variant},
        operator_mel,
        static_space::{BasisElements, SpaceBasis, SubspaceBasis},
    };
    use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum StaticBasis {
        Spin1(Spin),
        Spin2(Spin),
        Spin3(Spin),
        Spin4(Spin),
    }

    pub fn operator_static(bencher: Bencher, size: u32) {
        let spins = get_spin_basis(HalfU32::from_doubled(size));

        let mut basis = SpaceBasis::default();
        basis
            .push_subspace(SubspaceBasis::new(into_variant(spins.clone(), StaticBasis::Spin1)))
            .push_subspace(SubspaceBasis::new(into_variant(spins.clone(), StaticBasis::Spin2)))
            .push_subspace(SubspaceBasis::new(into_variant(spins.clone(), StaticBasis::Spin3)))
            .push_subspace(SubspaceBasis::new(into_variant(spins, StaticBasis::Spin4)));

        let basis: BasisElements<StaticBasis> = basis
            .iter_elements()
            .filter(|x| {
                let state1 = cast_variant!(x[0], StaticBasis::Spin1);
                let state2 = cast_variant!(x[1], StaticBasis::Spin2);
                let state3 = cast_variant!(x[2], StaticBasis::Spin3);
                let state4 = cast_variant!(x[3], StaticBasis::Spin4);

                (state1.m + state2.m + state3.m + state4.m).double_value() == 0
            })
            .collect();

        bencher.bench(|| {
            let mut operator: Operator<Mat<f64>> = operator_mel!(&basis, |[s2: StaticBasis::Spin2, s4: StaticBasis::Spin4]| {
                SpinOps::dot(s2, s4)
            });

            black_box(&mut operator);
        });
    }
}

mod dynamic_way {
    use diol::prelude::*;
    use hilbert_space::{
        cast_variant,
        dyn_space::{BasisElements, SpaceBasis, SubspaceBasis},
        faer::Mat,
        operator::Operator,
        operator_mel,
    };
    use spin_algebra::{Spin, SpinOps, get_spin_basis, half_integer::HalfU32};

    pub fn dynamic_operator(bencher: Bencher, size: u32) {
        let spins = get_spin_basis(HalfU32::from_doubled(size));

        let mut basis = SpaceBasis::default();
        _ = basis.push_subspace(SubspaceBasis::new(spins.clone()));
        let s2 = basis.push_subspace(SubspaceBasis::new(spins.clone()));
        _ = basis.push_subspace(SubspaceBasis::new(spins.clone()));
        let s4 = basis.push_subspace(SubspaceBasis::new(spins.clone()));

        let basis: BasisElements = basis.get_filtered_basis(|x| {
            let state1 = cast_variant!(dyn x[0], Spin);
            let state2 = cast_variant!(dyn x[1], Spin);
            let state3 = cast_variant!(dyn x[2], Spin);
            let state4 = cast_variant!(dyn x[3], Spin);

            (state1.m + state2.m + state3.m + state4.m).double_value() == 0
        });

        bencher.bench(|| {
            let mut operator: Operator<Mat<f64>> = operator_mel!(dyn &basis, [s2, s4], |[s2: Spin, s4: Spin]| {
                SpinOps::dot(s2, s4)
            });

            black_box(&mut operator);
        });
    }
}

pub fn manual_operator(bencher: Bencher, size: u32) {
    let s = HalfU32::from_doubled(size);
    let m = get_spin_basis(s).iter().map(|x| x.m).collect::<Vec<HalfI32>>();

    let mut states = vec![];
    for &m1 in &m {
        for &m2 in &m {
            for &m3 in &m {
                for &m4 in &m {
                    if (m1 + m2 + m3 + m4).double_value() == 0 {
                        states.push([m1, m2, m3, m4]);
                    }
                }
            }
        }
    }

    bencher.bench(|| {
        let mut operator = Mat::from_fn(states.len(), states.len(), |i, j| unsafe {
            let ms_bra = *states.get_unchecked(i);
            let ms_ket = *states.get_unchecked(j);

            if ms_bra[0] != ms_ket[0] || ms_bra[2] != ms_ket[2] {
                return 0.0;
            }

            let mut value = 0.0;

            if ms_bra[1] == ms_ket[1] && ms_bra[3] == ms_ket[3] {
                value += ms_bra[1].value() * ms_bra[3].value()
            }

            if ms_bra[1].double_value() == ms_ket[1].double_value() + 2
                && ms_bra[3].double_value() + 2 == ms_ket[3].double_value()
            {
                value += (s.value() * (s.value() + 1.)
                    - ms_bra[1].value() * ms_ket[1].value() * s.value() * (s.value() + 1.)
                    - ms_bra[3].value() * ms_ket[3].value())
                .sqrt()
            }

            if ms_bra[1].double_value() + 2 == ms_ket[1].double_value()
                && ms_bra[3].double_value() == ms_ket[3].double_value() + 2
            {
                value += (s.value() * (s.value() + 1.)
                    - ms_bra[1].value() * ms_ket[1].value() * s.value() * (s.value() + 1.)
                    - ms_bra[3].value() * ms_ket[3].value())
                .sqrt()
            }

            value
        });

        black_box(&mut operator);
    });
}
