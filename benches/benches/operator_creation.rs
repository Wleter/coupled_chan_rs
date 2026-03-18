use diol::prelude::*;
use hilbert_space::faer::Mat;
use spin_algebra::{
    dot,
    get_spin_basis,
    half_integer::{
        HalfI32,
        HalfU32,
    },
};

fn main() -> eyre::Result<()> {
    let bench = Bench::new(Config::from_args()?);

    bench.register("dynamic operator", dynamic_operator, [2, 4, 8, 16]);
    bench.register("manual operator", manual_operator, [2, 4, 8, 16]);

    bench.run()?;
    Ok(())
}

use hilbert_space::{
    operator::Operator,
    operator_mel,
    space::{
        BasisElements,
        SpaceBasis,
        SubspaceBasis,
    },
};

pub fn dynamic_operator(bencher: Bencher, size: u32) {
    let spins = get_spin_basis(HalfU32::from_doubled(size));

    let mut basis = SpaceBasis::default();
    let s1_id = basis.push_subspace(SubspaceBasis::new(spins.clone()));
    let s2_id = basis.push_subspace(SubspaceBasis::new(spins.clone()));
    let s3_id = basis.push_subspace(SubspaceBasis::new(spins.clone()));
    let s4_id = basis.push_subspace(SubspaceBasis::new(spins.clone()));

    let basis: BasisElements = basis.get_filtered_basis(|elements| {
        let s1 = elements[s1_id];
        let s2 = elements[s2_id];
        let s3 = elements[s3_id];
        let s4 = elements[s4_id];

        (s1.m + s2.m + s3.m + s4.m).double_value() == 0
    });

    bencher.bench(|| {
        let mut operator: Operator<Mat<f64>> = operator_mel!(&basis, [s2_id, s4_id], |[s2, s4]| dot(s2, s4));

        black_box(&mut operator);
    });
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
