use cc_problems::{
    AngularMomentum,
    atom_structure::AtomBasisRecipe,
    coupled_chan::{composite_int::CompositeInt, dispersion::Dispersion, log_derivative::diabatic::Johnson},
    homo_diatom_basis::{AlkaliHomoDiatom, HomoDiatomRecipe},
    prelude::*,
    spin_algebra::{hi32, hu32},
};

fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Li2 collision",
    "Li2 Levels" => |_| Self::li2_levels(),
    "Li2 Feshbach" => |_| Self::li2_feshbach(),
    "Li2 Bound" => |_| Self::li2_bound(),
    "Li2 Field" => |_| Self::li2_field(),
    "Li2 Wave" => |_| Self::li2_wave(),
);

impl Problems {
    fn li2_levels() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());

        let mag_fields = linspace(0., 1200., 1001);
        let saver = DataSaver::new("data/li2_levels.jsonl", JsonFormat, FileAccess::Create)?;

        DependenceProblem::new(li2_problem).dependence(mag_fields, |problem, &field| {
            problem.set_b_field(field * Gauss);
            let w_matrix = problem.w_matrix();

            saver.send(LevelsData::new(field, w_matrix.asymptote().levels()));

            Ok(())
        })?;

        Ok(())
    }

    fn li2_feshbach() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());

        let mag_fields: Vec<f64> = vec![
            linspace(0., 600., 601),
            linspace(600.1, 630., 5000),
            linspace(631., 1200., 570),
        ]
        .into_iter()
        .flatten()
        .collect();

        let li2_scattering = li2_scattering();
        let saver = DataSaver::new("data/li2_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        DependenceProblem::new(li2_problem).dependence(mag_fields, |problem, &field| {
            problem.set_b_field(field * Gauss);
            let w_matrix = problem.w_matrix();

            let s_matrix = li2_scattering.get_s_matrix(&w_matrix, RatioNumerov::new);

            saver.send(SMatrixData::new(field, s_matrix));

            Ok(())
        })?;

        Ok(())
    }

    fn li2_bound() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());

        let mag_fields = linspace(0., 1200., 1201);
        let e_min = -12. * GHz;
        let e_max = 0. * GHz;
        let err = 1. * MHz;
        let li2_bound = li2_bound();

        let saver = DataSaver::new("data/li2_bound.jsonl", JsonFormat, FileAccess::Create)?;

        DependenceProblem::new(li2_problem).dependence(mag_fields, |problem, &field| {
            problem.set_b_field(field * Gauss);

            let bound_finder = li2_bound.get_bound_finder::<_, Johnson>(
                (e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()),
                err.to(AuEnergy).value(),
                |e| {
                    let mut problem = problem.clone();
                    problem.system_params.energy = e * AuEnergy;

                    problem.w_matrix()
                },
            );

            let bounds: Result<Vec<BoundState>> = bound_finder.bound_states().collect();

            for b in bounds? {
                let data = BoundStateData::new(field, b);
                saver.send(data)
            }

            Ok(())
        })?;

        Ok(())
    }

    fn li2_field() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());

        let energies: Vec<f64> = linspace((-2. * GHz).to(AuEnergy).value().cbrt(), (0. * GHz).to(AuEnergy).value(), 101)
            .iter()
            .map(|&x| x.powi(3))
            .collect();

        let mag_min = 0.;
        let mag_max = 1200.;
        let err = 1e-2;
        let li2_bound = li2_bound();

        let saver = DataSaver::new("data/li2_field.jsonl", JsonFormat, FileAccess::Create)?;

        DependenceProblem::new(li2_problem).dependence(energies, |problem, &energy| {
            problem.system_params.energy = energy * AuEnergy;

            let bound_finder = li2_bound.get_bound_finder::<_, Johnson>((mag_min, mag_max), err, |f| {
                let mut problem = problem.clone();
                problem.set_b_field(f * Gauss);

                problem.w_matrix()
            });

            let bounds: Result<Vec<BoundState>> = bound_finder.bound_states().collect();

            for b in bounds? {
                let data = BoundStateData::new(energy, b);
                saver.send(data)
            }

            Ok(())
        })?;

        Ok(())
    }

    fn li2_wave() -> Result<()> {
        let mut li2_problem = li2_problem(li2_recipe());
        li2_problem.set_b_field(600. * Gauss);

        let e_min = -12. * GHz;
        let e_max = 0. * GHz;
        let err = 1. * MHz;
        let li2_bound = li2_bound();

        let saver = DataSaver::new("data/li2_wave_600G.jsonl", JsonFormat, FileAccess::Create)?;

        let bound_finder = li2_bound.get_bound_finder::<_, Johnson>(
            (e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()),
            err.to(AuEnergy).value(),
            |e| {
                let mut problem = li2_problem.clone();
                problem.system_params.energy = e * AuEnergy;

                problem.w_matrix()
            },
        );

        for b in bound_finder.bound_states() {
            let wave = bound_finder.bound_wave(&b.unwrap());

            saver.send(wave)
        }

        Ok(())
    }
}

pub fn li2_problem(recipe: HomoDiatomRecipe) -> AlkaliHomoDiatom<impl Interaction + Clone, impl Interaction + Clone> {
    let triplet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(2.19348e8, -12)]);
    let singlet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(1.112e7, -12)]);

    let mut diatom = AlkaliHomoDiatom::new(triplet, singlet, recipe);

    diatom.atom.hyperfine.a_hifi = (228.2 / 1.5 * MHz).to(AuEnergy);
    diatom.system_params.energy = (1e-7 * Kelvin).to(AuEnergy);
    diatom.system_params.mass = (6.015122 / 2. * Dalton).to(AuMass);
    diatom.system_params.entrance_channel = 0;

    diatom
}

pub fn li2_recipe() -> HomoDiatomRecipe {
    HomoDiatomRecipe {
        atom: AtomBasisRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1),
        },
        l_max: AngularMomentum(0),
        tot_projection: hi32!(0),
    }
}

pub fn li2_scattering() -> ScatteringProblem {
    ScatteringProblem {
        r_min: 4. * Bohr,
        r_max: 1.5e3 * Bohr,
        step_strat: LocalWavelengthStep::new(1e-4, f64::INFINITY, 400.).into(),
    }
}

pub fn li2_bound() -> BoundProblem {
    BoundProblem {
        r_min: 4. * Bohr,
        r_match: 20. * Bohr,
        r_max: 1.5e3 * Bohr,
        step_strat: LocalWavelengthStep::new(1e-4, 10., 400.).into(),
        node_range: None,
        node_monotony: NodeMonotony::Increasing,
    }
}

// truth table calculations from based on results from commit 8f1ecc4
#[cfg(test)]
mod tests {
    use cc_problems::coupled_chan::log_derivative::diabatic::ManolopoulosLogDerivative;
    use math_utils::assert_approx_eq;

    use super::*;

    const ASYMPTOTE_0: [f64; 5] = [
        -4.6243315181781235e-8,
        -1.1560828795445314e-8,
        -1.1560828795445304e-8,
        2.3121657590890608e-8,
        2.312165759089062e-8,
    ];

    const ASYMPTOTE_500: [f64; 5] = [
        -2.270294714100955e-7,
        -2.2987368572154435e-8,
        -1.3428901873619304e-10,
        2.312165759089062e-8,
        2.0390781381920488e-7,
    ];

    const ASYMPTOTE_1000: [f64; 5] = [
        -4.387466353978883e-7,
        -2.308771464894053e-8,
        -3.394294195009041e-11,
        2.312165759089064e-8,
        4.15624977806998e-7,
    ];

    #[test]
    fn test_li2_levels() {
        let mut li2_problem = li2_problem(li2_recipe());

        for (field, result) in [(0., ASYMPTOTE_0), (500., ASYMPTOTE_500), (1000., ASYMPTOTE_1000)] {
            li2_problem.set_b_field(field * Gauss);
            let w_matrix = li2_problem.w_matrix();
            let levels = w_matrix.asymptote().levels();

            assert!(levels.l.iter().all(|&x| x == 0));
            assert_approx_eq!(iter => levels.asymptote, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_NUMEROV: [f64; 3] = [-7.199240787436895, -283.4443841387517, -5286.252495345484];

    #[test]
    fn test_li2_numerov() {
        let mut li2_problem = li2_problem(li2_recipe());
        let li2_scattering = li2_scattering();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_NUMEROV) {
            li2_problem.set_b_field(field * Gauss);
            let w_matrix = li2_problem.w_matrix();
            let s_matrix = li2_scattering.get_s_matrix(&w_matrix, |w, s, b| RatioNumerov::new(w, s, b));

            assert_approx_eq!(s_matrix.get_scattering_length().re, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_JOHNSON: [f64; 3] = [-7.19409243425113, -283.2910419145843, -5283.14763314261];

    #[test]
    fn test_li2_johnson() {
        let mut li2_problem = li2_problem(li2_recipe());
        let li2_scattering = li2_scattering();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_JOHNSON) {
            li2_problem.set_b_field(field * Gauss);
            let w_matrix = li2_problem.w_matrix();
            let s_matrix = li2_scattering.get_s_matrix(&w_matrix, |w, s, b| JohnsonLogDerivative::new(w, s, b));

            assert_approx_eq!(s_matrix.get_scattering_length().re, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_MANOLOPOULOS: [f64; 3] = [-7.185909206685685, -283.1042109930902, -5280.378031416163];
    #[test]
    fn test_li2_manolopoulos() {
        let mut li2_problem = li2_problem(li2_recipe());
        let li2_scattering = li2_scattering();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_MANOLOPOULOS) {
            li2_problem.set_b_field(field * Gauss);
            let w_matrix = li2_problem.w_matrix();
            let s_matrix = li2_scattering.get_s_matrix(&w_matrix, |w, s, b| ManolopoulosLogDerivative::new(w, s, b));

            assert_approx_eq!(s_matrix.get_scattering_length().re, result, 1e-6);
        }
    }

    #[test]
    fn test_li2_bound() {
        let bound_states_600 = [
            BoundState {
                parameter: -1.4658787668201906e-8,
                node: 77,
                occupations: Some(vec![
                    0.7834516574720637,
                    0.00063549424270599,
                    0.0022352869919129207,
                    0.17297119811001713,
                    0.04070636318329788,
                ]),
            },
            BoundState {
                parameter: -8.881979759726907e-9,
                node: 78,
                occupations: Some(vec![
                    0.056575344519407586,
                    3.7644021693278268e-6,
                    0.0014968869912998207,
                    0.001606564738229966,
                    0.9403174393488927,
                ]),
            },
        ];

        let mut li2_problem = li2_problem(li2_recipe());
        li2_problem.set_b_field(600. * Gauss);

        let e_min = -12. * GHz;
        let e_max = 0. * GHz;
        let err = 1. * MHz;
        let li2_bound = li2_bound();

        let bound_finder = li2_bound.get_bound_finder::<_, Johnson>(
            (e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()),
            err.to(AuEnergy).value(),
            |e| {
                let mut problem = li2_problem.clone();
                problem.system_params.energy = e * AuEnergy;

                problem.w_matrix()
            },
        );

        let bound_states: Result<Vec<BoundState>> = bound_finder.bound_states().collect();
        let bound_states: Vec<BoundState> = bound_states
            .unwrap()
            .into_iter()
            .map(|mut b| {
                b.occupations = Some(bound_finder.bound_wave(&b).occupations());
                b
            })
            .collect();

        assert_eq!(bound_states.len(), 2);
        assert_eq!(bound_states[0].node, bound_states_600[0].node);
        assert_eq!(bound_states[1].node, bound_states_600[1].node);

        assert_approx_eq!(bound_states[0].parameter, bound_states_600[0].parameter, 1e-6);
        assert_approx_eq!(bound_states[1].parameter, bound_states_600[1].parameter, 1e-6);

        assert_approx_eq!(iter => bound_states[0].occupations.as_ref().unwrap(), bound_states_600[0].occupations.as_ref().unwrap(), 1e-6);
        assert_approx_eq!(iter => bound_states[1].occupations.as_ref().unwrap(), bound_states_600[1].occupations.as_ref().unwrap(), 1e-6);
    }

    #[test]
    fn test_li2_field_bound() {
        let bound_states_60 = [
            BoundState {
                parameter: 616.9785468255947,
                node: 77,
                occupations: Some(vec![
                    0.6630758988532024,
                    0.0005365589754639324,
                    0.002348626419279133,
                    0.29367257011049147,
                    0.040366345641559456,
                ]),
            },
            BoundState {
                parameter: 599.4381158015825,
                node: 78,
                occupations: Some(vec![
                    0.05619324675661696,
                    3.85549448131846e-6,
                    0.0014947905890375138,
                    0.0015907451195761205,
                    0.9407173620402844,
                ]),
            },
        ];

        let mut li2_problem = li2_problem(li2_recipe());
        li2_problem.system_params.energy = (-60. * MHz).to(AuEnergy);

        let mag_min = 0.;
        let mag_max = 1200.;
        let err = 1e-2;
        let li2_bound = li2_bound();

        let bound_finder = li2_bound.get_bound_finder::<_, Johnson>((mag_min, mag_max), err, |f| {
            let mut problem = li2_problem.clone();
            problem.set_b_field(f * Gauss);

            problem.w_matrix()
        });

        let bound_states: Result<Vec<BoundState>> = bound_finder.bound_states().collect();
        let bound_states: Vec<BoundState> = bound_states
            .unwrap()
            .into_iter()
            .map(|mut b| {
                b.occupations = Some(bound_finder.bound_wave(&b).occupations());
                b
            })
            .collect();

        assert_eq!(bound_states.len(), 2);
        assert_eq!(bound_states[0].node, bound_states_60[0].node);
        assert_eq!(bound_states[1].node, bound_states_60[1].node);

        assert_approx_eq!(bound_states[0].parameter, bound_states_60[0].parameter, 1e-6);
        assert_approx_eq!(bound_states[1].parameter, bound_states_60[1].parameter, 1e-6);

        assert_approx_eq!(iter => bound_states[0].occupations.as_ref().unwrap(), bound_states_60[0].occupations.as_ref().unwrap(), 1e-6);
        assert_approx_eq!(iter => bound_states[1].occupations.as_ref().unwrap(), bound_states_60[1].occupations.as_ref().unwrap(), 1e-6);
    }
}
