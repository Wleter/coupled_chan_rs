use cc_problems::{
    AngularMomentum, BoundStateData, LevelsData, SMatrixData,
    alkali_homo_diatom::{AlkaliHomoDiatom, AlkaliHomoDiatomBuilder, AlkaliHomoDiatomParams, AlkaliHomoDiatomRecipe},
    anyhow::Result,
    atom_structure::{AtomStructureParams, AtomStructureRecipe},
    bound_states::{BoundState, BoundStatesFinder},
    coupled_chan::{
        Interaction,
        composite_int::CompositeInt,
        constants::units::{
            Quantity,
            atomic_units::{AuEnergy, AuMass, Bohr, Dalton, GHz, Gauss, Kelvin, MHz},
        },
        coupling::WMatrix,
        dispersion::Dispersion,
        log_derivative::diabatic::JohnsonLogDerivative,
        propagator::{Direction, Propagator, step_strategy::LocalWavelengthStep},
        ratio_numerov::RatioNumerov,
        s_matrix::SMatrixGetter,
        vanishing_boundary,
    },
    linspace,
    qol_utils::{
        problem_selector::{ProblemSelector, get_args},
        problems_impl,
        saving::{DataSaver, FileAccess, JsonFormat},
    },
    spin_algebra::{hi32, hu32},
    system_structure::SystemParams,
};

use cc_problems::rayon::prelude::*;

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
        let li2_params = li2_params();

        let mag_fields = linspace(0., 1200., 1001);
        let saver = DataSaver::new("data/li2_levels.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields.par_iter().for_each_with(li2_params, |params, &field| {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));

            saver.send(LevelsData::new(field, w_matrix.asymptote().levels())).unwrap()
        });

        Ok(())
    }

    fn li2_feshbach() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());
        let li2_params = li2_params();

        let mag_fields: Vec<f64> = vec![
            linspace(0., 600., 601),
            linspace(600.1, 630., 5000),
            linspace(631., 1200., 570),
        ]
        .into_iter()
        .flatten()
        .collect();

        let saver = DataSaver::new("data/li2_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields.par_iter().for_each_with(li2_params, |params, &field| {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);
            let boundary = vanishing_boundary(Quantity(4., Bohr), Direction::Outwards, &w_matrix);

            let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
            let solution = propagator.propagate_to(Quantity(1500., Bohr).value());
            let s_matrix = solution.get_s_matrix(&w_matrix);

            saver.send(SMatrixData::new(field, s_matrix)).unwrap()
        });

        Ok(())
    }

    fn li2_bound() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());
        let li2_params = li2_params();

        let mag_fields = linspace(0., 1200., 1201);
        let e_min = Quantity(-12., GHz);
        let e_max = Quantity(0., GHz);
        let err = Quantity(1., MHz);
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);

        let saver = DataSaver::new("data/li2_bound.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields.par_iter().for_each_with(li2_params, |params, &field| {
            params.with_field(Quantity(field, Gauss));

            let bounds = BoundStatesFinder::default()
                .set_parameter_range(
                    [e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()],
                    err.to(AuEnergy).value(),
                )
                .set_problem(|e| {
                    let mut params = params.clone();
                    params.system.energy = Quantity(e, AuEnergy);

                    li2_problem.with_params(&params)
                })
                .set_r_range([Quantity(4., Bohr), Quantity(20., Bohr), Quantity(1.5e3, Bohr)])
                .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_strategy.into(), b));

            let bounds: Result<Vec<BoundState>> = bounds.bound_states().collect();

            for b in bounds.unwrap() {
                let data = BoundStateData::new(field, b);
                saver.send(data).unwrap()
            }
        });

        Ok(())
    }

    fn li2_field() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());
        let li2_params = li2_params();

        let energies: Vec<f64> = linspace(
            Quantity(-2., GHz).to(AuEnergy).value().cbrt(),
            Quantity(0., GHz).to(AuEnergy).value(),
            101,
        )
        .iter()
        .map(|&x| x.powi(3))
        .collect();

        let mag_min = 0.;
        let mag_max = 1200.;
        let err = 1e-2;
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);

        let saver = DataSaver::new("data/li2_field.jsonl", JsonFormat, FileAccess::Create)?;

        energies.par_iter().for_each_with(li2_params, |params, &energy| {
            params.system.energy = Quantity(energy, AuEnergy);

            let bounds = BoundStatesFinder::default()
                .set_parameter_range([mag_min, mag_max], err)
                .set_problem(|f| {
                    let mut params = params.clone();
                    params.with_field(Quantity(f, Gauss));

                    li2_problem.with_params(&params)
                })
                .set_r_range([Quantity(4., Bohr), Quantity(20., Bohr), Quantity(1.5e3, Bohr)])
                .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_strategy.into(), b));

            let bounds: Result<Vec<BoundState>> = bounds.bound_states().collect();

            for b in bounds.unwrap() {
                let data = BoundStateData::new(energy, b);
                saver.send(data).unwrap()
            }
        });

        Ok(())
    }

    fn li2_wave() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());
        let mut li2_params = li2_params();
        li2_params.with_field(Quantity(600., Gauss));

        let e_min = Quantity(-12., GHz);
        let e_max = Quantity(0., GHz);
        let err = Quantity(1., MHz);
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);

        let saver = DataSaver::new("data/li2_wave_600G.jsonl", JsonFormat, FileAccess::Create)?;

        let bounds = BoundStatesFinder::default()
            .set_parameter_range(
                [e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()],
                err.to(AuEnergy).value(),
            )
            .set_problem(|e| {
                let mut params = li2_params.clone();
                params.system.energy = Quantity(e, AuEnergy);

                li2_problem.with_params(&params)
            })
            .set_r_range([Quantity(4., Bohr), Quantity(20., Bohr), Quantity(1.5e3, Bohr)])
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_strategy.into(), b));

        for b in bounds.bound_states() {
            let wave = bounds.bound_wave(&b.unwrap());

            saver.send(wave).unwrap()
        }

        Ok(())
    }
}

pub fn li2_params() -> AlkaliHomoDiatomParams<impl Interaction + Clone, impl Interaction + Clone> {
    let singlet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(1.112e7, -12)]);
    let triplet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(2.19348e8, -12)]);

    AlkaliHomoDiatomParams {
        atom: AtomStructureParams {
            a_hifi: Quantity(228.2 / 1.5, MHz).to(AuEnergy),
            ..Default::default()
        },
        system: SystemParams {
            mass: Quantity(6.015122 / 2., Dalton).to(AuMass),
            energy: Quantity(1e-7, Kelvin).to(AuEnergy),
            entrance_channel: 0,
        },
        triplet,
        singlet,
    }
}

pub fn li2_problem(recipe: AlkaliHomoDiatomRecipe) -> AlkaliHomoDiatom {
    AlkaliHomoDiatomBuilder::new(recipe).build()
}

pub fn li2_recipe() -> AlkaliHomoDiatomRecipe {
    AlkaliHomoDiatomRecipe {
        atom: AtomStructureRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1),
        },
        l_max: AngularMomentum(0),
        tot_projection: hi32!(0),
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
        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();

        for (field, result) in [(0., ASYMPTOTE_0), (500., ASYMPTOTE_500), (1000., ASYMPTOTE_1000)] {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let levels = w_matrix.asymptote().levels();

            assert!(levels.l.iter().all(|&x| x == 0));
            assert_approx_eq!(iter => levels.asymptote, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_NUMEROV: [f64; 3] = [-7.199240787436895, -283.4443841387517, -5286.252495345484];

    #[test]
    fn test_li2_numerov() {
        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_NUMEROV) {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);
            let boundary = vanishing_boundary(Quantity(4., Bohr), Direction::Outwards, &w_matrix);

            let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
            let solution = propagator.propagate_to(Quantity(1500., Bohr).value());
            let s_matrix = solution.get_s_matrix(&w_matrix);
            assert_approx_eq!(s_matrix.get_scattering_length().re, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_JOHNSON: [f64; 3] = [-7.19409243425113, -283.2910419145843, -5283.14763314261];

    #[test]
    fn test_li2_johnson() {
        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_JOHNSON) {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);
            let boundary = vanishing_boundary(Quantity(4., Bohr), Direction::Outwards, &w_matrix);

            let mut propagator = JohnsonLogDerivative::new(&w_matrix, step_strategy.into(), boundary);
            let solution = propagator.propagate_to(Quantity(1500., Bohr).value());
            let s_matrix = solution.get_s_matrix(&w_matrix);

            assert_approx_eq!(s_matrix.get_scattering_length().re, result, 1e-6);
        }
    }

    const SCATTERING_LENGTHS_MANOLOPOULOS: [f64; 3] = [-7.185909206685685, -283.1042109930902, -5280.378031416163];
    #[test]
    fn test_li2_manolopoulos() {
        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();

        for (field, &result) in [0., 500., 1000.].into_iter().zip(&SCATTERING_LENGTHS_MANOLOPOULOS) {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);
            let boundary = vanishing_boundary(Quantity(4., Bohr), Direction::Outwards, &w_matrix);

            let mut propagator = ManolopoulosLogDerivative::new(&w_matrix, step_strategy.into(), boundary);
            let solution = propagator.propagate_to(Quantity(1500., Bohr).value());
            let s_matrix = solution.get_s_matrix(&w_matrix);

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

        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();
        params.with_field(Quantity(600., Gauss));

        let e_min = Quantity(-12., GHz);
        let e_max = Quantity(0., GHz);
        let err = Quantity(1., MHz);
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);

        let bounds = BoundStatesFinder::default()
            .set_parameter_range(
                [e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()],
                err.to(AuEnergy).value(),
            )
            .set_problem(|e| {
                let mut params = params.clone();
                params.system.energy = Quantity(e, AuEnergy);

                li2_problem.with_params(&params)
            })
            .set_r_range([Quantity(4., Bohr), Quantity(20., Bohr), Quantity(1.5e3, Bohr)])
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_strategy.into(), b));

        let bound_states: Result<Vec<BoundState>> = bounds.bound_states().collect();
        let bound_states: Vec<BoundState> = bound_states
            .unwrap()
            .into_iter()
            .map(|mut b| {
                b.occupations = Some(bounds.bound_wave(&b).occupations());
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

        let li2_problem = li2_problem(li2_recipe());
        let mut params = li2_params();
        params.system.energy = Quantity(-60., MHz).to(AuEnergy);

        let mag_min = 0.;
        let mag_max = 1200.;
        let err = 1e-2;
        let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);

        let bounds = BoundStatesFinder::default()
            .set_parameter_range([mag_min, mag_max], err)
            .set_problem(|f| {
                let mut params = params.clone();
                params.with_field(Quantity(f, Gauss));

                li2_problem.with_params(&params)
            })
            .set_r_range([Quantity(4., Bohr), Quantity(20., Bohr), Quantity(1.5e3, Bohr)])
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_strategy.into(), b));

        let bound_states: Result<Vec<BoundState>> = bounds.bound_states().collect();
        let bound_states: Vec<BoundState> = bound_states
            .unwrap()
            .into_iter()
            .map(|mut b| {
                b.occupations = Some(bounds.bound_wave(&b).occupations());
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
