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
