use cc_problems::{
    atom_rotor_basis::{AtomRotorTRAMRecipe, SinglePESAtomRotorTRAM},
    coupled_chan::{composite_int::CompositeInt, dispersion::Dispersion, log_derivative::diabatic::Johnson},
    prelude::*,
    rotor_structure::{Interaction2D, PESScaling},
    tram_basis::TRAMBasisRecipe,
};

pub struct ModelRotorAtomProblem;

problems_impl!(ModelRotorAtomProblem, "model atom rotor problem",
    "potential scaling bound_state dependence" => |_| Self::bound_states_dependence(),
    "bound wave" => |_| Self::bound_wave(),
);

impl ModelRotorAtomProblem {
    fn bound_states_dependence() -> Result<()> {
        let mut recipe = get_recipe();
        recipe.tram.n_max = 10;
        recipe.tram.l_max = 10;
        let scaling_1 = 1e-3;

        let scaling_min = 0.8;
        let scaling_max = 1.2;
        let scaling_err = 1e-8;

        let save_file = format!(
            "data/model_atom_rotor_bounds_n_max_{}_aniso_{:.0e}.jsonl",
            recipe.tram.n_max, scaling_1
        );

        let mut problem = get_problem(recipe);
        problem.pes.surface.scale(PESScaling::LegendreScaling(1, scaling_1));
        let bound = get_bound_problem();

        let energies: Vec<Quantity<GHz>> = linspace(-2., 0., 201).iter().map(|x| x.powi(3) * GHz).collect();

        let saver = DataSaver::new(&save_file, JsonFormat, FileAccess::Create)?;

        DependenceProblem::new(problem).dependence(energies, |problem, &energy| {
            problem.system_params.energy = energy.to(AuEnergy);

            let bound_finder = bound.get_bound_finder::<_, Johnson>((scaling_min, scaling_max), scaling_err, |s| {
                let mut problem = problem.clone();
                problem.pes.surface.scale(PESScaling::Scaling(s));

                problem.w_matrix()
            });

            let bounds: Result<Vec<BoundState>> = bound_finder.bound_states().collect();

            for b in bounds? {
                let occupations = bound_finder.bound_wave(&b).occupations();

                let mut data = BoundStateData::new(energy.value(), b);
                data.occupations = Some(occupations);

                saver.send(data)
            }

            Ok(())
        })?;

        Ok(())
    }

    fn bound_wave() -> Result<()> {
        let mut recipe = get_recipe();
        recipe.tram.n_max = 10;
        recipe.tram.l_max = 10;

        let scaling_1 = 1e-3;
        let scaling = 0.80353;
        let e_min = -125. * MHz;
        let e_max = -15.6 * MHz;
        let e_err = 1e-3 * MHz;

        let save_file = format!("data/model_atom_rotor_wave_n_max_{}.jsonl", recipe.tram.n_max);

        let mut problem = get_problem(recipe);
        problem.pes.surface.scale(PESScaling::Composite(vec![
            PESScaling::LegendreScaling(1, scaling_1),
            PESScaling::Scaling(scaling),
        ]));
        let bound = get_bound_problem();

        let saver = DataSaver::new(&save_file, JsonFormat, FileAccess::Create)?;
        let bound_finder = bound.get_bound_finder::<_, Johnson>(
            (e_min.to(AuEnergy).value(), e_max.to(AuEnergy).value()),
            e_err.to(AuEnergy).value(),
            |e| {
                let mut problem = problem.clone();
                problem.system_params.energy = e * AuEnergy;

                problem.w_matrix()
            },
        );

        for b in bound_finder.bound_states() {
            let mut b = b?;
            let wave = bound_finder.bound_wave(&b);

            let occupation = wave.occupations();
            b.occupations = Some(occupation);

            println!("{b:?}");

            saver.send(wave);
        }

        Ok(())
    }
}

const C6_0: f64 = -3495.30040855597;
const C8_0: f64 = -516911.950541056;
const C7_1: f64 = -17274.8363457991;
const C9_1: f64 = -768422.32042577;
const C12_0: f64 = 2e9;

fn get_problem(recipe: AtomRotorTRAMRecipe) -> SinglePESAtomRotorTRAM<impl Interaction + Clone + std::fmt::Debug> {
    let pes_iso = CompositeInt::new(vec![
        Dispersion::new(C6_0, -6),
        Dispersion::new(C8_0, -8),
        Dispersion::new(C12_0, -12),
    ]);

    let pes_aniso_1 = CompositeInt::new(vec![Dispersion::new(C7_1, -7), Dispersion::new(C9_1, -9)]);

    let pes = Interaction2D(vec![(0, pes_iso), (1, pes_aniso_1)]);

    let mut problem = SinglePESAtomRotorTRAM::new(pes, recipe);
    problem.rotational.rot_const = (0.24975935 * CmInv).to(AuEnergy);
    problem.system_params.energy = (1e-7 * Kelvin).to(AuEnergy);
    problem.system_params.mass = Quantity(47.9376046914861, Dalton).to(AuMass);

    problem
}

fn get_recipe() -> AtomRotorTRAMRecipe {
    AtomRotorTRAMRecipe {
        tram: TRAMBasisRecipe {
            l_max: 10,
            n_max: 10,
            ..Default::default()
        },
        tot_projection: hi32!(0),
        ..Default::default()
    }
}

fn get_bound_problem() -> BoundProblem {
    BoundProblem {
        r_min: 6. * Bohr,
        r_match: 25. * Bohr,
        r_max: 500. * Bohr,
        step_strat: LocalWavelengthStep::new(4e-3, 10., 400.).into(),
        node_monotony: NodeMonotony::Decreasing,
        node_range: None,
    }
}
