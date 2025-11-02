use cc_problems::{
    atom_rotor_basis::{AtomRotorTRAMRecipe, SinglePESAtomRotorTRAM},
    coupled_chan::{composite_int::CompositeInt, dispersion::Dispersion},
    prelude::*,
    rotor_structure::{Interaction2D, PESScaling},
    tram_basis::TRAMBasisRecipe,
};

pub struct ModelRotorAtomProblem;

problems_impl!(ModelRotorAtomProblem, "model atom rotor problem",
    "potential scaling bound_state dependence" => |_| Self::bound_states_dependence(),
);

impl ModelRotorAtomProblem {
    fn bound_states_dependence() -> Result<()> {
        let mut recipe = get_recipe();
        recipe.tram.n_max = 2;
        recipe.tram.l_max = 2;
        let scaling_1 = 2e-1;

        let scaling_min = 2.2;
        let scaling_max = 3.0;
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
        energies
            .par_iter()
            .progress_with_style(default_progress())
            .for_each_with(problem, |problem, &energy| {
                problem.system_params.energy = energy.to(AuEnergy);

                let bound_finder = bound.get_bound_finder(
                    (scaling_min, scaling_max),
                    scaling_err,
                    |s| {
                        let mut problem = problem.clone();
                        problem.pes.surface.scale(PESScaling::Scaling(s));

                        problem.w_matrix()
                    },
                    |w, s, b| JohnsonLogDerivative::new(w, s, b),
                );

                let bounds: Result<Vec<BoundState>> = bound_finder.bound_states().collect();

                for b in bounds.unwrap() {
                    let occupations = bound_finder.bound_wave(&b).occupations();

                    let mut data = BoundStateData::new(energy.value(), b);
                    data.occupations = Some(occupations);

                    saver.send(data)
                }
            });

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
        r_match: 50. * Bohr,
        r_max: 500. * Bohr,
        step_strat: LocalWavelengthStep::new(4e-3, 10., 400.).into(),
        node_monotony: NodeMonotony::Decreasing,
        node_range: None,
    }
}
