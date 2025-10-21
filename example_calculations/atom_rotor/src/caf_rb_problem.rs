use cc_problems::{
    atom_rotor_basis::{AlkaliAtomRotorTRAM, AtomRotorTRAMRecipe},
    atom_structure::AtomBasisRecipe,
    coupled_chan::{composite_int::CompositeInt, dispersion::Dispersion},
    prelude::*,
    rotor_structure::Interaction2D,
    system_structure::SystemParams,
    tram_basis::TRAMBasisRecipe,
};

pub struct CaFRbProblem;

problems_impl!(CaFRbProblem, "CaF + Rb problems",
    "CaF + Rb Feshbach - isotropic only" => |_| Self::caf_rb_iso_feshbach(),
    "CaF + Rb Feshbach" => |_| Self::caf_rb_feshbach(),
);

impl CaFRbProblem {
    fn caf_rb_iso_feshbach() -> Result<()> {
        let mut recipe = caf_rb_recipe();
        recipe.tram.l_max = 0;
        recipe.tram.n_max = 0;

        let caf_rb_problem = caf_rb_problem(0, 0, recipe);

        let mag_fields = linspace(0., 1000., 4001);
        let saver = DataSaver::new("data/caf_rb_iso_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields
            .par_iter()
            .progress_with_style(default_progress())
            .for_each_with(caf_rb_problem, |problem, &field| {
                problem.set_b_field(field * Gauss);
                let w_matrix = problem.w_matrix();
                let step_strategy = LocalWavelengthStep::new(1e-4, f64::INFINITY, 500.);
                let boundary = vanishing_boundary(7.2 * Bohr, Direction::Outwards, &w_matrix);

                let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
                let solution = propagator.propagate_to((1.5e3 * Bohr).value());
                let s_matrix = solution.get_s_matrix(&w_matrix);

                saver.send(SMatrixData::new(field, s_matrix))
            });

        Ok(())
    }

    fn caf_rb_feshbach() -> Result<()> {
        let recipe = caf_rb_recipe();
        let caf_rb_problem = caf_rb_problem(0, 0, recipe);

        let mag_fields = linspace(0., 1000., 4001);
        let saver = DataSaver::new("data/caf_rb_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        let mut problem = caf_rb_problem.clone();
        problem.set_b_field(100. * Gauss);
        let w_matrix = problem.w_matrix();
        println!("{:?}", w_matrix.asymptote());

        mag_fields
            .par_iter()
            .progress_with_style(default_progress())
            .for_each_with(caf_rb_problem, |problem, &field| {
                problem.set_b_field(field * Gauss);
                let w_matrix = problem.w_matrix();
                let step_strategy = LocalWavelengthStep::new(1e-4, f64::INFINITY, 500.);
                let boundary = vanishing_boundary(7.2 * Bohr, Direction::Outwards, &w_matrix);

                let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
                let solution = propagator.propagate_to((1.5e3 * Bohr).value());
                let s_matrix = solution.get_s_matrix(&w_matrix);

                saver.send(SMatrixData::new(field, s_matrix))
            });

        Ok(())
    }
}

const C6_0: f64 = -3084.;
const C6_2: f64 = -100.;
const C12_0_TRIPLET: f64 = 2e9;
const C12_0_SINGLET: f64 = 5e8;

fn caf_rb_problem(
    singlet_scaling_no: usize,
    triplet_scaling_no: usize,
    recipe: AtomRotorTRAMRecipe,
) -> AlkaliAtomRotorTRAM<impl Interaction + Clone, impl Interaction + Clone> {
    let factors_singlet = [1.0196, 0.9815, 1.0037];
    let factors_triplet = [1.0286, 0.9717, 1.00268];
    let c12_0_singlet = factors_singlet[singlet_scaling_no] * C12_0_SINGLET;
    let c12_0_triplet = factors_triplet[triplet_scaling_no] * C12_0_TRIPLET;

    let singlet_iso = CompositeInt::new(vec![Dispersion::new(C6_0, -6), Dispersion::new(c12_0_singlet, -12)]);

    let triplet_iso = CompositeInt::new(vec![Dispersion::new(C6_0, -6), Dispersion::new(c12_0_triplet, -12)]);

    let singlet = Interaction2D(vec![
        (0, singlet_iso),
        (2, CompositeInt::new(vec![Dispersion::new(C6_2, -6)])),
    ]);

    let triplet = Interaction2D(vec![
        (0, triplet_iso),
        (2, CompositeInt::new(vec![Dispersion::new(C6_2, -6)])),
    ]);

    let mut problem = AlkaliAtomRotorTRAM::new(triplet, singlet, recipe);
    problem.atom_a.hyperfine.a_hifi = (6.83 / 2. * GHz).to(AuEnergy);
    problem.atom_b.hyperfine.a_hifi = (120. * MHz).to(AuEnergy);
    problem.rotational.rot_const = (10.3 * GHz).to(AuEnergy);
    problem.system_params.energy = (1e-7 * Kelvin).to(AuEnergy);
    problem.system_params.mass = SystemParams::red_masses(&[
        Quantity(39.962590850 + 18.998403162, Dalton).to(AuMass),
        (86.90918053 * Dalton).to(AuMass),
    ]);

    problem
}

fn caf_rb_recipe() -> AtomRotorTRAMRecipe {
    AtomRotorTRAMRecipe {
        atom_a: AtomBasisRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
        },
        atom_b: AtomBasisRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1 / 2),
        },
        atom_c: AtomBasisRecipe::default(),
        tram: TRAMBasisRecipe {
            l_max: 5,
            n_max: 5,
            ..Default::default()
        },
        tot_projection: hi32!(1),
        anisotropy_lambda_max: 2,
    }
}
