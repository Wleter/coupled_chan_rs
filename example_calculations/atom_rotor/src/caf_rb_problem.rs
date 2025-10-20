use cc_problems::{
    alkali_atom_rotor::{
        AlkaliAtomRotorTRAM, AlkaliAtomRotorTRAMBuilder, AlkaliAtomRotorTRAMParams, AlkaliAtomRotorTRAMRecipe,
    },
    atom_structure::{AtomStructureParams, AtomStructureRecipe},
    coupled_chan::{composite_int::CompositeInt, dispersion::Dispersion},
    prelude::*,
    rotor_structure::{Interaction2D, RotorParams},
    system_structure::SystemParams,
    tram_basis::{TRAMBasisParams, TRAMBasisRecipe},
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

        let caf_rb_problem = caf_rb_problem(recipe);
        let caf_rb_params = caf_rb_params(0, 0);

        let mag_fields = linspace(0., 1000., 4001);
        let saver = DataSaver::new("data/caf_rb_iso_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields
            .par_iter()
            .progress_with_style(default_progress())
            .for_each_with(caf_rb_params, |params, &field| {
                let w_matrix = caf_rb_problem.with_params(params.with_field(Quantity(field, Gauss)));
                let step_strategy = LocalWavelengthStep::new(1e-4, f64::INFINITY, 500.);
                let boundary = vanishing_boundary(Quantity(7.2, Bohr), Direction::Outwards, &w_matrix);

                let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
                let solution = propagator.propagate_to(Quantity(1.5e3, Bohr).value());
                let s_matrix = solution.get_s_matrix(&w_matrix);

                saver.send(SMatrixData::new(field, s_matrix))
            });

        Ok(())
    }

    fn caf_rb_feshbach() -> Result<()> {
        let recipe = caf_rb_recipe();

        let caf_rb_problem = caf_rb_problem(recipe);
        let caf_rb_params = caf_rb_params(0, 0);

        let mag_fields = linspace(0., 1000., 4001);
        let saver = DataSaver::new("data/caf_rb_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields
            .par_iter()
            .progress_with_style(default_progress())
            .for_each_with(caf_rb_params, |params, &field| {
                let w_matrix = caf_rb_problem.with_params(params.with_field(Quantity(field, Gauss)));
                let step_strategy = LocalWavelengthStep::new(1e-4, f64::INFINITY, 500.);
                let boundary = vanishing_boundary(Quantity(7.2, Bohr), Direction::Outwards, &w_matrix);

                let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
                let solution = propagator.propagate_to(Quantity(1.5e3, Bohr).value());
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

fn caf_rb_params(
    singlet_scaling_no: usize,
    triplet_scaling_no: usize,
) -> AlkaliAtomRotorTRAMParams<impl Interaction + Clone, impl Interaction + Clone> {
    let factors_singlet = [1.0196, 0.9815, 1.0037];
    let factors_triplet = [1.0286, 0.9717, 1.00268];

    let singlet = Interaction2D(vec![
        (
            0,
            CompositeInt::new(vec![
                Dispersion::new(C6_0, -6),
                Dispersion::new(factors_singlet[singlet_scaling_no] * C12_0_SINGLET, -12),
            ]),
        ),
        (2, CompositeInt::new(vec![Dispersion::new(C6_2, -6)])),
    ]);

    let triplet = Interaction2D(vec![
        (
            0,
            CompositeInt::new(vec![
                Dispersion::new(C6_0, -6),
                Dispersion::new(factors_triplet[triplet_scaling_no] * C12_0_TRIPLET, -12),
            ]),
        ),
        (2, CompositeInt::new(vec![Dispersion::new(C6_2, -6)])),
    ]);

    AlkaliAtomRotorTRAMParams {
        atom_a: AtomStructureParams {
            a_hifi: Quantity(6.83 / 2., GHz).to(AuEnergy),
            ..Default::default()
        },
        atom_b: AtomStructureParams {
            a_hifi: Quantity(120., MHz).to(AuEnergy),
            ..Default::default()
        },
        atom_c: AtomStructureParams::default(),
        tram: TRAMBasisParams {
            rotor: RotorParams {
                rot_const: Quantity(10.3, GHz).to(AuEnergy),
                ..Default::default()
            },
            system: SystemParams {
                mass: SystemParams::red_masses(&[
                    Quantity(39.962590850 + 18.998403162, Dalton).to(AuMass),
                    Quantity(86.90918053, Dalton).to(AuMass),
                ]),
                energy: Quantity(1e-7, Kelvin).to(AuEnergy),
                entrance_channel: 0,
            },
        },
        triplet,
        singlet,
    }
}

fn caf_rb_problem(recipe: AlkaliAtomRotorTRAMRecipe) -> AlkaliAtomRotorTRAM {
    AlkaliAtomRotorTRAMBuilder::new(recipe).build()
}

fn caf_rb_recipe() -> AlkaliAtomRotorTRAMRecipe {
    AlkaliAtomRotorTRAMRecipe {
        atom_a: AtomStructureRecipe {
            s: hu32!(1 / 2),
            i: hu32!(3 / 2),
        },
        atom_b: AtomStructureRecipe {
            s: hu32!(1 / 2),
            i: hu32!(1 / 2),
        },
        atom_c: AtomStructureRecipe::default(),
        tram: TRAMBasisRecipe {
            l_max: 5,
            n_max: 5,
            ..Default::default()
        },
        tot_projection: hi32!(1),
        anisotropy_lambda_max: 2,
    }
}
