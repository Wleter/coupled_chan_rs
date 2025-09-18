use cc_problems::{
    alkali_homo_diatom::{AlkaliHomoDiatom, AlkaliHomoDiatomBuilder, AlkaliHomoDiatomParams, AlkaliHomoDiatomRecipe}, anyhow::Result, atom_structure::{AtomStructureParams, AtomStructureRecipe}, coupled_chan::{
        composite_int::CompositeInt, constants::units::{
            atomic_units::{AuEnergy, AuMass, Bohr, Dalton, Gauss, Kelvin, MHz}, Quantity
        }, dispersion::Dispersion, propagator::{step_strategy::LocalWavelengthStep, Direction, Propagator}, ratio_numerov::{get_s_matrix, RatioNumerov}, vanishing_boundary, Interaction
    }, linspace, qol_utils::{
        problem_selector::{get_args, ProblemSelector}, problems_impl, saving::{DataSaver, FileAccess, JsonFormat}
    }, spin_algebra::{hi32, hu32}, system_structure::SystemParams, AngularMomentum, SMatrixData
};

use cc_problems::rayon::prelude::*;

fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Li2 collision",
    "Li2 Feshbach" => |_| Self::li2_feshbach()
);

impl Problems {
    fn li2_feshbach() -> Result<()> {
        let li2_problem = li2_problem(li2_recipe());
        let li2_params = li2_params();

        let mag_fields = linspace(0., 1000., 1001);
        let saver = DataSaver::new("data/li2_feshbach.jsonl", JsonFormat, FileAccess::Create)?;

        mag_fields.par_iter().for_each_with(li2_params, |params, &field| {
            let w_matrix = li2_problem.with_params(params.with_field(Quantity(field, Gauss)));
            let step_strategy = LocalWavelengthStep::new(1e-4, 10., 400.);
            let boundary = vanishing_boundary(Quantity(4., Bohr), Direction::Outwards, &w_matrix);

            let mut propagator = RatioNumerov::new(&w_matrix, step_strategy.into(), boundary);
            let solution = propagator.propagate_to(Quantity(500., Bohr).value());
            let s_matrix = get_s_matrix(solution, &w_matrix);

            saver.send(SMatrixData::new(field, s_matrix)).unwrap()
        });

        Ok(())
    }
}

pub fn li2_params() -> AlkaliHomoDiatomParams<impl Interaction + Clone, impl Interaction + Clone> {
    let triplet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(1.112e7, -12)]);
    let singlet = CompositeInt::new(vec![Dispersion::new(-1381., -6), Dispersion::new(2.19348e8, -12)]);

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
