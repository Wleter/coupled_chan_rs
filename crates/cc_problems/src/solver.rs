use coupled_chan::cc_propagator::step_strategy::Step;
use serde::{
    Deserialize,
    Serialize,
};

use crate::hamiltonian::Hamiltonian;

pub struct SystemBuilder<Rec, H>
where
    Rec: Serialize + Deserialize<'static>,
    H: Fn(&Rec) -> Hamiltonian,
{
    recipe: Option<Rec>,
    hamiltonian: Option<H>,
}

impl<Recipe, H> Default for SystemBuilder<Recipe, H>
where
    Recipe: Serialize + Deserialize<'static>,
    H: Fn(&Recipe) -> Hamiltonian,
{
    fn default() -> Self {
        Self {
            recipe: Default::default(),
            hamiltonian: Default::default(),
        }
    }
}

impl<Recipe, H> SystemBuilder<Recipe, H>
where
    Recipe: Serialize + Deserialize<'static>,
    H: Fn(&Recipe) -> Hamiltonian,
{
    pub fn recipe(mut self, recipe: Recipe) -> Self {
        self.recipe = Some(recipe);
        self
    }

    pub fn hamiltonian(mut self, hamiltonian: H) -> Self {
        self.hamiltonian = Some(hamiltonian);
        self
    }
}

pub trait Calculation<Res: Serialize> {
    fn calculate(&self, hamiltonian: &Hamiltonian) -> Res;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CoupledChanSolver {
    RatioNumerov,
    JohnsonLogDeriv,
    ManolopoulosLogDeriv,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScatteringCalc<S: Step> {
    pub r_start: f64,
    pub r_stop: f64,
    pub step: S,
    pub solver: CoupledChanSolver,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundStateCalc<S: Step> {
    pub r_start: f64,
    pub r_stop: f64,
    pub r_match: f64,
    pub step: S,
    pub solver: CoupledChanSolver,
}

#[cfg(test)]
mod tests {
    use hilbert_space::space::SpaceBasis;
    use serde::{
        Deserialize,
        Serialize,
    };
    use spin_algebra::{
        hi32,
        hu32,
    };

    use crate::{
        OrbitalRecipe,
        atom_basis::{
            AtomRecipe,
            RecipeWithProj,
        },
        diatom_basis::{
            DiatomRecipe,
            UncoupledDiatomBasis,
        },
        solver::SystemBuilder,
    };

    #[derive(Clone, Copy, Debug, Serialize, Deserialize)]
    struct Recipe(RecipeWithProj<DiatomRecipe>);

    fn recipe() -> Recipe {
        Recipe(RecipeWithProj {
            recipe: DiatomRecipe {
                atom_a: AtomRecipe {
                    s: hu32!(1 / 2),
                    i: hu32!(1 / 2),
                },
                atom_b: AtomRecipe {
                    s: hu32!(1 / 2),
                    i: hu32!(3 / 2),
                },
                l: OrbitalRecipe::Single(0),
            },
            projection: hi32!(0),
        })
    }

    #[test]
    fn test_system_builder() {
        let problem = SystemBuilder::default().recipe(recipe()).hamiltonian(|recipe| {
            let mut basis = SpaceBasis::default();
            let diatom = UncoupledDiatomBasis::new(recipe.0.recipe, &mut basis);

            todo!()
        });
    }
}
