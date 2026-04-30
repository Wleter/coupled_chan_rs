use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::Hash,
    ops::Index,
    sync::Arc,
};

use anyhow::{
    Result,
    anyhow,
};
use cc_qol_utils::Composite;
use coupled_chan::{
    DynInteraction,
    Interaction,
    coupling::{
        AngularBlocks,
        Asymptote,
        SystemParams,
        masked::Masked,
    },
    scaled::Scaled,
};
use hilbert_space::{
    faer::Mat,
    space::BasisElementsRef,
};
use smallvec::SmallVec;

use crate::OrbitalBasisElements;

pub type Operator = hilbert_space::operator::Operator<Mat<f64>>;

#[derive(Clone)]
pub struct HamiltonianTerm {
    mel: Arc<dyn Fn(BasisElementsRef) -> Operator>,
}

impl HamiltonianTerm {
    pub fn new<F: Fn(BasisElementsRef) -> Operator + 'static>(mel: F) -> Self {
        Self { mel: Arc::new(mel) }
    }

    pub fn mel(&self, elements: BasisElementsRef) -> Operator {
        (self.mel)(elements)
    }
}

#[derive(Clone, Debug)]
pub struct Hamiltonian {
    constructor: HamiltonianConstructor,

    asymptote_constructor: AsymptoteConstructor,
    potential_constructor: PotentialConstructor,

    asymptote: Asymptote,
    potential: Composite<Masked<Scaled<DynInteraction>>>,
}

impl Hamiltonian {
    pub fn constructor(&self) -> &HamiltonianConstructor {
        &self.constructor
    }

    pub fn asymptote_constructor(&self) -> &AsymptoteConstructor {
        &self.asymptote_constructor
    }

    pub fn potential_constructor(&self) -> &PotentialConstructor {
        &self.potential_constructor
    }

    pub fn asymptote(&self) -> &Asymptote {
        &self.asymptote
    }

    pub fn potential(&self) -> &Composite<Masked<Scaled<DynInteraction>>> {
        &self.potential
    }

    // todo! maybe lazily change only when accessed or check if the value indeed changes
    pub fn modify<S: AsRef<str>, A: AsRef<[(S, f64)]>>(&mut self, params: A) -> Result<()> {
        let mut reconstruct_hamiltonian = vec![];
        let mut reconstruct_potential = vec![];
        let mut changed_hamiltonian = false;
        let mut changed_potential = false;

        for (key, val) in params.as_ref() {
            let val = *val;
            let index = self.constructor.params.map[key.as_ref()];
            if val == self.constructor.params.vec[index] {
                continue;
            }
            self.constructor.params.modify(key.as_ref(), val)?;

            for (name, term_index) in &self.constructor.hamiltonian_terms.map {
                let term = &self.constructor.hamiltonian_terms.vec[*term_index];

                if let Some(_) = term.params_replace.iter().find(|x| *x == &index) {
                    changed_hamiltonian = true;
                    reconstruct_hamiltonian.push(name);
                }

                if let Some((i, _)) = term.coupling_multiply.iter().enumerate().find(|(_, x)| *x == &index) {
                    changed_hamiltonian = true;
                    self.asymptote_constructor
                        .angular_blocks
                        .get_mut(name.as_ref())
                        .expect("Missing scaling in asymptote constructor")
                        .scaling[i] = val;
                }
            }

            for (name, index) in &self.constructor.interactions.map {
                let term = &self.constructor.interactions.vec[*index].term_builder;

                if let Some(_) = term.params_replace.iter().find(|x| *x == index) {
                    changed_potential = true;
                    reconstruct_potential.push(name)
                }

                if let Some((i, _)) = term.coupling_multiply.iter().enumerate().find(|(_, x)| *x == index) {
                    changed_potential = true;
                    self.potential_constructor
                        .potentials
                        .get_mut(name.as_ref())
                        .expect("Missing scaling in asymptote constructor")
                        .scaling[i] = val;
                }
            }
        }

        for name in reconstruct_hamiltonian {
            *self
                .asymptote_constructor
                .angular_blocks
                .get_mut(name.as_ref())
                .expect("Missing angular block in asymptote constructor") =
                self.constructor.single_angular_blocks_term(name);
        }

        for name in reconstruct_potential {
            *self
                .potential_constructor
                .potentials
                .get_mut(name.as_ref())
                .expect("Missing angular block in asymptote constructor") = self.constructor.single_potential_term(name);
        }

        if changed_hamiltonian {
            self.asymptote = self.asymptote_constructor.construct();
        }
        if changed_potential {
            self.potential = self.potential_constructor.construct();
        }

        Ok(())
    }
}

const VEC_BUFFER: usize = 4;

#[derive(Debug, Clone)]
pub struct HashVec<Key, Val> {
    pub map: HashMap<Key, usize>,
    pub vec: Vec<Val>,
}

impl<Key, Val> Default for HashVec<Key, Val> {
    fn default() -> Self {
        Self {
            map: Default::default(),
            vec: Default::default(),
        }
    }
}

impl<Key: Eq + Hash + Clone + std::fmt::Debug, Val> HashVec<Key, Val> {
    pub fn extend<KeyValPairs: IntoIterator<Item = (Key, Val)>>(&mut self, pairs: KeyValPairs)
    where
        KeyValPairs::IntoIter: ExactSizeIterator,
    {
        let pairs = pairs.into_iter();

        if self.map.is_empty() && self.vec.is_empty() {
            self.map = HashMap::with_capacity(pairs.len());
            self.vec = Vec::with_capacity(pairs.len());
        }

        for pair in pairs {
            self.insert(pair);
        }
    }

    pub fn extend_sep<Keys: IntoIterator<Item = Key>, Values: IntoIterator<Item = Val>>(
        &mut self,
        keys: Keys,
        values: Values,
    ) where
        Keys::IntoIter: ExactSizeIterator,
        Values::IntoIter: ExactSizeIterator,
    {
        let keys = keys.into_iter();
        let values = values.into_iter();

        assert_eq!(keys.len(), values.len());

        if self.map.is_empty() && self.vec.is_empty() {
            self.map = HashMap::with_capacity(keys.len());
            self.vec = Vec::with_capacity(values.len());
        }

        let index_start = values.len();
        self.map
            .extend(keys.into_iter().enumerate().map(|(i, x)| (x, index_start + i)));
        self.vec.extend(values);
    }

    pub fn insert(&mut self, pair: (Key, Val)) {
        if let Some(val) = self.map.get(&pair.0) {
            self.vec[*val] = pair.1;
        } else {
            let index = self.vec.len();
            self.vec.push(pair.1);
            self.map.insert(pair.0, index);
        }
    }
}

impl<Key: Eq + Hash, Val> HashVec<Key, Val> {
    pub fn modify<S>(&mut self, key: &S, val: Val) -> Result<()>
    where
        S: Hash + Eq + std::fmt::Debug + ?Sized,
        Key: Borrow<S>,
    {
        if let Some(index) = self.map.get(key) {
            self.vec[*index] = val;
            Ok(())
        } else {
            Err(anyhow!("Could not modify missing element {:?}", key.borrow()))
        }
    }
}

impl<S: ?Sized + Hash + Eq, Key: Eq + Hash + Borrow<S>, Val> Index<&S> for HashVec<Key, Val> {
    type Output = Val;

    fn index(&self, index: &S) -> &Self::Output {
        &self.vec[self.map[index]]
    }
}

#[derive(Clone, Debug)]
pub struct HamiltonianConstructor {
    basis: OrbitalBasisElements,
    system_params: SystemParams,

    params: HashVec<Box<str>, f64>,
    hamiltonian_terms: HashVec<Box<str>, TermBuilder>,
    interactions: HashVec<Box<str>, PotentialBuilder>,
}

impl HamiltonianConstructor {
    pub fn new(basis: OrbitalBasisElements, system_params: SystemParams) -> Self {
        Self {
            basis,
            system_params,
            params: Default::default(),
            hamiltonian_terms: Default::default(),
            interactions: Default::default(),
        }
    }

    pub fn add_param(&mut self, pair: (impl AsRef<str>, f64)) {
        self.params.insert((pair.0.as_ref().into(), pair.1));
    }

    pub fn extend_params<S: AsRef<str>, Params: IntoIterator<Item = (S, f64)>>(&mut self, params: Params)
    where
        Params::IntoIter: ExactSizeIterator,
    {
        let params = params.into_iter().map(|(s, v)| (s.as_ref().into(), v));
        self.params.extend(params);
    }

    pub fn param_index(&self, key: impl AsRef<str>) -> usize {
        self.params.map[key.as_ref()]
    }

    pub fn add_term(&mut self, recipe: TermRecipe) {
        self.hamiltonian_terms.insert(recipe.into_builder(&self.params));
    }

    pub fn extend_terms<Recipes: IntoIterator<Item = TermRecipe>>(&mut self, recipes: Recipes)
    where
        Recipes::IntoIter: ExactSizeIterator,
    {
        let builders = recipes.into_iter().map(|x| x.into_builder(&self.params));
        self.hamiltonian_terms.extend(builders);
    }

    pub fn add_interaction(&mut self, recipe: PotentialRecipe) {
        self.interactions.insert(recipe.into_builder(&self.params));
    }

    pub fn extend_interactions<Recipes: IntoIterator<Item = PotentialRecipe>>(&mut self, recipes: Recipes)
    where
        Recipes::IntoIter: ExactSizeIterator,
    {
        let builders = recipes.into_iter().map(|x| x.into_builder(&self.params));
        self.interactions.extend(builders);
    }

    fn single_angular_blocks_term(&self, name: impl AsRef<str>) -> AngularBlocksTerm {
        let name = name.as_ref();
        let index = self.hamiltonian_terms.map[name];

        let term_builder = &self.hamiltonian_terms.vec[index];
        let scaling = term_builder.coupling_multiply.iter().map(|&x| self.params.vec[x]).collect();
        let params: SmallVec<[f64; VEC_BUFFER]> = term_builder.params_replace.iter().map(|&x| self.params.vec[x]).collect();

        let term = (term_builder.operator)(&params);
        let angular_blocks = self.basis.get_angular_blocks(|_, e| term.mel(e));

        AngularBlocksTerm { scaling, angular_blocks }
    }

    fn single_potential_term(&self, name: impl AsRef<str>) -> PotentialTerm {
        let name = name.as_ref();
        let index = self.interactions.map[name];

        let interaction = &self.interactions.vec[index];
        let scaling = interaction
            .term_builder
            .coupling_multiply
            .iter()
            .map(|&x| self.params.vec[x])
            .collect();

        let params: SmallVec<[f64; VEC_BUFFER]> = interaction
            .term_builder
            .params_replace
            .iter()
            .map(|&x| self.params.vec[x])
            .collect();

        let term = (interaction.term_builder.operator)(&params);
        let operator = term.mel(self.basis.full_basis.as_ref());

        PotentialTerm {
            potential: Masked::new(Scaled::new(interaction.interaction.clone()), operator.0),
            scaling,
        }
    }

    pub fn asymptote_constructor(&self) -> AsymptoteConstructor {
        let angular_blocks = self
            .hamiltonian_terms
            .map
            .iter()
            .map(|(key, &index)| {
                let term_builder = &self.hamiltonian_terms.vec[index];

                let scaling = term_builder.coupling_multiply.iter().map(|&x| self.params.vec[x]).collect();

                let params: SmallVec<[f64; VEC_BUFFER]> =
                    term_builder.params_replace.iter().map(|&x| self.params.vec[x]).collect();

                let term = (term_builder.operator)(&params);
                let angular_blocks = self.basis.get_angular_blocks(|_, e| term.mel(e));

                (key.clone(), AngularBlocksTerm { scaling, angular_blocks })
            })
            .collect();

        AsymptoteConstructor {
            angular_blocks,
            system_params: self.system_params.clone(),
        }
    }

    pub fn potential_constructor(&self) -> PotentialConstructor {
        let potentials = self
            .interactions
            .map
            .iter()
            .map(|(key, &index)| {
                let interaction = &self.interactions.vec[index];

                let scaling = interaction
                    .term_builder
                    .coupling_multiply
                    .iter()
                    .map(|&x| self.params.vec[x])
                    .collect();

                let params: SmallVec<[f64; VEC_BUFFER]> = interaction
                    .term_builder
                    .params_replace
                    .iter()
                    .map(|&x| self.params.vec[x])
                    .collect();

                let term = (interaction.term_builder.operator)(&params);
                let operator = term.mel(self.basis.full_basis.as_ref());

                (
                    key.clone(),
                    PotentialTerm {
                        potential: Masked::new(Scaled::new(interaction.interaction.clone()), operator.0),
                        scaling,
                    },
                )
            })
            .collect();

        PotentialConstructor { potentials }
    }

    pub fn construct(self) -> Hamiltonian {
        let asymptote_constructor = self.asymptote_constructor();
        let potential_constructor = self.potential_constructor();
        let asymptote = asymptote_constructor.construct();
        let potential = potential_constructor.construct();

        Hamiltonian {
            constructor: self,
            asymptote_constructor,
            potential_constructor,
            asymptote,
            potential,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsymptoteConstructor {
    pub angular_blocks: HashMap<Box<str>, AngularBlocksTerm>,
    pub system_params: SystemParams,
}

impl AsymptoteConstructor {
    pub fn construct(&self) -> Asymptote {
        let angular_blocks = self
            .angular_blocks
            .iter()
            .map(|(_, x)| x.angular_blocks.scale(prod(&x.scaling)))
            .sum();

        Asymptote::new_angular_blocks(angular_blocks, self.system_params)
    }
}

#[derive(Debug, Clone)]
pub struct PotentialTerm {
    pub scaling: SmallVec<[f64; 3]>,
    pub potential: Masked<Scaled<DynInteraction>>,
}

#[derive(Debug, Clone)]
pub struct PotentialConstructor {
    pub potentials: HashMap<Box<str>, PotentialTerm>,
}

impl PotentialConstructor {
    pub fn construct(&self) -> Composite<Masked<Scaled<DynInteraction>>> {
        Composite::new(
            self.potentials
                .iter()
                .map(|(_, x)| {
                    let mut potential = x.potential.clone();
                    potential.interaction_mut().scale(prod(&x.scaling));

                    potential
                })
                .collect(),
        )
    }
}

fn prod(a: impl AsRef<[f64]>) -> f64 {
    a.as_ref().iter().product()
}

#[derive(Clone)]
struct TermBuilder {
    coupling_multiply: SmallVec<[usize; VEC_BUFFER]>,
    params_replace: SmallVec<[usize; VEC_BUFFER]>,
    operator: Arc<dyn Fn(&[f64]) -> HamiltonianTerm>,
}

impl std::fmt::Debug for TermBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermBuilder")
            .field("coupling_multiply", &self.coupling_multiply)
            .field("params_replace", &self.params_replace)
            .finish()
    }
}

#[derive(Clone, Debug)]
struct PotentialBuilder {
    term_builder: TermBuilder,
    interaction: DynInteraction,
}

#[derive(Debug, Clone)]
pub struct AngularBlocksTerm {
    scaling: SmallVec<[f64; VEC_BUFFER]>,
    angular_blocks: AngularBlocks,
}

#[derive(Clone)]
pub struct TermRecipe {
    pub name: Box<str>,
    pub coupling_multiply: Vec<Box<str>>,
    pub params_replace: Vec<Box<str>>,
    pub operator: Arc<dyn Fn(&[f64]) -> HamiltonianTerm>,
}

impl TermRecipe {
    pub fn new<'a, F, T1, T2>(name: impl AsRef<str>, coupling_multiply: T1, params_replace: T2, operator: F) -> Self
    where
        F: Fn(&[f64]) -> HamiltonianTerm + 'static,
        T1: IntoIterator<Item = &'a str>,
        T2: IntoIterator<Item = &'a str>,
    {
        Self {
            name: name.as_ref().into(),
            coupling_multiply: coupling_multiply.into_iter().map(|x| x.into()).collect(),
            params_replace: params_replace.into_iter().map(|x| x.into()).collect(),
            operator: Arc::new(operator),
        }
    }

    fn into_builder(self, params: &HashVec<Box<str>, f64>) -> (Box<str>, TermBuilder) {
        let builder = TermBuilder {
            coupling_multiply: self.coupling_multiply.into_iter().map(|x| params.map[&x]).collect(),
            params_replace: self.params_replace.into_iter().map(|x| params.map[&x]).collect(),
            operator: self.operator,
        };

        (self.name, builder)
    }
}

#[derive(Clone)]
pub struct PotentialRecipe {
    pub term: TermRecipe,
    pub interaction: DynInteraction,
}

impl PotentialRecipe {
    pub fn new<I: Interaction + Send + Sync + 'static>(term: TermRecipe, interaction: I) -> Self {
        Self {
            interaction: DynInteraction::new(interaction),
            term,
        }
    }

    fn into_builder(self, params: &HashVec<Box<str>, f64>) -> (Box<str>, PotentialBuilder) {
        let (name, term_builder) = self.term.into_builder(params);

        (
            name,
            PotentialBuilder {
                term_builder,
                interaction: self.interaction,
            },
        )
    }
}
