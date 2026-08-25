use std::{
    borrow::Borrow,
    collections::{
        HashMap,
        HashSet,
    },
    hash::Hash,
    ops::{
        Deref,
        Index,
    },
    sync::Arc,
};

use cc_qol_utils::{
    Composite,
    params::CloneAny,
};
use coupled_chan::{
    DynInteraction,
    Interaction,
    coupling::{
        AngularBlocks,
        masked::Masked,
    },
    scaled::Scaled,
};
use hilbert_space::space::BasisElementsRef;
use smallvec::SmallVec;

use crate::{
    Operator,
    OrbitalBasisElements,
    parameters::{
        ParamId,
        ParameterRegistry,
        TypedParamId,
    },
};

use anyhow::{
    Result,
    anyhow,
};

pub type CouplingPotential = Masked<Scaled<DynInteraction>>;
pub type Coupling = Composite<CouplingPotential>;

#[derive(Clone)]
pub struct System {
    registry: ParameterRegistry,

    pub basis: OrbitalBasisElements,
    pub operator_specs: HashVec<String, DynOperatorSpec>,
    pub potential_specs: HashVec<String, PotentialSpec>,

    pub operators: HashVec<String, (f64, AngularBlocks)>,
    pub potentials: HashVec<String, CouplingPotential>,
}

impl System {
    pub fn new(hamiltonian_spec: HamiltonianSpec, registry: ParameterRegistry) -> Self {
        let basis = hamiltonian_spec.basis;
        let operator_specs = HashVec::from_hashmap(hamiltonian_spec.operators);
        let potential_specs = HashVec::from_hashmap(hamiltonian_spec.potentials);

        let operators =
            operator_specs.mapped(|x| (x.coupling(&registry), basis.get_angular_blocks(|_, b| x.matrix(b, &registry))));

        let potentials = potential_specs.mapped(|x| {
            Masked::new(
                Scaled {
                    scaling: x.operator_builder.coupling(&registry),
                    interaction: x.potential_curve.clone(),
                },
                x.operator_builder.matrix(basis.full_basis.as_ref(), &registry).0,
            )
        });

        Self {
            registry,
            basis,
            operator_specs,
            potential_specs,
            operators,
            potentials,
        }
    }

    pub fn param_registry(&self) -> &ParameterRegistry {
        &self.registry
    }

    pub fn modify_params(&mut self, modifications: ParamModifications<impl FnOnce(&mut ParameterRegistry) -> Vec<ParamId>>) {
        let ids = (modifications.modification)(&mut self.registry);

        self.update_operators(&ids);
        self.update_potentials(&ids);
    }

    pub fn angular_blocks(&self) -> AngularBlocks {
        let mut iterator = self.operators.vec.iter();
        let mut acc = iterator
            .next()
            .map(|(f, m)| m.scale(*f))
            .unwrap_or(self.basis.zero_angular_block());

        for (f, m) in iterator {
            if *f == 0.0 {
                continue;
            }

            acc += m.scale(*f)
        }

        acc
    }

    pub fn coupling(&self) -> Coupling {
        Composite::new(self.potentials.vec.clone())
    }

    fn update_operators(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut coupling_queue = HashSet::new();

        let specs = &self.operator_specs.vec;
        let ops = &mut self.operators.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if let Some(_) = ops.build_params().iter().find(|x| *x == m) {
                    rebuild_queue.insert(i);
                }
                if let Some(_) = ops.coupling_params().iter().find(|x| *x == m) {
                    coupling_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            ops[ops_id].1 = self
                .basis
                .get_angular_blocks(|_, b| self.operator_specs.vec[ops_id].matrix(b, &self.registry));
        }

        for ops_id in coupling_queue {
            ops[ops_id].0 = self.operator_specs.vec[ops_id].coupling(&self.registry);
        }
    }

    fn update_potentials(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut coupling_queue = HashSet::new();

        let specs = &self.potential_specs.vec;
        let potentials = &mut self.potentials.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if let Some(_) = ops.operator_builder.build_params().iter().find(|x| *x == m) {
                    rebuild_queue.insert(i);
                }
                if let Some(_) = ops.operator_builder.coupling_params().iter().find(|x| *x == m) {
                    coupling_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            potentials[ops_id].masking = specs[ops_id]
                .operator_builder
                .matrix(self.basis.full_basis.as_ref(), &self.registry)
                .0
        }

        for ops_id in coupling_queue {
            potentials[ops_id].interaction.scaling = self.operator_specs.vec[ops_id].coupling(&self.registry);
        }
    }
}

pub struct HamiltonianSpec {
    basis: OrbitalBasisElements,
    operators: HashMap<String, DynOperatorSpec>,
    potentials: HashMap<String, PotentialSpec>,
}

impl HamiltonianSpec {
    pub fn new(basis: OrbitalBasisElements) -> Self {
        Self {
            basis,
            operators: Default::default(),
            potentials: Default::default(),
        }
    }

    pub fn add_operators<'a>(&mut self, operators: impl IntoIterator<Item = (&'a str, DynOperatorSpec)>) {
        self.operators.extend(operators.into_iter().map(|x| (x.0.to_string(), x.1)));
    }

    pub fn add_potentials<'a>(&mut self, potentials: impl IntoIterator<Item = (&'a str, PotentialSpec)>) {
        self.potentials.extend(potentials.into_iter().map(|x| (x.0.to_string(), x.1)));
    }
}

pub struct ParamIds(SmallVec<[ParamId; 4]>);

impl ParamIds {
    pub fn new(vec: SmallVec<[ParamId; 4]>) -> Self {
        Self(vec)
    }
}

impl Deref for ParamIds {
    type Target = [ParamId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[macro_export]
macro_rules! param_ids {
    ($($x:expr),*$(,)?) => ({
        $crate::system::ParamIds::new($crate::smallvec::smallvec!($($x),*))
    });
}

pub trait OperatorSpec: Send + Sync {
    fn build_params(&self) -> ParamIds;
    fn matrix(&self, elements: BasisElementsRef, params: &ParameterRegistry) -> Operator;

    fn coupling_params(&self) -> ParamIds;
    fn coupling(&self, params: &ParameterRegistry) -> f64;
}

#[derive(Clone)]
pub struct DynOperatorSpec(pub Arc<dyn OperatorSpec>);

impl Deref for DynOperatorSpec {
    type Target = dyn OperatorSpec;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DynOperatorSpec {
    pub fn new(builder: impl OperatorSpec + 'static) -> Self {
        Self(Arc::new(builder))
    }
}

#[derive(Clone)]
pub struct PotentialSpec {
    pub operator_builder: DynOperatorSpec,
    pub potential_curve: DynInteraction,
}

impl PotentialSpec {
    pub fn new<O, I>(operator: O, interaction: I) -> Self
    where
        O: OperatorSpec + 'static,
        I: Interaction + 'static + Sync + Send,
    {
        Self {
            operator_builder: DynOperatorSpec::new(operator),
            potential_curve: DynInteraction::new(interaction),
        }
    }
}

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
    pub fn from_hashmap(hashmap: HashMap<Key, Val>) -> Self {
        let mut hash_vec = Self::default();
        hash_vec.extend(hashmap);

        hash_vec
    }

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

    pub fn mapped<Val2>(&self, mapping: impl Fn(&Val) -> Val2) -> HashVec<Key, Val2> {
        let new_vec = self.vec.iter().map(mapping).collect();

        HashVec {
            map: self.map.clone(),
            vec: new_vec,
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

pub type DynParamModifications = ParamModifications<Box<dyn FnOnce(&mut ParameterRegistry) -> Vec<ParamId>>>;

pub struct ParamModifications<F: FnOnce(&mut ParameterRegistry) -> Vec<ParamId>> {
    modification: F,
}

pub fn new_param_modifications<T: PartialEq + CloneAny>(
    id: TypedParamId<T>,
    value: T,
) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> Vec<ParamId> + 'static> {
    let modification = move |registry: &mut ParameterRegistry| {
        if registry.modify(id, value) {
            vec![id.vanish()]
        } else {
            vec![]
        }
    };

    ParamModifications { modification }
}

impl<'a, F: FnOnce(&mut ParameterRegistry) -> Vec<ParamId> + 'static> ParamModifications<F> {
    pub fn modify<T: PartialEq + CloneAny>(
        self,
        id: TypedParamId<T>,
        value: T,
    ) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> Vec<ParamId> + 'static> {
        let modification = move |registry: &mut ParameterRegistry| {
            if registry.modify(id, value) {
                let mut ids = (self.modification)(registry);

                ids.push(id.vanish());
                ids
            } else {
                (self.modification)(registry)
            }
        };

        ParamModifications { modification }
    }

    pub fn into_dyn(self) -> DynParamModifications {
        DynParamModifications {
            modification: Box::new(self.modification),
        }
    }
}
