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

    basis: OrbitalBasisElements,
    pub operator_specs: HashVec<String, DynOperatorSpec>,
    pub potential_specs: HashVec<String, DynPotentialSpec>,

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
            let masked = x.r_coupling(basis.full_basis.as_ref(), &registry);

            Masked::new(
                Scaled {
                    scaling: x.scaling(&registry),
                    interaction: masked.interaction,
                },
                masked.masking,
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

    pub fn modify_params(&mut self, modifications: ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds>) {
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
        if self.potentials.vec.is_empty() {
            return Composite::new(vec![]);
        }

        let zeros = Operator::zeros(self.potentials.vec[0].masking.nrows()).0;
        let filtered = self
            .potentials
            .vec
            .iter()
            .filter(|x| x.interaction.scaling != 0.0 || x.masking == zeros)
            .cloned()
            .collect();

        Composite::new(filtered)
    }

    fn update_operators(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut coupling_queue = HashSet::new();

        let specs = &self.operator_specs.vec;
        let ops = &mut self.operators.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if ops.build_params().iter().find(|x| *x == m).is_some() {
                    rebuild_queue.insert(i);
                }
                if ops.coupling_params().iter().find(|x| *x == m).is_some() {
                    coupling_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            ops[ops_id].1 = self.basis.get_angular_blocks(|_, b| specs[ops_id].matrix(b, &self.registry));
        }

        for ops_id in coupling_queue {
            ops[ops_id].0 = specs[ops_id].coupling(&self.registry);
        }
    }

    fn update_potentials(&mut self, modified: &[ParamId]) {
        let mut rebuild_queue = HashSet::new();
        let mut scaling_queue = HashSet::new();

        let specs = &self.potential_specs.vec;
        let potentials = &mut self.potentials.vec;

        for m in modified {
            for (i, ops) in specs.iter().enumerate() {
                if ops.build_params().iter().find(|x| *x == m).is_some() {
                    rebuild_queue.insert(i);
                }
                if ops.scaling_params().iter().find(|x| *x == m).is_some() {
                    scaling_queue.insert(i);
                }
            }
        }

        for ops_id in rebuild_queue {
            let rebuilt = specs[ops_id].r_coupling(self.basis.full_basis.as_ref(), &self.registry);

            potentials[ops_id].masking = rebuilt.masking;
            potentials[ops_id].interaction.interaction = rebuilt.interaction;
        }

        for ops_id in scaling_queue {
            potentials[ops_id].interaction.scaling = specs[ops_id].scaling(&self.registry);
        }
    }

    pub fn basis(&self) -> &OrbitalBasisElements {
        &self.basis
    }
}

pub struct HamiltonianSpec {
    basis: OrbitalBasisElements,
    operators: HashMap<String, DynOperatorSpec>,
    potentials: HashMap<String, DynPotentialSpec>,
}

impl HamiltonianSpec {
    pub fn new(basis: OrbitalBasisElements) -> Self {
        Self {
            basis,
            operators: Default::default(),
            potentials: Default::default(),
        }
    }

    pub fn add_operators<S: AsRef<str>>(&mut self, operators: impl IntoIterator<Item = (S, DynOperatorSpec)>) {
        self.operators
            .extend(operators.into_iter().map(|x| (x.0.as_ref().to_string(), x.1)));
    }

    pub fn add_potentials<S: AsRef<str>>(&mut self, potentials: impl IntoIterator<Item = (S, DynPotentialSpec)>) {
        self.potentials
            .extend(potentials.into_iter().map(|x| (x.0.as_ref().to_string(), x.1)));
    }
}

pub struct ParamIds(SmallVec<[ParamId; 4]>);

impl ParamIds {
    pub fn new(vec: SmallVec<[ParamId; 4]>) -> Self {
        Self(vec)
    }

    pub fn push(&mut self, value: ParamId) {
        self.0.push(value)
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
    [$($x:expr),*$(,)?] => ({
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
    pub fn new(spec: impl OperatorSpec + 'static) -> Self {
        Self(Arc::new(spec))
    }
}

pub trait PotentialSpec: Send + Sync {
    fn build_params(&self) -> ParamIds;
    fn r_coupling(&self, elements: BasisElementsRef, params: &ParameterRegistry) -> Masked<DynInteraction>;

    fn scaling_params(&self) -> ParamIds;
    fn scaling(&self, params: &ParameterRegistry) -> f64;
}

#[derive(Clone)]
pub struct DynPotentialSpec(pub Arc<dyn PotentialSpec>);

impl Deref for DynPotentialSpec {
    type Target = dyn PotentialSpec;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl DynPotentialSpec {
    pub fn new(spec: impl PotentialSpec + 'static) -> Self {
        Self(Arc::new(spec))
    }
}

#[derive(Clone)]
pub struct HashVec<Key, Val> {
    pub map: HashMap<Key, usize>,
    pub vec: Vec<Val>,
}

impl<Key: std::fmt::Debug, Val: std::fmt::Debug> std::fmt::Debug for HashVec<Key, Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.map.iter().map(|(a, i)| (a, &self.vec[*i]))).finish()
    }
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

pub type DynParamModifications = ParamModifications<Box<dyn FnOnce(&mut ParameterRegistry) -> ParamIds>>;

#[derive(Clone, Debug)]
pub struct ParamModifications<F: FnOnce(&mut ParameterRegistry) -> ParamIds> {
    modification: F,
}

pub fn new_param_modifications<T: PartialEq + CloneAny>(
    id: TypedParamId<T>,
    value: T,
) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> {
    let modification = move |registry: &mut ParameterRegistry| {
        if registry.modify(id, value) {
            param_ids![id.vanish()]
        } else {
            param_ids![]
        }
    };

    ParamModifications { modification }
}

impl<F: FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> ParamModifications<F> {
    pub fn new(modification: F) -> Self {
        Self { modification }
    }

    pub fn modify<T: PartialEq + CloneAny>(
        self,
        id: TypedParamId<T>,
        value: T,
    ) -> ParamModifications<impl FnOnce(&mut ParameterRegistry) -> ParamIds + 'static> {
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
