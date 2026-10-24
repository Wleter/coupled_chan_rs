use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Index;

#[derive(Clone)]
pub struct HashVec<Key, Val> {
    pub map: HashMap<Key, usize>,
    pub vec: Vec<Val>,
}

impl<Key: std::fmt::Debug, Val: std::fmt::Debug> std::fmt::Debug for HashVec<Key, Val> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.map.iter().map(|(a, i)| (a, &self.vec[*i])))
            .finish()
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
    pub fn modify<S>(&mut self, key: &S, val: Val) -> anyhow::Result<()>
    where
        S: Hash + Eq + std::fmt::Debug + ?Sized,
        Key: Borrow<S>,
    {
        if let Some(index) = self.map.get(key) {
            self.vec[*index] = val;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Could not modify {:?}, key does not exist", key.borrow()))
        }
    }
}

impl<S: ?Sized + Hash + Eq, Key: Eq + Hash + Borrow<S>, Val> Index<&S> for HashVec<Key, Val> {
    type Output = Val;

    fn index(&self, index: &S) -> &Self::Output {
        &self.vec[self.map[index]]
    }
}
