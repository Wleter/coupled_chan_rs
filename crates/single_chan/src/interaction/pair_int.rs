use crate::interaction::Interaction;

#[derive(Debug, Clone)]
pub struct PairInt<P: Interaction, V: Interaction> {
    first: P,
    second: V
}

impl<P: Interaction, V: Interaction> PairInt<P, V> {
    pub fn new(first: P, second: V) -> Self {
        Self { first, second }
    }
}

impl<P: Interaction, V: Interaction> Interaction for PairInt<P, V> {
    fn value(&self, r: f64) -> f64 {
        self.first.value(r) + self.second.value(r)
    }
}
