use crate::interaction::Interaction;

pub type DynComposite = CompositeInt<Box<dyn Interaction>>;

#[derive(Debug, Clone)]
pub struct CompositeInt<P: Interaction> {
    interactions: Vec<P>,
}

impl<P: Interaction> Default for CompositeInt<P> {
    fn default() -> Self {
        Self {
            interactions: Vec::new(),
        }
    }
}

impl<P: Interaction> CompositeInt<P> {
    pub fn new(interactions: Vec<P>) -> Self {
        Self { interactions }
    }

    pub fn add_interaction(&mut self, interaction: P) -> &mut Self {
        self.interactions.push(interaction);

        self
    }
}

impl<P: Interaction> Interaction for CompositeInt<P> {
    fn value(&self, r: f64) -> f64 {
        self.interactions.iter().fold(0., |acc, p| acc + p.value(r))
    }
}
