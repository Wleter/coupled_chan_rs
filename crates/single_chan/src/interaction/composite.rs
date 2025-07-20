use crate::interaction::Interaction;

pub type DynComposite = Composite<Box<dyn Interaction>>;

#[derive(Debug, Clone)]
pub struct Composite<P: Interaction> {
    interactions: Vec<P>,
}

impl<P: Interaction> Default for Composite<P> {
    fn default() -> Self {
        Self {
            interactions: Vec::new(),
        }
    }
}

impl<P: Interaction> Composite<P> {
    pub fn new(interactions: Vec<P>) -> Self {
        Self { interactions }
    }

    pub fn add_interaction(&mut self, interaction: P) -> &mut Self {
        self.interactions.push(interaction);

        self
    }
}

impl<P: Interaction> Interaction for Composite<P> {
    fn value(&self, r: f64) -> f64 {
        self.interactions.iter().fold(0., |acc, p| acc + p.value(r))
    }

    fn asymptote(&self) -> f64 {
        self.interactions.iter().fold(0., |acc, p| acc + p.asymptote())
    }
}
