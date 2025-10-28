use crate::coupling::VanishingCoupling;

pub type DynComposite = Composite<Box<dyn VanishingCoupling>>;

#[derive(Debug, Clone)]
pub struct Composite<P: VanishingCoupling> {
    couplings: Vec<P>,
}

impl<P: VanishingCoupling> Default for Composite<P> {
    fn default() -> Self {
        Self { couplings: Vec::new() }
    }
}

impl<P: VanishingCoupling> Composite<P> {
    pub fn new(couplings: Vec<P>) -> Self {
        if !couplings.is_empty() {
            let first_size = couplings.first().unwrap().size();
            assert!(
                couplings.iter().all(|x| x.size() == first_size),
                "All couplings should have the same channel number"
            )
        }

        Self { couplings }
    }

    pub fn add_coupling(&mut self, coupling: P) -> &mut Self {
        if !self.couplings.is_empty() {
            assert_eq!(
                coupling.size(),
                self.size(),
                "All coupling should have the same channel number"
            )
        }
        self.couplings.push(coupling);

        self
    }
}

impl<P: VanishingCoupling> VanishingCoupling for Composite<P> {
    fn value_inplace(&self, r: f64, channels: &mut crate::Operator) {
        let mut couplings = self.couplings.iter();

        if let Some(c) = couplings.next() {
            c.value_inplace(r, channels);
        }

        for c in couplings {
            c.value_inplace_add(r, channels);
        }
    }

    fn value_inplace_add(&self, r: f64, channels: &mut crate::Operator) {
        for c in self.couplings.iter() {
            c.value_inplace_add(r, channels);
        }
    }

    fn size(&self) -> usize {
        if let Some(c) = self.couplings.first() { c.size() } else { 0 }
    }
}
