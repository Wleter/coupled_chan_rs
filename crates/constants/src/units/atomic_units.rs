use crate::impl_unit_group;

#[derive(Clone, Copy, Default, Debug)]
pub struct AuEnergy;
impl_unit_group!(AuEnergy, AuEnergy, 1.);

#[derive(Clone, Copy, Default, Debug)]
pub struct Kelvin;
impl_unit_group!(Kelvin, AuEnergy, 3.1668115634564e-6);

#[derive(Clone, Copy, Default, Debug)]
pub struct CmInv;
impl_unit_group!(CmInv, AuEnergy, 4.5563352529132e-6);

#[derive(Clone, Copy, Default, Debug)]
pub struct GHz;
impl_unit_group!(GHz, AuEnergy, 1.51982850071586e-07);

#[derive(Clone, Copy, Default, Debug)]
pub struct MHz;
impl_unit_group!(MHz, AuEnergy, 1.51982850071586e-10);

#[derive(Clone, Copy, Default, Debug)]
pub struct Bohr;
impl_unit_group!(Bohr, Bohr, 1.);

#[derive(Clone, Copy, Default, Debug)]
pub struct Angstrom;
impl_unit_group!(Angstrom, Bohr, 1. / 0.529177210544);

#[derive(Clone, Copy, Default, Debug)]
pub struct AuMass;
impl_unit_group!(AuMass, AuMass, 1.);

#[derive(Clone, Copy, Default, Debug)]
pub struct Dalton;
impl_unit_group!(Dalton, AuMass, 1. / 5.485799090441e-4);

#[derive(Clone, Copy, Default, Debug)]
pub struct Gauss;
impl_unit_group!(Gauss, Gauss, 1.);

#[derive(Clone, Copy, Default, Debug)]
pub struct Tesla;
impl_unit_group!(Tesla, Gauss, 10_000.);