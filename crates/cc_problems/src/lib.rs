pub mod atom_rotor_basis;
pub mod atom_structure;
pub mod bound_states;
pub mod diatom_basis;
pub mod homo_diatom_basis;
pub mod operator_mel;
pub mod prelude;
pub mod problems;
pub mod rotor_structure;
pub mod system_structure;
pub mod tram_basis;

pub use anyhow;
pub use coupled_chan;
pub use hilbert_space;
pub use math_utils::{linspace, logspace};
pub use qol_utils;
pub use rayon;
use serde::Serialize;
pub use spin_algebra;

use coupled_chan::{
    Operator,
    constants::{Bohr, Quantity},
    coupling::{AngularBlocks, Levels, WMatrix},
    log_derivative::diabatic::{DiabaticLogDerivative, LogDerivativeReference},
    propagator::{Boundary, Direction, Propagator, Repr, Solution, step_strategy::StepStrategy},
    s_matrix::{SMatrix, SMatrixGetter},
    vanishing_boundary,
};
use hilbert_space::{
    cast_variant,
    dyn_space::{BasisElementIndices, BasisElements, BasisElementsRef, BasisId, DynSubspaceElement},
};

use crate::{
    bound_states::BoundState,
    prelude::{BoundStatesFinder, NodeMonotony, NodeRangeTarget},
    system_structure::AngularBasis,
};

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct AngularMomentum(pub u32);

impl std::fmt::Debug for AngularMomentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone)]
pub struct AngularBasisElements {
    pub full_basis: BasisElements,
    ls: Vec<AngularMomentum>,
    separated_basis_indices: Vec<Vec<BasisElementIndices>>,
}

impl std::fmt::Debug for AngularBasisElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AngularBasisElements")
            .field("full_basis", &self.full_basis)
            .finish()
    }
}

impl AngularBasisElements {
    pub fn new_angular(full_basis: BasisElements, system: &AngularBasis) -> Self {
        Self::new(full_basis, system.l, |&a| a)
    }

    pub fn new<T: DynSubspaceElement>(
        full_basis: BasisElements,
        l_index: BasisId,
        conversion: impl Fn(&T) -> AngularMomentum,
    ) -> Self {
        let basis = full_basis.basis;

        let mut angular_indices: Vec<(AngularMomentum, BasisElementIndices)> = full_basis
            .elements_indices
            .into_iter()
            .map(|indices| (conversion(cast_variant!(dyn indices.index(l_index, &basis), T)), indices))
            .collect();
        angular_indices.sort_by_key(|(l, _)| *l);
        let ordered_indices = angular_indices.iter().map(|x| x.1.clone()).collect();

        let ordered_basis = BasisElements {
            basis,
            elements_indices: ordered_indices,
        };

        let mut l_prev: Option<AngularMomentum> = None;
        let mut separated_basis_indices = vec![];
        let mut ls = vec![];
        for (l, index) in angular_indices {
            let l_changed = if let Some(l_prev) = l_prev { l_prev != l } else { true };

            if l_changed {
                ls.push(l);
                separated_basis_indices.push(vec![index])
            } else {
                separated_basis_indices.last_mut().unwrap().push(index)
            }

            l_prev = Some(l)
        }

        Self {
            full_basis: ordered_basis,
            ls,
            separated_basis_indices,
        }
    }

    pub fn angular_iter<'a>(&'a self) -> impl Iterator<Item = BasisElementsRef<'a>> {
        self.separated_basis_indices.iter().map(|indices| BasisElementsRef {
            basis: &self.full_basis.basis,
            elements_indices: indices,
        })
    }

    pub fn get_angular_blocks(&self, mut f: impl FnMut(&BasisElementsRef) -> Operator) -> AngularBlocks {
        let blocks = self.angular_iter().map(|e| f(&e)).collect();

        AngularBlocks {
            l: self.ls.iter().map(|a| a.0).collect(),
            angular_blocks: blocks,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScatteringProblem {
    pub r_min: Quantity<Bohr>,
    pub r_max: Quantity<Bohr>,
    pub step_strat: StepStrategy,
}

impl ScatteringProblem {
    pub fn get_s_matrix<'a, W, R, P>(
        &self,
        w_matrix: &'a W,
        prop: impl Fn(&'a W, StepStrategy, Boundary<Operator>) -> P,
    ) -> SMatrix
    where
        W: WMatrix,
        R: Repr,
        P: Propagator<R> + 'a,
        Solution<R>: SMatrixGetter,
    {
        let boundary = vanishing_boundary(self.r_min.value(), Direction::Outwards, w_matrix);

        let mut propagator = prop(w_matrix, self.step_strat.clone(), boundary);
        let solution = propagator.propagate_to(self.r_max.value());

        solution.get_s_matrix(w_matrix)
    }
}

#[derive(Clone, Debug)]
pub struct BoundProblem {
    pub r_min: Quantity<Bohr>,
    pub r_match: Quantity<Bohr>,
    pub r_max: Quantity<Bohr>,
    pub step_strat: StepStrategy,

    pub node_range: Option<NodeRangeTarget>,
    pub node_monotony: NodeMonotony,
}

impl BoundProblem {
    pub fn get_bound_finder<'a, W, L>(
        &'a self,
        parameter_range: (f64, f64),
        parameter_err: f64,
        problem: impl Fn(f64) -> W + 'a,
        prop: impl for<'w> Fn(&'w W, StepStrategy, Boundary<Operator>) -> DiabaticLogDerivative<'w, L, W> + 'a,
    ) -> BoundStatesFinder<'a, W, L>
    where
        W: WMatrix,
        L: LogDerivativeReference,
    {
        let mut b = BoundStatesFinder::default()
            .set_parameter_range([parameter_range.0, parameter_range.1], parameter_err)
            .set_problem(problem)
            .set_r_range([self.r_min, self.r_match, self.r_max])
            .set_propagator(move |b, w| prop(w, self.step_strat.clone(), b))
            .set_node_monotony(self.node_monotony);

        if let Some(node_range) = self.node_range {
            b = b.set_node_range(node_range);
        }

        b
    }
}

#[derive(Serialize)]
pub struct SMatrixData<T> {
    pub parameter: T,
    pub s_length_re: f64,
    pub s_length_im: f64,
    pub elastic_cross_section: f64,
    pub tot_inelastic_cross_section: f64,
    pub inelastic_cross_sections: Vec<f64>,
}

impl<T> SMatrixData<T> {
    pub fn new(parameter: T, s_matrix: SMatrix) -> Self {
        let s_length = s_matrix.get_scattering_length();
        let elastic_cross_section = s_matrix.get_elastic_cross_sect();
        let tot_inelastic_cross_section = s_matrix.get_inelastic_cross_sect();
        let inelastic_cross_sections = s_matrix.get_inelastic_cross_sects();

        Self {
            parameter,
            s_length_re: s_length.re,
            s_length_im: s_length.im,
            elastic_cross_section,
            tot_inelastic_cross_section,
            inelastic_cross_sections,
        }
    }
}

#[derive(Serialize)]
pub struct LevelsData<T> {
    pub parameter: T,
    pub levels: Vec<f64>,
}

impl<T> LevelsData<T> {
    pub fn new(parameter: T, levels: &Levels) -> Self {
        Self {
            parameter,
            levels: levels.asymptote.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct BoundStateData<T> {
    pub parameter: T,

    pub bound_parameter: f64,
    pub nodes: u64,
    pub occupations: Option<Vec<f64>>,
}

impl<T> BoundStateData<T> {
    pub fn new(parameter: T, bound_state: BoundState) -> Self {
        Self {
            parameter,
            bound_parameter: bound_state.parameter,
            nodes: bound_state.node,
            occupations: bound_state.occupations,
        }
    }
}
