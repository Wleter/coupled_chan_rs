use std::mem::swap;

use anyhow::Result;
use coupled_chan::{
    Operator,
    constants::units::{Quantity, atomic_units::Bohr},
    coupling::WMatrix,
    log_derivative::diabatic::{DiabaticLogDerivative, LogDerivativeReference, WaveLogDerivStorage},
    propagator::{Boundary, Direction, NodeCountPropagator, Propagator},
    vanishing_boundary,
};
use hilbert_space::faer::{self, Mat};
use math_utils::brent_root_method;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct BoundMismatch {
    pub parameter: f64,
    pub nodes: u64,
    pub matching_matrix: Mat<f64>,
    pub matching_eigenvalues: Vec<f64>,
}

#[derive(Serialize, Clone, Debug)]
pub struct BoundState {
    pub parameter: f64,
    pub node: u64,
    pub occupations: Option<Vec<f64>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct WaveFunction {
    pub parameter: f64,
    pub distances: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

impl WaveFunction {
    pub fn reverse(&mut self) {
        self.distances.reverse();
        self.values.reverse();
    }

    pub fn normalize(mut self) -> Self {
        let normalization: f64 = self
            .distances
            .windows(2)
            .zip(self.values.windows(2))
            .map(|(x, f)| unsafe {
                let f1 = f.get_unchecked(1);
                let f0 = f.get_unchecked(0);
                let f1_norm = f1.iter().fold(0., |acc, x| acc + x * x);
                let f0_norm = f0.iter().fold(0., |acc, x| acc + x * x);

                0.5 * (x.get_unchecked(1) - x.get_unchecked(0)) * (f1_norm + f0_norm)
            })
            .sum();

        for v in &mut self.values {
            for p in v {
                *p /= normalization.sqrt()
            }
        }

        self
    }

    pub fn occupations(&self) -> Vec<f64> {
        self.distances
            .windows(2)
            .zip(self.values.windows(2))
            .fold(vec![0.; self.values[0].len()], |mut acc, (d, v)| {
                for (i, acc) in acc.iter_mut().enumerate() {
                    *acc += 0.5 * (d[1] - d[0]) * (v[1][i].powi(2) + v[0][i].powi(2))
                }

                acc
            })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum NodeMonotony {
    Decreasing,
    #[default]
    Increasing,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundMethod {
    Brent(u32),
    Bisection,
}

impl Default for BoundMethod {
    fn default() -> Self {
        Self::Brent(30)
    }
}

#[derive(Clone, Debug, Copy)]
pub enum NodeRangeTarget {
    Range(u64, u64),
    BottomRange(u64),
    TopRange(u64),
}

pub type Prop<'a, W, L> = Box<dyn for<'w> Fn(Boundary<Operator>, &'w W) -> DiabaticLogDerivative<'w, L, W> + 'a>;

// todo! currently works only with DiabaticLogDerivative, make it work for all
pub struct BoundStatesFinder<'a, W, L>
where
    W: WMatrix,
    L: LogDerivativeReference,
{
    prob: Option<Box<dyn Fn(f64) -> W + 'a>>,
    prop: Option<Prop<'a, W, L>>,

    parameter_range: Option<[f64; 2]>,
    parameter_err: Option<f64>,
    node_range: Option<NodeRangeTarget>,

    r_range: Option<[Quantity<Bohr>; 3]>,

    node_monotony: NodeMonotony,
    method: BoundMethod,
}

impl<W, L> Default for BoundStatesFinder<'_, W, L>
where
    W: WMatrix,
    L: LogDerivativeReference,
{
    fn default() -> Self {
        Self {
            prob: Default::default(),
            prop: Default::default(),
            parameter_range: Default::default(),
            parameter_err: Default::default(),
            node_range: Default::default(),
            r_range: Default::default(),
            node_monotony: Default::default(),
            method: Default::default(),
        }
    }
}

impl<'a, W, L> BoundStatesFinder<'a, W, L>
where
    W: WMatrix,
    L: LogDerivativeReference,
{
    pub fn set_propagator(
        mut self,
        prop: impl for<'w> Fn(Boundary<Operator>, &'w W) -> DiabaticLogDerivative<'w, L, W> + 'a,
    ) -> Self {
        self.prop = Some(Box::new(prop));
        self
    }

    pub fn set_problem(mut self, prob: impl Fn(f64) -> W + 'a) -> Self {
        self.prob = Some(Box::new(prob));
        self
    }

    pub fn set_r_range(mut self, r_range: [Quantity<Bohr>; 3]) -> Self {
        assert!(r_range[0].value() <= r_range[2].value(), "Invalid r range");
        self.r_range = Some(r_range);
        self
    }

    pub fn set_parameter_range(mut self, p_range: [f64; 2], p_err: f64) -> Self {
        assert!(p_range[0] <= p_range[1], "Invalid parameter range");
        self.parameter_range = Some(p_range);
        self.parameter_err = Some(p_err);
        self
    }

    pub fn set_node_monotony(mut self, monotony: NodeMonotony) -> Self {
        self.node_monotony = monotony;
        self
    }

    pub fn set_method(mut self, method: BoundMethod) -> Self {
        self.method = method;
        self
    }

    pub fn set_node_range(mut self, node_range: NodeRangeTarget) -> Self {
        if let NodeRangeTarget::Range(a, b) = node_range {
            assert!(a <= b, "Invalid nodes range");
        }

        self.node_range = Some(node_range);
        self
    }

    pub fn bound_mismatch(&self, parameter: f64) -> BoundMismatch {
        let problem = self.prob.as_ref().expect("Did not set problem via set_problem");
        let r_range = self.r_range.expect("Did not set r_range via set_r_range");
        let prop = self.prop.as_ref().expect("Did not set propagator via set_propagator");

        let w_matrix = problem(parameter);

        let boundary_out = vanishing_boundary(r_range[0], Direction::Outwards, &w_matrix);
        let boundary_in = vanishing_boundary(r_range[2], Direction::Inwards, &w_matrix);

        let mut propagator_in = prop(boundary_in, &w_matrix);
        let sol_in = propagator_in.propagate_to(r_range[1].value());

        let mut propagator_out = prop(boundary_out, &w_matrix);
        let sol_out = propagator_out.propagate_to(r_range[1].value());

        let matching_matrix = &sol_out.sol.0.0 - &sol_in.sol.0.0;
        let nodes = propagator_in.nodes().0 + propagator_out.nodes().0;

        let eigenvalues = matching_matrix
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("could not diagonalize matching matrix");

        let nodes = nodes + eigenvalues.iter().fold(0, |acc, &x| if x < 0. { acc + 1 } else { acc });

        BoundMismatch {
            parameter,
            nodes,
            matching_matrix,
            matching_eigenvalues: eigenvalues,
        }
    }

    pub fn bound_states(&self) -> impl Iterator<Item = Result<BoundState>> {
        let p_range = self
            .parameter_range
            .expect("Did not set parameter range via set_parameter_range");

        let mut lower_mismatch = self.bound_mismatch(p_range[0]);
        let mut upper_mismatch = self.bound_mismatch(p_range[1]);

        if lower_mismatch.nodes > upper_mismatch.nodes {
            swap(&mut lower_mismatch, &mut upper_mismatch)
        }

        let mut upper_node = upper_mismatch.nodes;
        let mut lower_node = lower_mismatch.nodes;

        if let Some(nodes_range) = self.node_range {
            match nodes_range {
                NodeRangeTarget::Range(a, b) => {
                    lower_node = lower_node.max(a);
                    upper_node = upper_node.min(b + 1);
                }
                NodeRangeTarget::BottomRange(a) => upper_node = upper_node.min(lower_node + a),
                NodeRangeTarget::TopRange(a) => lower_node = lower_node.max(upper_node.saturating_sub(a)),
            }
        }
        let states_no = (upper_node - lower_node) as usize;

        let mut mismatch_node = vec![None; states_no + 1];
        mismatch_node[0] = Some(lower_mismatch);
        mismatch_node[states_no] = Some(upper_mismatch);

        let nodes: Vec<u64> = match self.node_monotony {
            NodeMonotony::Increasing => (lower_node..upper_node).collect(),
            NodeMonotony::Decreasing => (lower_node..upper_node).rev().collect(),
        };

        nodes.into_iter().map(move |target_node| {
            let p = match self.method {
                BoundMethod::Brent(max_iter) => self.brent_search(lower_node, target_node, &mut mismatch_node, max_iter),
                BoundMethod::Bisection => self.bisection_search(lower_node, target_node, &mut mismatch_node),
            };

            p.map(|p| BoundState {
                parameter: p,
                node: target_node,
                occupations: None,
            })
        })
    }

    pub fn bound_wave(&self, bound: &BoundState) -> WaveFunction {
        let parameter = bound.parameter;
        let problem = self.prob.as_ref().expect("Did not set problem via set_problem");
        let r_range = self.r_range.expect("Did not set r_range via set_r_range");
        let prop = self.prop.as_ref().expect("Did not set propagator via set_propagator");

        let w_matrix = problem(parameter);

        let boundary_out = vanishing_boundary(r_range[0], Direction::Outwards, &w_matrix);
        let boundary_in = vanishing_boundary(r_range[2], Direction::Inwards, &w_matrix);

        let mut propagator_in = prop(boundary_in, &w_matrix);
        propagator_in.with_wave_storage(WaveLogDerivStorage::new(true));
        let sol_in = propagator_in.propagate_to(r_range[1].value());

        let mut propagator_out = prop(boundary_out, &w_matrix);
        propagator_out.with_wave_storage(WaveLogDerivStorage::new(true));
        let sol_out = propagator_out.propagate_to(r_range[1].value());

        let matching_matrix = &sol_out.sol.0.0 - &sol_in.sol.0.0;

        let eigen = matching_matrix
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("could not diagonalize matching matrix");

        let index = eigen
            .S()
            .column_vector()
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.abs().partial_cmp(&y.1.abs()).unwrap())
            .unwrap()
            .0;

        let init_wave = eigen.U().col(index);

        let wave_in = propagator_in.wave_storage().as_ref().unwrap().reconstruct(init_wave);
        let wave_in = WaveFunction {
            parameter,
            distances: wave_in.0,
            values: wave_in.1,
        };

        let wave_out = propagator_out.wave_storage().as_ref().unwrap().reconstruct(init_wave);
        let mut wave_out = WaveFunction {
            parameter,
            distances: wave_out.0,
            values: wave_out.1,
        };

        wave_out.reverse();
        wave_out.distances.extend(wave_in.distances);
        wave_out.values.extend(wave_in.values);

        wave_out.normalize()
    }

    fn bisection_search(
        &self,
        index_offset: u64,
        target_nodes: u64,
        mismatch_node: &mut [Option<BoundMismatch>],
    ) -> Result<f64> {
        let p_err = self.parameter_err.unwrap();

        let node_index = (target_nodes - index_offset) as usize;

        let mut lower_bound = mismatch_node
            .iter()
            .take(node_index + 1)
            .filter(|&x| x.is_some())
            .next_back()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        let mut upper_bound = mismatch_node
            .iter()
            .skip(node_index + 1)
            .find(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        while (upper_bound.parameter - lower_bound.parameter).abs() > p_err {
            let field_mid = (upper_bound.parameter + lower_bound.parameter) / 2.;

            let mid_mismatch = self.bound_mismatch(field_mid);

            let index = if mid_mismatch.nodes <= index_offset {
                0
            } else if mid_mismatch.nodes >= index_offset + mismatch_node.len() as u64 {
                mismatch_node.len() - 1
            } else {
                (mid_mismatch.nodes - index_offset) as usize
            };

            if mismatch_node[index].is_none() || index == 0 || index + 1 == mismatch_node.len() {
                mismatch_node[index] = Some(mid_mismatch.clone())
            }

            if mid_mismatch.nodes > target_nodes {
                upper_bound = mid_mismatch
            } else {
                lower_bound = mid_mismatch
            }
        }

        Ok((upper_bound.parameter + lower_bound.parameter) / 2.)
    }

    fn brent_search(
        &self,
        index_offset: u64,
        target_nodes: u64,
        mismatch_node: &mut [Option<BoundMismatch>],
        max_iter: u32,
    ) -> Result<f64> {
        let p_err = self.parameter_err.unwrap();

        let node_index = (target_nodes - index_offset) as usize;

        let mut lower_bound = mismatch_node
            .iter()
            .take(node_index + 1)
            .filter(|&x| x.is_some())
            .next_back()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        let mut upper_bound = mismatch_node
            .iter()
            .skip(node_index + 1)
            .find(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        let index = lower_bound.matching_eigenvalues.partition_point(|&x| x < 0.);
        let mut lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);

        let index = upper_bound.matching_eigenvalues.partition_point(|&x| x < 0.);
        let mut upper_eigenvalue = upper_bound.matching_eigenvalues.get(index - 1);

        while upper_bound.nodes != target_nodes + 1
            || lower_bound.nodes != target_nodes
            || lower_eigenvalue.is_none()
            || upper_eigenvalue.is_none()
        {
            let field_mid = (upper_bound.parameter + lower_bound.parameter) / 2.;

            if (upper_bound.parameter - lower_bound.parameter).abs() < p_err.abs() {
                return Ok(field_mid);
            }

            let mid_mismatch = self.bound_mismatch(field_mid);

            let index = if mid_mismatch.nodes <= index_offset {
                0
            } else if mid_mismatch.nodes >= index_offset + mismatch_node.len() as u64 {
                mismatch_node.len() - 1
            } else {
                (mid_mismatch.nodes - index_offset) as usize
            };

            if mismatch_node[index].is_none() || index == 0 || index + 1 == mismatch_node.len() {
                mismatch_node[index] = Some(mid_mismatch.clone());
            }

            if mid_mismatch.nodes > target_nodes {
                upper_bound = mid_mismatch;

                let index = upper_bound.matching_eigenvalues.partition_point(|&x| x < 0.);
                upper_eigenvalue = upper_bound.matching_eigenvalues.get(index - 1);
            } else {
                lower_bound = mid_mismatch;

                let index = lower_bound.matching_eigenvalues.partition_point(|&x| x < 0.);
                lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);
            }
        }

        Ok(brent_root_method(
            [lower_bound.parameter, *lower_eigenvalue.unwrap()],
            [upper_bound.parameter, *upper_eigenvalue.unwrap()],
            |x| {
                let mismatch = self.bound_mismatch(x);

                let index = mismatch.matching_eigenvalues.partition_point(|&x| x < 0.);

                if mismatch.nodes > target_nodes {
                    mismatch.matching_eigenvalues[index - 1]
                } else {
                    mismatch.matching_eigenvalues[index]
                }
            },
            p_err,
            max_iter,
        )?)
    }
}
