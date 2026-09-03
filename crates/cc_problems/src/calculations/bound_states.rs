use std::marker::PhantomData;

use cc_math_utils::brent_root_method;
use coupled_chan::{cc_propagator::{Direction, Propagator, WithNodeCount, WithWaveStorage}, coupling::{Asymptote, CollisionParams, CollisionWMatrix, RCoupling}, multi_channel::log_derivative::{JohnsonLogDerivative, ManolopoulosLogDerivative}};
use hilbert_space::faer::{self, Mat};
use serde::{
    Deserialize,
    Serialize, de::DeserializeOwned,
};
use serde_json::Number;
use unit_systems::quantities::{
    Scalar,
    phys_quantities::{
        Energy,
        Length, Mass,
    },
};

use crate::{UNITS_CONVERTER, calculations::{Modified, SingleCalc, dependence::ModifyParams, scattering::{Boundary, CoupledChanSolver, Step}}, parameters::TypedParamId, problems::Problem, system::{Coupling, System}};

#[derive(Clone, Debug, Deserialize)]
#[serde(bound = "D: DeserializeOwned")]
pub struct BoundStateCalcInput<D: ModifyParams> {
    #[serde(default)]
    pub entrance: usize,
    #[serde(default)]
    pub energy: Scalar<Energy>,

    pub dependant: (D, D),
    pub dependant_err: D,

    #[serde(default)]
    pub boundaries: (Boundary, Boundary),
    pub r_min: Scalar<Length>,
    pub r_match: Scalar<Length>,
    pub r_max: Scalar<Length>,

    pub step: Step,
    pub solver: LogDerivSolver,

    #[serde(default)]
    pub get_occupations: bool,
    #[serde(default)]
    pub get_wave_function: bool,

    #[serde(default)]
    node_monotony: NodeMonotony,
    node_range: Option<NodeRangeTarget>,

    #[serde(default)]
    search_method: BoundSearchMethod,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogDerivSolver {
    JohnsonLogDeriv,
    ManolopoulosLogDeriv,
}

impl Into<CoupledChanSolver> for LogDerivSolver {
    fn into(self) -> CoupledChanSolver {
        match self {
            LogDerivSolver::JohnsonLogDeriv => CoupledChanSolver::JohnsonLogDeriv,
            LogDerivSolver::ManolopoulosLogDeriv => CoupledChanSolver::ManolopoulosLogDeriv,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum NodeMonotony {
    Decreasing,
    #[default]
    Increasing,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundSearchMethod {
    Brent(u32),
    Bisection,
}

impl Default for BoundSearchMethod {
    fn default() -> Self {
        Self::Brent(30)
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum NodeRangeTarget {
    Range(u64, u64),
    BottomRange(u64),
    TopRange(u64),
}

#[derive(Clone, Debug, Serialize)]
pub struct BoundStateData {
    pub nodes: u64,
    pub parameter: f64,

    pub occupations: Option<Vec<f64>>,
    pub wave_function: Option<WaveFunction>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WaveFunction {
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

#[derive(Clone, Debug, Serialize)]
pub struct BoundStatesData(pub Vec<BoundStateData>);

pub struct BoundStateCalc<P: Problem, D: ModifyParams<P = P, C = BoundStateCalcInput<D>>> {
    pub mass: TypedParamId<Scalar<Mass>>,
    phantom: PhantomData<D>
}

impl<P, D> SingleCalc for BoundStateCalc<P, D> 
where
    P: Problem,
    D: ModifyParams<P = P, C = BoundStateCalcInput<D>> + Clone,
{
    type P = P;
    type CalcInput = BoundStateCalcInput<D>;
    type Data = BoundStateData;

    fn calculate(
        &self,
        mut modified: Modified<P, BoundStateCalcInput<D>>,
        _problem: &P,
    ) -> impl IntoIterator<Item = anyhow::Result<Self::Data>> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
        let r_start = converter.scalar_value(&modified.calc_input.r_min);
        let r_match = converter.scalar_value(&modified.calc_input.r_match);
        let r_stop = converter.scalar_value(&modified.calc_input.r_max);
        assert!(r_start < r_match && r_match < r_stop, "expected r_start < r_match < r_stop");

        let (p_start, p_end) = modified.calc_input.dependant.clone();
        let p_start_f64 = modify_to_f64(&p_start);
        let p_end_f64 = modify_to_f64(&p_end);

        p_start.modify(&mut modified);
        let w_matrix = self.get_w_matrix(modified.system, modified.calc_input);
        let mut lower_mismatch = bound_mismatch(&w_matrix, modified.calc_input, p_start_f64);

        p_end.modify(&mut modified);
        let w_matrix = self.get_w_matrix(modified.system, modified.calc_input);
        let mut upper_mismatch = bound_mismatch(&w_matrix, modified.calc_input, p_end_f64);

        if lower_mismatch.nodes > upper_mismatch.nodes {
            std::mem::swap(&mut lower_mismatch, &mut upper_mismatch)
        }

        let mut upper_node = upper_mismatch.nodes;
        let mut lower_node = lower_mismatch.nodes;

        if let Some(nodes_range) = modified.calc_input.node_range {
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

        let mut lower_bounds = vec![None; states_no];
        lower_bounds[0] = Some(lower_mismatch);
        let mut upper_bounds = vec![None; states_no];
        upper_bounds[states_no-1] = Some(upper_mismatch);

        let nodes: Vec<u64> = match modified.calc_input.node_monotony {
            NodeMonotony::Increasing => (lower_node..upper_node).collect(),
            NodeMonotony::Decreasing => (lower_node..upper_node).rev().collect(),
        };

        let mut p_mod = p_start.clone();
        nodes.into_iter().map(move |target_node| {
            let p = match modified.calc_input.search_method {
                BoundSearchMethod::Brent(max_iter) => self.brent_search(
                    &mut modified,
                    &mut lower_bounds, 
                    &mut upper_bounds, 
                    lower_node, 
                    target_node, 
                    max_iter
                ),
                BoundSearchMethod::Bisection => Ok(self.bisection_search(
                    &mut modified,
                    &mut lower_bounds, 
                    &mut upper_bounds, 
                    lower_node, 
                    target_node,
                )),
            };

            p.map(|p| {
                let mut occupations = None;
                let mut wave_function = None;

                if modified.calc_input.get_occupations || modified.calc_input.get_wave_function {
                    modify_from_f64(&mut p_mod, p);
                    p_mod.modify(&mut modified);
                    let w_matrix = self.get_w_matrix(modified.system, modified.calc_input);
                    
                    let wave = bound_wave(&w_matrix, modified.calc_input, target_node);

                    if modified.calc_input.get_occupations {
                        occupations = Some(wave.occupations())
                    }
                    if modified.calc_input.get_wave_function {
                        wave_function = Some(wave)
                    }
                }

                BoundStateData {
                    nodes: target_node,
                    parameter: p,
        
                    occupations,
                    wave_function,
                }
            })
        })
    }
}

impl<P, D> BoundStateCalc<P, D> 
where
    P: Problem,
    D: ModifyParams<P = P, C = BoundStateCalcInput<D>> + Clone,
{
    fn get_w_matrix(&self, system: &mut System, input: &BoundStateCalcInput<D>) -> CollisionWMatrix<Coupling> {
        let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");

        let blocks = system.angular_blocks();
        let collision_params = CollisionParams {
            mass: converter.scalar_value(system.param_registry().get(self.mass)),
            energy: converter.scalar_value(&input.energy),
            entrance: input.entrance,
        };
        let asymptote = Asymptote::new_angular_blocks(blocks, collision_params);

        CollisionWMatrix::new(system.coupling(), asymptote)
    }

    fn brent_search(
        &self, 
        modified: &mut Modified<P, BoundStateCalcInput<D>>,
        lower_bounds: &mut [Option<BoundMismatch>],
        upper_bounds: &mut [Option<BoundMismatch>],
        min_nodes: u64,
        target_nodes: u64,
        max_iter: u32
    ) -> anyhow::Result<f64> {
        let n = lower_bounds.len();
        let node_index = (target_nodes - min_nodes) as usize;
        let p_err = modify_to_f64(&modified.calc_input.dependant_err);

        let mut lower_bound = lower_bounds
            .iter()
            .take(node_index + 1)
            .filter(|&x| x.is_some())
            .next_back()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        let mut upper_bound = upper_bounds
            .iter()
            .skip(node_index + 1)
            .find(|&x| x.is_some())
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();

        let index = (lower_bound.nodes_match + target_nodes - lower_bound.nodes) as usize;
        let mut lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);

        let index = (upper_bound.nodes_match + target_nodes - upper_bound.nodes) as usize;
        let mut upper_eigenvalue = upper_bound.matching_eigenvalues.get(index);

        let monotony = upper_bound.parameter > lower_bound.parameter;

        let mut p_mod = modified.calc_input.dependant_err.clone();
        while upper_bound.nodes != target_nodes + 1
            || lower_bound.nodes != target_nodes
            || lower_eigenvalue.is_none()
            || upper_eigenvalue.is_none()
        {
            let p_mid = (upper_bound.parameter + lower_bound.parameter) / 2.;

            if (upper_bound.parameter - lower_bound.parameter).abs() < p_err.abs() {
                return Ok(p_mid);
            }

            modify_from_f64(&mut p_mod, p_mid);
            p_mod.modify(modified);
            let w_matrix = self.get_w_matrix(modified.system, modified.calc_input);
            let mid_mismatch = bound_mismatch(&w_matrix, modified.calc_input, p_mid);

            if mid_mismatch.nodes <= min_nodes
                && ((lower_bounds[0].as_ref().unwrap().parameter < mid_mismatch.parameter) ^ !monotony) {
                lower_bounds[0] = Some(mid_mismatch.clone())
            }
            else if mid_mismatch.nodes >= min_nodes + n as u64
                && ((upper_bounds[n - 1].as_ref().unwrap().parameter > mid_mismatch.parameter) ^ !monotony) {
                upper_bounds[n - 1] = Some(mid_mismatch.clone())
            } else {
                let index = (mid_mismatch.nodes - min_nodes) as usize;
                if let Some(lower) = &mut lower_bounds[index] 
                    && ((lower.parameter < mid_mismatch.parameter) ^ !monotony) {
                    *lower = mid_mismatch.clone()
                }

                if let Some(upper) = &mut upper_bounds[index - 1] 
                    && ((upper.parameter > mid_mismatch.parameter) ^ !monotony) {
                    *upper = mid_mismatch.clone()
                }
            }

            if mid_mismatch.nodes > target_nodes {
                upper_bound = mid_mismatch;

                let index = (upper_bound.nodes_match + target_nodes - upper_bound.nodes) as usize;
                upper_eigenvalue = upper_bound.matching_eigenvalues.get(index);
            } else {
                lower_bound = mid_mismatch;

                let index = (lower_bound.nodes_match + target_nodes - lower_bound.nodes) as usize;
                lower_eigenvalue = lower_bound.matching_eigenvalues.get(index);
            }
        }

        Ok(brent_root_method(
            [lower_bound.parameter, *lower_eigenvalue.unwrap()],
            [upper_bound.parameter, *upper_eigenvalue.unwrap()],
            |x| {
                modify_from_f64(&mut p_mod, x);
                p_mod.modify(modified);
                let w_matrix = self.get_w_matrix(modified.system, modified.calc_input);
                let mismatch = bound_mismatch(&w_matrix, modified.calc_input, x);

                let index = (mismatch.nodes_match + target_nodes - mismatch.nodes) as usize;

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

    fn bisection_search(        
        &self, 
        modified: &mut Modified<P, BoundStateCalcInput<D>>,
        lower_bounds: &mut [Option<BoundMismatch>],
        upper_bounds: &mut [Option<BoundMismatch>],
        min_nodes: u64,
        target_nodes: u64,
    ) -> f64 {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct BoundMismatch {
    parameter: f64,
    nodes: u64,
    nodes_match: u64,
    matching_eigenvalues: Vec<f64>,
}

pub fn bound_mismatch(w_matrix: &CollisionWMatrix<impl RCoupling>, input: &BoundStateCalcInput<impl ModifyParams>, parameter: f64) -> BoundMismatch {
    let boundary_out = input.boundaries.0.get_boundary(&input.r_min, Direction::Outwards, w_matrix);
    let boundary_in = input.boundaries.0.get_boundary(&input.r_max, Direction::Inwards, w_matrix);

    let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
    let r_match = converter.scalar_value(&input.r_match);

    let (matching_matrix, nodes) = match input.solver {
        LogDerivSolver::JohnsonLogDeriv => {
            let step = input.step.get_step();
            let mut solver_in = JohnsonLogDerivative::new(w_matrix, step, boundary_in);
            let sol_in = solver_in.propagate_to(r_match);

            let step = input.step.get_step();
            let mut solver_out = JohnsonLogDerivative::new(w_matrix, step, boundary_out);
            let sol_out = solver_out.propagate_to(r_match);

            let matching_matrix = &sol_out.sol.0 - &sol_in.sol.0;
            let nodes = solver_in.nodes().0 + solver_out.nodes().0;

            (matching_matrix, nodes)
        },
        LogDerivSolver::ManolopoulosLogDeriv => {
            let step = input.step.get_step();
            let mut solver_in = ManolopoulosLogDerivative::new(w_matrix, step, boundary_in);
            let sol_in = solver_in.propagate_to(r_match);

            let step = input.step.get_step();
            let mut solver_out = ManolopoulosLogDerivative::new(w_matrix, step, boundary_out);
            let sol_out = solver_out.propagate_to(r_match);

            let matching_matrix = &sol_out.sol.0 - &sol_in.sol.0;
            let nodes = solver_in.nodes().0 + solver_out.nodes().0;

            (matching_matrix, nodes)
        },
    };

    let eigenvalues = matching_matrix
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("could not diagonalize matching matrix");

    let nodes_match = eigenvalues.iter().fold(0, |acc, &x| if x < 0. { acc + 1 } else { acc });
    let nodes = nodes + nodes_match;

    BoundMismatch {
        parameter,
        nodes,
        nodes_match,
        matching_eigenvalues: eigenvalues,
    }
}

fn bound_wave(w_matrix: &CollisionWMatrix<impl RCoupling>, input: &BoundStateCalcInput<impl ModifyParams>, target_nodes: u64) -> WaveFunction {
    let boundary_out = input.boundaries.0.get_boundary(&input.r_min, Direction::Outwards, w_matrix);
    let boundary_in = input.boundaries.0.get_boundary(&input.r_max, Direction::Inwards, w_matrix);

    let converter = UNITS_CONVERTER.read().expect("Could not obtain UNITS_CONVERTER");
    let r_match = converter.scalar_value(&input.r_match);

    // todo! code duplication
    match input.solver {
        LogDerivSolver::JohnsonLogDeriv => {
            let step = input.step.get_step();
            let mut solver_in = JohnsonLogDerivative::new(w_matrix, step, boundary_in);
            solver_in.init_wave_storage();
            let sol_in = solver_in.propagate_to(r_match);

            let step = input.step.get_step();
            let mut solver_out = JohnsonLogDerivative::new(w_matrix, step, boundary_out);
            solver_out.init_wave_storage();
            let sol_out = solver_out.propagate_to(r_match);

            let matching_matrix = &sol_out.sol.0 - &sol_in.sol.0;
            let nodes = solver_in.nodes().0 + solver_out.nodes().0;

            let eigen = matching_matrix
                .self_adjoint_eigen(faer::Side::Lower)
                .expect("could not diagonalize matching matrix");

            let init_wave = eigen.U().col((target_nodes - nodes) as usize);

            let wave_in = solver_in.get_wave_storage().as_ref().unwrap().reconstruct(init_wave, false);
            let wave_in = WaveFunction {
                distances: wave_in.0,
                values: wave_in.1,
            };

            let wave_out = solver_out.get_wave_storage().as_ref().unwrap().reconstruct(init_wave, false);
            let mut wave_out = WaveFunction {
                distances: wave_out.0,
                values: wave_out.1,
            };

            wave_out.reverse();
            wave_out.distances.extend(wave_in.distances);
            wave_out.values.extend(wave_in.values);

            wave_out.normalize()
        },
        LogDerivSolver::ManolopoulosLogDeriv => {
            let step = input.step.get_step();
            let mut solver_in = ManolopoulosLogDerivative::new(w_matrix, step, boundary_in);
            let sol_in = solver_in.propagate_to(r_match);

            let step = input.step.get_step();
            let mut solver_out = ManolopoulosLogDerivative::new(w_matrix, step, boundary_out);
            let sol_out = solver_out.propagate_to(r_match);

            let matching_matrix = &sol_out.sol.0 - &sol_in.sol.0;
            let nodes = solver_in.nodes().0 + solver_out.nodes().0;
            
            let eigen = matching_matrix
                .self_adjoint_eigen(faer::Side::Lower)
                .expect("could not diagonalize matching matrix");


            let init_wave = eigen.U().col((target_nodes - nodes) as usize);

                        let wave_in = solver_in.get_wave_storage().as_ref().unwrap().reconstruct(init_wave, false);
            let wave_in = WaveFunction {
                distances: wave_in.0,
                values: wave_in.1,
            };

            let wave_out = solver_out.get_wave_storage().as_ref().unwrap().reconstruct(init_wave, false);
            let mut wave_out = WaveFunction {
                distances: wave_out.0,
                values: wave_out.1,
            };

            wave_out.reverse();
            wave_out.distances.extend(wave_in.distances);
            wave_out.values.extend(wave_in.values);

            wave_out.normalize()
        },
    }
}

fn modify_from_f64<D: ModifyParams>(modify: &mut D, value: f64) {
    modify.from_number(Number::from_f64(value).unwrap())
}

fn modify_to_f64<D: ModifyParams>(value: &D) -> f64 {
    value.as_number().as_f64().unwrap()
}