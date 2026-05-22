use pyo3::prelude::*;

#[pymodule]
mod coupled_chan_py {
    use pyo3::prelude::*;

    use coupled_chan::{
        CollisionWFunction,
        Interaction,
        cc_propagator::{
            Boundary,
            Direction,
            Propagator,
            step_strategy::{
                DynStep,
                LocalWavelengthStep,
                SingleStep,
                TransitionStep,
            },
        },
        coupling::{
            self,
            AngularBlocks,
            DynRCoupling,
            RCoupling,
        },
        multi_channel::{
            Matrix,
            log_derivative::{
                JohnsonLogDerivative,
                ManolopoulosLogDerivative,
            },
            numerov::RatioNumerov,
        },
        scaled::Scaled,
        single_chan,
    };
    use faer_ext::{
        IntoFaer,
        numpy::PyReadonlyArray2,
    };

    #[pyclass]
    struct NumpyInteraction {
        potential_func: Py<PyAny>,
    }

    impl Clone for NumpyInteraction {
        fn clone(&self) -> Self {
            let cloned = pyo3::Python::attach(|py| self.potential_func.clone_ref(py));

            Self { potential_func: cloned }
        }
    }

    impl Interaction for NumpyInteraction {
        fn value(&self, r: f64) -> f64 {
            pyo3::Python::attach(|py| {
                self.potential_func
                    .call1(py, (r,))
                    .expect("Expected function with signature fn(r) -> f64")
                    .extract(py)
                    .expect("Expected function with signature fn(r) -> f64")
            })
        }

        fn asymptote_dep(&self) -> coupled_chan::AsymptoteDep {
            coupled_chan::AsymptoteDep::Unknown
        }
    }

    #[pymethods]
    impl NumpyInteraction {
        #[new]
        #[pyo3(signature = (potential_func: "Callable[[float], float]") -> "NumpyInteraction")]
        pub fn new(potential_func: Py<PyAny>) -> Self {
            Self {
                potential_func: potential_func,
            }
        }
    }

    #[pyclass]
    struct NumpyCoupling {
        matrix_func: Py<PyAny>,
        size: usize,
    }

    impl Clone for NumpyCoupling {
        fn clone(&self) -> Self {
            let cloned = pyo3::Python::attach(|py| self.matrix_func.clone_ref(py));

            Self {
                matrix_func: cloned,
                size: self.size,
            }
        }
    }

    #[pymethods]
    impl NumpyCoupling {
        #[new]
        #[pyo3(signature = (size: "int", potential_func: "Callable[[float], np.array]") -> "NumpyInteraction")]
        pub fn new(size: usize, potential_func: Py<PyAny>) -> Self {
            Self {
                matrix_func: potential_func,
                size,
            }
        }
    }

    impl RCoupling for NumpyCoupling {
        fn value_inplace_add(&self, r: f64, channels: &mut Matrix) {
            pyo3::Python::attach(|py| {
                let array: PyReadonlyArray2<f64> = self
                    .matrix_func
                    .call1(py, (r,))
                    .expect("Expected function with signature fn(r) -> np.array")
                    .extract(py)
                    .expect("Expected function with signature fn(r) -> np.array");
                let array = array.into_faer(); // todo! incompatible crate versions

                for i in 0..self.size {
                    for j in 0..self.size {
                        channels[(i, j)] += array[(i, j)]
                    }
                }
            });
        }

        fn size(&self) -> usize {
            self.size
        }

        fn asymptote_dep(&self) -> coupled_chan::AsymptoteDep {
            coupled_chan::AsymptoteDep::Unknown
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct Asymptote(coupling::Asymptote);

    #[pymethods]
    impl Asymptote {
        #[new]
        pub fn new(l: Vec<u32>, matrices: Vec<PyReadonlyArray2<f64>>, entrance: usize) -> Self {
            let blocks = matrices
                .into_iter()
                .map(|x| {
                    let array = x.into_faer().cloned(); // todo! incompatible crate versions

                    Matrix::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)])
                })
                .collect();

            let angular_blocks = AngularBlocks { l, blocks };
            let params = coupling::SystemParams::new(0.0, 0.0, entrance);

            Self(coupling::Asymptote::new_angular_blocks(angular_blocks, params))
        }

        #[staticmethod]
        pub fn new_single(l: u32, matrix: PyReadonlyArray2<f64>, entrance: usize) -> Self {
            let array = matrix.into_faer().cloned(); // todo! incompatible crate versions
            let matrix = Matrix::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)]);

            let angular_blocks = AngularBlocks {
                l: vec![l],
                blocks: vec![matrix],
            };
            let params = coupling::SystemParams::new(0.0, 0.0, entrance);

            Self(coupling::Asymptote::new_angular_blocks(angular_blocks, params))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct CollisionWMatrix(coupling::CollisionWMatrix<DynRCoupling>);

    #[pymethods]
    impl CollisionWMatrix {
        #[new]
        pub fn new(asymptote: Asymptote, coupling: NumpyCoupling) -> Self {
            let w_matrix = coupling::CollisionWMatrix::new(DynRCoupling::new(coupling), asymptote.0);

            Self(w_matrix)
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct ScatteringStep(DynStep);

    #[pymethods]
    impl ScatteringStep {
        #[new]
        fn new(dr_min: f64, dr_max: f64, dr_ratio: f64) -> Self {
            Self(DynStep::new(LocalWavelengthStep::new(dr_min, dr_max, dr_ratio)))
        }

        #[staticmethod]
        fn new_single(dr: f64) -> Self {
            Self(DynStep::new(SingleStep::new(dr)))
        }

        #[staticmethod]
        fn new_transitioned(r_switch: f64, step_near: ScatteringStep, step_far: ScatteringStep) -> Self {
            Self(DynStep::new(TransitionStep {
                r_switch,
                step_near: step_near.0,
                step_far: step_far.0,
            }))
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct ScatteringConfig {
        #[pyo3(get, set)]
        r_start: f64,
        #[pyo3(get, set)]
        r_stop: f64,
        #[pyo3(get, set)]
        step: ScatteringStep,
    }

    #[pymethods]
    impl ScatteringConfig {
        #[new]
        fn new(r_start: f64, r_stop: f64, step: ScatteringStep) -> Self {
            Self { r_start, r_stop, step }
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct SingleChanData {
        #[pyo3(get, set)]
        mass: f64,
        #[pyo3(get, set)]
        energy: f64,
        #[pyo3(get, set)]
        l: u32,
        #[pyo3(get, set)]
        potential_scaling: f64,

        #[pyo3(get, set)]
        scattering_config: ScatteringConfig,

        #[pyo3(get, set)]
        potential: NumpyInteraction,
    }

    #[pyclass]
    enum SCPropagator {
        LogDerivative,
        Numerov,
    }

    #[pyclass]
    struct SValue {
        #[pyo3(get)]
        s: (f64, f64),
        #[pyo3(get)]
        a_length: f64,

        #[pyo3(get)]
        cross_sect: f64,

        #[pyo3(get)]
        phase_shift: f64,
    }

    #[pymethods]
    impl SingleChanData {
        #[new]
        fn new(mass: f64, energy: f64, l: u32, scattering_config: ScatteringConfig, potential: NumpyInteraction) -> Self {
            Self {
                mass,
                energy,
                potential_scaling: 1.,
                scattering_config,
                l,
                potential,
            }
        }

        fn scatter(&self, propagator: &SCPropagator) -> SValue {
            let w_function = CollisionWFunction::new(
                Scaled {
                    scaling: self.potential_scaling,
                    interaction: self.potential.clone(),
                },
                self.mass,
                self.energy,
                self.l,
            );

            let step = self.scattering_config.step.0.clone();
            let boundary = Boundary {
                r_start: self.scattering_config.r_start,
                direction: single_chan::cc_propagator::Direction::Outwards,
                value: 1e-50,
                derivative: 1.,
            };

            let s_value = match propagator {
                SCPropagator::Numerov => {
                    let mut numerov = single_chan::numerov::RatioNumerov::new(&w_function, step, boundary);
                    let sol = numerov.propagate_to(self.scattering_config.r_stop);
                    single_chan::s_matrix::SValue::from_ratio(sol, &w_function)
                }
                SCPropagator::LogDerivative => {
                    let mut johnson = single_chan::log_derivative::LogDerivative::new(&w_function, step, boundary);
                    let sol = johnson.propagate_to(self.scattering_config.r_stop);
                    single_chan::s_matrix::SValue::from_log_deriv(sol, &w_function)
                }
            };

            SValue {
                s: (s_value.value.re, s_value.value.im),
                a_length: s_value.scattering_length().re,
                cross_sect: s_value.elastic_cross_sect(),
                phase_shift: s_value.phase_shift(),
            }
        }

        fn scatter_dependence(&self, propagator: &SCPropagator, changes: usize, dependence: Py<PyAny>) -> Vec<SValue> {
            use rayon::prelude::*;

            (0..changes)
                .into_par_iter()
                .map(|i| {
                    let s = self.clone();
                    pyo3::Python::attach(|py| {
                        let s_new: Self = dependence
                            .call1(py, (s, i))
                            .expect("Expected function with signature fn(s: SingleChanData, i: int)) -> SingleChanData")
                            .extract(py)
                            .expect("Expected function with signature fn(s: SingleChanData, i: int)) -> SingleChanData");

                        s_new.scatter(propagator)
                    })
                })
                .collect()
        }
    }

    #[pyclass]
    #[derive(Clone)]
    struct CoupledChanData {
        #[pyo3(get, set)]
        mass: f64,
        #[pyo3(get, set)]
        energy: f64,

        #[pyo3(get, set)]
        scattering_config: ScatteringConfig,

        #[pyo3(get, set)]
        w_matrix: CollisionWMatrix,
    }

    #[pyclass]
    struct SMatrix(coupled_chan::s_matrix::SMatrix);

    #[pymethods]
    impl SMatrix {
        pub fn s_matrix(&self, i: usize, j: usize) -> (f64, f64) {
            let s = self.0.s_matrix()[(i, j)];
            (s.re, s.im)
        }

        pub fn entrance_number(&self) -> usize {
            self.0.entrance_number()
        }

        pub fn entrance_momentum(&self) -> f64 {
            self.0.entrance_momentum()
        }

        pub fn momenta(&self) -> &[f64] {
            self.0.momenta()
        }

        pub fn scattering_length(&self) -> (f64, f64) {
            let s = self.0.scattering_length();

            (s.re, s.im)
        }

        pub fn elastic_cross_sect(&self) -> f64 {
            self.0.elastic_cross_sect()
        }

        pub fn inelastic_cross_sect(&self) -> f64 {
            self.0.inelastic_cross_sect()
        }

        pub fn inelastic_cross_sect_to(&self, channel: usize) -> f64 {
            self.0.inelastic_cross_sect_to(channel)
        }

        pub fn cross_sect(&self, in_chan: usize, out_chan: usize) -> f64 {
            self.0.cross_sect(in_chan, out_chan)
        }

        pub fn scattering_length_in(&self, channel: usize) -> (f64, f64) {
            let s = self.0.scattering_length_in(channel);

            (s.re, s.im)
        }
    }

    #[pyclass]
    enum CCPropagator {
        Johnson,
        Manolopoulos,
        Numerov,
    }

    #[pymethods]
    impl CoupledChanData {
        #[new]
        fn new(mass: f64, energy: f64, scattering_config: ScatteringConfig, w_matrix: CollisionWMatrix) -> Self {
            Self {
                mass,
                energy,
                scattering_config,
                w_matrix,
            }
        }

        fn scatter(&self, propagator: &CCPropagator) -> SMatrix {
            let step = self.scattering_config.step.0.clone();
            let boundary = Boundary {
                r_start: self.scattering_config.r_start,
                direction: Direction::Outwards,
                value: 1e-50 * self.w_matrix.0.id(),
                derivative: 1. * self.w_matrix.0.id(),
            };

            let s = match propagator {
                CCPropagator::Johnson => {
                    let mut prop = JohnsonLogDerivative::new(&self.w_matrix.0, step.clone(), boundary);
                    let sol = prop.propagate_to(self.scattering_config.r_stop);
                    coupled_chan::s_matrix::SMatrix::from_log_deriv(sol, &self.w_matrix.0)
                }
                CCPropagator::Manolopoulos => {
                    let mut prop = ManolopoulosLogDerivative::new(&self.w_matrix.0, step.clone(), boundary);
                    let sol = prop.propagate_to(self.scattering_config.r_stop);
                    coupled_chan::s_matrix::SMatrix::from_log_deriv(sol, &self.w_matrix.0)
                }
                CCPropagator::Numerov => {
                    let mut prop = RatioNumerov::new(&self.w_matrix.0, step.clone(), boundary);
                    let sol = prop.propagate_to(self.scattering_config.r_stop);
                    coupled_chan::s_matrix::SMatrix::from_ratio(sol, &self.w_matrix.0)
                }
            };

            SMatrix(s)
        }

        fn scatter_dependence(&self, propagator: &CCPropagator, changes: usize, dependence: Py<PyAny>) -> Vec<SMatrix> {
            use rayon::prelude::*;

            (0..changes)
                .into_par_iter()
                .map(|i| {
                    let s = self.clone();
                    pyo3::Python::attach(|py| {
                        let s_new: Self = dependence
                            .call1(py, (s, i))
                            .expect("Expected function with signature fn(s: CoupledChanData, i: int)) -> CoupledChanData")
                            .extract(py)
                            .expect("Expected function with signature fn(s: CoupledChanData, i: int)) -> CoupledChanData");

                        s_new.scatter(propagator)
                    })
                })
                .collect()
        }
    }
}
