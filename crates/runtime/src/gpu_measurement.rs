//! Opt-in timing of the compiled column runner. Ordinary execution does not
//! create timing events or wait at these boundaries.

use crate::{
    gpu_memory::{GpuAdmissionError, GpuAdmittedPlanSummary},
    gpu_schedule::GpuColumnJob,
};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{GpuPreparedOccupancyMode, GpuPreparedStorage},
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, GpuDeviceTiming},
    },
};
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc, time::Instant};

/// Streaming observations: callers may aggregate wave classes without retaining
/// a record for every wave. Records never retain GPU input or output owners.
#[derive(Debug)]
pub enum GpuAdmittedMeasurement {
    /// IR dispatch provenance. Several compiled invocations can belong to this
    /// one node selection; `instances` is its batch of independent IR instances.
    Node(GpuMeasurementNode),
    /// Save the parent's observation while a child execution scope runs.
    EnterScope,
    /// Restore the parent before staging or consuming the child's outputs.
    ExitScope,
    /// Native operations removed by the actual executor optimization plan.
    OmittedNodes { scope: mxx_ir_core::FrozenGraphScopeId, nodes: Vec<mxx_ir_core::types::NodeId> },
    /// Actual normalization and redistribution, excluding isolated calibration
    /// pilots. A batch may share this preparation across several invocations.
    InputPreparation { operation: Option<[u8; 32]>, timing: GpuStageTiming },
    Invocation {
        operation: Option<[u8; 32]>,
        plan: GpuAdmittedPlanSummary,
        /// Host materialization has a separate dataflow timing owner.
        host_import: bool,
        /// Process-local concrete input and prepared-layout identity.
        scenario: [u8; 32],
    },
    /// Output initialization is reported separately from preparation and ranges.
    OutputInitialization(GpuStageTiming),
    Wave {
        jobs: Vec<GpuColumnJob>,
        timing: GpuStageTiming,
        /// False reuses the first measured wave of the same interval/width class.
        measured: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuMeasurementNode {
    pub scope: mxx_ir_core::FrozenGraphScopeId,
    pub node: mxx_ir_core::types::NodeId,
    pub instances: usize,
    /// Bindings of the representative member in this calibration-equivalent
    /// selection. Shared by its compiled invocations, only when observing.
    pub bindings: Arc<mxx_ir_core::ParamEnv>,
}

#[derive(Clone, Debug)]
pub struct GpuStageTiming {
    /// Physical device ID and its CUDA-event timeline span, in seconds.
    pub device_elapsed_seconds: Vec<(i32, f64)>,
    /// Coordinated submission through completion of every active device.
    pub fleet_wall_seconds: f64,
    /// Actual joint prepared device-span observations. Retained prepared inputs
    /// and full outputs are included in the baseline, not reported as scratch.
    /// External inputs, pinned host storage and opaque CUDA bytes are excluded.
    pub prepared_memory: Vec<GpuPreparedStageMemory>,
}

#[derive(Clone, Debug)]
pub struct GpuPreparedStageMemory {
    pub device_id: i32,
    pub baseline_bytes: usize,
    pub peak_bytes: usize,
}

pub type GpuAdmittedMeasurementSink = Box<dyn FnMut(GpuAdmittedMeasurement) + Send>;

/// A measurement-only boundary used by `GpuColumnMemoryPlan::execute`.
/// The caller owns exclusive benchmark access to the participating contexts.
pub struct GpuColumnMeasurement<'a> {
    parameters: BTreeMap<usize, GpuDCRTPolyParams>,
    stores: BTreeMap<usize, Vec<Arc<GpuPreparedStorage>>>,
    sink: &'a mut GpuAdmittedMeasurementSink,
    offset_sensitive: bool,
    wave_timings: BTreeMap<Vec<(usize, usize, usize)>, GpuStageTiming>,
}

pub(crate) struct GpuStageTimer {
    timings: Vec<GpuDeviceTiming>,
    start: Instant,
    baselines: Vec<(usize, usize)>,
}

pub(crate) enum GpuMeasuredStage {
    InputPreparation(Option<[u8; 32]>),
    OutputInitialization,
    Wave(Vec<GpuColumnJob>),
}

impl<'a> GpuColumnMeasurement<'a> {
    pub(crate) fn new(
        parameters: BTreeMap<usize, GpuDCRTPolyParams>,
        stores: BTreeMap<usize, Vec<Arc<GpuPreparedStorage>>>,
        sink: &'a mut GpuAdmittedMeasurementSink,
        invocation: Option<(Option<[u8; 32]>, GpuAdmittedPlanSummary, bool, [u8; 32])>,
        offset_sensitive: bool,
    ) -> Self {
        if let Some((operation, plan, host_import, scenario)) = invocation {
            sink(GpuAdmittedMeasurement::Invocation { operation, plan, host_import, scenario });
        }
        Self { parameters, stores, sink, offset_sensitive, wave_timings: BTreeMap::new() }
    }

    fn wave_class(jobs: &[GpuColumnJob]) -> Vec<(usize, usize, usize)> {
        // A compiled source interval fixes all source owners, contexts, levels,
        // formats, preparation mappings and destination geometry for its jobs.
        jobs.iter().map(|job| (job.device, job.source_interval, job.end - job.start)).collect()
    }

    pub(crate) fn begin_wave(
        &self,
        jobs: &[GpuColumnJob],
    ) -> Result<Option<GpuStageTimer>, GpuAdmissionError> {
        if !self.offset_sensitive && self.wave_timings.contains_key(&Self::wave_class(jobs)) {
            Ok(None)
        } else {
            self.begin(Some(jobs)).map(Some)
        }
    }

    pub(crate) fn reuse_wave(&mut self, jobs: Vec<GpuColumnJob>) {
        let timing = self.wave_timings[&Self::wave_class(&jobs)].clone();
        (self.sink)(GpuAdmittedMeasurement::Wave { jobs, timing, measured: false });
    }

    pub(crate) fn begin(
        &self,
        jobs: Option<&[GpuColumnJob]>,
    ) -> Result<GpuStageTimer, GpuAdmissionError> {
        let parameters = self
            .parameters
            .iter()
            .filter(|(device, _)| {
                jobs.is_none_or(|jobs| jobs.iter().any(|job| job.device == **device))
            })
            .collect::<Vec<_>>();
        let stores_by_device = &self.stores;
        // Memory baselines include retained owners on idle devices too; their
        // CUDA work remains absent from this wave's timing vector.
        let all_parameters = self.parameters.iter().collect::<Vec<_>>();
        let baselines = all_parameters
            .par_iter()
            .map(|(device, parameters)| {
                // Earlier waves may have run without timing waits. Complete
                // this owner's preceding compute outside the next sample's
                // wall timer, using benchmark-owned events (not device sync).
                parameters.begin_device_timing()?.finish()?;
                // This is an explicit measurement boundary, outside the timed
                // submission. Reconcile warmup/preceding-wave releases while
                // the invocation's reservations are parked on the coordinator.
                parameters.fence_released_memory();
                let stores = stores_by_device[device].iter().map(Arc::as_ref).collect::<Vec<_>>();
                let (baseline, _) = GpuPreparedStorage::joint_occupancy(
                    &stores,
                    GpuPreparedOccupancyMode::ResetMeasurement,
                )?;
                Ok((**device, baseline))
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(GpuAdmissionError::NativeReservation)?;
        let timings = parameters
            .par_iter()
            .map(|(_, parameters)| parameters.begin_device_timing())
            .collect::<Result<Vec<_>, _>>()
            .map_err(GpuAdmissionError::NativeReservation)?;
        Ok(GpuStageTimer { timings, start: Instant::now(), baselines })
    }

    pub(crate) fn finish(
        &mut self,
        mut timer: GpuStageTimer,
        stage: GpuMeasuredStage,
    ) -> Result<(), GpuAdmissionError> {
        // Stop every device before waiting for any result. Full output owners
        // remain in the runner across this boundary and all subsequent waves.
        timer
            .timings
            .par_iter_mut()
            .map(GpuDeviceTiming::stop)
            .collect::<Result<Vec<_>, _>>()
            .map_err(GpuAdmissionError::NativeReservation)?;
        let spans = timer
            .timings
            .into_par_iter()
            .map(GpuDeviceTiming::finish)
            .collect::<Result<Vec<_>, _>>()
            .map_err(GpuAdmissionError::NativeReservation)?;
        let fleet_wall_seconds = timer.start.elapsed().as_secs_f64();
        let stores_by_device = &self.stores;
        let parameters_by_device = &self.parameters;
        let prepared_memory = timer
            .baselines
            .into_par_iter()
            .map(|(device, baseline_bytes)| {
                let stores = stores_by_device[&device].iter().map(Arc::as_ref).collect::<Vec<_>>();
                let (_, peak_bytes) = GpuPreparedStorage::joint_occupancy(
                    &stores,
                    GpuPreparedOccupancyMode::Observe,
                )?;
                Ok(GpuPreparedStageMemory {
                    device_id: parameters_by_device[&device].device_ids()[0],
                    baseline_bytes,
                    peak_bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(GpuAdmissionError::NativeReservation)?;
        let timing = GpuStageTiming {
            device_elapsed_seconds: spans.into_iter().flatten().collect(),
            fleet_wall_seconds,
            prepared_memory,
        };
        (self.sink)(match stage {
            GpuMeasuredStage::InputPreparation(operation) => {
                GpuAdmittedMeasurement::InputPreparation { operation, timing }
            }
            GpuMeasuredStage::Wave(jobs) => {
                if !self.offset_sensitive {
                    self.wave_timings.insert(Self::wave_class(&jobs), timing.clone());
                }
                GpuAdmittedMeasurement::Wave { jobs, timing, measured: true }
            }
            GpuMeasuredStage::OutputInitialization => {
                GpuAdmittedMeasurement::OutputInitialization(timing)
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExecutionConfig, MemoryArtifactStore, RuntimeValue, backend::poly_gpu::gpu_backend_on,
        execute_with_config, transcript::SamplingMode,
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::ParamEnv;
    use mxx_primitives::{
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
        },
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_compiled_measurement_retains_outputs_and_records_actual_ranges() {
        for normalize in [false, true] {
            check_compiled_measurement(normalize);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_measurement_scenario_distinguishes_normalization_without_allocation_ids() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let parameters = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(parameters.modulus().as_ref().clone(), n as usize);
        let input = ring.input("input", (2, columns));
        let graph = DslContext::new("measurement-layout-identity")
            .output("result", input.clone() + input)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let value =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let expected = value.clone() + value.clone();
        let mut backend = gpu_backend_on([parameters.clone()], [device]);
        let mut scenarios = Vec::new();
        for coefficient in [false, true, false] {
            let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&parameters, &value);
            if coefficient {
                matrix.intt_all_in_place();
            }
            let (sender, receiver) = std::sync::mpsc::channel();
            backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
                sender.send(event).unwrap();
            })));
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                BTreeMap::from([("input".into(), RuntimeValue::matrix(matrix.into()))]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            backend.set_admitted_measurement_sink(None);
            let mut observed = receiver.into_iter().filter_map(|event| match event {
                GpuAdmittedMeasurement::Invocation { scenario, host_import: false, .. } => {
                    Some(scenario)
                }
                _ => None,
            });
            scenarios.push(observed.next().expect("one add invocation"));
            assert!(observed.next().is_none());
            let RuntimeValue::Matrix(output) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix result");
            };
            assert_eq!(output.shards()[0].value.to_cpu_matrix(), expected);
        }
        assert_eq!(scenarios[0], scenarios[2], "equivalent fresh owners share a layout identity");
        assert_ne!(scenarios[0], scenarios[1], "coefficient normalization changes the scenario");
    }

    fn check_compiled_measurement(normalize: bool) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let parameters = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(parameters.modulus().as_ref().clone(), n as usize);
        let input = ring.input("input", (2, columns));
        let graph = DslContext::new("compiled-measurement")
            .output("result", if normalize { input.clone() + input } else { -input })
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let value =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let expected =
            if normalize { value.clone() + value.clone() } else { value.negate_out_of_place() };
        let mut backend = gpu_backend_on([parameters.clone()], [device]);
        let (sender, receiver) = std::sync::mpsc::channel();
        backend.set_admitted_measurement_sink(Some(Box::new(move |record| {
            sender.send(record).unwrap();
        })));
        let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&parameters, &value);
        if normalize {
            matrix.intt_all_in_place();
        }
        let inputs = BTreeMap::from([("input".into(), RuntimeValue::matrix(matrix.into()))]);
        let mut store = MemoryArtifactStore::default();
        let mut result = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        backend.set_admitted_measurement_sink(None);
        let observations = receiver
            .into_iter()
            .filter(|event| {
                !matches!(
                    event,
                    GpuAdmittedMeasurement::Node(_) |
                        GpuAdmittedMeasurement::OmittedNodes { .. } |
                        GpuAdmittedMeasurement::EnterScope |
                        GpuAdmittedMeasurement::ExitScope
                )
            })
            .collect::<Vec<_>>();
        let (preparation, records) = if normalize {
            let GpuAdmittedMeasurement::InputPreparation { operation, timing } = &observations[0]
            else {
                panic!("coefficient input normalization must be measured before compilation")
            };
            assert!(operation.is_some());
            assert_eq!(timing.device_elapsed_seconds.len(), 1);
            assert!(timing.device_elapsed_seconds[0].1 > 0.0);
            assert!(timing.fleet_wall_seconds > 0.0);
            assert_eq!(timing.prepared_memory.len(), 1);
            assert!(
                timing.prepared_memory[0].peak_bytes > timing.prepared_memory[0].baseline_bytes
            );
            (Some(operation), &observations[1..])
        } else {
            (None, observations.as_slice())
        };
        let GpuAdmittedMeasurement::Invocation { operation, plan, .. } = &records[0] else {
            panic!("invocation metadata must precede timing records")
        };
        assert!(operation.is_some());
        if let Some(preparation) = preparation {
            assert_eq!(preparation, operation);
        }
        assert_eq!(plan.columns, columns);
        let GpuAdmittedMeasurement::OutputInitialization(initialization) = &records[1] else {
            panic!("output initialization must precede waves")
        };
        assert_eq!(initialization.prepared_memory.len(), 1);
        let initialized = &initialization.prepared_memory[0];
        assert!(
            initialized.peak_bytes > initialized.baseline_bytes,
            "initialization adds the complete output above retained preparation owners"
        );
        assert_eq!(records.len(), plan.wave_count + 2);
        for (record, expected_jobs) in records[2..].iter().zip(plan.schedule.waves()) {
            let GpuAdmittedMeasurement::Wave { jobs, timing, .. } = record else {
                panic!("expected actual wave timing")
            };
            assert_eq!(*jobs, expected_jobs);
            assert_eq!(timing.device_elapsed_seconds.len(), 1);
            assert_eq!(timing.device_elapsed_seconds[0].0, device);
            assert!(timing.device_elapsed_seconds[0].1 > 0.0);
            assert!(timing.fleet_wall_seconds > 0.0);
            assert_eq!(timing.prepared_memory.len(), 1);
            let memory = &timing.prepared_memory[0];
            assert_eq!(memory.device_id, device);
            assert_eq!(memory.baseline_bytes, initialized.peak_bytes);
            assert_eq!(
                memory.peak_bytes, memory.baseline_bytes,
                "addition and negation write retained output directly without range scratch"
            );
        }
        let RuntimeValue::Matrix(output) =
            result.materialize_output("result", &mut backend, &mut store).unwrap()
        else {
            panic!("matrix result")
        };
        // Reading after detaching the observer checks that measurement does not
        // retire the production output's native storage or event dependencies.
        assert_eq!(output.shards().len(), 1);
        assert_eq!(output.shards()[0].value.to_cpu_matrix(), expected);
    }
}
