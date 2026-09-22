use super::NodeMeasurement;
use std::{
    hint::black_box,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuWarmupMeasurementConfig {
    pub warm_up_iterations: usize,
    pub measured_iterations: usize,
    pub memory_poll_interval: Duration,
}

impl Default for GpuWarmupMeasurementConfig {
    fn default() -> Self {
        Self {
            warm_up_iterations: 1,
            measured_iterations: 2,
            memory_poll_interval: Duration::from_millis(1),
        }
    }
}

impl GpuWarmupMeasurementConfig {
    /// Build the setup harness from the runtime's already-frozen options.
    /// This keeps environment parsing at construction/preparation time and
    /// avoids a second parser or a late environment lookup in measurement.
    pub fn from_runtime_options(options: &crate::env::GpuRuntimeOptions) -> Self {
        Self {
            warm_up_iterations: options.measurement_warmups,
            measured_iterations: options.measurement_iterations.get(),
            ..Self::default()
        }
    }
}

pub(super) trait MemoryProbe: Sync {
    type Error: std::error::Error + Send;

    fn current_bytes(&self) -> Result<u64, Self::Error>;
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct BatchMeasurement {
    pub(super) batch_size: usize,
    pub(super) measurement: NodeMeasurement,
}

#[derive(Debug, Error)]
pub(super) enum MeasurementHarnessError<E: std::error::Error> {
    #[error("measured iteration count must be positive")]
    EmptyMeasurement,
    #[error("batch size must be positive")]
    EmptyBatch,
    #[error("memory probe failed: {0}")]
    MemoryProbe(E),
}

/// Measures the production batch entry point itself. The callback receives the
/// complete representative batch size on every warm-up and measured
/// invocation; no single-item timing is extrapolated.
pub(super) fn measure_batch_operation<P, F, R>(
    config: &GpuWarmupMeasurementConfig,
    probe: &P,
    batch_size: usize,
    mut operation: F,
) -> Result<BatchMeasurement, MeasurementHarnessError<P::Error>>
where
    P: MemoryProbe,
    F: FnMut(usize) -> R,
{
    if batch_size == 0 {
        return Err(MeasurementHarnessError::EmptyBatch);
    }
    let measurement = measure_operation(config, probe, || operation(batch_size))?;
    Ok(BatchMeasurement { batch_size, measurement })
}

/// Measures one production operation after warm-up while polling its transient
/// memory high-water mark. GPU callers must include their ordinary per-stream
/// completion fence in `operation`; this harness never performs a device-wide
/// synchronization.
pub(super) fn measure_operation<P, F, R>(
    config: &GpuWarmupMeasurementConfig,
    probe: &P,
    mut operation: F,
) -> Result<NodeMeasurement, MeasurementHarnessError<P::Error>>
where
    P: MemoryProbe,
    F: FnMut() -> R,
{
    if config.measured_iterations == 0 {
        return Err(MeasurementHarnessError::EmptyMeasurement);
    }
    for _ in 0..config.warm_up_iterations {
        black_box(operation());
    }

    let baseline = probe.current_bytes().map_err(MeasurementHarnessError::MemoryProbe)?;
    let peak = AtomicU64::new(baseline);
    let stop = AtomicBool::new(false);
    let probe_error = Mutex::new(None);
    let (elapsed, retained_delta, spread) = thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                match probe.current_bytes() {
                    Ok(bytes) => {
                        peak.fetch_max(bytes, Ordering::AcqRel);
                    }
                    Err(error) => {
                        *probe_error.lock().expect("memory probe error lock poisoned") =
                            Some(error);
                        stop.store(true, Ordering::Release);
                        break;
                    }
                }
                if config.memory_poll_interval.is_zero() {
                    thread::yield_now();
                } else {
                    thread::sleep(config.memory_poll_interval);
                }
            }
        });
        let mut retained_output = None;
        let mut retained_after = baseline;
        let mut sample_seconds = Vec::with_capacity(config.measured_iterations);
        for _ in 0..config.measured_iterations {
            // Keep the final returned value alive while sampling U1. This is
            // what distinguishes retained output/cache residency from the
            // transient high-water observation.
            let started = Instant::now();
            let output = operation();
            retained_output = Some(black_box(output));
            let after = probe.current_bytes().map_err(MeasurementHarnessError::MemoryProbe);
            match after {
                Ok(bytes) => {
                    retained_after = bytes;
                    peak.fetch_max(bytes, Ordering::AcqRel);
                }
                Err(error) => {
                    stop.store(true, Ordering::Release);
                    return Err(error);
                }
            }
            sample_seconds.push(started.elapsed().as_secs_f64());
        }
        stop.store(true, Ordering::Release);
        let elapsed = sample_seconds.iter().sum::<f64>();
        let mean = elapsed / sample_seconds.len() as f64;
        let spread = if sample_seconds.len() > 1 {
            let variance = sample_seconds
                .iter()
                .map(|sample| (sample - mean) * (sample - mean))
                .sum::<f64>() /
                sample_seconds.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let retained_delta = retained_after.saturating_sub(baseline);
        // Keep the last output alive until U1 has been sampled, then release
        // it before returning so the next candidate starts from a clean
        // allocator state. The timing summary is per invocation, not a
        // batch-wide wall-clock interval.
        drop(retained_output);
        Ok((elapsed, retained_delta, spread))
    })?;
    if let Some(error) = probe_error.lock().expect("memory probe error lock poisoned").take() {
        return Err(MeasurementHarnessError::MemoryProbe(error));
    }
    let seconds = elapsed / config.measured_iterations as f64;
    Ok(NodeMeasurement {
        work_seconds: seconds,
        latency_seconds: seconds,
        cumulative_wave_seconds: seconds,
        independent_wave_count: 1,
        measured_wave_workspace_bytes: peak.load(Ordering::Acquire).saturating_sub(baseline),
        workspace_bytes: peak.load(Ordering::Acquire).saturating_sub(baseline),
        retained_delta_bytes: retained_delta,
        spread_seconds: spread,
        graph_pool_workspace_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct Probe {
        bytes: AtomicU64,
        samples: AtomicUsize,
        observed_high: AtomicBool,
    }

    #[test]
    fn default_measurement_count_is_one_warmup_and_two_samples() {
        let config = GpuWarmupMeasurementConfig::default();
        assert_eq!(config.warm_up_iterations, 1);
        assert_eq!(config.measured_iterations, 2);
    }

    impl MemoryProbe for Probe {
        type Error = Infallible;

        fn current_bytes(&self) -> Result<u64, Self::Error> {
            self.samples.fetch_add(1, Ordering::AcqRel);
            let bytes = self.bytes.load(Ordering::Acquire);
            if bytes >= 80 {
                self.observed_high.store(true, Ordering::Release);
            }
            Ok(bytes)
        }
    }

    #[test]
    fn harness_warms_up_and_averages_the_requested_measurements() {
        let probe = Probe {
            bytes: AtomicU64::new(0),
            samples: AtomicUsize::new(0),
            observed_high: AtomicBool::new(false),
        };
        let calls = AtomicUsize::new(0);
        let measurement = measure_operation(
            &GpuWarmupMeasurementConfig {
                warm_up_iterations: 2,
                measured_iterations: 3,
                memory_poll_interval: Duration::ZERO,
            },
            &probe,
            || calls.fetch_add(1, Ordering::AcqRel),
        )
        .expect("measurement");
        assert_eq!(calls.load(Ordering::Acquire), 5);
        assert!(measurement.work_seconds >= 0.0);
        assert_eq!(measurement.work_seconds, measurement.latency_seconds);
        assert!(measurement.spread_seconds >= 0.0);
    }

    #[test]
    fn harness_observes_transient_memory_during_the_operation() {
        let probe = Probe {
            bytes: AtomicU64::new(16),
            samples: AtomicUsize::new(0),
            observed_high: AtomicBool::new(false),
        };
        let measurement = measure_operation(
            &GpuWarmupMeasurementConfig {
                warm_up_iterations: 0,
                measured_iterations: 1,
                memory_poll_interval: Duration::ZERO,
            },
            &probe,
            || {
                probe.bytes.store(80, Ordering::Release);
                // Wait until the polling thread has observed the elevated allocation before
                // releasing it.  Synchronizing only with a prior sample permits the operation
                // to finish before the poller ever sees the transient peak.
                while !probe.observed_high.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                probe.bytes.store(16, Ordering::Release);
            },
        )
        .expect("measurement");
        assert_eq!(measurement.workspace_bytes, 64);
        assert_eq!(measurement.retained_delta_bytes, 0);
    }

    #[test]
    fn harness_reports_output_retained_at_u1_separately_from_peak() {
        let probe = Probe {
            bytes: AtomicU64::new(16),
            samples: AtomicUsize::new(0),
            observed_high: AtomicBool::new(false),
        };
        let measurement = measure_operation(
            &GpuWarmupMeasurementConfig {
                warm_up_iterations: 0,
                measured_iterations: 1,
                memory_poll_interval: Duration::ZERO,
            },
            &probe,
            || {
                // The returned value represents a retained output owner. It
                // remains live through the explicit U1 probe below.
                probe.bytes.store(48, Ordering::Release);
                7u64
            },
        )
        .expect("measurement");
        assert_eq!(measurement.workspace_bytes, 32);
        assert_eq!(measurement.retained_delta_bytes, 32);
        assert!(measurement.spread_seconds >= 0.0);
    }

    #[test]
    fn harness_rejects_an_empty_measurement() {
        let probe = Probe {
            bytes: AtomicU64::new(0),
            samples: AtomicUsize::new(0),
            observed_high: AtomicBool::new(false),
        };
        assert!(matches!(
            measure_operation(
                &GpuWarmupMeasurementConfig {
                    warm_up_iterations: 0,
                    measured_iterations: 0,
                    memory_poll_interval: Duration::ZERO,
                },
                &probe,
                || (),
            ),
            Err(MeasurementHarnessError::EmptyMeasurement)
        ));
    }

    #[test]
    fn batch_harness_invokes_the_complete_representative_batch() {
        let probe = Probe {
            bytes: AtomicU64::new(0),
            samples: AtomicUsize::new(0),
            observed_high: AtomicBool::new(false),
        };
        let observed = Mutex::new(Vec::new());
        let measurement = measure_batch_operation(
            &GpuWarmupMeasurementConfig {
                warm_up_iterations: 1,
                measured_iterations: 2,
                memory_poll_interval: Duration::ZERO,
            },
            &probe,
            7,
            |batch_size| observed.lock().expect("observed batch lock").push(batch_size),
        )
        .expect("batch measurement");
        assert_eq!(measurement.batch_size, 7);
        assert_eq!(*observed.lock().expect("observed batch lock"), vec![7, 7, 7]);
        assert!(matches!(
            measure_batch_operation(&GpuWarmupMeasurementConfig::default(), &probe, 0, |_| (),),
            Err(MeasurementHarnessError::EmptyBatch)
        ));
    }
}
