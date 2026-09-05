use crate::NodeMeasurement;
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
pub struct MeasurementHarnessConfig {
    pub warm_up_iterations: usize,
    pub measured_iterations: usize,
    pub memory_poll_interval: Duration,
}

impl Default for MeasurementHarnessConfig {
    fn default() -> Self {
        Self {
            warm_up_iterations: 2,
            measured_iterations: 5,
            memory_poll_interval: Duration::from_millis(1),
        }
    }
}

pub trait MemoryProbe: Sync {
    type Error: std::error::Error + Send;

    fn current_bytes(&self) -> Result<u64, Self::Error>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchMeasurement {
    pub batch_size: usize,
    pub measurement: NodeMeasurement,
}

#[derive(Debug, Error)]
pub enum MeasurementHarnessError<E: std::error::Error> {
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
pub fn measure_batch_operation<P, F, R>(
    config: &MeasurementHarnessConfig,
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
pub fn measure_operation<P, F, R>(
    config: &MeasurementHarnessConfig,
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
    let elapsed = thread::scope(|scope| {
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
        let started = Instant::now();
        for _ in 0..config.measured_iterations {
            black_box(operation());
        }
        stop.store(true, Ordering::Release);
        started.elapsed().as_secs_f64()
    });
    if let Some(error) = probe_error.lock().expect("memory probe error lock poisoned").take() {
        return Err(MeasurementHarnessError::MemoryProbe(error));
    }
    let seconds = elapsed / config.measured_iterations as f64;
    Ok(NodeMeasurement {
        work_seconds: seconds,
        latency_seconds: seconds,
        workspace_bytes: peak.load(Ordering::Acquire).saturating_sub(baseline),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicUsize, Ordering},
    };

    struct Probe {
        bytes: AtomicU64,
        samples: AtomicUsize,
    }

    impl MemoryProbe for Probe {
        type Error = Infallible;

        fn current_bytes(&self) -> Result<u64, Self::Error> {
            self.samples.fetch_add(1, Ordering::AcqRel);
            Ok(self.bytes.load(Ordering::Acquire))
        }
    }

    #[test]
    fn harness_warms_up_and_averages_the_requested_measurements() {
        let probe = Probe { bytes: AtomicU64::new(0), samples: AtomicUsize::new(0) };
        let calls = AtomicUsize::new(0);
        let measurement = measure_operation(
            &MeasurementHarnessConfig {
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
    }

    #[test]
    fn harness_observes_transient_memory_during_the_operation() {
        let probe = Probe { bytes: AtomicU64::new(16), samples: AtomicUsize::new(0) };
        let measurement = measure_operation(
            &MeasurementHarnessConfig {
                warm_up_iterations: 0,
                measured_iterations: 1,
                memory_poll_interval: Duration::ZERO,
            },
            &probe,
            || {
                probe.bytes.store(80, Ordering::Release);
                let samples = probe.samples.load(Ordering::Acquire);
                while probe.samples.load(Ordering::Acquire) == samples {
                    thread::yield_now();
                }
                probe.bytes.store(16, Ordering::Release);
            },
        )
        .expect("measurement");
        assert_eq!(measurement.workspace_bytes, 64);
    }

    #[test]
    fn harness_rejects_an_empty_measurement() {
        let probe = Probe { bytes: AtomicU64::new(0), samples: AtomicUsize::new(0) };
        assert!(matches!(
            measure_operation(
                &MeasurementHarnessConfig {
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
        let probe = Probe { bytes: AtomicU64::new(0), samples: AtomicUsize::new(0) };
        let observed = Mutex::new(Vec::new());
        let measurement = measure_batch_operation(
            &MeasurementHarnessConfig {
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
            measure_batch_operation(&MeasurementHarnessConfig::default(), &probe, 0, |_| (),),
            Err(MeasurementHarnessError::EmptyBatch)
        ));
    }
}
