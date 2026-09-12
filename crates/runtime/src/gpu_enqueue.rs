//! Device enqueue workers whose progress does not depend on Rayon pool size.
//!
//! Each batch transfers its owned device states to long-lived workers and gets
//! every state back before returning. Production callbacks enqueue GPU work;
//! only explicit measurement/preflight callbacks may wait for device completion.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

type Command = Box<dyn FnOnce(bool) + Send>;

#[derive(Debug, thiserror::Error)]
pub enum GpuEnqueueError<E> {
    #[error("GPU worker {device} failed: {error}")]
    Operation { device: usize, error: E },
    #[error("GPU worker {device} panicked")]
    Panicked { device: usize },
    #[error("GPU worker {device} stopped")]
    Stopped { device: usize },
    #[error("GPU enqueue device count mismatch: expected {expected}, received {actual}")]
    DeviceCount { expected: usize, actual: usize },
    #[error("GPU enqueue workers are unavailable after an earlier worker failure")]
    Unavailable,
}

impl From<GpuEnqueueError<crate::backend::poly::PolyBackendError>>
    for crate::backend::poly::PolyBackendError
{
    fn from(error: GpuEnqueueError<Self>) -> Self {
        match error {
            GpuEnqueueError::Operation { error, .. } => error,
            error => Self::GpuSubmission(error.to_string()),
        }
    }
}

enum Outcome<R, E> {
    Complete(R),
    Failed(GpuEnqueueError<E>),
    Cancelled,
}

pub struct GpuEnqueuePool {
    devices: usize,
    senders: Vec<mpsc::SyncSender<Command>>,
    threads: Vec<JoinHandle<()>>,
    healthy: bool,
}

impl GpuEnqueuePool {
    pub fn new(devices: usize) -> std::io::Result<Self> {
        // Fleet constructors validate nonempty physical placement. An empty
        // worker set is also useful for metadata-only estimator setup; mapping
        // it submits no commands and never fabricates a device.
        let mut pool = Self { devices, senders: Vec::new(), threads: Vec::new(), healthy: true };
        // The single-device path stays on the caller, with the same error and
        // ownership contract and no extra host thread or channel hop.
        if devices > 1 {
            for device in 0..devices {
                let (sender, receiver) = mpsc::sync_channel::<Command>(1);
                let worker = thread::Builder::new().name(format!("gpu-enqueue-{device}")).spawn(
                    move || {
                        while let Ok(command) = receiver.recv() {
                            command(true);
                        }
                    },
                )?;
                pool.senders.push(sender);
                pool.threads.push(worker);
            }
        }
        Ok(pool)
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy
    }

    /// Submit one bounded host command per device and collect every reply.
    /// GPU completion is not a condition for a production callback to return.
    /// A failed callback cancels callbacks that have not started; already
    /// submitted device work keeps the callback's ordinary release ownership.
    pub fn map<T, R, E, F>(
        &mut self,
        states: &mut Vec<T>,
        operation: F,
    ) -> Result<Vec<R>, GpuEnqueueError<E>>
    where
        T: Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
        F: Fn(usize, &mut T) -> Result<R, E> + Send + Sync + 'static,
    {
        if !self.healthy {
            return Err(GpuEnqueueError::Unavailable);
        }
        if states.len() != self.devices {
            return Err(GpuEnqueueError::DeviceCount {
                expected: self.devices,
                actual: states.len(),
            });
        }
        if self.devices == 1 {
            return match catch_unwind(AssertUnwindSafe(|| operation(0, &mut states[0]))) {
                Ok(result) => result
                    .map(|value| vec![value])
                    .map_err(|error| GpuEnqueueError::Operation { device: 0, error }),
                Err(_) => {
                    self.healthy = false;
                    Err(GpuEnqueueError::Panicked { device: 0 })
                }
            };
        }
        let operation = Arc::new(operation);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::channel();
        let mut stopped = None;
        for (device, mut state) in std::mem::take(states).into_iter().enumerate() {
            let operation = operation.clone();
            let worker_cancelled = cancelled.clone();
            let sender = sender.clone();
            let command = Box::new(move |run: bool| {
                let outcome = if !run || worker_cancelled.load(Ordering::Acquire) {
                    Outcome::Cancelled
                } else {
                    match catch_unwind(AssertUnwindSafe(|| operation(device, &mut state))) {
                        Ok(Ok(value)) => Outcome::Complete(value),
                        Ok(Err(error)) => {
                            worker_cancelled.store(true, Ordering::Release);
                            Outcome::Failed(GpuEnqueueError::Operation { device, error })
                        }
                        Err(_) => {
                            worker_cancelled.store(true, Ordering::Release);
                            Outcome::Failed(GpuEnqueueError::Panicked { device })
                        }
                    }
                };
                drop(sender.send((device, state, outcome)));
            });
            // At most one batch can borrow this pool. All previous commands
            // have replied, so a queue slot never waits for prior GPU work.
            if let Err(mpsc::SendError(command)) = self.senders[device].send(command) {
                cancelled.store(true, Ordering::Release);
                stopped.get_or_insert(device);
                // The command still owns its device state and any unsubmitted
                // reservation leases. Return them through the ordinary reply
                // path without executing its GPU callback on the caller.
                command(false);
            }
        }
        drop(sender);
        let mut replies = Vec::with_capacity(self.devices);
        if rayon::current_thread_index().is_some() {
            // Blocking the last Rayon worker in recv() would strand nested CPU
            // work submitted by an enqueue callback. Keep helping that pool.
            loop {
                match receiver.try_recv() {
                    Ok(reply) => replies.push(reply),
                    Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {
                        rayon::yield_now();
                        thread::yield_now();
                    }
                }
            }
        } else {
            replies.extend(receiver);
        }
        replies.sort_unstable_by_key(|(device, _, _)| *device);
        let mut failure = stopped.map(|device| GpuEnqueueError::Stopped { device });
        if stopped.is_some() {
            self.healthy = false;
        }
        if replies.len() != self.devices {
            self.healthy = false;
            if failure.is_none() {
                let device = (0..self.devices)
                    .find(|device| !replies.iter().any(|(index, _, _)| index == device))
                    .expect("missing worker");
                failure = Some(GpuEnqueueError::Stopped { device });
            }
        }
        let mut values = Vec::with_capacity(self.devices);
        for (_, state, outcome) in replies {
            states.push(state);
            match outcome {
                Outcome::Complete(value) => values.push(value),
                Outcome::Failed(error) => {
                    if matches!(error, GpuEnqueueError::Panicked { .. }) {
                        self.healthy = false;
                    }
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
                Outcome::Cancelled => {}
            }
        }
        if let Some(error) = failure { Err(error) } else { Ok(values) }
    }
}

impl Drop for GpuEnqueuePool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.threads.drain(..) {
            drop(worker.join());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::{convert::Infallible, sync::Barrier, time::Duration};

    #[test]
    fn test_gpu_enqueue_stopped_worker_returns_owned_state_without_running_callback() {
        let mut pool = GpuEnqueuePool::new(3).unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(receiver);
        pool.senders[1] = sender;
        let mut states = vec![0usize, 10, 20];
        let result = pool.map(&mut states, |_, state| {
            *state += 1;
            Ok::<_, Infallible>(())
        });
        assert!(matches!(result, Err(GpuEnqueueError::Stopped { device: 1 })));
        assert_eq!(states.len(), 3);
        assert!([0, 1].contains(&states[0]));
        assert_eq!(states[1], 10);
        assert_eq!(states[2], 20);
        assert!(!pool.is_healthy());
        assert!(matches!(
            pool.map(&mut states, |_, _| Ok::<_, Infallible>(())),
            Err(GpuEnqueueError::Unavailable)
        ));
    }

    #[test]
    fn test_gpu_enqueue_persistent_workers_make_progress_with_nested_rayon() {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let result = rayon::join(
                || {
                    let mut pool = GpuEnqueuePool::new(3).unwrap();
                    let mut states = vec![0usize; 3];
                    let mut previous = None;
                    for _ in 0..3 {
                        let rendezvous = Arc::new(Barrier::new(3));
                        let threads = pool
                            .map(&mut states, move |_, state| {
                                *state += (0..16usize).into_par_iter().sum::<usize>();
                                // These are dedicated threads, not Rayon jobs. The
                                // rendezvous proves all commands can be active together.
                                rendezvous.wait();
                                Ok::<_, Infallible>(thread::current().id())
                            })
                            .unwrap();
                        if let Some(previous) = previous {
                            assert_eq!(threads, previous);
                        }
                        previous = Some(threads);
                    }
                    assert_eq!(states, vec![360; 3]);
                },
                || (),
            )
            .0;
            sender.send(result).unwrap();
        });
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("GPU enqueue workers stranded Rayon work");
    }

    #[test]
    fn test_gpu_enqueue_failure_returns_all_states_and_keeps_workers_usable() {
        let mut pool = GpuEnqueuePool::new(3).unwrap();
        let mut states = vec![0; 3];
        let result = pool.map(&mut states, |device, state| {
            *state += 1;
            if device == 1 { Err("injected launch failure") } else { Ok(()) }
        });
        assert!(matches!(result, Err(GpuEnqueueError::Operation { device: 1, .. })));
        assert_eq!(states.len(), 3);
        let before = states.clone();
        pool.map(&mut states, |_, state| {
            *state += 1;
            Ok::<_, Infallible>(())
        })
        .unwrap();
        assert_eq!(states, before.into_iter().map(|value| value + 1).collect::<Vec<_>>());
    }

    #[test]
    fn test_gpu_enqueue_panic_is_reported_without_losing_states() {
        let mut pool = GpuEnqueuePool::new(3).unwrap();
        let mut states = vec![0; 3];
        let result = pool.map(&mut states, |device, state| {
            *state += 1;
            if device == 1 {
                panic!("injected worker panic");
            }
            Ok::<_, Infallible>(())
        });
        assert!(matches!(result, Err(GpuEnqueueError::Panicked { device: 1 })));
        assert_eq!(states.len(), 3);
        assert!(!pool.is_healthy());
        assert!(matches!(
            pool.map(&mut states, |_, _| Ok::<_, Infallible>(())),
            Err(GpuEnqueueError::Unavailable)
        ));
    }

    #[test]
    fn test_gpu_enqueue_single_device_preserves_caller_thread() {
        let mut pool = GpuEnqueuePool::new(1).unwrap();
        let result =
            pool.map(&mut vec![()], |_, _| Ok::<_, Infallible>(thread::current().id())).unwrap();
        assert_eq!(result, [thread::current().id()]);
    }
}
