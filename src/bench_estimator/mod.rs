pub mod bgg_poly_encoding;
pub mod bgg_pubkey;
#[cfg(feature = "gpu")]
mod gpu;

pub use bgg_poly_encoding::*;
pub use bgg_pubkey::*;
#[cfg(feature = "gpu")]
pub(crate) use gpu::benchmark_gate_operation;
#[cfg(feature = "gpu")]
pub use gpu::*;

use std::{
    collections::HashMap,
    hint::black_box,
    sync::{Arc, RwLock},
    time::Instant,
};
#[cfg(any(test, feature = "gpu"))]
use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use num_bigint::BigUint;
use rayon::prelude::*;
use tracing::debug;

use crate::{
    circuit::{
        Evaluable, GroupedCallExecutionLayer, GroupedExecutionPlan, PolyCircuit, PolyGateType,
        SubCircuitParamValue,
    },
    poly::Poly,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BenchSummaryCacheKey {
    circuit_ptr: usize,
}

type BenchSummaryCache = RwLock<HashMap<BenchSummaryCacheKey, CircuitBenchSummary>>;

#[derive(Clone)]
struct PreparedBenchSummaryRequest<P: Poly> {
    sub_circuit: Arc<PolyCircuit<P>>,
    param_bindings: Vec<SubCircuitParamValue>,
    cache_key: BenchSummaryCacheKey,
}

#[derive(Debug, Default)]
struct CachedSummaryAggregate {
    total_time: f64,
    max_latency: f64,
    total_parallelism: u128,
    #[cfg(feature = "gpu")]
    peak_vram: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct BenchOperationMeasurement {
    pub time: f64,
    pub peak_vram: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CircuitBenchEstimate {
    pub total_time: f64,
    pub latency: f64,
    pub max_parallelism: u128,
    #[cfg(feature = "gpu")]
    pub peak_vram: usize,
}

impl CircuitBenchEstimate {
    pub(crate) fn new(
        total_time: f64,
        latency: f64,
        max_parallelism: u128,
        peak_vram: usize,
    ) -> Self {
        #[cfg(not(feature = "gpu"))]
        let _ = peak_vram;
        Self {
            total_time,
            latency,
            max_parallelism,
            #[cfg(feature = "gpu")]
            peak_vram,
        }
    }

    fn parallelism_factor(&self) -> u128 {
        if self.max_parallelism > 0 {
            return self.max_parallelism;
        }
        if self.total_time <= 0.0 || self.latency <= 0.0 {
            return 0;
        }
        (self.total_time / self.latency).ceil() as u128
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CircuitBenchSummary {
    pub total_time: f64,
    pub latency: f64,
    pub max_parallelism: u128,
    #[cfg(feature = "gpu")]
    pub peak_vram: usize,
}

impl CircuitBenchSummary {
    pub(crate) fn new(
        total_time: f64,
        latency: f64,
        max_parallelism: u128,
        peak_vram: usize,
    ) -> Self {
        #[cfg(not(feature = "gpu"))]
        let _ = peak_vram;
        Self {
            total_time,
            latency,
            max_parallelism,
            #[cfg(feature = "gpu")]
            peak_vram,
        }
    }
}

fn bench_summary_cache_get(
    summary_cache: &BenchSummaryCache,
    cache_key: &BenchSummaryCacheKey,
) -> Option<CircuitBenchSummary> {
    summary_cache.read().expect("bench summary cache read lock poisoned").get(cache_key).copied()
}

fn bench_summary_cache_contains(
    summary_cache: &BenchSummaryCache,
    cache_key: &BenchSummaryCacheKey,
) -> bool {
    summary_cache.read().expect("bench summary cache read lock poisoned").contains_key(cache_key)
}

fn bench_summary_cache_insert(
    summary_cache: &BenchSummaryCache,
    cache_key: BenchSummaryCacheKey,
    summary: CircuitBenchSummary,
) {
    summary_cache
        .write()
        .expect("bench summary cache write lock poisoned")
        .insert(cache_key, summary);
}

fn bench_summary_cache_len(summary_cache: &BenchSummaryCache) -> usize {
    summary_cache.read().expect("bench summary cache read lock poisoned").len()
}

fn prepared_bench_summary_request<P: Poly>(
    _sub_circuit_id: usize,
    sub_circuit: Arc<PolyCircuit<P>>,
    param_bindings: Vec<SubCircuitParamValue>,
) -> PreparedBenchSummaryRequest<P> {
    PreparedBenchSummaryRequest {
        cache_key: BenchSummaryCacheKey { circuit_ptr: Arc::as_ptr(&sub_circuit) as usize },
        sub_circuit,
        param_bindings,
    }
}

fn aggregate_cached_sub_summaries<P: Poly>(
    summary_cache: &BenchSummaryCache,
    requests: &[PreparedBenchSummaryRequest<P>],
) -> CachedSummaryAggregate {
    requests
        .par_iter()
        .map(|request| {
            let summary = bench_summary_cache_get(summary_cache, &request.cache_key)
                .expect("bench summary cache missing warmed subcircuit summary");
            CachedSummaryAggregate {
                total_time: summary.total_time,
                max_latency: summary.latency,
                total_parallelism: summary.max_parallelism,
                #[cfg(feature = "gpu")]
                peak_vram: summary.peak_vram,
            }
        })
        .reduce(CachedSummaryAggregate::default, |mut left, right| {
            left.total_time += right.total_time;
            left.max_latency = left.max_latency.max(right.max_latency);
            left.total_parallelism = left
                .total_parallelism
                .checked_add(right.total_parallelism)
                .expect("layer parallelism overflowed u128 while summing subcircuit requests");
            #[cfg(feature = "gpu")]
            {
                left.peak_vram = left.peak_vram.max(right.peak_vram);
            }
            left
        })
}

#[derive(Debug, Default)]
struct RegularGateAggregate {
    total_time: f64,
    max_latency: f64,
    total_parallelism: u128,
    #[cfg(feature = "gpu")]
    peak_vram: usize,
}

pub(crate) fn measure_bench_operation<R, F>(iterations: usize, mut op: F) -> f64
where
    F: FnMut() -> R,
{
    let iterations = iterations.max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(op());
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn benchmark_gate_operation<R, F>(iterations: usize, op: F) -> BenchOperationMeasurement
where
    F: FnMut() -> R,
{
    BenchOperationMeasurement { time: measure_bench_operation(iterations, op), peak_vram: 0 }
}

#[cfg(any(test, feature = "gpu"))]
pub(crate) fn measure_samples_with_interval<R, T, S, F>(
    interval: Duration,
    sample_once: S,
    op: F,
) -> Result<(R, Vec<T>), String>
where
    T: Send + 'static,
    S: FnMut(Duration) -> Result<T, String> + Send + 'static,
    F: FnOnce() -> R,
{
    if interval.is_zero() {
        return Err("sampling interval must be positive".to_string());
    }

    let start = Instant::now();
    let mut sample_once = sample_once;
    let initial_sample = sample_once(Duration::ZERO)?;
    let done = Arc::new(AtomicBool::new(false));
    let (sample_tx, sample_rx) = mpsc::channel();
    let done_for_thread = Arc::clone(&done);

    let sampler = thread::spawn(move || {
        let mut sample_once = sample_once;
        while !done_for_thread.load(Ordering::Acquire) {
            thread::park_timeout(interval);
            if done_for_thread.load(Ordering::Acquire) {
                break;
            }

            let sample = sample_once(start.elapsed());
            let should_stop = sample.is_err();
            if sample_tx.send(sample).is_err() || should_stop {
                break;
            }
        }
    });

    let op_result = catch_unwind(AssertUnwindSafe(op));
    done.store(true, Ordering::Release);
    sampler.thread().unpark();
    sampler.join().expect("sampling thread panicked unexpectedly");

    let op_result = match op_result {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    };

    let mut samples = vec![initial_sample];
    for sample in sample_rx {
        samples.push(sample?);
    }

    Ok((op_result, samples))
}

pub trait BenchEstimator<E: Evaluable> {
    fn estimate_input(&self) -> CircuitBenchEstimate;
    fn estimate_add(&self) -> CircuitBenchEstimate;
    fn estimate_sub(&self) -> CircuitBenchEstimate;
    fn estimate_mul(&self) -> CircuitBenchEstimate;
    fn estimate_small_scalar_mul(&self, scalar: &[u32]) -> CircuitBenchEstimate;
    fn estimate_large_scalar_mul(&self, scalar: &[BigUint]) -> CircuitBenchEstimate;
    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate;
    fn estimate_public_lookup(&self, lut_id: usize) -> CircuitBenchEstimate;

    fn estimate_gate_bench_with_bindings(
        &self,
        gate_type: &PolyGateType,
        param_bindings: &[SubCircuitParamValue],
    ) -> CircuitBenchEstimate {
        match gate_type {
            PolyGateType::Input => self.estimate_input(),
            PolyGateType::Add => self.estimate_add(),
            PolyGateType::Sub => self.estimate_sub(),
            PolyGateType::Mul => self.estimate_mul(),
            PolyGateType::SmallScalarMul { scalar } => {
                self.estimate_small_scalar_mul(scalar.resolve_small_scalar(param_bindings))
            }
            PolyGateType::LargeScalarMul { scalar } => {
                self.estimate_large_scalar_mul(scalar.resolve_large_scalar(param_bindings))
            }
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                self.estimate_slot_transfer(src_slots.as_ref())
            }
            PolyGateType::PubLut { lut_id } => {
                self.estimate_public_lookup(lut_id.resolve_public_lookup(param_bindings))
            }
            PolyGateType::SubCircuitOutput { .. } | PolyGateType::SummedSubCircuitOutput { .. } => {
                panic!(
                    "BenchEstimator::estimate_gate_bench received unexpected SubCircuitOutput or SummedSubCircuitOutput"
                )
            }
        }
    }

    fn estimate_gate_bench(&self, gate_type: &PolyGateType) -> CircuitBenchEstimate {
        self.estimate_gate_bench_with_bindings(gate_type, &[])
    }

    fn estimate_circuit_bench(&self, circuit: &PolyCircuit<E::P>) -> CircuitBenchSummary
    where
        Self: Sync,
    {
        let summary_cache = RwLock::new(HashMap::new());
        let start = Instant::now();
        debug!(
            "estimate_circuit_bench start: circuit_ptr={}, outputs={}, initial_param_bindings=0",
            circuit as *const PolyCircuit<E::P> as usize,
            circuit.num_output()
        );
        let estimate = estimate_circuit_bench_with_cache(self, circuit, &[], &summary_cache);
        debug!(
            "estimate_circuit_bench finished: circuit_ptr={}, outputs={}, total_time={:.6}, latency={:.6}, max_parallelism={}, cache_entries={}, elapsed_ms={:.3}",
            circuit as *const PolyCircuit<E::P> as usize,
            circuit.num_output(),
            estimate.total_time,
            estimate.latency,
            estimate.max_parallelism,
            bench_summary_cache_len(&summary_cache),
            start.elapsed().as_secs_f64() * 1000.0
        );
        estimate
    }
}

fn estimate_circuit_bench_with_cache<E, B>(
    estimator: &B,
    circuit: &PolyCircuit<E::P>,
    param_bindings: &[SubCircuitParamValue],
    summary_cache: &BenchSummaryCache,
) -> CircuitBenchSummary
where
    E: Evaluable,
    B: BenchEstimator<E> + Sync + ?Sized,
{
    let circuit_key =
        BenchSummaryCacheKey { circuit_ptr: circuit as *const PolyCircuit<E::P> as usize };
    if let Some(summary) = bench_summary_cache_get(summary_cache, &circuit_key) {
        debug!(
            "estimate_circuit_bench cache hit: circuit_ptr={}, param_bindings={}, total_time={:.6}, latency={:.6}, max_parallelism={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            summary.total_time,
            summary.latency,
            summary.max_parallelism
        );
        return summary;
    }

    let start = Instant::now();
    let mut estimate = CircuitBenchSummary::new(0.0, 0.0, 0, 0);
    #[cfg(feature = "gpu")]
    let mut peak_vram = 0;
    let grouped_plan = circuit.grouped_execution_plan();
    let GroupedExecutionPlan { layers: grouped_layers, reachable_input_gate_ids: reachable_inputs } =
        grouped_plan;
    debug!(
        "estimate_circuit_bench cache miss: circuit_ptr={}, param_bindings={}, reachable_inputs={} starting input estimate",
        circuit_key.circuit_ptr,
        param_bindings.len(),
        reachable_inputs.len()
    );
    if !reachable_inputs.is_empty() {
        let input_estimate = estimator.estimate_input();
        let input_count = reachable_inputs.len() as f64;
        estimate.total_time += input_estimate.total_time * input_count;
        estimate.latency += input_estimate.latency;
        estimate.max_parallelism = estimate.max_parallelism.max(
            input_estimate
                .parallelism_factor()
                .checked_mul(reachable_inputs.len() as u128)
                .expect("input parallelism overflowed u128 while scaling by input count"),
        );
        #[cfg(feature = "gpu")]
        {
            peak_vram = peak_vram.max(input_estimate.peak_vram);
        }
        debug!(
            "estimate_circuit_bench input estimate applied: circuit_ptr={}, param_bindings={}, reachable_inputs={}, input_total_time={:.6}, input_latency={:.6}, accumulated_total_time={:.6}, accumulated_latency={:.6}, accumulated_max_parallelism={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            reachable_inputs.len(),
            input_estimate.total_time * input_count,
            input_estimate.latency,
            estimate.total_time,
            estimate.latency,
            estimate.max_parallelism
        );
    }
    debug!(
        "estimate_circuit_bench grouped layers ready: circuit_ptr={}, param_bindings={}, layer_count={}, elapsed_ms={:.3}",
        circuit_key.circuit_ptr,
        param_bindings.len(),
        grouped_layers.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );

    for (
        layer_idx,
        GroupedCallExecutionLayer {
            sub_circuit_call_ids,
            summed_sub_circuit_call_ids,
            regular_gate_ids,
        },
    ) in grouped_layers.into_iter().enumerate()
    {
        let layer_start = Instant::now();
        debug!(
            "estimate_circuit_bench layer start: circuit_ptr={}, param_bindings={}, layer_index={}, sub_calls={}, summed_sub_calls={}, regular_gates={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            layer_idx,
            sub_circuit_call_ids.len(),
            summed_sub_circuit_call_ids.len(),
            regular_gate_ids.len()
        );
        let layer_counts: HashMap<PolyGateType, usize> = regular_gate_ids
            .into_par_iter()
            .map(|gate_id| circuit.gate(gate_id).gate_type.clone())
            .fold(HashMap::new, |mut counts, gate_type| {
                *counts.entry(gate_type).or_insert(0) += 1;
                counts
            })
            .reduce(HashMap::new, |mut left, right| {
                for (gate_type, count) in right {
                    *left.entry(gate_type).or_insert(0) += count;
                }
                left
            });
        let layer_regular_gate_total: usize = layer_counts.values().copied().sum();
        debug!(
            "estimate_circuit_bench layer regular counts ready: circuit_ptr={}, param_bindings={}, layer_index={}, regular_gate_kinds={}, regular_gate_total={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            layer_idx,
            layer_counts.len(),
            layer_regular_gate_total
        );
        let direct_requests = sub_circuit_call_ids
            .iter()
            .map(|call_id| {
                let call = circuit.sub_circuit_call_info(*call_id);
                let sub_circuit = circuit.registered_sub_circuit_ref(call.sub_circuit_id);
                prepared_bench_summary_request(
                    call.sub_circuit_id,
                    sub_circuit,
                    call.param_bindings.to_vec(),
                )
            })
            .collect::<Vec<_>>();
        let summed_requests = summed_sub_circuit_call_ids
            .iter()
            .flat_map(|summed_call_id| {
                let call = circuit.summed_sub_circuit_call_info(*summed_call_id);
                let sub_circuit = circuit.registered_sub_circuit_ref(call.sub_circuit_id);
                call.param_bindings
                    .iter()
                    .map(|bindings| {
                        prepared_bench_summary_request(
                            call.sub_circuit_id,
                            Arc::clone(&sub_circuit),
                            bindings.as_ref().to_vec(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut unique_requests_by_key =
            HashMap::<BenchSummaryCacheKey, PreparedBenchSummaryRequest<E::P>>::new();
        for request in direct_requests.iter().chain(summed_requests.iter()) {
            unique_requests_by_key
                .entry(request.cache_key.clone())
                .or_insert_with(|| request.clone());
        }
        let missing_requests = unique_requests_by_key
            .into_values()
            .filter(|request| !bench_summary_cache_contains(summary_cache, &request.cache_key))
            .collect::<Vec<_>>();
        if !missing_requests.is_empty() {
            debug!(
                "estimate_circuit_bench layer subcircuit warmup: circuit_ptr={}, param_bindings={}, layer_index={}, direct_requests={}, summed_requests={}, unique_missing_requests={}",
                circuit_key.circuit_ptr,
                param_bindings.len(),
                layer_idx,
                direct_requests.len(),
                summed_requests.len(),
                missing_requests.len()
            );
            for request in missing_requests {
                let _ = estimate_circuit_bench_with_cache::<E, B>(
                    estimator,
                    request.sub_circuit.as_ref(),
                    &request.param_bindings,
                    summary_cache,
                );
            }
        }
        let (direct_aggregate, (summed_aggregate, regular_aggregate)) = rayon::join(
            || aggregate_cached_sub_summaries(summary_cache, &direct_requests),
            || {
                rayon::join(
                    || aggregate_cached_sub_summaries(summary_cache, &summed_requests),
                    || {
                        layer_counts
                            .into_par_iter()
                            .map(|(gate_type, count)| {
                                let gate_estimate =
                                    estimator.estimate_gate_bench_with_bindings(&gate_type, param_bindings);
                                let gate_total_time = gate_estimate.total_time * count as f64;
                                let gate_parallelism = gate_estimate
                                    .parallelism_factor()
                                    .checked_mul(count as u128)
                                    .expect(
                                        "layer parallelism overflowed u128 while scaling by gate count",
                                    );
                                debug!(
                                    "estimate_circuit_bench regular gate kind: circuit_ptr={}, param_bindings={}, layer_index={}, gate_type={:?}, count={}, gate_total_time={:.6}, gate_latency={:.6}, gate_parallelism={}",
                                    circuit_key.circuit_ptr,
                                    param_bindings.len(),
                                    layer_idx,
                                    gate_type,
                                    count,
                                    gate_total_time,
                                    gate_estimate.latency,
                                    gate_parallelism
                                );
                                RegularGateAggregate {
                                    total_time: gate_total_time,
                                    max_latency: gate_estimate.latency,
                                    total_parallelism: gate_parallelism,
                                    #[cfg(feature = "gpu")]
                                    peak_vram: gate_estimate.peak_vram,
                                }
                            })
                            .reduce(RegularGateAggregate::default, |mut left, right| {
                                left.total_time += right.total_time;
                                left.max_latency = left.max_latency.max(right.max_latency);
                                left.total_parallelism = left
                                    .total_parallelism
                                    .checked_add(right.total_parallelism)
                                    .expect(
                                        "layer parallelism overflowed u128 while summing gate kinds",
                                    );
                                #[cfg(feature = "gpu")]
                                {
                                    left.peak_vram = left.peak_vram.max(right.peak_vram);
                                }
                                left
                            })
                    },
                )
            },
        );

        debug!(
            "estimate_circuit_bench layer cached aggregates ready: circuit_ptr={}, param_bindings={}, layer_index={}, direct_calls={}, direct_total_time={:.6}, direct_max_latency={:.6}, direct_total_parallelism={}, summed_invocations={}, summed_total_time={:.6}, summed_max_latency={:.6}, summed_total_parallelism={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            layer_idx,
            direct_requests.len(),
            direct_aggregate.total_time,
            direct_aggregate.max_latency,
            direct_aggregate.total_parallelism,
            summed_requests.len(),
            summed_aggregate.total_time,
            summed_aggregate.max_latency,
            summed_aggregate.total_parallelism
        );

        estimate.total_time += direct_aggregate.total_time + summed_aggregate.total_time;
        let layer_latency = direct_aggregate
            .max_latency
            .max(summed_aggregate.max_latency)
            .max(regular_aggregate.max_latency);
        let layer_parallelism = direct_aggregate
            .total_parallelism
            .checked_add(summed_aggregate.total_parallelism)
            .expect("layer parallelism overflowed u128 while summing direct and summed subcircuit requests")
            .max(regular_aggregate.total_parallelism)
            ;
        #[cfg(feature = "gpu")]
        {
            peak_vram = peak_vram.max(direct_aggregate.peak_vram).max(summed_aggregate.peak_vram);
        }

        estimate.total_time += regular_aggregate.total_time;
        #[cfg(feature = "gpu")]
        {
            peak_vram = peak_vram.max(regular_aggregate.peak_vram);
        }
        debug!(
            "estimate_circuit_bench regular aggregate applied: circuit_ptr={}, param_bindings={}, layer_index={}, regular_gate_total={}, regular_total_time={:.6}, regular_max_latency={:.6}, regular_parallelism={}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            layer_idx,
            layer_regular_gate_total,
            regular_aggregate.total_time,
            regular_aggregate.max_latency,
            regular_aggregate.total_parallelism
        );
        estimate.latency += layer_latency;
        estimate.max_parallelism = estimate.max_parallelism.max(layer_parallelism);
        debug!(
            "estimate_circuit_bench layer finished: circuit_ptr={}, param_bindings={}, layer_index={}, accumulated_total_time={:.6}, accumulated_latency={:.6}, accumulated_max_parallelism={}, elapsed_ms={:.3}",
            circuit_key.circuit_ptr,
            param_bindings.len(),
            layer_idx,
            estimate.total_time,
            estimate.latency,
            estimate.max_parallelism,
            layer_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    let estimate = CircuitBenchSummary::new(
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
        #[cfg(feature = "gpu")]
        peak_vram,
        #[cfg(not(feature = "gpu"))]
        0,
    );

    debug!(
        "estimate_circuit_bench cache store: circuit_ptr={}, param_bindings={}, total_time={:.6}, latency={:.6}, max_parallelism={}, elapsed_ms={:.3}",
        circuit_key.circuit_ptr,
        param_bindings.len(),
        estimate.total_time,
        estimate.latency,
        estimate.max_parallelism,
        start.elapsed().as_secs_f64() * 1000.0
    );
    bench_summary_cache_insert(summary_cache, circuit_key, estimate);
    estimate
}

#[cfg(test)]
mod tests {
    use super::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, measure_samples_with_interval,
    };
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, PolyGateType, SubCircuitParamValue},
        poly::dcrt::poly::DCRTPoly,
    };
    use sequential_test::sequential;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
    };

    fn bench(latency: f64, total_time: f64, peak_vram: usize) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(total_time, latency, 0, peak_vram)
    }

    fn summary(
        total_time: f64,
        latency: f64,
        max_parallelism: u128,
        peak_vram: usize,
    ) -> CircuitBenchSummary {
        CircuitBenchSummary::new(total_time, latency, max_parallelism, peak_vram)
    }

    struct TestBenchEstimator;

    impl BenchEstimator<DCRTPoly> for TestBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            bench(0.2, 0.25, 1)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            bench(1.0, 1.5, 5)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            bench(2.0, 2.5, 9)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            bench(3.0, 3.5, 7)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            bench(4.0, 4.5, 11)
        }

        fn estimate_large_scalar_mul(
            &self,
            _scalar: &[num_bigint::BigUint],
        ) -> CircuitBenchEstimate {
            bench(5.0, 5.5, 13)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            bench(6.0, 6.5, 17)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            bench(7.0, 7.5, 19)
        }
    }

    struct ExpandedSubCircuitBenchEstimator;

    impl BenchEstimator<DCRTPoly> for ExpandedSubCircuitBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            bench(0.5, 1.0, 2)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            bench(5.0, 6.0, 23)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            bench(2.0, 4.0, 29)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_large_scalar_mul(
            &self,
            _scalar: &[num_bigint::BigUint],
        ) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }
    }

    #[derive(Default)]
    struct CountingBenchEstimator {
        input_calls: AtomicUsize,
        add_calls: AtomicUsize,
        small_scalar_mul_calls: AtomicUsize,
    }

    impl BenchEstimator<DCRTPoly> for CountingBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            self.input_calls.fetch_add(1, Ordering::SeqCst);
            bench(1.0, 1.0, 0)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            self.add_calls.fetch_add(1, Ordering::SeqCst);
            bench(1.0, 1.0, 0)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            self.small_scalar_mul_calls.fetch_add(1, Ordering::SeqCst);
            bench(1.0, 1.0, 0)
        }

        fn estimate_large_scalar_mul(
            &self,
            _scalar: &[num_bigint::BigUint],
        ) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            bench(0.0, 0.0, 0)
        }
    }

    #[test]
    #[sequential]
    fn test_estimate_circuit_bench_accumulates_layer_latency_and_total_time() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4).to_vec();
        let add_0 = circuit.add_gate(inputs[0], inputs[1]);
        let add_1 = circuit.add_gate(inputs[2], inputs[3]);
        let sub = circuit.sub_gate(inputs[0], inputs[2]);
        let mul = circuit.mul_gate(add_0, sub);
        circuit.output(vec![mul, add_1]);

        let estimate = TestBenchEstimator.estimate_circuit_bench(&circuit);

        assert!((estimate.total_time - 10.0).abs() < 1e-9);
        assert!((estimate.latency - 5.2).abs() < 1e-9);
        assert_eq!(estimate.max_parallelism, 8);
        assert_eq!(estimate, summary(10.0, 5.2, 8, 9));
    }

    #[test]
    #[sequential]
    fn test_estimate_circuit_bench_counts_multi_output_subcircuit_call_once() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2).to_vec();
        let sub_add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add, sub_inputs[0].into()]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(2).to_vec();
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &main_inputs);
        main_circuit.output(sub_outputs);

        let estimate = ExpandedSubCircuitBenchEstimator.estimate_circuit_bench(&main_circuit);
        let expected = summary(10.0, 6.0, 4, 23);

        assert!((estimate.total_time - expected.total_time).abs() < 1e-9);
        assert!((estimate.latency - expected.latency).abs() < 1e-9);
        assert_eq!(estimate.max_parallelism, expected.max_parallelism);
        assert_eq!(estimate, expected);
    }

    #[test]
    #[sequential]
    fn test_estimate_circuit_bench_sequences_subcircuit_calls_before_regular_gates() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2).to_vec();
        let sub_add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(4).to_vec();
        let top_add = main_circuit.add_gate(main_inputs[0], main_inputs[1]);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[main_inputs[2], main_inputs[3]]);
        main_circuit.output(vec![top_add, sub_outputs[0]]);

        let estimate = ExpandedSubCircuitBenchEstimator.estimate_circuit_bench(&main_circuit);
        let expected = summary(18.0, 6.0, 8, 23);

        assert!((estimate.total_time - expected.total_time).abs() < 1e-9);
        assert!((estimate.latency - expected.latency).abs() < 1e-9);
        assert_eq!(estimate.max_parallelism, expected.max_parallelism);
        assert_eq!(estimate, expected);
    }

    #[test]
    #[sequential]
    fn test_estimate_circuit_bench_reuses_cached_subcircuit_summary() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2).to_vec();
        let sub_add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(4).to_vec();
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let first = main_circuit.call_sub_circuit(sub_id, &[main_inputs[0], main_inputs[1]]);
        let second = main_circuit.call_sub_circuit(sub_id, &[main_inputs[2], main_inputs[3]]);
        main_circuit.output(vec![first[0], second[0]]);

        let estimator = CountingBenchEstimator::default();
        let estimate = estimator.estimate_circuit_bench(&main_circuit);

        assert_eq!(estimate, summary(10.0, 3.0, 4, 0));
        assert_eq!(estimator.input_calls.load(Ordering::SeqCst), 2);
        assert_eq!(estimator.add_calls.load(Ordering::SeqCst), 1);
        assert_eq!(estimator.small_scalar_mul_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    #[sequential]
    fn test_estimate_circuit_bench_reuses_cached_subcircuit_summary_across_param_bindings() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let scalar_param = sub_circuit
            .register_sub_circuit_param(crate::circuit::SubCircuitParamKind::SmallScalarMul);
        let sub_inputs = sub_circuit.input(1).to_vec();
        let scaled = sub_circuit.small_scalar_mul_param(sub_inputs[0], scalar_param);
        sub_circuit.output(vec![scaled]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(1).to_vec();
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let doubled = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![2])],
        );
        let tripled = main_circuit.call_sub_circuit_with_bindings(
            sub_id,
            &[main_inputs[0]],
            &[SubCircuitParamValue::SmallScalarMul(vec![3])],
        );
        main_circuit.output(vec![doubled[0], tripled[0]]);

        let estimator = CountingBenchEstimator::default();
        let estimate = estimator.estimate_circuit_bench(&main_circuit);

        assert_eq!(estimate, summary(5.0, 3.0, 2, 0));
        assert_eq!(estimator.input_calls.load(Ordering::SeqCst), 2);
        assert_eq!(estimator.add_calls.load(Ordering::SeqCst), 0);
        assert_eq!(estimator.small_scalar_mul_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[should_panic(expected = "unexpected SubCircuitOutput")]
    #[sequential]
    fn test_estimate_gate_bench_panics_on_sub_circuit_output_placeholder() {
        let estimator = TestBenchEstimator;
        let gate_type = PolyGateType::SubCircuitOutput { call_id: 0, output_idx: 0, num_inputs: 1 };

        let _ = estimator.estimate_gate_bench(&gate_type);
    }

    #[test]
    #[sequential]
    fn test_measure_samples_with_interval_rejects_zero_interval() {
        let err = measure_samples_with_interval(Duration::ZERO, |_| Ok(()), || ())
            .expect_err("zero interval should be rejected");

        assert_eq!(err, "sampling interval must be positive");
    }

    #[test]
    #[sequential]
    fn test_measure_samples_with_interval_collects_samples_until_operation_finishes() {
        let interval = Duration::from_millis(5);
        let sample_count = Arc::new(AtomicUsize::new(0));
        let (ready_tx, ready_rx) = mpsc::channel();
        let sample_count_for_sampler = Arc::clone(&sample_count);

        let (result, samples) = measure_samples_with_interval(
            interval,
            move |elapsed| {
                let count = sample_count_for_sampler.fetch_add(1, Ordering::SeqCst) + 1;
                if count == 3 {
                    let _ = ready_tx.send(());
                }
                Ok(elapsed)
            },
            move || {
                ready_rx
                    .recv_timeout(Duration::from_secs(1))
                    .expect("sampler did not emit enough periodic samples");
                7usize
            },
        )
        .expect("sampling should succeed");

        assert_eq!(result, 7);
        assert!(samples.len() >= 3);
        assert_eq!(samples[0], Duration::ZERO);
        assert!(samples.iter().skip(1).any(|elapsed| *elapsed > Duration::ZERO));
        assert!(samples.windows(2).all(|pair| pair[0] <= pair[1]));

        let count_after_completion = sample_count.load(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(sample_count.load(Ordering::SeqCst), count_after_completion);
    }

    #[test]
    #[sequential]
    fn test_measure_samples_with_interval_returns_sampling_error() {
        let (ready_tx, ready_rx) = mpsc::channel();
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_for_sampler = Arc::clone(&call_count);

        let err = measure_samples_with_interval(
            Duration::from_millis(5),
            move |_elapsed| {
                let count = call_count_for_sampler.fetch_add(1, Ordering::SeqCst) + 1;
                if count == 2 {
                    let _ = ready_tx.send(());
                    return Err("sampling failed".to_string());
                }
                Ok(count)
            },
            move || {
                ready_rx
                    .recv_timeout(Duration::from_secs(1))
                    .expect("sampler did not return an error in time")
            },
        )
        .expect_err("sampling error should be propagated");

        assert_eq!(err, "sampling failed");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }
}
