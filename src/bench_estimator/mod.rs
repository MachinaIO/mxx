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

use std::{collections::HashMap, hint::black_box, time::Instant};
#[cfg(any(test, feature = "gpu"))]
use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use num_bigint::BigUint;

use crate::circuit::{
    Evaluable, GroupedCallExecutionLayer, PolyCircuit, PolyGateType, SubCircuitParamValue,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BenchSummaryCacheKey {
    circuit_ptr: usize,
    param_bindings: Vec<SubCircuitParamValue>,
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
    #[cfg(feature = "gpu")]
    pub peak_vram: usize,
}

impl CircuitBenchEstimate {
    pub(crate) fn new(total_time: f64, latency: f64) -> Self {
        Self {
            total_time,
            latency,
            #[cfg(feature = "gpu")]
            peak_vram: 0,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn with_peak_vram(mut self, peak_vram: usize) -> Self {
        self.peak_vram = peak_vram;
        self
    }

    #[cfg(not(feature = "gpu"))]
    pub(crate) fn with_peak_vram(self, _peak_vram: usize) -> Self {
        self
    }

    fn parallelism_factor(&self) -> u128 {
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
    pub(crate) fn new(total_time: f64, latency: f64, max_parallelism: u128) -> Self {
        Self {
            total_time,
            latency,
            max_parallelism,
            #[cfg(feature = "gpu")]
            peak_vram: 0,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn with_peak_vram(mut self, peak_vram: usize) -> Self {
        self.peak_vram = peak_vram;
        self
    }

    #[cfg(not(feature = "gpu"))]
    pub(crate) fn with_peak_vram(self, _peak_vram: usize) -> Self {
        self
    }
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

    fn estimate_circuit_bench(&self, circuit: &PolyCircuit<E::P>) -> CircuitBenchSummary {
        let mut summary_cache = HashMap::new();
        estimate_circuit_bench_with_cache(self, circuit, &[], &mut summary_cache)
    }
}

fn estimate_circuit_bench_with_cache<E, B>(
    estimator: &B,
    circuit: &PolyCircuit<E::P>,
    param_bindings: &[SubCircuitParamValue],
    summary_cache: &mut HashMap<BenchSummaryCacheKey, CircuitBenchSummary>,
) -> CircuitBenchSummary
where
    E: Evaluable,
    B: BenchEstimator<E> + ?Sized,
{
    let circuit_key = BenchSummaryCacheKey {
        circuit_ptr: circuit as *const PolyCircuit<E::P> as usize,
        param_bindings: param_bindings.to_vec(),
    };
    if let Some(summary) = summary_cache.get(&circuit_key) {
        return *summary;
    }

    let mut estimate = CircuitBenchSummary::new(0.0, 0.0, 0);
    #[cfg(feature = "gpu")]
    let mut peak_vram = 0;
    let reachable_inputs = circuit.reachable_input_gate_ids();
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
    }

    for GroupedCallExecutionLayer {
        sub_circuit_call_ids,
        summed_sub_circuit_call_ids,
        regular_gate_ids,
    } in circuit.grouped_execution_layers()
    {
        let mut layer_counts: HashMap<PolyGateType, usize> = HashMap::new();
        let mut layer_latency = 0.0_f64;
        let mut layer_parallelism = 0u128;

        for call_id in sub_circuit_call_ids {
            let call = circuit.sub_circuit_call_info(call_id);
            let sub_circuit = circuit.registered_sub_circuit_ref(call.sub_circuit_id);
            let sub_summary = estimate_circuit_bench_with_cache::<E, B>(
                estimator,
                sub_circuit.as_ref(),
                &call.param_bindings,
                summary_cache,
            );
            estimate.total_time += sub_summary.total_time;
            layer_latency += sub_summary.latency;
            layer_parallelism = layer_parallelism.max(sub_summary.max_parallelism);
            #[cfg(feature = "gpu")]
            {
                peak_vram = peak_vram.max(sub_summary.peak_vram);
            }
        }
        for summed_call_id in summed_sub_circuit_call_ids {
            let call = circuit.summed_sub_circuit_call_info(summed_call_id);
            let sub_circuit = circuit.registered_sub_circuit_ref(call.sub_circuit_id);
            for param_bindings in &call.param_bindings {
                let sub_summary = estimate_circuit_bench_with_cache::<E, B>(
                    estimator,
                    sub_circuit.as_ref(),
                    param_bindings.as_ref(),
                    summary_cache,
                );
                estimate.total_time += sub_summary.total_time;
                layer_latency += sub_summary.latency;
                layer_parallelism = layer_parallelism.max(sub_summary.max_parallelism);
                #[cfg(feature = "gpu")]
                {
                    peak_vram = peak_vram.max(sub_summary.peak_vram);
                }
            }
        }

        for gate_id in regular_gate_ids {
            let gate_type = circuit.gate(gate_id).gate_type.clone();
            *layer_counts.entry(gate_type).or_insert(0) += 1;
        }

        let mut regular_latency = 0.0_f64;
        let mut regular_parallelism = 0u128;
        for (gate_type, count) in layer_counts.into_iter() {
            let gate_estimate =
                estimator.estimate_gate_bench_with_bindings(&gate_type, param_bindings);
            estimate.total_time += gate_estimate.total_time * count as f64;
            regular_latency = regular_latency.max(gate_estimate.latency);
            let gate_parallelism = gate_estimate
                .parallelism_factor()
                .checked_mul(count as u128)
                .expect("layer parallelism overflowed u128 while scaling by gate count");
            regular_parallelism = regular_parallelism
                .checked_add(gate_parallelism)
                .expect("layer parallelism overflowed u128 while summing gate kinds");
            #[cfg(feature = "gpu")]
            {
                peak_vram = peak_vram.max(gate_estimate.peak_vram);
            }
        }

        layer_latency += regular_latency;
        layer_parallelism = layer_parallelism.max(regular_parallelism);
        estimate.latency += layer_latency;
        estimate.max_parallelism = estimate.max_parallelism.max(layer_parallelism);
    }

    let estimate = estimate.with_peak_vram({
        #[cfg(feature = "gpu")]
        {
            peak_vram
        }
        #[cfg(not(feature = "gpu"))]
        {
            0
        }
    });

    summary_cache.insert(circuit_key, estimate);
    estimate
}

#[cfg(test)]
mod tests {
    use super::{
        BenchEstimator, CircuitBenchEstimate, CircuitBenchSummary, measure_samples_with_interval,
    };
    use crate::{
        __PAIR, __TestState,
        circuit::{PolyCircuit, PolyGateType},
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
        CircuitBenchEstimate::new(total_time, latency).with_peak_vram(peak_vram)
    }

    fn summary(
        total_time: f64,
        latency: f64,
        max_parallelism: u128,
        peak_vram: usize,
    ) -> CircuitBenchSummary {
        CircuitBenchSummary::new(total_time, latency, max_parallelism).with_peak_vram(peak_vram)
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
        let expected = summary(18.0, 11.0, 8, 23);

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

        assert_eq!(estimate, summary(10.0, 5.0, 4, 0));
        assert_eq!(estimator.input_calls.load(Ordering::SeqCst), 2);
        assert_eq!(estimator.add_calls.load(Ordering::SeqCst), 1);
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
