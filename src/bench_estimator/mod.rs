use std::collections::HashMap;

use num_bigint::BigUint;

use crate::circuit::{Evaluable, PolyCircuit, PolyGateType};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CircuitBenchEstimate {
    pub total_time: f64,
    pub latency: f64,
}

pub trait BenchEstimator<E: Evaluable> {
    fn estimate_input(&self) -> f64;
    fn estimate_add(&self) -> f64;
    fn estimate_sub(&self) -> f64;
    fn estimate_mul(&self) -> f64;
    fn estimate_small_scalar_mul(&self, scalar: &[u32]) -> f64;
    fn estimate_large_scalar_mul(&self, scalar: &[BigUint]) -> f64;
    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> f64;
    fn estimate_public_lookup(&self, lut_id: usize) -> f64;

    fn estimate_gate_bench(&self, gate_type: &PolyGateType) -> f64 {
        match gate_type {
            PolyGateType::Input => self.estimate_input(),
            PolyGateType::Add => self.estimate_add(),
            PolyGateType::Sub => self.estimate_sub(),
            PolyGateType::Mul => self.estimate_mul(),
            PolyGateType::SmallScalarMul { scalar } => self.estimate_small_scalar_mul(scalar),
            PolyGateType::LargeScalarMul { scalar } => self.estimate_large_scalar_mul(scalar),
            PolyGateType::SlotTransfer { src_slots } => self.estimate_slot_transfer(src_slots),
            PolyGateType::PubLut { lut_id } => self.estimate_public_lookup(*lut_id),
            PolyGateType::SubCircuitOutput { .. } => {
                panic!("BenchEstimator::estimate_gate_bench received unexpected SubCircuitOutput")
            }
        }
    }

    fn estimate_circuit_bench(&self, circuit: &PolyCircuit<E::P>) -> CircuitBenchEstimate {
        let mut estimate = CircuitBenchEstimate::default();
        for layer in circuit.expanded_gate_types_by_level() {
            let mut layer_counts: HashMap<PolyGateType, usize> = HashMap::new();
            for gate_type in layer {
                *layer_counts.entry(gate_type).or_insert(0) += 1;
            }

            let mut layer_latency = 0.0_f64;
            for (gate_type, count) in layer_counts.into_iter() {
                let gate_time = self.estimate_gate_bench(&gate_type);
                estimate.total_time += gate_time * count as f64;
                layer_latency = layer_latency.max(gate_time);
            }
            estimate.latency += layer_latency;
        }
        estimate
    }
}

#[cfg(test)]
mod tests {
    use super::{BenchEstimator, CircuitBenchEstimate};
    use crate::{
        circuit::{PolyCircuit, PolyGateType},
        poly::dcrt::poly::DCRTPoly,
    };

    struct TestBenchEstimator;

    impl BenchEstimator<DCRTPoly> for TestBenchEstimator {
        fn estimate_input(&self) -> f64 {
            0.2
        }

        fn estimate_add(&self) -> f64 {
            1.0
        }

        fn estimate_sub(&self) -> f64 {
            2.0
        }

        fn estimate_mul(&self) -> f64 {
            3.0
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> f64 {
            4.0
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[num_bigint::BigUint]) -> f64 {
            5.0
        }

        fn estimate_slot_transfer(&self, _src_slots: &[(u32, Option<u32>)]) -> f64 {
            6.0
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> f64 {
            7.0
        }
    }

    struct ExpandedSubCircuitBenchEstimator;

    impl BenchEstimator<DCRTPoly> for ExpandedSubCircuitBenchEstimator {
        fn estimate_input(&self) -> f64 {
            0.5
        }

        fn estimate_add(&self) -> f64 {
            5.0
        }

        fn estimate_sub(&self) -> f64 {
            0.0
        }

        fn estimate_mul(&self) -> f64 {
            2.0
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> f64 {
            0.0
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[num_bigint::BigUint]) -> f64 {
            0.0
        }

        fn estimate_slot_transfer(&self, _src_slots: &[(u32, Option<u32>)]) -> f64 {
            0.0
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> f64 {
            0.0
        }
    }

    #[test]
    fn test_estimate_circuit_bench_accumulates_layer_latency_and_total_time() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4);
        let add_0 = circuit.add_gate(inputs[0], inputs[1]);
        let add_1 = circuit.add_gate(inputs[2], inputs[3]);
        let sub = circuit.sub_gate(inputs[0], inputs[2]);
        let mul = circuit.mul_gate(add_0, sub);
        circuit.output(vec![mul, add_1]);

        let estimate = TestBenchEstimator.estimate_circuit_bench(&circuit);

        assert!((estimate.total_time - 7.8).abs() < 1e-9);
        assert!((estimate.latency - 5.2).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_circuit_bench_expands_sub_circuit_without_counting_placeholders() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(2);
        let sub_add = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);
        sub_circuit.output(vec![sub_add]);

        let mut main_circuit = PolyCircuit::<DCRTPoly>::new();
        let main_inputs = main_circuit.input(3);
        let mul = main_circuit.mul_gate(main_inputs[0], main_inputs[1]);
        let sub_id = main_circuit.register_sub_circuit(sub_circuit);
        let sub_outputs = main_circuit.call_sub_circuit(sub_id, &[mul, main_inputs[2]]);
        main_circuit.output(vec![sub_outputs[0]]);

        let estimate = ExpandedSubCircuitBenchEstimator.estimate_circuit_bench(&main_circuit);
        let expected = CircuitBenchEstimate { total_time: 8.5, latency: 7.5 };

        assert!((estimate.total_time - expected.total_time).abs() < 1e-9);
        assert!((estimate.latency - expected.latency).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "unexpected SubCircuitOutput")]
    fn test_estimate_gate_bench_panics_on_sub_circuit_output_placeholder() {
        let estimator = TestBenchEstimator;
        let gate_type = PolyGateType::SubCircuitOutput { call_id: 0, output_idx: 0, num_inputs: 1 };

        let _ = estimator.estimate_gate_bench(&gate_type);
    }
}
