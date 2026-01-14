pub mod evaluable;
pub mod gate;
pub mod serde;

pub use evaluable::*;
pub use gate::{PolyGate, PolyGateKind, PolyGateType};

use dashmap::DashMap;
use num_bigint::BigUint;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};
use tracing::{debug, info};
#[derive(Debug, Clone, Default)]
pub struct PolyCircuit<P: Poly> {
    gates: BTreeMap<GateId, PolyGate>,
    print_value: BTreeMap<GateId, String>,
    sub_circuits: BTreeMap<usize, PolyCircuit<P>>,
    output_ids: Vec<GateId>,
    num_input: usize,
    gate_counts: HashMap<PolyGateKind, usize>,
    pub lookups: HashMap<usize, Arc<PublicLut<P>>>,
}

impl<P: Poly> PartialEq for PolyCircuit<P> {
    fn eq(&self, other: &Self) -> bool {
        self.gates == other.gates &&
            self.print_value == other.print_value &&
            self.sub_circuits == other.sub_circuits &&
            self.output_ids == other.output_ids &&
            self.gate_counts == other.gate_counts &&
            self.num_input == other.num_input
    }
}

impl<P: Poly> Eq for PolyCircuit<P> {}

impl<P: Poly> PolyCircuit<P> {
    pub fn new() -> Self {
        let mut gates = BTreeMap::new();
        // Ensure the reserved constant-one gate exists at GateId(0)
        gates.insert(GateId(0), PolyGate::new(GateId(0), PolyGateType::Input, vec![]));
        let mut gate_counts = HashMap::new();
        gate_counts.insert(PolyGateKind::Input, 1);
        Self {
            gates,
            print_value: BTreeMap::new(),
            sub_circuits: BTreeMap::new(),
            output_ids: vec![],
            num_input: 0,
            gate_counts,
            lookups: HashMap::new(),
        }
    }

    /// Get number of inputs
    pub fn num_input(&self) -> usize {
        self.num_input
    }

    /// Get number of outputs
    pub fn num_output(&self) -> usize {
        self.output_ids.len()
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn count_gates_by_type_vec(&self) -> HashMap<PolyGateKind, usize> {
        let mut counts = HashMap::new();
        self.count_helper(&mut counts);
        counts
    }

    fn count_helper(&self, counts: &mut HashMap<PolyGateKind, usize>) {
        for (&kind, &count) in &self.gate_counts {
            *counts.entry(kind).or_insert(0) += count;
        }
        for sub in self.sub_circuits.values() {
            sub.count_helper(counts);
        }
    }

    /// Computes the circuit depth excluding Add gates.
    ///
    /// Definition:
    /// - Inputs and the reserved constant-one gate contribute 0 to depth.
    /// - Add, Sub, SmallScalarMul, Rotate gates do not increase depth: level(add) =
    ///   max(level(inputs)).
    /// - Any other non-input gate increases depth by 1: level(g) = max(level(inputs)) + 1.
    /// - If there are no outputs, returns 0.
    pub fn non_free_depth(&self) -> usize {
        if self.output_ids.is_empty() {
            return 0;
        }

        // Compute a topo order of all gates needed for outputs
        let order = self.topological_order();
        let mut level_map: HashMap<GateId, usize> = HashMap::new();

        for gate_id in order.iter() {
            let gate = self.gates.get(gate_id).expect("gate not found");
            if gate.input_gates.is_empty() {
                // Inputs and consts
                level_map.insert(*gate_id, 0);
                continue;
            }

            let max_in = gate
                .input_gates
                .iter()
                .map(|id| level_map[id])
                .max()
                .expect("non-input gate must have inputs");

            let incr = match gate.gate_type {
                PolyGateType::Add |
                PolyGateType::Sub |
                PolyGateType::SmallScalarMul { scalar: _ } |
                PolyGateType::Rotate { shift: _ } => 0,
                _ => 1,
            };
            level_map.insert(*gate_id, max_in + incr);
        }

        // Max depth among outputs
        self.output_ids.iter().map(|id| level_map[id]).max().unwrap_or(0)
    }

    pub fn recompute_gate_counts(&mut self) {
        self.gate_counts.clear();
        for gate in self.gates.values() {
            let kind = gate.gate_type.kind();
            *self.gate_counts.entry(kind).or_insert(0) += 1;
        }
    }

    pub fn print(&mut self, gate_id: GateId, prefix: String) {
        self.print_value.insert(gate_id, prefix);
    }

    pub fn input(&mut self, num_input: usize) -> Vec<GateId> {
        let mut input_gates = Vec::with_capacity(num_input);
        for _ in 0..num_input {
            let next_id = self.gates.len();
            let gid = GateId(next_id);
            self.increment_gate_kind(PolyGateKind::Input);
            self.gates.insert(gid, PolyGate::new(gid, PolyGateType::Input, vec![]));
            input_gates.push(gid);
        }
        self.num_input += num_input;
        input_gates
    }

    pub fn output(&mut self, outputs: Vec<GateId>) {
        #[cfg(debug_assertions)]
        assert_eq!(self.output_ids.len(), 0);
        for gate_id in outputs.into_iter() {
            self.output_ids.push(gate_id);
        }
    }

    pub fn const_zero_gate(&mut self) -> GateId {
        self.not_gate(GateId(0))
    }

    /// index 0 have value 1
    pub fn const_one_gate(&mut self) -> GateId {
        GateId(0)
    }

    pub fn const_minus_one_gate(&mut self) -> GateId {
        let zero = self.const_zero_gate();
        self.sub_gate(zero, GateId(0))
    }

    pub fn and_gate(&mut self, left: GateId, right: GateId) -> GateId {
        self.mul_gate(left, right)
    }

    /// Computes the NOT gate using arithmetic inversion: `1 - x`.
    /// This operation assumes that `x` is restricted to binary values (0 or 1),
    /// meaning it should only be used with polynomials sampled from a bit distribution.
    /// The computation is achieved by subtracting `x` from 1 (i.e., `0 - x + 1`).
    pub fn not_gate(&mut self, input: GateId) -> GateId {
        self.sub_gate(GateId(0), input)
    }

    pub fn or_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let add = self.add_gate(left, right);
        let mul = self.mul_gate(left, right);
        self.sub_gate(add, mul) // A + B - A*B
    }

    /// Computes the NAND gate as `NOT(AND(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nand_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let and_result = self.and_gate(left, right);
        self.not_gate(and_result) // NOT AND
    }

    /// Computes the NOR gate as `NOT(OR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn nor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let or_result = self.or_gate(left, right);
        self.not_gate(or_result) // NOT OR
    }

    pub fn xor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let two = self.add_gate(GateId(0), GateId(0));
        let mul = self.mul_gate(left, right);
        let two_mul = self.mul_gate(two, mul);
        let add = self.add_gate(left, right);
        self.sub_gate(add, two_mul) // A + B - 2*A*B
    }

    /// Computes the XNOR gate as `NOT(XOR(left, right))`.
    /// This operation follows the same restriction as the NOT gate:
    /// `left` and `right` must be bit distribution (0 or 1)
    pub fn xnor_gate(&mut self, left: GateId, right: GateId) -> GateId {
        let xor_result = self.xor_gate(left, right);
        self.not_gate(xor_result) // NOT XOR
    }

    pub fn const_digits(&mut self, digits: &[u32]) -> GateId {
        let one = self.const_one_gate();
        self.small_scalar_mul(one, digits)
    }

    pub fn const_poly(&mut self, poly: &P) -> GateId {
        let one = self.const_one_gate();
        self.large_scalar_mul(one, &poly.coeffs_biguints())
    }

    pub fn add_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Add)
    }

    pub fn sub_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Sub)
    }

    pub fn mul_gate(&mut self, left_input: GateId, right_input: GateId) -> GateId {
        self.new_gate_generic(vec![left_input, right_input], PolyGateType::Mul)
    }

    pub fn small_scalar_mul(&mut self, input: GateId, scalar: &[u32]) -> GateId {
        self.new_gate_generic(vec![input], PolyGateType::SmallScalarMul { scalar: scalar.to_vec() })
    }

    pub fn large_scalar_mul(&mut self, input: GateId, scalar: &[BigUint]) -> GateId {
        self.new_gate_generic(vec![input], PolyGateType::LargeScalarMul { scalar: scalar.to_vec() })
    }

    pub fn poly_scalar_mul(&mut self, input: GateId, scalar: &P) -> GateId {
        self.new_gate_generic(
            vec![input],
            PolyGateType::LargeScalarMul { scalar: scalar.coeffs_biguints() },
        )
    }

    pub fn rotate_gate(&mut self, input: GateId, shift: i32) -> GateId {
        self.new_gate_generic(vec![input], PolyGateType::Rotate { shift })
    }

    pub fn public_lookup_gate(&mut self, input: GateId, lut_id: usize) -> GateId {
        self.new_gate_generic(vec![input], PolyGateType::PubLut { lut_id })
    }

    fn new_gate_generic(&mut self, inputs: Vec<GateId>, gate_type: PolyGateType) -> GateId {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.output_ids.len(), 0);
            assert_eq!(inputs.len(), gate_type.num_input());
            for gate_id in inputs.iter() {
                assert!(self.gates.contains_key(gate_id));
            }
        }
        let gate_id = self.gates.len();
        let gate_kind = gate_type.kind();
        self.increment_gate_kind(gate_kind);
        self.gates.insert(GateId(gate_id), PolyGate::new(GateId(gate_id), gate_type, inputs));
        GateId(gate_id)
    }

    fn increment_gate_kind(&mut self, kind: PolyGateKind) {
        *self.gate_counts.entry(kind).or_insert(0) += 1;
    }

    /// Computes a topological order (as a vector of gate IDs) for all gates that
    /// are needed to evaluate the outputs. This is done via a DFS from each output gate.
    fn topological_order(&self) -> Vec<GateId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = Vec::new();
        for &output_gate in &self.output_ids {
            if visited.insert(output_gate) {
                stack.push((output_gate, 0));
            }
        }

        while let Some((node, child_idx)) = stack.pop() {
            let gate = self.gates.get(&node).expect("gate not found");

            if child_idx < gate.input_gates.len() {
                stack.push((node, child_idx + 1));
                let child = gate.input_gates[child_idx];
                if visited.insert(child) {
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
            }
        }

        order
    }

    /// Computes a levelized grouping of gate ids.
    /// Input wires (keys 0..=num_input) are assigned level 0.
    /// Each non‐input gate's level is defined as max(levels of its inputs) + 1.
    fn compute_levels(&self) -> Vec<Vec<GateId>> {
        let mut gate_levels: HashMap<GateId, usize> = HashMap::new();
        let mut levels: Vec<Vec<GateId>> = vec![];
        let orders = self.topological_order();
        for gate_id in orders.into_iter() {
            let gate = self.gates.get(&gate_id).expect("gate not found");
            if gate.input_gates.is_empty() {
                // Inputs and consts have no dependencies; place them at level 0
                gate_levels.insert(gate_id, 0);
                if levels.is_empty() {
                    levels.push(vec![]);
                }
                levels[0].push(gate_id);
                continue;
            }
            // Find the maximum level among all input gates, then add 1.
            let max_input_level = gate
                .input_gates
                .iter()
                .map(|id| gate_levels[id])
                .max()
                .expect("gate has input_gates but max() returned None");
            let level = max_input_level + 1;
            gate_levels.insert(gate_id, level);
            if levels.len() <= level {
                levels.resize(level + 1, vec![]);
            }
            levels[level].push(gate_id);
        }
        levels
    }

    /// Returns the circuit depth defined as the maximum level index among
    /// all gates required to compute the outputs.
    ///
    /// - Inputs and constant-one gate reside at level 0.
    /// - Each non-input gate is assigned level = max(input levels) + 1.
    /// - If there are no outputs, depth is 0.
    pub fn depth(&self) -> usize {
        let levels = self.compute_levels();
        if levels.is_empty() { 0 } else { levels.len() - 1 }
    }

    /// Evaluate the circuit using an iterative approach over a precomputed topological order.
    ///
    /// `helper_lookup` is P_{x_L}
    pub fn eval<E, PE>(
        &self,
        params: &E::Params,
        one: &E,
        inputs: &[E],
        plt_evaluator: Option<&PE>,
    ) -> Vec<E>
    where
        E: Evaluable<P = P>,
        PE: PltEvaluator<E>,
    {
        #[cfg(debug_assertions)]
        {
            assert_eq!(self.num_input(), inputs.len());
            assert_ne!(self.num_output(), 0);
        }

        let wires = DashMap::new();
        let levels = self.compute_levels();
        debug!("{}", format!("Levels: {levels:?}"));
        debug!("Levels are computed");

        wires.insert(GateId(0), one.clone());
        // Collect all input gate IDs excluding the reserved constant-one gate (0)
        let mut input_gate_ids: Vec<GateId> = self
            .gates
            .iter()
            .filter_map(|(id, gate)| match gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect();
        input_gate_ids.sort_by_key(|gid| gid.0);
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );
        for (id, input) in input_gate_ids.into_iter().zip(inputs.iter()) {
            wires.insert(id, input.clone());
            if let Some(prefix) = self.print_value.get(&id) {
                info!("{}", format!("[{prefix}] Gate ID {id}, {:?}", input));
            }
        }
        debug!("Input wires are set");

        let parallel_gates = crate::env::circuit_parallel_gates();
        let use_parallel = parallel_gates.map(|n| n != 1).unwrap_or(true);
        let eval_gate = |gate_id: GateId| {
            debug!("{}", format!("Gate id {gate_id} started"));
            if wires.contains_key(&gate_id) {
                debug!("{}", format!("Gate id {gate_id} already evaluated"));
                return;
            }
            let gate = self.gates.get(&gate_id).expect("gate not found").clone();
            debug!("Get gate");
            let result = match &gate.gate_type {
                PolyGateType::Input => {
                    panic!("Input gate {gate:?} should already be preloaded");
                }
                PolyGateType::Add => {
                    debug!("Add gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Add").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Add").clone();
                    let result = left + right;
                    debug!("Add gate end");
                    result
                }
                PolyGateType::Sub => {
                    debug!("Sub gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Sub").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Sub").clone();
                    let result = left - right;
                    debug!("Sub gate end");
                    result
                }
                PolyGateType::Mul => {
                    debug!("Mul gate start");
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Mul").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Mul").clone();
                    let result = left * right;
                    debug!("Mul gate end");
                    result
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let result = input.small_scalar_mul(params, scalar);
                    debug!("Large scalar mul gate end");
                    result
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for LargeScalarMul")
                        .clone();
                    let result = input.large_scalar_mul(params, scalar);
                    debug!("Large scalar mul gate end");
                    result
                }
                PolyGateType::Rotate { shift } => {
                    debug!("Rotate gate start");
                    let input =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Rotate").clone();
                    let result = input.rotate(params, *shift);
                    debug!("Rotate gate end");
                    result
                }
                PolyGateType::PubLut { lut_id } => {
                    debug!("Public Lookup gate start");
                    let input = wires
                        .get(&gate.input_gates[0])
                        .expect("wire missing for Public Lookup")
                        .clone();
                    let lookup = self.lookups.get(lut_id).expect("lookup table missing").as_ref();
                    let result = plt_evaluator
                        .expect("public lookup evaluator missing")
                        .public_lookup(params, lookup, one.clone(), input, gate_id, *lut_id);
                    debug!("Public Lookup gate end");
                    result
                }
            };
            if let Some(prefix) = self.print_value.get(&gate_id) {
                info!("{}", format!("[{prefix}] Gate ID {gate_id}, {:?}", result));
            }
            wires.insert(gate_id, result);
            debug!("{}", format!("Gate id {gate_id} finished"));
        };

        for level in levels.iter() {
            debug!("New level started");
            // All gates in the same level can be processed in parallel.
            if use_parallel {
                if let Some(chunk_size) = parallel_gates {
                    level.chunks(chunk_size).for_each(|chunk| {
                        chunk.par_iter().copied().for_each(|gate_id| eval_gate(gate_id));
                    });
                } else {
                    level.par_iter().copied().for_each(|gate_id| eval_gate(gate_id));
                }
                debug!("Evaluated gate in parallel");
            } else {
                level.iter().copied().for_each(|gate_id| eval_gate(gate_id));
                debug!("Evaluated gate in single thread");
            }
        }

        let outputs = if use_parallel {
            if let Some(chunk_size) = parallel_gates {
                let mut out = Vec::with_capacity(self.output_ids.len());
                for chunk in self.output_ids.chunks(chunk_size) {
                    let mut chunk_out: Vec<E> = chunk
                        .par_iter()
                        .map(|&id| wires.get(&id).expect("output missing").clone())
                        .collect();
                    out.append(&mut chunk_out);
                }
                out
            } else {
                self.output_ids
                    .par_iter()
                    .map(|&id| wires.get(&id).expect("output missing").clone())
                    .collect()
            }
        } else {
            self.output_ids
                .iter()
                .map(|&id| wires.get(&id).expect("output missing").clone())
                .collect()
        };
        debug!("Outputs are collected");
        outputs
    }

    pub fn register_public_lookup(&mut self, public_lookup: PublicLut<P>) -> usize {
        let plt_id = self.lookups.len();
        self.lookups.insert(plt_id, Arc::new(public_lookup));
        plt_id
    }

    pub fn register_sub_circuit(&mut self, sub_circuit: Self) -> usize {
        let circuit_id = self.sub_circuits.len();
        self.sub_circuits.insert(circuit_id, sub_circuit);
        circuit_id
    }

    /// Inlines the subcircuit operations directly into the main circuit instead of using call
    /// gates.
    pub fn call_sub_circuit(&mut self, circuit_id: usize, inputs: &[GateId]) -> Vec<GateId> {
        #[cfg(debug_assertions)]
        {
            let sub_circuit = &self.sub_circuits[&circuit_id];
            assert_eq!(inputs.len(), sub_circuit.num_input());
        }
        let mut gate_map: BTreeMap<GateId, GateId> = BTreeMap::new();
        let sub_circuit = self.sub_circuits.get(&circuit_id).unwrap().clone();
        // Map the reserved constant-one gate
        gate_map.insert(GateId(0), GateId(0));
        // Collect sub-circuit input gate IDs (excluding 0) in ascending order
        let mut sub_input_ids: Vec<GateId> = sub_circuit
            .gates
            .iter()
            .filter_map(|(id, gate)| match gate.gate_type {
                PolyGateType::Input if id.0 != 0 => Some(*id),
                _ => None,
            })
            .collect();
        sub_input_ids.sort_by_key(|gid| gid.0);
        assert_eq!(sub_input_ids.len(), inputs.len(), "sub-circuit input count mismatch");
        for (sub_in_id, &main_in_id) in sub_input_ids.into_iter().zip(inputs.iter()) {
            gate_map.insert(sub_in_id, main_in_id);
        }

        let mut outputs = Vec::with_capacity(sub_circuit.num_output());
        for &output_id in &sub_circuit.output_ids {
            let main_gate_id = self.inline_gate(output_id, &sub_circuit, &mut gate_map);
            outputs.push(main_gate_id);
        }
        outputs
    }

    /// Iteratively inlines a gate and its dependencies from a subcircuit into the main circuit.
    /// Returns the ID of the corresponding gate in the main circuit.
    fn inline_gate(
        &mut self,
        start_gate_id: GateId,
        sub_circuit: &PolyCircuit<P>,
        gate_map: &mut BTreeMap<GateId, GateId>,
    ) -> GateId {
        let mut stack = Vec::new();
        stack.push(start_gate_id);

        while let Some(&current_gate_id) = stack.last() {
            if gate_map.contains_key(&current_gate_id) {
                stack.pop();
                continue;
            }
            let gate = sub_circuit.gates.get(&current_gate_id).unwrap();
            let mut all_inputs_inlined = true;
            for &input_id in &gate.input_gates {
                if !gate_map.contains_key(&input_id) {
                    all_inputs_inlined = false;
                    stack.push(input_id);
                }
            }
            if all_inputs_inlined {
                let main_inputs: Vec<GateId> =
                    gate.input_gates.iter().map(|input_id| gate_map[input_id]).collect();
                let main_gate_id = self.new_gate_generic(main_inputs, gate.gate_type.clone());
                gate_map.insert(current_gate_id, main_gate_id);
                stack.pop();
            }
        }
        gate_map[&start_gate_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        rlwe_enc::rlwe_encrypt,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        utils::{create_bit_random_poly, create_random_poly},
    };
    use num_bigint::BigUint;

    #[test]
    fn test_eval_add() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with an Add operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 + poly2
        let expected = poly1 + poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_eval_sub() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with a Sub operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);
        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 - poly2
        let expected = poly1 - poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_mul() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with a Mul operation
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);
        circuit.output(vec![mul_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: poly1 * poly2
        let expected = poly1 * poly2;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_digits() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_bit_poly gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);

        // Define a specific bit pattern
        // This will create a polynomial with coefficients:
        // [1, 0, 1, 1]
        // (where 1 is at positions 0, 2, 3, and 4)
        let digits = vec![1u32, 0u32, 1u32, 1u32];
        let digits_poly_gate = circuit.const_digits(&digits);
        circuit.output(vec![digits_poly_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[dummy_input],
            None::<&PolyPltEvaluator>,
        );

        // Verify the result
        assert_eq!(result.len(), 1);

        // Check that the coefficients match the bit pattern
        let coeffs = result[0].coeffs();
        for (i, digit) in digits.iter().enumerate() {
            if digit != &0 {
                assert_eq!(
                    coeffs[i].value(),
                    &BigUint::from(1u8),
                    "Coefficient at position {} should be 1",
                    i
                );
            } else {
                assert_eq!(
                    coeffs[i].value(),
                    &BigUint::from(0u8),
                    "Coefficient at position {} should be 0",
                    i
                );
            }
        }

        // Check that remaining coefficients are 0
        for (i, _) in coeffs.iter().enumerate().skip(digits.len()) {
            assert_eq!(
                coeffs[i].value(),
                &BigUint::from(0u8),
                "Coefficient at position {} should be 0",
                i
            );
        }
    }

    #[test]
    fn test_eval_complex() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);

        // Create a complex circuit: (poly1 + poly2) - poly3
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(3);

        // poly1 + poly2
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);

        // (poly1 + poly2) - poly3
        let sub_gate = circuit.sub_gate(add_gate, inputs[2]);

        circuit.output(vec![sub_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone(), poly3.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (poly1 + poly2) - poly3
        let expected = (poly1 + poly2) - poly3;

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_multiple_outputs() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit with multiple outputs
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);

        // poly1 + poly2
        let add_gate = circuit.add_gate(inputs[0], inputs[1]);

        // poly1 - poly2
        let sub_gate = circuit.sub_gate(inputs[0], inputs[1]);

        // poly1 * poly2
        let mul_gate = circuit.mul_gate(inputs[0], inputs[1]);

        circuit.output(vec![add_gate, sub_gate, mul_gate]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected results
        let expected_add = poly1.clone() + poly2.clone();
        let expected_sub = poly1.clone() - poly2.clone();
        let expected_mul = poly1 * poly2;

        // Verify the results
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], expected_add);
        assert_eq!(result[1], expected_sub);
        assert_eq!(result[2], expected_mul);
    }

    #[test]
    fn test_multiple_input_calls_with_nonconsecutive_gate_ids() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a circuit
        let mut circuit = PolyCircuit::new();

        // First input call: creates GateId(1) as input
        let inputs_first = circuit.input(1);
        assert_eq!(inputs_first.len(), 1);

        // Insert a gate between input calls so next input gate is non-consecutive
        // Use a const-digits gate which introduces a new gate with no inputs
        let _const_gate = circuit.const_digits(&[1u32, 0u32, 1u32]);

        // Second input call: creates a new input gate with a higher, non-consecutive GateId
        let inputs_second = circuit.input(1);
        assert_eq!(inputs_second.len(), 1);

        // Ensure non-consecutive input GateIds (there should be a gap)
        assert_ne!(inputs_second[0].0, inputs_first[0].0 + 1);

        // Build a simple circuit that adds the two inputs together
        let add_gate = circuit.add_gate(inputs_first[0], inputs_second[0]);
        circuit.output(vec![add_gate]);

        // Evaluate the circuit: inputs are assigned in ascending input GateId order
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        let expected = poly1 + poly2;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_eval_deep_complex() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);
        let poly4 = create_random_poly(&params);

        // Create a complex circuit with depth = 4
        // Circuit structure:
        // Level 1: a = poly1 + poly2, b = poly3 * poly4, d = poly1 - poly3
        // Level 2: c = a * b
        // Level 3: e = c + d
        // Level 4: f = e * e
        // Output: f
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(4);

        // Level 1
        let a = circuit.add_gate(inputs[0], inputs[1]); // poly1 + poly2
        let b = circuit.mul_gate(inputs[2], inputs[3]); // poly3 * poly4
        let d = circuit.sub_gate(inputs[0], inputs[2]); // poly1 - poly3

        // Level 2
        let c = circuit.mul_gate(a, b); // (poly1 + poly2) * (poly3 * poly4)

        // Level 3
        let e = circuit.add_gate(c, d); // ((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3)

        // Level 4
        let f = circuit.mul_gate(e, e); // (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2

        circuit.output(vec![f]);

        // Evaluate the circuit
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone(), poly3.clone(), poly4.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (((poly1 + poly2) * (poly3 * poly4)) + (poly1 - poly3))^2
        let expected = (((poly1.clone() + poly2.clone()) * (poly3.clone() * poly4.clone())) +
            (poly1.clone() - poly3.clone())) *
            (((poly1.clone() + poly2) * (poly3.clone() * poly4)) + (poly1 - poly3));

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_boolean_gate_and() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let and_result = circuit.and_gate(inputs[0], inputs[1]);
        circuit.output(vec![and_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = poly1.clone() * poly2;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_not() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let not_result = circuit.not_gate(inputs[0]);
        circuit.output(vec![not_result]);
        let poly1 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            std::slice::from_ref(&poly1),
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) - poly1.clone();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_or() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let or_result = circuit.or_gate(inputs[0], inputs[1]);
        circuit.output(vec![or_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_nand() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nand_result = circuit.nand_gate(inputs[0], inputs[1]);
        circuit.output(vec![nand_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) - (poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_nor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nor_result = circuit.nor_gate(inputs[0], inputs[1]); // poly1 AND poly2
        circuit.output(vec![nor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected =
            DCRTPoly::const_one(&params) - ((poly1.clone() + poly2.clone()) - (poly1 * poly2));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_xor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let nor_result = circuit.xor_gate(inputs[0], inputs[1]);
        circuit.output(vec![nor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = (poly1.clone() + poly2.clone()) -
            (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_boolean_gate_xnor() {
        let params = DCRTPolyParams::default();
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let xnor_result = circuit.xnor_gate(inputs[0], inputs[1]);
        circuit.output(vec![xnor_result]);
        let poly1 = create_bit_random_poly(&params);
        let poly2 = create_bit_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );
        let expected = DCRTPoly::const_one(&params) -
            ((poly1.clone() + poly2.clone()) -
                (DCRTPoly::from_usize_to_constant(&params, 2) * poly1 * poly2));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coeffs(), expected.coeffs());
    }

    #[test]
    fn test_mul_fhe_poly_bits_mul_by_poly_circuit() {
        let mut circuit = PolyCircuit::new();
        let params = DCRTPolyParams::default();
        let sampler = DCRTPolyUniformSampler::new();
        let sigma = 3.0;
        let log_q = params.modulus_bits();

        // encrypt a polynomial m using FHE secret key encryption
        // Generate random message bits
        let m = sampler.sample_poly(&params, &DistType::BitDist);

        // Encrypt the message
        let a = sampler.sample_poly(&params, &DistType::BitDist);
        let t = sampler.sample_poly(&params, &DistType::BitDist);

        let m_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![m.clone()]);
        let a_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![a.clone()]);
        let t_mat = DCRTPolyMatrix::from_poly_vec_row(&params, vec![t.clone()]);
        let b_mat = rlwe_encrypt(&params, &sampler, &t_mat, &a_mat, &m_mat, sigma);
        let b = b_mat.entry(0, 0);

        // ct = (a, b)
        let a_bits = a.decompose_base(&params);
        let b_bits = b.decompose_base(&params);

        let x = DCRTPoly::const_one(&params);

        let inputs = circuit.input(a_bits.len() + b_bits.len() + 1);
        assert_eq!(inputs.len(), params.modulus_bits() * 2 + 1);

        // Input: ct[bits], x
        // Output: ct[bits] * x
        let x_id = inputs[inputs.len() - 1];
        let output_ids = inputs
            .iter()
            .take(inputs.len() - 1)
            .map(|&input_id| circuit.mul_gate(input_id, x_id))
            .collect();

        circuit.output(output_ids);

        // concatenate decomposed_c0 and decomposed_c1 and x
        let input = [a_bits, b_bits, vec![x.clone()]].concat();
        let result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &input, None::<&PolyPltEvaluator>);

        assert_eq!(result.len(), log_q * 2);

        let a_bits_eval = result[..params.modulus_bits()].to_vec();
        let b_bits_eval = result[params.modulus_bits()..].to_vec();

        let a_eval = DCRTPoly::from_decomposed(&params, &a_bits_eval);
        let b_eval = DCRTPoly::from_decomposed(&params, &b_bits_eval);

        assert_eq!(a_eval, &a * &x);
        assert_eq!(b_eval, &b * &x);

        // decrypt the result
        let plaintext = b_eval - (a_eval * t);
        // recover the message bits
        let plaintext_bits = plaintext.extract_bits_with_threshold(&params);
        assert_eq!(plaintext_bits, (m * x).to_bool_vec());
    }

    #[test]
    fn test_register_and_call_sub_circuit() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials using UniformSampler
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);

        // Create a sub-circuit that performs addition and multiplication
        let mut sub_circuit = PolyCircuit::new();
        let sub_inputs = sub_circuit.input(2);

        // Add operation: poly1 + poly2
        let add_gate = sub_circuit.add_gate(sub_inputs[0], sub_inputs[1]);

        // Mul operation: poly1 * poly2
        let mul_gate = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[1]);

        // Set the outputs of the sub-circuit
        sub_circuit.output(vec![add_gate, mul_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(2);

        // Register the sub-circuit and get its ID
        let sub_circuit_id = main_circuit.register_sub_circuit(sub_circuit);

        // Call the sub-circuit with the main circuit's inputs
        let sub_outputs =
            main_circuit.call_sub_circuit(sub_circuit_id, &[main_inputs[0], main_inputs[1]]);

        // Verify we got two outputs from the sub-circuit
        assert_eq!(sub_outputs.len(), 2);

        // Use the sub-circuit outputs for further operations
        // For example, subtract the multiplication result from the addition result
        let final_gate = main_circuit.sub_gate(sub_outputs[0], sub_outputs[1]);

        // Set the output of the main circuit
        main_circuit.output(vec![final_gate]);

        // Evaluate the main circuit
        let result = main_circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: (poly1 + poly2) - (poly1 * poly2)
        let expected = (poly1.clone() + poly2.clone()) - (poly1 * poly2);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_nested_sub_circuits() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create input polynomials
        let poly1 = create_random_poly(&params);
        let poly2 = create_random_poly(&params);
        let poly3 = create_random_poly(&params);

        // Create the innermost sub-circuit that performs multiplication
        let mut inner_circuit = PolyCircuit::new();
        let inner_inputs = inner_circuit.input(2);
        let mul_gate = inner_circuit.mul_gate(inner_inputs[0], inner_inputs[1]);
        inner_circuit.output(vec![mul_gate]);

        // Create a middle sub-circuit that uses the inner sub-circuit
        let mut middle_circuit = PolyCircuit::new();
        let middle_inputs = middle_circuit.input(3);

        // Register the inner circuit
        let inner_circuit_id = middle_circuit.register_sub_circuit(inner_circuit);

        // Call the inner circuit with the first two inputs
        let inner_outputs = middle_circuit
            .call_sub_circuit(inner_circuit_id, &[middle_inputs[0], middle_inputs[1]]);

        // Add the result of the inner circuit with the third input
        let add_gate = middle_circuit.add_gate(inner_outputs[0], middle_inputs[2]);
        middle_circuit.output(vec![add_gate]);

        // Create the main circuit
        let mut main_circuit = PolyCircuit::new();
        let main_inputs = main_circuit.input(3);

        // Register the middle circuit
        let middle_circuit_id = main_circuit.register_sub_circuit(middle_circuit);

        // Call the middle circuit with all inputs
        let middle_outputs = main_circuit
            .call_sub_circuit(middle_circuit_id, &[main_inputs[0], main_inputs[1], main_inputs[2]]);

        let scalar_mul_gate = main_circuit.mul_gate(middle_outputs[0], middle_outputs[0]);

        // Set the output of the main circuit
        main_circuit.output(vec![scalar_mul_gate]);

        // Evaluate the main circuit
        let result = main_circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[poly1.clone(), poly2.clone(), poly3.clone()],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: ((poly1 * poly2) + poly3)^2
        let expected =
            ((poly1.clone() * poly2.clone()) + poly3.clone()) * ((poly1 * poly2) + poly3);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_zero_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_zero_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let zero_gate = circuit.const_zero_gate();
        circuit.output(vec![zero_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[dummy_input],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: 0
        let expected = DCRTPoly::const_zero(&params);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_one_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_one_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let one_gate = circuit.const_one_gate();
        circuit.output(vec![one_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[dummy_input],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: 1
        let expected = DCRTPoly::const_one(&params);

        // Verify the result
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_const_minus_one_gate() {
        // Create parameters for testing
        let params = DCRTPolyParams::default();

        // Create a circuit with a const_minus_one_gate
        let mut circuit = PolyCircuit::new();
        // We need to call input() to initialize the circuit
        circuit.input(1);
        let minus_one_gate = circuit.const_minus_one_gate();
        circuit.output(vec![minus_one_gate]);

        // Evaluate the circuit with any input (it won't be used)
        let dummy_input = create_random_poly(&params);
        let result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[dummy_input],
            None::<&PolyPltEvaluator>,
        );

        // Expected result: -1
        // We can compute -1 as 0 - 1
        let expected = DCRTPoly::const_zero(&params) - DCRTPoly::const_one(&params);

        // verify
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_depth_zero_with_direct_input_output() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        circuit.output(vec![inputs[0]]);
        assert_eq!(circuit.depth(), 0);
    }

    #[test]
    fn test_depth_one_with_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        circuit.output(vec![add]);
        assert_eq!(circuit.depth(), 1);
    }

    #[test]
    fn test_depth_two_with_chain() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(3);
        let add = circuit.add_gate(inputs[0], inputs[1]);
        let mul = circuit.mul_gate(add, inputs[2]);
        circuit.output(vec![mul]);
        assert_eq!(circuit.depth(), 2);
    }
}
