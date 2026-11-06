//! Layered Boolean circuits from polynomial circuits written with Boolean idioms.
//!
//! The layered Boolean evaluators ([`super::evaluate_boolean_family`] and the BGG+ evaluators built
//! on [`super::evaluate_boolean_matrix_family`]) read every gate of layer `l` from layer `l - 1`
//! and know only the gates false, true, copy, NOT, AND, and XOR. This module lowers a
//! [`PolyCircuit`] whose wires carry bits and whose gates use only the idioms
//!
//! - AND: `x y`,
//! - NOT: `1 - x`,
//! - XOR: `x + y - 2 x y`, where the product may also be used elsewhere,
//! - the constants zero (`1 - 1`) and one,
//!
//! into that representation. Additions, subtractions, and small integer scalars are tracked as
//! affine forms over the Boolean values computed so far; a form becomes a gate only where a bit is
//! needed (a multiplication operand, a sub-circuit boundary, or an output), and it must then be
//! one of the idioms above. Products with a constant operand are folded. A gate's level is one
//! more than the largest level of its operands, and the layered circuit places each gate at its
//! level and copies every value forward until its last use.
//!
//! [`lower_to_boolean_layers`] expands every sub-circuit call and materializes the layers.
//! [`boolean_layering_bound`] computes the same depth without expansion: it lowers each sub-circuit
//! once per profile of input levels, so its cost follows the materialized size of the circuit
//! rather than its expanded size.

use super::{
    BooleanCircuitData, BooleanCircuitShape, BooleanGateData, BooleanGateKind, PolyCircuit,
    gate::{GateId, GateParamSource, PolyGateType},
};
use crate::poly::Poly;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Arc};

/// A Boolean value: a constant, or a circuit input or gate output identified by its node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Atom {
    Constant(bool),
    Node(u32),
}

#[derive(Clone, Copy, Debug)]
struct Node {
    level: u32,
    /// The operands of an AND gate, for recognizing `x + y - 2 x y`.
    and_operands: Option<(u32, u32)>,
}

/// The affine form `sum_i coefficient_i node_i + constant` of one polynomial-circuit wire.
#[derive(Clone, Debug)]
struct Form {
    terms: Vec<(u32, i64)>,
    constant: i64,
}

impl Form {
    fn constant(value: i64) -> Self {
        Self { terms: Vec::new(), constant: value }
    }

    fn atom(atom: Atom) -> Self {
        match atom {
            Atom::Constant(value) => Self::constant(i64::from(value)),
            Atom::Node(node) => Self { terms: vec![(node, 1)], constant: 0 },
        }
    }

    fn combine(&self, other: &Self, other_sign: i64) -> Self {
        let mut terms = self.terms.clone();
        terms.extend(
            other.terms.iter().map(|&(node, coefficient)| (node, other_sign * coefficient)),
        );
        terms.sort_unstable_by_key(|&(node, _)| node);
        let mut merged: Vec<(u32, i64)> = Vec::with_capacity(terms.len());
        for (node, coefficient) in terms {
            match merged.last_mut() {
                Some((last, sum)) if *last == node => *sum += coefficient,
                _ => merged.push((node, coefficient)),
            }
        }
        merged.retain(|&(_, coefficient)| coefficient != 0);
        Self { terms: merged, constant: self.constant + other_sign * other.constant }
    }

    fn scale(&self, scalar: i64) -> Self {
        if scalar == 0 {
            return Self::constant(0);
        }
        Self {
            terms: self.terms.iter().map(|&(node, c)| (node, c * scalar)).collect(),
            constant: self.constant * scalar,
        }
    }
}

/// The public part of an atom that determines the lowering of a sub-circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum AtomProfile {
    Constant(bool),
    Level(u32),
}

struct BodySummary {
    outputs: Vec<AtomProfile>,
    gate_count: u128,
}

type BodyMemo = HashMap<(usize, Vec<AtomProfile>), Arc<BodySummary>>;

struct Lowering {
    nodes: Vec<Node>,
    /// Every created gate as `(kind, output, left, right)` when the layers are materialized.
    gates: Option<Vec<(BooleanGateKind, u32, u32, u32)>>,
    gate_count: u128,
}

impl Lowering {
    fn new(materialize: bool) -> Self {
        Self { nodes: Vec::new(), gates: materialize.then(Vec::new), gate_count: 0 }
    }

    fn node(&mut self, level: u32, and_operands: Option<(u32, u32)>) -> u32 {
        let index = u32::try_from(self.nodes.len()).expect("Boolean lowering node index fits u32");
        self.nodes.push(Node { level, and_operands });
        index
    }

    fn atom_with_profile(&mut self, profile: AtomProfile) -> Atom {
        match profile {
            AtomProfile::Constant(value) => Atom::Constant(value),
            AtomProfile::Level(level) => Atom::Node(self.node(level, None)),
        }
    }

    fn profile(&self, atom: Atom) -> AtomProfile {
        match atom {
            Atom::Constant(value) => AtomProfile::Constant(value),
            Atom::Node(node) => AtomProfile::Level(self.nodes[node as usize].level),
        }
    }

    fn gate(&mut self, kind: BooleanGateKind, left: u32, right: Option<u32>) -> u32 {
        let level = self.nodes[left as usize]
            .level
            .max(right.map_or(0, |right| self.nodes[right as usize].level)) +
            1;
        let and_operands = (kind == BooleanGateKind::And).then(|| (left, right.unwrap_or(left)));
        let output = self.node(level, and_operands);
        if let Some(gates) = self.gates.as_mut() {
            gates.push((kind, output, left, right.unwrap_or(0)));
        }
        self.gate_count += 1;
        output
    }

    /// The Boolean value of an affine form, which must be one of the idioms of the module or the
    /// NOT `1 - g` of one.
    fn materialize(&mut self, form: &Form) -> Atom {
        if let Some(atom) = self.materialize_idiom(form) {
            return atom;
        }
        let complement = Form::constant(1).combine(form, -1);
        match self.materialize_idiom(&complement) {
            Some(Atom::Node(node)) => Atom::Node(self.gate(BooleanGateKind::Not, node, None)),
            _ => panic!("the affine form {form:?} is not a Boolean idiom"),
        }
    }

    fn materialize_idiom(&mut self, form: &Form) -> Option<Atom> {
        Some(match form.terms.as_slice() {
            [] if form.constant == 0 || form.constant == 1 => Atom::Constant(form.constant == 1),
            [(node, 1)] if form.constant == 0 => Atom::Node(*node),
            [(node, -1)] if form.constant == 1 => {
                Atom::Node(self.gate(BooleanGateKind::Not, *node, None))
            }
            [_, _, _] if form.constant == 0 => {
                let product = form.terms.iter().find(|&&(_, coefficient)| coefficient == -2);
                let operands = form
                    .terms
                    .iter()
                    .filter(|&&(_, coefficient)| coefficient == 1)
                    .map(|&(node, _)| node)
                    .collect::<Vec<_>>();
                match (product, operands.as_slice()) {
                    (Some(&(product, _)), &[left, right])
                        if self.nodes[product as usize].and_operands == Some((left, right)) ||
                            self.nodes[product as usize].and_operands == Some((right, left)) =>
                    {
                        Atom::Node(self.gate(BooleanGateKind::Xor, left, Some(right)))
                    }
                    _ => return None,
                }
            }
            _ => return None,
        })
    }

    fn and(&mut self, left: Atom, right: Atom) -> Form {
        match (left, right) {
            (Atom::Constant(false), _) | (_, Atom::Constant(false)) => Form::constant(0),
            (Atom::Constant(true), other) | (other, Atom::Constant(true)) => Form::atom(other),
            (Atom::Node(left), Atom::Node(right)) if left == right => Form::atom(Atom::Node(left)),
            (Atom::Node(left), Atom::Node(right)) => {
                Form::atom(Atom::Node(self.gate(BooleanGateKind::And, left, Some(right))))
            }
        }
    }

    fn lower_body<P: Poly>(
        &mut self,
        circuit: &PolyCircuit<P>,
        inputs: &[Atom],
        memo: &mut BodyMemo,
    ) -> Vec<Atom> {
        let input_ids = circuit.sorted_input_gate_ids();
        assert_eq!(input_ids.len(), inputs.len(), "Boolean lowering input arity");
        let gate_count = circuit.num_gates();
        let mut forms: Vec<Option<Form>> = vec![None; gate_count];
        let mut atoms: Vec<Option<Atom>> = vec![None; gate_count];
        forms[0] = Some(Form::constant(1));
        atoms[0] = Some(Atom::Constant(true));
        for (&gate_id, &atom) in input_ids.iter().zip(inputs) {
            forms[gate_id.0] = Some(Form::atom(atom));
            atoms[gate_id.0] = Some(atom);
        }
        let mut call_outputs: HashMap<usize, Vec<Atom>> = HashMap::new();
        for (&gate_id, gate) in circuit.gates_in_id_order() {
            let form = match &gate.gate_type {
                PolyGateType::Input => continue,
                PolyGateType::Add | PolyGateType::Sub => {
                    let sign = if gate.gate_type == PolyGateType::Add { 1 } else { -1 };
                    let left = forms[gate.input_gates[0].0].as_ref().expect("an evaluated input");
                    let right = forms[gate.input_gates[1].0].as_ref().expect("an evaluated input");
                    left.combine(right, sign)
                }
                PolyGateType::SmallScalarMul { scalar: GateParamSource::Const(scalar) }
                    if scalar.len() == 1 =>
                {
                    forms[gate.input_gates[0].0]
                        .as_ref()
                        .expect("an evaluated input")
                        .scale(i64::from(scalar[0]))
                }
                PolyGateType::Mul => {
                    let left = self.materialize_wire(gate.input_gates[0], &mut forms, &mut atoms);
                    let right = self.materialize_wire(gate.input_gates[1], &mut forms, &mut atoms);
                    self.and(left, right)
                }
                PolyGateType::SubCircuitOutput { call_id, output_idx, .. } => {
                    if !call_outputs.contains_key(call_id) {
                        let outputs =
                            self.lower_call(circuit, *call_id, &mut forms, &mut atoms, memo);
                        call_outputs.insert(*call_id, outputs);
                    }
                    Form::atom(call_outputs[call_id][*output_idx])
                }
                other => panic!("the gate {other:?} has no Boolean lowering"),
            };
            forms[gate_id.0] = Some(form);
        }
        circuit
            .output_gate_ids()
            .iter()
            .map(|&output| self.materialize_wire(output, &mut forms, &mut atoms))
            .collect()
    }

    /// The Boolean value of a wire. Later gates read the wire as this value, so an XOR idiom
    /// over a materialized operand is recognized.
    fn materialize_wire(
        &mut self,
        gate_id: GateId,
        forms: &mut [Option<Form>],
        atoms: &mut [Option<Atom>],
    ) -> Atom {
        if let Some(atom) = atoms[gate_id.0] {
            return atom;
        }
        let atom = self.materialize(forms[gate_id.0].as_ref().expect("an evaluated wire"));
        atoms[gate_id.0] = Some(atom);
        forms[gate_id.0] = Some(Form::atom(atom));
        atom
    }

    fn lower_call<P: Poly>(
        &mut self,
        circuit: &PolyCircuit<P>,
        call_id: usize,
        forms: &mut [Option<Form>],
        atoms: &mut [Option<Atom>],
        memo: &mut BodyMemo,
    ) -> Vec<Atom> {
        let info = circuit.sub_circuit_call_info(call_id);
        assert!(info.param_bindings.is_empty(), "Boolean lowering requires unparameterized calls");
        let inputs = info
            .inputs
            .iter()
            .flat_map(|batch| batch.gate_ids())
            .map(|input| self.materialize_wire(input, forms, atoms))
            .collect::<Vec<_>>();
        let sub_circuit = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
        if self.gates.is_some() {
            return self.lower_body(sub_circuit.as_ref(), &inputs, memo);
        }
        let profile = inputs.iter().map(|&atom| self.profile(atom)).collect::<Vec<_>>();
        let key = (Arc::as_ptr(&sub_circuit) as usize, profile);
        let summary = match memo.get(&key) {
            Some(summary) => Arc::clone(summary),
            None => {
                let mut child = Lowering::new(false);
                let child_inputs = key
                    .1
                    .iter()
                    .map(|&profile| child.atom_with_profile(profile))
                    .collect::<Vec<_>>();
                let outputs = child.lower_body(sub_circuit.as_ref(), &child_inputs, memo);
                let summary = Arc::new(BodySummary {
                    outputs: outputs.iter().map(|&atom| child.profile(atom)).collect(),
                    gate_count: child.gate_count,
                });
                memo.insert(key, Arc::clone(&summary));
                summary
            }
        };
        self.gate_count += summary.gate_count;
        summary.outputs.iter().map(|&profile| self.atom_with_profile(profile)).collect()
    }
}

/// The depth and size of the layered lowering of a polynomial circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanLayeringBound {
    /// The number of layers: the largest output level, and at least one.
    pub depth: usize,
    /// The circuit inputs, which form layer zero.
    pub input_width: usize,
    /// The created gates, counted with their multiplicity in the expanded circuit.
    pub gate_count: u128,
    pub output_count: usize,
}

impl BooleanLayeringBound {
    /// An upper bound on the width of every layer: a layer holds each input or gate output at
    /// most once, plus the constant outputs of the last layer.
    pub fn max_layer_width_bound(&self) -> u128 {
        self.input_width as u128 + self.gate_count + self.output_count as u128
    }
}

/// The depth and gate count of the layered lowering of `circuit`, without expanding its
/// sub-circuits.
pub fn boolean_layering_bound<P: Poly>(circuit: &PolyCircuit<P>) -> BooleanLayeringBound {
    let mut lowering = Lowering::new(false);
    let input_width = circuit.num_input();
    let inputs = (0..input_width).map(|_| Atom::Node(lowering.node(0, None))).collect::<Vec<_>>();
    let outputs = lowering.lower_body(circuit, &inputs, &mut BodyMemo::new());
    let depth = outputs
        .iter()
        .map(|&atom| match atom {
            Atom::Constant(_) => 1,
            Atom::Node(node) => lowering.nodes[node as usize].level.max(1),
        })
        .max()
        .unwrap_or(1) as usize;
    BooleanLayeringBound {
        depth,
        input_width,
        gate_count: lowering.gate_count,
        output_count: outputs.len(),
    }
}

/// A polynomial circuit lowered to layers: `circuit` selects the first output, and
/// `output_slots[i]` is the slot of output `i` in the last layer.
#[derive(Clone, Debug)]
pub struct BooleanLayering {
    pub shape: BooleanCircuitShape,
    pub circuit: BooleanCircuitData,
    pub output_slots: Vec<usize>,
}

impl BooleanLayering {
    /// The circuit data whose output is output `index` of the polynomial circuit.
    pub fn with_output(&self, index: usize) -> BooleanCircuitData {
        BooleanCircuitData {
            layers: self.circuit.layers.clone(),
            output_source: self.output_slots[index],
        }
    }
}

/// Lowers `circuit` to the layered Boolean representation, expanding every sub-circuit call.
/// Gates that no output uses are dropped.
pub fn lower_to_boolean_layers<P: Poly>(circuit: &PolyCircuit<P>) -> BooleanLayering {
    let mut lowering = Lowering::new(true);
    let input_width = circuit.num_input();
    let inputs = (0..input_width).map(|_| Atom::Node(lowering.node(0, None))).collect::<Vec<_>>();
    let outputs = lowering.lower_body(circuit, &inputs, &mut BodyMemo::new());
    let gates = lowering.gates.take().expect("the lowering materializes gates");
    let nodes = lowering.nodes;

    // The gate defining every node, and the layer up to which every node is read.
    let mut definitions: Vec<Option<(BooleanGateKind, u32, u32)>> = vec![None; nodes.len()];
    for &(kind, output, left, right) in &gates {
        definitions[output as usize] = Some((kind, left, right));
    }
    let depth = outputs
        .iter()
        .map(|&atom| match atom {
            Atom::Constant(_) => 1,
            Atom::Node(node) => nodes[node as usize].level.max(1),
        })
        .max()
        .unwrap_or(1);
    let mut read_until: Vec<Option<u32>> = vec![None; nodes.len()];
    let mut stack = Vec::new();
    for atom in &outputs {
        if let Atom::Node(node) = *atom {
            if read_until[node as usize].is_none() {
                stack.push(node);
            }
            read_until[node as usize] = Some(depth);
        }
    }
    while let Some(node) = stack.pop() {
        if let Some((kind, left, right)) = definitions[node as usize] {
            let reader_layer = nodes[node as usize].level - 1;
            let operands = match kind {
                BooleanGateKind::And | BooleanGateKind::Xor => vec![left, right],
                _ => vec![left],
            };
            for operand in operands {
                let until = &mut read_until[operand as usize];
                if until.is_none() {
                    stack.push(operand);
                }
                *until = Some(until.map_or(reader_layer, |current| current.max(reader_layer)));
            }
        }
    }

    // Layer `l` holds the used gates of level `l` and copies of every node read after it.
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); depth as usize + 1];
    for (node, until) in read_until.iter().enumerate() {
        if let Some(until) = *until {
            let level = nodes[node].level;
            for layer in level..=until {
                members[layer as usize].push(node as u32);
            }
        }
    }
    let mut slots: Vec<HashMap<u32, usize>> = members
        .par_iter()
        .map(|layer| layer.iter().enumerate().map(|(slot, &node)| (node, slot)).collect())
        .collect();
    // Layer zero is the circuit input in input order.
    slots[0] = (0..input_width).map(|input| (input as u32, input)).collect();
    let mut layers = (1..=depth as usize)
        .into_par_iter()
        .map(|layer| {
            members[layer]
                .iter()
                .map(|&node| {
                    let preceding = &slots[layer - 1];
                    match definitions[node as usize] {
                        Some((kind, left, right))
                            if nodes[node as usize].level as usize == layer =>
                        {
                            let (left, right) = match kind {
                                BooleanGateKind::And | BooleanGateKind::Xor => {
                                    (preceding[&left], preceding[&right])
                                }
                                _ => (preceding[&left], 0),
                            };
                            BooleanGateData { kind, left, right }
                        }
                        _ => BooleanGateData {
                            kind: BooleanGateKind::Copy,
                            left: preceding[&node],
                            right: 0,
                        },
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let last = layers.last_mut().expect("the lowering has at least one layer");
    let output_slots = outputs
        .iter()
        .map(|&atom| match atom {
            Atom::Node(node) => slots[depth as usize][&node],
            Atom::Constant(value) => {
                last.push(BooleanGateData {
                    kind: if value {
                        BooleanGateKind::ConstantTrue
                    } else {
                        BooleanGateKind::ConstantFalse
                    },
                    left: 0,
                    right: 0,
                });
                last.len() - 1
            }
        })
        .collect::<Vec<_>>();
    // A layer without members keeps the canonical constant-false gate.
    for layer in &mut layers {
        if layer.is_empty() {
            layer.push(BooleanGateData { kind: BooleanGateKind::ConstantFalse, left: 0, right: 0 });
        }
    }
    let max_layer_width = layers.iter().map(Vec::len).max().unwrap_or(1).max(input_width).max(1);
    let shape = BooleanCircuitShape {
        instance_width: input_width,
        witness_width: 0,
        depth: depth as usize,
        max_layer_width,
    };
    let circuit =
        BooleanCircuitData { layers, output_source: output_slots.first().copied().unwrap_or(0) };
    BooleanLayering { shape, circuit, output_slots }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_gadgets::{
        arith::{
            CarryArithPoly, CarryArithPolyContext, CrtWindow, DecomposeArithmeticGadget,
            MontgomeryPoly, MontgomeryPolyContext,
        },
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    };
    use mxx_backends::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly as _, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        utils::gen_biguint_for_modulus,
    };
    use num_bigint::BigUint;
    use std::sync::Arc;

    fn bits(value: &BigUint, width: usize) -> Vec<bool> {
        (0..width).map(|bit| value.bit(bit as u64)).collect()
    }

    fn evaluate_outputs(layering: &BooleanLayering, inputs: &[bool]) -> Vec<bool> {
        (0..layering.output_slots.len())
            .into_par_iter()
            .map(|output| {
                layering
                    .with_output(output)
                    .evaluate(&layering.shape, inputs, &[])
                    .expect("a valid lowered circuit")
            })
            .collect()
    }

    fn from_bits(bits: &[bool]) -> BigUint {
        bits.iter().rev().fold(BigUint::ZERO, |acc, &bit| (acc << 1u8) + u8::from(bit))
    }

    #[test]
    fn lowered_bit_arithmetic_matches_integer_arithmetic() {
        let width = 9;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(CarryArithPolyContext::setup(&mut circuit, 1, false));
        let left = CarryArithPoly::input(context.clone(), &mut circuit, width);
        let right = CarryArithPoly::input(context, &mut circuit, width);
        let sum = left.add(&right, &mut circuit);
        let product = left.mul(&right, &mut circuit, None);
        let (is_less, difference) = left.less_than(&right, &mut circuit);
        let selected = left.cmux(&right, is_less, &mut circuit);
        circuit.output(
            sum.limbs
                .iter()
                .chain(&product.limbs)
                .chain(std::iter::once(&is_less))
                .chain(&difference.limbs)
                .chain(&selected.limbs)
                .copied()
                .collect::<Vec<_>>(),
        );
        let layering = lower_to_boolean_layers(&circuit);
        let bound = boolean_layering_bound(&circuit);
        assert_eq!(bound.depth, layering.shape.depth);
        assert!(bound.max_layer_width_bound() >= layering.shape.max_layer_width as u128);

        let mut rng = rand::rng();
        let modulus = BigUint::from(1u32) << width;
        for _ in 0..8 {
            let a = gen_biguint_for_modulus(&mut rng, &modulus);
            let b = gen_biguint_for_modulus(&mut rng, &modulus);
            let inputs = bits(&a, width).into_iter().chain(bits(&b, width)).collect::<Vec<_>>();
            let outputs = evaluate_outputs(&layering, &inputs);
            let (sum, rest) = outputs.split_at(width + 1);
            let (product, rest) = rest.split_at(2 * width);
            let (less, rest) = rest.split_at(1);
            let (difference, selected) = rest.split_at(width);
            assert_eq!(from_bits(sum), &a + &b);
            assert_eq!(from_bits(product), &a * &b);
            assert_eq!(less[0], a < b);
            assert_eq!(from_bits(difference), (&a + &modulus - &b) % &modulus);
            assert_eq!(from_bits(selected), if a < b { a.clone() } else { b.clone() });
        }
    }

    #[test]
    fn lowered_montgomery_ring_gsw_product_matches_the_native_product() {
        let parameters = DCRTPolyParams::new(4, 1, 8, 4, None, None);
        let dimension = parameters.ring_dimension() as usize;
        let window = CrtWindow::full(1);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let arithmetic =
            Arc::new(MontgomeryPolyContext::setup(&mut circuit, &parameters, 1, false));
        let context = Arc::new(RingGswContext::from_arith_context(
            &mut circuit,
            &parameters,
            dimension,
            arithmetic.clone(),
            window,
        ));
        let left = RingGswCiphertext::<DCRTPoly, MontgomeryPoly<DCRTPoly>>::input(
            context.clone(),
            None,
            &mut circuit,
        );
        let right = RingGswCiphertext::input(context.clone(), None, &mut circuit);
        let product = left.mul(&right, &mut circuit);
        let output_entries = product.rows.iter().flatten().cloned().collect::<Vec<_>>();
        circuit.output(
            output_entries
                .iter()
                .flat_map(|entry| entry.value[0].iter().flat_map(|residue| residue.limbs.clone()))
                .collect::<Vec<_>>(),
        );
        let layering = lower_to_boolean_layers(&circuit);
        let bound = boolean_layering_bound(&circuit);
        assert_eq!(bound.depth, layering.shape.depth);
        assert!(bound.max_layer_width_bound() >= layering.shape.max_layer_width as u128);

        let width = context.width();
        let modulus = parameters.modulus();
        let limbs = arithmetic.num_limbs;
        let radix = BigUint::from(1u32) << limbs;
        let mut rng = rand::rng();
        let random_entries = |rng: &mut rand::rngs::ThreadRng| {
            (0..2 * width)
                .map(|_| {
                    (0..dimension)
                        .map(|_| gen_biguint_for_modulus(rng, modulus.as_ref()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let left_entries = random_entries(&mut rng);
        let right_entries = random_entries(&mut rng);
        let montgomery_bits =
            |coefficient: &BigUint| bits(&((coefficient * &radix) % modulus.as_ref()), limbs);
        let inputs = left_entries
            .iter()
            .chain(&right_entries)
            .flatten()
            .flat_map(montgomery_bits)
            .collect::<Vec<_>>();
        let outputs = evaluate_outputs(&layering, &inputs);

        let native = |entries: &[Vec<BigUint>]| {
            DCRTPolyMatrix::from_poly_vec(
                &parameters,
                entries
                    .chunks(width)
                    .map(|row| {
                        row.iter()
                            .map(|coefficients| DCRTPoly::from_biguints(&parameters, coefficients))
                            .collect()
                    })
                    .collect(),
            )
        };
        let expected = native(&left_entries) *
            MontgomeryPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
                &parameters,
                &arithmetic,
                &native(&right_entries),
                window,
            );
        let radix_inverse = mxx_backends::utils::mod_inverse_biguints(&radix, modulus.as_ref())
            .expect("the Montgomery radix is invertible");
        for (index, coefficients) in outputs.chunks(dimension * limbs).enumerate() {
            let actual = coefficients
                .chunks(limbs)
                .map(|residue| (from_bits(residue) * &radix_inverse) % modulus.as_ref())
                .collect::<Vec<_>>();
            assert_eq!(
                DCRTPoly::from_biguints(&parameters, &actual),
                expected.entry(index / width, index % width),
                "Ring-GSW product entry {index}"
            );
        }
    }
}
