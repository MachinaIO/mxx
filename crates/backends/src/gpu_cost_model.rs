//! Static prediction of the arithmetic the GPU backend issues for a graph.
//!
//! [`gpu_operation_counts`] walks a frozen graph the way the GPU lowering does
//! (`gpu_physical_lowering` and `gpu_physical_control`, single logical device) and
//! counts the work of every kernel it would emit, without planning or running anything on a
//! device. All counts are per one-limb polynomial of `n` coefficients, `L` being the operand
//! ring's CRT depth:
//!
//! | counter | meaning |
//! |---|---|
//! | `ntt_forward`, `ntt_inverse` | one-limb negacyclic transforms |
//! | `shoup_mul` | Shoup modular multiplies (one `__umul64hi` and two low 64-bit multiplies) on limbs with `q >= 2^31` |
//! | `shoup_mul32` | Shoup multiplies on limbs with `q < 2^31`, which run on 32-bit words |
//! | `wide_mul` | 64x64->128 products of `raw_matrix_mul_kernel` |
//! | `wide_mod_calls` | `raw_wide_mod` calls of that kernel (a 128-bit `%`, a 64-bit `%`, or nothing) |
//! | `mulmod_u64` | `mul_mod_u64` calls of the gadget-scale, 1x1-scalar, scale, and tensor kernels |
//! | `gadget_decompositions` | polynomials given to `GadgetDecompose` |
//! | `gadget_digit_polynomials`, `gadget_digit_coefficients` | digits it writes (shift, mask, and compare only) |
//! | `mul64_instructions` | `wide_mul + 3 * shoup_mul + mulmod_u64`, a weighted operation count |
//! | `unmodeled(<kind>)` | matrix-valued nodes whose work is not counted |
//! | `unsupported(<reason>)` | sites costed by extrapolating beyond what the backend executes |
//!
//! The lowering keeps every matrix in the evaluation (NTT) or coefficient domain, or compact for
//! bounded digits. The model tracks the same domain per value and charges a transform the first
//! time a consumer needs the other domain, sharing it through the same bidirectional conversion
//! cache: one cache for the root, inlined subgraph calls, and parallel lanes, and a fresh one per
//! sequential-loop iteration. Samples, constants, and lifted integers start in the coefficient
//! domain; imports, products, sums, selections, family packs, and child outputs are in the
//! evaluation domain.
//!
//! A parallel loop runs `ceil(count / W)` waves of `W` lanes, the tail wave included; `W` is the
//! loop count inside a sequential body and `min(wave_instances, count)` elsewhere. Lane 0 is
//! simulated first; when the body is index invariant every later lane costs what lane 1 costs.
//! A sequential body that is index invariant is simulated once and multiplied by its count.
//!
//! Extrapolations, each reported under `unsupported(..)`: rings above `n = 131072` use the
//! wide-stage NTT counts with additional stages, and a matrix-family sequential carry is costed
//! like a matrix carry. Sampling arithmetic and multi-device replicas are not modeled.

use mxx_ir_core::{
    ConcreteRing, FrozenGraphScopeId, Graph, GraphScope, NodeId, ParamEnv, Port, ShapeDependencies,
    WireRef, bind_integers, concretize_wire_type, loop_body_is_index_invariant,
    node::{ConstantMatrix, LoopInputMode, MatrixBinaryOp, NodeKind, ParallelLoop, SequentialLoop},
    types::ConcreteWireType,
};
use num_traits::ToPrimitive;
use std::{collections::BTreeMap, num::NonZeroU64};

/// The largest ring the GPU NTT accepts (`MatrixNTT.cu`).
const MAX_NTT_RING_DIMENSION: u32 = 1024 * 128;
/// The inner-product terms `raw_matrix_mul_kernel` accumulates per 128-bit reduction for moduli
/// of at most 61 bits.
const WIDE_MOD_BATCH: u128 = 63;
/// The inner-dimension slices of `raw_matrix_mul_kernel`.
const MUL_SLICES: u128 = 4;
/// The most lanes or iterations simulated one by one for a body that is not index invariant.
const MAX_ENUMERATED_INSTANCES: u128 = 1 << 20;

/// Predicted GPU arithmetic of `graph` under `env` (the module docs define every counter).
///
/// `wave_instances` is the parallel wave width of loops outside sequential bodies (`None`: one
/// wave), and `ntt_radix` the `MXX_GPU_NTT_RADIX` the kernels use (4 by default).
pub fn gpu_operation_counts(
    graph: &Graph,
    env: &ParamEnv,
    wave_instances: Option<NonZeroU64>,
    ntt_radix: u32,
) -> Result<BTreeMap<String, u128>, String> {
    if !ntt_radix.is_power_of_two() || !(2..=32).contains(&ntt_radix) {
        return Err(format!("GPU NTT radix {ntt_radix} is not a power of two in 2..=32"));
    }
    let mut model = Model {
        graph,
        dependencies: graph.shape_dependencies(),
        wave_instances,
        ntt_radix,
        values: Vec::new(),
        converted: BTreeMap::new(),
        counts: BTreeMap::new(),
        factor: 1,
        device_body: false,
    };
    let wires = model.lower_scope(&FrozenGraphScopeId::Root, env, &[])?;
    for root in graph.outputs().values() {
        let id = *wires.get(&root.value).ok_or("GPU cost model: an output has no value")?;
        model.need_eval(id)?;
    }
    let counter = |name: &str| model.counts.get(name).copied().unwrap_or(0);
    let mul64 = counter("shoup_mul")
        .checked_mul(3)
        .and_then(|shoup| shoup.checked_add(counter("wide_mul")))
        .and_then(|sum| sum.checked_add(counter("mulmod_u64")))
        .ok_or("GPU cost model: an operation count overflows u128")?;
    model.counts.insert("mul64_instructions".to_owned(), mul64);
    Ok(model.counts)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    Eval,
    Coeff,
    Compact,
    Other,
}

struct Value {
    form: Form,
    ty: ConcreteWireType,
}

struct Model<'a> {
    graph: &'a Graph,
    dependencies: BTreeMap<FrozenGraphScopeId, ShapeDependencies>,
    wave_instances: Option<NonZeroU64>,
    ntt_radix: u32,
    values: Vec<Value>,
    converted: BTreeMap<usize, usize>,
    counts: BTreeMap<String, u128>,
    /// How many times the operations being simulated run.
    factor: u128,
    device_body: bool,
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

/// The polynomials of a matrix-like or matrix-family type and their ring.
fn layout(ty: &ConcreteWireType) -> Option<(u128, &ConcreteRing)> {
    match ty {
        ConcreteWireType::IndexedFamily { element, count } => {
            layout(element).map(|(polys, ring)| (polys * *count as u128, ring))
        }
        ty => ty.matrix_type().map(|matrix| ((matrix.rows * matrix.columns) as u128, &matrix.ring)),
    }
}

/// The domain a producer of `ty` leaves a fresh value in, given its matrix domain.
fn form_for(ty: &ConcreteWireType, matrix: Form) -> Form {
    match ty {
        ConcreteWireType::Matrix(_) => matrix,
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => Form::Compact,
        ConcreteWireType::IndexedFamily { element, .. } => form_for(element, matrix),
        _ => Form::Other,
    }
}

fn kind_name(kind: &NodeKind) -> String {
    serde_json::to_value(kind)
        .ok()
        .and_then(|kind| kind["tag"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "Unknown".to_owned())
}

/// Shoup multiplies of one one-limb transform (`MatrixNTT.cu`, `raw_ntt_tile` dispatch).
fn ntt_multiplies(direction: Direction, n: u128, radix: u128) -> u128 {
    let log2 = |value: u128| u128::from(value.max(1).ilog2());
    match direction {
        Direction::Forward => n / 2 * log2(n) + n,
        Direction::Inverse if n <= 4096 && n / radix.min(n) <= 1024 => n / 2 * log2(n) + 2 * n,
        Direction::Inverse => {
            let tiles = n / 1024;
            if tiles <= 32 {
                5 * n + n * log2(tiles) + 2 * n
            } else {
                10 * n + n / 2 * log2(tiles / 32) + 2 * n
            }
        }
    }
}

/// `raw_wide_mod` calls per output limb coefficient of a product with inner dimension `inner`.
fn wide_mod_calls(inner: u128) -> u128 {
    (0..MUL_SLICES)
        .map(|slice| inner.saturating_sub(slice).div_ceil(MUL_SLICES) / WIDE_MOD_BATCH + 1)
        .sum()
}

impl Model<'_> {
    fn charge(&mut self, name: &str, amount: u128) -> Result<(), String> {
        let total = self.counts.entry(name.to_owned()).or_insert(0);
        *total = amount
            .checked_mul(self.factor)
            .and_then(|amount| total.checked_add(amount))
            .ok_or("GPU cost model: an operation count overflows u128")?;
        Ok(())
    }

    fn value(&mut self, form: Form, ty: ConcreteWireType) -> usize {
        self.values.push(Value { form, ty });
        self.values.len() - 1
    }

    fn charge_ntt(&mut self, direction: Direction, ty: &ConcreteWireType) -> Result<(), String> {
        let Some((polys, ring)) = layout(ty) else { return Ok(()) };
        let n = ring.ring_dimension();
        let moduli = ring.crt_moduli().to_vec();
        let per_limb = ntt_multiplies(direction, u128::from(n), u128::from(self.ntt_radix));
        let narrow = moduli.iter().filter(|&&q| q < 1 << 31).count() as u128;
        let wide = moduli.len() as u128 - narrow;
        let name = match direction {
            Direction::Forward => "ntt_forward",
            Direction::Inverse => "ntt_inverse",
        };
        self.charge(name, polys * moduli.len() as u128)?;
        self.charge("shoup_mul", polys * wide * per_limb)?;
        if narrow > 0 {
            self.charge("shoup_mul32", polys * narrow * per_limb)?;
        }
        if n > MAX_NTT_RING_DIMENSION {
            self.charge("unsupported(ring_dimension_above_131072)", 1)?;
        }
        Ok(())
    }

    /// `id` in the evaluation domain (`full_eval_value`).
    fn need_eval(&mut self, id: usize) -> Result<usize, String> {
        self.convert(id, Form::Coeff, Form::Eval, Direction::Forward)
    }

    /// `id` in the coefficient domain (`full_coeff_value`).
    fn need_coeff(&mut self, id: usize) -> Result<usize, String> {
        self.convert(id, Form::Eval, Form::Coeff, Direction::Inverse)
    }

    fn convert(
        &mut self,
        id: usize,
        from: Form,
        to: Form,
        direction: Direction,
    ) -> Result<usize, String> {
        if self.values[id].form != from {
            return Ok(id);
        }
        if let Some(&converted) = self.converted.get(&id) {
            return Ok(converted);
        }
        let ty = self.values[id].ty.clone();
        self.charge_ntt(direction, &ty)?;
        let converted = self.value(to, ty);
        self.converted.insert(id, converted);
        self.converted.insert(converted, id);
        Ok(converted)
    }

    fn matrix_product(
        &mut self,
        rows: usize,
        inner: usize,
        columns: usize,
        ring: &ConcreteRing,
    ) -> Result<(), String> {
        let limb_coefficients =
            (rows * columns) as u128 * u128::from(ring.ring_dimension()) * ring.crt_depth() as u128;
        self.charge("wide_mul", limb_coefficients * inner as u128)?;
        self.charge("wide_mod_calls", limb_coefficients * wide_mod_calls(inner as u128))
    }

    /// One `mul_mod_u64` per limb coefficient of `ty`.
    fn coefficientwise_multiply(&mut self, ty: &ConcreteWireType) -> Result<(), String> {
        if let Some((polys, ring)) = layout(ty) {
            let amount = polys * u128::from(ring.ring_dimension()) * ring.crt_depth() as u128;
            self.charge("mulmod_u64", amount)?;
        }
        Ok(())
    }

    fn lower_child(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: &[usize],
    ) -> Result<Vec<usize>, String> {
        let wires = self.lower_scope(scope_id, env, inputs)?;
        let scope = self.graph.scope(scope_id).ok_or("GPU cost model: a scope is missing")?;
        // Child boundaries hand matrices to their parent in the evaluation domain.
        scope
            .outputs()
            .iter()
            .map(|wire| {
                let id = *wires.get(wire).ok_or("GPU cost model: a child output has no value")?;
                self.need_eval(id)
            })
            .collect()
    }

    fn lower_scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: &[usize],
    ) -> Result<BTreeMap<WireRef, usize>, String> {
        let graph = self.graph;
        let scope = graph.scope(scope_id).ok_or("GPU cost model: a scope is missing")?;
        let mut types = BTreeMap::new();
        for (position, node) in scope.nodes().iter().enumerate() {
            for (port, ty) in node.output_types().iter().enumerate() {
                let node_id = NodeId(position as u64);
                let concrete = concretize_wire_type(ty, env, scope_id, node_id)
                    .map_err(|error| format!("GPU cost model: {error}"))?;
                types.insert(WireRef { node: node_id, port: Port(port as u32) }, concrete);
            }
        }
        let mut wires = scope.inputs().iter().copied().zip(inputs.iter().copied()).collect();
        for position in 0..scope.nodes().len() {
            self.lower_node(scope_id, scope, env, NodeId(position as u64), &types, &mut wires)?;
        }
        Ok(wires)
    }

    fn lower_node(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        env: &ParamEnv,
        node_id: NodeId,
        types: &BTreeMap<WireRef, ConcreteWireType>,
        wires: &mut BTreeMap<WireRef, usize>,
    ) -> Result<(), String> {
        let node = &scope.nodes()[node_id.0 as usize];
        let output = |port: u32| WireRef { node: node_id, port: Port(port) };
        let first = output(0);
        let arguments = scope.arguments(node).ok_or("GPU cost model: an argument is outside")?;
        let argument = |wires: &BTreeMap<WireRef, usize>, index: usize| {
            wires.get(&arguments[index]).copied().ok_or("GPU cost model: an argument has no value")
        };
        let matrix = |id: usize, values: &[Value]| {
            values[id].ty.matrix_type().cloned().ok_or("GPU cost model: an operand is no matrix")
        };
        let output_ty = || types.get(&first).cloned().ok_or("GPU cost model: no output type");
        let fresh = |model: &mut Self, wires: &mut BTreeMap<WireRef, usize>, form: Form| {
            for port in 0..node.output_types().len() as u32 {
                let ty = types[&output(port)].clone();
                let id = model.value(form_for(&ty, form), ty);
                wires.insert(output(port), id);
            }
        };
        match node.kind() {
            NodeKind::Input { .. } if wires.contains_key(&first) => {}
            NodeKind::Input { .. } => fresh(self, wires, Form::Eval),
            NodeKind::ConstantMatrix { .. } | NodeKind::CenteredRebase { .. }
                if lazy_gadget(scope, first, types) => {}
            NodeKind::ConstantMatrix { .. } |
            NodeKind::LiftIntegerToConstantPolynomial { .. } |
            NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } => fresh(self, wires, Form::Coeff),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
                if !wires.contains_key(&arguments[1]) &&
                    lazy_gadget(scope, arguments[1], types) =>
            {
                // `lower_gadget_product`: one gadget-scale multiply per output limb coefficient.
                let left = argument(wires, 0)?;
                self.need_eval(left)?;
                self.coefficientwise_multiply(&output_ty()?)?;
                fresh(self, wires, Form::Eval);
            }
            NodeKind::MatrixBinary(op) => {
                // `lower_matrix_node`: both operands in the evaluation domain.
                let left = argument(wires, 0)?;
                let right = argument(wires, 1)?;
                let (left, right) = (self.need_eval(left)?, self.need_eval(right)?);
                if *op == MatrixBinaryOp::Multiply {
                    let (lhs, rhs) = (matrix(left, &self.values)?, matrix(right, &self.values)?);
                    if lhs.columns == rhs.rows {
                        self.matrix_product(lhs.rows, lhs.columns, rhs.columns, &lhs.ring)?;
                    } else {
                        // The 1x1-scalar kernel scales the other operand.
                        self.coefficientwise_multiply(&output_ty()?)?;
                    }
                }
                fresh(self, wires, Form::Eval);
            }
            NodeKind::MatrixMulSmallRhs => {
                // `lower_matrix_mul_small_rhs`: every digit polynomial is transformed again.
                let left = argument(wires, 0)?;
                let left = self.need_eval(left)?;
                let right = argument(wires, 1)?;
                let right_ty = self.values[right].ty.clone();
                self.charge_ntt(Direction::Forward, &right_ty)?;
                let (lhs, rhs) = (matrix(left, &self.values)?, matrix(right, &self.values)?);
                self.matrix_product(lhs.rows, lhs.columns, rhs.columns, &lhs.ring)?;
                fresh(self, wires, Form::Eval);
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                // `lower_matrix_mul_accumulate`: a product and a scale per term.
                if *has_bias {
                    let bias = argument(wires, 2 * coefficients.len())?;
                    self.need_eval(bias)?;
                }
                for term in 0..coefficients.len() {
                    let left = argument(wires, 2 * term)?;
                    let right = argument(wires, 2 * term + 1)?;
                    let (left, right) = (self.need_eval(left)?, self.need_eval(right)?);
                    let (lhs, rhs) = (matrix(left, &self.values)?, matrix(right, &self.values)?);
                    self.matrix_product(lhs.rows, lhs.columns, rhs.columns, &lhs.ring)?;
                    self.coefficientwise_multiply(&output_ty()?)?;
                }
                fresh(self, wires, Form::Eval);
            }
            NodeKind::MatrixScale { .. } | NodeKind::MatrixNegate => {
                let input = argument(wires, 0)?;
                if matches!(node.kind(), NodeKind::MatrixScale { .. }) {
                    self.coefficientwise_multiply(&output_ty()?)?;
                }
                let form = self.values[input].form;
                fresh(self, wires, form);
            }
            NodeKind::GadgetDecompose { .. } => {
                // `lower_gadget_decompose`: shifts and masks of the coefficient domain.
                let source = argument(wires, 0)?;
                let source = self.need_coeff(source)?;
                let (polys, _) =
                    layout(&self.values[source].ty).ok_or("GPU cost model: no source")?;
                let output = output_ty()?;
                let (digits, ring) = layout(&output).ok_or("GPU cost model: no digits")?;
                let n = u128::from(ring.ring_dimension());
                self.charge("gadget_decompositions", polys)?;
                self.charge("gadget_digit_polynomials", digits)?;
                self.charge("gadget_digit_coefficients", digits * n)?;
                fresh(self, wires, Form::Compact);
            }
            NodeKind::Concat { .. } | NodeKind::Slice { .. } | NodeKind::Transpose => {
                for index in 0..arguments.len() {
                    let input = argument(wires, index)?;
                    self.need_eval(input)?;
                }
                fresh(self, wires, Form::Eval);
            }
            NodeKind::Tensor => {
                for index in 0..arguments.len() {
                    let input = argument(wires, index)?;
                    self.need_eval(input)?;
                }
                self.coefficientwise_multiply(&output_ty()?)?;
                fresh(self, wires, Form::Eval);
            }
            NodeKind::ExtractCoefficient { .. } | NodeKind::ThresholdDecode { .. } => {
                let input = argument(wires, 0)?;
                self.need_coeff(input)?;
                fresh(self, wires, Form::Other);
            }
            NodeKind::CrtRecompose { .. } => {
                for index in 0..arguments.len() {
                    let input = argument(wires, index)?;
                    self.need_coeff(input)?;
                }
                self.charge("unmodeled(CrtRecompose)", 1)?;
                fresh(self, wires, Form::Coeff);
            }
            NodeKind::Select { .. } | NodeKind::FamilyPack { .. } => {
                let first_member = usize::from(matches!(node.kind(), NodeKind::Select { .. }));
                for index in first_member..arguments.len() {
                    let input = argument(wires, index)?;
                    self.need_eval(input)?;
                }
                fresh(self, wires, Form::Eval);
            }
            NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyGetDynamic => {
                let family = argument(wires, 0)?;
                let form = self.values[family].form;
                fresh(self, wires, form);
            }
            NodeKind::SubgraphCall(call) => {
                let child = self.child(scope_id, node_id)?;
                let inputs = (0..arguments.len())
                    .map(|index| argument(wires, index))
                    .collect::<Result<Vec<_>, _>>()?;
                let env = bind_integers(env, &call.bindings).map_err(|error| error.to_string())?;
                let outputs = self.lower_child(&child, &env, &inputs)?;
                for (port, id) in outputs.into_iter().enumerate() {
                    wires.insert(output(port as u32), id);
                }
            }
            NodeKind::ParallelLoop(spec) => {
                let child = self.child(scope_id, node_id)?;
                let inputs = (0..arguments.len())
                    .map(|index| argument(wires, index))
                    .collect::<Result<Vec<_>, _>>()?;
                self.lower_parallel(spec, &child, env, &inputs)?;
                fresh(self, wires, Form::Eval);
            }
            NodeKind::SequentialLoop(spec) => {
                let child = self.child(scope_id, node_id)?;
                let inputs = (0..arguments.len())
                    .map(|index| argument(wires, index))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = self.lower_sequential(spec, &child, env, &inputs)?;
                for (port, id) in outputs.into_iter().enumerate() {
                    wires.insert(output(port as u32), id);
                }
            }
            kind => {
                let matrix_valued = node
                    .output_types()
                    .iter()
                    .enumerate()
                    .any(|(port, _)| layout(&types[&output(port as u32)]).is_some());
                if matrix_valued {
                    self.charge(&format!("unmodeled({})", kind_name(kind)), 1)?;
                }
                fresh(self, wires, Form::Other);
            }
        }
        Ok(())
    }

    fn child(
        &self,
        scope_id: &FrozenGraphScopeId,
        node_id: NodeId,
    ) -> Result<FrozenGraphScopeId, String> {
        self.graph
            .child_scope_id(scope_id, node_id)
            .ok_or_else(|| "GPU cost model: a structural node has no body".to_owned())
    }

    fn iterations(count: &mxx_ir_core::IntExpr, env: &ParamEnv) -> Result<u128, String> {
        count
            .evaluate(env)
            .map_err(|error| error.to_string())?
            .to_u128()
            .ok_or_else(|| "GPU cost model: a loop count is negative or too large".to_owned())
    }

    fn loop_env(
        env: &ParamEnv,
        slot: u32,
        index: u128,
        bindings: &[(String, mxx_ir_core::IntExpr)],
    ) -> Result<ParamEnv, String> {
        let mut env = env.clone();
        env.loop_indices.insert(slot, index.into());
        bind_integers(&env, bindings).map_err(|error| error.to_string())
    }

    /// `lower_parallel_loop`: lanes share the current conversion cache, and the template of `W`
    /// lanes replays once per wave.
    fn lower_parallel(
        &mut self,
        spec: &ParallelLoop,
        child: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: &[usize],
    ) -> Result<(), String> {
        let count = Self::iterations(&spec.count, env)?;
        if count == 0 {
            return Ok(());
        }
        let width = match (self.device_body, self.wave_instances) {
            (false, Some(width)) => u128::from(width.get()).min(count),
            _ => count,
        };
        let waves = count.div_ceil(width);
        let invariant = loop_body_is_index_invariant(
            &self.dependencies[child],
            spec.index_slot,
            &spec.bindings,
        );
        if !invariant && width > MAX_ENUMERATED_INSTANCES {
            return Err("GPU cost model: an index-dependent parallel body is too wide".into());
        }
        let outer = self.factor;
        let lane = |model: &mut Self, index: u128, factor: u128| {
            model.factor = factor;
            // Broadcast inputs are the parent's values; zipped ones are this lane's member.
            let lane_inputs = inputs
                .iter()
                .zip(&spec.input_modes)
                .map(|(&id, mode)| match (mode, &model.values[id].ty) {
                    (LoopInputMode::Broadcast, _) => id,
                    (_, ConcreteWireType::IndexedFamily { element, .. }) => {
                        let (form, ty) = (model.values[id].form, element.as_ref().clone());
                        model.value(form, ty)
                    }
                    _ => id,
                })
                .collect::<Vec<_>>();
            let env = Self::loop_env(env, spec.index_slot, index, &spec.bindings)?;
            model.lower_child(child, &env, &lane_inputs).map(|_| ())
        };
        let per_wave = |factor: u128| {
            factor.checked_mul(waves).ok_or_else(|| "GPU cost model: overflow".to_owned())
        };
        lane(self, 0, per_wave(outer)?)?;
        if width > 1 {
            if invariant {
                let later = per_wave(outer)?
                    .checked_mul(width - 1)
                    .ok_or("GPU cost model: an operation count overflows u128")?;
                lane(self, 1, later)?;
            } else {
                for index in 1..width {
                    lane(self, index, per_wave(outer)?)?;
                }
            }
        }
        self.factor = outer;
        Ok(())
    }

    /// `lower_sequential_loop`: the body runs per iteration with a fresh conversion cache.
    fn lower_sequential(
        &mut self,
        spec: &SequentialLoop,
        child: &FrozenGraphScopeId,
        env: &ParamEnv,
        inputs: &[usize],
    ) -> Result<Vec<usize>, String> {
        let count = Self::iterations(&spec.count, env)?;
        let (carried, broadcast) = inputs.split_at(spec.carried_count);
        if count == 0 {
            // A zero-iteration loop hands its initial state through unchanged.
            return Ok(carried.to_vec());
        }
        let mut state = Vec::with_capacity(carried.len());
        for &id in carried {
            let ty = self.values[id].ty.clone();
            let id = match &ty {
                ConcreteWireType::Matrix(_) => {
                    self.need_eval(id)?;
                    self.value(Form::Eval, ty)
                }
                ConcreteWireType::IndexedFamily { element, .. } if layout(element).is_some() => {
                    // `allocate_sequential_carry` rejects matrix-family carries.
                    self.charge("unsupported(sequential_family_carry)", 1)?;
                    let form = form_for(&ty, Form::Eval);
                    if form == Form::Eval {
                        self.need_eval(id)?;
                    }
                    self.value(form, ty)
                }
                _ => {
                    let form = self.values[id].form;
                    self.value(form, ty)
                }
            };
            state.push(id);
        }
        let invariant = loop_body_is_index_invariant(
            &self.dependencies[child],
            spec.index_slot,
            &spec.bindings,
        );
        if !invariant && count > MAX_ENUMERATED_INSTANCES {
            return Err("GPU cost model: an index-dependent sequential body runs too often".into());
        }
        let (outer_factor, outer_device) = (self.factor, self.device_body);
        let outer_cache = std::mem::take(&mut self.converted);
        self.device_body = true;
        let iterations = if invariant { 1 } else { count };
        if invariant {
            self.factor = outer_factor
                .checked_mul(count)
                .ok_or("GPU cost model: an operation count overflows u128")?;
        }
        let mut result = Ok(());
        for index in 0..iterations {
            self.converted.clear();
            let mut body_inputs = state.clone();
            body_inputs.extend_from_slice(broadcast);
            let env = match Self::loop_env(env, spec.index_slot, index, &spec.bindings) {
                Ok(env) => env,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            };
            match self.lower_child(child, &env, &body_inputs) {
                Ok(outputs) => state = outputs,
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.converted = outer_cache;
        (self.factor, self.device_body) = (outer_factor, outer_device);
        result.map(|()| state)
    }
}

/// The gadget constant a lazy gadget read through `wire` comes from, and
/// whether a centered rebase lifts it into `wire`'s ring.
pub(crate) fn gadget_origin(scope: &GraphScope, wire: WireRef) -> Option<(WireRef, bool)> {
    let is_gadget = |wire: WireRef| {
        scope.nodes().get(wire.node.0 as usize).is_some_and(|node| {
            matches!(
                node.kind(),
                NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. }
            )
        })
    };
    if is_gadget(wire) {
        return Some((wire, false));
    }
    let node = scope.nodes().get(wire.node.0 as usize)?;
    match (node.kind(), scope.arguments(node)?.as_slice()) {
        (NodeKind::CenteredRebase { .. }, [source]) if is_gadget(*source) => Some((*source, true)),
        _ => None,
    }
}

/// Whether `wire` has readers in `scope`, each using it as the right factor
/// of a product with a one-row left factor, and is not a scope output.
fn read_by_row_products(
    scope: &GraphScope,
    wire: WireRef,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> bool {
    let Some(right) = wire_types.get(&wire).and_then(ConcreteWireType::matrix_type) else {
        return false;
    };
    if scope.outputs().contains(&wire) {
        return false;
    }
    let mut read = false;
    for consumer in scope.nodes() {
        let Some(arguments) = scope.arguments(consumer) else { return false };
        if !arguments.contains(&wire) {
            continue;
        }
        read = true;
        let row_product = matches!(
            consumer.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) && matches!(arguments.as_slice(), [left, right_wire] if *right_wire == wire && *left != wire &&
        wire_types.get(left).and_then(ConcreteWireType::matrix_type).is_some_and(|left| {
            left.rows == 1 && left.columns == right.rows && left.ring == right.ring
        }));
        if !row_product {
            return false;
        }
    }
    read
}

/// Whether `wire`, a gadget constant or its centered rebase, is never stored:
/// every product reading it scales its row by the gadget's entries (see
/// `lower_gadget_product`). A gadget constant qualifies when each reader is
/// such a product or a rebase that qualifies.
pub(crate) fn lazy_gadget(
    scope: &GraphScope,
    wire: WireRef,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> bool {
    match gadget_origin(scope, wire) {
        Some((_, true)) => read_by_row_products(scope, wire, wire_types),
        Some((_, false)) => {
            if scope.outputs().contains(&wire) {
                return false;
            }
            let mut read = false;
            for (index, consumer) in scope.nodes().iter().enumerate() {
                let Some(arguments) = scope.arguments(consumer) else { return false };
                if !arguments.contains(&wire) {
                    continue;
                }
                read = true;
                let rebase = WireRef { node: NodeId(index as u64), port: Port(0) };
                let lazy_rebase = matches!(consumer.kind(), NodeKind::CenteredRebase { .. }) &&
                    read_by_row_products(scope, rebase, wire_types);
                if !lazy_rebase && !read_by_row_products(scope, wire, wire_types) {
                    return false;
                }
            }
            read
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, GraphValue, Mat, Ring, Subgraph, iterate, parallel};

    /// n = 8 with two 33-bit limbs: 16 limb coefficients per polynomial.
    fn ring() -> Ring {
        let moduli = mxx_ir_core::generate_crt_basis(8, 2, 33).unwrap();
        Ring::from_crt_moduli(moduli.into_iter().map(Into::into).collect(), 8)
    }

    fn counts(output: Mat, wave_instances: Option<u64>) -> BTreeMap<String, u128> {
        let built = DslContext::new("cost").output("out", output).unwrap().build().unwrap();
        gpu_operation_counts(
            &built.graph,
            &ParamEnv::default(),
            wave_instances.map(|width| NonZeroU64::new(width).unwrap()),
            4,
        )
        .unwrap()
    }

    fn get(counts: &BTreeMap<String, u128>, name: &str) -> u128 {
        counts.get(name).copied().unwrap_or(0)
    }

    #[test]
    fn ntt_and_reduction_counts_follow_the_kernel_dispatch() {
        assert_eq!(ntt_multiplies(Direction::Forward, 8, 4), 4 * 3 + 8);
        assert_eq!(ntt_multiplies(Direction::Inverse, 8, 4), 4 * 3 + 16);
        assert_eq!(ntt_multiplies(Direction::Inverse, 4096, 4), 2048 * 12 + 2 * 4096);
        // Radix 2 tiles 4096: 5n in the blocked tiles, n log2(4) in the fused top, 2n at the end.
        assert_eq!(ntt_multiplies(Direction::Inverse, 4096, 2), 5 * 4096 + 2 * 4096 + 2 * 4096);
        assert_eq!(ntt_multiplies(Direction::Inverse, 8192, 4), 5 * 8192 + 3 * 8192 + 2 * 8192);
        assert_eq!(ntt_multiplies(Direction::Inverse, 65536, 4), 10 * 65536 + 32768 + 2 * 65536);
        assert_eq!(ntt_multiplies(Direction::Forward, 65536, 4), 32768 * 16 + 65536);
        // Four slices, each with one final reduction and one more per 63 terms.
        for (inner, calls) in [(1, 4), (3, 4), (252, 8), (253, 8), (504, 12)] {
            assert_eq!(wide_mod_calls(inner), calls, "inner dimension {inner}");
        }
    }

    #[test]
    fn products_count_wide_multiplies_and_the_scalar_and_gadget_paths_count_mulmods() {
        let ring = ring();
        let (a, b) = (ring.input("a", (2, 3)), ring.input("b", (3, 4)));
        let product = counts(&a * &b, None);
        assert_eq!(get(&product, "wide_mul"), 2 * 3 * 4 * 16);
        assert_eq!(get(&product, "wide_mod_calls"), 2 * 4 * 16 * 4);
        assert_eq!(get(&product, "ntt_forward"), 0);
        let scalar = counts(ring.input("x", (1, 1)) * ring.input("y", (4, 1)), None);
        assert_eq!((get(&scalar, "mulmod_u64"), get(&scalar, "wide_mul")), (4 * 16, 0));
        let gadget = counts(ring.input("row", (1, 2)) * ring.gadget(2, 4, 3), None);
        assert_eq!((get(&gadget, "mulmod_u64"), get(&gadget, "ntt_forward")), (6 * 16, 0));
    }

    #[test]
    fn conversions_are_shared_through_the_cache_and_inlined_calls() {
        let ring = ring();
        let (a, b) = (ring.input("a", (1, 1)), ring.input("b", (1, 1)));
        let sample = ring.uniform_residue((1, 1));
        let shared = counts(&sample * &a + &sample * &b, None);
        assert_eq!(get(&shared, "ntt_forward"), 2);
        let square = Subgraph::<Mat, Mat>::define("square", a.schema(), |x| Ok(&x * &x)).unwrap();
        let calls =
            counts(square.call(sample.clone()).unwrap() + square.call(sample).unwrap(), None);
        assert_eq!(get(&calls, "ntt_forward"), 2);
        // A decomposed sample is already in the coefficient domain.
        let decomposed = ring.uniform_residue((1, 1)).decompose(4, 34);
        let digits = counts(ring.input("row", (1, 34)).mul_small_rhs(decomposed), None);
        assert_eq!(get(&digits, "ntt_inverse"), 0);
        assert_eq!(get(&digits, "ntt_forward"), 34 * 2);
        assert_eq!(get(&digits, "gadget_digit_coefficients"), 34 * 8);
    }

    #[test]
    fn small_rhs_products_transform_their_digits_every_time() {
        let ring = ring();
        let digits = ring.input("key", (1, 2)).decompose(4, 17);
        let (left, right) = (ring.input("l", (1, 17)), ring.input("r", (2, 17)));
        let both = counts(
            Mat::concat(
                mxx_dsl::ConcatAxis::Rows,
                vec![left.mul_small_rhs(digits.clone()), right.mul_small_rhs(digits)],
            ),
            None,
        );
        assert_eq!(get(&both, "ntt_inverse"), 2 * 2);
        assert_eq!(get(&both, "ntt_forward"), 2 * 17 * 2 * 2);
        assert_eq!(get(&both, "wide_mul"), 3 * 17 * 2 * 16);
    }

    #[test]
    fn loops_charge_broadcast_conversions_per_iteration_or_wave() {
        let ring = ring();
        let a = ring.input("a", (1, 1));
        let sample = ring.uniform_residue((1, 1));
        let sequential =
            counts(iterate(5, a.clone(), |_, state| Ok(&state * &sample)).unwrap(), None);
        assert_eq!(get(&sequential, "ntt_forward"), 5 * 2);
        assert_eq!(get(&sequential, "wide_mul"), 5 * 16);
        let lanes = || parallel(10, |_| Ok(&a * &sample)).unwrap().at(0);
        for (width, waves) in [(None, 1), (Some(3), 4), (Some(20), 1)] {
            let parallel = counts(lanes(), width);
            assert_eq!(get(&parallel, "ntt_forward"), waves * 2, "width {width:?}");
            assert_eq!(
                get(&parallel, "wide_mul"),
                waves * width.unwrap_or(10).min(10) as u128 * 16
            );
        }
        // Inside a sequential body every lane runs once, whatever the wave width.
        let nested = iterate(2, a.clone(), |_, state| {
            Ok(parallel(10, |_| Ok(&state * &sample)).unwrap().at(0))
        })
        .unwrap();
        let nested = counts(nested, Some(3));
        assert_eq!((get(&nested, "ntt_forward"), get(&nested, "wide_mul")), (2 * 2, 2 * 10 * 16));
    }

    #[test]
    fn carries_keep_their_domain_and_empty_loops_cost_nothing() {
        let ring = ring();
        let sample = ring.uniform_residue((1, 1));
        let empty = iterate(0, sample.clone(), |_, state| Ok(&state * &state)).unwrap();
        let empty = counts(ring.input("row", (1, 34)).mul_small_rhs(empty.decompose(4, 34)), None);
        assert_eq!((get(&empty, "ntt_inverse"), get(&empty, "wide_mul")), (0, 34 * 16));
        let carried = iterate(3, sample, |_, state| Ok(-state)).unwrap();
        let carried = counts(carried, None);
        assert_eq!(get(&carried, "ntt_forward"), 2);
    }

    #[test]
    fn unmodeled_and_tensor_nodes_are_reported() {
        let ring = ring();
        let a = ring.input("a", (1, 2));
        let rotated = counts(a.clone().ring_automorphism(3), None);
        assert_eq!(get(&rotated, "unmodeled(RingAutomorphism)"), 1);
        let tensor = counts(a.clone().tensor(ring.input("b", (2, 1))), None);
        assert_eq!(get(&tensor, "mulmod_u64"), 4 * 16);
        let scaled = counts(a * mxx_ir_core::IntExpr::constant(3), None);
        assert_eq!(get(&scaled, "mulmod_u64"), 2 * 16);
    }
}
