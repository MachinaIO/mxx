use crate::{
    circuit::{
        ArithmeticCircuitLowering, BatchedWire, CircuitLoweringTypes, GateInstance, PolyCircuit,
        PolyGateKind, PublicLookupLowering, SlotOperationLowering, SlotTransferSpec,
        StructuredCircuitLowering, SubCircuitParamSpec, SubCircuitParamValue, gate::GateId,
        lower_circuit_structured,
    },
    circuit_gadgets::{
        arith::{
            BinaryPlannerResult, CrtWindow, DecomposeArithmeticGadget, ModularArithmeticContext,
            ModularArithmeticGadget, ModularArithmeticPlanner,
        },
        conv_mul::{NegacyclicConvolutionContext, RingGswConvolution},
    },
};
use mxx_dsl::{ConcatAxis, DslContext, Family, GraphValue, Mat, Ring, Subgraph};
use mxx_ir_core::{IntExpr, ParamEnv, node::IndexRange, validate::ValidatedGraph};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
    transcript::SamplingMode,
};
use num_bigint::{BigInt, BigUint};
use std::{collections::BTreeMap, convert::Infallible, sync::Arc};

struct RuntimeLowering {
    ring: Ring,
    parameters: DCRTPolyParams,
    wire_size: usize,
}

impl CircuitLoweringTypes for RuntimeLowering {
    type Wire = Mat;
    type Error = Infallible;
}

impl ArithmeticCircuitLowering<DCRTPoly> for RuntimeLowering {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Mat,
        rhs: &Mat,
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        Ok(match operation {
            PolyGateKind::Add => lhs.clone() + rhs.clone(),
            PolyGateKind::Sub => lhs.clone() - rhs.clone(),
            PolyGateKind::Mul => lhs.clone() * rhs.clone(),
            _ => unreachable!("binary lowering receives only arithmetic operations"),
        })
    }

    fn small_scalar_mul(
        &mut self,
        input: &Mat,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        Ok(input.clone() * self.ring.polynomial(scalar.iter().copied().map(IntExpr::constant)))
    }

    fn large_scalar_mul(
        &mut self,
        input: &Mat,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        Ok(input.clone() *
            self.ring.polynomial(scalar.iter().cloned().map(BigInt::from).map(IntExpr::constant)))
    }
}

impl SlotOperationLowering<DCRTPoly> for RuntimeLowering {
    fn slot_transfer(
        &mut self,
        input: &Mat,
        source_slots: &[(u32, Option<u32>)],
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        let diagonal = source_slots
            .iter()
            .map(|(source, scalar)| {
                let source = usize::try_from(*source).expect("slot index fits in usize");
                assert!(source < source_slots.len(), "slot-transfer source is in range");
                let range = IndexRange { start: source.into(), end: (source + 1).into() };
                let entry = input.clone().slice(Some(range.clone()), Some(range));
                match scalar {
                    None | Some(1) => entry,
                    Some(scalar) => entry * self.ring.polynomial([IntExpr::constant(*scalar)]),
                }
            })
            .collect::<Vec<_>>();
        Ok(Mat::concat(ConcatAxis::Diagonal, diagonal))
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Mat],
        slot_count: usize,
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        assert!(slot_count > 0, "slot reduction requires a positive slot count");
        assert!(!inputs.is_empty(), "slot reduction requires an input");
        assert!(inputs.len() <= slot_count, "slot reduction has too many inputs");
        assert!(slot_count <= self.wire_size, "slot reduction exceeds the runtime wire size");

        let mut output_entries = inputs
            .iter()
            .map(|input| {
                (0..slot_count)
                    .map(|source| {
                        let range = IndexRange { start: source.into(), end: (source + 1).into() };
                        let selected = input.clone().slice(Some(range.clone()), Some(range));
                        let rotation =
                            self.ring.polynomial((0..self.parameters.ring_dimension()).map(
                                |coefficient| IntExpr::constant(coefficient == source as u32),
                            ));
                        selected * rotation
                    })
                    .reduce(|left, right| left + right)
                    .expect("positive slot count yields a reduction term")
            })
            .collect::<Vec<_>>();
        output_entries
            .extend((output_entries.len()..self.wire_size).map(|_| self.ring.zero((1, 1))));
        Ok(Mat::concat(ConcatAxis::Diagonal, output_entries))
    }
}

impl PublicLookupLowering<DCRTPoly> for RuntimeLowering {
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<DCRTPoly>,
        lookup_id: usize,
        input: &Mat,
        _gate: GateInstance<'_>,
    ) -> Result<Mat, Infallible> {
        let branches = circuit
            .lookup_table(lookup_id)
            .entries()
            .map(|(_input, (_row, output))| self.ring.polynomial([IntExpr::constant(output)]))
            .collect::<Vec<_>>();
        if self.wire_size == 1 {
            return Ok(input
                .clone()
                .extract_coefficient(0)
                .select(branches)
                .expect("public LUT branches share one matrix type"));
        }
        Ok(Mat::concat(
            ConcatAxis::Diagonal,
            (0..self.wire_size)
                .map(|index| {
                    let range = IndexRange { start: index.into(), end: (index + 1).into() };
                    input
                        .clone()
                        .slice(Some(range.clone()), Some(range))
                        .extract_coefficient(0)
                        .select(branches.clone())
                        .expect("public LUT branches share one matrix type")
                })
                .collect(),
        ))
    }
}

impl StructuredCircuitLowering<DCRTPoly> for RuntimeLowering {
    type Subgraph = Subgraph<Vec<Mat>, Vec<Mat>>;

    fn call_site_identity_is_semantic(&self) -> bool {
        false
    }

    fn define_subgraph<F>(
        &mut self,
        name: &str,
        input_examples: &[Mat],
        body: F,
    ) -> Result<Self::Subgraph, crate::circuit::CircuitLowerError<Self::Error>>
    where
        F: FnOnce(
            &mut Self,
            Vec<Mat>,
        ) -> Result<Vec<Mat>, crate::circuit::CircuitLowerError<Self::Error>>,
    {
        let schemas = input_examples.iter().map(GraphValue::schema).collect::<Vec<_>>();
        let mut body_error = None;
        let definition = Subgraph::define(name, schemas, |inputs| match body(self, inputs) {
            Ok(outputs) => outputs,
            Err(error) => {
                body_error = Some(error);
                Vec::new()
            }
        });
        if let Some(error) = body_error {
            return Err(error);
        }
        definition
            .map_err(|error| crate::circuit::CircuitLowerError::GraphStructure(error.to_string()))
    }

    fn call_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Mat>,
    ) -> Result<Vec<Mat>, crate::circuit::CircuitLowerError<Self::Error>> {
        definition
            .call(inputs)
            .map_err(|error| crate::circuit::CircuitLowerError::GraphStructure(error.to_string()))
    }

    fn call_audited_constant_lut_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Mat>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<Mat>, crate::circuit::CircuitLowerError<Self::Error>> {
        // This unit-test lowerer executes DSL values only; it emits no checker evidence, so the
        // producer contract has no runtime effect here. The subgraph call itself is preserved.
        drop(canonical_input_exclusive_uppers);
        self.call_subgraph(definition, inputs)
    }

    fn call_subgraph_parallel(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Vec<Mat>>,
    ) -> Result<Vec<Vec<Mat>>, crate::circuit::CircuitLowerError<Self::Error>> {
        let Some(first) = inputs.first() else {
            return Ok(Vec::new());
        };
        let input_count = first.len();
        if input_count == 0 || inputs.iter().any(|instance| instance.len() != input_count) {
            return Err(crate::circuit::CircuitLowerError::GraphStructure(
                "parallel subgraph calls require a non-empty rectangular input set".to_owned(),
            ));
        }
        let count = inputs.len();
        let mut inputs_by_position =
            (0..input_count).map(|_| Vec::with_capacity(count)).collect::<Vec<_>>();
        for instance in inputs {
            for (position, input) in instance.into_iter().enumerate() {
                inputs_by_position[position].push(input);
            }
        }
        let families = inputs_by_position
            .into_iter()
            .map(Family::pack)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                crate::circuit::CircuitLowerError::GraphStructure(error.to_string())
            })?;
        let definition = definition.clone();
        let output_families = Family::parallel_zip_many_values(families, move |_, child_inputs| {
            definition
                .call(child_inputs)
                .expect("parallel subgraph inputs preserve the validated definition schema")
        })
        .map_err(|error| crate::circuit::CircuitLowerError::GraphStructure(error.to_string()))?;
        let output_count = output_families.len();
        let mut outputs = (0..count).map(|_| Vec::with_capacity(output_count)).collect::<Vec<_>>();
        for family in output_families {
            for (index, instance) in outputs.iter_mut().enumerate() {
                instance.push(family.get_static(index));
            }
        }
        Ok(outputs)
    }

    fn call_audited_constant_lut_subgraph_parallel(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Vec<Mat>>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<Vec<Mat>>, crate::circuit::CircuitLowerError<Self::Error>> {
        // This unit-test lowerer executes DSL values only; it emits no checker evidence, so the
        // producer contract has no runtime effect here. The subgraph calls themselves are kept.
        drop(canonical_input_exclusive_uppers);
        self.call_subgraph_parallel(definition, inputs)
    }
}

pub fn constant_matrix(parameters: &DCRTPolyParams, value: usize) -> DCRTPolyMatrix {
    DCRTPolyMatrix::from_poly_vec_row(
        parameters,
        vec![DCRTPoly::from_usize_to_constant(parameters, value)],
    )
}

pub fn diagonal_matrix(
    parameters: &DCRTPolyParams,
    diagonal: impl IntoIterator<Item = DCRTPoly>,
) -> DCRTPolyMatrix {
    let diagonal = diagonal.into_iter().collect::<Vec<_>>();
    let zero = DCRTPoly::const_zero(parameters);
    DCRTPolyMatrix::from_poly_vec(
        parameters,
        (0..diagonal.len())
            .map(|row| {
                (0..diagonal.len())
                    .map(|column| if row == column { diagonal[row].clone() } else { zero.clone() })
                    .collect()
            })
            .collect(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyVec(pub Vec<DCRTPoly>);

impl PolyVec {
    pub fn to_diagonal_matrix(&self, parameters: &DCRTPolyParams) -> DCRTPolyMatrix {
        diagonal_matrix(parameters, self.0.iter().cloned())
    }

    pub fn from_diagonal_matrix(matrix: &DCRTPolyMatrix) -> Self {
        let (rows, columns) = matrix.size();
        assert_eq!(rows, columns, "PolyVec requires a square diagonal matrix");
        Self((0..rows).map(|slot| matrix.entry(slot, slot).clone()).collect())
    }
}

pub fn execute_polyvec_circuit(
    name: &str,
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    inputs: Vec<PolyVec>,
    wire_size: usize,
) -> Vec<PolyVec> {
    assert!(inputs.iter().all(|input| input.0.len() == wire_size));
    let matrices =
        inputs.iter().map(|input| input.to_diagonal_matrix(parameters)).collect::<Vec<_>>();
    execute_circuit_with_shape(name, parameters, circuit, &matrices, (wire_size, wire_size))
        .iter()
        .map(PolyVec::from_diagonal_matrix)
        .collect()
}

pub fn execute_circuit(
    name: &str,
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    inputs: &[DCRTPolyMatrix],
) -> Vec<DCRTPolyMatrix> {
    execute_circuit_with_shape(name, parameters, circuit, inputs, (1, 1))
}

pub fn execute_circuit_with_shape(
    name: &str,
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    inputs: &[DCRTPolyMatrix],
    shape: (usize, usize),
) -> Vec<DCRTPolyMatrix> {
    assert!(inputs.iter().all(|input| input.size() == shape), "runtime input shape mismatch");
    let graph = build_circuit_graph(name, parameters, circuit, inputs.len(), shape);
    let result = execute(
        &graph,
        &mut cpu_backend([parameters.clone()]),
        inputs
            .iter()
            .enumerate()
            .map(|(index, value)| (format!("input-{index}"), RuntimeValue::matrix(value.clone())))
            .collect::<BTreeMap<_, _>>(),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("execute runtime unit-test graph");
    (0..circuit.output_gate_ids().len())
        .map(|index| {
            let RuntimeValue::Matrix(value) = &result.outputs[&format!("output-{index}")] else {
                panic!("gadget output must be a matrix")
            };
            value.as_ref().clone()
        })
        .collect()
}

pub(crate) fn build_circuit_graph(
    name: &str,
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    input_count: usize,
    shape: (usize, usize),
) -> ValidatedGraph {
    assert_eq!(
        circuit.sorted_input_gate_ids().len(),
        input_count,
        "each concrete input must correspond to one circuit input wire"
    );
    assert_eq!(shape.0, shape.1, "runtime test wires use square matrices");
    let ring = Ring::new(
        BigInt::from(parameters.modulus().as_ref().clone()),
        parameters.ring_dimension() as usize,
    );
    let input_wires = (0..input_count)
        .map(|index| ring.input(format!("input-{index}"), shape))
        .collect::<Vec<_>>();
    let outputs = lower_circuit_structured(
        circuit,
        ring.identity(shape.0),
        input_wires,
        &mut RuntimeLowering { ring, parameters: parameters.clone(), wire_size: shape.0 },
    )
    .expect("runtime unit-test lowering is infallible");
    let mut context = DslContext::new(name);
    for (index, output) in outputs.into_iter().enumerate() {
        context = context
            .public_output(format!("output-{index}"), output)
            .expect("output names are unique");
    }
    context
        .build()
        .expect("build runtime unit-test graph")
        .validate(&ParamEnv::default())
        .expect("validate runtime unit-test graph")
}

#[derive(Clone, Debug)]
pub struct ScalarArithmeticContext {
    pub q_modulus: u64,
}

impl ModularArithmeticContext<DCRTPoly> for ScalarArithmeticContext {
    fn q_moduli_depth(&self) -> usize {
        1
    }

    fn decomposition_len(&self) -> usize {
        1
    }

    fn q_level_row_width(&self) -> usize {
        1
    }

    fn randomizer_decomposition_bound(&self) -> u64 {
        1
    }

    fn decomposition_term_bound(&self, term_idx: usize) -> BigUint {
        assert_eq!(term_idx, 0);
        BigUint::from(1u8)
    }

    fn plaintext_capacity_bound(&self) -> BigUint {
        BigUint::from(self.q_modulus)
    }
}

impl NegacyclicConvolutionContext<DCRTPoly> for ScalarArithmeticContext {
    fn q_level_diagonal_product_param_specs(&self) -> Vec<SubCircuitParamSpec> {
        vec![
            SubCircuitParamSpec::SlotTransfer {
                max_scalar: u32::try_from(self.q_modulus - 1).expect("test modulus fits in u32"),
            },
            SubCircuitParamSpec::SlotTransfer { max_scalar: 1 },
        ]
    }

    fn q_level_diagonal_product_param_bindings(
        &self,
        diagonal: usize,
        num_slots: usize,
    ) -> Vec<SubCircuitParamValue> {
        vec![
            SubCircuitParamValue::SlotTransfer(SlotTransferSpec::repeated(
                diagonal,
                num_slots,
                diagonal,
                Some(u32::try_from(self.q_modulus - 1).expect("test modulus fits in u32")),
            )),
            SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(diagonal, num_slots)),
        ]
    }

    fn reduce_q_level_row(
        &self,
        row: &[GateId],
        input_norms: &[BigUint],
        _circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (Vec<GateId>, Vec<BigUint>) {
        (row.to_vec(), input_norms.to_vec())
    }

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        _left_norms: &[BigUint],
        _right_norms: &[BigUint],
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Vec<GateId> {
        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        vec![circuit.mul_gate(left[0], right[0]).as_single_wire()]
    }
}

#[derive(Clone, Debug)]
pub struct ScalarArithmeticEntry {
    pub context: Arc<ScalarArithmeticContext>,
    pub wire: GateId,
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

impl ScalarArithmeticEntry {
    fn with_wire(&self, wire: GateId) -> Self {
        Self { wire, ..self.clone() }
    }
}

impl ModularArithmeticGadget<DCRTPoly> for ScalarArithmeticEntry {
    type Context = ScalarArithmeticContext;

    fn context(&self) -> &Arc<Self::Context> {
        &self.context
    }

    fn crt_window(&self) -> CrtWindow {
        CrtWindow::full(1)
    }

    fn max_plaintexts(&self) -> &[BigUint] {
        &self.max_plaintexts
    }

    fn p_max_traces(&self) -> &[BigUint] {
        &self.p_max_traces
    }

    fn input(
        context: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        Self::input_with_metadata(
            context,
            num_coefficient_slots,
            window,
            vec![BigUint::from(1u8)],
            vec![BigUint::from(1u8)],
            circuit,
        )
    }

    fn input_with_metadata(
        context: Arc<Self::Context>,
        _num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        assert_eq!(window, CrtWindow::full(1));
        Self { context, wire: circuit.input(1).as_single_wire(), max_plaintexts, p_max_traces }
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        vec![self.context.q_modulus]
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        vec![self.wire.into()]
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        assert_eq!(outputs.len(), 1);
        Self { context: template.context.clone(), wire: outputs[0], max_plaintexts, p_max_traces }
    }

    fn q_level_row_batch(&self, q_idx: usize) -> BatchedWire {
        assert_eq!(q_idx, 0);
        self.wire.into()
    }

    fn sparse_level_poly_with_metadata(
        context: Arc<Self::Context>,
        _num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        _circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        assert_eq!(window, CrtWindow::full(1));
        assert_eq!(target_q_idx, 0);
        Self {
            context,
            wire: target_row.as_single_wire(),
            max_plaintexts: vec![max_plaintext],
            p_max_traces: vec![p_max_trace],
        }
    }

    fn slot_transfer(
        &self,
        source_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        let lowered = source_slots
            .iter()
            .map(|(source, scalar)| {
                (
                    *source,
                    scalar.as_ref().map(|values| {
                        assert_eq!(values.len(), 1);
                        u32::try_from(values[0]).expect("test scalar fits in u32")
                    }),
                )
            })
            .collect::<Vec<_>>();
        let wire = circuit.slot_transfer_gate(self.wire, &lowered).as_single_wire();
        self.with_wire(wire)
    }

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        let mut result = self.with_wire(circuit.add_gate(self.wire, other.wire).as_single_wire());
        result.max_plaintexts = vec![&self.max_plaintexts[0] + &other.max_plaintexts[0]];
        result.p_max_traces = vec![&self.p_max_traces[0] + &other.p_max_traces[0]];
        result
    }

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        let mut result = self.with_wire(circuit.sub_gate(self.wire, other.wire).as_single_wire());
        result.max_plaintexts = vec![&self.max_plaintexts[0] + &other.max_plaintexts[0]];
        result.p_max_traces = vec![&self.p_max_traces[0] + &other.p_max_traces[0]];
        result
    }

    fn mul(&self, other: &Self, circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        let mut result = self.with_wire(circuit.mul_gate(self.wire, other.wire).as_single_wire());
        result.max_plaintexts = vec![&self.max_plaintexts[0] * &other.max_plaintexts[0]];
        result.p_max_traces = vec![&self.p_max_traces[0] * &other.p_max_traces[0]];
        result
    }

    fn full_reduce(&self, _circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        self.clone()
    }

    fn prepare_for_reconstruct(&self, _circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        self.clone()
    }

    fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        assert_eq!(tower_constants.len(), 1);
        self.with_wire(
            circuit
                .large_scalar_mul(self.wire, &[BigUint::from(tower_constants[0])])
                .as_single_wire(),
        )
    }

    fn reconstruct(&self, _circuit: &mut PolyCircuit<DCRTPoly>) -> GateId {
        self.wire
    }
}

impl ModularArithmeticPlanner<DCRTPoly> for ScalarArithmeticEntry {
    type Metadata = (Vec<BigUint>, Vec<BigUint>);
    type AddPlanKey = ();
    type SubPlanKey = ();

    fn metadata(entry: &Self) -> Self::Metadata {
        (entry.max_plaintexts.clone(), entry.p_max_traces.clone())
    }

    fn normalized_metadata(_context: &Self::Context, _window: CrtWindow) -> Self::Metadata {
        (vec![BigUint::from(1u8)], vec![BigUint::from(1u8)])
    }

    fn input_with_planner_metadata(
        context: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        metadata: &Self::Metadata,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        Self::input_with_metadata(
            context,
            num_coefficient_slots,
            window,
            metadata.0.clone(),
            metadata.1.clone(),
            circuit,
        )
    }

    fn from_flat_outputs_with_planner_metadata(
        template: &Self,
        outputs: &[GateId],
        metadata: &Self::Metadata,
    ) -> Self {
        Self::from_flat_outputs(template, outputs, metadata.0.clone(), metadata.1.clone())
    }

    fn compute_add_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::AddPlanKey, Self::Metadata> {
        BinaryPlannerResult {
            cache_key: (),
            output_metadata: (
                vec![&left.max_plaintexts[0] + &right.max_plaintexts[0]],
                vec![&left.p_max_traces[0] + &right.p_max_traces[0]],
            ),
        }
    }

    fn compute_sub_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::SubPlanKey, Self::Metadata> {
        Self::compute_add_plan_and_output(left, right)
    }

    fn normalize_mul_input(entry: &Self, _circuit: &mut PolyCircuit<DCRTPoly>) -> Self {
        entry.clone()
    }
}

impl DecomposeArithmeticGadget<DCRTPoly> for ScalarArithmeticEntry {
    fn gadget_matrix<M: PolyMatrix<P = DCRTPoly>>(
        parameters: &DCRTPolyParams,
        _context: &Self::Context,
        _window: CrtWindow,
    ) -> M {
        M::identity(parameters, 1, None)
    }

    fn gadget_decomposed<M: PolyMatrix<P = DCRTPoly>>(
        _parameters: &DCRTPolyParams,
        _context: &Self::Context,
        target: &M,
        _window: CrtWindow,
    ) -> M {
        target.clone()
    }

    fn gadget_decomposition_norm_bound(_context: &Self::Context, _window: CrtWindow) -> BigUint {
        BigUint::from(1u8)
    }

    fn gadget_vector(
        context: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Vec<Self> {
        vec![Self::input(context, num_coefficient_slots, window, circuit)]
    }

    fn gadget_decompose(&self, _circuit: &mut PolyCircuit<DCRTPoly>) -> Vec<Self> {
        vec![self.clone()]
    }

    fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        _circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (Vec<GateId>, GateId) {
        assert_eq!(q_idx, 0);
        (vec![self.wire], self.wire)
    }

    fn conv_mul_right_decomposed_many(
        &self,
        _parameters: &DCRTPolyParams,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Vec<Self> {
        assert_eq!(num_slots, 1);
        left_rows
            .iter()
            .map(|row| {
                assert_eq!(row.len(), 1);
                row[0].mul(self, circuit)
            })
            .collect()
    }
}

impl RingGswConvolution<DCRTPoly> for ScalarArithmeticEntry {
    fn q_level_row_max_plaintext_norms(&self, physical_q_row: usize) -> Vec<BigUint> {
        vec![self.p_max_traces[physical_q_row].clone()]
    }

    fn from_diagonal_q_level_outputs(
        template: &Self,
        q_level_outputs: Vec<Vec<BatchedWire>>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        assert_eq!(q_level_outputs.len(), 1);
        assert_eq!(q_level_outputs[0].len(), 1);
        Self {
            context: template.context.clone(),
            wire: q_level_outputs[0][0].as_single_wire(),
            max_plaintexts,
            p_max_traces,
        }
    }

    fn from_sparse_diagonal_q_level_output(
        template: &Self,
        target_q_idx: usize,
        q_level_output: Vec<BatchedWire>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        _circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> Self {
        assert_eq!(target_q_idx, 0);
        assert_eq!(q_level_output.len(), 1);
        Self {
            context: template.context.clone(),
            wire: q_level_output[0].as_single_wire(),
            max_plaintexts: vec![max_plaintext],
            p_max_traces: vec![p_max_trace],
        }
    }
}
