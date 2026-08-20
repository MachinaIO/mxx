//! PolyCircuit evaluation into declarative BGG+ DAG values.

use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire,
    EncodingCompileError, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
    NaiveBggSlotTransferCompiler, NaiveBggVecCompiler, NaiveVecCompileError,
    SlotFamilyCompileError,
    tall_encoding::{
        BggTallEncodingCompiler, BggTallEncodingWire, BggTallPlaintext, TallCompileError,
    },
};
use mxx_dsl::{GraphValue, Subgraph};
use mxx_gadgets::{
    Poly,
    circuit::{
        ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
        PolyCircuit, PolyGateKind, PublicLookupLowering, SlotOperationLowering,
        StructuredCircuitLowering, lower_circuit,
    },
};
use num_bigint::BigUint;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Clone)]
pub struct PolyCircuitCompiler {
    pub public_key: BggPublicKeyCompiler,
}

struct ConfiguredCircuitLowering<'a, A, L, S> {
    arithmetic: A,
    lookup: &'a mut L,
    slots: &'a mut S,
}

impl<A, L, S> CircuitLoweringTypes for ConfiguredCircuitLowering<'_, A, L, S>
where
    A: CircuitLoweringTypes,
{
    type Wire = A::Wire;
    type Error = A::Error;

    fn enter_subcircuit_inputs(
        &mut self,
        inputs: Vec<Self::Wire>,
        input_max_plaintext_norm_ranges: Option<
            &[mxx_gadgets::circuit::SubCircuitInputMaxPlaintextNormRange],
        >,
    ) -> Result<Vec<Self::Wire>, Self::Error> {
        self.arithmetic.enter_subcircuit_inputs(inputs, input_max_plaintext_norm_ranges)
    }
}

impl<P, A, L, S> ArithmeticCircuitLowering<P> for ConfiguredCircuitLowering<'_, A, L, S>
where
    P: Poly,
    A: ArithmeticCircuitLowering<P>,
    L: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
    S: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
{
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.arithmetic.binary(operation, lhs, rhs, gate)
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.arithmetic.small_scalar_mul(input, scalar, gate)
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.arithmetic.large_scalar_mul(input, scalar, gate)
    }
}

impl<P, A, L, S> SlotOperationLowering<P> for ConfiguredCircuitLowering<'_, A, L, S>
where
    P: Poly,
    A: CircuitLoweringTypes,
    L: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
    S: SlotOperationLowering<P, Wire = A::Wire, Error = A::Error>,
{
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.slots.slot_transfer(input, source_slots, gate)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.slots.slot_reduce(inputs, slot_count, gate)
    }

    fn slot_rotation(
        &mut self,
        input: &Self::Wire,
        offset: u32,
        num_slots: u32,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.slots.slot_rotation(input, offset, num_slots, gate)
    }
}

impl<P, A, L, S> PublicLookupLowering<P> for ConfiguredCircuitLowering<'_, A, L, S>
where
    P: Poly,
    A: CircuitLoweringTypes,
    L: PublicLookupLowering<P, Wire = A::Wire, Error = A::Error>,
    S: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
{
    fn public_lookup(
        &mut self,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        self.lookup.public_lookup(circuit, lookup_id, input, gate)
    }
}

impl<'a, P, A, L, S> StructuredCircuitLowering<P> for ConfiguredCircuitLowering<'a, A, L, S>
where
    P: Poly,
    A: CircuitLoweringTypes,
    A::Wire: GraphValue,
    L: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
    S: CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
    Self: mxx_gadgets::circuit::GraphCircuitLowering<P>
        + CircuitLoweringTypes<Wire = A::Wire, Error = A::Error>,
{
    type Subgraph = Subgraph<Vec<A::Wire>, Vec<A::Wire>>;

    fn define_subgraph<F>(
        &mut self,
        name: &str,
        input_examples: &[A::Wire],
        body: F,
    ) -> Result<Self::Subgraph, CircuitLowerError<Self::Error>>
    where
        F: FnOnce(&mut Self, Vec<A::Wire>) -> Result<Vec<A::Wire>, CircuitLowerError<Self::Error>>,
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
        definition.map_err(|error| CircuitLowerError::GraphStructure(error.to_string()))
    }

    fn call_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<A::Wire>,
    ) -> Result<Vec<A::Wire>, CircuitLowerError<Self::Error>> {
        definition
            .call(inputs)
            .map_err(|error| CircuitLowerError::GraphStructure(error.to_string()))
    }

    fn call_audited_constant_lut_subgraph(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<A::Wire>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<A::Wire>, CircuitLowerError<A::Error>> {
        definition
            .call_with_canonical_input_exclusive_uppers(inputs, canonical_input_exclusive_uppers)
            .map_err(|error| CircuitLowerError::GraphStructure(error.to_string()))
    }

    fn call_audited_constant_lut_subgraph_parallel(
        &mut self,
        definition: &Self::Subgraph,
        inputs: Vec<Vec<A::Wire>>,
        canonical_input_exclusive_uppers: Vec<Option<BigUint>>,
    ) -> Result<Vec<Vec<A::Wire>>, CircuitLowerError<A::Error>> {
        inputs
            .into_iter()
            .map(|inputs| {
                definition
                    .call_with_canonical_input_exclusive_uppers(
                        inputs,
                        canonical_input_exclusive_uppers.clone(),
                    )
                    .map_err(|error| CircuitLowerError::GraphStructure(error.to_string()))
            })
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum CircuitCompileError {
    #[error(transparent)]
    Encoding(#[from] EncodingCompileError),
    #[error(transparent)]
    NaiveVector(#[from] NaiveVecCompileError),
    #[error(transparent)]
    TallEncoding(#[from] TallCompileError),
    #[error(transparent)]
    SlotFamily(#[from] SlotFamilyCompileError),
    #[error(transparent)]
    Dsl(#[from] mxx_dsl::DslError),
    #[error(
        "gate {gate} uses a feature that is not represented by the initial declarative BGG evaluator: {feature}"
    )]
    Unsupported { gate: usize, feature: &'static str },
    #[error("circuit structure is invalid: {0}")]
    Structure(String),
    #[error("gate {gate} has an invalid slot-transfer layout")]
    InvalidSlotTransfer { gate: usize },
    #[error("slot-transfer artifact is missing: {name}")]
    MissingSlotTransferArtifact { name: String },
    #[error("tall rotation encoding ({num_slots}, {offset}) is unavailable")]
    MissingTallRotationEncoding { num_slots: u32, offset: u32 },
    #[error("gate {gate}: {source}")]
    LweLookup {
        gate: usize,
        #[source]
        source: crate::LweLookupCompileError,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct NoPublicLookup<W>(PhantomData<fn() -> W>);

impl<W> Default for NoPublicLookup<W> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: Clone> CircuitLoweringTypes for NoPublicLookup<W> {
    type Wire = W;
    type Error = CircuitCompileError;
}

impl<P: Poly, W: Clone> PublicLookupLowering<P> for NoPublicLookup<W> {
    fn public_lookup(
        &mut self,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        unsupported(gate, "public lookup")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NoSlotOperations<W>(PhantomData<fn() -> W>);

impl<W> Default for NoSlotOperations<W> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: Clone> CircuitLoweringTypes for NoSlotOperations<W> {
    type Wire = W;
    type Error = CircuitCompileError;
}

impl<P: Poly, W: Clone> SlotOperationLowering<P> for NoSlotOperations<W> {
    fn slot_transfer(
        &mut self,
        _input: &Self::Wire,
        _source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        unsupported(gate, "slot transfer")
    }

    fn slot_reduce(
        &mut self,
        _inputs: &[Self::Wire],
        _slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        unsupported(gate, "slot reduction")
    }
}

impl PolyCircuitCompiler {
    pub fn compile_public_keys_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggPublicKeyWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggPublicKeyWire, Error = CircuitCompileError>,
    {
        let arithmetic = PublicKeyLowering::<P> { compiler: &self.public_key, marker: PhantomData };
        let mut lowering = ConfiguredCircuitLowering { arithmetic, lookup, slots };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_tall_encodings_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggTallEncodingWire,
        inputs: impl IntoIterator<Item = BggTallEncodingWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<BggTallEncodingWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggTallEncodingWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggTallEncodingWire, Error = CircuitCompileError>,
    {
        let compiler = BggTallEncodingCompiler { public_key: self.public_key.clone() };
        let arithmetic = TallEncodingLowering::<P> { compiler: &compiler, marker: PhantomData };
        let mut lowering = ConfiguredCircuitLowering { arithmetic, lookup, slots };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_encodings_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggEncodingWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggEncodingWire, Error = CircuitCompileError>,
    {
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let arithmetic = EncodingLowering::<P> { compiler: &compiler, marker: PhantomData };
        let mut lowering = ConfiguredCircuitLowering { arithmetic, lookup, slots };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_naive_public_keys_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: NaiveBggPublicKeyVecWire,
        inputs: impl IntoIterator<Item = NaiveBggPublicKeyVecWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = NaiveBggPublicKeyVecWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = NaiveBggPublicKeyVecWire, Error = CircuitCompileError>,
    {
        let arithmetic = NaivePublicKeyLowering::<P> {
            compiler: NaiveBggVecCompiler { public_key: self.public_key.clone() },
            marker: PhantomData,
        };
        let mut lowering = ConfiguredCircuitLowering { arithmetic, lookup, slots };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_naive_encodings_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: NaiveBggEncodingVecWire,
        inputs: impl IntoIterator<Item = NaiveBggEncodingVecWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<NaiveBggEncodingVecWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = NaiveBggEncodingVecWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = NaiveBggEncodingVecWire, Error = CircuitCompileError>,
    {
        let arithmetic = NaiveEncodingLowering::<P> {
            compiler: NaiveBggVecCompiler { public_key: self.public_key.clone() },
            marker: PhantomData,
        };
        let mut lowering = ConfiguredCircuitLowering { arithmetic, lookup, slots };
        lower_circuit(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_public_keys<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NoSlotOperations::default();
        self.compile_public_keys_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
    }

    pub fn compile_encodings<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NoSlotOperations::default();
        self.compile_encodings_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
    }

    pub fn compile_naive_public_keys<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: NaiveBggPublicKeyVecWire,
        inputs: impl IntoIterator<Item = NaiveBggPublicKeyVecWire>,
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NaivePublicKeySlotOperations;
        self.compile_naive_public_keys_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
    }

    pub fn compile_tall_encodings<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggTallEncodingWire,
        inputs: impl IntoIterator<Item = BggTallEncodingWire>,
    ) -> Result<Vec<BggTallEncodingWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NoSlotOperations::default();
        self.compile_tall_encodings_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
    }

    pub fn compile_naive_encodings<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: NaiveBggEncodingVecWire,
        inputs: impl IntoIterator<Item = NaiveBggEncodingVecWire>,
    ) -> Result<Vec<NaiveBggEncodingVecWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NaiveEncodingSlotOperations;
        self.compile_naive_encodings_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
    }
}

struct NaivePublicKeyLowering<P> {
    compiler: NaiveBggVecCompiler,
    marker: PhantomData<P>,
}

struct TallEncodingLowering<'a, P> {
    compiler: &'a BggTallEncodingCompiler,
    marker: PhantomData<P>,
}

impl<P> CircuitLoweringTypes for TallEncodingLowering<'_, P> {
    type Wire = BggTallEncodingWire;
    type Error = CircuitCompileError;

    fn enter_subcircuit_inputs(
        &mut self,
        mut inputs: Vec<Self::Wire>,
        input_max_plaintext_norm_ranges: Option<
            &[mxx_gadgets::circuit::SubCircuitInputMaxPlaintextNormRange],
        >,
    ) -> Result<Vec<Self::Wire>, Self::Error> {
        let Some(ranges) = input_max_plaintext_norm_ranges else {
            return Ok(inputs);
        };
        for range in ranges {
            let exclusive_upper = &range.norm + BigUint::from(1u8);
            for input in &mut inputs[range.start..range.end] {
                if !matches!(input.plaintext, BggTallPlaintext::Diagonal(_)) {
                    return Err(CircuitCompileError::Structure(
                        "sub-circuit plaintext norm metadata requires revealed tall plaintexts"
                            .to_owned(),
                    ));
                }
                input.canonical_input_exclusive_upper = Some(exclusive_upper.clone());
            }
        }
        Ok(inputs)
    }
}

impl<P: Poly> ArithmeticCircuitLowering<P> for TallEncodingLowering<'_, P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(self.compiler.add(lhs, rhs)?),
            PolyGateKind::Sub => Ok(self.compiler.sub(lhs, rhs)?),
            PolyGateKind::Mul => Ok(self.compiler.simd_mul(lhs, rhs)?),
            _ => unsupported(gate, "non-binary operation"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self
            .compiler
            .public_key
            .ring
            .polynomial(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        Ok(self.compiler.small_scalar_mul(input, &scalar)?)
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self.compiler.public_key.ring.polynomial(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        Ok(self.compiler.large_scalar_mul(input, &scalar)?)
    }
}

impl<P> CircuitLoweringTypes for NaivePublicKeyLowering<P> {
    type Wire = NaiveBggPublicKeyVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for NaivePublicKeyLowering<P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(self.compiler.add_public_keys(lhs, rhs)?),
            PolyGateKind::Sub => Ok(self.compiler.sub_public_keys(lhs, rhs)?),
            PolyGateKind::Mul => Ok(self.compiler.mul_public_keys(lhs, rhs)?),
            _ => unsupported(gate, "non-binary operation"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self
            .compiler
            .public_key
            .ring
            .polynomial(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        let matrices = input.matrices.clone().parallel_map({
            let compiler = self.compiler.public_key.clone();
            let scalar = scalar.clone();
            let reveal = input.reveal_plaintext;
            move |_, matrix| {
                compiler
                    .small_scalar_mul(
                        &BggPublicKeyWire { matrix, reveal_plaintext: reveal },
                        &scalar,
                    )
                    .matrix
            }
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: input.reveal_plaintext })
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self.compiler.public_key.ring.polynomial(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        let matrices = input.matrices.clone().parallel_map({
            let compiler = self.compiler.public_key.clone();
            let scalar = scalar.clone();
            let reveal = input.reveal_plaintext;
            move |_, matrix| {
                compiler
                    .large_scalar_mul(
                        &BggPublicKeyWire { matrix, reveal_plaintext: reveal },
                        &scalar,
                    )
                    .matrix
            }
        })?;
        Ok(NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: input.reveal_plaintext })
    }
}

struct NaiveEncodingLowering<P> {
    compiler: NaiveBggVecCompiler,
    marker: PhantomData<P>,
}

impl<P> CircuitLoweringTypes for NaiveEncodingLowering<P> {
    type Wire = NaiveBggEncodingVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for NaiveEncodingLowering<P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(self.compiler.add_encodings(lhs, rhs)?),
            PolyGateKind::Sub => Ok(self.compiler.sub_encodings(lhs, rhs)?),
            PolyGateKind::Mul => Ok(self.compiler.mul_encodings(lhs, rhs)?),
            _ => unsupported(gate, "non-binary operation"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self
            .compiler
            .public_key
            .ring
            .polynomial(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        Ok(self.compiler.small_scalar_mul_encodings(input, &scalar)?)
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self.compiler.public_key.ring.polynomial(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        Ok(self.compiler.large_scalar_mul_encodings(input, &scalar)?)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NaivePublicKeySlotOperations;

impl CircuitLoweringTypes for NaivePublicKeySlotOperations {
    type Wire = NaiveBggPublicKeyVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> SlotOperationLowering<P> for NaivePublicKeySlotOperations {
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(NaiveBggSlotTransferCompiler.transfer_public_keys(input, source_slots)?)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(NaiveBggSlotTransferCompiler.reduce_public_keys(inputs, slot_count)?)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveEncodingSlotOperations;

impl CircuitLoweringTypes for NaiveEncodingSlotOperations {
    type Wire = NaiveBggEncodingVecWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> SlotOperationLowering<P> for NaiveEncodingSlotOperations {
    fn slot_transfer(
        &mut self,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(NaiveBggSlotTransferCompiler.transfer_encodings(input, source_slots)?)
    }

    fn slot_reduce(
        &mut self,
        inputs: &[Self::Wire],
        slot_count: usize,
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        Ok(NaiveBggSlotTransferCompiler.reduce_encodings(inputs, slot_count)?)
    }
}

struct PublicKeyLowering<'a, P> {
    compiler: &'a BggPublicKeyCompiler,
    marker: PhantomData<P>,
}

impl<P> CircuitLoweringTypes for PublicKeyLowering<'_, P> {
    type Wire = BggPublicKeyWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for PublicKeyLowering<'_, P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match operation {
            PolyGateKind::Add => Ok(self.compiler.add(lhs, rhs)),
            PolyGateKind::Sub => Ok(self.compiler.sub(lhs, rhs)),
            PolyGateKind::Mul => Ok(self.compiler.mul(lhs, rhs)),
            _ => unsupported(gate, "non-binary operation"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self
            .compiler
            .ring
            .polynomial(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        Ok(self.compiler.small_scalar_mul(input, &scalar))
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self.compiler.ring.polynomial(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        Ok(self.compiler.large_scalar_mul(input, &scalar))
    }
}

struct EncodingLowering<'a, P> {
    compiler: &'a BggEncodingCompiler,
    marker: PhantomData<P>,
}

impl<P> CircuitLoweringTypes for EncodingLowering<'_, P> {
    type Wire = BggEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for EncodingLowering<'_, P> {
    fn binary(
        &mut self,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match operation {
            PolyGateKind::Add => self.compiler.add(lhs, rhs).map_err(Into::into),
            PolyGateKind::Sub => self.compiler.sub(lhs, rhs).map_err(Into::into),
            PolyGateKind::Mul => self.compiler.mul(lhs, rhs).map_err(Into::into),
            _ => unsupported(gate, "non-binary operation"),
        }
    }

    fn small_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self
            .compiler
            .public_key
            .ring
            .polynomial(scalar.iter().copied().map(mxx_ir_core::IntExpr::constant));
        Ok(self.compiler.small_scalar_mul(input, &scalar))
    }

    fn large_scalar_mul(
        &mut self,
        input: &Self::Wire,
        scalar: &[BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = self.compiler.public_key.ring.polynomial(
            scalar
                .iter()
                .cloned()
                .map(num_bigint::BigInt::from)
                .map(mxx_ir_core::IntExpr::constant),
        );
        Ok(self.compiler.large_scalar_mul(input, &scalar))
    }
}

fn unsupported<T>(gate: GateInstance<'_>, feature: &'static str) -> Result<T, CircuitCompileError> {
    Err(CircuitCompileError::Unsupported { gate: gate.local_gate().index(), feature })
}

fn map_lower_error(error: CircuitLowerError<CircuitCompileError>) -> CircuitCompileError {
    match error {
        CircuitLowerError::Operation { source, .. } => source,
        other => CircuitCompileError::Structure(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BggSlotTransferPublicKeyLowering, LweLookupArtifacts, LweLookupCompiler, LweLookupIdentity,
        LweLookupInvocation, LweLookupPublicKeyLowering, LweLookupTable,
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_gadgets::circuit::{LutExpr, PublicLutProgram};
    use mxx_ir_core::{
        IntExpr, ParamEnv,
        artifact::{ProductionId, SpecHash},
        node::NodeKind,
        types::MatrixType,
    };
    use mxx_primitives::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::{BigInt, BigUint};

    #[derive(Default)]
    struct RecordingTallLookup {
        canonical_input_exclusive_upper: Option<BigUint>,
    }

    impl CircuitLoweringTypes for RecordingTallLookup {
        type Wire = BggTallEncodingWire;
        type Error = CircuitCompileError;
    }

    impl PublicLookupLowering<DCRTPoly> for RecordingTallLookup {
        fn public_lookup(
            &mut self,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &Self::Wire,
            _gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.canonical_input_exclusive_upper = input.canonical_input_exclusive_upper.clone();
            Ok(input.clone())
        }
    }

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn naive_public_key_circuit_lowers_slot_transfer() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(1, None), (0, Some(3))]);
        circuit.output([transferred]);

        let ring = Ring::new(17, 8);
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                ring: ring.clone(),
                base: 2.into(),
                digit_count: 2.into(),
            },
        };
        let one = NaiveBggPublicKeyVecWire {
            matrices: ring.input_family("one", 2, (1, 1)),
            reveal_plaintext: true,
        };
        let input = NaiveBggPublicKeyVecWire {
            matrices: ring.input_family("input", 2, (1, 1)),
            reveal_plaintext: true,
        };
        let outputs = compiler
            .compile_naive_public_keys(&circuit, one, [input])
            .expect("slot transfer lowering");
        let built = DslContext::new("slot-transfer")
            .family_output("output", outputs[0].matrices.clone())
            .expect("output")
            .build()
            .expect("build");
        built.validate(&ParamEnv::default()).expect("validation");
    }

    #[test]
    fn configured_lowering_forwards_audited_lut_ranges_for_normal_and_parallel_calls() {
        let context = DslContext::new("audited-lut-range-forwarding");
        let ring = Ring::new(17, 8);
        let public_key =
            BggPublicKeyCompiler { ring: ring.clone(), base: 2.into(), digit_count: 2.into() };
        let arithmetic =
            PublicKeyLowering::<DCRTPoly> { compiler: &public_key, marker: PhantomData };
        let mut lookup = NoPublicLookup::<BggPublicKeyWire>::default();
        let mut slots = NoSlotOperations::<BggPublicKeyWire>::default();
        let mut lowering =
            ConfiguredCircuitLowering { arithmetic, lookup: &mut lookup, slots: &mut slots };
        let input =
            |name| BggPublicKeyWire { matrix: ring.input(name, (1, 1)), reveal_plaintext: true };
        let definition = Subgraph::define(
            "audited-lut-child",
            vec![input("definition-input").schema()],
            |values| values,
        )
        .expect("subgraph definition");
        let bounds = vec![Some(BigUint::from(4u8))];
        let normal = lowering
            .call_audited_constant_lut_subgraph(
                &definition,
                vec![input("normal-input")],
                bounds.clone(),
            )
            .expect("normal call");
        let parallel = lowering
            .call_audited_constant_lut_subgraph_parallel(
                &definition,
                vec![vec![input("parallel-left")], vec![input("parallel-right")]],
                bounds.clone(),
            )
            .expect("parallel calls");
        let graph = context
            .output("normal", normal[0].matrix.clone())
            .expect("normal output")
            .output("parallel-left", parallel[0][0].matrix.clone())
            .expect("parallel left output")
            .output("parallel-right", parallel[1][0].matrix.clone())
            .expect("parallel right output")
            .build()
            .expect("graph");
        let calls = graph
            .graph
            .root_scope()
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::SubgraphCall(call) => Some(&call.canonical_input_exclusive_uppers),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls, vec![&bounds, &bounds, &bounds]);
        graph.validate(&ParamEnv::default()).expect("validation");
    }

    #[test]
    fn unstructured_tall_nested_call_attaches_the_exclusive_plaintext_upper() {
        let mut parent = PolyCircuit::<DCRTPoly>::new();
        let lookup = parent.register_public_lookup(
            PublicLutProgram::new(8, LutExpr::input()).expect("identity lookup"),
        );
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_input = child.input(1).as_single_wire();
        let child_output = child.public_lookup_gate(child_input, lookup);
        child.output([child_output]);

        let parent_input = parent.input(1).as_single_wire();
        let child_id = parent.register_sub_circuit(child);
        let output = parent.call_sub_circuit_with_max_plaintext_norms(
            child_id,
            [parent_input],
            [mxx_gadgets::circuit::SubCircuitInputMaxPlaintextNormRange::new(
                0,
                1,
                BigUint::from(6u8),
            )],
        );
        parent.output(output);

        let ring = Ring::new(17, 8);
        let public_key =
            BggPublicKeyCompiler { ring: ring.clone(), base: 2.into(), digit_count: 2.into() };
        let tall = |name: &str| BggTallEncodingWire {
            rows: ring.input_family(format!("{name}-rows"), 1, (1, 2)),
            pubkey: BggPublicKeyWire {
                matrix: ring.input(format!("{name}-public"), (1, 2)),
                reveal_plaintext: true,
            },
            plaintext: BggTallPlaintext::Diagonal(ring.input_family(
                format!("{name}-plaintext"),
                1,
                (1, 1),
            )),
            canonical_input_exclusive_upper: None,
        };
        let mut lookup = RecordingTallLookup::default();
        let mut slots = NoSlotOperations::<BggTallEncodingWire>::default();
        PolyCircuitCompiler { public_key }
            .compile_tall_encodings_with_lowerings(
                &parent,
                tall("one"),
                [tall("input")],
                &mut lookup,
                &mut slots,
            )
            .expect("unstructured tall lowering");
        assert_eq!(lookup.canonical_input_exclusive_upper, Some(BigUint::from(7u8)));
    }

    #[test]
    fn lookup_and_slot_providers_compose_in_one_circuit() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(0, None)]);
        let lookup_id = circuit.register_public_lookup(
            PublicLutProgram::new(2, LutExpr::input()).expect("identity LUT"),
        );
        let looked_up = circuit.public_lookup_gate(transferred, lookup_id);
        circuit.output([looked_up]);

        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: looked_up.as_single_wire().index(),
            occurrence: 0,
            lookup: lookup_id,
            slot: None,
        };
        let table = LweLookupTable::from_public_lut(circuit.lookup_table(lookup_id).as_ref())
            .expect("lookup table");
        let lookup = LweLookupCompiler {
            identity,
            table,
            public_key_type: matrix_type(&parameters, 1, digit_count),
            low_matrix_type: matrix_type(&parameters, digit_count, digit_count),
            high_matrix_type: matrix_type(&parameters, digit_count + 2, digit_count),
            gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
            digit_count: IntExpr::constant(digit_count),
        };
        let invocation = LweLookupInvocation::bind(
            lookup.clone(),
            LweLookupArtifacts::for_compiler(
                ProductionId { spec_hash: SpecHash([60; 32]), execution_nonce: [61; 32] },
                &lookup,
            ),
            &parameters,
            &circuit,
        )
        .expect("lookup invocation");
        let ring = Ring::new(
            lookup.public_key_type.modulus.clone(),
            lookup.public_key_type.ring_dimension.clone(),
        );
        let public_key_compiler = BggPublicKeyCompiler {
            ring: ring.clone(),
            base: lookup.gadget_base.clone(),
            digit_count: lookup.digit_count.clone(),
        };
        let mut lookup_provider =
            LweLookupPublicKeyLowering::new([invocation]).expect("lookup provider");
        let mut slot_provider = BggSlotTransferPublicKeyLowering {
            compiler: public_key_compiler.clone(),
            hash_key: ring.bytes_input("slot-hash-key", 32),
            public_key_type: lookup.public_key_type.clone(),
            configured_slot_count: 1,
            output_public_key_production: None,
            requests: Vec::new(),
        };
        let public_key = |name: &str| BggPublicKeyWire {
            matrix: ring.input(name, (1, digit_count)),
            reveal_plaintext: true,
        };
        let outputs = PolyCircuitCompiler { public_key: public_key_compiler }
            .compile_public_keys_with_lowerings(
                &circuit,
                public_key("one"),
                [public_key("input")],
                &mut lookup_provider,
                &mut slot_provider,
            )
            .expect("combined lowering");

        assert_eq!(slot_provider.requests.len(), 1);
        let graph = DslContext::new("lookup-and-slot-transfer")
            .output("output", outputs[0].matrix.clone())
            .expect("output")
            .build()
            .expect("graph");
        assert_eq!(
            graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::Input { artifact: Some(_), .. }))
                .count(),
            1
        );
    }
}
