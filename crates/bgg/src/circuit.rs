//! PolyCircuit evaluation into declarative BGG+ DAG values.

use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPolyEncodingCompiler, BggPolyEncodingWire,
    BggPublicKeyCompiler, BggPublicKeyWire, EncodingCompileError, NaiveBggEncodingVecWire,
    NaiveBggPublicKeyVecWire, NaiveBggSlotTransferCompiler, NaiveBggVecCompiler,
    NaiveVecCompileError, PolyEncodingCompileError, SlotFamilyCompileError,
};
use mxx_dsl::{GraphValue, Subgraph};
use mxx_gadgets::{
    Poly,
    circuit::{
        ArithmeticCircuitLowering, CircuitLowerError, CircuitLoweringTypes, GateInstance,
        PolyCircuit, PolyGateKind, PublicLookupLowering, SlotOperationLowering,
        StructuredCircuitLowering, lower_circuit, lower_circuit_structured,
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
}

#[derive(Debug, Error)]
pub enum CircuitCompileError {
    #[error(transparent)]
    Encoding(#[from] EncodingCompileError),
    #[error(transparent)]
    NaiveVector(#[from] NaiveVecCompileError),
    #[error(transparent)]
    PolyEncoding(#[from] PolyEncodingCompileError),
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
        lower_circuit_structured(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
    }

    pub fn compile_poly_encodings_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggPolyEncodingWire,
        inputs: impl IntoIterator<Item = BggPolyEncodingWire>,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<Vec<BggPolyEncodingWire>, CircuitCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggPolyEncodingWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggPolyEncodingWire, Error = CircuitCompileError>,
    {
        let compiler = BggPolyEncodingCompiler { public_key: self.public_key.clone() };
        let arithmetic = PolyEncodingLowering::<P> { compiler: &compiler, marker: PhantomData };
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
        lower_circuit_structured(circuit, one, inputs, &mut lowering).map_err(map_lower_error)
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

    pub fn compile_poly_encodings<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        one: BggPolyEncodingWire,
        inputs: impl IntoIterator<Item = BggPolyEncodingWire>,
    ) -> Result<Vec<BggPolyEncodingWire>, CircuitCompileError> {
        let mut lookup = NoPublicLookup::default();
        let mut slots = NoSlotOperations::default();
        self.compile_poly_encodings_with_lowerings(circuit, one, inputs, &mut lookup, &mut slots)
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

struct PolyEncodingLowering<'a, P> {
    compiler: &'a BggPolyEncodingCompiler,
    marker: PhantomData<P>,
}

impl<P> CircuitLoweringTypes for PolyEncodingLowering<'_, P> {
    type Wire = BggPolyEncodingWire;
    type Error = CircuitCompileError;
}

impl<P: Poly> ArithmeticCircuitLowering<P> for PolyEncodingLowering<'_, P> {
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
            PolyGateKind::Mul => Ok(self.compiler.mul(lhs, rhs)?),
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
    use mxx_gadgets::{PolyElem, circuit::PublicLut};
    use mxx_ir_core::{
        IntExpr, ParamEnv,
        artifact::{ProductionId, SpecHash},
        node::NodeKind,
        types::MatrixType,
    };
    use mxx_primitives::poly::{
        Poly as ConcretePoly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::BigInt;

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
    fn lookup_and_slot_providers_compose_in_one_circuit() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input_gate = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input_gate, &[(0, None)]);
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((
                    input,
                    <DCRTPoly as ConcretePoly>::Elem::constant(&parameters.modulus(), input),
                ))
            },
            None,
        ));
        let looked_up = circuit.public_lookup_gate(transferred, lookup_id);
        circuit.output([looked_up]);

        let identity = LweLookupIdentity {
            call_path: Vec::new(),
            gate: looked_up.as_single_wire().index(),
            occurrence: 0,
            lookup: lookup_id,
            slot: None,
        };
        let table =
            LweLookupTable::from_public_lut(&parameters, circuit.lookup_table(lookup_id).as_ref())
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
