use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPolyEncodingCompiler, BggPolyEncodingWire,
    BggPublicKeyCompiler, BggPublicKeyWire, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
    NaiveBggVecCompiler,
};
use mxx_gadgets::{
    Poly,
    circuit::{
        CircuitLowerError, GateInstance, GraphCircuitLowering, PolyCircuit, PolyGateKind,
        lower_circuit,
    },
};
use mxx_ir_core::{
    GraphBuilder, MatrixWire, SubgraphBuildError, artifact::ArtifactConfidentiality,
    node::MatrixBinaryOp,
};
use num_bigint::BigInt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CircuitCompileError {
    #[error("gate {gate} references unavailable input gate {input}")]
    MissingInput { gate: usize, input: usize },
    #[error("gate {gate} has an invalid input arity")]
    InvalidArity { gate: usize },
    #[error("gate {gate} requires lowering context for {kind}")]
    MissingGateContext { gate: usize, kind: &'static str },
    #[error("the circuit contains advanced gates and requires an advanced lowering context")]
    AdvancedGateContextRequired,
    #[error("gate {gate} has an invalid slot-transfer or slot-reduction specification")]
    InvalidSlotTransfer { gate: usize },
    #[error("slot-transfer artifact {name} is unavailable")]
    MissingSlotTransferArtifact { name: String },
    #[error("the circuit compiler received more input bundles than the circuit consumes")]
    ExtraInputs,
    #[error("gate {gate}: {source}")]
    Encoding {
        gate: usize,
        #[source]
        source: crate::encoding::EncodingCompileError,
    },
    #[error("gate {gate}: {source}")]
    PolyEncoding {
        gate: usize,
        #[source]
        source: crate::poly_encoding::PolyEncodingCompileError,
    },
    #[error("gate {gate}: {source}")]
    NaiveVec {
        gate: usize,
        #[source]
        source: crate::naive_vec::NaiveVecCompileError,
    },
    #[error("gate {gate}: {source}")]
    LweLookup {
        gate: usize,
        #[source]
        source: crate::lwe_lookup::LweLookupCompileError,
    },
    #[error(transparent)]
    Subgraph(#[from] SubgraphBuildError),
}

#[derive(Clone, Debug)]
pub struct PolyCircuitCompiler {
    pub public_key: BggPublicKeyCompiler,
}

struct PublicKeyLowering<'a, P: Poly> {
    compiler: &'a BggPublicKeyCompiler,
    advanced: Option<&'a mut (dyn AdvancedGateLowering<P, BggPublicKeyWire> + 'a)>,
}

struct EncodingLowering<'a, P: Poly> {
    compiler: &'a BggEncodingCompiler,
    advanced: Option<&'a mut (dyn AdvancedGateLowering<P, BggEncodingWire> + 'a)>,
}

struct PolyEncodingLowering<'a, P: Poly> {
    compiler: &'a BggPolyEncodingCompiler,
    advanced: Option<&'a mut (dyn AdvancedGateLowering<P, BggPolyEncodingWire> + 'a)>,
}

struct NaivePublicKeyLowering<'a, P: Poly> {
    compiler: &'a NaiveBggVecCompiler,
    advanced: Option<&'a mut (dyn AdvancedGateLowering<P, NaiveBggPublicKeyVecWire> + 'a)>,
}

struct NaiveEncodingLowering<'a, P: Poly> {
    compiler: &'a NaiveBggVecCompiler,
    advanced: Option<&'a mut (dyn AdvancedGateLowering<P, NaiveBggEncodingVecWire> + 'a)>,
}

impl<P: Poly> GraphCircuitLowering<P> for PublicKeyLowering<'_, P> {
    type Wire = BggPublicKeyWire;
    type Error = CircuitCompileError;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let operation = match operation {
            PolyGateKind::Add => MatrixBinaryOp::Add,
            PolyGateKind::Sub => MatrixBinaryOp::Subtract,
            PolyGateKind::Mul => MatrixBinaryOp::Multiply,
            _ => {
                return Err(CircuitCompileError::InvalidArity { gate: gate.local_gate().index() });
            }
        };
        compile_public_key_binary_template(builder, self.compiler, operation, lhs, rhs)
    }

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(*value)),
        );
        compile_public_key_scalar_template(builder, self.compiler, input, &scalar, false)
    }

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[num_bigint::BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(value.clone())),
        );
        compile_public_key_scalar_template(builder, self.compiler, input, &scalar, true)
    }

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_transfer(builder, input, source_slots, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            }),
        }
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_reduce(builder, inputs, slot_count, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            }),
        }
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.public_lookup(builder, circuit, lookup_id, input, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            }),
        }
    }
}

impl<P: Poly> GraphCircuitLowering<P> for EncodingLowering<'_, P> {
    type Wire = BggEncodingWire;
    type Error = CircuitCompileError;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let operation = match operation {
            PolyGateKind::Add => MatrixBinaryOp::Add,
            PolyGateKind::Sub => MatrixBinaryOp::Subtract,
            PolyGateKind::Mul => MatrixBinaryOp::Multiply,
            _ => {
                return Err(CircuitCompileError::InvalidArity { gate: gate.local_gate().index() });
            }
        };
        compile_encoding_binary_template(
            builder,
            self.compiler,
            operation,
            lhs,
            rhs,
            gate.local_gate().index(),
        )
    }

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkey.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(*value)),
        );
        compile_encoding_scalar_template(builder, self.compiler, input, &scalar, false)
    }

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[num_bigint::BigUint],
        _gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkey.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(value.clone())),
        );
        compile_encoding_scalar_template(builder, self.compiler, input, &scalar, true)
    }

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_transfer(builder, input, source_slots, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            }),
        }
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_reduce(builder, inputs, slot_count, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            }),
        }
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.public_lookup(builder, circuit, lookup_id, input, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            }),
        }
    }
}

impl<P: Poly> GraphCircuitLowering<P> for PolyEncodingLowering<'_, P> {
    type Wire = BggPolyEncodingWire;
    type Error = CircuitCompileError;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let output = match operation {
            PolyGateKind::Add => self.compiler.add(builder, lhs, rhs),
            PolyGateKind::Sub => self.compiler.sub(builder, lhs, rhs),
            PolyGateKind::Mul => self.compiler.mul(builder, lhs, rhs),
            _ => {
                return Err(CircuitCompileError::InvalidArity { gate: gate.local_gate().index() });
            }
        };
        output.map_err(|source| CircuitCompileError::PolyEncoding {
            gate: gate.local_gate().index(),
            source,
        })
    }

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkey.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(*value)),
        );
        self.compiler.small_scalar_mul(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::PolyEncoding { gate: gate.local_gate().index(), source }
        })
    }

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[num_bigint::BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkey.matrix.matrix_type),
            scalar.iter().map(|value| BigInt::from(value.clone())),
        );
        self.compiler.large_scalar_mul(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::PolyEncoding { gate: gate.local_gate().index(), source }
        })
    }

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_transfer(builder, input, source_slots, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            }),
        }
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_reduce(builder, inputs, slot_count, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            }),
        }
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.public_lookup(builder, circuit, lookup_id, input, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            }),
        }
    }
}

impl<P: Poly> GraphCircuitLowering<P> for NaivePublicKeyLowering<'_, P> {
    type Wire = NaiveBggPublicKeyVecWire;
    type Error = CircuitCompileError;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let output = match operation {
            PolyGateKind::Add => self.compiler.add_public_keys(builder, lhs, rhs),
            PolyGateKind::Sub => self.compiler.sub_public_keys(builder, lhs, rhs),
            PolyGateKind::Mul => self.compiler.mul_public_keys(builder, lhs, rhs),
            _ => {
                return Err(CircuitCompileError::InvalidArity { gate: gate.local_gate().index() });
            }
        };
        output.map_err(|source| CircuitCompileError::NaiveVec {
            gate: gate.local_gate().index(),
            source,
        })
    }

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.matrices.matrix_type),
            scalar.iter().map(|value| BigInt::from(*value)),
        );
        self.compiler.small_scalar_mul_public_keys(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::NaiveVec { gate: gate.local_gate().index(), source }
        })
    }

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[num_bigint::BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.matrices.matrix_type),
            scalar.iter().map(|value| BigInt::from(value.clone())),
        );
        self.compiler.large_scalar_mul_public_keys(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::NaiveVec { gate: gate.local_gate().index(), source }
        })
    }

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_transfer(builder, input, source_slots, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            }),
        }
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_reduce(builder, inputs, slot_count, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            }),
        }
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.public_lookup(builder, circuit, lookup_id, input, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            }),
        }
    }
}

impl<P: Poly> GraphCircuitLowering<P> for NaiveEncodingLowering<'_, P> {
    type Wire = NaiveBggEncodingVecWire;
    type Error = CircuitCompileError;

    fn binary(
        &mut self,
        builder: &mut GraphBuilder,
        operation: PolyGateKind,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let output = match operation {
            PolyGateKind::Add => self.compiler.add_encodings(builder, lhs, rhs),
            PolyGateKind::Sub => self.compiler.sub_encodings(builder, lhs, rhs),
            PolyGateKind::Mul => self.compiler.mul_encodings(builder, lhs, rhs),
            _ => {
                return Err(CircuitCompileError::InvalidArity { gate: gate.local_gate().index() });
            }
        };
        output.map_err(|source| CircuitCompileError::NaiveVec {
            gate: gate.local_gate().index(),
            source,
        })
    }

    fn small_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[u32],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkeys.matrix_type),
            scalar.iter().map(|value| BigInt::from(*value)),
        );
        self.compiler.small_scalar_mul_encodings(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::NaiveVec { gate: gate.local_gate().index(), source }
        })
    }

    fn large_scalar_mul(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        scalar: &[num_bigint::BigUint],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        let scalar = builder.constant_polynomial(
            scalar_type(&input.pubkeys.matrix_type),
            scalar.iter().map(|value| BigInt::from(value.clone())),
        );
        self.compiler.large_scalar_mul_encodings(builder, input, &scalar).map_err(|source| {
            CircuitCompileError::NaiveVec { gate: gate.local_gate().index(), source }
        })
    }

    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &Self::Wire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_transfer(builder, input, source_slots, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            }),
        }
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[Self::Wire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.slot_reduce(builder, inputs, slot_count, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            }),
        }
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &Self::Wire,
        gate: GateInstance<'_>,
    ) -> Result<Self::Wire, Self::Error> {
        match self.advanced.as_deref_mut() {
            Some(advanced) => advanced.public_lookup(builder, circuit, lookup_id, input, gate),
            None => Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            }),
        }
    }
}

fn map_lower_error(error: CircuitLowerError<CircuitCompileError>) -> CircuitCompileError {
    match error {
        CircuitLowerError::MissingInput { gate, input } => {
            CircuitCompileError::MissingInput { gate, input }
        }
        CircuitLowerError::InvalidArity { gate } => CircuitCompileError::InvalidArity { gate },
        CircuitLowerError::ExtraInputs => CircuitCompileError::ExtraInputs,
        CircuitLowerError::ParameterizedPublicLookup { gate, .. } => {
            CircuitCompileError::MissingGateContext { gate, kind: "parameterized public lookup" }
        }
        CircuitLowerError::MissingParameter { gate, .. } |
        CircuitLowerError::ParameterKind { gate, .. } => {
            CircuitCompileError::MissingGateContext { gate, kind: "sub-circuit parameter binding" }
        }
        CircuitLowerError::Operation { source, .. } => source,
    }
}

/// Scheme-specific lowering for gates whose concrete construction depends on
/// lookup or slot-transfer preprocessing that is intentionally not stored in a
/// [`PolyCircuit`].
pub trait AdvancedGateLowering<P: Poly, W> {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &W,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError>;

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[W],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError>;

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &W,
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError>;
}

/// Combines one public-lookup lowering with one slot-transfer lowering.
///
/// The two constructions have independent preprocessing artifacts. Keeping
/// them as delegates avoids a product enum over lookup schemes and slot
/// schemes while still allowing one circuit to contain both gate families.
#[derive(Clone, Debug)]
pub struct CompositeAdvancedGateLowering<L, S> {
    pub lookup: L,
    pub slots: S,
}

impl<L, S> CompositeAdvancedGateLowering<L, S> {
    pub fn new(lookup: L, slots: S) -> Self {
        Self { lookup, slots }
    }

    pub fn into_parts(self) -> (L, S) {
        (self.lookup, self.slots)
    }
}

impl<P: Poly, W, L, S> AdvancedGateLowering<P, W> for CompositeAdvancedGateLowering<L, S>
where
    L: AdvancedGateLowering<P, W>,
    S: AdvancedGateLowering<P, W>,
{
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &W,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError> {
        self.slots.slot_transfer(builder, input, source_slots, gate)
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[W],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError> {
        self.slots.slot_reduce(builder, inputs, slot_count, gate)
    }

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &W,
        gate: GateInstance<'_>,
    ) -> Result<W, CircuitCompileError> {
        self.lookup.public_lookup(builder, circuit, lookup_id, input, gate)
    }
}

impl PolyCircuitCompiler {
    pub fn compile_public_keys<P: Poly>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError> {
        if circuit.requires_advanced_lowering() {
            return Err(CircuitCompileError::AdvancedGateContextRequired);
        }
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut PublicKeyLowering::<P> { compiler: &self.public_key, advanced: None },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_encodings<P: Poly>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError> {
        if circuit.requires_advanced_lowering() {
            return Err(CircuitCompileError::AdvancedGateContextRequired);
        }
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut EncodingLowering::<P> { compiler: &compiler, advanced: None },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_poly_encodings<P: Poly>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPolyEncodingWire,
        inputs: impl IntoIterator<Item = BggPolyEncodingWire>,
    ) -> Result<Vec<BggPolyEncodingWire>, CircuitCompileError> {
        if circuit.requires_advanced_lowering() {
            return Err(CircuitCompileError::AdvancedGateContextRequired);
        }
        let compiler = BggPolyEncodingCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut PolyEncodingLowering::<P> { compiler: &compiler, advanced: None },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_public_keys_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
        lowering: &mut L,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggPublicKeyWire>,
    {
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut PublicKeyLowering::<P> { compiler: &self.public_key, advanced: Some(lowering) },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_encodings_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
        lowering: &mut L,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggEncodingWire>,
    {
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut EncodingLowering::<P> { compiler: &compiler, advanced: Some(lowering) },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_poly_encodings_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPolyEncodingWire,
        inputs: impl IntoIterator<Item = BggPolyEncodingWire>,
        lowering: &mut L,
    ) -> Result<Vec<BggPolyEncodingWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggPolyEncodingWire>,
    {
        let compiler = BggPolyEncodingCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut PolyEncodingLowering::<P> { compiler: &compiler, advanced: Some(lowering) },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_naive_public_keys_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: NaiveBggPublicKeyVecWire,
        inputs: impl IntoIterator<Item = NaiveBggPublicKeyVecWire>,
        lowering: &mut L,
    ) -> Result<Vec<NaiveBggPublicKeyVecWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, NaiveBggPublicKeyVecWire>,
    {
        let compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut NaivePublicKeyLowering::<P> { compiler: &compiler, advanced: Some(lowering) },
        )
        .map_err(map_lower_error)
    }

    pub fn compile_naive_encodings_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: NaiveBggEncodingVecWire,
        inputs: impl IntoIterator<Item = NaiveBggEncodingVecWire>,
        lowering: &mut L,
    ) -> Result<Vec<NaiveBggEncodingVecWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, NaiveBggEncodingVecWire>,
    {
        let compiler = NaiveBggVecCompiler { public_key: self.public_key.clone() };
        lower_circuit(
            builder,
            circuit,
            one,
            inputs,
            &mut NaiveEncodingLowering::<P> { compiler: &compiler, advanced: Some(lowering) },
        )
        .map_err(map_lower_error)
    }
}

fn compile_public_key_binary_template(
    builder: &mut GraphBuilder,
    compiler: &BggPublicKeyCompiler,
    operation: MatrixBinaryOp,
    lhs: &BggPublicKeyWire,
    rhs: &BggPublicKeyWire,
) -> Result<BggPublicKeyWire, CircuitCompileError> {
    let name = match operation {
        MatrixBinaryOp::Add => "bgg-public-key-add",
        MatrixBinaryOp::Subtract => "bgg-public-key-sub",
        MatrixBinaryOp::Multiply => "bgg-public-key-mul",
    };
    let mut template = GraphBuilder::new(name, Vec::new());
    let template_lhs = BggPublicKeyWire {
        matrix: template.input("lhs", lhs.matrix.matrix_type.clone()),
        reveal_plaintext: lhs.reveal_plaintext,
    };
    let template_rhs = BggPublicKeyWire {
        matrix: template.input("rhs", rhs.matrix.matrix_type.clone()),
        reveal_plaintext: rhs.reveal_plaintext,
    };
    let output = match operation {
        MatrixBinaryOp::Add => compiler.add(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Subtract => compiler.sub(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Multiply => compiler.mul(&mut template, &template_lhs, &template_rhs),
    };
    template.output("0_matrix", &output.matrix, ArtifactConfidentiality::Public);
    let mut outputs = builder.subgraph_call(
        template.finish(),
        vec![lhs.matrix.wire, rhs.matrix.wire],
        &[output.matrix.matrix_type],
    )?;
    Ok(BggPublicKeyWire { matrix: outputs.remove(0), reveal_plaintext: output.reveal_plaintext })
}

fn compile_encoding_binary_template(
    builder: &mut GraphBuilder,
    compiler: &BggEncodingCompiler,
    operation: MatrixBinaryOp,
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    gate: usize,
) -> Result<BggEncodingWire, CircuitCompileError> {
    let plaintext_kind = match (&lhs.plaintext, &rhs.plaintext) {
        (Some(_), Some(_)) => "revealed-revealed",
        (Some(_), None) => "revealed-hidden",
        (None, Some(_)) => "hidden-revealed",
        (None, None) => "hidden-hidden",
    };
    let operation_name = match operation {
        MatrixBinaryOp::Add => "add",
        MatrixBinaryOp::Subtract => "sub",
        MatrixBinaryOp::Multiply => "mul",
    };
    let mut template =
        GraphBuilder::new(format!("bgg-encoding-{operation_name}-{plaintext_kind}"), Vec::new());
    let template_lhs = BggEncodingWire {
        vector: template.input("0_lhs_vector", lhs.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("1_lhs_pubkey", lhs.pubkey.matrix.matrix_type.clone()),
            reveal_plaintext: lhs.pubkey.reveal_plaintext,
        },
        plaintext: lhs
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("2_lhs_plaintext", plaintext.matrix_type.clone())),
    };
    let template_rhs = BggEncodingWire {
        vector: template.input("3_rhs_vector", rhs.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("4_rhs_pubkey", rhs.pubkey.matrix.matrix_type.clone()),
            reveal_plaintext: rhs.pubkey.reveal_plaintext,
        },
        plaintext: rhs
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("5_rhs_plaintext", plaintext.matrix_type.clone())),
    };
    let output = match operation {
        MatrixBinaryOp::Add => compiler.add(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Subtract => compiler.sub(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Multiply => compiler.mul(&mut template, &template_lhs, &template_rhs),
    }
    .map_err(|source| CircuitCompileError::Encoding { gate, source })?;
    template.output("0_vector", &output.vector, ArtifactConfidentiality::Public);
    template.output("1_pubkey", &output.pubkey.matrix, ArtifactConfidentiality::Public);
    if let Some(plaintext) = &output.plaintext {
        template.output("2_plaintext", plaintext, ArtifactConfidentiality::Public);
    }
    let mut args = vec![lhs.vector.wire, lhs.pubkey.matrix.wire];
    if let Some(plaintext) = &lhs.plaintext {
        args.push(plaintext.wire);
    }
    args.extend([rhs.vector.wire, rhs.pubkey.matrix.wire]);
    if let Some(plaintext) = &rhs.plaintext {
        args.push(plaintext.wire);
    }
    let mut output_types =
        vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
    if let Some(plaintext) = &output.plaintext {
        output_types.push(plaintext.matrix_type.clone());
    }
    let mut outputs = builder.subgraph_call(template.finish(), args, &output_types)?;
    let vector = outputs.remove(0);
    let pubkey = BggPublicKeyWire {
        matrix: outputs.remove(0),
        reveal_plaintext: output.pubkey.reveal_plaintext,
    };
    let plaintext = (!outputs.is_empty()).then(|| outputs.remove(0));
    Ok(BggEncodingWire { vector, pubkey, plaintext })
}

fn compile_public_key_scalar_template(
    builder: &mut GraphBuilder,
    compiler: &BggPublicKeyCompiler,
    input: &BggPublicKeyWire,
    scalar: &MatrixWire,
    large: bool,
) -> Result<BggPublicKeyWire, CircuitCompileError> {
    let name = if large { "bgg-public-key-large-scalar" } else { "bgg-public-key-small-scalar" };
    let mut template = GraphBuilder::new(name, Vec::new());
    let template_input = BggPublicKeyWire {
        matrix: template.input("input", input.matrix.matrix_type.clone()),
        reveal_plaintext: input.reveal_plaintext,
    };
    let template_scalar = template.input("scalar", scalar.matrix_type.clone());
    let output = if large {
        compiler.large_scalar_mul(&mut template, &template_input, &template_scalar)
    } else {
        compiler.small_scalar_mul(&mut template, &template_input, &template_scalar)
    };
    template.output("0_matrix", &output.matrix, ArtifactConfidentiality::Public);
    let mut outputs = builder.subgraph_call(
        template.finish(),
        vec![input.matrix.wire, scalar.wire],
        &[output.matrix.matrix_type],
    )?;
    Ok(BggPublicKeyWire { matrix: outputs.remove(0), reveal_plaintext: output.reveal_plaintext })
}

fn compile_encoding_scalar_template(
    builder: &mut GraphBuilder,
    compiler: &BggEncodingCompiler,
    input: &BggEncodingWire,
    scalar: &MatrixWire,
    large: bool,
) -> Result<BggEncodingWire, CircuitCompileError> {
    let plaintext_kind = if input.plaintext.is_some() { "revealed" } else { "hidden" };
    let operation = if large { "large" } else { "small" };
    let mut template =
        GraphBuilder::new(format!("bgg-encoding-{operation}-scalar-{plaintext_kind}"), Vec::new());
    let template_input = BggEncodingWire {
        vector: template.input("0_vector", input.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("1_pubkey", input.pubkey.matrix.matrix_type.clone()),
            reveal_plaintext: input.pubkey.reveal_plaintext,
        },
        plaintext: input
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("2_plaintext", plaintext.matrix_type.clone())),
    };
    let template_scalar = template.input("3_scalar", scalar.matrix_type.clone());
    let output = if large {
        compiler.large_scalar_mul(&mut template, &template_input, &template_scalar)
    } else {
        compiler.small_scalar_mul(&mut template, &template_input, &template_scalar)
    };
    template.output("0_vector", &output.vector, ArtifactConfidentiality::Public);
    template.output("1_pubkey", &output.pubkey.matrix, ArtifactConfidentiality::Public);
    if let Some(plaintext) = &output.plaintext {
        template.output("2_plaintext", plaintext, ArtifactConfidentiality::Public);
    }
    let mut args = vec![input.vector.wire, input.pubkey.matrix.wire];
    if let Some(plaintext) = &input.plaintext {
        args.push(plaintext.wire);
    }
    args.push(scalar.wire);
    let mut output_types =
        vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
    if let Some(plaintext) = &output.plaintext {
        output_types.push(plaintext.matrix_type.clone());
    }
    let mut outputs = builder.subgraph_call(template.finish(), args, &output_types)?;
    let vector = outputs.remove(0);
    let pubkey = BggPublicKeyWire {
        matrix: outputs.remove(0),
        reveal_plaintext: output.pubkey.reveal_plaintext,
    };
    let plaintext = (!outputs.is_empty()).then(|| outputs.remove(0));
    Ok(BggEncodingWire { vector, pubkey, plaintext })
}

fn scalar_type(ambient: &mxx_ir_core::types::MatrixType) -> mxx_ir_core::types::MatrixType {
    mxx_ir_core::types::MatrixType {
        modulus: ambient.modulus.clone(),
        ring_dimension: ambient.ring_dimension.clone(),
        rows: mxx_ir_core::IntExpr::constant(1),
        columns: mxx_ir_core::IntExpr::constant(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_gadgets::{PolyElem, circuit::PublicLut};
    use mxx_ir_core::{IntExpr, types::MatrixType};
    use mxx_primitives::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    struct NoAdvancedGates;

    impl<W> AdvancedGateLowering<DCRTPoly, W> for NoAdvancedGates {
        fn slot_transfer(
            &mut self,
            _builder: &mut GraphBuilder,
            _input: &W,
            _source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot transfer",
            })
        }

        fn slot_reduce(
            &mut self,
            _builder: &mut GraphBuilder,
            _inputs: &[W],
            _slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            })
        }

        fn public_lookup(
            &mut self,
            _builder: &mut GraphBuilder,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            _input: &W,
            gate: GateInstance<'_>,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "public lookup",
            })
        }
    }

    #[derive(Default)]
    struct RecordingAdvancedGates {
        slot_instances: Vec<(Vec<usize>, usize, usize)>,
        lookup_instances: Vec<(Vec<usize>, usize, usize)>,
    }

    impl AdvancedGateLowering<DCRTPoly, BggPublicKeyWire> for RecordingAdvancedGates {
        fn slot_transfer(
            &mut self,
            _builder: &mut GraphBuilder,
            input: &BggPublicKeyWire,
            _source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<BggPublicKeyWire, CircuitCompileError> {
            self.slot_instances.push((
                gate.call_path().to_vec(),
                gate.local_gate().index(),
                gate.operation_occurrence(),
            ));
            Ok(input.clone())
        }

        fn slot_reduce(
            &mut self,
            _builder: &mut GraphBuilder,
            _inputs: &[BggPublicKeyWire],
            _slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<BggPublicKeyWire, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.local_gate().index(),
                kind: "slot reduction",
            })
        }

        fn public_lookup(
            &mut self,
            _builder: &mut GraphBuilder,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &BggPublicKeyWire,
            gate: GateInstance<'_>,
        ) -> Result<BggPublicKeyWire, CircuitCompileError> {
            self.lookup_instances.push((
                gate.call_path().to_vec(),
                gate.local_gate().index(),
                gate.operation_occurrence(),
            ));
            Ok(input.clone())
        }
    }

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn recursively_lowers_registered_subcircuits() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_inputs = child.input(2).to_vec();
        let child_output = child.add_gate(child_inputs[0], child_inputs[1]);
        child.output([child_output]);

        let mut parent = PolyCircuit::<DCRTPoly>::new();
        let parent_inputs = parent.input(2).to_vec();
        let child_id = parent.register_sub_circuit(child);
        let outputs = parent.call_sub_circuit(child_id, parent_inputs);
        parent.output(outputs);

        let mut builder = GraphBuilder::new("subcircuit", Vec::new());
        let one = BggPublicKeyWire {
            matrix: builder.input("one", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let inputs = [
            BggPublicKeyWire {
                matrix: builder.input("left", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: builder.input("right", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
        ];
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output = compiler
            .compile_public_keys_with_lowering(
                &mut builder,
                &parent,
                one,
                inputs,
                &mut NoAdvancedGates,
            )
            .expect("sub-circuit should lower");
        assert_eq!(output.len(), 1);
        builder.output("result", &output[0].matrix, ArtifactConfidentiality::Public);
        let graph = builder.finish();
        assert!(matches!(
            graph.nodes[graph.nodes.len() - 2].kind,
            mxx_ir_core::node::NodeKind::SubgraphCall(_)
        ));
        assert!(matches!(
            graph.nodes.last().expect("output node").kind,
            mxx_ir_core::node::NodeKind::Output { .. }
        ));
        let add_template = graph.subgraphs.get("bgg-public-key-add").expect("shared add template");
        assert!(add_template.nodes.iter().any(|node| matches!(
            node.kind,
            mxx_ir_core::node::NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Add)
        )));
    }

    #[test]
    fn advanced_lowering_receives_distinct_scoped_gate_instances() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_input = child.input(1).as_single_wire();
        let transferred = child.slot_transfer_gate(child_input, &[(0, None)]);
        child.output([transferred]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let child_id = circuit.register_sub_circuit(child);
        let first = circuit.call_sub_circuit(child_id, [input]);
        let second = circuit.call_sub_circuit(child_id, [input]);
        circuit.output([first[0], second[0]]);

        let mut builder = GraphBuilder::new("scoped-advanced-lowering", Vec::new());
        let one = BggPublicKeyWire {
            matrix: builder.input("one", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let input = BggPublicKeyWire {
            matrix: builder.input("input", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let mut lowering = RecordingAdvancedGates::default();
        let outputs = compiler
            .compile_public_keys_with_lowering(&mut builder, &circuit, one, [input], &mut lowering)
            .expect("advanced circuit should lower");
        assert_eq!(outputs.len(), 2);
        assert_eq!(lowering.slot_instances.len(), 2);
        assert_ne!(lowering.slot_instances[0].0, lowering.slot_instances[1].0);
        assert!(lowering.slot_instances.iter().all(|(_, _, occurrence)| *occurrence == 0));
    }

    #[test]
    fn composite_advanced_lowering_routes_lookup_and_slot_gates_to_separate_delegates() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        let transferred = circuit.slot_transfer_gate(input, &[(0, None)]);
        let lookup_id = circuit.register_public_lookup(PublicLut::new(
            &parameters,
            2,
            |parameters: &DCRTPolyParams, input| {
                Some((input, <DCRTPoly as Poly>::Elem::constant(&parameters.modulus(), input)))
            },
            None,
        ));
        let looked_up = circuit.public_lookup_gate(transferred, lookup_id);
        circuit.output([looked_up]);

        let mut builder = GraphBuilder::new("composite-advanced-lowering", Vec::new());
        let one = BggPublicKeyWire {
            matrix: builder.input("one", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let input = BggPublicKeyWire {
            matrix: builder.input("input", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let mut lowering = CompositeAdvancedGateLowering::new(
            RecordingAdvancedGates::default(),
            RecordingAdvancedGates::default(),
        );
        let outputs = compiler
            .compile_public_keys_with_lowering(&mut builder, &circuit, one, [input], &mut lowering)
            .expect("mixed advanced circuit should lower");
        assert_eq!(outputs.len(), 1);

        let (lookup, slots) = lowering.into_parts();
        assert_eq!(lookup.lookup_instances.len(), 1);
        assert!(lookup.slot_instances.is_empty());
        assert_eq!(slots.slot_instances.len(), 1);
        assert!(slots.lookup_instances.is_empty());
    }

    #[test]
    fn context_free_compilers_reject_advanced_gates_before_lowering() {
        let mut slot_transfer = PolyCircuit::<DCRTPoly>::new();
        let input = slot_transfer.input(1).as_single_wire();
        let output = slot_transfer.slot_transfer_gate(input, &[(0, None)]);
        slot_transfer.output([output]);

        let mut slot_reduce = PolyCircuit::<DCRTPoly>::new();
        let input = slot_reduce.input(1).as_single_wire();
        let output = slot_reduce.slot_reduce_gate(&[input], 1);
        slot_reduce.output([output]);

        let mut public_lookup = PolyCircuit::<DCRTPoly>::new();
        let input = public_lookup.input(1).as_single_wire();
        let output = public_lookup.public_lookup_gate(input, 0);
        public_lookup.output([output]);

        let mut nested_child = PolyCircuit::<DCRTPoly>::new();
        let input = nested_child.input(1).as_single_wire();
        let output = nested_child.slot_transfer_gate(input, &[(0, None)]);
        nested_child.output([output]);
        let mut nested = PolyCircuit::<DCRTPoly>::new();
        let input = nested.input(1).as_single_wire();
        let child = nested.register_sub_circuit(nested_child);
        let output = nested.call_sub_circuit(child, [input]);
        nested.output(output);

        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        for circuit in [slot_transfer, slot_reduce, public_lookup, nested] {
            assert!(circuit.requires_advanced_lowering());

            let mut builder = GraphBuilder::new("context-free-public-key", Vec::new());
            let one = BggPublicKeyWire {
                matrix: builder.input("one", matrix_type(2, 10)),
                reveal_plaintext: true,
            };
            let input = BggPublicKeyWire {
                matrix: builder.input("input", matrix_type(2, 10)),
                reveal_plaintext: true,
            };
            assert!(matches!(
                compiler.compile_public_keys(&mut builder, &circuit, one, [input]),
                Err(CircuitCompileError::AdvancedGateContextRequired)
            ));
            assert_eq!(builder.finish().nodes.len(), 2);

            let mut builder = GraphBuilder::new("context-free-encoding", Vec::new());
            let one = BggEncodingWire {
                vector: builder.input("one_vector", matrix_type(1, 10)),
                pubkey: BggPublicKeyWire {
                    matrix: builder.input("one_pubkey", matrix_type(2, 10)),
                    reveal_plaintext: false,
                },
                plaintext: None,
            };
            let input = BggEncodingWire {
                vector: builder.input("input_vector", matrix_type(1, 10)),
                pubkey: BggPublicKeyWire {
                    matrix: builder.input("input_pubkey", matrix_type(2, 10)),
                    reveal_plaintext: false,
                },
                plaintext: None,
            };
            assert!(matches!(
                compiler.compile_encodings(&mut builder, &circuit, one, [input]),
                Err(CircuitCompileError::AdvancedGateContextRequired)
            ));
            assert_eq!(builder.finish().nodes.len(), 4);

            let mut builder = GraphBuilder::new("context-free-poly-encoding", Vec::new());
            let one_vector = builder.input("one_vector", matrix_type(1, 10));
            let one = BggPolyEncodingWire {
                vectors: builder.family_pack(&[one_vector]).expect("one vector family"),
                pubkey: BggPublicKeyWire {
                    matrix: builder.input("one_pubkey", matrix_type(2, 10)),
                    reveal_plaintext: false,
                },
                plaintexts: None,
            };
            let input_vector = builder.input("input_vector", matrix_type(1, 10));
            let input = BggPolyEncodingWire {
                vectors: builder.family_pack(&[input_vector]).expect("input vector family"),
                pubkey: BggPublicKeyWire {
                    matrix: builder.input("input_pubkey", matrix_type(2, 10)),
                    reveal_plaintext: false,
                },
                plaintexts: None,
            };
            assert!(matches!(
                compiler.compile_poly_encodings(&mut builder, &circuit, one, [input]),
                Err(CircuitCompileError::AdvancedGateContextRequired)
            ));
            assert_eq!(builder.finish().nodes.len(), 6);
        }
    }

    #[test]
    fn repeated_gate_kind_reuses_one_registered_subgraph_template() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let first = circuit.add_gate(inputs[0], inputs[1]);
        let second = circuit.add_gate(first, inputs[0]);
        circuit.output([second]);

        let mut builder = GraphBuilder::new("template-reuse", Vec::new());
        let one = BggPublicKeyWire {
            matrix: builder.input("one", matrix_type(2, 10)),
            reveal_plaintext: true,
        };
        let supplied = [
            BggPublicKeyWire {
                matrix: builder.input("left", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
            BggPublicKeyWire {
                matrix: builder.input("right", matrix_type(2, 10)),
                reveal_plaintext: true,
            },
        ];
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output =
            compiler.compile_public_keys(&mut builder, &circuit, one, supplied).expect("circuit");
        builder.output("result", &output[0].matrix, ArtifactConfidentiality::Public);
        let graph = builder.finish();
        assert_eq!(graph.subgraphs.len(), 1);
        assert!(graph.subgraphs.contains_key("bgg-public-key-add"));
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, mxx_ir_core::node::NodeKind::SubgraphCall(_)))
                .count(),
            2
        );
    }
}
