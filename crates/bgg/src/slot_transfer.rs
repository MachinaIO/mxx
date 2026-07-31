use crate::{
    AdvancedGateLowering, CircuitCompileError, NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire,
};
use mxx_gadgets::{
    Poly,
    circuit::{GateInstance, PolyCircuit},
};
use mxx_ir_core::{
    GraphBuilder, IntExpr, MatrixFamilyWire, OutputFamilyError, node::MatrixBinaryOp,
};
use num_bigint::BigInt;
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SlotFamilyCompileError {
    #[error("slot transfer requires at least one destination slot")]
    EmptyTransfer,
    #[error("slot reduction requires at least one input family and one source slot")]
    EmptyReduction,
    #[error("slot reduction input count exceeds its source-slot count")]
    TooManyReductionInputs,
    #[error("naive Graph IR slot reduction requires homogeneous public-key reveal metadata")]
    RevealMetadataMismatch,
    #[error("naive Graph IR slot reduction requires homogeneous plaintext availability")]
    PlaintextAvailabilityMismatch,
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
}

#[derive(Clone, Debug, Default)]
pub struct NaiveBggSlotTransferCompiler;

impl NaiveBggSlotTransferCompiler {
    pub fn transfer_public_keys(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        source_slots: &[(u32, Option<u32>)],
    ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
        Ok(NaiveBggPublicKeyVecWire {
            matrices: transfer_matrix_family(builder, &input.matrices, source_slots)?,
            reveal_plaintext: input.reveal_plaintext,
        })
    }

    pub fn reduce_public_keys(
        &self,
        builder: &mut GraphBuilder,
        inputs: &[NaiveBggPublicKeyVecWire],
        source_slot_count: usize,
    ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
        // Graph IR indexed families have one bundle-level metadata value. The
        // old concrete container could carry different reveal flags in
        // different output members; this compiler deliberately supports the
        // homogeneous family invariant used by the Graph IR samplers and
        // rejects a reduction that would create heterogeneous metadata.
        let Some(first) = inputs.first() else {
            return Err(SlotFamilyCompileError::EmptyReduction);
        };
        if inputs.iter().any(|input| input.reveal_plaintext != first.reveal_plaintext) {
            return Err(SlotFamilyCompileError::RevealMetadataMismatch);
        }
        Ok(NaiveBggPublicKeyVecWire {
            matrices: reduce_matrix_families(
                builder,
                &inputs.iter().map(|input| input.matrices.clone()).collect::<Vec<_>>(),
                source_slot_count,
            )?,
            reveal_plaintext: first.reveal_plaintext,
        })
    }

    pub fn transfer_encodings(
        &self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        source_slots: &[(u32, Option<u32>)],
    ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
        Ok(NaiveBggEncodingVecWire {
            vectors: transfer_matrix_family(builder, &input.vectors, source_slots)?,
            pubkeys: transfer_matrix_family(builder, &input.pubkeys, source_slots)?,
            pubkey_reveal_plaintext: input.pubkey_reveal_plaintext,
            plaintexts: input
                .plaintexts
                .as_ref()
                .map(|plaintexts| transfer_matrix_family(builder, plaintexts, source_slots))
                .transpose()?,
        })
    }

    pub fn reduce_encodings(
        &self,
        builder: &mut GraphBuilder,
        inputs: &[NaiveBggEncodingVecWire],
        source_slot_count: usize,
    ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
        // As above, optional plaintext presence is homogeneous for one Graph
        // IR family. Reject mixed inputs instead of manufacturing placeholder
        // plaintext matrices or silently losing availability metadata.
        let Some(first) = inputs.first() else {
            return Err(SlotFamilyCompileError::EmptyReduction);
        };
        if inputs.iter().any(|input| input.pubkey_reveal_plaintext != first.pubkey_reveal_plaintext)
        {
            return Err(SlotFamilyCompileError::RevealMetadataMismatch);
        }
        let has_plaintexts = first.plaintexts.is_some();
        if inputs.iter().any(|input| input.plaintexts.is_some() != has_plaintexts) {
            return Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch);
        }
        let vectors = inputs.iter().map(|input| input.vectors.clone()).collect::<Vec<_>>();
        let pubkeys = inputs.iter().map(|input| input.pubkeys.clone()).collect::<Vec<_>>();
        let plaintexts = has_plaintexts
            .then(|| {
                inputs
                    .iter()
                    .map(|input| input.plaintexts.as_ref().expect("checked availability").clone())
                    .collect::<Vec<_>>()
            })
            .map(|plaintexts| reduce_matrix_families(builder, &plaintexts, source_slot_count))
            .transpose()?;
        Ok(NaiveBggEncodingVecWire {
            vectors: reduce_matrix_families(builder, &vectors, source_slot_count)?,
            pubkeys: reduce_matrix_families(builder, &pubkeys, source_slot_count)?,
            pubkey_reveal_plaintext: first.pubkey_reveal_plaintext,
            plaintexts,
        })
    }
}

impl<P: Poly> AdvancedGateLowering<P, NaiveBggPublicKeyVecWire> for NaiveBggSlotTransferCompiler {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &NaiveBggPublicKeyVecWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        self.transfer_public_keys(builder, input, source_slots).map_err(|_| {
            CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
        })
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[NaiveBggPublicKeyVecWire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        self.reduce_public_keys(builder, inputs, slot_count).map_err(|_| {
            CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
        })
    }

    fn public_lookup(
        &mut self,
        _builder: &mut GraphBuilder,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &NaiveBggPublicKeyVecWire,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggPublicKeyVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "public lookup",
        })
    }
}

impl<P: Poly> AdvancedGateLowering<P, NaiveBggEncodingVecWire> for NaiveBggSlotTransferCompiler {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &NaiveBggEncodingVecWire,
        source_slots: &[(u32, Option<u32>)],
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        self.transfer_encodings(builder, input, source_slots).map_err(|_| {
            CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
        })
    }

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[NaiveBggEncodingVecWire],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        self.reduce_encodings(builder, inputs, slot_count).map_err(|_| {
            CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
        })
    }

    fn public_lookup(
        &mut self,
        _builder: &mut GraphBuilder,
        _circuit: &PolyCircuit<P>,
        _lookup_id: usize,
        _input: &NaiveBggEncodingVecWire,
        gate: GateInstance<'_>,
    ) -> Result<NaiveBggEncodingVecWire, CircuitCompileError> {
        Err(CircuitCompileError::MissingGateContext {
            gate: gate.local_gate().index(),
            kind: "public lookup",
        })
    }
}

/// Selects and optionally scales members of one BGG matrix family.
///
/// Each `(source, scalar)` pair creates one destination member. Scalars use
/// the same constant-polynomial convention as `PolyCircuit` small scalar
/// multiplication.
pub(crate) fn transfer_matrix_family(
    builder: &mut GraphBuilder,
    input: &MatrixFamilyWire,
    source_slots: &[(u32, Option<u32>)],
) -> Result<MatrixFamilyWire, SlotFamilyCompileError> {
    if source_slots.is_empty() {
        return Err(SlotFamilyCompileError::EmptyTransfer);
    }
    let outputs = source_slots
        .iter()
        .map(|(source, scalar)| {
            let selected = builder.family_get_static(input, IntExpr::constant(*source));
            match scalar {
                Some(scalar) => {
                    let scalar =
                        builder.constant_polynomial(scalar_type(input), [BigInt::from(*scalar)]);
                    builder.matrix_binary(
                        MatrixBinaryOp::Multiply,
                        &selected,
                        &scalar,
                        input.matrix_type.clone(),
                    )
                }
                None => selected,
            }
        })
        .collect::<Vec<_>>();
    Ok(builder.family_pack(&outputs)?)
}

/// Reduces the first `source_slot_count` members of each input family into one
/// output member, using the basis polynomials `1, X, X^2, ...`.
///
/// Output member `i` is derived only from input family `i`, matching the
/// historical BGG slot-reduction layout.
///
/// Validation of the generated rotation constants also enforces
/// `source_slot_count <= ring_dimension`, so an oversized reduction is
/// rejected before execution instead of reaching a backend coefficient index.
pub(crate) fn reduce_matrix_families(
    builder: &mut GraphBuilder,
    inputs: &[MatrixFamilyWire],
    source_slot_count: usize,
) -> Result<MatrixFamilyWire, SlotFamilyCompileError> {
    if inputs.is_empty() || source_slot_count == 0 {
        return Err(SlotFamilyCompileError::EmptyReduction);
    }
    if inputs.len() > source_slot_count {
        return Err(SlotFamilyCompileError::TooManyReductionInputs);
    }
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut terms = Vec::with_capacity(source_slot_count);
        for source in 0..source_slot_count {
            let selected = builder.family_get_static(input, IntExpr::constant(source));
            let mut coefficients = vec![BigInt::from(0); source + 1];
            coefficients[source] = BigInt::from(1);
            let scalar = builder.constant_polynomial(scalar_type(input), coefficients);
            terms.push(builder.matrix_binary(
                MatrixBinaryOp::Multiply,
                &selected,
                &scalar,
                input.matrix_type.clone(),
            ));
        }
        let mut terms = terms.into_iter();
        let mut output = terms.next().expect("source_slot_count was checked nonzero");
        for term in terms {
            output = builder.matrix_binary(
                MatrixBinaryOp::Add,
                &output,
                &term,
                input.matrix_type.clone(),
            );
        }
        outputs.push(output);
    }
    Ok(builder.family_pack(&outputs)?)
}

fn scalar_type(input: &MatrixFamilyWire) -> mxx_ir_core::types::MatrixType {
    mxx_ir_core::types::MatrixType {
        modulus: input.matrix_type.modulus.clone(),
        ring_dimension: input.matrix_type.ring_dimension.clone(),
        rows: IntExpr::constant(1),
        columns: IntExpr::constant(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality, types::MatrixType, validate};
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
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn row(parameters: &DCRTPolyParams, columns: usize, offset: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            parameters,
            (0..columns)
                .map(|column| {
                    DCRTPoly::const_rotate_poly(
                        parameters,
                        (offset + column) % parameters.ring_dimension() as usize,
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn naive_transfer_and_reduce_match_the_legacy_slotwise_formulas() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let row_type = matrix_type(&parameters, 1, columns);
        let scalar_type = matrix_type(&parameters, 1, 1);
        let mut builder = GraphBuilder::new("naive-slot-transfer", Vec::new());
        let mut make_encoding = |prefix: &str| {
            let vectors = (0..3)
                .map(|slot| builder.input(format!("{prefix}_vector_{slot}"), row_type.clone()))
                .collect::<Vec<_>>();
            let pubkeys = (0..3)
                .map(|slot| builder.input(format!("{prefix}_pubkey_{slot}"), row_type.clone()))
                .collect::<Vec<_>>();
            let plaintexts = (0..3)
                .map(|slot| {
                    builder.input(format!("{prefix}_plaintext_{slot}"), scalar_type.clone())
                })
                .collect::<Vec<_>>();
            NaiveBggEncodingVecWire {
                vectors: builder.family_pack(&vectors).expect("vector family"),
                pubkeys: builder.family_pack(&pubkeys).expect("public-key family"),
                pubkey_reveal_plaintext: true,
                plaintexts: Some(builder.family_pack(&plaintexts).expect("plaintext family")),
            }
        };
        let first = make_encoding("first");
        let second = make_encoding("second");
        let compiler = NaiveBggSlotTransferCompiler;
        let transferred = compiler
            .transfer_encodings(&mut builder, &first, &[(2, Some(3)), (0, None)])
            .expect("valid transfer");
        let reduced =
            compiler.reduce_encodings(&mut builder, &[first, second], 3).expect("valid reduction");
        for (prefix, output, count) in
            [("transferred", &transferred, 2usize), ("reduced", &reduced, 2usize)]
        {
            for slot in 0..count {
                for (component, family) in [
                    ("vector", &output.vectors),
                    ("pubkey", &output.pubkeys),
                    ("plaintext", output.plaintexts.as_ref().expect("revealed plaintext family")),
                ] {
                    let value = builder.family_get_static(family, IntExpr::constant(slot));
                    builder.output(
                        format!("{prefix}_{component}_{slot}"),
                        &value,
                        ArtifactConfidentiality::Public,
                    );
                }
            }
        }
        let validated =
            validate(&builder.finish(), &ParamEnv::default()).expect("valid slot graph");

        let mut inputs = BTreeMap::new();
        let mut source_values = BTreeMap::<String, Vec<DCRTPolyMatrix>>::new();
        for (input_index, prefix) in ["first", "second"].into_iter().enumerate() {
            for (component_index, (component, width)) in
                [("vector", columns), ("pubkey", columns), ("plaintext", 1)].into_iter().enumerate()
            {
                let values = (0..3)
                    .map(|slot| {
                        row(&parameters, width, input_index * 9 + component_index * 3 + slot)
                    })
                    .collect::<Vec<_>>();
                for (slot, value) in values.iter().enumerate() {
                    inputs.insert(
                        format!("{prefix}_{component}_{slot}"),
                        RuntimeValue::matrix(value.clone()),
                    );
                }
                source_values.insert(format!("{prefix}_{component}"), values);
            }
        }
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(&validated, &mut backend, inputs, &mut store, SamplingMode::Fresh)
            .expect("slot graph execution");

        let transfer_scalar = DCRTPoly::from_u32s(&parameters, &[3]);
        for component in ["vector", "pubkey", "plaintext"] {
            let source = &source_values[&format!("first_{component}")];
            let expected = [source[2].clone() * transfer_scalar.clone(), source[0].clone()];
            for (slot, expected) in expected.into_iter().enumerate() {
                let RuntimeValue::Matrix(actual) =
                    &result.outputs[&format!("transferred_{component}_{slot}")]
                else {
                    panic!("transferred output")
                };
                assert_eq!(actual.as_ref(), &expected);
            }
        }
        for (output_slot, prefix) in ["first", "second"].into_iter().enumerate() {
            for component in ["vector", "pubkey", "plaintext"] {
                let source = &source_values[&format!("{prefix}_{component}")];
                let expected = source
                    .iter()
                    .enumerate()
                    .map(|(slot, value)| {
                        let mut coefficients = vec![0u32; slot + 1];
                        coefficients[slot] = 1;
                        value.clone() * DCRTPoly::from_u32s(&parameters, &coefficients)
                    })
                    .reduce(|lhs, rhs| lhs + rhs)
                    .expect("three source slots");
                let RuntimeValue::Matrix(actual) =
                    &result.outputs[&format!("reduced_{component}_{output_slot}")]
                else {
                    panic!("reduced output")
                };
                assert_eq!(actual.as_ref(), &expected);
            }
        }
    }

    #[test]
    fn naive_reduction_rejects_oversized_rotations_during_graph_validation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let row_type = matrix_type(&parameters, 1, parameters.modulus_digits());
        let mut builder = GraphBuilder::new("oversized-naive-reduction", Vec::new());
        let members = (0..9)
            .map(|slot| builder.input(format!("slot_{slot}"), row_type.clone()))
            .collect::<Vec<_>>();
        let input = NaiveBggPublicKeyVecWire {
            matrices: builder.family_pack(&members).expect("input family"),
            reveal_plaintext: true,
        };
        let output = NaiveBggSlotTransferCompiler
            .reduce_public_keys(&mut builder, &[input], 9)
            .expect("graph construction leaves parameter validation to ir-core");
        let first = builder.family_get_static(&output.matrices, IntExpr::constant(0));
        builder.output("output", &first, ArtifactConfidentiality::Public);

        let error = validate(&builder.finish(), &ParamEnv::default())
            .expect_err("source slot count exceeds ring dimension");
        assert!(error.to_string().contains("rotation exponent is out of range"));
    }

    #[test]
    fn naive_reduction_rejects_heterogeneous_bundle_metadata() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let row_type = matrix_type(&parameters, 1, parameters.modulus_digits());
        let scalar_type = matrix_type(&parameters, 1, 1);
        let mut builder = GraphBuilder::new("heterogeneous-naive-reduction", Vec::new());
        let rows = (0..2)
            .map(|slot| builder.input(format!("row_{slot}"), row_type.clone()))
            .collect::<Vec<_>>();
        let scalars = (0..2)
            .map(|slot| builder.input(format!("scalar_{slot}"), scalar_type.clone()))
            .collect::<Vec<_>>();
        let row_family = builder.family_pack(&rows).expect("row family");
        let scalar_family = builder.family_pack(&scalars).expect("scalar family");
        let public =
            NaiveBggPublicKeyVecWire { matrices: row_family.clone(), reveal_plaintext: true };
        let hidden =
            NaiveBggPublicKeyVecWire { matrices: row_family.clone(), reveal_plaintext: false };
        assert_eq!(
            NaiveBggSlotTransferCompiler.reduce_public_keys(&mut builder, &[public, hidden], 2),
            Err(SlotFamilyCompileError::RevealMetadataMismatch)
        );

        let revealed = NaiveBggEncodingVecWire {
            vectors: row_family.clone(),
            pubkeys: row_family.clone(),
            pubkey_reveal_plaintext: true,
            plaintexts: Some(scalar_family),
        };
        let unavailable = NaiveBggEncodingVecWire {
            vectors: row_family.clone(),
            pubkeys: row_family,
            pubkey_reveal_plaintext: true,
            plaintexts: None,
        };
        assert_eq!(
            NaiveBggSlotTransferCompiler.reduce_encodings(
                &mut builder,
                &[revealed, unavailable],
                2
            ),
            Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch)
        );
    }
}
