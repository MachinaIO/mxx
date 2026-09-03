//! Slot transfer, reduction, and rotation support for BGG+ graph values.

mod naive {
    use crate::{NaiveBggEncodingVecWire, NaiveBggPublicKeyVecWire};
    use mxx_dsl::{DslError, Family, Mat, Ring};
    use mxx_ir_core::IntExpr;
    use rayon::prelude::*;
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum SlotFamilyCompileError {
        #[error("slot transfer requires at least one destination slot")]
        EmptyTransfer,
        #[error("slot reduction requires at least one input family and one source slot")]
        EmptyReduction,
        #[error("slot reduction input count exceeds its source-slot count")]
        TooManyReductionInputs,
        #[error("slot reduction requires homogeneous public-key reveal metadata")]
        RevealMetadataMismatch,
        #[error("slot reduction requires homogeneous plaintext availability")]
        PlaintextAvailabilityMismatch,
        #[error(transparent)]
        Dsl(#[from] DslError),
    }

    #[derive(Clone, Debug, Default)]
    pub struct NaiveBggSlotTransferCompiler;

    impl NaiveBggSlotTransferCompiler {
        pub fn transfer_public_keys(
            &self,
            input: &NaiveBggPublicKeyVecWire,
            source_slots: &[(u32, Option<u32>)],
        ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
            Ok(NaiveBggPublicKeyVecWire {
                matrices: transfer_matrix_family(&input.matrices, source_slots)?,
                reveal_plaintext: input.reveal_plaintext,
            })
        }

        pub fn reduce_public_keys(
            &self,
            inputs: &[NaiveBggPublicKeyVecWire],
            source_slot_count: usize,
        ) -> Result<NaiveBggPublicKeyVecWire, SlotFamilyCompileError> {
            let Some(first) = inputs.first() else {
                return Err(SlotFamilyCompileError::EmptyReduction);
            };
            if inputs.par_iter().any(|input| input.reveal_plaintext != first.reveal_plaintext) {
                return Err(SlotFamilyCompileError::RevealMetadataMismatch);
            }
            Ok(NaiveBggPublicKeyVecWire {
                matrices: reduce_matrix_families(
                    &inputs.par_iter().map(|input| input.matrices.clone()).collect::<Vec<_>>(),
                    source_slot_count,
                )?,
                reveal_plaintext: first.reveal_plaintext,
            })
        }

        pub fn transfer_encodings(
            &self,
            input: &NaiveBggEncodingVecWire,
            source_slots: &[(u32, Option<u32>)],
        ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
            Ok(NaiveBggEncodingVecWire {
                vectors: transfer_matrix_family(&input.vectors, source_slots)?,
                pubkeys: transfer_matrix_family(&input.pubkeys, source_slots)?,
                pubkey_reveal_plaintext: input.pubkey_reveal_plaintext,
                plaintexts: input
                    .plaintexts
                    .as_ref()
                    .map(|plaintexts| transfer_matrix_family(plaintexts, source_slots))
                    .transpose()?,
            })
        }

        pub fn reduce_encodings(
            &self,
            inputs: &[NaiveBggEncodingVecWire],
            source_slot_count: usize,
        ) -> Result<NaiveBggEncodingVecWire, SlotFamilyCompileError> {
            let Some(first) = inputs.first() else {
                return Err(SlotFamilyCompileError::EmptyReduction);
            };
            if inputs
                .par_iter()
                .any(|input| input.pubkey_reveal_plaintext != first.pubkey_reveal_plaintext)
            {
                return Err(SlotFamilyCompileError::RevealMetadataMismatch);
            }
            let has_plaintexts = first.plaintexts.is_some();
            if inputs.par_iter().any(|input| input.plaintexts.is_some() != has_plaintexts) {
                return Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch);
            }
            let vectors = inputs.par_iter().map(|input| input.vectors.clone()).collect::<Vec<_>>();
            let pubkeys = inputs.par_iter().map(|input| input.pubkeys.clone()).collect::<Vec<_>>();
            let plaintexts = has_plaintexts
                .then(|| {
                    inputs
                        .par_iter()
                        .map(|input| input.plaintexts.as_ref().expect("checked").clone())
                        .collect::<Vec<_>>()
                })
                .map(|families| reduce_matrix_families(&families, source_slot_count))
                .transpose()?;
            Ok(NaiveBggEncodingVecWire {
                vectors: reduce_matrix_families(&vectors, source_slot_count)?,
                pubkeys: reduce_matrix_families(&pubkeys, source_slot_count)?,
                pubkey_reveal_plaintext: first.pubkey_reveal_plaintext,
                plaintexts,
            })
        }
    }

    fn transfer_matrix_family(
        input: &Family<Mat>,
        source_slots: &[(u32, Option<u32>)],
    ) -> Result<Family<Mat>, SlotFamilyCompileError> {
        if source_slots.is_empty() {
            return Err(SlotFamilyCompileError::EmptyTransfer);
        }
        let descriptors = source_slots
            .par_iter()
            .map(|(source, scalar)| (usize::try_from(*source).expect("u32 fits usize"), *scalar))
            .collect::<Vec<_>>();
        let ty = input.element_type();
        let ring = Ring::new(ty.modulus.clone(), ty.ring_dimension.clone());
        let outputs = descriptors
            .into_iter()
            .map(|(source, scalar)| {
                let selected = input.get_static(source);
                scalar.map_or(selected.clone(), |scalar| {
                    selected * ring.polynomial([IntExpr::constant(scalar)])
                })
            })
            .collect();
        Ok(Family::pack(outputs)?)
    }

    fn reduce_matrix_families(
        inputs: &[Family<Mat>],
        source_slot_count: usize,
    ) -> Result<Family<Mat>, SlotFamilyCompileError> {
        if inputs.is_empty() || source_slot_count == 0 {
            return Err(SlotFamilyCompileError::EmptyReduction);
        }
        if inputs.len() > source_slot_count {
            return Err(SlotFamilyCompileError::TooManyReductionInputs);
        }
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let ty = input.element_type();
            let ring = Ring::new(ty.modulus.clone(), ty.ring_dimension.clone());
            let mut terms = (0..source_slot_count).map(|source| {
                input.get_static(source) *
                    ring.polynomial(
                        (0..=source).map(|index| IntExpr::constant(usize::from(index == source))),
                    )
            });
            let first = terms.next().expect("nonzero source slot count");
            outputs.push(terms.fold(first, |sum, term| sum + term));
        }
        Ok(Family::pack(outputs)?)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_utils::{execute_graph, matrix_output, row};
        use mxx_dsl::{DslContext, Family};
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::{
            matrix::dcrt_poly::DCRTPolyMatrix,
            poly::{
                Poly, PolyParams,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use mxx_runtime::RuntimeValue;
        use num_bigint::BigInt;
        use std::collections::BTreeMap;

        #[test]
        fn runtime_transfer_and_reduce_match_the_slotwise_primitive_formulas() {
            let parameters = DCRTPolyParams::new(8, 1, 20, 4);
            let columns = parameters.modulus_digits();
            let ring = Ring::new(
                BigInt::from(parameters.modulus().as_ref().clone()),
                parameters.ring_dimension() as usize,
            );
            let mut inputs = BTreeMap::new();
            let mut source_values = BTreeMap::<String, Vec<DCRTPolyMatrix>>::new();
            let mut make_encoding = |prefix: &str| {
                let mut make_family = |component: &str, width: usize, component_index: usize| {
                    let values = (0..3)
                        .map(|slot| {
                            row(
                                &parameters,
                                width,
                                usize::from(prefix == "second") * 9 + component_index * 3 + slot,
                            )
                        })
                        .collect::<Vec<_>>();
                    let wires = values
                        .iter()
                        .enumerate()
                        .map(|(slot, value)| {
                            let name = format!("{prefix}_{component}_{slot}");
                            inputs.insert(name.clone(), RuntimeValue::matrix(value.clone()));
                            ring.input(name, (1, width))
                        })
                        .collect();
                    source_values.insert(format!("{prefix}_{component}"), values);
                    Family::pack(wires).expect("three-member input family")
                };
                NaiveBggEncodingVecWire {
                    vectors: make_family("vector", columns, 0),
                    pubkeys: make_family("pubkey", columns, 1),
                    pubkey_reveal_plaintext: true,
                    plaintexts: Some(make_family("plaintext", 1, 2)),
                }
            };
            let first = make_encoding("first");
            let second = make_encoding("second");
            let compiler = NaiveBggSlotTransferCompiler;
            let transferred = compiler
                .transfer_encodings(&first, &[(2, Some(3)), (0, None)])
                .expect("valid transfer");
            let reduced = compiler.reduce_encodings(&[first, second], 3).expect("valid reduction");

            let mut context = DslContext::new("naive-slot-transfer-runtime");
            for (prefix, output, count) in
                [("transferred", &transferred, 2usize), ("reduced", &reduced, 2usize)]
            {
                for slot in 0..count {
                    for (component, family) in [
                        ("vector", &output.vectors),
                        ("pubkey", &output.pubkeys),
                        ("plaintext", output.plaintexts.as_ref().expect("plaintext family")),
                    ] {
                        context = context
                            .output(format!("{prefix}_{component}_{slot}"), family.get_static(slot))
                            .expect("matrix output");
                    }
                }
            }
            let result =
                execute_graph(context.build().expect("runtime graph"), parameters.clone(), inputs);

            let transfer_scalar = DCRTPoly::from_u32s(&parameters, &[3]);
            for component in ["vector", "pubkey", "plaintext"] {
                let source = &source_values[&format!("first_{component}")];
                let expected = [source[2].clone() * transfer_scalar.clone(), source[0].clone()];
                for (slot, expected) in expected.into_iter().enumerate() {
                    assert_eq!(
                        matrix_output(&result, &format!("transferred_{component}_{slot}")),
                        &expected
                    );
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
                        .reduce(|left, right| left + right)
                        .expect("three source slots");
                    assert_eq!(
                        matrix_output(&result, &format!("reduced_{component}_{output_slot}")),
                        &expected
                    );
                }
            }
        }

        #[test]
        fn slot_transfer_and_reduction_preserve_heterogeneous_member_graphs() {
            let ring = Ring::new(257, 8);
            let matrices =
                Family::pack(vec![ring.gaussian((1, 1), 2, 13), ring.gaussian((1, 1), 3, 20)])
                    .expect("family");
            let input = NaiveBggPublicKeyVecWire { matrices, reveal_plaintext: true };
            let compiler = NaiveBggSlotTransferCompiler;
            let transferred =
                compiler.transfer_public_keys(&input, &[(1, None), (0, None)]).expect("transfer");
            let reduced = compiler.reduce_public_keys(&[input], 2).expect("reduction");
            let built = DslContext::new("slot-symbolics")
                .output("transferred", transferred.matrices.get_static(0))
                .expect("transfer output")
                .output("reduced", reduced.matrices.get_static(0))
                .expect("reduction output")
                .build()
                .expect("build");
            built.validate(&ParamEnv::default()).expect("valid executable graph");
        }

        #[test]
        fn reduction_rejects_oversized_rotations_during_validation() {
            let ring = Ring::new(17, 8);
            let input = NaiveBggPublicKeyVecWire {
                matrices: Family::pack(
                    (0..9).map(|slot| ring.input(format!("slot-{slot}"), (1, 2))).collect(),
                )
                .unwrap(),
                reveal_plaintext: true,
            };
            let output = NaiveBggSlotTransferCompiler
                .reduce_public_keys(&[input], 9)
                .expect("construction leaves rotation validation to ir-core");
            let graph = DslContext::new("oversized-slot-reduction")
                .output("output", output.matrices.get_static(0))
                .unwrap()
                .build()
                .unwrap();
            let error = graph
                .validate(&ParamEnv::default())
                .expect_err("rotation exponent exceeds ring dimension");
            assert!(error.to_string().contains("constant polynomial exceeds the ring dimension"));
        }

        #[test]
        fn reduction_rejects_heterogeneous_family_metadata() {
            let ring = Ring::new(17, 8);
            let rows = Family::pack(vec![ring.input("row-0", (1, 2)), ring.input("row-1", (1, 2))])
                .unwrap();
            let scalars =
                Family::pack(vec![ring.input("scalar-0", (1, 1)), ring.input("scalar-1", (1, 1))])
                    .unwrap();
            let public =
                NaiveBggPublicKeyVecWire { matrices: rows.clone(), reveal_plaintext: true };
            let hidden =
                NaiveBggPublicKeyVecWire { matrices: rows.clone(), reveal_plaintext: false };
            assert!(matches!(
                NaiveBggSlotTransferCompiler.reduce_public_keys(&[public, hidden], 2),
                Err(SlotFamilyCompileError::RevealMetadataMismatch)
            ));

            let revealed = NaiveBggEncodingVecWire {
                vectors: rows.clone(),
                pubkeys: rows.clone(),
                pubkey_reveal_plaintext: true,
                plaintexts: Some(scalars),
            };
            let unavailable = NaiveBggEncodingVecWire {
                vectors: rows.clone(),
                pubkeys: rows,
                pubkey_reveal_plaintext: true,
                plaintexts: None,
            };
            assert!(matches!(
                NaiveBggSlotTransferCompiler.reduce_encodings(&[revealed, unavailable], 2),
                Err(SlotFamilyCompileError::PlaintextAvailabilityMismatch)
            ));
        }
    }
}
pub use naive::*;

mod public_key {
    use crate::{BggPublicKeyCompiler, BggPublicKeyWire, CircuitCompileError};
    use mxx_dsl::{Bytes, HashTag, Mat};
    use mxx_gadgets::{
        Poly,
        circuit::{CircuitLoweringTypes, GateInstance, SlotOperationLowering},
    };
    use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId};
    use rayon::prelude::*;

    #[derive(Clone)]
    pub enum BggSlotTransferGateRequest {
        Transfer {
            identity: String,
            input_public_key: Mat,
            output_public_key: Mat,
            source_slots: Vec<(u32, Option<u32>)>,
        },
        Reduce {
            identity: String,
            input_public_keys: Vec<Mat>,
            output_public_key: Mat,
            source_slot_count: usize,
        },
    }

    #[derive(Clone)]
    pub struct BggSlotTransferPublicKeyLowering {
        pub compiler: BggPublicKeyCompiler,
        pub hash_key: Bytes,
        pub public_key_type: mxx_ir_core::types::MatrixType,
        pub configured_slot_count: usize,
        /// Producer containing the exact gate output public keys, when this
        /// lowering is used by a consumer graph.
        pub output_public_key_production: Option<ProductionId>,
        pub requests: Vec<BggSlotTransferGateRequest>,
    }

    impl BggSlotTransferPublicKeyLowering {
        fn output_public_key(&self, gate: GateInstance<'_>, reduction: bool) -> BggPublicKeyWire {
            let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
            let identity = gate_token(gate);
            let matrix = self.output_public_key_production.as_ref().map_or_else(
                || {
                    self.compiler.ring.hash_matrix(
                        self.hash_key.clone(),
                        HashTag::from(format!("{operation}_gate_a_out_{identity}").into_bytes()),
                        (self.public_key_type.rows.clone(), self.public_key_type.columns.clone()),
                    )
                },
                |production| {
                    self.compiler.ring.artifact_input(
                        production.clone(),
                        super::slot_gate_public_key_name(reduction, &identity),
                        (self.public_key_type.rows.clone(), self.public_key_type.columns.clone()),
                        ArtifactConfidentiality::Public,
                    )
                },
            );
            BggPublicKeyWire { matrix, reveal_plaintext: true }
        }

        fn valid_type(&self, input: &BggPublicKeyWire) -> bool {
            input.matrix.matrix_type() == &self.public_key_type
        }
    }

    impl CircuitLoweringTypes for BggSlotTransferPublicKeyLowering {
        type Wire = BggPublicKeyWire;
        type Error = CircuitCompileError;
    }

    impl<P: Poly> SlotOperationLowering<P> for BggSlotTransferPublicKeyLowering {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            if !self.valid_type(input) ||
                source_slots.len() > self.configured_slot_count ||
                source_slots
                    .par_iter()
                    .any(|(source, _)| *source as usize >= self.configured_slot_count)
            {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            let output = self.output_public_key(gate, false);
            self.requests.push(BggSlotTransferGateRequest::Transfer {
                identity: gate_token(gate),
                input_public_key: input.matrix.clone(),
                output_public_key: output.matrix.clone(),
                source_slots: source_slots.to_vec(),
            });
            Ok(output)
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            if inputs.is_empty() ||
                inputs.len() > slot_count ||
                slot_count > self.configured_slot_count ||
                inputs.par_iter().any(|input| !self.valid_type(input))
            {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            let output = self.output_public_key(gate, true);
            self.requests.push(BggSlotTransferGateRequest::Reduce {
                identity: gate_token(gate),
                input_public_keys: inputs.par_iter().map(|input| input.matrix.clone()).collect(),
                output_public_key: output.matrix.clone(),
                source_slot_count: slot_count,
            });
            Ok(output)
        }

        fn slot_anchor_reduce(
            &mut self,
            _input: &Self::Wire,
            _num_blocks: u32,
            _lane_scalars: &[num_bigint::BigUint],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            Err(CircuitCompileError::Unsupported {
                gate: gate.local_gate().index(),
                feature: "anchor reduction in preimage-based slot lowering",
            })
        }
    }

    pub(crate) fn gate_token(gate: GateInstance<'_>) -> String {
        let mut token = gate.call_path().iter().map(usize::to_string).collect::<Vec<_>>().join("_");
        if !token.is_empty() {
            token.push('_');
        }
        token.push_str(&format!("g{}_o{}", gate.local_gate().index(), gate.operation_occurrence()));
        token
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{NoPublicLookup, PolyCircuitCompiler};
        use mxx_dsl::{DslContext, Ring};
        use mxx_gadgets::circuit::PolyCircuit;
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::poly::dcrt::poly::DCRTPoly;

        #[test]
        fn public_key_slot_transfer_lowering_builds_and_symbolically_elaborates() {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let input_gate = circuit.input(1).as_single_wire();
            let transferred = circuit.slot_transfer_gate(input_gate, &[(1, None), (0, Some(3))]);
            circuit.output([transferred]);

            let ring = Ring::new(257, 8);
            let public_key_type = ring.matrix_type((2, 8));
            let key = |name: &str| BggPublicKeyWire {
                matrix: ring.input(name, (2, 8)),
                reveal_plaintext: true,
            };
            let public_key_compiler =
                BggPublicKeyCompiler { ring: ring.clone(), base: 4.into(), digit_count: 4.into() };
            let mut lowering = BggSlotTransferPublicKeyLowering {
                compiler: public_key_compiler.clone(),
                hash_key: ring.bytes_input("hash-key", 32),
                public_key_type,
                configured_slot_count: 2,
                output_public_key_production: None,
                requests: Vec::new(),
            };
            let mut lookup = NoPublicLookup::default();
            let outputs = PolyCircuitCompiler { public_key: public_key_compiler }
                .compile_public_keys_with_lowerings(
                    &circuit,
                    key("one"),
                    [key("input")],
                    &mut lookup,
                    &mut lowering,
                )
                .expect("slot-transfer lowering");
            assert_eq!(lowering.requests.len(), 1);
            let built = DslContext::new("slot-transfer-public-key")
                .output("output", outputs[0].matrix.clone())
                .expect("output")
                .build()
                .expect("build");
            built.validate(&ParamEnv::default()).expect("valid executable graph");
        }
    }
}
pub use public_key::*;

fn slot_gate_public_key_name(reduction: bool, identity: &str) -> String {
    let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
    format!("{operation}_gate_{identity}_public_key")
}

mod artifact {
    use crate::BggSlotTransferGateRequest;
    use mxx_dsl::{
        Bytes, DslContext, DslError, Family, HashTag, Mat, Parallel, Preimage, Ring, Trapdoor,
    };
    use mxx_ir_core::{
        IntExpr, RealExpr,
        artifact::{ArtifactConfidentiality, ProductionId},
        node::{ConcatAxis, ConstantMatrix, IndexRange},
        types::MatrixType,
    };
    use rayon::prelude::*;
    use std::collections::BTreeMap;
    use thiserror::Error;

    const B0_PUBLIC: &str = "slot_transfer_b0_public";
    const B0_TRAPDOOR: &str = "slot_transfer_b0_trapdoor";
    const B1_PUBLIC: &str = "slot_transfer_b1_public";
    const B1_TRAPDOOR: &str = "slot_transfer_b1_trapdoor";
    const SLOT_SECRET: &str = "slot_transfer_slot_secret";
    const SLOT_PUBLIC_KEY: &str = "slot_transfer_slot_a";

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct BggSlotTransferArtifactCompiler {
        pub modulus: IntExpr,
        pub ring_dimension: IntExpr,
        pub secret_size: usize,
        pub slot_count: usize,
        pub digit_count: usize,
        pub chunk_columns: usize,
        pub gadget_base: IntExpr,
        pub trapdoor_sigma: RealExpr,
        pub error_sigma: RealExpr,
        pub preimage_max_coefficient_bound: IntExpr,
        pub error_max_coefficient_bound: IntExpr,
    }

    #[derive(Debug, Error)]
    pub enum BggSlotTransferArtifactError {
        #[error("slot-transfer dimensions, slot count, and chunk width must be nonzero")]
        EmptyLayout,
        #[error("slot-transfer gate request is incompatible with the artifact layout")]
        InvalidGateRequest,
        #[error("slot-transfer artifact family is missing: {0}")]
        MissingArtifact(String),
        #[error(transparent)]
        Dsl(#[from] DslError),
    }

    #[derive(Clone)]
    pub struct BggSlotTransferBaseWires {
        pub b0: Trapdoor,
        pub b1: Trapdoor,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct BggSlotTransferBaseArtifacts {
        pub production_id: ProductionId,
    }

    #[derive(Clone)]
    pub struct BggSlotTransferSlotWires {
        pub secrets: Family<Mat>,
        pub public_keys: Family<Mat>,
        pub b0_preimage_chunks: Vec<Family<Preimage>>,
        pub b1_preimage_chunks: Vec<Family<Preimage>>,
    }

    #[derive(Clone)]
    pub struct BggSlotTransferPublicSlotWires {
        pub public_keys: Family<Mat>,
        pub b0_preimage_chunks: Vec<Family<Preimage>>,
        pub b1_preimage_chunks: Vec<Family<Preimage>>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct BggSlotTransferSlotArtifacts {
        pub production_id: ProductionId,
    }

    #[derive(Clone, Default)]
    pub struct BggSlotTransferGateWires {
        pub preimage_chunks: BTreeMap<String, Family<Preimage>>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct BggSlotTransferGateArtifacts {
        pub production_id: ProductionId,
    }

    impl BggSlotTransferArtifactCompiler {
        pub fn validate_layout(&self) -> Result<(), BggSlotTransferArtifactError> {
            if self.secret_size == 0 ||
                self.slot_count == 0 ||
                self.digit_count == 0 ||
                self.chunk_columns == 0
            {
                return Err(BggSlotTransferArtifactError::EmptyLayout);
            }
            Ok(())
        }

        pub fn public_key_type(&self) -> MatrixType {
            self.matrix_type(self.secret_size, self.gadget_columns())
        }

        pub fn build_base(&self) -> Result<BggSlotTransferBaseWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let ring = self.ring();
            Ok(BggSlotTransferBaseWires {
                b0: ring.sample_trapdoor(
                    self.secret_size,
                    self.trapdoor_sigma.clone(),
                    self.gadget_base.clone(),
                    self.digit_count,
                    self.preimage_max_coefficient_bound.clone(),
                ),
                b1: ring.sample_trapdoor(
                    self.secret_size * 2,
                    self.trapdoor_sigma.clone(),
                    self.gadget_base.clone(),
                    self.digit_count,
                    self.preimage_max_coefficient_bound.clone(),
                ),
            })
        }

        pub fn export_base(
            &self,
            context: DslContext,
            base: BggSlotTransferBaseWires,
        ) -> Result<DslContext, BggSlotTransferArtifactError> {
            Ok(context
                .public_output(B0_PUBLIC, base.b0.public_matrix())?
                .private_trapdoor_output(B0_TRAPDOOR, base.b0)?
                .public_output(B1_PUBLIC, base.b1.public_matrix())?
                .private_trapdoor_output(B1_TRAPDOOR, base.b1)?)
        }

        pub fn import_base(
            &self,
            artifacts: &BggSlotTransferBaseArtifacts,
        ) -> Result<BggSlotTransferBaseWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let ring = self.ring();
            Ok(BggSlotTransferBaseWires {
                b0: ring.trapdoor_artifact_input(
                    artifacts.production_id.clone(),
                    B0_PUBLIC,
                    B0_TRAPDOOR,
                    self.secret_size,
                    self.trapdoor_sigma.clone(),
                    self.gadget_base.clone(),
                    self.digit_count,
                    self.preimage_max_coefficient_bound.clone(),
                ),
                b1: ring.trapdoor_artifact_input(
                    artifacts.production_id.clone(),
                    B1_PUBLIC,
                    B1_TRAPDOOR,
                    self.secret_size * 2,
                    self.trapdoor_sigma.clone(),
                    self.gadget_base.clone(),
                    self.digit_count,
                    self.preimage_max_coefficient_bound.clone(),
                ),
            })
        }

        pub fn build_slots(
            &self,
            hash_key: Bytes,
            base: &BggSlotTransferBaseWires,
        ) -> Result<BggSlotTransferSlotWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let ring = self.ring();
            let secret_size = self.secret_size;
            let public_columns = self.gadget_columns();
            let (secrets, public_keys) = Parallel::range(self.slot_count).map_values({
                let ring = ring.clone();
                move |index| {
                    let mut tag = HashTag::from(b"slot_transfer_slot_a_".as_slice());
                    tag.push_decimal(index);
                    (
                        ring.uniform_interval((secret_size, secret_size), -1, 1),
                        ring.hash_matrix(hash_key.clone(), tag, (secret_size, public_columns)),
                    )
                }
            })?;

            let identity = ring.identity(secret_size);
            let b1_public = base.b1.public_matrix();
            let b0_preimage_chunks = self
                .chunks(self.b1_public_columns())
                .into_iter()
                .map(|columns| {
                    let target_columns = columns.clone();
                    let error_sigma = self.error_sigma.clone();
                    let ring = ring.clone();
                    let b0 = base.b0.clone();
                    let b1_public = b1_public.clone();
                    let identity = identity.clone();
                    secrets.clone().parallel_map_values(move |_, secret| {
                        let secret_identity =
                            Mat::concat(ConcatAxis::Columns, vec![secret, identity.clone()]);
                        let target = secret_identity *
                            b1_public.clone().slice(None, Some(target_columns.clone()));
                        let columns = target.matrix_type().columns.clone();
                        b0.sample_preimage(
                            target +
                                ring.gaussian(
                                    (secret_size, columns.clone()),
                                    error_sigma.clone(),
                                    self.error_max_coefficient_bound.clone(),
                                ),
                            (b0.public_matrix().matrix_type().columns.clone(), columns),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let gadget = ring.gadget(secret_size, self.gadget_base.clone(), self.digit_count);
            let b1_preimage_chunks = self
                .chunks(self.gadget_columns())
                .into_iter()
                .map(|columns| {
                    let target_columns = columns.clone();
                    let error_sigma = self.error_sigma.clone();
                    let ring = ring.clone();
                    let b1 = base.b1.clone();
                    let gadget = gadget.clone();
                    secrets.clone().parallel_zip_values(
                        public_keys.clone(),
                        move |_, secret, public_key| {
                            let a_chunk = public_key.slice(None, Some(target_columns.clone()));
                            let gadget_chunk =
                                gadget.clone().slice(None, Some(target_columns.clone()));
                            let secret_gadget = -(secret * gadget_chunk);
                            let target =
                                Mat::concat(ConcatAxis::Rows, vec![a_chunk, secret_gadget]);
                            let columns = target.matrix_type().columns.clone();
                            b1.sample_preimage(
                                target +
                                    ring.gaussian(
                                        (secret_size * 2, columns.clone()),
                                        error_sigma.clone(),
                                        self.error_max_coefficient_bound.clone(),
                                    ),
                                (b1.public_matrix().matrix_type().columns.clone(), columns),
                            )
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(BggSlotTransferSlotWires {
                secrets,
                public_keys,
                b0_preimage_chunks,
                b1_preimage_chunks,
            })
        }

        pub fn export_slots(
            &self,
            context: DslContext,
            slots: BggSlotTransferSlotWires,
        ) -> Result<DslContext, BggSlotTransferArtifactError> {
            let context = context
                .private_family_output(SLOT_SECRET, slots.secrets)?
                .public_family_output(SLOT_PUBLIC_KEY, slots.public_keys)?;
            let context = slots.b0_preimage_chunks.into_iter().enumerate().try_fold(
                context,
                |context, (chunk, family)| {
                    context.public_preimage_family_output(b0_preimage_name(chunk), family)
                },
            )?;
            Ok(slots.b1_preimage_chunks.into_iter().enumerate().try_fold(
                context,
                |context, (chunk, family)| {
                    context.public_preimage_family_output(b1_preimage_name(chunk), family)
                },
            )?)
        }

        pub fn import_slots(
            &self,
            artifacts: &BggSlotTransferSlotArtifacts,
        ) -> Result<BggSlotTransferSlotWires, BggSlotTransferArtifactError> {
            let ring = self.ring();
            let public = self.import_slots_public(artifacts)?;
            Ok(BggSlotTransferSlotWires {
                secrets: ring.family_artifact_input(
                    artifacts.production_id.clone(),
                    SLOT_SECRET,
                    self.slot_count,
                    (self.secret_size, self.secret_size),
                    ArtifactConfidentiality::Private,
                ),
                public_keys: public.public_keys,
                b0_preimage_chunks: public.b0_preimage_chunks,
                b1_preimage_chunks: public.b1_preimage_chunks,
            })
        }

        pub fn import_slots_public(
            &self,
            artifacts: &BggSlotTransferSlotArtifacts,
        ) -> Result<BggSlotTransferPublicSlotWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let ring = self.ring();
            Ok(BggSlotTransferPublicSlotWires {
                public_keys: ring.family_artifact_input(
                    artifacts.production_id.clone(),
                    SLOT_PUBLIC_KEY,
                    self.slot_count,
                    (self.secret_size, self.gadget_columns()),
                    ArtifactConfidentiality::Public,
                ),
                b0_preimage_chunks: self.import_slot_chunks(
                    &artifacts.production_id,
                    true,
                    self.b1_public_columns(),
                    self.b0_public_columns(),
                ),
                b1_preimage_chunks: self.import_slot_chunks(
                    &artifacts.production_id,
                    false,
                    self.gadget_columns(),
                    self.b1_public_columns(),
                ),
            })
        }

        pub fn build_gate_preimages(
            &self,
            base: &BggSlotTransferBaseWires,
            slots: &BggSlotTransferSlotWires,
            requests: &[BggSlotTransferGateRequest],
        ) -> Result<BggSlotTransferGateWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let mut preimage_chunks = BTreeMap::new();
            for request in requests {
                self.validate_gate_request(request)?;
                for (chunk, columns) in self.chunks(self.gadget_columns()).into_iter().enumerate() {
                    let (name, family) = match request {
                        BggSlotTransferGateRequest::Transfer {
                            identity,
                            input_public_key,
                            output_public_key,
                            source_slots,
                        } => (
                            gate_preimage_name(false, identity, chunk),
                            self.build_transfer_gate_chunk(
                                base,
                                slots,
                                input_public_key,
                                output_public_key,
                                source_slots,
                                columns,
                            )?,
                        ),
                        BggSlotTransferGateRequest::Reduce {
                            identity,
                            input_public_keys,
                            output_public_key,
                            source_slot_count,
                        } => (
                            gate_preimage_name(true, identity, chunk),
                            self.build_reduce_gate_chunk(
                                base,
                                slots,
                                input_public_keys,
                                output_public_key,
                                *source_slot_count,
                                columns,
                            )?,
                        ),
                    };
                    preimage_chunks.insert(name, family);
                }
            }
            Ok(BggSlotTransferGateWires { preimage_chunks })
        }

        /// Returns the exact number of trapdoor preimage samples performed by
        /// [`Self::build_slots`] followed by [`Self::build_gate_preimages`] for
        /// `requests`.
        ///
        /// This deliberately derives the count from the same chunking and
        /// family cardinalities as graph construction.  Callers therefore do
        /// not need to duplicate assumptions about the current gadget width or
        /// about how many destination rows a transfer/reduction request has.
        pub fn preprocessing_preimage_count(
            &self,
            requests: &[BggSlotTransferGateRequest],
        ) -> Result<usize, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let slot_samples = self.slot_count *
                (self.chunks(self.b1_public_columns()).len() +
                    self.chunks(self.gadget_columns()).len());
            let gate_chunks = self.chunks(self.gadget_columns()).len();
            let gate_samples = requests.iter().try_fold(0usize, |total, request| {
                self.validate_gate_request(request)?;
                let family_count = match request {
                    BggSlotTransferGateRequest::Transfer { source_slots, .. } => source_slots.len(),
                    BggSlotTransferGateRequest::Reduce { input_public_keys, .. } => {
                        input_public_keys.len()
                    }
                };
                Ok::<usize, BggSlotTransferArtifactError>(total + family_count * gate_chunks)
            })?;
            Ok(slot_samples + gate_samples)
        }

        pub fn export_gate_preimages(
            &self,
            context: DslContext,
            gates: BggSlotTransferGateWires,
        ) -> Result<DslContext, BggSlotTransferArtifactError> {
            Ok(gates.preimage_chunks.into_iter().try_fold(context, |context, (name, family)| {
                context.public_preimage_family_output(name, family)
            })?)
        }

        /// Exports the exact public-key expressions used as gate-preimage targets.
        pub fn export_gate_public_keys(
            &self,
            context: DslContext,
            requests: &[BggSlotTransferGateRequest],
        ) -> Result<DslContext, BggSlotTransferArtifactError> {
            requests.iter().try_fold(context, |context, request| {
                let (reduction, identity, output) = match request {
                    BggSlotTransferGateRequest::Transfer {
                        identity, output_public_key, ..
                    } => (false, identity, output_public_key),
                    BggSlotTransferGateRequest::Reduce { identity, output_public_key, .. } => {
                        (true, identity, output_public_key)
                    }
                };
                Ok(context.public_output(
                    super::slot_gate_public_key_name(reduction, identity),
                    output.clone(),
                )?)
            })
        }

        pub fn import_gate_preimages(
            &self,
            artifacts: &BggSlotTransferGateArtifacts,
            requests: &[BggSlotTransferGateRequest],
        ) -> Result<BggSlotTransferGateWires, BggSlotTransferArtifactError> {
            self.validate_layout()?;
            let ring = self.ring();
            let mut preimage_chunks = BTreeMap::new();
            for request in requests {
                self.validate_gate_request(request)?;
                let (reduction, identity, count) = match request {
                    BggSlotTransferGateRequest::Transfer { identity, source_slots, .. } => {
                        (false, identity, source_slots.len())
                    }
                    BggSlotTransferGateRequest::Reduce { identity, input_public_keys, .. } => {
                        (true, identity, input_public_keys.len())
                    }
                };
                for (chunk, columns) in self.chunks(self.gadget_columns()).into_iter().enumerate() {
                    let name = gate_preimage_name(reduction, identity, chunk);
                    let family = ring.preimage_family_artifact_input(
                        artifacts.production_id.clone(),
                        name.clone(),
                        vec![IntExpr::constant(count)],
                        (self.b0_public_columns(), range_len(&columns)),
                        self.preimage_max_coefficient_bound.clone(),
                        ArtifactConfidentiality::Public,
                    );
                    preimage_chunks.insert(name, family);
                }
            }
            Ok(BggSlotTransferGateWires { preimage_chunks })
        }

        fn build_transfer_gate_chunk(
            &self,
            base: &BggSlotTransferBaseWires,
            slots: &BggSlotTransferSlotWires,
            input: &Mat,
            output: &Mat,
            source_slots: &[(u32, Option<u32>)],
            columns: IndexRange,
        ) -> Result<Family<Preimage>, BggSlotTransferArtifactError> {
            let ring = self.ring();
            if source_slots.is_empty() {
                let rows = self.b0_public_columns();
                let columns = range_len(&columns);
                let target_rows = self.secret_size;
                let b0 = base.b0.clone();
                return Ok(Parallel::range(0).map_values(move |_| {
                    b0.sample_preimage(ring.zero((target_rows, columns)), (rows, columns))
                })?);
            }
            let results = source_slots
                .iter()
                .enumerate()
                .map(|(destination, (source, scalar))| {
                    let source = usize::try_from(*source).expect("u32 fits usize");
                    let source_secret = slots.secrets.get_static(source);
                    let destination_secret = slots.secrets.get_static(destination);
                    let destination_public = slots.public_keys.get_static(destination);
                    let destination_chunk = destination_public.slice(None, Some(columns.clone()));
                    let rhs = source_secret *
                        input.clone().mul_small_rhs(
                            destination_chunk.decompose(self.gadget_base.clone(), self.digit_count),
                        ) *
                        ring.polynomial([IntExpr::constant(scalar.unwrap_or(1))]);
                    let lhs =
                        destination_secret * output.clone().slice(None, Some(columns.clone()));
                    let target = lhs - rhs +
                        ring.gaussian(
                            (self.secret_size, range_len(&columns)),
                            self.error_sigma.clone(),
                            self.error_max_coefficient_bound.clone(),
                        );
                    base.b0.sample_preimage(target, (self.b0_public_columns(), range_len(&columns)))
                })
                .collect::<Vec<_>>();
            Ok(Family::pack(results)?)
        }

        fn build_reduce_gate_chunk(
            &self,
            base: &BggSlotTransferBaseWires,
            slots: &BggSlotTransferSlotWires,
            inputs: &[Mat],
            output: &Mat,
            source_slot_count: usize,
            columns: IndexRange,
        ) -> Result<Family<Preimage>, BggSlotTransferArtifactError> {
            let ring = self.ring();
            let results = inputs
                .iter()
                .enumerate()
                .map(|(destination, input)| {
                    let destination_secret = slots.secrets.get_static(destination);
                    let destination_chunk = slots
                        .public_keys
                        .get_static(destination)
                        .slice(None, Some(columns.clone()));
                    let rhs = (0..source_slot_count)
                        .map(|source| {
                            slots.secrets.get_static(source) *
                                input.clone().mul_small_rhs(
                                    destination_chunk
                                        .clone()
                                        .decompose(self.gadget_base.clone(), self.digit_count),
                                ) *
                                ring.constant(
                                    (1, 1),
                                    ConstantMatrix::Rotation {
                                        exponent: IntExpr::constant(source),
                                    },
                                )
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .reduce(|sum, term| sum + term)
                        .expect("validated nonzero source slots");
                    let lhs =
                        destination_secret * output.clone().slice(None, Some(columns.clone()));
                    let target = lhs - rhs +
                        ring.gaussian(
                            (self.secret_size, range_len(&columns)),
                            self.error_sigma.clone(),
                            self.error_max_coefficient_bound.clone(),
                        );
                    base.b0.sample_preimage(target, (self.b0_public_columns(), range_len(&columns)))
                })
                .collect::<Vec<_>>();
            Ok(Family::pack(results)?)
        }

        fn validate_gate_request(
            &self,
            request: &BggSlotTransferGateRequest,
        ) -> Result<(), BggSlotTransferArtifactError> {
            let valid = match request {
                BggSlotTransferGateRequest::Transfer {
                    input_public_key,
                    output_public_key,
                    source_slots,
                    ..
                } => {
                    input_public_key.matrix_type() == &self.public_key_type() &&
                        output_public_key.matrix_type() == &self.public_key_type() &&
                        source_slots.len() <= self.slot_count &&
                        source_slots
                            .par_iter()
                            .all(|(source, _)| (*source as usize) < self.slot_count)
                }
                BggSlotTransferGateRequest::Reduce {
                    input_public_keys,
                    output_public_key,
                    source_slot_count,
                    ..
                } => {
                    !input_public_keys.is_empty() &&
                        input_public_keys.len() <= *source_slot_count &&
                        *source_slot_count <= self.slot_count &&
                        output_public_key.matrix_type() == &self.public_key_type() &&
                        input_public_keys
                            .par_iter()
                            .all(|input| input.matrix_type() == &self.public_key_type())
                }
            };
            if valid { Ok(()) } else { Err(BggSlotTransferArtifactError::InvalidGateRequest) }
        }

        fn import_slot_chunks(
            &self,
            production_id: &ProductionId,
            b0: bool,
            columns: usize,
            rows: usize,
        ) -> Vec<Family<Preimage>> {
            let ring = self.ring();
            self.chunks(columns)
                .into_iter()
                .enumerate()
                .map(|(chunk, range)| {
                    ring.preimage_family_artifact_input(
                        production_id.clone(),
                        if b0 { b0_preimage_name(chunk) } else { b1_preimage_name(chunk) },
                        vec![IntExpr::constant(self.slot_count)],
                        (rows, range_len(&range)),
                        self.preimage_max_coefficient_bound.clone(),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect()
        }

        pub(crate) fn ring(&self) -> Ring {
            Ring::new(self.modulus.clone(), self.ring_dimension.clone())
        }
        pub(crate) fn matrix_type(&self, rows: usize, columns: usize) -> MatrixType {
            self.ring().matrix_type((rows, columns))
        }
        pub(crate) fn gadget_columns(&self) -> usize {
            self.secret_size * self.digit_count
        }
        pub(crate) fn b0_public_columns(&self) -> usize {
            self.secret_size * (self.digit_count + 2)
        }
        pub(crate) fn b1_public_columns(&self) -> usize {
            self.secret_size * 2 * (self.digit_count + 2)
        }
        pub(crate) fn chunks(&self, columns: usize) -> Vec<IndexRange> {
            (0..columns)
                .step_by(self.chunk_columns)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|start| IndexRange {
                    start: IntExpr::constant(start),
                    end: IntExpr::constant((start + self.chunk_columns).min(columns)),
                })
                .collect()
        }
    }

    fn range_len(range: &IndexRange) -> usize {
        let (IntExpr::Const(start), IntExpr::Const(end)) = (&range.start, &range.end) else {
            unreachable!("slot-transfer chunk ranges are static")
        };
        usize::try_from(end - start).expect("nonnegative static chunk length")
    }

    fn b0_preimage_name(chunk: usize) -> String {
        format!("slot_transfer_slot_preimage_b0_chunk_{chunk}")
    }
    fn b1_preimage_name(chunk: usize) -> String {
        format!("slot_transfer_slot_preimage_b1_chunk_{chunk}")
    }
    pub(crate) fn gate_preimage_name(reduction: bool, identity: &str, chunk: usize) -> String {
        let operation = if reduction { "slot_reduce" } else { "slot_transfer" };
        format!("{operation}_gate_{identity}_preimage_chunk_{chunk}")
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_utils::{execute_graph, matrix_output, row, small_matrix_output};
        use mxx_ir_core::ParamEnv;
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::{
                Poly, PolyParams,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
            sampler::{DistType, PolyHashSampler, hash::DCRTPolyHashSampler},
        };
        use mxx_runtime::{
            RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };
        use num_bigint::BigInt;
        use std::collections::BTreeMap;

        fn compiler() -> BggSlotTransferArtifactCompiler {
            BggSlotTransferArtifactCompiler {
                modulus: 65_537.into(),
                ring_dimension: 8.into(),
                secret_size: 2,
                slot_count: 3,
                digit_count: 4,
                chunk_columns: 3,
                gadget_base: 4.into(),
                trapdoor_sigma: RealExpr::from_integer(5),
                error_sigma: RealExpr::from_integer(3),
                preimage_max_coefficient_bound: 32.into(),
                error_max_coefficient_bound: 19.into(),
            }
        }

        fn static_range(range: &IndexRange) -> (usize, usize) {
            let (IntExpr::Const(start), IntExpr::Const(end)) = (&range.start, &range.end) else {
                panic!("test slot-transfer chunks must be static")
            };
            (
                usize::try_from(start).expect("nonnegative start"),
                usize::try_from(end).expect("nonnegative end"),
            )
        }

        #[test]
        fn runtime_preprocessing_and_gate_preimages_satisfy_the_primitive_relations() {
            let parameters = DCRTPolyParams::new(2, 1, 5, 3);
            let compiler = BggSlotTransferArtifactCompiler {
                modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
                ring_dimension: IntExpr::constant(parameters.ring_dimension()),
                secret_size: 1,
                slot_count: 2,
                digit_count: parameters.modulus_digits(),
                chunk_columns: 2,
                gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
                error_sigma: RealExpr::from_integer(0),
                preimage_max_coefficient_bound: 1_000_000.into(),
                error_max_coefficient_bound: 0.into(),
            };
            let ring = compiler.ring();
            let base = compiler.build_base().expect("base");
            let slots = compiler
                .build_slots(ring.bytes_input("slot_hash_key", 32), &base)
                .expect("slot preprocessing");
            let input_key_value = row(&parameters, compiler.gadget_columns(), 1);
            let input_key_two_value = row(&parameters, compiler.gadget_columns(), 3);
            let transfer_output_value = row(&parameters, compiler.gadget_columns(), 5);
            let reduce_output_value = row(&parameters, compiler.gadget_columns(), 7);
            let input_key = ring.input("input_key", (1, compiler.gadget_columns()));
            let input_key_two = ring.input("input_key_two", (1, compiler.gadget_columns()));
            let transfer_output = ring.input("transfer_output", (1, compiler.gadget_columns()));
            let reduce_output = ring.input("reduce_output", (1, compiler.gadget_columns()));
            let requests = [
                BggSlotTransferGateRequest::Transfer {
                    identity: "transfer".to_owned(),
                    input_public_key: input_key,
                    output_public_key: transfer_output,
                    source_slots: vec![(1, None), (0, Some(3))],
                },
                BggSlotTransferGateRequest::Reduce {
                    identity: "reduce".to_owned(),
                    input_public_keys: vec![input_key_two.clone(), input_key_two],
                    output_public_key: reduce_output,
                    source_slot_count: 2,
                },
            ];
            let gates =
                compiler.build_gate_preimages(&base, &slots, &requests).expect("gate preimages");

            let mut context = DslContext::new("slot-transfer-runtime-relations")
                .output("b0", base.b0.public_matrix())
                .expect("b0")
                .output("b1", base.b1.public_matrix())
                .expect("b1");
            for slot in 0..compiler.slot_count {
                context = context
                    .output(format!("secret_{slot}"), slots.secrets.get_static(slot))
                    .expect("secret")
                    .output(format!("public_{slot}"), slots.public_keys.get_static(slot))
                    .expect("public key");
                for (chunk, family) in slots.b0_preimage_chunks.iter().enumerate() {
                    context = context
                        .preimage_output(format!("slot_b0_{chunk}_{slot}"), family.get_static(slot))
                        .expect("b0 preimage");
                }
                for (chunk, family) in slots.b1_preimage_chunks.iter().enumerate() {
                    context = context
                        .preimage_output(format!("slot_b1_{chunk}_{slot}"), family.get_static(slot))
                        .expect("b1 preimage");
                }
            }
            for (reduction, identity) in [(false, "transfer"), (true, "reduce")] {
                for chunk in 0..compiler.chunks(compiler.gadget_columns()).len() {
                    let family =
                        &gates.preimage_chunks[&gate_preimage_name(reduction, identity, chunk)];
                    for destination in 0..2 {
                        context = context
                            .preimage_output(
                                format!("gate_{identity}_{chunk}_{destination}"),
                                family.get_static(destination),
                            )
                            .expect("gate preimage");
                    }
                }
            }
            let result = execute_graph(
                context.build().expect("runtime graph"),
                parameters.clone(),
                BTreeMap::from([
                    ("slot_hash_key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32])),
                    ("input_key".to_owned(), RuntimeValue::matrix(input_key_value.clone())),
                    ("input_key_two".to_owned(), RuntimeValue::matrix(input_key_two_value.clone())),
                    (
                        "transfer_output".to_owned(),
                        RuntimeValue::matrix(transfer_output_value.clone()),
                    ),
                    ("reduce_output".to_owned(), RuntimeValue::matrix(reduce_output_value.clone())),
                ]),
            );

            let b0 = matrix_output(&result, "b0");
            let b1 = matrix_output(&result, "b1");
            let identity = DCRTPolyMatrix::identity(&parameters, compiler.secret_size, None);
            let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
            for slot in 0..compiler.slot_count {
                let secret = matrix_output(&result, &format!("secret_{slot}"));
                let public = matrix_output(&result, &format!("public_{slot}"));
                let secret_identity = secret.clone().concat_columns(&[&identity]);
                for (chunk, range) in
                    compiler.chunks(compiler.b1_public_columns()).into_iter().enumerate()
                {
                    let (start, end) = static_range(&range);
                    assert_eq!(
                        b0.clone() *
                            small_matrix_output(&result, &format!("slot_b0_{chunk}_{slot}")),
                        secret_identity.clone() * &b1.slice_columns(start, end)
                    );
                }
                for (chunk, range) in
                    compiler.chunks(compiler.gadget_columns()).into_iter().enumerate()
                {
                    let (start, end) = static_range(&range);
                    let expected = public
                        .slice_columns(start, end)
                        .concat_rows(&[&-(secret.clone() * &gadget.slice_columns(start, end))]);
                    assert_eq!(
                        b1.clone() *
                            small_matrix_output(&result, &format!("slot_b1_{chunk}_{slot}")),
                        expected
                    );
                }
            }

            for (chunk, range) in compiler.chunks(compiler.gadget_columns()).into_iter().enumerate()
            {
                let (start, end) = static_range(&range);
                for destination in 0..2 {
                    let source = [1usize, 0][destination];
                    let scalar =
                        DCRTPoly::from_usize_to_constant(&parameters, [1usize, 3][destination]);
                    let source_secret = matrix_output(&result, &format!("secret_{source}"));
                    let destination_secret =
                        matrix_output(&result, &format!("secret_{destination}"));
                    let destination_public =
                        matrix_output(&result, &format!("public_{destination}"));
                    let rhs = ((source_secret.clone() * &input_key_value) *
                        &destination_public.slice_columns(start, end).decompose()) *
                        &scalar;
                    let expected = destination_secret.clone() *
                        &transfer_output_value.slice_columns(start, end) -
                        &rhs;
                    assert_eq!(
                        b0.clone() *
                            small_matrix_output(
                                &result,
                                &format!("gate_transfer_{chunk}_{destination}"),
                            ),
                        expected
                    );

                    let decomposed = destination_public.slice_columns(start, end).decompose();
                    let mut rhs =
                        DCRTPolyMatrix::zero(&parameters, compiler.secret_size, end - start);
                    for source in 0..2 {
                        let rotation = DCRTPoly::const_rotate_poly(&parameters, source);
                        let source_secret = matrix_output(&result, &format!("secret_{source}"));
                        rhs = rhs +
                            &(((source_secret.clone() * &input_key_two_value) * &decomposed) *
                                &rotation);
                    }
                    let expected = destination_secret.clone() *
                        &reduce_output_value.slice_columns(start, end) -
                        &rhs;
                    assert_eq!(
                        b0.clone() *
                            small_matrix_output(
                                &result,
                                &format!("gate_reduce_{chunk}_{destination}"),
                            ),
                        expected
                    );
                }
            }
        }

        #[test]
        #[ignore = "large CPU preimage artifact-family integration test"]
        fn runtime_artifact_productions_preserve_import_order_tail_chunks_and_gate_families() {
            let parameters = DCRTPolyParams::new(8, 1, 20, 4);
            let compiler = BggSlotTransferArtifactCompiler {
                modulus: IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
                ring_dimension: IntExpr::constant(parameters.ring_dimension()),
                secret_size: 2,
                slot_count: 11,
                digit_count: parameters.modulus_digits(),
                chunk_columns: 3,
                gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                trapdoor_sigma: RealExpr::from_f64_exact(4.578).expect("finite sigma"),
                error_sigma: RealExpr::from_integer(0),
                preimage_max_coefficient_bound: 1_000_000.into(),
                error_max_coefficient_bound: 0.into(),
            };
            let mut backend = cpu_backend([parameters.clone()]);
            let mut store = MemoryArtifactStore::default();

            let base = compiler.build_base().expect("base");
            let base_graph = compiler
                .export_base(DslContext::new("slot-base-production"), base)
                .unwrap()
                .build()
                .unwrap()
                .validate(&ParamEnv::default())
                .unwrap();
            let base_result = execute(
                &base_graph,
                &mut backend,
                BTreeMap::new(),
                &mut store,
                SamplingMode::Fresh,
            )
            .unwrap();
            let base_production = base_result.production_id.expect("base production");
            let base_manifest = store.manifest(&base_production).unwrap().clone();

            let imported_base = compiler
                .import_base(&BggSlotTransferBaseArtifacts {
                    production_id: base_production.clone(),
                })
                .unwrap();
            let slots = compiler
                .build_slots(compiler.ring().bytes_input("slot-hash-key", 32), &imported_base)
                .unwrap();
            let slot_graph = compiler
                .export_slots(DslContext::new("slot-production"), slots)
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([(base_production.clone(), base_manifest.clone())]),
                )
                .unwrap();
            let slot_result = execute(
                &slot_graph,
                &mut backend,
                BTreeMap::from([("slot-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
                &mut store,
                SamplingMode::Fresh,
            )
            .unwrap();
            let slot_production = slot_result.production_id.expect("slot production");
            let slot_manifest = store.manifest(&slot_production).unwrap().clone();
            let b0_ranges = compiler.chunks(compiler.b1_public_columns());
            assert!(b0_ranges.len() >= 10, "test must cross the chunk-9 name boundary");
            assert!(
                b0_ranges.iter().any(|range| range_len(range) < compiler.chunk_columns),
                "test must contain a nonmultiple tail chunk"
            );

            let imported_base = compiler
                .import_base(&BggSlotTransferBaseArtifacts {
                    production_id: base_production.clone(),
                })
                .unwrap();
            let imported_slots = compiler
                .import_slots(&BggSlotTransferSlotArtifacts {
                    production_id: slot_production.clone(),
                })
                .unwrap();
            let inspected_slot = 10;
            let mut inspect = DslContext::new("slot-import-inspection")
                .output("b0", imported_base.b0.public_matrix())
                .unwrap()
                .output("b1", imported_base.b1.public_matrix())
                .unwrap()
                .output("secret", imported_slots.secrets.get_static(inspected_slot))
                .unwrap()
                .output("public", imported_slots.public_keys.get_static(inspected_slot))
                .unwrap();
            for (chunk, family) in imported_slots.b0_preimage_chunks.iter().enumerate() {
                inspect = inspect
                    .preimage_output(
                        format!("b0-preimage-{chunk}"),
                        family.get_static(inspected_slot),
                    )
                    .unwrap();
            }
            for (chunk, family) in imported_slots.b1_preimage_chunks.iter().enumerate() {
                inspect = inspect
                    .preimage_output(
                        format!("b1-preimage-{chunk}"),
                        family.get_static(inspected_slot),
                    )
                    .unwrap();
            }
            let inspect = inspect
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([
                        (base_production.clone(), base_manifest.clone()),
                        (slot_production.clone(), slot_manifest.clone()),
                    ]),
                )
                .unwrap();
            let inspected =
                execute(&inspect, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
                    .unwrap();
            let b0 = matrix_output(&inspected, "b0");
            let b1 = matrix_output(&inspected, "b1");
            let secret = matrix_output(&inspected, "secret");
            let public = matrix_output(&inspected, "public");
            let expected_public = DCRTPolyHashSampler::<keccak_asm::Keccak256>::new().sample_hash(
                &parameters,
                [0x42; 32],
                format!("slot_transfer_slot_a_{inspected_slot}"),
                compiler.secret_size,
                compiler.gadget_columns(),
                DistType::FinRingDist,
            );
            assert_eq!(public, &expected_public);
            let identity = DCRTPolyMatrix::identity(&parameters, compiler.secret_size, None);
            let secret_identity = secret.clone().concat_columns(&[&identity]);
            for (chunk, range) in b0_ranges.iter().enumerate() {
                let (start, end) = static_range(range);
                assert_eq!(
                    b0.clone() * small_matrix_output(&inspected, &format!("b0-preimage-{chunk}")),
                    secret_identity.clone() * &b1.slice_columns(start, end)
                );
            }
            let gadget = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
            for (chunk, range) in compiler.chunks(compiler.gadget_columns()).iter().enumerate() {
                let (start, end) = static_range(range);
                let expected = public
                    .slice_columns(start, end)
                    .concat_rows(&[&-(secret.clone() * &gadget.slice_columns(start, end))]);
                assert_eq!(
                    b1.clone() * small_matrix_output(&inspected, &format!("b1-preimage-{chunk}")),
                    expected
                );
            }

            let ring = compiler.ring();
            let base = compiler
                .import_base(&BggSlotTransferBaseArtifacts {
                    production_id: base_production.clone(),
                })
                .unwrap();
            let slots = compiler
                .import_slots(&BggSlotTransferSlotArtifacts {
                    production_id: slot_production.clone(),
                })
                .unwrap();
            let gate_hash = ring.bytes_input("gate-hash-key", 32);
            let input_key = ring.gadget(
                compiler.secret_size,
                compiler.gadget_base.clone(),
                compiler.digit_count,
            );
            let transfer_output = ring.hash_matrix(
                gate_hash.clone(),
                b"slot_transfer_gate_a_out_7".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let reduce_output = ring.hash_matrix(
                gate_hash,
                b"slot_reduce_gate_a_out_8".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let requests = vec![
                BggSlotTransferGateRequest::Transfer {
                    identity: "7".to_owned(),
                    input_public_key: input_key.clone(),
                    output_public_key: transfer_output.clone(),
                    source_slots: vec![(1, None), (0, Some(3))],
                },
                BggSlotTransferGateRequest::Reduce {
                    identity: "8".to_owned(),
                    input_public_keys: vec![input_key.clone(), input_key.clone()],
                    output_public_key: reduce_output.clone(),
                    source_slot_count: 2,
                },
                BggSlotTransferGateRequest::Transfer {
                    identity: "9".to_owned(),
                    input_public_key: input_key.clone(),
                    output_public_key: transfer_output.clone(),
                    source_slots: Vec::new(),
                },
            ];
            let gates = compiler.build_gate_preimages(&base, &slots, &requests).unwrap();
            let gate_graph = compiler
                .export_gate_preimages(DslContext::new("slot-gate-production"), gates)
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([
                        (base_production.clone(), base_manifest.clone()),
                        (slot_production.clone(), slot_manifest.clone()),
                    ]),
                )
                .unwrap();
            let gate_result = execute(
                &gate_graph,
                &mut backend,
                BTreeMap::from([("gate-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
                &mut store,
                SamplingMode::Fresh,
            )
            .unwrap();
            let gate_production = gate_result.production_id.expect("gate production");
            let gate_manifest = store.manifest(&gate_production).unwrap().clone();
            let invalid = BggSlotTransferGateRequest::Transfer {
                identity: "7".to_owned(),
                input_public_key: ring.identity(1),
                output_public_key: transfer_output.clone(),
                source_slots: vec![(0, None)],
            };
            assert!(matches!(
                compiler.import_gate_preimages(
                    &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                    &[invalid],
                ),
                Err(BggSlotTransferArtifactError::InvalidGateRequest)
            ));

            let imported_base = compiler
                .import_base(&BggSlotTransferBaseArtifacts {
                    production_id: base_production.clone(),
                })
                .unwrap();
            let imported_slots = compiler
                .import_slots(&BggSlotTransferSlotArtifacts {
                    production_id: slot_production.clone(),
                })
                .unwrap();
            let imported_gates = compiler
                .import_gate_preimages(
                    &BggSlotTransferGateArtifacts { production_id: gate_production.clone() },
                    &requests,
                )
                .unwrap();
            let transfer_name = gate_preimage_name(false, "7", 0);
            let reduce_name = gate_preimage_name(true, "8", 0);
            let gate_hash = ring.bytes_input("gate-hash-key", 32);
            let transfer_output = ring.hash_matrix(
                gate_hash.clone(),
                b"slot_transfer_gate_a_out_7".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let reduce_output = ring.hash_matrix(
                gate_hash,
                b"slot_reduce_gate_a_out_8".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let consumer = DslContext::new("slot-gate-import-consumer")
                .output("b0", imported_base.b0.public_matrix())
                .unwrap()
                .output("secret-0", imported_slots.secrets.get_static(0))
                .unwrap()
                .output("secret-1", imported_slots.secrets.get_static(1))
                .unwrap()
                .output("public-0", imported_slots.public_keys.get_static(0))
                .unwrap()
                .output("transfer-output", transfer_output)
                .unwrap()
                .output("reduce-output", reduce_output)
                .unwrap()
                .preimage_output(
                    "transfer-preimage",
                    imported_gates.preimage_chunks[&transfer_name].get_static(0),
                )
                .unwrap()
                .preimage_output(
                    "reduce-preimage",
                    imported_gates.preimage_chunks[&reduce_name].get_static(0),
                )
                .unwrap()
                .build()
                .unwrap()
                .validate_with_manifests(
                    &ParamEnv::default(),
                    &BTreeMap::from([
                        (base_production, base_manifest),
                        (slot_production, slot_manifest),
                        (gate_production, gate_manifest),
                    ]),
                )
                .unwrap();
            let consumed = execute(
                &consumer,
                &mut backend,
                BTreeMap::from([("gate-hash-key".to_owned(), RuntimeValue::Bytes(vec![0x42; 32]))]),
                &mut store,
                SamplingMode::Fresh,
            )
            .unwrap();
            let (start, end) = static_range(&compiler.chunks(compiler.gadget_columns())[0]);
            let b0 = matrix_output(&consumed, "b0");
            let secret_0 = matrix_output(&consumed, "secret-0");
            let secret_1 = matrix_output(&consumed, "secret-1");
            let public_0 = matrix_output(&consumed, "public-0");
            let input_key = DCRTPolyMatrix::gadget_matrix(&parameters, compiler.secret_size);
            let decomposed = public_0.slice_columns(start, end).decompose();
            let transfer_rhs = (secret_1.clone() * &input_key) * &decomposed;
            let transfer_expected = secret_0.clone() *
                &matrix_output(&consumed, "transfer-output").slice_columns(start, end) -
                &transfer_rhs;
            assert_eq!(
                b0.clone() * small_matrix_output(&consumed, "transfer-preimage"),
                transfer_expected
            );
            let mut reduce_rhs =
                DCRTPolyMatrix::zero(&parameters, compiler.secret_size, end - start);
            for (source, secret) in [secret_0, secret_1].into_iter().enumerate() {
                reduce_rhs = reduce_rhs +
                    &(((secret.clone() * &input_key) * &decomposed) *
                        &DCRTPoly::const_rotate_poly(&parameters, source));
            }
            let reduce_expected = matrix_output(&consumed, "secret-0").clone() *
                &matrix_output(&consumed, "reduce-output").slice_columns(start, end) -
                &reduce_rhs;
            assert_eq!(
                b0.clone() * small_matrix_output(&consumed, "reduce-preimage"),
                reduce_expected
            );
        }

        #[test]
        fn base_and_slot_preprocessing_build_valid_graphs() {
            let compiler = compiler();
            let base = compiler.build_base().expect("base");
            compiler
                .export_base(DslContext::new("slot-base"), base.clone())
                .expect("base outputs")
                .build()
                .expect("base graph")
                .validate(&ParamEnv::default())
                .expect("valid base graph");
            let slots = compiler
                .build_slots(compiler.ring().bytes_input("hash-key", 32), &base)
                .expect("slots");
            let slot_graph = compiler
                .export_slots(DslContext::new("slot-preprocessing"), slots.clone())
                .expect("slot outputs")
                .build()
                .expect("slot graph");
            slot_graph.validate(&ParamEnv::default()).expect("valid slot graph");

            let key = compiler.ring().bytes_input("gate-hash-key", 32);
            let input = compiler.ring().hash_matrix(
                key.clone(),
                b"input".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let output = compiler.ring().hash_matrix(
                key,
                b"output".as_slice(),
                (compiler.secret_size, compiler.gadget_columns()),
            );
            let gates = compiler
                .build_gate_preimages(
                    &base,
                    &slots,
                    &[
                        BggSlotTransferGateRequest::Transfer {
                            identity: "test".to_owned(),
                            input_public_key: input.clone(),
                            output_public_key: output.clone(),
                            source_slots: vec![(1, None), (0, Some(2))],
                        },
                        BggSlotTransferGateRequest::Transfer {
                            identity: "empty".to_owned(),
                            input_public_key: input,
                            output_public_key: output,
                            source_slots: Vec::new(),
                        },
                    ],
                )
                .expect("gate preimages");
            let gate_graph = compiler
                .export_gate_preimages(DslContext::new("slot-gates"), gates)
                .expect("gate outputs")
                .build()
                .expect("gate graph");
            gate_graph.validate(&ParamEnv::default()).expect("valid gate graph");
        }
    }
}
pub use artifact::*;

mod tall {
    use crate::{
        BggPublicKeyWire, CircuitCompileError,
        tall_encoding::{BggTallEncodingCompiler, BggTallEncodingSampler, BggTallEncodingWire},
        tall_rotation_encoding::{
            TallLinearTransformEncodingWires, TallLinearTransformPublicWires,
            TallRotationEncodingKey,
        },
    };
    use mxx_dsl::{Family, Int, Mat, Parallel};
    use mxx_gadgets::{
        Poly,
        circuit::{CircuitLoweringTypes, GateInstance, SlotOperationLowering},
    };
    use mxx_ir_core::IntExpr;
    use std::collections::BTreeMap;

    /// Public-key slot lowering for the secret-transfer-free Tall subset.
    ///
    /// The only ordinary transfer it accepts is an identity-source diagonal
    /// mask.  Its public key is the fixed mask public matrix multiplied by the
    /// input public key; the matching mask encoding is built online below from
    /// the one supplied Tall secret family.
    #[derive(Clone)]
    pub struct BggTallSlotPublicKeyLowering {
        /// BGG+ public-key arithmetic.
        pub compiler: crate::BggPublicKeyCompiler,
        /// Public key used for every per-row diagonal mask.
        pub diagonal_mask_public_key: BggPublicKeyWire,
        /// Exact physical Tall slot count.
        pub configured_slot_count: usize,
        /// Exact preprocessed public matrices for cyclic rotations.
        pub rotations: BTreeMap<TallRotationEncodingKey, TallLinearTransformPublicWires>,
        /// The sole CRT reconstruction spec and its generic fixed linear transform.
        pub anchor_reduce:
            Option<((u32, Vec<num_bigint::BigUint>), TallLinearTransformPublicWires)>,
    }

    impl CircuitLoweringTypes for BggTallSlotPublicKeyLowering {
        type Wire = BggPublicKeyWire;
        type Error = CircuitCompileError;
    }

    impl<P: Poly> SlotOperationLowering<P> for BggTallSlotPublicKeyLowering {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            validate_identity_sources(source_slots, self.configured_slot_count, gate)?;
            if input.matrix.matrix_type() != self.diagonal_mask_public_key.matrix.matrix_type() {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            Ok(self.compiler.mul(&self.diagonal_mask_public_key, input))
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            let _ = (inputs, slot_count);
            Err(CircuitCompileError::Unsupported {
                gate: gate.local_gate().index(),
                feature: "nonidentity Tall slot reduction",
            })
        }

        fn slot_anchor_reduce(
            &mut self,
            input: &Self::Wire,
            num_blocks: u32,
            lane_scalars: &[num_bigint::BigUint],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            let (_, transform) = self
                .anchor_reduce
                .as_ref()
                .filter(|((blocks, scalars), _)| {
                    *blocks == num_blocks && scalars.as_slice() == lane_scalars
                })
                .ok_or(CircuitCompileError::Unsupported {
                    gate: gate.local_gate().index(),
                    feature: "missing Tall anchor-reduction encoding",
                })?;
            let tall = BggTallEncodingCompiler { public_key: self.compiler.clone() };
            let helper_public = tall.linear_transform_public_key(
                input,
                &transform.left_matrix,
                &transform.right_matrix,
            );
            let scalar = self.compiler.ring.polynomial([mxx_ir_core::IntExpr::constant(
                num_bigint::BigInt::from(lane_scalars[0].clone()),
            )]);
            Ok(self.compiler.add(&self.compiler.large_scalar_mul(input, &scalar), &helper_public))
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            offset: u32,
            num_slots: u32,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            let Some(key) =
                TallRotationEncodingKey::normalize(num_slots, offset).map_err(|_| {
                    CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
                })?
            else {
                return Ok(input.clone());
            };
            if usize::try_from(num_slots)
                .ok()
                .is_none_or(|slots| slots != self.configured_slot_count) ||
                input.matrix.matrix_type() != self.diagonal_mask_public_key.matrix.matrix_type()
            {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            let transform = self.rotations.get(&key).ok_or(
                CircuitCompileError::MissingTallRotationEncoding {
                    num_slots: key.num_slots,
                    offset: key.offset,
                },
            )?;
            Ok(BggTallEncodingCompiler { public_key: self.compiler.clone() }
                .linear_transform_public_key(
                    input,
                    &transform.left_matrix,
                    &transform.right_matrix,
                ))
        }

        fn slot_identity_repeated_lanes(
            &mut self,
            input: &Self::Wire,
            num_blocks: u32,
            lane_scalars: &[Option<u32>],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            validate_identity_repeated_lanes(
                num_blocks,
                lane_scalars,
                self.configured_slot_count,
                gate,
            )?;
            if input.matrix.matrix_type() != self.diagonal_mask_public_key.matrix.matrix_type() {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            Ok(self.compiler.mul(&self.diagonal_mask_public_key, input))
        }
    }

    /// Encoding-side lowering for the secret-transfer-free Tall subset.
    ///
    /// Ordinary transfers are limited to identity-source per-row masks.  The
    /// fixed public mask key is encoded afresh under `secret_rows`; no
    /// trapdoor, transfer matrix, per-slot artifact, or gate preimage exists
    /// on this path.  General transfer/broadcast/reduction remains explicitly
    /// unsupported until it has a separately reviewed secret-free construction.
    #[derive(Clone)]
    pub struct BggTallSlotLowering {
        /// Tall arithmetic compiler.
        compiler: BggTallEncodingCompiler,
        /// Public key used for every per-row diagonal mask.
        diagonal_mask_public_key: BggPublicKeyWire,
        /// The one fresh Tall secret-row family owned by the online graph.
        secret_rows: Family<Mat>,
        /// Error configuration for direct diagonal-mask encodings.
        sampler: BggTallEncodingSampler,
        /// Direct tall-rotation encodings keyed by `(num_slots, normalized_offset)`.
        rotations: BTreeMap<TallRotationEncodingKey, TallLinearTransformEncodingWires>,
        anchor_reduce: Option<((u32, Vec<num_bigint::BigUint>), TallLinearTransformEncodingWires)>,
    }

    impl BggTallSlotLowering {
        pub fn new(
            compiler: BggTallEncodingCompiler,
            diagonal_mask_public_key: BggPublicKeyWire,
            secret_rows: Family<Mat>,
            sampler: BggTallEncodingSampler,
            rotations: BTreeMap<TallRotationEncodingKey, TallLinearTransformEncodingWires>,
            anchor_reduce: Option<(
                (u32, Vec<num_bigint::BigUint>),
                TallLinearTransformEncodingWires,
            )>,
        ) -> Self {
            Self {
                compiler,
                diagonal_mask_public_key,
                secret_rows,
                sampler,
                rotations,
                anchor_reduce,
            }
        }

        fn transfer(
            &mut self,
            input: &BggTallEncodingWire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<BggTallEncodingWire, CircuitCompileError> {
            validate_identity_sources(source_slots, self.configured_slot_count(), gate)?;
            let ring = self.sampler.layout.ring();
            let masks = Family::pack(
                source_slots
                    .iter()
                    .map(|(_, scalar)| ring.polynomial([IntExpr::constant(scalar.unwrap_or(1))]))
                    .collect(),
            )?;
            let mask = self.sampler.sample_diagonal(
                self.secret_rows.clone(),
                self.diagonal_mask_public_key.clone(),
                masks,
            )?;
            Ok(self.compiler.simd_mul(&mask, input)?)
        }

        fn configured_slot_count(&self) -> usize {
            match self.secret_rows.count() {
                IntExpr::Const(count) => {
                    usize::try_from(count).expect("Tall slot count must fit usize")
                }
                _ => unreachable!("Tall slot lowering requires a concrete secret-row family"),
            }
        }

        fn transfer_identity_repeated_lanes(
            &mut self,
            input: &BggTallEncodingWire,
            num_blocks: u32,
            lane_scalars: &[Option<u32>],
            gate: GateInstance<'_>,
        ) -> Result<BggTallEncodingWire, CircuitCompileError> {
            let total_slots = validate_identity_repeated_lanes(
                num_blocks,
                lane_scalars,
                self.configured_slot_count(),
                gate,
            )?;
            let masks = identity_repeated_lane_masks(
                &self.sampler.layout.ring(),
                total_slots,
                lane_scalars,
            )?;
            let mask = self.sampler.sample_diagonal(
                self.secret_rows.clone(),
                self.diagonal_mask_public_key.clone(),
                masks,
            )?;
            Ok(self.compiler.simd_mul(&mask, input)?)
        }
    }

    impl CircuitLoweringTypes for BggTallSlotLowering {
        type Wire = BggTallEncodingWire;
        type Error = CircuitCompileError;
    }

    impl<P: Poly> SlotOperationLowering<P> for BggTallSlotLowering {
        fn slot_transfer(
            &mut self,
            input: &Self::Wire,
            source_slots: &[(u32, Option<u32>)],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.transfer(input, source_slots, gate)
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Self::Wire],
            slot_count: usize,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            let _ = (inputs, slot_count);
            Err(CircuitCompileError::Unsupported {
                gate: gate.local_gate().index(),
                feature: "nonidentity Tall slot reduction",
            })
        }

        fn slot_anchor_reduce(
            &mut self,
            input: &Self::Wire,
            num_blocks: u32,
            lane_scalars: &[num_bigint::BigUint],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            let ((blocks, scalars), transform) = self
                .anchor_reduce
                .as_ref()
                .filter(|((blocks, scalars), _)| {
                    *blocks == num_blocks && scalars.as_slice() == lane_scalars
                })
                .ok_or(CircuitCompileError::Unsupported {
                    gate: gate.local_gate().index(),
                    feature: "missing Tall anchor-reduction encoding",
                })?;
            self.compiler.anchor_reduce(input, *blocks, scalars, transform).map_err(Into::into)
        }

        fn slot_rotation(
            &mut self,
            input: &Self::Wire,
            offset: u32,
            num_slots: u32,
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            if input.rows.count() !=
                &IntExpr::constant(usize::try_from(num_slots).map_err(|_| {
                    CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
                })?)
            {
                return Err(CircuitCompileError::InvalidSlotTransfer {
                    gate: gate.local_gate().index(),
                });
            }
            let Some(key) =
                TallRotationEncodingKey::normalize(num_slots, offset).map_err(|_| {
                    CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() }
                })?
            else {
                return Ok(input.clone());
            };
            let transform = self.rotations.get(&key).ok_or(
                CircuitCompileError::MissingTallRotationEncoding {
                    num_slots: key.num_slots,
                    offset: key.offset,
                },
            )?;
            self.compiler.rotate(input, key, transform).map_err(Into::into)
        }

        fn slot_identity_repeated_lanes(
            &mut self,
            input: &Self::Wire,
            num_blocks: u32,
            lane_scalars: &[Option<u32>],
            gate: GateInstance<'_>,
        ) -> Result<Self::Wire, Self::Error> {
            self.transfer_identity_repeated_lanes(input, num_blocks, lane_scalars, gate)
        }
    }

    fn validate_identity_sources(
        source_slots: &[(u32, Option<u32>)],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<(), CircuitCompileError> {
        if source_slots.len() != slot_count {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        if source_slots
            .iter()
            .enumerate()
            .any(|(destination, (source, _))| *source != destination as u32)
        {
            return Err(CircuitCompileError::Unsupported {
                gate: gate.local_gate().index(),
                feature: "nonidentity Tall slot transfer",
            });
        }
        Ok(())
    }

    fn validate_identity_repeated_lanes(
        num_blocks: u32,
        lane_scalars: &[Option<u32>],
        slot_count: usize,
        gate: GateInstance<'_>,
    ) -> Result<usize, CircuitCompileError> {
        let total_slots = usize::try_from(num_blocks)
            .ok()
            .and_then(|blocks| blocks.checked_mul(lane_scalars.len()))
            .filter(|_| num_blocks > 0 && !lane_scalars.is_empty())
            .ok_or(CircuitCompileError::InvalidSlotTransfer { gate: gate.local_gate().index() })?;
        if total_slots != slot_count {
            return Err(CircuitCompileError::InvalidSlotTransfer {
                gate: gate.local_gate().index(),
            });
        }
        Ok(total_slots)
    }

    fn identity_repeated_lane_masks(
        ring: &mxx_dsl::Ring,
        total_slots: usize,
        lane_scalars: &[Option<u32>],
    ) -> Result<Family<Mat>, mxx_dsl::DslError> {
        let lanes = lane_scalars.len();
        Parallel::range(total_slots).try_map(|index| {
            index.as_int().rem(Int::constant(lanes)).select(
                lane_scalars
                    .iter()
                    .map(|scalar| ring.polynomial([IntExpr::constant(scalar.unwrap_or(1))]))
                    .collect(),
            )
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_utils::{execute_graph, matrix_output};
        use mxx_dsl::{DslContext, Ring};
        use mxx_primitives::{
            matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
            poly::{
                PolyParams,
                dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
            },
        };
        use std::collections::BTreeMap;

        #[test]
        fn compact_identity_lane_masks_match_explicit_runtime_sequence() {
            let parameters = DCRTPolyParams::new(8, 1, 20, 4);
            let ring = Ring::new(
                num_bigint::BigInt::from(parameters.modulus().as_ref().clone()),
                parameters.ring_dimension() as usize,
            );
            let lane_scalars = [Some(0), None, Some(5)];
            let masks = identity_repeated_lane_masks(&ring, 6, &lane_scalars)
                .expect("compact identity lane masks");
            let mut context = DslContext::new("compact-identity-lane-mask-runtime");
            for slot in 0..6 {
                context = context
                    .output(format!("mask-{slot}"), masks.get_static(slot))
                    .expect("mask output");
            }
            let result = execute_graph(
                context.build().expect("compact mask graph"),
                parameters.clone(),
                BTreeMap::new(),
            );
            for slot in 0..6 {
                let scalar = lane_scalars[slot % lane_scalars.len()].unwrap_or(1);
                let expected = DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, scalar as usize)],
                );
                assert_eq!(matrix_output(&result, &format!("mask-{slot}")), &expected);
            }
        }
    }
}
pub use tall::*;
