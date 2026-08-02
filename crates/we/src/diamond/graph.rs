use super::{DiamondArtifactNames, DiamondConfigError, DiamondWeConfig};
use mxx_bgg::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeySampler,
    BggPublicKeyWire, BggSamplerLayout, CircuitCompileError, PolyCircuitCompiler,
};
use mxx_dsl::{Bool, BuiltGraph, DslContext, DslError, Family, Int, Mat};
use mxx_gadgets::{
    Poly,
    circuit::{PolyCircuit, PublicLookupLowering, SlotOperationLowering},
    input_injector::{DiamondInputInjector, DiamondInputPreprocessError},
};
use mxx_ir_core::{
    artifact::{ArtifactConfidentiality, ProductionId},
    node::{ConcatAxis, IndexRange},
};
use num_bigint::BigInt;
use thiserror::Error;

pub const HASH_KEY_INPUT: &str = "diamond-hash-key";
pub const MESSAGE_INPUT: &str = "diamond-message";
pub const DECODED_OUTPUT: &str = "diamond-decoded";
pub const NOISY_PLAINTEXT_OUTPUT: &str = "diamond-noisy-plaintext";

pub struct DiamondEncryptionGraph {
    pub graph: BuiltGraph,
}

pub struct DiamondDecryptionGraph {
    pub graph: BuiltGraph,
}

#[derive(Clone)]
pub struct DiamondWeCompiler {
    pub config: DiamondWeConfig,
}

#[derive(Debug, Error)]
pub enum DiamondCompileError {
    #[error(transparent)]
    Config(#[from] DiamondConfigError),
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error(transparent)]
    Circuit(#[from] CircuitCompileError),
    #[error(transparent)]
    Input(#[from] DiamondInputPreprocessError),
    #[error("Diamond WE requires exactly one circuit output")]
    OutputCount,
    #[error("the circuit input count does not equal witness_size + instance_size")]
    InputCount,
}

impl DiamondWeCompiler {
    pub fn new(config: DiamondWeConfig) -> Result<Self, DiamondConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn build_encryption<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        instance: &[bool],
    ) -> Result<DiamondEncryptionGraph, DiamondCompileError> {
        let mut lookup = mxx_bgg::NoPublicLookup::default();
        let mut slots = mxx_bgg::NoSlotOperations::default();
        self.build_encryption_with_lowerings(circuit, instance, &mut lookup, &mut slots)
    }

    pub fn build_encryption_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        instance: &[bool],
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<DiamondEncryptionGraph, DiamondCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggPublicKeyWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggPublicKeyWire, Error = CircuitCompileError>,
    {
        self.validate_circuit(circuit, instance.len())?;
        let ring = self.config.ring();
        let message = ring.input(MESSAGE_INPUT, (1, 1));
        let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
        let input_preprocessing =
            DiamondInputInjector::new(self.config.input_config())?.preprocess(message)?;
        let public_key_compiler = self.public_key_compiler();
        let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
        let witness_size = self.config.witness_size()?;
        let public_keys = BggPublicKeySampler { layout: self.sampler_layout() }.sample(
            hash_key.clone(),
            self.tag(b":witness_public_keys"),
            &vec![true; witness_size],
        );
        let one_public_key = public_keys[0].clone();
        let zero_public_key =
            public_key_compiler.small_scalar_mul(&one_public_key, &ring.zero((1, 1)));
        let mut circuit_inputs = public_keys[1..].to_vec();
        circuit_inputs.extend(
            instance
                .iter()
                .map(|bit| if *bit { one_public_key.clone() } else { zero_public_key.clone() }),
        );
        let circuit_outputs = circuit_compiler.compile_public_keys_with_lowerings(
            circuit,
            one_public_key.clone(),
            circuit_inputs,
            lookup,
            slots,
        )?;
        let circuit_output =
            circuit_outputs.into_iter().next().ok_or(DiamondCompileError::OutputCount)?;

        let gadget = ring.gadget(1, self.config.gadget_base_expr(), self.config.digit_count_expr());
        let public_columns = self.config.public_key_columns()?;
        let state_columns = self.config.state_columns()?;
        let zero_row = ring.zero((1, public_columns));
        let one_target = Mat::concat(
            ConcatAxis::Rows,
            vec![one_public_key.matrix.clone() - gadget.clone(), zero_row.clone()],
        );
        let one_preimage = input_preprocessing.final_trapdoors[0]
            .sample_preimage(one_target, (state_columns, public_columns))
            .as_mat();

        let mut witness_preimages = Vec::with_capacity(witness_size);
        for (bit, public_key) in public_keys[1..].iter().enumerate() {
            let digit = bit / self.config.batch_bits;
            let bit_in_digit = bit % self.config.batch_bits;
            let state = self.config.bit_state_index(digit, bit_in_digit)?;
            let target =
                Mat::concat(ConcatAxis::Rows, vec![public_key.matrix.clone(), -gadget.clone()]);
            witness_preimages.push(
                input_preprocessing.final_trapdoors[state]
                    .sample_preimage(target, (state_columns, public_columns))
                    .as_mat(),
            );
        }

        let k_public_key = BggPublicKeyWire {
            matrix: ring.hash_matrix(
                hash_key.clone(),
                self.tag(b":k_public_key"),
                (1, public_columns),
            ),
            reveal_plaintext: false,
        };
        let first_column = Some(IndexRange { start: 0.into(), end: 1.into() });
        let k_public_key_first = k_public_key.matrix.clone().slice(None, first_column.clone());
        let half_modulus = &self.config.modulus / BigInt::from(2);
        let k_selector = ring.polynomial([half_modulus.into()]);
        let k_target = Mat::concat(ConcatAxis::Rows, vec![k_public_key_first.clone(), k_selector]);
        let k_preimage = input_preprocessing.final_trapdoors[0]
            .sample_preimage(k_target, (state_columns, 1))
            .as_mat();

        let r = ring.hash_matrix(hash_key, self.tag(b":r"), (1, public_columns));
        let r_decomposed = r
            .slice(None, first_column)
            .decompose(self.config.gadget_base_expr(), self.config.digit_count_expr())
            .as_mat();
        let difference = public_key_compiler.sub(&one_public_key, &circuit_output);
        let decoder_public_key = k_public_key_first + difference.matrix * r_decomposed.clone();
        let decoder_target =
            Mat::concat(ConcatAxis::Rows, vec![decoder_public_key, ring.zero((1, 1))]);
        let decoder_preimage = input_preprocessing.final_trapdoors[0]
            .sample_preimage(decoder_target, (state_columns, 1))
            .as_mat();

        let mut context = DslContext::new("diamond-we-encryption")
            .public_output(DiamondArtifactNames::INITIAL_STATE, input_preprocessing.p)?
            .public_output(DiamondArtifactNames::ONE_PREIMAGE, one_preimage)?
            .public_output(DiamondArtifactNames::K_PREIMAGE, k_preimage)?
            .public_output(DiamondArtifactNames::DECODER_PREIMAGE, decoder_preimage)?
            .public_output(DiamondArtifactNames::R_DECOMPOSED, r_decomposed)?;
        for (index, public_key) in public_keys.into_iter().enumerate() {
            context = context
                .public_output(DiamondArtifactNames::public_key(index), public_key.matrix)?;
        }
        for (level_index, level) in input_preprocessing.transitions.into_iter().enumerate() {
            for (digit, transitions) in level.into_iter().enumerate() {
                for (state, transition) in transitions.into_iter().enumerate() {
                    context = context.public_output(
                        DiamondArtifactNames::transition(level_index + 1, digit, state),
                        transition,
                    )?;
                }
            }
        }
        for (bit, preimage) in witness_preimages.into_iter().enumerate() {
            context =
                context.public_output(DiamondArtifactNames::witness_preimage(bit), preimage)?;
        }
        Ok(DiamondEncryptionGraph { graph: context.build()? })
    }

    pub fn build_decryption<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        instance: &[bool],
        encryption: ProductionId,
    ) -> Result<DiamondDecryptionGraph, DiamondCompileError> {
        let mut lookup = mxx_bgg::NoPublicLookup::default();
        let mut slots = mxx_bgg::NoSlotOperations::default();
        self.build_decryption_with_lowerings(circuit, instance, encryption, &mut lookup, &mut slots)
    }

    pub fn build_decryption_with_lowerings<P, L, S>(
        &self,
        circuit: &PolyCircuit<P>,
        instance: &[bool],
        encryption: ProductionId,
        lookup: &mut L,
        slots: &mut S,
    ) -> Result<DiamondDecryptionGraph, DiamondCompileError>
    where
        P: Poly,
        L: PublicLookupLowering<P, Wire = BggEncodingWire, Error = CircuitCompileError>,
        S: SlotOperationLowering<P, Wire = BggEncodingWire, Error = CircuitCompileError>,
    {
        self.validate_circuit(circuit, instance.len())?;
        let ring = self.config.ring();
        let state_columns = self.config.state_columns()?;
        let public_columns = self.config.public_key_columns()?;
        let mut states = vec![ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::INITIAL_STATE,
            (1, state_columns),
            ArtifactConfidentiality::Public,
        )];
        let witness_digits = (0..self.config.input_count)
            .map(|digit| {
                ring.input(format!("witness-digit-{digit}"), (1, 1)).extract_coefficient(0)
            })
            .collect::<Vec<_>>();
        for level in 1..=self.config.input_count {
            let first_new_state = 1 + (level - 1) * self.config.batch_bits;
            let state_count = self.config.state_count_at_level(level)?;
            let mut source_states = Vec::with_capacity(state_count);
            let mut selected_transitions = Vec::with_capacity(state_count);
            for state in 0..state_count {
                let branches = (0..self.config.digit_base)
                    .map(|digit| {
                        ring.artifact_input(
                            encryption.clone(),
                            DiamondArtifactNames::transition(level, digit, state),
                            (state_columns, state_columns),
                            ArtifactConfidentiality::Public,
                        )
                    })
                    .collect();
                let transition = witness_digits[level - 1].clone().select(branches)?;
                let source = if state >= first_new_state { 0 } else { state };
                source_states.push(states[source].clone());
                selected_transitions.push(transition);
            }
            let next_states = Family::pack(source_states)?
                .parallel_zip(Family::pack(selected_transitions)?, |_, state, transition| {
                    state * transition
                })?;
            states = (0..state_count).map(|state| next_states.get_static(state)).collect();
        }

        let public_key_compiler = self.public_key_compiler();
        let encoding_compiler = BggEncodingCompiler { public_key: public_key_compiler.clone() };
        let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
        let witness_size = self.config.witness_size()?;
        let public_keys = (0..=witness_size)
            .map(|index| BggPublicKeyWire {
                matrix: ring.artifact_input(
                    encryption.clone(),
                    DiamondArtifactNames::public_key(index),
                    (1, public_columns),
                    ArtifactConfidentiality::Public,
                ),
                reveal_plaintext: true,
            })
            .collect::<Vec<_>>();
        let one_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::ONE_PREIMAGE,
            (state_columns, public_columns),
            ArtifactConfidentiality::Public,
        );
        let k_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::K_PREIMAGE,
            (state_columns, 1),
            ArtifactConfidentiality::Public,
        );
        let decoder_preimage = ring.artifact_input(
            encryption.clone(),
            DiamondArtifactNames::DECODER_PREIMAGE,
            (state_columns, 1),
            ArtifactConfidentiality::Public,
        );
        let one_vector = states[0].clone() * one_preimage;
        let k_vector = states[0].clone() * k_preimage;
        let decoder = states[0].clone() * decoder_preimage;
        let one_encoding = BggEncodingWire {
            vector: one_vector,
            pubkey: public_keys[0].clone(),
            plaintext: Some(ring.identity(1)),
        };
        let witness_preimages = Family::pack(
            (0..witness_size)
                .map(|bit| {
                    ring.artifact_input(
                        encryption.clone(),
                        DiamondArtifactNames::witness_preimage(bit),
                        (state_columns, public_columns),
                        ArtifactConfidentiality::Public,
                    )
                })
                .collect(),
        )?;
        let witness_states = Family::pack(
            (0..witness_size)
                .map(|bit| {
                    let digit = bit / self.config.batch_bits;
                    let bit_in_digit = bit % self.config.batch_bits;
                    self.config
                        .bit_state_index(digit, bit_in_digit)
                        .map(|state| states[state].clone())
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let witness_vectors = witness_states
            .parallel_zip(witness_preimages, |_, state, preimage| state * preimage)?;
        let mut circuit_inputs = Vec::with_capacity(circuit.num_input());
        for bit in 0..witness_size {
            let digit = bit / self.config.batch_bits;
            let bit_in_digit = bit % self.config.batch_bits;
            let plaintext = witness_digits[digit]
                .clone()
                .bit(bit_in_digit)
                .to_int()
                .select(vec![ring.zero((1, 1)), ring.identity(1)])?;
            circuit_inputs.push(BggEncodingWire {
                vector: witness_vectors.get_static(bit),
                pubkey: public_keys[bit + 1].clone(),
                plaintext: Some(plaintext),
            });
        }
        let zero_encoding = encoding_compiler.small_scalar_mul(&one_encoding, &ring.zero((1, 1)));
        circuit_inputs.extend(
            instance.iter().map(
                |bit| {
                    if *bit { one_encoding.clone() } else { zero_encoding.clone() }
                },
            ),
        );
        let circuit_output = circuit_compiler
            .compile_encodings_with_lowerings(
                circuit,
                one_encoding.clone(),
                circuit_inputs,
                lookup,
                slots,
            )?
            .into_iter()
            .next()
            .ok_or(DiamondCompileError::OutputCount)?;

        let r_decomposed = ring.artifact_input(
            encryption,
            DiamondArtifactNames::R_DECOMPOSED,
            (public_columns, 1),
            ArtifactConfidentiality::Public,
        );
        let projected_difference = (one_encoding.vector - circuit_output.vector) * r_decomposed;
        let noisy_plaintext = decoder - (k_vector + projected_difference);
        let decoded = decode_boolean_interval(noisy_plaintext.clone(), &self.config.modulus);
        let graph = DslContext::new("diamond-we-decryption")
            .output(NOISY_PLAINTEXT_OUTPUT, noisy_plaintext)?
            .bool_output(DECODED_OUTPUT, decoded)?
            .build()?;
        Ok(DiamondDecryptionGraph { graph })
    }

    fn validate_circuit<P: Poly>(
        &self,
        circuit: &PolyCircuit<P>,
        instance_size: usize,
    ) -> Result<(), DiamondCompileError> {
        if circuit.num_output() != 1 {
            return Err(DiamondCompileError::OutputCount);
        }
        if self.config.witness_size()?.checked_add(instance_size) != Some(circuit.num_input()) {
            return Err(DiamondCompileError::InputCount);
        }
        Ok(())
    }

    fn public_key_compiler(&self) -> BggPublicKeyCompiler {
        BggPublicKeyCompiler {
            ring: self.config.ring(),
            base: self.config.gadget_base_expr(),
            digit_count: self.config.digit_count_expr(),
        }
    }

    fn sampler_layout(&self) -> BggSamplerLayout {
        BggSamplerLayout {
            modulus: self.config.modulus.clone().into(),
            ring_dimension: self.config.ring_dimension.into(),
            secret_dimension: 1,
            digit_count: self.config.digit_count,
            gadget_base: self.config.gadget_base_expr(),
        }
    }

    fn tag(&self, suffix: &[u8]) -> Vec<u8> {
        let mut tag = self.config.bgg_tag.clone();
        tag.extend_from_slice(suffix);
        tag
    }
}

/// Decodes a Boolean from the closed interval
/// `[floor(q/4), 3*floor(q/4)]` over canonical residues.
fn decode_boolean_interval(noisy_plaintext: Mat, modulus: &BigInt) -> Bool {
    let coefficient = noisy_plaintext.extract_coefficient(0);
    let quarter = modulus / BigInt::from(4);
    let upper = &quarter * BigInt::from(3);
    let lower_ok = Int::constant(quarter).less_equal(coefficient.clone());
    let upper_ok = coefficient.less_equal(Int::constant(upper));
    lower_ok.to_int().add(upper_ok.to_int()).equal(Int::constant(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_gadgets::circuit::PolyCircuit;
    use mxx_ir_core::{
        ParamEnv, RealExpr,
        artifact::{SpecHash, export_validated_manifest},
        types::ConcreteWireType,
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::PolyBackend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigUint, Sign};
    use std::collections::BTreeMap;

    fn compiler() -> DiamondWeCompiler {
        DiamondWeCompiler::new(DiamondWeConfig {
            modulus: 257.into(),
            ring_dimension: 8,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: 4.into(),
            digit_count: 2,
            trapdoor_sigma: RealExpr::from_integer(4),
            error_sigma: RealExpr::from_integer(3),
            bgg_tag: b"diamond-graph-test".to_vec(),
        })
        .unwrap()
    }

    fn identity_circuit() -> PolyCircuit<DCRTPoly> {
        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1);
        circuit.output([input]);
        circuit
    }

    #[test]
    fn encryption_and_decryption_are_valid_manifest_linked_graphs() {
        let compiler = compiler();
        let circuit = identity_circuit();
        let bindings = ParamEnv::default();
        let encryption = compiler.build_encryption(&circuit, &[]).unwrap();
        let validated_encryption = encryption.graph.validate(&bindings).unwrap();
        for (name, expected_rows, expected_columns) in [
            (DiamondArtifactNames::K_PREIMAGE, compiler.config.state_columns().unwrap(), 1),
            (DiamondArtifactNames::DECODER_PREIMAGE, compiler.config.state_columns().unwrap(), 1),
            (DiamondArtifactNames::R_DECOMPOSED, compiler.config.public_key_columns().unwrap(), 1),
        ] {
            let output = validated_encryption.source.outputs().get(name).unwrap();
            let ConcreteWireType::Matrix(matrix_type) =
                &validated_encryption.root_scope().wire_types[&output.value]
            else {
                panic!("{name} must be a matrix")
            };
            assert_eq!((matrix_type.rows, matrix_type.columns), (expected_rows, expected_columns));
        }
        assert!((0..=compiler.config.witness_size().unwrap()).all(|index| {
            validated_encryption
                .source
                .outputs()
                .contains_key(&DiamondArtifactNames::public_key(index))
        }));
        let tags = validated_encryption
            .scopes
            .values()
            .flat_map(|scope| &scope.execution_order)
            .filter_map(|node| match node.kind() {
                mxx_ir_core::node::NodeKind::HashSample { tag_prefix, .. } => {
                    Some(tag_prefix.as_slice())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(tags.contains(&b"diamond-graph-test:witness_public_keys".as_slice()));
        assert!(tags.contains(&b"diamond-graph-test:k_public_key".as_slice()));
        assert!(tags.contains(&b"diamond-graph-test:r".as_slice()));
        let production = ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [9; 32] };
        let manifest =
            export_validated_manifest(production.clone(), &validated_encryption).unwrap();
        let decryption = compiler.build_decryption(&circuit, &[], production.clone()).unwrap();
        let validated_decryption = decryption
            .graph
            .validate_with_manifests(&bindings, &[(production, manifest)].into_iter().collect())
            .unwrap();
        assert!(
            validated_decryption
                .source
                .root_scope()
                .nodes()
                .iter()
                .any(|node| matches!(node.kind(), mxx_ir_core::node::NodeKind::ParallelLoop(_)))
        );
    }

    #[test]
    fn boolean_decoder_accepts_the_closed_interval_at_odd_modulus_boundaries() {
        let parameters = DCRTPolyParams::new(4, 2, 51, 4);
        let modulus: std::sync::Arc<BigUint> = parameters.modulus();
        let modulus_int = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
        let ring = mxx_dsl::Ring::new(modulus_int.clone(), parameters.ring_dimension() as usize);
        let input = ring.input("input", (1, 1));
        let decoded = decode_boolean_interval(input, &modulus_int);
        let graph = DslContext::new("diamond-we-boolean-decode")
            .bool_output("message", decoded)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();

        let quarter = modulus.as_ref() / 4u32;
        let upper = &quarter * 3u32;
        for (residue, expected) in [
            (&quarter - 1u32, false),
            (quarter.clone(), true),
            (upper.clone(), true),
            (&upper + 1u32, false),
        ] {
            let value = DCRTPoly::from_biguint_to_constant(&parameters, residue.clone());
            let value = DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![value]]);
            let mut backend = PolyBackend::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<keccak_asm::Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new([parameters.clone()]);
            let result = execute(
                &graph,
                &mut backend,
                BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(value))]),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
            )
            .unwrap();
            let Some(RuntimeValue::Bool(actual)) = result.outputs.get("message") else {
                panic!("decoder must return a boolean")
            };
            assert_eq!(
                *actual, expected,
                "raw residue {residue}, modulus {modulus}, quarter {quarter}, upper {upper}"
            );
        }
    }
}
