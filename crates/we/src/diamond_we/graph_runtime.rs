use super::{
    DiamondWE,
    graph::{
        DiamondInjectorGraphConfig, DiamondInjectorGraphError, DiamondWEArtifactNames,
        DiamondWEGraphConfig, build_keygen_graph, build_we_evaluation_graph,
    },
};
use crate::{
    circuit::PolyCircuit,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
};
use mxx_ir_core::{
    IntExpr, ParamEnv, RealExpr, ValidatedGraph,
    artifact::ProductionId,
    encoding::{EncodingError, hash_canonical, spec_hash},
    expr::ExprError,
    validate, validate_with_manifests,
};
use mxx_runtime::{
    FilesystemArtifactStore, FilesystemStoreError, RuntimeValue, SessionAliasDescriptor,
    SessionStore, artifact::ArtifactStore, backend::poly::PolyBackend, execute, execute_in_session,
    executor::ExecutionError, transcript::SamplingMode,
};
use num_bigint::{BigInt, Sign};
use num_traits::One;
use serde::Serialize;
use std::{collections::BTreeMap, fmt::Debug};
use thiserror::Error;

#[derive(Serialize)]
struct DiamondWEEncryptionRequest<'a> {
    domain: &'static str,
    message: bool,
    instance: &'a [bool],
}

#[derive(Clone, Debug)]
pub struct DiamondWECiphertext<P: Poly> {
    pub circuit: PolyCircuit<P>,
    pub instance: Vec<bool>,
    pub hash_key: [u8; 32],
    pub production_id: ProductionId,
}

#[derive(Debug, Error)]
pub enum DiamondWEGraphRuntimeError {
    #[error(transparent)]
    Build(#[from] DiamondInjectorGraphError),
    #[error(transparent)]
    Expression(#[from] ExprError),
    #[error(transparent)]
    Encoding(#[from] EncodingError),
    #[error(transparent)]
    Filesystem(#[from] FilesystemStoreError),
    #[error(transparent)]
    Execution(#[from] ExecutionError),
    #[error("Graph IR validation failed: {0}")]
    Validation(String),
    #[error("Diamond WE key generation did not produce a production identity")]
    MissingProduction,
    #[error("Diamond WE evaluation did not produce a boolean message")]
    MissingMessage,
    #[error("a Diamond WE graph dimension overflowed usize")]
    DimensionOverflow,
    #[error("Diamond WE witness length must equal the configured witness size")]
    WitnessLength,
    #[error("Diamond WE fixed-witness artifacts can only evaluate the witness selected at setup")]
    FixedWitnessMismatch,
    #[error("Diamond WE ciphertext nonce does not match its production identity")]
    ProductionNonceMismatch,
    #[error("Diamond WE ciphertext circuit does not match its production identity")]
    ProductionGraphMismatch,
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix + Send + Sync + 'static,
    M::P: 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    TS::Trapdoor: Clone + Debug,
{
    fn graph_config(
        &self,
        instance_size: usize,
    ) -> Result<DiamondWEGraphConfig, DiamondWEGraphRuntimeError> {
        let parameters = &self.injector.params;
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus().into();
        let secret_rows = crate::input_injector::DIAMOND_SECRET_SIZE
            .checked_mul(2)
            .ok_or(DiamondWEGraphRuntimeError::DimensionOverflow)?;
        let state_columns = secret_rows
            .checked_mul(
                parameters
                    .modulus_digits()
                    .checked_add(2)
                    .ok_or(DiamondWEGraphRuntimeError::DimensionOverflow)?,
            )
            .ok_or(DiamondWEGraphRuntimeError::DimensionOverflow)?;
        let bgg_columns = crate::input_injector::DIAMOND_SECRET_SIZE
            .checked_mul(parameters.modulus_digits())
            .ok_or(DiamondWEGraphRuntimeError::DimensionOverflow)?;
        let gadget_base = BigInt::one() << parameters.base_bits() as usize;
        Ok(DiamondWEGraphConfig {
            injector: DiamondInjectorGraphConfig {
                modulus: IntExpr::constant(BigInt::from_biguint(
                    Sign::Plus,
                    modulus.as_ref().clone(),
                )),
                ring_dimension: IntExpr::constant(parameters.ring_dimension()),
                state_columns: IntExpr::constant(state_columns),
                concrete_state_columns: state_columns,
                preimage_chunk_columns: mxx_gadgets::env::aux_sampling_chunk_width().max(1),
                input_count: self.injector.input_count,
                base: self.injector.base,
                batch_bits: self.injector.batch_bits,
                trapdoor_sigma: RealExpr::from_f64_exact(self.injector.trapdoor_sigma)?,
                gadget_base: IntExpr::constant(gadget_base.clone()),
                gadget_digit_count: IntExpr::constant(parameters.modulus_digits()),
                error_sigma: RealExpr::from_f64_exact(self.injector.error_sigma)?,
            },
            witness_size: self.witness_size,
            instance_size,
            bgg_columns: IntExpr::constant(bgg_columns),
            concrete_bgg_columns: bgg_columns,
            gadget_base: IntExpr::constant(gadget_base),
            bgg_tag: self.bgg_tag.clone(),
        })
    }

    fn graph_store(&self) -> Result<FilesystemArtifactStore, FilesystemStoreError> {
        FilesystemArtifactStore::open(self.artifact_dir.join("graph-runtime"))
    }

    fn validated_keygen_graph(
        &self,
        circuit: &PolyCircuit<M::P>,
        instance_size: usize,
    ) -> Result<ValidatedGraph, DiamondWEGraphRuntimeError> {
        let config = self.graph_config(instance_size)?;
        let graph = build_keygen_graph(&config, &DiamondWEArtifactNames::default(), circuit)?;
        validate(&graph, &ParamEnv::default())
            .map_err(|error| DiamondWEGraphRuntimeError::Validation(error.to_string()))
    }

    fn execute_keygen_graph(
        &self,
        message: bool,
        circuit: PolyCircuit<M::P>,
        instance: &[bool],
        hash_key: [u8; 32],
        graph: &ValidatedGraph,
        store: &mut FilesystemArtifactStore,
    ) -> Result<DiamondWECiphertext<M::P>, DiamondWEGraphRuntimeError> {
        let parameters = self.injector.params.clone();
        let plaintext =
            if message { M::P::const_one(&parameters) } else { M::P::const_zero(&parameters) };
        let plaintext = M::from_poly_vec(&parameters, vec![vec![plaintext]]);
        let mut inputs = BTreeMap::from([
            ("k".to_owned(), RuntimeValue::matrix(plaintext)),
            ("hash_key".to_owned(), RuntimeValue::Bytes(hash_key.to_vec())),
        ]);
        for (index, bit) in instance.iter().copied().enumerate() {
            inputs.insert(format!("instance_{index}"), RuntimeValue::Bool(bit));
        }
        let mut backend = PolyBackend::<M, US, HS, TS>::new_for_execution_on(
            [parameters],
            &self.injector.gpu_device_ids,
        );
        let result = execute_in_session(graph, &mut backend, inputs, store, hash_key)?;
        let production_id =
            result.production_id.ok_or(DiamondWEGraphRuntimeError::MissingProduction)?;
        Ok(DiamondWECiphertext { circuit, instance: instance.to_vec(), hash_key, production_id })
    }

    pub fn try_encrypt(
        &self,
        message: bool,
        circuit: PolyCircuit<M::P>,
        instance: &[bool],
    ) -> Result<DiamondWECiphertext<M::P>, DiamondWEGraphRuntimeError> {
        let graph = self.validated_keygen_graph(&circuit, instance.len())?;
        let request_digest = hash_canonical(&DiamondWEEncryptionRequest {
            domain: "mxx-diamond-we-encryption-request-v1",
            message,
            instance,
        })?;
        let descriptor = SessionAliasDescriptor::new(
            "diamond-we-encryption",
            graph.source.name.clone(),
            spec_hash(&graph.source, &graph.bindings)?,
            request_digest,
        );
        let mut store = self.graph_store()?;
        let hash_key = store.resolve_session_nonce(&descriptor)?;
        self.execute_keygen_graph(message, circuit, instance, hash_key, &graph, &mut store)
    }

    pub fn try_encrypt_with_nonce(
        &self,
        message: bool,
        circuit: PolyCircuit<M::P>,
        instance: &[bool],
        hash_key: [u8; 32],
    ) -> Result<DiamondWECiphertext<M::P>, DiamondWEGraphRuntimeError> {
        let graph = self.validated_keygen_graph(&circuit, instance.len())?;
        let mut store = self.graph_store()?;
        self.execute_keygen_graph(message, circuit, instance, hash_key, &graph, &mut store)
    }

    pub fn try_decrypt(
        &self,
        ciphertext: &DiamondWECiphertext<M::P>,
        witness: &[bool],
    ) -> Result<bool, DiamondWEGraphRuntimeError> {
        if witness.len() != self.witness_size {
            return Err(DiamondWEGraphRuntimeError::WitnessLength);
        }
        if self.target_witness.as_deref().is_some_and(|target| target != witness) {
            return Err(DiamondWEGraphRuntimeError::FixedWitnessMismatch);
        }
        if ciphertext.hash_key != ciphertext.production_id.execution_nonce {
            return Err(DiamondWEGraphRuntimeError::ProductionNonceMismatch);
        }
        let config = self.graph_config(ciphertext.instance.len())?;
        let names = DiamondWEArtifactNames::default();
        let keygen_graph = build_keygen_graph(&config, &names, &ciphertext.circuit)?;
        if spec_hash(&keygen_graph, &ParamEnv::default())? != ciphertext.production_id.spec_hash {
            return Err(DiamondWEGraphRuntimeError::ProductionGraphMismatch);
        }
        let mut store = self.graph_store()?;
        let manifest = store.load_manifest(&ciphertext.production_id)?;
        let graph = build_we_evaluation_graph(
            &config,
            &names,
            ciphertext.production_id.clone(),
            &ciphertext.circuit,
        )?;
        let graph = validate_with_manifests(
            &graph,
            &ParamEnv::default(),
            &BTreeMap::from([(ciphertext.production_id.clone(), manifest)]),
        )
        .map_err(|error| DiamondWEGraphRuntimeError::Validation(error.to_string()))?;
        let witness_digits = witness
            .chunks_exact(self.injector.batch_bits)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |digit, (bit, value)| digit | (u32::from(*value) << bit))
            })
            .collect::<Vec<_>>();
        let mut inputs = BTreeMap::new();
        for (level, digit) in witness_digits.into_iter().enumerate() {
            inputs.insert(format!("digit_{}", level + 1), RuntimeValue::Int(BigInt::from(digit)));
        }
        for (index, bit) in ciphertext.instance.iter().copied().enumerate() {
            inputs.insert(format!("instance_{index}"), RuntimeValue::Bool(bit));
        }
        let parameters = self.injector.params.clone();
        let mut backend = PolyBackend::<M, US, HS, TS>::new_for_execution_on(
            [parameters],
            &self.injector.gpu_device_ids,
        );
        let result = execute(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh)?;
        match result.outputs.get("message") {
            Some(RuntimeValue::Bool(message)) => Ok(*message),
            _ => Err(DiamondWEGraphRuntimeError::MissingMessage),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        input_injector::DiamondInjector,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use mxx_ir_core::{GraphBuilder, types::MatrixType};
    use mxx_runtime::artifact::MemoryArtifactStore;
    use num_bigint::BigUint;

    type TestDiamondWE = DiamondWE<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<keccak_asm::Keccak256>,
        DCRTPolyTrapdoorSampler,
    >;

    #[test]
    fn graph_decoder_matches_legacy_closed_interval_at_odd_modulus_boundaries() {
        let parameters = DCRTPolyParams::new(4, 2, 51, 4);
        let modulus: std::sync::Arc<BigUint> = parameters.modulus().into();
        let modulus_int = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(modulus_int.clone()),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let mut builder = GraphBuilder::new("diamond-we-legacy-boolean-decode", Vec::new());
        let input = builder.input("input", matrix_type);
        let decoded =
            crate::diamond_we::graph::legacy_boolean_decode(&mut builder, &input, &modulus_int);
        builder.value_output_wire("message", decoded);
        let graph = validate(&builder.finish(), &ParamEnv::default())
            .expect("legacy decoder graph validates");

        let quarter = modulus.as_ref() / 4u32;
        let upper = &quarter * 3u32;
        let cases = [
            (&quarter - 1u32, false),
            (quarter.clone(), true),
            (upper.clone(), true),
            (&upper + 1u32, false),
        ];
        for (residue, expected) in cases {
            let value = DCRTPoly::from_biguint_to_constant(&parameters, residue.clone());
            let value = DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![value]]);
            let mut backend = PolyBackend::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<keccak_asm::Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new([parameters.clone()]);
            let mut store = MemoryArtifactStore::default();
            let result = execute(
                &graph,
                &mut backend,
                BTreeMap::from([("input".to_owned(), RuntimeValue::matrix(value))]),
                &mut store,
                SamplingMode::Fresh,
            )
            .expect("legacy decoder graph executes");
            let Some(RuntimeValue::Bool(actual)) = result.outputs.get("message") else {
                panic!("legacy decoder graph must return a boolean")
            };
            assert_eq!(
                *actual, expected,
                "raw residue {residue} must preserve the legacy decode boundary"
            );
        }
    }

    #[test]
    fn graph_runtime_encrypts_decrypts_and_replays_one_session() {
        let parameters = DCRTPolyParams::new(4, 2, 51, 4);
        let injector = DiamondInjector::new(parameters, 1, 2, 1, 4.578, 0.0);
        let directory = tempfile::tempdir().expect("temporary graph store");
        let scheme = TestDiamondWE::new(
            injector,
            1,
            directory.path(),
            b"diamond-we-graph-runtime".to_vec(),
            None,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1).as_single_wire();
        circuit.output([input]);
        let hash_key = [41; 32];

        let first = scheme
            .try_encrypt_with_nonce(true, circuit.clone(), &[], hash_key)
            .expect("first graph encryption");
        let replayed = scheme
            .try_encrypt_with_nonce(true, circuit, &[], hash_key)
            .expect("replayed graph encryption");
        assert_eq!(first.production_id, replayed.production_id);
        assert_eq!(first.hash_key, replayed.hash_key);
        assert!(
            scheme.try_decrypt(&first, &[true]).expect("graph decryption"),
            "the identity circuit accepts the true witness and reveals the encrypted message"
        );
        let mut wrong_nonce = first.clone();
        wrong_nonce.hash_key = [0; 32];
        assert!(matches!(
            scheme.try_decrypt(&wrong_nonce, &[true]),
            Err(DiamondWEGraphRuntimeError::ProductionNonceMismatch)
        ));
        let mut different_circuit = PolyCircuit::<DCRTPoly>::new();
        different_circuit.input(1);
        let one = different_circuit.const_one_gate();
        different_circuit.output([one]);
        let mut wrong_circuit = first.clone();
        wrong_circuit.circuit = different_circuit;
        assert!(matches!(
            scheme.try_decrypt(&wrong_circuit, &[true]),
            Err(DiamondWEGraphRuntimeError::ProductionGraphMismatch)
        ));

        let false_ciphertext = scheme
            .try_encrypt_with_nonce(false, first.circuit.clone(), &[], [42; 32])
            .expect("false graph encryption");
        assert!(
            !scheme.try_decrypt(&false_ciphertext, &[true]).expect("false graph decryption"),
            "the same accepting witness reveals the encrypted false message"
        );

        let production_ciphertext =
            crate::WitnessEnc::enc(&scheme, &true, first.circuit.clone(), &Vec::new());
        let resumed_production_ciphertext =
            crate::WitnessEnc::enc(&scheme, &true, first.circuit.clone(), &Vec::new());
        assert_eq!(
            production_ciphertext.production_id, resumed_production_ciphertext.production_id,
            "the public encryption API must resume its named runtime session"
        );
        assert!(
            crate::WitnessEnc::dec(&scheme, &production_ciphertext, &vec![true]),
            "the public WitnessEnc API must execute the Graph IR path"
        );
        assert!(matches!(
            scheme.try_encrypt(false, first.circuit.clone(), &[]),
            Err(DiamondWEGraphRuntimeError::Filesystem(FilesystemStoreError::Conflict(_)))
        ));

        let mut advanced = PolyCircuit::<DCRTPoly>::new();
        let input = advanced.input(1).as_single_wire();
        let transferred = advanced.slot_transfer_gate(input, &[(0, None)]).as_single_wire();
        advanced.output([transferred]);
        assert!(matches!(
            scheme.try_encrypt_with_nonce(true, advanced, &[], [43; 32]),
            Err(DiamondWEGraphRuntimeError::Build(_))
        ));
    }
}
