use super::{DiamondCompileError, DiamondConfigError, DiamondWeCompiler};
use crate::WitnessEnc;
use mxx_gadgets::{Poly, circuit::PolyCircuit};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ProductionId, production_id},
    encoding::spec_hash,
};
use mxx_primitives::{matrix::PolyMatrix, poly::PolyParams};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue, SessionStore, backend::poly::PolyBackend, execute,
    execute_in_session_with_config, transcript::SamplingMode,
};
use rand::random;
use std::collections::BTreeMap;
use thiserror::Error;

use super::graph::{DECODED_OUTPUT, HASH_KEY_INPUT, MESSAGE_INPUT};

#[derive(Clone)]
pub struct DiamondWeCiphertext<P: Poly> {
    pub circuit: PolyCircuit<P>,
    pub instance: Vec<bool>,
    pub hash_key: [u8; 32],
    pub encryption: ProductionId,
}

pub struct DiamondWeRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
{
    pub compiler: DiamondWeCompiler,
    pub parameters: <M::P as Poly>::Params,
    pub backend: PolyBackend<M, U, H, T>,
    pub store: S,
    pub execution_config: ExecutionConfig,
}

#[derive(Debug, Error)]
pub enum DiamondRuntimeError {
    #[error(transparent)]
    Config(#[from] DiamondConfigError),
    #[error(transparent)]
    Compile(#[from] DiamondCompileError),
    #[error("Diamond runtime graph validation failed: {0}")]
    Validation(String),
    #[error("Diamond runtime execution failed: {0}")]
    Execution(String),
    #[error("Diamond artifact store failed: {0}")]
    Store(String),
    #[error("the runtime parameters do not match the Diamond compiler layout")]
    ParameterMismatch,
    #[error("the supplied witness has the wrong length")]
    WitnessLength,
    #[error("the ciphertext hash key does not match its production identity")]
    ProductionNonceMismatch,
    #[error("the ciphertext circuit does not match its production identity")]
    ProductionGraphMismatch,
    #[error("the Diamond decryption graph did not return a boolean")]
    DecodeOutput,
}

impl<M, U, H, T, S> DiamondWeRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    pub fn new(
        compiler: DiamondWeCompiler,
        parameters: <M::P as Poly>::Params,
        store: S,
    ) -> Result<Self, DiamondRuntimeError> {
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus().into();
        if compiler.config.modulus != num_bigint::BigInt::from(modulus.as_ref().clone()) ||
            compiler.config.ring_dimension != parameters.ring_dimension() as usize ||
            compiler.config.digit_count != parameters.modulus_digits() ||
            compiler.config.gadget_base !=
                num_bigint::BigInt::from(1u64 << parameters.base_bits())
        {
            return Err(DiamondRuntimeError::ParameterMismatch);
        }
        Ok(Self {
            compiler,
            backend: PolyBackend::new_for_execution([parameters.clone()]),
            parameters,
            store,
            execution_config: ExecutionConfig::default(),
        })
    }

    pub fn with_execution_config(mut self, execution_config: ExecutionConfig) -> Self {
        self.execution_config = execution_config;
        self
    }

    pub fn encrypt(
        &mut self,
        message: bool,
        circuit: PolyCircuit<M::P>,
        instance: &[bool],
    ) -> Result<DiamondWeCiphertext<M::P>, DiamondRuntimeError> {
        self.encrypt_with_hash_key(message, circuit, instance, random())
    }

    pub fn encrypt_with_hash_key(
        &mut self,
        message: bool,
        circuit: PolyCircuit<M::P>,
        instance: &[bool],
        hash_key: [u8; 32],
    ) -> Result<DiamondWeCiphertext<M::P>, DiamondRuntimeError> {
        let built = self.compiler.build_encryption(&circuit, instance)?.graph;
        let bindings = ParamEnv::default();
        let validated = built
            .validate(&bindings)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        // The hash key is also the deterministic production nonce that binds
        // the sampled matrices to this session.
        let nonce = hash_key;
        let production = production_id(
            spec_hash(&validated.source, &validated.bindings)
                .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?,
            nonce,
        );
        let message_matrix = self.scalar_matrix(usize::from(message));
        execute_in_session_with_config(
            &validated,
            &mut self.backend,
            BTreeMap::from([
                (HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec())),
                (MESSAGE_INPUT.to_owned(), RuntimeValue::matrix(message_matrix)),
            ]),
            &mut self.store,
            nonce,
            self.execution_config,
        )
        .map_err(|error| DiamondRuntimeError::Execution(error.to_string()))?;
        Ok(DiamondWeCiphertext {
            circuit,
            instance: instance.to_vec(),
            hash_key,
            encryption: production,
        })
    }

    pub fn decrypt(
        &mut self,
        ciphertext: &DiamondWeCiphertext<M::P>,
        witness: &[bool],
    ) -> Result<bool, DiamondRuntimeError> {
        if witness.len() != self.compiler.config.witness_size()? {
            return Err(DiamondRuntimeError::WitnessLength);
        }
        if ciphertext.hash_key != ciphertext.encryption.execution_nonce {
            return Err(DiamondRuntimeError::ProductionNonceMismatch);
        }
        let encryption_graph =
            self.compiler.build_encryption(&ciphertext.circuit, &ciphertext.instance)?.graph;
        let encryption_graph = encryption_graph
            .validate(&ParamEnv::default())
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        let graph_hash = spec_hash(&encryption_graph.source, &encryption_graph.bindings)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        if graph_hash != ciphertext.encryption.spec_hash {
            return Err(DiamondRuntimeError::ProductionGraphMismatch);
        }
        let built = self
            .compiler
            .build_decryption(
                &ciphertext.circuit,
                &ciphertext.instance,
                ciphertext.encryption.clone(),
            )?
            .graph;
        let manifest = self
            .store
            .load_manifest(&ciphertext.encryption)
            .map_err(|error| DiamondRuntimeError::Store(error.to_string()))?;
        let manifests = BTreeMap::from([(ciphertext.encryption.clone(), manifest)]);
        let validated = built
            .validate_with_manifests(&ParamEnv::default(), &manifests)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        let mut inputs = BTreeMap::new();
        for (digit, value) in self.pack_witness_digits(witness)?.into_iter().enumerate() {
            inputs.insert(
                format!("witness-digit-{digit}"),
                RuntimeValue::matrix(self.scalar_matrix(value)),
            );
        }
        let result =
            execute(&validated, &mut self.backend, inputs, &mut self.store, SamplingMode::Fresh)
                .map_err(|error| DiamondRuntimeError::Execution(error.to_string()))?;
        let Some(RuntimeValue::Bool(decoded)) = result.outputs.get(DECODED_OUTPUT) else {
            return Err(DiamondRuntimeError::DecodeOutput);
        };
        Ok(*decoded)
    }

    fn scalar_matrix(&self, value: usize) -> M {
        M::from_poly_vec(
            &self.parameters,
            vec![vec![M::P::from_usize_to_constant(&self.parameters, value)]],
        )
    }

    fn pack_witness_digits(&self, witness: &[bool]) -> Result<Vec<usize>, DiamondRuntimeError> {
        if witness.len() != self.compiler.config.witness_size()? {
            return Err(DiamondRuntimeError::WitnessLength);
        }
        Ok(witness
            .chunks_exact(self.compiler.config.batch_bits)
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .fold(0usize, |digit, (bit, value)| digit | (usize::from(*value) << bit))
            })
            .collect())
    }
}

impl<M, U, H, T, S> WitnessEnc<M::P> for DiamondWeRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    type Msg = bool;
    type Inst = Vec<bool>;
    type Wtns = Vec<bool>;
    type Ciphertext = DiamondWeCiphertext<M::P>;
    type Error = DiamondRuntimeError;

    fn enc(
        &mut self,
        msg: &Self::Msg,
        circuit: PolyCircuit<M::P>,
        instance: &Self::Inst,
    ) -> Result<Self::Ciphertext, Self::Error> {
        self.encrypt(*msg, circuit, instance)
    }

    fn dec(
        &mut self,
        ciphertext: &Self::Ciphertext,
        witness: &Self::Wtns,
    ) -> Result<Self::Msg, Self::Error> {
        self.decrypt(ciphertext, witness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{DiamondWeCompiler, DiamondWeConfig};
    use keccak_asm::Keccak256;
    use mxx_ir_core::RealExpr;
    use mxx_primitives::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use mxx_runtime::artifact::MemoryArtifactStore;
    use num_bigint::BigInt;

    type TestRuntime = DiamondWeRuntime<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        MemoryArtifactStore,
    >;

    fn identity_circuit() -> PolyCircuit<<DCRTPolyMatrix as PolyMatrix>::P> {
        let mut circuit = PolyCircuit::new();
        let input = circuit.input(1);
        circuit.output([input]);
        circuit
    }

    fn runtime() -> TestRuntime {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus();
        let compiler = DiamondWeCompiler::new(DiamondWeConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count: parameters.modulus_digits(),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
            error_sigma: RealExpr::from_integer(0),
            bgg_tag: b"diamond-runtime-test".to_vec(),
        })
        .unwrap();
        TestRuntime::new(compiler, parameters, MemoryArtifactStore::default()).unwrap()
    }

    fn batched_runtime() -> TestRuntime {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus();
        let compiler = DiamondWeCompiler::new(DiamondWeConfig {
            modulus: BigInt::from(modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            input_count: 1,
            digit_base: 4,
            batch_bits: 2,
            gadget_base: BigInt::from(1u64 << parameters.base_bits()),
            digit_count: parameters.modulus_digits(),
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
            error_sigma: RealExpr::from_integer(0),
            bgg_tag: b"diamond-runtime-batched-test".to_vec(),
        })
        .unwrap();
        TestRuntime::new(compiler, parameters, MemoryArtifactStore::default()).unwrap()
    }

    #[test]
    fn identity_witness_circuit_round_trips_both_messages() {
        for message in [false, true] {
            let mut runtime = runtime();
            let ciphertext = runtime
                .encrypt_with_hash_key(message, identity_circuit(), &[], [0x42; 32])
                .unwrap();
            assert_eq!(runtime.decrypt(&ciphertext, &[true]).unwrap(), message);
        }
    }

    #[test]
    fn ciphertext_identity_is_bound_to_the_hash_key_and_circuit() {
        let mut runtime = runtime();
        let ciphertext =
            runtime.encrypt_with_hash_key(true, identity_circuit(), &[], [0x42; 32]).unwrap();
        let replay =
            runtime.encrypt_with_hash_key(true, identity_circuit(), &[], [0x42; 32]).unwrap();
        assert_eq!(ciphertext.encryption, replay.encryption);

        let mut wrong_nonce = ciphertext.clone();
        wrong_nonce.hash_key = [0; 32];
        assert!(matches!(
            runtime.decrypt(&wrong_nonce, &[true]),
            Err(DiamondRuntimeError::ProductionNonceMismatch)
        ));

        let mut different_circuit = PolyCircuit::new();
        different_circuit.input(1);
        let one = different_circuit.const_one_gate();
        different_circuit.output([one]);
        let mut wrong_circuit = ciphertext;
        wrong_circuit.circuit = different_circuit;
        assert!(matches!(
            runtime.decrypt(&wrong_circuit, &[true]),
            Err(DiamondRuntimeError::ProductionGraphMismatch)
        ));
    }

    #[test]
    fn batched_witness_bits_drive_the_same_and_circuit_as_the_runtime_digit() {
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(2);
        let output = circuit.and_gate(inputs.at(0), inputs.at(1));
        circuit.output([output]);
        for message in [false, true] {
            let mut runtime = batched_runtime();
            let ciphertext =
                runtime.encrypt_with_hash_key(message, circuit.clone(), &[], [0x24; 32]).unwrap();
            assert_eq!(runtime.decrypt(&ciphertext, &[true, true]).unwrap(), message);
        }
    }
}
