use super::{DiamondCompileError, DiamondConfigError, DiamondWeCompiler};
use crate::WitnessEncryptionRuntime;
use mxx_gadgets::{
    Poly,
    circuit::{
        BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitData, BooleanCircuitError,
        BooleanCircuitShape,
    },
};
use mxx_ir_core::{
    artifact::{ProductionId, production_id},
    encoding::spec_hash,
};
use mxx_primitives::{matrix::PolyMatrix, poly::PolyParams};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue, SessionStore, backend::poly::PolyBackend, execute,
    execute_in_session_with_config, transcript::SamplingMode,
};
use rand::random;
use std::{collections::BTreeMap, time::Instant};
use thiserror::Error;
use tracing::{debug, info};

use super::graph::{DECODED_OUTPUT, HASH_KEY_INPUT, MESSAGE_INPUT, NOISY_PLAINTEXT_OUTPUT};

#[derive(Clone)]
pub struct DiamondWeCiphertext {
    pub hash_key: [u8; 32],
    pub encryption: ProductionId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondDecryptionResult {
    pub decoded: bool,
    /// Canonical coefficients of the noisy plaintext polynomial in [0, q).
    pub noisy_plaintext_coefficients: Vec<num_bigint::BigInt>,
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
    Circuit(#[from] BooleanCircuitError),
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
    #[error("the supplied instance has the wrong length")]
    InstanceLength,
    #[error("the supplied witness has the wrong length")]
    WitnessLength,
    #[error("the ciphertext hash key does not match its production identity")]
    ProductionNonceMismatch,
    #[error("the ciphertext production graph does not match this protocol family")]
    ProductionGraphMismatch,
    #[error("the Diamond decryption graph did not return a boolean")]
    DecodeOutput,
    #[error("the Diamond decryption graph did not return its noisy plaintext matrix")]
    NoisyPlaintextOutput,
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
        circuit: &BooleanCircuitData,
        instance: &[bool],
        message: bool,
    ) -> Result<DiamondWeCiphertext, DiamondRuntimeError> {
        self.encrypt_with_hash_key(circuit, instance, message, random())
    }

    pub fn encrypt_with_hash_key(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        message: bool,
        hash_key: [u8; 32],
    ) -> Result<DiamondWeCiphertext, DiamondRuntimeError> {
        let total_started = Instant::now();
        let validation_started = Instant::now();
        self.validate_public_inputs(circuit, instance)?;
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond encryption inputs"
        );
        let graph_started = Instant::now();
        let built = self.compiler.build_encryption()?.graph;
        debug!(
            elapsed_seconds = graph_started.elapsed().as_secs_f64(),
            "built Diamond encryption graph"
        );
        let validation_started = Instant::now();
        let bindings = self.compiler.circuit_bindings()?;
        let validated = built
            .validate(&bindings)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond encryption graph"
        );
        let production_started = Instant::now();
        let production = production_id(
            spec_hash(&validated.source, &validated.bindings)
                .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?,
            hash_key,
        );
        debug!(
            elapsed_seconds = production_started.elapsed().as_secs_f64(),
            "constructed Diamond encryption production identity"
        );
        let inputs_started = Instant::now();
        let mut inputs = circuit_inputs::<PolyBackend<M, U, H, T>>(circuit, &self.compiler.shape);
        insert_boolean_family_input(
            &mut inputs,
            BOOLEAN_INSTANCE_INPUT,
            instance,
            self.compiler.shape.analyze()?.maximum_layer_width,
        );
        inputs.insert(HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec()));
        inputs.insert(MESSAGE_INPUT.to_owned(), RuntimeValue::Bool(message));
        debug!(
            elapsed_seconds = inputs_started.elapsed().as_secs_f64(),
            "constructed Diamond encryption runtime inputs"
        );
        let execution_started = Instant::now();
        info!("starting Diamond encryption graph execution");
        execute_in_session_with_config(
            &validated,
            &mut self.backend,
            inputs,
            &mut self.store,
            hash_key,
            self.execution_config,
        )
        .map_err(|error| DiamondRuntimeError::Execution(error.to_string()))?;
        info!(
            execution_elapsed_seconds = execution_started.elapsed().as_secs_f64(),
            total_elapsed_seconds = total_started.elapsed().as_secs_f64(),
            "finished Diamond encryption graph execution"
        );
        Ok(DiamondWeCiphertext { hash_key, encryption: production })
    }

    pub fn decrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &DiamondWeCiphertext,
    ) -> Result<bool, DiamondRuntimeError> {
        self.decrypt_internal(circuit, instance, witness, ciphertext, false)
            .map(|(decoded, _)| decoded)
    }

    pub fn decrypt_with_diagnostics(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &DiamondWeCiphertext,
    ) -> Result<DiamondDecryptionResult, DiamondRuntimeError> {
        let (decoded, noisy_plaintext_coefficients) =
            self.decrypt_internal(circuit, instance, witness, ciphertext, true)?;
        Ok(DiamondDecryptionResult {
            decoded,
            noisy_plaintext_coefficients: noisy_plaintext_coefficients
                .ok_or(DiamondRuntimeError::NoisyPlaintextOutput)?,
        })
    }

    fn decrypt_internal(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &DiamondWeCiphertext,
        include_diagnostics: bool,
    ) -> Result<(bool, Option<Vec<num_bigint::BigInt>>), DiamondRuntimeError> {
        let total_started = Instant::now();
        let validation_started = Instant::now();
        self.validate_public_inputs(circuit, instance)?;
        if witness.len() != self.compiler.shape.witness_width {
            return Err(DiamondRuntimeError::WitnessLength);
        }
        if ciphertext.hash_key != ciphertext.encryption.execution_nonce {
            return Err(DiamondRuntimeError::ProductionNonceMismatch);
        }
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond decryption inputs"
        );
        let encryption_graph_started = Instant::now();
        let encryption_graph = self
            .compiler
            .build_encryption()?
            .graph
            .validate(&self.compiler.circuit_bindings()?)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        let graph_hash = spec_hash(&encryption_graph.source, &encryption_graph.bindings)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        if graph_hash != ciphertext.encryption.spec_hash {
            return Err(DiamondRuntimeError::ProductionGraphMismatch);
        }
        debug!(
            elapsed_seconds = encryption_graph_started.elapsed().as_secs_f64(),
            "validated Diamond encryption provenance for decryption"
        );
        let graph_started = Instant::now();
        let built = self.compiler.build_decryption(ciphertext.encryption.clone())?.graph;
        debug!(
            elapsed_seconds = graph_started.elapsed().as_secs_f64(),
            "built Diamond decryption graph"
        );
        let manifest_started = Instant::now();
        let manifest = self
            .store
            .load_manifest(&ciphertext.encryption)
            .map_err(|error| DiamondRuntimeError::Store(error.to_string()))?;
        debug!(
            elapsed_seconds = manifest_started.elapsed().as_secs_f64(),
            "loaded Diamond encryption manifest for decryption"
        );
        let validation_started = Instant::now();
        let validated = built
            .validate_with_manifests(
                &self.compiler.circuit_bindings()?,
                &BTreeMap::from([(ciphertext.encryption.clone(), manifest)]),
            )
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond decryption graph"
        );
        let inputs_started = Instant::now();
        let maximum_width = self.compiler.shape.analyze()?.maximum_layer_width;
        let mut inputs = circuit_inputs::<PolyBackend<M, U, H, T>>(circuit, &self.compiler.shape);
        insert_boolean_family_input(&mut inputs, BOOLEAN_INSTANCE_INPUT, instance, maximum_width);
        insert_boolean_family_input(&mut inputs, BOOLEAN_WITNESS_INPUT, witness, maximum_width);
        debug!(
            elapsed_seconds = inputs_started.elapsed().as_secs_f64(),
            "constructed Diamond decryption runtime inputs"
        );
        let execution_started = Instant::now();
        info!("starting Diamond decryption graph execution");
        let result =
            execute(&validated, &mut self.backend, inputs, &mut self.store, SamplingMode::Fresh)
                .map_err(|error| DiamondRuntimeError::Execution(error.to_string()))?;
        let Some(RuntimeValue::Bool(decoded)) = result.outputs.get(DECODED_OUTPUT) else {
            return Err(DiamondRuntimeError::DecodeOutput);
        };
        let decoded = *decoded;
        let noisy_plaintext_coefficients = if include_diagnostics {
            let Some(RuntimeValue::Matrix(noisy_plaintext)) =
                result.outputs.get(NOISY_PLAINTEXT_OUTPUT)
            else {
                return Err(DiamondRuntimeError::NoisyPlaintextOutput);
            };
            // The ideal plaintext is a constant polynomial. Returning every
            // canonical coefficient lets diagnostics measure the maximum
            // modular error, including nonconstant coefficients that decoding
            // ignores. The ordinary decrypt path deliberately avoids this
            // device-to-host materialization and synchronization.
            Some(
                noisy_plaintext
                    .entry(0, 0)
                    .coeffs_biguints()
                    .into_iter()
                    .map(num_bigint::BigInt::from)
                    .collect(),
            )
        } else {
            None
        };
        info!(
            execution_elapsed_seconds = execution_started.elapsed().as_secs_f64(),
            total_elapsed_seconds = total_started.elapsed().as_secs_f64(),
            "finished Diamond decryption graph execution"
        );
        Ok((decoded, noisy_plaintext_coefficients))
    }

    fn validate_public_inputs(
        &self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
    ) -> Result<(), DiamondRuntimeError> {
        circuit.validate(&self.compiler.shape)?;
        if instance.len() != self.compiler.shape.instance_width {
            return Err(DiamondRuntimeError::InstanceLength);
        }
        Ok(())
    }
}

fn circuit_inputs<B: Backend>(
    circuit: &BooleanCircuitData,
    shape: &BooleanCircuitShape,
) -> BTreeMap<String, RuntimeValue<B>> {
    let maximum_width = shape.analyze().expect("validated Boolean shape").maximum_layer_width;
    let family = |values: Vec<num_bigint::BigInt>| {
        RuntimeValue::Family(values.into_iter().map(RuntimeValue::Int).collect())
    };
    let mut active_gate_counts = Vec::with_capacity(circuit.layers.len());
    let mut kinds = Vec::with_capacity(circuit.layers.len() * maximum_width);
    let mut left = Vec::with_capacity(circuit.layers.len() * maximum_width);
    let mut right = Vec::with_capacity(circuit.layers.len() * maximum_width);
    for gates in &circuit.layers {
        active_gate_counts.push(gates.len().into());
        for slot in 0..maximum_width {
            let gate = gates.get(slot);
            kinds.push(gate.map_or(0, |gate| gate.kind as u8).into());
            left.push(gate.map_or(0, |gate| gate.left).into());
            right.push(gate.map_or(0, |gate| gate.right).into());
        }
    }
    BTreeMap::from([
        ("circuit-active-gate-count".to_owned(), family(active_gate_counts)),
        ("circuit-gate-kind".to_owned(), family(kinds)),
        ("circuit-left-source".to_owned(), family(left)),
        ("circuit-right-source".to_owned(), family(right)),
        ("circuit-output-source".to_owned(), family(vec![circuit.output_source.into()])),
    ])
}

fn insert_boolean_family_input<B: Backend>(
    inputs: &mut BTreeMap<String, RuntimeValue<B>>,
    name: &str,
    values: &[bool],
    maximum_width: usize,
) {
    let mut padded =
        values.iter().map(|value| num_bigint::BigInt::from(*value)).collect::<Vec<_>>();
    padded.resize(maximum_width, 0.into());
    inputs.insert(
        name.to_owned(),
        RuntimeValue::Family(padded.into_iter().map(RuntimeValue::Int).collect()),
    );
}

impl<M, U, H, T, S> WitnessEncryptionRuntime for DiamondWeRuntime<M, U, H, T, S>
where
    M: PolyMatrix,
    S: SessionStore,
    PolyBackend<M, U, H, T>: Backend<Matrix = M>,
{
    type Ciphertext = DiamondWeCiphertext;
    type Message = bool;
    type Error = DiamondRuntimeError;

    fn shape(&self) -> &mxx_gadgets::circuit::BooleanCircuitShape {
        &self.compiler.shape
    }

    fn encrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        message: &bool,
    ) -> Result<Self::Ciphertext, Self::Error> {
        DiamondWeRuntime::encrypt(self, circuit, instance, *message)
    }

    fn decrypt(
        &mut self,
        circuit: &BooleanCircuitData,
        instance: &[bool],
        witness: &[bool],
        ciphertext: &Self::Ciphertext,
    ) -> Result<bool, Self::Error> {
        DiamondWeRuntime::decrypt(self, circuit, instance, witness, ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{
        DiamondArtifactNames, DiamondWeConfig, default_preimage_max_coefficient_bound,
    };
    use keccak_asm::Keccak256;
    use mxx_gadgets::circuit::{BooleanGateData, BooleanGateKind};
    use mxx_ir_core::{RealExpr, artifact::SpecHash};
    use mxx_primitives::{
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::params::DCRTPolyParams,
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };
    use mxx_runtime::artifact::MemoryArtifactStore;
    use num_bigint::BigInt;
    use std::collections::BTreeSet;

    type TestRuntime = DiamondWeRuntime<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        MemoryArtifactStore,
    >;

    fn runtime() -> TestRuntime {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus();
        let trapdoor_sigma = RealExpr::from_f64_exact(4.578).unwrap();
        let gadget_base = BigInt::from(1u64 << parameters.base_bits());
        let preimage_max_coefficient_bound = default_preimage_max_coefficient_bound(
            &trapdoor_sigma,
            parameters.ring_dimension() as usize,
            parameters.modulus_digits(),
            &gadget_base,
        )
        .unwrap();
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: BigInt::from(modulus.as_ref().clone()),
                ring_dimension: parameters.ring_dimension() as usize,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base,
                digit_count: parameters.modulus_digits(),
                trapdoor_sigma,
                error_sigma: RealExpr::from_integer(0),
                error_max_coefficient_bound: 0.into(),
                preimage_max_coefficient_bound,
                bgg_tag: b"diamond-runtime-test".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 1,
                witness_width: 1,
                depth: 1,
                max_layer_width: 2,
            },
        )
        .unwrap();
        TestRuntime::new(compiler, parameters, MemoryArtifactStore::default()).unwrap()
    }

    fn and_xor_circuit(output_source: usize) -> BooleanCircuitData {
        BooleanCircuitData {
            layers: vec![vec![
                BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 },
                BooleanGateData { kind: BooleanGateKind::Xor, left: 0, right: 1 },
            ]],
            output_source,
        }
    }

    fn and_circuit() -> BooleanCircuitData {
        BooleanCircuitData {
            layers: vec![vec![BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 }]],
            output_source: 0,
        }
    }

    #[test]
    fn small_and_round_trip() {
        let circuit = and_circuit();
        let instance = [true];
        let mut runtime = runtime();
        let ciphertext =
            runtime.encrypt_with_hash_key(&circuit, &instance, true, [0x3f; 32]).unwrap();
        assert_eq!(runtime.decrypt(&circuit, &instance, &[true], &ciphertext).unwrap(), true);
    }

    #[test]
    fn and_xor_dynamic_outputs_round_trip_both_messages() {
        for (case, (circuit, instance)) in
            [(and_xor_circuit(0), vec![true]), (and_xor_circuit(1), vec![false])]
                .into_iter()
                .enumerate()
        {
            assert!(circuit.evaluate(&runtime().compiler.shape, &instance, &[true]).unwrap());
            for message in [false, true] {
                let mut runtime = runtime();
                let hash_key = [0x40 + (case as u8) * 2 + u8::from(message); 32];
                let ciphertext =
                    runtime.encrypt_with_hash_key(&circuit, &instance, message, hash_key).unwrap();
                assert_eq!(
                    runtime.decrypt(&circuit, &instance, &[true], &ciphertext).unwrap(),
                    message
                );
            }
        }
    }

    #[test]
    fn ciphertext_rejects_hash_key_and_production_graph_mismatches() {
        let circuit = and_xor_circuit(0);
        let instance = [true];
        let mut runtime = runtime();
        let ciphertext =
            runtime.encrypt_with_hash_key(&circuit, &instance, true, [0x52; 32]).unwrap();

        let mut wrong_hash_key = ciphertext.clone();
        wrong_hash_key.hash_key = [0x53; 32];
        assert!(matches!(
            runtime.decrypt(&circuit, &instance, &[true], &wrong_hash_key),
            Err(DiamondRuntimeError::ProductionNonceMismatch)
        ));

        let mut wrong_production = ciphertext;
        wrong_production.encryption.spec_hash = SpecHash([0x54; 32]);
        assert!(matches!(
            runtime.decrypt(&circuit, &instance, &[true], &wrong_production),
            Err(DiamondRuntimeError::ProductionGraphMismatch)
        ));
    }

    #[test]
    fn ciphertext_and_manifest_do_not_store_gate_rhs_decompositions() {
        let circuit = and_xor_circuit(0);
        let mut runtime = runtime();
        let ciphertext =
            runtime.encrypt_with_hash_key(&circuit, &[true], true, [0x61; 32]).unwrap();

        let DiamondWeCiphertext { hash_key: _, encryption } = ciphertext;
        let artifact_names = runtime
            .store
            .manifest(&encryption)
            .expect("encryption manifest")
            .artifacts
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            artifact_names,
            BTreeSet::from([
                DiamondArtifactNames::INITIAL_STATE,
                DiamondArtifactNames::ONE_PREIMAGE,
                DiamondArtifactNames::K_PREIMAGE,
                DiamondArtifactNames::DECODER_PREIMAGE,
                DiamondArtifactNames::R_DECOMPOSED,
                DiamondArtifactNames::TRANSITIONS,
                DiamondArtifactNames::WITNESS_PREIMAGES,
                DiamondArtifactNames::PUBLIC_KEYS,
            ])
        );
    }
}
