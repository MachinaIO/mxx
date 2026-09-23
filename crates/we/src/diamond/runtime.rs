use super::{DiamondCompileError, DiamondConfigError, DiamondWeCompiler};
use crate::WitnessEncryptionRuntime;
#[cfg(feature = "gpu")]
use mxx_backends::{GpuExecutionResult, GpuRuntime};
use mxx_backends::{
    RuntimeValue, SessionStore, authority::ExecutionAuthority, executor::ExecutionResult,
};
use mxx_gadgets::circuit::{
    BOOLEAN_INSTANCE_INPUT, BOOLEAN_WITNESS_INPUT, BooleanCircuitData, BooleanCircuitError,
    BooleanCircuitShape,
};
use mxx_ir_core::{artifact::ProductionId, encoding::spec_hash};
use rand::random;
use std::{collections::BTreeMap, time::Instant};
use thiserror::Error;
use tracing::{debug, info};

use super::graph::{DECODED_OUTPUT, HASH_KEY_INPUT, MESSAGE_INPUT};

/// The execution authority deliberately chooses its concrete result type (CPU
/// execution and compiled GPU execution have different ownership metadata).
/// Diamond only needs the boolean value produced by its decryption graph, so
/// keep that result contract local to the protocol caller instead of exposing
/// executor-specific fields through `ExecutionAuthority`.
pub trait DiamondBooleanOutput<E> {
    fn boolean_output(&self, execution: &E, name: &str) -> Result<Option<bool>, String>;
}

impl<E> DiamondBooleanOutput<E> for ExecutionResult {
    fn boolean_output(&self, _execution: &E, name: &str) -> Result<Option<bool>, String> {
        Ok(match self.outputs.get(name) {
            Some(RuntimeValue::Bool(value)) => Some(*value),
            _ => None,
        })
    }
}

#[cfg(feature = "gpu")]
impl DiamondBooleanOutput<GpuRuntime> for GpuExecutionResult<'_> {
    fn boolean_output(&self, execution: &GpuRuntime, name: &str) -> Result<Option<bool>, String> {
        self.output(name)
            .map(|output| {
                execution.download_bool_output(&output).map_err(|error| error.to_string())
            })
            .transpose()
    }
}

#[derive(Clone)]
pub struct DiamondWeCiphertext {
    pub hash_key: [u8; 32],
    pub encryption: ProductionId,
}

pub struct DiamondWeRuntime<E, S>
where
    S: SessionStore,
    E: ExecutionAuthority<S>,
{
    pub compiler: DiamondWeCompiler,
    pub execution: E,
    pub store: S,
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
}

impl<E, S> DiamondWeRuntime<E, S>
where
    S: SessionStore,
    E: ExecutionAuthority<S>,
    for<'a> E::Result<'a>: DiamondBooleanOutput<E>,
{
    pub fn new(
        compiler: DiamondWeCompiler,
        execution: E,
        store: S,
    ) -> Result<Self, DiamondRuntimeError> {
        Ok(Self { compiler, execution, store })
    }

    pub fn encrypt(
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
            .validate(&bindings, mxx_backends::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond encryption graph"
        );
        let production_started = Instant::now();
        let production = ProductionId {
            spec_hash: spec_hash(&validated.source, &validated.bindings)
                .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?,
            execution_nonce: hash_key,
        };
        debug!(
            elapsed_seconds = production_started.elapsed().as_secs_f64(),
            "constructed Diamond encryption production identity"
        );
        let inputs_started = Instant::now();
        let mut inputs = circuit_inputs(circuit, &self.compiler.shape);
        insert_boolean_family_input(
            &mut inputs,
            BOOLEAN_INSTANCE_INPUT,
            instance,
            self.compiler.shape.analyze()?.maximum_layer_width,
        );
        inputs.insert(HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec().into()));
        inputs.insert(MESSAGE_INPUT.to_owned(), RuntimeValue::Bool(message));
        debug!(
            elapsed_seconds = inputs_started.elapsed().as_secs_f64(),
            "constructed Diamond encryption runtime inputs"
        );
        let execution_started = Instant::now();
        info!("starting Diamond encryption graph execution");
        let mut prepared =
            self.execution.prepare(validated, &inputs).map_err(DiamondRuntimeError::Execution)?;
        self.execution
            .run(&mut prepared, inputs, &mut self.store, hash_key)
            .map_err(DiamondRuntimeError::Execution)?;
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
            .validate(
                &self.compiler.circuit_bindings()?,
                mxx_backends::openfhe_guard::gen_modulus_and_warmup,
            )
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
        // Diamond decryption is a production consumer: it may only observe a
        // finalized session snapshot, never a standalone or in-progress
        // manifest that could still be mutated by its producer.
        let manifest = self
            .store
            .load_finalized_manifest(&ciphertext.encryption)
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
                mxx_backends::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| DiamondRuntimeError::Validation(error.to_string()))?;
        debug!(
            elapsed_seconds = validation_started.elapsed().as_secs_f64(),
            "validated Diamond decryption graph"
        );
        let inputs_started = Instant::now();
        let maximum_width = self.compiler.shape.analyze()?.maximum_layer_width;
        let mut inputs = circuit_inputs(circuit, &self.compiler.shape);
        insert_boolean_family_input(&mut inputs, BOOLEAN_INSTANCE_INPUT, instance, maximum_width);
        insert_boolean_family_input(&mut inputs, BOOLEAN_WITNESS_INPUT, witness, maximum_width);
        debug!(
            elapsed_seconds = inputs_started.elapsed().as_secs_f64(),
            "constructed Diamond decryption runtime inputs"
        );
        let execution_started = Instant::now();
        info!("starting Diamond decryption graph execution");
        let mut prepared =
            self.execution.prepare(validated, &inputs).map_err(DiamondRuntimeError::Execution)?;
        let result = self
            .execution
            .run(&mut prepared, inputs, &mut self.store, [0; 32])
            .map_err(DiamondRuntimeError::Execution)?;
        let decoded = result
            .boolean_output(&self.execution, DECODED_OUTPUT)
            .map_err(DiamondRuntimeError::Execution)?
            .ok_or(DiamondRuntimeError::DecodeOutput)?;
        info!(
            execution_elapsed_seconds = execution_started.elapsed().as_secs_f64(),
            total_elapsed_seconds = total_started.elapsed().as_secs_f64(),
            "finished Diamond decryption graph execution"
        );
        Ok(decoded)
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

fn circuit_inputs(
    circuit: &BooleanCircuitData,
    shape: &BooleanCircuitShape,
) -> BTreeMap<String, RuntimeValue> {
    let maximum_width = shape.analyze().expect("validated Boolean shape").maximum_layer_width;
    let family = RuntimeValue::integer_values;
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

fn insert_boolean_family_input(
    inputs: &mut BTreeMap<String, RuntimeValue>,
    name: &str,
    values: &[bool],
    maximum_width: usize,
) {
    let mut padded =
        values.iter().map(|value| num_bigint::BigInt::from(*value)).collect::<Vec<_>>();
    padded.resize(maximum_width, 0.into());
    inputs.insert(name.to_owned(), RuntimeValue::integer_values(padded));
}

impl<E, S> WitnessEncryptionRuntime for DiamondWeRuntime<E, S>
where
    S: SessionStore,
    E: ExecutionAuthority<S>,
    for<'a> E::Result<'a>: DiamondBooleanOutput<E>,
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
        DiamondWeRuntime::encrypt(self, circuit, instance, *message, random())
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
    use mxx_backends::{
        artifact::MemoryArtifactStore,
        backend::poly::cpu_backend,
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_gadgets::circuit::{BooleanGateData, BooleanGateKind};
    use mxx_ir_core::{RealExpr, artifact::SpecHash};
    use num_bigint::BigInt;
    use std::collections::BTreeSet;

    type TestRuntime = DiamondWeRuntime<mxx_backends::authority::CpuExecution, MemoryArtifactStore>;

    fn runtime() -> TestRuntime {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
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
                crt_moduli: parameters.to_crt().0,
                ring_dimension: parameters.ring_dimension(),
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
        TestRuntime::new(
            compiler,
            mxx_backends::authority::CpuExecution::new(cpu_backend([parameters])),
            MemoryArtifactStore::default(),
        )
        .unwrap()
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
        let ciphertext = runtime.encrypt(&circuit, &instance, true, [0x3f; 32]).unwrap();
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
                let ciphertext = runtime.encrypt(&circuit, &instance, message, hash_key).unwrap();
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
        let ciphertext = runtime.encrypt(&circuit, &instance, true, [0x52; 32]).unwrap();

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
        let ciphertext = runtime.encrypt(&circuit, &[true], true, [0x61; 32]).unwrap();

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
