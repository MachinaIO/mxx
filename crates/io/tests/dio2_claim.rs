//! The correctness claim of a Diamond iO 2 evaluation, exported to Lean
//! (`mxx_io::dio2::claim`).
//!
//! The obfuscation and the evaluation run on an execution that builds and validates every stage
//! graph but executes only the payload secret's sampling ([`GraphOnly`]), so the export needs no
//! GPU: the claim is about the recorded graphs, which do not depend on the values. The parameters
//! and the function are those of the GPU round trip (`tests/gpu_round_trip.rs`); the evaluator's
//! chunks are `MXX_IO_DIO2_CLAIM_X` (one bit per step, default `0,1`), and the claim has the
//! failure probability `2^-MXX_IO_DIO2_CLAIM_FAILURE_EXPONENT` (default 40) under the
//! average-case model, or is about every execution when it is `worst`. `MXX_IO_DIO2_CLAIM_EXPORT`
//! names the directory of the Lean modules (none: the protocol is only built).
//! Ring/PRF overrides (same prefix): `RING_DIMENSION`, `PRIMES`, `CRT_BITS`, `BASE_BITS`,
//! `LEVELS`, `SMUDGE_DIGITS`, `PRF_NU`, `PRF_H`, `PRF_Q`, `PRF_P`, `PRF_BUCKET_WIDTH`,
//! `ERROR_SIGMA`, `TRAPDOOR_SIGMA`. Defaults retain the toy round trip. Production values are
//! `65536,27,60,6,38,306,1024,31,256,32,0,4,4`, respectively. This is graph-only execution,
//! not a noise/security check or an executable production-sized obfuscation.

use bigdecimal::BigDecimal;
use mxx_backends::{
    MemoryArtifactStore, RuntimeValue, authority::CpuExecution, backend::poly::cpu_backend,
    matrix::dcrt_poly::DCRTPolyMatrix, openfhe_guard::gen_modulus_and_warmup,
};
use mxx_exponent_lut::{
    ExpLayout,
    api::{ApiError, StageExecution, StageOutputs},
    refresh::PrfSettings,
};
use mxx_io::{
    Obfuscation,
    dio2::{Dio2, Dio2Function, Dio2Parameters, StreamingFunction},
};
use mxx_ir_core::{ValidatedGraph, artifact::ProductionId};
use num_bigint::BigInt;
use std::collections::BTreeMap;

fn read(name: &str, default: &str) -> String {
    std::env::var(format!("MXX_IO_DIO2_{name}")).unwrap_or_else(|_| default.to_owned())
}

/// The toy round trip, with explicit ring, chain, sampler and PRF overrides for graph export.
fn toy() -> (Dio2Parameters, StreamingFunction) {
    let crt_bits = read("CRT_BITS", "50").parse().unwrap();
    let ring_dimension = read("RING_DIMENSION", "8").parse().unwrap();
    let primes = gen_modulus_and_warmup(
        ring_dimension,
        read("PRIMES", "5").parse().unwrap(),
        crt_bits,
        None,
    )
    .expect("CRT basis");
    let parameters = Dio2Parameters {
        ring_dimension,
        crt_bits,
        base_bits: read("BASE_BITS", "12").parse().unwrap(),
        primes,
        levels: read("LEVELS", "6").parse().unwrap(),
        prf: PrfSettings {
            lwr_dimension: read("PRF_NU", "1").parse().unwrap(),
            support_weight: read("PRF_H", "1").parse().unwrap(),
            input_modulus: read("PRF_Q", "16").parse().unwrap(),
            output_modulus: read("PRF_P", "8").parse().unwrap(),
            bucket_width: read("PRF_BUCKET_WIDTH", "0").parse().unwrap(),
        },
        smudge_digits: read("SMUDGE_DIGITS", "1").parse().unwrap(),
        trapdoor_sigma: read("TRAPDOOR_SIGMA", "4").parse().unwrap(),
        encoding_error_sigma: read("ERROR_SIGMA", "0").parse::<BigDecimal>().unwrap(),
    };
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 });
    let function = [vec![1, 0, 1, 0], vec![0, 1, 1, 0]]
        .into_iter()
        .fold(function.clone(), |acc, table| acc.with_step_row(table).unwrap());
    (parameters, function)
}

/// An execution that runs only the payload secret's sampling, on the CPU. Every other graph is
/// validated by its caller and not executed: a producing stage returns the production its
/// execution would have (which depends only on the validated graph and the nonce), and an
/// evaluated stage returns the integer `0` for each of its outputs, so the stages after it can
/// build their graphs.
struct GraphOnly(CpuExecution);

impl StageExecution<MemoryArtifactStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: ValidatedGraph,
        _: BTreeMap<String, RuntimeValue>,
        _: &mut MemoryArtifactStore,
        execution_nonce: [u8; 32],
    ) -> Result<ProductionId, ApiError> {
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: ValidatedGraph,
        inputs: BTreeMap<String, RuntimeValue>,
        store: &mut MemoryArtifactStore,
        execution_nonce: [u8; 32],
    ) -> Result<StageOutputs, ApiError> {
        if validated.source.name() == "rns-bgg-secret" {
            return self.0.evaluate(validated, inputs, store, execution_nonce);
        }
        Ok(StageOutputs {
            outputs: validated
                .source
                .outputs()
                .keys()
                .map(|name| (name.clone(), RuntimeValue::Int(BigInt::from(0))))
                .collect(),
        })
    }

    fn download_matrix(&self, value: &RuntimeValue) -> Result<DCRTPolyMatrix, ApiError> {
        StageExecution::<MemoryArtifactStore>::download_matrix(&self.0, value)
    }

    fn download_integers(&self, value: &RuntimeValue) -> Result<Vec<BigInt>, ApiError> {
        StageExecution::<MemoryArtifactStore>::download_integers(&self.0, value)
    }
}

#[test]
fn test_dio2_claim_export() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let (parameters, function) = toy();
    let x =
        read("CLAIM_X", "0,1").split(',').map(|bit| vec![bit.trim() == "1"]).collect::<Vec<_>>();
    assert_eq!(x.len(), function.stream_length(), "one chunk per step");
    let failure = match read("CLAIM_FAILURE_EXPONENT", "40").as_str() {
        "worst" => None,
        exponent => Some(exponent.parse().unwrap()),
    };
    let execution = GraphOnly(CpuExecution::new(cpu_backend(parameters.rings().unwrap())));
    let mut dio2 = Dio2::new(parameters, execution, MemoryArtifactStore::default())
        .unwrap()
        .with_inputs_by_shape();
    let steps = function.stream_length();
    let y = (0..steps).map(|_| vec![false; function.obf_chunk_bits]).collect::<Vec<_>>();
    // A fixed hash key, so that every export writes the same modules.
    let obfuscation = dio2
        .obfuscate_with_hash_key(
            &Dio2Function { function: function.clone(), y, initial_state: 0 },
            *b"mxx-io/dio2/claim-export/key/v01",
        )
        .unwrap();
    dio2.evaluate(&obfuscation, &x).unwrap();
    let protocol = dio2.protocol(&obfuscation, &x, failure).unwrap();
    for requirement in &protocol.bundle.requirements {
        mxx_ir_core::validate(requirement.graph(), &protocol.bindings)
            .expect("claim requirement validates");
    }
    if let Ok(directory) = std::env::var("MXX_IO_DIO2_CLAIM_EXPORT") {
        std::fs::create_dir_all(&directory).unwrap();
        mxx_ir_core::lean::protocol::export(&protocol, std::path::Path::new(&directory), true)
            .unwrap();
        // The names of every stage's root inputs and outputs, in the order of its Lean tuples (the
        // proof generators read them).
        let quote = |name: &str| format!("\"{}\"", name.replace('\\', "\\\\").replace('"', "\\\""));
        let stages = protocol
            .bundle
            .workflow
            .stages
            .iter()
            .map(|stage| {
                let inputs = stage
                    .graph
                    .root_scope()
                    .nodes()
                    .iter()
                    .filter_map(|node| match node.kind() {
                        mxx_ir_core::node::NodeKind::Input { name, .. } => Some(quote(name)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let outputs =
                    stage.graph.outputs().keys().map(|name| quote(name)).collect::<Vec<_>>();
                format!(
                    "{}: {{\"inputs\": [{}], \"outputs\": [{}]}}",
                    quote(&stage.id.0),
                    inputs.join(", "),
                    outputs.join(", ")
                )
            })
            .collect::<Vec<_>>();
        std::fs::write(
            std::path::Path::new(&directory).join("stages.json"),
            format!("{{\n{}\n}}\n", stages.join(",\n")),
        )
        .unwrap();
    }
}
