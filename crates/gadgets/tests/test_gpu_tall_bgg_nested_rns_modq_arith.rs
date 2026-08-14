#![cfg(feature = "gpu")]

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    CostReport, EstimateConfig, estimate, gpu::GpuNodeMeasurementBackend,
    harness::MeasurementHarnessConfig,
};
use mxx_bgg::{
    BggPublicKeyCompiler, BggPublicKeySampler, BggPublicKeyWire, BggSamplerLayout,
    BggSlotTransferArtifactCompiler, BggSlotTransferGateArtifacts,
    BggSlotTransferPublicKeyLowering, BggSlotTransferPublicSlotWires, BggSlotTransferSlotArtifacts,
    BggTallEncodingCompiler, BggTallEncodingSampler, BggTallPlaintext, BggTallSlotLowering,
    BggTallSlotPublicKeyLowering, LweLookupPreprocessingLowering, LweLookupPublicKeyLowering,
    LweLookupTallEncodingLowering, PolyCircuitCompiler, TallRotationEncodingArtifacts,
    TallRotationEncodingCompiler, bind_lwe_lookup_invocations, required_tall_rotation_encodings,
};
use mxx_correctness::{
    ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
    EndpointSemanticBinding, EndpointSpecId, ExactMatrixInputMetadata, OperationalDecoderKind,
    OperationalDecoderTarget, OutputRef, StageId,
    operational_noise::{
        OperationalCheckRequest, OperationalGadgetLayout, OperationalSimulationReport,
        ProgressEventKind, check_operational_noise_candidate_with_progress,
    },
    operational_protocol_from_graphs,
};
use mxx_dsl::{
    Bool, BuiltGraph, DslContext, Family, IdealSpec, Ring, SemanticAnchor, parallel_zip,
};
use mxx_gadgets::{
    circuit::{PolyCircuit, PolyGateKind},
    circuit_gadgets::arith::nested_rns::{
        NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly_with_offset,
        minimum_p_moduli_bits,
    },
};
use mxx_ir_core::{
    IntExpr, ParamEnv, RealExpr,
    artifact::{
        ArtifactConfidentiality, Manifest as RuntimeManifest, ProductionId,
        export_validated_manifest, production_id,
    },
    encoding::spec_hash,
    node::{IndexRange, NodeKind},
};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::bounds::{compute_preimage_sigma, hard_cutoff_from_sigma_bound},
    utils::gen_biguint_for_modulus,
};
use mxx_runtime::{
    ExecutionConfig, ExecutionResult, PreimageProgressConfig, RuntimeValue,
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore, MemoryArtifactStore},
    backend::poly::gpu::{GpuDcrtBackend, gpu_backend_on},
    execute_in_session_with_config, execute_with_config,
    transcript::SamplingMode,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    num::NonZeroUsize,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::{debug, error, info};

const HASH_KEY_INPUT: &str = "tall_nested_rns_hash_key";
const SECRET_ARTIFACT: &str = "tall_nested_rns_secret";
const C_B0_ARTIFACT: &str = "tall_nested_rns_c_b0";
const LOOKUP_C_B_ARTIFACT: &str = "tall_nested_rns_lookup_c_b";
const INPUT_PUBLIC_KEY_PREFIX: &str = "tall_nested_rns_input_public_key";
const TALL_OPERATIONAL_TARGET_ID: &str = "tall-threshold-decode";
const TALL_OPERATIONAL_RESIDUAL: &str = "operational_residual";
const TALL_OPERATIONAL_DECODED: &str = "operational_decoded";
const TALL_DECODER_RESIDUAL_ANCHOR: &str = "tall.decoder.residual";
const TALL_DECODER_RESULT_ANCHOR: &str = "tall.decoder.result";

#[derive(Clone, Debug)]
struct TestConfig {
    mul_count: usize,
    min_crt_depth: usize,
    max_crt_depth: usize,
    min_log_ring_dimension: usize,
    max_log_ring_dimension: usize,
    selected_parameters: Option<(usize, usize)>,
    security_bits: u64,
    crt_modulus_bits: usize,
    gadget_base_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    error_sigma: f64,
    trapdoor_sigma: f64,
    benchmark_warmups: usize,
    benchmark_iterations: usize,
    stop_after_benchmark_estimation: bool,
    parameter_simulation_parallelism: usize,
    preimage_progress_interval: usize,
    max_parallel_instances: usize,
    preprocessing_parallel_instances: usize,
    release_fence_interval: usize,
    checkpoint_root: PathBuf,
    reuse_checkpoint: Option<PathBuf>,
}

impl TestConfig {
    fn from_env() -> Result<Self, String> {
        // For n = 8, 5-bit CRT moduli support only one NTT level. Ten bits is the smallest
        // practical basis width for the multi-level search below. Keeping the base at half that
        // width minimizes the gadget representation without an independent tuning knob; the
        // nested-RNS p basis is selected from the concrete q basis below.
        let crt_modulus_bits = env_usize("MXX_TALL_NESTED_RNS_CRT_MODULUS_BITS", 10)?;
        let selected_crt_depth = env_optional_usize("MXX_TALL_NESTED_RNS_SELECTED_CRT_DEPTH")?;
        let selected_log_ring_dimension =
            env_optional_usize("MXX_TALL_NESTED_RNS_SELECTED_LOG_RING_DIMENSION")?;
        let selected_parameters = match (selected_crt_depth, selected_log_ring_dimension) {
            (None, None) => None,
            (Some(crt_depth), Some(log_ring_dimension)) => Some((crt_depth, log_ring_dimension)),
            _ => {
                return Err(
                    "MXX_TALL_NESTED_RNS_SELECTED_CRT_DEPTH and MXX_TALL_NESTED_RNS_SELECTED_LOG_RING_DIMENSION must be supplied together"
                        .to_owned(),
                );
            }
        };
        let config = Self {
            mul_count: env_usize("MXX_TALL_NESTED_RNS_MUL_COUNT", 1)?,
            min_crt_depth: env_usize("MXX_TALL_NESTED_RNS_MIN_CRT_DEPTH", 1)?,
            max_crt_depth: env_usize("MXX_TALL_NESTED_RNS_MAX_CRT_DEPTH", 16)?,
            min_log_ring_dimension: env_usize("MXX_TALL_NESTED_RNS_MIN_LOG_RING_DIMENSION", 3)?,
            max_log_ring_dimension: env_usize("MXX_TALL_NESTED_RNS_MAX_LOG_RING_DIMENSION", 3)?,
            selected_parameters,
            // n = 8 is intentionally an execution smoke parameter and has no positive lattice
            // security estimate. A caller may request a positive target, which will reject it.
            security_bits: env_u64("MXX_TALL_NESTED_RNS_SECURITY_BITS", 0)?,
            crt_modulus_bits,
            gadget_base_bits: env_usize(
                "MXX_TALL_NESTED_RNS_GADGET_BASE_BITS",
                crt_modulus_bits.div_ceil(2),
            )?,
            // The multiplication's full-reduce intermediate exceeds the one-product p basis;
            // retain the two-product budget required by the nested-RNS bound check.
            max_unreduced_muls: env_usize("MXX_TALL_NESTED_RNS_MAX_UNREDUCED_MULS", 2)?,
            scale: env_u64("MXX_TALL_NESTED_RNS_SCALE", 1 << 10)?,
            error_sigma: env_f64("MXX_TALL_NESTED_RNS_ERROR_SIGMA", 1.0)?,
            trapdoor_sigma: env_f64("MXX_TALL_NESTED_RNS_TRAPDOOR_SIGMA", 4.578)?,
            benchmark_warmups: env_usize("MXX_TALL_NESTED_RNS_BENCH_WARMUPS", 1)?,
            benchmark_iterations: env_usize("MXX_TALL_NESTED_RNS_BENCH_ITERATIONS", 2)?,
            stop_after_benchmark_estimation: env_bool(
                "MXX_TALL_NESTED_RNS_STOP_AFTER_BENCHMARK_ESTIMATION",
                false,
            )?,
            // Operational checking of the generated workflow is CPU-only and independent across
            // parameter candidates. Keep the default batch small because each checker can use
            // tens of GiB for large LUT graphs.
            parameter_simulation_parallelism: env_usize(
                "MXX_TALL_NESTED_RNS_PARAMETER_SIMULATION_PARALLELISM",
                2,
            )?,
            // Report exact runtime sampler completion frequently enough to make the long
            // preprocessing phase observable without emitting one line per preimage.
            preimage_progress_interval: env_usize(
                "MXX_TALL_NESTED_RNS_PREIMAGE_PROGRESS_INTERVAL",
                32,
            )?,
            // The nested-RNS LUT preprocessing has large artifact families. The GPU DCRT
            // allocator exhausts its live-buffer budget at two concurrent instances, so keep
            // one instance in flight by default.
            max_parallel_instances: env_usize("MXX_TALL_NESTED_RNS_MAX_PARALLEL_INSTANCES", 1)?,
            // Preprocessing only produces serialized artifacts, so it can safely use the
            // established two-instance GPU batch while consumer evaluation remains at its
            // tighter live-buffer limit above.
            preprocessing_parallel_instances: env_usize(
                "MXX_TALL_NESTED_RNS_PREPROCESSING_PARALLEL_INSTANCES",
                2,
            )?,
            // This fences context-owned release streams only after substantial graph work.
            // It bounds queued frees without waiting unrelated live matrices.
            release_fence_interval: env_usize("MXX_TALL_NESTED_RNS_RELEASE_FENCE_INTERVAL", 1024)?,
            checkpoint_root: env::var_os("MXX_TALL_NESTED_RNS_CHECKPOINT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("test_data/tall_nested_rns_gpu")),
            reuse_checkpoint: env::var_os("MXX_TALL_NESTED_RNS_REUSE_CHECKPOINT")
                .map(PathBuf::from),
        };
        if config.min_crt_depth == 0 ||
            config.min_crt_depth > config.max_crt_depth ||
            config.min_log_ring_dimension > config.max_log_ring_dimension ||
            u32::try_from(config.max_log_ring_dimension).is_err() ||
            1u32.checked_shl(config.max_log_ring_dimension as u32).is_none() ||
            config.crt_modulus_bits == 0 ||
            config.gadget_base_bits == 0 ||
            config.max_unreduced_muls == 0 ||
            config.scale == 0 ||
            config.error_sigma <= 0.0 ||
            !config.error_sigma.is_finite() ||
            config.trapdoor_sigma <= 0.0 ||
            !config.trapdoor_sigma.is_finite() ||
            config.benchmark_iterations == 0 ||
            config.parameter_simulation_parallelism == 0 ||
            config.preimage_progress_interval == 0 ||
            config.max_parallel_instances == 0 ||
            config.preprocessing_parallel_instances == 0 ||
            config.release_fence_interval == 0
        {
            return Err("invalid Tall nested-RNS GPU test configuration".to_owned());
        }
        if let Some((crt_depth, log_ring_dimension)) = config.selected_parameters &&
            (crt_depth == 0 ||
                u32::try_from(log_ring_dimension).is_err() ||
                1u32.checked_shl(log_ring_dimension as u32).is_none())
        {
            return Err("invalid selected Tall nested-RNS parameters".to_owned());
        }
        Ok(config)
    }

    fn candidate_dimensions(&self) -> Vec<(usize, usize)> {
        candidate_dimensions(
            self.min_crt_depth,
            self.max_crt_depth,
            self.min_log_ring_dimension,
            self.max_log_ring_dimension,
            self.selected_parameters,
        )
    }
}

fn candidate_dimensions(
    min_crt_depth: usize,
    max_crt_depth: usize,
    min_log_ring_dimension: usize,
    max_log_ring_dimension: usize,
    selected: Option<(usize, usize)>,
) -> Vec<(usize, usize)> {
    if let Some(selected) = selected {
        vec![selected]
    } else {
        (min_crt_depth..=max_crt_depth)
            .flat_map(|crt_depth| {
                (min_log_ring_dimension..=max_log_ring_dimension)
                    .map(move |log_ring_dimension| (crt_depth, log_ring_dimension))
            })
            .collect()
    }
}

struct CircuitBundle {
    circuit: PolyCircuit<DCRTPoly>,
    nested: Arc<NestedRnsPolyContext>,
}

struct PreparedCandidate {
    parameters: DCRTPolyParams,
    circuit: PolyCircuit<DCRTPoly>,
    nested: Arc<NestedRnsPolyContext>,
    layout: BggSamplerLayout,
    artifact_compiler: BggSlotTransferArtifactCompiler,
    producer: BuiltGraph,
    preprocessing_graph_construction: Duration,
    lookup_preimage_count: usize,
    slot_preimage_count: usize,
    preprocessing_preimage_count: usize,
    production: ProductionId,
    runtime_manifest: RuntimeManifest,
    lookup_compilers: Vec<mxx_bgg::LweLookupCompiler>,
    slot_requests: Vec<mxx_bgg::BggSlotTransferGateRequest>,
    rotation_offsets: Vec<u32>,
    public_key_graph: BuiltGraph,
    encoding_graph: BuiltGraph,
    operational_noise_bound: BigUint,
    operational_report: OperationalSimulationReport,
    achieved_security_bits: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PreprocessingCheckpoint {
    hash_key: [u8; 32],
    manifest: RuntimeManifest,
    payloads: Vec<(ArtifactKey, ArtifactPayload)>,
}

struct EndToEndOutputs {
    public_key: GpuDCRTPolyMatrix,
    encoding_public_key: GpuDCRTPolyMatrix,
    encoding_rows: Vec<GpuDCRTPolyMatrix>,
    output_plaintexts: Vec<GpuDCRTPolyMatrix>,
    transformed_secrets: Vec<GpuDCRTPolyMatrix>,
    expected_evaluation_slots: Vec<BigUint>,
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    env::var(name).map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("{name} must be a nonnegative integer"))
    })
}

fn env_u64(name: &str, default: u64) -> Result<u64, String> {
    env::var(name).map_or(Ok(default), |value| {
        value.parse().map_err(|_| format!("{name} must be a nonnegative integer"))
    })
}

fn env_optional_usize(name: &str) -> Result<Option<usize>, String> {
    env::var(name).map_or(Ok(None), |value| {
        if value.is_empty() {
            return Err(format!("{name} must not be empty"));
        }
        value.parse().map(Some).map_err(|_| format!("{name} must be a nonnegative integer"))
    })
}

fn env_bool(name: &str, default: bool) -> Result<bool, String> {
    env::var(name).map_or(Ok(default), |value| match value.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("{name} must be 0 or 1")),
    })
}

fn env_f64(name: &str, default: f64) -> Result<f64, String> {
    env::var(name)
        .map_or(Ok(default), |value| value.parse().map_err(|_| format!("{name} must be a number")))
}

fn build_modq_multiplication_circuit(
    parameters: &DCRTPolyParams,
    config: &TestConfig,
    evaluation_slots: usize,
    p_modulus_bits: usize,
) -> CircuitBundle {
    let mut circuit = PolyCircuit::new();
    let nested = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        parameters,
        p_modulus_bits,
        config.max_unreduced_muls,
        config.scale,
        false,
        None,
    ));
    let inputs = (0..=config.mul_count)
        .map(|_| {
            NestedRnsPoly::input(
                nested.clone(),
                evaluation_slots,
                Some(nested.q_moduli_depth),
                Some(0),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let mut inputs = inputs.into_iter();
    let first = inputs.next().expect("mul_count + 1 is positive");
    let product = inputs.fold(first, |left, right| left.mul(&right, &mut circuit));
    let output = product.reconstruct(&mut circuit);
    circuit.output([output]);
    CircuitBundle { circuit, nested }
}

fn gate_kind_counts(circuit: &PolyCircuit<DCRTPoly>) -> HashMap<PolyGateKind, usize> {
    let mut counts = HashMap::new();
    for (_, gate) in circuit.gates_in_id_order() {
        *counts.entry(gate.gate_type.kind()).or_default() += 1;
    }
    counts
}

fn run_tall_operational_check(
    producer: &BuiltGraph,
    encoding: &BuiltGraph,
    parameters: &DCRTPolyParams,
    nested: &NestedRnsPolyContext,
) -> Result<OperationalSimulationReport, String> {
    if nested.p_moduli.is_empty() {
        return Err("nested-RNS plaintext contract requires a nonempty p-basis".to_owned());
    }
    let graph_counts = |graph: &mxx_ir_core::Graph| {
        (
            graph.scopes().len(),
            graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>(),
            graph.outputs().len(),
        )
    };
    let (producer_scopes, producer_nodes, producer_outputs) = graph_counts(&producer.graph);
    let (encoding_scopes, encoding_nodes, encoding_outputs) = graph_counts(&encoding.graph);
    info!(
        ring_dimension = parameters.ring_dimension(),
        crt_depth = parameters.to_crt().2,
        producer_scopes,
        producer_nodes,
        producer_outputs,
        encoding_scopes,
        encoding_nodes,
        encoding_outputs,
        "constructed Tall operational checker graphs"
    );
    let mut exact_input_metadata = BTreeMap::new();
    for node in encoding.graph.root_scope().nodes() {
        let NodeKind::Input { name, .. } = node.kind() else { continue };
        let Some(indices) = name.strip_prefix("plaintext_") else { continue };
        let Some((input_index, _slot_index)) = indices.split_once('_') else { continue };
        let input_index = input_index
            .parse::<usize>()
            .map_err(|_| format!("invalid nested-RNS plaintext input name {name}"))?;
        let modulus = nested
            .p_moduli
            .get(input_index % nested.p_moduli.len())
            .ok_or_else(|| "nested-RNS plaintext input has no p-modulus".to_owned())?;
        exact_input_metadata.insert(
            name.clone(),
            ExactMatrixInputMetadata {
                canonical_coefficient_exclusive_upper_bound: Some(IntExpr::constant(*modulus)),
                is_constant_polynomial: true,
            },
        );
    }
    let (crt_moduli, crt_bits, _) = parameters.to_crt();
    let q_max = crt_moduli.iter().copied().max().expect("nonempty CRT basis");
    let decoder_stage = StageId("encoding".to_owned());
    let decoder_node = encoding
        .graph
        .outputs()
        .get(TALL_OPERATIONAL_DECODED)
        .ok_or_else(|| "Tall operational decoder output is absent".to_owned())?
        .value
        .node;
    let endpoint = EndpointSpecId::ToyThresholdDecode;
    let ideal = IdealSpec::new(
        DslContext::new("gpu-tall-nested-rns-operational-ideal")
            .bool_output(TALL_OPERATIONAL_DECODED, Bool::constant(false))
            .map_err(|error| error.to_string())?
            .build()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let protocol = operational_protocol_from_graphs(
        vec![("producer".to_owned(), producer), ("encoding".to_owned(), encoding)],
        "encoding",
        &exact_input_metadata,
        &BTreeMap::new(),
        |bundle| {
            bundle.ideal = ideal;
            bundle.comparator = ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: TALL_OPERATIONAL_DECODED.to_owned(),
                    ideal_input: TALL_OPERATIONAL_DECODED.to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            };
            bundle.endpoints = EndpointAnchors {
                entries: vec![EndpointAnchor {
                    spec: endpoint,
                    stage: decoder_stage.clone(),
                    semantic_anchor: TALL_DECODER_RESULT_ANCHOR.to_owned(),
                    semantics: EndpointSemanticBinding::ThresholdDecode,
                    workflow_output: OutputRef {
                        stage: decoder_stage.clone(),
                        output: TALL_OPERATIONAL_DECODED.to_owned(),
                    },
                    ideal_output: TALL_OPERATIONAL_DECODED.to_owned(),
                }],
            };
            bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                target_id: TALL_OPERATIONAL_TARGET_ID.to_owned(),
                residual_stage: decoder_stage.clone(),
                residual_output: TALL_OPERATIONAL_RESIDUAL.to_owned(),
                decoder_stage,
                decoder_node,
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(q_max),
                },
            }];
            bundle.endpoint_specs = vec![endpoint];
        },
    )
    .map_err(|error| error.to_string())?;
    let base_bits = parameters.base_bits() as usize;
    let small_digit_count = crt_bits.div_ceil(base_bits);
    let request = OperationalCheckRequest {
        environment: Vec::new(),
        layouts: vec![OperationalGadgetLayout {
            params_id: "tall-nested-rns".to_owned(),
            ring_dimension: parameters.ring_dimension() as usize,
            smallest_crt_modulus: *crt_moduli.iter().min().expect("nonempty CRT basis"),
            regular_digit_count: small_digit_count * crt_moduli.len(),
            small_digit_count,
            base: BigInt::from(1u8) << base_bits,
            base_bits,
            crt_bits,
            crt_moduli,
        }],
        target_id: TALL_OPERATIONAL_TARGET_ID.to_owned(),
    };
    let evaluation_started = Instant::now();
    info!(target = request.target_id, "begin Tall operational noise checker");
    let report = check_operational_noise_candidate_with_progress(&protocol, &request, |event| {
        if event.event == ProgressEventKind::Progress {
            debug!(
                phase = ?event.phase,
                event = ?event.event,
                elapsed_ms = event.elapsed_ms,
                processed = event.processed,
                total_or_discovered = ?event.total_or_discovered,
                owned_elements = event.owned_elements,
                egraph_nodes = ?event.egraph_nodes,
                rewrite_iterations = event.rewrite_iterations,
                program = ?event.program,
                scope = ?event.scope,
                node = ?event.node,
                "Tall operational checker progress"
            );
        } else if event.event != ProgressEventKind::Progress {
            info!(
                phase = ?event.phase,
                event = ?event.event,
                elapsed_ms = event.elapsed_ms,
                processed = event.processed,
                owned_elements = event.owned_elements,
                egraph_nodes = ?event.egraph_nodes,
                rewrite_iterations = event.rewrite_iterations,
                program = ?event.program,
                scope = ?event.scope,
                node = ?event.node,
                "Tall operational checker phase summary"
            );
        }
    })
    .map_err(|simulation_error| {
        error!(
            elapsed = ?evaluation_started.elapsed(),
            error = %simulation_error,
            "Tall operational noise checker failed"
        );
        simulation_error.to_string()
    })?;
    info!(
        elapsed = ?evaluation_started.elapsed(),
        accepted = report.accepted,
        noise_bound = %report.noise_bound,
        "evaluated Tall parameter request with Rust operational checker"
    );
    Ok(report)
}

fn prepare_candidate(
    parameters: DCRTPolyParams,
    config: &TestConfig,
    achieved_security_bits: u64,
) -> Result<PreparedCandidate, String> {
    let ring_dimension = parameters.ring_dimension() as usize;
    let (q_moduli, _, _) = parameters.to_crt();
    let p_modulus_bits = minimum_p_moduli_bits(
        *q_moduli.iter().max().expect("CRT basis is nonempty"),
        config.max_unreduced_muls,
    )
    .ok_or_else(|| "no nested-RNS p-modulus basis supports the selected q basis".to_owned())?;
    let CircuitBundle { circuit, nested } =
        build_modq_multiplication_circuit(&parameters, config, ring_dimension, p_modulus_bits);
    let scalar_circuit = build_modq_multiplication_circuit(&parameters, config, 1, p_modulus_bits);
    if gate_kind_counts(&circuit) != gate_kind_counts(&scalar_circuit.circuit) {
        return Err("nested-RNS gate counts depend on the coefficient slot count".to_owned());
    }
    let preprocessing_graph_started = Instant::now();
    let physical_slots = ring_dimension * nested.q_moduli_depth;
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let gadget_base = BigInt::from(1u8) << config.gadget_base_bits;
    let error_sigma_decimal = BigDecimal::from_f64(config.error_sigma)
        .ok_or_else(|| "error sigma must be finite".to_owned())?;
    let error_max_coefficient_bound =
        BigInt::from(hard_cutoff_from_sigma_bound(&error_sigma_decimal));
    let preimage_sigma = compute_preimage_sigma(
        &BigDecimal::from(ring_dimension as u64).sqrt().expect("positive ring dimension"),
        u64::try_from(parameters.modulus_digits())
            .map_err(|_| "digit count exceeds u64".to_owned())?,
        &BigDecimal::from(gadget_base.clone()),
        Some(1),
        Some(config.trapdoor_sigma),
    );
    let preimage_max_coefficient_bound =
        BigInt::from(hard_cutoff_from_sigma_bound(&preimage_sigma));
    let layout = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_dimension: 1,
        digit_count: parameters.modulus_digits(),
        gadget_base: gadget_base.clone().into(),
    };
    let ring = layout.ring();
    let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
    let secret = ring.uniform_interval((1, 1), -1, 1);
    let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
        hash_key.clone(),
        b"tall-nested-rns-input-public-keys".as_slice(),
        &vec![true; circuit.num_input()],
    );
    let public_key_compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: gadget_base.clone().into(),
        digit_count: parameters.modulus_digits().into(),
    };
    let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
    let artifact_compiler = BggSlotTransferArtifactCompiler {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_size: 1,
        slot_count: physical_slots,
        digit_count: parameters.modulus_digits(),
        chunk_columns: parameters.modulus_digits().max(1),
        gadget_base: gadget_base.clone().into(),
        trapdoor_sigma: RealExpr::from_f64_exact(config.trapdoor_sigma)
            .map_err(|error| error.to_string())?,
        error_sigma: RealExpr::from_f64_exact(config.error_sigma)
            .map_err(|error| error.to_string())?,
        preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone().into(),
        error_max_coefficient_bound: error_max_coefficient_bound.clone().into(),
    };
    let base = artifact_compiler.build_base().map_err(|error| error.to_string())?;
    let slots = artifact_compiler
        .build_slots(hash_key.clone(), &base)
        .map_err(|error| error.to_string())?;
    let mut lookup_preprocessing = LweLookupPreprocessingLowering::new(
        parameters.clone(),
        hash_key.clone(),
        base.b0.clone(),
        gadget_base.clone().into(),
        parameters.modulus_digits().into(),
        Vec::new(),
    );
    let mut slot_preprocessing = BggTallSlotPublicKeyLowering {
        inner: BggSlotTransferPublicKeyLowering {
            compiler: public_key_compiler.clone(),
            hash_key: hash_key.clone(),
            public_key_type: ring.matrix_type((1, parameters.modulus_digits())),
            configured_slot_count: physical_slots,
            output_public_key_production: None,
            requests: Vec::new(),
        },
    };
    circuit_compiler
        .compile_public_keys_with_lowerings(
            &circuit,
            public_keys[0].clone(),
            public_keys.iter().skip(1).cloned(),
            &mut lookup_preprocessing,
            &mut slot_preprocessing,
        )
        .map_err(|error| error.to_string())?;
    let lookup_entries = lookup_preprocessing.into_entries();
    let mut lookup_preimage_start = 0usize;
    for (lookup_index, entry) in lookup_entries.iter().enumerate() {
        let table_length = entry.compiler.preprocessing_row_count();
        let lookup_preimage_end = lookup_preimage_start.saturating_add(table_length);
        info!(
            lookup_index,
            table_length,
            preimage_start = lookup_preimage_start,
            preimage_end_exclusive = lookup_preimage_end,
            "lookup preprocessing preimage range"
        );
        lookup_preimage_start = lookup_preimage_end;
    }
    let lookup_preimage_count =
        lookup_entries.iter().map(|entry| entry.compiler.preprocessing_row_count()).sum::<usize>();
    let lookup_compilers =
        lookup_entries.iter().map(|entry| entry.compiler.clone()).collect::<Vec<_>>();
    let slot_requests = slot_preprocessing.inner.requests;
    let slot_preimage_count = artifact_compiler
        .preprocessing_preimage_count(&slot_requests)
        .map_err(|error| error.to_string())?;
    let preprocessing_preimage_count = lookup_preimage_count + slot_preimage_count;
    info!(
        lookup_tables = lookup_entries.len(),
        lookup_preimage_count,
        slot_preimage_count,
        total_preimage_count = preprocessing_preimage_count,
        "preprocessing preimage plan"
    );
    let gate_wires = artifact_compiler
        .build_gate_preimages(&base, &slots, &slot_requests)
        .map_err(|error| error.to_string())?;
    let transformed_secrets = slots
        .secrets
        .clone()
        .parallel_map({
            let secret = secret.clone();
            move |_, transform| secret.clone() * transform
        })
        .map_err(|error| error.to_string())?;
    let error_sigma =
        RealExpr::from_f64_exact(config.error_sigma).map_err(|error| error.to_string())?;
    let b0_columns = 1usize
        .checked_mul(parameters.modulus_digits() + 2)
        .ok_or_else(|| "B0 column count overflow".to_owned())?;
    let c_b0 = secret.clone() * base.b0.public_matrix() +
        ring.gaussian((1, b0_columns), error_sigma.clone(), error_max_coefficient_bound.clone());
    let lookup_c_b = transformed_secrets
        .clone()
        .parallel_map({
            let ring = ring.clone();
            let b0 = base.b0.public_matrix();
            let error_max_coefficient_bound = error_max_coefficient_bound.clone();
            move |_, slot_secret| {
                slot_secret * b0.clone() +
                    ring.gaussian(
                        (1, b0_columns),
                        error_sigma.clone(),
                        error_max_coefficient_bound.clone(),
                    )
            }
        })
        .map_err(|error| error.to_string())?;
    let rotation_keys =
        required_tall_rotation_encodings(&circuit).map_err(|error| error.to_string())?;
    let rotation_offsets = rotation_keys.iter().map(|key| key.offset).collect::<Vec<_>>();
    let rotation_compiler = TallRotationEncodingCompiler {
        modulus: modulus.clone().into(),
        ring_dimension: ring_dimension.into(),
        secret_size: 1,
        slot_count: physical_slots,
        gadget_base: gadget_base.clone().into(),
        digit_count: parameters.modulus_digits(),
        error_sigma: RealExpr::from_f64_exact(config.error_sigma)
            .map_err(|error| error.to_string())?,
        error_max_coefficient_bound: error_max_coefficient_bound.clone().into(),
    };
    let rotations = rotation_compiler
        .preprocess(hash_key.clone(), secret.clone(), slots.secrets.clone(), &rotation_offsets)
        .map_err(|error| error.to_string())?;

    let mut producer_context = DslContext::new("gpu-tall-nested-rns-preprocessing")
        .private_output(SECRET_ARTIFACT, secret)
        .map_err(|error| error.to_string())?
        .public_output(C_B0_ARTIFACT, c_b0)
        .map_err(|error| error.to_string())?
        .public_family_output(LOOKUP_C_B_ARTIFACT, lookup_c_b)
        .map_err(|error| error.to_string())?;
    for (index, public_key) in public_keys.iter().enumerate() {
        producer_context = producer_context
            .public_output(format!("{INPUT_PUBLIC_KEY_PREFIX}_{index}"), public_key.matrix.clone())
            .map_err(|error| error.to_string())?;
    }
    producer_context = artifact_compiler
        .export_slots(producer_context, slots)
        .map_err(|error| error.to_string())?;
    producer_context = artifact_compiler
        .export_gate_public_keys(producer_context, &slot_requests)
        .map_err(|error| error.to_string())?;
    producer_context = artifact_compiler
        .export_gate_preimages(producer_context, gate_wires)
        .map_err(|error| error.to_string())?;
    for entry in &lookup_entries {
        producer_context = entry.export(producer_context).map_err(|error| error.to_string())?;
    }
    producer_context = rotation_compiler
        .export_preprocessing(producer_context, rotations)
        .map_err(|error| error.to_string())?;
    let producer = producer_context.build().map_err(|error| error.to_string())?;
    let preprocessing_graph_construction = preprocessing_graph_started.elapsed();
    let bindings = ParamEnv::default();
    let validated_producer = producer.validate(&bindings).map_err(|error| error.to_string())?;
    let production = production_id(
        spec_hash(&producer.graph, &bindings).map_err(|error| error.to_string())?,
        [0x71; 32],
    );
    let runtime_manifest = export_validated_manifest(production.clone(), &validated_producer)
        .map_err(|error| error.to_string())?;
    info!(
        ring_dimension,
        crt_depth = parameters.to_crt().2,
        p_modulus_bits,
        physical_slots,
        gate_counts = ?gate_kind_counts(&circuit),
        artifact_count = runtime_manifest.artifacts.len(),
        "constructed Tall nested-RNS candidate graphs"
    );
    debug!(q_moduli = ?parameters.to_crt().0, "candidate CRT moduli");
    let public_key_graph = build_public_key_graph(
        &parameters,
        &circuit,
        &layout,
        &artifact_compiler,
        production.clone(),
        &lookup_compilers,
    )?;
    let encoding_graph = build_encoding_graph(
        &parameters,
        &circuit,
        &layout,
        &artifact_compiler,
        production.clone(),
        &lookup_compilers,
        &slot_requests,
        &rotation_offsets,
        physical_slots,
        config.error_sigma,
        false,
    )?;
    let operational_encoding_graph = build_encoding_graph(
        &parameters,
        &circuit,
        &layout,
        &artifact_compiler,
        production.clone(),
        &lookup_compilers,
        &slot_requests,
        &rotation_offsets,
        physical_slots,
        config.error_sigma,
        true,
    )?;
    for output in
        ["encoding_rows", "encoding_public_key", "output_plaintexts", "transformed_secrets"]
    {
        if encoding_graph.graph.outputs().get(output) !=
            operational_encoding_graph.graph.outputs().get(output)
        {
            return Err(format!(
                "operational residual suffix changed executable output identity {output}"
            ));
        }
    }
    let manifests = BTreeMap::from([(production.clone(), runtime_manifest.clone())]);
    public_key_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    encoding_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    operational_encoding_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    let operational_report =
        run_tall_operational_check(&producer, &operational_encoding_graph, &parameters, &nested)?;
    let operational_noise_bound = operational_report.noise_bound.clone();
    Ok(PreparedCandidate {
        parameters,
        circuit,
        nested,
        layout,
        artifact_compiler,
        producer,
        preprocessing_graph_construction,
        lookup_preimage_count,
        slot_preimage_count,
        preprocessing_preimage_count,
        production,
        runtime_manifest,
        lookup_compilers,
        slot_requests,
        rotation_offsets,
        public_key_graph,
        encoding_graph,
        operational_noise_bound,
        operational_report,
        achieved_security_bits,
    })
}

fn imported_public_keys(
    ring: &Ring,
    production: &ProductionId,
    circuit_input_count: usize,
    columns: usize,
) -> Vec<BggPublicKeyWire> {
    (0..=circuit_input_count)
        .map(|index| BggPublicKeyWire {
            matrix: ring.artifact_input(
                production.clone(),
                format!("{INPUT_PUBLIC_KEY_PREFIX}_{index}"),
                (1, columns),
                ArtifactConfidentiality::Public,
            ),
            reveal_plaintext: true,
        })
        .collect()
}

fn build_public_key_graph(
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    layout: &BggSamplerLayout,
    _artifact_compiler: &BggSlotTransferArtifactCompiler,
    production: ProductionId,
    lookup_compilers: &[mxx_bgg::LweLookupCompiler],
) -> Result<BuiltGraph, String> {
    let ring = layout.ring();
    let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
    let public_key_compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: layout.gadget_base.clone(),
        digit_count: layout.digit_count.into(),
    };
    let public_keys =
        imported_public_keys(&ring, &production, circuit.num_input(), layout.public_key_columns());
    let invocations = bind_lwe_lookup_invocations(
        parameters,
        circuit,
        production.clone(),
        lookup_compilers.iter().cloned(),
    )
    .map_err(|error| error.to_string())?;
    let mut lookup =
        LweLookupPublicKeyLowering::new(invocations).map_err(|error| error.to_string())?;
    let mut slots = BggTallSlotPublicKeyLowering {
        inner: BggSlotTransferPublicKeyLowering {
            compiler: public_key_compiler.clone(),
            hash_key,
            public_key_type: ring.matrix_type((1, layout.public_key_columns())),
            configured_slot_count: layout
                .ring_dimension
                .evaluate(&ParamEnv::default())
                .map_err(|error| error.to_string())?
                .to_usize()
                .ok_or_else(|| "ring dimension does not fit usize".to_owned())? *
                parameters.to_crt().2,
            output_public_key_production: Some(production),
            requests: Vec::new(),
        },
    };
    let output = PolyCircuitCompiler { public_key: public_key_compiler }
        .compile_public_keys_with_lowerings(
            circuit,
            public_keys[0].clone(),
            public_keys.into_iter().skip(1),
            &mut lookup,
            &mut slots,
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "public-key circuit has no output".to_owned())?;
    DslContext::new("gpu-tall-nested-rns-public-key")
        .output("public_key", output.matrix)
        .map_err(|error| error.to_string())?
        .build()
        .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn build_encoding_graph(
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    layout: &BggSamplerLayout,
    artifact_compiler: &BggSlotTransferArtifactCompiler,
    production: ProductionId,
    lookup_compilers: &[mxx_bgg::LweLookupCompiler],
    slot_requests: &[mxx_bgg::BggSlotTransferGateRequest],
    rotation_offsets: &[u32],
    physical_slots: usize,
    error_sigma: f64,
    include_operational_residual: bool,
) -> Result<BuiltGraph, String> {
    let ring = layout.ring();
    let hash_key = ring.bytes_input(HASH_KEY_INPUT, 32);
    let secret = ring.artifact_input(
        production.clone(),
        SECRET_ARTIFACT,
        (1, layout.secret_dimension),
        ArtifactConfidentiality::Private,
    );
    let public_keys =
        imported_public_keys(&ring, &production, circuit.num_input(), layout.public_key_columns());
    let plaintexts = (0..circuit.num_input())
        .map(|input| {
            Family::pack(
                (0..physical_slots)
                    .map(|slot| ring.input(format!("plaintext_{input}_{slot}"), (1, 1)))
                    .collect(),
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let imported_slots = artifact_compiler
        .import_slots(&BggSlotTransferSlotArtifacts { production_id: production.clone() })
        .map_err(|error| error.to_string())?;
    let sample = BggTallEncodingSampler {
        layout: layout.clone(),
        gaussian_sigma: Some(
            RealExpr::from_f64_exact(error_sigma).map_err(|error| error.to_string())?,
        ),
        gaussian_max_coefficient_bound: Some(
            BigInt::from(hard_cutoff_from_sigma_bound(
                &BigDecimal::from_f64(error_sigma)
                    .ok_or_else(|| "error sigma must be finite".to_owned())?,
            ))
            .into(),
        ),
    }
    .sample(
        secret.clone(),
        &public_keys,
        &plaintexts,
        physical_slots.into(),
        Some(imported_slots.secrets.clone()),
    )
    .map_err(|error| error.to_string())?;
    let transformed_secrets = imported_slots
        .secrets
        .clone()
        .parallel_map({
            let secret = secret.clone();
            move |_, transform| secret.clone() * transform
        })
        .map_err(|error| error.to_string())?;
    let invocations = bind_lwe_lookup_invocations(
        parameters,
        circuit,
        production.clone(),
        lookup_compilers.iter().cloned(),
    )
    .map_err(|error| error.to_string())?;
    let lookup_c_b = ring.family_artifact_input(
        production.clone(),
        LOOKUP_C_B_ARTIFACT,
        physical_slots,
        (1, layout.secret_dimension * (layout.digit_count + 2)),
        ArtifactConfidentiality::Public,
    );
    let mut lookup = LweLookupTallEncodingLowering::new(invocations, lookup_c_b)
        .map_err(|error| error.to_string())?;
    let public_slots = BggSlotTransferPublicSlotWires {
        public_keys: imported_slots.public_keys,
        b0_preimage_chunks: imported_slots.b0_preimage_chunks,
        b1_preimage_chunks: imported_slots.b1_preimage_chunks,
    };
    let gate_wires = artifact_compiler
        .import_gate_preimages(
            &BggSlotTransferGateArtifacts { production_id: production.clone() },
            slot_requests,
        )
        .map_err(|error| error.to_string())?;
    let rotation_compiler = TallRotationEncodingCompiler {
        modulus: layout.modulus.clone(),
        ring_dimension: layout.ring_dimension.clone(),
        secret_size: layout.secret_dimension,
        slot_count: physical_slots,
        gadget_base: layout.gadget_base.clone(),
        digit_count: layout.digit_count,
        error_sigma: RealExpr::from_f64_exact(error_sigma).map_err(|error| error.to_string())?,
        error_max_coefficient_bound: BigInt::from(hard_cutoff_from_sigma_bound(
            &BigDecimal::from_f64(error_sigma)
                .ok_or_else(|| "error sigma must be finite".to_owned())?,
        ))
        .into(),
    };
    let rotation_artifacts = TallRotationEncodingArtifacts {
        production_id: production.clone(),
        slot_count: u32::try_from(physical_slots)
            .map_err(|_| "physical slot count exceeds u32".to_owned())?,
    };
    let rotations = rotation_offsets
        .iter()
        .filter_map(|offset| {
            rotation_compiler
                .import_artifacts(&rotation_artifacts, *offset)
                .transpose()
                .map(|result| result.map(|rotation| (rotation.key, rotation)))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| error.to_string())?;
    let public_key_compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: layout.gadget_base.clone(),
        digit_count: layout.digit_count.into(),
    };
    let mut slots = BggTallSlotLowering {
        compiler: BggTallEncodingCompiler { public_key: public_key_compiler.clone() },
        artifact: artifact_compiler.clone(),
        hash_key,
        output_public_key_production: Some(production.clone()),
        c_b0: ring.artifact_input(
            production,
            C_B0_ARTIFACT,
            (1, layout.secret_dimension * (layout.digit_count + 2)),
            ArtifactConfidentiality::Public,
        ),
        slots: public_slots,
        gates: gate_wires,
        rotations,
    };
    let output = PolyCircuitCompiler { public_key: public_key_compiler }
        .compile_tall_encodings_with_lowerings(
            circuit,
            sample.encodings[0].clone(),
            sample.encodings.into_iter().skip(1),
            &mut lookup,
            &mut slots,
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "encoding circuit has no output".to_owned())?;
    let BggTallPlaintext::Diagonal(output_plaintexts) = output.plaintext else {
        return Err("nested-RNS output plaintext is hidden".to_owned());
    };
    let encoding_rows = output.rows;
    let output_public_key = output.pubkey.matrix;
    let mut context = DslContext::new("gpu-tall-nested-rns-encoding")
        .family_output("encoding_rows", encoding_rows.clone())
        .map_err(|error| error.to_string())?
        .output("encoding_public_key", output_public_key.clone())
        .map_err(|error| error.to_string())?
        .family_output("output_plaintexts", output_plaintexts.clone())
        .map_err(|error| error.to_string())?
        .family_output("transformed_secrets", transformed_secrets.clone())
        .map_err(|error| error.to_string())?;
    if include_operational_residual {
        let gadget =
            ring.gadget(layout.secret_dimension, layout.gadget_base.clone(), layout.digit_count);
        let public_key = output_public_key.clone();
        let residuals = parallel_zip(
            (encoding_rows, output_plaintexts, transformed_secrets),
            move |_, (encoding, plaintext, transformed_secret)| {
                let signal = transformed_secret.clone() * public_key.clone() -
                    plaintext * (transformed_secret * gadget.clone());
                encoding - signal
            },
        )
        .map_err(|error| error.to_string())?;
        // The operational target remains the complete residual family.  The
        // executable decoder is a fixed lane witness taken from that exact
        // family, rather than an independently computed same-modulus matrix.
        let decoder_input = residuals
            .get_static(0)
            .slice(
                Some(IndexRange { start: 0.into(), end: 1.into() }),
                Some(IndexRange { start: 0.into(), end: 1.into() }),
            )
            .semantic_anchor(TALL_DECODER_RESIDUAL_ANCHOR)
            .map_err(|error| error.to_string())?;
        let q_max = parameters.to_crt().0.into_iter().max().expect("nonempty CRT basis");
        let decoded = decoder_input
            .threshold_decode_bools(IntExpr::constant(q_max), 1)
            .into_iter()
            .next()
            .ok_or_else(|| "Tall operational decoder has no Boolean output".to_owned())?
            .semantic_anchor(TALL_DECODER_RESULT_ANCHOR)
            .map_err(|error| error.to_string())?;
        context = context
            .family_output(TALL_OPERATIONAL_RESIDUAL, residuals)
            .map_err(|error| error.to_string())?
            .bool_output(TALL_OPERATIONAL_DECODED, decoded)
            .map_err(|error| error.to_string())?;
    }
    context.build().map_err(|error| error.to_string())
}

fn lattice_security_bits(parameters: &DCRTPolyParams, sigma: f64) -> Result<u64, String> {
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(parameters.modulus().to_string())
        .arg("--s-dist")
        .arg(r#"{"name":"Ternary"}"#)
        .arg("--e-dist")
        .arg(format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#))
        .output()
        .map_err(|error| format!("failed to start lattice-estimator-cli: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "lattice-estimator-cli failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim()
        .parse()
        .map_err(|_| format!("invalid lattice-estimator-cli output: {stdout}"))
}

fn select_parameters(config: &TestConfig) -> Result<PreparedCandidate, String> {
    info!("stage 1/4: parameter simulation");
    let candidate_dimensions = config.candidate_dimensions();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parameter_simulation_parallelism)
        .thread_name(|index| format!("tall-parameter-simulation-{index}"))
        .build()
        .map_err(|error| format!("could not create parameter simulation pool: {error}"))?;
    info!(
        candidate_count = candidate_dimensions.len(),
        selected_mode = config.selected_parameters.is_some(),
        selected_crt_depth = config.selected_parameters.map(|selected| selected.0),
        selected_log_ring_dimension = config.selected_parameters.map(|selected| selected.1),
        min_crt_depth = config.min_crt_depth,
        max_crt_depth = config.max_crt_depth,
        min_log_ring_dimension = config.min_log_ring_dimension,
        max_log_ring_dimension = config.max_log_ring_dimension,
        gadget_base_bits = config.gadget_base_bits,
        scale = config.scale,
        parallelism = config.parameter_simulation_parallelism,
        "configured bounded parallel parameter simulation"
    );
    for batch in candidate_dimensions.chunks(config.parameter_simulation_parallelism) {
        let batch_results = pool.install(|| {
            batch
                .par_iter()
                .map(|&(crt_depth, log_ring_dimension)| -> Result<_, String> {
                    let ring_dimension = 1u32
                        .checked_shl(
                            u32::try_from(log_ring_dimension)
                                .map_err(|_| "log ring dimension exceeds u32".to_owned())?,
                        )
                        .ok_or_else(|| "ring dimension overflow".to_owned())?;
                    let parameters = DCRTPolyParams::new(
                        ring_dimension,
                        crt_depth,
                        config.crt_modulus_bits,
                        u32::try_from(config.gadget_base_bits)
                            .map_err(|_| "gadget base bits exceed u32".to_owned())?,
                    );
                    // A zero target is the explicitly configured execution-smoke mode for n = 8.
                    // Its security predicate is tautological, so do not require
                    // Sage/lattice-estimator just to establish 0 >= 0. Any
                    // positive target retains the existing concrete
                    // estimator check.
                    let achieved_security_bits = if config.security_bits == 0 {
                        0
                    } else {
                        lattice_security_bits(&parameters, config.error_sigma)?
                    };
                    info!(
                        crt_depth,
                        ring_dimension,
                        achieved_security_bits,
                        required_security_bits = config.security_bits,
                        "evaluated lattice-security candidate"
                    );
                    if achieved_security_bits < config.security_bits {
                        return Ok(None);
                    }
                    let candidate = prepare_candidate(parameters, config, achieved_security_bits)?;
                    Ok(Some(candidate))
                })
                .collect::<Vec<_>>()
        });
        for ((crt_depth, log_ring_dimension), result) in batch.iter().zip(batch_results) {
            let Some(candidate) = result? else {
                continue;
            };
            let accepted = candidate.operational_report.accepted;
            info!(
                crt_depth = *crt_depth,
                ring_dimension = 1usize << *log_ring_dimension,
                operational_noise_bound = %candidate.operational_noise_bound,
                diagnostics = ?candidate.operational_report.diagnostics,
                accepted,
                "evaluated Tall BGG+ operational candidate"
            );
            if accepted {
                return Ok(candidate);
            }
            debug!("rejected Tall BGG+ operational candidate");
        }
    }
    Err("no configured CRT depth and ring dimension satisfy security and noise".to_owned())
}

fn log_cost_report(label: &str, report: &CostReport) {
    info!(
        label,
        total_work_seconds = report.total_work_seconds,
        preimage_sampling_work_seconds = report.preimage_sampling_work_seconds,
        critical_path_seconds = report.critical_path_seconds,
        maximum_parallelism = report.maximum_parallelism,
        workspace_high_water_bytes = report.workspace_high_water_bytes,
        peak_memory_bytes = report.peak_memory_bytes,
        "GPU benchmark estimate"
    );
    for (scope, cost) in &report.per_subgraph {
        debug!(label, scope, ?cost, "GPU benchmark subgraph estimate");
    }
}

fn benchmark_estimation(
    selected: &PreparedCandidate,
    config: &TestConfig,
    gpu_parameters: &GpuDCRTPolyParams,
    device_ids: &[i32],
) -> Result<(CostReport, CostReport, CostReport), String> {
    info!("stage 2/4: benchmark estimation");
    let bindings = ParamEnv::default();
    let manifests =
        BTreeMap::from([(selected.production.clone(), selected.runtime_manifest.clone())]);
    let preprocessing_graph =
        selected.producer.validate(&bindings).map_err(|error| error.to_string())?;
    let public_graph = selected
        .public_key_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    let encoding_graph = selected
        .encoding_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: config.benchmark_warmups,
        measured_iterations: config.benchmark_iterations,
        memory_poll_interval: Duration::from_millis(1),
    };
    let estimator_config = EstimateConfig {
        device_pool_size: config.max_parallel_instances,
        per_instance_occupancy: 1,
    };
    let make_backend = || {
        GpuNodeMeasurementBackend::new(
            gpu_backend_on([gpu_parameters.clone()], device_ids.iter().copied()),
            device_ids[0],
            harness.clone(),
            selected.parameters.to_crt().2,
        )
    };
    let preprocessing_estimator_config = EstimateConfig {
        device_pool_size: config.preprocessing_parallel_instances,
        per_instance_occupancy: 1,
    };
    let mut preprocessing_backend = make_backend();
    let preprocessing_started = Instant::now();
    info!(subgraph = "preprocessing", "benchmark subgraph estimation begin");
    let preprocessing_report =
        estimate(&preprocessing_graph, &mut preprocessing_backend, &preprocessing_estimator_config)
            .map_err(|error| error.to_string())?;
    info!(subgraph = "preprocessing", elapsed = ?preprocessing_started.elapsed(), "benchmark subgraph estimation complete");
    info!(
        lookup_preimage_count = selected.lookup_preimage_count,
        slot_preimage_count = selected.slot_preimage_count,
        total_preimage_count = selected.preprocessing_preimage_count,
        "estimated lookup and slot-operation preimage sampling"
    );
    log_cost_report("TallBggPreprocessing", &preprocessing_report);
    let mut public_backend = make_backend();
    let public_started = Instant::now();
    info!(subgraph = "public-key", "benchmark subgraph estimation begin");
    let public_report = estimate(&public_graph, &mut public_backend, &estimator_config)
        .map_err(|error| error.to_string())?;
    info!(subgraph = "public-key", elapsed = ?public_started.elapsed(), "benchmark subgraph estimation complete");
    log_cost_report("TallBggPublicKey", &public_report);
    let mut encoding_backend = make_backend();
    let encoding_started = Instant::now();
    info!(subgraph = "encoding", "benchmark subgraph estimation begin");
    let encoding_report = estimate(&encoding_graph, &mut encoding_backend, &estimator_config)
        .map_err(|error| error.to_string())?;
    info!(subgraph = "encoding", elapsed = ?encoding_started.elapsed(), "benchmark subgraph estimation complete");
    log_cost_report("TallBggEncoding", &encoding_report);
    Ok((preprocessing_report, public_report, encoding_report))
}

fn execution_config(
    config: &TestConfig,
    max_parallel_instances: usize,
    preprocessing_preimage_count: Option<usize>,
) -> Result<ExecutionConfig, String> {
    Ok(ExecutionConfig {
        max_parallel_instances: NonZeroUsize::new(max_parallel_instances)
            .ok_or_else(|| "maximum parallel instances must be positive".to_owned())?,
        preimage_progress: preprocessing_preimage_count.map(|total| PreimageProgressConfig {
            total,
            report_interval: NonZeroUsize::new(config.preimage_progress_interval)
                .expect("validated nonzero progress interval"),
        }),
        // Release-stream epochs bound queued frees without synchronizing live values.
        release_fence_interval: Some(
            NonZeroUsize::new(config.release_fence_interval)
                .expect("validated nonzero fence interval"),
        ),
    })
}

fn matrix_output(
    result: &ExecutionResult<GpuDcrtBackend>,
    name: &str,
) -> Result<GpuDCRTPolyMatrix, String> {
    match result.outputs.get(name) {
        Some(RuntimeValue::Matrix(value)) => Ok(value.as_ref().clone()),
        Some(_) => Err(format!("output {name} is not a matrix")),
        None => Err(format!("missing output {name}")),
    }
}

fn matrix_family_output(
    result: &mut ExecutionResult<GpuDcrtBackend>,
    name: &str,
    backend: &GpuDcrtBackend,
    store: &mut MemoryArtifactStore,
) -> Result<Vec<GpuDCRTPolyMatrix>, String> {
    let RuntimeValue::IndexedFamily(values) =
        result.materialize_output(name, backend, store).map_err(|error| error.to_string())?
    else {
        return Err(format!("output {name} is not an indexed family"));
    };
    values
        .iter()
        .map(|value| match value {
            RuntimeValue::Matrix(matrix) => Ok(matrix.as_ref().clone()),
            _ => Err(format!("output family {name} contains a non-matrix member")),
        })
        .collect()
}

fn save_preprocessing(
    config: &TestConfig,
    store: &MemoryArtifactStore,
    manifest: &RuntimeManifest,
    hash_key: [u8; 32],
) -> Result<PathBuf, String> {
    let payloads = store.snapshot_manifest_payloads(manifest).map_err(|error| error.to_string())?;
    let payload_size = |payload: &ArtifactPayload| match payload {
        ArtifactPayload::Matrix(bytes) |
        ArtifactPayload::Bytes(bytes) |
        ArtifactPayload::TypedBlob(bytes) => bytes.len(),
        ArtifactPayload::Trapdoor { public_bytes, secret_bytes } => {
            public_bytes.len().saturating_add(secret_bytes.len())
        }
    };
    let total_payload_bytes =
        payloads.iter().map(|(_, payload)| payload_size(payload)).sum::<usize>();
    info!(
        manifest_artifacts = manifest.artifacts.len(),
        payload_count = payloads.len(),
        total_payload_bytes,
        "preprocessing artifact totals"
    );
    for (key, payload) in &payloads {
        let descriptor = manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| format!("manifest omits checkpoint payload {}", key.name))?;
        debug!(
            name = key.name,
            index = ?key.index,
            artifact_type = ?descriptor.artifact_type,
            confidentiality = ?descriptor.confidentiality,
            payload_bytes = payload_size(payload),
            "preprocessing artifact payload"
        );
    }
    let unique =
        SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| error.to_string())?.as_nanos();
    let directory =
        config.checkpoint_root.join(format!("run-{unique}-{:016x}", rand::random::<u64>()));
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join("preprocessing.json");
    let checkpoint = PreprocessingCheckpoint { hash_key, manifest: manifest.clone(), payloads };
    fs::write(&path, serde_json::to_vec(&checkpoint).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    Ok(path)
}

fn reload_preprocessing(
    path: &PathBuf,
) -> Result<(MemoryArtifactStore, RuntimeManifest, [u8; 32]), String> {
    let restored: PreprocessingCheckpoint =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let mut reloaded = MemoryArtifactStore::default();
    for (key, payload) in restored.payloads {
        let descriptor = restored
            .manifest
            .artifacts
            .get(&key.name)
            .ok_or_else(|| format!("checkpoint manifest omits {}", key.name))?;
        reloaded
            .store(
                key,
                &descriptor.artifact_type,
                descriptor.confidentiality,
                descriptor.layout.as_deref(),
                payload,
            )
            .map_err(|error| error.to_string())?;
    }
    reloaded.store_manifest(restored.manifest.clone()).map_err(|error| error.to_string())?;
    Ok((reloaded, restored.manifest, restored.hash_key))
}

fn random_operands(selected: &PreparedCandidate, config: &TestConfig) -> Vec<Vec<BigUint>> {
    let modulus = selected.parameters.modulus().as_ref().clone();
    (0..=config.mul_count)
        .into_par_iter()
        .map(|_| {
            (0..selected.parameters.ring_dimension() as usize)
                .into_par_iter()
                .map_init(rand::rng, |rng, _| gen_biguint_for_modulus(rng, &modulus))
                .collect()
        })
        .collect()
}

fn expected_product_on_gpu(
    selected: &PreparedCandidate,
    gpu_parameters: &GpuDCRTPolyParams,
    operands: &[Vec<BigUint>],
) -> Result<Vec<BigUint>, String> {
    let mut matrices = operands.iter().map(|coefficients| {
        let polynomial = DCRTPoly::from_biguints_eval(&selected.parameters, coefficients);
        let cpu = DCRTPolyMatrix::from_poly_vec_row(&selected.parameters, vec![polynomial]);
        GpuDCRTPolyMatrix::from_cpu_matrix(gpu_parameters, &cpu)
    });
    let first = matrices.next().ok_or_else(|| "expected at least one operand".to_owned())?;
    let product = matrices.fold(first, |left, right| left * right);
    product.wait_until_ready();
    Ok(product.to_cpu_matrix().entry(0, 0).eval_slots())
}

fn encoding_inputs(
    selected: &PreparedCandidate,
    gpu_parameters: &GpuDCRTPolyParams,
    operands: &[Vec<BigUint>],
    hash_key: [u8; 32],
) -> Result<BTreeMap<String, RuntimeValue<GpuDcrtBackend>>, String> {
    let encoded = operands
        .iter()
        .flat_map(|coefficients| {
            encode_nested_rns_poly_with_offset::<DCRTPoly>(
                selected.nested.p_moduli_bits,
                selected.nested.max_unreduced_muls,
                &selected.parameters,
                coefficients,
                0,
                Some(selected.nested.q_moduli_depth),
            )
        })
        .collect::<Vec<_>>();
    if encoded.len() != selected.circuit.num_input() {
        return Err(format!(
            "nested-RNS encoding produced {} circuit inputs, expected {}",
            encoded.len(),
            selected.circuit.num_input()
        ));
    }
    let mut inputs =
        BTreeMap::from([(HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]);
    let matrices = encoded
        .into_par_iter()
        .enumerate()
        .flat_map_iter(|(input, lanes)| {
            lanes.into_iter().enumerate().map(move |(slot, value)| (input, slot, value))
        })
        .map(|(input, slot, value)| {
            let polynomial = DCRTPoly::from_biguint_to_constant(&selected.parameters, value);
            let cpu = DCRTPolyMatrix::from_poly_vec_row(&selected.parameters, vec![polynomial]);
            (
                format!("plaintext_{input}_{slot}"),
                RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(gpu_parameters, &cpu)),
            )
        })
        .collect::<Vec<_>>();
    for (name, matrix) in matrices {
        inputs.insert(name, matrix);
    }
    Ok(inputs)
}

fn end_to_end_processing(
    selected: &PreparedCandidate,
    config: &TestConfig,
    gpu_parameters: &GpuDCRTPolyParams,
    device_ids: &[i32],
) -> Result<EndToEndOutputs, String> {
    info!("stage 3/4: end-to-end processing");
    info!(
        achieved_security_bits = selected.achieved_security_bits,
        runtime_artifacts = selected.runtime_manifest.artifacts.len(),
        "selected parameter evidence retained for end-to-end processing"
    );
    let bindings = ParamEnv::default();
    let producer_execution_config = execution_config(
        config,
        config.preprocessing_parallel_instances,
        Some(selected.preprocessing_preimage_count),
    )?;
    let runtime_execution_config = execution_config(config, config.max_parallel_instances, None)?;
    info!(
        elapsed = ?selected.preprocessing_graph_construction,
        "timed preprocessing graph construction during selected-candidate preparation"
    );
    let (mut store, manifest, hash_key) = if let Some(checkpoint_path) = &config.reuse_checkpoint {
        let started = Instant::now();
        let (store, manifest, hash_key) = reload_preprocessing(checkpoint_path)?;
        if manifest.production_id != selected.production {
            return Err(
                "reused checkpoint production id differs from selected parameters".to_owned()
            );
        }
        info!(
            elapsed = ?started.elapsed(),
            path = %checkpoint_path.display(),
            "reused preprocessing checkpoint; skipped preprocessing execution"
        );
        (store, manifest, hash_key)
    } else {
        let mut hash_key = [0u8; 32];
        rand::rng().fill(&mut hash_key);
        let started = Instant::now();
        let producer = selected.producer.validate(&bindings).map_err(|error| error.to_string())?;
        info!(elapsed = ?started.elapsed(), "timed preprocessing graph validation");

        let mut producer_store = MemoryArtifactStore::default();
        // Preprocessing artifacts are serialized by the store below. Use a dedicated context so
        // the consumer passes start without the producer's allocator pool and transient preimage
        // buffers still resident on the GPU.
        let production = {
            let mut producer_backend =
                gpu_backend_on([gpu_parameters.clone()], device_ids.iter().copied());
            let started = Instant::now();
            let producer_result = execute_in_session_with_config(
                &producer,
                &mut producer_backend,
                BTreeMap::from([(
                    HASH_KEY_INPUT.to_owned(),
                    RuntimeValue::Bytes(hash_key.to_vec()),
                )]),
                &mut producer_store,
                [0x71; 32],
                producer_execution_config,
            )
            .map_err(|error| error.to_string())?;
            info!(elapsed = ?started.elapsed(), "timed preprocessing execution");
            producer_result
                .production_id
                .ok_or_else(|| "preprocessing execution returned no production id".to_owned())?
        };
        if production != selected.production {
            return Err("preprocessing production id differs from the selected manifest".to_owned());
        }
        let manifest = producer_store
            .manifest(&production)
            .cloned()
            .ok_or_else(|| "preprocessing manifest was not committed".to_owned())?;

        let started = Instant::now();
        let checkpoint_path = save_preprocessing(config, &producer_store, &manifest, hash_key)?;
        info!(elapsed = ?started.elapsed(), path = %checkpoint_path.display(), "timed checkpoint serialization");
        let started = Instant::now();
        let (store, manifest, hash_key) = reload_preprocessing(&checkpoint_path)?;
        info!(elapsed = ?started.elapsed(), path = %checkpoint_path.display(), "timed checkpoint reload");
        (store, manifest, hash_key)
    };
    let manifests = BTreeMap::from([(selected.production.clone(), manifest)]);
    let mut backend = gpu_backend_on([gpu_parameters.clone()], device_ids.iter().copied());

    let started = Instant::now();
    let public_graph_source = build_public_key_graph(
        &selected.parameters,
        &selected.circuit,
        &selected.layout,
        &selected.artifact_compiler,
        selected.production.clone(),
        &selected.lookup_compilers,
    )?;
    info!(elapsed = ?started.elapsed(), "timed Tall public-key graph construction");
    let public_pass_started = Instant::now();
    let started = Instant::now();
    let public_graph = public_graph_source
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    info!(elapsed = ?started.elapsed(), "timed Tall public-key graph validation");
    let started = Instant::now();
    let public_result = execute_with_config(
        &public_graph,
        &mut backend,
        BTreeMap::from([(HASH_KEY_INPUT.to_owned(), RuntimeValue::Bytes(hash_key.to_vec()))]),
        &mut store,
        SamplingMode::Fresh,
        runtime_execution_config,
    )
    .map_err(|error| error.to_string())?;
    let public_key = matrix_output(&public_result, "public_key")?;
    public_key.wait_until_ready();
    info!(elapsed = ?started.elapsed(), "timed TallBggPublicKey end-to-end evaluation");
    info!(elapsed = ?public_pass_started.elapsed(), "timed public-key-pass total");

    let started = Instant::now();
    let operands = random_operands(selected, config);
    let expected_evaluation_slots = expected_product_on_gpu(selected, gpu_parameters, &operands)?;
    let inputs = encoding_inputs(selected, gpu_parameters, &operands, hash_key)?;
    info!(elapsed = ?started.elapsed(), "timed random plaintext generation, GPU oracle, and Tall input encoding");
    let started = Instant::now();
    let encoding_graph_source = build_encoding_graph(
        &selected.parameters,
        &selected.circuit,
        &selected.layout,
        &selected.artifact_compiler,
        selected.production.clone(),
        &selected.lookup_compilers,
        &selected.slot_requests,
        &selected.rotation_offsets,
        selected.parameters.ring_dimension() as usize * selected.nested.q_moduli_depth,
        config.error_sigma,
        false,
    )?;
    info!(elapsed = ?started.elapsed(), "timed Tall encoding graph construction");
    let encoding_pass_started = Instant::now();
    let started = Instant::now();
    let encoding_graph = encoding_graph_source
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    info!(elapsed = ?started.elapsed(), "timed Tall encoding graph validation");
    let started = Instant::now();
    let mut encoding_result = execute_with_config(
        &encoding_graph,
        &mut backend,
        inputs,
        &mut store,
        SamplingMode::Fresh,
        runtime_execution_config,
    )
    .map_err(|error| error.to_string())?;
    let encoding_public_key = matrix_output(&encoding_result, "encoding_public_key")?;
    let encoding_rows =
        matrix_family_output(&mut encoding_result, "encoding_rows", &backend, &mut store)?;
    let output_plaintexts =
        matrix_family_output(&mut encoding_result, "output_plaintexts", &backend, &mut store)?;
    let transformed_secrets =
        matrix_family_output(&mut encoding_result, "transformed_secrets", &backend, &mut store)?;
    encoding_result.cleanup_staged(&mut store).map_err(|error| error.to_string())?;
    encoding_public_key.wait_until_ready();
    encoding_rows.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    output_plaintexts.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    transformed_secrets.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    info!(elapsed = ?started.elapsed(), "timed TallBggEncoding end-to-end evaluation");
    let started = Instant::now();
    let _transferred_public_key = public_key.to_cpu_matrix();
    let _transferred_encoding_public_key = encoding_public_key.to_cpu_matrix();
    let _transferred_encoding_rows =
        encoding_rows.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    let _transferred_plaintexts =
        output_plaintexts.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    let _transferred_secrets =
        transformed_secrets.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    info!(elapsed = ?started.elapsed(), "timed output transfer");
    info!(elapsed = ?encoding_pass_started.elapsed(), "timed encoding-pass total");
    Ok(EndToEndOutputs {
        public_key,
        encoding_public_key,
        encoding_rows,
        output_plaintexts,
        transformed_secrets,
        expected_evaluation_slots,
    })
}

fn runtime_verification(
    selected: &PreparedCandidate,
    gpu_parameters: &GpuDCRTPolyParams,
    outputs: EndToEndOutputs,
) -> Result<(), String> {
    info!("stage 4/4: runtime verification");
    if outputs.public_key != outputs.encoding_public_key {
        return Err("Tall encoding output public key differs from public-key evaluation".to_owned());
    }
    let physical_slots =
        selected.parameters.ring_dimension() as usize * selected.nested.q_moduli_depth;
    if outputs.encoding_rows.len() != physical_slots ||
        outputs.output_plaintexts.len() != physical_slots ||
        outputs.transformed_secrets.len() != physical_slots
    {
        return Err(
            "Tall output family cardinality differs from the physical SIMD lane count".to_owned()
        );
    }
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(gpu_parameters, selected.layout.secret_dimension);
    let modulus = selected.parameters.modulus().as_ref().clone();
    let half_modulus = &modulus / BigUint::from(2u8);
    let mut maximum_noise = BigUint::zero();
    let mut maximum_location = (0usize, 0usize, 0usize);
    for slot in 0..physical_slots {
        let evaluation_slot = slot / selected.nested.q_moduli_depth;
        let expected = outputs
            .expected_evaluation_slots
            .get(evaluation_slot)
            .cloned()
            .ok_or_else(|| "GPU oracle omitted an expected evaluation slot".to_owned())?;
        let expected_poly = DCRTPoly::from_biguint_to_constant(&selected.parameters, expected);
        let expected_cpu =
            DCRTPolyMatrix::from_poly_vec_row(&selected.parameters, vec![expected_poly]);
        let expected_matrix = GpuDCRTPolyMatrix::from_cpu_matrix(gpu_parameters, &expected_cpu);
        if outputs.output_plaintexts[slot] != expected_matrix {
            return Err(format!("runtime plaintext mismatch at physical slot {slot}"));
        }
        let signal = &outputs.transformed_secrets[slot] * &outputs.encoding_public_key -
            &expected_matrix * (&outputs.transformed_secrets[slot] * &gadget);
        let residual = &outputs.encoding_rows[slot] - &signal;
        residual.wait_until_ready();
        let cpu = residual.to_cpu_matrix();
        for column in 0..cpu.col_size() {
            for (coefficient_index, value) in
                cpu.entry(0, column).coeffs_biguints().into_iter().enumerate()
            {
                let centered = if value > half_modulus { &modulus - value } else { value };
                if centered > maximum_noise {
                    maximum_noise = centered;
                    maximum_location = (slot, column, coefficient_index);
                    debug!(
                        maximum_noise = %maximum_noise,
                        ?maximum_location,
                        "updated Tall BGG+ residual maximum"
                    );
                }
            }
        }
    }
    let q_max = selected
        .parameters
        .to_crt()
        .0
        .into_iter()
        .max()
        .ok_or_else(|| "runtime verification requires a nonempty CRT basis".to_owned())?;
    let threshold_lhs = BigUint::from(2u8) * BigUint::from(q_max) * &maximum_noise;
    info!(
        maximum_noise = %maximum_noise,
        ?maximum_location,
        operational_noise_bound = %selected.operational_noise_bound,
        within_operational_envelope = maximum_noise <= selected.operational_noise_bound,
        threshold_lhs = %threshold_lhs,
        ciphertext_modulus = %modulus,
        "measured Tall BGG+ residual"
    );
    if maximum_noise > selected.operational_noise_bound {
        return Err(format!(
            "measured residual {maximum_noise} at {maximum_location:?} exceeds operational bound {}",
            selected.operational_noise_bound
        ));
    }
    if threshold_lhs >= modulus {
        return Err(format!(
            "measured residual {maximum_noise} at {maximum_location:?} fails 2*q_max*noise < q"
        ));
    }
    Ok(())
}

#[test]
fn selected_parameters_produce_exactly_one_candidate() {
    assert_eq!(candidate_dimensions(1, 16, 3, 8, Some((7, 5))), vec![(7, 5)]);
}

#[test]
fn range_parameters_preserve_the_cartesian_candidate_search() {
    assert_eq!(candidate_dimensions(2, 3, 4, 5, None), vec![(2, 4), (2, 5), (3, 4), (3, 5)]);
}

#[test]
#[ignore = "runs the Rust Tall BGG+ operational parameter simulation"]
fn test_tall_bgg_nested_rns_parameter_simulation() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let config = TestConfig::from_env().expect("valid Tall nested-RNS configuration");
    info!(?config, "effective Tall nested-RNS parameter-simulation configuration");
    let selected = select_parameters(&config).expect("Rust operational parameter simulation");
    info!(
        crt_depth = selected.parameters.to_crt().2,
        ring_dimension = selected.parameters.ring_dimension(),
        operational_noise_bound = %selected.operational_noise_bound,
        operational_accepted = selected.operational_report.accepted,
        "completed Rust-only Tall parameter simulation"
    );
}

#[test]
#[ignore = "requires a CUDA GPU and lattice-estimator-cli; runs the full Tall BGG+ round trip"]
fn test_gpu_tall_bgg_nested_rns_modq_arithmetic() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let config = TestConfig::from_env().expect("valid GPU Tall nested-RNS configuration");
    info!(?config, "effective GPU Tall nested-RNS integration configuration");
    let selected = select_parameters(&config).expect("parameter simulation");
    info!(
        crt_depth = selected.parameters.to_crt().2,
        ring_dimension = selected.parameters.ring_dimension(),
        physical_slots = selected.parameters.ring_dimension() as usize *
            selected.nested.q_moduli_depth,
        achieved_security_bits = selected.achieved_security_bits,
        operational_noise_bound = %selected.operational_noise_bound,
        operational_accepted = selected.operational_report.accepted,
        "selected Tall nested-RNS parameters"
    );
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "at least one CUDA GPU");
    info!(?device_ids, gpu_count = device_ids.len(), "using all detected CUDA GPUs");
    let (moduli, _, _) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
    );
    let _reports = benchmark_estimation(&selected, &config, &gpu_parameters, &device_ids)
        .expect("benchmark estimation");
    if config.stop_after_benchmark_estimation {
        info!("stopped successfully after benchmark estimation before preprocessing execution");
        return;
    }
    let outputs = end_to_end_processing(&selected, &config, &gpu_parameters, &device_ids)
        .expect("end-to-end processing");
    runtime_verification(&selected, &gpu_parameters, outputs).expect("runtime verification");
}
