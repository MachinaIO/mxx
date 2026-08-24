#![cfg(feature = "gpu")]

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    CostReport, EstimateConfig, estimate, gpu::GpuNodeMeasurementBackend,
    harness::MeasurementHarnessConfig,
};
use mxx_bgg::{
    BggEncodingWire, BggPublicKeyCompiler, BggPublicKeySampler, BggPublicKeyWire, BggSamplerLayout,
    BggTallEncodingCompiler, BggTallEncodingSampler, BggTallPlaintext, BggTallSlotLowering,
    BggTallSlotPublicKeyLowering, LweLookupArtifactNames, LweLookupArtifacts, LweLookupCompiler,
    LweLookupIdentity, LweLookupPreprocessingLowering, LweLookupTable,
    LweLookupTallEncodingLowering, NoSlotOperations, PolyCircuitCompiler,
    TallRotationEncodingArtifactNames, TallRotationEncodingArtifacts, TallRotationEncodingCompiler,
    TallRotationEncodingKey, bind_lwe_lookup_invocations, collect_lwe_lookup_identities,
    required_tall_rotation_encodings,
};
use mxx_correctness::{
    ComparatorEndpointBinding, ComparatorSpec, EndpointAnchor, EndpointAnchors,
    EndpointSemanticBinding, EndpointSpecId, ExactMatrixInputMetadata, OperationalDecoderKind,
    OperationalDecoderTarget, OutputRef, StageId,
    operational_noise::{
        OperationalAcceptanceReport, OperationalCheckRequest, OperationalGadgetLayout,
        OperationalSimulationError, OperationalSimulationReport, ProgressEventKind,
        check_operational_noise_candidate_with_progress,
    },
    operational_protocol_from_graphs,
};
use mxx_dsl::{
    Bool, BuiltGraph, DslContext, IdealSpec, Int, Parallel, Ring, SemanticAnchor, parallel_zip,
};
use mxx_gadgets::{
    circuit::{PolyCircuit, PolyGateKind},
    circuit_gadgets::arith::{
        CrtWindow,
        nested_rns::{
            NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly, minimum_p_moduli_bits,
        },
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
    types::MatrixType,
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
use num_traits::{FromPrimitive, Zero};
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
const LOOKUP_PUBLIC_B_ARTIFACT: &str = "tall_nested_rns_lookup_public_b";
const INPUT_PUBLIC_KEY_PREFIX: &str = "tall_nested_rns_input_public_key";
const DIAGONAL_MASK_PUBLIC_KEY_ARTIFACT: &str = "tall_nested_rns_diagonal_mask_public_key";
const OUTPUT_PUBLIC_KEY_ARTIFACT: &str = "tall_nested_rns_output_public_key";
const TALL_OPERATIONAL_TARGET_ID: &str = "tall-threshold-decode";
const TALL_OPERATIONAL_RESIDUAL: &str = "operational_residual";
const TALL_OPERATIONAL_DECODED: &str = "operational_decoded";
const TALL_DECODER_RESIDUAL_ANCHOR: &str = "tall.decoder.residual";
const TALL_DECODER_RESULT_ANCHOR: &str = "tall.decoder.result";

fn log_graph_phase(phase: &'static str, state: &'static str, started: Option<&Instant>) {
    let (vm_rss_kib, vm_hwm_kib) = process_memory_kib();
    info!(
        phase,
        state,
        elapsed = ?started.map(Instant::elapsed),
        vm_rss_kib = ?vm_rss_kib,
        vm_hwm_kib = ?vm_hwm_kib,
        "Tall graph preparation phase"
    );
}

#[cfg(target_os = "linux")]
fn process_memory_kib() -> (Option<u64>, Option<u64>) {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return (None, None);
    };
    let value = |name: &str| {
        status.lines().find_map(|line| {
            let remainder = line.strip_prefix(name)?.trim();
            remainder.split_whitespace().next()?.parse().ok()
        })
    };
    (value("VmRSS:"), value("VmHWM:"))
}

#[cfg(not(target_os = "linux"))]
fn process_memory_kib() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[derive(Clone, Debug)]
struct TestConfig {
    mul_count: usize,
    min_crt_depth: usize,
    max_crt_depth: usize,
    min_log_ring_dimension: usize,
    max_log_ring_dimension: usize,
    selected_parameters: Option<(usize, usize)>,
    /// Number of DCRT evaluation slots carried through the nested-RNS/Tall encoding. `None`
    /// retains all slots from the ambient DCRT ring.
    encoding_ring_dimension: Option<usize>,
    /// Number of leading q-CRT towers carried through the nested-RNS/Tall encoding. `None`
    /// retains the full DCRT basis.
    encoding_crt_depth: Option<usize>,
    security_bits: u64,
    crt_modulus_bits: usize,
    /// Explicit nested-RNS p-basis width. `None` selects the smallest width supporting the
    /// configured unreduced-multiplication budget for the candidate's q basis.
    p_moduli_bits: Option<usize>,
    gadget_base_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    error_sigma: f64,
    trapdoor_sigma: f64,
    benchmark_warmups: usize,
    benchmark_iterations: usize,
    run_mode: TallRunMode,
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
        let crt_modulus_bits = env_usize("MXX_TALL_NESTED_RNS_CRT_MODULUS_BITS", 28)?;
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
            min_crt_depth: env_usize("MXX_TALL_NESTED_RNS_MIN_CRT_DEPTH", 2)?,
            max_crt_depth: env_usize("MXX_TALL_NESTED_RNS_MAX_CRT_DEPTH", 40)?,
            min_log_ring_dimension: env_usize("MXX_TALL_NESTED_RNS_MIN_LOG_RING_DIMENSION", 5)?,
            max_log_ring_dimension: env_usize("MXX_TALL_NESTED_RNS_MAX_LOG_RING_DIMENSION", 16)?,
            selected_parameters,
            encoding_ring_dimension: env_optional_usize(
                "MXX_TALL_NESTED_RNS_ENCODING_RING_DIMENSION",
            )?,
            encoding_crt_depth: env_optional_usize("MXX_TALL_NESTED_RNS_ENCODING_CRT_DEPTH")?,
            // n = 8 is intentionally an execution smoke parameter and has no positive lattice
            // security estimate. A caller may request a positive target, which will reject it.
            security_bits: env_u64("MXX_TALL_NESTED_RNS_SECURITY_BITS", 100)?,
            crt_modulus_bits,
            p_moduli_bits: env_optional_usize("MXX_TALL_NESTED_RNS_P_MODULI_BITS")?,
            gadget_base_bits: env_usize(
                "MXX_TALL_NESTED_RNS_GADGET_BASE_BITS",
                crt_modulus_bits.div_ceil(2),
            )?,
            // A multiplication consumes the product of two full-reduce outputs, so the
            // two-product budget closes the reduce/multiply loop at any multiplication depth.
            max_unreduced_muls: env_usize("MXX_TALL_NESTED_RNS_MAX_UNREDUCED_MULS", 2)?,
            scale: env_u64("MXX_TALL_NESTED_RNS_SCALE", 1 << 6)?,
            error_sigma: env_f64("MXX_TALL_NESTED_RNS_ERROR_SIGMA", 4.0)?,
            trapdoor_sigma: env_f64("MXX_TALL_NESTED_RNS_TRAPDOOR_SIGMA", 4.578)?,
            benchmark_warmups: env_usize("MXX_TALL_NESTED_RNS_BENCH_WARMUPS", 1)?,
            benchmark_iterations: env_usize("MXX_TALL_NESTED_RNS_BENCH_ITERATIONS", 2)?,
            run_mode: TallRunMode::from_env()?,
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
            config.encoding_ring_dimension.is_some_and(|dimension| !dimension.is_power_of_two()) ||
            config.encoding_crt_depth == Some(0) ||
            config.crt_modulus_bits == 0 ||
            config.gadget_base_bits == 0 ||
            config.max_unreduced_muls == 0 ||
            config.scale == 0 ||
            config.error_sigma < 0.0 ||
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
        validate_error_sigma(config.run_mode, config.error_sigma)?;
        if let Some((crt_depth, log_ring_dimension)) = config.selected_parameters &&
            (crt_depth == 0 ||
                u32::try_from(log_ring_dimension).is_err() ||
                1u32.checked_shl(log_ring_dimension as u32).is_none())
        {
            return Err("invalid selected Tall nested-RNS parameters".to_owned());
        }
        let (largest_crt_depth, largest_log_ring_dimension) = config
            .selected_parameters
            .unwrap_or((config.max_crt_depth, config.max_log_ring_dimension));
        let largest_ring_dimension = 1usize
            .checked_shl(largest_log_ring_dimension as u32)
            .ok_or_else(|| "largest available ring dimension exceeds usize".to_owned())?;
        config.encoding_ring_dimension(largest_ring_dimension)?;
        config.encoding_crt_depth(largest_crt_depth)?;
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
        .into_iter()
        .filter(|(crt_depth, log_ring_dimension)| {
            self.encoding_crt_depth
                .is_none_or(|encoding_crt_depth| encoding_crt_depth <= *crt_depth) &&
                self.encoding_ring_dimension.is_none_or(|encoding_ring_dimension| {
                    1usize
                        .checked_shl(*log_ring_dimension as u32)
                        .is_some_and(|ring_dimension| encoding_ring_dimension <= ring_dimension)
                })
        })
        .collect()
    }

    fn encoding_ring_dimension(&self, dcrt_ring_dimension: usize) -> Result<usize, String> {
        let encoding_ring_dimension = self.encoding_ring_dimension.unwrap_or(dcrt_ring_dimension);
        if !encoding_ring_dimension.is_power_of_two() {
            return Err("encoding ring dimension must be a positive power of two".to_owned());
        }
        if encoding_ring_dimension > dcrt_ring_dimension {
            return Err(format!(
                "encoding ring dimension {encoding_ring_dimension} exceeds DCRT ring dimension {dcrt_ring_dimension}"
            ));
        }
        Ok(encoding_ring_dimension)
    }

    fn encoding_crt_depth(&self, dcrt_crt_depth: usize) -> Result<usize, String> {
        let encoding_crt_depth = self.encoding_crt_depth.unwrap_or(dcrt_crt_depth);
        if encoding_crt_depth == 0 {
            return Err("encoding CRT depth must be positive".to_owned());
        }
        if encoding_crt_depth > dcrt_crt_depth {
            return Err(format!(
                "encoding CRT depth {encoding_crt_depth} exceeds DCRT CRT depth {dcrt_crt_depth}"
            ));
        }
        Ok(encoding_crt_depth)
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
    encoding_ring_dimension: usize,
    encoding_crt_depth: usize,
    physical_slots: usize,
    circuit: PolyCircuit<DCRTPoly>,
    nested: Arc<NestedRnsPolyContext>,
    layout: BggSamplerLayout,
    preprocessing: BuiltGraph,
    preprocessing_graph_construction: Duration,
    lookup_preimage_count: usize,
    preprocessing_preimage_count: usize,
    production: ProductionId,
    runtime_manifest: RuntimeManifest,
    lookup_compilers: Vec<mxx_bgg::LweLookupCompiler>,
    rotation_offsets: Vec<u32>,
    encoding_graph: BuiltGraph,
    operational_report: Option<OperationalSimulationReport>,
    achieved_security_bits: u64,
}

/// Whether graph preparation also runs the CPU-only operational-noise checker.
///
/// The noiseless runtime test deliberately skips it: it verifies the executable
/// Tall construction against an independent plaintext product, rather than
/// making that runtime check depend on the checker currently under development.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CandidatePreparation {
    OperationalChecked,
    RuntimeOnly,
}

#[derive(Debug, Serialize, Deserialize)]
struct PreprocessingCheckpoint {
    hash_key: [u8; 32],
    manifest: RuntimeManifest,
    payloads: Vec<(ArtifactKey, ArtifactPayload)>,
}

struct EndToEndOutputs {
    encoding_rows: Vec<GpuDCRTPolyMatrix>,
    output_plaintexts: Vec<GpuDCRTPolyMatrix>,
    residuals: Vec<GpuDCRTPolyMatrix>,
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

/// Chooses how far the existing Tall integration target runs. The modes are
/// stages of one test, so every mode uses the same protocol definition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TallRunMode {
    Graph,
    ZeroNoise,
    Simulation,
    Benchmark,
    BenchmarkSelected,
    Full,
}

impl TallRunMode {
    fn from_env() -> Result<Self, String> {
        match env::var("MXX_TALL_NESTED_RNS_MODE").as_deref().unwrap_or("full") {
            "graph" => Ok(Self::Graph),
            "zero-noise" => Ok(Self::ZeroNoise),
            "simulation" => Ok(Self::Simulation),
            "benchmark" => Ok(Self::Benchmark),
            "benchmark-selected" => Ok(Self::BenchmarkSelected),
            "full" => Ok(Self::Full),
            value => Err(format!(
                "MXX_TALL_NESTED_RNS_MODE must be graph, zero-noise, simulation, benchmark, benchmark-selected, or full (got {value})"
            )),
        }
    }
}

fn env_f64(name: &str, default: f64) -> Result<f64, String> {
    env::var(name)
        .map_or(Ok(default), |value| value.parse().map_err(|_| format!("{name} must be a number")))
}

fn validate_error_sigma(run_mode: TallRunMode, error_sigma: f64) -> Result<(), String> {
    if !error_sigma.is_finite() || error_sigma < 0.0 {
        return Err("MXX_TALL_NESTED_RNS_ERROR_SIGMA must be finite and nonnegative".to_owned());
    }
    if matches!(
        run_mode,
        TallRunMode::Simulation |
            TallRunMode::Benchmark |
            TallRunMode::BenchmarkSelected |
            TallRunMode::Full
    ) && error_sigma == 0.0
    {
        return Err(format!(
            "MXX_TALL_NESTED_RNS_ERROR_SIGMA must be greater than zero in {run_mode:?} mode; use zero-noise mode for an exact noiseless smoke test"
        ));
    }
    Ok(())
}

fn log_invocation(config: &TestConfig, selected: Option<&PreparedCandidate>) {
    let command =
        env::args_os().map(|argument| argument.to_string_lossy().into_owned()).collect::<Vec<_>>();
    let environment = env::vars()
        .filter(|(name, _)| {
            name.starts_with("MXX_TALL_NESTED_RNS_") ||
                matches!(
                    name.as_str(),
                    "CUDA_VISIBLE_DEVICES" | "CUDA_DEVICE_ORDER" | "RUST_LOG"
                )
        })
        .collect::<BTreeMap<_, _>>();
    info!(
        ?command,
        ?environment,
        requested_mode = ?config.run_mode,
        configured_selected_parameters = ?config.selected_parameters,
        configured_encoding_ring_dimension = ?config.encoding_ring_dimension,
        configured_encoding_crt_depth = ?config.encoding_crt_depth,
        error_sigma = config.error_sigma,
        "Tall nested-RNS reproducibility invocation"
    );
    if let Some(selected) = selected {
        info!(
            selected_crt_depth = selected.parameters.to_crt().2,
            selected_ring_dimension = selected.parameters.ring_dimension(),
            selected_log_ring_dimension = selected.parameters.ring_dimension().ilog2(),
            selected_encoding_ring_dimension = selected.encoding_ring_dimension,
            selected_encoding_crt_depth = selected.encoding_crt_depth,
            selected_physical_slots = selected.physical_slots,
            selected_security_bits = selected.achieved_security_bits,
            selected_operational_noise_bound = ?selected
                .operational_report
                .as_ref()
                .map(|report| &report.noise_bound),
            "Tall nested-RNS selected parameters"
        );
    }
}

fn build_modq_multiplication_circuit(
    parameters: &DCRTPolyParams,
    config: &TestConfig,
    evaluation_slots: usize,
    encoding_crt_depth: usize,
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
                CrtWindow::new(0, encoding_crt_depth, nested.q_moduli_depth),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let mut inputs = inputs.into_iter();
    let first = inputs.next().expect("mul_count + 1 is positive");
    let product = inputs.fold(first, |left, right| left.mul(&right, &mut circuit));
    let output = product.reconstruct_q1_anchors(&mut circuit);
    circuit.output([output.anchor_wire()]);
    CircuitBundle { circuit, nested }
}

fn gate_kind_counts(circuit: &PolyCircuit<DCRTPoly>) -> HashMap<PolyGateKind, usize> {
    let mut counts = HashMap::new();
    for (_, gate) in circuit.gates_in_id_order() {
        *counts.entry(gate.gate_type.kind()).or_default() += 1;
    }
    counts
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LookupPlanningStats {
    occurrences: usize,
    preimages: usize,
}

/// Counts concrete lookup invocations without constructing lookup tables, DSL graphs, or samples.
///
/// The identity collector recursively expands registered sub-circuit calls in exactly the same way
/// as the LWE preprocessing lowering. Table lengths are deliberately summed per occurrence: sharing
/// a registered LUT does not remove any required preimage.
fn lookup_planning_stats(circuit: &PolyCircuit<DCRTPoly>) -> Result<LookupPlanningStats, String> {
    let identities = collect_lwe_lookup_identities(circuit).map_err(|error| error.to_string())?;
    let preimages = identities.iter().try_fold(0usize, |total, identity| {
        total
            .checked_add(circuit.lookup_table(identity.lookup).len())
            .ok_or_else(|| "lookup planning preimage count exceeds usize".to_owned())
    })?;
    Ok(LookupPlanningStats { occurrences: identities.len(), preimages })
}

fn selected_cpu_parameters(
    config: &TestConfig,
    test_name: &str,
) -> Result<(usize, usize, DCRTPolyParams), String> {
    let (crt_depth, log_ring_dimension) = config.selected_parameters.ok_or_else(|| {
        format!(
            "{test_name} requires both MXX_TALL_NESTED_RNS_SELECTED_CRT_DEPTH and \
             MXX_TALL_NESTED_RNS_SELECTED_LOG_RING_DIMENSION"
        )
    })?;
    let shift = u32::try_from(log_ring_dimension)
        .map_err(|_| "selected log ring dimension exceeds u32".to_owned())?;
    let ring_dimension = 1u32
        .checked_shl(shift)
        .ok_or_else(|| "selected ring dimension overflows u32".to_owned())?;
    let base_bits = u32::try_from(config.gadget_base_bits)
        .map_err(|_| "configured gadget base bits exceed u32".to_owned())?;
    Ok((
        crt_depth,
        log_ring_dimension,
        DCRTPolyParams::new(ring_dimension, crt_depth, config.crt_modulus_bits, base_bits),
    ))
}

/// `log2` of an arbitrary-precision unsigned value for human-readable noise-margin logs.
fn log2_biguint(value: &BigUint) -> f64 {
    if value.is_zero() {
        return f64::NEG_INFINITY;
    }
    let bits = value.bits();
    if bits <= 53 {
        return (value.iter_u64_digits().next().unwrap_or(0) as f64).log2();
    }
    let shift = bits - 53;
    let top = (value >> shift).iter_u64_digits().next().unwrap_or(0) as f64;
    top.log2() + shift as f64
}

/// The evaluated noise bound and the decoder rule's maximum acceptable noise, both in bits.
/// Threshold decode accepts `2 * p * noise < q`, so the acceptable-noise budget is `q / (2p)`;
/// the boolean interval accepts noise inside the quarter-width band around each residue.
fn acceptance_log2(report: &OperationalSimulationReport) -> (f64, f64) {
    let noise_bound_log2 = log2_biguint(&report.noise_bound);
    let noise_threshold_log2 = match &report.acceptance {
        OperationalAcceptanceReport::Threshold { plaintext_modulus, .. } => {
            log2_biguint(&report.ciphertext_modulus) - 1.0 - log2_biguint(plaintext_modulus)
        }
        OperationalAcceptanceReport::BooleanInterval { quarter, .. } => {
            log2_biguint(quarter.magnitude())
        }
    };
    (noise_bound_log2, noise_threshold_log2)
}

fn run_tall_operational_check(
    preprocessing: &BuiltGraph,
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
    let (preprocessing_scopes, preprocessing_nodes, preprocessing_outputs) =
        graph_counts(&preprocessing.graph);
    let (encoding_scopes, encoding_nodes, encoding_outputs) = graph_counts(&encoding.graph);
    info!(
        ring_dimension = parameters.ring_dimension(),
        crt_depth = parameters.to_crt().2,
        preprocessing_scopes,
        preprocessing_nodes,
        preprocessing_outputs,
        encoding_scopes,
        encoding_nodes,
        encoding_outputs,
        "constructed Tall operational checker graphs"
    );
    let mut exact_input_metadata = BTreeMap::new();
    for node in encoding.graph.root_scope().nodes() {
        let NodeKind::Input { name, .. } = node.kind() else { continue };
        let Some(indices) = name.strip_prefix("plaintext_") else { continue };
        // Plaintexts are checker-visible indexed-family inputs. Accept the family name directly;
        // the suffixed form remains understood for older graph fixtures.
        let input_index = indices.split_once('_').map_or(indices, |(input_index, _)| input_index);
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
        vec![("preprocessing".to_owned(), preprocessing), ("encoding".to_owned(), encoding)],
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
                normalization_nodes_processed = event.normalization_nodes_processed,
                normalization_nodes_total = event.normalization_nodes_total,
                normalization_exact_term_count = event.normalization_exact_term_count,
                normalization_bounded_fold_count = event.normalization_bounded_fold_count,
                normalization_relation_candidates = event.normalization_relation_candidates,
                normalization_relations_applied = event.normalization_relations_applied,
                normalization_relations_remaining = event.normalization_relations_remaining,
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
                normalization_nodes_processed = event.normalization_nodes_processed,
                normalization_nodes_total = event.normalization_nodes_total,
                normalization_exact_term_count = event.normalization_exact_term_count,
                normalization_bounded_fold_count = event.normalization_bounded_fold_count,
                normalization_relation_candidates = event.normalization_relation_candidates,
                normalization_relations_applied = event.normalization_relations_applied,
                normalization_relations_remaining = event.normalization_relations_remaining,
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
    let (noise_bound_log2, noise_threshold_log2) = acceptance_log2(&report);
    info!(
        elapsed = ?evaluation_started.elapsed(),
        accepted = report.accepted,
        noise_bound = %report.noise_bound,
        noise_bound_log2 = (noise_bound_log2 * 10.0).round() / 10.0,
        noise_threshold_log2 = (noise_threshold_log2 * 10.0).round() / 10.0,
        excess_log2 = ((noise_bound_log2 - noise_threshold_log2) * 10.0).round() / 10.0,
        "evaluated Tall parameter request with Rust operational checker"
    );
    Ok(report)
}

fn prepare_candidate(
    parameters: DCRTPolyParams,
    config: &TestConfig,
    achieved_security_bits: u64,
    preparation: CandidatePreparation,
) -> Result<PreparedCandidate, String> {
    let ring_dimension = parameters.ring_dimension() as usize;
    let encoding_ring_dimension = config.encoding_ring_dimension(ring_dimension)?;
    let (q_moduli, _, dcrt_crt_depth) = parameters.to_crt();
    let encoding_crt_depth = config.encoding_crt_depth(dcrt_crt_depth)?;
    let p_modulus_bits = match config.p_moduli_bits {
        Some(bits) => bits,
        None => minimum_p_moduli_bits(
            *q_moduli.iter().max().expect("CRT basis is nonempty"),
            config.max_unreduced_muls,
        )
        .ok_or_else(|| "no nested-RNS p-modulus basis supports the selected q basis".to_owned())?,
    };
    let CircuitBundle { circuit, nested } = build_modq_multiplication_circuit(
        &parameters,
        config,
        encoding_ring_dimension,
        encoding_crt_depth,
        p_modulus_bits,
    );
    let scalar_circuit = build_modq_multiplication_circuit(
        &parameters,
        config,
        1,
        encoding_crt_depth,
        p_modulus_bits,
    );
    if gate_kind_counts(&circuit) != gate_kind_counts(&scalar_circuit.circuit) {
        return Err("nested-RNS gate counts depend on the coefficient slot count".to_owned());
    }
    let preprocessing_graph_started = Instant::now();
    log_graph_phase("preprocessing_construction", "start", None);
    let physical_slots = encoding_ring_dimension
        .checked_mul(encoding_crt_depth)
        .ok_or_else(|| "physical slot count exceeds usize".to_owned())?;
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
    let public_keys = BggPublicKeySampler { layout: layout.clone() }.sample(
        hash_key.clone(),
        b"tall-nested-rns-input-public-keys".as_slice(),
        &vec![true; circuit.num_input()],
    );
    let diagonal_mask_public_key = BggPublicKeySampler { layout: layout.clone() }
        .sample(hash_key.clone(), b"tall-nested-rns-diagonal-mask".as_slice(), &[true])
        .into_iter()
        .next()
        .expect("one diagonal-mask public key");
    let public_key_compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: gadget_base.clone().into(),
        digit_count: parameters.modulus_digits().into(),
    };
    let circuit_compiler = PolyCircuitCompiler { public_key: public_key_compiler.clone() };
    let lookup_trapdoor = ring.sample_trapdoor(
        1,
        RealExpr::from_f64_exact(config.trapdoor_sigma).map_err(|error| error.to_string())?,
        gadget_base.clone(),
        parameters.modulus_digits(),
        preimage_max_coefficient_bound.clone(),
    );
    let mut lookup_preprocessing = LweLookupPreprocessingLowering::new(
        parameters.clone(),
        hash_key.clone(),
        lookup_trapdoor.clone(),
        gadget_base.clone().into(),
        parameters.modulus_digits().into(),
        Vec::new(),
    );
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
        .preprocess(hash_key.clone(), &rotation_offsets)
        .map_err(|error| error.to_string())?;
    let mut slot_preprocessing = BggTallSlotPublicKeyLowering {
        compiler: public_key_compiler.clone(),
        diagonal_mask_public_key: diagonal_mask_public_key.clone(),
        configured_slot_count: physical_slots,
        rotations: rotations.rotations.clone(),
    };
    let output_public_key = circuit_compiler
        .compile_public_keys_with_lowerings(
            &circuit,
            public_keys[0].clone(),
            public_keys.iter().skip(1).cloned(),
            &mut lookup_preprocessing,
            &mut slot_preprocessing,
        )
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "Tall preprocessing public-key circuit has no output".to_owned())?;
    let lookup_entries = lookup_preprocessing.into_entries();
    let mut lookup_preimage_start = 0usize;
    for (lookup_index, entry) in lookup_entries.iter().enumerate() {
        let table_length = entry.compiler.table_length();
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
        lookup_entries.iter().map(|entry| entry.compiler.table_length()).sum::<usize>();
    let lookup_compilers =
        lookup_entries.iter().map(|entry| entry.compiler.clone()).collect::<Vec<_>>();
    let preprocessing_preimage_count = lookup_preimage_count;
    info!(
        lookup_tables = lookup_entries.len(),
        lookup_preimage_count,
        total_preimage_count = preprocessing_preimage_count,
        "lookup preprocessing preimage plan"
    );
    let mut preprocessing_context = DslContext::new("gpu-tall-nested-rns-preprocessing")
        .public_output(LOOKUP_PUBLIC_B_ARTIFACT, lookup_trapdoor.public_matrix())
        .map_err(|error| error.to_string())?;
    for (index, public_key) in public_keys.iter().enumerate() {
        preprocessing_context = preprocessing_context
            .public_output(format!("{INPUT_PUBLIC_KEY_PREFIX}_{index}"), public_key.matrix.clone())
            .map_err(|error| error.to_string())?;
    }
    preprocessing_context = preprocessing_context
        .public_output(DIAGONAL_MASK_PUBLIC_KEY_ARTIFACT, diagonal_mask_public_key.matrix.clone())
        .map_err(|error| error.to_string())?;
    preprocessing_context = preprocessing_context
        .public_output(OUTPUT_PUBLIC_KEY_ARTIFACT, output_public_key.matrix)
        .map_err(|error| error.to_string())?;
    for entry in &lookup_entries {
        preprocessing_context =
            entry.export(preprocessing_context).map_err(|error| error.to_string())?;
    }
    preprocessing_context = rotation_compiler
        .export_preprocessing(preprocessing_context, rotations)
        .map_err(|error| error.to_string())?;
    let preprocessing = preprocessing_context.build().map_err(|error| error.to_string())?;
    let preprocessing_graph_construction = preprocessing_graph_started.elapsed();
    log_graph_phase("preprocessing_construction", "end", Some(&preprocessing_graph_started));
    let bindings = ParamEnv::default();
    let preprocessing_validate_started = Instant::now();
    log_graph_phase("preprocessing_validate", "start", None);
    let validated_preprocessing =
        preprocessing.validate(&bindings).map_err(|error| error.to_string())?;
    log_graph_phase("preprocessing_validate", "end", Some(&preprocessing_validate_started));

    let spec_hash_started = Instant::now();
    log_graph_phase("spec_hash", "start", None);
    let preprocessing_spec_hash =
        spec_hash(&preprocessing.graph, &bindings).map_err(|error| error.to_string())?;
    log_graph_phase("spec_hash", "end", Some(&spec_hash_started));
    let production = production_id(preprocessing_spec_hash, [0x71; 32]);

    let manifest_export_started = Instant::now();
    log_graph_phase("manifest_export", "start", None);
    let runtime_manifest = export_validated_manifest(production.clone(), &validated_preprocessing)
        .map_err(|error| error.to_string())?;
    log_graph_phase("manifest_export", "end", Some(&manifest_export_started));
    validate_tall_preprocessing_manifest(
        &runtime_manifest,
        circuit.num_input(),
        &lookup_compilers,
        rotation_keys,
    )?;
    drop(validated_preprocessing);
    drop(lookup_entries);
    log_graph_phase("validated_and_lookup_drop", "end", None);
    info!(
        ring_dimension,
        encoding_ring_dimension,
        encoding_crt_depth,
        crt_depth = parameters.to_crt().2,
        p_modulus_bits,
        physical_slots,
        gate_counts = ?gate_kind_counts(&circuit),
        artifact_count = runtime_manifest.artifacts.len(),
        "constructed Tall nested-RNS candidate graphs"
    );
    debug!(q_moduli = ?parameters.to_crt().0, "candidate CRT moduli");
    let encoding_construction_started = Instant::now();
    log_graph_phase("encoding_construction", "start", None);
    let encoding_graph = build_encoding_graph(
        &parameters,
        &circuit,
        &layout,
        production.clone(),
        &lookup_compilers,
        &rotation_offsets,
        physical_slots,
        encoding_crt_depth,
        config.error_sigma,
        true,
    )?;
    log_graph_phase("encoding_construction", "end", Some(&encoding_construction_started));
    let manifests = BTreeMap::from([(production.clone(), runtime_manifest.clone())]);
    let encoding_validate_started = Instant::now();
    log_graph_phase("encoding_validate", "start", None);
    encoding_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    log_graph_phase("encoding_validate", "end", Some(&encoding_validate_started));
    let operational_report = if preparation == CandidatePreparation::OperationalChecked {
        Some(run_tall_operational_check(&preprocessing, &encoding_graph, &parameters, &nested)?)
    } else {
        None
    };
    Ok(PreparedCandidate {
        parameters,
        encoding_ring_dimension,
        encoding_crt_depth,
        physical_slots,
        circuit,
        nested,
        layout,
        preprocessing,
        preprocessing_graph_construction,
        lookup_preimage_count,
        preprocessing_preimage_count,
        production,
        runtime_manifest,
        lookup_compilers,
        rotation_offsets,
        encoding_graph,
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

fn validate_tall_preprocessing_manifest(
    manifest: &RuntimeManifest,
    input_count: usize,
    lookup_compilers: &[mxx_bgg::LweLookupCompiler],
    rotations: impl IntoIterator<Item = TallRotationEncodingKey>,
) -> Result<(), String> {
    let forbidden = [
        "slot_transfer",
        "slot_reduce",
        "slot_secret",
        "transformed_secret",
        "lookup_c_b",
        "c_forward",
        "c_backward",
    ];
    if let Some(name) = manifest
        .artifacts
        .keys()
        .find(|name| forbidden.iter().any(|fragment| name.contains(fragment)))
    {
        return Err(format!("Tall preprocessing exported forbidden legacy artifact {name}"));
    }
    let mut required = vec![
        LOOKUP_PUBLIC_B_ARTIFACT.to_owned(),
        DIAGONAL_MASK_PUBLIC_KEY_ARTIFACT.to_owned(),
        OUTPUT_PUBLIC_KEY_ARTIFACT.to_owned(),
    ];
    required.extend((0..input_count).map(|index| format!("{INPUT_PUBLIC_KEY_PREFIX}_{index}")));
    for compiler in lookup_compilers {
        let names = LweLookupArtifactNames::for_compiler(compiler);
        required.extend([
            names.output_public_key,
            names.low_matrices,
            names.high_matrices,
            names.output_plaintexts,
        ]);
    }
    for key in rotations {
        let names = TallRotationEncodingArtifactNames::for_key(key);
        required.extend([names.a_forward, names.a_backward]);
    }
    if let Some(name) = required.into_iter().find(|name| !manifest.artifacts.contains_key(name)) {
        return Err(format!("Tall preprocessing manifest omits required public artifact {name}"));
    }
    if let Some((name, _)) = manifest
        .artifacts
        .iter()
        .find(|(_, descriptor)| descriptor.confidentiality != ArtifactConfidentiality::Public)
    {
        return Err(format!("Tall preprocessing exports non-public artifact {name}"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_encoding_graph(
    parameters: &DCRTPolyParams,
    circuit: &PolyCircuit<DCRTPoly>,
    layout: &BggSamplerLayout,
    production: ProductionId,
    lookup_compilers: &[mxx_bgg::LweLookupCompiler],
    rotation_offsets: &[u32],
    physical_slots: usize,
    encoding_crt_depth: usize,
    error_sigma: f64,
    include_operational_residual: bool,
) -> Result<BuiltGraph, String> {
    let ring = layout.ring();
    let public_keys =
        imported_public_keys(&ring, &production, circuit.num_input(), layout.public_key_columns());
    let plaintexts = (0..circuit.num_input())
        .map(|input| ring.input_family(format!("plaintext_{input}"), physical_slots, (1, 1)))
        .collect::<Vec<_>>();
    // The online protocol owns exactly one fresh Tall secret matrix.  The
    // family below contains row views of that one `physical_slots × n` sample,
    // not independently sampled per-slot secrets.
    let tall_secret = ring.uniform_interval((physical_slots, layout.secret_dimension), -1, 1);
    let secret_rows = Parallel::range(physical_slots)
        .map_values(move |slot| {
            let start = slot.expression();
            tall_secret.clone().slice(
                Some(IndexRange {
                    end: IntExpr::Add(Box::new(start.clone()), Box::new(IntExpr::constant(1))),
                    start,
                }),
                None,
            )
        })
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
    .sample(secret_rows.clone(), &public_keys, &plaintexts, physical_slots.into())
    .map_err(|error| error.to_string())?;
    let invocations = bind_lwe_lookup_invocations(
        parameters,
        circuit,
        production.clone(),
        lookup_compilers.iter().cloned(),
    )
    .map_err(|error| error.to_string())?;
    let lookup_public_b = ring.artifact_input(
        production.clone(),
        LOOKUP_PUBLIC_B_ARTIFACT,
        (layout.secret_dimension, layout.secret_dimension * (layout.digit_count + 2)),
        ArtifactConfidentiality::Public,
    );
    let lookup_error_sigma =
        RealExpr::from_f64_exact(error_sigma).map_err(|error| error.to_string())?;
    let lookup_error_bound: IntExpr = BigInt::from(hard_cutoff_from_sigma_bound(
        &BigDecimal::from_f64(error_sigma)
            .ok_or_else(|| "error sigma must be finite".to_owned())?,
    ))
    .into();
    let lookup_ring = ring.clone();
    let lookup_columns = layout.secret_dimension * (layout.digit_count + 2);
    let lookup_public_b_for_c_b = lookup_public_b.clone();
    let lookup_c_b = secret_rows
        .clone()
        .parallel_map(move |_, secret_row| {
            secret_row * lookup_public_b_for_c_b.clone() +
                lookup_ring.gaussian(
                    (1, lookup_columns),
                    lookup_error_sigma.clone(),
                    lookup_error_bound.clone(),
                )
        })
        .map_err(|error| error.to_string())?;
    let mut lookup = LweLookupTallEncodingLowering::new(invocations, lookup_c_b)
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
    let secret_gadget_rows = rotation_compiler
        .secret_gadget_rows(secret_rows.clone())
        .map_err(|error| error.to_string())?;
    let mut rotations = BTreeMap::new();
    for offset in rotation_offsets {
        if let Some(public) = rotation_compiler
            .import_artifacts(&rotation_artifacts, *offset)
            .map_err(|error| error.to_string())?
        {
            let rotation = rotation_compiler
                .encode(&public, secret_rows.clone(), secret_gadget_rows.clone())
                .map_err(|error| error.to_string())?;
            rotations.insert(rotation.key, rotation);
        }
    }
    let public_key_compiler = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: layout.gadget_base.clone(),
        digit_count: layout.digit_count.into(),
    };
    let diagonal_mask_public_key = BggPublicKeyWire {
        matrix: ring.artifact_input(
            production.clone(),
            DIAGONAL_MASK_PUBLIC_KEY_ARTIFACT,
            (layout.secret_dimension, layout.public_key_columns()),
            ArtifactConfidentiality::Public,
        ),
        reveal_plaintext: true,
    };
    let mut slots = BggTallSlotLowering {
        compiler: BggTallEncodingCompiler { public_key: public_key_compiler.clone() },
        diagonal_mask_public_key,
        secret_rows: secret_rows.clone(),
        sampler: BggTallEncodingSampler {
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
        },
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
    let (q_moduli, _, dcrt_crt_depth) = parameters.to_crt();
    if encoding_crt_depth == 0 || encoding_crt_depth > dcrt_crt_depth {
        return Err(format!(
            "encoding CRT depth {encoding_crt_depth} is outside DCRT CRT depth {dcrt_crt_depth}"
        ));
    }
    if physical_slots % encoding_crt_depth != 0 {
        return Err(format!(
            "physical slot count {physical_slots} is not coefficient-major for encoding CRT depth {encoding_crt_depth}"
        ));
    }
    let anchor_count = physical_slots / encoding_crt_depth;
    // Keep the anchor selection as one generated gather family. A packed list of static gets
    // lowers to an opaque explicit family and prevents the checker from beta-reducing the source
    // row expression; this generated index map retains one shared mapped-index authority.
    let anchor_index_family = Parallel::range(anchor_count)
        .map_values(|index| index.as_int().mul(Int::constant(encoding_crt_depth)))
        .map_err(|error| error.to_string())?;
    let encoding_rows = output
        .rows
        .clone()
        .parallel_gather(anchor_index_family.clone())
        .map_err(|error| error.to_string())?;
    let output_plaintexts = output_plaintexts
        .clone()
        .parallel_gather(anchor_index_family.clone())
        .map_err(|error| error.to_string())?;
    let residual_secret_rows = secret_rows
        .clone()
        .parallel_gather(anchor_index_family)
        .map_err(|error| error.to_string())?;
    let output_public_key = ring.artifact_input(
        production.clone(),
        OUTPUT_PUBLIC_KEY_ARTIFACT,
        (layout.secret_dimension, layout.public_key_columns()),
        ArtifactConfidentiality::Public,
    );
    let mut context = DslContext::new("gpu-tall-nested-rns-encoding")
        .family_output("encoding_rows", encoding_rows.clone())
        .map_err(|error| error.to_string())?
        .family_output("output_plaintexts", output_plaintexts.clone())
        .map_err(|error| error.to_string())?;
    if include_operational_residual {
        let gadget =
            ring.gadget(layout.secret_dimension, layout.gadget_base.clone(), layout.digit_count);
        let public_key = output_public_key.clone();
        let residuals = parallel_zip(
            (encoding_rows.clone(), output_plaintexts.clone(), residual_secret_rows),
            move |_, (encoding, plaintext, secret_row)| {
                let signal = secret_row.clone() * public_key.clone() -
                    plaintext * (secret_row * gadget.clone());
                encoding - signal
            },
        )
        .map_err(|error| error.to_string())?;
        // The operational target contains only the authoritative q1 anchors.
        let decoder_input = residuals
            .get_static(0)
            .slice(
                Some(IndexRange { start: 0.into(), end: 1.into() }),
                Some(IndexRange { start: 0.into(), end: 1.into() }),
            )
            .semantic_anchor(TALL_DECODER_RESIDUAL_ANCHOR)
            .map_err(|error| error.to_string())?;
        let q_max = q_moduli.into_iter().max().expect("nonempty CRT basis");
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
        encoding_ring_dimension = ?config.encoding_ring_dimension,
        encoding_crt_depth = ?config.encoding_crt_depth,
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
                    let candidate = prepare_candidate(
                        parameters,
                        config,
                        achieved_security_bits,
                        CandidatePreparation::OperationalChecked,
                    )?;
                    Ok(Some(candidate))
                })
                .collect::<Vec<_>>()
        });
        for ((crt_depth, log_ring_dimension), result) in batch.iter().zip(batch_results) {
            let Some(candidate) = result? else {
                continue;
            };
            let operational_report = candidate
                .operational_report
                .as_ref()
                .ok_or_else(|| "operational candidate omitted its checker report".to_owned())?;
            let accepted = operational_report.accepted;
            let (noise_bound_log2, noise_threshold_log2) = acceptance_log2(operational_report);
            info!(
                crt_depth = *crt_depth,
                ring_dimension = 1usize << *log_ring_dimension,
                encoding_ring_dimension = candidate.encoding_ring_dimension,
                encoding_crt_depth = candidate.encoding_crt_depth,
                physical_slots = candidate.physical_slots,
                operational_noise_bound = %operational_report.noise_bound,
                noise_bound_log2 = (noise_bound_log2 * 10.0).round() / 10.0,
                noise_threshold_log2 = (noise_threshold_log2 * 10.0).round() / 10.0,
                excess_log2 = ((noise_bound_log2 - noise_threshold_log2) * 10.0).round() / 10.0,
                diagnostics = ?operational_report.diagnostics,
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

fn prepare_selected_benchmark_candidate(config: &TestConfig) -> Result<PreparedCandidate, String> {
    let (crt_depth, log_ring_dimension, parameters) =
        selected_cpu_parameters(config, "selected-parameter Tall benchmark")?;
    let achieved_security_bits = if config.security_bits == 0 {
        0
    } else {
        lattice_security_bits(&parameters, config.error_sigma)?
    };
    if achieved_security_bits < config.security_bits {
        return Err(format!(
            "selected Tall benchmark parameters provide {achieved_security_bits} security bits, below the required {}",
            config.security_bits
        ));
    }
    info!(
        crt_depth,
        log_ring_dimension,
        ring_dimension = parameters.ring_dimension(),
        achieved_security_bits,
        required_security_bits = config.security_bits,
        "preparing previously accepted selected parameters without rerunning operational simulation"
    );
    prepare_candidate(parameters, config, achieved_security_bits, CandidatePreparation::RuntimeOnly)
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
) -> Result<(CostReport, CostReport), String> {
    info!("stage 2/4: benchmark estimation");
    let bindings = ParamEnv::default();
    let manifests =
        BTreeMap::from([(selected.production.clone(), selected.runtime_manifest.clone())]);
    let preprocessing_graph =
        selected.preprocessing.validate(&bindings).map_err(|error| error.to_string())?;
    let encoding_graph = selected
        .encoding_graph
        .validate_with_manifests(&bindings, &manifests)
        .map_err(|error| error.to_string())?;
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: config.benchmark_warmups,
        measured_iterations: config.benchmark_iterations,
        memory_poll_interval: Duration::from_millis(1),
    };
    let encoding_parallel_instances = config.max_parallel_instances;
    let preprocessing_parallel_instances = config.preprocessing_parallel_instances;
    let encoding_column_wave_size = encoding_parallel_instances.div_ceil(device_ids.len());
    let preprocessing_column_wave_size =
        preprocessing_parallel_instances.div_ceil(device_ids.len());
    let estimator_config =
        EstimateConfig { device_pool_size: encoding_parallel_instances, per_instance_occupancy: 1 };
    let preprocessing_estimator_config = EstimateConfig {
        device_pool_size: preprocessing_parallel_instances,
        per_instance_occupancy: 1,
    };
    info!(
        gpu_count = device_ids.len(),
        measurement_workers = device_ids.len(),
        encoding_column_wave_size,
        preprocessing_column_wave_size,
        encoding_parallel_instances,
        preprocessing_parallel_instances,
        "effective benchmark estimator parallelism"
    );
    let backends = device_ids
        .iter()
        .copied()
        .map(|device_id| (gpu_backend_on([gpu_parameters.clone()], [device_id]), device_id))
        .collect();
    let mut backend = GpuNodeMeasurementBackend::new(
        backends,
        harness,
        selected.parameters.to_crt().2,
        preprocessing_column_wave_size,
    );
    info!("collecting unique GPU measurement shapes");
    estimate(&preprocessing_graph, &mut backend, &preprocessing_estimator_config)
        .map_err(|error| error.to_string())?;
    backend.set_column_wave_size(encoding_column_wave_size);
    estimate(&encoding_graph, &mut backend, &estimator_config)
        .map_err(|error| error.to_string())?;
    let measurement_started = Instant::now();
    backend.measure_collected().map_err(|error| error.to_string())?;
    info!(
        elapsed = ?measurement_started.elapsed(),
        gpu_count = device_ids.len(),
        "parallel GPU measurement collection complete"
    );
    let preprocessing_started = Instant::now();
    info!(subgraph = "preprocessing", "benchmark subgraph estimation begin");
    backend.set_column_wave_size(preprocessing_column_wave_size);
    let preprocessing_report =
        estimate(&preprocessing_graph, &mut backend, &preprocessing_estimator_config)
            .map_err(|error| error.to_string())?;
    info!(subgraph = "preprocessing", elapsed = ?preprocessing_started.elapsed(), "benchmark subgraph estimation complete");
    info!(
        lookup_preimage_count = selected.lookup_preimage_count,
        total_preimage_count = selected.preprocessing_preimage_count,
        "estimated lookup preimage sampling"
    );
    log_cost_report("TallBggPreprocessing", &preprocessing_report);
    let encoding_started = Instant::now();
    info!(subgraph = "encoding", "benchmark subgraph estimation begin");
    backend.set_column_wave_size(encoding_column_wave_size);
    let encoding_report = estimate(&encoding_graph, &mut backend, &estimator_config)
        .map_err(|error| error.to_string())?;
    info!(subgraph = "encoding", elapsed = ?encoding_started.elapsed(), "benchmark subgraph estimation complete");
    log_cost_report("TallBggEncoding", &encoding_report);
    Ok((preprocessing_report, encoding_report))
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
    // Keep a complete ambient-ring evaluation vector for the independent DCRT oracle. The Tall
    // input encoder below intentionally consumes only the configured active prefix.
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
) -> Result<BTreeMap<String, RuntimeValue<GpuDcrtBackend>>, String> {
    let encoded = operands
        .par_iter()
        .map(|evaluation_slots| {
            // DCRT evaluation multiplication is pointwise, so the active prefix can be checked
            // directly against the corresponding prefix of the full-ring GPU oracle.
            let active_slots = evaluation_slots
                .get(..selected.encoding_ring_dimension)
                .ok_or_else(|| {
                    format!(
                        "DCRT operand has {} evaluation slots, fewer than the configured encoding ring dimension {}",
                        evaluation_slots.len(),
                        selected.encoding_ring_dimension
                    )
                })?;
            Ok(encode_nested_rns_poly::<DCRTPoly>(
                selected.nested.p_moduli_bits,
                selected.nested.max_unreduced_muls,
                &selected.parameters,
                active_slots,
                CrtWindow::new(
                    0,
                    selected.encoding_crt_depth,
                    selected.nested.q_moduli_depth,
                ),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if encoded.len() != selected.circuit.num_input() {
        return Err(format!(
            "nested-RNS encoding produced {} circuit inputs, expected {}",
            encoded.len(),
            selected.circuit.num_input()
        ));
    }
    let mut inputs = BTreeMap::new();
    let matrices = encoded
        .into_par_iter()
        .enumerate()
        .map(|(input, lanes)| {
            let rows = lanes
                .into_iter()
                .map(|value| vec![DCRTPoly::from_biguint_to_constant(&selected.parameters, value)])
                .collect::<Vec<_>>();
            let cpu = DCRTPolyMatrix::from_poly_vec(&selected.parameters, rows);
            (
                format!("plaintext_{input}"),
                RuntimeValue::IndexedFamily(
                    (0..cpu.row_size())
                        .map(|slot| {
                            RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                                gpu_parameters,
                                &cpu.slice(slot, slot + 1, 0, 1),
                            ))
                        })
                        .collect(),
                ),
            )
        })
        .collect::<Vec<_>>();
    for (name, family) in matrices {
        inputs.insert(name, family);
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
    let (mut store, manifest) = if let Some(checkpoint_path) = &config.reuse_checkpoint {
        let started = Instant::now();
        let (store, manifest, _) = reload_preprocessing(checkpoint_path)?;
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
        (store, manifest)
    } else {
        let mut hash_key = [0u8; 32];
        rand::rng().fill(&mut hash_key);
        let started = Instant::now();
        let preprocessing =
            selected.preprocessing.validate(&bindings).map_err(|error| error.to_string())?;
        info!(elapsed = ?started.elapsed(), "timed preprocessing graph validation");

        let mut preprocessing_store = MemoryArtifactStore::default();
        // Preprocessing artifacts are serialized by the store below. Use a dedicated context so
        // the consumer starts without the preprocessing allocator pool and transient preimage
        // buffers still resident on the GPU.
        let production = {
            let mut preprocessing_backend =
                gpu_backend_on([gpu_parameters.clone()], device_ids.iter().copied());
            let started = Instant::now();
            let preprocessing_result = execute_in_session_with_config(
                &preprocessing,
                &mut preprocessing_backend,
                BTreeMap::from([(
                    HASH_KEY_INPUT.to_owned(),
                    RuntimeValue::Bytes(hash_key.to_vec()),
                )]),
                &mut preprocessing_store,
                [0x71; 32],
                producer_execution_config,
            )
            .map_err(|error| error.to_string())?;
            info!(elapsed = ?started.elapsed(), "timed preprocessing execution");
            preprocessing_result
                .production_id
                .ok_or_else(|| "preprocessing execution returned no production id".to_owned())?
        };
        if production != selected.production {
            return Err("preprocessing production id differs from the selected manifest".to_owned());
        }
        let manifest = preprocessing_store
            .manifest(&production)
            .cloned()
            .ok_or_else(|| "preprocessing manifest was not committed".to_owned())?;

        let started = Instant::now();
        let checkpoint_path =
            save_preprocessing(config, &preprocessing_store, &manifest, hash_key)?;
        info!(elapsed = ?started.elapsed(), path = %checkpoint_path.display(), "timed checkpoint serialization");
        let started = Instant::now();
        let (store, manifest, _) = reload_preprocessing(&checkpoint_path)?;
        info!(elapsed = ?started.elapsed(), path = %checkpoint_path.display(), "timed checkpoint reload");
        (store, manifest)
    };
    let manifests = BTreeMap::from([(selected.production.clone(), manifest)]);
    let mut backend = gpu_backend_on([gpu_parameters.clone()], device_ids.iter().copied());

    let started = Instant::now();
    let operands = random_operands(selected, config);
    let expected_evaluation_slots = expected_product_on_gpu(selected, gpu_parameters, &operands)?;
    let inputs = encoding_inputs(selected, gpu_parameters, &operands)?;
    info!(elapsed = ?started.elapsed(), "timed random plaintext generation, GPU oracle, and Tall input encoding");
    let started = Instant::now();
    let encoding_graph_source = build_encoding_graph(
        &selected.parameters,
        &selected.circuit,
        &selected.layout,
        selected.production.clone(),
        &selected.lookup_compilers,
        &selected.rotation_offsets,
        selected.physical_slots,
        selected.encoding_crt_depth,
        config.error_sigma,
        true,
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
    let encoding_rows =
        matrix_family_output(&mut encoding_result, "encoding_rows", &backend, &mut store)?;
    let output_plaintexts =
        matrix_family_output(&mut encoding_result, "output_plaintexts", &backend, &mut store)?;
    let residuals = matrix_family_output(
        &mut encoding_result,
        TALL_OPERATIONAL_RESIDUAL,
        &backend,
        &mut store,
    )?;
    encoding_result.cleanup_staged(&mut store).map_err(|error| error.to_string())?;
    encoding_rows.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    output_plaintexts.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    residuals.par_iter().for_each(GpuDCRTPolyMatrix::wait_until_ready);
    info!(elapsed = ?started.elapsed(), "timed TallBggEncoding end-to-end evaluation");
    let started = Instant::now();
    let _transferred_encoding_rows =
        encoding_rows.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    let _transferred_plaintexts =
        output_plaintexts.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    let _transferred_residuals =
        residuals.par_iter().map(GpuDCRTPolyMatrix::to_cpu_matrix).collect::<Vec<_>>();
    info!(elapsed = ?started.elapsed(), "timed output transfer");
    info!(elapsed = ?encoding_pass_started.elapsed(), "timed encoding-pass total");
    Ok(EndToEndOutputs { encoding_rows, output_plaintexts, residuals, expected_evaluation_slots })
}

/// Verifies the executable encoding against the independently evaluated
/// plaintext circuit and returns its largest centered encoding residual.
fn measure_runtime_residual(
    selected: &PreparedCandidate,
    gpu_parameters: &GpuDCRTPolyParams,
    outputs: EndToEndOutputs,
) -> Result<(BigUint, (usize, usize, usize)), String> {
    info!("stage 4/4: runtime verification");
    let anchor_count = selected.encoding_ring_dimension;
    if outputs.encoding_rows.len() != anchor_count ||
        outputs.output_plaintexts.len() != anchor_count ||
        outputs.residuals.len() != anchor_count
    {
        return Err("Tall output family cardinality differs from the q1 anchor count".to_owned());
    }
    let modulus = selected.parameters.modulus().as_ref().clone();
    let active_modulus = selected.parameters.to_crt().0[..selected.encoding_crt_depth]
        .iter()
        .fold(BigUint::from(1u8), |product, modulus| product * BigUint::from(*modulus));
    let half_modulus = &modulus / BigUint::from(2u8);
    let mut maximum_noise = BigUint::zero();
    let mut maximum_location = (0usize, 0usize, 0usize);
    for anchor in 0..anchor_count {
        let expected = outputs
            .expected_evaluation_slots
            .get(anchor)
            .cloned()
            .ok_or_else(|| "GPU oracle omitted an expected evaluation slot".to_owned())? %
            &active_modulus;
        let expected_poly = DCRTPoly::from_biguint_to_constant(&selected.parameters, expected);
        let expected_cpu =
            DCRTPolyMatrix::from_poly_vec_row(&selected.parameters, vec![expected_poly]);
        let expected_matrix = GpuDCRTPolyMatrix::from_cpu_matrix(gpu_parameters, &expected_cpu);
        if outputs.output_plaintexts[anchor] != expected_matrix {
            return Err(format!("runtime plaintext mismatch at q1 anchor {anchor}"));
        }
        let residual = &outputs.residuals[anchor];
        residual.wait_until_ready();
        let cpu = residual.to_cpu_matrix();
        for column in 0..cpu.col_size() {
            for (coefficient_index, value) in
                cpu.entry(0, column).coeffs_biguints().into_iter().enumerate()
            {
                let centered = if value > half_modulus { &modulus - value } else { value };
                if centered > maximum_noise {
                    maximum_noise = centered;
                    maximum_location = (anchor, column, coefficient_index);
                    debug!(
                        maximum_noise = %maximum_noise,
                        ?maximum_location,
                        "updated Tall BGG+ residual maximum"
                    );
                }
            }
        }
    }
    Ok((maximum_noise, maximum_location))
}

fn runtime_verification(
    selected: &PreparedCandidate,
    gpu_parameters: &GpuDCRTPolyParams,
    outputs: EndToEndOutputs,
) -> Result<(), String> {
    let (maximum_noise, maximum_location) =
        measure_runtime_residual(selected, gpu_parameters, outputs)?;
    let operational_report = selected
        .operational_report
        .as_ref()
        .ok_or_else(|| "runtime verification requires an operational checker report".to_owned())?;
    let q_max = selected
        .parameters
        .to_crt()
        .0
        .into_iter()
        .max()
        .ok_or_else(|| "runtime verification requires a nonempty CRT basis".to_owned())?;
    let modulus = selected.parameters.modulus().as_ref().clone();
    let threshold_lhs = BigUint::from(2u8) * BigUint::from(q_max) * &maximum_noise;
    info!(
        maximum_noise = %maximum_noise,
        ?maximum_location,
        operational_noise_bound = %operational_report.noise_bound,
        within_operational_envelope = maximum_noise <= operational_report.noise_bound,
        threshold_lhs = %threshold_lhs,
        ciphertext_modulus = %modulus,
        "measured Tall BGG+ residual"
    );
    if maximum_noise > operational_report.noise_bound {
        return Err(format!(
            "measured residual {maximum_noise} at {maximum_location:?} exceeds operational bound {}",
            operational_report.noise_bound
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
fn encoding_ring_dimension_defaults_to_dcrt_and_rejects_invalid_subdimensions() {
    let mut config = noiseless_runtime_config();
    assert_eq!(config.encoding_ring_dimension(8), Ok(8));

    config.encoding_ring_dimension = Some(4);
    assert_eq!(config.encoding_ring_dimension(8), Ok(4));

    config.encoding_ring_dimension = Some(16);
    assert!(config.encoding_ring_dimension(8).is_err());

    config.encoding_ring_dimension = Some(3);
    assert!(config.encoding_ring_dimension(8).is_err());
}

#[test]
fn candidate_search_excludes_dcrt_rings_smaller_than_the_encoding_ring() {
    let mut config = noiseless_runtime_config();
    config.selected_parameters = None;
    config.min_log_ring_dimension = 1;
    config.max_log_ring_dimension = 3;
    config.encoding_ring_dimension = Some(4);
    assert_eq!(config.candidate_dimensions(), vec![(1, 2), (1, 3)]);
}

#[test]
fn encoding_crt_depth_defaults_to_dcrt_and_rejects_invalid_subdepths() {
    let mut config = noiseless_runtime_config();
    assert_eq!(config.encoding_crt_depth(8), Ok(8));

    config.encoding_crt_depth = Some(4);
    assert_eq!(config.encoding_crt_depth(8), Ok(4));

    config.encoding_crt_depth = Some(0);
    assert!(config.encoding_crt_depth(8).is_err());

    config.encoding_crt_depth = Some(16);
    assert!(config.encoding_crt_depth(8).is_err());
}

#[test]
fn candidate_search_excludes_dcrt_depths_smaller_than_the_encoding_depth() {
    let mut config = noiseless_runtime_config();
    config.selected_parameters = None;
    config.min_crt_depth = 1;
    config.max_crt_depth = 3;
    config.encoding_crt_depth = Some(2);
    assert_eq!(config.candidate_dimensions(), vec![(2, 1), (3, 1)]);
}

#[test]
fn lookup_planning_stats_match_preprocessing_for_repeated_subcircuit() {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4);
    let digit_count = parameters.modulus_digits();
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let ring = Ring::new(modulus, parameters.ring_dimension() as usize);

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let lookup_id = circuit.register_public_lookup(
        mxx_gadgets::circuit::PublicLutProgram::new(3, mxx_gadgets::circuit::LutExpr::input())
            .expect("identity public LUT"),
    );
    let mut child = circuit.fresh_sub_circuit();
    let child_input = child.input(1).as_single_wire();
    let child_output = child.public_lookup_gate(child_input, lookup_id);
    child.output([child_output]);
    let child_id = circuit.register_sub_circuit(child);
    let inputs = circuit.input(2).to_vec();
    let first = circuit.call_sub_circuit(child_id, [inputs[0]]);
    let second = circuit.call_sub_circuit(child_id, [inputs[1]]);
    circuit.output([first[0], second[0]]);

    let planning = lookup_planning_stats(&circuit).expect("lookup planning stats");
    assert_eq!(planning, LookupPlanningStats { occurrences: 2, preimages: 6 });
    assert_eq!(
        planning.preimages,
        circuit.lut_vector_len_with_subcircuits(),
        "the recursive circuit helper must retain per-call LUT multiplicity"
    );

    let gadget_base = BigInt::from(1u64 << parameters.base_bits());
    let trapdoor = ring.sample_trapdoor(1, 5, gadget_base.clone(), digit_count, 1_000_000);
    let mut lookup = LweLookupPreprocessingLowering::new(
        parameters.clone(),
        ring.bytes_input("planning-parity-hash-key", 32),
        trapdoor,
        gadget_base.clone().into(),
        digit_count.into(),
        Vec::new(),
    );
    let mut slots = NoSlotOperations::default();
    let public_key = |name: &str| BggPublicKeyWire {
        matrix: ring.input(name, (1, digit_count)),
        reveal_plaintext: true,
    };
    PolyCircuitCompiler {
        public_key: BggPublicKeyCompiler {
            ring: ring.clone(),
            base: gadget_base.into(),
            digit_count: digit_count.into(),
        },
    }
    .compile_public_keys_with_lowerings(
        &circuit,
        public_key("planning-parity-one"),
        [public_key("planning-parity-input-0"), public_key("planning-parity-input-1")],
        &mut lookup,
        &mut slots,
    )
    .expect("public-key preprocessing lowering");
    let lowered_occurrences = lookup.entries().len();
    let lowered_preimages = lookup
        .entries()
        .iter()
        .try_fold(0usize, |total, entry| total.checked_add(entry.compiler.table_length()));
    assert_eq!(Some(planning.preimages), lowered_preimages);
    assert_eq!(planning.occurrences, lowered_occurrences);
    assert_eq!(lookup.entries()[0].compiler.identity.lookup, lookup_id);
    assert_eq!(lookup.entries()[1].compiler.identity.lookup, lookup_id);
    assert_ne!(
        lookup.entries()[0].compiler.identity.call_path,
        lookup.entries()[1].compiler.identity.call_path,
        "the two sub-circuit calls must remain distinct lookup occurrences"
    );
    assert_eq!(lookup.entries()[0].compiler.table, lookup.entries()[1].compiler.table);
}

#[test]
#[ignore = "CPU-only planning report; requires explicitly selected CRT depth and ring dimension"]
fn test_tall_bgg_nested_rns_planning_stats() -> Result<(), String> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let config = TestConfig::from_env()?;
    let (crt_depth, log_ring_dimension, parameters) =
        selected_cpu_parameters(&config, "Tall nested-RNS planning")?;
    let (q_moduli, _, actual_crt_depth) = parameters.to_crt();
    if actual_crt_depth != crt_depth || q_moduli.len() != crt_depth {
        return Err(format!(
            "DCRT parameter schema mismatch: requested depth {crt_depth}, actual depth \
             {actual_crt_depth}, q-modulus count {}",
            q_moduli.len()
        ));
    }
    let q_max = q_moduli
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "planning requires a nonempty q-modulus basis".to_owned())?;
    let p_modulus_bits = match config.p_moduli_bits {
        Some(bits) => bits,
        None => minimum_p_moduli_bits(q_max, config.max_unreduced_muls).ok_or_else(|| {
            "no nested-RNS p-modulus basis supports the selected q basis".to_owned()
        })?,
    };
    let encoding_crt_depth = config.encoding_crt_depth(actual_crt_depth)?;
    let CircuitBundle { circuit, nested } = build_modq_multiplication_circuit(
        &parameters,
        &config,
        1,
        encoding_crt_depth,
        p_modulus_bits,
    );
    let encoding_ring_dimension =
        config.encoding_ring_dimension(parameters.ring_dimension() as usize)?;
    let physical_slots = encoding_ring_dimension
        .checked_mul(encoding_crt_depth)
        .ok_or_else(|| "physical slot count exceeds usize".to_owned())?;
    let lookup_stats = lookup_planning_stats(&circuit)?;
    info!(
        planning_only = true,
        crt_depth,
        log_ring_dimension,
        ring_dimension = parameters.ring_dimension(),
        encoding_ring_dimension,
        encoding_crt_depth,
        q_moduli = ?q_moduli,
        q_modulus_bits = parameters.modulus().bits(),
        p_modulus_bits = nested.p_moduli_bits,
        p_moduli = ?nested.p_moduli,
        p_moduli_depth = nested.p_moduli.len(),
        physical_slots,
        gate_counts = ?circuit.count_gates_by_type_vec(),
        lookup_occurrences = lookup_stats.occurrences,
        lookup_preimages = lookup_stats.preimages,
        "Tall nested-RNS planning-only statistics; this is not security, noise, graph, or runtime acceptance"
    );
    Ok(())
}

#[test]
#[ignore = "CPU-only lattice estimator report; requires explicitly selected CRT depth and ring dimension"]
fn test_tall_bgg_nested_rns_security_estimation() -> Result<(), String> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let config = TestConfig::from_env()?;
    let (crt_depth, log_ring_dimension, parameters) =
        selected_cpu_parameters(&config, "Tall nested-RNS security estimation")?;
    let (q_moduli, _, actual_crt_depth) = parameters.to_crt();
    if actual_crt_depth != crt_depth || q_moduli.len() != crt_depth {
        return Err(format!(
            "DCRT parameter schema mismatch: requested depth {crt_depth}, actual depth \
             {actual_crt_depth}, q-modulus count {}",
            q_moduli.len()
        ));
    }
    let achieved_security_bits = lattice_security_bits(&parameters, config.error_sigma)?;
    info!(
        estimator_only = true,
        crt_depth,
        log_ring_dimension,
        ring_dimension = parameters.ring_dimension(),
        q_moduli = ?q_moduli,
        q_modulus_bits = parameters.modulus().bits(),
        achieved_security_bits,
        required_security_bits = config.security_bits,
        meets_requested_security = achieved_security_bits >= config.security_bits,
        "Tall nested-RNS lattice-security estimate; no circuit, graph, noise, or runtime acceptance was evaluated"
    );
    Ok(())
}

#[test]
fn noisy_modes_require_positive_error_sigma_but_zero_noise_does_not() {
    for mode in [
        TallRunMode::Simulation,
        TallRunMode::Benchmark,
        TallRunMode::BenchmarkSelected,
        TallRunMode::Full,
    ] {
        assert!(validate_error_sigma(mode, 0.0).is_err(), "{mode:?} must reject zero sigma");
        validate_error_sigma(mode, f64::MIN_POSITIVE).expect("positive sigma is valid");
    }
    validate_error_sigma(TallRunMode::ZeroNoise, 0.0)
        .expect("zero-noise mode intentionally permits zero sigma");
    validate_error_sigma(TallRunMode::Graph, 0.0)
        .expect("graph-only mode may inspect a noiseless graph");
}

fn single_lwe_public_lut_signal_check(
    residual_from_signal: impl FnOnce(mxx_dsl::Mat) -> mxx_dsl::Mat,
) -> Result<OperationalSimulationReport, OperationalSimulationError> {
    let parameters = DCRTPolyParams::new(8, 1, 20, 4);
    let digit_count = parameters.modulus_digits();
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let ring = Ring::new(modulus.clone(), parameters.ring_dimension() as usize);
    let matrix_type = |rows, columns| MatrixType {
        modulus: IntExpr::constant(modulus.clone()),
        ring_dimension: IntExpr::constant(parameters.ring_dimension()),
        rows: IntExpr::constant(rows),
        columns: IntExpr::constant(columns),
    };
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let input_gate = circuit.input(1).as_single_wire();
    let lookup_id = circuit.register_public_lookup(
        mxx_gadgets::circuit::PublicLutProgram::new(2, mxx_gadgets::circuit::LutExpr::input())
            .expect("identity public LUT"),
    );
    let output_gate = circuit.public_lookup_gate(input_gate, lookup_id);
    circuit.output([output_gate]);
    let lookup = LweLookupCompiler {
        identity: LweLookupIdentity {
            call_path: Vec::new(),
            gate: output_gate.as_single_wire().index(),
            occurrence: 0,
            lookup: lookup_id,
            slot: None,
        },
        table: LweLookupTable::from_public_lut(circuit.lookup_table(lookup_id).as_ref())
            .expect("lookup table"),
        public_key_type: matrix_type(1, digit_count),
        low_matrix_type: matrix_type(digit_count, digit_count),
        high_matrix_type: matrix_type(digit_count + 2, digit_count),
        gadget_base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
        digit_count: IntExpr::constant(digit_count),
    };
    let production = ProductionId {
        spec_hash: mxx_ir_core::artifact::SpecHash([6; 32]),
        execution_nonce: [9; 32],
    };
    let artifacts = LweLookupArtifacts::for_compiler(production, &lookup);
    let trapdoor = ring.sample_trapdoor(
        1,
        5,
        lookup.gadget_base.clone(),
        lookup.digit_count.clone(),
        1_000_000,
    );
    let preprocessing = lookup
        .preprocess(
            ring.bytes_input("single-lwe-public-lut-hash-key", 32),
            &BggPublicKeyWire { matrix: ring.zero((1, digit_count)), reveal_plaintext: true },
            &trapdoor,
        )
        .expect("LWE public-LUT preprocessing");
    let producer = lookup
        .export_preprocessing(
            DslContext::new("single-lwe-public-lut-producer"),
            preprocessing,
            &LweLookupArtifactNames::for_compiler(&lookup),
        )
        .expect("public-LUT artifacts")
        .build()
        .expect("public-LUT producer graph");
    let output = lookup
        .encoding(
            &BggEncodingWire {
                vector: ring.zero((1, digit_count)),
                pubkey: BggPublicKeyWire {
                    matrix: ring.zero((1, digit_count)),
                    reveal_plaintext: true,
                },
                plaintext: Some(ring.polynomial([IntExpr::constant(0)])),
            },
            &ring.zero((1, digit_count + 2)),
            &lookup.import_artifacts(&artifacts).expect("imported public-LUT artifacts"),
        )
        .expect("one public-LUT evaluation");
    let output_plaintext = output.plaintext.expect("public-LUT output plaintext");
    let secret = ring.uniform_interval((1, 1), -1, 1);
    let gadget = ring.gadget(1, lookup.gadget_base.clone(), lookup.digit_count.clone());
    let first_column = Some(IndexRange { start: IntExpr::constant(0), end: IntExpr::constant(1) });
    let signal = secret.clone() * output.pubkey.matrix.slice(None, first_column.clone()) -
        output_plaintext * secret * gadget.slice(None, first_column);
    let residual = residual_from_signal(signal);
    let decoded = residual
        .clone()
        .threshold_decode_bools(2, 1)
        .into_iter()
        .next()
        .expect("one threshold output")
        .semantic_anchor("single-lwe-public-lut.decoder")
        .expect("decoder anchor");
    let encoding = DslContext::new("single-lwe-public-lut-encoding")
        .private_output("residual", residual)
        .expect("residual output")
        .bool_output("decoded", decoded)
        .expect("decoder output")
        .build()
        .expect("public-LUT encoding graph");
    let decoder_node = encoding.graph.outputs()["decoded"].value.node;
    let endpoint = EndpointSpecId::ToyThresholdDecode;
    let ideal = IdealSpec::new(
        DslContext::new("single-lwe-public-lut-ideal")
            .bool_output("decoded", Bool::constant(false))
            .expect("ideal output")
            .build()
            .expect("ideal graph"),
    )
    .expect("sampler-free ideal");
    let decoder_stage = StageId("encoding".to_owned());
    let protocol = operational_protocol_from_graphs(
        vec![("producer".to_owned(), &producer), ("encoding".to_owned(), &encoding)],
        "encoding",
        &BTreeMap::new(),
        &BTreeMap::new(),
        |bundle| {
            bundle.ideal = ideal;
            bundle.comparator = ComparatorSpec::Equality {
                endpoints: vec![ComparatorEndpointBinding {
                    endpoint,
                    actual_input: "decoded".to_owned(),
                    ideal_input: "decoded".to_owned(),
                    result_output: "failure".to_owned(),
                    failure_value: true,
                }],
            };
            bundle.endpoints = EndpointAnchors {
                entries: vec![EndpointAnchor {
                    spec: endpoint,
                    stage: decoder_stage.clone(),
                    semantic_anchor: "single-lwe-public-lut.decoder".to_owned(),
                    semantics: EndpointSemanticBinding::ThresholdDecode,
                    workflow_output: OutputRef {
                        stage: decoder_stage.clone(),
                        output: "decoded".to_owned(),
                    },
                    ideal_output: "decoded".to_owned(),
                }],
            };
            bundle.operational_decoder_targets = vec![OperationalDecoderTarget {
                target_id: "single-lwe-public-lut".to_owned(),
                residual_stage: decoder_stage.clone(),
                residual_output: "residual".to_owned(),
                decoder_stage,
                decoder_node,
                kind: OperationalDecoderKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(2),
                },
            }];
            bundle.endpoint_specs = vec![endpoint];
        },
    )
    .expect("operational public-LUT protocol");
    let (crt_moduli, crt_bits, _) = parameters.to_crt();
    let base_bits = parameters.base_bits() as usize;
    let request = OperationalCheckRequest {
        environment: Vec::new(),
        layouts: vec![OperationalGadgetLayout {
            params_id: "single-lwe-public-lut".to_owned(),
            ring_dimension: parameters.ring_dimension() as usize,
            crt_moduli: crt_moduli.clone(),
            crt_bits,
            base_bits,
            base: BigInt::from(1u8) << base_bits,
            regular_digit_count: crt_bits.div_ceil(base_bits) * crt_moduli.len(),
            small_digit_count: crt_bits.div_ceil(base_bits),
            smallest_crt_modulus: *crt_moduli.iter().min().expect("CRT modulus"),
        }],
        target_id: "single-lwe-public-lut".to_owned(),
    };
    check_operational_noise_candidate_with_progress(&protocol, &request, |_| {})
}

#[test]
fn single_lwe_public_lut_raw_signal_is_an_unconsumed_large_term() {
    let error = single_lwe_public_lut_signal_check(|signal| signal)
        .expect_err("an uncancelled public-LUT signal must remain Large");
    let OperationalSimulationError::Production(_) = error else {
        panic!("raw public-LUT signal must reject as an unconsumed exact residual: {error:?}")
    };
}

#[test]
fn single_lwe_public_lut_signal_subtraction_cancels_in_the_operational_checker() {
    let report = single_lwe_public_lut_signal_check(|signal| signal.clone() - signal)
        .expect("the exact same public-LUT signal must cancel under subtraction");
    assert_eq!(report.noise_bound, BigUint::zero());
}

#[test]
fn single_lwe_public_lut_signal_add_negate_cancels_in_the_operational_checker() {
    let report = single_lwe_public_lut_signal_check(|signal| signal.clone() + -signal)
        .expect("the exact same public-LUT signal must cancel under Add + Negate");
    assert_eq!(report.noise_bound, BigUint::zero());
}

#[test]
fn range_parameters_preserve_the_cartesian_candidate_search() {
    assert_eq!(candidate_dimensions(2, 3, 4, 5, None), vec![(2, 4), (2, 5), (3, 4), (3, 5)]);
}

/// Uses the same small arithmetic parameters as the nested-RNS Ring-GSW unit
/// tests.  Setting this one value to zero reaches every additive Gaussian
/// error source in the current Tall construction: input and diagonal-mask
/// public keys, online LUT `C_B`, online rotation `C`, and sampled Tall rows.
/// The compact q1-anchor residual then checks the resulting zero-noise output.
fn noiseless_runtime_config() -> TestConfig {
    TestConfig {
        mul_count: 1,
        min_crt_depth: 1,
        max_crt_depth: 1,
        min_log_ring_dimension: 1,
        max_log_ring_dimension: 1,
        selected_parameters: Some((1, 1)),
        encoding_ring_dimension: None,
        encoding_crt_depth: None,
        security_bits: 0,
        crt_modulus_bits: 10,
        p_moduli_bits: None,
        gadget_base_bits: 5,
        max_unreduced_muls: 2,
        scale: 16,
        error_sigma: 0.0,
        // Trapdoors and preimages remain sampled.  They are not additive
        // encryption errors: each preimage satisfies its target relation
        // exactly, which this test checks through the final residual.
        trapdoor_sigma: 4.578,
        benchmark_warmups: 1,
        benchmark_iterations: 1,
        run_mode: TallRunMode::ZeroNoise,
        parameter_simulation_parallelism: 1,
        // Keep the 31,232-preimage smoke run observable without logging once
        // per preimage (about 122 quantitative progress reports).
        preimage_progress_interval: 256,
        max_parallel_instances: 1,
        preprocessing_parallel_instances: 1,
        release_fence_interval: 1,
        checkpoint_root: PathBuf::from("test_data/tall_nested_rns_noiseless_gpu"),
        reuse_checkpoint: None,
    }
}

#[test]
#[ignore = "requires a CUDA GPU; checks a small noiseless Tall BGG+ execution against an independent plaintext product"]
fn test_gpu_tall_bgg_nested_rns_noiseless_encoding_matches_ideal_product() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let config = noiseless_runtime_config();
    let parameters = DCRTPolyParams::new(2, 1, 10, 5);
    let selected = prepare_candidate(parameters, &config, 0, CandidatePreparation::RuntimeOnly)
        .expect("small noiseless Tall graphs");
    assert!(
        selected.operational_report.is_none(),
        "the executable noiseless check must not depend on checker acceptance"
    );
    let device_ids = detected_gpu_device_ids();
    assert!(!device_ids.is_empty(), "at least one CUDA GPU");
    let (moduli, _, _) = selected.parameters.to_crt();
    let gpu_parameters = GpuDCRTPolyParams::new(
        selected.parameters.ring_dimension(),
        moduli,
        selected.parameters.base_bits(),
    );
    let outputs = end_to_end_processing(&selected, &config, &gpu_parameters, &device_ids)
        .expect("small noiseless Tall execution");
    let (maximum_residual, location) =
        measure_runtime_residual(&selected, &gpu_parameters, outputs)
            .expect("Tall output plaintext must equal the independent product");
    assert_eq!(
        maximum_residual,
        BigUint::zero(),
        "noiseless Tall encoding residual must be exactly zero (first nonzero location: {location:?})"
    );
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
    let operational_report = selected
        .operational_report
        .as_ref()
        .expect("parameter simulation runs the operational checker");
    info!(
        crt_depth = selected.parameters.to_crt().2,
        ring_dimension = selected.parameters.ring_dimension(),
        encoding_ring_dimension = selected.encoding_ring_dimension,
        encoding_crt_depth = selected.encoding_crt_depth,
        physical_slots = selected.physical_slots,
        operational_noise_bound = %operational_report.noise_bound,
        operational_accepted = operational_report.accepted,
        "completed Rust-only Tall parameter simulation"
    );
}

#[test]
#[ignore = "requires a CUDA GPU and lattice-estimator-cli; runs the full Tall BGG+ round trip"]
fn test_gpu_tall_bgg_nested_rns_modq_arithmetic() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let requested_config = TestConfig::from_env().expect("valid GPU Tall nested-RNS configuration");
    let requested_mode = requested_config.run_mode;
    log_invocation(&requested_config, None);
    let (config, selected) = match requested_mode {
        TallRunMode::ZeroNoise => {
            let config = noiseless_runtime_config();
            let parameters = DCRTPolyParams::new(2, 1, 10, 5);
            let selected =
                prepare_candidate(parameters, &config, 0, CandidatePreparation::RuntimeOnly)
                    .expect("small noiseless Tall graphs");
            (config, selected)
        }
        TallRunMode::Graph => {
            let (crt_depth, log_ring_dimension) = requested_config
                .candidate_dimensions()
                .into_iter()
                .next()
                .expect("validated candidate dimensions are nonempty");
            let ring_dimension = 1u32
                .checked_shl(u32::try_from(log_ring_dimension).expect("validated ring dimension"))
                .expect("validated ring dimension");
            let parameters = DCRTPolyParams::new(
                ring_dimension,
                crt_depth,
                requested_config.crt_modulus_bits,
                u32::try_from(requested_config.gadget_base_bits).expect("validated gadget base"),
            );
            let selected = prepare_candidate(
                parameters,
                &requested_config,
                0,
                CandidatePreparation::RuntimeOnly,
            )
            .expect("Tall graph construction");
            (requested_config, selected)
        }
        TallRunMode::BenchmarkSelected => {
            let selected = prepare_selected_benchmark_candidate(&requested_config)
                .expect("previously accepted selected parameters");
            (requested_config, selected)
        }
        TallRunMode::Simulation | TallRunMode::Benchmark | TallRunMode::Full => {
            let selected = select_parameters(&requested_config).expect("parameter simulation");
            (requested_config, selected)
        }
    };
    info!(?config, ?requested_mode, "effective GPU Tall nested-RNS integration configuration");
    log_invocation(&config, Some(&selected));
    if requested_mode == TallRunMode::Graph {
        info!("completed Tall graph-only mode");
        return;
    }
    let operational_report = selected.operational_report.as_ref();
    info!(
        crt_depth = selected.parameters.to_crt().2,
        ring_dimension = selected.parameters.ring_dimension(),
        encoding_ring_dimension = selected.encoding_ring_dimension,
        encoding_crt_depth = selected.encoding_crt_depth,
        physical_slots = selected.physical_slots,
        achieved_security_bits = selected.achieved_security_bits,
        operational_noise_bound = ?operational_report.map(|report| &report.noise_bound),
        operational_accepted = ?operational_report.map(|report| report.accepted),
        "selected Tall nested-RNS parameters"
    );
    if requested_mode == TallRunMode::Simulation {
        info!("completed Tall simulation-only mode");
        return;
    }
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
    if matches!(requested_mode, TallRunMode::Benchmark | TallRunMode::BenchmarkSelected) {
        info!(
            skipped_operational_simulation = requested_mode == TallRunMode::BenchmarkSelected,
            "completed Tall benchmark-only mode before preprocessing execution"
        );
        return;
    }
    let outputs = end_to_end_processing(&selected, &config, &gpu_parameters, &device_ids)
        .expect("end-to-end processing");
    runtime_verification(&selected, &gpu_parameters, outputs).expect("runtime verification");
}
