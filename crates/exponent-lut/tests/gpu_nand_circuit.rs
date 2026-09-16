#![cfg(feature = "gpu")]

//! GPU round trip for the NAND-to-unary-LUT compiler.
//!
//! Parameter search and noise/security admission remain explicit in this test;
//! execution uses the public RNS-BGG facade so setup cannot diverge from the
//! production path.

use bigdecimal::BigDecimal;
use mxx_bench_estimator::{
    CostReport, gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig, measure_graphs,
};
use mxx_exponent_lut::{
    CompiledNandCircuit, ExpLayout, ExpectedEncoding, ExponentLut, NandCircuit,
    RuntimeConfiguration,
    advanced::CompiledNandCircuitDetails,
    noise_refresh::simulation::AverageCaseConfig,
    program::{
        Gate, LutId, ProgramBindings, ProgramFamilyBindings, ProgramGateInstance, ProgramInputId,
    },
    rns_bgg::{
        RnsBggNoiseBound, RnsBggParameters, RnsBggPlanRequest,
        noise::{RnsBggGateNoiseMaterial, RnsBggNoiseRhs, RnsBggNoiseValue},
    },
};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{
        gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
        params::DCRTPolyParams,
    },
};
use mxx_runtime::{ExecutionConfig, backend::poly::gpu::gpu_backend_on};
use num_bigint::BigUint;
use num_traits::One;
use std::{collections::BTreeMap, process::Command, time::Duration};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("integer environment value"))
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        Err(error) => panic!("invalid {name}: {error}"),
        Ok(value) => panic!("{name} must be either 0 or 1, got {value:?}"),
    }
}

/// Run the same ternary-secret/Gaussian-error model used by parameter search.
fn lattice_security_bits(parameters: &DCRTPolyParams, sigma: &BigDecimal) -> u64 {
    let output = Command::new("lattice-estimator-cli")
        .arg(parameters.ring_dimension().to_string())
        .arg(parameters.modulus().as_ref().to_string())
        .arg("--s-dist")
        .arg(r#"{"name":"Ternary"}"#)
        .arg("--e-dist")
        .arg(format!(r#"{{"name":"DiscreteGaussian","stddev":{sigma}}}"#))
        .output()
        .expect("lattice-estimator-cli and Sage must be available on PATH");
    assert!(
        output.status.success(),
        "lattice-estimator-cli failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("lattice-estimator output must be UTF-8");
    let estimates = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && line.bytes().all(|byte| byte.is_ascii_digit()))
        .collect::<Vec<_>>();
    let [estimate] = estimates.as_slice() else {
        panic!("lattice-estimator-cli must print exactly one integer estimate: {stdout}");
    };
    estimate.parse().expect("lattice security estimate must fit u64")
}

fn log_cost_report(stage: &str, report: &CostReport) {
    println!(
        "nand_cost_estimate stage={stage} total_work_seconds={} total_time_seconds={} critical_path_seconds={} peak_memory_bytes={} transmitted_bytes={} oracle_excluded=true",
        report.total_work_seconds,
        report.total_time_seconds,
        report.critical_path_seconds,
        report.peak_memory_bytes,
        report.transmitted_bytes,
    );
}

/// Build the benchmark adapter only for the duration of one observation. The
/// facade graph passed to this callback is the graph that will immediately be
/// executed; no benchmark-only graph is constructed.
fn benchmark_observer(
    parameters: RnsBggParameters,
    devices: Vec<i32>,
) -> impl FnMut(&str, &mxx_ir_core::ValidatedGraph) -> Result<(), String> {
    move |stage, graph| {
        let backends = devices
            .iter()
            .map(|device| {
                let rings = parameters.levels().iter().map(|parameter| {
                    GpuDCRTPolyParams::new(
                        parameter.ring_dimension(),
                        parameter.to_crt().0,
                        parameter.base_bits(),
                        None,
                    )
                });
                (gpu_backend_on(rings, [*device]), *device)
            })
            .collect();
        let mut estimator = GpuNodeMeasurementBackend::new(
            backends,
            MeasurementHarnessConfig {
                warm_up_iterations: env_usize("MXX_EXPONENT_LUT_BENCH_WARMUPS", 1),
                measured_iterations: env_usize("MXX_EXPONENT_LUT_BENCH_ITERATIONS", 1),
                memory_poll_interval: Duration::from_millis(1),
            },
        );
        let report = measure_graphs(&[graph], &mut estimator)
            .map_err(|error| format!("benchmark observation for {stage} failed: {error}"))?
            .into_iter()
            .next()
            .expect("one graph produces one cost report");
        log_cost_report(stage, &report);
        drop(estimator);
        Ok(())
    }
}

fn balanced_circuit(depth: usize) -> NandCircuit {
    assert!((1..=3).contains(&depth));
    let leaves = 1usize << depth;
    let mut nodes: Vec<usize> = (0..leaves).collect();
    let mut gates = Vec::with_capacity(leaves - 1);
    while nodes.len() > 1 {
        let next = nodes
            .chunks_exact(2)
            .map(|pair| {
                let index = leaves + gates.len();
                gates.push([pair[0], pair[1]]);
                index
            })
            .collect();
        nodes = next;
    }
    NandCircuit::new(leaves, gates, nodes[0]).unwrap()
}

fn oracle(circuit: &NandCircuit, inputs: &[bool]) -> bool {
    let mut values = inputs.to_vec();
    for [lhs, rhs] in circuit.gates() {
        values.push(!(values[*lhs] & values[*rhs]));
    }
    values[circuit.output()]
}

fn noise_bindings(
    compiled: &CompiledNandCircuit,
    encoding_error_bound: &BigUint,
) -> (
    BTreeMap<ProgramInputId, RnsBggNoiseValue>,
    BTreeMap<mxx_exponent_lut::program::RhsId, RnsBggNoiseRhs>,
    BTreeMap<LutId, RnsBggGateNoiseMaterial>,
    BTreeMap<ProgramGateInstance, RnsBggGateNoiseMaterial>,
) {
    let input = RnsBggNoiseValue {
        level: 0,
        bound: RnsBggNoiseBound { error: encoding_error_bound.clone(), payload_l1: 1u8.into() },
        payload_l2_squared: 1u8.into(),
        moments: Some(mxx_exponent_lut::noise_utils::NoiseMoments {
            mean_numerator: 0.into(),
            mean_denominator: BigUint::one(),
            mean_radius: mxx_exponent_lut::noise_utils::NonnegativeRational::zero(),
            variance: mxx_exponent_lut::noise_utils::NonnegativeRational::new(
                16u8.into(),
                1u8.into(),
            )
            .unwrap(),
        }),
        public: false,
        range: None,
        one_hot: false,
    };
    let inputs = BTreeMap::from([(compiled.state_input(), input)]);
    let rhs = compiled
        .rhs_ids()
        .iter()
        .copied()
        .map(|id| {
            (
                id,
                RnsBggNoiseRhs {
                    payload_l1: 1u8.into(),
                    payload_l2_squared: 1u8.into(),
                    one_hot: None,
                },
            )
        })
        .collect();
    let mut helpers = BTreeMap::new();
    let mut gate_helpers = BTreeMap::new();
    for (step, operation) in compiled.program().steps().iter().enumerate() {
        let mxx_exponent_lut::program::ProgramStep::Gate(operation) = operation else { continue };
        let material = match operation {
            Gate::MulPrivate { .. } => RnsBggGateNoiseMaterial {
                level: step,
                table: None,
                fresh_error_gain: mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
            },
            Gate::Lut { lut_id, .. } => RnsBggGateNoiseMaterial {
                level: step,
                table: Some(compiled.program().lut(*lut_id).unwrap().commitment()),
                fresh_error_gain: mxx_exponent_lut::noise_utils::RightActionNoiseGain::unit(),
            },
            _ => continue,
        };
        gate_helpers.insert(ProgramGateInstance::Gate { step }, material.clone());
        if let Gate::Lut { lut_id, .. } = operation {
            helpers.insert(*lut_id, material);
        }
    }
    (inputs, rhs, helpers, gate_helpers)
}

#[test]
fn test_gpu_nand_circuit_noise_search_and_artifact_round_trip() {
    let depth = env_usize("MXX_EXPONENT_LUT_NAND_DEPTH", 1);
    let n = env_usize("MXX_PRIMITIVE_TEST_RING_DIMENSION", 32);
    let large_parameter_run = env_flag("MXX_EXPONENT_LUT_NAND_LARGE_PARAMETER_RUN");
    assert!(
        n <= 1 << 8 || large_parameter_run,
        "ring dimensions above 2^8 require MXX_EXPONENT_LUT_NAND_LARGE_PARAMETER_RUN=1"
    );
    let operations_per_drop = env_usize("MXX_EXPONENT_LUT_NAND_OPERATIONS_PER_DROP", 2);
    assert!(operations_per_drop > 1);
    let circuit = balanced_circuit(depth);
    let rounds = 1usize.checked_shl((2 * depth) as u32).expect("round count overflow");
    let compiled = circuit.compile(ExpLayout { stride_log2: 0, width_log2: 4 }, n, rounds).unwrap();
    assert_eq!(compiled.rhs_ids().len(), rounds);
    assert_eq!(compiled.program().steps().len(), rounds * 2);
    let request = RnsBggPlanRequest {
        ring_dimension: n as u32,
        widths: (0..rounds).flat_map(|_| [8, 16]).collect(),
        encoding_error_sigma: BigDecimal::from(4u8),
        crt_bits: env_usize("MXX_EXPONENT_LUT_NAND_CRT_BITS", 32),
        terminal_depth: 1,
        max_crt_depth: env_usize("MXX_EXPONENT_LUT_NAND_MAX_CRT_DEPTH", 32),
        terminal_value_bound: BigUint::one(),
        operations_per_drop,
    };
    if !large_parameter_run {
        assert_eq!(request.crt_bits, 32);
    }
    let noiseless = env_flag("MXX_EXPONENT_LUT_NAND_NOISELESS");
    let parameter_only = env_flag("MXX_EXPONENT_LUT_NAND_PARAMETER_ONLY");
    let failure_exponent =
        u32::try_from(env_usize("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", 1)).unwrap();
    let average = match std::env::var("MXX_EXPONENT_LUT_NAND_NOISE_MODEL").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("average") => Some(AverageCaseConfig {
            failure_exponent,
            allow_average_acceptance: true,
            ..Default::default()
        }),
        Ok("worst") => None,
        Err(error) => panic!("invalid noise model: {error}"),
        Ok(value) => panic!("noise model must be `average` or `worst`, got {value:?}"),
    };
    let bound = mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(
        &request.encoding_error_sigma,
    );
    let (inputs, rhs, helpers, gate_helpers) = noise_bindings(&compiled, &bound);
    let bindings = ProgramBindings::new()
        .inputs(inputs.iter().map(|(&id, value)| (id, value.clone())))
        .rhs_values(rhs.iter().map(|(&id, value)| (id, value.clone())))
        .helpers(helpers.iter().map(|(&id, value)| (id, value.clone())))
        .gate_helpers(gate_helpers.iter().map(|(&id, value)| (id, value.clone())));
    let ranges = ProgramFamilyBindings::new();
    let output = compiled.program().outputs()[0];
    let terminal_bounds = BTreeMap::from([(output, BigUint::one())]);
    let evaluate = |depth: usize| {
        let mut candidate = request.clone();
        candidate.terminal_depth = depth;
        RnsBggParameters::plan_program(&candidate, compiled.program(), &bindings, &ranges)
            .ok()
            .and_then(|planned| {
                let parameters = planned.0;
                parameters
                    .audit_program(
                        compiled.program(),
                        &bindings,
                        &ranges,
                        &terminal_bounds,
                        average.as_ref(),
                    )
                    .ok()
                    .map(|trace| (parameters, trace))
            })
    };
    let (selected_depth, parameters, trace) = (1..=request.max_crt_depth)
        .find_map(|depth| evaluate(depth).map(|(parameters, trace)| (depth, parameters, trace)))
        .unwrap_or_else(|| panic!("no CRT depth passed the Program noise audit"));
    let required_security_bits =
        u64::try_from(env_usize("MXX_EXPONENT_LUT_NAND_SECURITY_BITS", 1)).unwrap();
    let achieved = match std::env::var("MXX_EXPONENT_LUT_NAND_PREVALIDATED_SECURITY_BITS") {
        Ok(value) => value.parse().unwrap(),
        Err(std::env::VarError::NotPresent) => {
            lattice_security_bits(parameters.level(0).unwrap(), &request.encoding_error_sigma)
        }
        Err(error) => panic!("invalid security evidence: {error}"),
    };
    assert!(achieved >= required_security_bits);
    println!(
        "nand_security_candidate ring_dimension={} crt_depth={} modulus_bits={} security_bits={} evidence={} selected_depth={selected_depth}",
        parameters.levels()[0].ring_dimension(),
        parameters.levels()[0].crt_depth(),
        parameters.levels()[0].modulus().as_ref().bits(),
        achieved,
        match std::env::var("MXX_EXPONENT_LUT_NAND_PREVALIDATED_SECURITY_BITS") {
            Ok(_) => "prevalidated",
            Err(std::env::VarError::NotPresent) => "lattice-estimator-cli",
            Err(error) => panic!("invalid prevalidated security bits: {error}"),
        },
    );
    println!(
        "nand_noise_audit model={} failure_exponent={} rounds={rounds} cadence={} crt_depth={} base_bits={} events={} bound={} lattice_security_bits={} required_lattice_security_bits={}",
        if average.is_some() { "average-case" } else { "worst-case" },
        average.map_or(0, |config| config.failure_exponent),
        request.operations_per_drop,
        parameters.levels()[0].crt_depth(),
        parameters.levels()[0].base_bits(),
        trace.joint_event_count,
        trace.wires[&output].bound.error,
        achieved,
        required_security_bits,
    );
    if parameter_only {
        return;
    }

    let assignment = (0..circuit.input_count()).map(|_| rand::random::<bool>()).collect::<Vec<_>>();
    let expected = oracle(&circuit, &assignment);
    let devices = detected_gpu_device_ids();
    assert!(!devices.is_empty(), "NAND test requires a detected GPU");
    let rings = parameters.levels().iter().map(|p| {
        mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams::new(
            p.ring_dimension(),
            p.to_crt().0,
            p.base_bits(),
            None,
        )
    });
    let backend = gpu_backend_on(rings, devices.iter().copied());
    let context = rand::random();
    let benchmark_stage = std::env::var("MXX_EXPONENT_LUT_BENCH_STAGE").unwrap_or_default();
    if benchmark_stage != "" && benchmark_stage != "estimate" && benchmark_stage != "graph" {
        panic!(
            "MXX_EXPONENT_LUT_BENCH_STAGE must be `graph` or `estimate`, got {benchmark_stage:?}"
        );
    }
    let runtime = RuntimeConfiguration::new(backend, ExecutionConfig::default());
    let graph_only_error = "graph-only benchmark observation requested";
    let runtime = match benchmark_stage.as_str() {
        "estimate" => {
            runtime.with_graph_observer(benchmark_observer(parameters.clone(), devices.clone()))
        }
        "graph" => runtime.with_graph_observer(move |stage, graph| {
            println!(
                "nand_graph_stage={stage} facade_production_graph=true graph_name={}",
                graph.source.name()
            );
            Err(graph_only_error.into())
        }),
        "" => runtime,
        _ => unreachable!("benchmark stage was validated above"),
    };
    #[cfg(feature = "test-utils")]
    let engine = if noiseless {
        ExponentLut::new_noiseless_for_testing(parameters, context, runtime)
    } else {
        ExponentLut::new(parameters, context, runtime)
    };
    #[cfg(not(feature = "test-utils"))]
    let engine = {
        if noiseless {
            panic!("MXX_EXPONENT_LUT_NAND_NOISELESS=1 requires feature `test-utils`");
        }
        ExponentLut::new(parameters, context, runtime)
    };
    let secrets = compiled.sample_secrets(&engine).unwrap();
    let base = std::env::var_os("MXX_EXPONENT_LUT_NAND_ARTIFACT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("mxx-exponent-lut-{:032x}", rand::random::<u128>()))
        });
    std::fs::create_dir_all(&base).unwrap();
    let prepared = match engine.preprocess(
        compiled.program(),
        compiled.fixed_rhs(&assignment).unwrap(),
        secrets.payload(),
        base.join("enc"),
        base.join("eval"),
    ) {
        Ok(prepared) => prepared,
        Err(error)
            if benchmark_stage == "graph" && error.to_string().contains(graph_only_error) =>
        {
            return;
        }
        Err(error) => panic!("facade preprocessing failed: {error}"),
    };
    let encrypted =
        engine.encrypt(&prepared.encryption, secrets.mask(), base.join("input")).unwrap();
    let outputs = engine.online_eval(compiled.program(), &prepared.evaluation, &encrypted).unwrap();
    let output = compiled.output(&outputs).unwrap();
    engine
        .assert_encoding(
            output,
            ExpectedEncoding::Exponent(BigUint::from(expected as u8).into()),
            &secrets,
        )
        .unwrap();
    if benchmark_stage == "estimate" {
        println!("nand_benchmark_stage=estimate");
        return;
    }
    println!(
        "nand_round_trip_ok=true strict_q_over_4=true expected_exponent={expected} stride_log2=0 readout_applied=false"
    );
}
