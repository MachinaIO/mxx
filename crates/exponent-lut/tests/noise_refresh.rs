#![cfg(feature = "gpu")]

use mxx_backends::{
    GpuRuntime,
    artifact::FileArtifactStore,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_dsl::{GraphValue, GraphValueSchema};
use mxx_exponent_lut::{
    CompiledNandCircuit, ExpLayout, NandCircuit, api,
    noise_refresh::{
        pbc::{PbcGeneratedKeyLayout, PbcParameters, PbcRootSeed, generate_key_layout},
        prf::{PreparedSparseLwrPrf, SparseLwrPrfProgram},
        refresh::{RefreshDeclaration, RefreshDigitPlan, RefreshPrfLabelIndex, RefreshSlot},
        simulation::RefreshTranscriptTopology,
    },
    noise_utils::{NoiseMomentPolicy, NonnegativeRational},
    program::{FamilyRange, InputBinding},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
        RnsBggNoiseTrace, RnsBggNoiseValue, RnsBggParameters, RnsBggPlanRequest, RnsEncryptBuild,
        RnsNoiseBuild, RnsOnlineBuild, RnsPreprocessBuild, checkpoint_refreshed_output,
        resume_refreshed_output, verify_refreshed_output,
    },
    utils::{
        audit_program_at_crt_depth, env_positive_decimal, env_usize, lattice_security_bits,
        noise_model, ring_dimension_candidates, ring_dimension_stride_log2,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::One;
use serial_test::serial as sequential;
use std::collections::BTreeMap;

const NOISE_REFRESH_PREFIX: &str = "MXX_EXPONENT_LUT_NOISE_REFRESH_";

struct SelectedCandidate {
    poly_parameters: DCRTPolyParams,
    parameters: RnsBggParameters,
    prf_parameters: RnsBggParameters,
    source: CompiledNandCircuit,
    declaration: RefreshDeclaration,
    prf: SparseLwrPrfProgram,
    /// The generated sparse key of each PRF output, in output order.
    prf_keys: Vec<PbcGeneratedKeyLayout>,
    refreshed_bound: BigUint,
}

fn select_rns_candidate() -> SelectedCandidate {
    let security_bits = env_usize("MXX_EXPONENT_LUT_NOISE_REFRESH_SECURITY_BITS", 1)
        .expect("valid noise-refresh security setting");
    assert!(security_bits > 0, "MXX_EXPONENT_LUT_NOISE_REFRESH_SECURITY_BITS must be positive");
    let ring_dimensions = ring_dimension_candidates(
        "MXX_EXPONENT_LUT_NOISE_REFRESH_MIN_LOG_RING_DIMENSION",
        "MXX_EXPONENT_LUT_NOISE_REFRESH_MAX_LOG_RING_DIMENSION",
        5,
        16,
    )
    .expect("valid ring dimension candidates");
    let crt_bits =
        env_usize("MXX_EXPONENT_LUT_NOISE_REFRESH_CRT_BITS", 40).expect("valid CRT bit setting");
    assert!(crt_bits > 0, "MXX_EXPONENT_LUT_NOISE_REFRESH_CRT_BITS must be positive");
    let expected_base_bits = crt_bits.div_ceil(2);
    assert!(
        expected_base_bits <= 31,
        "MXX_EXPONENT_LUT_NOISE_REFRESH_CRT_BITS is unsupported above 62 bits"
    );
    let max_crt_depth = env_usize("MXX_EXPONENT_LUT_NOISE_REFRESH_MAX_CRT_DEPTH", 32)
        .expect("valid maximum CRT depth setting");
    assert!(max_crt_depth >= 2, "MXX_EXPONENT_LUT_NOISE_REFRESH_MAX_CRT_DEPTH must be at least 2");
    let operations_per_drop = env_usize("MXX_EXPONENT_LUT_NOISE_REFRESH_OPERATIONS_PER_DROP", 1)
        .expect("valid operations-per-drop setting");
    // Each drop removes up to this many CRT primes.
    let max_primes_per_drop = env_usize("MXX_EXPONENT_LUT_NOISE_REFRESH_MAX_PRIMES_PER_DROP", 4)
        .expect("valid maximum primes-per-drop setting");
    assert!(max_primes_per_drop > 0);
    assert!(!NOISE_REFRESH_PREFIX.is_empty());
    assert!(operations_per_drop > 0);
    let average =
        noise_model(NOISE_REFRESH_PREFIX).expect("valid noise-refresh noise-model setting");
    let encoding_error_sigma =
        env_positive_decimal("MXX_EXPONENT_LUT_NOISE_REFRESH_ERROR_SIGMA", "4")
            .expect("valid noise-refresh encoding sigma");

    for ring_dimension in ring_dimensions {
        let Some(stride_log2) = ring_dimension_stride_log2(ring_dimension, 4) else {
            continue;
        };
        let source = match NandCircuit::new(2, vec![[0, 1]], 2)
            .expect("valid source circuit")
            .compile(ExpLayout { stride_log2, width_log2: 4 }, ring_dimension as usize, 4)
        {
            Ok(source) => source,
            Err(_) => continue,
        };
        let depth_candidates = (2..=max_crt_depth).flat_map(|depth| {
            (1..=max_primes_per_drop).map(move |primes_per_drop| (depth, primes_per_drop))
        });
        for (crt_depth, primes_per_drop) in depth_candidates {
            let request = RnsBggPlanRequest {
                ring_dimension,
                encoding_error_sigma: encoding_error_sigma.clone(),
                crt_bits,
                base_bits: expected_base_bits as u32,
                terminal_depth: 2,
                max_crt_depth: crt_depth,
                terminal_value_bound: BigUint::one(),
                operations_per_drop,
                primes_per_drop,
            };
            let source_program = source.program();
            let source_bindings = RnsBggNoiseBindings::from_inputs(
                source_program,
                &vec![
                    RnsBggNoiseValue::sampled_encoding_sigma(
                        &request.encoding_error_sigma,
                        format!("noise-refresh-source-{ring_dimension}").as_bytes(),
                        &[BigInt::from(1u8)],
                    )
                    .expect("valid noise-refresh encoding sigma");
                    source_program.encryption_inputs().inputs().len()
                ],
                &vec![
                    RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() };
                    source.rhs_ids().len()
                ],
            )
            .expect("noise-refresh source bindings");
            let Ok((parameters, _trace)) = audit_program_at_crt_depth(
                &request,
                source_program,
                &source_bindings,
                &vec![BigUint::one(); source_program.outputs().len()],
                average.as_ref(),
                2,
            ) else {
                continue;
            };
            if parameters
                .levels()
                .iter()
                .any(|level| level.base_bits() != expected_base_bits as u32)
            {
                continue;
            }
            let poly_parameters = match parameters.level(0) {
                Ok(level) => level.clone(),
                Err(_) => continue,
            };
            let Ok(candidate) =
                build_candidate(poly_parameters, parameters, source.clone(), average.as_ref())
            else {
                continue;
            };
            if lattice_security_bits(
                &candidate.poly_parameters,
                candidate.parameters.encoding_error_sigma(),
            )
            .map_or(false, |bits| bits >= security_bits as u64)
            {
                return candidate;
            }
        }
    }
    panic!("no noise-refresh RNS candidate satisfies the configured audit and security target")
}

fn build_candidate(
    poly_parameters: DCRTPolyParams,
    parameters: RnsBggParameters,
    source: CompiledNandCircuit,
    average: Option<&mxx_exponent_lut::noise_refresh::simulation::AverageCaseConfig>,
) -> Result<SelectedCandidate, String> {
    let source_program = source.program();
    let source_output = source_program
        .outputs()
        .iter()
        .find_map(|operand| match operand {
            mxx_exponent_lut::program::Operand::Scalar(wire) => Some(*wire),
            mxx_exponent_lut::program::Operand::Member { .. } => None,
        })
        .ok_or_else(|| "source exponent output is missing".to_owned())?;
    let source_workload =
        RnsBggDerivedWorkload::from_program(source_program).map_err(|error| error.to_string())?;
    let source_level = source_workload.level(source_output).map_err(|error| error.to_string())?;
    let p_level = source_level
        .checked_add(1)
        .ok_or_else(|| "source output logical level overflows".to_owned())?;
    let target_level = p_level
        .checked_add(1)
        .ok_or_else(|| "refresh target logical level overflows".to_owned())?;
    let target_modulus = parameters
        .level(target_level)
        .map_err(|error| error.to_string())?
        .modulus()
        .as_ref()
        .clone();
    let target_primes = parameters
        .level(target_level)
        .map_err(|error| error.to_string())?
        .to_crt()
        .0
        .into_iter()
        .map(BigUint::from)
        .collect::<Vec<_>>();
    if target_primes.iter().product::<BigUint>() != target_modulus {
        return Err("refresh q_t product does not match the target modulus".to_owned());
    }
    let p_modulus =
        parameters.level(p_level).map_err(|error| error.to_string())?.modulus().as_ref().clone();
    if &p_modulus % &target_modulus != BigUint::from(0u8) {
        return Err("refresh p modulus is not divisible by the target modulus".to_owned());
    }
    let target_columns =
        parameters.component_columns(target_level).map_err(|error| error.to_string())?;
    let target_ring = parameters.ring(target_level).map_err(|error| error.to_string())?;
    let target_matrix = (
        (
            target_ring.input("refresh-candidate-00", (1, target_columns)),
            target_ring.input("refresh-candidate-01", (1, target_columns)),
        ),
        (
            target_ring.input("refresh-candidate-10", (1, target_columns)),
            target_ring.input("refresh-candidate-11", (1, target_columns)),
        ),
    );
    let output_schema = target_matrix.schema().wire_types();
    let topology = RefreshTranscriptTopology::new(
        target_columns,
        poly_parameters.ring_dimension() as usize,
        vec![1; target_primes.len()],
        1,
    )
    .map_err(|error| error.to_string())?;
    let labels =
        RefreshPrfLabelIndex::new([0x72; 32], &topology).map_err(|error| error.to_string())?;
    // Every PRF output evaluates the same public vector `a` in Z_Q^4 under
    // its own weight-one key, placed by its own PBC layout.
    let lwr_dimension = 4;
    let pbc_parameters =
        PbcParameters::custom(lwr_dimension, 1, 2, 2, poly_parameters.ring_dimension(), None);
    let prf_labels = labels.prf_labels().map_err(|error| error.to_string())?;
    let keys = (0..prf_labels.len())
        .map(|output| {
            let mut seed = [0x5a; 32];
            seed[..8].copy_from_slice(&(output as u64).to_le_bytes());
            generate_key_layout(&pbc_parameters, PbcRootSeed(seed), &[output % lwr_dimension])
                .map_err(|error| format!("PBC layout generation failed: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let profile = mxx_exponent_lut::noise_refresh::prf::SparseLwrPrfProfile::new(
        ExpLayout {
            stride_log2: ring_dimension_stride_log2(poly_parameters.ring_dimension(), 4)
                .ok_or_else(|| "ring dimension must be a power of two with log2 >= 4".to_owned())?,
            width_log2: 4,
        },
        2,
        lwr_dimension,
    )
    .map_err(|error| error.to_string())?;
    let prf = SparseLwrPrfProgram::new(
        profile,
        keys.iter().map(|key| key.public_layout.clone()).collect(),
        prf_labels,
    )
    .map_err(|error| error.to_string())?;
    let prepared =
        PreparedSparseLwrPrf::new(prf.clone(), [0x73; 32]).map_err(|error| error.to_string())?;
    let source_q = parameters.level(source_level).map_err(|error| error.to_string())?;
    let prf_terminal = parameters.level(p_level).map_err(|error| error.to_string())?.clone();
    // The PRF runs at q through the level after its coefficient outputs,
    // followed by p, which supplies the width of those q carriers.
    let prf_workload =
        RnsBggDerivedWorkload::from_program(prf.program()).map_err(|error| error.to_string())?;
    let prf_output_level = prf_workload
        .level(prf.ordered_outputs().next().ok_or("PRF has no output")?.1)
        .map_err(|error| error.to_string())?;
    let prf_parameters = RnsBggParameters::new(
        std::iter::repeat_n(source_q.clone(), prf_output_level + 2)
            .chain(std::iter::once(prf_terminal))
            .collect(),
        parameters.encoding_error_sigma().clone(),
        parameters.operations_per_drop(),
    )
    .map_err(|error| error.to_string())?;
    let prf_q = prf_parameters.level(prf_output_level + 1).map_err(|error| error.to_string())?;
    if prf_q.modulus() != source_q.modulus() ||
        prf_q.modulus_digits() != source_q.modulus_digits() ||
        prf_q.ring_dimension() != source_q.ring_dimension() ||
        prf_parameters
            .component_columns(prf_output_level + 1)
            .map_err(|error| error.to_string())? !=
            parameters.component_columns(source_level).map_err(|error| error.to_string())?
    {
        return Err("PRF coefficient output schema does not match source q carrier".to_owned());
    }
    let q = parameters
        .level(source_level)
        .map_err(|error| error.to_string())?
        .modulus()
        .as_ref()
        .clone();
    let p = p_modulus;
    let slots = target_primes
        .iter()
        .enumerate()
        .map(|(slot, qt)| RefreshSlot {
            slot,
            q: BigInt::from_biguint(Sign::Plus, q.clone()),
            p: BigInt::from_biguint(Sign::Plus, p.clone()),
            q_t: BigInt::from_biguint(Sign::Plus, qt.clone()),
        })
        .collect::<Vec<_>>();
    let declaration = RefreshDeclaration::new(
        source_program,
        source_output,
        output_schema,
        slots,
        RefreshDigitPlan { base: 4, mask_counts: vec![1; target_primes.len()], fresh_count: 1 },
        labels,
        prepared,
        target_level,
        NoiseMomentPolicy::Continuous,
    )
    .map_err(|error| error.to_string())?;

    let source_noise_bindings = RnsBggNoiseBindings::from_inputs(
        source_program,
        &source_program
            .encryption_inputs()
            .inputs()
            .iter()
            .enumerate()
            .map(|(index, _)| noise_value(&parameters, format!("source-input-{index}").as_bytes()))
            .collect::<Vec<_>>(),
        &vec![
            RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() };
            source.rhs_ids().len()
        ],
    )
    .map_err(|error| error.to_string())?;
    parameters
        .audit_program(
            source_program,
            &source_noise_bindings,
            &vec![BigUint::one(); source_program.outputs().len()],
            average,
        )
        .map_err(|error| error.to_string())?;

    let prf_program = prf.program();
    let mut prf_noise_inputs = BTreeMap::new();
    for (index, (wire, declaration)) in prf_program.encryption_inputs().inputs().iter().enumerate()
    {
        let value = noise_value(&prf_parameters, format!("prf-input-{index}").as_bytes());
        let binding = match declaration.shape() {
            InputBinding::Scalar(()) => InputBinding::Scalar(value),
            // Every member of an input family is a separate encryption.
            InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                element: value.independent_family_members(),
                range: FamilyRange::full(count.clone()).map_err(|error| error.to_string())?,
                max_nonzero_payloads: 1u8.into(),
            }),
        };
        prf_noise_inputs.insert(*wire, binding);
    }
    // One one-hot selector family per PRF output and block.
    let prf_noise_rhs = prf
        .selection_families()
        .iter()
        .map(|family| {
            Ok((
                family.selector,
                InputBinding::Family(RnsBggNoiseFamily {
                    element: RnsBggNoiseRhs {
                        payload_l1: 1u8.into(),
                        payload_l2_squared: 1u8.into(),
                    },
                    range: FamilyRange::full(IntExpr::constant(family.capacity))
                        .map_err(|error| error.to_string())?,
                    max_nonzero_payloads: 1u8.into(),
                }),
            ))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let prf_noise_bindings = RnsBggNoiseBindings::new(prf_noise_inputs, prf_noise_rhs)
        .map_err(|error| error.to_string())?;
    let prf_terminal_bound = prf
        .profile()
        .canonical_readout()
        .map_err(|error| error.to_string())?
        .into_iter()
        .max()
        .map(BigUint::from)
        .ok_or_else(|| "PRF canonical readout is empty".to_owned())?;
    prf_parameters
        .audit_program(
            prf_program,
            &prf_noise_bindings,
            &vec![prf_terminal_bound; prf_program.outputs().len()],
            average,
        )
        .map_err(|error| error.to_string())?;

    let refresh_noise =
        parameters.refresh_noise_parameters(&declaration).map_err(|error| error.to_string())?;
    let prf_workload =
        RnsBggDerivedWorkload::from_program(prf_program).map_err(|error| error.to_string())?;
    let mut noise_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child =
            RnsNoiseBuild::new(&prf_parameters, prf_program, &prf_workload, &prf_noise_bindings)
                .map_err(|error| error.to_string())?;
        child.bind_refresh_prf(&declaration, slot).map_err(|error| error.to_string())?;
        noise_children.insert(slot, child);
    }
    let noise_declaration = declaration.clone();
    let noise_report = api::noise_simulate(
        source_program,
        RnsNoiseBuild::new(&parameters, source_program, &source_workload, &source_noise_bindings)
            .map_err(|error| error.to_string())?,
        move |build, outputs| {
            build.with_refresh_noise_parameters(refresh_noise).expect("refresh noise parameters");
            build
                .apply_refresh(
                    0,
                    &noise_declaration,
                    (0..noise_declaration.slots().len())
                        .map(|slot| (slot, outputs[0].clone()))
                        .collect(),
                    noise_children,
                )
                .expect("source noise refresh");
            Ok(outputs)
        },
    )
    .map_err(|error| error.to_string())?;
    let refreshed_output = noise_report
        .refreshed_output
        .as_ref()
        .ok_or_else(|| "refresh simulation did not produce an output envelope".to_owned())?;
    let target_modulus = parameters
        .level(declaration.target_level())
        .map_err(|error| error.to_string())?
        .modulus()
        .as_ref()
        .clone();
    let payload_message_limit = &refreshed_output.bound.payload_l1 * 2u8;
    if target_modulus <= payload_message_limit {
        return Err(
            "refreshed output payload violates the strict q/2 message-scale bound".to_owned()
        );
    }
    let refreshed_fits = if average.is_some() {
        average_refreshed_output_fits(
            &parameters,
            &declaration,
            average.expect("average mode is present"),
            &noise_report,
            refreshed_output,
        )?
    } else {
        parameters
            .noise_fits(declaration.target_level(), &refreshed_output.bound)
            .map_err(|error| error.to_string())?
    };
    if !refreshed_fits || refreshed_output.bound.payload_l1 > BigUint::one() {
        return Err("refreshed output fails the correctness predicate".to_owned());
    }

    Ok(SelectedCandidate {
        poly_parameters,
        parameters,
        prf_parameters,
        source,
        declaration,
        prf,
        prf_keys: keys,
        refreshed_bound: refreshed_output.bound.error.clone(),
    })
}

fn average_refreshed_output_fits(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    average: &mxx_exponent_lut::noise_refresh::simulation::AverageCaseConfig,
    trace: &RnsBggNoiseTrace,
    output: &RnsBggNoiseValue,
) -> Result<bool, String> {
    let moments = output
        .moments
        .as_ref()
        .ok_or_else(|| "average refresh output is missing its moment model".to_owned())?;
    // Match RnsBggParameters::audit_program's authoritative physical event
    // budget: node_event_counts already includes ring coordinates, component
    // columns, family multiplicities, and refresh topology events.
    let events = trace.node_event_counts.values().sum::<BigUint>().max(BigUint::one());
    let mut z_squared = average.z_squared_for_events(&events).map_err(|error| error.to_string())?;
    let correction = average
        .tail_correction_bits
        .checked_mul(2)
        .ok_or_else(|| "invalid average-case tail correction".to_owned())?;
    z_squared = z_squared.scaled(&(BigUint::one() << correction));
    let modulus = parameters
        .level(declaration.target_level())
        .map_err(|error| error.to_string())?
        .modulus()
        .as_ref()
        .clone();
    let limit = NonnegativeRational::new(modulus, 4u8.into()).map_err(|error| error.to_string())?;
    Ok(moments.fits(&limit, &z_squared))
}

fn noise_value(parameters: &RnsBggParameters, source: &[u8]) -> RnsBggNoiseValue {
    RnsBggNoiseValue::sampled_encoding(parameters, source, &[BigInt::from(1u8)])
}

#[test]
#[sequential(dcrt_runtime)]
fn noise_refresh_end_to_end() {
    // This scenario is intentionally serialized because the DCRT runtime and
    // CUDA context are shared by the refresh channels.  Keep the device gate
    // before allocating any stage state so CPU-only environments exit cleanly.
    let devices = mxx_backends::poly::dcrt::gpu::detected_gpu_device_ids();
    if devices.is_empty() {
        eprintln!("skipping RNS refresh round trip: no CUDA device detected");
        return;
    }
    // Build one source NAND Program and a nested modulus chain deep enough for
    // its exponent workload; the refresh target is a product of CRT slot
    // moduli at a lower level.
    let SelectedCandidate {
        poly_parameters,
        parameters,
        prf_parameters,
        source,
        declaration,
        prf,
        prf_keys: keys,
        refreshed_bound,
    } = select_rns_candidate();
    let source_program = source.program();

    // Construct stage-local PRF bindings and secrets.  RHS values are used
    // only while preparing the PRF; the online stage receives no RHS
    // plaintexts, while the LHS plaintext projection remains public to the
    // encryption stage as required by MulPublic.
    let hash_key = [0x43; 32];
    let prf_program = prf.program();
    let prf_poly_parameters = prf_parameters.level(0).expect("PRF level zero").clone();
    let zero = DCRTPoly::const_zero(&prf_poly_parameters);
    let one = DCRTPoly::const_rotate_poly(&prf_poly_parameters, 0);
    // The shared public vector `a`, as stride-aware signed monomials,
    // followed by the encoding of zero read by dummy and padding cells.
    let a_values = (0..prf.profile().lwr_dimension)
        .map(|coordinate| {
            let dimension = prf_poly_parameters.ring_dimension() as usize;
            let exponent = (7 * coordinate + 3) % prf.profile().q().expect("Q exponent");
            let rotation = (exponent * prf.profile().layout.stride()) % (2 * dimension);
            let monomial = DCRTPoly::const_rotate_poly(&prf_poly_parameters, rotation % dimension);
            if rotation >= dimension { -monomial } else { monomial }
        })
        .chain(std::iter::once(one.clone()))
        .collect::<Vec<_>>();
    let mut prf_lhs = BTreeMap::new();
    for (&wire, declaration) in prf_program.encryption_inputs().inputs() {
        let value = match declaration.shape() {
            InputBinding::Scalar(_) => InputBinding::Scalar(one.clone()),
            InputBinding::Family(_) => InputBinding::Family(a_values.clone()),
        };
        prf_lhs.insert(wire, value);
    }
    // Each output's key selects one cell per block of its layout.
    let prf_rhs = prf
        .selection_families()
        .iter()
        .map(|family| {
            let bits = prf.selector_bits(family.output, &keys[family.output]).expect("key bits");
            let values = bits[family.bucket]
                .iter()
                .map(|bit| if *bit == 1 { one.clone() } else { zero.clone() })
                .collect();
            (family.selector, InputBinding::Family(values))
        })
        .collect::<BTreeMap<_, _>>();
    // Only after simulation succeeds do we allocate the CUDA runtime. Each
    // of the three production stages is planned on it (measuring its W/C
    // candidates, which is logged as the benchmark estimate) before it runs.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    // The refresh decodes each CRT slot in its own single-prime ring.
    let residues = parameters
        .refresh_residue_parameters(declaration.target_level())
        .expect("refresh residue parameters");
    let gpu_levels =
        parameters.distinct_levels().into_iter().cloned().chain(residues).map(|level| {
            GpuDCRTPolyParams::new(
                level.ring_dimension(),
                level.to_crt().0,
                level.base_bits(),
                None,
            )
        });
    let mut runtime = GpuRuntime::new(gpu_backend_on(gpu_levels, devices)).expect("GPU runtime");
    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = FileArtifactStore::new(directory.path()).expect("artifact store");
    // The secrets are sampled on the stage runtime and stay resident there.
    let payload = RnsEncryptBuild::new(&parameters, [0x51; 32])
        .sample_payload_secret(&mut runtime, &mut store)
        .expect("payload secret");
    let mask = RnsEncryptBuild::new(&parameters, [0x51; 32])
        .sample_mask_secret(&mut runtime, &mut store)
        .expect("mask secret");
    // The PRF schedule has a source-q carrier distinct from the outer
    // schedule's level zero.  Import the already sampled outer secrets into
    // that ring; never create an unrelated child secret relation.
    let prf_payload = payload.at_level(&prf_parameters, 0).expect("PRF payload import");
    let prf_mask = mask.at_level(&prf_parameters, 0).expect("PRF mask import");
    let source_lhs_rhs = source
        .plaintext_inputs::<DCRTPoly>(&poly_parameters, &[false, false])
        .expect("source plaintext bindings");
    let source_lhs = source_lhs_rhs.0;
    let source_rhs = source_lhs_rhs.1;
    // Prepare one child graph per CRT slot.  These children contribute PRF
    // artifacts to the owning stage graph and are never executed separately.
    let mut preprocess_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsPreprocessBuild::new(&prf_parameters, [0x51; 32]);
        child
            .bind_rhs_inputs(prf_program, &ParamEnv::default(), &prf_rhs, &prf_payload, &hash_key)
            .expect("PRF preprocess child inputs");
        child.bind_refresh_prf(&declaration, slot).expect("PRF preprocess child binding");
        preprocess_children.insert(slot, child);
    }

    let preprocess_declaration = declaration.clone();
    // Preprocess executes once and publishes two typed stage records from the
    // same named session: one for encryption and one for online evaluation.
    let (preprocess, online) = api::preprocess(
        source_program,
        &ParamEnv::default(),
        &source_rhs,
        &payload,
        &hash_key,
        RnsPreprocessBuild::new(&parameters, [0x51; 32]),
        &mut runtime,
        &mut store,
        move |build, outputs| {
            build
                .bind_refresh_declaration(&preprocess_declaration)
                .expect("source refresh declaration");
            build
                .apply_refresh(
                    0,
                    &preprocess_declaration,
                    (0..preprocess_declaration.slots().len())
                        .map(|slot| (slot, outputs[0].clone()))
                        .collect(),
                    preprocess_children,
                )
                .expect("source preprocess refresh");
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("source preprocess");

    // Encryption consumes only the public LHS plaintext projection plus the
    // payload/mask secrets.  It imports the preprocess-encrypt stage record;
    // no RHS plaintext enters this stage or the later online stage.
    let mut encrypt_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsEncryptBuild::new(&prf_parameters, [0x51; 32]);
        child
            .bind_lhs_inputs(
                &prf_program.encryption_inputs(),
                &ParamEnv::default(),
                &prf_lhs,
                &prf_payload,
                &prf_mask,
                &hash_key,
            )
            .expect("PRF encryption child inputs");
        child
            .bind_refresh_prf(&declaration, slot, prf_program)
            .expect("PRF encryption child binding");
        child.bind_preprocess_artifacts(&preprocess).expect("PRF encryption child artifacts");
        encrypt_children.insert(slot, child);
    }
    let encrypt_declaration = declaration.clone();
    let encrypted = api::encrypt(
        &source_program.encryption_inputs(),
        &ParamEnv::default(),
        &source_lhs,
        &preprocess,
        &payload,
        &mask,
        &hash_key,
        RnsEncryptBuild::new(&parameters, [0x51; 32]),
        &mut runtime,
        &mut store,
        move |build| {
            build
                .bind_refresh_declaration(&encrypt_declaration)
                .expect("source encryption declaration");
            build
                .bind_refresh_prf_children(&encrypt_declaration, encrypt_children)
                .expect("source encryption refresh children");
            Ok(())
        },
        |_| Ok(()),
    )
    .expect("source encryption");

    // Online children use only the online preprocess stage record and the
    // encryption stage record.  Refresh is bound as a graph operation here;
    // it is not materialized as a caller-owned carrier or rerun by resume.
    let mut online_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsOnlineBuild::new(&prf_parameters, [0x51; 32]);
        child
            .bind_refresh_artifacts(&declaration, slot, &online, &encrypted, &hash_key)
            .expect("PRF online child artifacts");
        child.bind_refresh_prf(&declaration, slot).expect("PRF online child binding");
        online_children.insert(slot, child);
    }
    let online_declaration = declaration.clone();
    let online_result = api::online_eval(
        source_program,
        &ParamEnv::default(),
        &online,
        &encrypted,
        &hash_key,
        RnsOnlineBuild::new(&parameters, [0x51; 32]),
        &mut runtime,
        &mut store,
        move |build, outputs| {
            build
                .apply_refresh(0, &online_declaration, online_children)
                .expect("source online refresh");
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("source online evaluation");
    // Verify the independently expected NAND output against the refreshed
    // carrier returned by online_eval before checkpointing that carrier.
    let expected = DCRTPoly::const_rotate_poly(&poly_parameters, 1);
    verify_refreshed_output(
        source_program,
        &declaration,
        &online_result,
        0,
        &expected,
        &parameters,
        &payload,
        &mask,
        &mut runtime,
        &mut store,
    )
    .expect("RNS-BGG refresh application verification");
    assert!(!online_result.outputs.is_empty(), "refresh online evaluation has no outputs");
    assert!(refreshed_bound > 0u8.into(), "refresh simulation must produce a bound");

    // The refresh result is already materialized by online execution.
    // Checkpointing imports only its key/row ports and optional public payload;
    // it does not lower or execute refresh again.  Resumption then runs the
    // built-in single-input identity continuation over that checkpoint.
    let checkpointed = checkpoint_refreshed_output(
        source_program,
        &declaration,
        &online_result,
        0,
        &online,
        &encrypted,
        &parameters,
        &hash_key,
        &mut runtime,
        &mut store,
    )
    .expect("checkpoint refreshed output");
    let resumed_parameters = RnsBggParameters::new(
        vec![parameters.level(declaration.target_level()).expect("target level").clone()],
        parameters.encoding_error_sigma().clone(),
        parameters.operations_per_drop(),
    )
    .expect("resumed RNS parameters");
    let resumed = resume_refreshed_output(
        &declaration,
        &checkpointed,
        &resumed_parameters,
        &mut runtime,
        &mut store,
    )
    .expect("resume refreshed output");
    assert!(
        resumed.outputs.keys().any(|name| name.starts_with("output-0.")),
        "identity continuation must return the checkpointed output"
    );
}
