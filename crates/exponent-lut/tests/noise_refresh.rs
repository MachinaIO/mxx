#![cfg(feature = "gpu")]

use mxx_dsl::{DslContext, GraphValue, GraphValueSchema, Ring};
use mxx_exponent_lut::{
    ExpLayout, NandCircuit, api,
    noise_refresh::{
        pbc::{PbcParameters, PbcRootSeed, generate_key_layout},
        prf::{PreparedSparseLwrPrf, SparseLwrPrfProgram},
        refresh::{RefreshDigitPlan, RefreshPrfLabelIndex, RefreshSlot},
        simulation::{RefreshRouteNoiseGains, RefreshTranscriptTopology},
    },
    noise_utils::{NoiseMomentPolicy, RightActionNoiseGain},
    program::{FamilyRange, InputBinding},
    rns_bgg::{
        RnsBggDerivedWorkload, RnsBggNoiseBindings, RnsBggNoiseFamily, RnsBggNoiseRhs,
        RnsBggNoiseValue, RnsBggParameters, RnsEncryptBuild, RnsNoiseBuild, RnsOnlineBuild,
        RnsPreprocessBuild,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv, types::WireType};
use mxx_primitives::{
    matrix::PolyMatrix,
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    Backend, ExecutionConfig, RuntimeValue,
    artifact::FileArtifactStore,
    backend::poly::gpu::{GpuDcrtBackend, gpu_backend_on},
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, Zero};
use serial_test::serial as sequential;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

fn nested_rns_parameters() -> (DCRTPolyParams, RnsBggParameters) {
    let pool = DCRTPolyParams::new(32, 4, 10, 5, None, None);
    let primes = pool.to_crt().0;
    let levels = [4usize, 3, 2]
        .into_iter()
        .map(|depth| {
            let modulus =
                primes.iter().take(depth).fold(BigUint::one(), |product, prime| product * *prime);
            pool.select_modulus(&modulus).expect("nested RNS level")
        })
        .collect();
    let parameters = RnsBggParameters::new(levels, bigdecimal::BigDecimal::from(1u8), 1)
        .expect("RNS parameters");
    (pool, parameters)
}

fn noise_value() -> RnsBggNoiseValue {
    RnsBggNoiseValue::public(6u8.into(), &[BigInt::from(1u8)])
}

#[test]
#[sequential(dcrt_runtime)]
fn noise_refresh_end_to_end() {
    let (poly_parameters, parameters) = nested_rns_parameters();
    let source = NandCircuit::new(2, vec![[0, 1]], 2)
        .expect("valid source circuit")
        .compile(ExpLayout { stride_log2: 0, width_log2: 4 }, 32, 4)
        .expect("source program");
    let source_program = source.program();
    let source_output = source_program
        .outputs()
        .iter()
        .find_map(|operand| match operand {
            mxx_exponent_lut::program::Operand::Scalar(wire) => Some(*wire),
            mxx_exponent_lut::program::Operand::Member { .. } => None,
        })
        .expect("source exponent output");

    let target_modulus = parameters.level(2).expect("target level").modulus().as_ref().clone();
    let target_primes = parameters
        .level(2)
        .expect("target level")
        .to_crt()
        .0
        .into_iter()
        .map(BigUint::from)
        .collect::<Vec<_>>();
    let target_columns = parameters.component_columns(2).expect("target columns");
    let target_ring = Ring::new(
        BigInt::from_biguint(Sign::Plus, target_modulus.clone()),
        poly_parameters.ring_dimension() as usize,
    );
    let target_matrix = (
        (
            target_ring.input("refresh-schema-00", (1, target_columns)),
            target_ring.input("refresh-schema-01", (1, target_columns)),
        ),
        (
            target_ring.input("refresh-schema-10", (1, target_columns)),
            target_ring.input("refresh-schema-11", (1, target_columns)),
        ),
    );
    let output_schema = target_matrix.schema().wire_types();
    assert_eq!(output_schema.len(), 4, "RNS-BGG public matrix has four ports");
    assert!(output_schema.iter().all(|wire_type| matches!(
        wire_type,
        WireType::Matrix(matrix)
            if matrix.rows == 1.into() && matrix.columns == target_columns.into()
    )));
    let topology = RefreshTranscriptTopology::new(
        target_columns,
        poly_parameters.ring_dimension() as usize,
        vec![1; target_primes.len()],
        1,
    )
    .expect("refresh topology");
    let labels = RefreshPrfLabelIndex::new([0x72; 32], &topology).expect("refresh labels");
    let pbc_parameters = PbcParameters::custom(4, 1, 2, 2, 32, None);
    let generated = (0..32u8)
        .find_map(|seed| generate_key_layout(&pbc_parameters, PbcRootSeed([seed; 32]), &[0]).ok())
        .expect("PBC layout");
    let profile = mxx_exponent_lut::noise_refresh::prf::SparseLwrPrfProfile::new(
        ExpLayout { stride_log2: 1, width_log2: 4 },
        2,
    )
    .expect("native PRF profile");
    let prf_labels = labels.prf_labels().expect("PRF labels");
    let prf = SparseLwrPrfProgram::new_structural(
        profile,
        generated.public_layout.bucket_width,
        generated.public_layout.parameters.bucket_count,
        b"noise-refresh-prf",
        &prf_labels,
    )
    .expect("structural PRF program");
    let prepared = PreparedSparseLwrPrf::new(prf.clone(), &generated.public_layout, [0x73; 32])
        .expect("prepared PRF");
    let q = parameters.level(0).expect("source level").modulus().as_ref().clone();
    let p = parameters.level(1).expect("switch level").modulus().as_ref().clone();
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
    let declaration = mxx_exponent_lut::noise_refresh::refresh::RefreshDeclaration::new(
        source_program,
        source_output,
        output_schema,
        slots,
        RefreshDigitPlan { base: 2, mask_counts: vec![1; target_primes.len()], fresh_count: 1 },
        labels,
        prepared,
        2,
        NoiseMomentPolicy::Continuous,
    )
    .expect("refresh declaration");

    let hash_key = [0x43; 32];
    let prf_program = prf.program();
    let d_width = prf.profile().transform_column_count().expect("D width");
    let selector_width = prf.block_count() * prf.bucket_width();
    let zero = DCRTPoly::const_zero(&poly_parameters);
    let one = DCRTPoly::const_rotate_poly(&poly_parameters, 0);
    let d_values = (0..d_width)
        .map(|index| DCRTPoly::const_rotate_poly(&poly_parameters, index))
        .collect::<Vec<_>>();
    let mut prf_lhs = BTreeMap::new();
    for (&wire, declaration) in prf_program.encryption_inputs().inputs() {
        let value = if wire == prf.d0_input() {
            InputBinding::Family(d_values.clone())
        } else {
            match declaration.shape() {
                InputBinding::Scalar(_) => InputBinding::Scalar(one.clone()),
                InputBinding::Family(_) => InputBinding::Family(d_values.clone()),
            }
        };
        prf_lhs.insert(wire, value);
    }
    let mut selector_values = vec![zero; selector_width];
    for bucket in 0..prf.block_count() {
        selector_values[bucket * prf.bucket_width()] =
            DCRTPoly::const_rotate_poly(&poly_parameters, 0);
    }
    let prf_rhs = BTreeMap::from([(
        prf.selection_families()[0].selector,
        InputBinding::Family(selector_values),
    )]);
    let payload = RnsEncryptBuild::<mxx_runtime::backend::poly::gpu::GpuDcrtBackend>::new(
        &parameters,
        [0x51; 32],
    )
    .sample_payload_secret()
    .expect("payload secret");
    let mask = RnsEncryptBuild::<mxx_runtime::backend::poly::gpu::GpuDcrtBackend>::new(
        &parameters,
        [0x51; 32],
    )
    .sample_mask_secret()
    .expect("mask secret");

    let input_noise = std::array::from_fn::<_, 3, _>(|_| noise_value());
    let source_noise_inputs = source_program
        .encryption_inputs()
        .inputs()
        .keys()
        .map(|wire| (*wire, InputBinding::Scalar(input_noise[0].clone())))
        .collect();
    let source_noise_rhs = source
        .rhs_ids()
        .iter()
        .copied()
        .map(|id| {
            (
                id,
                InputBinding::Scalar(RnsBggNoiseRhs {
                    payload_l1: 1u8.into(),
                    payload_l2_squared: 1u8.into(),
                }),
            )
        })
        .collect();
    let source_noise_bindings = RnsBggNoiseBindings::new(source_noise_inputs, source_noise_rhs)
        .expect("source noise bindings");
    let mut prf_noise_inputs = BTreeMap::new();
    for ((wire, declaration), value) in
        prf_program.encryption_inputs().inputs().iter().zip(input_noise)
    {
        let binding = match declaration.shape() {
            InputBinding::Scalar(()) => InputBinding::Scalar(value),
            InputBinding::Family(count) => InputBinding::Family(RnsBggNoiseFamily {
                element: value,
                range: FamilyRange::full(count.clone()).expect("input family range"),
                max_nonzero_payloads: 1u8.into(),
            }),
        };
        prf_noise_inputs.insert(*wire, binding);
    }
    let prf_noise_rhs = BTreeMap::from([(
        prf.selection_families()[0].selector,
        InputBinding::Family(RnsBggNoiseFamily {
            element: RnsBggNoiseRhs { payload_l1: 1u8.into(), payload_l2_squared: 1u8.into() },
            range: FamilyRange::full(IntExpr::constant(selector_width)).expect("selector range"),
            max_nonzero_payloads: 1u8.into(),
        }),
    )]);
    let prf_noise_bindings =
        RnsBggNoiseBindings::new(prf_noise_inputs, prf_noise_rhs).expect("PRF noise bindings");

    let devices = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids();
    assert!(!devices.is_empty(), "RNS refresh requires a CUDA device");
    let gpu_levels = parameters.levels().iter().map(|level| {
        GpuDCRTPolyParams::new(level.ring_dimension(), level.to_crt().0, level.base_bits(), None)
    });
    let mut backend = gpu_backend_on(gpu_levels, devices);
    let source_lhs_rhs = source
        .plaintext_inputs::<DCRTPoly>(&poly_parameters, &[false, false])
        .expect("source plaintext bindings");
    let source_lhs = source_lhs_rhs.0;
    let source_rhs = source_lhs_rhs.1;
    let mut preprocess_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsPreprocessBuild::new(&parameters, [0x51; 32]);
        api::PreprocessBuild::bind_inputs(
            &mut child,
            prf_program,
            &ParamEnv::default(),
            &prf_rhs,
            &payload,
            &hash_key,
        )
        .expect("PRF preprocess child inputs");
        child.bind_refresh_prf(&declaration, slot).expect("PRF preprocess child binding");
        preprocess_children.insert(slot, child);
    }

    let directory = tempfile::tempdir().expect("artifact directory");
    let mut store = FileArtifactStore::new(directory.path()).expect("artifact store");
    let preprocess_encrypt = directory.path().join("preprocess-encrypt.json");
    let preprocess_online = directory.path().join("preprocess-online.json");
    let encryption = directory.path().join("encryption.json");
    let config = ExecutionConfig::default();
    let preprocess_declaration = declaration.clone();
    let (preprocess, online) = api::preprocess(
        source_program,
        &ParamEnv::default(),
        &source_rhs,
        &payload,
        &hash_key,
        RnsPreprocessBuild::new(&parameters, [0x51; 32]),
        &mut backend,
        &mut store,
        [&preprocess_encrypt, &preprocess_online],
        config.clone(),
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

    let mut encrypt_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsEncryptBuild::new(&parameters, [0x51; 32]);
        api::EncryptBuild::bind_inputs(
            &mut child,
            &prf_program.encryption_inputs(),
            &prf_lhs,
            &payload,
            &mask,
        )
        .expect("PRF encryption child inputs");
        <RnsEncryptBuild<'_, mxx_runtime::backend::poly::gpu::GpuDcrtBackend> as api::EncryptBuild<
            mxx_runtime::backend::poly::gpu::GpuDcrtBackend,
            DCRTPoly,
        >>::bind_artifacts(&mut child, &preprocess)
            .expect("PRF encryption child artifacts");
        child
            .bind_refresh_prf(&declaration, slot, prf_program)
            .expect("PRF encryption child binding");
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
        RnsEncryptBuild::new(&parameters, [0x51; 32]),
        &mut backend,
        api::ArtifactDestination { store: &mut store, checkpoint: &encryption },
        config.clone(),
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

    let mut online_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child = RnsOnlineBuild::new(&parameters, [0x51; 32]);
        child
            .bind_refresh_artifacts(&declaration, slot, &preprocess, &encrypted)
            .expect("PRF online child artifacts");
        child.bind_refresh_prf(&declaration, slot).expect("PRF online child binding");
        online_children.insert(slot, child);
    }
    let online_declaration = declaration.clone();
    let refreshed_carrier = Rc::new(RefCell::new(None));
    let refreshed_carrier_slot = Rc::clone(&refreshed_carrier);
    let refresh_checkpoint_graph = Rc::new(RefCell::new(None));
    let refresh_checkpoint_graph_slot = Rc::clone(&refresh_checkpoint_graph);
    let mut online_result = api::online_eval(
        source_program,
        &ParamEnv::default(),
        &online,
        &encrypted,
        RnsOnlineBuild::new(&parameters, [0x51; 32]),
        &mut backend,
        &mut store,
        config,
        move |build, mut outputs| {
            let refreshed = build
                .compose_refresh(0, &online_declaration, online_children, &mut outputs)
                .expect("source online refresh");
            let target = build
                .refreshed_public_key()
                .cloned()
                .expect("refresh must expose its matching target key");
            let checkpoint_graph = build
                .build_refresh_checkpoint_graph::<GpuDcrtBackend>(
                    &online_declaration,
                    refreshed.clone(),
                    target,
                )
                .expect("refresh checkpoint graph");
            *refreshed_carrier_slot.borrow_mut() = Some(refreshed);
            *refresh_checkpoint_graph_slot.borrow_mut() = Some(checkpoint_graph);
            Ok(outputs)
        },
        |_| Ok(()),
    )
    .expect("source online evaluation");
    assert!(
        refreshed_carrier.borrow().is_some(),
        "online compose must return the refreshed encoding carrier"
    );

    let (checkpoint_graph, checkpoint_descriptor) = refresh_checkpoint_graph
        .borrow_mut()
        .take()
        .expect("online compose must produce a refresh checkpoint graph");
    let refresh_checkpoint_path = directory.path().join("refresh-checkpoint.json");
    let refresh_checkpoint = api::execute_checkpoint(
        checkpoint_graph,
        &ParamEnv::default(),
        checkpoint_descriptor,
        *declaration.commitment(),
        &mut backend,
        &mut store,
        &refresh_checkpoint_path,
        ExecutionConfig::default(),
        |_| Ok(()),
    )
    .expect("refresh checkpoint execution");
    let reopened = refresh_checkpoint
        .checkpoint()
        .reopen_in_store(&mut store)
        .expect("refresh checkpoint reopen");
    assert_eq!(reopened.expected_identity(), refresh_checkpoint.checkpoint().expected_identity());

    let continuation = NandCircuit::new(1, Vec::new(), 0)
        .expect("identity continuation circuit")
        .compile(ExpLayout { stride_log2: 0, width_log2: 4 }, 32, 4)
        .expect("identity continuation program");
    let continuation_parameters = RnsBggParameters::new(
        vec![parameters.level(2).expect("target level").clone()],
        bigdecimal::BigDecimal::from(1u8),
        1,
    )
    .expect("target-chain parameters");
    let mut resumed = RnsOnlineBuild::new(&continuation_parameters, [0x51; 32]);
    resumed
        .bind_refreshed_input(
            continuation.program(),
            continuation.state_input(),
            &declaration,
            &refresh_checkpoint,
        )
        .expect("bind refreshed input to identity continuation");
    let imported_encoding = resumed
        .refreshed_encoding()
        .cloned()
        .expect("resumption must import the refreshed encoding");
    let public_key = resumed
        .refreshed_public_key()
        .cloned()
        .expect("resumption must import the refreshed target");
    let expected_plaintext = target_ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
    let residual = imported_encoding
        .verification_residual(
            &continuation_parameters,
            0,
            &public_key,
            &payload,
            &mask,
            expected_plaintext,
        )
        .expect("compute caller-owned verification residual");
    let residual_graph = DslContext::new("rns-bgg-refresh-verification")
        .artifact_output("verification-residual.0", residual.0)
        .expect("verification residual graph port")
        .artifact_output("verification-residual.1", residual.1)
        .expect("verification residual graph port")
        .build()
        .expect("verification residual graph");
    let residual_graph = api::StageGraph::new(
        residual_graph,
        BTreeMap::new(),
        refresh_checkpoint.checkpoint().catalog().clone(),
    );
    let mut residual_result = api::execute_graph(
        residual_graph,
        &ParamEnv::default(),
        &mut backend,
        &mut store,
        ExecutionConfig::default(),
        |_| Ok(()),
    )
    .expect("verification residual execution");
    let mut read_residual = |name: &str| {
        let value = residual_result
            .materialize_output(name, &mut backend, &mut store)
            .expect("materialize verification residual");
        let RuntimeValue::Matrix(matrix) = value else {
            panic!("verification residual output must be a matrix");
        };
        let (rows, columns) = matrix.size();
        let cpu_matrix =
            backend.gather_matrix_for_host(&matrix).expect("read verification residual matrix");
        let modulus = BigInt::from_biguint(Sign::Plus, target_modulus.clone());
        let half_modulus = &modulus / 2u8;
        let norms = (0..rows)
            .flat_map(|row| (0..columns).map(move |column| (row, column)))
            .flat_map(|(row, column)| {
                cpu_matrix
                    .entry(row, column)
                    .coeffs_biguints()
                    .into_iter()
                    .map(|coefficient| {
                        let coefficient = BigInt::from_biguint(Sign::Plus, coefficient);
                        if coefficient > half_modulus {
                            coefficient - &modulus
                        } else {
                            coefficient
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(norms.len(), rows * columns * poly_parameters.ring_dimension() as usize);
        let decoded = backend
            .threshold_decode(
                &matrix,
                &BigInt::from(2u8),
                poly_parameters.ring_dimension() as usize,
            )
            .expect("decode verification residual");
        assert!(
            decoded.iter().all(Zero::is_zero),
            "verification residual decodes to a nonzero plaintext"
        );
        norms
    };
    let residual_norms =
        [read_residual("verification-residual.0"), read_residual("verification-residual.1")];
    let residual_norms = residual_norms.into_iter().flatten().collect::<Vec<_>>();
    let resumed_graph = <RnsOnlineBuild<'_> as api::OnlineBuild<GpuDcrtBackend>>::build_graph(
        resumed,
        vec![imported_encoding],
    )
    .expect("identity continuation graph");
    let mut continuation_result = api::execute_graph(
        resumed_graph,
        &ParamEnv::default(),
        &mut backend,
        &mut store,
        ExecutionConfig::default(),
        |_| Ok(()),
    )
    .expect("identity continuation execution");
    let online_outputs = ["output-0.0", "output-0.1"].map(|name| {
        let value = online_result
            .materialize_output(name, &mut backend, &mut store)
            .expect("materialize refreshed online output");
        let RuntimeValue::Matrix(matrix) = value else {
            panic!("refreshed online output must be a matrix");
        };
        backend.matrix_to_bytes(&matrix).expect("read refreshed online output")
    });
    let continuation_outputs = ["output-0.0", "output-0.1"].map(|name| {
        let value = continuation_result
            .materialize_output(name, &mut backend, &mut store)
            .expect("materialize continuation output");
        let RuntimeValue::Matrix(matrix) = value else {
            panic!("continuation output must be a matrix");
        };
        backend.matrix_to_bytes(&matrix).expect("read continuation output")
    });
    assert_eq!(
        continuation_outputs, online_outputs,
        "identity continuation changed refreshed encoding"
    );

    let mut refresh_noise =
        mxx_exponent_lut::noise_refresh::simulation::RefreshNoiseParameters::new(
            target_primes.clone(),
            vec![p.clone(); target_primes.len()],
            2u8.into(),
            vec![1; target_primes.len()],
            1,
        )
        .expect("refresh noise schema")
        .with_global_modulus(q)
        .expect("refresh global modulus")
        .with_transcript_topology(topology)
        .expect("refresh noise topology")
        .with_route_gains(RefreshRouteNoiseGains {
            state_scaling: vec![RightActionNoiseGain::unit(); target_primes.len()],
            mask: vec![RightActionNoiseGain::unit(); target_primes.len()],
            fresh: vec![RightActionNoiseGain::unit(); target_primes.len()],
            decoder: vec![RightActionNoiseGain::unit(); target_primes.len()],
        })
        .expect("refresh route gains");
    refresh_noise.route_bias_bounds = vec![BigUint::zero(); target_primes.len()];
    let source_workload =
        RnsBggDerivedWorkload::from_program(source_program).expect("source noise workload");
    let prf_workload =
        RnsBggDerivedWorkload::from_program(prf_program).expect("PRF noise workload");
    let mut noise_children = BTreeMap::new();
    for slot in 0..declaration.slots().len() {
        let mut child =
            RnsNoiseBuild::new(&parameters, prf_program, &prf_workload, &prf_noise_bindings)
                .expect("PRF noise child");
        child.bind_refresh_prf(&declaration, slot).expect("PRF noise child binding");
        noise_children.insert(slot, child);
    }
    let noise_declaration = declaration.clone();
    let noise_report = api::noise_simulate(
        source_program,
        RnsNoiseBuild::new(&parameters, source_program, &source_workload, &source_noise_bindings)
            .expect("source noise builder"),
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
    .expect("source noise simulation");
    assert_eq!(noise_report.refreshed_declaration_commitment, Some(*declaration.commitment()));
    let refreshed_bound = noise_report
        .refreshed_output
        .as_ref()
        .expect("refresh must produce an output envelope")
        .bound
        .error
        .clone();
    assert!(residual_norms.iter().all(|norm| norm.magnitude() <= &refreshed_bound));
}
