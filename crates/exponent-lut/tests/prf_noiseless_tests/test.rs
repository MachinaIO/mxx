//! Exact production PRF accumulator/readout execution with encoding errors disabled.
use super::*;
use crate::{
    ReadoutPreimageSchema,
    noise_refresh::pbc::{
        PbcCell, PbcParameters, PbcRootSeed, clear_pbc_inner_product, generate_key_layout,
    },
};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_runtime::{
    ExecutionConfig, RuntimeValue,
    artifact::{ArtifactStore, FileArtifactStore},
    backend::poly::cpu_backend,
    execute_in_session, execute_with_config,
    transcript::SamplingMode,
};
use rand::Rng;

#[test]
fn test_lwr_prf_noiseless_round_trip() {
    let env = |suffix: &str, default: usize| {
        std::env::var(format!("MXX_EXPONENT_LUT_REFRESH_{suffix}"))
            .map(|v| v.parse::<usize>().expect("integer parameter"))
            .unwrap_or(default)
    };
    let log_n = env("MIN_LOG_RING_DIMENSION", 2);
    let n = 1usize << log_n;
    let q_native = env("SPARSE_LWR_MODULUS", 8);
    let p = env("SPARSE_LWR_OUTPUT_MODULUS", 4);
    let universe = env("SPARSE_LWR_UNIVERSE", 4);
    let base_bits = env("BASE_BITS", 4) as u32;
    let depth = env("MIN_CRT_DEPTH", 2);
    assert!(depth >= 2, "exercise at least two CRT limbs");
    assert!(q_native.is_power_of_two() && q_native >= 4 && q_native <= 2 * n);
    assert!(p.is_power_of_two() && p >= 4 && p <= q_native);
    let params = DCRTPolyParams::new(n as u32, depth, env("CRT_BITS", 20), base_bits, None, None);
    let modulus = BigInt::from(params.modulus().as_ref().clone());
    let layout = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: n.into(),
        secret_dimension: 2,
        digit_count: params.modulus_digits(),
        gadget_base: (BigInt::one() << base_bits).into(),
    };
    let width_log2 = q_native.ilog2() - 1;
    let profile = SparseLwrPrfProfile::new(
        crate::ExpLayout { stride_log2: log_n as u32 - width_log2, width_log2 },
        p / 2,
    )
    .unwrap();
    let mut rng = rand::rng();
    let support = [rng.random_range(0..universe)];
    let generated = generate_key_layout(
        &PbcParameters::custom(universe, 1, 2, 3, 1, None),
        PbcRootSeed(rng.random()),
        &support,
    )
    .unwrap();
    let public_layout = &generated.public_layout;
    let active = PbcActiveCellIndex::build(public_layout).unwrap();
    let program = SparseLwrPrfProgram::new(
        profile,
        public_layout.bucket_width,
        public_layout.parameters.bucket_count,
    )
    .unwrap();
    let ring = layout.ring();
    let columns = layout.public_key_columns();
    let sampler = ExponentLutEncodingSampler {
        layout: layout.clone(),
        exponent_layout: profile.layout,
        gaussian_sigma: None,
        gaussian_max_coefficient_bound: None,
    };
    let t = sampler.sample_secret().unwrap();
    let s = sampler.sample_secret().unwrap();
    let hash = ring.bytes_input("prf-hash", 32);
    let key_id = rng.random();
    let bits = PbcTrustedSelectorBits::from_schedule(&generated, &ring, key_id).unwrap();
    let runtime_bits = bits.runtime_binding(&ring).unwrap();
    let selectors = build_structural_selector_families(
        &sampler,
        bits.family().clone(),
        t.clone(),
        t.clone(),
        hash.clone(),
        public_layout,
        key_id,
    )
    .unwrap();
    let (public_inputs, label_rhs) = sample_setup_public_key_families(
        &mxx_bgg::BggPublicKeySampler { layout: layout.clone() },
        hash.clone(),
        active.len(),
    )
    .unwrap();
    let initial = mxx_bgg::BggPublicKeySampler { layout: layout.clone() }
        .sample(hash.clone(), b"prf-initial".as_slice(), &[false])
        .remove(1);
    let bgg_public = BggPublicKeyCompiler {
        ring: ring.clone(),
        base: layout.gadget_base.clone(),
        digit_count: layout.digit_count.into(),
    };
    let public_compiler =
        ExponentLutPublicKeyCompiler::with_layout(bgg_public.clone(), profile.layout);
    let compiler = ExponentLutEncodingCompiler::with_layout(bgg_public, profile.layout);
    let public_batch = program
        .compile_public_batch_with_preprocessing(
            &public_compiler,
            initial.clone(),
            &PublicRhsFamily::new_for_layout(selectors.gsw.clone(), public_layout.layout_id)
                .unwrap(),
            &NativePrfPublicInputs::new_for_layout(public_inputs, 1, public_layout.layout_id)
                .unwrap(),
            public_layout,
        )
        .unwrap();
    let table_values = program
        .readout
        .canonical_coefficients()
        .unwrap()
        .iter()
        .map(|v| v.to_usize().unwrap())
        .collect();
    let table =
        LutTable::unary(profile.native_width(), profile.native_width(), table_values).unwrap();
    let helpers = sampler
        .sample_native_readout_helper_families(
            t.clone(),
            t.clone(),
            hash.clone(),
            &table,
            b"prf-mask".as_slice(),
            b"prf-helper".as_slice(),
        )
        .unwrap();
    let gadget = ring.gadget(2, layout.gadget_base.clone(), layout.digit_count);
    let readout = program
        .produce_public_readout_family_artifacts(
            &public_compiler,
            public_batch.outputs.clone(),
            &helpers,
            gadget.clone(),
        )
        .unwrap();
    let public_output = readout
        .project_public_batch(
            &program.readout,
            &public_compiler.public_key,
            public_batch.outputs,
            helpers.public_masks.clone(),
        )
        .unwrap();
    let names = canonical_program_cache_names(&program, public_layout, rng.random()).unwrap();
    let mut context = DslContext::new("one-output-prf-preprocessing")
        .private_output("t", t)
        .unwrap()
        .public_output("initial", initial.matrix)
        .unwrap()
        .public_output("selectors", selectors.gsw)
        .unwrap()
        .public_output("label-rhs", label_rhs)
        .unwrap()
        .public_output("output-public", public_output)
        .unwrap()
        .public_output("public-masks", helpers.public_masks.clone())
        .unwrap()
        .public_output("readout-target", readout.target_decompositions().clone())
        .unwrap()
        .public_output("readout-switch", readout.helper_switch_decompositions().clone())
        .unwrap()
        .public_output("readout-public", readout.helper_public_decompositions().clone())
        .unwrap()
        .public_output("readout-normalization", readout.normalization_decomposition().clone())
        .unwrap();
    for entry in &names.families {
        context = context
            .public_output(
                entry.rhs.clone(),
                public_batch.cache.family_entry(entry.step, entry.body_gate).unwrap().2.clone(),
            )
            .unwrap();
    }
    let preprocessing = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let nonce = rng.random();
    let production = ProductionId {
        spec_hash: spec_hash(&preprocessing.source, &preprocessing.bindings).unwrap(),
        execution_nonce: nonce,
    };
    let manifest =
        mxx_ir_core::artifact::export_validated_manifest(production.clone(), &preprocessing)
            .unwrap();
    let imported_matrix = |name: &str, rows: usize, cols: usize, private: bool| {
        ring.artifact_input(
            production.clone(),
            name,
            (rows, cols),
            if private {
                ArtifactConfidentiality::Private
            } else {
                ArtifactConfidentiality::Public
            },
        )
    };
    let import_preimage_family = |name: &str| {
        let artifact = &manifest.artifacts[name];
        let ArtifactType::Preimage { matrix, max_coefficient_bound } = &artifact.artifact_type
        else {
            panic!("preimage schema")
        };
        ring.preimage_family_artifact_input(
            production.clone(),
            name,
            artifact.family_count.unwrap(),
            (matrix.rows, matrix.columns),
            max_coefficient_bound.clone(),
            ArtifactConfidentiality::Public,
        )
    };
    let import_matrix_family = |name: &str| {
        let artifact = &manifest.artifacts[name];
        let ArtifactType::Matrix(matrix) = &artifact.artifact_type else { panic!("matrix schema") };
        ring.family_artifact_input(
            production.clone(),
            name,
            artifact.family_count.unwrap(),
            (matrix.rows, matrix.columns),
            ArtifactConfidentiality::Public,
        )
    };
    let imported_t = imported_matrix("t", 1, 2, true);
    let imported_initial = BggPublicKeyWire {
        matrix: imported_matrix("initial", 2, columns, false),
        reveal_plaintext: false,
    };
    let initial_vector = sampler
        .sample_encodings_for_public_matrices(
            s.clone(),
            Some(imported_t.clone()),
            &[imported_initial],
            &[ring.identity(1)],
        )
        .unwrap()
        .remove(0);
    let plaintexts = ring.input_family("prf-monomials", active.len(), (1, 1));
    let vectors = (import_preimage_family("label-rhs"), plaintexts.clone())
        .parallel_zip_tuple_result(|_, (rhs, plaintext)| {
            sample_online_label_vector_from_preimage(&sampler, &s, rhs, plaintext)
                .map_err(|_| mxx_dsl::DslError::Schema)
        })
        .unwrap();
    let inputs =
        NativePrfOnlineBatchInputs::new_for_layout(vectors, plaintexts, 1, public_layout.layout_id)
            .unwrap();
    let imported_selectors = EncodingRhsFamily::new_for_layout(
        import_matrix_family("selectors"),
        public_layout.layout_id,
    )
    .unwrap();
    let mut cache = ProgramCacheBatchBindings::new(1).unwrap();
    for entry in &names.families {
        let family = import_preimage_family(&entry.rhs);
        if entry.output_indexed {
            cache.insert_family(entry.step, entry.body_gate, entry.count, family).unwrap();
        } else {
            cache.insert_shared_family(entry.step, entry.body_gate, entry.count, family).unwrap();
        }
    }
    let accumulation = program
        .compile_encoding_batch_with_preprocessed_artifacts(
            &compiler,
            initial_vector,
            &imported_selectors,
            &inputs,
            public_layout,
            &cache,
        )
        .unwrap();
    let normalization = &manifest.artifacts["readout-normalization"].artifact_type;
    let ArtifactType::Preimage { matrix, max_coefficient_bound } = normalization else {
        panic!("normalization preimage")
    };
    let schema = ReadoutPreimageSchema::new(
        layout.gadget_base.evaluate(&ParamEnv::default()).unwrap(),
        layout.digit_count,
        max_coefficient_bound.clone(),
    )
    .unwrap();
    let imported_readout = NativePrfReadoutFamilyArtifacts::new(
        import_preimage_family("readout-target"),
        schema.clone(),
        helpers.sigmas.as_ref().clone(),
        import_preimage_family("readout-switch"),
        import_preimage_family("readout-public"),
        ring.preimage_artifact_input(
            production.clone(),
            "readout-normalization",
            (matrix.rows, matrix.columns),
            max_coefficient_bound.clone(),
            ArtifactConfidentiality::Public,
        ),
        schema,
        1,
    )
    .unwrap();
    let mask_publics = import_matrix_family("public-masks");
    let masks = mask_publics
        .parallel_map(|index, matrix| {
            let sigma = profile
                .layout
                .automorphism_index_expression(
                    n,
                    IntExpr::Add(Box::new(index.expression()), Box::new(IntExpr::constant(1))),
                )
                .unwrap();
            let mask_sigma = s.clone().ring_automorphism(sigma);
            mxx_bgg::BggEncodingSampler {
                layout: layout.clone(),
                gaussian_sigma: sampler.gaussian_sigma.clone(),
                gaussian_max_coefficient_bound: sampler.gaussian_max_coefficient_bound.clone(),
            }
            .sample(
                s.clone(),
                Some(mask_sigma),
                &[
                    BggPublicKeyWire { matrix: ring.zero((2, columns)), reveal_plaintext: false },
                    BggPublicKeyWire { matrix, reveal_plaintext: false },
                ],
                &[ring.identity(1)],
            )
            .unwrap()
            .remove(1)
            .vector
        })
        .unwrap();
    let output = program
        .execute_value_readout_batch_with_mask_family(
            &compiler.bgg,
            &accumulation,
            &imported_readout,
            masks,
        )
        .unwrap()
        .vectors()
        .get_static(0);

    // The oracle uses the independent signed LWR definition, not the compiler's table.
    let g = ring.input("expected-value", (1, 1));
    let anchor =
        s * import_matrix_family("output-public").get_static(0) - g * (imported_t * gadget);
    let online = DslContext::new("noiseless-prf-online")
        .private_output("vector", output)
        .unwrap()
        .private_output("anchor", anchor)
        .unwrap()
        .build()
        .unwrap();
    validate_cache_required_online_graph(&online).unwrap();
    online
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest.clone())]),
        )
        .unwrap();
    let mut backend = cpu_backend([params.clone()]);
    let mut runtime_inputs = runtime_bits.materialize(&mut backend).unwrap();
    runtime_inputs
        .insert("prf-hash".into(), RuntimeValue::Bytes(rng.random::<[u8; 32]>().to_vec()));
    std::fs::create_dir_all("test_data/prf_noiseless_round_trip").unwrap();
    let directory = tempfile::tempdir_in("test_data/prf_noiseless_round_trip").unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    execute_in_session(&preprocessing, &mut backend, runtime_inputs, &mut store, nonce)
        .expect("production PRF preprocessing");
    drop(store);
    let mut store = FileArtifactStore::open(directory.path()).unwrap();
    let manifest = store.load_manifest(&production).unwrap();
    let online = online
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .unwrap();
    for z in 0..q_native {
        let mut public_vector =
            (0..universe).map(|_| rng.random_range(0..q_native as u64)).collect::<Vec<_>>();
        public_vector[support[0]] = z as u64;
        assert_eq!(
            clear_pbc_inner_product(
                public_layout,
                generated.private_schedule(),
                &public_vector,
                q_native as u64
            )
            .unwrap(),
            z as u64
        );
        let magnitude = BigInt::from(2 * (p * (z % (q_native / 2)) / q_native) + 1);
        let doubled = if z < q_native / 2 { magnitude } else { -magnitude };
        let g = (doubled * ((&modulus + BigInt::one()) / BigInt::from(2u8))) % &modulus;
        let g = if g < BigInt::zero() { g + &modulus } else { g };
        let expected = DCRTPolyMatrix::from_poly_vec_row(
            &params,
            vec![DCRTPoly::from_biguints(&params, &[g.to_biguint().unwrap()])],
        );
        let monomials = active
            .iter()
            .map(|(bucket, slot, _)| {
                let value = match public_layout.cells[bucket][slot] {
                    PbcCell::Real { coordinate, .. } => public_vector[coordinate] as usize,
                    PbcCell::Dummy => 0,
                    PbcCell::Padding => unreachable!(),
                };
                let exponent = value << profile.layout.stride_log2;
                let poly = DCRTPoly::const_rotate_poly(&params, exponent % n);
                let poly = if exponent / n % 2 == 0 { poly } else { -poly };
                RuntimeValue::Matrix(DCRTPolyMatrix::from_poly_vec_row(&params, vec![poly]).into())
            })
            .collect::<Vec<_>>();
        for run in 0..2 {
            let result = execute_with_config(
                &online,
                &mut backend,
                BTreeMap::from([
                    ("prf-monomials".into(), RuntimeValue::IndexedFamily(monomials.clone())),
                    ("expected-value".into(), RuntimeValue::Matrix(expected.clone().into())),
                ]),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .expect("production PRF online execution");
            let RuntimeValue::Matrix(actual) = &result.outputs["vector"] else { panic!("matrix") };
            let RuntimeValue::Matrix(expected) = &result.outputs["anchor"] else {
                panic!("matrix")
            };
            assert_eq!(actual, expected, "exact PRF round trip z={z} fresh_run={run}");
        }
    }
    println!(
        "noiseless_prf_pass N={n} depth={depth} q={modulus} base_bits={base_bits} Q={q_native} P={p} inputs={q_native} fresh_runs=2"
    );
}
