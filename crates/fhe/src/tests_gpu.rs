//! GPU execution of the production FHE graphs; all artifacts stay in memory.
use crate::{
    BgvCiphertext, BgvHybridParams, BgvParams, FheCommonParams, FheScheme,
    utils::{common, gpu::integers},
};
use mxx_backends::{
    GpuRuntime, MemoryArtifactStore, RuntimeValue,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend},
    poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams},
};
use mxx_dsl::{DslContext, Ring};
use mxx_ir_core::{ParamEnv, artifact::ArtifactAvailability};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use std::collections::BTreeMap;

#[test]
fn test_gpu_integer_family_permutation_across_waves() {
    let width = std::env::var("MXX_GPU_MAX_PARALLEL_INSTANCES")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(64);
    let count = width * 2;
    let context = DslContext::new("integer-permutation-waves");
    let input_family = context.int_family_input("values", count);
    let indices =
        mxx_dsl::Family::pack((0..count).rev().map(mxx_dsl::Int::constant).collect()).unwrap();
    let result = mxx_dsl::parallel(count, |i| Ok(input_family.at(indices.at(i)))).unwrap();
    let common = common();
    let ring = crate::utils::ring(&common.ring);
    let graph = context
        .output("result", result)
        .unwrap()
        .output("anchor", ring.zero((1, 1)))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut runtime = GpuRuntime::new(backend(&common, None)).unwrap();
    let input_values = (0..count).map(|i| i as i64).collect::<Vec<_>>();
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("values".into(), BigInt::from(0)..=BigInt::from(count - 1));
    let inputs = BTreeMap::from([("values".into(), input(&input_values))]);
    let mut plan = runtime.plan(graph, &inputs).unwrap();
    let mut store = MemoryArtifactStore::default();
    let result = runtime.execute_with_artifacts(&mut plan, inputs, &mut store, [0; 32]).unwrap();
    assert_eq!(
        runtime.download_integer_family_output(&result.output("result").unwrap()).unwrap(),
        input_values.iter().rev().map(|value| BigInt::from(*value)).collect::<Vec<_>>()
    );
}

fn gpu_parameters(common: &FheCommonParams, bgv: Option<&BgvParams>) -> Vec<GpuDCRTPolyParams> {
    if let Some(bgv) = bgv {
        return crate::utils::gpu::bgv_gpu_parameters(bgv);
    }
    let (primes, _, depth) = common.ring.to_crt();
    crate::utils::gpu::related_gpu_parameters(
        (0..depth)
            .map(|level| common.parameters_at(level).unwrap())
            .chain(primes.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap())),
    )
}

fn backend(common: &FheCommonParams, bgv: Option<&BgvParams>) -> GpuDcrtBackend {
    gpu_backend(gpu_parameters(common, bgv))
}

/// Execute once and return every output with the production id.
fn prepare_and_run(
    graph: mxx_ir_core::ValidatedGraph,
    runtime: &mut GpuRuntime,
    inputs: BTreeMap<String, RuntimeValue>,
    store: &mut MemoryArtifactStore,
) -> (BTreeMap<String, RuntimeValue>, Option<mxx_ir_core::artifact::ProductionId>) {
    let mut plan = runtime.plan(graph, &inputs).expect("prepare FHE GPU graph");
    let launches_before = plan.compiled_launch_count();
    let result = runtime
        .execute_with_artifacts(&mut plan, inputs, store, [0; 32])
        .expect("execute FHE GPU graph");
    let production_id = result.production_id.clone();
    let outputs = result.into_outputs();
    assert!(
        plan.compiled_launch_count() > launches_before,
        "FHE GPU execution must submit the compiled production path"
    );
    (outputs, production_id)
}

fn input(values: &[i64]) -> RuntimeValue {
    RuntimeValue::integer_values(values.iter().map(|v| BigInt::from(*v)).collect())
}

#[test]
fn test_gpu_compiled_matrix_product_rebinds_sources() {
    let common = common();
    let n = common.ring.ring_dimension() as usize;
    let mut runtime = GpuRuntime::new(backend(&common, None)).unwrap();
    let context = DslContext::new("compiled-product-rebinding");
    let input_values = context.int_family_input("values", n);
    let matrix = common.ring().from_coefficients(&input_values);
    let wide = mxx_dsl::Mat::concat(
        mxx_ir_core::node::ConcatAxis::Columns,
        vec![matrix.clone(), matrix.clone(), matrix.clone()],
    );
    let partial =
        wide.slice(None, Some(mxx_ir_core::node::IndexRange { start: 1.into(), end: 2.into() }));
    let rounded = partial.clone().centered_round_divide(1);
    let product = matrix.clone() * rounded.clone();
    let graph = context
        .output("source", matrix.coefficients())
        .unwrap()
        .output("partial", partial.coefficients())
        .unwrap()
        .output("rounded", rounded.coefficients())
        .unwrap()
        .output("product", product.coefficients())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut coefficients = vec![0; n];
    coefficients[0] = 1;
    let initial = BTreeMap::from([("values".into(), input(&coefficients))]);
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("values".into(), BigInt::from(-2)..=BigInt::from(3));
    let mut store = MemoryArtifactStore::default();
    let mut plan = runtime.plan(graph, &initial).unwrap();
    for coefficient in [1i64, 3, -2] {
        coefficients[0] = coefficient;
        let inputs = BTreeMap::from([("values".into(), input(&coefficients))]);
        let launches = plan.compiled_launch_count();
        let result =
            runtime.execute_with_artifacts(&mut plan, inputs, &mut store, [0; 32]).unwrap();
        let modulus = BigInt::from(common.ring.modulus().as_ref().clone());
        let mut expected = vec![BigInt::from(0); n];
        expected[0] = BigInt::from(coefficient).mod_floor(&modulus);
        let values = |name: &str| {
            runtime.download_integer_family_output(&result.output(name).unwrap()).unwrap()
        };
        for name in ["source", "partial", "rounded"] {
            assert_eq!(values(name), expected, "{name}");
        }
        expected[0] = BigInt::from(coefficient * coefficient);
        assert_eq!(values("product"), expected);
        drop(result);
        assert!(plan.compiled_launch_count() > launches);
    }
}

#[test]
fn test_gpu_fhe_bgv_simd_staged_runtime() {
    let common = common();
    let n = common.ring.ring_dimension() as usize;
    let top = common.ring.to_crt().2 - 1;
    assert!(top >= 2);
    let t = (1..).map(|k| k * 2 * n as u64 + 1).find(|&t| crate::utils::is_prime(t)).unwrap();
    let bgv = BgvParams::new(common.clone(), t, None).unwrap();
    let context = DslContext::new("gpu-bgv-encrypt");
    let slots = context.int_family_input("slots", n);
    let (secret, pk) = bgv.keygen().unwrap();
    let ct = bgv.encrypt(&pk, &slots).unwrap();
    let relin = bgv.relinearization_key(&secret, top).unwrap();
    let rotation = bgv.rotation_key(&secret, top - 1, 1).unwrap();
    let backwards = bgv.rotation_key(&secret, top - 1, -1).unwrap();
    let swap = bgv.row_swap_key(&secret, top - 1).unwrap();
    // The following three graphs model separate protocol participants. Only
    // public ciphertext/evaluation-key artifacts cross into the evaluator;
    // the secret key is imported privately by the final decryption graph.
    let encryption_noise = ct.noise_bound.clone();
    let encryption = context
        .transferred_output("secret", secret)
        .unwrap()
        .transferred_output("ciphertext", ct.components)
        .unwrap()
        .transferred_output("relin", relin)
        .unwrap()
        .transferred_output("rotation", rotation)
        .unwrap()
        .transferred_output("backwards", backwards)
        .unwrap()
        .transferred_output("swap", swap)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let backend = backend(&common, Some(&bgv));
    let mut store = MemoryArtifactStore::default();
    let message = (0..n).map(|i| (i as u64 % t) as i64).collect::<Vec<_>>();
    let mut runtime = GpuRuntime::new(backend).expect("construct FHE GPU runtime");
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("slots".into(), BigInt::from(0)..=BigInt::from(t - 1));
    let (_, encryption_id) = prepare_and_run(
        encryption,
        &mut runtime,
        BTreeMap::from([("slots".into(), input(&message))]),
        &mut store,
    );
    runtime.options_mut().integer_input_ranges.clear();
    let encryption_id = encryption_id.unwrap();
    let mut manifests =
        BTreeMap::from([(encryption_id.clone(), store.manifest(&encryption_id).unwrap().clone())]);
    let ct = BgvCiphertext {
        components: common.ring().artifact_input(
            encryption_id.clone(),
            "ciphertext",
            (2, 1),
            ArtifactAvailability::Transferred,
        ),
        correction_factor: 1,
        noise_bound: encryption_noise,
    };
    let (relin_parameters, relin_width) = bgv.key_switch_parameters(top).unwrap();
    let relin = crate::utils::ring(&relin_parameters).artifact_input(
        encryption_id.clone(),
        "relin",
        (2, relin_width),
        ArtifactAvailability::Transferred,
    );
    let (lower, lower_width) = bgv.key_switch_parameters(top - 1).unwrap();
    let ring = crate::utils::ring(&lower);
    let import_key = |name| {
        ring.artifact_input(
            encryption_id.clone(),
            name,
            (2, lower_width),
            ArtifactAvailability::Transferred,
        )
    };
    let sum = bgv.add(&ct, &ct).unwrap();
    let quadratic = bgv.mul_unrelinearized(&ct, &ct).unwrap();
    let product = bgv.relinearize(&quadratic, &relin).unwrap();
    let reduced = bgv.mod_switch_to(&product, top - 1).unwrap();
    let twice = bgv.mod_switch_to(&reduced, top - 2).unwrap();
    let aligned = bgv.match_correction_factor(&reduced, 1).unwrap();
    let forward = bgv.rotate_rows(Some(&import_key("rotation")), &reduced, 1).unwrap();
    let backward = bgv.rotate_rows(Some(&import_key("backwards")), &reduced, -1).unwrap();
    let wrapped = bgv.rotate_rows(None, &reduced, n as i32 / 2).unwrap();
    let swapped = bgv.swap_rows(&import_key("swap"), &reduced).unwrap();
    // Cover both ciphertext degrees, two level drops, factor alignment, and
    // rotations in both directions; a full-row rotation needs no switch key.
    let outputs = [
        ("sum", sum),
        ("quadratic", quadratic),
        ("product", product),
        ("reduced", reduced),
        ("twice", twice),
        ("aligned", aligned),
        ("forward", forward),
        ("backward", backward),
        ("wrapped", wrapped),
        ("swapped", swapped),
    ];
    let mut evaluator = DslContext::new("gpu-bgv-public-evaluator");
    for (name, ct) in &outputs {
        evaluator = evaluator.transferred_output(*name, ct.components.clone()).unwrap();
    }
    let evaluator = evaluator
        .build()
        .unwrap()
        .validate_with_manifests(
            &ParamEnv::default(),
            &manifests,
            mxx_backends::openfhe_guard::gen_modulus_and_warmup,
        )
        .unwrap();
    let (_, evaluation_id) = prepare_and_run(evaluator, &mut runtime, BTreeMap::new(), &mut store);
    let evaluation_id = evaluation_id.unwrap();
    manifests.insert(evaluation_id.clone(), store.manifest(&evaluation_id).unwrap().clone());
    let secret = common.ring().artifact_input(
        encryption_id,
        "secret",
        (1, 1),
        // Secret keys are randomized producer outputs, not cache entries.
        ArtifactAvailability::Transferred,
    );
    let mut decryption = DslContext::new("gpu-bgv-decrypt");
    for (name, ct) in &outputs {
        let ty = ct.components.matrix_type();
        let imported = BgvCiphertext {
            components: Ring::from_ref(ty.ring.clone()).artifact_input(
                evaluation_id.clone(),
                *name,
                (ty.rows.clone(), 1),
                ArtifactAvailability::Transferred,
            ),
            correction_factor: ct.correction_factor,
            // Bounds are public graph metadata, not part of a Matrix artifact.
            // Carry the evaluator's bound into the specialized decryption graph.
            noise_bound: ct.noise_bound.clone(),
        };
        decryption =
            decryption.transferred_output(*name, bgv.decrypt(&secret, &imported).unwrap()).unwrap();
    }
    let decryption = decryption
        .build()
        .unwrap()
        .validate_with_manifests(
            &ParamEnv::default(),
            &manifests,
            mxx_backends::openfhe_guard::gen_modulus_and_warmup,
        )
        .unwrap();
    let (result, _) = prepare_and_run(decryption, &mut runtime, BTreeMap::new(), &mut store);
    for (name, _) in outputs {
        let expected = (0..n)
            .map(|i| {
                let row = i / (n / 2);
                let col = i % (n / 2);
                let source = match name {
                    "forward" => row * (n / 2) + (col + 1) % (n / 2),
                    "backward" => row * (n / 2) + (col + n / 2 - 1) % (n / 2),
                    "swapped" => (1 - row) * (n / 2) + col,
                    _ => i,
                };
                let m = message[source] as u64;
                BigInt::from(if name == "sum" { (2 * m) % t } else { (m * m) % t })
            })
            .collect::<Vec<_>>();
        assert_eq!(integers(&runtime, &result, name), expected, "{name}");
    }
}

#[test]
fn test_gpu_fhe_bgv_short_slot_inputs() {
    let common = common();
    let n = common.ring.ring_dimension() as usize;
    let top = common.ring.to_crt().2 - 1;
    let t = (1..).map(|k| k * 2 * n as u64 + 1).find(|&t| crate::utils::is_prime(t)).unwrap();
    let bgv = BgvParams::new(common.clone(), t, None).unwrap();
    let context = DslContext::new("gpu-bgv-short-slots");
    let single = context.int_family_input("single", 1);
    let partial = context.int_family_input("partial", n - 1);
    let (secret, key) = bgv.keygen().unwrap();
    // A scalar occupies slot zero only; neither encryption nor decryption
    // broadcasts it. The unused slots must survive as zeros on the GPU too.
    let single = bgv.encrypt(&key, &single).unwrap();
    let partial = bgv.encrypt(&key, &partial).unwrap();
    let rotation = bgv.rotation_key(&secret, top, 1).unwrap();
    let rotated = bgv.rotate_rows(Some(&rotation), &single, 1).unwrap();
    let graph = context
        .transferred_output("single", bgv.decrypt(&secret, &single).unwrap())
        .unwrap()
        .transferred_output("partial", bgv.decrypt(&secret, &partial).unwrap())
        .unwrap()
        .transferred_output("rotated", bgv.decrypt(&secret, &rotated).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let backend = backend(&common, Some(&bgv));
    let mut store = MemoryArtifactStore::default();
    let mut runtime = GpuRuntime::new(backend).expect("construct FHE GPU runtime");
    let partial_values = (0..n - 1).map(|i| i as i64 - t as i64 - 1).collect::<Vec<_>>();
    let ranges = &mut runtime.options_mut().integer_input_ranges;
    ranges.insert("single".into(), BigInt::from(-1)..=BigInt::from(-1));
    ranges.insert("partial".into(), BigInt::from(-(t as i64) - 1)..=BigInt::from(n as i64));
    let (result, _) = prepare_and_run(
        graph,
        &mut runtime,
        BTreeMap::from([
            ("single".into(), input(&[-1])),
            ("partial".into(), input(&partial_values)),
        ]),
        &mut store,
    );
    let mut expected = vec![BigInt::from(0); n];
    expected[0] = BigInt::from(t - 1);
    assert_eq!(integers(&runtime, &result, "single"), expected);
    // Positive rotation moves slot zero to the last position of its row,
    // which was outside the single-value input. Decryption must still return it.
    expected[..n / 2].rotate_left(1);
    assert_eq!(integers(&runtime, &result, "rotated"), expected);
    let mut expected = partial_values
        .into_iter()
        .map(|v| BigInt::from(v).mod_floor(&BigInt::from(t)))
        .collect::<Vec<_>>();
    expected.push(BigInt::from(0));
    assert_eq!(integers(&runtime, &result, "partial"), expected);
}

#[test]
fn test_gpu_fhe_bgv_hybrid_multilimb_all_levels() {
    let common = common();
    let n = common.ring.ring_dimension() as usize;
    let depth = common.ring.crt_depth();
    let order = 2 * n as u64;
    let t = (1..).map(|k| k * order + 1).find(|&t| crate::utils::is_prime(t)).unwrap();
    let defaults = BgvParams::new(common.clone(), t, None).unwrap();
    let q_primes = common.ring.to_crt().0;
    let mut auxiliary_primes = defaults
        .key_switch_parameters(depth - 1)
        .unwrap()
        .0
        .to_crt()
        .0
        .into_iter()
        .filter(|p| !q_primes.contains(p))
        .collect::<Vec<_>>();
    let mut candidate = *auxiliary_primes.last().unwrap() - order;
    while auxiliary_primes.len() < 2 {
        if !q_primes.contains(&candidate) && t % candidate != 0 && crate::utils::is_prime(candidate)
        {
            auxiliary_primes.push(candidate);
        }
        candidate -= order;
    }
    let bgv = BgvParams::new(
        common.clone(),
        t,
        Some(BgvHybridParams { digit_size: depth.min(2), auxiliary_primes }),
    )
    .unwrap();
    let mut context = DslContext::new("gpu-bgv-hybrid-multilimb");
    let slots = context.int_family_input("slots", n);
    let (secret, public) = bgv.keygen().unwrap();
    let fresh = bgv.encrypt(&public, &slots).unwrap();
    // Default depth three exercises a two-prime digit and an uneven final
    // digit. Every level uses both approximate ModUp and multi-prime ModDown.
    for level in 0..depth {
        let ct = bgv.mod_switch_to(&fresh, level).unwrap();
        let quadratic = bgv.mul_unrelinearized(&ct, &ct).unwrap();
        let key = bgv.relinearization_key(&secret, level).unwrap();
        let product = bgv.relinearize(&quadratic, &key).unwrap();
        let rotation_key = bgv.rotation_key(&secret, level, 1).unwrap();
        let rotated = bgv.rotate_rows(Some(&rotation_key), &product, 1).unwrap();
        assert!(bgv.can_decrypt(&product).unwrap(), "product level {level}");
        assert!(bgv.can_decrypt(&rotated).unwrap(), "rotation level {level}");
        context = context
            .transferred_output(format!("product{level}"), bgv.decrypt(&secret, &product).unwrap())
            .unwrap()
            .transferred_output(format!("rotated{level}"), bgv.decrypt(&secret, &rotated).unwrap())
            .unwrap();
    }
    let graph = context
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let backend = backend(&common, Some(&bgv));
    let mut store = MemoryArtifactStore::default();
    let mut runtime = GpuRuntime::new(backend).expect("construct FHE GPU runtime");
    let message = (0..n).map(|i| (i as u64 % t) as i64).collect::<Vec<_>>();
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("slots".into(), BigInt::from(0)..=BigInt::from(t - 1));
    let (result, _) = prepare_and_run(
        graph,
        &mut runtime,
        BTreeMap::from([("slots".into(), input(&message))]),
        &mut store,
    );
    let expected = message
        .iter()
        .map(|&m| {
            BigInt::from((u128::from(m as u64) * u128::from(m as u64) % u128::from(t)) as u64)
        })
        .collect::<Vec<_>>();
    let mut rotated = expected.clone();
    rotated[..n / 2].rotate_left(1);
    rotated[n / 2..].rotate_left(1);
    for level in 0..depth {
        assert_eq!(integers(&runtime, &result, &format!("product{level}")), expected);
        assert_eq!(integers(&runtime, &result, &format!("rotated{level}")), rotated);
    }
}
