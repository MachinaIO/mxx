//! GPU execution of the production FHE graphs; all artifacts stay in memory.
use crate::{
    BgvCiphertext, BgvHybridParams, BgvParams, FheCommonParams, FheScheme, RingGswParams,
    utils::common,
};
use mxx_dsl::{DslContext, Ring};
use mxx_ir_core::{ParamEnv, artifact::ArtifactConfidentiality};
use mxx_primitives::poly::{PolyParams, dcrt::gpu::GpuDCRTPolyParams};
use mxx_runtime::{
    ExecutionResult, MemoryArtifactStore, RuntimeValue,
    backend::poly_gpu::{GpuDcrtBackend, gpu_backend},
    execute,
    transcript::SamplingMode,
};
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use std::collections::BTreeMap;

#[path = "gpu_test_utils.rs"]
mod gpu_test_utils;
use gpu_test_utils::configure_widths;

fn backend(common: &FheCommonParams, bgv: Option<&BgvParams>) -> GpuDcrtBackend {
    let rings = if let Some(bgv) = bgv {
        bgv.runtime_parameters().unwrap()
    } else {
        let (primes, _, depth) = common.ring.to_crt();
        let mut rings =
            (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
        rings
            .extend(primes.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
        rings
    };
    gpu_backend(
        rings
            .iter()
            .map(|p| GpuDCRTPolyParams::new(p.ring_dimension(), p.to_crt().0, p.base_bits(), None)),
    )
}

fn input(values: &[i64]) -> RuntimeValue<GpuDcrtBackend> {
    RuntimeValue::IndexedFamily(
        values.iter().map(|v| RuntimeValue::Int(BigInt::from(*v))).collect(),
    )
}

fn values(
    result: &mut ExecutionResult<GpuDcrtBackend>,
    name: &str,
    backend: &mut GpuDcrtBackend,
    store: &mut MemoryArtifactStore,
) -> Vec<BigInt> {
    let RuntimeValue::IndexedFamily(values) =
        result.materialize_output(name, backend, store).unwrap()
    else {
        panic!("integer family")
    };
    values
        .iter()
        .map(|v| {
            let RuntimeValue::Int(v) = v else { panic!("integer") };
            v.clone()
        })
        .collect()
}

#[test]
fn test_gpu_fhe_ring_gsw_runtime() {
    let common = common();
    let n = common.ring.ring_dimension() as usize;
    let scheme =
        RingGswParams::new(common.clone(), BigUint::from(1u64 << 22), BigUint::from(2u8)).unwrap();
    let context = DslContext::new("gpu-ring-gsw");
    let message = context.int_family_input("message", n);
    let multiplier = context.int_family_input("multiplier", n);
    let (secret, key) = scheme.keygen().unwrap();
    let ct = scheme.encrypt(&key, &common.ring().from_coefficients(&message)).unwrap();
    let gsw = scheme.encrypt_gsw(&secret, &common.ring().from_coefficients(&multiplier)).unwrap();
    let product = scheme.mul(&ct, &gsw, &()).unwrap();
    let sum = scheme.add(&ct, &ct).unwrap();
    let graph = context
        .private_output("roundtrip", scheme.decrypt(&secret, &ct).unwrap().coefficients())
        .unwrap()
        .private_output("sum", scheme.decrypt(&secret, &sum).unwrap().coefficients())
        .unwrap()
        .private_output("product", scheme.decrypt(&secret, &product).unwrap().coefficients())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let message = (0..n).map(|i| i as i64 % 5 - 2).collect::<Vec<_>>();
    let mut multiplier = vec![0; n];
    multiplier[1] = 1;
    let mut backend = backend(&common, None);
    let mut store = MemoryArtifactStore::default();
    configure_widths(&mut backend, &graph);
    let mut result = execute(
        &graph,
        &mut backend,
        BTreeMap::from([
            ("message".into(), input(&message)),
            ("multiplier".into(), input(&multiplier)),
        ]),
        &mut store,
        SamplingMode::Fresh,
    )
    .unwrap();
    assert_eq!(
        values(&mut result, "roundtrip", &mut backend, &mut store),
        message
            .iter()
            .map(|v| BigInt::from(*v)
                .mod_floor(&BigInt::from(common.ring.modulus().as_ref().clone())))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        values(&mut result, "sum", &mut backend, &mut store),
        message
            .iter()
            .map(|v| BigInt::from(2 * v)
                .mod_floor(&BigInt::from(common.ring.modulus().as_ref().clone())))
            .collect::<Vec<_>>()
    );
    // Multiplication by X shifts coefficients, with a negated wrapped term
    // because this is R_q = Z_q[X]/(X^N + 1), not a cyclic polynomial ring.
    let mut expected = message;
    expected.rotate_right(1);
    expected[0] = -expected[0];
    assert_eq!(
        values(&mut result, "product", &mut backend, &mut store),
        expected
            .into_iter()
            .map(|v| BigInt::from(v)
                .mod_floor(&BigInt::from(common.ring.modulus().as_ref().clone())))
            .collect::<Vec<_>>()
    );
    result.cleanup_staged(&mut store).unwrap();
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
        .private_output("secret", secret)
        .unwrap()
        .public_output("ciphertext", ct.components)
        .unwrap()
        .public_output("relin", relin)
        .unwrap()
        .public_output("rotation", rotation)
        .unwrap()
        .public_output("backwards", backwards)
        .unwrap()
        .public_output("swap", swap)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut backend = backend(&common, Some(&bgv));
    let mut store = MemoryArtifactStore::default();
    let message = (0..n).map(|i| (i as u64 % t) as i64).collect::<Vec<_>>();
    configure_widths(&mut backend, &encryption);
    let encrypted = execute(
        &encryption,
        &mut backend,
        BTreeMap::from([("slots".into(), input(&message))]),
        &mut store,
        SamplingMode::Fresh,
    )
    .unwrap();
    let encryption_id = encrypted.production_id.unwrap();
    let mut manifests =
        BTreeMap::from([(encryption_id.clone(), store.manifest(&encryption_id).unwrap().clone())]);
    let ct = BgvCiphertext {
        components: common.ring().artifact_input(
            encryption_id.clone(),
            "ciphertext",
            (2, 1),
            ArtifactConfidentiality::Public,
        ),
        correction_factor: 1,
        noise_bound: encryption_noise,
    };
    let (relin_parameters, relin_width) = bgv.key_switch_parameters(top).unwrap();
    let relin = Ring::new(relin_parameters.modulus().as_ref().clone(), n).artifact_input(
        encryption_id.clone(),
        "relin",
        (2, relin_width),
        ArtifactConfidentiality::Public,
    );
    let (lower, lower_width) = bgv.key_switch_parameters(top - 1).unwrap();
    let ring = Ring::new(lower.modulus().as_ref().clone(), n);
    let import_key = |name| {
        ring.artifact_input(
            encryption_id.clone(),
            name,
            (2, lower_width),
            ArtifactConfidentiality::Public,
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
        evaluator = evaluator.public_output(*name, ct.components.clone()).unwrap();
    }
    let evaluator = evaluator
        .build()
        .unwrap()
        .validate_with_manifests(&ParamEnv::default(), &manifests)
        .unwrap();
    configure_widths(&mut backend, &evaluator);
    let evaluated =
        execute(&evaluator, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
            .unwrap();
    let evaluation_id = evaluated.production_id.unwrap();
    manifests.insert(evaluation_id.clone(), store.manifest(&evaluation_id).unwrap().clone());
    let secret = common.ring().artifact_input(
        encryption_id,
        "secret",
        (1, 1),
        ArtifactConfidentiality::Private,
    );
    let mut decryption = DslContext::new("gpu-bgv-decrypt");
    for (name, ct) in &outputs {
        let ty = ct.components.matrix_type();
        let imported = BgvCiphertext {
            components: Ring::new(ty.modulus.clone(), n).artifact_input(
                evaluation_id.clone(),
                *name,
                (ty.rows.clone(), 1),
                ArtifactConfidentiality::Public,
            ),
            correction_factor: ct.correction_factor,
            // Bounds are public graph metadata, not part of a Matrix artifact.
            // Carry the evaluator's bound into the specialized decryption graph.
            noise_bound: ct.noise_bound.clone(),
        };
        decryption =
            decryption.private_output(*name, bgv.decrypt(&secret, &imported).unwrap()).unwrap();
    }
    let decryption = decryption
        .build()
        .unwrap()
        .validate_with_manifests(&ParamEnv::default(), &manifests)
        .unwrap();
    configure_widths(&mut backend, &decryption);
    let mut result =
        execute(&decryption, &mut backend, BTreeMap::new(), &mut store, SamplingMode::Fresh)
            .unwrap();
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
        assert_eq!(values(&mut result, name, &mut backend, &mut store), expected, "{name}");
    }
    result.cleanup_staged(&mut store).unwrap();
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
        .private_output("single", bgv.decrypt(&secret, &single).unwrap())
        .unwrap()
        .private_output("partial", bgv.decrypt(&secret, &partial).unwrap())
        .unwrap()
        .private_output("rotated", bgv.decrypt(&secret, &rotated).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut backend = backend(&common, Some(&bgv));
    configure_widths(&mut backend, &graph);
    let mut store = MemoryArtifactStore::default();
    let partial_values = (0..n - 1).map(|i| i as i64 - t as i64 - 1).collect::<Vec<_>>();
    let mut result = execute(
        &graph,
        &mut backend,
        BTreeMap::from([
            ("single".into(), input(&[-1])),
            ("partial".into(), input(&partial_values)),
        ]),
        &mut store,
        SamplingMode::Fresh,
    )
    .unwrap();
    let mut expected = vec![BigInt::from(0); n];
    expected[0] = BigInt::from(t - 1);
    assert_eq!(values(&mut result, "single", &mut backend, &mut store), expected);
    // Positive rotation moves slot zero to the last position of its row,
    // which was outside the single-value input. Decryption must still return it.
    expected[..n / 2].rotate_left(1);
    assert_eq!(values(&mut result, "rotated", &mut backend, &mut store), expected);
    let mut expected = partial_values
        .into_iter()
        .map(|v| BigInt::from(v).mod_floor(&BigInt::from(t)))
        .collect::<Vec<_>>();
    expected.push(BigInt::from(0));
    assert_eq!(values(&mut result, "partial", &mut backend, &mut store), expected);
    result.cleanup_staged(&mut store).unwrap();
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
            .private_output(format!("product{level}"), bgv.decrypt(&secret, &product).unwrap())
            .unwrap()
            .private_output(format!("rotated{level}"), bgv.decrypt(&secret, &rotated).unwrap())
            .unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let mut backend = backend(&common, Some(&bgv));
    configure_widths(&mut backend, &graph);
    let mut store = MemoryArtifactStore::default();
    let message = (0..n).map(|i| (i as u64 % t) as i64).collect::<Vec<_>>();
    let mut result = execute(
        &graph,
        &mut backend,
        BTreeMap::from([("slots".into(), input(&message))]),
        &mut store,
        SamplingMode::Fresh,
    )
    .unwrap();
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
        assert_eq!(
            values(&mut result, &format!("product{level}"), &mut backend, &mut store),
            expected
        );
        assert_eq!(
            values(&mut result, &format!("rotated{level}"), &mut backend, &mut store),
            rotated
        );
    }
    result.cleanup_staged(&mut store).unwrap();
}
