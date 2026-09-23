//! End-to-end Ring-GSW GPU round-trip.
//!
//! The test builds DSL programs, plans them on one [`GpuRuntime`], copies each
//! execution's resident outputs on the device into the next program's inputs,
//! and checks decryption.

use mxx_fhe::utils::{
    self,
    gpu::{copy_outputs, input, integers},
};

use mxx_backends::{
    GpuRuntime, MemoryArtifactStore,
    backend::poly_gpu::gpu_backend,
    poly::{
        PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
    },
};
use mxx_dsl::DslContext;
use mxx_fhe::{FheCommonParams, FheScheme, RingCiphertext};
use mxx_ir_core::ParamEnv;
use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use rand::Rng;
use std::collections::BTreeMap;

fn ring(parameters: &DCRTPolyParams) -> mxx_dsl::Ring {
    mxx_dsl::Ring::from_crt_moduli(
        parameters.to_crt().0.into_iter().map(Into::into).collect(),
        parameters.ring_dimension(),
    )
}

fn ring_gsw_ring_parameters(common: &FheCommonParams) -> Vec<DCRTPolyParams> {
    let (primes, _, depth) = common.ring.to_crt();
    let mut rings =
        (0..depth).map(|level| common.parameters_at(level).unwrap()).collect::<Vec<_>>();
    rings.extend(primes.iter().map(|p| common.ring.select_modulus(&BigUint::from(*p)).unwrap()));
    // The first ordered prefix is also the first single-prime correction
    // parameter. Register each logical ring only once: the GPU backend maps
    // parameters by their CRT layout, so duplicate entries make placement
    // ambiguous even when they share a context.
    let mut unique_rings = Vec::with_capacity(rings.len());
    for ring in rings {
        if !unique_rings.contains(&ring) {
            unique_rings.push(ring);
        }
    }
    unique_rings
}

fn ring_gsw_parameters(common: &FheCommonParams) -> Vec<GpuDCRTPolyParams> {
    let rings = ring_gsw_ring_parameters(common);
    let mut parameters: Vec<GpuDCRTPolyParams> = Vec::with_capacity(rings.len());
    for ring in rings {
        let parameter = if let Some(related) = parameters.first() {
            GpuDCRTPolyParams::new_with_gpu(
                ring.ring_dimension(),
                ring.to_crt().0,
                ring.base_bits(),
                related.gpu_ids().to_vec(),
                Some(1),
                Some(related),
                None,
            )
        } else {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        };
        parameters.push(parameter);
    }
    parameters
}

#[test]
fn test_gpu_ring_gsw() {
    let scheme = utils::ring_gsw_params();
    let common = &scheme.common;
    let n = common.ring.ring_dimension() as usize;
    let ring = ring(&common.ring);
    let q = common.ring.modulus();
    // Construct the symbolic encryption program first; its named inputs become
    // the bindings supplied when this graph is executed on the GPU.
    let context = DslContext::new("integration-ring-gsw-encrypt");
    let secret = ring.input("sk", (1, 1));
    let message = ring.from_coefficients(&context.int_family_input("message", n));
    let bit = ring.from_coefficients(&context.int_family_input("bit", n));
    let ct = scheme.encrypt(&secret, &message).unwrap();
    let gsw = scheme.encrypt_gsw(&secret, &bit).unwrap();
    let encryption = context
        // Plain outputs remain resident owners in the transient execution
        // result.  Intermediate ciphertext components must not be exported
        // to the artifact store and re-imported through host memory.
        .output("a", ct.a.clone())
        .unwrap()
        .output("b", ct.b.clone())
        .unwrap()
        .output("ga", gsw.a.clone())
        .unwrap()
        .output("gb", gsw.b.clone())
        .unwrap()
        .build()
        .unwrap();
    // Rebind the resident ciphertext components as graph inputs so the
    // evaluator and the final check can consume the returned GPU owners.
    let ct = RingCiphertext { a: ring.input("a", (1, 1)), b: ring.input("b", (1, 1)), ..ct };
    let width = 2 * common.ring.modulus_digits();
    let gsw =
        RingCiphertext { a: ring.input("ga", (1, width)), b: ring.input("gb", (1, width)), ..gsw };
    let product = scheme.external_product(&gsw, &ct).unwrap();
    let parameters = ring_gsw_parameters(common);
    // One runtime owns planning and execution for every graph in this round-trip.
    let backend = gpu_backend(parameters.iter().cloned());
    let mut runtime = GpuRuntime::new(backend).expect("construct Ring-GSW GPU runtime");
    let (keygen_secret, _) = scheme.keygen().unwrap();
    let keygen_graph = DslContext::new("integration-ring-gsw-keygen")
        // The secret key is an intermediate GPU owner consumed by encryption;
        // keeping this a plain output avoids a producer-session artifact export.
        .output("sk", keygen_secret)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("valid integration graph");
    // Plan fixes the executable GPU path; execute returns the key material that
    // is passed as inputs to the encryption graph below.
    let mut keygen =
        runtime.plan(keygen_graph, &BTreeMap::new()).expect("production-equivalent GPU warmup");
    let mut keygen_store = MemoryArtifactStore::default();
    let keygen_result = runtime
        .execute(&mut keygen, BTreeMap::new(), &mut keygen_store, [0; 32])
        .expect("GPU execution");
    // A plain graph output is a resident value borrowed from its plan; the
    // device copy keeps it for later programs without any producer-session
    // artifact or host serialization of this intermediate key material.
    let keys = copy_outputs(&runtime, &keygen_result);

    let exponent = rand::rng().random_range(0..2 * n);
    let mut message = vec![0i64; n];
    message[exponent % n] = if exponent < n { 1 } else { -1 };
    let bit_value = rand::rng().random_range(0..=1);
    let mut bit = vec![0i64; n];
    bit[0] = bit_value;
    let encryption_inputs = BTreeMap::from([
        ("sk".into(), keys["sk"].clone()),
        ("message".into(), input(&message)),
        ("bit".into(), input(&bit)),
    ]);
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("message".into(), BigInt::from(-1)..=BigInt::from(1));
    runtime
        .options_mut()
        .integer_input_ranges
        .insert("bit".into(), BigInt::from(0)..=BigInt::from(1));
    let mut encryption = runtime
        .plan(
            encryption
                .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
                .expect("valid integration graph"),
            &encryption_inputs,
        )
        .expect("production-equivalent GPU warmup");
    let mut encryption_store = MemoryArtifactStore::default();
    let encryption_result = runtime
        .execute(&mut encryption, encryption_inputs, &mut encryption_store, [0; 32])
        .expect("GPU execution");
    let encrypted = copy_outputs(&runtime, &encryption_result);
    runtime.options_mut().integer_input_ranges.clear();

    let evaluator_graph = DslContext::new("integration-ring-gsw-external-product")
        .output("a", product.a.clone())
        .unwrap()
        .output("b", product.b.clone())
        .unwrap()
        .build()
        .unwrap();
    let mut evaluator = runtime
        .plan(
            evaluator_graph
                .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
                .expect("valid integration graph"),
            &encrypted,
        )
        .expect("production-equivalent GPU warmup");
    let mut evaluator_store = MemoryArtifactStore::default();
    let evaluator_result = runtime
        .execute(&mut evaluator, encrypted, &mut evaluator_store, [0; 32])
        .expect("GPU execution");
    let evaluated = copy_outputs(&runtime, &evaluator_result);

    // Decryption is the final DSL program: it consumes the evaluated
    // ciphertext and produces the value checked by the round-trip assertion.
    let imported = RingCiphertext {
        a: ring.input("a", (1, 1)),
        b: ring.input("b", (1, 1)),
        noise_bound: product.noise_bound.clone(),
        plaintext_bound: product.plaintext_bound.clone(),
    };
    let decryption_graph = DslContext::new("integration-ring-gsw-check")
        // The final integer download below is the sole explicit host boundary.
        .output("decoded", scheme.decrypt(&secret, &imported).unwrap().coefficients())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .expect("valid integration graph");
    let decryption_inputs = BTreeMap::from([
        ("sk".into(), keys["sk"].clone()),
        ("a".into(), evaluated["a"].clone()),
        ("b".into(), evaluated["b"].clone()),
    ]);
    let mut decryption = runtime
        .plan(decryption_graph, &decryption_inputs)
        .expect("production-equivalent GPU warmup");
    let mut decryption_store = MemoryArtifactStore::default();
    let decryption_result = runtime
        .execute(&mut decryption, decryption_inputs, &mut decryption_store, [0; 32])
        .expect("GPU execution");
    let decoded = copy_outputs(&runtime, &decryption_result);
    let expected = message
        .iter()
        .map(|v| BigInt::from(v * bit_value).mod_floor(&BigInt::from(q.as_ref().clone())))
        .collect::<Vec<_>>();
    // The decoded coefficients are the sole semantic check for the round-trip.
    assert_eq!(integers(&runtime, &decoded, "decoded"), expected);
}
