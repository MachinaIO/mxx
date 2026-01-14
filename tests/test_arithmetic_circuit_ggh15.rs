use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    lookup::{
        ggh15_eval::{GGH15BGGEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler, hash::DCRTPolyHashSampler,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::gen_biguint_for_modulus,
};
use num_bigint::BigUint;
use std::sync::Arc;
use tempfile::tempdir;
use tracing::info;

#[tokio::test]
async fn test_arithmetic_circuit_operations_ggh15() {
    // Mixed operations in a single circuit: (a + b) * c - a.
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 7;
    const BASE_BITS: u32 = 8;
    let _ = tracing_subscriber::fmt::try_init();

    // Use parameters where NestedRnsPoly is known to be correct.
    let params = DCRTPolyParams::new(4, 3, 18, BASE_BITS);
    let mut rng = rand::rng();

    let modulus = params.modulus();

    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx =
        Arc::new(NestedRnsPolyContext::setup(&mut circuit, &params, P_MODULI_BITS, SCALE, false));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_c = NestedRnsPoly::input(ctx.clone(), &mut circuit);

    let sum = poly_a.add_full_reduce(&poly_b, None, &mut circuit);
    let prod = sum.mul_full_reduce(&poly_c, None, &mut circuit);
    let out_poly = prod.sub_full_reduce(&poly_a, None, &mut circuit);
    let out = out_poly.reconstruct(&params, None, &mut circuit);
    circuit.output(vec![out]);
    info!("{}", format!("non-free depth: {}", circuit.non_free_depth()));

    // 1) Plain polynomial evaluation.
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let c_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value);
    let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value);
    let c_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &c_value);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone(), c_inputs.clone()].concat();

    let plt_evaluator = PolyPltEvaluator::new();
    let eval_results = circuit.eval(
        &params,
        &DCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&plt_evaluator),
    );
    assert_eq!(eval_results.len(), 1);

    let q = modulus.as_ref();
    let aa = &a_value % q;
    let bb = &b_value % q;
    let cc = &c_value % q;
    let t = (&aa + &bb) % q;
    let t = (t * &cc) % q;
    let expected = (t + (q - &aa)) % q;
    let expected_poly = DCRTPoly::from_biguint_to_constant(&params, expected);

    assert_eq!(eval_results[0], expected_poly, "mixed operations should be correct");

    // 2) BGG+ public key evaluation (GGH15 PLT).
    let tmp_dir = tempdir().unwrap();
    let seed: [u8; 32] = [0u8; 32];
    let d = 1usize;
    let trapdoor_sigma = 4.578;
    let error_sigma = 0.0;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, trapdoor_sigma);
    let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, d);
    let b0_trapdoor = Arc::new(b0_trapdoor);
    let b0_matrix = Arc::new(b0_matrix);

    init_storage_system(tmp_dir.path().to_path_buf());
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pk_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(seed, d);
    let pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);

    let insert_1_to_s = false;
    let pk_evaluator = GGH15BGGPubKeyPltEvaluator::<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
    >::new(
        seed,
        trapdoor_sigma,
        error_sigma,
        &params,
        b0_matrix.clone(),
        b0_trapdoor.clone(),
        tmp_dir.path().to_path_buf(),
        insert_1_to_s,
    );
    info!("start pubkey evaluation");
    let start = std::time::Instant::now();
    let pubkey_out = circuit.eval(&params, &pubkeys[0], &pubkeys[1..], Some(&pk_evaluator));
    info!("{}", format!("end pubkey evaluation in {:?}", start.elapsed()));
    assert_eq!(pubkey_out.len(), 1);
    info!("wait for all writes");
    pk_evaluator.sample_aux_matrices(&params);
    wait_for_all_writes(tmp_dir.path().to_path_buf()).await.unwrap();
    info!("finish writing");

    // 3) BGG+ encoding evaluation.
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());
    let c_b0 = s.clone() * b0_matrix.as_ref();

    let bgg_encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
    let zero_plaintexts = vec![DCRTPoly::const_zero(&params); circuit.num_input()];
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &zero_plaintexts);
    let enc_evaluator = GGH15BGGEncodingPltEvaluator::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
    >::new(seed, &params, tmp_dir.path().to_path_buf(), d, c_b0);
    info!("start encoding evaluation");
    let start = std::time::Instant::now();
    let encoding_out = circuit.eval(&params, &encodings[0], &encodings[1..], Some(&enc_evaluator));
    info!("{}", format!("end encoding evaluation in {:?}", start.elapsed()));
    assert_eq!(encoding_out.len(), 1);
    assert_eq!(encoding_out[0].plaintext.as_ref().unwrap(), &DCRTPoly::const_zero(&params));
    assert_eq!(encoding_out[0].pubkey, pubkey_out[0]);

    let encoding_expected = s.clone() * &pubkey_out[0].matrix;
    assert_eq!(encoding_out[0].vector, encoding_expected);
}
