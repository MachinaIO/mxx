use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    commit::wee25::{Wee25Commit, Wee25PublicParams},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext, encode_nested_rns_poly},
    lookup::{
        commit_eval::{CommitBGGEncodingPltEvaluator, CommitBGGPubKeyPltEvaluator},
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
use std::{fs, path::Path, sync::Arc};
use tracing::info;

#[tokio::test]
async fn test_arithmetic_circuit_operations_commit() {
    // Mixed operations in a single circuit: (a + b) * c - a.
    const P_MODULI_BITS: usize = 6;
    const SCALE: u64 = 1 << 7;
    const BASE_BITS: u32 = 9;
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
    // let poly_c = NestedRnsPoly::input(ctx.clone(), &mut circuit);

    // let sum = poly_a.add_full_reduce(&poly_b, None, &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    // let out_poly = prod.sub_full_reduce(&poly_a, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    info!("{}", format!("non-free depth: {}", circuit.non_free_depth()));

    // 1) Plain polynomial evaluation.
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    // let c_value: BigUint = gen_biguint_for_modulus(&mut rng, modulus.as_ref());
    let a_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value);
    let b_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value);
    // let c_inputs = encode_nested_rns_poly(P_MODULI_BITS, &params, &c_value);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone()].concat();

    let plt_evaluator = PolyPltEvaluator::new();
    let eval_results = circuit.eval(
        &params,
        &DCRTPoly::const_one(&params),
        &plaintext_inputs,
        Some(&plt_evaluator),
    );
    assert_eq!(eval_results.len(), 1);

    let q = modulus.as_ref();
    // let aa = &a_value % q;
    // let bb = &b_value % q;
    // let cc = &c_value % q;
    // let t = (&aa + &bb) % q;
    // let t = (t * &cc) % q;
    // let expected = (t + (q - &aa)) % q;
    let expected = (&a_value * &b_value) % q;
    let expected_poly = DCRTPoly::from_biguint_to_constant(&params, expected);

    assert_eq!(eval_results[0], expected_poly, "mixed operations should be correct");

    // 2) BGG+ public key evaluation (Commit PLT).
    let storage_dir = Path::new("test_data/arithmetic_circuit_operations_commit");
    if !storage_dir.exists() {
        fs::create_dir_all(storage_dir).unwrap();
    }
    let seed: [u8; 32] = [0u8; 32];
    let d = 1usize;
    let trapdoor_sigma = 4.578;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, trapdoor_sigma);
    let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, d);

    init_storage_system(storage_dir.to_path_buf());
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pk_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(seed, d);
    let pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);

    let tree_base = 4;
    info!("wee25 public params sampling start");
    let wee25_commit = Wee25Commit::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
        &params,
        d,
        tree_base,
        trapdoor_sigma,
    );
    wee25_commit.sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
        &params,
        seed,
        storage_dir,
    );
    wait_for_all_writes(storage_dir.to_path_buf()).await.unwrap();
    let wee25_public_params = Wee25PublicParams::<DCRTPolyMatrix>::read_from_storage(
        &params,
        storage_dir,
        &wee25_commit,
        seed,
    )
    .expect("wee25 public params not found");
    info!("wee25 public params sampling done");
    let pk_evaluator =
        CommitBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup(
            &params,
            d,
            trapdoor_sigma,
            tree_base,
            seed,
            wee25_public_params,
        );
    info!("start pubkey evaluation");
    let start = std::time::Instant::now();
    let pubkey_out = circuit.eval(&params, &pubkeys[0], &pubkeys[1..], Some(&pk_evaluator));
    info!("{}", format!("end pubkey evaluation in {:?}", start.elapsed()));
    assert_eq!(pubkey_out.len(), 1);
    info!("commit all LUT matrices");
    pk_evaluator.commit_all_lut_matrices::<DCRTPolyTrapdoorSampler>(
        &params,
        &b0_matrix,
        &b0_trapdoor,
    );
    info!("wait for all writes");
    wait_for_all_writes(storage_dir.to_path_buf()).await.unwrap();
    info!("finish writing");

    // 3) BGG+ encoding evaluation.
    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::BitDist).get_row(0);
    let s = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());
    let c_b0 = s.clone() * &b0_matrix;
    let c_b = s.clone() * pk_evaluator.wee25_public_params.b.clone();

    let bgg_encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
    let zero_plaintexts = vec![DCRTPoly::const_zero(&params); circuit.num_input()];
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &zero_plaintexts);
    let enc_evaluator =
        CommitBGGEncodingPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup(
            &params,
            trapdoor_sigma,
            tree_base,
            seed,
            &circuit,
            &pubkeys[0],
            &pubkeys[1..],
            &c_b0,
            &c_b,
            &storage_dir.to_path_buf(),
            pk_evaluator.wee25_public_params.clone(),
        );
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
