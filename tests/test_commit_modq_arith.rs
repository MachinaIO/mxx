use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    commit::wee25::{Wee25Commit, Wee25PublicParams},
    element::PolyElem,
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
    simulator::{SimulatorContext, error_norm::NormPltCommitEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::{bigdecimal_bits_ceil, gen_biguint_for_modulus},
};
use num_bigint::BigUint;
use std::{fs, path::Path, sync::Arc};
use tracing::info;

const RING_DIM: u32 = 1 << 8;
const CRT_BITS: usize = 24;
const P_MODULI_BITS: usize = 6;
const SCALE: u64 = 1 << 7;
const BASE_BITS: u32 = 12;
const MAX_CRT_DEPTH: usize = 32;
const ERROR_SIGMA: f64 = 4.0;
const D_SECRET: usize = 1;
const TREE_BASE: usize = 4;

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn q_level_from_env() -> Option<usize> {
    std::env::var("COMMIT_MODQ_ARITH_Q_LEVEL").ok().map(|raw| {
        let level =
            raw.parse::<usize>().expect("COMMIT_MODQ_ARITH_Q_LEVEL must be a positive integer");
        assert!(level > 0, "COMMIT_MODQ_ARITH_Q_LEVEL must be greater than or equal to 1");
        level
    })
}

fn active_q_moduli_and_modulus<T: PolyParams>(
    params: &T,
    q_level: Option<usize>,
) -> (Vec<u64>, BigUint, usize) {
    let (q_moduli, _, max_q_level) = params.to_crt();
    let active_q_level = q_level.unwrap_or(max_q_level);
    assert!(
        active_q_level <= max_q_level,
        "q_level exceeds CRT depth: q_level={}, crt_depth={}",
        active_q_level,
        max_q_level
    );
    let active_q_moduli = q_moduli.into_iter().take(active_q_level).collect::<Vec<_>>();
    let active_q =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    (active_q_moduli, active_q, active_q_level)
}

fn assert_value_matches_q_level(
    value: &BigUint,
    expected_mod_active_q: &BigUint,
    active_q: &BigUint,
    active_q_moduli: &[u64],
    all_q_moduli: &[u64],
) {
    if active_q_moduli.len() == all_q_moduli.len() {
        assert_eq!(
            value, expected_mod_active_q,
            "value must match expected modulo full q when q_level covers all CRT levels"
        );
        return;
    }

    assert_eq!(
        value % active_q,
        expected_mod_active_q.clone(),
        "value modulo active q must match expected modulo active q"
    );
    for &q_i in all_q_moduli.iter().skip(active_q_moduli.len()) {
        assert_eq!(
            value % BigUint::from(q_i),
            BigUint::from(0u64),
            "inactive CRT residues must be zero when q_level is limited"
        );
    }
}

fn build_modq_arith_circuit(
    params: &DCRTPolyParams,
    q_level: Option<usize>,
) -> (PolyCircuit<DCRTPoly>, Arc<NestedRnsPolyContext>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);

    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit(
    params: &DCRTPolyParams,
    q_level: Option<usize>,
) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx = Arc::new(NestedRnsPolyContext::setup(
        &mut circuit,
        params,
        P_MODULI_BITS,
        SCALE,
        false,
        q_level,
    ));

    let poly_a = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let poly_b = NestedRnsPoly::input(ctx.clone(), &mut circuit);
    let prod = poly_a.mul_full_reduce(&poly_b, None, &mut circuit);
    let out = prod.reconstruct(None, &mut circuit);
    circuit.output(vec![out]);
    circuit
}

fn find_crt_depth_for_modq_arith(
    q_level: Option<usize>,
) -> (usize, DCRTPolyParams, PolyCircuit<DCRTPoly>) {
    let ring_dim_sqrt = BigDecimal::from_u32(RING_DIM).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << BASE_BITS, 0);
    let error_sigma = BigDecimal::from_f64(ERROR_SIGMA).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << P_MODULI_BITS) - 1);
    let e_init_norm = &error_sigma * BigDecimal::from(6u64);

    for crt_depth in 1..=MAX_CRT_DEPTH {
        let params = DCRTPolyParams::new(RING_DIM, crt_depth, CRT_BITS, BASE_BITS);
        let (active_q_moduli, _, crt_depth) = active_q_moduli_and_modulus(&params, q_level);
        let full_q = params.modulus();
        let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
        let (circuit, _ctx) = build_modq_arith_circuit(&params, q_level);

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            D_SECRET,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator =
            NormPltCommitEvaluator::new(ctx.clone(), &error_sigma, TREE_BASE, &circuit);

        let out_errors = circuit.simulate_max_error_norm(
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
        );

        assert_eq!(out_errors.len(), 1);

        let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
        let error = &out_errors[0].matrix_norm.poly_norm.norm;
        let max_error_bits = bigdecimal_bits_ceil(error);
        let all_ok = *error < BigDecimal::from_biguint(threshold, 0);

        info!(
            "crt_depth={} q_bits={} max_error_bits={}",
            crt_depth,
            params.modulus_bits(),
            max_error_bits
        );

        if all_ok {
            return (crt_depth, params, circuit);
        }
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_i) for all CRT moduli not found up to MAX_CRT_DEPTH"
    );
}

#[tokio::test]
#[ignore]
async fn test_commit_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();

    let q_level = q_level_from_env();
    let (crt_depth, params, circuit) = find_crt_depth_for_modq_arith(q_level);
    let (all_q_moduli, _, _) = params.to_crt();
    let (active_q_moduli, active_q, active_q_level) = active_q_moduli_and_modulus(&params, q_level);
    let full_q = params.modulus();
    info!(
        "selected crt_depth={} ring_dim={} crt_bits={} base_bits={} q_level={:?} q_moduli={:?}",
        crt_depth,
        params.ring_dimension(),
        CRT_BITS,
        BASE_BITS,
        q_level,
        all_q_moduli
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?}",
        circuit.non_free_depth(),
        circuit.count_gates_by_type_vec()
    );

    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        active_q.bits(),
        active_q_moduli.len()
    );
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");

    let mut rng = rand::rng();
    let a_value: BigUint = gen_biguint_for_modulus(&mut rng, &active_q);
    let b_value: BigUint = gen_biguint_for_modulus(&mut rng, &active_q);

    let expected = (&a_value * &b_value) % &active_q;

    let a_inputs: Vec<DCRTPoly> = encode_nested_rns_poly(P_MODULI_BITS, &params, &a_value, q_level);
    let b_inputs: Vec<DCRTPoly> = encode_nested_rns_poly(P_MODULI_BITS, &params, &b_value, q_level);
    let plaintext_inputs = [a_inputs.clone(), b_inputs.clone()].concat();
    let plaintext_inputs_shared: Vec<Arc<DCRTPoly>> =
        plaintext_inputs.iter().cloned().map(Arc::new).collect();

    let dry_circuit = build_modq_arith_value_circuit(&params, q_level);
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_one = Arc::new(DCRTPoly::const_one(&params));
    let dry_out =
        dry_circuit.eval(&params, &dry_one, &plaintext_inputs_shared, Some(&dry_plt_evaluator));
    assert_eq!(dry_out.len(), 1, "plain PolyCircuit dry-run should output one value polynomial");
    let dry_const_term = dry_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("dry-run output polynomial must have at least one coefficient")
        .value()
        .clone();
    assert_value_matches_q_level(
        &dry_const_term,
        &expected,
        &active_q,
        &active_q_moduli,
        &all_q_moduli,
    );
    info!("plain PolyCircuit dry-run succeeded with expected constant term");

    let plt_evaluator = PolyPltEvaluator::new();
    let plain_one = Arc::new(DCRTPoly::const_one(&params));
    let plain_out =
        circuit.eval(&params, &plain_one, &plaintext_inputs_shared, Some(&plt_evaluator));
    assert_eq!(plain_out.len(), 1);
    let plain_const = plain_out[0]
        .coeffs()
        .into_iter()
        .next()
        .expect("output poly must have at least one coefficient")
        .value()
        .clone();
    assert_value_matches_q_level(
        &plain_const,
        &expected,
        &active_q,
        &active_q_moduli,
        &all_q_moduli,
    );

    let seed: [u8; 32] = [0u8; 32];
    let trapdoor_sigma = 4.578;
    let d_secret = D_SECRET;

    let dir_name = format!("test_data/test_commit_modq_arith_qlevel_{}", active_q_level);
    let storage_dir = Path::new(&dir_name);
    if !storage_dir.exists() {
        fs::create_dir_all(storage_dir).unwrap();
    }
    init_storage_system(storage_dir.to_path_buf());

    let uniform_sampler = DCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());

    let pk_sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);

    let encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    let enc_one = encodings[0].clone();

    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, trapdoor_sigma);
    let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, d_secret);

    info!("wee25 public params sampling start");
    let wee25_commit = Wee25Commit::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::new(
        &params,
        d_secret,
        TREE_BASE,
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
            d_secret,
            trapdoor_sigma,
            TREE_BASE,
            seed,
            wee25_public_params,
        );

    let pubkey_one = Arc::new(enc_one.pubkey.clone());
    let input_pubkeys: Vec<Arc<_>> = pubkeys[1..].iter().cloned().map(Arc::new).collect();
    let pubkey_out = circuit.eval(&params, &pubkey_one, &input_pubkeys, Some(&pk_evaluator));
    assert_eq!(pubkey_out.len(), 1);

    pk_evaluator.commit_all_lut_matrices::<DCRTPolyTrapdoorSampler>(
        &params,
        &b0_matrix,
        &b0_trapdoor,
    );
    wait_for_all_writes(storage_dir.to_path_buf()).await.unwrap();

    let c_b0 = s_vec.clone() * &b0_matrix;
    let c_b = s_vec.clone() * pk_evaluator.wee25_public_params.b.clone();

    let enc_evaluator =
        CommitBGGEncodingPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>::setup(
            &params,
            trapdoor_sigma,
            TREE_BASE,
            seed,
            &circuit,
            &enc_one.pubkey,
            &pubkeys[1..],
            &c_b0,
            &c_b,
            &storage_dir.to_path_buf(),
            pk_evaluator.wee25_public_params.clone(),
        );

    let encoding_one = Arc::new(enc_one.clone());
    let input_encodings: Vec<Arc<_>> = encodings[1..].iter().cloned().map(Arc::new).collect();
    let encoding_out = circuit.eval(&params, &encoding_one, &input_encodings, Some(&enc_evaluator));
    assert_eq!(encoding_out.len(), 1);

    assert_eq!(encoding_out[0].pubkey, pubkey_out[0]);
    let expected_poly = DCRTPoly::from_biguint_to_constant(&params, expected.clone());
    let expected_times_gadget =
        s_vec.clone() * (DCRTPolyMatrix::gadget_matrix(&params, d_secret) * expected_poly.clone());
    let s_times_pk = s_vec.clone() * &pubkey_out[0].matrix;
    let diff = encoding_out[0].vector.clone() - s_times_pk + expected_times_gadget;
    let coeff = diff
        .entry(0, 0)
        .coeffs()
        .into_iter()
        .next()
        .expect("diff poly must have at least one coefficient")
        .value()
        .clone();

    let random_int: u64 = rand::random::<u64>() % q_max;
    let q_over_qmax = full_q.as_ref() / BigUint::from(q_max);
    let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
    let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
    let decoded_random: u64 = (&rounded % BigUint::from(q_max))
        .try_into()
        .expect("decoded random coefficient must fit u64");
    assert_eq!(decoded_random, random_int);
}
