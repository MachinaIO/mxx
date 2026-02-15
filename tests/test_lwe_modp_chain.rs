use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    element::PolyElem,
    lookup::{
        PublicLut,
        lwe_eval::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
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
    simulator::{SimulatorContext, error_norm::NormPltLWEEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use rand::Rng;
use std::{fs, path::Path, sync::Arc};
use tracing::info;

const CRT_BITS: usize = 51;
const RING_DIM: u32 = 1 << 14;
const ERROR_SIGMA: f64 = 4.0;
const BASE_BITS: u32 = 17;
const MAX_CRT_DEPTH: usize = 12;
const P: u64 = 7;
const D_SECRET: usize = 2;

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn decode_mod_p_from_scaled_plaintext(plaintext: &DCRTPoly, q_over_p: &BigUint, p: u64) -> u64 {
    let coeff = plaintext
        .coeffs()
        .into_iter()
        .next()
        .expect("plaintext polynomial must have at least one coefficient");
    let rounded = round_div_biguint(coeff.value(), q_over_p);
    (&rounded % BigUint::from(p)).try_into().expect("decoded value must fit u64")
}

fn max_output_row_from_biguint(
    params: &DCRTPolyParams,
    idx: usize,
    value: BigUint,
) -> (usize, <DCRTPoly as Poly>::Elem) {
    let poly = DCRTPoly::from_biguint_to_constant(params, value);
    let coeff =
        poly.coeffs().into_iter().max().expect("max_output_row_from_biguint requires coefficients");
    (idx, coeff)
}

fn build_mod_p_lut(params: &DCRTPolyParams, p: u64) -> PublicLut<DCRTPoly> {
    let lut_len = (p * p) as usize;
    let max_row = max_output_row_from_biguint(params, (p - 1) as usize, BigUint::from(p - 1));
    PublicLut::<DCRTPoly>::new_from_usize_range(
        params,
        lut_len,
        move |params, t| {
            let output = BigUint::from((t as u64) % p);
            (t, DCRTPoly::from_biguint_to_constant(params, output))
        },
        Some(max_row),
    )
}

fn build_modp_chain_circuit(
    params: &DCRTPolyParams,
    p: u64,
    q_over_p: &BigUint,
) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(3);

    let lut_id = circuit.register_public_lookup(build_mod_p_lut(params, p));

    let t1 = circuit.mul_gate(inputs[0], inputs[1]);
    let t1_mod = circuit.public_lookup_gate(t1, lut_id);

    let t2 = circuit.mul_gate(t1_mod, inputs[2]);
    let t2_mod = circuit.public_lookup_gate(t2, lut_id);

    let scalar = DCRTPoly::from_biguint_to_constant(params, q_over_p.clone());
    let scaled = circuit.poly_scalar_mul(t2_mod, &scalar);
    circuit.output(vec![scaled]);
    circuit
}

fn find_crt_depth_for_modp_chain() -> (usize, DCRTPolyParams, PolyCircuit<DCRTPoly>, BigUint) {
    let ring_dim_sqrt = BigDecimal::from_u32(RING_DIM).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << BASE_BITS, 0);
    let error_sigma = BigDecimal::from_f64(ERROR_SIGMA).expect("valid error sigma");

    for crt_depth in 1..=MAX_CRT_DEPTH {
        let params = DCRTPolyParams::new(RING_DIM, crt_depth, CRT_BITS, BASE_BITS);
        let (q_moduli, _q_bits, _q_depth) = params.to_crt();
        let q_moduli_min = *q_moduli.iter().min().expect("q_moduli must not be empty");
        assert!(
            (P as u128) * (P as u128) < q_moduli_min as u128,
            "p^2 must be smaller than any CRT modulus"
        );

        let q = params.modulus();
        let q_over_p = q.as_ref() / BigUint::from(P);
        let q_over_2p = q.as_ref() / BigUint::from(2 * P);
        let circuit = build_modp_chain_circuit(&params, P, &q_over_p);

        let log_base_q = params.modulus_digits();
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            D_SECRET,
            log_base_q,
        ));
        let plt_evaluator = NormPltLWEEvaluator::new(ctx.clone(), &error_sigma);
        let e_init_norm = &error_sigma * BigDecimal::from(6u64);
        let input_bound = BigDecimal::from((P - 1) as u64);

        let out_errors = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound,
            circuit.num_input(),
            &e_init_norm,
            Some(&plt_evaluator),
        );

        let max_error = out_errors
            .into_iter()
            .map(|e| e.matrix_norm.poly_norm.norm)
            .max_by(|a, b| a.partial_cmp(b).expect("comparable BigDecimal"))
            .expect("non-empty output");

        // Post-processing projection by G^{-1}(unit_vector) is a 1-hot selection path here,
        // so we conservatively reuse max_error without extra inflation.
        let projected_max_error = max_error.clone();
        let q_over_2p_bd = BigDecimal::from_biguint(q_over_2p.clone(), 0);
        info!(
            "crt_depth={} q_over_p_bits={} q_over_2p_bits={} max_error_bits={} projected_max_error_bits={}",
            crt_depth,
            q_over_p.bits(),
            q_over_2p.bits(),
            bigdecimal_bits_ceil(&max_error),
            bigdecimal_bits_ceil(&projected_max_error)
        );
        if projected_max_error < q_over_2p_bd {
            return (crt_depth, params, circuit, q_over_p);
        }
    }

    panic!("crt_depth satisfying error < q/p not found up to MAX_CRT_DEPTH");
}

#[tokio::test]
async fn test_lwe_modp_chain_rounding() {
    let _ = tracing_subscriber::fmt::try_init();

    let (_crt_depth, params, circuit, q_over_p) = find_crt_depth_for_modp_chain();
    info!(
        "selected crt_depth={} ring_dim={} base_bits={}",
        params.crt_depth(),
        params.ring_dimension(),
        params.base_bits()
    );
    let mut rng = rand::rng();
    let a: u64 = rng.random_range(0..P);
    let b: u64 = rng.random_range(0..P);
    let c: u64 = rng.random_range(0..P);
    let expected_mod_p = ((a * b) % P) * c % P;

    let plaintexts = vec![
        DCRTPoly::from_usize_to_constant(&params, a as usize),
        DCRTPoly::from_usize_to_constant(&params, b as usize),
        DCRTPoly::from_usize_to_constant(&params, c as usize),
    ];

    let d_secret = D_SECRET;
    let key: [u8; 32] = [
        0x5f, 0x92, 0x10, 0x6a, 0xa0, 0xd4, 0x55, 0xf2, 0x64, 0x96, 0x74, 0x8f, 0xc6, 0xdb, 0xed,
        0x20, 0x1e, 0x0c, 0xd6, 0x6a, 0xe4, 0xa4, 0x3a, 0x9a, 0x41, 0xb5, 0xf8, 0x30, 0xe7, 0x3a,
        0x08, 0xd9,
    ];
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d_secret);
    let tag_bytes: &[u8] = b"bgg_pubkey";

    let uniform_sampler = DCRTPolyUniformSampler::new();
    let s = uniform_sampler.sample_uniform(&params, 1, d_secret - 1, DistType::BitDist).get_row(0);
    let mut secrets = s;
    secrets.push(DCRTPoly::const_minus_one(&params));
    let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets.to_vec());

    let reveal_plaintexts = vec![true; circuit.num_input()];
    let pubkeys = bgg_pubkey_sampler.sample(&params, tag_bytes, &reveal_plaintexts);
    let bgg_encoding_sampler =
        BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
    let enc_one = encodings[0].clone();
    let input_pubkeys = pubkeys[1..].to_vec();
    let input_encodings = encodings[1..].to_vec();

    let trapdoor_sigma = 4.578;
    let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, trapdoor_sigma);
    let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, d_secret);
    let b0_matrix = Arc::new(b0_matrix);
    let b0_trapdoor = Arc::new(b0_trapdoor);

    let dir = Path::new("test_data/lwe_modp_chain_rounding");
    if !dir.exists() {
        fs::create_dir_all(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    info!("plt pubkey evaluator setup start");
    let plt_pubkey_evaluator =
        LWEBGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(
            key, trapdoor_sampler.clone(), b0_matrix.clone(), b0_trapdoor, dir.to_path_buf()
        );
    info!("plt pubkey evaluator setup done");

    info!("circuit eval pubkey start");
    let result_pubkey =
        circuit.eval(&params, &enc_one.pubkey, &input_pubkeys, Some(&plt_pubkey_evaluator));
    info!("circuit eval pubkey done");
    assert_eq!(result_pubkey.len(), 1);
    plt_pubkey_evaluator.sample_aux_matrices(&params);

    wait_for_all_writes(dir.to_path_buf()).await.unwrap();

    let c_b = s_vec.clone() * b0_matrix.as_ref();
    info!("plt encoding evaluator setup start");
    let plt_encoding_evaluator = LWEBGGEncodingPltEvaluator::<
        DCRTPolyMatrix,
        DCRTPolyHashSampler<Keccak256>,
    >::new(key, dir.to_path_buf(), c_b);
    info!("plt encoding evaluator setup done");

    info!("circuit eval encoding start");
    let result_encoding =
        circuit.eval(&params, &enc_one, &input_encodings, Some(&plt_encoding_evaluator));
    info!("circuit eval encoding done");
    assert_eq!(result_encoding.len(), 1);

    let encoding = &result_encoding[0];
    let s_times_pk = s_vec.clone() * &encoding.pubkey.matrix;
    let diff = encoding.vector.clone() - s_times_pk;

    let expected = BigUint::from(expected_mod_p);
    let expected_plaintext =
        DCRTPoly::from_biguint_to_constant(&params, expected * q_over_p.clone());
    let plaintext =
        encoding.plaintext.as_ref().expect("encoding plaintext should be available for this test");
    assert_eq!(
        plaintext, &expected_plaintext,
        "symbolic plaintext must match expected scaled mod-p value"
    );
    let decoded_from_plaintext = decode_mod_p_from_scaled_plaintext(plaintext, &q_over_p, P);
    assert_eq!(
        decoded_from_plaintext, expected_mod_p,
        "decoded value from encoding plaintext must match mod p result"
    );

    let unit_vector = DCRTPolyMatrix::unit_column_vector(&params, d_secret, d_secret - 1);
    let projected = diff.mul_decompose(&unit_vector);
    assert_eq!(projected.row_size(), 1);
    assert_eq!(projected.col_size(), 1);
    let projected_poly = projected.entry(0, 0);
    let coeff = projected_poly
        .coeffs()
        .into_iter()
        .next()
        .expect("projected poly must have at least one coefficient");
    let rounded = round_div_biguint(coeff.value(), &q_over_p);
    let decoded: u64 =
        (&rounded % BigUint::from(P)).try_into().expect("decoded coefficient must fit u64");
    assert_eq!(
        decoded, expected_mod_p,
        "decoded value from projected encoding.vector must match mod p result"
    );
}
