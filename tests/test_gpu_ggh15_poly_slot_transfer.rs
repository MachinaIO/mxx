#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bgg::sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    circuit::{PolyCircuit, gate::PolyGateKind},
    element::PolyElem,
    lookup::{
        PltEvaluator, PublicLut,
        ggh15_eval::{GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_count, detected_gpu_device_ids,
                gpu_device_sync,
            },
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{
        SimulatorContext,
        error_norm::{ErrorNorm, NormBggPolyEncodingSTEvaluator, NormPltGGH15Evaluator},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use rand::Rng;
use std::{env, fs, path::Path, sync::Arc};
use tracing::info;

const DEFAULT_RING_DIM: u32 = 1 << 5;
const DEFAULT_CRT_BITS: usize = 24;
const DEFAULT_BASE_BITS: u32 = 12;
const DEFAULT_MAX_CRT_DEPTH: usize = 24;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 2;
const DEFAULT_LUT_SIZE: usize = 16;
const TRAPDOOR_SIGMA: f64 = 4.578;

const INPUT_CONSTANTS: [u64; 3] = [5, 9, 4];
const SRC_SLOTS: [(u32, Option<u32>); 3] = [(2, Some(4)), (0, None), (1, Some(3))];

#[derive(Debug, Clone)]
struct SlotTransferConfig {
    ring_dim: u32,
    crt_bits: usize,
    base_bits: u32,
    max_crt_depth: usize,
    error_sigma: f64,
    d_secret: usize,
    lut_size: usize,
    dir_name_override: Option<String>,
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be a valid u32: {e}")),
        Err(_) => default,
    }
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    match env::var(key) {
        Ok(raw) => {
            raw.parse::<usize>().unwrap_or_else(|e| panic!("{key} must be a valid usize: {e}"))
        }
        Err(_) => default,
    }
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    match env::var(key) {
        Ok(raw) => raw.parse::<f64>().unwrap_or_else(|e| panic!("{key} must be a valid f64: {e}")),
        Err(_) => default,
    }
}

impl SlotTransferConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("GGH15_POLY_SLOT_TRANSFER_RING_DIM", DEFAULT_RING_DIM);
        let crt_bits = env_or_parse_usize("GGH15_POLY_SLOT_TRANSFER_CRT_BITS", DEFAULT_CRT_BITS);
        let base_bits = env_or_parse_u32("GGH15_POLY_SLOT_TRANSFER_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("GGH15_POLY_SLOT_TRANSFER_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let error_sigma =
            env_or_parse_f64("GGH15_POLY_SLOT_TRANSFER_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("GGH15_POLY_SLOT_TRANSFER_D_SECRET", DEFAULT_D_SECRET);
        let lut_size = env_or_parse_usize("GGH15_POLY_SLOT_TRANSFER_LUT_SIZE", DEFAULT_LUT_SIZE);
        let dir_name_override = env::var("GGH15_POLY_SLOT_TRANSFER_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "GGH15_POLY_SLOT_TRANSFER_RING_DIM must be > 0");
        assert!(
            ring_dim as usize >= INPUT_CONSTANTS.len(),
            "GGH15_POLY_SLOT_TRANSFER_RING_DIM must cover {} slots",
            INPUT_CONSTANTS.len()
        );
        assert!(crt_bits > 0, "GGH15_POLY_SLOT_TRANSFER_CRT_BITS must be > 0");
        assert!(base_bits > 0, "GGH15_POLY_SLOT_TRANSFER_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_POLY_SLOT_TRANSFER_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma >= 0.0, "GGH15_POLY_SLOT_TRANSFER_ERROR_SIGMA must be >= 0");
        assert!(d_secret > 0, "GGH15_POLY_SLOT_TRANSFER_D_SECRET must be > 0");
        assert!(lut_size > 0, "GGH15_POLY_SLOT_TRANSFER_LUT_SIZE must be > 0");
        assert!(
            INPUT_CONSTANTS.iter().all(|&x| x < lut_size as u64),
            "GGH15_POLY_SLOT_TRANSFER_LUT_SIZE must exceed all input constants"
        );

        Self {
            ring_dim,
            crt_bits,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            lut_size,
            dir_name_override,
        }
    }

    fn dir_name(&self, crt_depth: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!("test_data/test_gpu_ggh15_poly_slot_transfer_crtdepth_{crt_depth}")
        })
    }
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn build_lsb_bit_lut_cpu(lut_size: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
    PublicLut::<DCRTPoly>::new(
        params,
        lut_size as u64,
        move |params, k| {
            if k >= lut_size as u64 {
                return None;
            }
            let y_elem = <<DCRTPoly as Poly>::Elem as PolyElem>::constant(&params.modulus(), k % 2);
            Some((k, y_elem))
        },
        None,
    )
}

fn build_lsb_bit_lut_gpu(lut_size: usize, params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
    PublicLut::<GpuDCRTPoly>::new(
        params,
        lut_size as u64,
        move |params, k| {
            if k >= lut_size as u64 {
                return None;
            }
            let y_elem =
                <<GpuDCRTPoly as Poly>::Elem as PolyElem>::constant(&params.modulus(), k % 2);
            Some((k, y_elem))
        },
        None,
    )
}

fn build_simple_slot_transfer_circuit_cpu(
    params: &DCRTPolyParams,
    lut_size: usize,
) -> PolyCircuit<DCRTPoly> {
    let plt = build_lsb_bit_lut_cpu(lut_size, params);
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let inputs = circuit.input(1);
    let lut_id = circuit.register_public_lookup(plt);
    let looked_up = circuit.public_lookup_gate(inputs.at(0), lut_id);
    let transferred = circuit.slot_transfer_gate(looked_up, &SRC_SLOTS);
    circuit.output(vec![transferred]);
    circuit
}

fn build_simple_slot_transfer_circuit_gpu(
    params: &GpuDCRTPolyParams,
    lut_size: usize,
) -> PolyCircuit<GpuDCRTPoly> {
    let plt = build_lsb_bit_lut_gpu(lut_size, params);
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let inputs = circuit.input(1);
    let lut_id = circuit.register_public_lookup(plt);
    let looked_up = circuit.public_lookup_gate(inputs.at(0), lut_id);
    let transferred = circuit.slot_transfer_gate(looked_up, &SRC_SLOTS);
    circuit.output(vec![transferred]);
    circuit
}

fn simulate_max_error_norm_with_slot_transfer<P: PltEvaluator<ErrorNorm>>(
    circuit: &PolyCircuit<DCRTPoly>,
    ctx: Arc<SimulatorContext>,
    input_norm_bound: BigDecimal,
    input_size: usize,
    e_init_norm: &BigDecimal,
    plt_evaluator: &P,
    slot_transfer_evaluator: &NormBggPolyEncodingSTEvaluator,
) -> Vec<ErrorNorm> {
    let one_error = ErrorNorm::new(
        PolyNorm::one(ctx.clone()),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
    );
    let input_error = ErrorNorm::new(
        PolyNorm::new(ctx.clone(), input_norm_bound),
        PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
    );
    let input_errors = vec![input_error; input_size];
    circuit.eval(
        &(),
        one_error,
        input_errors,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        None,
    )
}

fn find_crt_depth_for_simple_slot_transfer(cfg: &SlotTransferConfig) -> DCRTPolyParams {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();
    let input_bound =
        BigDecimal::from(*INPUT_CONSTANTS.iter().max().expect("input constants must not be empty"));

    for crt_depth in 1..=cfg.max_crt_depth {
        let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
        let circuit = build_simple_slot_transfer_circuit_cpu(&params, cfg.lut_size);
        let q_moduli = params.to_crt().0;
        let q_max = *q_moduli.iter().max().expect("CRT modulus list must not be empty");

        let log_base_q = params.modulus_digits();
        let log_base_q_small = log_base_q / crt_depth;
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt.clone(),
            base.clone(),
            cfg.d_secret,
            log_base_q,
            log_base_q_small,
        ));
        let plt_evaluator =
            NormPltGGH15Evaluator::new(ctx.clone(), &error_sigma, &error_sigma, None);
        let slot_transfer_evaluator =
            NormBggPolyEncodingSTEvaluator::new(ctx.clone(), cfg.error_sigma, &error_sigma, None);

        let out_errors = simulate_max_error_norm_with_slot_transfer(
            &circuit,
            ctx,
            input_bound.clone(),
            circuit.num_input(),
            &e_init_norm,
            &plt_evaluator,
            &slot_transfer_evaluator,
        );
        assert_eq!(out_errors.len(), 1);

        let threshold = params.modulus().as_ref() / BigUint::from(2u64 * q_max);
        let error = &out_errors[0].matrix_norm.poly_norm.norm;
        info!(
            "crt_depth={} q_bits={} max_error_bits={} threshold_bits={}",
            crt_depth,
            params.modulus_bits(),
            bigdecimal_bits_ceil(error),
            bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold.clone(), 0))
        );
        if *error < BigDecimal::from_biguint(threshold, 0) {
            info!(
                "selected crt_depth={} for simple slot-transfer circuit with max_error_bits={}",
                crt_depth,
                bigdecimal_bits_ceil(error)
            );
            return params;
        }
    }

    panic!(
        "crt_depth satisfying error < q / (2 * q_max) was not found up to {}",
        cfg.max_crt_depth
    );
}

fn expected_slot_outputs() -> Vec<u64> {
    let lookup_outputs = INPUT_CONSTANTS.iter().map(|value| value % 2).collect::<Vec<_>>();
    SRC_SLOTS
        .iter()
        .map(|(src_slot, scalar)| {
            let mut value = lookup_outputs[*src_slot as usize];
            if let Some(scalar) = scalar {
                value *= u64::from(*scalar);
            }
            value
        })
        .collect()
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_ggh15_poly_slot_transfer() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    gpu_device_sync();

    let cfg = SlotTransferConfig::from_env();
    info!("simple slot-transfer config: {:?}", cfg);

    let cpu_params = find_crt_depth_for_simple_slot_transfer(&cfg);
    let crt_depth = cpu_params.to_crt().2;
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_ids = detected_gpu_device_ids();
    let detected_gpu_count = detected_gpu_device_count();
    assert_eq!(
        detected_gpu_count,
        detected_gpu_ids.len(),
        "detected GPU count and ids length must match"
    );
    let single_gpu_id = *detected_gpu_ids
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_poly_slot_transfer");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    assert_eq!(params.modulus(), cpu_params.modulus());

    let circuit = build_simple_slot_transfer_circuit_gpu(&params, cfg.lut_size);
    assert_eq!(
        circuit.non_free_depth_contributions().values().sum::<usize>(),
        2,
        "circuit depth should stay simple"
    );
    let gate_counts = circuit.count_gates_by_type_vec();
    assert_eq!(gate_counts.get(&PolyGateKind::PubLut), Some(&1));
    assert_eq!(gate_counts.get(&PolyGateKind::SlotTransfer), Some(&1));

    let seed = [0u8; 32];
    let tag: u64 = 7;
    let dir_name = cfg.dir_name(crt_depth);
    let dir = Path::new(&dir_name);
    if dir.exists() {
        fs::remove_dir_all(dir).expect("failed to clear existing test directory");
    }
    fs::create_dir_all(dir).expect("failed to create test directory");
    init_storage_system(dir.to_path_buf());

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());

    let pk_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(seed, cfg.d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    let mut pubkeys = pk_sampler.sample(&params, &tag.to_le_bytes(), &reveal_plaintexts);
    let pubkeys_for_poly_encodings = pubkeys.clone();

    let plt_pubkey_evaluator =
        GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(seed, cfg.d_secret, TRAPDOOR_SIGMA, cfg.error_sigma, dir.to_path_buf());
    let slot_pubkey_evaluator = BggPublicKeySTEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >::new(
        seed,
        cfg.d_secret,
        INPUT_CONSTANTS.len(),
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        dir.to_path_buf(),
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("const-one public key must exist");
    let pubkey_out = circuit.eval(
        &params,
        one_pubkey,
        input_pubkeys,
        Some(&plt_pubkey_evaluator),
        Some(&slot_pubkey_evaluator),
        None,
    );
    assert_eq!(pubkey_out.len(), 1);

    plt_pubkey_evaluator.sample_aux_matrices(&params);
    slot_pubkey_evaluator.sample_aux_matrices(&params);
    wait_for_all_writes(dir.to_path_buf()).await.expect("storage writes should complete");

    let slot_secret_mats = slot_pubkey_evaluator
        .load_slot_secret_mats_checkpoint(&params)
        .expect("slot secret matrix checkpoints should exist after auxiliary sampling");

    let plaintext_rows = vec![
        INPUT_CONSTANTS
            .iter()
            .map(|&value| {
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, value as usize).to_compact_bytes(),
                )
            })
            .collect::<Vec<_>>(),
    ];
    let encoding_sampler = BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &secrets,
        Some(cfg.error_sigma),
    );
    let mut poly_encodings = encoding_sampler.sample(
        &params,
        &pubkeys_for_poly_encodings,
        &plaintext_rows,
        Some(&slot_secret_mats),
    );

    let plt_b0_matrix = plt_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("public-lookup b0 checkpoint should exist after auxiliary sampling");
    let slot_b0_matrix = slot_pubkey_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("slot-transfer b0 checkpoint should exist after auxiliary sampling");
    let plt_c_b0_compact_bytes_by_slot = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::build_c_b0_compact_bytes_by_slot::<
        GpuDCRTPolyUniformSampler,
    >(
        &params,
        &s_vec,
        &plt_b0_matrix,
        &slot_secret_mats,
        Some(cfg.error_sigma),
    );
    let mut slot_c_b0 = s_vec.clone() * &slot_b0_matrix;
    if cfg.error_sigma != 0.0 {
        let slot_c_b0_error = uniform_sampler.sample_uniform(
            &params,
            slot_c_b0.row_size(),
            slot_c_b0.col_size(),
            DistType::GaussDist { sigma: cfg.error_sigma },
        );
        slot_c_b0 = slot_c_b0 + slot_c_b0_error;
    }

    let plt_poly_evaluator = GGH15BGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(
        seed,
        dir.to_path_buf(),
        plt_pubkey_evaluator.checkpoint_prefix(&params),
        plt_c_b0_compact_bytes_by_slot,
    );
    let slot_poly_evaluator =
        BggPolyEncodingSTEvaluator::<GpuDCRTPolyMatrix, GpuDCRTPolyHashSampler<Keccak256>>::new(
            seed,
            dir.to_path_buf(),
            slot_pubkey_evaluator.checkpoint_prefix(&params),
            slot_c_b0.to_compact_bytes(),
        );

    let input_poly_encodings = poly_encodings.split_off(1);
    let one_poly_encoding = poly_encodings.pop().expect("const-one poly encoding must exist");
    let poly_out = circuit.eval(
        &params,
        one_poly_encoding,
        input_poly_encodings,
        Some(&plt_poly_evaluator),
        Some(&slot_poly_evaluator),
        None,
    );
    assert_eq!(poly_out.len(), 1);

    let poly_out = &poly_out[0];
    assert_eq!(poly_out.pubkey, pubkey_out[0]);
    assert_eq!(poly_out.num_slots(), INPUT_CONSTANTS.len());

    let expected_outputs = expected_slot_outputs();
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, cfg.d_secret);
    let q_max = params.to_crt().0.into_iter().max().expect("CRT modulus list must not be empty");
    let q_over_qmax = params.modulus().as_ref() / BigUint::from(q_max);
    let mut rng = rand::rng();

    for (slot, expected_value) in expected_outputs.into_iter().enumerate() {
        let result_plaintext = poly_out
            .plaintext_for_params(&params, slot)
            .expect("poly output should reveal plaintexts");
        let expected_poly = GpuDCRTPoly::from_usize_to_constant(&params, expected_value as usize);
        assert_eq!(
            result_plaintext, expected_poly,
            "result plaintext polynomial must match expected output at slot {}",
            slot
        );

        let slot_secret_mat =
            GpuDCRTPolyMatrix::from_compact_bytes(&params, slot_secret_mats[slot].as_ref());
        let transformed_secret_vec = s_vec.clone() * &slot_secret_mat;
        let expected_times_gadget =
            transformed_secret_vec.clone() * (gadget.clone() * poly_out.plaintext(slot).unwrap());
        let s_times_pk = transformed_secret_vec.clone() * &poly_out.pubkey.matrix;
        let diff = poly_out.vector(slot) - s_times_pk + expected_times_gadget;
        let coeff = diff
            .entry(0, 0)
            .coeffs()
            .into_iter()
            .next()
            .expect("diff polynomial must contain at least one coefficient")
            .value()
            .clone();

        let random_int = rng.random_range(0..q_max);
        let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
        let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
        let decoded_random: u64 = (&rounded % BigUint::from(q_max))
            .try_into()
            .expect("decoded coefficient must fit in u64");
        assert_eq!(decoded_random, random_int);
    }

    gpu_device_sync();
}
