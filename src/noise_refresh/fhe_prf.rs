//! Evaluable representation helpers for the noise-refresh FHE PRF.

use crate::{
    circuit::evaluable::Evaluable,
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::RingGswContext,
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    slot_transfer::SlotTransferEvaluator,
};
use rayon::prelude::*;
use std::sync::Arc;

use super::circuits::{
    NoiseRefreshPrgParameters, build_next_seed_encodings_circuit_with_output_chunk_limit,
};

fn combine_next_seed_encoding_chunks<M, E>(
    params: &<M::P as Poly>::Params,
    next_seed_chunk: &[E],
    errors_plus_masks_chunk: &[E],
    unit_column_target: &M,
) -> Vec<E>
where
    M: PolyMatrix,
    E: Evaluable<P = M::P, Params = <M::P as Poly>::Params>,
{
    let log_base_q = params.modulus_digits();
    let crt_depth = next_seed_chunk.len();
    assert_ne!(crt_depth, 0, "crt_depth must be positive");
    assert_eq!(
        errors_plus_masks_chunk.len(),
        log_base_q * crt_depth,
        "errors_plus_masks chunk length must be log_base_q * crt_depth"
    );

    let crt_terms = next_seed_chunk
        .par_iter()
        .enumerate()
        .map(|(crt_idx, next_seed_term)| {
            let refresh_columns = errors_plus_masks_chunk
                [crt_idx * log_base_q..(crt_idx + 1) * log_base_q]
                .par_iter()
                .map(|refresh_encoding| refresh_encoding.matrix_mul(params, unit_column_target))
                .collect::<Vec<_>>();
            let (first_refresh, rest_refresh) =
                refresh_columns.split_first().expect("log_base_q must be positive");
            let refresh_term = first_refresh.concat_columns(rest_refresh);
            next_seed_term.clone() + &refresh_term
        })
        .collect::<Vec<_>>();

    crt_terms
}

/// Evaluates the next-seed-encoding circuit over an arbitrary `Evaluable` representation.
///
/// The input `enc_seeds` must have exactly `ciphertext_wire_count * seed_bits` entries, one for
/// each flattened encrypted seed ciphertext wire.  The constant-one representation is passed
/// separately as `one`, matching the `PolyCircuit::eval` API.  The `plt_evaluator` and
/// `slot_transfer_evaluator` are also explicit required arguments and are forwarded to
/// `PolyCircuit::eval` as `Some(...)`, so the caller always controls the evaluator set used by
/// circuit evaluation.  The next-seed-encoding circuit also uses the constant-one wire as the
/// encoded Ring-GSW decryption key, so it is not included in the ordinary circuit input vector.
///
/// The final compression keeps CRT levels separate.  For each output chunk and each CRT level, it
/// adds the circuit-produced `q/q_i`-scaled next-seed term to the decoded error-plus-mask term for
/// the same CRT modulus.  The decoded refresh columns all use the same multiplication by
/// `G^-1(u_1)`; the resulting one-column values are concatenated into the `log_base_q` refresh
/// columns before being added to the corresponding next-seed term.
pub fn fhe_prf_evaluable_for_step_fn<P, A, M, E, PE, ST>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    one: &E,
    enc_seeds: &[E],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    parallel_gates: Option<usize>,
) -> Vec<E>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
    E: Evaluable<P = P, Params = <P as Poly>::Params>,
    PE: PltEvaluator<E>,
    ST: SlotTransferEvaluator<E>,
{
    fhe_prf_evaluable_for_step_fn_with_output_chunk_limit::<P, A, M, E, PE, ST>(
        ring_gsw,
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        one,
        enc_seeds,
        plt_evaluator,
        slot_transfer_evaluator,
        parallel_gates,
        None,
    )
}

pub fn fhe_prf_evaluable_for_step_fn_with_output_chunk_limit<P, A, M, E, PE, ST>(
    ring_gsw: Arc<RingGswContext<P, A>>,
    seed_bits: usize,
    input_bits_per_step: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    one: &E,
    enc_seeds: &[E],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    parallel_gates: Option<usize>,
    output_chunk_limit: Option<usize>,
) -> Vec<E>
where
    P: Poly + 'static,
    A: DecomposeArithmeticGadget<P> + ModularArithmeticPlanner<P>,
    M: PolyMatrix<P = P>,
    E: Evaluable<P = P, Params = <P as Poly>::Params>,
    PE: PltEvaluator<E>,
    ST: SlotTransferEvaluator<E>,
{
    let secret_size = 1usize;
    let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
    let expected_enc_seeds = ciphertext_wire_count
        .checked_mul(seed_bits)
        .expect("noise-refresh FHE PRF input count overflow");
    assert_eq!(
        enc_seeds.len(),
        expected_enc_seeds,
        "fhe_prf_evaluable_for_step_fn expects one value per encrypted seed ciphertext wire"
    );

    let circuit = build_next_seed_encodings_circuit_with_output_chunk_limit::<P, A, M>(
        ring_gsw.clone(),
        seed_bits,
        input_bits_per_step,
        v_bits,
        graph_seed,
        cbd_n,
        output_chunk_limit,
    );

    let circuit_outputs = circuit.eval(
        &ring_gsw.params,
        one.clone(),
        enc_seeds.to_vec(),
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        parallel_gates,
    );

    let expanded_seed_bits = seed_bits
        .checked_mul(input_bits_per_step)
        .expect("seed_bits * input_bits_per_step overflow");
    let full_output_chunk_count = expanded_seed_bits
        .checked_mul(ciphertext_wire_count)
        .expect("next-seed output chunk count overflow");
    let output_chunk_count = output_chunk_limit.unwrap_or(full_output_chunk_count);
    assert!(
        output_chunk_count <= full_output_chunk_count,
        "output chunk limit must not exceed the full next-seed wire count"
    );
    let log_base_q = ring_gsw.log_base_q();
    let crt_depth = ring_gsw.crt_depth();
    let next_seed_output_len = output_chunk_count
        .checked_mul(crt_depth)
        .expect("CRT-scaled next-seed output count overflow");
    let errors_plus_masks_output_len = output_chunk_count
        .checked_mul(log_base_q)
        .and_then(|value| value.checked_mul(crt_depth))
        .expect("decoded refresh output count overflow");
    assert_eq!(
        circuit_outputs.len(),
        next_seed_output_len + errors_plus_masks_output_len,
        "next-seed-encoding circuit output count must match next_seed plus decoded refresh suffix"
    );
    let (next_seed_outputs, errors_plus_masks_outputs) =
        circuit_outputs.split_at(next_seed_output_len);

    let params = &ring_gsw.params;
    let unit_column_target =
        M::scaled_unit_column_vector(params, secret_size, 0, M::P::const_one(params));
    let (q_moduli, _crt_bits, q_moduli_depth) = params.to_crt();
    assert_eq!(q_moduli_depth, crt_depth, "Ring-GSW CRT depth must match params.to_crt()");
    assert_eq!(q_moduli.len(), crt_depth, "one CRT modulus is required for each CRT level");

    next_seed_outputs
        .par_chunks_exact(crt_depth)
        .zip(errors_plus_masks_outputs.par_chunks_exact(log_base_q * crt_depth))
        .flat_map(|(next_seed_chunk, errors_plus_masks_chunk)| {
            combine_next_seed_encoding_chunks::<M, E>(
                params,
                next_seed_chunk,
                errors_plus_masks_chunk,
                &unit_column_target,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use crate::{
        __PAIR, __TestState,
        bgg::{
            poly_encoding::BggPolyEncoding,
            sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
        },
        circuit::{PolyCircuit, evaluable::PolyVec},
        gadgets::arith::{DEFAULT_MAX_UNREDUCED_MULS, NestedRnsPoly, NestedRnsPolyContext},
        lookup::{
            lwe::{LWEBGGPolyEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
            poly::PolyPltEvaluator,
            poly_vec::PolyVecPltEvaluator,
        },
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::dcrt::gpu::{
            GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync,
        },
        sampler::{
            PolyTrapdoorSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::{
            PolyVecSlotTransferEvaluator, bgg_poly_encoding::BggPolyEncodingSTEvaluator,
            bgg_pubkey::BggPublicKeySTEvaluator,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use crate::{
        bgg::public_key::BggPublicKey,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    #[cfg(feature = "gpu")]
    use keccak_asm::Keccak256;
    #[cfg(feature = "gpu")]
    use std::{path::Path, sync::Arc, time::Instant};
    #[cfg(feature = "gpu")]
    use tracing::info;

    #[cfg(feature = "gpu")]
    fn log_relation_step(test_start: Instant, step_start: &mut Instant, label: &str) {
        let now = Instant::now();
        info!(
            target: "noise_refresh::fhe_prf_relation",
            step = label,
            step_elapsed_ms = now.duration_since(*step_start).as_millis(),
            total_elapsed_ms = now.duration_since(test_start).as_millis(),
            "relation test step completed"
        );
        *step_start = now;
    }

    fn constant_matrix(
        params: &DCRTPolyParams,
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            params,
            (0..rows)
                .map(|row| {
                    (0..cols)
                        .map(|col| {
                            DCRTPoly::from_usize_to_constant(params, offset + row * cols + col + 1)
                        })
                        .collect()
                })
                .collect(),
        )
    }

    #[test]
    fn combine_next_seed_encoding_chunks_outputs_log_base_q_columns() {
        let params = DCRTPolyParams::default();
        let secret_size = 1usize;
        let log_base_q = params.modulus_digits();
        let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let gadget_cols = secret_size * log_base_q;
        let next_seed_chunk =
            BggPublicKey::new(constant_matrix(&params, secret_size, gadget_cols, 0), true);
        let errors_plus_masks_chunk = (0..log_base_q * crt_depth)
            .map(|idx| {
                BggPublicKey::new(
                    constant_matrix(&params, secret_size, gadget_cols, 100 + idx * 100),
                    true,
                )
            })
            .collect::<Vec<_>>();
        let unit_column_target = DCRTPolyMatrix::scaled_unit_column_vector(
            &params,
            secret_size,
            0,
            DCRTPoly::const_one(&params),
        );
        let full_q: Arc<num_bigint::BigUint> = params.modulus().into();
        let next_seed_terms = q_moduli
            .par_iter()
            .map(|&q_i| {
                let q_over_qi = full_q.as_ref() / num_bigint::BigUint::from(q_i);
                next_seed_chunk.large_scalar_mul(&params, &[q_over_qi])
            })
            .collect::<Vec<_>>();

        let combined = combine_next_seed_encoding_chunks::<DCRTPolyMatrix, BggPublicKey<_>>(
            &params,
            next_seed_terms.as_slice(),
            errors_plus_masks_chunk.as_slice(),
            &unit_column_target,
        );

        let expected_terms = next_seed_terms
            .par_iter()
            .enumerate()
            .map(|(crt_idx, next_seed_term)| {
                let refresh_columns = errors_plus_masks_chunk
                    [crt_idx * log_base_q..(crt_idx + 1) * log_base_q]
                    .par_iter()
                    .map(|refresh_encoding| {
                        refresh_encoding.matrix_mul(&params, &unit_column_target)
                    })
                    .collect::<Vec<_>>();
                let (first_refresh, rest_refresh) =
                    refresh_columns.split_first().expect("log_base_q must be positive");
                let refresh_term = first_refresh.concat_columns(rest_refresh);
                next_seed_term.clone() + &refresh_term
            })
            .collect::<Vec<_>>();
        assert_eq!(combined.len(), crt_depth);
        assert!(combined.iter().all(|term| term.matrix.col_size() == log_base_q));
        assert_eq!(combined, expected_terms);
    }

    #[cfg(feature = "gpu")]
    type TestArithmetic = NestedRnsPoly<GpuDCRTPoly>;
    #[cfg(feature = "gpu")]
    type TestRingGswContext =
        crate::gadgets::fhe::ring_gsw_nested_rns::NestedRnsRingGswContext<GpuDCRTPoly>;

    #[cfg(feature = "gpu")]
    fn create_test_context(
        circuit: &mut PolyCircuit<GpuDCRTPoly>,
        gpu_ids: Vec<i32>,
    ) -> (GpuDCRTPolyParams, Arc<TestRingGswContext>) {
        let ring_dim = 2u32;
        let num_slots = 2usize;
        let active_levels = 1usize;
        let crt_bits = 12usize;
        let base_bits = 12u32;
        let p_moduli_bits = 5usize;
        let scale = 1u64 << 5;
        let cpu_params = DCRTPolyParams::new(ring_dim, active_levels, crt_bits, base_bits);
        let (moduli, _crt_bits, _crt_depth) = cpu_params.to_crt();
        let params = GpuDCRTPolyParams::new_with_gpu(
            ring_dim,
            moduli,
            base_bits,
            gpu_ids,
            Some(num_slots as u32),
        );
        let nested_rns = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &params,
            p_moduli_bits,
            DEFAULT_MAX_UNREDUCED_MULS,
            scale,
            false,
            Some(active_levels),
        ));
        let ring_gsw = Arc::new(RingGswContext::from_arith_context(
            circuit,
            &params,
            num_slots,
            nested_rns,
            Some(active_levels),
            None,
        ));
        (params, ring_gsw)
    }

    #[cfg(feature = "gpu")]
    fn constant_plaintext_row<P: Poly>(
        params: &P::Params,
        values: impl IntoIterator<Item = usize>,
    ) -> Vec<Arc<[u8]>> {
        values
            .into_iter()
            .map(|value| {
                Arc::<[u8]>::from(P::from_usize_to_constant(params, value).to_compact_bytes())
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn round_qi_scaled_poly<P: Poly>(params: &P::Params, value: &P, q_i: u64) -> P {
        let q: Arc<num_bigint::BigUint> = params.modulus().into();
        let q_i = num_bigint::BigUint::from(q_i);
        let half_q = q.as_ref() / num_bigint::BigUint::from(2u32);
        let rounded_coeffs = value
            .coeffs_biguints()
            .into_iter()
            .map(|coeff| ((coeff * &q_i) + &half_q) / q.as_ref())
            .collect::<Vec<_>>();
        P::from_biguints(params, &rounded_coeffs)
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    #[sequential_test::sequential]
    #[ignore = "expensive Nested RNS BGG relation test; run explicitly when needed"]
    async fn fhe_prf_evaluable_for_step_fn_bgg_poly_encoding_matches_public_key_relation() {
        let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
        let test_start = Instant::now();
        let mut step_start = test_start;
        info!(target: "noise_refresh::fhe_prf_relation", "relation test started");
        let _storage_lock = storage_test_lock().await;
        log_relation_step(test_start, &mut step_start, "acquired storage lock");

        gpu_device_sync();
        let gpu_ids = detected_gpu_device_ids();
        if gpu_ids.is_empty() {
            info!(
                target: "noise_refresh::fhe_prf_relation",
                "skipping GpuDCRTPolyMatrix relation test because no GPU devices were detected"
            );
            return;
        }
        info!(
            target: "noise_refresh::fhe_prf_relation",
            gpu_device_count = gpu_ids.len(),
            ?gpu_ids,
            "detected GPU devices"
        );
        let mut setup_circuit = PolyCircuit::new();
        let (params, ring_gsw) = create_test_context(&mut setup_circuit, gpu_ids);
        let seed_bits = 1usize;
        let input_bits_per_step = 1usize;
        let v_bits = 1usize;
        let cbd_n = 2usize;
        let output_chunk_limit = 1usize;
        let graph_seed = [0x5au8; 32];
        let hash_key = [0x91u8; 32];
        let secret_size = 1usize;
        let num_slots = ring_gsw.ring_dim();
        let ciphertext_wire_count = ring_gsw.encrypted_bit_wire_count();
        let enc_seed_count = seed_bits * ciphertext_wire_count;
        let reveal_plaintexts = vec![true; enc_seed_count];
        info!(
            target: "noise_refresh::fhe_prf_relation",
            seed_bits,
            input_bits_per_step,
            v_bits,
            cbd_n,
            output_chunk_limit,
            ring_dim = num_slots,
            ciphertext_wire_count,
            enc_seed_count,
            log_base_q = params.modulus_digits(),
            crt_depth = ring_gsw.crt_depth(),
            "relation test parameters"
        );
        log_relation_step(test_start, &mut step_start, "created Nested RNS Ring-GSW context");

        let dir_path = "test_data/noise_refresh_fhe_prf_poly_encoding_relation";
        let dir = Path::new(dir_path);
        if dir.exists() {
            std::fs::remove_dir_all(dir).unwrap();
        }
        std::fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
        log_relation_step(test_start, &mut step_start, "initialized storage");

        let pubkey_sampler =
            BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(hash_key, secret_size);
        let public_keys =
            pubkey_sampler.sample(&params, b"noise_refresh_fhe_prf", &reveal_plaintexts);
        log_relation_step(test_start, &mut step_start, "sampled BGG public keys");
        let pubkey_st_evaluator =
            BggPublicKeySTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyUniformSampler,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(hash_key, secret_size, num_slots, 4.578, 0.0, dir.to_path_buf());
        let lwe_trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, 4.578);
        let (lwe_trapdoor, lwe_pub_matrix) = lwe_trapdoor_sampler.trapdoor(&params, secret_size);
        let pubkey_plt_evaluator = LWEBGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(
            hash_key,
            lwe_trapdoor_sampler,
            Arc::new(lwe_pub_matrix.clone()),
            Arc::new(lwe_trapdoor),
            dir.to_path_buf(),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            "evaluating output-limited circuit over BGGPublicKey"
        );
        let pubkey_outputs = fhe_prf_evaluable_for_step_fn_with_output_chunk_limit::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
            BggPublicKey<GpuDCRTPolyMatrix>,
            _,
            _,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            &public_keys[0],
            &public_keys[1..],
            &pubkey_plt_evaluator,
            &pubkey_st_evaluator,
            Some(1),
            Some(output_chunk_limit),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = pubkey_outputs.len(),
            first_col_size = pubkey_outputs.first().map(|pk| pk.matrix.col_size()).unwrap_or(0),
            "BGGPublicKey outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated BGGPublicKey outputs");

        info!(
            target: "noise_refresh::fhe_prf_relation",
            "sampling PublicLUT and slot-transfer auxiliary matrices"
        );
        pubkey_plt_evaluator.sample_aux_matrices(&params);
        pubkey_st_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        log_relation_step(
            test_start,
            &mut step_start,
            "sampled and flushed PublicLUT and slot-transfer aux matrices",
        );
        let slot_secret_mats = pubkey_st_evaluator
            .load_slot_secret_mats_checkpoint(&params)
            .expect("slot secret matrix checkpoints should exist after sample_aux_matrices");
        log_relation_step(test_start, &mut step_start, "loaded slot secret matrices");

        let plaintext_rows = (0..enc_seed_count)
            .map(|input_idx| {
                constant_plaintext_row::<GpuDCRTPoly>(
                    &params,
                    (0..num_slots).map(|slot| (input_idx + slot) % 2),
                )
            })
            .collect::<Vec<_>>();
        log_relation_step(test_start, &mut step_start, "built plaintext rows");
        let secrets = vec![GpuDCRTPoly::const_one(&params)];
        let poly_encoding_sampler =
            BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
        let input_encodings = poly_encoding_sampler.sample(
            &params,
            &public_keys,
            &plaintext_rows,
            Some(&slot_secret_mats),
        );
        log_relation_step(test_start, &mut step_start, "sampled zero-error BGGPolyEncoding inputs");
        let b0_matrix = pubkey_st_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        let base_secret_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
        let c_b0 = base_secret_vec.clone() * &b0_matrix;
        let c_lwe_pub_matrix_compact_bytes_by_slot = LWEBGGPolyEncodingPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::build_c_b_compact_bytes_by_slot::<
            GpuDCRTPolyUniformSampler,
        >(
            &params,
            &base_secret_vec,
            &lwe_pub_matrix,
            &slot_secret_mats,
            None,
        );
        let poly_encoding_plt_evaluator = LWEBGGPolyEncodingPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::new(
            hash_key,
            dir.to_path_buf(),
            c_lwe_pub_matrix_compact_bytes_by_slot,
        );
        let poly_encoding_st_evaluator = BggPolyEncodingSTEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
        >::new(
            hash_key,
            dir.to_path_buf(),
            pubkey_st_evaluator.checkpoint_prefix(&params),
            c_b0.to_compact_bytes(),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            "evaluating output-limited circuit over BGGPolyEncoding"
        );
        let poly_encoding_outputs = fhe_prf_evaluable_for_step_fn_with_output_chunk_limit::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
            BggPolyEncoding<GpuDCRTPolyMatrix>,
            _,
            _,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            &input_encodings[0],
            &input_encodings[1..],
            &poly_encoding_plt_evaluator,
            &poly_encoding_st_evaluator,
            Some(1),
            Some(output_chunk_limit),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = poly_encoding_outputs.len(),
            first_col_size = poly_encoding_outputs
                .first()
                .map(|enc| enc.pubkey.matrix.col_size())
                .unwrap_or(0),
            "BGGPolyEncoding outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated BGGPolyEncoding outputs");
        assert_eq!(poly_encoding_outputs.len(), pubkey_outputs.len());

        info!(
            target: "noise_refresh::fhe_prf_relation",
            "building output-limited PolyVec reference circuit"
        );
        let reference_circuit = build_next_seed_encodings_circuit_with_output_chunk_limit::<
            GpuDCRTPoly,
            TestArithmetic,
            GpuDCRTPolyMatrix,
        >(
            ring_gsw.clone(),
            seed_bits,
            input_bits_per_step,
            v_bits,
            graph_seed,
            cbd_n,
            Some(output_chunk_limit),
        );
        log_relation_step(test_start, &mut step_start, "built PolyVec reference circuit");
        let polyvec_inputs = plaintext_rows
            .iter()
            .map(|row| {
                PolyVec::new(
                    row.iter()
                        .map(|bytes| GpuDCRTPoly::from_compact_bytes(&params, bytes.as_ref()))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let polyvec_one = PolyVec::new(vec![GpuDCRTPoly::const_one(&params); num_slots]);
        let polyvec_plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        info!(target: "noise_refresh::fhe_prf_relation", "evaluating PolyVec reference outputs");
        let reference_outputs = reference_circuit.eval(
            &params,
            polyvec_one,
            polyvec_inputs,
            Some(&polyvec_plt_evaluator),
            Some(&PolyVecSlotTransferEvaluator::new()),
            Some(1),
        );
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_count = reference_outputs.len(),
            "PolyVec reference outputs"
        );
        log_relation_step(test_start, &mut step_start, "evaluated PolyVec reference outputs");

        let output_chunk_count = output_chunk_limit;
        let log_base_q = params.modulus_digits();
        let (q_moduli, _crt_bits, crt_depth) = params.to_crt();
        let next_seed_output_len = output_chunk_count * crt_depth;
        let (reference_next_seed, reference_refresh) =
            reference_outputs.split_at(next_seed_output_len);
        let gadget_row = GpuDCRTPolyMatrix::gadget_matrix(&params, secret_size).get_row(0);
        info!(
            target: "noise_refresh::fhe_prf_relation",
            output_chunk_count,
            log_base_q,
            crt_depth,
            num_slots,
            "checking public-key/poly-encoding relation"
        );

        for output_idx in 0..output_chunk_count {
            for crt_idx in 0..crt_depth {
                let flat_idx = output_idx * crt_depth + crt_idx;
                let pubkey_output = &pubkey_outputs[flat_idx];
                let encoding_output = &poly_encoding_outputs[flat_idx];
                assert_eq!(encoding_output.pubkey, *pubkey_output);
                for slot_idx in 0..num_slots {
                    let slot_secret = GpuDCRTPolyMatrix::from_compact_bytes(
                        &params,
                        slot_secret_mats[slot_idx].as_ref(),
                    );
                    let lhs = slot_secret.clone() * pubkey_output.matrix.clone() -
                        encoding_output.vector(slot_idx);
                    let next_seed_plain =
                        reference_next_seed[flat_idx].as_slice()[slot_idx].clone();
                    let expected_cols = (0..log_base_q)
                        .map(|digit_idx| {
                            let refresh_idx = output_idx * log_base_q * crt_depth +
                                crt_idx * log_base_q +
                                digit_idx;
                            let refresh_plain =
                                reference_refresh[refresh_idx].as_slice()[slot_idx].clone();
                            let column_plain =
                                (next_seed_plain.clone() * &gadget_row[digit_idx]) + refresh_plain;
                            slot_secret.entry(0, 0) * column_plain
                        })
                        .collect::<Vec<_>>();
                    let expected = GpuDCRTPolyMatrix::from_poly_vec_row(&params, expected_cols);
                    assert_eq!(
                        lhs, expected,
                        "output {output_idx}, crt {crt_idx}, slot {slot_idx}"
                    );

                    let q_i = q_moduli[crt_idx];
                    let rounded_lhs_cols = lhs
                        .get_row(0)
                        .into_iter()
                        .map(|poly| round_qi_scaled_poly(&params, &poly, q_i))
                        .collect::<Vec<_>>();
                    let rounded_expected_cols = (0..log_base_q)
                        .map(|digit_idx| {
                            let refresh_idx = output_idx * log_base_q * crt_depth +
                                crt_idx * log_base_q +
                                digit_idx;
                            let next_seed_plain = round_qi_scaled_poly(
                                &params,
                                &reference_next_seed[flat_idx].as_slice()[slot_idx],
                                q_i,
                            );
                            let error_plain = round_qi_scaled_poly(
                                &params,
                                &reference_refresh[refresh_idx].as_slice()[slot_idx],
                                q_i,
                            );
                            let column_plain =
                                (next_seed_plain * &gadget_row[digit_idx]) + error_plain;
                            slot_secret.entry(0, 0) * column_plain
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        GpuDCRTPolyMatrix::from_poly_vec_row(&params, rounded_lhs_cols),
                        GpuDCRTPolyMatrix::from_poly_vec_row(&params, rounded_expected_cols),
                        "rounded output {output_idx}, crt {crt_idx}, slot {slot_idx}"
                    );
                }
            }
        }
        log_relation_step(test_start, &mut step_start, "checked public-key/poly-encoding relation");
    }
}
