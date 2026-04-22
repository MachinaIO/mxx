mod encoding;
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests;
mod poly_encoding;
mod pubkey;
mod utils;

pub use encoding::LWEBGGEncodingPltEvaluator;
#[cfg(feature = "gpu")]
#[allow(unused_imports)]
pub(crate) use encoding::public_lookup_gpu_device_ids;
#[cfg(all(test, feature = "gpu"))]
#[allow(unused_imports)]
pub(crate) use encoding::public_lookup_round_robin_device_slot;
pub use poly_encoding::LWEBGGPolyEncodingPltEvaluator;
pub use pubkey::LWEBGGPubKeyPltEvaluator;
pub(crate) use utils::{
    column_chunk_bounds, column_chunk_count, column_chunk_id_prefix, derive_a_lt_matrix,
    derive_k_low, derive_k_low_chunk, k_high_checkpoint_prefix, k_high_chunk_count,
    k_high_row_checkpoint_prefix, read_k_high_chunk, read_k_high_row,
};

#[cfg(test)]
mod tests {
    use super::{
        LWEBGGEncodingPltEvaluator, LWEBGGPolyEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
        derive_k_low, k_high_chunk_count, read_k_high_row,
    };
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{BggPolyEncodingBenchSamples, PolyEncodingPublicLutBenchEstimator},
        bgg::sampler::{BGGEncodingSampler, BGGPolyEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        lookup::PublicLut,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            PolyTrapdoorSampler, hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    fn setup_lsb_bit_lut(t_n: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::<DCRTPoly>::new(
            params,
            t_n as u64,
            move |params, k| {
                if k >= t_n as u64 {
                    return None;
                }
                let y_elem = <<DCRTPoly as Poly>::Elem as crate::element::PolyElem>::constant(
                    &params.modulus(),
                    k % 2,
                );
                Some((k, y_elem))
            },
            None,
        )
    }

    const SIGMA: f64 = 4.578;

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_lwe_plt_eval() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::default();
        let plt = setup_lsb_bit_lut(16, &params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output_gate]);

        let d = 3;
        let input_size = 1;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let secrets = vec![create_bit_random_poly(&params); d];
        let rand_int = (rand::random::<u64>() % 16) as usize;
        let plaintexts = vec![DCRTPoly::from_usize_to_constant(&params, rand_int); input_size];

        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b = s_vec.clone() * &b;

        let dir_path = "test_data/test_lwe_plt_eval";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let plt_pubkey_evaluator =
            LWEBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>, _>::new(
                key,
                trapdoor_sampler,
                Arc::new(b),
                Arc::new(b_trapdoor),
                dir_path.into(),
            );
        let one_pubkey = enc_one.pubkey.clone();
        let input_pubkeys = vec![enc1.pubkey.clone()];
        let result_pubkey = circuit.eval(
            &params,
            one_pubkey,
            input_pubkeys,
            Some(&plt_pubkey_evaluator),
            None,
            None,
        );
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        let result_pubkey = &result_pubkey[0];

        let plt_encoding_evaluator = LWEBGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, dir_path.into(), c_b.clone());

        let one_encoding = enc_one.clone();
        let input_encodings = vec![enc1.clone()];
        let result_encoding = circuit.eval(
            &params,
            one_encoding,
            input_encodings,
            Some(&plt_encoding_evaluator),
            None,
            None,
        );
        assert_eq!(result_encoding.len(), 1);
        let result_encoding = &result_encoding[0];
        assert_eq!(result_encoding.pubkey, result_pubkey.clone());

        let expected_input = plaintexts[0].const_coeff_u64();
        let (lut_entry_idx, expected_plaintext_elem) = plt.get(&params, expected_input).unwrap();
        let lut_entry_idx =
            usize::try_from(lut_entry_idx).expect("LUT row index must fit in usize");
        let expected_plaintext = DCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
        assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext.clone());
        let output_gate = output_gate.as_single_wire();

        let k_low = derive_k_low::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>(
            &params,
            d,
            key,
            output_gate,
            plt_id,
            lut_entry_idx,
        );
        let k_high =
            read_k_high_row::<DCRTPolyMatrix>(&params, dir, output_gate, plt_id, d, lut_entry_idx);

        let expected_vector = (c_b * &k_high) + &(enc1.vector.clone() * &k_low);
        assert_eq!(result_encoding.vector, expected_vector);
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_lwe_poly_encoding_plt_eval() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_bit_lut(16, &params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1).to_vec();
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output_gate]);

        let d = 3;
        let num_slots = 3;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintext_bytes = (0..num_slots)
            .map(|_| {
                Arc::<[u8]>::from(
                    DCRTPoly::from_usize_to_constant(
                        &params,
                        (rand::random::<u64>() % 16) as usize,
                    )
                    .to_compact_bytes(),
                )
            })
            .collect::<Vec<_>>();

        let reveal_plaintexts = vec![true];
        let bgg_poly_encoding_sampler =
            BGGPolyEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let (poly_encodings, slot_secret_mats) = bgg_poly_encoding_sampler
            .sample_with_fresh_slot_secret_mats(&params, &pubkeys, &[plaintext_bytes]);
        let enc_one_poly = poly_encodings[0].clone();
        let enc_input_poly = poly_encodings[1].clone();

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);

        let dir_path = "test_data/test_lwe_poly_encoding_plt_eval";
        let dir = Path::new(dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let plt_pubkey_evaluator =
            LWEBGGPubKeyPltEvaluator::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>, _>::new(
                key,
                trapdoor_sampler,
                Arc::new(b.clone()),
                Arc::new(b_trapdoor),
                dir_path.into(),
            );
        let result_pubkey = circuit.eval(
            &params,
            enc_one_poly.pubkey.clone(),
            vec![enc_input_poly.pubkey.clone()],
            Some(&plt_pubkey_evaluator),
            None,
            None,
        );
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);

        let c_b_compact_bytes_by_slot = LWEBGGPolyEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::build_c_b_compact_bytes_by_slot::<DCRTPolyUniformSampler>(
            &params,
            &s_vec,
            &b,
            &slot_secret_mats,
            None,
        );
        let poly_evaluator = LWEBGGPolyEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(key, dir_path.into(), c_b_compact_bytes_by_slot.clone());

        let result_poly = circuit.eval(
            &params,
            enc_one_poly.clone(),
            vec![enc_input_poly.clone()],
            Some(&poly_evaluator),
            None,
            Some(1),
        );
        assert_eq!(result_poly.len(), 1);
        let result_poly = &result_poly[0];
        assert_eq!(result_poly.pubkey, result_pubkey[0]);
        assert_eq!(result_poly.num_slots(), num_slots);

        let output_gate = output_gate.as_single_wire();
        for slot in 0..num_slots {
            let input_plaintext = enc_input_poly
                .plaintext_for_params(&params, slot)
                .expect("input poly encoding should reveal plaintexts");
            let (lut_entry_idx, expected_plaintext_elem) =
                plt.get(&params, input_plaintext.const_coeff_u64()).unwrap();
            let lut_entry_idx =
                usize::try_from(lut_entry_idx).expect("LUT row index must fit in usize");
            let expected_plaintext =
                DCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
            assert_eq!(
                result_poly
                    .plaintext_for_params(&params, slot)
                    .expect("poly lookup result should reveal plaintexts"),
                expected_plaintext
            );

            let c_b = DCRTPolyMatrix::from_compact_bytes(
                &params,
                c_b_compact_bytes_by_slot[slot].as_ref(),
            );
            let k_low = derive_k_low::<DCRTPolyMatrix, DCRTPolyHashSampler<Keccak256>>(
                &params,
                d,
                key,
                output_gate,
                plt_id,
                lut_entry_idx,
            );
            let k_high = read_k_high_row::<DCRTPolyMatrix>(
                &params,
                dir,
                output_gate,
                plt_id,
                d,
                lut_entry_idx,
            );
            let expected_vector =
                (c_b * &k_high) + &(enc_input_poly.vector_for_params(&params, slot) * &k_low);
            assert_eq!(result_poly.vector_for_params(&params, slot), expected_vector);
        }

        let single_slot_one = crate::bgg::poly_encoding::BggPolyEncoding::new(
            params.clone(),
            vec![enc_one_poly.vector_bytes[0].clone()],
            enc_one_poly.pubkey.clone(),
            Some(vec![
                enc_one_poly
                    .plaintext_bytes
                    .as_ref()
                    .expect("poly encoding benchmark sample should reveal plaintexts")[0]
                    .clone(),
            ]),
        );
        let single_slot_input = crate::bgg::poly_encoding::BggPolyEncoding::new(
            params.clone(),
            vec![enc_input_poly.vector_bytes[0].clone()],
            enc_input_poly.pubkey.clone(),
            Some(vec![
                enc_input_poly
                    .plaintext_bytes
                    .as_ref()
                    .expect("poly encoding benchmark sample should reveal plaintexts")[0]
                    .clone(),
            ]),
        );
        let bench_samples = BggPolyEncodingBenchSamples {
            num_slots: 1,
            params: &params,
            add_lhs: &single_slot_input,
            add_rhs: &single_slot_input,
            sub_lhs: &single_slot_input,
            sub_rhs: &single_slot_input,
            mul_lhs: &single_slot_input,
            mul_rhs: &single_slot_input,
            small_scalar_input: &single_slot_input,
            small_scalar: &[1u32],
            large_scalar_input: &single_slot_input,
            large_scalar: &[num_bigint::BigUint::from(1u32)],
            public_lut_one: &single_slot_one,
            public_lut_input: &single_slot_input,
            public_lut: &plt,
            public_lut_gate_id: output_gate,
            public_lut_id: plt_id,
            slot_transfer_input: &single_slot_input,
            slot_transfer_src_slots: &[(0, None)],
            slot_transfer_gate_id: output_gate,
        };
        let bench = poly_evaluator.benchmark_public_lookup_chunk(&bench_samples, 1);
        assert!(bench.latency >= 0.0);
        assert!(bench.total_time >= bench.latency);
        #[cfg(feature = "gpu")]
        let expected_parallelism = (2 * k_high_chunk_count::<DCRTPolyMatrix>(&params, d)) as u128;
        #[cfg(not(feature = "gpu"))]
        let expected_parallelism = k_high_chunk_count::<DCRTPolyMatrix>(&params, d) as u128;
        assert_eq!(bench.max_parallelism, expected_parallelism);
    }
}
