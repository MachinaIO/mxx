mod encoding;
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests;
mod poly_encoding;
mod pubkey;

pub use encoding::GGH15BGGEncodingPltEvaluator;
pub use poly_encoding::GGH15BGGPolyEncodingPltEvaluator;
pub use pubkey::GGH15BGGPubKeyPltEvaluator;

#[cfg(test)]
mod tests {
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPolyEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        lookup::PublicLut,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            DistType, PolyUniformSampler, hash::DCRTPolyHashSampler,
            trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    use super::{
        GGH15BGGEncodingPltEvaluator, GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator,
    };

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

    #[test]
    fn test_ggh15_checkpoint_prefix_keeps_false_mode_suffix() {
        let params = DCRTPolyParams::default();
        let key = [0x5au8; 32];
        let evaluator =
            GGH15BGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(key, 2, SIGMA, 0.0, "test_data/test_ggh15_checkpoint_prefix".into());

        let checkpoint_prefix = evaluator.checkpoint_prefix(&params);

        assert!(
            checkpoint_prefix.contains("_ins0_key"),
            "checkpoint prefix should keep the legacy false-mode marker: {checkpoint_prefix}"
        );
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_ggh15_plt_eval_single_input() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_bit_lut(16, &params);

        // Create a simple circuit with the lookup table
        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output]);

        let d = 2;
        let input_size = 1;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let secrets =
            uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
        let rand_int = (rand::random::<u64>() % 16) as usize;
        let plaintexts = vec![DCRTPoly::from_usize_to_constant(&params, rand_int); input_size];

        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let enc1 = encodings[1].clone();

        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);

        // Storage directory
        let dir_path = "test_data/test_ggh15_plt_eval_single_input";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let error_sigma = 0.0;
        let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(key, d, SIGMA, error_sigma, dir_path.into());

        let one_pubkey = enc_one.pubkey.clone();
        let input_pubkeys = vec![enc1.pubkey.clone()];
        let result_pubkey =
            circuit.eval(&params, one_pubkey, input_pubkeys, Some(&plt_pubkey_evaluator));
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);
        let result_pubkey = &result_pubkey[0];
        let b0_matrix = plt_pubkey_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        let c_b0 = s_vec.clone() * &b0_matrix;
        let checkpoint_prefix = plt_pubkey_evaluator.checkpoint_prefix(&params);

        let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(
            key, dir_path.into(), checkpoint_prefix, &params, c_b0
        );

        let one_encoding = enc_one.clone();
        let input_encodings = vec![enc1.clone()];
        let result_encoding =
            circuit.eval(&params, one_encoding, input_encodings, Some(&plt_encoding_evaluator));
        assert_eq!(result_encoding.len(), 1);
        let result_encoding = &result_encoding[0];
        assert_eq!(result_encoding.pubkey, result_pubkey.clone());

        let expected_input = u64::try_from(plaintexts[0].to_const_int())
            .expect("test plaintext constant term must fit in u64");
        let expected_plaintext_elem = plt.get(&params, expected_input).unwrap().1;
        let expected_plaintext = DCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
        assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext.clone());

        let expected_vector = s_vec.clone() *
            (result_encoding.pubkey.matrix.clone() -
                (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
        assert_eq!(result_encoding.vector, expected_vector);
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_ggh15_plt_eval_multi_inputs() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_bit_lut(16, &params);

        // Create a simple circuit with the lookup table
        let mut circuit = PolyCircuit::new();
        let input_size = 5;
        let inputs = circuit.input(input_size);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let outputs = inputs
            .iter()
            .map(|&input| circuit.public_lookup_gate(input, plt_id))
            .collect::<Vec<_>>();
        circuit.output(outputs);

        let d = 2;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let secrets =
            uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
        let rand_ints =
            (0..input_size).map(|_| (rand::random::<u64>() % 16) as usize).collect::<Vec<_>>();
        let plaintexts = rand_ints
            .iter()
            .map(|&rand_int| DCRTPoly::from_usize_to_constant(&params, rand_int))
            .collect::<Vec<_>>();

        let reveal_plaintexts = vec![true; input_size];
        let bgg_encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
        let enc_one = encodings[0].clone();
        let input_pubkeys = pubkeys[1..].to_vec();
        let input_encodings = encodings[1..].to_vec();

        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);

        // Storage directory
        let dir_path = "test_data/test_ggh15_plt_eval_multi_inputs";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let error_sigma = 0.0;
        let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(key, d, SIGMA, error_sigma, dir_path.into());

        let one_pubkey = enc_one.pubkey.clone();
        let result_pubkey =
            circuit.eval(&params, one_pubkey, input_pubkeys.clone(), Some(&plt_pubkey_evaluator));
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), input_size);
        let b0_matrix = plt_pubkey_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        let c_b0 = s_vec.clone() * &b0_matrix;
        let checkpoint_prefix = plt_pubkey_evaluator.checkpoint_prefix(&params);

        let plt_encoding_evaluator = GGH15BGGEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(
            key, dir_path.into(), checkpoint_prefix, &params, c_b0
        );

        let one_encoding = enc_one.clone();
        let result_encoding = circuit.eval(
            &params,
            one_encoding,
            input_encodings.clone(),
            Some(&plt_encoding_evaluator),
        );
        assert_eq!(result_encoding.len(), input_size);

        for i in 0..input_size {
            let result_encoding_i = &result_encoding[i];
            assert_eq!(result_encoding_i.pubkey, result_pubkey[i].clone());

            let expected_input = u64::try_from(plaintexts[i].to_const_int())
                .expect("test plaintext constant term must fit in u64");
            let expected_plaintext_elem = plt.get(&params, expected_input).unwrap().1;
            let expected_plaintext =
                DCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
            assert_eq!(result_encoding_i.plaintext.clone().unwrap(), expected_plaintext.clone());

            let expected_vector = s_vec.clone() *
                (result_encoding_i.pubkey.matrix.clone() -
                    (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
            assert_eq!(result_encoding_i.vector, expected_vector);
        }
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_ggh15_poly_encoding_plt_eval_matches_single_slot_evaluator() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_bit_lut(16, &params);

        let mut circuit = PolyCircuit::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt.clone());
        let output = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![output]);

        let d = 2;
        let num_slots = 3;
        let key: [u8; 32] = rand::random();
        let bgg_pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);

        let tag: u64 = rand::random();
        let tag_bytes = tag.to_le_bytes();

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let secrets =
            uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
        let plaintexts = (0..num_slots)
            .map(|_| {
                DCRTPoly::from_usize_to_constant(&params, (rand::random::<u64>() % 16) as usize)
            })
            .collect::<Vec<_>>();

        let reveal_plaintexts = vec![true];
        let bgg_poly_encoding_sampler =
            BGGPolyEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let slot_secret_mats =
            bgg_poly_encoding_sampler.sample_slot_secret_mats(&params, num_slots);
        let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
        let poly_encodings = bgg_poly_encoding_sampler.sample_with_slot_secret_mats(
            &params,
            &pubkeys,
            &[plaintexts],
            &slot_secret_mats,
        );
        let enc_one_poly = poly_encodings[0].clone();
        let enc_input_poly = poly_encodings[1].clone();

        let s_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);

        let dir_path = "test_data/test_ggh15_poly_encoding_plt_eval";
        let dir = Path::new(&dir_path);
        if !dir.exists() {
            fs::create_dir(dir).unwrap();
        } else {
            fs::remove_dir_all(dir).unwrap();
            fs::create_dir(dir).unwrap();
        }
        init_storage_system(dir.to_path_buf());

        let error_sigma = 0.0;
        let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(key, d, SIGMA, error_sigma, dir_path.into());

        let result_pubkey = circuit.eval(
            &params,
            enc_one_poly.pubkey.clone(),
            vec![enc_input_poly.pubkey.clone()],
            Some(&plt_pubkey_evaluator),
        );
        plt_pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(dir.to_path_buf()).await.unwrap();
        assert_eq!(result_pubkey.len(), 1);

        let b0_matrix = plt_pubkey_evaluator
            .load_b0_matrix_checkpoint(&params)
            .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
        let checkpoint_prefix = plt_pubkey_evaluator.checkpoint_prefix(&params);
        let poly_evaluator = GGH15BGGPolyEncodingPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyHashSampler<Keccak256>,
        >::new(
            key,
            dir_path.into(),
            checkpoint_prefix,
            &params,
            s_vec.clone(),
            b0_matrix,
            slot_secret_mats.clone(),
        );

        let result_poly = circuit.eval(
            &params,
            enc_one_poly.clone(),
            vec![enc_input_poly.clone()],
            Some(&poly_evaluator),
        );
        assert_eq!(result_poly.len(), 1);
        let result_poly = &result_poly[0];
        assert_eq!(result_poly.pubkey, result_pubkey[0]);

        let result_plaintexts =
            result_poly.plaintexts.as_ref().expect("poly lookup result should reveal plaintexts");
        assert_eq!(result_plaintexts.len(), num_slots);
        assert_eq!(result_poly.num_slots(), num_slots);
        let gadget = DCRTPolyMatrix::gadget_matrix(&params, d);

        for slot in 0..num_slots {
            let transformed_secret_vec = s_vec.clone() * &slot_secret_mats[slot];
            let expected_vector = transformed_secret_vec.clone() *
                result_poly.pubkey.matrix.clone() -
                (transformed_secret_vec * (gadget.clone() * result_plaintexts[slot].clone()));
            assert_eq!(result_poly.vector(slot), expected_vector);
        }
    }
}
