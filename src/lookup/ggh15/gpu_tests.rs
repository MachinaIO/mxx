use crate::{
    __PAIR, __TestState,
    bgg::sampler::{BGGEncodingSampler, BGGPolyEncodingSampler, BGGPublicKeySampler},
    circuit::PolyCircuit,
    lookup::PublicLut,
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
        },
    },
    sampler::{
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
};
use keccak_asm::Keccak256;
use std::{fs, path::Path};

use super::{
    GGH15BGGEncodingPltEvaluator, GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator,
};

const SIGMA: f64 = 4.578;

fn setup_lsb_bit_lut_gpu(t_n: usize, params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
    PublicLut::<GpuDCRTPoly>::new(
        params,
        t_n as u64,
        move |params, k| {
            if k >= t_n as u64 {
                return None;
            }
            let y_elem = <<GpuDCRTPoly as Poly>::Elem as crate::element::PolyElem>::constant(
                &params.modulus(),
                k % 2,
            );
            Some((k, y_elem))
        },
        None,
    )
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_ggh15_plt_eval_multi_inputs() {
    let _storage_lock = storage_test_lock().await;
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();

    let cpu_params = DCRTPolyParams::default();
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params
        .gpu_ids()
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_plt_eval_multi_inputs");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
        detected_gpu_params.batch(),
    );

    let plt = setup_lsb_bit_lut_gpu(16, &params);

    let mut circuit = PolyCircuit::new();
    let input_size = 5;
    let inputs = circuit.input(input_size);
    let plt_id = circuit.register_public_lookup(plt.clone());
    let outputs =
        inputs.iter().map(|&input| circuit.public_lookup_gate(input, plt_id)).collect::<Vec<_>>();
    circuit.output(outputs);

    let d = 2;
    let key: [u8; 32] = rand::random();
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, d);

    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
    let rand_ints =
        (0..input_size).map(|_| (rand::random::<u64>() % 16) as usize).collect::<Vec<_>>();
    let plaintexts = rand_ints
        .iter()
        .map(|&rand_int| GpuDCRTPoly::from_usize_to_constant(&params, rand_int))
        .collect::<Vec<_>>();

    let reveal_plaintexts = vec![true; input_size];
    let bgg_encoding_sampler =
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
    let enc_one = encodings[0].clone();
    let input_pubkeys = pubkeys[1..].to_vec();
    let input_encodings = encodings[1..].to_vec();

    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets);

    let dir_path = "test_data/test_gpu_ggh15_plt_eval_multi_inputs";
    let dir = Path::new(&dir_path);
    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    } else {
        fs::remove_dir_all(dir).unwrap();
        fs::create_dir(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    let error_sigma = 1e-9;
    let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
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
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(key, dir_path.into(), checkpoint_prefix, &params, c_b0);

    let one_encoding = enc_one.clone();
    let result_encoding =
        circuit.eval(&params, one_encoding, input_encodings.clone(), Some(&plt_encoding_evaluator));
    assert_eq!(result_encoding.len(), input_size);

    for i in 0..input_size {
        let result_encoding_i = &result_encoding[i];
        assert_eq!(result_encoding_i.pubkey, result_pubkey[i].clone());

        let expected_input = u64::try_from(plaintexts[i].to_const_int())
            .expect("test plaintext constant term must fit in u64");
        let expected_plaintext_elem = plt.get(&params, expected_input).unwrap().1;
        let expected_plaintext =
            GpuDCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
        assert_eq!(result_encoding_i.plaintext.clone().unwrap(), expected_plaintext.clone());

        let expected_vector = s_vec.clone() *
            (result_encoding_i.pubkey.matrix.clone() -
                (GpuDCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
        assert_eq!(result_encoding_i.vector, expected_vector);
    }
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_ggh15_poly_encoding_plt_eval_slot_secret_relation() {
    let _storage_lock = storage_test_lock().await;
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();

    let cpu_params = DCRTPolyParams::default();
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params.gpu_ids().first().expect(
        "at least one GPU device is required for test_gpu_ggh15_poly_encoding_plt_eval_slot_secret_relation",
    );
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
        detected_gpu_params.batch(),
    );

    let plt = setup_lsb_bit_lut_gpu(16, &params);

    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(1);
    let plt_id = circuit.register_public_lookup(plt.clone());
    let output = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![output]);

    let d = 2;
    let num_slots = 3;
    let key: [u8; 32] = rand::random();
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, d);

    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
    let plaintexts = (0..num_slots)
        .map(|_| {
            GpuDCRTPoly::from_usize_to_constant(&params, (rand::random::<u64>() % 16) as usize)
        })
        .collect::<Vec<_>>();

    let reveal_plaintexts = vec![true];
    let bgg_poly_encoding_sampler =
        BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let slot_secret_mats = bgg_poly_encoding_sampler.sample_slot_secret_mats(&params, num_slots);
    let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
    let poly_encodings = bgg_poly_encoding_sampler.sample_with_slot_secret_mats(
        &params,
        &pubkeys,
        &[plaintexts],
        &slot_secret_mats,
    );
    let enc_one_poly = poly_encodings[0].clone();
    let enc_input_poly = poly_encodings[1].clone();

    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets);

    let dir_path = "test_data/test_gpu_ggh15_poly_encoding_plt_eval";
    let dir = Path::new(&dir_path);
    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    } else {
        fs::remove_dir_all(dir).unwrap();
        fs::create_dir(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    let error_sigma = 1e-9;
    let plt_pubkey_evaluator = GGH15BGGPubKeyPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
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
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
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
    assert_eq!(result_poly.vectors.len(), num_slots);
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, d);

    for slot in 0..num_slots {
        let transformed_secret_vec = s_vec.clone() * &slot_secret_mats[slot];
        let expected_vector = transformed_secret_vec.clone() * result_poly.pubkey.matrix.clone() -
            (transformed_secret_vec * (gadget.clone() * result_plaintexts[slot].clone()));
        assert_eq!(result_poly.vectors[slot], expected_vector);
    }

    gpu_device_sync();
}
