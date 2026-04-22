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
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
};
use keccak_asm::Keccak256;
use std::{fs, path::Path, sync::Arc};

fn setup_lsb_bit_lut(t_n: usize, params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
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

const SIGMA: f64 = 4.578;

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_lwe_plt_eval() {
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
        .expect("at least one GPU device is required for test_gpu_lwe_plt_eval");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );

    let plt = setup_lsb_bit_lut(16, &params);
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt.clone());
    let output_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![output_gate]);

    let d = 3;
    let key: [u8; 32] = rand::random();
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, d);

    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
    let rand_int = (rand::random::<u64>() % 16) as usize;
    let plaintexts = vec![GpuDCRTPoly::from_usize_to_constant(&params, rand_int)];

    let reveal_plaintexts = vec![true; 1];
    let bgg_encoding_sampler =
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
    let encodings = bgg_encoding_sampler.sample(&params, &pubkeys, &plaintexts);
    let enc_one = encodings[0].clone();
    let enc1 = encodings[1].clone();

    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
    let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets);
    let c_b = s_vec.clone() * &b;

    let dir_path = "test_data/test_gpu_lwe_plt_eval";
    let dir = Path::new(dir_path);
    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    } else {
        fs::remove_dir_all(dir).unwrap();
        fs::create_dir(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    let plt_pubkey_evaluator =
        LWEBGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(key, trapdoor_sampler, Arc::new(b), Arc::new(b_trapdoor), dir_path.into());
    let result_pubkey = circuit.eval(
        &params,
        enc_one.pubkey.clone(),
        vec![enc1.pubkey.clone()],
        Some(&plt_pubkey_evaluator),
        None,
        None,
    );
    plt_pubkey_evaluator.sample_aux_matrices(&params);
    wait_for_all_writes(dir.to_path_buf()).await.unwrap();
    assert_eq!(result_pubkey.len(), 1);

    let plt_encoding_evaluator = LWEBGGEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::new(key, dir_path.into(), c_b.clone());
    let result_encoding = circuit.eval(
        &params,
        enc_one.clone(),
        vec![enc1.clone()],
        Some(&plt_encoding_evaluator),
        None,
        None,
    );
    assert_eq!(result_encoding.len(), 1);

    let result_encoding = &result_encoding[0];
    assert_eq!(result_encoding.pubkey, result_pubkey[0].clone());
    let expected_input = plaintexts[0].const_coeff_u64();
    let (lut_entry_idx, expected_plaintext_elem) = plt.get(&params, expected_input).unwrap();
    let lut_entry_idx = usize::try_from(lut_entry_idx).expect("LUT row index must fit in usize");
    let expected_plaintext = GpuDCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
    assert_eq!(result_encoding.plaintext.clone().unwrap(), expected_plaintext);

    let output_gate = output_gate.as_single_wire();
    let k_low = derive_k_low::<GpuDCRTPolyMatrix, GpuDCRTPolyHashSampler<Keccak256>>(
        &params,
        d,
        key,
        output_gate,
        plt_id,
        lut_entry_idx,
    );
    let k_high =
        read_k_high_row::<GpuDCRTPolyMatrix>(&params, dir, output_gate, plt_id, d, lut_entry_idx);
    let expected_vector = (c_b * &k_high) + &(enc1.vector.clone() * &k_low);
    assert_eq!(result_encoding.vector, expected_vector);
    gpu_device_sync();
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_lwe_poly_encoding_plt_eval() {
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
        .expect("at least one GPU device is required for test_gpu_lwe_poly_encoding_plt_eval");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );

    let plt = setup_lsb_bit_lut(16, &params);
    let mut circuit = PolyCircuit::new();
    let inputs = circuit.input(1).to_vec();
    let plt_id = circuit.register_public_lookup(plt.clone());
    let output_gate = circuit.public_lookup_gate(inputs[0], plt_id);
    circuit.output(vec![output_gate]);

    let d = 3;
    let num_slots = detected_gpu_params.gpu_ids().len() + 2;
    let key: [u8; 32] = rand::random();
    let bgg_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(key, d);

    let tag: u64 = rand::random();
    let tag_bytes = tag.to_le_bytes();
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets = uniform_sampler.sample_uniform(&params, 1, d, DistType::TernaryDist).get_row(0);
    let plaintext_bytes = (0..num_slots)
        .map(|_| {
            Arc::<[u8]>::from(
                GpuDCRTPoly::from_usize_to_constant(&params, (rand::random::<u64>() % 16) as usize)
                    .to_compact_bytes(),
            )
        })
        .collect::<Vec<_>>();

    let reveal_plaintexts = vec![true];
    let bgg_poly_encoding_sampler =
        BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
    let pubkeys = bgg_pubkey_sampler.sample(&params, &tag_bytes, &reveal_plaintexts);
    let (poly_encodings, slot_secret_mats) = bgg_poly_encoding_sampler
        .sample_with_fresh_slot_secret_mats(&params, &pubkeys, &[plaintext_bytes]);
    let enc_one_poly = poly_encodings[0].clone();
    let enc_input_poly = poly_encodings[1].clone();

    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
    let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets);

    let dir_path = "test_data/test_gpu_lwe_poly_encoding_plt_eval";
    let dir = Path::new(dir_path);
    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    } else {
        fs::remove_dir_all(dir).unwrap();
        fs::create_dir(dir).unwrap();
    }
    init_storage_system(dir.to_path_buf());

    let plt_pubkey_evaluator =
        LWEBGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(
            key, trapdoor_sampler, Arc::new(b.clone()), Arc::new(b_trapdoor), dir_path.into()
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
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
    >::build_c_b_compact_bytes_by_slot::<GpuDCRTPolyUniformSampler>(
        &params,
        &s_vec,
        &b,
        &slot_secret_mats,
        None,
    );
    let poly_evaluator = LWEBGGPolyEncodingPltEvaluator::<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyHashSampler<Keccak256>,
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
            GpuDCRTPoly::from_elem_to_constant(&params, &expected_plaintext_elem);
        assert_eq!(
            result_poly
                .plaintext_for_params(&params, slot)
                .expect("poly lookup result should reveal plaintexts"),
            expected_plaintext
        );

        let c_b = GpuDCRTPolyMatrix::from_compact_bytes(
            &params,
            c_b_compact_bytes_by_slot[slot].as_ref(),
        );
        let k_low = derive_k_low::<GpuDCRTPolyMatrix, GpuDCRTPolyHashSampler<Keccak256>>(
            &params,
            d,
            key,
            output_gate,
            plt_id,
            lut_entry_idx,
        );
        let k_high = read_k_high_row::<GpuDCRTPolyMatrix>(
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
    assert_eq!(
        bench.max_parallelism,
        (2 * k_high_chunk_count::<GpuDCRTPolyMatrix>(&params, d)) as u128
    );

    gpu_device_sync();
}
