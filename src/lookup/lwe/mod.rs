mod encoding;
mod pubkey;

pub use encoding::LWEBGGEncodingPltEvaluator;
pub use pubkey::LWEBGGPubKeyPltEvaluator;

use crate::{
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler},
};

pub(crate) fn derive_a_lt_matrix<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = row_size * params.modulus_digits();
    let hash_sampler = SH::new();
    let tag = format!("A_LT_{gate_id}");
    hash_sampler.sample_hash(
        params,
        hash_key,
        tag.into_bytes(),
        row_size,
        m_g,
        DistType::FinRingDist,
    )
}

pub(crate) fn k_high_checkpoint_prefix(gate_id: GateId, lut_id: usize) -> String {
    format!("LWE_K_H_{gate_id}_{lut_id}")
}

fn k_low_tag(gate_id: GateId, lut_id: usize, lut_entry_idx: usize) -> String {
    format!("LWE_R_G_{gate_id}_{lut_id}_{lut_entry_idx}")
}

pub(crate) fn derive_k_low<M, SH>(
    params: &<M::P as Poly>::Params,
    row_size: usize,
    hash_key: [u8; 32],
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
) -> M
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = row_size * params.modulus_digits();
    SH::new().sample_hash_decomposed(
        params,
        hash_key,
        k_low_tag(gate_id, lut_id, lut_entry_idx),
        row_size,
        m_g,
        DistType::FinRingDist,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator, derive_k_low,
        k_high_checkpoint_prefix,
    };
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
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
        storage::{
            read::read_matrix_from_multi_batch,
            write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        },
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
        let k_high = read_matrix_from_multi_batch::<DCRTPolyMatrix>(
            &params,
            dir,
            &k_high_checkpoint_prefix(output_gate, plt_id),
            lut_entry_idx,
        )
        .expect("k_high checkpoint must exist for the looked-up LUT row");

        let expected_vector = (c_b * &k_high) + &(enc1.vector.clone() * &k_low);
        assert_eq!(result_encoding.vector, expected_vector);
    }
}
