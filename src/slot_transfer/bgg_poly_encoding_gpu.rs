use super::*;
use crate::poly::{PolyParams, dcrt::gpu::detected_gpu_device_ids};
use rayon::prelude::*;

pub(super) struct GpuSlotTransferSharedByDevice<M: PolyMatrix> {
    pub device_id: i32,
    pub params: <<M as PolyMatrix>::P as Poly>::Params,
    pub c_b0: M,
    pub out_pubkey_matrix: M,
}

pub(super) fn effective_gpu_slot_parallelism(slot_parallelism: usize) -> usize {
    let device_ids = detected_gpu_device_ids();
    assert!(
        !device_ids.is_empty(),
        "at least one GPU device is required for BggPolyEncoding slot transfer"
    );
    slot_parallelism.min(device_ids.len()).max(1)
}

fn slot_device_ids(slot_parallelism: usize) -> Vec<i32> {
    let clamped_parallelism = effective_gpu_slot_parallelism(slot_parallelism);
    detected_gpu_device_ids().into_iter().take(clamped_parallelism).collect()
}

fn prepare_slot_transfer_shared_by_device<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    gate_id: GateId,
    secret_size: usize,
    slot_parallelism: usize,
) -> Vec<GpuSlotTransferSharedByDevice<M>>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    slot_device_ids(slot_parallelism)
        .into_iter()
        .map(|device_id| {
            let local_params = params.params_for_device(device_id);
            let out_pubkey =
                evaluator.output_pubkey_for_params(&local_params, secret_size, gate_id);
            let c_b0 = M::from_compact_bytes(&local_params, &evaluator.c_b0_bytes);
            GpuSlotTransferSharedByDevice {
                device_id,
                params: local_params,
                c_b0,
                out_pubkey_matrix: out_pubkey.matrix,
            }
        })
        .collect()
}

pub(super) fn evaluate_slot_transfer_slots_gpu<M, HS>(
    evaluator: &BggPolyEncodingSTEvaluator<M, HS>,
    params: &<M::P as Poly>::Params,
    input: &BggPolyEncoding<M>,
    plaintext_bytes_by_slot: &[Arc<[u8]>],
    src_slots: &[(u32, Option<u32>)],
    gate_id: GateId,
    out_pubkey: &BggPublicKey<M>,
    configured_parallelism: usize,
) -> (Vec<Arc<[u8]>>, Vec<Arc<[u8]>>)
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: Send + Sync,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    let slot_count = input.num_slots();
    let shared_by_device = prepare_slot_transfer_shared_by_device::<M, HS>(
        evaluator,
        params,
        gate_id,
        out_pubkey.matrix.row_size(),
        configured_parallelism,
    );
    let chunk_width = shared_by_device.len();
    let mut output_vector_bytes = Vec::with_capacity(slot_count);
    let mut output_plaintext_bytes = Vec::with_capacity(slot_count);

    for slot_start in (0..slot_count).step_by(chunk_width.max(1)) {
        let chunk_len = (slot_start + chunk_width).min(slot_count) - slot_start;
        let chunk_outputs = (0..chunk_len)
            .into_par_iter()
            .map(|offset| {
                let device_shared = &shared_by_device[offset];
                let _device_id = device_shared.device_id;
                evaluator.output_slot_for_params(
                    &device_shared.params,
                    &device_shared.c_b0,
                    input,
                    plaintext_bytes_by_slot,
                    src_slots,
                    &device_shared.out_pubkey_matrix,
                    gate_id,
                    slot_start + offset,
                )
            })
            .collect::<Vec<_>>();
        let (chunk_vector_bytes, chunk_plaintext_bytes): (Vec<_>, Vec<_>) =
            chunk_outputs.into_iter().unzip();
        output_vector_bytes.extend(chunk_vector_bytes);
        output_plaintext_bytes.extend(chunk_plaintext_bytes);
    }

    (output_vector_bytes, output_plaintext_bytes)
}

#[cfg(test)]
mod tests {
    use super::slot_device_ids;
    use crate::{
        __PAIR, __TestState,
        bgg::{
            poly_encoding::BggPolyEncoding,
            public_key::BggPublicKey,
            sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
        },
        circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
        },
        sampler::{
            DistType, PolyHashSampler, PolyTrapdoorSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    struct EnvVarGuard {
        key: &'static str,
        old_value: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old_value = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, old_value }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.old_value {
                unsafe { std::env::set_var(self.key, value) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    struct DummyGpuPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<GpuDCRTPolyMatrix>> for DummyGpuPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPublicKey<GpuDCRTPolyMatrix>,
            _input: &BggPublicKey<GpuDCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPublicKey<GpuDCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer GPU tests")
        }
    }

    struct DummyGpuPolyEncodingPltEvaluator;

    impl PltEvaluator<BggPolyEncoding<GpuDCRTPolyMatrix>> for DummyGpuPolyEncodingPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPolyEncoding<GpuDCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPolyEncoding<GpuDCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPolyEncoding<GpuDCRTPolyMatrix>,
            _input: &BggPolyEncoding<GpuDCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPolyEncoding<GpuDCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer GPU tests")
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_slot_transfer_bgg_poly_encoding_gpu_uses_detected_gpu_ids() {
        let detected_gpu_ids = detected_gpu_device_ids();
        if detected_gpu_ids.is_empty() {
            let panic = std::panic::catch_unwind(|| slot_device_ids(1))
                .expect_err("without detected GPUs the helper should reject slot processing");
            let panic_msg = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .expect("panic payload should be a string");
            assert!(panic_msg.contains("at least one GPU device is required"));
            return;
        }

        let slot_parallelism = detected_gpu_ids.len().min(2);
        assert_eq!(
            slot_device_ids(slot_parallelism),
            detected_gpu_ids[..slot_parallelism].to_vec()
        );
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_slot_transfer_bgg_poly_encoding_gpu_runs_on_gpu_repeatedly() {
        let detected_gpu_ids = detected_gpu_device_ids();
        assert!(
            !detected_gpu_ids.is_empty(),
            "at least one GPU device is required for BggPolyEncoding slot-transfer GPU tests"
        );

        let _storage_lock = storage_test_lock().await;
        let _parallelism_guard = EnvVarGuard::set(
            "SLOT_TRANSFER_SLOT_PARALLELISM",
            &(detected_gpu_ids.len() + 2).to_string(),
        );

        for iter in 0..5usize {
            gpu_device_sync();
            let params = GpuDCRTPolyParams::new(4, vec![131041, 131009], 1);
            let hash_key = [0x41u8; 32];
            let secret_size = 2usize;
            let num_slots = 3usize;
            let dir_path = format!("test_data/test_slot_transfer_bgg_poly_encoding_gpu_{iter}");
            let dir = Path::new(&dir_path);
            if dir.exists() {
                fs::remove_dir_all(dir).unwrap();
            }
            fs::create_dir_all(dir).unwrap();
            init_storage_system(dir.to_path_buf());

            let tag: u64 = (iter as u64) + 11;
            let bgg_pubkey_sampler =
                BGGPublicKeySampler::<_, GpuDCRTPolyHashSampler<Keccak256>>::new(
                    hash_key,
                    secret_size,
                );
            let public_keys = bgg_pubkey_sampler.sample(&params, &tag.to_le_bytes(), &[true]);

            let secrets = (0..secret_size)
                .map(|idx| GpuDCRTPoly::from_usize_to_constant(&params, idx + 2))
                .collect::<Vec<_>>();
            let plaintext_rows = vec![vec![
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 5).to_compact_bytes(),
                ),
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 9).to_compact_bytes(),
                ),
                Arc::<[u8]>::from(
                    GpuDCRTPoly::from_usize_to_constant(&params, 4).to_compact_bytes(),
                ),
            ]];
            let encoding_sampler =
                BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(&params, &secrets, None);
            let slot_secret_mats = encoding_sampler.sample_slot_secret_mats(&params, num_slots);
            let encodings = encoding_sampler.sample(
                &params,
                &public_keys,
                &plaintext_rows,
                Some(&slot_secret_mats),
            );
            let one = encodings[0].clone();
            let input = encodings[1].clone();
            let pubkey_evaluator =
                BggPublicKeySTEvaluator::<
                    GpuDCRTPolyMatrix,
                    GpuDCRTPolyUniformSampler,
                    GpuDCRTPolyHashSampler<Keccak256>,
                    GpuDCRTPolyTrapdoorSampler,
                >::new(
                    hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf()
                );

            let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];
            let mut circuit = PolyCircuit::new();
            let inputs = circuit.input(1);
            let transferred = circuit.slot_transfer_gate(inputs[0], &src_slots);
            circuit.output(vec![transferred]);

            let result_pubkey = circuit.eval(
                &params,
                one.pubkey.clone(),
                vec![input.pubkey.clone()],
                None::<&DummyGpuPubKeyPltEvaluator>,
                Some(&pubkey_evaluator),
                None,
            );
            assert_eq!(result_pubkey.len(), 1);

            pubkey_evaluator.sample_aux_matrices(&params, slot_secret_mats.clone());
            wait_for_all_writes(dir.to_path_buf()).await.unwrap();

            let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
            let b0_matrix = pubkey_evaluator
                .load_b0_matrix_checkpoint(&params)
                .expect("b0 matrix checkpoint should exist after sample_aux_matrices");
            let c_b0 = s_vec.clone() * &b0_matrix;
            let evaluator = BggPolyEncodingSTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyHashSampler<Keccak256>,
            >::new(
                hash_key,
                dir.to_path_buf(),
                pubkey_evaluator.checkpoint_prefix(&params),
                c_b0.to_compact_bytes(),
            );

            let result = circuit.eval(
                &params,
                one,
                vec![input.clone()],
                None::<&DummyGpuPolyEncodingPltEvaluator>,
                Some(&evaluator),
                Some(1),
            );
            assert_eq!(result.len(), 1);
            let output = &result[0];
            assert_eq!(output.pubkey, result_pubkey[0]);

            let a_out = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_gate_a_out_{}", transferred),
                secret_size,
                secret_size * params.modulus_digits(),
                DistType::FinRingDist,
            );
            let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, secret_size);

            for (dst_slot, (src_slot_u32, scalar)) in src_slots.into_iter().enumerate() {
                let src_slot = src_slot_u32 as usize;
                let s_dst = GpuDCRTPolyMatrix::from_compact_bytes(
                    &params,
                    slot_secret_mats[dst_slot].as_ref(),
                );
                let source_plaintext = input
                    .plaintext(src_slot)
                    .expect("input encoding should reveal plaintext constants");
                let constant_term = source_plaintext
                    .coeffs_biguints()
                    .into_iter()
                    .next()
                    .expect("plaintext must have a constant coefficient");
                let mut scaled_plaintext =
                    GpuDCRTPoly::from_biguint_to_constant(&params, constant_term);
                if let Some(scalar) = scalar {
                    let scalar_poly = GpuDCRTPoly::from_usize_to_constant(&params, scalar as usize);
                    scaled_plaintext = scaled_plaintext * scalar_poly;
                }
                let expected_vector = (s_vec.clone() * &s_dst) *
                    &(a_out.clone() - (gadget_matrix.clone() * &scaled_plaintext));
                assert_eq!(output.vector(dst_slot), expected_vector);

                let output_plaintext = output
                    .plaintext(dst_slot)
                    .expect("slot-transfer output should reveal plaintext constants");
                assert_eq!(output_plaintext, scaled_plaintext);
            }

            gpu_device_sync();
        }
    }
}
