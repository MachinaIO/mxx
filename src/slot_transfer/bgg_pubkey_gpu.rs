use crate::{
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        trapdoor::GpuPreimageRequest,
    },
    slot_transfer::bgg_pubkey::{
        BggPublicKeySTEvaluator, BggPublicKeySTGateState, GateAuxSample, SlotAuxSample,
    },
};
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) struct GpuSlotTransferDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: T,
    b0_matrix: M,
    b1_trapdoor: T,
    b1_matrix: M,
    identity: M,
    gadget_matrix: M,
}

struct GpuSlotAuxTarget<M>
where
    M: PolyMatrix,
{
    slot_idx: usize,
    device_idx: usize,
    slot_a: M,
    slot_a_bytes: Vec<u8>,
    target_b0: M,
    target_b1: M,
}

impl<M, US, HS, TS> BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub(crate) fn prepare_gpu_device_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        parallelism: usize,
    ) -> Vec<GpuSlotTransferDeviceShared<M, TS::Trapdoor>> {
        let device_ids = detected_gpu_device_ids();
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for slot-transfer auxiliary sampling"
        );
        assert!(
            parallelism <= device_ids.len(),
            "slot-transfer GPU parallelism must be <= detected GPU count: requested={}, devices={}",
            parallelism,
            device_ids.len()
        );

        let b0_matrix_bytes = b0_matrix.to_compact_bytes();
        let b1_matrix_bytes = b1_matrix.to_compact_bytes();
        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);
        let b1_trapdoor_bytes = TS::trapdoor_to_bytes(b1_trapdoor);

        device_ids
            .into_par_iter()
            .take(parallelism)
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                GpuSlotTransferDeviceShared {
                    identity: M::identity(&local_params, self.secret_size, None),
                    gadget_matrix: M::gadget_matrix(&local_params, self.secret_size),
                    b0_trapdoor: TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                        .unwrap_or_else(|| {
                            panic!(
                                "failed to decode slot-transfer b0 trapdoor for device {device_id}"
                            )
                        }),
                    b0_matrix: M::from_compact_bytes(&local_params, &b0_matrix_bytes),
                    b1_trapdoor: TS::trapdoor_from_bytes(&local_params, &b1_trapdoor_bytes)
                        .unwrap_or_else(|| {
                            panic!(
                                "failed to decode slot-transfer b1 trapdoor for device {device_id}"
                            )
                        }),
                    b1_matrix: M::from_compact_bytes(&local_params, &b1_matrix_bytes),
                    params: local_params,
                }
            })
            .collect()
    }

    pub(crate) fn copy_compact_matrices_to_gpu_devices(
        &self,
        matrix_bytes: &[Vec<u8>],
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
    ) -> Vec<Vec<M>> {
        shared_by_device
            .par_iter()
            .map(|shared| {
                matrix_bytes
                    .iter()
                    .map(|bytes| M::from_compact_bytes(&shared.params, bytes))
                    .collect()
            })
            .collect()
    }

    pub(crate) fn sample_slot_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        slot_secret_mats_by_device: &[Vec<M>],
        batch_slot_indices: &[usize],
    ) -> Vec<SlotAuxSample<M>> {
        let b1_size = self.secret_size * 2;
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let slot_targets = batch_slot_indices
            .par_iter()
            .copied()
            .enumerate()
            .map(|(batch_pos, slot_idx)| {
                let device_idx = batch_pos % shared_by_device.len();
                let shared = &shared_by_device[device_idx];
                let m_g = self.secret_size * shared.params.modulus_digits();
                let slot_secret_mat = &slot_secret_mats_by_device[device_idx][slot_idx];
                assert_eq!(
                    slot_secret_mat.row_size(),
                    self.secret_size,
                    "slot secret matrix {} row size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.row_size(),
                    self.secret_size
                );
                assert_eq!(
                    slot_secret_mat.col_size(),
                    self.secret_size,
                    "slot secret matrix {} col size {} must match secret_size {}",
                    slot_idx,
                    slot_secret_mat.col_size(),
                    self.secret_size
                );
                let a_i = HS::new().sample_hash(
                    &shared.params,
                    self.hash_key,
                    format!("slot_transfer_slot_a_{}", slot_idx),
                    self.secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                let s_concat_identity = slot_secret_mat.clone().concat_columns(&[&shared.identity]);
                let mut target_b0 = s_concat_identity * &shared.b1_matrix;
                target_b0.add_in_place(&self.sample_error_matrix(
                    &shared.params,
                    self.secret_size,
                    shared.b1_matrix.col_size(),
                ));

                let neg_slot_secret_gadget = -(slot_secret_mat.clone() * &shared.gadget_matrix);
                let mut target_b1 = a_i.clone().concat_rows(&[&neg_slot_secret_gadget]);
                target_b1.add_in_place(&self.sample_error_matrix(&shared.params, b1_size, m_g));
                let slot_a_bytes = a_i.to_compact_bytes();
                GpuSlotAuxTarget {
                    slot_idx,
                    device_idx,
                    slot_a: a_i,
                    slot_a_bytes,
                    target_b0,
                    target_b1,
                }
            })
            .collect::<Vec<_>>();
        let mut slot_targets_by_idx = slot_targets
            .into_iter()
            .map(|target| (target.slot_idx, target))
            .collect::<HashMap<usize, GpuSlotAuxTarget<M>>>();

        let requests_b0 = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| {
                let slot_target = slot_targets_by_idx
                    .get_mut(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu slot target for slot_idx={slot_idx}"));
                let shared = &shared_by_device[slot_target.device_idx];
                GpuPreimageRequest {
                    entry_idx: slot_idx,
                    params: &shared.params,
                    trapdoor: &shared.b0_trapdoor,
                    public_matrix: &shared.b0_matrix,
                    target: std::mem::replace(
                        &mut slot_target.target_b0,
                        M::zero(&shared.params, self.secret_size, shared.b1_matrix.col_size()),
                    ),
                }
            })
            .collect::<Vec<_>>();
        let mut preimages_b0 = trap_sampler
            .preimage_batched_sharded(requests_b0)
            .into_iter()
            .collect::<HashMap<usize, M>>();

        let requests_b1 = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| {
                let slot_target = slot_targets_by_idx
                    .get_mut(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu slot target for slot_idx={slot_idx}"));
                let shared = &shared_by_device[slot_target.device_idx];
                GpuPreimageRequest {
                    entry_idx: slot_idx,
                    params: &shared.params,
                    trapdoor: &shared.b1_trapdoor,
                    public_matrix: &shared.b1_matrix,
                    target: std::mem::replace(
                        &mut slot_target.target_b1,
                        M::zero(&shared.params, b1_size, shared.b1_matrix.col_size()),
                    ),
                }
            })
            .collect::<Vec<_>>();
        let mut preimages_b1 = trap_sampler
            .preimage_batched_sharded(requests_b1)
            .into_iter()
            .collect::<HashMap<usize, M>>();

        batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| {
                let slot_target = slot_targets_by_idx
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu slot target for slot_idx={slot_idx}"));
                let preimage_b0 = preimages_b0
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu b0 preimage for slot_idx={slot_idx}"));
                let preimage_b1 = preimages_b1
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu b1 preimage for slot_idx={slot_idx}"));
                SlotAuxSample {
                    slot_idx,
                    slot_a: slot_target.slot_a,
                    slot_a_bytes: slot_target.slot_a_bytes,
                    preimage_b0,
                    preimage_b1,
                }
            })
            .collect()
    }

    pub(crate) fn sample_gate_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        slot_secret_mats_by_device: &[Vec<M>],
        slot_a_mats_by_device: &[Vec<M>],
        batch: &[(GateId, BggPublicKeySTGateState)],
    ) -> Vec<GateAuxSample<M>> {
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let request_targets = batch
            .par_iter()
            .enumerate()
            .map(|(batch_pos, (gate_id, state))| {
                let device_idx = batch_pos % shared_by_device.len();
                let shared = &shared_by_device[device_idx];
                let m_g = self.secret_size * shared.params.modulus_digits();
                let a_in = M::from_compact_bytes(&shared.params, &state.input_pubkey_bytes);
                let a_out = HS::new().sample_hash(
                    &shared.params,
                    self.hash_key,
                    format!("slot_transfer_gate_a_out_{}", gate_id),
                    self.secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                state
                    .src_slots
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(dst_slot, src_slot_u32)| {
                        let src_slot = usize::try_from(src_slot_u32)
                            .expect("source slot index must fit in usize");
                        assert!(
                            src_slot < self.num_slots,
                            "source slot index {} out of range for num_slots {}",
                            src_slot,
                            self.num_slots
                        );
                        let s_j = &slot_secret_mats_by_device[device_idx][dst_slot];
                        let s_i = &slot_secret_mats_by_device[device_idx][src_slot];
                        let a_j = &slot_a_mats_by_device[device_idx][dst_slot];
                        let lhs = s_j.clone() * &a_out;
                        let rhs = (s_i.clone() * &a_in) * a_j.decompose();
                        let mut target = lhs - rhs;
                        target.add_in_place(&self.sample_error_matrix(
                            &shared.params,
                            self.secret_size,
                            m_g,
                        ));
                        (*gate_id, dst_slot, device_idx, target)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(entry_idx, (gate_id, dst_slot, device_idx, target))| {
                (entry_idx, gate_id, dst_slot, device_idx, target)
            })
            .collect::<Vec<_>>();

        let requests = request_targets
            .iter()
            .map(|(entry_idx, _, _, device_idx, target)| {
                let shared = &shared_by_device[*device_idx];
                GpuPreimageRequest {
                    entry_idx: *entry_idx,
                    params: &shared.params,
                    trapdoor: &shared.b0_trapdoor,
                    public_matrix: &shared.b0_matrix,
                    target: target.clone(),
                }
            })
            .collect::<Vec<_>>();
        let mut preimages_by_entry = trap_sampler
            .preimage_batched_sharded(requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let mut grouped = batch
            .par_iter()
            .map(|(gate_id, _)| {
                (*gate_id, GateAuxSample { gate_id: *gate_id, dst_preimages: Vec::new() })
            })
            .collect::<HashMap<GateId, GateAuxSample<M>>>();

        for (entry_idx, gate_id, dst_slot, _, _) in request_targets {
            let preimage = preimages_by_entry
                .remove(&entry_idx)
                .unwrap_or_else(|| panic!("missing gpu gate preimage for entry_idx={entry_idx}"));
            grouped
                .get_mut(&gate_id)
                .unwrap_or_else(|| panic!("missing gate grouping for gate_id={gate_id}"))
                .dst_preimages
                .push((dst_slot, preimage));
        }

        batch
            .iter()
            .map(|(gate_id, _)| {
                grouped.remove(gate_id).unwrap_or_else(|| {
                    panic!("missing grouped gpu gate sample for gate_id={gate_id}")
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        __PAIR, __TestState,
        bgg::public_key::BggPublicKey,
        circuit::{PolyCircuit, evaluable::Evaluable, gate::GateId},
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
        sampler::{
            DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::BggPublicKeySTEvaluator,
        storage::{
            read::read_matrix_from_multi_batch,
            write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        },
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    const SIGMA: f64 = 4.578;

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<GpuDCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::Params,
            _plt: &PublicLut<<BggPublicKey<GpuDCRTPolyMatrix> as Evaluable>::P>,
            _one: &BggPublicKey<GpuDCRTPolyMatrix>,
            _input: &BggPublicKey<GpuDCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPublicKey<GpuDCRTPolyMatrix> {
            unreachable!("dummy evaluator should never be called in slot-transfer GPU-path tests")
        }
    }

    fn single_gpu_params() -> GpuDCRTPolyParams {
        let cpu_params = DCRTPolyParams::default();
        let (moduli, _, _) = cpu_params.to_crt();
        let detected_gpu_params = GpuDCRTPolyParams::new(
            cpu_params.ring_dimension(),
            moduli.clone(),
            cpu_params.base_bits(),
        );
        let single_gpu_id = *detected_gpu_params
            .gpu_ids()
            .first()
            .expect("at least one GPU device is required for slot-transfer GPU tests");
        GpuDCRTPolyParams::new_with_gpu(
            cpu_params.ring_dimension(),
            moduli,
            cpu_params.base_bits(),
            vec![single_gpu_id],
            Some(1),
        )
    }

    #[sequential_test::sequential]
    #[test]
    fn bgg_public_key_st_evaluator_uses_parallel_slot_transfer_path_with_gpu_feature() {
        gpu_device_sync();
        let params = single_gpu_params();
        let hash_key = [0x73u8; 32];
        let num_inputs = 3usize;
        let src_slots = [1, 0];
        let m_g = 2 * params.modulus_digits();
        let one = BggPublicKey::new(
            GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                "slot_transfer_gpu_one".to_string(),
                2,
                m_g,
                DistType::FinRingDist,
            ),
            true,
        );
        let inputs = (0..num_inputs)
            .map(|idx| {
                let scalar = GpuDCRTPoly::from_usize_to_constant(&params, idx + 1);
                let base = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &params,
                    hash_key,
                    format!("slot_transfer_gpu_input_{}", idx),
                    2,
                    m_g,
                    DistType::FinRingDist,
                );
                BggPublicKey::new(base * scalar, idx % 2 == 0)
            })
            .collect::<Vec<_>>();

        let mut circuit = PolyCircuit::new();
        let input_gates = circuit.input(num_inputs);
        let transferred_gates = input_gates
            .iter()
            .map(|&gate| circuit.slot_transfer_gate(gate, &src_slots))
            .collect::<Vec<_>>();
        circuit.output(transferred_gates.clone());

        let evaluator = BggPublicKeySTEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new(
            hash_key,
            2,
            src_slots.len(),
            SIGMA,
            0.0,
            "test_data/test_slot_transfer_gpu".into(),
        );
        let outputs = circuit.eval(
            &params,
            one,
            inputs.clone(),
            None::<&DummyPubKeyPltEvaluator>,
            Some(&evaluator),
            Some(1),
        );

        assert_eq!(outputs.len(), transferred_gates.len());
        for ((output, input), gate_id) in
            outputs.iter().zip(inputs.iter()).zip(transferred_gates.iter())
        {
            let expected_matrix = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_gate_a_out_{}", gate_id),
                input.matrix.row_size(),
                input.matrix.row_size() * params.modulus_digits(),
                DistType::FinRingDist,
            );
            assert_eq!(*output, BggPublicKey::new(expected_matrix, true));

            let stored = evaluator.gate_state(*gate_id).expect("missing stored gate state");
            assert_eq!(stored.input_pubkey_bytes, input.matrix.to_compact_bytes());
            assert_eq!(stored.src_slots, src_slots);
        }
        gpu_device_sync();
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn bgg_public_key_st_evaluator_samples_aux_matrices_with_gpu_feature() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        for iter in 0..5usize {
            gpu_device_sync();
            let params = single_gpu_params();
            let hash_key = [0x55u8; 32];
            let secret_size = 2usize;
            let num_slots = 2usize;
            let dir_path = format!("test_data/test_slot_transfer_gpu_aux_{iter}");
            let dir = Path::new(&dir_path);
            if dir.exists() {
                fs::remove_dir_all(dir).unwrap();
            }
            fs::create_dir_all(dir).unwrap();
            init_storage_system(dir.to_path_buf());

            let input_pubkey = BggPublicKey::new(
                GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &params,
                    hash_key,
                    format!("slot_transfer_gpu_aux_input_{iter}"),
                    secret_size,
                    secret_size * params.modulus_digits(),
                    DistType::FinRingDist,
                ),
                true,
            );
            let one = BggPublicKey::new(
                GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &params,
                    hash_key,
                    format!("slot_transfer_gpu_aux_one_{iter}"),
                    secret_size,
                    secret_size * params.modulus_digits(),
                    DistType::FinRingDist,
                ),
                true,
            );

            let mut circuit = PolyCircuit::new();
            let inputs = circuit.input(1);
            let transferred = circuit.slot_transfer_gate(inputs[0], &[1, 0]);
            circuit.output(vec![transferred]);

            let evaluator = BggPublicKeySTEvaluator::<
                GpuDCRTPolyMatrix,
                GpuDCRTPolyUniformSampler,
                GpuDCRTPolyHashSampler<Keccak256>,
                GpuDCRTPolyTrapdoorSampler,
            >::new(
                hash_key, secret_size, num_slots, SIGMA, 0.0, dir.to_path_buf()
            );
            let result = circuit.eval(
                &params,
                one,
                vec![input_pubkey.clone()],
                None::<&DummyPubKeyPltEvaluator>,
                Some(&evaluator),
                None,
            );
            assert_eq!(result.len(), 1);

            let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(&params, SIGMA);
            let (b0_trapdoor, b0_matrix) = trapdoor_sampler.trapdoor(&params, secret_size);
            let slot_secret_mats = (0..num_slots)
                .map(|_| {
                    GpuDCRTPolyUniformSampler::new()
                        .sample_uniform(&params, secret_size, secret_size, DistType::TernaryDist)
                        .into_compact_bytes()
                })
                .collect::<Vec<_>>();

            evaluator.sample_aux_matrices(
                &params,
                &b0_matrix,
                &b0_trapdoor,
                slot_secret_mats.clone(),
            );
            wait_for_all_writes(dir.to_path_buf()).await.unwrap();

            let checkpoint_prefix = evaluator.checkpoint_prefix(&params);
            let b1_matrix = evaluator
                .load_b1_matrix_checkpoint(&params)
                .expect("b1 matrix checkpoint should exist after gpu sample_aux_matrices");
            let loaded_b0 = evaluator
                .load_b0_matrix_checkpoint(&params)
                .expect("b0 matrix checkpoint should exist after gpu sample_aux_matrices");
            assert_eq!(loaded_b0, b0_matrix);

            let identity = GpuDCRTPolyMatrix::identity(&params, secret_size, None);
            let gadget_matrix = GpuDCRTPolyMatrix::gadget_matrix(&params, secret_size);
            let m_g = secret_size * params.modulus_digits();

            for slot_idx in 0..num_slots {
                let s_i = GpuDCRTPolyMatrix::from_compact_bytes(
                    &params,
                    slot_secret_mats[slot_idx].as_ref(),
                );
                let a_i = read_matrix_from_multi_batch::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_slot_a_{slot_idx}"),
                    0,
                )
                .expect("gpu slot A_i checkpoint should exist");
                let expected_a_i = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &params,
                    hash_key,
                    format!("slot_transfer_slot_a_{}", slot_idx),
                    secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                assert_eq!(a_i, expected_a_i);

                let slot_preimage_b0 = read_matrix_from_multi_batch::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_slot_preimage_b0_{slot_idx}"),
                    0,
                )
                .expect("gpu slot preimage in b0 basis should exist");
                let slot_preimage_b1 = read_matrix_from_multi_batch::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_slot_preimage_b1_{slot_idx}"),
                    0,
                )
                .expect("gpu slot preimage in b1 basis should exist");

                let expected_target_b0 = s_i.clone().concat_columns(&[&identity]) * &b1_matrix;
                let neg_slot_secret_gadget = -(s_i.clone() * &gadget_matrix);
                let expected_target_b1 = a_i.clone().concat_rows(&[&neg_slot_secret_gadget]);
                assert_eq!(b0_matrix.clone() * &slot_preimage_b0, expected_target_b0);
                assert_eq!(b1_matrix.clone() * &slot_preimage_b1, expected_target_b1);
            }

            let a_out = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_gate_a_out_{}", transferred),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );

            for (dst_slot, src_slot) in [1usize, 0usize].into_iter().enumerate() {
                let s_j = GpuDCRTPolyMatrix::from_compact_bytes(
                    &params,
                    slot_secret_mats[dst_slot].as_ref(),
                );
                let s_i = GpuDCRTPolyMatrix::from_compact_bytes(
                    &params,
                    slot_secret_mats[src_slot].as_ref(),
                );
                let a_j = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                    &params,
                    hash_key,
                    format!("slot_transfer_slot_a_{}", dst_slot),
                    secret_size,
                    m_g,
                    DistType::FinRingDist,
                );
                let gate_preimage = read_matrix_from_multi_batch::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_gate_preimage_{}_dst{}", transferred, dst_slot),
                    0,
                )
                .expect("gpu gate preimage checkpoint should exist");

                let expected_target = s_j * &a_out - (s_i * &input_pubkey.matrix) * a_j.decompose();
                assert_eq!(b0_matrix.clone() * &gate_preimage, expected_target);
            }

            gpu_device_sync();
        }
    }
}
