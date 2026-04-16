use crate::{
    bench_estimator::{SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator},
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        trapdoor::GpuPreimageRequest,
    },
    slot_transfer::bgg_pubkey::{BggPublicKeySTEvaluator, BggPublicKeySTGateState, SlotAuxSample},
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, sync_channel},
    thread,
    time::Instant,
};

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
    target_b0: M,
    target_b1: M,
}

struct PreparedSlotBatch<M>
where
    M: PolyMatrix,
{
    batch_slot_indices: Vec<usize>,
    slot_secret_mats_by_device: Vec<Vec<M>>,
}

struct ComputedSlotBatch<M>
where
    M: PolyMatrix,
{
    samples: Vec<SlotAuxSample<M>>,
}

struct PreparedGateBatch<M>
where
    M: PolyMatrix,
{
    selected_slot_indices: Vec<usize>,
    slot_chunk: Vec<(usize, (u32, Option<u32>))>,
    slot_secret_mats_by_device: Vec<Vec<M>>,
    slot_a_mats_by_device: Vec<Vec<M>>,
}

struct ComputedGateBatch<M>
where
    M: PolyMatrix,
{
    gate_preimages: Vec<(usize, M)>,
}

fn slot_aux_samples_compact_bytes<M>(samples: &[SlotAuxSample<M>]) -> u64
where
    M: PolyMatrix,
{
    samples.iter().fold(0u64, |total, sample| {
        total
            .checked_add(
                u64::try_from(sample.slot_a.to_compact_bytes().len())
                    .expect("slot_a compact_bytes length overflowed u64"),
            )
            .and_then(|total| {
                total.checked_add(
                    u64::try_from(sample.preimage_b0.to_compact_bytes().len())
                        .expect("slot preimage_b0 compact_bytes length overflowed u64"),
                )
            })
            .and_then(|total| {
                total.checked_add(
                    u64::try_from(sample.preimage_b1.to_compact_bytes().len())
                        .expect("slot preimage_b1 compact_bytes length overflowed u64"),
                )
            })
            .expect("slot auxiliary compact_bytes total overflowed u64")
    })
}

fn gate_preimages_compact_bytes<M>(preimages: &[(usize, M)]) -> u64
where
    M: PolyMatrix,
{
    preimages.iter().fold(0u64, |total, (_, preimage)| {
        total
            .checked_add(
                u64::try_from(preimage.to_compact_bytes().len())
                    .expect("gate preimage compact_bytes length overflowed u64"),
            )
            .expect("gate preimage compact_bytes total overflowed u64")
    })
}

fn sharded_batch_device_idx(
    num_devices: usize,
    batch_device_offset: usize,
    batch_pos: usize,
) -> usize {
    assert!(num_devices > 0, "batch sharding requires at least one device");
    (batch_device_offset + batch_pos) % num_devices
}

fn recv_next<T>(rx: &Receiver<T>, stage: &'static str) -> Option<T> {
    match rx.recv() {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::debug!(stage, "slot-transfer gpu pipeline stage completed");
            None
        }
    }
}

impl<M, US, HS, TS> BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync,
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

    fn sample_slot_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        prepared_batch: PreparedSlotBatch<M>,
    ) -> ComputedSlotBatch<M> {
        let b1_size = self.secret_size * 2;
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let PreparedSlotBatch { batch_slot_indices, slot_secret_mats_by_device } = prepared_batch;
        let slot_targets = batch_slot_indices
            .par_iter()
            .copied()
            .enumerate()
            .map(|(local_idx, slot_idx)| {
                let device_idx = sharded_batch_device_idx(shared_by_device.len(), 0, local_idx);
                let shared = &shared_by_device[device_idx];
                let m_g = self.secret_size * shared.params.modulus_digits();
                let slot_secret_mat = &slot_secret_mats_by_device[device_idx][local_idx];
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
                GpuSlotAuxTarget { slot_idx, device_idx, slot_a: a_i, target_b0, target_b1 }
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

        let samples = batch_slot_indices
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
                SlotAuxSample { slot_idx, slot_a: slot_target.slot_a, preimage_b0, preimage_b1 }
            })
            .collect();
        ComputedSlotBatch { samples }
    }

    fn prepare_slot_batch_gpu(
        &self,
        slot_secret_mats: &[Vec<u8>],
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        batch_slot_indices: &[usize],
    ) -> PreparedSlotBatch<M> {
        let selected_slot_secret_mats = batch_slot_indices
            .iter()
            .map(|slot_idx| slot_secret_mats[*slot_idx].clone())
            .collect::<Vec<_>>();
        let slot_secret_mats_by_device =
            self.copy_compact_matrices_to_gpu_devices(&selected_slot_secret_mats, shared_by_device);
        PreparedSlotBatch {
            batch_slot_indices: batch_slot_indices.to_vec(),
            slot_secret_mats_by_device,
        }
    }

    fn finalize_slot_batch_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        computed_batch: ComputedSlotBatch<M>,
        slot_a_bytes_by_slot: &mut [Vec<u8>],
    ) {
        let stored_samples = computed_batch
            .samples
            .into_par_iter()
            .map(|sample| {
                let slot_a_bytes = sample.slot_a.to_compact_bytes();
                let preimage_b0_bytes = sample.preimage_b0.to_compact_bytes();
                let preimage_b1_bytes = sample.preimage_b1.to_compact_bytes();
                (sample.slot_idx, slot_a_bytes, preimage_b0_bytes, preimage_b1_bytes)
            })
            .collect::<Vec<_>>();

        for (slot_idx, slot_a_bytes, preimage_b0_bytes, preimage_b1_bytes) in stored_samples {
            slot_a_bytes_by_slot[slot_idx] = slot_a_bytes.clone();
            Self::store_bytes_checkpoint(slot_a_bytes, &self.slot_a_id_prefix(params, slot_idx));
            Self::store_bytes_checkpoint(
                preimage_b0_bytes,
                &self.slot_preimage_b0_id_prefix(params, slot_idx),
            );
            Self::store_bytes_checkpoint(
                preimage_b1_bytes,
                &self.slot_preimage_b1_id_prefix(params, slot_idx),
            );
        }
    }

    pub(crate) fn sample_slot_batches_gpu_pipelined(
        &self,
        params: &<M::P as Poly>::Params,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        slot_secret_mats: &[Vec<u8>],
        slot_parallelism: usize,
    ) -> Vec<Vec<u8>> {
        let chunks = (0..self.num_slots)
            .collect::<Vec<_>>()
            .chunks(slot_parallelism.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let mut slot_a_bytes_by_slot = vec![Vec::new(); self.num_slots];
        if chunks.is_empty() {
            return slot_a_bytes_by_slot;
        }

        thread::scope(|scope| {
            let slot_a_bytes_by_slot_ref = &mut slot_a_bytes_by_slot;
            let (prepared_tx, prepared_rx) = sync_channel::<PreparedSlotBatch<M>>(1);
            let (computed_tx, computed_rx) = sync_channel::<ComputedSlotBatch<M>>(1);

            let load_handle = scope.spawn(move || {
                for batch_slot_indices in chunks {
                    let prepared_batch = self.prepare_slot_batch_gpu(
                        slot_secret_mats,
                        shared_by_device,
                        &batch_slot_indices,
                    );
                    prepared_tx.send(prepared_batch).expect("slot pipeline compute stage dropped");
                }
            });
            let compute_handle = scope.spawn(move || {
                while let Some(prepared_batch) = recv_next(&prepared_rx, "slot-load") {
                    let computed_batch =
                        self.sample_slot_batch_gpu(shared_by_device, prepared_batch);
                    computed_tx.send(computed_batch).expect("slot pipeline store stage dropped");
                }
            });
            let store_handle = scope.spawn(move || {
                while let Some(computed_batch) = recv_next(&computed_rx, "slot-compute") {
                    self.finalize_slot_batch_gpu(params, computed_batch, slot_a_bytes_by_slot_ref);
                }
            });

            load_handle.join().expect("slot pipeline load stage panicked");
            compute_handle.join().expect("slot pipeline compute stage panicked");
            store_handle.join().expect("slot pipeline store stage panicked");
        });

        slot_a_bytes_by_slot
    }

    fn sample_gate_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        gate_id: GateId,
        state: &BggPublicKeySTGateState,
        prepared_batch: PreparedGateBatch<M>,
    ) -> ComputedGateBatch<M> {
        let PreparedGateBatch {
            selected_slot_indices,
            slot_chunk,
            slot_secret_mats_by_device,
            slot_a_mats_by_device,
        } = prepared_batch;
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let selected_slot_positions = selected_slot_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(local_idx, slot_idx)| (slot_idx, local_idx))
            .collect::<HashMap<usize, usize>>();
        let request_targets = slot_chunk
            .par_iter()
            .copied()
            .enumerate()
            .map(|(batch_pos, (dst_slot, (src_slot_u32, scalar)))| {
                let device_idx = sharded_batch_device_idx(shared_by_device.len(), 0, batch_pos);
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
                let src_slot =
                    usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
                assert!(
                    src_slot < self.num_slots,
                    "source slot index {} out of range for num_slots {}",
                    src_slot,
                    self.num_slots
                );
                let dst_local_idx = *selected_slot_positions
                    .get(&dst_slot)
                    .unwrap_or_else(|| panic!("missing selected dst_slot={dst_slot}"));
                let src_local_idx = *selected_slot_positions
                    .get(&src_slot)
                    .unwrap_or_else(|| panic!("missing selected src_slot={src_slot}"));
                let s_j = &slot_secret_mats_by_device[device_idx][dst_local_idx];
                let s_i = &slot_secret_mats_by_device[device_idx][src_local_idx];
                let a_j = &slot_a_mats_by_device[device_idx][dst_local_idx];
                let lhs = s_j.clone() * &a_out;
                let rhs = match scalar {
                    Some(scalar) => {
                        let scalar_poly =
                            M::P::from_usize_to_constant(&shared.params, scalar as usize);
                        ((s_i.clone() * &a_in) * a_j.decompose()) * scalar_poly
                    }
                    None => (s_i.clone() * &a_in) * a_j.decompose(),
                };
                let mut target = lhs - rhs;
                target.add_in_place(&self.sample_error_matrix(
                    &shared.params,
                    self.secret_size,
                    m_g,
                ));
                (dst_slot, device_idx, target)
            })
            .collect::<Vec<_>>();

        let requests = request_targets
            .iter()
            .map(|(dst_slot, device_idx, target)| {
                let shared = &shared_by_device[*device_idx];
                GpuPreimageRequest {
                    entry_idx: *dst_slot,
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

        let gate_preimages = slot_chunk
            .iter()
            .map(|(dst_slot, _)| {
                let preimage = preimages_by_entry
                    .remove(dst_slot)
                    .unwrap_or_else(|| panic!("missing gpu gate preimage for dst_slot={dst_slot}"));
                (*dst_slot, preimage)
            })
            .collect();
        ComputedGateBatch { gate_preimages }
    }

    fn prepare_gate_batch_gpu(
        &self,
        slot_secret_mats: &[Vec<u8>],
        slot_a_bytes_by_slot: &[Vec<u8>],
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        slot_chunk: &[(usize, (u32, Option<u32>))],
    ) -> PreparedGateBatch<M> {
        let mut selected_slot_indices = slot_chunk
            .iter()
            .flat_map(|(dst_slot, (src_slot_u32, _))| {
                let src_slot =
                    usize::try_from(*src_slot_u32).expect("source slot index must fit in usize");
                [*dst_slot, src_slot]
            })
            .collect::<Vec<_>>();
        selected_slot_indices.sort_unstable();
        selected_slot_indices.dedup();

        let selected_slot_secret_mats = selected_slot_indices
            .iter()
            .map(|slot_idx| slot_secret_mats[*slot_idx].clone())
            .collect::<Vec<_>>();
        let slot_secret_mats_by_device =
            self.copy_compact_matrices_to_gpu_devices(&selected_slot_secret_mats, shared_by_device);
        let selected_slot_a_mats = selected_slot_indices
            .iter()
            .map(|slot_idx| slot_a_bytes_by_slot[*slot_idx].clone())
            .collect::<Vec<_>>();
        let slot_a_mats_by_device =
            self.copy_compact_matrices_to_gpu_devices(&selected_slot_a_mats, shared_by_device);

        PreparedGateBatch {
            selected_slot_indices,
            slot_chunk: slot_chunk.to_vec(),
            slot_secret_mats_by_device,
            slot_a_mats_by_device,
        }
    }

    fn finalize_gate_batch_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        computed_batch: ComputedGateBatch<M>,
    ) {
        let stored_preimages = computed_batch
            .gate_preimages
            .into_par_iter()
            .map(|(dst_slot, preimage)| (dst_slot, preimage.to_compact_bytes()))
            .collect::<Vec<_>>();

        for (dst_slot, preimage_bytes) in stored_preimages {
            Self::store_bytes_checkpoint(
                preimage_bytes,
                &self.gate_preimage_id_prefix(params, gate_id, dst_slot),
            );
        }
    }

    pub(crate) fn sample_gate_batches_gpu_pipelined(
        &self,
        params: &<M::P as Poly>::Params,
        shared_by_device: &[GpuSlotTransferDeviceShared<M, TS::Trapdoor>],
        slot_secret_mats: &[Vec<u8>],
        slot_a_bytes_by_slot: &[Vec<u8>],
        gate_id: GateId,
        state: &BggPublicKeySTGateState,
        gate_parallelism: usize,
    ) {
        let chunks = state
            .src_slots
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>()
            .chunks(gate_parallelism.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        if chunks.is_empty() {
            return;
        }

        thread::scope(|scope| {
            let (prepared_tx, prepared_rx) = sync_channel::<PreparedGateBatch<M>>(1);
            let (computed_tx, computed_rx) = sync_channel::<ComputedGateBatch<M>>(1);

            let load_handle = scope.spawn(move || {
                for slot_chunk in chunks {
                    let prepared_batch = self.prepare_gate_batch_gpu(
                        slot_secret_mats,
                        slot_a_bytes_by_slot,
                        shared_by_device,
                        &slot_chunk,
                    );
                    prepared_tx.send(prepared_batch).expect("gate pipeline compute stage dropped");
                }
            });
            let compute_handle = scope.spawn(move || {
                while let Some(prepared_batch) = recv_next(&prepared_rx, "gate-load") {
                    let computed_batch = self.sample_gate_batch_gpu(
                        shared_by_device,
                        gate_id,
                        state,
                        prepared_batch,
                    );
                    computed_tx.send(computed_batch).expect("gate pipeline store stage dropped");
                }
            });
            let store_handle = scope.spawn(move || {
                while let Some(computed_batch) = recv_next(&computed_rx, "gate-compute") {
                    self.finalize_gate_batch_gpu(params, gate_id, computed_batch);
                }
            });

            load_handle.join().expect("gate pipeline load stage panicked");
            compute_handle.join().expect("gate pipeline compute stage panicked");
            store_handle.join().expect("gate pipeline store stage panicked");
        });
    }
}

impl<M, US, HS, TS> SlotTransferSampleAuxBenchEstimator for BggPublicKeySTEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    type Params = <M::P as Poly>::Params;

    fn sample_aux_matrices_slot_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let b1_size =
            self.secret_size.checked_mul(2).expect("slot-transfer benchmark b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        let gpu_shared = self.prepare_gpu_device_shared(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            1,
        );
        let batch_slot_indices = vec![0usize];
        let slot_secret_mats = vec![
            US::new()
                .sample_uniform(&params, self.secret_size, self.secret_size, DistType::TernaryDist)
                .into_compact_bytes(),
        ];
        let prepared_batch =
            self.prepare_slot_batch_gpu(&slot_secret_mats, &gpu_shared, &batch_slot_indices);
        let start = Instant::now();
        let computed_batch = self.sample_slot_batch_gpu(&gpu_shared, prepared_batch);
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate {
            latency: elapsed,
            total_time: elapsed,
            compact_bytes: slot_aux_samples_compact_bytes(&computed_batch.samples),
        }
    }

    fn sample_aux_matrices_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let b1_size =
            self.secret_size.checked_mul(2).expect("slot-transfer benchmark b1 size overflow");
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        let benchmark_num_slots = self.num_slots.max(1);
        let benchmark_parallelism = crate::env::slot_transfer_slot_parallelism()
            .max(1)
            .min(benchmark_num_slots)
            .min(detected_gpu_device_ids().len().max(1));
        let gpu_shared = self.prepare_gpu_device_shared(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix,
            &b1_trapdoor,
            benchmark_parallelism,
        );
        let slot_secret_mats = (0..benchmark_num_slots)
            .map(|_| {
                US::new()
                    .sample_uniform(
                        params,
                        self.secret_size,
                        self.secret_size,
                        DistType::TernaryDist,
                    )
                    .into_compact_bytes()
            })
            .collect::<Vec<_>>();
        let slot_indices = (0..benchmark_num_slots).collect::<Vec<_>>();
        let prepared_slot_batch =
            self.prepare_slot_batch_gpu(&slot_secret_mats, &gpu_shared, &slot_indices);
        let sampled_slots = self.sample_slot_batch_gpu(&gpu_shared, prepared_slot_batch);
        let mut slot_a_bytes_by_slot = vec![Vec::new(); benchmark_num_slots];
        for sample in sampled_slots.samples {
            slot_a_bytes_by_slot[sample.slot_idx] = sample.slot_a.to_compact_bytes();
        }
        let state = BggPublicKeySTGateState {
            input_pubkey_bytes: M::gadget_matrix(params, self.secret_size).into_compact_bytes(),
            src_slots: (0..benchmark_num_slots).map(|slot_idx| (slot_idx as u32, None)).collect(),
        };
        let slot_chunk = state.src_slots.iter().copied().enumerate().collect::<Vec<_>>();
        let prepared_batch = self.prepare_gate_batch_gpu(
            &slot_secret_mats,
            &slot_a_bytes_by_slot,
            &gpu_shared,
            &slot_chunk,
        );
        let start = Instant::now();
        let computed_batch =
            self.sample_gate_batch_gpu(&gpu_shared, GateId(0), &state, prepared_batch);
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate {
            latency: elapsed,
            total_time: elapsed,
            compact_bytes: gate_preimages_compact_bytes(&computed_batch.gate_preimages),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sharded_batch_device_idx;
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
            DistType, PolyHashSampler,
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
        storage::{
            read::read_matrix_from_multi_batch,
            write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        },
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

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

    #[test]
    fn sharded_batch_device_idx_rotates_with_batch_position() {
        assert_eq!(sharded_batch_device_idx(3, 0, 0), 0);
        assert_eq!(sharded_batch_device_idx(3, 0, 1), 1);
        assert_eq!(sharded_batch_device_idx(3, 0, 2), 2);
        assert_eq!(sharded_batch_device_idx(3, 0, 3), 0);
    }

    #[sequential_test::sequential]
    #[test]
    fn bgg_public_key_st_evaluator_uses_parallel_slot_transfer_path_with_gpu_feature() {
        gpu_device_sync();
        let params = single_gpu_params();
        let hash_key = [0x73u8; 32];
        let num_inputs = 3usize;
        let src_slots = [(1, None), (0, Some(3))];
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
            .map(|gate| circuit.slot_transfer_gate(gate, &src_slots))
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
        for ((output, input), gate) in
            outputs.iter().zip(inputs.iter()).zip(transferred_gates.iter())
        {
            let gate_id = gate.as_single_wire();
            let expected_matrix = GpuDCRTPolyHashSampler::<Keccak256>::new().sample_hash(
                &params,
                hash_key,
                format!("slot_transfer_gate_a_out_{}", gate_id),
                input.matrix.row_size(),
                input.matrix.row_size() * params.modulus_digits(),
                DistType::FinRingDist,
            );
            assert_eq!(*output, BggPublicKey::new(expected_matrix, true));

            let stored = evaluator.gate_state(gate_id).expect("missing stored gate state");
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
        let _parallelism_guard = EnvVarGuard::set("SLOT_TRANSFER_SLOT_PARALLELISM", "1");

        for iter in 0..5usize {
            gpu_device_sync();
            let params = single_gpu_params();
            let hash_key = [0x55u8; 32];
            let secret_size = 2usize;
            let num_slots = 3usize;
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
            let src_slots = [(1, None), (2, Some(3)), (0, Some(5))];
            let transferred = circuit.slot_transfer_gate(inputs.at(0), &src_slots);
            let transferred_gate = transferred.as_single_wire();
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

            evaluator.sample_aux_matrices(&params);
            wait_for_all_writes(dir.to_path_buf()).await.unwrap();
            let slot_secret_mats = evaluator.load_slot_secret_mats_checkpoint(&params).expect(
                "gpu slot secret matrix checkpoints should exist after sample_aux_matrices",
            );

            let checkpoint_prefix = evaluator.checkpoint_prefix(&params);
            let b1_matrix = evaluator
                .load_b1_matrix_checkpoint(&params)
                .expect("b1 matrix checkpoint should exist after gpu sample_aux_matrices");
            let b0_matrix = evaluator
                .load_b0_matrix_checkpoint(&params)
                .expect("b0 matrix checkpoint should exist after gpu sample_aux_matrices");

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
                format!("slot_transfer_gate_a_out_{}", transferred_gate),
                secret_size,
                m_g,
                DistType::FinRingDist,
            );

            for (dst_slot, (src_slot, scalar)) in src_slots.into_iter().enumerate() {
                let src_slot = src_slot as usize;
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
                    &format!(
                        "{checkpoint_prefix}_gate_preimage_{}_dst{}",
                        transferred_gate, dst_slot
                    ),
                    0,
                )
                .expect("gpu gate preimage checkpoint should exist");

                let rhs = match scalar {
                    Some(scalar) => {
                        let scalar_poly =
                            <GpuDCRTPolyMatrix as PolyMatrix>::P::from_usize_to_constant(
                                &params,
                                scalar as usize,
                            );
                        ((s_i * &input_pubkey.matrix) * a_j.decompose()) * scalar_poly
                    }
                    None => (s_i * &input_pubkey.matrix) * a_j.decompose(),
                };
                let expected_target = s_j * &a_out - rhs;
                assert_eq!(b0_matrix.clone() * &gate_preimage, expected_target);
            }

            gpu_device_sync();
        }
    }
}
