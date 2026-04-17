use crate::{
    bench_estimator::{SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator},
    circuit::gate::GateId,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::gpu::detected_gpu_device_ids},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        trapdoor::GpuPreimageRequest,
    },
    slot_transfer::bgg_pubkey::{
        BggPublicKeySTEvaluator, BggPublicKeySTGateState, column_chunk_bounds, column_chunk_count,
        column_chunk_id_prefix,
    },
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, sync_channel},
    thread,
    time::Instant,
};

pub(crate) struct GpuSlotAuxB0DeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: T,
    b0_matrix: M,
    b1_matrix: M,
    identity: M,
}

pub(crate) struct GpuSlotAuxB1DeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b1_trapdoor: T,
    b1_matrix: M,
    gadget_matrix: M,
}

pub(crate) struct GpuSlotGateDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: T,
    b0_matrix: M,
}

struct GpuSlotAuxTarget {
    slot_idx: usize,
    slot_secret_bytes: Vec<u8>,
    slot_a_bytes: Vec<u8>,
}

struct PreparedSlotBatch<M>
where
    M: PolyMatrix,
{
    batch_slot_indices: Vec<usize>,
    slot_secret_bytes_by_slot: HashMap<usize, Vec<u8>>,
    _m: std::marker::PhantomData<M>,
}

struct StoredSlotB0AuxSample {
    slot_idx: usize,
    preimage_b0_bytes: Vec<Vec<u8>>,
}

struct ComputedSlotB0Batch {
    samples: Vec<StoredSlotB0AuxSample>,
}

struct StoredSlotB1AuxSample {
    slot_idx: usize,
    slot_a_bytes: Vec<u8>,
    preimage_b1_bytes: Vec<Vec<u8>>,
}

struct ComputedSlotB1Batch {
    samples: Vec<StoredSlotB1AuxSample>,
}

struct PreparedGateBatch<M>
where
    M: PolyMatrix,
{
    slot_chunk: Vec<(usize, (u32, Option<u32>))>,
    slot_secret_bytes_by_slot: HashMap<usize, Vec<u8>>,
    slot_a_bytes_by_slot: HashMap<usize, Vec<u8>>,
    _m: std::marker::PhantomData<M>,
}

struct ComputedGateBatch {
    gate_preimages: Vec<(usize, Vec<Vec<u8>>)>,
}

#[cfg(test)]
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
    pub(crate) fn prepare_gpu_slot_aux_b0_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        parallelism: usize,
    ) -> Vec<GpuSlotAuxB0DeviceShared<M, TS::Trapdoor>> {
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

        device_ids
            .into_par_iter()
            .take(parallelism)
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                GpuSlotAuxB0DeviceShared {
                    identity: M::identity(&local_params, self.secret_size, None),
                    b0_trapdoor: TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                        .unwrap_or_else(|| {
                            panic!(
                                "failed to decode slot-transfer b0 trapdoor for device {device_id}"
                            )
                        }),
                    b0_matrix: M::from_compact_bytes(&local_params, &b0_matrix_bytes),
                    b1_matrix: M::from_compact_bytes(&local_params, &b1_matrix_bytes),
                    params: local_params,
                }
            })
            .collect()
    }

    pub(crate) fn prepare_gpu_slot_aux_b1_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b1_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        parallelism: usize,
    ) -> Vec<GpuSlotAuxB1DeviceShared<M, TS::Trapdoor>> {
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

        let b1_matrix_bytes = b1_matrix.to_compact_bytes();
        let b1_trapdoor_bytes = TS::trapdoor_to_bytes(b1_trapdoor);

        device_ids
            .into_par_iter()
            .take(parallelism)
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                GpuSlotAuxB1DeviceShared {
                    gadget_matrix: M::gadget_matrix(&local_params, self.secret_size),
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

    pub(crate) fn prepare_gpu_slot_gate_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        parallelism: usize,
    ) -> Vec<GpuSlotGateDeviceShared<M, TS::Trapdoor>> {
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
        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);

        device_ids
            .into_par_iter()
            .take(parallelism)
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                GpuSlotGateDeviceShared {
                    b0_trapdoor: TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                        .unwrap_or_else(|| {
                            panic!(
                                "failed to decode slot-transfer b0 trapdoor for device {device_id}"
                            )
                        }),
                    b0_matrix: M::from_compact_bytes(&local_params, &b0_matrix_bytes),
                    params: local_params,
                }
            })
            .collect()
    }

    fn benchmark_gpu_params(&self, params: &<M::P as Poly>::Params) -> <M::P as Poly>::Params {
        let device_id = *detected_gpu_device_ids()
            .first()
            .expect("at least one GPU device is required for slot-transfer auxiliary benchmarking");
        params.params_for_device(device_id)
    }

    fn prepare_gpu_slot_aux_b0_bench_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
        b1_matrix_bytes: &[u8],
    ) -> GpuSlotAuxB0DeviceShared<M, TS::Trapdoor> {
        let local_params = self.benchmark_gpu_params(params);
        let b0_matrix_bytes = b0_matrix.to_compact_bytes();
        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);
        GpuSlotAuxB0DeviceShared {
            identity: M::identity(&local_params, self.secret_size, None),
            b0_trapdoor: TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                .expect("failed to decode slot-transfer benchmark b0 trapdoor"),
            b0_matrix: M::from_compact_bytes(&local_params, &b0_matrix_bytes),
            b1_matrix: M::from_compact_bytes(&local_params, b1_matrix_bytes),
            params: local_params,
        }
    }

    fn prepare_gpu_slot_aux_b1_bench_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b1_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
    ) -> GpuSlotAuxB1DeviceShared<M, TS::Trapdoor> {
        let local_params = self.benchmark_gpu_params(params);
        let b1_matrix_bytes = b1_matrix.to_compact_bytes();
        let b1_trapdoor_bytes = TS::trapdoor_to_bytes(b1_trapdoor);
        GpuSlotAuxB1DeviceShared {
            gadget_matrix: M::gadget_matrix(&local_params, self.secret_size),
            b1_trapdoor: TS::trapdoor_from_bytes(&local_params, &b1_trapdoor_bytes)
                .expect("failed to decode slot-transfer benchmark b1 trapdoor"),
            b1_matrix: M::from_compact_bytes(&local_params, &b1_matrix_bytes),
            params: local_params,
        }
    }

    fn prepare_gpu_slot_gate_bench_shared(
        &self,
        params: &<M::P as Poly>::Params,
        b0_matrix: &M,
        b0_trapdoor: &TS::Trapdoor,
    ) -> GpuSlotGateDeviceShared<M, TS::Trapdoor> {
        let local_params = self.benchmark_gpu_params(params);
        let b0_matrix_bytes = b0_matrix.to_compact_bytes();
        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);
        GpuSlotGateDeviceShared {
            b0_trapdoor: TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                .expect("failed to decode slot-transfer benchmark b0 trapdoor"),
            b0_matrix: M::from_compact_bytes(&local_params, &b0_matrix_bytes),
            params: local_params,
        }
    }

    fn build_slot_aux_b1_target_chunk_gpu(
        &self,
        shared: &GpuSlotAuxB1DeviceShared<M, TS::Trapdoor>,
        slot_idx: usize,
        slot_secret_mat: &M,
        chunk_idx: usize,
    ) -> M {
        let m_g = self.secret_size * shared.params.modulus_digits();
        let b1_size = self.secret_size * 2;
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let a_i_chunk = HS::new().sample_hash_columns(
            &shared.params,
            self.hash_key,
            format!("slot_transfer_slot_a_{}", slot_idx),
            self.secret_size,
            m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        let neg_slot_secret_gadget_chunk = -(slot_secret_mat.clone() *
            &shared.gadget_matrix.slice(0, self.secret_size, col_start, col_end));
        let mut target_chunk = a_i_chunk.concat_rows(&[&neg_slot_secret_gadget_chunk]);
        target_chunk.add_in_place(&self.sample_error_matrix(&shared.params, b1_size, col_len));
        target_chunk
    }

    fn build_slot_aux_b0_target_chunk_gpu(
        &self,
        shared: &GpuSlotAuxB0DeviceShared<M, TS::Trapdoor>,
        slot_secret_mat: &M,
        chunk_idx: usize,
    ) -> M {
        let b1_size = self.secret_size * 2;
        let b0_target_cols = shared.b1_matrix.col_size();
        let (col_start, col_len) = column_chunk_bounds(b0_target_cols, chunk_idx);
        let col_end = col_start + col_len;
        let s_concat_identity = slot_secret_mat.clone().concat_columns(&[&shared.identity]);
        let mut target_chunk =
            s_concat_identity * &shared.b1_matrix.slice(0, b1_size, col_start, col_end);
        target_chunk.add_in_place(&self.sample_error_matrix(
            &shared.params,
            self.secret_size,
            col_len,
        ));
        target_chunk
    }

    fn build_gate_target_chunk_gpu(
        &self,
        shared: &GpuSlotGateDeviceShared<M, TS::Trapdoor>,
        gate_id: GateId,
        lhs_input: &M,
        s_j: &M,
        a_j: &M,
        scalar_poly: Option<&M::P>,
        chunk_idx: usize,
    ) -> M {
        let m_g = self.secret_size * shared.params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let lhs_chunk = s_j.clone() *
            &HS::new().sample_hash_columns(
                &shared.params,
                self.hash_key,
                format!("slot_transfer_gate_a_out_{}", gate_id),
                self.secret_size,
                m_g,
                col_start,
                col_len,
                DistType::FinRingDist,
            );
        let a_j_chunk = a_j.slice_columns(col_start, col_start + col_len).decompose();
        let rhs_chunk = lhs_input.clone() * a_j_chunk;
        let rhs_chunk = match scalar_poly {
            Some(poly) => rhs_chunk * poly,
            None => rhs_chunk,
        };
        let mut target_chunk = lhs_chunk - rhs_chunk;
        target_chunk.add_in_place(&self.sample_error_matrix(
            &shared.params,
            self.secret_size,
            col_len,
        ));
        target_chunk
    }

    fn sample_slot_b0_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotAuxB0DeviceShared<M, TS::Trapdoor>],
        prepared_batch: PreparedSlotBatch<M>,
    ) -> ComputedSlotB0Batch {
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let PreparedSlotBatch { batch_slot_indices, slot_secret_bytes_by_slot, .. } =
            prepared_batch;
        let b0_target_cols = shared_by_device[0].b1_matrix.col_size();
        let b0_chunk_count = column_chunk_count(b0_target_cols);
        let mut preimages_b0 = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| (slot_idx, vec![Vec::new(); b0_chunk_count]))
            .collect::<HashMap<usize, Vec<Vec<u8>>>>();

        let b0_tasks = batch_slot_indices
            .iter()
            .copied()
            .flat_map(|slot_idx| (0..b0_chunk_count).map(move |chunk_idx| (slot_idx, chunk_idx)))
            .collect::<Vec<_>>();
        for wave in b0_tasks.chunks(shared_by_device.len().max(1)) {
            let requests_b0 = wave
                .par_iter()
                .enumerate()
                .map(|(device_idx, (slot_idx, chunk_idx))| {
                    let shared = &shared_by_device[device_idx];
                    let slot_secret_mat = M::from_compact_bytes(
                        &shared.params,
                        slot_secret_bytes_by_slot.get(slot_idx).unwrap_or_else(|| {
                            panic!("missing slot secret bytes for slot_idx={slot_idx}")
                        }),
                    );
                    let target_chunk = self.build_slot_aux_b0_target_chunk_gpu(
                        shared,
                        &slot_secret_mat,
                        *chunk_idx,
                    );
                    GpuPreimageRequest {
                        entry_idx: device_idx,
                        params: &shared.params,
                        trapdoor: &shared.b0_trapdoor,
                        public_matrix: &shared.b0_matrix,
                        target: target_chunk,
                    }
                })
                .collect::<Vec<_>>();
            let chunk_preimages = trap_sampler.preimage_batched_sharded(requests_b0);
            let mut preimages_by_device =
                chunk_preimages.into_iter().collect::<HashMap<usize, M>>();
            for (device_idx, (slot_idx, chunk_idx)) in wave.iter().enumerate() {
                let preimage_chunk = preimages_by_device.remove(&device_idx).unwrap_or_else(|| {
                    panic!(
                        "missing gpu b0 preimage chunk for slot_idx={}, chunk_idx={}",
                        slot_idx, chunk_idx
                    )
                });
                preimages_b0.get_mut(slot_idx).unwrap_or_else(|| {
                    panic!("missing gpu b0 preimage chunk vec for slot_idx={slot_idx}")
                })[*chunk_idx] = preimage_chunk.to_compact_bytes();
            }
        }

        let samples = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| {
                let preimage_b0_chunks = preimages_b0
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu b0 preimage for slot_idx={slot_idx}"));
                StoredSlotB0AuxSample { slot_idx, preimage_b0_bytes: preimage_b0_chunks }
            })
            .collect();
        ComputedSlotB0Batch { samples }
    }

    fn sample_slot_b1_batch_gpu(
        &self,
        shared_by_device: &[GpuSlotAuxB1DeviceShared<M, TS::Trapdoor>],
        prepared_batch: PreparedSlotBatch<M>,
    ) -> ComputedSlotB1Batch {
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let PreparedSlotBatch { batch_slot_indices, slot_secret_bytes_by_slot, .. } =
            prepared_batch;
        let m_g = self.secret_size * shared_by_device[0].params.modulus_digits();
        let b1_chunk_count = column_chunk_count(m_g);
        let slot_targets = batch_slot_indices
            .par_iter()
            .copied()
            .map(|slot_idx| {
                let shared = &shared_by_device[slot_idx % shared_by_device.len()];
                let slot_a_bytes = HS::new()
                    .sample_hash(
                        &shared.params,
                        self.hash_key,
                        format!("slot_transfer_slot_a_{}", slot_idx),
                        self.secret_size,
                        m_g,
                        DistType::FinRingDist,
                    )
                    .into_compact_bytes();
                let slot_secret_bytes = slot_secret_bytes_by_slot
                    .get(&slot_idx)
                    .unwrap_or_else(|| panic!("missing slot secret bytes for slot_idx={slot_idx}"))
                    .clone();
                GpuSlotAuxTarget { slot_idx, slot_secret_bytes, slot_a_bytes }
            })
            .collect::<Vec<_>>();
        let mut slot_targets_by_idx = slot_targets
            .into_iter()
            .map(|target| (target.slot_idx, target))
            .collect::<HashMap<usize, GpuSlotAuxTarget>>();
        let mut preimages_b1 = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| (slot_idx, vec![Vec::new(); b1_chunk_count]))
            .collect::<HashMap<usize, Vec<Vec<u8>>>>();

        let b1_tasks = batch_slot_indices
            .iter()
            .copied()
            .flat_map(|slot_idx| (0..b1_chunk_count).map(move |chunk_idx| (slot_idx, chunk_idx)))
            .collect::<Vec<_>>();
        for wave in b1_tasks.chunks(shared_by_device.len().max(1)) {
            let requests_b1 = wave
                .par_iter()
                .enumerate()
                .map(|(device_idx, (slot_idx, chunk_idx))| {
                    let slot_target = slot_targets_by_idx.get(slot_idx).unwrap_or_else(|| {
                        panic!("missing gpu slot target for slot_idx={slot_idx}")
                    });
                    let shared = &shared_by_device[device_idx];
                    let slot_secret_mat =
                        M::from_compact_bytes(&shared.params, &slot_target.slot_secret_bytes);
                    let target_chunk = self.build_slot_aux_b1_target_chunk_gpu(
                        shared,
                        *slot_idx,
                        &slot_secret_mat,
                        *chunk_idx,
                    );
                    GpuPreimageRequest {
                        entry_idx: device_idx,
                        params: &shared.params,
                        trapdoor: &shared.b1_trapdoor,
                        public_matrix: &shared.b1_matrix,
                        target: target_chunk,
                    }
                })
                .collect::<Vec<_>>();
            let chunk_preimages = trap_sampler.preimage_batched_sharded(requests_b1);
            let mut preimages_by_device =
                chunk_preimages.into_iter().collect::<HashMap<usize, M>>();
            for (device_idx, (slot_idx, chunk_idx)) in wave.iter().enumerate() {
                let preimage_chunk = preimages_by_device.remove(&device_idx).unwrap_or_else(|| {
                    panic!(
                        "missing gpu b1 preimage chunk for slot_idx={}, chunk_idx={}",
                        slot_idx, chunk_idx
                    )
                });
                preimages_b1.get_mut(slot_idx).unwrap_or_else(|| {
                    panic!("missing gpu b1 preimage chunk vec for slot_idx={slot_idx}")
                })[*chunk_idx] = preimage_chunk.to_compact_bytes();
            }
        }

        let samples = batch_slot_indices
            .iter()
            .copied()
            .map(|slot_idx| {
                let slot_target = slot_targets_by_idx
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu slot target for slot_idx={slot_idx}"));
                let preimage_b1_chunks = preimages_b1
                    .remove(&slot_idx)
                    .unwrap_or_else(|| panic!("missing gpu b1 preimage for slot_idx={slot_idx}"));
                StoredSlotB1AuxSample {
                    slot_idx,
                    slot_a_bytes: slot_target.slot_a_bytes,
                    preimage_b1_bytes: preimage_b1_chunks,
                }
            })
            .collect();
        ComputedSlotB1Batch { samples }
    }

    fn prepare_slot_batch_gpu(
        &self,
        slot_secret_mats: &[Vec<u8>],
        batch_slot_indices: &[usize],
    ) -> PreparedSlotBatch<M> {
        PreparedSlotBatch {
            batch_slot_indices: batch_slot_indices.to_vec(),
            slot_secret_bytes_by_slot: batch_slot_indices
                .iter()
                .copied()
                .map(|slot_idx| (slot_idx, slot_secret_mats[slot_idx].clone()))
                .collect(),
            _m: std::marker::PhantomData,
        }
    }

    fn finalize_slot_b0_batch_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        computed_batch: ComputedSlotB0Batch,
    ) {
        computed_batch.samples.into_par_iter().for_each(|sample| {
            for (chunk_idx, bytes) in sample.preimage_b0_bytes.into_iter().enumerate() {
                Self::store_bytes_checkpoint(
                    bytes,
                    &column_chunk_id_prefix(
                        &self.slot_preimage_b0_id_prefix(params, sample.slot_idx),
                        chunk_idx,
                    ),
                );
            }
        });
    }

    fn finalize_slot_b1_batch_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        computed_batch: ComputedSlotB1Batch,
        slot_a_bytes_by_slot: &mut [Vec<u8>],
    ) {
        let samples = computed_batch.samples;
        samples.par_iter().for_each(|sample| {
            let slot_idx = sample.slot_idx;
            Self::store_bytes_checkpoint(
                sample.slot_a_bytes.clone(),
                &self.slot_a_id_prefix(params, slot_idx),
            );
            for (chunk_idx, bytes) in sample.preimage_b1_bytes.iter().cloned().enumerate() {
                Self::store_bytes_checkpoint(
                    bytes,
                    &column_chunk_id_prefix(
                        &self.slot_preimage_b1_id_prefix(params, slot_idx),
                        chunk_idx,
                    ),
                );
            }
        });
        for sample in samples {
            slot_a_bytes_by_slot[sample.slot_idx] = sample.slot_a_bytes;
        }
    }

    pub(crate) fn sample_slot_b0_batches_gpu_pipelined(
        &self,
        params: &<M::P as Poly>::Params,
        shared_by_device: &[GpuSlotAuxB0DeviceShared<M, TS::Trapdoor>],
        slot_secret_mats: &[Vec<u8>],
        slot_parallelism: usize,
    ) {
        let chunks = (0..self.num_slots)
            .collect::<Vec<_>>()
            .chunks(slot_parallelism.max(1))
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        if chunks.is_empty() {
            return;
        }

        thread::scope(|scope| {
            let (prepared_tx, prepared_rx) = sync_channel::<PreparedSlotBatch<M>>(1);
            let (computed_tx, computed_rx) = sync_channel::<ComputedSlotB0Batch>(1);

            let load_handle = scope.spawn(move || {
                for batch_slot_indices in chunks {
                    let prepared_batch =
                        self.prepare_slot_batch_gpu(slot_secret_mats, &batch_slot_indices);
                    prepared_tx.send(prepared_batch).expect("slot pipeline compute stage dropped");
                }
            });
            let compute_handle = scope.spawn(move || {
                while let Some(prepared_batch) = recv_next(&prepared_rx, "slot-load") {
                    let computed_batch =
                        self.sample_slot_b0_batch_gpu(shared_by_device, prepared_batch);
                    computed_tx.send(computed_batch).expect("slot pipeline store stage dropped");
                }
            });
            let store_handle = scope.spawn(move || {
                while let Some(computed_batch) = recv_next(&computed_rx, "slot-compute") {
                    self.finalize_slot_b0_batch_gpu(params, computed_batch);
                }
            });

            load_handle.join().expect("slot pipeline load stage panicked");
            compute_handle.join().expect("slot pipeline compute stage panicked");
            store_handle.join().expect("slot pipeline store stage panicked");
        });
    }

    pub(crate) fn sample_slot_b1_batches_gpu_pipelined(
        &self,
        params: &<M::P as Poly>::Params,
        shared_by_device: &[GpuSlotAuxB1DeviceShared<M, TS::Trapdoor>],
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
            let (computed_tx, computed_rx) = sync_channel::<ComputedSlotB1Batch>(1);

            let load_handle = scope.spawn(move || {
                for batch_slot_indices in chunks {
                    let prepared_batch =
                        self.prepare_slot_batch_gpu(slot_secret_mats, &batch_slot_indices);
                    prepared_tx.send(prepared_batch).expect("slot pipeline compute stage dropped");
                }
            });
            let compute_handle = scope.spawn(move || {
                while let Some(prepared_batch) = recv_next(&prepared_rx, "slot-load") {
                    let computed_batch =
                        self.sample_slot_b1_batch_gpu(shared_by_device, prepared_batch);
                    computed_tx.send(computed_batch).expect("slot pipeline store stage dropped");
                }
            });
            let store_handle = scope.spawn(move || {
                while let Some(computed_batch) = recv_next(&computed_rx, "slot-compute") {
                    self.finalize_slot_b1_batch_gpu(
                        params,
                        computed_batch,
                        slot_a_bytes_by_slot_ref,
                    );
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
        shared_by_device: &[GpuSlotGateDeviceShared<M, TS::Trapdoor>],
        gate_id: GateId,
        state: &BggPublicKeySTGateState,
        prepared_batch: PreparedGateBatch<M>,
    ) -> ComputedGateBatch {
        let PreparedGateBatch {
            slot_chunk, slot_secret_bytes_by_slot, slot_a_bytes_by_slot, ..
        } = prepared_batch;
        let trap_sampler = TS::new(&shared_by_device[0].params, self.trapdoor_sigma);
        let m_g = self.secret_size * shared_by_device[0].params.modulus_digits();
        let chunk_count = column_chunk_count(m_g);
        let mut gate_preimages = slot_chunk
            .iter()
            .map(|(dst_slot, _)| (*dst_slot, vec![Vec::new(); chunk_count]))
            .collect::<HashMap<usize, Vec<Vec<u8>>>>();
        let tasks = slot_chunk
            .iter()
            .copied()
            .flat_map(|slot_entry| (0..chunk_count).map(move |chunk_idx| (slot_entry, chunk_idx)))
            .collect::<Vec<_>>();
        for wave in tasks.chunks(shared_by_device.len().max(1)) {
            let requests = wave
                .par_iter()
                .enumerate()
                .map(|(device_idx, ((dst_slot, (src_slot_u32, scalar)), chunk_idx))| {
                    let shared = &shared_by_device[device_idx];
                    let a_in = M::from_compact_bytes(&shared.params, &state.input_pubkey_bytes);
                    let src_slot = usize::try_from(*src_slot_u32)
                        .expect("source slot index must fit in usize");
                    assert!(
                        src_slot < self.num_slots,
                        "source slot index {} out of range for num_slots {}",
                        src_slot,
                        self.num_slots
                    );
                    let s_j = M::from_compact_bytes(
                        &shared.params,
                        slot_secret_bytes_by_slot.get(dst_slot).unwrap_or_else(|| {
                            panic!("missing dst slot secret bytes for dst_slot={dst_slot}")
                        }),
                    );
                    let s_i = M::from_compact_bytes(
                        &shared.params,
                        slot_secret_bytes_by_slot.get(&src_slot).unwrap_or_else(|| {
                            panic!("missing src slot secret bytes for src_slot={src_slot}")
                        }),
                    );
                    let a_j = M::from_compact_bytes(
                        &shared.params,
                        slot_a_bytes_by_slot.get(dst_slot).unwrap_or_else(|| {
                            panic!("missing dst slot a bytes for dst_slot={dst_slot}")
                        }),
                    );
                    let lhs_input = s_i * &a_in;
                    let scalar_poly = scalar
                        .map(|value| M::P::from_usize_to_constant(&shared.params, value as usize));
                    let target_chunk = self.build_gate_target_chunk_gpu(
                        shared,
                        gate_id,
                        &lhs_input,
                        &s_j,
                        &a_j,
                        scalar_poly.as_ref(),
                        *chunk_idx,
                    );
                    GpuPreimageRequest {
                        entry_idx: device_idx,
                        params: &shared.params,
                        trapdoor: &shared.b0_trapdoor,
                        public_matrix: &shared.b0_matrix,
                        target: target_chunk,
                    }
                })
                .collect::<Vec<_>>();
            let mut preimages_by_device = trap_sampler
                .preimage_batched_sharded(requests)
                .into_iter()
                .collect::<HashMap<usize, M>>();
            for (device_idx, ((dst_slot, _), chunk_idx)) in wave.iter().enumerate() {
                let preimage_chunk = preimages_by_device.remove(&device_idx).unwrap_or_else(|| {
                    panic!(
                        "missing gpu gate preimage for dst_slot={}, chunk_idx={}",
                        dst_slot, chunk_idx
                    )
                });
                gate_preimages.get_mut(dst_slot).unwrap_or_else(|| {
                    panic!("missing gpu gate preimage vec for dst_slot={dst_slot}")
                })[*chunk_idx] = preimage_chunk.to_compact_bytes();
            }
        }

        let gate_preimages = slot_chunk
            .iter()
            .map(|(dst_slot, _)| {
                let preimages = gate_preimages
                    .remove(dst_slot)
                    .unwrap_or_else(|| panic!("missing gpu gate preimage for dst_slot={dst_slot}"));
                (*dst_slot, preimages)
            })
            .collect();
        ComputedGateBatch { gate_preimages }
    }

    fn prepare_gate_batch_gpu(
        &self,
        slot_secret_mats: &[Vec<u8>],
        slot_a_bytes_by_slot: &[Vec<u8>],
        slot_chunk: &[(usize, (u32, Option<u32>))],
    ) -> PreparedGateBatch<M> {
        let mut slot_secret_bytes_by_slot = HashMap::<usize, Vec<u8>>::new();
        let mut slot_a_bytes_by_slot_map = HashMap::<usize, Vec<u8>>::new();
        for (dst_slot, (src_slot_u32, _)) in slot_chunk.iter().copied() {
            let src_slot =
                usize::try_from(src_slot_u32).expect("source slot index must fit in usize");
            slot_secret_bytes_by_slot
                .entry(dst_slot)
                .or_insert_with(|| slot_secret_mats[dst_slot].clone());
            slot_secret_bytes_by_slot
                .entry(src_slot)
                .or_insert_with(|| slot_secret_mats[src_slot].clone());
            slot_a_bytes_by_slot_map
                .entry(dst_slot)
                .or_insert_with(|| slot_a_bytes_by_slot[dst_slot].clone());
        }

        PreparedGateBatch {
            slot_chunk: slot_chunk.to_vec(),
            slot_secret_bytes_by_slot,
            slot_a_bytes_by_slot: slot_a_bytes_by_slot_map,
            _m: std::marker::PhantomData,
        }
    }

    fn finalize_gate_batch_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        computed_batch: ComputedGateBatch,
    ) {
        computed_batch.gate_preimages.into_par_iter().for_each(|(dst_slot, preimage_bytes)| {
            for (chunk_idx, bytes) in preimage_bytes.into_iter().enumerate() {
                Self::store_bytes_checkpoint(
                    bytes,
                    &column_chunk_id_prefix(
                        &self.gate_preimage_id_prefix(params, gate_id, dst_slot),
                        chunk_idx,
                    ),
                );
            }
        });
    }

    pub(crate) fn sample_gate_batches_gpu_pipelined(
        &self,
        params: &<M::P as Poly>::Params,
        shared_by_device: &[GpuSlotGateDeviceShared<M, TS::Trapdoor>],
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
            let (computed_tx, computed_rx) = sync_channel::<ComputedGateBatch>(1);

            let load_handle = scope.spawn(move || {
                for slot_chunk in chunks {
                    let prepared_batch = self.prepare_gate_batch_gpu(
                        slot_secret_mats,
                        slot_a_bytes_by_slot,
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
        let b1_size =
            self.secret_size.checked_mul(2).expect("slot-transfer benchmark b1 size overflow");
        let total_chunk_count =
            crate::slot_transfer::bgg_pubkey::slot_aux_chunk_count::<M>(params, self.secret_size);
        let benchmark_params = self.benchmark_gpu_params(params);
        let slot_secret_mat = US::new().sample_uniform(
            &benchmark_params,
            self.secret_size,
            self.secret_size,
            DistType::TernaryDist,
        );
        let m_g = self.secret_size * benchmark_params.modulus_digits();
        let b1_chunk_count = column_chunk_count(m_g);
        let slot_a_bytes = HS::new()
            .sample_hash(
                &benchmark_params,
                self.hash_key,
                "slot_transfer_slot_a_0".to_string(),
                self.secret_size,
                m_g,
                DistType::FinRingDist,
            )
            .into_compact_bytes();

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (unused_b1_trapdoor, b1_matrix_for_b0) = trap_sampler.trapdoor(params, b1_size);
        let b1_matrix_bytes_for_b0 = b1_matrix_for_b0.to_compact_bytes();
        drop(b1_matrix_for_b0);
        drop(unused_b1_trapdoor);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let b0_shared = self.prepare_gpu_slot_aux_b0_bench_shared(
            params,
            &b0_matrix,
            &b0_trapdoor,
            &b1_matrix_bytes_for_b0,
        );
        let b0_chunk_count = column_chunk_count(b0_shared.b1_matrix.col_size());
        let trap_sampler_b0 = TS::new(&b0_shared.params, self.trapdoor_sigma);
        let start = Instant::now();
        let b0_target_chunk =
            self.build_slot_aux_b0_target_chunk_gpu(&b0_shared, &slot_secret_mat, 0);
        let b0_chunk_bytes = trap_sampler_b0
            .preimage(
                &b0_shared.params,
                &b0_shared.b0_trapdoor,
                &b0_shared.b0_matrix,
                &b0_target_chunk,
            )
            .into_compact_bytes();
        let b0_latency = start.elapsed().as_secs_f64();
        drop(b0_target_chunk);
        drop(b0_shared);
        drop(b0_matrix);
        drop(b0_trapdoor);

        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, b1_size);
        let b1_shared = self.prepare_gpu_slot_aux_b1_bench_shared(params, &b1_matrix, &b1_trapdoor);
        let trap_sampler_b1 = TS::new(&b1_shared.params, self.trapdoor_sigma);
        let start = Instant::now();
        let b1_target_chunk =
            self.build_slot_aux_b1_target_chunk_gpu(&b1_shared, 0, &slot_secret_mat, 0);
        let b1_chunk_bytes = trap_sampler_b1
            .preimage(
                &b1_shared.params,
                &b1_shared.b1_trapdoor,
                &b1_shared.b1_matrix,
                &b1_target_chunk,
            )
            .into_compact_bytes();
        let b1_latency = start.elapsed().as_secs_f64();
        drop(b1_target_chunk);
        drop(b1_shared);
        drop(b1_matrix);
        drop(b1_trapdoor);

        let b0_estimate =
            SampleAuxBenchEstimate::from_chunk(b0_latency, b0_chunk_count, b0_chunk_bytes.len());
        let b1_estimate = SampleAuxBenchEstimate::from_chunk_with_base(
            b1_latency,
            b1_chunk_count,
            b1_chunk_bytes.len(),
            BigUint::from(slot_a_bytes.len()),
        );
        debug_assert_eq!(
            total_chunk_count,
            b0_chunk_count + b1_chunk_count,
            "slot auxiliary chunk-count decomposition mismatch"
        );
        SampleAuxBenchEstimate {
            total_time: b0_estimate.total_time + b1_estimate.total_time,
            latency: b0_estimate.latency + b1_estimate.latency,
            compact_bytes: b0_estimate.compact_bytes + b1_estimate.compact_bytes,
        }
    }

    fn sample_aux_matrices_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.secret_size);
        let total_chunk_count =
            crate::slot_transfer::bgg_pubkey::slot_gate_chunk_count::<M>(params, self.secret_size);
        let gate_shared = self.prepare_gpu_slot_gate_bench_shared(params, &b0_matrix, &b0_trapdoor);
        let m_g = self.secret_size * gate_shared.params.modulus_digits();
        let slot_secret_bytes = US::new()
            .sample_uniform(
                &gate_shared.params,
                self.secret_size,
                self.secret_size,
                DistType::TernaryDist,
            )
            .into_compact_bytes();
        let slot_a_bytes = HS::new()
            .sample_hash(
                &gate_shared.params,
                self.hash_key,
                "slot_transfer_slot_a_0".to_string(),
                self.secret_size,
                m_g,
                DistType::FinRingDist,
            )
            .into_compact_bytes();
        let state = BggPublicKeySTGateState {
            input_pubkey_bytes: M::gadget_matrix(&gate_shared.params, self.secret_size)
                .into_compact_bytes(),
            src_slots: vec![(0u32, None)],
        };
        let trap_sampler_gate = TS::new(&gate_shared.params, self.trapdoor_sigma);
        let start = Instant::now();
        let a_in = M::from_compact_bytes(&gate_shared.params, &state.input_pubkey_bytes);
        let s_j = M::from_compact_bytes(&gate_shared.params, &slot_secret_bytes);
        let s_i = M::from_compact_bytes(&gate_shared.params, &slot_secret_bytes);
        let a_j = M::from_compact_bytes(&gate_shared.params, &slot_a_bytes);
        let lhs_input = s_i * &a_in;
        let target_chunk = self.build_gate_target_chunk_gpu(
            &gate_shared,
            GateId(0),
            &lhs_input,
            &s_j,
            &a_j,
            None,
            0,
        );
        let preimage_chunk = trap_sampler_gate.preimage(
            &gate_shared.params,
            &gate_shared.b0_trapdoor,
            &gate_shared.b0_matrix,
            &target_chunk,
        );
        let chunk_bytes = preimage_chunk.into_compact_bytes();
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate::from_chunk(elapsed, total_chunk_count, chunk_bytes.len())
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
        slot_transfer::bgg_pubkey::{BggPublicKeySTEvaluator, trapdoor_public_column_count},
        storage::{
            read::{read_bytes_from_multi_batch, read_matrix_from_multi_batch},
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

    fn concat_column_chunks<M>(chunks: Vec<M>) -> M
    where
        M: PolyMatrix,
    {
        let mut chunk_iter = chunks.into_iter();
        let first = chunk_iter.next().expect("column chunk list must be non-empty");
        first.concat_columns_owned(chunk_iter.collect())
    }

    fn read_matrix_from_column_chunks<M>(
        params: &<M::P as Poly>::Params,
        dir: &Path,
        id_prefix: &str,
        total_cols: usize,
    ) -> M
    where
        M: PolyMatrix,
    {
        let chunk_count = super::column_chunk_count(total_cols);
        let mut chunks = Vec::with_capacity(chunk_count);
        for chunk_idx in 0..chunk_count {
            let (_, expected_cols) = super::column_chunk_bounds(total_cols, chunk_idx);
            let chunk_prefix = super::column_chunk_id_prefix(id_prefix, chunk_idx);
            let chunk = if let Some(matrix) =
                read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, 0)
            {
                matrix
            } else {
                let bytes = read_bytes_from_multi_batch(dir, &chunk_prefix, 0)
                    .unwrap_or_else(|| {
                        panic!(
                            "missing slot-transfer checkpoint bytes for {id_prefix} chunk {chunk_idx}"
                        )
                    });
                M::from_compact_bytes(params, &bytes)
            };
            assert_eq!(
                chunk.col_size(),
                expected_cols,
                "slot-transfer checkpoint chunk {} must have {} columns for {}",
                chunk_idx,
                expected_cols,
                id_prefix
            );
            chunks.push(chunk);
        }
        concat_column_chunks(chunks)
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

                let slot_preimage_b0 = read_matrix_from_column_chunks::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_slot_preimage_b0_{slot_idx}"),
                    trapdoor_public_column_count::<GpuDCRTPolyMatrix>(&params, secret_size * 2),
                );
                let slot_preimage_b1 = read_matrix_from_column_chunks::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!("{checkpoint_prefix}_slot_preimage_b1_{slot_idx}"),
                    m_g,
                );

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
                let gate_preimage = read_matrix_from_column_chunks::<GpuDCRTPolyMatrix>(
                    &params,
                    dir,
                    &format!(
                        "{checkpoint_prefix}_gate_preimage_{}_dst{}",
                        transferred_gate, dst_slot
                    ),
                    m_g,
                );

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
