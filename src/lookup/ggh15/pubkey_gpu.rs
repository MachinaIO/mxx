use super::*;
use crate::{
    bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate},
    poly::dcrt::gpu::detected_gpu_device_ids,
    sampler::trapdoor::GpuPreimageRequest,
};
use std::ops::Deref;

enum DeviceReplica<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> DeviceReplica<'_, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<T> Deref for DeviceReplica<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

fn source_device_first(mut device_ids: Vec<i32>, source_device_id: i32) -> Vec<i32> {
    if let Some(source_pos) = device_ids.iter().position(|&device_id| device_id == source_device_id)
    {
        device_ids.swap(0, source_pos);
    }
    device_ids
}

pub(super) struct GpuLutBaseDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: DeviceReplica<'a, T>,
    b1_matrix: DeviceReplica<'a, M>,
}

struct GpuLutDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: &'a T,
    b1_matrix: &'a M,
    w_block_identity: M,
    w_block_gy: M,
    w_block_v: M,
    gadget_matrix: M,
}

pub(super) struct GpuGateBaseDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: DeviceReplica<'a, T>,
    b0_matrix: DeviceReplica<'a, M>,
    b1_matrix: DeviceReplica<'a, M>,
}

struct GpuGateDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: &'a T,
    b0_matrix: &'a M,
    b1_matrix: &'a M,
    w_block_identity: DeviceReplica<'a, M>,
    w_block_gy: DeviceReplica<'a, M>,
    w_block_v: DeviceReplica<'a, M>,
}

#[cfg(feature = "gpu")]
fn round_robin_device_slot(logical_idx: usize, device_count: usize) -> usize {
    assert!(device_count > 0, "round-robin device selection requires at least one device");
    logical_idx % device_count
}

#[cfg(feature = "gpu")]
fn shared_for_logical_idx<T>(shared: &[T], logical_idx: usize) -> &T {
    let device_slot = round_robin_device_slot(logical_idx, shared.len());
    &shared[device_slot]
}

#[cfg(feature = "gpu")]
fn prepare_lookup_buffers(jobs: Vec<CompactBytesJob>) -> Vec<BatchLookupBuffer> {
    jobs.into_par_iter().map(CompactBytesJob::into_lookup_buffer).collect()
}

#[cfg(feature = "gpu")]
fn store_lookup_buffers(buffers: Vec<BatchLookupBuffer>) {
    buffers.into_par_iter().for_each(|buffer| {
        let _ = add_lookup_buffer(buffer);
    });
}

#[cfg(feature = "gpu")]
fn pipeline_lookup_stage<F>(
    pending_store_buffers: &mut Option<Vec<BatchLookupBuffer>>,
    build_current_buffers: F,
) where
    F: FnOnce() -> Vec<BatchLookupBuffer> + Send,
{
    let current_buffers = if let Some(previous_buffers) = pending_store_buffers.take() {
        let (_, current_buffers) =
            rayon::join(|| store_lookup_buffers(previous_buffers), build_current_buffers);
        current_buffers
    } else {
        build_current_buffers()
    };
    *pending_store_buffers = Some(current_buffers);
}

impl<M, US, HS, TS> GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    #[cfg(feature = "gpu")]
    pub(super) fn prepare_gpu_lut_base_device_shared<'a>(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b1_trapdoor: &'a TS::Trapdoor,
        b1_matrix: &'a M,
    ) -> Vec<GpuLutBaseDeviceShared<'a, M, TS::Trapdoor>> {
        let source_device_id = params
            .device_ids()
            .into_iter()
            .next()
            .expect("gpu params must include at least one device id");
        let device_ids = source_device_first(detected_gpu_device_ids(), source_device_id);
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for gpu sample_lut_preimages"
        );

        let needs_cross_device_copy =
            device_ids.iter().any(|&device_id| device_id != source_device_id);
        let b1_trapdoor_bytes = needs_cross_device_copy.then(|| TS::trapdoor_to_bytes(b1_trapdoor));
        let b1_matrix_bytes = needs_cross_device_copy.then(|| b1_matrix.to_compact_bytes());

        device_ids
            .into_iter()
            .map(|device_id| {
                let local_params = if device_id == source_device_id {
                    params.clone()
                } else {
                    params.params_for_device(device_id)
                };
                let (local_trapdoor, local_b1_matrix) = if device_id == source_device_id {
                    (DeviceReplica::Borrowed(b1_trapdoor), DeviceReplica::Borrowed(b1_matrix))
                } else {
                    let trapdoor_bytes = b1_trapdoor_bytes.as_ref().expect(
                        "cross-device LUT trapdoor copy requested without serialized source bytes",
                    );
                    let matrix_bytes = b1_matrix_bytes.as_ref().expect(
                        "cross-device LUT matrix copy requested without serialized source bytes",
                    );
                    (
                        DeviceReplica::Owned(
                            TS::trapdoor_from_bytes(&local_params, trapdoor_bytes).expect(
                                "failed to deserialize trapdoor for preloaded LUT device resources",
                            ),
                        ),
                        DeviceReplica::Owned(M::from_compact_bytes(&local_params, matrix_bytes)),
                    )
                };
                GpuLutBaseDeviceShared {
                    params: local_params,
                    trapdoor: local_trapdoor,
                    b1_matrix: local_b1_matrix,
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn prepare_gpu_lut_device_shared<'a>(
        &self,
        lut_id: usize,
        base_shared: &'a [GpuLutBaseDeviceShared<'_, M, TS::Trapdoor>],
    ) -> Vec<GpuLutDeviceShared<'a, M, TS::Trapdoor>> {
        base_shared
            .iter()
            .map(|base| {
                let w_block_identity = self.derive_w_block_identity(&base.params, lut_id);
                let w_block_gy = self.derive_w_block_gy(&base.params, lut_id);
                let w_block_v = self.derive_w_block_v(&base.params, lut_id);
                let gadget_matrix = M::gadget_matrix(&base.params, self.d);
                GpuLutDeviceShared {
                    params: &base.params,
                    trapdoor: base.trapdoor.as_ref(),
                    b1_matrix: base.b1_matrix.as_ref(),
                    w_block_identity,
                    w_block_gy,
                    w_block_v,
                    gadget_matrix,
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    pub(super) fn prepare_gpu_gate_base_device_shared<'a>(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b0_trapdoor: &'a TS::Trapdoor,
        b0_matrix: &'a M,
        b1_matrix: &'a M,
    ) -> Vec<GpuGateBaseDeviceShared<'a, M, TS::Trapdoor>> {
        let source_device_id = params
            .device_ids()
            .into_iter()
            .next()
            .expect("gpu params must include at least one device id");
        let device_ids = source_device_first(detected_gpu_device_ids(), source_device_id);
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for gpu gate preimage sampling"
        );

        let needs_cross_device_copy =
            device_ids.iter().any(|&device_id| device_id != source_device_id);
        let b0_trapdoor_bytes = needs_cross_device_copy.then(|| TS::trapdoor_to_bytes(b0_trapdoor));
        let b0_matrix_bytes = needs_cross_device_copy.then(|| b0_matrix.to_compact_bytes());
        let b1_matrix_bytes = needs_cross_device_copy.then(|| b1_matrix.to_compact_bytes());

        device_ids
            .into_iter()
            .map(|device_id| {
                let local_params =
                    if device_id == source_device_id { params.clone() } else { params.params_for_device(device_id) };
                let (local_b0_trapdoor, local_b0_matrix, local_b1_matrix) =
                    if device_id == source_device_id {
                        (
                            DeviceReplica::Borrowed(b0_trapdoor),
                            DeviceReplica::Borrowed(b0_matrix),
                            DeviceReplica::Borrowed(b1_matrix),
                        )
                    } else {
                        let trapdoor_bytes = b0_trapdoor_bytes.as_ref().expect(
                            "cross-device gate trapdoor copy requested without serialized source bytes",
                        );
                        let b0_bytes = b0_matrix_bytes.as_ref().expect(
                            "cross-device gate b0 copy requested without serialized source bytes",
                        );
                        let b1_bytes = b1_matrix_bytes.as_ref().expect(
                            "cross-device gate b1 copy requested without serialized source bytes",
                        );
                        (
                            DeviceReplica::Owned(
                                TS::trapdoor_from_bytes(&local_params, trapdoor_bytes).expect(
                                    "failed to deserialize b0 trapdoor for preloaded gate device resources",
                                ),
                            ),
                            DeviceReplica::Owned(M::from_compact_bytes(&local_params, b0_bytes)),
                            DeviceReplica::Owned(M::from_compact_bytes(&local_params, b1_bytes)),
                        )
                    };
                GpuGateBaseDeviceShared {
                    device_id,
                    params: local_params,
                    b0_trapdoor: local_b0_trapdoor,
                    b0_matrix: local_b0_matrix,
                    b1_matrix: local_b1_matrix,
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn copy_matrix_to_gpu_gate_devices<'a>(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        matrix: &'a M,
        base_shared: &[GpuGateBaseDeviceShared<'_, M, TS::Trapdoor>],
    ) -> Vec<DeviceReplica<'a, M>> {
        let source_device_id = params
            .device_ids()
            .into_iter()
            .next()
            .expect("gpu params must include at least one device id");
        let needs_cross_device_copy =
            base_shared.iter().any(|device_shared| device_shared.device_id != source_device_id);
        let bytes = needs_cross_device_copy.then(|| matrix.to_compact_bytes());
        base_shared
            .iter()
            .map(|device_shared| {
                if device_shared.device_id == source_device_id {
                    DeviceReplica::Borrowed(matrix)
                } else {
                    DeviceReplica::Owned(M::from_compact_bytes(
                        &device_shared.params,
                        bytes.as_ref().expect(
                            "cross-device gate replica requested without serialized source bytes",
                        ),
                    ))
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn prepare_gpu_gate_device_shared<'a>(
        &self,
        base_shared: &'a [GpuGateBaseDeviceShared<'_, M, TS::Trapdoor>],
        w_block_identity_by_device: Vec<DeviceReplica<'a, M>>,
        w_block_gy_by_device: Vec<DeviceReplica<'a, M>>,
        w_block_v_by_device: Vec<DeviceReplica<'a, M>>,
    ) -> Vec<GpuGateDeviceShared<'a, M, TS::Trapdoor>> {
        let len = base_shared.len();
        assert_eq!(
            len,
            w_block_identity_by_device.len(),
            "w_block_identity device replicas must match gate base shared length"
        );
        assert_eq!(
            len,
            w_block_gy_by_device.len(),
            "w_block_gy device replicas must match gate base shared length"
        );
        assert_eq!(
            len,
            w_block_v_by_device.len(),
            "w_block_v device replicas must match gate base shared length"
        );

        let mut w_identity_it = w_block_identity_by_device.into_iter();
        let mut w_gy_it = w_block_gy_by_device.into_iter();
        let mut w_v_it = w_block_v_by_device.into_iter();

        (0..len)
            .map(|idx| {
                let base = &base_shared[idx];
                GpuGateDeviceShared {
                    params: &base.params,
                    b0_trapdoor: base.b0_trapdoor.as_ref(),
                    b0_matrix: base.b0_matrix.as_ref(),
                    b1_matrix: base.b1_matrix.as_ref(),
                    w_block_identity: w_identity_it
                        .next()
                        .expect("missing w_block_identity replica for gate device"),
                    w_block_gy: w_gy_it.next().expect("missing w_block_gy replica for gate device"),
                    w_block_v: w_v_it.next().expect("missing w_block_v replica for gate device"),
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn build_lut_preimage_target_chunk_gpu(
        &self,
        shared_dev: &GpuLutDeviceShared<'_, M, TS::Trapdoor>,
        lut_id: usize,
        idx: usize,
        local_y_poly: &M::P,
        idx_scalar_by_digit: &[M::P],
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m = d * shared_dev.params.modulus_digits();
        let (_, _, crt_depth) = shared_dev.params.to_crt();
        let k_small = shared_dev.params.modulus_digits() / crt_depth;
        let (col_start, col_len) = column_chunk_bounds(m, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = shared_dev.w_block_identity.slice(0, d, col_start, col_end);
        let gy_chunk = (shared_dev.gadget_matrix.slice(0, d, col_start, col_end) *
            local_y_poly.clone())
        .decompose();
        target_chunk.add_in_place(&(shared_dev.w_block_gy.clone() * gy_chunk));

        let v_idx_chunk = HS::new().sample_hash_decomposed_columns(
            shared_dev.params,
            self.hash_key,
            format!("ggh15_lut_v_idx_{}_{}", lut_id, idx),
            d,
            m,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        target_chunk.add_in_place(&(shared_dev.w_block_v.clone() * &v_idx_chunk));

        for small_chunk_idx in 0..k_small {
            let w_vx_chunk = self.derive_w_block_with_tag_columns(
                shared_dev.params,
                lut_id,
                "block_vx",
                m * k_small,
                small_chunk_idx * m,
                m,
            );
            let vx_rhs_chunk = build_small_decomposed_scalar_mul_chunk::<M>(
                shared_dev.params,
                &v_idx_chunk,
                idx_scalar_by_digit,
                small_chunk_idx,
            );
            target_chunk.add_in_place(&(w_vx_chunk * vx_rhs_chunk));
        }

        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn build_gate_stage1_target_chunk_gpu(
        &self,
        shared: &GpuGateDeviceShared<'_, M, TS::Trapdoor>,
        s_g: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let (col_start, col_len) = column_chunk_bounds(shared.b1_matrix.col_size(), chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk = s_g.clone() * &shared.b1_matrix.slice(0, d, col_start, col_end);
        target_chunk.add_in_place(&self.sample_error_matrix(shared.params, d, col_len));
        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn build_gate_stage2_identity_target_chunk_gpu(
        &self,
        shared: &GpuGateDeviceShared<'_, M, TS::Trapdoor>,
        gate_id: GateId,
        s_g: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * shared.params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk =
            s_g.clone() * &shared.w_block_identity.as_ref().slice(0, d, col_start, col_end);
        let out_chunk = HS::new().sample_hash_columns(
            shared.params,
            self.hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        target_chunk.add_in_place(&out_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(shared.params, d, col_len));
        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn build_gate_stage3_gy_target_chunk_gpu(
        &self,
        shared: &GpuGateDeviceShared<'_, M, TS::Trapdoor>,
        s_g: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * shared.params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk =
            s_g.clone() * &shared.w_block_gy.as_ref().slice(0, d, col_start, col_end);
        let target_high_chunk = -M::gadget_matrix(shared.params, d).slice(0, d, col_start, col_end);
        target_chunk.add_in_place(&target_high_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(shared.params, d, col_len));
        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn build_gate_stage4_v_target_chunk_gpu(
        &self,
        shared: &GpuGateDeviceShared<'_, M, TS::Trapdoor>,
        gate_id: GateId,
        s_g: &M,
        input_matrix: &M,
        chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * shared.params.modulus_digits();
        let (col_start, col_len) = column_chunk_bounds(m_g, chunk_idx);
        let col_end = col_start + col_len;
        let mut target_chunk =
            s_g.clone() * &shared.w_block_v.as_ref().slice(0, d, col_start, col_end);
        let u_g_decomposed_chunk = HS::new().sample_hash_decomposed_columns(
            shared.params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            col_start,
            col_len,
            DistType::FinRingDist,
        );
        let target_high_chunk = -(input_matrix.clone() * u_g_decomposed_chunk);
        target_chunk.add_in_place(&target_high_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(shared.params, d, col_len));
        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn build_gate_stage5_target_chunk_gpu(
        &self,
        shared: &GpuGateDeviceShared<'_, M, TS::Trapdoor>,
        lut_id: usize,
        s_g: &M,
        u_g_matrix: &M,
        small_chunk_idx: usize,
        col_chunk_idx: usize,
    ) -> M {
        let d = self.d;
        let m_g = d * shared.params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(shared.params);
        let (col_start, col_len) = column_chunk_bounds(m_g, col_chunk_idx);
        let small_chunk_start =
            small_chunk_idx.checked_mul(m_g).expect("stage5 small-chunk start column overflow");
        let target_high_vx_chunk = self.build_stage5_target_high_vx_chunk(
            shared.params,
            u_g_matrix,
            small_chunk_idx,
            col_start,
            col_len,
        );
        let w_block_vx_chunk = self.derive_w_block_with_tag_columns(
            shared.params,
            lut_id,
            "block_vx",
            m_g * k_small,
            small_chunk_start + col_start,
            col_len,
        );
        let mut target_chunk = s_g.clone() * &w_block_vx_chunk;
        target_chunk.add_in_place(&target_high_vx_chunk);
        target_chunk.add_in_place(&self.sample_error_matrix(shared.params, d, col_len));
        target_chunk
    }

    #[cfg(feature = "gpu")]
    fn sample_lut_preimages_gpu(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        lut_aux_id_prefix: &str,
        shared: &[GpuLutDeviceShared<'_, M, TS::Trapdoor>],
        batch: &[(usize, usize, M::P)],
    ) -> Vec<CompactBytesJob> {
        let sample_lut_preimages_start = Instant::now();
        debug!("Sampling LUT preimages started: lut_id={}, batch_size={}", lut_id, batch.len());
        let d = self.d;
        let m = d * params.modulus_digits();
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        debug_assert!(!shared.is_empty(), "gpu shared resources must not be empty");
        debug_assert_eq!(
            shared[0].w_block_identity.col_size(),
            m,
            "w_block_identity columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            shared[0].w_block_gy.col_size(),
            m,
            "w_block_gy columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            shared[0].w_block_v.col_size(),
            m,
            "w_block_v columns must equal d * modulus_digits^2"
        );

        let jobs = {
            if batch.is_empty() {
                Vec::new()
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let row_states = batch
                    .par_iter()
                    .map(|(idx, _device_slot, local_y_poly)| {
                        let idx_poly = M::P::from_usize_to_constant(shared[0].params, *idx);
                        let idx_scalar_by_digit = small_decomposed_scalar_digits::<M>(
                            shared[0].params,
                            &idx_poly,
                            k_small,
                        );
                        (*idx, local_y_poly.clone(), idx_scalar_by_digit)
                    })
                    .collect::<Vec<_>>();
                let chunk_count = column_chunk_count(m);
                let tasks = row_states
                    .iter()
                    .enumerate()
                    .flat_map(|(state_idx, _)| {
                        (0..chunk_count).map(move |chunk_idx| (state_idx, chunk_idx))
                    })
                    .collect::<Vec<_>>();
                let mut jobs = Vec::with_capacity(tasks.len());
                for wave in tasks.chunks(shared.len().max(1)) {
                    let requests = wave
                        .par_iter()
                        .enumerate()
                        .map(|(device_slot, (state_idx, chunk_idx))| {
                            let shared_dev = &shared[device_slot];
                            let (idx, local_y_poly, idx_scalar_by_digit) = &row_states[*state_idx];
                            let target_chunk = self.build_lut_preimage_target_chunk_gpu(
                                shared_dev,
                                lut_id,
                                *idx,
                                local_y_poly,
                                idx_scalar_by_digit,
                                *chunk_idx,
                            );
                            GpuPreimageRequest {
                                entry_idx: device_slot,
                                params: shared_dev.params,
                                trapdoor: shared_dev.trapdoor,
                                public_matrix: shared_dev.b1_matrix,
                                target: target_chunk,
                            }
                        })
                        .collect::<Vec<_>>();
                    let mut preimage_by_device = trap_sampler
                        .preimage_batched_sharded(requests)
                        .into_iter()
                        .collect::<HashMap<usize, M>>();
                    for (device_slot, (state_idx, chunk_idx)) in wave.iter().enumerate() {
                        let (idx, _, _) = &row_states[*state_idx];
                        let lut_aux_id = format!("{lut_aux_id_prefix}_idx{idx}");
                        let preimage_chunk =
                            preimage_by_device.remove(&device_slot).unwrap_or_else(|| {
                                panic!(
                                    "missing LUT preimage chunk for row idx={}, chunk_idx={}",
                                    idx, chunk_idx
                                )
                            });
                        jobs.push(CompactBytesJob::new(
                            column_chunk_id_prefix(&lut_aux_id, *chunk_idx),
                            vec![(0usize, preimage_chunk)],
                        ));
                    }
                }
                jobs
            }
        };

        let sample_lut_preimages_elapsed = sample_lut_preimages_start.elapsed();
        debug!(
            "Finished sampling LUT preimages: lut_id={}, batch_size={}, elapsed={}",
            lut_id,
            batch.len(),
            Self::format_duration(sample_lut_preimages_elapsed)
        );
        jobs
    }

    #[cfg(feature = "gpu")]
    fn sample_gate_preimages_batch_gpu(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        pending: Vec<(GateId, GateState<M>)>,
        shared: &[GpuGateDeviceShared<'_, M, TS::Trapdoor>],
    ) -> u64 {
        if pending.is_empty() {
            return 0;
        }

        let d = self.d;
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        debug_assert!(
            shared.iter().all(|entry| entry.b1_matrix.row_size() == d),
            "gate stage1 expects b1_matrix to have d rows"
        );
        let mut pending_store_buffers: Option<Vec<BatchLookupBuffer>> = None;
        let mut total_compact_bytes = 0u64;

        let stage1_chunk_count = column_chunk_count(shared[0].b1_matrix.col_size());
        let stage_mg_chunk_count = column_chunk_count(m_g);

        let stage_inputs = pending
            .into_par_iter()
            .enumerate()
            .map(|(entry_pos, (gate_id, state))| {
                let shared = shared_for_logical_idx(shared, entry_pos);
                let uniform_sampler = US::new();
                let s_g_bytes = uniform_sampler
                    .sample_uniform(shared.params, d, d, DistType::TernaryDist)
                    .into_compact_bytes();
                (entry_pos, gate_id, state.input_pubkey_bytes, s_g_bytes)
            })
            .collect::<Vec<_>>();

        let stage1_tasks = stage_inputs
            .iter()
            .enumerate()
            .flat_map(|(input_idx, _)| {
                (0..stage1_chunk_count).map(move |chunk_idx| (input_idx, chunk_idx))
            })
            .collect::<Vec<_>>();
        for wave in stage1_tasks.chunks(shared.len().max(1)) {
            let mut stage1_compact_bytes = 0u64;
            pipeline_lookup_stage(&mut pending_store_buffers, || {
                let requests = wave
                    .par_iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let shared = &shared[device_slot];
                        let (_, _, _, s_g_bytes) = &stage_inputs[*input_idx];
                        let s_g = M::from_compact_bytes(shared.params, s_g_bytes);
                        let target_chunk =
                            self.build_gate_stage1_target_chunk_gpu(shared, &s_g, *chunk_idx);
                        GpuPreimageRequest {
                            entry_idx: device_slot,
                            params: shared.params,
                            trapdoor: shared.b0_trapdoor,
                            public_matrix: shared.b0_matrix,
                            target: target_chunk,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut preimages = trap_sampler
                    .preimage_batched_sharded(requests)
                    .into_iter()
                    .collect::<HashMap<usize, M>>();
                let jobs = wave
                    .iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let (_, gate_id, _, _) = &stage_inputs[*input_idx];
                        let preimage_chunk = preimages.remove(&device_slot).unwrap_or_else(|| {
                            panic!(
                                "missing gate stage1 chunk for input_idx={}, chunk_idx={}",
                                input_idx, chunk_idx
                            )
                        });
                        CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate1_id_prefix(params, *gate_id),
                                *chunk_idx,
                            ),
                            vec![(0usize, preimage_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
                stage1_compact_bytes = compact_bytes_job_total(&jobs);
                prepare_lookup_buffers(jobs)
            });
            total_compact_bytes = total_compact_bytes
                .checked_add(stage1_compact_bytes)
                .expect("public LUT gpu stage1 compact_bytes overflowed u64");
        }

        let stage_mg_tasks = stage_inputs
            .iter()
            .enumerate()
            .flat_map(|(input_idx, _)| {
                (0..stage_mg_chunk_count).map(move |chunk_idx| (input_idx, chunk_idx))
            })
            .collect::<Vec<_>>();
        for wave in stage_mg_tasks.chunks(shared.len().max(1)) {
            let mut stage2_compact_bytes = 0u64;
            pipeline_lookup_stage(&mut pending_store_buffers, || {
                let requests = wave
                    .par_iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let shared = &shared[device_slot];
                        let (_, gate_id, _, s_g_bytes) = &stage_inputs[*input_idx];
                        let s_g = M::from_compact_bytes(shared.params, s_g_bytes);
                        let target_chunk = self.build_gate_stage2_identity_target_chunk_gpu(
                            shared, *gate_id, &s_g, *chunk_idx,
                        );
                        GpuPreimageRequest {
                            entry_idx: device_slot,
                            params: shared.params,
                            trapdoor: shared.b0_trapdoor,
                            public_matrix: shared.b0_matrix,
                            target: target_chunk,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut preimages = trap_sampler
                    .preimage_batched_sharded(requests)
                    .into_iter()
                    .collect::<HashMap<usize, M>>();
                let jobs = wave
                    .iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let (_, gate_id, _, _) = &stage_inputs[*input_idx];
                        let preimage_chunk = preimages.remove(&device_slot).unwrap_or_else(|| {
                            panic!(
                                "missing gate stage2 chunk for input_idx={}, chunk_idx={}",
                                input_idx, chunk_idx
                            )
                        });
                        CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate2_identity_id_prefix(params, *gate_id),
                                *chunk_idx,
                            ),
                            vec![(0usize, preimage_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
                stage2_compact_bytes = compact_bytes_job_total(&jobs);
                prepare_lookup_buffers(jobs)
            });
            total_compact_bytes = total_compact_bytes
                .checked_add(stage2_compact_bytes)
                .expect("public LUT gpu stage2 compact_bytes overflowed u64");
        }

        for wave in stage_mg_tasks.chunks(shared.len().max(1)) {
            let mut stage3_compact_bytes = 0u64;
            pipeline_lookup_stage(&mut pending_store_buffers, || {
                let requests = wave
                    .par_iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let shared = &shared[device_slot];
                        let (_, _, _, s_g_bytes) = &stage_inputs[*input_idx];
                        let s_g = M::from_compact_bytes(shared.params, s_g_bytes);
                        let target_chunk =
                            self.build_gate_stage3_gy_target_chunk_gpu(shared, &s_g, *chunk_idx);
                        GpuPreimageRequest {
                            entry_idx: device_slot,
                            params: shared.params,
                            trapdoor: shared.b0_trapdoor,
                            public_matrix: shared.b0_matrix,
                            target: target_chunk,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut preimages = trap_sampler
                    .preimage_batched_sharded(requests)
                    .into_iter()
                    .collect::<HashMap<usize, M>>();
                let jobs = wave
                    .iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let (_, gate_id, _, _) = &stage_inputs[*input_idx];
                        let preimage_chunk = preimages.remove(&device_slot).unwrap_or_else(|| {
                            panic!(
                                "missing gate stage3 chunk for input_idx={}, chunk_idx={}",
                                input_idx, chunk_idx
                            )
                        });
                        CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate2_gy_id_prefix(params, *gate_id),
                                *chunk_idx,
                            ),
                            vec![(0usize, preimage_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
                stage3_compact_bytes = compact_bytes_job_total(&jobs);
                prepare_lookup_buffers(jobs)
            });
            total_compact_bytes = total_compact_bytes
                .checked_add(stage3_compact_bytes)
                .expect("public LUT gpu stage3 compact_bytes overflowed u64");
        }

        for wave in stage_mg_tasks.chunks(shared.len().max(1)) {
            let mut stage4_compact_bytes = 0u64;
            pipeline_lookup_stage(&mut pending_store_buffers, || {
                let requests = wave
                    .par_iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let shared = &shared[device_slot];
                        let (_, gate_id, input_pubkey_bytes, s_g_bytes) = &stage_inputs[*input_idx];
                        let s_g = M::from_compact_bytes(shared.params, s_g_bytes);
                        let input_matrix = M::from_compact_bytes(shared.params, input_pubkey_bytes);
                        let target_chunk = self.build_gate_stage4_v_target_chunk_gpu(
                            shared,
                            *gate_id,
                            &s_g,
                            &input_matrix,
                            *chunk_idx,
                        );
                        GpuPreimageRequest {
                            entry_idx: device_slot,
                            params: shared.params,
                            trapdoor: shared.b0_trapdoor,
                            public_matrix: shared.b0_matrix,
                            target: target_chunk,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut preimages = trap_sampler
                    .preimage_batched_sharded(requests)
                    .into_iter()
                    .collect::<HashMap<usize, M>>();
                let jobs = wave
                    .iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, chunk_idx))| {
                        let (_, gate_id, _, _) = &stage_inputs[*input_idx];
                        let preimage_chunk = preimages.remove(&device_slot).unwrap_or_else(|| {
                            panic!(
                                "missing gate stage4 chunk for input_idx={}, chunk_idx={}",
                                input_idx, chunk_idx
                            )
                        });
                        CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate2_v_id_prefix(params, *gate_id),
                                *chunk_idx,
                            ),
                            vec![(0usize, preimage_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
                stage4_compact_bytes = compact_bytes_job_total(&jobs);
                prepare_lookup_buffers(jobs)
            });
            total_compact_bytes = total_compact_bytes
                .checked_add(stage4_compact_bytes)
                .expect("public LUT gpu stage4 compact_bytes overflowed u64");
        }

        let stage5_inputs = stage_inputs
            .iter()
            .map(|(entry_pos, gate_id, _, s_g_bytes)| {
                let shared = shared_for_logical_idx(shared, *entry_pos);
                let u_g_bytes = HS::new()
                    .sample_hash(
                        shared.params,
                        self.hash_key,
                        format!("ggh15_lut_u_g_matrix_{}", gate_id),
                        d,
                        m_g,
                        DistType::FinRingDist,
                    )
                    .into_compact_bytes();
                (*entry_pos, *gate_id, s_g_bytes.clone(), u_g_bytes)
            })
            .collect::<Vec<_>>();
        let stage5_tasks = stage5_inputs
            .iter()
            .enumerate()
            .flat_map(|(input_idx, _)| {
                (0..k_small).flat_map(move |small_chunk_idx| {
                    (0..stage_mg_chunk_count)
                        .map(move |col_chunk_idx| (input_idx, small_chunk_idx, col_chunk_idx))
                })
            })
            .collect::<Vec<_>>();
        for wave in stage5_tasks.chunks(shared.len().max(1)) {
            let mut stage5_compact_bytes = 0u64;
            pipeline_lookup_stage(&mut pending_store_buffers, || {
                let requests = wave
                    .par_iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, small_chunk_idx, col_chunk_idx))| {
                        let shared = &shared[device_slot];
                        let (_, _gate_id, s_g_bytes, u_g_bytes) = &stage5_inputs[*input_idx];
                        let s_g = M::from_compact_bytes(shared.params, s_g_bytes);
                        let u_g_matrix = M::from_compact_bytes(shared.params, u_g_bytes);
                        let target_gate2_vx_chunk = self.build_gate_stage5_target_chunk_gpu(
                            shared,
                            lut_id,
                            &s_g,
                            &u_g_matrix,
                            *small_chunk_idx,
                            *col_chunk_idx,
                        );
                        GpuPreimageRequest {
                            entry_idx: device_slot,
                            params: shared.params,
                            trapdoor: shared.b0_trapdoor,
                            public_matrix: shared.b0_matrix,
                            target: target_gate2_vx_chunk,
                        }
                    })
                    .collect::<Vec<_>>();
                let mut stage5_preimages = trap_sampler
                    .preimage_batched_sharded(requests)
                    .into_iter()
                    .collect::<HashMap<usize, M>>();
                let stage5_jobs = wave
                    .iter()
                    .enumerate()
                    .map(|(device_slot, (input_idx, small_chunk_idx, col_chunk_idx))| {
                        let (_, gate_id, _, _) = &stage5_inputs[*input_idx];
                        let preimage_gate2_vx_chunk =
                            stage5_preimages.remove(&device_slot).unwrap_or_else(|| {
                                panic!(
                                    "missing gate stage5 preimage for input_idx={}, small_chunk_idx={}, col_chunk_idx={}",
                                    input_idx, small_chunk_idx, col_chunk_idx
                                )
                            });
                        debug!(
                            "Sampled gate preimage 2 (vx part): gate_id={}, lut_id={}, small_chunk_idx={}, col_chunk_idx={}",
                            gate_id,
                            lut_id,
                            small_chunk_idx,
                            col_chunk_idx
                        );
                        CompactBytesJob::new(
                            column_chunk_id_prefix(
                                &self.preimage_gate2_vx_small_id_prefix(
                                    params,
                                    *gate_id,
                                    *small_chunk_idx,
                                ),
                                *col_chunk_idx,
                            ),
                            vec![(0, preimage_gate2_vx_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
                stage5_compact_bytes = compact_bytes_job_total(&stage5_jobs);
                prepare_lookup_buffers(stage5_jobs)
            });
            total_compact_bytes = total_compact_bytes
                .checked_add(stage5_compact_bytes)
                .expect("public LUT gpu stage5 compact_bytes overflowed u64");
        }

        if let Some(previous_buffers) = pending_store_buffers.take() {
            store_lookup_buffers(previous_buffers);
        }
        total_compact_bytes
    }

    pub(super) fn sample_lut_preimage_batches_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        lut_aux_id_prefix: &str,
        pending_input_indices: &[usize],
        chunk_size: usize,
        gpu_lut_base_shared: &[GpuLutBaseDeviceShared<M, TS::Trapdoor>],
        start: &Instant,
        total_lut_rows: usize,
        processed_lut_rows: &mut usize,
    ) {
        let gpu_lut_shared = self.prepare_gpu_lut_device_shared(lut_id, gpu_lut_base_shared);
        let sample_lut_batch = |current_batch: &[(usize, usize, M::P)]| {
            self.sample_lut_preimages_gpu(
                params,
                lut_id,
                lut_aux_id_prefix,
                &gpu_lut_shared,
                current_batch,
            )
        };

        let mut batch: Vec<(usize, usize, M::P)> = Vec::with_capacity(chunk_size);
        let mut pending_store_jobs: Option<Vec<CompactBytesJob>> = None;
        for input_idx in pending_input_indices.iter().copied() {
            let logical_idx = batch.len();
            let device_slot = round_robin_device_slot(logical_idx, gpu_lut_shared.len());
            let device_shared = &gpu_lut_shared[device_slot];
            let input_u64 = u64::try_from(input_idx).expect("LUT input index must fit in u64");
            let (idx, y_elem) = plt
                .get(device_shared.params, input_u64)
                .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", input_idx));
            let idx_usize = usize::try_from(idx).expect("LUT row index must fit in usize");
            let y_poly = M::P::from_elem_to_constant(device_shared.params, &y_elem);
            batch.push((idx_usize, device_slot, y_poly));

            if batch.len() >= chunk_size {
                let current_batch = std::mem::take(&mut batch);
                let lut_preimage_batch_start = Instant::now();
                let sampled_jobs = if let Some(previous_jobs) = pending_store_jobs.take() {
                    let (_, jobs) = rayon::join(
                        || {
                            let wait_start = Instant::now();
                            previous_jobs
                                .into_par_iter()
                                .for_each(CompactBytesJob::wait_then_store);
                            debug!(
                                "Previous batch store completed in {}",
                                Self::format_duration(wait_start.elapsed())
                            );
                        },
                        || sample_lut_batch(&current_batch),
                    );
                    jobs
                } else {
                    sample_lut_batch(&current_batch)
                };
                let lut_preimage_batch_elapsed = lut_preimage_batch_start.elapsed();
                debug!(
                    "Sampled LUT preimages: lut_id={}, batch_size={}, elapsed={}",
                    lut_id,
                    current_batch.len(),
                    Self::format_duration(lut_preimage_batch_elapsed)
                );
                *processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                let pct = if total_lut_rows == 0 {
                    100.0
                } else {
                    (*processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                };
                debug!(
                    "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                    *processed_lut_rows,
                    total_lut_rows,
                    Self::format_duration(start.elapsed())
                );
                pending_store_jobs = Some(sampled_jobs);
            }
        }
        if !batch.is_empty() {
            let current_batch = std::mem::take(&mut batch);
            let lut_preimage_batch_start = Instant::now();
            let sampled_jobs = if let Some(previous_jobs) = pending_store_jobs.take() {
                let (_, jobs) = rayon::join(
                    || {
                        previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                    },
                    || sample_lut_batch(&current_batch),
                );
                jobs
            } else {
                sample_lut_batch(&current_batch)
            };
            let lut_preimage_batch_elapsed = lut_preimage_batch_start.elapsed();
            debug!(
                "Sampled LUT preimages: lut_id={}, batch_size={}, elapsed={}",
                lut_id,
                current_batch.len(),
                Self::format_duration(lut_preimage_batch_elapsed)
            );
            pending_store_jobs = Some(sampled_jobs);
            *processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
            let pct = if total_lut_rows == 0 {
                100.0
            } else {
                (*processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
            };
            debug!(
                "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                *processed_lut_rows,
                total_lut_rows,
                Self::format_duration(start.elapsed())
            );
        }
        if let Some(previous_jobs) = pending_store_jobs.take() {
            previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
        }
    }

    pub(super) fn process_gate_group_gpu(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        gates: &mut Vec<(GateId, GateState<M>)>,
        chunk_size: usize,
        gpu_gate_base_shared: &[GpuGateBaseDeviceShared<M, TS::Trapdoor>],
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
        total_gate_count: usize,
        total_gates: &mut usize,
        start: &Instant,
    ) {
        let w_block_identity_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, w_block_identity, gpu_gate_base_shared);
        let w_block_gy_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, w_block_gy, gpu_gate_base_shared);
        let w_block_v_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, w_block_v, gpu_gate_base_shared);
        let gpu_gate_shared = self.prepare_gpu_gate_device_shared(
            gpu_gate_base_shared,
            w_block_identity_by_device,
            w_block_gy_by_device,
            w_block_v_by_device,
        );

        while !gates.is_empty() {
            let take = gates.len().min(chunk_size);
            let pending: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();

            if !pending.is_empty() {
                let pending_len = pending.len();
                *total_gates = total_gates.saturating_add(pending_len);
                let gate_preimage_batch_start = Instant::now();
                debug!(
                    "Sampling gate preimages batch started: lut_id={}, batch_size={}",
                    lut_id, pending_len
                );
                self.sample_gate_preimages_batch_gpu(params, lut_id, pending, &gpu_gate_shared);
                let gate_preimage_batch_elapsed = gate_preimage_batch_start.elapsed();
                debug!(
                    "Finished sampling gate preimages batch: lut_id={}, batch_size={}, elapsed={}",
                    lut_id,
                    pending_len,
                    Self::format_duration(gate_preimage_batch_elapsed)
                );
            }
            let pct = if total_gate_count == 0 {
                100.0
            } else {
                (*total_gates as f64) * 100.0 / (total_gate_count as f64)
            };
            debug!(
                "Gates processed: {}/{} ({pct:.1}%), elapsed={}",
                *total_gates,
                total_gate_count,
                Self::format_duration(start.elapsed())
            );
        }
    }
}

impl<M, US, HS, TS> PublicLutSampleAuxBenchEstimator for GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    type Params = <M::P as Poly>::Params;

    fn sample_aux_matrices_lut_entry_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let lut_id = 0usize;
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, self.d);
        let gpu_lut_base_shared =
            self.prepare_gpu_lut_base_device_shared(params, &b1_trapdoor, &b1_matrix);
        let gpu_lut_shared = self.prepare_gpu_lut_device_shared(lut_id, &gpu_lut_base_shared);
        let shared = &gpu_lut_shared[0];
        let d = self.d;
        let chunk_count = lut_entry_chunk_count::<M>(params, d);
        let y_poly = M::P::from_usize_to_constant(shared.params, 1usize);
        let k_small = small_gadget_chunk_count::<M>(shared.params);
        let idx_poly = M::P::from_usize_to_constant(shared.params, 0usize);
        let idx_digits = small_decomposed_scalar_digits::<M>(shared.params, &idx_poly, k_small);
        let start = Instant::now();
        let target_chunk =
            self.build_lut_preimage_target_chunk_gpu(shared, lut_id, 0, &y_poly, &idx_digits, 0);
        let preimage_chunk =
            trap_sampler.preimage(shared.params, shared.trapdoor, shared.b1_matrix, &target_chunk);
        let chunk_bytes = preimage_chunk.into_compact_bytes();
        let elapsed = start.elapsed().as_secs_f64();
        SampleAuxBenchEstimate::from_chunk(elapsed, chunk_count, chunk_bytes.len())
    }

    fn sample_aux_matrices_lut_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate {
        let lut_id = 0usize;
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, self.d);
        let (_b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, self.d);
        let gpu_gate_base_shared =
            self.prepare_gpu_gate_base_device_shared(params, &b0_trapdoor, &b0_matrix, &b1_matrix);
        let w_block_identity = self.derive_w_block_identity(params, lut_id);
        let w_block_gy = self.derive_w_block_gy(params, lut_id);
        let w_block_v = self.derive_w_block_v(params, lut_id);
        let w_block_identity_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, &w_block_identity, &gpu_gate_base_shared);
        let w_block_gy_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, &w_block_gy, &gpu_gate_base_shared);
        let w_block_v_by_device =
            self.copy_matrix_to_gpu_gate_devices(params, &w_block_v, &gpu_gate_base_shared);
        let gpu_gate_shared = self.prepare_gpu_gate_device_shared(
            &gpu_gate_base_shared,
            w_block_identity_by_device,
            w_block_gy_by_device,
            w_block_v_by_device,
        );
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let stage1_chunk_count = column_chunk_count(gpu_gate_shared[0].b1_matrix.col_size());
        let stage_mg_chunk_count = column_chunk_count(m_g);
        let k_small = small_gadget_chunk_count::<M>(params);
        let shared = &gpu_gate_shared[0];
        let s_g = US::new().sample_uniform(shared.params, d, d, DistType::TernaryDist);
        let input_matrix = M::gadget_matrix(shared.params, d);

        let start = Instant::now();
        let stage1_target = self.build_gate_stage1_target_chunk_gpu(shared, &s_g, 0);
        let stage1_bytes = trap_sampler
            .preimage(shared.params, shared.b0_trapdoor, shared.b0_matrix, &stage1_target)
            .into_compact_bytes();
        let stage1_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage2_target =
            self.build_gate_stage2_identity_target_chunk_gpu(shared, GateId(0), &s_g, 0);
        let stage2_bytes = trap_sampler
            .preimage(shared.params, shared.b0_trapdoor, shared.b0_matrix, &stage2_target)
            .into_compact_bytes();
        let stage2_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage3_target = self.build_gate_stage3_gy_target_chunk_gpu(shared, &s_g, 0);
        let stage3_bytes = trap_sampler
            .preimage(shared.params, shared.b0_trapdoor, shared.b0_matrix, &stage3_target)
            .into_compact_bytes();
        let stage3_latency = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let stage4_target =
            self.build_gate_stage4_v_target_chunk_gpu(shared, GateId(0), &s_g, &input_matrix, 0);
        let stage4_bytes = trap_sampler
            .preimage(shared.params, shared.b0_trapdoor, shared.b0_matrix, &stage4_target)
            .into_compact_bytes();
        let stage4_latency = start.elapsed().as_secs_f64();

        let u_g_matrix = HS::new().sample_hash(
            shared.params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", GateId(0)),
            d,
            m_g,
            DistType::FinRingDist,
        );
        let start = Instant::now();
        let target_chunk =
            self.build_gate_stage5_target_chunk_gpu(shared, lut_id, &s_g, &u_g_matrix, 0, 0);
        let preimage_chunk = trap_sampler.preimage(
            shared.params,
            shared.b0_trapdoor,
            shared.b0_matrix,
            &target_chunk,
        );
        let chunk_bytes = preimage_chunk.into_compact_bytes();
        let stage5_latency = start.elapsed().as_secs_f64();
        let stage1_estimate = SampleAuxBenchEstimate::from_chunk(
            stage1_latency,
            stage1_chunk_count,
            stage1_bytes.len(),
        );
        let stage2_estimate = SampleAuxBenchEstimate::from_chunk(
            stage2_latency,
            stage_mg_chunk_count,
            stage2_bytes.len(),
        );
        let stage3_estimate = SampleAuxBenchEstimate::from_chunk(
            stage3_latency,
            stage_mg_chunk_count,
            stage3_bytes.len(),
        );
        let stage4_estimate = SampleAuxBenchEstimate::from_chunk(
            stage4_latency,
            stage_mg_chunk_count,
            stage4_bytes.len(),
        );
        let stage5_estimate = SampleAuxBenchEstimate::from_chunk(
            stage5_latency,
            k_small * stage_mg_chunk_count,
            chunk_bytes.len(),
        );
        SampleAuxBenchEstimate {
            total_time: stage1_estimate.total_time +
                stage2_estimate.total_time +
                stage3_estimate.total_time +
                stage4_estimate.total_time +
                stage5_estimate.total_time,
            latency: stage1_estimate.latency +
                stage2_estimate.latency +
                stage3_estimate.latency +
                stage4_estimate.latency +
                stage5_estimate.latency,
            compact_bytes: stage1_estimate.compact_bytes +
                stage2_estimate.compact_bytes +
                stage3_estimate.compact_bytes +
                stage4_estimate.compact_bytes +
                stage5_estimate.compact_bytes,
        }
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{
        build_small_decomposed_scalar_mul_chunk, round_robin_device_slot,
        small_decomposed_scalar_digits, source_device_first,
    };
    use crate::{
        __PAIR, __TestState,
        matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
                params::DCRTPolyParams,
            },
        },
    };
    use num_bigint::BigUint;
    use sequential_test::sequential;

    #[test]
    fn round_robin_device_slot_wraps_after_last_device() {
        assert_eq!(round_robin_device_slot(0, 3), 0);
        assert_eq!(round_robin_device_slot(1, 3), 1);
        assert_eq!(round_robin_device_slot(2, 3), 2);
        assert_eq!(round_robin_device_slot(3, 3), 0);
        assert_eq!(round_robin_device_slot(4, 3), 1);
    }

    #[test]
    fn round_robin_device_slot_balances_logical_work_items() {
        let device_count = 3;
        let work_items = 11;
        let mut counts = vec![0usize; device_count];
        for logical_idx in 0..work_items {
            counts[round_robin_device_slot(logical_idx, device_count)] += 1;
        }

        let min_count = *counts.iter().min().expect("counts must be non-empty");
        let max_count = *counts.iter().max().expect("counts must be non-empty");
        assert!(max_count - min_count <= 1, "counts={counts:?}");
    }

    #[test]
    fn source_device_first_moves_source_to_front() {
        assert_eq!(source_device_first(vec![2, 0, 1], 0), vec![0, 2, 1]);
        assert_eq!(source_device_first(vec![0, 1, 2], 0), vec![0, 1, 2]);
    }

    #[test]
    #[sequential]
    fn gpu_small_decomposed_scalar_mul_chunk_matches_dense_identity_chunk() {
        gpu_device_sync();
        let cpu_params = DCRTPolyParams::new(128, 2, 16, 4);
        let (moduli, _, _) = cpu_params.to_crt();
        let params =
            GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli, cpu_params.base_bits());
        let chunk_count = super::small_gadget_chunk_count::<GpuDCRTPolyMatrix>(&params);
        let nrow = 8usize;
        let ncol = 3usize;
        assert_eq!(nrow % chunk_count, 0, "test requires nrow divisible by chunk_count");

        let scalar = GpuDCRTPoly::from_biguint_to_constant(&params, BigUint::from(13u32));
        let scalar_by_digit =
            small_decomposed_scalar_digits::<GpuDCRTPolyMatrix>(&params, &scalar, chunk_count);
        let source = GpuDCRTPolyMatrix::from_poly_vec(
            &params,
            (0..nrow)
                .map(|row| {
                    (0..ncol)
                        .map(|col| GpuDCRTPoly::from_usize_to_constant(&params, row * 10 + col + 1))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        );

        for chunk_idx in 0..chunk_count {
            let dense_chunk = GpuDCRTPolyMatrix::small_decomposed_identity_chunk_from_scalar(
                &params,
                nrow,
                &scalar,
                chunk_idx,
                chunk_count,
            );
            let expected = dense_chunk * source.clone();
            let actual = build_small_decomposed_scalar_mul_chunk::<GpuDCRTPolyMatrix>(
                &params,
                &source,
                &scalar_by_digit,
                chunk_idx,
            );
            assert_eq!(actual, expected, "chunk mismatch at chunk_idx={chunk_idx}");
        }
    }
}
