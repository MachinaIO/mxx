use super::*;
use crate::{poly::dcrt::gpu::detected_gpu_device_ids, sampler::trapdoor::GpuPreimageRequest};

pub(super) struct GpuLutBaseDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: T,
    b1_matrix: M,
}

struct GpuLutDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: &'a T,
    b1_matrix: &'a M,
    w_block_identity: M,
    w_block_gy: M,
    w_block_v: M,
    w_block_vx: M,
    gadget_matrix: M,
}

pub(super) struct GpuGateBaseDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: T,
    b0_matrix: M,
    b1_matrix: M,
}

struct GpuGateDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: &'a T,
    b0_matrix: &'a M,
    b1_matrix: &'a M,
    w_block_identity: M,
    w_block_gy: M,
    w_block_v: M,
    w_block_vx: M,
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
    pub(super) fn prepare_gpu_lut_base_device_shared(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
    ) -> Vec<GpuLutBaseDeviceShared<M, TS::Trapdoor>> {
        let device_ids = detected_gpu_device_ids();
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for gpu sample_lut_preimages"
        );

        let b1_trapdoor_bytes = TS::trapdoor_to_bytes(b1_trapdoor);
        let b1_matrix_bytes = b1_matrix.to_compact_bytes();

        device_ids
            .into_iter()
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                let local_trapdoor = TS::trapdoor_from_bytes(&local_params, &b1_trapdoor_bytes)
                    .expect("failed to deserialize trapdoor for preloaded LUT device resources");
                let local_b1_matrix = M::from_compact_bytes(&local_params, &b1_matrix_bytes);
                GpuLutBaseDeviceShared {
                    device_id,
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
        base_shared: &'a [GpuLutBaseDeviceShared<M, TS::Trapdoor>],
    ) -> Vec<GpuLutDeviceShared<'a, M, TS::Trapdoor>> {
        base_shared
            .iter()
            .map(|base| {
                let w_block_identity = self.derive_w_block_identity(&base.params, lut_id);
                let w_block_gy = self.derive_w_block_gy(&base.params, lut_id);
                let w_block_v = self.derive_w_block_v(&base.params, lut_id);
                let w_block_vx = self.derive_w_block_vx(&base.params, lut_id);
                let gadget_matrix = M::gadget_matrix(&base.params, self.d);
                GpuLutDeviceShared {
                    device_id: base.device_id,
                    params: &base.params,
                    trapdoor: &base.trapdoor,
                    b1_matrix: &base.b1_matrix,
                    w_block_identity,
                    w_block_gy,
                    w_block_v,
                    w_block_vx,
                    gadget_matrix,
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    pub(super) fn prepare_gpu_gate_base_device_shared(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b0_trapdoor: &TS::Trapdoor,
        b0_matrix: &M,
        b1_matrix: &M,
    ) -> Vec<GpuGateBaseDeviceShared<M, TS::Trapdoor>> {
        let device_ids = detected_gpu_device_ids();
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for gpu gate preimage sampling"
        );

        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);
        let b0_matrix_bytes = b0_matrix.to_compact_bytes();
        let b1_matrix_bytes = b1_matrix.to_compact_bytes();

        device_ids
            .into_iter()
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                let local_b0_trapdoor = TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                    .expect(
                        "failed to deserialize b0 trapdoor for preloaded gate device resources",
                    );
                let local_b0_matrix = M::from_compact_bytes(&local_params, &b0_matrix_bytes);
                let local_b1_matrix = M::from_compact_bytes(&local_params, &b1_matrix_bytes);
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
    fn copy_matrix_to_gpu_gate_devices(
        &self,
        matrix: &M,
        base_shared: &[GpuGateBaseDeviceShared<M, TS::Trapdoor>],
    ) -> Vec<M> {
        let bytes = matrix.to_compact_bytes();
        base_shared
            .iter()
            .map(|device_shared| M::from_compact_bytes(&device_shared.params, &bytes))
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn prepare_gpu_gate_device_shared<'a>(
        &self,
        base_shared: &'a [GpuGateBaseDeviceShared<M, TS::Trapdoor>],
        w_block_identity_by_device: Vec<M>,
        w_block_gy_by_device: Vec<M>,
        w_block_v_by_device: Vec<M>,
        w_block_vx_by_device: Vec<M>,
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
        assert_eq!(
            len,
            w_block_vx_by_device.len(),
            "w_block_vx device replicas must match gate base shared length"
        );

        let mut w_identity_it = w_block_identity_by_device.into_iter();
        let mut w_gy_it = w_block_gy_by_device.into_iter();
        let mut w_v_it = w_block_v_by_device.into_iter();
        let mut w_vx_it = w_block_vx_by_device.into_iter();

        (0..len)
            .map(|idx| {
                let base = &base_shared[idx];
                GpuGateDeviceShared {
                    device_id: base.device_id,
                    params: &base.params,
                    b0_trapdoor: &base.b0_trapdoor,
                    b0_matrix: &base.b0_matrix,
                    b1_matrix: &base.b1_matrix,
                    w_block_identity: w_identity_it
                        .next()
                        .expect("missing w_block_identity replica for gate device"),
                    w_block_gy: w_gy_it.next().expect("missing w_block_gy replica for gate device"),
                    w_block_v: w_v_it.next().expect("missing w_block_v replica for gate device"),
                    w_block_vx: w_vx_it.next().expect("missing w_block_vx replica for gate device"),
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu")]
    fn sample_lut_preimages_gpu(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        lut_aux_id_prefix: &str,
        shared: &[GpuLutDeviceShared<'_, M, TS::Trapdoor>],
        batch: &[(usize, i32, M::P)],
    ) -> Vec<CompactBytesJob<M>> {
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
        debug_assert_eq!(
            shared[0].w_block_vx.col_size(),
            m * k_small,
            "w_block_vx columns must equal d * modulus_digits^2"
        );

        let jobs = {
            if batch.is_empty() {
                Vec::new()
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                assert!(
                    batch.len() <= shared.len(),
                    "LUT_PREIMAGE_CHUNK_SIZE must be <= available GPU devices: batch_size={}, devices={}",
                    batch.len(),
                    shared.len()
                );
                let shared_by_device = shared
                    .iter()
                    .map(|entry| (entry.device_id, entry))
                    .collect::<HashMap<i32, &GpuLutDeviceShared<'_, M, TS::Trapdoor>>>();
                debug_assert_eq!(
                    shared_by_device.len(),
                    shared.len(),
                    "gpu shared resources must have unique device ids"
                );

                let requests = batch
                        .par_iter()
                        .map(|(idx, device_id, local_y_poly)| {
                            let shared_dev = shared_by_device
                                .get(device_id)
                                .copied()
                                .unwrap_or_else(|| panic!("missing gpu shared entry for device {}", device_id));

                            let gy = shared_dev.gadget_matrix.clone() * local_y_poly;
                            let gy_decomposed = gy.decompose();
                            let mut target = shared_dev.w_block_gy.clone() * gy_decomposed;

                            let v_idx = derive_lut_v_idx_from_hash::<M, HS>(
                                shared_dev.params,
                                self.hash_key,
                                lut_id,
                                *idx,
                                d,
                            );
                            let w_v = shared_dev.w_block_v.clone() * &v_idx;
                            target.add_in_place(&w_v);

                            let idx_poly = M::P::from_usize_to_constant(shared_dev.params, *idx);
                            let idx_identity_decomposed =
                                M::identity(shared_dev.params, m, Some(idx_poly)).small_decompose();
                            let vx_rhs = idx_identity_decomposed * &v_idx;
                            let w_vx = shared_dev.w_block_vx.clone() * vx_rhs;
                            target.add_in_place(&w_vx);
                            target.add_in_place(&shared_dev.w_block_identity);
                            debug!(
                                "Constructed device-local target for LUT preimage: lut_id={}, row_idx={}, device_id={}",
                                lut_id,
                                idx,
                                shared_dev.device_id
                            );

                            GpuPreimageRequest {
                                entry_idx: *idx,
                                params: shared_dev.params,
                                trapdoor: shared_dev.trapdoor,
                                public_matrix: shared_dev.b1_matrix,
                                target,
                            }
                        })
                        .collect::<Vec<_>>();

                let preimages = trap_sampler.preimage_batched_sharded(requests);
                let mut preimage_by_idx = preimages.into_iter().collect::<HashMap<usize, M>>();
                batch
                    .iter()
                    .map(|(idx, _, _)| {
                        let lut_aux_id = format!("{lut_aux_id_prefix}_idx{idx}");
                        let preimage = preimage_by_idx
                            .remove(idx)
                            .unwrap_or_else(|| panic!("missing preimage for LUT row idx={idx}"));
                        CompactBytesJob::new(lut_aux_id, vec![(0usize, preimage)])
                    })
                    .collect::<Vec<_>>()
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
    ) {
        if pending.is_empty() {
            return;
        }
        assert!(
            pending.len() <= shared.len(),
            "GGH15_GATE_PARALLELISM must be <= available GPU devices for gate preimage sampling: pending={}, devices={}",
            pending.len(),
            shared.len()
        );

        let d = self.d;
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        debug_assert!(
            shared.iter().all(|entry| entry.w_block_vx.col_size() == m_g * k_small),
            "w_block_vx columns must equal m_g * k_small"
        );
        debug_assert!(
            shared.iter().all(|entry| entry.b1_matrix.row_size() == d),
            "gate stage1 expects b1_matrix to have d rows"
        );

        let stage1_entries = pending
                .into_par_iter()
                .enumerate()
                .map(|(entry_pos, (gate_id, state))| {
                    let shared = &shared[entry_pos];
                    let uniform_sampler = US::new();
                    let s_g = uniform_sampler.sample_uniform(shared.params, d, d, DistType::TernaryDist);
                    let gate_target1 = {
                        let error = uniform_sampler.sample_uniform(
                            shared.params,
                            d,
                            shared.b1_matrix.col_size(),
                            DistType::GaussDist { sigma: self.error_sigma },
                        );
                        s_g.clone() * shared.b1_matrix + error
                    };
                    debug!(
                        "Constructed stage1 target for gate preimage: gate_id={}, lut_id={}, device_id={}",
                        gate_id,
                        lut_id,
                        shared.device_id
                    );
                    (entry_pos, gate_id, state, s_g, gate_target1)
                })
                .collect::<Vec<_>>();
        let mut stage2_inputs = Vec::with_capacity(stage1_entries.len());
        let mut stage1_requests = Vec::with_capacity(stage1_entries.len());
        for (entry_pos, gate_id, state, s_g, target) in stage1_entries {
            let shared = &shared[entry_pos];
            stage2_inputs.push((entry_pos, gate_id, state, s_g));
            stage1_requests.push(GpuPreimageRequest {
                entry_idx: entry_pos,
                params: shared.params,
                trapdoor: shared.b0_trapdoor,
                public_matrix: shared.b0_matrix,
                target,
            });
        }
        let mut stage1_preimages = trap_sampler
            .preimage_batched_sharded(stage1_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage1_jobs = stage2_inputs
            .iter()
            .map(|(entry_pos, gate_id, _, _)| {
                let preimage_gate1 = stage1_preimages.remove(entry_pos).unwrap_or_else(|| {
                    panic!("missing gate stage1 preimage for entry_pos={entry_pos}")
                });
                debug!("Sampled gate preimage 1: gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate1_id_prefix(params, *gate_id),
                    vec![(0, preimage_gate1)],
                )
            })
            .collect::<Vec<_>>();
        stage1_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage2_entries = stage2_inputs
                .into_par_iter()
                .map(|(entry_pos, gate_id, state, s_g)| {
                    let shared = &shared[entry_pos];
                    let hash_sampler = HS::new();
                    let uniform_sampler = US::new();
                    let out_matrix = hash_sampler.sample_hash(
                        shared.params,
                        self.hash_key,
                        format!("ggh15_gate_a_out_{}", gate_id),
                        d,
                        m_g,
                        DistType::FinRingDist,
                    );
                    let mut target_gate2_identity = s_g.clone() * &shared.w_block_identity;
                    target_gate2_identity.add_in_place(&out_matrix);
                    let error = uniform_sampler.sample_uniform(
                        shared.params,
                        d,
                        m_g,
                        DistType::GaussDist { sigma: self.error_sigma },
                    );
                    target_gate2_identity.add_in_place(&error);
                    debug!(
                        "Constructed stage2 identity target for gate preimage: gate_id={}, lut_id={}, device_id={}",
                        gate_id,
                        lut_id,
                        shared.device_id
                    );
                    (entry_pos, gate_id, state, s_g, target_gate2_identity)
                })
                .collect::<Vec<_>>();
        let mut stage3_inputs = Vec::with_capacity(stage2_entries.len());
        let mut stage2_requests = Vec::with_capacity(stage2_entries.len());
        for (entry_pos, gate_id, state, s_g, target) in stage2_entries {
            let shared = &shared[entry_pos];
            stage3_inputs.push((entry_pos, gate_id, state, s_g));
            stage2_requests.push(GpuPreimageRequest {
                entry_idx: entry_pos,
                params: shared.params,
                trapdoor: shared.b0_trapdoor,
                public_matrix: shared.b0_matrix,
                target,
            });
        }
        let mut stage2_preimages = trap_sampler
            .preimage_batched_sharded(stage2_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage2_jobs = stage3_inputs
            .iter()
            .map(|(entry_pos, gate_id, _, _)| {
                let preimage_gate2_identity =
                    stage2_preimages.remove(entry_pos).unwrap_or_else(|| {
                        panic!("missing gate stage2 preimage for entry_pos={entry_pos}")
                    });
                debug!(
                    "Sampled gate preimage 2 (identity part): gate_id={}, lut_id={}",
                    gate_id, lut_id
                );
                CompactBytesJob::new(
                    self.preimage_gate2_identity_id_prefix(params, *gate_id),
                    vec![(0, preimage_gate2_identity)],
                )
            })
            .collect::<Vec<_>>();
        stage2_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage3_requests = stage3_inputs
            .iter()
            .map(|(entry_pos, _, _, s_g)| {
                let shared = &shared[*entry_pos];
                let uniform_sampler = US::new();
                let mut target_gate2_gy = s_g.clone() * &shared.w_block_gy;
                let target_high_gy = -M::gadget_matrix(shared.params, d);
                target_gate2_gy.add_in_place(&target_high_gy);
                let error = uniform_sampler.sample_uniform(
                    shared.params,
                    d,
                    m_g,
                    DistType::GaussDist { sigma: self.error_sigma },
                );
                target_gate2_gy.add_in_place(&error);
                GpuPreimageRequest {
                    entry_idx: *entry_pos,
                    params: shared.params,
                    trapdoor: shared.b0_trapdoor,
                    public_matrix: shared.b0_matrix,
                    target: target_gate2_gy,
                }
            })
            .collect::<Vec<_>>();
        let mut stage3_preimages = trap_sampler
            .preimage_batched_sharded(stage3_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage3_jobs = stage3_inputs
            .iter()
            .map(|(entry_pos, gate_id, _, _)| {
                let preimage_gate2_gy = stage3_preimages.remove(entry_pos).unwrap_or_else(|| {
                    panic!("missing gate stage3 preimage for entry_pos={entry_pos}")
                });
                debug!("Sampled gate preimage 2 (gy part): gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate2_gy_id_prefix(params, *gate_id),
                    vec![(0, preimage_gate2_gy)],
                )
            })
            .collect::<Vec<_>>();
        stage3_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage4_entries = stage3_inputs
                .into_par_iter()
                .map(|(entry_pos, gate_id, state, s_g)| {
                    let shared = &shared[entry_pos];
                    let hash_sampler = HS::new();
                    let uniform_sampler = US::new();
                    let input_matrix = M::from_compact_bytes(shared.params, &state.input_pubkey_bytes);
                    let u_g_decomposed = hash_sampler.sample_hash_decomposed(
                        shared.params,
                        self.hash_key,
                        format!("ggh15_lut_u_g_matrix_{}", gate_id),
                        d,
                        m_g,
                        DistType::FinRingDist,
                    );
                    debug!(
                        "Derived decomposed u_g_matrix for gate: gate_id={}, lut_id={}, rows={}, cols={}, device_id={}",
                        gate_id,
                        lut_id,
                        u_g_decomposed.row_size(),
                        u_g_decomposed.col_size(),
                        shared.device_id
                    );
                    let mut target_gate2_v = s_g.clone() * &shared.w_block_v;
                    let target_high_v = -(input_matrix * u_g_decomposed);
                    target_gate2_v.add_in_place(&target_high_v);
                    let error = uniform_sampler.sample_uniform(
                        shared.params,
                        d,
                        m_g,
                        DistType::GaussDist { sigma: self.error_sigma },
                    );
                    target_gate2_v.add_in_place(&error);
                    (entry_pos, gate_id, s_g, target_gate2_v)
                })
                .collect::<Vec<_>>();
        let mut stage5_inputs = Vec::with_capacity(stage4_entries.len());
        let mut stage4_requests = Vec::with_capacity(stage4_entries.len());
        for (entry_pos, gate_id, s_g, target) in stage4_entries {
            let shared = &shared[entry_pos];
            stage5_inputs.push((entry_pos, gate_id, s_g));
            stage4_requests.push(GpuPreimageRequest {
                entry_idx: entry_pos,
                params: shared.params,
                trapdoor: shared.b0_trapdoor,
                public_matrix: shared.b0_matrix,
                target,
            });
        }

        let mut stage4_preimages = trap_sampler
            .preimage_batched_sharded(stage4_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage4_jobs = stage5_inputs
            .iter()
            .map(|(entry_pos, gate_id, _)| {
                let preimage_gate2_v = stage4_preimages.remove(entry_pos).unwrap_or_else(|| {
                    panic!("missing gate stage4 preimage for entry_pos={entry_pos}")
                });
                debug!("Sampled gate preimage 2 (v part): gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate2_v_id_prefix(params, *gate_id),
                    vec![(0, preimage_gate2_v)],
                )
            })
            .collect::<Vec<_>>();
        stage4_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        for chunk_idx in 0..k_small {
            let chunk_col_start =
                chunk_idx.checked_mul(m_g).expect("stage5 chunk start column overflow");
            let chunk_col_end = chunk_col_start + m_g;
            let stage5_requests = stage5_inputs
                .iter()
                .map(|(entry_pos, gate_id, s_g)| {
                    let shared = &shared[*entry_pos];
                    let hash_sampler = HS::new();
                    let uniform_sampler = US::new();
                    let u_g_matrix = hash_sampler.sample_hash(
                        shared.params,
                        self.hash_key,
                        format!("ggh15_lut_u_g_matrix_{}", gate_id),
                        d,
                        m_g,
                        DistType::FinRingDist,
                    );
                    let target_high_vx_chunk = self.build_stage5_target_high_vx_chunk(
                        shared.params,
                        &u_g_matrix,
                        chunk_idx,
                        m_g,
                    );
                    let w_block_vx_chunk = shared.w_block_vx.slice(
                        0,
                        shared.w_block_vx.row_size(),
                        chunk_col_start,
                        chunk_col_end,
                    );
                    let mut target_gate2_vx_chunk = s_g.clone() * &w_block_vx_chunk;
                    target_gate2_vx_chunk.add_in_place(&target_high_vx_chunk);
                    let error = uniform_sampler.sample_uniform(
                        shared.params,
                        d,
                        m_g,
                        DistType::GaussDist { sigma: self.error_sigma },
                    );
                    target_gate2_vx_chunk.add_in_place(&error);
                    GpuPreimageRequest {
                        entry_idx: *entry_pos,
                        params: shared.params,
                        trapdoor: shared.b0_trapdoor,
                        public_matrix: shared.b0_matrix,
                        target: target_gate2_vx_chunk,
                    }
                })
                .collect::<Vec<_>>();
            let mut stage5_preimages = trap_sampler
                .preimage_batched_sharded(stage5_requests)
                .into_iter()
                .collect::<HashMap<usize, M>>();
            let stage5_jobs = stage5_inputs
                    .iter()
                    .map(|(entry_pos, gate_id, _)| {
                        let preimage_gate2_vx_chunk = stage5_preimages
                            .remove(entry_pos)
                            .unwrap_or_else(|| {
                                panic!("missing gate stage5 preimage for entry_pos={entry_pos}, chunk_idx={chunk_idx}")
                            });
                        debug!(
                            "Sampled gate preimage 2 (vx part): gate_id={}, lut_id={}, chunk_idx={}",
                            gate_id,
                            lut_id,
                            chunk_idx
                        );
                        CompactBytesJob::new(
                            self.preimage_gate2_vx_chunk_id_prefix(params, *gate_id, chunk_idx),
                            vec![(0, preimage_gate2_vx_chunk)],
                        )
                    })
                    .collect::<Vec<_>>();
            stage5_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
        }
    }

    #[cfg(feature = "gpu")]
    fn build_stage5_target_high_vx_chunk(
        &self,
        params: &<M::P as Poly>::Params,
        u_g_matrix: &M,
        chunk_idx: usize,
        m_g: usize,
    ) -> M {
        let d = self.d;
        let k_small = small_gadget_chunk_count::<M>(params);
        debug_assert!(chunk_idx < k_small, "chunk_idx must be < k_small");

        let chunk_start = chunk_idx.checked_mul(m_g).expect("stage5 chunk column offset overflow");
        let mut target_high_vx_chunk = M::zero(params, d, m_g);
        let scalar_by_digit = (0..k_small)
            .map(|digit| M::P::from_power_of_base_to_constant(params, digit))
            .collect::<Vec<_>>();

        for local_col in 0..m_g {
            let global_col = chunk_start + local_col;
            let src_col = global_col / k_small;
            let digit = global_col % k_small;
            let src_column = u_g_matrix.slice(0, d, src_col, src_col + 1);
            let scaled_column = src_column * &scalar_by_digit[digit];
            target_high_vx_chunk.copy_block_from(&scaled_column, 0, local_col, 0, 0, d, 1);
        }

        target_high_vx_chunk
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
        assert!(
            chunk_size <= gpu_lut_shared.len(),
            "LUT_PREIMAGE_CHUNK_SIZE must be <= available GPU devices: chunk_size={}, devices={}",
            chunk_size,
            gpu_lut_shared.len()
        );
        let sample_lut_batch = |current_batch: &[(usize, i32, M::P)]| {
            self.sample_lut_preimages_gpu(
                params,
                lut_id,
                lut_aux_id_prefix,
                &gpu_lut_shared,
                current_batch,
            )
        };

        let mut batch: Vec<(usize, i32, M::P)> = Vec::with_capacity(chunk_size);
        let mut pending_store_jobs: Option<Vec<CompactBytesJob<M>>> = None;
        for input_idx in pending_input_indices.iter().copied() {
            let device_slot = batch.len();
            let device_shared = &gpu_lut_shared[device_slot];
            let input_u64 = u64::try_from(input_idx).expect("LUT input index must fit in u64");
            let (idx, y_elem) = plt
                .get(device_shared.params, input_u64)
                .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", input_idx));
            let idx_usize = usize::try_from(idx).expect("LUT row index must fit in usize");
            let y_poly = M::P::from_elem_to_constant(device_shared.params, &y_elem);
            batch.push((idx_usize, device_shared.device_id, y_poly));

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
        w_block_vx: &M,
        total_gate_count: usize,
        total_gates: &mut usize,
        start: &Instant,
    ) {
        let w_block_identity_by_device =
            self.copy_matrix_to_gpu_gate_devices(w_block_identity, gpu_gate_base_shared);
        let w_block_gy_by_device =
            self.copy_matrix_to_gpu_gate_devices(w_block_gy, gpu_gate_base_shared);
        let w_block_v_by_device =
            self.copy_matrix_to_gpu_gate_devices(w_block_v, gpu_gate_base_shared);
        let w_block_vx_by_device =
            self.copy_matrix_to_gpu_gate_devices(w_block_vx, gpu_gate_base_shared);
        let gpu_gate_shared = self.prepare_gpu_gate_device_shared(
            gpu_gate_base_shared,
            w_block_identity_by_device,
            w_block_gy_by_device,
            w_block_v_by_device,
            w_block_vx_by_device,
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
