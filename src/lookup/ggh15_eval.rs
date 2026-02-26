#[cfg(feature = "gpu")]
use crate::poly::dcrt::gpu::detected_gpu_device_ids;
#[cfg(feature = "gpu")]
use crate::sampler::trapdoor::GpuPreimageRequest;
use crate::{
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::{read_bytes_from_multi_batch, read_matrix_from_multi_batch},
        write::{GlobalTableIndex, add_lookup_buffer, get_lookup_buffer, get_lookup_buffer_bytes},
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::read_to_string,
    marker::PhantomData,
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    input_pubkey_bytes: Vec<u8>,
    _m: PhantomData<M>,
}

struct CompactBytesJob<M>
where
    M: PolyMatrix,
{
    id_prefix: String,
    matrices: Vec<(usize, M)>,
}

impl<M> CompactBytesJob<M>
where
    M: PolyMatrix,
{
    fn new(id_prefix: String, matrices: Vec<(usize, M)>) -> Self {
        Self { id_prefix, matrices }
    }

    fn wait_then_store(self) {
        let mut payloads = Vec::with_capacity(self.matrices.len());
        let mut max_len = 0usize;
        for (idx, matrix) in self.matrices {
            let bytes = matrix.into_compact_bytes();
            max_len = max_len.max(bytes.len());
            payloads.push((idx, bytes));
        }
        // Match get_lookup_buffer behavior for variable compact payload lengths.
        let padded_len = max_len.saturating_add(16);
        for (_, bytes) in payloads.iter_mut() {
            if bytes.len() < padded_len {
                bytes.resize(padded_len, 0);
            }
        }
        add_lookup_buffer(get_lookup_buffer_bytes(payloads, &self.id_prefix));
    }
}

#[cfg(feature = "gpu")]
struct GpuLutBaseDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    trapdoor: T,
    b1_matrix: M,
}

#[cfg(feature = "gpu")]
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

#[cfg(feature = "gpu")]
struct GpuGateBaseDeviceShared<M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: T,
    b0_matrix: M,
    b1_trapdoor: T,
    b1_matrix: M,
}

#[cfg(feature = "gpu")]
struct GpuGateDeviceShared<'a, M, T>
where
    M: PolyMatrix,
    T: Send + Sync,
{
    device_id: i32,
    params: &'a <<M as PolyMatrix>::P as Poly>::Params,
    b0_trapdoor: &'a T,
    b0_matrix: &'a M,
    b1_trapdoor: &'a T,
    b1_matrix: &'a M,
    w_block_identity: M,
    w_block_gy: M,
    w_block_v: M,
    w_block_vx: M,
}

fn derive_lut_v_idx_from_hash<M, HS>(
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    lut_id: usize,
    idx: usize,
    d: usize,
) -> M
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    let m_g = d * params.modulus_digits();
    let v_idx = HS::new().sample_hash_decomposed(
        params,
        hash_key,
        format!("ggh15_lut_v_idx_{}_{}", lut_id, idx),
        d,
        m_g,
        DistType::FinRingDist,
    );
    debug_assert_eq!(v_idx.row_size(), m_g, "derived v_idx rows must equal d * modulus_digits");
    debug_assert_eq!(v_idx.col_size(), m_g, "derived v_idx cols must equal d * modulus_digits");
    v_idx
}

fn small_gadget_chunk_count<M>(params: &<M::P as Poly>::Params) -> usize
where
    M: PolyMatrix,
{
    let (_, _, crt_depth) = params.to_crt();
    params.modulus_digits() / crt_depth
}

fn read_matrix_from_chunks<M>(
    params: &<M::P as Poly>::Params,
    dir: &Path,
    id_prefix: &str,
    expected_chunk_cols: usize,
    chunk_count: usize,
    label: &str,
) -> M
where
    M: PolyMatrix,
{
    assert!(chunk_count > 0, "{label} chunk_count must be > 0 (id_prefix={id_prefix})");
    let first = read_matrix_from_multi_batch::<M>(params, dir, id_prefix, 0)
        .unwrap_or_else(|| panic!("{label} (index 0) not found: id_prefix={id_prefix}"));
    assert_eq!(
        first.col_size(),
        expected_chunk_cols,
        "{label} chunk 0 must have {expected_chunk_cols} columns (id_prefix={id_prefix})"
    );
    if chunk_count == 1 {
        return first;
    }

    let mut chunks = Vec::with_capacity(chunk_count - 1);
    for chunk_idx in 1..chunk_count {
        let chunk = read_matrix_from_multi_batch::<M>(params, dir, id_prefix, chunk_idx)
            .unwrap_or_else(|| {
                panic!("{label} chunk {chunk_idx} not found: id_prefix={id_prefix}")
            });
        assert_eq!(
            chunk.col_size(),
            expected_chunk_cols,
            "{label} chunk {} must have {expected_chunk_cols} columns (id_prefix={id_prefix})",
            chunk_idx
        );
        chunks.push(chunk);
    }
    first.concat_columns_owned(chunks)
}

pub struct GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M>,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub d: usize,
    pub trapdoor_sigma: f64,
    pub error_sigma: f64,
    pub dir_path: PathBuf,
    pub insert_1_to_s: bool,
    lut_state: DashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
    gate_state: DashMap<GateId, GateState<M>>,
    _us: PhantomData<US>,
    _hs: PhantomData<HS>,
    _ts: PhantomData<TS>,
}

impl<M, US, HS, TS> GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        d: usize,
        trapdoor_sigma: f64,
        error_sigma: f64,
        dir_path: PathBuf,
        insert_1_to_s: bool,
    ) -> Self {
        debug_assert!(!insert_1_to_s || d > 1, "cannot insert 1 into s when d = 1");

        Self {
            hash_key,
            d,
            trapdoor_sigma,
            error_sigma,
            dir_path,
            insert_1_to_s,
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _us: PhantomData,
            _hs: PhantomData,
            _ts: PhantomData,
        }
    }

    #[cfg(feature = "gpu")]
    fn prepare_gpu_lut_base_device_shared(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
    ) -> Vec<GpuLutBaseDeviceShared<M, TS::Trapdoor>> {
        let device_ids = params.device_ids();
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
    fn prepare_gpu_gate_base_device_shared(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        b0_trapdoor: &TS::Trapdoor,
        b0_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
    ) -> Vec<GpuGateBaseDeviceShared<M, TS::Trapdoor>> {
        let device_ids = detected_gpu_device_ids();
        assert!(
            !device_ids.is_empty(),
            "at least one GPU device is required for gpu gate preimage sampling"
        );

        let b0_trapdoor_bytes = TS::trapdoor_to_bytes(b0_trapdoor);
        let b0_matrix_bytes = b0_matrix.to_compact_bytes();
        let b1_trapdoor_bytes = TS::trapdoor_to_bytes(b1_trapdoor);
        let b1_matrix_bytes = b1_matrix.to_compact_bytes();

        device_ids
            .into_iter()
            .map(|device_id| {
                let local_params = params.params_for_device(device_id);
                let local_b0_trapdoor = TS::trapdoor_from_bytes(&local_params, &b0_trapdoor_bytes)
                    .expect(
                        "failed to deserialize b0 trapdoor for preloaded gate device resources",
                    );
                let local_b1_trapdoor = TS::trapdoor_from_bytes(&local_params, &b1_trapdoor_bytes)
                    .expect(
                        "failed to deserialize b1 trapdoor for preloaded gate device resources",
                    );
                let local_b0_matrix = M::from_compact_bytes(&local_params, &b0_matrix_bytes);
                let local_b1_matrix = M::from_compact_bytes(&local_params, &b1_matrix_bytes);
                GpuGateBaseDeviceShared {
                    device_id,
                    params: local_params,
                    b0_trapdoor: local_b0_trapdoor,
                    b0_matrix: local_b0_matrix,
                    b1_trapdoor: local_b1_trapdoor,
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
                    b1_trapdoor: &base.b1_trapdoor,
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
    fn sample_lut_preimages(
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
                        let mut w_t_idx = shared_dev.w_block_gy.clone() * gy_decomposed;

                        let v_idx = derive_lut_v_idx_from_hash::<M, HS>(
                            shared_dev.params,
                            self.hash_key,
                            lut_id,
                            *idx,
                            d,
                        );
                        let w_v = shared_dev.w_block_v.clone() * &v_idx;
                        w_t_idx.add_in_place(&w_v);

                        let idx_poly = M::P::from_usize_to_constant(shared_dev.params, *idx);
                        let idx_identity_decomposed =
                            M::identity(shared_dev.params, m, Some(idx_poly)).small_decompose();
                        let vx_rhs = idx_identity_decomposed * &v_idx;
                        let w_vx = shared_dev.w_block_vx.clone() * vx_rhs;
                        w_t_idx.add_in_place(&w_vx);
                        w_t_idx.add_in_place(&shared_dev.w_block_identity);

                        let mut target = M::zero(shared_dev.params, d + w_t_idx.row_size(), m);
                        target.copy_block_from(&w_t_idx, d, 0, 0, 0, w_t_idx.row_size(), m);

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

    #[cfg(not(feature = "gpu"))]
    fn sample_lut_preimages(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        lut_aux_id_prefix: &str,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
        w_block_vx: &M,
        batch: &[(usize, M::P)],
    ) -> Vec<CompactBytesJob<M>> {
        let sample_lut_preimages_start = Instant::now();
        debug!("Sampling LUT preimages started: lut_id={}, batch_size={}", lut_id, batch.len());
        let d = self.d;
        let m = d * params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let gadget_matrix = M::gadget_matrix(params, d);
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        debug_assert_eq!(
            w_block_identity.col_size(),
            m,
            "w_block_identity columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            w_block_gy.col_size(),
            m,
            "w_block_gy columns must equal d * modulus_digits"
        );
        debug_assert_eq!(
            w_block_v.col_size(),
            m,
            "w_block_v columns must equal d * modulus_digits^2"
        );
        debug_assert_eq!(
            w_block_vx.col_size(),
            m * k_small,
            "w_block_vx columns must equal d * modulus_digits^2"
        );

        let jobs = if batch.is_empty() {
            Vec::new()
        } else {
            let batch_size = batch.len();
            let batched_cols =
                m.checked_mul(batch_size).expect("batched target column count overflow");
            let entry_indices = batch.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();

            let mut w_t_idx_batched = if batch_size == 1 {
                w_block_identity.clone()
            } else {
                let identity_refs =
                    std::iter::repeat(w_block_identity).take(batch_size).collect::<Vec<_>>();
                identity_refs[0].concat_columns(&identity_refs[1..])
            };

            let gy_terms = batch
                .par_iter()
                .map(|(idx, y_poly)| {
                    let gy = gadget_matrix.clone() * y_poly;
                    let gy_decomposed = gy.decompose();
                    drop(gy);
                    debug!(
                        "Computed gy_decomposed for LUT preimage: lut_id={}, row_idx={}",
                        lut_id, idx
                    );
                    gy_decomposed
                })
                .collect::<Vec<_>>();
            let gy_cat = if gy_terms.len() == 1 {
                gy_terms.into_iter().next().expect("gy_terms must contain one matrix")
            } else {
                let mut gy_iter = gy_terms.into_iter();
                let gy_first = gy_iter.next().expect("gy_terms must be non-empty");
                gy_first.concat_columns_owned(gy_iter.collect())
            };
            let w_gy_batched = w_block_gy.clone() * gy_cat;
            w_t_idx_batched.add_in_place(&w_gy_batched);
            drop(w_gy_batched);

            let v_idx_terms = batch
                .par_iter()
                .map(|(idx, _)| {
                    let v_idx =
                        derive_lut_v_idx_from_hash::<M, HS>(params, self.hash_key, lut_id, *idx, d);
                    debug!(
                        "Derived v_idx for LUT preimage from hash: lut_id={}, row_idx={}",
                        lut_id, idx
                    );
                    (*idx, v_idx)
                })
                .collect::<Vec<_>>();
            let v_refs = v_idx_terms.iter().map(|(_, v_idx)| v_idx).collect::<Vec<_>>();
            let v_cat = if v_refs.len() == 1 {
                v_refs[0].clone()
            } else {
                v_refs[0].concat_columns(&v_refs[1..])
            };
            drop(v_refs);
            let w_v_batched = w_block_v.clone() * v_cat;
            w_t_idx_batched.add_in_place(&w_v_batched);
            drop(w_v_batched);

            let vx_rhs_terms = v_idx_terms
                .par_iter()
                .map(|(idx, v_idx)| {
                    let idx_poly = M::P::from_usize_to_constant(params, *idx);
                    let idx_identity_decomposed =
                        M::identity(params, m, Some(idx_poly)).small_decompose();
                    let vx_rhs = idx_identity_decomposed * v_idx;
                    debug!("Computed vx_rhs for LUT preimage: lut_id={}, row_idx={}", lut_id, idx);
                    vx_rhs
                })
                .collect::<Vec<_>>();
            let vx_rhs_cat = if vx_rhs_terms.len() == 1 {
                vx_rhs_terms.into_iter().next().expect("vx_rhs_terms must contain one matrix")
            } else {
                let mut rhs_iter = vx_rhs_terms.into_iter();
                let rhs_first = rhs_iter.next().expect("vx_rhs_terms must be non-empty");
                rhs_first.concat_columns_owned(rhs_iter.collect())
            };
            drop(v_idx_terms);
            let w_vx_batched = w_block_vx.clone() * vx_rhs_cat;
            w_t_idx_batched.add_in_place(&w_vx_batched);
            drop(w_vx_batched);

            let mut batched_target = M::zero(params, d + w_t_idx_batched.row_size(), batched_cols);
            batched_target.copy_block_from(
                &w_t_idx_batched,
                d,
                0,
                0,
                0,
                w_t_idx_batched.row_size(),
                batched_cols,
            );
            drop(w_t_idx_batched);

            let batched_preimage =
                trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &batched_target);
            let preimage_nrow = batched_preimage.row_size();
            entry_indices
                .into_iter()
                .enumerate()
                .map(|(entry_pos, idx)| {
                    let col_start =
                        entry_pos.checked_mul(m).expect("preimage slice column offset overflow");
                    let col_end = col_start + m;
                    let k_l_preimage = batched_preimage.slice(0, preimage_nrow, col_start, col_end);
                    let lut_aux_id = format!("{lut_aux_id_prefix}_idx{idx}");
                    CompactBytesJob::new(lut_aux_id, vec![(0usize, k_l_preimage)])
                })
                .collect::<Vec<_>>()
        };
        drop(gadget_matrix);
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
    fn sample_gate_preimages_batch(
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

        let stage1_entries = pending
            .into_par_iter()
            .enumerate()
            .map(|(entry_pos, (gate_id, state))| {
                let shared = &shared[entry_pos];
                let uniform_sampler = US::new();
                let s_g = if self.insert_1_to_s {
                    let s_g_bar = uniform_sampler.sample_uniform(
                        shared.params,
                        d - 1,
                        d - 1,
                        DistType::TernaryDist,
                    );
                    s_g_bar.concat_diag_owned(vec![M::identity(shared.params, 1, None)])
                } else {
                    uniform_sampler.sample_uniform(shared.params, d, d, DistType::TernaryDist)
                };
                let s_g_concat = M::identity(shared.params, d, None).concat_columns_owned(vec![s_g]);
                let gate_target1 = {
                    let error = uniform_sampler.sample_uniform(
                        shared.params,
                        d,
                        shared.b1_matrix.col_size(),
                        DistType::GaussDist { sigma: self.error_sigma },
                    );
                    s_g_concat.clone() * shared.b1_matrix + error
                };
                debug!(
                    "Constructed stage1 target for gate preimage: gate_id={}, lut_id={}, device_id={}",
                    gate_id,
                    lut_id,
                    shared.device_id
                );
                (entry_pos, gate_id, state, gate_target1)
            })
            .collect::<Vec<_>>();
        let mut stage2_inputs = Vec::with_capacity(stage1_entries.len());
        let mut stage1_requests = Vec::with_capacity(stage1_entries.len());
        for (entry_pos, gate_id, state, target) in stage1_entries {
            let shared = &shared[entry_pos];
            stage2_inputs.push((entry_pos, gate_id, state));
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
            .map(|(entry_pos, gate_id, _)| {
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
            .map(|(entry_pos, gate_id, state)| {
                let shared = &shared[entry_pos];
                let hash_sampler = HS::new();
                let out_matrix = hash_sampler.sample_hash(
                    shared.params,
                    self.hash_key,
                    format!("ggh15_gate_a_out_{}", gate_id),
                    d,
                    m_g,
                    DistType::FinRingDist,
                );
                let target_gate2_identity = out_matrix.concat_rows(&[&shared.w_block_identity]);
                debug!(
                    "Constructed stage2 identity target for gate preimage: gate_id={}, lut_id={}, device_id={}",
                    gate_id,
                    lut_id,
                    shared.device_id
                );
                (entry_pos, gate_id, state, target_gate2_identity)
            })
            .collect::<Vec<_>>();
        let mut stage3_inputs = Vec::with_capacity(stage2_entries.len());
        let mut stage2_requests = Vec::with_capacity(stage2_entries.len());
        for (entry_pos, gate_id, state, target) in stage2_entries {
            let shared = &shared[entry_pos];
            stage3_inputs.push((entry_pos, gate_id, state));
            stage2_requests.push(GpuPreimageRequest {
                entry_idx: entry_pos,
                params: shared.params,
                trapdoor: shared.b1_trapdoor,
                public_matrix: shared.b1_matrix,
                target,
            });
        }
        let mut stage2_preimages = trap_sampler
            .preimage_batched_sharded(stage2_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage2_jobs = stage3_inputs
            .iter()
            .map(|(entry_pos, gate_id, _)| {
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
            .map(|(entry_pos, _, _)| {
                let shared = &shared[*entry_pos];
                let target_high_gy = -M::gadget_matrix(shared.params, d);
                let target_gate2_gy = target_high_gy.concat_rows(&[&shared.w_block_gy]);
                GpuPreimageRequest {
                    entry_idx: *entry_pos,
                    params: shared.params,
                    trapdoor: shared.b1_trapdoor,
                    public_matrix: shared.b1_matrix,
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
            .map(|(entry_pos, gate_id, _)| {
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
            .map(|(entry_pos, gate_id, state)| {
                let shared = &shared[entry_pos];
                let hash_sampler = HS::new();
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
                let target_high_v = -(input_matrix * u_g_decomposed);
                let target_gate2_v = target_high_v.concat_rows(&[&shared.w_block_v]);
                (entry_pos, gate_id, target_gate2_v)
            })
            .collect::<Vec<_>>();
        let mut stage5_inputs = Vec::with_capacity(stage4_entries.len());
        let mut stage4_requests = Vec::with_capacity(stage4_entries.len());
        for (entry_pos, gate_id, target) in stage4_entries {
            let shared = &shared[entry_pos];
            stage5_inputs.push((entry_pos, gate_id));
            stage4_requests.push(GpuPreimageRequest {
                entry_idx: entry_pos,
                params: shared.params,
                trapdoor: shared.b1_trapdoor,
                public_matrix: shared.b1_matrix,
                target,
            });
        }

        let mut stage4_preimages = trap_sampler
            .preimage_batched_sharded(stage4_requests)
            .into_iter()
            .collect::<HashMap<usize, M>>();
        let stage4_jobs = stage5_inputs
            .iter()
            .map(|(entry_pos, gate_id)| {
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
                .map(|(entry_pos, gate_id)| {
                    let shared = &shared[*entry_pos];
                    let hash_sampler = HS::new();
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
                    let target_gate2_vx_chunk =
                        target_high_vx_chunk.concat_rows(&[&w_block_vx_chunk]);
                    GpuPreimageRequest {
                        entry_idx: *entry_pos,
                        params: shared.params,
                        trapdoor: shared.b1_trapdoor,
                        public_matrix: shared.b1_matrix,
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
                .map(|(entry_pos, gate_id)| {
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

    #[cfg(not(feature = "gpu"))]
    fn sample_gate_preimages_batch(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        lut_id: usize,
        pending: Vec<(GateId, GateState<M>)>,
        b0_trapdoor: &TS::Trapdoor,
        b0_matrix: &M,
        b1_trapdoor: &TS::Trapdoor,
        b1_matrix: &M,
        w_block_identity: &M,
        w_block_gy: &M,
        w_block_v: &M,
        w_block_vx: &M,
    ) {
        if pending.is_empty() {
            return;
        }
        let d = self.d;
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);

        let stage1_entries = pending
            .into_par_iter()
            .map(|(gate_id, state)| {
                let uniform_sampler = US::new();
                let s_g = if self.insert_1_to_s {
                    let s_g_bar =
                        uniform_sampler.sample_uniform(params, d - 1, d - 1, DistType::TernaryDist);
                    s_g_bar.concat_diag_owned(vec![M::identity(params, 1, None)])
                } else {
                    uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
                };
                let s_g_concat = M::identity(params, d, None).concat_columns_owned(vec![s_g]);
                let gate_target1 = {
                    let error = uniform_sampler.sample_uniform(
                        params,
                        d,
                        b1_matrix.col_size(),
                        DistType::GaussDist { sigma: self.error_sigma },
                    );
                    s_g_concat.clone() * b1_matrix + error
                };
                drop(s_g_concat);
                (gate_id, state, gate_target1)
            })
            .collect::<Vec<_>>();
        let stage1_target_cols = stage1_entries[0].2.col_size();
        debug_assert!(
            stage1_entries.iter().all(|(_, _, target)| target.col_size() == stage1_target_cols),
            "stage1 target columns must be identical across gates"
        );
        let mut stage1_gate_ids = Vec::with_capacity(stage1_entries.len());
        let mut stage2_inputs = Vec::with_capacity(stage1_entries.len());
        let mut stage1_targets = Vec::with_capacity(stage1_entries.len());
        for (gate_id, state, target) in stage1_entries {
            stage1_gate_ids.push(gate_id);
            stage2_inputs.push((gate_id, state));
            stage1_targets.push(target);
        }
        let stage1_batched_target = if stage1_targets.len() == 1 {
            stage1_targets.pop().expect("stage1_targets must contain one matrix")
        } else {
            let mut target_iter = stage1_targets.into_iter();
            let target_first = target_iter.next().expect("stage1_targets must be non-empty");
            target_first.concat_columns_owned(target_iter.collect())
        };
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let stage1_batched_preimage =
            trap_sampler.preimage(params, b0_trapdoor, b0_matrix, &stage1_batched_target);
        let stage1_preimage_nrow = stage1_batched_preimage.row_size();
        let stage1_jobs = stage1_gate_ids
            .into_iter()
            .enumerate()
            .map(|(gate_pos, gate_id)| {
                let col_start = gate_pos
                    .checked_mul(stage1_target_cols)
                    .expect("stage1 preimage slice column offset overflow");
                let col_end = col_start + stage1_target_cols;
                let preimage_gate1 =
                    stage1_batched_preimage.slice(0, stage1_preimage_nrow, col_start, col_end);
                debug!("Sampled gate preimage 1: gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate1_id_prefix(params, gate_id),
                    vec![(0, preimage_gate1)],
                )
            })
            .collect::<Vec<_>>();
        stage1_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage2_entries = stage2_inputs
            .into_par_iter()
            .map(|(gate_id, state)| {
                let hash_sampler = HS::new();
                let out_matrix = hash_sampler.sample_hash(
                    params,
                    self.hash_key,
                    format!("ggh15_gate_a_out_{}", gate_id),
                    d,
                    m_g,
                    DistType::FinRingDist,
                );
                let target_gate2_identity = out_matrix.concat_rows(&[w_block_identity]);
                (gate_id, state, target_gate2_identity)
            })
            .collect::<Vec<_>>();
        let stage2_target_cols = stage2_entries[0].2.col_size();
        debug_assert!(
            stage2_entries.iter().all(|(_, _, target)| target.col_size() == stage2_target_cols),
            "stage2 target columns must be identical across gates"
        );
        let mut stage2_gate_ids = Vec::with_capacity(stage2_entries.len());
        let mut stage3_inputs = Vec::with_capacity(stage2_entries.len());
        let mut stage2_targets = Vec::with_capacity(stage2_entries.len());
        for (gate_id, state, target) in stage2_entries {
            stage2_gate_ids.push(gate_id);
            stage3_inputs.push((gate_id, state));
            stage2_targets.push(target);
        }
        let stage2_batched_target = if stage2_targets.len() == 1 {
            stage2_targets.pop().expect("stage2_targets must contain one matrix")
        } else {
            let mut target_iter = stage2_targets.into_iter();
            let target_first = target_iter.next().expect("stage2_targets must be non-empty");
            target_first.concat_columns_owned(target_iter.collect())
        };
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let stage2_batched_preimage =
            trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &stage2_batched_target);
        let stage2_preimage_nrow = stage2_batched_preimage.row_size();
        let stage2_jobs = stage2_gate_ids
            .into_iter()
            .enumerate()
            .map(|(gate_pos, gate_id)| {
                let col_start = gate_pos
                    .checked_mul(stage2_target_cols)
                    .expect("stage2 preimage slice column offset overflow");
                let col_end = col_start + stage2_target_cols;
                let preimage_gate2_identity =
                    stage2_batched_preimage.slice(0, stage2_preimage_nrow, col_start, col_end);
                debug!(
                    "Sampled gate preimage 2 (identity part): gate_id={}, lut_id={}",
                    gate_id, lut_id
                );
                CompactBytesJob::new(
                    self.preimage_gate2_identity_id_prefix(params, gate_id),
                    vec![(0, preimage_gate2_identity)],
                )
            })
            .collect::<Vec<_>>();
        stage2_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage3_gate_ids = stage3_inputs.iter().map(|(gate_id, _)| *gate_id).collect::<Vec<_>>();
        let target_high_gy = -M::gadget_matrix(params, d);
        let target_gate2_gy = target_high_gy.concat_rows(&[w_block_gy]);
        drop(target_high_gy);
        let stage3_target_cols = target_gate2_gy.col_size();
        let stage3_batched_target = if stage3_gate_ids.len() == 1 {
            target_gate2_gy.clone()
        } else {
            let stage3_target_refs =
                std::iter::repeat(&target_gate2_gy).take(stage3_gate_ids.len()).collect::<Vec<_>>();
            stage3_target_refs[0].concat_columns(&stage3_target_refs[1..])
        };
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let stage3_batched_preimage =
            trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &stage3_batched_target);
        let stage3_preimage_nrow = stage3_batched_preimage.row_size();
        let stage3_jobs = stage3_gate_ids
            .into_iter()
            .enumerate()
            .map(|(gate_pos, gate_id)| {
                let col_start = gate_pos
                    .checked_mul(stage3_target_cols)
                    .expect("stage3 preimage slice column offset overflow");
                let col_end = col_start + stage3_target_cols;
                let preimage_gate2_gy =
                    stage3_batched_preimage.slice(0, stage3_preimage_nrow, col_start, col_end);
                debug!("Sampled gate preimage 2 (gy part): gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate2_gy_id_prefix(params, gate_id),
                    vec![(0, preimage_gate2_gy)],
                )
            })
            .collect::<Vec<_>>();
        stage3_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let stage4_entries = stage3_inputs
            .into_par_iter()
            .map(|(gate_id, state)| {
                let hash_sampler = HS::new();
                let input_matrix = M::from_compact_bytes(params, &state.input_pubkey_bytes);
                let u_g_decomposed = hash_sampler.sample_hash_decomposed(
                    params,
                    self.hash_key,
                    format!("ggh15_lut_u_g_matrix_{}", gate_id),
                    d,
                    m_g,
                    DistType::FinRingDist,
                );
                debug!(
                    "Derived decomposed u_g_matrix for gate: gate_id={}, lut_id={}, rows={}, cols={}",
                    gate_id,
                    lut_id,
                    u_g_decomposed.row_size(),
                    u_g_decomposed.col_size()
                );
                let target_high_v = -(input_matrix * u_g_decomposed);
                let target_gate2_v = target_high_v.concat_rows(&[w_block_v]);
                (gate_id, target_gate2_v)
            })
            .collect::<Vec<_>>();
        let stage4_target_cols = stage4_entries[0].1.col_size();
        debug_assert!(
            stage4_entries.iter().all(|(_, target)| target.col_size() == stage4_target_cols),
            "stage4 target columns must be identical across gates"
        );
        let mut stage4_gate_ids = Vec::with_capacity(stage4_entries.len());
        let mut stage5_inputs = Vec::with_capacity(stage4_entries.len());
        let mut stage4_targets = Vec::with_capacity(stage4_entries.len());
        for (gate_id, target) in stage4_entries {
            stage4_gate_ids.push(gate_id);
            stage5_inputs.push(gate_id);
            stage4_targets.push(target);
        }
        let stage4_batched_target = if stage4_targets.len() == 1 {
            stage4_targets.pop().expect("stage4_targets must contain one matrix")
        } else {
            let mut target_iter = stage4_targets.into_iter();
            let target_first = target_iter.next().expect("stage4_targets must be non-empty");
            target_first.concat_columns_owned(target_iter.collect())
        };
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let stage4_batched_preimage =
            trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &stage4_batched_target);
        let stage4_preimage_nrow = stage4_batched_preimage.row_size();
        let stage4_jobs = stage4_gate_ids
            .into_iter()
            .enumerate()
            .map(|(gate_pos, gate_id)| {
                let col_start = gate_pos
                    .checked_mul(stage4_target_cols)
                    .expect("stage4 preimage slice column offset overflow");
                let col_end = col_start + stage4_target_cols;
                let preimage_gate2_v =
                    stage4_batched_preimage.slice(0, stage4_preimage_nrow, col_start, col_end);
                debug!("Sampled gate preimage 2 (v part): gate_id={}, lut_id={}", gate_id, lut_id);
                CompactBytesJob::new(
                    self.preimage_gate2_v_id_prefix(params, gate_id),
                    vec![(0, preimage_gate2_v)],
                )
            })
            .collect::<Vec<_>>();
        stage4_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

        let small_gadget_matrix = M::small_gadget_matrix(params, m_g);
        let stage5_entries = stage5_inputs
            .into_par_iter()
            .map(|gate_id| {
                let hash_sampler = HS::new();
                let u_g_matrix = hash_sampler.sample_hash(
                    params,
                    self.hash_key,
                    format!("ggh15_lut_u_g_matrix_{}", gate_id),
                    d,
                    m_g,
                    DistType::FinRingDist,
                );
                let target_high_vx = u_g_matrix * &small_gadget_matrix;
                let target_gate2_vx = target_high_vx.concat_rows(&[w_block_vx]);
                drop(target_high_vx);
                (gate_id, target_gate2_vx)
            })
            .collect::<Vec<_>>();
        let stage5_target_cols = stage5_entries[0].1.col_size();
        debug_assert!(
            stage5_entries.iter().all(|(_, target)| target.col_size() == stage5_target_cols),
            "stage5 target columns must be identical across gates"
        );
        let mut stage5_gate_ids = Vec::with_capacity(stage5_entries.len());
        let mut stage5_targets = Vec::with_capacity(stage5_entries.len());
        for (gate_id, target) in stage5_entries {
            stage5_gate_ids.push(gate_id);
            stage5_targets.push(target);
        }
        let stage5_batched_target = if stage5_targets.len() == 1 {
            stage5_targets.pop().expect("stage5_targets must contain one matrix")
        } else {
            let mut target_iter = stage5_targets.into_iter();
            let target_first = target_iter.next().expect("stage5_targets must be non-empty");
            target_first.concat_columns_owned(target_iter.collect())
        };
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let stage5_batched_preimage =
            trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &stage5_batched_target);
        let stage5_preimage_nrow = stage5_batched_preimage.row_size();
        let stage5_jobs = stage5_gate_ids
            .into_iter()
            .enumerate()
            .flat_map(|(gate_pos, gate_id)| {
                let col_start = gate_pos
                    .checked_mul(stage5_target_cols)
                    .expect("stage5 preimage slice column offset overflow");
                let col_end = col_start + stage5_target_cols;
                let preimage_gate2_vx =
                    stage5_batched_preimage.slice(0, stage5_preimage_nrow, col_start, col_end);
                debug!("Sampled gate preimage 2 (vx part): gate_id={}, lut_id={}", gate_id, lut_id);
                debug_assert_eq!(
                    preimage_gate2_vx.col_size(),
                    m_g * k_small,
                    "stage5 preimage columns must equal m_g * k_small"
                );
                (0..k_small)
                    .map(|chunk_idx| {
                        let chunk_col_start =
                            chunk_idx.checked_mul(m_g).expect("stage5 chunk start column overflow");
                        let chunk_col_end = chunk_col_start + m_g;
                        let chunk = preimage_gate2_vx.slice(
                            0,
                            preimage_gate2_vx.row_size(),
                            chunk_col_start,
                            chunk_col_end,
                        );
                        CompactBytesJob::new(
                            self.preimage_gate2_vx_chunk_id_prefix(params, gate_id, chunk_idx),
                            vec![(0, chunk)],
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        stage5_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
    }

    fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs_f64();
        if secs >= 1.0 { format!("{secs:.3}s") } else { format!("{:.1}ms", secs * 1000.0) }
    }

    fn derive_w_block_with_tag(
        &self,
        params: &<M::P as Poly>::Params,
        lut_id: usize,
        tag: &str,
        cols: usize,
    ) -> M {
        let d = self.d;
        HS::new().sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_w_{tag}_{}", lut_id),
            d,
            cols,
            DistType::FinRingDist,
        )
    }

    fn derive_w_block_identity(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_identity", m_g)
    }

    fn derive_w_block_gy(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_gy", m_g)
    }

    fn derive_w_block_v(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        self.derive_w_block_with_tag(params, lut_id, "block_v", m_g)
    }

    fn derive_w_block_vx(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> M {
        let m_g = self.d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);
        self.derive_w_block_with_tag(params, lut_id, "block_vx", m_g * k_small)
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

    fn aux_checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        let (_, crt_bits, crt_depth) = params.to_crt();
        format!(
            "ggh15_aux_d{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_err{:.6}_ins{}_key{}",
            self.d,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            self.error_sigma,
            if self.insert_1_to_s { 1 } else { 0 },
            self.hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    fn lut_aux_id_prefix(&self, params: &<M::P as Poly>::Params, lut_id: usize) -> String {
        format!("{}_lut_aux_{}", self.aux_checkpoint_prefix(params), lut_id)
    }

    fn preimage_gate1_id_prefix(&self, params: &<M::P as Poly>::Params, gate_id: GateId) -> String {
        format!("{}_preimage_gate1_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_identity_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_identity_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_gy_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_gy_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_v_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_v_{}", self.aux_checkpoint_prefix(params), gate_id)
    }

    fn preimage_gate2_vx_chunk_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
        chunk_idx: usize,
    ) -> String {
        format!(
            "{}_preimage_gate2_vx_{}_chunk{}",
            self.aux_checkpoint_prefix(params),
            gate_id,
            chunk_idx
        )
    }

    pub fn checkpoint_prefix(&self, params: &<M::P as Poly>::Params) -> String {
        self.aux_checkpoint_prefix(params)
    }

    fn load_checkpoint_index(&self) -> Option<GlobalTableIndex> {
        let index_path = self.dir_path.join("lookup_tables.index");
        match read_to_string(&index_path) {
            Ok(index_data) => match serde_json::from_str::<GlobalTableIndex>(&index_data) {
                Ok(global_index) => {
                    info!(
                        "Loaded checkpoint index from {} (entries={})",
                        index_path.display(),
                        global_index.entries.len()
                    );
                    Some(global_index)
                }
                Err(err) => {
                    warn!("Failed to parse checkpoint index {}: {}", index_path.display(), err);
                    None
                }
            },
            Err(err) => {
                info!("Checkpoint index not available at {}: {}", index_path.display(), err);
                None
            }
        }
    }

    fn checkpoint_has_index(
        checkpoint_index: Option<&GlobalTableIndex>,
        part_index_cache: Option<&HashMap<String, HashSet<usize>>>,
        id_prefix: &str,
        target_k: usize,
    ) -> bool {
        if let Some(entry_info) = checkpoint_index.and_then(|idx| idx.entries.get(id_prefix)) &&
            entry_info.indices.contains(&target_k)
        {
            return true;
        }
        part_index_cache
            .and_then(|cache| cache.get(id_prefix))
            .is_some_and(|indices| indices.contains(&target_k))
    }

    fn strip_part_suffix(key: &str) -> &str {
        if let Some((base, part)) = key.rsplit_once("_part") &&
            !part.is_empty() &&
            part.bytes().all(|c| c.is_ascii_digit())
        {
            base
        } else {
            key
        }
    }

    fn build_part_index_cache(
        checkpoint_index: Option<&GlobalTableIndex>,
    ) -> Option<HashMap<String, HashSet<usize>>> {
        let checkpoint_index = checkpoint_index?;
        let mut cache: HashMap<String, HashSet<usize>> = HashMap::new();
        for (key, entry) in &checkpoint_index.entries {
            if let Some((base, part)) = key.rsplit_once("_part") &&
                !part.is_empty() &&
                part.bytes().all(|c| c.is_ascii_digit())
            {
                cache.entry(base.to_string()).or_default().extend(entry.indices.iter().copied());
            }
        }
        Some(cache)
    }

    fn collect_lut_completed_rows(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        lut_ids: &[usize],
    ) -> HashMap<usize, HashSet<usize>> {
        let Some(checkpoint_index) = checkpoint_index else {
            return HashMap::new();
        };
        if lut_ids.is_empty() {
            return HashMap::new();
        }
        let target_lut_ids: HashSet<usize> = lut_ids.iter().copied().collect();
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_");
        let mut completed_rows: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (raw_key, entry) in &checkpoint_index.entries {
            if !entry.indices.contains(&0) {
                continue;
            }
            let key = Self::strip_part_suffix(raw_key);
            let Some(rest) = key.strip_prefix(&lut_aux_prefix) else {
                continue;
            };
            let Some((lut_id_str, row_idx_str)) = rest.split_once("_idx") else {
                continue;
            };
            let Ok(lut_id) = lut_id_str.parse::<usize>() else {
                continue;
            };
            if !target_lut_ids.contains(&lut_id) {
                continue;
            }
            let Ok(row_idx) = row_idx_str.parse::<usize>() else {
                continue;
            };
            completed_rows.entry(lut_id).or_default().insert(row_idx);
        }
        completed_rows
    }

    fn gate_checkpoint_complete(
        checkpoint_index: Option<&GlobalTableIndex>,
        part_index_cache: Option<&HashMap<String, HashSet<usize>>>,
        checkpoint_prefix: &str,
        gate_id: GateId,
        gate_v_chunk_count: usize,
        gate_vx_chunk_count: usize,
    ) -> bool {
        let gate1_prefix = format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id);
        let gate2_identity_prefix =
            format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id);
        let gate2_gy_prefix = format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id);
        let gate2_v_prefix = format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id);
        Self::checkpoint_has_index(checkpoint_index, part_index_cache, &gate1_prefix, 0) &&
            Self::checkpoint_has_index(
                checkpoint_index,
                part_index_cache,
                &gate2_identity_prefix,
                0,
            ) &&
            Self::checkpoint_has_index(checkpoint_index, part_index_cache, &gate2_gy_prefix, 0) &&
            (0..gate_v_chunk_count).into_par_iter().all(|chunk_idx| {
                Self::checkpoint_has_index(
                    checkpoint_index,
                    part_index_cache,
                    &gate2_v_prefix,
                    chunk_idx,
                )
            }) &&
            (0..gate_vx_chunk_count).into_par_iter().all(|chunk_idx| {
                let gate2_vx_chunk_prefix =
                    format!("{checkpoint_prefix}_preimage_gate2_vx_{}_chunk{}", gate_id, chunk_idx);
                Self::checkpoint_has_index(
                    checkpoint_index,
                    part_index_cache,
                    &gate2_vx_chunk_prefix,
                    0,
                )
            })
    }

    fn has_resume_candidates(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        lut_ids: &[usize],
        gate_ids: &[GateId],
        gate_v_chunk_count: usize,
        gate_vx_chunk_count: usize,
    ) -> bool {
        let Some(checkpoint_index) = checkpoint_index else {
            return false;
        };

        #[derive(Default)]
        struct GateResumeState {
            gate1: bool,
            gate2_identity: bool,
            gate2_gy: bool,
            gate2_v_chunks: HashSet<usize>,
            gate2_vx_chunks: HashSet<usize>,
        }

        impl GateResumeState {
            fn is_complete(&self, gate_v_chunk_count: usize, gate_vx_chunk_count: usize) -> bool {
                self.gate1 &&
                    self.gate2_identity &&
                    self.gate2_gy &&
                    self.gate2_v_chunks.len() == gate_v_chunk_count &&
                    self.gate2_vx_chunks.len() == gate_vx_chunk_count
            }
        }

        let target_lut_ids: HashSet<usize> = lut_ids.iter().copied().collect();
        let target_gate_ids: HashSet<GateId> = gate_ids.iter().copied().collect();
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_");
        let gate1_prefix = format!("{checkpoint_prefix}_preimage_gate1_");
        let gate2_identity_prefix = format!("{checkpoint_prefix}_preimage_gate2_identity_");
        let gate2_gy_prefix = format!("{checkpoint_prefix}_preimage_gate2_gy_");
        let gate2_v_prefix = format!("{checkpoint_prefix}_preimage_gate2_v_");
        let gate2_vx_prefix = format!("{checkpoint_prefix}_preimage_gate2_vx_");

        let mut gate_states: HashMap<GateId, GateResumeState> = HashMap::new();
        for (raw_key, entry) in &checkpoint_index.entries {
            let key = Self::strip_part_suffix(raw_key);

            if let Some(rest) = key.strip_prefix(&lut_aux_prefix) &&
                let Some((lut_id_str, _)) = rest.split_once("_idx") &&
                let Ok(lut_id) = lut_id_str.parse::<usize>() &&
                target_lut_ids.contains(&lut_id) &&
                entry.indices.contains(&0)
            {
                return true;
            }

            if let Some(gate_id_str) = key.strip_prefix(&gate1_prefix) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                entry.indices.contains(&0)
            {
                let gate_id = GateId(gate_id_raw);
                let state = gate_states.entry(gate_id).or_default();
                state.gate1 = true;
                if state.is_complete(gate_v_chunk_count, gate_vx_chunk_count) {
                    return true;
                }
                continue;
            }

            if let Some(gate_id_str) = key.strip_prefix(&gate2_identity_prefix) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                entry.indices.contains(&0)
            {
                let gate_id = GateId(gate_id_raw);
                let state = gate_states.entry(gate_id).or_default();
                state.gate2_identity = true;
                if state.is_complete(gate_v_chunk_count, gate_vx_chunk_count) {
                    return true;
                }
                continue;
            }

            if let Some(gate_id_str) = key.strip_prefix(&gate2_gy_prefix) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                entry.indices.contains(&0)
            {
                let gate_id = GateId(gate_id_raw);
                let state = gate_states.entry(gate_id).or_default();
                state.gate2_gy = true;
                if state.is_complete(gate_v_chunk_count, gate_vx_chunk_count) {
                    return true;
                }
                continue;
            }

            if let Some(gate_id_str) = key.strip_prefix(&gate2_v_prefix) &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw))
            {
                let gate_id = GateId(gate_id_raw);
                let state = gate_states.entry(gate_id).or_default();
                entry
                    .indices
                    .iter()
                    .copied()
                    .filter(|chunk_idx| *chunk_idx < gate_v_chunk_count)
                    .for_each(|chunk_idx| {
                        state.gate2_v_chunks.insert(chunk_idx);
                    });
                if state.is_complete(gate_v_chunk_count, gate_vx_chunk_count) {
                    return true;
                }
                continue;
            }

            if let Some(rest) = key.strip_prefix(&gate2_vx_prefix) &&
                let Some((gate_id_str, chunk_idx_str)) = rest.split_once("_chunk") &&
                let Ok(gate_id_raw) = gate_id_str.parse::<usize>() &&
                target_gate_ids.contains(&GateId(gate_id_raw)) &&
                entry.indices.contains(&0) &&
                let Ok(chunk_idx) = chunk_idx_str.parse::<usize>() &&
                chunk_idx < gate_vx_chunk_count
            {
                let gate_id = GateId(gate_id_raw);
                let state = gate_states.entry(gate_id).or_default();
                state.gate2_vx_chunks.insert(chunk_idx);
                if state.is_complete(gate_v_chunk_count, gate_vx_chunk_count) {
                    return true;
                }
            }
        }

        false
    }

    fn load_b1_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        checkpoint_prefix: &str,
    ) -> Option<(TS::Trapdoor, M)> {
        let dir = self.dir_path.as_path();
        let b1_id_prefix = format!("{checkpoint_prefix}_b1");
        let b1_trapdoor_id_prefix = format!("{checkpoint_prefix}_b1_trapdoor");
        info!(
            "Trying B1 checkpoint load from {} (matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b1_id_prefix,
            b1_trapdoor_id_prefix
        );
        let b1_bytes = if let Some(bytes) = read_bytes_from_multi_batch(dir, &b1_id_prefix, 0) {
            bytes
        } else {
            info!(
                "B1 checkpoint matrix not found (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b1_id_prefix
            );
            return None;
        };
        let trapdoor_bytes =
            if let Some(bytes) = read_bytes_from_multi_batch(dir, &b1_trapdoor_id_prefix, 0) {
                bytes
            } else {
                info!(
                    "B1 checkpoint trapdoor not found (dir={}, id_prefix={}, index=0)",
                    dir.display(),
                    b1_trapdoor_id_prefix
                );
                return None;
            };
        let b1_trapdoor = if let Some(td) = TS::trapdoor_from_bytes(params, &trapdoor_bytes) {
            td
        } else {
            warn!(
                "Failed to decode B1 trapdoor bytes (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b1_trapdoor_id_prefix
            );
            return None;
        };
        let b1_matrix = M::from_compact_bytes(params, &b1_bytes);
        info!(
            "Loaded B1 checkpoint (dir={}, matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b1_id_prefix,
            b1_trapdoor_id_prefix
        );
        Some((b1_trapdoor, b1_matrix))
    }

    fn load_b0_checkpoint(
        &self,
        params: &<M::P as Poly>::Params,
        checkpoint_prefix: &str,
    ) -> Option<(TS::Trapdoor, M)> {
        let dir = self.dir_path.as_path();
        let b0_id_prefix = format!("{checkpoint_prefix}_b0");
        let b0_trapdoor_id_prefix = format!("{checkpoint_prefix}_b0_trapdoor");
        info!(
            "Trying B0 checkpoint load from {} (matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b0_id_prefix,
            b0_trapdoor_id_prefix
        );
        let b0_bytes = if let Some(bytes) = read_bytes_from_multi_batch(dir, &b0_id_prefix, 0) {
            bytes
        } else {
            info!(
                "B0 checkpoint matrix not found (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b0_id_prefix
            );
            return None;
        };
        let trapdoor_bytes =
            if let Some(bytes) = read_bytes_from_multi_batch(dir, &b0_trapdoor_id_prefix, 0) {
                bytes
            } else {
                info!(
                    "B0 checkpoint trapdoor not found (dir={}, id_prefix={}, index=0)",
                    dir.display(),
                    b0_trapdoor_id_prefix
                );
                return None;
            };
        let b0_trapdoor = if let Some(td) = TS::trapdoor_from_bytes(params, &trapdoor_bytes) {
            td
        } else {
            warn!(
                "Failed to decode B0 trapdoor bytes (dir={}, id_prefix={}, index=0)",
                dir.display(),
                b0_trapdoor_id_prefix
            );
            return None;
        };
        let b0_matrix = M::from_compact_bytes(params, &b0_bytes);
        info!(
            "Loaded B0 checkpoint (dir={}, matrix_id_prefix={}, trapdoor_id_prefix={})",
            dir.display(),
            b0_id_prefix,
            b0_trapdoor_id_prefix
        );
        Some((b0_trapdoor, b0_matrix))
    }

    pub fn load_b0_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        let checkpoint_prefix = self.aux_checkpoint_prefix(params);
        let b0_id_prefix = format!("{checkpoint_prefix}_b0");
        let dir = self.dir_path.as_path();
        let bytes = read_bytes_from_multi_batch(dir, &b0_id_prefix, 0)?;
        Some(M::from_compact_bytes(params, &bytes))
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!("Sampling LUT and gate auxiliary matrices");
        let start = Instant::now();
        let chunk_size = crate::env::lut_preimage_chunk_size();
        let checkpoint_prefix = self.aux_checkpoint_prefix(params);
        let checkpoint_index = self.load_checkpoint_index();

        let lut_ids: Vec<usize> = self.lut_state.iter().map(|entry| *entry.key()).collect();
        let mut lut_entries = Vec::with_capacity(lut_ids.len());
        for &lut_id in &lut_ids {
            if let Some((_, plt)) = self.lut_state.remove(&lut_id) {
                lut_entries.push((lut_id, plt));
            }
        }

        let total_lut_rows: usize = lut_entries.iter().map(|(_, plt)| plt.len()).sum();
        debug!(
            "LUT sampling start: lut_count={}, total_rows={}, chunk_size={}",
            lut_entries.len(),
            total_lut_rows,
            chunk_size
        );
        let gate_ids: Vec<GateId> = self.gate_state.iter().map(|entry| *entry.key()).collect();
        let mut gate_entries = Vec::with_capacity(gate_ids.len());
        for &gate_id in &gate_ids {
            if let Some((_, state)) = self.gate_state.remove(&gate_id) {
                gate_entries.push((gate_id, state));
            }
        }
        let gate_v_chunk_count = 1usize;
        let gate_vx_chunk_count = small_gadget_chunk_count::<M>(params);
        let has_resume_candidates = Self::has_resume_candidates(
            checkpoint_index.as_ref(),
            &checkpoint_prefix,
            &lut_ids,
            &gate_ids,
            gate_v_chunk_count,
            gate_vx_chunk_count,
        );
        let total_gate_count = gate_entries.len();
        debug!("Gate sampling start: total_gates={}, chunk_size={}", total_gate_count, chunk_size);

        let d = self.d;
        info!("Sampling auxiliary matrices with d = {}", d);
        let mut persist_b0_checkpoint: Option<(M, Vec<u8>)> = None;
        let (b0_trapdoor, b0_matrix, b0_loaded_from_checkpoint) =
            if let Some((b0_trapdoor, b0_matrix)) =
                self.load_b0_checkpoint(params, &checkpoint_prefix)
            {
                info!("Resumed B0 checkpoint with prefix={checkpoint_prefix}");
                (b0_trapdoor, b0_matrix, true)
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let (b0_trapdoor, b0_matrix) = trap_sampler.trapdoor(params, d);
                persist_b0_checkpoint =
                    Some((b0_matrix.clone(), TS::trapdoor_to_bytes(&b0_trapdoor)));
                (b0_trapdoor, b0_matrix, false)
            };
        let mut persist_b1_checkpoint: Option<(M, Vec<u8>)> = None;
        let (b1_trapdoor, b1_matrix, b1_loaded_from_checkpoint) =
            if let Some((b1_trapdoor, b1_matrix)) =
                self.load_b1_checkpoint(params, &checkpoint_prefix)
            {
                info!("Resumed B1 checkpoint with prefix={checkpoint_prefix}");
                (b1_trapdoor, b1_matrix, true)
            } else {
                let trap_sampler = TS::new(params, self.trapdoor_sigma);
                let (b1_trapdoor, b1_matrix) = trap_sampler.trapdoor(params, 2 * d);
                persist_b1_checkpoint =
                    Some((b1_matrix.clone(), TS::trapdoor_to_bytes(&b1_trapdoor)));
                (b1_trapdoor, b1_matrix, false)
            };

        let checkpoint_index_for_resume = if has_resume_candidates &&
            (!b0_loaded_from_checkpoint || !b1_loaded_from_checkpoint)
        {
            warn!(
                "Auxiliary outputs exist but B0/B1 checkpoint is missing (prefix={checkpoint_prefix}, b0_loaded={}, b1_loaded={}); \
resuming is disabled and auxiliary matrices will be resampled from scratch",
                b0_loaded_from_checkpoint, b1_loaded_from_checkpoint
            );
            None
        } else {
            checkpoint_index.as_ref()
        };
        info!(
            "Checkpoint resume {} (dir={}, prefix={})",
            if checkpoint_index_for_resume.is_some() { "enabled" } else { "disabled" },
            self.dir_path.display(),
            checkpoint_prefix
        );
        let checkpoint_part_index_cache = Self::build_part_index_cache(checkpoint_index_for_resume);

        if let Some((b0_matrix_for_save, b0_trapdoor_bytes)) = persist_b0_checkpoint {
            let b0_id_prefix = format!("{checkpoint_prefix}_b0");
            let b0_trapdoor_id_prefix = format!("{checkpoint_prefix}_b0_trapdoor");
            info!(
                "Persisting newly generated B0 checkpoint (matrix_id_prefix={}, trapdoor_id_prefix={})",
                b0_id_prefix, b0_trapdoor_id_prefix
            );
            add_lookup_buffer(get_lookup_buffer(vec![(0, b0_matrix_for_save)], &b0_id_prefix));
            add_lookup_buffer(get_lookup_buffer_bytes(
                vec![(0, b0_trapdoor_bytes)],
                &b0_trapdoor_id_prefix,
            ));
        }
        if let Some((b1_matrix_for_save, b1_trapdoor_bytes)) = persist_b1_checkpoint {
            let b1_id_prefix = format!("{checkpoint_prefix}_b1");
            let b1_trapdoor_id_prefix = format!("{checkpoint_prefix}_b1_trapdoor");
            info!(
                "Persisting newly generated B1 checkpoint (matrix_id_prefix={}, trapdoor_id_prefix={})",
                b1_id_prefix, b1_trapdoor_id_prefix
            );
            add_lookup_buffer(get_lookup_buffer(vec![(0, b1_matrix_for_save)], &b1_id_prefix));
            add_lookup_buffer(get_lookup_buffer_bytes(
                vec![(0, b1_trapdoor_bytes)],
                &b1_trapdoor_id_prefix,
            ));
        }

        // Checkpoint verification phase.
        let mut processed_lut_rows = 0usize;
        let completed_lut_rows_by_id = Self::collect_lut_completed_rows(
            checkpoint_index_for_resume,
            &checkpoint_prefix,
            &lut_ids,
        );
        let lut_plans_with_prefix = lut_entries
            .into_par_iter()
            .map(|(lut_id, plt)| {
                let lut_aux_prefix = self.lut_aux_id_prefix(params, lut_id);
                let completed_rows: HashSet<usize> =
                    if let Some(index_rows) = completed_lut_rows_by_id.get(&lut_id) {
                        plt.entries(params)
                            .map(|(_, (idx, _))| idx)
                            .filter(|idx| index_rows.contains(idx))
                            .collect()
                    } else {
                        HashSet::new()
                    };
                let resumed_rows_for_lut = completed_rows.len();
                (lut_id, plt, completed_rows, resumed_rows_for_lut, lut_aux_prefix)
            })
            .collect::<Vec<_>>();
        lut_plans_with_prefix
            .par_iter()
            .filter(|(_, _, _, resumed_rows_for_lut, _)| *resumed_rows_for_lut > 0)
            .for_each(|(lut_id, _, _, resumed_rows_for_lut, lut_aux_prefix)| {
                info!(
                    "LUT checkpoint resumed: lut_id={}, rows={}, aux_prefix={}",
                    lut_id, resumed_rows_for_lut, lut_aux_prefix
                );
            });
        let resumed_lut_rows = lut_plans_with_prefix
            .par_iter()
            .map(|(_, _, _, resumed_rows_for_lut, _)| *resumed_rows_for_lut)
            .sum::<usize>();
        processed_lut_rows = processed_lut_rows.saturating_add(resumed_lut_rows);
        let lut_plans = lut_plans_with_prefix
            .into_iter()
            .map(|(lut_id, plt, completed_rows, resumed_rows_for_lut, _)| {
                (lut_id, plt, completed_rows, resumed_rows_for_lut)
            })
            .collect::<Vec<_>>();

        let (resumed_gates, gates_by_lut) = gate_entries
            .into_par_iter()
            .fold(
                || (0usize, HashMap::<usize, Vec<(GateId, GateState<M>)>>::new()),
                |(mut resumed, mut grouped), (gate_id, state)| {
                    if Self::gate_checkpoint_complete(
                        checkpoint_index_for_resume,
                        checkpoint_part_index_cache.as_ref(),
                        &checkpoint_prefix,
                        gate_id,
                        gate_v_chunk_count,
                        gate_vx_chunk_count,
                    ) {
                        resumed = resumed.saturating_add(1);
                    } else {
                        grouped.entry(state.lut_id).or_default().push((gate_id, state));
                    }
                    (resumed, grouped)
                },
            )
            .reduce(
                || (0usize, HashMap::<usize, Vec<(GateId, GateState<M>)>>::new()),
                |(left_resumed, mut left_grouped), (right_resumed, right_grouped)| {
                    right_grouped.into_iter().for_each(|(lut_id, mut gates)| {
                        left_grouped.entry(lut_id).or_default().append(&mut gates);
                    });
                    (left_resumed.saturating_add(right_resumed), left_grouped)
                },
            );
        let mut total_gates = resumed_gates;
        info!(
            "Checkpoint verification completed (pending_lut_rows={}, pending_gates={})",
            total_lut_rows.saturating_sub(resumed_lut_rows),
            total_gate_count.saturating_sub(resumed_gates)
        );
        #[cfg(feature = "gpu")]
        let gpu_lut_base_shared =
            self.prepare_gpu_lut_base_device_shared(params, &b1_trapdoor, &b1_matrix);

        for (lut_id, plt, completed_rows, resumed_rows_for_lut) in lut_plans {
            let lut_start = Instant::now();
            let lut_aux_id_prefix = self.lut_aux_id_prefix(params, lut_id);
            let pending_input_indices: Vec<usize> = if completed_rows.is_empty() {
                (0..plt.len()).collect()
            } else {
                let pending_scan_start = Instant::now();
                let pending = (0..plt.len())
                    .into_par_iter()
                    .filter_map(|input_idx| {
                        let x = M::P::from_usize_to_constant(params, input_idx);
                        let (row_idx, _) = plt.get(params, &x).unwrap_or_else(|| {
                            panic!("LUT entry {} missing from 0..len range", input_idx)
                        });
                        (!completed_rows.contains(&row_idx)).then_some(input_idx)
                    })
                    .collect::<Vec<_>>();
                debug!(
                    "Computed pending LUT entries: lut_id={}, pending_entries={}, resumed_rows={}, elapsed={}",
                    lut_id,
                    pending.len(),
                    resumed_rows_for_lut,
                    Self::format_duration(pending_scan_start.elapsed())
                );
                pending
            };
            if pending_input_indices.is_empty() {
                debug!(
                    "LUT {} complete in {} (resumed_rows={})",
                    lut_id,
                    Self::format_duration(lut_start.elapsed()),
                    resumed_rows_for_lut
                );
                continue;
            }
            let w_block_identity = self.derive_w_block_identity(params, lut_id);
            let w_block_gy = self.derive_w_block_gy(params, lut_id);
            let w_block_v = self.derive_w_block_v(params, lut_id);
            let w_block_vx = self.derive_w_block_vx(params, lut_id);
            #[cfg(feature = "gpu")]
            let gpu_lut_shared = self.prepare_gpu_lut_device_shared(lut_id, &gpu_lut_base_shared);

            #[cfg(feature = "gpu")]
            {
                assert!(
                    chunk_size <= gpu_lut_shared.len(),
                    "LUT_PREIMAGE_CHUNK_SIZE must be <= available GPU devices: chunk_size={}, devices={}",
                    chunk_size,
                    gpu_lut_shared.len()
                );
                let sample_lut_batch = |current_batch: &[(usize, i32, M::P)]| {
                    self.sample_lut_preimages(
                        params,
                        lut_id,
                        &lut_aux_id_prefix,
                        &gpu_lut_shared,
                        current_batch,
                    )
                };

                let mut batch: Vec<(usize, i32, M::P)> = Vec::with_capacity(chunk_size);
                let mut pending_store_jobs: Option<Vec<CompactBytesJob<M>>> = None;
                for input_idx in pending_input_indices.iter().copied() {
                    let device_slot = batch.len();
                    let device_shared = &gpu_lut_shared[device_slot];
                    let x = M::P::from_usize_to_constant(device_shared.params, input_idx);
                    let (idx, y_poly) = plt.get(device_shared.params, &x).unwrap_or_else(|| {
                        panic!("LUT entry {} missing from 0..len range", input_idx)
                    });
                    batch.push((idx, device_shared.device_id, y_poly));

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
                        processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                        let pct = if total_lut_rows == 0 {
                            100.0
                        } else {
                            (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                        };
                        debug!(
                            "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                            processed_lut_rows,
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
                                previous_jobs
                                    .into_par_iter()
                                    .for_each(CompactBytesJob::wait_then_store);
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
                    processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                    let pct = if total_lut_rows == 0 {
                        100.0
                    } else {
                        (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                    };
                    debug!(
                        "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                        processed_lut_rows,
                        total_lut_rows,
                        Self::format_duration(start.elapsed())
                    );
                }
                if let Some(previous_jobs) = pending_store_jobs.take() {
                    previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                }
            }

            #[cfg(not(feature = "gpu"))]
            {
                let sample_lut_batch = |current_batch: &[(usize, M::P)]| {
                    self.sample_lut_preimages(
                        params,
                        lut_id,
                        &lut_aux_id_prefix,
                        &b1_trapdoor,
                        &b1_matrix,
                        &w_block_identity,
                        &w_block_gy,
                        &w_block_v,
                        &w_block_vx,
                        current_batch,
                    )
                };

                let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);
                let mut pending_store_jobs: Option<Vec<CompactBytesJob<M>>> = None;
                for input_idx in pending_input_indices.iter().copied() {
                    let x = M::P::from_usize_to_constant(params, input_idx);
                    let (idx, y_poly) = plt.get(params, &x).unwrap_or_else(|| {
                        panic!("LUT entry {} missing from 0..len range", input_idx)
                    });
                    batch.push((idx, y_poly));

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
                        processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                        let pct = if total_lut_rows == 0 {
                            100.0
                        } else {
                            (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                        };
                        debug!(
                            "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                            processed_lut_rows,
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
                                previous_jobs
                                    .into_par_iter()
                                    .for_each(CompactBytesJob::wait_then_store);
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
                    processed_lut_rows = processed_lut_rows.saturating_add(current_batch.len());
                    let pct = if total_lut_rows == 0 {
                        100.0
                    } else {
                        (processed_lut_rows as f64) * 100.0 / (total_lut_rows as f64)
                    };
                    debug!(
                        "LUT rows processed: {}/{} ({pct:.1}%), elapsed={}",
                        processed_lut_rows,
                        total_lut_rows,
                        Self::format_duration(start.elapsed())
                    );
                }
                if let Some(previous_jobs) = pending_store_jobs.take() {
                    previous_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                }
            }
            #[cfg(feature = "gpu")]
            drop(gpu_lut_shared);
            drop(w_block_vx);
            drop(w_block_v);
            drop(w_block_gy);
            drop(w_block_identity);
            debug!(
                "LUT {} complete in {} (resumed_rows={})",
                lut_id,
                Self::format_duration(lut_start.elapsed()),
                resumed_rows_for_lut
            );
        }
        #[cfg(feature = "gpu")]
        drop(gpu_lut_base_shared);

        if total_gate_count == 0 {
            info!("No gate auxiliary matrices to sample");
            info!(
                "Sampled LUT and gate auxiliary matrices in {} (0 gates, resumed_lut_rows={})",
                Self::format_duration(start.elapsed()),
                resumed_lut_rows
            );
            return;
        }

        let chunk_size = crate::env::ggh15_gate_parallelism();
        info!(
            "GGH15 gate preimage parallelism uses Rayon global pool (GGH15_GATE_PARALLELISM={})",
            chunk_size
        );
        #[cfg(feature = "gpu")]
        let gpu_gate_base_shared = self.prepare_gpu_gate_base_device_shared(
            params,
            &b0_trapdoor,
            &b0_matrix,
            &b1_trapdoor,
            &b1_matrix,
        );

        for (lut_id, mut gates) in gates_by_lut {
            let lut_gate_start = Instant::now();
            let w_block_identity = self.derive_w_block_identity(params, lut_id);
            let w_block_gy = self.derive_w_block_gy(params, lut_id);
            let w_block_v = self.derive_w_block_v(params, lut_id);
            let w_block_vx = self.derive_w_block_vx(params, lut_id);
            #[cfg(feature = "gpu")]
            let w_block_identity_by_device =
                self.copy_matrix_to_gpu_gate_devices(&w_block_identity, &gpu_gate_base_shared);
            #[cfg(feature = "gpu")]
            let w_block_gy_by_device =
                self.copy_matrix_to_gpu_gate_devices(&w_block_gy, &gpu_gate_base_shared);
            #[cfg(feature = "gpu")]
            let w_block_v_by_device =
                self.copy_matrix_to_gpu_gate_devices(&w_block_v, &gpu_gate_base_shared);
            #[cfg(feature = "gpu")]
            let w_block_vx_by_device =
                self.copy_matrix_to_gpu_gate_devices(&w_block_vx, &gpu_gate_base_shared);
            #[cfg(feature = "gpu")]
            let gpu_gate_shared = self.prepare_gpu_gate_device_shared(
                &gpu_gate_base_shared,
                w_block_identity_by_device,
                w_block_gy_by_device,
                w_block_v_by_device,
                w_block_vx_by_device,
            );

            #[cfg(feature = "gpu")]
            let sample_gate_batch = |pending: Vec<(GateId, GateState<M>)>| {
                self.sample_gate_preimages_batch(params, lut_id, pending, &gpu_gate_shared)
            };
            #[cfg(not(feature = "gpu"))]
            let sample_gate_batch = |pending: Vec<(GateId, GateState<M>)>| {
                self.sample_gate_preimages_batch(
                    params,
                    lut_id,
                    pending,
                    &b0_trapdoor,
                    &b0_matrix,
                    &b1_trapdoor,
                    &b1_matrix,
                    &w_block_identity,
                    &w_block_gy,
                    &w_block_v,
                    &w_block_vx,
                )
            };

            while !gates.is_empty() {
                let take = gates.len().min(chunk_size);
                let pending: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();

                if !pending.is_empty() {
                    let pending_len = pending.len();
                    total_gates = total_gates.saturating_add(pending_len);
                    let gate_preimage_batch_start = Instant::now();
                    debug!(
                        "Sampling gate preimages batch started: lut_id={}, batch_size={}",
                        lut_id, pending_len
                    );
                    sample_gate_batch(pending);
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
                    (total_gates as f64) * 100.0 / (total_gate_count as f64)
                };
                debug!(
                    "Gates processed: {}/{} ({pct:.1}%), elapsed={}",
                    total_gates,
                    total_gate_count,
                    Self::format_duration(start.elapsed())
                );
            }
            drop(w_block_vx);
            drop(w_block_v);
            drop(w_block_gy);
            drop(w_block_identity);
            debug!(
                "Gate group for LUT {} complete in {}",
                lut_id,
                Self::format_duration(lut_gate_start.elapsed())
            );
        }

        info!(
            "Sampled LUT and gate auxiliary matrices in {} ({} gates, resumed_lut_rows={}, resumed_gates={})",
            Self::format_duration(start.elapsed()),
            total_gates,
            resumed_lut_rows,
            resumed_gates
        );
    }
}

impl<M, US, HS, TS> PltEvaluator<BggPublicKey<M>> for GGH15BGGPubKeyPltEvaluator<M, US, HS, TS>
where
    M: PolyMatrix + Send + Sync + 'static,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        _: &BggPublicKey<M>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggPublicKey<M> {
        let d = input.matrix.row_size();
        debug!("Starting public lookup for gate {}", gate_id);
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());

        let hash_sampler = HS::new();
        let a_out = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_gate_a_out_{}", gate_id),
            d,
            d * params.modulus_digits(),
            DistType::FinRingDist,
        );
        let output_pubkey = BggPublicKey { matrix: a_out, reveal_plaintext: true };
        self.gate_state.insert(
            gate_id,
            GateState {
                lut_id,
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                _m: PhantomData,
            },
        );
        debug!("Public lookup for gate {} recorded", gate_id);
        output_pubkey
    }
}

#[derive(Debug, Clone)]
pub struct GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub hash_key: [u8; 32],
    pub dir_path: PathBuf,
    pub checkpoint_prefix: String,
    c_b0_by_params: Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)>,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(
        hash_key: [u8; 32],
        dir_path: PathBuf,
        checkpoint_prefix: String,
        params: &<M::P as Poly>::Params,
        c_b0: M,
    ) -> Self {
        let c_b0_compact_bytes = c_b0.into_compact_bytes();
        let mut c_b0_by_params: Vec<(<<M as PolyMatrix>::P as Poly>::Params, Arc<M>)> = Vec::new();

        #[cfg(feature = "gpu")]
        {
            let mut device_ids = params.device_ids();
            if device_ids.is_empty() {
                device_ids.push(0);
            }
            for device_id in device_ids {
                let local_params = params.params_for_device(device_id);
                let local_c_b0 =
                    Arc::new(M::from_compact_bytes(&local_params, &c_b0_compact_bytes));
                c_b0_by_params.push((local_params, local_c_b0));
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            c_b0_by_params.push((
                params.clone(),
                Arc::new(M::from_compact_bytes(params, &c_b0_compact_bytes)),
            ));
        }

        Self { hash_key, dir_path, checkpoint_prefix, c_b0_by_params, _hs: PhantomData }
    }

    fn c_b0_for_params(&self, params: &<M::P as Poly>::Params) -> Arc<M> {
        if let Some((_, cached)) =
            self.c_b0_by_params.iter().find(|(cached_params, _)| cached_params == params)
        {
            return Arc::clone(cached);
        }
        panic!("c_b0 for params not found in preloaded cache");
    }
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<BggEncoding<M> as Evaluable>::Params,
        plt: &PublicLut<<BggEncoding<M> as Evaluable>::P>,
        _: &BggEncoding<M>,
        input: &BggEncoding<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> BggEncoding<M> {
        let public_lookup_started = Instant::now();
        debug!(
            "GGH15BGGEncodingPltEvaluator::public_lookup start: gate_id={}, lut_id={}",
            gate_id, lut_id
        );
        let x = input
            .plaintext
            .as_ref()
            .expect("the BGG encoding should reveal plaintext for public lookup");
        let (k, y) = plt.get(params, x).unwrap_or_else(|| {
            panic!("{:?} not found in LUT for gate {}", x.to_const_int(), gate_id)
        });

        let dir = std::path::Path::new(&self.dir_path);
        let checkpoint_prefix = &self.checkpoint_prefix;
        let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
        let lut_aux_row_id = format!("{lut_aux_prefix}_idx{k}");

        let hash_sampler = HS::new();
        let d = input.pubkey.matrix.row_size();
        let m_g = d * params.modulus_digits();
        let k_small = small_gadget_chunk_count::<M>(params);

        let preimage_gate2_identity = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_identity for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_identity.col_size(),
            m_g,
            "preimage_gate2_identity must have m_g columns"
        );
        let mut c_const_rhs = preimage_gate2_identity;

        let preimage_gate2_gy = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_gy for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_gy.col_size(),
            m_g,
            "preimage_gate2_gy must have m_g columns"
        );
        let gy = M::gadget_matrix(params, d) * y.clone();
        let gy_decomposed = gy.decompose();
        drop(gy);
        let gy_term = preimage_gate2_gy * gy_decomposed;
        c_const_rhs.add_in_place(&gy_term);
        drop(gy_term);

        let v_idx = derive_lut_v_idx_from_hash::<M, HS>(params, self.hash_key, lut_id, k, d);
        let preimage_gate2_v = read_matrix_from_chunks::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id),
            m_g,
            1,
            "preimage_gate2_v",
        );
        debug_assert_eq!(
            preimage_gate2_v.col_size(),
            m_g,
            "preimage_gate2_v must have m_g columns"
        );
        let v_term = preimage_gate2_v * &v_idx;
        c_const_rhs.add_in_place(&v_term);
        drop(v_term);

        let mut vx_product_acc: Option<M> = None;
        for chunk_idx in 0..k_small {
            let preimage_gate2_vx_chunk = read_matrix_from_multi_batch::<M>(
                params,
                dir,
                &format!("{checkpoint_prefix}_preimage_gate2_vx_{}_chunk{}", gate_id, chunk_idx),
                0,
            )
            .unwrap_or_else(|| {
                panic!("preimage_gate2_vx chunk {} for gate {} not found", chunk_idx, gate_id)
            });
            debug_assert_eq!(
                preimage_gate2_vx_chunk.col_size(),
                m_g,
                "preimage_gate2_vx chunk must have m_g columns"
            );
            let x_identity_decomposed_chunk =
                M::small_decomposed_identity_chunk_from_scalar(params, m_g, x, chunk_idx, k_small);
            let vx_chunk_product = preimage_gate2_vx_chunk * &x_identity_decomposed_chunk;
            if let Some(acc) = vx_product_acc.as_mut() {
                acc.add_in_place(&vx_chunk_product);
            } else {
                vx_product_acc = Some(vx_chunk_product);
            }
        }
        let vx_product_acc =
            vx_product_acc.expect("gate2_vx chunk accumulation must have at least one chunk");
        let vx_term = vx_product_acc * &v_idx;
        c_const_rhs.add_in_place(&vx_term);
        drop(vx_term);

        let preimage_lut = read_matrix_from_multi_batch::<M>(params, dir, &lut_aux_row_id, 0)
            .unwrap_or_else(|| panic!("preimage_lut (index {}) for lut {} not found", k, lut_id));
        c_const_rhs = c_const_rhs - preimage_lut;

        let preimage_gate1 = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate1 for gate {} not found", gate_id));
        let c_b0 = self.c_b0_for_params(params);
        let sg_times_b1 = c_b0.as_ref() * &preimage_gate1;
        let c_const = sg_times_b1 * c_const_rhs;

        let u_g_decomposed = hash_sampler.sample_hash_decomposed(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            DistType::FinRingDist,
        );
        debug!(
            "Derived decomposed u_g_matrix for gate encoding: gate_id={}, lut_id={}, rows={}, cols={}",
            gate_id,
            lut_id,
            u_g_decomposed.row_size(),
            u_g_decomposed.col_size()
        );
        let c_x_randomized = input.vector.clone() * u_g_decomposed * v_idx;
        debug!("Computed c_x_randomized for gate encoding: gate_id={}, lut_id={}", gate_id, lut_id);
        let c_out = c_const + c_x_randomized;
        debug!("Computed c_out for gate encoding: gate_id={}, lut_id={}", gate_id, lut_id);
        let out_pubkey = BggPublicKey {
            matrix: hash_sampler.sample_hash(
                params,
                self.hash_key,
                format!("ggh15_gate_a_out_{}", gate_id),
                d,
                m_g,
                DistType::FinRingDist,
            ),
            reveal_plaintext: true,
        };
        let out = BggEncoding::new(c_out, out_pubkey, Some(y));
        debug!(
            "GGH15BGGEncodingPltEvaluator::public_lookup end: gate_id={}, lut_id={}, elapsed_ms={:.3}",
            gate_id,
            lut_id,
            public_lookup_started.elapsed().as_secs_f64() * 1000.0
        );
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        bgg::sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        circuit::PolyCircuit,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
    };
    use keccak_asm::Keccak256;
    use std::{fs, path::Path};

    fn setup_lsb_constant_binary_plt(t_n: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::<DCRTPoly>::new_from_usize_range(
            params,
            t_n,
            |params, k| (k, DCRTPoly::from_usize_to_lsb(params, k)),
            None,
        )
    }

    const SIGMA: f64 = 4.578;

    #[test]
    fn test_small_decomposed_identity_chunk_equivalence() {
        let params = DCRTPolyParams::default();
        let d = 2usize;
        let m_g = d * params.modulus_digits();
        let k_small = super::small_gadget_chunk_count::<DCRTPolyMatrix>(&params);
        let x = DCRTPoly::from_usize_to_constant(&params, 13);

        let full = DCRTPolyMatrix::identity(&params, m_g, Some(x.clone())).small_decompose();
        let x_digit_decomposed = DCRTPolyMatrix::identity(&params, 1, Some(x)).small_decompose();
        let x_digit_by_chunk =
            (0..k_small).map(|digit| x_digit_decomposed.entry(digit, 0)).collect::<Vec<_>>();

        for chunk_idx in 0..k_small {
            let expected = full.slice(chunk_idx * m_g, (chunk_idx + 1) * m_g, 0, m_g);
            let actual = DCRTPolyMatrix::small_decomposed_identity_chunk(
                &params,
                m_g,
                chunk_idx,
                k_small,
                &x_digit_by_chunk,
            );
            assert_eq!(actual, expected, "chunk mismatch at chunk_idx={chunk_idx}");
        }
    }

    #[tokio::test]
    #[sequential_test::sequential]
    async fn test_ggh15_plt_eval_single_input() {
        let _storage_lock = storage_test_lock().await;
        let _ = tracing_subscriber::fmt::try_init();

        let params = DCRTPolyParams::default();
        let plt = setup_lsb_constant_binary_plt(16, &params);

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
        let insert_1_to_s = false;
        let plt_pubkey_evaluator =
            GGH15BGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(key, d, SIGMA, error_sigma, dir_path.into(), insert_1_to_s);

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

        let expected_plaintext = plt.get(&params, &plaintexts[0]).unwrap().1;
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
        let plt = setup_lsb_constant_binary_plt(16, &params);

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
        let insert_1_to_s = false;
        let plt_pubkey_evaluator =
            GGH15BGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(key, d, SIGMA, error_sigma, dir_path.into(), insert_1_to_s);

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

            let expected_plaintext = plt.get(&params, &plaintexts[i]).unwrap().1;
            assert_eq!(result_encoding_i.plaintext.clone().unwrap(), expected_plaintext.clone());

            let expected_vector = s_vec.clone() *
                (result_encoding_i.pubkey.matrix.clone() -
                    (DCRTPolyMatrix::gadget_matrix(&params, d) * expected_plaintext));
            assert_eq!(result_encoding_i.vector, expected_vector);
        }
    }
}
