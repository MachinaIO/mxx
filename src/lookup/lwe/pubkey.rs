#[cfg(feature = "gpu")]
#[path = "pubkey_gpu.rs"]
mod gpu;

#[cfg(not(feature = "gpu"))]
use crate::bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate};
use crate::{
    bgg::public_key::BggPublicKey,
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler},
    storage::{
        read::read_bytes_from_multi_batch,
        write::{
            BatchLookupBuffer, GlobalTableIndex, add_lookup_buffer, get_lookup_buffer,
            get_lookup_buffer_bytes,
        },
    },
};
use dashmap::DashMap;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::read_to_string,
    marker::PhantomData,
    path::PathBuf,
    sync::Arc,
};
use tracing::{info, warn};

use super::{
    column_chunk_count, derive_a_lt_matrix, derive_a_lt_matrix_for_slot, derive_k_low_for_slot,
    k_high_checkpoint_prefix_for_slot, k_high_row_checkpoint_prefix_for_slot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LweLookupGateKey {
    gate_id: GateId,
    slot_idx: usize,
}

impl LweLookupGateKey {
    pub(super) fn new(gate_id: GateId, slot_idx: Option<usize>) -> Self {
        Self { gate_id, slot_idx: slot_idx.unwrap_or(0) }
    }

    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(super) fn gate_id(self) -> GateId {
        self.gate_id
    }

    pub(super) fn slot_idx_option(self) -> Option<usize> {
        Some(self.slot_idx)
    }
}

#[cfg(not(feature = "gpu"))]
use super::{column_chunk_bounds, derive_k_low_chunk};
#[cfg(not(feature = "gpu"))]
use crate::poly::PolyParams;
#[cfg(not(feature = "gpu"))]
use num_bigint::BigUint;

#[derive(Debug)]
pub(super) struct GateState<M>
where
    M: PolyMatrix,
{
    pub(super) lut_id: usize,
    pub(super) input_pubkey_bytes: Vec<u8>,
    pub(super) output_pubkey_bytes: Vec<u8>,
    pub(super) _m: PhantomData<M>,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) struct CompactBytesJob {
    pub(super) id_prefix: String,
    pub(super) matrices: Vec<(usize, Vec<u8>)>,
}

impl CompactBytesJob {
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(super) fn new<M>(id_prefix: String, matrices: Vec<(usize, M)>) -> Self
    where
        M: PolyMatrix,
    {
        Self {
            id_prefix,
            matrices: matrices
                .into_iter()
                .map(|(idx, matrix)| (idx, matrix.into_compact_bytes()))
                .collect(),
        }
    }

    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(super) fn into_lookup_buffer(self) -> BatchLookupBuffer {
        let mut payloads = Vec::with_capacity(self.matrices.len());
        let mut max_len = 0usize;
        for (idx, bytes) in self.matrices {
            max_len = max_len.max(bytes.len());
            payloads.push((idx, bytes));
        }
        let padded_len = max_len.saturating_add(16);
        for (_, bytes) in &mut payloads {
            if bytes.len() < padded_len {
                bytes.resize(padded_len, 0);
            }
        }
        get_lookup_buffer_bytes(payloads, &self.id_prefix)
    }

    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(super) fn wait_then_store(self) {
        let _ = add_lookup_buffer(self.into_lookup_buffer());
    }
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn compact_bytes_job_total(jobs: &[CompactBytesJob]) -> u64 {
    jobs.iter().flat_map(|job| job.matrices.iter()).fold(0u64, |total, (_, bytes)| {
        total
            .checked_add(u64::try_from(bytes.len()).expect("compact_bytes length overflowed u64"))
            .expect("compact_bytes total overflowed u64")
    })
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn load_checkpoint_index(dir_path: &PathBuf) -> Option<GlobalTableIndex> {
    let index_path = dir_path.join("lookup_tables.index");
    match read_to_string(&index_path) {
        Ok(index_data) => match serde_json::from_str::<GlobalTableIndex>(&index_data) {
            Ok(global_index) => Some(global_index),
            Err(err) => {
                warn!("Failed to parse checkpoint index {}: {}", index_path.display(), err);
                None
            }
        },
        Err(_) => None,
    }
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn checkpoint_has_index(
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

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn build_part_index_cache(
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

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(super) fn k_high_row_checkpoint_complete(
    checkpoint_index: Option<&GlobalTableIndex>,
    part_index_cache: Option<&HashMap<String, HashSet<usize>>>,
    gate_id: GateId,
    lut_id: usize,
    row_size: usize,
    modulus_digits: usize,
    lut_entry_idx: usize,
    slot_idx: Option<usize>,
) -> bool {
    let base_prefix = k_high_checkpoint_prefix_for_slot(gate_id, lut_id, slot_idx);
    if checkpoint_has_index(checkpoint_index, part_index_cache, &base_prefix, lut_entry_idx) {
        return true;
    }

    let chunk_count = column_chunk_count(row_size * modulus_digits);
    (0..chunk_count).into_par_iter().all(|chunk_idx| {
        checkpoint_has_index(
            checkpoint_index,
            part_index_cache,
            &super::column_chunk_id_prefix(&base_prefix, chunk_idx),
            lut_entry_idx,
        ) || checkpoint_has_index(
            checkpoint_index,
            part_index_cache,
            &super::column_chunk_id_prefix(
                &k_high_row_checkpoint_prefix_for_slot(gate_id, lut_id, lut_entry_idx, slot_idx),
                chunk_idx,
            ),
            0,
        )
    })
}

#[derive(Debug)]
pub struct LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub hash_key: [u8; 32],
    pub trap_sampler: ST,
    pub pub_matrix: Arc<M>,
    pub trapdoor: Arc<ST::Trapdoor>,
    pub dir_path: PathBuf,
    pub(super) lut_state: DashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
    pub(super) gate_state: DashMap<LweLookupGateKey, GateState<M>>,
    _sh: PhantomData<SH>,
    _st: PhantomData<ST>,
}

impl<M, SH, ST> PltEvaluator<BggPublicKey<M>> for LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
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
        self.public_lookup_for_slot(params, plt, input, gate_id, lut_id, None)
    }
}

impl<M, SH, ST> LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn public_lookup_for_slot(
        &self,
        params: &<BggPublicKey<M> as Evaluable>::Params,
        plt: &PublicLut<<BggPublicKey<M> as Evaluable>::P>,
        input: &BggPublicKey<M>,
        gate_id: GateId,
        lut_id: usize,
        slot_idx: Option<usize>,
    ) -> BggPublicKey<M> {
        let row_size = input.matrix.row_size();
        let slot_key = LweLookupGateKey::new(gate_id, slot_idx);
        let a_lt = derive_a_lt_matrix_for_slot::<M, SH>(
            params,
            row_size,
            self.hash_key,
            gate_id,
            slot_key.slot_idx_option(),
        );
        self.lut_state.entry(lut_id).or_insert_with(|| plt.clone());
        self.gate_state.insert(
            slot_key,
            GateState {
                lut_id,
                input_pubkey_bytes: input.matrix.to_compact_bytes(),
                output_pubkey_bytes: a_lt.to_compact_bytes(),
                _m: PhantomData,
            },
        );
        BggPublicKey { matrix: a_lt, reveal_plaintext: true }
    }
}

impl<M, SH, ST> LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(
        hash_key: [u8; 32],
        trap_sampler: ST,
        pub_matrix: Arc<M>,
        trapdoor: Arc<ST::Trapdoor>,
        dir_path: PathBuf,
    ) -> Self {
        Self {
            hash_key,
            trap_sampler,
            pub_matrix,
            trapdoor,
            dir_path,
            lut_state: DashMap::new(),
            gate_state: DashMap::new(),
            _sh: PhantomData,
            _st: PhantomData,
        }
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        info!("Sampling LWE LUT auxiliary matrices");

        let lut_ids: Vec<usize> = self.lut_state.iter().map(|entry| *entry.key()).collect();
        let mut lut_entries: HashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>> =
            HashMap::with_capacity(lut_ids.len());
        for lut_id in lut_ids {
            if let Some((_, plt)) = self.lut_state.remove(&lut_id) {
                lut_entries.insert(lut_id, plt);
            }
        }

        let gate_keys: Vec<LweLookupGateKey> =
            self.gate_state.iter().map(|entry| *entry.key()).collect();
        let mut gate_entries = Vec::with_capacity(gate_keys.len());
        for gate_key in gate_keys {
            if let Some((_, state)) = self.gate_state.remove(&gate_key) {
                gate_entries.push((gate_key, state));
            }
        }

        if gate_entries.is_empty() {
            info!("No LWE gate auxiliary matrices to sample");
            return;
        }

        #[cfg(feature = "gpu")]
        {
            gpu::sample_aux_matrices_gpu(self, params, gate_entries, lut_entries);
            return;
        }

        #[cfg(not(feature = "gpu"))]
        {
            sample_aux_matrices_cpu::<M, SH, ST>(self, params, gate_entries, lut_entries);
        }
    }

    pub fn load_pub_matrix_checkpoint(&self, params: &<M::P as Poly>::Params) -> Option<M> {
        let bytes = read_bytes_from_multi_batch(self.dir_path.as_path(), "LWE_PUB_MATRIX", 0)?;
        Some(M::from_compact_bytes(params, &bytes))
    }
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
fn sample_aux_matrices_cpu<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    params: &<M::P as Poly>::Params,
    gate_entries: Vec<(LweLookupGateKey, GateState<M>)>,
    lut_entries: HashMap<usize, PublicLut<<BggPublicKey<M> as Evaluable>::P>>,
) where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    let total_gates = gate_entries.len();
    for (gate_key, gate_state) in gate_entries {
        let gate_id = gate_key.gate_id;
        let slot_idx = gate_key.slot_idx_option();
        let plt = lut_entries.get(&gate_state.lut_id).unwrap_or_else(|| {
            panic!(
                "LUT state for lut_id {} not found while sampling gate {}",
                gate_state.lut_id, gate_id
            )
        });
        let input_pubkey = M::from_compact_bytes(params, &gate_state.input_pubkey_bytes);
        let output_pubkey = M::from_compact_bytes(params, &gate_state.output_pubkey_bytes);
        let buffer = sample_k_high_buffer::<M, SH, ST, _>(
            plt,
            params,
            evaluator.hash_key,
            &evaluator.trap_sampler,
            &evaluator.pub_matrix,
            &evaluator.trapdoor,
            &input_pubkey,
            &output_pubkey,
            gate_id,
            gate_state.lut_id,
            slot_idx,
        );
        add_lookup_buffer(buffer);
    }

    info!("Sampled {} LWE gate auxiliary matrices", total_gates);
}

#[cfg_attr(feature = "gpu", allow(dead_code))]
fn sample_k_high_buffer<M, SH, ST, P>(
    plt: &PublicLut<P>,
    params: &<M::P as Poly>::Params,
    hash_key: [u8; 32],
    trap_sampler: &ST,
    pub_matrix: &M,
    trapdoor: &ST::Trapdoor,
    a_z: &M,
    a_lt: &M,
    gate_id: GateId,
    lut_id: usize,
    slot_idx: Option<usize>,
) -> BatchLookupBuffer
where
    P: Poly + 'static,
    M: PolyMatrix<P = P> + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    let row_size = pub_matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let chunk_size = crate::env::lut_preimage_chunk_size().max(1);
    info!(
        "start collecting LWE k_high matrices for gate {} (entries={}, parallel_batch={})",
        gate_id,
        plt.len(),
        chunk_size
    );
    let mut k_high_by_entry = Vec::with_capacity(plt.len());
    let mut batch = Vec::with_capacity(chunk_size);
    for (x_k, (k, y_k)) in plt.entries(params) {
        batch.push((x_k, k, y_k));
        if batch.len() >= chunk_size {
            let chunk = std::mem::take(&mut batch);
            let mut partial = chunk
                .into_par_iter()
                .map(|(x_k, k, y_k)| {
                    let x_k_usize = usize::try_from(x_k).expect("LUT input must fit in usize");
                    let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
                    let x_poly = P::from_usize_to_constant(params, x_k_usize);
                    let y_poly = P::from_elem_to_constant(params, &y_k);
                    let ext_matrix = a_z.clone() - &(gadget.clone() * x_poly);
                    let target = a_lt.clone() - &(gadget.clone() * y_poly);
                    let k_low = derive_k_low_for_slot::<M, SH>(
                        params, row_size, hash_key, gate_id, lut_id, k_usize, slot_idx,
                    );
                    let adjusted_target = target - &(ext_matrix * &k_low);
                    (k_usize, trap_sampler.preimage(params, trapdoor, pub_matrix, &adjusted_target))
                })
                .collect::<Vec<_>>();
            k_high_by_entry.append(&mut partial);
            batch = Vec::with_capacity(chunk_size);
        }
    }
    if !batch.is_empty() {
        let mut partial = batch
            .into_par_iter()
            .map(|(x_k, k, y_k)| {
                let x_k_usize = usize::try_from(x_k).expect("LUT input must fit in usize");
                let k_usize = usize::try_from(k).expect("LUT row index must fit in usize");
                let x_poly = P::from_usize_to_constant(params, x_k_usize);
                let y_poly = P::from_elem_to_constant(params, &y_k);
                let ext_matrix = a_z.clone() - &(gadget.clone() * x_poly);
                let target = a_lt.clone() - &(gadget.clone() * y_poly);
                let k_low = derive_k_low_for_slot::<M, SH>(
                    params, row_size, hash_key, gate_id, lut_id, k_usize, slot_idx,
                );
                let adjusted_target = target - &(ext_matrix * &k_low);
                (k_usize, trap_sampler.preimage(params, trapdoor, pub_matrix, &adjusted_target))
            })
            .collect::<Vec<_>>();
        k_high_by_entry.append(&mut partial);
    }
    info!("finish collecting LWE k_high matrices for gate {}", gate_id);
    get_lookup_buffer(
        k_high_by_entry,
        &k_high_checkpoint_prefix_for_slot(gate_id, lut_id, slot_idx),
    )
}

#[cfg(not(feature = "gpu"))]
fn sample_bench_preimage_chunk<M, SH, ST>(
    evaluator: &LWEBGGPubKeyPltEvaluator<M, SH, ST>,
    params: &<M::P as Poly>::Params,
    gate_id: GateId,
    lut_id: usize,
    lut_entry_idx: usize,
    x_k_usize: usize,
    y_k_usize: usize,
    chunk_idx: usize,
) -> M
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    let row_size = evaluator.pub_matrix.row_size();
    let gadget = M::gadget_matrix(params, row_size);
    let total_cols = row_size * params.modulus_digits();
    let (col_start, col_len) = column_chunk_bounds(total_cols, chunk_idx);
    let x_poly = M::P::from_usize_to_constant(params, x_k_usize);
    let y_poly = M::P::from_usize_to_constant(params, y_k_usize);
    let a_z = SH::new().sample_hash(
        params,
        evaluator.hash_key,
        b"lwe_bench_input_pubkey",
        row_size,
        total_cols,
        crate::sampler::DistType::FinRingDist,
    );
    let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, evaluator.hash_key, gate_id);
    let ext_matrix = a_z - &(gadget.clone() * x_poly);
    let target = a_lt.slice_columns(col_start, col_start + col_len) -
        &(gadget.slice_columns(col_start, col_start + col_len) * y_poly);
    let k_low_chunk = derive_k_low_chunk::<M, SH>(
        params,
        row_size,
        evaluator.hash_key,
        gate_id,
        lut_id,
        lut_entry_idx,
        chunk_idx,
    );
    let adjusted_target = target - &(ext_matrix * &k_low_chunk);
    evaluator.trap_sampler.preimage(
        params,
        evaluator.trapdoor.as_ref(),
        evaluator.pub_matrix.as_ref(),
        &adjusted_target,
    )
}

#[cfg(not(feature = "gpu"))]
impl<M, SH, ST> PublicLutSampleAuxBenchEstimator<M> for LWEBGGPubKeyPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    type Params = <M::P as Poly>::Params;

    fn estimate_public_lut_sample_aux_matrices(
        &self,
        params: &Self::Params,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> SampleAuxBenchEstimate {
        let total_preimages = total_lut_entries
            .checked_mul(total_lut_gates)
            .expect("total sampled LWE preimages overflowed usize");
        if total_preimages == 0 {
            return SampleAuxBenchEstimate::default();
        }
        let chunk_count = column_chunk_count(self.pub_matrix.row_size() * params.modulus_digits());
        let start = std::time::Instant::now();
        let chunk =
            sample_bench_preimage_chunk::<M, SH, ST>(self, params, GateId(0), 0, 0, 0, 1, 0);
        let elapsed = start.elapsed().as_secs_f64();
        let total_chunk_count = BigUint::from(total_preimages) * BigUint::from(chunk_count);
        SampleAuxBenchEstimate::from_chunk_big_count(
            elapsed,
            total_chunk_count,
            chunk.into_compact_bytes().len(),
        )
    }

    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        plt: &PublicLut<M::P>,
        _used_inputs: &[u64],
        lut_id: usize,
        gate_id: GateId,
        _error_sigma: f64,
    ) {
        let row_size = self.pub_matrix.row_size();
        let a_z = SH::new().sample_hash(
            params,
            self.hash_key,
            b"lwe_bench_input_pubkey",
            row_size,
            row_size * params.modulus_digits(),
            crate::sampler::DistType::FinRingDist,
        );
        let a_lt = derive_a_lt_matrix::<M, SH>(params, row_size, self.hash_key, gate_id);
        let buffer = sample_k_high_buffer::<M, SH, ST, _>(
            plt,
            params,
            self.hash_key,
            &self.trap_sampler,
            self.pub_matrix.as_ref(),
            self.trapdoor.as_ref(),
            &a_z,
            &a_lt,
            gate_id,
            lut_id,
            None,
        );
        add_lookup_buffer(buffer);
    }
}
