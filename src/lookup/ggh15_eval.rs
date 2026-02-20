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
    path::PathBuf,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

struct GateState<M>
where
    M: PolyMatrix,
{
    lut_id: usize,
    input_pubkey_bytes: Vec<u8>,
    output_pubkey_bytes: Vec<u8>,
    _m: PhantomData<M>,
}

#[derive(Default)]
struct CheckpointEntryInfo {
    indices: HashSet<usize>,
    max_part_idx: Option<usize>,
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
            let bytes = matrix.to_compact_bytes();
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

fn decode_lut_aux_pair<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    bytes: &[u8],
) -> Option<(M, M)> {
    if bytes.len() < 16 {
        return None;
    }
    let preimage_len =
        u64::from_le_bytes(bytes[0..8].try_into().expect("slice has fixed length")) as usize;
    let v_idx_len =
        u64::from_le_bytes(bytes[8..16].try_into().expect("slice has fixed length")) as usize;
    let payload_len = 16usize.saturating_add(preimage_len).saturating_add(v_idx_len);
    if payload_len != bytes.len() || preimage_len == 0 || v_idx_len == 0 {
        return None;
    }
    let preimage_start = 16;
    let preimage_end = preimage_start + preimage_len;
    let v_idx_start = preimage_end;
    let preimage = M::from_compact_bytes(params, &bytes[preimage_start..preimage_end]);
    let v_idx = M::from_compact_bytes(params, &bytes[v_idx_start..]);
    Some((preimage, v_idx))
}

fn parse_lut_aux_row_idx_from_key(id_prefix: &str, key: &str) -> Option<usize> {
    let rest = key.strip_prefix(&format!("{id_prefix}_part"))?;
    let idx_pos = rest.find("_idx")?;
    let digits = rest[(idx_pos + "_idx".len())..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() { None } else { digits.parse::<usize>().ok() }
}

fn read_lut_aux_matrix_from_checkpoint<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    lut_aux_id_prefix: &str,
    idx: usize,
    entry_key: Option<&str>,
    matrix_index: usize,
) -> Option<M> {
    if let Some(key) = entry_key {
        return read_matrix_from_multi_batch::<M>(params, dir, key, matrix_index);
    }

    let index_path = dir.join("lookup_tables.index");
    if let Ok(index_data) = read_to_string(index_path) &&
        let Ok(global_index) = serde_json::from_str::<GlobalTableIndex>(&index_data)
    {
        let mut keys = global_index
            .entries
            .keys()
            .filter(|key| {
                parse_lut_aux_row_idx_from_key(lut_aux_id_prefix, key.as_str()) == Some(idx)
            })
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            if let Some(matrix) = read_matrix_from_multi_batch::<M>(params, dir, &key, matrix_index)
            {
                return Some(matrix);
            }
        }
    }

    let bytes = read_bytes_from_multi_batch(dir, lut_aux_id_prefix, idx)?;
    let (preimage, v_idx) = decode_lut_aux_pair::<M>(params, &bytes)?;
    match matrix_index {
        0 => Some(v_idx),
        1 => Some(preimage),
        _ => None,
    }
}

fn find_lut_aux_row_key_with_full_pair(
    dir: &std::path::Path,
    lut_aux_id_prefix: &str,
    idx: usize,
) -> Option<String> {
    let index_path = dir.join("lookup_tables.index");
    let index_data = read_to_string(index_path).ok()?;
    let global_index = serde_json::from_str::<GlobalTableIndex>(&index_data).ok()?;
    let mut keys = global_index
        .entries
        .iter()
        .filter_map(|(key, entry)| {
            if parse_lut_aux_row_idx_from_key(lut_aux_id_prefix, key.as_str()) != Some(idx) {
                return None;
            }
            if !entry.indices.contains(&0) || !entry.indices.contains(&1) {
                return None;
            }
            Some(key.clone())
        })
        .collect::<Vec<_>>();
    keys.sort();
    keys.into_iter().next()
}

fn read_lut_aux_v_idx_from_checkpoint<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    lut_aux_id_prefix: &str,
    idx: usize,
    entry_key: Option<&str>,
) -> Option<M> {
    read_lut_aux_matrix_from_checkpoint::<M>(params, dir, lut_aux_id_prefix, idx, entry_key, 0)
}

fn read_lut_aux_preimage_from_checkpoint<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    dir: &std::path::Path,
    lut_aux_id_prefix: &str,
    idx: usize,
    entry_key: Option<&str>,
) -> Option<M> {
    read_lut_aux_matrix_from_checkpoint::<M>(params, dir, lut_aux_id_prefix, idx, entry_key, 1)
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
        part_idx: usize,
    ) {
        debug!(
            "Sampling LUT preimages: lut_id={}, part_idx={}, batch_size={}",
            lut_id,
            part_idx,
            batch.len()
        );
        let d = self.d;
        let m = d * params.modulus_digits();
        let trap_sampler = TS::new(params, self.trapdoor_sigma);
        let uniform_sampler = US::new();
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
        let jobs = batch
            .par_iter()
            .map(|(idx, y_poly)| {
                let v_idx = uniform_sampler.sample_uniform(
                    params,
                    m,
                    m,
                    DistType::GaussDist { sigma: self.trapdoor_sigma },
                );
                debug!(
                    "Sampled v_idx for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                // Compute w_matrix * t_idx without materializing t_idx:
                // t_idx = [I_d; G^-1(G*y); v_idx; G_small^-1(idx * I_m) * v_idx].
                let mut w_t_idx = w_block_identity.clone();
                debug!(
                    "Constructed w_block_identity for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let gy = gadget_matrix.clone() * y_poly;
                let gy_decomposed = gy.decompose();
                drop(gy);
                let w_gy = w_block_gy.clone() * gy_decomposed;
                w_t_idx.add_in_place(&w_gy);
                debug!(
                    "Constructed w_block_gy contribution for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let w_v = w_block_v.clone() * &v_idx;
                w_t_idx.add_in_place(&w_v);
                debug!(
                    "Constructed w_block_v contribution for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let idx_poly = M::P::from_usize_to_constant(params, *idx);
                let idx_small_decomposed = M::identity(params, m, Some(idx_poly)).small_decompose();
                let w_v_idx = w_block_vx.clone() * idx_small_decomposed * &v_idx;
                w_t_idx.add_in_place(&w_v_idx);
                debug!(
                    "Constructed w_block_v_idx contribution for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let mut target = M::zero(params, d + w_t_idx.row_size(), m);
                target.copy_block_from(&w_t_idx, d, 0, 0, 0, w_t_idx.row_size(), m);
                debug!(
                    "Constructed target for LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let k_l_preimage = trap_sampler.preimage(params, b1_trapdoor, b1_matrix, &target);
                debug!(
                    "Sampled LUT preimage: lut_id={}, part_idx={}, row_idx={}",
                    lut_id, part_idx, idx
                );
                let lut_aux_id = format!("{lut_aux_id_prefix}_part{part_idx}_idx{idx}");
                CompactBytesJob::new(lut_aux_id, vec![(0usize, v_idx), (1usize, k_l_preimage)])
            })
            .collect::<Vec<_>>();
        drop(gadget_matrix);
        debug!("Finished sampling LUT preimages for part_idx={}", part_idx);
        jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
        debug!("Finished storing LUT preimages for part_idx={}", part_idx);
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
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        self.derive_w_block_with_tag(params, lut_id, "block_vx", m_g * k_small)
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

    fn preimage_gate2_vx_id_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        gate_id: GateId,
    ) -> String {
        format!("{}_preimage_gate2_vx_{}", self.aux_checkpoint_prefix(params), gate_id)
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

    fn parse_checkpoint_part_idx(id_prefix: &str, key: &str) -> Option<usize> {
        if key == id_prefix {
            return Some(0);
        }
        let part_prefix = format!("{id_prefix}_part");
        let suffix = key.strip_prefix(&part_prefix)?;
        let digits = suffix.chars().take_while(|c| c.is_ascii_digit()).collect::<String>();
        if digits.is_empty() { None } else { digits.parse::<usize>().ok() }
    }

    fn collect_checkpoint_entry_info(
        checkpoint_index: Option<&GlobalTableIndex>,
        id_prefix: &str,
    ) -> CheckpointEntryInfo {
        let Some(checkpoint_index) = checkpoint_index else {
            return CheckpointEntryInfo::default();
        };

        let mut info = CheckpointEntryInfo::default();
        let mut row_to_presence: HashMap<usize, HashSet<usize>> = HashMap::new();
        if let Some(entry) = checkpoint_index.entries.get(id_prefix) {
            // Legacy: row index is encoded directly as table index.
            info.indices.extend(entry.indices.iter().copied());
            info.max_part_idx = Some(0);
        }

        let part_prefix = format!("{id_prefix}_part");
        for (key, entry) in &checkpoint_index.entries {
            if !key.starts_with(&part_prefix) {
                continue;
            }
            if let Some(part_idx) = Self::parse_checkpoint_part_idx(id_prefix, key) {
                info.max_part_idx = Some(info.max_part_idx.map_or(part_idx, |v| v.max(part_idx)));
            }
            if let Some(row_idx) = parse_lut_aux_row_idx_from_key(id_prefix, key) {
                row_to_presence.entry(row_idx).or_default().extend(entry.indices.iter().copied());
            } else {
                // Legacy: row index is encoded directly as table index.
                info.indices.extend(entry.indices.iter().copied());
            }
        }
        for (row_idx, presence) in row_to_presence {
            if presence.contains(&0) && presence.contains(&1) {
                info.indices.insert(row_idx);
            }
        }
        info
    }

    fn checkpoint_has_index(
        checkpoint_index: Option<&GlobalTableIndex>,
        id_prefix: &str,
        target_k: usize,
    ) -> bool {
        if let Some(entry_info) = checkpoint_index.and_then(|idx| idx.entries.get(id_prefix)) &&
            entry_info.indices.contains(&target_k)
        {
            return true;
        }
        let Some(checkpoint_index) = checkpoint_index else {
            return false;
        };
        let part_prefix = format!("{id_prefix}_part");
        checkpoint_index
            .entries
            .iter()
            .any(|(key, entry)| key.starts_with(&part_prefix) && entry.indices.contains(&target_k))
    }

    fn gate_checkpoint_complete(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        gate_id: GateId,
        gate_chunk_count: usize,
    ) -> bool {
        let gate1_prefix = format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id);
        let gate2_identity_prefix =
            format!("{checkpoint_prefix}_preimage_gate2_identity_{}", gate_id);
        let gate2_gy_prefix = format!("{checkpoint_prefix}_preimage_gate2_gy_{}", gate_id);
        let gate2_v_prefix = format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id);
        let gate2_vx_prefix = format!("{checkpoint_prefix}_preimage_gate2_vx_{}", gate_id);
        Self::checkpoint_has_index(checkpoint_index, &gate1_prefix, 0) &&
            Self::checkpoint_has_index(checkpoint_index, &gate2_identity_prefix, 0) &&
            Self::checkpoint_has_index(checkpoint_index, &gate2_gy_prefix, 0) &&
            (0..gate_chunk_count).all(|chunk_idx| {
                Self::checkpoint_has_index(checkpoint_index, &gate2_v_prefix, chunk_idx)
            }) &&
            (0..gate_chunk_count).all(|chunk_idx| {
                Self::checkpoint_has_index(checkpoint_index, &gate2_vx_prefix, chunk_idx)
            })
    }

    fn has_resume_candidates(
        checkpoint_index: Option<&GlobalTableIndex>,
        checkpoint_prefix: &str,
        lut_ids: &[usize],
        gate_ids: &[GateId],
        gate_chunk_count: usize,
    ) -> bool {
        let Some(checkpoint_index) = checkpoint_index else {
            return false;
        };
        for lut_id in lut_ids {
            let lut_aux_prefix = format!("{checkpoint_prefix}_lut_aux_{}", lut_id);
            let info = Self::collect_checkpoint_entry_info(Some(checkpoint_index), &lut_aux_prefix);
            if !info.indices.is_empty() {
                return true;
            }
        }
        for gate_id in gate_ids {
            if Self::gate_checkpoint_complete(
                Some(checkpoint_index),
                checkpoint_prefix,
                *gate_id,
                gate_chunk_count,
            ) {
                return true;
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
        let gate_chunk_count = params.modulus_digits();
        let has_resume_candidates = Self::has_resume_candidates(
            checkpoint_index.as_ref(),
            &checkpoint_prefix,
            &lut_ids,
            &gate_ids,
            gate_chunk_count,
        );
        let total_gate_count = gate_entries.len();
        debug!("Gate sampling start: total_gates={}, chunk_size={}", total_gate_count, chunk_size);

        let d = self.d;
        info!("Sampling auxiliary matrices with d = {}", d);
        let m_g = d * params.modulus_digits();
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
        let mut resumed_lut_rows = 0usize;
        let mut lut_plans = Vec::with_capacity(lut_entries.len());

        for (lut_id, plt) in lut_entries {
            let lut_aux_prefix = self.lut_aux_id_prefix(params, lut_id);
            let CheckpointEntryInfo { indices: aux_indices, max_part_idx: aux_max_part } =
                Self::collect_checkpoint_entry_info(checkpoint_index_for_resume, &lut_aux_prefix);

            let completed_rows: HashSet<usize> = aux_indices;
            let resumed_rows_for_lut = completed_rows.len();
            let mut part_idx = 0usize;
            resumed_lut_rows = resumed_lut_rows.saturating_add(resumed_rows_for_lut);
            processed_lut_rows = processed_lut_rows.saturating_add(resumed_rows_for_lut);
            if resumed_rows_for_lut > 0 {
                let max_part_idx = aux_max_part.unwrap_or(0);
                part_idx = max_part_idx.saturating_add(1);
                info!(
                    "LUT checkpoint resumed: lut_id={}, rows={}, aux_prefix={}, next_part_idx={}",
                    lut_id, resumed_rows_for_lut, lut_aux_prefix, part_idx
                );
            }
            lut_plans.push((lut_id, plt, completed_rows, part_idx, resumed_rows_for_lut));
        }

        let mut gates_by_lut: HashMap<usize, Vec<(GateId, GateState<M>)>> = HashMap::new();
        let mut resumed_gates = 0usize;
        let mut total_gates = 0usize;
        for (gate_id, state) in gate_entries {
            if Self::gate_checkpoint_complete(
                checkpoint_index_for_resume,
                &checkpoint_prefix,
                gate_id,
                gate_chunk_count,
            ) {
                resumed_gates = resumed_gates.saturating_add(1);
                total_gates = total_gates.saturating_add(1);
            } else {
                gates_by_lut.entry(state.lut_id).or_default().push((gate_id, state));
            }
        }
        info!(
            "Checkpoint verification completed (pending_lut_rows={}, pending_gates={})",
            total_lut_rows.saturating_sub(resumed_lut_rows),
            total_gate_count.saturating_sub(resumed_gates)
        );

        for (lut_id, plt, completed_rows, mut part_idx, resumed_rows_for_lut) in lut_plans {
            let lut_start = Instant::now();
            let lut_aux_id_prefix = self.lut_aux_id_prefix(params, lut_id);
            let w_block_identity = self.derive_w_block_identity(params, lut_id);
            let w_block_gy = self.derive_w_block_gy(params, lut_id);
            let w_block_v = self.derive_w_block_v(params, lut_id);
            let w_block_vx = self.derive_w_block_vx(params, lut_id);
            let mut batch: Vec<(usize, M::P)> = Vec::with_capacity(chunk_size);
            for (_, (idx, y_poly)) in plt.entries(params) {
                if completed_rows.contains(&idx) {
                    continue;
                }
                batch.push((idx, y_poly));
                if batch.len() >= chunk_size {
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
                        &batch,
                        part_idx,
                    );
                    debug!(
                        "Sampled LUT preimages: lut_id={}, part_idx={}, batch_size={}",
                        lut_id,
                        part_idx,
                        batch.len()
                    );
                    processed_lut_rows = processed_lut_rows.saturating_add(batch.len());
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
                    part_idx += 1;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
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
                    &batch,
                    part_idx,
                );
                processed_lut_rows = processed_lut_rows.saturating_add(batch.len());
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

        if total_gate_count == 0 {
            info!("No gate auxiliary matrices to sample");
            info!(
                "Sampled LUT and gate auxiliary matrices in {} (0 gates, resumed_lut_rows={})",
                Self::format_duration(start.elapsed()),
                resumed_lut_rows
            );
            return;
        }

        let error_sigma = self.error_sigma;
        let trapdoor_sigma = self.trapdoor_sigma;
        let chunk_size = crate::env::ggh15_gate_parallelism();
        info!(
            "GGH15 gate preimage parallelism uses Rayon global pool (GGH15_GATE_PARALLELISM={})",
            chunk_size
        );

        for (lut_id, mut gates) in gates_by_lut {
            let lut_gate_start = Instant::now();

            while !gates.is_empty() {
                let take = gates.len().min(chunk_size);
                let pending: Vec<(GateId, GateState<M>)> = gates.drain(..take).collect();

                if !pending.is_empty() {
                    total_gates = total_gates.saturating_add(pending.len());
                    let stage1 = pending
                        .into_par_iter()
                        .map(|(gate_id, state)| {
                            let uniform_sampler = US::new();
                            let trap_sampler = TS::new(params, trapdoor_sigma);
                            let s_g = if self.insert_1_to_s {
                                let s_g_bar = uniform_sampler.sample_uniform(
                                    params,
                                    d - 1,
                                    d - 1,
                                    DistType::TernaryDist,
                                );
                                s_g_bar.concat_diag(&[&M::identity(params, 1, None)])
                            } else {
                                uniform_sampler.sample_uniform(params, d, d, DistType::TernaryDist)
                            };
                            let s_g_concat = M::identity(params, d, None).concat_columns(&[&s_g]);
                            let gate_target1 = {
                                let error = uniform_sampler.sample_uniform(
                                    params,
                                    d,
                                    b1_matrix.col_size(),
                                    DistType::GaussDist { sigma: error_sigma },
                                );
                                s_g_concat.clone() * &b1_matrix + error
                            };
                            let preimage_gate1 = trap_sampler.preimage(
                                params,
                                &b0_trapdoor,
                                &b0_matrix,
                                &gate_target1,
                            );
                            drop(gate_target1);
                            drop(s_g_concat);
                            drop(s_g);
                            debug!(
                                "Sampled gate preimage 1: gate_id={}, lut_id={}",
                                gate_id, lut_id
                            );
                            let job = CompactBytesJob::new(
                                self.preimage_gate1_id_prefix(params, gate_id),
                                vec![(0, preimage_gate1)],
                            );
                            (job, (gate_id, state))
                        })
                        .collect::<Vec<_>>();
                    let (stage1_jobs, stage2_inputs): (Vec<_>, Vec<_>) = stage1.into_iter().unzip();
                    stage1_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

                    let w_block_identity = self.derive_w_block_identity(params, lut_id);
                    let stage2 = stage2_inputs
                        .into_par_iter()
                        .map(|(gate_id, state)| {
                            let trap_sampler = TS::new(params, trapdoor_sigma);
                            let out_matrix =
                                M::from_compact_bytes(params, &state.output_pubkey_bytes);
                            let target_gate2_identity =
                                out_matrix.concat_rows(&[&w_block_identity]);
                            let preimage_gate2_identity = trap_sampler.preimage(
                                params,
                                &b1_trapdoor,
                                &b1_matrix,
                                &target_gate2_identity,
                            );
                            drop(target_gate2_identity);
                            debug!(
                                "Sampled gate preimage 2 (identity part): gate_id={}, lut_id={}",
                                gate_id, lut_id
                            );
                            let job = CompactBytesJob::new(
                                self.preimage_gate2_identity_id_prefix(params, gate_id),
                                vec![(0, preimage_gate2_identity)],
                            );
                            (job, (gate_id, state))
                        })
                        .collect::<Vec<_>>();
                    drop(w_block_identity);
                    let (stage2_jobs, stage3_inputs): (Vec<_>, Vec<_>) = stage2.into_iter().unzip();
                    stage2_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

                    let w_block_gy = self.derive_w_block_gy(params, lut_id);
                    let stage3 = stage3_inputs
                        .into_par_iter()
                        .map(|(gate_id, state)| {
                            let trap_sampler = TS::new(params, trapdoor_sigma);
                            let target_high_gy = -M::gadget_matrix(params, d);
                            let target_gate2_gy = target_high_gy.concat_rows(&[&w_block_gy]);
                            drop(target_high_gy);
                            let preimage_gate2_gy = trap_sampler.preimage(
                                params,
                                &b1_trapdoor,
                                &b1_matrix,
                                &target_gate2_gy,
                            );
                            drop(target_gate2_gy);
                            debug!(
                                "Sampled gate preimage 2 (gy part): gate_id={}, lut_id={}",
                                gate_id, lut_id
                            );
                            let job = CompactBytesJob::new(
                                self.preimage_gate2_gy_id_prefix(params, gate_id),
                                vec![(0, preimage_gate2_gy)],
                            );
                            (job, (gate_id, state))
                        })
                        .collect::<Vec<_>>();
                    drop(w_block_gy);
                    let (stage3_jobs, stage4_inputs): (Vec<_>, Vec<_>) = stage3.into_iter().unzip();
                    stage3_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);

                    let w_block_v = self.derive_w_block_v(params, lut_id);
                    let stage4 = stage4_inputs
                        .into_par_iter()
                        .map(|(gate_id, state)| {
                            let trap_sampler = TS::new(params, trapdoor_sigma);
                            let hash_sampler = HS::new();
                            let input_matrix =
                                M::from_compact_bytes(params, &state.input_pubkey_bytes);
                            let u_g_matrix = hash_sampler.sample_hash(
                                params,
                                self.hash_key,
                                format!("ggh15_lut_u_g_matrix_{}", gate_id),
                                d,
                                m_g,
                                DistType::FinRingDist,
                            );
                            debug!(
                                "Derived u_g_matrix for gate: gate_id={}, lut_id={}, rows={}, cols={}",
                                gate_id,
                                lut_id,
                                u_g_matrix.row_size(),
                                u_g_matrix.col_size()
                            );
                            let taget_high_v = - input_matrix * u_g_matrix.decompose();
                            let target_gate2_v = taget_high_v.concat_rows(&[&w_block_v]);
                            drop(taget_high_v);
                            let preimage_gate2_v = trap_sampler.preimage(
                                params,
                                &b1_trapdoor,
                                &b1_matrix,
                                &target_gate2_v,
                            );
                            drop(target_gate2_v);
                            debug!(
                                "Sampled gate preimage 2 (v part): gate_id={}, lut_id={}",
                                gate_id, lut_id
                            );
                            let job = CompactBytesJob::new(
                                self.preimage_gate2_v_id_prefix(params, gate_id),
                                vec![(0, preimage_gate2_v)],
                            );
                            (job, (gate_id, u_g_matrix))
                        })
                        .collect::<Vec<_>>();
                    let (stage4_jobs, stage5_inputs): (Vec<_>, Vec<_>) = stage4.into_iter().unzip();
                    stage4_jobs.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                    drop(w_block_v);

                    let w_block_vx = self.derive_w_block_vx(params, lut_id);
                    let small_gadget_matrix = M::small_gadget_matrix(params, m_g);
                    let stage5 = stage5_inputs
                        .into_par_iter()
                        .map(|(gate_id, u_g_matrix)| {
                            let trap_sampler = TS::new(params, trapdoor_sigma);
                            let target_high_vx = u_g_matrix * &small_gadget_matrix;
                            let target_gate2_vx = target_high_vx.concat_rows(&[&w_block_vx]);
                            drop(target_high_vx);
                            let preimage_gate2_vx = trap_sampler.preimage(
                                params,
                                &b1_trapdoor,
                                &b1_matrix,
                                &target_gate2_vx,
                            );
                            drop(target_gate2_vx);
                            debug!(
                                "Sampled gate preimage 2 (vx part): gate_id={}, lut_id={}",
                                gate_id, lut_id
                            );
                            let job = CompactBytesJob::new(
                                self.preimage_gate2_vx_id_prefix(params, gate_id),
                                vec![(0, preimage_gate2_vx)],
                            );
                            job
                        })
                        .collect::<Vec<_>>();
                    stage5.into_par_iter().for_each(CompactBytesJob::wait_then_store);
                    drop(w_block_vx);
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
                output_pubkey_bytes: output_pubkey.matrix.to_compact_bytes(),
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
    pub c_b0: M,
    _hs: PhantomData<HS>,
}

impl<M, HS> GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(hash_key: [u8; 32], dir_path: PathBuf, checkpoint_prefix: String, c_b0: M) -> Self {
        Self { hash_key, dir_path, checkpoint_prefix, c_b0, _hs: PhantomData }
    }
}

impl<M, HS> PltEvaluator<BggEncoding<M>> for GGH15BGGEncodingPltEvaluator<M, HS>
where
    M: PolyMatrix + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
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
        let lut_aux_entry_key = find_lut_aux_row_key_with_full_pair(dir, &lut_aux_prefix, k);

        let preimage_gate1 = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate1_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate1 for gate {} not found", gate_id));
        let sg_times_b1 = self.c_b0.clone() * preimage_gate1;

        let hash_sampler = HS::new();
        let d = input.pubkey.matrix.row_size();
        let m_g = d * params.modulus_digits();
        let (_, _, crt_depth) = params.to_crt();
        let k_small = params.modulus_digits() / crt_depth;
        // let mut c_const: Option<M> = None;

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
        let mut c_const = sg_times_b1.clone() * preimage_gate2_identity;

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
        c_const = c_const + sg_times_b1.clone() * preimage_gate2_gy * gy_decomposed;

        let v_idx = read_lut_aux_v_idx_from_checkpoint::<M>(
            params,
            dir,
            &lut_aux_prefix,
            k,
            lut_aux_entry_key.as_deref(),
        )
        .unwrap_or_else(|| panic!("v_idx (index {}) for lut {} not found", k, lut_id));
        let preimage_gate2_v = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_v_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_v for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_v.col_size(),
            m_g,
            "preimage_gate2_v must have m_g columns"
        );
        c_const = c_const + sg_times_b1.clone() * preimage_gate2_v * &v_idx;

        let preimage_gate2_vx = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{checkpoint_prefix}_preimage_gate2_vx_{}", gate_id),
            0,
        )
        .unwrap_or_else(|| panic!("preimage_gate2_vx for gate {} not found", gate_id));
        debug_assert_eq!(
            preimage_gate2_vx.col_size(),
            m_g * k_small,
            "preimage_gate2_vx must have m_g * k_small columns"
        );
        let x_small_decomposed = M::identity(params, m_g, Some(x.clone())).small_decompose();
        c_const = c_const + sg_times_b1.clone() * preimage_gate2_vx * x_small_decomposed * &v_idx;

        let preimage_lut = read_lut_aux_preimage_from_checkpoint::<M>(
            params,
            dir,
            &lut_aux_prefix,
            k,
            lut_aux_entry_key.as_deref(),
        )
        .unwrap_or_else(|| panic!("preimage_lut (index {}) for lut {} not found", k, lut_id));
        c_const = c_const - sg_times_b1.clone() * preimage_lut;
        drop(sg_times_b1);

        let u_g = hash_sampler.sample_hash(
            params,
            self.hash_key,
            format!("ggh15_lut_u_g_matrix_{}", gate_id),
            d,
            m_g,
            DistType::FinRingDist,
        );
        debug!(
            "Derived u_g_matrix for gate encoding: gate_id={}, lut_id={}, rows={}, cols={}",
            gate_id,
            lut_id,
            u_g.row_size(),
            u_g.col_size()
        );

        let c_x_randomized = input.vector.clone() * u_g.decompose() * v_idx;
        debug!("Computed c_x_randomized for gate encoding: gate_id={}, lut_id={}", gate_id, lut_id);
        drop(u_g);
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
        BggEncoding::new(c_out, out_pubkey, Some(y))
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
    use std::{fs, path::Path, sync::Arc};

    fn setup_lsb_constant_binary_plt(t_n: usize, params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::<DCRTPoly>::new_from_usize_range(
            params,
            t_n,
            |params, k| (k, DCRTPoly::from_usize_to_lsb(params, k)),
            None,
        )
    }

    const SIGMA: f64 = 4.578;

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

        let one_pubkey = Arc::new(enc_one.pubkey.clone());
        let input_pubkeys = vec![Arc::new(enc1.pubkey.clone())];
        let result_pubkey =
            circuit.eval(&params, &one_pubkey, &input_pubkeys, Some(&plt_pubkey_evaluator));
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
        >::new(key, dir_path.into(), checkpoint_prefix, c_b0);

        let one_encoding = Arc::new(enc_one.clone());
        let input_encodings = vec![Arc::new(enc1.clone())];
        let result_encoding =
            circuit.eval(&params, &one_encoding, &input_encodings, Some(&plt_encoding_evaluator));
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

        let one_pubkey = Arc::new(enc_one.pubkey.clone());
        let input_pubkeys_arc = input_pubkeys.iter().cloned().map(Arc::new).collect::<Vec<_>>();
        let result_pubkey =
            circuit.eval(&params, &one_pubkey, &input_pubkeys_arc, Some(&plt_pubkey_evaluator));
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
        >::new(key, dir_path.into(), checkpoint_prefix, c_b0);

        let one_encoding = Arc::new(enc_one.clone());
        let input_encodings_arc = input_encodings.iter().cloned().map(Arc::new).collect::<Vec<_>>();
        let result_encoding = circuit.eval(
            &params,
            &one_encoding,
            &input_encodings_arc,
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
