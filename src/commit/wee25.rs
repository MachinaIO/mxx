use std::{
    collections::HashMap,
    marker::PhantomData,
    ops::Range,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    env,
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::{read_bytes_from_multi_batch, read_matrix_from_multi_batch},
        write::{add_lookup_buffer, get_lookup_buffer, get_lookup_buffer_bytes},
    },
};
use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub const WEE25_PUBLIC_PARAMS_PREFIX: &str = "wee25_public_params";
pub const WEE25_COMMIT_CACHE_PREFIX: &str = "wee25_commit_cache";

#[derive(Debug, Clone)]
pub struct Wee25Commit<M: PolyMatrix, HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync> {
    pub secret_size: usize,
    pub tree_base: usize,
    pub m_b: usize,
    pub m_g: usize,
    pub trapdoor_sigma: f64,
    _marker: PhantomData<(M, HS)>,
}

#[derive(Debug, Clone)]
pub struct Wee25PublicParams<M: PolyMatrix> {
    pub b: M,
    pub top_j: Vec<Vec<u8>>,
    pub top_j_last: Vec<Vec<u8>>,
    pub t_bottom_j_2m: Vec<u8>,
    pub t_bottom_j_2m_last: Vec<u8>,
    pub hash_key: [u8; 32],
}

#[derive(Clone)]
pub struct MsgMatrixStream<'a, M: PolyMatrix> {
    reader: Arc<dyn Fn(Range<usize>) -> Vec<M> + Send + Sync + 'a>,
    offset: usize,
    len: usize,
}

impl<'a, M: PolyMatrix + 'a> MsgMatrixStream<'a, M> {
    pub fn new<F>(len: usize, reader: F) -> Self
    where
        F: Fn(Range<usize>) -> Vec<M> + Send + Sync + 'a,
    {
        Self { reader: Arc::new(reader), offset: 0, len }
    }

    pub fn from_blocks(blocks: Vec<M>) -> Self {
        let len = blocks.len();
        Self::new(len, move |range: Range<usize>| {
            blocks[range.start..range.end].iter().cloned().collect()
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice(&self, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end, "stream slice must be ordered");
        debug_assert!(
            range.end <= self.len,
            "stream slice end {} exceeds length {}",
            range.end,
            self.len
        );
        Self {
            reader: Arc::clone(&self.reader),
            offset: self.offset + range.start,
            len: range.end - range.start,
        }
    }

    pub fn read(&self, range: Range<usize>) -> Vec<M> {
        debug_assert!(range.start <= range.end, "read range must be ordered");
        debug_assert!(
            range.end <= self.len,
            "read range end {} exceeds length {}",
            range.end,
            self.len
        );
        let absolute = (self.offset + range.start)..(self.offset + range.end);
        let out = (self.reader)(absolute);
        debug_assert_eq!(
            out.len(),
            range.end - range.start,
            "reader returned {} blocks, expected {}",
            out.len(),
            range.end - range.start
        );
        out
    }
}

#[derive(Debug, Clone)]
pub struct CommitCache<M: PolyMatrix> {
    pub id_prefix: String,
    pub cols: usize,
    tree_base: usize,
    nodes: Arc<DashMap<(usize, usize), M>>,
    persist_batch_size: usize,
    pending_persist: Arc<Mutex<Vec<(usize, M)>>>,
    next_chunk_seq: Arc<AtomicUsize>,
}

impl<M: PolyMatrix> CommitCache<M> {
    pub fn new(id_prefix: String, cols: usize, tree_base: usize) -> Self {
        Self {
            id_prefix,
            cols,
            tree_base,
            nodes: Arc::new(DashMap::new()),
            persist_batch_size: env::wee25_commit_cache_persist_batch(),
            pending_persist: Arc::new(Mutex::new(Vec::new())),
            next_chunk_seq: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn chunk_key_prefix(&self) -> String {
        format!("{}_chunk_", self.id_prefix)
    }

    fn node_index(&self, offset: usize, len: usize) -> usize {
        debug_assert!(
            len >= self.tree_base,
            "node len {} must be at least tree_base {}",
            len,
            self.tree_base
        );
        debug_assert_eq!(offset % len, 0, "offset {} must align with len {}", offset, len);
        let mut level_len = self.cols;
        let mut level_nodes = 1usize;
        let mut nodes_before_level = 0usize;
        while level_len > len {
            debug_assert_eq!(
                level_len % self.tree_base,
                0,
                "level_len {} must be divisible by tree_base {}",
                level_len,
                self.tree_base
            );
            nodes_before_level += level_nodes;
            level_nodes = level_nodes
                .checked_mul(self.tree_base)
                .expect("node count overflow while indexing commit cache");
            level_len /= self.tree_base;
        }
        debug_assert_eq!(
            level_len, len,
            "cannot map offset/len ({},{}) into tree rooted at {}",
            offset, len, self.cols
        );
        let pos_in_level = offset / len;
        debug_assert!(
            pos_in_level < level_nodes,
            "position {} exceeds nodes at level {}",
            pos_in_level,
            level_nodes
        );
        nodes_before_level + pos_in_level
    }

    pub fn get(&self, offset: usize, len: usize) -> Option<M> {
        self.nodes.get(&(offset, len)).map(|entry| entry.clone())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn load<HS>(
        params: &<M::P as Poly>::Params,
        dir: &Path,
        wee25_commit: &Wee25Commit<M, HS>,
        hash_key: [u8; 32],
        cols: usize,
    ) -> Option<Self>
    where
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    {
        wee25_commit.assert_stream_len(cols);
        let id_prefix = wee25_commit.commit_cache_prefix(params, hash_key, cols);
        let cache = Self::new(id_prefix, cols, wee25_commit.tree_base);
        let index_path = dir.join("lookup_tables.index");
        let index_json = std::fs::read_to_string(&index_path).ok()?;
        let global_index =
            serde_json::from_str::<crate::storage::write::GlobalTableIndex>(&index_json).ok()?;
        let mut node_index_to_chunk = HashMap::new();
        let chunk_key_prefix = cache.chunk_key_prefix();
        for (key, entry) in global_index.entries.iter() {
            if key.starts_with(&chunk_key_prefix) {
                for node_index in entry.indices.iter() {
                    node_index_to_chunk.insert(*node_index, key.clone());
                }
            }
        }
        if node_index_to_chunk.is_empty() {
            return None;
        }
        cache.load_subtree(params, dir, wee25_commit.tree_base, 0, cols, &node_index_to_chunk)?;
        Some(cache)
    }

    fn load_subtree(
        &self,
        params: &<M::P as Poly>::Params,
        dir: &Path,
        tree_base: usize,
        offset: usize,
        len: usize,
        node_index_to_chunk: &HashMap<usize, String>,
    ) -> Option<()> {
        let node_index = self.node_index(offset, len);
        let chunk_prefix = node_index_to_chunk.get(&node_index)?;
        let commit = read_matrix_from_multi_batch::<M>(params, dir, chunk_prefix, node_index)?;
        self.nodes.insert((offset, len), commit);
        if len == tree_base {
            return Some(());
        }
        let child_len = len / tree_base;
        for idx in 0..tree_base {
            let child_offset = offset + idx * child_len;
            self.load_subtree(
                params,
                dir,
                tree_base,
                child_offset,
                child_len,
                node_index_to_chunk,
            )?;
        }
        Some(())
    }

    fn insert_and_persist(&self, offset: usize, len: usize, commit: M) {
        if self.persist_batch_size <= 1 {
            self.persist_batch(vec![(self.node_index(offset, len), commit)]);
            return;
        }
        let mut batch: Option<Vec<(usize, M)>> = None;
        {
            let mut pending = self.pending_persist.lock().expect("pending_persist lock poisoned");
            pending.push((self.node_index(offset, len), commit));
            if pending.len() >= self.persist_batch_size {
                batch = Some(std::mem::take(&mut *pending));
            }
        }
        if let Some(batch) = batch {
            self.persist_batch(batch);
        }
    }

    fn persist_batch(&self, batch: Vec<(usize, M)>) {
        let chunk_seq = self.next_chunk_seq.fetch_add(1, Ordering::Relaxed);
        let chunk_id = format!("{}_chunk_{}", self.id_prefix, chunk_seq);
        let ok = add_lookup_buffer(get_lookup_buffer(batch, &chunk_id));
        debug_assert!(ok, "failed to enqueue commit cache chunk {}", chunk_id);
    }

    fn flush_pending_persist(&self) {
        let batch = {
            let mut pending = self.pending_persist.lock().expect("pending_persist lock poisoned");
            if pending.is_empty() {
                return;
            }
            std::mem::take(&mut *pending)
        };
        self.persist_batch(batch);
    }
}

impl<M: PolyMatrix> Wee25PublicParams<M> {
    pub fn new(
        b: M,
        top_j: Vec<Vec<u8>>,
        top_j_last: Vec<Vec<u8>>,
        t_bottom_j_2m: Vec<u8>,
        t_bottom_j_2m_last: Vec<u8>,
        hash_key: [u8; 32],
    ) -> Self {
        Self { b, top_j, top_j_last, t_bottom_j_2m, t_bottom_j_2m_last, hash_key }
    }

    pub fn read_from_storage<HS>(
        params: &<M::P as Poly>::Params,
        dir: &std::path::Path,
        wee25_commit: &Wee25Commit<M, HS>,
        hash_key: [u8; 32],
    ) -> Option<Self>
    where
        HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    {
        let id_prefix = wee25_commit.checkpoint_prefix(params, hash_key);
        let b = read_matrix_from_multi_batch::<M>(params, dir, &format!("{id_prefix}_b"), 0)?;
        let pp_size = wee25_commit.tree_base * wee25_commit.m_b * wee25_commit.m_g;
        let (_t_block_batch, top_j_batch, _t_bottom_batch) = env::wee25_parallel_batches();
        let mut top_j_parts = Vec::with_capacity(pp_size);
        let mut top_j_last_parts = Vec::with_capacity(pp_size);
        for chunk_start in (0..pp_size).step_by(top_j_batch) {
            let chunk_end = (chunk_start + top_j_batch).min(pp_size);
            let chunk_prefix = format!("{id_prefix}_top_j_part_{chunk_start}");
            let first = read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, chunk_start);
            let chunk_len = chunk_end - chunk_start;
            let first_last = read_matrix_from_multi_batch::<M>(
                params,
                dir,
                &chunk_prefix,
                chunk_start + chunk_len,
            );
            if first.is_none() || first_last.is_none() {
                break;
            }
            for offset in 0..chunk_len {
                let top_idx = chunk_start + offset;
                let last_idx = chunk_start + chunk_len + offset;
                let part = read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, top_idx)?;
                let part_last =
                    read_matrix_from_multi_batch::<M>(params, dir, &chunk_prefix, last_idx)?;
                top_j_parts.push(part.to_compact_bytes());
                top_j_last_parts.push(part_last.to_compact_bytes());
            }
        }
        let t_bottom_prefix = format!("{id_prefix}_t_bottom_j_2m");
        let t_bottom_j_2m = read_matrix_from_multi_batch::<M>(params, dir, &t_bottom_prefix, 0)?;
        let t_bottom_j_2m_last =
            read_matrix_from_multi_batch::<M>(params, dir, &t_bottom_prefix, 1)?;
        if top_j_parts.len() != pp_size || top_j_last_parts.len() != pp_size {
            return None;
        }
        Some(Self {
            b,
            top_j: top_j_parts,
            top_j_last: top_j_last_parts,
            t_bottom_j_2m: t_bottom_j_2m.to_compact_bytes(),
            t_bottom_j_2m_last: t_bottom_j_2m_last.to_compact_bytes(),
            hash_key,
        })
    }

    pub fn default_storage_prefix() -> &'static str {
        WEE25_PUBLIC_PARAMS_PREFIX
    }
}

impl<M, HS> Wee25Commit<M, HS>
where
    M: PolyMatrix,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
{
    fn read_public_params_checkpoint<TS: PolyTrapdoorSampler<M = M>>(
        &self,
        params: &<M::P as Poly>::Params,
        checkpoint_dir: &std::path::Path,
        checkpoint_prefix: &str,
        t_block_checkpoint_prefix: &str,
    ) -> (Option<M>, Option<TS::Trapdoor>, usize, Vec<Vec<u8>>, bool, usize) {
        let tree_base = self.tree_base;
        let m_b = self.m_b;
        let m_g = self.m_g;
        let pp_size = tree_base * m_b * m_g;
        let mut t_bottom_ready = false;
        let trapdoor_prefix = format!("{checkpoint_prefix}_trapdoor");

        let mut can_resume = false;
        let mut b = None;
        let mut trapdoor = None;
        let b_loaded =
            read_bytes_from_multi_batch(checkpoint_dir, &format!("{checkpoint_prefix}_b"), 0);
        let trapdoor_bytes = read_bytes_from_multi_batch(checkpoint_dir, &trapdoor_prefix, 0);
        if let (Some(b_bytes), Some(bytes)) = (b_loaded, trapdoor_bytes) {
            if let Some(td) = TS::trapdoor_from_bytes(params, &bytes) {
                b = Some(M::from_compact_bytes(params, &b_bytes));
                trapdoor = Some(td);
                can_resume = true;
                tracing::debug!("Wee25Commit::sample_public_params loaded b/trapdoor checkpoint");
            }
        }

        let (t_block_batch, top_j_batch, _) = env::wee25_parallel_batches();
        let mut t_block_bytes = vec![Vec::new(); pp_size];
        let mut t_block_resume_start = 0usize;
        if can_resume {
            loop {
                if t_block_resume_start >= pp_size {
                    break;
                }
                let chunk_start = t_block_resume_start;
                let chunk_end = (chunk_start + t_block_batch).min(pp_size);
                let chunk_prefix = format!("{t_block_checkpoint_prefix}_chunk_{chunk_start}");
                let read_chunks = parallel_iter!(chunk_start..chunk_end)
                    .map(|block_idx| {
                        (
                            block_idx,
                            read_bytes_from_multi_batch(checkpoint_dir, &chunk_prefix, block_idx),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut chunk_ok = true;
                for (block_idx, chunk) in read_chunks {
                    match chunk {
                        Some(bytes) => t_block_bytes[block_idx] = bytes,
                        None => {
                            chunk_ok = false;
                            break;
                        }
                    }
                }
                if !chunk_ok {
                    break;
                }
                t_block_resume_start = chunk_end;
            }
            if t_block_resume_start < pp_size {
                can_resume = false;
                t_bottom_ready = false;
            } else {
                let t_bottom_prefix = format!("{checkpoint_prefix}_t_bottom_j_2m");
                let t_bottom_loaded =
                    read_bytes_from_multi_batch(checkpoint_dir, &t_bottom_prefix, 0);
                let t_bottom_last_loaded =
                    read_bytes_from_multi_batch(checkpoint_dir, &t_bottom_prefix, 1);
                if let (Some(_t_bottom), Some(_t_bottom_last)) =
                    (t_bottom_loaded, t_bottom_last_loaded)
                {
                    t_bottom_ready = true;
                    tracing::debug!(
                        "Wee25Commit::sample_public_params resumed t_block at {} of {}",
                        t_block_resume_start,
                        pp_size
                    );
                } else {
                    t_bottom_ready = false;
                    can_resume = false;
                }
            }
        }

        let mut top_j_resume_start = 0usize;
        if can_resume {
            loop {
                if top_j_resume_start >= pp_size {
                    break;
                }
                let chunk_start = top_j_resume_start;
                let chunk_end = (chunk_start + top_j_batch).min(pp_size);
                let chunk_prefix = format!("{checkpoint_prefix}_top_j_part_{chunk_start}");
                let chunk_len = chunk_end - chunk_start;
                let chunk_ok = parallel_iter!(0..chunk_len).all(|offset| {
                    let top_idx = chunk_start + offset;
                    let last_idx = chunk_start + chunk_len + offset;
                    let part = read_bytes_from_multi_batch(checkpoint_dir, &chunk_prefix, top_idx);
                    let part_last =
                        read_bytes_from_multi_batch(checkpoint_dir, &chunk_prefix, last_idx);
                    match (part, part_last) {
                        (Some(_), Some(_)) => true,
                        _ => false,
                    }
                });
                if !chunk_ok {
                    break;
                }
                top_j_resume_start = chunk_end;
            }
            if top_j_resume_start > 0 {
                tracing::debug!(
                    "Wee25Commit::sample_public_params resumed top_j_parts at {} of {}",
                    top_j_resume_start,
                    pp_size
                );
            }
        }

        (b, trapdoor, t_block_resume_start, t_block_bytes, t_bottom_ready, top_j_resume_start)
    }

    pub fn new(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        tree_base: usize,
        trapdoor_sigma: f64,
    ) -> Self {
        debug_assert!(tree_base >= 2, "tree_base must be at least 2");
        let log_base_q = params.modulus_digits();
        let m_g = secret_size * log_base_q;
        // For the current trapdoor sampler, b has size d x (2d + d*log_base_q).
        let m_b = secret_size * (2 + log_base_q);
        Self { secret_size, tree_base, m_b, m_g, trapdoor_sigma, _marker: PhantomData }
    }

    pub fn checkpoint_prefix(&self, params: &<M::P as Poly>::Params, hash_key: [u8; 32]) -> String {
        let (_crt_moduli, crt_bits, crt_depth) = params.to_crt();
        let (t_block_batch, top_j_batch, t_bottom_batch) = env::wee25_parallel_batches();
        format!(
            "{}_s{}_tb{}_mb{}_mg{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_tblock{}_topj{}_tbottom{}_key{}",
            Wee25PublicParams::<M>::default_storage_prefix(),
            self.secret_size,
            self.tree_base,
            self.m_b,
            self.m_g,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            t_block_batch,
            top_j_batch,
            t_bottom_batch,
            hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    pub fn commit_cache_prefix(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        cols: usize,
    ) -> String {
        let (_crt_moduli, crt_bits, crt_depth) = params.to_crt();
        format!(
            "{}_s{}_tb{}_mb{}_mg{}_crtbits{}_crtdepth{}_ring{}_base{}_sigma{:.6}_cols{}_key{}",
            WEE25_COMMIT_CACHE_PREFIX,
            self.secret_size,
            self.tree_base,
            self.m_b,
            self.m_g,
            crt_bits,
            crt_depth,
            params.ring_dimension(),
            params.base_bits(),
            self.trapdoor_sigma,
            cols,
            hash_key.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        )
    }

    pub fn sample_public_params<
        US: PolyUniformSampler<M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M>,
    >(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        checkpoint_dir: &std::path::Path,
    ) {
        let secret_size = self.secret_size;
        let tree_base = self.tree_base;
        let m_b = self.m_b;
        let m_g = self.m_g;
        let trapdoor_sigma = self.trapdoor_sigma;
        let checkpoint_prefix = self.checkpoint_prefix(params, hash_key);
        let t_block_checkpoint_prefix = format!("{checkpoint_prefix}_t_block_ckpt");
        let pp_size = tree_base * m_b * m_g;
        tracing::debug!(
            "Wee25Commit::sample_public_params secret_size={} m_b={} m_g={} pp_size={}",
            secret_size,
            m_b,
            m_g,
            pp_size
        );
        let l = tree_base * m_b;
        let log_base_q = params.modulus_digits();
        let j_2m_cols = l * log_base_q;
        let (
            mut b,
            mut trapdoor,
            t_block_resume_start,
            mut t_block_bytes,
            t_bottom_ready,
            top_j_resume_start,
        ) = self.read_public_params_checkpoint::<TS>(
            params,
            checkpoint_dir,
            &checkpoint_prefix,
            &t_block_checkpoint_prefix,
        );
        let gadget_vec = M::gadget_matrix(params, 1);
        let gadget_row = gadget_vec.get_row(0);
        drop(gadget_vec);
        debug_assert_eq!(
            gadget_row.len(),
            log_base_q,
            "gadget row length mismatch: expected {}, got {}",
            log_base_q,
            gadget_row.len()
        );
        let gadget = M::gadget_matrix(params, secret_size);
        let t_top_parts_start = std::time::Instant::now();

        let build_j_2m_block = |block_idx: usize, block_group: usize| -> M {
            debug_assert_eq!(
                m_g,
                secret_size * log_base_q,
                "m_g must equal secret_size * log_base_q"
            );
            debug_assert!(block_group < l, "block_group {} exceeds l {}", block_group, l);
            let mut row_blocks = (0..secret_size)
                .into_par_iter()
                .map(|i| {
                    let r = block_idx * secret_size + i;
                    let r_g_start = r * log_base_q;
                    let slice_start = block_group * m_g * m_g;
                    let offset = r_g_start - slice_start;
                    let step = m_g + 1;
                    let mut row_mat = M::zero(params, 1, j_2m_cols);
                    let c = (offset + step - 1) / step;
                    if c < m_g {
                        let pos = slice_start + c * step;
                        if pos <= r_g_start + log_base_q - 1 {
                            let k = pos - r_g_start;
                            let coeff = gadget_row[k].clone();
                            for s in 0..log_base_q {
                                let entry = coeff.clone() * &gadget_row[s];
                                row_mat.set_entry(0, block_group * log_base_q + s, entry);
                            }
                        }
                    }
                    let row_decomp = row_mat.decompose();
                    (i, row_decomp)
                })
                .collect::<Vec<_>>();
            row_blocks.sort_by_key(|(i, _)| *i);
            let row_refs = row_blocks.iter().map(|(_, row)| row).collect::<Vec<_>>();
            row_refs[0].concat_rows(&row_refs[1..])
        };

        // Sample t_block once, store compact bytes, and update t_bottom aggregates.
        let (t_block_batch, top_j_batch, t_bottom_batch) = env::wee25_parallel_batches();
        if b.is_none() || trapdoor.is_none() {
            let trapdoor_sampler = TS::new(params, trapdoor_sigma);
            let (td, b_mat) = trapdoor_sampler.trapdoor(params, secret_size);
            let td_bytes = TS::trapdoor_to_bytes(&td);
            add_lookup_buffer(get_lookup_buffer(
                vec![(0, b_mat.clone())],
                &format!("{checkpoint_prefix}_b"),
            ));
            add_lookup_buffer(get_lookup_buffer_bytes(
                vec![(0, td_bytes)],
                &format!("{checkpoint_prefix}_trapdoor"),
            ));
            b = Some(b_mat);
            trapdoor = Some(td);
        }
        let b = b.expect("trapdoor/b must be available");
        let trapdoor = trapdoor.expect("trapdoor/b must be available");
        debug_assert_eq!(
            b.col_size(),
            m_b,
            "Wee25PublicParams m_b mismatch: expected {}, got {}",
            m_b,
            b.col_size()
        );
        let uniform_sampler = US::new();
        let hash_sampler = HS::new();
        tracing::info!(
            "Wee25Commit::sample_public_params sampling t_block from {}",
            t_block_resume_start
        );
        for chunk_start in (t_block_resume_start..pp_size).step_by(t_block_batch) {
            let chunk_end = (chunk_start + t_block_batch).min(pp_size);
            tracing::debug!(
                "Wee25Commit::sample_public_params processing blocks {}..{} of {}",
                chunk_start,
                chunk_end,
                pp_size
            );
            let chunk_results = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|block_idx| {
                    let t_block = uniform_sampler.sample_uniform(
                        params,
                        m_b,
                        m_g,
                        DistType::GaussDist { sigma: trapdoor_sigma },
                    );
                    let t_block_bytes = t_block.to_compact_bytes();
                    (block_idx, t_block_bytes, t_block)
                })
                .collect::<Vec<_>>();
            let mut ckpt_blocks = Vec::with_capacity(chunk_results.len());
            for (block_idx, _bytes, t_block_ckpt) in &chunk_results {
                ckpt_blocks.push((*block_idx, t_block_ckpt.clone()));
            }
            if !ckpt_blocks.is_empty() {
                let chunk_prefix = format!("{t_block_checkpoint_prefix}_chunk_{chunk_start}");
                add_lookup_buffer(get_lookup_buffer(ckpt_blocks, &chunk_prefix));
            }
            for (block_idx, bytes, _t_block_ckpt) in chunk_results {
                t_block_bytes[block_idx] = bytes;
            }
        }

        if !t_bottom_ready {
            tracing::info!("Wee25Commit::sample_public_params computing t_bottom_j_2m aggregates");
            let mut t_bottom_j_2m = M::zero(params, m_b, j_2m_cols);
            let mut t_bottom_j_2m_last = M::zero(params, m_b, l);
            for chunk_start in (0..pp_size).step_by(t_bottom_batch) {
                let chunk_end = (chunk_start + t_bottom_batch).min(pp_size);
                tracing::debug!(
                    "Wee25Commit::sample_public_params t_bottom_j_2m chunk {}..{} of {}",
                    chunk_start,
                    chunk_end,
                    pp_size
                );
                let (chunk_j_2m, chunk_j_2m_last) = (chunk_start..chunk_end)
                    .into_par_iter()
                    .map(|block_idx| {
                        let t_block = M::from_compact_bytes(params, &t_block_bytes[block_idx]);
                        let block_group = block_idx / m_g;
                        let block_in_group = block_idx % m_g;
                        let unit_row = M::unit_row_vector(params, l, block_group);
                        let j_2m_block = build_j_2m_block(block_idx, block_group);
                        let contrib_j_2m = t_block.clone() * &j_2m_block;
                        let t_block_col = t_block.slice_columns(block_in_group, block_in_group + 1);
                        let contrib_j_2m_last = t_block_col * &unit_row;
                        (contrib_j_2m, contrib_j_2m_last)
                    })
                    .reduce(
                        || (M::zero(params, m_b, j_2m_cols), M::zero(params, m_b, l)),
                        |(acc_j, acc_last), (c_j, c_last)| (acc_j + c_j, acc_last + c_last),
                    );
                t_bottom_j_2m = t_bottom_j_2m + chunk_j_2m;
                t_bottom_j_2m_last = t_bottom_j_2m_last + chunk_j_2m_last;
            }
            add_lookup_buffer(get_lookup_buffer(
                vec![(0, t_bottom_j_2m.clone()), (1, t_bottom_j_2m_last.clone())],
                &format!("{checkpoint_prefix}_t_bottom_j_2m"),
            ));
        }
        tracing::debug!("Wee25Commit::sample_public_params completed t_block sampling");

        // Reverse the loop order: outer over j_2m columns (idx), inner over t_block/j_2m rows.
        tracing::info!(
            "Wee25Commit::sample_public_params computing top_j_parts from {}",
            top_j_resume_start
        );
        for chunk_start in (top_j_resume_start..pp_size).step_by(top_j_batch) {
            let chunk_end = (chunk_start + top_j_batch).min(pp_size);
            tracing::debug!(
                "Wee25Commit::sample_public_params top_j_parts processing idx {}..{} of {}",
                chunk_start,
                chunk_end,
                pp_size
            );
            let chunk_parts = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|idx| {
                    tracing::debug!(
                        "Wee25Commit::sample_public_params top_j_parts idx={}/{}",
                        idx,
                        pp_size
                    );
                    let mut tag = Vec::with_capacity(b"wee25_w_block_".len() + 8);
                    tag.extend_from_slice(b"wee25_w_block_");
                    tag.extend_from_slice(&idx.to_le_bytes());
                    let w_block = hash_sampler.sample_hash(
                        params,
                        hash_key,
                        tag,
                        secret_size,
                        m_b,
                        DistType::FinRingDist,
                    );
                    let mut top_j_acc = M::zero(params, m_b, j_2m_cols);
                    let mut top_j_last_acc = M::zero(params, m_b, l);
                    for block_idx in 0..pp_size {
                        if block_idx % 10 == 0 {
                            tracing::debug!(
                                "Wee25Commit::sample_public_params top_j_parts idx={}/{} block_idx={}/{}",
                                idx,
                                pp_size,
                                block_idx,
                                pp_size
                            );
                        }
                        let t_block = M::from_compact_bytes(params, &t_block_bytes[block_idx]);
                        let block_group = block_idx / m_g;
                        let block_in_group = block_idx % m_g;
                        let unit_row = M::unit_row_vector(params, l, block_group);
                        let j_2m_block = build_j_2m_block(block_idx, block_group);
                        let wt = w_block.clone() * &t_block;
                        let target_block = if idx == block_idx { gadget.clone() - &wt } else { -wt };
                        let local_sampler = TS::new(params, trapdoor_sigma);
                        let t_top_piece =
                            local_sampler.preimage(params, &trapdoor, &b, &target_block);
                        let contrib_j = t_top_piece.clone() * &j_2m_block;
                        let t_top_col =
                            t_top_piece.slice_columns(block_in_group, block_in_group + 1);
                        let contrib_j_last = t_top_col * &unit_row;
                        top_j_acc = top_j_acc + contrib_j;
                        top_j_last_acc = top_j_last_acc + contrib_j_last;
                    }
                    debug_assert_eq!(
                        top_j_acc.row_size(),
                        m_b,
                        "top_j_acc row_size {} must equal m_b {}",
                        top_j_acc.row_size(),
                        m_b
                    );
                    debug_assert_eq!(
                        top_j_last_acc.row_size(),
                        m_b,
                        "top_j_last_acc row_size {} must equal m_b {}",
                        top_j_last_acc.row_size(),
                        m_b
                    );
                    tracing::debug!(
                        "Wee25Commit::sample_public_params top_j_parts idx={}/{} done",
                        idx,
                        pp_size
                    );
                    (idx, top_j_acc, top_j_last_acc)
                })
                .collect::<Vec<_>>();
            let mut chunk_parts = chunk_parts;
            chunk_parts.sort_by_key(|(idx, _, _)| *idx);
            let mut ckpt_top_j = Vec::with_capacity(chunk_parts.len() * 2);
            let chunk_prefix = format!("{checkpoint_prefix}_top_j_part_{chunk_start}");
            let chunk_len = chunk_end - chunk_start;
            for (idx, top_j, top_j_last) in &chunk_parts {
                let offset = idx - chunk_start;
                ckpt_top_j.push((*idx, top_j.clone()));
                ckpt_top_j.push((chunk_start + chunk_len + offset, top_j_last.clone()));
            }
            add_lookup_buffer(get_lookup_buffer(ckpt_top_j, &chunk_prefix));
        }
        tracing::info!(
            "Wee25Commit::sample_public_params t_top_parts elapsed_s={}",
            t_top_parts_start.elapsed().as_secs()
        );
    }

    pub fn commit(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
        public_params: &Wee25PublicParams<M>,
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        let cache_prefix =
            self.commit_cache_prefix(params, public_params.hash_key, msg_stream.len());
        let commit_cache = CommitCache::<M>::new(cache_prefix, msg_stream.len(), self.tree_base);
        let commitment = self.commit_recursive(params, msg_stream, public_params, &commit_cache);
        commit_cache.flush_pending_persist();
        commitment
    }

    pub fn verify(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        commit: &M,
        opening: &M,
        col_range: Option<std::ops::Range<usize>>,
        public_params: &Wee25PublicParams<M>,
    ) -> bool {
        debug_assert_eq!(msg.row_size(), self.secret_size);
        debug_assert_eq!(msg.col_size() % self.m_b, 0);
        let msg_size = msg.col_size() / self.m_b;
        let verifier = self.verifier(params, msg_size, col_range.clone(), public_params);
        let target_msg = if let Some(col_range) = col_range {
            msg.slice_columns(self.m_b * col_range.start, self.m_b * col_range.end)
        } else {
            msg.clone()
        };
        let lhs = commit.clone() * verifier;
        let rhs = target_msg - public_params.b.clone() * opening;
        lhs == rhs
    }

    fn commit_recursive(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
        public_params: &Wee25PublicParams<M>,
        commit_cache: &CommitCache<M>,
    ) -> M {
        let cols = msg_stream.len();
        debug_assert!(
            cols >= self.tree_base,
            "commit expects at least tree_base={} blocks (got {})",
            self.tree_base,
            cols
        );
        if cols == self.tree_base {
            let parts = msg_stream.read(0..cols);
            let refs = parts.iter().collect::<Vec<_>>();
            let msg = parts[0].concat_columns(&refs[1..]);
            let commit = self.commit_base(params, &msg, public_params);
            commit_cache.insert_and_persist(msg_stream.offset, cols, commit.clone());
            return commit;
        }
        debug_assert!(
            cols % self.tree_base == 0,
            "commit expects block count divisible by tree_base={} (got {})",
            self.tree_base,
            cols
        );
        let child_cols = cols / self.tree_base;
        let commits = parallel_iter!(0..self.tree_base)
            .map(|idx| {
                let start = idx * child_cols;
                let end = start + child_cols;
                let part_stream = msg_stream.slice(start..end);
                self.commit_recursive(params, &part_stream, public_params, commit_cache)
            })
            .collect::<Vec<_>>();
        let commit_refs = commits.iter().collect::<Vec<_>>();
        let combined = commits[0].concat_columns(&commit_refs[1..]);
        let commit = self.commit_base(params, &combined, public_params);
        commit_cache.insert_and_persist(msg_stream.offset, cols, commit.clone());
        commit
    }

    fn commit_base(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        public_params: &Wee25PublicParams<M>,
    ) -> M {
        let base_cols = self.tree_base * self.m_b;
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, base_cols),
            "base commit expects shape ({}, {})",
            self.secret_size,
            base_cols
        );
        let hash_sampler = HS::new();
        let cols = msg.col_size();
        (0..cols)
            .into_par_iter()
            .fold(
                || M::zero(params, self.secret_size, self.m_b),
                |mut acc, j| {
                    let decomposed_col = msg.get_column_matrix_decompose(j);
                    for r in 0..self.m_g {
                        let a = decomposed_col.entry(r, 0);
                        let block_idx = j * self.m_g + r;
                        let mut tag = Vec::with_capacity(b"wee25_w_block_".len() + 8);
                        tag.extend_from_slice(b"wee25_w_block_");
                        tag.extend_from_slice(&block_idx.to_le_bytes());
                        let w_block = hash_sampler.sample_hash(
                            params,
                            public_params.hash_key,
                            tag,
                            self.secret_size,
                            self.m_b,
                            DistType::FinRingDist,
                        );
                        acc = acc + (w_block * &a);
                    }
                    acc
                },
            )
            .reduce(|| M::zero(params, self.secret_size, self.m_b), |left, right| left + right)
    }

    pub fn open(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
        col_range: Option<Range<usize>>,
        public_params: &Wee25PublicParams<M>,
        commit_cache: &CommitCache<M>,
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        debug_assert_eq!(
            commit_cache.cols,
            msg_stream.len(),
            "commit cache cols {} mismatch msg cols {}",
            commit_cache.cols,
            msg_stream.len()
        );
        let v_base = self.verifier_base(params, public_params, false);
        let v_base_last = self.verifier_base(params, public_params, true);
        let z_prime_cache: DashMap<(usize, usize, usize), M> = DashMap::new();
        let verifier_cache: DashMap<(usize, usize), M> = DashMap::new();
        let cols = msg_stream.len();
        let col_range = col_range.unwrap_or(0..cols);
        debug_assert!(col_range.start < col_range.end, "column range must be non-empty");
        debug_assert!(
            col_range.end <= cols,
            "column range end {} exceeds total columns {}",
            col_range.end,
            cols
        );
        let openings = col_range
            .map(|col_idx| {
                self.open_recursive(
                    params,
                    msg_stream,
                    col_idx,
                    &v_base,
                    &v_base_last,
                    &public_params.top_j,
                    &public_params.top_j_last,
                    commit_cache,
                    &z_prime_cache,
                    &verifier_cache,
                )
            })
            .collect::<Vec<_>>();
        openings[0].concat_columns(&openings[1..].iter().collect::<Vec<_>>())
    }

    fn open_recursive(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
        col_idx: usize,
        v_base: &M,
        v_base_last: &M,
        t_top_j: &[Vec<u8>],
        t_top_j_last: &[Vec<u8>],
        commit_cache: &CommitCache<M>,
        z_prime_cache: &DashMap<(usize, usize, usize), M>,
        verifier_cache: &DashMap<(usize, usize), M>,
    ) -> M {
        let cols = msg_stream.len();
        if cols == self.tree_base {
            let parts = msg_stream.read(0..cols);
            let refs = parts.iter().collect::<Vec<_>>();
            let msg = parts[0].concat_columns(&refs[1..]);
            return self.open_base(params, &msg, col_idx, t_top_j, t_top_j_last, true);
        }
        let child_cols = cols / self.tree_base;
        let child_col_idx = col_idx % child_cols;
        let sibling_idx = col_idx / child_cols;
        let commits = (0..self.tree_base)
            .map(|j| {
                let child_offset = msg_stream.offset + j * child_cols;
                commit_cache.get(child_offset, child_cols).unwrap_or_else(|| {
                    panic!(
                        "missing commit cache node for offset={} len={}",
                        child_offset, child_cols
                    )
                })
            })
            .collect::<Vec<_>>();
        let commits_msg = commits[0].concat_columns(&commits[1..].iter().collect::<Vec<_>>());
        let z_prime_key = (msg_stream.offset, cols, sibling_idx);
        let z_prime = if let Some(entry) = z_prime_cache.get(&z_prime_key) {
            entry.clone()
        } else {
            let value =
                self.open_base(params, &commits_msg, sibling_idx, t_top_j, t_top_j_last, false);
            z_prime_cache.insert(z_prime_key, value.clone());
            value
        };
        let child_stream =
            msg_stream.slice(child_cols * sibling_idx..child_cols * (sibling_idx + 1));
        let z_child = self.open_recursive(
            params,
            &child_stream,
            child_col_idx,
            v_base,
            v_base_last,
            t_top_j,
            t_top_j_last,
            commit_cache,
            z_prime_cache,
            verifier_cache,
        );
        let verifier =
            self.verifier_recursive(v_base, v_base_last, child_cols, child_col_idx, verifier_cache);
        z_prime * verifier.decompose() + z_child
    }

    fn open_base(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        col_idx: usize,
        t_top_j: &[Vec<u8>],
        t_top_j_last: &[Vec<u8>],
        is_leaf: bool,
    ) -> M {
        let base_cols = self.tree_base * self.m_b;
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, base_cols),
            "base open expects shape ({}, {})",
            self.secret_size,
            base_cols
        );
        let t_top_parts = if is_leaf { t_top_j_last } else { t_top_j };
        debug_assert!(!t_top_parts.is_empty(), "t_top_parts must be non-empty");
        let sample = M::from_compact_bytes(params, &t_top_parts[0]);
        debug_assert_eq!(
            sample.row_size(),
            self.m_b,
            "t_top part row_size {} must equal m_b {}",
            sample.row_size(),
            self.m_b
        );
        debug_assert!(
            sample.col_size() % self.tree_base == 0,
            "t_top part col_size {} must be divisible by tree_base {}",
            sample.col_size(),
            self.tree_base
        );
        let slice_width = sample.col_size() / self.tree_base;
        let expected_parts = self.tree_base * self.m_b * self.m_g;
        debug_assert_eq!(
            t_top_parts.len(),
            expected_parts,
            "t_top_parts len {} must equal {}",
            t_top_parts.len(),
            expected_parts
        );
        let cols = msg.col_size();
        (0..cols)
            .into_par_iter()
            .fold(
                || M::zero(params, self.m_b, slice_width),
                |mut acc, j| {
                    let decomposed_col = msg.get_column_matrix_decompose(j);
                    for r in 0..self.m_g {
                        let a = decomposed_col.entry(r, 0);
                        let part_idx = j * self.m_g + r;
                        let t_part = M::from_compact_bytes(params, &t_top_parts[part_idx]);
                        debug_assert_eq!(
                            t_part.row_size(),
                            self.m_b,
                            "t_top part row_size {} must equal m_b {}",
                            t_part.row_size(),
                            self.m_b
                        );
                        let t_block = t_part
                            .slice_columns(slice_width * col_idx, slice_width * (col_idx + 1));
                        acc = acc + (t_block * &a);
                    }
                    acc
                },
            )
            .reduce(|| M::zero(params, self.m_b, slice_width), |left, right| left + right)
    }

    pub fn verifier(
        &self,
        params: &<M::P as Poly>::Params,
        cols: usize,
        col_range: Option<Range<usize>>,
        public_params: &Wee25PublicParams<M>,
    ) -> M {
        let col_range = col_range.unwrap_or(0..cols);
        debug_assert!(col_range.start < col_range.end, "column range must be non-empty");
        let mut cursor = cols;
        while cursor > self.tree_base {
            debug_assert!(
                cursor % self.tree_base == 0,
                "cols must be a power of tree_base={} (got {})",
                self.tree_base,
                cols
            );
            cursor /= self.tree_base;
        }
        debug_assert_eq!(
            cursor, self.tree_base,
            "cols must be a power of tree_base={} (got {})",
            self.tree_base, cols
        );
        let log_base_q = self.m_g / self.secret_size;
        let total_cols = cols * log_base_q;
        debug_assert!(
            col_range.end <= total_cols,
            "column range end {} exceeds total columns {}",
            col_range.end,
            total_cols
        );
        let base = self.verifier_base(params, public_params, false);
        let base_last = self.verifier_base(params, public_params, true);
        let cache: DashMap<(usize, usize), M> = DashMap::new();
        let cols_vec = col_range
            .map(|col_idx| self.verifier_recursive(&base, &base_last, cols, col_idx, &cache))
            .collect::<Vec<_>>();
        cols_vec[0].concat_columns(&cols_vec[1..].iter().collect::<Vec<_>>())
    }

    fn verifier_recursive(
        &self,
        base: &M,
        base_last: &M,
        cols: usize,
        col_idx: usize,
        cache: &DashMap<(usize, usize), M>,
    ) -> M {
        // tracing::debug!(
        //     "verifier_recursive start cols={} col_idx={} tree_base={}",
        //     cols,
        //     col_idx,
        //     self.tree_base
        // );
        if let Some(entry) = cache.get(&(cols, col_idx)) {
            // tracing::debug!("verifier_recursive cache hit cols={} col_idx={}", cols, col_idx);
            return entry.clone();
        }
        if cols == self.tree_base {
            // verifier for tree_base * msg matrices, each of which has m_b columns
            let result = base_last.slice_columns(self.m_b * col_idx, self.m_b * (col_idx + 1));
            cache.insert((cols, col_idx), result.clone());
            // tracing::debug!(
            //     "verifier_recursive leaf cols={} col_idx={} m_b={}",
            //     cols,
            //     col_idx,
            //     self.m_b
            // );
            return result;
        }
        let child_cols = cols / self.tree_base;
        let child_idx = col_idx % child_cols;
        let child_col = self.verifier_recursive(base, base_last, child_cols, child_idx, cache);
        let slice_width = base.col_size() / self.tree_base;
        let sibling_idx = col_idx / child_cols;
        let slice = base.slice_columns(slice_width * sibling_idx, slice_width * (sibling_idx + 1));
        let decomposed = child_col.decompose();
        let result = slice * &decomposed;
        cache.insert((cols, col_idx), result.clone());
        // tracing::debug!(
        //     "verifier_recursive computed cols={} col_idx={} child_cols={} child_idx={}
        // sibling_idx={}",     cols,
        //     col_idx,
        //     child_cols,
        //     child_idx,
        //     sibling_idx
        // );
        result
    }

    fn verifier_base(
        &self,
        params: &<M::P as Poly>::Params,
        public_params: &Wee25PublicParams<M>,
        is_leaf: bool,
    ) -> M {
        if is_leaf {
            M::from_compact_bytes(params, &public_params.t_bottom_j_2m_last)
        } else {
            M::from_compact_bytes(params, &public_params.t_bottom_j_2m)
        }
    }

    fn assert_stream_len(&self, cols: usize) {
        debug_assert!(
            cols >= self.tree_base,
            "message block count must be at least tree_base={} (got {})",
            self.tree_base,
            cols
        );
        let mut cursor = cols;
        while cursor > self.tree_base {
            debug_assert!(
                cursor % self.tree_base == 0,
                "message block count must be divisible by tree_base={} (got {})",
                self.tree_base,
                cursor
            );
            cursor /= self.tree_base;
        }
        debug_assert_eq!(
            cursor, self.tree_base,
            "message block count must be a power of tree_base (got {})",
            cols
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::params::DCRTPolyParams,
        sampler::{trapdoor::sampler::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler},
        storage::write::{init_storage_system, wait_for_all_writes},
    };
    use std::time::Instant;
    use tempfile::tempdir;
    use tokio::runtime::Runtime;
    use tracing::info;
    const SIGMA: f64 = 4.578;

    fn concat_blocks<M: PolyMatrix>(blocks: &[M]) -> M {
        let refs = blocks.iter().collect::<Vec<_>>();
        blocks[0].concat_columns(&refs[1..])
    }

    fn sample_public_params_for_test(
        wee25_commit: &Wee25Commit<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >,
        params: &DCRTPolyParams,
        hash_key: [u8; 32],
    ) -> (Wee25PublicParams<DCRTPolyMatrix>, tempfile::TempDir) {
        let tmp_dir = tempdir().unwrap();
        init_storage_system(tmp_dir.path().to_path_buf());
        wee25_commit.sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
            params,
            hash_key,
            tmp_dir.path(),
        );
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let public_params = Wee25PublicParams::<DCRTPolyMatrix>::read_from_storage(
            params,
            tmp_dir.path(),
            wee25_commit,
            hash_key,
        )
        .expect("wee25 public params not found");
        (public_params, tmp_dir)
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;
        let hash_key = [0u8; 32];

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base, SIGMA);
        let (public_params, tmp_dir) =
            sample_public_params_for_test(&wee25_commit, &params, hash_key);
        info!("commit params generated in {:?}", start.elapsed());

        let msg_blocks = (0..cols)
            .map(|_| DCRTPolyMatrix::zero(&params, secret_size, wee25_commit.m_b))
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let start = Instant::now();
        let commitment = wee25_commit.commit(&params, &msg_stream, &public_params);
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let commit_cache = CommitCache::<DCRTPolyMatrix>::load(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
            cols,
        )
        .expect("commit cache not found");
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = wee25_commit.verifier(&params, cols, None, &public_params);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = wee25_commit.open(&params, &msg_stream, None, &public_params, &commit_cache);
        info!("opening generated in {:?}", start.elapsed());

        assert!(wee25_commit.verify(
            &params,
            &msg_matrix,
            &commitment,
            &opening,
            None,
            &public_params
        ));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;
        let hash_key = [0u8; 32];

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base, SIGMA);
        let (public_params, tmp_dir) =
            sample_public_params_for_test(&wee25_commit, &params, hash_key);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(
                    &params,
                    secret_size,
                    wee25_commit.m_b,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let start = Instant::now();
        let commitment = wee25_commit.commit(&params, &msg_stream, &public_params);
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let commit_cache = CommitCache::<DCRTPolyMatrix>::load(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
            cols,
        )
        .expect("commit cache not found");
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = wee25_commit.verifier(&params, cols, None, &public_params);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = wee25_commit.open(&params, &msg_stream, None, &public_params, &commit_cache);
        info!("opening generated in {:?}", start.elapsed());

        assert!(wee25_commit.verify(
            &params,
            &msg_matrix,
            &commitment,
            &opening,
            None,
            &public_params
        ));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_invalid_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;
        let hash_key = [0u8; 32];

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base, SIGMA);
        let (public_params, tmp_dir) =
            sample_public_params_for_test(&wee25_commit, &params, hash_key);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks1 = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(
                    &params,
                    secret_size,
                    wee25_commit.m_b,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let msg_stream1 = MsgMatrixStream::from_blocks(msg_blocks1);
        let start = Instant::now();
        let commitment = wee25_commit.commit(&params, &msg_stream1, &public_params);
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let commit_cache = CommitCache::<DCRTPolyMatrix>::load(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
            cols,
        )
        .expect("commit cache not found");
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = wee25_commit.verifier(&params, cols, None, &public_params);
        info!("verifier generated in {:?}", start.elapsed());
        let msg_blocks2 = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(
                    &params,
                    secret_size,
                    wee25_commit.m_b,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let msg_matrix2 = concat_blocks(&msg_blocks2);
        let msg_stream2 = MsgMatrixStream::from_blocks(msg_blocks2);
        let start = Instant::now();
        let opening = wee25_commit.open(&params, &msg_stream2, None, &public_params, &commit_cache);
        info!("opening generated in {:?}", start.elapsed());

        assert!(!wee25_commit.verify(
            &params,
            &msg_matrix2,
            &commitment,
            &opening,
            None,
            &public_params
        ));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_partial_open_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;
        let hash_key = [0u8; 32];

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base, SIGMA);
        let (public_params, tmp_dir) =
            sample_public_params_for_test(&wee25_commit, &params, hash_key);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(
                    &params,
                    secret_size,
                    wee25_commit.m_b,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);

        let start = Instant::now();
        let commitment = wee25_commit.commit(&params, &msg_stream, &public_params);
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let commit_cache = CommitCache::<DCRTPolyMatrix>::load(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
            cols,
        )
        .expect("commit cache not found");
        info!("commitment generated in {:?}", start.elapsed());

        let col_range = 1..3;
        let start = Instant::now();
        let opening = wee25_commit.open(
            &params,
            &msg_stream,
            Some(col_range.clone()),
            &public_params,
            &commit_cache,
        );
        info!("opening generated in {:?}", start.elapsed());

        assert!(wee25_commit.verify(
            &params,
            &msg_matrix,
            &commitment,
            &opening,
            Some(col_range),
            &public_params
        ));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_commit_cache_load_open_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;
        let hash_key = [7u8; 32];

        let tmp_dir = tempdir().unwrap();
        init_storage_system(tmp_dir.path().to_path_buf());

        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base, SIGMA);
        wee25_commit.sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
            &params,
            hash_key,
            tmp_dir.path(),
        );
        let rt = Runtime::new().unwrap();
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");
        let public_params = Wee25PublicParams::<DCRTPolyMatrix>::read_from_storage(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
        )
        .expect("wee25 public params not found");

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(
                    &params,
                    secret_size,
                    wee25_commit.m_b,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let commitment = wee25_commit.commit(&params, &msg_stream, &public_params);
        rt.block_on(wait_for_all_writes(tmp_dir.path().to_path_buf()))
            .expect("wait_for_all_writes failed");

        let loaded_cache = CommitCache::<DCRTPolyMatrix>::load(
            &params,
            tmp_dir.path(),
            &wee25_commit,
            hash_key,
            cols,
        )
        .expect("commit cache not found");
        assert!(loaded_cache.len() > 0);
        let opening = wee25_commit.open(&params, &msg_stream, None, &public_params, &loaded_cache);

        assert!(wee25_commit.verify(
            &params,
            &msg_matrix,
            &commitment,
            &opening,
            None,
            &public_params
        ));
    }
}
