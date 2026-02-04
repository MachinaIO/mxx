use std::{marker::PhantomData, ops::Range, sync::Arc};

use crate::{
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    storage::{
        read::read_matrix_from_multi_batch,
        write::{add_lookup_buffer, get_lookup_buffer},
    },
};
use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub const WEE25_PUBLIC_PARAMS_PREFIX: &str = "wee25_public_params";

#[derive(Debug, Clone)]
pub struct Wee25Commit<M: PolyMatrix, HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync> {
    pub secret_size: usize,
    pub tree_base: usize,
    pub m_b: usize,
    pub m_g: usize,
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

    pub fn write_to_storage(&self, params: &<M::P as Poly>::Params, id_prefix: &str) -> bool
    where
        M: Send,
    {
        let mut ok = true;
        ok &= add_lookup_buffer(get_lookup_buffer(
            vec![(0, self.b.clone())],
            &format!("{id_prefix}_b"),
        ));
        for (idx, bytes) in self.top_j.iter().enumerate() {
            let part = M::from_compact_bytes(params, bytes);
            ok &= add_lookup_buffer(get_lookup_buffer(
                vec![(0, part)],
                &format!("{id_prefix}_top_j_part_{idx}"),
            ));
        }
        for (idx, bytes) in self.top_j_last.iter().enumerate() {
            let part = M::from_compact_bytes(params, bytes);
            ok &= add_lookup_buffer(get_lookup_buffer(
                vec![(0, part)],
                &format!("{id_prefix}_top_j_last_part_{idx}"),
            ));
        }
        let t_bottom_j_2m = M::from_compact_bytes(params, &self.t_bottom_j_2m);
        let t_bottom_j_2m_last = M::from_compact_bytes(params, &self.t_bottom_j_2m_last);
        ok &= add_lookup_buffer(get_lookup_buffer(
            vec![(0, t_bottom_j_2m)],
            &format!("{id_prefix}_t_bottom_j_2m"),
        ));
        ok &= add_lookup_buffer(get_lookup_buffer(
            vec![(0, t_bottom_j_2m_last)],
            &format!("{id_prefix}_t_bottom_j_2m_last"),
        ));
        ok
    }

    pub fn read_from_storage(
        params: &<M::P as Poly>::Params,
        dir: &std::path::Path,
        id_prefix: &str,
        hash_key: [u8; 32],
    ) -> Option<Self> {
        let b = read_matrix_from_multi_batch::<M>(params, dir, &format!("{id_prefix}_b"), 0)?;
        let mut top_j_parts = Vec::new();
        for idx in 0.. {
            let part = read_matrix_from_multi_batch::<M>(
                params,
                dir,
                &format!("{id_prefix}_top_j_part_{idx}"),
                0,
            );
            match part {
                Some(mat) => top_j_parts.push(mat.to_compact_bytes()),
                None => break,
            }
        }
        let mut top_j_last_parts = Vec::new();
        for idx in 0.. {
            let part = read_matrix_from_multi_batch::<M>(
                params,
                dir,
                &format!("{id_prefix}_top_j_last_part_{idx}"),
                0,
            );
            match part {
                Some(mat) => top_j_last_parts.push(mat.to_compact_bytes()),
                None => break,
            }
        }
        let t_bottom_j_2m = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{id_prefix}_t_bottom_j_2m"),
            0,
        )?;
        let t_bottom_j_2m_last = read_matrix_from_multi_batch::<M>(
            params,
            dir,
            &format!("{id_prefix}_t_bottom_j_2m_last"),
            0,
        )?;
        if top_j_parts.is_empty() || top_j_last_parts.is_empty() {
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
    pub fn new(params: &<M::P as Poly>::Params, secret_size: usize, tree_base: usize) -> Self {
        debug_assert!(tree_base >= 2, "tree_base must be at least 2");
        let log_base_q = params.modulus_digits();
        let m_g = secret_size * log_base_q;
        // For the current trapdoor sampler, b has size d x (2d + d*log_base_q).
        let m_b = secret_size * (2 + log_base_q);
        Self { secret_size, tree_base, m_b, m_g, _marker: PhantomData }
    }

    pub fn sample_public_params<
        US: PolyUniformSampler<M = M> + Send + Sync,
        TS: PolyTrapdoorSampler<M = M>,
    >(
        &self,
        params: &<M::P as Poly>::Params,
        hash_key: [u8; 32],
        trapdoor_sigma: f64,
    ) -> Wee25PublicParams<M> {
        let secret_size = self.secret_size;
        let tree_base = self.tree_base;
        let m_b = self.m_b;
        let m_g = self.m_g;

        let trapdoor_sampler = TS::new(params, trapdoor_sigma);
        let (trapdoor, b) = trapdoor_sampler.trapdoor(params, secret_size);
        debug_assert_eq!(
            b.col_size(),
            m_b,
            "Wee25PublicParams m_b mismatch: expected {}, got {}",
            m_b,
            b.col_size()
        );
        let uniform_sampler = US::new();
        let pp_size = tree_base * m_b * m_g;
        tracing::debug!(
            "Wee25Commit::sample_public_params secret_size={} m_b={} m_g={} pp_size={}",
            secret_size,
            m_b,
            m_g,
            pp_size
        );
        let hash_sampler = HS::new();
        let l = tree_base * m_b;
        let log_base_q = params.modulus_digits();
        let j_2m_cols = l * log_base_q;
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
        let zero_top_j = M::zero(params, m_b, j_2m_cols);
        let zero_top_j_last = M::zero(params, m_b, l);
        let mut t_bottom_j_2m = zero_top_j.clone();
        let mut t_bottom_j_2m_last = zero_top_j_last.clone();

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
        tracing::debug!(
            "Wee25Commit::sample_public_params starting t_block sampling for pp_size={}",
            pp_size
        );

        // Sample t_block once, store compact bytes, and update t_bottom aggregates.
        let t_block_batch = std::env::var("WEE25_TBLOCK_PARALLEL_BATCH")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .filter(|&val| val > 0)
            .unwrap_or(10);
        let mut t_block_bytes = vec![Vec::new(); pp_size];
        for chunk_start in (0..pp_size).step_by(t_block_batch) {
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
                    let block_group = block_idx / m_g;
                    let block_in_group = block_idx % m_g;
                    let unit_row = M::unit_row_vector(params, l, block_group);
                    let j_2m_block = build_j_2m_block(block_idx, block_group);
                    let contrib_j_2m = t_block.clone() * &j_2m_block;
                    let t_block_col = t_block.slice_columns(block_in_group, block_in_group + 1);
                    let contrib_j_2m_last = t_block_col * &unit_row;
                    (block_idx, t_block_bytes, contrib_j_2m, contrib_j_2m_last)
                })
                .collect::<Vec<_>>();
            for (block_idx, bytes, contrib_j_2m, contrib_j_2m_last) in chunk_results {
                t_block_bytes[block_idx] = bytes;
                t_bottom_j_2m = t_bottom_j_2m + contrib_j_2m;
                t_bottom_j_2m_last = t_bottom_j_2m_last + contrib_j_2m_last;
            }
        }
        let t_bottom_j_2m_bytes = t_bottom_j_2m.to_compact_bytes();
        drop(t_bottom_j_2m);
        let t_bottom_j_2m_last_bytes = t_bottom_j_2m_last.to_compact_bytes();
        drop(t_bottom_j_2m_last);
        tracing::debug!("Wee25Commit::sample_public_params completed t_block sampling");

        // Reverse the loop order: outer over j_2m columns (idx), inner over t_block/j_2m rows.
        let top_j_parts: Vec<(Vec<u8>, Vec<u8>)> = (0..pp_size)
            .into_par_iter()
            .map(|idx| {
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
                let mut top_j_acc = zero_top_j.clone();
                let mut top_j_last_acc = zero_top_j_last.clone();
                for block_idx in 0..pp_size {
                    let t_block = M::from_compact_bytes(params, &t_block_bytes[block_idx]);
                    let block_group = block_idx / m_g;
                    let block_in_group = block_idx % m_g;
                    let unit_row = M::unit_row_vector(params, l, block_group);
                    let j_2m_block = build_j_2m_block(block_idx, block_group);
                    let wt = w_block.clone() * &t_block;
                    let target_block = if idx == block_idx { gadget.clone() - &wt } else { -wt };
                    let local_sampler = TS::new(params, trapdoor_sigma);
                    let t_top_piece = local_sampler.preimage(params, &trapdoor, &b, &target_block);
                    let contrib_j = t_top_piece.clone() * &j_2m_block;
                    let t_top_col = t_top_piece.slice_columns(block_in_group, block_in_group + 1);
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
                (top_j_acc.to_compact_bytes(), top_j_last_acc.to_compact_bytes())
            })
            .collect();
        tracing::info!(
            "Wee25Commit::sample_public_params t_top_parts elapsed_s={}",
            t_top_parts_start.elapsed().as_secs()
        );
        let (top_j_parts, top_j_last_parts): (Vec<Vec<u8>>, Vec<Vec<u8>>) =
            top_j_parts.into_iter().unzip();
        drop(t_block_bytes);
        drop(zero_top_j);
        drop(zero_top_j_last);
        Wee25PublicParams::new(
            b,
            top_j_parts,
            top_j_last_parts,
            t_bottom_j_2m_bytes,
            t_bottom_j_2m_last_bytes,
            hash_key,
        )
    }

    pub fn commit(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
        public_params: &Wee25PublicParams<M>,
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        self.commit_recursive(params, msg_stream, public_params)
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
            return self.commit_base(params, &msg, public_params);
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
                self.commit_recursive(params, &part_stream, public_params)
            })
            .collect::<Vec<_>>();
        let commit_refs = commits.iter().collect::<Vec<_>>();
        let combined = commits[0].concat_columns(&commit_refs[1..]);
        self.commit_base(params, &combined, public_params)
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
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        let v_base = self.verifier_base(params, public_params, false);
        let v_base_last = self.verifier_base(params, public_params, true);
        let commit_cache: DashMap<(usize, usize), M> = DashMap::new();
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
                    &commit_cache,
                    &z_prime_cache,
                    &verifier_cache,
                    public_params,
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
        commit_cache: &DashMap<(usize, usize), M>,
        z_prime_cache: &DashMap<(usize, usize, usize), M>,
        verifier_cache: &DashMap<(usize, usize), M>,
        public_params: &Wee25PublicParams<M>,
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
        let commits = parallel_iter!(0..self.tree_base)
            .map(|j| {
                let start = j * child_cols;
                let end = start + child_cols;
                let part_stream = msg_stream.slice(start..end);
                let key = (part_stream.offset, part_stream.len);
                if let Some(entry) = commit_cache.get(&key) {
                    return entry.clone();
                }
                let commit = self.commit_recursive(params, &part_stream, public_params);
                commit_cache.insert(key, commit.clone());
                commit
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
            public_params,
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
    };
    use std::time::Instant;
    use tracing::info;
    const SIGMA: f64 = 4.578;

    fn concat_blocks<M: PolyMatrix>(blocks: &[M]) -> M {
        let refs = blocks.iter().collect::<Vec<_>>();
        blocks[0].concat_columns(&refs[1..])
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let tree_base = 2;
        let cols = 4;

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base);
        let public_params = wee25_commit
            .sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
                &params, [0u8; 32], SIGMA,
            );
        info!("commit params generated in {:?}", start.elapsed());

        let msg_blocks = (0..cols)
            .map(|_| DCRTPolyMatrix::zero(&params, secret_size, wee25_commit.m_b))
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let start = Instant::now();
        let commitment = wee25_commit.commit(&params, &msg_stream, &public_params);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = wee25_commit.verifier(&params, cols, None, &public_params);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = wee25_commit.open(&params, &msg_stream, None, &public_params);
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

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base);
        let public_params = wee25_commit
            .sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
                &params, [0u8; 32], SIGMA,
            );
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
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = wee25_commit.verifier(&params, cols, None, &public_params);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = wee25_commit.open(&params, &msg_stream, None, &public_params);
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

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base);
        let public_params = wee25_commit
            .sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
                &params, [0u8; 32], SIGMA,
            );
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
        let opening = wee25_commit.open(&params, &msg_stream2, None, &public_params);
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

        let start = Instant::now();
        let wee25_commit = Wee25Commit::<
            DCRTPolyMatrix,
            crate::sampler::hash::DCRTPolyHashSampler<keccak_asm::Keccak256>,
        >::new(&params, secret_size, tree_base);
        let public_params = wee25_commit
            .sample_public_params::<DCRTPolyUniformSampler, DCRTPolyTrapdoorSampler>(
                &params, [0u8; 32], SIGMA,
            );
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
        info!("commitment generated in {:?}", start.elapsed());

        let col_range = 1..3;
        let start = Instant::now();
        let opening =
            wee25_commit.open(&params, &msg_stream, Some(col_range.clone()), &public_params);
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
}
