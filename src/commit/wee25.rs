use std::{ops::Range, sync::Arc};

use crate::{
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler},
};
use dashmap::DashMap;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone)]
pub struct Wee25Commit<M: PolyMatrix> {
    pub secret_size: usize,
    pub tree_base: usize,
    pub m_b: usize,
    pub m_g: usize,
    pub b: M,
    pub w: M,
    pub t_top: M,
    pub t_bottom: M,
    pub j_2m: M,
    pub j_2m_last: M,
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

impl<M: PolyMatrix> Wee25Commit<M> {
    pub fn setup<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
        tree_base: usize,
    ) -> Self {
        debug_assert!(tree_base >= 2, "tree_base must be at least 2");
        let trapdoor_sampler = TS::new(params, trapdoor_sigma);
        let (trapdoor, b) = trapdoor_sampler.trapdoor(params, secret_size);
        let uniform_sampler = US::new();
        let m_b = b.col_size();
        let log_base_q = params.modulus_digits();
        let m_g = secret_size * log_base_q;
        let pp_size = tree_base * m_b * m_g;
        tracing::debug!(
            "Wee25Commit::setup secret_size={} m_b={} m_g={} pp_size={}",
            secret_size,
            m_b,
            m_g,
            pp_size
        );
        let w = uniform_sampler.sample_uniform(
            params,
            pp_size * secret_size,
            m_b,
            DistType::FinRingDist,
        );
        let t_ncol = pp_size * m_g;
        let t_bottom = uniform_sampler.sample_uniform(
            params,
            m_b,
            t_ncol,
            DistType::GaussDist { sigma: trapdoor_sigma },
        );
        let mut t_tops = Vec::with_capacity(pp_size);
        let gadget = M::gadget_matrix(params, secret_size);
        for idx in 0..pp_size {
            let target = {
                let zeros_before_g = M::zero(params, secret_size, idx * m_g);
                let zeros_after_g = M::zero(params, secret_size, (pp_size - idx - 1) * m_g);
                let g_concated = zeros_before_g.concat_columns(&[&gadget, &zeros_after_g]);
                let wt = w.slice_rows(idx * secret_size, (idx + 1) * secret_size) * &t_bottom;
                g_concated - wt
            };
            t_tops.push(trapdoor_sampler.preimage(params, &trapdoor, &b, &target));
        }
        let t_top = t_tops[0].concat_rows(&t_tops[1..].iter().collect::<Vec<_>>());
        let identity_m = M::identity(params, m_g, None);
        let identity_m_vectorized = identity_m.vectorize_columns();
        let l = tree_base * m_b;
        let j_2m_last = identity_m_vectorized.concat_diag(&vec![&identity_m_vectorized; l - 1]);
        let j_2m = {
            let lmn = l * secret_size * m_g;
            let g_lmn = M::gadget_matrix(params, lmn);
            let g_l = M::gadget_matrix(params, l);
            let mul1 = g_lmn.mul_tensor_identity(&identity_m_vectorized, l);
            let mul2 = mul1 * g_l;
            mul2.decompose()
        };

        Self { secret_size, tree_base, m_b, m_g, b, w, t_top, t_bottom, j_2m, j_2m_last }
    }

    pub fn commit(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        self.commit_recursive(params, msg_stream)
    }

    pub fn verify(
        &self,
        msg: &M,
        commit: &M,
        opening: &M,
        col_range: Option<std::ops::Range<usize>>,
    ) -> bool {
        debug_assert_eq!(msg.row_size(), self.secret_size);
        debug_assert_eq!(msg.col_size() % self.m_b, 0);
        let msg_size = msg.col_size() / self.m_b;
        let verifier = self.verifier(msg_size, col_range.clone());
        let target_msg = if let Some(col_range) = col_range {
            msg.slice_columns(self.m_b * col_range.start, self.m_b * col_range.end)
        } else {
            msg.clone()
        };
        let lhs = commit.clone() * verifier;
        let rhs = target_msg - self.b.clone() * opening;
        lhs == rhs
    }

    fn commit_recursive(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
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
            return self.commit_base(params, &msg);
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
                self.commit_recursive(params, &part_stream)
            })
            .collect::<Vec<_>>();
        let commit_refs = commits.iter().collect::<Vec<_>>();
        let combined = commits[0].concat_columns(&commit_refs[1..]);
        self.commit_base(params, &combined)
    }

    fn commit_base(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        let base_cols = self.tree_base * self.m_b;
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, base_cols),
            "base commit expects shape ({}, {})",
            self.secret_size,
            base_cols
        );
        let cols = msg.col_size();
        (0..cols)
            .into_par_iter()
            .fold(
                || M::zero(params, self.secret_size, self.m_b),
                |mut acc, j| {
                    let decomposed_col = msg.get_column_matrix_decompose(j);
                    for r in 0..self.m_g {
                        let a = decomposed_col.entry(r, 0);
                        let row_start = (j * self.m_g + r) * self.secret_size;
                        let row_end = row_start + self.secret_size;
                        let w_block = self.w.slice_rows(row_start, row_end);
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
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        let v_base = self.verifier_base(false);
        let v_base_last = self.verifier_base(true);
        let top_j = self.t_top.clone() * &self.j_2m;
        let top_j_last = self.t_top.clone() * &self.j_2m_last;
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
        let openings = parallel_iter!(col_range)
            .map(|col_idx| {
                self.open_recursive(
                    params,
                    msg_stream,
                    col_idx,
                    &v_base,
                    &v_base_last,
                    &top_j,
                    &top_j_last,
                    &commit_cache,
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
        t_top_j: &M,
        t_top_j_last: &M,
        commit_cache: &DashMap<(usize, usize), M>,
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
        let commits = parallel_iter!(0..self.tree_base)
            .map(|j| {
                let start = j * child_cols;
                let end = start + child_cols;
                let part_stream = msg_stream.slice(start..end);
                let key = (part_stream.offset, part_stream.len);
                if let Some(entry) = commit_cache.get(&key) {
                    return entry.clone();
                }
                let commit = self.commit_recursive(params, &part_stream);
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
        t_top_j: &M,
        t_top_j_last: &M,
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
        let t_top_j = if is_leaf { t_top_j_last } else { t_top_j };
        let slice_width = t_top_j.col_size() / self.tree_base;
        let t_top_j = t_top_j.slice_columns(slice_width * col_idx, slice_width * (col_idx + 1));
        let cols = msg.col_size();
        (0..cols)
            .into_par_iter()
            .fold(
                || M::zero(params, self.m_b, slice_width),
                |mut acc, j| {
                    let decomposed_col = msg.get_column_matrix_decompose(j);
                    for r in 0..self.m_g {
                        let a = decomposed_col.entry(r, 0);
                        let row_start = (j * self.m_g + r) * self.m_b;
                        let row_end = row_start + self.m_b;
                        let t_block = t_top_j.slice_rows(row_start, row_end);
                        acc = acc + (t_block * &a);
                    }
                    acc
                },
            )
            .reduce(|| M::zero(params, self.m_b, slice_width), |left, right| left + right)
    }

    pub fn verifier(&self, cols: usize, col_range: Option<Range<usize>>) -> M {
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
        let base = self.verifier_base(false);
        let base_last = self.verifier_base(true);
        let cache: DashMap<(usize, usize), M> = DashMap::new();
        let cols_vec = parallel_iter!(col_range)
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
        tracing::debug!(
            "verifier_recursive start cols={} col_idx={} tree_base={}",
            cols,
            col_idx,
            self.tree_base
        );
        if let Some(entry) = cache.get(&(cols, col_idx)) {
            tracing::debug!("verifier_recursive cache hit cols={} col_idx={}", cols, col_idx);
            return entry.clone();
        }
        if cols == self.tree_base {
            // verifier for tree_base * msg matrices, each of which has m_b columns
            let result = base_last.slice_columns(self.m_b * col_idx, self.m_b * (col_idx + 1));
            cache.insert((cols, col_idx), result.clone());
            tracing::debug!(
                "verifier_recursive leaf cols={} col_idx={} m_b={}",
                cols,
                col_idx,
                self.m_b
            );
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
        tracing::debug!(
            "verifier_recursive computed cols={} col_idx={} child_cols={} child_idx={} sibling_idx={}",
            cols,
            col_idx,
            child_cols,
            child_idx,
            sibling_idx
        );
        result
    }

    fn verifier_base(&self, is_leaf: bool) -> M {
        let j_2m = if is_leaf { &self.j_2m_last } else { &self.j_2m };
        self.t_bottom.clone() * j_2m
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
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let cols = 4;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let msg_blocks =
            (0..cols).map(|_| DCRTPolyMatrix::zero(&params, secret_size, m_b)).collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let start = Instant::now();
        let commitment = commit_params.commit(&params, &msg_stream);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier(cols, None);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open(&params, &msg_stream, None);
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify(&msg_matrix, &commitment, &opening, None));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let cols = 4;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(&params, secret_size, m_b, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);
        let start = Instant::now();
        let commitment = commit_params.commit(&params, &msg_stream);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier(cols, None);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open(&params, &msg_stream, None);
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify(&msg_matrix, &commitment, &opening, None));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_invalid_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let cols = 4;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks1 = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(&params, secret_size, m_b, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let msg_stream1 = MsgMatrixStream::from_blocks(msg_blocks1);
        let start = Instant::now();
        let commitment = commit_params.commit(&params, &msg_stream1);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier(cols, None);
        info!("verifier generated in {:?}", start.elapsed());
        let msg_blocks2 = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(&params, secret_size, m_b, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let msg_matrix2 = concat_blocks(&msg_blocks2);
        let msg_stream2 = MsgMatrixStream::from_blocks(msg_blocks2);
        let start = Instant::now();
        let opening = commit_params.open(&params, &msg_stream2, None);
        info!("opening generated in {:?}", start.elapsed());

        assert!(!commit_params.verify(&msg_matrix2, &commitment, &opening, None));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_partial_open_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let cols = 4;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_blocks = (0..cols)
            .map(|_| {
                uniform_sampler.sample_uniform(&params, secret_size, m_b, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let msg_matrix = concat_blocks(&msg_blocks);
        let msg_stream = MsgMatrixStream::from_blocks(msg_blocks);

        let start = Instant::now();
        let commitment = commit_params.commit(&params, &msg_stream);
        info!("commitment generated in {:?}", start.elapsed());

        let col_range = 1..3;
        let start = Instant::now();
        let opening = commit_params.open(&params, &msg_stream, Some(col_range.clone()));
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify(&msg_matrix, &commitment, &opening, Some(col_range)));
    }
}
