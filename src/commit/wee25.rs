use std::{ops::Range, sync::Arc};

use crate::{
    matrix::PolyMatrix,
    parallel_iter,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler},
};
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

    // pub fn setup_vector<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
    //     params: &<M::P as Poly>::Params,
    //     secret_size: usize,
    //     trapdoor_sigma: f64,
    //     tree_base: usize,
    // ) -> Self {
    //     Self::setup::<US, TS>(params, secret_size, trapdoor_sigma, tree_base)
    // }

    pub fn commit(
        &self,
        params: &<M::P as Poly>::Params,
        msg_stream: &MsgMatrixStream<'_, M>,
    ) -> M {
        self.assert_stream_len(msg_stream.len());
        self.commit_recursive(params, msg_stream)
    }

    // pub fn commit_vector(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
    //     let identity = M::identity(params, self.secret_size, None);
    //     let matrix = msg.tensor(&identity);
    //     self.commit_matrix(params, &matrix)
    // }

    // pub fn open_vector(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
    //     let identity = M::identity(params, self.secret_size, None);
    //     let matrix = msg.tensor(&identity);
    //     self.open_matrix(params, &matrix)
    // }

    // pub fn verify_matrix(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     commit: &M,
    //     opening: &M,
    //     col_range: Option<Range<usize>>,
    // ) -> bool {
    //     debug_assert_eq!(msg.row_size(), self.secret_size);
    //     let msg_size = msg.col_size();
    //     let g_l = M::gadget_matrix(params, msg_size);
    //     let v = self.verifier(msg_size, col_range);
    //     let lhs = commit.clone() * v;
    //     let rhs_msg = msg.clone() * g_l;
    //     let rhs_open = self.b.clone() * opening.clone();
    //     let rhs = rhs_msg - rhs_open;
    //     lhs == rhs
    // }

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
        // debug_assert!(col_range.start < col_range.end, "column range must be non-empty");
        // let msg_size = msg.col_size();
        // let log_base_q = self.m_g / self.secret_size;
        // let total_cols = msg_size * log_base_q;
        // debug_assert!(
        //     col_range.end <= total_cols,
        //     "column range end {} exceeds total columns {}",
        //     col_range.end,
        //     total_cols
        // );
        // debug_assert_eq!(
        //     opening_cols.col_size(),
        //     col_range.end - col_range.start,
        //     "opening column count must match range length"
        // );
        // let g_l = M::gadget_matrix(params, msg_size);
        // let g_l_slice = g_l.slice_columns(col_range.start, col_range.end);
        // let v = self.verifier_for_length_columns(msg_size, col_range);
        // let lhs = commit.clone() * v;
        // let rhs_msg = msg.clone() * g_l_slice;
        // let rhs_open = self.b.clone() * opening_cols.clone();
        // let rhs = rhs_msg - rhs_open;
        // lhs == rhs
    }

    // pub fn verify_vector(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     commit: &M,
    //     opening: &M,
    // ) -> bool {
    //     let identity = M::identity(params, self.secret_size, None);
    //     let matrix = msg.tensor(&identity);
    //     self.verify_matrix(params, &matrix, commit, opening)
    // }

    // pub fn verify_vector_range(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg_vector: &M,
    //     commit: &M,
    //     opening_cols: &M,
    //     vector_range: std::ops::Range<usize>,
    // ) -> bool {
    //     let vector_len = msg_vector.col_size();
    //     let col_range = self.vector_range_to_opening_columns(vector_len, vector_range);
    //     let identity = M::identity(params, self.secret_size, None);
    //     let matrix = msg_vector.tensor(&identity);
    //     self.verify_matrix_columns(params, &matrix, commit, opening_cols, col_range)
    // }

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

    // fn open_matrix_recursive(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     verifier_cache: &mut std::collections::HashMap<usize, M>,
    // ) -> (M, M) {
    //     let cols = msg.col_size();
    //     debug_assert!(
    //         cols % self.m_b == 0,
    //         "open_matrix expects column count divisible by m_b={} (got {})",
    //         self.m_b,
    //         cols
    //     );
    //     let base_cols = self.tree_base * self.m_b;
    //     if cols == base_cols {
    //         let opening = self.open_base(params, msg);
    //         let commitment = self.commit_base(params, msg);
    //         return (opening, commitment);
    //     }
    //     debug_assert!(
    //         cols % self.tree_base == 0,
    //         "open_matrix expects column count divisible by tree_base={} (got {})",
    //         self.tree_base,
    //         cols
    //     );
    //     let part_cols = cols / self.tree_base;
    //     let mut zs = Vec::with_capacity(self.tree_base);
    //     let mut cs = Vec::with_capacity(self.tree_base);
    //     for idx in 0..self.tree_base {
    //         let start = idx * part_cols;
    //         let end = start + part_cols;
    //         let part = msg.slice_columns(start, end);
    //         let (z, c) = self.open_matrix_recursive(params, &part, verifier_cache);
    //         zs.push(z);
    //         cs.push(c);
    //     }
    //     let c_refs = cs.iter().collect::<Vec<_>>();
    //     let combined_c = cs[0].concat_columns(&c_refs[1..]);
    //     let z_prime = self.open_base(params, &combined_c);
    //     let v_part = match verifier_cache.get(&part_cols) {
    //         Some(v) => v.clone(),
    //         None => {
    //             let v = self.verifier_for_length(part_cols);
    //             verifier_cache.insert(part_cols, v.clone());
    //             v
    //         }
    //     };
    //     let adjusted = z_prime.mul_tensor_identity_decompose(&v_part, self.tree_base);
    //     let z_refs = zs.iter().collect::<Vec<_>>();
    //     let z_concat = zs[0].concat_columns(&z_refs[1..]);
    //     let opening = adjusted + z_concat;
    //     let commitment = self.commit_base(params, &combined_c);
    //     (opening, commitment)
    // }

    // fn open_base_intermediate(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
    //     let cols = msg.col_size();
    //     (0..cols)
    //         .into_par_iter()
    //         .fold(
    //             || M::zero(params, self.m_b, self.t_top.col_size()),
    //             |mut acc, j| {
    //                 let decomposed_col = msg.get_column_matrix_decompose(j);
    //                 for r in 0..self.m_g {
    //                     let a = decomposed_col.entry(r, 0);
    //                     let row_start = (j * self.m_g + r) * self.m_b;
    //                     let row_end = row_start + self.m_b;
    //                     let t_block = self.t_top.slice_rows(row_start, row_end);
    //                     acc = acc + (t_block * &a);
    //                 }
    //                 acc
    //             },
    //         )
    //         .reduce(|| M::zero(params, self.m_b, self.t_top.col_size()), |left, right| left +
    // right) }

    // fn open_base_columns_from_acc(&self, acc: &M, col_range: std::ops::Range<usize>) -> M {
    //     debug_assert!(
    //         col_range.start < col_range.end,
    //         "open_base_columns expects non-empty column range"
    //     );
    //     debug_assert!(
    //         col_range.end <= self.j_2m.col_size(),
    //         "open_base_columns range end {} exceeds {}",
    //         col_range.end,
    //         self.j_2m.col_size()
    //     );
    //     let j_slice = self.j_2m.slice_columns(col_range.start, col_range.end);
    //     acc.clone() * &j_slice
    // }

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
                self.commit_recursive(params, &part_stream)
            })
            .collect::<Vec<_>>();
        let commits_msg = commits[0].concat_columns(&commits[1..].iter().collect::<Vec<_>>());
        let z_prime =
            self.open_base(params, &commits_msg, sibling_idx, t_top_j, t_top_j_last, false);
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
        );
        let verifier = self.verifier_recursive(v_base, v_base_last, child_cols, child_col_idx);
        z_prime * verifier.decompose() + z_child

        // let part_cols = cols / self.tree_base;
        // let part_open_cols = part_cols * log_base_q;

        // let mut commits = Vec::with_capacity(self.tree_base);
        // let mut parts = Vec::with_capacity(self.tree_base);
        // for idx in 0..self.tree_base {
        //     let start = idx * part_cols;
        //     let end = start + part_cols;
        //     let part = msg.slice_columns(start, end);
        //     commits.push(self.commit_matrix_recursive(params, &part));
        //     parts.push(part);
        // }
        // let commit_refs = commits.iter().collect::<Vec<_>>();
        // let combined_c = commits[0].concat_columns(&commit_refs[1..]);
        // let acc_prime = self.open_base_intermediate(params, &combined_c);

        // let mut adjusted_segments = Vec::new();
        // for i in 0..self.tree_base {
        //     let block_start = i * part_open_cols;
        //     let block_end = block_start + part_open_cols;
        //     let start = col_range.start.max(block_start);
        //     let end = col_range.end.min(block_end);
        //     if start >= end {
        //         continue;
        //     }
        //     let sub_range = (start - block_start)..(end - block_start);
        //     let v_part_cols = self.verifier_for_length_columns(part_cols, sub_range);
        //     let slice_width = self.m_b * log_base_q;
        //     let z_slice_range = (i * slice_width)..((i + 1) * slice_width);
        //     let z_prime_slice = self.open_base_columns_from_acc(&acc_prime, z_slice_range);
        //     let adjusted_block = z_prime_slice.mul_decompose(&v_part_cols);
        //     adjusted_segments.push(adjusted_block);
        // }
        // let adjusted = if adjusted_segments.is_empty() {
        //     M::zero(params, self.m_b, 0)
        // } else {
        //     let refs = adjusted_segments.iter().collect::<Vec<_>>();
        //     adjusted_segments[0].concat_columns(&refs[1..])
        // };

        // let mut z_segments = Vec::new();
        // for (idx, part) in parts.iter().enumerate() {
        //     let block_start = idx * part_open_cols;
        //     let block_end = block_start + part_open_cols;
        //     let start = col_range.start.max(block_start);
        //     let end = col_range.end.min(block_end);
        //     if start >= end {
        //         continue;
        //     }
        //     let sub_range = (start - block_start)..(end - block_start);
        //     let z_part = self.opening_columns_recursive(params, part, sub_range, log_base_q);
        //     z_segments.push(z_part);
        // }
        // let z_concat = if z_segments.is_empty() {
        //     M::zero(params, self.m_b, 0)
        // } else {
        //     let refs = z_segments.iter().collect::<Vec<_>>();
        //     z_segments[0].concat_columns(&refs[1..])
        // };
        // adjusted + z_concat
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

    // fn open_base(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     col_range: Range<usize>,
    //     is_leaf: bool,
    // ) -> M {
    //     // let acc = self.open_base_intermediate(params, msg);
    //     // self.open_base_columns_from_acc(&acc, col_range)
    // }

    // pub fn verifier_for_length(&self, cols: usize) -> M {
    //     let log_base_q = self.m_g / self.secret_size;
    //     let total_cols = cols * log_base_q;
    //     self.verifier_for_length_columns(cols, 0..total_cols)
    // }

    // fn verifier_for_length_full(&self, cols: usize) -> M {
    //     let base_len = self.tree_base * self.m_b;
    //     debug_assert!(
    //         cols % base_len == 0,
    //         "verifier_for_length expects multiple of {} cols (got {})",
    //         base_len,
    //         cols
    //     );
    //     let base = self.verifier_base();
    //     if cols == base_len {
    //         return base;
    //     }
    //     let mut current = base.clone();
    //     let mut current_len = base_len;
    //     while current_len < cols {
    //         current = base.clone().mul_tensor_identity_decompose(&current, self.tree_base);
    //         current_len *= self.tree_base;
    //     }
    //     current
    // }

    pub fn verifier(&self, cols: usize, col_range: Option<Range<usize>>) -> M {
        let col_range = col_range.unwrap_or(0..cols);
        debug_assert!(col_range.start < col_range.end, "column range must be non-empty");
        debug_assert!(cols.is_power_of_two(), "cols must be a power of two (got {})", cols);
        let log_base_q = self.m_g / self.secret_size;
        let total_cols = cols * log_base_q;
        debug_assert!(
            col_range.end <= total_cols,
            "column range end {} exceeds total columns {}",
            col_range.end,
            total_cols
        );
        // if col_range.start == 0 && col_range.end == total_cols {
        //     return self.verifier_for_length_full(cols);
        // }
        let base = self.verifier_base(false);
        let base_last = self.verifier_base(true);
        let cols_vec = parallel_iter!(col_range)
            .map(|col_idx| self.verifier_recursive(&base, &base_last, cols, col_idx))
            .collect::<Vec<_>>();
        cols_vec[0].concat_columns(&cols_vec[1..].iter().collect::<Vec<_>>())
        // let mut iter = col_range.into_iter();
        // let first_idx = iter.next().expect("column range must be non-empty");
        // let first = self.verifier_column_with_base(&base, &base_last, cols, first_idx,
        // log_base_q); let mut cols_vec = Vec::new();
        // for idx in iter {
        //     cols_vec.push(self.verifier_column_with_base(&base, &base_last, cols, idx,
        // log_base_q)); }
        // if cols_vec.is_empty() {
        //     first
        // } else {
        //     let refs = cols_vec.iter().collect::<Vec<_>>();
        //     first.concat_columns(&refs)
        // }
    }

    // pub fn verifier_for_vector_range(
    //     &self,
    //     vector_len: usize,
    //     vector_range: std::ops::Range<usize>,
    // ) -> M {
    //     let col_range = self.vector_range_to_opening_columns(vector_len, vector_range);
    //     let cols = vector_len * self.secret_size;
    //     self.verifier_for_length_columns(cols, col_range)
    // }

    // pub fn opening_for_vector_range(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg_vector: &M,
    //     vector_range: std::ops::Range<usize>,
    // ) -> M {
    //     let vector_len = msg_vector.col_size();
    //     let col_range = self.vector_range_to_opening_columns(vector_len, vector_range);
    //     let identity = M::identity(params, self.secret_size, None);
    //     let matrix = msg_vector.tensor(&identity);
    //     self.opening_for_length_columns(params, &matrix, col_range)
    // }

    fn verifier_recursive(
        &self,
        base: &M,
        base_last: &M,
        cols: usize,
        col_idx: usize,
        // log_base_q: usize,
    ) -> M {
        if cols == self.tree_base {
            // verifier for tree_base * msg matrices, each of which has m_b columns
            return base_last.slice_columns(self.m_b * col_idx, self.m_b * (col_idx + 1));
        }
        let child_cols = cols / self.tree_base;
        let child_idx = col_idx % child_cols;
        let child_col = self.verifier_recursive(base, base_last, child_cols, child_idx);
        let slice_width = base.col_size() / self.tree_base;
        let sibling_idx = col_idx / child_cols;
        let slice = base.slice_columns(slice_width * sibling_idx, slice_width * (sibling_idx + 1));
        let decomposed = child_col.decompose();
        slice * &decomposed
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

    // fn vector_range_to_opening_columns(
    //     &self,
    //     vector_len: usize,
    //     vector_range: std::ops::Range<usize>,
    // ) -> std::ops::Range<usize> {
    //     debug_assert!(vector_range.start < vector_range.end, "vector range must be non-empty");
    //     debug_assert!(
    //         vector_range.end <= vector_len,
    //         "vector range end {} exceeds vector length {}",
    //         vector_range.end,
    //         vector_len
    //     );
    //     let log_base_q = self.m_g / self.secret_size;
    //     let start = vector_range.start * self.secret_size * log_base_q;
    //     let end = vector_range.end * self.secret_size * log_base_q;
    //     start..end
    // }

    // pub fn opening_for_length_columns(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     col_range: std::ops::Range<usize>,
    // ) -> M {
    //     debug_assert_eq!(msg.row_size(), self.secret_size);
    //     debug_assert!(col_range.start < col_range.end, "opening column range must be non-empty");
    //     let log_base_q = self.m_g / self.secret_size;
    //     let total_cols = msg.col_size() * log_base_q;
    //     debug_assert!(
    //         col_range.end <= total_cols,
    //         "opening column range end {} exceeds total columns {}",
    //         col_range.end,
    //         total_cols
    //     );
    //     // if col_range.start == 0 && col_range.end == total_cols {
    //     //     let mut verifier_cache = std::collections::HashMap::new();
    //     //     let (opening, _commitment) =
    //     //         self.open_matrix_recursive(params, msg, &mut verifier_cache);
    //     //     return opening;
    //     // }
    //     self.opening_columns_recursive(params, msg, col_range, log_base_q)
    // }

    // fn opening_columns_recursive(
    //     &self,
    //     params: &<M::P as Poly>::Params,
    //     msg: &M,
    //     col_range: std::ops::Range<usize>,
    //     log_base_q: usize,
    // ) -> M {
    //     let cols = msg.col_size();
    //     let base_cols = self.tree_base * self.m_b;
    //     if cols == base_cols {
    //         return self.open_base_columns(params, msg, col_range);
    //     }
    //     let part_cols = cols / self.tree_base;
    //     let part_open_cols = part_cols * log_base_q;

    //     let mut commits = Vec::with_capacity(self.tree_base);
    //     let mut parts = Vec::with_capacity(self.tree_base);
    //     for idx in 0..self.tree_base {
    //         let start = idx * part_cols;
    //         let end = start + part_cols;
    //         let part = msg.slice_columns(start, end);
    //         commits.push(self.commit_matrix_recursive(params, &part));
    //         parts.push(part);
    //     }
    //     let commit_refs = commits.iter().collect::<Vec<_>>();
    //     let combined_c = commits[0].concat_columns(&commit_refs[1..]);
    //     let acc_prime = self.open_base_intermediate(params, &combined_c);

    //     let mut adjusted_segments = Vec::new();
    //     for i in 0..self.tree_base {
    //         let block_start = i * part_open_cols;
    //         let block_end = block_start + part_open_cols;
    //         let start = col_range.start.max(block_start);
    //         let end = col_range.end.min(block_end);
    //         if start >= end {
    //             continue;
    //         }
    //         let sub_range = (start - block_start)..(end - block_start);
    //         let v_part_cols = self.verifier_for_length_columns(part_cols, sub_range);
    //         let slice_width = self.m_b * log_base_q;
    //         let z_slice_range = (i * slice_width)..((i + 1) * slice_width);
    //         let z_prime_slice = self.open_base_columns_from_acc(&acc_prime, z_slice_range);
    //         let adjusted_block = z_prime_slice.mul_decompose(&v_part_cols);
    //         adjusted_segments.push(adjusted_block);
    //     }
    //     let adjusted = if adjusted_segments.is_empty() {
    //         M::zero(params, self.m_b, 0)
    //     } else {
    //         let refs = adjusted_segments.iter().collect::<Vec<_>>();
    //         adjusted_segments[0].concat_columns(&refs[1..])
    //     };

    //     let mut z_segments = Vec::new();
    //     for (idx, part) in parts.iter().enumerate() {
    //         let block_start = idx * part_open_cols;
    //         let block_end = block_start + part_open_cols;
    //         let start = col_range.start.max(block_start);
    //         let end = col_range.end.min(block_end);
    //         if start >= end {
    //             continue;
    //         }
    //         let sub_range = (start - block_start)..(end - block_start);
    //         let z_part = self.opening_columns_recursive(params, part, sub_range, log_base_q);
    //         z_segments.push(z_part);
    //     }
    //     let z_concat = if z_segments.is_empty() {
    //         M::zero(params, self.m_b, 0)
    //     } else {
    //         let refs = z_segments.iter().collect::<Vec<_>>();
    //         z_segments[0].concat_columns(&refs[1..])
    //     };
    //     adjusted + z_concat
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
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

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_zero_commit_invalid_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = base_len * 2;

    //     let start = Instant::now();
    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);
    //     info!("commit params generated in {:?}", start.elapsed());

    //     let mut zero_matrix = DCRTPolyMatrix::zero(&params, secret_size, msg_size);
    //     let start = Instant::now();
    //     let commitment = commit_params.commit_matrix(&params, &zero_matrix);
    //     info!("commitment generated in {:?}", start.elapsed());
    //     zero_matrix.set_entry(0, 0, DCRTPoly::one(&params));
    //     let start = Instant::now();
    //     let _verifier = commit_params.verifier_for_length(msg_size);
    //     info!("verifier generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let opening = commit_params.open_matrix(&params, &zero_matrix);
    //     info!("opening generated in {:?}", start.elapsed());

    //     assert!(!commit_params.verify_matrix(&params, &zero_matrix, &commitment, &opening));
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_random_commit_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = base_len * 2;

    //     let start = Instant::now();
    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);
    //     info!("commit params generated in {:?}", start.elapsed());

    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let msg_matrix =
    //         uniform_sampler.sample_uniform(&params, secret_size, msg_size,
    // DistType::FinRingDist);

    //     let start = Instant::now();
    //     let commitment = commit_params.commit_matrix(&params, &msg_matrix);
    //     info!("commitment generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let _verifier = commit_params.verifier_for_length(msg_size);
    //     info!("verifier generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let opening = commit_params.open_matrix(&params, &msg_matrix);
    //     info!("opening generated in {:?}", start.elapsed());

    //     assert!(commit_params.verify_matrix(&params, &msg_matrix, &commitment, &opening));
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_random_commit_invalid_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = base_len * 2;

    //     let start = Instant::now();
    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);
    //     info!("commit params generated in {:?}", start.elapsed());

    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let msg_matrix =
    //         uniform_sampler.sample_uniform(&params, secret_size, msg_size,
    // DistType::FinRingDist);     let start = Instant::now();
    //     let commitment = commit_params.commit_matrix(&params, &msg_matrix);
    //     info!("commitment generated in {:?}", start.elapsed());

    //     let mut tampered = msg_matrix.clone();
    //     let original_entry = tampered.entry(0, 0);
    //     tampered.set_entry(0, 0, original_entry + DCRTPoly::const_one(&params));

    //     let start = Instant::now();
    //     let _verifier = commit_params.verifier_for_length(msg_size);
    //     info!("verifier generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let opening = commit_params.open_matrix(&params, &tampered);
    //     info!("opening generated in {:?}", start.elapsed());

    //     assert!(!commit_params.verify_matrix(&params, &tampered, &commitment, &opening));
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_random_vector_commit_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = 2 * base_len;

    //     let start = Instant::now();
    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);
    //     info!("commit params generated in {:?}", start.elapsed());

    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let msg_vector =
    //         uniform_sampler.sample_uniform(&params, 1, msg_size / 2, DistType::FinRingDist);

    //     let start = Instant::now();
    //     let commitment = commit_params.commit_vector(&params, &msg_vector);
    //     info!("commitment generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let _verifier = commit_params.verifier_for_length(msg_size);
    //     info!("verifier generated in {:?}", start.elapsed());
    //     let start = Instant::now();
    //     let opening = commit_params.open_vector(&params, &msg_vector);
    //     info!("opening generated in {:?}", start.elapsed());

    //     assert!(commit_params.verify_vector(&params, &msg_vector, &commitment, &opening));
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_partial_matrix_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = base_len * 2;
    //     let log_base_q = params.modulus_digits();
    //     let part_open_cols = base_len * log_base_q;
    //     let total_cols = msg_size * log_base_q;

    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);

    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let msg_matrix =
    //         uniform_sampler.sample_uniform(&params, secret_size, msg_size,
    // DistType::FinRingDist);

    //     let commitment = commit_params.commit_matrix(&params, &msg_matrix);
    //     let full_opening = commit_params.open_matrix(&params, &msg_matrix);
    //     let start = part_open_cols - 1;
    //     let end = (part_open_cols + 2).min(total_cols);
    //     let partial_opening =
    //         commit_params.opening_for_length_columns(&params, &msg_matrix, start..end);
    //     let expected = full_opening.slice_columns(start, end);
    //     assert_eq!(partial_opening, expected);
    //     assert!(commit_params.verify_matrix_columns(
    //         &params,
    //         &msg_matrix,
    //         &commitment,
    //         &partial_opening,
    //         start..end
    //     ));
    // }

    // #[test]
    // #[sequential_test::sequential]
    // fn test_wee25_partial_vector_verify() {
    //     let _ = tracing_subscriber::fmt::try_init();
    //     let params = DCRTPolyParams::new(4, 2, 17, 15);
    //     let secret_size = 1;
    //     let m_b = (&params.modulus_digits() + 2) * secret_size;
    //     let tree_base = 2;
    //     let base_len = tree_base * m_b;
    //     let msg_size = 2 * base_len;
    //     let log_base_q = params.modulus_digits();

    //     let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
    //         DCRTPolyUniformSampler,
    //         DCRTPolyTrapdoorSampler,
    //     >(&params, secret_size, SIGMA, tree_base);

    //     let uniform_sampler = DCRTPolyUniformSampler::new();
    //     let msg_vector = uniform_sampler.sample_uniform(
    //         &params,
    //         1,
    //         msg_size / secret_size,
    //         DistType::FinRingDist,
    //     );

    //     let commitment = commit_params.commit_vector(&params, &msg_vector);
    //     let full_opening = commit_params.open_vector(&params, &msg_vector);
    //     let full_verifier = commit_params.verifier_for_length(msg_size);

    //     let vec_start = 1usize;
    //     let vec_end = msg_vector.col_size().min(3);
    //     let partial_opening =
    //         commit_params.opening_for_vector_range(&params, &msg_vector, vec_start..vec_end);
    //     let partial_verifier =
    //         commit_params.verifier_for_vector_range(msg_vector.col_size(), vec_start..vec_end);

    //     let col_start = vec_start * secret_size * log_base_q;
    //     let col_end = vec_end * secret_size * log_base_q;
    //     let expected_opening = full_opening.slice_columns(col_start, col_end);
    //     let expected_verifier = full_verifier.slice_columns(col_start, col_end);

    //     assert_eq!(partial_opening, expected_opening);
    //     assert_eq!(partial_verifier, expected_verifier);
    //     assert!(commit_params.verify_vector_range(
    //         &params,
    //         &msg_vector,
    //         &commitment,
    //         &partial_opening,
    //         vec_start..vec_end
    //     ));
    // }
}
