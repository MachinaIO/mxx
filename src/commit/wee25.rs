use crate::{
    matrix::PolyMatrix,
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
        let j_2m = {
            let l = tree_base * m_b;
            let lmn = l * secret_size * m_g;
            let g_lmn = M::gadget_matrix(params, lmn);
            let g_l = M::gadget_matrix(params, l);
            let identity_m = M::identity(params, m_g, None);
            let identity_m_cols = (0..m_g)
                .into_par_iter()
                .map(|i| identity_m.slice_columns(i, i + 1))
                .collect::<Vec<_>>();
            let identity_m_vec =
                identity_m_cols[0].concat_rows(&identity_m_cols[1..].iter().collect::<Vec<_>>());
            let mul1 = g_lmn.mul_tensor_identity(&identity_m_vec, l);
            let mul2 = mul1 * g_l;
            mul2.decompose()
        };

        Self { secret_size, tree_base, m_b, m_g, b, w, t_top, t_bottom, j_2m }
    }

    pub fn setup_vector<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
        tree_base: usize,
    ) -> Self {
        Self::setup::<US, TS>(params, secret_size, trapdoor_sigma, tree_base)
    }

    pub fn commit_matrix(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        debug_assert_eq!(msg.row_size(), self.secret_size);
        self.commit_matrix_recursive(params, msg)
    }

    pub fn commit_vector(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        let identity = M::identity(params, self.secret_size, None);
        let matrix = msg.tensor(&identity);
        self.commit_matrix(params, &matrix)
    }

    pub fn open_matrix(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        debug_assert_eq!(msg.row_size(), self.secret_size);
        let mut verifier_cache = std::collections::HashMap::new();
        let (opening, _commitment) = self.open_matrix_recursive(params, msg, &mut verifier_cache);
        opening
    }

    pub fn open_vector(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        let identity = M::identity(params, self.secret_size, None);
        let matrix = msg.tensor(&identity);
        self.open_matrix(params, &matrix)
    }

    pub fn verify_matrix(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        commit: &M,
        opening: &M,
    ) -> bool {
        debug_assert_eq!(msg.row_size(), self.secret_size);
        let msg_size = msg.col_size();
        let g_l = M::gadget_matrix(params, msg_size);
        let v = self.verifier_for_length(msg_size);
        let lhs = commit.clone() * v;
        let rhs_msg = msg.clone() * g_l;
        let rhs_open = self.b.clone() * opening.clone();
        let rhs = rhs_msg - rhs_open;
        lhs == rhs
    }

    pub fn verify_vector(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        commit: &M,
        opening: &M,
    ) -> bool {
        let identity = M::identity(params, self.secret_size, None);
        let matrix = msg.tensor(&identity);
        self.verify_matrix(params, &matrix, commit, opening)
    }

    fn commit_matrix_recursive(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        let cols = msg.col_size();
        debug_assert!(
            cols % self.m_b == 0,
            "commit_matrix expects column count divisible by m_b={} (got {})",
            self.m_b,
            cols
        );
        let base_cols = self.tree_base * self.m_b;
        if cols == base_cols {
            return self.commit_base(params, msg);
        }
        debug_assert!(
            cols % self.tree_base == 0,
            "commit_matrix expects column count divisible by tree_base={} (got {})",
            self.tree_base,
            cols
        );
        let part_cols = cols / self.tree_base;
        let mut commits = Vec::with_capacity(self.tree_base);
        for idx in 0..self.tree_base {
            let start = idx * part_cols;
            let end = start + part_cols;
            let part = msg.slice_columns(start, end);
            commits.push(self.commit_matrix_recursive(params, &part));
        }
        let commit_refs = commits.iter().collect::<Vec<_>>();
        let combined = commits[0].concat_columns(&commit_refs[1..]);
        self.commit_base(params, &combined)
    }

    fn commit_base(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, 2 * self.m_b),
            "base commit expects shape ({}, {})",
            self.secret_size,
            2 * self.m_b
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

    fn open_matrix_recursive(
        &self,
        params: &<M::P as Poly>::Params,
        msg: &M,
        verifier_cache: &mut std::collections::HashMap<usize, M>,
    ) -> (M, M) {
        let cols = msg.col_size();
        debug_assert!(
            cols % self.m_b == 0,
            "open_matrix expects column count divisible by m_b={} (got {})",
            self.m_b,
            cols
        );
        let base_cols = self.tree_base * self.m_b;
        if cols == base_cols {
            let opening = self.open_base(params, msg);
            let commitment = self.commit_base(params, msg);
            return (opening, commitment);
        }
        debug_assert!(
            cols % self.tree_base == 0,
            "open_matrix expects column count divisible by tree_base={} (got {})",
            self.tree_base,
            cols
        );
        let part_cols = cols / self.tree_base;
        let mut zs = Vec::with_capacity(self.tree_base);
        let mut cs = Vec::with_capacity(self.tree_base);
        for idx in 0..self.tree_base {
            let start = idx * part_cols;
            let end = start + part_cols;
            let part = msg.slice_columns(start, end);
            let (z, c) = self.open_matrix_recursive(params, &part, verifier_cache);
            zs.push(z);
            cs.push(c);
        }
        let c_refs = cs.iter().collect::<Vec<_>>();
        let combined_c = cs[0].concat_columns(&c_refs[1..]);
        let z_prime = self.open_base(params, &combined_c);
        let v_part = match verifier_cache.get(&part_cols) {
            Some(v) => v.clone(),
            None => {
                let v = self.verifier_for_length(part_cols);
                verifier_cache.insert(part_cols, v.clone());
                v
            }
        };
        let adjusted = z_prime.mul_tensor_identity_decompose(&v_part, self.tree_base);
        let z_refs = zs.iter().collect::<Vec<_>>();
        let z_concat = zs[0].concat_columns(&z_refs[1..]);
        let opening = adjusted + z_concat;
        let commitment = self.commit_base(params, &combined_c);
        (opening, commitment)
    }

    fn open_base(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, 2 * self.m_b),
            "base open expects shape ({}, {})",
            self.secret_size,
            2 * self.m_b
        );
        let cols = msg.col_size();
        let acc = (0..cols)
            .into_par_iter()
            .fold(
                || M::zero(params, self.m_b, self.t_top.col_size()),
                |mut acc, j| {
                    let decomposed_col = msg.get_column_matrix_decompose(j);
                    for r in 0..self.m_g {
                        let a = decomposed_col.entry(r, 0);
                        let row_start = (j * self.m_g + r) * self.m_b;
                        let row_end = row_start + self.m_b;
                        let t_block = self.t_top.slice_rows(row_start, row_end);
                        acc = acc + (t_block * &a);
                    }
                    acc
                },
            )
            .reduce(
                || M::zero(params, self.m_b, self.t_top.col_size()),
                |left, right| left + right,
            );
        acc * &self.j_2m
    }

    pub fn verifier_for_length(&self, cols: usize) -> M {
        let base_len = self.tree_base * self.m_b;
        debug_assert!(
            cols % base_len == 0,
            "verifier_for_length expects multiple of {} cols (got {})",
            base_len,
            cols
        );
        let base = self.verifier_base();
        if cols == base_len {
            return base;
        }
        let mut current = base.clone();
        let mut current_len = base_len;
        while current_len < cols {
            current = base.clone().mul_tensor_identity_decompose(&current, self.tree_base);
            current_len *= self.tree_base;
        }
        current
    }

    fn verifier_base(&self) -> M {
        self.t_bottom.clone() * &self.j_2m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        matrix::{MatrixElem, dcrt_poly::DCRTPolyMatrix},
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        sampler::{trapdoor::sampler::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler},
    };
    use std::time::Instant;
    use tracing::info;
    const SIGMA: f64 = 4.578;

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let base_len = tree_base * m_b;
        let msg_size = base_len * 2;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let zero_matrix = DCRTPolyMatrix::zero(&params, secret_size, msg_size);
        let start = Instant::now();
        let commitment = commit_params.commit_matrix(&params, &zero_matrix);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier_for_length(msg_size);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open_matrix(&params, &zero_matrix);
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify_matrix(&params, &zero_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_invalid_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let base_len = tree_base * m_b;
        let msg_size = base_len * 2;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let mut zero_matrix = DCRTPolyMatrix::zero(&params, secret_size, msg_size);
        let start = Instant::now();
        let commitment = commit_params.commit_matrix(&params, &zero_matrix);
        info!("commitment generated in {:?}", start.elapsed());
        zero_matrix.set_entry(0, 0, DCRTPoly::one(&params));
        let start = Instant::now();
        let _verifier = commit_params.verifier_for_length(msg_size);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open_matrix(&params, &zero_matrix);
        info!("opening generated in {:?}", start.elapsed());

        assert!(!commit_params.verify_matrix(&params, &zero_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let base_len = tree_base * m_b;
        let msg_size = base_len * 2;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_matrix =
            uniform_sampler.sample_uniform(&params, secret_size, msg_size, DistType::FinRingDist);

        let start = Instant::now();
        let commitment = commit_params.commit_matrix(&params, &msg_matrix);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier_for_length(msg_size);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open_matrix(&params, &msg_matrix);
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify_matrix(&params, &msg_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_invalid_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let base_len = tree_base * m_b;
        let msg_size = base_len * 2;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_matrix =
            uniform_sampler.sample_uniform(&params, secret_size, msg_size, DistType::FinRingDist);
        let start = Instant::now();
        let commitment = commit_params.commit_matrix(&params, &msg_matrix);
        info!("commitment generated in {:?}", start.elapsed());

        let mut tampered = msg_matrix.clone();
        let original_entry = tampered.entry(0, 0);
        tampered.set_entry(0, 0, original_entry + DCRTPoly::const_one(&params));

        let start = Instant::now();
        let _verifier = commit_params.verifier_for_length(msg_size);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open_matrix(&params, &tampered);
        info!("opening generated in {:?}", start.elapsed());

        assert!(!commit_params.verify_matrix(&params, &tampered, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_vector_commit_verify() {
        let _ = tracing_subscriber::fmt::try_init();
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let tree_base = 2;
        let base_len = tree_base * m_b;
        let msg_size = 2 * base_len;

        let start = Instant::now();
        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA, tree_base);
        info!("commit params generated in {:?}", start.elapsed());

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_vector =
            uniform_sampler.sample_uniform(&params, 1, msg_size / 2, DistType::FinRingDist);

        let start = Instant::now();
        let commitment = commit_params.commit_vector(&params, &msg_vector);
        info!("commitment generated in {:?}", start.elapsed());
        let start = Instant::now();
        let _verifier = commit_params.verifier_for_length(msg_size);
        info!("verifier generated in {:?}", start.elapsed());
        let start = Instant::now();
        let opening = commit_params.open_vector(&params, &msg_vector);
        info!("opening generated in {:?}", start.elapsed());

        assert!(commit_params.verify_vector(&params, &msg_vector, &commitment, &opening));
    }
}
