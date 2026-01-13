use crate::{
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    sampler::{DistType, PolyTrapdoorSampler, PolyUniformSampler},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone)]
pub struct Wee25Commit<M: PolyMatrix> {
    pub secret_size: usize,
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
    ) -> Self {
        let trapdoor_sampler = TS::new(params, trapdoor_sigma);
        let (trapdoor, b) = trapdoor_sampler.trapdoor(params, secret_size);
        let uniform_sampler = US::new();
        let m_b = b.col_size();
        let log_base_q = params.modulus_digits();
        let m_g = secret_size * log_base_q;
        let pp_size = 2 * m_b * m_g;
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
            let l = 2 * m_b;
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

        Self { secret_size, m_b, m_g, b, w, t_top, t_bottom, j_2m }
    }

    pub fn setup_vector<US: PolyUniformSampler<M = M>, TS: PolyTrapdoorSampler<M = M>>(
        params: &<M::P as Poly>::Params,
        secret_size: usize,
        trapdoor_sigma: f64,
    ) -> Self {
        Self::setup::<US, TS>(params, secret_size, trapdoor_sigma)
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
        self.open_matrix_recursive(params, msg)
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
        if cols == 2 * self.m_b {
            return self.commit_base(params, msg);
        }
        debug_assert!(
            cols % 2 == 0,
            "commit_matrix expects even number of column blocks, got {}",
            cols
        );
        let mid = cols / 2;
        let left = msg.slice_columns(0, mid);
        let right = msg.slice_columns(mid, cols);
        let c0 = self.commit_matrix_recursive(params, &left);
        let c1 = self.commit_matrix_recursive(params, &right);
        let combined = c0.concat_columns(&[&c1]);
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
        let bits_row = self.bits_row(msg);
        let identity = M::identity(&params, self.secret_size, None);
        let bits_tensor = bits_row.tensor(&identity);
        bits_tensor * &self.w
    }

    fn open_matrix_recursive(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        let cols = msg.col_size();
        debug_assert!(
            cols % self.m_b == 0,
            "open_matrix expects column count divisible by m_b={} (got {})",
            self.m_b,
            cols
        );
        if cols == 2 * self.m_b {
            return self.open_base(params, msg);
        }
        debug_assert!(
            cols % 2 == 0,
            "open_matrix expects even number of column blocks, got {}",
            cols
        );
        let mid = cols / 2;
        let left = msg.slice_columns(0, mid);
        let right = msg.slice_columns(mid, cols);
        let z0 = self.open_matrix_recursive(params, &left);
        let z1 = self.open_matrix_recursive(params, &right);
        let c0 = self.commit_matrix_recursive(params, &left);
        let c1 = self.commit_matrix_recursive(params, &right);
        let combined_c = c0.concat_columns(&[&c1]);
        let z_prime = self.open_base(params, &combined_c);
        let v_half = self.verifier_for_length(mid);
        let adjusted = z_prime.mul_tensor_identity_decompose(&v_half, 2);
        let z_concat = z0.concat_columns(&[&z1]);
        adjusted + z_concat
    }

    fn open_base(&self, params: &<M::P as Poly>::Params, msg: &M) -> M {
        debug_assert_eq!(
            msg.size(),
            (self.secret_size, 2 * self.m_b),
            "base open expects shape ({}, {})",
            self.secret_size,
            2 * self.m_b
        );
        let bits_row = self.bits_row(msg);
        let m_b = self.b.col_size();
        let identity = M::identity(params, m_b, None);
        let bits_tensor = bits_row.tensor(&identity);
        // let t_matrix = self.t_top.clone().concat_rows(&[&self.t_bottom]);
        let intermediate = bits_tensor * &self.t_top;
        intermediate * &self.j_2m
    }

    fn bits_row(&self, msg: &M) -> M {
        let decomposed = msg.decompose();
        debug_assert_eq!(
            decomposed.row_size(),
            self.m_g,
            "decomposed row size {} must equal m_g={}",
            decomposed.row_size(),
            self.m_g
        );
        let cols = decomposed.col_size();
        debug_assert!(cols > 0, "bits_row expects at least one column in the decomposed matrix");
        let mut transposed_cols = Vec::with_capacity(cols);
        for j in 0..cols {
            let col = decomposed.slice_columns(j, j + 1);
            transposed_cols.push(col.transpose());
        }
        let mut iter = transposed_cols.into_iter();
        let first = iter.next().expect("transposed_cols not empty");
        let rest: Vec<M> = iter.collect();
        if rest.is_empty() {
            first
        } else {
            let rest_refs = rest.iter().collect::<Vec<_>>();
            first.concat_columns(&rest_refs)
        }
    }

    pub fn verifier_for_length(&self, cols: usize) -> M {
        let base_len = 2 * self.m_b;
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
            current = base.clone().mul_tensor_identity_decompose(&current, 2);
            current_len *= 2;
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
    const SIGMA: f64 = 4.578;

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_verify() {
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let base_len = 2 * m_b;
        let msg_size = base_len * 2;

        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA);

        let zero_matrix = DCRTPolyMatrix::zero(&params, secret_size, msg_size);
        let commitment = commit_params.commit_matrix(&params, &zero_matrix);
        let opening = commit_params.open_matrix(&params, &zero_matrix);

        assert!(commit_params.verify_matrix(&params, &zero_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_zero_commit_invalid_verify() {
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let base_len = 2 * m_b;
        let msg_size = base_len * 2;

        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA);

        let mut zero_matrix = DCRTPolyMatrix::zero(&params, secret_size, msg_size);
        let commitment = commit_params.commit_matrix(&params, &zero_matrix);
        zero_matrix.set_entry(0, 0, DCRTPoly::one(&params));
        let opening = commit_params.open_matrix(&params, &zero_matrix);

        assert!(!commit_params.verify_matrix(&params, &zero_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_verify() {
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let base_len = 2 * m_b;
        let msg_size = base_len * 2;

        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA);

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_matrix =
            uniform_sampler.sample_uniform(&params, secret_size, msg_size, DistType::FinRingDist);

        let commitment = commit_params.commit_matrix(&params, &msg_matrix);
        let opening = commit_params.open_matrix(&params, &msg_matrix);

        assert!(commit_params.verify_matrix(&params, &msg_matrix, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_commit_invalid_verify() {
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 1;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let base_len = 2 * m_b;
        let msg_size = base_len * 2;

        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA);

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_matrix =
            uniform_sampler.sample_uniform(&params, secret_size, msg_size, DistType::FinRingDist);
        let commitment = commit_params.commit_matrix(&params, &msg_matrix);

        let mut tampered = msg_matrix.clone();
        let original_entry = tampered.entry(0, 0);
        tampered.set_entry(0, 0, original_entry + DCRTPoly::const_one(&params));

        let opening = commit_params.open_matrix(&params, &tampered);

        assert!(!commit_params.verify_matrix(&params, &tampered, &commitment, &opening));
    }

    #[test]
    #[sequential_test::sequential]
    fn test_wee25_random_vector_commit_verify() {
        let params = DCRTPolyParams::new(4, 2, 17, 15);
        let secret_size = 2;
        let m_b = (&params.modulus_digits() + 2) * secret_size;
        let base_len = 2 * m_b;
        let msg_size = 2 * base_len;

        let commit_params = Wee25Commit::<DCRTPolyMatrix>::setup::<
            DCRTPolyUniformSampler,
            DCRTPolyTrapdoorSampler,
        >(&params, secret_size, SIGMA);

        let uniform_sampler = DCRTPolyUniformSampler::new();
        let msg_vector =
            uniform_sampler.sample_uniform(&params, 1, msg_size / 2, DistType::FinRingDist);

        let commitment = commit_params.commit_vector(&params, &msg_vector);
        let opening = commit_params.open_vector(&params, &msg_vector);

        assert!(commit_params.verify_vector(&params, &msg_vector, &commitment, &opening));
    }
}
