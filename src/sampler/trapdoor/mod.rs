use crate::{
    matrix::{
        PolyMatrix,
        cpp_matrix::CppMatrix,
        dcrt_poly::DCRTPolyMatrix,
        i64::{I64Matrix, I64MatrixParams},
    },
    openfhe_guard::ensure_openfhe_warmup,
    parallel_iter,
    poly::{PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
use openfhe::ffi::{FormatMatrixCoefficient, SampleP1ForPertMat};
use rayon::iter::ParallelIterator;
pub use sampler::DCRTPolyTrapdoorSampler;
use std::{
    ops::Range,
    sync::{Mutex, OnceLock},
};
use tracing::debug;
use utils::{gen_dgg_int_vec, gen_int_karney, split_int64_mat_to_elems};

pub mod sampler;
pub mod utils;

pub(crate) const KARNEY_THRESHOLD: f64 = 300.0;
static SAMPLE_P1_FOR_PERT_MAT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DCRTTrapdoor {
    pub r: DCRTPolyMatrix,
    pub e: DCRTPolyMatrix,
}

impl DCRTTrapdoor {
    pub fn new(params: &DCRTPolyParams, size: usize, sigma: f64) -> Self {
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma };
        let r = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let e = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        Self { r, e }
    }

    pub fn sample_pert_square_mat(
        &self,
        s: f64,
        c: f64,
        dgg: f64,
        dgg_large_params: (Option<f64>, f64, Option<&[f64]>),
        peikert: bool,
        total_ncol: usize,
    ) -> DCRTPolyMatrix {
        let r = &self.r;
        let e = &self.e;
        let params = &r.params;
        ensure_openfhe_warmup(params);
        let n = params.ring_dimension() as usize;
        let (d, dk) = r.size();
        let sigma_large = dgg_large_params.1;
        let num_blocks = total_ncol.div_ceil(d);
        let padded_ncol = num_blocks * d;
        let padding_ncol = padded_ncol - total_ncol;
        debug!("{}", "sample_pert_square_mat parameters computed");
        // for distribution parameters up to the experimentally found threshold, use
        // the Peikert's inversion method otherwise, use Karney's method
        let p2z_vec = if sigma_large > KARNEY_THRESHOLD {
            let mut matrix = I64Matrix::new_empty(&I64MatrixParams, n * dk, padded_ncol);
            let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<i64>> {
                parallel_iter!(row_offsets)
                    .map(|_| {
                        parallel_iter!(col_offsets.clone())
                            .map(|_| gen_int_karney(0.0, sigma_large))
                            .collect()
                    })
                    .collect()
            };
            matrix.replace_entries(0..n * dk, 0..padded_ncol, f);
            matrix
        } else {
            let dgg_vectors = gen_dgg_int_vec(
                n * dk * padded_ncol,
                peikert,
                dgg_large_params.0.unwrap(),
                dgg_large_params.1,
                dgg_large_params.2.unwrap(),
            );
            let vecs = parallel_iter!(0..n * dk)
                .map(|i| {
                    dgg_vectors.slice(i * padded_ncol, (i + 1) * padded_ncol, 0, 1).transpose()
                })
                .collect::<Vec<_>>();
            vecs[0].concat_rows(&vecs[1..].iter().collect::<Vec<_>>())
        };
        debug!("{}", "p2z_vec generated");
        // create a matrix of d*k x padded_ncol ring elements in coefficient representation
        let p2 = split_int64_mat_to_elems(&p2z_vec, params);
        // parallel_iter!(0..padded_ncol)
        //     .map(|i| split_int64_vec_to_elems(&p2z_vec.slice(0, n * dk, i, i + 1), params))
        //     .collect::<Vec<_>>();
        // debug_mem("p2_vecs generated");
        // let p2 = p2_vecs[0].concat_columns(&p2_vecs[1..].iter().collect::<Vec<_>>());
        debug!("{}", "p2 generated");
        let a_mat = r.clone() * r.transpose(); // d x d
        let b_mat = r.clone() * e.transpose(); // d x d
        let d_mat = e.clone() * e.transpose(); // d x d
        debug!("{}", "a_mat, b_mat, d_mat generated");
        let re = r.concat_rows(&[e]);
        debug!("{}", "re generated");
        let tp2 = re * &p2;
        debug!("{}", "tp2 generated");
        let p1 = sample_p1_for_pert_mat(a_mat, b_mat, d_mat, tp2, params, c, s, dgg, padded_ncol);
        debug!("{}", "p1 generated");
        let mut p = p1.concat_rows(&[&p2]);
        debug!("{}", "p1 and p2 concatenated");
        if padding_ncol > 0 {
            p = p.slice_columns(0, total_ncol);
        }
        p
    }
}

// A function corresponding to lines 425-473 (except for the line 448) in the `SamplePertSquareMat` function in the trapdoor.h of OpenFHE. https://github.com/openfheorg/openfhe-development/blob/main/src/core/include/lattice/trapdoor.h#L425-L473
fn sample_p1_for_pert_mat(
    a_mat: DCRTPolyMatrix,
    b_mat: DCRTPolyMatrix,
    d_mat: DCRTPolyMatrix,
    tp2: DCRTPolyMatrix,
    params: &DCRTPolyParams,
    c: f64,
    s: f64,
    dgg_stddev: f64,
    padded_ncol: usize,
) -> DCRTPolyMatrix {
    ensure_openfhe_warmup(params);
    let n = params.ring_dimension();
    let depth = params.crt_depth();
    let k_res = params.crt_bits();
    debug!("{}", "sample_p1_for_pert_square_mat parameters computed");
    let mut a_mat = a_mat.to_cpp_matrix_ptr();
    FormatMatrixCoefficient(a_mat.inner.as_mut().unwrap());
    let mut b_mat = b_mat.to_cpp_matrix_ptr();
    FormatMatrixCoefficient(b_mat.inner.as_mut().unwrap());
    let mut d_mat = d_mat.to_cpp_matrix_ptr();
    FormatMatrixCoefficient(d_mat.inner.as_mut().unwrap());
    debug!("{}", "a_mat, b_mat, d_mat are converted to cpp matrices");
    let mut tp2_cpp = tp2.to_cpp_matrix_ptr();
    FormatMatrixCoefficient(tp2_cpp.inner.as_mut().unwrap());
    let cpp_matrix = {
        let _lock = SAMPLE_P1_FOR_PERT_MAT_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        SampleP1ForPertMat(
            &a_mat.inner,
            &b_mat.inner,
            &d_mat.inner,
            &tp2_cpp.inner,
            n,
            depth,
            k_res,
            padded_ncol,
            c,
            s,
            dgg_stddev,
        )
    };
    debug!("{}", "SampleP1ForPertSquareMat called");
    DCRTPolyMatrix::from_cpp_matrix_ptr(params, &CppMatrix::new(params, cpp_matrix))
}
