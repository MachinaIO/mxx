use crate::{
    matrix::{
        CompactMatrixDecodeError, PolyMatrix,
        cpp_matrix::CppMatrix,
        dcrt_poly::DCRTPolyMatrix,
        i64::{I64Matrix, I64MatrixParams},
    },
    openfhe_guard::ensure_openfhe_warmup,
    parallel_iter,
    poly::{
        PolyParams,
        dcrt::{native::ffi::exact_basis_matrix_coefficients, params::DCRTPolyParams},
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};
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

#[derive(Debug, Clone)]
pub struct DCRTTrapdoor {
    pub r: DCRTPolyMatrix,
    pub e: DCRTPolyMatrix,
    pub a_mat: DCRTPolyMatrix,
    pub b_mat: DCRTPolyMatrix,
    pub d_mat: DCRTPolyMatrix,
    pub re: DCRTPolyMatrix,
}

impl PartialEq for DCRTTrapdoor {
    fn eq(&self, other: &Self) -> bool {
        self.r_cpu() == other.r_cpu() &&
            self.e_cpu() == other.e_cpu() &&
            self.a_mat_cpu() == other.a_mat_cpu() &&
            self.b_mat_cpu() == other.b_mat_cpu() &&
            self.d_mat_cpu() == other.d_mat_cpu() &&
            self.re_cpu() == other.re_cpu()
    }
}

impl Eq for DCRTTrapdoor {}

impl DCRTTrapdoor {
    pub fn new(params: &DCRTPolyParams, size: usize, sigma: f64) -> Self {
        assert_eq!(
            params.dropped_moduli(),
            0,
            "exact trapdoor sampling requires dropped_moduli = 0"
        );
        let uniform_sampler = DCRTPolyUniformSampler::new();
        let log_base_q = params.modulus_digits();
        let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
        let r_cpu = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist.clone());
        let e_cpu = uniform_sampler.sample_uniform(params, size, size * log_base_q, dist);
        let a_mat_cpu = &r_cpu * &r_cpu.transpose(); // d x d
        let b_mat_cpu = &r_cpu * &e_cpu.transpose(); // d x d
        let d_mat_cpu = &e_cpu * &e_cpu.transpose(); // d x d
        let re_cpu = r_cpu.concat_rows(&[&e_cpu]);
        Self {
            r: r_cpu,
            e: e_cpu,
            a_mat: a_mat_cpu,
            b_mat: b_mat_cpu,
            d_mat: d_mat_cpu,
            re: re_cpu,
        }
    }

    pub fn r_cpu(&self) -> DCRTPolyMatrix {
        self.r.clone()
    }

    pub fn e_cpu(&self) -> DCRTPolyMatrix {
        self.e.clone()
    }

    pub(crate) fn a_mat_cpu(&self) -> DCRTPolyMatrix {
        self.a_mat.clone()
    }

    pub(crate) fn b_mat_cpu(&self) -> DCRTPolyMatrix {
        self.b_mat.clone()
    }

    pub(crate) fn d_mat_cpu(&self) -> DCRTPolyMatrix {
        self.d_mat.clone()
    }

    pub(crate) fn re_cpu(&self) -> DCRTPolyMatrix {
        self.re.clone()
    }

    /// Validate all six CPU components against one concrete ordered ring.
    pub fn matches_ordered_ring(&self, moduli: &[u64], ring_dimension: u32) -> bool {
        [&self.r, &self.e, &self.a_mat, &self.b_mat, &self.d_mat, &self.re].into_iter().all(
            |matrix| {
                matrix.params().ring_dimension() == ring_dimension &&
                    matrix.params().moduli() == moduli
            },
        )
    }

    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mats = [
            self.r_cpu(),
            self.e_cpu(),
            self.a_mat_cpu(),
            self.b_mat_cpu(),
            self.d_mat_cpu(),
            self.re_cpu(),
        ];
        let mut parts = Vec::with_capacity(mats.len());
        let mut total_len = 0usize;
        for mat in mats.iter() {
            let bytes = mat.to_compact_bytes();
            total_len += 8 + bytes.len();
            parts.push(bytes);
        }
        let mut out = Vec::with_capacity(total_len);
        for bytes in parts {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_compact_bytes(params: &DCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        Self::try_from_compact_bytes(params, bytes).ok()
    }

    pub fn try_from_compact_bytes(
        params: &DCRTPolyParams,
        bytes: &[u8],
    ) -> Result<Self, CompactMatrixDecodeError> {
        let mut offset = 0usize;
        let next = |buf: &[u8], offset: &mut usize| -> Result<Vec<u8>, CompactMatrixDecodeError> {
            let header_end = (*offset).checked_add(8).ok_or(
                CompactMatrixDecodeError::InvalidHeader("trapdoor compact length overflows"),
            )?;
            if header_end > buf.len() {
                return Err(CompactMatrixDecodeError::InvalidHeader(
                    "truncated trapdoor compact length",
                ));
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&buf[*offset..header_end]);
            let len = u64::from_le_bytes(len_bytes) as usize;
            *offset = header_end;
            let payload_end =
                (*offset).checked_add(len).ok_or(CompactMatrixDecodeError::InvalidHeader(
                    "trapdoor compact payload length overflows",
                ))?;
            if payload_end > buf.len() {
                return Err(CompactMatrixDecodeError::InvalidHeader(
                    "truncated trapdoor compact payload",
                ));
            }
            let out = buf[*offset..payload_end].to_vec();
            *offset = payload_end;
            Ok(out)
        };
        let r_bytes = next(bytes, &mut offset)?;
        let e_bytes = next(bytes, &mut offset)?;
        let a_bytes = next(bytes, &mut offset)?;
        let b_bytes = next(bytes, &mut offset)?;
        let d_bytes = next(bytes, &mut offset)?;
        let re_bytes = next(bytes, &mut offset)?;
        if offset != bytes.len() {
            return Err(CompactMatrixDecodeError::InvalidHeader("trailing trapdoor compact bytes"));
        }
        let r = DCRTPolyMatrix::try_from_compact_bytes(params, &r_bytes)?;
        let e = DCRTPolyMatrix::try_from_compact_bytes(params, &e_bytes)?;
        let a_mat = DCRTPolyMatrix::try_from_compact_bytes(params, &a_bytes)?;
        let b_mat = DCRTPolyMatrix::try_from_compact_bytes(params, &b_bytes)?;
        let d_mat = DCRTPolyMatrix::try_from_compact_bytes(params, &d_bytes)?;
        let re = DCRTPolyMatrix::try_from_compact_bytes(params, &re_bytes)?;
        Ok(Self { r, e, a_mat, b_mat, d_mat, re })
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
        let r = self.r_cpu();
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
        drop(p2z_vec);
        // parallel_iter!(0..padded_ncol)
        //     .map(|i| split_int64_vec_to_elems(&p2z_vec.slice(0, n * dk, i, i + 1), params))
        //     .collect::<Vec<_>>();
        // debug_mem("p2_vecs generated");
        // let p2 = p2_vecs[0].concat_columns(&p2_vecs[1..].iter().collect::<Vec<_>>());
        debug!("{}", "p2 generated");
        let a_mat = self.a_mat_cpu();
        let b_mat = self.b_mat_cpu();
        let d_mat = self.d_mat_cpu();
        debug!("{}", "a_mat, b_mat, d_mat loaded");
        let re = self.re_cpu();
        debug!("{}", "re loaded");
        let tp2 = &re * &p2;
        debug!("{}", "tp2 generated");
        let p1 = sample_p1_for_pert_mat(a_mat, b_mat, d_mat, tp2, params, c, s, dgg, padded_ncol);
        debug!("{}", "p1 generated");
        let mut p = p1.concat_rows(&[&p2]);
        drop(p1);
        drop(p2);
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
    debug!("{}", "sample_p1_for_pert_square_mat parameters computed");
    let mut a_mat = a_mat.to_cpp_matrix_ptr();
    exact_basis_matrix_coefficients(a_mat.inner.as_mut().unwrap())
        .expect("A coefficient conversion");
    let mut b_mat = b_mat.to_cpp_matrix_ptr();
    exact_basis_matrix_coefficients(b_mat.inner.as_mut().unwrap())
        .expect("B coefficient conversion");
    let mut d_mat = d_mat.to_cpp_matrix_ptr();
    exact_basis_matrix_coefficients(d_mat.inner.as_mut().unwrap())
        .expect("D coefficient conversion");
    debug!("{}", "a_mat, b_mat, d_mat are converted to cpp matrices");
    let mut tp2_cpp = tp2.to_cpp_matrix_ptr();
    exact_basis_matrix_coefficients(tp2_cpp.inner.as_mut().unwrap())
        .expect("target coefficient conversion");
    drop(tp2);
    let cpp_matrix = {
        let _lock = SAMPLE_P1_FOR_PERT_MAT_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        crate::poly::dcrt::native::ffi::exact_basis_p1(
            &a_mat.inner,
            &b_mat.inner,
            &d_mat.inner,
            &tp2_cpp.inner,
            padded_ncol,
            c,
            s,
            dgg_stddev,
        )
        .expect("exact CRT perturbation sampling failed")
    };
    debug!("{}", "SampleP1ForPertSquareMat called");
    drop(a_mat);
    drop(b_mat);
    drop(d_mat);
    drop(tp2_cpp);
    DCRTPolyMatrix::from_cpp_matrix_ptr(params, &CppMatrix::new(cpp_matrix))
}
