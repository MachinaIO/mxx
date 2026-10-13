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
use mxx_ir_core::types::TrapdoorInstantiation;
use rayon::iter::ParallelIterator;
pub use sampler::DCRTPolyTrapdoorSampler;
use std::{
    ops::Range,
    sync::{Mutex, OnceLock},
};
use tracing::debug;
use utils::{gen_dgg_int_vec, gen_int_karney, split_int64_mat_to_elems};

pub mod sampler;
pub mod statistical;
pub mod utils;

pub(crate) const KARNEY_THRESHOLD: f64 = 300.0;
static SAMPLE_P1_FOR_PERT_MAT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// A sampled gadget trapdoor, in one of the two instantiations of
/// [`TrapdoorInstantiation`].
///
/// - `Computational`: `A = [ā | I | G - (ā R + E)]` with the trapdoor `[R; E; I]`. `a_mat`,
///   `b_mat`, and `d_mat` are `R Rᵀ`, `R Eᵀ`, and `E Eᵀ`, and `re` is `[R; E]`, as the OpenFHE
///   perturbation sampler consumes them (with the plain, not the ring-adjoint, transpose).
/// - `Statistical`: `A = [ā | g - ā T]` with the trapdoor `[T; I]` for `T` of `m_bar` rows and `D`
///   columns. `T` is kept as row chunks of at most [`statistical::ROW_CHUNK`] rows, and every
///   operation streams over them, so no `m_bar x m_bar` or `D x D` matrix is formed.
// One value per trapdoor; boxing the computational components would only add an indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum DCRTTrapdoor {
    Computational {
        r: DCRTPolyMatrix,
        e: DCRTPolyMatrix,
        a_mat: DCRTPolyMatrix,
        b_mat: DCRTPolyMatrix,
        d_mat: DCRTPolyMatrix,
        re: DCRTPolyMatrix,
    },
    Statistical {
        /// Row chunks of `T`, in row order.
        t: Vec<DCRTPolyMatrix>,
    },
}

impl PartialEq for DCRTTrapdoor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Computational { .. }, Self::Computational { .. }) => {
                self.r_cpu() == other.r_cpu() &&
                    self.e_cpu() == other.e_cpu() &&
                    self.a_mat_cpu() == other.a_mat_cpu() &&
                    self.b_mat_cpu() == other.b_mat_cpu() &&
                    self.d_mat_cpu() == other.d_mat_cpu() &&
                    self.re_cpu() == other.re_cpu()
            }
            (Self::Statistical { t }, Self::Statistical { t: other }) => {
                statistical::rows_equal(t, other)
            }
            _ => false,
        }
    }
}

impl Eq for DCRTTrapdoor {}

/// Leading bytes of a tagged trapdoor encoding. Read as the little-endian length prefix of the
/// untagged computational encoding, they exceed every encodable length, so the two are
/// distinguished without ambiguity.
pub const TAGGED_MAGIC: [u8; 8] = *b"mxxtrpd\xff";
/// The tag of a computational trapdoor: the components `R`, `E`, `R Rᵀ`, `R Eᵀ`, `E Eᵀ`, `[R; E]`.
pub const TAG_COMPUTATIONAL: u8 = 0;
/// The tag of a statistical trapdoor: the row chunks of `T`.
pub const TAG_STATISTICAL: u8 = 1;

/// The `(rows, columns)` of every component of a trapdoor of `rows` public rows and `digits`
/// digits per row, in encoding order: `R`, `E`, `R Rᵀ`, `R Eᵀ`, `E Eᵀ`, `[R; E]` for the
/// computational instantiation, the row chunks of `T` for the statistical one. `None` for an
/// invalid instantiation.
pub fn trapdoor_component_shapes(
    rows: usize,
    digits: usize,
    instantiation: TrapdoorInstantiation,
) -> Option<Vec<(usize, usize)>> {
    match instantiation {
        TrapdoorInstantiation::Computational => {
            let width = rows.checked_mul(digits)?;
            let square = (rows, rows);
            Some(vec![(rows, width), (rows, width), square, square, square, (2 * rows, width)])
        }
        TrapdoorInstantiation::Statistical { m_bar } if rows == 1 && m_bar > 0 => Some(
            statistical::chunk_row_counts(m_bar).into_iter().map(|rows| (rows, digits)).collect(),
        ),
        TrapdoorInstantiation::Statistical { .. } => None,
    }
}

/// The tag and the length-prefixed component payloads of a trapdoor encoding, tagged or untagged
/// (an untagged encoding is computational).
pub fn trapdoor_encoding_parts(bytes: &[u8]) -> Result<(u8, Vec<&[u8]>), CompactMatrixDecodeError> {
    let (tag, mut offset) = if bytes.starts_with(&TAGGED_MAGIC) {
        let tag = *bytes
            .get(TAGGED_MAGIC.len())
            .ok_or(CompactMatrixDecodeError::InvalidHeader("truncated trapdoor encoding tag"))?;
        (tag, TAGGED_MAGIC.len() + 1)
    } else {
        (TAG_COMPUTATIONAL, 0)
    };
    let mut parts = Vec::new();
    while offset < bytes.len() {
        let header_end = offset
            .checked_add(8)
            .ok_or(CompactMatrixDecodeError::InvalidHeader("trapdoor compact length overflows"))?;
        if header_end > bytes.len() {
            return Err(CompactMatrixDecodeError::InvalidHeader(
                "truncated trapdoor compact length",
            ));
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&bytes[offset..header_end]);
        let len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| {
            CompactMatrixDecodeError::InvalidHeader("trapdoor compact payload length overflows")
        })?;
        let payload_end = header_end.checked_add(len).ok_or(
            CompactMatrixDecodeError::InvalidHeader("trapdoor compact payload length overflows"),
        )?;
        if payload_end > bytes.len() {
            return Err(CompactMatrixDecodeError::InvalidHeader(
                "truncated trapdoor compact payload",
            ));
        }
        parts.push(&bytes[header_end..payload_end]);
        offset = payload_end;
    }
    if tag == TAG_COMPUTATIONAL && parts.len() != 6 {
        return Err(CompactMatrixDecodeError::InvalidHeader(
            "a computational trapdoor has six components",
        ));
    }
    if tag == TAG_STATISTICAL && parts.is_empty() {
        return Err(CompactMatrixDecodeError::InvalidHeader("a statistical trapdoor has rows"));
    }
    if tag > TAG_STATISTICAL {
        return Err(CompactMatrixDecodeError::InvalidHeader("unknown trapdoor encoding tag"));
    }
    Ok((tag, parts))
}

impl DCRTTrapdoor {
    /// Samples a computational trapdoor of `size` rows: `R` and `E` Gaussian of width `sigma`
    /// without truncation.
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
        Self::Computational {
            r: r_cpu,
            e: e_cpu,
            a_mat: a_mat_cpu,
            b_mat: b_mat_cpu,
            d_mat: d_mat_cpu,
            re: re_cpu,
        }
    }

    /// Statistical TrapGen: `ā` uniform `1 x m_bar`, `T` Gaussian `m_bar x D` of width `sigma`
    /// without truncation, and `A = [ā | g - ā T]`, with `ā T` accumulated over the row chunks of
    /// `T`. Panics unless `size` is one and `instantiation` is statistical.
    pub fn sample_statistical(
        params: &DCRTPolyParams,
        size: usize,
        sigma: f64,
        instantiation: TrapdoorInstantiation,
    ) -> (Self, DCRTPolyMatrix) {
        let TrapdoorInstantiation::Statistical { m_bar } = instantiation else {
            panic!("statistical TrapGen requires the statistical instantiation");
        };
        assert_eq!(size, 1, "the statistical trapdoor has one public row");
        assert_eq!(
            params.dropped_moduli(),
            0,
            "exact trapdoor sampling requires dropped_moduli = 0"
        );
        let (t, public) = statistical::trapgen(params, m_bar, sigma);
        (Self::Statistical { t }, public)
    }

    /// The instantiation of this trapdoor.
    pub fn instantiation(&self) -> TrapdoorInstantiation {
        match self {
            Self::Computational { .. } => TrapdoorInstantiation::Computational,
            Self::Statistical { t } => TrapdoorInstantiation::Statistical {
                m_bar: t.iter().map(|chunk| chunk.row_size()).sum(),
            },
        }
    }

    fn computational_part(&self, part: usize) -> &DCRTPolyMatrix {
        let Self::Computational { r, e, a_mat, b_mat, d_mat, re } = self else {
            panic!("computational component requested from a statistical trapdoor");
        };
        [r, e, a_mat, b_mat, d_mat, re][part]
    }

    pub fn r_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(0).clone()
    }

    pub fn e_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(1).clone()
    }

    pub(crate) fn a_mat_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(2).clone()
    }

    pub(crate) fn b_mat_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(3).clone()
    }

    pub(crate) fn d_mat_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(4).clone()
    }

    pub(crate) fn re_cpu(&self) -> DCRTPolyMatrix {
        self.computational_part(5).clone()
    }

    /// The row chunks of `T` of a statistical trapdoor.
    pub fn t_chunks(&self) -> &[DCRTPolyMatrix] {
        let Self::Statistical { t } = self else {
            panic!("statistical component requested from a computational trapdoor");
        };
        t
    }

    /// Every stored matrix: the six computational components, or the row chunks of `T`.
    pub fn matrices(&self) -> Vec<&DCRTPolyMatrix> {
        match self {
            Self::Computational { r, e, a_mat, b_mat, d_mat, re } => {
                vec![r, e, a_mat, b_mat, d_mat, re]
            }
            Self::Statistical { t } => t.iter().collect(),
        }
    }

    /// Validate all CPU components against one concrete ordered ring.
    pub fn matches_ordered_ring(&self, moduli: &[u64], ring_dimension: u32) -> bool {
        self.matrices().into_iter().all(|matrix| {
            matrix.params().ring_dimension() == ring_dimension && matrix.params().moduli() == moduli
        })
    }

    /// The tagged encoding: `TAGGED_MAGIC`, a tag byte, and the length-prefixed compact matrices
    /// (the six computational components, or the row chunks of `T`). Each chunk is a separate
    /// record, so the encoding can be written and read one chunk at a time.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let tag = match self {
            Self::Computational { .. } => TAG_COMPUTATIONAL,
            Self::Statistical { .. } => TAG_STATISTICAL,
        };
        let parts = parallel_iter!(self.matrices())
            .map(|matrix| matrix.to_compact_bytes())
            .collect::<Vec<_>>();
        let mut out = Vec::with_capacity(
            TAGGED_MAGIC.len() + 1 + parts.iter().map(|bytes| 8 + bytes.len()).sum::<usize>(),
        );
        out.extend_from_slice(&TAGGED_MAGIC);
        out.push(tag);
        for bytes in parts {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn from_compact_bytes(params: &DCRTPolyParams, bytes: &[u8]) -> Option<Self> {
        Self::try_from_compact_bytes(params, bytes).ok()
    }

    /// Decodes [`Self::to_compact_bytes`], or the untagged six-part encoding of a computational
    /// trapdoor written before the tag was introduced.
    pub fn try_from_compact_bytes(
        params: &DCRTPolyParams,
        bytes: &[u8],
    ) -> Result<Self, CompactMatrixDecodeError> {
        let (tag, parts) = trapdoor_encoding_parts(bytes)?;
        let matrices = parallel_iter!(parts)
            .map(|part| DCRTPolyMatrix::try_from_compact_bytes(params, part))
            .collect::<Result<Vec<_>, _>>()?;
        match tag {
            TAG_COMPUTATIONAL => {
                let [r, e, a_mat, b_mat, d_mat, re]: [DCRTPolyMatrix; 6] =
                    matrices.try_into().map_err(|_| {
                        CompactMatrixDecodeError::InvalidHeader(
                            "a computational trapdoor has six components",
                        )
                    })?;
                Ok(Self::Computational { r, e, a_mat, b_mat, d_mat, re })
            }
            TAG_STATISTICAL => {
                let digits = params.modulus_digits();
                if matrices.is_empty() ||
                    matrices
                        .iter()
                        .any(|chunk| chunk.col_size() != digits || chunk.row_size() == 0)
                {
                    return Err(CompactMatrixDecodeError::InvalidHeader(
                        "statistical trapdoor chunks must be nonempty with one column per digit",
                    ));
                }
                Ok(Self::Statistical { t: matrices })
            }
            _ => Err(CompactMatrixDecodeError::InvalidHeader("unknown trapdoor encoding tag")),
        }
    }

    /// A perturbation `p` of width matrix `s² I - c² [T'; I][T'; I]*` for the top block `T'` of
    /// the trapdoor, one column per target column. See [`statistical::perturbation`] for the
    /// statistical instantiation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_perturbation(
        &self,
        params: &DCRTPolyParams,
        s: f64,
        c: f64,
        dgg: f64,
        dgg_large_params: (Option<f64>, f64, Option<&[f64]>),
        peikert: bool,
        total_ncol: usize,
    ) -> DCRTPolyMatrix {
        match self {
            Self::Computational { .. } => {
                self.sample_pert_square_mat(s, c, dgg, dgg_large_params, peikert, total_ncol)
            }
            Self::Statistical { t } => {
                let p2z_vec = sample_p2_integers(
                    params.ring_dimension() as usize * params.modulus_digits(),
                    total_ncol,
                    dgg_large_params,
                    peikert,
                );
                statistical::perturbation(params, t, p2z_vec, s, c, dgg)
            }
        }
    }

    /// The preimage `[p_top + T' z; p_bottom + z]` of a perturbation `p` and a gadget preimage
    /// `z` (`D·rows x columns`).
    pub(crate) fn combine_preimage(
        &self,
        p_hat: DCRTPolyMatrix,
        z_hat: &DCRTPolyMatrix,
    ) -> DCRTPolyMatrix {
        match self {
            Self::Computational { r, e, .. } => {
                let d = r.row_size();
                let r_z_hat = r * z_hat;
                let e_z_hat = e * z_hat;
                let top = (p_hat.slice_rows(0, d) + r_z_hat)
                    .concat_rows(&[&(p_hat.slice_rows(d, 2 * d) + e_z_hat)]);
                let bottom = p_hat.slice_rows(2 * d, p_hat.row_size()) + z_hat;
                top.concat_rows(&[&bottom])
            }
            Self::Statistical { t } => statistical::combine(t, p_hat, z_hat),
        }
    }

    fn sample_pert_square_mat(
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
        let num_blocks = total_ncol.div_ceil(d);
        let padded_ncol = num_blocks * d;
        let padding_ncol = padded_ncol - total_ncol;
        debug!("{}", "sample_pert_square_mat parameters computed");
        let p2z_vec = sample_p2_integers(n * dk, padded_ncol, dgg_large_params, peikert);
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

/// `rows x columns` integers of the perturbation part `p2`, from the discrete Gaussian of width
/// `dgg_large_params.1`: Peikert's inversion table up to [`KARNEY_THRESHOLD`] (an experimentally
/// found threshold), Karney's method above it.
pub(crate) fn sample_p2_integers(
    rows: usize,
    columns: usize,
    dgg_large_params: (Option<f64>, f64, Option<&[f64]>),
    peikert: bool,
) -> I64Matrix {
    let sigma_large = dgg_large_params.1;
    if sigma_large > KARNEY_THRESHOLD {
        let mut matrix = I64Matrix::new_empty(&I64MatrixParams, rows, columns);
        let f = |row_offsets: Range<usize>, col_offsets: Range<usize>| -> Vec<Vec<i64>> {
            parallel_iter!(row_offsets)
                .map(|_| {
                    parallel_iter!(col_offsets.clone())
                        .map(|_| gen_int_karney(0.0, sigma_large))
                        .collect()
                })
                .collect()
        };
        matrix.replace_entries(0..rows, 0..columns, f);
        matrix
    } else {
        let dgg_vectors = gen_dgg_int_vec(
            rows * columns,
            peikert,
            dgg_large_params.0.unwrap(),
            dgg_large_params.1,
            dgg_large_params.2.unwrap(),
        );
        let vecs = parallel_iter!(0..rows)
            .map(|i| dgg_vectors.slice(i * columns, (i + 1) * columns, 0, 1).transpose())
            .collect::<Vec<_>>();
        vecs[0].concat_rows(&vecs[1..].iter().collect::<Vec<_>>())
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
