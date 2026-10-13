//! The statistical instantiation of the gadget trapdoor.
//!
//! TrapGen draws `ā` uniform in `R_q^{1 x m̄}` and `T` from the discrete Gaussian of width `σ` on
//! `R^{m̄ x D}` without truncation, and publishes `A = [ā | g - ā T]`, so `A [T; I] = g`. With
//! enough columns `m̄`, `A` is statistically close to uniform by the regularity lemma; no
//! computational assumption is needed for it.
//!
//! A preimage of `u` is `[p_1 + T z; p_2 + z]` for a perturbation `p = [p_1; p_2]` and a gadget
//! preimage `z` of `u - A p` of width `c = c_G`, as in the computational instantiation. The
//! perturbation has covariance `Σ_p = s² I - c² [T; I][T; I]*`, where `*` is the ring adjoint
//! (`x ↦ x^{-1}` entrywise) followed by transposition. Widths here are standard deviations, and
//! a covariance is the covariance of the coefficient vector.
//!
//! The perturbation is drawn by Peikert's convolution sampler (CRYPTO 2010, Algorithm 2), as in
//! the statistical trapdoor of DRTV26: `p_2` from `D_{Z^{nD}, √(s² - c²)}`, and `p_1` from the
//! conditional distribution, whose center and covariance are
//!
//! ```text
//! μ   = -c² / (s² - c²) · T p_2
//! Σ_1 = s² I - α T T*,   α = c² s² / (s² - c²),
//! ```
//!
//! as the randomized rounding `p_1 = ⌊μ + y⌉_r` (a sample of `D_{Z, μ + y, r}` per coefficient)
//! of a continuous `y` of covariance `Σ' = Σ_1 - r² I = (s² - r²) I - α T T*`, with rounding
//! width `r = σ`.
//!
//! The negacyclic Fourier transform diagonalizes ring multiplication, so `Σ'` is block diagonal
//! over the evaluation points `ω_j = exp(iπ(2j + 1)/n)`: its block at `ω_j` is the Hermitian
//! `m̄ x m̄` matrix `Σ'_j = (s² - r²) I - α T(ω_j) T(ω_j)*`. The points pair up as
//! `ω_{n-1-j} = conj(ω_j)`, and the values of a real polynomial at a pair are conjugate, so TrapGen
//! keeps the values of `T` and the Cholesky factors `L_j L_j* = Σ'_j` at the `n / 2` points
//! `j < n / 2` only ([`StatisticalPrecomputation`]). The sampler draws `q̂_j` for `j < n / 2` as
//! independent circular complex Gaussians with `E|q̂_j|² = n` (the law of the transform `q̂ = F q`
//! of a real standard normal `q`, whose partner values are the conjugates), takes
//! `ŷ_j = L_j q̂_j` and `ŷ_{n-1-j} = conj(ŷ_j)`, and transforms back with the factor `1 / n`, so
//! `y` is real with covariance `F^{-1} diag(Σ'_j) F^{-*} n = Σ'`. The center `μ` is added in the
//! same domain.
//!
//! The factorization is also the check of the trapdoor: it succeeds iff every `Σ'_j` is positive
//! definite, i.e. `s_1(T(ω_j))² < (s² - r²) / α` at every point, which the minimum width of
//! [`crate::sampler::bounds::minimum_preimage_sigma`] predicts for the estimate
//! [`crate::sampler::bounds::trapdoor_singular_value_estimate`]. A pivot below
//! [`PIVOT_FLOOR`]` · s²` counts as a failure, since a binary64 factorization of a nearly singular
//! matrix may still complete. TrapGen redraws `T` after a failure and reports an error after
//! [`TRAPGEN_ATTEMPTS`] failures, which means that the estimate is wrong for the parameters. The
//! largest point singular value of the accepted `T` (by power iteration, a lower bound of
//! `s_1(T)`) is logged next to the estimate.
//!
//! The floating-point evaluation and the convolution of the continuous and discrete parts are
//! heuristic, as are the samplers of the computational instantiation. The precomputation holds
//! `(n / 2) (m̄ D + m̄ (m̄ + 1) / 2)` complex values.

use crate::{
    matrix::{
        PolyMatrix,
        dcrt_poly::DCRTPolyMatrix,
        i64::{I64Matrix, I64MatrixParams},
    },
    poly::{
        PolyParams,
        dcrt::{native::ffi::exact_basis_small_coefficients, params::DCRTPolyParams},
    },
    sampler::{
        DistType, PolyUniformSampler, trapdoor::utils::split_int64_mat_to_elems,
        uniform::DCRTPolyUniformSampler,
    },
};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{f64::consts::PI, ops::Range};
use tracing::{info, warn};

/// Rows of `T` per stored chunk.
pub const ROW_CHUNK: usize = 8;
/// Draws of `T` before TrapGen reports that the perturbation covariance is not positive definite.
pub const TRAPGEN_ATTEMPTS: usize = 8;
/// The smallest accepted Cholesky pivot, relative to `s²`.
pub const PIVOT_FLOOR: f64 = 1.0 / (1u64 << 20) as f64;
/// Iterations of the power iteration that measures the point singular values of `T`.
pub const POWER_ITERATIONS: usize = 64;
/// Tail cut, in rounding widths, of the rejection sampler of the randomized rounding.
const ROUNDING_TAIL: f64 = 12.0;

/// The row counts of the chunks of a `T` of `m_bar` rows: [`ROW_CHUNK`] each but the last.
pub fn chunk_row_counts(m_bar: usize) -> Vec<usize> {
    (0..m_bar).step_by(ROW_CHUNK).map(|start| ROW_CHUNK.min(m_bar - start)).collect()
}

/// The entries `m_bar (m_bar + 1) / 2` of a packed `m_bar x m_bar` lower triangle.
pub fn triangle_len(m_bar: usize) -> Option<usize> {
    Some(m_bar.checked_mul(m_bar.checked_add(1)?)? / 2)
}

/// The index of entry `(i, k)`, `k ≤ i`, of a packed lower triangle.
pub(crate) const fn triangle_index(i: usize, k: usize) -> usize {
    i * (i + 1) / 2 + k
}

/// The bytes of the header of a [`StatisticalPrecomputation`]: the width and the measured
/// singular value.
pub const PRECOMPUTATION_HEADER_BYTES: usize = 16;

/// The byte length of the [`StatisticalPrecomputation`] of an `m_bar x digits` trapdoor in ring
/// dimension `n`: the header, then `(n / 2) (m_bar · digits + m_bar (m_bar + 1) / 2)` complex
/// binary64 values.
pub fn precomputation_bytes(n: usize, m_bar: usize, digits: usize) -> Option<usize> {
    let complex =
        (n / 2).checked_mul(m_bar.checked_mul(digits)?.checked_add(triangle_len(m_bar)?)?)?;
    complex.checked_mul(16)?.checked_add(PRECOMPUTATION_HEADER_BYTES)
}

/// What statistical TrapGen precomputes for the preimages of width `preimage_sigma`, with the
/// largest point singular value `measured_singular_value` of `T` by power iteration (recorded
/// next to the estimate), in the negacyclic Fourier domain at the points `j < n / 2` (module
/// documentation), as interleaved complex binary64 values with consecutive points adjacent:
///
/// - `spectra`: `T_{ik}(ω_j)` at `(i · D + k) · n / 2 + j`;
/// - `factor`: entry `(i, k)`, `k ≤ i`, of the Cholesky factor `L_j` at `triangle_index(i, k) · n /
///   2 + j`.
///
/// Its encoding ([`Self::to_bytes`]) is also the layout of the trapdoor's precomputation leaf on
/// the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct StatisticalPrecomputation {
    pub preimage_sigma: f64,
    pub measured_singular_value: f64,
    pub spectra: Vec<f64>,
    pub factor: Vec<f64>,
}

impl StatisticalPrecomputation {
    /// `preimage_sigma`, `measured_singular_value`, `spectra`, and `factor`, little-endian
    /// binary64.
    pub fn to_bytes(&self) -> Vec<u8> {
        [self.preimage_sigma, self.measured_singular_value]
            .into_iter()
            .chain(self.spectra.iter().copied())
            .chain(self.factor.iter().copied())
            .flat_map(f64::to_le_bytes)
            .collect()
    }

    /// Decodes [`Self::to_bytes`] of an `m_bar x digits` trapdoor in ring dimension `n`.
    pub fn from_bytes(bytes: &[u8], n: usize, m_bar: usize, digits: usize) -> Option<Self> {
        if bytes.len() != precomputation_bytes(n, m_bar, digits)? {
            return None;
        }
        let words = bytes
            .chunks_exact(8)
            .map(|word| f64::from_le_bytes(word.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        let spectra_end = 2 + n * m_bar * digits;
        Some(Self {
            preimage_sigma: words[0],
            measured_singular_value: words[1],
            spectra: words[2..spectra_end].to_vec(),
            factor: words[spectra_end..].to_vec(),
        })
    }
}

/// The scalars of the statistical perturbation sampler for preimage width `s`, gadget width `c`,
/// and rounding width `r`.
#[derive(Clone, Copy, Debug)]
pub struct PerturbationWidths {
    /// `-c² / (s² - c²)`, the scale of the center `μ = center_scale · T p_2`.
    pub center_scale: f64,
    /// `α = c² s² / (s² - c²)`.
    pub alpha: f64,
    /// `s² - r²`, the diagonal of `Σ'`.
    pub variance: f64,
    /// The smallest accepted pivot, [`PIVOT_FLOOR`]` · s²`.
    pub pivot_floor: f64,
    /// The rounding width `r`.
    pub rounding: f64,
}

impl PerturbationWidths {
    /// `None` unless `s > c` and `s > r`.
    pub fn new(s: f64, c: f64, rounding: f64) -> Option<Self> {
        (s.is_finite() && c > 0.0 && rounding > 0.0 && s > c && s > rounding).then(|| Self {
            center_scale: -c * c / (s * s - c * c),
            alpha: c * c * s * s / (s * s - c * c),
            variance: s * s - rounding * rounding,
            pivot_floor: PIVOT_FLOOR * s * s,
            rounding,
        })
    }
}

/// Statistical TrapGen for preimages of width `preimage_sigma` (`widths`): the row chunks of
/// `T`, the precomputation, and the public matrix `[ā | g - ā T]`. Redraws `T` while the
/// factorization fails, at most [`TRAPGEN_ATTEMPTS`] times; `None` after that many failures.
/// `estimate` is `ŝ_1`, which is logged next to the measured value.
pub(crate) fn trapgen(
    params: &DCRTPolyParams,
    m_bar: usize,
    sigma: f64,
    preimage_sigma: f64,
    widths: &PerturbationWidths,
    estimate: f64,
) -> Option<(Vec<DCRTPolyMatrix>, StatisticalPrecomputation, DCRTPolyMatrix)> {
    let sampler = DCRTPolyUniformSampler::new();
    let digits = params.modulus_digits();
    let n = params.ring_dimension() as usize;
    let fft = NegacyclicFft::new(n);
    let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
    for attempt in 1..=TRAPGEN_ATTEMPTS {
        let chunks = chunk_row_counts(m_bar)
            .into_iter()
            .map(|rows| sampler.sample_uniform(params, rows, digits, dist.clone()))
            .collect::<Vec<_>>();
        let spectra = half_spectra(&fft, &chunks);
        let measured = measured_singular_value(&spectra, n, m_bar, digits);
        match factor(&spectra, n, m_bar, digits, widths) {
            Ok(factor) => {
                info!(
                    attempt,
                    measured_s1 = measured,
                    estimated_s1 = estimate,
                    "statistical trapdoor accepted"
                );
                let a_bar = sampler.sample_uniform(params, 1, m_bar, DistType::FinRingDist);
                let mut a_bar_t = DCRTPolyMatrix::zero(params, 1, digits);
                for (range, chunk) in chunk_row_ranges(&chunks).into_iter().zip(&chunks) {
                    a_bar_t = a_bar_t + &(a_bar.slice_columns(range.start, range.end) * chunk);
                }
                let public = a_bar
                    .concat_columns(&[&(DCRTPolyMatrix::gadget_matrix(params, 1, None) - a_bar_t)]);
                let precomputation = StatisticalPrecomputation {
                    preimage_sigma,
                    measured_singular_value: measured,
                    spectra,
                    factor,
                };
                return Some((chunks, precomputation, public));
            }
            Err(point) => warn!(
                attempt,
                point,
                measured_s1 = measured,
                estimated_s1 = estimate,
                "statistical trapdoor redrawn: its perturbation covariance is not positive definite"
            ),
        }
    }
    None
}

/// Whether two chunked `T` hold the same rows, whatever their chunking.
pub(crate) fn rows_equal(left: &[DCRTPolyMatrix], right: &[DCRTPolyMatrix]) -> bool {
    let rows = |chunks: &[DCRTPolyMatrix]| {
        chunks
            .iter()
            .flat_map(|chunk| (0..chunk.row_size()).map(|row| chunk.get_row(row)))
            .collect::<Vec<_>>()
    };
    !left.is_empty() && rows(left) == rows(right)
}

fn chunk_row_ranges(chunks: &[DCRTPolyMatrix]) -> Vec<Range<usize>> {
    let mut start = 0;
    chunks
        .iter()
        .map(|chunk| {
            let range = start..start + chunk.row_size();
            start = range.end;
            range
        })
        .collect()
}

/// The preimage `[p_1 + T z; p_2 + z]`.
pub(crate) fn combine(
    t: &[DCRTPolyMatrix],
    p_hat: DCRTPolyMatrix,
    z_hat: &DCRTPolyMatrix,
) -> DCRTPolyMatrix {
    let ranges = chunk_row_ranges(t);
    let m_bar = ranges.last().map_or(0, |range| range.end);
    let top = ranges
        .iter()
        .zip(t)
        .map(|(range, chunk)| p_hat.slice_rows(range.start, range.end) + &(chunk * z_hat))
        .collect::<Vec<_>>();
    let bottom = p_hat.slice_rows(m_bar, p_hat.row_size()) + z_hat;
    top[0].concat_rows(&top[1..].iter().chain(std::iter::once(&bottom)).collect::<Vec<_>>())
}

/// The values of the entries of `T` at the points `j < n / 2`, in the layout of
/// [`StatisticalPrecomputation::spectra`].
pub(crate) fn half_spectra(fft: &NegacyclicFft, t: &[DCRTPolyMatrix]) -> Vec<f64> {
    let spectra = t.iter().flat_map(|chunk| chunk_spectra(fft, chunk)).collect::<Vec<_>>();
    let points = fft.n / 2;
    spectra.par_iter().flat_map_iter(|spectrum| spectrum[..2 * points].iter().copied()).collect()
}

/// The value at the point `j` of entry `index` of a point-major array of `points` points.
fn at(values: &[f64], points: usize, index: usize, j: usize) -> (f64, f64) {
    let offset = 2 * (index * points + j);
    (values[offset], values[offset + 1])
}

/// The packed Cholesky factors of `Σ'_j = variance I - α T(ω_j) T(ω_j)*` of the points
/// `j < n / 2` for the `spectra` of [`StatisticalPrecomputation`], in the layout of its `factor`,
/// or the first point with a pivot below `pivot_floor`.
pub(crate) fn factor(
    spectra: &[f64],
    n: usize,
    m_bar: usize,
    digits: usize,
    widths: &PerturbationWidths,
) -> Result<Vec<f64>, usize> {
    let points = n / 2;
    let triangle = m_bar * (m_bar + 1) / 2;
    let factors = (0..points)
        .into_par_iter()
        .map(|j| {
            let entry = |i: usize, k: usize| at(spectra, points, i * digits + k, j);
            // The lower triangle of Σ'_j, overwritten by its factor.
            let mut l = vec![(0.0, 0.0); triangle];
            for i in 0..m_bar {
                for p in 0..=i {
                    let (mut re, mut im) = (0.0, 0.0);
                    for k in 0..digits {
                        // T_ik conj(T_pk)
                        let (a, b) = entry(i, k);
                        let (c, d) = entry(p, k);
                        re += a * c + b * d;
                        im += b * c - a * d;
                    }
                    let diagonal = if i == p { widths.variance } else { 0.0 };
                    l[triangle_index(i, p)] = (diagonal - widths.alpha * re, -widths.alpha * im);
                }
            }
            for k in 0..m_bar {
                let mut pivot = l[triangle_index(k, k)].0;
                for p in 0..k {
                    let (a, b) = l[triangle_index(k, p)];
                    pivot -= a * a + b * b;
                }
                // `!(≥)` also rejects a NaN pivot.
                if !(pivot >= widths.pivot_floor) {
                    return Err(j);
                }
                let diagonal = pivot.sqrt();
                l[triangle_index(k, k)] = (diagonal, 0.0);
                for i in k + 1..m_bar {
                    let (mut re, mut im) = l[triangle_index(i, k)];
                    for p in 0..k {
                        // L_ip conj(L_kp)
                        let (a, b) = l[triangle_index(i, p)];
                        let (c, d) = l[triangle_index(k, p)];
                        re -= a * c + b * d;
                        im -= b * c - a * d;
                    }
                    l[triangle_index(i, k)] = (re / diagonal, im / diagonal);
                }
            }
            Ok(l)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = vec![0.0; 2 * triangle * points];
    values.par_chunks_mut(2 * points).enumerate().for_each(|(entry, out)| {
        for (j, l) in factors.iter().enumerate() {
            out[2 * j] = l[entry].0;
            out[2 * j + 1] = l[entry].1;
        }
    });
    Ok(values)
}

/// The largest singular value of `T(ω_j)` over the points `j < n / 2` for the `spectra` of
/// [`StatisticalPrecomputation`], by [`POWER_ITERATIONS`] steps of the power iteration on
/// `T(ω_j)* T(ω_j)` from the all-ones vector: a lower bound of `s_1(T)`.
pub(crate) fn measured_singular_value(
    spectra: &[f64],
    n: usize,
    m_bar: usize,
    digits: usize,
) -> f64 {
    let points = n / 2;
    (0..points)
        .into_par_iter()
        .map(|j| {
            let entry = |i: usize, k: usize| at(spectra, points, i * digits + k, j);
            let mut v = vec![(1.0, 0.0); digits];
            let mut lambda = 0.0f64;
            for _ in 0..POWER_ITERATIONS {
                let norm = v.iter().map(|(a, b)| a * a + b * b).sum::<f64>().sqrt();
                if norm == 0.0 {
                    return 0.0;
                }
                let mut next = vec![(0.0, 0.0); digits];
                for i in 0..m_bar {
                    // u = T_i· v / ‖v‖
                    let (mut ur, mut ui) = (0.0, 0.0);
                    for (k, (a, b)) in v.iter().enumerate() {
                        let (c, d) = entry(i, k);
                        ur += (c * a - d * b) / norm;
                        ui += (c * b + d * a) / norm;
                    }
                    // next += conj(T_i·) u
                    for (k, out) in next.iter_mut().enumerate() {
                        let (c, d) = entry(i, k);
                        out.0 += c * ur + d * ui;
                        out.1 += c * ui - d * ur;
                    }
                }
                // ‖G v‖ / ‖v‖ for G = T* T increases to λ_max(G).
                lambda = next.iter().map(|(a, b)| a * a + b * b).sum::<f64>().sqrt();
                v = next;
            }
            lambda.sqrt()
        })
        .reduce(|| 0.0, f64::max)
}

/// The perturbation `[p_1; p_2]` of the module documentation for an `m_bar`-row trapdoor with
/// `precomputation`, the integers `p2z_vec` of `p_2` (`n·D x columns`), and `widths`.
pub(crate) fn perturbation(
    params: &DCRTPolyParams,
    m_bar: usize,
    precomputation: &StatisticalPrecomputation,
    p2z_vec: I64Matrix,
    widths: &PerturbationWidths,
) -> DCRTPolyMatrix {
    let n = params.ring_dimension() as usize;
    let digits = params.modulus_digits();
    let columns = p2z_vec.ncol;
    let fft = NegacyclicFft::new(n);
    let center_spectra = (0..digits * columns)
        .into_par_iter()
        .map(|index| {
            let (digit, column) = (index / columns, index % columns);
            fft.forward(
                &(0..n)
                    .map(|k| widths.center_scale * p2z_vec.entry(digit * n + k, column) as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut seed = [0u8; 32];
    rand::rng().fill_bytes(&mut seed);
    let centers =
        continuous_centers(&fft, precomputation, m_bar, digits, &center_spectra, columns, &seed);
    // `rounded[row · columns + column]` holds the `n` coefficients of entry `(row, column)`.
    let rounded = centers
        .into_par_iter()
        .map(|centers| {
            let mut rng = rand::rng();
            centers
                .into_iter()
                .map(|center| round_gaussian(&mut rng, center, widths.rounding))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut p1 = I64Matrix::new_empty(&I64MatrixParams, m_bar * n, columns);
    p1.replace_entries(0..m_bar * n, 0..columns, |row_range, column_range| {
        row_range
            .map(|index| {
                column_range
                    .clone()
                    .map(|column| rounded[(index / n) * columns + column][index % n])
                    .collect()
            })
            .collect()
    });
    let p1 = split_int64_mat_to_elems(&p1, params);
    let p2 = split_int64_mat_to_elems(&p2z_vec, params);
    p1.concat_rows(&[&p2])
}

/// The coefficients of `μ + y = F^{-1}(T̂ ĉ + L q̂)` of every entry `(row, column)` of `p_1`, at
/// `row · columns + column`, for the centers `ĉ` (`D x columns` Fourier vectors of
/// `center_scale · p_2`) and `q̂` derived from `seed`.
fn continuous_centers(
    fft: &NegacyclicFft,
    precomputation: &StatisticalPrecomputation,
    m_bar: usize,
    digits: usize,
    center_spectra: &[Vec<f64>],
    columns: usize,
    seed: &[u8; 32],
) -> Vec<Vec<f64>> {
    let n = fft.n;
    let points = n / 2;
    // q̂ at the points j < n / 2: the real and imaginary parts are N(0, n / 2).
    let scale = (n as f64 / 2.0).sqrt();
    let normals = (0..m_bar * columns)
        .into_par_iter()
        .map(|index| {
            standard_normal(seed, index / columns, index % columns, n)
                .into_iter()
                .map(|z| scale * z)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (0..m_bar * columns)
        .into_par_iter()
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            let mut spectrum = vec![0.0; 2 * n];
            for j in 0..points {
                let (mut re, mut im) = (0.0, 0.0);
                for digit in 0..digits {
                    let (a, b) = at(&precomputation.spectra, points, row * digits + digit, j);
                    let center = &center_spectra[digit * columns + column];
                    let (c, d) = (center[2 * j], center[2 * j + 1]);
                    re += a * c - b * d;
                    im += a * d + b * c;
                }
                for k in 0..=row {
                    let (a, b) = at(&precomputation.factor, points, triangle_index(row, k), j);
                    let q = &normals[k * columns + column];
                    let (c, d) = (q[2 * j], q[2 * j + 1]);
                    re += a * c - b * d;
                    im += a * d + b * c;
                }
                spectrum[2 * j] = re;
                spectrum[2 * j + 1] = im;
                spectrum[2 * (n - 1 - j)] = re;
                spectrum[2 * (n - 1 - j) + 1] = -im;
            }
            fft.inverse(&spectrum)
        })
        .collect()
}

/// The Fourier vectors of the entries of one row chunk of `T`, row-major.
fn chunk_spectra(fft: &NegacyclicFft, chunk: &DCRTPolyMatrix) -> Vec<Vec<f64>> {
    let (rows, digits) = chunk.size();
    (0..rows * digits)
        .into_par_iter()
        .map(|index| {
            let entry = chunk.entry(index / digits, index % digits);
            let coefficients = exact_basis_small_coefficients(entry.get_poly())
                .expect("small trapdoor coefficients")
                .into_iter()
                .map(|value| value as f64)
                .collect::<Vec<_>>();
            fft.forward(&coefficients)
        })
        .collect()
}

/// `n` standard normal values determined by `seed`, `row`, and `column`.
fn standard_normal(seed: &[u8; 32], row: usize, column: usize, n: usize) -> Vec<f64> {
    let digest = Sha256::new()
        .chain_update(b"mxx-statistical-perturbation")
        .chain_update(seed)
        .chain_update((row as u64).to_le_bytes())
        .chain_update((column as u64).to_le_bytes())
        .finalize();
    let mut rng = StdRng::from_seed(digest.into());
    (0..n).map(|_| gaussian(&mut rng)).collect()
}

/// One standard normal value (Box–Muller).
fn gaussian(rng: &mut impl Rng) -> f64 {
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// One sample of `D_{Z, center, width}` by rejection from the integers within
/// [`ROUNDING_TAIL`] widths of the center.
fn round_gaussian(rng: &mut impl Rng, center: f64, width: f64) -> i64 {
    let low = (center - ROUNDING_TAIL * width).floor() as i64;
    let high = (center + ROUNDING_TAIL * width).ceil() as i64;
    loop {
        let x = rng.random_range(low..=high);
        let distance = x as f64 - center;
        if rng.random::<f64>() < (-distance * distance / (2.0 * width * width)).exp() {
            return x;
        }
    }
}

/// The negacyclic Fourier transform of `R = R[x]/(x^n + 1)`: a real polynomial `a` maps to its
/// values `a(ω_j)` at `ω_j = exp(iπ(2j + 1)/n)`, stored as interleaved complex numbers. Ring
/// multiplication is pointwise, and the adjoint `a(x^{-1})` is the complex conjugate.
pub struct NegacyclicFft {
    n: usize,
    /// `exp(iπ l / n)`, interleaved.
    twist: Vec<f64>,
    /// `exp(2πi m / n)` for `m < n / 2`, interleaved.
    roots: Vec<f64>,
}

impl NegacyclicFft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "the ring dimension is a power of two");
        let unit = |angle: f64| [angle.cos(), angle.sin()];
        Self {
            n,
            twist: (0..n).flat_map(|l| unit(PI * l as f64 / n as f64)).collect(),
            roots: (0..n / 2).flat_map(|m| unit(2.0 * PI * m as f64 / n as f64)).collect(),
        }
    }

    pub fn forward(&self, coefficients: &[f64]) -> Vec<f64> {
        debug_assert_eq!(coefficients.len(), self.n);
        let mut data = coefficients
            .iter()
            .zip(self.twist.chunks_exact(2))
            .flat_map(|(a, twist)| [a * twist[0], a * twist[1]])
            .collect::<Vec<_>>();
        self.transform(&mut data, false);
        data
    }

    pub fn inverse(&self, spectrum: &[f64]) -> Vec<f64> {
        let mut data = spectrum.to_vec();
        self.transform(&mut data, true);
        let scale = 1.0 / self.n as f64;
        data.chunks_exact(2)
            .zip(self.twist.chunks_exact(2))
            .map(|(value, twist)| (value[0] * twist[0] + value[1] * twist[1]) * scale)
            .collect()
    }

    /// The in-place radix-2 DFT `Σ_l x_l exp(±2πi jl / n)` (`-` when `inverse`).
    fn transform(&self, data: &mut [f64], inverse: bool) {
        let n = self.n;
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                data.swap(2 * i, 2 * j);
                data.swap(2 * i + 1, 2 * j + 1);
            }
        }
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) =
                        (self.roots[2 * k * stride], sign * self.roots[2 * k * stride + 1]);
                    let (u, v) = (start + k, start + k + len / 2);
                    let (vr, vi) = (data[2 * v], data[2 * v + 1]);
                    let (tr, ti) = (vr * wr - vi * wi, vr * wi + vi * wr);
                    let (ur, ui) = (data[2 * u], data[2 * u + 1]);
                    data[2 * u] = ur + tr;
                    data[2 * u + 1] = ui + ti;
                    data[2 * v] = ur - tr;
                    data[2 * v + 1] = ui - ti;
                }
            }
            len <<= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::ToPrimitive;

    /// `acc += a b` for interleaved complex vectors.
    fn mul_acc(acc: &mut [f64], a: &[f64], b: &[f64]) {
        for ((acc, a), b) in acc.chunks_exact_mut(2).zip(a.chunks_exact(2)).zip(b.chunks_exact(2)) {
            acc[0] += a[0] * b[0] - a[1] * b[1];
            acc[1] += a[0] * b[1] + a[1] * b[0];
        }
    }

    /// `acc += conj(a) b` for interleaved complex vectors.
    fn conj_mul_acc(acc: &mut [f64], a: &[f64], b: &[f64]) {
        for ((acc, a), b) in acc.chunks_exact_mut(2).zip(a.chunks_exact(2)).zip(b.chunks_exact(2)) {
            acc[0] += a[0] * b[0] + a[1] * b[1];
            acc[1] += a[0] * b[1] - a[1] * b[0];
        }
    }

    fn negacyclic_product(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut out = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                let k = i + j;
                if k < n {
                    out[k] += a[i] * b[j];
                } else {
                    out[k - n] -= a[i] * b[j];
                }
            }
        }
        out
    }

    #[test]
    fn negacyclic_fft_multiplies_pointwise_and_conjugates_the_adjoint() {
        let n = 16;
        let fft = NegacyclicFft::new(n);
        let a = (0..n).map(|i| (i as f64 * 0.7).sin() * 5.0).collect::<Vec<_>>();
        let b = (0..n).map(|i| (i as f64 * 1.3).cos() * 3.0 - 1.0).collect::<Vec<_>>();
        let roundtrip = fft.inverse(&fft.forward(&a));
        assert!(roundtrip.iter().zip(&a).all(|(x, y)| (x - y).abs() < 1e-12));
        let mut product = vec![0.0; 2 * n];
        mul_acc(&mut product, &fft.forward(&a), &fft.forward(&b));
        let expected = negacyclic_product(&a, &b);
        assert!(fft.inverse(&product).iter().zip(&expected).all(|(x, y)| (x - y).abs() < 1e-9));
        // The adjoint a(x^{-1}) = a_0 - Σ_{l>0} a_l x^{n-l} is the conjugate, unlike the plain
        // coefficient reversal of a transpose.
        let adjoint = (0..n).map(|l| if l == 0 { a[0] } else { -a[n - l] }).collect::<Vec<_>>();
        let mut conjugate = vec![0.0; 2 * n];
        conj_mul_acc(&mut conjugate, &fft.forward(&a), &fft.forward(&b));
        let expected = negacyclic_product(&adjoint, &b);
        assert!(fft.inverse(&conjugate).iter().zip(&expected).all(|(x, y)| (x - y).abs() < 1e-9));
        let reversed = (0..n).map(|l| a[n - 1 - l]).collect::<Vec<_>>();
        let plain = negacyclic_product(&reversed, &b);
        assert!(plain.iter().zip(&expected).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    /// The real `m̄n x Dn` coefficient-embedding matrix of a chunked `T`: column `k n + b` holds
    /// the coefficients of `T_{·k} x^b`.
    fn embedding(t: &[DCRTPolyMatrix], n: usize) -> Vec<Vec<f64>> {
        let digits = t[0].col_size();
        let rows = t.iter().flat_map(|chunk| (0..chunk.row_size()).map(move |row| (chunk, row)));
        let mut matrix = Vec::new();
        for (chunk, row) in rows {
            let entries = (0..digits)
                .map(|k| exact_basis_small_coefficients(chunk.entry(row, k).get_poly()).unwrap())
                .collect::<Vec<_>>();
            for a in 0..n {
                let mut line = vec![0.0; digits * n];
                for (k, coefficients) in entries.iter().enumerate() {
                    for b in 0..n {
                        // The coefficient of x^a in t x^b is t_{a-b}, negated on wrap-around.
                        line[k * n + b] = if a >= b {
                            coefficients[a - b] as f64
                        } else {
                            -(coefficients[a + n - b] as f64)
                        };
                    }
                }
                matrix.push(line);
            }
        }
        matrix
    }

    fn sample_t(params: &DCRTPolyParams, m_bar: usize, sigma: f64) -> Vec<DCRTPolyMatrix> {
        let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
        chunk_row_counts(m_bar)
            .into_iter()
            .map(|rows| {
                DCRTPolyUniformSampler::new().sample_uniform(
                    params,
                    rows,
                    params.modulus_digits(),
                    dist.clone(),
                )
            })
            .collect()
    }

    /// The sample covariance of the continuous part `y` (zero centers) of the coefficient vector.
    fn sampled_covariance(
        t: &[DCRTPolyMatrix],
        n: usize,
        widths: &PerturbationWidths,
    ) -> Vec<Vec<f64>> {
        let columns = 20_000;
        let fft = NegacyclicFft::new(n);
        let digits = t[0].col_size();
        let m_bar = t.iter().map(|chunk| chunk.row_size()).sum::<usize>();
        let spectra = half_spectra(&fft, t);
        let factor = factor(&spectra, n, m_bar, digits, widths).expect("positive definite");
        let precomputation = StatisticalPrecomputation {
            preimage_sigma: 0.0,
            measured_singular_value: 0.0,
            spectra,
            factor,
        };
        let zero = vec![vec![0.0; 2 * n]; digits * columns];
        let mut seed = [0u8; 32];
        rand::rng().fill_bytes(&mut seed);
        let mut samples = vec![vec![0.0; m_bar * n]; columns];
        for (index, centers) in
            continuous_centers(&fft, &precomputation, m_bar, digits, &zero, columns, &seed)
                .into_iter()
                .enumerate()
        {
            let (row, column) = (index / columns, index % columns);
            for (k, center) in centers.into_iter().enumerate() {
                samples[column][row * n + k] = center;
            }
        }
        (0..m_bar * n)
            .map(|i| {
                (0..m_bar * n)
                    .map(|j| samples.iter().map(|y| y[i] * y[j]).sum::<f64>() / columns as f64)
                    .collect()
            })
            .collect()
    }

    /// `M Mᵀ` of the coefficient embedding `M` of `T`.
    fn embedding_gram(t: &[DCRTPolyMatrix], n: usize) -> Vec<Vec<f64>> {
        let m = embedding(t, n);
        (0..m.len())
            .map(|i| {
                (0..m.len()).map(|j| m[i].iter().zip(&m[j]).map(|(x, y)| x * y).sum()).collect()
            })
            .collect()
    }

    /// Widths of `s = 10`, `r = 1` whose `α` makes `α λ = variance / 2`.
    fn widths_for(lambda: f64) -> PerturbationWidths {
        let (s, r) = (10.0f64, 1.0f64);
        let alpha = 0.5 * (s * s - r * r) / lambda;
        let c = (alpha * s * s / (s * s + alpha)).sqrt();
        let widths = PerturbationWidths::new(s, c, r).unwrap();
        assert!((widths.alpha - alpha).abs() < 1e-9 * alpha);
        widths
    }

    #[test]
    #[serial_test::serial]
    fn continuous_part_has_the_target_covariance_with_the_ring_adjoint() {
        let n = 4;
        let params = DCRTPolyParams::new(n as u32, 1, 20, 10, None, None);
        let t = sample_t(&params, 3, 2.0);
        let gram = embedding_gram(&t, n);
        let fft = NegacyclicFft::new(n);
        let lambda =
            measured_singular_value(&half_spectra(&fft, &t), n, 3, params.modulus_digits()).powi(2);
        let widths = widths_for(lambda);
        let covariance = sampled_covariance(&t, n, &widths);
        for i in 0..gram.len() {
            for j in 0..gram.len() {
                let expected =
                    if i == j { widths.variance } else { 0.0 } - widths.alpha * gram[i][j];
                assert!(
                    (covariance[i][j] - expected).abs() < 0.06 * widths.variance,
                    "covariance ({i}, {j}) = {}, expected {expected}",
                    covariance[i][j]
                );
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn continuous_part_of_a_zero_trapdoor_is_isotropic() {
        let n = 4;
        let params = DCRTPolyParams::new(n as u32, 1, 20, 10, None, None);
        let t = vec![DCRTPolyMatrix::zero(&params, 3, params.modulus_digits())];
        let widths = widths_for(1.0);
        let covariance = sampled_covariance(&t, n, &widths);
        for (i, row) in covariance.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = if i == j { widths.variance } else { 0.0 };
                assert!((value - expected).abs() < 0.06 * widths.variance);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn factorization_accepts_exactly_the_widths_above_the_singular_value() {
        let n = 16;
        let params = DCRTPolyParams::new(n as u32, 1, 20, 10, None, None);
        let t = sample_t(&params, 5, 3.0);
        let fft = NegacyclicFft::new(n);
        let spectra = half_spectra(&fft, &t);
        let digits = params.modulus_digits();
        let s1 = measured_singular_value(&spectra, n, 5, digits);
        // Σ' = variance I - T T* is positive definite iff variance > s_1(T)².
        let widths = |variance: f64| PerturbationWidths {
            center_scale: 0.0,
            alpha: 1.0,
            variance,
            pivot_floor: 0.0,
            rounding: 1.0,
        };
        assert!(factor(&spectra, n, 5, digits, &widths(1.01 * s1 * s1)).is_ok());
        assert!(factor(&spectra, n, 5, digits, &widths(0.99 * s1 * s1)).is_err());
        // A pivot floor above every pivot rejects a positive definite covariance.
        let floored = PerturbationWidths { pivot_floor: 2.0 * s1 * s1, ..widths(1.5 * s1 * s1) };
        assert!(factor(&spectra, n, 5, digits, &floored).is_err());
    }

    /// The statistical singular-value estimate bounds `s_1(T)` of sampled trapdoors: the
    /// factorization of `ŝ_1² I - T T*` succeeds for each sample (it fails with probability about
    /// [`crate::sampler::bounds::STATISTICAL_SPECTRAL_FAILURE`] each).
    #[test]
    #[serial_test::serial]
    fn statistical_singular_value_estimate_bounds_sampled_trapdoors() {
        let samples = std::env::var("MXX_TEST_STATISTICAL_ESTIMATE_SAMPLES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(4usize);
        let sigma = 4.578;
        for (n, m_bar, crt_bits, base_bits) in
            [(64u32, 12usize, 20usize, 10u32), (256, 20, 20, 5), (512, 9, 30, 15)]
        {
            let params = DCRTPolyParams::new(n, 2, crt_bits, base_bits, None, None);
            let digits = params.modulus_digits();
            let estimate = crate::sampler::bounds::trapdoor_singular_value_estimate(
                n,
                1,
                digits,
                sigma,
                mxx_ir_core::types::TrapdoorInstantiation::Statistical { m_bar },
            )
            .unwrap()
            .to_f64()
            .unwrap();
            let fft = NegacyclicFft::new(n as usize);
            for _ in 0..samples {
                let t = sample_t(&params, m_bar, sigma);
                let spectra = half_spectra(&fft, &t);
                let measured = measured_singular_value(&spectra, n as usize, m_bar, digits);
                let widths = PerturbationWidths {
                    center_scale: 0.0,
                    alpha: 1.0,
                    variance: estimate * estimate,
                    pivot_floor: 0.0,
                    rounding: 1.0,
                };
                assert!(
                    factor(&spectra, n as usize, m_bar, digits, &widths).is_ok(),
                    "n = {n}, m̄ = {m_bar}, D = {digits}: s_1 ≥ {measured} exceeds {estimate}"
                );
                // The estimate is not loose by more than a constant factor.
                assert!(measured > 0.5 * estimate, "n = {n}: s_1 ≈ {measured}, ŝ_1 = {estimate}");
            }
        }
    }

    /// TrapGen at a width far below the trapdoor's minimum rejects every draw and gives up after
    /// [`TRAPGEN_ATTEMPTS`].
    #[test]
    #[serial_test::serial]
    fn trapgen_fails_when_the_covariance_is_never_positive_definite() {
        let params = DCRTPolyParams::new(16, 1, 20, 10, None, None);
        let sigma = 4.0;
        let c = crate::sampler::bounds::gadget_sampler_sigma(1 << params.base_bits(), sigma);
        let s = 1.01 * c;
        let widths = PerturbationWidths::new(s, c, sigma).unwrap();
        assert!(trapgen(&params, 5, sigma, s, &widths, 1.0).is_none());
    }

    #[test]
    fn precomputation_encoding_round_trips() {
        let (n, m_bar, digits) = (8, 3, 2);
        let precomputation = StatisticalPrecomputation {
            preimage_sigma: 123.25,
            measured_singular_value: 17.5,
            spectra: (0..n * m_bar * digits).map(|i| i as f64 * 0.25 + 1.0).collect(),
            factor: (0..n / 2 * 2 * triangle_len(m_bar).unwrap())
                .map(|i| i as f64 * 0.5 - 3.0)
                .collect(),
        };
        let bytes = precomputation.to_bytes();
        assert_eq!(bytes.len(), precomputation_bytes(n, m_bar, digits).unwrap());
        assert_eq!(
            StatisticalPrecomputation::from_bytes(&bytes, n, m_bar, digits),
            Some(precomputation)
        );
        assert_eq!(StatisticalPrecomputation::from_bytes(&bytes, n, m_bar + 1, digits), None);
    }

    #[test]
    fn rounding_and_normal_samplers_have_their_moments() {
        let mut rng = rand::rng();
        let count = 200_000;
        let normals = (0..count).map(|_| gaussian(&mut rng)).collect::<Vec<_>>();
        let mean = normals.iter().sum::<f64>() / count as f64;
        let variance = normals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        assert!(mean.abs() < 0.01 && (variance - 1.0).abs() < 0.02);
        let rounded =
            (0..count).map(|_| round_gaussian(&mut rng, 10.3, 4.0) as f64).collect::<Vec<_>>();
        let mean = rounded.iter().sum::<f64>() / count as f64;
        let variance = rounded.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        assert!((mean - 10.3).abs() < 0.05 && (variance - 16.0).abs() < 0.4);
    }
}
