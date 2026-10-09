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
//! The perturbation is drawn as `p_2` from `D_{Z^{nD}, √(s² - c²)}` and `p_1` from the conditional
//! distribution, whose center and covariance are
//!
//! ```text
//! μ   = -c² / (s² - c²) · T p_2
//! Σ_1 = s² I - α T T*,   α = c² s² / (s² - c²).
//! ```
//!
//! `Σ_1` is never formed. With `ŝ_1` the estimate of `s_1(T)` from
//! [`crate::sampler::bounds::trapdoor_singular_value_estimate`], `a = 2 α ŝ_1²` and `β = α / a =
//! 1 / (2 ŝ_1²)`,
//!
//! ```text
//! Σ_1 - r² I = s_iso² I + a (I - β T T*),   s_iso² = s² - r² - a,
//! ```
//!
//! and `s_iso² ≥ 0` is the condition of [`crate::sampler::bounds::minimum_preimage_sigma`]. The
//! correlated part is `y_corr = √a (z - T f(G) T* z)` for `z` standard normal and `G = T* T`,
//! where `f(g) = (1 - √(1 - β g)) / g = β / (1 + √(1 - β g))` is the smaller root of
//! `2 k - k² g = β`, so that `(I - T f(G) T*)² = I - β T T*`. `f(G) v` is evaluated by the
//! Chebyshev expansion of `f` on `[0, ŝ_1²]` with the Clenshaw recurrence, and `G v = T* (T v)`
//! streams over the row chunks of `T`. On `[0, ŝ_1²]`, `β g ≤ 1/2`, and the branch point of `f` at
//! `β g = 1` maps to `x = 3` of the Chebyshev variable, so the expansion converges like
//! `(3 + 2√2)^{-K}`. Every product runs in the negacyclic Fourier domain, where ring
//! multiplication is pointwise and the adjoint is complex conjugation.
//!
//! Finally `p_1` is the randomized rounding (Peikert, CRYPTO 2010) `D_{Z, μ + y, r}` of
//! `y = y_corr + s_iso N(0, I)` with rounding width `r = σ`.
//!
//! The floating-point evaluation, the Chebyshev truncation, the estimate `ŝ_1 ≥ s_1(T)`, and the
//! convolution of the continuous and discrete parts are heuristic, as are the samplers of the
//! computational instantiation. Peak memory holds one row chunk of `T` in the Fourier domain and
//! `D x columns` Fourier vectors, besides the perturbation itself.

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
        DistType, PolyUniformSampler, bounds::trapdoor_singular_value_estimate,
        trapdoor::utils::split_int64_mat_to_elems, uniform::DCRTPolyUniformSampler,
    },
};
use mxx_ir_core::types::TrapdoorInstantiation;
use num_traits::ToPrimitive;
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{f64::consts::PI, ops::Range};

/// Rows of `T` per stored chunk.
pub const ROW_CHUNK: usize = 8;
/// Degree of the Chebyshev expansion of `f`; its relative error is about `(3 + 2√2)^{-24} <
/// 2^{-60}`.
pub(crate) const CHEBYSHEV_DEGREE: usize = 24;
/// Tail cut, in rounding widths, of the rejection sampler of the randomized rounding.
const ROUNDING_TAIL: f64 = 12.0;

/// The row counts of the chunks of a `T` of `m_bar` rows: [`ROW_CHUNK`] each but the last.
pub fn chunk_row_counts(m_bar: usize) -> Vec<usize> {
    (0..m_bar).step_by(ROW_CHUNK).map(|start| ROW_CHUNK.min(m_bar - start)).collect()
}

/// Statistical TrapGen: the row chunks of `T` and the public matrix `[ā | g - ā T]`.
pub(crate) fn trapgen(
    params: &DCRTPolyParams,
    m_bar: usize,
    sigma: f64,
) -> (Vec<DCRTPolyMatrix>, DCRTPolyMatrix) {
    let sampler = DCRTPolyUniformSampler::new();
    let digits = params.modulus_digits();
    let a_bar = sampler.sample_uniform(params, 1, m_bar, DistType::FinRingDist);
    let dist = DistType::GaussDist { sigma, max_coefficient_bound: None };
    let mut a_bar_t = DCRTPolyMatrix::zero(params, 1, digits);
    let mut chunks = Vec::with_capacity(m_bar.div_ceil(ROW_CHUNK));
    let mut start = 0;
    for rows in chunk_row_counts(m_bar) {
        let chunk = sampler.sample_uniform(params, rows, digits, dist.clone());
        a_bar_t = a_bar_t + &(a_bar.slice_columns(start, start + rows) * &chunk);
        chunks.push(chunk);
        start += rows;
    }
    let public =
        a_bar.concat_columns(&[&(DCRTPolyMatrix::gadget_matrix(params, 1, None) - a_bar_t)]);
    (chunks, public)
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

/// The perturbation `[p_1; p_2]` of the module documentation for the integers `p2z_vec` of `p_2`
/// (`n·D x columns`), preimage width `s`, gadget width `c`, and TrapGen width `sigma` (also the
/// rounding width).
pub(crate) fn perturbation(
    params: &DCRTPolyParams,
    t: &[DCRTPolyMatrix],
    p2z_vec: I64Matrix,
    s: f64,
    c: f64,
    sigma: f64,
) -> DCRTPolyMatrix {
    let n = params.ring_dimension() as usize;
    let digits = params.modulus_digits();
    let columns = p2z_vec.ncol;
    let m_bar = t.iter().map(|chunk| chunk.row_size()).sum::<usize>();
    let s1 = trapdoor_singular_value_estimate(
        params.ring_dimension(),
        1,
        digits,
        sigma,
        TrapdoorInstantiation::Statistical { m_bar },
    )
    .and_then(|estimate| estimate.to_f64())
    .expect("valid statistical trapdoor parameters");
    let widths = PerturbationWidths::new(s, c, sigma, s1);
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
    let mut p1 = I64Matrix::new_empty(&I64MatrixParams, m_bar * n, columns);
    correlated_centers(t, &fft, &widths, &center_spectra, columns, &seed, |range, rows| {
        let rows = rows
            .into_par_iter()
            .map(|(row, column, centers)| {
                let mut rng = rand::rng();
                let rounded = centers
                    .into_iter()
                    .map(|center| {
                        let center = center + widths.isotropic * gaussian(&mut rng);
                        round_gaussian(&mut rng, center, widths.rounding)
                    })
                    .collect::<Vec<_>>();
                (row, column, rounded)
            })
            .collect::<Vec<_>>();
        p1.replace_entries(range.start * n..range.end * n, 0..columns, |row_range, _| {
            let mut block = vec![vec![0i64; columns]; row_range.len()];
            for (row, column, values) in &rows {
                for (k, value) in values.iter().enumerate() {
                    block[(row - range.start) * n + k][*column] = *value;
                }
            }
            block
        });
    });
    let p1 = split_int64_mat_to_elems(&p1, params);
    let p2 = split_int64_mat_to_elems(&p2z_vec, params);
    p1.concat_rows(&[&p2])
}

/// `μ + y_corr = T (center - √a f(G) T* z) + √a z` for `center = center_scale · p_2` given as
/// `D x columns` Fourier vectors and `z` standard normal derived from `seed`. The values of each
/// row chunk, in the coefficient domain, are passed to `consume` with the chunk's row range as
/// `(row, column, coefficients)`.
fn correlated_centers(
    t: &[DCRTPolyMatrix],
    fft: &NegacyclicFft,
    widths: &PerturbationWidths,
    center_spectra: &[Vec<f64>],
    columns: usize,
    seed: &[u8; 32],
    mut consume: impl FnMut(Range<usize>, Vec<(usize, usize, Vec<f64>)>),
) {
    let n = fft.n;
    let digits = t[0].col_size();
    let operator = TrapdoorOperator { t, fft, columns };
    // w = T* z, regenerating z from `seed` row by row.
    let w = operator.adjoint(|row, column| fft.forward(&standard_normal(seed, row, column, n)));
    // h = f(G) w.
    let coefficients = chebyshev_coefficients(widths.beta, widths.gmax, CHEBYSHEV_DEGREE);
    let h = clenshaw(&coefficients, widths.gmax, &w, |v| operator.gram(v));
    drop(w);
    let sqrt_a = widths.a.sqrt();
    let q = center_spectra
        .par_iter()
        .zip(h)
        .map(|(center, h)| center.iter().zip(h).map(|(center, h)| center - sqrt_a * h).collect())
        .collect::<Vec<Vec<f64>>>();
    for (range, chunk) in chunk_row_ranges(t).into_iter().zip(t) {
        let spectra = chunk_spectra(fft, chunk);
        let rows = range
            .clone()
            .into_par_iter()
            .flat_map(|row| (0..columns).into_par_iter().map(move |column| (row, column)))
            .map(|(row, column)| {
                let mut spectrum = vec![0.0; 2 * n];
                for digit in 0..digits {
                    mul_acc(
                        &mut spectrum,
                        &spectra[(row - range.start) * digits + digit],
                        &q[digit * columns + column],
                    );
                }
                let centers = fft
                    .inverse(&spectrum)
                    .into_iter()
                    .zip(standard_normal(seed, row, column, n))
                    .map(|(tq, z)| tq + sqrt_a * z)
                    .collect::<Vec<_>>();
                (row, column, centers)
            })
            .collect::<Vec<_>>();
        consume(range, rows);
    }
}

/// The scalars of the statistical perturbation sampler.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PerturbationWidths {
    /// `-c² / (s² - c²)`, the scale of the center `μ = center_scale · T p_2`.
    pub center_scale: f64,
    /// `a = 2 α ŝ_1²`, the scale of the correlated part.
    pub a: f64,
    /// `β = α / a`.
    pub beta: f64,
    /// `ŝ_1²`, the right end of the Chebyshev interval.
    pub gmax: f64,
    /// `s_iso = √(s² - r² - a)`.
    pub isotropic: f64,
    /// The rounding width `r`.
    pub rounding: f64,
}

impl PerturbationWidths {
    pub(crate) fn new(s: f64, c: f64, rounding: f64, s1: f64) -> Self {
        assert!(s > c, "the preimage width is below the gadget sampler's width");
        let alpha = c * c * s * s / (s * s - c * c);
        let a = 2.0 * alpha * s1 * s1;
        // s² - a = s² (s² - c² (1 + 2 ŝ_1²)) / (s² - c²); at the minimum width of
        // `minimum_preimage_sigma` it is about r², and the isotropic variance about 0. Rounding
        // errors of the order of `s²` ulps are clamped.
        let isotropic_variance =
            s * s * (s * s - c * c * (1.0 + 2.0 * s1 * s1)) / (s * s - c * c) - rounding * rounding;
        assert!(
            isotropic_variance >= -1e-12 * s * s,
            "the preimage width is below the statistical trapdoor's minimum"
        );
        let isotropic_variance = isotropic_variance.max(0.0);
        Self {
            center_scale: -c * c / (s * s - c * c),
            a,
            beta: alpha / a,
            gmax: s1 * s1,
            isotropic: isotropic_variance.sqrt(),
            rounding,
        }
    }
}

/// Streaming products with the chunked `T` on Fourier vectors.
struct TrapdoorOperator<'a> {
    t: &'a [DCRTPolyMatrix],
    fft: &'a NegacyclicFft,
    columns: usize,
}

impl TrapdoorOperator<'_> {
    fn digits(&self) -> usize {
        self.t[0].col_size()
    }

    /// `T* x` for `x` given row by row (`x(row, column)`), as `D x columns` Fourier vectors.
    fn adjoint(&self, x: impl Fn(usize, usize) -> Vec<f64> + Sync) -> Vec<Vec<f64>> {
        let (digits, columns, n2) = (self.digits(), self.columns, 2 * self.fft.n);
        let mut out = vec![vec![0.0; n2]; digits * columns];
        for (range, chunk) in chunk_row_ranges(self.t).into_iter().zip(self.t) {
            let spectra = chunk_spectra(self.fft, chunk);
            let rows = range
                .clone()
                .into_par_iter()
                .flat_map(|row| (0..columns).into_par_iter().map(move |column| (row, column)))
                .map(|(row, column)| ((row, column), x(row, column)))
                .collect::<std::collections::BTreeMap<_, _>>();
            out.par_iter_mut().enumerate().for_each(|(index, out)| {
                let (digit, column) = (index / columns, index % columns);
                for row in range.clone() {
                    conj_mul_acc(
                        out,
                        &spectra[(row - range.start) * digits + digit],
                        &rows[&(row, column)],
                    );
                }
            });
        }
        out
    }

    /// `G v = T* (T v)` for `D x columns` Fourier vectors `v`.
    fn gram(&self, v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (digits, columns, n2) = (self.digits(), self.columns, 2 * self.fft.n);
        let mut out = vec![vec![0.0; n2]; digits * columns];
        for (range, chunk) in chunk_row_ranges(self.t).into_iter().zip(self.t) {
            let spectra = chunk_spectra(self.fft, chunk);
            let tv = (0..range.len() * columns)
                .into_par_iter()
                .map(|index| {
                    let (row, column) = (index / columns, index % columns);
                    let mut acc = vec![0.0; n2];
                    for digit in 0..digits {
                        mul_acc(
                            &mut acc,
                            &spectra[row * digits + digit],
                            &v[digit * columns + column],
                        );
                    }
                    acc
                })
                .collect::<Vec<_>>();
            out.par_iter_mut().enumerate().for_each(|(index, out)| {
                let (digit, column) = (index / columns, index % columns);
                for row in 0..range.len() {
                    conj_mul_acc(out, &spectra[row * digits + digit], &tv[row * columns + column]);
                }
            });
        }
        out
    }
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

/// `f(g) = β / (1 + √(1 - β g))`.
pub(crate) fn correction(beta: f64, g: f64) -> f64 {
    beta / (1.0 + (1.0 - beta * g).sqrt())
}

/// Coefficients of the degree-`degree` Chebyshev interpolant of [`correction`] on `[0, gmax]` in
/// the variable `x = 2 g / gmax - 1`, with the constant coefficient halved.
pub(crate) fn chebyshev_coefficients(beta: f64, gmax: f64, degree: usize) -> Vec<f64> {
    let nodes = degree + 1;
    let values = (0..nodes)
        .map(|m| {
            let x = (PI * (m as f64 + 0.5) / nodes as f64).cos();
            correction(beta, gmax * (x + 1.0) / 2.0)
        })
        .collect::<Vec<_>>();
    (0..nodes)
        .map(|k| {
            let sum = values
                .iter()
                .enumerate()
                .map(|(m, value)| value * (PI * k as f64 * (m as f64 + 0.5) / nodes as f64).cos())
                .sum::<f64>();
            let coefficient = 2.0 * sum / nodes as f64;
            if k == 0 { coefficient / 2.0 } else { coefficient }
        })
        .collect()
}

/// `Σ_k c_k T_k(X) w` for `X = 2 G / gmax - I` by the Clenshaw recurrence, with `apply_g`
/// computing `G v`.
pub(crate) fn clenshaw(
    coefficients: &[f64],
    gmax: f64,
    w: &[Vec<f64>],
    apply_g: impl Fn(&[Vec<f64>]) -> Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let apply_x = |v: &[Vec<f64>]| {
        apply_g(v)
            .into_par_iter()
            .zip(v)
            .map(|(gv, v)| gv.iter().zip(v).map(|(gv, v)| 2.0 * gv / gmax - v).collect())
            .collect::<Vec<Vec<f64>>>()
    };
    let zero = || w.iter().map(|v| vec![0.0; v.len()]).collect::<Vec<_>>();
    let combine = |scale: f64, x: &[Vec<f64>], twice: f64, b2: &[Vec<f64>]| {
        w.par_iter()
            .zip(x)
            .zip(b2)
            .map(|((w, x), b2)| {
                w.iter().zip(x).zip(b2).map(|((w, x), b2)| scale * w + twice * x - b2).collect()
            })
            .collect::<Vec<Vec<f64>>>()
    };
    let degree = coefficients.len() - 1;
    let (mut b1, mut b2) = (zero(), zero());
    for k in (1..=degree).rev() {
        let xb1 = if k == degree { zero() } else { apply_x(&b1) };
        let bk = combine(coefficients[k], &xb1, 2.0, &b2);
        b2 = std::mem::replace(&mut b1, bk);
    }
    let xb1 = if degree == 0 { zero() } else { apply_x(&b1) };
    combine(coefficients[0], &xb1, 1.0, &b2)
}

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

    #[test]
    fn chebyshev_expansion_of_the_correction_is_accurate_on_its_interval() {
        let gmax = 1234.5;
        let beta = 1.0 / (2.0 * gmax);
        let coefficients = chebyshev_coefficients(beta, gmax, CHEBYSHEV_DEGREE);
        for step in 0..=200 {
            let g = gmax * step as f64 / 200.0;
            let x = 2.0 * g / gmax - 1.0;
            let (mut b1, mut b2) = (0.0, 0.0);
            for coefficient in coefficients[1..].iter().rev() {
                let bk = coefficient + 2.0 * x * b1 - b2;
                b2 = b1;
                b1 = bk;
            }
            let value = coefficients[0] + x * b1 - b2;
            let exact = correction(beta, g);
            assert!((value - exact).abs() <= 1e-14 * exact, "g = {g}");
            // 2 f - f² g = β, so (I - T f T*)² = I - β T T*.
            assert!((2.0 * exact - exact * exact * g - beta).abs() < 1e-15 * beta);
        }
    }

    #[test]
    fn clenshaw_matches_the_scalar_expansion_on_diagonal_operators() {
        let gmax = 50.0;
        let beta = 1.0 / (2.0 * gmax);
        let coefficients = chebyshev_coefficients(beta, gmax, CHEBYSHEV_DEGREE);
        let eigenvalues = [0.0, 3.0, 17.5, 50.0];
        let w = eigenvalues.iter().map(|_| vec![1.0, -2.0]).collect::<Vec<_>>();
        let result = clenshaw(&coefficients, gmax, &w, |v| {
            v.iter().zip(eigenvalues).map(|(v, g)| v.iter().map(|x| g * x).collect()).collect()
        });
        for (out, g) in result.iter().zip(eigenvalues) {
            let expected = correction(beta, g);
            assert!((out[0] - expected).abs() < 1e-14 && (out[1] + 2.0 * expected).abs() < 1e-14);
        }
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

    fn sampled_covariance(
        t: &[DCRTPolyMatrix],
        n: usize,
        widths: &PerturbationWidths,
    ) -> Vec<Vec<f64>> {
        let columns = 20_000;
        let fft = NegacyclicFft::new(n);
        let digits = t[0].col_size();
        let zero = vec![vec![0.0; 2 * n]; digits * columns];
        let rows = t.iter().map(|chunk| chunk.row_size()).sum::<usize>() * n;
        let mut samples = vec![vec![0.0; rows]; columns];
        correlated_centers(t, &fft, widths, &zero, columns, &[3u8; 32], |_, values| {
            for (row, column, centers) in values {
                for (k, center) in centers.into_iter().enumerate() {
                    samples[column][row * n + k] = center;
                }
            }
        });
        (0..rows)
            .map(|i| {
                (0..rows)
                    .map(|j| samples.iter().map(|y| y[i] * y[j]).sum::<f64>() / columns as f64)
                    .collect()
            })
            .collect()
    }

    #[test]
    #[serial_test::serial]
    fn correlated_part_has_the_target_covariance_with_the_ring_adjoint() {
        let n = 4;
        let params = DCRTPolyParams::new(n as u32, 1, 20, 10, None, None);
        let (t, _) = trapgen(&params, 3, 2.0);
        let m = embedding(&t, n);
        // s_1(T)² by power iteration on M Mᵀ.
        let gram = (0..m.len())
            .map(|i| {
                (0..m.len()).map(|j| m[i].iter().zip(&m[j]).map(|(x, y)| x * y).sum()).collect()
            })
            .collect::<Vec<Vec<f64>>>();
        let mut v = vec![1.0; m.len()];
        let mut lambda = 0.0;
        for _ in 0..500 {
            let next = gram
                .iter()
                .map(|row| row.iter().zip(&v).map(|(g, x)| g * x).sum())
                .collect::<Vec<f64>>();
            lambda = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            v = next.iter().map(|x| x / lambda).collect();
        }
        let widths = PerturbationWidths::new(1000.0, 1.0, 1.0, 1.1 * lambda.sqrt());
        let covariance = sampled_covariance(&t, n, &widths);
        for i in 0..m.len() {
            for j in 0..m.len() {
                let expected =
                    widths.a * (if i == j { 1.0 } else { 0.0 } - widths.beta * gram[i][j]);
                assert!(
                    (covariance[i][j] - expected).abs() < 0.06 * widths.a,
                    "covariance ({i}, {j}) = {}, expected {expected}",
                    covariance[i][j]
                );
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn correlated_part_of_a_zero_trapdoor_is_isotropic() {
        let n = 4;
        let params = DCRTPolyParams::new(n as u32, 1, 20, 10, None, None);
        let t = vec![DCRTPolyMatrix::zero(&params, 3, params.modulus_digits())];
        let widths = PerturbationWidths::new(1000.0, 1.0, 1.0, 5.0);
        let covariance = sampled_covariance(&t, n, &widths);
        for (i, row) in covariance.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = if i == j { widths.a } else { 0.0 };
                assert!((value - expected).abs() < 0.06 * widths.a);
            }
        }
    }

    #[test]
    fn rounding_and_normal_samplers_have_their_moments() {
        let mut rng = StdRng::seed_from_u64(7);
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
