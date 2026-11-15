//! Preimages under a computational (MP12) trapdoor, as the CPU backend samples them: the GM18
//! perturbation sampler (eprint 2017/844) as OpenFHE implements it (`SampleMat`,
//! `ZSampleSigma2x2`, `ZSampleF` over `Field2n`), the arbitrary-base gadget sampler per CRT tower,
//! and the combination `x = p + [R; E; I] z`.
//!
//! For a trapdoor of `d` rows and `D` digits per row (`m = d D` columns of `R` and `E`), public
//! matrix `A = [ā | I | G - (ā R + E)]` of `W = 2 d + m` columns, preimage width `s`, and gadget
//! width `c = (b + 1) σ`, one candidate for a target `u` (`d x C`) is:
//!
//! 1. `p_2`, `m x C` integers of the discrete Gaussian of width `sqrt(s² - c²)`;
//! 2. per column, `p_1` (`2 d` ring elements) from `SampleMat` of the covariance `Σ = s² I - c² [R;
//!    E][R; E]ᵀ` (the plain transpose, as OpenFHE and the CPU backend form it) and center `-c² /
//!    (s² - c²) [R; E] p_2`;
//! 3. per tower `q_l`, the gadget preimage `z` of the syndrome `u - A [p_1; p_2] mod q_l`, `k`
//!    digits per coefficient (`D = L k`), digit `l k + t` of row `i` in row `i D + l k + t`;
//! 4. `x = [p_1 + [R; E] z; p_2 + z]`, written modulo every tower.
//!
//! A statistical trapdoor (one row, `T` of `m̄` rows in place of `[R; E]`, `A = [ā | g - ā T]`)
//! draws `p_1` by Peikert's convolution sampler as the CPU backend does: the Cholesky factors
//! `L_j` of `Σ'_j = (s² - σ²) I - α T(ω_j) T(ω_j)*` (`α = c² s² / (s² - c²)`, the conjugate
//! transpose) at the points `j < n / 2`, the continuous `y = F^-1(L q̂)`, and the rounding of
//! `-c² / (s² - c²) T p_2 + y` to the integers by `D_{Z, ·, σ}`; [`trapgen`] draws `T` and checks
//! that every `Σ'_j` is positive definite.
//!
//! Field elements are negacyclic polynomials of `m` real coefficients, in coefficient form or as
//! their values at `ζ^(2 j + 1)`, `ζ = e^(iπ / m)` (OpenFHE's evaluation order), stored as
//! interleaved real and imaginary parts. The transpose of an evaluated real polynomial is its
//! conjugate. Every buffer comes from the caller's scratch in stack order ([`Arena`]), and
//! [`scratch_words`] is its exact high-water mark.

use crate::gauss::{MAX_DIGITS, Stream, ceil, gadget_sample, karney, sqrt};

/// The scratch, handed out in stack order: a caller records `used` and restores it.
pub struct Arena {
    base: *mut u64,
    used: usize,
    peak: usize,
}

impl Arena {
    /// The arena over the scratch at `base`.
    ///
    /// # Safety
    ///
    /// `base` must point to at least the words the draw takes ([`scratch_words`]).
    pub unsafe fn new(base: *mut u64) -> Self {
        Self { base, used: 0, peak: 0 }
    }

    /// The most words in use at once so far.
    pub fn peak(&self) -> usize {
        self.peak
    }

    fn take(&mut self, count: usize) -> *mut u64 {
        let start = self.used;
        self.used += count;
        if self.used > self.peak {
            self.peak = self.used;
        }
        // SAFETY: within the scratch by the contract of `new`.
        unsafe { self.base.add(start) }
    }

    fn floats<'b>(&mut self, count: usize) -> &'b mut [f64] {
        // SAFETY: a fresh range of the scratch; words and binary64 share size and alignment.
        unsafe { core::slice::from_raw_parts_mut(self.take(count) as *mut f64, count) }
    }

    fn ints<'b>(&mut self, count: usize) -> &'b mut [i64] {
        // SAFETY: as `floats`.
        unsafe { core::slice::from_raw_parts_mut(self.take(count) as *mut i64, count) }
    }
}

/// The negacyclic transforms of sizes dividing `n`, from the table `cos(π j / n)`, `j <= n`.
struct Fft<'a> {
    cos: &'a [f64],
    n: usize,
}

impl Fft<'_> {
    /// Fills `cos[j] = cos(π j / n)` for `j <= n` (`n` a power of two, at least 2) by halving
    /// angles: `cos(x / 2) = sqrt((1 + cos x) / 2)`, and between known neighbours
    /// `cos y = (cos(y - h) + cos(y + h)) / (2 cos h)`.
    fn table(cos: &mut [f64], n: usize) {
        cos[0] = 1.0;
        cos[n] = -1.0;
        cos[n / 2] = 0.0;
        let mut step = n / 2;
        while step >= 2 {
            let half = step / 2;
            let ch = sqrt((1.0 + cos[step]) * 0.5);
            cos[half] = ch;
            let mut j = half + step;
            while j < n {
                cos[j] = (cos[j - half] + cos[j + half]) / (2.0 * ch);
                j += step;
            }
            step = half;
        }
    }

    /// `(cos, sin)` of `π idx / n` for `idx <= n`.
    #[inline(always)]
    fn root(&self, idx: usize) -> (f64, f64) {
        let half = self.n / 2;
        let sin = if idx <= half { self.cos[half - idx] } else { self.cos[idx - half] };
        (self.cos[idx], sin)
    }

    /// The cyclic DFT of `m` complex values in place, by `e^(±2πi jk / m)`.
    fn transform(&self, x: &mut [f64], m: usize, inverse: bool) {
        let mut j = 0;
        let mut i = 1;
        while i < m {
            let mut bit = m >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                x.swap(2 * i, 2 * j);
                x.swap(2 * i + 1, 2 * j + 1);
            }
            i += 1;
        }
        let mut len = 2;
        while len <= m {
            let step = 2 * self.n / len;
            let half = len / 2;
            let mut start = 0;
            while start < m {
                let mut t = 0;
                while t < half {
                    let (c, s) = self.root(t * step);
                    let s = if inverse { -s } else { s };
                    let (a, b) = (start + t, start + t + half);
                    let (vr, vi) =
                        (x[2 * b] * c - x[2 * b + 1] * s, x[2 * b] * s + x[2 * b + 1] * c);
                    let (ur, ui) = (x[2 * a], x[2 * a + 1]);
                    x[2 * a] = ur + vr;
                    x[2 * a + 1] = ui + vi;
                    x[2 * b] = ur - vr;
                    x[2 * b + 1] = ui - vi;
                    t += 1;
                }
                start += len;
            }
            len <<= 1;
        }
    }

    /// The values at `ζ^(2 j + 1)` of the `m` real coefficients `a`, into `out` (`2 m` reals).
    fn forward(&self, a: &[f64], out: &mut [f64], m: usize) {
        let scale = self.n / m;
        let mut j = 0;
        while j < m {
            let (c, s) = self.root(j * scale);
            out[2 * j] = a[j] * c;
            out[2 * j + 1] = a[j] * s;
            j += 1;
        }
        self.transform(out, m, false);
    }

    /// The real coefficients of the values `x` (`2 m` reals, overwritten), into `out`.
    fn inverse(&self, x: &mut [f64], out: &mut [f64], m: usize) {
        self.transform(x, m, true);
        let scale = self.n / m;
        let inv = 1.0 / m as f64;
        let mut j = 0;
        while j < m {
            let (c, s) = self.root(j * scale);
            out[j] = (x[2 * j] * c + x[2 * j + 1] * s) * inv;
            j += 1;
        }
    }
}

/// `x[j] *= y[j]` slotwise.
#[inline(always)]
fn mul_assign(x: &mut [f64], y: &[f64], m: usize) {
    let mut j = 0;
    while j < m {
        let (a, b, c, d) = (x[2 * j], x[2 * j + 1], y[2 * j], y[2 * j + 1]);
        x[2 * j] = a * c - b * d;
        x[2 * j + 1] = a * d + b * c;
        j += 1;
    }
}

/// `acc[j] += x[j] y[j]` slotwise.
#[inline(always)]
fn mul_add(acc: &mut [f64], x: &[f64], y: &[f64], m: usize) {
    let mut j = 0;
    while j < m {
        let (a, b, c, d) = (x[2 * j], x[2 * j + 1], y[2 * j], y[2 * j + 1]);
        acc[2 * j] += a * c - b * d;
        acc[2 * j + 1] += a * d + b * c;
        j += 1;
    }
}

/// The words [`zsample_f`] takes at size `m` at most at once.
fn zsample_words(m: usize) -> usize {
    if m == 1 { 0 } else { 9 * (m / 2) + sigma2x2_words(m / 2) }
}

/// The words [`sigma2x2`] takes at size `h` at most at once.
fn sigma2x2_words(h: usize) -> usize {
    11 * h + zsample_words(h)
}

/// A sample of the discrete Gaussian of covariance `f` (real coefficients, `m` of them) centered
/// at `c`, into `out`: OpenFHE's `ZSampleF` (Algorithm 4 of eprint 2017/844).
fn zsample_f(
    fft: &Fft<'_>,
    g: &mut Stream,
    arena: &mut Arena,
    f: &[f64],
    c: &[f64],
    out: &mut [i64],
    m: usize,
) {
    if m == 1 {
        if !(f[0] > 0.0) {
            g.fail();
            out[0] = 0;
            return;
        }
        out[0] = karney(g, c[0], sqrt(f[0]));
        return;
    }
    let h = m / 2;
    let mark = arena.used;
    let (a, b, tmp) = (arena.floats(2 * h), arena.floats(2 * h), arena.floats(h));
    let (c0, c1) = (arena.floats(h), arena.floats(h));
    let (q1, q2) = (arena.ints(h), arena.ints(h));
    let mut i = 0;
    while i < h {
        tmp[i] = f[2 * i];
        i += 1;
    }
    fft.forward(tmp, a, h);
    let mut i = 0;
    while i < h {
        tmp[i] = f[2 * i + 1];
        c0[i] = c[2 * i];
        c1[i] = c[2 * i + 1];
        i += 1;
    }
    fft.forward(tmp, b, h);
    sigma2x2(fft, g, arena, a, b, a, c0, c1, q1, q2, h);
    let mut i = 0;
    while i < h {
        out[2 * i] = q1[i];
        out[2 * i + 1] = q2[i];
        i += 1;
    }
    arena.used = mark;
}

/// A sample of the discrete Gaussian of covariance `[[a, b], [bᵀ, d]]` (evaluated, size `h`)
/// centered at `(c0, c1)` (coefficients), into `(q1, q2)`: OpenFHE's `ZSampleSigma2x2`.
#[allow(clippy::too_many_arguments)]
fn sigma2x2(
    fft: &Fft<'_>,
    g: &mut Stream,
    arena: &mut Arena,
    a: &[f64],
    b: &[f64],
    d: &[f64],
    c0: &[f64],
    c1: &[f64],
    q1: &mut [i64],
    q2: &mut [i64],
    h: usize,
) {
    let mark = arena.used;
    let (dcoef, tmp, bdinv) = (arena.floats(h), arena.floats(2 * h), arena.floats(2 * h));
    let (fe, fcoef, diff) = (arena.floats(2 * h), arena.floats(h), arena.floats(h));
    let (prod, c1n) = (arena.floats(h), arena.floats(h));
    tmp.copy_from_slice(&d[..2 * h]);
    fft.inverse(tmp, dcoef, h);
    // `b d^-1`, and `f = a - b d^-1 b̄`.
    let mut j = 0;
    while j < h {
        let (dr, di) = (d[2 * j], d[2 * j + 1]);
        let norm = dr * dr + di * di;
        let (ir, ii) = (dr / norm, -di / norm);
        let (br, bi) = (b[2 * j], b[2 * j + 1]);
        let (xr, xi) = (br * ir - bi * ii, br * ii + bi * ir);
        bdinv[2 * j] = xr;
        bdinv[2 * j + 1] = xi;
        // (xr + i xi)(br - i bi)
        fe[2 * j] = a[2 * j] - (xr * br + xi * bi);
        fe[2 * j + 1] = a[2 * j + 1] - (xi * br - xr * bi);
        j += 1;
    }
    fft.inverse(fe, fcoef, h);
    zsample_f(fft, g, arena, dcoef, c1, q2, h);
    let mut i = 0;
    while i < h {
        diff[i] = q2[i] as f64 - c1[i];
        i += 1;
    }
    fft.forward(diff, tmp, h);
    mul_assign(tmp, bdinv, h);
    fft.inverse(tmp, prod, h);
    let mut i = 0;
    while i < h {
        c1n[i] = c0[i] + prod[i];
        i += 1;
    }
    zsample_f(fft, g, arena, fcoef, c1n, q1, h);
    arena.used = mark;
}

/// The words [`sample_mat`] takes for a `dim x dim` covariance of size-`n` elements at most at
/// once.
fn sample_mat_words(dim: usize, n: usize) -> usize {
    match dim {
        1 => 3 * n + zsample_words(n),
        2 => sigma2x2_words(n),
        _ => {
            let a = dim.div_ceil(2);
            let e = dim - a;
            let rest = e * e * 2 * n +
                4 * e * e +
                a * e * 2 * n +
                a * a * 2 * n +
                e * 2 * n +
                n +
                a * n +
                2 * n;
            let tail = rest + sample_mat_words(a, n);
            e * e * 2 * n +
                if tail > sample_mat_words(e, n) { tail } else { sample_mat_words(e, n) }
        }
    }
}

/// A sample of the discrete Gaussian of covariance `sigma` (`dim x dim` evaluated elements of size
/// `n`, row-major, entry `(i, j)` at `2 n (i dim + j)`) centered at `c` (`dim` coefficient
/// vectors), into `out`: OpenFHE's `SampleMat`, splitting at `ceil(dim / 2)`, sampling the lower
/// block, and the upper block from its Schur complement `A - B D^-1 Bᵀ` (plain transpose) and the
/// shifted center. The inverse of `D` is taken slot by slot.
#[allow(clippy::too_many_arguments)]
fn sample_mat(
    fft: &Fft<'_>,
    g: &mut Stream,
    arena: &mut Arena,
    sigma: &[f64],
    dim: usize,
    c: &[f64],
    out: &mut [i64],
    n: usize,
) {
    let w = 2 * n;
    let entry = |i: usize, j: usize| &sigma[w * (i * dim + j)..w * (i * dim + j + 1)];
    if dim == 1 {
        let mark = arena.used;
        let (fc, tmp) = (arena.floats(n), arena.floats(w));
        tmp.copy_from_slice(entry(0, 0));
        fft.inverse(tmp, fc, n);
        zsample_f(fft, g, arena, fc, c, out, n);
        arena.used = mark;
        return;
    }
    if dim == 2 {
        let (o0, o1) = out.split_at_mut(n);
        sigma2x2(fft, g, arena, entry(0, 0), entry(0, 1), entry(1, 1), &c[..n], &c[n..], o0, o1, n);
        return;
    }
    let a = dim.div_ceil(2);
    let e = dim - a;
    let mark = arena.used;
    let sdd = arena.floats(e * e * w);
    let mut i = 0;
    while i < e {
        let mut j = 0;
        while j < e {
            sdd[w * (i * e + j)..w * (i * e + j + 1)].copy_from_slice(entry(a + i, a + j));
            j += 1;
        }
        i += 1;
    }
    let (upper, lower) = out.split_at_mut(a * n);
    sample_mat(fft, g, arena, sdd, e, &c[a * n..], lower, n);
    // `D^-1` slot by slot, by Gauss-Jordan elimination with partial pivoting on `[D | I]`.
    let dinv = arena.floats(e * e * w);
    let work = arena.floats(4 * e * e);
    let width = 2 * e;
    let mut slot = 0;
    while slot < n {
        let mut i = 0;
        while i < e {
            let mut j = 0;
            while j < width {
                let (re, im) = if j < e {
                    let x = &sdd[w * (i * e + j)..];
                    (x[2 * slot], x[2 * slot + 1])
                } else {
                    (if j - e == i { 1.0 } else { 0.0 }, 0.0)
                };
                work[2 * (i * width + j)] = re;
                work[2 * (i * width + j) + 1] = im;
                j += 1;
            }
            i += 1;
        }
        let mut col = 0;
        while col < e {
            let mut pivot = col;
            let mut best = -1.0;
            let mut r = col;
            while r < e {
                let (re, im) = (work[2 * (r * width + col)], work[2 * (r * width + col) + 1]);
                let norm = re * re + im * im;
                if norm > best {
                    best = norm;
                    pivot = r;
                }
                r += 1;
            }
            if !(best > 0.0) {
                g.fail();
                arena.used = mark;
                return;
            }
            if pivot != col {
                let mut j = 0;
                while j < 2 * width {
                    work.swap(2 * pivot * width + j, 2 * col * width + j);
                    j += 1;
                }
            }
            let (pr, pi) = (work[2 * (col * width + col)], work[2 * (col * width + col) + 1]);
            let (ir, ii) = (pr / best, -pi / best);
            let mut j = 0;
            while j < width {
                let (xr, xi) = (work[2 * (col * width + j)], work[2 * (col * width + j) + 1]);
                work[2 * (col * width + j)] = xr * ir - xi * ii;
                work[2 * (col * width + j) + 1] = xr * ii + xi * ir;
                j += 1;
            }
            let mut r = 0;
            while r < e {
                if r != col {
                    let (fr, fi) = (work[2 * (r * width + col)], work[2 * (r * width + col) + 1]);
                    let mut j = 0;
                    while j < width {
                        let (xr, xi) =
                            (work[2 * (col * width + j)], work[2 * (col * width + j) + 1]);
                        work[2 * (r * width + j)] -= fr * xr - fi * xi;
                        work[2 * (r * width + j) + 1] -= fr * xi + fi * xr;
                        j += 1;
                    }
                }
                r += 1;
            }
            col += 1;
        }
        let mut i = 0;
        while i < e {
            let mut j = 0;
            while j < e {
                let x = &mut dinv[w * (i * e + j)..];
                x[2 * slot] = work[2 * (i * width + e + j)];
                x[2 * slot + 1] = work[2 * (i * width + e + j) + 1];
                j += 1;
            }
            i += 1;
        }
        slot += 1;
    }
    // `B D^-1`, the Schur complement, and the shifted center.
    let bd = arena.floats(a * e * w);
    bd.fill(0.0);
    let mut i = 0;
    while i < a {
        let mut j = 0;
        while j < e {
            let mut k = 0;
            while k < e {
                mul_add(
                    &mut bd[w * (i * e + j)..w * (i * e + j + 1)],
                    entry(i, a + k),
                    &dinv[w * (k * e + j)..w * (k * e + j + 1)],
                    n,
                );
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    let s = arena.floats(a * a * w);
    let mut i = 0;
    while i < a {
        let mut j = 0;
        while j < a {
            let target = &mut s[w * (i * a + j)..w * (i * a + j + 1)];
            target.copy_from_slice(entry(i, j));
            let mut k = 0;
            while k < e {
                let (x, y) = (&bd[w * (i * e + k)..], entry(j, a + k));
                let mut t = 0;
                while t < n {
                    let (p, q, r, u) = (x[2 * t], x[2 * t + 1], y[2 * t], y[2 * t + 1]);
                    target[2 * t] -= p * r - q * u;
                    target[2 * t + 1] -= p * u + q * r;
                    t += 1;
                }
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    let (diff, real) = (arena.floats(e * w), arena.floats(n));
    let mut j = 0;
    while j < e {
        let mut t = 0;
        while t < n {
            real[t] = lower[j * n + t] as f64 - c[(a + j) * n + t];
            t += 1;
        }
        fft.forward(real, &mut diff[w * j..w * (j + 1)], n);
        j += 1;
    }
    let (cnew, acc) = (arena.floats(a * n), arena.floats(w));
    let mut i = 0;
    while i < a {
        acc.fill(0.0);
        let mut j = 0;
        while j < e {
            mul_add(acc, &bd[w * (i * e + j)..w * (i * e + j + 1)], &diff[w * j..w * (j + 1)], n);
            j += 1;
        }
        let row = &mut cnew[i * n..(i + 1) * n];
        fft.inverse(acc, row, n);
        let mut t = 0;
        while t < n {
            row[t] += c[i * n + t];
            t += 1;
        }
        i += 1;
    }
    sample_mat(fft, g, arena, s, a, cnew, upper, n);
    arena.used = mark;
}

/// The shape of one draw.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// The ring dimension (a power of two, at least 2).
    pub n: usize,
    /// The CRT towers.
    pub towers: usize,
    /// The rows of the public matrix (1 for a statistical trapdoor).
    pub rows: usize,
    /// The gadget digits per row (`towers * tower_digits`).
    pub digits: usize,
    /// The gadget digits per tower.
    pub tower_digits: usize,
    /// The target's columns.
    pub columns: usize,
    /// The rows `m̄` of `T` of a statistical trapdoor, or 0 for a computational one.
    pub statistical: usize,
}

impl Shape {
    /// The rows of the secret block: `2 rows` (`[R; E]`) or `m̄` (`T`).
    pub fn secret_rows(&self) -> usize {
        if self.statistical > 0 { self.statistical } else { 2 * self.rows }
    }

    /// The columns of the secret block, `rows * digits`.
    pub fn trapdoor_columns(&self) -> usize {
        self.rows * self.digits
    }

    /// The columns of the public matrix (the preimage's rows): the secret rows and the secret
    /// columns.
    pub fn public_columns(&self) -> usize {
        self.secret_rows() + self.trapdoor_columns()
    }

    /// The words of one limb of the trapdoor value: the public matrix and the secret block.
    pub fn trapdoor_limb_words(&self) -> usize {
        self.n * (self.rows * self.public_columns() + self.secret_rows() * self.trapdoor_columns())
    }

    /// The words of one limb of the target.
    pub fn target_limb_words(&self) -> usize {
        self.n * self.rows * self.columns
    }

    /// The words of one limb of the preimage.
    pub fn preimage_limb_words(&self) -> usize {
        self.n * self.public_columns() * self.columns
    }

    /// The Gaussian samples one draw takes: `p_2`, `p_1` (twice for a statistical trapdoor: the
    /// continuous part and the rounding), and per tower and coefficient of the syndrome `2 k`
    /// (the gadget sampler's continuous and discrete parts).
    pub fn gaussian_samples(&self) -> usize {
        let p1 = if self.statistical > 0 { 2 } else { 1 } * self.secret_rows();
        self.n * self.columns * (self.trapdoor_columns() + p1 + 2 * self.rows * self.digits)
    }
}

/// The scratch words of one draw of shape `shape`, exactly the high-water mark of its [`Arena`].
pub fn scratch_words(shape: &Shape) -> usize {
    let (n, dim, m, c) = (shape.n, shape.secret_rows(), shape.trapdoor_columns(), shape.columns);
    let fixed = (n + 1) + dim * dim * 2 * n + dim * c * 2 * n + m * c * n;
    let per_t = dim * 2 * n + n + 2 * n;
    let per_column = dim * n + dim * n + m * n;
    let sampler = if shape.statistical > 0 { dim * n + 3 * n } else { sample_mat_words(dim, n) };
    fixed + per_t + per_column + sampler
}

/// The smallest accepted Cholesky pivot of a statistical trapdoor's perturbation covariance,
/// relative to `s²` (as the CPU backend's `PIVOT_FLOOR`).
pub const PIVOT_FLOOR: f64 = 1.0 / (1u64 << 20) as f64;

/// `acc[j] += x[j] conj(y[j])` slotwise.
#[inline(always)]
fn mul_conj_add(acc: &mut [f64], x: &[f64], y: &[f64], m: usize) {
    let mut j = 0;
    while j < m {
        let (a, b, c, d) = (x[2 * j], x[2 * j + 1], y[2 * j], y[2 * j + 1]);
        acc[2 * j] += a * c + b * d;
        acc[2 * j + 1] += b * c - a * d;
        j += 1;
    }
}

/// Replaces `sigma` (`dim x dim` evaluated elements, entry `(i, k)` at `2 n (i dim + k)`, holding
/// `T(ω_j) T(ω_j)*`) by the Cholesky factor `L_j` of `variance I - alpha sigma` at the points
/// `j < n / 2`, in its lower triangle: false when a pivot is below `pivot_floor` (or NaN), as the
/// CPU backend's statistical factorization.
fn factor(sigma: &mut [f64], dim: usize, n: usize, variance: f64, alpha: f64, floor: f64) -> bool {
    let w = 2 * n;
    let at = |i: usize, k: usize, j: usize| w * (i * dim + k) + 2 * j;
    let mut j = 0;
    while j < n / 2 {
        let mut i = 0;
        while i < dim {
            let mut p = 0;
            while p <= i {
                let x = at(i, p, j);
                let diagonal = if i == p { variance } else { 0.0 };
                sigma[x] = diagonal - alpha * sigma[x];
                sigma[x + 1] = -alpha * sigma[x + 1];
                p += 1;
            }
            i += 1;
        }
        let mut k = 0;
        while k < dim {
            let mut pivot = sigma[at(k, k, j)];
            let mut p = 0;
            while p < k {
                let (a, b) = (sigma[at(k, p, j)], sigma[at(k, p, j) + 1]);
                pivot -= a * a + b * b;
                p += 1;
            }
            if !(pivot >= floor) {
                return false;
            }
            let diagonal = sqrt(pivot);
            sigma[at(k, k, j)] = diagonal;
            sigma[at(k, k, j) + 1] = 0.0;
            let mut i = k + 1;
            while i < dim {
                let (mut re, mut im) = (sigma[at(i, k, j)], sigma[at(i, k, j) + 1]);
                let mut p = 0;
                while p < k {
                    let (a, b) = (sigma[at(i, p, j)], sigma[at(i, p, j) + 1]);
                    let (c, d) = (sigma[at(k, p, j)], sigma[at(k, p, j) + 1]);
                    re -= a * c + b * d;
                    im -= b * c - a * d;
                    p += 1;
                }
                sigma[at(i, k, j)] = re / diagonal;
                sigma[at(i, k, j) + 1] = im / diagonal;
                i += 1;
            }
            k += 1;
        }
        j += 1;
    }
    true
}

/// `p_1` of a statistical trapdoor for one column, into `out`: the randomized rounding, a sample
/// of `D_{Z, μ + y, r}` per coefficient, of the centers `center` (`μ`, `dim` coefficient vectors)
/// plus the continuous `y = F^-1(L q̂)`, for `q̂_j` (`j < n / 2`) circular complex Gaussians with
/// `E|q̂_j|² = n` and the conjugate values at the partner points `n - 1 - j`, as the CPU backend
/// draws it (Peikert's convolution sampler).
#[allow(clippy::too_many_arguments)]
fn sample_statistical(
    fft: &Fft<'_>,
    g: &mut Stream,
    arena: &mut Arena,
    l: &[f64],
    dim: usize,
    center: &[f64],
    out: &mut [i64],
    n: usize,
    rounding: f64,
) {
    let (w, half) = (2 * n, n / 2);
    let mark = arena.used;
    let (q, spectrum, y) = (arena.floats(dim * n), arena.floats(w), arena.floats(n));
    let scale = sqrt(n as f64 / 2.0);
    let mut i = 0;
    while i < dim * n {
        q[i] = crate::gauss::normal(g, scale);
        i += 1;
    }
    let mut row = 0;
    while row < dim {
        let mut j = 0;
        while j < half {
            let (mut re, mut im) = (0.0, 0.0);
            let mut k = 0;
            while k <= row {
                let x = w * (row * dim + k) + 2 * j;
                let (a, b) = (l[x], l[x + 1]);
                let (c, d) = (q[k * n + 2 * j], q[k * n + 2 * j + 1]);
                re += a * c - b * d;
                im += a * d + b * c;
                k += 1;
            }
            spectrum[2 * j] = re;
            spectrum[2 * j + 1] = im;
            spectrum[2 * (n - 1 - j)] = re;
            spectrum[2 * (n - 1 - j) + 1] = -im;
            j += 1;
        }
        fft.inverse(spectrum, y, n);
        let mut a = 0;
        while a < n {
            out[row * n + a] = karney(g, center[row * n + a] + y[a], rounding);
            a += 1;
        }
        row += 1;
    }
    arena.used = mark;
}

/// The centered value of a residue modulo `q`.
#[inline(always)]
fn centered(v: u64, q: u64) -> i64 {
    if v > q / 2 { v as i64 - q as i64 } else { v as i64 }
}

/// `x mod q` of a signed 128-bit integer.
#[inline(always)]
fn signed_mod(x: i128, q: u64) -> u64 {
    let magnitude = x.unsigned_abs();
    let r = crate::reduce((magnitude >> 64) as u64, magnitude as u64, q);
    if x < 0 && r != 0 { q - r } else { r }
}

/// The nearest integer to `x` (`|x| < 2^62`), halves away from zero.
#[inline(always)]
fn round(x: f64) -> f64 {
    if x >= 0.0 { -ceil(-(x + 0.5)) } else { ceil(x - 0.5) }
}

/// Adds coefficient `c` of the negacyclic product of the residues `e` and the signed integers
/// `p` (`n` each) to `positive` and `negative` (its positive and negative terms).
///
/// # Safety
///
/// `e` and `p` must hold `n` values, `c < n`.
#[inline(always)]
unsafe fn residue_times_signed(
    e: *const u64,
    p: *const i64,
    n: usize,
    c: usize,
    positive: &mut u128,
    negative: &mut u128,
) {
    unsafe {
        let (mut plus, mut minus) = (0u128, 0u128);
        let mut a = 0;
        while a <= c {
            let v = *p.add(c - a);
            let term = *e.add(a) as u128 * v.unsigned_abs() as u128;
            if v < 0 {
                minus += term
            } else {
                plus += term
            }
            a += 1;
        }
        while a < n {
            let v = *p.add(n + c - a);
            let term = *e.add(a) as u128 * v.unsigned_abs() as u128;
            if v < 0 {
                plus += term
            } else {
                minus += term
            }
            a += 1;
        }
        *positive += plus;
        *negative += minus;
    }
}

/// Coefficient `c` of the negacyclic product of the residues `e` modulo `q`, centered, and the
/// signed integers `z` (`n` each).
///
/// # Safety
///
/// `e` and `z` must hold `n` values, `c < n`.
#[inline(always)]
unsafe fn centered_times_signed(e: *const u64, q: u64, z: *const i64, n: usize, c: usize) -> i128 {
    unsafe {
        let mut acc = 0i128;
        let mut a = 0;
        while a <= c {
            acc += centered(*e.add(a), q) as i128 * *z.add(c - a) as i128;
            a += 1;
        }
        while a < n {
            acc -= centered(*e.add(a), q) as i128 * *z.add(n + c - a) as i128;
            a += 1;
        }
        acc
    }
}

/// One candidate preimage, written to `out` (each limb `preimage_limb_words` words): whether it
/// was drawn within the stream's budget, with every coefficient's absolute value at most `bound`.
///
/// # Safety
///
/// The pointers must hold the trapdoor and target families and the `towers` moduli (below `2^62`)
/// of `shape`, `out` the preimage family, and `arena` [`scratch_words`] free words; `shape.n` is a
/// power of two at least 2, `shape.digits == shape.towers * shape.tower_digits`, the tower digits
/// at most [`MAX_DIGITS`], and `base^tower_digits` at least every modulus.
#[allow(clippy::too_many_arguments)]
pub unsafe fn draw(
    shape: &Shape,
    g: &mut Stream,
    base: u64,
    sigma: f64,
    width: f64,
    bound: u64,
    trapdoor: *const u64,
    target: *const u64,
    moduli: *const u64,
    out: *mut u64,
    arena: &mut Arena,
) -> bool {
    unsafe {
        let (n, towers, d, cols) = (shape.n, shape.towers, shape.rows, shape.columns);
        let (m, wpub, k) = (shape.trapdoor_columns(), shape.public_columns(), shape.tower_digits);
        let (digits, w, statistical) = (shape.digits, 2 * n, shape.statistical > 0);
        let cos = arena.floats(n + 1);
        Fft::table(cos, n);
        let fft = Fft { cos, n };
        let c = (base as f64 + 1.0) * sigma;
        let (s2, c2) = (width * width, c * c);
        let scale = -c2 / (s2 - c2);
        let large = sqrt(s2 - c2);
        let q0 = *moduli;
        // Entries of the secret block (`[R; E]` or `T`) of limb 0, centered.
        let secret_at = |i: usize, t: usize, a: usize| {
            centered(*trapdoor.add(n * (d * wpub + i * m + t) + a), q0)
        };
        // The covariance, the center accumulators `S p_2` (evaluated), and `p_2`.
        let dim = shape.secret_rows();
        let sigma_m = arena.floats(dim * dim * w);
        sigma_m.fill(0.0);
        let acc = arena.floats(dim * cols * w);
        acc.fill(0.0);
        let p2 = arena.ints(m * cols * n);
        let mut i = 0;
        while i < m * cols * n {
            p2[i] = karney(g, 0.0, large);
            i += 1;
        }
        let hats = arena.floats(dim * w);
        let (real, phat) = (arena.floats(n), arena.floats(w));
        let mut t = 0;
        while t < m {
            let mut i = 0;
            while i < dim {
                let mut a = 0;
                while a < n {
                    real[a] = secret_at(i, t, a) as f64;
                    a += 1;
                }
                fft.forward(real, &mut hats[w * i..w * (i + 1)], n);
                i += 1;
            }
            let mut i = 0;
            while i < dim {
                let mut j = 0;
                while j < dim {
                    let (x, y) = (&hats[w * i..w * (i + 1)], &hats[w * j..w * (j + 1)]);
                    let target = &mut sigma_m[w * (i * dim + j)..w * (i * dim + j + 1)];
                    if statistical {
                        mul_conj_add(target, x, y, n)
                    } else {
                        mul_add(target, x, y, n)
                    }
                    j += 1;
                }
                i += 1;
            }
            let mut col = 0;
            while col < cols {
                let mut a = 0;
                while a < n {
                    real[a] = p2[(t * cols + col) * n + a] as f64;
                    a += 1;
                }
                fft.forward(real, phat, n);
                let mut i = 0;
                while i < dim {
                    mul_add(
                        &mut acc[w * (i * cols + col)..w * (i * cols + col + 1)],
                        &hats[w * i..w * (i + 1)],
                        phat,
                        n,
                    );
                    i += 1;
                }
                col += 1;
            }
            t += 1;
        }
        if statistical {
            // The factors of `Σ' = (s² - r²) I - α T T*`, `α = c² s² / (s² - c²)`, `r = σ`.
            if !factor(sigma_m, dim, n, s2 - sigma * sigma, c2 * s2 / (s2 - c2), PIVOT_FLOOR * s2) {
                g.fail();
                return false;
            }
        } else {
            // `Σ = s² I - c² [R; E][R; E]ᵀ`.
            let mut i = 0;
            while i < dim * dim * w {
                sigma_m[i] *= -c2;
                i += 1;
            }
            let mut i = 0;
            while i < dim {
                let mut slot = 0;
                while slot < n {
                    sigma_m[w * (i * dim + i) + 2 * slot] += s2;
                    slot += 1;
                }
                i += 1;
            }
        }
        let (center, p1, z) = (arena.floats(dim * n), arena.ints(dim * n), arena.ints(m * n));
        let mut ok = true;
        let mut col = 0;
        while col < cols {
            let mut i = 0;
            while i < dim {
                phat.copy_from_slice(&acc[w * (i * cols + col)..w * (i * cols + col + 1)]);
                let row = &mut center[i * n..(i + 1) * n];
                fft.inverse(phat, row, n);
                let mut a = 0;
                while a < n {
                    row[a] = scale * round(row[a]);
                    a += 1;
                }
                i += 1;
            }
            if statistical {
                sample_statistical(&fft, g, arena, sigma_m, dim, center, p1, n, sigma);
            } else {
                sample_mat(&fft, g, arena, sigma_m, dim, center, p1, n);
            }
            let p1_base = p1.as_ptr();
            let p2_base = p2.as_ptr();
            let p_row = |j: usize| {
                if j < dim { p1_base.add(j * n) } else { p2_base.add(((j - dim) * cols + col) * n) }
            };
            // The syndrome per tower, and its gadget preimage.
            let mut l = 0;
            while l < towers {
                let q = *moduli.add(l);
                let limb = trapdoor.add(l * shape.trapdoor_limb_words());
                let u = target.add(l * shape.target_limb_words());
                let mut i = 0;
                while i < d {
                    let mut coefficient = 0;
                    while coefficient < n {
                        let (mut positive, mut negative) = (0u128, 0u128);
                        let mut j = 0;
                        while j < wpub {
                            residue_times_signed(
                                limb.add(n * (i * wpub + j)),
                                p_row(j),
                                n,
                                coefficient,
                                &mut positive,
                                &mut negative,
                            );
                            j += 1;
                        }
                        let ap = crate::sub_mod(
                            crate::reduce((positive >> 64) as u64, positive as u64, q),
                            crate::reduce((negative >> 64) as u64, negative as u64, q),
                            q,
                        );
                        let v =
                            crate::sub_mod(*u.add(n * (i * cols + col) + coefficient) % q, ap, q);
                        let mut digits_out = [0i64; MAX_DIGITS];
                        gadget_sample(g, v, q, base, k, c, &mut digits_out);
                        let mut tau = 0;
                        while tau < k {
                            z[(i * digits + l * k + tau) * n + coefficient] = digits_out[tau];
                            tau += 1;
                        }
                        coefficient += 1;
                    }
                    i += 1;
                }
                l += 1;
            }
            // `x = [p_1 + [R; E] z; p_2 + z]`, checked and written modulo every tower.
            let secret = trapdoor.add(n * d * wpub);
            let mut row = 0;
            while row < wpub {
                let mut coefficient = 0;
                while coefficient < n {
                    let mut x = *p_row(row).add(coefficient) as i128;
                    if row < dim {
                        let mut t = 0;
                        while t < m {
                            x += centered_times_signed(
                                secret.add(n * (row * m + t)),
                                q0,
                                z.as_ptr().add(t * n),
                                n,
                                coefficient,
                            );
                            t += 1;
                        }
                    } else {
                        x += z[(row - dim) * n + coefficient] as i128;
                    }
                    if x.unsigned_abs() > bound as u128 {
                        ok = false;
                    }
                    let mut l = 0;
                    while l < towers {
                        let at =
                            l * shape.preimage_limb_words() + n * (row * cols + col) + coefficient;
                        *out.add(at) = signed_mod(x, *moduli.add(l));
                        l += 1;
                    }
                    coefficient += 1;
                }
                row += 1;
            }
            col += 1;
        }
        ok && !g.exhausted()
    }
}

/// The scratch words of [`trapgen`] for `m̄` rows of `digits` entries in ring dimension `n`.
pub fn trapgen_scratch_words(n: usize, m_bar: usize, digits: usize) -> usize {
    (n + 1) + m_bar * digits * n + m_bar * m_bar * 2 * n + m_bar * 2 * n + n
}

/// Statistical TrapGen's draw of `T` (`m̄ x digits`, entries of the discrete Gaussian of width
/// `sigma` without truncation, by Karney's sampler), written to `out` modulo each of the `towers`
/// moduli (each limb `m̄ digits n` words), and whether the perturbation covariance of preimages
/// of width `width` is positive definite at every point, as the CPU backend checks it (and the
/// stream sufficed). A loop redraws on false, as TrapGen does.
///
/// # Safety
///
/// `moduli` must hold `towers` moduli below `2^62`, `out` the family, and `arena`
/// [`trapgen_scratch_words`] free words; `n` is a power of two at least 2.
#[allow(clippy::too_many_arguments)]
pub unsafe fn trapgen(
    g: &mut Stream,
    n: usize,
    towers: usize,
    m_bar: usize,
    digits: usize,
    base: u64,
    sigma: f64,
    width: f64,
    moduli: *const u64,
    out: *mut u64,
    arena: &mut Arena,
) -> bool {
    unsafe {
        let w = 2 * n;
        let cos = arena.floats(n + 1);
        Fft::table(cos, n);
        let fft = Fft { cos, n };
        let t = arena.ints(m_bar * digits * n);
        let mut i = 0;
        while i < m_bar * digits * n {
            t[i] = karney(g, 0.0, sigma);
            let mut l = 0;
            while l < towers {
                *out.add(l * m_bar * digits * n + i) = signed_mod(t[i] as i128, *moduli.add(l));
                l += 1;
            }
            i += 1;
        }
        let sigma_m = arena.floats(m_bar * m_bar * w);
        sigma_m.fill(0.0);
        let (hats, real) = (arena.floats(m_bar * w), arena.floats(n));
        let mut k = 0;
        while k < digits {
            let mut i = 0;
            while i < m_bar {
                let mut a = 0;
                while a < n {
                    real[a] = t[(i * digits + k) * n + a] as f64;
                    a += 1;
                }
                fft.forward(real, &mut hats[w * i..w * (i + 1)], n);
                i += 1;
            }
            let mut i = 0;
            while i < m_bar {
                let mut p = 0;
                while p <= i {
                    mul_conj_add(
                        &mut sigma_m[w * (i * m_bar + p)..w * (i * m_bar + p + 1)],
                        &hats[w * i..w * (i + 1)],
                        &hats[w * p..w * (p + 1)],
                        n,
                    );
                    p += 1;
                }
                i += 1;
            }
            k += 1;
        }
        let c = (base as f64 + 1.0) * sigma;
        let (s2, c2) = (width * width, c * c);
        let ok =
            factor(sigma_m, m_bar, n, s2 - sigma * sigma, c2 * s2 / (s2 - c2), PIVOT_FLOOR * s2);
        ok && !g.exhausted()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;

    /// The transforms invert each other and multiply negacyclically, at every size up to `n`.
    #[test]
    fn test_fft_is_negacyclic_evaluation() {
        for n in [2usize, 4, 16, 256] {
            let mut cos = vec![0.0; n + 1];
            Fft::table(&mut cos, n);
            for (j, value) in cos.iter().enumerate() {
                let host = (core::f64::consts::PI * j as f64 / n as f64).cos();
                assert!((value - host).abs() < 1e-14, "cos {j}/{n}: {value} vs {host}");
            }
            let fft = Fft { cos: &cos, n };
            let mut m = 1;
            while m <= n {
                let a: vec::Vec<f64> =
                    (0..m).map(|i| (i as f64 * 7.0 + 3.0) % 11.0 - 5.0).collect();
                let b: vec::Vec<f64> = (0..m).map(|i| (i as f64 * 5.0 + 1.0) % 7.0 - 3.0).collect();
                let mut expected = vec![0.0; m];
                for i in 0..m {
                    for j in 0..m {
                        if i + j < m {
                            expected[i + j] += a[i] * b[j];
                        } else {
                            expected[i + j - m] -= a[i] * b[j];
                        }
                    }
                }
                let (mut x, mut y) = (vec![0.0; 2 * m], vec![0.0; 2 * m]);
                fft.forward(&a, &mut x, m);
                fft.forward(&b, &mut y, m);
                mul_assign(&mut x, &y, m);
                let mut product = vec![0.0; m];
                fft.inverse(&mut x, &mut product, m);
                for i in 0..m {
                    assert!((product[i] - expected[i]).abs() < 1e-9, "{m}/{n} at {i}");
                }
                m *= 2;
            }
        }
    }

    /// `zsample_f` draws integers with the covariance it is given: for `f = v` (a constant), each
    /// coefficient has variance `v`; the arena's peak matches `zsample_words`.
    #[test]
    fn test_zsample_f_has_the_requested_variance() {
        let n = 16;
        let mut cos = vec![0.0; n + 1];
        Fft::table(&mut cos, n);
        let fft = Fft { cos: &cos, n };
        let mut scratch = vec![0u64; zsample_words(n)];
        let mut g = Stream::new(&[9, 9]);
        let v = 400.0;
        let mut f = vec![0.0; n];
        f[0] = v;
        let c: vec::Vec<f64> = (0..n).map(|i| i as f64 * 0.37).collect();
        let (mut sum, mut squares, mut count) = (0.0, 0.0, 0.0);
        let mut out = vec![0i64; n];
        let mut peak = 0;
        for _ in 0..4000 {
            let mut arena = unsafe { Arena::new(scratch.as_mut_ptr()) };
            zsample_f(&fft, &mut g, &mut arena, &f, &c, &mut out, n);
            peak = arena.peak();
            for i in 0..n {
                let x = out[i] as f64 - c[i];
                sum += x;
                squares += x * x;
                count += 1.0;
            }
        }
        assert_eq!(peak, zsample_words(n));
        let (mean, var) = (sum / count, squares / count);
        assert!(mean.abs() < 0.5 && (var / v - 1.0).abs() < 0.05, "mean {mean}, variance {var}");
    }

    /// A random instance of `shape` in base `2^base_bits` over `moduli`: the trapdoor value (per
    /// tower `A ‖ R ‖ E` with `A = [ā | I | G - (ā R + E)]`, `R` and `E` uniform on `[-9, 9]`) and
    /// a target of residues of integers uniform modulo the product.
    fn instance(
        rng: &mut impl rand::Rng,
        shape: &Shape,
        base_bits: usize,
        moduli: &[u64],
    ) -> (vec::Vec<u64>, vec::Vec<u64>) {
        let (n, rows) = (shape.n, shape.rows);
        let (m, wpub) = (shape.trapdoor_columns(), shape.public_columns());
        let r: vec::Vec<i128> = (0..rows * m * n).map(|_| rng.random_range(-9..10)).collect();
        let e: vec::Vec<i128> = (0..rows * m * n).map(|_| rng.random_range(-9..10)).collect();
        let a_bar: vec::Vec<u64> = (0..rows * rows * n).map(|_| rng.random::<u64>()).collect();
        let mut trapdoor = vec![0u64; moduli.len() * shape.trapdoor_limb_words()];
        for (l, &q) in moduli.iter().enumerate() {
            let qi = q as i128;
            let limb = &mut trapdoor[l * shape.trapdoor_limb_words()..];
            for i in 0..rows {
                for j in 0..wpub {
                    let mut value = vec![0i128; n];
                    if j < rows {
                        for a in 0..n {
                            value[a] = (a_bar[(i * rows + j) * n + a] % q) as i128;
                        }
                    } else if j < 2 * rows {
                        value[0] = i128::from(j - rows == i);
                    } else {
                        let t = j - 2 * rows;
                        // `G`: row `i` holds digits `i D ..`, digit `l k + τ` is `e_l base^τ`.
                        let digit = t % shape.digits;
                        if t / shape.digits == i && digit / shape.tower_digits == l {
                            value[0] = (1i128 << (base_bits * (digit % shape.tower_digits))) % qi;
                        }
                        for kk in 0..rows {
                            let abar: vec::Vec<i128> = (0..n)
                                .map(|a| (a_bar[(i * rows + kk) * n + a] % q) as i128)
                                .collect();
                            let rr = &r[(kk * m + t) * n..(kk * m + t + 1) * n];
                            let prod = negacyclic(&abar, rr, qi);
                            for a in 0..n {
                                value[a] -= prod[a];
                            }
                        }
                        for a in 0..n {
                            value[a] -= e[(i * m + t) * n + a];
                        }
                    }
                    for a in 0..n {
                        limb[n * (i * wpub + j) + a] = value[a].rem_euclid(qi) as u64;
                    }
                }
                for t in 0..m {
                    for a in 0..n {
                        let at = (i * m + t) * n + a;
                        limb[n * (rows * wpub + i * m + t) + a] = r[at].rem_euclid(qi) as u64;
                        limb[n * (rows * wpub + rows * m + i * m + t) + a] =
                            e[at].rem_euclid(qi) as u64;
                    }
                }
            }
        }
        let total: u128 = moduli.iter().map(|&q| q as u128).product();
        let mut target = vec![0u64; moduli.len() * shape.target_limb_words()];
        for idx in 0..shape.target_limb_words() {
            let value = rng.random::<u128>() % total;
            for (l, &q) in moduli.iter().enumerate() {
                target[l * shape.target_limb_words() + idx] = (value % q as u128) as u64;
            }
        }
        (trapdoor, target)
    }

    /// The negacyclic product of two polynomials modulo `q`, canonical.
    fn negacyclic(x: &[i128], y: &[i128], q: i128) -> vec::Vec<i128> {
        let n = x.len();
        let mut out = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let t = x[i] * y[j] % q;
                if i + j < n { out[i + j] += t } else { out[i + j - n] -= t }
            }
        }
        out.iter().map(|v| v.rem_euclid(q)).collect()
    }

    /// The computational sampler's minimum preimage width (`crates/backends` `bounds`).
    fn minimum_width(shape: &Shape, base: u64, sigma: f64) -> f64 {
        let n = shape.n as f64;
        1.8 * (base as f64 + 1.0) *
            sigma *
            sigma *
            (((shape.n * shape.rows * shape.digits) as f64).sqrt() + (2.0 * n).sqrt() + 4.7)
    }

    /// One draw of `shape`: the preimage family and the flag; checks that the arena's peak is
    /// `scratch_words`.
    #[allow(clippy::too_many_arguments)]
    fn run(
        shape: &Shape,
        seed: [u64; 4],
        base: u64,
        sigma: f64,
        width: f64,
        bound: u64,
        trapdoor: &[u64],
        target: &[u64],
        moduli: &[u64],
    ) -> (vec::Vec<u64>, bool) {
        let mut out = vec![0u64; moduli.len() * shape.preimage_limb_words()];
        let mut scratch = vec![0u64; scratch_words(shape)];
        let mut arena = unsafe { Arena::new(scratch.as_mut_ptr()) };
        let mut g = Stream::new(&seed);
        let ok = unsafe {
            draw(
                shape,
                &mut g,
                base,
                sigma,
                width,
                bound,
                trapdoor.as_ptr(),
                target.as_ptr(),
                moduli.as_ptr(),
                out.as_mut_ptr(),
                &mut arena,
            )
        };
        assert_eq!(arena.peak(), scratch_words(shape), "{shape:?}");
        if std::env::var("MXX_PRINT_WORDS").is_ok() {
            std::println!(
                "words per sample {}",
                g.drawn() as f64 / shape.gaussian_samples() as f64
            );
        }
        (out, ok)
    }

    /// Random small instances: each draw satisfies `A x = u` modulo every tower, and its flag says
    /// whether every coefficient is within the bound (random shapes; override the count with
    /// `MXX_TEST_PREIMAGE_DRAWS`).
    #[test]
    fn test_draws_are_preimages() {
        use rand::{Rng, SeedableRng};
        let draws: usize =
            std::env::var("MXX_TEST_PREIMAGE_DRAWS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let mut rng = rand::rngs::StdRng::seed_from_u64(rand::random());
        let moduli_pool = [(1u64 << 30) - 35, (1 << 31) - 1, (1 << 29) - 3];
        for _ in 0..draws {
            let towers = rng.random_range(1..3);
            let base_bits = rng.random_range(1..4) * 5;
            let tower_digits = 31usize.div_ceil(base_bits);
            let shape = Shape {
                n: 1 << rng.random_range(1..4),
                towers,
                rows: rng.random_range(1..4),
                digits: towers * tower_digits,
                tower_digits,
                columns: rng.random_range(1..3),
                statistical: 0,
            };
            let moduli = &moduli_pool[..towers];
            let (trapdoor, target) = instance(&mut rng, &shape, base_bits, moduli);
            let (base, sigma) = (1u64 << base_bits, 4.578);
            let width = minimum_width(&shape, base, sigma);
            let bound = (6.5 * width) as u64;
            let (out, ok) =
                run(&shape, rng.random(), base, sigma, width, bound, &trapdoor, &target, moduli);
            let (n, wpub, columns) = (shape.n, shape.public_columns(), shape.columns);
            let mut largest = 0i64;
            for (l, &q) in moduli.iter().enumerate() {
                let qi = q as i128;
                let limb = &trapdoor[l * shape.trapdoor_limb_words()..];
                let x =
                    &out[l * shape.preimage_limb_words()..(l + 1) * shape.preimage_limb_words()];
                for col in 0..columns {
                    for i in 0..shape.rows {
                        let mut sum = vec![0i128; n];
                        for j in 0..wpub {
                            let aa: vec::Vec<i128> =
                                (0..n).map(|a| limb[n * (i * wpub + j) + a] as i128).collect();
                            let xx: vec::Vec<i128> =
                                (0..n).map(|a| x[n * (j * columns + col) + a] as i128).collect();
                            for (a, value) in negacyclic(&aa, &xx, qi).into_iter().enumerate() {
                                sum[a] += value;
                            }
                        }
                        for a in 0..n {
                            let at = l * shape.target_limb_words() + n * (i * columns + col) + a;
                            assert_eq!(
                                sum[a].rem_euclid(qi) as u64,
                                target[at],
                                "{shape:?} A x = u"
                            );
                        }
                    }
                }
                largest = x.iter().fold(largest, |acc, v| acc.max(centered(*v, q).abs()));
            }
            assert_eq!(ok, largest as u64 <= bound, "{shape:?}: flag {ok}, largest {largest}");
        }
    }

    /// The preimages spread with the requested width: over many draws of one shape, the mean
    /// square coefficient is within 10% of `s²` (override the count with
    /// `MXX_TEST_PREIMAGE_SPREAD_DRAWS`).
    #[test]
    fn test_preimages_have_the_requested_width() {
        use rand::{Rng, SeedableRng};
        let draws: usize = std::env::var("MXX_TEST_PREIMAGE_SPREAD_DRAWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let mut rng = rand::rngs::StdRng::seed_from_u64(rand::random());
        let moduli = [(1u64 << 31) - 1];
        let shape = Shape {
            n: 8,
            towers: 1,
            rows: 2,
            digits: 4,
            tower_digits: 4,
            columns: 1,
            statistical: 0,
        };
        let (trapdoor, target) = instance(&mut rng, &shape, 8, &moduli);
        let (base, sigma) = (256u64, 4.578);
        let width = minimum_width(&shape, base, sigma);
        let (mut squares, mut count) = (0.0f64, 0.0f64);
        for _ in 0..draws {
            let (out, _) = run(
                &shape,
                rng.random(),
                base,
                sigma,
                width,
                u64::MAX,
                &trapdoor,
                &target,
                &moduli,
            );
            for value in &out {
                let v = centered(*value, moduli[0]) as f64;
                squares += v * v;
                count += 1.0;
            }
        }
        let ratio = squares / count / (width * width);
        assert!((ratio - 1.0).abs() < 0.1, "mean square / s² = {ratio}");
    }

    /// Statistical trapdoors: TrapGen accepts a `T` within its attempts at the minimum width, and
    /// each preimage satisfies `A x = u` modulo every tower with the flag saying whether it is
    /// within the bound; over the draws of each instance the mean square coefficient is within
    /// 15% of `s²` (random shapes; override the count with `MXX_TEST_STATISTICAL_DRAWS`).
    #[test]
    fn test_statistical_draws_are_preimages() {
        use rand::{Rng, SeedableRng};
        let instances: usize = std::env::var("MXX_TEST_STATISTICAL_DRAWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(rand::random());
        let moduli_pool = [(1u64 << 30) - 35, (1 << 31) - 1];
        for _ in 0..instances {
            let towers = rng.random_range(1..3);
            let base_bits = rng.random_range(2..4) * 5;
            let tower_digits = 31usize.div_ceil(base_bits);
            let m_bar = rng.random_range(1..6);
            let shape = Shape {
                n: 1 << rng.random_range(1..5),
                towers,
                rows: 1,
                digits: towers * tower_digits,
                tower_digits,
                columns: rng.random_range(1..3),
                statistical: m_bar,
            };
            let moduli = &moduli_pool[..towers];
            let (n, digits, wpub) = (shape.n, shape.digits, shape.public_columns());
            let (base, sigma) = (1u64 << base_bits, 4.578f64);
            let c = (base as f64 + 1.0) * sigma;
            let deviation = (n as f64 / (2.0 / 1024.0)).ln().max(0.0);
            let s1 = sigma *
                (n as f64).sqrt() *
                ((m_bar as f64).sqrt() + (digits as f64).sqrt() + deviation.sqrt());
            let width = (c * c * (1.0 + s1 * s1) + sigma * sigma).sqrt();
            // TrapGen.
            let t_words = m_bar * digits * n;
            let mut t_limbs = vec![0u64; towers * t_words];
            let mut scratch = vec![0u64; trapgen_scratch_words(n, m_bar, digits)];
            let accepted = (0..8).any(|_| {
                let mut arena = unsafe { Arena::new(scratch.as_mut_ptr()) };
                let mut g = Stream::new(&rng.random::<[u64; 4]>());
                let ok = unsafe {
                    trapgen(
                        &mut g,
                        n,
                        towers,
                        m_bar,
                        digits,
                        base,
                        sigma,
                        width,
                        moduli.as_ptr(),
                        t_limbs.as_mut_ptr(),
                        &mut arena,
                    )
                };
                assert_eq!(arena.peak(), trapgen_scratch_words(n, m_bar, digits));
                ok
            });
            assert!(accepted, "{shape:?}: TrapGen rejected every T");
            let t: vec::Vec<i128> =
                t_limbs[..t_words].iter().map(|v| centered(*v, moduli[0]) as i128).collect();
            // A = [ā | g - ā T] per tower, then T.
            let a_bar: vec::Vec<u64> = (0..m_bar * n).map(|_| rng.random::<u64>()).collect();
            let mut trapdoor = vec![0u64; towers * shape.trapdoor_limb_words()];
            for (l, &q) in moduli.iter().enumerate() {
                let qi = q as i128;
                let limb = &mut trapdoor[l * shape.trapdoor_limb_words()..];
                let abar = |k: usize| -> vec::Vec<i128> {
                    (0..n).map(|a| (a_bar[k * n + a] % q) as i128).collect()
                };
                for j in 0..wpub {
                    let mut value = vec![0i128; n];
                    if j < m_bar {
                        value = abar(j);
                    } else {
                        let tt = j - m_bar;
                        if tt / tower_digits == l {
                            value[0] = (1i128 << (base_bits * (tt % tower_digits))) % qi;
                        }
                        for k in 0..m_bar {
                            let prod = negacyclic(
                                &abar(k),
                                &t[(k * digits + tt) * n..(k * digits + tt + 1) * n],
                                qi,
                            );
                            for a in 0..n {
                                value[a] -= prod[a];
                            }
                        }
                    }
                    for a in 0..n {
                        limb[n * j + a] = value[a].rem_euclid(qi) as u64;
                    }
                }
                limb[n * wpub..n * wpub + t_words]
                    .copy_from_slice(&t_limbs[l * t_words..(l + 1) * t_words]);
            }
            let total: u128 = moduli.iter().map(|&q| q as u128).product();
            let mut target = vec![0u64; towers * shape.target_limb_words()];
            for idx in 0..shape.target_limb_words() {
                let value = rng.random::<u128>() % total;
                for (l, &q) in moduli.iter().enumerate() {
                    target[l * shape.target_limb_words() + idx] = (value % q as u128) as u64;
                }
            }
            let bound = (6.5 * width) as u64;
            let (mut squares, mut count) = (0.0f64, 0.0f64);
            for _ in 0..200 {
                let (out, ok) = run(
                    &shape,
                    rng.random(),
                    base,
                    sigma,
                    width,
                    bound,
                    &trapdoor,
                    &target,
                    moduli,
                );
                let mut largest = 0i64;
                for (l, &q) in moduli.iter().enumerate() {
                    let qi = q as i128;
                    let limb = &trapdoor[l * shape.trapdoor_limb_words()..];
                    let x = &out
                        [l * shape.preimage_limb_words()..(l + 1) * shape.preimage_limb_words()];
                    for col in 0..shape.columns {
                        let mut sum = vec![0i128; n];
                        for j in 0..wpub {
                            let aa: vec::Vec<i128> =
                                (0..n).map(|a| limb[n * j + a] as i128).collect();
                            let xx: vec::Vec<i128> = (0..n)
                                .map(|a| x[n * (j * shape.columns + col) + a] as i128)
                                .collect();
                            for (a, value) in negacyclic(&aa, &xx, qi).into_iter().enumerate() {
                                sum[a] += value;
                            }
                        }
                        for a in 0..n {
                            let at = l * shape.target_limb_words() + n * col + a;
                            assert_eq!(
                                sum[a].rem_euclid(qi) as u64,
                                target[at],
                                "{shape:?} A x = u"
                            );
                        }
                    }
                    largest = x.iter().fold(largest, |acc, v| acc.max(centered(*v, q).abs()));
                    if l == 0 {
                        for v in x {
                            let v = centered(*v, q) as f64;
                            squares += v * v;
                            count += 1.0;
                        }
                    }
                }
                assert_eq!(ok, largest as u64 <= bound, "{shape:?}: flag {ok}, largest {largest}");
            }
            let ratio = squares / count / (width * width);
            assert!((ratio - 1.0).abs() < 0.15, "{shape:?}: mean square / s² = {ratio}");
        }
    }
}
