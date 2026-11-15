//! Discrete Gaussian sampling from coins, for the trapdoor samplers.
//!
//! [`Stream`] expands a seed of coins into a stream of uniform words by SHAKE256 (Keccak-f[1600]
//! at rate 136 with the `0x1f ... 0x80` padding), so a leaf draws as many uniform values as its
//! rejection loops need from one fixed-size coin input. [`karney`] samples the discrete Gaussian
//! of any real center and width exactly, after Karney ("Sampling exactly from the normal
//! distribution", 2016) as OpenFHE's `GenerateIntegerKarney` implements it, with every uniform
//! real of 53-bit precision. All arithmetic is on 64-bit integers and binary64, as the machine has
//! no 32-bit word instructions.

use crate::keccak_f;

/// The rate of SHAKE256 in bytes, and in words.
const RATE_WORDS: usize = 17;

/// A SHAKE256 stream of words from a seed, of at most `budget` words: past it every word is 0 and
/// [`Stream::exhausted`] holds, so a leaf's rejection loops end within a step bound and the leaf
/// reports the draw as failed instead.
pub struct Stream {
    state: [u64; 25],
    /// Words of the current block already used.
    used: usize,
    /// Words drawn, and the most that may be.
    drawn: u64,
    budget: u64,
}

impl Stream {
    /// The stream of `seed` (whole words, absorbed little-endian), without a budget.
    pub fn new(seed: &[u64]) -> Self {
        Self::with_budget(seed, u64::MAX)
    }

    /// The stream of `seed` of at most `budget` words.
    pub fn with_budget(seed: &[u64], budget: u64) -> Self {
        let mut state = [0u64; 25];
        let mut lane = 0;
        let mut i = 0;
        while i < seed.len() {
            state[lane] ^= seed[i];
            lane += 1;
            if lane == RATE_WORDS {
                keccak_f(&mut state);
                lane = 0;
            }
            i += 1;
        }
        // SHAKE padding: `0x1f` after the message, `0x80` at the end of the block.
        state[lane] ^= 0x1f;
        state[RATE_WORDS - 1] ^= 0x80u64 << 56;
        keccak_f(&mut state);
        Self { state, used: 0, drawn: 0, budget }
    }

    /// The words drawn so far.
    pub fn drawn(&self) -> u64 {
        self.drawn
    }

    /// Whether the budget ran out: a word past it was asked for.
    pub fn exhausted(&self) -> bool {
        self.drawn > self.budget
    }

    /// Marks the draw as failed, as an exhausted budget does.
    pub fn fail(&mut self) {
        self.drawn = u64::MAX;
        self.budget = 0;
    }

    /// The next uniform word, or 0 past the budget.
    pub fn word(&mut self) -> u64 {
        if self.drawn >= self.budget {
            self.drawn = self.budget.saturating_add(1);
            return 0;
        }
        self.drawn += 1;
        if self.used == RATE_WORDS {
            keccak_f(&mut self.state);
            self.used = 0;
        }
        let w = self.state[self.used];
        self.used += 1;
        w
    }

    /// A uniform real in `[0, 1)` with 53 random bits.
    pub fn unit(&mut self) -> f64 {
        (self.word() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// A uniform bit.
    pub fn bit(&mut self) -> bool {
        self.word() & 1 == 1
    }

    /// A uniform integer in `[0, n)`, `n > 0`, by rejection.
    pub fn below(&mut self, n: u64) -> u64 {
        let zone = u64::MAX - (u64::MAX % n + 1) % n;
        loop {
            let w = self.word();
            if w <= zone {
                return w % n;
            }
        }
    }
}

/// Bernoulli of probability `exp(-1/2)` (Karney's algorithm H).
fn algorithm_h(g: &mut Stream) -> bool {
    let mut a = g.unit();
    if !(a < 0.5) {
        return true;
    }
    loop {
        let b = g.unit();
        if !(b < a) {
            return false;
        }
        a = g.unit();
        if !(a < b) {
            return true;
        }
    }
}

/// The number of successes of algorithm H before its first failure (Karney's algorithm G).
fn algorithm_g(g: &mut Stream) -> u64 {
    let mut n = 0;
    while algorithm_h(g) {
        n += 1;
    }
    n
}

/// Bernoulli of probability `exp(-n/2)` (Karney's algorithm P).
fn algorithm_p(g: &mut Stream, n: u64) -> bool {
    let mut i = 0;
    while i < n {
        if !algorithm_h(g) {
            return false;
        }
        i += 1;
    }
    true
}

/// Bernoulli of probability `exp(-x (2 k + x) / (2 k + 2))` (Karney's algorithm B).
fn algorithm_b(g: &mut Stream, k: u64, x: f64) -> bool {
    let mut y = x;
    let mut n = 0u64;
    let m = (2 * k + 2) as f64;
    let bound = (2 * k) as f64;
    loop {
        let z = g.unit();
        if !(z < y) {
            break;
        }
        let r = g.unit();
        if !(r < (bound + x) / m) {
            break;
        }
        y = z;
        n += 1;
    }
    n.is_multiple_of(2)
}

/// A sample of the discrete Gaussian of center `mean` and width `stddev` (`stddev > 0`), exactly;
/// 0 once the stream is exhausted.
pub fn karney(g: &mut Stream, mean: f64, stddev: f64) -> i64 {
    let span = ceil(stddev) as u64;
    loop {
        if g.exhausted() {
            return 0;
        }
        // D1, D2.
        let k = algorithm_g(g);
        if !algorithm_p(g, k * k.saturating_sub(1)) {
            continue;
        }
        // D3.
        let s: i64 = if g.bit() { 1 } else { -1 };
        // D4.
        let di0 = stddev * k as f64 + s as f64 * mean;
        let i0 = ceil(di0) as i64;
        let x0 = (i0 as f64 - di0) / stddev;
        let j = g.below(span) as i64;
        let x = x0 + j as f64 / stddev;
        // D5, D6.
        if !(x < 1.0) || (x == 0.0 && s < 0 && k == 0) {
            continue;
        }
        // D7.
        let mut h = k + 1;
        let mut accepted = true;
        while h > 0 {
            if !algorithm_b(g, k, x) {
                accepted = false;
                break;
            }
            h -= 1;
        }
        if !accepted {
            continue;
        }
        // D8.
        return s * (i0 + j);
    }
}

/// The square root of a nonnegative binary64, correctly rounded.
pub fn sqrt(x: f64) -> f64 {
    f64::from_bits(crate::sqrt_bits(x.to_bits()))
}

/// The natural logarithm of a positive normal binary64, within a few ulps: `x = m 2^e` with `m` in
/// `[sqrt(1/2), sqrt(2))`, and `ln m = 2 atanh((m - 1) / (m + 1))` by its series.
pub fn ln(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let (mut term, mut sum, mut k) = (s, 0.0, 1.0);
    while k < 40.0 {
        sum += term / k;
        term *= s2;
        k += 2.0;
    }
    2.0 * sum + e as f64 * core::f64::consts::LN_2
}

/// A sample of the continuous Gaussian of center 0 and width `sigma`, by Marsaglia's polar method;
/// 0 once the stream is exhausted.
pub fn normal(g: &mut Stream, sigma: f64) -> f64 {
    loop {
        if g.exhausted() {
            return 0.0;
        }
        let u = 2.0 * g.unit() - 1.0;
        let v = 2.0 * g.unit() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return sigma * u * sqrt(-2.0 * ln(s) / s);
        }
    }
}

/// The most gadget digits per coefficient.
pub const MAX_DIGITS: usize = 64;

/// The `k` digits of `v` in base `b`, least significant first.
fn digits(mut v: u64, b: u64, k: usize, out: &mut [i64; MAX_DIGITS]) {
    let mut t = 0;
    while t < k {
        out[t] = (v % b) as i64;
        v /= b;
        t += 1;
    }
}

/// A Gaussian sample of the coset `{z : Σ z_t b^t ≡ v (mod q)}` of the gadget lattice of `k`
/// digits in base `b` (`b^k >= q`), of width `stddev`: OpenFHE's `GaussSampGqArbBase` for one
/// coefficient (Algorithm 3 of eprint 2017/844), with its continuous perturbation.
#[allow(clippy::needless_range_loop)]
pub fn gadget_sample(
    g: &mut Stream,
    v: u64,
    q: u64,
    b: u64,
    k: usize,
    stddev: f64,
    z: &mut [i64; MAX_DIGITS],
) {
    let base = b as f64;
    let sigma = stddev / (base + 1.0);
    let (mut m_digits, mut v_digits) = ([0i64; MAX_DIGITS], [0i64; MAX_DIGITS]);
    digits(q, b, k, &mut m_digits);
    digits(v, b, k, &mut v_digits);
    let (mut l, mut h, mut c) = ([0f64; MAX_DIGITS], [0f64; MAX_DIGITS], [0f64; MAX_DIGITS]);
    // As OpenFHE, whose `1 / k` divides integers: 0 unless `k` is 1.
    let first = if k == 1 { 1.0 } else { 0.0 };
    l[0] = sqrt(base * (1.0 + first) + 1.0);
    let mut i = 1;
    while i < k {
        l[i] = sqrt(base * (1.0 + 1.0 / (k - i) as f64));
        h[i] = sqrt(base * (1.0 - 1.0 / (k - (i - 1)) as f64));
        i += 1;
    }
    c[0] = m_digits[0] as f64 / base;
    let mut i = 1;
    while i < k {
        c[i] = (c[i - 1] + m_digits[i] as f64) / base;
        i += 1;
    }
    // The perturbation `p = L x` for continuous Gaussian `x` of width `sigma`.
    let mut x = [0f64; MAX_DIGITS];
    let mut i = 0;
    while i < k {
        x[i] = normal(g, sigma);
        i += 1;
    }
    let mut p = [0f64; MAX_DIGITS];
    let mut i = 0;
    while i + 1 < k {
        p[i] = l[i] * x[i] + h[i + 1] * x[i + 1];
        i += 1;
    }
    p[k - 1] = h[k - 1] * x[k - 1];
    let mut a = [0f64; MAX_DIGITS];
    a[0] = (v_digits[0] as f64 - p[0]) / base;
    let mut t = 1;
    while t < k {
        a[t] = (a[t - 1] + v_digits[t] as f64 - p[t]) / base;
        t += 1;
    }
    // SampleC.
    let mut zj = [0i64; MAX_DIGITS];
    zj[k - 1] = karney(g, -a[k - 1] / c[k - 1], sigma / c[k - 1]);
    let mut i = 0;
    while i < k {
        a[i] += zj[k - 1] as f64 * c[i];
        i += 1;
    }
    let mut i = 0;
    while i + 1 < k {
        zj[i] = karney(g, -a[i], sigma);
        i += 1;
    }
    let bi = b as i64;
    z[0] = bi * zj[0] + m_digits[0] * zj[k - 1] + v_digits[0];
    let mut t = 1;
    while t + 1 < k {
        z[t] = bi * zj[t] - zj[t - 1] + m_digits[t] * zj[k - 1] + v_digits[t];
        t += 1;
    }
    z[k - 1] = m_digits[k - 1] * zj[k - 1] - zj[k - 2] + v_digits[k - 1];
}

/// The least integer not below `x`, for `|x| < 2^62`.
pub fn ceil(x: f64) -> f64 {
    let t = x as i64 as f64;
    if t < x { t + 1.0 } else { t }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    extern crate std;
    use super::*;

    /// Karney's sampler has the discrete Gaussian's mean and variance for random centers and
    /// widths (`MXX_TEST_KARNEY_SAMPLES` samples, default 200000).
    #[test]
    fn test_karney_samples_have_the_gaussian_moments() {
        let samples: usize = std::env::var("MXX_TEST_KARNEY_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);
        let seed: [u64; 4] =
            core::array::from_fn(|i| 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1));
        let mut g = Stream::new(&seed);
        for (mean, stddev) in [(0.0, 3.0), (0.37, 5.5), (-12.6, 20.0), (1000.25, 1.5)] {
            let (mut sum, mut squares) = (0.0f64, 0.0f64);
            for _ in 0..samples {
                let x = karney(&mut g, mean, stddev) as f64 - mean;
                sum += x;
                squares += x * x;
            }
            let n = samples as f64;
            let (m, v) = (sum / n, squares / n - (sum / n) * (sum / n));
            assert!(m.abs() < 6.0 * stddev / n.sqrt(), "mean {m} at ({mean}, {stddev})");
            assert!(
                (v / (stddev * stddev) - 1.0).abs() < 0.02,
                "variance {v} at ({mean}, {stddev})"
            );
        }
    }

    /// `ln` agrees with the host's logarithm, and `normal` has the Gaussian's moments.
    #[test]
    fn test_ln_and_normal() {
        for x in [1e-300, 0.001, 0.5, 0.7, 1.0, 1.5, 2.0, 3.7, 1e10, 1.7e308] {
            let (ours, host) = (ln(x), x.ln());
            assert!((ours - host).abs() <= 1e-14 * host.abs().max(1.0), "ln {x}: {ours} vs {host}");
        }
        let mut g = Stream::new(&[7, 8]);
        let n = 200_000;
        let (mut sum, mut squares) = (0.0, 0.0);
        for _ in 0..n {
            let x = normal(&mut g, 3.0);
            sum += x;
            squares += x * x;
        }
        let (m, v) = (sum / n as f64, squares / n as f64);
        assert!(m.abs() < 0.05 && (v / 9.0 - 1.0).abs() < 0.02, "mean {m}, variance {v}");
    }

    /// The gadget sampler's digits reconstruct the coefficient modulo `q`, for random moduli,
    /// bases, and coefficients, and spread with about the requested width.
    #[test]
    fn test_gadget_samples_lie_in_the_coset() {
        let mut g = Stream::new(&[1, 2, 3]);
        for (q, b) in [((1u64 << 40) - 87, 16u64), ((1 << 61) - 1, 1 << 20), (998244353, 2)] {
            let mut k = 0;
            let mut power: u128 = 1;
            while power < u128::from(q) {
                power *= u128::from(b);
                k += 1;
            }
            let stddev = (b as f64 + 1.0) * 4.0;
            let mut largest = 0i64;
            for _ in 0..2000 {
                let v = g.below(q);
                let mut z = [0i64; MAX_DIGITS];
                gadget_sample(&mut g, v, q, b, k, stddev, &mut z);
                let mut sum: i128 = 0;
                let mut weight: i128 = 1;
                for t in 0..k {
                    sum += i128::from(z[t]) * weight;
                    weight *= i128::from(b);
                    largest = largest.max(z[t].abs());
                }
                assert_eq!(sum.rem_euclid(i128::from(q)), i128::from(v), "a coset member");
            }
            assert!((largest as f64) < 20.0 * stddev, "digits {largest} for width {stddev}");
        }
    }

    /// The stream is SHAKE256: its first words for the empty seed are SHAKE256("")'s first bytes.
    #[test]
    fn test_stream_is_shake256() {
        let mut g = Stream::new(&[]);
        // SHAKE256("") = 46b9dd2b0ba88d13233b3feb743eeb24...
        assert_eq!(g.word().to_le_bytes(), [0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13]);
        assert_eq!(g.word().to_le_bytes(), [0x23, 0x3b, 0x3f, 0xeb, 0x74, 0x3e, 0xeb, 0x24]);
    }
}
