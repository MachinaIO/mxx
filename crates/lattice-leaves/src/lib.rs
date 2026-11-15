//! Leaves of the lattice DSL as `no_std` Rust.
//!
//! Each leaf is an `extern "C"` function of one argument, the address of a descriptor of
//! little-endian `u64` words laid out as `mxx_core::leaf` documents: the leaf's arguments, then
//! the address of each input, then the address of each output, then the scratch address. The same
//! source runs on the host, where `mxx_ir_core::lower` calls it with host addresses, and is
//! compiled to RV64IM for proof images by `scripts/build_rv64_leaves.sh`. The code must not
//! panic, allocate, or keep state, and uses only 64-bit integers and binary64 (soft float), since
//! the mathcal-A machine has no floating point and no RV64 word operations other than `ADDIW`.
//!
//! Matrices are limbs as `mxx_ir_core::lower` encodes them: residues of one CRT modulus `q`,
//! entries row-major, each entry `N` coefficients of the negacyclic ring `Z_q[X]/(X^N + 1)`.

#![no_std]

// Negated comparisons keep NaN on the rejecting side, as OpenFHE's samplers do.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub mod gauss;
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub mod preimage;

/// The descriptor of one call.
struct Descriptor(*const u64);

impl Descriptor {
    #[inline(always)]
    unsafe fn word(&self, i: usize) -> u64 {
        unsafe { *self.0.add(i) }
    }

    #[inline(always)]
    unsafe fn input(&self, i: usize) -> *const u64 {
        unsafe { self.word(i) as *const u64 }
    }

    #[inline(always)]
    unsafe fn output(&self, i: usize) -> *mut u64 {
        unsafe { self.word(i) as *mut u64 }
    }
}

/// `a b mod q` for `q < 2^63`, by reducing the 128-bit product one bit at a time from the top, so
/// only 64-bit operations run: a 128-bit `%` would call compiler_builtins' division, which uses
/// RV64 word operations the mathcal-A machine lacks. Every partial remainder is below `q`, so
/// doubling it and adding a bit stays below `2^64`.
#[inline(always)]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    let product = a as u128 * b as u128;
    reduce((product >> 64) as u64, product as u64, q)
}

/// `(high 2^64 + low) mod m` for `m < 2^63`, one bit at a time (see [`mul_mod`]).
#[inline(always)]
fn reduce(high: u64, low: u64, m: u64) -> u64 {
    let mut remainder = 0u64;
    let mut i = 128u64;
    while i > 0 {
        i -= 1;
        let bit = if i >= 64 { (high >> (i - 64)) & 1 } else { (low >> i) & 1 };
        remainder = (remainder << 1) | bit;
        if remainder >= m {
            remainder -= m;
        }
    }
    remainder
}

#[inline(always)]
fn add_mod(a: u64, b: u64, q: u64) -> u64 {
    let s = a + b;
    if s >= q { s - q } else { s }
}

#[inline(always)]
fn sub_mod(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

/// `I_count ⊗ A` for an `rows x columns` matrix `A`. Descriptor: `N, rows, columns, count, A, out,
/// scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, and the output must not overlap the input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_block_diagonal(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, rows, columns, count) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize, d.word(3) as usize);
        let (a, out) = (d.input(4), d.output(5));
        let total_columns = columns * count;
        let mut i = 0;
        while i < rows * count * total_columns * n {
            *out.add(i) = 0;
            i += 1;
        }
        let mut block = 0;
        while block < count {
            let mut r = 0;
            while r < rows {
                let mut c = 0;
                while c < columns {
                    let (from, to) = (
                        a.add((r * columns + c) * n),
                        out.add(((block * rows + r) * total_columns + block * columns + c) * n),
                    );
                    let mut k = 0;
                    while k < n {
                        *to.add(k) = *from.add(k);
                        k += 1;
                    }
                    c += 1;
                }
                r += 1;
            }
            block += 1;
        }
    }
}

// Integers are two's complement of `W` words (`1 <= W <= BIG`), held in `[u64; BIG]` sign-extended
// to `BIG` words, so every value of `W` words keeps its sign; results are stored as their low `W`
// words, which wraps them modulo `2^(64 W)`.

/// The `W`-word integer at `p`, sign-extended.
#[inline(always)]
unsafe fn int_load(p: *const u64, w: usize) -> [u64; BIG] {
    let mut a = [0u64; BIG];
    let mut i = 0;
    while i < w {
        a[i] = unsafe { *p.add(i) };
        i += 1;
    }
    let fill = if a[w - 1] >> 63 == 1 { u64::MAX } else { 0 };
    while i < BIG {
        a[i] = fill;
        i += 1;
    }
    a
}

/// Stores the low `W` words of `a` at `p`.
#[inline(always)]
unsafe fn int_store(p: *mut u64, w: usize, a: &[u64; BIG]) {
    let mut i = 0;
    while i < w {
        unsafe { *p.add(i) = a[i] };
        i += 1;
    }
}

#[inline(always)]
fn int_negative(a: &[u64; BIG]) -> bool {
    a[BIG - 1] >> 63 == 1
}

/// `-a`, wrapping.
#[inline(always)]
fn int_neg(a: &[u64; BIG]) -> [u64; BIG] {
    let mut out = [0u64; BIG];
    let mut carry = 1u64;
    let mut i = 0;
    while i < BIG {
        let (s, c) = (!a[i]).overflowing_add(carry);
        out[i] = s;
        carry = c as u64;
        i += 1;
    }
    out
}

/// `|a|`.
#[inline(always)]
fn int_abs(a: &[u64; BIG]) -> [u64; BIG] {
    if int_negative(a) { int_neg(a) } else { *a }
}

/// `a b` modulo `2^(64 W)`, in the low `W` words.
#[inline(always)]
fn int_mul(a: &[u64; BIG], b: &[u64; BIG], w: usize) -> [u64; BIG] {
    let mut out = [0u64; BIG];
    let mut i = 0;
    while i < w {
        let mut carry = 0u128;
        let mut j = 0;
        while i + j < w {
            let t = out[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
            j += 1;
        }
        i += 1;
    }
    out
}

/// Whether `a < b` as signed integers.
#[inline(always)]
fn int_lt(a: &[u64; BIG], b: &[u64; BIG]) -> bool {
    match (int_negative(a), int_negative(b)) {
        (true, false) => true,
        (false, true) => false,
        _ => !big_ge(a, b),
    }
}

/// The quotient and remainder of `a` by `|b|` for `b != 0`, rounding down, as the graph's
/// integer division: `a = q |b| + r` with `0 <= r < |b|`.
#[inline(always)]
fn int_div_rem(a: &[u64; BIG], b: &[u64; BIG]) -> ([u64; BIG], [u64; BIG]) {
    let (magnitude, divisor) = (int_abs(a), int_abs(b));
    let (mut q, mut r) = big_div_rem(&magnitude, &divisor);
    if int_negative(a) && r != [0u64; BIG] {
        big_add(&mut q, &big_word(1));
        let mut rest = divisor;
        big_sub(&mut rest, &r);
        r = rest;
    }
    if int_negative(a) {
        q = int_neg(&q);
    }
    (q, r)
}

/// `W`-word integer arithmetic: `op` 0 adds, 1 subtracts, 2 multiplies (all wrapping), 3 and 4
/// are the quotient and remainder of `a` by `|b|`, rounding down (`0 <= r < |b|`). A zero divisor
/// gives 0: the graph's semantics make it an error, which the lowering does not reproduce.
/// Descriptor: `W, op, a, b, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_int_binary(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let w = d.word(0) as usize;
        let (a, b) = (int_load(d.input(2), w), int_load(d.input(3), w));
        let out = match d.word(1) {
            0 => {
                let mut s = a;
                big_add(&mut s, &b);
                s
            }
            1 => {
                let mut s = a;
                big_sub(&mut s, &b);
                s
            }
            2 => int_mul(&a, &b, w),
            _ if b == [0u64; BIG] => [0u64; BIG],
            3 => int_div_rem(&a, &b).0,
            _ => int_div_rem(&a, &b).1,
        };
        int_store(d.output(4), w, &out);
    }
}

/// `W`-word integer comparison to a Bool: `op` 0 is `=`, 1 is `<`, 2 is `<=`. Descriptor: `W, op,
/// a, b, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_int_compare(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let w = d.word(0) as usize;
        let (a, b) = (int_load(d.input(2), w), int_load(d.input(3), w));
        *d.output(4) = match d.word(1) {
            0 => a == b,
            1 => int_lt(&a, &b),
            _ => !int_lt(&b, &a),
        } as u64;
    }
}

/// Bit `bit` of a `W`-word integer in two's complement, as a Bool; bits above the width are the
/// sign. Descriptor: `W, bit, a, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_bit_extract(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (w, bit) = (d.word(0) as usize, d.word(1) as usize);
        let a = int_load(d.input(2), w);
        *d.output(3) =
            if bit >= 64 * BIG { a[BIG - 1] >> 63 } else { (a[bit / 64] >> (bit % 64)) & 1 };
    }
}

/// `a mod q` in `[0, q)` of the integer `a`.
#[inline(always)]
fn int_rem_small(a: &[u64; BIG], q: u64) -> u64 {
    let r = big_rem_small(&int_abs(a), q);
    if int_negative(a) && r != 0 { q - r } else { r }
}

/// The transpose of a `rows x columns` matrix of `N`-coefficient entries. Descriptor: `N, rows,
/// columns, A, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, and the output must not overlap the input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_transpose(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, rows, columns) = (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize);
        let (a, out) = (d.input(3), d.output(4));
        let mut r = 0;
        while r < rows {
            let mut c = 0;
            while c < columns {
                let (from, to) = (a.add((r * columns + c) * n), out.add((c * rows + r) * n));
                let mut k = 0;
                while k < n {
                    *to.add(k) = *from.add(k);
                    k += 1;
                }
                c += 1;
            }
            r += 1;
        }
    }
}

/// Writes `sign * x` modulo `q` at coefficient `position` (taken modulo `2N`) of `entry`, where
/// `X^N = -1`.
#[inline(always)]
unsafe fn place(entry: *mut u64, n: usize, q: u64, position: usize, x: u64) {
    unsafe {
        let position = position % (2 * n);
        if position < n {
            *entry.add(position) = x;
        } else {
            *entry.add(position - n) = if x == 0 { 0 } else { q - x };
        }
    }
}

/// `X -> X^k` on each of `entries` entries, for odd `0 < k < 2N`. Descriptor: `N, q, k, entries,
/// A, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, and the output must not overlap the input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_automorphism(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q, k, entries) =
            (d.word(0) as usize, d.word(1), d.word(2) as usize, d.word(3) as usize);
        let (a, out) = (d.input(4), d.output(5));
        let mut e = 0;
        while e < entries {
            let mut i = 0;
            while i < n {
                place(out.add(e * n), n, q, i * k, *a.add(e * n + i));
                i += 1;
            }
            e += 1;
        }
    }
}

/// `n` residues uniform modulo `q`, each from 16 bytes of coins reduced modulo `q` (statistical
/// distance below `q / 2^128` per residue). Descriptor: `n, q, coins, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `16 n` bytes of coins and `q < 2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_uniform_mod(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q) = (d.word(0) as usize, d.word(1));
        let (coins, out) = (d.input(2), d.output(3));
        let mut i = 0;
        while i < n {
            *out.add(i) = reduce(*coins.add(2 * i + 1), *coins.add(2 * i), q);
            i += 1;
        }
    }
}

/// `n` integers uniform on `[minimum, minimum + width)`, each from 16 bytes of coins reduced
/// modulo `width`, written modulo `q`; `minimum_mod_q` is `minimum mod q`. Every limb reads the
/// same coins, so the residues are of the same integers. Descriptor: `n, q, minimum_mod_q, width,
/// coins, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `16 n` bytes of coins, `width < 2^63`, and
/// `q < 2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_interval_mod(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q, minimum, width) = (d.word(0) as usize, d.word(1), d.word(2), d.word(3));
        let (coins, out) = (d.input(4), d.output(5));
        let mut i = 0;
        while i < n {
            let offset = reduce(*coins.add(2 * i + 1), *coins.add(2 * i), width);
            *out.add(i) = add_mod(reduce(0, offset, q), minimum, q);
            i += 1;
        }
    }
}

/// `n` integers drawn by a cumulative distribution table: integer `i` is the number of the
/// `entries` table words at most its 63-bit uniform coin, minus the table's offset, written modulo
/// `q` by adding `offset_mod_q`. Every limb reads the same coins. The whole table is read for
/// every integer, so the time does not depend on the coins. Descriptor: `n, q, entries,
/// offset_mod_q, coins, table, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `8 n` bytes of coins and `entries < q < 2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_cdt_mod(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q, entries, offset) =
            (d.word(0) as usize, d.word(1), d.word(2) as usize, d.word(3));
        let (coins, table, out) = (d.input(4), d.input(5), d.output(6));
        let mut i = 0;
        while i < n {
            let u = *coins.add(i) >> 1;
            let mut index = 0u64;
            let mut k = 0;
            while k < entries {
                index += (u >= *table.add(k)) as u64;
                k += 1;
            }
            *out.add(i) = add_mod(index, offset, q);
            i += 1;
        }
    }
}

/// The balanced base-`base` digit step of `value` (ties to even): `(digit, next)` with
/// `value = digit + base next` and `digit` in `[-base/2, base/2]`.
#[inline(always)]
fn balanced_digit(value: i64, base: i64) -> (i64, i64) {
    let (quotient, remainder) = (value.div_euclid(base), value.rem_euclid(base));
    let half = base / 2;
    if remainder < half {
        (remainder, quotient)
    } else if remainder > half {
        (remainder - base, quotient + 1)
    } else if quotient % 2 == 0 {
        (half, quotient)
    } else {
        (half - base, quotient + 1)
    }
}

/// The balanced CRT gadget decompositions of the CPU backend. A coefficient's residue modulo a
/// tower `q_t` is lifted to `(-q_t/2, q_t/2]` and split into `digits_per_tower` balanced
/// base-`base` digits (ties to even).
///
/// - Without `small`: digit `d` belongs to tower `t = d / digits_per_tower`, for the first
///   `retained` towers, and is written modulo every `q_l`. With dropped towers (`retained < L`,
///   ePrint 2024/909, section 3.2), the residues are first corrected by the CRT sum of the dropped
///   towers' centered twisted residues: tower `t` subtracts `sum_p weight[p][t] centered(z_p
///   twist[p] mod p)` over the dropped `p`, where `twist[p]` is the inverse of `P / p` modulo `p`
///   and `weight[p][t] = (P / p) mod q_t`, `P` the product of the dropped towers.
/// - With `small`: digit `d` holds, in limb `l`, digit `d` of tower `l`'s own residue.
///
/// Row `i D + d` of the output holds digit `d` of row `i`, `D` the digits per row. Descriptor:
/// `N, L, rows, columns, base, digits_per_tower, limb_words_in, limb_words_out, retained, small,
/// matrix, moduli, twist, weight, out, scratch`, with `weight` source-major (`L x L`).
///
/// # Safety
///
/// `descriptor` must follow the layout above, with moduli below `2^62` and `base` a power of two.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_gadget_decompose(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, depth, rows, columns) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize, d.word(3) as usize);
        let (base, per_tower, words_in, words_out) =
            (d.word(4) as i64, d.word(5) as usize, d.word(6) as usize, d.word(7) as usize);
        let (retained, small) = (d.word(8) as usize, d.word(9) != 0);
        let (matrix, moduli, twist, weight, out) =
            (d.input(10), d.input(11), d.input(12), d.input(13), d.output(14));
        let digits = if small { per_tower } else { retained * per_tower };
        let mut i = 0;
        while i < rows {
            let mut j = 0;
            while j < columns {
                let mut k = 0;
                while k < n {
                    let entry = (i * columns + j) * n + k;
                    let mut tower = 0;
                    while tower < if small { depth } else { retained } {
                        let q = *moduli.add(tower);
                        let mut residue = *matrix.add(tower * words_in + entry) % q;
                        // The correction by the dropped towers.
                        let mut p = retained;
                        while p < depth {
                            let prime = *moduli.add(p);
                            let twisted = mul_mod(
                                *matrix.add(p * words_in + entry) % prime,
                                *twist.add(p),
                                prime,
                            );
                            let centered = if twisted * 2 > prime {
                                sub_mod(0, (prime - twisted) % q, q)
                            } else {
                                twisted % q
                            };
                            residue = sub_mod(
                                residue,
                                mul_mod(*weight.add(p * depth + tower), centered, q),
                                q,
                            );
                            p += 1;
                        }
                        let mut value = if residue * 2 > q {
                            residue as i64 - q as i64
                        } else {
                            residue as i64
                        };
                        let mut local = 0;
                        while local < per_tower {
                            let (digit, next) = balanced_digit(value, base);
                            value = next;
                            if small {
                                let row = i * digits + local;
                                *out.add(tower * words_out + (row * columns + j) * n + k) =
                                    digit.rem_euclid(q as i64) as u64;
                            } else {
                                let row = i * digits + tower * per_tower + local;
                                let mut l = 0;
                                while l < depth {
                                    let ql = *moduli.add(l) as i64;
                                    *out.add(l * words_out + (row * columns + j) * n + k) =
                                        digit.rem_euclid(ql) as u64;
                                    l += 1;
                                }
                            }
                            local += 1;
                        }
                        tower += 1;
                    }
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
    }
}

/// `m mod q` on the diagonal of a `size x size` matrix, as constant polynomials, for the
/// `W`-word integer `m`. Descriptor: `N, q, size, W, m, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_lift_int(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q, size) = (d.word(0) as usize, d.word(1), d.word(2) as usize);
        let residue = int_rem_small(&int_load(d.input(4), d.word(3) as usize), q);
        let out = d.output(5);
        let mut i = 0;
        while i < size * size * n {
            *out.add(i) = 0;
            i += 1;
        }
        let mut k = 0;
        while k < size {
            *out.add((k * size + k) * n) = residue;
            k += 1;
        }
    }
}

/// The most words of the big integers of [`mxx_threshold_decode`]: CRT bases of up to 16 limbs.
const BIG: usize = 20;

/// `a += b`, both little-endian of `BIG` words.
#[inline(always)]
fn big_add(a: &mut [u64; BIG], b: &[u64; BIG]) {
    let mut carry = 0u64;
    let mut i = 0;
    while i < BIG {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry);
        a[i] = s2;
        carry = (c1 as u64) + (c2 as u64);
        i += 1;
    }
}

/// `a = a * m + c`.
#[inline(always)]
fn big_mul_add(a: &mut [u64; BIG], m: u64, c: u64) {
    let mut carry = c as u128;
    let mut i = 0;
    while i < BIG {
        let p = a[i] as u128 * m as u128 + carry;
        a[i] = p as u64;
        carry = p >> 64;
        i += 1;
    }
}

/// Whether `a >= b`.
#[inline(always)]
fn big_ge(a: &[u64; BIG], b: &[u64; BIG]) -> bool {
    let mut i = BIG;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// `a -= b` for `a >= b`.
#[inline(always)]
fn big_sub(a: &mut [u64; BIG], b: &[u64; BIG]) {
    let mut borrow = 0u64;
    let mut i = 0;
    while i < BIG {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        a[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
        i += 1;
    }
}

/// `a << s` for `s < 64`.
#[inline(always)]
fn big_shl(a: &[u64; BIG], s: u64) -> [u64; BIG] {
    let mut out = [0u64; BIG];
    let mut i = BIG;
    while i > 0 {
        i -= 1;
        out[i] =
            if s == 0 { a[i] } else { (a[i] << s) | if i > 0 { a[i - 1] >> (64 - s) } else { 0 } };
    }
    out
}

/// The inverse of `a` modulo the prime `q`, as `a^(q-2)`.
#[inline(always)]
fn inverse_mod(a: u64, q: u64) -> u64 {
    let (mut result, mut base, mut exponent) = (1u64, a % q, q - 2);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, q);
        }
        base = mul_mod(base, base, q);
        exponent >>= 1;
    }
    result
}

/// `prod moduli[0..depth]` as a big integer.
#[inline(always)]
unsafe fn big_product(moduli: *const u64, depth: usize) -> [u64; BIG] {
    let mut product = [0u64; BIG];
    product[0] = 1;
    let mut l = 0;
    while l < depth {
        big_mul_add(&mut product, unsafe { *moduli.add(l) }, 0);
        l += 1;
    }
    product
}

/// `values[0..count]` as an array of at most 16 words.
#[inline(always)]
unsafe fn load16(values: *const u64, count: usize) -> [u64; 16] {
    let mut out = [0u64; 16];
    let mut i = 0;
    while i < count {
        out[i] = unsafe { *values.add(i) };
        i += 1;
    }
    out
}

/// For Garner's reconstruction over `moduli[0..depth]`: entry `k` is the inverse of
/// `q_0 ... q_{k-1}` modulo `q_k`.
#[inline(always)]
fn garner_inverses(moduli: &[u64; 16], depth: usize) -> [u64; 16] {
    let mut inverses = [0u64; 16];
    let mut k = 0;
    while k < depth {
        let qk = moduli[k];
        let (mut radix, mut i) = (1u64, 0);
        while i < k {
            radix = mul_mod(radix, moduli[i] % qk, qk);
            i += 1;
        }
        inverses[k] = inverse_mod(radix, qk);
        k += 1;
    }
    inverses
}

/// The integer `x` in `[0, q_0 ... q_{depth-1})` whose residue modulo `q_k` is `residue(k)`, by
/// Garner's mixed-radix reconstruction `x = v_0 + q_0 (v_1 + q_1 (v_2 + ...))`.
#[inline(always)]
fn garner(
    residue: impl Fn(usize) -> u64,
    moduli: &[u64; 16],
    depth: usize,
    inverses: &[u64; 16],
) -> [u64; BIG] {
    let mut digits = [0u64; 16];
    let mut k = 0;
    while k < depth {
        let qk = moduli[k];
        // The value of the digits so far modulo q_k.
        let (mut value, mut radix, mut i) = (0u64, 1u64, 0);
        while i < k {
            value = add_mod(value, mul_mod(digits[i] % qk, radix, qk), qk);
            radix = mul_mod(radix, moduli[i] % qk, qk);
            i += 1;
        }
        digits[k] = mul_mod(sub_mod(residue(k) % qk, value, qk), inverses[k], qk);
        k += 1;
    }
    let mut x = [0u64; BIG];
    let mut k = depth;
    while k > 0 {
        k -= 1;
        big_mul_add(&mut x, moduli[k], digits[k]);
    }
    x
}

/// `a mod q` for `q < 2^63`.
#[inline(always)]
fn big_rem_small(a: &[u64; BIG], q: u64) -> u64 {
    let (mut r, mut i) = (0u64, BIG);
    while i > 0 {
        i -= 1;
        r = reduce(r, a[i], q);
    }
    r
}

/// `(a / b, a mod b)` for `b > 0`, by restoring division one bit at a time.
#[inline(always)]
fn big_div_rem(a: &[u64; BIG], b: &[u64; BIG]) -> ([u64; BIG], [u64; BIG]) {
    let (mut quotient, mut remainder) = ([0u64; BIG], [0u64; BIG]);
    let mut top = BIG;
    while top > 0 && a[top - 1] == 0 {
        top -= 1;
    }
    let mut bit = top * 64;
    while bit > 0 {
        bit -= 1;
        remainder = big_shl(&remainder, 1);
        remainder[0] |= (a[bit / 64] >> (bit % 64)) & 1;
        if big_ge(&remainder, b) {
            big_sub(&mut remainder, b);
            quotient[bit / 64] |= 1 << (bit % 64);
        }
    }
    (quotient, remainder)
}

/// `floor((2 t x + Q) / (2 Q)) mod t`, which is `round(t x / Q) mod t` with ties up, for `x` in
/// `[0, Q)` and `t < 2^63`, so the quotient fits 64 bits.
#[inline(always)]
fn round_scale(x: &[u64; BIG], t: u64, q_big: &[u64; BIG], twice_q: &[u64; BIG]) -> u64 {
    let mut numerator = *x;
    big_mul_add(&mut numerator, 2 * t, 0);
    big_add(&mut numerator, q_big);
    let mut quotient = 0u64;
    let mut bit = 64u64;
    while bit > 0 {
        bit -= 1;
        let shifted = big_shl(twice_q, bit);
        // The shift must not lose high words: with t < 2^63 the quotient is below 2^64.
        if shifted[BIG - 1] >> 63 == 0 && big_ge(&numerator, &shifted) {
            big_sub(&mut numerator, &shifted);
            quotient |= 1 << bit;
        }
    }
    quotient % t
}

/// One level of a CRT recomposition: for every coefficient `x` of a matrix over the `L` source
/// moduli (product `Q`), `round(t x / Q) mod t` times `c`, modulo each of the `M` destination
/// moduli; `c_residues[m]` is `c mod q_m`. Descriptor: `limb words, L, M, t, source limbs, source
/// moduli, destination moduli, c residues, destination limbs, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `L <= 16` and `t < 2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_round_lift(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (limb_words, depth, targets, t) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize, d.word(3));
        let (limbs, moduli, destination, c, out) =
            (d.input(4), load16(d.input(5), depth), d.input(6), d.input(7), d.output(8));
        let q_big = big_product(moduli.as_ptr(), depth);
        let inverses = garner_inverses(&moduli, depth);
        let mut twice_q = q_big;
        big_add(&mut twice_q, &q_big);
        let mut j = 0;
        while j < limb_words {
            let x = garner(|k| *limbs.add(k * limb_words + j), &moduli, depth, &inverses);
            let v = round_scale(&x, t, &q_big, &twice_q);
            let mut m = 0;
            while m < targets {
                let q = *destination.add(m);
                *out.add(m * limb_words + j) = mul_mod(v % q, *c.add(m), q);
                m += 1;
            }
            j += 1;
        }
    }
}

/// A scalar polynomial from canonical coefficient bits: coefficient `k` is
/// `sum_b bit[k B + b] 2^b` (`B` bits each, little-endian), written modulo each of the `L` moduli.
/// A coefficient at or above the ring modulus is an error of the graph, which this leaf does not
/// reproduce. Descriptor: `N, B, L, bits, moduli, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `B <= 64 BIG`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_pack_bits(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, bits, depth) = (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize);
        let (input, moduli, out) = (d.input(3), d.input(4), d.output(5));
        let mut k = 0;
        while k < n {
            let mut value = [0u64; BIG];
            let mut b = 0;
            while b < bits {
                value[b / 64] |= (*input.add(k * bits + b) & 1) << (b % 64);
                b += 1;
            }
            let mut l = 0;
            while l < depth {
                *out.add(l * n + k) = big_rem_small(&value, *moduli.add(l));
                l += 1;
            }
            k += 1;
        }
    }
}

/// For coefficients `0..length` of entry `(0, 0)` of a matrix over the CRT basis `moduli` (primes
/// below `2^63`), `round(t x / Q) mod t` where `x` in `[0, Q)` is the coefficient, computed
/// exactly: `x` is rebuilt from its residues (Garner), and the rounding is
/// `floor((2 t x + Q) / (2 Q))`, whose quotient fits 64 bits, as `W`-word integers. Descriptor:
/// `N, L, t, length, limb words, W, matrix, moduli, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `L <= 16`, `t < 2^63`, and `length <= N`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_threshold_decode(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (depth, t, length, limb_words) =
            (d.word(1) as usize, d.word(2), d.word(3) as usize, d.word(4) as usize);
        let w = d.word(5) as usize;
        let (matrix, moduli, out) = (d.input(6), d.input(7), d.output(8));
        let q_big = big_product(moduli, depth);
        let inverses = garner_inverses(&load16(moduli, depth), depth);
        let mut twice_q = q_big;
        big_add(&mut twice_q, &q_big);
        let mut j = 0;
        while j < length {
            let numerator = garner(
                |k| *matrix.add(k * limb_words + j),
                &load16(moduli, depth),
                depth,
                &inverses,
            );
            let quotient = round_scale(&numerator, t, &q_big, &twice_q);
            int_store(out.add(j * w), w, &big_word(quotient % t));
            j += 1;
        }
    }
}

/// `[0; BIG]` with its low word `value`.
#[inline(always)]
fn big_word(value: u64) -> [u64; BIG] {
    let mut out = [0u64; BIG];
    out[0] = value;
    out
}

/// `(a - 1) / 2`, which is `floor(a / 2)` for odd `a`.
#[inline(always)]
fn big_half(a: &[u64; BIG]) -> [u64; BIG] {
    let mut half = *a;
    big_sub(&mut half, &big_word(1));
    let mut i = 0;
    while i < BIG {
        half[i] = (half[i] >> 1) | if i + 1 < BIG { half[i + 1] << 63 } else { 0 };
        i += 1;
    }
    half
}

/// The magnitude of the centered value of `x` in `[0, modulus)`, and whether it is negative: the
/// centered value is `x - modulus` when `x > half = (modulus - 1) / 2`, and `x` otherwise.
#[inline(always)]
fn big_centered(x: &[u64; BIG], modulus: &[u64; BIG], half: &[u64; BIG]) -> ([u64; BIG], bool) {
    if big_ge(half, x) {
        (*x, false)
    } else {
        let mut magnitude = *modulus;
        big_sub(&mut magnitude, x);
        (magnitude, true)
    }
}

/// An exact coefficientwise map from a matrix over the CRT basis `source` (`L` odd primes below
/// `2^63`, product `Q`) to one over the basis `destination` (`M` primes, product `Q'`). Each
/// coefficient `x` in `[0, Q)` is rebuilt (Garner); its centered value is `x - Q` when
/// `x > (Q - 1) / 2` and `x` otherwise. With `D` the `big` input (`BIG` little-endian words), by
/// `mode`:
///
/// - 0, modulus switch (`destination` a subset of `source`): `floor((x Q' + floor(Q / 2)) / Q)`;
/// - 1, centered rebase: the centered value;
/// - 2, centered round division: the centered value divided by `D > 0`, rounded to the nearest
///   integer with ties toward positive infinity;
/// - 3, block modulus switch (`destination` a strict subset, `P` the product of the dropped primes,
///   `D = t` invertible modulo `P`): `(x - t r) / P`, where `r` is the centered value of `x t^-1
///   mod P` over `P`.
///
/// The result is reduced modulo each destination prime. Descriptor: `limb words, L, M, mode,
/// source limbs, source moduli, destination moduli, big, destination limbs, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `L, M <= 16`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_crt_map(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (limb_words, depth, targets, mode) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize, d.word(3));
        let (limbs, out) = (d.input(4), d.output(8));
        let (source, destination) = (load16(d.input(5), depth), load16(d.input(6), targets));
        let mut big = [0u64; BIG];
        let mut w = 0;
        while w < BIG {
            big[w] = *d.input(7).add(w);
            w += 1;
        }
        // The dropped source primes, and where each destination prime is in the source.
        let (mut dropped, mut dropped_count, mut retained) = ([0u64; 16], 0, [0usize; 16]);
        let mut l = 0;
        while l < depth {
            let (mut kept, mut m) = (false, 0);
            while m < targets {
                if destination[m] == source[l] {
                    kept = true;
                    retained[m] = l;
                }
                m += 1;
            }
            if !kept {
                dropped[dropped_count] = source[l];
                dropped_count += 1;
            }
            l += 1;
        }
        let (q_big, inverses) =
            (big_product(source.as_ptr(), depth), garner_inverses(&source, depth));
        let half = big_half(&q_big);
        let p_big = big_product(dropped.as_ptr(), dropped_count);
        let (p_half, p_inverses) = (big_half(&p_big), garner_inverses(&dropped, dropped_count));
        // Mode 3: t^-1 modulo each dropped prime; t and P^-1 modulo each destination prime.
        let (mut t_inverse, mut t_target, mut p_inverse) = ([0u64; 16], [0u64; 16], [0u64; 16]);
        if mode == 3 {
            let mut k = 0;
            while k < dropped_count {
                t_inverse[k] = inverse_mod(big_rem_small(&big, dropped[k]), dropped[k]);
                k += 1;
            }
            let mut m = 0;
            while m < targets {
                t_target[m] = big_rem_small(&big, destination[m]);
                p_inverse[m] = inverse_mod(big_rem_small(&p_big, destination[m]), destination[m]);
                m += 1;
            }
        }
        let mut j = 0;
        while j < limb_words {
            let residue = |l: usize| *limbs.add(l * limb_words + j);
            if mode == 3 {
                // The dropped residues of x t^-1, in the order of `dropped`.
                let mut scaled = [0u64; 16];
                let (mut k, mut l) = (0, 0);
                while l < depth {
                    if k < dropped_count && source[l] == dropped[k] {
                        scaled[k] = mul_mod(residue(l) % dropped[k], t_inverse[k], dropped[k]);
                        k += 1;
                    }
                    l += 1;
                }
                let r = garner(|k| scaled[k], &dropped, dropped_count, &p_inverses);
                let (magnitude, negative) = big_centered(&r, &p_big, &p_half);
                let mut m = 0;
                while m < targets {
                    let q = destination[m];
                    let mut c = big_rem_small(&magnitude, q);
                    if negative && c != 0 {
                        c = q - c;
                    }
                    let numerator =
                        sub_mod(residue(retained[m]) % q, mul_mod(t_target[m], c, q), q);
                    *out.add(m * limb_words + j) = mul_mod(numerator, p_inverse[m], q);
                    m += 1;
                }
                j += 1;
                continue;
            }
            let x = garner(residue, &source, depth, &inverses);
            let (magnitude, negative) = if mode == 0 {
                // With x = a P + b: x Q' + floor(Q/2) = a Q + (b Q' + floor(Q/2)), and the second
                // term is below 2Q, so the quotient is a or a + 1.
                let (mut a, b) = big_div_rem(&x, &p_big);
                // b Q' < Q fits.
                let (mut t, mut m) = (b, targets);
                while m > 0 {
                    m -= 1;
                    big_mul_add(&mut t, destination[m], 0);
                }
                big_add(&mut t, &half);
                if big_ge(&t, &q_big) {
                    big_add(&mut a, &big_word(1));
                }
                (a, false)
            } else {
                let (mut magnitude, negative) = big_centered(&x, &q_big, &half);
                if mode == 2 {
                    let (mut quotient, remainder) = big_div_rem(&magnitude, &big);
                    let mut doubled = remainder;
                    big_add(&mut doubled, &remainder);
                    let up =
                        if negative { !big_ge(&big, &doubled) } else { big_ge(&doubled, &big) };
                    if up {
                        big_add(&mut quotient, &big_word(1));
                    }
                    magnitude = quotient;
                }
                (magnitude, negative)
            };
            let mut m = 0;
            while m < targets {
                let q = destination[m];
                let r = big_rem_small(&magnitude, q);
                *out.add(m * limb_words + j) = if negative && r != 0 { q - r } else { r };
                m += 1;
            }
            j += 1;
        }
    }
}

/// Fast (approximate) RNS base conversion, as OpenFHE's: the source primes are split into groups,
/// and for each group `g`, destination prime `q`, and coefficient,
/// `s = sum_{i in g} centered(z_i scale_i mod p_i) weight_{i,q} mod q`, where `centered(u)` is
/// `u` or `u - p_i` for `u > (p_i - 1) / 2`. Primes of scale 0 take no part. With `down`, the
/// output is instead `(s t + z_{retained(q)}) inverse_q mod q`. The tables are public constants:
/// `group` and `scale` per source prime, `weight` source-major (`L x M`), and `down` per
/// destination prime as `t mod q, inverse_q, retained index`. Output limb `m` holds the groups
/// one after another, each `limb words` words. Descriptor: `limb words, L, M, groups, down, source
/// limbs, source moduli, destination moduli, group, scale, weight, down table, destination limbs,
/// scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with every group below `groups`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_fast_base_conversion(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (limb_words, depth, targets, groups, down) = (
            d.word(0) as usize,
            d.word(1) as usize,
            d.word(2) as usize,
            d.word(3) as usize,
            d.word(4) != 0,
        );
        let (limbs, source, destination) = (d.input(5), d.input(6), d.input(7));
        let (group, scale, weight, down_table, out) =
            (d.input(8), d.input(9), d.input(10), d.input(11), d.output(12));
        let mut m = 0;
        while m < targets {
            let q = *destination.add(m);
            let mut g = 0;
            while g < groups {
                let mut j = 0;
                while j < limb_words {
                    let mut sum = 0u64;
                    let mut i = 0;
                    while i < depth {
                        let s = *scale.add(i);
                        if s != 0 && *group.add(i) as usize == g {
                            let p = *source.add(i);
                            let u = mul_mod(*limbs.add(i * limb_words + j) % p, s, p);
                            let centered = if u <= p / 2 {
                                u % q
                            } else {
                                let magnitude = (p - u) % q;
                                if magnitude == 0 { 0 } else { q - magnitude }
                            };
                            sum =
                                add_mod(sum, mul_mod(centered, *weight.add(i * targets + m), q), q);
                        }
                        i += 1;
                    }
                    if down {
                        let (t, inverse, retained) = (
                            *down_table.add(3 * m),
                            *down_table.add(3 * m + 1),
                            *down_table.add(3 * m + 2) as usize,
                        );
                        sum = add_mod(
                            mul_mod(sum, t, q),
                            *limbs.add(retained * limb_words + j) % q,
                            q,
                        );
                        sum = mul_mod(sum, inverse, q);
                    }
                    *out.add((m * groups + g) * limb_words + j) = sum;
                    j += 1;
                }
                g += 1;
            }
            m += 1;
        }
    }
}

/// The negacyclic NTT of `count` polynomials of one limb, in place of OpenFHE's
/// `ForwardTransformToBitReverseInPlace` (`inverse` 0) or its inverse (`inverse` 1). `table` holds
/// `psi^brv(k)` (forward) or `psi^-brv(k)` (inverse) for `k < N`, with `brv` the `log2 N`-bit
/// reversal and `psi` the 2N-th root of unity of the limb, so the forward transform's slot `i` is
/// the polynomial at `psi^(2 brv(i) + 1)`. `n_inverse` is `N^-1 mod q`. Descriptor: `N, q,
/// count, inverse, n_inverse, values, table, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `N` a power of two and `q < 2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_ntt(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, q, count, inverse, n_inverse) =
            (d.word(0) as usize, d.word(1), d.word(2) as usize, d.word(3) != 0, d.word(4));
        let (values, table, out) = (d.input(5), d.input(6), d.output(7));
        let mut p = 0;
        while p < count {
            let a = out.add(p * n);
            let mut j = 0;
            while j < n {
                *a.add(j) = *values.add(p * n + j) % q;
                j += 1;
            }
            if !inverse {
                // Cooley-Tukey, natural to bit-reversed order.
                let (mut t, mut m) = (n, 1);
                while m < n {
                    t >>= 1;
                    let mut i = 0;
                    while i < m {
                        let s = *table.add(m + i);
                        let mut j = 2 * i * t;
                        while j < 2 * i * t + t {
                            let (u, v) = (*a.add(j), mul_mod(*a.add(j + t), s, q));
                            *a.add(j) = add_mod(u, v, q);
                            *a.add(j + t) = sub_mod(u, v, q);
                            j += 1;
                        }
                        i += 1;
                    }
                    m <<= 1;
                }
            } else {
                // Gentleman-Sande, bit-reversed to natural order.
                let (mut t, mut m) = (1, n);
                while m > 1 {
                    let h = m >> 1;
                    let mut i = 0;
                    while i < h {
                        let s = *table.add(h + i);
                        let mut j = 2 * i * t;
                        while j < 2 * i * t + t {
                            let (u, v) = (*a.add(j), *a.add(j + t));
                            *a.add(j) = add_mod(u, v, q);
                            *a.add(j + t) = mul_mod(sub_mod(u, v, q), s, q);
                            j += 1;
                        }
                        i += 1;
                    }
                    t <<= 1;
                    m = h;
                }
                let mut j = 0;
                while j < n {
                    *a.add(j) = mul_mod(*a.add(j), n_inverse, q);
                    j += 1;
                }
            }
            p += 1;
        }
    }
}

/// The values `x` in `[0, Q)` of `count` CRT coefficients as `W`-word integers, `Q` the product
/// of the `L` moduli (below `2^(64 W - 1)`). Descriptor: `count, L, W, limbs, moduli, out,
/// scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `L <= 16` and `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_crt_values(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (count, depth, w) = (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize);
        let (limbs, moduli, out) = (d.input(3), load16(d.input(4), depth), d.output(5));
        let inverses = garner_inverses(&moduli, depth);
        let mut j = 0;
        while j < count {
            let x = garner(|k| *limbs.add(k * count + j), &moduli, depth, &inverses);
            int_store(out.add(j * w), w, &x);
            j += 1;
        }
    }
}

/// The residues of `count` `W`-word integers modulo each of `L` moduli, one limb per modulus.
/// Descriptor: `count, L, W, values, moduli, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_int_residues(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (count, depth, w) = (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize);
        let (values, moduli, out) = (d.input(3), d.input(4), d.output(5));
        let mut j = 0;
        while j < count {
            let v = int_load(values.add(j * w), w);
            let mut l = 0;
            while l < depth {
                *out.add(l * count + j) = int_rem_small(&v, *moduli.add(l));
                l += 1;
            }
            j += 1;
        }
    }
}

/// The wrapping `W`-word integer product of the row-major `outer x inner` matrix `M` (or, with
/// `transpose`, the `inner x outer` matrix whose transpose is used) with a vector of `inner`
/// integers: `out[i] = sum_j M[i, j] v[j]`, or `sum_j v[j] M[j, i]`. Descriptor: `outer, inner,
/// transpose, W, matrix, vector, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_int_mat_vec(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (outer, inner, transpose, w) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) != 0, d.word(3) as usize);
        let (matrix, vector, out) = (d.input(4), d.input(5), d.output(6));
        let mut i = 0;
        while i < outer {
            let mut sum = [0u64; BIG];
            let mut j = 0;
            while j < inner {
                let entry = if transpose { j * outer + i } else { i * inner + j };
                let product = int_mul(
                    &int_load(matrix.add(entry * w), w),
                    &int_load(vector.add(j * w), w),
                    w,
                );
                big_add(&mut sum, &product);
                j += 1;
            }
            int_store(out.add(i * w), w, &sum);
            i += 1;
        }
    }
}

/// The round constants of Keccak-f[1600].
const KECCAK_ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// The rotation offsets of Keccak-f[1600], by lane `x + 5 y`.
const KECCAK_ROTATIONS: [u32; 25] =
    [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14];

fn keccak_f(a: &mut [u64; 25]) {
    let mut round = 0;
    while round < 24 {
        let mut c = [0u64; 5];
        let mut x = 0;
        while x < 5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
            x += 1;
        }
        let mut x = 0;
        while x < 5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            let mut y = 0;
            while y < 25 {
                a[x + y] ^= d;
                y += 5;
            }
            x += 1;
        }
        let mut b = [0u64; 25];
        let mut x = 0;
        while x < 5 {
            let mut y = 0;
            while y < 5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] =
                    a[x + 5 * y].rotate_left(KECCAK_ROTATIONS[x + 5 * y]);
                y += 1;
            }
            x += 1;
        }
        let mut y = 0;
        while y < 25 {
            let mut x = 0;
            while x < 5 {
                a[x + y] = b[x + y] ^ (!b[(x + 1) % 5 + y] & b[(x + 2) % 5 + y]);
                x += 1;
            }
            y += 5;
        }
        a[0] ^= KECCAK_ROUND_CONSTANTS[round];
        round += 1;
    }
}

/// Keccak-256 with the original padding (`0x01 ... 0x80`), as `keccak_asm::Keccak256`.
#[derive(Clone, Copy)]
struct Keccak256 {
    state: [u64; 25],
    /// Bytes absorbed into the current block.
    position: usize,
}

const KECCAK_RATE: usize = 136;

impl Keccak256 {
    fn new() -> Self {
        Self { state: [0; 25], position: 0 }
    }

    #[inline(always)]
    fn absorb_byte(&mut self, byte: u8) {
        let lane = self.position / 8;
        self.state[lane] ^= (byte as u64) << (8 * (self.position % 8));
        self.position += 1;
        if self.position == KECCAK_RATE {
            keccak_f(&mut self.state);
            self.position = 0;
        }
    }

    unsafe fn update(&mut self, bytes: *const u8, length: usize) {
        let mut i = 0;
        while i < length {
            self.absorb_byte(unsafe { *bytes.add(i) });
            i += 1;
        }
    }

    fn update_u64(&mut self, value: u64) {
        // Bytes through memory, not shifts: extracting a byte by shifting lets the compiler choose
        // a 32-bit shift, which the machine lacks.
        let bytes = value.to_le_bytes();
        let mut i = 0;
        while i < 8 {
            self.absorb_byte(bytes[i]);
            i += 1;
        }
    }

    fn finalize(mut self, out: &mut [u8; 32]) {
        let (lane, last) = (self.position / 8, (KECCAK_RATE - 1) / 8);
        self.state[lane] ^= 0x01u64 << (8 * (self.position % 8));
        self.state[last] ^= 0x80u64 << (8 * ((KECCAK_RATE - 1) % 8));
        keccak_f(&mut self.state);
        let mut lane = 0;
        while lane < 4 {
            let bytes = self.state[lane].to_le_bytes();
            let mut i = 0;
            while i < 8 {
                out[8 * lane + i] = bytes[i];
                i += 1;
            }
            lane += 1;
        }
    }
}

/// A matrix of residues uniform modulo `Q`, derived from a 32-byte key and a tag exactly as the
/// CPU backend's hash sampler does: coefficient `k` of entry `(i, j)` is the first candidate
/// below `Q` among `Keccak256(key || tag || i || j || k || attempt || block)` streams (each index a
/// little-endian `u64`), `bytes_per_coefficient` bytes per candidate with the top byte masked by
/// `high_mask`. Descriptor: `N, L, rows, columns, bytes_per_coefficient, high_mask, limb_words,
/// tag_words, modulus_words, key, tag, moduli, modulus_bytes, out, scratch`; the output is `L`
/// limbs of `limb_words` words, the tag buffer is its byte length in one word and then its bytes
/// (see [`mxx_hash_tag`]), and `tag_words` and `modulus_words` are the sizes of the tag and
/// modulus buffers.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `bytes_per_coefficient <= 128` and moduli
/// below `2^63`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_hash_uniform(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (n, depth, rows, columns) =
            (d.word(0) as usize, d.word(1) as usize, d.word(2) as usize, d.word(3) as usize);
        let (width, high_mask, limb_words) =
            (d.word(4) as usize, d.word(5) as u8, d.word(6) as usize);
        let (key, tag, moduli, modulus) =
            (d.input(9) as *const u8, d.input(10), d.input(11), d.input(12) as *const u8);
        let out = d.output(13);
        let mut prefix = Keccak256::new();
        prefix.update(key, 32);
        prefix.update((tag as *const u8).add(8), *tag as usize);
        let mut sample = [0u8; 128];
        let mut digest = [0u8; 32];
        let mut i = 0;
        while i < rows {
            let mut j = 0;
            while j < columns {
                let mut entry = prefix;
                entry.update_u64(i as u64);
                entry.update_u64(j as u64);
                let mut k = 0;
                while k < n {
                    let mut attempt = 0u64;
                    loop {
                        let (mut filled, mut block) = (0usize, 0u64);
                        while filled < width {
                            let mut h = entry;
                            h.update_u64(k as u64);
                            h.update_u64(attempt);
                            h.update_u64(block);
                            h.finalize(&mut digest);
                            let mut t = 0;
                            while t < 32 && filled < width {
                                sample[filled] = digest[t];
                                filled += 1;
                                t += 1;
                            }
                            block += 1;
                        }
                        sample[width - 1] &= high_mask;
                        // Accept when the little-endian candidate is below Q.
                        let mut below = false;
                        let mut t = width;
                        while t > 0 {
                            t -= 1;
                            let (c, m) = (sample[t], *modulus.add(t));
                            if c != m {
                                below = c < m;
                                break;
                            }
                        }
                        if below {
                            break;
                        }
                        attempt += 1;
                    }
                    let mut l = 0;
                    while l < depth {
                        let q = *moduli.add(l);
                        let mut residue = 0u64;
                        let mut t = width;
                        while t > 0 {
                            t -= 1;
                            residue =
                                add_mod(mul_mod(residue, 256 % q, q), sample[t] as u64 % q, q);
                        }
                        *out.add(l * limb_words + (i * columns + j) * n + k) = residue;
                        l += 1;
                    }
                    k += 1;
                }
                j += 1;
            }
            i += 1;
        }
    }
}

/// `count` integers uniform on `[0, 2^b)`, as the CPU backend's hash integer family: integer `k`
/// is the candidate of coefficient `k` of entry `(0, 0)` of [`mxx_hash_uniform`]'s stream (attempt
/// 0, which a power-of-two modulus always accepts), `bytes_per_coefficient` bytes with the top
/// byte masked by `high_mask`, as a `W`-word integer. Descriptor: `count, bytes_per_coefficient,
/// high_mask, tag_words, W, key, tag, out, scratch`, the tag as [`mxx_hash_uniform`]'s.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `bytes_per_coefficient < 8 W` and `W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_hash_ints(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (count, width, high_mask) = (d.word(0) as usize, d.word(1) as usize, d.word(2) as u8);
        let w = d.word(4) as usize;
        let (key, tag, out) = (d.input(5) as *const u8, d.input(6), d.output(7));
        let mut entry = Keccak256::new();
        entry.update(key, 32);
        entry.update((tag as *const u8).add(8), *tag as usize);
        entry.update_u64(0);
        entry.update_u64(0);
        let mut digest = [0u8; 32];
        let mut k = 0;
        while k < count {
            let mut value = [0u64; BIG];
            let (mut filled, mut block) = (0usize, 0u64);
            while filled < width {
                let mut h = entry;
                h.update_u64(k as u64);
                h.update_u64(0);
                h.update_u64(block);
                h.finalize(&mut digest);
                let mut t = 0;
                while t < 32 && filled < width {
                    let byte = if filled + 1 == width { digest[t] & high_mask } else { digest[t] };
                    value[filled / 8] |= (byte as u64) << (8 * (filled % 8));
                    filled += 1;
                    t += 1;
                }
                block += 1;
            }
            int_store(out.add(k * w), w, &value);
            k += 1;
        }
    }
}

/// A hash tag: its byte length in one word, then its bytes, built from a template of `u64` words
/// and a family of `W`-word integer operands. The template is a sequence of items: `[0, n, bytes]`
/// appends `n` literal bytes (packed little-endian into words); `[1, k]` appends operand `k` as an
/// integer component (`1`, its sign byte, its magnitude's byte length as a big-endian `u64`, and
/// the magnitude big-endian without leading zeros, at least one byte); `[2, k]` as a decimal
/// component (`2`, the
/// length of its decimal text as a big-endian `u64`, and the text); `[3, k]` as a little-endian
/// component (`3` and its low word, little-endian). Descriptor: `W, template words, out words,
/// operand count, template, operands, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with an output large enough for the tag and
/// `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_hash_tag(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (w, template_words) = (d.word(0) as usize, d.word(1) as usize);
        let (template, operands, out) = (d.input(4), d.input(5), d.output(6));
        let bytes = (out as *mut u8).add(8);
        let mut length = 0usize;
        let mut push = |byte: u8| {
            *bytes.add(length) = byte;
            length += 1;
        };
        let mut at = 0;
        while at < template_words {
            let kind = *template.add(at);
            if kind == 0 {
                let n = *template.add(at + 1) as usize;
                let mut i = 0;
                while i < n {
                    push((*template.add(at + 2 + i / 8) >> (8 * (i % 8))) as u8);
                    i += 1;
                }
                at += 2 + n.div_ceil(8);
                continue;
            }
            let value = int_load(operands.add(*template.add(at + 1) as usize * w), w);
            at += 2;
            match kind {
                1 => {
                    let magnitude = int_abs(&value);
                    let mut top = 8 * BIG;
                    while top > 1 && (magnitude[(top - 1) / 8] >> (8 * ((top - 1) % 8))) as u8 == 0
                    {
                        top -= 1;
                    }
                    push(1);
                    push(int_negative(&value) as u8);
                    let mut i = 8;
                    while i > 0 {
                        i -= 1;
                        push(((top as u64) >> (8 * i)) as u8);
                    }
                    while top > 0 {
                        top -= 1;
                        push((magnitude[top / 8] >> (8 * (top % 8))) as u8);
                    }
                }
                2 => {
                    // The decimal digits, least significant first.
                    let mut magnitude = int_abs(&value);
                    let mut digits = [0u8; 400];
                    let mut count = 0;
                    loop {
                        let (q, r) = big_div_rem(&magnitude, &big_word(10));
                        digits[count] = b'0' + r[0] as u8;
                        count += 1;
                        magnitude = q;
                        if magnitude == [0u64; BIG] {
                            break;
                        }
                    }
                    let negative = int_negative(&value);
                    let text = count + negative as usize;
                    push(2);
                    let mut i = 8;
                    while i > 0 {
                        i -= 1;
                        push(((text as u64) >> (8 * i)) as u8);
                    }
                    if negative {
                        push(b'-');
                    }
                    while count > 0 {
                        count -= 1;
                        push(digits[count]);
                    }
                }
                _ => {
                    push(3);
                    let mut i = 0;
                    while i < 8 {
                        push((value[0] >> (8 * i)) as u8);
                        i += 1;
                    }
                }
            }
        }
        *out = length as u64;
    }
}

/// The correctly rounded square root of a non-negative binary64 (its bits), with integer
/// arithmetic only: for `x = m 2^(e - 52)` with `e` even, `sqrt(x) = sqrt(m 2^52) 2^(e/2 - 52)`,
/// and the integer square root `q` of `m 2^52` (53 bits) rounds up exactly when the remainder
/// exceeds `q`, since no square root of an integer lies halfway.
#[inline(always)]
fn sqrt_bits(x: u64) -> u64 {
    let (exponent, fraction) = ((x >> 52) & 0x7ff, x & ((1 << 52) - 1));
    if x << 1 == 0 || exponent == 0x7ff {
        return x;
    }
    let (mut m, mut e) = if exponent == 0 {
        (fraction, -1022i64)
    } else {
        (fraction | (1 << 52), exponent as i64 - 1023)
    };
    while m < 1 << 52 {
        m <<= 1;
        e -= 1;
    }
    if e & 1 != 0 {
        m <<= 1;
        e -= 1;
    }
    // Bit by bit: the largest q with q^2 <= n.
    let n = (m as u128) << 52;
    let (mut q, mut rem, mut bit) = (0u128, n, 1u128 << 106);
    while bit > n {
        bit >>= 2;
    }
    while bit != 0 {
        if rem >= q + bit {
            rem -= q + bit;
            q = (q >> 1) + bit;
        } else {
            q >>= 1;
        }
        bit >>= 2;
    }
    let mut q = q as u64;
    if rem > q as u128 {
        q += 1;
    }
    let mut half = e / 2;
    if q == 1 << 53 {
        q >>= 1;
        half += 1;
    }
    (((half + 1023) as u64) << 52) | (q & ((1 << 52) - 1))
}

/// An IEEE 754 binary64 operation, as the host computes it: `op` 0 adds, 1 subtracts, 2
/// multiplies, 3 divides, and 4 is the square root of `a` (`b` unused). A zero divisor, a negative
/// square root, or a non-finite result is an error of the graph, which this leaf does not
/// reproduce. Descriptor: `op, a, b, out, scratch`, each real one word of its bits.
///
/// # Safety
///
/// `descriptor` must follow the layout above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_real_op(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (a, b) = (f64::from_bits(*d.input(1)), f64::from_bits(*d.input(2)));
        *d.output(3) = match d.word(0) {
            0 => a + b,
            1 => a - b,
            2 => a * b,
            3 => a / b,
            _ => f64::from_bits(sqrt_bits(a.to_bits())),
        }
        .to_bits();
    }
}

/// A `W`-word integer as the nearest binary64 (ties to even), as `num_bigint`'s `to_f64`, which
/// rounds the top 64 bits to odd and then to nearest. Descriptor: `W, a, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with `1 <= W <= 20`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_int_to_real(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let a = int_load(d.input(1), d.word(0) as usize);
        let (magnitude, negative) = (int_abs(&a), int_negative(&a));
        let mut top = BIG;
        while top > 0 && magnitude[top - 1] == 0 {
            top -= 1;
        }
        let sign = (negative as u64) << 63;
        if top == 0 {
            *d.output(2) = 0;
            return;
        }
        // The bit length, and the top 64 bits with the rest folded into the last (round to odd).
        let lead = 64 - magnitude[top - 1].leading_zeros() as usize;
        let bits = 64 * (top - 1) + lead;
        let mut m = 0u64;
        let mut sticky = false;
        let mut i = bits;
        let mut taken = 0;
        while i > 0 {
            i -= 1;
            let bit = (magnitude[i / 64] >> (i % 64)) & 1;
            if taken < 64 {
                m = (m << 1) | bit;
                taken += 1;
            } else if bit != 0 {
                sticky = true;
                break;
            }
        }
        m <<= 64 - taken;
        if sticky {
            m |= 1;
        }
        // Round the 64 bits (leading bit set) to 53, ties to even.
        let (mut mantissa, rest) = (m >> 11, m & 0x7ff);
        if rest > 0x400 || (rest == 0x400 && mantissa & 1 == 1) {
            mantissa += 1;
        }
        let mut exponent = bits as u64 - 1;
        if mantissa == 1 << 53 {
            mantissa >>= 1;
            exponent += 1;
        }
        *d.output(2) = if exponent + 1023 >= 2047 {
            sign | 0x7ff0_0000_0000_0000
        } else {
            sign | ((exponent + 1023) << 52) | (mantissa & ((1 << 52) - 1))
        };
    }
}

/// One candidate preimage under a sampled trapdoor (computational MP12 or statistical), as the CPU
/// backend samples it ([`preimage::draw`]): from a SHAKE256 stream of the 32 coin bytes of at
/// most `budget` words, with `ok` true iff the stream sufficed, a statistical trapdoor's
/// covariance factored, and every coefficient's absolute value is at most `bound`. Descriptor:
/// `N, L, rows, digits, tower_digits, columns, secret_rows, statistical, base, sigma, width,
/// bound, budget, steps, scratch_words, trapdoor, target, moduli, coins, out, ok, scratch`, with
/// `statistical` 1 for a statistical trapdoor (whose `T` has `secret_rows` rows), `sigma` (the
/// trapdoor's width) and `width` (the preimage's) the bits of binary64 values, and `steps` and
/// `scratch_words` read only by the signature.
///
/// # Safety
///
/// `descriptor` must follow the layout above over buffers of [`preimage::Shape`]'s sizes, with
/// the conditions of [`preimage::draw`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_trapdoor_preimage(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let shape = preimage::Shape {
            n: d.word(0) as usize,
            towers: d.word(1) as usize,
            rows: d.word(2) as usize,
            digits: d.word(3) as usize,
            tower_digits: d.word(4) as usize,
            columns: d.word(5) as usize,
            statistical: if d.word(7) != 0 { d.word(6) as usize } else { 0 },
        };
        let (base, sigma, width) =
            (d.word(8), f64::from_bits(d.word(9)), f64::from_bits(d.word(10)));
        let (bound, budget) = (d.word(11), d.word(12));
        let coins = d.input(18);
        let seed = [*coins, *coins.add(1), *coins.add(2), *coins.add(3)];
        let mut g = gauss::Stream::with_budget(&seed, budget);
        let mut arena = preimage::Arena::new(d.word(21) as *mut u64);
        let ok = preimage::draw(
            &shape,
            &mut g,
            base,
            sigma,
            width,
            bound,
            d.input(15),
            d.input(16),
            d.input(17),
            d.output(19),
            &mut arena,
        );
        *d.output(20) = ok as u64;
    }
}

/// Statistical TrapGen's draw of `T` and its check ([`preimage::trapgen`]), from a SHAKE256
/// stream of the 32 coin bytes of at most `budget` words. Descriptor: `N, L, m_bar, digits, base,
/// sigma, width, budget, steps, scratch_words, moduli, coins, out, ok, scratch`, with `sigma` and
/// `width` the bits of binary64 values and `steps` and `scratch_words` read only by the
/// signature.
///
/// # Safety
///
/// `descriptor` must follow the layout above, with the conditions of [`preimage::trapgen`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_statistical_trapgen(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let coins = d.input(11);
        let seed = [*coins, *coins.add(1), *coins.add(2), *coins.add(3)];
        let mut g = gauss::Stream::with_budget(&seed, d.word(7));
        let mut arena = preimage::Arena::new(d.word(14) as *mut u64);
        let ok = preimage::trapgen(
            &mut g,
            d.word(0) as usize,
            d.word(1) as usize,
            d.word(2) as usize,
            d.word(3) as usize,
            d.word(4),
            f64::from_bits(d.word(5)),
            f64::from_bits(d.word(6)),
            d.input(10),
            d.output(12),
            &mut arena,
        );
        *d.output(13) = ok as u64;
    }
}

/// `count` buffers of `words` zero words. Descriptor: `words, count, out, scratch`.
///
/// # Safety
///
/// `descriptor` must follow the layout above over an output of `words * count` words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mxx_zero(descriptor: *const u64) {
    unsafe {
        let d = Descriptor(descriptor);
        let (words, out) = (d.word(0) as usize * d.word(1) as usize, d.output(2));
        let mut i = 0;
        while i < words {
            core::ptr::write_volatile(out.add(i), 0);
            i += 1;
        }
    }
}

#[cfg(target_os = "none")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
