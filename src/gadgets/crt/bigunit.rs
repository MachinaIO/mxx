use crate::{
    circuit::{PolyCircuit, gate::GateId},
    gadgets::packed_plt::PackedPlt,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use rayon::prelude::*;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

//
// Carry-save multiplication and column compression helpers
// -------------------------------------------------------
// We represent base-B limbs with B = 2^w. Every partial product a_i * b_j is
// immediately split via public LUTs into (lo = p % B, hi = p / B) and placed
// into column banks: lo goes to column i+j, hi to column i+j+1. This ensures
// every cell flowing into compressors is < B, so a single lookup(x) =
// (x % B, x / B) remains valid for inputs < B^2.
//
// Column compression is done by a Wallace-style tree of
// compressors (summing triples per column into (digit, carry) with one lookup)
// until height <= 2, and performing exactly one lookup to
// emit (digit, carry) for that column. The final normalization to canonical
// base-B digits is deferred to the end of the entire pipeline.
//
// The public API of BigUintPoly is preserved; mul() now constructs columns,
// compresses to a carry-save pair (S, C), then performs a single ripple
// normalization to match prior outputs. A future enhancement can swap this
// ripple for a parallel-prefix CPA without changing the output shape.

type Columns = Vec<Vec<GateId>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigUintPolyContext<P: Poly> {
    pub limb_bit_size: usize,
    pub crt_idx: usize,
    pub max_degree: usize,
    pub const_zero: GateId,
    pub const_base: GateId,
    pub luts: Option<(PackedPlt<P>, PackedPlt<P>)>,
}

impl<P: Poly> BigUintPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        crt_idx: usize,
        max_degree: usize,
    ) -> Self {
        // Assume base < 2^32
        debug_assert!(limb_bit_size < 32);
        let base = 1 << limb_bit_size;
        let const_zero = circuit.const_zero_gate();
        let const_base = circuit.const_digits_poly(&[base as u32]);
        let luts = if limb_bit_size > 1 {
            let (mod_lut, floor_lut) =
                Self::setup_packed_luts(circuit, params, base, base * base, crt_idx, max_degree);
            Some((mod_lut, floor_lut))
        } else {
            None
        };
        Self { limb_bit_size, crt_idx, max_degree, const_zero, const_base, luts }
    }

    fn setup_packed_luts(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        base: usize,
        nrows: usize,
        crt_idx: usize,
        max_degree: usize,
    ) -> (PackedPlt<P>, PackedPlt<P>) {
        let entries: Vec<_> = (0..nrows)
            .into_par_iter()
            .map(|k| {
                let input = BigUint::from(k);
                let output_mod = BigUint::from(k % base);
                let output_floor = BigUint::from(k / base);
                P::from_usize_to_constant(params, k % base);
                (input, k, output_mod, output_floor)
            })
            .collect();
        let mut map_mod = HashMap::with_capacity(nrows);
        let mut map_floor = HashMap::with_capacity(nrows);
        for (input, k, output_mod, output_floor) in entries {
            map_mod.insert(input.clone(), (k, output_mod));
            map_floor.insert(input, (k, output_floor));
        }
        (
            PackedPlt::setup(circuit, params, crt_idx, max_degree, map_mod),
            PackedPlt::setup(circuit, params, crt_idx, max_degree, map_floor),
        )
    }
}

#[derive(Debug, Clone)]
pub struct BigUintPoly<P: Poly> {
    pub ctx: Arc<BigUintPolyContext<P>>,
    pub limbs: Vec<GateId>,
    _p: PhantomData<P>,
}

impl<P: Poly> BigUintPoly<P> {
    pub fn new(ctx: Arc<BigUintPolyContext<P>>, limbs: Vec<GateId>) -> Self {
        Self { ctx, limbs, _p: PhantomData }
    }

    #[inline]
    pub fn bit_size(&self) -> usize {
        self.limbs.len() * self.ctx.limb_bit_size
    }

    pub fn zero(ctx: Arc<BigUintPolyContext<P>>, bit_size: usize) -> Self {
        let limb_len = bit_size.div_ceil(ctx.limb_bit_size);
        let limbs = vec![ctx.const_zero; limb_len];
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn const_u64(
        ctx: Arc<BigUintPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        value: u64,
    ) -> Self {
        let mut limbs = vec![];
        let mut remaining_value = value;
        let base = 1u64 << ctx.limb_bit_size;
        while remaining_value > 0 {
            limbs.push(circuit.const_digits_poly(&[(remaining_value % base) as u32]));
            remaining_value /= base;
        }

        debug_assert_eq!(remaining_value, 0);
        Self { ctx, limbs, _p: PhantomData }
    }

    /// Allocate input polynomials for a BigUintPoly
    pub fn input(
        ctx: Arc<BigUintPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        input_bit_size: usize,
    ) -> Self {
        let num_limbs = input_bit_size.div_ceil(ctx.limb_bit_size);
        let limb_gateids = circuit.input(num_limbs);
        Self { ctx, limbs: limb_gateids, _p: PhantomData }
    }

    #[inline]
    pub fn extend_size(&self, new_bit_size: usize) -> Self {
        debug_assert!(new_bit_size >= self.bit_size());
        debug_assert_eq!(new_bit_size % self.ctx.limb_bit_size, 0);
        let limb_len = new_bit_size / self.ctx.limb_bit_size;
        let mut limbs = self.limbs.clone();
        limbs.resize(limb_len, self.ctx.const_zero);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);

        let (w, a, b) = if self.limbs.len() >= other.limbs.len() {
            (self.limbs.len(), &self.limbs, &other.limbs)
        } else {
            (other.limbs.len(), &other.limbs, &self.limbs)
        };

        let zero = circuit.const_zero_gate();
        let mut ss = Vec::with_capacity(w);
        let mut gs = Vec::with_capacity(w);
        let mut ps = Vec::with_capacity(w);

        let one = circuit.const_one_gate();
        for i in 0..w {
            let ai = a[i];
            let bi = if i < b.len() { b[i] } else { self.ctx.const_zero };
            let (s, g) = match self.ctx.luts.as_ref() {
                None => {
                    let s = circuit.xor_gate(ai, bi);
                    let g = circuit.and_gate(ai, bi);
                    (s, g)
                }
                Some((mod_lut, floor_lut)) => {
                    let t = circuit.add_gate(ai, bi);
                    let s = mod_lut.lookup_all(circuit, t);
                    let g = floor_lut.lookup_all(circuit, t);
                    (s, g)
                }
            };

            // p = 1 iff s == B-1, i.e., floor((s + 1)/B) = 1
            let p = match self.ctx.luts.as_ref() {
                None => s,
                Some((_, floor_lut)) => {
                    let s_plus = circuit.add_gate(s, one);
                    floor_lut.lookup_all(circuit, s_plus)
                }
            };
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }

        // Parallel-prefix (Kogge–Stone) on (g,p)
        let (g_pref, _) = Self::prefix_gp(circuit, &gs, &ps);

        // Compute final limbs: s_i = (s'_i + c_i) mod B
        let mut limbs = Vec::with_capacity(w + 1);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            match self.ctx.luts.as_ref() {
                None => {
                    // For single bit: (s'_i + c_i) mod 2 = s'_i XOR c_i
                    let digit = circuit.xor_gate(ss[i], carry_in);
                    limbs.push(digit);
                }
                Some((mod_lut, _)) => {
                    // For multi-bit: use lookup table for modular reduction.
                    let t = circuit.add_gate(ss[i], carry_in);
                    let digit = mod_lut.lookup_all(circuit, t);
                    limbs.push(digit);
                }
            }
        }
        let last_carry = if w == 0 { zero } else { g_pref[w - 1] };
        limbs.push(last_carry);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn less_than(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> (GateId, Self) {
        // Parallel-prefix borrow-lookahead comparator with O(log t) depth.
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        debug_assert_eq!(self.ctx, other.ctx);

        let w = self.limbs.len();
        let zero = circuit.const_zero_gate();
        let mut g = Vec::with_capacity(w);
        let mut p = Vec::with_capacity(w);
        let mut ab_pairs = Vec::new();

        let one = circuit.const_one_gate();
        let base_minus_one = {
            let b_minus_1 = (1u32 << self.ctx.limb_bit_size) - 1;
            circuit.const_digits_poly(&[b_minus_1])
        };

        // For each limb i, compute t_i = a_i + (B-1) - b_i.
        // Then: h_i = floor(t_i / B) is 1 iff a_i >= b_i, 0 otherwise.
        //        s_i = t_i % B; s_i == B-1 iff a_i == b_i.
        // Borrow generate g_i = (a_i < b_i) = (!h_i) & (!eq_i)
        // Borrow propagate p_i = (a_i == b_i) = eq_i
        for i in 0..w {
            let a = self.limbs[i];
            let b = other.limbs[i];
            let (g_i, p_i) = match self.ctx.luts.as_ref() {
                None => {
                    // g_i = !a_i AND b_i (borrow generate: a < b).
                    let not_a = circuit.not_gate(a);
                    let gi = circuit.and_gate(not_a, b);
                    // p_i = !(a_i XOR b_i) (borrow propagate: a == b).
                    let xor_ab = circuit.xor_gate(a, b);
                    let pi = circuit.not_gate(xor_ab);
                    (gi, pi)
                }
                Some((mod_lut, floor_lut)) => {
                    let t0 = circuit.add_gate(a, base_minus_one);
                    let t = circuit.sub_gate(t0, b);
                    let s = mod_lut.lookup_all(circuit, t);
                    let h = floor_lut.lookup_all(circuit, t);
                    let not_h = circuit.not_gate(h);
                    // eq_i: s == B-1 <=> floor((s + 1)/B) == 1
                    let s_plus = circuit.add_gate(s, one);
                    let eq = floor_lut.lookup_all(circuit, s_plus);
                    let not_eq = circuit.not_gate(eq);
                    let gi_and = circuit.and_gate(not_h, not_eq);
                    (gi_and, eq)
                }
            };
            g.push(g_i);
            p.push(p_i);
            ab_pairs.push((a, b));
        }

        // Parallel-prefix (Kogge–Stone) on (g,p) - same for both cases.
        let (g_pref, _) = Self::prefix_gp(circuit, &g, &p);

        // Compute final difference limbs: d_i = (a_i + B - b_i - b_{in,i}) mod B
        let mut diff_limbs = Vec::with_capacity(w);
        for i in 0..w {
            let b_in = if i == 0 { zero } else { g_pref[i - 1] };
            let (a, b) = ab_pairs[i];
            match self.ctx.luts.as_ref() {
                None => {
                    let xor = circuit.xor_gate(a, b);
                    let diff = circuit.xor_gate(xor, b_in);
                    diff_limbs.push(diff);
                }
                Some((mod_lut, _)) => {
                    let (a, b) = ab_pairs[i];
                    let pre = circuit.add_gate(a, self.ctx.const_base);
                    let pre2 = circuit.sub_gate(pre, b);
                    let t = circuit.sub_gate(pre2, b_in);
                    let d = mod_lut.lookup_all(circuit, t);
                    diff_limbs.push(d);
                }
            }
        }
        let borrow_out = if w == 0 { zero } else { g_pref[w - 1] };
        (borrow_out, Self { ctx: self.ctx.clone(), limbs: diff_limbs, _p: PhantomData })
    }

    pub fn mul(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_bit_size: Option<usize>,
    ) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);

        let max_bit_size = max_bit_size.unwrap_or(self.bit_size() + other.bit_size());
        debug_assert!(max_bit_size.is_multiple_of(self.ctx.limb_bit_size));
        let max_limbs = max_bit_size / self.ctx.limb_bit_size;
        let (sum_vec, carry_vec) = self.mul_without_cpa(other, circuit, max_limbs);

        // Final single normalization (ripple-style) to match the existing API.
        // Note: We normalize S + (C shifted by 1) with exactly one pass of lookups.
        let limbs = self.final_cpa(circuit, sum_vec, carry_vec, max_limbs);

        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    #[inline]
    pub fn left_shift(&self, shift: usize) -> Self {
        debug_assert!(shift < self.limbs.len());
        let limbs = self.limbs[shift..].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    #[inline]
    pub fn mod_limbs(&self, num_limbs: usize) -> Self {
        debug_assert!(num_limbs <= self.limbs.len());
        let limbs = self.limbs[0..num_limbs].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    // return self if selector is 1, other if selector is 0
    pub fn cmux(&self, other: &Self, selector: GateId, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let not = circuit.not_gate(selector);
        for i in 0..self.limbs.len() {
            let case1 = circuit.mul_gate(self.limbs[i], selector);
            let case2 = circuit.mul_gate(other.limbs[i], not);
            let cmuxed = circuit.add_gate(case1, case2);
            limbs.push(cmuxed);
        }
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    // return a gate id of an integeter corresponding to the big-integer representation of `limbs`.
    // namely, `out = (limbs[0] + 2^{limb_bit_size} * limbs[1] + ... + 2^{limb_bit_size * (k-1)} *
    // limbs[k-1]) * (q/q_i)`
    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        debug_assert!(!self.limbs.is_empty(), "limbs should not be empty");

        let mut result = self.limbs[0];

        for i in 1..self.limbs.len() {
            // Create BigUint for 2^{limb_bit_size * i}
            let power_exponent = self.ctx.limb_bit_size * i;
            let power_of_two = BigUint::from(1u32) << power_exponent;

            let weighted_limb = circuit.large_scalar_mul(self.limbs[i], &[power_of_two]);
            result = circuit.add_gate(result, weighted_limb);
        }
        result
    }

    #[inline]
    fn schoolbook_partial_products_columns(
        &self,
        circuit: &mut PolyCircuit<P>,
        a: &[GateId],
        b: &[GateId],
        max_limbs: usize,
    ) -> Columns {
        // Columns sized up to max_limbs; carries to the last column beyond max_limbs are dropped.
        let mut columns: Columns = vec![vec![]; max_limbs];
        let pairs: Vec<(usize, usize)> = (0..a.len())
            .flat_map(|i| (0..b.len()).map(move |j| (i, j)))
            .filter(|(i, j)| i + j < max_limbs)
            .collect();
        for (i, j) in pairs {
            let k = i + j;
            let prod = circuit.mul_gate(a[i], b[j]);
            match self.ctx.luts.as_ref() {
                None => {
                    columns[k].push(prod);
                }
                Some((mod_lut, floor_lut)) => {
                    let lo = mod_lut.lookup_all(circuit, prod);
                    columns[k].push(lo);
                    if k + 1 < max_limbs {
                        let hi = floor_lut.lookup_all(circuit, prod);
                        columns[k + 1].push(hi);
                    }
                }
            };
        }
        columns
    }

    // Wallace tree using compressors per column until height <= 2
    #[inline]
    fn compress_columns_wallace(
        &self,
        circuit: &mut PolyCircuit<P>,
        columns: &mut Columns,
    ) -> (Vec<GateId>, Vec<GateId>) {
        let w = columns.len();
        if w == 0 {
            return (vec![], vec![]);
        }
        // Choose a safe compressor arity per column.
        // Each cell is < B, and we reduce up to comp_rate cells in one shot
        // using a single lookup pair (x % B, x / B) defined for inputs < B^2.
        // Safety condition: comp_rate * (B-1) <= B^2 - 1  ==> comp_rate <= B + 1.
        // To guarantee convergence for B=2 (limb_bit_size=1), use 3:2 compression.
        let base = 1usize << self.ctx.limb_bit_size;
        let comp_rate = if base == 2 { 3 } else { base + 1 };

        loop {
            let w = columns.len();
            let mut next: Columns = vec![vec![]; w + 1];
            for k in 0..w {
                let col = &columns[k];
                let mut idx = 0;
                while idx < col.len() {
                    let group_len = comp_rate.min(col.len() - idx);
                    if group_len <= 2 {
                        // Preserve <= 2 items
                        next[k].extend_from_slice(&col[idx..idx + group_len]);
                    } else {
                        // Compress this chunk with one lookup pair.
                        let (digit, carry) = match self.ctx.luts.as_ref() {
                            None => {
                                let (a, b, c) = (col[idx], col[idx + 1], col[idx + 2]);
                                // digit = a xor b xor c
                                // carry = (a xor b) and c or (a and b)
                                let xor_ab = circuit.xor_gate(a, b);
                                let digit = circuit.xor_gate(xor_ab, c); // sum mod 2
                                let and_ab = circuit.and_gate(a, b);
                                let and_abc = circuit.and_gate(xor_ab, c);
                                let carry = circuit.or_gate(and_ab, and_abc); // carry out
                                (digit, carry)
                            }
                            Some((mod_lut, floor_lut)) => {
                                let mut sum = col[idx];
                                for i in (idx + 1)..(idx + group_len) {
                                    sum = circuit.add_gate(sum, col[i]);
                                }
                                let digit = mod_lut.lookup_all(circuit, sum); // x % B
                                let carry = floor_lut.lookup_all(circuit, sum); // x / B
                                (digit, carry)
                            }
                        };
                        next[k].push(digit);
                        next[k + 1].push(carry);
                    }
                    idx += group_len;
                }
            }
            if next.last().map_or(false, |v| v.is_empty()) {
                next.pop();
            }
            let need_more = next.iter().any(|col| col.len() > 2);
            *columns = next;
            if !need_more {
                break;
            }
        }

        // with height <= 2 per column, split any pair into (digit, carry).
        let w = columns.len();
        let zero = circuit.const_zero_gate();
        let mut sum_vec = Vec::with_capacity(w);
        let mut carry_vec = vec![zero; w + 1];

        for k in 0..w {
            match columns[k].as_slice() {
                [] => sum_vec.push(zero),
                [x] => sum_vec.push(*x),
                [x, y] => {
                    let (digit, carry) = match self.ctx.luts.as_ref() {
                        None => {
                            let digit = circuit.xor_gate(*x, *y);
                            let carry = circuit.and_gate(*x, *y);
                            (digit, carry)
                        }
                        Some((mod_lut, floor_lut)) => {
                            let s = circuit.add_gate(*x, *y);
                            let digit = mod_lut.lookup_all(circuit, s);
                            let carry = floor_lut.lookup_all(circuit, s);
                            (digit, carry)
                        }
                    };
                    sum_vec.push(digit);
                    carry_vec[k + 1] = carry;
                }
                _ => unreachable!("column height should be <= 2 after compression"),
            }
        }
        (sum_vec, carry_vec)
    }

    // Final normalization by a parallel-prefix CPA: add S and C (shifted) once to produce digits.
    #[inline]
    pub(crate) fn final_cpa(
        &self,
        circuit: &mut PolyCircuit<P>,
        mut sum_vec: Vec<GateId>,
        carry_vec: Vec<GateId>,
        max_limbs: usize,
    ) -> Vec<GateId> {
        // Ensure width
        let w = sum_vec.len().min(max_limbs);
        sum_vec.truncate(w);
        // Precompute t_k = S_k + C_k
        let zero = circuit.const_zero_gate();
        let one = circuit.const_one_gate();
        let mut ss = Vec::with_capacity(w);
        let mut gs = Vec::with_capacity(w);
        let mut ps = Vec::with_capacity(w);
        for k in 0..w {
            let c = carry_vec.get(k).copied().unwrap_or(zero);
            let (s, g, p) = match self.ctx.luts.as_ref() {
                None => {
                    let s = circuit.xor_gate(sum_vec[k], c);
                    let g = circuit.and_gate(sum_vec[k], c);
                    let p = s;
                    (s, g, p)
                }
                Some((mod_lut, floor_lut)) => {
                    let t = circuit.add_gate(sum_vec[k], c);
                    let s = mod_lut.lookup_all(circuit, t);
                    let g = floor_lut.lookup_all(circuit, t);
                    // p = 1 iff s == B-1 <=> floor((s + 1)/B) = 1
                    let s_plus = circuit.add_gate(s, one);
                    let p = floor_lut.lookup_all(circuit, s_plus);
                    (s, g, p)
                }
            };
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }
        let (g_pref, _) = Self::prefix_gp(circuit, &gs, &ps);
        let mut out = Vec::with_capacity(w);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let digit = match self.ctx.luts.as_ref() {
                None => circuit.xor_gate(ss[i], carry_in),
                Some((mod_lut, _)) => {
                    let t = circuit.add_gate(ss[i], carry_in);
                    mod_lut.lookup_all(circuit, t)
                }
            };
            out.push(digit);
        }
        out
    }

    // Kogge–Stone parallel prefix on (g, p)
    #[inline]
    fn prefix_gp(
        circuit: &mut PolyCircuit<P>,
        g: &[GateId],
        p: &[GateId],
    ) -> (Vec<GateId>, Vec<GateId>) {
        let w = g.len();
        let mut gs = g.to_vec();
        let mut ps = p.to_vec();
        let mut d = 1usize;
        while d < w {
            let mut gs_next = gs.clone();
            let mut ps_next = ps.clone();
            for k in 0..w {
                if k >= d {
                    let gj = gs[k - d];
                    let pj = ps[k - d];
                    let gk = gs[k];
                    let pk = ps[k];
                    // G' = gk OR (pk AND gj); P' = pk AND pj
                    let pk_and_gj = circuit.and_gate(pk, gj);
                    let g_new = circuit.or_gate(gk, pk_and_gj);
                    let p_new = circuit.and_gate(pk, pj);
                    gs_next[k] = g_new;
                    ps_next[k] = p_new;
                }
            }
            gs = gs_next;
            ps = ps_next;
            d <<= 1;
        }
        (gs, ps)
    }

    #[inline]
    pub(crate) fn mul_without_cpa(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_limbs: usize,
    ) -> (Vec<GateId>, Vec<GateId>) {
        // 1) Schoolbook partial products with immediate split and column placement
        let mut columns =
            self.schoolbook_partial_products_columns(circuit, &self.limbs, &other.limbs, max_limbs);

        // 2) Compress column
        self.compress_columns_wallace(circuit, &mut columns)
    }
}

pub fn u64_vec_to_biguint_poly<P: Poly>(
    limb_bit_size: usize,
    crt_idx: usize,
    params: &P::Params,
    inputs: &[u64],
    num_limbs: Option<usize>,
) -> Vec<P> {
    let ring_n = params.ring_dimension() as usize;
    let base = 1u64 << limb_bit_size;

    // Determine total limbs to output
    let mut max_needed_limbs = 1usize;
    for &v in inputs.iter() {
        let mut x = v;
        let mut cnt = 0usize;
        while x > 0 {
            cnt += 1;
            x /= base;
        }
        if cnt == 0 {
            cnt = 1;
        }
        if cnt > max_needed_limbs {
            max_needed_limbs = cnt;
        }
    }
    let total_limbs = num_limbs.map(|n| n.max(max_needed_limbs)).unwrap_or(max_needed_limbs);

    // Build slots for each limb as a ring_n-sized vector with first `degree` populated
    let mut limb_slots: Vec<Vec<BigUint>> = vec![vec![BigUint::zero(); ring_n]; total_limbs];
    for (idx, &v0) in inputs.iter().enumerate() {
        let mut v = v0;
        for limb_idx in 0..total_limbs {
            let digit = (v % base) as u64;
            limb_slots[limb_idx][idx] = BigUint::from(digit);
            v /= base;
        }
    }

    limb_slots
        .iter()
        .map(|slots| P::from_biguints_eval_single_mod(params, crt_idx, slots))
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use std::sync::Arc;

    const INPUT_BIT_SIZE: usize = 20;
    // LIMB_BIT_SIZE works if INPUT_BIT_SIZE % LIMB_BIT_SIZE = 0. (3 doesn't works)
    const LIMB_BIT_SIZE: usize = 3;
    const LIMB_LEN: usize = INPUT_BIT_SIZE.div_ceil(LIMB_BIT_SIZE);
    const CRT_IDX: usize = 1;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<BigUintPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            circuit,
            &params,
            LIMB_BIT_SIZE,
            CRT_IDX,
            params.ring_dimension() as usize,
        ));
        (params, ctx)
    }

    #[test]
    fn test_biguint_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [15, 1, 8, 9];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [20, 11, 4, 6];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let result = big_a.add(&big_b, &mut circuit).mod_limbs(LIMB_LEN + 1);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_sums =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN + 1];
        for i in 0..d {
            for j in 0..LIMB_LEN + 1 {
                if expected_sums[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_sums[i] % (1u64 << ctx.limb_bit_size));
                expected_sums[i] /= 1u64 << ctx.limb_bit_size;
            }
        }
        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_add_with_carry() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause carry for some slots
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [(1u64 << INPUT_BIT_SIZE) - 1, 123, (1u64 << INPUT_BIT_SIZE) - 2, 0];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [1, 456, 3, 0];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let result = big_a.add(&big_b, &mut circuit).mod_limbs(LIMB_LEN + 1);

        // Scale outputs to the selected CRT slot for direct comparison
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_sums =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN + 1];
        for i in 0..d {
            for j in 0..LIMB_LEN + 1 {
                if i >= expected_sums.len() || expected_sums[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_sums[i] % (1u64 << ctx.limb_bit_size));
                expected_sums[i] /= 1u64 << ctx.limb_bit_size;
            }
        }
        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_less_than_smaller() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a < b (500 < 1000), so less_than should return 1 (true)
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [500, 7, 10, 1];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [1000, 8, 20, 2];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let mut output_gates = vec![circuit.large_scalar_mul(lt_result, &scalar)];
        for limb in diff.limbs.into_iter() {
            output_gates.push(circuit.large_scalar_mul(limb, &scalar));
        }
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        // Expected lt_result is all ones in active slots
        let d = params.ring_dimension() as usize;
        let mut ones = vec![BigUint::zero(); d];
        for i in 0..a_u64s.len() {
            if a_u64s[i] < b_u64s[i] {
                ones[i] = BigUint::from(1u32);
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_lt =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &ones) * &q_over_qi;
        assert_eq!(eval_result[0], expected_lt);
    }

    #[test]
    fn test_biguint_less_than_equal() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a == b in all slots, so less_than should return 0 (false)
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [12345, 0, 1, 999];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s = a_u64s;
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let mut output_gates = vec![circuit.large_scalar_mul(lt_result, &scalar)];
        for limb in diff.limbs.into_iter() {
            output_gates.push(circuit.large_scalar_mul(limb, &scalar));
        }
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let d = params.ring_dimension() as usize;
        let zeros = vec![BigUint::zero(); d];
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_lt =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &zeros) * &q_over_qi;
        assert_eq!(eval_result[0], expected_lt);
        // Diff should be zero in all limbs
        for i in 1..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &zeros) * &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_less_than_greater() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a > b in all slots
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [1000, 15, 100, 7];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [500, 10, 99, 3];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let mut output_gates = vec![circuit.large_scalar_mul(lt_result, &scalar)];
        for limb in diff.limbs.clone().into_iter() {
            output_gates.push(circuit.large_scalar_mul(limb, &scalar));
        }
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        // Expected lt_result is all zeros
        let d = params.ring_dimension() as usize;
        let zeros = vec![BigUint::zero(); d];
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_lt =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &zeros) * &q_over_qi;
        assert_eq!(eval_result[0], expected_lt);

        // Expected diff = a - b per slot
        let mut diffs = a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a - b).collect::<Vec<_>>();
        let mut expected_cols = vec![vec![BigUint::zero(); d]; LIMB_LEN];
        for i in 0..d {
            for j in 0..LIMB_LEN {
                if i >= diffs.len() || diffs[i] == 0 {
                    break;
                }
                expected_cols[j][i] = BigUint::from(diffs[i] % (1u64 << ctx.limb_bit_size));
                diffs[i] /= 1u64 << ctx.limb_bit_size;
            }
        }
        for i in 0..LIMB_LEN {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_cols[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i + 1], expected_poly);
        }
    }

    #[test]
    fn test_biguint_mul_simple() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [123, 5, 17, 0];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [456, 3, 2, 7];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let result = big_a.mul(&big_b, &mut circuit, Some(LIMB_LEN * LIMB_BIT_SIZE));
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_products =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
        let max_bit_size = LIMB_LEN * LIMB_BIT_SIZE;
        let output_limb_len = max_bit_size / LIMB_BIT_SIZE;
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; output_limb_len];
        for i in 0..d {
            for j in 0..output_limb_len {
                if i >= expected_products.len() || expected_products[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_products[i] % (1u64 << ctx.limb_bit_size));
                expected_products[i] /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), output_limb_len);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_mul_with_overflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause overflow with 20-bit input size
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [1023, 2047, 1, 0];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [1023, 1023, 2, 0];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let result = big_a.mul(&big_b, &mut circuit, Some(LIMB_LEN * LIMB_BIT_SIZE));
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_products =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
        let max_bit_size = LIMB_LEN * LIMB_BIT_SIZE;
        let output_limb_len = max_bit_size / LIMB_BIT_SIZE;
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; output_limb_len];
        for i in 0..d {
            for j in 0..output_limb_len {
                if i >= expected_products.len() || expected_products[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_products[i] % (1u64 << ctx.limb_bit_size));
                expected_products[i] /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), output_limb_len);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let zero = BigUintPoly::zero(ctx.clone(), INPUT_BIT_SIZE);
        circuit.output(zero.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &[], Some(plt_evaluator));

        assert_eq!(eval_result.len(), LIMB_LEN);

        for limb_result in eval_result {
            let limb_coeffs = limb_result.coeffs();
            assert_eq!(*limb_coeffs[0].value(), 0u32.into());
        }
    }

    #[test]
    fn test_biguint_extend_size() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a_full = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [12345, 7, 0, 999];
        let a_value =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_a = big_a_full.mod_limbs(LIMB_LEN);

        // Extend to next multiple of LIMB_BIT_SIZE after INPUT_BIT_SIZE
        let new_bit_size = ((INPUT_BIT_SIZE / LIMB_BIT_SIZE) + 1) * LIMB_BIT_SIZE;
        let extended = big_a.extend_size(new_bit_size);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = extended
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_value].concat(),
            Some(plt_evaluator),
        );

        let extended_limb_len = new_bit_size / LIMB_BIT_SIZE; // limb +1 from extend_size
        assert_eq!(eval_result.len(), extended_limb_len);

        // Check that the original values are preserved in the first limbs
        let d = params.ring_dimension() as usize;
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; extended_limb_len];
        for i in 0..d {
            let mut v = if i < a_u64s.len() { a_u64s[i] } else { 0 };
            for j in 0..LIMB_LEN {
                if v == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(v % (1u64 << ctx.limb_bit_size));
                v /= 1u64 << ctx.limb_bit_size;
            }
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_add_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [100, 10, 5, 3];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));

        // For big_b, we're using a smaller bit size
        let b_bit_size = 2 * LIMB_BIT_SIZE;
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, b_bit_size);
        let b_limb_len = b_bit_size / ctx.limb_bit_size;
        // Choose values that fit within b_bit_size
        let b_u64s: [u64; 4] = [50, 1, 2, 0];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(b_limb_len));

        let result = big_a.add(&big_b, &mut circuit).mod_limbs(LIMB_LEN + 1);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_sums =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN + 1];
        for i in 0..d {
            for j in 0..LIMB_LEN + 1 {
                if i >= expected_sums.len() || expected_sums[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_sums[i] % (1u64 << ctx.limb_bit_size));
                expected_sums[i] /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_mul_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [100, 3, 7, 8];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));

        // For big_b, we're using a smaller bit size
        let b_bit_size = 2 * LIMB_BIT_SIZE;
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, b_bit_size);
        let b_limb_len = b_bit_size / ctx.limb_bit_size;
        let b_u64s: [u64; 4] = [50, 2, 0, 9];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(b_limb_len));

        let result = big_a.mul(&big_b, &mut circuit, None);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let d = params.ring_dimension() as usize;
        let mut expected_products =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
        let max_bit_size = INPUT_BIT_SIZE + b_bit_size;
        let output_limb_len = max_bit_size.div_ceil(LIMB_BIT_SIZE);
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; output_limb_len];
        for i in 0..d {
            for j in 0..output_limb_len {
                if i >= expected_products.len() || expected_products[i] == 0 {
                    break;
                }
                expected_limbs[j][i] =
                    BigUint::from(expected_products[i] % (1u64 << ctx.limb_bit_size));
                expected_products[i] /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), output_limb_len);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_left_shift() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [12345, 7, 0, 999];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        // After shift, restrict to the first LIMB_LEN - 1 limbs for assertion
        let shifted = big_a.left_shift(1);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = shifted
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        // Left shift by 1 means removing the first limb
        let d = params.ring_dimension() as usize;
        let expected_vals = a_u64s.map(|x| x);
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN];
        for i in 0..d {
            let mut v = if i < expected_vals.len() { expected_vals[i] } else { 0 };
            for j in 0..LIMB_LEN {
                if v == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(v % (1u64 << ctx.limb_bit_size));
                v /= 1u64 << ctx.limb_bit_size;
            }
        }
        // After shifting by 1, we expect to see limbs[1..] from the original
        assert_eq!(eval_result.len(), LIMB_LEN - 1);

        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i + 1]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_cmux() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [123, 7, 88, 5];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [456, 6, 22, 9];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let selector = circuit.input(1)[0];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        // selector per slot: [1,0,1,0]
        let d = params.ring_dimension() as usize;
        let mut sel_slots = vec![BigUint::zero(); d];
        for (idx, &v) in [1u64, 0, 1, 0].iter().enumerate() {
            sel_slots[idx] = BigUint::from(v);
        }
        let selector_value =
            vec![DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &sel_slots)];
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // Expected per slot selection
        let mut expected_vals = [0u64; 4];
        for i in 0..4 {
            expected_vals[i] = if i % 2 == 0 { a_u64s[i] } else { b_u64s[i] };
        }
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN];
        for i in 0..d {
            let mut v = if i < expected_vals.len() { expected_vals[i] } else { 0 };
            for j in 0..LIMB_LEN {
                if v == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(v % (1u64 << ctx.limb_bit_size));
                v /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), LIMB_LEN);
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_cmux_select_other() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a_u64s: [u64; 4] = [123, 7, 88, 5];
        let a =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &a_u64s, Some(LIMB_LEN));
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let b_u64s: [u64; 4] = [456, 6, 22, 9];
        let b =
            u64_vec_to_biguint_poly(ctx.limb_bit_size, CRT_IDX, &params, &b_u64s, Some(LIMB_LEN));
        let selector = circuit.input(1)[0];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        // selector per slot: all zeros
        let d = params.ring_dimension() as usize;
        let zeros_slots = vec![BigUint::zero(); d];
        let selector_value =
            vec![DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &zeros_slots)];
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // Expected per slot selection equals b
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; LIMB_LEN];
        for i in 0..d {
            let mut v = if i < b_u64s.len() { b_u64s[i] } else { 0 };
            for j in 0..LIMB_LEN {
                if v == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(v % (1u64 << ctx.limb_bit_size));
                v /= 1u64 << ctx.limb_bit_size;
            }
        }

        assert_eq!(eval_result.len(), LIMB_LEN);
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_biguint_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let test_values: [u64; 4] = [12345, 7, 0, 999];
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a = u64_vec_to_biguint_poly(
            ctx.limb_bit_size,
            CRT_IDX,
            &params,
            &test_values,
            Some(LIMB_LEN),
        );
        let finalized = big_a.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected = vec![BigUint::zero(); d];
        for i in 0..test_values.len() {
            expected[i] = BigUint::from(test_values[i]);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_biguint_finalize_large_value() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that span multiple limbs
        let test_values: [u64; 4] = [1_048_575, 1, 2, (1 << 19)];
        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, INPUT_BIT_SIZE);
        let a = u64_vec_to_biguint_poly(
            ctx.limb_bit_size,
            CRT_IDX,
            &params,
            &test_values,
            Some(LIMB_LEN),
        );
        let finalized = big_a.finalize(&mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let out = circuit.large_scalar_mul(finalized, &scalar);
        circuit.output(vec![out]);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected = vec![BigUint::zero(); d];
        for i in 0..test_values.len() {
            expected[i] = BigUint::from(test_values[i]);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_single_bit_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            &mut circuit,
            &params,
            1, // limb_bit_size = 1
            CRT_IDX,
            params.ring_dimension() as usize,
        ));

        // SIMD cases: element-wise add
        let a_u64s: [u64; 4] = [15, 1, 8, 9];
        let b_u64s: [u64; 4] = [20, 11, 4, 6];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 8);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 8);

        let a = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &a_u64s, Some(8));
        let b = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &b_u64s, Some(8));

        let result = big_a.add(&big_b, &mut circuit).mod_limbs(9);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            None::<PolyPltEvaluator>,
        );

        let d = params.ring_dimension() as usize;
        let mut expected_sums =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; 9];
        for i in 0..d {
            for j in 0..9 {
                if i >= expected_sums.len() || expected_sums[i] == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(expected_sums[i] % 2);
                expected_sums[i] /= 2;
            }
        }
        assert_eq!(eval_result.len(), 9);
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }

    #[test]
    fn test_single_bit_less_than() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            &mut circuit,
            &params,
            1, // limb_bit_size = 1
            CRT_IDX,
            params.ring_dimension() as usize,
        ));

        // SIMD: choose mixed cases
        let a_u64s: [u64; 4] = [10, 25, 30, 3];
        let b_u64s: [u64; 4] = [25, 10, 30, 4];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 8);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 8);

        let a = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &a_u64s, Some(8));
        let b = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &b_u64s, Some(8));

        let (lt_result, _diff) = big_a.less_than(&big_b, &mut circuit);
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let scaled_lt = circuit.large_scalar_mul(lt_result, &scalar);
        circuit.output(vec![scaled_lt]);
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            None::<PolyPltEvaluator>,
        );

        assert_eq!(eval_result.len(), 1);
        let d = params.ring_dimension() as usize;
        let mut expected = vec![BigUint::zero(); d];
        for i in 0..a_u64s.len() {
            expected[i] = BigUint::from((a_u64s[i] < b_u64s[i]) as u32);
        }
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        let expected_poly =
            DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected) * &q_over_qi;
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_single_bit_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            &mut circuit,
            &params,
            1, // limb_bit_size = 1
            CRT_IDX,
            params.ring_dimension() as usize,
        ));

        // SIMD: element-wise mul in base-2
        let a_u64s: [u64; 4] = [7, 2, 3, 1];
        let b_u64s: [u64; 4] = [5, 3, 4, 1];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 4);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 4);

        let a = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &a_u64s, Some(4));
        let b = u64_vec_to_biguint_poly(1, CRT_IDX, &params, &b_u64s, Some(4));

        let result = big_a.mul(&big_b, &mut circuit, Some(8));
        let (moduli, _, _) = params.to_crt();
        let q_over_qi = params.modulus().as_ref() / BigUint::from(moduli[CRT_IDX]);
        let scalar = [q_over_qi.clone()];
        let outs = result
            .limbs
            .into_iter()
            .map(|l| circuit.large_scalar_mul(l, &scalar))
            .collect::<Vec<_>>();
        circuit.output(outs);

        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            None::<PolyPltEvaluator>,
        );

        let d = params.ring_dimension() as usize;
        let mut expected_products =
            a_u64s.iter().zip(b_u64s.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
        let mut expected_limbs = vec![vec![BigUint::zero(); d]; 8];
        for i in 0..d {
            for j in 0..8 {
                if i >= expected_products.len() || expected_products[i] == 0 {
                    break;
                }
                expected_limbs[j][i] = BigUint::from(expected_products[i] % 2);
                expected_products[i] /= 2;
            }
        }

        assert_eq!(eval_result.len(), 8);
        let q_over_qi = DCRTPoly::from_biguint_to_constant(&params, q_over_qi);
        for i in 0..eval_result.len() {
            let expected_poly =
                DCRTPoly::from_biguints_eval_single_mod(&params, CRT_IDX, &expected_limbs[i]) *
                    &q_over_qi;
            assert_eq!(eval_result[i], expected_poly);
        }
    }
}
