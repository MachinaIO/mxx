use crate::{
    circuit::{PolyCircuit, gate::GateId},
    lookup::PublicLut,
    poly::Poly,
};
use num_bigint::BigUint;
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
// Column compression is done either by a Wallace-style tree of
// compressors (summing triples per column into (digit, carry) with one lookup)
// until height <= 2, or in one-shot when H_max < B by summing all values in a
// column linearly (add gates are linear) and performing exactly one lookup to
// emit (digit, carry) for that column. The final normalization to canonical
// base-B digits is deferred to the end of the entire pipeline.
//
// The public API of BigUintPoly is preserved; mul() now constructs columns,
// compresses to a carry-save pair (S, C), then performs a single ripple
// normalization to match prior outputs. A future enhancement can swap this
// ripple for a parallel-prefix CPA without changing the output shape.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigUintPolyContext<P: Poly> {
    pub limb_bit_size: usize,
    pub const_zero: GateId,
    pub const_base: GateId,
    pub lut_ids: (usize, usize),
    _p: PhantomData<P>,
}

impl<P: Poly> BigUintPolyContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, params: &P::Params, limb_bit_size: usize) -> Self {
        let base = 1 << limb_bit_size;
        // Assume base < 2^32
        debug_assert!(limb_bit_size < 32);
        let const_zero = circuit.const_zero_gate();
        let const_base = circuit.const_digits_poly(&[base as u32]);
        let mul_luts = Self::setup_split_lut(params, base, base * base);
        let lut_ids = (
            circuit.register_public_lookup(mul_luts.0),
            circuit.register_public_lookup(mul_luts.1),
        );
        // Note: Use a single pair of LUTs (split into (x % base, x / base)) for all operations.
        Self { limb_bit_size, const_zero, const_base, lut_ids, _p: PhantomData }
    }

    fn setup_split_lut(
        params: &P::Params,
        base: usize,
        nrows: usize,
    ) -> (PublicLut<P>, PublicLut<P>) {
        let mut f = HashMap::with_capacity(nrows);
        let mut g = HashMap::with_capacity(nrows);
        for k in 0..nrows {
            let input = P::from_usize_to_constant(params, k);
            let output_f = P::from_usize_to_constant(params, k % base);
            let output_g = P::from_usize_to_constant(params, k / base);
            f.insert(input.clone(), (k, output_f));
            g.insert(input, (k, output_g));
        }
        (PublicLut::new(f), PublicLut::new(g))
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

    pub fn bit_size(&self) -> usize {
        self.limbs.len() * self.ctx.limb_bit_size
    }

    pub fn zero(ctx: Arc<BigUintPolyContext<P>>, bit_size: usize) -> Self {
        debug_assert_eq!(bit_size % ctx.limb_bit_size, 0);
        let limb_len = bit_size / ctx.limb_bit_size;
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

        // Build per-column t_k = a_k + b_k, then s_k, g_k, p_k
        let mut ss = Vec::with_capacity(w);
        let mut gs = Vec::with_capacity(w);
        let mut ps = Vec::with_capacity(w);
        let one = circuit.const_one_gate();
        for i in 0..w {
            let ai = a[i];
            let bi = if i < b.len() { b[i] } else { self.ctx.const_zero };
            let t = circuit.add_gate(ai, bi);
            let s = circuit.public_lookup_gate(t, self.ctx.lut_ids.0);
            let g = circuit.public_lookup_gate(t, self.ctx.lut_ids.1); // in {0,1}
            // p = 1 iff s == B-1, i.e., floor((s + 1)/B) = 1
            let s_plus = circuit.add_gate(s, one);
            let p = circuit.public_lookup_gate(s_plus, self.ctx.lut_ids.1);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }

        // Parallel-prefix (Kogge–Stone) on (g,p)
        let (g_pref, _) = Self::prefix_gp(circuit, &gs, &ps);

        // carry_in for limb i is g_pref[i-1], with c0 = 0
        let zero = circuit.const_zero_gate();
        let mut limbs = Vec::with_capacity(w + 1);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let t = circuit.add_gate(ss[i], carry_in);
            let digit = circuit.public_lookup_gate(t, self.ctx.lut_ids.0);
            limbs.push(digit);
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
        let one = circuit.const_one_gate();
        let base_minus_one = {
            let b_minus_1 = ((1u32 << self.ctx.limb_bit_size) - 1) as u32;
            circuit.const_digits_poly(&[b_minus_1])
        };

        // For each limb i, compute t_i = a_i + (B-1) - b_i.
        // Then: h_i = floor(t_i / B) is 1 iff a_i > b_i, 0 otherwise.
        //        s_i = t_i % B; s_i == B-1 iff a_i == b_i.
        // Borrow generate g_i = (a_i < b_i) = (!h_i) & (!eq_i)
        // Borrow propagate p_i = (a_i == b_i) = eq_i
        let mut g = Vec::with_capacity(w);
        let mut p = Vec::with_capacity(w);
        for i in 0..w {
            let a = self.limbs[i];
            let b = other.limbs[i];
            let t0 = circuit.add_gate(a, base_minus_one);
            let t = circuit.sub_gate(t0, b);
            let s = circuit.public_lookup_gate(t, self.ctx.lut_ids.0);
            let h = circuit.public_lookup_gate(t, self.ctx.lut_ids.1);
            let not_h = circuit.not_gate(h);
            // eq_i: s == B-1 <=> floor((s + 1)/B) == 1
            let s_plus = circuit.add_gate(s, one);
            let eq = circuit.public_lookup_gate(s_plus, self.ctx.lut_ids.1);
            let not_eq = circuit.not_gate(eq);
            let gi_and = circuit.and_gate(not_h, not_eq);
            // p_i = eq_i (since eq_i=1 implies h_i=0, so (!h_i) is redundant)
            g.push(gi_and);
            p.push(eq);
        }

        // Prefix on (g,p) to compute borrows: b_{i+1} = g_i OR (p_i AND b_i), with b_0 = 0.
        let (g_pref, _) = Self::prefix_gp(circuit, &g, &p);

        // Final difference digits using borrow-in b_i and constant base: a_i + B - b_i - b_in
        let mut diff_limbs = Vec::with_capacity(w);
        for i in 0..w {
            let a = self.limbs[i];
            let b = other.limbs[i];
            let pre = circuit.add_gate(a, self.ctx.const_base);
            let pre2 = circuit.sub_gate(pre, b);
            let b_in = if i == 0 { zero } else { g_pref[i - 1] };
            let t = circuit.sub_gate(pre2, b_in);
            let d = circuit.public_lookup_gate(t, self.ctx.lut_ids.0);
            diff_limbs.push(d);
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
        debug_assert!(max_bit_size % self.ctx.limb_bit_size == 0);
        let max_limbs = max_bit_size / self.ctx.limb_bit_size;
        let (sum_vec, carry_vec) = self.mul_without_cpa(other, circuit, max_limbs);

        // 3) Final single normalization (ripple-style) to match the existing API
        // Note: We normalize S + (C shifted by 1) with exactly one pass of lookups.
        let limbs = Self::final_cpa(circuit, sum_vec, carry_vec, self.ctx.lut_ids, max_limbs);

        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn left_shift(&self, shift: usize) -> Self {
        debug_assert!(shift < self.limbs.len());
        let limbs = self.limbs[shift..].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

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
    // namely, `out = limbs[0] + 2^{limb_bit_size} * limbs[1] + ... + 2^{limb_bit_size * (k-1)} *
    // limbs[k-1]`
    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        debug_assert!(!self.limbs.is_empty(), "limbs should not be empty");

        let mut result = self.limbs[0];

        for i in 1..self.limbs.len() {
            // Create BigUint for 2^{limb_bit_size * i}
            let power_exponent = self.ctx.limb_bit_size * i;
            let power_of_two = BigUint::from(1u32) << power_exponent;

            let weighted_limb = circuit.large_scalar_mul(self.limbs[i], vec![power_of_two]);
            result = circuit.add_gate(result, weighted_limb);
        }

        result
    }
}

type Columns = Vec<Vec<GateId>>;

impl<P: Poly> BigUintPoly<P> {
    #[inline]
    fn schoolbook_partial_products_columns(
        circuit: &mut PolyCircuit<P>,
        a: &[GateId],
        b: &[GateId],
        max_limbs: usize,
        lut_ids: (usize, usize),
    ) -> Columns {
        // Columns sized up to max_limbs; carries to the last column beyond max_limbs are dropped.
        let mut columns: Columns = vec![vec![]; max_limbs];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let k = i + j;
                if k >= max_limbs {
                    continue;
                }
                let prod = circuit.mul_gate(ai, bj);
                let lo = circuit.public_lookup_gate(prod, lut_ids.0);
                columns[k].push(lo);
                if k + 1 < max_limbs {
                    let hi = circuit.public_lookup_gate(prod, lut_ids.1);
                    columns[k + 1].push(hi);
                }
            }
        }
        columns
    }

    // Wallace tree using compressors per column until height <= 2
    fn compress_columns_wallace(
        &self,
        circuit: &mut PolyCircuit<P>,
        columns: &mut Columns,
    ) -> (Vec<GateId>, Vec<GateId>) {
        let w = columns.len();
        if w == 0 {
            return (vec![], vec![]);
        }
        // base * comp_rate < base^2
        let comp_rate = (1usize << self.ctx.limb_bit_size) - 1;
        let lut_ids = self.ctx.lut_ids;
        // Iteratively reduce column heights by applying comp_rate:2 compressions.
        loop {
            let mut next: Columns = vec![vec![]; w + 1]; // +1 for carries spilling into w
            let mut done = true;
            for k in 0..w {
                let col = &mut columns[k];
                if col.len() > 2 {
                    done = false;
                }
                // Process comp_rate inputs
                let mut idx: usize = 0;
                while idx < col.len() {
                    let last_col_idx = (idx + comp_rate - 1).min(col.len() - 1);
                    let mut sum = col[idx].clone();
                    for i in idx + 1..=last_col_idx {
                        sum = circuit.add_gate(sum, col[i]);
                    }
                    let digit = circuit.public_lookup_gate(sum, lut_ids.0); // % B
                    let carry = circuit.public_lookup_gate(sum, lut_ids.1); // / B
                    next[k].push(digit);
                    if k + 1 < next.len() {
                        next[k + 1].push(carry);
                    }
                    idx = last_col_idx + 1;
                }
            }
            // Drop any potential tail beyond max width
            columns.truncate(w);
            // Move back carries from column w into range if any (we discard overflow beyond max)
            let tail = next.pop();
            let mut compact = vec![vec![]; w];
            for k in 0..w {
                compact[k] = std::mem::take(&mut next[k]);
            }
            if let Some(t) = tail {
                if w > 0 {
                    /* overflow beyond max width ignored */
                    let _ = t;
                }
            }
            *columns = compact;
            if done {
                break;
            }
        }

        // At this point, each column has height <= 2. For columns with two
        // residual values at the same weight, perform one lookup to convert to
        // a proper (digit, carry) pair.
        let zero = circuit.const_zero_gate();
        let mut sum_vec = Vec::with_capacity(w);
        let mut carry_vec = vec![zero; w + 1]; // shifted by +1 to the right
        for k in 0..w {
            match columns[k].as_slice() {
                [] => sum_vec.push(zero),
                [x] => sum_vec.push(*x),
                [x, y] => {
                    let s = circuit.add_gate(*x, *y);
                    let digit = circuit.public_lookup_gate(s, lut_ids.0);
                    let carry = circuit.public_lookup_gate(s, lut_ids.1);
                    sum_vec.push(digit);
                    carry_vec[k + 1] = carry;
                }
                _ => unreachable!("column height should be <= 2 after compression"),
            }
        }
        (sum_vec, carry_vec)
    }

    // Final normalization by a parallel-prefix CPA: add S and C (shifted) once to produce digits.
    pub(crate) fn final_cpa(
        circuit: &mut PolyCircuit<P>,
        mut sum_vec: Vec<GateId>,
        carry_vec: Vec<GateId>,
        lut_ids: (usize, usize),
        max_limbs: usize,
    ) -> Vec<GateId> {
        // Ensure width
        let w = sum_vec.len().min(max_limbs);
        sum_vec.truncate(w);
        // Precompute t_k = S_k + C_k
        let zero = circuit.const_zero_gate();
        let mut ss = Vec::with_capacity(w);
        let mut gs = Vec::with_capacity(w);
        let mut ps = Vec::with_capacity(w);
        for k in 0..w {
            let c = carry_vec.get(k).copied().unwrap_or(zero);
            let t = circuit.add_gate(sum_vec[k], c);
            let s = circuit.public_lookup_gate(t, lut_ids.0);
            let g = circuit.public_lookup_gate(t, lut_ids.1);
            // p = 1 iff s == B-1 <=> floor((s + 1)/B) = 1
            let s_plus = circuit.add_gate(s, GateId(0));
            let p = circuit.public_lookup_gate(s_plus, lut_ids.1);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }
        let (g_pref, _) = Self::prefix_gp(circuit, &gs, &ps);
        let mut out = Vec::with_capacity(w);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let t = circuit.add_gate(ss[i], carry_in);
            let digit = circuit.public_lookup_gate(t, lut_ids.0);
            out.push(digit);
        }
        out
    }

    // Kogge–Stone parallel prefix on (g, p)
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

    pub(crate) fn mul_without_cpa(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_limbs: usize,
    ) -> (Vec<GateId>, Vec<GateId>) {
        // 1) Schoolbook partial products with immediate split and column placement
        let mut columns = Self::schoolbook_partial_products_columns(
            circuit,
            &self.limbs,
            &other.limbs,
            max_limbs,
            self.ctx.lut_ids,
        );

        // 2) Compress columns (one-shot if H_max < B, else Wallace compressors)
        // let base: usize = 1usize << self.ctx.limb_bit_size;
        // let h_max = columns.iter().map(|c| c.len()).max().unwrap_or(0);
        self.compress_columns_wallace(circuit, &mut columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        element::PolyElem,
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use std::sync::Arc;

    const INPUT_BIT_SIZE: usize = 20;
    const LIMB_BIT_SIZE: usize = 5;
    const LIMB_LEN: usize = INPUT_BIT_SIZE / LIMB_BIT_SIZE;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        total_limbs: usize,
    ) -> (Vec<GateId>, DCRTPolyParams, Arc<BigUintPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let inputs = circuit.input(total_limbs);
        let ctx = Arc::new(BigUintPolyContext::setup(circuit, &params, LIMB_BIT_SIZE));
        (inputs, params, ctx)
    }

    fn create_test_biguint_from_value(
        ctx: Arc<BigUintPolyContext<DCRTPoly>>,
        params: &DCRTPolyParams,
        value: u32,
    ) -> Vec<DCRTPoly> {
        let limb_len = LIMB_LEN;
        let mut limbs = Vec::with_capacity(limb_len);
        let mut remaining_value = value;
        let base = 1u32 << ctx.limb_bit_size;

        for _ in 0..limb_len {
            let limb_value = remaining_value % base;
            limbs.push(DCRTPoly::from_usize_to_constant(&params, limb_value as usize));
            remaining_value /= base;
        }

        debug_assert_eq!(remaining_value, 0);
        limbs
    }

    #[test]
    fn test_biguint_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let result = big_a.add(&big_b, &mut circuit);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 15);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 20);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 15u32 + 20u32;
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), 5);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_add_with_carry() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let result = big_a.add(&big_b, &mut circuit);
        circuit.output(result.limbs.clone());

        // Use values that will cause carry with 20-bit input size (4 limbs of 5 bits each)
        let a = create_test_biguint_from_value(ctx.clone(), &params, 1048575); // 2^20 - 1
        let b = create_test_biguint_from_value(ctx.clone(), &params, 1);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 1048575u32 + 1u32; // This will be 1048576 = 2^20
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_less_than_smaller() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        // a < b (500 < 1000), so less_than should return 1 (true)
        let a = create_test_biguint_from_value(ctx.clone(), &params, 500);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 1000);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 1u32.into());

        let mut expected_diff = 500u32 + (1u32 << INPUT_BIT_SIZE) - 1000u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_less_than_equal() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        // a == b (12345 == 12345), so less_than should return 0 (false)
        let a = create_test_biguint_from_value(ctx.clone(), &params, 12345);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 12345);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 0u32.into());

        let mut expected_diff = 1u32 << INPUT_BIT_SIZE; // base
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_less_than_greater() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        // a > b (1000 > 500), so less_than should return 0 (false)
        let a = create_test_biguint_from_value(ctx.clone(), &params, 1000);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 500);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + LIMB_LEN);

        let lt_coeffs = eval_result[0].coeffs();
        assert_eq!(*lt_coeffs[0].value(), 0u32.into());

        let mut expected_diff = 1000u32 - 500u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_diff == 0 {
                break;
            }
            expected_limbs[i] = expected_diff % (1u32 << ctx.limb_bit_size);
            expected_diff /= 1u32 << ctx.limb_bit_size;
        }

        for i in 1..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i - 1].into());
        }
    }

    #[test]
    fn test_biguint_mul_simple() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 123);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 456);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 123u32 * 456u32;
        let output_limb_len = 40 / LIMB_BIT_SIZE; // 8 limbs for 40-bit output
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_mul_with_overflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..].to_vec());
        // Use values that will cause overflow with 20-bit input size
        let result = big_a.mul(&big_b, &mut circuit, Some(40));
        circuit.output(result.limbs.clone());

        // Use larger values that will produce overflow across multiple limbs
        let a = create_test_biguint_from_value(ctx.clone(), &params, 1023); // near max for 10 bits
        let b = create_test_biguint_from_value(ctx.clone(), &params, 1023);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 1023u32 * 1023u32;
        let output_limb_len = 40 / LIMB_BIT_SIZE; // 8 limbs for 40-bit output
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_zero() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, params, ctx) = create_test_context(&mut circuit, LIMB_LEN);

        let zero = BigUintPoly::zero(ctx.clone(), INPUT_BIT_SIZE);
        circuit.output(zero.limbs.clone());

        let dummy_input = create_test_biguint_from_value(ctx.clone(), &params, 0);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &dummy_input, Some(plt_evaluator));

        assert_eq!(eval_result.len(), LIMB_LEN);

        for limb_result in eval_result {
            let limb_coeffs = limb_result.coeffs();
            assert_eq!(*limb_coeffs[0].value(), 0u32.into());
        }
    }

    #[test]
    fn test_biguint_extend_size() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());

        // Extend from 20 bits to 25 bits (5 limbs)
        let extended = big_a.extend_size(25);
        circuit.output(extended.limbs.clone());

        let a_value = create_test_biguint_from_value(ctx.clone(), &params, 12345);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &a_value, Some(plt_evaluator));

        let extended_limb_len = 25 / LIMB_BIT_SIZE; // 6 limbs for 25 bits (with +1 from extend_size)
        assert_eq!(eval_result.len(), extended_limb_len);

        // Check that the original value is preserved in the first limbs
        let mut expected_value = 12345u32;
        let mut expected_limbs = vec![0; extended_limb_len];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }
        // Remaining limbs should be 0

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_add_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN + 2);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b =
            BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..LIMB_LEN + 2].to_vec()); // only 2 limbs
        let result = big_a.add(&big_b, &mut circuit);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 100);
        let b_limbs = vec![
            DCRTPoly::from_usize_to_constant(&params, 50 % (1u32 << ctx.limb_bit_size) as usize),
            DCRTPoly::from_usize_to_constant(&params, 50 / (1u32 << ctx.limb_bit_size) as usize),
        ];
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b_limbs].concat(),
            Some(plt_evaluator),
        );

        let mut expected_sum = 100u32 + 50u32;
        let mut expected_limbs = vec![0; LIMB_LEN + 1];
        for i in 0..LIMB_LEN + 1 {
            if expected_sum == 0 {
                break;
            }
            expected_limbs[i] = expected_sum % (1u32 << ctx.limb_bit_size);
            expected_sum /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN + 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_mul_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN + 2);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b =
            BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..LIMB_LEN + 2].to_vec()); // only 2 limbs
        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 100);
        let b_limbs = vec![
            DCRTPoly::from_usize_to_constant(&params, 50 % (1u32 << ctx.limb_bit_size) as usize),
            DCRTPoly::from_usize_to_constant(&params, 50 / (1u32 << ctx.limb_bit_size) as usize),
        ];
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b_limbs].concat(),
            Some(plt_evaluator),
        );

        let mut expected_product = 100u32 * 50u32;
        let output_limb_len = (INPUT_BIT_SIZE + 10) / LIMB_BIT_SIZE; // a bit size + b bit size
        let mut expected_limbs = vec![0; output_limb_len];
        for i in 0..output_limb_len {
            if expected_product == 0 {
                break;
            }
            expected_limbs[i] = expected_product % (1u32 << ctx.limb_bit_size);
            expected_product /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), output_limb_len);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_left_shift() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let shifted = big_a.left_shift(1);
        circuit.output(shifted.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 12345);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &a, Some(plt_evaluator));

        // Left shift by 1 means removing the first limb
        let mut expected_value = 12345u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        // After shifting by 1, we expect to see limbs[1..] from the original
        assert_eq!(eval_result.len(), LIMB_LEN - 1);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i + 1].into());
        }
    }

    #[test]
    fn test_biguint_cmux() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN + 1);

        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b =
            BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..2 * LIMB_LEN].to_vec());
        let selector = inputs[2 * LIMB_LEN];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 123);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 456);
        let selector_value = vec![DCRTPoly::from_usize_to_constant(&params, 1)]; // selector = 1, should return 'a'
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // With selector = 1, expect to get 'a' (123)
        let mut expected_value = 123u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_cmux_select_other() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, 2 * LIMB_LEN + 1);

        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let big_b =
            BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[LIMB_LEN..2 * LIMB_LEN].to_vec());
        let selector = inputs[2 * LIMB_LEN];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        circuit.output(result.limbs.clone());

        let a = create_test_biguint_from_value(ctx.clone(), &params, 123);
        let b = create_test_biguint_from_value(ctx.clone(), &params, 456);
        let selector_value = vec![DCRTPoly::from_usize_to_constant(&params, 0)]; // selector = 0, should return 'b'
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a, b, selector_value].concat(),
            Some(plt_evaluator),
        );

        // With selector = 0, expect to get 'b' (456)
        let mut expected_value = 456u32;
        let mut expected_limbs = vec![0; LIMB_LEN];
        for i in 0..LIMB_LEN {
            if expected_value == 0 {
                break;
            }
            expected_limbs[i] = expected_value % (1u32 << ctx.limb_bit_size);
            expected_value /= 1u32 << ctx.limb_bit_size;
        }

        assert_eq!(eval_result.len(), LIMB_LEN);

        for i in 0..eval_result.len() {
            let coeffs = eval_result[i].coeffs();
            assert_eq!(*coeffs[0].value(), expected_limbs[i].into());
        }
    }

    #[test]
    fn test_biguint_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        let test_value = 12345u32;
        let a = create_test_biguint_from_value(ctx.clone(), &params, test_value);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &a, Some(plt_evaluator));

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }

    #[test]
    fn test_biguint_finalize_large_value() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (inputs, params, ctx) = create_test_context(&mut circuit, LIMB_LEN);
        let big_a = BigUintPoly::<DCRTPoly>::new(ctx.clone(), inputs[0..LIMB_LEN].to_vec());
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);

        // Use a value that spans multiple limbs (2^20 - 1 = 1048575)
        let test_value = 1048575u32;
        let a = create_test_biguint_from_value(ctx.clone(), &params, test_value);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result =
            circuit.eval(&params, &DCRTPoly::const_one(&params), &a, Some(plt_evaluator));

        assert_eq!(eval_result.len(), 1);
        let coeffs = eval_result[0].coeffs();
        assert_eq!(*coeffs[0].value(), test_value.into());
    }
}
