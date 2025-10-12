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
    pub max_degree: usize,
    pub const_zero: GateId,
    pub const_base: GateId,
    pub scalar_base: BigUint,
    // (add_mod, add_floor, mul_mod, mul_floor, kss_g, kss_p)
    pub luts: Option<(
        PackedPlt<P>,
        PackedPlt<P>,
        PackedPlt<P>,
        PackedPlt<P>,
        PackedPlt<P>,
        PackedPlt<P>,
    )>,
}

impl<P: Poly> BigUintPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        max_degree: usize,
        dummy_scalar: bool,
    ) -> Self {
        // Assume base < 2^32
        debug_assert!(limb_bit_size < 32);
        let base = 1 << limb_bit_size;
        let const_zero = circuit.const_zero_gate();
        let const_base = circuit.const_digits(&[base as u32]);
        let scalar_base = BigUint::from(base);
        let luts = if limb_bit_size > 1 {
            let luts = Self::setup_packed_luts(circuit, params, base, max_degree, dummy_scalar);
            Some(luts)
        } else {
            None
        };
        Self { limb_bit_size, max_degree, const_zero, const_base, scalar_base, luts }
    }

    fn setup_packed_luts(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        base: usize,
        max_degree: usize,
        dummy_scalar: bool,
    ) -> (PackedPlt<P>, PackedPlt<P>, PackedPlt<P>, PackedPlt<P>, PackedPlt<P>, PackedPlt<P>) {
        // B_p base
        let b = base;

        // Addition LUTs over D_+ x D_+
        // D_+ = [0, floor((B^2 - 1)/2)]
        // Key t = k1 + k2 ∈ [0, 2*add_max] (<= B^2 - 1)
        let add_max = (b * b - 1) / 2;
        let add_rows = 2 * add_max + 1; // inclusive range [0, 2*add_max]
        let add_map_mod: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..add_rows).into_par_iter().map(|t| {
                let input = BigUint::from(t);
                let output = BigUint::from(t % b);
                (input, (t, output))
            }));
        let add_map_floor: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..add_rows).into_par_iter().map(|t| {
                let input = BigUint::from(t);
                let output = BigUint::from(t / b);
                (input, (t, output))
            }));

        // Multiplication LUTs over D_x x D_x with key t = k1 + k2*B
        // k1, k2 ∈ [0, B-1] => t ∈ [0, B^2 - 1]
        let mul_rows = b * b;
        let mul_map_mod: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..mul_rows).into_par_iter().map(|t| {
                let k1 = t % b;
                let k2 = t / b;
                let out = (k1 * k2) % b;
                (BigUint::from(t), (t, BigUint::from(out)))
            }));
        let mul_map_floor: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..mul_rows).into_par_iter().map(|t| {
                let k1 = t % b;
                let k2 = t / b;
                let out = (k1 * k2) / b;
                (BigUint::from(t), (t, BigUint::from(out)))
            }));

        // Kogge–Stone LUTs
        // L_g: key t = g_k + 2*g_{k-d} + 4*p_k in [0,7], output: g_k OR (p_k AND g_{k-d})
        let or_and_map: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..8).into_par_iter().map(|t| {
                let gk = (t & 1) != 0;
                let gj = (t & 2) != 0; // bit1
                let pk = (t & 4) != 0; // bit2
                let out = (gk as u8) | ((pk as u8) & (gj as u8));
                (BigUint::from(t), (t, BigUint::from(out)))
            }));
        // L_p: key t = p_k + 2*p_{k-d} in [0,3], output: p_k AND p_{k-d}
        let and_map: HashMap<BigUint, (usize, BigUint)> =
            HashMap::from_par_iter((0..4).into_par_iter().map(|t| {
                let pk = (t & 1) != 0;
                let pj = (t & 2) != 0;
                let out = (pk as u8) & (pj as u8);
                (BigUint::from(t), (t, BigUint::from(out)))
            }));

        (
            PackedPlt::setup(circuit, params, max_degree, add_map_mod, dummy_scalar),
            PackedPlt::setup(circuit, params, max_degree, add_map_floor, dummy_scalar),
            PackedPlt::setup(circuit, params, max_degree, mul_map_mod, dummy_scalar),
            PackedPlt::setup(circuit, params, max_degree, mul_map_floor, dummy_scalar),
            PackedPlt::setup(circuit, params, max_degree, or_and_map, dummy_scalar),
            PackedPlt::setup(circuit, params, max_degree, and_map, dummy_scalar),
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

    pub fn const_limbs(
        ctx: Arc<BigUintPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        limbs: &[P],
    ) -> Self {
        let limbs = limbs.into_iter().map(|poly| circuit.const_poly(poly)).collect();
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
            // s_i, g_i from IntMod&Floor_{+,Bp}
            let (s, g) = self.add_mod_floor(circuit, ai, bi);

            // p_i = floor((s_i + 1)/B) via another IntMod&Floor_{+,Bp}
            let p = self.add_floor(circuit, s, one);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }

        // Parallel-prefix (Kogge–Stone) on (g,p)
        let (g_pref, _) = self.prefix_gp(circuit, &gs, &ps);

        // Compute final limbs: s_i = (s'_i + c_i) mod B
        let mut limbs = Vec::with_capacity(w + 1);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let digit = self.add_mod(circuit, ss[i], carry_in);
            limbs.push(digit);
        }
        let last_carry = if w == 0 { zero } else { g_pref[w - 1] };
        limbs.push(last_carry);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    #[inline]
    fn add_mod_floor(
        &self,
        circuit: &mut PolyCircuit<P>,
        x: GateId,
        y: GateId,
    ) -> (GateId, GateId) {
        match self.ctx.luts.as_ref() {
            None => {
                let s = circuit.xor_gate(x, y);
                let g = circuit.and_gate(x, y);
                (s, g)
            }
            Some((add_mod_lut, add_floor_lut, _, _, _, _)) => {
                let t = circuit.add_gate(x, y);
                let s = add_mod_lut.lookup_all(circuit, t);
                let g = add_floor_lut.lookup_all(circuit, t);
                (s, g)
            }
        }
    }

    #[inline]
    fn add_mod(&self, circuit: &mut PolyCircuit<P>, x: GateId, y: GateId) -> GateId {
        match self.ctx.luts.as_ref() {
            None => circuit.xor_gate(x, y),
            Some((add_mod_lut, _, _, _, _, _)) => {
                let t = circuit.add_gate(x, y);
                add_mod_lut.lookup_all(circuit, t)
            }
        }
    }

    #[inline]
    fn add_floor(&self, circuit: &mut PolyCircuit<P>, x: GateId, y: GateId) -> GateId {
        match self.ctx.luts.as_ref() {
            None => circuit.and_gate(x, y),
            Some((_, add_floor_lut, _, _, _, _)) => {
                let t = circuit.add_gate(x, y);
                add_floor_lut.lookup_all(circuit, t)
            }
        }
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
            circuit.const_digits(&[b_minus_1])
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
                Some((_, _, _, _, _, and_map)) => {
                    let y = circuit.sub_gate(base_minus_one, b);
                    let (s, h) = self.add_mod_floor(circuit, a, y);
                    // eq_i: s == B-1 <=> floor((s + 1)/B) == 1
                    let eq = self.add_floor(circuit, s, one);
                    let not_h = circuit.not_gate(h);
                    let not_eq = circuit.not_gate(eq);
                    // gi = (1-h_i) AND (1-p_i) via and_map with key = x + 2*y
                    let two_not_eq = circuit.add_gate(not_eq, not_eq);
                    let key = circuit.add_gate(not_h, two_not_eq);
                    let gi_and = and_map.lookup_all(circuit, key);
                    (gi_and, eq)
                }
            };
            g.push(g_i);
            p.push(p_i);
            ab_pairs.push((a, b));
        }

        // Parallel-prefix (Kogge–Stone) on (g,p) - same for both cases.
        let (g_pref, _) = self.prefix_gp(circuit, &g, &p);

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
                Some((_, _, _, _, _, _)) => {
                    // Difference digit via IntMod&Floor_{+,B}(a, B - b - b_in)
                    let b_comp = circuit.sub_gate(self.ctx.const_base, b);
                    let y = circuit.sub_gate(b_comp, b_in);
                    let digit = self.add_mod(circuit, a, y);
                    diff_limbs.push(digit);
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
            let (case1, _) = self.mul_mod_floor(circuit, self.limbs[i], selector, false);
            let (case2, _) = self.mul_mod_floor(circuit, other.limbs[i], not, false);
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
    fn mul_mod_floor(
        &self,
        circuit: &mut PolyCircuit<P>,
        x: GateId,
        y: GateId,
        output_floor: bool,
    ) -> (GateId, Option<GateId>) {
        match self.ctx.luts.as_ref() {
            None => (circuit.and_gate(x, y), None),
            Some((_, _, mul_mod_lut, mul_floor_lut, _, _)) => {
                let shifted = circuit.large_scalar_mul(y, &[self.ctx.scalar_base.clone()]);
                let key = circuit.add_gate(x, shifted);
                let mod_out = mul_mod_lut.lookup_all(circuit, key);
                let floor_out =
                    if output_floor { Some(mul_floor_lut.lookup_all(circuit, key)) } else { None };
                (mod_out, floor_out)
            }
        }
    }

    #[inline]
    // Final normalization by a parallel-prefix CPA: add S and C (shifted) once to produce digits.
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
            let (s, g) = self.add_mod_floor(circuit, sum_vec[k], c);
            let p = self.add_floor(circuit, s, one);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }
        let (g_pref, _) = self.prefix_gp(circuit, &gs, &ps);
        let mut out = Vec::with_capacity(w);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let digit = self.add_mod(circuit, ss[i], carry_in);
            out.push(digit);
        }
        out
    }

    // Kogge–Stone parallel prefix on (g, p)
    #[inline]
    fn prefix_gp(
        &self,
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
                    match self.ctx.luts.as_ref() {
                        None => {
                            // G' = gk OR (pk AND gj); P' = pk AND pj
                            let pk_and_gj = circuit.and_gate(pk, gj);
                            let g_new = circuit.or_gate(gk, pk_and_gj);
                            let p_new = circuit.and_gate(pk, pj);
                            gs_next[k] = g_new;
                            ps_next[k] = p_new;
                        }
                        Some((_, _, _, _, kss_g_lut, kss_p_lut)) => {
                            // key_g = gk + 2*gj + 4*pk
                            let two_gj = circuit.add_gate(gj, gj);
                            let two_pk = circuit.add_gate(pk, pk);
                            let four_pk = circuit.add_gate(two_pk, two_pk);
                            let sum = circuit.add_gate(two_gj, four_pk);
                            let key_g = circuit.add_gate(gk, sum);
                            let g_new = kss_g_lut.lookup_all(circuit, key_g);
                            // key_p = pk + 2*pj
                            let two_pj = circuit.add_gate(pj, pj);
                            let key_p = circuit.add_gate(pk, two_pj);
                            let p_new = kss_p_lut.lookup_all(circuit, key_p);
                            gs_next[k] = g_new;
                            ps_next[k] = p_new;
                        }
                    }
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
        // Columns sized up to max_limbs; carries to the last column beyond max_limbs are dropped.
        let mut columns: Columns = vec![vec![]; max_limbs];
        for (i, &ai) in self.limbs.iter().enumerate() {
            for (j, &bj) in other.limbs.iter().enumerate() {
                let k = i + j;
                if k >= max_limbs {
                    continue;
                }
                match self.ctx.luts.as_ref() {
                    None => {
                        let (prod, _) = self.mul_mod_floor(circuit, ai, bj, false);
                        columns[k].push(prod);
                    }
                    Some((_, _, _, _, _, _)) => {
                        let (lo, hi) = self.mul_mod_floor(circuit, ai, bj, k + 1 < max_limbs);
                        columns[k].push(lo);
                        if let Some(hi) = hi {
                            columns[k + 1].push(hi);
                        }
                    }
                }
            }
        }

        // Wallace tree compression until column height <= 2
        let base = 1usize << self.ctx.limb_bit_size;
        let comp_rate = if base == 2 { 3 } else { base - 1 };

        loop {
            let w = columns.len();
            let mut next: Columns = vec![vec![]; w + 1];
            for k in 0..w {
                let col = &columns[k];
                let mut idx = 0;
                while idx < col.len() {
                    let group_len = comp_rate.min(col.len() - idx);
                    if group_len <= 2 {
                        next[k].extend_from_slice(&col[idx..idx + group_len]);
                    } else {
                        let (digit, carry) = match self.ctx.luts.as_ref() {
                            None => {
                                let (a, b, c) = (col[idx], col[idx + 1], col[idx + 2]);
                                let xor_ab = circuit.xor_gate(a, b);
                                let digit = circuit.xor_gate(xor_ab, c);
                                let and_ab = circuit.and_gate(a, b);
                                let and_abc = circuit.and_gate(xor_ab, c);
                                let carry = circuit.or_gate(and_ab, and_abc);
                                (digit, carry)
                            }
                            Some((add_mod_lut, add_floor_lut, _, _, _, _)) => {
                                let mut sum = col[idx];
                                for item in &col[(idx + 1)..(idx + group_len)] {
                                    sum = circuit.add_gate(sum, *item);
                                }
                                let digit = add_mod_lut.lookup_all(circuit, sum);
                                let carry = add_floor_lut.lookup_all(circuit, sum);
                                (digit, carry)
                            }
                        };
                        next[k].push(digit);
                        next[k + 1].push(carry);
                    }
                    idx += group_len;
                }
            }
            if next.last().is_some_and(|v| v.is_empty()) {
                next.pop();
            }
            let need_more = next.iter().any(|col| col.len() > 2);
            columns = next;
            if !need_more {
                break;
            }
        }

        let w = columns.len();
        if w == 0 {
            return (vec![], vec![]);
        }

        let zero = circuit.const_zero_gate();
        let mut sum_vec = Vec::with_capacity(w);
        let mut carry_vec = vec![zero; w + 1];

        for k in 0..w {
            match columns[k].as_slice() {
                [] => sum_vec.push(zero),
                [x] => sum_vec.push(*x),
                [x, y] => {
                    let (digit, carry) = self.add_mod_floor(circuit, *x, *y);
                    sum_vec.push(digit);
                    carry_vec[k + 1] = carry;
                }
                _ => unreachable!("column height should be <= 2 after compression"),
            }
        }

        (sum_vec, carry_vec)
    }
}

pub fn encode_biguint_poly<P: Poly>(
    limb_bit_size: usize,
    num_limbs_per_slot: usize,
    params: &P::Params,
    inputs: &[Vec<u64>],
) -> Vec<P> {
    let ring_n = params.ring_dimension() as usize;
    let mask = (1u64 << limb_bit_size) - 1;
    let modulus_big: Arc<BigUint> = params.modulus().into();
    let total_limbs = num_limbs_per_slot;
    // Build slots for each limb: each evaluation slot is reconstructed via CRT combination.
    let mut limb_slots: Vec<Vec<BigUint>> = vec![vec![BigUint::zero(); ring_n]; total_limbs];
    for (crt_idx, inputs_per_crt) in inputs.iter().enumerate() {
        // CRT reconstruction coefficients c_i = (q/qi) * (q/qi)^{-1} (mod qi)
        let (_, reconst_coeff) = params.to_crt_coeffs(crt_idx);
        for (eval_idx, input_raw) in inputs_per_crt.iter().enumerate() {
            let mut input = *input_raw;
            if input == 0 {
                continue;
            }
            let mut limb_idx = 0;
            while input > 0 {
                if limb_idx >= total_limbs {
                    panic!(
                        "the input {input_raw} at crt_idx {crt_idx} and eval_idx {eval_idx} is too large for the number of limbs {num_limbs_per_slot}"
                    );
                }
                limb_slots[limb_idx][eval_idx] = (&limb_slots[limb_idx][eval_idx] +
                    &reconst_coeff * BigUint::from(input & mask)) %
                    modulus_big.as_ref();
                input >>= limb_bit_size;
                limb_idx += 1;
            }
        }
    }
    limb_slots.iter().map(|slots| P::from_biguints_eval(params, slots)).collect::<Vec<_>>()
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

    const LIMB_BIT_SIZE: usize = 5;
    const DEFAULT_LIMB_LEN: usize = 4;
    const INPUT_BIT_SIZE: usize = DEFAULT_LIMB_LEN * LIMB_BIT_SIZE;

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<BigUintPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            circuit,
            &params,
            LIMB_BIT_SIZE,
            params.ring_dimension() as usize,
            false,
        ));
        (params, ctx)
    }

    #[test]
    fn test_biguint_add() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let a_poly = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![15, 1, 3, 9], vec![23, 16, 81, 74]];
        let a_inputs = encode_biguint_poly(LIMB_BIT_SIZE, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let b_poly = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![20, 111, 0, 15], vec![90, 651, 63, 34]];
        let b_inputs = encode_biguint_poly(LIMB_BIT_SIZE, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let result = a_poly.add(&b_poly, &mut circuit);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .into_iter()
            .zip(b_inputs_raw.into_iter())
            .map(|(a_vec, b_vec)| {
                { a_vec.into_iter().zip(b_vec.into_iter()).map(|(a, b)| a + b) }.collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs =
            encode_biguint_poly(LIMB_BIT_SIZE, DEFAULT_LIMB_LEN + 1, &params, &expected_raw);

        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN + 1);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_biguint_add_with_carry() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause carry for some slots
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [
            vec![(1u64 << INPUT_BIT_SIZE) - 1, 123, (1u64 << INPUT_BIT_SIZE) - 2, 0],
            vec![(1u64 << INPUT_BIT_SIZE) - 5, 400, (1u64 << INPUT_BIT_SIZE) - 7, 3],
        ];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);

        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![1, 456, 3, 0], vec![12, 33, 7, 5]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);

        let result = big_a.add(&big_b, &mut circuit).mod_limbs(DEFAULT_LIMB_LEN + 1);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .into_iter()
            .zip(b_inputs_raw.into_iter())
            .map(|(a_vec, b_vec)| {
                a_vec.into_iter().zip(b_vec.into_iter()).map(|(a, b)| a + b).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN + 1, &params, &expected_raw);

        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN + 1);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_biguint_less_than_smaller() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a < b (500 < 1000), so less_than should return 1 (true)
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![500, 7, 10, 1], vec![500, 7, 10, 1]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![1000, 8, 20, 2], vec![1000, 8, 20, 2]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + DEFAULT_LIMB_LEN);

        let expected_lt_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| if a < b { 1u64 } else { 0 })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let base_pow = 1u128 << (ctx.limb_bit_size * DEFAULT_LIMB_LEN);
        let expected_diff_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| {
                        let diff = (base_pow + (*a as u128) - (*b as u128)) % base_pow;
                        diff as u64
                    })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_lt = encode_biguint_poly(ctx.limb_bit_size, 1, &params, &expected_lt_raw);
        let expected_diff =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &expected_diff_raw);
        let mut expected = Vec::with_capacity(1 + DEFAULT_LIMB_LEN);
        expected.extend(expected_lt);
        expected.extend(expected_diff);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_less_than_equal() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a == b in all slots, so less_than should return 0 (false)
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![12345, 0, 1, 999], vec![12345, 0, 1, 999]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = a_inputs_raw.clone();
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + DEFAULT_LIMB_LEN);

        let expected_lt_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| if a < b { 1u64 } else { 0 })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let base_pow = 1u128 << (ctx.limb_bit_size * DEFAULT_LIMB_LEN);
        let expected_diff_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| {
                        let diff = (base_pow + (*a as u128) - (*b as u128)) % base_pow;
                        diff as u64
                    })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_lt = encode_biguint_poly(ctx.limb_bit_size, 1, &params, &expected_lt_raw);
        let expected_diff =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &expected_diff_raw);
        let mut expected = Vec::with_capacity(1 + DEFAULT_LIMB_LEN);
        expected.extend(expected_lt);
        expected.extend(expected_diff);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_less_than_greater() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // a > b in all slots
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![1000, 15, 100, 7], vec![1000, 15, 100, 7]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![500, 10, 99, 3], vec![500, 10, 99, 3]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let (lt_result, diff) = big_a.less_than(&big_b, &mut circuit);
        let mut output_gates = vec![lt_result];
        output_gates.extend(diff.limbs.clone());
        circuit.output(output_gates);

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1 + DEFAULT_LIMB_LEN);

        // Expected lt_result is all zeros and diff corresponds to (a - b) mod base
        let expected_lt_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| if a < b { 1u64 } else { 0 })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let base_pow = 1u128 << (ctx.limb_bit_size * DEFAULT_LIMB_LEN);
        let expected_diff_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| {
                        let diff_val = (base_pow + (*a as u128) - (*b as u128)) % base_pow;
                        diff_val as u64
                    })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_lt = encode_biguint_poly(ctx.limb_bit_size, 1, &params, &expected_lt_raw);
        let expected_diff =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &expected_diff_raw);
        let mut expected = Vec::with_capacity(1 + DEFAULT_LIMB_LEN);
        expected.extend(expected_lt);
        expected.extend(expected_diff);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_mul_simple() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![123, 5, 17, 0], vec![123, 5, 17, 0]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![456, 3, 2, 7], vec![456, 3, 2, 7]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a * b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs =
            encode_biguint_poly(ctx.limb_bit_size, 2 * DEFAULT_LIMB_LEN, &params, &expected_raw);

        assert_eq!(eval_result.len(), 2 * DEFAULT_LIMB_LEN);

        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_biguint_mul_with_overflow() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that will cause overflow with 20-bit input size
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![1023, 2047, 1, 0], vec![1023, 2047, 1, 0]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![1023, 1023, 2, 0], vec![1023, 1023, 2, 0]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a * b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs =
            encode_biguint_poly(ctx.limb_bit_size, 2 * DEFAULT_LIMB_LEN, &params, &expected_raw);

        assert_eq!(eval_result.len(), 2 * DEFAULT_LIMB_LEN);
        assert_eq!(eval_result, expected_limbs);
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

        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN);

        for limb_result in eval_result {
            let limb_coeffs = limb_result.coeffs();
            assert_eq!(*limb_coeffs[0].value(), 0u32.into());
        }
    }

    #[test]
    fn test_biguint_extend_size() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a_full = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![12345, 7, 0, 999], vec![12345, 7, 0, 999]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_a = big_a_full.mod_limbs(DEFAULT_LIMB_LEN);

        // Extend to next multiple of LIMB_BIT_SIZE after INPUT_BIT_SIZE
        let new_bit_size = ((INPUT_BIT_SIZE / LIMB_BIT_SIZE) + 1) * LIMB_BIT_SIZE;
        let extended = big_a.extend_size(new_bit_size);
        circuit.output(extended.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs].concat(),
            Some(plt_evaluator),
        );

        let extended_limb_len = new_bit_size / LIMB_BIT_SIZE; // limb +1 from extend_size
        assert_eq!(eval_result.len(), extended_limb_len);

        let expected_raw: Vec<Vec<u64>> = a_inputs_raw.iter().cloned().collect();
        let expected =
            encode_biguint_poly(ctx.limb_bit_size, extended_limb_len, &params, &expected_raw);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_add_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![100, 10, 5, 3], vec![100, 10, 5, 3]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);

        // For big_b, we're using a smaller bit size
        let b_bit_size = 2 * LIMB_BIT_SIZE;
        let b_limb_len = b_bit_size / ctx.limb_bit_size;
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, b_bit_size);
        // Choose values that fit within b_bit_size
        let b_inputs_raw = [vec![50, 1, 2, 0], vec![50, 1, 2, 0]];
        let b_inputs = encode_biguint_poly(ctx.limb_bit_size, b_limb_len, &params, &b_inputs_raw);

        let result = big_a.add(&big_b, &mut circuit).mod_limbs(DEFAULT_LIMB_LEN + 1);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a + b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN + 1, &params, &expected_raw);

        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN + 1);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_biguint_mul_different_limb_sizes() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        // Create BigUints with different limb sizes
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![100, 3, 7, 8], vec![100, 3, 7, 8]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);

        // For big_b, we're using a smaller bit size
        let b_bit_size = 2 * LIMB_BIT_SIZE;
        let b_limb_len = b_bit_size / ctx.limb_bit_size;
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, b_bit_size);
        let b_inputs_raw = [vec![50, 2, 0, 9], vec![50, 2, 0, 9]];
        let b_inputs = encode_biguint_poly(ctx.limb_bit_size, b_limb_len, &params, &b_inputs_raw);

        let result = big_a.mul(&big_b, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a * b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let max_bit_size = INPUT_BIT_SIZE + b_bit_size;
        let output_limb_len = max_bit_size.div_ceil(LIMB_BIT_SIZE);
        let expected_limbs =
            encode_biguint_poly(ctx.limb_bit_size, output_limb_len, &params, &expected_raw);

        assert_eq!(eval_result.len(), output_limb_len);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_biguint_left_shift() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![12345, 7, 0, 999], vec![12345, 7, 0, 999]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        // After shift, restrict to the first DEFAULT_LIMB_LEN - 1 limbs for assertion
        let shifted = big_a.left_shift(1);
        circuit.output(shifted.limbs.clone());

        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs].concat(),
            Some(plt_evaluator),
        );

        // Left shift by 1 means dividing by the base once
        let base = 1u64 << ctx.limb_bit_size;
        let expected_raw: Vec<Vec<u64>> = a_inputs_raw
            .iter()
            .map(|slot_vec| slot_vec.iter().map(|&value| value / base).collect())
            .collect();
        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN - 1);
        let expected =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN - 1, &params, &expected_raw);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_cmux() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);

        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [vec![123, 7, 88, 5], vec![123, 7, 88, 5]];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let big_b = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let b_inputs_raw = [vec![456, 6, 22, 9], vec![456, 6, 22, 9]];
        let b_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &b_inputs_raw);
        let selector = circuit.input(1)[0];
        let result = big_a.cmux(&big_b, selector, &mut circuit);
        circuit.output(result.limbs.clone());

        // selector per slot: [1, 0, 1, 0]
        let (_, _, crt_depth) = params.to_crt();
        let selector_inputs_raw = vec![vec![1u64, 0, 1, 0]; crt_depth];
        let selector_inputs =
            encode_biguint_poly(ctx.limb_bit_size, 1, &params, &selector_inputs_raw);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs, selector_inputs].concat(),
            Some(plt_evaluator),
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .zip(selector_inputs_raw.iter())
            .map(|((a_vec, b_vec), sel_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .zip(sel_vec.iter())
                    .map(|((a, b), sel)| if *sel == 1 { *a } else { *b })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(eval_result.len(), DEFAULT_LIMB_LEN);
        let expected =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &expected_raw);
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_biguint_finalize() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let test_values: [u64; 4] = [12345, 7, 0, 999];
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [test_values.to_vec(), test_values.to_vec()];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let encoded_limbs: Vec<DCRTPoly> =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let mut expected_poly = DCRTPoly::const_zero(&params);
        for (idx, limb_poly) in encoded_limbs.into_iter().enumerate() {
            let shift = BigUint::from(1u64) << (ctx.limb_bit_size * idx);
            let shift_const = DCRTPoly::from_biguint_to_constant(&params, shift);
            expected_poly += limb_poly * shift_const;
        }
        assert_eq!(eval_result[0], expected_poly);
    }

    #[test]
    fn test_biguint_finalize_large_value() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        // Use values that span multiple limbs
        let test_values: [u64; 4] = [1_048_575, 1, 2, (1 << 19)];
        let big_a = BigUintPoly::<DCRTPoly>::input(
            ctx.clone(),
            &mut circuit,
            DEFAULT_LIMB_LEN * ctx.limb_bit_size,
        );
        let a_inputs_raw = [test_values.to_vec(), test_values.to_vec()];
        let a_inputs =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let finalized = big_a.finalize(&mut circuit);
        circuit.output(vec![finalized]);
        let plt_evaluator = PolyPltEvaluator::new();
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs].concat(),
            Some(plt_evaluator),
        );

        assert_eq!(eval_result.len(), 1);
        let encoded_limbs: Vec<DCRTPoly> =
            encode_biguint_poly(ctx.limb_bit_size, DEFAULT_LIMB_LEN, &params, &a_inputs_raw);
        let mut expected_poly = DCRTPoly::const_zero(&params);
        for (idx, limb_poly) in encoded_limbs.into_iter().enumerate() {
            let shift = BigUint::from(1u64) << (ctx.limb_bit_size * idx);
            let shift_const = DCRTPoly::from_biguint_to_constant(&params, shift);
            expected_poly += limb_poly * shift_const;
        }
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
            params.ring_dimension() as usize,
            false,
        ));

        // SIMD cases: element-wise add
        let a_inputs_raw = [vec![1, 0, 1, 0], vec![1, 0, 1, 0]];
        let b_inputs_raw = [vec![0, 1, 0, 1], vec![0, 1, 0, 1]];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);

        let a_inputs = encode_biguint_poly(1, 1, &params, &a_inputs_raw);
        let b_inputs = encode_biguint_poly(1, 1, &params, &b_inputs_raw);

        let result = big_a.add(&big_b, &mut circuit).mod_limbs(1);
        circuit.output(result.limbs.clone());

        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            None::<PolyPltEvaluator>,
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a + b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs = encode_biguint_poly(1, 1, &params, &expected_raw);
        assert_eq!(eval_result.len(), 1);
        assert_eq!(eval_result, expected_limbs);
    }

    #[test]
    fn test_single_bit_less_than() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            &mut circuit,
            &params,
            1, // limb_bit_size = 1
            params.ring_dimension() as usize,
            false,
        ));

        // SIMD: choose mixed cases
        let a_inputs_raw = [vec![0, 1, 1, 0], vec![0, 1, 1, 0]];
        let b_inputs_raw = [vec![1, 0, 1, 0], vec![1, 0, 1, 0]];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);

        let a_inputs = encode_biguint_poly(1, 1, &params, &a_inputs_raw);
        let b_inputs = encode_biguint_poly(1, 1, &params, &b_inputs_raw);

        let (lt_result, _diff) = big_a.less_than(&big_b, &mut circuit);
        circuit.output(vec![lt_result]);
        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            None::<PolyPltEvaluator>,
        );

        assert_eq!(eval_result.len(), 1);
        let expected_lt_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| if a < b { 1u64 } else { 0 })
                    .collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected = encode_biguint_poly(1, 1, &params, &expected_lt_raw);
        assert_eq!(eval_result[0], expected[0]);
    }

    #[test]
    fn test_single_bit_mul() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let params = DCRTPolyParams::default();
        let ctx = Arc::new(BigUintPolyContext::setup(
            &mut circuit,
            &params,
            1, // limb_bit_size = 1
            params.ring_dimension() as usize,
            false,
        ));

        // SIMD: element-wise mul in base-2
        let a_inputs_raw = [vec![1, 0, 1, 1], vec![1, 0, 1, 1]];
        let b_inputs_raw = [vec![1, 1, 0, 1], vec![1, 1, 0, 1]];

        let big_a = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);
        let big_b = BigUintPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, 1);

        let a_inputs = encode_biguint_poly(1, 1, &params, &a_inputs_raw);
        let b_inputs = encode_biguint_poly(1, 1, &params, &b_inputs_raw);

        let result = big_a.mul(&big_b, &mut circuit, Some(1));
        circuit.output(result.limbs.clone());

        let eval_result = circuit.eval(
            &params,
            &DCRTPoly::const_one(&params),
            &[a_inputs, b_inputs].concat(),
            None::<PolyPltEvaluator>,
        );

        let expected_raw = a_inputs_raw
            .iter()
            .zip(b_inputs_raw.iter())
            .map(|(a_vec, b_vec)| {
                a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a * b).collect::<Vec<u64>>()
            })
            .collect::<Vec<_>>();
        let expected_limbs = encode_biguint_poly(1, 1, &params, &expected_raw);

        assert_eq!(eval_result.len(), 1);
        assert_eq!(eval_result, expected_limbs);
    }
}
