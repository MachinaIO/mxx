use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    element::PolyElem,
    lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use num_traits::Zero;
use std::{marker::PhantomData, sync::Arc};

// Carry-based big-integer arithmetic over circuit gates.
//
// The representation is a little-endian base-`2^w` limb vector:
// `value = limbs[0] + limbs[1] * 2^w + ...`.
// Each limb is stored as one gate and is expected to stay in `[0, 2^w)`.
//
// The implementation keeps the arithmetic shallow by decomposing each operation
// into small limb-local steps plus one prefix pass:
// - addition computes per-limb `(sum_digit, generate, propagate)` pairs and then resolves carries
//   with a Kogge-Stone-style prefix network,
// - comparison computes borrow generate/propagate signals with the same prefix idea,
// - multiplication accumulates partial products into columns, compresses those columns down to a
//   carry-save form, and performs one final carry-propagate add.
//
// For `w = 1`, the logic degenerates to ordinary bitwise gates.
// For `w > 1`, the same operations are expressed through small public lookup tables
// such as `x + y -> (mod base, floor / base)` and `x * y -> (mod base, floor / base)`.
// Because this port stores exactly one plaintext value in the constant term,
// these lookups can be applied directly without any slot-packing machinery.

type Columns = Vec<Vec<GateId>>;

#[inline]
fn single_wire<I: Into<BatchedWire>>(wire: I) -> GateId {
    wire.into().as_single_wire()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CarryArithLutIds {
    add_mod: usize,
    add_floor: usize,
    mul_mod: usize,
    mul_floor: usize,
    kss_g: usize,
    kss_p: usize,
}

fn constant_elem<P: Poly>(params: &P::Params, value: u64) -> P::Elem {
    P::Elem::constant(&params.modulus(), value)
}

fn register_lookup<P, F>(
    circuit: &mut PolyCircuit<P>,
    params: &P::Params,
    len: u64,
    max_output_row: (u64, u64),
    output: F,
) -> usize
where
    P: Poly + 'static,
    F: Fn(u64) -> u64 + Send + Sync + 'static,
{
    let max_output_row = (max_output_row.0, constant_elem::<P>(params, max_output_row.1));
    let lut = PublicLut::<P>::new(
        params,
        len,
        move |params, x| {
            if x >= len {
                return None;
            }
            Some((x, constant_elem::<P>(params, output(x))))
        },
        Some(max_output_row),
    );
    circuit.register_public_lookup(lut)
}

fn register_dummy_lookup<P: Poly + 'static>(
    circuit: &mut PolyCircuit<P>,
    params: &P::Params,
) -> usize {
    register_lookup::<P, _>(circuit, params, 1, (0, 0), |_| 0)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarryArithPolyContext<P: Poly> {
    pub limb_bit_size: usize,
    pub const_zero: GateId,
    pub const_base: GateId,
    pub scalar_base: BigUint,
    luts: Option<CarryArithLutIds>,
    _p: PhantomData<P>,
}

impl<P: Poly + 'static> CarryArithPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        dummy_scalar: bool,
    ) -> Self {
        debug_assert!(limb_bit_size < 32);
        let base = 1usize << limb_bit_size;
        let const_zero = single_wire(circuit.const_zero_gate());
        let const_base = single_wire(circuit.const_digits(&[base as u32]));
        let scalar_base = BigUint::from(base);
        let luts = if limb_bit_size > 1 {
            Some(Self::setup_luts(circuit, params, base, dummy_scalar))
        } else {
            None
        };
        Self { limb_bit_size, const_zero, const_base, scalar_base, luts, _p: PhantomData }
    }

    fn setup_luts(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        base: usize,
        dummy_scalar: bool,
    ) -> CarryArithLutIds {
        // The lookup tables encode the limb-local arithmetic used throughout the gadget:
        // - add_mod/add_floor split `x + y` into digit/carry under the base,
        // - mul_mod/mul_floor split one partial product into low/high limbs,
        // - kss_g/kss_p implement the binary combine step of the prefix network.
        if dummy_scalar {
            return CarryArithLutIds {
                add_mod: register_dummy_lookup(circuit, params),
                add_floor: register_dummy_lookup(circuit, params),
                mul_mod: register_dummy_lookup(circuit, params),
                mul_floor: register_dummy_lookup(circuit, params),
                kss_g: register_dummy_lookup(circuit, params),
                kss_p: register_dummy_lookup(circuit, params),
            };
        }

        let b = u64::try_from(base).expect("base must fit in u64");
        let add_max = (b * b - 1) / 2;
        let add_len = 2 * add_max + 1;
        let add_mod =
            register_lookup::<P, _>(circuit, params, add_len, (b - 1, b - 1), move |t| t % b);
        let add_floor =
            register_lookup::<P, _>(circuit, params, add_len, (b * (b - 1), b - 1), move |t| t / b);
        let mul_len = b * b;
        let mul_mod =
            register_lookup::<P, _>(circuit, params, mul_len, ((b - 1) * b + 1, b - 1), move |t| {
                let x = t % b;
                let y = t / b;
                (x * y) % b
            });
        let mul_floor = register_lookup::<P, _>(
            circuit,
            params,
            mul_len,
            (mul_len - 1, b.saturating_sub(2)),
            move |t| {
                let x = t % b;
                let y = t / b;
                (x * y) / b
            },
        );
        let kss_g = register_lookup::<P, _>(circuit, params, 8, (7, 1), move |t| {
            let gk = (t & 1) != 0;
            let gj = (t & 2) != 0;
            let pk = (t & 4) != 0;
            u64::from(gk || (pk && gj))
        });
        let kss_p = register_lookup::<P, _>(circuit, params, 4, (3, 1), move |t| {
            let pk = (t & 1) != 0;
            let pj = (t & 2) != 0;
            u64::from(pk && pj)
        });

        CarryArithLutIds { add_mod, add_floor, mul_mod, mul_floor, kss_g, kss_p }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarryArithPoly<P: Poly> {
    pub ctx: Arc<CarryArithPolyContext<P>>,
    pub limbs: Vec<GateId>,
    _p: PhantomData<P>,
}

impl<P: Poly + 'static> CarryArithPoly<P> {
    pub fn new(ctx: Arc<CarryArithPolyContext<P>>, limbs: Vec<GateId>) -> Self {
        Self { ctx, limbs, _p: PhantomData }
    }

    #[inline]
    pub fn bit_size(&self) -> usize {
        self.limbs.len() * self.ctx.limb_bit_size
    }

    pub fn zero(ctx: Arc<CarryArithPolyContext<P>>, bit_size: usize) -> Self {
        let limb_len = bit_size.div_ceil(ctx.limb_bit_size);
        let limbs = vec![ctx.const_zero; limb_len];
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn const_limbs(
        ctx: Arc<CarryArithPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        limbs: &[P],
    ) -> Self {
        let limbs = limbs.iter().map(|poly| single_wire(circuit.const_poly(poly))).collect();
        Self { ctx, limbs, _p: PhantomData }
    }

    pub fn input(
        ctx: Arc<CarryArithPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        input_bit_size: usize,
    ) -> Self {
        let num_limbs = input_bit_size.div_ceil(ctx.limb_bit_size);
        let limb_gateids = circuit.input(num_limbs).to_vec();
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

        let zero = single_wire(circuit.const_zero_gate());
        let one = single_wire(circuit.const_one_gate());
        let mut ss = Vec::with_capacity(w);
        let mut gs = Vec::with_capacity(w);
        let mut ps = Vec::with_capacity(w);

        // First pass: compute local digit/carry information for each limb independently.
        // `g` means "this limb definitely emits a carry".
        // `p` means "an incoming carry will pass through this limb".
        for i in 0..w {
            let ai = a[i];
            let bi = if i < b.len() { b[i] } else { self.ctx.const_zero };
            let (s, g) = self.add_mod_floor(circuit, ai, bi);
            let p = self.add_floor(circuit, s, one);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }

        let (g_pref, _) = self.prefix_gp(circuit, &gs, &ps);

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
        match self.ctx.luts {
            None => {
                let s = single_wire(circuit.xor_gate(x, y));
                let g = single_wire(circuit.and_gate(x, y));
                (s, g)
            }
            Some(luts) => {
                let t = single_wire(circuit.add_gate(x, y));
                let s = single_wire(circuit.public_lookup_gate(t, luts.add_mod));
                let g = single_wire(circuit.public_lookup_gate(t, luts.add_floor));
                (s, g)
            }
        }
    }

    #[inline]
    fn add_mod(&self, circuit: &mut PolyCircuit<P>, x: GateId, y: GateId) -> GateId {
        match self.ctx.luts {
            None => single_wire(circuit.xor_gate(x, y)),
            Some(luts) => {
                let t = single_wire(circuit.add_gate(x, y));
                single_wire(circuit.public_lookup_gate(t, luts.add_mod))
            }
        }
    }

    #[inline]
    fn add_floor(&self, circuit: &mut PolyCircuit<P>, x: GateId, y: GateId) -> GateId {
        match self.ctx.luts {
            None => single_wire(circuit.and_gate(x, y)),
            Some(luts) => {
                let t = single_wire(circuit.add_gate(x, y));
                single_wire(circuit.public_lookup_gate(t, luts.add_floor))
            }
        }
    }

    pub fn less_than(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> (GateId, Self) {
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        debug_assert_eq!(self.ctx, other.ctx);

        let w = self.limbs.len();
        let zero = single_wire(circuit.const_zero_gate());
        let one = single_wire(circuit.const_one_gate());
        let base_minus_one = {
            let b_minus_1 = (1u32 << self.ctx.limb_bit_size) - 1;
            single_wire(circuit.const_digits(&[b_minus_1]))
        };

        let mut g = Vec::with_capacity(w);
        let mut p = Vec::with_capacity(w);
        let mut ab_pairs = Vec::with_capacity(w);

        // This mirrors addition, but for borrow propagation.
        // Each limb reports whether it generates a borrow (`g`) or only propagates one (`p`).
        for i in 0..w {
            let a = self.limbs[i];
            let b = other.limbs[i];
            let (g_i, p_i) = match self.ctx.luts {
                None => {
                    let not_a = single_wire(circuit.not_gate(a));
                    let gi = single_wire(circuit.and_gate(not_a, b));
                    let xor_ab = single_wire(circuit.xor_gate(a, b));
                    let pi = single_wire(circuit.not_gate(xor_ab));
                    (gi, pi)
                }
                Some(luts) => {
                    let y = single_wire(circuit.sub_gate(base_minus_one, b));
                    let (s, h) = self.add_mod_floor(circuit, a, y);
                    let eq = self.add_floor(circuit, s, one);
                    let not_h = single_wire(circuit.not_gate(h));
                    let not_eq = single_wire(circuit.not_gate(eq));
                    let two_not_eq = single_wire(circuit.add_gate(not_eq, not_eq));
                    let key = single_wire(circuit.add_gate(not_h, two_not_eq));
                    let gi = single_wire(circuit.public_lookup_gate(key, luts.kss_p));
                    (gi, eq)
                }
            };
            g.push(g_i);
            p.push(p_i);
            ab_pairs.push((a, b));
        }

        let (g_pref, _) = self.prefix_gp(circuit, &g, &p);

        let mut diff_limbs = Vec::with_capacity(w);
        for i in 0..w {
            let b_in = if i == 0 { zero } else { g_pref[i - 1] };
            let (a, b) = ab_pairs[i];
            match self.ctx.luts {
                None => {
                    let xor = single_wire(circuit.xor_gate(a, b));
                    let diff = single_wire(circuit.xor_gate(xor, b_in));
                    diff_limbs.push(diff);
                }
                Some(_) => {
                    let b_comp = single_wire(circuit.sub_gate(self.ctx.const_base, b));
                    let y = single_wire(circuit.sub_gate(b_comp, b_in));
                    diff_limbs.push(self.add_mod(circuit, a, y));
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
        // Multiplication keeps carries deferred as long as possible:
        // partial products are compressed to a carry-save pair, then normalized once.
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
        let limbs = self.limbs[..num_limbs].to_vec();
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn cmux(&self, other: &Self, selector: GateId, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let not = single_wire(circuit.not_gate(selector));
        for i in 0..self.limbs.len() {
            let (case1, _) = self.mul_mod_floor(circuit, self.limbs[i], selector, false);
            let (case2, _) = self.mul_mod_floor(circuit, other.limbs[i], not, false);
            limbs.push(single_wire(circuit.add_gate(case1, case2)));
        }
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        debug_assert!(!self.limbs.is_empty(), "limbs should not be empty");
        let mut result = self.limbs[0];
        for i in 1..self.limbs.len() {
            let shift = BigUint::from(1u32) << (self.ctx.limb_bit_size * i);
            let weighted_limb = single_wire(circuit.large_scalar_mul(self.limbs[i], &[shift]));
            result = single_wire(circuit.add_gate(result, weighted_limb));
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
        match self.ctx.luts {
            None => (single_wire(circuit.and_gate(x, y)), None),
            Some(luts) => {
                let shifted = single_wire(
                    circuit.large_scalar_mul(y, std::slice::from_ref(&self.ctx.scalar_base)),
                );
                let key = single_wire(circuit.add_gate(x, shifted));
                let mod_out = single_wire(circuit.public_lookup_gate(key, luts.mul_mod));
                let floor_out = if output_floor {
                    Some(single_wire(circuit.public_lookup_gate(key, luts.mul_floor)))
                } else {
                    None
                };
                (mod_out, floor_out)
            }
        }
    }

    #[inline]
    pub(crate) fn final_cpa(
        &self,
        circuit: &mut PolyCircuit<P>,
        mut sum_vec: Vec<GateId>,
        carry_vec: Vec<GateId>,
        max_limbs: usize,
    ) -> Vec<GateId> {
        // Normalize a carry-save representation `(sum_vec, carry_vec)` into ordinary digits.
        // This uses the same generate/propagate structure as `add`.
        let w = sum_vec.len().min(max_limbs);
        sum_vec.truncate(w);

        let zero = single_wire(circuit.const_zero_gate());
        let one = single_wire(circuit.const_one_gate());
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
            out.push(self.add_mod(circuit, ss[i], carry_in));
        }
        out
    }

    #[inline]
    fn prefix_gp(
        &self,
        circuit: &mut PolyCircuit<P>,
        g: &[GateId],
        p: &[GateId],
    ) -> (Vec<GateId>, Vec<GateId>) {
        // Kogge-Stone-style prefix scan:
        // after this finishes, `gs[i]` tells whether the carry/borrow into limb `i + 1`
        // is forced by any less-significant suffix.
        let w = g.len();
        let mut gs = g.to_vec();
        let mut ps = p.to_vec();
        let mut d = 1usize;
        while d < w {
            let mut gs_next = gs.clone();
            let mut ps_next = ps.clone();
            for k in 0..w {
                if k < d {
                    continue;
                }
                let gj = gs[k - d];
                let pj = ps[k - d];
                let gk = gs[k];
                let pk = ps[k];
                match self.ctx.luts {
                    None => {
                        let pk_and_gj = single_wire(circuit.and_gate(pk, gj));
                        gs_next[k] = single_wire(circuit.or_gate(gk, pk_and_gj));
                        ps_next[k] = single_wire(circuit.and_gate(pk, pj));
                    }
                    Some(luts) => {
                        let two_gj = single_wire(circuit.add_gate(gj, gj));
                        let two_pk = single_wire(circuit.add_gate(pk, pk));
                        let four_pk = single_wire(circuit.add_gate(two_pk, two_pk));
                        let sum = single_wire(circuit.add_gate(two_gj, four_pk));
                        let key_g = single_wire(circuit.add_gate(gk, sum));
                        gs_next[k] = single_wire(circuit.public_lookup_gate(key_g, luts.kss_g));

                        let two_pj = single_wire(circuit.add_gate(pj, pj));
                        let key_p = single_wire(circuit.add_gate(pk, two_pj));
                        ps_next[k] = single_wire(circuit.public_lookup_gate(key_p, luts.kss_p));
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
        // Schoolbook partial products are first grouped by output column `i + j`.
        // Each column is then repeatedly compressed until only a carry-save pair remains.
        let mut columns: Columns = vec![vec![]; max_limbs];
        for (i, &ai) in self.limbs.iter().enumerate() {
            for (j, &bj) in other.limbs.iter().enumerate() {
                let k = i + j;
                if k >= max_limbs {
                    continue;
                }
                match self.ctx.luts {
                    None => {
                        let (prod, _) = self.mul_mod_floor(circuit, ai, bj, false);
                        columns[k].push(prod);
                    }
                    Some(_) => {
                        let (lo, hi) = self.mul_mod_floor(circuit, ai, bj, k + 1 < max_limbs);
                        columns[k].push(lo);
                        if let Some(hi) = hi {
                            columns[k + 1].push(hi);
                        }
                    }
                }
            }
        }

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
                        // Compress several same-column terms into one digit that stays in this
                        // column and one carry that moves to the next column.
                        let (digit, carry) = match self.ctx.luts {
                            None => {
                                let (a, b, c) = (col[idx], col[idx + 1], col[idx + 2]);
                                let xor_ab = single_wire(circuit.xor_gate(a, b));
                                let digit = single_wire(circuit.xor_gate(xor_ab, c));
                                let and_ab = single_wire(circuit.and_gate(a, b));
                                let and_abc = single_wire(circuit.and_gate(xor_ab, c));
                                let carry = single_wire(circuit.or_gate(and_ab, and_abc));
                                (digit, carry)
                            }
                            Some(luts) => {
                                let mut sum = col[idx];
                                for item in &col[(idx + 1)..(idx + group_len)] {
                                    sum = single_wire(circuit.add_gate(sum, *item));
                                }
                                let digit =
                                    single_wire(circuit.public_lookup_gate(sum, luts.add_mod));
                                let carry =
                                    single_wire(circuit.public_lookup_gate(sum, luts.add_floor));
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

        let zero = single_wire(circuit.const_zero_gate());
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

pub fn encode_carry_arith_poly<P: Poly>(
    limb_bit_size: usize,
    num_limbs_per_value: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    // Encode exactly one integer into constant polynomials, one polynomial per limb.
    // Unlike the older packed-slot design, there is no per-slot multiplexing here.
    let mask = (BigUint::from(1u32) << limb_bit_size) - BigUint::from(1u32);
    let mut value = input.clone();
    let mut limb_constants = Vec::with_capacity(num_limbs_per_value);

    for _ in 0..num_limbs_per_value {
        limb_constants.push(&value & &mask);
        value >>= limb_bit_size;
    }
    if !value.is_zero() {
        panic!("the input {} is too large for the number of limbs {}", input, num_limbs_per_value);
    }

    limb_constants
        .into_iter()
        .map(|constant| P::from_biguint_to_constant(params, constant))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };

    const LIMB_BIT_SIZE: usize = 5;

    fn eval_with_const_one(
        circuit: &PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        inputs: &[DCRTPoly],
    ) -> Vec<DCRTPoly> {
        let plt_evaluator = PolyPltEvaluator::new();
        circuit.eval(
            params,
            DCRTPoly::const_one(params),
            inputs.to_vec(),
            Some(&plt_evaluator),
            None,
            None,
        )
    }

    fn create_test_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> (DCRTPolyParams, Arc<CarryArithPolyContext<DCRTPoly>>) {
        let params = DCRTPolyParams::default();
        let ctx =
            Arc::new(CarryArithPolyContext::setup(&mut *circuit, &params, LIMB_BIT_SIZE, false));
        (params, ctx)
    }

    fn input_limb_len(params: &DCRTPolyParams) -> usize {
        params.modulus_bits().div_ceil(LIMB_BIT_SIZE)
    }

    fn input_bit_size(params: &DCRTPolyParams) -> usize {
        input_limb_len(params) * LIMB_BIT_SIZE
    }

    fn max_value_for_modulus(modulus: &BigUint) -> BigUint {
        modulus - BigUint::from(1u32)
    }

    fn test_carry_arith_add_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let limb_len = input_limb_len(&params);
        let lhs =
            CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, input_bit_size(&params));
        let rhs =
            CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, input_bit_size(&params));

        let lhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &lhs_value);
        let rhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &rhs_value);

        let result = lhs.add(&rhs, &mut circuit);
        circuit.output(result.limbs.clone());

        let mut eval_inputs = lhs_inputs;
        eval_inputs.extend(rhs_inputs);
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected =
            encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len + 1, &params, &(lhs_value + rhs_value));
        assert_eq!(eval_result, expected);
    }

    fn test_carry_arith_less_than_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let limb_len = input_limb_len(&params);
        let bit_size = input_bit_size(&params);
        let lhs = CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, bit_size);
        let rhs = CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, bit_size);

        let lhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &lhs_value);
        let rhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &rhs_value);

        let (lt_result, diff) = lhs.less_than(&rhs, &mut circuit);
        let mut outputs = vec![lt_result];
        outputs.extend(diff.limbs.clone());
        circuit.output(outputs);

        let mut eval_inputs = lhs_inputs;
        eval_inputs.extend(rhs_inputs);
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);

        let (expected_lt_value, expected_diff_value) = if lhs_value < rhs_value {
            (BigUint::from(1u32), (BigUint::from(1u32) << bit_size) + lhs_value - rhs_value)
        } else {
            (BigUint::from(0u32), lhs_value - rhs_value)
        };
        let expected_lt = encode_carry_arith_poly(LIMB_BIT_SIZE, 1, &params, &expected_lt_value);
        let expected_diff =
            encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &expected_diff_value);
        let mut expected = expected_lt;
        expected.extend(expected_diff);
        assert_eq!(eval_result, expected);
    }

    fn test_carry_arith_mul_case(lhs_value: BigUint, rhs_value: BigUint) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context(&mut circuit);
        let limb_len = input_limb_len(&params);
        let lhs =
            CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, input_bit_size(&params));
        let rhs =
            CarryArithPoly::<DCRTPoly>::input(ctx.clone(), &mut circuit, input_bit_size(&params));

        let lhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &lhs_value);
        let rhs_inputs = encode_carry_arith_poly(LIMB_BIT_SIZE, limb_len, &params, &rhs_value);

        let result = lhs.mul(&rhs, &mut circuit, None);
        circuit.output(result.limbs.clone());

        let mut eval_inputs = lhs_inputs;
        eval_inputs.extend(rhs_inputs);
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected =
            encode_carry_arith_poly(LIMB_BIT_SIZE, 2 * limb_len, &params, &(lhs_value * rhs_value));
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_carry_arith_add_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_carry_arith_add_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_carry_arith_add_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_carry_arith_add_case(min_value.clone(), max_value.clone());
        test_carry_arith_add_case(max_value.clone(), max_value);
    }

    #[test]
    fn test_carry_arith_less_than_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_carry_arith_less_than_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_carry_arith_less_than_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_carry_arith_less_than_case(min_value.clone(), max_value.clone());
        test_carry_arith_less_than_case(max_value, min_value);
    }

    #[test]
    fn test_carry_arith_mul_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_carry_arith_mul_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_carry_arith_mul_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_carry_arith_mul_case(min_value, max_value.clone());
        test_carry_arith_mul_case(max_value.clone(), max_value);
    }
}
