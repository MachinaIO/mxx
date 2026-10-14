use crate::{
    circuit::{BatchedWire, LutExpr, PolyCircuit, PublicLutProgram, gate::GateId},
    poly::Poly,
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
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

/// Digits grouped by limb position: `columns[k]` holds digits of weight `base^k`.
pub type Columns = Vec<Vec<GateId>>;

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

fn register_lookup<P: Poly>(circuit: &mut PolyCircuit<P>, len: u64, value: LutExpr) -> usize {
    circuit.register_public_lookup(
        PublicLutProgram::new(len, value).expect("carry-arithmetic LUT program must be valid"),
    )
}

fn register_dummy_lookup<P: Poly>(circuit: &mut PolyCircuit<P>) -> usize {
    register_lookup(circuit, 1, LutExpr::constant(0))
}

/// Limb-local arithmetic shared by every [`CarryArithPoly`] of one limb size.
///
/// The context holds no gate of a particular circuit: every constant wire is created in the
/// circuit that uses it, so one context also serves the sub-circuits built from that circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarryArithPolyContext<P: Poly> {
    pub limb_bit_size: usize,
    pub scalar_base: BigUint,
    luts: Option<CarryArithLutIds>,
    _p: PhantomData<P>,
}

impl<P: Poly + 'static> CarryArithPolyContext<P> {
    pub fn setup(circuit: &mut PolyCircuit<P>, limb_bit_size: usize, dummy_scalar: bool) -> Self {
        debug_assert!(limb_bit_size < 32);
        let base = 1usize << limb_bit_size;
        let scalar_base = BigUint::from(base);
        let luts = if limb_bit_size > 1 {
            Some(Self::setup_luts(circuit, base, dummy_scalar))
        } else {
            None
        };
        Self { limb_bit_size, scalar_base, luts, _p: PhantomData }
    }

    fn setup_luts(
        circuit: &mut PolyCircuit<P>,
        base: usize,
        dummy_scalar: bool,
    ) -> CarryArithLutIds {
        // The lookup tables encode the limb-local arithmetic used throughout the gadget:
        // - add_mod/add_floor split `x + y` into digit/carry under the base,
        // - mul_mod/mul_floor split one partial product into low/high limbs,
        // - kss_g/kss_p implement the binary combine step of the prefix network.
        if dummy_scalar {
            return CarryArithLutIds {
                add_mod: register_dummy_lookup(circuit),
                add_floor: register_dummy_lookup(circuit),
                mul_mod: register_dummy_lookup(circuit),
                mul_floor: register_dummy_lookup(circuit),
                kss_g: register_dummy_lookup(circuit),
                kss_p: register_dummy_lookup(circuit),
            };
        }

        let b = u64::try_from(base).expect("base must fit in u64");
        let add_max = (b * b - 1) / 2;
        let add_len = 2 * add_max + 1;
        let input = LutExpr::input();
        let add_mod = register_lookup(circuit, add_len, input.clone().modulo(b));
        let add_floor = register_lookup(circuit, add_len, input.clone().floor_div(b));
        let mul_len = b * b;
        let product = input.clone().modulo(b).mul(input.clone().floor_div(b));
        let mul_mod = register_lookup(circuit, mul_len, product.clone().modulo(b));
        let mul_floor = register_lookup(circuit, mul_len, product.floor_div(b));

        let gk = input.clone().modulo(2u8);
        let gj = input.clone().floor_div(2u8).modulo(2u8);
        let pk = input.clone().floor_div(4u8).modulo(2u8);
        let propagated = pk.mul(gj);
        let kss_g =
            register_lookup(circuit, 8, gk.clone().add(propagated.clone()).sub(gk.mul(propagated)));
        let pk = input.clone().modulo(2u8);
        let pj = input.floor_div(2u8).modulo(2u8);
        let kss_p = register_lookup(circuit, 4, pk.mul(pj));

        CarryArithLutIds { add_mod, add_floor, mul_mod, mul_floor, kss_g, kss_p }
    }

    /// `x XOR y = x + y - 2 x y` for bits, reusing an already computed `x y`.
    fn xor_with_and(circuit: &mut PolyCircuit<P>, x: GateId, y: GateId, and: GateId) -> GateId {
        let sum = single_wire(circuit.add_gate(x, y));
        let twice_and = single_wire(circuit.small_scalar_mul(and, &[2]));
        single_wire(circuit.sub_gate(sum, twice_and))
    }

    /// Splits `x + y` into the digit `(x + y) mod base` and the carry `floor((x + y) / base)`.
    #[inline]
    fn add_mod_floor(
        &self,
        circuit: &mut PolyCircuit<P>,
        x: GateId,
        y: GateId,
    ) -> (GateId, GateId) {
        match self.luts {
            None => {
                let g = single_wire(circuit.and_gate(x, y));
                (Self::xor_with_and(circuit, x, y, g), g)
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
        match self.luts {
            None => single_wire(circuit.xor_gate(x, y)),
            Some(luts) => {
                let t = single_wire(circuit.add_gate(x, y));
                single_wire(circuit.public_lookup_gate(t, luts.add_mod))
            }
        }
    }

    #[inline]
    fn add_floor(&self, circuit: &mut PolyCircuit<P>, x: GateId, y: GateId) -> GateId {
        match self.luts {
            None => single_wire(circuit.and_gate(x, y)),
            Some(luts) => {
                let t = single_wire(circuit.add_gate(x, y));
                single_wire(circuit.public_lookup_gate(t, luts.add_floor))
            }
        }
    }

    /// Whether an incoming carry passes through a limb whose digit sum is `s`, that is, whether
    /// `s = base - 1`. For bits this is `s` itself.
    #[inline]
    fn propagate(&self, circuit: &mut PolyCircuit<P>, s: GateId, one: GateId) -> GateId {
        match self.luts {
            None => s,
            Some(_) => self.add_floor(circuit, s, one),
        }
    }

    #[inline]
    fn mul_mod_floor(
        &self,
        circuit: &mut PolyCircuit<P>,
        x: GateId,
        y: GateId,
        output_floor: bool,
    ) -> (GateId, Option<GateId>) {
        match self.luts {
            None => (single_wire(circuit.and_gate(x, y)), None),
            Some(luts) => {
                let shifted = single_wire(
                    circuit.large_scalar_mul(y, std::slice::from_ref(&self.scalar_base)),
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

    /// Normalizes a carry-save representation `(sum_vec, carry_vec)` into ordinary digits,
    /// keeping the lowest `max_limbs` limbs. This uses the same generate/propagate structure as
    /// [`CarryArithPoly::add`].
    #[inline]
    pub(crate) fn final_cpa(
        &self,
        circuit: &mut PolyCircuit<P>,
        mut sum_vec: Vec<GateId>,
        carry_vec: Vec<GateId>,
        max_limbs: usize,
    ) -> Vec<GateId> {
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
            let p = self.propagate(circuit, s, one);
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

    /// Kogge-Stone-style prefix scan: afterwards `gs[i]` tells whether the carry/borrow into limb
    /// `i + 1` is forced by any less-significant suffix.
    ///
    /// Each generate signal excludes its propagate signal, and the combine step preserves this,
    /// so for bits the OR of the combine step is an XOR.
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
            for k in d..w {
                let gj = gs[k - d];
                let pj = ps[k - d];
                let gk = gs[k];
                let pk = ps[k];
                match self.luts {
                    None => {
                        let pk_and_gj = single_wire(circuit.and_gate(pk, gj));
                        gs_next[k] = single_wire(circuit.xor_gate(gk, pk_and_gj));
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

    /// Compresses every column of `sum_k base^k sum(columns[k])` to at most two digits.
    ///
    /// Each compressor replaces several digits of one column by one digit of that column and one
    /// carry of the next column, so the represented integer is preserved exactly and carries may
    /// open new columns. Bits are reduced by [`Self::compress_bit_columns`].
    pub(crate) fn compress_columns(
        &self,
        circuit: &mut PolyCircuit<P>,
        mut columns: Columns,
    ) -> Columns {
        if self.luts.is_none() {
            return Self::compress_bit_columns(circuit, columns);
        }
        let base = 1usize << self.limb_bit_size;
        let comp_rate = if base == 2 { 3 } else { base - 1 };
        while columns.iter().any(|column| column.len() > 2) {
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
                        let luts = self.luts.expect("bit columns are compressed by Dadda");
                        let mut sum = col[idx];
                        for item in &col[(idx + 1)..(idx + group_len)] {
                            sum = single_wire(circuit.add_gate(sum, *item));
                        }
                        let digit = single_wire(circuit.public_lookup_gate(sum, luts.add_mod));
                        let carry = single_wire(circuit.public_lookup_gate(sum, luts.add_floor));
                        next[k].push(digit);
                        next[k + 1].push(carry);
                    }
                    idx += group_len;
                }
            }
            if next.last().is_some_and(|v| v.is_empty()) {
                next.pop();
            }
            columns = next;
        }
        columns
    }

    /// Dadda's reduction of bit columns to at most two bits each.
    ///
    /// Stage `j` lowers every column to the height `d_j` of the sequence `2, 3, 4, 6, 9, ...`
    /// (`d_(j+1) = floor(3 d_j / 2)`), counting the carries the stage sends into the column, with
    /// full adders (three bits to a bit and a carry) and half adders (two bits to a bit and a
    /// carry). Every adder reads bits of the previous stage only, so a stage has the depth of one
    /// full adder and `h` bits per column need `O(log h)` stages; a carry never ripples through a
    /// stage.
    fn compress_bit_columns(circuit: &mut PolyCircuit<P>, mut columns: Columns) -> Columns {
        let max_height = columns.iter().map(Vec::len).max().unwrap_or(0);
        let mut targets = vec![2usize];
        while let Some(&target) = targets.last().filter(|&&target| target < max_height) {
            targets.push(target * 3 / 2);
        }
        for &target in targets.iter().rev().filter(|&&target| target < max_height) {
            let mut next: Columns = vec![vec![]; columns.len() + 1];
            let mut carries_in = 0;
            for (k, column) in columns.iter().enumerate() {
                let mut height = column.len() + carries_in;
                let mut idx = 0;
                let mut carries_out = 0;
                while height > target {
                    let (digit, carry) = if height - target >= 2 {
                        let (a, b, c) = (column[idx], column[idx + 1], column[idx + 2]);
                        idx += 3;
                        height -= 2;
                        let and_ab = single_wire(circuit.and_gate(a, b));
                        let xor_ab = Self::xor_with_and(circuit, a, b, and_ab);
                        let and_c = single_wire(circuit.and_gate(xor_ab, c));
                        let digit = Self::xor_with_and(circuit, xor_ab, c, and_c);
                        // `a b` and `c (a XOR b)` are never both one, so their OR is their XOR.
                        (digit, single_wire(circuit.xor_gate(and_ab, and_c)))
                    } else {
                        let (a, b) = (column[idx], column[idx + 1]);
                        idx += 2;
                        height -= 1;
                        let and_ab = single_wire(circuit.and_gate(a, b));
                        (Self::xor_with_and(circuit, a, b, and_ab), and_ab)
                    };
                    next[k].push(digit);
                    next[k + 1].push(carry);
                    carries_out += 1;
                }
                next[k].extend_from_slice(&column[idx..]);
                carries_in = carries_out;
            }
            if next.last().is_some_and(|v| v.is_empty()) {
                next.pop();
            }
            columns = next;
        }
        debug_assert!(columns.iter().all(|column| column.len() <= 2));
        columns
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

    pub fn zero(
        ctx: Arc<CarryArithPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        bit_size: usize,
    ) -> Self {
        let limb_len = bit_size.div_ceil(ctx.limb_bit_size);
        let limbs = vec![single_wire(circuit.const_zero_gate()); limb_len];
        Self { ctx, limbs, _p: PhantomData }
    }

    /// The constant `value` in `num_limbs` limbs. Limbs zero and one are the constant zero and
    /// one wires; larger limbs are small-scalar multiples of one.
    pub fn constant(
        ctx: Arc<CarryArithPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        num_limbs: usize,
        value: &BigUint,
    ) -> Self {
        let mask = (BigUint::from(1u32) << ctx.limb_bit_size) - BigUint::from(1u32);
        let zero = single_wire(circuit.const_zero_gate());
        let one = single_wire(circuit.const_one_gate());
        let mut remaining = value.clone();
        let limbs = (0..num_limbs)
            .map(|_| {
                let digit = (&remaining & &mask).to_u32().expect("a limb digit fits in u32");
                remaining >>= ctx.limb_bit_size;
                match digit {
                    0 => zero,
                    1 => one,
                    digit => single_wire(circuit.const_digits(&[digit])),
                }
            })
            .collect();
        assert!(remaining.is_zero(), "the constant {value} does not fit in {num_limbs} limbs");
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

    /// The exact integer `sum_k base^k sum(columns[k])` in ordinary limbs: the columns are
    /// compressed to a carry-save pair and added once, with one extra limb for the final carry.
    pub fn from_columns(
        ctx: Arc<CarryArithPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        columns: Columns,
    ) -> Self {
        let columns = ctx.compress_columns(circuit, columns);
        let zero = single_wire(circuit.const_zero_gate());
        let width = columns.len() + 1;
        let mut sum_vec = Vec::with_capacity(width);
        let mut carry_vec = Vec::with_capacity(width);
        for column in &columns {
            sum_vec.push(column.first().copied().unwrap_or(zero));
            carry_vec.push(column.get(1).copied().unwrap_or(zero));
        }
        sum_vec.push(zero);
        carry_vec.push(zero);
        let limbs = ctx.final_cpa(circuit, sum_vec, carry_vec, width);
        Self { ctx, limbs, _p: PhantomData }
    }

    #[inline]
    pub fn extend_size(&self, circuit: &mut PolyCircuit<P>, new_bit_size: usize) -> Self {
        debug_assert!(new_bit_size >= self.bit_size());
        debug_assert_eq!(new_bit_size % self.ctx.limb_bit_size, 0);
        let limb_len = new_bit_size / self.ctx.limb_bit_size;
        let mut limbs = self.limbs.clone();
        if limbs.len() < limb_len {
            limbs.resize(limb_len, single_wire(circuit.const_zero_gate()));
        }
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
            let bi = if i < b.len() { b[i] } else { zero };
            let (s, g) = self.ctx.add_mod_floor(circuit, ai, bi);
            let p = self.ctx.propagate(circuit, s, one);
            ss.push(s);
            gs.push(g);
            ps.push(p);
        }

        let (g_pref, _) = self.ctx.prefix_gp(circuit, &gs, &ps);

        let mut limbs = Vec::with_capacity(w + 1);
        for i in 0..w {
            let carry_in = if i == 0 { zero } else { g_pref[i - 1] };
            let digit = self.ctx.add_mod(circuit, ss[i], carry_in);
            limbs.push(digit);
        }
        let last_carry = if w == 0 { zero } else { g_pref[w - 1] };
        limbs.push(last_carry);
        Self { ctx: self.ctx.clone(), limbs, _p: PhantomData }
    }

    pub fn less_than(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> (GateId, Self) {
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        debug_assert_eq!(self.ctx, other.ctx);

        let w = self.limbs.len();
        let zero = single_wire(circuit.const_zero_gate());
        let one = single_wire(circuit.const_one_gate());
        let lut_constants = self.ctx.luts.map(|_| {
            let base = 1u32 << self.ctx.limb_bit_size;
            let base_minus_one = single_wire(circuit.const_digits(&[base - 1]));
            let base = single_wire(circuit.const_digits(&[base]));
            (base_minus_one, base)
        });

        let mut g = Vec::with_capacity(w);
        let mut p = Vec::with_capacity(w);
        let mut bit_xors = Vec::with_capacity(w);

        // This mirrors addition, but for borrow propagation.
        // Each limb reports whether it generates a borrow (`g`) or only propagates one (`p`).
        for i in 0..w {
            let a = self.limbs[i];
            let b = other.limbs[i];
            let (g_i, p_i) = match (self.ctx.luts, lut_constants) {
                (Some(luts), Some((base_minus_one, _))) => {
                    let y = single_wire(circuit.sub_gate(base_minus_one, b));
                    let (s, h) = self.ctx.add_mod_floor(circuit, a, y);
                    let eq = self.ctx.add_floor(circuit, s, one);
                    let not_h = single_wire(circuit.not_gate(h));
                    let not_eq = single_wire(circuit.not_gate(eq));
                    let two_not_eq = single_wire(circuit.add_gate(not_eq, not_eq));
                    let key = single_wire(circuit.add_gate(not_h, two_not_eq));
                    let gi = single_wire(circuit.public_lookup_gate(key, luts.kss_p));
                    (gi, eq)
                }
                _ => {
                    let and_ab = single_wire(circuit.and_gate(a, b));
                    let xor_ab = CarryArithPolyContext::xor_with_and(circuit, a, b, and_ab);
                    let not_a = single_wire(circuit.not_gate(a));
                    let gi = single_wire(circuit.and_gate(not_a, b));
                    let pi = single_wire(circuit.not_gate(xor_ab));
                    bit_xors.push(xor_ab);
                    (gi, pi)
                }
            };
            g.push(g_i);
            p.push(p_i);
        }

        let (g_pref, _) = self.ctx.prefix_gp(circuit, &g, &p);

        let mut diff_limbs = Vec::with_capacity(w);
        for i in 0..w {
            let b_in = if i == 0 { zero } else { g_pref[i - 1] };
            match lut_constants {
                None => diff_limbs.push(single_wire(circuit.xor_gate(bit_xors[i], b_in))),
                Some((_, base)) => {
                    let b_comp = single_wire(circuit.sub_gate(base, other.limbs[i]));
                    let y = single_wire(circuit.sub_gate(b_comp, b_in));
                    diff_limbs.push(self.ctx.add_mod(circuit, self.limbs[i], y));
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
        let limbs = self.ctx.final_cpa(circuit, sum_vec, carry_vec, max_limbs);
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

    /// Returns `self` when `selector` is one and `other` when it is zero.
    pub fn cmux(&self, other: &Self, selector: GateId, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.limbs.len(), other.limbs.len());
        let limbs = match self.ctx.luts {
            // `(self AND selector) XOR (other AND NOT selector)`: the two terms are exclusive.
            None => {
                let not = single_wire(circuit.not_gate(selector));
                self.limbs
                    .iter()
                    .zip(&other.limbs)
                    .map(|(&selected, &unselected)| {
                        let chosen = single_wire(circuit.and_gate(selected, selector));
                        let kept = single_wire(circuit.and_gate(unselected, not));
                        single_wire(circuit.xor_gate(chosen, kept))
                    })
                    .collect()
            }
            Some(_) => {
                let not = single_wire(circuit.not_gate(selector));
                self.limbs
                    .iter()
                    .zip(&other.limbs)
                    .map(|(&selected, &unselected)| {
                        let (case1, _) = self.ctx.mul_mod_floor(circuit, selected, selector, false);
                        let (case2, _) = self.ctx.mul_mod_floor(circuit, unselected, not, false);
                        single_wire(circuit.add_gate(case1, case2))
                    })
                    .collect()
            }
        };
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

    /// Schoolbook partial products grouped by output column `i + j`, compressed to a
    /// carry-save pair `(sum_vec, carry_vec)` of equal weights.
    #[inline]
    pub(crate) fn mul_without_cpa(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_limbs: usize,
    ) -> (Vec<GateId>, Vec<GateId>) {
        let mut columns: Columns = vec![vec![]; max_limbs];
        for (i, &ai) in self.limbs.iter().enumerate() {
            for (j, &bj) in other.limbs.iter().enumerate() {
                let k = i + j;
                if k >= max_limbs {
                    continue;
                }
                let (lo, hi) = self.ctx.mul_mod_floor(circuit, ai, bj, k + 1 < max_limbs);
                columns[k].push(lo);
                if let Some(hi) = hi {
                    columns[k + 1].push(hi);
                }
            }
        }

        let columns = self.ctx.compress_columns(circuit, columns);
        let zero = single_wire(circuit.const_zero_gate());
        columns
            .iter()
            .map(|column| {
                (column.first().copied().unwrap_or(zero), column.get(1).copied().unwrap_or(zero))
            })
            .unzip()
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
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        test_utils::execute_circuit,
        utils::gen_biguint_for_modulus,
    };
    use mxx_backends::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::Zero;
    use std::sync::Arc;

    fn test_parameters() -> DCRTPolyParams {
        DCRTPolyParams::new(4, 2, 16, 8, None, None)
    }

    fn execute_polys(
        name: &str,
        parameters: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<DCRTPoly>,
    ) -> Vec<DCRTPoly> {
        let inputs = inputs
            .into_iter()
            .map(|poly| DCRTPolyMatrix::from_poly_vec_row(parameters, vec![poly]))
            .collect::<Vec<_>>();
        execute_circuit(name, parameters, circuit, &inputs)
            .into_iter()
            .map(|matrix| matrix.entry(0, 0).clone())
            .collect()
    }

    fn limb_len(parameters: &DCRTPolyParams, limb_bit_size: usize) -> usize {
        parameters.modulus_bits().div_ceil(limb_bit_size)
    }

    fn random_value(parameters: &DCRTPolyParams) -> BigUint {
        let mut rng = rand::rng();
        gen_biguint_for_modulus(&mut rng, parameters.modulus().as_ref())
    }

    fn test_add_case(limb_bit_size: usize, left_value: BigUint, right_value: BigUint) {
        let parameters = test_parameters();
        let limb_len = limb_len(&parameters, limb_bit_size);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(CarryArithPolyContext::setup(&mut circuit, limb_bit_size, false));
        let input_bits = limb_len * limb_bit_size;
        let left = CarryArithPoly::input(context.clone(), &mut circuit, input_bits);
        let right = CarryArithPoly::input(context, &mut circuit, input_bits);
        let sum = left.add(&right, &mut circuit);
        circuit.output(sum.limbs);

        let mut inputs = encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &left_value);
        inputs.extend(encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &right_value));
        let actual = execute_polys("carry-add-runtime", &parameters, &circuit, inputs);
        let expected = encode_carry_arith_poly(
            limb_bit_size,
            limb_len + 1,
            &parameters,
            &(left_value + right_value),
        );
        assert_eq!(actual, expected);
    }

    fn test_less_than_case(limb_bit_size: usize, left_value: BigUint, right_value: BigUint) {
        let parameters = test_parameters();
        let limb_len = limb_len(&parameters, limb_bit_size);
        let input_bits = limb_len * limb_bit_size;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(CarryArithPolyContext::setup(&mut circuit, limb_bit_size, false));
        let left = CarryArithPoly::input(context.clone(), &mut circuit, input_bits);
        let right = CarryArithPoly::input(context, &mut circuit, input_bits);
        let (is_less, difference) = left.less_than(&right, &mut circuit);
        circuit.output(std::iter::once(is_less).chain(difference.limbs));

        let mut inputs = encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &left_value);
        inputs.extend(encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &right_value));
        let actual = execute_polys("carry-less-than-runtime", &parameters, &circuit, inputs);
        let (expected_flag, expected_difference) = if left_value < right_value {
            (BigUint::from(1u8), (BigUint::from(1u8) << input_bits) + left_value - right_value)
        } else {
            (BigUint::zero(), left_value - right_value)
        };
        let mut expected = encode_carry_arith_poly(limb_bit_size, 1, &parameters, &expected_flag);
        expected.extend(encode_carry_arith_poly(
            limb_bit_size,
            limb_len,
            &parameters,
            &expected_difference,
        ));
        assert_eq!(actual, expected);
    }

    fn test_mul_case(limb_bit_size: usize, left_value: BigUint, right_value: BigUint) {
        let parameters = test_parameters();
        let limb_len = limb_len(&parameters, limb_bit_size);
        let input_bits = limb_len * limb_bit_size;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(CarryArithPolyContext::setup(&mut circuit, limb_bit_size, false));
        let left = CarryArithPoly::input(context.clone(), &mut circuit, input_bits);
        let right = CarryArithPoly::input(context, &mut circuit, input_bits);
        let product = left.mul(&right, &mut circuit, None);
        circuit.output(product.limbs);

        let mut inputs = encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &left_value);
        inputs.extend(encode_carry_arith_poly(limb_bit_size, limb_len, &parameters, &right_value));
        let actual = execute_polys("carry-mul-runtime", &parameters, &circuit, inputs);
        let expected = encode_carry_arith_poly(
            limb_bit_size,
            2 * limb_len,
            &parameters,
            &(left_value * right_value),
        );
        assert_eq!(actual, expected);
    }

    fn boundary_values(parameters: &DCRTPolyParams) -> (BigUint, BigUint) {
        (BigUint::zero(), parameters.modulus().as_ref() - BigUint::from(1u8))
    }

    fn test_random_cases(limb_bit_size: usize) {
        let parameters = test_parameters();
        let left = random_value(&parameters);
        let right = random_value(&parameters);
        test_add_case(limb_bit_size, left.clone(), right.clone());
        test_less_than_case(limb_bit_size, left.clone(), right.clone());
        test_mul_case(limb_bit_size, left, right);
    }

    fn test_boundary_cases(limb_bit_size: usize) {
        let parameters = test_parameters();
        let (minimum, maximum) = boundary_values(&parameters);
        test_add_case(limb_bit_size, minimum.clone(), maximum.clone());
        test_add_case(limb_bit_size, maximum.clone(), maximum.clone());
        test_less_than_case(limb_bit_size, minimum.clone(), maximum.clone());
        test_less_than_case(limb_bit_size, maximum.clone(), minimum.clone());
        test_mul_case(limb_bit_size, minimum, maximum.clone());
        test_mul_case(limb_bit_size, maximum.clone(), maximum);
    }

    #[test]
    fn lookup_backed_limbs_match_random_integer_arithmetic_at_runtime() {
        test_random_cases(3);
    }

    #[test]
    fn lookup_backed_limbs_match_boundary_integer_arithmetic_at_runtime() {
        test_boundary_cases(3);
    }

    #[test]
    fn bit_limbs_match_random_integer_arithmetic_at_runtime() {
        test_random_cases(1);
    }

    #[test]
    fn bit_limbs_match_boundary_integer_arithmetic_at_runtime() {
        test_boundary_cases(1);
    }
}
