use super::carry_arith::{CarryArithPoly, CarryArithPolyContext, encode_carry_arith_poly};
use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    gadgets::arith::{
        DecomposeArithmeticGadget, ModularArithmeticContext, ModularArithmeticGadget,
    },
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    utils::{mod_inverse, mod_inverse_biguints},
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryPolyContext<P: Poly> {
    pub params: P::Params,
    pub carry_arith_ctx: Arc<CarryArithPolyContext<P>>,
    pub modulus: Arc<BigUint>,
    pub q_moduli: Vec<u64>,
    pub q_moduli_big: Vec<BigUint>,
    pub reconst_coeffs: Vec<Vec<Vec<BigUint>>>,
    pub num_limbs: usize,
    pub moduli_poly: Vec<CarryArithPoly<P>>,
    pub r2_poly: Vec<CarryArithPoly<P>>,
    pub moduli_prime_poly: Vec<CarryArithPoly<P>>,
}

impl<P: Poly + 'static> MontgomeryPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        dummy_scalar: bool,
    ) -> Self {
        let modulus: Arc<BigUint> = params.modulus().into();
        let (q_moduli, _, _) = params.to_crt();
        assert!(!q_moduli.is_empty(), "MontgomeryPolyContext requires at least one CRT modulus");
        let q_moduli_big = q_moduli.iter().copied().map(BigUint::from).collect::<Vec<_>>();
        let reconst_coeffs = Self::precompute_reconst_coeffs(&q_moduli);
        let num_limbs = q_moduli_big
            .iter()
            .map(|q_i| q_i.bits() as usize)
            .max()
            .expect("q_moduli_big must be non-empty")
            .div_ceil(limb_bit_size);
        let carry_arith_ctx =
            Arc::new(CarryArithPolyContext::setup(circuit, params, limb_bit_size, dummy_scalar));

        let r_bits = limb_bit_size * num_limbs;
        let r = BigUint::one() << r_bits;
        let moduli_poly = q_moduli_big
            .iter()
            .map(|q_i| {
                Self::constant_carry_arith_poly(
                    circuit,
                    params,
                    carry_arith_ctx.clone(),
                    num_limbs,
                    q_i,
                )
            })
            .collect::<Vec<_>>();
        let r2_poly = q_moduli_big
            .iter()
            .map(|q_i| {
                let r2_value = (&r * &r) % q_i;
                Self::constant_carry_arith_poly(
                    circuit,
                    params,
                    carry_arith_ctx.clone(),
                    num_limbs,
                    &r2_value,
                )
            })
            .collect::<Vec<_>>();
        let moduli_prime_poly = q_moduli_big
            .iter()
            .map(|q_i| {
                let moduli_prime_value = Self::calculate_modulus_prime(q_i, &r);
                Self::constant_carry_arith_poly(
                    circuit,
                    params,
                    carry_arith_ctx.clone(),
                    num_limbs,
                    &moduli_prime_value,
                )
            })
            .collect::<Vec<_>>();

        Self {
            params: params.clone(),
            carry_arith_ctx,
            modulus,
            q_moduli,
            q_moduli_big,
            reconst_coeffs,
            num_limbs,
            moduli_poly,
            r2_poly,
            moduli_prime_poly,
        }
    }

    fn calculate_modulus_prime(modulus: &BigUint, r: &BigUint) -> BigUint {
        let n_inv =
            mod_inverse_biguints(modulus, r).expect("Montgomery modulus must be coprime with R");
        r - n_inv
    }

    fn constant_carry_arith_poly(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        ctx: Arc<CarryArithPolyContext<P>>,
        num_limbs_per_value: usize,
        value: &BigUint,
    ) -> CarryArithPoly<P> {
        let limbs = encode_carry_arith_poly(ctx.limb_bit_size, num_limbs_per_value, params, value);
        CarryArithPoly::const_limbs(ctx, circuit, &limbs)
    }

    fn precompute_reconst_coeffs(q_moduli: &[u64]) -> Vec<Vec<Vec<BigUint>>> {
        (0..q_moduli.len())
            .into_par_iter()
            .map(|level_offset| {
                (1..=(q_moduli.len() - level_offset))
                    .into_par_iter()
                    .map(|levels| {
                        Self::window_reconst_coeffs(
                            &q_moduli[level_offset..(level_offset + levels)],
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn window_reconst_coeffs(active_moduli: &[u64]) -> Vec<BigUint> {
        let active_modulus = active_moduli
            .par_iter()
            .map(|&q_i| BigUint::from(q_i))
            .reduce(BigUint::one, |acc, modulus| acc * modulus);
        active_moduli
            .par_iter()
            .map(|&q_i| {
                let q_i_big = BigUint::from(q_i);
                let q_over_qi = &active_modulus / &q_i_big;
                let q_over_qi_mod =
                    (&q_over_qi % &q_i_big).to_u64().expect("active CRT residue must fit in u64");
                let inv = mod_inverse(q_over_qi_mod, q_i)
                    .expect("active CRT moduli must remain pairwise coprime");
                (&q_over_qi * BigUint::from(inv)) % &active_modulus
            })
            .collect()
    }

    fn active_range(
        &self,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) -> std::ops::Range<usize> {
        let levels = <Self as ModularArithmeticContext<P>>::active_levels(
            self,
            enable_levels,
            Some(level_offset),
        );
        level_offset..(level_offset + levels)
    }

    fn reconst_coeffs_for_window(
        &self,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) -> &[BigUint] {
        let levels = self.active_range(enable_levels, level_offset).len();
        &self.reconst_coeffs[level_offset][levels - 1]
    }

    fn active_modulus(&self, enable_levels: Option<usize>, level_offset: usize) -> BigUint {
        self.q_moduli[self.active_range(enable_levels, level_offset)]
            .par_iter()
            .map(|&q_i| BigUint::from(q_i))
            .reduce(BigUint::one, |acc, modulus| acc * modulus)
    }

    fn sparse_native_constant(
        &self,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        value: &BigUint,
    ) -> BigUint {
        let active_modulus = self.active_modulus(enable_levels, level_offset);
        (value * &self.reconst_coeffs_for_window(enable_levels, level_offset)[target_q_idx]) %
            active_modulus
    }
}

#[derive(Debug, Clone)]
pub struct MontgomeryPoly<P: Poly> {
    pub ctx: Arc<MontgomeryPolyContext<P>>,
    pub value: Vec<CarryArithPoly<P>>,
    pub level_offset: usize,
    pub enable_levels: Option<usize>,
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

impl<P: Poly + 'static> MontgomeryPoly<P> {
    fn validate_window(
        ctx: &MontgomeryPolyContext<P>,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) {
        let levels = <MontgomeryPolyContext<P> as ModularArithmeticContext<P>>::active_levels(
            ctx,
            enable_levels,
            Some(level_offset),
        );
        assert!(levels > 0, "MontgomeryPoly must keep at least one active q level");
    }

    fn default_max_plaintexts(
        ctx: &MontgomeryPolyContext<P>,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) -> Vec<BigUint> {
        ctx.active_range(enable_levels, level_offset)
            .into_par_iter()
            .map(|q_idx| &ctx.q_moduli_big[q_idx] - BigUint::from(1u64))
            .collect()
    }

    fn default_p_max_traces(
        ctx: &MontgomeryPolyContext<P>,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) -> Vec<BigUint> {
        Self::default_max_plaintexts(ctx, enable_levels, level_offset)
    }

    fn with_metadata(
        ctx: Arc<MontgomeryPolyContext<P>>,
        value: Vec<CarryArithPoly<P>>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset);
        let levels = ctx.active_range(enable_levels, level_offset).len();
        assert_eq!(value.len(), levels, "MontgomeryPoly value rows must match active q levels");
        assert_eq!(
            max_plaintexts.len(),
            levels,
            "MontgomeryPoly metadata must match active q levels",
        );
        assert_eq!(
            p_max_traces.len(),
            levels,
            "MontgomeryPoly trace metadata must match active q levels",
        );
        Self { ctx, value, level_offset, enable_levels, max_plaintexts, p_max_traces }
    }

    fn bit_size(ctx: &MontgomeryPolyContext<P>) -> usize {
        ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size
    }

    fn resolve_enable_levels(&self) -> usize {
        self.enable_levels.unwrap_or(self.value.len())
    }

    fn zero_level(ctx: &MontgomeryPolyContext<P>) -> CarryArithPoly<P> {
        CarryArithPoly::zero(ctx.carry_arith_ctx.clone(), Self::bit_size(ctx))
    }

    fn digit_radix(ctx: &MontgomeryPolyContext<P>) -> BigUint {
        BigUint::one() << ctx.carry_arith_ctx.limb_bit_size
    }

    fn active_range(&self) -> std::ops::Range<usize> {
        self.level_offset..(self.level_offset + self.resolve_enable_levels())
    }

    pub fn new(ctx: Arc<MontgomeryPolyContext<P>>, value: Vec<CarryArithPoly<P>>) -> Self {
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref(), None, 0);
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref(), None, 0);
        Self::with_metadata(ctx, value, None, Some(0), max_plaintexts, p_max_traces)
    }

    pub fn input(ctx: Arc<MontgomeryPolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let bit_size = Self::bit_size(ctx.as_ref());
        let values = (0..ctx.q_moduli_depth())
            .map(|_| CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size))
            .collect::<Vec<_>>();
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref(), None, 0);
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref(), None, 0);
        Self::with_metadata(ctx, values, None, Some(0), max_plaintexts, p_max_traces)
    }

    fn pad_value(
        ctx: &MontgomeryPolyContext<P>,
        mut value: CarryArithPoly<P>,
    ) -> CarryArithPoly<P> {
        if value.limbs.len() != ctx.num_limbs {
            value = value.extend_size(Self::bit_size(ctx));
        }
        value
    }

    pub fn from_regular_values(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<MontgomeryPolyContext<P>>,
        values: Vec<CarryArithPoly<P>>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        let range = ctx.active_range(enable_levels, level_offset);
        assert_eq!(values.len(), range.len(), "regular Montgomery inputs must match active levels");
        let mont_values = values
            .into_iter()
            .enumerate()
            .map(|(local_q_idx, value)| {
                let q_idx = range.start + local_q_idx;
                let value = Self::pad_value(ctx.as_ref(), value);
                let r2_mul = value.mul(&ctx.r2_poly[q_idx], circuit, None);
                Self::montgomery_reduce_level(ctx.as_ref(), q_idx, circuit, &r2_mul)
            })
            .collect::<Vec<_>>();
        let max_plaintexts =
            Self::default_max_plaintexts(ctx.as_ref(), enable_levels, level_offset);
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref(), enable_levels, level_offset);
        Self::with_metadata(
            ctx,
            mont_values,
            enable_levels,
            Some(level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.level_offset, other.level_offset);
        debug_assert_eq!(self.enable_levels, other.enable_levels);
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.carry_arith_ctx.limb_bit_size;
        let reduced = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs, rhs), q_idx)| {
                let sum_full = lhs.add(rhs, circuit);
                let n_ext = self.ctx.moduli_poly[q_idx].extend_size(n_ext_bits);
                let (is_less, diff) = sum_full.less_than(&n_ext, circuit);
                let reduced_full = sum_full.cmux(&diff, is_less, circuit);
                reduced_full.mod_limbs(self.ctx.num_limbs)
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::add", circuit.count_gates_by_type_vec());
        Self::with_metadata(
            self.ctx.clone(),
            reduced,
            self.enable_levels,
            Some(self.level_offset),
            Self::default_max_plaintexts(self.ctx.as_ref(), self.enable_levels, self.level_offset),
            Self::default_p_max_traces(self.ctx.as_ref(), self.enable_levels, self.level_offset),
        )
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.level_offset, other.level_offset);
        debug_assert_eq!(self.enable_levels, other.enable_levels);
        let result = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs, rhs), q_idx)| {
                let (is_less, raw_sub) = lhs.less_than(rhs, circuit);
                let added = raw_sub
                    .add(&self.ctx.moduli_poly[q_idx], circuit)
                    .mod_limbs(self.ctx.num_limbs);
                added.cmux(&raw_sub, is_less, circuit)
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::sub", circuit.count_gates_by_type_vec());
        Self::with_metadata(
            self.ctx.clone(),
            result,
            self.enable_levels,
            Some(self.level_offset),
            Self::default_max_plaintexts(self.ctx.as_ref(), self.enable_levels, self.level_offset),
            Self::default_p_max_traces(self.ctx.as_ref(), self.enable_levels, self.level_offset),
        )
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.level_offset, other.level_offset);
        debug_assert_eq!(self.enable_levels, other.enable_levels);
        let reduced = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs, rhs), q_idx)| {
                let product = lhs.mul(rhs, circuit, None);
                Self::montgomery_reduce_level(self.ctx.as_ref(), q_idx, circuit, &product)
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::mul", circuit.count_gates_by_type_vec());
        Self::with_metadata(
            self.ctx.clone(),
            reduced,
            self.enable_levels,
            Some(self.level_offset),
            Self::default_max_plaintexts(self.ctx.as_ref(), self.enable_levels, self.level_offset),
            Self::default_p_max_traces(self.ctx.as_ref(), self.enable_levels, self.level_offset),
        )
    }

    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> Vec<CarryArithPoly<P>> {
        self.value
            .iter()
            .zip(self.active_range())
            .map(|(value, q_idx)| {
                Self::montgomery_reduce_level(self.ctx.as_ref(), q_idx, circuit, value)
            })
            .collect()
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct_value(circuit)
    }

    fn montgomery_reduce_level(
        ctx: &MontgomeryPolyContext<P>,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
        t: &CarryArithPoly<P>,
    ) -> CarryArithPoly<P> {
        let r = ctx.num_limbs;
        let limb_bits = ctx.carry_arith_ctx.limb_bit_size;

        let t_low = t.mod_limbs(r);
        let m = t_low.mul(&ctx.moduli_prime_poly[q_idx], circuit, Some(r * limb_bits));
        let m_times_n = m.mul(&ctx.moduli_poly[q_idx], circuit, None);
        let u = t.add(&m_times_n, circuit);

        let u_shifted = u.left_shift(r);
        let n_ext_bits = (r + 1) * limb_bits;
        let n_ext = ctx.moduli_poly[q_idx].extend_size(n_ext_bits);
        let (is_less, diff) = u_shifted.less_than(&n_ext, circuit);
        let reduced_full = u_shifted.cmux(&diff, is_less, circuit);
        reduced_full.mod_limbs(r)
    }

    pub fn reconstruct_value(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let residues = self.to_regular(circuit);
        let reconst_coeffs =
            self.ctx.reconst_coeffs_for_window(self.enable_levels, self.level_offset);
        let mut sum = circuit.const_zero_gate();
        for (residue, reconst_coeff) in residues.iter().zip(reconst_coeffs.iter()) {
            let residue_gate = residue.finalize(circuit);
            let scaled =
                circuit.large_scalar_mul(residue_gate, std::slice::from_ref(reconst_coeff));
            sum = circuit.add_gate(sum, scaled);
        }
        sum.as_single_wire()
    }

    pub fn active_q_moduli(&self) -> Vec<u64> {
        self.ctx.q_moduli[self.active_range()].to_vec()
    }

    fn digit_value_poly(
        ctx: &MontgomeryPolyContext<P>,
        digit_gate: GateId,
        circuit: &mut PolyCircuit<P>,
    ) -> CarryArithPoly<P> {
        let mut limbs = vec![circuit.const_zero_gate().as_single_wire(); ctx.num_limbs];
        limbs[0] = digit_gate;
        CarryArithPoly::new(ctx.carry_arith_ctx.clone(), limbs)
    }

    fn sparse_regular_level_poly_with_metadata(
        ctx: Arc<MontgomeryPolyContext<P>>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_value: CarryArithPoly<P>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(target_q_idx < active_levels, "target_q_idx must lie within active_levels");
        let mut values =
            (0..active_levels).map(|_| Self::zero_level(ctx.as_ref())).collect::<Vec<_>>();
        values[target_q_idx] = Self::pad_value(ctx.as_ref(), target_value);
        let mont = Self::from_regular_values(
            circuit,
            ctx.clone(),
            values,
            enable_levels,
            Some(level_offset),
        );
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;
        Self::with_metadata(
            ctx,
            mont.value,
            enable_levels,
            Some(level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn native_sparse_gadget_constant(
        params: &P::Params,
        ctx: &MontgomeryPolyContext<P>,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        value: &BigUint,
    ) -> P {
        P::from_biguint_to_constant(
            params,
            ctx.sparse_native_constant(enable_levels, level_offset, target_q_idx, value),
        )
    }

    fn decompose_regular_value(ctx: &MontgomeryPolyContext<P>, value: &BigUint) -> Vec<BigUint> {
        let radix = Self::digit_radix(ctx);
        let mut digits = Vec::with_capacity(ctx.num_limbs);
        let mut remaining = value.clone();
        for _ in 0..ctx.num_limbs {
            digits.push(&remaining % &radix);
            remaining /= &radix;
        }
        digits
    }
}

impl<P: Poly + 'static> ModularArithmeticContext<P> for MontgomeryPolyContext<P> {
    fn register_local_in(&self, _circuit: &mut PolyCircuit<P>) -> Self {
        self.clone()
    }

    fn register_shared_in(
        &self,
        _source_circuit: &PolyCircuit<P>,
        _circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.clone()
    }

    fn q_moduli_depth(&self) -> usize {
        self.q_moduli.len()
    }

    fn decomposition_len(&self) -> usize {
        self.num_limbs
    }

    fn q_level_row_width(&self) -> usize {
        self.num_limbs
    }

    fn full_reduce_output_metadata(
        &self,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> (Vec<BigUint>, Vec<BigUint>) {
        let level_offset = level_offset.unwrap_or(0);
        MontgomeryPoly::<P>::validate_window(self, enable_levels, level_offset);
        (
            MontgomeryPoly::<P>::default_max_plaintexts(self, enable_levels, level_offset),
            MontgomeryPoly::<P>::default_p_max_traces(self, enable_levels, level_offset),
        )
    }

    fn reduced_p_max_trace(&self) -> BigUint {
        self.q_moduli_big.iter().max().expect("q_moduli_big must not be empty") -
            BigUint::from(1u64)
    }

    fn randomizer_decomposition_bound(&self) -> u64 {
        self.q_moduli.iter().copied().max().expect("q_moduli must not be empty").saturating_sub(1)
    }

    fn decomposition_term_bound(&self, _term_idx: usize) -> BigUint {
        MontgomeryPoly::<P>::digit_radix(self) - BigUint::from(1u64)
    }

    fn full_reduce_level_plaintext_bound(&self, q_idx: usize) -> BigUint {
        &self.q_moduli_big[q_idx] - BigUint::from(1u64)
    }

    fn plaintext_capacity_bound(&self) -> BigUint {
        self.modulus.as_ref().clone()
    }

    fn trace_capacity_bound(&self) -> BigUint {
        self.modulus.as_ref().clone()
    }
}

impl<P: Poly + 'static> ModularArithmeticGadget<P> for MontgomeryPoly<P> {
    type Context = MontgomeryPolyContext<P>;

    fn context(&self) -> &Arc<Self::Context> {
        &self.ctx
    }

    fn level_offset(&self) -> usize {
        self.level_offset
    }

    fn enable_levels(&self) -> Option<usize> {
        self.enable_levels
    }

    fn max_plaintexts(&self) -> &[BigUint] {
        &self.max_plaintexts
    }

    fn p_max_traces(&self) -> &[BigUint] {
        &self.p_max_traces
    }

    fn input(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset);
        let bit_size = Self::bit_size(ctx.as_ref());
        let levels = ctx.active_range(enable_levels, level_offset).len();
        let value = (0..levels)
            .map(|_| CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size))
            .collect::<Vec<_>>();
        let max_plaintexts =
            Self::default_max_plaintexts(ctx.as_ref(), enable_levels, level_offset);
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref(), enable_levels, level_offset);
        Self::with_metadata(
            ctx,
            value,
            enable_levels,
            Some(level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset);
        let bit_size = Self::bit_size(ctx.as_ref());
        let levels = ctx.active_range(enable_levels, level_offset).len();
        let value = (0..levels)
            .map(|_| CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size))
            .collect::<Vec<_>>();
        Self::with_metadata(
            ctx,
            value,
            enable_levels,
            Some(level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        self.active_q_moduli()
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        self.value
            .iter()
            .flat_map(|residue| residue.limbs.iter().copied().map(BatchedWire::single))
            .collect()
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = template.resolve_enable_levels();
        assert_eq!(
            outputs.len(),
            levels * template.ctx.num_limbs,
            "Montgomery flattened outputs must match active_levels * num_limbs",
        );
        let value = outputs
            .par_chunks(template.ctx.num_limbs)
            .map(|row| CarryArithPoly::new(template.ctx.carry_arith_ctx.clone(), row.to_vec()))
            .collect::<Vec<_>>();
        Self::with_metadata(
            template.ctx.clone(),
            value,
            template.enable_levels,
            Some(template.level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn q_level_row_batch(&self, q_idx: usize) -> BatchedWire {
        BatchedWire::from_batches(self.value[q_idx].limbs.iter().copied())
    }

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        _circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert!(target_q_idx < active_levels, "target_q_idx must lie within active_levels");
        let mut value = Vec::with_capacity(active_levels);
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        let mut target_row = Some(target_row.gate_ids().collect::<Vec<_>>());
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;
        for q_idx in 0..active_levels {
            if q_idx == target_q_idx {
                value.push(CarryArithPoly::new(
                    ctx.carry_arith_ctx.clone(),
                    target_row.take().expect("target row must be present exactly once"),
                ));
            } else {
                value.push(Self::zero_level(ctx.as_ref()));
            }
        }
        Self::with_metadata(
            ctx,
            value,
            enable_levels,
            Some(level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let lowered_src_slots = src_slots
            .iter()
            .map(|(src_slot, scalar)| {
                let scalar = scalar
                    .as_ref()
                    .map(|values| u32::try_from(values[0]).expect("slot-transfer scalar fits u32"));
                (*src_slot, scalar)
            })
            .collect::<Vec<_>>();
        let value = self
            .value
            .iter()
            .map(|residue| {
                let limbs = residue
                    .limbs
                    .iter()
                    .map(|&gate_id| {
                        circuit.slot_transfer_gate(gate_id, &lowered_src_slots).as_single_wire()
                    })
                    .collect::<Vec<_>>();
                CarryArithPoly::new(self.ctx.carry_arith_ctx.clone(), limbs)
            })
            .collect::<Vec<_>>();
        Self::with_metadata(
            self.ctx.clone(),
            value,
            self.enable_levels,
            Some(self.level_offset),
            self.max_plaintexts.clone(),
            self.p_max_traces.clone(),
        )
    }

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        MontgomeryPoly::add(self, other, circuit)
    }

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        MontgomeryPoly::sub(self, other, circuit)
    }

    fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        MontgomeryPoly::mul(self, other, circuit)
    }

    fn full_reduce(&self, _circuit: &mut PolyCircuit<P>) -> Self {
        self.clone()
    }

    fn prepare_for_reconstruct(&self, _circuit: &mut PolyCircuit<P>) -> Self {
        self.clone()
    }

    fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        assert_eq!(
            tower_constants.len(),
            self.resolve_enable_levels(),
            "MontgomeryPoly const_mul requires one scalar per active q-level"
        );
        let constants = tower_constants
            .iter()
            .map(|constant| {
                MontgomeryPolyContext::constant_carry_arith_poly(
                    circuit,
                    &self.ctx.params,
                    self.ctx.carry_arith_ctx.clone(),
                    self.ctx.num_limbs,
                    &BigUint::from(*constant),
                )
            })
            .collect::<Vec<_>>();
        let mont_constant = MontgomeryPoly::from_regular_values(
            circuit,
            self.ctx.clone(),
            constants,
            self.enable_levels,
            Some(self.level_offset),
        );
        self.mul(&mont_constant, circuit)
    }

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct_value(circuit)
    }
}

impl<P: Poly + 'static> DecomposeArithmeticGadget<P> for MontgomeryPoly<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M {
        let level_offset = level_offset.unwrap_or(0);
        let active_levels = ctx.active_levels(enable_levels, Some(level_offset));
        let radix = Self::digit_radix(ctx);
        let row = (0..active_levels * ctx.num_limbs)
            .into_par_iter()
            .map(|entry_idx| {
                let q_idx = entry_idx / ctx.num_limbs;
                let digit_idx = entry_idx % ctx.num_limbs;
                let value = radix
                    .pow(u32::try_from(digit_idx).expect("Montgomery digit index must fit in u32"));
                Self::native_sparse_gadget_constant(
                    params,
                    ctx,
                    enable_levels,
                    level_offset,
                    q_idx,
                    &value,
                )
            })
            .collect::<Vec<_>>();
        M::from_poly_vec_row(params, row)
    }

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> M {
        let level_offset = level_offset.unwrap_or(0);
        let active_q_moduli = &ctx.q_moduli[ctx.active_range(enable_levels, level_offset)];
        let gadget_len = active_q_moduli.len() * ctx.num_limbs;
        let (row_size, col_size) = target.size();
        let entry_polys = (0..row_size * col_size)
            .into_par_iter()
            .map(|entry_idx| {
                let row_idx = entry_idx / col_size;
                let col_idx = entry_idx % col_size;
                let coeffs = target.entry(row_idx, col_idx).coeffs_biguints();
                let mut output_polys =
                    (0..gadget_len).map(|_| vec![BigUint::ZERO; coeffs.len()]).collect::<Vec<_>>();
                for (coeff_idx, coeff) in coeffs.iter().enumerate() {
                    for (q_idx, &q_i) in active_q_moduli.iter().enumerate() {
                        let residue = coeff % BigUint::from(q_i);
                        for (digit_idx, digit) in
                            Self::decompose_regular_value(ctx, &residue).into_iter().enumerate()
                        {
                            let target_row = q_idx * ctx.num_limbs + digit_idx;
                            output_polys[target_row][coeff_idx] = ctx.sparse_native_constant(
                                enable_levels,
                                level_offset,
                                q_idx,
                                &digit,
                            );
                        }
                    }
                }
                (
                    row_idx,
                    col_idx,
                    output_polys
                        .into_iter()
                        .map(|coeffs| P::from_biguints(params, &coeffs))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let mut decomposed = (0..row_size * gadget_len)
            .map(|_| vec![P::const_zero(params); col_size])
            .collect::<Vec<_>>();
        for (row_idx, col_idx, output_polys) in entry_polys {
            for (target_row, poly) in output_polys.into_iter().enumerate() {
                decomposed[row_idx * gadget_len + target_row][col_idx] = poly;
            }
        }

        M::from_poly_vec(params, decomposed)
    }

    fn gadget_decomposition_norm_bound(
        ctx: &Self::Context,
        _enable_levels: Option<usize>,
        _level_offset: Option<usize>,
    ) -> BigUint {
        Self::digit_radix(ctx) - BigUint::from(1u64)
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        let level_offset = level_offset.unwrap_or(0);
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset);
        let active_levels = ctx.active_range(enable_levels, level_offset).len();
        let radix = Self::digit_radix(ctx.as_ref());
        let mut gadget = Vec::with_capacity(active_levels * ctx.num_limbs);
        for q_idx in 0..active_levels {
            for digit_idx in 0..ctx.num_limbs {
                let value = radix
                    .pow(u32::try_from(digit_idx).expect("Montgomery digit index must fit in u32"));
                let carry = MontgomeryPolyContext::constant_carry_arith_poly(
                    circuit,
                    &ctx.params,
                    ctx.carry_arith_ctx.clone(),
                    ctx.num_limbs,
                    &value,
                );
                gadget.push(Self::sparse_regular_level_poly_with_metadata(
                    ctx.clone(),
                    active_levels,
                    enable_levels,
                    level_offset,
                    q_idx,
                    carry,
                    value.clone(),
                    value.clone(),
                    circuit,
                ));
            }
        }
        gadget
    }

    fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        let regular_rows = self.to_regular(circuit);
        let active_levels = self.resolve_enable_levels();
        let digit_bound = Self::digit_radix(self.ctx.as_ref()) - BigUint::from(1u64);
        let mut decomposition = Vec::with_capacity(active_levels * self.ctx.num_limbs);
        for (q_idx, regular_row) in regular_rows.into_iter().enumerate() {
            for digit_gate in regular_row.limbs {
                let digit_value = Self::digit_value_poly(self.ctx.as_ref(), digit_gate, circuit);
                decomposition.push(Self::sparse_regular_level_poly_with_metadata(
                    self.ctx.clone(),
                    active_levels,
                    self.enable_levels,
                    self.level_offset,
                    q_idx,
                    digit_value,
                    digit_bound.clone(),
                    digit_bound.clone(),
                    circuit,
                ));
            }
        }
        decomposition
    }

    fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        let regular_rows = self.to_regular(circuit);
        let row = &regular_rows[q_idx].limbs;
        debug_assert_eq!(row.len(), self.ctx.num_limbs);
        (
            row[..row.len().saturating_sub(1)].to_vec(),
            *row.last().expect("Montgomery decomposition requires at least one limb"),
        )
    }

    fn conv_mul_right_decomposed_many(
        &self,
        _params: &P::Params,
        left_rows: &[&[Self]],
        _num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        let decomposed_rhs = self.gadget_decompose(circuit);
        left_rows
            .iter()
            .map(|row| {
                assert_eq!(
                    row.len(),
                    decomposed_rhs.len(),
                    "Montgomery decomposed rows must match the gadget decomposition width",
                );
                let mut row_iter = row.iter().zip(decomposed_rhs.iter());
                let (first_left, first_right) =
                    row_iter.next().expect("Montgomery decomposition width must be non-zero");
                row_iter.fold(first_left.mul(first_right, circuit), |acc, (left, right)| {
                    let product = left.mul(right, circuit);
                    acc.add(&product, circuit)
                })
            })
            .collect()
    }
}

pub fn encode_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    encode_montgomery_poly_with_window(limb_bit_size, params, input, None, 0)
}

pub fn encode_montgomery_poly_with_window<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    input: &BigUint,
    enable_levels: Option<usize>,
    level_offset: usize,
) -> Vec<P> {
    let (q_moduli, _, _) = params.to_crt();
    let levels = enable_levels.unwrap_or(q_moduli.len().saturating_sub(level_offset));
    assert!(level_offset + levels <= q_moduli.len(), "active Montgomery q range exceeds CRT depth",);
    let num_limbs = q_moduli
        .iter()
        .copied()
        .map(|q_i| BigUint::from(q_i).bits() as usize)
        .max()
        .expect("q_moduli must not be empty")
        .div_ceil(limb_bit_size);
    let r = BigUint::one() << (limb_bit_size * num_limbs);
    q_moduli[level_offset..(level_offset + levels)]
        .par_iter()
        .map(|q_i| {
            let modulus = BigUint::from(*q_i);
            let montgomery_value = ((input % &modulus) * (&r % &modulus)) % &modulus;
            encode_carry_arith_poly(limb_bit_size, num_limbs, params, &montgomery_value)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        __PAIR, __TestState,
        element::PolyElem,
        gadgets::arith::{DecomposeArithmeticGadget, ModularArithmeticGadget},
        lookup::poly::PolyPltEvaluator,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        utils::gen_biguint_for_modulus,
    };

    const LIMB_BIT_SIZE: usize = 3;
    const TEST_CRT_DEPTH: usize = 4;
    const TEST_CRT_BITS: usize = 16;
    const TEST_BASE_BITS: u32 = 8;
    const TEST_RING_DIM: u32 = 4;

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

    fn create_test_context_with_limb_size(
        circuit: &mut PolyCircuit<DCRTPoly>,
        params: DCRTPolyParams,
        limb_bit_size: usize,
    ) -> (DCRTPolyParams, Arc<MontgomeryPolyContext<DCRTPoly>>) {
        let ctx = Arc::new(MontgomeryPolyContext::setup(circuit, &params, limb_bit_size, false));
        (params, ctx)
    }

    fn default_test_params() -> DCRTPolyParams {
        DCRTPolyParams::new(TEST_RING_DIM, TEST_CRT_DEPTH, TEST_CRT_BITS, TEST_BASE_BITS)
    }

    fn active_modulus(
        params: &DCRTPolyParams,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) -> BigUint {
        let (q_moduli, _, _) = params.to_crt();
        let levels = enable_levels.unwrap_or(q_moduli.len().saturating_sub(level_offset));
        q_moduli[level_offset..(level_offset + levels)]
            .par_iter()
            .map(|q_i| BigUint::from(*q_i))
            .reduce(BigUint::one, |acc, modulus| acc * modulus)
    }

    fn eval_const_term(
        circuit: &PolyCircuit<DCRTPoly>,
        params: &DCRTPolyParams,
        inputs: &[DCRTPoly],
    ) -> BigUint {
        let outputs = eval_with_const_one(circuit, params, inputs);
        assert_eq!(outputs.len(), 1, "circuit should emit a single reconstruction output");
        outputs[0]
            .coeffs()
            .into_iter()
            .next()
            .expect("output polynomial must have a constant coefficient")
            .value()
            .clone()
    }

    fn build_window_input(
        ctx: Arc<MontgomeryPolyContext<DCRTPoly>>,
        enable_levels: Option<usize>,
        level_offset: usize,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> MontgomeryPoly<DCRTPoly> {
        <MontgomeryPoly<DCRTPoly> as ModularArithmeticGadget<DCRTPoly>>::input(
            ctx,
            enable_levels,
            Some(level_offset),
            circuit,
        )
    }

    fn test_montgomery_roundtrip_case_with_window(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let input = build_window_input(ctx.clone(), enable_levels, level_offset, &mut circuit);
        let output = input.finalize(&mut circuit);
        circuit.output(vec![output]);

        let eval_inputs = encode_montgomery_poly_with_window(
            limb_bit_size,
            &params,
            &value,
            enable_levels,
            level_offset,
        );
        let result = eval_const_term(&circuit, &params, &eval_inputs);
        let modulus = active_modulus(&params, enable_levels, level_offset);
        assert_eq!(result % &modulus, value.clone() % &modulus);
    }

    fn test_montgomery_binary_case_with_window<F>(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        lhs_value: BigUint,
        rhs_value: BigUint,
        op: F,
        expected: BigUint,
    ) where
        F: Fn(
            &MontgomeryPoly<DCRTPoly>,
            &MontgomeryPoly<DCRTPoly>,
            &mut PolyCircuit<DCRTPoly>,
        ) -> MontgomeryPoly<DCRTPoly>,
    {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let lhs = build_window_input(ctx.clone(), enable_levels, level_offset, &mut circuit);
        let rhs = build_window_input(ctx.clone(), enable_levels, level_offset, &mut circuit);
        let output = op(&lhs, &rhs, &mut circuit).finalize(&mut circuit);
        circuit.output(vec![output]);

        let mut eval_inputs = encode_montgomery_poly_with_window(
            limb_bit_size,
            &params,
            &lhs_value,
            enable_levels,
            level_offset,
        );
        eval_inputs.extend(encode_montgomery_poly_with_window(
            limb_bit_size,
            &params,
            &rhs_value,
            enable_levels,
            level_offset,
        ));
        let result = eval_const_term(&circuit, &params, &eval_inputs);
        let modulus = active_modulus(&params, enable_levels, level_offset);
        assert_eq!(result % &modulus, expected % &modulus);
    }

    fn test_montgomery_gadget_decompose_identity_with_window(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let input = build_window_input(ctx.clone(), enable_levels, level_offset, &mut circuit);
        let gadget = MontgomeryPoly::<DCRTPoly>::gadget_vector(
            ctx.clone(),
            enable_levels,
            Some(level_offset),
            &mut circuit,
        );
        let decomposition = input.gadget_decompose(&mut circuit);
        assert_eq!(gadget.len(), decomposition.len());
        let recomposed = gadget
            .iter()
            .zip(decomposition.iter())
            .fold(None, |acc: Option<MontgomeryPoly<DCRTPoly>>, (g, d)| {
                let product = g.mul(d, &mut circuit);
                Some(match acc {
                    Some(acc) => acc.add(&product, &mut circuit),
                    None => product,
                })
            })
            .expect("Montgomery decomposition width must be non-zero");
        let output = recomposed.finalize(&mut circuit);
        circuit.output(vec![output]);

        let eval_inputs = encode_montgomery_poly_with_window(
            limb_bit_size,
            &params,
            &value,
            enable_levels,
            level_offset,
        );
        let result = eval_const_term(&circuit, &params, &eval_inputs);
        let modulus = active_modulus(&params, enable_levels, level_offset);
        assert_eq!(result % &modulus, value.clone() % &modulus);

        let native_gadget = MontgomeryPoly::<DCRTPoly>::gadget_matrix::<DCRTPolyMatrix>(
            &params,
            ctx.as_ref(),
            enable_levels,
            Some(level_offset),
        );
        let native_decomp = MontgomeryPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
            &params,
            ctx.as_ref(),
            &DCRTPolyMatrix::from_poly_vec(
                &params,
                vec![vec![DCRTPoly::from_biguint_to_constant(&params, value.clone())]],
            ),
            enable_levels,
            Some(level_offset),
        );
        assert_eq!(
            native_gadget.col_size(),
            native_decomp.row_size(),
            "native Montgomery gadget matrix width must match the decomposition height",
        );
    }

    fn random_value(modulus: &BigUint) -> BigUint {
        let mut rng = rand::rng();
        gen_biguint_for_modulus(&mut rng, modulus)
    }

    fn run_roundtrip_cases(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) {
        let modulus = active_modulus(&params, enable_levels, level_offset);
        let cases = [BigUint::ZERO, &modulus - BigUint::from(1u64), random_value(&modulus)];
        for value in cases {
            test_montgomery_roundtrip_case_with_window(
                params.clone(),
                limb_bit_size,
                enable_levels,
                level_offset,
                value,
            );
        }
    }

    fn run_add_cases(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) {
        let modulus = active_modulus(&params, enable_levels, level_offset);
        let random_lhs = random_value(&modulus);
        let random_rhs = random_value(&modulus);
        let cases = [
            (BigUint::ZERO, &modulus - BigUint::from(1u64)),
            (&modulus - BigUint::from(1u64), BigUint::ZERO),
            (random_lhs, random_rhs),
        ];
        for (lhs, rhs) in cases {
            let expected = (&lhs + &rhs) % &modulus;
            test_montgomery_binary_case_with_window(
                params.clone(),
                limb_bit_size,
                enable_levels,
                level_offset,
                lhs,
                rhs,
                |lhs, rhs, circuit| lhs.add(rhs, circuit),
                expected,
            );
        }
    }

    fn run_sub_cases(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) {
        let modulus = active_modulus(&params, enable_levels, level_offset);
        let random_lhs = random_value(&modulus);
        let random_rhs = random_value(&modulus);
        let cases = [
            (BigUint::ZERO, &modulus - BigUint::from(1u64)),
            (&modulus - BigUint::from(1u64), BigUint::ZERO),
            (random_lhs, random_rhs),
        ];
        for (lhs, rhs) in cases {
            let expected = (&lhs + &modulus - &rhs) % &modulus;
            test_montgomery_binary_case_with_window(
                params.clone(),
                limb_bit_size,
                enable_levels,
                level_offset,
                lhs,
                rhs,
                |lhs, rhs, circuit| lhs.sub(rhs, circuit),
                expected,
            );
        }
    }

    fn run_mul_cases(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
    ) {
        let modulus = active_modulus(&params, enable_levels, level_offset);
        let random_lhs = random_value(&modulus);
        let random_rhs = random_value(&modulus);
        let cases = [
            (BigUint::ZERO, &modulus - BigUint::from(1u64)),
            (&modulus - BigUint::from(1u64), &modulus - BigUint::from(1u64)),
            (random_lhs, random_rhs),
        ];
        for (lhs, rhs) in cases {
            let expected = (&lhs * &rhs) % &modulus;
            test_montgomery_binary_case_with_window(
                params.clone(),
                limb_bit_size,
                enable_levels,
                level_offset,
                lhs,
                rhs,
                |lhs, rhs, circuit| lhs.mul(rhs, circuit),
                expected,
            );
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_roundtrip_reconstruct_full_depth() {
        let params = default_test_params();
        run_roundtrip_cases(params, LIMB_BIT_SIZE, None, 0);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_roundtrip_reconstruct_partial_levels() {
        let params = default_test_params();
        run_roundtrip_cases(params, LIMB_BIT_SIZE, Some(2), 1);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_add_reconstruct_full_depth() {
        let params = default_test_params();
        run_add_cases(params, LIMB_BIT_SIZE, None, 0);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_add_reconstruct_partial_levels() {
        let params = default_test_params();
        run_add_cases(params, LIMB_BIT_SIZE, Some(2), 1);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_sub_reconstruct_full_depth() {
        let params = default_test_params();
        run_sub_cases(params, LIMB_BIT_SIZE, None, 0);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_sub_reconstruct_partial_levels() {
        let params = default_test_params();
        run_sub_cases(params, LIMB_BIT_SIZE, Some(2), 1);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_mul_reconstruct_full_depth() {
        let params = default_test_params();
        run_mul_cases(params, LIMB_BIT_SIZE, None, 0);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_mul_reconstruct_partial_levels() {
        let params = default_test_params();
        run_mul_cases(params, LIMB_BIT_SIZE, Some(2), 1);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_roundtrip_reconstruct_full_depth_limb1() {
        let params = default_test_params();
        let modulus = active_modulus(&params, None, 0);
        let value = random_value(&modulus);
        test_montgomery_roundtrip_case_with_window(params, 1, None, 0, value);
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_gadget_decompose_identity_full_depth() {
        let params = default_test_params();
        let modulus = active_modulus(&params, None, 0);
        let value = random_value(&modulus);
        test_montgomery_gadget_decompose_identity_with_window(
            params,
            LIMB_BIT_SIZE,
            None,
            0,
            value,
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_montgomery_gadget_decompose_identity_partial_levels() {
        let params = default_test_params();
        let modulus = active_modulus(&params, Some(2), 1);
        let value = random_value(&modulus);
        test_montgomery_gadget_decompose_identity_with_window(
            params,
            LIMB_BIT_SIZE,
            Some(2),
            1,
            value,
        );
    }
}
