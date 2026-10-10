//! Montgomery arithmetic modulo every q-level of a CRT basis, over carry-arithmetic limbs.
//!
//! A [`MontgomeryPoly`] holds a polynomial of `N` coefficients: at every active q-level `q_i`
//! each coefficient is the Montgomery residue `x R mod q_i` for the radix `R = 2^(w L)` of `L`
//! limbs of `w` bits. Additions, subtractions, and products act coefficient by coefficient, and
//! the Ring-GSW product (`DecomposeArithmeticGadget`) multiplies negacyclically in
//! `Z_q[X] / (X^N + 1)`.
//!
//! With one-bit limbs (`w = 1`) every operation is a Boolean circuit of AND, XOR, and NOT gates
//! and no public lookup table is registered. Wider limbs use the lookup tables of
//! [`CarryArithPoly`], and the Ring-GSW product requires one-bit limbs.

use super::carry_arith::{CarryArithPoly, CarryArithPolyContext, encode_carry_arith_poly};
use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    circuit_gadgets::{
        arith::{
            BinaryPlannerResult, CrtWindow, DecomposeArithmeticGadget, ModularArithmeticContext,
            ModularArithmeticGadget, ModularArithmeticPlanner,
        },
        conv_mul::montgomery::negacyclic_decomposed_products,
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
pub struct MontgomeryPlannerMetadata {
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MontgomeryAddPlanKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MontgomerySubPlanKey;

/// The public constants of Montgomery arithmetic over the CRT basis of `params`.
///
/// The context holds no gate of a particular circuit: constants become wires of the circuit that
/// uses them, so one context also serves the sub-circuits built from that circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryPolyContext<P: Poly> {
    pub params: P::Params,
    pub carry_arith_ctx: Arc<CarryArithPolyContext<P>>,
    pub modulus: Arc<BigUint>,
    pub q_moduli: Vec<u64>,
    pub q_moduli_big: Vec<BigUint>,
    pub reconst_coeffs: Vec<Vec<Vec<BigUint>>>,
    pub num_limbs: usize,
    /// `R^2 mod q_i` for the Montgomery radix `R`.
    pub r2_values: Vec<BigUint>,
    /// `-q_i^(-1) mod R`.
    pub moduli_prime_values: Vec<BigUint>,
}

/// The constant wires of one q-level used by a Montgomery reduction.
struct LevelConstants<P: Poly> {
    modulus: CarryArithPoly<P>,
    modulus_prime: CarryArithPoly<P>,
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
            Arc::new(CarryArithPolyContext::setup(circuit, limb_bit_size, dummy_scalar));
        let r = BigUint::one() << (limb_bit_size * num_limbs);
        let r2_values = q_moduli_big.par_iter().map(|q_i| (&r * &r) % q_i).collect();
        let moduli_prime_values =
            q_moduli_big.par_iter().map(|q_i| Self::calculate_modulus_prime(q_i, &r)).collect();

        Self {
            params: params.clone(),
            carry_arith_ctx,
            modulus,
            q_moduli,
            q_moduli_big,
            reconst_coeffs,
            num_limbs,
            r2_values,
            moduli_prime_values,
        }
    }

    fn calculate_modulus_prime(modulus: &BigUint, r: &BigUint) -> BigUint {
        let n_inv =
            mod_inverse_biguints(modulus, r).expect("Montgomery modulus must be coprime with R");
        r - n_inv
    }

    /// The Montgomery radix `R = 2^(limb_bit_size num_limbs)`.
    fn radix(&self) -> BigUint {
        BigUint::one() << (self.carry_arith_ctx.limb_bit_size * self.num_limbs)
    }

    /// The Montgomery residue `value R mod q_i` of the q-level `q_idx` of the full basis.
    pub(crate) fn montgomery_residue(&self, q_idx: usize, value: &BigUint) -> BigUint {
        let q_i = &self.q_moduli_big[q_idx];
        ((value % q_i) * (self.radix() % q_i)) % q_i
    }

    /// The constant `value` as `num_limbs` limbs of `circuit`.
    pub(crate) fn constant_level(
        &self,
        circuit: &mut PolyCircuit<P>,
        value: &BigUint,
    ) -> CarryArithPoly<P> {
        CarryArithPoly::constant(self.carry_arith_ctx.clone(), circuit, self.num_limbs, value)
    }

    fn level_constants(&self, circuit: &mut PolyCircuit<P>, q_idx: usize) -> LevelConstants<P> {
        LevelConstants {
            modulus: self.constant_level(circuit, &self.q_moduli_big[q_idx]),
            modulus_prime: self.constant_level(circuit, &self.moduli_prime_values[q_idx]),
        }
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

    fn active_range(&self, window: CrtWindow) -> std::ops::Range<usize> {
        let window = self.validate_window(window);
        window.offset..window.end()
    }

    fn reconst_coeffs_for_window(&self, window: CrtWindow) -> &[BigUint] {
        &self.reconst_coeffs[window.offset][window.depth - 1]
    }

    fn active_modulus(&self, window: CrtWindow) -> BigUint {
        self.q_moduli[self.active_range(window)]
            .par_iter()
            .map(|&q_i| BigUint::from(q_i))
            .reduce(BigUint::one, |acc, modulus| acc * modulus)
    }

    fn sparse_native_constant(
        &self,
        window: CrtWindow,
        target_q_idx: usize,
        value: &BigUint,
    ) -> BigUint {
        let active_modulus = self.active_modulus(window);
        (value * &self.reconst_coeffs_for_window(window)[target_q_idx]) % active_modulus
    }
}

#[derive(Debug, Clone)]
pub struct MontgomeryPoly<P: Poly> {
    pub ctx: Arc<MontgomeryPolyContext<P>>,
    /// `value[level][coefficient]`: the Montgomery residue of one coefficient at one active
    /// q-level.
    pub value: Vec<Vec<CarryArithPoly<P>>>,
    pub window: CrtWindow,
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

impl<P: Poly + 'static> MontgomeryPoly<P> {
    fn validate_window(ctx: &MontgomeryPolyContext<P>, window: CrtWindow) {
        ctx.validate_window(window);
    }

    fn default_max_plaintexts(ctx: &MontgomeryPolyContext<P>, window: CrtWindow) -> Vec<BigUint> {
        ctx.active_range(window)
            .into_par_iter()
            .map(|q_idx| &ctx.q_moduli_big[q_idx] - BigUint::from(1u64))
            .collect()
    }

    fn default_p_max_traces(ctx: &MontgomeryPolyContext<P>, window: CrtWindow) -> Vec<BigUint> {
        Self::default_max_plaintexts(ctx, window)
    }

    fn with_metadata(
        ctx: Arc<MontgomeryPolyContext<P>>,
        value: Vec<Vec<CarryArithPoly<P>>>,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        Self::validate_window(ctx.as_ref(), window);
        let levels = window.depth;
        assert_eq!(value.len(), levels, "MontgomeryPoly value rows must match active q levels");
        let num_coefficients = value[0].len();
        assert!(num_coefficients > 0, "MontgomeryPoly requires at least one coefficient");
        assert!(
            value.iter().all(|level| level.len() == num_coefficients),
            "every MontgomeryPoly q-level must hold the same number of coefficients",
        );
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
        Self { ctx, value, window, max_plaintexts, p_max_traces }
    }

    /// An entry whose metadata bounds every residue by `q_i - 1`.
    pub(crate) fn normalized(
        ctx: Arc<MontgomeryPolyContext<P>>,
        value: Vec<Vec<CarryArithPoly<P>>>,
        window: CrtWindow,
    ) -> Self {
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref(), window);
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref(), window);
        Self::with_metadata(ctx, value, window, max_plaintexts, p_max_traces)
    }

    fn planner_metadata(&self) -> MontgomeryPlannerMetadata {
        MontgomeryPlannerMetadata {
            max_plaintexts: self.max_plaintexts.clone(),
            p_max_traces: self.p_max_traces.clone(),
        }
    }

    fn normalized_planner_metadata(
        ctx: &MontgomeryPolyContext<P>,
        window: CrtWindow,
    ) -> MontgomeryPlannerMetadata {
        Self::validate_window(ctx, window);
        MontgomeryPlannerMetadata {
            max_plaintexts: Self::default_max_plaintexts(ctx, window),
            p_max_traces: Self::default_p_max_traces(ctx, window),
        }
    }

    fn bit_size(ctx: &MontgomeryPolyContext<P>) -> usize {
        ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size
    }

    fn zero_coefficients(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        num_coefficients: usize,
    ) -> Vec<CarryArithPoly<P>> {
        let zero = CarryArithPoly::zero(ctx.carry_arith_ctx.clone(), circuit, Self::bit_size(ctx));
        vec![zero; num_coefficients]
    }

    fn digit_radix(ctx: &MontgomeryPolyContext<P>) -> BigUint {
        BigUint::one() << ctx.carry_arith_ctx.limb_bit_size
    }

    fn active_range(&self) -> std::ops::Range<usize> {
        self.window.offset..self.window.end()
    }

    pub fn num_coefficients(&self) -> usize {
        self.value[0].len()
    }

    fn input_value(
        ctx: &MontgomeryPolyContext<P>,
        num_coefficients: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Vec<CarryArithPoly<P>>> {
        Self::validate_window(ctx, window);
        let bit_size = Self::bit_size(ctx);
        (0..window.depth)
            .map(|_| {
                (0..num_coefficients)
                    .map(|_| CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size))
                    .collect()
            })
            .collect()
    }

    fn pad_value(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        value: CarryArithPoly<P>,
    ) -> CarryArithPoly<P> {
        if value.limbs.len() < ctx.num_limbs {
            value.extend_size(circuit, Self::bit_size(ctx))
        } else {
            value
        }
    }

    /// Converts regular residues `x < q_i` of the q-level `q_idx` to Montgomery residues.
    fn montgomery_level(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        q_idx: usize,
        values: Vec<CarryArithPoly<P>>,
    ) -> Vec<CarryArithPoly<P>> {
        let r2 = ctx.constant_level(circuit, &ctx.r2_values[q_idx]);
        let constants = ctx.level_constants(circuit, q_idx);
        values
            .into_iter()
            .map(|value| {
                let value = Self::pad_value(ctx, circuit, value);
                let r2_mul = value.mul(&r2, circuit, None);
                Self::montgomery_reduce(ctx, circuit, &r2_mul, &constants)
            })
            .collect()
    }

    pub fn from_regular_values(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<MontgomeryPolyContext<P>>,
        values: Vec<Vec<CarryArithPoly<P>>>,
        window: CrtWindow,
    ) -> Self {
        let range = ctx.active_range(window);
        assert_eq!(values.len(), range.len(), "regular Montgomery inputs must match active levels");
        let mont_values = values
            .into_iter()
            .zip(range)
            .map(|(level, q_idx)| Self::montgomery_level(ctx.as_ref(), circuit, q_idx, level))
            .collect::<Vec<_>>();
        Self::normalized(ctx, mont_values, window)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.window, other.window);
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.carry_arith_ctx.limb_bit_size;
        let reduced = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs_level, rhs_level), q_idx)| {
                let n_ext = self
                    .ctx
                    .constant_level(circuit, &self.ctx.q_moduli_big[q_idx])
                    .extend_size(circuit, n_ext_bits);
                lhs_level
                    .iter()
                    .zip(rhs_level)
                    .map(|(lhs, rhs)| {
                        let sum_full = lhs.add(rhs, circuit);
                        let (is_less, diff) = sum_full.less_than(&n_ext, circuit);
                        let reduced_full = sum_full.cmux(&diff, is_less, circuit);
                        reduced_full.mod_limbs(self.ctx.num_limbs)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::add", circuit.count_gates_by_type_vec());
        Self::normalized(self.ctx.clone(), reduced, self.window)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.window, other.window);
        let result = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs_level, rhs_level), q_idx)| {
                let modulus = self.ctx.constant_level(circuit, &self.ctx.q_moduli_big[q_idx]);
                lhs_level
                    .iter()
                    .zip(rhs_level)
                    .map(|(lhs, rhs)| {
                        let (is_less, raw_sub) = lhs.less_than(rhs, circuit);
                        let added = raw_sub.add(&modulus, circuit).mod_limbs(self.ctx.num_limbs);
                        added.cmux(&raw_sub, is_less, circuit)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::sub", circuit.count_gates_by_type_vec());
        Self::normalized(self.ctx.clone(), result, self.window)
    }

    /// The coefficient-wise product.
    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        debug_assert_eq!(self.window, other.window);
        let reduced = self
            .value
            .iter()
            .zip(other.value.iter())
            .zip(self.active_range())
            .map(|((lhs_level, rhs_level), q_idx)| {
                let constants = self.ctx.level_constants(circuit, q_idx);
                lhs_level
                    .iter()
                    .zip(rhs_level)
                    .map(|(lhs, rhs)| {
                        let product = lhs.mul(rhs, circuit, None);
                        Self::montgomery_reduce(self.ctx.as_ref(), circuit, &product, &constants)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        debug!("num gates {:?} at MontgomeryPoly::mul", circuit.count_gates_by_type_vec());
        Self::normalized(self.ctx.clone(), reduced, self.window)
    }

    /// The regular residues `x < q_i`, as `[level][coefficient]`. Every q-level registers one
    /// reduction sub-circuit and calls it once per coefficient.
    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> Vec<Vec<CarryArithPoly<P>>> {
        let num_limbs = self.ctx.num_limbs;
        self.value
            .iter()
            .zip(self.active_range())
            .map(|(level, q_idx)| {
                let mut reduction = circuit.fresh_sub_circuit();
                let input = CarryArithPoly::input(
                    self.ctx.carry_arith_ctx.clone(),
                    &mut reduction,
                    Self::bit_size(self.ctx.as_ref()),
                );
                let constants = self.ctx.level_constants(&mut reduction, q_idx);
                let regular =
                    Self::montgomery_reduce(self.ctx.as_ref(), &mut reduction, &input, &constants);
                reduction.output(regular.limbs);
                let reduction_id = circuit.register_sub_circuit(reduction);
                level
                    .iter()
                    .map(|value| {
                        let limbs = circuit
                            .call_sub_circuit(reduction_id, value.limbs.clone())
                            .into_iter()
                            .map(BatchedWire::as_single_wire)
                            .collect::<Vec<_>>();
                        debug_assert_eq!(limbs.len(), num_limbs);
                        CarryArithPoly::new(self.ctx.carry_arith_ctx.clone(), limbs)
                    })
                    .collect()
            })
            .collect()
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct_value(circuit)
    }

    /// REDC: `t R^(-1) mod q_i` for `t < q_i R`.
    fn montgomery_reduce(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        t: &CarryArithPoly<P>,
        constants: &LevelConstants<P>,
    ) -> CarryArithPoly<P> {
        let r = ctx.num_limbs;
        let limb_bits = ctx.carry_arith_ctx.limb_bit_size;

        let t_low = t.mod_limbs(r);
        let m = t_low.mul(&constants.modulus_prime, circuit, Some(r * limb_bits));
        let m_times_n = m.mul(&constants.modulus, circuit, None);
        let u = t.add(&m_times_n, circuit);

        let u_shifted = u.left_shift(r);
        let n_ext_bits = (r + 1) * limb_bits;
        let n_ext = constants.modulus.extend_size(circuit, n_ext_bits);
        let (is_less, diff) = u_shifted.less_than(&n_ext, circuit);
        let reduced_full = u_shifted.cmux(&diff, is_less, circuit);
        reduced_full.mod_limbs(r)
    }

    /// One gate whose plaintext is `sum_i X^i c_i` for the CRT reconstructions `c_i` of the
    /// coefficients.
    pub fn reconstruct_value(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let residues = self.to_regular(circuit);
        let reconst_coeffs = self.ctx.reconst_coeffs_for_window(self.window);
        let mut sum = circuit.const_zero_gate();
        for (residue_level, reconst_coeff) in residues.iter().zip(reconst_coeffs.iter()) {
            for (coefficient, residue) in residue_level.iter().enumerate() {
                let residue_gate = residue.finalize(circuit);
                let mut scalar = vec![BigUint::ZERO; coefficient + 1];
                scalar[coefficient] = reconst_coeff.clone();
                let scaled = circuit.large_scalar_mul(residue_gate, &scalar);
                sum = circuit.add_gate(sum, scaled);
            }
        }
        sum.as_single_wire()
    }

    /// One gate whose plaintext is `sum_i X^i gates[i]`.
    fn coefficient_embedding(circuit: &mut PolyCircuit<P>, gates: &[GateId]) -> GateId {
        let mut sum = gates[0];
        for (coefficient, &gate) in gates.iter().enumerate().skip(1) {
            let rotated = circuit.rotate_gate(gate, coefficient as u64);
            sum = circuit.add_gate(sum, rotated).as_single_wire();
        }
        sum
    }

    pub fn active_q_moduli(&self) -> Vec<u64> {
        self.ctx.q_moduli[self.active_range()].to_vec()
    }

    /// The entry whose only nonzero q-level is `target_q_idx`, holding the regular residues
    /// `target_values`, with the given metadata at that level.
    fn sparse_regular_level_poly_with_metadata(
        ctx: Arc<MontgomeryPolyContext<P>>,
        circuit: &mut PolyCircuit<P>,
        window: CrtWindow,
        target_q_idx: usize,
        target_values: Vec<CarryArithPoly<P>>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
    ) -> Self {
        let active_levels = window.depth;
        assert!(target_q_idx < active_levels, "target_q_idx must lie within active_levels");
        let num_coefficients = target_values.len();
        let target_level = Self::montgomery_level(
            ctx.as_ref(),
            circuit,
            window.offset + target_q_idx,
            target_values,
        );
        let mut target_level = Some(target_level);
        let value = (0..active_levels)
            .map(|q_idx| {
                if q_idx == target_q_idx {
                    target_level.take().expect("the target level is used once")
                } else {
                    Self::zero_coefficients(ctx.as_ref(), circuit, num_coefficients)
                }
            })
            .collect::<Vec<_>>();
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;
        Self::with_metadata(ctx, value, window, max_plaintexts, p_max_traces)
    }

    fn native_sparse_gadget_constant(
        params: &P::Params,
        ctx: &MontgomeryPolyContext<P>,
        window: CrtWindow,
        target_q_idx: usize,
        value: &BigUint,
    ) -> P {
        P::from_biguint_to_constant(params, ctx.sparse_native_constant(window, target_q_idx, value))
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
    fn q_moduli_depth(&self) -> usize {
        self.q_moduli.len()
    }

    fn decomposition_len(&self) -> usize {
        self.num_limbs
    }

    fn q_level_row_width(&self) -> usize {
        self.num_limbs
    }

    fn randomizer_decomposition_bound(&self) -> u64 {
        self.q_moduli.iter().copied().max().expect("q_moduli must not be empty").saturating_sub(1)
    }

    fn decomposition_term_bound(&self, _term_idx: usize) -> BigUint {
        MontgomeryPoly::<P>::digit_radix(self) - BigUint::from(1u64)
    }

    fn plaintext_capacity_bound(&self) -> BigUint {
        self.modulus.as_ref().clone()
    }
}

impl<P: Poly + 'static> ModularArithmeticGadget<P> for MontgomeryPoly<P> {
    type Context = MontgomeryPolyContext<P>;

    fn context(&self) -> &Arc<Self::Context> {
        &self.ctx
    }

    fn crt_window(&self) -> CrtWindow {
        self.window
    }

    fn max_plaintexts(&self) -> &[BigUint] {
        &self.max_plaintexts
    }

    fn p_max_traces(&self) -> &[BigUint] {
        &self.p_max_traces
    }

    fn num_coefficient_slots(&self) -> usize {
        self.num_coefficients()
    }

    fn input(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let value = Self::input_value(ctx.as_ref(), num_coefficient_slots, window, circuit);
        Self::normalized(ctx, value, window)
    }

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let value = Self::input_value(ctx.as_ref(), num_coefficient_slots, window, circuit);
        Self::with_metadata(ctx, value, window, max_plaintexts, p_max_traces)
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        self.active_q_moduli()
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        self.value
            .iter()
            .flatten()
            .flat_map(|residue| residue.limbs.iter().copied().map(BatchedWire::single))
            .collect()
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let levels = template.window.depth;
        let num_coefficients = template.num_coefficients();
        let num_limbs = template.ctx.num_limbs;
        assert_eq!(
            outputs.len(),
            levels * num_coefficients * num_limbs,
            "Montgomery flattened outputs must match active_levels * coefficients * num_limbs",
        );
        let value = outputs
            .par_chunks(num_coefficients * num_limbs)
            .map(|level| {
                level
                    .chunks(num_limbs)
                    .map(|limbs| {
                        CarryArithPoly::new(template.ctx.carry_arith_ctx.clone(), limbs.to_vec())
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        Self::with_metadata(
            template.ctx.clone(),
            value,
            template.window,
            max_plaintexts,
            p_max_traces,
        )
    }

    fn q_level_row_batch(&self, q_idx: usize) -> BatchedWire {
        BatchedWire::from_batches(
            self.value[q_idx].iter().flat_map(|residue| residue.limbs.iter().copied()),
        )
    }

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let active_levels = window.depth;
        assert!(target_q_idx < active_levels, "target_q_idx must lie within active_levels");
        let target_row = target_row.gate_ids().collect::<Vec<_>>();
        assert_eq!(
            target_row.len(),
            num_coefficient_slots * ctx.num_limbs,
            "a sparse Montgomery row must hold every limb of every coefficient",
        );
        let mut target_level = Some(
            target_row
                .chunks(ctx.num_limbs)
                .map(|limbs| CarryArithPoly::new(ctx.carry_arith_ctx.clone(), limbs.to_vec()))
                .collect::<Vec<_>>(),
        );
        let value = (0..active_levels)
            .map(|q_idx| {
                if q_idx == target_q_idx {
                    target_level.take().expect("the target level is used once")
                } else {
                    Self::zero_coefficients(ctx.as_ref(), circuit, num_coefficient_slots)
                }
            })
            .collect::<Vec<_>>();
        let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
        let mut p_max_traces = vec![BigUint::ZERO; active_levels];
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;
        Self::with_metadata(ctx, value, window, max_plaintexts, p_max_traces)
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
            .map(|level| {
                level
                    .iter()
                    .map(|residue| {
                        let limbs = residue
                            .limbs
                            .iter()
                            .map(|&gate_id| {
                                circuit
                                    .slot_transfer_gate(gate_id, &lowered_src_slots)
                                    .as_single_wire()
                            })
                            .collect::<Vec<_>>();
                        CarryArithPoly::new(self.ctx.carry_arith_ctx.clone(), limbs)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        Self::with_metadata(
            self.ctx.clone(),
            value,
            self.window,
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
            self.window.depth,
            "MontgomeryPoly const_mul requires one scalar per active q-level"
        );
        let value = self
            .value
            .iter()
            .zip(self.active_range())
            .zip(tower_constants)
            .map(|((level, q_idx), constant)| {
                // REDC(x R * c R) = x c R.
                let mont_constant = self.ctx.constant_level(
                    circuit,
                    &self.ctx.montgomery_residue(q_idx, &BigUint::from(*constant)),
                );
                let constants = self.ctx.level_constants(circuit, q_idx);
                level
                    .iter()
                    .map(|residue| {
                        let product = residue.mul(&mont_constant, circuit, None);
                        Self::montgomery_reduce(self.ctx.as_ref(), circuit, &product, &constants)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        Self::normalized(self.ctx.clone(), value, self.window)
    }

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct_value(circuit)
    }
}

impl<P: Poly + 'static> ModularArithmeticPlanner<P> for MontgomeryPoly<P> {
    type Metadata = MontgomeryPlannerMetadata;
    type AddPlanKey = MontgomeryAddPlanKey;
    type SubPlanKey = MontgomerySubPlanKey;

    fn metadata(entry: &Self) -> Self::Metadata {
        entry.planner_metadata()
    }

    fn normalized_metadata(ctx: &Self::Context, window: CrtWindow) -> Self::Metadata {
        Self::normalized_planner_metadata(ctx, window)
    }

    fn input_with_planner_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        metadata: &Self::Metadata,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
            circuit,
        )
    }

    fn from_flat_outputs_with_planner_metadata(
        template: &Self,
        outputs: &[GateId],
        metadata: &Self::Metadata,
    ) -> Self {
        Self::from_flat_outputs(
            template,
            outputs,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
        )
    }

    fn compute_add_plan_and_output(
        left: &Self,
        _right: &Self,
    ) -> BinaryPlannerResult<Self::AddPlanKey, Self::Metadata> {
        BinaryPlannerResult {
            cache_key: MontgomeryAddPlanKey,
            output_metadata: Self::normalized_planner_metadata(left.ctx.as_ref(), left.window),
        }
    }

    fn compute_sub_plan_and_output(
        left: &Self,
        _right: &Self,
    ) -> BinaryPlannerResult<Self::SubPlanKey, Self::Metadata> {
        BinaryPlannerResult {
            cache_key: MontgomerySubPlanKey,
            output_metadata: Self::normalized_planner_metadata(left.ctx.as_ref(), left.window),
        }
    }

    fn normalize_mul_input(entry: &Self, _circuit: &mut PolyCircuit<P>) -> Self {
        entry.clone()
    }
}

impl<P: Poly + 'static> DecomposeArithmeticGadget<P> for MontgomeryPoly<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        window: CrtWindow,
    ) -> M {
        let window = ctx.validate_window(window);
        let active_levels = window.depth;
        let radix = Self::digit_radix(ctx);
        let row = (0..active_levels * ctx.num_limbs)
            .into_par_iter()
            .map(|entry_idx| {
                let q_idx = entry_idx / ctx.num_limbs;
                let digit_idx = entry_idx % ctx.num_limbs;
                let value = radix
                    .pow(u32::try_from(digit_idx).expect("Montgomery digit index must fit in u32"));
                Self::native_sparse_gadget_constant(params, ctx, window, q_idx, &value)
            })
            .collect::<Vec<_>>();
        M::from_poly_vec_row(params, row)
    }

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        window: CrtWindow,
    ) -> M {
        let window = ctx.validate_window(window);
        let active_q_moduli = &ctx.q_moduli[ctx.active_range(window)];
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
                            output_polys[target_row][coeff_idx] =
                                ctx.sparse_native_constant(window, q_idx, &digit);
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

    fn gadget_decomposition_norm_bound(ctx: &Self::Context, _window: CrtWindow) -> BigUint {
        Self::digit_radix(ctx) - BigUint::from(1u64)
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        Self::validate_window(ctx.as_ref(), window);
        let active_levels = window.depth;
        let radix = Self::digit_radix(ctx.as_ref());
        let mut gadget = Vec::with_capacity(active_levels * ctx.num_limbs);
        for q_idx in 0..active_levels {
            for digit_idx in 0..ctx.num_limbs {
                let value = radix
                    .pow(u32::try_from(digit_idx).expect("Montgomery digit index must fit in u32"));
                // The constant polynomial `value` at q-level `q_idx`, already in Montgomery form.
                let mut target_level =
                    Self::zero_coefficients(ctx.as_ref(), circuit, num_coefficient_slots);
                target_level[0] = ctx.constant_level(
                    circuit,
                    &ctx.montgomery_residue(window.offset + q_idx, &value),
                );
                let mut target_level = Some(target_level);
                let entry_value = (0..active_levels)
                    .map(|level| {
                        if level == q_idx {
                            target_level.take().expect("the target level is used once")
                        } else {
                            Self::zero_coefficients(ctx.as_ref(), circuit, num_coefficient_slots)
                        }
                    })
                    .collect::<Vec<_>>();
                let mut max_plaintexts = vec![BigUint::ZERO; active_levels];
                let mut p_max_traces = vec![BigUint::ZERO; active_levels];
                max_plaintexts[q_idx] = value.clone();
                p_max_traces[q_idx] = value;
                gadget.push(Self::with_metadata(
                    ctx.clone(),
                    entry_value,
                    window,
                    max_plaintexts,
                    p_max_traces,
                ));
            }
        }
        gadget
    }

    fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        let regular_rows = self.to_regular(circuit);
        let digit_bound = Self::digit_radix(self.ctx.as_ref()) - BigUint::from(1u64);
        let mut decomposition = Vec::with_capacity(self.window.depth * self.ctx.num_limbs);
        for (q_idx, regular_level) in regular_rows.iter().enumerate() {
            for digit_idx in 0..self.ctx.num_limbs {
                let digit_values = regular_level
                    .iter()
                    .map(|regular| {
                        CarryArithPoly::new(
                            self.ctx.carry_arith_ctx.clone(),
                            vec![regular.limbs[digit_idx]],
                        )
                    })
                    .collect::<Vec<_>>();
                decomposition.push(Self::sparse_regular_level_poly_with_metadata(
                    self.ctx.clone(),
                    circuit,
                    self.window,
                    q_idx,
                    digit_values,
                    digit_bound.clone(),
                    digit_bound.clone(),
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
        let level = &regular_rows[q_idx];
        let mut digits = (0..self.ctx.num_limbs)
            .map(|digit_idx| {
                let coefficient_digits =
                    level.iter().map(|regular| regular.limbs[digit_idx]).collect::<Vec<_>>();
                Self::coefficient_embedding(circuit, &coefficient_digits)
            })
            .collect::<Vec<_>>();
        let last = digits.pop().expect("Montgomery decomposition requires at least one limb");
        (digits, last)
    }

    fn conv_mul_right_decomposed_many(
        &self,
        _params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        assert_eq!(num_slots, self.num_coefficients(), "num_slots must match the coefficients");
        negacyclic_decomposed_products(&[self], left_rows, circuit)
    }

    /// Both rows of a Ring-GSW column product in one sum per row: the top and bottom halves of
    /// each row are accumulated together and reduced once.
    fn mul_rows_with_decomposed_rhs(
        _params: &P::Params,
        lhs_row0: &[Self],
        lhs_row1: &[Self],
        rhs_top: &Self,
        rhs_bottom: &Self,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Self; 2] {
        assert_eq!(num_slots, rhs_top.num_coefficients(), "num_slots must match the coefficients");
        let mut rows =
            negacyclic_decomposed_products(&[rhs_top, rhs_bottom], &[lhs_row0, lhs_row1], circuit)
                .into_iter();
        let row0 = rows.next().expect("the product has a first row");
        let row1 = rows.next().expect("the product has a second row");
        [row0, row1]
    }
}

/// The inputs of a [`MontgomeryPoly`] of the given coefficients over the full CRT basis.
pub fn encode_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    coefficients: &[BigUint],
) -> Vec<P> {
    encode_montgomery_poly_with_window(limb_bit_size, params, coefficients, None, 0)
}

/// The inputs of a [`MontgomeryPoly`] of the given coefficients over `enable_levels` q-levels
/// starting at `level_offset`, ordered by level, then coefficient, then limb.
pub fn encode_montgomery_poly_with_window<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    coefficients: &[BigUint],
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
            coefficients
                .iter()
                .flat_map(|coefficient| {
                    let montgomery_value = ((coefficient % &modulus) * (&r % &modulus)) % &modulus;
                    encode_carry_arith_poly(limb_bit_size, num_limbs, params, &montgomery_value)
                })
                .collect::<Vec<_>>()
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
        circuit_gadgets::arith::{DecomposeArithmeticGadget, ModularArithmeticGadget},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        test_utils::execute_circuit,
        utils::gen_biguint_for_modulus,
    };
    use mxx_backends::poly::{
        Poly as ConcretePoly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_traits::Zero;

    const LIMB_BIT_SIZE: usize = 3;

    fn test_parameters() -> DCRTPolyParams {
        DCRTPolyParams::new(4, 3, 12, 6, None, None)
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

    fn active_modulus(parameters: &DCRTPolyParams, window: CrtWindow) -> BigUint {
        let (q_moduli, _, _) = parameters.to_crt();
        let window = CrtWindow::new(window.offset, window.depth, q_moduli.len());
        q_moduli[window.offset..window.end()]
            .par_iter()
            .copied()
            .map(BigUint::from)
            .reduce(BigUint::one, |left, right| left * right)
    }

    fn random_value(modulus: &BigUint) -> BigUint {
        let mut rng = rand::rng();
        gen_biguint_for_modulus(&mut rng, modulus)
    }

    fn build_input(
        context: Arc<MontgomeryPolyContext<DCRTPoly>>,
        window: CrtWindow,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> MontgomeryPoly<DCRTPoly> {
        <MontgomeryPoly<DCRTPoly> as ModularArithmeticGadget<DCRTPoly>>::input(
            context, 1, window, circuit,
        )
    }

    fn execute_constant_output(
        name: &str,
        parameters: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: Vec<DCRTPoly>,
    ) -> BigUint {
        let outputs = execute_polys(name, parameters, circuit, inputs);
        assert_eq!(outputs.len(), 1);
        outputs[0].coeffs_biguints()[0].clone()
    }

    fn test_roundtrip_case(limb_bit_size: usize, window: CrtWindow, value: BigUint) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context =
            Arc::new(MontgomeryPolyContext::setup(&mut circuit, &parameters, limb_bit_size, false));
        let input = build_input(context, window, &mut circuit);
        let output = input.finalize(&mut circuit);
        circuit.output([output]);
        let inputs = encode_montgomery_poly_with_window(
            limb_bit_size,
            &parameters,
            std::slice::from_ref(&value),
            Some(window.depth),
            window.offset,
        );
        let actual =
            execute_constant_output("montgomery-roundtrip-runtime", &parameters, &circuit, inputs);
        assert_eq!(actual % &modulus, value % modulus);
    }

    #[derive(Clone, Copy)]
    enum BinaryOperation {
        Add,
        Sub,
        Mul,
    }

    fn test_binary_case(
        operation: BinaryOperation,
        window: CrtWindow,
        left_value: BigUint,
        right_value: BigUint,
    ) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context =
            Arc::new(MontgomeryPolyContext::setup(&mut circuit, &parameters, LIMB_BIT_SIZE, false));
        let left = build_input(context.clone(), window, &mut circuit);
        let right = build_input(context, window, &mut circuit);
        let result = match operation {
            BinaryOperation::Add => left.add(&right, &mut circuit),
            BinaryOperation::Sub => left.sub(&right, &mut circuit),
            BinaryOperation::Mul => left.mul(&right, &mut circuit),
        };
        let output = result.finalize(&mut circuit);
        circuit.output([output]);
        let mut inputs = encode_montgomery_poly_with_window(
            LIMB_BIT_SIZE,
            &parameters,
            std::slice::from_ref(&left_value),
            Some(window.depth),
            window.offset,
        );
        inputs.extend(encode_montgomery_poly_with_window(
            LIMB_BIT_SIZE,
            &parameters,
            std::slice::from_ref(&right_value),
            Some(window.depth),
            window.offset,
        ));
        let actual =
            execute_constant_output("montgomery-binary-runtime", &parameters, &circuit, inputs);
        let expected = match operation {
            BinaryOperation::Add => left_value + right_value,
            BinaryOperation::Sub => left_value + &modulus - right_value,
            BinaryOperation::Mul => left_value * right_value,
        };
        assert_eq!(actual % &modulus, expected % modulus);
    }

    fn run_roundtrip_cases(limb_bit_size: usize, window: CrtWindow) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        [BigUint::zero(), &modulus - BigUint::from(1u8), random_value(&modulus)]
            .into_par_iter()
            .for_each(|value| {
                test_roundtrip_case(limb_bit_size, window, value);
            });
    }

    fn run_binary_cases(operation: BinaryOperation, window: CrtWindow) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        let cases = match operation {
            BinaryOperation::Add => vec![
                (BigUint::zero(), &modulus - BigUint::from(1u8)),
                (&modulus - BigUint::from(1u8), &modulus - BigUint::from(1u8)),
                (random_value(&modulus), random_value(&modulus)),
            ],
            BinaryOperation::Sub => vec![
                (BigUint::zero(), &modulus - BigUint::from(1u8)),
                (&modulus - BigUint::from(1u8), BigUint::zero()),
                (random_value(&modulus), random_value(&modulus)),
            ],
            BinaryOperation::Mul => vec![
                (BigUint::zero(), &modulus - BigUint::from(1u8)),
                (&modulus - BigUint::from(1u8), &modulus - BigUint::from(1u8)),
                (random_value(&modulus), random_value(&modulus)),
            ],
        };
        cases.into_par_iter().for_each(|(left, right)| {
            test_binary_case(operation, window, left, right);
        });
    }

    fn test_decomposition_case(window: CrtWindow) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        let value = random_value(&modulus);
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context =
            Arc::new(MontgomeryPolyContext::setup(&mut circuit, &parameters, LIMB_BIT_SIZE, false));
        let input = build_input(context.clone(), window, &mut circuit);
        let gadget = MontgomeryPoly::gadget_vector(context.clone(), 1, window, &mut circuit);
        let decomposition = input.gadget_decompose(&mut circuit);
        assert_eq!(gadget.len(), decomposition.len());
        let mut terms = gadget.iter().zip(&decomposition);
        let (first_gadget, first_digit) =
            terms.next().expect("Montgomery gadget decomposition is non-empty");
        let mut recomposed = first_gadget.mul(first_digit, &mut circuit);
        for (gadget, digit) in terms {
            let product = gadget.mul(digit, &mut circuit);
            recomposed = recomposed.add(&product, &mut circuit);
        }
        let output = recomposed.finalize(&mut circuit);
        circuit.output([output]);
        let inputs = encode_montgomery_poly_with_window(
            LIMB_BIT_SIZE,
            &parameters,
            std::slice::from_ref(&value),
            Some(window.depth),
            window.offset,
        );
        let actual = execute_constant_output(
            "montgomery-decomposition-runtime",
            &parameters,
            &circuit,
            inputs,
        );
        assert_eq!(actual % &modulus, value.clone() % &modulus);

        let target = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_biguint_to_constant(&parameters, value)],
        );
        let native_gadget = MontgomeryPoly::<DCRTPoly>::gadget_matrix::<DCRTPolyMatrix>(
            &parameters,
            &context,
            window,
        );
        let native_decomposition = MontgomeryPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
            &parameters,
            &context,
            &target,
            window,
        );
        assert_eq!(native_gadget.col_size(), native_decomposition.row_size());
    }

    #[test]
    fn roundtrip_reconstructs_full_depth_values_at_runtime() {
        run_roundtrip_cases(LIMB_BIT_SIZE, CrtWindow::full(3));
    }

    #[test]
    fn roundtrip_reconstructs_partial_window_values_at_runtime() {
        run_roundtrip_cases(LIMB_BIT_SIZE, CrtWindow::new(1, 2, 3));
    }

    #[test]
    fn one_bit_limbs_roundtrip_at_runtime() {
        let parameters = test_parameters();
        let window = CrtWindow::full(3);
        let modulus = active_modulus(&parameters, window);
        test_roundtrip_case(1, window, random_value(&modulus));
    }

    #[test]
    fn addition_reconstructs_full_depth_values_at_runtime() {
        run_binary_cases(BinaryOperation::Add, CrtWindow::full(3));
    }

    #[test]
    fn addition_reconstructs_partial_window_values_at_runtime() {
        run_binary_cases(BinaryOperation::Add, CrtWindow::new(1, 2, 3));
    }

    #[test]
    fn subtraction_reconstructs_full_depth_values_at_runtime() {
        run_binary_cases(BinaryOperation::Sub, CrtWindow::full(3));
    }

    #[test]
    fn subtraction_reconstructs_partial_window_values_at_runtime() {
        run_binary_cases(BinaryOperation::Sub, CrtWindow::new(1, 2, 3));
    }

    #[test]
    fn multiplication_reconstructs_full_depth_values_at_runtime() {
        run_binary_cases(BinaryOperation::Mul, CrtWindow::full(3));
    }

    #[test]
    fn multiplication_reconstructs_partial_window_values_at_runtime() {
        run_binary_cases(BinaryOperation::Mul, CrtWindow::new(1, 2, 3));
    }

    #[test]
    #[ignore = "full-depth Montgomery gadget decomposition runtime coverage takes several minutes"]
    fn gadget_decomposition_recomposes_full_depth_value_at_runtime() {
        test_decomposition_case(CrtWindow::full(3));
    }

    #[test]
    #[ignore = "partial-window Montgomery gadget decomposition runtime coverage takes about two minutes"]
    fn gadget_decomposition_recomposes_partial_window_value_at_runtime() {
        test_decomposition_case(CrtWindow::new(1, 2, 3));
    }
}
