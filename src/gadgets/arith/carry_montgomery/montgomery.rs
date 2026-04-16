use super::carry_arith::{CarryArithPoly, CarryArithPolyContext, encode_carry_arith_poly};
use crate::{
    circuit::{BatchedWire, PolyCircuit, gate::GateId},
    gadgets::arith::{ModularArithmeticContext, ModularArithmeticGadget},
    poly::{Poly, PolyParams},
    utils::mod_inverse_biguints,
};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryPolyContext<P: Poly> {
    pub params: P::Params,
    pub carry_arith_ctx: Arc<CarryArithPolyContext<P>>,
    pub modulus: Arc<BigUint>,
    pub modulus_u64: u64,
    pub num_limbs: usize,
    pub moduli_poly: CarryArithPoly<P>,
    pub r2_poly: CarryArithPoly<P>,
    pub moduli_prime_poly: CarryArithPoly<P>,
}

impl<P: Poly + 'static> MontgomeryPolyContext<P> {
    pub fn setup(
        circuit: &mut PolyCircuit<P>,
        params: &P::Params,
        limb_bit_size: usize,
        dummy_scalar: bool,
    ) -> Self {
        let modulus: Arc<BigUint> = params.modulus().into();
        let modulus_u64 =
            modulus.as_ref().to_u64().expect("Montgomery modulus must fit in u64 for gadget APIs");
        let num_limbs = params.modulus_bits().div_ceil(limb_bit_size);
        let carry_arith_ctx =
            Arc::new(CarryArithPolyContext::setup(circuit, params, limb_bit_size, dummy_scalar));

        let moduli_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            modulus.as_ref(),
        );

        let r_bits = limb_bit_size * num_limbs;
        let r = BigUint::one() << r_bits;
        let r2_value = (&r * &r) % modulus.as_ref();
        let r2_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            &r2_value,
        );
        let moduli_prime_value = Self::calculate_modulus_prime(modulus.as_ref(), &r);
        let moduli_prime_poly = Self::constant_carry_arith_poly(
            circuit,
            params,
            carry_arith_ctx.clone(),
            num_limbs,
            &moduli_prime_value,
        );

        Self {
            params: params.clone(),
            carry_arith_ctx,
            modulus,
            modulus_u64,
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
}

#[derive(Debug, Clone)]
pub struct MontgomeryPoly<P: Poly> {
    pub ctx: Arc<MontgomeryPolyContext<P>>,
    pub value: CarryArithPoly<P>,
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
        let levels = enable_levels.unwrap_or(1);
        assert_eq!(levels, 1, "MontgomeryPoly supports exactly one active q level");
        assert_eq!(level_offset, 0, "MontgomeryPoly only supports level_offset == 0");
        assert_eq!(ctx.modulus_u64 > 0, true, "Montgomery modulus must be positive");
    }

    fn default_max_plaintexts(ctx: &MontgomeryPolyContext<P>) -> Vec<BigUint> {
        vec![ctx.modulus.as_ref() - BigUint::from(1u64)]
    }

    fn default_p_max_traces(ctx: &MontgomeryPolyContext<P>) -> Vec<BigUint> {
        vec![ctx.modulus.as_ref() - BigUint::from(1u64)]
    }

    fn with_metadata(
        ctx: Arc<MontgomeryPolyContext<P>>,
        value: CarryArithPoly<P>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let level_offset = level_offset.unwrap_or(0);
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset);
        assert_eq!(max_plaintexts.len(), 1, "MontgomeryPoly metadata must contain one level");
        assert_eq!(p_max_traces.len(), 1, "MontgomeryPoly metadata must contain one trace level");
        Self { ctx, value, level_offset, enable_levels, max_plaintexts, p_max_traces }
    }

    pub fn new(ctx: Arc<MontgomeryPolyContext<P>>, value: CarryArithPoly<P>) -> Self {
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref());
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref());
        Self::with_metadata(ctx, value, Some(1), Some(0), max_plaintexts, p_max_traces)
    }

    pub fn input(ctx: Arc<MontgomeryPolyContext<P>>, circuit: &mut PolyCircuit<P>) -> Self {
        let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
        let value = CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size);
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref());
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref());
        Self::with_metadata(ctx, value, Some(1), Some(0), max_plaintexts, p_max_traces)
    }

    pub fn from_regular(
        circuit: &mut PolyCircuit<P>,
        ctx: Arc<MontgomeryPolyContext<P>>,
        mut value: CarryArithPoly<P>,
    ) -> Self {
        if value.limbs.len() != ctx.num_limbs {
            let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
            value = value.extend_size(bit_size);
        }
        let r2_mul = value.mul(&ctx.r2_poly, circuit, None);
        let reduced = Self::montgomery_reduce(ctx.as_ref(), circuit, &r2_mul);
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref());
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref());
        Self::with_metadata(ctx, reduced, Some(1), Some(0), max_plaintexts, p_max_traces)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let sum_full = self.value.add(&other.value, circuit);
        let n_ext_bits = (self.ctx.num_limbs + 1) * self.ctx.carry_arith_ctx.limb_bit_size;
        let n_ext = self.ctx.moduli_poly.extend_size(n_ext_bits);
        let (is_less, diff) = sum_full.less_than(&n_ext, circuit);
        let reduced_full = sum_full.cmux(&diff, is_less, circuit);
        let reduced = reduced_full.mod_limbs(self.ctx.num_limbs);
        debug!("num gates {:?} at MontgomeryPoly::add", circuit.count_gates_by_type_vec());
        Self::new(self.ctx.clone(), reduced)
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let (is_less, raw_sub) = self.value.less_than(&other.value, circuit);
        let added = raw_sub.add(&self.ctx.moduli_poly, circuit).mod_limbs(self.ctx.num_limbs);
        let result = added.cmux(&raw_sub, is_less, circuit);
        debug!("num gates {:?} at MontgomeryPoly::sub", circuit.count_gates_by_type_vec());
        Self::new(self.ctx.clone(), result)
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        debug_assert_eq!(self.ctx, other.ctx);
        let product = self.value.mul(&other.value, circuit, None);
        let reduced = Self::montgomery_reduce(self.ctx.as_ref(), circuit, &product);
        debug!("num gates {:?} at MontgomeryPoly::mul", circuit.count_gates_by_type_vec());
        Self::new(self.ctx.clone(), reduced)
    }

    pub fn to_regular(&self, circuit: &mut PolyCircuit<P>) -> CarryArithPoly<P> {
        Self::montgomery_reduce(self.ctx.as_ref(), circuit, &self.value)
    }

    pub fn finalize(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.to_regular(circuit).finalize(circuit)
    }

    fn montgomery_reduce(
        ctx: &MontgomeryPolyContext<P>,
        circuit: &mut PolyCircuit<P>,
        t: &CarryArithPoly<P>,
    ) -> CarryArithPoly<P> {
        let r = ctx.num_limbs;
        let limb_bits = ctx.carry_arith_ctx.limb_bit_size;

        let t_low = t.mod_limbs(r);
        let m = t_low.mul(&ctx.moduli_prime_poly, circuit, Some(r * limb_bits));
        let m_times_n = m.mul(&ctx.moduli_poly, circuit, None);
        let u = t.add(&m_times_n, circuit);

        let u_shifted = u.left_shift(r);
        let n_ext_bits = (r + 1) * limb_bits;
        let n_ext = ctx.moduli_poly.extend_size(n_ext_bits);
        let (is_less, diff) = u_shifted.less_than(&n_ext, circuit);
        let reduced_full = u_shifted.cmux(&diff, is_less, circuit);
        reduced_full.mod_limbs(r)
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
        1
    }

    fn decomposition_len(&self) -> usize {
        1
    }

    fn q_level_row_width(&self) -> usize {
        self.num_limbs
    }

    fn full_reduce_output_metadata(
        &self,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
    ) -> (Vec<BigUint>, Vec<BigUint>) {
        MontgomeryPoly::<P>::validate_window(self, enable_levels, level_offset.unwrap_or(0));
        (
            MontgomeryPoly::<P>::default_max_plaintexts(self),
            MontgomeryPoly::<P>::default_p_max_traces(self),
        )
    }

    fn reduced_p_max_trace(&self) -> BigUint {
        self.modulus.as_ref() - BigUint::from(1u64)
    }

    fn randomizer_decomposition_bound(&self) -> u64 {
        self.modulus_u64.saturating_sub(1)
    }

    fn decomposition_term_bound(&self, _term_idx: usize) -> BigUint {
        BigUint::from(1u64)
    }

    fn full_reduce_level_plaintext_bound(&self, _q_idx: usize) -> BigUint {
        self.modulus.as_ref() - BigUint::from(1u64)
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
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset.unwrap_or(0));
        let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
        let value = CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size);
        let max_plaintexts = Self::default_max_plaintexts(ctx.as_ref());
        let p_max_traces = Self::default_p_max_traces(ctx.as_ref());
        Self::with_metadata(ctx, value, Some(1), Some(0), max_plaintexts, p_max_traces)
    }

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset.unwrap_or(0));
        let bit_size = ctx.num_limbs * ctx.carry_arith_ctx.limb_bit_size;
        let value = CarryArithPoly::input(ctx.carry_arith_ctx.clone(), circuit, bit_size);
        Self::with_metadata(ctx, value, Some(1), Some(0), max_plaintexts, p_max_traces)
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        vec![self.ctx.modulus_u64]
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        self.value.limbs.iter().copied().map(BatchedWire::single).collect()
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let value = CarryArithPoly::new(template.ctx.carry_arith_ctx.clone(), outputs.to_vec());
        Self::with_metadata(
            template.ctx.clone(),
            value,
            template.enable_levels,
            Some(template.level_offset),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn q_level_row_batch(&self, _q_idx: usize) -> BatchedWire {
        BatchedWire::from_batches(self.value.limbs.iter().copied())
    }

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        _active_levels: usize,
        enable_levels: Option<usize>,
        level_offset: usize,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        _circuit: &mut PolyCircuit<P>,
    ) -> Self {
        assert_eq!(target_q_idx, 0, "MontgomeryPoly has exactly one q-level");
        let value =
            CarryArithPoly::new(ctx.carry_arith_ctx.clone(), target_row.gate_ids().collect());
        Self::with_metadata(
            ctx,
            value,
            enable_levels,
            Some(level_offset),
            vec![max_plaintext],
            vec![p_max_trace],
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
        let limbs = self
            .value
            .limbs
            .iter()
            .map(|&gate_id| {
                circuit.slot_transfer_gate(gate_id, &lowered_src_slots).as_single_wire()
            })
            .collect::<Vec<_>>();
        let value = CarryArithPoly::new(self.ctx.carry_arith_ctx.clone(), limbs);
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
            1,
            "MontgomeryPoly const_mul requires exactly one active q-level scalar"
        );
        let constant = MontgomeryPolyContext::constant_carry_arith_poly(
            circuit,
            &self.ctx.params,
            self.ctx.carry_arith_ctx.clone(),
            self.ctx.num_limbs,
            &BigUint::from(tower_constants[0]),
        );
        let mont_constant = MontgomeryPoly::from_regular(circuit, self.ctx.clone(), constant);
        self.mul(&mont_constant, circuit)
    }

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.finalize(circuit)
    }
}

/* impl<P: Poly + 'static> DecomposeArithmeticGadget<P> for MontgomeryPoly<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        _ctx: &Self::Context,
        _enable_levels: Option<usize>,
        _level_offset: Option<usize>,
    ) -> M {
        M::from_poly_vec_row(params, vec![P::const_one(params)])
    }

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        _params: &P::Params,
        _ctx: &Self::Context,
        target: &M,
        _enable_levels: Option<usize>,
        _level_offset: Option<usize>,
    ) -> M {
        target.clone()
    }

    fn gadget_decomposition_norm_bound(
        _ctx: &Self::Context,
        _enable_levels: Option<usize>,
        _level_offset: Option<usize>,
    ) -> BigUint {
        BigUint::from(1u64)
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        enable_levels: Option<usize>,
        level_offset: Option<usize>,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        Self::validate_window(ctx.as_ref(), enable_levels, level_offset.unwrap_or(0));
        let one = MontgomeryPolyContext::constant_carry_arith_poly(
            circuit,
            &ctx.params,
            ctx.carry_arith_ctx.clone(),
            ctx.num_limbs,
            &BigUint::from(1u64),
        );
        vec![MontgomeryPoly::from_regular(circuit, ctx, one)]
    }

    fn conv_mul_right_decomposed_many(
        &self,
        _params: &P::Params,
        left_rows: &[&[Self]],
        _num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        left_rows
            .iter()
            .map(|row| {
                assert_eq!(row.len(), 1, "MontgomeryPoly decomposed rows must have length 1");
                row[0].mul(self, circuit)
            })
            .collect()
    }

    fn gadget_decompose(&self, _circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        vec![self.clone()]
    }

    fn decomposition_terms_for_level(
        &self,
        q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        assert_eq!(q_idx, 0, "MontgomeryPoly has exactly one q-level");
        (Vec::new(), self.reconstruct(circuit))
    }
} */

pub fn encode_montgomery_poly<P: Poly>(
    limb_bit_size: usize,
    params: &P::Params,
    input: &BigUint,
) -> Vec<P> {
    let modulus: Arc<BigUint> = params.modulus().into();
    let num_limbs = params.modulus_bits().div_ceil(limb_bit_size);
    let r = BigUint::one() << (limb_bit_size * num_limbs);
    let montgomery_value =
        ((input % modulus.as_ref()) * (&r % modulus.as_ref())) % modulus.as_ref();
    encode_carry_arith_poly(limb_bit_size, num_limbs, params, &montgomery_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lookup::poly::PolyPltEvaluator,
        poly::dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };

    const LIMB_BIT_SIZE: usize = 3;

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

    fn encode_regular_poly(
        ctx: &MontgomeryPolyContext<DCRTPoly>,
        params: &DCRTPolyParams,
        value: &BigUint,
    ) -> Vec<DCRTPoly> {
        encode_carry_arith_poly(ctx.carry_arith_ctx.limb_bit_size, ctx.num_limbs, params, value)
    }

    fn max_value_for_modulus(modulus: &BigUint) -> BigUint {
        modulus - BigUint::from(1u32)
    }

    fn test_montgomery_roundtrip_case(value: BigUint) {
        test_montgomery_roundtrip_case_with_params(DCRTPolyParams::default(), LIMB_BIT_SIZE, value);
    }

    fn test_montgomery_add_case(lhs_value: BigUint, rhs_value: BigUint) {
        test_montgomery_add_case_with_params(
            DCRTPolyParams::default(),
            LIMB_BIT_SIZE,
            lhs_value,
            rhs_value,
        );
    }

    fn test_montgomery_sub_case(lhs_value: BigUint, rhs_value: BigUint) {
        test_montgomery_sub_case_with_params(
            DCRTPolyParams::default(),
            LIMB_BIT_SIZE,
            lhs_value,
            rhs_value,
        );
    }

    fn test_montgomery_mul_case(lhs_value: BigUint, rhs_value: BigUint) {
        test_montgomery_mul_case_with_params(
            DCRTPolyParams::default(),
            LIMB_BIT_SIZE,
            lhs_value,
            rhs_value,
        );
    }

    fn test_montgomery_roundtrip_case_with_params(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let input = CarryArithPoly::<DCRTPoly>::input(
            ctx.carry_arith_ctx.clone(),
            &mut circuit,
            ctx.num_limbs * limb_bit_size,
        );
        let mont = MontgomeryPoly::from_regular(&mut circuit, ctx.clone(), input);
        let regular = mont.to_regular(&mut circuit);
        circuit.output(regular.limbs.clone());

        let eval_inputs = encode_regular_poly(ctx.as_ref(), &params, &value);
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_regular_poly(ctx.as_ref(), &params, &value);
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_add_case_with_params(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        lhs_value: BigUint,
        rhs_value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.add(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(limb_bit_size, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(limb_bit_size, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_montgomery_poly(limb_bit_size, &params, &(lhs_value + rhs_value));
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_sub_case_with_params(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        lhs_value: BigUint,
        rhs_value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.sub(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(limb_bit_size, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(limb_bit_size, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let modulus = params.modulus();
        let expected_value = if lhs_value >= rhs_value {
            lhs_value - rhs_value
        } else {
            lhs_value + modulus.as_ref() - rhs_value
        };
        let expected = encode_montgomery_poly(limb_bit_size, &params, &expected_value);
        assert_eq!(eval_result, expected);
    }

    fn test_montgomery_mul_case_with_params(
        params: DCRTPolyParams,
        limb_bit_size: usize,
        lhs_value: BigUint,
        rhs_value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (params, ctx) = create_test_context_with_limb_size(&mut circuit, params, limb_bit_size);
        let lhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let rhs = MontgomeryPoly::input(ctx.clone(), &mut circuit);
        let result = lhs.mul(&rhs, &mut circuit);
        circuit.output(result.value.limbs.clone());

        let mut eval_inputs = encode_montgomery_poly(limb_bit_size, &params, &lhs_value);
        eval_inputs.extend(encode_montgomery_poly(limb_bit_size, &params, &rhs_value));
        let eval_result = eval_with_const_one(&circuit, &params, &eval_inputs);
        let expected = encode_montgomery_poly(limb_bit_size, &params, &(lhs_value * rhs_value));
        assert_eq!(eval_result, expected);
    }

    #[test]
    fn test_montgomery_roundtrip_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_roundtrip_case(value);
    }

    #[test]
    fn test_montgomery_roundtrip_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_roundtrip_case(min_value);
        test_montgomery_roundtrip_case(max_value);
    }

    #[test]
    fn test_montgomery_add_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_add_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_add_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_add_case(min_value.clone(), max_value.clone());
        test_montgomery_add_case(max_value.clone(), max_value);
    }

    #[test]
    fn test_montgomery_sub_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_sub_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_sub_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_sub_case(min_value.clone(), max_value.clone());
        test_montgomery_sub_case(max_value, min_value);
    }

    #[test]
    fn test_montgomery_mul_random() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_mul_case(lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_mul_min_max() {
        let params = DCRTPolyParams::default();
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_mul_case(min_value, max_value.clone());
        test_montgomery_mul_case(max_value.clone(), max_value);
    }

    #[test]
    fn test_montgomery_roundtrip_random_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_roundtrip_case_with_params(params, 1, value);
    }

    #[test]
    fn test_montgomery_roundtrip_min_max_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_roundtrip_case_with_params(params.clone(), 1, min_value);
        test_montgomery_roundtrip_case_with_params(params, 1, max_value);
    }

    #[test]
    fn test_montgomery_add_random_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_add_case_with_params(params, 1, lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_add_min_max_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_add_case_with_params(
            params.clone(),
            1,
            min_value.clone(),
            max_value.clone(),
        );
        test_montgomery_add_case_with_params(params, 1, max_value.clone(), max_value);
    }

    #[test]
    fn test_montgomery_sub_random_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_sub_case_with_params(params, 1, lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_sub_min_max_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_sub_case_with_params(
            params.clone(),
            1,
            min_value.clone(),
            max_value.clone(),
        );
        test_montgomery_sub_case_with_params(params, 1, max_value, min_value);
    }

    #[test]
    fn test_montgomery_mul_random_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let mut rng = rand::rng();
        let lhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        let rhs_value = crate::utils::gen_biguint_for_modulus(&mut rng, modulus.as_ref());
        test_montgomery_mul_case_with_params(params, 1, lhs_value, rhs_value);
    }

    #[test]
    fn test_montgomery_mul_min_max_limb1() {
        let params = DCRTPolyParams::new(4, 2, 15, 13);
        let modulus = params.modulus();
        let min_value = BigUint::from(0u32);
        let max_value = max_value_for_modulus(modulus.as_ref());
        test_montgomery_mul_case_with_params(params.clone(), 1, min_value, max_value.clone());
        test_montgomery_mul_case_with_params(params, 1, max_value.clone(), max_value);
    }
}
