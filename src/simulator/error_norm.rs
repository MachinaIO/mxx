use crate::{
    circuit::{Evaluable, PolyCircuit, gate::GateId},
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
    utils::bigdecimal_bits_ceil,
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};
use tracing::{debug, info};

impl PolyCircuit<DCRTPoly> {
    pub fn simulate_max_error_norm<P: PltEvaluator<ErrorNorm>>(
        &self,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
        input_size: usize,
        e_init_norm: &BigDecimal,
        plt_evaluator: Option<&P>,
    ) -> Vec<ErrorNorm>
    where
        ErrorNorm: Evaluable<P = DCRTPoly>,
    {
        let one_error = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        let input_error = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        info!("e_init_norm bits {}", bigdecimal_bits_ceil(e_init_norm));
        self.eval(&(), &one_error, &vec![input_error; input_size], plt_evaluator)
    }
}

// Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorNorm {
    pub plaintext_norm: PolyNorm,
    pub matrix_norm: PolyMatrixNorm,
}

impl ErrorNorm {
    pub fn new(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        debug_assert_eq!(plaintext_norm.ctx, matrix_norm.clone_ctx());
        Self { plaintext_norm, matrix_norm }
    }

    #[inline]
    pub fn ctx(&self) -> &SimulatorContext {
        &self.plaintext_norm.ctx
    }
    #[inline]
    pub fn clone_ctx(&self) -> Arc<SimulatorContext> {
        self.plaintext_norm.ctx.clone()
    }
}

impl_binop_with_refs!(ErrorNorm => Add::add(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(ErrorNorm => Sub::sub(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

impl_binop_with_refs!(ErrorNorm => Mul::mul(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm * PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g) + rhs.matrix_norm.clone() * &self.plaintext_norm
    }
});

impl Evaluable for ErrorNorm {
    type Params = ();
    type P = DCRTPoly;

    fn rotate(&self, _: &Self::Params, _: i32) -> Self {
        self.clone()
    }

    fn small_scalar_mul(&self, _: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_max);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_bd);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLWEEvaluator {
    pub e_b_times_preimage: PolyMatrixNorm,
    pub preimage_lower: PolyMatrixNorm,
}

impl NormPltLWEEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, e_b_sigma: &BigDecimal) -> Self {
        let c_0 = BigDecimal::from_f64(1.8).unwrap();
        let c_1 = BigDecimal::from_f64(4.7).unwrap();
        let sigma = BigDecimal::from_f64(4.578).unwrap();
        let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
        let m_g_sqrt = BigDecimal::from(ctx.m_g as u64).sqrt().expect("sqrt(m_g) failed");
        let term = ctx.ring_dim_sqrt.clone() * m_g_sqrt +
            two_sqrt * ctx.ring_dim_sqrt.clone() +
            c_1.clone();
        let norm: BigDecimal = c_0 * 6 * sigma.clone() * ((ctx.base.clone() + 1) * sigma) * term;
        let norm_bits = bigdecimal_bits_ceil(&norm);
        info!("{}", format!("preimage norm bits {}", norm_bits));
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * 6, None);
        let e_b_times_preimage =
            &e_b_init * &PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, norm.clone(), None);
        let preimage_lower = PolyMatrixNorm::new(ctx.clone(), ctx.m_g, ctx.m_g, norm.clone(), None);
        Self { e_b_times_preimage, preimage_lower }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltLWEEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let matrix_norm = &self.e_b_times_preimage + (&input.matrix_norm * &self.preimage_lower);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltGGH15Evaluator {
    pub const_term: PolyMatrixNorm,
    pub s_times_errs: PolyMatrixNorm,
    pub e_one_multiplier: PolyMatrixNorm,
    pub e_input_multiplier: PolyMatrixNorm,
}

impl NormPltGGH15Evaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b_sigma: &BigDecimal,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let preimage_norm = compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base);
        // : BigDecimal =
        //     c_0 * 6 * sigma.clone() * ((ctx.base.clone() + 1) * sigma) * term;
        // let preimage_norm_bits = bigdecimal_bits_ceil(&preimage_norm);
        // info!("{}", format!("preimage norm bits {}", preimage_norm_bits));
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * 6, None);

        let k_to_ggh =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, preimage_norm.clone(), None);
        let k_g =
            PolyMatrixNorm::new(ctx.clone(), 2 * ctx.m_b, 3 * ctx.m_b, preimage_norm.clone(), None);
        let k_lut =
            PolyMatrixNorm::new(ctx.clone(), 3 * ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        let s_times_errs = s_vec *
            PolyMatrixNorm::new(ctx.clone(), ctx.secret_size, 3 * ctx.m_b, e_mat_sigma * 6, None) *
            &k_lut;
        debug!("s_times_errs norm bits {}", bigdecimal_bits_ceil(&s_times_errs.poly_norm.norm));
        let const_term = e_b_init * k_to_ggh * &k_g * &k_lut + &s_times_errs;
        let const_term_bits = bigdecimal_bits_ceil(&const_term.poly_norm.norm);
        debug!("const_term norm bits {}", const_term_bits);
        info!("{}", format!("GGH15 PLT const term norm bits {}", const_term_bits));
        let decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), 2 * ctx.m_b);
        let e_one_multiplier = &decomposed * &k_g * &k_lut;
        let e_one_multiplier_bits = bigdecimal_bits_ceil(&e_one_multiplier.poly_norm.norm);
        debug!("e_one_multiplier norm bits {}", e_one_multiplier_bits);
        info!("{}", format!("GGH15 PLT e_one multiplier norm bits {}", e_one_multiplier_bits));
        let e_input_multiplier = decomposed * &k_g * &k_lut;
        Self { const_term, s_times_errs, e_one_multiplier, e_input_multiplier }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltGGH15Evaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        one: &ErrorNorm,
        input: &ErrorNorm,
        gate_id: GateId,
        lut_id: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        let max_input_norm = PolyNorm::new(
            input.clone_ctx(),
            BigDecimal::from(num_bigint::BigInt::from(plt.len() as u64 - 1u64)),
        );
        let input_bits = bigdecimal_bits_ceil(&input.matrix_norm.poly_norm.norm);
        debug!(
            "{}",
            format!(
                "GGH15 PLT input matrix bits {} (gate {}, lut {})",
                input_bits, gate_id, lut_id
            )
        );
        let matrix_norm = &self.const_term +
            &self.s_times_errs * &max_input_norm +
            &one.matrix_norm * &self.e_one_multiplier +
            &input.matrix_norm * &self.e_input_multiplier;
        debug!(
            "{}",
            format!(
                "GGH15 PLT output matrix bits {}",
                bigdecimal_bits_ceil(&matrix_norm.poly_norm.norm)
            )
        );
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

pub fn compute_preimage_norm(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(4.578).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let term = ring_dim_sqrt.clone() * m_g_sqrt + two_sqrt * ring_dim_sqrt + c_1;
    let preimage_norm = c_0 * 6 * sigma.clone() * ((base + 1) * sigma) * term;
    let preimage_norm_bits = bigdecimal_bits_ceil(&preimage_norm);
    info!("{}", format!("preimage norm bits {}", preimage_norm_bits));
    preimage_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::PublicLut,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        simulator::SimulatorContext,
    };
    use bigdecimal::BigDecimal;

    fn make_ctx() -> Arc<SimulatorContext> {
        // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
        Arc::new(SimulatorContext::new(
            BigDecimal::from(1024u64), // ring_dim_sqrt
            BigDecimal::from(32u64),   // base
            2,
            28, // log_base_q
        ))
    }

    const E_B_SIGMA: f64 = 4.0;
    const E_INIT_NORM: u32 = 1 << 14;

    #[test]
    fn test_wire_norm_addition() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.add_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);

        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
        );
        assert_eq!(out.len(), 1);
        // Build expected from input wires and add them
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire + &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_subtraction() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.sub_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
        );
        assert_eq!(out.len(), 1);
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_multiplication() {
        // ctx: secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=28
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
        );
        assert_eq!(out.len(), 1);

        // Build expected = in_wire * in_wire
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire * &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_lwe_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new_from_usize_range(
            &params,
            2,
            |params, idx| match idx {
                0 => (0usize, DCRTPoly::from_usize_to_constant(params, 5)),
                1 => (1usize, DCRTPoly::from_usize_to_constant(params, 7)),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(E_B_SIGMA).unwrap());
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_ggh15_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new_from_usize_range(
            &params,
            2,
            |params, idx| match idx {
                0 => (0usize, DCRTPoly::from_usize_to_constant(params, 5)),
                1 => (1usize, DCRTPoly::from_usize_to_constant(params, 7)),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }
}
