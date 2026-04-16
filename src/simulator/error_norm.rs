use crate::{
    circuit::Evaluable,
    impl_binop_with_refs,
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

pub use super::eval_error::{
    AffinePltEvaluator, AffineSlotTransferEvaluator, NormBggPolyEncodingSTEvaluator,
    NormPltCommitEvaluator, NormPltGGH15Evaluator, NormPltLWEEvaluator, compute_preimage_norm,
};

// Note: h_norm and plaintext_norm computed here can be larger than the modulus .
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
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm,
    }
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(ErrorNorm => Sub::sub(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm,
    }
});

impl_binop_with_refs!(ErrorNorm => Mul::mul(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm *
            PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g) +
            rhs.matrix_norm.clone() * &self.plaintext_norm,
    }
});

impl Evaluable for ErrorNorm {
    type Params = ();
    type P = DCRTPoly;
    type Compact = ErrorNorm;

    fn to_compact(self) -> Self::Compact {
        self
    }

    fn from_compact(_: &Self::Params, compact: &Self::Compact) -> Self {
        compact.clone()
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
