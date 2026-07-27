use crate::{
    circuit::Evaluable,
    impl_binop_with_refs,
    poly::dcrt::poly::DCRTPoly,
    simulator::{
        SimulatorContext, dependency_set::DependencySet, poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use num_traits::Zero;
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};
use tracing::debug;

pub use super::eval_error::{
    AffinePltEvaluator, AffineSlotTransferEvaluator, NormBggPolyEncodingSTEvaluator,
    NormNaiveBggEncodingVecSTEvaluator, NormPltCommitEvaluator, NormPltGGH15Evaluator,
    NormPltLWEEvaluator, compute_preimage_norm, compute_preimage_sigma,
};

// Note: h_norm and plaintext_norm computed here can be larger than the modulus .
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone)]
pub struct ErrorNorm {
    pub plaintext_norm: PolyNorm,
    pub matrix_norm: PolyMatrixNorm,
    pub pubkey_deps: DependencySet,
    pub is_pubkey_random: bool,
}

impl PartialEq for ErrorNorm {
    fn eq(&self, other: &Self) -> bool {
        self.plaintext_norm == other.plaintext_norm &&
            self.matrix_norm == other.matrix_norm &&
            self.pubkey_deps == other.pubkey_deps &&
            self.is_pubkey_random == other.is_pubkey_random
    }
}

impl Eq for ErrorNorm {}

impl ErrorNorm {
    pub fn new(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        Self::deterministic_pubkey(plaintext_norm, matrix_norm)
    }

    pub fn fresh_input(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        let ctx = plaintext_norm.ctx.clone();
        Self::from_parts(
            plaintext_norm,
            matrix_norm,
            DependencySet::singleton(ctx.fresh_source_id()),
            true,
        )
    }

    pub fn from_parts(
        plaintext_norm: PolyNorm,
        matrix_norm: PolyMatrixNorm,
        pubkey_deps: DependencySet,
        is_pubkey_random: bool,
    ) -> Self {
        debug_assert_eq!(plaintext_norm.ctx, matrix_norm.clone_ctx());
        Self {
            plaintext_norm: plaintext_norm.into_constant_poly(),
            matrix_norm,
            pubkey_deps,
            is_pubkey_random,
        }
    }

    pub fn deterministic_pubkey(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        Self::from_parts(plaintext_norm, matrix_norm, DependencySet::empty(), false)
    }

    pub fn fresh_lut_output(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        let ctx = plaintext_norm.ctx.clone();
        Self::from_parts(
            plaintext_norm,
            matrix_norm,
            DependencySet::singleton(ctx.fresh_source_id()),
            true,
        )
    }

    pub fn rhs_pubkey_gadget_norm(&self, gadget_poly: PolyNorm) -> PolyMatrixNorm {
        let ctx = gadget_poly.ctx.clone();
        debug!(
            rule = "RHS_PUBKEY_GADGET",
            rhs_is_pubkey_random = self.is_pubkey_random,
            rhs_pubkey_deps = ?self.pubkey_deps,
            "simulator BGG RHS public-key gadget norm"
        );
        PolyMatrixNorm::from_parts(
            ctx.m_g,
            ctx.m_g,
            gadget_poly,
            None,
            self.pubkey_deps.clone(),
            self.is_pubkey_random,
        )
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
    ErrorNorm::from_parts(
        &self.plaintext_norm + &rhs.plaintext_norm,
        &self.matrix_norm + &rhs.matrix_norm,
        self.pubkey_deps.union(&rhs.pubkey_deps),
        false,
    )
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(ErrorNorm => Sub::sub(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm::from_parts(
        &self.plaintext_norm + &rhs.plaintext_norm,
        &self.matrix_norm + &rhs.matrix_norm,
        self.pubkey_deps.union(&rhs.pubkey_deps),
        false,
    )
});

impl_binop_with_refs!(ErrorNorm => Mul::mul(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    let rhs_gadget = rhs.rhs_pubkey_gadget_norm(
        PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g).poly_norm,
    );
    ErrorNorm::from_parts(
        &self.plaintext_norm * &rhs.plaintext_norm,
        &self.matrix_norm * &rhs_gadget + rhs.matrix_norm.clone() * &self.plaintext_norm,
        self.pubkey_deps.union(&rhs.pubkey_deps),
        false,
    )
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
        let scalar_poly = PolyNorm::constant(self.clone_ctx(), scalar_max.clone());
        let (pubkey_deps, is_pubkey_random) = if scalar_max.is_zero() {
            (DependencySet::empty(), false)
        } else {
            (self.pubkey_deps.clone(), self.is_pubkey_random)
        };
        ErrorNorm::from_parts(
            self.plaintext_norm.clone() * &scalar_poly,
            self.matrix_norm.clone() * &scalar_poly,
            pubkey_deps,
            is_pubkey_random,
        )
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::constant(self.clone_ctx(), scalar_bd.clone());
        let (pubkey_deps, is_pubkey_random) = if scalar_bd.is_zero() {
            (DependencySet::empty(), false)
        } else {
            (self.pubkey_deps.clone(), false)
        };
        ErrorNorm::from_parts(
            self.plaintext_norm.clone() * &scalar_poly,
            self.matrix_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g),
            pubkey_deps,
            is_pubkey_random,
        )
    }
}
