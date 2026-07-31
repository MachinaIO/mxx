use super::{
    dependency_set::DependencySet,
    poly_norm::{PolyNorm, high_probability_envelope_from_sigma},
};
use crate::SimulatorContext;
use bigdecimal::BigDecimal;
use mxx_primitives::impl_binop_with_refs;
use num_traits::Zero;
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};
use tracing::debug;

#[derive(Debug, Clone)]
pub struct PolyMatrixNorm {
    pub nrow: usize,
    pub ncol: usize,
    pub ncol_sqrt: BigDecimal,
    pub poly_norm: PolyNorm,
    pub zero_rows: Option<usize>,
    pub deps: DependencySet,
    pub clt_ready: bool,
}

impl PartialEq for PolyMatrixNorm {
    fn eq(&self, other: &Self) -> bool {
        self.nrow == other.nrow &&
            self.ncol == other.ncol &&
            self.ncol_sqrt == other.ncol_sqrt &&
            self.poly_norm == other.poly_norm &&
            self.zero_rows == other.zero_rows &&
            self.deps == other.deps &&
            self.clt_ready == other.clt_ready
    }
}

impl Eq for PolyMatrixNorm {}

impl PolyMatrixNorm {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        norm: BigDecimal,
        zero_rows: Option<usize>,
    ) -> Self {
        Self::from_parts(
            nrow,
            ncol,
            PolyNorm::new(ctx, norm),
            zero_rows,
            DependencySet::empty(),
            false,
        )
    }

    pub fn fresh_preimage(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        preimage_sigma: BigDecimal,
        zero_rows: Option<usize>,
    ) -> Self {
        let deps = DependencySet::singleton(ctx.fresh_source_id());
        Self::from_parts(
            nrow,
            ncol,
            PolyNorm::new(ctx, high_probability_envelope_from_sigma(&preimage_sigma)),
            zero_rows,
            deps,
            true,
        )
    }

    pub fn fresh_random_with_norm(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        norm: BigDecimal,
        zero_rows: Option<usize>,
    ) -> Self {
        let deps = DependencySet::singleton(ctx.fresh_source_id());
        Self::from_parts(nrow, ncol, PolyNorm::new(ctx, norm), zero_rows, deps, true)
    }

    pub fn from_parts(
        nrow: usize,
        ncol: usize,
        poly_norm: PolyNorm,
        zero_rows: Option<usize>,
        deps: DependencySet,
        clt_ready: bool,
    ) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm,
            zero_rows,
            deps,
            clt_ready,
        }
    }

    pub fn sample_gauss(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        sigma: BigDecimal,
    ) -> Self {
        let deps = DependencySet::singleton(ctx.fresh_source_id());
        Self::from_parts(nrow, ncol, PolyNorm::sample_gauss(ctx, sigma), None, deps, true)
    }

    fn balanced_gadget_digit_norm(ctx: &SimulatorContext) -> BigDecimal {
        let sigma = ((&ctx.base * &ctx.base + BigDecimal::from(2u64)) / BigDecimal::from(12u64))
            .sqrt()
            .expect("sqrt balanced gadget digit variance failed");
        high_probability_envelope_from_sigma(&sigma)
    }

    // this only support d = 1
    pub fn gadget_decomposed(ctx: Arc<SimulatorContext>, ncol: usize) -> Self {
        let digit_norm = Self::balanced_gadget_digit_norm(&ctx);
        Self::from_parts(
            ctx.m_g,
            ncol,
            PolyNorm::new(ctx.clone(), digit_norm),
            None,
            DependencySet::empty(),
            false,
        )
    }

    pub fn gadget_decomposed_with_secret_size(
        ctx: Arc<SimulatorContext>,
        secret_size: usize,
        ncol: usize,
    ) -> Self {
        let digit_norm = Self::balanced_gadget_digit_norm(&ctx);
        Self::from_parts(
            secret_size * ctx.log_base_q,
            ncol,
            PolyNorm::new(ctx.clone(), digit_norm),
            None,
            DependencySet::empty(),
            false,
        )
    }

    pub fn with_deps(mut self, deps: DependencySet, clt_ready: bool) -> Self {
        self.deps = deps;
        self.clt_ready = clt_ready;
        self
    }

    pub fn rhs_pubkey_gadget(
        ctx: Arc<SimulatorContext>,
        ncol: usize,
        deps: DependencySet,
        clt_ready: bool,
    ) -> Self {
        Self::gadget_decomposed(ctx, ncol).with_deps(deps, clt_ready)
    }

    pub fn maximum_coefficient_bound(&self) -> BigDecimal {
        self.poly_norm.norm.clone()
    }

    #[inline]
    pub fn ctx(&self) -> &SimulatorContext {
        &self.poly_norm.ctx
    }
    #[inline]
    pub fn clone_ctx(&self) -> Arc<SimulatorContext> {
        self.poly_norm.ctx.clone()
    }

    pub fn split_rows(&self, top_row_size: usize) -> (Self, Self) {
        assert!(top_row_size <= self.nrow);
        let mut top = self.clone();
        top.nrow = top_row_size;
        let mut bottom = self.clone();
        bottom.nrow = self.nrow - top_row_size;
        (top, bottom)
    }

    pub fn split_cols(&self, left_col_size: usize) -> (Self, Self) {
        assert!(left_col_size <= self.ncol);
        let mut left = self.clone();
        left.ncol = left_col_size;
        left.ncol_sqrt =
            BigDecimal::from(left_col_size as u64).sqrt().expect("sqrt(ncol) to failed");
        let mut right = self.clone();
        right.ncol = self.ncol - left_col_size;
        right.ncol_sqrt = BigDecimal::from(right.ncol as u64).sqrt().expect("sqrt(ncol) to failed");
        (left, right)
    }
}

impl_binop_with_refs!(PolyMatrixNorm => Add::add(self, rhs: &PolyMatrixNorm) -> PolyMatrixNorm {
    assert!(self.poly_norm.ctx == rhs.poly_norm.ctx, "ctx must match");
    assert!(self.nrow == rhs.nrow && self.ncol == rhs.ncol, "matrix dims must match");
    PolyMatrixNorm {
        nrow: self.nrow,
        ncol: self.ncol,
        ncol_sqrt: self.ncol_sqrt.clone(),
        poly_norm: &self.poly_norm + &rhs.poly_norm,
        zero_rows: None,
        deps: self.deps.union(&rhs.deps),
        clt_ready: false,
    }
});

impl AddAssign for PolyMatrixNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.nrow == rhs.nrow && self.ncol == rhs.ncol, "matrix dims must match");
        self.poly_norm += rhs.poly_norm;
        self.zero_rows = None;
        self.deps = self.deps.union(&rhs.deps);
        self.clt_ready = false;
        // nrow, ncol, ncol_sqrt unchanged
    }
}

impl_binop_with_refs!(PolyMatrixNorm => Mul::mul(self, rhs: &PolyMatrixNorm) -> PolyMatrixNorm {
    assert!(self.poly_norm.ctx == rhs.poly_norm.ctx, "ctx must match");
    assert!(self.ncol == rhs.nrow, "inner dims must match for multiplication");
    let effective_inner_dim = if let Some(z) = rhs.zero_rows { self.ncol - z } else { self.ncol };
    let deps_disjoint = self.deps.is_disjoint(&rhs.deps);
    let use_clt = deps_disjoint && (self.clt_ready || rhs.clt_ready);
    let out_clt_ready = deps_disjoint && self.clt_ready && rhs.clt_ready;
    let inner = BigDecimal::from(effective_inner_dim as u64);
    let contraction = if self.poly_norm.is_const_poly || rhs.poly_norm.is_const_poly {
        inner
    } else {
        inner * &self.ctx().ring_dim_sqrt * &self.ctx().ring_dim_sqrt
    };
    let scale = if use_clt { contraction.sqrt().expect("sqrt(K) failed") } else { contraction.clone() };
    let norm = scale * &self.poly_norm.norm * &rhs.poly_norm.norm;
    debug!(
        operation = "matrix_mul",
        k = %contraction,
        lhs_norm = %self.poly_norm.norm,
        rhs_norm = %rhs.poly_norm.norm,
        out_norm = %norm,
        lhs_deps = ?self.deps,
        rhs_deps = ?rhs.deps,
        deps_disjoint,
        lhs_clt_ready = self.clt_ready,
        rhs_clt_ready = rhs.clt_ready,
        use_clt,
        out_clt_ready,
        rule = if use_clt { "CLT" } else { "WORST_CASE" },
        "simulator matrix norm multiplication"
    );
    if out_clt_ready {
        debug!(rule = "PRODUCT_CLOSURE", "simulator matrix norm product closure");
    }
    PolyMatrixNorm {
        nrow: self.nrow,
        ncol: rhs.ncol,
        ncol_sqrt: rhs.ncol_sqrt.clone(),
        poly_norm: PolyNorm::new(self.clone_ctx(), norm),
        zero_rows: None,
        deps: self.deps.union(&rhs.deps),
        clt_ready: out_clt_ready,
    }
});

impl MulAssign for PolyMatrixNorm {
    fn mul_assign(&mut self, rhs: Self) {
        let out = self.clone() * rhs;
        *self = out;
    }
}

impl Mul<&PolyNorm> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: &PolyNorm) -> Self::Output {
        assert!(self.poly_norm.ctx == rhs.ctx, "ctx must match");
        let is_zero = rhs.norm.is_zero();
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
            deps: if is_zero { DependencySet::empty() } else { self.deps },
            clt_ready: if is_zero { false } else { self.clt_ready },
        }
    }
}

impl Mul<&PolyNorm> for &PolyMatrixNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: &PolyNorm) -> Self::Output {
        assert!(self.poly_norm.ctx == rhs.ctx, "ctx must match");
        let is_zero = rhs.norm.is_zero();
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt.clone(),
            poly_norm: &self.poly_norm * rhs,
            zero_rows: None,
            deps: if is_zero { DependencySet::empty() } else { self.deps.clone() },
            clt_ready: if is_zero { false } else { self.clt_ready },
        }
    }
}

impl Mul<PolyMatrixNorm> for PolyNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: PolyMatrixNorm) -> Self::Output {
        rhs * &self
    }
}

impl Mul<BigDecimal> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: BigDecimal) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&BigDecimal> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: &BigDecimal) -> Self::Output {
        let is_zero = rhs.is_zero();
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
            deps: if is_zero { DependencySet::empty() } else { self.deps },
            clt_ready: if is_zero { false } else { self.clt_ready },
        }
    }
}

impl Mul<PolyMatrixNorm> for BigDecimal {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: PolyMatrixNorm) -> Self::Output {
        rhs * self
    }
}

impl Mul<u32> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: u32) -> Self::Output {
        self * BigDecimal::from(rhs)
    }
}

impl Mul<u32> for &PolyMatrixNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: u32) -> Self::Output {
        self.clone() * BigDecimal::from(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;

    fn test_ctx_with_base(base: u64) -> Arc<SimulatorContext> {
        Arc::new(SimulatorContext::new(BigDecimal::from(4u64), BigDecimal::from(base), 2, 6, 3))
    }

    fn balanced_digit_step(value: i64, base: i64) -> (i64, i64) {
        let quotient = value.div_euclid(base);
        let remainder = value.rem_euclid(base);
        let half = base / 2;
        if remainder < half {
            (remainder, quotient)
        } else if remainder > half {
            (remainder - base, quotient + 1)
        } else if quotient % 2 == 0 {
            (half, quotient)
        } else {
            (half - base, quotient + 1)
        }
    }

    #[test]
    fn gadget_decomposed_uses_balanced_digit_second_moment_sigma() {
        let base = 8u64;
        let half = i64::try_from(base / 2).expect("small test base fits i64");
        let mut weighted_digits = Vec::new();
        weighted_digits.push((-half, 1u64));
        for digit in (-half + 1)..half {
            weighted_digits.push((digit, 2u64));
        }
        weighted_digits.push((half, 1u64));

        assert_eq!(weighted_digits.first().expect("support is nonempty").0, -half);
        assert_eq!(weighted_digits.last().expect("support is nonempty").0, half);

        let weight_sum = weighted_digits.iter().map(|(_, weight)| *weight).sum::<u64>();
        assert_eq!(weight_sum, 2 * base);
        let mean_numerator = weighted_digits
            .iter()
            .map(|(digit, weight)| i128::from(*digit) * i128::from(*weight))
            .sum::<i128>();
        assert_eq!(mean_numerator, 0);

        let second_moment_numerator = weighted_digits
            .iter()
            .map(|(digit, weight)| {
                BigDecimal::from((digit * digit) as u64) * BigDecimal::from(*weight)
            })
            .fold(BigDecimal::zero(), |acc, value| acc + value);
        let second_moment = second_moment_numerator / BigDecimal::from(weight_sum);
        let expected_variance =
            (BigDecimal::from(base * base) + BigDecimal::from(2u64)) / BigDecimal::from(12u64);
        assert_eq!(second_moment, expected_variance);

        assert_eq!(balanced_digit_step(4, base as i64), (4, 0));
        assert_eq!(balanced_digit_step(12, base as i64), (-4, 2));

        let ctx = test_ctx_with_base(base);
        let expected_sigma = expected_variance.sqrt().expect("variance sqrt should exist");
        let expected_norm = high_probability_envelope_from_sigma(&expected_sigma);
        let decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), 3);
        assert_eq!(decomposed.nrow, ctx.m_g);
        assert_eq!(decomposed.ncol, 3);
        assert_eq!(decomposed.poly_norm.norm, expected_norm);

        let secret_decomposed =
            PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), 5, 2);
        assert_eq!(secret_decomposed.nrow, 5 * ctx.log_base_q);
        assert_eq!(secret_decomposed.ncol, 2);
        assert_eq!(secret_decomposed.poly_norm.norm, expected_norm);
    }

    #[test]
    fn dependency_set_tracks_disjoint_clone_overlap_and_union() {
        let ctx = test_ctx_with_base(8);
        let a = DependencySet::singleton(ctx.fresh_source_id());
        let b = DependencySet::singleton(ctx.fresh_source_id());
        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&a.clone()));
        let union = b.union(&a);
        assert_eq!(union, a.union(&b));
        assert!(!DependencySet::Unknown.is_disjoint(&a));
    }

    #[test]
    fn matrix_mul_uses_clt_only_for_disjoint_ready_operands() {
        let ctx = test_ctx_with_base(8);
        let lhs = PolyMatrixNorm::sample_gauss(ctx.clone(), 1, 3, BigDecimal::from(2u64));
        let rhs = PolyMatrixNorm::sample_gauss(ctx.clone(), 3, 1, BigDecimal::from(5u64));
        let out = lhs.clone() * &rhs;
        let lhs_norm = high_probability_envelope_from_sigma(&BigDecimal::from(2u64));
        let rhs_norm = high_probability_envelope_from_sigma(&BigDecimal::from(5u64));
        let k = BigDecimal::from(3u64) * &ctx.ring_dim_sqrt * &ctx.ring_dim_sqrt;
        assert_eq!(out.poly_norm.norm, k.sqrt().unwrap() * &lhs_norm * &rhs_norm);
        assert!(out.clt_ready);

        let overlap = lhs.clone() * &lhs.clone().split_cols(1).0.transpose_like_for_test(3);
        let k_overlap = BigDecimal::from(3u64) * &ctx.ring_dim_sqrt * &ctx.ring_dim_sqrt;
        assert_eq!(overlap.poly_norm.norm, k_overlap * &lhs_norm * &lhs_norm);
        assert!(!overlap.clt_ready);
    }

    #[test]
    fn matrix_mul_with_exactly_one_clt_ready_uses_clt_without_product_closure() {
        let ctx = test_ctx_with_base(8);
        let lhs = PolyMatrixNorm::sample_gauss(ctx.clone(), 1, 2, BigDecimal::from(2u64));
        let rhs = PolyMatrixNorm::new(ctx.clone(), 2, 1, BigDecimal::from(7u64), None)
            .with_deps(DependencySet::singleton(ctx.fresh_source_id()), false);
        let out = lhs * &rhs;
        let k = BigDecimal::from(2u64) * &ctx.ring_dim_sqrt * &ctx.ring_dim_sqrt;
        assert_eq!(
            out.poly_norm.norm,
            k.sqrt().unwrap() * BigDecimal::from(13u64) * BigDecimal::from(7u64)
        );
        assert!(!out.clt_ready);
    }

    trait TestTransposeShape {
        fn transpose_like_for_test(self, nrow: usize) -> Self;
    }

    impl TestTransposeShape for PolyMatrixNorm {
        fn transpose_like_for_test(mut self, nrow: usize) -> Self {
            self.nrow = nrow;
            self.ncol = 1;
            self.ncol_sqrt = BigDecimal::from(1u64);
            self
        }
    }
}
