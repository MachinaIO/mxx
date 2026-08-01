use crate::SimulatorContext;
use bigdecimal::BigDecimal;
use mxx_primitives::impl_binop_with_refs;
use num_traits::{One, Zero};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

pub use mxx_primitives::sampler::bounds::{
    high_probability_envelope_from_sigma, maximum_coefficient_bound_from_sigma,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyNorm {
    pub ctx: Arc<SimulatorContext>,
    pub norm: BigDecimal,
    pub sigma: BigDecimal,
    pub is_const_poly: bool,
}

impl PolyNorm {
    fn assert_nonnegative_norm(norm: &BigDecimal) {
        assert!(*norm >= BigDecimal::zero(), "norm must be nonnegative");
    }

    fn from_norm(ctx: Arc<SimulatorContext>, norm: BigDecimal, is_const_poly: bool) -> Self {
        Self::assert_nonnegative_norm(&norm);
        PolyNorm { ctx, sigma: norm.clone(), norm, is_const_poly }
    }

    pub fn new(ctx: Arc<SimulatorContext>, norm: BigDecimal) -> Self {
        Self::from_norm(ctx, norm, false)
    }

    pub fn constant(ctx: Arc<SimulatorContext>, norm: BigDecimal) -> Self {
        Self::from_norm(ctx, norm, true)
    }

    pub fn into_constant_poly(mut self) -> Self {
        self.is_const_poly = true;
        self
    }

    pub fn one(ctx: Arc<SimulatorContext>) -> Self {
        Self::constant(ctx, BigDecimal::one())
    }

    pub fn sample_gauss(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        Self::from_norm(ctx, high_probability_envelope_from_sigma(&sigma), false)
    }

    pub fn maximum_coefficient_bound(&self) -> BigDecimal {
        self.norm.clone()
    }

    pub fn max_coefficients_bound(&self) -> BigDecimal {
        self.norm.clone()
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm::from_norm(
        self.ctx.clone(),
        &self.norm + &rhs.norm,
        self.is_const_poly && rhs.is_const_poly,
    )
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm += rhs.norm;
        self.sigma = self.norm.clone();
        self.is_const_poly = self.is_const_poly && rhs.is_const_poly;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    let mut norm = &self.norm * &rhs.norm;
    if !self.is_const_poly && !rhs.is_const_poly {
        norm *= &self.ctx.ring_dim_sqrt;
    }
    PolyNorm::from_norm(self.ctx.clone(), norm, self.is_const_poly && rhs.is_const_poly)
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        let lhs_is_const_poly = self.is_const_poly;
        let rhs_is_const_poly = rhs.is_const_poly;
        self.norm = &self.norm * rhs.norm;
        if !lhs_is_const_poly && !rhs_is_const_poly {
            self.norm *= &self.ctx.ring_dim_sqrt;
        }
        self.sigma = self.norm.clone();
        self.is_const_poly = lhs_is_const_poly && rhs_is_const_poly;
    }
}

impl Mul<BigDecimal> for PolyNorm {
    type Output = Self;
    fn mul(self, rhs: BigDecimal) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&BigDecimal> for PolyNorm {
    type Output = Self;
    fn mul(self, rhs: &BigDecimal) -> Self::Output {
        Self::assert_nonnegative_norm(rhs);
        PolyNorm::from_norm(self.ctx, self.norm * rhs, self.is_const_poly)
    }
}

impl Mul<PolyNorm> for BigDecimal {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm::assert_nonnegative_norm(&self);
        PolyNorm::from_norm(rhs.ctx, rhs.norm * self, rhs.is_const_poly)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_envelope_remains_six_point_five_sigma() {
        let sigma = BigDecimal::from(4u32);
        assert_eq!(high_probability_envelope_from_sigma(&sigma), BigDecimal::from(26u32));
    }

    #[test]
    fn two_nonconstant_factors_pay_the_ring_sqrt_factor() {
        let ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(4u32),
            BigDecimal::from(2u32),
            1,
            1,
            1,
        ));
        let product = PolyNorm::new(ctx.clone(), BigDecimal::from(2u32)) *
            PolyNorm::new(ctx, BigDecimal::from(3u32));
        assert_eq!(product.norm, BigDecimal::from(24u32));
    }
}
