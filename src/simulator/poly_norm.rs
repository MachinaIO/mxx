use crate::{impl_binop_with_refs, simulator::SimulatorContext};
use bigdecimal::BigDecimal;
use num_traits::{One, Zero};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

pub fn maximum_coefficient_bound_from_sigma(sigma: &BigDecimal) -> BigDecimal {
    assert!(*sigma >= BigDecimal::zero(), "sigma must be nonnegative");
    sigma * BigDecimal::from(13u64) / BigDecimal::from(2u64)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyNorm {
    pub ctx: Arc<SimulatorContext>,
    pub sigma: BigDecimal,
    pub is_constant_poly: bool,
}

impl PolyNorm {
    fn assert_nonnegative_sigma(sigma: &BigDecimal) {
        assert!(*sigma >= BigDecimal::zero(), "sigma must be nonnegative");
    }

    pub fn new(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        Self::assert_nonnegative_sigma(&sigma);
        PolyNorm { ctx, sigma, is_constant_poly: false }
    }

    pub fn constant(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        Self::assert_nonnegative_sigma(&sigma);
        PolyNorm { ctx, sigma, is_constant_poly: true }
    }

    pub fn into_constant_poly(mut self) -> Self {
        self.is_constant_poly = true;
        self
    }

    pub fn one(ctx: Arc<SimulatorContext>) -> Self {
        Self::constant(ctx, BigDecimal::one())
    }

    pub fn sample_gauss(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        Self::assert_nonnegative_sigma(&sigma);
        PolyNorm { ctx, sigma, is_constant_poly: false }
    }

    pub fn maximum_coefficient_bound(&self) -> BigDecimal {
        maximum_coefficient_bound_from_sigma(&self.sigma)
    }

    pub fn max_coefficients_bound(&self) -> BigDecimal {
        if self.is_constant_poly { self.sigma.clone() } else { self.maximum_coefficient_bound() }
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm {
        ctx: self.ctx.clone(),
        sigma: &self.sigma + &rhs.sigma,
        is_constant_poly: self.is_constant_poly && rhs.is_constant_poly,
    }
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.sigma += rhs.sigma;
        self.is_constant_poly = self.is_constant_poly && rhs.is_constant_poly;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    let mut sigma = &self.sigma * &rhs.sigma;
    if !self.is_constant_poly && !rhs.is_constant_poly {
        sigma *= &self.ctx.ring_dim_sqrt;
    }
    PolyNorm {
        ctx: self.ctx.clone(),
        sigma,
        is_constant_poly: self.is_constant_poly && rhs.is_constant_poly,
    }
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        let lhs_is_constant_poly = self.is_constant_poly;
        let rhs_is_constant_poly = rhs.is_constant_poly;
        self.sigma = &self.sigma * rhs.sigma;
        if !lhs_is_constant_poly && !rhs_is_constant_poly {
            self.sigma *= &self.ctx.ring_dim_sqrt;
        }
        self.is_constant_poly = lhs_is_constant_poly && rhs_is_constant_poly;
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
        Self::assert_nonnegative_sigma(rhs);
        PolyNorm { ctx: self.ctx, sigma: self.sigma * rhs, is_constant_poly: self.is_constant_poly }
    }
}

impl Mul<PolyNorm> for BigDecimal {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm::assert_nonnegative_sigma(&self);
        PolyNorm { ctx: rhs.ctx, sigma: rhs.sigma * self, is_constant_poly: rhs.is_constant_poly }
    }
}
