use crate::{impl_binop_with_refs, simulator::SimulatorContext};
use bigdecimal::BigDecimal;
use num_traits::{FromPrimitive, One};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyNorm {
    pub ctx: Arc<SimulatorContext>,
    pub norm: BigDecimal,
    pub is_constant: bool,
}

impl PolyNorm {
    pub fn new(ctx: Arc<SimulatorContext>, norm: BigDecimal) -> Self {
        PolyNorm { ctx, norm, is_constant: false }
    }

    pub fn constant(ctx: Arc<SimulatorContext>, norm: BigDecimal) -> Self {
        PolyNorm { ctx, norm, is_constant: true }
    }

    pub fn into_constant(mut self) -> Self {
        self.is_constant = true;
        self
    }

    pub fn one(ctx: Arc<SimulatorContext>) -> Self {
        Self::constant(ctx, BigDecimal::one())
    }

    pub fn sample_gauss(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        let norm = sigma * BigDecimal::from_f32(6.5).unwrap();
        PolyNorm { ctx, norm, is_constant: false }
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm {
        ctx: self.ctx.clone(),
        norm: &self.norm + &rhs.norm,
        is_constant: self.is_constant && rhs.is_constant,
    }
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm += rhs.norm;
        self.is_constant = self.is_constant && rhs.is_constant;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    let mut norm = &self.norm * &rhs.norm;
    if !self.is_constant && !rhs.is_constant {
        norm *= &self.ctx.ring_dim_sqrt;
    }
    PolyNorm {
        ctx: self.ctx.clone(),
        norm,
        is_constant: self.is_constant && rhs.is_constant,
    }
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        let lhs_is_constant = self.is_constant;
        let rhs_is_constant = rhs.is_constant;
        self.norm = &self.norm * rhs.norm;
        if !lhs_is_constant && !rhs_is_constant {
            self.norm *= &self.ctx.ring_dim_sqrt;
        }
        self.is_constant = lhs_is_constant && rhs_is_constant;
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
        PolyNorm { ctx: self.ctx, norm: self.norm * rhs, is_constant: self.is_constant }
    }
}

impl Mul<PolyNorm> for BigDecimal {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm { ctx: rhs.ctx, norm: rhs.norm * self, is_constant: rhs.is_constant }
    }
}
