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
}

impl PolyNorm {
    pub fn new(ctx: Arc<SimulatorContext>, norm: BigDecimal) -> Self {
        PolyNorm { ctx, norm }
    }

    pub fn one(ctx: Arc<SimulatorContext>) -> Self {
        Self::new(ctx, BigDecimal::one())
    }

    pub fn sample_gauss(ctx: Arc<SimulatorContext>, sigma: BigDecimal) -> Self {
        let norm = sigma * BigDecimal::from_f32(6.5).unwrap();
        PolyNorm { ctx, norm }
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm { ctx: self.ctx.clone(), norm: &self.norm + &rhs.norm }
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm += rhs.norm;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm { ctx: self.ctx.clone(), norm: &self.norm * &rhs.norm * &self.ctx.ring_dim_sqrt }
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm = &self.norm * rhs.norm * &self.ctx.ring_dim_sqrt;
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
        PolyNorm { ctx: self.ctx, norm: self.norm * rhs }
    }
}

impl Mul<PolyNorm> for BigDecimal {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm { ctx: rhs.ctx, norm: rhs.norm * self }
    }
}
