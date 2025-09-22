use crate::{impl_binop_with_refs, simulator::SimulatorContext};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PolyNorm {
    pub ctx: Arc<SimulatorContext>,
    pub norm: f64,
}

impl PolyNorm {
    pub fn new(ctx: Arc<SimulatorContext>, norm: f64) -> Self {
        PolyNorm { ctx, norm }
    }

    pub fn one(ctx: Arc<SimulatorContext>) -> Self {
        Self::new(ctx, 1.0)
    }

    pub fn sample_gauss(ctx: Arc<SimulatorContext>, sigma: f64) -> Self {
        let norm = ctx.secpar_sqrt * sigma;
        PolyNorm { ctx, norm }
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm { ctx: self.ctx.clone(), norm: self.norm + rhs.norm }
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm += rhs.norm;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(self.ctx == rhs.ctx, "ctx must match");
    PolyNorm { ctx: self.ctx.clone(), norm: self.norm * rhs.norm * self.ctx.ring_dim_sqrt }
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ctx == rhs.ctx, "ctx must match");
        self.norm = self.norm * rhs.norm * self.ctx.ring_dim_sqrt;
    }
}

impl Mul<f64> for PolyNorm {
    type Output = PolyNorm;
    fn mul(self, rhs: f64) -> Self::Output {
        PolyNorm { ctx: self.ctx, norm: self.norm * rhs }
    }
}

impl Mul<PolyNorm> for f64 {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm { ctx: rhs.ctx, norm: rhs.norm * self }
    }
}
