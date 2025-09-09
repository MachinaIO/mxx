use crate::impl_binop_with_refs;
use std::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyNorm {
    pub ring_dim_sqrt: f64,
    pub norm: f64,
}

impl PolyNorm {
    pub fn sample_uniform(ring_dim_sqrt: f64, bound: f64) -> Self {
        PolyNorm { ring_dim_sqrt, norm: bound }
    }

    pub fn sample_gauss(ring_dim_sqrt: f64, secpar_sqrt: f64, sigma: f64) -> Self {
        PolyNorm { ring_dim_sqrt, norm: secpar_sqrt * sigma }
    }
}

impl_binop_with_refs!(PolyNorm => Add::add(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(
        (self.ring_dim_sqrt - rhs.ring_dim_sqrt).abs() <= f64::EPSILON,
        "ring_dim_sqrt must match"
    );
    PolyNorm { ring_dim_sqrt: self.ring_dim_sqrt, norm: self.norm + rhs.norm }
});

impl AddAssign for PolyNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(
            (self.ring_dim_sqrt - rhs.ring_dim_sqrt).abs() <= f64::EPSILON,
            "ring_dim_sqrt must match"
        );
        self.norm += rhs.norm;
    }
}

impl_binop_with_refs!(PolyNorm => Mul::mul(self, rhs: &PolyNorm) -> PolyNorm {
    assert!(
        (self.ring_dim_sqrt - rhs.ring_dim_sqrt).abs() <= f64::EPSILON,
        "ring_dim_sqrt must match"
    );
    PolyNorm { ring_dim_sqrt: self.ring_dim_sqrt, norm: self.norm * rhs.norm * self.ring_dim_sqrt }
});

impl MulAssign for PolyNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(
            (self.ring_dim_sqrt - rhs.ring_dim_sqrt).abs() <= f64::EPSILON,
            "ring_dim_sqrt must match"
        );
        self.norm = self.norm * rhs.norm * self.ring_dim_sqrt;
    }
}

impl Mul<f64> for PolyNorm {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        PolyNorm { ring_dim_sqrt: self.ring_dim_sqrt, norm: self.norm * rhs }
    }
}

impl Mul<PolyNorm> for f64 {
    type Output = PolyNorm;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyNorm { ring_dim_sqrt: rhs.ring_dim_sqrt, norm: rhs.norm * self }
    }
}
