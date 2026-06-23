use super::poly_norm::{PolyNorm, maximum_coefficient_bound_from_sigma};
use crate::{impl_binop_with_refs, simulator::SimulatorContext};
use bigdecimal::BigDecimal;
use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyMatrixNorm {
    pub nrow: usize,
    pub ncol: usize,
    pub ncol_sqrt: BigDecimal,
    pub poly_norm: PolyNorm,
    pub zero_rows: Option<usize>,
}

impl PolyMatrixNorm {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        sigma: BigDecimal,
        zero_rows: Option<usize>,
    ) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::new(ctx, sigma),
            zero_rows,
        }
    }

    pub fn sample_gauss(
        ctx: Arc<SimulatorContext>,
        nrow: usize,
        ncol: usize,
        sigma: BigDecimal,
    ) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::sample_gauss(ctx, sigma),
            zero_rows: None,
        }
    }

    fn balanced_gadget_digit_sigma(ctx: &SimulatorContext) -> BigDecimal {
        ((&ctx.base * &ctx.base + BigDecimal::from(2u64)) / BigDecimal::from(12u64))
            .sqrt()
            .expect("sqrt balanced gadget digit variance failed")
    }

    // this only support d = 1
    pub fn gadget_decomposed(ctx: Arc<SimulatorContext>, ncol: usize) -> Self {
        let digit_sigma = Self::balanced_gadget_digit_sigma(&ctx);
        PolyMatrixNorm {
            nrow: ctx.m_g,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::new(ctx.clone(), digit_sigma),
            zero_rows: None,
        }
    }

    pub fn gadget_decomposed_with_secret_size(
        ctx: Arc<SimulatorContext>,
        secret_size: usize,
        ncol: usize,
    ) -> Self {
        let digit_sigma = Self::balanced_gadget_digit_sigma(&ctx);
        PolyMatrixNorm {
            nrow: secret_size * ctx.log_base_q,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::new(ctx.clone(), digit_sigma),
            zero_rows: None,
        }
    }

    pub fn maximum_coefficient_bound(&self) -> BigDecimal {
        maximum_coefficient_bound_from_sigma(&self.poly_norm.sigma)
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
    }
});

impl AddAssign for PolyMatrixNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.nrow == rhs.nrow && self.ncol == rhs.ncol, "matrix dims must match");
        self.poly_norm += rhs.poly_norm;
        self.zero_rows = None;
        // nrow, ncol, ncol_sqrt unchanged
    }
}

impl_binop_with_refs!(PolyMatrixNorm => Mul::mul(self, rhs: &PolyMatrixNorm) -> PolyMatrixNorm {
    assert!(self.poly_norm.ctx == rhs.poly_norm.ctx, "ctx must match");
    assert!(self.ncol == rhs.nrow, "inner dims must match for multiplication");
    let scale = if let Some(z) = rhs.zero_rows {
        BigDecimal::from(self.ncol as u64 - z as u64).sqrt().expect("sqrt(ncol) to failed")
    } else {
        self.ncol_sqrt.clone()
    };
    let pn = (&self.poly_norm * &rhs.poly_norm) * scale;
    PolyMatrixNorm { nrow: self.nrow, ncol: rhs.ncol, ncol_sqrt: rhs.ncol_sqrt.clone(), poly_norm: pn, zero_rows: None }
});

impl MulAssign for PolyMatrixNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ncol == rhs.nrow, "inner dims must match for multiplication");
        let scale = if let Some(z) = rhs.zero_rows {
            BigDecimal::from(self.ncol as u64 - z as u64).sqrt().expect("sqrt(ncol) to failed")
        } else {
            self.ncol_sqrt.clone()
        };
        self.poly_norm = (self.poly_norm.clone() * rhs.poly_norm) * scale;
        self.ncol = rhs.ncol;
        self.ncol_sqrt = rhs.ncol_sqrt;
        self.zero_rows = None;
    }
}

impl Mul<&PolyNorm> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: &PolyNorm) -> Self::Output {
        assert!(self.poly_norm.ctx == rhs.ctx, "ctx must match");
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
        }
    }
}

impl Mul<&PolyNorm> for &PolyMatrixNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: &PolyNorm) -> Self::Output {
        assert!(self.poly_norm.ctx == rhs.ctx, "ctx must match");
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt.clone(),
            poly_norm: &self.poly_norm * rhs,
            zero_rows: None,
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
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
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
        let decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), 3);
        assert_eq!(decomposed.nrow, ctx.m_g);
        assert_eq!(decomposed.ncol, 3);
        assert_eq!(decomposed.poly_norm.sigma, expected_sigma);

        let secret_decomposed =
            PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), 5, 2);
        assert_eq!(secret_decomposed.nrow, 5 * ctx.log_base_q);
        assert_eq!(secret_decomposed.ncol, 2);
        assert_eq!(secret_decomposed.poly_norm.sigma, expected_sigma);
    }
}
