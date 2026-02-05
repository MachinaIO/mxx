use super::poly_norm::PolyNorm;
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
        norm: BigDecimal,
        zero_rows: Option<usize>,
    ) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::new(ctx, norm),
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

    // this only support d = 1
    pub fn gadget_decomposed(ctx: Arc<SimulatorContext>, ncol: usize) -> Self {
        PolyMatrixNorm {
            nrow: ctx.m_g,
            ncol,
            ncol_sqrt: BigDecimal::from(ncol as u64).sqrt().expect("sqrt(ncol) to failed"),
            poly_norm: PolyNorm::new(ctx.clone(), ctx.base.clone() - BigDecimal::from(1u64)),
            zero_rows: None,
        }
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
