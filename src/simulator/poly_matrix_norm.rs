use super::poly_norm::PolyNorm;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use crate::impl_binop_with_refs;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyMatrixNorm {
    pub nrow: usize,
    pub ncol: usize,
    pub ncol_sqrt: f64,
    pub poly_norm: PolyNorm,
    pub zero_rows: Option<usize>,
}

impl PolyMatrixNorm {
    pub fn sample_uniform(nrow: usize, ncol: usize, ring_dim_sqrt: f64, bound: f64) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: (ncol as f64).sqrt(),
            poly_norm: PolyNorm::sample_uniform(ring_dim_sqrt, bound),
            zero_rows: None,
        }
    }

    pub fn sample_gauss(
        nrow: usize,
        ncol: usize,
        ring_dim_sqrt: f64,
        secpar_sqrt: f64,
        sigma: f64,
    ) -> Self {
        PolyMatrixNorm {
            nrow,
            ncol,
            ncol_sqrt: (ncol as f64).sqrt(),
            poly_norm: PolyNorm::sample_gauss(ring_dim_sqrt, secpar_sqrt, sigma),
            zero_rows: None,
        }
    }

    pub fn sample_preimage(ring_dim_sqrt: f64, base: f64, log_base_q: usize) -> Self {
        let ncol = log_base_q + 2;
        let c_0 = 1.8_f64;
        let c_1 = 4.7_f64;
        let sigma = 4.578_f64;
        let norm = 6.0 *
            c_0 *
            sigma *
            ((base + 1.0) * sigma) *
            (ring_dim_sqrt * ring_dim_sqrt.sqrt() + 1.414 * ring_dim_sqrt + c_1);
        PolyMatrixNorm {
            nrow: 1,
            ncol,
            ncol_sqrt: (ncol as f64).sqrt(),
            poly_norm: PolyNorm { ring_dim_sqrt, norm },
            zero_rows: None,
        }
    }
}

impl_binop_with_refs!(PolyMatrixNorm => Add::add(self, rhs: &PolyMatrixNorm) -> PolyMatrixNorm {
    assert!(self.nrow == rhs.nrow && self.ncol == rhs.ncol, "matrix dims must match");
    PolyMatrixNorm {
        nrow: self.nrow,
        ncol: self.ncol,
        ncol_sqrt: self.ncol_sqrt,
        poly_norm: &self.poly_norm + &rhs.poly_norm,
        zero_rows: None,
    }
});

impl AddAssign for PolyMatrixNorm {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.nrow == rhs.nrow && self.ncol == rhs.ncol, "matrix dims must match");
        self.poly_norm = self.poly_norm + rhs.poly_norm;
        self.zero_rows = None;
        // nrow, ncol, ncol_sqrt unchanged
    }
}

impl_binop_with_refs!(PolyMatrixNorm => Mul::mul(self, rhs: &PolyMatrixNorm) -> PolyMatrixNorm {
    assert!(self.ncol == rhs.nrow, "inner dims must match for multiplication");
    let zero_rows_f = rhs.zero_rows.map(|z| z as f64).unwrap_or(0.0);
    let scale = self.ncol_sqrt - zero_rows_f;
    let pn = (&self.poly_norm * &rhs.poly_norm) * scale;
    PolyMatrixNorm { nrow: self.nrow, ncol: rhs.ncol, ncol_sqrt: rhs.ncol_sqrt, poly_norm: pn, zero_rows: None }
});

impl MulAssign for PolyMatrixNorm {
    fn mul_assign(&mut self, rhs: Self) {
        assert!(self.ncol == rhs.nrow, "inner dims must match for multiplication");
        let zero_rows_f = rhs.zero_rows.map(|z| z as f64).unwrap_or(0.0);
        let scale = self.ncol_sqrt - zero_rows_f;
        self.poly_norm = (self.poly_norm * rhs.poly_norm) * scale;
        self.nrow = self.nrow;
        self.ncol = rhs.ncol;
        self.ncol_sqrt = rhs.ncol_sqrt;
        self.zero_rows = None;
    }
}

impl Mul<PolyNorm> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: PolyNorm) -> Self::Output {
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
        }
    }
}

impl Mul<PolyMatrixNorm> for PolyNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: PolyMatrixNorm) -> Self::Output {
        rhs * self
    }
}

// reference-reference heterogeneous multiplication
impl<'a, 'b> Mul<&'b PolyNorm> for &'a PolyMatrixNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: &'b PolyNorm) -> Self::Output {
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: &self.poly_norm * rhs,
            zero_rows: None,
        }
    }
}

impl<'a, 'b> Mul<&'b PolyMatrixNorm> for &'a PolyNorm {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: &'b PolyMatrixNorm) -> Self::Output {
        PolyMatrixNorm {
            nrow: rhs.nrow,
            ncol: rhs.ncol,
            ncol_sqrt: rhs.ncol_sqrt,
            poly_norm: &rhs.poly_norm * self,
            zero_rows: None,
        }
    }
}

impl Mul<f64> for PolyMatrixNorm {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        PolyMatrixNorm {
            nrow: self.nrow,
            ncol: self.ncol,
            ncol_sqrt: self.ncol_sqrt,
            poly_norm: self.poly_norm * rhs,
            zero_rows: None,
        }
    }
}

impl Mul<PolyMatrixNorm> for f64 {
    type Output = PolyMatrixNorm;
    fn mul(self, rhs: PolyMatrixNorm) -> Self::Output {
        rhs * self
    }
}
