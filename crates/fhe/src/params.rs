use crate::FheError;
use mxx_dsl::{Mat, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, node::SampleRange};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::Zero;

/// Shared public parameters. The exact CRT layout is owned by the primitive
/// parameter type; no duplicate per-level or gadget metadata is maintained.
#[derive(Clone, Debug)]
pub struct FheCommonParams {
    pub ring: DCRTPolyParams,
    pub secret_range: SampleRange,
    pub error_sigma: f64,
    pub error_cutoff: BigUint,
}

impl FheCommonParams {
    pub fn validate(&self) -> Result<(), FheError> {
        if self.ring.ring_dimension() < 2 || self.ring.dropped_moduli() != 0 {
            return Err(FheError::InvalidParameters(
                "an exact nonempty negacyclic CRT ring is required",
            ));
        }
        if !self.error_sigma.is_finite() || self.error_sigma <= 0.0 || self.error_cutoff.is_zero() {
            return Err(FheError::InvalidParameters(
                "positive finite Gaussian sigma and cutoff are required",
            ));
        }
        let env = ParamEnv::default();
        let lo = self.secret_range.minimum.evaluate(&env).ok();
        let hi = self.secret_range.maximum.evaluate(&env).ok();
        if hi != Some(BigInt::from(1)) ||
            (lo != Some(BigInt::from(-1)) && lo != Some(BigInt::from(0)))
        {
            return Err(FheError::InvalidParameters("secret interval must be [0,1] or [-1,1]"));
        }
        if self.error_cutoff.clone() * 2u8 >= *self.ring.modulus() {
            return Err(FheError::InvalidParameters(
                "Gaussian cutoff must be below half the ring modulus",
            ));
        }
        Ok(())
    }

    pub fn parameters_at(&self, level: usize) -> Result<DCRTPolyParams, FheError> {
        let (moduli, _, _) = self.ring.to_crt();
        let primes = moduli.get(..=level).ok_or(FheError::LevelMismatch)?;
        let modulus = primes.iter().map(|p| BigUint::from(*p)).product::<BigUint>();
        self.ring.select_modulus(&modulus).ok_or(FheError::LevelMismatch)
    }

    pub(crate) fn ring(&self) -> Ring {
        Ring::new(
            IntExpr::constant(BigInt::from(self.ring.modulus().as_ref().clone())),
            self.ring.ring_dimension(),
        )
    }

    pub(crate) fn sample_secret(&self) -> Mat {
        self.ring().uniform_interval(
            (1, 1),
            self.secret_range.minimum.clone(),
            self.secret_range.maximum.clone(),
        )
    }

    pub(crate) fn gaussian(&self, parameters: &DCRTPolyParams, rows: usize, columns: usize) -> Mat {
        Ring::new(
            IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            parameters.ring_dimension(),
        )
        .gaussian(
            (rows, columns),
            mxx_ir_core::RealExpr::from_f64_exact(self.error_sigma)
                .expect("validated finite sigma"),
            IntExpr::constant(BigInt::from(self.error_cutoff.clone())),
        )
    }
}
