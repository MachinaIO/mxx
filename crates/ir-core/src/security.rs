//! The hardness assumptions a security statement may use.
//!
//! [`AssumptionId`] is the closed list; each assumption's game is written by hand in
//! `MxxSecurity.Assumptions` (`crates/ir-core/lean/security`). A protocol's security declaration,
//! such as the public-key FHE binding of `mxx-fhe`, names the identifiers it uses with
//! [`AssumptionParams`]; nothing outside this list can enter a statement as an assumption.

use crate::expr::Rational;
use num_bigint::{BigInt, BigUint};

/// The hardness assumptions a security statement may use. Mirrors `MxxSecurity.AssumptionId`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AssumptionId {
    /// Decision Ring-LWE: `(a, s a + e)` is indistinguishable from uniform.
    DecisionRlwe,
    /// Circular security of a key-switching key from `s²` to `s`, next to a public key.
    RlweKeySwitchCircular,
}

impl AssumptionId {
    /// The constructor of `MxxSecurity.AssumptionId`.
    pub fn lean_name(self) -> &'static str {
        match self {
            Self::DecisionRlwe => "MxxSecurity.AssumptionId.decisionRlwe",
            Self::RlweKeySwitchCircular => "MxxSecurity.AssumptionId.rlweKeySwitchCircular",
        }
    }

    /// The names of the challenge components, in the order of the Lean challenge type: each
    /// names an input of the simulator's setup graph.
    pub fn challenge_components(self) -> &'static [&'static str] {
        match self {
            Self::DecisionRlwe => &["a", "b"],
            Self::RlweKeySwitchCircular => &["pk_a", "pk_b", "key_a", "key_b"],
        }
    }
}

/// The law of one sampled coefficient. Mirrors `MxxRuntime.SamplerLaw`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplerLaw {
    /// The discrete Gaussian of width `sigma` conditioned on `|x| ≤ cutoff`.
    Gaussian { sigma: Rational, cutoff: BigInt },
    /// The uniform distribution on `[minimum, maximum]`.
    Interval { minimum: BigInt, maximum: BigInt },
    /// The uniform distribution on `[0, modulus)`.
    Residue { modulus: BigUint },
}

impl SamplerLaw {
    /// The Lean term of this law.
    pub fn lean(&self) -> String {
        match self {
            Self::Gaussian { sigma, cutoff } => format!(
                "(MxxRuntime.SamplerLaw.gaussian ({}/{}) ({cutoff}))",
                sigma.numerator(),
                sigma.denominator()
            ),
            Self::Interval { minimum, maximum } => {
                format!("(MxxRuntime.SamplerLaw.interval ({minimum}) ({maximum}))")
            }
            Self::Residue { modulus } => format!("(MxxRuntime.SamplerLaw.residue {modulus})"),
        }
    }
}

/// The parameters of one use of an assumption. Mirrors `MxxSecurity.AssumptionId.Params`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssumptionParams {
    DecisionRlwe {
        modulus: BigUint,
        ring_dimension: u32,
        secret: SamplerLaw,
        error: SamplerLaw,
        samples: usize,
    },
    RlweKeySwitchCircular {
        modulus: BigUint,
        extended_modulus: BigUint,
        ring_dimension: u32,
        secret: SamplerLaw,
        error: SamplerLaw,
        plaintext_modulus: BigUint,
        factors: Vec<BigUint>,
    },
}

impl AssumptionParams {
    pub fn id(&self) -> AssumptionId {
        match self {
            Self::DecisionRlwe { .. } => AssumptionId::DecisionRlwe,
            Self::RlweKeySwitchCircular { .. } => AssumptionId::RlweKeySwitchCircular,
        }
    }

    /// Checks the values every game needs: positive moduli and dimensions, a nonempty key.
    pub fn validate(&self) -> Result<(), SecurityError> {
        let positive = |value: &BigUint| *value > BigUint::from(1u8);
        let ok = match self {
            Self::DecisionRlwe { modulus, ring_dimension, samples, .. } => {
                positive(modulus) && *ring_dimension > 0 && *samples > 0
            }
            Self::RlweKeySwitchCircular {
                modulus,
                extended_modulus,
                ring_dimension,
                plaintext_modulus,
                factors,
                ..
            } => {
                positive(modulus) &&
                    positive(extended_modulus) &&
                    *ring_dimension > 0 &&
                    positive(plaintext_modulus) &&
                    !factors.is_empty()
            }
        };
        if ok { Ok(()) } else { Err(SecurityError::InvalidAssumption(self.id())) }
    }

    /// The Lean term of these parameters.
    pub fn lean(&self) -> String {
        match self {
            Self::DecisionRlwe { modulus, ring_dimension, secret, error, samples } => format!(
                "({{ modulus := {modulus}, ringDimension := {ring_dimension}, secretLaw := {}, errorLaw := {}, samples := {samples} }} : MxxSecurity.DecisionRlwe.Params)",
                secret.lean(),
                error.lean()
            ),
            Self::RlweKeySwitchCircular {
                modulus,
                extended_modulus,
                ring_dimension,
                secret,
                error,
                plaintext_modulus,
                factors,
            } => format!(
                "({{ modulus := {modulus}, extendedModulus := {extended_modulus}, ringDimension := {ring_dimension}, secretLaw := {}, errorLaw := {}, plaintextModulus := {plaintext_modulus}, factors := [{}] }} : MxxSecurity.RlweKeySwitchCircular.Params)",
                secret.lean(),
                error.lean(),
                factors.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum SecurityError {
    #[error("invalid parameters for assumption {0:?}")]
    InvalidAssumption(AssumptionId),
}
