//! The shared, scheme-independent RNS-BGG core.
//!
//! This module intentionally contains only the contracts shared by the stage
//! owners. Public-key preparation, encoding/encryption, and noise evaluation
//! live in their respective modules; this file does not provide a scheme
//! facade or an artifact/material wrapper hierarchy.

mod encoding;
#[cfg(all(test, feature = "gpu"))]
pub(crate) mod gpu;
mod noise;
mod public_key;

use bigdecimal::BigDecimal;
use mxx_dsl::{Family, Mat, Preimage, Ring};
use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::{
    noise_refresh::{
        prf::PrfError,
        refresh::{RefreshDeclaration, RefreshError},
        simulation::{RefreshNoiseParameters, RefreshRouteNoiseGains, RefreshTranscriptTopology},
    },
    program::{ExponentLUTProgramLowering, ExponentLutProgramId, WireId},
};

pub use encoding::{
    EncodingLowering, RnsBggEncoding, RnsBggMaskSecret, RnsBggPayloadSecret, RnsEncryptBuild,
    RnsOnlineBuild, checkpoint_refreshed_output, resume_refreshed_output, verify_output,
    verify_refreshed_output,
};
pub use noise::{
    RnsBggDerivedWorkload, RnsBggMaskSecretComponent, RnsBggNoiseBindings, RnsBggNoiseBound,
    RnsBggNoiseEvaluator, RnsBggNoiseFamily, RnsBggNoiseRhs, RnsBggNoiseTrace, RnsBggNoiseValue,
    RnsBggParameterManifest, RnsBggPlanRequest, RnsNoiseBuild,
};
pub use public_key::{PublicKeyLowering, RnsBggPublicKey, RnsPreprocessBuild};

/// A two-polynomial row block.
pub type RnsBggRow = (Mat, Mat);
/// A two-by-two polynomial block matrix.
pub type RnsBggMatrix = ((Mat, Mat), (Mat, Mat));
/// A two-by-two matrix of signed compact preimages.
pub(crate) type RnsBggDecomposition = ((Preimage, Preimage), (Preimage, Preimage));
fn refresh_commitment_hex(commitment: [u8; 32]) -> String {
    commitment.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Build the canonical per-slot refresh artifact path.
pub(crate) fn refresh_artifact_name(commitment: [u8; 32], slot: usize, role: &str) -> String {
    format!("refresh/{}/slot/{slot}/{role}", refresh_commitment_hex(commitment))
}

/// Build the canonical per-slot prepared-PRF artifact path.
pub(crate) fn refresh_prf_artifact_name(commitment: [u8; 32], slot: usize, role: &str) -> String {
    refresh_artifact_name(commitment, slot, &format!("prf/{role}"))
}

/// Build the canonical prepared-PRF input public-key artifact path.
pub(crate) fn refresh_prf_input_key_artifact_name(
    commitment: [u8; 32],
    slot: usize,
    wire: WireId,
    port: usize,
) -> String {
    refresh_prf_artifact_name(commitment, slot, &format!("input-key/{wire:?}.{port}"))
}

/// Build the canonical label-specific target-action artifact path.
///
/// Label actions are currently target actions; the fixed role is part of this
/// helper so producer and consumer cannot accidentally diverge.
pub(crate) fn refresh_label_artifact_name(
    commitment: [u8; 32],
    slot: usize,
    label: usize,
) -> String {
    refresh_artifact_name(commitment, slot, &format!("target/{label}"))
}

/// Build the canonical refresh-checkpoint artifact path.
pub(crate) fn refresh_checkpoint_artifact_name(commitment: [u8; 32], role: &str) -> String {
    format!("refresh/{}/{role}", refresh_commitment_hex(commitment))
}

/// Errors raised by the retained RNS-BGG contracts.
#[derive(Debug, Error)]
pub enum RnsBggError {
    #[error("invalid RNS-BGG parameters: {0}")]
    Parameters(&'static str),
    #[error("RNS-BGG budget exceeded: {0}")]
    Budget(&'static str),
    #[error("invalid RNS-BGG artifact or graph binding: {0}")]
    Binding(&'static str),
    #[error("RHS {rhs:?} is consumed at level {expected}, not {actual}")]
    RhsLevelMismatch { rhs: crate::program::RhsId, expected: usize, actual: usize },
    #[error(transparent)]
    Prf(#[from] PrfError),
    #[error(transparent)]
    Refresh(#[from] crate::noise_refresh::refresh::RefreshError),
    #[error("invalid RNS-BGG LUT")]
    Lut,
    #[error("RNS-BGG graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for RnsBggError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

fn canonical_encoding_sigma(sigma: BigDecimal) -> Result<BigDecimal, RnsBggError> {
    let sigma = sigma.normalized();
    let (_, exponent) = sigma.as_bigint_and_exponent();
    if sigma <= BigDecimal::zero() || exponent.unsigned_abs() > u64::from(u32::MAX) {
        return Err(RnsBggError::Parameters("encoding_error_sigma must be positive"));
    }
    Ok(sigma)
}

fn encoding_error_bound(sigma: &BigDecimal) -> Result<BigUint, RnsBggError> {
    let bound = std::panic::catch_unwind(|| {
        mxx_primitives::sampler::bounds::hard_cutoff_from_sigma_bound(sigma)
    })
    .map_err(|_| RnsBggError::Parameters("encoding_error_sigma is unrepresentable"))?;
    if bound.is_zero() {
        return Err(RnsBggError::Parameters("encoding_error_sigma derives a zero cutoff"));
    }
    Ok(bound)
}

/// Validated modulus-chain and encoding-noise parameters.
#[derive(Clone, Debug)]
pub struct RnsBggParameters {
    levels: Vec<DCRTPolyParams>,
    drops: Vec<BigUint>,
    encoding_error_sigma: BigDecimal,
    encoding_error_bound: BigUint,
    operations_per_drop: usize,
}

/// Validated source-to-target route for a public operand.
///
/// `source_level` and `target_level` describe the physical rings, while
/// `ratio` is the exact coefficient-modulus quotient used by scaled
/// ModDown.  A ratio of one is a compatible physical no-op; it is not an
/// instruction to insert a synthetic level transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RnsBggPublicAlignment {
    source_level: usize,
    target_level: usize,
    ratio: BigUint,
}

impl RnsBggPublicAlignment {
    pub(crate) fn ratio(&self) -> &BigUint {
        &self.ratio
    }

    pub(crate) fn is_noop(&self) -> bool {
        self.ratio == BigUint::one()
    }
}

impl RnsBggParameters {
    /// Validates the ring chain and computes every modulus-drop scale once.
    pub fn new(
        levels: Vec<DCRTPolyParams>,
        sigma: impl Into<BigDecimal>,
        operations_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        if levels.is_empty() {
            return Err(RnsBggError::Parameters("empty modulus chain"));
        }
        if operations_per_drop == 0 {
            return Err(RnsBggError::Parameters("operations_per_drop must be positive"));
        }
        let sigma = canonical_encoding_sigma(sigma.into())?;
        let encoding_error_bound = encoding_error_bound(&sigma)?;
        let first = &levels[0];
        if first.dropped_moduli() != 0 {
            return Err(RnsBggError::Parameters("partial modulus bases are not supported"));
        }
        if levels.iter().any(|level| {
            level.ring_dimension() != first.ring_dimension() ||
                level.base_bits() != first.base_bits() ||
                level.dropped_moduli() != 0
        }) {
            return Err(RnsBggError::Parameters("incompatible ring or partial basis"));
        }

        let mut drops = Vec::with_capacity(levels.len());
        for pair in levels.windows(2) {
            if pair[0].modulus() == pair[1].modulus() {
                if !compatible_level_parameters(&pair[0], &pair[1]) {
                    return Err(RnsBggError::Parameters(
                        "equal modulus levels have incompatible ring or gadget parameters",
                    ));
                }
                drops.push(BigUint::from(1u8));
                continue;
            }
            let source = pair[0].to_crt().0;
            let target = pair[1].to_crt().0;
            if target.len() >= source.len() || target.iter().any(|prime| !source.contains(prime)) {
                return Err(RnsBggError::Parameters(
                    "levels must be equal or strictly nested prime bases",
                ));
            }
            let source_modulus = pair[0].modulus();
            let target_modulus = pair[1].modulus();
            let quotient = source_modulus.as_ref() / target_modulus.as_ref();
            if &quotient * target_modulus.as_ref() != *source_modulus.as_ref() {
                return Err(RnsBggError::Parameters("modulus drop is not exact"));
            }
            drops.push(quotient);
        }
        drops.push(BigUint::from(1u8));
        Ok(Self {
            levels,
            drops,
            encoding_error_sigma: sigma,
            encoding_error_bound,
            operations_per_drop,
        })
    }

    pub fn levels(&self) -> &[DCRTPolyParams] {
        &self.levels
    }

    pub fn level(&self, level: usize) -> Result<&DCRTPolyParams, RnsBggError> {
        self.levels.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    pub fn scale(&self, level: usize) -> Result<&BigUint, RnsBggError> {
        self.drops.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    pub fn component_columns(&self, level: usize) -> Result<usize, RnsBggError> {
        self.level(level)?;
        Ok(self.levels.get(level + 1).map_or(1, PolyParams::modulus_digits))
    }

    /// Return the number of key columns for a wire with the supplied
    /// attribute knowledge.  Public attributes use the full gadget at their
    /// current modulus; private payloads use the next-level state gadget.
    /// Keeping this choice in the parameter contract prevents callers from
    /// inferring cryptographic role from a matrix shape.
    pub fn key_component_columns(
        &self,
        level: usize,
        knowledge: crate::program::AttributeKnowledge,
    ) -> Result<usize, RnsBggError> {
        let parameters = self.level(level)?;
        match knowledge {
            crate::program::AttributeKnowledge::PublicPayload => Ok(parameters.modulus_digits()),
            crate::program::AttributeKnowledge::PrivatePayload => self.component_columns(level),
        }
    }

    /// Validate one public source-to-target route.
    ///
    /// The source may be at a strictly larger modulus than the target, or
    /// both levels may describe the same compatible physical ring.  Upward
    /// conversion and unrelated CRT bases are rejected before any graph node
    /// is created.
    pub(crate) fn public_alignment(
        &self,
        source_level: usize,
        target_level: usize,
    ) -> Result<RnsBggPublicAlignment, RnsBggError> {
        let source = self.level(source_level)?;
        let target = self.level(target_level)?;
        if !compatible_level_parameters_except_basis(source, target) {
            return Err(RnsBggError::Parameters("incompatible public alignment levels"));
        }
        let source_modulus = source.modulus();
        let target_modulus = target.modulus();
        if source_modulus == target_modulus {
            if source.to_crt() != target.to_crt() {
                return Err(RnsBggError::Parameters(
                    "equal modulus levels have incompatible CRT bases",
                ));
            }
            return Ok(RnsBggPublicAlignment { source_level, target_level, ratio: BigUint::one() });
        }
        if source_modulus < target_modulus {
            return Err(RnsBggError::Parameters("public alignment cannot increase modulus"));
        }
        let source_basis = source.to_crt().0;
        let target_basis = target.to_crt().0;
        if target_basis.len() >= source_basis.len() ||
            target_basis.iter().any(|prime| !source_basis.contains(prime))
        {
            return Err(RnsBggError::Parameters("public alignment requires a nested CRT drop"));
        }
        let ratio = source_modulus.as_ref() / target_modulus.as_ref();
        if ratio.is_zero() || &ratio * target_modulus.as_ref() != *source_modulus.as_ref() {
            return Err(RnsBggError::Parameters("public alignment modulus ratio is not exact"));
        }
        Ok(RnsBggPublicAlignment { source_level, target_level, ratio })
    }

    /// Select the target level for `MulPublic`.
    ///
    /// The public LHS is aligned to the RHS-owned level when the RHS is
    /// private. For a public RHS, both carriers are aligned to the compatible
    /// lower physical modulus (and the larger logical index when equal).
    /// No upward conversion is accepted in either case.
    pub fn mul_public_target_level(
        &self,
        lhs_level: usize,
        rhs_level: usize,
        rhs_knowledge: crate::program::AttributeKnowledge,
    ) -> Result<usize, RnsBggError> {
        match rhs_knowledge {
            crate::program::AttributeKnowledge::PrivatePayload => {
                self.public_alignment(lhs_level, rhs_level)?;
                Ok(rhs_level)
            }
            crate::program::AttributeKnowledge::PublicPayload => {
                let lhs = self.level(lhs_level)?;
                let rhs = self.level(rhs_level)?;
                let target_level = match lhs.modulus().cmp(&rhs.modulus()) {
                    std::cmp::Ordering::Greater => rhs_level,
                    std::cmp::Ordering::Less => lhs_level,
                    std::cmp::Ordering::Equal => lhs_level.max(rhs_level),
                };
                self.public_alignment(lhs_level, target_level)?;
                self.public_alignment(rhs_level, target_level)?;
                Ok(target_level)
            }
        }
    }

    /// Construct the source-ring public alignment action
    /// `H = Dec_(G_source)(alpha * G_target)`.
    ///
    /// The returned preimage is shared by every component of a public key or
    /// encoding.  Equal compatible physical rings return `None`, which is a
    /// deliberate no-op and does not create a graph node.
    pub(crate) fn public_alignment_action(
        &self,
        source_level: usize,
        target_level: usize,
    ) -> Result<Option<Preimage>, RnsBggError> {
        let alignment = self.public_alignment(source_level, target_level)?;
        if alignment.is_noop() {
            return Ok(None);
        }
        let source = self.level(source_level)?;
        let source_modulus = BigInt::from(source.modulus().as_ref().clone());
        let target_gadget =
            self.public_gadget(target_level)?.centered_extend(source_modulus.clone());
        let alpha = self.scalar(source_level, BigInt::from(alignment.ratio().clone()))?;
        let scaled_target_gadget = alpha * target_gadget;
        let base = 1u64
            .checked_shl(source.base_bits())
            .ok_or(RnsBggError::Parameters("alignment decomposition base overflow"))?;
        Ok(Some(scaled_target_gadget.decompose(base, source.modulus_digits())))
    }

    pub fn encoding_error_sigma(&self) -> &BigDecimal {
        &self.encoding_error_sigma
    }

    pub fn encoding_error_bound(&self) -> &BigUint {
        &self.encoding_error_bound
    }

    pub fn operations_per_drop(&self) -> usize {
        self.operations_per_drop
    }

    pub fn refresh_residue_parameters(
        &self,
        level: usize,
    ) -> Result<Vec<DCRTPolyParams>, RnsBggError> {
        let parameters = self.level(level)?;
        parameters
            .to_crt()
            .0
            .into_iter()
            .map(|prime| {
                parameters
                    .select_modulus(&BigUint::from(prime))
                    .ok_or(RnsBggError::Parameters("refresh residue is outside the CRT basis"))
            })
            .collect()
    }

    /// Validate the workload-owned refresh level contract.  The source wire
    /// determines the physical route: its level is `q`, the immediate next
    /// level is `p`, and the following level is the CRT-recomposed target.
    /// This check belongs at production refresh dispatch because deriving a
    /// noise snapshot alone has no workload from which to resolve the source.
    pub(crate) fn validate_refresh_level_contract(
        &self,
        declaration: &RefreshDeclaration,
        source_level: usize,
    ) -> Result<(), RnsBggError> {
        let p_level = source_level
            .checked_add(1)
            .ok_or(RnsBggError::Binding("refresh source level has no q/p/target chain"))?;
        let target_level = source_level
            .checked_add(2)
            .ok_or(RnsBggError::Binding("refresh source level has no q/p/target chain"))?;
        let source_parameters = self
            .level(source_level)
            .map_err(|_| RnsBggError::Binding("refresh source level is not configured"))?;
        let p_parameters = self
            .level(p_level)
            .map_err(|_| RnsBggError::Binding("refresh p level is not configured"))?;
        let target_parameters = self
            .level(target_level)
            .map_err(|_| RnsBggError::Binding("refresh target level is not configured"))?;
        if declaration.target_level() != target_level {
            return Err(RnsBggError::Binding("refresh target level is not source+2"));
        }

        let q = source_parameters.modulus().as_ref().clone();
        let p = p_parameters.modulus().as_ref().clone();
        let target_modulus = target_parameters.modulus().as_ref().clone();
        let mut target_modulus_product = BigUint::one();
        for slot in declaration.slots() {
            let declared_q = slot
                .q
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let declared_p = slot
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let qt = slot
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            if declared_q != q || declared_p != p {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            if declared_q <= declared_p ||
                &declared_q % &declared_p != BigUint::zero() ||
                declared_p < qt ||
                &declared_p % &qt != BigUint::zero()
            {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            target_modulus_product *= qt;
        }
        if target_modulus_product != target_modulus {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        Ok(())
    }

    /// Derive the refresh noise contract from the actual RNS-BGG setup.
    ///
    /// Every prepared refresh action is decomposed in its slot's `p` ring.
    /// For an evaluation row with `N` ring coordinates, `D_qp` input/decomposition
    /// columns, and radix `B`, the conservative coefficient gain is
    /// `K1 = N*D_qp*B`; the squared-L2 envelope is `K1²`. Label weights are
    /// split using their canonical positional sums, so the evaluator can apply
    /// each label's digit/`H` weight exactly once. The decoder gain separately
    /// envelopes the two `N`-coefficient mask-secret convolutions. The returned
    /// bias also includes the bounded fresh `eta` sampled by every prepared
    /// label/data action and the canonical positive-digit offsets.
    pub fn refresh_noise_parameters(
        &self,
        declaration: &RefreshDeclaration,
    ) -> Result<RefreshNoiseParameters, RnsBggError> {
        if declaration.slots().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let target_level = declaration.target_level();
        let target_columns = self.component_columns(target_level)?;
        let coefficient_count = self.level(target_level)?.ring_dimension() as usize;
        let target_modulus = self.level(target_level)?.modulus().as_ref().clone();
        let base = BigUint::from(declaration.digits().base);
        let mask_counts = declaration.digits().mask_counts.clone();
        let fresh_count = declaration.digits().fresh_count;
        if mask_counts.len() != declaration.slots().len() || target_columns == 0 {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }

        let mut global_modulus = None;
        let mut plaintext_moduli = Vec::with_capacity(declaration.slots().len());
        let mut switch_moduli = Vec::with_capacity(declaration.slots().len());
        let mut state_scaling = Vec::with_capacity(declaration.slots().len());
        let mut mask = Vec::with_capacity(declaration.slots().len());
        let mut fresh = Vec::with_capacity(declaration.slots().len());
        let mut decoder = Vec::with_capacity(declaration.slots().len());
        let mut route_bias_bounds = Vec::with_capacity(declaration.slots().len());
        let coordinate_count = BigUint::from(target_columns) * BigUint::from(coefficient_count);
        let mut target_modulus_product = BigUint::one();

        for (slot, metadata) in declaration.slots().iter().enumerate() {
            let q = metadata
                .q
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let p = metadata
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let qt = metadata
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            if q <= p || &q % &p != BigUint::zero() || p < qt || &p % &qt != BigUint::zero() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
            }
            if let Some(expected) = &global_modulus {
                if expected != &q {
                    return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
                }
            } else {
                global_modulus = Some(q.clone());
            }
            let q_level = self
                .levels
                .iter()
                .position(|level| level.modulus().as_ref() == &q)
                .ok_or(RnsBggError::Binding("refresh q modulus is not configured"))?;
            let p_level = q_level
                .checked_add(1)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let p_parameters = self.level(p_level).map_err(|_| {
                RnsBggError::Binding("refresh p modulus is not the immediate q state level")
            })?;
            if p_parameters.modulus().as_ref() != &p {
                return Err(RnsBggError::Binding(
                    "refresh p modulus is not the immediate q state level",
                ));
            }
            let decomposition_width = p_parameters.modulus_digits();
            if self.component_columns(q_level)? != decomposition_width {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
            let decomposition_base = BigUint::one() << p_parameters.base_bits();
            let k1 = BigUint::from(p_parameters.ring_dimension()) *
                BigUint::from(decomposition_width) *
                &decomposition_base;
            let k2_squared = k1.pow(2u32);
            let single_action_gain = crate::noise_utils::RightActionNoiseGain {
                l1: k1.clone(),
                l2_squared: k2_squared.clone(),
            };
            state_scaling.push(single_action_gain);
            let decoder_l1 = BigUint::from(p_parameters.ring_dimension()) * 2u8;
            decoder.push(crate::noise_utils::RightActionNoiseGain {
                l2_squared: decoder_l1.pow(2u32),
                l1: decoder_l1,
            });

            let mask_weights = refresh_noise_weight_sums(
                &base,
                mask_counts[slot],
                &coordinate_count,
                &BigUint::one(),
            );
            let fresh_weights = refresh_noise_weight_sums(
                &base,
                fresh_count,
                &coordinate_count,
                &(p.clone() / &qt),
            );
            // The evaluator splits each aggregate route gain by these exact
            // sums before applying the individual label's positional
            // weight. Every independently prepared dense action must retain
            // the full single-action K1 envelope, so persist the weighted
            // aggregate rather than K1 itself.
            let mask_aggregate_l1 = &k1 * &mask_weights.0;
            let fresh_aggregate_l1 = &k1 * &fresh_weights.0;
            mask.push(crate::noise_utils::RightActionNoiseGain {
                l1: mask_aggregate_l1.clone(),
                l2_squared: &k2_squared * &mask_weights.1,
            });
            fresh.push(crate::noise_utils::RightActionNoiseGain {
                l1: fresh_aggregate_l1.clone(),
                l2_squared: &k2_squared * &fresh_weights.1,
            });
            let mask_unit_gain = refresh_noise_ceil_div(&mask_aggregate_l1, &mask_weights.0);
            let fresh_unit_gain = refresh_noise_ceil_div(&fresh_aggregate_l1, &fresh_weights.0);
            let eta_bound = self.encoding_error_bound();
            let label_eta = eta_bound *
                (&mask_weights.0 * &mask_unit_gain + &fresh_weights.0 * &fresh_unit_gain);
            let data_eta = eta_bound * &k1;
            let q_to_p = &q / &p;
            let q_to_qt = &q / &qt;
            let mask_offset = crate::noise_refresh::refresh::canonical_digit_offset(
                &base,
                mask_counts[slot],
                true,
            )
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidLayout))?
            .magnitude() *
                &q_to_p *
                &coordinate_count *
                mask_unit_gain;
            let fresh_offset =
                crate::noise_refresh::refresh::canonical_digit_offset(&base, fresh_count, false)
                    .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidLayout))?
                    .magnitude() *
                    &q_to_qt *
                    &coordinate_count *
                    fresh_unit_gain;
            plaintext_moduli.push(qt);
            switch_moduli.push(p);
            target_modulus_product *= &plaintext_moduli[slot];
            route_bias_bounds.push(label_eta + data_eta + mask_offset + fresh_offset);
        }

        if target_modulus_product != target_modulus {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }

        let topology = RefreshTranscriptTopology::new(
            target_columns,
            coefficient_count,
            mask_counts.clone(),
            fresh_count,
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let mut parameters = RefreshNoiseParameters::new(
            plaintext_moduli,
            switch_moduli,
            base,
            mask_counts,
            fresh_count,
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?
        .with_global_modulus(
            global_modulus.ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?,
        )
        .map_err(|error| RnsBggError::Graph(error.to_string()))?
        .with_transcript_topology(topology)
        .map_err(|error| RnsBggError::Graph(error.to_string()))?
        .with_route_gains(RefreshRouteNoiseGains { state_scaling, mask, fresh, decoder })
        .map_err(|error| RnsBggError::Graph(error.to_string()))?;
        parameters.route_bias_bounds = route_bias_bounds;
        parameters.validate().map_err(|error| RnsBggError::Graph(error.to_string()))?;
        Ok(parameters)
    }

    pub fn ring(&self, level: usize) -> Result<Ring, RnsBggError> {
        let parameters = self.level(level)?;
        Ok(Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        ))
    }

    pub fn scalar(&self, level: usize, value: impl Into<BigInt>) -> Result<Mat, RnsBggError> {
        Ok(self.ring(level)?.polynomial([value.into().into()]))
    }

    /// Stable identity of the validated parameter contract.
    pub(crate) fn parameter_identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-rns-bgg-parameters/v1");
        digest.update(self.encoding_error_sigma.normalized().to_string().as_bytes());
        digest.update(self.encoding_error_bound.to_bytes_be());
        digest.update((self.operations_per_drop as u64).to_le_bytes());
        for level in &self.levels {
            digest.update(level.ring_dimension().to_le_bytes());
            digest.update((level.base_bits() as u64).to_le_bytes());
            digest.update(level.modulus().to_bytes_be());
            for prime in level.to_crt().0 {
                digest.update(prime.to_le_bytes());
            }
        }
        for drop in &self.drops {
            digest.update(drop.to_bytes_be());
        }
        digest.finalize().into()
    }

    /// Validate the exact four-level schedule required by a prepared PRF
    /// child: `[q, q, q, p]`.  The final successor supplies the coefficient
    /// decomposition width for the level-two q carrier.
    pub(crate) fn validate_refresh_prf_schedule(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<(), RnsBggError> {
        if self.levels.len() != 4 || slot >= declaration.slots().len() {
            return Err(RnsBggError::Binding("refresh PRF parameter schedule is not [q,q,q,p]"));
        }
        let declared_slot = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let q = declared_slot
            .q
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let p = declared_slot
            .p
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        if self.level(0)?.modulus().as_ref() != &q ||
            self.level(1)?.modulus().as_ref() != &q ||
            self.level(2)?.modulus().as_ref() != &q ||
            self.level(3)?.modulus().as_ref() != &p ||
            self.level(1)? != self.level(0)? ||
            self.level(2)? != self.level(0)?
        {
            return Err(RnsBggError::Binding("refresh PRF parameter schedule is not [q,q,q,p]"));
        }
        if self.component_columns(2)? != self.level(3)?.modulus_digits() {
            return Err(RnsBggError::Binding(
                "refresh PRF coefficient output width does not match successor level",
            ));
        }
        Ok(())
    }

    /// Derive the exact child schedule from the outer source chain and return
    /// its full parameter identity for retained child/artifact validation.
    pub(crate) fn expected_refresh_prf_parameter_identity(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<[u8; 32], RnsBggError> {
        let declared_slot = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let q = declared_slot
            .q
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let p = declared_slot
            .p
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let q_level = (0..self.levels.len().saturating_sub(1))
            .find(|level| {
                self.levels[*level].modulus().as_ref() == &q &&
                    self.levels[*level + 1].modulus().as_ref() == &p
            })
            .ok_or(RnsBggError::Binding("refresh q/p successor is not configured"))?;
        let q_parameters = self.level(q_level)?.clone();
        let p_parameters = self.level(q_level + 1)?.clone();
        let expected = Self::new(
            vec![
                q_parameters.clone(),
                q_parameters.clone(),
                q_parameters.clone(),
                p_parameters.clone(),
            ],
            self.encoding_error_sigma.clone(),
            self.operations_per_drop,
        )?;
        if expected.component_columns(2)? != self.component_columns(q_level)? {
            return Err(RnsBggError::Binding(
                "refresh PRF coefficient output width does not match source q carrier",
            ));
        }
        Ok(expected.parameter_identity())
    }

    pub(crate) fn state_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let high = self.level(level)?;
        match self.levels.get(level + 1) {
            Some(low) => {
                let base = 1u64
                    .checked_shl(low.base_bits())
                    .ok_or(RnsBggError::Parameters("state gadget base overflow"))?;
                Ok(self
                    .ring(level + 1)?
                    .gadget(1, base, low.modulus_digits())
                    .centered_extend(BigInt::from(high.modulus().as_ref().clone())))
            }
            None => Ok(self.ring(level)?.identity(1)),
        }
    }

    pub(crate) fn public_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        Ok(self.ring(level)?.gadget(1, base, parameters.modulus_digits()))
    }
}

fn refresh_noise_ceil_div(numerator: &BigUint, denominator: &BigUint) -> BigUint {
    if numerator.is_zero() {
        BigUint::zero()
    } else {
        (numerator + denominator - BigUint::one()) / denominator
    }
}

fn refresh_noise_weight_sums(
    base: &BigUint,
    count: usize,
    coordinate_count: &BigUint,
    scalar: &BigUint,
) -> (BigUint, BigUint) {
    let mut l1 = BigUint::zero();
    let mut l2_squared = BigUint::zero();
    let mut weight = scalar.clone();
    for _ in 0..count {
        l1 += &weight;
        l2_squared += weight.pow(2u32);
        weight *= base;
    }
    l1 = &l1 * coordinate_count;
    l2_squared = &l2_squared * coordinate_count;
    (l1, l2_squared)
}

fn public_gadget_base(base_bits: u32) -> Result<u64, RnsBggError> {
    1u64.checked_shl(base_bits).ok_or(RnsBggError::Parameters("public gadget base overflow"))
}

/// Check the metadata that must remain identical when no physical modulus
/// conversion is performed.  Equal numeric moduli alone are insufficient:
/// gadget decomposition and ring coordinates must describe the same basis.
fn compatible_level_parameters(lhs: &DCRTPolyParams, rhs: &DCRTPolyParams) -> bool {
    compatible_level_parameters_except_basis(lhs, rhs) && lhs.to_crt() == rhs.to_crt()
}

fn compatible_level_parameters_except_basis(lhs: &DCRTPolyParams, rhs: &DCRTPolyParams) -> bool {
    lhs.ring_dimension() == rhs.ring_dimension() &&
        lhs.base_bits() == rhs.base_bits() &&
        lhs.dropped_moduli() == rhs.dropped_moduli()
}

/// Map a logical operation level to the CRT depth selected by a drop cadence.
/// Logical chain slots remain present; adjacent slots in one cadence window
/// therefore intentionally carry equal physical moduli.
pub(crate) fn planned_crt_depth(
    logical_level: usize,
    operations_per_drop: usize,
    max_crt_depth: usize,
    terminal_depth: usize,
) -> Result<usize, RnsBggError> {
    if operations_per_drop == 0 || terminal_depth == 0 || max_crt_depth < terminal_depth {
        return Err(RnsBggError::Parameters("invalid CRT drop cadence"));
    }
    let physical_drop = logical_level / operations_per_drop;
    Ok(max_crt_depth.saturating_sub(physical_drop).max(terminal_depth))
}

/// Return the reusable LUT mask-helper keys required by the encryption
/// projection.  These keys are deliberately derived from the complete LUT
/// vector rather than from the Program body: an unreferenced registered LUT
/// still describes part of the input contract, and every nonterminal logical
/// level needs the same deterministic public mask matrix when it can host a
/// LUT operation.  Logical levels remain part of the key even when adjacent
/// levels have equal physical moduli.
pub(crate) fn canonical_lut_helper_keys(
    parameters: &RnsBggParameters,
    inputs: &crate::program::EncryptionInputs,
) -> Result<Vec<(usize, usize)>, RnsBggError> {
    let mut keys = std::collections::BTreeSet::new();
    let nonterminal_levels = parameters.levels().len().saturating_sub(1);
    for level in 0..nonterminal_levels {
        let ring_dimension = parameters.level(level)?.ring_dimension() as usize;
        for table in inputs.luts() {
            let layout = table.input_layout();
            layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
            for sigma in
                layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?
            {
                keys.insert((level, sigma));
            }
        }
    }
    Ok(keys.into_iter().collect())
}

fn update_projection_json<T: Serialize>(digest: &mut Sha256, value: &T) -> Result<(), RnsBggError> {
    let encoded = serde_json::to_vec(value)
        .map_err(|_| RnsBggError::Binding("encryption projection is not serializable"))?;
    digest.update((encoded.len() as u64).to_le_bytes());
    digest.update(encoded);
    Ok(())
}

fn resolved_input_cardinality(
    shape: &crate::program::InputShape,
    parameters: &mxx_ir_core::ParamEnv,
) -> Result<Option<usize>, RnsBggError> {
    match shape {
        crate::program::InputBinding::Scalar(()) => Ok(None),
        crate::program::InputBinding::Family(count) => count
            .evaluate(parameters)
            .ok()
            .and_then(|value| value.to_usize())
            .filter(|value| *value > 0)
            .map(Some)
            .ok_or(RnsBggError::Binding("encryption input family count is not concrete")),
    }
}

/// Hash the exact projection and resolved invocation cardinalities consumed by
/// encryption.  This intentionally does not include the Program identifier,
/// computation body, or gate schedule.
pub(crate) fn encryption_projection_identity(
    inputs: &crate::program::EncryptionInputs,
    parameters: &mxx_ir_core::ParamEnv,
    rns_parameters: &RnsBggParameters,
) -> Result<[u8; 32], RnsBggError> {
    let mut digest = Sha256::new();
    digest.update(b"mxx-rns-bgg/encryption-projection/v1");
    for (wire, declaration) in inputs.inputs() {
        digest.update(wire.0.to_le_bytes());
        digest.update([match declaration.knowledge() {
            crate::program::AttributeKnowledge::PrivatePayload => 0,
            crate::program::AttributeKnowledge::PublicPayload => 1,
        }]);
        update_projection_json(&mut digest, declaration.shape())?;
        match resolved_input_cardinality(declaration.shape(), parameters)? {
            Some(cardinality) => digest.update(cardinality.to_le_bytes()),
            None => digest.update([0xff; std::mem::size_of::<usize>()]),
        }
    }
    digest.update([0xfe]);
    for (index, table) in inputs.luts().iter().enumerate() {
        digest.update((index as u64).to_le_bytes());
        update_projection_json(&mut digest, table)?;
    }
    digest.update([0xfd]);
    for (level, sigma) in canonical_lut_helper_keys(rns_parameters, inputs)? {
        digest.update((level as u64).to_le_bytes());
        digest.update((sigma as u64).to_le_bytes());
    }
    Ok(digest.finalize().into())
}

/// The sole descriptor shared by RNS-BGG stage checkpoints.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct RnsBggArtifactBinding {
    context: [u8; 32],
    program: ExponentLutProgramId,
    parameters: [u8; 32],
    hash_key_commitment: [u8; 32],
    sources: Vec<mxx_ir_core::artifact::ProductionId>,
    refresh_commitment: Option<[u8; 32]>,
}

impl RnsBggArtifactBinding {
    pub(crate) fn new(
        context: [u8; 32],
        program: ExponentLutProgramId,
        parameters: [u8; 32],
        hash_key_commitment: [u8; 32],
        sources: Vec<mxx_ir_core::artifact::ProductionId>,
    ) -> Result<Self, RnsBggError> {
        if sources.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RnsBggError::Binding("artifact sources must be canonical and unique"));
        }
        let mut binding = Self {
            context,
            program,
            parameters,
            hash_key_commitment,
            sources: Vec::new(),
            refresh_commitment: None,
        };
        for source in sources {
            binding = binding.with_source(source)?;
        }
        Ok(binding)
    }

    /// Binds a refresh declaration to this stage's artifact identity.
    pub(crate) fn with_refresh_declaration(mut self, declaration: &RefreshDeclaration) -> Self {
        self.refresh_commitment = Some(*declaration.commitment());
        self
    }

    /// Attach the producing checkpoint once execution assigns its nonce.
    pub(crate) fn with_source(
        mut self,
        source: mxx_ir_core::artifact::ProductionId,
    ) -> Result<Self, RnsBggError> {
        if self.sources.binary_search(&source).is_ok() {
            return Err(RnsBggError::Binding("artifact sources must be unique"));
        }
        self.sources.push(source);
        self.sources.sort();
        Ok(self)
    }

    pub(crate) fn context(&self) -> &[u8; 32] {
        &self.context
    }

    pub(crate) fn program(&self) -> ExponentLutProgramId {
        self.program
    }

    pub(crate) fn parameters(&self) -> &[u8; 32] {
        &self.parameters
    }

    pub(crate) fn hash_key_commitment(&self) -> &[u8; 32] {
        &self.hash_key_commitment
    }

    pub(crate) fn sources(&self) -> &[mxx_ir_core::artifact::ProductionId] {
        &self.sources
    }

    pub(crate) fn refresh_commitment(&self) -> Option<&[u8; 32]> {
        self.refresh_commitment.as_ref()
    }
}

/// A stage descriptor is deliberately distinct at the type level even though
/// all three RNS-BGG checkpoints carry the same binding metadata.  The
/// transparent wrapper keeps the serialized representation owned by the
/// binding while preventing a preprocess, encryption, or online checkpoint
/// from being passed to the wrong stage API by accident.
macro_rules! rns_bgg_stage_descriptor {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(transparent)]
        #[doc = "Typed RNS-BGG stage artifact descriptor."]
        pub struct $name(RnsBggArtifactBinding);

        impl $name {
            pub(crate) fn from_binding(binding: RnsBggArtifactBinding) -> Self {
                Self(binding)
            }

            pub(crate) fn binding(&self) -> &RnsBggArtifactBinding {
                &self.0
            }
        }
    };
}

rns_bgg_stage_descriptor!(RnsBggPreprocessEncryptDescriptor);

rns_bgg_stage_descriptor!(RnsBggPreprocessOnlineDescriptor);

rns_bgg_stage_descriptor!(RnsBggEncryptionDescriptor);

rns_bgg_stage_descriptor!(RnsBggRefreshCheckpointDescriptor);

impl RnsBggPreprocessEncryptDescriptor {
    pub(crate) fn with_source(
        self,
        source: mxx_ir_core::artifact::ProductionId,
    ) -> Result<Self, RnsBggError> {
        Ok(Self::from_binding(self.0.with_source(source)?))
    }
}

pub(crate) fn hash_key_commitment(hash_key: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-rns-bgg/hash-key/v1");
    digest.update(hash_key);
    digest.finalize().into()
}

/// Small, one-shot refresh invocation shared by the three stage lowerers.
pub struct RnsBggRefresh<'a, E: ExponentLUTProgramLowering> {
    lowerer: &'a mut E,
    inputs: BTreeMap<usize, E::Wire>,
}

impl<'a, E: ExponentLUTProgramLowering> RnsBggRefresh<'a, E> {
    pub fn new(lowerer: &'a mut E, inputs: BTreeMap<usize, E::Wire>) -> Self {
        // RefreshDeclaration owns and validates all immutable refresh
        // metadata. Stage implementations receive it explicitly in
        // `finish_noise_refresh`; this runtime carrier stores only lowering
        // state and concrete inputs.
        Self { lowerer, inputs }
    }

    pub(crate) fn lowerer(&mut self) -> &mut E {
        self.lowerer
    }

    pub(crate) fn inputs(&self) -> &BTreeMap<usize, E::Wire> {
        &self.inputs
    }
}

fn map_row(row: RnsBggRow, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggRow {
    (operation(row.0), operation(row.1))
}

fn map_matrix(matrix: RnsBggMatrix, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggMatrix {
    ((operation(matrix.0.0), operation(matrix.0.1)), (operation(matrix.1.0), operation(matrix.1.1)))
}

/// Convert a matrix between concrete CRT rings without constructing an
/// invalid modulus-reduction node.  Reduction is only valid when the source
/// ring is larger; extending is used for a larger destination and equal rings
/// are preserved as-is.
pub(crate) fn checked_modulus_convert(
    value: Mat,
    destination: impl Into<BigInt>,
) -> Result<Mat, RnsBggError> {
    let destination = destination.into();
    if destination <= BigInt::zero() {
        return Err(RnsBggError::Parameters("destination modulus must be positive"));
    }
    let source = value
        .matrix_type()
        .modulus
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| RnsBggError::Parameters("source modulus must be concrete"))?;
    if source <= BigInt::zero() {
        return Err(RnsBggError::Parameters("source modulus must be positive"));
    }
    match source.cmp(&destination) {
        std::cmp::Ordering::Greater => {
            if &source % &destination != BigInt::zero() {
                return Err(RnsBggError::Parameters("destination modulus must divide source"));
            }
            Ok(value.reduce_modulus(destination))
        }
        std::cmp::Ordering::Equal => Ok(value),
        std::cmp::Ordering::Less => Ok(value.centered_extend(destination)),
    }
}

pub(crate) fn checked_modulus_convert_row(
    row: RnsBggRow,
    destination: impl Into<BigInt> + Clone,
) -> Result<RnsBggRow, RnsBggError> {
    let destination = destination.into();
    Ok((
        checked_modulus_convert(row.0, destination.clone())?,
        checked_modulus_convert(row.1, destination)?,
    ))
}

pub(crate) fn checked_modulus_convert_matrix(
    matrix: RnsBggMatrix,
    destination: impl Into<BigInt> + Clone,
) -> Result<RnsBggMatrix, RnsBggError> {
    let destination = destination.into();
    Ok((
        (
            checked_modulus_convert(matrix.0.0, destination.clone())?,
            checked_modulus_convert(matrix.0.1, destination.clone())?,
        ),
        (
            checked_modulus_convert(matrix.1.0, destination.clone())?,
            checked_modulus_convert(matrix.1.1, destination)?,
        ),
    ))
}

pub(crate) fn add_public_matrices(lhs: RnsBggMatrix, rhs: RnsBggMatrix) -> RnsBggMatrix {
    ((lhs.0.0 + rhs.0.0, lhs.0.1 + rhs.0.1), (lhs.1.0 + rhs.1.0, lhs.1.1 + rhs.1.1))
}

pub(crate) fn add_encoding_rows(lhs: RnsBggRow, rhs: RnsBggRow) -> RnsBggRow {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

/// Decompose a compact right-action matrix at its declared decomposition
/// level and transport only the resulting bounded digits to the evaluation
/// ring.  The dense matrix is never extended to the evaluation modulus.
pub(crate) fn compact_decompose_matrix(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    matrix: RnsBggMatrix,
) -> Result<RnsBggDecomposition, RnsBggError> {
    let evaluation_modulus = parameters.level(evaluation_level)?.modulus().as_ref().clone();
    let decomposition = parameters.level(decomposition_level)?;
    let decomposition_modulus = decomposition.modulus().as_ref().clone();
    let expected_dimension = BigInt::from(decomposition.ring_dimension());
    for entry in [&matrix.0.0, &matrix.0.1, &matrix.1.0, &matrix.1.1] {
        let matrix_type = entry.matrix_type();
        let modulus = matrix_type
            .modulus
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .map_err(|_| RnsBggError::Parameters("decomposition modulus must be concrete"))?;
        let dimension =
            matrix_type.ring_dimension.evaluate(&mxx_ir_core::ParamEnv::default()).map_err(
                |_| RnsBggError::Parameters("decomposition ring dimension must be concrete"),
            )?;
        if modulus != BigInt::from(decomposition_modulus.clone()) || dimension != expected_dimension
        {
            return Err(RnsBggError::Parameters(
                "compact action matrix does not match its decomposition ring",
            ));
        }
    }
    if evaluation_modulus < decomposition_modulus {
        return Err(RnsBggError::Parameters(
            "evaluation modulus must contain the decomposition modulus",
        ));
    }
    let base = 1u64
        .checked_shl(decomposition.base_bits())
        .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
    let decomposition_count = decomposition.modulus_digits();
    let value = (
        (
            matrix.0.0.decompose(base, decomposition_count),
            matrix.0.1.decompose(base, decomposition_count),
        ),
        (
            matrix.1.0.decompose(base, decomposition_count),
            matrix.1.1.decompose(base, decomposition_count),
        ),
    );
    if evaluation_modulus == decomposition_modulus {
        Ok(value)
    } else {
        Ok(centered_extend_compact(value, BigInt::from(evaluation_modulus)))
    }
}

fn centered_extend_compact(value: RnsBggDecomposition, modulus: BigInt) -> RnsBggDecomposition {
    (
        (value.0.0.centered_extend(modulus.clone()), value.0.1.centered_extend(modulus.clone())),
        (value.1.0.centered_extend(modulus.clone()), value.1.1.centered_extend(modulus)),
    )
}

/// Apply a compact right action to an encoding row.
pub(crate) fn compact_right_action_row(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    row: RnsBggRow,
    matrix: RnsBggMatrix,
) -> Result<RnsBggRow, RnsBggError> {
    let decomposition =
        compact_decompose_matrix(parameters, evaluation_level, decomposition_level, matrix)?;
    Ok(multiply_encoding_action(row, decomposition))
}

/// Apply a compact right action to a public matrix.
pub(crate) fn compact_right_action_public(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    matrix: RnsBggMatrix,
    action: RnsBggMatrix,
) -> Result<RnsBggMatrix, RnsBggError> {
    let decomposition =
        compact_decompose_matrix(parameters, evaluation_level, decomposition_level, action)?;
    Ok(multiply_public_action(matrix, decomposition))
}

fn validate_public_alignment_entry(
    parameters: &RnsBggParameters,
    source_level: usize,
    entry: &Mat,
) -> Result<(), RnsBggError> {
    let expected = parameters.ring(source_level)?.matrix_type((
        1,
        parameters.key_component_columns(
            source_level,
            crate::program::AttributeKnowledge::PublicPayload,
        )?,
    ));
    if entry.matrix_type() == &expected {
        Ok(())
    } else {
        Err(RnsBggError::Binding("public alignment operand shape mismatch"))
    }
}

/// Apply one shared public alignment action to both encoding-row components.
pub(crate) fn apply_public_alignment_row(
    parameters: &RnsBggParameters,
    source_level: usize,
    target_level: usize,
    row: RnsBggRow,
) -> Result<RnsBggRow, RnsBggError> {
    validate_public_alignment_entry(parameters, source_level, &row.0)?;
    validate_public_alignment_entry(parameters, source_level, &row.1)?;
    let Some(action) = parameters.public_alignment_action(source_level, target_level)? else {
        return Ok(row);
    };
    let aligned = (action.clone().mul_small_rhs(row.0), action.mul_small_rhs(row.1));
    let target_modulus = BigInt::from(parameters.level(target_level)?.modulus().as_ref().clone());
    checked_mod_down_encoding(aligned, target_modulus)
}

/// Apply one shared public alignment action to every component of a public
/// matrix.  The same `H` is cloned only as an immutable graph handle; no
/// component receives a separately generated action.
pub(crate) fn apply_public_alignment_matrix(
    parameters: &RnsBggParameters,
    source_level: usize,
    target_level: usize,
    matrix: RnsBggMatrix,
) -> Result<RnsBggMatrix, RnsBggError> {
    validate_public_alignment_entry(parameters, source_level, &matrix.0.0)?;
    validate_public_alignment_entry(parameters, source_level, &matrix.0.1)?;
    validate_public_alignment_entry(parameters, source_level, &matrix.1.0)?;
    validate_public_alignment_entry(parameters, source_level, &matrix.1.1)?;
    let Some(action) = parameters.public_alignment_action(source_level, target_level)? else {
        return Ok(matrix);
    };
    let aligned = (
        (action.clone().mul_small_rhs(matrix.0.0), action.clone().mul_small_rhs(matrix.0.1)),
        (action.clone().mul_small_rhs(matrix.1.0), action.mul_small_rhs(matrix.1.1)),
    );
    let target_modulus = BigInt::from(parameters.level(target_level)?.modulus().as_ref().clone());
    checked_mod_down_public(aligned, target_modulus)
}

pub(crate) fn multiply_public_action(lhs: RnsBggMatrix, rhs: RnsBggDecomposition) -> RnsBggMatrix {
    let ((a, b), (c, d)) = lhs;
    let ((u, v), (w, x)) = rhs;
    (
        (
            u.clone().mul_small_rhs(a.clone()) + w.clone().mul_small_rhs(b.clone()),
            v.clone().mul_small_rhs(a.clone()) + x.clone().mul_small_rhs(b.clone()),
        ),
        (
            u.mul_small_rhs(c.clone()) + w.mul_small_rhs(d.clone()),
            v.mul_small_rhs(c) + x.mul_small_rhs(d),
        ),
    )
}

pub(crate) fn multiply_encoding_action(lhs: RnsBggRow, rhs: RnsBggDecomposition) -> RnsBggRow {
    let (a, b) = lhs;
    let ((u, v), (w, x)) = rhs;
    (
        u.mul_small_rhs(a.clone()) + w.mul_small_rhs(b.clone()),
        v.mul_small_rhs(a) + x.mul_small_rhs(b),
    )
}

fn concrete_entry_modulus(entry: &Mat) -> Result<BigInt, RnsBggError> {
    entry
        .matrix_type()
        .modulus
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| RnsBggError::Parameters("source modulus must be concrete"))
}

fn validate_mod_down_entry(entry: &Mat, modulus: &BigInt) -> Result<bool, RnsBggError> {
    if modulus <= &BigInt::zero() {
        return Err(RnsBggError::Parameters("destination modulus must be positive"));
    }
    let source = concrete_entry_modulus(entry)?;
    match source.cmp(modulus) {
        std::cmp::Ordering::Less => {
            Err(RnsBggError::Parameters("modulus switch cannot increase modulus"))
        }
        std::cmp::Ordering::Equal => Ok(true),
        std::cmp::Ordering::Greater => {
            if &source % modulus != BigInt::zero() {
                return Err(RnsBggError::Parameters("destination modulus must divide source"));
            }
            Ok(false)
        }
    }
}

fn checked_mod_down_matrix(
    value: RnsBggMatrix,
    modulus: BigInt,
) -> Result<RnsBggMatrix, RnsBggError> {
    let entries = [&value.0.0, &value.0.1, &value.1.0, &value.1.1];
    let source = concrete_entry_modulus(entries[0])?;
    if entries[1..]
        .iter()
        .map(|entry| concrete_entry_modulus(entry))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|entry| *entry != source)
    {
        return Err(RnsBggError::Parameters("matrix entries have incompatible source moduli"));
    }
    let no_op = entries
        .iter()
        .map(|entry| validate_mod_down_entry(entry, &modulus))
        .collect::<Result<Vec<_>, _>>()?;
    if no_op.iter().all(|entry| *entry) {
        return Ok(value);
    }
    Ok(map_matrix(value, |entry| entry.block_mod_switch(modulus.clone(), 1)))
}

fn checked_mod_down_row(value: RnsBggRow, modulus: BigInt) -> Result<RnsBggRow, RnsBggError> {
    let source = concrete_entry_modulus(&value.0)?;
    if concrete_entry_modulus(&value.1)? != source {
        return Err(RnsBggError::Parameters("row entries have incompatible source moduli"));
    }
    let no_op = [
        validate_mod_down_entry(&value.0, &modulus)?,
        validate_mod_down_entry(&value.1, &modulus)?,
    ];
    if no_op[0] {
        return Ok(value);
    }
    Ok(map_row(value, |entry| entry.block_mod_switch(modulus.clone(), 1)))
}

/// Checked public-role ModDown.  Equal physical moduli return the original
/// graph nodes; upward conversion and non-exact drops fail before graph
/// construction.
pub(crate) fn checked_mod_down_public(
    value: RnsBggMatrix,
    modulus: BigInt,
) -> Result<RnsBggMatrix, RnsBggError> {
    checked_mod_down_matrix(value, modulus)
}

/// Checked encoding-role ModDown with the same source/target policy as public
/// matrices.
pub(crate) fn checked_mod_down_encoding(
    value: RnsBggRow,
    modulus: BigInt,
) -> Result<RnsBggRow, RnsBggError> {
    checked_mod_down_row(value, modulus)
}

/// Existing graph callers use this infallible boundary after their enclosing
/// validation.  Keep the error visible as a panic rather than constructing an
/// invalid upward or non-exact switch node.
pub(crate) fn mod_down_public(value: RnsBggMatrix, modulus: BigInt) -> RnsBggMatrix {
    checked_mod_down_public(value, modulus)
        .unwrap_or_else(|error| panic!("invalid public modulus switch: {error}"))
}

pub(crate) fn mod_down_encoding(value: RnsBggRow, modulus: BigInt) -> RnsBggRow {
    checked_mod_down_encoding(value, modulus)
        .unwrap_or_else(|error| panic!("invalid encoding modulus switch: {error}"))
}

pub(crate) fn recompose_public(
    values: Family<RnsBggMatrix>,
    coefficients: Family<Mat>,
) -> Result<RnsBggMatrix, RnsBggError> {
    if values.count() != coefficients.count() {
        return Err(RnsBggError::Binding("CRT value and coefficient families differ"));
    }
    let weighted = mxx_dsl::parallel(values.count().clone(), |index| {
        let coefficient = coefficients.at(index.clone());
        Ok(map_matrix(values.at(index), |entry| entry * coefficient.clone()))
    })?;
    crate::utils::balanced_sum_family(weighted, add_public_matrices)
        .map_err(|error| RnsBggError::Graph(error.to_string()))
}

pub(crate) fn recompose_encoding(
    values: Family<RnsBggRow>,
    coefficients: Family<Mat>,
) -> Result<RnsBggRow, RnsBggError> {
    if values.count() != coefficients.count() {
        return Err(RnsBggError::Binding("CRT value and coefficient families differ"));
    }
    let weighted = mxx_dsl::parallel(values.count().clone(), |index| {
        let coefficient = coefficients.at(index.clone());
        let row = values.at(index);
        Ok((coefficient.clone() * row.0, coefficient * row.1))
    })?;
    crate::utils::balanced_sum_family(weighted, add_encoding_rows)
        .map_err(|error| RnsBggError::Graph(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::GraphValue;
    use mxx_ir_core::IntExpr;
    use num_traits::One;

    #[test]
    fn parameter_drops_are_exact_and_terminal_is_one() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low], 1, 2).unwrap();
        assert_eq!(
            parameters.scale(0).unwrap(),
            &(high.modulus().as_ref() / parameters.level(1).unwrap().modulus().as_ref())
        );
        assert_eq!(parameters.scale(1).unwrap(), &BigUint::one());
    }

    #[test]
    fn public_alignment_selects_nested_target_and_rejects_reverse_route() {
        let high = DCRTPolyParams::new(16, 3, 17, 4, None, None);
        let middle_modulus =
            high.to_crt().0[..2].iter().map(|prime| BigUint::from(*prime)).product::<BigUint>();
        let middle = high.select_modulus(&middle_modulus).unwrap();
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![high.clone(), middle.clone(), low.clone()], 1, 1).unwrap();

        let direct = parameters.public_alignment(0, 2).unwrap();
        assert_eq!(direct.source_level, 0);
        assert_eq!(direct.target_level, 2);
        assert_eq!(direct.ratio(), &(high.modulus().as_ref() / low.modulus().as_ref()));
        assert!(!direct.is_noop());
        let action = parameters.public_alignment_action(0, 2).unwrap().unwrap();
        assert_eq!(
            action.matrix_type().modulus,
            IntExpr::constant(BigInt::from(high.modulus().as_ref().clone()))
        );
        assert_eq!(
            action.matrix_type().rows,
            IntExpr::constant(BigInt::from(high.modulus_digits()))
        );
        assert_eq!(
            action.matrix_type().columns,
            IntExpr::constant(BigInt::from(low.modulus_digits()))
        );
        assert!(parameters.public_alignment(2, 0).is_err());
        assert_eq!(
            parameters
                .mul_public_target_level(0, 2, crate::program::AttributeKnowledge::PublicPayload)
                .unwrap(),
            2
        );
        assert_eq!(
            parameters
                .mul_public_target_level(2, 1, crate::program::AttributeKnowledge::PublicPayload)
                .unwrap(),
            2
        );
        assert_eq!(
            parameters
                .mul_public_target_level(0, 1, crate::program::AttributeKnowledge::PrivatePayload)
                .unwrap(),
            1
        );
        assert!(
            parameters
                .mul_public_target_level(1, 0, crate::program::AttributeKnowledge::PrivatePayload)
                .is_err()
        );
    }

    #[test]
    fn equal_levels_have_unit_drop() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone(), level], 1, 1).unwrap();
        assert_eq!(parameters.scale(0).unwrap(), &BigUint::one());
        assert_eq!(parameters.scale(1).unwrap(), &BigUint::one());
    }

    #[test]
    fn equal_physical_alignment_is_a_noop_and_upward_switch_is_rejected() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone(), level.clone()], 1, 1).unwrap();
        let alignment = parameters.public_alignment(0, 1).unwrap();
        assert!(alignment.is_noop());
        assert_eq!(alignment.ratio(), &BigUint::one());
        assert!(parameters.public_alignment_action(0, 1).unwrap().is_none());

        let ring = parameters.ring(0).unwrap();
        let entry = ring.input("equal-modulus-entry", (1, 1));
        let matrix = ((entry.clone(), entry.clone()), (entry.clone(), entry.clone()));
        let converted =
            checked_mod_down_public(matrix.clone(), BigInt::from(level.modulus().as_ref().clone()))
                .unwrap();
        assert_eq!(converted.0.0.flatten(), matrix.0.0.flatten());
        let row = (entry.clone(), entry.clone());
        let converted_row =
            checked_mod_down_encoding(row.clone(), BigInt::from(level.modulus().as_ref().clone()))
                .unwrap();
        assert_eq!(converted_row.0.flatten(), row.0.flatten());

        let low = level.select_modulus(&BigUint::from(level.to_crt().0[0])).unwrap();
        let low_ring = mxx_dsl::Ring::new(
            BigInt::from(low.modulus().as_ref().clone()),
            low.ring_dimension() as usize,
        );
        let lower_entry = low_ring.input("upward-entry", (1, 1));
        let lower_matrix = (
            (lower_entry.clone(), lower_entry.clone()),
            (lower_entry.clone(), lower_entry.clone()),
        );
        assert!(matches!(
            checked_mod_down_public(lower_matrix, BigInt::from(level.modulus().as_ref().clone())),
            Err(RnsBggError::Parameters("modulus switch cannot increase modulus"))
        ));
        let lower_row = (lower_entry.clone(), lower_entry);
        assert!(matches!(
            checked_mod_down_encoding(lower_row, BigInt::from(level.modulus().as_ref().clone())),
            Err(RnsBggError::Parameters("modulus switch cannot increase modulus"))
        ));
    }

    #[test]
    fn key_component_columns_follow_attribute_knowledge() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low.clone()], 1, 1).unwrap();
        assert_eq!(
            parameters
                .key_component_columns(0, crate::program::AttributeKnowledge::PublicPayload)
                .unwrap(),
            high.modulus_digits()
        );
        assert_eq!(
            parameters
                .key_component_columns(0, crate::program::AttributeKnowledge::PrivatePayload)
                .unwrap(),
            low.modulus_digits()
        );
    }

    #[test]
    fn parameter_chain_rejects_mismatched_gadget_bases() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = DCRTPolyParams::new(16, 1, 17, 2, Some(vec![high.to_crt().0[0]]), None);
        assert!(matches!(
            RnsBggParameters::new(vec![high, low], 1, 1),
            Err(RnsBggError::Parameters("incompatible ring or partial basis"))
        ));
    }

    #[test]
    fn planned_crt_depth_keeps_three_operation_windows() {
        let depths =
            (0..=7).map(|level| planned_crt_depth(level, 3, 6, 2).unwrap()).collect::<Vec<_>>();
        assert_eq!(depths, vec![6, 6, 6, 5, 5, 5, 4, 4]);
        assert!(planned_crt_depth(0, 0, 6, 2).is_err());
    }

    #[test]
    fn compact_right_action_extends_only_bounded_digits() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low.clone()], 1, 1).unwrap();
        let low_ring = parameters.ring(1).unwrap();
        let low_one = low_ring.polynomial([mxx_ir_core::IntExpr::constant(BigInt::from(1u8))]);
        let action = ((low_one.clone(), low_one.clone()), (low_one.clone(), low_one.clone()));
        let decomposition = compact_decompose_matrix(&parameters, 0, 1, action.clone()).unwrap();
        assert_eq!(
            decomposition.0.0.matrix_type().modulus,
            mxx_ir_core::IntExpr::constant(BigInt::from(high.modulus().as_ref().clone()))
        );
        let high_ring = parameters.ring(0).unwrap();
        let high_one = high_ring.polynomial([mxx_ir_core::IntExpr::constant(BigInt::from(1u8))]);
        let row = (high_one.clone(), high_one);
        let result = compact_right_action_row(&parameters, 0, 1, row.clone(), action).unwrap();
        assert_eq!(result.0.matrix_type(), row.0.matrix_type());
    }

    #[test]
    fn compact_decomposition_preserves_gadget_shape_for_negative_coefficients() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low.clone()], 1, 1).unwrap();
        let gadget = parameters.public_gadget(1).unwrap();
        assert_eq!(
            gadget.matrix_type(),
            &parameters.ring(1).unwrap().matrix_type((1, low.modulus_digits()))
        );

        let negative =
            parameters.ring(1).unwrap().polynomial([IntExpr::constant(BigInt::from(-1i8))]);
        let action = ((negative.clone(), negative.clone()), (negative.clone(), negative.clone()));
        let decomposition = compact_decompose_matrix(&parameters, 0, 1, action).unwrap();
        assert_eq!(
            decomposition.0.0.matrix_type().ring_dimension,
            IntExpr::constant(BigInt::from(16u8))
        );
    }

    #[test]
    fn public_alignment_applies_one_action_to_rows_and_all_matrix_components() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high.clone(), low.clone()], 1, 1).unwrap();
        let source_ring = parameters.ring(0).unwrap();
        let source_columns = high.modulus_digits();
        let entry = source_ring.input("alignment-source", (1, source_columns));
        let row = (entry.clone(), entry.clone());
        let aligned_row = apply_public_alignment_row(&parameters, 0, 1, row).unwrap();
        let target_type = parameters.ring(1).unwrap().matrix_type((1, low.modulus_digits()));
        assert_eq!(aligned_row.0.matrix_type(), &target_type);
        assert_eq!(aligned_row.1.matrix_type(), &target_type);

        let matrix = ((entry.clone(), entry.clone()), (entry.clone(), entry));
        let aligned_matrix = apply_public_alignment_matrix(&parameters, 0, 1, matrix).unwrap();
        for entry in
            [&aligned_matrix.0.0, &aligned_matrix.0.1, &aligned_matrix.1.0, &aligned_matrix.1.1]
        {
            assert_eq!(entry.matrix_type(), &target_type);
        }
    }

    #[test]
    fn public_gadget_rejects_shift_overflow() {
        assert!(matches!(
            public_gadget_base(u64::BITS),
            Err(RnsBggError::Parameters("public gadget base overflow"))
        ));
    }

    #[test]
    fn checked_modulus_conversion_handles_greater_equal_and_less() {
        let source_ring = Ring::new(BigInt::from(105u8), 4);
        let source = source_ring.input("checked-modulus-source", (1, 1));
        let reduced = checked_modulus_convert(source.clone(), BigInt::from(15u8)).unwrap();
        let equal = checked_modulus_convert(source.clone(), BigInt::from(105u8)).unwrap();
        let less_source = Ring::new(BigInt::from(15u8), 4).input("checked-modulus-less", (1, 1));
        let extended = checked_modulus_convert(less_source, BigInt::from(105u8)).unwrap();

        assert_eq!(reduced.matrix_type().modulus, IntExpr::constant(BigInt::from(15u8)));
        assert_eq!(equal.flatten(), source.flatten());
        assert_eq!(extended.matrix_type().modulus, IntExpr::constant(BigInt::from(105u8)));

        let graph = mxx_dsl::DslContext::new("checked-modulus-conversion")
            .artifact_output("reduced", reduced)
            .unwrap()
            .artifact_output("equal", equal)
            .unwrap()
            .artifact_output("extended", extended)
            .unwrap()
            .build()
            .unwrap();
        graph.validate(&mxx_ir_core::ParamEnv::default()).unwrap();
    }

    fn helper_test_parameters() -> RnsBggParameters {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        RnsBggParameters::new(vec![high.clone(), high, low], 1, 1).unwrap()
    }

    fn helper_test_program(depth: usize) -> crate::program::ExponentLutProgram {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let layout = crate::program::ExpLayout { stride_log2: 0, width_log2: 1 };
        let table = crate::program::LutTable::unary(
            2,
            layout,
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(layout),
        )
        .unwrap();
        builder.register_lut(table.clone()).unwrap();
        builder.register_lut(table).unwrap();
        let mut output = input;
        for lut in 0..depth {
            output = builder
                .lut(crate::program::Operand::Scalar(output), crate::program::LutId(lut as u32))
                .unwrap();
        }
        builder.finish(vec![crate::program::Operand::Scalar(output)]).unwrap()
    }

    #[test]
    fn projection_helper_keys_include_unused_luts_and_deduplicate() {
        let parameters = helper_test_parameters();
        let program = helper_test_program(0);
        let keys = canonical_lut_helper_keys(&parameters, &program.encryption_inputs()).unwrap();
        assert!(!keys.is_empty(), "an unused registered LUT is still in the projection");
        assert_eq!(keys.iter().filter(|(level, _)| *level == 0).count(), 2);
        assert_eq!(keys.iter().filter(|(level, _)| *level == 1).count(), 2);
        assert!(keys.iter().any(|(level, _)| *level == 0));
        assert!(keys.iter().any(|(level, _)| *level == 1));
    }

    #[test]
    fn projection_helper_keys_ignore_body_and_keep_equal_physical_levels_distinct() {
        let parameters = helper_test_parameters();
        let unused = helper_test_program(0);
        let used = helper_test_program(2);
        let unused_projection = unused.encryption_inputs();
        let used_projection = used.encryption_inputs();
        assert_eq!(
            canonical_lut_helper_keys(&parameters, &unused_projection).unwrap(),
            canonical_lut_helper_keys(&parameters, &used_projection).unwrap()
        );
        assert_eq!(
            encryption_projection_identity(
                &unused_projection,
                &mxx_ir_core::ParamEnv::default(),
                &parameters,
            )
            .unwrap(),
            encryption_projection_identity(
                &used_projection,
                &mxx_ir_core::ParamEnv::default(),
                &parameters,
            )
            .unwrap()
        );
        let keys = canonical_lut_helper_keys(&parameters, &unused_projection).unwrap();
        assert!(keys.contains(&(0, 1)));
        assert!(keys.contains(&(1, 1)));
    }

    #[test]
    fn stage_descriptors_are_distinct_compile_time_types() {
        use std::any::TypeId;

        assert_ne!(
            TypeId::of::<RnsBggPreprocessEncryptDescriptor>(),
            TypeId::of::<RnsBggPreprocessOnlineDescriptor>()
        );
        assert_ne!(
            TypeId::of::<RnsBggPreprocessEncryptDescriptor>(),
            TypeId::of::<RnsBggEncryptionDescriptor>()
        );
        assert_ne!(
            TypeId::of::<RnsBggPreprocessOnlineDescriptor>(),
            TypeId::of::<RnsBggEncryptionDescriptor>()
        );
    }

    #[test]
    fn refresh_noise_weight_sums_include_coordinates_and_fresh_h() {
        let base = BigUint::from(4u8);
        let coordinates = BigUint::from(3u8);
        assert_eq!(
            refresh_noise_weight_sums(&base, 2, &coordinates, &BigUint::one()),
            (BigUint::from(5u8) * &coordinates, BigUint::from(17u8) * &coordinates)
        );
        assert_eq!(
            refresh_noise_weight_sums(&base, 2, &coordinates, &BigUint::from(2u8)),
            (BigUint::from(10u8) * &coordinates, BigUint::from(68u8) * &coordinates)
        );
    }

    #[test]
    fn refresh_noise_ceil_div_rounds_outward() {
        assert_eq!(refresh_noise_ceil_div(&BigUint::from(10u8), &BigUint::from(3u8)), 4u8.into());
        assert_eq!(refresh_noise_ceil_div(&BigUint::from(9u8), &BigUint::from(3u8)), 3u8.into());
        assert_eq!(refresh_noise_ceil_div(&BigUint::zero(), &BigUint::from(3u8)), BigUint::zero());
    }
}
