//! The shared, scheme-independent RNS-BGG core.
//!
//! This module intentionally contains only the contracts shared by the stage
//! owners. Public-key preparation, encoding/encryption, and noise evaluation
//! live in their respective modules; this file does not provide a scheme
//! facade or an artifact/material wrapper hierarchy.

mod dynamic;
mod encoding;
#[cfg(all(test, feature = "gpu"))]
pub(crate) mod gpu;
mod noise;
mod public_key;
mod setup;
mod workload;

use bigdecimal::BigDecimal;
use mxx_backends::poly::{PolyParams, dcrt::params::DCRTPolyParams};
use mxx_dsl::{Family, Mat, Preimage, Ring, SmallMatrix};
use mxx_ir_core::{IntExpr, ring::ConcreteRing};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::{
    decoder::decoding::{DecoderDeclaration, DecoderError},
    program::{
        ExponentLUTProgramLowering, ExponentLutProgramId, InputBinding, InputShape, LutId, RhsId,
        WireId,
    },
    program_gadgets::prf::PrfError,
};

pub use encoding::{
    EncodingLowering, RnsBggEncoding, RnsBggMaskSecret, RnsBggPayloadSecret, RnsEncryptBuild,
    RnsOnlineEvalBuild, decode_output, decode_public_output, verify_output,
};
pub(crate) use noise::simulate_decoder_noise;
pub use noise::{
    RnsBggDecoderNoise, RnsBggDynamicNoise, RnsBggMaskSecretComponent, RnsBggNoiseBindings,
    RnsBggNoiseBound, RnsBggNoiseEvaluator, RnsBggNoiseFamily, RnsBggNoiseRhs, RnsBggNoiseTrace,
    RnsBggNoiseValue, RnsBggPackageNoise, RnsBggParameterManifest, RnsBggPlanRequest,
    RnsNoiseBuild,
};
pub use public_key::{PublicKeyLowering, RnsBggPublicKey, RnsPreprocessBuild};
pub(crate) use setup::RnsBggPackage;
pub use setup::{
    RnsBggDynamicBinding, RnsBggDynamicDistributions, RnsBggDynamicRhs, RnsBggDynamicSetup,
    RnsBggGaussian, RnsBggPackageSource, covering_automorphism_layout,
};
pub use workload::RnsBggDerivedWorkload;

/// Name of the graph input carrying the shared 32-byte hash key. A stage
/// graph declares it only where it samples hash-derived public matrices.
pub(crate) const HASH_KEY_INPUT: &str = "rns-bgg/hash-key";

/// Whether `graph` declares a root input named `name`, so a stage binds an
/// optional runtime input only when its graph consumes it.
pub(crate) fn graph_declares_input(graph: &mxx_dsl::BuiltGraph, name: &str) -> bool {
    graph.graph.root_scope().nodes().iter().any(|node| {
        matches!(node.kind(), mxx_ir_core::node::NodeKind::Input { name: input, .. } if input == name)
    })
}

/// A two-polynomial row block.
pub type RnsBggRow = (Mat, Mat);
/// A two-by-two polynomial block matrix.
pub type RnsBggMatrix = ((Mat, Mat), (Mat, Mat));
/// A two-by-two matrix of signed compact preimages.
/// Raw signed gadget digits of a right-action matrix in its decomposition ring.
pub(crate) type RnsBggDigits = ((Preimage, Preimage), (Preimage, Preimage));
/// Bounded digits of a right action, represented in the ring that consumes them.
pub(crate) type RnsBggDecomposition = ((SmallMatrix, SmallMatrix), (SmallMatrix, SmallMatrix));
fn decoder_commitment_hex(commitment: [u8; 32]) -> String {
    commitment.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `decoder/<commitment>/<role>`: the namespace of every artifact of one
/// decoder declaration.
pub(crate) fn decoder_artifact_name(commitment: [u8; 32], role: &str) -> String {
    format!("decoder/{}/{role}", decoder_commitment_hex(commitment))
}

/// `decoder/<commitment>/smudge/<role>`: an artifact of the decoder's smudge
/// Program child.
pub(crate) fn decoder_smudge_artifact_name(commitment: [u8; 32], role: &str) -> String {
    decoder_artifact_name(commitment, &format!("smudge/{role}"))
}

/// The public-key artifact `port` of smudge Program input `wire`.
pub(crate) fn decoder_smudge_input_key_artifact_name(
    commitment: [u8; 32],
    wire: WireId,
    port: usize,
) -> String {
    decoder_smudge_artifact_name(commitment, &format!("input-key/{wire:?}.{port}"))
}

/// Build the canonical path of the switching ciphertexts of LUT `lut`
/// evaluated at `level` with input layout `layout`, followed by `suffix`.
pub(crate) fn lut_switching_role(
    level: usize,
    lut: LutId,
    layout: crate::program::ExpLayout,
    suffix: &str,
) -> String {
    format!("lut-switching/{level}/{}/{lut:?}/{layout:?}{suffix}", level + 1)
}

/// Number `m^2` of digit-companion encodings of one Dynamic package read at
/// `level` whose two components have `output_columns` columns each: one per
/// digit row and output column.
pub(crate) fn companion_count(
    parameters: &RnsBggParameters,
    level: usize,
    output_columns: usize,
) -> Result<usize, RnsBggError> {
    Ok(4 * parameters.component_columns(level)? * output_columns)
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
    Decoder(#[from] crate::decoder::decoding::DecoderError),
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
    if sigma < BigDecimal::zero() || exponent.unsigned_abs() > u64::from(u32::MAX) {
        return Err(RnsBggError::Parameters("encoding_error_sigma must be nonnegative"));
    }
    Ok(sigma)
}

/// A zero sigma selects noise-free encodings: no error is sampled or added.
fn encoding_error_bound(sigma: &BigDecimal) -> Result<BigUint, RnsBggError> {
    if sigma.is_zero() {
        return Ok(BigUint::zero());
    }
    let bound = std::panic::catch_unwind(|| {
        mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound(sigma)
    })
    .map_err(|_| RnsBggError::Parameters("encoding_error_sigma is unrepresentable"))?;
    if bound.is_zero() {
        return Err(RnsBggError::Parameters("encoding_error_sigma derives a zero cutoff"));
    }
    Ok(bound)
}

/// When the RHS values of a Program, and every secret-dependent helper, are
/// fixed. One execution of a Program has exactly one mode.
///
/// `Static` binds the RHS plaintexts during preprocessing under a payload
/// secret distinct from the encryption mask, and multiplies by setup-fixed
/// ciphertexts. `Dynamic` finishes preprocessing from public material only,
/// against a [`RnsBggDynamicSetup`] on an equal-basis chain: encryption binds
/// one secret `s = (s_bar, 1)` as both mask and payload, and publishes one
/// package per RHS coordinate and per automorphism of the setup, each a
/// gadget ciphertext `C` with the public encodings `z = s B - d s G + e` of
/// its gadget digits `d`. A private product then has the
/// ciphertext-independent key `T = sum_a B_a U_a`, where `U = Dec(A)` is the
/// gadget decomposition of the state key, and a LUT is a sum of public
/// scalar actions on automorphism-switched encodings (see [`dynamic`]).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum RhsMode {
    #[default]
    Static,
    Dynamic,
}

/// How the consumer of a Program uses its outputs. Preprocessing, online
/// evaluation, and noise simulation resolve every output carrier through
/// [`RnsBggParameters::output_carrier`] from the same value, so they agree on
/// its ring, column width, and payload form.
///
/// Normalization is separate from this choice: a coefficient output is always
/// moved once to the level after its LUT (a ModDown by the level's drop, an
/// identity on an equal-basis chain), whatever its use.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum RnsOutputUse {
    /// The normalized carrier itself: a private output is the full-gadget
    /// encoding `s (A - v G) + e` with `A` of shape `2 x m`. It is consumed by
    /// a further key-homomorphic computation (for example a predicate
    /// encryption key equation), never decoded directly.
    Gadget,
    /// The normalized carrier followed by the direct-decoding action: a
    /// coefficient output is right-multiplied by `G^-1(k)` (see
    /// [`RnsBggParameters::coefficient_output_scale`]); a Dynamic exponent
    /// output below the terminal level by `G^-1(floor(q/2))`. Terminal
    /// exponent outputs already have that scalar form.
    #[default]
    DirectDecode,
}

/// The payload gadget of a resolved output carrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RnsOutputGadget {
    /// The state gadget of the carrier's level (a private exponent wire).
    State,
    /// The full public gadget of the carrier's level (a public wire, or a
    /// normalized coefficient wire).
    Public,
    /// A scalar column per component: the payload is `scale v s_j`.
    Scalar,
}

/// The resolved carrier of one Program output: its ring level, its columns
/// per component, the payload scale and gadget, the weight of the direct
/// decoding action applied after normalization, and the plaintext modulus a
/// direct decode reads it with. The public-key, encoding, and noise paths
/// derive it from one call, so they cannot choose different actions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RnsOutputCarrier {
    /// Ring level of the normalized carrier.
    pub level: usize,
    /// Columns per component.
    pub columns: usize,
    /// Payload scale: the row is `s A - scale v s G + e` for gadget `G`.
    pub scale: BigUint,
    /// Payload gadget.
    pub gadget: RnsOutputGadget,
    /// Weight `w` of the action `G^-1(w)` applied after normalization.
    pub action: Option<BigUint>,
    /// Plaintext modulus of a direct decode, when the carrier is decodable.
    pub plaintext_modulus: Option<u64>,
}

/// Validated modulus-chain and encoding-noise parameters.
#[derive(Clone, Debug)]
pub struct RnsBggParameters {
    levels: Vec<DCRTPolyParams>,
    drops: Vec<BigUint>,
    encoding_error_sigma: BigDecimal,
    encoding_error_bound: BigUint,
    operations_per_drop: usize,
    rhs_mode: RhsMode,
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
            rhs_mode: RhsMode::Static,
        })
    }

    /// RHS timing mode shared by preprocessing, encryption, online
    /// evaluation, and noise analysis.
    pub fn rhs_mode(&self) -> RhsMode {
        self.rhs_mode
    }

    /// These parameters with RHS timing mode `mode`.
    pub fn with_rhs_mode(mut self, mode: RhsMode) -> Self {
        self.rhs_mode = mode;
        self
    }

    pub fn levels(&self) -> &[DCRTPolyParams] {
        &self.levels
    }

    /// Levels with distinct ordered CRT bases, in chain order. Adjacent
    /// logical levels between drops share one physical ring; an execution
    /// backend registers each physical ring once.
    pub fn distinct_levels(&self) -> Vec<&DCRTPolyParams> {
        let mut levels = Vec::<&DCRTPolyParams>::new();
        for level in &self.levels {
            if !levels.iter().any(|known| known.to_crt().0 == level.to_crt().0) {
                levels.push(level);
            }
        }
        levels
    }

    pub fn level(&self, level: usize) -> Result<&DCRTPolyParams, RnsBggError> {
        self.levels.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    pub fn scale(&self, level: usize) -> Result<&BigUint, RnsBggError> {
        self.drops.get(level).ok_or(RnsBggError::Parameters("level outside modulus chain"))
    }

    /// Public scale of the message in a level-`level` payload relation.
    /// Below the terminal level it is the level's drop `p_level`, which the
    /// next ModDown removes. The terminal level has no further drop and an
    /// identity state gadget, so its message carries the public scale
    /// `floor(q / 2)`: distinct monomial plaintexts then differ by about
    /// `q / 2` per coefficient and fail the `q / 4` residual decode.
    pub fn message_scale(&self, level: usize) -> Result<BigUint, RnsBggError> {
        if level + 1 == self.levels.len() {
            Ok(self.level(level)?.modulus().as_ref() / 2u8)
        } else {
            self.scale(level).cloned()
        }
    }

    pub fn component_columns(&self, level: usize) -> Result<usize, RnsBggError> {
        self.level(level)?;
        Ok(self.levels.get(level + 1).map_or(1, PolyParams::modulus_digits))
    }

    /// Columns per component of the switching target of a LUT whose switching
    /// ciphertexts live at `next_level`, and so of its raw output carrier.
    ///
    /// An `Exponent` output uses the state gadget of `next_level`, i.e.
    /// `component_columns(next_level)`. A `Coefficient` output uses the full
    /// public gadget of `next_level`, i.e. its `modulus_digits`: its payload
    /// carries no scale that could annihilate the multiples of a smaller
    /// gadget modulus, so the gadget modulus must be the finalized carrier's
    /// own (see `lut_switching_target`).
    pub fn lut_output_columns(
        &self,
        next_level: usize,
        representation: crate::program::LutOutputRepresentation,
    ) -> Result<usize, RnsBggError> {
        match representation {
            crate::program::LutOutputRepresentation::Coefficient => {
                Ok(self.level(next_level)?.modulus_digits())
            }
            crate::program::LutOutputRepresentation::Exponent(_) => {
                self.component_columns(next_level)
            }
        }
    }

    /// The scale and gadget of the switching target of a LUT whose switching
    /// ciphertexts live at `next_level`. An `Exponent` output uses
    /// `message_scale(next_level)` and `state_gadget(next_level)`; a
    /// `Coefficient` output uses scale 1 and `public_gadget(next_level)`, so
    /// that after the source-level action (`Delta_l`) and the one finalizing
    /// ModDown its payload is exactly `v r G_(next_level)`.
    pub(crate) fn lut_switching_target(
        &self,
        next_level: usize,
        representation: crate::program::LutOutputRepresentation,
    ) -> Result<(BigUint, Mat), RnsBggError> {
        match representation {
            crate::program::LutOutputRepresentation::Coefficient => {
                Ok((BigUint::from(1u8), self.public_gadget(next_level)?))
            }
            crate::program::LutOutputRepresentation::Exponent(_) => {
                Ok((self.message_scale(next_level)?, self.state_gadget(next_level)?))
            }
        }
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
        let target_gadget =
            self.public_gadget(target_level)?.centered_rebase(&self.ring(source_level)?);
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

    /// The default sampling contract of fresh encoding errors: the encoding
    /// sigma with its cutoff. Static stages sample every encoding role under
    /// it; Dynamic stages sample under their setup's contracts.
    pub fn encoding_contract(&self) -> RnsBggGaussian {
        RnsBggGaussian::new(self.encoding_error_sigma.clone(), self.encoding_error_bound.clone())
            .expect("validated encoding sigma and cutoff")
    }

    pub fn operations_per_drop(&self) -> usize {
        self.operations_per_drop
    }

    pub fn ring(&self, level: usize) -> Result<Ring, RnsBggError> {
        let parameters = self.level(level)?;
        Ok(Ring::from_crt_moduli(
            parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
            parameters.ring_dimension(),
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
        if self.rhs_mode == RhsMode::Dynamic {
            digest.update(b"dynamic-rhs");
        }
        digest.finalize().into()
    }

    /// Check that every level has the same ordered CRT basis and gadget and
    /// that the chain has at least two levels. On such a chain every drop is
    /// one, every ModDown is an identity with no rounding, and every level
    /// below the terminal one has the full gadget of its successor as its
    /// state gadget and message scale one. Dynamic evaluation requires it:
    /// a package is read at any logical level, so every level must be its
    /// physical ring.
    pub fn validate_equal_basis(&self) -> Result<(), RnsBggError> {
        let first = &self.levels[0];
        if self.levels.len() < 2 ||
            self.levels.iter().any(|level| !compatible_level_parameters(first, level))
        {
            return Err(RnsBggError::Parameters(
                "Dynamic evaluation needs an equal-basis chain with a gadget successor",
            ));
        }
        Ok(())
    }

    /// Check one Dynamic evaluation at `level`: Dynamic parameters on an
    /// equal-basis chain, with a gadget successor after `level`. A route with
    /// a real modulus drop is rejected; there is no fallback.
    pub(crate) fn dynamic_route(&self, level: usize) -> Result<(), RnsBggError> {
        if self.rhs_mode != RhsMode::Dynamic {
            return Err(RnsBggError::Parameters("a Dynamic route needs Dynamic parameters"));
        }
        self.validate_equal_basis()?;
        if level + 1 >= self.levels.len() {
            return Err(RnsBggError::Parameters("a Dynamic evaluation level needs a successor"));
        }
        Ok(())
    }

    pub(crate) fn state_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        self.level(level)?;
        match self.levels.get(level + 1) {
            Some(low) => {
                let base = 1u64
                    .checked_shl(low.base_bits())
                    .ok_or(RnsBggError::Parameters("state gadget base overflow"))?;
                Ok(self
                    .ring(level + 1)?
                    .gadget(1, base, low.modulus_digits())
                    .centered_rebase(&self.ring(level)?))
            }
            None => Ok(self.ring(level)?.identity(1)),
        }
    }

    /// The full public gadget row `g` of `level`: the RNS gadget of its CRT
    /// basis with its base and digit count, `1 x modulus_digits`.
    pub fn public_gadget(&self, level: usize) -> Result<Mat, RnsBggError> {
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        Ok(self.ring(level)?.gadget(1, base, parameters.modulus_digits()))
    }

    /// `G^-1(weight)`: the gadget decomposition of the `1 x 1` `weight` (in
    /// the ring of `gadget_level`) under that level's full gadget, rebased to
    /// `ring_level`.
    ///
    /// Right-multiplying a carrier whose payload is `v G` by it selects
    /// `v weight`: per tower it selects the base digits of
    /// `weight mod q_t`, and `G G^-1(weight) = weight` modulo the gadget
    /// modulus. Its digits are bounded by the gadget base, so its noise gain
    /// [`Self::gadget_inverse_l1`] does not grow with `weight`. `weight` is a
    /// graph value, so a loop body may select it per member.
    pub(crate) fn gadget_inverse(
        &self,
        gadget_level: usize,
        ring_level: usize,
        weight: Mat,
    ) -> Result<SmallMatrix, RnsBggError> {
        let parameters = self.level(gadget_level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        Ok(weight
            .decompose(base, parameters.modulus_digits())
            .centered_rebase(&self.ring(ring_level)?))
    }

    /// The constant `1 x n` row of `weights` in the ring of `level`.
    pub(crate) fn weight_row(&self, level: usize, weights: &[BigUint]) -> Result<Mat, RnsBggError> {
        let ring = self.ring(level)?;
        let entries = weights
            .iter()
            .map(|weight| ring.polynomial([IntExpr::constant(weight.clone())]))
            .collect::<Vec<_>>();
        Ok(match entries.len() {
            1 => entries.into_iter().next().expect("one weight"),
            _ => Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, entries),
        })
    }

    /// The largest `l1` norm of [`Self::gadget_inverse`] over `weights`: the
    /// sum over towers of the base-`B` digits of `weight mod q_t`.
    pub(crate) fn gadget_inverse_l1(
        &self,
        gadget_level: usize,
        weights: &[BigUint],
    ) -> Result<BigUint, RnsBggError> {
        let parameters = self.level(gadget_level)?;
        let base = BigUint::from(public_gadget_base(parameters.base_bits())?);
        let primes = parameters.to_crt().0;
        Ok(weights
            .iter()
            .map(|weight| {
                primes
                    .iter()
                    .map(|prime| {
                        let mut residue = weight % prime;
                        let mut sum = BigUint::zero();
                        while !residue.is_zero() {
                            sum += &residue % &base;
                            residue /= &base;
                        }
                        sum
                    })
                    .sum::<BigUint>()
            })
            .max()
            .unwrap_or_default())
    }

    /// The decryption modulus `t = 2 (W + 1)` of a coefficient output with
    /// payload bound `W` (see
    /// [`crate::program::ExponentLutProgram::wire_coefficient_bound`]), so
    /// that its payload lies in `(-t/2, t/2)`.
    pub fn coefficient_decoding_modulus(bound: u64) -> Result<u64, RnsBggError> {
        bound
            .checked_add(1)
            .and_then(|bound| bound.checked_mul(2))
            .ok_or(RnsBggError::Parameters("coefficient decoding modulus overflow"))
    }

    /// The scale `k` of a coefficient output finalized at `level` and
    /// decrypted modulo `t`, and the loss `r = q - t k`.
    ///
    /// `k = floor(floor(q/t) / 2^b) 2^b` for the largest `b >= 0` with
    /// `t 2^b <= q/(32 t)` (`b = 0` if there is none), so `0 <= r < t 2^b`:
    /// clearing the low bits of `k` keeps the balanced digits of `G^-1(k)`
    /// small while the decoding loss `r W / t` stays below `q/(64 t)`.
    pub fn coefficient_output_scale(
        &self,
        level: usize,
        t: u64,
    ) -> Result<(BigUint, BigUint), RnsBggError> {
        if t < 2 {
            return Err(RnsBggError::Parameters("coefficient decoding modulus below two"));
        }
        let q: BigUint = self.level(level)?.modulus().as_ref().clone();
        let t = BigUint::from(t);
        let limit = &q / (&t * &t * 32u8);
        let low_bits = if limit.is_zero() { 0 } else { limit.bits() - 1 };
        let k = ((&q / &t) >> low_bits) << low_bits;
        if k.is_zero() {
            return Err(RnsBggError::Parameters(
                "coefficient output modulus below its decoding modulus",
            ));
        }
        let loss = &q - &t * &k;
        Ok((k, loss))
    }

    /// The decryption of the coefficient output `wire` of `program`, finalized
    /// at `level`: its decryption modulus `t` (from the payload bound), its
    /// scale `k` and the loss `r = q - t k` (see
    /// [`Self::coefficient_output_scale`]).
    pub(crate) fn coefficient_output_decoding(
        &self,
        program: &crate::program::ExponentLutProgram,
        wire: WireId,
        level: usize,
    ) -> Result<(u64, BigUint, BigUint), RnsBggError> {
        let bound = program
            .wire_coefficient_bound(wire)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
            .ok_or(RnsBggError::Binding("not a coefficient output"))?;
        let t = Self::coefficient_decoding_modulus(bound)?;
        let (k, loss) = self.coefficient_output_scale(level, t)?;
        Ok((t, k, loss))
    }

    /// The direct-decoding action `G^-1(w)` of an output carrier normalized at
    /// `level` with weight `w`: right-multiplying a carrier whose payload is
    /// `v G` gives the payload `w v` against the identity gadget.
    pub(crate) fn output_action(
        &self,
        level: usize,
        w: &BigUint,
    ) -> Result<SmallMatrix, RnsBggError> {
        self.gadget_inverse(level, level, self.weight_row(level, std::slice::from_ref(w))?)
    }

    /// The carrier of output `wire` of `program`, whose lowered value is at
    /// logical level `level`, under `output_use`.
    ///
    /// A coefficient output is normalized once to `level + 1`: its payload is
    /// then `v s G` against the full public gadget there. `DirectDecode`
    /// applies `G^-1(k)` (a scalar column with payload `k v s_j`, decoded
    /// modulo `t`); `Gadget` keeps the full gadget. An exponent output is not
    /// moved. A terminal exponent output already has the scalar form `floor(q
    /// / 2) v s_j` and has no gadget successor, so `Gadget` rejects it. A
    /// Dynamic exponent output below the terminal level has scale one on an
    /// equal-basis chain; `DirectDecode` applies `G^-1(floor(q / 2))` to it.
    /// A Static exponent output below the terminal level keeps its scaled
    /// state-gadget form, which is not directly decodable. Public outputs
    /// other than coefficient outputs keep their form.
    pub fn output_carrier(
        &self,
        program: &crate::program::ExponentLutProgram,
        wire: WireId,
        level: usize,
        output_use: RnsOutputUse,
    ) -> Result<RnsOutputCarrier, RnsBggError> {
        let metadata = |_| RnsBggError::Binding("output wire metadata");
        if let Some(bound) = program.wire_coefficient_bound(wire).map_err(metadata)? {
            let normalized =
                level.checked_add(1).ok_or(RnsBggError::Parameters("output level overflow"))?;
            let parameters = self.level(normalized)?;
            return Ok(match output_use {
                RnsOutputUse::Gadget => RnsOutputCarrier {
                    level: normalized,
                    columns: parameters.modulus_digits(),
                    scale: BigUint::one(),
                    gadget: RnsOutputGadget::Public,
                    action: None,
                    plaintext_modulus: None,
                },
                RnsOutputUse::DirectDecode => {
                    let t = Self::coefficient_decoding_modulus(bound)?;
                    let (k, _) = self.coefficient_output_scale(normalized, t)?;
                    RnsOutputCarrier {
                        level: normalized,
                        columns: 1,
                        scale: k.clone(),
                        gadget: RnsOutputGadget::Scalar,
                        action: Some(k),
                        plaintext_modulus: Some(t),
                    }
                }
            });
        }
        if program.wire_knowledge(wire).map_err(metadata)? ==
            crate::program::AttributeKnowledge::PublicPayload
        {
            return Ok(RnsOutputCarrier {
                level,
                columns: self.level(level)?.modulus_digits(),
                scale: BigUint::one(),
                gadget: RnsOutputGadget::Public,
                action: None,
                plaintext_modulus: None,
            });
        }
        let terminal = level + 1 == self.levels.len();
        let state = RnsOutputCarrier {
            level,
            columns: self.component_columns(level)?,
            scale: self.message_scale(level)?,
            gadget: if terminal { RnsOutputGadget::Scalar } else { RnsOutputGadget::State },
            action: None,
            plaintext_modulus: terminal.then_some(2),
        };
        match (output_use, terminal) {
            (RnsOutputUse::Gadget, true) => {
                Err(RnsBggError::Parameters("a gadget output needs a gadget successor level"))
            }
            (RnsOutputUse::DirectDecode, false)
                if self.rhs_mode == RhsMode::Dynamic && self.scale(level)?.is_one() =>
            {
                let half = self.level(level)?.modulus().as_ref() / 2u8;
                Ok(RnsOutputCarrier {
                    level,
                    columns: 1,
                    scale: half.clone(),
                    gadget: RnsOutputGadget::Scalar,
                    action: Some(half),
                    plaintext_modulus: Some(2),
                })
            }
            _ => Ok(state),
        }
    }

    /// The gadget row of `carrier`'s payload in its ring: the state or public
    /// gadget of its level, or `1` for a scalar column.
    pub(crate) fn output_gadget(&self, carrier: &RnsOutputCarrier) -> Result<Mat, RnsBggError> {
        match carrier.gadget {
            RnsOutputGadget::State => self.state_gadget(carrier.level),
            RnsOutputGadget::Public => self.public_gadget(carrier.level),
            RnsOutputGadget::Scalar => Ok(self.ring(carrier.level)?.identity(1)),
        }
    }

    /// The `l1` norm and the squared `l2` norm of `G^-1(k)` at `level`, the
    /// finalization action of a coefficient output: per tower, the balanced
    /// base-`B` digits of the centered residue of `k`, exactly as the runtime
    /// decomposition computes them. The level's gadget must be exact (no
    /// dropped moduli), so that `G G^-1(k) = k`.
    pub(crate) fn gadget_inverse_norms(
        &self,
        level: usize,
        k: &BigUint,
    ) -> Result<(BigUint, BigUint), RnsBggError> {
        let parameters = self.level(level)?;
        if parameters.dropped_moduli() != 0 {
            return Err(RnsBggError::Parameters("a coefficient output needs an exact gadget"));
        }
        let (moduli, _, _) = parameters.to_crt();
        let digits = parameters.modulus_digits() / moduli.len();
        let base = 1i128 << parameters.base_bits();
        let (mut l1, mut l2_squared) = (BigUint::zero(), BigUint::zero());
        for modulus in moduli {
            let residue = (k % modulus).to_u64().expect("a residue fits its tower");
            let mut value = if 2 * u128::from(residue) > u128::from(modulus) {
                i128::from(residue) - i128::from(modulus)
            } else {
                i128::from(residue)
            };
            for _ in 0..digits {
                let (quotient, remainder) = (value.div_euclid(base), value.rem_euclid(base));
                let (digit, next) = if remainder < base / 2 {
                    (remainder, quotient)
                } else if remainder > base / 2 || quotient % 2 != 0 {
                    (remainder - base, quotient + 1)
                } else {
                    (remainder, quotient)
                };
                l1 += digit.unsigned_abs();
                l2_squared += digit.unsigned_abs() * digit.unsigned_abs();
                value = next;
            }
        }
        Ok((l1, l2_squared))
    }

    /// The unmask trapdoor of a decoder at `level`: its gadget base
    /// and digit count are the level's own, and the preimage coefficient
    /// bound is the sampler's default cutoff for trapdoor sigma `sigma`.
    pub fn unmask_gadget(
        &self,
        level: usize,
        sigma: f64,
    ) -> Result<(u64, usize, BigUint), RnsBggError> {
        let parameters = self.level(level)?;
        let base = public_gadget_base(parameters.base_bits())?;
        let digits = parameters.modulus_digits();
        let bound = mxx_backends::sampler::bounds::default_preimage_cutoff(
            parameters.ring_dimension(),
            crate::decoder::unmask::UNMASK_ROWS,
            digits,
            u32::try_from(base).map_err(|_| RnsBggError::Parameters("unmask gadget base"))?,
            sigma,
            mxx_ir_core::types::TrapdoorInstantiation::Computational,
        )
        .ok_or(RnsBggError::Parameters("invalid unmask trapdoor sigma"))?;
        Ok((base, digits, bound))
    }

    /// The level of the source output of `declaration` in `source_program`.
    pub(crate) fn decoder_source_level(
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
    ) -> Result<usize, RnsBggError> {
        if source_program.id() != declaration.source_program() {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        RnsBggDerivedWorkload::from_program(source_program)?.level(declaration.source_output())
    }

    /// The plaintext modulus `p` of `declaration`'s decoder of
    /// `source_program`, its CRT primes, and the decoding spacing
    /// `Delta = q_L / p` at the source level `L`.
    ///
    /// A Static decoder has no explicit modulus: `p = q_(L+1)`, the strict
    /// drop after the source, and `Delta` is that drop. A Dynamic chain keeps
    /// one modulus `q`, so its declaration names `p` explicitly
    /// ([`DecoderDeclaration::with_plaintext_modulus`]): an odd proper CRT
    /// divisor of `q`, the product of a nonempty strict subset of its primes.
    pub fn decoder_plaintext(
        &self,
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
    ) -> Result<(BigUint, Vec<u64>, BigUint), RnsBggError> {
        let relation = || RnsBggError::Decoder(DecoderError::InvalidModulusRelation);
        let level = Self::decoder_source_level(declaration, source_program)?;
        let decoding = self.level(level)?;
        let q: BigUint = decoding.modulus().as_ref().clone();
        let (p, primes) = match (self.rhs_mode, declaration.plaintext_modulus()) {
            (RhsMode::Static, None) => {
                let plaintext = self.level(level + 1).map_err(|_| relation())?;
                (plaintext.modulus().as_ref().clone(), plaintext.to_crt().0)
            }
            (RhsMode::Dynamic, Some(p)) => {
                let primes = decoding
                    .to_crt()
                    .0
                    .into_iter()
                    .filter(|prime| (p % prime).is_zero())
                    .collect::<Vec<_>>();
                let product: BigUint = primes.iter().copied().map(BigUint::from).product();
                if &product != p || primes.is_empty() || p == &q {
                    return Err(relation());
                }
                (p.clone(), primes)
            }
            _ => return Err(relation()),
        };
        if p.is_zero() || !(&q % &p).is_zero() || p == q {
            return Err(relation());
        }
        let spacing = &q / &p;
        Ok((p, primes, spacing))
    }

    /// The ring `R_p` of the decoded plaintext.
    pub(crate) fn decoder_plaintext_ring(
        &self,
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
    ) -> Result<Ring, RnsBggError> {
        let (_, primes, _) = self.decoder_plaintext(declaration, source_program)?;
        let dimension = self.level(0)?.ring_dimension();
        Ok(Ring::from_crt_moduli(primes.into_iter().map(IntExpr::from).collect(), dimension))
    }

    /// The source selection of `declaration`'s decoder: the action
    /// `Dec_g(w)`, in the source ring, on component 1 of the source output,
    /// and its gadget level and weight. A Static source at level `L` carries
    /// `p_L mu s G_(L+1)`, and `Dec_(g_(L+1))(1)` selects `Delta mu` with
    /// `Delta = p_L`. A Dynamic source carries `mu s G` against the full
    /// gadget of `q`, and `Dec_g(Delta)` selects `Delta mu`.
    pub(crate) fn decoder_selection(
        &self,
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
    ) -> Result<(SmallMatrix, usize, BigUint), RnsBggError> {
        let level = Self::decoder_source_level(declaration, source_program)?;
        let (gadget_level, weight) = match self.rhs_mode {
            RhsMode::Static => (level + 1, BigUint::one()),
            RhsMode::Dynamic => {
                self.dynamic_route(level)?;
                (level, self.decoder_plaintext(declaration, source_program)?.2)
            }
        };
        let action = self.gadget_inverse(
            gadget_level,
            level,
            self.ring(gadget_level)?.polynomial([IntExpr::constant(weight.clone())]),
        )?;
        Ok((action, gadget_level, weight))
    }

    /// The level `s` of the final smudge LUT input of `declaration`: its
    /// smudge chain has `s + 2` levels, the last being the decoding level.
    pub fn decoder_smudge_level(declaration: &DecoderDeclaration) -> Result<usize, RnsBggError> {
        RnsBggDerivedWorkload::from_program(declaration.smudge().program())?
            .level(declaration.smudge().output())
    }

    /// The smudge chain of `declaration`'s decoder of `source_program`: `s + 2`
    /// levels `j` whose moduli are the first `depth(L) + primes_per_drop *
    /// ceil((s + 1 - j) / operations_per_drop)` primes of `pool`, where
    /// `depth(L)` is the CRT depth of the decoding level `L` and `pool` lists
    /// the primes of `q_L` first. The last level is `q_L` itself and the step
    /// into it is always a drop, so every smudge LUT, including the final
    /// coefficient LUT, divides its noise at a cadence boundary.
    pub fn decoder_smudge_parameters(
        &self,
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
        pool: &DCRTPolyParams,
        primes_per_drop: usize,
    ) -> Result<Self, RnsBggError> {
        let level = Self::decoder_source_level(declaration, source_program)?;
        let decoding = self.level(level)?;
        let smudge_level = Self::decoder_smudge_level(declaration)?;
        if self.rhs_mode == RhsMode::Dynamic {
            // The smudge is evaluated at the source modulus itself, with a
            // gadget successor after its normalized members.
            let smudge = Self::new(
                vec![decoding.clone(); smudge_level + 3],
                self.encoding_error_sigma.clone(),
                self.operations_per_drop,
            )?
            .with_rhs_mode(RhsMode::Dynamic);
            self.validate_decoder_smudge_parameters(&smudge, declaration, source_program)?;
            return Ok(smudge);
        }
        let decoding_primes = decoding.to_crt().0;
        let primes = pool.to_crt().0;
        if primes_per_drop == 0 || !primes.starts_with(&decoding_primes) {
            return Err(RnsBggError::Decoder(DecoderError::InvalidModulusRelation));
        }
        let levels = (0..smudge_level + 2)
            .map(|j| {
                let drops = (smudge_level + 1 - j).div_ceil(self.operations_per_drop);
                let depth = decoding_primes.len() + drops * primes_per_drop;
                let modulus = primes
                    .get(..depth)
                    .ok_or(RnsBggError::Decoder(DecoderError::InvalidModulusRelation))?
                    .iter()
                    .copied()
                    .map(BigUint::from)
                    .product();
                pool.select_modulus(&modulus)
                    .ok_or(RnsBggError::Decoder(DecoderError::InvalidModulusRelation))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let smudge =
            Self::new(levels, self.encoding_error_sigma.clone(), self.operations_per_drop)?
                .with_rhs_mode(self.rhs_mode);
        self.validate_decoder_smudge_parameters(&smudge, declaration, source_program)?;
        Ok(smudge)
    }

    /// Check that `smudge` is a smudge chain of `declaration`'s decoder of
    /// `source_program` at these parameters, and that the decoder has a
    /// valid plaintext modulus (see [`Self::decoder_plaintext`]). A Static
    /// chain has `s + 2` levels of non-increasing moduli, each a CRT
    /// sub-basis of the one before, with a strict drop into the last level,
    /// which is the decoding level `L` with its CRT basis. A Dynamic chain has
    /// `s + 3` levels, every one the decoding level's basis. Source and smudge
    /// share the encoding sigma and RHS mode, and the declaration must smudge
    /// every coefficient of the decoding ring.
    pub(crate) fn validate_decoder_smudge_parameters(
        &self,
        smudge: &Self,
        declaration: &DecoderDeclaration,
        source_program: &crate::program::ExponentLutProgram,
    ) -> Result<(), RnsBggError> {
        let relation = RnsBggError::Decoder(DecoderError::InvalidModulusRelation);
        let level = Self::decoder_source_level(declaration, source_program)?;
        let decoding = self.level(level)?;
        self.decoder_plaintext(declaration, source_program)?;
        // Every coefficient of the decoded row carries noise, so every
        // coefficient is smudged.
        if declaration.coefficients() != decoding.ring_dimension() as usize {
            return Err(RnsBggError::Decoder(DecoderError::InvalidSmudgeProgram));
        }
        let smudge_level = Self::decoder_smudge_level(declaration)?;
        if smudge.encoding_error_sigma != self.encoding_error_sigma ||
            smudge.rhs_mode != self.rhs_mode
        {
            return Err(relation);
        }
        if self.rhs_mode == RhsMode::Dynamic {
            if smudge.levels.len() != smudge_level + 3 ||
                smudge.levels.iter().any(|level| !compatible_level_parameters(level, decoding))
            {
                return Err(relation);
            }
            return Ok(());
        }
        let plaintext = self
            .level(level + 1)
            .map_err(|_| RnsBggError::Decoder(DecoderError::InvalidModulusRelation))?;
        let levels = smudge.levels();
        let depth = |params: &DCRTPolyParams| params.to_crt().0.len();
        if plaintext.modulus() == decoding.modulus() ||
            levels.len() != smudge_level + 2 ||
            !compatible_level_parameters(&levels[smudge_level + 1], decoding) ||
            depth(&levels[smudge_level]) <= depth(decoding)
        {
            return Err(relation);
        }
        for pair in levels.windows(2) {
            if !compatible_level_parameters_except_basis(&pair[0], &pair[1]) ||
                !pair[0].to_crt().0.starts_with(&pair[1].to_crt().0)
            {
                return Err(relation);
            }
        }
        Ok(())
    }

    /// `2^-1 w mod q_level` for every weight numerator `w`: `q_level` is a
    /// product of odd primes, so 2 is invertible.
    pub(crate) fn halved_weights(
        &self,
        level: usize,
        numerators: &[BigUint],
    ) -> Result<Vec<BigUint>, RnsBggError> {
        let modulus: BigUint = self.level(level)?.modulus().as_ref().clone();
        let half = (&modulus + 1u8) / 2u8;
        Ok(numerators.iter().map(|numerator| numerator * &half % &modulus).collect())
    }
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
    primes_per_drop: usize,
    max_crt_depth: usize,
    terminal_depth: usize,
) -> Result<usize, RnsBggError> {
    if operations_per_drop == 0 ||
        primes_per_drop == 0 ||
        terminal_depth == 0 ||
        max_crt_depth < terminal_depth
    {
        return Err(RnsBggError::Parameters("invalid CRT drop cadence"));
    }
    let dropped = (logical_level / operations_per_drop).saturating_mul(primes_per_drop);
    Ok(max_crt_depth.saturating_sub(dropped).max(terminal_depth))
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

/// Where the public key of a Program input comes from: Static keys are
/// sampled from the hash key per wire and logical level, Dynamic keys per
/// setup input coordinate (see [`RnsBggDynamicSetup::input_key`]), so they
/// do not depend on the Program.
#[derive(Clone)]
pub enum InputKeySource {
    Static { level: usize, knowledge: crate::program::AttributeKnowledge },
    Dynamic { setup: std::sync::Arc<RnsBggDynamicSetup>, coordinate: u32 },
}

impl InputKeySource {
    /// The key of input `wire`, or of its member `member`.
    pub fn key(
        &self,
        parameters: &RnsBggParameters,
        hash_key: mxx_dsl::Bytes,
        wire: WireId,
        member: Option<IntExpr>,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        match self {
            Self::Static { level, knowledge } => {
                public_key::sampled_key(parameters, *level, hash_key, wire, *knowledge, member)
            }
            Self::Dynamic { setup, coordinate } => Ok(RnsBggPublicKey::new(
                wire,
                setup.input_key(parameters, hash_key, *coordinate, member)?,
            )),
        }
    }
}

/// The sole descriptor shared by RNS-BGG stage checkpoints.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct RnsBggArtifactBinding {
    context: [u8; 32],
    /// The Program of a preprocessing stage, and of a Static encryption; a
    /// Dynamic encryption belongs to its setup only.
    program: Option<ExponentLutProgramId>,
    parameters: [u8; 32],
    hash_key_commitment: [u8; 32],
    sources: Vec<mxx_ir_core::artifact::ProductionId>,
    /// The applied decoder: its declaration commitment and the coefficient
    /// bound of its unmask preimage `K`.
    decoder: Option<([u8; 32], BigUint)>,
    /// The logical level of each Program input, which encryption encodes it
    /// at.
    input_levels: BTreeMap<WireId, usize>,
    /// RHS timing mode of the bound stages.
    rhs_mode: RhsMode,
    /// Dynamic only: the identity of the setup the stage belongs to.
    dynamic_setup: Option<[u8; 32]>,
    /// Dynamic preprocessing only: the mapping of the Program's inputs and
    /// RHS to setup coordinates.
    dynamic_binding: Option<RnsBggDynamicBinding>,
    /// Dynamic encryption only: the prelude production holding every RHS
    /// family's ciphertexts, and its manifest.
    ciphertext_prelude:
        Option<(mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest)>,
    /// Preprocessing only: how the Program's outputs are consumed, and the
    /// resolved carrier of each declared output.
    output_use: RnsOutputUse,
    output_carriers: Vec<RnsOutputCarrier>,
}

impl RnsBggArtifactBinding {
    pub(crate) fn new(
        context: [u8; 32],
        program: Option<ExponentLutProgramId>,
        parameters: [u8; 32],
        hash_key_commitment: [u8; 32],
        sources: Vec<mxx_ir_core::artifact::ProductionId>,
        rhs_mode: RhsMode,
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
            decoder: None,
            ciphertext_prelude: None,
            input_levels: BTreeMap::new(),
            rhs_mode,
            dynamic_setup: None,
            dynamic_binding: None,
            output_use: RnsOutputUse::default(),
            output_carriers: Vec::new(),
        };
        for source in sources {
            binding = binding.with_source(source)?;
        }
        Ok(binding)
    }

    /// Records the logical level of each Program input.
    pub(crate) fn with_input_levels(mut self, input_levels: BTreeMap<WireId, usize>) -> Self {
        self.input_levels = input_levels;
        self
    }

    /// The logical level of each Program input.
    pub(crate) fn input_levels(&self) -> &BTreeMap<WireId, usize> {
        &self.input_levels
    }

    /// Records the Dynamic setup of the stage and, for preprocessing, the
    /// Program's mapping to its coordinates.
    pub(crate) fn with_dynamic_setup(
        mut self,
        setup: [u8; 32],
        binding: Option<RnsBggDynamicBinding>,
    ) -> Self {
        self.dynamic_setup = Some(setup);
        self.dynamic_binding = binding;
        self
    }

    pub(crate) fn rhs_mode(&self) -> RhsMode {
        self.rhs_mode
    }

    /// Records the output use and the resolved carrier of every declared
    /// output.
    pub(crate) fn with_outputs(
        mut self,
        output_use: RnsOutputUse,
        carriers: Vec<RnsOutputCarrier>,
    ) -> Self {
        self.output_use = output_use;
        self.output_carriers = carriers;
        self
    }

    /// How the Program's outputs are consumed.
    pub(crate) fn output_use(&self) -> RnsOutputUse {
        self.output_use
    }

    /// The resolved carrier of every declared output.
    pub(crate) fn output_carriers(&self) -> &[RnsOutputCarrier] {
        &self.output_carriers
    }

    /// The identity of the stage's Dynamic setup.
    pub(crate) fn dynamic_setup(&self) -> Option<&[u8; 32]> {
        self.dynamic_setup.as_ref()
    }

    /// The Program's mapping to setup coordinates of a Dynamic
    /// preprocessing stage.
    pub(crate) fn dynamic_binding(&self) -> Option<&RnsBggDynamicBinding> {
        self.dynamic_binding.as_ref()
    }

    /// Binds a decoder declaration to this stage's artifact identity.
    pub(crate) fn with_decoder(
        mut self,
        declaration: &DecoderDeclaration,
        preimage_bound: BigUint,
    ) -> Self {
        self.decoder = Some((*declaration.commitment(), preimage_bound));
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

    pub(crate) fn program(&self) -> Option<ExponentLutProgramId> {
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

    pub(crate) fn with_ciphertext_prelude(
        mut self,
        production: mxx_ir_core::artifact::ProductionId,
        manifest: mxx_ir_core::artifact::Manifest,
    ) -> Self {
        self.ciphertext_prelude = Some((production, manifest));
        self
    }

    /// The Dynamic encryption prelude holding the RHS family ciphertexts.
    pub(crate) fn ciphertext_prelude(
        &self,
    ) -> Option<&(mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest)> {
        self.ciphertext_prelude.as_ref()
    }

    pub(crate) fn decoder_commitment(&self) -> Option<&[u8; 32]> {
        self.decoder.as_ref().map(|(commitment, _)| commitment)
    }

    /// The coefficient bound of the applied decoder's unmask preimage.
    pub(crate) fn unmask_preimage_bound(&self) -> Option<&BigUint> {
        self.decoder.as_ref().map(|(_, bound)| bound)
    }
}

/// A stage descriptor is deliberately distinct at the type level even though
/// all three RNS-BGG checkpoints carry the same binding metadata.  The
/// transparent wrapper keeps the serialized representation owned by the
/// binding while preventing a preprocess, encryption, or online evaluation checkpoint
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

rns_bgg_stage_descriptor!(RnsBggPreprocessOnlineEvalDescriptor);

rns_bgg_stage_descriptor!(RnsBggEncryptionDescriptor);

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

/// One-shot decoder invocation shared by the stage lowerers: the source
/// carrier of the decoded output, lowered by the outer stage.
pub(crate) struct RnsBggDecoder<'a, E: ExponentLUTProgramLowering> {
    pub(crate) lowerer: &'a mut E,
    pub(crate) source: E::Wire,
}

/// The member transform of the decoder smudge: member `member` of the smudge
/// Program, whose finalized carrier at `level` has payload `v r G_level`, is
/// right-multiplied by `G_level^-1(w)` for its halved weight `w = 2^-1 P^digit
/// mod q_level` and then by `X^coefficient`. Returns that action and that
/// monomial; both are graph values chosen per member, built without
/// loop-dependent constants.
pub(crate) fn smudge_member_action(
    parameters: &RnsBggParameters,
    declaration: &DecoderDeclaration,
    member: IntExpr,
    level: usize,
) -> Result<(SmallMatrix, mxx_dsl::Int), RnsBggError> {
    let weights = parameters.halved_weights(level, &declaration.member_weights())?;
    let (coefficient, index) = declaration.member_placement(member);
    let row = parameters.weight_row(level, &weights)?;
    let weight = if weights.len() == 1 { row } else { row_entry(&row, index) };
    Ok((parameters.gadget_inverse(level, level, weight)?, mxx_dsl::Int::evaluate(coefficient)))
}

/// `X^coefficient * entry * action` for one carrier entry.
pub(crate) fn act_on_entry(entry: Mat, action: &SmallMatrix, coefficient: &mxx_dsl::Int) -> Mat {
    action.clone().right_multiply(entry).multiply_monomial(coefficient.clone())
}

fn map_row(row: RnsBggRow, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggRow {
    (operation(row.0), operation(row.1))
}

fn map_matrix(matrix: RnsBggMatrix, mut operation: impl FnMut(Mat) -> Mat) -> RnsBggMatrix {
    ((operation(matrix.0.0), operation(matrix.0.1)), (operation(matrix.1.0), operation(matrix.1.1)))
}

/// The nonzero coefficients of `value`, centered into `(-q/2, q/2]`.
fn centered_terms<P: mxx_backends::poly::Poly>(value: &P) -> Vec<(usize, BigInt)> {
    use mxx_backends::element::PolyElem;
    value
        .coeffs()
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| !coefficient.value().is_zero())
        .map(|(index, coefficient)| {
            let modulus: std::sync::Arc<BigUint> = coefficient.modulus().clone().into();
            let value = BigInt::from(coefficient.value().clone());
            let centered = if coefficient.value() > &(modulus.as_ref() >> 1usize) {
                value - BigInt::from(modulus.as_ref().clone())
            } else {
                value
            };
            (index, centered)
        })
        .collect()
}

/// The coefficient of a constant polynomial with these terms.
fn constant_term(terms: &[(usize, BigInt)]) -> Option<BigInt> {
    match terms {
        [] => Some(BigInt::zero()),
        [(0, coefficient)] => Some(coefficient.clone()),
        _ => None,
    }
}

/// The position `k < n` and sign of a signed monomial `+-X^k` with these
/// terms; `true` marks `-X^k`.
fn monomial_term(terms: &[(usize, BigInt)]) -> Option<(usize, bool)> {
    match terms {
        [(position, coefficient)] if coefficient.is_one() => Some((*position, false)),
        [(position, coefficient)] if (-coefficient).is_one() => Some((*position, true)),
        _ => None,
    }
}

/// A graph constant with these terms, sized by them rather than by the ring
/// dimension where it can be: a constant polynomial keeps one coefficient and a
/// signed monomial is a (negated) rotation. Any other polynomial keeps its
/// coefficients up to its last nonzero one.
fn constant_from_terms(ring: &Ring, terms: &[(usize, BigInt)]) -> Mat {
    if let Some(coefficient) = constant_term(terms) {
        return ring.polynomial([IntExpr::constant(coefficient)]);
    }
    if let Some((position, negated)) = monomial_term(terms) {
        let rotation = ring.constant(
            (1, 1),
            mxx_ir_core::node::ConstantMatrix::Rotation { exponent: IntExpr::constant(position) },
        );
        return if negated { -rotation } else { rotation };
    }
    let length = terms.last().map_or(0, |(position, _)| position + 1);
    let mut coefficients = vec![IntExpr::constant(0); length];
    for (position, coefficient) in terms {
        coefficients[*position] = IntExpr::constant(coefficient.clone());
    }
    ring.polynomial(coefficients)
}

/// A caller's plaintext polynomial as a graph constant over `ring`, sized by
/// its nonzero terms (see `constant_from_terms`).
pub(crate) fn plaintext_constant<P: mxx_backends::poly::Poly>(ring: &Ring, value: &P) -> Mat {
    constant_from_terms(ring, &centered_terms(value))
}

/// The signed monomial `c * X^k` of a caller's plaintext, as `(c, k)` with
/// `c` centered and `0 <= k < n`; zero is `(0, 0)`. A plaintext with more
/// than one nonzero coefficient is refused.
fn plaintext_monomial<P: mxx_backends::poly::Poly>(value: &P) -> Result<(BigInt, usize), String> {
    match centered_terms(value).as_slice() {
        [] => Ok((BigInt::zero(), 0)),
        [(exponent, coefficient)] => Ok((coefficient.clone(), *exponent)),
        _ => Err("a plaintext must have at most one nonzero coefficient".into()),
    }
}

/// A plaintext family read member by member inside loop bodies, so it is
/// never a resident family of full matrices: member `i` is
/// `coefficients[i] * X^(exponents[i])`.
#[derive(Clone)]
pub(crate) struct PlaintextFamily {
    coefficients: Family<mxx_dsl::Int>,
    exponents: Family<mxx_dsl::Int>,
}

impl PlaintextFamily {
    pub(crate) fn count(&self) -> &IntExpr {
        self.coefficients.count()
    }

    /// Member `index`, built where it is read in the ring `ring`.
    pub(crate) fn member(&self, ring: &Ring, index: mxx_dsl::Int) -> Mat {
        signed_monomial(ring, self.coefficients.at(index.clone()), self.exponents.at(index))
    }
}

fn signed_monomial(ring: &Ring, coefficient: mxx_dsl::Int, exponent: mxx_dsl::Int) -> Mat {
    coefficient.lift_to_constant_polynomial(ring.matrix_type((1, 1))).multiply_monomial(exponent)
}

/// Bind a caller's plaintext, or plaintext family, as graph integer inputs:
/// the plaintext `c * X^k` is read from the inputs `{name}.coefficient` and
/// `{name}.exponent`, whose values are inserted into `runtime`. A stage graph
/// therefore does not depend on its plaintexts, and an exported claim
/// quantifies over them. Callers make `name` unique per binding, as they do
/// for secrets, so stages with different plaintexts have distinct
/// specifications.
pub(crate) fn plaintext_inputs<P: mxx_backends::poly::Poly>(
    ring: &Ring,
    name: &str,
    value: &InputBinding<P, Vec<P>>,
    runtime: &mut BTreeMap<String, mxx_backends::RuntimeValue>,
) -> Result<InputBinding<Mat, PlaintextFamily>, String> {
    use mxx_backends::RuntimeValue;
    use rayon::prelude::*;
    let (coefficient_name, exponent_name) =
        (format!("{name}.coefficient"), format!("{name}.exponent"));
    let context = mxx_dsl::DslContext::new("plaintext");
    Ok(match value {
        InputBinding::Scalar(value) => {
            let (coefficient, exponent) = plaintext_monomial(value)?;
            runtime.insert(coefficient_name.clone(), RuntimeValue::Int(coefficient));
            runtime.insert(exponent_name.clone(), RuntimeValue::Int(BigInt::from(exponent)));
            let input = |name: String| {
                context.input::<mxx_dsl::Int>(name, mxx_dsl::IntType).map_err(|e| e.to_string())
            };
            InputBinding::Scalar(signed_monomial(
                ring,
                input(coefficient_name)?,
                input(exponent_name)?,
            ))
        }
        InputBinding::Family(values) => {
            let (coefficients, exponents): (Vec<_>, Vec<_>) = values
                .par_iter()
                .map(plaintext_monomial)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|(coefficient, exponent)| (coefficient, BigInt::from(exponent)))
                .unzip();
            runtime.insert(coefficient_name.clone(), RuntimeValue::integer_values(coefficients));
            runtime.insert(exponent_name.clone(), RuntimeValue::integer_values(exponents));
            InputBinding::Family(PlaintextFamily {
                coefficients: context.int_family_input(coefficient_name, values.len()),
                exponents: context.int_family_input(exponent_name, values.len()),
            })
        }
    })
}

/// A random name prefix for one binding's plaintext inputs.
pub(crate) fn plaintext_input_prefix() -> String {
    format!("rns-bgg/plaintext/{:032x}", rand::random::<u128>())
}

pub(crate) fn concrete_ring(value: &Mat) -> Result<ConcreteRing, RnsBggError> {
    resolve_ring(&value.matrix_type().ring)
}

pub(crate) fn resolve_ring(ring: &mxx_ir_core::RingRef) -> Result<ConcreteRing, RnsBggError> {
    ring.resolve(&mxx_ir_core::ParamEnv::default())
        .map_err(|_| RnsBggError::Parameters("matrix ring must be concrete"))
}

/// Convert a matrix between concrete CRT rings without constructing an
/// invalid modulus-reduction node.  Reduction is only valid when the
/// destination basis is a strict CRT subset of the source; a destination
/// that contains the source modulus is reached by centered rebasing, and
/// equal rings are preserved as-is.
pub(crate) fn checked_modulus_convert(value: Mat, destination: &Ring) -> Result<Mat, RnsBggError> {
    let source = concrete_ring(&value)?;
    let target = resolve_ring(destination.as_ref())?;
    if source == target {
        return Ok(value);
    }
    if target.crt_moduli().iter().all(|prime| source.crt_moduli().contains(prime)) &&
        target.crt_depth() < source.crt_depth()
    {
        return Ok(value.reduce_modulus(destination));
    }
    if (target.modulus() % source.modulus()).is_zero() {
        return Ok(value.centered_rebase(destination));
    }
    Err(RnsBggError::Parameters("destination modulus must divide or contain the source"))
}

pub(crate) fn checked_modulus_convert_row(
    row: RnsBggRow,
    destination: &Ring,
) -> Result<RnsBggRow, RnsBggError> {
    Ok((checked_modulus_convert(row.0, destination)?, checked_modulus_convert(row.1, destination)?))
}

pub(crate) fn checked_modulus_convert_matrix(
    matrix: RnsBggMatrix,
    destination: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    Ok((
        (
            checked_modulus_convert(matrix.0.0, destination)?,
            checked_modulus_convert(matrix.0.1, destination)?,
        ),
        (
            checked_modulus_convert(matrix.1.0, destination)?,
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

/// The ciphertext `C = [top; target - prefix * top + e]` at `level`, with a
/// fresh error row `e` of `columns` columns sampled under `contract`, so
/// `(prefix, 1) C = target + e`. RHS ciphertexts have this form.
pub(crate) fn ciphertext_matrix(
    parameters: &RnsBggParameters,
    level: usize,
    columns: usize,
    prefix: Mat,
    top: RnsBggRow,
    target: RnsBggRow,
    contract: &RnsBggGaussian,
) -> Result<RnsBggMatrix, RnsBggError> {
    let error = contract.sample_row(&parameters.ring(level)?, columns)?;
    let bottom = (
        target.0 - prefix.clone() * top.0.clone() + error.0,
        target.1 - prefix * top.1.clone() + error.1,
    );
    Ok((top, bottom))
}

/// Artifact role of a Dynamic root site's public action in the
/// preprocessing production: `part` is `key` (the site's output key, ports
/// `.0` to `.3`) or `branches` (a LUT's product keys `T_k`, every switched
/// branch side by side, ports `.0` and `.1`).
pub(crate) fn public_action_role(wire: WireId, part: &str) -> String {
    format!("public-action/{wire:?}/{part}")
}

/// The public key `B_{a,j}` of digit row `a` of output column `j` of the
/// package named by `tag` (see [`RnsBggDynamicSetup::package_tag`]): a
/// `2 x M` key at `level`, as four `1 x D` blocks, since a digit is a
/// public attribute encoded under the full public gadget.
pub(crate) fn companion_key(
    parameters: &RnsBggParameters,
    level: usize,
    hash_key: mxx_dsl::Bytes,
    tag: &mxx_dsl::HashTag,
    a: IntExpr,
    j: IntExpr,
) -> Result<RnsBggMatrix, RnsBggError> {
    let mut tag = tag.clone();
    tag.push("column");
    tag.push(j);
    tag.push("digit-row");
    tag.push(a);
    Ok(public_key::matrix_from_hash(
        &parameters.ring(level)?,
        hash_key,
        tag,
        parameters.level(level)?.modulus_digits(),
    ))
}

/// The `1 x width` unit row at a loop-dependent `index`: a window of one
/// constant row, so no constant depends on the loop.
pub(crate) fn unit_row(ring: &Ring, width: usize, index: IntExpr) -> Mat {
    let start = (IntExpr::constant(width - 1) - index).canonicalize();
    ring.constant(
        (1, 2 * width - 1),
        mxx_ir_core::node::ConstantMatrix::UnitRow { index: IntExpr::constant(width - 1) },
    )
    .slice(
        None,
        Some(mxx_ir_core::node::IndexRange {
            start: start.clone(),
            end: (start + IntExpr::constant(width)).canonicalize(),
        }),
    )
}

/// Entry `index` of a `1 x n` row.
pub(crate) fn row_entry(row: &Mat, index: IntExpr) -> Mat {
    row.clone().slice(
        None,
        Some(mxx_ir_core::node::IndexRange {
            start: index.clone(),
            end: (index + IntExpr::constant(1)).canonicalize(),
        }),
    )
}

/// The two blocks of a `1 x 2c` row.
pub(crate) fn split_row(row: &Mat, columns: usize) -> RnsBggRow {
    let block = |start: usize| {
        row.clone().slice(
            None,
            Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(start),
                end: IntExpr::constant(start + columns),
            }),
        )
    };
    (block(0), block(columns))
}

/// A row's two blocks as one row.
pub(crate) fn join_row(row: &RnsBggRow) -> Mat {
    Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![row.0.clone(), row.1.clone()])
}

/// The pre-rounding keys `T_s` and companion rows `E_s` of the Dynamic
/// products of `branches` state keys `A_s` at `level` (spec Eq. 049, 086,
/// 087): for every branch `s`, output column `j < m_(i+1)`, and digit row
/// `a < m_i`,
///
/// `T_{s,j} = sum_a B_{s,a,j} U_{s,:,a}` and `E_{s,j} = sum_a z_{s,a,j} U_{s,:,a}`,
///
/// with `U_s = Dec(A_s)`. `key_entry(s, r, a)` is entry `(r, a)` of
/// `A_s`, `companion_key(s, a, j)` the key `B_{s,a,j}`, and `companion(s, a,
/// j)` the encoding `z = s B - d s G + eps` of digit `d = Dec(C_s)_{a,j}`.
/// Since `G U_s = A_s`, `E_s = s T_s - s A_s Dec(C_s) + eps U_s`, and for
/// `c_s = s (A_s - mu G) + e` and `s C_s = nu s G + e_C`,
/// `E_s + c_s Dec(C_s) = s (T_s - mu nu G) + eps U_s - mu e_C + e Dec(C_s)`.
/// `T_s` depends only on public keys. The caller adds `c_s Dec(C_s)` (see
/// [`column_action`]).
///
/// An outer loop visits every `(s, a)` and decomposes column `a` of `A_s`
/// once; its inner loop visits every `j`, holding one key and two
/// companions (the current one and the next one, read ahead), and
/// accumulates the `1 x m_(i+1)` block of branch `s`, which the outer body
/// then places. Both loops read companions, so on a GPU both are host-driven
/// and the inner one reads each companion one iteration ahead. The graph and
/// the working set are linear in `M_i + m_i + branches m_(i+1)`. Returns, given
/// `companion_key`, the key rows and, given companions, the companion row, each
/// `1 x branches m_(i+1)` with branch `s` at columns `s m_(i+1) ..`. An online
/// evaluation that imports `T_s` from preprocessing (its public actions) passes no
/// `companion_key`, so it expands no `B_{s,a,j}` and accumulates the companion
/// row alone.
pub(crate) fn companion_columns(
    parameters: &RnsBggParameters,
    level: usize,
    branches: usize,
    output_columns: usize,
    key_entry: &dyn Fn(IntExpr, usize, IntExpr) -> Result<Mat, RnsBggError>,
    companion_key: Option<&dyn Fn(IntExpr, IntExpr, IntExpr) -> Result<RnsBggMatrix, RnsBggError>>,
    companion: Option<&dyn Fn(IntExpr, IntExpr, IntExpr) -> Result<RnsBggRow, RnsBggError>>,
) -> Result<Vec<Mat>, RnsBggError> {
    if companion_key.is_none() && companion.is_none() {
        return Err(RnsBggError::Binding("a companion loop accumulates no row"));
    }
    let ring = parameters.ring(level)?;
    let current = parameters.level(level)?;
    let base = public_gadget_base(current.base_bits())?;
    let digits = current.modulus_digits();
    let digit_rows = 2 * parameters.component_columns(level)?;
    let block = 2 * output_columns;
    let schema = |_| mxx_dsl::DslError::Schema;
    let sums = 2 * usize::from(companion_key.is_some()) + usize::from(companion.is_some());
    let rows = vec![ring.zero((1, branches * block)); sums];
    Ok(mxx_dsl::iterate(branches * digit_rows, rows, |pair, rows: Vec<Mat>| {
        let pair = Box::new(pair.expression()?);
        let digit_rows = Box::new(IntExpr::constant(digit_rows));
        let branch = IntExpr::FloorDiv(pair.clone(), digit_rows.clone()).canonicalize();
        let a = IntExpr::Rem(pair, digit_rows).canonicalize();
        let [u0, u1] = [0, 1].map(|r| {
            key_entry(branch.clone(), r, a.clone()).map(|entry| entry.decompose(base, digits))
        });
        let (u0, u1) = (u0.map_err(schema)?, u1.map_err(schema)?);
        let act = |(left, right): RnsBggRow| {
            left.mul_small_rhs(u0.clone()) + right.mul_small_rhs(u1.clone())
        };
        let read = |j: IntExpr| {
            companion.map(|companion| companion(branch.clone(), a.clone(), j)).transpose()
        };
        // Each inner iteration reads the next companion (the first again
        // after the last) and carries it, so that load overlaps this
        // iteration's work.
        let mut initial = vec![ring.zero((1, block)); sums];
        if let Some((left, right)) = read(IntExpr::constant(0)).map_err(schema)? {
            initial.extend([left, right]);
        }
        let blocks = mxx_dsl::iterate(block, initial, |j, mut carried: Vec<Mat>| {
            let j = j.expression()?;
            let next = IntExpr::Rem(
                Box::new(j.clone() + IntExpr::constant(1)),
                Box::new(IntExpr::constant(block)),
            )
            .canonicalize();
            let mut terms = Vec::with_capacity(sums);
            if let Some(companion_key) = companion_key {
                let (top, bottom) =
                    companion_key(branch.clone(), a.clone(), j.clone()).map_err(schema)?;
                terms.extend([act(top), act(bottom)]);
            }
            if let Some((left, right)) = read(next).map_err(schema)? {
                let current = carried.split_off(sums);
                terms.push(act((current[0].clone(), current[1].clone())));
                carried.extend([left, right]);
            }
            let unit = unit_row(&ring, block, j);
            for (row, term) in carried.iter_mut().zip(terms) {
                *row = row.clone() + term * unit.clone();
            }
            Ok(carried)
        })?;
        let place = unit_row(&ring, branches, branch);
        Ok(rows
            .into_iter()
            .zip(blocks)
            .map(|(row, block)| row + place.clone().tensor(block))
            .collect())
    })?)
}

/// `row Dec(matrix)` for every row of `rows`, one column at a time: column
/// `j` of `matrix` (a `2 x m_(i+1)` matrix at `decomposition_level` whose
/// components have `columns` columns each) is
/// sliced before it is decomposed, so only its `m_i` digits are held, never
/// the whole `m_i x m_(i+1)` decomposition. Each row is `1 x m_i` at
/// `evaluation_level`; the results are `1 x m_(i+1)`.
pub(crate) fn column_action(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    matrix: &RnsBggMatrix,
    columns: usize,
    rows: Vec<Mat>,
) -> Result<Vec<Mat>, RnsBggError> {
    let ring = parameters.ring(evaluation_level)?;
    let decomposition = parameters.level(decomposition_level)?;
    let base = public_gadget_base(decomposition.base_bits())?;
    let digits = decomposition.modulus_digits();
    let width = 2 * columns;
    let [top, bottom] = [&matrix.0, &matrix.1].map(join_row);
    Ok(mxx_dsl::iterate(width, vec![ring.zero((1, width)); rows.len()], |j, sums: Vec<Mat>| {
        let j = j.expression()?;
        let column = Mat::concat(
            mxx_ir_core::node::ConcatAxis::Rows,
            vec![row_entry(&top, j.clone()), row_entry(&bottom, j.clone())],
        )
        .decompose(base, digits)
        .centered_rebase(&ring);
        let unit = unit_row(&ring, width, j);
        Ok(sums
            .into_iter()
            .zip(&rows)
            .map(|(sum, row)| sum + row.clone().mul_small_rhs(column.clone()) * unit.clone())
            .collect())
    })?)
}

/// The compact decomposition of each column of `matrix`, its two entries
/// stacked: `G^-1([m00; m10])` and `G^-1([m01; m11])`. Digits of stacked
/// rows are stacked, so a product with `[a | b]` gives `a u + b w` in one
/// product, with no product of each entry held for a sum.
pub(crate) fn compact_decompose_columns(
    parameters: &RnsBggParameters,
    evaluation_level: usize,
    decomposition_level: usize,
    matrix: RnsBggMatrix,
) -> Result<(SmallMatrix, SmallMatrix), RnsBggError> {
    let evaluation_ring = parameters.ring(evaluation_level)?;
    let evaluation = resolve_ring(evaluation_ring.as_ref())?;
    let decomposition = parameters.level(decomposition_level)?;
    let decomposition_ring = resolve_ring(parameters.ring(decomposition_level)?.as_ref())?;
    for entry in [&matrix.0.0, &matrix.0.1, &matrix.1.0, &matrix.1.1] {
        if concrete_ring(entry)? != decomposition_ring {
            return Err(RnsBggError::Parameters(
                "compact action matrix does not match its decomposition ring",
            ));
        }
    }
    if evaluation.modulus() < decomposition_ring.modulus() {
        return Err(RnsBggError::Parameters(
            "evaluation modulus must contain the decomposition modulus",
        ));
    }
    let base = 1u64
        .checked_shl(decomposition.base_bits())
        .ok_or(RnsBggError::Parameters("decomposition base overflow"))?;
    let count = decomposition.modulus_digits();
    let ((top_left, top_right), (bottom_left, bottom_right)) = matrix;
    let column = |top: Mat, bottom: Mat| {
        Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![top, bottom])
            .decompose(base, count)
            .centered_rebase(&evaluation_ring)
    };
    Ok((column(top_left, bottom_left), column(top_right, bottom_right)))
}

/// `row * action` for every row of `rows`, with the action given as the
/// stacked decompositions of its columns (`compact_decompose_columns`):
/// each column is one product of all rows, both halves side by side.
pub(crate) fn multiply_rows_by_columns<D: RightAction>(
    rows: Vec<RnsBggRow>,
    columns: (D, D),
) -> Vec<RnsBggRow> {
    let heights = rows.iter().map(|(a, _)| a.matrix_type().rows.clone()).collect::<Vec<_>>();
    let (lefts, rights): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    let stack = |values: Vec<Mat>| match <[Mat; 1]>::try_from(values) {
        Ok([value]) => value,
        Err(values) => Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, values),
    };
    let both =
        Mat::concat(mxx_ir_core::node::ConcatAxis::Columns, vec![stack(lefts), stack(rights)]);
    let split = |value: Mat| {
        if heights.len() == 1 {
            return vec![value];
        }
        let mut start = IntExpr::constant(0);
        heights
            .iter()
            .map(|height| {
                let end = (start.clone() + height.clone()).canonicalize();
                let range =
                    mxx_ir_core::node::IndexRange { start: start.clone(), end: end.clone() };
                start = end;
                value.clone().slice(Some(range), None)
            })
            .collect::<Vec<_>>()
    };
    let (first, second) = columns;
    let first = split(first.right_multiply(both.clone()));
    let second = split(second.right_multiply(both));
    first.into_iter().zip(second).collect()
}

/// An encoding's row and public-key matrix under an action given by its
/// stacked column decompositions, as one product per column.
pub(crate) fn multiply_row_and_public_by_columns<D: RightAction>(
    row: RnsBggRow,
    matrix: RnsBggMatrix,
    columns: (D, D),
) -> (RnsBggRow, RnsBggMatrix) {
    let (top, bottom) = matrix;
    let mut rows = multiply_rows_by_columns(vec![row, top, bottom], columns).into_iter();
    let row = rows.next().expect("three rows");
    let top = rows.next().expect("three rows");
    (row, (top, rows.next().expect("three rows")))
}

pub(crate) fn centered_rebase_compact(value: RnsBggDigits, ring: &Ring) -> RnsBggDecomposition {
    (
        (value.0.0.centered_rebase(ring), value.0.1.centered_rebase(ring)),
        (value.1.0.centered_rebase(ring), value.1.1.centered_rebase(ring)),
    )
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
    let aligned = (row.0.mul_small_rhs(action.clone()), row.1.mul_small_rhs(action));
    checked_mod_down_encoding(aligned, &parameters.ring(target_level)?)
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
        (matrix.0.0.mul_small_rhs(action.clone()), matrix.0.1.mul_small_rhs(action.clone())),
        (matrix.1.0.mul_small_rhs(action.clone()), matrix.1.1.mul_small_rhs(action)),
    );
    checked_mod_down_public(aligned, &parameters.ring(target_level)?)
}

/// A bounded right operand of a matrix product.
///
/// Raw gadget digits multiply as a preimage; digits rebased into another
/// ring multiply as an ordinary bounded matrix.
pub(crate) trait RightAction: Clone {
    fn right_multiply(self, lhs: Mat) -> Mat;
}

impl RightAction for Preimage {
    fn right_multiply(self, lhs: Mat) -> Mat {
        lhs.mul_small_rhs(self)
    }
}

impl RightAction for SmallMatrix {
    fn right_multiply(self, lhs: Mat) -> Mat {
        lhs.mul_small_rhs(self)
    }
}

/// `row * rhs` for every row of `rows`, each a pair of blocks. Each right
/// factor multiplies all rows at once, stacked, so every factor has one use:
/// the rows' later consumers never keep a factor, which is as large as the
/// product's column count squared, alive after its products.
pub(crate) fn multiply_rows_action<D: RightAction>(
    rows: Vec<RnsBggRow>,
    rhs: ((D, D), (D, D)),
) -> Vec<RnsBggRow> {
    let ((u, v), (w, x)) = rhs;
    if let [(a, b)] = rows.as_slice() {
        let (a, b) = (a.clone(), b.clone());
        return vec![(
            u.right_multiply(a.clone()) + w.right_multiply(b.clone()),
            v.right_multiply(a) + x.right_multiply(b),
        )];
    }
    let heights = rows.iter().map(|(a, _)| a.matrix_type().rows.clone()).collect::<Vec<_>>();
    let (lefts, rights): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    let left = Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, lefts);
    let right = Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, rights);
    let split = |value: Mat| {
        let mut start = IntExpr::constant(0);
        heights
            .iter()
            .map(|height| {
                let end = (start.clone() + height.clone()).canonicalize();
                let range =
                    mxx_ir_core::node::IndexRange { start: start.clone(), end: end.clone() };
                start = end;
                value.clone().slice(Some(range), None)
            })
            .collect::<Vec<_>>()
    };
    let first = split(u.right_multiply(left.clone()) + w.right_multiply(right.clone()));
    let second = split(v.right_multiply(left) + x.right_multiply(right));
    first.into_iter().zip(second).collect()
}

/// `lhs * rhs` for 2x2 matrices of blocks.
pub(crate) fn multiply_public_action<D: RightAction>(
    lhs: RnsBggMatrix,
    rhs: ((D, D), (D, D)),
) -> RnsBggMatrix {
    let (top, bottom) = lhs;
    let mut rows = multiply_rows_action(vec![top, bottom], rhs).into_iter();
    let top = rows.next().expect("two rows");
    (top, rows.next().expect("two rows"))
}

/// Whether a ModDown of `entry` into `destination` is the identity. Upward
/// switches and destinations outside the source basis are rejected.
fn validate_mod_down_entry(entry: &Mat, destination: &ConcreteRing) -> Result<bool, RnsBggError> {
    let source = concrete_ring(entry)?;
    if &source == destination {
        return Ok(true);
    }
    if destination.crt_depth() >= source.crt_depth() {
        return Err(RnsBggError::Parameters("modulus switch cannot increase modulus"));
    }
    if !destination.crt_moduli().iter().all(|prime| source.crt_moduli().contains(prime)) {
        return Err(RnsBggError::Parameters("destination modulus must divide source"));
    }
    Ok(false)
}

fn checked_mod_down_matrix(
    value: RnsBggMatrix,
    destination: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    let entries = [&value.0.0, &value.0.1, &value.1.0, &value.1.1];
    let source = concrete_ring(entries[0])?;
    if entries[1..]
        .iter()
        .map(|entry| concrete_ring(entry))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|entry| *entry != source)
    {
        return Err(RnsBggError::Parameters("matrix entries have incompatible source moduli"));
    }
    let target = resolve_ring(destination.as_ref())?;
    if validate_mod_down_entry(entries[0], &target)? {
        return Ok(value);
    }
    Ok(map_matrix(value, |entry| entry.block_mod_switch(destination, 1)))
}

fn checked_mod_down_row(value: RnsBggRow, destination: &Ring) -> Result<RnsBggRow, RnsBggError> {
    let source = concrete_ring(&value.0)?;
    if concrete_ring(&value.1)? != source {
        return Err(RnsBggError::Parameters("row entries have incompatible source moduli"));
    }
    let target = resolve_ring(destination.as_ref())?;
    if validate_mod_down_entry(&value.0, &target)? {
        return Ok(value);
    }
    Ok(map_row(value, |entry| entry.block_mod_switch(destination, 1)))
}

/// Checked public-role ModDown.  Equal physical moduli return the original
/// graph nodes; upward conversion and non-exact drops fail before graph
/// construction.
pub(crate) fn checked_mod_down_public(
    value: RnsBggMatrix,
    destination: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    checked_mod_down_matrix(value, destination)
}

/// Checked encoding-role ModDown with the same source/target policy as public
/// matrices.
pub(crate) fn checked_mod_down_encoding(
    value: RnsBggRow,
    destination: &Ring,
) -> Result<RnsBggRow, RnsBggError> {
    checked_mod_down_row(value, destination)
}

/// Existing graph callers use this infallible boundary after their enclosing
/// validation.  Keep the error visible as a panic rather than constructing an
/// invalid upward or non-exact switch node.
pub(crate) fn mod_down_public(value: RnsBggMatrix, destination: &Ring) -> RnsBggMatrix {
    checked_mod_down_public(value, destination)
        .unwrap_or_else(|error| panic!("invalid public modulus switch: {error}"))
}

pub(crate) fn mod_down_encoding(value: RnsBggRow, destination: &Ring) -> RnsBggRow {
    checked_mod_down_encoding(value, destination)
        .unwrap_or_else(|error| panic!("invalid encoding modulus switch: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::GraphValue;
    use mxx_ir_core::IntExpr;
    use num_traits::One;

    /// `k = floor(floor(q/t)/2^b) 2^b` with the largest admissible `b`
    /// (`t 2^b <= q/(32 t)`), so `0 <= r = q - t k < t 2^b` and
    /// `r <= q/(32 t)`; without an admissible `b`, `b = 0`; `t` above `q` is
    /// rejected.
    #[test]
    fn coefficient_output_scale_clears_low_bits_within_the_loss() {
        let wide = DCRTPolyParams::new(16, 3, 17, 4, None, None);
        let narrow = wide.select_modulus(&BigUint::from(wide.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![wide, narrow], 1, 1).unwrap();
        for (level, t) in [(0, 8u64), (0, 64), (1, 8), (1, 256)] {
            let q: BigUint = parameters.level(level).unwrap().modulus().as_ref().clone();
            let (k, r) = parameters.coefficient_output_scale(level, t).unwrap();
            let t_big = BigUint::from(t);
            assert_eq!(&t_big * &k + &r, q);
            let limit = &q / (&t_big * &t_big * 32u8);
            if limit.is_zero() {
                assert_eq!(k, &q / &t_big, "no admissible b: k = floor(q/t)");
                assert!(r < t_big);
            } else {
                let low_bits = limit.bits() - 1;
                assert_eq!(k.trailing_zeros().unwrap_or(0) >= low_bits, true);
                assert!(r < &t_big << low_bits);
                assert!(&r * &t_big * 32u8 <= q);
            }
        }
        let q = parameters.level(1).unwrap().modulus().as_ref().to_u64().unwrap();
        assert!(parameters.coefficient_output_scale(1, q + 1).is_err());
        assert!(parameters.coefficient_output_scale(1, 1).is_err());
    }

    /// The norms of `G^-1(k)` are those of the digits the runtime
    /// decomposition computes, and the digits reconstruct `k`.
    #[test]
    fn gadget_inverse_norms_match_the_runtime_decomposition() {
        use mxx_backends::{
            MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend, poly::Poly,
        };
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let q: BigUint = level.modulus().as_ref().clone();
        let ring = parameters.ring(0).unwrap();
        let columns = level.modulus_digits();
        let (scaled, _) = parameters.coefficient_output_scale(0, 12).unwrap();
        for k in [scaled, &q - 1u8, BigUint::from(12345u32)] {
            let action = parameters.output_action(0, &k).unwrap();
            let digits = action.clone().right_multiply(ring.identity(columns));
            let reconstruction = action.right_multiply(parameters.public_gadget(0).unwrap());
            let graph = mxx_dsl::DslContext::new("gadget-inverse-norms")
                .output("digits", digits)
                .unwrap()
                .output("reconstruction", reconstruction)
                .unwrap()
                .build()
                .unwrap();
            let mut execution = CpuExecution::new(cpu_backend([level.clone()]));
            let mut store = MemoryArtifactStore::default();
            let result = crate::api::execute_graph(
                crate::api::StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &mxx_ir_core::ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let download = |name: &str| {
                <CpuExecution as crate::api::StageExecution<MemoryArtifactStore>>::download_matrix(
                    &execution,
                    &result.outputs[name],
                )
                .unwrap()
            };
            let centered = |value: BigUint| {
                let value = BigInt::from(value);
                let q = BigInt::from(q.clone());
                if &value * 2u8 > q { value - q } else { value }
            };
            let digits = download("digits");
            let (mut l1, mut l2_squared) = (BigUint::zero(), BigUint::zero());
            for row in 0..columns {
                let digit = centered(digits.entry(row, 0).coeffs_biguints()[0].clone());
                l1 += digit.magnitude();
                l2_squared += digit.magnitude() * digit.magnitude();
            }
            assert_eq!(parameters.gadget_inverse_norms(0, &k).unwrap(), (l1, l2_squared));
            assert_eq!(download("reconstruction").entry(0, 0).coeffs_biguints()[0], k);
        }
    }

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
            crate::test_support::modulus_of(action.matrix_type()),
            BigInt::from(high.modulus().as_ref().clone())
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
        let converted = checked_mod_down_public(matrix.clone(), &ring).unwrap();
        assert_eq!(converted.0.0.flatten(), matrix.0.0.flatten());
        let row = (entry.clone(), entry.clone());
        let converted_row = checked_mod_down_encoding(row.clone(), &ring).unwrap();
        assert_eq!(converted_row.0.flatten(), row.0.flatten());

        let low = level.select_modulus(&BigUint::from(level.to_crt().0[0])).unwrap();
        let low_ring = crate::test_support::ring_of(&low);
        let lower_entry = low_ring.input("upward-entry", (1, 1));
        let lower_matrix = (
            (lower_entry.clone(), lower_entry.clone()),
            (lower_entry.clone(), lower_entry.clone()),
        );
        assert!(matches!(
            checked_mod_down_public(lower_matrix, &ring),
            Err(RnsBggError::Parameters("modulus switch cannot increase modulus"))
        ));
        let lower_row = (lower_entry.clone(), lower_entry);
        assert!(matches!(
            checked_mod_down_encoding(lower_row, &ring),
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

    /// Pool-prefix parameters: a 12-prime pool, the decoding level `q_L` of
    /// its first three primes and the plaintext level of its first prime.
    pub(crate) fn decoder_test_chain() -> (DCRTPolyParams, RnsBggParameters) {
        let pool = DCRTPolyParams::new(8, 12, 30, 15, None, None);
        let prefix = |depth: usize| {
            let modulus =
                pool.to_crt().0[..depth].iter().map(|prime| BigUint::from(*prime)).product();
            pool.select_modulus(&modulus).unwrap()
        };
        let parameters = RnsBggParameters::new(vec![prefix(3), prefix(1)], 1, 1).unwrap();
        (pool, parameters)
    }

    /// The smudge chain descends by `primes_per_drop` primes per smudge level
    /// to the decoding level; a repeated decoding level, a source without a
    /// plaintext level below it, or a declaration that does not smudge every
    /// coefficient, is rejected.
    #[test]
    fn decoder_smudge_parameters_descend_to_the_decoding_level_and_check_the_contract() {
        let (pool, parameters) = decoder_test_chain();
        let level = parameters.level(0).unwrap().clone();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(crate::program::AttributeKnowledge::PrivatePayload).unwrap();
        let source = builder.finish(vec![crate::program::Operand::Scalar(input)]).unwrap();
        let (declaration, _) = crate::decoder::decoding::fixture::declaration(&source, input, 8, 1);
        let smudge = parameters.decoder_smudge_parameters(&declaration, &source, &pool, 1).unwrap();
        let smudge_level = RnsBggParameters::decoder_smudge_level(&declaration).unwrap();
        assert_eq!(smudge.levels().len(), smudge_level + 2);
        let depths = smudge.levels().iter().map(|entry| entry.to_crt().0.len()).collect::<Vec<_>>();
        assert_eq!(
            depths,
            (0..smudge_level + 2).map(|j| 3 + smudge_level + 1 - j).collect::<Vec<_>>()
        );
        assert_eq!(smudge.levels().last(), Some(&level));

        let repeated = RnsBggParameters::new(vec![level.clone(); smudge_level + 2], 1, 1).unwrap();
        assert!(matches!(
            parameters.validate_decoder_smudge_parameters(&repeated, &declaration, &source),
            Err(RnsBggError::Decoder(DecoderError::InvalidModulusRelation))
        ));
        let terminal = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        assert!(matches!(
            terminal.decoder_smudge_parameters(&declaration, &source, &pool, 1),
            Err(RnsBggError::Decoder(DecoderError::InvalidModulusRelation))
        ));
        let (partial, _) = crate::decoder::decoding::fixture::declaration(&source, input, 4, 1);
        assert!(matches!(
            parameters.decoder_smudge_parameters(&partial, &source, &pool, 1),
            Err(RnsBggError::Decoder(DecoderError::InvalidSmudgeProgram))
        ));
    }

    #[test]
    fn planned_crt_depth_keeps_three_operation_windows() {
        let depths =
            (0..=7).map(|level| planned_crt_depth(level, 3, 1, 6, 2).unwrap()).collect::<Vec<_>>();
        assert_eq!(depths, vec![6, 6, 6, 5, 5, 5, 4, 4]);
        assert!(planned_crt_depth(0, 0, 1, 6, 2).is_err());
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

    /// A negated monomial plaintext `-X^k` is a valid negated rotation, and
    /// evaluates to `-X^k` together with `X^k` and a general polynomial.
    #[test]
    fn plaintext_constants_keep_negated_monomials_valid() {
        use mxx_backends::poly::{Poly, dcrt::poly::DCRTPoly};
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let ring = crate::test_support::ring_of(&level);
        let monomial = DCRTPoly::const_rotate_poly(&level, 5);
        let cases = [
            ("negated", -monomial.clone()),
            ("positive", monomial.clone()),
            ("general", monomial.clone() + DCRTPoly::const_rotate_poly(&level, 2)),
        ];
        let mut context = mxx_dsl::DslContext::new("plaintext-constants");
        for (name, value) in &cases {
            context = context.output(*name, plaintext_constant(&ring, value)).unwrap();
        }
        let graph = crate::test_support::validate(&context.build().unwrap());
        let result = crate::test_support::execute_cpu(&graph, [level], BTreeMap::new());
        for (name, value) in cases {
            assert_eq!(
                crate::test_support::full(&result.outputs[name]).entry(0, 0),
                value,
                "{name}"
            );
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
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let source_ring = explicit_ring(&primes, 4);
        let small_ring = explicit_ring(&primes[..2], 4);
        let source = source_ring.input("checked-modulus-source", (1, 1));
        let reduced = checked_modulus_convert(source.clone(), &small_ring).unwrap();
        let equal = checked_modulus_convert(source.clone(), &source_ring).unwrap();
        let less_source = small_ring.input("checked-modulus-less", (1, 1));
        let extended = checked_modulus_convert(less_source, &source_ring).unwrap();

        assert_eq!(modulus_of(reduced.matrix_type()), BigInt::from(primes[0]) * primes[1]);
        assert_eq!(equal.flatten(), source.flatten());
        assert_eq!(
            modulus_of(extended.matrix_type()),
            BigInt::from(primes[0]) * primes[1] * primes[2]
        );

        let graph = mxx_dsl::DslContext::new("checked-modulus-conversion")
            .transferred_output("reduced", reduced)
            .unwrap()
            .transferred_output("equal", equal)
            .unwrap()
            .transferred_output("extended", extended)
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
            TypeId::of::<RnsBggPreprocessOnlineEvalDescriptor>()
        );
        assert_ne!(
            TypeId::of::<RnsBggPreprocessEncryptDescriptor>(),
            TypeId::of::<RnsBggEncryptionDescriptor>()
        );
        assert_ne!(
            TypeId::of::<RnsBggPreprocessOnlineEvalDescriptor>(),
            TypeId::of::<RnsBggEncryptionDescriptor>()
        );
    }
}
