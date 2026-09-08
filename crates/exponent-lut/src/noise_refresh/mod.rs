//! Sparse-LWR PRF evaluation and native exponent-encoding noise refresh.
//!
//! This module contains the application-specific layers built on the generic
//! Exponent-LUT program, encoding, public-key, and RHS modules. Setup artifacts,
//! PBC routing, PRF evaluation, value readout, CRT refresh, and their dedicated
//! noise simulation live together because they form one refresh protocol.

pub mod pbc;
pub mod prf;
pub mod refresh;
pub mod setup;
pub mod simulation;

use crate::scheme::ExponentLUTEncodingScheme;
use mxx_dsl::GraphValue;

/// Setup supplies compatible, already-scaled states, masks, fresh errors and
/// bound decoders. Operations below rely on those construction contracts;
/// they do not inspect ciphertext coefficients or recover plaintexts.
///
/// The application owns addition/switch/decoder ordering, homogeneous slot
/// grouping and structural parallelism. A scheme supplies individual
/// cryptographic operations, never an opaque implementation of the protocol.
/// This capability requires a specified decoder; ordinary LUT and terminal
/// support alone does not imply refresh support.
pub trait ExponentLUTRefreshScheme: ExponentLUTEncodingScheme {
    type RefreshRow: GraphValue;
    type RefreshDecoder: GraphValue;
    type RefreshLevel: Clone + Ord;
    type RefreshRecomposition;
    fn add_refresh_rows(&self, lhs: Self::RefreshRow, rhs: Self::RefreshRow) -> Self::RefreshRow;
    fn switch_refresh_row(
        &self,
        row: Self::RefreshRow,
        level: &Self::RefreshLevel,
    ) -> Self::RefreshRow;
    fn apply_refresh_decoder(
        &self,
        switched: Self::RefreshRow,
        decoder: Self::RefreshDecoder,
    ) -> Self::RefreshRow;
    /// Extract a decoded CRT residue. BFV rounds after decoder subtraction;
    /// BGG has already converted its carrier before subtraction.
    fn extract_refresh_residue(
        &self,
        row: Self::RefreshRow,
        level: &Self::RefreshLevel,
    ) -> Self::RefreshRow;
    fn recompose_refresh_rows(
        &self,
        rows: Vec<Self::RefreshRow>,
        preparation: &Self::RefreshRecomposition,
    ) -> Self::Encoding;
    /// Public preprocessing operations never accept the online mask secret.
    /// Setup supplies compatible levels, shapes and sampling contracts.
    type RefreshPublicRow: GraphValue;
    type RefreshTargetScale: GraphValue;
    fn add_refresh_public(
        &self,
        lhs: Self::RefreshPublicRow,
        rhs: Self::RefreshPublicRow,
    ) -> Self::RefreshPublicRow;
    fn reduce_refresh_public(
        &self,
        value: Self::RefreshPublicRow,
        level: &Self::RefreshLevel,
    ) -> Self::RefreshPublicRow;
    fn switch_refresh_public(
        &self,
        value: Self::RefreshPublicRow,
        level: &Self::RefreshLevel,
    ) -> Self::RefreshPublicRow;
    fn scale_refresh_public(
        &self,
        value: Self::RefreshPublicRow,
        scale: Self::RefreshTargetScale,
    ) -> Self::RefreshPublicRow;
    fn subtract_refresh_public(
        &self,
        lhs: Self::RefreshPublicRow,
        rhs: Self::RefreshPublicRow,
    ) -> Self::RefreshPublicRow;
}

/// Optional witness-based preparation. Two-column direct decoders do not
/// implement this capability and never need a dummy anchor or preimage.
pub trait ExponentLUTRefreshWitnessScheme: ExponentLUTRefreshScheme {
    type RefreshScaling;
    type RefreshDecoderBase: Clone;
    type RefreshDecoderWitness: GraphValue;

    /// Online preparation applies a preprocessing-produced scaling witness.
    fn scale_refresh_input(
        &self,
        state: &Self::Encoding,
        scaling: &Self::RefreshScaling,
    ) -> Self::RefreshRow;
    /// Combine an online decoder base with its fixed public witness.
    fn prepare_refresh_decoder(
        &self,
        base: Self::RefreshDecoderBase,
        witness: Self::RefreshDecoderWitness,
    ) -> Self::RefreshDecoder;

    type RefreshPublicScaling: GraphValue;

    /// Apply a persisted route witness to an online row. Public and online
    /// routing consume the same witness; neither reconstructs its target.
    fn route_refresh_row(
        &self,
        row: Self::RefreshRow,
        route: Self::RefreshPublicScaling,
    ) -> Self::RefreshRow;

    fn scale_refresh_public_input(
        &self,
        state: Self::RefreshPublicRow,
        scaling: Self::RefreshPublicScaling,
    ) -> Self::RefreshPublicRow;
    type RefreshAnchor;
    type RefreshAnchorParameters;

    fn sample_refresh_anchor(
        &self,
        level: &Self::RefreshLevel,
        parameters: &Self::RefreshAnchorParameters,
    ) -> Self::RefreshAnchor;
    fn refresh_anchor_public(&self, anchor: &Self::RefreshAnchor) -> Self::RefreshPublicRow;
    fn sample_refresh_witness(
        &self,
        anchor: &Self::RefreshAnchor,
        target: Self::RefreshPublicRow,
    ) -> Self::RefreshDecoderWitness;
    type RefreshSecret;
    type RefreshBaseParameters;

    /// Online preparation samples a decoder base using the current secret s
    /// and the fixed public anchor. All context is carried by these inputs.
    fn sample_refresh_decoder_base(
        secret: &Self::RefreshSecret,
        public: &Self::RefreshPublicRow,
        level: &Self::RefreshLevel,
        parameters: &Self::RefreshBaseParameters,
    ) -> Self::RefreshDecoderBase;
}

/// Direct refresh preparation: weighted terminal targets are preprocessed,
/// while decoder issuance uses the online secret s. The common application
/// owns digit routing, aggregation and finalization; these are individual
/// operations, not a callback implementing the complete refresh protocol.
pub trait ExponentLUTRefreshDirectScheme:
    ExponentLUTRefreshScheme + crate::readout::ExponentLUTTerminalScheme
{
    /// Concrete artifact binding and output-key schema supplied to shared transport.
    type RefreshArtifactBinding: serde::Serialize + PartialEq;
    type RefreshOutputKey: GraphValue<Schema: serde::Serialize>;

    fn refresh_artifact_binding(&self) -> Self::RefreshArtifactBinding;
    fn refresh_output_key(preparation: &Self::RefreshRecomposition) -> &Self::RefreshOutputKey;

    /// Restore only key-dependent recomposition constants and field schemas.
    /// Slot ordering, artifact import, and identity checks belong to the application.
    fn import_refresh_recomposition(
        &self,
        key: Self::RefreshOutputKey,
    ) -> Result<
        (
            Self::RefreshRecomposition,
            <Self::RefreshPublicRow as GraphValue>::Schema,
            <Self::RefreshRow as GraphValue>::Schema,
        ),
        Self::Error,
    >;

    type RefreshSecret;
    type RefreshDecoderParameters;

    fn prepare_direct_refresh_decoder(
        &self,
        secret: &Self::RefreshSecret,
        target: Self::RefreshPublicRow,
        level: &Self::RefreshLevel,
        parameters: &Self::RefreshDecoderParameters,
    ) -> Self::RefreshDecoder;

    fn offset_refresh_decoder(
        &self,
        decoder: Self::RefreshDecoder,
        offset: Self::RefreshRow,
    ) -> Self::RefreshDecoder;
    fn refresh_terminal_public(
        &self,
        raw: Self::TerminalPublicAccumulator,
    ) -> Self::RefreshPublicRow;
    fn refresh_terminal_encoding(&self, raw: Self::TerminalEncodingAccumulator)
    -> Self::RefreshRow;

    /// Payload-secret preparation of weighted digits, before online s exists.
    type RefreshDigitSecret;
    type RefreshDigitMaskBank;

    fn sample_refresh_digit_material(
        &self,
        secret: &Self::RefreshDigitSecret,
        readout: &crate::readout::ValueReadout,
        bank: &Self::RefreshDigitMaskBank,
        routing: &crate::noise_refresh::refresh::RefreshDigitRouting,
        hash_key: mxx_dsl::Bytes,
        tag: mxx_dsl::HashTag,
        id: [u8; 32],
    ) -> Result<
        (
            crate::program::ProgramValue<
                Self::TerminalMaterial,
                mxx_dsl::Family<Self::TerminalMaterial>,
            >,
            Self::RefreshRow,
        ),
        Self::Error,
    >;
}
