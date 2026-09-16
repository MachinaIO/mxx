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
    /// Extract a decoded CRT residue. RNS-BGG rounds after decoder subtraction;
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
