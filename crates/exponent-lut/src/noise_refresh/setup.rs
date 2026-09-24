//! Scheme-independent refresh setup contracts and verification metadata.
//!
//! Concrete key, encoding, and graph producers belong to backend adapters.
//! This module only owns errors, manifest validation, and the generic
//! verification boundary shared by those adapters.

use crate::noise_refresh::{prf::PrfError, refresh::RefreshError};
use mxx_dsl::{Bool, BuiltGraph, DslContext, GraphValue, Mat};
use mxx_ir_core::{
    IntExpr, ParamEnv, ScopedWireRef,
    types::{NodeId, WireType},
};
use std::error::Error;
use thiserror::Error as ThisError;

/// Errors shared by backend refresh setup adapters.
#[derive(Debug, ThisError)]
pub enum RefreshSetupError {
    #[error(transparent)]
    Prf(#[from] PrfError),
    #[error(transparent)]
    Refresh(#[from] RefreshError),
    #[error("backend refresh setup failed: {0}")]
    Backend(#[source] Box<dyn Error + Send + Sync>),
    #[error("PBC construction failed: {0}")]
    Pbc(String),
    #[error("the DSL rejected refresh setup graph construction: {0}")]
    Dsl(#[from] mxx_dsl::DslError),
    #[error("invalid refresh setup parameters: {0}")]
    InvalidParameters(&'static str),
    #[error("refresh setup manifest is invalid")]
    InvalidManifest,
    #[error("refresh setup stage `{stage}` failed: {detail}")]
    SetupStage { stage: &'static str, detail: String },
    #[error("refresh setup identity mismatch")]
    IdentityMismatch,
    #[error("refresh verification decoder residuals do not match the bound operands")]
    DecoderResidualMismatch,
}

/// Checks a manifest's typed artifact ports against a backend-supplied field
/// list and a generic readout schema.

/// Verification graph outputs and their exact threshold-decoder operands.
/// Backend adapters supply the relation residual and scalar components; this
/// boundary connects them to decoders without changing the row carrier.
pub struct RefreshVerification<V: GraphValue> {
    encoding_vector: V,
    anchor: V,
    residual: V,
    decoder_residuals: Vec<Mat>,
    decoded: Vec<Bool>,
}

impl<V: GraphValue> RefreshVerification<V> {
    pub fn new(
        encoding_vector: V,
        anchor: V,
        residual: V,
        decoder_residuals: Vec<Mat>,
        plaintext_modulus: impl Into<IntExpr>,
        decode_length: usize,
    ) -> Result<Self, RefreshSetupError> {
        let decoded_count = decoder_residuals
            .len()
            .checked_mul(decode_length)
            .ok_or(RefreshSetupError::InvalidParameters("decoded output count overflow"))?;
        let plaintext_modulus = plaintext_modulus.into();
        let mut decoded = Vec::with_capacity(decoded_count);
        for scalar_residual in &decoder_residuals {
            decoded.extend(
                scalar_residual
                    .clone()
                    .threshold_decode_bools(plaintext_modulus.clone(), decode_length),
            );
        }
        Ok(Self { encoding_vector, anchor, residual, decoder_residuals, decoded })
    }

    pub fn residual(&self) -> &V {
        &self.residual
    }

    pub fn decoder_residuals(&self) -> &[Mat] {
        &self.decoder_residuals
    }

    pub fn validate_decoder_residuals(&self, residuals: &[Mat]) -> Result<(), RefreshSetupError> {
        if residuals.len() != self.decoder_residuals.len() ||
            residuals.iter().zip(&self.decoder_residuals).any(|(candidate, expected)| {
                candidate.value_handle() != expected.value_handle() ||
                    candidate.matrix_type() != expected.matrix_type()
            })
        {
            return Err(RefreshSetupError::DecoderResidualMismatch);
        }
        Ok(())
    }

    pub fn decoded(&self) -> &[Bool] {
        &self.decoded
    }

    pub fn add_outputs(
        &self,
        context: DslContext,
        residual_name: impl Into<String>,
        decoded_prefix: impl Into<String>,
    ) -> Result<DslContext, RefreshSetupError> {
        let mut context = context
            .transferred_output("refresh-verification-vector", self.encoding_vector.clone())?
            .transferred_output("refresh-verification-anchor", self.anchor.clone())?
            .transferred_output(residual_name, self.residual.clone())?;
        for (column, residual) in self.decoder_residuals.iter().enumerate() {
            context = context.transferred_output(
                format!("refresh-decoder-residual-{column}"),
                residual.clone(),
            )?;
        }
        let decoded_prefix = decoded_prefix.into();
        for (index, value) in self.decoded.iter().enumerate() {
            context = context.output(format!("{decoded_prefix}_{index}"), value.clone())?;
        }
        Ok(context)
    }
}

/// Exact matrix facts exported for request-level external-input validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSimulationMatrixInputMetadata {
    pub canonical_coefficient_exclusive_upper_bound: Option<IntExpr>,
    pub is_constant_polynomial: bool,
}

/// One exact residual/decoder linkage produced by a verification graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSimulationDecoderTarget {
    pub target_id: String,
    pub residual_output_name: String,
    pub residual_anchor: String,
    pub residual: ScopedWireRef,
    pub residual_type: WireType,
    pub decoder: ScopedWireRef,
    pub decoder_node: NodeId,
    pub decoder_output_type: WireType,
    pub decoded_output_name: String,
    pub decoder_anchor: String,
    pub plaintext_modulus: IntExpr,
}

/// Validates a frozen graph using the common parameter environment.
pub fn validate_graph(graph: &BuiltGraph) -> Result<(), RefreshSetupError> {
    graph
        .validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)
        .map(|_| ())
        .map_err(|_| RefreshSetupError::InvalidManifest)
}
