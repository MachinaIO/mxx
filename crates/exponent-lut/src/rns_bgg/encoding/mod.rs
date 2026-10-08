//! Encoding, input encryption, and online evaluation lowering for RNS-BGG.
//!
//! Secret-dependent sampling belongs to [`RnsEncryptBuild`].  The online
//! evaluation lowerer only resolves lazy, compact artifact handles and evaluates the
//! graph operations described by the common Program walker.
#![warn(missing_docs)]

use super::*;
use crate::{
    api::{
        ApiError, EncryptBuild, OnlineEvalBuild, PlaintextInputs, StageArtifacts, StageExecution,
        StageGraph, StageOutputs, apply_decoder,
    },
    decoder::decoding::{Decoder, DecoderDeclaration, DecoderError},
    program::{
        AttributeKnowledge, Environment, ExpLayout, ExponentLUTProgramLowering, ExponentLutProgram,
        GraphParallelLowering, InputBinding, InputShape, LoweringSite, LutId, LutTable, Operand,
        ProgramLoweringContext, RhsId, RhsRef, WireId, family_of_fields,
    },
};
use mxx_backends::{
    RuntimeValue, SessionStore, backend::poly::threshold_decode_coefficients,
    matrix::dcrt_poly::DCRTPolyMatrix, poly::Poly,
};
use mxx_dsl::{Bytes, DslContext, DslError, Family, GraphValue, GraphValueSchema, Int, Mat};
use mxx_ir_core::{
    IntExpr, ParamEnv, ValueHandle,
    artifact::{ArtifactAvailability, ArtifactType, Manifest, ProductionId},
    types::{ConcreteMatrixType, WireType},
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

mod decoder;
mod dynamic;
mod encrypt;
mod lowering;
mod online_eval;
mod verify;

pub(super) use self::lowering::*;
pub use self::verify::{decode_output, decode_public_output, refresh_error, verify_output};
#[cfg(test)]
use crate::{api::execute_graph, program::ExponentLutProgramBuilder};
#[cfg(test)]
use mxx_ir_core::node::ConstantMatrix;

/// Encoded row paired with the public key for its Program wire and, for
/// public wires, plaintext.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RnsBggEncoding<R = RnsBggRow, A = Mat, K = RnsBggPublicKey> {
    pub(crate) public_key: K,
    pub(crate) row: R,
    pub(crate) public_payload: Option<A>,
}

impl<R, A> RnsBggEncoding<R, A, RnsBggPublicKey> {
    /// Construct an encoding carrier paired with its public key.
    ///
    /// Callers are responsible for preserving the Program wire identity,
    /// row schema, and public-payload presence implied by that wire's
    /// validated knowledge; stage builders validate these invariants when
    /// the carrier is consumed.
    pub fn new(public_key: RnsBggPublicKey, row: R, public_payload: Option<A>) -> Self {
        Self { public_key, row, public_payload }
    }
    /// Program wire represented by this encoding.
    pub fn wire(&self) -> WireId {
        self.public_key.wire
    }
    /// Public key paired with this encoding.
    pub fn public_key(&self) -> &RnsBggPublicKey {
        &self.public_key
    }
    /// Encoded row.
    pub fn row(&self) -> &R {
        &self.row
    }
    /// Paired public payload, if present.
    pub fn public_payload(&self) -> Option<&A> {
        self.public_payload.as_ref()
    }
}

impl RnsBggEncoding<RnsBggRow, Mat> {
    /// Returns the encryption residual against the deterministic BGG relation,
    /// a private payload being scaled by `message_scale`. Fresh Gaussian
    /// error is therefore the returned value; no plaintext is reconstructed
    /// and public payload pairing remains metadata-driven.
    pub fn verification_residual(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        expected: Mat,
        message_scale: &BigUint,
    ) -> Result<RnsBggRow, RnsBggError> {
        let ring = parameters.ring(level)?;
        if expected.matrix_type() != &ring.matrix_type((1, 1)) {
            return Err(RnsBggError::Binding("verification residual plaintext ring"));
        }
        let mask = reduce_secret(parameters, &mask.row, level)?;
        let mask_matrix = (
            mask.0.clone() * self.public_key.matrix.0.0.clone() +
                mask.1.clone() * self.public_key.matrix.1.0.clone(),
            mask.0.clone() * self.public_key.matrix.0.1.clone() +
                mask.1.clone() * self.public_key.matrix.1.1.clone(),
        );
        let deterministic = if self.public_payload.is_some() {
            let gadget = parameters.public_gadget(level)?;
            let scale = expected.clone() * gadget;
            (mask.0.clone() * scale.clone(), mask.1.clone() * scale)
        } else {
            let payload = reduce_secret(parameters, &payload.row, level)?;
            let gadget = parameters.state_gadget(level)?;
            let target = (payload.0 * gadget.clone(), payload.1 * gadget);
            let delta = parameters.scalar(level, message_scale.clone())?;
            let scale = delta * expected;
            (target.0 * scale.clone(), target.1 * scale)
        };
        Ok((
            self.row.0.clone() - mask_matrix.0 + deterministic.0,
            self.row.1.clone() - mask_matrix.1 + deterministic.1,
        ))
    }
}

impl<R: GraphValue, A: GraphValue, K: GraphValue> GraphValue for RnsBggEncoding<R, A, K> {
    type Schema = RnsBggEncoding<R::Schema, A::Schema, K::Schema>;
    fn flatten(&self) -> Vec<ValueHandle> {
        let mut values = self.public_key.flatten();
        values.extend(self.row.flatten());
        if let Some(payload) = &self.public_payload {
            values.extend(payload.flatten());
        }
        values
    }
    fn schema(&self) -> Self::Schema {
        RnsBggEncoding {
            public_key: self.public_key.schema(),
            row: self.row.schema(),
            public_payload: self.public_payload.as_ref().map(GraphValue::schema),
        }
    }
    fn from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError> {
        let key_len = schema.public_key.wire_types().len();
        let row_len = schema.row.wire_types().len();
        if values.len() < key_len + row_len {
            return Err(DslError::Schema);
        }
        let public_key = K::from_values(&schema.public_key, &values[..key_len])?;
        let row = R::from_values(&schema.row, &values[key_len..key_len + row_len])?;
        let public_payload = match &schema.public_payload {
            Some(payload) => Some(A::from_values(payload, &values[key_len + row_len..])?),
            None if values.len() == key_len + row_len => None,
            None => return Err(DslError::Schema),
        };
        Ok(Self { public_key, row, public_payload })
    }
}

impl<R: GraphValueSchema, A: GraphValueSchema, K: GraphValueSchema> GraphValueSchema
    for RnsBggEncoding<R, A, K>
{
    type Value = RnsBggEncoding<R::Value, A::Value, K::Value>;
    fn placeholders_from(&self, next: &mut usize) -> Self::Value {
        RnsBggEncoding {
            public_key: self.public_key.placeholders_from(next),
            row: self.row.placeholders_from(next),
            public_payload: self.public_payload.as_ref().map(|p| p.placeholders_from(next)),
        }
    }
    fn wire_types(&self) -> Vec<WireType> {
        let mut values = self.public_key.wire_types();
        values.extend(self.row.wire_types());
        if let Some(payload) = &self.public_payload {
            values.extend(payload.wire_types());
        }
        values
    }
}

/// Secret payload row used for private input and RHS encryption.
#[derive(Clone)]
pub struct RnsBggPayloadSecret {
    pub(crate) row: RnsBggRow,
    pub(crate) binding: Option<SecretBinding>,
}

/// Secret mask row used for input and helper encryption.
#[derive(Clone)]
pub struct RnsBggMaskSecret {
    pub(crate) row: RnsBggRow,
    pub(crate) binding: Option<SecretBinding>,
}

/// A sampled secret polynomial resident on the stage execution. Every stage
/// graph reads it through the graph input `name`, bound to this same value,
/// so preprocessing, encryption, and verification share one secret.
#[derive(Clone)]
pub(crate) struct SecretBinding {
    pub(crate) name: String,
    pub(crate) value: RuntimeValue,
}

impl SecretBinding {
    /// Bind this secret's graph input in a stage's runtime inputs.
    pub(crate) fn bind(&self, runtime: &mut BTreeMap<String, RuntimeValue>) {
        runtime.insert(self.name.clone(), self.value.clone());
    }
}

impl RnsBggPayloadSecret {
    /// The graph input through which every stage reads this sampled secret,
    /// or `None` for a secret given as a graph expression.
    pub fn input_name(&self) -> Option<&str> {
        self.binding.as_ref().map(|binding| binding.name.as_str())
    }

    /// Re-express the same sampled coefficients in a requested RNS level.
    ///
    /// This preserves the secret relation across stage-local parameter
    /// schedules; it does not sample a second payload.
    pub fn at_level(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
    ) -> Result<Self, RnsBggError> {
        Ok(Self {
            row: reduce_secret(parameters, &self.row, level)?,
            binding: self.binding.clone(),
        })
    }
}

impl RnsBggMaskSecret {
    /// The graph input through which every stage reads this sampled secret,
    /// or `None` for a secret given as a graph expression.
    pub fn input_name(&self) -> Option<&str> {
        self.binding.as_ref().map(|binding| binding.name.as_str())
    }

    /// Re-express the same sampled coefficients in a requested RNS level.
    ///
    /// This preserves the secret relation across stage-local parameter
    /// schedules; it does not sample a second mask.
    pub fn at_level(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
    ) -> Result<Self, RnsBggError> {
        Ok(Self {
            row: reduce_secret(parameters, &self.row, level)?,
            binding: self.binding.clone(),
        })
    }
}

/// The mask secret of a stage: the supplied Static mask, or in Dynamic
/// mode the single secret `s` itself, aliasing the payload's sampled rows
/// and runtime binding so no second secret exists.
fn stage_mask(
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: Option<&RnsBggMaskSecret>,
) -> Result<RnsBggMaskSecret, ApiError> {
    match (parameters.rhs_mode(), mask) {
        (RhsMode::Static, Some(mask)) => Ok(mask.clone()),
        (RhsMode::Dynamic, None) => {
            Ok(RnsBggMaskSecret { row: payload.row.clone(), binding: payload.binding.clone() })
        }
        (RhsMode::Static, None) => Err(ApiError::input("Static stages require a mask secret")),
        (RhsMode::Dynamic, Some(_)) => {
            Err(ApiError::input("Dynamic stages take no mask: the payload secret is the mask"))
        }
    }
}

fn graph<E: std::fmt::Display>(error: E) -> RnsBggError {
    RnsBggError::Graph(error.to_string())
}

fn merge_runtime_inputs(
    destination: &mut BTreeMap<String, RuntimeValue>,
    source: &BTreeMap<String, RuntimeValue>,
) -> Result<(), ApiError> {
    for (name, value) in source {
        match destination.get(name) {
            Some(RuntimeValue::Bytes(existing)) => match value {
                RuntimeValue::Bytes(candidate) if existing == candidate => {}
                RuntimeValue::Bytes(_) => {
                    return Err(ApiError::input("decoder child runtime input conflicts"));
                }
                _ => return Err(ApiError::input("decoder child runtime input schema")),
            },
            // A secret input name is unique to one sampled secret.
            Some(RuntimeValue::Matrix(_)) if matches!(value, RuntimeValue::Matrix(_)) => {}
            Some(_) => return Err(ApiError::input("decoder child runtime input schema")),
            None => {
                destination.insert(name.clone(), value.clone());
            }
        }
    }
    Ok(())
}

/// Sample one secret row `(s_bar, 1)` with uniform ternary `s_bar`. The
/// polynomial is sampled once by its own graph on `execution` and stays
/// resident there; the returned row reads it through a uniquely named graph
/// input, so each stage that binds the value sees the same secret.
fn sample_secret_row<E, S>(
    parameters: &RnsBggParameters,
    execution: &mut E,
    store: &mut S,
    before_execute: impl FnOnce(&mxx_ir_core::ValidatedGraph) -> Result<(), ApiError>,
) -> Result<(RnsBggRow, SecretBinding), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let ring = parameters.ring(0).map_err(ApiError::scheme)?;
    let graph = DslContext::new("rns-bgg-secret")
        .output("secret", ring.uniform_interval((1, 1), -1, 1))
        .map_err(ApiError::scheme)?
        .build()
        .map_err(ApiError::scheme)?;
    let sampled = crate::api::execute_graph(
        StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
        &ParamEnv::default(),
        execution,
        store,
        before_execute,
    )?;
    let value = sampled
        .outputs
        .get("secret")
        .cloned()
        .ok_or_else(|| ApiError::input("sampled secret output is missing"))?;
    let name = format!("rns-bgg/secret/{:032x}", rand::random::<u128>());
    let row = (ring.input(name.clone(), (1, 1)), ring.identity(1));
    Ok((row, SecretBinding { name, value }))
}

/// Keep only the runtime inputs `graph` declares; a stage binds every secret
/// it holds, but a graph may not read all of them.
pub(crate) fn declared_runtime_inputs(
    graph: &mxx_dsl::BuiltGraph,
    mut runtime: BTreeMap<String, RuntimeValue>,
) -> BTreeMap<String, RuntimeValue> {
    runtime.retain(|name, _| super::graph_declares_input(graph, name));
    runtime
}

fn reduce_secret(
    parameters: &RnsBggParameters,
    row: &RnsBggRow,
    level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let ring = parameters.ring(level)?;
    checked_modulus_convert_row(row.clone(), &ring)
}

pub(crate) fn fresh_error(
    parameters: &RnsBggParameters,
    level: usize,
    columns: usize,
) -> Result<RnsBggRow, RnsBggError> {
    encoding_error_row(parameters, &parameters.ring(level)?, columns)
}

/// One row of encoding error over `ring`; zero when the encoding sigma is
/// zero, so noise-free encodings add no sampled error.
fn encoding_error_row(
    parameters: &RnsBggParameters,
    ring: &mxx_dsl::Ring,
    columns: usize,
) -> Result<RnsBggRow, RnsBggError> {
    if parameters.encoding_error_sigma().is_zero() {
        let zero = ring.zero((1, columns));
        return Ok((zero.clone(), zero));
    }
    let sigma = crate::utils::decimal_to_real_expr(parameters.encoding_error_sigma())
        .map_err(|_| RnsBggError::Parameters("invalid encoding error sigma"))?;
    Ok((
        ring.gaussian((1, columns), sigma.clone(), parameters.encoding_error_bound().clone()),
        ring.gaussian((1, columns), sigma, parameters.encoding_error_bound().clone()),
    ))
}

fn encode_private_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
) -> Result<RnsBggEncoding, RnsBggError> {
    let ring = parameters.ring(level)?;
    if *plaintext.matrix_type() != ring.matrix_type((1, 1)) {
        return Err(RnsBggError::Binding("private plaintext ring"));
    }
    let s = reduce_secret(parameters, &mask.row, level)?;
    let r = reduce_secret(parameters, &payload.row, level)?;
    let gadget = parameters.state_gadget(level)?;
    let target = (r.0.clone() * gadget.clone(), r.1.clone() * gadget.clone());
    let delta = parameters.scalar(level, parameters.message_scale(level)?)?;
    let matrix = key.matrix.clone();
    let mask_matrix =
        (s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0, s.0 * matrix.0.1 + s.1 * matrix.1.1);
    let scale = delta * plaintext;
    let error = fresh_error(parameters, level, parameters.component_columns(level)?)?;
    let row = (
        mask_matrix.0 - target.0 * scale.clone() + error.0,
        mask_matrix.1 - target.1 * scale + error.1,
    );
    Ok(RnsBggEncoding::new(key.clone(), row, None))
}

fn encode_public_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
) -> Result<RnsBggEncoding, RnsBggError> {
    let ring = parameters.ring(level)?;
    if *plaintext.matrix_type() != ring.matrix_type((1, 1)) {
        return Err(RnsBggError::Binding("public plaintext ring"));
    }
    let row = public_encoding_row(
        parameters,
        key.matrix.clone(),
        &mask.row,
        plaintext.clone() * parameters.public_gadget(level)?,
        parameters.level(level)?.modulus_digits(),
        level,
    )?;
    Ok(RnsBggEncoding::new(key.clone(), row, Some(plaintext)))
}

/// The row `s A - s (d G) + e` encoding a public attribute `d` under key `A`
/// at `level`, with `public_scale = d G` for the full (unscaled)
/// current-level public gadget `G` and a fresh error row of `columns`
/// columns. Ordinary public inputs and Dynamic digit companions share this
/// equation.
fn public_encoding_row(
    parameters: &RnsBggParameters,
    matrix: RnsBggMatrix,
    mask: &RnsBggRow,
    public_scale: Mat,
    columns: usize,
    level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let s = reduce_secret(parameters, mask, level)?;
    let mask_matrix = (
        s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0,
        s.0.clone() * matrix.0.1 + s.1.clone() * matrix.1.1,
    );
    let error = fresh_error(parameters, level, columns)?;
    Ok((
        mask_matrix.0 - s.0 * public_scale.clone() + error.0,
        mask_matrix.1 - s.1 * public_scale + error.1,
    ))
}

/// Online evaluation lowering state.  Its artifact fields are metadata and lazy handles,
/// never resident matrix payloads.
pub struct EncodingLowering<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
    pub(crate) program: Arc<ExponentLutProgram>,
    pub(crate) workload: Arc<RnsBggDerivedWorkload>,
    pub(crate) preprocess: ProductionId,
    pub(crate) encryption: ProductionId,
    pub(crate) manifests: BTreeMap<ProductionId, Manifest>,
    pub(crate) decoder_smudge: Option<Arc<DecoderDeclaration>>,
    /// For a child Program: the namespace `<prefix>/<role>` its imported
    /// artifacts are published under (see `decoder_smudge_artifact_name` and
    /// `refresh_fresh_artifact_name`).
    pub(crate) artifact_prefix: Option<String>,
    pub(crate) current_site: Option<LoweringSite>,
    /// Graph input of the shared hash key, from which `A_sigma` is sampled.
    pub(crate) hash_key: Bytes,
    /// Setup context, part of the Dynamic digit-companion key namespace.
    pub(crate) context: [u8; 32],
    /// Dynamic: each RHS's digit-companion family, imported where the
    /// Program is bound so a Program family body reads its members in place.
    pub(crate) rhs_companions: BTreeMap<RhsId, Family<RnsBggRow>>,
    /// The material of each `(level, LUT)` evaluated in a family body,
    /// imported where the Program is bound so every member reads it in place.
    pub(crate) lut_material: BTreeMap<(usize, LutId), LutMaterial>,
}

/// A LUT's imported switching ciphertexts (one family per matrix port), its
/// automorphism indices, its helpers, and its Dynamic digit companions.
pub(crate) type LutMaterial =
    (Vec<Family<Mat>>, Vec<usize>, (Family<Mat>, Family<Mat>), Option<Family<RnsBggRow>>);

/// Encryption builder; secrets and plaintexts are stage-local.
pub struct RnsEncryptBuild<'a> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    hash_key: Vec<u8>,
    inputs: Option<crate::program::EncryptionInputs>,
    projection_identity: Option<[u8; 32]>,
    payload_secret: Option<RnsBggPayloadSecret>,
    mask_secret: Option<RnsBggMaskSecret>,
    /// Input plaintexts, read from graph inputs; a family is read member by
    /// member.
    plaintexts: BTreeMap<WireId, InputBinding<Mat, super::PlaintextFamily>>,
    preprocess: Option<ProductionId>,
    binding: Option<RnsBggArtifactBinding>,
    decoder_declaration: Option<DecoderDeclaration>,
    manifests: BTreeMap<ProductionId, Manifest>,
    runtime: BTreeMap<String, RuntimeValue>,
    /// The bound smudge child of an applied decoder.
    decoder_smudge_child: Option<Box<RnsEncryptBuild<'a>>>,
    /// For a smudge child: the decoder declaration whose smudge Program it
    /// encrypts the inputs of.
    decoder_smudge: Option<DecoderDeclaration>,
    /// Dynamic: the shape and consuming level of each declared RHS and each
    /// evaluated `(level, LUT)`, known once preprocessing (or, for a smudge
    /// child, its Program) is bound.
    dynamic_schema: Option<(BTreeMap<RhsId, (InputShape, usize)>, BTreeSet<(usize, LutId)>)>,
    /// Dynamic: the encryption batch identifier and the RHS plaintexts, read
    /// from graph inputs; a family is read member by member.
    dynamic_rhs: Option<([u8; 32], BTreeMap<RhsId, InputBinding<Mat, super::PlaintextFamily>>)>,
    /// Dynamic: the executed prelude production holding the RHS family
    /// ciphertexts, with its manifest.
    prelude: Option<(ProductionId, Manifest)>,
    /// For a refresh child: the namespace its hash key is derived under.
    hash_namespace: Option<[u8; 32]>,
    /// The bound noise refresh: its fresh-noise child and its decoders'
    /// smudge children, in decoder order.
    refresh: Option<(
        crate::refresh::RefreshDeclaration,
        Box<RnsEncryptBuild<'a>>,
        Vec<RnsEncryptBuild<'a>>,
    )>,
}

/// Online evaluation builder with no backend, secrets, or plaintexts.
pub struct RnsOnlineEvalBuild<'a> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    hash_key: Vec<u8>,
    evaluator: Option<EncodingLowering<'a>>,
    families: OnlineEvalFamilies<'a>,
    environment: Environment<
        RnsBggEncoding,
        (RnsBggMatrix, RhsRef<Int>),
        OnlineEvalWireFamily,
        OnlineEvalRhsFamily,
    >,
    /// The applied decoder: its output index, declaration and smudge child.
    pending_decoder: Option<(usize, DecoderDeclaration, Box<RnsOnlineEvalBuild<'a>>)>,
    /// The coefficient bound of the unmask preimage, from the
    /// preprocess binding.
    unmask_preimage_bound: Option<BigUint>,
    /// For a refresh child: the namespace its hash key is derived under.
    hash_namespace: Option<[u8; 32]>,
    /// The applied noise refresh: its target output index, declaration, and
    /// fresh-noise and smudge children (in decoder order).
    pending_refresh: Option<(
        usize,
        crate::refresh::RefreshDeclaration,
        Box<RnsOnlineEvalBuild<'a>>,
        Vec<RnsOnlineEvalBuild<'a>>,
    )>,
}
