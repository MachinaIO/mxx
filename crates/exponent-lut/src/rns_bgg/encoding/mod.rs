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
        GraphParallelLowering, InputBinding, LoweringSite, LutId, LutTable, Operand,
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
use std::{collections::BTreeMap, sync::Arc};

mod decoder;
mod dynamic;
pub(crate) use self::dynamic::AUTOMORPHISM_HELPER_ROLE;
mod encrypt;
mod lowering;
mod online_eval;
mod verify;

pub(super) use self::lowering::*;
pub use self::verify::{
    TOP_REFRESH_TERMS_INPUT, decode_output, decode_public_output, decode_top_refresh,
    refresh_error, top_refresh_error, verify_output,
};
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
    /// Returns the encryption residual against the deterministic BGG relation
    /// of `carrier` (see [`RnsBggParameters::output_carrier`]): a private
    /// payload is scaled by the carrier's scale against its gadget. Fresh
    /// Gaussian error is therefore the returned value; no plaintext is
    /// reconstructed and public payload pairing remains metadata-driven.
    pub fn verification_residual(
        &self,
        parameters: &RnsBggParameters,
        carrier: &RnsOutputCarrier,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        expected: Mat,
    ) -> Result<RnsBggRow, RnsBggError> {
        let level = carrier.level;
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
            let gadget = parameters.output_gadget(carrier)?;
            let target = (payload.0 * gadget.clone(), payload.1 * gadget);
            let delta = parameters.scalar(level, carrier.scale.clone())?;
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

    /// The secret row `(s_bar, 1)` at level zero, as graph expressions that
    /// read the sampled polynomial through [`Self::input_name`].
    pub fn row(&self) -> &RnsBggRow {
        &self.row
    }

    /// Bind the sampled secret to its graph input in `runtime`, for a graph
    /// that reads [`Self::row`].
    pub fn bind_runtime(&self, runtime: &mut BTreeMap<String, RuntimeValue>) {
        if let Some(binding) = &self.binding {
            binding.bind(runtime);
        }
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
    parameters.encoding_contract().sample_row(&parameters.ring(level)?, columns)
}

/// The private encoding `s A - Delta x r G + e` of `plaintext` at `level`
/// under `key`, with its error sampled under `contract`.
fn encode_private_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
    contract: &RnsBggGaussian,
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
    let error = contract.sample_row(&ring, parameters.component_columns(level)?)?;
    let row = (
        mask_matrix.0 - target.0 * scale.clone() + error.0,
        mask_matrix.1 - target.1 * scale + error.1,
    );
    Ok(RnsBggEncoding::new(key.clone(), row, None))
}

/// The public encoding of `plaintext` at `level` under `key` (see
/// [`public_encoding_row`]), with its error sampled under `contract`.
fn encode_public_inner(
    parameters: &RnsBggParameters,
    key: &RnsBggPublicKey,
    mask: &RnsBggMaskSecret,
    plaintext: Mat,
    level: usize,
    contract: &RnsBggGaussian,
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
        contract,
    )?;
    Ok(RnsBggEncoding::new(key.clone(), row, Some(plaintext)))
}

/// The row `s A - s (d G) + e` encoding a public attribute `d` under key `A`
/// at `level`, with `public_scale = d G` for the full (unscaled)
/// current-level public gadget `G` and a fresh error row of `columns`
/// columns sampled under `contract`. Ordinary public inputs and Dynamic
/// digit companions share this equation.
fn public_encoding_row(
    parameters: &RnsBggParameters,
    matrix: RnsBggMatrix,
    mask: &RnsBggRow,
    public_scale: Mat,
    columns: usize,
    level: usize,
    contract: &RnsBggGaussian,
) -> Result<RnsBggRow, RnsBggError> {
    let s = reduce_secret(parameters, mask, level)?;
    let mask_matrix = (
        s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0,
        s.0.clone() * matrix.0.1 + s.1.clone() * matrix.1.1,
    );
    let error = contract.sample_row(&parameters.ring(level)?, columns)?;
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
    /// Dynamic: each RHS's digit-companion family, imported where the
    /// Program is bound so a Program family body reads its members in place.
    pub(crate) rhs_companions: BTreeMap<RhsId, Family<RnsBggRow>>,
    /// Static: the material of each `(level, LUT)` evaluated in a family
    /// body, imported where the Program is bound so every member reads it in
    /// place.
    pub(crate) lut_material: BTreeMap<(usize, LutId), LutMaterial>,
    /// How the Program's outputs are consumed (see [`RnsOutputUse`]).
    pub(crate) output_use: RnsOutputUse,
    /// Dynamic: the setup, the Program's mapping to its coordinates, and the
    /// imported automorphism packages.
    pub(crate) dynamic: Option<DynamicEvaluation>,
    /// The Program's integer data inputs ([`crate::program::data_inputs`]),
    /// read by data-backed LUT tables in either RHS mode.
    pub(crate) data: BTreeMap<crate::program::DataId, Family<Int>>,
}

/// The Dynamic state of an online evaluation.
#[derive(Clone)]
pub(crate) struct DynamicEvaluation {
    pub(crate) setup: Arc<RnsBggDynamicSetup>,
    pub(crate) binding: RnsBggDynamicBinding,
    /// The helpers `h_k` and the digit companions of `C_k` of every
    /// nonidentity setup automorphism, member `i` for the `i`-th one; absent
    /// when `H` is trivial.
    pub(crate) automorphisms: Option<(Family<RnsBggRow>, Family<RnsBggRow>)>,
}

/// A Static LUT's imported switching ciphertexts (one family per matrix
/// port), its automorphism indices, and its helpers.
pub(crate) type LutMaterial = (Vec<Family<Mat>>, Vec<usize>, (Family<Mat>, Family<Mat>));

/// The bound value of one Dynamic RHS coordinate of an encryption.
#[derive(Clone)]
pub(crate) enum DynamicRhsValue {
    /// The plaintext of a coordinate, from which the encryption's own
    /// prelude samples the bottom rows of its gadget ciphertexts.
    Plaintext(InputBinding<Mat, super::PlaintextFamily>),
    /// A coordinate whose bottom rows the bound prelude production exports.
    Prelude,
}

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
    /// Dynamic: the setup this encryption belongs to.
    dynamic_setup: Option<Arc<RnsBggDynamicSetup>>,
    /// Dynamic: the encryption session identifier and the value of every
    /// RHS coordinate of the setup.
    dynamic_rhs: Option<([u8; 32], BTreeMap<u32, DynamicRhsValue>)>,
    /// Dynamic: the prelude production exporting the bottom rows of the RHS
    /// gadget ciphertexts (the encryption's own, or one an enclosing scheme
    /// binds), with its manifest.
    prelude: Option<(ProductionId, Manifest)>,
    /// For a refresh child: the namespace its hash key is derived under.
    hash_namespace: Option<[u8; 32]>,
    /// The bound noise refresh: its fresh-noise child and its decoders'
    /// smudge children, in decoder order.
    /// A top-refresh part also binds its mask child (first).
    refresh: Option<(
        crate::refresh::RefreshDeclaration,
        Option<Box<RnsEncryptBuild<'a>>>,
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
    /// How the Program's outputs are consumed; it must match the
    /// preprocessing binding.
    output_use: RnsOutputUse,
    /// The output carriers the preprocessing binding resolved; online
    /// evaluation resolves the same ones.
    output_carriers: Vec<RnsOutputCarrier>,
    /// Dynamic: the setup and the Program's mapping to its coordinates.
    dynamic: Option<(Arc<RnsBggDynamicSetup>, RnsBggDynamicBinding)>,
    /// The values of the Program's integer data, which must be the preprocessed data.
    data: crate::program::ProgramData,
    /// For a refresh child: the namespace its hash key is derived under.
    hash_namespace: Option<[u8; 32]>,
    /// The applied noise refresh: its target output index, declaration, and
    /// fresh-noise and smudge children (in decoder order).
    /// A top-refresh part also has its mask child (after the declaration).
    pending_refresh: Option<(
        usize,
        crate::refresh::RefreshDeclaration,
        Option<Box<RnsOnlineEvalBuild<'a>>>,
        Box<RnsOnlineEvalBuild<'a>>,
        Vec<RnsOnlineEvalBuild<'a>>,
    )>,
    /// The namespace and key of a top refresh's `A'`.
    refresh_matrix_key: Option<([u8; 32], Vec<u8>)>,
    /// The integer data of the Program and of its bound children, by graph
    /// input name: the Program's given data ([`RnsOnlineEvalBuild::with_data`])
    /// and the children's resident inputs holding the entries preprocessing
    /// exported under the same artifact names.
    runtime: BTreeMap<String, RuntimeValue>,
}
