//! Encoding, input encryption, and online lowering for RNS-BGG.
//!
//! Secret-dependent sampling belongs to [`RnsEncryptBuild`].  The online
//! lowerer only resolves lazy, compact artifact handles and evaluates the
//! graph operations described by the common Program walker.
#![warn(missing_docs)]

use super::*;
use crate::{
    api::{
        ApiError, EncryptBuild, OnlineBuild, PlaintextInputs, RefreshApplicationError,
        StageArtifacts, StageExecution, StageGraph, StageOutputs, apply_refresh, domain_identity,
        execute_artifacts, execute_graph,
    },
    noise_refresh::{
        prf::PrfLabel,
        refresh::{NoiseRefresh, RefreshDeclaration, RefreshError, RefreshPrfLabel},
    },
    program::{
        AttributeKnowledge, Environment, ExpLayout, ExponentLUTProgramLowering, ExponentLutProgram,
        ExponentLutProgramBuilder, GraphParallelLowering, InputBinding, LoweringSite, LutTable,
        Operand, ProgramLoweringContext, RhsId, WireId, family_of_fields,
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
    node::ConstantMatrix,
    types::{ConcreteMatrixType, WireType},
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

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
    /// Returns the encryption residual against the deterministic BGG relation.
    /// Fresh Gaussian error is therefore the returned value; no plaintext is
    /// reconstructed and public payload pairing remains metadata-driven.
    pub fn verification_residual(
        &self,
        parameters: &RnsBggParameters,
        level: usize,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        expected: Mat,
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
            let delta = parameters.scalar(level, parameters.message_scale(level)?)?;
            let scale = delta * expected;
            (target.0 * scale.clone(), target.1 * scale)
        };
        Ok((
            self.row.0.clone() - mask_matrix.0 + deterministic.0,
            self.row.1.clone() - mask_matrix.1 + deterministic.1,
        ))
    }
}

/// Verify one output returned by [`crate::api::online_eval`].
///
/// The output is a paired RNS-BGG carrier. Its public-key ports, encoded-row
/// ports, and (for public outputs) public payload are consumed from the
/// ordinary runtime result owner. The residual graph and its final decode are
/// concrete RNS-BGG application operations; no generic runtime decoder or
/// verification artifact is exposed. `payload` and `mask` remain
/// caller-owned, and neither is passed to online evaluation.
pub fn verify_output<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let wire = match program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?
    {
        Operand::Scalar(wire) => *wire,
        Operand::Member { .. } => {
            return Err(ApiError::input("verification output must be scalar"));
        }
    };
    let workload = RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?;
    let source_level = workload.level(wire).map_err(ApiError::scheme)?;
    let coefficient_output = program
        .wire_is_coefficient_output(wire)
        .map_err(|_| ApiError::input("verification output metadata is invalid"))?;
    let output_level = source_level
        .checked_add(usize::from(coefficient_output))
        .ok_or_else(|| ApiError::input("verification output level overflow"))?;
    verify_output_at_level(
        program,
        result,
        output_index,
        wire,
        output_level,
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

/// Verify one output after a declared RNS-BGG refresh.
///
/// A refresh changes the carrier modulus and public key, so the ordinary
/// Program workload level is no longer the level of the returned carrier.
/// The declaration supplies that target level while the rest of the
/// verification remains the same application-facing operation as
/// [`verify_output`].
pub fn verify_refreshed_output<E, S, P>(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
    result: &StageOutputs,
    output_index: usize,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    let wire = declaration.source_output();
    let output = program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("verification output index is out of range"))?;
    if !matches!(output, Operand::Scalar(actual) if *actual == wire) {
        return Err(ApiError::input("refreshed verification output does not match declaration"));
    }
    verify_output_at_level(
        program,
        result,
        output_index,
        wire,
        declaration.target_level(),
        expected,
        parameters,
        payload,
        mask,
        execution,
        store,
    )
}

fn verify_output_at_level<E, S, P>(
    program: &ExponentLutProgram,
    result: &StageOutputs,
    output_index: usize,
    wire: WireId,
    output_level: usize,
    expected: &P,
    parameters: &RnsBggParameters,
    payload: &RnsBggPayloadSecret,
    mask: &RnsBggMaskSecret,
    execution: &mut E,
    store: &mut S,
) -> Result<(), ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
    P: Poly,
{
    // Only the terminal relation (identity state gadget, public scale
    // floor(q / 2)) separates plaintexts by more than the q / 4 decoding
    // threshold; an intermediate carrier's gadget powers can be smaller than
    // its noise, so it cannot be verified by decoding.
    if output_level + 1 != parameters.levels().len() {
        return Err(ApiError::input(format!(
            "verification output is at level {output_level}, not the terminal level {}",
            parameters.levels().len() - 1
        )));
    }
    let output_ring = parameters.ring(output_level).map_err(ApiError::scheme)?;
    let knowledge = program.wire_knowledge(wire).map_err(ApiError::scheme)?;
    let private_output = matches!(knowledge, AttributeKnowledge::PrivatePayload);
    let columns =
        parameters.key_component_columns(output_level, knowledge).map_err(ApiError::scheme)?;
    let expected_ports = if private_output { 6 } else { 7 };
    let prefix = format!("output-{output_index}.");
    let mut ports = BTreeMap::new();
    for name in result.outputs.keys().filter(|name| name.starts_with(&prefix)) {
        let port = name
            .strip_prefix(&prefix)
            .and_then(|port| port.parse::<usize>().ok())
            .ok_or_else(|| ApiError::input("verification output port is malformed"))?;
        if port >= expected_ports || ports.insert(port, name.clone()).is_some() {
            return Err(ApiError::input("verification output port is out of range"));
        }
    }
    if ports.len() != expected_ports || ports.keys().copied().ne(0..expected_ports) {
        return Err(ApiError::input(format!(
            "verification output has {} ports, expected {expected_ports}",
            ports.len()
        )));
    }

    let mut runtime_inputs = BTreeMap::new();
    for (port, name) in ports {
        let value = result
            .outputs
            .get(&name)
            .ok_or_else(|| ApiError::input("verification output port is missing"))?;
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("verification output port is not a matrix"));
        }
        let input_name = match port {
            0..=3 => format!("verification-key-{port}"),
            4..=5 => format!("verification-row-{}", port - 4),
            6 => "verification-payload".to_owned(),
            _ => unreachable!("port range checked above"),
        };
        runtime_inputs.insert(input_name, value.clone());
    }

    let key_ports = (0..4)
        .map(|port| output_ring.input(format!("verification-key-{port}"), (1, columns)))
        .collect::<Vec<_>>();
    let key = RnsBggPublicKey::new(
        wire,
        (
            (key_ports[0].clone(), key_ports[1].clone()),
            (key_ports[2].clone(), key_ports[3].clone()),
        ),
    );
    let row = (
        output_ring.input("verification-row-0", (1, columns)),
        output_ring.input("verification-row-1", (1, columns)),
    );
    let public_payload =
        if private_output { None } else { Some(output_ring.input("verification-payload", (1, 1))) };

    // At source level l, the LUT switching target contributes
    // Delta_l * Delta_(l+1) * mu * r * G_(l+2). The one production
    // ModDown removes Delta_l. The finalized carrier therefore obeys the
    // ordinary level-(l+1) relation, with its own scale and gadget width.
    // Verify that target-ring relation directly; extending a finalized
    // carrier cannot recover its pre-ModDown representative.
    let expected = plaintext_constant(&output_ring, expected);
    let public_payload_residual =
        public_payload.as_ref().map(|payload| payload.clone() - expected.clone());
    let encoding = RnsBggEncoding::new(key, row, public_payload);
    let residual = encoding
        .verification_residual(parameters, output_level, payload, mask, expected)
        .map_err(ApiError::scheme)?;
    // The backend decoder consumes one polynomial (entry 0,0). Check every
    // gadget column, including the high powers which distinguish plaintexts.
    let mut graph = DslContext::new("rns-bgg-output-verification");
    let mut residual_names = Vec::with_capacity(2 * columns);
    for (component, row) in [residual.0, residual.1].into_iter().enumerate() {
        for column in 0..columns {
            let name = format!("verification-residual.{component}.{column}");
            let value = row.clone().slice(
                None,
                Some(mxx_ir_core::node::IndexRange {
                    start: IntExpr::constant(column),
                    end: IntExpr::constant(column + 1),
                }),
            );
            graph = graph.output(name.clone(), value).map_err(ApiError::scheme)?;
            residual_names.push(name);
        }
    }
    if let Some(payload_residual) = public_payload_residual {
        graph = graph
            .output("verification-payload-residual", payload_residual)
            .map_err(ApiError::scheme)?;
    }
    let graph = graph.build().map_err(ApiError::scheme)?;
    let mut runtime_inputs = runtime_inputs;
    for binding in [&payload.binding, &mask.binding].into_iter().flatten() {
        binding.bind(&mut runtime_inputs);
    }
    let runtime_inputs = declared_runtime_inputs(&graph, runtime_inputs);
    let verification = crate::api::execute_graph(
        StageGraph::new(graph, runtime_inputs, BTreeMap::new()),
        &mxx_ir_core::ParamEnv::default(),
        execution,
        store,
        |_| Ok(()),
    )?;
    let output_parameters = parameters.level(output_level).map_err(ApiError::scheme)?;
    let modulus = BigInt::from(output_parameters.modulus().as_ref().clone());
    let download = |name: &str| -> Result<DCRTPolyMatrix, ApiError> {
        let value = verification
            .outputs
            .get(name)
            .ok_or_else(|| ApiError::input("output verification residual is missing"))?;
        execution.download_matrix(value)
    };
    for name in residual_names {
        let residual = download(&name)?;
        let decoded = threshold_decode_coefficients(
            residual.entry(0, 0).coeffs_biguints(),
            &modulus,
            &BigInt::from(2u8),
            output_parameters.ring_dimension() as usize,
        );
        if decoded.iter().any(|value| value != &BigInt::from(0u8)) {
            return Err(ApiError::input("verification residual is nonzero"));
        }
    }
    if !private_output {
        let residual = download("verification-payload-residual")?;
        if residual.entry(0, 0).coeffs_biguints().iter().any(|value| !value.is_zero()) {
            return Err(ApiError::input("public payload does not match expected plaintext"));
        }
    }
    Ok(())
}

/// The family of input encodings whose fields are the imported families:
/// four public-key parts, two row parts, and an optional payload.
fn encoding_family(
    wire: WireId,
    key_parts: &[Family<Mat>],
    rows: &[Family<Mat>],
    payload: Option<&Family<Mat>>,
    count: IntExpr,
) -> Result<Family<RnsBggEncoding>, ApiError> {
    let ([k0, k1, k2, k3], [r0, r1]) = (key_parts, rows) else {
        return Err(ApiError::input("an input encoding family has the wrong field count"));
    };
    let first = |family: &Family<Mat>| family.at(Int::constant(0));
    let element = RnsBggEncoding::new(
        RnsBggPublicKey::new(wire, ((first(k0), first(k1)), (first(k2), first(k3)))),
        (first(r0), first(r1)),
        payload.map(first),
    );
    let fields = key_parts
        .iter()
        .chain(rows)
        .chain(payload)
        .map(|family| family.value_handle().clone())
        .collect::<Vec<_>>();
    family_of_fields(&element, count, &fields).map_err(ApiError::scheme)
}

/// A family bound for online evaluation.
///
/// A stored family holds complete members. An input family holds only what
/// encryption stores, each member's row and, for a public input, its payload;
/// a member read samples the member's public key from the hash key in place
/// (`sampled_key`), so no stage materializes the keys of a whole family.
#[derive(Clone)]
pub enum OnlineWireFamily {
    Stored(Family<RnsBggEncoding>),
    Input {
        wire: WireId,
        level: usize,
        knowledge: AttributeKnowledge,
        hash_key: Bytes,
        rows: Family<RnsBggRow>,
        payloads: Option<Family<Mat>>,
    },
}

/// A family RHS bound for online evaluation: the stored bottom row of each
/// member's compact ciphertext. A member read samples its top row from the
/// hash key in place (`rhs_top`).
#[derive(Clone)]
pub struct OnlineRhsFamily {
    rhs: RhsId,
    program: crate::program::ExponentLutProgramId,
    level: usize,
    hash_key: Bytes,
    bottoms: Family<RnsBggRow>,
}

/// Family execution over graph families, which derives the hash-sampled parts
/// of input and RHS family members where they are read. Online evaluation and
/// preprocessing both use it.
pub struct OnlineFamilies<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
}

impl OnlineFamilies<'_> {
    fn member(
        &self,
        family: &OnlineWireFamily,
        index: IntExpr,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        match family {
            OnlineWireFamily::Stored(family) => Ok(family.at(Int::evaluate(index))),
            OnlineWireFamily::Input { wire, level, knowledge, hash_key, rows, payloads } => {
                let key = super::public_key::sampled_key(
                    self.parameters,
                    *level,
                    hash_key.clone(),
                    *wire,
                    *knowledge,
                    Some(index.clone()),
                )?;
                let index = Int::evaluate(index);
                let payload = payloads.as_ref().map(|payloads| payloads.at(index.clone()));
                Ok(RnsBggEncoding::new(key, rows.at(index), payload))
            }
        }
    }

    fn count(family: &OnlineWireFamily) -> IntExpr {
        match family {
            OnlineWireFamily::Stored(family) => family.count().clone(),
            OnlineWireFamily::Input { rows, .. } => rows.count().clone(),
        }
    }
}

impl<'a, 'b> crate::program::FamilyExecution<EncodingLowering<'a>> for OnlineFamilies<'b> {
    type WireFamily = OnlineWireFamily;
    type RhsFamily = OnlineRhsFamily;

    fn wire_at(
        &mut self,
        family: &Self::WireFamily,
        index: &IntExpr,
    ) -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>> {
        self.member(family, index.clone()).map_err(crate::program::ProgramLowerError::Backend)
    }

    fn rhs_at(
        &mut self,
        family: &Self::RhsFamily,
        index: &IntExpr,
    ) -> Result<RnsBggMatrix, crate::program::ProgramLowerError<RnsBggError>> {
        let top = super::public_key::rhs_top(
            self.parameters,
            family.program,
            family.hash_key.clone(),
            family.rhs,
            family.level,
            Some(index.clone()),
        )
        .map_err(crate::program::ProgramLowerError::Backend)?;
        Ok((top, family.bottoms.at(Int::evaluate(index.clone()))))
    }

    fn parallel(
        &mut self,
        evaluator: &mut EncodingLowering<'a>,
        range: &crate::program::FamilyRange,
        body: impl FnOnce(
            &mut Self,
            &mut EncodingLowering<'a>,
            IntExpr,
        )
            -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>>,
    ) -> Result<Self::WireFamily, crate::program::ProgramLowerError<RnsBggError>> {
        crate::program::graph_parallel(self, evaluator, range, body).map(OnlineWireFamily::Stored)
    }

    fn sum(
        &mut self,
        evaluator: &mut EncodingLowering<'a>,
        family: &Self::WireFamily,
    ) -> Result<RnsBggEncoding, crate::program::ProgramLowerError<RnsBggError>> {
        use crate::program::ProgramLowerError;
        crate::program::graph_sum(evaluator, Self::count(family), |index| {
            let index = index.expression().map_err(ProgramLowerError::Dsl)?;
            self.member(family, index).map_err(ProgramLowerError::Backend)
        })
    }
}

/// Import every RHS of `program` from `preprocess`, for binding before the
/// Program is lowered. Preprocessing publishes the bottom row of each compact
/// RHS at the level after its consumer's, so it is imported once at that
/// level; a family RHS is one family per row port, and each member's top row
/// is sampled where the member is read. `name` maps a role to its artifact
/// name.
fn import_rhs_bindings(
    parameters: &RnsBggParameters,
    program: &ExponentLutProgram,
    workload: &RnsBggDerivedWorkload,
    preprocess: &ProductionId,
    hash_key: &Bytes,
    name: impl Fn(String) -> String,
) -> Result<Vec<(RhsId, InputBinding<RnsBggMatrix, OnlineRhsFamily>)>, ApiError> {
    let mut bindings = Vec::with_capacity(program.rhs_inputs().len());
    for (rhs, shape) in program.rhs_inputs() {
        let consumer_level = workload.rhs_level(*rhs).map_err(ApiError::scheme)?;
        let level = consumer_level.checked_add(1).ok_or_else(|| {
            ApiError::scheme(RnsBggError::Parameters("RHS target level overflow"))
        })?;
        let ring = parameters.ring(level).map_err(ApiError::scheme)?;
        let columns = parameters.component_columns(level).map_err(ApiError::scheme)?;
        let binding = match shape {
            InputBinding::Scalar(()) => {
                let port = |port: usize| {
                    ring.artifact_input(
                        preprocess.clone(),
                        name(format!("rhs/{rhs:?}.{port}")),
                        (1, columns),
                        ArtifactAvailability::Transferred,
                    )
                };
                let top = super::public_key::rhs_top(
                    parameters,
                    program.id(),
                    hash_key.clone(),
                    *rhs,
                    consumer_level,
                    None,
                )
                .map_err(ApiError::scheme)?;
                InputBinding::Scalar((top, (port(0), port(1))))
            }
            InputBinding::Family(count) => {
                let ports = (0..2)
                    .map(|port| {
                        ring.family_artifact_input(
                            preprocess.clone(),
                            name(format!("rhs/{rhs:?}/family.{port}")),
                            count.clone(),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    })
                    .collect::<Vec<_>>();
                let element: RnsBggRow =
                    (ports[0].at(Int::constant(0)), ports[1].at(Int::constant(0)));
                let fields =
                    ports.iter().map(|family| family.value_handle().clone()).collect::<Vec<_>>();
                InputBinding::Family(OnlineRhsFamily {
                    rhs: *rhs,
                    program: program.id(),
                    level: consumer_level,
                    hash_key: hash_key.clone(),
                    bottoms: family_of_fields(&element, count.clone(), &fields)
                        .map_err(ApiError::scheme)?,
                })
            }
        };
        bindings.push((*rhs, binding));
    }
    Ok(bindings)
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
                    return Err(ApiError::input("refresh child runtime input conflicts"));
                }
                _ => return Err(ApiError::input("refresh child runtime input schema")),
            },
            // A secret input name is unique to one sampled secret.
            Some(RuntimeValue::Matrix(_)) if matches!(value, RuntimeValue::Matrix(_)) => {}
            Some(_) => return Err(ApiError::input("refresh child runtime input schema")),
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
        |_| Ok(()),
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
    let s = reduce_secret(parameters, &mask.row, level)?;
    let gadget = parameters.public_gadget(level)?;
    let matrix = key.matrix.clone();
    let mask_matrix = (
        s.0.clone() * matrix.0.0 + s.1.clone() * matrix.1.0,
        s.0.clone() * matrix.0.1 + s.1.clone() * matrix.1.1,
    );
    let public_scale = plaintext.clone() * gadget;
    let error = fresh_error(parameters, level, parameters.level(level)?.modulus_digits())?;
    let row = (
        mask_matrix.0 - s.0 * public_scale.clone() + error.0,
        mask_matrix.1 - s.1 * public_scale + error.1,
    );
    Ok(RnsBggEncoding::new(key.clone(), row, Some(plaintext)))
}

/// Online lowering state.  Its artifact fields are metadata and lazy handles,
/// never resident matrix payloads.
pub struct EncodingLowering<'a> {
    pub(crate) parameters: &'a RnsBggParameters,
    pub(crate) program: Arc<ExponentLutProgram>,
    pub(crate) workload: Arc<RnsBggDerivedWorkload>,
    pub(crate) preprocess: ProductionId,
    pub(crate) encryption: ProductionId,
    pub(crate) manifests: BTreeMap<ProductionId, Manifest>,
    pub(crate) refresh_prf_namespace: Option<([u8; 32], usize)>,
    pub(crate) current_site: Option<LoweringSite>,
    /// Graph input of the shared hash key, from which `A_sigma` is sampled.
    pub(crate) hash_key: Bytes,
}

impl EncodingLowering<'_> {
    fn preprocess_artifact_name(&self, role: String) -> String {
        self.refresh_prf_namespace.map_or(role.clone(), |(commitment, slot)| {
            refresh_prf_artifact_name(commitment, slot, &role)
        })
    }

    fn encryption_artifact_name(&self, role: String) -> String {
        self.refresh_prf_namespace.map_or(role.clone(), |(commitment, slot)| {
            refresh_prf_artifact_name(commitment, slot, &role)
        })
    }

    /// Compute the native Walsh/automorphism branch polynomial for one unary
    /// LUT branch.  The coefficient depends on the complete table, not only
    /// its width; public-key preparation uses this same helper.
    pub(crate) fn lut_coefficient(
        parameters: &RnsBggParameters,
        level: usize,
        input_layout: ExpLayout,
        representation: crate::program::LutOutputRepresentation,
        sigma: impl Into<IntExpr>,
        values: &[usize],
    ) -> Result<Mat, RnsBggError> {
        let width = values.len();
        if width == 0 || !width.is_power_of_two() {
            return Err(RnsBggError::Lut);
        }
        let ring = parameters.ring(level)?;
        input_layout
            .validate(parameters.level(level)?.ring_dimension() as usize)
            .map_err(|_| RnsBggError::Lut)?;
        if input_layout.width() != width {
            return Err(RnsBggError::Lut);
        }
        let modulus = parameters.level(level)?.modulus().as_ref().clone();
        let inverse =
            mxx_backends::utils::mod_inverse_biguints(&(BigUint::from(width) % &modulus), &modulus)
                .ok_or(RnsBggError::Lut)?;
        let sigma = sigma.into();
        let mut terms = values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, output)| {
                let left = match representation {
                    crate::program::LutOutputRepresentation::Coefficient => {
                        ring.polynomial([IntExpr::constant(BigInt::from(output))])
                    }
                    crate::program::LutOutputRepresentation::Exponent(layout) => {
                        layout.mono(&ring, &BigInt::from(output)).map_err(|_| RnsBggError::Lut)?
                    }
                };
                let right = input_layout
                    .mono(&ring, &-BigInt::from(index))
                    .map_err(|_| RnsBggError::Lut)?
                    .ring_automorphism(sigma.clone());
                Ok(left * right)
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        while terms.len() > 1 {
            let mut next = Vec::with_capacity(terms.len().div_ceil(2));
            let mut iterator = terms.into_iter();
            while let Some(left) = iterator.next() {
                next.push(match iterator.next() {
                    Some(right) => left + right,
                    None => left,
                });
            }
            terms = next;
        }
        Ok(ring.polynomial([IntExpr::constant(BigInt::from(inverse))]) * terms.pop().unwrap())
    }

    /// Evaluate a public LUT payload with the same Walsh/automorphism
    /// coefficients used by the online encoding.  The payload is public, so
    /// the complete evaluation can remain a graph expression; private and
    /// unknown payloads never enter this path.
    fn lut_public_payload(
        &self,
        level: usize,
        payload: Mat,
        table: &LutTable,
    ) -> Result<Mat, RnsBggError> {
        let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
        let input_layout = table.input_layout();
        input_layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let output_level = if matches!(
            table.output_representation(),
            crate::program::LutOutputRepresentation::Coefficient
        ) {
            level
        } else {
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?
        };
        let output_ring = self.parameters.ring(output_level)?;
        let payload = checked_modulus_convert(payload, &output_ring)?;
        let sigmas =
            input_layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        // One loop body over the automorphism indices, as preprocessing builds
        // its switching ciphertexts, so the graph does not grow with the
        // square of the table width.
        let terms = mxx_dsl::parallel(IntExpr::constant(sigmas.len()), |index| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let coefficient = Self::lut_coefficient(
                self.parameters,
                output_level,
                input_layout,
                table.output_representation(),
                sigma.clone(),
                table.values(),
            )
            .map_err(|_| DslError::Schema)?;
            Ok(coefficient * payload.clone().ring_automorphism(sigma))
        })
        .map_err(graph)?;
        crate::utils::balanced_sum_family(terms, |lhs, rhs| lhs + rhs).map_err(graph)
    }

    fn site(&self) -> Result<&LoweringSite, RnsBggError> {
        self.current_site.as_ref().ok_or(RnsBggError::Binding("missing lowering site"))
    }

    /// The encryption's LUT helpers of `level` for `table`'s input layout,
    /// member `i` for its `i`-th automorphism.
    fn lut_helpers(
        &self,
        level: usize,
        table: &LutTable,
        count: usize,
    ) -> Result<(Family<Mat>, Family<Mat>), RnsBggError> {
        let ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(level)?;
        let layout = table.input_layout();
        let member = |port: usize| {
            ring.family_artifact_input(
                self.encryption.clone(),
                self.encryption_artifact_name(format!("lut-helper/{level}/{layout:?}.{port}")),
                count,
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
        Ok((member(0), member(1)))
    }

    fn lut_public_data(&self, level: usize, sigma: Int) -> Result<RnsBggMatrix, RnsBggError> {
        super::public_key::lut_public_matrix(self.parameters, self.hash_key.clone(), level, sigma)
    }

    fn rhs_decomposition(
        &self,
        rhs: &RnsBggMatrix,
        evaluation_level: usize,
    ) -> Result<(SmallMatrix, SmallMatrix), RnsBggError> {
        // Decompose immediately before the RHS action consumes the compact
        // artifact.  The resulting bounded digits are transported to the
        // evaluation ring, but never persisted at the stage boundary.
        let decomposition_level = evaluation_level
            .checked_add(1)
            .ok_or(RnsBggError::Parameters("RHS target level overflow"))?;
        compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            rhs.clone(),
        )
    }

    /// The imported compact `B_sigma` switching ciphertexts of one LUT, one
    /// family per matrix port, with its automorphism branch indices.
    fn lut_switching(
        &self,
        input: &RnsBggEncoding,
        lut_id: crate::program::LutId,
        table: &LutTable,
    ) -> Result<(Vec<Family<Mat>>, Vec<usize>), RnsBggError> {
        let count = IntExpr::constant(table.values().len());
        let evaluation_level = self.workload.level(input.wire())?;
        let decomposition_level =
            evaluation_level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?;
        let decomposition_ring = self.parameters.ring(decomposition_level)?;
        let input_layout = table.input_layout();
        let ring_dimension = self.parameters.level(evaluation_level)?.ring_dimension() as usize;
        input_layout.validate(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let target_columns = self.parameters.component_columns(decomposition_level)?;
        let sigma_indices =
            input_layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
        let families = (0..4)
            .map(|port| {
                decomposition_ring.family_artifact_input(
                    self.preprocess.clone(),
                    self.preprocess_artifact_name(format!(
                        "lut-switching/{evaluation_level}/{decomposition_level}/{lut_id:?}/{input_layout:?}.{port}"
                    )),
                    count.clone(),
                    (1, target_columns),
                    ArtifactAvailability::Transferred,
                )
            })
            .collect::<Vec<_>>();
        Ok((families, sigma_indices))
    }

    /// For a LUT branch with automorphism `sigma` and switching ciphertext
    /// `b_sigma`: the routed input row switched by `b_sigma`, and the
    /// decomposition of the switched public key. The switching decomposition acts on the
    /// routed row and public key in one product each.
    fn lut_branch_action(
        &self,
        input: &RnsBggEncoding,
        b_sigma: RnsBggMatrix,
        sigma: IntExpr,
    ) -> Result<(RnsBggRow, (SmallMatrix, SmallMatrix)), DslError> {
        let evaluation_level = self.workload.level(input.wire()).map_err(|_| DslError::Schema)?;
        let decomposition_level = evaluation_level.checked_add(1).ok_or(DslError::Schema)?;
        let target_ring =
            self.parameters.ring(decomposition_level).map_err(|_| DslError::Schema)?;
        let switching_decomposition = compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            b_sigma.clone(),
        )
        .map_err(|_| DslError::Schema)?;
        let routed = map_matrix(input.public_key.matrix.clone(), |entry| {
            entry.ring_automorphism(sigma.clone())
        });
        let routed_row = (
            input.row.0.clone().ring_automorphism(sigma.clone()),
            input.row.1.clone().ring_automorphism(sigma.clone()),
        );
        let (switched_row, switched_matrix) =
            multiply_row_and_public_by_columns(routed_row, routed, switching_decomposition);
        let switched_public_key = mod_down_public(switched_matrix, &target_ring);
        let switched_key_decomposition = compact_decompose_columns(
            self.parameters,
            evaluation_level,
            decomposition_level,
            switched_public_key,
        )
        .map_err(|_| DslError::Schema)?;
        Ok((switched_row, switched_key_decomposition))
    }
    fn refresh_action(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
        role: &str,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        self.refresh_action_for_label(declaration, None, slot, role)
    }

    fn refresh_action_for_label(
        &self,
        declaration: &RefreshDeclaration,
        label: Option<usize>,
        slot: usize,
        role: &str,
    ) -> Result<RnsBggDecomposition, RnsBggError> {
        let q_level = refresh_slot_q_level(self.parameters, declaration, slot)?;
        let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
        let ring = self.parameters.ring(p_level)?;
        // The preprocess stage stores the compact p-ring action matrix.  It
        // is decomposed only at this consuming q-level; no decomposition is
        // persisted in the stage boundary.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let prefix = match label {
            Some(label) => refresh_label_artifact_name(*declaration.commitment(), slot, label),
            None => refresh_artifact_name(*declaration.commitment(), slot, role),
        };
        let matrix = |port: usize| {
            ring.artifact_input(
                self.preprocess.clone(),
                format!("{prefix}.{port}"),
                (1, columns),
                ArtifactAvailability::Transferred,
            )
        };
        let matrix = ((matrix(0), matrix(1)), (matrix(2), matrix(3)));
        compact_decompose_matrix(self.parameters, q_level, p_level, matrix)
    }

    fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        let ring = self.parameters.ring(refresh_slot_level(self.parameters, declaration, slot)?)?;
        // The decoder is a target-matrix artifact.  Keep the slot p-level for
        // its ring and decomposition contract, but use the declared target
        // level for the matrix width.
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let prefix = refresh_artifact_name(*declaration.commitment(), slot, "decoder");
        Ok((
            ring.artifact_input(
                self.encryption.clone(),
                format!("{prefix}.0"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
            ring.artifact_input(
                self.encryption.clone(),
                format!("{prefix}.1"),
                (1, columns),
                ArtifactAvailability::Transferred,
            ),
        ))
    }

    fn refresh_crt_weight(
        &self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<Mat, RnsBggError> {
        let mut product = BigUint::from(1u8);
        for value in declaration.slots() {
            product *= value
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        }
        let selected = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?
            .q_t
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        let base = &product / &selected;
        let inverse = mxx_backends::utils::mod_inverse_biguints(&(&base % &selected), &selected)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        Ok(self
            .parameters
            .ring(declaration.target_level())?
            .polynomial([IntExpr::constant(BigInt::from(base * inverse))]))
    }

    /// Reconstruct the public refresh target from the public per-slot p-ring
    /// anchor emitted by preprocessing.  The anchor is reduced to `q_t`, centered
    /// extended to `Q*`, and then CRT recomposed.  The decoder's
    /// `A_sum-H*A_star` term is deliberately kept separate in the encryption
    /// graph; it is not part of the public target key.
    fn refresh_public_target(
        &self,
        declaration: &RefreshDeclaration,
        wire: WireId,
    ) -> Result<RnsBggPublicKey, RnsBggError> {
        let level = declaration.target_level();
        let target_ring = self.parameters.ring(level)?;
        let production = self.preprocess.clone();
        let commitment = *declaration.commitment();
        let mut values = Vec::with_capacity(declaration.slots().len());
        let mut coefficients = Vec::with_capacity(declaration.slots().len());
        for slot in declaration.slots() {
            let slot_modulus = slot
                .p
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
            let source_level = self
                .parameters
                .levels()
                .iter()
                .position(|candidate| candidate.modulus().as_ref() == &slot_modulus)
                .ok_or(RnsBggError::Binding("refresh slot modulus is not configured"))?;
            // A-star is a public target-matrix artifact.  Its p-ring is tied
            // to the slot's source modulus, while its matrix width is the
            // declared refresh target width.
            let target_columns = self.parameters.component_columns(level)?;
            let ring = self.parameters.ring(source_level)?;
            let matrix = |role: &str| -> RnsBggMatrix {
                (
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.0", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.1", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.2", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            format!("{}.3", refresh_artifact_name(commitment, slot.slot, role)),
                            (1, target_columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                )
            };
            let a_star = matrix("a-star");
            let target = reduce_anchor_channel(
                a_star,
                &self.parameters.ring_for_modulus(&slot.q_t)?,
                &target_ring,
            )?;
            values.push(target);
            coefficients.push(self.refresh_crt_weight(declaration, slot.slot)?);
        }
        let values = Family::pack(values).map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let coefficients =
            Family::pack(coefficients).map_err(|error| RnsBggError::Graph(error.to_string()))?;
        let matrix = recompose_public(values, coefficients)?;
        Ok(RnsBggPublicKey::new(wire, matrix))
    }
    fn finish_outputs(
        &mut self,
        outputs: Vec<RnsBggEncoding>,
    ) -> Result<Vec<RnsBggEncoding>, RnsBggError> {
        outputs
            .into_iter()
            .map(|output| {
                if self
                    .program
                    .wire_is_coefficient_output(output.wire())
                    .map_err(|_| RnsBggError::Binding("invalid output wire metadata"))?
                {
                    let level = self.workload.level(output.wire())?;
                    let target = level
                        .checked_add(1)
                        .ok_or(RnsBggError::Parameters("coefficient output level overflow"))?;
                    let ring = self.parameters.ring(target)?;
                    let public_key = RnsBggPublicKey::new(
                        output.public_key.wire,
                        mod_down_public(output.public_key.matrix, &ring),
                    );
                    Ok(RnsBggEncoding::new(
                        public_key,
                        mod_down_encoding(output.row, &ring),
                        output
                            .public_payload
                            .map(|payload| checked_modulus_convert(payload, &ring))
                            .transpose()?,
                    ))
                } else {
                    Ok(output)
                }
            })
            .collect()
    }

    fn checked_payload(
        &self,
        wire: WireId,
        payload: Option<Mat>,
    ) -> Result<Option<Mat>, RnsBggError> {
        let public = matches!(
            self.program.wire_knowledge(wire).map_err(|_| RnsBggError::Binding("wire metadata"))?,
            AttributeKnowledge::PublicPayload
        );
        if public != payload.is_some() {
            return Err(RnsBggError::Binding("public payload does not match Program metadata"));
        }
        Ok(payload)
    }
}

impl ExponentLUTProgramLowering for EncodingLowering<'_> {
    type Wire = RnsBggEncoding;
    type Rhs = RnsBggMatrix;
    type Error = RnsBggError;
    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire) -> Result<Self::Wire, Self::Error> {
        if self.workload.level(lhs.wire())? != self.workload.level(rhs.wire())? {
            return Err(RnsBggError::Binding("Add key level mismatch"));
        }
        let public_payload = match (&lhs.public_payload, &rhs.public_payload) {
            (Some(lhs), Some(rhs)) => Some(lhs.clone() + rhs.clone()),
            (None, None) => None,
            _ => return Err(RnsBggError::Binding("mixed public/private Add")),
        };
        let output = self.site()?.output;
        let public_key = RnsBggPublicKey::new(
            output,
            add_public_matrices(lhs.public_key.matrix.clone(), rhs.public_key.matrix.clone()),
        );
        let public_payload = self.checked_payload(output, public_payload)?;
        Ok(RnsBggEncoding::new(
            public_key,
            add_encoding_rows(lhs.row.clone(), rhs.row.clone()),
            public_payload,
        ))
    }
    fn mul_public(
        &mut self,
        lhs: &Self::Wire,
        rhs: &Self::Wire,
    ) -> Result<Self::Wire, Self::Error> {
        let lhs_knowledge = self
            .program
            .wire_knowledge(lhs.wire())
            .map_err(|_| RnsBggError::Binding("MulPublic LHS metadata"))?;
        let rhs_knowledge = self
            .program
            .wire_knowledge(rhs.wire())
            .map_err(|_| RnsBggError::Binding("MulPublic RHS metadata"))?;
        if lhs_knowledge != AttributeKnowledge::PublicPayload {
            return Err(RnsBggError::Binding("MulPublic LHS must be public"));
        }
        let lhs_level = self.workload.level(lhs.wire())?;
        let rhs_level = self.workload.level(rhs.wire())?;
        let target_level =
            self.parameters.mul_public_target_level(lhs_level, rhs_level, rhs_knowledge)?;
        let target_ring = self.parameters.ring(target_level)?;

        // The public LHS is aligned to the private RHS-owned level. The RHS
        // remains untouched when private; public/public additionally aligns
        // its public carrier to the selected lower modulus.
        let lhs_matrix = apply_public_alignment_matrix(
            self.parameters,
            lhs_level,
            target_level,
            lhs.public_key.matrix.clone(),
        )?;
        let lhs_row =
            apply_public_alignment_row(self.parameters, lhs_level, target_level, lhs.row.clone())?;
        let rhs_matrix = if rhs_knowledge == AttributeKnowledge::PublicPayload {
            apply_public_alignment_matrix(
                self.parameters,
                rhs_level,
                target_level,
                rhs.public_key.matrix.clone(),
            )?
        } else {
            rhs.public_key.matrix.clone()
        };
        let rhs_row = if rhs_knowledge == AttributeKnowledge::PublicPayload {
            apply_public_alignment_row(self.parameters, rhs_level, target_level, rhs.row.clone())?
        } else {
            rhs.row.clone()
        };
        let lhs_payload = checked_modulus_convert(
            lhs.public_payload.clone().ok_or(RnsBggError::Binding("public LHS payload missing"))?,
            &target_ring,
        )?;
        let rhs_payload = rhs
            .public_payload
            .clone()
            .map(|payload| checked_modulus_convert(payload, &target_ring))
            .transpose()?;
        // One decomposition of the RHS key acts on the LHS row and key.
        let columns =
            compact_decompose_columns(self.parameters, target_level, target_level, rhs_matrix)?;
        let (acted_row, acted_matrix) =
            multiply_row_and_public_by_columns(lhs_row, lhs_matrix, columns);
        let row = add_encoding_rows(
            acted_row,
            (rhs_row.0.clone() * lhs_payload.clone(), rhs_row.1.clone() * lhs_payload.clone()),
        );
        let output = self.site()?.output;
        let public_key = RnsBggPublicKey::new(output, acted_matrix);
        let output_payload = rhs_payload.map(|rhs_payload| lhs_payload * rhs_payload);
        Ok(RnsBggEncoding::new(public_key, row, self.checked_payload(output, output_payload)?))
    }
    fn mul_private(
        &mut self,
        input: &Self::Wire,
        rhs: &Self::Rhs,
    ) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire())
            .map_err(|_| RnsBggError::Binding("MulPrivate input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG MulPrivate requires a private input"));
        }
        let evaluation_level = self.workload.level(input.wire())?;
        let action = self.rhs_decomposition(rhs, evaluation_level)?;
        let output = self.site()?.output;
        let target_level = self.workload.level(output)?;
        let target_ring = self.parameters.ring(target_level)?;
        let (row, matrix) = multiply_row_and_public_by_columns(
            input.row.clone(),
            input.public_key.matrix.clone(),
            action,
        );
        let public_key = RnsBggPublicKey::new(output, mod_down_public(matrix, &target_ring));
        Ok(RnsBggEncoding::new(
            public_key,
            mod_down_encoding(row, &target_ring),
            self.checked_payload(output, None)?,
        ))
    }
    fn lut(&mut self, input: &Self::Wire, table: &LutTable) -> Result<Self::Wire, Self::Error> {
        if self
            .program
            .wire_knowledge(input.wire())
            .map_err(|_| RnsBggError::Binding("LUT input metadata"))? ==
            AttributeKnowledge::PublicPayload
        {
            return Err(RnsBggError::Binding("RNS-BGG LUT requires a private input"));
        }
        let lut_id = self.program.lut_id_at(self.site()?.output).map_err(|_| RnsBggError::Lut)?;
        let level = self.workload.level(input.wire())?;
        let (switching, sigmas) = self.lut_switching(input, lut_id, table)?;
        let output = self.site()?.output;
        let helper = self.lut_helpers(level, table, sigmas.len())?;
        // Accumulate the branches sequentially, as preprocessing does: the sum
        // and its noise bound equal the balanced tree, while only one branch,
        // including its sampled A_sigma, is resident at a time.
        let evaluation_ring = self.parameters.ring(level)?;
        let columns = self.parameters.component_columns(
            level.checked_add(1).ok_or(RnsBggError::Parameters("LUT level overflow"))?,
        )?;
        let zero = evaluation_ring.zero((1, columns));
        let initial =
            ((zero.clone(), zero.clone()), ((zero.clone(), zero.clone()), (zero.clone(), zero)));
        let b_at = |index: Int| {
            (
                (switching[0].at(index.clone()), switching[1].at(index.clone())),
                (switching[2].at(index.clone()), switching[3].at(index)),
            )
        };
        let this = &*self;
        let branch = |index: Int, (row_sum, key_sum), b_sigma| {
            let sigma = IntExpr::Select {
                selector: Box::new(index.expression()?),
                branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
            };
            let (switched_row, switched_key_decomposition) =
                this.lut_branch_action(input, b_sigma, sigma.clone())?;
            let mask = (helper.0.at(index.clone()), helper.1.at(index));
            let public_data =
                this.lut_public_data(level, Int::from(sigma)).map_err(|_| DslError::Schema)?;
            let (masked, switched_key) =
                multiply_row_and_public_by_columns(mask, public_data, switched_key_decomposition);
            let row = add_encoding_rows(row_sum, add_encoding_rows(masked, switched_row));
            let public_key = add_public_matrices(key_sum, switched_key);
            Ok::<_, DslError>((row, public_key))
        };
        // Each iteration first reads the next branch's switching ciphertext
        // and carries it to the next iteration, so that load overlaps this
        // branch's work; the last branch runs after the loop.
        let last = table.values().len().checked_sub(1).ok_or(RnsBggError::Lut)?;
        let (sums, b_last) = mxx_dsl::iterate(
            IntExpr::constant(last),
            (initial, b_at(Int::constant(0))),
            |index, (sums, b_sigma)| {
                let next = b_at(index.clone().add(1usize));
                Ok((branch(index, sums, b_sigma)?, next))
            },
        )
        .map_err(graph)?;
        let (row, public_key) = branch(Int::constant(last), sums, b_last).map_err(graph)?;
        let mut public_key = RnsBggPublicKey::new(output, public_key);
        let mut row = row;
        if !self
            .program
            .wire_is_coefficient_output(output)
            .map_err(|_| RnsBggError::Binding("coefficient output metadata"))?
        {
            let target = self.workload.level(output)?;
            let ring = self.parameters.ring(target)?;
            public_key.matrix = mod_down_public(public_key.matrix, &ring);
            row = mod_down_encoding(row, &ring);
        }
        let public_payload = if matches!(
            self.program
                .wire_knowledge(output)
                .map_err(|_| RnsBggError::Binding("wire metadata"))?,
            AttributeKnowledge::PublicPayload
        ) {
            let payload = input
                .public_payload
                .clone()
                .ok_or(RnsBggError::Binding("public LUT input payload missing"))?;
            let payload = self.lut_public_payload(level, payload, table)?;
            let target_level = self.workload.level(output)?;
            Some(checked_modulus_convert(payload, &self.parameters.ring(target_level)?)?)
        } else {
            None
        };
        Ok(RnsBggEncoding::new(public_key, row, self.checked_payload(output, public_payload)?))
    }
}

impl EncodingLowering<'_> {
    fn validate_refresh_declaration(
        &self,
        declaration: &RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.source_program() != self.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        self.program
            .wire_is_coefficient_output(declaration.source_output())
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
            .and_then(|coefficient| {
                if coefficient {
                    Err(RnsBggError::Refresh(RefreshError::InvalidSourceOutput))
                } else {
                    Ok(())
                }
            })?;
        self.workload.level(declaration.source_output())?;
        self.parameters.level(declaration.target_level())?;
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        Ok(())
    }
}

impl ProgramLoweringContext for EncodingLowering<'_> {
    fn replace_site(&mut self, site: Option<LoweringSite>) -> Option<LoweringSite> {
        std::mem::replace(&mut self.current_site, site)
    }
}

impl GraphParallelLowering for EncodingLowering<'_> {
    type SideOutput = Vec<Mat>;
    fn begin_graph_parallel(&mut self, _: &IntExpr, _: &IntExpr) {}
    fn finish_graph_parallel(&mut self) -> Result<Self::SideOutput, Self::Error> {
        Ok(Vec::new())
    }
    fn commit_graph_parallel(&mut self, _: Family<Self::SideOutput>) -> Result<(), Self::Error> {
        Ok(())
    }
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
    /// Input plaintexts; a family keeps `plaintext_terms`, read member by member.
    plaintexts: BTreeMap<WireId, InputBinding<Mat, Vec<(BigInt, usize)>>>,
    preprocess: Option<ProductionId>,
    binding: Option<RnsBggArtifactBinding>,
    refresh_declaration: Option<RefreshDeclaration>,
    manifests: BTreeMap<ProductionId, Manifest>,
    runtime: BTreeMap<String, RuntimeValue>,
    refresh_prf_children: BTreeMap<usize, Box<RnsEncryptBuild<'a>>>,
    refresh_prf_binding: Option<([u8; 32], usize, crate::program::ExponentLutProgramId, [u8; 32])>,
}

fn validate_refresh_prf_input_key_artifacts(
    manifests: &BTreeMap<ProductionId, Manifest>,
    production: &ProductionId,
    parameters: &RnsBggParameters,
    inputs: &crate::program::EncryptionInputs,
    commitment: [u8; 32],
    slot: usize,
) -> Result<(), RnsBggError> {
    let manifest = manifests
        .get(production)
        .ok_or(RnsBggError::Binding("refresh PRF preprocess manifest is missing"))?;
    let ring = resolve_ring(parameters.ring(0)?.as_ref())?;
    for (wire, declaration) in inputs.inputs() {
        let columns = parameters.key_component_columns(0, declaration.knowledge())?;
        let artifact_type =
            ArtifactType::Matrix(ConcreteMatrixType { ring: ring.clone(), rows: 1, columns });
        let family_count = match declaration.shape() {
            InputBinding::Scalar(()) => None,
            InputBinding::Family(count) => Some(
                count
                    .evaluate(&ParamEnv::default())
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(RnsBggError::Binding("refresh PRF family count is not concrete"))?,
            ),
        };
        for port in 0..4 {
            let name = refresh_prf_input_key_artifact_name(commitment, slot, *wire, port);
            let artifact = manifest
                .artifacts
                .get(&name)
                .ok_or(RnsBggError::Binding("refresh PRF input-key artifact is missing"))?;
            if artifact.artifact_type != artifact_type || artifact.family_count != family_count {
                return Err(RnsBggError::Binding("refresh PRF input-key artifact schema mismatch"));
            }
        }
    }
    Ok(())
}

impl<'a> RnsEncryptBuild<'a> {
    /// Creates an unbound encryption builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            hash_key: Vec::new(),
            inputs: None,
            projection_identity: None,
            payload_secret: None,
            mask_secret: None,
            plaintexts: BTreeMap::new(),
            preprocess: None,
            binding: None,
            refresh_declaration: None,
            manifests: BTreeMap::new(),
            runtime: BTreeMap::new(),
            refresh_prf_children: BTreeMap::new(),
            refresh_prf_binding: None,
        }
    }

    /// Bind public LHS plaintexts and secrets for a prepared PRF child.
    /// This application-facing method keeps the generic stage trait out of
    /// refresh child setup code.
    pub fn bind_lhs_inputs<P: Poly>(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        <Self as EncryptBuild<P>>::bind_inputs(
            self, inputs, parameters, lhs, payload, mask, hash_key,
        )
    }

    fn bind_preprocess_artifacts_inner(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        let binding = preprocess.descriptor().binding().clone();
        if binding.context() != &self.context ||
            (self.refresh_prf_binding.is_none() &&
                binding.parameters() != &self.parameters.parameter_identity()) ||
            binding.hash_key_commitment() != &self.hash_key_commitment
        {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess binding does not match encryption",
            ));
        }
        let production = preprocess.production().clone();
        if binding.sources().len() != 1 || !binding.sources().contains(&production) {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess production is not a declared source",
            ));
        }
        self.preprocess = Some(production.clone());
        self.manifests = BTreeMap::from([(production.clone(), preprocess.manifest().clone())]);
        if self.inputs.is_none() {
            return Err(ApiError::input("encryption inputs are not bound"));
        }
        self.binding = Some(binding);
        Ok(())
    }

    /// Bind the preprocess artifacts used by a prepared PRF child.
    pub fn bind_preprocess_artifacts(
        &mut self,
        preprocess: &StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(preprocess)
    }

    /// Bind this encryption child to one slot of the prepared PRF Program.
    /// The ordinary encryption checkpoint remains sourced from the outer
    /// Program; this explicit binding records the prepared identity and the
    /// slot projection used when its D0/initial carriers are published.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        program: &ExponentLutProgram,
    ) -> Result<(), RnsBggError> {
        self.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        if slot >= declaration.slots().len() ||
            program.id() != declaration.prepared().program().id()
        {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let source_q = declaration
            .slots()
            .get(slot)
            .and_then(|slot| slot.q.to_biguint())
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        if self
            .parameters
            .level(0)
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?
            .modulus()
            .as_ref() !=
            &source_q
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        let inputs = self
            .inputs
            .as_ref()
            .ok_or(RnsBggError::Binding("refresh PRF child inputs are not bound"))?;
        if inputs.inputs() != program.inputs() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if self.refresh_prf_binding.is_some() {
            return Err(RnsBggError::Binding("refresh PRF child is already bound"));
        }
        self.refresh_prf_binding = Some((
            *declaration.commitment(),
            slot,
            program.id(),
            self.parameters.parameter_identity(),
        ));
        Ok(())
    }

    /// Attach already-bound encryption children for the prepared PRF.
    ///
    /// Child outputs are emitted into the outer graph under a commitment and
    /// slot namespace.  No child checkpoint or second execution is created.
    pub fn bind_refresh_prf_children(
        &mut self,
        declaration: &RefreshDeclaration,
        children: BTreeMap<usize, RnsEncryptBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if children.len() != declaration.slots().len() ||
            children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        if children.keys().any(|slot| self.refresh_prf_children.contains_key(slot)) {
            return Err(RnsBggError::Binding("refresh PRF child slot is already bound"));
        }
        let expected_program = declaration.prepared().program().id();
        for (slot, child) in &children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            let binding = child
                .binding
                .as_ref()
                .ok_or(RnsBggError::Binding("refresh PRF child artifacts are not bound"))?;
            if binding.context() != &self.context ||
                binding.parameters() != &self.parameters.parameter_identity() ||
                binding.hash_key_commitment() != &self.hash_key_commitment ||
                binding.program() != declaration.source_program() ||
                child.inputs.is_none() ||
                child.payload_secret.is_none() ||
                child.mask_secret.is_none() ||
                child.preprocess.is_none() ||
                child.parameters.parameter_identity() != expected_child_parameters ||
                child.refresh_prf_binding.map(|(commitment, child_slot, program, identity)| {
                    commitment == *declaration.commitment() &&
                        child_slot == *slot &&
                        program == expected_program &&
                        identity == expected_child_parameters
                }) != Some(true)
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            validate_refresh_prf_input_key_artifacts(
                &child.manifests,
                child
                    .preprocess
                    .as_ref()
                    .ok_or(RnsBggError::Binding("refresh PRF child preprocess is not bound"))?,
                &child.parameters,
                child
                    .inputs
                    .as_ref()
                    .ok_or(RnsBggError::Binding("refresh PRF child inputs are not bound"))?,
                *declaration.commitment(),
                *slot,
            )?;
            if child
                .refresh_declaration
                .as_ref()
                .is_some_and(|bound| bound.commitment() != declaration.commitment())
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
            if declaration.slots().get(*slot).is_none() {
                return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
            }
        }
        self.refresh_prf_children =
            children.into_iter().map(|(slot, child)| (slot, Box::new(child))).collect();
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }

    fn append_refresh_prf_child(
        child: &RnsEncryptBuild<'a>,
        declaration: &RefreshDeclaration,
        slot: usize,
        mut graph: DslContext,
        hash_key: &Bytes,
    ) -> Result<DslContext, ApiError> {
        let inputs = child
            .inputs
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child inputs are not bound"))?;
        let production = child
            .preprocess
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child preprocess is not bound"))?;
        let payload = child
            .payload_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child payload is not bound"))?;
        let mask = child
            .mask_secret
            .as_ref()
            .ok_or_else(|| ApiError::input("refresh PRF child mask is not bound"))?;
        let commitment = *declaration.commitment();
        for (name, family) in
            child.lut_helper_families(mask, hash_key, inputs).map_err(ApiError::scheme)?
        {
            graph = graph
                .transferred_output(refresh_prf_artifact_name(commitment, slot, &name), family)
                .map_err(ApiError::scheme)?;
        }
        let workload = RnsBggDerivedWorkload::from_program(declaration.prepared().program())
            .map_err(ApiError::scheme)?;
        for (wire, declaration_input) in inputs.inputs() {
            // Each input is encoded at its logical level in the PRF Program.
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = child.parameters.ring(level).map_err(ApiError::scheme)?;
            let columns = child
                .parameters
                .key_component_columns(level, declaration_input.knowledge())
                .map_err(ApiError::scheme)?;
            let key = RnsBggPublicKey {
                wire: *wire,
                matrix: (
                    (
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 0),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 1),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                    (
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 2),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                        ring.artifact_input(
                            production.clone(),
                            refresh_prf_input_key_artifact_name(commitment, slot, *wire, 3),
                            (1, columns),
                            ArtifactAvailability::Cached,
                        ),
                    ),
                ),
            };
            let value = child
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("refresh PRF child plaintext is not bound"))?;
            let encode = |key: RnsBggPublicKey, plaintext: Mat| {
                let plaintext = checked_modulus_convert(plaintext, &ring)?;
                if matches!(declaration_input.knowledge(), AttributeKnowledge::PublicPayload) {
                    child.encode_public(&key, level, mask, plaintext)
                } else {
                    child.encode_private(&key, level, payload, mask, plaintext)
                }
            };
            graph = match value {
                InputBinding::Scalar(value) => graph
                    .transferred_output(
                        refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                        encode(key, value.clone()).map_err(ApiError::scheme)?,
                    )
                    .map_err(ApiError::scheme)?,
                InputBinding::Family(terms) => {
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                production.clone(),
                                refresh_prf_input_key_artifact_name(commitment, slot, *wire, port),
                                IntExpr::constant(terms.len()),
                                (1, columns),
                                ArtifactAvailability::Cached,
                            )
                        })
                        .collect::<Vec<_>>();
                    let plaintext_ring = child.parameters.ring(0).map_err(ApiError::scheme)?;
                    let family = mxx_dsl::parallel(IntExpr::constant(terms.len()), |index| {
                        let plaintext = plaintext_member(&plaintext_ring, terms, index.clone())?;
                        let member_key = RnsBggPublicKey::new(
                            *wire,
                            (
                                (key_parts[0].at(index.clone()), key_parts[1].at(index.clone())),
                                (key_parts[2].at(index.clone()), key_parts[3].at(index.clone())),
                            ),
                        );
                        encode(member_key, plaintext).map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    graph
                        .transferred_output(
                            refresh_prf_artifact_name(commitment, slot, &format!("input-{wire:?}")),
                            family,
                        )
                        .map_err(ApiError::scheme)?
                }
            };
        }
        Ok(graph)
    }

    /// Samples a payload secret.
    pub fn sample_payload_secret<E, S>(
        &self,
        execution: &mut E,
        store: &mut S,
    ) -> Result<RnsBggPayloadSecret, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let (row, binding) = sample_secret_row(self.parameters, execution, store)?;
        Ok(RnsBggPayloadSecret { row, binding: Some(binding) })
    }
    /// Samples a mask secret on `execution`; it stays resident there.
    pub fn sample_mask_secret<E, S>(
        &self,
        execution: &mut E,
        store: &mut S,
    ) -> Result<RnsBggMaskSecret, ApiError>
    where
        E: StageExecution<S>,
        S: SessionStore,
    {
        let (row, binding) = sample_secret_row(self.parameters, execution, store)?;
        Ok(RnsBggMaskSecret { row, binding: Some(binding) })
    }

    /// Binds the immutable refresh declaration used to publish decoder and
    /// per-label helper artifacts during encryption.
    pub fn bind_refresh_declaration(
        &mut self,
        declaration: &RefreshDeclaration,
    ) -> Result<(), RnsBggError> {
        if declaration.output_schema().is_empty() || declaration.prf_labels().is_empty() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        self.refresh_declaration = Some(declaration.clone());
        Ok(())
    }
    fn encode_private(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        payload: &RnsBggPayloadSecret,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_private_inner(self.parameters, key, payload, mask, plaintext, level)
    }
    fn encode_public(
        &self,
        key: &RnsBggPublicKey,
        level: usize,
        mask: &RnsBggMaskSecret,
        plaintext: Mat,
    ) -> Result<RnsBggEncoding, RnsBggError> {
        encode_public_inner(self.parameters, key, mask, plaintext, level)
    }
    /// The LUT helpers, one family per level and LUT input layout, whose
    /// member `i` belongs to the layout's `i`-th automorphism: its exports
    /// stream from a slot per lane rather than a slot per helper.
    fn lut_helper_families(
        &self,
        mask: &RnsBggMaskSecret,
        hash_key: &Bytes,
        inputs: &crate::program::EncryptionInputs,
    ) -> Result<Vec<(String, Family<RnsBggRow>)>, RnsBggError> {
        let mut families = Vec::new();
        for level in 0..self.parameters.levels().len().saturating_sub(1) {
            let ring_dimension = self.parameters.level(level)?.ring_dimension() as usize;
            let layouts = inputs
                .luts()
                .iter()
                .map(|table| (format!("{:?}", table.input_layout()), table.input_layout()))
                .collect::<BTreeMap<_, _>>();
            for (name, layout) in layouts {
                let sigmas =
                    layout.automorphism_indices(ring_dimension).map_err(|_| RnsBggError::Lut)?;
                let family = mxx_dsl::parallel(sigmas.len(), |index| {
                    let sigma = IntExpr::Select {
                        selector: Box::new(index.expression()?),
                        branches: sigmas.iter().map(|sigma| IntExpr::constant(*sigma)).collect(),
                    };
                    let public_data = super::public_key::lut_public_matrix(
                        self.parameters,
                        hash_key.clone(),
                        level,
                        Int::from(sigma.clone()),
                    )
                    .map_err(|_| DslError::Schema)?;
                    self.encrypt_lut_helper(mask, public_data, level, sigma)
                        .map_err(|_| DslError::Schema)
                })
                .map_err(graph)?;
                families.push((format!("lut-helper/{level}/{name}"), family));
            }
        }
        Ok(families)
    }

    /// Encrypt one preprocess-provided public mask-alignment matrix with the
    /// bound mask.  The `(level, sigma)` key is fixed by the Program and is
    /// not inferred from a LUT width or a family scope.
    fn encrypt_lut_helper(
        &self,
        mask: &RnsBggMaskSecret,
        public_data: RnsBggMatrix,
        level: usize,
        sigma: IntExpr,
    ) -> Result<RnsBggRow, RnsBggError> {
        let mask = reduce_secret(self.parameters, &mask.row, level)?;
        let scale = self.parameters.scalar(level, self.parameters.message_scale(level)?)?;
        let gadget = self.parameters.state_gadget(level)?;
        let columns = self.parameters.component_columns(level)?;
        let correction = (
            scale.clone() * mask.0.clone().ring_automorphism(sigma.clone()) * gadget.clone(),
            scale * mask.1.clone().ring_automorphism(sigma) * gadget,
        );
        let error = fresh_error(self.parameters, level, columns)?;
        Ok((
            mask.0.clone() * public_data.0.0 + mask.1.clone() * public_data.1.0 - correction.0 +
                error.0,
            mask.0 * public_data.0.1 + mask.1 * public_data.1.1 - correction.1 + error.1,
        ))
    }

    fn refresh_decoder(
        &self,
        declaration: &RefreshDeclaration,
        mask: &RnsBggMaskSecret,
        slot: usize,
    ) -> Result<RnsBggRow, RnsBggError> {
        let p_level = refresh_slot_level(self.parameters, declaration, slot)?;
        let metadata = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let p_ring = self.parameters.ring_for_modulus(&metadata.p)?;
        let columns = self.parameters.component_columns(declaration.target_level())?;
        let commitment = *declaration.commitment();
        let production =
            self.preprocess.clone().ok_or(RnsBggError::Binding("missing preprocess"))?;
        let matrix = |role: &str| -> Result<RnsBggMatrix, RnsBggError> {
            let availability = match role {
                "a-sum" => ArtifactAvailability::Transferred,
                "a-star" => ArtifactAvailability::Cached,
                _ => return Err(RnsBggError::Binding("unknown refresh decoder matrix")),
            };
            Ok((
                (
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.0", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                        availability,
                    ),
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.1", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                        availability,
                    ),
                ),
                (
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.2", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                        availability,
                    ),
                    p_ring.artifact_input(
                        production.clone(),
                        format!("{}.3", refresh_artifact_name(commitment, slot, role)),
                        (1, columns),
                        availability,
                    ),
                ),
            ))
        };
        let sum = matrix("a-sum")?;
        let anchor = matrix("a-star")?;
        let h = p_ring.polynomial([IntExpr::constant(&metadata.p / &metadata.q_t)]);
        let delta = (
            (sum.0.0 - h.clone() * anchor.0.0, sum.0.1 - h.clone() * anchor.0.1),
            (sum.1.0 - h.clone() * anchor.1.0, sum.1.1 - h * anchor.1.1),
        );
        let secret = checked_modulus_convert_row(mask.row.clone(), &p_ring)?;
        let affine_offset = refresh_affine_offset_row(self.parameters, declaration, slot, p_level)?;
        let decoder = (
            secret.0.clone() * delta.0.0 + secret.1.clone() * delta.1.0 - affine_offset.0,
            secret.0 * delta.0.1 + secret.1 * delta.1.1 - affine_offset.1,
        );
        let error = encoding_error_row(self.parameters, &p_ring, columns)?;
        Ok((decoder.0 + error.0, decoder.1 + error.1))
    }
}

fn refresh_slot_level(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<usize, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let modulus = metadata
        .p
        .to_biguint()
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    parameters
        .levels()
        .iter()
        .position(|level| level.modulus().as_ref() == &modulus)
        .ok_or(RnsBggError::Binding("refresh slot modulus is not configured"))
}

fn refresh_slot_q_level(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<usize, RnsBggError> {
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let modulus = metadata
        .q
        .to_biguint()
        .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
    parameters
        .levels()
        .iter()
        .position(|level| level.modulus().as_ref() == &modulus)
        .ok_or(RnsBggError::Binding("refresh slot q modulus is not configured"))
}

fn refresh_digit_sum(base: usize, digits: usize) -> BigInt {
    let base = BigInt::from(base);
    let mut power = BigInt::from(1u8);
    let mut sum = BigInt::from(0u8);
    for _ in 0..digits {
        sum += power.clone();
        power *= &base;
    }
    sum
}

fn refresh_affine_coefficients(
    declaration: &RefreshDeclaration,
    slot: usize,
) -> Result<(BigInt, BigInt), RnsBggError> {
    let digits = declaration.digits();
    let mask_digits =
        *digits.mask_counts.get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let base = BigInt::from(digits.base);
    let mask_support = base.pow(mask_digits as u32);
    let mask_offset = -refresh_digit_sum(digits.base, mask_digits) - &mask_support / 2u8;
    let fresh_sum = refresh_digit_sum(digits.base, digits.fresh_count);
    let metadata =
        declaration.slots().get(slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
    let h = metadata.p.clone() / metadata.q_t.clone();
    let fresh_offset = h * (-(base / 2u8) * fresh_sum);
    Ok((mask_offset, fresh_offset))
}

fn refresh_affine_route(
    ring: &mxx_dsl::Ring,
    columns: usize,
    component: usize,
    coefficient: usize,
    scalar: BigInt,
) -> Result<RnsBggRow, RnsBggError> {
    let component_limit = columns
        .checked_mul(2)
        .ok_or(RnsBggError::Parameters("refresh component index overflow"))?;
    if component >= component_limit {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let block = component / columns;
    let column = component % columns;
    let rotation = ring
        .constant((1, 1), ConstantMatrix::Rotation { exponent: IntExpr::constant(coefficient) });
    let unit =
        ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(column) });
    let row = ring.polynomial([IntExpr::constant(scalar)]) * rotation * unit;
    let first = ring.polynomial([IntExpr::constant(1usize - block)]);
    let second = ring.polynomial([IntExpr::constant(block)]);
    let zero = ring.zero((1, columns));
    Ok((first.clone() * row.clone() + second.clone() * zero.clone(), second * row + first * zero))
}

fn refresh_affine_offset_row(
    parameters: &RnsBggParameters,
    declaration: &RefreshDeclaration,
    slot: usize,
    level: usize,
) -> Result<RnsBggRow, RnsBggError> {
    let ring = parameters.ring(level)?;
    let columns = parameters.component_columns(declaration.target_level())?;
    let (mask_offset, fresh_offset) = refresh_affine_coefficients(declaration, slot)?;
    let mut coordinates = BTreeSet::new();
    for index in 0..declaration.labels().len() {
        let label = declaration
            .labels()
            .label(index)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
        let (component, coefficient) = match label {
            RefreshPrfLabel::Mask { component, coefficient, .. } |
            RefreshPrfLabel::FreshError { component, coefficient, .. } => (component, coefficient),
        };
        coordinates.insert((component, coefficient));
    }
    if coordinates.is_empty() {
        return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
    }
    let zero = ring.zero((1, columns));
    let mut offset = (zero.clone(), zero);
    for (component, coefficient) in coordinates {
        let mask =
            refresh_affine_route(&ring, columns, component, coefficient, mask_offset.clone())?;
        let fresh =
            refresh_affine_route(&ring, columns, component, coefficient, fresh_offset.clone())?;
        offset = add_encoding_rows(offset, add_encoding_rows(mask, fresh));
    }
    Ok(offset)
}

impl<'a, P: Poly> EncryptBuild<P> for RnsEncryptBuild<'a> {
    type PayloadSecret = RnsBggPayloadSecret;
    type MaskSecret = RnsBggMaskSecret;
    type PreprocessDescriptor = RnsBggPreprocessEncryptDescriptor;
    type Descriptor = RnsBggEncryptionDescriptor;

    fn bind_inputs(
        &mut self,
        inputs: &crate::program::EncryptionInputs,
        parameters: &mxx_ir_core::ParamEnv,
        lhs: &PlaintextInputs<WireId, P>,
        payload: &Self::PayloadSecret,
        mask: &Self::MaskSecret,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        if lhs.len() != inputs.inputs().len() ||
            lhs.keys().any(|wire| !inputs.inputs().contains_key(wire))
        {
            return Err(ApiError::input("encryption inputs do not exactly match Program inputs"));
        }
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        for (wire, declaration) in inputs.inputs() {
            let value = lhs.get(wire).ok_or_else(|| ApiError::input("missing encryption input"))?;
            let converted = match value {
                InputBinding::Scalar(value)
                    if matches!(declaration.shape(), InputBinding::Scalar(())) =>
                {
                    InputBinding::Scalar(plaintext_constant(&ring, value))
                }
                InputBinding::Family(values)
                    if matches!(declaration.shape(), InputBinding::Family(_)) =>
                {
                    let expected = match declaration.shape() {
                        InputBinding::Family(count) => count
                            .evaluate(parameters)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or_else(|| {
                                ApiError::input("encryption input family count is not concrete")
                            })?,
                        InputBinding::Scalar(()) => unreachable!(),
                    };
                    if values.len() != expected {
                        return Err(ApiError::input(
                            "encryption input family count does not match Program",
                        ));
                    }
                    InputBinding::Family(plaintext_terms(values).map_err(ApiError::input)?)
                }
                _ => return Err(ApiError::input("encryption input shape does not match Program")),
            };
            self.plaintexts.insert(*wire, converted);
        }
        self.inputs = Some(inputs.clone());
        self.projection_identity = Some(
            encryption_projection_identity(inputs, parameters, self.parameters)
                .map_err(ApiError::scheme)?,
        );
        self.payload_secret = Some(payload.clone());
        self.mask_secret = Some(mask.clone());
        for binding in [&payload.binding, &mask.binding].into_iter().flatten() {
            binding.bind(&mut self.runtime);
        }
        self.hash_key_commitment = hash_key_commitment(hash_key);
        self.hash_key = hash_key.to_vec();
        Ok(())
    }

    fn bind_artifacts(
        &mut self,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
    ) -> Result<(), ApiError> {
        self.bind_preprocess_artifacts_inner(preprocess)
    }

    fn artifact_identity(&self) -> [u8; 32] {
        let mut digest = sha2::Sha256::new();
        digest.update(b"mxx-rns-bgg/encryption/v1");
        digest.update(self.context);
        digest.update(self.parameters.parameter_identity());
        digest.update(self.hash_key_commitment);
        if let Some(projection) = self.projection_identity {
            digest.update(projection);
        }
        if let Some(production) = &self.preprocess {
            digest.update(production.spec_hash.0);
            digest.update(production.execution_nonce);
        }
        if let Some(inputs) = &self.inputs {
            for wire in inputs.inputs().keys() {
                digest.update(format!("public-key/{wire:?}").as_bytes());
            }
        }
        if let Some(binding) = &self.binding {
            for source in binding.sources() {
                digest.update(source.spec_hash.0);
                digest.update(source.execution_nonce);
            }
        }
        digest.finalize().into()
    }

    fn build_graph(mut self) -> Result<(StageGraph, Self::Descriptor), ApiError> {
        let inputs = self
            .inputs
            .clone()
            .ok_or_else(|| ApiError::input("encryption inputs are not bound"))?;
        if self.preprocess.is_none() {
            return Err(ApiError::input("preprocess artifacts are not bound"));
        }
        let payload = self
            .payload_secret
            .clone()
            .ok_or_else(|| ApiError::input("payload secret is not bound"))?;
        let mask =
            self.mask_secret.clone().ok_or_else(|| ApiError::input("mask secret is not bound"))?;
        let ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        let hash_key = ring.bytes_input(super::HASH_KEY_INPUT, self.hash_key.len());
        let mut graph = DslContext::new("rns-bgg-encrypt");
        if let Some(declaration) = self.refresh_declaration.as_ref() {
            for (slot, child) in &self.refresh_prf_children {
                graph =
                    Self::append_refresh_prf_child(child, declaration, *slot, graph, &hash_key)?;
                merge_runtime_inputs(&mut self.runtime, &child.runtime)?;
                self.manifests.extend(child.manifests.clone());
            }
        }
        for (name, family) in
            self.lut_helper_families(&mask, &hash_key, &inputs).map_err(ApiError::scheme)?
        {
            graph = graph.transferred_output(name, family).map_err(ApiError::scheme)?;
        }
        // Each input is encoded at its logical level under a public key
        // sampled from the hash key; only the row and, for a public input, the
        // payload are stored, and every later stage samples the key again.
        let binding =
            self.binding.as_ref().ok_or_else(|| ApiError::input("preprocess is not bound"))?;
        for (wire, declaration) in inputs.inputs() {
            let level = binding.input_levels().get(wire).copied().unwrap_or(0);
            let input_ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            // Plaintexts are bound in the natural input ring, level zero.
            let plaintext_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
            let knowledge = declaration.knowledge();
            let public = matches!(knowledge, AttributeKnowledge::PublicPayload);
            let value = self
                .plaintexts
                .get(wire)
                .ok_or_else(|| ApiError::input("missing encryption plaintext"))?;
            let encode = |member: Option<IntExpr>, plaintext: Mat| {
                let key = super::public_key::sampled_key(
                    self.parameters,
                    level,
                    hash_key.clone(),
                    *wire,
                    knowledge,
                    member,
                )?;
                let plaintext = checked_modulus_convert(plaintext, &input_ring)?;
                if public {
                    self.encode_public(&key, level, &mask, plaintext)
                } else {
                    self.encode_private(&key, level, &payload, &mask, plaintext)
                }
            };
            let name = format!("input-{wire:?}");
            graph = match (value, public) {
                (InputBinding::Scalar(value), true) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    let payload = encoding
                        .public_payload
                        .ok_or_else(|| ApiError::input("public encoding has no payload"))?;
                    graph.transferred_output(name, (encoding.row.0, encoding.row.1, payload))
                }
                (InputBinding::Scalar(value), false) => {
                    let encoding = encode(None, value.clone()).map_err(ApiError::scheme)?;
                    graph.transferred_output(name, encoding.row)
                }
                (InputBinding::Family(terms), true) => {
                    let stored = mxx_dsl::parallel(IntExpr::constant(terms.len()), |index| {
                        let plaintext = plaintext_member(&plaintext_ring, terms, index.clone())?;
                        let encoding = encode(Some(index.expression()?), plaintext)
                            .map_err(|_| DslError::Schema)?;
                        let payload = encoding.public_payload.ok_or(DslError::Schema)?;
                        Ok((encoding.row.0, encoding.row.1, payload))
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(name, stored)
                }
                (InputBinding::Family(terms), false) => {
                    let stored = mxx_dsl::parallel(IntExpr::constant(terms.len()), |index| {
                        let plaintext = plaintext_member(&plaintext_ring, terms, index.clone())?;
                        encode(Some(index.expression()?), plaintext)
                            .map(|encoding| encoding.row)
                            .map_err(|_| DslError::Schema)
                    })
                    .map_err(ApiError::scheme)?;
                    graph.transferred_output(name, stored)
                }
            }
            .map_err(ApiError::scheme)?;
        }
        if let Some(declaration) = self.refresh_declaration.clone() {
            for slot in declaration.slots() {
                let decoder = self
                    .refresh_decoder(&declaration, &mask, slot.slot)
                    .map_err(ApiError::scheme)?;
                graph = graph
                    .transferred_output(
                        refresh_artifact_name(*declaration.commitment(), slot.slot, "decoder"),
                        decoder.clone(),
                    )
                    .map_err(ApiError::scheme)?;
            }
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        if super::graph_declares_input(&graph, super::HASH_KEY_INPUT) {
            self.runtime.insert(
                super::HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.hash_key.as_slice().into()),
            );
        }
        let mut descriptor =
            self.binding.ok_or_else(|| ApiError::input("artifact binding is not available"))?;
        if let Some(declaration) = &self.refresh_declaration {
            descriptor = descriptor.with_refresh_declaration(declaration);
        }
        let runtime = declared_runtime_inputs(&graph, self.runtime);
        Ok((
            StageGraph::new(graph, runtime, self.manifests),
            RnsBggEncryptionDescriptor::from_binding(descriptor),
        ))
    }
}

/// Online builder with no backend, secrets, or plaintexts.
pub struct RnsOnlineBuild<'a> {
    parameters: &'a RnsBggParameters,
    context: [u8; 32],
    hash_key_commitment: [u8; 32],
    hash_key: Vec<u8>,
    evaluator: Option<EncodingLowering<'a>>,
    families: OnlineFamilies<'a>,
    environment: Environment<RnsBggEncoding, RnsBggMatrix, OnlineWireFamily, OnlineRhsFamily>,
    pending_refresh: Option<(usize, RefreshDeclaration, BTreeMap<usize, RnsOnlineBuild<'a>>)>,
    refreshed_input: Option<(RefreshDeclaration, RnsBggEncoding)>,
    refresh_prf: Option<(usize, crate::program::ExponentLutProgramId, [u8; 32])>,
    refreshed_public_key: Option<RnsBggPublicKey>,
}

impl<'a> RnsOnlineBuild<'a> {
    /// Creates an unbound online builder.
    pub fn new(parameters: &'a RnsBggParameters, context: [u8; 32]) -> Self {
        Self {
            parameters,
            context,
            hash_key_commitment: [0; 32],
            hash_key: Vec::new(),
            evaluator: None,
            families: OnlineFamilies { parameters },
            environment: Environment::new(),
            pending_refresh: None,
            refreshed_input: None,
            refresh_prf: None,
            refreshed_public_key: None,
        }
    }

    /// Bind a slot-scoped online child to the prepared PRF Program.
    ///
    /// Ordinary artifact binding retains strict source-Program equality. This
    /// explicit child path validates the combined source checkpoints and then
    /// imports only the slot-scoped PRF input ports.
    pub fn bind_refresh_artifacts(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
        preprocess: &StageArtifacts<RnsBggPreprocessOnlineDescriptor>,
        encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        let prepared = declaration.prepared().program();
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_hash_key = hash_key_commitment(hash_key);
        if slot >= declaration.slots().len() || declaration.output_schema().is_empty() {
            return Err(ApiError::input("invalid refresh PRF slot or schema"));
        }
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.program() != declaration.source_program() ||
                binding.hash_key_commitment() != &expected_hash_key ||
                binding.refresh_commitment() != Some(declaration.commitment())
            {
                return Err(ApiError::compatibility(
                    "refresh PRF artifacts do not match source declaration",
                ));
            }
        }
        if !preprocess_binding.sources().is_empty() {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh preprocess root unexpectedly declares sources",
            ));
        }
        if encryption_binding.sources().len() != 1 ||
            !encryption_binding.sources().contains(preprocess.production())
        {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh encryption source is not preprocess production",
            ));
        }
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption.production().clone();
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(prepared).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        // The prepared PRF encryption stage publishes each input at its
        // logical level (`append_refresh_prf_child`), with its public key; the
        // refresh route applies the per-slot p-level action to the outputs.
        let commitment = *declaration.commitment();
        self.environment = Environment::new();
        for (wire, input) in prepared.inputs() {
            let public = matches!(input.knowledge(), AttributeKnowledge::PublicPayload);
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            let columns = self
                .parameters
                .key_component_columns(level, input.knowledge())
                .map_err(ApiError::scheme)?;
            let scalar = || {
                RnsBggEncoding::new(
                    RnsBggPublicKey::new(
                        *wire,
                        (
                            (
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.0"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.1"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                            ),
                            (
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.2"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                                ring.artifact_input(
                                    encryption_production.clone(),
                                    refresh_prf_artifact_name(
                                        commitment,
                                        slot,
                                        &format!("input-{wire:?}.3"),
                                    ),
                                    (1, columns),
                                    ArtifactAvailability::Transferred,
                                ),
                            ),
                        ),
                    ),
                    (
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.4"),
                            ),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.5"),
                            ),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        ),
                    ),
                    public.then(|| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.6"),
                            ),
                            (1, 1),
                            ArtifactAvailability::Transferred,
                        )
                    }),
                )
            };
            let binding = match input.shape() {
                InputBinding::Scalar(()) => InputBinding::Scalar(scalar()),
                InputBinding::Family(count) => {
                    let key_parts = (0..4)
                        .map(|port| {
                            ring.family_artifact_input(
                                encryption_production.clone(),
                                refresh_prf_artifact_name(
                                    commitment,
                                    slot,
                                    &format!("input-{wire:?}.{port}"),
                                ),
                                count.clone(),
                                (1, columns),
                                ArtifactAvailability::Transferred,
                            )
                        })
                        .collect::<Vec<_>>();
                    let rows = (0..2)
                        .map(|port| {
                            ring.family_artifact_input(
                                encryption_production.clone(),
                                refresh_prf_artifact_name(
                                    commitment,
                                    slot,
                                    &format!("input-{wire:?}.{}", port + 4),
                                ),
                                count.clone(),
                                (1, columns),
                                ArtifactAvailability::Transferred,
                            )
                        })
                        .collect::<Vec<_>>();
                    let payload = public.then(|| {
                        ring.family_artifact_input(
                            encryption_production.clone(),
                            refresh_prf_artifact_name(
                                commitment,
                                slot,
                                &format!("input-{wire:?}.6"),
                            ),
                            count.clone(),
                            (1, 1),
                            ArtifactAvailability::Transferred,
                        )
                    });
                    InputBinding::Family(OnlineWireFamily::Stored(encoding_family(
                        *wire,
                        &key_parts,
                        &rows,
                        payload.as_ref(),
                        count.clone(),
                    )?))
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let commitment = *declaration.commitment();
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            prepared,
            &workload,
            &preprocess_production,
            &hash_key,
            |role: String| refresh_prf_artifact_name(commitment, slot, &role),
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(prepared.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            refresh_prf_namespace: Some((*declaration.commitment(), slot)),
            current_site: None,
            hash_key,
        });
        Ok(())
    }

    /// Bind one child to the structural PRF program frozen in `declaration`.
    ///
    /// The PRF is a separate Program from the outer computation.  Checking
    /// that identity here prevents a child lowerer for an unrelated source
    /// graph from being accepted merely because its carrier has the right
    /// row shape.
    pub fn bind_refresh_prf(
        &mut self,
        declaration: &RefreshDeclaration,
        slot: usize,
    ) -> Result<(), RnsBggError> {
        self.parameters.validate_refresh_prf_schedule(declaration, slot)?;
        if slot >= declaration.slots().len() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let evaluator =
            self.evaluator.as_ref().ok_or(RnsBggError::Binding("PRF artifacts are not bound"))?;
        if declaration.prepared().program().id() != evaluator.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        let declared_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(_, wire)| wire)
            .collect::<Vec<_>>();
        let program_outputs = evaluator
            .program
            .outputs()
            .iter()
            .filter_map(|output| match output {
                Operand::Scalar(wire) => Some(*wire),
                Operand::Member { .. } => None,
            })
            .collect::<Vec<_>>();
        if declared_outputs.iter().any(|wire| !program_outputs.contains(wire)) {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        if declared_outputs.iter().any(|wire| {
            evaluator
                .program
                .wire_is_coefficient_output(*wire)
                .map(|coefficient| !coefficient)
                .unwrap_or(true)
        }) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidSourceOutput));
        }
        let slot_modulus = declaration
            .slots()
            .get(slot)
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?
            .q
            .to_biguint()
            .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        if self
            .parameters
            .level(0)
            .map_err(|_| RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?
            .modulus()
            .as_ref() !=
            &slot_modulus
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        if declaration.prepared().setup_identity().program != evaluator.program.id() {
            return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
        }
        self.refresh_prf =
            Some((slot, evaluator.program.id(), self.parameters.parameter_identity()));
        Ok(())
    }

    /// Bind a checkpointed refresh as the sole input of an identity
    /// continuation. Only lazy artifact handles are created. The checkpoint's
    /// row and matching target public-key ports are imported from its
    /// production; caller code cannot substitute an unrelated carrier.
    fn bind_refreshed_input(
        &mut self,
        program: &ExponentLutProgram,
        declaration: &RefreshDeclaration,
        artifacts: &StageArtifacts<RnsBggRefreshCheckpointDescriptor>,
    ) -> Result<(), ApiError> {
        let (&input_wire, declaration_input) = program
            .inputs()
            .iter()
            .next()
            .ok_or_else(|| ApiError::input("resumption Program has no input"))?;
        let descriptor = artifacts.descriptor().binding();
        if descriptor.context() != &self.context ||
            descriptor.program() != declaration.source_program() ||
            descriptor.refresh_commitment() != Some(declaration.commitment())
        {
            return Err(ApiError::compatibility(
                "refreshed input checkpoint does not match its declaration",
            ));
        }
        if declaration.output_schema().is_empty() {
            return Err(ApiError::input("refreshed input schema is empty"));
        }
        // A resumed Program is evaluated against the target chain, whose
        // level zero is the refresh output level.  The old source Program's
        // parameter digest intentionally does not identify this builder.
        let resumed_level = self.parameters.level(0).map_err(ApiError::scheme)?;
        let resumed_ring = self.parameters.ring(0).map_err(ApiError::scheme)?;
        if program.inputs().len() != 1 ||
            !program.inputs().contains_key(&input_wire) ||
            program.outputs() != [Operand::Scalar(input_wire)]
        {
            return Err(ApiError::compatibility(
                "only an identity continuation can resume from a refresh checkpoint without stage artifacts",
            ));
        }
        if !matches!(declaration_input.shape(), InputBinding::Scalar(())) {
            return Err(ApiError::input("refreshed input must be scalar"));
        }
        let knowledge = program.wire_knowledge(input_wire).map_err(ApiError::scheme)?;
        let key_columns =
            self.parameters.key_component_columns(0, knowledge).map_err(ApiError::scheme)?;
        let row_columns = key_columns;
        let expected_port = resumed_ring.matrix_type((1, key_columns));
        let schema_ports = declaration
            .output_schema()
            .iter()
            .filter_map(|wire| match wire {
                WireType::Matrix(matrix) => Some(matrix),
                _ => None,
            })
            .collect::<Vec<_>>();
        let target_modulus =
            declaration.slots().iter().try_fold(BigUint::from(1u8), |product, slot| {
                slot.q_t
                    .to_biguint()
                    .map(|modulus| product * modulus)
                    .ok_or(ApiError::compatibility("refresh target modulus is invalid"))
            })?;
        if schema_ports.len() != 4 ||
            schema_ports.iter().any(|port| **port != expected_port) ||
            target_modulus != *resumed_level.modulus().as_ref()
        {
            return Err(ApiError::compatibility(
                "resumption builder must use the target modulus chain at level zero",
            ));
        }
        let target_ring = resumed_ring;
        let production = artifacts.production().clone();
        let manifest = artifacts.manifest();
        let checkpoint_name =
            |role: &str| refresh_checkpoint_artifact_name(*declaration.commitment(), role);
        let resumed_concrete_ring = resolve_ring(target_ring.as_ref()).map_err(ApiError::scheme)?;
        let matrix_type = |columns| {
            ArtifactType::Matrix(ConcreteMatrixType {
                ring: resumed_concrete_ring.clone(),
                rows: 1,
                columns,
            })
        };
        let require_artifact = |name: &str, artifact_type: ArtifactType| {
            let artifact = manifest.artifacts.get(name).ok_or_else(|| {
                ApiError::compatibility("refresh checkpoint schema is missing a required port")
            })?;
            if artifact.artifact_type != artifact_type || artifact.family_count.is_some() {
                return Err(ApiError::compatibility(
                    "refresh checkpoint artifact schema does not match the declaration",
                ));
            }
            Ok(())
        };
        require_artifact(&checkpoint_name("row.0"), matrix_type(row_columns))?;
        require_artifact(&checkpoint_name("row.1"), matrix_type(row_columns))?;
        for port in 0..4 {
            require_artifact(
                &checkpoint_name(&format!("target.{port}")),
                matrix_type(key_columns),
            )?;
        }
        let row = (
            target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.0"),
                (1, row_columns),
                ArtifactAvailability::Transferred,
            ),
            target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.1"),
                (1, row_columns),
                ArtifactAvailability::Transferred,
            ),
        );
        let public_key = RnsBggPublicKey::new(
            input_wire,
            (
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.0"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.1"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                ),
                (
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.2"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                    target_ring.artifact_input(
                        production.clone(),
                        checkpoint_name("target.3"),
                        (1, key_columns),
                        ArtifactAvailability::Transferred,
                    ),
                ),
            ),
        );
        if public_key.schema().wire_types() != declaration.output_schema() {
            return Err(ApiError::input("refresh public-key schema does not match declaration"));
        }
        let public_payload = if matches!(knowledge, AttributeKnowledge::PublicPayload) {
            require_artifact(&checkpoint_name("row.public"), matrix_type(1))?;
            Some(target_ring.artifact_input(
                production.clone(),
                checkpoint_name("row.public"),
                (1, 1),
                ArtifactAvailability::Transferred,
            ))
        } else {
            None
        };
        // Initialize the lowerer here rather than requiring an ordinary
        // bind_artifacts call.  The checkpoint's production is the only
        // runtime input needed by the identity continuation; source
        // dependency productions remain available for a later gated Program.
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?);
        let mut environment = Environment::new();
        environment
            .bind_wire(
                input_wire,
                InputBinding::Scalar(RnsBggEncoding::new(
                    public_key.clone(),
                    row.clone(),
                    public_payload.clone(),
                )),
            )
            .map_err(|error| ApiError::input(error.to_string()))?;
        let mut sources = descriptor.sources().to_vec();
        sources.sort();
        sources.dedup();
        let preprocess = sources.first().cloned().unwrap_or_else(|| production.clone());
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess,
            &hash_key,
            |role| role,
        )? {
            environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        let encryption = sources.get(1).cloned().unwrap_or_else(|| preprocess.clone());
        let manifests = BTreeMap::from([(production.clone(), artifacts.manifest().clone())]);
        if !manifests.contains_key(&production) {
            return Err(ApiError::compatibility(
                "refresh checkpoint production is not in its manifest map",
            ));
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess,
            encryption,
            manifests,
            refresh_prf_namespace: None,
            current_site: None,
            hash_key,
        });
        self.environment = environment;
        self.refreshed_public_key = Some(public_key.clone());
        self.refreshed_input =
            Some((declaration.clone(), RnsBggEncoding::new(public_key, row, public_payload)));
        Ok(())
    }

    /// Returns the lazily imported target public key for a resumed refresh.
    pub fn refreshed_public_key(&self) -> Option<&RnsBggPublicKey> {
        self.refreshed_public_key.as_ref()
    }

    /// Returns the lazily imported refreshed encoding for resumption.
    pub fn refreshed_encoding(&self) -> Option<&RnsBggEncoding> {
        self.refreshed_input.as_ref().map(|(_, encoding)| encoding)
    }

    /// Bind one declared refresh and replace the selected output at graph
    /// finalization.  The prepared PRF children are lowered internally when
    /// the graph is finalized; callers provide no carrier map or evaluator
    /// closure.
    pub fn apply_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        prf_children: BTreeMap<usize, RnsOnlineBuild<'a>>,
    ) -> Result<(), RnsBggError> {
        if self.pending_refresh.is_some() {
            return Err(RnsBggError::Binding("refresh was already bound"));
        }
        if prf_children.len() != declaration.slots().len() ||
            prf_children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        for (slot, child) in &prf_children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.refresh_prf.map(|(bound_slot, program, identity)| {
                bound_slot == *slot &&
                    program == declaration.prepared().program().id() &&
                    identity == expected_child_parameters
            }) != Some(true) ||
                child.parameters.parameter_identity() != expected_child_parameters
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
        }
        self.pending_refresh = Some((output_index, declaration.clone(), prf_children));
        Ok(())
    }

    fn lower_refresh_children(
        declaration: &RefreshDeclaration,
        children: &mut BTreeMap<usize, RnsOnlineBuild<'a>>,
    ) -> Result<BTreeMap<PrfLabel, BTreeMap<usize, RnsBggEncoding>>, RnsBggError> {
        let labels = declaration.prf_labels();
        let ordered_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        let mut outputs = BTreeMap::<PrfLabel, BTreeMap<usize, RnsBggEncoding>>::new();
        for (slot, child) in children.iter_mut() {
            let (evaluator, families, environment) = child
                .lowering_state_inner()
                .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            let values = crate::program::lower_program(
                declaration.prepared().program(),
                evaluator,
                families,
                environment,
            )
            .map_err(|error| RnsBggError::Graph(error.to_string()))?;
            if values.len() != ordered_outputs.len() {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            for (index, ((label, wire), value)) in ordered_outputs.iter().zip(values).enumerate() {
                let refresh_label = declaration
                    .labels()
                    .label(index)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                if !refresh_label
                    .slots(declaration.slots().len())
                    .map_err(RnsBggError::Refresh)?
                    .contains(slot)
                {
                    continue;
                }
                if value.wire() != *wire {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                outputs.entry(label.clone()).or_default().insert(*slot, value);
            }
        }
        for label in labels {
            let channels =
                outputs.get(label).ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let index = declaration
                .prf_labels()
                .iter()
                .position(|expected| expected == label)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let refresh_label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let expected_slots = match refresh_label {
                RefreshPrfLabel::Mask { slot, .. } => vec![slot],
                RefreshPrfLabel::FreshError { .. } => (0..children.len()).collect(),
            };
            if channels.len() != expected_slots.len() ||
                channels.keys().copied().ne(expected_slots.iter().copied())
            {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
        }
        Ok(outputs)
    }

    /// Finalize the pending refresh against lowered outer outputs and return
    /// the resulting symbolic carrier before graph finalization.
    fn finalize_refresh(
        &mut self,
        output_index: usize,
        declaration: &RefreshDeclaration,
        mut children: BTreeMap<usize, RnsOnlineBuild<'a>>,
        outputs: &mut [RnsBggEncoding],
    ) -> Result<RnsBggEncoding, RnsBggError> {
        if children.len() != declaration.slots().len() ||
            children.keys().copied().ne(0..declaration.slots().len())
        {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        for (slot, child) in &children {
            let expected_child_parameters =
                self.parameters.expected_refresh_prf_parameter_identity(declaration, *slot)?;
            if child.refresh_prf.map(|(bound_slot, program, identity)| {
                bound_slot == *slot &&
                    program == declaration.prepared().program().id() &&
                    identity == expected_child_parameters
            }) != Some(true) ||
                child.parameters.parameter_identity() != expected_child_parameters
            {
                return Err(RnsBggError::Refresh(RefreshError::Commitment));
            }
        }
        let source = outputs
            .get(output_index)
            .ok_or(RnsBggError::Refresh(RefreshError::TargetMismatch))?
            .clone();
        let inputs = self.refresh_inputs_from_output(declaration, &source)?;
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?;
        let mut refresh = RnsBggRefresh::new(evaluator, inputs);
        let refreshed = apply_refresh(declaration, &mut refresh, |callback_declaration, _| {
            Self::lower_refresh_children(callback_declaration, &mut children)
        })
        .map_err(|error| match error {
            RefreshApplicationError::Refresh(error) => error,
            RefreshApplicationError::OutputMismatch => {
                RnsBggError::Refresh(RefreshError::PrfOutputMismatch)
            }
        })?;
        let refreshed_target = evaluator.refresh_public_target(declaration, source.wire())?;
        for child in children.values() {
            if let Some(child_evaluator) = child.evaluator.as_ref() {
                evaluator.manifests.extend(child_evaluator.manifests.clone());
            }
        }
        outputs[output_index] = refreshed.clone();
        self.refreshed_public_key = Some(refreshed_target);
        Ok(refreshed)
    }

    fn refresh_inputs_from_output(
        &self,
        declaration: &RefreshDeclaration,
        source: &RnsBggEncoding,
    ) -> Result<BTreeMap<usize, RnsBggEncoding>, RnsBggError> {
        if source.wire() != declaration.source_output() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let evaluator = self
            .evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?;
        let level = evaluator.workload.level(source.wire())?;
        let ring = self.parameters.ring(level)?;
        let expected_public =
            matches!(self.program_knowledge(source.wire())?, AttributeKnowledge::PublicPayload);
        let key_knowledge = if expected_public {
            AttributeKnowledge::PublicPayload
        } else {
            AttributeKnowledge::PrivatePayload
        };
        let key_columns = self.parameters.key_component_columns(level, key_knowledge)?;
        let row_columns = key_columns;
        let expected_row = ring.matrix_type((1, row_columns));
        let expected_key = ring.matrix_type((1, key_columns));
        if source.row.0.matrix_type() != &expected_row ||
            source.row.1.matrix_type() != &expected_row
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if source.public_key.wire != source.wire() ||
            source.public_key.matrix.0.0.matrix_type() != &expected_key ||
            source.public_key.matrix.0.1.matrix_type() != &expected_key ||
            source.public_key.matrix.1.0.matrix_type() != &expected_key ||
            source.public_key.matrix.1.1.matrix_type() != &expected_key
        {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if expected_public != source.public_payload.is_some() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        if let Some(payload) = &source.public_payload {
            if payload.matrix_type() != &ring.matrix_type((1, 1)) {
                return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
            }
        }
        declaration
            .slots()
            .iter()
            .map(|slot| {
                // Each slot receives the same raw q-level carrier.  The
                // refresh route owns the single q→p ModDown; reducing here
                // would silently remove that protocol stage.
                Ok((
                    slot.slot,
                    RnsBggEncoding::new(
                        source.public_key.clone(),
                        (source.row.0.clone(), source.row.1.clone()),
                        source.public_payload.clone(),
                    ),
                ))
            })
            .collect()
    }

    fn program_knowledge(&self, wire: WireId) -> Result<AttributeKnowledge, RnsBggError> {
        self.evaluator
            .as_ref()
            .ok_or(RnsBggError::Binding("online artifacts are not bound"))?
            .program
            .wire_knowledge(wire)
            .map_err(|_| RnsBggError::Binding("wire metadata"))
    }

    fn lowering_state_inner(
        &mut self,
    ) -> Result<
        (
            &mut EncodingLowering<'a>,
            &mut OnlineFamilies<'a>,
            &mut Environment<RnsBggEncoding, RnsBggMatrix, OnlineWireFamily, OnlineRhsFamily>,
        ),
        ApiError,
    > {
        if let Some((declaration, input)) = self.refreshed_input.clone() {
            if declaration.output_schema().is_empty() {
                return Err(ApiError::input("refreshed input declaration schema is empty"));
            }
            let evaluator = self
                .evaluator
                .as_ref()
                .ok_or_else(|| ApiError::input("online artifacts are not bound"))?;
            let input_declaration = evaluator
                .program
                .inputs()
                .get(&input.wire())
                .ok_or_else(|| ApiError::input("refreshed wire is not a Program input"))?;
            if !matches!(input_declaration.shape(), InputBinding::Scalar(())) {
                return Err(ApiError::input("refreshed input must be scalar"));
            }
            self.environment.wires.insert(input.wire(), InputBinding::Scalar(input));
        }
        Ok((
            self.evaluator
                .as_mut()
                .ok_or_else(|| ApiError::input("online artifacts are not bound"))?,
            &mut self.families,
            &mut self.environment,
        ))
    }
}

/// Checkpoint a refresh output that has already been returned by
/// [`crate::api::online_eval`].
///
/// This operation only imports the returned output ports into a small
/// artifact graph. It does not lower or execute the refresh again. The
/// checkpoint stores the row tuple, an optional public payload, and the
/// matching target public key as separate outputs so that a resumed carrier
/// cannot accidentally treat the whole paired encoding as its row.
pub fn checkpoint_refreshed_output<E, S>(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
    result: &StageOutputs,
    output_index: usize,
    preprocess: &StageArtifacts<RnsBggPreprocessOnlineDescriptor>,
    encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
    parameters: &RnsBggParameters,
    hash_key: &[u8],
    execution: &mut E,
    store: &mut S,
) -> Result<StageArtifacts<RnsBggRefreshCheckpointDescriptor>, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let (graph, descriptor) = build_refresh_checkpoint_graph(
        program,
        declaration,
        result,
        output_index,
        preprocess,
        encryption,
        parameters,
        hash_key,
    )?;
    execute_artifacts(
        graph,
        &ParamEnv::default(),
        descriptor,
        domain_identity(*declaration.commitment(), b"refresh-checkpoint"),
        execution,
        store,
        |_| Ok(()),
    )
}

/// Resume a refresh through the built-in single-input identity continuation.
///
/// The continuation is deliberately outside the original estimated graph:
/// it is created and executed only after the caller has checkpointed the
/// already-returned refresh output. No refresh callback or intermediate
/// carrier is accepted by this API.
pub fn resume_refreshed_output<E, S>(
    declaration: &RefreshDeclaration,
    checkpoint: &StageArtifacts<RnsBggRefreshCheckpointDescriptor>,
    parameters: &RnsBggParameters,
    execution: &mut E,
    store: &mut S,
) -> Result<StageOutputs, ApiError>
where
    E: StageExecution<S>,
    S: SessionStore,
{
    let binding = checkpoint.descriptor().binding();
    let manifest = checkpoint.manifest();
    let public_payload = manifest
        .artifacts
        .contains_key(&refresh_checkpoint_artifact_name(*declaration.commitment(), "row.public"));
    let knowledge = if public_payload {
        AttributeKnowledge::PublicPayload
    } else {
        AttributeKnowledge::PrivatePayload
    };
    let mut builder = ExponentLutProgramBuilder::new();
    let input = builder.input(knowledge).map_err(ApiError::scheme)?;
    let program = builder.finish(vec![Operand::Scalar(input)]).map_err(ApiError::scheme)?;

    let mut online = RnsOnlineBuild::new(parameters, *binding.context());
    online.bind_refreshed_input(&program, declaration, checkpoint)?;
    let outputs = {
        let (evaluator, families, environment) = online.lowering_state_inner()?;
        crate::program::lower_program(&program, evaluator, families, environment)
            .map_err(ApiError::lowering)?
    };
    let graph = online.build_graph(outputs)?;
    execute_graph(graph, &ParamEnv::default(), execution, store, |_| Ok(()))
}

fn build_refresh_checkpoint_graph(
    program: &ExponentLutProgram,
    declaration: &RefreshDeclaration,
    result: &StageOutputs,
    output_index: usize,
    preprocess: &StageArtifacts<RnsBggPreprocessOnlineDescriptor>,
    encryption: &StageArtifacts<RnsBggEncryptionDescriptor>,
    parameters: &RnsBggParameters,
    hash_key: &[u8],
) -> Result<(StageGraph, RnsBggRefreshCheckpointDescriptor), ApiError> {
    let output = program
        .outputs()
        .get(output_index)
        .ok_or_else(|| ApiError::input("refresh checkpoint output index is out of range"))?;
    let source_wire = match output {
        Operand::Scalar(wire) => *wire,
        Operand::Member { .. } => {
            return Err(ApiError::input("refresh checkpoint output must be scalar"));
        }
    };
    if source_wire != declaration.source_output() {
        return Err(ApiError::input("refresh checkpoint output does not match declaration"));
    }
    if declaration.output_schema().len() != 4 {
        return Err(ApiError::input("refresh checkpoint target schema must contain four ports"));
    }
    let source_knowledge = program.wire_knowledge(source_wire).map_err(ApiError::scheme)?;
    let expected_ports =
        if matches!(source_knowledge, AttributeKnowledge::PrivatePayload) { 6 } else { 7 };
    let prefix = format!("output-{output_index}.");
    let mut ports = BTreeMap::new();
    for name in result.outputs.keys().filter(|name| name.starts_with(&prefix)) {
        let port = name
            .strip_prefix(&prefix)
            .and_then(|port| port.parse::<usize>().ok())
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is malformed"))?;
        if port >= expected_ports || ports.insert(port, name.clone()).is_some() {
            return Err(ApiError::input("refresh checkpoint output port is out of range"));
        }
    }
    if ports.len() != expected_ports || ports.keys().copied().ne(0..expected_ports) {
        return Err(ApiError::input("refresh checkpoint output port set is incomplete"));
    }

    let target_level = declaration.target_level();
    let ring = parameters.ring(target_level).map_err(ApiError::scheme)?;
    let columns = parameters
        .key_component_columns(target_level, source_knowledge)
        .map_err(ApiError::scheme)?;
    let expected_port = ring.matrix_type((1, columns));
    let mut runtime_inputs = BTreeMap::new();
    let mut input = |port: usize, name: String| -> Result<Mat, ApiError> {
        let output_name = ports
            .get(&port)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?;
        let value = result
            .outputs
            .get(output_name)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?
            .clone();
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("refresh checkpoint output is not a matrix"));
        }
        runtime_inputs.insert(name.clone(), value);
        Ok(ring.input(name, (1, columns)))
    };
    let key = (
        (
            input(0, "refresh-checkpoint-key-0".to_owned())?,
            input(1, "refresh-checkpoint-key-1".to_owned())?,
        ),
        (
            input(2, "refresh-checkpoint-key-2".to_owned())?,
            input(3, "refresh-checkpoint-key-3".to_owned())?,
        ),
    );
    let row = (
        input(4, "refresh-checkpoint-row-0".to_owned())?,
        input(5, "refresh-checkpoint-row-1".to_owned())?,
    );
    let schema_ports = declaration
        .output_schema()
        .iter()
        .filter_map(|port| match port {
            WireType::Matrix(matrix) => Some(matrix),
            _ => None,
        })
        .collect::<Vec<_>>();
    if schema_ports.len() != 4 || schema_ports.iter().any(|port| **port != expected_port) {
        return Err(ApiError::input("refresh checkpoint target schema mismatch"));
    }
    if key.0.0.matrix_type() != &expected_port ||
        key.0.1.matrix_type() != &expected_port ||
        key.1.0.matrix_type() != &expected_port ||
        key.1.1.matrix_type() != &expected_port ||
        row.0.matrix_type() != &expected_port ||
        row.1.matrix_type() != &expected_port
    {
        return Err(ApiError::input("refresh checkpoint carrier schema mismatch"));
    }
    let public_payload = if matches!(source_knowledge, AttributeKnowledge::PublicPayload) {
        let output_name = ports
            .get(&6)
            .ok_or_else(|| ApiError::input("refresh checkpoint public payload is missing"))?;
        let value = result
            .outputs
            .get(output_name)
            .ok_or_else(|| ApiError::input("refresh checkpoint output port is missing"))?
            .clone();
        if !matches!(value, RuntimeValue::Matrix(_)) {
            return Err(ApiError::input("refresh checkpoint public payload is not a matrix"));
        }
        let name = "refresh-checkpoint-row-public".to_owned();
        runtime_inputs.insert(name.clone(), value);
        Some(ring.input(name, (1, 1)))
    } else {
        None
    };

    let preprocess_binding = preprocess.descriptor().binding();
    let encryption_binding = encryption.descriptor().binding();
    let expected_parameters = parameters.parameter_identity();
    let expected_hash_key = hash_key_commitment(hash_key);
    for binding in [preprocess_binding, encryption_binding] {
        if binding.context() != encryption_binding.context() ||
            binding.parameters() != &expected_parameters ||
            binding.program() != declaration.source_program() ||
            binding.hash_key_commitment() != &expected_hash_key ||
            binding.refresh_commitment() != Some(declaration.commitment())
        {
            return Err(ApiError::compatibility("refresh checkpoint source binding mismatch"));
        }
    }
    if !preprocess_binding.sources().is_empty() ||
        encryption_binding.sources().len() != 1 ||
        !encryption_binding.sources().contains(preprocess.production())
    {
        return Err(ApiError::compatibility("refresh checkpoint source artifacts are invalid"));
    }
    let descriptor = RnsBggArtifactBinding::new(
        *encryption_binding.context(),
        declaration.source_program(),
        expected_parameters,
        expected_hash_key,
        Vec::new(),
    )
    .map_err(ApiError::scheme)?
    .with_refresh_declaration(declaration);
    let checkpoint_name =
        |role: &str| refresh_checkpoint_artifact_name(*declaration.commitment(), role);
    let mut graph = DslContext::new("rns-bgg-refresh-checkpoint")
        .transferred_output(checkpoint_name("row"), row)
        .map_err(ApiError::scheme)?;
    if let Some(payload) = public_payload {
        graph = graph
            .transferred_output(checkpoint_name("row.public"), payload)
            .map_err(ApiError::scheme)?;
    }
    graph = graph.transferred_output(checkpoint_name("target"), key).map_err(ApiError::scheme)?;
    let mut manifests = BTreeMap::new();
    manifests.insert(preprocess.production().clone(), preprocess.manifest().clone());
    manifests.insert(encryption.production().clone(), encryption.manifest().clone());
    for source in [preprocess.production(), encryption.production()] {
        if !manifests.contains_key(source) {
            return Err(ApiError::compatibility("refresh checkpoint source manifest is missing"));
        }
    }
    Ok((
        StageGraph::new(graph.build().map_err(ApiError::scheme)?, runtime_inputs, manifests),
        RnsBggRefreshCheckpointDescriptor::from_binding(descriptor),
    ))
}

impl<'a> OnlineBuild for RnsOnlineBuild<'a> {
    type Evaluator = EncodingLowering<'a>;
    type Families = OnlineFamilies<'a>;
    type PreprocessDescriptor = RnsBggPreprocessOnlineDescriptor;
    type EncryptDescriptor = RnsBggEncryptionDescriptor;

    fn lowering_state(
        &mut self,
    ) -> Result<
        (
            &mut Self::Evaluator,
            &mut Self::Families,
            &mut Environment<RnsBggEncoding, RnsBggMatrix, OnlineWireFamily, OnlineRhsFamily>,
        ),
        ApiError,
    > {
        self.lowering_state_inner()
    }

    fn bind_artifacts(
        &mut self,
        program: &ExponentLutProgram,
        parameters: &mxx_ir_core::ParamEnv,
        preprocess: &StageArtifacts<Self::PreprocessDescriptor>,
        encryption: &StageArtifacts<Self::EncryptDescriptor>,
        hash_key: &[u8],
    ) -> Result<(), ApiError> {
        for declaration in program.inputs().values() {
            if let InputBinding::Family(count) = declaration.shape() {
                let count =
                    count.evaluate(parameters).ok().and_then(|value| value.to_usize()).ok_or_else(
                        || ApiError::input("online input family count is not concrete"),
                    )?;
                if count == 0 {
                    return Err(ApiError::input("online input family count is empty"));
                }
            }
        }
        let preprocess_binding = preprocess.descriptor().binding();
        let encryption_binding = encryption.descriptor().binding();
        let expected_parameters = self.parameters.parameter_identity();
        let preprocess_production = preprocess.production().clone();
        let encryption_production = encryption.production().clone();
        let expected_hash_key = hash_key_commitment(hash_key);
        for binding in [preprocess_binding, encryption_binding] {
            if binding.context() != &self.context ||
                binding.parameters() != &expected_parameters ||
                binding.program() != program.id() ||
                binding.hash_key_commitment() != &expected_hash_key
            {
                return Err(ApiError::compatibility(
                    "RNS-BGG artifact binding does not match online Program",
                ));
            }
        }
        if !preprocess_binding.sources().is_empty() {
            return Err(ApiError::compatibility(
                "RNS-BGG preprocess root unexpectedly declares sources",
            ));
        }
        if encryption_binding.sources().len() != 1 ||
            !encryption_binding.sources().contains(&preprocess_production)
        {
            return Err(ApiError::compatibility(
                "RNS-BGG encryption source is not the preprocess production",
            ));
        }
        if preprocess_binding.refresh_commitment() != encryption_binding.refresh_commitment() {
            return Err(ApiError::compatibility(
                "RNS-BGG refresh declaration commitment differs between stages",
            ));
        }
        let mut manifests = BTreeMap::new();
        manifests.insert(preprocess_production.clone(), preprocess.manifest().clone());
        manifests.insert(encryption_production.clone(), encryption.manifest().clone());
        let workload =
            Arc::new(RnsBggDerivedWorkload::from_program(program).map_err(ApiError::scheme)?);
        self.hash_key_commitment = expected_hash_key;
        self.hash_key = hash_key.to_vec();
        // Encryption stores each input's row (and public payload) at its
        // logical level; its public key is sampled from the hash key here, or
        // where a family member is read.
        let hash_key = self
            .parameters
            .ring(0)
            .map_err(ApiError::scheme)?
            .bytes_input(super::HASH_KEY_INPUT, 32);
        for (wire, declaration) in program.inputs() {
            let level = workload.level(*wire).map_err(ApiError::scheme)?;
            let ring = self.parameters.ring(level).map_err(ApiError::scheme)?;
            let knowledge = declaration.knowledge();
            let columns = self
                .parameters
                .key_component_columns(level, knowledge)
                .map_err(ApiError::scheme)?;
            let public = matches!(knowledge, AttributeKnowledge::PublicPayload);
            let binding = match declaration.shape() {
                InputBinding::Scalar(()) => {
                    let port = |port: usize, columns: usize| {
                        ring.artifact_input(
                            encryption_production.clone(),
                            format!("input-{wire:?}.{port}"),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    };
                    let key = super::public_key::sampled_key(
                        self.parameters,
                        level,
                        hash_key.clone(),
                        *wire,
                        knowledge,
                        None,
                    )
                    .map_err(ApiError::scheme)?;
                    InputBinding::Scalar(RnsBggEncoding::new(
                        key,
                        (port(0, columns), port(1, columns)),
                        public.then(|| port(2, 1)),
                    ))
                }
                InputBinding::Family(count) => {
                    let port = |port: usize, columns: usize| {
                        ring.family_artifact_input(
                            encryption_production.clone(),
                            format!("input-{wire:?}.{port}"),
                            count.clone(),
                            (1, columns),
                            ArtifactAvailability::Transferred,
                        )
                    };
                    let row_ports = [port(0, columns), port(1, columns)];
                    let element: RnsBggRow =
                        (row_ports[0].at(Int::constant(0)), row_ports[1].at(Int::constant(0)));
                    let fields = row_ports
                        .iter()
                        .map(|family| family.value_handle().clone())
                        .collect::<Vec<_>>();
                    InputBinding::Family(OnlineWireFamily::Input {
                        wire: *wire,
                        level,
                        knowledge,
                        hash_key: hash_key.clone(),
                        rows: family_of_fields(&element, count.clone(), &fields)
                            .map_err(ApiError::scheme)?,
                        payloads: public.then(|| port(2, 1)),
                    })
                }
            };
            self.environment
                .bind_wire(*wire, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        for (rhs, binding) in import_rhs_bindings(
            self.parameters,
            program,
            &workload,
            &preprocess_production,
            &hash_key,
            |role| role,
        )? {
            self.environment
                .bind_rhs(rhs, binding)
                .map_err(|error| ApiError::input(error.to_string()))?;
        }
        self.evaluator = Some(EncodingLowering {
            parameters: self.parameters,
            program: Arc::new(program.clone()),
            workload,
            preprocess: preprocess_production,
            encryption: encryption_production,
            manifests,
            refresh_prf_namespace: None,
            current_site: None,
            hash_key,
        });
        Ok(())
    }

    fn build_graph(mut self, outputs: Vec<RnsBggEncoding>) -> Result<StageGraph, ApiError> {
        let mut outputs = outputs;
        let pending = self.pending_refresh.take();
        if let Some((index, declaration, mut children)) = pending {
            self.finalize_refresh(index, &declaration, std::mem::take(&mut children), &mut outputs)
                .map_err(ApiError::scheme)?;
        }
        let evaluator = self
            .evaluator
            .as_mut()
            .ok_or_else(|| ApiError::input("online artifacts are not bound"))?;
        let outputs = evaluator.finish_outputs(outputs).map_err(ApiError::scheme)?;
        let mut graph = DslContext::new("rns-bgg-online");
        for (index, output) in outputs.into_iter().enumerate() {
            graph = graph.output(format!("output-{index}"), output).map_err(ApiError::scheme)?;
        }
        let graph = graph.build().map_err(ApiError::scheme)?;
        let mut runtime = BTreeMap::new();
        if super::graph_declares_input(&graph, super::HASH_KEY_INPUT) {
            runtime.insert(
                super::HASH_KEY_INPUT.to_owned(),
                RuntimeValue::Bytes(self.hash_key.as_slice().into()),
            );
        }
        Ok(StageGraph::new(graph, runtime, evaluator.manifests.clone()))
    }
}

/// Encoding-side implementation of the common one-shot refresh contract.
impl NoiseRefresh for RnsBggRefresh<'_, EncodingLowering<'_>> {
    type PrfOutput = BTreeMap<usize, RnsBggEncoding>;
    type Output = RnsBggEncoding;
    type Error = RnsBggError;
    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        outputs: BTreeMap<PrfLabel, Self::PrfOutput>,
    ) -> Result<Self::Output, Self::Error> {
        let inputs = self.inputs().clone();
        let slots = declaration.slots();
        let digits = declaration.digits();
        let labels = declaration.prf_labels();
        let label_count = declaration.labels().len();
        let target_level = declaration.target_level();
        if inputs.len() != slots.len() || inputs.keys().copied().ne(0..slots.len()) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }
        let source =
            inputs.get(&0).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?.clone();
        let lowerer = self.lowerer();
        for input in inputs.values() {
            if input.wire() != declaration.source_output() {
                return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
            }
        }
        lowerer.validate_refresh_declaration(declaration)?;
        let target_parameters = lowerer.parameters.level(target_level)?;
        let target_ring = lowerer.parameters.ring(target_level)?;
        let target_columns = lowerer.parameters.component_columns(target_level)?;
        let expected_port = target_ring.matrix_type((1, target_columns));
        let schema_ports = declaration
            .output_schema()
            .iter()
            .filter_map(|wire| match wire {
                WireType::Matrix(matrix) => Some(matrix),
                _ => None,
            })
            .collect::<Vec<_>>();
        if schema_ports.len() != 4 || schema_ports.iter().any(|port| **port != expected_port) {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        let mut target_modulus = BigUint::from(1u8);
        for slot in slots {
            target_modulus *= slot
                .q_t
                .to_biguint()
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidModulusRelation))?;
        }
        if target_modulus != *target_parameters.modulus().as_ref() {
            return Err(RnsBggError::Refresh(RefreshError::InvalidModulusRelation));
        }
        if digits.mask_counts.len() != slots.len() || digits.fresh_count == 0 {
            return Err(RnsBggError::Refresh(RefreshError::InvalidLayout));
        }

        let mut channels = Vec::with_capacity(slots.len());
        for slot in slots {
            channels.push((slot.slot, None::<RnsBggRow>));
        }
        let ordered_outputs = declaration
            .prepared()
            .declaration()
            .ordered_outputs()
            .map(|(label, wire)| (label.clone(), wire))
            .collect::<Vec<_>>();
        for index in 0..label_count {
            let label = declaration
                .labels()
                .label(index)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let prf_label =
                labels.get(index).ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let expected_wire = ordered_outputs
                .get(index)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?
                .1;
            let channel_outputs = outputs
                .get(prf_label)
                .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
            let expected = match label {
                RefreshPrfLabel::Mask { slot, .. } => vec![slot],
                RefreshPrfLabel::FreshError { .. } => slots.iter().map(|slot| slot.slot).collect(),
            };
            if channel_outputs.keys().copied().collect::<Vec<_>>() != expected {
                return Err(RnsBggError::Refresh(RefreshError::PrfOutputMismatch));
            }
            for slot in expected {
                let value = channel_outputs
                    .get(&slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::PrfOutputMismatch))?;
                if value.wire() != expected_wire {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                if source.public_payload.is_some() != value.public_payload.is_some() {
                    return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                }
                if let (Some(source_payload), Some(payload)) =
                    (&source.public_payload, &value.public_payload)
                {
                    if source_payload.matrix_type() != payload.matrix_type() {
                        return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
                    }
                }
                let terminal =
                    lowerer.refresh_action_for_label(declaration, Some(index), slot, "target")?;
                // The preprocessing action already includes the positional
                // digit weight (and H for shared fresh error). Applying it to
                // the row here would weight every PRF digit twice.
                let routed = multiply_encoding_action(value.row.clone(), terminal);
                let (_, accumulator) = channels
                    .get_mut(slot)
                    .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
                *accumulator = Some(match accumulator.take() {
                    Some(previous) => add_encoding_rows(previous, routed),
                    None => routed,
                });
            }
        }
        let mut channel_rows = Vec::with_capacity(channels.len());
        let mut channel_weights = Vec::with_capacity(channels.len());
        for (slot, row) in channels {
            let mut row = row.ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let target = lowerer.refresh_action(declaration, slot, "target")?;
            let input =
                inputs.get(&slot).ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            row = add_encoding_rows(multiply_encoding_action(input.row.clone(), target), row);
            // Terminal actions aggregate in the source (q) ring.  Collapse
            // that aggregate once to the slot's producer modulus before the
            // decoder, which is emitted in the p-level ring.
            let slot_metadata = declaration
                .slots()
                .get(slot)
                .ok_or(RnsBggError::Refresh(RefreshError::InvalidLayout))?;
            let producer_level = refresh_slot_level(lowerer.parameters, declaration, slot)?;
            row = mod_down_encoding(row, &lowerer.parameters.ring(producer_level)?);
            let decoder = lowerer.refresh_decoder(declaration, slot)?;
            row = add_encoding_rows(row, (-decoder.0, -decoder.1));
            row = mod_down_lift_encoding(
                row,
                &lowerer.parameters.ring_for_modulus(&slot_metadata.q_t)?,
                &target_ring,
            );
            channel_rows.push(row);
            channel_weights.push(lowerer.refresh_crt_weight(declaration, slot)?);
        }
        let values = Family::pack(channel_rows).map_err(graph)?;
        let coefficients = Family::pack(channel_weights).map_err(graph)?;
        let result = recompose_encoding(values, coefficients)?;
        if result.0.matrix_type() != &expected_port || result.1.matrix_type() != &expected_port {
            return Err(RnsBggError::Refresh(RefreshError::InvalidOutputSchema));
        }
        let public_payload = match source.public_payload.clone() {
            Some(payload) => Some(checked_modulus_convert(payload, &target_ring)?),
            None => None,
        };
        let refreshed_public_key = lowerer.refresh_public_target(declaration, source.wire())?;
        Ok(RnsBggEncoding::new(refreshed_public_key, result, public_payload))
    }
}

fn mod_down_lift_encoding(row: RnsBggRow, channel_ring: &Ring, target_ring: &Ring) -> RnsBggRow {
    map_row(row, |entry| entry.block_mod_switch(channel_ring, 1).centered_rebase(target_ring))
}

fn reduce_anchor_channel(
    a_star: RnsBggMatrix,
    channel_ring: &Ring,
    target_ring: &Ring,
) -> Result<RnsBggMatrix, RnsBggError> {
    let reduced = checked_modulus_convert_matrix(a_star, channel_ring)?;
    Ok(map_matrix(reduced, |entry| entry.centered_rebase(target_ring)))
}

#[cfg(test)]
mod test {
    use super::*;
    use mxx_backends::{
        MemoryArtifactStore, backend::poly::cpu_backend, poly::dcrt::poly::DCRTPoly,
    };

    #[test]
    fn lut_coefficients_evaluate_values_with_distinct_input_and_output_layouts() {
        let level = DCRTPolyParams::new(16, 1, 17, 2, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let input_layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let output_layout = ExpLayout { stride_log2: 2, width_log2: 2 };
        let mut graph = DslContext::new("lut-layout-arithmetic");
        let values = [0usize, 1, 3, 2];
        for (role, representation) in [
            crate::program::LutOutputRepresentation::Coefficient,
            crate::program::LutOutputRepresentation::Exponent(output_layout),
        ]
        .into_iter()
        .enumerate()
        {
            for (input, expected) in values.iter().copied().enumerate() {
                let monomial = input_layout.mono(&ring, &BigInt::from(input)).unwrap();
                let mut actual = ring.zero((1, 1));
                for sigma in input_layout.automorphism_indices(16).unwrap() {
                    let coefficient = EncodingLowering::lut_coefficient(
                        &parameters,
                        0,
                        input_layout,
                        representation,
                        IntExpr::constant(sigma),
                        &values,
                    )
                    .unwrap();
                    actual = actual + coefficient * monomial.clone().ring_automorphism(sigma);
                }
                let expected = match representation {
                    crate::program::LutOutputRepresentation::Coefficient => {
                        ring.polynomial([IntExpr::constant(expected)])
                    }
                    crate::program::LutOutputRepresentation::Exponent(layout) => {
                        layout.mono(&ring, &BigInt::from(expected)).unwrap()
                    }
                };
                graph =
                    graph.output(format!("difference-{role}-{input}"), actual - expected).unwrap();
            }
        }
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend([level]));
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            StageGraph::new(graph.build().unwrap(), BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for name in result.outputs.keys().cloned().collect::<Vec<_>>() {
            let matrix = crate::test_support::full(&result.outputs[&name]);
            assert!(
                matrix
                    .entry(0, 0)
                    .coeffs_biguints()
                    .iter()
                    .all(|value| *value == BigUint::from(0u8)),
                "{name}"
            );
        }
    }

    /// A finalized output below the terminal level is refused: its gadget
    /// powers need not exceed its noise.
    #[test]
    fn intermediate_coefficient_verification_is_refused() {
        let high = DCRTPolyParams::new(16, 3, 17, 1, None, None);
        let primes = high.to_crt().0;
        let middle =
            high.select_modulus(&(BigUint::from(primes[0]) * BigUint::from(primes[1]))).unwrap();
        let low = high.select_modulus(&BigUint::from(primes[0])).unwrap();
        let parameters =
            RnsBggParameters::new(vec![high.clone(), middle.clone(), low], 1, 1).unwrap();
        check_finalized_coefficient_verification(&parameters, &middle, false);
    }

    /// The terminal level has an identity state gadget, so only its public
    /// message scale separates a correct output from a wrong one.
    #[test]
    fn terminal_coefficient_verification_rejects_wrong_plaintext() {
        let high = DCRTPolyParams::new(16, 2, 17, 1, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low.clone()], 1, 1).unwrap();
        check_finalized_coefficient_verification(&parameters, &low, true);
    }

    fn check_finalized_coefficient_verification(
        parameters: &RnsBggParameters,
        output_level: &DCRTPolyParams,
        terminal: bool,
    ) {
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let lut = builder
            .register_lut(
                LutTable::unary(
                    2,
                    ExpLayout { stride_log2: 0, width_log2: 1 },
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Coefficient,
                )
                .unwrap(),
            )
            .unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash(rand::random()),
            execution_nonce: rand::random(),
        };
        let mut lowerer = EncodingLowering {
            parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            workload: Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap()),
            program: program.clone(),
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: None,
        };
        let source_ring = parameters.ring(0).unwrap();
        let payload = RnsBggPayloadSecret {
            row: (source_ring.identity(1), source_ring.identity(1)),
            binding: None,
        };
        let mask = RnsBggMaskSecret {
            row: (source_ring.zero((1, 1)), source_ring.identity(1)),
            binding: None,
        };
        let columns = parameters.component_columns(1).unwrap();
        assert_ne!(columns, parameters.component_columns(0).unwrap());
        let zero = source_ring.zero((1, columns));
        let key =
            RnsBggPublicKey::new(output, ((zero.clone(), zero.clone()), (zero.clone(), zero)));
        let target_gadget =
            parameters.state_gadget(1).unwrap().centered_rebase(&parameters.ring(0).unwrap());
        let scale = parameters.message_scale(0).unwrap() * parameters.message_scale(1).unwrap();
        let raw_signal = target_gadget * parameters.scalar(0, scale).unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend(
            parameters.levels().iter().cloned(),
        ));
        let mut store = MemoryArtifactStore::default();
        for bit in [0usize, 1] {
            let row = -raw_signal.clone() * source_ring.polynomial([IntExpr::constant(bit)]);
            let raw = RnsBggEncoding::new(key.clone(), (row.clone(), row), None);
            let finalized = lowerer.finish_outputs(vec![raw]).unwrap().pop().unwrap();
            let graph = DslContext::new("coefficient-finalization-verification")
                .output("output-0", finalized)
                .unwrap()
                .build()
                .unwrap();
            let result = execute_graph(
                StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let verified = verify_output(
                &program,
                &result,
                0,
                &DCRTPoly::from_usize_to_constant(output_level, bit),
                parameters,
                &payload,
                &mask,
                &mut execution,
                &mut store,
            );
            if !terminal {
                assert!(verified.is_err(), "an intermediate output must not be verified");
                continue;
            }
            verified.unwrap();
            assert!(
                verify_output(
                    &program,
                    &result,
                    0,
                    &DCRTPoly::from_usize_to_constant(output_level, 1 - bit),
                    parameters,
                    &payload,
                    &mask,
                    &mut execution,
                    &mut store,
                )
                .is_err(),
                "wrong coefficient plaintext must be rejected"
            );
        }
    }

    #[test]
    fn public_payload_verification_rejects_corruption() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let mut builder = ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(input)]).unwrap());
        let ring = parameters.ring(0).unwrap();
        let columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let zero = ring.zero((1, columns));
        let key = RnsBggPublicKey::new(
            input,
            ((zero.clone(), zero.clone()), (zero.clone(), zero.clone())),
        );
        let expected_poly = DCRTPoly::from_usize_to_constant(&level, 1);
        let expected = plaintext_constant(&ring, &expected_poly);
        let mask = RnsBggMaskSecret { row: (ring.zero((1, 1)), ring.identity(1)), binding: None };
        let delta = parameters.scalar(0, parameters.scale(0).unwrap().clone()).unwrap();
        let deterministic =
            mask.row.1.clone() * expected.clone() * parameters.public_gadget(0).unwrap() * delta;
        let row = (ring.zero((1, columns)), -deterministic);
        let payload =
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None };

        for (name, public_payload, expected_result) in [
            ("public-payload-valid", expected.clone(), Ok(())),
            ("public-payload-corrupt", expected + ring.identity(1), Err(())),
        ] {
            let graph = DslContext::new(name)
                .output(
                    "output-0",
                    RnsBggEncoding::new(key.clone(), row.clone(), Some(public_payload)),
                )
                .unwrap()
                .build()
                .unwrap();
            let mut execution =
                mxx_backends::authority::CpuExecution::new(cpu_backend([level.clone()]));
            let mut store = MemoryArtifactStore::default();
            let result = execute_graph(
                StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
                &ParamEnv::default(),
                &mut execution,
                &mut store,
                |_| Ok(()),
            )
            .unwrap();
            let verified = verify_output(
                &program,
                &result,
                0,
                &expected_poly,
                &parameters,
                &payload,
                &mask,
                &mut execution,
                &mut store,
            )
            .map(|_| ());
            assert_eq!(verified.is_ok(), expected_result.is_ok(), "{name}");
        }
    }

    #[test]
    fn encryption_lut_helper_produces_one_compact_member() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let one = ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
        let public_data = ((one.clone(), one.clone()), (one.clone(), one.clone()));
        let builder = RnsEncryptBuild::new(&parameters, [0; 32]);
        let mask = RnsBggMaskSecret {
            row: (ring.uniform_interval((1, 1), -1, 1), ring.identity(1)),
            binding: None,
        };
        let helper =
            builder.encrypt_lut_helper(&mask, public_data, 0, IntExpr::constant(1)).unwrap();
        assert_eq!(helper.0.matrix_type().columns, IntExpr::constant(1));
    }

    #[test]
    fn encryption_projection_identity_is_sensitive_to_lut_semantics() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let layout = ExpLayout { stride_log2: 0, width_log2: 1 };
        let mut first = crate::program::ExponentLutProgramBuilder::new();
        let input = first.input(AttributeKnowledge::PrivatePayload).unwrap();
        first
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![0, 1],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let first = first.finish(vec![Operand::Scalar(input)]).unwrap();
        let mut second = crate::program::ExponentLutProgramBuilder::new();
        let input = second.input(AttributeKnowledge::PrivatePayload).unwrap();
        second
            .register_lut(
                LutTable::unary(
                    2,
                    layout,
                    2,
                    vec![1, 0],
                    crate::program::LutOutputRepresentation::Exponent(layout),
                )
                .unwrap(),
            )
            .unwrap();
        let second = second.finish(vec![Operand::Scalar(input)]).unwrap();
        let env = mxx_ir_core::ParamEnv::default();
        assert_ne!(
            encryption_projection_identity(&first.encryption_inputs(), &env, &parameters).unwrap(),
            encryption_projection_identity(&second.encryption_inputs(), &env, &parameters).unwrap()
        );
    }

    #[test]
    fn lut_coefficients_depend_on_values_not_only_width() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let constant = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 0],
        )
        .unwrap();
        let identity = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Coefficient,
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(constant.flatten(), identity.flatten());
        let declared_layout = EncodingLowering::lut_coefficient(
            &parameters,
            0,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            crate::program::LutOutputRepresentation::Exponent(ExpLayout {
                stride_log2: 1,
                width_log2: 1,
            }),
            IntExpr::constant(1),
            &[0, 1],
        )
        .unwrap();
        assert_ne!(identity.flatten(), declared_layout.flatten());
    }

    #[test]
    fn lowering_lut_path_consumes_table_coefficients() {
        fn lower(
            values: Vec<usize>,
            representation: crate::program::LutOutputRepresentation,
        ) -> RnsBggEncoding {
            let table = LutTable::unary(
                2,
                ExpLayout { stride_log2: 0, width_log2: 1 },
                2,
                values,
                representation,
            )
            .unwrap();
            let mut builder = crate::program::ExponentLutProgramBuilder::new();
            let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
            let lut = builder.register_lut(table.clone()).unwrap();
            let output = builder.lut(Operand::Scalar(input), lut).unwrap();
            let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
            let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
            let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
            let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
            let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
            let production = ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([7; 32]),
                execution_nonce: [9; 32],
            };
            let mut lowerer = EncodingLowering {
                parameters: &parameters,
                hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
                program,
                workload,
                preprocess: production.clone(),
                encryption: production,
                manifests: BTreeMap::new(),
                refresh_prf_namespace: None,
                current_site: Some(LoweringSite {
                    output,
                    indices: Vec::new(),
                    ranges: Vec::new(),
                }),
            };
            let ring = parameters.ring(0).unwrap();
            let columns = parameters.component_columns(0).unwrap();
            let value = ring.input("lut-lowering-input", (1, columns));
            let key = RnsBggPublicKey::new(
                input,
                ((value.clone(), value.clone()), (value.clone(), value.clone())),
            );
            let input = RnsBggEncoding::new(key, (value.clone(), value), None);
            <EncodingLowering<'_> as ExponentLUTProgramLowering>::lut(&mut lowerer, &input, &table)
                .unwrap()
        }

        let constant = lower(vec![0, 0], crate::program::LutOutputRepresentation::Coefficient);
        let identity = lower(vec![0, 1], crate::program::LutOutputRepresentation::Coefficient);
        assert_ne!(constant.flatten(), identity.flatten());
        let exponent = lower(
            vec![0, 1],
            crate::program::LutOutputRepresentation::Exponent(ExpLayout {
                stride_log2: 0,
                width_log2: 1,
            }),
        );
        assert_ne!(identity.flatten(), exponent.flatten());
    }

    #[test]
    fn rhs_imports_use_next_level_compact_schema_for_scalar_and_family() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let source_columns = parameters.component_columns(0).unwrap();
        let compact_columns = parameters.component_columns(1).unwrap();
        assert_ne!(source_columns, compact_columns);
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([29; 32]),
            execution_nonce: [31; 32],
        };
        let compact_type = parameters.ring(1).unwrap().matrix_type((1, compact_columns));
        let decomposed_type = parameters
            .ring(0)
            .unwrap()
            .matrix_type((2 * parameters.level(1).unwrap().modulus_digits(), compact_columns));
        let decomposed_bound = IntExpr::RoundDiv(
            Box::new(IntExpr::constant(
                BigInt::from(1u8) << parameters.level(1).unwrap().base_bits(),
            )),
            Box::new(IntExpr::constant(BigInt::from(2u8))),
        )
        .canonicalize();
        let lowerer = |program: &Arc<ExponentLutProgram>, output| EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program: Arc::clone(program),
            workload: Arc::new(RnsBggDerivedWorkload::from_program(program).unwrap()),
            preprocess: production.clone(),
            encryption: production.clone(),
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let assert_compact = |matrix: &RnsBggMatrix| {
            for entry in [&matrix.0.0, &matrix.0.1, &matrix.1.0, &matrix.1.1] {
                assert_eq!(entry.matrix_type(), &compact_type);
            }
        };

        let mut scalar_builder = crate::program::ExponentLutProgramBuilder::new();
        let scalar_input = scalar_builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let scalar_rhs = scalar_builder.rhs_input().unwrap();
        let scalar_output = scalar_builder
            .mul_private(Operand::Scalar(scalar_input), Operand::Scalar(scalar_rhs))
            .unwrap();
        let scalar_program =
            Arc::new(scalar_builder.finish(vec![Operand::Scalar(scalar_output)]).unwrap());
        let scalar_lowerer = lowerer(&scalar_program, scalar_output);
        let hash_key = scalar_lowerer.hash_key.clone();
        let bindings = import_rhs_bindings(
            &parameters,
            &scalar_program,
            &scalar_lowerer.workload,
            &production,
            &hash_key,
            |role| role,
        )
        .unwrap();
        let [(id, InputBinding::Scalar(scalar_compact))] = bindings.as_slice() else {
            panic!("one scalar RHS binding expected");
        };
        assert_eq!(*id, scalar_rhs);
        assert_compact(scalar_compact);
        let scalar_action = scalar_lowerer.rhs_decomposition(scalar_compact, 0).unwrap();
        assert_eq!(scalar_action.0.matrix_type(), &decomposed_type);
        assert_eq!(scalar_action.0.max_coefficient_bound(), &decomposed_bound);
        let scalar_name = format!("rhs/{scalar_rhs:?}");
        let scalar_graph = DslContext::new("rhs-scalar-consumer-schema")
            .transferred_output(scalar_name.clone(), scalar_action)
            .unwrap()
            .build()
            .unwrap();
        assert!(scalar_graph.graph.outputs().contains_key(&format!("{scalar_name}.0")));

        let mut family_builder = crate::program::ExponentLutProgramBuilder::new();
        let family_input = family_builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let family_rhs = family_builder.rhs_family(IntExpr::constant(2)).unwrap();
        let family_products = family_builder
            .parallel(
                crate::program::FamilyRange::full(IntExpr::constant(2)).unwrap(),
                |builder, index| {
                    builder
                        .mul_private(
                            Operand::Scalar(family_input),
                            Operand::Member { family: family_rhs, index },
                        )
                        .map(Operand::Scalar)
                },
            )
            .unwrap();
        let family_output = family_builder.sum(family_products).unwrap();
        let family_program =
            Arc::new(family_builder.finish(vec![Operand::Scalar(family_output)]).unwrap());
        let family_lowerer = lowerer(&family_program, family_output);
        let bindings = import_rhs_bindings(
            &parameters,
            &family_program,
            &family_lowerer.workload,
            &production,
            &hash_key,
            |role| role,
        )
        .unwrap();
        let [(id, InputBinding::Family(family))] = bindings.as_slice() else {
            panic!("one family RHS binding expected");
        };
        assert_eq!(*id, family_rhs);
        // Only the bottom rows are imported, as the port families themselves:
        // no loop reads them.
        assert_eq!(family.bottoms.flatten().len(), 2);
        assert!(family.bottoms.flatten().iter().all(|value| matches!(
            value.node().kind(),
            mxx_ir_core::node::NodeKind::Input { artifact: Some(_), .. }
        )));
        let mut families = OnlineFamilies { parameters: &parameters };
        let mut member_at = |index: IntExpr| {
            <OnlineFamilies<'_> as crate::program::FamilyExecution<EncodingLowering<'_>>>::rhs_at(
                &mut families,
                family,
                &index,
            )
            .unwrap()
        };
        let member = member_at(IntExpr::constant(1));
        assert_compact(&member);
        let family_action = family_lowerer.rhs_decomposition(&member, 0).unwrap();
        assert_eq!(family_action.0.matrix_type(), &decomposed_type);
        assert_eq!(family_action.0.max_coefficient_bound(), &decomposed_bound);
        let family_name = format!("rhs/{family_rhs:?}/family");
        let family_members = mxx_dsl::parallel(IntExpr::constant(2), |index| {
            family_lowerer
                .rhs_decomposition(&member_at(index.expression()?), 0)
                .map_err(|_| DslError::Schema)
        })
        .unwrap();
        let family_graph = DslContext::new("rhs-family-consumer-schema")
            .transferred_output(family_name.clone(), family_members)
            .unwrap()
            .build()
            .unwrap();
        assert!(family_graph.graph.outputs().contains_key(&format!("{family_name}.0")));
    }

    #[test]
    fn mul_public_uses_only_second_payload_and_propagates_public_product() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_wire = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let output = builder
            .mul_public(Operand::Scalar(public_wire), Operand::Scalar(private_wire))
            .unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([19; 32]),
            execution_nonce: [23; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production.clone(),
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let ring = parameters.ring(0).unwrap();
        let private_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_row = ring.input("mul-public-private-row", (1, private_columns));
        let public_row = ring.input("mul-public-public-row", (1, public_columns));
        let public_payload = ring.input("mul-public-payload", (1, 1));
        let private_key = RnsBggPublicKey::new(
            private_wire,
            (
                (private_row.clone(), private_row.clone()),
                (private_row.clone(), private_row.clone()),
            ),
        );
        let public_key = RnsBggPublicKey::new(
            public_wire,
            ((public_row.clone(), public_row.clone()), (public_row.clone(), public_row.clone())),
        );
        let private =
            RnsBggEncoding::new(private_key, (private_row.clone(), private_row.clone()), None);
        let public = RnsBggEncoding::new(
            public_key.clone(),
            (public_row.clone(), public_row.clone()),
            Some(public_payload.clone()),
        );
        let result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public,
            &private,
        )
        .unwrap();
        assert!(result.public_payload.is_none());
        let public_with_other_payload = RnsBggEncoding::new(
            public_key.clone(),
            (public.row.0.clone(), public.row.1.clone()),
            Some(ring.input("mul-public-other-payload", (1, 1))),
        );
        let other_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public_with_other_payload,
            &private,
        )
        .unwrap();
        assert_ne!(result.row.flatten(), other_result.row.flatten());

        let mut public_builder = crate::program::ExponentLutProgramBuilder::new();
        let public_lhs_wire = public_builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_rhs_wire = public_builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let public_output = public_builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(public_rhs_wire))
            .unwrap();
        let public_program =
            Arc::new(public_builder.finish(vec![Operand::Scalar(public_output)]).unwrap());
        let public_workload =
            Arc::new(RnsBggDerivedWorkload::from_program(&public_program).unwrap());
        let mut public_lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program: public_program,
            workload: public_workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite {
                output: public_output,
                indices: Vec::new(),
                ranges: Vec::new(),
            }),
        };
        let lhs_payload = ring.input("mul-public-lhs-payload", (1, 1));
        let rhs_payload = public_payload.clone();
        let public_lhs = RnsBggEncoding::new(
            RnsBggPublicKey::new(
                public_lhs_wire,
                (
                    (public_row.clone(), public_row.clone()),
                    (public_row.clone(), public_row.clone()),
                ),
            ),
            (public_row.clone(), public_row.clone()),
            Some(lhs_payload.clone()),
        );
        let public_rhs = RnsBggEncoding::new(
            RnsBggPublicKey::new(
                public_rhs_wire,
                (
                    (public_row.clone(), public_row.clone()),
                    (public_row.clone(), public_row.clone()),
                ),
            ),
            (public_row.clone(), public_row),
            Some(rhs_payload.clone()),
        );
        let public_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut public_lowerer,
            &public_lhs,
            &public_rhs,
        )
        .unwrap();
        let output_payload = public_result.public_payload.expect("public product payload");
        assert_eq!(output_payload.matrix_type(), &ring.matrix_type((1, 1)));
    }

    #[test]
    fn mul_public_aligns_public_carrier_before_private_same_level_action() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let rhs = builder.rhs_input().unwrap();
        let private_level_one =
            builder.mul_private(Operand::Scalar(private_input), Operand::Scalar(rhs)).unwrap();
        let output = builder
            .mul_public(Operand::Scalar(public_input), Operand::Scalar(private_level_one))
            .unwrap();
        // Also an output, the public input stays at level zero and is aligned.
        let program = Arc::new(
            builder.finish(vec![Operand::Scalar(output), Operand::Scalar(public_input)]).unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([41; 32]),
            execution_nonce: [43; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        let low_ring = parameters.ring(1).unwrap();
        let high_ring = parameters.ring(0).unwrap();
        let private_columns =
            parameters.key_component_columns(1, AttributeKnowledge::PrivatePayload).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_entry = low_ring.input("mul-public-private-level-one", (1, private_columns));
        let public_entry = high_ring.input("mul-public-public-level-zero", (1, public_columns));
        let private_key = RnsBggPublicKey::new(
            private_level_one,
            (
                (private_entry.clone(), private_entry.clone()),
                (private_entry.clone(), private_entry.clone()),
            ),
        );
        let public_key = RnsBggPublicKey::new(
            public_input,
            (
                (public_entry.clone(), public_entry.clone()),
                (public_entry.clone(), public_entry.clone()),
            ),
        );
        let private =
            RnsBggEncoding::new(private_key, (private_entry.clone(), private_entry.clone()), None);
        let public_payload = high_ring.input("mul-public-level-zero-payload", (1, 1));
        let public = RnsBggEncoding::new(
            public_key,
            (public_entry.clone(), public_entry),
            Some(public_payload),
        );
        let result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &public,
            &private,
        )
        .unwrap();
        let expected = low_ring.matrix_type((1, private_columns));
        assert_eq!(result.row.0.matrix_type(), &expected);
        assert_eq!(result.row.1.matrix_type(), &expected);
        for entry in [
            &result.public_key.matrix.0.0,
            &result.public_key.matrix.0.1,
            &result.public_key.matrix.1.0,
            &result.public_key.matrix.1.1,
        ] {
            assert_eq!(entry.matrix_type(), &expected);
        }
        assert!(result.public_payload.is_none());
    }

    #[test]
    fn mul_public_encoding_rows_match_independent_residual_oracle() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let public_lhs_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_rhs_wire = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let public_rhs_wire = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let private_output = builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(private_rhs_wire))
            .unwrap();
        let public_output = builder
            .mul_public(Operand::Scalar(public_lhs_wire), Operand::Scalar(public_rhs_wire))
            .unwrap();
        let program = Arc::new(
            builder
                .finish(vec![Operand::Scalar(private_output), Operand::Scalar(public_output)])
                .unwrap(),
        );
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level.clone()], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let public_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PublicPayload).unwrap();
        let private_columns =
            parameters.key_component_columns(0, AttributeKnowledge::PrivatePayload).unwrap();
        let entry = |columns, index| {
            ring.constant((1, columns), ConstantMatrix::UnitRow { index: IntExpr::constant(index) })
        };
        let lhs_entry = entry(public_columns, 0);
        let private_entry = entry(private_columns, 0);
        let private_entry_alt = private_entry.clone() * IntExpr::constant(2);
        let public_rhs_entry = entry(public_columns, 1);
        let lhs_key = ((lhs_entry.clone(), lhs_entry.clone()), (lhs_entry.clone(), lhs_entry));
        let private_rhs_key = (
            (private_entry.clone(), private_entry_alt.clone()),
            (private_entry_alt.clone(), private_entry),
        );
        let public_rhs_key = (
            (public_rhs_entry.clone(), public_rhs_entry.clone()),
            (public_rhs_entry.clone(), public_rhs_entry),
        );
        let mask = (ring.identity(1), ring.zero((1, 1)));
        let payload = (ring.identity(1), ring.zero((1, 1)));
        let lhs_plaintext = ring.polynomial([IntExpr::constant(2)]);
        let private_plaintext = ring.polynomial([IntExpr::constant(3)]);
        let public_rhs_plaintext = ring.polynomial([IntExpr::constant(5)]);
        let public_gadget = parameters.public_gadget(0).unwrap();
        let state_gadget = parameters.state_gadget(0).unwrap();
        let delta = parameters.scalar(0, parameters.message_scale(0).unwrap()).unwrap();
        let mask_times_key = |key: &RnsBggMatrix| {
            (
                mask.0.clone() * key.0.0.clone() + mask.1.clone() * key.1.0.clone(),
                mask.0.clone() * key.0.1.clone() + mask.1.clone() * key.1.1.clone(),
            )
        };
        let public_signal =
            (mask.0.clone() * public_gadget.clone(), mask.1.clone() * public_gadget.clone());
        let private_signal = (
            payload.0.clone() * state_gadget.clone() * delta.clone() * private_plaintext.clone(),
            payload.1.clone() * state_gadget * delta * private_plaintext,
        );
        let lhs_row = {
            let carrier = mask_times_key(&lhs_key);
            (
                carrier.0 - public_signal.0.clone() * lhs_plaintext.clone(),
                carrier.1 - public_signal.1.clone() * lhs_plaintext.clone(),
            )
        };
        let private_rhs_row = {
            let carrier = mask_times_key(&private_rhs_key);
            (carrier.0 - private_signal.0.clone(), carrier.1 - private_signal.1.clone())
        };
        let public_rhs_row = {
            let carrier = mask_times_key(&public_rhs_key);
            (
                carrier.0 - public_signal.0.clone() * public_rhs_plaintext.clone(),
                carrier.1 - public_signal.1.clone() * public_rhs_plaintext.clone(),
            )
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([127; 32]),
                execution_nonce: [129; 32],
            },
            encryption: ProductionId {
                spec_hash: mxx_ir_core::artifact::SpecHash([127; 32]),
                execution_nonce: [129; 32],
            },
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite {
                output: private_output,
                indices: Vec::new(),
                ranges: Vec::new(),
            }),
        };
        let private_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_lhs_wire, lhs_key.clone()),
                lhs_row.clone(),
                Some(lhs_plaintext.clone()),
            ),
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(private_rhs_wire, private_rhs_key.clone()),
                private_rhs_row.clone(),
                None,
            ),
        )
        .unwrap();
        lowerer.current_site =
            Some(LoweringSite { output: public_output, indices: Vec::new(), ranges: Vec::new() });
        let public_result = <EncodingLowering<'_> as ExponentLUTProgramLowering>::mul_public(
            &mut lowerer,
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_lhs_wire, lhs_key.clone()),
                lhs_row,
                Some(lhs_plaintext.clone()),
            ),
            &RnsBggEncoding::new(
                RnsBggPublicKey::new(public_rhs_wire, public_rhs_key.clone()),
                public_rhs_row,
                Some(public_rhs_plaintext.clone()),
            ),
        )
        .unwrap();
        let base = 1u64 << level.base_bits();
        let columns = level.modulus_digits();
        let direct_decompose = |matrix: RnsBggMatrix| {
            (
                (matrix.0.0.decompose(base, columns), matrix.0.1.decompose(base, columns)),
                (matrix.1.0.decompose(base, columns), matrix.1.1.decompose(base, columns)),
            )
        };
        let direct_action = |lhs: RnsBggMatrix, rhs: RnsBggDigits| {
            (
                (
                    lhs.0.0.clone().mul_small_rhs(rhs.0.0.clone()) +
                        lhs.0.1.clone().mul_small_rhs(rhs.1.0.clone()),
                    lhs.0.0.mul_small_rhs(rhs.0.1.clone()) + lhs.0.1.mul_small_rhs(rhs.1.1.clone()),
                ),
                (
                    lhs.1.0.clone().mul_small_rhs(rhs.0.0) + lhs.1.1.clone().mul_small_rhs(rhs.1.0),
                    lhs.1.0.mul_small_rhs(rhs.0.1) + lhs.1.1.mul_small_rhs(rhs.1.1),
                ),
            )
        };
        let expected_private_key =
            direct_action(lhs_key.clone(), direct_decompose(private_rhs_key.clone()));
        let expected_public_key =
            direct_action(lhs_key.clone(), direct_decompose(public_rhs_key.clone()));
        let direct_row_action = |row: RnsBggRow, key: RnsBggMatrix| {
            let decomposition = direct_decompose(key);
            (
                row.0.clone().mul_small_rhs(decomposition.0.0) +
                    row.1.clone().mul_small_rhs(decomposition.1.0),
                row.0.mul_small_rhs(decomposition.0.1) + row.1.mul_small_rhs(decomposition.1.1),
            )
        };
        let mask_of = |key: &RnsBggMatrix| mask_times_key(key);
        let public_action_for_private =
            direct_row_action(public_signal.clone(), private_rhs_key.clone());
        let private_rhs_carrier = mask_of(&private_rhs_key);
        let private_expected_signal = (
            public_action_for_private.0 * lhs_plaintext.clone() +
                private_rhs_carrier.0 * -lhs_plaintext.clone() +
                private_signal.0.clone() * lhs_plaintext.clone(),
            public_action_for_private.1 * lhs_plaintext.clone() +
                private_rhs_carrier.1 * -lhs_plaintext.clone() +
                private_signal.1.clone() * lhs_plaintext.clone(),
        );
        let private_residual = (
            private_result.row.0.clone() -
                mask.0.clone() * expected_private_key.0.0 -
                mask.1.clone() * expected_private_key.1.0 +
                private_expected_signal.0.clone(),
            private_result.row.1.clone() -
                mask.0.clone() * expected_private_key.0.1 -
                mask.1.clone() * expected_private_key.1.1 +
                private_expected_signal.1,
        );
        let public_action = direct_row_action(public_signal.clone(), public_rhs_key.clone());
        let public_rhs_carrier = mask_of(&public_rhs_key);
        let public_expected_signal = (
            public_action.0 * lhs_plaintext.clone() - public_rhs_carrier.0 * lhs_plaintext.clone() +
                public_signal.0.clone() * lhs_plaintext * public_rhs_plaintext.clone(),
            public_action.1 * ring.polynomial([IntExpr::constant(2)]) -
                public_rhs_carrier.1 * ring.polynomial([IntExpr::constant(2)]) +
                public_signal.1 * ring.polynomial([IntExpr::constant(2)]) * public_rhs_plaintext,
        );
        let public_residual = (
            public_result.row.0.clone() -
                mask.0.clone() * expected_public_key.0.0 -
                mask.1.clone() * expected_public_key.1.0 +
                public_expected_signal.0,
            public_result.row.1.clone() -
                mask.0.clone() * expected_public_key.0.1 -
                mask.1.clone() * expected_public_key.1.1 +
                public_expected_signal.1,
        );
        let wrong_rhs_key = (
            (private_entry_alt.clone(), private_entry_alt.clone()),
            (private_entry_alt.clone(), private_entry_alt),
        );
        let wrong_key = direct_action(lhs_key, direct_decompose(wrong_rhs_key));
        let wrong_residual = private_result.row.0 -
            mask.0.clone() * wrong_key.0.0 -
            mask.1.clone() * wrong_key.1.0 +
            private_expected_signal.0.clone();
        let graph = DslContext::new("mul-public-encoding-residual-oracle")
            .output("private-residual-0", private_residual.0)
            .unwrap()
            .output("private-residual-1", private_residual.1)
            .unwrap()
            .output("public-residual-0", public_residual.0)
            .unwrap()
            .output("public-residual-1", public_residual.1)
            .unwrap()
            .output("wrong-residual", wrong_residual)
            .unwrap()
            .build()
            .unwrap();
        let mut execution = mxx_backends::authority::CpuExecution::new(cpu_backend([level]));
        let mut store = MemoryArtifactStore::default();
        let result = execute_graph(
            StageGraph::new(graph, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            &mut execution,
            &mut store,
            |_| Ok(()),
        )
        .unwrap();
        for name in
            ["private-residual-0", "private-residual-1", "public-residual-0", "public-residual-1"]
        {
            let matrix = crate::test_support::full(&result.outputs[name]);
            assert!(
                matrix.entry(0, 0).coeffs_biguints().iter().all(|value| value.is_zero()),
                "{name} residual is nonzero: {:?}",
                matrix.entry(0, 0).coeffs_biguints()
            );
        }
        let wrong = crate::test_support::full(&result.outputs["wrong-residual"]);
        assert!(wrong.entry(0, 0).coeffs_biguints().iter().any(|value| !value.is_zero()));
    }

    #[test]
    fn coefficient_finish_outputs_finalizes_the_paired_key_and_row_once() {
        let high = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let low = high.select_modulus(&BigUint::from(high.to_crt().0[0])).unwrap();
        let parameters = RnsBggParameters::new(vec![high, low], 1, 1).unwrap();
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PrivatePayload).unwrap();
        let table = LutTable::unary(
            2,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Coefficient,
        )
        .unwrap();
        let lut = builder.register_lut(table).unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = Arc::new(builder.finish(vec![Operand::Scalar(output)]).unwrap());
        let workload = Arc::new(RnsBggDerivedWorkload::from_program(&program).unwrap());
        let production = ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([47; 32]),
            execution_nonce: [53; 32],
        };
        let mut lowerer = EncodingLowering {
            parameters: &parameters,
            hash_key: parameters.ring(0).unwrap().bytes_input(super::HASH_KEY_INPUT, 32),
            program,
            workload,
            preprocess: production.clone(),
            encryption: production,
            manifests: BTreeMap::new(),
            refresh_prf_namespace: None,
            current_site: Some(LoweringSite { output, indices: Vec::new(), ranges: Vec::new() }),
        };
        assert!(lowerer.program.wire_is_coefficient_output(output).unwrap());
        let source_ring = parameters.ring(0).unwrap();
        let columns = parameters.component_columns(0).unwrap();
        let value = source_ring.input("coefficient-finish", (1, columns));
        let key = RnsBggPublicKey::new(
            output,
            ((value.clone(), value.clone()), (value.clone(), value.clone())),
        );
        let encoding = RnsBggEncoding::new(key, (value.clone(), value), None);
        let finalized = lowerer.finish_outputs(vec![encoding]).unwrap().pop().unwrap();
        use crate::test_support::modulus_of;
        let target_modulus = BigInt::from(parameters.level(1).unwrap().modulus().as_ref().clone());
        assert_eq!(modulus_of(finalized.row.0.matrix_type()), target_modulus);
        assert_eq!(modulus_of(finalized.row.1.matrix_type()), target_modulus);
        for entry in [
            &finalized.public_key.matrix.0.0,
            &finalized.public_key.matrix.0.1,
            &finalized.public_key.matrix.1.0,
            &finalized.public_key.matrix.1.1,
        ] {
            assert_eq!(modulus_of(entry.matrix_type()), target_modulus);
        }
    }

    #[test]
    fn public_lut_input_is_rejected_before_payload_propagation() {
        let mut builder = crate::program::ExponentLutProgramBuilder::new();
        let input = builder.input(AttributeKnowledge::PublicPayload).unwrap();
        let table = LutTable::unary(
            2,
            ExpLayout { stride_log2: 0, width_log2: 1 },
            2,
            vec![0, 1],
            crate::program::LutOutputRepresentation::Coefficient,
        )
        .unwrap();
        let lut = builder.register_lut(table.clone()).unwrap();
        let output = builder.lut(Operand::Scalar(input), lut).unwrap();
        let program = builder.finish(vec![Operand::Scalar(output)]).unwrap();
        assert!(RnsBggDerivedWorkload::from_program(&program).is_err());
    }

    #[test]
    fn verification_residual_uses_the_paired_public_key() {
        let level = DCRTPolyParams::new(16, 2, 17, 4, None, None);
        let parameters = RnsBggParameters::new(vec![level], 1, 1).unwrap();
        let ring = parameters.ring(0).unwrap();
        let a = ring.input("verification-paired-key-a", (1, 1));
        let b = ring.input("verification-paired-key-b", (1, 1));
        let key_a =
            RnsBggPublicKey::new(WireId(0), ((a.clone(), a.clone()), (a.clone(), a.clone())));
        let key_b =
            RnsBggPublicKey::new(WireId(0), ((b.clone(), b.clone()), (b.clone(), b.clone())));
        let row = (
            ring.input("verification-paired-row-0", (1, 1)),
            ring.input("verification-paired-row-1", (1, 1)),
        );
        let payload =
            RnsBggPayloadSecret { row: (ring.identity(1), ring.identity(1)), binding: None };
        let mask = RnsBggMaskSecret { row: (ring.identity(1), ring.identity(1)), binding: None };
        let expected = ring.polynomial([IntExpr::constant(BigInt::from(1u8))]);
        let residual_a = RnsBggEncoding::new(key_a, row.clone(), None)
            .verification_residual(&parameters, 0, &payload, &mask, expected.clone())
            .unwrap();
        let residual_b = RnsBggEncoding::new(key_b, row, None)
            .verification_residual(&parameters, 0, &payload, &mask, expected)
            .unwrap();
        assert_ne!(residual_a.flatten(), residual_b.flatten());
    }

    #[test]
    fn refresh_two_prime_encoding_channels_mod_down_then_lift_to_combined_ring() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let source = explicit_ring(&primes, 4);
        let producer = explicit_ring(&[primes[0], primes[2]], 4);
        let combined_modulus = BigInt::from(primes[0]) * primes[1];
        let value = source.input("refresh-source", (1, 1));
        let row = (value.clone(), value);
        let producer_row = mod_down_encoding(row, &producer);
        assert_eq!(modulus_of(producer_row.0.matrix_type()), BigInt::from(primes[0]) * primes[2]);
        let raw_reduction = checked_modulus_convert(producer_row.0.clone(), &producer).unwrap();
        assert_eq!(producer_row.0.flatten(), raw_reduction.flatten());
        let lifted = mod_down_lift_encoding(
            producer_row,
            &explicit_ring(&primes[..1], 4),
            &explicit_ring(&primes[..2], 4),
        );
        assert_eq!(modulus_of(lifted.0.matrix_type()), combined_modulus);
        assert_eq!(modulus_of(lifted.1.matrix_type()), combined_modulus);
    }

    #[test]
    fn refresh_target_uses_only_anchor_and_lifts_from_qt_to_combined_ring() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let ring = explicit_ring(&primes, 4);
        let channel = explicit_ring(&primes[1..2], 4);
        let combined = explicit_ring(&primes[..2], 4);
        let zero = ring.zero((1, 1));
        let zero_matrix = ((zero.clone(), zero.clone()), (zero.clone(), zero.clone()));
        let anchor =
            ((ring.input("refresh-a-star", (1, 1)), zero.clone()), (zero.clone(), zero.clone()));
        let zero_target = reduce_anchor_channel(zero_matrix, &channel, &combined).unwrap();
        let anchored_target = reduce_anchor_channel(anchor, &channel, &combined).unwrap();
        assert_ne!(zero_target.flatten(), anchored_target.flatten());
        assert_eq!(
            modulus_of(anchored_target.0.0.matrix_type()),
            BigInt::from(primes[0]) * primes[1]
        );
    }

    #[test]
    fn refresh_decoder_uses_p_ring_target_without_an_extra_moddown() {
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let p_modulus = BigInt::from(primes[0]) * primes[2];
        let p_ring = explicit_ring(&[primes[0], primes[2]], 4);
        let sum = p_ring.input("refresh-a-sum-p", (1, 1));
        let zero = p_ring.zero((1, 1));
        let a_sum = ((sum.clone(), zero.clone()), (zero.clone(), zero.clone()));
        let a_star =
            ((p_ring.input("refresh-a-star-p", (1, 1)), zero.clone()), (zero.clone(), zero));
        let h = p_ring.polynomial([IntExpr::constant(BigInt::from(21u8))]);
        let p_target = (
            (a_sum.0.0 - h.clone() * a_star.0.0, a_sum.0.1 - h.clone() * a_star.0.1),
            (a_sum.1.0 - h.clone() * a_star.1.0, a_sum.1.1 - h * a_star.1.1),
        );
        assert_eq!(modulus_of(p_target.0.0.matrix_type()), p_modulus);
    }

    #[test]
    fn refresh_affine_offsets_cover_components_coefficients_and_b2_d1() {
        assert_eq!(refresh_digit_sum(2, 1), BigInt::from(1u8));
        assert_eq!(refresh_digit_sum(2, 2), BigInt::from(3u8));
        use crate::test_support::{explicit_ring, modulus_of, primes};
        let primes = primes(4, 3, 17);
        let ring = explicit_ring(&[primes[0], primes[2]], 4);
        let columns = 2;
        let mask = refresh_affine_route(&ring, columns, 0, 0, BigInt::from(-2i8)).unwrap();
        let fresh = refresh_affine_route(&ring, columns, 3, 1, BigInt::from(-5i8)).unwrap();
        let offset = add_encoding_rows(mask, fresh);
        assert_eq!(modulus_of(offset.0.matrix_type()), BigInt::from(primes[0]) * primes[2]);
        assert_ne!(offset.flatten(), ring.zero((1, columns)).flatten());
        let secret0 = ring.input("refresh-secret-0", (1, columns));
        let secret1 = ring.input("refresh-secret-1", (1, columns));
        let decoder0 = secret0.clone() - offset.0.clone();
        let decoder1 = secret1.clone() - offset.0.clone();
        assert_ne!(decoder0.flatten(), decoder1.flatten());
    }
}
