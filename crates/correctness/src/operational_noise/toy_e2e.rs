//! Fixed Rust input and in-memory adapter for the singleton-preimage Gaussian toy slice.
//!
//! The adapter intentionally has no generic proof language.  Its deterministic serialization is
//! test/audit output only; the fixed source remains the sole generator input.
//! It retains exactly the variants reached by this one honest run and rejects any other variant.

mod lean;

use super::{
    OperationalCheckRequest,
    bound::MatrixProductFacts,
    certificate_schema::CertificateDocumentV1,
    simulation::{
        ProofPayloadAuthority, ProofPayloadCoefficientMerge, ProofPayloadCoefficientMergeSource,
        ProofPayloadEvent, ProofPayloadFactorEvidence, ProofPayloadMonomial, ProofPayloadOwner,
        ProofPayloadPreFoldPolynomial, ProofPayloadRange, ProofPayloadRelationRule,
        ProofPayloadRule, ProofPayloadScale, ProofPayloadSurvivorFold, ProofPayloadValue,
        ProofPayloadValueRef, derive_certificate_documents, prepare_operational_certificate,
    },
};
use serde::{Deserialize, Serialize, Serializer};

const TOY_SOURCE_SCHEMA_ID: &str = "mxx.operational-noise.toy-source";
const TOY_SOURCE_SCHEMA_VERSION: u32 = 1;
const TOY_ABI: &str = "singleton-preimage-gaussian-v1";
const RUST_PROJECTION_VERSION: &str = "operational-noise-certificate-v1";
const LEAN_ABI_VERSION: &str = "toy-replay-v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ToySourceV1 {
    schema_id: String,
    schema_version: u32,
    abi: String,
    rust_projection_version: String,
    lean_abi_version: String,
    request: ToyRequestV1,
    parameters: ToyParametersV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ToyRequestV1 {
    target_id: String,
    environment: Vec<serde_json::Value>,
    layouts: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ToyParametersV1 {
    plaintext_modulus: String,
    ciphertext_modulus: String,
    ring_dimension: u64,
    trapdoor_rows: String,
    trapdoor_sigma: String,
    gadget_base: String,
    digit_count: String,
    preimage_maximum_absolute_coefficient: String,
    gaussian_sigma: String,
    gaussian_maximum_absolute_coefficient: String,
}

impl ToySourceV1 {
    fn expected() -> Self {
        Self {
            schema_id: TOY_SOURCE_SCHEMA_ID.to_owned(),
            schema_version: TOY_SOURCE_SCHEMA_VERSION,
            abi: TOY_ABI.to_owned(),
            rust_projection_version: RUST_PROJECTION_VERSION.to_owned(),
            lean_abi_version: LEAN_ABI_VERSION.to_owned(),
            request: ToyRequestV1 {
                target_id: "singleton-preimage-gaussian".to_owned(),
                environment: Vec::new(),
                layouts: Vec::new(),
            },
            parameters: ToyParametersV1 {
                plaintext_modulus: "2".to_owned(),
                ciphertext_modulus: "257".to_owned(),
                ring_dimension: 1,
                trapdoor_rows: "1".to_owned(),
                trapdoor_sigma: "3".to_owned(),
                gadget_base: "4".to_owned(),
                digit_count: "2".to_owned(),
                preimage_maximum_absolute_coefficient: "8".to_owned(),
                gaussian_sigma: "1".to_owned(),
                gaussian_maximum_absolute_coefficient: "1".to_owned(),
            },
        }
    }

    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let source = serde_json::from_slice::<Self>(bytes)
            .map_err(|error| format!("invalid toy source JSON: {error}"))?;
        if source != Self::expected() {
            return Err("toy source does not match the fixed singleton-preimage Gaussian ABI".into());
        }
        Ok(source)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ToySliceV1 {
    statement: CertificateDocumentV1,
    events: Vec<ToyEventV1>,
}

/// Deterministic, filesystem-free Lean output for the fixed toy vertical slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToyGeneratedLean {
    pub cert: Vec<u8>,
    pub proof: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyAuthorityV1 {
    Operator,
    RelationPreimageSource { source: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyRuleV1 {
    Authority(ToyAuthorityV1),
    Sum { inputs: Vec<ProofPayloadValueRef> },
    Scale { value: ProofPayloadValueRef, scale: ProofPayloadScale },
    MonomialProduct { monomial: ProofPayloadMonomial, factors: Vec<ProofPayloadFactorEvidence> },
    Product { left: ProofPayloadValueRef, right: ProofPayloadValueRef, facts: MatrixProductFacts },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyMergeSourceV1 {
    Operator { inputs: [super::simulation::ProofPayloadTermRef; 2] },
    Relation { application: u64, source_term_ordinal: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ToyMergeV1 {
    owner: ProofPayloadOwner,
    source: ToyMergeSourceV1,
    output: ProofPayloadMonomial,
    signed_contribution: num_bigint::BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ToyEventV1 {
    InvocationStart {
        root: ProofPayloadOwner,
    },
    Predecessor {
        consumer: ProofPayloadOwner,
        input_position: u32,
        predecessor: u64,
        source_result: u64,
    },
    Result {
        owner: ProofPayloadOwner,
        value: ProofPayloadValue,
    },
    InvocationEnd {
        root: ProofPayloadOwner,
        result: ProofPayloadValue,
    },
    SpecializationComputed {
        owner: ProofPayloadOwner,
        dispatch: super::simulation::ProofPayloadUniversalDispatch,
        source: ProofPayloadRange,
    },
    AppliedUniversal {
        owner: ProofPayloadOwner,
        source_monomial: ProofPayloadMonomial,
        outer_coefficient: num_bigint::BigInt,
        ordered_start: u32,
        ordered_end_exclusive: u32,
        computed: u64,
        lhs: ProofPayloadMonomial,
        lhs_layout: Option<super::arena::MatrixLayout>,
        rhs_result: u64,
    },
    BoundTransfer {
        owner: ProofPayloadOwner,
        rule: ToyRuleV1,
    },
    CoefficientMerge(ToyMergeV1),
    PreFoldPolynomial(ProofPayloadPreFoldPolynomial),
    SurvivorFold(ProofPayloadSurvivorFold),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditSlice<'a> {
    statement: &'a CertificateDocumentV1,
    events: Vec<ToyAuditEvent>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditScope {
    Closed { root_expression_row: u64 },
    Program { program_row: u64 },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditOwner {
    scope: ToyAuditScope,
    expression_row: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditMonomial {
    central_factors: Vec<ToyAuditOwner>,
    ordered_factors: Vec<ToyAuditOwner>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditTerm {
    monomial: ToyAuditMonomial,
    coefficient: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditBound {
    ExactZero,
    Finite { maximum_absolute_coefficient: String },
    Large,
    Missing,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditValue {
    Exact { terms: Vec<ToyAuditTerm>, summary: ToyAuditBound },
    Coefficient { bound: ToyAuditBound },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum ToyAuditProjection {
    Coefficient,
    Summary,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditValueRef {
    Predecessor { input_position: u32, projection: ToyAuditProjection },
    Result { event: u64, projection: ToyAuditProjection },
    Transfer { event: u64 },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditScale {
    Value { value: ToyAuditValueRef },
    Magnitude { magnitude: String },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditFactorEvidence {
    bound: ToyAuditValueRef,
    is_constant_polynomial: bool,
    support_upper: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditProductFacts {
    left_is_constant_polynomial: bool,
    right_is_constant_polynomial: bool,
    right_known_zero_rows: Option<String>,
    left_support_upper: Option<usize>,
    right_support_upper: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditAuthority {
    Operator,
    RelationPreimageSource { source: u64 },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditRule {
    Authority { authority: ToyAuditAuthority },
    Sum { inputs: Vec<ToyAuditValueRef> },
    Scale { value: ToyAuditValueRef, scale: ToyAuditScale },
    MonomialProduct { monomial: ToyAuditMonomial, factors: Vec<ToyAuditFactorEvidence> },
    Product { left: ToyAuditValueRef, right: ToyAuditValueRef, facts: ToyAuditProductFacts },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditDispatch {
    preimage_family: u64,
    preimage_source: u64,
    trapdoor_source: u64,
}

#[derive(Serialize)]
struct ToyAuditRange {
    start: u64,
    end: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditLayout {
    name: String,
    row_stride: usize,
    column_stride: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditTermRef {
    value_event: u64,
    term_ordinal: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditMergeSource {
    Operator { inputs: [ToyAuditTermRef; 2] },
    Relation { application: u64, source_term_ordinal: u64 },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ToyAuditMerge {
    owner: ToyAuditOwner,
    source: ToyAuditMergeSource,
    output: ToyAuditMonomial,
    signed_contribution: String,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", rename_all_fields = "camelCase", tag = "kind")]
enum ToyAuditEvent {
    InvocationStart {
        root: ToyAuditOwner,
    },
    Predecessor {
        consumer: ToyAuditOwner,
        input_position: u32,
        predecessor: u64,
        source_result: u64,
    },
    Result {
        owner: ToyAuditOwner,
        value: ToyAuditValue,
    },
    InvocationEnd {
        root: ToyAuditOwner,
        result: ToyAuditValue,
    },
    SpecializationComputed {
        owner: ToyAuditOwner,
        dispatch: ToyAuditDispatch,
        source: ToyAuditRange,
    },
    AppliedUniversal {
        owner: ToyAuditOwner,
        source_monomial: ToyAuditMonomial,
        outer_coefficient: String,
        ordered_start: u32,
        ordered_end_exclusive: u32,
        computed: u64,
        lhs: ToyAuditMonomial,
        lhs_layout: Option<ToyAuditLayout>,
        rhs_result: u64,
    },
    BoundTransfer {
        owner: ToyAuditOwner,
        rule: ToyAuditRule,
    },
    CoefficientMerge {
        merge: ToyAuditMerge,
    },
    PreFoldPolynomial {
        terms: Vec<ToyAuditTerm>,
        summary: ToyAuditBound,
        summary_evidence: Option<ToyAuditValueRef>,
    },
    SurvivorFold {
        coefficient: String,
        bound: u64,
    },
}

fn audit_owner(owner: &ProofPayloadOwner) -> ToyAuditOwner {
    let scope = match owner.scope {
        super::simulation::ProofPayloadScope::Closed { root_expression_row } => {
            ToyAuditScope::Closed { root_expression_row }
        }
        super::simulation::ProofPayloadScope::Program { program_row } => {
            ToyAuditScope::Program { program_row }
        }
    };
    ToyAuditOwner { scope, expression_row: owner.expression_row }
}

fn audit_monomial(monomial: &ProofPayloadMonomial) -> ToyAuditMonomial {
    ToyAuditMonomial {
        central_factors: monomial.central_factors.iter().map(audit_owner).collect(),
        ordered_factors: monomial.ordered_factors.iter().map(audit_owner).collect(),
    }
}

fn audit_term(term: &super::simulation::ProofPayloadTerm) -> ToyAuditTerm {
    ToyAuditTerm {
        monomial: audit_monomial(&term.monomial),
        coefficient: term.coefficient.to_string(),
    }
}

fn audit_coefficient_bound(
    bound: &super::facts::NumericContract<super::facts::CoefficientBound>,
) -> ToyAuditBound {
    match bound {
        super::facts::NumericContract::Missing => ToyAuditBound::Missing,
        super::facts::NumericContract::Known(super::facts::CoefficientBound::ExactZero) => {
            ToyAuditBound::ExactZero
        }
        super::facts::NumericContract::Known(super::facts::CoefficientBound::Finite(bound)) => {
            ToyAuditBound::Finite {
                maximum_absolute_coefficient: bound.maximum_absolute_coefficient.to_string(),
            }
        }
        super::facts::NumericContract::Known(super::facts::CoefficientBound::Large) => {
            ToyAuditBound::Large
        }
    }
}

fn audit_summary(summary: &super::normal_form::BoundedSummary) -> ToyAuditBound {
    audit_coefficient_bound(&summary.coefficient_bound())
}

fn audit_value(value: &ProofPayloadValue) -> ToyAuditValue {
    match value {
        ProofPayloadValue::Exact { terms, summary } => ToyAuditValue::Exact {
            terms: terms.iter().map(audit_term).collect(),
            summary: audit_summary(summary),
        },
        ProofPayloadValue::Coefficient { bound } => {
            ToyAuditValue::Coefficient { bound: audit_coefficient_bound(bound) }
        }
    }
}

fn audit_projection(projection: &super::g0::BoundProjection) -> ToyAuditProjection {
    match projection {
        super::g0::BoundProjection::Coefficient => ToyAuditProjection::Coefficient,
        super::g0::BoundProjection::Summary => ToyAuditProjection::Summary,
    }
}

fn audit_value_ref(value: &ProofPayloadValueRef) -> ToyAuditValueRef {
    match value {
        ProofPayloadValueRef::Predecessor { input_position, projection } => {
            ToyAuditValueRef::Predecessor {
                input_position: *input_position,
                projection: audit_projection(projection),
            }
        }
        ProofPayloadValueRef::Result { event, projection } => {
            ToyAuditValueRef::Result { event: *event, projection: audit_projection(projection) }
        }
        ProofPayloadValueRef::Transfer(event) => ToyAuditValueRef::Transfer { event: *event },
    }
}

fn audit_scale(scale: &ProofPayloadScale) -> ToyAuditScale {
    match scale {
        ProofPayloadScale::Value(value) => ToyAuditScale::Value { value: audit_value_ref(value) },
        ProofPayloadScale::Magnitude(magnitude) => {
            ToyAuditScale::Magnitude { magnitude: magnitude.to_string() }
        }
    }
}

fn audit_rule(rule: &ToyRuleV1) -> ToyAuditRule {
    match rule {
        ToyRuleV1::Authority(authority) => ToyAuditRule::Authority {
            authority: match authority {
                ToyAuthorityV1::Operator => ToyAuditAuthority::Operator,
                ToyAuthorityV1::RelationPreimageSource { source } => {
                    ToyAuditAuthority::RelationPreimageSource { source: *source }
                }
            },
        },
        ToyRuleV1::Sum { inputs } => {
            ToyAuditRule::Sum { inputs: inputs.iter().map(audit_value_ref).collect() }
        }
        ToyRuleV1::Scale { value, scale } => {
            ToyAuditRule::Scale { value: audit_value_ref(value), scale: audit_scale(scale) }
        }
        ToyRuleV1::MonomialProduct { monomial, factors } => ToyAuditRule::MonomialProduct {
            monomial: audit_monomial(monomial),
            factors: factors
                .iter()
                .map(|factor| ToyAuditFactorEvidence {
                    bound: audit_value_ref(&factor.bound),
                    is_constant_polynomial: factor.is_constant_polynomial,
                    support_upper: factor.support_upper,
                })
                .collect(),
        },
        ToyRuleV1::Product { left, right, facts } => ToyAuditRule::Product {
            left: audit_value_ref(left),
            right: audit_value_ref(right),
            facts: ToyAuditProductFacts {
                left_is_constant_polynomial: facts.left_is_constant_polynomial,
                right_is_constant_polynomial: facts.right_is_constant_polynomial,
                right_known_zero_rows: facts
                    .right_known_zero_rows
                    .as_ref()
                    .map(ToString::to_string),
                left_support_upper: facts.left_support_upper,
                right_support_upper: facts.right_support_upper,
            },
        },
    }
}

fn audit_merge(merge: &ToyMergeV1) -> ToyAuditMerge {
    let source = match &merge.source {
        ToyMergeSourceV1::Operator { inputs } => ToyAuditMergeSource::Operator {
            inputs: inputs.each_ref().map(|input| ToyAuditTermRef {
                value_event: input.value_event,
                term_ordinal: input.term_ordinal,
            }),
        },
        ToyMergeSourceV1::Relation { application, source_term_ordinal } => {
            ToyAuditMergeSource::Relation {
                application: *application,
                source_term_ordinal: *source_term_ordinal,
            }
        }
    };
    ToyAuditMerge {
        owner: audit_owner(&merge.owner),
        source,
        output: audit_monomial(&merge.output),
        signed_contribution: merge.signed_contribution.to_string(),
    }
}

fn audit_event(event: &ToyEventV1) -> ToyAuditEvent {
    match event {
        ToyEventV1::InvocationStart { root } => {
            ToyAuditEvent::InvocationStart { root: audit_owner(root) }
        }
        ToyEventV1::Predecessor { consumer, input_position, predecessor, source_result } => {
            ToyAuditEvent::Predecessor {
                consumer: audit_owner(consumer),
                input_position: *input_position,
                predecessor: *predecessor,
                source_result: *source_result,
            }
        }
        ToyEventV1::Result { owner, value } => {
            ToyAuditEvent::Result { owner: audit_owner(owner), value: audit_value(value) }
        }
        ToyEventV1::InvocationEnd { root, result } => {
            ToyAuditEvent::InvocationEnd { root: audit_owner(root), result: audit_value(result) }
        }
        ToyEventV1::SpecializationComputed { owner, dispatch, source } => {
            ToyAuditEvent::SpecializationComputed {
                owner: audit_owner(owner),
                dispatch: ToyAuditDispatch {
                    preimage_family: dispatch.preimage_family,
                    preimage_source: dispatch.preimage_source,
                    trapdoor_source: dispatch.trapdoor_source,
                },
                source: ToyAuditRange { start: source.start, end: source.end },
            }
        }
        ToyEventV1::AppliedUniversal {
            owner,
            source_monomial,
            outer_coefficient,
            ordered_start,
            ordered_end_exclusive,
            computed,
            lhs,
            lhs_layout,
            rhs_result,
        } => ToyAuditEvent::AppliedUniversal {
            owner: audit_owner(owner),
            source_monomial: audit_monomial(source_monomial),
            outer_coefficient: outer_coefficient.to_string(),
            ordered_start: *ordered_start,
            ordered_end_exclusive: *ordered_end_exclusive,
            computed: *computed,
            lhs: audit_monomial(lhs),
            lhs_layout: lhs_layout.as_ref().map(|layout| ToyAuditLayout {
                name: layout.name.clone(),
                row_stride: layout.row_stride,
                column_stride: layout.column_stride,
            }),
            rhs_result: *rhs_result,
        },
        ToyEventV1::BoundTransfer { owner, rule } => {
            ToyAuditEvent::BoundTransfer { owner: audit_owner(owner), rule: audit_rule(rule) }
        }
        ToyEventV1::CoefficientMerge(merge) => {
            ToyAuditEvent::CoefficientMerge { merge: audit_merge(merge) }
        }
        ToyEventV1::PreFoldPolynomial(value) => ToyAuditEvent::PreFoldPolynomial {
            terms: value.terms.iter().map(audit_term).collect(),
            summary: audit_summary(&value.summary),
            summary_evidence: value.summary_evidence.as_ref().map(audit_value_ref),
        },
        ToyEventV1::SurvivorFold(value) => ToyAuditEvent::SurvivorFold {
            coefficient: value.coefficient.to_string(),
            bound: value.bound,
        },
    }
}

impl Serialize for ToySliceV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ToyAuditSlice {
            statement: &self.statement,
            events: self.events.iter().map(audit_event).collect(),
        }
        .serialize(serializer)
    }
}

impl ToySliceV1 {
    fn encode_audit_pretty(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec_pretty(self)
            .map_err(|error| format!("toy projected-slice encoding failed: {error}"))
    }
}

fn toy_rule(rule: ProofPayloadRule) -> Result<ToyRuleV1, &'static str> {
    Ok(match rule {
        ProofPayloadRule::Authority(ProofPayloadAuthority::Operator) => {
            ToyRuleV1::Authority(ToyAuthorityV1::Operator)
        }
        ProofPayloadRule::Authority(ProofPayloadAuthority::RelationPreimageSource { source }) => {
            ToyRuleV1::Authority(ToyAuthorityV1::RelationPreimageSource { source })
        }
        ProofPayloadRule::Sum { inputs } => ToyRuleV1::Sum { inputs },
        ProofPayloadRule::Scale { value, scale } => ToyRuleV1::Scale { value, scale },
        ProofPayloadRule::MonomialProduct { monomial, factors } => {
            ToyRuleV1::MonomialProduct { monomial, factors }
        }
        ProofPayloadRule::Product { left, right, facts } => {
            ToyRuleV1::Product { left, right, facts }
        }
        _ => return Err("unsupported bound rule in fixed toy slice"),
    })
}

fn toy_merge(merge: ProofPayloadCoefficientMerge) -> ToyMergeV1 {
    let source = match merge.source {
        ProofPayloadCoefficientMergeSource::Operator { inputs } => {
            ToyMergeSourceV1::Operator { inputs }
        }
        ProofPayloadCoefficientMergeSource::Relation { application, source_term_ordinal } => {
            ToyMergeSourceV1::Relation { application, source_term_ordinal }
        }
    };
    ToyMergeV1 {
        owner: merge.owner,
        source,
        output: merge.output,
        signed_contribution: merge.signed_contribution,
    }
}

fn toy_event(event: ProofPayloadEvent) -> Result<ToyEventV1, &'static str> {
    Ok(match event {
        ProofPayloadEvent::InvocationStart { root } => ToyEventV1::InvocationStart { root },
        ProofPayloadEvent::Predecessor { consumer, input_position, predecessor, source_result } => {
            ToyEventV1::Predecessor { consumer, input_position, predecessor, source_result }
        }
        ProofPayloadEvent::Result { owner, value } => ToyEventV1::Result { owner, value },
        ProofPayloadEvent::InvocationEnd { root, result } => {
            ToyEventV1::InvocationEnd { root, result }
        }
        ProofPayloadEvent::SpecializationComputed { owner, dispatch, source } => {
            ToyEventV1::SpecializationComputed { owner, dispatch, source }
        }
        ProofPayloadEvent::AppliedRelation {
            owner,
            source_monomial,
            outer_coefficient,
            ordered_start,
            ordered_end_exclusive,
            rule: ProofPayloadRelationRule::Universal { computed, lhs, lhs_layout, rhs_result },
        } => ToyEventV1::AppliedUniversal {
            owner,
            source_monomial,
            outer_coefficient,
            ordered_start,
            ordered_end_exclusive,
            computed,
            lhs,
            lhs_layout,
            rhs_result,
        },
        ProofPayloadEvent::BoundTransfer { owner, rule } => {
            ToyEventV1::BoundTransfer { owner, rule: toy_rule(rule)? }
        }
        ProofPayloadEvent::CoefficientMerge(merge) => {
            ToyEventV1::CoefficientMerge(toy_merge(merge))
        }
        ProofPayloadEvent::PreFoldPolynomial(value) => ToyEventV1::PreFoldPolynomial(value),
        ProofPayloadEvent::SurvivorFold(value) => ToyEventV1::SurvivorFold(value),
        _ => return Err("unsupported event in fixed toy slice"),
    })
}

fn prepare_toy_slice(source: ToySourceV1) -> Result<ToySliceV1, String> {
    let protocol = super::lower::singleton_preimage_protocol(Some(1_u8.into()));
    let request = OperationalCheckRequest {
        environment: Vec::new(),
        layouts: Vec::new(),
        target_id: source.request.target_id,
    };
    let run =
        prepare_operational_certificate(&protocol, &request).map_err(|error| error.to_string())?;
    let documents = derive_certificate_documents(&run).map_err(|error| error.to_string())?;
    let events = documents
        .proof
        .payload
        .events
        .into_iter()
        .map(toy_event)
        .collect::<Result<Vec<_>, _>>()
        .map_err(str::to_owned)?;
    Ok(ToySliceV1 { statement: documents.cert, events })
}

/// Runs the fixed opt-in Rust authority and verifies that it projects into the narrow toy slice.
pub fn check_toy_operational_slice_source(source_json: &[u8]) -> Result<(), String> {
    prepare_toy_slice(ToySourceV1::parse(source_json)?)?.encode_audit_pretty().map(|_| ())
}

/// Runs the fixed Rust authority once and renders the reached toy certificate and proof as Lean.
pub fn generate_toy_operational_slice_lean(source_json: &[u8]) -> Result<ToyGeneratedLean, String> {
    let source = ToySourceV1::parse(source_json)?;
    let slice = prepare_toy_slice(source.clone())?;
    lean::render(&source, &slice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_noise::{
        g0::{StableSamplerOperation, StableValueType},
        simulation::{ProofPayloadScope, ProofPayloadValueRef},
    };

    const SOURCE_BYTES: &[u8] =
        include_bytes!("../../testdata/operational-noise-toy-source-v1.json");

    fn slice() -> ToySliceV1 {
        prepare_toy_slice(ToySourceV1::parse(SOURCE_BYTES).expect("fixed source"))
            .expect("fixed toy slice")
    }

    #[test]
    fn fixed_source_is_strict_and_exact() {
        let source = ToySourceV1::parse(SOURCE_BYTES).expect("fixed source");
        assert_eq!(source, ToySourceV1::expected());

        let mut unknown = serde_json::to_value(&source).unwrap();
        unknown.as_object_mut().unwrap().insert("unknown".into(), true.into());
        assert!(ToySourceV1::parse(&serde_json::to_vec(&unknown).unwrap()).is_err());

        let mut missing = serde_json::to_value(&source).unwrap();
        missing.as_object_mut().unwrap().remove("abi");
        assert!(ToySourceV1::parse(&serde_json::to_vec(&missing).unwrap()).is_err());

        let mut mismatch = source;
        mismatch.parameters.gaussian_maximum_absolute_coefficient = "2".into();
        assert!(ToySourceV1::parse(&serde_json::to_vec(&mismatch).unwrap()).is_err());
    }

    #[test]
    fn projected_slice_audit_is_complete_deterministic_and_matches_golden() {
        let first = slice().encode_audit_pretty().expect("first projected slice");
        let second = slice().encode_audit_pretty().expect("second projected slice");
        assert_eq!(first, second);

        let document: serde_json::Value = serde_json::from_slice(&first).expect("audit JSON");
        let top = document.as_object().expect("audit document");
        assert_eq!(
            top.keys().map(String::as_str).collect::<std::collections::BTreeSet<_>>(),
            ["statement", "events"].into_iter().collect()
        );
        let events = document["events"].as_array().expect("audit events");
        assert_eq!(events.len(), 59);
        for event in events {
            let event = event.as_object().expect("typed audit event");
            let kind = event["kind"].as_str().expect("event kind");
            let expected: &[&str] = match kind {
                "invocation_start" => &["kind", "root"],
                "predecessor" => {
                    &["kind", "consumer", "inputPosition", "predecessor", "sourceResult"]
                }
                "result" => &["kind", "owner", "value"],
                "invocation_end" => &["kind", "root", "result"],
                "specialization_computed" => &["kind", "owner", "dispatch", "source"],
                "applied_universal" => &[
                    "kind",
                    "owner",
                    "sourceMonomial",
                    "outerCoefficient",
                    "orderedStart",
                    "orderedEndExclusive",
                    "computed",
                    "lhs",
                    "lhsLayout",
                    "rhsResult",
                ],
                "bound_transfer" => &["kind", "owner", "rule"],
                "coefficient_merge" => &["kind", "merge"],
                "pre_fold_polynomial" => &["kind", "terms", "summary", "summaryEvidence"],
                "survivor_fold" => &["kind", "coefficient", "bound"],
                other => panic!("unexpected reached event {other}"),
            };
            let actual =
                event.keys().map(String::as_str).collect::<std::collections::BTreeSet<_>>();
            let expected = expected.iter().copied().collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual, expected, "complete fields for {kind}");
        }
        let text = std::str::from_utf8(&first).expect("UTF-8 audit JSON");
        for forbidden in
            ["arena", "cache", "coverage", "metric", "canonicalPayload", "generatorPeak"]
        {
            assert!(!text.contains(forbidden), "audit output contains {forbidden}");
        }

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/operational-noise-toy-v1/projected-slice.json");
        if std::env::var_os("MXX_REGENERATE_CORRECTNESS").as_deref() ==
            Some(std::ffi::OsStr::new("1"))
        {
            std::fs::create_dir_all(path.parent().expect("golden parent"))
                .expect("create toy audit testdata");
            std::fs::write(&path, &first).expect("write toy projected-slice golden");
        }
        assert_eq!(first, std::fs::read(path).expect("read toy projected-slice golden"));
    }

    #[test]
    fn generated_lean_is_source_driven_deterministic_and_matches_committed_files() {
        let first = generate_toy_operational_slice_lean(SOURCE_BYTES).expect("first Lean output");
        let second = generate_toy_operational_slice_lean(SOURCE_BYTES).expect("second Lean output");
        assert_eq!(first, second);

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../lean/Mxx/Certificate/OperationalNoise/ToyGenerated");
        let cert = root.join("Cert.lean");
        let proof = root.join("Proof.lean");
        if std::env::var_os("MXX_REGENERATE_CORRECTNESS").as_deref() ==
            Some(std::ffi::OsStr::new("1"))
        {
            std::fs::create_dir_all(&root).expect("create generated Lean directory");
            std::fs::write(&cert, &first.cert).expect("write generated Cert.lean");
            std::fs::write(&proof, &first.proof).expect("write generated Proof.lean");
        }
        assert_eq!(first.cert, std::fs::read(cert).expect("read generated Cert.lean"));
        assert_eq!(first.proof, std::fs::read(proof).expect("read generated Proof.lean"));

        let cert = std::str::from_utf8(&first.cert).expect("UTF-8 Cert.lean");
        let proof = std::str::from_utf8(&first.proof).expect("UTF-8 Proof.lean");
        assert!(cert.contains("def source : ToySource"));
        assert!(cert.contains("def document : Document"));
        assert!(cert.contains("def rows : ToyRows"));
        assert!(proof.contains("def events : List ToyEvent"));
        assert!(proof.contains("proofValid : ToyValid source document rows events"));
    }

    #[test]
    fn statement_preserves_actual_rows_and_sampler_contract_authorities() {
        let slice = slice();
        assert_eq!(slice.statement.expressions.len(), 13);
        assert_eq!(slice.statement.programs.len(), 1);
        assert_eq!(slice.statement.sources.len(), 2);
        assert_eq!(slice.statement.events.len(), 4);
        assert_eq!(slice.statement.ring_dimension, 1);
        assert_eq!(slice.statement.plaintext_modulus, "2");
        assert_eq!(slice.statement.ciphertext_modulus, "257");
        assert!(matches!(
            &slice.statement.events[2],
            super::super::certificate_schema::CertificateEventRowV1::Sampler {
                operation: StableSamplerOperation::Preimage {
                    output: StableValueType::Matrix { rows: 4, columns: 1, .. },
                    max_coefficient_bound,
                },
                contract: None,
                ..
            } if max_coefficient_bound == "8"
        ));
        assert!(matches!(
            &slice.statement.events[3],
            super::super::certificate_schema::CertificateEventRowV1::Sampler {
                operation: StableSamplerOperation::Gaussian {
                    output: StableValueType::Matrix { rows: 1, columns: 1, .. },
                    max_coefficient_bound,
                    ..
                },
                contract: None,
                ..
            } if max_coefficient_bound == "1"
        ));
    }

    fn kind(event: &ToyEventV1) -> &'static str {
        match event {
            ToyEventV1::InvocationStart { .. } => "start",
            ToyEventV1::Predecessor { .. } => "predecessor",
            ToyEventV1::Result { .. } => "result",
            ToyEventV1::InvocationEnd { .. } => "end",
            ToyEventV1::SpecializationComputed { .. } => "specialization",
            ToyEventV1::AppliedUniversal { .. } => "universal",
            ToyEventV1::BoundTransfer { rule, .. } => match rule {
                ToyRuleV1::Authority(ToyAuthorityV1::Operator) => "bound.authority.operator",
                ToyRuleV1::Authority(ToyAuthorityV1::RelationPreimageSource { .. }) => {
                    "bound.authority.preimage"
                }
                ToyRuleV1::Sum { .. } => "bound.sum",
                ToyRuleV1::Scale { .. } => "bound.scale",
                ToyRuleV1::MonomialProduct { .. } => "bound.monomial_product",
                ToyRuleV1::Product { .. } => "bound.product",
            },
            ToyEventV1::CoefficientMerge(merge) => match merge.source {
                ToyMergeSourceV1::Operator { .. } => "merge.operator",
                ToyMergeSourceV1::Relation { .. } => "merge.relation",
            },
            ToyEventV1::PreFoldPolynomial { .. } => "prefold",
            ToyEventV1::SurvivorFold { .. } => "survivor",
        }
    }

    #[test]
    fn actual_inventory_is_the_fixed_59_event_sequence() {
        let slice = slice();
        let actual = slice.events.iter().map(kind).collect::<Vec<_>>();
        let expected = [
            "start",
            "bound.authority.operator",
            "result",
            "bound.authority.operator",
            "result",
            "bound.authority.operator",
            "result",
            "bound.authority.operator",
            "result",
            "predecessor",
            "bound.authority.preimage",
            "result",
            "predecessor",
            "predecessor",
            "bound.scale",
            "result",
            "bound.authority.operator",
            "result",
            "predecessor",
            "predecessor",
            "bound.product",
            "merge.operator",
            "start",
            "bound.authority.operator",
            "result",
            "predecessor",
            "bound.authority.preimage",
            "result",
            "bound.authority.operator",
            "result",
            "predecessor",
            "predecessor",
            "bound.product",
            "merge.operator",
            "result",
            "prefold",
            "end",
            "start",
            "bound.authority.operator",
            "result",
            "prefold",
            "end",
            "specialization",
            "universal",
            "merge.relation",
            "result",
            "predecessor",
            "predecessor",
            "bound.sum",
            "merge.operator",
            "result",
            "predecessor",
            "predecessor",
            "bound.sum",
            "bound.monomial_product",
            "survivor",
            "result",
            "prefold",
            "end",
        ];
        assert_eq!(actual, expected);
    }

    fn event_ref(value: &ProofPayloadValueRef, current: usize) {
        match value {
            ProofPayloadValueRef::Result { event, .. } | ProofPayloadValueRef::Transfer(event) => {
                assert!((*event as usize) < current)
            }
            ProofPayloadValueRef::Predecessor { .. } => {}
        }
    }

    #[test]
    fn adapter_preserves_chronology_relation_and_lifecycle() {
        let slice = slice();
        let root = match &slice.events[0] {
            ToyEventV1::InvocationStart { root } => *root,
            _ => panic!("root start"),
        };
        assert_eq!(root.expression_row, 12);
        assert_eq!(root.scope, ProofPayloadScope::Closed { root_expression_row: 12 });
        for (index, event) in slice.events.iter().enumerate() {
            match event {
                ToyEventV1::Predecessor { source_result, .. } => {
                    assert!((*source_result as usize) < index)
                }
                ToyEventV1::AppliedUniversal { computed, rhs_result, .. } => {
                    assert!((*computed as usize) < index);
                    assert!((*rhs_result as usize) < index);
                }
                ToyEventV1::BoundTransfer { rule, .. } => match rule {
                    ToyRuleV1::Sum { inputs } => {
                        inputs.iter().for_each(|value| event_ref(value, index));
                    }
                    ToyRuleV1::Scale { value, scale } => {
                        event_ref(value, index);
                        if let ProofPayloadScale::Value(value) = scale {
                            event_ref(value, index);
                        }
                    }
                    ToyRuleV1::MonomialProduct { factors, .. } => {
                        factors.iter().for_each(|factor| event_ref(&factor.bound, index))
                    }
                    ToyRuleV1::Product { left, right, .. } => {
                        event_ref(left, index);
                        event_ref(right, index);
                    }
                    ToyRuleV1::Authority(_) => {}
                },
                ToyEventV1::CoefficientMerge(merge) => match &merge.source {
                    ToyMergeSourceV1::Operator { inputs } => inputs.iter().for_each(|input| {
                        assert!((input.value_event as usize) < index);
                    }),
                    ToyMergeSourceV1::Relation { application, .. } => {
                        assert!((*application as usize) < index)
                    }
                },
                ToyEventV1::PreFoldPolynomial(value) => {
                    if let Some(value) = &value.summary_evidence {
                        event_ref(value, index);
                    }
                }
                ToyEventV1::SurvivorFold(value) => assert!((value.bound as usize) < index),
                _ => {}
            }
        }
        assert!(matches!(
            &slice.events[42],
            ToyEventV1::SpecializationComputed {
                dispatch: super::super::simulation::ProofPayloadUniversalDispatch {
                    preimage_family: 0,
                    preimage_source: 2,
                    trapdoor_source: 4,
                },
                source: ProofPayloadRange { start: 22, end: 42 },
                ..
            }
        ));
        let ToyEventV1::AppliedUniversal {
            source_monomial,
            lhs,
            outer_coefficient,
            computed: 42,
            rhs_result: 41,
            lhs_layout: None,
            ordered_start: 0,
            ordered_end_exclusive: 2,
            ..
        } = &slice.events[43]
        else {
            panic!("fixed universal relation")
        };
        assert_eq!(source_monomial, lhs);
        assert_eq!(outer_coefficient, &num_bigint::BigInt::from(1));
        assert!(matches!(
            &slice.events[20],
            ToyEventV1::BoundTransfer {
                rule: ToyRuleV1::Product {
                    facts: MatrixProductFacts {
                        left_is_constant_polynomial: false,
                        right_is_constant_polynomial: false,
                        right_known_zero_rows: None,
                        left_support_upper: None,
                        right_support_upper: None,
                    },
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            &slice.events[44],
            ToyEventV1::CoefficientMerge(ToyMergeV1 {
                source: ToyMergeSourceV1::Relation {
                    application: 43,
                    source_term_ordinal: 0,
                },
                signed_contribution,
                ..
            }) if signed_contribution == &num_bigint::BigInt::from(1)
        ));
        assert!(matches!(
            slice.events.last(),
            Some(ToyEventV1::InvocationEnd { root: final_root, .. }) if *final_root == root
        ));
    }
}
