//! RNS-BGG-specific terminal preparation, aggregation, and artifact transport.
//!
//! Terminal outputs are not part of the scheme-independent LUT protocol.  The
//! raw accumulators intentionally stay at their input modulus and are reduced
//! exactly once after all logical stages have been combined.

use super::{
    RnsBggEncoding, RnsBggError, RnsBggMatrix, RnsBggOnlineMasks, RnsBggPreparedTerminal,
    RnsBggPreparedTerminalDescriptor, RnsBggPublicKey, RnsBggRow, RnsBggScheme,
    RnsBggTerminalAccumulator, RnsBggTerminalEncoding, RnsBggTerminalMaterial,
    RnsBggTerminalPublicKey,
};
use crate::{
    program::ProgramValue,
    scheme::{EncodingProgramValue, PublicProgramValue},
};
use mxx_dsl::{DslContext, DslError, Family, GraphValue, GraphValueSchema};
use mxx_ir_core::artifact::{ArtifactConfidentiality, ProductionId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

type PreparedTerm = RnsBggPreparedTerminal<RnsBggPublicKey, RnsBggMatrix, ()>;
type PublicAccumulator = RnsBggTerminalAccumulator<RnsBggMatrix>;
type EncodingAccumulator = RnsBggTerminalAccumulator<RnsBggRow>;
type OnlineHelpers = RnsBggOnlineMasks;

#[derive(Debug, Error)]
/// Errors raised while preparing or evaluating RNS-BGG terminal stages.
pub enum TerminalError {
    #[error("terminal preparation requires at least one term")]
    /// No terminal terms were supplied.
    Empty,
    #[error("terminal inputs do not match the prepared term count")]
    /// Inputs and prepared terms have different structural counts.
    InputCount,
    #[error(transparent)]
    /// The RNS-BGG operation rejected the stage context.
    Scheme(#[from] RnsBggError),
    #[error(transparent)]
    /// The DSL graph could not be constructed.
    Graph(#[from] DslError),
}

/// Construct one terminal scope while preserving structural families.
pub(crate) fn prepare_terminal_value<O: GraphValue>(
    scheme: &RnsBggScheme,
    input: &PublicProgramValue<RnsBggScheme>,
    material: &ProgramValue<RnsBggTerminalMaterial, Family<RnsBggTerminalMaterial>>,
    aggregate: bool,
    project: impl FnOnce(PublicAccumulator) -> Result<O, RnsBggError>,
) -> Result<
    (ProgramValue<PreparedTerm, Family<PreparedTerm>>, ProgramValue<O, Family<O>>),
    TerminalError,
> {
    let count = match (input, material) {
        (ProgramValue::Family(values), ProgramValue::Family(materials))
            if values.count() != materials.count() =>
        {
            return Err(TerminalError::InputCount)
        }
        (ProgramValue::Family(values), _) => Some(values.count().clone()),
        (_, ProgramValue::Family(values)) => Some(values.count().clone()),
        _ => None,
    };
    let Some(count) = count else {
        let (cache, raw) = scheme.prepare_terminal_term(&input.at(0), &material.at(0))?;
        return Ok((ProgramValue::Scalar(cache), ProgramValue::Scalar(project(raw)?)));
    };
    let mut error = None;
    if aggregate {
        let values = mxx_dsl::parallel(count, |index| {
            scheme.prepare_terminal_term(&input.at(&index), &material.at(index)).map_err(|source| {
                error = Some(source);
                DslError::Schema
            })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        let cache = values.field(|value| value.0)?;
        let raw = scheme.reduce_terminal_public(values.field(|value| value.1)?)?;
        Ok((ProgramValue::Family(cache), ProgramValue::Scalar(project(raw)?)))
    } else {
        let values = mxx_dsl::parallel(count, |index| {
            scheme
                .prepare_terminal_term(&input.at(&index), &material.at(index))
                .and_then(|(cache, raw)| project(raw).map(|value| (cache, value)))
                .map_err(|source| {
                    error = Some(source);
                    DslError::Schema
                })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        Ok((
            ProgramValue::Family(values.field(|value| value.0)?),
            ProgramValue::Family(values.field(|value| value.1)?),
        ))
    }
}

/// Evaluate a matching prepared scope without rebuilding public products.
pub(crate) fn evaluate_terminal_value<O: GraphValue>(
    scheme: &RnsBggScheme,
    input: &EncodingProgramValue<RnsBggScheme>,
    prepared: &ProgramValue<PreparedTerm, Family<PreparedTerm>>,
    helpers: &OnlineHelpers,
    aggregate: bool,
    project: impl FnOnce(EncodingAccumulator) -> Result<O, RnsBggError>,
) -> Result<ProgramValue<O, Family<O>>, TerminalError> {
    let ProgramValue::Family(terms) = prepared else {
        if matches!(input, ProgramValue::Family(_)) {
            return Err(TerminalError::InputCount);
        }
        let raw = scheme.terminal_encoding_term(&input.at(0), &prepared.at(0), helpers)?;
        return Ok(ProgramValue::Scalar(project(raw)?));
    };
    if let ProgramValue::Family(values) = input &&
        values.count() != terms.count()
    {
        return Err(TerminalError::InputCount);
    }
    let mut error = None;
    if aggregate {
        let raw = mxx_dsl::parallel(terms.count().clone(), |index| {
            scheme.terminal_encoding_term(&input.at(&index), &terms.at(index), helpers).map_err(
                |source| {
                    error = Some(source);
                    DslError::Schema
                },
            )
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        Ok(ProgramValue::Scalar(project(scheme.reduce_terminal_encoding(raw)?)?))
    } else {
        let values = mxx_dsl::parallel(terms.count().clone(), |index| {
            scheme
                .terminal_encoding_term(&input.at(&index), &terms.at(index), helpers)
                .and_then(project)
                .map_err(|source| {
                    error = Some(source);
                    DslError::Schema
                })
        })
        .map_err(|source| {
            error.map(TerminalError::Scheme).unwrap_or(TerminalError::Graph(source))
        })?;
        Ok(ProgramValue::Family(values))
    }
}

impl RnsBggScheme {
    pub(crate) fn reduce_terminal_public(
        &self,
        values: Family<PublicAccumulator>,
    ) -> Result<PublicAccumulator, RnsBggError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }
    pub(crate) fn reduce_terminal_encoding(
        &self,
        values: Family<EncodingAccumulator>,
    ) -> Result<EncodingAccumulator, RnsBggError> {
        let metadata = values.schema().element;
        let value = crate::utils::balanced_sum_family(values.field(|value| value.value)?)
            .map_err(|e| RnsBggError::Graph(e.to_string()))?;
        Ok(RnsBggTerminalAccumulator {
            value,
            level: metadata.level,
            context: metadata.context,
            stages: metadata.stages,
        })
    }
    pub(crate) fn prepare_terminal_term(
        &self,
        input: &(RnsBggPublicKey, Option<mxx_dsl::Mat>),
        material: &RnsBggTerminalMaterial,
    ) -> Result<(PreparedTerm, PublicAccumulator), RnsBggError> {
        let prepared = self.prepare_terminal(&input.0, material)?;
        let raw = RnsBggTerminalAccumulator {
            value: prepared.raw_public,
            level: prepared.input.level,
            context: prepared.input.context,
            stages: vec![prepared.id],
        };
        let cache = RnsBggPreparedTerminal {
            binding: prepared.binding,
            input: prepared.input,
            id: prepared.id,
            branches: prepared.branches,
            raw_public: (),
        };
        Ok((cache, raw))
    }
    pub(crate) fn terminal_encoding_term(
        &self,
        input: &(RnsBggEncoding, Option<mxx_dsl::Mat>),
        prepared: &PreparedTerm,
        helpers: &OnlineHelpers,
    ) -> Result<EncodingAccumulator, RnsBggError> {
        self.evaluate_terminal_raw(&input.0, prepared, helpers)
    }
    pub(crate) fn add_terminal_public(
        &self,
        lhs: PublicAccumulator,
        rhs: PublicAccumulator,
    ) -> Result<PublicAccumulator, RnsBggError> {
        lhs.combine(rhs)
    }
    pub(crate) fn add_terminal_encoding(
        &self,
        lhs: EncodingAccumulator,
        rhs: EncodingAccumulator,
    ) -> Result<EncodingAccumulator, RnsBggError> {
        lhs.combine(rhs)
    }
    pub(crate) fn finish_terminal_public(
        &self,
        raw: PublicAccumulator,
    ) -> Result<RnsBggTerminalPublicKey, RnsBggError> {
        self.finalize_terminal_public(raw)
    }
    pub(crate) fn finish_terminal_encoding(
        &self,
        raw: EncodingAccumulator,
        public: &RnsBggTerminalPublicKey,
    ) -> Result<RnsBggTerminalEncoding, RnsBggError> {
        self.finalize_terminal_encoding(raw, public)
    }
}

/// Ordered terminal artifacts and their cached finalized public output.
pub struct PreparedTerminal<'a> {
    scheme: &'a RnsBggScheme,
    terms: Vec<PreparedTerm>,
    public: RnsBggTerminalPublicKey,
}

#[derive(Clone, Serialize, Deserialize)]
/// Serializable descriptor for ordered RNS-BGG terminal terms.
pub struct PreparedTerminalDescriptor {
    prefix: String,
    terms: Vec<RnsBggPreparedTerminalDescriptor<(String, <PreparedTerm as GraphValue>::Schema)>>,
    output_name: String,
    output: <RnsBggMatrix as GraphValue>::Schema,
    output_level: usize,
    output_context: [u8; 32],
    output_stages: Vec<[u8; 32]>,
}

/// Opaque trust anchor retained from terminal preparation for artifact import.
///
/// The descriptor is serialized transport metadata and must not be treated as
/// an authentication source.  Callers must retain this value from `export`
/// (or the corresponding trusted preparation step) and pass it to `import`.
#[derive(Clone)]
pub struct PreparedTerminalExpectation {
    digest: [u8; 32],
}

fn expectation_digest(
    descriptor: &PreparedTerminalDescriptor,
    expected: &[(super::RnsBggLutBinding, [u8; 32], [u8; 32])],
) -> [u8; 32] {
    let encoded = serde_json::to_vec(&(
        "rns-bgg-prepared-terminal-v1",
        &descriptor.prefix,
        &descriptor.terms,
        &descriptor.output_name,
        &descriptor.output,
        descriptor.output_level,
        descriptor.output_context,
        &descriptor.output_stages,
        expected,
    ))
    .expect("canonical prepared terminal expectation serialization");
    Sha256::digest(encoded).into()
}

impl<'a> PreparedTerminal<'a> {
    /// Export ordered terms and the finalized public output into a graph.
    pub fn export(
        &self,
        mut graph: DslContext,
        prefix: &str,
    ) -> Result<(DslContext, PreparedTerminalDescriptor, PreparedTerminalExpectation), RnsBggError>
    {
        let mut terms = Vec::with_capacity(self.terms.len());
        for (index, term) in self.terms.iter().enumerate() {
            let next = graph.public_output(format!("{prefix}.term.{index}"), term.clone())?;
            graph = next;
            terms.push(RnsBggPreparedTerminalDescriptor {
                prepared: (format!("{prefix}.term.{index}"), term.schema()),
            });
        }
        graph = graph.public_output(format!("{prefix}.output"), self.public.matrix.clone())?;
        let descriptor = PreparedTerminalDescriptor {
            prefix: prefix.into(),
            terms,
            output_name: format!("{prefix}.output"),
            output: self.public.matrix.schema(),
            output_level: self.public.level,
            output_context: self.public.context,
            output_stages: self.public.stages.clone(),
        };
        let expected = self
            .terms
            .iter()
            .map(|term| (term.binding().clone(), term.input.id, term.id))
            .collect::<Vec<_>>();
        let expectation =
            PreparedTerminalExpectation { digest: expectation_digest(&descriptor, &expected) };
        Ok((graph, descriptor, expectation))
    }
    /// Import and validate ordered terms from public artifacts.
    pub fn import(
        scheme: &'a RnsBggScheme,
        descriptor: &PreparedTerminalDescriptor,
        expectation: &PreparedTerminalExpectation,
        production: ProductionId,
        expected: &[(super::RnsBggLutBinding, [u8; 32], [u8; 32])],
    ) -> Result<Self, TerminalError> {
        if expected.is_empty() || descriptor.terms.is_empty() {
            return Err(TerminalError::Empty);
        }
        if descriptor.output_name != format!("{}.output", descriptor.prefix) {
            return Err(TerminalError::Scheme(RnsBggError::Context));
        }
        if expected.len() != descriptor.terms.len() {
            return Err(TerminalError::InputCount);
        }
        for (index, (data, binding)) in descriptor.terms.iter().zip(expected).enumerate() {
            let (prefix, schema) = &data.prepared;
            if prefix != &format!("{}.term.{index}", descriptor.prefix) {
                return Err(TerminalError::Scheme(RnsBggError::Context));
            }
            if schema.binding != binding.0 ||
                schema.input.id != binding.1 ||
                schema.id != binding.2 ||
                schema.input.context != scheme.context
            {
                return Err(TerminalError::Scheme(RnsBggError::Context));
            }
            if schema.branches.is_empty() {
                return Err(TerminalError::Scheme(RnsBggError::Context));
            }
        }
        if expectation.digest != expectation_digest(descriptor, expected) {
            return Err(TerminalError::Scheme(RnsBggError::Context));
        }
        let terms = descriptor
            .terms
            .iter()
            .map(|data| {
                let (prefix, schema) = &data.prepared;
                let term = schema.artifact_input(
                    production.clone(),
                    prefix,
                    ArtifactConfidentiality::Public,
                )?;
                scheme.check_key(&term.input)?;
                if term.branches.is_empty() {
                    return Err(RnsBggError::Context);
                }
                let columns =
                    term.branches[0].lifted_ciphertext.entry(0, 0).matrix_type().columns.clone();
                if term.branches.iter().any(|branch| {
                    branch.lifted_ciphertext.entry(0, 0).matrix_type().columns != columns
                }) {
                    return Err(RnsBggError::Context);
                }
                Ok(term)
            })
            .collect::<Result<Vec<_>, RnsBggError>>()?;
        let input_level = terms[0].input.level;
        if terms.iter().any(|term| term.input.level != input_level) {
            return Err(TerminalError::Scheme(RnsBggError::Context));
        }
        let level = input_level + 1;
        let columns = terms[0]
            .branches
            .first()
            .ok_or(TerminalError::Scheme(RnsBggError::Context))?
            .lifted_ciphertext
            .entry(0, 0)
            .matrix_type()
            .columns
            .clone();
        if terms.iter().any(|term| {
            term.branches
                .iter()
                .any(|branch| branch.lifted_ciphertext.entry(0, 0).matrix_type().columns != columns)
        }) {
            return Err(TerminalError::Scheme(RnsBggError::Context));
        }
        let output = super::import_blocks(
            &scheme.ring(level)?,
            &production,
            &descriptor.output_name,
            columns,
        );
        if descriptor.output != output.schema() ||
            descriptor.output_level != level ||
            descriptor.output_context != scheme.context ||
            descriptor.output_stages != terms.iter().map(|term| term.id).collect::<Vec<_>>()
        {
            return Err(TerminalError::Scheme(RnsBggError::Context));
        }
        let public = RnsBggTerminalPublicKey {
            matrix: output,
            level,
            context: scheme.context,
            stages: terms.iter().map(|term| term.id).collect(),
        };
        Ok(Self { scheme, terms, public })
    }
    /// Return the ordered prepared terms.
    pub fn prepared_terms(&self) -> &[PreparedTerm] {
        &self.terms
    }
    /// Prepare and aggregate terminal terms before the final modulus switch.
    pub fn new(
        scheme: &'a RnsBggScheme,
        inputs: &[(&(RnsBggPublicKey, Option<mxx_dsl::Mat>), &RnsBggTerminalMaterial)],
    ) -> Result<Self, TerminalError> {
        if inputs.is_empty() {
            return Err(TerminalError::Empty);
        }
        let mut terms = Vec::with_capacity(inputs.len());
        let raw = sum_terms(
            inputs.iter().map(|(input, material)| {
                let (prepared, raw) =
                    scheme.prepare_terminal_term(input, material).map_err(TerminalError::Scheme)?;
                terms.push(prepared);
                Ok(raw)
            }),
            |lhs, rhs| scheme.add_terminal_public(lhs, rhs).map_err(TerminalError::Scheme),
        )?;
        let public = scheme.finish_terminal_public(raw)?;
        Ok(Self { scheme, terms, public })
    }
    /// Return the finalized public terminal output.
    pub fn public(&self) -> &RnsBggTerminalPublicKey {
        &self.public
    }
    /// Evaluate and finalize the matching ordered encoded terms.
    pub fn evaluate(
        &self,
        inputs: &[(&(RnsBggEncoding, Option<mxx_dsl::Mat>), &OnlineHelpers)],
    ) -> Result<RnsBggTerminalEncoding, TerminalError> {
        if inputs.len() != self.terms.len() {
            return Err(TerminalError::InputCount);
        }
        let raw = sum_terms(
            inputs.iter().zip(&self.terms).map(|((input, helpers), term)| {
                self.scheme
                    .terminal_encoding_term(input, term, helpers)
                    .map_err(TerminalError::Scheme)
            }),
            |lhs, rhs| self.scheme.add_terminal_encoding(lhs, rhs).map_err(TerminalError::Scheme),
        )?;
        Ok(self.scheme.finish_terminal_encoding(raw, &self.public)?)
    }
}

pub(crate) fn sum_terms<T>(
    terms: impl IntoIterator<Item = Result<T, TerminalError>>,
    add: impl Fn(T, T) -> Result<T, TerminalError>,
) -> Result<T, TerminalError> {
    let mut levels = Vec::<Option<T>>::new();
    for term in terms {
        let mut value = term?;
        let mut level = 0;
        while level < levels.len() {
            let Some(left) = levels[level].take() else { break };
            value = add(left, value)?;
            level += 1;
        }
        if level == levels.len() {
            levels.push(Some(value));
        } else {
            levels[level] = Some(value);
        }
    }
    let mut result = None;
    for value in levels.into_iter().flatten() {
        result = Some(match result {
            None => value,
            Some(right) => add(value, right)?,
        });
    }
    result.ok_or(TerminalError::Empty)
}
