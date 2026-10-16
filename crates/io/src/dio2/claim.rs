//! The correctness claim of an evaluation as a closed DSL protocol
//! ([`mxx_ir_core::protocol::ProtocolDecl`]), which `mxx_ir_core::lean::protocol::export` turns
//! into the Lean statement `GeneratedClaim.CorrectnessClaim`.
//!
//! The protocol has two stages, the graphs a [`Dio2`] recorded: [`OBFUSCATE_STAGE`] (the
//! obfuscation, one production) and [`EVALUATE_STAGE`] (the evaluation, which reads that
//! production by its artifact inputs). Every plain graph input is fed by an external input:
//!
//! - every hash key (32 bytes): the root hash key and each Program kind's stage hash key, read by
//!   both stages;
//! - `z0`, the initial state, and `y`, the obfuscator's bits (bit `b` of step `i` at `i
//!   obf_chunk_bits + b`), read by the obfuscation and the ideal functionality;
//! - `x`, the evaluator's chunks (step `i`'s little-endian value at `i`), read by the evaluation
//!   and the ideal functionality;
//! - `a0`, the initial labels `A_0` (entry `(r, c)` at `r (m_v + 1) + c`), and `rows`, the
//!   function's step rows, read by both stages (`rows` by the ideal functionality too);
//! - the keys of every PRF role and bucket: its selector bits (`selectors <role> <bucket>`, read by
//!   the obfuscation) and its coordinate tables (`table <role> <bucket>`, read by both stages),
//!   every chunk's at `x members ..`.
//!
//! The requirement checks that every selector row is one-hot; the contracts bound every other
//! input. The ideal functionality evaluates the function on `z0`, `x`, `y` and `rows` in one loop
//! over the steps ([`super::StepProgram::ideal`]); the claim compares it with the decoded state.

use super::{
    Dio2, Dio2Error, Dio2Obfuscation,
    scheme::{
        EVALUATE_STAGE, INITIAL_STATE_INPUT, Kind, LABELS_INPUT, OBFUSCATE_STAGE, STATE_OUTPUT,
        STEP_DATA_INPUT, X_INPUT, Y_INPUT, selector_input, table_input,
    },
};
use mxx_backends::SessionStore;
use mxx_dsl::{Bool, DslContext, IdealSpec, Int};
use mxx_exponent_lut::{
    api::{ApiError, StageExecution},
    claim::{external, stage, to_stage},
};
use mxx_ir_core::{
    Graph, IntExpr,
    node::NodeKind,
    protocol::{
        ClosedProtocolBundle, ComparatorEndpointBinding, ComparatorSpec, EndpointBinding,
        EndpointBindings, EndpointSemanticBinding, EndpointSpecId, InputContract,
        InputValueContract, OutputRef, ProtocolDecl, ProtocolInputDestination,
        ProtocolPreconditionSpec, PurePredicateSpec, StageId, Workflow,
    },
};
use std::collections::BTreeMap;

/// The names of the plain (non-artifact) root inputs of `graph`.
fn plain_inputs(graph: &Graph) -> Vec<String> {
    graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: None, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// The external inputs of a claim: every external's contract and destinations, by name.
type Externals = BTreeMap<String, (InputValueContract, Vec<ProtocolInputDestination>)>;

/// `[lower, upper]`.
fn range(lower: impl Into<IntExpr>, upper: impl Into<IntExpr>) -> InputValueContract {
    InputValueContract::IntegerRange { lower: lower.into(), upper: upper.into() }
}

/// `count` members in `[lower, upper]`, without requiring an addressable runtime vector.
fn family(
    count: impl Into<IntExpr>,
    lower: impl Into<IntExpr>,
    upper: impl Into<IntExpr>,
) -> InputValueContract {
    InputValueContract::Family { count: count.into(), element: Box::new(range(lower, upper)) }
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// The external input that feeds the plain input `name`, with its contract.
    fn source(
        &self,
        obfuscation: &Dio2Obfuscation,
        name: &str,
    ) -> Result<Option<(String, InputValueContract)>, Dio2Error> {
        let (function, plan) = (&obfuscation.function, &obfuscation.plan);
        let steps = function.stream_length();
        let chunks = 1u128 << function.eval_chunk_bits;
        if name.contains("hash-key") {
            return Ok(Some((
                format!("hash_key {name}"),
                InputValueContract::Bytes { length: 32.into() },
            )));
        }
        Ok(Some(match name {
            INITIAL_STATE_INPUT => ("z0".to_owned(), range(0, (1usize << function.state_bits) - 1)),
            LABELS_INPUT => (
                "a0".to_owned(),
                family(
                    plan.label_rows * (plan.label_width + 1),
                    0,
                    self.parameters.label_modulus() - 1,
                ),
            ),
            Y_INPUT => ("y".to_owned(), family(steps * function.obf_chunk_bits, 0, 1)),
            X_INPUT => ("x".to_owned(), family(steps, 0, chunks - 1)),
            STEP_DATA_INPUT => {
                ("rows".to_owned(), family(steps * function.data.width, 0, function.data.bound - 1))
            }
            _ => {
                for kind in [Kind::Refreshing, Kind::Last] {
                    let Ok(material) = obfuscation.material(kind) else { continue };
                    for (role, prf) in kind.prfs(&material.program) {
                        for (bucket, selector) in prf.selector_families().iter().enumerate() {
                            let count = chunks * selector.members;
                            if name == selector_input(role, bucket) {
                                return Ok(Some((
                                    format!("selectors {role} {bucket}"),
                                    family(count, 0, 1),
                                )));
                            }
                            if name == table_input(role, bucket) {
                                return Ok(Some((
                                    format!("table {role} {bucket}"),
                                    family(count, 0, self.parameters.prf.lwr_dimension),
                                )));
                            }
                        }
                    }
                }
                return Ok(None);
            }
        }))
    }

    /// The correctness claim of the evaluation of `obfuscation` (both run on this instance),
    /// with failure probability `2^-failure_probability_log2` (`None`: every execution): see
    /// the module documentation.
    pub fn protocol(
        &self,
        obfuscation: &Dio2Obfuscation,
        failure_probability_log2: Option<u32>,
    ) -> Result<ProtocolDecl, Dio2Error> {
        let function = &obfuscation.function;
        let productions = self
            .productions
            .iter()
            .map(|(production, stage)| (production.clone(), StageId(stage.clone())))
            .collect();
        let mut externals = Externals::new();
        let mut unknown = Vec::new();
        let mut stages = Vec::new();
        for name in [OBFUSCATE_STAGE, EVALUATE_STAGE] {
            let graph = self
                .graphs
                .get(name)
                .ok_or(Dio2Error::Input("the obfuscation and the evaluation were not recorded"))?;
            for input in plain_inputs(graph) {
                match self.source(obfuscation, &input)? {
                    Some((external, contract)) => {
                        let entry = externals
                            .entry(external)
                            .or_insert_with(|| (contract.clone(), Vec::new()));
                        if entry.0 != contract {
                            return Err(Dio2Error::Input("an external input has two contracts"));
                        }
                        entry.1.push(to_stage(name, &input));
                    }
                    None => unknown.push(format!("{name}: {input}")),
                }
            }
            stages.push(stage(name, graph, &productions, &BTreeMap::new())?);
        }
        if !unknown.is_empty() {
            return Err(Dio2Error::Graph(format!("stage inputs without a source: {unknown:?}")));
        }
        let requirement = Self::requirement(obfuscation, &mut externals)?;
        let ideal = Self::ideal(function, &mut externals)?;
        let (contracts, bindings): (Vec<_>, Vec<_>) = externals
            .into_iter()
            .map(|(name, (contract, destinations))| external(&name, contract, destinations))
            .unzip();
        let endpoint = EndpointSpecId::Exact;
        let id = |name: &str| StageId(name.to_owned());
        Ok(ProtocolDecl::new(ProtocolDecl {
            params: Vec::new(),
            bindings: Default::default(),
            failure_probability_log2,
            bundle: ClosedProtocolBundle {
                workflow: Workflow { stages, entrypoint: id(EVALUATE_STAGE) },
                ideal,
                requirements: vec![requirement],
                comparator: ComparatorSpec::Equality {
                    endpoints: vec![ComparatorEndpointBinding {
                        endpoint,
                        actual_input: STATE_OUTPUT.to_owned(),
                        ideal_input: "state".to_owned(),
                        result_output: "failure".to_owned(),
                        failure_value: true,
                    }],
                },
                endpoints: EndpointBindings {
                    entries: vec![EndpointBinding {
                        spec: endpoint,
                        semantics: EndpointSemanticBinding::Exact,
                        workflow_output: OutputRef {
                            stage: id(EVALUATE_STAGE),
                            output: STATE_OUTPUT.to_owned(),
                        },
                        ideal_output: "state".to_owned(),
                    }],
                },
                operational_decoder_targets: Vec::new(),
                endpoint_specs: vec![endpoint],
                input_contract: InputContract { inputs: contracts },
                input_bindings: bindings,
                precondition_spec: ProtocolPreconditionSpec {
                    requirement_outputs: vec!["valid".to_owned()],
                },
            },
        })
        .map_err(ApiError::scheme)?)
    }

    /// The requirement: every row of `W` selector bits of every PRF bucket is one-hot.
    fn requirement(
        obfuscation: &Dio2Obfuscation,
        externals: &mut Externals,
    ) -> Result<PurePredicateSpec, Dio2Error> {
        let chunks = 1u128 << obfuscation.function.eval_chunk_bits;
        let requirement = DslContext::new("dio2-inputs");
        let mut valid = Bool::constant(true);
        for kind in [Kind::Refreshing, Kind::Last] {
            let Ok(material) = obfuscation.material(kind) else { continue };
            for (role, prf) in kind.prfs(&material.program) {
                for (bucket, selector) in prf.selector_families().iter().enumerate() {
                    let name = format!("selectors {role} {bucket}");
                    let Some((_, destinations)) = externals.get_mut(&name) else { continue };
                    destinations.push(ProtocolInputDestination::Requirement {
                        requirement: 0,
                        input: name.clone(),
                    });
                    let row_length = selector
                        .row_length
                        .ok_or(Dio2Error::Input("a selector family is row-wise one-hot"))?;
                    let bits = requirement.int_family_input(&name, chunks * selector.members);
                    let rows = chunks * selector.members / row_length as u128;
                    valid = mxx_dsl::iterate(rows, valid, |row, valid| {
                        let total =
                            mxx_dsl::iterate(row_length, Int::constant(0), |slot, total| {
                                Ok(total.add(bits.at(row.clone().mul(row_length).add(slot))))
                            })?;
                        Ok(valid & total.equal(1))
                    })?;
                }
            }
        }
        Ok(PurePredicateSpec::new(requirement.output("valid", valid)?.build()?.graph)
            .map_err(ApiError::scheme)?)
    }

    /// The ideal functionality: the function on `z0`, `x`, `y` and `rows`, one loop over the
    /// steps.
    fn ideal(
        function: &super::StreamingFunction,
        externals: &mut Externals,
    ) -> Result<IdealSpec, Dio2Error> {
        let steps = function.stream_length();
        let (obf, width) = (function.obf_chunk_bits, function.data.width);
        let ideal = DslContext::new("dio2-ideal");
        let z0 = ideal.input::<Int>("z0", mxx_dsl::IntType)?;
        let x = ideal.int_family_input("x", steps);
        let y = ideal.int_family_input("y", steps * obf);
        let rows = ideal.int_family_input("rows", steps * width);
        let state = mxx_dsl::iterate(steps, z0, |step, state| {
            let row =
                mxx_dsl::parallel(width, |entry| Ok(rows.at(step.clone().mul(width).add(entry))))?;
            let bits = (0..obf).map(|bit| y.at(step.clone().mul(obf).add(bit))).collect::<Vec<_>>();
            function.body.ideal(&row, state, x.at(step), &bits)
        })?;
        for name in ["z0", "x", "y", "rows"] {
            externals
                .get_mut(name)
                .ok_or(Dio2Error::Input("the ideal functionality reads a recorded input"))?
                .1
                .push(ProtocolInputDestination::Ideal { input: name.to_owned() });
        }
        Ok(IdealSpec::new(ideal.output("state", state)?.build()?.graph)
            .map_err(ApiError::scheme)?)
    }
}
