//! The correctness claim of an evaluation as a closed DSL protocol
//! ([`mxx_ir_core::protocol::ProtocolDecl`]), which `mxx_ir_core::lean::protocol::export` turns
//! into the Lean statement `GeneratedClaim.CorrectnessClaim`.
//!
//! A [`Dio2`] records the graph of every stage it runs under a stage name (and the stage of every
//! production). The protocol of the evaluation of the evaluator's chunks `x` is the obfuscation's
//! stages on the path of `x` and the evaluation's stages, in execution order:
//!
//! | Stage | Graph |
//! | --- | --- |
//! | `payload`, `mask` | the payload secret `t` and the mask `s_0` |
//! | `anchors`, `anchor_publics` | the anchor trapdoors and their public matrices `B_i` |
//! | `initial`, `zero` | the step input `S_0` and the zero step input of the decoded keys |
//! | `preprocess_<i>_<c>` | the Program of step `i` on chunk `c = x_i` |
//! | `keys_import_<i>_<c>`, `keys_online_<i>_<c>` | its composed run on the zero input: the decoded keys |
//! | `preimages_<i>_<c>` | their preimages under the anchor `B_i` |
//! | `import_<i>`, `online_<i>` | the evaluator's run of step `i`: the step input `S_(i+1)` or the output decoder |
//! | `decode` | the final state of the output decoder's plaintext |
//!
//! Every plain stage input is linked by name to the stage output it reads (a step input is output
//! under the names its import reads, see [`super::step::StepPlan::input_groups`]) or fed by an
//! external input:
//!
//! - every hash key (32 bytes);
//! - `z0`, the initial state, and `a0`, the initial labels `A_0` (entry `(r, c)` at `r (m_v + 1) +
//!   c`), read by the initial input;
//! - `y`, the obfuscator's bits (bit `b` of step `i` at `i obf_chunk_bits + b`), and the monomials
//!   `X^(y_monomial(b, y_b))` of every step's RHS family (`y<i>_coefficient`, `y<i>_exponent`);
//! - the labels' public plaintexts `X^(stride a)` of every step's import, by root tag
//!   (`label<i>_<tag>_{coefficient,exponent}`), and those of its run on the zero input
//!   (`keys_label<i>_<tag>_...`, which the decoded keys do not read);
//! - the clear current-label family of each symbolic label forest (`labels<i>`), constrained
//!   member-wise to the transition of `a0`; the key-only copy is independently public;
//! - the keys of every PRF `p` of step `i`: bucket `b`'s selector bits (`selector<i>_<p>_<b>`, with
//!   zero exponents) and coordinate table (`table<i>_<p>_<b>`, read by preprocessing and both
//!   online evaluations).
//!
//! The requirement checks that every label plaintext is the monomial of `A_i`, the labels the
//! label transitions of `x` give from `A_0`, that every `y` monomial is that of its bit, and that
//! every selector row is one-hot. The ideal functionality evaluates the function on `z0`, `x` and
//! `y` ([`super::StepProgram::ideal`]); the claim compares it with the decoded state.

use super::{
    Dio2, Dio2Error, Dio2Obfuscation,
    refresh::{HINT_INPUT, NEXT_ANCHOR_INPUT, OUTPUT_PLAINTEXT_OUTPUT, OUTPUT_PREIMAGE_INPUT},
    scheme::{
        ANCHOR_INPUT, INITIAL_LABELS_INPUT, INITIAL_STATE_INPUT, PLAINTEXT_INPUT, STATE_OUTPUT,
        anchor_output, chunk_index,
    },
    step::{ChainProgram, LABEL_TAG, LABEL_VALUES_INPUT, LabelProducts, PUBLIC_TAG},
};
use mxx_backends::SessionStore;
use mxx_dsl::{Bool, DslContext, IdealSpec, Int};
use mxx_exponent_lut::{
    api::{ApiError, StageExecution},
    claim::{external, stage, to_stage},
    program::ProgramData,
    program_gadgets::prf::SparseLwrPrf,
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

/// The field (`.coefficient` or `.exponent`) of a plaintext input named
/// `<prefix>/<role>/<id:?>.<field>`, when it is the plaintext of `id`.
fn plaintext_field<'a>(name: &'a str, role: &str, id: impl std::fmt::Debug) -> Option<&'a str> {
    let marker = format!("/{role}/{id:?}.");
    let start = name.find(&marker)? + marker.len() - 1;
    Some(&name[start..])
}

/// The PRFs of a step Program in claim order: the smudges, the σ sampler's, the CBD samplers'.
fn prfs(program: &ChainProgram) -> Vec<&SparseLwrPrf> {
    std::iter::once(&program.smudge)
        .chain(program.sigma.iter().map(|sampler| sampler.prf()))
        .chain(program.cbd.iter().map(|sampler| sampler.prf()))
        .collect()
}

/// The external inputs of a claim: every external's contract and destinations, by name.
type Externals = BTreeMap<String, (InputValueContract, Vec<ProtocolInputDestination>)>;

/// Feed `destination` from the external `name`, which a stage reads.
fn add(
    externals: &mut Externals,
    name: &str,
    destination: ProtocolInputDestination,
) -> Result<(), Dio2Error> {
    externals
        .get_mut(name)
        .ok_or(Dio2Error::Input("the claim reads an external no stage reads"))?
        .1
        .push(destination);
    Ok(())
}

/// How a stage input is fed.
enum Source {
    /// The output `.1` of stage `.0`.
    Link(String, String),
    /// The external input `.0` with contract `.1`.
    External(String, InputValueContract),
}

/// `[lower, upper]`.
fn range(lower: i64, upper: i64) -> InputValueContract {
    InputValueContract::IntegerRange { lower: lower.into(), upper: upper.into() }
}

/// `count` members in `[lower, upper]`, without requiring an addressable runtime vector.
fn family(count: impl Into<IntExpr>, lower: i64, upper: i64) -> InputValueContract {
    InputValueContract::Family { count: count.into(), element: Box::new(range(lower, upper)) }
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// The stages of the evaluation of `x`, in execution order.
    fn path_stages(x: &[Vec<bool>]) -> Vec<String> {
        let mut stages = ["payload", "anchors", "anchor_publics", "mask", "initial", "zero"]
            .map(str::to_owned)
            .to_vec();
        for (step, chunk) in x.iter().enumerate() {
            let chunk = chunk_index(chunk);
            for phase in ["preprocess", "keys_import", "keys_online", "preimages"] {
                stages.push(format!("{phase}_{step}_{chunk}"));
            }
            stages.push(format!("import_{step}"));
            stages.push(format!("online_{step}"));
        }
        stages.push("decode".to_owned());
        stages
    }

    /// The source of input `name` of stage `stage` of the evaluation of `x` on `obfuscation`.
    fn source(
        &self,
        obfuscation: &Dio2Obfuscation,
        stage: &str,
        name: &str,
        x: &[Vec<bool>],
    ) -> Option<Source> {
        let link = |producer: String, output: &str| Some(Source::Link(producer, output.to_owned()));
        let n = self.parameters().ring_dimension as i64;
        let plan = &obfuscation.plan;
        let function = &obfuscation.function;
        let steps = x.len();
        if let Some((producer, output)) = self.secrets.get(name) {
            return link(producer.clone(), output);
        }
        if name.starts_with("rns-bgg/hash-key") || name.starts_with("rns-bgg/root-hash-key") {
            return Some(Source::External(
                format!("hash_key {name}"),
                InputValueContract::Bytes { length: 32.into() },
            ));
        }
        match (stage, name) {
            ("initial", ANCHOR_INPUT) => {
                return link("anchor_publics".to_owned(), &anchor_output(0))
            }
            ("initial", INITIAL_STATE_INPUT) => {
                return Some(Source::External(
                    "z0".to_owned(),
                    range(0, (1i64 << function.state_bits) - 1),
                ));
            }
            ("initial", INITIAL_LABELS_INPUT) => {
                let count = plan.label_rows * (plan.label_width + 1);
                let q = self.parameters().label_modulus() as i64;
                return Some(Source::External("a0".to_owned(), family(count, 0, q - 1)));
            }
            ("decode", PLAINTEXT_INPUT) => {
                return link(format!("online_{}", steps - 1), OUTPUT_PLAINTEXT_OUTPUT);
            }
            _ => {}
        }
        // `(phase, step, chunk)` of a stage `<phase>_<step>[_<chunk>]`.
        let (phase, step) =
            ["preprocess", "keys_import", "keys_online", "preimages", "import", "online"]
                .into_iter()
                .find_map(|phase| {
                    let rest = stage.strip_prefix(phase)?.strip_prefix('_')?;
                    Some((phase, rest.split('_').next()?.parse::<usize>().ok()?))
                })?;
        let chunk = chunk_index(&x[step]);
        let program = &obfuscation.steps[step][chunk].program;
        if name == LABEL_VALUES_INPUT {
            return Some(Source::External(
                format!("{}labels{step}", if phase == "keys_online" { "keys_" } else { "" }),
                family(
                    plan.label_rows * (plan.label_width + 1),
                    0,
                    self.parameters().label_modulus() as i64 - 1,
                ),
            ));
        }
        let previous =
            || if step == 0 { "initial".to_owned() } else { format!("online_{}", step - 1) };
        let step_input = name.starts_with("rns-bgg/root-row/") ||
            name.starts_with("rns-bgg/root-helper/") ||
            name == HINT_INPUT;
        match phase {
            "keys_import" | "keys_online" if step_input => return link("zero".to_owned(), name),
            "import" | "online" if step_input => return link(previous(), name),
            "keys_online" | "online" if name == NEXT_ANCHOR_INPUT => {
                return link("anchor_publics".to_owned(), &anchor_output(step + 1));
            }
            "preimages" => return link(format!("keys_online_{step}_{chunk}"), name),
            "online" if name.starts_with("dio2/preimage") || name == OUTPUT_PREIMAGE_INPUT => {
                return link(format!("preimages_{step}_{chunk}"), name);
            }
            _ => {}
        }
        // A coordinate table of a PRF, read by preprocessing and both online evaluations.
        if name.starts_with("exponent-lut/data.") {
            for (p, prf) in prfs(program).into_iter().enumerate() {
                if let Some(bucket) =
                    prf.tables().iter().position(|table| ProgramData::input_name(*table) == name)
                {
                    let members = prf.selector_families()[bucket].members;
                    let lwr = self.parameters().prf.lwr_dimension as i64;
                    return Some(Source::External(
                        format!("table{step}_{p}_{bucket}"),
                        family(members, 0, lwr),
                    ));
                }
            }
            return None;
        }
        // An RHS plaintext of preprocessing: the obfuscator's monomials or a selector family.
        if name.contains("/rhs/") {
            let obf = function.obf_chunk_bits;
            if let Some(field) = plaintext_field(name, "rhs", program.y) {
                return Some(match field {
                    ".coefficient" => {
                        Source::External(format!("y{step}_coefficient"), family(obf, -1, 1))
                    }
                    _ => Source::External(format!("y{step}_exponent"), family(obf, 0, n - 1)),
                });
            }
            for (p, prf) in prfs(program).into_iter().enumerate() {
                for (bucket, selector) in prf.selector_families().iter().enumerate() {
                    if let Some(field) = plaintext_field(name, "rhs", selector.rhs) {
                        return Some(match field {
                            ".coefficient" => Source::External(
                                format!("selector{step}_{p}_{bucket}"),
                                family(selector.members, 0, 1),
                            ),
                            _ => Source::External(
                                format!("zero_{}", selector.members),
                                family(selector.members, 0, 0),
                            ),
                        });
                    }
                }
            }
            return None;
        }
        // A public label plaintext of an import, by the root tag of its input.
        if let Some((tag, field)) = program
            .root_inputs
            .iter()
            .find_map(|(wire, tag)| Some((tag, plaintext_field(name, "input", wire)?)))
        {
            let count = match tag.as_str() {
                PUBLIC_TAG => plan.label_rows + 1,
                LABEL_TAG => plan.label_rows * plan.label_width,
                _ => return None,
            };
            let prefix = if phase == "keys_import" { "keys_label" } else { "label" };
            let tag = if tag == PUBLIC_TAG { "public" } else { "label" };
            return Some(match field {
                ".coefficient" => Source::External(
                    format!("{prefix}{step}_{tag}_coefficient"),
                    family(count, -1, 1),
                ),
                _ => Source::External(
                    format!("{prefix}{step}_{tag}_exponent"),
                    family(count, 0, n - 1),
                ),
            });
        }
        None
    }

    /// The correctness claim of the evaluation of the evaluator's chunks `x` on `obfuscation`
    /// (both run on this instance), with failure probability `2^-failure_probability_log2`
    /// (`None`: every execution): see the module documentation.
    pub fn protocol(
        &self,
        obfuscation: &Dio2Obfuscation,
        x: &[Vec<bool>],
        failure_probability_log2: Option<u32>,
    ) -> Result<ProtocolDecl, Dio2Error> {
        let function = &obfuscation.function;
        let plan = &obfuscation.plan;
        let steps = function.stream_length();
        if x.len() != steps {
            return Err(Dio2Error::Input("one evaluator chunk per step"));
        }
        let productions = self
            .productions
            .iter()
            .map(|(production, stage)| (production.clone(), StageId(stage.clone())))
            .collect();
        let mut externals = Externals::new();
        let mut unknown = Vec::new();
        let mut stages = Vec::new();
        for name in Self::path_stages(x) {
            let graph = self
                .graphs
                .get(&name)
                .ok_or(Dio2Error::Input("a stage of the evaluated path was not recorded"))?;
            let mut links = BTreeMap::new();
            for input in plain_inputs(graph) {
                match self.source(obfuscation, &name, &input, x) {
                    Some(Source::Link(producer, output)) => {
                        let produced = self
                            .graphs
                            .get(&producer)
                            .is_some_and(|graph| graph.outputs().contains_key(&output));
                        if !produced {
                            unknown
                                .push(format!("{name}: {input} <- {producer}.{output} (missing)"));
                        }
                        links.insert(input, (StageId(producer), output));
                    }
                    Some(Source::External(external, contract)) => {
                        let entry = externals
                            .entry(external)
                            .or_insert_with(|| (contract.clone(), Vec::new()));
                        if entry.0 != contract {
                            return Err(Dio2Error::Input("an external input has two contracts"));
                        }
                        entry.1.push(to_stage(&name, &input));
                    }
                    None => unknown.push(format!("{name}: {input}")),
                }
            }
            stages.push(stage(&name, graph, &productions, &links)?);
        }
        if !unknown.is_empty() {
            return Err(Dio2Error::Graph(format!("stage inputs without a source: {unknown:?}")));
        }

        let requirement_input = |input: &str| ProtocolInputDestination::Requirement {
            requirement: 0,
            input: input.to_owned(),
        };
        let ideal_input = |input: &str| ProtocolInputDestination::Ideal { input: input.to_owned() };
        let obf = function.obf_chunk_bits;
        externals.insert("y".to_owned(), (family(steps * obf, 0, 1), Vec::new()));
        add(&mut externals, "z0", ideal_input("z0"))?;
        add(&mut externals, "y", ideal_input("y"))?;
        add(&mut externals, "y", requirement_input("y"))?;
        add(&mut externals, "a0", requirement_input("a0"))?;

        // The requirement.
        let n = self.parameters().ring_dimension as i64;
        let requirement = DslContext::new("dio2-inputs");
        let input_family = |name: &str| -> Result<mxx_dsl::Family<Int>, Dio2Error> {
            let count =
                match &externals.get(name).ok_or(Dio2Error::Input("an external is missing"))?.0 {
                    InputValueContract::Family { count, .. } => count.clone(),
                    _ => return Err(Dio2Error::Input("an external is not a family")),
                };
            Ok(requirement.int_family_input(name, count))
        };
        let mut read = Vec::new();
        let mut valid = Bool::constant(true);
        // `c X^e` is the monomial `X^(2 stride a)` of the exponent `value`: `c^2 = 1` and
        // `2 e + n (1 - c) = 2 value` (a negative coefficient past `X^n`).
        let monomial = |coefficient: Int, exponent: Int, value: Int| {
            coefficient.clone().mul(coefficient.clone()).equal(1) &
                exponent.mul(2).add(Int::constant(1).sub(coefficient).mul(n)).equal(value.mul(2))
        };
        // The labels `A_i` from `A_0` by the label transitions of `x`.
        let a0 = input_family("a0")?;
        let width = plan.label_width;
        let mut labels = a0;
        let prf_stride = self.parameters().profile()?.layout.stride() as i64;
        let state_stride = function.state_layout.stride() as i64;
        let y = input_family("y")?;
        for (step, chunk) in x.iter().enumerate() {
            let values_name = format!("labels{step}");
            if externals.contains_key(&values_name) {
                let values = input_family(&values_name)?;
                valid = mxx_dsl::iterate(plan.label_rows * (width + 1), valid, |member, valid| {
                    Ok(valid & values.at(member.clone()).equal(labels.at(member)))
                })?;
                read.push(values_name);
            }
            for (tag, count) in
                [("public", plan.label_rows + 1), ("label", plan.label_rows * width)]
            {
                let coefficient = format!("label{step}_{tag}_coefficient");
                let exponent = format!("label{step}_{tag}_exponent");
                if !externals.contains_key(&coefficient) {
                    continue;
                }
                let (c, e) = (input_family(&coefficient)?, input_family(&exponent)?);
                valid = mxx_dsl::iterate(count, valid, |index, valid| {
                    let value = if tag == "public" {
                        mxx_dsl::select(
                            index.clone().equal(plan.label_rows).to_int(),
                            vec![
                                labels.at(index.clone().rem(plan.label_rows).mul(width + 1)),
                                Int::constant(0),
                            ],
                        )?
                    } else {
                        labels.at(index
                            .clone()
                            .div(width)
                            .mul(width + 1)
                            .add(index.clone().rem(width))
                            .add(1usize))
                    };
                    Ok(valid & monomial(c.at(index.clone()), e.at(index), value.mul(prf_stride)))
                })?;
                read.extend([coefficient, exponent]);
            }
            let coefficient = format!("y{step}_coefficient");
            if externals.contains_key(&coefficient) {
                let exponent = format!("y{step}_exponent");
                let (c, e) = (input_family(&coefficient)?, input_family(&exponent)?);
                for bit in 0..obf {
                    let index = Int::constant(bit as i64);
                    let value =
                        y.at(Int::constant((step * obf + bit) as i64)).mul(state_stride << bit);
                    valid = valid & monomial(c.at(index.clone()), e.at(index), value);
                }
                read.extend([coefficient, exponent]);
            }
            if step + 1 < x.len() {
                labels = LabelProducts::new(self.parameters(), function.eval_chunk_bits).next(
                    self.parameters(),
                    &labels,
                    Int::constant(super::function::chunk_value(chunk) as i64),
                )?;
            }
        }
        // Every selector row is one-hot.
        for (step, chunk) in x.iter().enumerate() {
            let program = &obfuscation.steps[step][chunk_index(chunk)].program;
            for (p, prf) in prfs(program).into_iter().enumerate() {
                for (bucket, selector) in prf.selector_families().iter().enumerate() {
                    let name = format!("selector{step}_{p}_{bucket}");
                    if !externals.contains_key(&name) {
                        continue;
                    }
                    let row_length = selector
                        .row_length
                        .ok_or(Dio2Error::Input("a selector family is not row-wise one-hot"))?;
                    let bits = input_family(&name)?;
                    valid = mxx_dsl::iterate(
                        selector.members / row_length as u128,
                        valid,
                        |row, valid| {
                            let total =
                                mxx_dsl::iterate(row_length, Int::constant(0), |slot, total| {
                                    Ok(total
                                        .add(bits.at(row.clone().mul(row_length as i64).add(slot))))
                                })?;
                            Ok(valid & total.equal(1))
                        },
                    )?;
                    read.push(name);
                }
            }
        }
        for name in read {
            add(&mut externals, &name, requirement_input(&name))?;
        }
        let requirement =
            PurePredicateSpec::new(requirement.output("valid", valid)?.build()?.graph)
                .map_err(ApiError::scheme)?;

        // The ideal functionality.
        let ideal = DslContext::new("dio2-ideal");
        let mut state = ideal.input::<Int>("z0", mxx_dsl::IntType)?;
        let bits = ideal.int_family_input("y", steps * obf);
        for (step, chunk) in x.iter().enumerate() {
            let y = (0..obf)
                .map(|bit| bits.at(Int::constant((step * obf + bit) as i64)))
                .collect::<Vec<_>>();
            let row = mxx_dsl::parallel(function.data.width, |entry| {
                mxx_dsl::select(
                    entry,
                    function
                        .step_row(step)
                        .iter()
                        .map(|value| Int::constant(*value as i64))
                        .collect(),
                )
            })?;
            state = function.body.ideal(
                &row,
                state,
                Int::constant(super::function::chunk_value(chunk) as i64),
                &y,
            )?;
        }
        let ideal = IdealSpec::new(ideal.output("state", state)?.build()?.graph)
            .map_err(ApiError::scheme)?;

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
                workflow: Workflow { stages, entrypoint: id("decode") },
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
                            stage: id("decode"),
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
}
