//! The evaluator (spec §6, §8): one graph that reads the obfuscation's production.
//!
//! A sequential loop over the refreshing steps carries the step input `S_i` (every root row
//! and the anchor hint, [`StepRows`]) and the clear labels `A_i`. Its body evaluates the uniform
//! step Program once: the root rows are graph values (the evaluator-bit family is the chunk's
//! public monomials times `ONEPUB`), the integer data are the step's row and the chunk's PRF
//! tables, and every Static preprocessing artifact is read at member `(i, x_i)` by the runtime
//! chunk `x_i`. The row composition of the refresh ([`Compose::next_rows`]) decodes the next
//! step input, and the label transition of `x_i` the next labels. After the loop the last
//! step's Program and its output decoder give the state monomial modulo the first prime, which
//! [`decode_state`] reads as the final state.

use super::{
    Dio2, Dio2Error, Dio2Parameters, Failure,
    carrier::Row,
    compose::{Compose, NextStep, split_deferred},
    labels::{LabelColumns, LabelProducts},
    scheme::{
        ANCHORS_OUTPUT, Dio2Obfuscation, EVALUATE_STAGE, INITIAL_OUTPUT, Kind, LABELS_INPUT,
        STATE_OUTPUT, X_INPUT, derive, record, step_chain, step_root,
    },
    step::{PRIVATE_TAG, PUBLIC_TAG, STATE_TAG, StepRows, XBITS_TAG, xbits_exponent, xbits_row},
};
use mxx_backends::{RuntimeValue, SessionStore};
use mxx_dsl::{DslContext, Family, Int, Mat, Preimage, Ring};
use mxx_exponent_lut::{
    api::{self, GraphEmission, ProgramBuild, StageExecution},
    program::{GraphFamily, InputBinding},
    rns_bgg::{RnsBggEncoding, RnsBggRoot, RnsBggRow, RnsOnlineEvalBuild, RnsOutputUse},
};
use mxx_ir_core::{IntExpr, ParamEnv, artifact::ArtifactAvailability};
use std::{cell::RefCell, collections::BTreeMap};

/// The construction of an evaluation graph.
struct Evaluator<'a> {
    parameters: &'a Dio2Parameters,
    obfuscation: &'a Dio2Obfuscation,
    root: RnsBggRoot,
    /// The evaluator's chunk of every step ([`X_INPUT`]).
    chunks: Family<Int>,
    columns: LabelColumns,
    /// The resident inputs of the graph's builders (hash keys).
    runtime: BTreeMap<String, RuntimeValue>,
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// Evaluate `obfuscation` on the evaluator's chunks `x`, one per step.
    pub(crate) fn evaluate_chunks(
        &mut self,
        obfuscation: &Dio2Obfuscation,
        x: &[Vec<bool>],
    ) -> Result<usize, Dio2Error> {
        let function = &obfuscation.function;
        let steps = function.stream_length();
        if x.len() != steps || x.iter().any(|chunk| chunk.len() != function.eval_chunk_bits) {
            return Err(Dio2Error::Input("the evaluator's chunks do not fit the function"));
        }
        let parameters = self.parameters.clone();
        let mut evaluator = Evaluator {
            parameters: &parameters,
            obfuscation,
            root: parameters.root(derive(&obfuscation.hash_key, "root", &[]))?,
            chunks: mxx_dsl::int_family_input(X_INPUT, steps),
            columns: LabelProducts::new(&parameters, function.eval_chunk_bits).columns(),
            runtime: BTreeMap::new(),
        };
        let state = evaluator.state()?;
        let graph = DslContext::new("dio2-evaluate").output(STATE_OUTPUT, state)?;
        let mut runtime = std::mem::take(&mut evaluator.runtime);
        if !self.inputs_by_shape {
            runtime.extend(obfuscation.public.clone());
            runtime.insert(
                X_INPUT.to_owned(),
                RuntimeValue::integer_values(
                    x.iter().map(|chunk| super::function::chunk_value(chunk).into()).collect(),
                ),
            );
        }
        let production = &obfuscation.production;
        let (graph, ()) = GraphEmission {
            graph,
            runtime_inputs: runtime,
            manifests: obfuscation
                .input_storage
                .manifests(false)
                .into_iter()
                .chain([(production.production().clone(), production.manifest().clone())])
                .collect(),
            descriptor: (),
        }
        .build()?;
        let mut outputs = api::execute_graph(
            graph,
            &ParamEnv::default(),
            &mut self.execution,
            &mut self.store,
            record(&mut self.graphs, EVALUATE_STAGE),
        )?
        .outputs;
        let state = outputs.remove(STATE_OUTPUT).ok_or(Dio2Error::Input("the state is missing"))?;
        self.execution
            .download_integers(&state)?
            .first()
            .and_then(num_traits::ToPrimitive::to_usize)
            .ok_or(Dio2Error::Input("the decoded state is not a state"))
    }
}

impl Evaluator<'_> {
    /// The ring of `Q_0`.
    fn ring(&self) -> Ring {
        self.root.ring()
    }

    /// The final state: the refreshing steps in one sequential loop, then the last step.
    fn state(&mut self) -> Result<Int, Dio2Error> {
        let obfuscation = self.obfuscation;
        let (plan, function) = (&obfuscation.plan, &obfuscation.function);
        let production = obfuscation.production.production().clone();
        let initial = DslContext::new("dio2-evaluate")
            .artifact_input::<super::step::StepRowsValue>(
                production,
                INITIAL_OUTPUT,
                StepRows::schema(plan, &self.ring()),
                ArtifactAvailability::Transferred,
            )?;
        let labels =
            mxx_dsl::int_family_input(LABELS_INPUT, plan.label_rows * (plan.label_width + 1));
        let steps = function.stream_length();
        let failure = Failure::default();
        let carried = mxx_dsl::iterate(steps - 1, (initial, labels), |step, (rows, labels)| {
            let rows = StepRows::from_value(rows);
            let (rows, labels) = failure.keep(self.refresh(step, &rows, &labels))?;
            Ok((rows.into_value(), labels))
        });
        let (rows, labels) = failure.finish(carried)?;
        self.output(&StepRows::from_value(rows), &labels)
    }

    /// The anchors `B_i` of the obfuscation.
    fn anchors(&self) -> Family<Mat> {
        let plan = &self.obfuscation.plan;
        self.ring().family_artifact_input(
            self.obfuscation.production.production().clone(),
            ANCHORS_OUTPUT,
            self.obfuscation.function.stream_length(),
            (2, super::anchor_columns(plan.columns)),
            ArtifactAvailability::Transferred,
        )
    }

    /// The preimages of `kind`'s instances: `per` of each instance side by side.
    fn preimages(
        &self,
        kind: Kind,
        per: usize,
        columns: usize,
    ) -> Result<Family<Preimage>, Dio2Error> {
        let plan = &self.obfuscation.plan;
        let members = kind.members(&self.obfuscation.function) * per;
        Ok(self.ring().preimage_family_artifact_input(
            self.obfuscation.production.production().clone(),
            kind.preimages(),
            members,
            (super::anchor_columns(plan.columns), columns),
            num_bigint::BigInt::from(super::preimage_bound(self.parameters)?),
            ArtifactAvailability::Transferred,
        ))
    }

    /// The composition of the online evaluation of `kind`'s Program at step `step` on the step
    /// input `rows` and the clear labels `labels`, on the instance's chain `chain`, given to
    /// `compose`: the lowered outputs as row carriers, the deferred rows read by blocks.
    fn compose<'c, T>(
        &mut self,
        kind: Kind,
        (step, chunk): (&Int, &Int),
        (rows, labels): (&StepRows, &Family<Int>),
        chain: &'c mxx_exponent_lut::rns_bgg::RnsBggParameters,
        compose: impl FnOnce(&Compose<'_, Row>) -> Result<T, Dio2Error>,
    ) -> Result<T, Dio2Error> {
        let obfuscation = self.obfuscation;
        let (plan, function) = (&obfuscation.plan, &obfuscation.function);
        let material = obfuscation.material(kind)?;
        let program = &material.program;
        let (context, stage_key) = kind.identity(&obfuscation.hash_key);
        let mut build = RnsOnlineEvalBuild::new(chain, context)
            .with_artifact_prefix(kind.prefix())
            .with_root_inputs(program.root_inputs.clone())
            .with_output_use(RnsOutputUse::Gadget)
            .with_material_member(kind.member(function, step, chunk), kind.members(function));
        build.bind_data_values(
            &program.program,
            kind.data(function, program, (step, chunk), &obfuscation.input_storage)?,
        )?;
        let ring = self.ring();
        let stride = self.parameters.profile()?.layout.stride();
        let one = rows.public.at(Int::constant(plan.label_rows));
        let mut graph_rows = BTreeMap::new();
        let mut payloads = BTreeMap::new();
        // The root rows of every input; a family's members (and their public payloads) are
        // formed where the Program reads them.
        for (wire, tag) in &program.root_inputs {
            let (row, payload) = match tag.as_str() {
                PRIVATE_TAG => (InputBinding::Scalar(rows.private.clone()), None),
                STATE_TAG => (InputBinding::Scalar(rows.state.clone()), None),
                PUBLIC_TAG => {
                    let (ring, labels) = (ring.clone(), labels.clone());
                    let (label_rows, label_width) = (plan.label_rows, plan.label_width);
                    (
                        InputBinding::Family(GraphFamily::Values(rows.public.clone())),
                        Some(GraphFamily::computed(label_rows + 1, move |member| {
                            let entry =
                                labels.at(member.clone().rem(label_rows).mul(label_width + 1));
                            let exponent = mxx_dsl::select(
                                member.equal(label_rows).to_int(),
                                vec![entry.mul(stride), Int::constant(0)],
                            )?;
                            Ok(ring.identity(1).multiply_monomial(exponent))
                        })),
                    )
                }
                XBITS_TAG => {
                    let (row_function, one, row_chunk) =
                        (function.clone(), one.clone(), chunk.clone());
                    let (payload_function, payload_chunk, ring) =
                        (function.clone(), chunk.clone(), ring.clone());
                    (
                        InputBinding::Family(GraphFamily::computed(
                            function.eval_chunk_bits,
                            move |bit| xbits_row(&row_function, &one, bit, row_chunk.clone()),
                        )),
                        Some(GraphFamily::computed(function.eval_chunk_bits, move |bit| {
                            Ok(ring.identity(1).multiply_monomial(xbits_exponent(
                                &payload_function,
                                bit,
                                payload_chunk.clone(),
                            )?))
                        })),
                    )
                }
                _ => {
                    return Err(Dio2Error::Input("a production Program reads only step input rows"))
                }
            };
            graph_rows.insert(*wire, row);
            if let Some(payload) = payload {
                payloads.insert(*wire, InputBinding::Family(payload));
            }
        }
        build.bind_rows(
            &program.program,
            &ParamEnv::default(),
            &obfuscation.preprocessing(kind)?,
            graph_rows,
            payloads,
            &stage_key,
        )?;
        build.bind_lut_helper_rows(self.helper_rows(&rows.helpers)?)?;
        let lowered = build.lower_deferred(&program.program, &program.deferred_outputs())?;
        let (outputs, deferred) = split_deferred(lowered);
        let row =
            |output: RnsBggEncoding| (output.row().clone(), output.public_key().matrix().clone());
        let build = RefCell::new(build);
        let blocks = |output: usize, block: IntExpr| -> Result<Row, Dio2Error> {
            let columns =
                deferred.get(&output).ok_or(Dio2Error::Input("an output is not deferred"))?;
            Ok(row(build.borrow_mut().deferred_block(&program.program, columns, &block)?))
        };
        let composed = compose(&Compose::new(
            self.parameters,
            chain,
            plan,
            program,
            outputs.into_iter().map(|output| output.map(row)).collect(),
            Box::new(blocks),
        )?)?;
        self.runtime.extend(build.into_inner().runtime_inputs());
        Ok(composed)
    }

    /// The LUT helper rows of every helper family of the plan: branch `b` of a family is the
    /// helper `HLP[k]` of its `b`-th automorphism `k`, the member of `helpers` at the position of
    /// `k` in [`super::step::StepPlan::automorphisms`].
    #[allow(clippy::type_complexity)]
    fn helper_rows(
        &self,
        helpers: &Family<RnsBggRow>,
    ) -> Result<BTreeMap<String, (Family<RnsBggRow>, Vec<IntExpr>)>, Dio2Error> {
        let plan = &self.obfuscation.plan;
        plan.helper_families
            .iter()
            .map(|(key, sigmas)| {
                let positions = sigmas
                    .iter()
                    .map(|sigma| {
                        plan.automorphisms
                            .iter()
                            .position(|known| known == sigma)
                            .map(IntExpr::constant)
                            .ok_or(Dio2Error::Parameters("a helper automorphism is unplanned"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((key.clone(), (helpers.clone(), positions)))
            })
            .collect()
    }

    /// Refreshing step `step` on the step input `rows` and the labels `labels`: the next step
    /// input and labels.
    fn refresh(
        &mut self,
        step: Int,
        rows: &StepRows,
        labels: &Family<Int>,
    ) -> Result<(StepRows, Family<Int>), Dio2Error> {
        let obfuscation = self.obfuscation;
        let (plan, function) = (&obfuscation.plan, &obfuscation.function);
        let chunk = self.chunks.at(step.clone());
        let root = step_root(&self.root, function, plan, step.clone(), Some(chunk.clone()));
        let chain = step_chain(self.parameters, &root, (step.clone(), chunk.clone()))?;
        let next = NextStep {
            root: step_root(&self.root, function, plan, step.clone().add(1usize), None),
            hash_key: self.root.hash_key_input(),
            anchor: self.anchors().at(step.clone().add(1usize)),
        };
        // The instance's preimages, read by their index in the family of every instance's.
        let per = plan.blocks() * plan.primes;
        let preimages = self.preimages(Kind::Refreshing, per, plan.columns)?;
        let first = Kind::Refreshing.member(function, &step, &chunk).mul(per);
        let columns = &self.columns.clone();
        let rows =
            self.compose(Kind::Refreshing, (&step, &chunk), (rows, labels), &chain, |compose| {
                compose.next_rows(
                    &next,
                    (columns, chunk.clone(), labels),
                    rows,
                    (&preimages, first),
                )
            })?;
        let products = LabelProducts::new(self.parameters, function.eval_chunk_bits);
        let labels = products.next(self.parameters, &self.columns, labels, chunk)?;
        Ok((rows, labels))
    }

    /// The last step on the step input `rows` and the labels `labels`: its output decoder's
    /// plaintext read as the final state.
    fn output(&mut self, rows: &StepRows, labels: &Family<Int>) -> Result<Int, Dio2Error> {
        let obfuscation = self.obfuscation;
        let (plan, function) = (&obfuscation.plan, &obfuscation.function);
        let step = Int::constant(function.stream_length() - 1);
        let chunk = self.chunks.at(step.clone());
        let root = step_root(&self.root, function, plan, step.clone(), Some(chunk.clone()));
        let chain = step_chain(self.parameters, &root, (step.clone(), chunk.clone()))?;
        let preimage = self.preimages(Kind::Last, 1, 1)?.at(chunk.clone());
        let plaintext =
            self.compose(Kind::Last, (&step, &chunk), (rows, labels), &chain, |compose| {
                compose.output_plaintext(&rows.hint, preimage)
            })?;
        let q = self.parameters.primes[0];
        decode_state(
            plaintext,
            q,
            self.parameters.ring_dimension as usize,
            function.state_layout.stride(),
        )
    }
}

/// The state `z` of the plaintext `mono(z) = X^(z stride)` modulo the odd prime `q` (`1 x 1`, ring
/// dimension `n`): with every coefficient `c_k` centered to `v_k in {-1, 0, 1}`, the exponent
/// `sum_k (k v_k (v_k + 1) / 2 + (k + n) v_k (v_k - 1) / 2)` of the one nonzero coefficient
/// (negative past `X^n`), divided by `stride`. One loop over the coefficients, each extracted
/// where it is read, so the graph does not grow with `n` and no family of coefficients is formed.
fn decode_state(plaintext: Mat, q: u64, n: usize, stride: usize) -> Result<Int, Dio2Error> {
    let exponent = mxx_dsl::iterate(n, Int::constant(0), |k, total| {
        let c = plaintext.clone().extract_coefficient(k.expression()?);
        let v = c.clone().sub(c.mul(2).div(q as i64).mul(q as i64));
        let up = v.clone().mul(v.clone().add(1)).div(2);
        let down = v.clone().mul(v.sub(1)).div(2);
        Ok(total.add(k.clone().mul(up)).add(k.add(n as i64).mul(down)))
    })?;
    Ok(exponent.div(stride as i64))
}
