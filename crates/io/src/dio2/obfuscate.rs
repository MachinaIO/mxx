//! The obfuscator (spec §7): one graph, executed once, whose outputs (one artifact production)
//! are the obfuscation.
//!
//! The graph samples the payload secret `t`, the binary mask `s_0` and one anchor trapdoor per
//! step (a parallel loop over the steps), and encrypts the step input `S_0` at the root modulus
//! (the only encryption: `t` and `s_0` are read by nothing else but the Static preprocessing
//! ciphertexts under `t`). Then, for every instance of each Program kind ([`Kind`]: every
//! refreshing step and chunk, and every chunk of the last step) as one parallel loop, it lowers
//! the Static preprocessing of the uniform step Program: the RHS plaintexts (the step's
//! obfuscator monomials, the chunk's PRF selectors and the sampler offsets) and the integer data
//! (the step's row, the chunk's coordinate tables) are graph values selected by the loop index,
//! and the Program's hashed public matrices carry the instance's hash domain. The refresh's
//! decoded keys of the instance come from the key-only composition ([`Compose`] on the lowered
//! public keys): no encryption, no online evaluation, no payload or mask. The anchor preimages
//! of every part and prime are sampled in a parallel loop.
//!
//! Every per-instance artifact role (an RHS ciphertext, LUT-switching ciphertexts, the
//! preimages) is published as one family: an instance's matrices side by side, so the evaluator
//! reads instance `(i, x_i)` by its runtime index (`RnsOnlineEvalBuild::with_material_member`).

use super::{
    Dio2, Dio2Error, Dio2Function, Dio2Parameters, Failure,
    carrier::Key,
    compose::{Compose, NextStep, split_deferred},
    labels::{LabelColumns, LabelProducts},
    scheme::{
        ANCHORS_OUTPUT, Dio2Obfuscation, INITIAL_OUTPUT, INITIAL_STATE_INPUT, Kind, KindMaterial,
        LABELS_INPUT, OBFUSCATE_STAGE, STEP_DATA_INPUT, Y_INPUT, derive, record, selector_input,
        step_chain, step_root, table_input,
    },
    step::{RootObject, StepPlan, StepRows},
};
use mxx_backends::{RuntimeValue, SessionStore};
use mxx_dsl::{DslContext, Family, Int, Leaf, Mat, Trapdoor};
use mxx_exponent_lut::{
    api::{self, GraphEmission, ProgramBuild, StageExecution},
    decoder::unmask::{sample_trapdoor, unmask_hint},
    program::InputBinding,
    rns_bgg::{
        RhsPlaintexts, RnsBggParameters, RnsBggPayloadSecret, RnsBggPreprocessOnlineEvalDescriptor,
        RnsBggRoot, RnsBggRow, RnsOutputUse, RnsPreprocessBuild,
    },
};
use mxx_ir_core::{IntExpr, ParamEnv, node::ConcatAxis};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{cell::RefCell, collections::BTreeMap};

/// The secrets and public matrices of an obfuscation graph that every part reads.
struct Secrets {
    /// `t_bar`, uniform ternary.
    payload: Mat,
    /// `s_0 = (s_bar_0, 1)`, `s_bar_0` uniform binary.
    mask: RnsBggRow,
    /// The anchor trapdoor `T_i` of every step.
    anchors: Family<Trapdoor>,
}

/// The construction of an obfuscation graph.
struct Obfuscator<'a> {
    parameters: &'a Dio2Parameters,
    input: &'a Dio2Function,
    plan: &'a StepPlan,
    /// The root of the obfuscation (step roots add their domain).
    root: RnsBggRoot,
    hash_key: [u8; 32],
    input_storage: &'a super::input_storage::InputStorage,
    columns: LabelColumns,
    /// The resident inputs of the graph's builders (hash keys).
    runtime: BTreeMap<String, RuntimeValue>,
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// The obfuscation of `input` whose public matrices and namespaces are derived from
    /// `hash_key`: an obfuscation keyed by a fixed key builds the same graph on every run (its
    /// claim's export is reproducible); [`crate::Obfuscation::obfuscate`] samples the key.
    pub fn obfuscate_with_hash_key(
        &mut self,
        input: &Dio2Function,
        hash_key: [u8; 32],
    ) -> Result<Dio2Obfuscation, Dio2Error> {
        let function = &input.function;
        let steps = function.stream_length();
        if steps == 0 ||
            input.y.len() != steps ||
            input.y.iter().any(|chunk| chunk.len() != function.obf_chunk_bits) ||
            input.initial_state >= 1 << function.state_bits
        {
            return Err(Dio2Error::Input(
                "the obfuscator's chunks or state do not fit the function",
            ));
        }
        let parameters = self.parameters.clone();
        let plan = StepPlan::new(&parameters, function)?;
        let kinds = if steps > 1 { vec![Kind::Refreshing, Kind::Last] } else { vec![Kind::Last] };
        let programs = kinds
            .iter()
            .map(|kind| kind.program(&parameters, function, &plan))
            .collect::<Result<Vec<_>, _>>()?;
        let input_storage = super::input_storage::InputStorage::new(
            &hash_key,
            function,
            &kinds.iter().copied().zip(programs.iter()).collect::<Vec<_>>(),
            self.inputs_by_shape,
        );
        let mut obfuscator = Obfuscator {
            parameters: &parameters,
            input,
            plan: &plan,
            root: parameters.root(derive(&hash_key, "root", &[]))?,
            hash_key,
            input_storage: &input_storage,
            columns: LabelProducts::new(&parameters, function.eval_chunk_bits).columns(),
            runtime: BTreeMap::new(),
        };
        let secrets = obfuscator.secrets()?;
        let mut graph = DslContext::new("dio2-obfuscate")
            .transferred_output(INITIAL_OUTPUT, obfuscator.initial(&secrets)?.into_value())?
            .transferred_output(ANCHORS_OUTPUT, secrets.anchors.public_matrices())?;
        let mut descriptors = Vec::new();
        for (kind, program) in kinds.iter().zip(&programs) {
            let (leaves, descriptor) = obfuscator.material(*kind, program, &secrets)?;
            for (name, leaf) in leaves {
                graph = graph.transferred_output(name, leaf)?;
            }
            descriptors.push(descriptor);
        }
        let mut runtime = std::mem::take(&mut obfuscator.runtime);
        let public = if self.inputs_by_shape {
            BTreeMap::new()
        } else {
            let (mut public, mut secret) = self.key_values(input, &plan, &kinds, &programs)?;
            if let Err(error) = input_storage.publish(&mut self.store, &mut public, &mut secret) {
                input_storage.discard_secret(&mut self.store)?;
                return Err(error);
            }
            runtime.extend(secret);
            public
        };
        runtime.extend(public.clone());
        let graph = GraphEmission {
            graph,
            runtime_inputs: runtime,
            manifests: input_storage.manifests(true),
            descriptor: (),
        }
        .build();
        let (graph, ()) = match graph {
            Ok(graph) => graph,
            Err(error) => {
                if !self.inputs_by_shape {
                    input_storage.discard_secret(&mut self.store)?;
                }
                return Err(error.into());
            }
        };
        let produced = api::execute_artifacts(
            graph,
            &ParamEnv::default(),
            (),
            derive(&hash_key, "obfuscation", &[]),
            &mut self.execution,
            &mut self.store,
            record(&mut self.graphs, OBFUSCATE_STAGE),
        );
        if !self.inputs_by_shape {
            input_storage.discard_secret(&mut self.store)?;
        }
        let production = produced?;
        self.productions.insert(production.production().clone(), OBFUSCATE_STAGE.to_owned());
        let mut materials = programs
            .into_iter()
            .zip(descriptors)
            .map(|(program, descriptor)| KindMaterial { program, descriptor })
            .collect::<Vec<_>>();
        let last = materials.pop().expect("the last kind");
        Ok(Dio2Obfuscation {
            function: function.clone(),
            plan,
            hash_key,
            production,
            refreshing: materials.pop(),
            last,
            public,
            input_storage: input_storage.public_only(),
        })
    }

    /// The values of the graph inputs the obfuscator samples or knows: the public ones both
    /// graphs read (the labels `A_0`, the step rows, the PRF coordinate tables of every chunk),
    /// and the obfuscator's own (the initial state, its bits, the PRF selectors).
    #[allow(clippy::type_complexity)]
    fn key_values(
        &self,
        input: &Dio2Function,
        plan: &StepPlan,
        kinds: &[Kind],
        programs: &[super::step::ChainProgram],
    ) -> Result<(BTreeMap<String, RuntimeValue>, BTreeMap<String, RuntimeValue>), Dio2Error> {
        let function = &input.function;
        let integers = |values: Vec<usize>| {
            RuntimeValue::integer_values(values.into_iter().map(Into::into).collect())
        };
        let q = self.parameters.label_modulus();
        let labels = (0..plan.label_rows * (plan.label_width + 1))
            .map(|_| rand::rng().random_range(0..q))
            .collect::<Vec<_>>();
        let rows = (0..function.stream_length())
            .flat_map(|step| function.step_row(step).iter().copied())
            .collect::<Vec<_>>();
        let mut public = BTreeMap::from([
            (LABELS_INPUT.to_owned(), integers(labels)),
            (STEP_DATA_INPUT.to_owned(), integers(rows)),
        ]);
        let bits = input.y.iter().flatten().map(|bit| usize::from(*bit)).collect();
        let mut secret = BTreeMap::from([
            (INITIAL_STATE_INPUT.to_owned(), RuntimeValue::Int(input.initial_state.into())),
            (Y_INPUT.to_owned(), integers(bits)),
        ]);
        // Keys of every chunk (spec §4.5, §4.6): indexed by the chunk, shared by every step.
        let mut rng = StdRng::from_rng(&mut rand::rng());
        for (kind, program) in kinds.iter().zip(programs) {
            for (role, prf) in kind.prfs(program) {
                let mut selectors = vec![Vec::new(); prf.bucket_count()];
                let mut tables = vec![Vec::new(); prf.bucket_count()];
                for _ in 0..1usize << function.eval_chunk_bits {
                    let keys = self
                        .parameters
                        .prf
                        .generate_keys(&mut rng, prf.members())
                        .map_err(|_| Dio2Error::Parameters("PRF keys"))?;
                    let (bits, data) = prf.key_values(&keys)?;
                    for (bucket, bits) in bits.into_iter().enumerate() {
                        selectors[bucket].extend(bits.into_iter().map(usize::from));
                        let table = data
                            .values(prf.tables()[bucket])
                            .ok_or(Dio2Error::Parameters("a PRF table"))?;
                        tables[bucket].extend_from_slice(table);
                    }
                }
                for (bucket, (selectors, table)) in selectors.into_iter().zip(tables).enumerate() {
                    secret.insert(selector_input(role, bucket), integers(selectors));
                    public.insert(table_input(role, bucket), integers(table));
                }
            }
        }
        Ok((public, secret))
    }
}

impl Obfuscator<'_> {
    /// The ring of `Q_0`.
    fn ring(&self) -> mxx_dsl::Ring {
        self.root.ring()
    }

    /// The secrets: `t_bar` uniform ternary, `s_bar_0` uniform binary, and the anchor trapdoors.
    fn secrets(&self) -> Result<Secrets, Dio2Error> {
        let ring = self.ring();
        let bound = super::preimage_bound(self.parameters)?;
        let anchors = mxx_dsl::parallel(self.input.function.stream_length(), |_| {
            Ok(sample_trapdoor(
                &ring,
                mxx_ir_core::RealExpr::from(self.parameters.trapdoor_sigma),
                1u64 << self.parameters.base_bits,
                self.plan.columns,
                num_bigint::BigInt::from(bound.clone()),
            ))
        })?;
        Ok(Secrets {
            payload: ring.uniform_interval((1, 1), -1, 1),
            mask: (ring.uniform_interval((1, 1), 0, 1), ring.identity(1)),
            anchors,
        })
    }

    /// A CBD(1) error row of `columns` columns.
    fn error(&self, columns: usize) -> Mat {
        let ring = self.ring();
        ring.uniform_interval((1, columns), 0, 1) - ring.uniform_interval((1, columns), 0, 1)
    }

    /// The step input `S_0` (spec §7): every root object of step 0 under the mask `s_0` with
    /// CBD(1) errors, the targets `(t_bar, 1)` (`ONEPRIV`), `mono(z_0) (t_bar, 1)` (state),
    /// `mono(a) s_0` (labels and `ONEPUB`) and `(sigma_k(s_bar_0), 1)` (helpers), and the hint
    /// `b_0 = s_0 B_0 + e`.
    fn initial(&mut self, secrets: &Secrets) -> Result<StepRows, Dio2Error> {
        let (plan, ring) = (self.plan, self.ring());
        let root = self.root.clone().with_domain(vec![Int::constant(0)]);
        let hash_key = root.hash_key_input();
        self.runtime.insert(
            root.hash_key_input_name(),
            RuntimeValue::Bytes(root.hash_key().as_slice().into()),
        );
        let s = &secrets.mask;
        let t = (secrets.payload.clone(), ring.identity(1));
        let labels =
            mxx_dsl::int_family_input(LABELS_INPUT, plan.label_rows * (plan.label_width + 1));
        let initial_state = mxx_dsl::DslContext::new("dio2-initial")
            .input::<Int>(INITIAL_STATE_INPUT, mxx_dsl::IntType)?;
        let prf_stride = self.parameters.profile()?.layout.stride();
        let state_stride = self.input.function.state_layout.stride();
        let monomial = |exponent: Int| ring.identity(1).multiply_monomial(exponent);
        let times =
            |value: Mat, row: &RnsBggRow| (value.clone() * row.0.clone(), value * row.1.clone());
        let failure = Failure::default();
        let encode = |object: RootObject, member: Option<Int>, target: (Mat, Mat)| {
            let key = plan.root_key(&root, hash_key.clone(), object, member)?;
            Ok(root.encode(
                &key,
                s,
                target,
                (self.error(plan.columns), self.error(plan.columns)),
            )?)
        };
        let private = encode(RootObject::Private, None, times(ring.identity(1), &t))?;
        let state =
            encode(RootObject::State, None, times(monomial(initial_state.mul(state_stride)), &t))?;
        let public = mxx_dsl::parallel(plan.label_rows + 1, |member| {
            let entry = labels.at(member.clone().rem(plan.label_rows).mul(plan.label_width + 1));
            let exponent = mxx_dsl::select(
                member.clone().equal(plan.label_rows).to_int(),
                vec![entry.mul(prf_stride), Int::constant(0)],
            )?;
            failure.keep(encode(RootObject::Public, Some(member), times(monomial(exponent), s)))
        });
        let label_rows = mxx_dsl::parallel(plan.label_rows * plan.label_width, |member| {
            let entry = labels.at(member
                .clone()
                .div(plan.label_width)
                .mul(plan.label_width + 1)
                .add(member.clone().rem(plan.label_width))
                .add(1usize));
            failure.keep(encode(
                RootObject::Label,
                Some(member),
                times(monomial(entry.mul(prf_stride)), s),
            ))
        });
        let helpers = mxx_dsl::parallel(plan.automorphisms.len(), |member| {
            let sigma = plan.automorphism(member.expression()?);
            let target = (s.0.clone().ring_automorphism(sigma.expression()?), ring.identity(1));
            failure.keep(encode(RootObject::Helper, Some(member), target))
        });
        let (public, labels, helpers) =
            (failure.finish(public)?, failure.finish(label_rows)?, failure.finish(helpers)?);
        let anchor_columns = super::anchor_columns(plan.columns);
        let hint = unmask_hint(
            Mat::concat(ConcatAxis::Columns, vec![s.0.clone(), s.1.clone()]),
            secrets.anchors.public_matrices().at(Int::constant(0)),
            self.error(anchor_columns),
        );
        Ok(StepRows { private, state, public, labels, helpers, hint })
    }

    /// The RHS plaintexts of the instance of step `step` on chunk `chunk`: the monomials
    /// `mono(y_monomial(b, y_b))` of the step's obfuscator bits, the chunk's PRF selector bits
    /// (constant polynomials) and the samplers' offsets (one).
    fn rhs_values(
        &self,
        kind: Kind,
        program: &super::step::ChainProgram,
        (step, chunk): (&Int, &Int),
    ) -> Result<
        BTreeMap<mxx_exponent_lut::program::RhsId, InputBinding<Mat, RhsPlaintexts>>,
        Dio2Error,
    > {
        let function = &self.input.function;
        let ring = self.ring();
        let obf = function.obf_chunk_bits;
        let y = mxx_dsl::int_family_input(Y_INPUT, function.stream_length() * obf);
        let first = step.clone().mul(obf);
        let mut rhs = BTreeMap::from([(
            program.y,
            InputBinding::Family(RhsPlaintexts::Values(mxx_dsl::parallel(obf, |bit| {
                let set = y.at(first.clone().add(bit.clone()));
                Ok(ring.identity(1).multiply_monomial((function.y_monomial)(bit, set)?))
            })?)),
        )]);
        for (role, prf) in kind.prfs(program) {
            for (bucket, selector) in prf.selector_families().into_iter().enumerate() {
                let count = IntExpr::constant(selector.members);
                let chunks = IntExpr::constant(1usize << function.eval_chunk_bits);
                let bits = self.input_storage.family(
                    selector_input(role, bucket),
                    IntExpr::Mul(Box::new(chunks), Box::new(count.clone())).canonicalize(),
                )?;
                // The chunk's selector bits, lifted where each member's ciphertext is sampled.
                let first = chunk.clone().mul(Int::evaluate(count.clone()));
                rhs.insert(
                    selector.rhs,
                    InputBinding::Family(RhsPlaintexts::Lifted { integers: bits, first, count }),
                );
            }
        }
        for sampler in program.sigma.iter().chain(&program.cbd) {
            for offset in sampler.offset_rhs() {
                rhs.insert(*offset, InputBinding::Scalar(ring.identity(1)));
            }
        }
        Ok(rhs)
    }

    /// The obfuscation's material of `kind`: for every instance (a parallel loop), the
    /// Static preprocessing of `program` and the anchor preimages of its decoded keys, each
    /// artifact role as one family of the instances' values side by side. Returns the published
    /// leaves by artifact name and the descriptor of the preprocessing.
    fn material(
        &mut self,
        kind: Kind,
        program: &super::step::ChainProgram,
        secrets: &Secrets,
    ) -> Result<(Vec<(String, Leaf)>, RnsBggPreprocessOnlineEvalDescriptor), Dio2Error> {
        let function = &self.input.function;
        let members = kind.members(function);
        let (context, stage_key) = kind.identity(&self.hash_key);
        let failure = Failure::default();
        let mut published = None;
        let mut runtime = BTreeMap::new();
        let instances = mxx_dsl::parallel(members, |member| {
            let (step, chunk) = kind.instance(function, &member);
            let root =
                step_root(&self.root, function, self.plan, step.clone(), Some(chunk.clone()));
            let chain =
                failure.keep(step_chain(self.parameters, &root, (step.clone(), chunk.clone())))?;
            let (leaves, descriptor, inputs) = failure.keep(self.instance(
                kind,
                program,
                &chain,
                (&step, &chunk),
                (context, &stage_key),
                secrets,
            ))?;
            runtime = inputs;
            let (names, values): (Vec<_>, Vec<_>) = leaves.into_iter().unzip();
            published = Some((names, descriptor));
            Ok(values)
        });
        let instances = failure.finish(instances)?;
        self.runtime.extend(runtime);
        let (names, descriptor) =
            published.ok_or(Dio2Error::Input("a Program kind without instances"))?;
        let leaves = names
            .into_iter()
            .enumerate()
            .map(|(position, name)| {
                let family = instances.field(|leaves| leaves[position].clone())?;
                Ok((name, flatten(family)?))
            })
            .collect::<Result<Vec<_>, Dio2Error>>()?;
        Ok((leaves, descriptor))
    }

    /// One instance of `kind`'s material (see [`Self::material`]) on the instance's chain
    /// `chain`: its published artifact leaves (the Static RHS and LUT-switching ciphertexts,
    /// then the preimages), the preprocessing descriptor and the resident inputs it binds.
    #[allow(clippy::type_complexity)]
    fn instance(
        &self,
        kind: Kind,
        program: &super::step::ChainProgram,
        chain: &RnsBggParameters,
        (step, chunk): (&Int, &Int),
        (context, stage_key): ([u8; 32], &[u8; 32]),
        secrets: &Secrets,
    ) -> Result<
        (Vec<(String, Leaf)>, RnsBggPreprocessOnlineEvalDescriptor, BTreeMap<String, RuntimeValue>),
        Dio2Error,
    > {
        let function = &self.input.function;
        let plan = self.plan;
        let mut build = RnsPreprocessBuild::new(chain, context)
            .with_artifact_prefix(kind.prefix())
            .with_root_inputs(program.root_inputs.clone())
            .with_output_use(RnsOutputUse::Gadget);
        build.bind_data_values(
            &program.program,
            kind.data(function, program, (step, chunk), self.input_storage)?,
        )?;
        let payload = RnsBggPayloadSecret::from_row(chain, secrets.payload.clone())?;
        build.bind_rhs_values(
            &program.program,
            &ParamEnv::default(),
            self.rhs_values(kind, program, (step, chunk))?,
            Some(&payload),
            stage_key,
        )?;
        let lowered = build.lower_deferred(&program.program, &program.deferred_outputs())?;
        let (keys, deferred) = split_deferred(lowered);
        let build = RefCell::new(build);
        let blocks = |output: usize, block: IntExpr| -> Result<Key, Dio2Error> {
            let columns =
                deferred.get(&output).ok_or(Dio2Error::Input("an output is not deferred"))?;
            Ok(build
                .borrow_mut()
                .deferred_block(&program.program, columns, &block)?
                .matrix()
                .clone())
        };
        let compose = Compose::<Key>::new(
            self.parameters,
            chain,
            plan,
            program,
            keys.iter().map(|key| key.as_ref().map(|key| key.matrix().clone())).collect(),
            Box::new(blocks),
        )?;
        // The step's anchor trapdoor is selected where each preimage is sampled, with its public
        // matrix at the same index.
        let trapdoor = || secrets.anchors.at(step.clone());
        let anchor = super::anchor_columns(plan.columns);
        let preimages: Leaf = match kind {
            Kind::Last => {
                let key = compose.output_key()?;
                Leaf::of(trapdoor().sample_preimage(key, (anchor, 1))).expect("one wire")
            }
            Kind::Refreshing => {
                let next = NextStep {
                    root: step_root(&self.root, function, plan, step.clone().add(1usize), None),
                    hash_key: self.root.hash_key_input(),
                    anchor: secrets.anchors.public_matrices().at(step.clone().add(1usize)),
                };
                let preimages =
                    compose.decoded_keys(&next, (&self.columns, chunk.clone()), |key| {
                        trapdoor().sample_preimage(key, (anchor, plan.columns))
                    })?;
                Leaf::of(preimages).expect("one wire")
            }
        };
        drop(compose);
        let emission = build
            .into_inner()
            .emit(DslContext::new("dio2-material"), keys.into_iter().flatten().collect())?;
        let prefix = kind.prefix();
        let mut leaves = emission
            .graph
            .into_outputs()
            .into_iter()
            .filter(|(name, _, _)| {
                ["rhs/", "lut-switching/"]
                    .iter()
                    .any(|role| name.starts_with(&format!("{prefix}/{role}")))
            })
            .map(|(name, leaf, _)| (name, leaf))
            .collect::<Vec<_>>();
        leaves.push((kind.preimages(), preimages));
        Ok((leaves, emission.descriptor.1, emission.runtime_inputs))
    }
}

/// The family of every instance's values of one artifact role side by side: the instances'
/// matrices, or, for a role that is itself a family of `count` matrices, the `members count`
/// matrices of the instances in order.
fn flatten(instances: Family<Leaf>) -> Result<Leaf, Dio2Error> {
    let element = instances.at(Int::constant(0));
    let mxx_ir_core::types::WireType::IndexedFamily { count, .. } = element.wire_type().clone()
    else {
        return Ok(Leaf::of(instances).expect("one wire"));
    };
    let members =
        IntExpr::Mul(Box::new(instances.count().clone()), Box::new(count.clone())).canonicalize();
    let per = Int::evaluate(count);
    let flat = mxx_dsl::parallel(members, |member| {
        instances
            .at(member.clone().div(per.clone()))
            .at(member.rem(per.clone()))
            .ok_or(mxx_dsl::DslError::Schema)
    })?;
    Ok(Leaf::of(flat).expect("one wire"))
}
