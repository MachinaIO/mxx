//! The obfuscator and the evaluator of Diamond iO 2 (spec §5–§8) on any stage executor.
//!
//! [`Dio2::obfuscate`] samples the payload secret `t`, the binary mask `s_0`, the initial labels
//! `A_0`, the PRF keys of every chunk `x`, and one anchor trapdoor per step (spec §7). It
//! encrypts the step input `S_0` at the root modulus, and for every step and chunk it
//! preprocesses the step's Program, runs its composed refresh on zero rows to obtain the decoded
//! key of every part and prime (the keys do not depend on the rows), and samples their preimages
//! under the step's anchor.
//!
//! [`Dio2::evaluate`] runs, for every step, the chunk's Program on the imported rows of `S_i`,
//! whose composed refresh unmasks and rounds every part for every prime and recombines the
//! residues into the rows of `S_(i+1)` (spec §6.8); the last step's output decoder returns the
//! state monomial modulo the first prime, read as the final state.

use super::{
    Dio2Error, Dio2Parameters, StreamingFunction, anchor_columns,
    refresh::{
        ComposeMode, HINT_INPUT, NEXT_ANCHOR_INPUT, OUTPUT_KEY_OUTPUT, OUTPUT_PLAINTEXT_OUTPUT,
        OUTPUT_PREIMAGE_INPUT, StepCompose, key_output, preimage_input,
    },
    step::{ChainKeys, ChainProgram, LABEL_TAG, PUBLIC_TAG, StepPlan, Target, next_labels},
};
use crate::Obfuscation;
use mxx_backends::{
    RuntimeValue, SessionStore,
    poly::{
        Poly, PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
};
use mxx_dsl::{DslContext, Int, Mat, Ring};
use mxx_exponent_lut::{
    api::{self, ApiError, PlaintextInputs, StageArtifacts, StageExecution, StageGraph},
    decoder::unmask::{UNMASK_ROWS, sample_trapdoor, unmask_hint},
    program::{InputBinding, ProgramData, RhsId, WireId},
    program_gadgets::pbc::PbcGeneratedKeyLayout,
    rns_bgg::{
        RnsBggMaskSecret, RnsBggParameters, RnsBggPayloadSecret, RnsBggPreprocessEncryptDescriptor,
        RnsBggPreprocessOnlineEvalDescriptor, RnsBggRoot, RnsEncryptBuild, RnsOnlineEvalBuild,
        RnsOutputUse, RnsPreprocessBuild, root_helper_input_name, root_row_input_name,
    },
};
use mxx_ir_core::{
    Graph, IntExpr, ParamEnv, ValidatedGraph, artifact::ProductionId, node::ConcatAxis,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The obfuscator's input: a streaming function, its chunk `y_i` of every step, and the initial
/// state `z_0`.
#[derive(Clone)]
pub struct Dio2Function {
    pub function: StreamingFunction,
    pub y: Vec<Vec<bool>>,
    pub initial_state: usize,
}

/// A step input `S_i` (spec §5): the root rows at `Q_0` by the graph input names an imported
/// encryption reads ([`StepPlan::input_rows`]), the anchor hint ([`HINT_INPUT`]), and the clear
/// labels.
#[derive(Clone)]
pub struct StepInput {
    values: BTreeMap<String, RuntimeValue>,
    labels: Vec<Vec<usize>>,
}

/// The graph input of the initial state `z_0` (the initial input stage).
pub const INITIAL_STATE_INPUT: &str = "dio2/initial-state";
/// The graph input of the initial labels `A_0`, entry `(r, c)` at `r (m_v + 1) + c`.
pub const INITIAL_LABELS_INPUT: &str = "dio2/initial-labels";
/// The graph input of the first anchor `B_0` (the initial input stage).
pub const ANCHOR_INPUT: &str = "dio2/anchor";
/// The graph input of the output decoder's plaintext (the decode stage).
pub const PLAINTEXT_INPUT: &str = "dio2/plaintext";
/// The graph output of the decoded final state (the decode stage).
pub const STATE_OUTPUT: &str = "dio2/state";

/// The output of step `step`'s anchor `B_step` (the anchors' public stage).
pub fn anchor_output(step: usize) -> String {
    format!("anchor-public/{step}")
}

/// The material of a step and chunk: its Program, the Program's data, its preprocessing, and the
/// preimages of its parts and primes (or of its output decoder).
pub struct StepMaterial {
    pub(crate) program: ChainProgram,
    data: ProgramData,
    preprocess: (
        StageArtifacts<RnsBggPreprocessEncryptDescriptor>,
        StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>,
    ),
    preimages: BTreeMap<String, RuntimeValue>,
}

/// An obfuscation (spec §7): the initial step input, the anchors, and the material of every step
/// and chunk.
pub struct Dio2Obfuscation {
    pub(crate) function: StreamingFunction,
    pub(crate) plan: StepPlan,
    hash_key: [u8; 32],
    initial: StepInput,
    anchors: Vec<RuntimeValue>,
    pub(crate) steps: Vec<Vec<StepMaterial>>,
}

/// A Diamond iO 2 instance: its parameters and the executor and artifact store its stages run
/// on, whose backend hosts every ring of [`Dio2Parameters::rings`]. It records the graph of every
/// stage it runs, by stage name, and the stage of every production, for the claim of a run
/// ([`super::claim`]).
pub struct Dio2<E, S> {
    parameters: Dio2Parameters,
    execution: E,
    store: S,
    pub(crate) graphs: BTreeMap<String, Graph>,
    pub(crate) productions: BTreeMap<ProductionId, String>,
    /// The stage output every stage input of a sampled secret reads, by input name.
    pub(crate) secrets: BTreeMap<String, (String, String)>,
}

/// A stage observer that records the validated graph as stage `stage` in `graphs`.
fn record(
    graphs: &mut BTreeMap<String, Graph>,
    stage: String,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), ApiError> + '_ {
    move |validated| {
        graphs.insert(stage, validated.source.clone());
        Ok(())
    }
}

/// The key `H(hash_key, role, indices)` of a role of the obfuscation.
fn derive(hash_key: &[u8; 32], role: &str, indices: &[usize]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-io/dio2/derive/v1");
    digest.update(hash_key);
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    for index in indices {
        digest.update((*index as u64).to_le_bytes());
    }
    digest.finalize().into()
}

/// The chunk `bits` as an index.
pub(crate) fn chunk_index(bits: &[bool]) -> usize {
    bits.iter().enumerate().map(|(bit, set)| usize::from(*set) << bit).sum()
}

/// The chunk of index `index`.
fn chunk_bits(index: usize, bits: usize) -> Vec<bool> {
    (0..bits).map(|bit| index >> bit & 1 == 1).collect()
}

/// `X^exponent` in the negacyclic ring of `parameters`.
fn monomial(parameters: &DCRTPolyParams, exponent: usize) -> DCRTPoly {
    let dimension = parameters.ring_dimension() as usize;
    let rotation = exponent % (2 * dimension);
    let value = DCRTPoly::const_rotate_poly(parameters, rotation % dimension);
    if rotation >= dimension { -value } else { value }
}

fn output(
    outputs: &mut BTreeMap<String, RuntimeValue>,
    name: &str,
) -> Result<RuntimeValue, Dio2Error> {
    outputs.remove(name).ok_or(Dio2Error::Input("a stage output is missing"))
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// An instance on `parameters` running on `execution` and `store`.
    pub fn new(parameters: Dio2Parameters, execution: E, store: S) -> Result<Self, Dio2Error> {
        parameters.validate()?;
        Ok(Self {
            parameters,
            execution,
            store,
            graphs: BTreeMap::new(),
            productions: BTreeMap::new(),
            secrets: BTreeMap::new(),
        })
    }

    pub fn parameters(&self) -> &Dio2Parameters {
        &self.parameters
    }

    /// The root of step `step` of the obfuscation keyed by `hash_key`.
    fn root(&self, hash_key: &[u8; 32], step: usize) -> Result<RnsBggRoot, Dio2Error> {
        self.parameters.root(derive(hash_key, "root", &[step]))
    }

    /// The stage hash key and context of the Program of `(step, chunk)`.
    fn stage(hash_key: &[u8; 32], step: usize, chunk: usize) -> ([u8; 32], [u8; 32]) {
        (
            derive(hash_key, "stage-hash-key", &[step, chunk]),
            derive(hash_key, "stage-context", &[step, chunk]),
        )
    }

    /// Run a graph and return its outputs.
    fn execute(
        &mut self,
        stage: &str,
        graph: DslContext,
        runtime: BTreeMap<String, RuntimeValue>,
    ) -> Result<BTreeMap<String, RuntimeValue>, Dio2Error> {
        self.execute_with(stage, graph, runtime, BTreeMap::new())
    }

    /// Run a graph as stage `stage`, reading the artifacts of `manifests`.
    fn execute_with(
        &mut self,
        stage: &str,
        graph: DslContext,
        runtime: BTreeMap<String, RuntimeValue>,
        manifests: BTreeMap<ProductionId, mxx_ir_core::artifact::Manifest>,
    ) -> Result<BTreeMap<String, RuntimeValue>, Dio2Error> {
        let graph = graph.build()?;
        Ok(api::execute_graph(
            StageGraph::new(graph, runtime, manifests),
            &ParamEnv::default(),
            &mut self.execution,
            &mut self.store,
            record(&mut self.graphs, stage.to_owned()),
        )?
        .outputs)
    }

    /// The width of the anchor's preimages as a real expression.
    fn trapdoor_sigma(&self) -> mxx_ir_core::RealExpr {
        mxx_ir_core::RealExpr::from(self.parameters.trapdoor_sigma)
    }
}

impl<E: StageExecution<S>, S: SessionStore> Obfuscation for Dio2<E, S> {
    type Function = Dio2Function;
    type Obfuscation = Dio2Obfuscation;
    type Input = Vec<Vec<bool>>;
    type Output = usize;
    type Error = Dio2Error;

    fn obfuscate(&mut self, function: &Self::Function) -> Result<Self::Obfuscation, Self::Error> {
        self.obfuscate_with_hash_key(function, rand::random())
    }

    fn evaluate(
        &mut self,
        obfuscation: &Self::Obfuscation,
        input: &Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        self.evaluate_chunks(obfuscation, input)
    }
}

/// The PRF keys of one chunk: the σ sampler's, the CBD sampler's ([`StepPlan::cbd_members`]),
/// the smudge PRF's ([`StepPlan::smudge_members`]), and the output decoder's smudge keys.
struct ChunkKeys {
    sigma: Vec<PbcGeneratedKeyLayout>,
    cbd: Vec<PbcGeneratedKeyLayout>,
    smudge: Vec<PbcGeneratedKeyLayout>,
    output: Vec<PbcGeneratedKeyLayout>,
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// The obfuscation of `input` whose public matrices and stage namespaces are derived from
    /// `hash_key`: an obfuscation keyed by a fixed key builds the same stage graphs on every run
    /// (its claim's export is reproducible); [`Obfuscation::obfuscate`] samples the key.
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
        let mut rng = StdRng::from_rng(&mut rand::rng());
        let chunks = 1usize << function.eval_chunk_bits;
        // Keys of every chunk (spec §4.5, §4.6): indexed by the chunk, shared by every step.
        let generate = |rng: &mut StdRng, count| {
            parameters.prf.generate_keys(rng, count).map_err(|_| Dio2Error::Parameters("PRF keys"))
        };
        let keys = (0..chunks)
            .map(|_| {
                Ok(ChunkKeys {
                    sigma: generate(&mut rng, plan.ring_dimension)?,
                    cbd: generate(&mut rng, plan.cbd_members())?,
                    smudge: generate(&mut rng, plan.smudge_members())?,
                    output: generate(&mut rng, plan.output_smudge_members())?,
                })
            })
            .collect::<Result<Vec<_>, Dio2Error>>()?;
        let root = self.root(&hash_key, 0)?;
        let chain = parameters.chain(&root)?;
        let payload = RnsEncryptBuild::new(&chain, derive(&hash_key, "payload", &[]))
            .sample_payload_secret(
                &mut self.execution,
                &mut self.store,
                record(&mut self.graphs, "payload".to_owned()),
            )?;
        if let Some(name) = payload.input_name() {
            self.secrets.insert(name.to_owned(), ("payload".to_owned(), "secret".to_owned()));
        }
        let (anchor_production, anchors) = self.sample_anchors(&hash_key, steps)?;
        let initial = self.initial_input(input, &plan, &chain, &root, &payload, &anchors[0])?;
        let mut materials = Vec::with_capacity(steps);
        for step in 0..steps {
            let last = step + 1 == steps;
            let root = self.root(&hash_key, step)?;
            let next_root = (!last).then(|| self.root(&hash_key, step + 1)).transpose()?;
            let chain = parameters.chain(&root)?;
            let mut step_material = Vec::with_capacity(chunks);
            for chunk in 0..chunks {
                let x = chunk_bits(chunk, function.eval_chunk_bits);
                let material = self.preprocess(
                    function,
                    &plan,
                    &input.y[step],
                    &x,
                    &keys[chunk],
                    (step, chunk),
                    &hash_key,
                    &chain,
                    &payload,
                    last,
                )?;
                // The decoded keys, from a run on zero rows.
                let zero = self.zero_input(&plan, &initial)?;
                let keys = self.run_step(
                    &chain,
                    &material,
                    &plan,
                    &zero,
                    (step, chunk),
                    &hash_key,
                    next_root.as_ref(),
                    anchors.get(step + 1),
                    ComposeMode::Keys,
                )?;
                let mut material = material;
                self.sample_preimages(
                    &anchor_production,
                    (step, chunk),
                    &plan,
                    &mut material,
                    keys,
                )?;
                step_material.push(material);
            }
            materials.push(step_material);
        }
        Ok(Dio2Obfuscation {
            function: function.clone(),
            plan,
            hash_key,
            initial,
            anchors,
            steps: materials,
        })
    }

    /// Sample the anchor `B_i` of every step with its trapdoor (spec §4.7), published as one
    /// artifact production; returns it and the anchors' public matrices.
    fn sample_anchors(
        &mut self,
        hash_key: &[u8; 32],
        steps: usize,
    ) -> Result<(StageArtifacts<()>, Vec<RuntimeValue>), Dio2Error> {
        let ring = self.parameters.root([0; 32])?.ring();
        let columns = self.parameters.columns()?;
        let bound = super::preimage_bound(&self.parameters)?;
        let mut graph = DslContext::new("dio2-anchors");
        for step in 0..steps {
            let trapdoor = sample_trapdoor(
                &ring,
                self.trapdoor_sigma(),
                1u64 << self.parameters.base_bits,
                columns,
                num_bigint::BigInt::from(bound.clone()),
            );
            graph = graph.transferred_output(format!("anchor/{step}"), trapdoor)?;
        }
        let production = api::execute_artifacts(
            StageGraph::new(graph.build()?, BTreeMap::new(), BTreeMap::new()),
            &ParamEnv::default(),
            (),
            derive(hash_key, "anchors", &[]),
            &mut self.execution,
            &mut self.store,
            record(&mut self.graphs, "anchors".to_owned()),
        )?;
        self.productions.insert(production.production().clone(), "anchors".to_owned());
        let mut graph = DslContext::new("dio2-anchor-publics");
        for step in 0..steps {
            graph = graph
                .output(anchor_output(step), self.trapdoor(&production, step)?.public_matrix())?;
        }
        let manifests = self.manifests(&production);
        let mut outputs = self.execute_with("anchor_publics", graph, BTreeMap::new(), manifests)?;
        let anchors = (0..steps)
            .map(|step| output(&mut outputs, &anchor_output(step)))
            .collect::<Result<_, _>>()?;
        Ok((production, anchors))
    }

    fn manifests(
        &self,
        production: &StageArtifacts<()>,
    ) -> BTreeMap<mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest> {
        BTreeMap::from([(production.production().clone(), production.manifest().clone())])
    }

    /// The trapdoor of step `step`'s anchor, read from the anchors' production.
    fn trapdoor(
        &self,
        production: &StageArtifacts<()>,
        step: usize,
    ) -> Result<mxx_dsl::Trapdoor, Dio2Error> {
        let ring = self.parameters.root([0; 32])?.ring();
        // A trapdoor is published as its public matrix (leaf `.0`) and the trapdoor (leaf `.1`).
        Ok(ring.trapdoor_artifact_input(
            production.production().clone(),
            format!("anchor/{step}.1"),
            UNMASK_ROWS,
            self.trapdoor_sigma(),
            1u64 << self.parameters.base_bits,
            self.parameters.columns()?,
            num_bigint::BigInt::from(super::preimage_bound(&self.parameters)?),
        ))
    }

    /// The step input `S_0` (spec §7): every root row under the binary mask `s_0`, the payload
    /// secret `t`, the initial state `z_0` ([`INITIAL_STATE_INPUT`]) and the initial labels
    /// ([`INITIAL_LABELS_INPUT`]), with CBD(1) errors, and the hint `b_0 = s_0 B_0 + e`.
    fn initial_input(
        &mut self,
        input: &Dio2Function,
        plan: &StepPlan,
        chain: &RnsBggParameters,
        root: &RnsBggRoot,
        payload: &RnsBggPayloadSecret,
        anchor: &RuntimeValue,
    ) -> Result<StepInput, Dio2Error> {
        let ring = root.ring();
        let secret = self.execute(
            "mask",
            DslContext::new("dio2-mask").output("mask", ring.uniform_interval((1, 1), 0, 1))?,
            BTreeMap::new(),
        )?;
        let mask = RnsBggMaskSecret::from_sampled(
            chain,
            secret.get("mask").cloned().ok_or(Dio2Error::Input("the mask is missing"))?,
        )?;
        if let Some(name) = mask.input_name() {
            self.secrets.insert(name.to_owned(), ("mask".to_owned(), "mask".to_owned()));
        }
        let q = self.parameters.label_modulus();
        let labels = (0..plan.label_rows)
            .map(|_| (0..=plan.label_width).map(|_| rand::rng().random_range(0..q)).collect())
            .collect::<Vec<Vec<usize>>>();
        let hash_key = root.hash_key_input();
        let (s, t) = (mask.row().clone(), payload.row().clone());
        let error = |columns: usize| {
            ring.uniform_interval((1, columns), 0, 1) - ring.uniform_interval((1, columns), 0, 1)
        };
        let monomial = |exponent: Int| ring.identity(1).multiply_monomial(exponent);
        let prf_stride = self.parameters.profile()?.layout.stride() as i64;
        let state_stride = input.function.state_layout.stride() as i64;
        let mut graph = DslContext::new("dio2-initial");
        let initial_state = graph.input::<Int>(INITIAL_STATE_INPUT, mxx_dsl::IntType)?;
        let label_values = mxx_dsl::int_family_input(
            INITIAL_LABELS_INPUT,
            plan.label_rows * (plan.label_width + 1),
        );
        let mut rows = BTreeMap::new();
        for target in &plan.targets {
            let pair = |value: Mat, row: &(Mat, Mat)| {
                (value.clone() * row.0.clone(), value * row.1.clone())
            };
            let payload = match target {
                Target::Private => pair(ring.identity(1), &t),
                Target::State => pair(monomial(initial_state.clone().mul(state_stride)), &t),
                Target::Public => pair(ring.identity(1), &s),
                Target::Label { row, column } => {
                    let entry = label_values
                        .at(Int::constant((row * (plan.label_width + 1) + column) as i64));
                    pair(monomial(entry.mul(prf_stride)), &s)
                }
                Target::Helper(sigma) => {
                    (s.0.clone().ring_automorphism(IntExpr::constant(*sigma)), ring.identity(1))
                }
                Target::Anchor(_) => continue,
            };
            let tag = target.key_tag(plan).ok_or(Dio2Error::Input("a root target has a tag"))?;
            let key = root.key(hash_key.clone(), tag);
            rows.insert(
                *target,
                root.encode(&key, &s, payload, (error(plan.columns), error(plan.columns)))?,
            );
        }
        for (name, port, targets, is_family) in plan.input_rows() {
            let members = targets
                .iter()
                .map(|target| {
                    let row = rows.get(target).ok_or(Dio2Error::Input("a root row is missing"))?;
                    Ok(if port == 0 { row.0.clone() } else { row.1.clone() })
                })
                .collect::<Result<Vec<_>, Dio2Error>>()?;
            graph = if is_family {
                graph.output(name, mxx_dsl::Family::pack(members)?)?
            } else {
                graph.output(
                    name,
                    members.into_iter().next().ok_or(Dio2Error::Input("a root row is missing"))?,
                )?
            };
        }
        let anchor_columns = anchor_columns(plan.columns);
        let secret_row = Mat::concat(ConcatAxis::Columns, vec![s.0.clone(), s.1.clone()]);
        let hint = unmask_hint(
            secret_row,
            ring.input(ANCHOR_INPUT, (UNMASK_ROWS, anchor_columns)),
            error(anchor_columns),
        );
        graph = graph.output(HINT_INPUT, hint)?;
        let mut runtime = BTreeMap::from([
            (root.hash_key_input_name(), RuntimeValue::Bytes(root.hash_key().as_slice().into())),
            (ANCHOR_INPUT.to_owned(), anchor.clone()),
            (INITIAL_STATE_INPUT.to_owned(), RuntimeValue::Int(input.initial_state.into())),
            (
                INITIAL_LABELS_INPUT.to_owned(),
                RuntimeValue::integer_values(
                    labels.iter().flatten().map(|value| (*value).into()).collect(),
                ),
            ),
        ]);
        payload.bind_runtime(&mut runtime);
        mask.bind_runtime(&mut runtime);
        Ok(StepInput { values: self.execute("initial", graph, runtime)?, labels })
    }

    /// A step input of zero rows (and the labels of `like`): the keys of a composed run do not
    /// depend on its rows.
    fn zero_input(&mut self, plan: &StepPlan, like: &StepInput) -> Result<StepInput, Dio2Error> {
        let ring = self.parameters.root([0; 32])?.ring();
        let mut graph = DslContext::new("dio2-zero");
        for (name, _, targets, is_family) in plan.input_rows() {
            let zero = || ring.zero((1, plan.columns));
            graph = if is_family {
                graph.output(
                    name,
                    mxx_dsl::Family::pack(targets.iter().map(|_| zero()).collect())?,
                )?
            } else {
                graph.output(name, zero())?
            };
        }
        graph = graph.output(HINT_INPUT, ring.zero((1, anchor_columns(plan.columns))))?;
        Ok(StepInput {
            values: self.execute("zero", graph, BTreeMap::new())?,
            labels: like.labels.clone(),
        })
    }

    /// Preprocess the Program of `(step, chunk)`: its RHS plaintexts are the obfuscator's
    /// chunk and the PRF keys' selectors and offsets, its data the keys' coordinate tables.
    #[allow(clippy::too_many_arguments)]
    fn preprocess(
        &mut self,
        function: &StreamingFunction,
        plan: &StepPlan,
        y: &[bool],
        x: &[bool],
        keys: &ChunkKeys,
        (step, chunk): (usize, usize),
        hash_key: &[u8; 32],
        chain_parameters: &RnsBggParameters,
        payload: &RnsBggPayloadSecret,
        last: bool,
    ) -> Result<StepMaterial, Dio2Error> {
        let shape = |layouts: &[PbcGeneratedKeyLayout]| {
            mxx_exponent_lut::program_gadgets::prf::PrfKeyShape::new(
                &layouts.iter().map(|key| key.public_layout.clone()).collect::<Vec<_>>(),
            )
        };
        let smudge_keys = if last { &keys.output } else { &keys.smudge };
        let chain_keys = ChainKeys {
            sigma: (!last).then(|| shape(&keys.sigma)).transpose()?,
            cbd: (!last).then(|| shape(&keys.cbd)).transpose()?,
            smudge: shape(smudge_keys)?,
        };
        let program = if last {
            ChainProgram::last(&self.parameters, function, plan, step, x, &chain_keys)?
        } else {
            ChainProgram::refreshing(&self.parameters, function, plan, step, x, &chain_keys)?
        };
        let level = chain_parameters.level(0)?.clone();
        let mut rhs: PlaintextInputs<RhsId, DCRTPoly> = BTreeMap::from([(
            program.y,
            InputBinding::Family(
                y.iter()
                    .enumerate()
                    .map(|(bit, set)| monomial(&level, (function.y_monomial)(bit, *set)))
                    .collect(),
            ),
        )]);
        let mut data = ProgramData::new();
        let mut merge = |(more, tables): (PlaintextInputs<RhsId, DCRTPoly>, ProgramData),
                         ids: &[mxx_exponent_lut::program::DataId]| {
            rhs.extend(more);
            for id in ids {
                if let Some(values) = tables.values(*id) {
                    data = std::mem::take(&mut data).with(*id, values.to_vec());
                }
            }
        };
        merge(program.smudge.key_inputs::<DCRTPoly>(&level, smudge_keys)?, program.smudge.tables());
        if let Some(cbd) = &program.cbd {
            merge(cbd.key_inputs::<DCRTPoly>(&level, &keys.cbd)?, cbd.prf().tables());
        }
        if let Some(sigma) = &program.sigma {
            merge(sigma.key_inputs::<DCRTPoly>(&level, &keys.sigma)?, sigma.prf().tables());
        }
        let (stage_key, context) = Self::stage(hash_key, step, chunk);
        let preprocess = api::preprocess(
            &program.program,
            &ParamEnv::default(),
            &rhs,
            Some(payload),
            &stage_key,
            RnsPreprocessBuild::new(chain_parameters, context)
                .with_data(data.clone())
                .with_root_inputs(program.root_inputs.clone())
                .with_output_use(RnsOutputUse::Gadget),
            &mut self.execution,
            &mut self.store,
            |_, outputs| Ok(outputs),
            record(&mut self.graphs, format!("preprocess_{step}_{chunk}")),
        )?;
        for production in [preprocess.0.production(), preprocess.1.production()] {
            self.productions.insert(production.clone(), format!("preprocess_{step}_{chunk}"));
        }
        Ok(StepMaterial { program, data, preprocess, preimages: BTreeMap::new() })
    }

    /// The imported rows of `input` that `program`'s inputs and LUT helpers read, and the
    /// public payloads of its public inputs (spec §6.1).
    fn imported(
        &self,
        program: &ChainProgram,
        plan: &StepPlan,
        input: &StepInput,
        level: &DCRTPolyParams,
    ) -> Result<(BTreeMap<String, RuntimeValue>, PlaintextInputs<WireId, DCRTPoly>), Dio2Error>
    {
        let prf_stride = self.parameters.profile()?.layout.stride();
        let tags = program.root_inputs.values().collect::<std::collections::BTreeSet<_>>();
        let keys = program
            .program
            .encryption_inputs()
            .luts()
            .iter()
            .map(|table| table.helper_family_key())
            .collect::<std::collections::BTreeSet<_>>();
        let mut rows = BTreeMap::new();
        for (name, port, _, _) in plan.input_rows() {
            let read = tags.iter().any(|tag| root_row_input_name(tag, port) == name) ||
                keys.iter().any(|key| root_helper_input_name(key, port) == name);
            if read {
                let value =
                    input.values.get(&name).ok_or(Dio2Error::Input("a root row is missing"))?;
                rows.insert(name, value.clone());
            }
        }
        let mut lhs = BTreeMap::new();
        for (wire, tag) in &program.root_inputs {
            let members = plan
                .input_rows()
                .into_iter()
                .find(|(name, _, _, _)| *name == root_row_input_name(tag, 0))
                .map(|(_, _, targets, _)| targets)
                .ok_or(Dio2Error::Input("an unknown root tag"))?;
            if matches!(tag.as_str(), PUBLIC_TAG | LABEL_TAG) {
                let label = |target: &Target| match target {
                    Target::Label { row, column } => input.labels[*row][*column],
                    _ => 0,
                };
                lhs.insert(
                    *wire,
                    InputBinding::Family(
                        members
                            .iter()
                            .map(|target| monomial(level, label(target) * prf_stride))
                            .collect(),
                    ),
                );
            }
        }
        Ok((rows, lhs))
    }

    /// Run the composed online evaluation of a step's Program on the step input `input`.
    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &mut self,
        chain_parameters: &RnsBggParameters,
        material: &StepMaterial,
        plan: &StepPlan,
        input: &StepInput,
        (step, chunk): (usize, usize),
        hash_key: &[u8; 32],
        next_root: Option<&RnsBggRoot>,
        next_anchor: Option<&RuntimeValue>,
        mode: ComposeMode,
    ) -> Result<BTreeMap<String, RuntimeValue>, Dio2Error> {
        let (stage_key, context) = Self::stage(hash_key, step, chunk);
        let level = chain_parameters.level(0)?.clone();
        let (rows, lhs) = self.imported(&material.program, plan, input, &level)?;
        let (import_stage, online_stage) = match mode {
            ComposeMode::Keys => {
                (format!("keys_import_{step}_{chunk}"), format!("keys_online_{step}_{chunk}"))
            }
            ComposeMode::Decode => (format!("import_{step}"), format!("online_{step}")),
        };
        let encrypted = RnsEncryptBuild::new(chain_parameters, context).import(
            &material.program.program.encryption_inputs(),
            &ParamEnv::default(),
            &lhs,
            material.program.root_inputs.clone(),
            rows,
            &material.preprocess.0,
            &stage_key,
            &mut self.execution,
            &mut self.store,
            record(&mut self.graphs, import_stage.clone()),
        )?;
        self.productions.insert(encrypted.production().clone(), import_stage);
        let mut builder = RnsOnlineEvalBuild::new(chain_parameters, context)
            .with_data(material.data.clone())
            .with_root_inputs(material.program.root_inputs.clone())
            .with_output_use(RnsOutputUse::Gadget);
        if let Some(next_root) = next_root {
            builder.bind_runtime_input(
                next_root.hash_key_input_name(),
                RuntimeValue::Bytes(next_root.hash_key().as_slice().into()),
            );
        }
        if let Some(next_anchor) = next_anchor {
            builder.bind_runtime_input(NEXT_ANCHOR_INPUT, next_anchor.clone());
        }
        if mode == ComposeMode::Decode {
            let hint =
                input.values.get(HINT_INPUT).ok_or(Dio2Error::Input("the hint is missing"))?;
            builder.bind_runtime_input(HINT_INPUT, hint.clone());
            for (name, value) in &material.preimages {
                builder.bind_runtime_input(name.clone(), value.clone());
            }
        }
        let compose = StepCompose {
            parameters: &self.parameters,
            chain_parameters,
            plan,
            program: &material.program,
            next_root,
            mode,
        };
        Ok(api::online_eval(
            &material.program.program,
            &ParamEnv::default(),
            &material.preprocess.1,
            &encrypted,
            &stage_key,
            builder,
            &mut self.execution,
            &mut self.store,
            |builder, outputs| {
                compose.compose(builder, &outputs).map_err(ApiError::scheme)?;
                Ok(outputs)
            },
            record(&mut self.graphs, online_stage),
        )?
        .outputs)
    }

    /// Sample the preimage of every decoded key of step `step` on chunk `chunk` under the step's
    /// anchor (spec §4.7): the keys `keys` (of `Q_0`, every level's modulus) are the stage's
    /// inputs and its preimages its outputs, by the names the online evaluation reads.
    fn sample_preimages(
        &mut self,
        anchors: &StageArtifacts<()>,
        (step, chunk): (usize, usize),
        plan: &StepPlan,
        material: &mut StepMaterial,
        mut keys: BTreeMap<String, RuntimeValue>,
    ) -> Result<(), Dio2Error> {
        let trapdoor = self.trapdoor(anchors, step)?;
        let ring = self.parameters.root([0; 32])?.ring();
        let anchor = anchor_columns(plan.columns);
        let mut graph = DslContext::new("dio2-preimages");
        let mut runtime = BTreeMap::new();
        let mut names = Vec::new();
        if keys.contains_key(OUTPUT_KEY_OUTPUT) {
            runtime.insert(OUTPUT_KEY_OUTPUT.to_owned(), output(&mut keys, OUTPUT_KEY_OUTPUT)?);
            graph = graph.output(
                OUTPUT_PREIMAGE_INPUT,
                trapdoor.sample_preimage(ring.input(OUTPUT_KEY_OUTPUT, (2, 1)), (anchor, 1)),
            )?;
            names.push(OUTPUT_PREIMAGE_INPUT.to_owned());
        } else {
            // One loop over the parts per prime.
            for prime in 0..plan.primes {
                let key = key_output(prime);
                runtime.insert(key.clone(), output(&mut keys, &key)?);
                let keys = ring.input_family(key, plan.blocks(), (2, plan.columns));
                let preimages = mxx_dsl::parallel(plan.blocks(), |part| {
                    Ok(trapdoor.sample_preimage(keys.at(part), (anchor, plan.columns)))
                })?;
                graph = graph.output(preimage_input(prime), preimages)?;
                names.push(preimage_input(prime));
            }
        }
        let manifests = self.manifests(anchors);
        let mut outputs =
            self.execute_with(&format!("preimages_{step}_{chunk}"), graph, runtime, manifests)?;
        for name in names {
            material.preimages.insert(name.clone(), output(&mut outputs, &name)?);
        }
        Ok(())
    }

    fn evaluate_chunks(
        &mut self,
        obfuscation: &Dio2Obfuscation,
        x: &[Vec<bool>],
    ) -> Result<usize, Dio2Error> {
        let function = &obfuscation.function;
        let steps = function.stream_length();
        if x.len() != steps || x.iter().any(|chunk| chunk.len() != function.eval_chunk_bits) {
            return Err(Dio2Error::Input("the evaluator's chunks do not fit the function"));
        }
        let plan = &obfuscation.plan;
        let hash_key = &obfuscation.hash_key;
        let mut input = obfuscation.initial.clone();
        for (step, chunk_bits_) in x.iter().enumerate() {
            let chunk = chunk_index(chunk_bits_);
            let root = self.root(hash_key, step)?;
            let chain = self.parameters.chain(&root)?;
            let material = &obfuscation.steps[step][chunk];
            let last = step + 1 == steps;
            let next_root = (!last).then(|| self.root(hash_key, step + 1)).transpose()?;
            let mut outputs = self.run_step(
                &chain,
                material,
                plan,
                &input,
                (step, chunk),
                hash_key,
                next_root.as_ref(),
                obfuscation.anchors.get(step + 1),
                ComposeMode::Decode,
            )?;
            if last {
                return self.decode(output(&mut outputs, OUTPUT_PLAINTEXT_OUTPUT)?, function);
            }
            input = StepInput {
                values: outputs,
                labels: next_labels(&self.parameters, &input.labels, chunk_bits_),
            };
        }
        Err(Dio2Error::Input("an obfuscation has at least one step"))
    }

    /// The decode stage: the final state `z` of the output decoder's plaintext `mono(z) mod q`
    /// ([`decode_state`]).
    fn decode(
        &mut self,
        plaintext: RuntimeValue,
        function: &StreamingFunction,
    ) -> Result<usize, Dio2Error> {
        let q = self.parameters.primes[0];
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(q)], self.parameters.ring_dimension);
        let state = decode_state(
            ring.input(PLAINTEXT_INPUT, (1, 1)),
            q,
            self.parameters.ring_dimension as usize,
            function.state_layout.stride(),
        )?;
        let mut outputs = self.execute(
            "decode",
            DslContext::new("dio2-decode").output(STATE_OUTPUT, state)?,
            BTreeMap::from([(PLAINTEXT_INPUT.to_owned(), plaintext)]),
        )?;
        let state = self.execution.download_integers(&output(&mut outputs, STATE_OUTPUT)?)?;
        state
            .first()
            .and_then(num_traits::ToPrimitive::to_usize)
            .ok_or(Dio2Error::Input("the decoded state is not a state"))
    }
}

/// The state `z` of the plaintext `mono(z) = X^(z stride)` modulo the odd prime `q` (`1 x 1`, ring
/// dimension `n`): with every coefficient `c_k` centered to `v_k in {-1, 0, 1}`, the exponent
/// `sum_k (k v_k (v_k + 1) / 2 + (k + n) v_k (v_k - 1) / 2)` of the one nonzero coefficient
/// (negative past `X^n`), divided by `stride`. One loop over the coefficients, so the graph does
/// not grow with `n`.
fn decode_state(plaintext: Mat, q: u64, n: usize, stride: usize) -> Result<Int, Dio2Error> {
    let coefficients = plaintext.coefficients();
    let exponent = mxx_dsl::iterate(n, Int::constant(0), |k, total| {
        let c = coefficients.at(k.clone());
        let v = c.clone().sub(c.mul(2).div(q as i64).mul(q as i64));
        let up = v.clone().mul(v.clone().add(1)).div(2);
        let down = v.clone().mul(v.sub(1)).div(2);
        Ok(total.add(k.clone().mul(up)).add(k.add(n as i64).mul(down)))
    })?;
    Ok(exponent.div(stride as i64))
}
