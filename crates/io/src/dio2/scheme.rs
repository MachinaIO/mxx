//! The Diamond iO 2 instance (spec §5–§8) on any stage executor: an obfuscation is one graph
//! ([`super::obfuscate`]) executed once into one artifact production, and an evaluation is one
//! graph ([`super::evaluate`]) that reads it.
//!
//! This module holds what both graphs share: the two kinds of the uniform step Program
//! ([`Kind`]), the root of every step ([`step_root`]), the graph inputs of the public data both
//! graphs read (the label matrix `A_0`, the function's step rows, the PRF coordinate tables), and
//! the integer data a Program instance of a step and chunk binds ([`Kind::data`]).

use super::{
    Dio2Error, Dio2Parameters, StreamingFunction,
    step::{ChainKeys, ChainProgram, PUBLIC_TAG, StepPlan, UNIFORM_SEED, XBITS_TAG, xbits_key},
};
use crate::Obfuscation;
use mxx_backends::{RuntimeValue, SessionStore};
use mxx_dsl::{Family, Int};
use mxx_exponent_lut::{
    api::{ApiError, StageArtifacts, StageExecution},
    program::DataId,
    program_gadgets::prf::SparseLwrPrf,
    rns_bgg::{RnsBggParameters, RnsBggPreprocessOnlineEvalDescriptor, RnsBggRoot},
};
use mxx_ir_core::{Graph, IntExpr, ValidatedGraph, artifact::ProductionId};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

/// The obfuscator's input: a streaming function, its chunk `y_i` of every step, and the initial
/// state `z_0`.
#[derive(Clone)]
pub struct Dio2Function {
    pub function: StreamingFunction,
    pub y: Vec<Vec<bool>>,
    pub initial_state: usize,
}

/// The graph input of the initial state `z_0` (obfuscation).
pub const INITIAL_STATE_INPUT: &str = "dio2/initial-state";
/// The graph input of the initial labels `A_0`, entry `(r, c)` at `r (m_v + 1) + c` (both
/// graphs).
pub const LABELS_INPUT: &str = "dio2/initial-labels";
/// The graph input of the obfuscator's bits, bit `b` of step `i` at `i obf_chunk_bits + b`
/// (obfuscation).
pub const Y_INPUT: &str = "dio2/y";
/// The graph input of the function's step rows, row `i` at `i width` (both graphs).
pub const STEP_DATA_INPUT: &str = "dio2/step-data";
/// The graph input of the evaluator's chunks, the little-endian value of step `i` at `i`
/// (evaluation).
pub const X_INPUT: &str = "dio2/x";
/// The obfuscation's step input `S_0` ([`super::step::StepRows`]).
pub const INITIAL_OUTPUT: &str = "dio2/initial";
/// The obfuscation's anchors `B_i`, member `i`.
pub const ANCHORS_OUTPUT: &str = "dio2/anchors";
/// The evaluation's decoded final state.
pub const STATE_OUTPUT: &str = "dio2/state";

/// The graph input of the selector bits of bucket `bucket` of the PRF `role` of every chunk:
/// chunk `x`'s family at `x members ..` (obfuscation).
pub fn selector_input(role: &str, bucket: usize) -> String {
    format!("dio2/prf/{role}/{bucket}/selectors")
}

/// The graph input of the coordinate table of bucket `bucket` of the PRF `role` of every chunk,
/// laid out as [`selector_input`] (both graphs).
pub fn table_input(role: &str, bucket: usize) -> String {
    format!("dio2/prf/{role}/{bucket}/table")
}

/// The kind of a step Program: a step followed by a refresh, or the last step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Refreshing,
    Last,
}

impl Kind {
    /// The builder context and the stage hash key of the kind's Program in the obfuscation
    /// keyed by `hash_key`: one of each for every instance (their hash domains separate them).
    pub(crate) fn identity(self, hash_key: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
        let index = match self {
            Kind::Refreshing => 0,
            Kind::Last => 1,
        };
        (derive(hash_key, "context", &[index]), derive(hash_key, "stage-hash-key", &[index]))
    }

    /// The namespace of the kind's Program inputs and artifacts.
    pub(crate) fn prefix(self) -> &'static str {
        match self {
            Kind::Refreshing => "dio2/refreshing",
            Kind::Last => "dio2/last",
        }
    }

    /// The artifact of the anchor preimages of every instance: per part and prime (member
    /// `(m blocks + p) |P| + k` for instance `m`) or of the output decoder (member `m`).
    pub(crate) fn preimages(self) -> String {
        format!("{}/preimages", self.prefix())
    }

    /// The instances of the kind: every refreshing step and chunk, or every chunk of the last
    /// step.
    pub(crate) fn members(self, function: &StreamingFunction) -> usize {
        let chunks = 1usize << function.eval_chunk_bits;
        match self {
            Kind::Refreshing => (function.stream_length() - 1) * chunks,
            Kind::Last => chunks,
        }
    }

    /// The step and the chunk of instance `member`.
    pub(crate) fn instance(self, function: &StreamingFunction, member: &Int) -> (Int, Int) {
        let chunks = 1usize << function.eval_chunk_bits;
        match self {
            Kind::Refreshing => (member.clone().div(chunks), member.clone().rem(chunks)),
            Kind::Last => (Int::constant(function.stream_length() - 1), member.clone()),
        }
    }

    /// The instance of step `step` on chunk `chunk`.
    pub(crate) fn member(self, function: &StreamingFunction, step: &Int, chunk: &Int) -> Int {
        match self {
            Kind::Refreshing => {
                step.clone().mul(1usize << function.eval_chunk_bits).add(chunk.clone())
            }
            Kind::Last => chunk.clone(),
        }
    }

    /// The kind's Program on its uniform declared PRF key shapes.
    pub(crate) fn program(
        self,
        parameters: &Dio2Parameters,
        function: &StreamingFunction,
        plan: &StepPlan,
    ) -> Result<ChainProgram, Dio2Error> {
        let shape = |role: &str, members| {
            parameters
                .prf
                .declared_shape(UNIFORM_SEED, role, members)
                .map_err(|error| Dio2Error::Api(ApiError::scheme(error)))
        };
        Ok(match self {
            Kind::Refreshing => ChainProgram::refreshing(
                parameters,
                function,
                plan,
                &ChainKeys {
                    sigma: Some(shape("sigma", plan.ring_dimension)?),
                    cbd: Some(shape("cbd", plan.cbd_members())?),
                    smudge: shape("smudge", plan.smudge_members())?,
                },
            )?,
            Kind::Last => ChainProgram::last(
                parameters,
                function,
                plan,
                &ChainKeys {
                    sigma: None,
                    cbd: None,
                    smudge: shape("output", plan.output_smudge_members())?,
                },
            )?,
        })
    }

    /// The PRFs of the kind's Program by key role, in claim order.
    pub(crate) fn prfs(self, program: &ChainProgram) -> Vec<(&'static str, &SparseLwrPrf)> {
        match self {
            Kind::Last => vec![("output", &program.smudge)],
            Kind::Refreshing => std::iter::once(("smudge", &program.smudge))
                .chain(program.sigma.iter().map(|sampler| ("sigma", sampler.prf())))
                .chain(program.cbd.iter().map(|sampler| ("cbd", sampler.prf())))
                .collect(),
        }
    }

    /// The integer data of the instance of step `step` on chunk `chunk`: the step's row of the
    /// function and the chunk's PRF coordinate tables, members of the graph inputs
    /// [`STEP_DATA_INPUT`] and [`table_input`].
    pub(crate) fn data(
        self,
        function: &StreamingFunction,
        program: &ChainProgram,
        (step, chunk): (&Int, &Int),
        inputs: &super::input_storage::InputStorage,
    ) -> Result<BTreeMap<DataId, (Family<Int>, Int)>, Dio2Error> {
        let width = function.data.width;
        let rows = step_data_input(function);
        // The step's row, a LUT's data: its own family of the row width.
        let mut data = BTreeMap::from([(
            program.data,
            (
                mxx_dsl::parallel(width, |entry| Ok(rows.at(step.clone().mul(width).add(entry))))?,
                Int::constant(0),
            ),
        )]);
        // The chunk's coordinate tables, read in the family of every chunk's at their first
        // member: no family of an instance's table entries is formed.
        for (role, prf) in self.prfs(program) {
            for (bucket, table) in prf.tables().iter().enumerate() {
                let count = IntExpr::constant(prf.selector_families()[bucket].members);
                let first = chunk.clone().mul(Int::evaluate(count.clone()));
                data.insert(*table, (table_family(role, bucket, &count, function, inputs)?, first));
            }
        }
        Ok(data)
    }
}

/// The graph input [`STEP_DATA_INPUT`] of `function`'s rows.
pub(crate) fn step_data_input(function: &StreamingFunction) -> Family<Int> {
    mxx_dsl::int_family_input(STEP_DATA_INPUT, function.stream_length() * function.data.width)
}

/// The graph input [`table_input`] of `count` entries per chunk.
pub(crate) fn table_family(
    role: &str,
    bucket: usize,
    count: &IntExpr,
    function: &StreamingFunction,
    inputs: &super::input_storage::InputStorage,
) -> Result<Family<Int>, Dio2Error> {
    let chunks = IntExpr::constant(1usize << function.eval_chunk_bits);
    inputs.family(
        table_input(role, bucket),
        IntExpr::Mul(Box::new(chunks), Box::new(count.clone())).canonicalize(),
    )
}

/// The root of step `step` under the root `root` of an obfuscation: its hash domain is the
/// step, so every step's root objects have their own keys under one root hash key. With the
/// chunk `chunk`, its evaluator-bit inputs ([`XBITS_TAG`]) read the public monomials of the
/// chunk's bits times the key of `ONEPUB` ([`xbits_key`]), as the evaluator's rows do.
pub(crate) fn step_root(
    root: &RnsBggRoot,
    function: &StreamingFunction,
    plan: &StepPlan,
    step: Int,
    chunk: Option<Int>,
) -> RnsBggRoot {
    let root = root.clone().with_domain(vec![step]);
    let Some(chunk) = chunk else { return root };
    let (function, one) = (function.clone(), plan.label_rows);
    root.with_derived_input(
        XBITS_TAG,
        Arc::new(move |root, hash_key, member| {
            let key = root.input_key(hash_key, PUBLIC_TAG, Some(Int::constant(one)))?;
            let member = member
                .ok_or(mxx_exponent_lut::rns_bgg::RnsBggError::Binding("an evaluator bit"))?;
            xbits_key(&function, &key, member, chunk.clone())
                .map_err(|error| mxx_exponent_lut::rns_bgg::RnsBggError::Graph(error.to_string()))
        }),
    )
}

/// The chain of the instance of step `step` on chunk `chunk`: rooted at the step's root, with
/// the hash domain `(step, chunk)` for the Program's sampled public matrices.
pub(crate) fn step_chain(
    parameters: &Dio2Parameters,
    root: &RnsBggRoot,
    (step, chunk): (Int, Int),
) -> Result<RnsBggParameters, Dio2Error> {
    Ok(parameters.chain(root)?.with_hash_domain(vec![step, chunk]))
}

/// The key `H(hash_key, role, indices)` of a role of the obfuscation.
pub(crate) fn derive(hash_key: &[u8; 32], role: &str, indices: &[usize]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"mxx-io/dio2/derive/v2");
    digest.update(hash_key);
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    for index in indices {
        digest.update((*index as u64).to_le_bytes());
    }
    digest.finalize().into()
}

/// A Program kind of an obfuscation: its Program and the descriptor of its preprocessing in
/// the obfuscation's production.
pub(crate) struct KindMaterial {
    pub(crate) program: ChainProgram,
    pub(crate) descriptor: RnsBggPreprocessOnlineEvalDescriptor,
}

/// An obfuscation (spec §7): the production of the obfuscation graph (the step input `S_0`, the
/// anchors, and the Static preprocessing and anchor preimages of every step and chunk), the
/// Programs it was preprocessed for, and the public data the evaluation also reads.
pub struct Dio2Obfuscation {
    pub(crate) function: StreamingFunction,
    pub(crate) plan: StepPlan,
    pub(crate) hash_key: [u8; 32],
    pub(crate) production: StageArtifacts<()>,
    pub(crate) refreshing: Option<KindMaterial>,
    pub(crate) last: KindMaterial,
    /// The public runtime inputs of both graphs: `A_0`, the step rows and the PRF tables (none
    /// in shape-only mode).
    pub(crate) public: BTreeMap<String, RuntimeValue>,
    pub(crate) input_storage: super::input_storage::InputStorage,
}

impl Dio2Obfuscation {
    /// The material of `kind`.
    pub(crate) fn material(&self, kind: Kind) -> Result<&KindMaterial, Dio2Error> {
        match kind {
            Kind::Refreshing => {
                self.refreshing.as_ref().ok_or(Dio2Error::Input("a one-step function"))
            }
            Kind::Last => Ok(&self.last),
        }
    }

    /// The obfuscation's preprocessing of `kind` as the stage an online evaluation reads.
    pub(crate) fn preprocessing(
        &self,
        kind: Kind,
    ) -> Result<StageArtifacts<RnsBggPreprocessOnlineEvalDescriptor>, Dio2Error> {
        Ok(StageArtifacts::new(
            self.production.production().clone(),
            self.production.manifest().clone(),
            self.material(kind)?.descriptor.clone(),
            self.production.session().clone(),
        ))
    }
}

/// A Diamond iO 2 instance: its parameters and the executor and artifact store its graphs run
/// on, whose backend hosts every ring of [`Dio2Parameters::rings`]. It records the obfuscation
/// and evaluation graphs it ran ([`OBFUSCATE_STAGE`], [`EVALUATE_STAGE`]) and the stage of every
/// production, for the claim of a run ([`super::claim`]).
pub struct Dio2<E, S> {
    pub(crate) inputs_by_shape: bool,
    pub(crate) parameters: Dio2Parameters,
    pub(crate) execution: E,
    pub(crate) store: S,
    pub(crate) graphs: BTreeMap<String, Graph>,
    pub(crate) productions: BTreeMap<ProductionId, String>,
}

/// The stage name of the obfuscation graph.
pub const OBFUSCATE_STAGE: &str = "obfuscate";
/// The stage name of the evaluation graph.
pub const EVALUATE_STAGE: &str = "evaluate";

/// A stage observer that records the validated graph as stage `stage` in `graphs`.
pub(crate) fn record<'a>(
    graphs: &'a mut BTreeMap<String, Graph>,
    stage: &str,
) -> impl FnOnce(&ValidatedGraph) -> Result<(), ApiError> + 'a {
    let stage = stage.to_owned();
    move |validated| {
        graphs.insert(stage, validated.source.clone());
        Ok(())
    }
}

impl<E: StageExecution<S>, S: SessionStore> Dio2<E, S> {
    /// An instance on `parameters` running on `execution` and `store`.
    pub fn new(parameters: Dio2Parameters, execution: E, store: S) -> Result<Self, Dio2Error> {
        parameters.validate()?;
        Ok(Self {
            inputs_by_shape: false,
            parameters,
            execution,
            store,
            graphs: BTreeMap::new(),
            productions: BTreeMap::new(),
        })
    }

    pub fn parameters(&self) -> &Dio2Parameters {
        &self.parameters
    }

    /// Build graphs whose PRF keys, tables, labels, chunks and states are declared by shape
    /// only, for graph-only claims: no keys are generated and no value of these inputs is
    /// bound, so the graphs need external bindings before execution. Use only with an executor
    /// that records graphs without running them.
    pub fn with_inputs_by_shape(mut self) -> Self {
        self.inputs_by_shape = true;
        self
    }

    /// The validated graph of stage `stage` this instance recorded.
    pub fn graph(&self, stage: &str) -> Option<&Graph> {
        self.graphs.get(stage)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio2::test_support::toy;
    use mxx_backends::{MemoryArtifactStore, authority::CpuExecution, backend::poly::cpu_backend};
    use rand::Rng;

    /// A uniform table function on `eval_bits` evaluator bits whose body reads evaluator bit 0
    /// and obfuscator bit 0: its Program is the same for every chunk width.
    fn first_bit_function(eval_bits: usize, steps: usize) -> StreamingFunction {
        use mxx_exponent_lut::{
            ExpLayout,
            program::{LutOutputRepresentation, LutTable, Operand},
        };
        let layout = ExpLayout { stride_log2: 1, width_log2: 2 };
        let mut function = StreamingFunction::additive(1, eval_bits, 1, layout);
        function.body = super::super::StepProgram::new(
            move |builder, wires| {
                let state = builder.mul_public(
                    Operand::Member { family: wires.x, index: IntExpr::constant(0) },
                    Operand::Scalar(wires.state),
                )?;
                let state = builder.mul_private(
                    Operand::Scalar(state),
                    Operand::Member { family: wires.y, index: IntExpr::constant(0) },
                )?;
                let lut = builder.register_lut(LutTable::from_data(
                    layout.width(),
                    layout,
                    2,
                    wires.data,
                    0,
                    LutOutputRepresentation::Exponent(layout),
                )?)?;
                builder.lut(Operand::Scalar(state), lut)
            },
            |row, state, x, y| row[state + usize::from(x[0]) + usize::from(y[0])],
            |row, state, x, y| Ok(row.at(state.add(x.rem(2usize)).add(y[0].clone()))),
        );
        let mut rng = rand::rng();
        (0..steps).fold(function, |function, _| {
            function.with_step_row((0..4).map(|_| rng.random_range(0..2)).collect()).unwrap()
        })
    }

    /// The obfuscation and evaluation graphs (graph-only, inputs by shape) have the same node
    /// count for parameters that differ in the ring dimension, the number of primes, the label
    /// width, the smudge digits, the stream length and the chunk width. (One smudge digit folds
    /// index arithmetic by the digit count away, so both sets have at least two digits.)
    #[test]
    fn graphs_do_not_grow_with_the_parameters() {
        use crate::dio2::test_support::GraphOnly;
        let counts = [(8u32, 5usize, 1usize, 2usize, 2usize, 1usize), (16, 3, 2, 3, 3, 2)].map(
            |(ring_dimension, primes, label_rows, digits, steps, eval_bits)| {
                let (mut parameters, _) = toy();
                parameters.ring_dimension = ring_dimension;
                parameters.primes = mxx_backends::openfhe_guard::gen_modulus_and_warmup(
                    ring_dimension,
                    primes,
                    parameters.crt_bits,
                    None,
                )
                .unwrap();
                parameters.prf.lwr_dimension = label_rows;
                parameters.smudge_digits = digits;
                let function = first_bit_function(eval_bits, steps);
                let execution = GraphOnly(
                    CpuExecution::new(cpu_backend(parameters.rings().unwrap())),
                    Vec::new(),
                );
                let mut dio2 = Dio2::new(parameters, execution, MemoryArtifactStore::default())
                    .unwrap()
                    .with_inputs_by_shape();
                let input = Dio2Function {
                    function: function.clone(),
                    y: vec![vec![false; function.obf_chunk_bits]; steps],
                    initial_state: 0,
                };
                let obfuscation = dio2.obfuscate_with_hash_key(&input, [7; 32]).unwrap();
                dio2.evaluate(&obfuscation, &vec![vec![false; eval_bits]; steps]).unwrap();
                [OBFUSCATE_STAGE, EVALUATE_STAGE].map(|stage| {
                    dio2.graph(stage)
                        .unwrap()
                        .scopes()
                        .values()
                        .map(|scope| scope.nodes().len())
                        .sum::<usize>()
                })
            },
        );
        eprintln!("graph node counts (obfuscate, evaluate): {counts:?}");
        assert_eq!(counts[0], counts[1]);
    }

    /// No matrix a GPU plan of the obfuscation or evaluation graph holds grows with the parts, the
    /// number of primes (beyond the `O(w_0)` rows and `O(w_0^2)` words of one operation), the
    /// label rows, the smudge digits, the PRF outputs, the steps or the chunks (graph-only, the
    /// parameter sets of [`graphs_do_not_grow_with_the_parameters`]): every family of matrices the
    /// plan keeps in a member arena has the same member count for both sets (the anchor trapdoors
    /// of the steps excepted), every other one is exported, spilled to its export, or a streamed
    /// carry, and every matrix (or family member) has at most `2 (w_0 + 2) w_0` rows and columns
    /// and `4 (w_0 + 2)^2` entries. The resident families of one-word integers (the clear labels
    /// and the packed label-transition columns, `O(v m_v)` and `O(2^c m_v)` words) are listed.
    #[test]
    fn test_vram_audit_resident_families() {
        use crate::dio2::test_support::{GraphOnly, residency};
        let audits = [(8u32, 5usize, 1usize, 2usize, 2usize, 1usize), (16, 3, 2, 3, 3, 2)].map(
            |(ring_dimension, primes, label_rows, digits, steps, eval_bits)| {
                let (mut parameters, _) = toy();
                parameters.ring_dimension = ring_dimension;
                parameters.primes = mxx_backends::openfhe_guard::gen_modulus_and_warmup(
                    ring_dimension,
                    primes,
                    parameters.crt_bits,
                    None,
                )
                .unwrap();
                parameters.prf.lwr_dimension = label_rows;
                parameters.smudge_digits = digits;
                let function = first_bit_function(eval_bits, steps);
                let execution = GraphOnly(
                    CpuExecution::new(cpu_backend(parameters.rings().unwrap())),
                    Vec::new(),
                );
                let mut dio2 =
                    Dio2::new(parameters.clone(), execution, MemoryArtifactStore::default())
                        .unwrap()
                        .with_inputs_by_shape();
                let input = Dio2Function {
                    function: function.clone(),
                    y: vec![vec![false; function.obf_chunk_bits]; steps],
                    initial_state: 0,
                };
                let obfuscation = dio2.obfuscate_with_hash_key(&input, [7; 32]).unwrap();
                dio2.evaluate(&obfuscation, &vec![vec![false; eval_bits]; steps]).unwrap();
                let w0 = crate::dio2::step::StepPlan::new(&parameters, &function).unwrap().columns;
                let audit = dio2
                    .execution
                    .1
                    .iter()
                    .zip(["obfuscate", "evaluate"])
                    .flat_map(|(graph, name)| {
                        residency(graph).into_iter().map(move |entry| (name, entry))
                    })
                    .collect::<Vec<_>>();
                (w0, audit)
            },
        );
        let mut violations = Vec::new();
        for (w0, audit) in &audits {
            let bound = 2 * (w0 + 2) * w0;
            let mut totals = std::collections::BTreeMap::<_, u128>::new();
            for (graph, entry) in audit {
                *totals.entry((*graph, entry.class)).or_default() += entry.bytes;
                if let Some((rows, columns)) = entry.dimensions &&
                    (rows.max(columns) > bound || rows * columns > 4 * (w0 + 2) * (w0 + 2))
                {
                    violations
                        .push(format!("w0 {w0}: {graph} {} {} wide", entry.scope, entry.shape));
                }
            }
            eprintln!("residency (w0 {w0}): {totals:?}");
            for (graph, entry) in audit.iter().filter(|(_, entry)| entry.class == "integer") {
                eprintln!("integer family (w0 {w0}): {graph} {} {}", entry.scope, entry.shape);
            }
        }
        // The anchor trapdoors of the steps (and their public matrices) stay one family over the
        // steps, read by every instance.
        let resident = |audit: &[(&str, crate::dio2::test_support::Residency)]| {
            audit
                .iter()
                .filter(|(_, entry)| entry.class == "resident" && !entry.trapdoors)
                .map(|(graph, entry)| {
                    (
                        (graph.to_string(), entry.key.clone()),
                        (entry.count, entry.node.clone(), entry.shape.clone()),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        let (first, second) = (resident(&audits[0].1), resident(&audits[1].1));
        for (key, (count, node, shape)) in &first {
            let other = second.get(key).map(|(count, _, shape)| (count, shape));
            if other.map(|(count, _)| count) != Some(count) {
                violations.push(format!("{key:?} {node}: {shape} vs {other:?}"));
            }
        }
        for key in second.keys().filter(|key| !first.contains_key(*key)) {
            violations.push(format!("{key:?} is resident only for the second set"));
        }
        assert!(
            violations.is_empty(),
            "{} violations:\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
}
