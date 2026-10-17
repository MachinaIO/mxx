//! The toy parameters and function of the unit tests.

use super::{Dio2Parameters, StreamingFunction};
use bigdecimal::BigDecimal;
use mxx_backends::openfhe_guard::gen_modulus_and_warmup;
use mxx_exponent_lut::{ExpLayout, refresh::PrfSettings};

/// Ring dimension `8`; five primes of 50 bits with gadget base `2^12` and six levels of `Q_0`
/// (decoding level 4); the PRF on `Q = 16`, `P = 8` with one LWR coordinate and weight one; one
/// smudge digit; noise-free preprocessing (`MXX_IO_DIO2_TEST_{CRT_BITS,PRIMES,BASE_BITS,LEVELS}`
/// override the ring; `N`, `INPUT_MODULUS`, `OUTPUT_MODULUS`, and `STRIDE_LOG2`
/// under the same prefix also permit checking the GPU toy parameters). The
/// function has one state bit, one evaluator bit and one obfuscator bit, on the state layout
/// `(1, 2)` (`mono(e) = X^(2 e)`, `e < 4`), and two table steps.
pub(crate) fn toy() -> (Dio2Parameters, StreamingFunction) {
    let read = |name: &str, default: usize| {
        std::env::var(format!("MXX_IO_DIO2_TEST_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    };
    let n = read("N", 8) as u32;
    let crt_bits = read("CRT_BITS", 50);
    let primes =
        gen_modulus_and_warmup(n, read("PRIMES", 5), crt_bits, None).expect("generated CRT basis");
    let parameters = Dio2Parameters {
        ring_dimension: n,
        crt_bits,
        base_bits: read("BASE_BITS", 12) as u32,
        primes,
        levels: read("LEVELS", 6),
        prf: PrfSettings {
            lwr_dimension: 1,
            support_weight: 1,
            input_modulus: read("INPUT_MODULUS", 16),
            output_modulus: read("OUTPUT_MODULUS", 8),
            bucket_width: 0,
        },
        smudge_digits: 1,
        trapdoor_sigma: 4,
        encoding_error_sigma: BigDecimal::from(0),
    };
    let function = StreamingFunction::additive(
        1,
        1,
        1,
        ExpLayout { stride_log2: read("STRIDE_LOG2", 1) as u32, width_log2: 2 },
    );
    let steps = [vec![1, 0, 1, 0], vec![0, 1, 1, 0]];
    let function =
        steps.into_iter().fold(function.clone(), |acc, table| acc.with_step_row(table).unwrap());
    (parameters, function)
}

/// An execution that validates graphs without executing them, for graph-only tests: a producing
/// graph returns the production its execution would have, an evaluated graph the integer `0`
/// for each output. Downloads read from the wrapped CPU execution.
pub(crate) struct GraphOnly(
    pub(crate) mxx_backends::authority::CpuExecution,
    /// Every validated graph, in order.
    pub(crate) Vec<mxx_ir_core::ValidatedGraph>,
);

impl mxx_exponent_lut::api::StageExecution<mxx_backends::MemoryArtifactStore> for GraphOnly {
    fn produce(
        &mut self,
        validated: mxx_ir_core::ValidatedGraph,
        _: std::collections::BTreeMap<String, mxx_backends::RuntimeValue>,
        _: &mut mxx_backends::MemoryArtifactStore,
        execution_nonce: [u8; 32],
    ) -> Result<mxx_ir_core::artifact::ProductionId, mxx_exponent_lut::api::ApiError> {
        self.1.push(validated.clone());
        let specification =
            mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
                .expect("stage specification hash");
        Ok(mxx_ir_core::artifact::production_id(specification, execution_nonce))
    }

    fn evaluate(
        &mut self,
        validated: mxx_ir_core::ValidatedGraph,
        _: std::collections::BTreeMap<String, mxx_backends::RuntimeValue>,
        _: &mut mxx_backends::MemoryArtifactStore,
        _: [u8; 32],
    ) -> Result<mxx_exponent_lut::api::StageOutputs, mxx_exponent_lut::api::ApiError> {
        self.1.push(validated.clone());
        Ok(mxx_exponent_lut::api::StageOutputs {
            outputs: validated
                .source
                .outputs()
                .keys()
                .map(|name| (name.clone(), mxx_backends::RuntimeValue::Int(0.into())))
                .collect(),
        })
    }

    fn download_matrix(
        &self,
        value: &mxx_backends::RuntimeValue,
    ) -> Result<mxx_backends::matrix::dcrt_poly::DCRTPolyMatrix, mxx_exponent_lut::api::ApiError>
    {
        mxx_exponent_lut::api::StageExecution::<mxx_backends::MemoryArtifactStore>::download_matrix(
            &self.0, value,
        )
    }

    fn download_integers(
        &self,
        value: &mxx_backends::RuntimeValue,
    ) -> Result<Vec<num_bigint::BigInt>, mxx_exponent_lut::api::ApiError> {
        mxx_exponent_lut::api::StageExecution::<mxx_backends::MemoryArtifactStore>::download_integers(
            &self.0, value,
        )
    }
}

/// One family or matrix value of a validated graph that a GPU plan may hold: its scope (the
/// loop counts enclosing it, and the scope without them, which identifies the value across
/// parameter sets of one graph shape), producing node and port, shape, bytes and class.
#[derive(Debug)]
pub(crate) struct Residency {
    pub(crate) scope: String,
    pub(crate) key: String,
    /// The producing node's kind, abbreviated.
    pub(crate) node: String,
    pub(crate) shape: String,
    /// The member count of a family; `None` for a matrix.
    pub(crate) count: Option<u128>,
    /// The rows and columns of a matrix (of a family's members).
    pub(crate) dimensions: Option<(usize, usize)>,
    pub(crate) bytes: u128,
    pub(crate) class: &'static str,
    /// Whether the producing node outputs a trapdoor family (and so its public matrices).
    pub(crate) trapdoors: bool,
}

/// The bytes of a value of `ty`: full matrices `rows columns crt_depth n` words, bounded ones as
/// many (an upper bound), a trapdoor its matrix leaves, a family its members.
pub(crate) fn wire_bytes(ty: &mxx_ir_core::types::ConcreteWireType) -> u128 {
    use mxx_ir_core::types::ConcreteWireType as T;
    let matrix = |m: &mxx_ir_core::types::ConcreteMatrixType| {
        (m.rows * m.columns) as u128 *
            m.ring.crt_depth() as u128 *
            m.ring.ring_dimension() as u128 *
            8
    };
    match ty {
        T::Matrix(m) => matrix(m),
        T::SmallMatrix { matrix: m, .. } | T::Preimage { matrix: m, .. } => matrix(m),
        T::Trapdoor { matrix: m, digit_count, .. } => {
            let d = m.rows as u128;
            let k = *digit_count as u128;
            (4 * d * d * k + 3 * d * d) *
                m.ring.crt_depth() as u128 *
                m.ring.ring_dimension() as u128 *
                8
        }
        T::IndexedFamily { element, count } => count * wire_bytes(element),
        _ => 8,
    }
}

fn shape(ty: &mxx_ir_core::types::ConcreteWireType) -> String {
    use mxx_ir_core::types::ConcreteWireType as T;
    match ty {
        T::Matrix(m) => format!("M{}x{}", m.rows, m.columns),
        T::SmallMatrix { matrix: m, .. } => format!("S{}x{}", m.rows, m.columns),
        T::Preimage { matrix: m, .. } => format!("P{}x{}", m.rows, m.columns),
        T::Trapdoor { matrix: m, .. } => format!("T{}x{}", m.rows, m.columns),
        T::IndexedFamily { element, count } => format!("{count}x{}", shape(element)),
        other => format!("{other:?}").chars().take(10).collect(),
    }
}

/// The rows and columns of a matrix type, or of the innermost members of a family type.
fn dimensions(ty: &mxx_ir_core::types::ConcreteWireType) -> Option<(usize, usize)> {
    use mxx_ir_core::types::ConcreteWireType as T;
    match ty {
        T::Matrix(m) |
        T::SmallMatrix { matrix: m, .. } |
        T::Preimage { matrix: m, .. } |
        T::Trapdoor { matrix: m, .. } => Some((m.rows, m.columns)),
        T::IndexedFamily { element, .. } => dimensions(element),
        _ => None,
    }
}

/// Every loop output family and sequential carry of `graph` and every matrix of its scopes,
/// with its class under the GPU planner's policy:
///
/// - `exported`: a family written member by member to an artifact and read by nothing else: a root
///   loop's family that only artifact outputs read, or that only the root loops flattening it into
///   an artifact output read (with the members those loops select), and, nested, a loop body's
///   output family that the enclosing loop's exported family returns per lane;
/// - `spilled`: an exported matrix family that loops nested in the same lane also read member by
///   member, which the GPU reads back from its export instead of holding it;
/// - `streamed-carry`: a matrix-family carry of a sequential loop (streamed through the store);
/// - `summed`: a parallel producer whose only consumer is the recognized member sum;
/// - `dead`: an unread family;
/// - `op`: a matrix (or a family member) of one operation, allocated in the plan's scratch;
/// - `integer`: a resident family of one-word integers (or Booleans);
/// - `resident`: every other family of matrices, held in its loop's member arena for the whole
///   plan.
pub(crate) fn residency(graph: &mxx_ir_core::ValidatedGraph) -> Vec<Residency> {
    use mxx_ir_core::{
        FrozenGraphScopeId as Scope,
        node::NodeKind,
        types::{ConcreteWireType as T, NodeId, Port, WireRef},
    };
    use std::collections::BTreeSet;
    struct Visit<'g> {
        graph: &'g mxx_ir_core::ValidatedGraph,
        rows: Vec<Residency>,
    }
    impl Visit<'_> {
        /// Visit scope `id`, whose body outputs at `exported` are families an enclosing loop
        /// exports per lane.
        fn scope(&mut self, id: &Scope, path: (String, String), exported: &BTreeSet<usize>) {
            let graph = self.graph;
            let (Some(scope), Some(types)) = (graph.source.scope(id), graph.scope(id)) else {
                return;
            };
            let readers = |wire: WireRef| {
                scope
                    .nodes()
                    .iter()
                    .enumerate()
                    .filter(|(_, node)| scope.arguments(node).is_some_and(|a| a.contains(&wire)))
                    .map(|(index, _)| NodeId(index as u64))
                    .collect::<Vec<_>>()
            };
            let root_output = |wire: WireRef| {
                matches!(id, Scope::Root) &&
                    graph.source.outputs().values().any(|output| output.value == wire)
            };
            let wire = |node: NodeId, port: usize| WireRef { node, port: Port(port as u32) };
            // A root loop flattening a family into an artifact output: read only by outputs.
            let flattens = |node: NodeId| {
                matches!(id, Scope::Root) &&
                    matches!(scope.nodes()[node.0 as usize].kind(), NodeKind::ParallelLoop(_)) &&
                    (0..scope.nodes()[node.0 as usize].output_types().len()).all(|port| {
                        let output = wire(node, port);
                        readers(output).is_empty() && root_output(output)
                    })
            };
            for (index, node) in scope.nodes().iter().enumerate() {
                if matches!(node.kind(), NodeKind::Input { .. }) {
                    // A capture of a value counted where it is produced, or an artifact
                    // imported member by member.
                    continue;
                }
                let node_id = NodeId(index as u64);
                let (looped, count, child) = match node.kind() {
                    NodeKind::ParallelLoop(spec) => (
                        Some(false),
                        spec.count.evaluate(&graph.bindings).ok(),
                        Some(Scope::ParallelBody { parent: Box::new(id.clone()), owner: node_id }),
                    ),
                    NodeKind::SequentialLoop(spec) => (
                        Some(true),
                        spec.count.evaluate(&graph.bindings).ok(),
                        Some(Scope::SequentialBody {
                            parent: Box::new(id.clone()),
                            owner: node_id,
                        }),
                    ),
                    _ => (None, None, None),
                };
                let summed = if looped == Some(false) {
                    count.as_ref().and_then(|n| usize::try_from(n.clone()).ok()).and_then(|count| {
                        mxx_ir_core::parallel_reduction::parallel_member_sums(
                            &graph.source,
                            id,
                            node_id,
                            node.output_types().len(),
                            count,
                            &graph.bindings,
                        )
                    })
                } else {
                    None
                };
                let mut child_exports = BTreeSet::new();
                let trapdoors = (0..node.output_types().len()).any(|port| {
                    types.wire_types.get(&wire(node_id, port)).is_some_and(|ty| {
                        matches!(ty, T::IndexedFamily { element, .. }
                            if matches!(element.as_ref(), T::Trapdoor { .. }))
                    })
                });
                for port in 0..node.output_types().len() {
                    let output = wire(node_id, port);
                    let Some(ty) = types.wire_types.get(&output) else { continue };
                    let family = matches!(ty, T::IndexedFamily { .. });
                    if !family && dimensions(ty).is_none() {
                        continue;
                    }
                    let readers = readers(output);
                    let scope_output = scope.outputs().iter().position(|out| *out == output);
                    let lane_export =
                        scope_output.is_some_and(|position| exported.contains(&position));
                    let streamed_matrix = matches!(ty, T::IndexedFamily { element, .. }
                        if matches!(element.as_ref(), T::Matrix(_) | T::SmallMatrix { .. }));
                    // A sequential body returns its matrix-family carries from its top-level
                    // parallel loops, which write them to the carry's next generation.
                    let carry_update = matches!(id, Scope::SequentialBody { .. }) &&
                        looped == Some(false) &&
                        scope_output.is_some() &&
                        readers.is_empty();
                    let class = if summed.as_ref().is_some_and(|(_, ports)| ports[port].is_some()) {
                        "summed"
                    } else if !family {
                        "op"
                    } else if (looped == Some(true) || carry_update) && streamed_matrix {
                        "streamed-carry"
                    } else if readers.is_empty() && scope_output.is_none() && !root_output(output) {
                        "dead"
                    } else if matches!(id, Scope::Root) &&
                        looped == Some(false) &&
                        readers.iter().all(|reader| flattens(*reader))
                    {
                        "exported"
                    } else if lane_export && readers.is_empty() {
                        "exported"
                    } else if lane_export && streamed_matrix {
                        "spilled"
                    } else if looped.is_none() && exported.contains(&usize::MAX) {
                        // A member of the family a flattening loop exports.
                        "exported"
                    } else if dimensions(ty).is_none() {
                        "integer"
                    } else {
                        "resident"
                    };
                    if class == "exported" || class == "spilled" {
                        child_exports.insert(port);
                    }
                    self.rows.push(Residency {
                        scope: path.0.clone(),
                        key: format!("{}/{index}:{port}", path.1),
                        node: format!("{:?}", node.kind()).chars().take(40).collect(),
                        shape: shape(ty),
                        count: match ty {
                            T::IndexedFamily { count, .. } => Some(*count),
                            _ => None,
                        },
                        dimensions: dimensions(ty),
                        bytes: wire_bytes(ty),
                        class,
                        trapdoors,
                    });
                }
                if let Some(child) = child {
                    // A flattening loop's body reads members of the exported family.
                    if flattens(node_id) {
                        child_exports.insert(usize::MAX);
                    }
                    let count = count.map_or("?".into(), |count| count.to_string());
                    let path =
                        (format!("{}/{index}[{count}]", path.0), format!("{}/{index}", path.1));
                    self.scope(&child, path, &child_exports);
                }
            }
        }
    }
    let mut visit = Visit { graph, rows: Vec::new() };
    visit.scope(&Scope::Root, (String::new(), String::new()), &BTreeSet::new());
    visit.rows
}
