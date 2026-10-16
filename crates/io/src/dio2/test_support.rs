//! The toy parameters and function of the unit tests.

use super::{Dio2Parameters, StreamingFunction};
use bigdecimal::BigDecimal;
use mxx_backends::openfhe_guard::gen_modulus_and_warmup;
use mxx_exponent_lut::{ExpLayout, refresh::PrfSettings};

/// Ring dimension `8`; five primes of 50 bits with gadget base `2^12` and six levels of `Q_0`
/// (decoding level 4); the PRF on `Q = 16`, `P = 8` with one LWR coordinate and weight one; one
/// smudge digit; noise-free preprocessing (`MXX_IO_DIO2_TEST_{CRT_BITS,PRIMES,BASE_BITS,LEVELS}`
/// override the ring). The
/// function has one state bit, one evaluator bit and one obfuscator bit, on the state layout
/// `(1, 2)` (`mono(e) = X^(2 e)`, `e < 4`), and two table steps.
pub(crate) fn toy() -> (Dio2Parameters, StreamingFunction) {
    let read = |name: &str, default: usize| {
        std::env::var(format!("MXX_IO_DIO2_TEST_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    };
    let crt_bits = read("CRT_BITS", 50);
    let primes =
        gen_modulus_and_warmup(8, read("PRIMES", 5), crt_bits, None).expect("generated CRT basis");
    let parameters = Dio2Parameters {
        ring_dimension: 8,
        crt_bits,
        base_bits: read("BASE_BITS", 12) as u32,
        primes,
        levels: read("LEVELS", 6),
        prf: PrfSettings {
            lwr_dimension: 1,
            support_weight: 1,
            input_modulus: 16,
            output_modulus: 8,
            bucket_width: 0,
        },
        smudge_digits: 1,
        trapdoor_sigma: 4,
        encoding_error_sigma: BigDecimal::from(0),
    };
    let function =
        StreamingFunction::additive(1, 1, 1, ExpLayout { stride_log2: 1, width_log2: 2 });
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

/// One family or matrix of a validated graph that a GPU plan may hold at once: its scope (the
/// loop counts enclosing it), producing node, shape, bytes and residency class.
#[derive(Debug)]
pub(crate) struct Residency {
    pub(crate) scope: String,
    pub(crate) node: String,
    pub(crate) shape: String,
    pub(crate) bytes: u128,
    pub(crate) class: &'static str,
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

/// Every loop output family and sequential carry of `graph`, and every matrix wire of at least
/// `wide` bytes, with its class under the GPU planner's policy: a root loop's family that only
/// artifact outputs read is exported member by member (`exported`), a matrix-family carry is
/// streamed (`streamed-carry`), an unread family is `dead`, and every other family is `resident`.
pub(crate) fn residency(graph: &mxx_ir_core::ValidatedGraph, wide: u128) -> Vec<Residency> {
    use mxx_ir_core::{
        FrozenGraphScopeId as Scope,
        node::NodeKind,
        types::{NodeId, Port, WireRef},
    };
    fn visit(
        graph: &mxx_ir_core::ValidatedGraph,
        id: &Scope,
        path: String,
        wide: u128,
        rows: &mut Vec<Residency>,
    ) {
        let (Some(scope), Some(types)) = (graph.source.scope(id), graph.scope(id)) else { return };
        let reads = |wire: WireRef| {
            scope
                .nodes()
                .iter()
                .any(|node| scope.arguments(node).is_some_and(|a| a.contains(&wire)))
        };
        for (index, node) in scope.nodes().iter().enumerate() {
            // An input is a capture of a value counted where it is produced, or an artifact
            // imported member by member.
            if matches!(node.kind(), NodeKind::Input { .. }) {
                continue;
            }
            let node_id = NodeId(index as u64);
            let ports = node.output_types().len();
            let wire = |port: usize| WireRef { node: node_id, port: Port(port as u32) };
            let (label, count, child) = match node.kind() {
                NodeKind::ParallelLoop(spec) => (
                    "parallel",
                    spec.count
                        .evaluate(&graph.bindings)
                        .ok()
                        .map(|c| c.to_string())
                        .unwrap_or("?".into()),
                    Some(Scope::ParallelBody { parent: Box::new(id.clone()), owner: node_id }),
                ),
                NodeKind::SequentialLoop(spec) => (
                    "sequential",
                    spec.count
                        .evaluate(&graph.bindings)
                        .ok()
                        .map(|c| c.to_string())
                        .unwrap_or("?".into()),
                    Some(Scope::SequentialBody { parent: Box::new(id.clone()), owner: node_id }),
                ),
                _ => ("", String::new(), None),
            };
            for port in 0..ports {
                let Some(ty) = types.wire_types.get(&wire(port)) else { continue };
                let bytes = wire_bytes(ty);
                let family =
                    matches!(ty, mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. });
                let in_loop = child.is_some();
                if !(in_loop && family) && bytes < wide {
                    continue;
                }
                let output = scope.outputs().contains(&wire(port)) ||
                    matches!(id, Scope::Root) &&
                        graph.source.outputs().values().any(|o| o.value == wire(port));
                let class = if label == "sequential" &&
                    matches!(ty, mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. }
                        if matches!(element.as_ref(), mxx_ir_core::types::ConcreteWireType::Matrix(_) | mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. }))
                {
                    "streamed-carry"
                } else if !reads(wire(port)) && !output {
                    "dead"
                } else if matches!(id, Scope::Root) && label == "parallel" && !reads(wire(port)) {
                    "exported"
                } else {
                    "resident"
                };
                rows.push(Residency {
                    scope: path.clone(),
                    node: format!(
                        "{index}:{}{}",
                        if label.is_empty() {
                            format!("{:?}", node.kind()).chars().take(24).collect::<String>()
                        } else {
                            label.to_owned()
                        },
                        if count.is_empty() { String::new() } else { format!("[{count}]") }
                    ),
                    shape: shape(ty),
                    bytes,
                    class,
                });
            }
            if let Some(child) = child {
                visit(graph, &child, format!("{path}/{index}[{count}]"), wide, rows);
            }
        }
    }
    let mut rows = Vec::new();
    visit(graph, &Scope::Root, String::new(), wide, &mut rows);
    rows
}
