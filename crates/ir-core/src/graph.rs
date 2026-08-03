use crate::{
    artifact::ArtifactConfidentiality,
    expr::RealExpr,
    node::{NodeKind, ParallelLoop, SubgraphCall},
    types::{NodeId, Port, WireRef, WireType},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompileParameter {
    pub name: String,
    pub kind: CompileParameterKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CompileParameterKind {
    Integer,
    Real,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl SourceLocation {
    #[track_caller]
    pub fn caller() -> Self {
        let location = std::panic::Location::caller();
        Self { file: location.file().to_owned(), line: location.line(), column: location.column() }
    }
}

static NEXT_CONSTRUCTION_SCOPE: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static CONSTRUCTION_SCOPES: RefCell<Vec<ConstructionScopeId>> =
        const { RefCell::new(Vec::new()) };
}

/// Process-local scope marker used only while sealing closure bodies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ConstructionScopeId(u64);

impl ConstructionScopeId {
    fn fresh() -> Self {
        Self(NEXT_CONSTRUCTION_SCOPE.fetch_add(1, Ordering::Relaxed))
    }
}

pub fn current_construction_scope() -> ConstructionScopeId {
    CONSTRUCTION_SCOPES.with(|scopes| scopes.borrow().last().copied().unwrap_or(SelfRoot::ID))
}

struct SelfRoot;
impl SelfRoot {
    const ID: ConstructionScopeId = ConstructionScopeId(0);
}

pub fn with_new_construction_scope<T>(f: impl FnOnce(ConstructionScopeId) -> T) -> T {
    struct PopScope;
    impl Drop for PopScope {
        fn drop(&mut self) {
            CONSTRUCTION_SCOPES.with(|scopes| {
                scopes.borrow_mut().pop();
            });
        }
    }

    let scope = ConstructionScopeId::fresh();
    CONSTRUCTION_SCOPES.with(|scopes| scopes.borrow_mut().push(scope));
    let guard = PopScope;
    let output = f(scope);
    drop(guard);
    output
}

#[derive(Clone)]
pub struct NodeHandle(Arc<GraphNode>);

impl NodeHandle {
    #[track_caller]
    pub fn new(kind: NodeKind, arguments: Vec<ValueHandle>, output_types: Vec<WireType>) -> Self {
        Self::new_in_scope(
            current_construction_scope(),
            kind,
            arguments,
            output_types,
            Some(SourceLocation::caller()),
            None,
        )
    }

    fn new_in_scope(
        scope: ConstructionScopeId,
        kind: NodeKind,
        arguments: Vec<ValueHandle>,
        output_types: Vec<WireType>,
        source_location: Option<SourceLocation>,
        child: Option<StructuralChild>,
    ) -> Self {
        Self(Arc::new(GraphNode {
            kind,
            arguments,
            output_types,
            source_location,
            construction_scope: scope,
            child,
        }))
    }

    #[track_caller]
    pub fn subgraph_call(
        definition: SubgraphHandle,
        arguments: Vec<ValueHandle>,
        bindings: Vec<(String, crate::IntExpr)>,
    ) -> Self {
        let output_types = definition.output_types();
        Self::new_in_scope(
            current_construction_scope(),
            NodeKind::SubgraphCall(SubgraphCall {
                definition: definition.name().to_owned(),
                bindings,
            }),
            arguments,
            output_types,
            Some(SourceLocation::caller()),
            Some(StructuralChild::Subgraph(definition)),
        )
    }

    #[track_caller]
    pub fn parallel_loop(
        body: SubgraphHandle,
        arguments: Vec<ValueHandle>,
        output_types: Vec<WireType>,
        loop_spec: ParallelLoop,
    ) -> Self {
        Self::new_in_scope(
            current_construction_scope(),
            NodeKind::ParallelLoop(loop_spec),
            arguments,
            output_types,
            Some(SourceLocation::caller()),
            Some(StructuralChild::Parallel(body)),
        )
    }

    pub fn kind(&self) -> &NodeKind {
        &self.0.kind
    }

    pub fn arguments(&self) -> &[ValueHandle] {
        &self.0.arguments
    }

    pub fn output_types(&self) -> &[WireType] {
        &self.0.output_types
    }

    pub fn output(&self, port: u32) -> Option<ValueHandle> {
        ((port as usize) < self.0.output_types.len())
            .then(|| ValueHandle { node: self.clone(), port: Port(port) })
    }

    pub fn source_location(&self) -> Option<&SourceLocation> {
        self.0.source_location.as_ref()
    }

    pub fn construction_scope(&self) -> ConstructionScopeId {
        self.0.construction_scope
    }

    fn identity(&self) -> NodeIdentity {
        NodeIdentity(Arc::as_ptr(&self.0) as usize)
    }

    fn child(&self) -> Option<&StructuralChild> {
        self.0.child.as_ref()
    }
}

impl PartialEq for NodeHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for NodeHandle {}

impl Hash for NodeHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

impl fmt::Debug for NodeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeHandle")
            .field("instance", &self.identity().0)
            .field("kind", self.kind())
            .field("outputs", &self.output_types())
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct ValueHandle {
    node: NodeHandle,
    port: Port,
}

impl ValueHandle {
    pub fn node(&self) -> &NodeHandle {
        &self.node
    }

    pub fn port(&self) -> Port {
        self.port
    }

    pub fn wire_type(&self) -> &WireType {
        &self.node.output_types()[self.port.0 as usize]
    }

    pub fn construction_scope(&self) -> ConstructionScopeId {
        self.node.construction_scope()
    }
}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        self.port == other.port && self.node == other.node
    }
}

impl Eq for ValueHandle {}

impl Hash for ValueHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.port.hash(state);
    }
}

#[derive(Clone)]
pub struct SubgraphHandle(Arc<SubgraphDefinition>);

impl SubgraphHandle {
    pub fn new(
        name: impl Into<String>,
        scope: ConstructionScopeId,
        inputs: Vec<ValueHandle>,
        outputs: Vec<ValueHandle>,
    ) -> Result<Self, FreezeError> {
        let name = name.into();
        if inputs.iter().chain(&outputs).any(|value| value.construction_scope() != scope) {
            return Err(FreezeError::ForeignScope { graph: name });
        }
        Ok(Self(Arc::new(SubgraphDefinition { name, scope, inputs, outputs })))
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn inputs(&self) -> &[ValueHandle] {
        &self.0.inputs
    }

    pub fn outputs(&self) -> &[ValueHandle] {
        &self.0.outputs
    }

    pub fn output_types(&self) -> Vec<WireType> {
        self.outputs().iter().map(|value| value.wire_type().clone()).collect()
    }

    pub fn construction_scope(&self) -> ConstructionScopeId {
        self.0.scope
    }

    /// Seals a closure body and makes every permitted foreign value an
    /// explicit input. The returned captures are ordered exactly like the
    /// appended placeholder inputs.
    pub fn seal(
        name: impl Into<String>,
        scope: ConstructionScopeId,
        explicit_inputs: Vec<ValueHandle>,
        outputs: Vec<ValueHandle>,
        captures: CapturePolicy,
    ) -> Result<SealedSubgraph, FreezeError> {
        let name = name.into();
        if explicit_inputs.iter().any(|value| value.construction_scope() != scope) {
            return Err(FreezeError::ForeignScope { graph: name });
        }
        let mut sealer = ScopeSealer {
            scope,
            policy: captures,
            nodes: HashMap::new(),
            captured: Vec::new(),
            capture_inputs: HashMap::new(),
        };
        for input in &explicit_inputs {
            sealer.nodes.insert(input.node.identity(), input.node.clone());
        }
        let outputs =
            outputs.iter().map(|value| sealer.value(value)).collect::<Result<Vec<_>, _>>()?;
        let mut inputs = explicit_inputs;
        inputs.extend(sealer.captured.iter().map(|capture| capture.placeholder.clone()));
        let handle = SubgraphHandle::new(name, scope, inputs, outputs)?;
        let mut values = HashMap::new();
        for (old, new) in &sealer.nodes {
            for port in 0..new.output_types().len() {
                values.insert(
                    (*old, Port(port as u32)),
                    new.output(port as u32).expect("known output port"),
                );
            }
        }
        Ok(SealedSubgraph { handle, captures: sealer.captured, remap: SealMap { values } })
    }

    fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapturePolicy {
    Reject,
    /// Broadcast scalar values and read-only artifact families into a
    /// structural child scope.
    ///
    /// Arbitrary executable families remain rejected because their member
    /// dataflow cannot be reconstructed from an opaque family input.
    BroadcastScalarsAndArtifactFamilies,
}

#[derive(Clone, Debug)]
pub struct CapturedValue {
    pub outer: ValueHandle,
    pub placeholder: ValueHandle,
}

#[derive(Clone, Debug)]
pub struct SealedSubgraph {
    pub handle: SubgraphHandle,
    pub captures: Vec<CapturedValue>,
    pub remap: SealMap,
}

#[derive(Clone, Debug, Default)]
pub struct SealMap {
    values: HashMap<(NodeIdentity, Port), ValueHandle>,
}

impl SealMap {
    pub fn resolve(&self, value: &ValueHandle) -> Option<&ValueHandle> {
        self.values.get(&(value.node.identity(), value.port))
    }
}

struct ScopeSealer {
    scope: ConstructionScopeId,
    policy: CapturePolicy,
    nodes: HashMap<NodeIdentity, NodeHandle>,
    captured: Vec<CapturedValue>,
    capture_inputs: HashMap<(NodeIdentity, Port), ValueHandle>,
}

impl ScopeSealer {
    fn value(&mut self, value: &ValueHandle) -> Result<ValueHandle, FreezeError> {
        if value.construction_scope() != self.scope {
            return self.capture(value);
        }
        let identity = value.node.identity();
        let node = if let Some(node) = self.nodes.get(&identity) {
            node.clone()
        } else {
            let arguments = value
                .node
                .arguments()
                .iter()
                .map(|argument| self.value(argument))
                .collect::<Result<Vec<_>, _>>()?;
            let unchanged =
                arguments.iter().zip(value.node.arguments()).all(|(left, right)| left == right);
            let node = if unchanged {
                value.node.clone()
            } else {
                NodeHandle::new_in_scope(
                    self.scope,
                    value.node.kind().clone(),
                    arguments,
                    value.node.output_types().to_vec(),
                    value.node.source_location().cloned(),
                    value.node.0.child.clone(),
                )
            };
            self.nodes.insert(identity, node.clone());
            node
        };
        node.output(value.port.0).ok_or(FreezeError::InvalidPort { port: value.port.0 })
    }

    fn capture(&mut self, value: &ValueHandle) -> Result<ValueHandle, FreezeError> {
        if self.policy == CapturePolicy::Reject {
            return Err(FreezeError::ForeignScope { graph: "subgraph capture".to_owned() });
        }
        let artifact = match value.node.kind() {
            NodeKind::Input { artifact, .. } => artifact.clone(),
            _ => None,
        };
        let is_artifact_family = artifact.is_some();
        if matches!(value.wire_type(), WireType::IndexedFamily { .. }) && !is_artifact_family {
            return Err(FreezeError::ForeignScope { graph: "parallel family capture".to_owned() });
        }
        let key = (value.node.identity(), value.port);
        if let Some(input) = self.capture_inputs.get(&key) {
            return Ok(input.clone());
        }
        let name = format!("__capture_{}", self.captured.len());
        let ty = value.wire_type().clone();
        let input = NodeHandle::new_in_scope(
            self.scope,
            NodeKind::Input { name, wire_type: ty.clone(), artifact },
            Vec::new(),
            vec![ty],
            value.node.source_location().cloned(),
            None,
        )
        .output(0)
        .expect("input has one output");
        self.capture_inputs.insert(key, input.clone());
        self.captured.push(CapturedValue { outer: value.clone(), placeholder: input.clone() });
        Ok(input)
    }
}

impl fmt::Debug for SubgraphHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubgraphHandle")
            .field("name", &self.name())
            .field("instance", &self.identity())
            .finish()
    }
}

impl PartialEq for SubgraphHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SubgraphHandle {}

struct GraphNode {
    kind: NodeKind,
    arguments: Vec<ValueHandle>,
    output_types: Vec<WireType>,
    source_location: Option<SourceLocation>,
    construction_scope: ConstructionScopeId,
    child: Option<StructuralChild>,
}

#[derive(Clone)]
enum StructuralChild {
    Subgraph(SubgraphHandle),
    Parallel(SubgraphHandle),
}

struct SubgraphDefinition {
    name: String,
    scope: ConstructionScopeId,
    inputs: Vec<ValueHandle>,
    outputs: Vec<ValueHandle>,
}

#[derive(Clone, Debug)]
pub struct GraphOutput {
    pub value: ValueHandle,
    pub confidentiality: Option<ArtifactConfidentiality>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OutputRoot {
    pub value: WireRef,
    pub confidentiality: Option<ArtifactConfidentiality>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "tag", content = "value")]
pub enum FrozenGraphScopeId {
    Root,
    Subgraph { canonical_name: String },
    ParallelBody { parent: Box<FrozenGraphScopeId>, owner: NodeId },
}

#[derive(Clone)]
pub struct GraphScope {
    id: FrozenGraphScopeId,
    nodes: Vec<NodeHandle>,
    node_ids: HashMap<NodeIdentity, NodeId>,
    inputs: Vec<WireRef>,
    outputs: Vec<WireRef>,
}

impl GraphScope {
    pub fn id(&self) -> &FrozenGraphScopeId {
        &self.id
    }

    pub fn nodes(&self) -> &[NodeHandle] {
        &self.nodes
    }

    pub fn inputs(&self) -> &[WireRef] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[WireRef] {
        &self.outputs
    }

    pub fn node(&self, id: NodeId) -> Option<&NodeHandle> {
        self.nodes.get(id.0 as usize)
    }

    pub fn node_id(&self, node: &NodeHandle) -> Option<NodeId> {
        self.node_ids.get(&node.identity()).copied()
    }

    pub fn wire_ref(&self, value: &ValueHandle) -> Option<WireRef> {
        self.node_id(value.node()).map(|node| WireRef { node, port: value.port() })
    }

    pub fn arguments(&self, node: &NodeHandle) -> Option<Vec<WireRef>> {
        node.arguments().iter().map(|value| self.wire_ref(value)).collect()
    }
}

impl fmt::Debug for GraphScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphScope")
            .field("id", &self.id)
            .field("node_count", &self.nodes.len())
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

#[derive(Clone)]
pub struct Graph {
    name: String,
    parameters: Vec<CompileParameter>,
    outputs: BTreeMap<String, OutputRoot>,
    effect_roots: Vec<WireRef>,
    scopes: BTreeMap<FrozenGraphScopeId, GraphScope>,
    real_constants: BTreeMap<String, RealExpr>,
}

impl Graph {
    pub fn freeze(
        name: impl Into<String>,
        parameters: Vec<CompileParameter>,
        outputs: BTreeMap<String, GraphOutput>,
        retained_roots: Vec<ValueHandle>,
        effect_roots: Vec<ValueHandle>,
        real_constants: BTreeMap<String, RealExpr>,
    ) -> Result<(Self, FreezeMap), FreezeError> {
        let name = name.into();
        let root_values = outputs
            .values()
            .map(|output| output.value.clone())
            .chain(retained_roots)
            .collect::<Vec<_>>();
        let mut freeze_map = FreezeMap::default();
        let mut named = BTreeMap::<String, SubgraphHandle>::new();
        let root = freeze_scope(
            FrozenGraphScopeId::Root,
            SelfRoot::ID,
            &[],
            &root_values,
            &effect_roots,
            &mut freeze_map,
        )?;
        register_named_children(&root, &mut named)?;

        let mut scopes = BTreeMap::new();
        freeze_parallel_children(&root, &mut scopes, &mut freeze_map, &mut named)?;
        scopes.insert(FrozenGraphScopeId::Root, root);

        let mut completed_names = BTreeSet::new();
        loop {
            let next = named.keys().find(|name| !completed_names.contains(*name)).cloned();
            let Some(name) = next else { break };
            let definition = named.get(&name).expect("registered definition").clone();
            let id = FrozenGraphScopeId::Subgraph { canonical_name: name.clone() };
            let scope = freeze_scope(
                id.clone(),
                definition.construction_scope(),
                definition.inputs(),
                definition.outputs(),
                &[],
                &mut freeze_map,
            )?;
            register_named_children(&scope, &mut named)?;
            freeze_parallel_children(&scope, &mut scopes, &mut freeze_map, &mut named)?;
            scopes.insert(id, scope);
            completed_names.insert(name);
        }

        let root_scope = scopes.get(&FrozenGraphScopeId::Root).expect("root scope");
        let frozen_outputs = outputs
            .into_iter()
            .map(|(name, output)| {
                let value = root_scope
                    .wire_ref(&output.value)
                    .ok_or_else(|| FreezeError::UnreachableOutput { name: name.clone() })?;
                Ok((name, OutputRoot { value, confidentiality: output.confidentiality }))
            })
            .collect::<Result<BTreeMap<_, _>, FreezeError>>()?;
        let frozen_effects = effect_roots
            .iter()
            .map(|value| {
                root_scope
                    .wire_ref(value)
                    .ok_or_else(|| FreezeError::ForeignScope { graph: "effect root".to_owned() })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((
            Self {
                name,
                parameters,
                outputs: frozen_outputs,
                effect_roots: frozen_effects,
                scopes,
                real_constants,
            },
            freeze_map,
        ))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn parameters(&self) -> &[CompileParameter] {
        &self.parameters
    }

    pub fn outputs(&self) -> &BTreeMap<String, OutputRoot> {
        &self.outputs
    }

    pub fn effect_roots(&self) -> &[WireRef] {
        &self.effect_roots
    }

    pub fn scopes(&self) -> &BTreeMap<FrozenGraphScopeId, GraphScope> {
        &self.scopes
    }

    pub fn scope(&self, id: &FrozenGraphScopeId) -> Option<&GraphScope> {
        self.scopes.get(id)
    }

    pub fn root_scope(&self) -> &GraphScope {
        self.scope(&FrozenGraphScopeId::Root).expect("a frozen graph always has a root")
    }

    pub fn real_constants(&self) -> &BTreeMap<String, RealExpr> {
        &self.real_constants
    }

    pub fn child_scope_id(
        &self,
        parent: &FrozenGraphScopeId,
        node: NodeId,
    ) -> Option<FrozenGraphScopeId> {
        let handle = self.scope(parent)?.node(node)?;
        match handle.kind() {
            NodeKind::SubgraphCall(call) => {
                Some(FrozenGraphScopeId::Subgraph { canonical_name: call.definition.clone() })
            }
            NodeKind::ParallelLoop(_) => Some(FrozenGraphScopeId::ParallelBody {
                parent: Box::new(parent.clone()),
                owner: node,
            }),
            _ => None,
        }
    }

    fn serialized(&self) -> SerializedGraph {
        SerializedGraph {
            name: self.name.clone(),
            parameters: self.parameters.clone(),
            outputs: self.outputs.clone(),
            effect_roots: self.effect_roots.clone(),
            scopes: self
                .scopes
                .iter()
                .map(|(id, scope)| SerializedScopeEntry {
                    id: id.clone(),
                    scope: SerializedScope::from_scope(scope),
                })
                .collect(),
            real_constants: self.real_constants.clone(),
        }
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.serialized().fmt(f)
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.serialized() == other.serialized()
    }
}

impl Eq for Graph {}

impl Serialize for Graph {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialized().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Graph {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serialized = SerializedGraph::deserialize(deserializer)?;
        serialized.into_graph().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Default)]
pub struct FreezeMap {
    values: HashMap<(NodeIdentity, Port), ScopedWireRef>,
}

impl FreezeMap {
    pub fn resolve(&self, value: &ValueHandle) -> Option<&ScopedWireRef> {
        self.values.get(&(value.node.identity(), value.port))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ScopedWireRef {
    pub scope: FrozenGraphScopeId,
    pub wire: WireRef,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NodeIdentity(usize);

#[derive(Debug, Error)]
pub enum FreezeError {
    #[error("graph scope {graph} contains an edge to a foreign construction scope")]
    ForeignScope { graph: String },
    #[error("node argument refers to output port {port} that does not exist")]
    InvalidPort { port: u32 },
    #[error("distinct input nodes in one graph scope reuse name {name}")]
    DuplicateInput { name: String },
    #[error("different subgraph definitions reuse canonical name {name}")]
    DuplicateSubgraph { name: String },
    #[error("graph output {name} is not reachable")]
    UnreachableOutput { name: String },
    #[error("a graph node cycle was detected during freezing")]
    Cycle,
    #[error("invalid serialized graph: {0}")]
    InvalidSerialization(String),
}

fn freeze_scope(
    id: FrozenGraphScopeId,
    construction_scope: ConstructionScopeId,
    inputs: &[ValueHandle],
    outputs: &[ValueHandle],
    effects: &[ValueHandle],
    freeze_map: &mut FreezeMap,
) -> Result<GraphScope, FreezeError> {
    let mut roots = outputs.to_vec();
    roots.extend_from_slice(effects);
    roots.extend_from_slice(inputs);
    let nodes = canonical_postorder(&roots, construction_scope)?;
    let node_ids = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity(), NodeId(index as u64)))
        .collect::<HashMap<_, _>>();
    let resolve = |value: &ValueHandle| {
        let node = node_ids.get(&value.node.identity()).copied()?;
        Some(WireRef { node, port: value.port })
    };
    let frozen_inputs = inputs
        .iter()
        .map(&resolve)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| FreezeError::ForeignScope { graph: format!("{id:?}") })?;
    let frozen_outputs = outputs
        .iter()
        .map(&resolve)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| FreezeError::ForeignScope { graph: format!("{id:?}") })?;
    let mut input_names = BTreeSet::new();
    for node in &nodes {
        if let NodeKind::Input { name, .. } = node.kind() {
            if !input_names.insert(name.clone()) {
                return Err(FreezeError::DuplicateInput { name: name.clone() });
            }
        }
        let node_id = node_ids[&node.identity()];
        for port in 0..node.output_types().len() {
            freeze_map.values.insert(
                (node.identity(), Port(port as u32)),
                ScopedWireRef {
                    scope: id.clone(),
                    wire: WireRef { node: node_id, port: Port(port as u32) },
                },
            );
        }
    }
    Ok(GraphScope { id, nodes, node_ids, inputs: frozen_inputs, outputs: frozen_outputs })
}

fn canonical_postorder(
    roots: &[ValueHandle],
    scope: ConstructionScopeId,
) -> Result<Vec<NodeHandle>, FreezeError> {
    let mut states = HashMap::<NodeIdentity, u8>::new();
    let mut nodes = Vec::new();
    for root in roots {
        if root.construction_scope() != scope {
            return Err(FreezeError::ForeignScope { graph: "root".to_owned() });
        }
        let mut stack = vec![(root.node.clone(), false)];
        while let Some((node, exiting)) = stack.pop() {
            let identity = node.identity();
            if exiting {
                states.insert(identity, 2);
                nodes.push(node);
                continue;
            }
            match states.get(&identity).copied() {
                Some(2) => continue,
                Some(1) => return Err(FreezeError::Cycle),
                _ => {}
            }
            if node.construction_scope() != scope {
                return Err(FreezeError::ForeignScope { graph: "node".to_owned() });
            }
            states.insert(identity, 1);
            stack.push((node.clone(), true));
            for argument in node.arguments().iter().rev() {
                if argument.port.0 as usize >= argument.node.output_types().len() {
                    return Err(FreezeError::InvalidPort { port: argument.port.0 });
                }
                if argument.construction_scope() != scope {
                    return Err(FreezeError::ForeignScope { graph: "argument".to_owned() });
                }
                if states.get(&argument.node.identity()).copied() != Some(2) {
                    stack.push((argument.node.clone(), false));
                }
            }
        }
    }
    Ok(nodes)
}

fn register_named_children(
    scope: &GraphScope,
    named: &mut BTreeMap<String, SubgraphHandle>,
) -> Result<(), FreezeError> {
    for node in scope.nodes() {
        let Some(StructuralChild::Subgraph(definition)) = node.child() else { continue };
        if let Some(existing) = named.get(definition.name()) {
            if existing != definition {
                return Err(FreezeError::DuplicateSubgraph { name: definition.name().to_owned() });
            }
        } else {
            named.insert(definition.name().to_owned(), definition.clone());
        }
    }
    Ok(())
}

fn freeze_parallel_children(
    parent: &GraphScope,
    scopes: &mut BTreeMap<FrozenGraphScopeId, GraphScope>,
    freeze_map: &mut FreezeMap,
    named: &mut BTreeMap<String, SubgraphHandle>,
) -> Result<(), FreezeError> {
    for node in parent.nodes() {
        let Some(StructuralChild::Parallel(body)) = node.child() else { continue };
        let owner = parent.node_id(node).expect("frozen parent node");
        let id = FrozenGraphScopeId::ParallelBody { parent: Box::new(parent.id().clone()), owner };
        let scope = freeze_scope(
            id.clone(),
            body.construction_scope(),
            body.inputs(),
            body.outputs(),
            &[],
            freeze_map,
        )?;
        register_named_children(&scope, named)?;
        freeze_parallel_children(&scope, scopes, freeze_map, named)?;
        scopes.insert(id, scope);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct SerializedGraph {
    name: String,
    parameters: Vec<CompileParameter>,
    outputs: BTreeMap<String, OutputRoot>,
    effect_roots: Vec<WireRef>,
    scopes: Vec<SerializedScopeEntry>,
    real_constants: BTreeMap<String, RealExpr>,
}

impl SerializedGraph {
    fn into_graph(self) -> Result<Graph, FreezeError> {
        let mut scopes = BTreeMap::new();
        for entry in self.scopes {
            if scopes.insert(entry.id.clone(), entry.scope.into_scope(entry.id.clone())?).is_some()
            {
                return Err(FreezeError::InvalidSerialization(
                    "duplicate frozen graph scope id".to_owned(),
                ));
            }
        }
        if !scopes.contains_key(&FrozenGraphScopeId::Root) {
            return Err(FreezeError::InvalidSerialization("missing root scope".to_owned()));
        }
        Ok(Graph {
            name: self.name,
            parameters: self.parameters,
            outputs: self.outputs,
            effect_roots: self.effect_roots,
            scopes,
            real_constants: self.real_constants,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct SerializedScopeEntry {
    id: FrozenGraphScopeId,
    scope: SerializedScope,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct SerializedScope {
    nodes: Vec<SerializedNode>,
    inputs: Vec<WireRef>,
    outputs: Vec<WireRef>,
}

impl SerializedScope {
    fn from_scope(scope: &GraphScope) -> Self {
        let nodes = scope
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| SerializedNode {
                id: NodeId(index as u64),
                kind: node.kind().clone(),
                arguments: scope.arguments(node).expect("same-scope arguments"),
                output_types: node.output_types().to_vec(),
            })
            .collect();
        Self { nodes, inputs: scope.inputs.clone(), outputs: scope.outputs.clone() }
    }

    fn into_scope(self, id: FrozenGraphScopeId) -> Result<GraphScope, FreezeError> {
        let scope = ConstructionScopeId::fresh();
        let mut nodes = Vec::<NodeHandle>::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.into_iter().enumerate() {
            if node.id != NodeId(index as u64) {
                return Err(FreezeError::InvalidSerialization(
                    "node ids are not canonical and contiguous".to_owned(),
                ));
            }
            let arguments = node
                .arguments
                .iter()
                .map(|wire| {
                    let source = nodes.get(wire.node.0 as usize).ok_or_else(|| {
                        FreezeError::InvalidSerialization("forward or missing edge".to_owned())
                    })?;
                    source.output(wire.port.0).ok_or(FreezeError::InvalidPort { port: wire.port.0 })
                })
                .collect::<Result<Vec<_>, _>>()?;
            nodes.push(NodeHandle::new_in_scope(
                scope,
                node.kind,
                arguments,
                node.output_types,
                None,
                None,
            ));
        }
        let node_ids = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.identity(), NodeId(index as u64)))
            .collect();
        Ok(GraphScope { id, nodes, node_ids, inputs: self.inputs, outputs: self.outputs })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct SerializedNode {
    id: NodeId,
    kind: NodeKind,
    arguments: Vec<WireRef>,
    output_types: Vec<WireType>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntExpr, node::MatrixBinaryOp, types::MatrixType};

    fn matrix_type() -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        }
    }

    fn input(name: &str) -> ValueHandle {
        let ty = WireType::Matrix(matrix_type());
        NodeHandle::new(
            NodeKind::Input { name: name.to_owned(), wire_type: ty.clone(), artifact: None },
            Vec::new(),
            vec![ty],
        )
        .output(0)
        .unwrap()
    }

    #[test]
    fn shared_handles_freeze_once_and_unreachable_nodes_are_omitted() {
        let value = input("x");
        let _unused = input("unused");
        let sum = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            vec![value.clone(), value],
            vec![WireType::Matrix(matrix_type())],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "sharing",
            Vec::new(),
            BTreeMap::from([("out".to_owned(), GraphOutput { value: sum, confidentiality: None })]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(graph.root_scope().nodes().len(), 2);
    }

    #[test]
    fn canonical_serialization_round_trips_sharing() {
        let value = input("x");
        let (graph, _) = Graph::freeze(
            "round-trip",
            Vec::new(),
            BTreeMap::from([("out".to_owned(), GraphOutput { value, confidentiality: None })]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap();
        let encoded = serde_json::to_vec(&graph).unwrap();
        let decoded: Graph = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(graph, decoded);
    }
}
