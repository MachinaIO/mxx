//! Core typed graph IR for lattice-cryptography computations.
//!
//! This crate owns executable graph structure, compile expressions, concrete
//! type validation, canonical identities, and runtime artifact metadata.

pub mod artifact;
pub mod checks;
pub mod constraints;
pub mod encoding;
pub mod expr;
pub mod graph;
pub mod inventory;
pub mod lean;
pub mod linked;
pub mod node;
mod serde_support;
pub mod types;
pub mod validate;

pub use constraints::{ParamConstraint, derive_param_constraints};
pub use expr::{IndexExpr, IndexMap, IntExpr, ParamEnv, Rational, RealExpr};
pub use graph::{
    CapturePolicy, CapturedValue, CompileParameter, CompileParameterKind, ConstructionScopeId,
    FreezeError, FreezeMap, FreezeResolveError, FrozenGraphScopeId, FrozenStructuralIntExpr,
    FrozenValueRef, Graph, GraphOutput, GraphScope, NodeHandle, OutputRoot, ScopedWireRef, SealMap,
    SealedSubgraph, SourceLocation, SubgraphHandle, ValueHandle, current_construction_scope,
    with_new_construction_scope,
};
pub use lean::{
    LeanEmissionError, RenderedLeanModule, RenderedLeanProgram, render_child_input_hop,
    render_child_input_path, render_lean_program, render_parallel_output_hop,
    render_structural_value_route,
};
pub use linked::{
    ChildInputHop, ChildInputPathError, ConcreteArtifactInput, ConcreteArtifactLink,
    ConcreteGridInputMode, ConcreteIndexMap, ConcreteIndexMapExpr, ConcreteIndexRange,
    ConcreteLinkedProgram, ConcreteLinkedStage, ConcreteMatrixLiteral, ConcreteNamedOutput,
    ConcreteNode, ConcreteNodePayload, ConcreteParallelGrid, ConcreteRealExpr, ConcreteSampleRange,
    ConcreteScope, ConcreteSemanticWireRef, ConcreteSequentialLoop, ConcreteStructuralIntExpr,
    ConcreteSubgraphPayload, ConcreteWireRef, LinkedArtifactLink, LinkedProducerOutput,
    LinkedProgramError, LinkedProgramStage, ParallelOutputHop, SemanticLinkedProgram,
    StructuralSlotDecl, StructuralSlotKind, StructuralValueRoute, TypedScopedWireRef,
    ValidatedLinkedProgram, derive_child_input_path, derive_concrete_child_input_path,
    derive_structural_value_route, follow_child_input_hop, follow_concrete_child_input_hop,
    follow_concrete_parallel_output_hop, follow_concrete_structural_value_route,
    follow_parallel_output_hop, follow_structural_value_route, follows_child_input_path,
    follows_concrete_child_input_path, follows_concrete_structural_value_route,
    follows_structural_value_route,
};
pub use types::{NodeId, Port, WireRef, WireType};
pub use validate::{
    LivenessSchedule, ValidatedGraph, ValidatedScope, ValidationError, validate,
    validate_structure, validate_with_manifests,
};
