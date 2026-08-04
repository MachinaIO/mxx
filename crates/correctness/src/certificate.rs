//! Serializable structural certificates for protocol graphs.
//!
//! A certificate identifies executable core-IR wiring. It does not introduce
//! equations, proof claims, runtime values, ciphertext fields, or artifacts.
//! In particular, Boolean-gate RHS decompositions remain deterministic local
//! graph operations and are never exported as protocol data.

use crate::{ProtocolDecl, StageId};
use mxx_ir_core::{
    FrozenGraphScopeId, IntExpr, NodeId, Port, WireRef,
    node::{ConcatAxis, IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp, NodeKind},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CoreNodeRef {
    pub stage: StageId,
    pub scope: FrozenGraphScopeId,
    pub node: NodeId,
}

impl CoreNodeRef {
    pub fn new(stage: StageId, scope: FrozenGraphScopeId, node: NodeId) -> Self {
        Self { stage, scope, node }
    }

    pub fn wire(&self, port: u32) -> CoreWireRef {
        CoreWireRef { node: self.clone(), port: Port(port) }
    }

    pub fn operand(&self, operand: u32, wire: CoreWireRef) -> CoreOperandRef {
        CoreOperandRef { node: self.clone(), operand, wire }
    }

    pub fn parameter(&self, parameter: CoreNodeParameter) -> CoreNodeParameterRef {
        CoreNodeParameterRef { node: self.clone(), parameter }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CoreWireRef {
    pub node: CoreNodeRef,
    pub port: Port,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CoreOperandRef {
    pub node: CoreNodeRef,
    pub operand: u32,
    pub wire: CoreWireRef,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum CoreNodeParameter {
    GadgetDecomposeBase,
    GadgetDecomposeDigitCount,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CoreNodeParameterRef {
    pub node: CoreNodeRef,
    pub parameter: CoreNodeParameter,
}

/// One complete stage interface. The vectors use frozen root-node/output order.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageInterfaceLayout {
    pub stage: StageId,
    pub inputs: Vec<StageInputLayout>,
    pub outputs: Vec<StageOutputLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageInputLayout {
    pub name: String,
    pub node: CoreNodeRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageOutputLayout {
    pub name: String,
    pub wire: CoreWireRef,
}

/// Exact producer/consumer provenance for one workflow artifact binding.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ArtifactProvenance {
    pub producer_stage: StageId,
    pub producer_output: StageOutputLayout,
    pub consumer_stage: StageId,
    pub consumer_input: StageInputLayout,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DiamondWorkflowLayout {
    pub encryption: StageInterfaceLayout,
    pub decryption: StageInterfaceLayout,
    pub artifacts: Vec<ArtifactProvenance>,
}

/// Exact operands and outputs of one construction-time identified core operation.
/// The validator, rather than this untrusted record, determines the operation's kind.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct OperationRef {
    pub operation: CoreNodeRef,
    pub inputs: Vec<CoreOperandRef>,
    pub outputs: Vec<CoreWireRef>,
}

/// One parallel-loop boundary whose child performs one identified operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelOperationRef {
    pub parallel_loop: ParallelLoopRef,
    pub body: OperationRef,
}

/// One parallel loop whose scalar output is defined by a fixed, verifier-known
/// integer formula rooted at `body_output`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelIndexFormulaRef {
    pub parallel_loop: ParallelLoopRef,
    pub body_output: CoreWireRef,
}

/// The online input-injection base family.  Its fixed formula places the
/// imported initial state at index zero and zero matrices at every other index.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct InitialStateExpansionRef {
    pub parallel_loop: ParallelLoopRef,
    pub body_output: CoreWireRef,
}

/// Parallel little-endian packing whose body contains one sequential
/// `(sum, weight)` scan.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WitnessDigitPackingRef {
    pub parallel_loop: ParallelLoopRef,
    pub body_output: CoreWireRef,
    pub bit_scan: SequentialLoopRef,
}

/// One sequential-loop boundary, preserving the exact carried/invariant
/// operands and the child-scope interface in frozen order.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SequentialLoopRef {
    pub operation: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub count: IntExpr,
    pub index_slot: u32,
    pub bindings: Vec<(String, IntExpr)>,
    pub carried_count: usize,
    pub arguments: Vec<CoreOperandRef>,
    pub body_inputs: Vec<CoreWireRef>,
    pub body_outputs: Vec<CoreWireRef>,
    pub outputs: Vec<CoreWireRef>,
}

/// The Boolean message-to-matrix conversion used by Diamond encryption.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MessageConstructionLayout {
    pub to_int: OperationRef,
    pub zero: OperationRef,
    pub one: OperationRef,
    pub select: OperationRef,
}

/// Packed hash sampling followed by deterministic per-key slices.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct BggPublicKeySamplingLayout {
    pub public_keys_artifact: ArtifactProvenance,
    pub packed_hash: OperationRef,
    pub slices: ParallelOperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelPackedPublicKeyLayout {
    pub parallel_loop: ParallelLoopRef,
    pub in_range: OperationRef,
    pub padded: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelCircuitInputPublicKeyLayout {
    pub parallel_loop: ParallelLoopRef,
    pub selected_instance: OperationRef,
    pub selected_source: OperationRef,
}

/// Construction of the public-key family consumed by the encryption Boolean
/// evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EncryptionInitialPublicKeysLayout {
    pub one_public_key: OperationRef,
    pub zero_public_key: OperationRef,
    pub instance_width: EvaluateIntRef,
    pub public_indices: ParallelLoopRef,
    pub public_candidates: ParallelGatherRef,
    pub packed_inputs: ParallelPackedPublicKeyLayout,
    pub circuit_inputs: ParallelCircuitInputPublicKeyLayout,
}

/// Exact preimage sampling followed by its matrix materialization.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct PreimageRef {
    pub sample: OperationRef,
    pub materialize: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelPreimageRef {
    pub parallel_loop: ParallelLoopRef,
    pub body: PreimageRef,
}

/// Exact elementwise gather for either a scalar family or a multi-wire value
/// such as a trapdoor.  Entries retain construction order throughout the loop
/// boundary and the child `FamilyGetDynamic` operations.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelGatherRef {
    pub parallel_loop: ParallelLoopRef,
    pub index_family: CoreOperandRef,
    pub source_families: Vec<CoreOperandRef>,
    pub body_index: CoreWireRef,
    pub body_sources: Vec<CoreWireRef>,
    pub gets: Vec<DynamicFamilyGetRef>,
    pub output_families: Vec<CoreWireRef>,
}

/// One iteration of the selector's bit-by-bit sequential refinement.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TransitionSelectorBitLayout {
    pub bit_extract: OperationRef,
    pub bit_to_int: OperationRef,
    pub bit_zero: OperationRef,
    pub bit_one: OperationRef,
    pub bit_select: OperationRef,
    pub special_product: OperationRef,
    pub special_top: OperationRef,
    pub special_bottom: OperationRef,
    pub special: OperationRef,
    pub state_match: OperationRef,
    pub state_match_to_int: OperationRef,
    pub selector: OperationRef,
}

/// Exact executable construction of one transition selector.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TransitionSelectorLayout {
    pub regular: OperationRef,
    pub k_identity: OperationRef,
    pub k: OperationRef,
    pub initial_select: OperationRef,
    pub bit_scan: SequentialLoopRef,
    pub bit_body: TransitionSelectorBitLayout,
}

/// One target matrix `selector * target_public + error` used by preprocessing.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TransitionTargetRef {
    pub digit_secret: CoreWireRef,
    pub target_public: CoreWireRef,
    pub selector: CoreWireRef,
    pub selector_construction: TransitionSelectorLayout,
    pub error_sample: OperationRef,
    pub selector_product: OperationRef,
    pub target_sum: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelTransitionTargetRef {
    pub parallel_loop: ParallelLoopRef,
    pub body: TransitionTargetRef,
}

/// Complete construction trace for the reusable Diamond input preprocessing gadget.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DiamondInputPreprocessingLayout {
    pub initial_state_artifact: ArtifactProvenance,
    pub transitions_artifact: ArtifactProvenance,
    pub trapdoor_samples: ParallelOperationRef,
    pub secret_sample: OperationRef,
    pub message_selector: OperationRef,
    pub initial_error_sample: OperationRef,
    pub initial_public_product: OperationRef,
    pub initial_state: OperationRef,
    pub transition_source_indices: ParallelIndexFormulaRef,
    pub transition_target_indices: ParallelIndexFormulaRef,
    pub digit_secret_indices: ParallelIndexFormulaRef,
    pub digit_secret_samples: ParallelOperationRef,
    pub digit_secrets: ParallelGatherRef,
    pub transition_sources: ParallelGatherRef,
    pub target_public_matrices: ParallelGatherRef,
    pub transition_targets: ParallelTransitionTargetRef,
    pub transition_preimages: ParallelPreimageRef,
    pub final_indices: ParallelLoopRef,
    pub final_trapdoors: ParallelGatherRef,
}

/// The one online Diamond input-injection recurrence.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct InputInjectionLayout {
    pub state_scan: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub initial_states_expansion: InitialStateExpansionRef,
    pub initial_states: CoreOperandRef,
    pub packed_digits: CoreOperandRef,
    pub transition_family: CoreOperandRef,
    pub final_states: CoreWireRef,
    pub body_initial_states: CoreWireRef,
    pub body_packed_digits: CoreWireRef,
    pub body_transition_family: CoreWireRef,
    pub selected_digit: DynamicFamilyGetRef,
    pub source_indices: ParallelIndexFormulaRef,
    pub source_states: ParallelFamilyGetRef,
    pub transition_indices: ParallelIndexFormulaRef,
    pub selected_transitions: ParallelFamilyGetRef,
    pub body_final_states: CoreWireRef,
    pub state_product: ParallelMatrixBinaryRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct OneTargetLayout {
    pub gadget: OperationRef,
    pub difference: OperationRef,
    pub zero_row: OperationRef,
    pub target: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StaticTrapdoorLayout {
    pub public: OperationRef,
    pub secret: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelWitnessTargetLayout {
    pub parallel_loop: ParallelLoopRef,
    pub negated_gadget: OperationRef,
    pub target: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct KTargetLayout {
    pub public_key_hash: OperationRef,
    pub first_column: OperationRef,
    pub half_modulus: OperationRef,
    pub target: OperationRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecoderTargetLayout {
    pub public_key_difference: OperationRef,
    pub projected_difference: OperationRef,
    pub public_key_sum: OperationRef,
    pub zero: OperationRef,
    pub target: OperationRef,
}

/// Construction of all application-specific Diamond artifacts after reusable
/// input preprocessing and Boolean public-key evaluation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DiamondArtifactPreprocessingLayout {
    pub one_preimage_artifact: ArtifactProvenance,
    pub witness_preimages_artifact: ArtifactProvenance,
    pub k_preimage_artifact: ArtifactProvenance,
    pub r_decomposed_artifact: ArtifactProvenance,
    pub decoder_preimage_artifact: ArtifactProvenance,
    pub projection_trapdoor: StaticTrapdoorLayout,
    pub one_target: OneTargetLayout,
    pub one_preimage: PreimageRef,
    pub witness_indices: ParallelLoopRef,
    pub witness_trapdoors: ParallelGatherRef,
    pub witness_public_keys: ParallelGatherRef,
    pub witness_targets: ParallelWitnessTargetLayout,
    pub witness_preimages: ParallelPreimageRef,
    pub k_target: KTargetLayout,
    pub k_preimage: PreimageRef,
    pub r_hash: OperationRef,
    pub r_slice: OperationRef,
    pub r_decomposition: OperationRef,
    pub r_materialization: OperationRef,
    pub r_reshape: OperationRef,
    pub decoder_target: DecoderTargetLayout,
    pub decoder_preimage: PreimageRef,
}

/// Three synchronized vector/public-key/plaintext parallel operations.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EncodingComponentOperationsLayout {
    pub vectors: ParallelOperationRef,
    pub public_keys: ParallelOperationRef,
    pub plaintexts: ParallelOperationRef,
}

/// The complete construction of the initial encoding families consumed by the
/// decryption Boolean evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecryptionInitialEncodingsLayout {
    pub initial_state_artifact: ArtifactProvenance,
    pub one_preimage_artifact: ArtifactProvenance,
    pub witness_preimages_artifact: ArtifactProvenance,
    pub public_keys_artifact: ArtifactProvenance,
    pub witness_indices: ParallelLoopRef,
    pub witness_bits: ParallelGatherRef,
    pub witness_digits: WitnessDigitPackingRef,
    pub initial_projection_state: OperationRef,
    pub one_public_key: OperationRef,
    pub one_plaintext: OperationRef,
    pub zero_encoding: [OperationRef; 3],
    pub witness_state_indices: ParallelLoopRef,
    pub witness_states: ParallelGatherRef,
    pub witness_vectors: ParallelMatrixBinaryRef,
    pub witness_public_indices: ParallelLoopRef,
    pub witness_public_keys: ParallelGatherRef,
    pub witness_plaintext_constants: [ParallelLoopRef; 2],
    pub witness_plaintexts: ParallelOperationRef,
    pub instance_width: EvaluateIntRef,
    pub packed_indices: ParallelLoopRef,
    pub packed_vectors: ParallelGatherRef,
    pub packed_public_keys: ParallelGatherRef,
    pub packed_plaintexts: ParallelGatherRef,
    pub active_witness: ParallelLoopRef,
    pub active_witness_zeroes: [ParallelLoopRef; 3],
    pub active_witness_selection: EncodingComponentOperationsLayout,
    pub instance_constants: [[ParallelLoopRef; 2]; 3],
    pub selected_instance: EncodingComponentOperationsLayout,
    pub active_instance: ParallelLoopRef,
    pub circuit_inputs: EncodingComponentOperationsLayout,
}

/// Exact dynamic selection from the final carried family.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DynamicFamilyGetRef {
    pub operation: CoreNodeRef,
    pub family: CoreOperandRef,
    pub index: CoreOperandRef,
    pub output: CoreWireRef,
}

/// One complete parallel-loop boundary, including the outer operands, child
/// inputs and outputs, and exposed output families.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelLoopRef {
    pub operation: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub count: IntExpr,
    pub index_slot: u32,
    pub bindings: Vec<(String, IntExpr)>,
    pub input_modes: Vec<CertifiedLoopInputMode>,
    pub arguments: Vec<CoreOperandRef>,
    pub body_inputs: Vec<CoreWireRef>,
    pub body_outputs: Vec<CoreWireRef>,
    pub outputs: Vec<CoreWireRef>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum CertifiedLoopInputMode {
    Broadcast,
    Zip,
    ZipOffset { offset: usize },
}

impl From<&LoopInputMode> for CertifiedLoopInputMode {
    fn from(value: &LoopInputMode) -> Self {
        match value {
            LoopInputMode::Broadcast => Self::Broadcast,
            LoopInputMode::Zip => Self::Zip,
            LoopInputMode::ZipOffset { offset } => Self::ZipOffset { offset: *offset },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EvaluateIntRef {
    pub operation: CoreNodeRef,
    pub expression: IntExpr,
    pub evaluated: CoreWireRef,
    pub materialization: Option<BinaryNodeRef>,
    pub output: CoreWireRef,
}

/// Exact elementwise family gather: the first loop input is the index family,
/// the second is the source family, and the body performs one dynamic get.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelFamilyGetRef {
    pub parallel_loop: ParallelLoopRef,
    pub index_family: CoreOperandRef,
    pub source_family: CoreOperandRef,
    pub body_index: CoreWireRef,
    pub body_source: CoreWireRef,
    pub get: DynamicFamilyGetRef,
    pub output_family: CoreWireRef,
}

/// The active-count scalar selected for one sequential layer.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct LayerScalarMetadataRef {
    pub source_input_name: String,
    pub root_input: CoreWireRef,
    pub sequential_operand: CoreOperandRef,
    pub body_source: CoreWireRef,
    pub layer_index: EvaluateIntRef,
    pub selected: DynamicFamilyGetRef,
}

/// One opcode/source family sliced at the current sequential layer.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct LayerFamilyMetadataRef {
    pub source_input_name: String,
    pub root_input: CoreWireRef,
    pub sequential_operand: CoreOperandRef,
    pub body_source: CoreWireRef,
    pub flattened_indices: ParallelLoopRef,
    pub flattened_index: EvaluateIntRef,
    pub gathered: ParallelFamilyGetRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct BooleanLayerMetadataLayout {
    pub active_gate_count: LayerScalarMetadataRef,
    pub opcode: LayerFamilyMetadataRef,
    pub left_source: LayerFamilyMetadataRef,
    pub right_source: LayerFamilyMetadataRef,
}

/// The public-key-only Boolean layer scan in the encryption stage.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct PublicKeyBooleanLoopLayout {
    pub layer_scan: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub initial_public_keys: CoreOperandRef,
    pub active_gate_counts: CoreOperandRef,
    pub gate_kinds: CoreOperandRef,
    pub left_sources: CoreOperandRef,
    pub right_sources: CoreOperandRef,
    pub one_public_key: CoreOperandRef,
    pub final_public_keys: CoreWireRef,
    pub body_initial_public_keys: CoreWireRef,
    pub body_active_gate_counts: CoreWireRef,
    pub body_gate_kinds: CoreWireRef,
    pub body_left_sources: CoreWireRef,
    pub body_right_sources: CoreWireRef,
    pub body_one_public_key: CoreWireRef,
    pub body_final_public_keys: CoreWireRef,
    pub metadata: BooleanLayerMetadataLayout,
    pub selected_output: DynamicFamilyGetRef,
}

/// The vector/public-key/plaintext Boolean layer scan in the decryption stage.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EncodingBooleanLoopLayout {
    pub layer_scan: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub initial_vectors: CoreOperandRef,
    pub initial_public_keys: CoreOperandRef,
    pub initial_plaintexts: CoreOperandRef,
    pub active_gate_counts: CoreOperandRef,
    pub gate_kinds: CoreOperandRef,
    pub left_sources: CoreOperandRef,
    pub right_sources: CoreOperandRef,
    pub one_vector: CoreOperandRef,
    pub one_public_key: CoreOperandRef,
    pub one_plaintext: CoreOperandRef,
    pub final_vectors: CoreWireRef,
    pub final_public_keys: CoreWireRef,
    pub final_plaintexts: CoreWireRef,
    pub body_initial_vectors: CoreWireRef,
    pub body_initial_public_keys: CoreWireRef,
    pub body_initial_plaintexts: CoreWireRef,
    pub body_active_gate_counts: CoreWireRef,
    pub body_gate_kinds: CoreWireRef,
    pub body_left_sources: CoreWireRef,
    pub body_right_sources: CoreWireRef,
    pub body_one_vector: CoreWireRef,
    pub body_one_public_key: CoreWireRef,
    pub body_one_plaintext: CoreWireRef,
    pub body_final_vectors: CoreWireRef,
    pub body_final_public_keys: CoreWireRef,
    pub body_final_plaintexts: CoreWireRef,
    pub metadata: BooleanLayerMetadataLayout,
    pub selected_vector: DynamicFamilyGetRef,
}

/// Common local node data for deterministic gadget decomposition.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct LocalGadgetDecompositionRef {
    pub decomposition_node: CoreNodeRef,
    pub right_public_key: CoreOperandRef,
    pub base: CoreNodeParameterRef,
    pub digit_count: CoreNodeParameterRef,
    pub decomposition: CoreWireRef,
    pub materialized: CoreWireRef,
}

impl LocalGadgetDecompositionRef {
    pub fn new(decomposition_node: CoreNodeRef, right_public_key: CoreWireRef) -> Self {
        Self {
            right_public_key: decomposition_node.operand(0, right_public_key),
            base: decomposition_node.parameter(CoreNodeParameter::GadgetDecomposeBase),
            digit_count: decomposition_node.parameter(CoreNodeParameter::GadgetDecomposeDigitCount),
            decomposition: decomposition_node.wire(0),
            materialized: decomposition_node.wire(0),
            decomposition_node,
        }
    }
}

/// Encryption computes the RHS decomposition in the gate-evaluation loop body
/// and has exactly one local matrix-multiplication consumer.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct EncryptPublicKeyRhsDecomposition {
    pub right_selection: ParallelFamilyGetRef,
    pub enclosing_parallel_loop: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub right_public_key_family: CoreOperandRef,
    pub body_right_public_key: CoreWireRef,
    pub local: LocalGadgetDecompositionRef,
    pub multiplication_consumer: CoreOperandRef,
}

/// One consumer of a family produced by a deterministic parallel decomposition.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelDecompositionConsumer {
    pub consumer_loop: CoreNodeRef,
    pub decomposition_family: CoreOperandRef,
    pub body_scope: FrozenGraphScopeId,
    pub body_decomposition: CoreWireRef,
    pub multiplication_consumer: CoreOperandRef,
}

/// Decryption computes one RHS decomposition family and reuses it in exactly
/// the public-key and encoding-vector multiplication loops.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecryptEncodingRhsDecomposition {
    pub right_selection: ParallelFamilyGetRef,
    pub decomposition_loop: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub right_public_key_family: CoreOperandRef,
    pub body_right_public_key: CoreWireRef,
    pub local: LocalGadgetDecompositionRef,
    pub body_output: CoreWireRef,
    pub decomposition_family: CoreWireRef,
    pub public_key_consumer: ParallelDecompositionConsumer,
    pub vector_consumer: ParallelDecompositionConsumer,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct BooleanLayersLayout {
    pub public_keys_artifact: ArtifactProvenance,
    pub encryption: PublicKeyBooleanLoopLayout,
    pub decryption: EncodingBooleanLoopLayout,
    pub encrypt_public_key_rhs_decomposition: EncryptPublicKeyRhsDecomposition,
    pub decrypt_encoding_rhs_decomposition: DecryptEncodingRhsDecomposition,
    pub encryption_gate: Box<LocalBooleanGateLayout>,
    pub decryption_vectors: Box<FamilyBooleanGateLayout>,
    pub decryption_public_keys: Box<FamilyBooleanGateLayout>,
    pub decryption_plaintexts: Box<FamilyBooleanGateLayout>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct MatrixBinaryRef {
    pub operation: CoreNodeRef,
    pub left: CoreOperandRef,
    pub right: CoreOperandRef,
    pub output: CoreWireRef,
}

impl MatrixBinaryRef {
    pub fn new(operation: CoreNodeRef, left: CoreWireRef, right: CoreWireRef) -> Self {
        Self {
            left: operation.operand(0, left),
            right: operation.operand(1, right),
            output: operation.wire(0),
            operation,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SixWaySelectRef {
    pub operation: CoreNodeRef,
    pub selector: CoreOperandRef,
    pub branches: [CoreOperandRef; 6],
    pub output: CoreWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct TwoWaySelectRef {
    pub operation: CoreNodeRef,
    pub selector: CoreOperandRef,
    pub branches: [CoreOperandRef; 2],
    pub output: CoreWireRef,
}

/// Complete local gate body used by the encryption public-key evaluator.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct LocalBooleanGateLayout {
    pub body_scope: FrozenGraphScopeId,
    pub parent_loop: ParallelLoopRef,
    pub opcode_family: CoreOperandRef,
    pub left_family: CoreOperandRef,
    pub right_family: CoreOperandRef,
    pub one_public_key: CoreOperandRef,
    pub active_gate_count: CoreOperandRef,
    pub left_selection: ParallelFamilyGetRef,
    pub body_opcode: CoreWireRef,
    pub body_left: CoreWireRef,
    pub body_right: CoreWireRef,
    pub body_one_public_key: CoreWireRef,
    pub body_active_gate_count: CoreWireRef,
    pub zero: MatrixBinaryRef,
    pub one: CoreWireRef,
    pub copy: CoreWireRef,
    pub not: MatrixBinaryRef,
    pub product: MatrixBinaryRef,
    pub sum: MatrixBinaryRef,
    pub two_product: MatrixBinaryRef,
    pub xor: MatrixBinaryRef,
    pub candidate_select: SixWaySelectRef,
    pub active_select: TwoWaySelectRef,
}

/// A matrix binary operation performed elementwise by one parallel loop.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelMatrixBinaryRef {
    pub parallel_loop: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub left_family: CoreOperandRef,
    pub right_family: CoreOperandRef,
    pub body_left: CoreWireRef,
    pub body_right: CoreWireRef,
    pub operation: MatrixBinaryRef,
    pub body_output: CoreWireRef,
    pub output_family: CoreWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelSixWaySelectRef {
    pub parallel_loop: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub selector_family: CoreOperandRef,
    pub branch_families: [CoreOperandRef; 6],
    pub body_selector: CoreWireRef,
    pub body_branches: [CoreWireRef; 6],
    pub select: SixWaySelectRef,
    pub body_output: CoreWireRef,
    pub output_family: CoreWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct ParallelTwoWaySelectRef {
    pub parallel_loop: CoreNodeRef,
    pub body_scope: FrozenGraphScopeId,
    pub selector_family: CoreOperandRef,
    pub branch_families: [CoreOperandRef; 2],
    pub body_selector: CoreWireRef,
    pub body_branches: [CoreWireRef; 2],
    pub select: TwoWaySelectRef,
    pub body_output: CoreWireRef,
    pub output_family: CoreWireRef,
}

/// One carried BGG component's product/XOR reuse and two-stage selection.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct FamilyBooleanGateLayout {
    pub state_input: CoreWireRef,
    pub state_output: CoreWireRef,
    pub left_selection: ParallelFamilyGetRef,
    pub right_selection: ParallelFamilyGetRef,
    pub opcode_family: CoreWireRef,
    pub active_family: CoreWireRef,
    pub zero: ParallelMatrixBinaryRef,
    pub one_repetition: ParallelLoopRef,
    pub one_family: CoreWireRef,
    pub copy_family: CoreWireRef,
    pub not: ParallelMatrixBinaryRef,
    pub product: FamilyProductRef,
    pub sum: ParallelMatrixBinaryRef,
    pub two_product: ParallelMatrixBinaryRef,
    pub xor: ParallelMatrixBinaryRef,
    pub candidate_select: ParallelSixWaySelectRef,
    pub active_mask: ParallelLoopRef,
    pub active_select: ParallelTwoWaySelectRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "operation")]
pub enum FamilyProductRef {
    Direct(ParallelMatrixBinaryRef),
    EncodingVector {
        left_times_right_decomposition: ParallelMatrixBinaryRef,
        right_times_left_plaintext: ParallelMatrixBinaryRef,
        sum: ParallelMatrixBinaryRef,
    },
}

impl FamilyProductRef {
    pub fn output_family(&self) -> &CoreWireRef {
        match self {
            Self::Direct(operation) => &operation.output_family,
            Self::EncodingVector { sum, .. } => &sum.output_family,
        }
    }
}

/// One actual dataflow edge on the path from the residual to the decoded bit.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct UnaryNodeRef {
    pub operation: CoreNodeRef,
    pub input: CoreOperandRef,
    pub output: CoreWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct BinaryNodeRef {
    pub operation: CoreNodeRef,
    pub left: CoreOperandRef,
    pub right: CoreOperandRef,
    pub output: CoreWireRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DecoderLayout {
    pub one_vector: MatrixBinaryRef,
    pub k_vector: MatrixBinaryRef,
    pub decoder_vector: MatrixBinaryRef,
    pub one_preimage: CoreWireRef,
    pub k_preimage: CoreWireRef,
    pub decoder_preimage: CoreWireRef,
    pub r_decomposed: CoreWireRef,
    pub selected_circuit_vector: CoreWireRef,
    pub one_minus_circuit: MatrixBinaryRef,
    pub projected_difference: MatrixBinaryRef,
    pub k_plus_projection: MatrixBinaryRef,
    pub residual: MatrixBinaryRef,
    pub extract_coefficient: UnaryNodeRef,
    pub threshold: EvaluateIntRef,
    pub lower_compare: BinaryNodeRef,
    pub upper_scale: BinaryNodeRef,
    pub upper_compare: BinaryNodeRef,
    pub lower_to_int: UnaryNodeRef,
    pub upper_to_int: UnaryNodeRef,
    pub comparison_sum: BinaryNodeRef,
    pub equals_two: BinaryNodeRef,
    pub decoded: CoreWireRef,
}

/// The nonempty, cardinality-fixed Diamond semantic certificate.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct DiamondCertificate {
    pub workflow: DiamondWorkflowLayout,
    pub message: Box<MessageConstructionLayout>,
    pub input_preprocessing: DiamondInputPreprocessingLayout,
    pub public_key_sampling: Box<BggPublicKeySamplingLayout>,
    pub encryption_initial_public_keys: Box<EncryptionInitialPublicKeysLayout>,
    pub artifact_preprocessing: Box<DiamondArtifactPreprocessingLayout>,
    pub input_injection: InputInjectionLayout,
    pub decryption_initial_encodings: Box<DecryptionInitialEncodingsLayout>,
    pub boolean_layers: BooleanLayersLayout,
    pub decoder: DecoderLayout,
}

/// Protocols without a specialized certificate remain representable, but only
/// `Diamond` is accepted as a Diamond proof input.
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "certificate")]
pub enum SemanticCertificate {
    #[default]
    None,
    Diamond(Box<DiamondCertificate>),
}

impl SemanticCertificate {
    pub fn diamond(&self) -> Option<&DiamondCertificate> {
        match self {
            Self::None => None,
            Self::Diamond(certificate) => Some(certificate),
        }
    }

    pub fn sha256(&self) -> Result<[u8; 32], serde_json::Error> {
        Ok(Sha256::digest(serde_json::to_vec(self)?).into())
    }

    pub fn validate_references(
        &self,
        protocol: &ProtocolDecl,
    ) -> Result<(), CertificateValidationError> {
        match self {
            Self::None => Ok(()),
            Self::Diamond(certificate) => certificate.validate_references(protocol),
        }
    }
}

impl DiamondCertificate {
    pub fn sha256(&self) -> Result<[u8; 32], serde_json::Error> {
        Ok(Sha256::digest(serde_json::to_vec(self)?).into())
    }

    pub fn validate_references(
        &self,
        protocol: &ProtocolDecl,
    ) -> Result<(), CertificateValidationError> {
        validate_workflow(protocol, &self.workflow)?;
        validate_message_construction(protocol, &self.workflow, &self.message)?;
        validate_input_preprocessing(protocol, &self.workflow, &self.input_preprocessing)?;
        validate_public_key_sampling(protocol, &self.workflow, &self.public_key_sampling)?;
        validate_encryption_initial_public_keys(
            protocol,
            &self.workflow,
            &self.public_key_sampling,
            &self.encryption_initial_public_keys,
        )?;
        validate_artifact_preprocessing(
            protocol,
            &self.workflow,
            &self.input_preprocessing,
            &self.public_key_sampling,
            &self.boolean_layers,
            &self.artifact_preprocessing,
        )?;
        validate_input_injection(protocol, &self.workflow, &self.input_injection)?;
        validate_decryption_initial_encodings(
            protocol,
            &self.workflow,
            &self.input_injection,
            &self.boolean_layers,
            &self.decryption_initial_encodings,
        )?;
        validate_boolean_layers(protocol, &self.workflow, &self.boolean_layers)?;
        validate_decoder(protocol, &self.workflow, &self.boolean_layers, &self.decoder)?;

        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CertificateValidationError {
    #[error("semantic reference names a missing protocol stage `{0}`")]
    MissingStage(String),
    #[error("semantic reference names a missing core scope")]
    MissingScope,
    #[error("semantic reference names missing core node {0:?}")]
    MissingNode(NodeId),
    #[error("semantic reference names missing output port {port:?} on node {node:?}")]
    MissingPort { node: NodeId, port: Port },
    #[error("semantic reference names missing operand {operand} on node {node:?}")]
    MissingOperand { node: NodeId, operand: u32 },
    #[error("the referenced operand does not contain the certified wire")]
    OperandMismatch,
    #[error("the referenced node kind does not implement the required operation")]
    WrongNodeKind,
    #[error("the semantic output is not produced by the certified operation")]
    OutputMismatch,
    #[error("the referenced node does not have the certified embedded parameter")]
    MissingNodeParameter,
    #[error("a fixed Diamond workflow stage is missing or duplicated")]
    WorkflowStageMismatch,
    #[error("a stage interface does not exactly enumerate its inputs and outputs")]
    StageInterfaceMismatch,
    #[error("an artifact provenance entry does not match the protocol binding")]
    ArtifactProvenanceMismatch,
    #[error("a fixed loop body does not belong to the certified loop")]
    LoopBodyMismatch,
    #[error("a fixed loop does not have the required complete argument/output layout")]
    LoopLayoutMismatch,
    #[error("Diamond input preprocessing does not match the certified executable wiring")]
    InputPreprocessingMismatch,
    #[error("a Diamond construction trace does not match the certified executable wiring")]
    ConstructionTraceMismatch,
    #[error("encrypt and decrypt Boolean loops do not consume the same protocol input")]
    CircuitInputMismatch,
    #[error("Boolean layer metadata is not the exact certified layer slice")]
    LayerMetadataMismatch,
    #[error("a certified parallel-loop boundary does not match the executable graph")]
    ParallelLoopBoundaryMismatch,
    #[error("a deterministic RHS decomposition is not local to the certified Boolean loop")]
    DecompositionScopeMismatch,
    #[error("a deterministic RHS decomposition consumer is not the required multiplication")]
    DecompositionConsumerMismatch,
    #[error("a gate RHS decomposition is exported as a stage output")]
    ExportedGateDecomposition,
    #[error("the decoder matrix operations are not connected in the required order")]
    DecoderWiringMismatch,
    #[error("the decoder path is empty or does not form exact executable dataflow")]
    DecodePathMismatch,
}

fn validate_workflow(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
) -> Result<(), CertificateValidationError> {
    if workflow.encryption.stage == workflow.decryption.stage ||
        protocol.stages.len() != 2 ||
        !protocol.stages.iter().any(|stage| stage.id == workflow.encryption.stage) ||
        !protocol.stages.iter().any(|stage| stage.id == workflow.decryption.stage)
    {
        return Err(CertificateValidationError::WorkflowStageMismatch);
    }
    validate_stage_interface(protocol, &workflow.encryption)?;
    validate_stage_interface(protocol, &workflow.decryption)?;

    let expected_binding_count: usize =
        protocol.stages.iter().map(|stage| stage.bindings.len()).sum();
    if workflow.artifacts.len() != expected_binding_count {
        return Err(CertificateValidationError::ArtifactProvenanceMismatch);
    }
    let mut seen = BTreeSet::new();
    for provenance in &workflow.artifacts {
        validate_artifact_provenance(protocol, provenance)?;
        if !seen.insert((provenance.consumer_stage.clone(), provenance.consumer_input.name.clone()))
        {
            return Err(CertificateValidationError::ArtifactProvenanceMismatch);
        }
    }
    Ok(())
}

fn validate_artifact_provenance(
    protocol: &ProtocolDecl,
    provenance: &ArtifactProvenance,
) -> Result<(), CertificateValidationError> {
    if provenance.producer_output.wire.node.stage != provenance.producer_stage ||
        provenance.consumer_input.node.stage != provenance.consumer_stage
    {
        return Err(CertificateValidationError::ArtifactProvenanceMismatch);
    }
    validate_stage_output(protocol, &provenance.producer_stage, &provenance.producer_output)?;
    validate_stage_input(protocol, &provenance.consumer_stage, &provenance.consumer_input)?;
    let consumer_stage = stage(protocol, &provenance.consumer_stage)?;
    let binding = consumer_stage.bindings.iter().find(|binding| {
        binding.consumer_input.0 == provenance.consumer_input.name &&
            binding.producer_stage == provenance.producer_stage &&
            binding.producer_output.0 == provenance.producer_output.name
    });
    let input = resolve_node(protocol, &provenance.consumer_input.node)?;
    if binding.is_none() ||
        !matches!(input.kind(), NodeKind::Input { name, artifact: Some(artifact), .. }
            if name == &provenance.consumer_input.name
                && artifact.artifact_name == provenance.producer_output.name)
    {
        return Err(CertificateValidationError::ArtifactProvenanceMismatch);
    }
    Ok(())
}

fn validate_stage_interface(
    protocol: &ProtocolDecl,
    layout: &StageInterfaceLayout,
) -> Result<(), CertificateValidationError> {
    let stage = stage(protocol, &layout.stage)?;
    let root = stage.graph.root_scope();
    let actual_inputs = root
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, .. } => Some((name.as_str(), root.node_id(node))),
            _ => None,
        })
        .collect::<Vec<_>>();
    if actual_inputs.len() != layout.inputs.len() {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    for (declared, (name, node)) in layout.inputs.iter().zip(actual_inputs) {
        if declared.name != name ||
            declared.node.stage != layout.stage ||
            declared.node.scope != FrozenGraphScopeId::Root ||
            Some(declared.node.node) != node
        {
            return Err(CertificateValidationError::StageInterfaceMismatch);
        }
    }

    if stage.graph.outputs().len() != layout.outputs.len() {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    for (declared, (name, output)) in layout.outputs.iter().zip(stage.graph.outputs()) {
        if declared.name != *name ||
            declared.wire.node.stage != layout.stage ||
            declared.wire.node.scope != FrozenGraphScopeId::Root ||
            declared.wire.as_wire_ref() != output.value
        {
            return Err(CertificateValidationError::StageInterfaceMismatch);
        }
    }
    Ok(())
}

fn validate_stage_input(
    protocol: &ProtocolDecl,
    stage_id: &StageId,
    input: &StageInputLayout,
) -> Result<(), CertificateValidationError> {
    if &input.node.stage != stage_id || input.node.scope != FrozenGraphScopeId::Root {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    let node = resolve_node(protocol, &input.node)?;
    if !matches!(node.kind(), NodeKind::Input { name, .. } if name == &input.name) {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    Ok(())
}

fn validate_stage_output(
    protocol: &ProtocolDecl,
    stage_id: &StageId,
    output: &StageOutputLayout,
) -> Result<(), CertificateValidationError> {
    if &output.wire.node.stage != stage_id || output.wire.node.scope != FrozenGraphScopeId::Root {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    validate_wire(protocol, &output.wire)?;
    let stage = stage(protocol, stage_id)?;
    if stage.graph.outputs().get(&output.name).map(|root| root.value) !=
        Some(output.wire.as_wire_ref())
    {
        return Err(CertificateValidationError::StageInterfaceMismatch);
    }
    Ok(())
}

fn validate_operation_ref(
    protocol: &ProtocolDecl,
    reference: &OperationRef,
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, &reference.operation)?;
    let stage = stage(protocol, &reference.operation.stage)?;
    let scope = stage
        .graph
        .scope(&reference.operation.scope)
        .ok_or(CertificateValidationError::MissingScope)?;
    let arguments = scope.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    if arguments.len() != reference.inputs.len() ||
        node.output_types().len() != reference.outputs.len()
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    for (index, (actual, declared)) in arguments.iter().zip(&reference.inputs).enumerate() {
        validate_operand(protocol, declared)?;
        if declared.node != reference.operation ||
            declared.operand as usize != index ||
            declared.wire.as_wire_ref() != *actual
        {
            return Err(CertificateValidationError::InputPreprocessingMismatch);
        }
    }
    for (index, output) in reference.outputs.iter().enumerate() {
        validate_operation_output(protocol, &reference.operation, output)?;
        if output.port.0 as usize != index {
            return Err(CertificateValidationError::InputPreprocessingMismatch);
        }
    }
    Ok(())
}

fn validate_parallel_operation(
    protocol: &ProtocolDecl,
    reference: &ParallelOperationRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    validate_operation_ref(protocol, &reference.body)?;
    if reference.body.operation.scope != reference.parallel_loop.body_scope ||
        reference.parallel_loop.body_outputs != reference.body.outputs ||
        reference.parallel_loop.outputs.len() != reference.body.outputs.len()
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    Ok(())
}

fn validate_sequential_loop_ref(
    protocol: &ProtocolDecl,
    layout: &SequentialLoopRef,
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, &layout.operation)?;
    let NodeKind::SequentialLoop(specification) = node.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    let stage = stage(protocol, &layout.operation.stage)?;
    let parent = stage
        .graph
        .scope(&layout.operation.scope)
        .ok_or(CertificateValidationError::MissingScope)?;
    let body =
        stage.graph.scope(&layout.body_scope).ok_or(CertificateValidationError::MissingScope)?;
    let arguments = parent.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    if stage.graph.child_scope_id(&layout.operation.scope, layout.operation.node).as_ref() !=
        Some(&layout.body_scope) ||
        layout.count != specification.count ||
        layout.index_slot != specification.index_slot ||
        layout.bindings != specification.bindings ||
        layout.carried_count != specification.carried_count ||
        arguments.len() != layout.arguments.len() ||
        body.inputs().len() != layout.body_inputs.len() ||
        body.outputs().len() != layout.body_outputs.len() ||
        node.output_types().len() != layout.outputs.len() ||
        layout.outputs.len() != layout.carried_count ||
        layout.body_outputs.len() != layout.carried_count
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    for (index, ((actual, declared), body_input)) in
        arguments.iter().zip(&layout.arguments).zip(&layout.body_inputs).enumerate()
    {
        validate_operand(protocol, declared)?;
        validate_wire(protocol, body_input)?;
        if declared.node != layout.operation ||
            declared.operand as usize != index ||
            declared.wire.as_wire_ref() != *actual ||
            body_input.as_wire_ref() != body.inputs()[index]
        {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    for (index, (body_output, output)) in
        layout.body_outputs.iter().zip(&layout.outputs).enumerate()
    {
        validate_wire(protocol, body_output)?;
        validate_operation_output(protocol, &layout.operation, output)?;
        if body_output.as_wire_ref() != body.outputs()[index] || output.port.0 as usize != index {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    Ok(())
}

fn validate_select_operation(
    protocol: &ProtocolDecl,
    reference: &OperationRef,
) -> Result<(), CertificateValidationError> {
    validate_operation_ref(protocol, reference)?;
    if reference.inputs.is_empty() ||
        reference.outputs.len() != 1 ||
        !matches!(resolve_node(protocol, &reference.operation)?.kind(),
            NodeKind::Select { count }
                if count == &IntExpr::constant(reference.inputs.len() - 1))
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_parallel_body_operation(
    protocol: &ProtocolDecl,
    reference: &ParallelOperationRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_operation(protocol, reference)?;
    if reference.body.operation.scope != reference.parallel_loop.body_scope {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_parallel_index_formula(
    protocol: &ProtocolDecl,
    reference: &ParallelIndexFormulaRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    validate_wire(protocol, &reference.body_output)?;
    if reference.body_output.node.scope != reference.parallel_loop.body_scope ||
        reference.parallel_loop.body_outputs != [reference.body_output.clone()] ||
        reference.parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn scoped_wire(context: &CoreNodeRef, wire: WireRef) -> CoreWireRef {
    CoreWireRef {
        node: CoreNodeRef::new(context.stage.clone(), context.scope.clone(), wire.node),
        port: wire.port,
    }
}

/// Validate the exact lower-bound expression used by the online input-injection source lookup:
/// `loop_index(0) * diamond_batch_bits + 1`.
fn validate_online_source_lower_bound(
    protocol: &ProtocolDecl,
    wire: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    validate_wire(protocol, wire)?;
    let add = resolve_node(protocol, &wire.node)?;
    let add_scope = stage(protocol, &wire.node.stage)?
        .graph
        .scope(&wire.node.scope)
        .ok_or(CertificateValidationError::MissingScope)?;
    let add_arguments =
        add_scope.arguments(add).ok_or(CertificateValidationError::OperandMismatch)?;
    let [product_wire, one_wire] = add_arguments.as_slice() else {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    };
    if wire.port != Port(0) ||
        add.output_types().len() != 1 ||
        !matches!(add.kind(), NodeKind::IntBinary(IntBinaryOp::Add))
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }

    let product_wire = scoped_wire(&wire.node, *product_wire);
    let one_wire = scoped_wire(&wire.node, *one_wire);
    if product_wire.port != Port(0) ||
        one_wire.port != Port(0) ||
        !is_constant_int(protocol, &one_wire, 1)?
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }

    let product = resolve_node(protocol, &product_wire.node)?;
    let product_arguments =
        add_scope.arguments(product).ok_or(CertificateValidationError::OperandMismatch)?;
    let [level_wire, width_wire] = product_arguments.as_slice() else {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    };
    if product.output_types().len() != 1 ||
        !matches!(product.kind(), NodeKind::IntBinary(IntBinaryOp::Multiply))
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }

    let level_wire = scoped_wire(&wire.node, *level_wire);
    let width_wire = scoped_wire(&wire.node, *width_wire);
    let level = resolve_node(protocol, &level_wire.node)?;
    let width = resolve_node(protocol, &width_wire.node)?;
    let level_arguments =
        add_scope.arguments(level).ok_or(CertificateValidationError::OperandMismatch)?;
    let width_arguments =
        add_scope.arguments(width).ok_or(CertificateValidationError::OperandMismatch)?;
    if level_wire.port != Port(0) ||
        width_wire.port != Port(0) ||
        !level_arguments.is_empty() ||
        !width_arguments.is_empty() ||
        level.output_types().len() != 1 ||
        width.output_types().len() != 1 ||
        !matches!(level.kind(), NodeKind::EvaluateInt(IntExpr::LoopIndex(0))) ||
        !matches!(width.kind(), NodeKind::EvaluateInt(IntExpr::Var(name)) if name == "diamond_batch_bits")
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_initial_state_expansion(
    protocol: &ProtocolDecl,
    reference: &InitialStateExpansionRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    validate_wire(protocol, &reference.body_output)?;
    if reference.body_output.node.scope != reference.parallel_loop.body_scope ||
        reference.parallel_loop.body_outputs != [reference.body_output.clone()] ||
        reference.parallel_loop.arguments.len() != 1 ||
        reference.parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_witness_digit_packing(
    protocol: &ProtocolDecl,
    reference: &WitnessDigitPackingRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    validate_wire(protocol, &reference.body_output)?;
    validate_sequential_loop_ref(protocol, &reference.bit_scan)?;
    if reference.body_output.node.scope != reference.parallel_loop.body_scope ||
        reference.parallel_loop.body_outputs != [reference.body_output.clone()] ||
        reference.parallel_loop.outputs.len() != 1 ||
        !scope_is_within(
            &reference.bit_scan.operation.scope,
            &reference.parallel_loop.body_scope,
        ) ||
        reference.bit_scan.outputs.first() != Some(&reference.body_output)
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_parallel_gather(
    protocol: &ProtocolDecl,
    reference: &ParallelGatherRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    let source_count = reference.source_families.len();
    if source_count == 0 ||
        reference.body_sources.len() != source_count ||
        reference.gets.len() != source_count ||
        reference.output_families.len() != source_count ||
        reference.parallel_loop.arguments.len() != source_count + 1 ||
        reference.parallel_loop.body_inputs.len() != source_count + 1 ||
        reference.parallel_loop.body_outputs.len() != source_count ||
        reference.parallel_loop.outputs.len() != source_count ||
        reference.parallel_loop.arguments[0] != reference.index_family ||
        reference.parallel_loop.body_inputs[0] != reference.body_index ||
        reference.parallel_loop.arguments[1..] != reference.source_families ||
        reference.parallel_loop.body_inputs[1..] != reference.body_sources ||
        reference.parallel_loop.body_outputs !=
            reference.gets.iter().map(|get| get.output.clone()).collect::<Vec<_>>() ||
        reference.parallel_loop.outputs != reference.output_families
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    for (get, source) in reference.gets.iter().zip(&reference.body_sources) {
        validate_dynamic_get(protocol, get, source)?;
        if get.index.wire != reference.body_index {
            return Err(CertificateValidationError::InputPreprocessingMismatch);
        }
    }
    Ok(())
}

fn validate_preimage_ref(
    protocol: &ProtocolDecl,
    reference: &PreimageRef,
) -> Result<(), CertificateValidationError> {
    validate_operation_ref(protocol, &reference.sample)?;
    validate_operation_ref(protocol, &reference.materialize)?;
    if !matches!(
        resolve_node(protocol, &reference.sample.operation)?.kind(),
        NodeKind::PreimageSample { .. }
    ) || !matches!(resolve_node(protocol, &reference.materialize.operation)?.kind(), NodeKind::MatrixScale { scalar } if scalar == &IntExpr::constant(1)) ||
        reference.sample.outputs.len() != 1 ||
        reference.materialize.inputs.len() != 1 ||
        reference.materialize.outputs.len() != 1 ||
        reference.materialize.inputs[0].wire != reference.sample.outputs[0]
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    Ok(())
}

fn validate_parallel_preimage(
    protocol: &ProtocolDecl,
    reference: &ParallelPreimageRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    validate_preimage_ref(protocol, &reference.body)?;
    if reference.body.sample.operation.scope != reference.parallel_loop.body_scope ||
        reference.body.materialize.operation.scope != reference.parallel_loop.body_scope ||
        reference.parallel_loop.body_inputs !=
            reference
                .body
                .sample
                .inputs
                .iter()
                .map(|input| input.wire.clone())
                .collect::<Vec<_>>() ||
        reference.parallel_loop.body_outputs != reference.body.materialize.outputs ||
        reference.parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    Ok(())
}

fn validate_transition_selector(
    protocol: &ProtocolDecl,
    reference: &TransitionSelectorLayout,
) -> Result<(), CertificateValidationError> {
    for operation in [&reference.regular, &reference.k_identity, &reference.k] {
        validate_operation_ref(protocol, operation)?;
    }
    validate_select_operation(protocol, &reference.initial_select)?;
    validate_sequential_loop_ref(protocol, &reference.bit_scan)?;
    let body = &reference.bit_body;
    for operation in [
        &body.bit_extract,
        &body.bit_to_int,
        &body.bit_zero,
        &body.bit_one,
        &body.special_product,
        &body.special_top,
        &body.special_bottom,
        &body.special,
        &body.state_match,
        &body.state_match_to_int,
    ] {
        validate_operation_ref(protocol, operation)?;
        if operation.operation.scope != reference.bit_scan.body_scope {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    validate_select_operation(protocol, &body.bit_select)?;
    validate_select_operation(protocol, &body.selector)?;
    if body.bit_select.operation.scope != reference.bit_scan.body_scope ||
        body.selector.operation.scope != reference.bit_scan.body_scope ||
        !matches!(
            resolve_node(protocol, &reference.regular.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Diagonal }
        ) ||
        !matches!(
            resolve_node(protocol, &reference.k_identity.operation)?.kind(),
            NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Identity, .. }
        ) ||
        !matches!(
            resolve_node(protocol, &reference.k.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Diagonal }
        ) ||
        !matches!(
            resolve_node(protocol, &body.bit_extract.operation)?.kind(),
            NodeKind::BitExtract { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &body.bit_to_int.operation)?.kind(),
            NodeKind::BoolToInt
        ) ||
        !matches!(
            resolve_node(protocol, &body.special_product.operation)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) ||
        !matches!(
            resolve_node(protocol, &body.special_top.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Columns }
        ) ||
        !matches!(
            resolve_node(protocol, &body.special.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Rows }
        ) ||
        !matches!(
            resolve_node(protocol, &body.state_match.operation)?.kind(),
            NodeKind::IntCompare(IntCompareOp::Equal)
        ) ||
        !matches!(
            resolve_node(protocol, &body.state_match_to_int.operation)?.kind(),
            NodeKind::BoolToInt
        ) ||
        reference.initial_select.outputs != [reference.bit_scan.arguments[0].wire.clone()] ||
        reference.bit_scan.body_outputs != body.selector.outputs ||
        reference.bit_scan.outputs.len() != 1 ||
        body.bit_select.inputs.first().map(|input| &input.wire) !=
            body.bit_to_int.outputs.first() ||
        body.special_product.inputs.get(1).map(|input| &input.wire) !=
            body.bit_select.outputs.first() ||
        body.special_top.inputs.get(1).map(|input| &input.wire) !=
            body.special_product.outputs.first() ||
        body.special.inputs.first().map(|input| &input.wire) != body.special_top.outputs.first() ||
        body.selector.inputs.first().map(|input| &input.wire) !=
            body.state_match_to_int.outputs.first() ||
        body.selector.inputs.get(2).map(|input| &input.wire) != body.special.outputs.first()
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_transition_target(
    protocol: &ProtocolDecl,
    reference: &ParallelTransitionTargetRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &reference.parallel_loop)?;
    let body = &reference.body;
    validate_wire(protocol, &body.digit_secret)?;
    validate_wire(protocol, &body.target_public)?;
    validate_wire(protocol, &body.selector)?;
    validate_transition_selector(protocol, &body.selector_construction)?;
    validate_operation_ref(protocol, &body.error_sample)?;
    validate_operation_ref(protocol, &body.selector_product)?;
    validate_operation_ref(protocol, &body.target_sum)?;
    if !matches!(
        resolve_node(protocol, &body.error_sample.operation)?.kind(),
        NodeKind::GaussianSample { .. }
    ) || !matches!(
        resolve_node(protocol, &body.selector_product.operation)?.kind(),
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
    ) || !matches!(
        resolve_node(protocol, &body.target_sum.operation)?.kind(),
        NodeKind::MatrixBinary(MatrixBinaryOp::Add)
    ) || reference.parallel_loop.arguments.len() < 2 ||
        reference.parallel_loop.body_inputs.len() < 2 ||
        reference.parallel_loop.body_inputs[..2] !=
            [body.digit_secret.clone(), body.target_public.clone()] ||
        body.selector_construction.bit_scan.outputs != [body.selector.clone()] ||
        body.error_sample.inputs.len() != 0 ||
        body.error_sample.outputs.len() != 1 ||
        body.selector_product.inputs.iter().map(|input| &input.wire).collect::<Vec<_>>() !=
            vec![&body.selector, &body.target_public] ||
        body.selector_product.outputs.len() != 1 ||
        body.target_sum.inputs.iter().map(|input| &input.wire).collect::<Vec<_>>() !=
            vec![&body.selector_product.outputs[0], &body.error_sample.outputs[0]] ||
        body.target_sum.outputs.len() != 1 ||
        reference.parallel_loop.body_outputs != body.target_sum.outputs ||
        reference.parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    Ok(())
}

fn validate_static_get(
    protocol: &ProtocolDecl,
    value: &CoreWireRef,
    family: &CoreWireRef,
    index: usize,
) -> Result<(), CertificateValidationError> {
    validate_wire(protocol, value)?;
    let node = resolve_node(protocol, &value.node)?;
    let stage = stage(protocol, &value.node.stage)?;
    let scope =
        stage.graph.scope(&value.node.scope).ok_or(CertificateValidationError::MissingScope)?;
    let arguments = scope.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    if value.port != Port(0) ||
        !matches!(node.kind(), NodeKind::FamilyGetStatic { index: actual }
            if actual == &IntExpr::constant(index)) ||
        arguments.as_slice() != [family.as_wire_ref()]
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    Ok(())
}

fn require_construction_stage(
    operation: &CoreNodeRef,
    expected: &StageId,
) -> Result<(), CertificateValidationError> {
    if &operation.stage != expected {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_workflow_artifact(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    artifact: &ArtifactProvenance,
) -> Result<(), CertificateValidationError> {
    validate_artifact_provenance(protocol, artifact)?;
    if !workflow.artifacts.contains(artifact) {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_message_construction(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    layout: &MessageConstructionLayout,
) -> Result<(), CertificateValidationError> {
    for operation in [&layout.to_int, &layout.zero, &layout.one] {
        validate_operation_ref(protocol, operation)?;
        require_construction_stage(&operation.operation, &workflow.encryption.stage)?;
    }
    validate_select_operation(protocol, &layout.select)?;
    require_construction_stage(&layout.select.operation, &workflow.encryption.stage)?;
    if !matches!(resolve_node(protocol, &layout.to_int.operation)?.kind(), NodeKind::BoolToInt) ||
        !matches!(
            resolve_node(protocol, &layout.zero.operation)?.kind(),
            NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Zero, .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.one.operation)?.kind(),
            NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Identity, .. }
        ) ||
        layout.select.inputs.first().map(|input| &input.wire) != layout.to_int.outputs.first() ||
        layout.select.inputs.get(1).map(|input| &input.wire) != layout.zero.outputs.first() ||
        layout.select.inputs.get(2).map(|input| &input.wire) != layout.one.outputs.first()
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_public_key_sampling(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    layout: &BggPublicKeySamplingLayout,
) -> Result<(), CertificateValidationError> {
    validate_workflow_artifact(protocol, workflow, &layout.public_keys_artifact)?;
    validate_operation_ref(protocol, &layout.packed_hash)?;
    validate_parallel_body_operation(protocol, &layout.slices)?;
    let encryption = &workflow.encryption.stage;
    require_construction_stage(&layout.packed_hash.operation, encryption)?;
    require_construction_stage(&layout.slices.parallel_loop.operation, encryption)?;
    if !matches!(
        resolve_node(protocol, &layout.packed_hash.operation)?.kind(),
        NodeKind::HashSample { .. }
    ) || !matches!(
        resolve_node(protocol, &layout.slices.body.operation)?.kind(),
        NodeKind::Slice { .. }
    ) || layout.slices.parallel_loop.arguments.len() != 1 ||
        layout.slices.parallel_loop.arguments[0].wire != layout.packed_hash.outputs[0] ||
        layout.slices.body.inputs.len() != 1 ||
        layout.slices.body.inputs[0].wire != layout.slices.parallel_loop.body_inputs[0] ||
        layout.public_keys_artifact.producer_output.wire !=
            layout.slices.parallel_loop.outputs[0]
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_parallel_select_chain(
    protocol: &ProtocolDecl,
    parallel_loop: &ParallelLoopRef,
    operations: &[&OperationRef],
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, parallel_loop)?;
    for operation in operations {
        validate_select_operation(protocol, operation)?;
        if operation.operation.scope != parallel_loop.body_scope {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    if operations.is_empty() ||
        parallel_loop.body_outputs != operations.last().expect("nonempty").outputs ||
        parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_encryption_initial_public_keys(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    sampling: &BggPublicKeySamplingLayout,
    layout: &EncryptionInitialPublicKeysLayout,
) -> Result<(), CertificateValidationError> {
    validate_operation_ref(protocol, &layout.one_public_key)?;
    validate_operation_ref(protocol, &layout.zero_public_key)?;
    validate_evaluate_int(protocol, &layout.instance_width)?;
    validate_parallel_loop_ref(protocol, &layout.public_indices)?;
    validate_parallel_gather(protocol, &layout.public_candidates)?;
    validate_parallel_select_chain(
        protocol,
        &layout.packed_inputs.parallel_loop,
        &[&layout.packed_inputs.in_range, &layout.packed_inputs.padded],
    )?;
    validate_parallel_select_chain(
        protocol,
        &layout.circuit_inputs.parallel_loop,
        &[&layout.circuit_inputs.selected_instance, &layout.circuit_inputs.selected_source],
    )?;
    let encryption = &workflow.encryption.stage;
    for operation in [
        &layout.one_public_key.operation,
        &layout.zero_public_key.operation,
        &layout.instance_width.operation,
        &layout.public_indices.operation,
        &layout.public_candidates.parallel_loop.operation,
        &layout.packed_inputs.parallel_loop.operation,
        &layout.circuit_inputs.parallel_loop.operation,
    ] {
        require_construction_stage(operation, encryption)?;
    }
    if !matches!(
        resolve_node(protocol, &layout.one_public_key.operation)?.kind(),
        NodeKind::FamilyGetStatic { index } if index == &IntExpr::constant(0)
    ) || layout.one_public_key.inputs.len() != 1 ||
        layout.one_public_key.outputs.len() != 1 ||
        layout.one_public_key.inputs[0].wire != sampling.slices.parallel_loop.outputs[0] ||
        !matches!(
            resolve_node(protocol, &layout.zero_public_key.operation)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)
        ) ||
        layout.zero_public_key.inputs.len() != 2 ||
        layout.zero_public_key.inputs[0].wire != layout.zero_public_key.inputs[1].wire ||
        layout.zero_public_key.inputs[0].wire != layout.one_public_key.outputs[0] ||
        layout.public_candidates.index_family.wire != layout.public_indices.outputs[0] ||
        layout.public_candidates.source_families.len() != 1 ||
        layout.public_candidates.source_families[0].wire !=
            sampling.slices.parallel_loop.outputs[0] ||
        layout.packed_inputs.parallel_loop.arguments.first().map(|input| &input.wire) !=
            layout.public_candidates.output_families.first() ||
        layout.packed_inputs.padded.inputs.get(2).map(|input| &input.wire) !=
            layout.packed_inputs.in_range.outputs.first() ||
        layout.circuit_inputs.selected_source.inputs.get(1).map(|input| &input.wire) !=
            layout.circuit_inputs.parallel_loop.body_inputs.get(1) ||
        layout.circuit_inputs.selected_source.inputs.get(2).map(|input| &input.wire) !=
            layout.circuit_inputs.selected_instance.outputs.first()
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_static_trapdoor(
    protocol: &ProtocolDecl,
    layout: &StaticTrapdoorLayout,
) -> Result<(), CertificateValidationError> {
    for operation in [&layout.public, &layout.secret] {
        validate_operation_ref(protocol, operation)?;
        if operation.inputs.len() != 1 ||
            operation.outputs.len() != 1 ||
            !matches!(resolve_node(protocol, &operation.operation)?.kind(),
                NodeKind::FamilyGetStatic { index } if index == &IntExpr::constant(0))
        {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    Ok(())
}

fn validate_parallel_witness_target(
    protocol: &ProtocolDecl,
    layout: &ParallelWitnessTargetLayout,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &layout.parallel_loop)?;
    validate_operation_ref(protocol, &layout.negated_gadget)?;
    validate_operation_ref(protocol, &layout.target)?;
    if layout.negated_gadget.operation.scope != layout.parallel_loop.body_scope ||
        layout.target.operation.scope != layout.parallel_loop.body_scope ||
        !matches!(
            resolve_node(protocol, &layout.negated_gadget.operation)?.kind(),
            NodeKind::MatrixNegate
        ) ||
        !matches!(
            resolve_node(protocol, &layout.target.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Rows }
        ) ||
        layout.target.inputs.get(1).map(|input| &input.wire) !=
            layout.negated_gadget.outputs.first() ||
        layout.parallel_loop.body_outputs != layout.target.outputs ||
        layout.parallel_loop.outputs.len() != 1
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_artifact_preprocessing(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    input_preprocessing: &DiamondInputPreprocessingLayout,
    sampling: &BggPublicKeySamplingLayout,
    boolean_layers: &BooleanLayersLayout,
    layout: &DiamondArtifactPreprocessingLayout,
) -> Result<(), CertificateValidationError> {
    for artifact in [
        &layout.one_preimage_artifact,
        &layout.witness_preimages_artifact,
        &layout.k_preimage_artifact,
        &layout.r_decomposed_artifact,
        &layout.decoder_preimage_artifact,
    ] {
        validate_workflow_artifact(protocol, workflow, artifact)?;
        if artifact.producer_stage != workflow.encryption.stage {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    validate_static_trapdoor(protocol, &layout.projection_trapdoor)?;
    for operation in [
        &layout.one_target.gadget,
        &layout.one_target.difference,
        &layout.one_target.zero_row,
        &layout.one_target.target,
    ] {
        validate_operation_ref(protocol, operation)?;
    }
    validate_preimage_ref(protocol, &layout.one_preimage)?;
    validate_parallel_loop_ref(protocol, &layout.witness_indices)?;
    validate_parallel_gather(protocol, &layout.witness_trapdoors)?;
    validate_parallel_gather(protocol, &layout.witness_public_keys)?;
    validate_parallel_witness_target(protocol, &layout.witness_targets)?;
    validate_parallel_preimage(protocol, &layout.witness_preimages)?;
    for operation in [
        &layout.k_target.public_key_hash,
        &layout.k_target.first_column,
        &layout.k_target.half_modulus,
        &layout.k_target.target,
        &layout.r_hash,
        &layout.r_slice,
        &layout.r_decomposition,
        &layout.r_materialization,
        &layout.r_reshape,
        &layout.decoder_target.public_key_difference,
        &layout.decoder_target.projected_difference,
        &layout.decoder_target.public_key_sum,
        &layout.decoder_target.zero,
        &layout.decoder_target.target,
    ] {
        validate_operation_ref(protocol, operation)?;
    }
    validate_preimage_ref(protocol, &layout.k_preimage)?;
    validate_preimage_ref(protocol, &layout.decoder_preimage)?;

    let encryption = &workflow.encryption.stage;
    let operations = [
        &layout.projection_trapdoor.public.operation,
        &layout.projection_trapdoor.secret.operation,
        &layout.one_target.gadget.operation,
        &layout.one_target.difference.operation,
        &layout.one_target.zero_row.operation,
        &layout.one_target.target.operation,
        &layout.one_preimage.sample.operation,
        &layout.witness_indices.operation,
        &layout.witness_trapdoors.parallel_loop.operation,
        &layout.witness_public_keys.parallel_loop.operation,
        &layout.witness_targets.parallel_loop.operation,
        &layout.witness_preimages.parallel_loop.operation,
        &layout.k_target.public_key_hash.operation,
        &layout.k_target.first_column.operation,
        &layout.k_target.half_modulus.operation,
        &layout.k_target.target.operation,
        &layout.k_preimage.sample.operation,
        &layout.r_hash.operation,
        &layout.r_slice.operation,
        &layout.r_decomposition.operation,
        &layout.r_materialization.operation,
        &layout.r_reshape.operation,
        &layout.decoder_target.public_key_difference.operation,
        &layout.decoder_target.projected_difference.operation,
        &layout.decoder_target.public_key_sum.operation,
        &layout.decoder_target.zero.operation,
        &layout.decoder_target.target.operation,
        &layout.decoder_preimage.sample.operation,
    ];
    for operation in operations {
        require_construction_stage(operation, encryption)?;
    }
    let projection_public = layout.projection_trapdoor.public.outputs.first();
    let projection_secret = layout.projection_trapdoor.secret.outputs.first();
    for sample in
        [&layout.one_preimage.sample, &layout.k_preimage.sample, &layout.decoder_preimage.sample]
    {
        if sample.inputs.first().map(|input| &input.wire) != projection_public ||
            sample.inputs.get(1).map(|input| &input.wire) != projection_secret
        {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    if !matches!(
        resolve_node(protocol, &layout.one_target.gadget.operation)?.kind(),
        NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Gadget { .. }, .. }
    ) || !matches!(
        resolve_node(protocol, &layout.one_target.difference.operation)?.kind(),
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)
    ) || !matches!(
        resolve_node(protocol, &layout.one_target.target.operation)?.kind(),
        NodeKind::Concat { axis: ConcatAxis::Rows }
    ) || layout.one_target.target.inputs.first().map(|input| &input.wire) !=
        layout.one_target.difference.outputs.first() ||
        layout.one_target.target.inputs.get(1).map(|input| &input.wire) !=
            layout.one_target.zero_row.outputs.first() ||
        layout.one_target.difference.inputs.get(1).map(|input| &input.wire) !=
            layout.one_target.gadget.outputs.first() ||
        layout.one_preimage.sample.inputs.get(2).map(|input| &input.wire) !=
            layout.one_target.target.outputs.first() ||
        layout.one_preimage_artifact.producer_output.wire !=
            layout.one_preimage.materialize.outputs[0] ||
        layout.projection_trapdoor.public.inputs[0].wire !=
            input_preprocessing.final_trapdoors.output_families[0] ||
        layout.projection_trapdoor.secret.inputs[0].wire !=
            input_preprocessing.final_trapdoors.output_families[1] ||
        layout.witness_trapdoors.index_family.wire != layout.witness_indices.outputs[0] ||
        layout.witness_public_keys.index_family.wire != layout.witness_indices.outputs[0] ||
        layout.witness_public_keys.source_families.first().map(|source| &source.wire) !=
            sampling.slices.parallel_loop.outputs.first() ||
        layout.witness_targets.parallel_loop.arguments.first().map(|argument| &argument.wire) !=
            layout.witness_public_keys.output_families.first() ||
        layout.witness_preimages.parallel_loop.arguments !=
            layout
                .witness_trapdoors
                .output_families
                .iter()
                .chain(layout.witness_targets.parallel_loop.outputs.iter())
                .enumerate()
                .map(|(index, wire)| {
                    layout
                        .witness_preimages
                        .parallel_loop
                        .operation
                        .operand(index as u32, wire.clone())
                })
                .collect::<Vec<_>>() ||
        layout.witness_preimages_artifact.producer_output.wire !=
            layout.witness_preimages.parallel_loop.outputs[0] ||
        !matches!(
            resolve_node(protocol, &layout.k_target.public_key_hash.operation)?.kind(),
            NodeKind::HashSample { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.k_target.first_column.operation)?.kind(),
            NodeKind::Slice { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.k_target.target.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Rows }
        ) ||
        layout.k_target.public_key_hash.inputs.first().map(|input| &input.wire) !=
            sampling.packed_hash.inputs.first().map(|input| &input.wire) ||
        layout.k_target.first_column.inputs.first().map(|input| &input.wire) !=
            layout.k_target.public_key_hash.outputs.first() ||
        layout.k_target.target.inputs.first().map(|input| &input.wire) !=
            layout.k_target.first_column.outputs.first() ||
        layout.k_target.target.inputs.get(1).map(|input| &input.wire) !=
            layout.k_target.half_modulus.outputs.first() ||
        layout.k_preimage.sample.inputs.get(2).map(|input| &input.wire) !=
            layout.k_target.target.outputs.first() ||
        layout.k_preimage_artifact.producer_output.wire !=
            layout.k_preimage.materialize.outputs[0] ||
        !matches!(
            resolve_node(protocol, &layout.r_hash.operation)?.kind(),
            NodeKind::HashSample { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.r_slice.operation)?.kind(),
            NodeKind::Slice { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.r_decomposition.operation)?.kind(),
            NodeKind::GadgetDecompose { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.r_reshape.operation)?.kind(),
            NodeKind::Reshape { .. }
        ) ||
        layout.r_hash.inputs.first().map(|input| &input.wire) !=
            sampling.packed_hash.inputs.first().map(|input| &input.wire) ||
        layout.r_slice.inputs.first().map(|input| &input.wire) != layout.r_hash.outputs.first() ||
        layout.r_decomposition.inputs.first().map(|input| &input.wire) !=
            layout.r_slice.outputs.first() ||
        layout.r_materialization.inputs.first().map(|input| &input.wire) !=
            layout.r_decomposition.outputs.first() ||
        layout.r_reshape.inputs.first().map(|input| &input.wire) !=
            layout.r_materialization.outputs.first() ||
        layout.r_decomposed_artifact.producer_output.wire != layout.r_reshape.outputs[0] ||
        layout.decoder_target.public_key_difference.inputs.get(1).map(|input| &input.wire) !=
            Some(&boolean_layers.encryption.selected_output.output) ||
        layout.decoder_target.public_key_difference.inputs.first().map(|input| &input.wire) !=
            layout.one_target.difference.inputs.first().map(|input| &input.wire) ||
        layout.decoder_target.projected_difference.inputs.first().map(|input| &input.wire) !=
            layout.decoder_target.public_key_difference.outputs.first() ||
        layout.decoder_target.projected_difference.inputs.get(1).map(|input| &input.wire) !=
            layout.r_reshape.outputs.first() ||
        layout.decoder_target.public_key_sum.inputs.first().map(|input| &input.wire) !=
            layout.k_target.first_column.outputs.first() ||
        layout.decoder_target.public_key_sum.inputs.get(1).map(|input| &input.wire) !=
            layout.decoder_target.projected_difference.outputs.first() ||
        layout.decoder_target.target.inputs.first().map(|input| &input.wire) !=
            layout.decoder_target.public_key_sum.outputs.first() ||
        layout.decoder_preimage.sample.inputs.get(2).map(|input| &input.wire) !=
            layout.decoder_target.target.outputs.first() ||
        layout.decoder_preimage_artifact.producer_output.wire !=
            layout.decoder_preimage.materialize.outputs[0]
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_encoding_component_operations(
    protocol: &ProtocolDecl,
    layout: &EncodingComponentOperationsLayout,
) -> Result<(), CertificateValidationError> {
    for operation in [&layout.vectors, &layout.public_keys, &layout.plaintexts] {
        validate_parallel_body_operation(protocol, operation)?;
        validate_select_operation(protocol, &operation.body)?;
    }
    Ok(())
}

fn validate_decryption_initial_encodings(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    input_injection: &InputInjectionLayout,
    boolean_layers: &BooleanLayersLayout,
    layout: &DecryptionInitialEncodingsLayout,
) -> Result<(), CertificateValidationError> {
    for artifact in [
        &layout.initial_state_artifact,
        &layout.one_preimage_artifact,
        &layout.witness_preimages_artifact,
        &layout.public_keys_artifact,
    ] {
        validate_workflow_artifact(protocol, workflow, artifact)?;
    }
    validate_parallel_loop_ref(protocol, &layout.witness_indices)?;
    validate_parallel_gather(protocol, &layout.witness_bits)?;
    validate_witness_digit_packing(protocol, &layout.witness_digits)?;
    for operation in
        [&layout.initial_projection_state, &layout.one_public_key, &layout.one_plaintext]
    {
        validate_operation_ref(protocol, operation)?;
    }
    for operation in &layout.zero_encoding {
        validate_operation_ref(protocol, operation)?;
    }
    validate_parallel_loop_ref(protocol, &layout.witness_state_indices)?;
    validate_parallel_gather(protocol, &layout.witness_states)?;
    validate_parallel_matrix_binary(protocol, &layout.witness_vectors, MatrixBinaryOp::Multiply)?;
    validate_parallel_loop_ref(protocol, &layout.witness_public_indices)?;
    validate_parallel_gather(protocol, &layout.witness_public_keys)?;
    for constant in &layout.witness_plaintext_constants {
        validate_parallel_loop_ref(protocol, constant)?;
    }
    validate_parallel_body_operation(protocol, &layout.witness_plaintexts)?;
    validate_select_operation(protocol, &layout.witness_plaintexts.body)?;
    validate_evaluate_int(protocol, &layout.instance_width)?;
    validate_parallel_loop_ref(protocol, &layout.packed_indices)?;
    validate_parallel_gather(protocol, &layout.packed_vectors)?;
    validate_parallel_gather(protocol, &layout.packed_public_keys)?;
    validate_parallel_gather(protocol, &layout.packed_plaintexts)?;
    validate_parallel_loop_ref(protocol, &layout.active_witness)?;
    for zeroes in &layout.active_witness_zeroes {
        validate_parallel_loop_ref(protocol, zeroes)?;
    }
    validate_encoding_component_operations(protocol, &layout.active_witness_selection)?;
    for component in &layout.instance_constants {
        for constant in component {
            validate_parallel_loop_ref(protocol, constant)?;
        }
    }
    validate_encoding_component_operations(protocol, &layout.selected_instance)?;
    validate_parallel_loop_ref(protocol, &layout.active_instance)?;
    validate_encoding_component_operations(protocol, &layout.circuit_inputs)?;

    let decryption = &workflow.decryption.stage;
    for operation in [
        &layout.witness_indices.operation,
        &layout.witness_bits.parallel_loop.operation,
        &layout.witness_digits.parallel_loop.operation,
        &layout.initial_projection_state.operation,
        &layout.one_public_key.operation,
        &layout.one_plaintext.operation,
        &layout.witness_state_indices.operation,
        &layout.witness_states.parallel_loop.operation,
        &layout.witness_vectors.parallel_loop,
        &layout.witness_public_indices.operation,
        &layout.witness_public_keys.parallel_loop.operation,
        &layout.witness_plaintexts.parallel_loop.operation,
        &layout.instance_width.operation,
        &layout.packed_indices.operation,
        &layout.packed_vectors.parallel_loop.operation,
        &layout.packed_public_keys.parallel_loop.operation,
        &layout.packed_plaintexts.parallel_loop.operation,
        &layout.active_witness.operation,
        &layout.active_instance.operation,
    ] {
        require_construction_stage(operation, decryption)?;
    }
    for operation in &layout.zero_encoding {
        require_construction_stage(&operation.operation, decryption)?;
        if !matches!(
            resolve_node(protocol, &operation.operation)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)
        ) || operation.inputs.len() != 2 ||
            operation.inputs[0].wire != operation.inputs[1].wire
        {
            return Err(CertificateValidationError::ConstructionTraceMismatch);
        }
    }
    let public_keys_input = layout.public_keys_artifact.consumer_input.node.wire(0);
    if !matches!(resolve_node(protocol, &layout.initial_projection_state.operation)?.kind(),
            NodeKind::FamilyGetStatic { index } if index == &IntExpr::constant(0)) ||
        !matches!(resolve_node(protocol, &layout.one_public_key.operation)?.kind(),
            NodeKind::FamilyGetStatic { index } if index == &IntExpr::constant(0)) ||
        !matches!(
            resolve_node(protocol, &layout.one_plaintext.operation)?.kind(),
            NodeKind::ConstantMatrix { value: mxx_ir_core::node::ConstantMatrix::Identity, .. }
        ) ||
        layout.initial_projection_state.inputs.first().map(|input| &input.wire) !=
            Some(&input_injection.final_states) ||
        layout.one_public_key.inputs.first().map(|input| &input.wire) != Some(&public_keys_input) ||
        layout.witness_bits.index_family.wire != layout.witness_indices.outputs[0] ||
        layout.witness_digits.parallel_loop.arguments.first().map(|input| &input.wire) !=
            layout.witness_bits.output_families.first() ||
        layout.witness_state_indices.outputs.len() != 1 ||
        layout.witness_states.index_family.wire != layout.witness_state_indices.outputs[0] ||
        layout.witness_states.source_families.first().map(|source| &source.wire) !=
            Some(&input_injection.final_states) ||
        layout.witness_vectors.left_family.wire != layout.witness_states.output_families[0] ||
        layout.witness_vectors.right_family.wire !=
            layout.witness_preimages_artifact.consumer_input.node.wire(0) ||
        layout.witness_public_keys.index_family.wire != layout.witness_public_indices.outputs[0] ||
        layout.witness_public_keys.source_families.first().map(|source| &source.wire) !=
            Some(&public_keys_input) ||
        layout.witness_plaintexts.parallel_loop.arguments.first().map(|argument| &argument.wire) !=
            layout.witness_bits.output_families.first() ||
        layout.packed_vectors.index_family.wire != layout.packed_indices.outputs[0] ||
        layout.packed_public_keys.index_family.wire != layout.packed_indices.outputs[0] ||
        layout.packed_plaintexts.index_family.wire != layout.packed_indices.outputs[0] ||
        layout.circuit_inputs.vectors.parallel_loop.outputs.first() !=
            Some(&boolean_layers.decryption.initial_vectors.wire) ||
        layout.circuit_inputs.public_keys.parallel_loop.outputs.first() !=
            Some(&boolean_layers.decryption.initial_public_keys.wire) ||
        layout.circuit_inputs.plaintexts.parallel_loop.outputs.first() !=
            Some(&boolean_layers.decryption.initial_plaintexts.wire)
    {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    }
    Ok(())
}

fn validate_input_preprocessing(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    layout: &DiamondInputPreprocessingLayout,
) -> Result<(), CertificateValidationError> {
    for artifact in [&layout.initial_state_artifact, &layout.transitions_artifact] {
        validate_artifact_provenance(protocol, artifact)?;
        if !workflow.artifacts.contains(artifact) ||
            artifact.producer_stage != workflow.encryption.stage
        {
            return Err(CertificateValidationError::InputPreprocessingMismatch);
        }
    }
    validate_parallel_operation(protocol, &layout.trapdoor_samples)?;
    validate_operation_ref(protocol, &layout.secret_sample)?;
    validate_operation_ref(protocol, &layout.message_selector)?;
    validate_operation_ref(protocol, &layout.initial_error_sample)?;
    validate_operation_ref(protocol, &layout.initial_public_product)?;
    validate_operation_ref(protocol, &layout.initial_state)?;
    validate_parallel_index_formula(protocol, &layout.transition_source_indices)?;
    validate_parallel_index_formula(protocol, &layout.transition_target_indices)?;
    validate_parallel_index_formula(protocol, &layout.digit_secret_indices)?;
    validate_parallel_operation(protocol, &layout.digit_secret_samples)?;
    validate_parallel_gather(protocol, &layout.digit_secrets)?;
    validate_parallel_gather(protocol, &layout.transition_sources)?;
    validate_parallel_gather(protocol, &layout.target_public_matrices)?;
    validate_transition_target(protocol, &layout.transition_targets)?;
    validate_parallel_preimage(protocol, &layout.transition_preimages)?;
    validate_parallel_loop_ref(protocol, &layout.final_indices)?;
    validate_parallel_gather(protocol, &layout.final_trapdoors)?;

    let encryption = &workflow.encryption.stage;
    let all_operations = [
        &layout.trapdoor_samples.parallel_loop.operation,
        &layout.secret_sample.operation,
        &layout.message_selector.operation,
        &layout.initial_error_sample.operation,
        &layout.initial_public_product.operation,
        &layout.initial_state.operation,
        &layout.transition_source_indices.parallel_loop.operation,
        &layout.transition_target_indices.parallel_loop.operation,
        &layout.digit_secret_indices.parallel_loop.operation,
        &layout.digit_secret_samples.parallel_loop.operation,
        &layout.digit_secrets.parallel_loop.operation,
        &layout.transition_sources.parallel_loop.operation,
        &layout.target_public_matrices.parallel_loop.operation,
        &layout.transition_targets.parallel_loop.operation,
        &layout.transition_preimages.parallel_loop.operation,
        &layout.final_indices.operation,
        &layout.final_trapdoors.parallel_loop.operation,
    ];
    if all_operations.iter().any(|operation| &operation.stage != encryption) ||
        !matches!(
            resolve_node(protocol, &layout.trapdoor_samples.body.operation)?.kind(),
            NodeKind::TrapdoorSample { .. }
        ) ||
        !matches!(resolve_node(protocol, &layout.secret_sample.operation)?.kind(),
            NodeKind::UniformSample { range, .. }
                if range.minimum == IntExpr::constant(-1) &&
                    range.maximum == IntExpr::constant(1)) ||
        !matches!(
            resolve_node(protocol, &layout.message_selector.operation)?.kind(),
            NodeKind::Concat { axis: ConcatAxis::Columns }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.initial_error_sample.operation)?.kind(),
            NodeKind::GaussianSample { .. }
        ) ||
        !matches!(
            resolve_node(protocol, &layout.initial_public_product.operation)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        ) ||
        !matches!(
            resolve_node(protocol, &layout.initial_state.operation)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Add)
        ) ||
        !matches!(resolve_node(protocol, &layout.digit_secret_samples.body.operation)?.kind(),
            NodeKind::UniformSample { range, .. }
                if range.minimum == IntExpr::constant(-1) &&
                    range.maximum == IntExpr::constant(1)) ||
        layout.trapdoor_samples.parallel_loop.outputs.len() != 2 ||
        layout.secret_sample.outputs.len() != 1 ||
        layout.message_selector.inputs.len() != 2 ||
        layout.message_selector.outputs.len() != 1 ||
        layout.message_selector.inputs[0].wire != layout.secret_sample.outputs[0] ||
        layout.initial_error_sample.inputs.len() != 0 ||
        layout.initial_error_sample.outputs.len() != 1 ||
        layout.initial_public_product.inputs.len() != 2 ||
        layout.initial_public_product.inputs[0].wire != layout.message_selector.outputs[0] ||
        layout.initial_public_product.outputs.len() != 1 ||
        layout.initial_state.inputs.iter().map(|input| &input.wire).collect::<Vec<_>>() !=
            vec![
                &layout.initial_public_product.outputs[0],
                &layout.initial_error_sample.outputs[0],
            ] ||
        layout.initial_state.outputs.len() != 1 ||
        layout.initial_state_artifact.producer_output.wire != layout.initial_state.outputs[0] ||
        layout.transition_source_indices.parallel_loop.outputs.len() != 1 ||
        layout.transition_target_indices.parallel_loop.outputs.len() != 1 ||
        layout.digit_secret_indices.parallel_loop.outputs.len() != 1 ||
        layout.digit_secret_samples.parallel_loop.outputs.len() != 1 ||
        layout.digit_secrets.index_family.wire !=
            layout.digit_secret_indices.parallel_loop.outputs[0] ||
        layout
            .digit_secrets
            .source_families
            .iter()
            .map(|source| &source.wire)
            .collect::<Vec<_>>() !=
            vec![&layout.digit_secret_samples.parallel_loop.outputs[0]] ||
        layout.transition_sources.index_family.wire !=
            layout.transition_source_indices.parallel_loop.outputs[0] ||
        layout
            .transition_sources
            .source_families
            .iter()
            .map(|source| &source.wire)
            .collect::<Vec<_>>() !=
            layout.trapdoor_samples.parallel_loop.outputs.iter().collect::<Vec<_>>() ||
        layout.target_public_matrices.index_family.wire !=
            layout.transition_target_indices.parallel_loop.outputs[0] ||
        layout.target_public_matrices.source_families.len() != 1 ||
        layout.target_public_matrices.source_families[0].wire !=
            layout.trapdoor_samples.parallel_loop.outputs[0] ||
        layout
            .transition_targets
            .parallel_loop
            .arguments
            .iter()
            .map(|argument| &argument.wire)
            .collect::<Vec<_>>() !=
            vec![
                &layout.digit_secrets.output_families[0],
                &layout.target_public_matrices.output_families[0],
            ] ||
        layout
            .transition_preimages
            .parallel_loop
            .arguments
            .iter()
            .map(|argument| &argument.wire)
            .collect::<Vec<_>>() !=
            layout
                .transition_sources
                .output_families
                .iter()
                .chain(layout.transition_targets.parallel_loop.outputs.iter())
                .collect::<Vec<_>>() ||
        layout.transitions_artifact.producer_output.wire !=
            layout.transition_preimages.parallel_loop.outputs[0] ||
        layout.final_indices.outputs.len() != 1 ||
        layout.final_trapdoors.index_family.wire != layout.final_indices.outputs[0] ||
        layout
            .final_trapdoors
            .source_families
            .iter()
            .map(|source| &source.wire)
            .collect::<Vec<_>>() !=
            layout.trapdoor_samples.parallel_loop.outputs.iter().collect::<Vec<_>>()
    {
        return Err(CertificateValidationError::InputPreprocessingMismatch);
    }
    validate_static_get(
        protocol,
        &layout.initial_public_product.inputs[1].wire,
        &layout.trapdoor_samples.parallel_loop.outputs[0],
        0,
    )?;
    Ok(())
}

fn validate_input_injection(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    layout: &InputInjectionLayout,
) -> Result<(), CertificateValidationError> {
    if layout.state_scan.stage != workflow.decryption.stage {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    validate_initial_state_expansion(protocol, &layout.initial_states_expansion)?;
    validate_sequential_loop(
        protocol,
        &layout.state_scan,
        &layout.body_scope,
        &[&layout.initial_states],
        &[&layout.packed_digits, &layout.transition_family],
        &[&layout.final_states],
    )?;
    validate_parallel_matrix_binary(protocol, &layout.state_product, MatrixBinaryOp::Multiply)?;
    validate_boolean_loop_body(
        protocol,
        &layout.state_scan,
        &layout.body_scope,
        &[&layout.initial_states, &layout.packed_digits, &layout.transition_family],
        &[&layout.body_initial_states, &layout.body_packed_digits, &layout.body_transition_family],
        &[&layout.body_final_states],
        &[&layout.final_states],
    )?;
    validate_dynamic_get(protocol, &layout.selected_digit, &layout.body_packed_digits)?;
    validate_parallel_index_formula(protocol, &layout.source_indices)?;
    let [source_lower_bound] = layout.source_indices.parallel_loop.arguments.as_slice() else {
        return Err(CertificateValidationError::ConstructionTraceMismatch);
    };
    validate_online_source_lower_bound(protocol, &source_lower_bound.wire)?;
    validate_parallel_family_get(protocol, &layout.source_states)?;
    validate_parallel_index_formula(protocol, &layout.transition_indices)?;
    validate_parallel_family_get(protocol, &layout.selected_transitions)?;
    let stage = stage(protocol, &layout.state_scan.stage)?;
    let NodeKind::SequentialLoop(scan) = resolve_node(protocol, &layout.state_scan)?.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    let selected_index = resolve_node(protocol, &layout.selected_digit.index.wire.node)?;
    let body =
        stage.graph.scope(&layout.body_scope).ok_or(CertificateValidationError::MissingScope)?;
    if !scope_is_within(&layout.state_product.parallel_loop.scope, &layout.body_scope) ||
        body.outputs() != [layout.state_product.output_family.as_wire_ref()] ||
        layout.body_final_states != layout.state_product.output_family ||
        layout.initial_states_expansion.parallel_loop.outputs !=
            [layout.initial_states.wire.clone()] ||
        layout.source_indices.parallel_loop.outputs.len() != 1 ||
        layout.transition_indices.parallel_loop.outputs.len() != 1 ||
        layout.source_states.index_family.wire != layout.source_indices.parallel_loop.outputs[0] ||
        layout.source_states.source_family.wire != layout.body_initial_states ||
        layout.selected_transitions.index_family.wire !=
            layout.transition_indices.parallel_loop.outputs[0] ||
        layout.selected_transitions.source_family.wire != layout.body_transition_family ||
        layout.state_product.left_family.wire != layout.source_states.output_family ||
        layout.state_product.right_family.wire != layout.selected_transitions.output_family ||
        layout.source_indices.parallel_loop.operation.scope != layout.body_scope ||
        layout.transition_indices.parallel_loop.operation.scope != layout.body_scope ||
        layout.selected_digit.index.wire.port != Port(0) ||
        !selected_index.arguments().is_empty() ||
        selected_index.output_types().len() != 1 ||
        !matches!(
            selected_index.kind(),
            NodeKind::EvaluateInt(IntExpr::LoopIndex(slot)) if *slot == scan.index_slot
        )
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_boolean_layers(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    layout: &BooleanLayersLayout,
) -> Result<(), CertificateValidationError> {
    validate_artifact_provenance(protocol, &layout.public_keys_artifact)?;
    if !workflow.artifacts.contains(&layout.public_keys_artifact) {
        return Err(CertificateValidationError::ArtifactProvenanceMismatch);
    }
    let encryption = &layout.encryption;
    if encryption.layer_scan.stage != workflow.encryption.stage {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    validate_sequential_loop(
        protocol,
        &encryption.layer_scan,
        &encryption.body_scope,
        &[&encryption.initial_public_keys],
        &[
            &encryption.active_gate_counts,
            &encryption.gate_kinds,
            &encryption.left_sources,
            &encryption.right_sources,
            &encryption.one_public_key,
        ],
        &[&encryption.final_public_keys],
    )?;
    validate_dynamic_get(protocol, &encryption.selected_output, &encryption.final_public_keys)?;
    validate_boolean_loop_body(
        protocol,
        &encryption.layer_scan,
        &encryption.body_scope,
        &[
            &encryption.initial_public_keys,
            &encryption.active_gate_counts,
            &encryption.gate_kinds,
            &encryption.left_sources,
            &encryption.right_sources,
            &encryption.one_public_key,
        ],
        &[
            &encryption.body_initial_public_keys,
            &encryption.body_active_gate_counts,
            &encryption.body_gate_kinds,
            &encryption.body_left_sources,
            &encryption.body_right_sources,
            &encryption.body_one_public_key,
        ],
        &[&encryption.body_final_public_keys],
        &[&encryption.final_public_keys],
    )?;
    validate_boolean_metadata(
        protocol,
        &encryption.layer_scan,
        &encryption.metadata,
        [
            (&encryption.active_gate_counts, &encryption.body_active_gate_counts),
            (&encryption.gate_kinds, &encryption.body_gate_kinds),
            (&encryption.left_sources, &encryption.body_left_sources),
            (&encryption.right_sources, &encryption.body_right_sources),
        ],
    )?;

    let decryption = &layout.decryption;
    if decryption.layer_scan.stage != workflow.decryption.stage {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    validate_sequential_loop(
        protocol,
        &decryption.layer_scan,
        &decryption.body_scope,
        &[
            &decryption.initial_vectors,
            &decryption.initial_public_keys,
            &decryption.initial_plaintexts,
        ],
        &[
            &decryption.active_gate_counts,
            &decryption.gate_kinds,
            &decryption.left_sources,
            &decryption.right_sources,
            &decryption.one_vector,
            &decryption.one_public_key,
            &decryption.one_plaintext,
        ],
        &[&decryption.final_vectors, &decryption.final_public_keys, &decryption.final_plaintexts],
    )?;
    validate_dynamic_get(protocol, &decryption.selected_vector, &decryption.final_vectors)?;
    validate_boolean_loop_body(
        protocol,
        &decryption.layer_scan,
        &decryption.body_scope,
        &[
            &decryption.initial_vectors,
            &decryption.initial_public_keys,
            &decryption.initial_plaintexts,
            &decryption.active_gate_counts,
            &decryption.gate_kinds,
            &decryption.left_sources,
            &decryption.right_sources,
            &decryption.one_vector,
            &decryption.one_public_key,
            &decryption.one_plaintext,
        ],
        &[
            &decryption.body_initial_vectors,
            &decryption.body_initial_public_keys,
            &decryption.body_initial_plaintexts,
            &decryption.body_active_gate_counts,
            &decryption.body_gate_kinds,
            &decryption.body_left_sources,
            &decryption.body_right_sources,
            &decryption.body_one_vector,
            &decryption.body_one_public_key,
            &decryption.body_one_plaintext,
        ],
        &[
            &decryption.body_final_vectors,
            &decryption.body_final_public_keys,
            &decryption.body_final_plaintexts,
        ],
        &[&decryption.final_vectors, &decryption.final_public_keys, &decryption.final_plaintexts],
    )?;
    validate_boolean_metadata(
        protocol,
        &decryption.layer_scan,
        &decryption.metadata,
        [
            (&decryption.active_gate_counts, &decryption.body_active_gate_counts),
            (&decryption.gate_kinds, &decryption.body_gate_kinds),
            (&decryption.left_sources, &decryption.body_left_sources),
            (&decryption.right_sources, &decryption.body_right_sources),
        ],
    )?;

    for (left, right) in [
        (&encryption.active_gate_counts, &decryption.active_gate_counts),
        (&encryption.gate_kinds, &decryption.gate_kinds),
        (&encryption.left_sources, &decryption.left_sources),
        (&encryption.right_sources, &decryption.right_sources),
        (&encryption.selected_output.index, &decryption.selected_vector.index),
    ] {
        if protocol_input_name(protocol, &left.wire)? != protocol_input_name(protocol, &right.wire)?
        {
            return Err(CertificateValidationError::CircuitInputMismatch);
        }
    }

    validate_encrypt_decomposition(
        protocol,
        &encryption.body_scope,
        &layout.encrypt_public_key_rhs_decomposition,
    )?;
    validate_decrypt_decomposition(
        protocol,
        &decryption.body_scope,
        &layout.decrypt_encoding_rhs_decomposition,
    )?;
    validate_local_boolean_gate(protocol, &layout.encryption_gate)?;
    if layout.encryption_gate.body_scope != layout.encrypt_public_key_rhs_decomposition.body_scope {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    for component in
        [&layout.decryption_vectors, &layout.decryption_public_keys, &layout.decryption_plaintexts]
    {
        validate_family_boolean_gate(protocol, component)?;
    }
    for (component, state_input, state_output, one) in [
        (
            &layout.decryption_vectors,
            &decryption.body_initial_vectors,
            &decryption.body_final_vectors,
            &decryption.body_one_vector,
        ),
        (
            &layout.decryption_public_keys,
            &decryption.body_initial_public_keys,
            &decryption.body_final_public_keys,
            &decryption.body_one_public_key,
        ),
        (
            &layout.decryption_plaintexts,
            &decryption.body_initial_plaintexts,
            &decryption.body_final_plaintexts,
            &decryption.body_one_plaintext,
        ),
    ] {
        if component.state_input != *state_input ||
            component.state_output != *state_output ||
            component.left_selection.index_family.wire !=
                decryption.metadata.left_source.gathered.output_family ||
            component.right_selection.index_family.wire !=
                decryption.metadata.right_source.gathered.output_family ||
            component.opcode_family != decryption.metadata.opcode.gathered.output_family ||
            !component.one_repetition.arguments.iter().any(|argument| argument.wire == *one) ||
            !component.active_mask.arguments.iter().any(|argument| {
                argument.wire == decryption.metadata.active_gate_count.selected.output
            })
        {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    if layout.encryption_gate.parent_loop.operation !=
        layout.encrypt_public_key_rhs_decomposition.enclosing_parallel_loop ||
        layout.encryption_gate.right_family !=
            layout.encrypt_public_key_rhs_decomposition.right_public_key_family ||
        layout.encryption_gate.body_right !=
            layout.encrypt_public_key_rhs_decomposition.body_right_public_key ||
        layout.encryption_gate.product.right.wire !=
            layout.encrypt_public_key_rhs_decomposition.local.materialized ||
        layout.encryption_gate.sum.right.wire != layout.encryption_gate.body_right ||
        layout.encryption_gate.opcode_family.wire !=
            encryption.metadata.opcode.gathered.output_family ||
        layout.encryption_gate.left_selection.source_family.wire !=
            encryption.body_initial_public_keys ||
        layout.encryption_gate.left_selection.index_family.wire !=
            encryption.metadata.left_source.gathered.output_family ||
        layout.encryption_gate.left_selection.output_family !=
            layout.encryption_gate.left_family.wire ||
        layout.encrypt_public_key_rhs_decomposition.right_selection.source_family.wire !=
            encryption.body_initial_public_keys ||
        layout.encrypt_public_key_rhs_decomposition.right_selection.index_family.wire !=
            encryption.metadata.right_source.gathered.output_family ||
        layout.encrypt_public_key_rhs_decomposition.right_selection.output_family !=
            layout.encryption_gate.right_family.wire ||
        layout.encryption_gate.one_public_key.wire != encryption.body_one_public_key ||
        layout.encryption_gate.active_gate_count.wire !=
            encryption.metadata.active_gate_count.selected.output ||
        layout.encryption_gate.parent_loop.outputs !=
            vec![encryption.body_final_public_keys.clone()] ||
        layout.decryption_public_keys.state_input != decryption.body_initial_public_keys ||
        layout.decryption_public_keys.state_output != decryption.body_final_public_keys ||
        layout.decryption_public_keys.right_selection !=
            layout.decrypt_encoding_rhs_decomposition.right_selection ||
        layout.decryption_public_keys.right_selection.output_family !=
            layout.decrypt_encoding_rhs_decomposition.right_public_key_family.wire ||
        !matches!(
            &layout.decryption_public_keys.product,
            FamilyProductRef::Direct(product)
                if product.right_family.wire ==
                    layout.decrypt_encoding_rhs_decomposition.decomposition_family
        ) ||
        !matches!(
            &layout.decryption_vectors.product,
            FamilyProductRef::EncodingVector {
                left_times_right_decomposition,
                right_times_left_plaintext,
                ..
            }
                if left_times_right_decomposition.right_family.wire ==
                    layout.decrypt_encoding_rhs_decomposition.decomposition_family &&
                    left_times_right_decomposition.left_family.wire ==
                        layout.decryption_vectors.left_selection.output_family &&
                    right_times_left_plaintext.left_family.wire ==
                        layout.decryption_vectors.right_selection.output_family &&
                    right_times_left_plaintext.right_family.wire ==
                        layout.decryption_plaintexts.left_selection.output_family
        ) ||
        !matches!(
            &layout.decryption_public_keys.product,
            FamilyProductRef::Direct(product)
                if product.left_family.wire ==
                    layout.decryption_public_keys.left_selection.output_family
        ) ||
        !matches!(
            &layout.decryption_plaintexts.product,
            FamilyProductRef::Direct(product)
                if product.left_family.wire ==
                    layout.decryption_plaintexts.left_selection.output_family &&
                    product.right_family.wire ==
                    layout.decryption_plaintexts.right_selection.output_family
        )
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }

    for (left, right) in [
        (
            &encryption.metadata.active_gate_count.source_input_name,
            &decryption.metadata.active_gate_count.source_input_name,
        ),
        (
            &encryption.metadata.opcode.source_input_name,
            &decryption.metadata.opcode.source_input_name,
        ),
        (
            &encryption.metadata.left_source.source_input_name,
            &decryption.metadata.left_source.source_input_name,
        ),
        (
            &encryption.metadata.right_source.source_input_name,
            &decryption.metadata.right_source.source_input_name,
        ),
    ] {
        if left != right {
            return Err(CertificateValidationError::CircuitInputMismatch);
        }
    }

    let decompositions = [
        &layout.encrypt_public_key_rhs_decomposition.local.materialized,
        &layout.decrypt_encoding_rhs_decomposition.local.materialized,
        &layout.decrypt_encoding_rhs_decomposition.decomposition_family,
    ];
    for interface in [&workflow.encryption, &workflow.decryption] {
        if interface.outputs.iter().any(|output| decompositions.contains(&&output.wire)) {
            return Err(CertificateValidationError::ExportedGateDecomposition);
        }
    }
    Ok(())
}

fn validate_sequential_loop(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    body_scope: &FrozenGraphScopeId,
    carried: &[&CoreOperandRef],
    invariants: &[&CoreOperandRef],
    outputs: &[&CoreWireRef],
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, operation)?;
    let NodeKind::SequentialLoop(specification) = node.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    let stage = stage(protocol, &operation.stage)?;
    if stage.graph.child_scope_id(&operation.scope, operation.node).as_ref() != Some(body_scope) {
        return Err(CertificateValidationError::LoopBodyMismatch);
    }
    let scope =
        stage.graph.scope(&operation.scope).ok_or(CertificateValidationError::MissingScope)?;
    let arguments = scope.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    if specification.carried_count != carried.len() ||
        arguments.len() != carried.len() + invariants.len() ||
        outputs.len() != carried.len()
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    for (index, operand) in carried.iter().chain(invariants).enumerate() {
        validate_operand(protocol, operand)?;
        if operand.node != *operation || operand.operand as usize != index {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    for (index, output) in outputs.iter().enumerate() {
        validate_operation_output(protocol, operation, output)?;
        if output.port.0 as usize != index {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    Ok(())
}

fn validate_boolean_loop_body(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    body_scope: &FrozenGraphScopeId,
    outer: &[&CoreOperandRef],
    inner: &[&CoreWireRef],
    body_outputs: &[&CoreWireRef],
    outputs: &[&CoreWireRef],
) -> Result<(), CertificateValidationError> {
    let stage = stage(protocol, &operation.stage)?;
    let body = stage.graph.scope(body_scope).ok_or(CertificateValidationError::MissingScope)?;
    if outer.len() != inner.len() ||
        body.outputs().len() != body_outputs.len() ||
        body.inputs().len() != inner.len() ||
        outputs.len() != body_outputs.len()
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    for ((outer, inner), actual) in outer.iter().zip(inner).zip(body.inputs()) {
        validate_operand(protocol, outer)?;
        validate_wire(protocol, inner)?;
        if outer.node != *operation || inner.as_wire_ref() != *actual {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    for (index, ((body_output, output), actual)) in
        body_outputs.iter().zip(outputs).zip(body.outputs()).enumerate()
    {
        validate_wire(protocol, body_output)?;
        validate_operation_output(protocol, operation, output)?;
        if body_output.as_wire_ref() != *actual || output.port.0 as usize != index {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    Ok(())
}

fn validate_parallel_loop_ref(
    protocol: &ProtocolDecl,
    layout: &ParallelLoopRef,
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, &layout.operation)?;
    let NodeKind::ParallelLoop(specification) = node.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    let stage = stage(protocol, &layout.operation.stage)?;
    if stage.graph.child_scope_id(&layout.operation.scope, layout.operation.node).as_ref() !=
        Some(&layout.body_scope) ||
        layout.count != specification.count ||
        layout.index_slot != specification.index_slot ||
        layout.bindings != specification.bindings ||
        layout.input_modes !=
            specification
                .input_modes
                .iter()
                .map(CertifiedLoopInputMode::from)
                .collect::<Vec<_>>()
    {
        return Err(CertificateValidationError::ParallelLoopBoundaryMismatch);
    }
    let parent = stage
        .graph
        .scope(&layout.operation.scope)
        .ok_or(CertificateValidationError::MissingScope)?;
    let body =
        stage.graph.scope(&layout.body_scope).ok_or(CertificateValidationError::MissingScope)?;
    let arguments = parent.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    if arguments.len() != layout.arguments.len() ||
        body.inputs().len() != layout.body_inputs.len() ||
        body.outputs().len() != layout.body_outputs.len() ||
        node.output_types().len() != layout.outputs.len()
    {
        return Err(CertificateValidationError::ParallelLoopBoundaryMismatch);
    }
    for (index, ((argument, declared), body_input)) in
        arguments.iter().zip(&layout.arguments).zip(&layout.body_inputs).enumerate()
    {
        validate_operand(protocol, declared)?;
        validate_wire(protocol, body_input)?;
        if declared.node != layout.operation ||
            declared.operand as usize != index ||
            declared.wire.as_wire_ref() != *argument ||
            body_input.as_wire_ref() != body.inputs()[index]
        {
            return Err(CertificateValidationError::ParallelLoopBoundaryMismatch);
        }
    }
    for (index, (body_output, output)) in
        layout.body_outputs.iter().zip(&layout.outputs).enumerate()
    {
        validate_wire(protocol, body_output)?;
        validate_operation_output(protocol, &layout.operation, output)?;
        if body_output.as_wire_ref() != body.outputs()[index] || output.port.0 as usize != index {
            return Err(CertificateValidationError::ParallelLoopBoundaryMismatch);
        }
    }
    Ok(())
}

fn validate_evaluate_int(
    protocol: &ProtocolDecl,
    layout: &EvaluateIntRef,
) -> Result<(), CertificateValidationError> {
    if !matches!(resolve_node(protocol, &layout.operation)?.kind(), NodeKind::EvaluateInt(expression) if expression == &layout.expression)
    {
        return Err(CertificateValidationError::LayerMetadataMismatch);
    }
    validate_operation_output(protocol, &layout.operation, &layout.evaluated)?;
    match &layout.materialization {
        Some(materialization) => {
            if !matches!(
                resolve_node(protocol, &materialization.operation)?.kind(),
                NodeKind::IntBinary(IntBinaryOp::Add)
            ) {
                return Err(CertificateValidationError::LayerMetadataMismatch);
            }
            validate_binary_node(protocol, materialization)?;
            if materialization.left.wire != layout.evaluated ||
                !is_constant_int(protocol, &materialization.right.wire, 0)? ||
                layout.output != materialization.output
            {
                return Err(CertificateValidationError::LayerMetadataMismatch);
            }
        }
        None if layout.output == layout.evaluated => {}
        None => return Err(CertificateValidationError::LayerMetadataMismatch),
    }
    Ok(())
}

fn validate_parallel_family_get(
    protocol: &ProtocolDecl,
    layout: &ParallelFamilyGetRef,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &layout.parallel_loop)?;
    validate_dynamic_get(protocol, &layout.get, &layout.body_source)?;
    if layout.parallel_loop.arguments.len() != 2 ||
        layout.parallel_loop.body_inputs.len() != 2 ||
        layout.parallel_loop.body_outputs.len() != 1 ||
        layout.parallel_loop.outputs.len() != 1 ||
        layout.index_family != layout.parallel_loop.arguments[0] ||
        layout.source_family != layout.parallel_loop.arguments[1] ||
        layout.body_index != layout.parallel_loop.body_inputs[0] ||
        layout.body_source != layout.parallel_loop.body_inputs[1] ||
        layout.get.index.wire != layout.body_index ||
        layout.get.output != layout.parallel_loop.body_outputs[0] ||
        layout.output_family != layout.parallel_loop.outputs[0]
    {
        return Err(CertificateValidationError::LayerMetadataMismatch);
    }
    Ok(())
}

fn validate_boolean_metadata(
    protocol: &ProtocolDecl,
    sequential: &CoreNodeRef,
    layout: &BooleanLayerMetadataLayout,
    expected: [(&CoreOperandRef, &CoreWireRef); 4],
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, sequential)?;
    let NodeKind::SequentialLoop(specification) = node.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    let scalar = &layout.active_gate_count;
    validate_evaluate_int(protocol, &scalar.layer_index)?;
    validate_dynamic_get(protocol, &scalar.selected, &scalar.body_source)?;
    if scalar.sequential_operand != *expected[0].0 ||
        scalar.body_source != *expected[0].1 ||
        scalar.root_input != scalar.sequential_operand.wire ||
        scalar.layer_index.expression != IntExpr::LoopIndex(specification.index_slot) ||
        scalar.selected.index.wire != scalar.layer_index.output ||
        input_name(protocol, &scalar.root_input)? != scalar.source_input_name
    {
        return Err(CertificateValidationError::LayerMetadataMismatch);
    }
    for (metadata, (outer, inner)) in [
        (&layout.opcode, expected[1]),
        (&layout.left_source, expected[2]),
        (&layout.right_source, expected[3]),
    ] {
        validate_parallel_loop_ref(protocol, &metadata.flattened_indices)?;
        validate_evaluate_int(protocol, &metadata.flattened_index)?;
        validate_parallel_family_get(protocol, &metadata.gathered)?;
        let expected_expression = IntExpr::Add(
            Box::new(IntExpr::Mul(
                Box::new(IntExpr::LoopIndex(specification.index_slot)),
                Box::new(metadata.flattened_indices.count.clone()),
            )),
            Box::new(IntExpr::LoopIndex(metadata.flattened_indices.index_slot)),
        )
        .canonicalize();
        if metadata.sequential_operand != *outer ||
            metadata.body_source != *inner ||
            metadata.root_input != metadata.sequential_operand.wire ||
            metadata.flattened_indices.arguments.len() != 0 ||
            metadata.flattened_indices.body_outputs !=
                vec![metadata.flattened_index.output.clone()] ||
            metadata.flattened_index.expression.canonicalize() != expected_expression ||
            metadata.gathered.index_family.wire != metadata.flattened_indices.outputs[0] ||
            metadata.gathered.source_family.wire != metadata.body_source ||
            input_name(protocol, &metadata.root_input)? != metadata.source_input_name
        {
            return Err(CertificateValidationError::LayerMetadataMismatch);
        }
    }
    Ok(())
}

fn validate_dynamic_get(
    protocol: &ProtocolDecl,
    operation: &DynamicFamilyGetRef,
    expected_family: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    if !matches!(resolve_node(protocol, &operation.operation)?.kind(), NodeKind::FamilyGetDynamic) {
        return Err(CertificateValidationError::WrongNodeKind);
    }
    validate_operand(protocol, &operation.family)?;
    validate_operand(protocol, &operation.index)?;
    if operation.family.node != operation.operation ||
        operation.family.operand != 0 ||
        operation.family.wire != *expected_family ||
        operation.index.node != operation.operation ||
        operation.index.operand != 1
    {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_operation_output(protocol, &operation.operation, &operation.output)
}

fn validate_encrypt_decomposition(
    protocol: &ProtocolDecl,
    boolean_body: &FrozenGraphScopeId,
    layout: &EncryptPublicKeyRhsDecomposition,
) -> Result<(), CertificateValidationError> {
    validate_parallel_family_get(protocol, &layout.right_selection)?;
    validate_parallel_input(
        protocol,
        &layout.enclosing_parallel_loop,
        &layout.body_scope,
        &layout.right_public_key_family,
        &layout.body_right_public_key,
    )?;
    if !scope_is_within(&layout.enclosing_parallel_loop.scope, boolean_body) {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    if layout.right_selection.output_family != layout.right_public_key_family.wire {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    validate_local_decomposition(protocol, &layout.local, &layout.body_scope)?;
    if layout.local.right_public_key.wire != layout.body_right_public_key {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_multiply_consumer(
        protocol,
        &layout.multiplication_consumer,
        &layout.local.materialized,
    )
}

fn validate_decrypt_decomposition(
    protocol: &ProtocolDecl,
    boolean_body: &FrozenGraphScopeId,
    layout: &DecryptEncodingRhsDecomposition,
) -> Result<(), CertificateValidationError> {
    validate_parallel_family_get(protocol, &layout.right_selection)?;
    validate_parallel_input(
        protocol,
        &layout.decomposition_loop,
        &layout.body_scope,
        &layout.right_public_key_family,
        &layout.body_right_public_key,
    )?;
    if !scope_is_within(&layout.decomposition_loop.scope, boolean_body) {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    if layout.right_selection.output_family != layout.right_public_key_family.wire {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    validate_local_decomposition(protocol, &layout.local, &layout.body_scope)?;
    if layout.local.right_public_key.wire != layout.body_right_public_key {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_wire(protocol, &layout.body_output)?;
    let stage = stage(protocol, &layout.decomposition_loop.stage)?;
    let body =
        stage.graph.scope(&layout.body_scope).ok_or(CertificateValidationError::MissingScope)?;
    if body.outputs() != [layout.body_output.as_wire_ref()] ||
        !is_decomposition_materialization(
            protocol,
            &layout.local.decomposition,
            &layout.body_output,
        )?
    {
        return Err(CertificateValidationError::OutputMismatch);
    }
    validate_operation_output(protocol, &layout.decomposition_loop, &layout.decomposition_family)?;
    validate_parallel_consumer(
        protocol,
        boolean_body,
        &layout.decomposition_family,
        &layout.public_key_consumer,
    )?;
    validate_parallel_consumer(
        protocol,
        boolean_body,
        &layout.decomposition_family,
        &layout.vector_consumer,
    )?;
    if layout.public_key_consumer.consumer_loop == layout.vector_consumer.consumer_loop {
        return Err(CertificateValidationError::DecompositionConsumerMismatch);
    }
    Ok(())
}

fn validate_local_decomposition(
    protocol: &ProtocolDecl,
    local: &LocalGadgetDecompositionRef,
    expected_scope: &FrozenGraphScopeId,
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, &local.decomposition_node)?;
    let NodeKind::GadgetDecompose { small, digit_count, .. } = node.kind() else {
        return Err(CertificateValidationError::WrongNodeKind);
    };
    if *small || digit_count.is_none() || &local.decomposition_node.scope != expected_scope {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    validate_parameter(
        &local.decomposition_node,
        &local.base,
        CoreNodeParameter::GadgetDecomposeBase,
    )?;
    validate_parameter(
        &local.decomposition_node,
        &local.digit_count,
        CoreNodeParameter::GadgetDecomposeDigitCount,
    )?;
    validate_operand(protocol, &local.right_public_key)?;
    if local.right_public_key.node != local.decomposition_node ||
        local.right_public_key.operand != 0
    {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_operation_output(protocol, &local.decomposition_node, &local.decomposition)?;
    if !is_decomposition_materialization(protocol, &local.decomposition, &local.materialized)? {
        return Err(CertificateValidationError::OutputMismatch);
    }
    Ok(())
}

fn validate_parallel_input(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    body_scope: &FrozenGraphScopeId,
    outer_operand: &CoreOperandRef,
    body_input: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    if !matches!(resolve_node(protocol, operation)?.kind(), NodeKind::ParallelLoop(_)) {
        return Err(CertificateValidationError::WrongNodeKind);
    }
    let stage = stage(protocol, &operation.stage)?;
    if stage.graph.child_scope_id(&operation.scope, operation.node).as_ref() != Some(body_scope) {
        return Err(CertificateValidationError::LoopBodyMismatch);
    }
    validate_operand(protocol, outer_operand)?;
    if outer_operand.node != *operation {
        return Err(CertificateValidationError::OperandMismatch);
    }
    let body = stage.graph.scope(body_scope).ok_or(CertificateValidationError::MissingScope)?;
    let expected = body
        .inputs()
        .get(outer_operand.operand as usize)
        .ok_or(CertificateValidationError::LoopLayoutMismatch)?;
    if body_input.node.stage != operation.stage ||
        body_input.node.scope != *body_scope ||
        body_input.as_wire_ref() != *expected
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_parallel_consumer(
    protocol: &ProtocolDecl,
    boolean_body: &FrozenGraphScopeId,
    decomposition_family: &CoreWireRef,
    consumer: &ParallelDecompositionConsumer,
) -> Result<(), CertificateValidationError> {
    validate_parallel_input(
        protocol,
        &consumer.consumer_loop,
        &consumer.body_scope,
        &consumer.decomposition_family,
        &consumer.body_decomposition,
    )?;
    if !scope_is_within(&consumer.consumer_loop.scope, boolean_body) ||
        consumer.decomposition_family.wire != *decomposition_family
    {
        return Err(CertificateValidationError::DecompositionScopeMismatch);
    }
    validate_multiply_consumer(
        protocol,
        &consumer.multiplication_consumer,
        &consumer.body_decomposition,
    )
}

fn validate_multiply_consumer(
    protocol: &ProtocolDecl,
    consumer: &CoreOperandRef,
    expected: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    validate_operand(protocol, consumer)?;
    if consumer.operand != 1 ||
        consumer.wire != *expected ||
        !matches!(
            resolve_node(protocol, &consumer.node)?.kind(),
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply)
        )
    {
        return Err(CertificateValidationError::DecompositionConsumerMismatch);
    }
    Ok(())
}

fn validate_local_boolean_gate(
    protocol: &ProtocolDecl,
    layout: &LocalBooleanGateLayout,
) -> Result<(), CertificateValidationError> {
    validate_parallel_loop_ref(protocol, &layout.parent_loop)?;
    validate_parallel_family_get(protocol, &layout.left_selection)?;
    validate_matrix_binary(protocol, &layout.zero, MatrixBinaryOp::Subtract)?;
    validate_matrix_binary(protocol, &layout.not, MatrixBinaryOp::Subtract)?;
    validate_matrix_binary(protocol, &layout.product, MatrixBinaryOp::Multiply)?;
    validate_matrix_binary(protocol, &layout.sum, MatrixBinaryOp::Add)?;
    validate_matrix_binary(protocol, &layout.two_product, MatrixBinaryOp::Multiply)?;
    validate_matrix_binary(protocol, &layout.xor, MatrixBinaryOp::Subtract)?;
    validate_six_way_select(protocol, &layout.candidate_select)?;
    validate_two_way_select(protocol, &layout.active_select)?;
    let in_scope = |node: &CoreNodeRef| node.scope == layout.body_scope;
    if !in_scope(&layout.zero.operation) ||
        !in_scope(&layout.not.operation) ||
        !in_scope(&layout.product.operation) ||
        !in_scope(&layout.sum.operation) ||
        !in_scope(&layout.two_product.operation) ||
        !in_scope(&layout.xor.operation) ||
        !in_scope(&layout.candidate_select.operation) ||
        !in_scope(&layout.active_select.operation) ||
        layout.parent_loop.body_scope != layout.body_scope ||
        layout.parent_loop.arguments.len() != 5 ||
        layout.parent_loop.body_inputs.len() != 5 ||
        layout.parent_loop.body_outputs != vec![layout.active_select.output.clone()] ||
        layout.parent_loop.outputs.len() != 1 ||
        layout.body_opcode != layout.candidate_select.selector.wire ||
        layout.body_left != layout.copy ||
        layout.left_selection.output_family != layout.left_family.wire ||
        layout.zero.left.wire != layout.one ||
        layout.zero.right.wire != layout.one ||
        layout.not.left.wire != layout.one ||
        layout.not.right.wire != layout.copy ||
        layout.product.left.wire != layout.copy ||
        layout.sum.left.wire != layout.copy ||
        layout.sum.right.wire != layout.body_right ||
        layout.two_product.left.wire != layout.product.output ||
        layout.xor.left.wire != layout.sum.output ||
        layout.xor.right.wire != layout.two_product.output ||
        layout.candidate_select.branches[0].wire != layout.zero.output ||
        layout.candidate_select.branches[1].wire != layout.one ||
        layout.candidate_select.branches[2].wire != layout.copy ||
        layout.candidate_select.branches[3].wire != layout.not.output ||
        layout.candidate_select.branches[4].wire != layout.product.output ||
        layout.candidate_select.branches[5].wire != layout.xor.output ||
        layout.active_select.branches[0].wire != layout.candidate_select.branches[0].wire ||
        layout.active_select.branches[1].wire != layout.candidate_select.output
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    for (outer, inner) in [
        (&layout.opcode_family, &layout.body_opcode),
        (&layout.left_family, &layout.body_left),
        (&layout.right_family, &layout.body_right),
        (&layout.one_public_key, &layout.body_one_public_key),
        (&layout.active_gate_count, &layout.body_active_gate_count),
    ] {
        let index = outer.operand as usize;
        if layout.parent_loop.arguments.get(index) != Some(outer) ||
            layout.parent_loop.body_inputs.get(index) != Some(inner)
        {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    Ok(())
}

fn validate_family_boolean_gate(
    protocol: &ProtocolDecl,
    layout: &FamilyBooleanGateLayout,
) -> Result<(), CertificateValidationError> {
    validate_parallel_family_get(protocol, &layout.left_selection)?;
    validate_parallel_family_get(protocol, &layout.right_selection)?;
    validate_parallel_loop_ref(protocol, &layout.one_repetition)?;
    validate_parallel_loop_ref(protocol, &layout.active_mask)?;
    validate_parallel_matrix_binary(protocol, &layout.zero, MatrixBinaryOp::Subtract)?;
    validate_parallel_matrix_binary(protocol, &layout.not, MatrixBinaryOp::Subtract)?;
    match &layout.product {
        FamilyProductRef::Direct(product) => {
            validate_parallel_matrix_binary(protocol, product, MatrixBinaryOp::Multiply)?;
        }
        FamilyProductRef::EncodingVector {
            left_times_right_decomposition,
            right_times_left_plaintext,
            sum,
        } => {
            validate_parallel_matrix_binary(
                protocol,
                left_times_right_decomposition,
                MatrixBinaryOp::Multiply,
            )?;
            validate_parallel_matrix_binary(
                protocol,
                right_times_left_plaintext,
                MatrixBinaryOp::Multiply,
            )?;
            validate_parallel_matrix_binary(protocol, sum, MatrixBinaryOp::Add)?;
            if sum.left_family.wire != left_times_right_decomposition.output_family ||
                sum.right_family.wire != right_times_left_plaintext.output_family
            {
                return Err(CertificateValidationError::LoopLayoutMismatch);
            }
        }
    }
    validate_parallel_matrix_binary(protocol, &layout.sum, MatrixBinaryOp::Add)?;
    validate_parallel_matrix_binary(protocol, &layout.two_product, MatrixBinaryOp::Multiply)?;
    validate_parallel_matrix_binary(protocol, &layout.xor, MatrixBinaryOp::Subtract)?;
    validate_parallel_six_way_select(protocol, &layout.candidate_select)?;
    validate_parallel_two_way_select(protocol, &layout.active_select)?;
    if layout.left_selection.source_family.wire != layout.state_input ||
        layout.right_selection.source_family.wire != layout.state_input ||
        layout.one_repetition.outputs != vec![layout.one_family.clone()] ||
        layout.active_mask.outputs != vec![layout.active_family.clone()] ||
        layout.zero.left_family.wire != layout.one_family ||
        layout.zero.right_family.wire != layout.one_family ||
        layout.not.left_family.wire != layout.one_family ||
        layout.not.right_family.wire != layout.copy_family ||
        layout.sum.left_family.wire != layout.copy_family ||
        layout.sum.right_family.wire != layout.right_selection.output_family ||
        layout.two_product.left_family.wire != *layout.product.output_family() ||
        layout.xor.left_family.wire != layout.sum.output_family ||
        layout.xor.right_family.wire != layout.two_product.output_family ||
        layout.candidate_select.selector_family.wire != layout.opcode_family ||
        layout.candidate_select.branch_families[0].wire != layout.zero.output_family ||
        layout.candidate_select.branch_families[1].wire != layout.one_family ||
        layout.candidate_select.branch_families[2].wire != layout.copy_family ||
        layout.candidate_select.branch_families[3].wire != layout.not.output_family ||
        layout.candidate_select.branch_families[4].wire != *layout.product.output_family() ||
        layout.candidate_select.branch_families[5].wire != layout.xor.output_family ||
        layout.active_select.selector_family.wire != layout.active_family ||
        layout.active_select.branch_families[0].wire !=
            layout.candidate_select.branch_families[0].wire ||
        layout.active_select.branch_families[1].wire != layout.candidate_select.output_family ||
        layout.active_select.output_family != layout.state_output
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_parallel_matrix_binary(
    protocol: &ProtocolDecl,
    layout: &ParallelMatrixBinaryRef,
    expected: MatrixBinaryOp,
) -> Result<(), CertificateValidationError> {
    validate_parallel_layout(
        protocol,
        &layout.parallel_loop,
        &layout.body_scope,
        &[&layout.left_family, &layout.right_family],
        &[&layout.body_left, &layout.body_right],
        &layout.body_output,
        &layout.output_family,
    )?;
    validate_matrix_binary(protocol, &layout.operation, expected)?;
    if layout.operation.left.wire != layout.body_left ||
        layout.operation.right.wire != layout.body_right ||
        layout.operation.output != layout.body_output
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_parallel_six_way_select(
    protocol: &ProtocolDecl,
    layout: &ParallelSixWaySelectRef,
) -> Result<(), CertificateValidationError> {
    let outer = std::iter::once(&layout.selector_family)
        .chain(layout.branch_families.iter())
        .collect::<Vec<_>>();
    let inner = std::iter::once(&layout.body_selector)
        .chain(layout.body_branches.iter())
        .collect::<Vec<_>>();
    validate_parallel_layout(
        protocol,
        &layout.parallel_loop,
        &layout.body_scope,
        &outer,
        &inner,
        &layout.body_output,
        &layout.output_family,
    )?;
    validate_six_way_select(protocol, &layout.select)?;
    if layout.select.selector.wire != layout.body_selector ||
        layout
            .select
            .branches
            .iter()
            .map(|operand| &operand.wire)
            .ne(layout.body_branches.iter()) ||
        layout.select.output != layout.body_output
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_parallel_two_way_select(
    protocol: &ProtocolDecl,
    layout: &ParallelTwoWaySelectRef,
) -> Result<(), CertificateValidationError> {
    let outer = std::iter::once(&layout.selector_family)
        .chain(layout.branch_families.iter())
        .collect::<Vec<_>>();
    let inner = std::iter::once(&layout.body_selector)
        .chain(layout.body_branches.iter())
        .collect::<Vec<_>>();
    validate_parallel_layout(
        protocol,
        &layout.parallel_loop,
        &layout.body_scope,
        &outer,
        &inner,
        &layout.body_output,
        &layout.output_family,
    )?;
    validate_two_way_select(protocol, &layout.select)?;
    if layout.select.selector.wire != layout.body_selector ||
        layout
            .select
            .branches
            .iter()
            .map(|operand| &operand.wire)
            .ne(layout.body_branches.iter()) ||
        layout.select.output != layout.body_output
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_parallel_layout(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    body_scope: &FrozenGraphScopeId,
    outer_inputs: &[&CoreOperandRef],
    body_inputs: &[&CoreWireRef],
    body_output: &CoreWireRef,
    output_family: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    if outer_inputs.len() != body_inputs.len() ||
        !matches!(resolve_node(protocol, operation)?.kind(), NodeKind::ParallelLoop(_))
    {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    let stage = stage(protocol, &operation.stage)?;
    if stage.graph.child_scope_id(&operation.scope, operation.node).as_ref() != Some(body_scope) {
        return Err(CertificateValidationError::LoopBodyMismatch);
    }
    let body = stage.graph.scope(body_scope).ok_or(CertificateValidationError::MissingScope)?;
    if body.inputs().len() != body_inputs.len() || body.outputs() != [body_output.as_wire_ref()] {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    for (index, (outer, inner)) in outer_inputs.iter().zip(body_inputs).enumerate() {
        validate_operand(protocol, outer)?;
        validate_wire(protocol, inner)?;
        if outer.node != *operation ||
            outer.operand as usize != index ||
            inner.node.stage != operation.stage ||
            inner.node.scope != *body_scope ||
            inner.as_wire_ref() != body.inputs()[index]
        {
            return Err(CertificateValidationError::LoopLayoutMismatch);
        }
    }
    validate_operation_output(protocol, operation, output_family)?;
    if output_family.port != Port(0) {
        return Err(CertificateValidationError::LoopLayoutMismatch);
    }
    Ok(())
}

fn validate_six_way_select(
    protocol: &ProtocolDecl,
    layout: &SixWaySelectRef,
) -> Result<(), CertificateValidationError> {
    validate_select(protocol, &layout.operation, &layout.selector, &layout.branches, &layout.output)
}

fn validate_two_way_select(
    protocol: &ProtocolDecl,
    layout: &TwoWaySelectRef,
) -> Result<(), CertificateValidationError> {
    validate_select(protocol, &layout.operation, &layout.selector, &layout.branches, &layout.output)
}

fn validate_select<const N: usize>(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    selector: &CoreOperandRef,
    branches: &[CoreOperandRef; N],
    output: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    if !matches!(resolve_node(protocol, operation)?.kind(), NodeKind::Select { count } if count == &IntExpr::constant(N))
    {
        return Err(CertificateValidationError::WrongNodeKind);
    }
    validate_operand(protocol, selector)?;
    if selector.node != *operation || selector.operand != 0 {
        return Err(CertificateValidationError::OperandMismatch);
    }
    for (index, branch) in branches.iter().enumerate() {
        validate_operand(protocol, branch)?;
        if branch.node != *operation || branch.operand as usize != index + 1 {
            return Err(CertificateValidationError::OperandMismatch);
        }
    }
    validate_operation_output(protocol, operation, output)
}

fn validate_decoder(
    protocol: &ProtocolDecl,
    workflow: &DiamondWorkflowLayout,
    boolean_layers: &BooleanLayersLayout,
    layout: &DecoderLayout,
) -> Result<(), CertificateValidationError> {
    for (operation, expected) in [
        (&layout.one_vector, MatrixBinaryOp::Multiply),
        (&layout.k_vector, MatrixBinaryOp::Multiply),
        (&layout.decoder_vector, MatrixBinaryOp::Multiply),
        (&layout.one_minus_circuit, MatrixBinaryOp::Subtract),
        (&layout.projected_difference, MatrixBinaryOp::Multiply),
        (&layout.k_plus_projection, MatrixBinaryOp::Add),
        (&layout.residual, MatrixBinaryOp::Subtract),
    ] {
        validate_matrix_binary(protocol, operation, expected)?;
        if operation.operation.stage != workflow.decryption.stage {
            return Err(CertificateValidationError::DecoderWiringMismatch);
        }
    }
    for artifact in
        [&layout.one_preimage, &layout.k_preimage, &layout.decoder_preimage, &layout.r_decomposed]
    {
        validate_wire(protocol, artifact)?;
        if !matches!(
            resolve_node(protocol, &artifact.node)?.kind(),
            NodeKind::Input { artifact: Some(_), .. }
        ) {
            return Err(CertificateValidationError::DecoderWiringMismatch);
        }
    }
    if layout.one_vector.right.wire != layout.one_preimage ||
        layout.k_vector.right.wire != layout.k_preimage ||
        layout.decoder_vector.right.wire != layout.decoder_preimage ||
        layout.one_vector.left.wire != layout.k_vector.left.wire ||
        layout.one_vector.left.wire != layout.decoder_vector.left.wire ||
        layout.selected_circuit_vector != boolean_layers.decryption.selected_vector.output ||
        layout.one_minus_circuit.left.wire != layout.one_vector.output ||
        layout.one_minus_circuit.right.wire != layout.selected_circuit_vector ||
        layout.projected_difference.left.wire != layout.one_minus_circuit.output ||
        layout.projected_difference.right.wire != layout.r_decomposed ||
        layout.k_plus_projection.left.wire != layout.k_vector.output ||
        layout.k_plus_projection.right.wire != layout.projected_difference.output ||
        layout.residual.left.wire != layout.decoder_vector.output ||
        layout.residual.right.wire != layout.k_plus_projection.output
    {
        return Err(CertificateValidationError::DecoderWiringMismatch);
    }
    validate_unary(protocol, &layout.extract_coefficient)?;
    validate_evaluate_int(protocol, &layout.threshold)?;
    validate_binary_node(protocol, &layout.lower_compare)?;
    validate_binary_node(protocol, &layout.upper_scale)?;
    validate_binary_node(protocol, &layout.upper_compare)?;
    validate_unary(protocol, &layout.lower_to_int)?;
    validate_unary(protocol, &layout.upper_to_int)?;
    validate_binary_node(protocol, &layout.comparison_sum)?;
    validate_binary_node(protocol, &layout.equals_two)?;
    if !matches!(resolve_node(protocol, &layout.extract_coefficient.operation)?.kind(),
            NodeKind::ExtractCoefficient { position } if position == &IntExpr::constant(0)) ||
        layout.threshold.expression !=
            IntExpr::RoundDiv(
                Box::new(IntExpr::Sub(
                    Box::new(IntExpr::Var("diamond_modulus".to_owned())),
                    Box::new(IntExpr::constant(2)),
                )),
                Box::new(IntExpr::constant(4)),
            ) ||
        layout.threshold.operation.stage != workflow.decryption.stage ||
        !matches!(
            resolve_node(protocol, &layout.lower_compare.operation)?.kind(),
            NodeKind::IntCompare(IntCompareOp::LessEqual)
        ) ||
        !matches!(
            resolve_node(protocol, &layout.upper_scale.operation)?.kind(),
            NodeKind::IntBinary(IntBinaryOp::Multiply)
        ) ||
        !matches!(
            resolve_node(protocol, &layout.upper_compare.operation)?.kind(),
            NodeKind::IntCompare(IntCompareOp::LessEqual)
        ) ||
        !matches!(
            resolve_node(protocol, &layout.lower_to_int.operation)?.kind(),
            NodeKind::BoolToInt
        ) ||
        !matches!(
            resolve_node(protocol, &layout.upper_to_int.operation)?.kind(),
            NodeKind::BoolToInt
        ) ||
        !matches!(
            resolve_node(protocol, &layout.comparison_sum.operation)?.kind(),
            NodeKind::IntBinary(IntBinaryOp::Add)
        ) ||
        !matches!(
            resolve_node(protocol, &layout.equals_two.operation)?.kind(),
            NodeKind::IntCompare(IntCompareOp::Equal)
        ) ||
        layout.extract_coefficient.input.wire != layout.residual.output ||
        layout.lower_compare.left.wire != layout.threshold.output ||
        layout.lower_compare.right.wire != layout.extract_coefficient.output ||
        layout.upper_compare.left.wire != layout.extract_coefficient.output ||
        layout.upper_scale.left.wire != layout.lower_compare.left.wire ||
        layout.upper_compare.right.wire != layout.upper_scale.output ||
        layout.lower_to_int.input.wire != layout.lower_compare.output ||
        layout.upper_to_int.input.wire != layout.upper_compare.output ||
        !binary_node_consumes(&layout.comparison_sum, &layout.lower_to_int.output) ||
        !binary_node_consumes(&layout.comparison_sum, &layout.upper_to_int.output) ||
        layout.equals_two.left.wire != layout.comparison_sum.output ||
        !is_constant_int(protocol, &layout.upper_scale.right.wire, 3)? ||
        !is_constant_int(protocol, &layout.equals_two.right.wire, 2)? ||
        layout.equals_two.output != layout.decoded
    {
        return Err(CertificateValidationError::DecodePathMismatch);
    }
    Ok(())
}

fn validate_unary(
    protocol: &ProtocolDecl,
    operation: &UnaryNodeRef,
) -> Result<(), CertificateValidationError> {
    validate_operand(protocol, &operation.input)?;
    if operation.input.node != operation.operation || operation.input.operand != 0 {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_operation_output(protocol, &operation.operation, &operation.output)
}

fn validate_binary_node(
    protocol: &ProtocolDecl,
    operation: &BinaryNodeRef,
) -> Result<(), CertificateValidationError> {
    validate_operand(protocol, &operation.left)?;
    validate_operand(protocol, &operation.right)?;
    if operation.left.node != operation.operation ||
        operation.left.operand != 0 ||
        operation.right.node != operation.operation ||
        operation.right.operand != 1
    {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_operation_output(protocol, &operation.operation, &operation.output)
}

fn binary_node_consumes(operation: &BinaryNodeRef, wire: &CoreWireRef) -> bool {
    operation.left.wire == *wire || operation.right.wire == *wire
}

fn is_constant_int(
    protocol: &ProtocolDecl,
    wire: &CoreWireRef,
    expected: i64,
) -> Result<bool, CertificateValidationError> {
    validate_wire(protocol, wire)?;
    Ok(
        matches!(resolve_node(protocol, &wire.node)?.kind(), NodeKind::ConstantInt(value) if value == &expected.into()),
    )
}

fn validate_matrix_binary(
    protocol: &ProtocolDecl,
    operation: &MatrixBinaryRef,
    expected: MatrixBinaryOp,
) -> Result<(), CertificateValidationError> {
    if !matches!(resolve_node(protocol, &operation.operation)?.kind(), NodeKind::MatrixBinary(actual) if *actual == expected)
    {
        return Err(CertificateValidationError::WrongNodeKind);
    }
    validate_operand(protocol, &operation.left)?;
    validate_operand(protocol, &operation.right)?;
    if operation.left.node != operation.operation ||
        operation.left.operand != 0 ||
        operation.right.node != operation.operation ||
        operation.right.operand != 1
    {
        return Err(CertificateValidationError::OperandMismatch);
    }
    validate_operation_output(protocol, &operation.operation, &operation.output)
}

fn protocol_input_name(
    protocol: &ProtocolDecl,
    wire: &CoreWireRef,
) -> Result<String, CertificateValidationError> {
    let node = resolve_node(protocol, &wire.node)?;
    let name = match node.kind() {
        NodeKind::Input { name, artifact: None, .. } => name,
        NodeKind::FamilyGetDynamic => {
            let stage = stage(protocol, &wire.node.stage)?;
            let scope = stage
                .graph
                .scope(&wire.node.scope)
                .ok_or(CertificateValidationError::MissingScope)?;
            let arguments =
                scope.arguments(node).ok_or(CertificateValidationError::CircuitInputMismatch)?;
            let source =
                arguments.first().ok_or(CertificateValidationError::CircuitInputMismatch)?;
            return protocol_input_name(
                protocol,
                &CoreWireRef {
                    node: CoreNodeRef::new(
                        wire.node.stage.clone(),
                        wire.node.scope.clone(),
                        source.node,
                    ),
                    port: source.port,
                },
            );
        }
        _ => return Err(CertificateValidationError::CircuitInputMismatch),
    };
    protocol
        .correctness
        .protocol_inputs
        .iter()
        .find_map(|(protocol_name, destinations)| {
            destinations
                .iter()
                .any(|(stage, input)| stage == &wire.node.stage && input.0 == *name)
                .then(|| protocol_name.0.clone())
        })
        .ok_or(CertificateValidationError::CircuitInputMismatch)
}

fn input_name(
    protocol: &ProtocolDecl,
    wire: &CoreWireRef,
) -> Result<String, CertificateValidationError> {
    let node = resolve_node(protocol, &wire.node)?;
    match node.kind() {
        NodeKind::Input { name, artifact: None, .. } if wire.port == Port(0) => Ok(name.clone()),
        _ => Err(CertificateValidationError::LayerMetadataMismatch),
    }
}

fn stage<'a>(
    protocol: &'a ProtocolDecl,
    stage: &StageId,
) -> Result<&'a crate::ProtocolStage, CertificateValidationError> {
    protocol
        .stages
        .iter()
        .find(|candidate| &candidate.id == stage)
        .ok_or_else(|| CertificateValidationError::MissingStage(stage.0.clone()))
}

fn resolve_node<'a>(
    protocol: &'a ProtocolDecl,
    reference: &CoreNodeRef,
) -> Result<&'a mxx_ir_core::NodeHandle, CertificateValidationError> {
    let stage = stage(protocol, &reference.stage)?;
    let scope =
        stage.graph.scope(&reference.scope).ok_or(CertificateValidationError::MissingScope)?;
    scope.node(reference.node).ok_or(CertificateValidationError::MissingNode(reference.node))
}

fn validate_wire(
    protocol: &ProtocolDecl,
    reference: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    let node = resolve_node(protocol, &reference.node)?;
    if reference.port.0 as usize >= node.output_types().len() {
        return Err(CertificateValidationError::MissingPort {
            node: reference.node.node,
            port: reference.port,
        });
    }
    Ok(())
}

fn validate_operand(
    protocol: &ProtocolDecl,
    reference: &CoreOperandRef,
) -> Result<(), CertificateValidationError> {
    validate_wire(protocol, &reference.wire)?;
    let stage = stage(protocol, &reference.node.stage)?;
    let scope =
        stage.graph.scope(&reference.node.scope).ok_or(CertificateValidationError::MissingScope)?;
    let node = scope
        .node(reference.node.node)
        .ok_or(CertificateValidationError::MissingNode(reference.node.node))?;
    let arguments = scope.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    let actual = arguments.get(reference.operand as usize).ok_or(
        CertificateValidationError::MissingOperand {
            node: reference.node.node,
            operand: reference.operand,
        },
    )?;
    if reference.node.stage != reference.wire.node.stage ||
        reference.node.scope != reference.wire.node.scope ||
        *actual != reference.wire.as_wire_ref()
    {
        return Err(CertificateValidationError::OperandMismatch);
    }
    Ok(())
}

fn validate_parameter(
    operation: &CoreNodeRef,
    parameter: &CoreNodeParameterRef,
    expected: CoreNodeParameter,
) -> Result<(), CertificateValidationError> {
    if parameter.node != *operation || parameter.parameter != expected {
        return Err(CertificateValidationError::MissingNodeParameter);
    }
    Ok(())
}

fn validate_operation_output(
    protocol: &ProtocolDecl,
    operation: &CoreNodeRef,
    output: &CoreWireRef,
) -> Result<(), CertificateValidationError> {
    validate_wire(protocol, output)?;
    if output.node != *operation {
        return Err(CertificateValidationError::OutputMismatch);
    }
    Ok(())
}

fn is_decomposition_materialization(
    protocol: &ProtocolDecl,
    decomposition: &CoreWireRef,
    value: &CoreWireRef,
) -> Result<bool, CertificateValidationError> {
    if value == decomposition {
        return Ok(true);
    }
    if value.node.stage != decomposition.node.stage ||
        value.node.scope != decomposition.node.scope ||
        value.port != Port(0)
    {
        return Ok(false);
    }
    let node = resolve_node(protocol, &value.node)?;
    if !matches!(node.kind(), NodeKind::MatrixScale { scalar } if scalar == &IntExpr::constant(1)) {
        return Ok(false);
    }
    let stage = stage(protocol, &value.node.stage)?;
    let scope =
        stage.graph.scope(&value.node.scope).ok_or(CertificateValidationError::MissingScope)?;
    let arguments = scope.arguments(node).ok_or(CertificateValidationError::OperandMismatch)?;
    Ok(arguments.as_slice() == [decomposition.as_wire_ref()])
}

fn scope_is_within(scope: &FrozenGraphScopeId, ancestor: &FrozenGraphScopeId) -> bool {
    if scope == ancestor {
        return true;
    }
    match scope {
        FrozenGraphScopeId::ParallelBody { parent, .. } |
        FrozenGraphScopeId::SequentialBody { parent, .. } => scope_is_within(parent, ancestor),
        FrozenGraphScopeId::Root | FrozenGraphScopeId::Subgraph { .. } => false,
    }
}

impl CoreWireRef {
    fn as_wire_ref(&self) -> WireRef {
        WireRef { node: self.node.node, port: self.port }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};

    #[test]
    fn semantic_certificate_none_round_trips_and_hashes() {
        let certificate = SemanticCertificate::None;
        let encoded = serde_json::to_vec(&certificate).expect("serialize");
        let decoded: SemanticCertificate = serde_json::from_slice(&encoded).expect("deserialize");
        assert_eq!(certificate, decoded);
        assert_eq!(certificate.sha256().unwrap(), decoded.sha256().unwrap());
    }

    #[test]
    fn local_decomposition_commits_to_exact_key_and_parameters() {
        let node =
            CoreNodeRef::new(StageId("decrypt".to_owned()), FrozenGraphScopeId::Root, NodeId(7));
        let key = node.wire(3);
        let decomposition = LocalGadgetDecompositionRef::new(node.clone(), key.clone());
        assert_eq!(decomposition.right_public_key, node.operand(0, key));
        assert_eq!(decomposition.base, node.parameter(CoreNodeParameter::GadgetDecomposeBase));
        assert_eq!(
            decomposition.digit_count,
            node.parameter(CoreNodeParameter::GadgetDecomposeDigitCount)
        );
    }

    #[test]
    fn deterministic_decomposition_materialization_is_not_protocol_data() {
        let ring = Ring::new(257, 8);
        let right = ring.input("right-public-key", (1, 2));
        let decomposition = right.decompose(2, 2).as_mat();
        let graph = DslContext::new("local-decomposition")
            .output("result", ring.input("left-public-key", (1, 2)) * decomposition)
            .expect("output")
            .build()
            .expect("graph")
            .graph;
        assert!(graph.outputs().keys().all(|name| !name.contains("decomposition")));
    }
}
