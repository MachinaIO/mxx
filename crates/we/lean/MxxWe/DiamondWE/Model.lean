import MxxIrCore.Program
import MxxIrCore.Structural
import MxxIrCore.Eval
import MxxIrCore.NodeEquation
import MxxIrCore.Sampler
import MxxGadgets.BooleanCircuit
import MxxBgg.Trace
import MxxRuntime
import MxxWe.DiamondWE.Parameters

namespace Mxx.We.DiamondWE

open Mxx.IR
open Mxx.Primitives
open Mxx.Gadgets

noncomputable abbrev RuntimeBackend (oracle : Mxx.Runtime.RuntimeGadgetOracle) :=
  Mxx.Runtime.irBackendWithGadgetOracle oracle
abbrev RuntimeValue (oracle : Mxx.Runtime.RuntimeGadgetOracle) := Value (RuntimeBackend oracle)
abbrev RuntimeDynamicValue (oracle : Mxx.Runtime.RuntimeGadgetOracle) := DynamicValue (RuntimeBackend oracle)
abbrev RuntimeTrace (oracle : Mxx.Runtime.RuntimeGadgetOracle) := Trace (RuntimeBackend oracle)
abbrev RuntimeEvalEnv (oracle : Mxx.Runtime.RuntimeGadgetOracle) (data : ProgramData) :=
  EvalEnv (RuntimeBackend oracle) data
abbrev RuntimeMatrixValue (matrixType : MatrixType) :=
  ExactMatrix matrixType.modulus.toNat matrixType.ringDimension matrixType.rows matrixType.columns

@[simp] theorem runtimeValue_matrix (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (matrixType : MatrixType) :
    RuntimeValue oracle (.matrix matrixType) = RuntimeMatrixValue matrixType := rfl

/- A typed reference is local to one validated program.  The proof field is intentionally based on
   the program's declared output type; callers cannot provide a reference from another program or
   silently cast a wire to a different type. -/
def wireTypeAt (program : Mxx.IR.Program) (stage : Nat) (wire : WireRef) : Option WireType :=
  program.data.stages[stage]?.bind (fun value => value.wireType? wire)

structure TypedWireRef (program : Mxx.IR.Program) (wireType : WireType) where
  stage : Nat
  stage_valid : stage < program.data.stages.size
  wire : WireRef
  type_correct : wireTypeAt program stage wire = some wireType

def externalInputIndexAt (program : Mxx.IR.Program) (stage : Nat)
    (wire : WireRef) : Option Nat := do
  let stageData ← program.data.stages[stage]?
  if wire.scope = stageData.root then
    let scope ← scopeAt stageData stageData.root
    let node ← nodeAt scope wire.node
    match node.payload with
    | .input index => some index
    | _ => none
  else none

structure TypedExternalInputRef (program : Mxx.IR.Program) (wireType : WireType) where
  reference : TypedWireRef program wireType
  inputIndex : Nat
  input_stored :
    externalInputIndexAt program reference.stage reference.wire = some inputIndex

abbrev AnyTypedWireRef (program : Mxx.IR.Program) :=
  Sigma (fun wireType => TypedWireRef program wireType)

structure BooleanCircuitInputRefs (program : Mxx.IR.Program)
    (shape : LayeredBoolCircuitShape) where
  activeGateCountsInput : TypedExternalInputRef program (.family [shape.depth] .int)
  circuitGateKindsInput :
    TypedExternalInputRef program (.family [shape.depth * shape.maxLayerWidth] .int)
  circuitLeftSourcesInput :
    TypedExternalInputRef program (.family [shape.depth * shape.maxLayerWidth] .int)
  circuitRightSourcesInput :
    TypedExternalInputRef program (.family [shape.depth * shape.maxLayerWidth] .int)
  circuitOutputSourceInput : TypedExternalInputRef program (.family [1] .int)

structure Refs (program : Mxx.IR.Program)
    (circuitShape : LayeredBoolCircuitShape)
    (plaintextMatrixType : MatrixType) where
  messageInput : TypedExternalInputRef program .bool
  encryptionInstanceBitsInput :
    TypedExternalInputRef program (.family [circuitShape.maxLayerWidth] .int)
  decryptionInstanceBitsInput :
    TypedExternalInputRef program (.family [circuitShape.maxLayerWidth] .int)
  witnessBitsInput : TypedExternalInputRef program (.family [circuitShape.maxLayerWidth] .int)
  encryptionCircuit : BooleanCircuitInputRefs program circuitShape
  decryptionCircuit : BooleanCircuitInputRefs program circuitShape
  encryptionCircuitOutput : AnyTypedWireRef program
  decryptionCircuitOutput : AnyTypedWireRef program
  noisyPlaintextOutput : TypedWireRef program (.matrix plaintextMatrixType)
  decodedOutput : TypedWireRef program .bool

structure Candidate where
  program : Mxx.IR.Program
  parameters : Parameters
  circuitShape : LayeredBoolCircuitShape
  plaintextMatrixType : MatrixType
  refs : Refs program circuitShape plaintextMatrixType
  bound : BoundData

/- Runtime DCRT representation is deliberately separate from the semantic IR
   candidate.  The IR theorem does not assume that a backend's CRT layout is
   a faithful refinement until this record is supplied. -/
structure RuntimeDcrtRepresentation where
  moduli : Array Nat
  baseBits : Nat
  actualModulusBits : Array Nat
  nonempty : moduli.size > 0
  pairwiseCoprime : ∀ i j, i < moduli.size → j < moduli.size → i ≠ j →
    Nat.Coprime (moduli[i]!) (moduli[j]!)

def RuntimeDcrtRepresentation.depth (representation : RuntimeDcrtRepresentation) : Nat :=
  representation.moduli.size

def RuntimeDcrtRepresentation.modulus (representation : RuntimeDcrtRepresentation) : Nat :=
  representation.moduli.foldl (· * ·) 1

def RuntimeDcrtRepresentation.validFor (representation : RuntimeDcrtRepresentation)
    (parameters : Parameters) : Prop :=
  representation.baseBits = parameters.gadgetBase ∧
    representation.modulus > parameters.modulus ∧
    representation.actualModulusBits.size = representation.depth


/- A site is a frozen IR wire together with the operation class and the exact
   argument/output arrays stored by that node.  The arrays are copied from the
   generated program at the boundary; they are not equations supplied by a
   correctness caller. -/
inductive DiamondSiteKind
  | input
  | artifactInput
  | matrixMultiply
  | matrixAddSub
  | applyPreimage
  | preimageSample
  | familyPreimageSample
  | gadgetDecompose
  | sequentialLoop
  | parallelGrid
  | select
  | familyOperation
  | coefficientExtraction
  | thresholdDecode
  | other
  deriving Repr, DecidableEq

def _root_.Mxx.IR.NodePayload.siteKind : Mxx.IR.NodePayload → Mxx.We.DiamondWE.DiamondSiteKind
  | .input _ => .input
  | .artifactInput _ => .artifactInput
  | .matrixBinary .multiply => .matrixMultiply
  | .matrixBinary .add | .matrixBinary .subtract => .matrixAddSub
  | .applyPreimage => .applyPreimage
  | .preimageSample _ _ => .preimageSample
  | .familyPreimageSample _ _ => .familyPreimageSample
  | .gadgetDecompose _ _ _ => .gadgetDecompose
  | .sequentialLoop _ => .sequentialLoop
  | .parallelGrid _ => .parallelGrid
  | .select _ => .select
  | .familyPack _ | .familyGetStatic _ | .familyGetDynamic _ |
      .familySelectAxis _ | .familyReindex _ _ | .familyGather _ _ => .familyOperation
  | .extractCoefficient _ _ => .coefficientExtraction
  | .thresholdDecode _ _ _ => .thresholdDecode
  | _ => .other


example : Mxx.IR.NodePayload.siteKind (.applyPreimage) = DiamondSiteKind.applyPreimage := by rfl

example : Mxx.IR.NodePayload.siteKind (.matrixBinary .multiply) =
    DiamondSiteKind.matrixMultiply := by rfl

def nodeAtReference {program : Program} (reference : AnyTypedWireRef program) : Option Node := do
  let typed := reference.2
  let stage ← program.data.stages[typed.stage]?
  let scope ← scopeAt stage typed.wire.scope
  nodeAt scope typed.wire.node

def nodeKindAt {program : Program} (reference : AnyTypedWireRef program) : Option DiamondSiteKind :=
  (nodeAtReference reference).map (fun node => _root_.Mxx.IR.NodePayload.siteKind node.payload)

def nodePayloadAt {program : Program} (reference : AnyTypedWireRef program) : Option NodePayload :=
  (nodeAtReference reference).map Node.payload

def nodeArgumentsAt {program : Program} (reference : AnyTypedWireRef program) : Option (Array WireRef) :=
  (nodeAtReference reference).map Node.arguments

def nodeOutputsAt {program : Program} (reference : AnyTypedWireRef program) : Option (Array WireType) :=
  (nodeAtReference reference).map Node.outputs

structure StoredNodeRef (program : Program) where
  reference : AnyTypedWireRef program
  stage : Nat
  scope : Nat
  payload : NodePayload
  kind : DiamondSiteKind
  arguments : Array WireRef
  outputs : Array WireType
  payload_stored : nodePayloadAt reference = some payload
  kind_stored : nodeKindAt reference = some kind
  arguments_stored : nodeArgumentsAt reference = some arguments
  outputs_stored : nodeOutputsAt reference = some outputs
  ownership_stored : reference.2.stage = stage ∧ reference.2.wire.scope = scope

/- A named edge is checked against the consumer's concrete argument array.  It
   cannot be satisfied merely by matching operation classes or node numbers. -/
structure ArgumentWireEdge {program : Program} (consumer producer : StoredNodeRef program) (argument : Nat) : Prop where
  exact_argument : consumer.arguments[argument]? = some producer.reference.2.wire

def hasArgumentWireEdge {program : Program} (consumer producer : StoredNodeRef program) (argument : Nat) : Prop :=
  consumer.arguments[argument]? = some producer.reference.2.wire

def isCoefficientAtZero (payload : NodePayload) : Prop :=
  match payload with
  | .extractCoefficient position upper => position = .literal 0 ∧ upper = none
  | _ => False

def isIntegerComparison (payload : NodePayload) : Prop :=
  match payload with
  | .intCompare _ => True
  | _ => False

def isBoolToInt (payload : NodePayload) : Prop := payload = .boolToInt

def isIntegerAdd (payload : NodePayload) : Prop := payload = .intBinary .add
def isLessEqualComparison (payload : NodePayload) : Prop := payload = .intCompare .lessEqual
def isEqualComparison (payload : NodePayload) : Prop := payload = .intCompare .equal
def isIntegerMultiply (payload : NodePayload) : Prop := payload = .intBinary .multiply

def isFamilyGetDynamicZero (payload : NodePayload) : Prop :=
  payload = .familyGetDynamic 0

def quarterPayload (modulus : Nat) : NodePayload :=
  .evaluateInt (.roundDivide (.subtract (.literal modulus) (.literal 2)) (.literal 4))

def isQuarterPayload (payload : NodePayload) (modulus : Nat) : Prop :=
  payload = quarterPayload modulus

def loopCarries (payload : NodePayload) (arity : Nat) : Prop :=
  match payload with
  | .sequentialLoop loop => loop.carriedCount = arity ∧ loop.count ≠ .literal 0
  | _ => False

def injectorLoopExact (payload : NodePayload) (inputCount : Nat) (arity : Nat) : Prop :=
  match payload with
  | .sequentialLoop loop =>
      loop.count = .literal inputCount ∧ loop.indexSlot = 0 ∧
        loop.carriedCount = arity ∧ loop.bindings.size = 0
  | _ => False

abbrev BggTraceRole := Mxx.Bgg.TraceRole
abbrev BggTraceLane := Mxx.Bgg.TraceLane
abbrev BggTraceSubrole := Mxx.Bgg.TraceSubrole
abbrev BggTraceStep := Mxx.Bgg.TraceStep
abbrev BggOperandSource := Mxx.Bgg.OperandSource
abbrev BggOperandSourceDescriptor := Mxx.Bgg.OperandSourceDescriptor

structure BggExpectedTraceEntry where
  lane : BggTraceLane
  subrole : BggTraceSubrole
  role : BggTraceRole
  operands : Nat
  hasLayer : Bool
  hasGateSlot : Bool
  deriving Repr, DecidableEq

def expectedBggLayerTrace : Array BggExpectedTraceEntry := #[
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .decompose, .decomposition, 1, true, false⟩,
  ⟨.publicKey, .materializeExact, .materializePreimageExact, 1, true, false⟩,
  ⟨.publicKey, .multiply, .matrixMultiply, 2, true, false⟩,
  ⟨.vector, .decompose, .decomposition, 1, true, false⟩,
  ⟨.vector, .applyPreimage, .applyPreimage, 2, true, false⟩,
  ⟨.vector, .multiply, .matrixMultiply, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.plaintext, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.publicKey, .gateOutput, .gateOutput, 2, true, false⟩,
  ⟨.vector, .select, .candidateSelect, 7, true, true⟩,
  ⟨.publicKey, .select, .candidateSelect, 7, true, true⟩,
  ⟨.plaintext, .select, .candidateSelect, 7, true, true⟩,
  ⟨.vector, .select, .activeSelect, 3, true, true⟩,
  ⟨.publicKey, .select, .activeSelect, 3, true, true⟩,
  ⟨.plaintext, .select, .activeSelect, 3, true, true⟩
]

def expectedBggEpilogue : BggExpectedTraceEntry :=
  ⟨.vector, .gateOutput, .gateOutput, 0, false, false⟩

def expectedBggSteps : Array BggTraceStep := #[
  .zeroPlaintext, .zeroVector, .zeroPublicKey,
  .notPlaintext, .notVector, .notPublicKey,
  .productPublicKeyDecompose, .productPublicKeyMaterialize,
  .productPublicKeyMultiply, .productVectorDecompose,
  .productVectorApplyPreimage, .productVectorMultiply,
  .productVectorOutput, .productPlaintextOutput,
  .sumPlaintext, .sumVector, .sumPublicKey,
  .twoProductPublicKey, .twoProductVector, .twoProductPlaintext,
  .xorPlaintext, .xorVector, .xorPublicKey,
  .candidateVectorSelect, .candidatePublicKeySelect, .candidatePlaintextSelect,
  .activeVectorSelect, .activePublicKeySelect, .activePlaintextSelect, .layerOutput]

def expectedBggOperandDescriptors : Array (Array BggOperandSourceDescriptor) := #[
  #[.external .one, .external .one],
  #[.external .one, .external .one],
  #[.external .one, .external .one],
  #[.external .one, .external .left],
  #[.external .one, .external .left],
  #[.external .one, .external .left],
  #[.external .right],
  #[.prior .productPublicKeyDecompose],
  #[.external .left, .prior .productPublicKeyMaterialize],
  #[.external .right],
  #[.external .left, .prior .productVectorDecompose],
  #[.external .left, .external .right],
  #[.external .left, .external .right],
  #[.external .left, .external .right],
  #[.external .left, .external .right],
  #[.external .left, .external .right],
  #[.external .left, .external .right],
  #[.external .left, .external .scalar],
  #[.external .left, .external .scalar],
  #[.external .left, .external .scalar],
  #[.prior .sumPlaintext, .prior .twoProductPlaintext],
  #[.prior .sumVector, .prior .twoProductVector],
  #[.prior .sumPublicKey, .prior .twoProductPublicKey],
  #[.external .selector, .prior .zeroVector, .external .one, .external .left,
    .prior .notVector, .prior .productVectorOutput, .prior .xorVector],
  #[.external .selector, .prior .zeroPublicKey, .external .one, .external .left,
    .prior .notPublicKey, .prior .productPublicKeyMultiply, .prior .xorPublicKey],
  #[.external .selector, .prior .zeroPlaintext, .external .one, .external .left,
    .prior .notPlaintext, .prior .productPlaintextOutput, .prior .xorPlaintext],
  #[.external .active, .prior .zeroVector, .prior .candidateVectorSelect],
  #[.external .active, .prior .zeroPublicKey, .prior .candidatePublicKeySelect],
  #[.external .active, .prior .zeroPlaintext, .prior .candidatePlaintextSelect],
  #[]]

structure TaggedBggSite (program : Program) where
  site : StoredNodeRef program
  step : BggTraceStep
  role : BggTraceRole
  lane : BggTraceLane
  subrole : BggTraceSubrole
  layer : Option StructuralIntExpr
  gateSlot : Option StructuralIntExpr
  candidate : Option StructuralIntExpr
  operands : Array WireRef
  operandSources : Array BggOperandSource
  arguments_eq : site.arguments = operands

inductive InjectorTraceRole
  | packedInputDigits | sourceStateReindex | selectedTransition | bodyApplyPreimage
  | transitionReindex | carriedPreviousState | nextStateBodyOutput
  deriving Repr, DecidableEq, Inhabited

structure TaggedInjectorSite (program : Program) where
  site : StoredNodeRef program
  role : InjectorTraceRole
  coordinate : StructuralIntExpr
  operands : Array WireRef
  arguments_eq : site.arguments = operands

def injectorRoleKindMatches {program : Program} (entry : TaggedInjectorSite program) : Prop :=
  match entry.role with
  | .packedInputDigits | .sourceStateReindex | .transitionReindex | .selectedTransition =>
      entry.site.kind = .familyOperation
  | .bodyApplyPreimage => entry.site.kind = .applyPreimage
  | .carriedPreviousState => True
  | .nextStateBodyOutput => entry.site.kind = .parallelGrid

def injectorRolePayloadMatches {program : Program} (entry : TaggedInjectorSite program) : Prop :=
  match entry.role, entry.site.payload with
  | .packedInputDigits, .familyGetDynamic 0 => True
  | .sourceStateReindex, .familyReindex _ _ => True
  | .transitionReindex, .familyReindex _ _ => True
  | .selectedTransition, .familySelectAxis 1 => True
  | .bodyApplyPreimage, .applyPreimage => True
  | .carriedPreviousState, _ => True
  | .nextStateBodyOutput, .parallelGrid _ => True
  | _, _ => False

def sourceStateReindexMap (batchBits : Nat) : IndexMap :=
  { sourceRank := 1, outputRank := 1,
    inputIndices := #[.select
      (.lessEqual (.add (.mul (.structuralSlot 0) (.literal batchBits)) (.literal 1)) (.axis 0))
      #[.axis 0, .literal 0]] }

def transitionReindexMap : IndexMap :=
  { sourceRank := 3, outputRank := 2,
    inputIndices := #[.structuralSlot 0, .axis 0, .axis 1] }

def injectorPayloadIsSourceReindex (payload : NodePayload) (batchBits stateCount : Nat) : Prop :=
  match payload with
  | .familyReindex shape map => shape = #[.literal stateCount] ∧ map = sourceStateReindexMap batchBits
  | _ => False

def injectorPayloadIsTransitionReindex (payload : NodePayload) (stateCount digitBase : Nat) : Prop :=
  match payload with
  | .familyReindex shape map => shape = #[.literal stateCount, .literal digitBase] ∧ map = transitionReindexMap
  | _ => False

def injectorAxisZeroInputMode : GridInputMode :=
  { reindex := true
    map := some {
      sourceRank := 1
      outputRank := 1
      inputIndices := #[.axis 0] } }

def nextStateGridPayloadIsExact (payload : NodePayload) (bodyScope stateCount : Nat) : Prop :=
  match payload with
  | .parallelGrid grid =>
      grid.child = bodyScope ∧ grid.shape = #[.literal stateCount] ∧ grid.indexSlots = #[0] ∧
        grid.bindings = #[] ∧ grid.inputModes = #[injectorAxisZeroInputMode,
          injectorAxisZeroInputMode]
  | _ => False

def declaredGridChildOutput {program : Program} (site : StoredNodeRef program) : Option WireRef := do
  let stage ← program.data.stages[site.stage]?
  let payload ← match site.payload with
    | .parallelGrid grid => some grid
    | _ => none
  let child ← scopeAt stage payload.child
  child.outputs[0]?

def declaredGridChildInput {program : Program} (site : StoredNodeRef program)
    (index : Nat) : Option WireRef := do
  let stage ← program.data.stages[site.stage]?
  let payload ← match site.payload with
    | .parallelGrid grid => some grid
    | _ => none
  let child ← scopeAt stage payload.child
  child.inputs[index]?

def injectorTraceRoles : Array InjectorTraceRole := #[
  .packedInputDigits, .sourceStateReindex, .transitionReindex, .selectedTransition,
  .bodyApplyPreimage,
  .carriedPreviousState, .nextStateBodyOutput
]

inductive InjectorTargetTraceRole
  | selector | selectorProduct | gaussianError | targetAdd
  deriving Repr, DecidableEq, Inhabited

def injectorTargetTraceRoles : Array InjectorTargetTraceRole := #[
  .selector, .selectorProduct, .gaussianError, .targetAdd
]

inductive SelectorMagnitudeTraceRole
  | digitSecretSample | selectedSecret | regularDiagonal | identity | kDiagonal
  | initialSelect | bitZero | bitIdentity | bitValueSelect | secretTimesBitValue
  | specialTop | specialBottom | specialConcat | carriedVsSpecialSelect
  deriving Repr, DecidableEq, Inhabited

def selectorMagnitudeTraceRoles : Array SelectorMagnitudeTraceRole := #[
  .digitSecretSample, .selectedSecret, .regularDiagonal, .identity, .kDiagonal,
  .initialSelect, .bitZero, .bitIdentity, .bitValueSelect, .secretTimesBitValue,
  .specialTop, .specialBottom, .specialConcat, .carriedVsSpecialSelect
]

structure TaggedSelectorMagnitudeSite (program : Program) where
  site : StoredNodeRef program
  role : SelectorMagnitudeTraceRole
  /- Only matrix operands are retained. Select controls are intentionally omitted because a
     maximum-of-branches magnitude proof is independent of the chosen Boolean branch. -/
  operands : Array WireRef

def selectorMagnitudeRolePayloadMatches {program : Program}
    (entry : TaggedSelectorMagnitudeSite program) : Prop :=
  match entry.role, entry.site.payload with
  | .digitSecretSample, .uniformIntervalSample _ range =>
      range.start = .literal (-1) ∧ range.stop = .literal 1
  | .selectedSecret, .familyGetDynamic 1 => True
  | .regularDiagonal, .concat .diagonal => True
  | .identity, .constantMatrix _ .identity => True
  | .kDiagonal, .concat .diagonal => True
  | .initialSelect, .select (.literal 2) => True
  | .bitZero, .constantMatrix _ .zero => True
  | .bitIdentity, .constantMatrix _ .identity => True
  | .bitValueSelect, .select (.literal 2) => True
  | .secretTimesBitValue, .matrixBinary .multiply => True
  | .specialTop, .concat .columns => True
  | .specialBottom, .constantMatrix _ .zero => True
  | .specialConcat, .concat .rows => True
  | .carriedVsSpecialSelect, .select (.literal 2) => True
  | _, _ => False

def selectorMagnitudeTraceMatches {program : Program}
    (entries : Array (TaggedSelectorMagnitudeSite program)) : Prop :=
  entries.size = selectorMagnitudeTraceRoles.size ∧
    ∀ index, index < selectorMagnitudeTraceRoles.size →
      ∃ entry, entries[index]? = some entry ∧
        entry.role = selectorMagnitudeTraceRoles[index]! ∧
        selectorMagnitudeRolePayloadMatches entry

structure TaggedInjectorTargetSite (program : Program) where
  site : StoredNodeRef program
  role : InjectorTargetTraceRole
  operands : Array WireRef
  arguments_eq : site.arguments = operands

def injectorTargetRolePayloadMatches {program : Program}
    (entry : TaggedInjectorTargetSite program) (batchBits errorBound : Nat) : Prop :=
  match entry.role, entry.site.payload with
  | .selector, .sequentialLoop loop =>
      loop.count = .literal batchBits ∧ loop.carriedCount = 1 ∧ loop.bindings = #[]
  | .selectorProduct, .matrixBinary .multiply => True
  | .gaussianError, .gaussianSample _ _ (.literal bound) => bound = errorBound
  | .targetAdd, .matrixBinary .add => True
  | _, _ => False

def injectorTargetTraceMatches {program : Program}
    (entries : Array (TaggedInjectorTargetSite program)) (batchBits errorBound : Nat) : Prop :=
  entries.size = injectorTargetTraceRoles.size ∧
    ∀ index, index < injectorTargetTraceRoles.size →
      ∃ entry, entries[index]? = some entry ∧ entry.role = injectorTargetTraceRoles[index]! ∧
        injectorTargetRolePayloadMatches entry batchBits errorBound

def injectorTargetReindexMap (stateCount digitBase : Nat) : IndexMap := {
  sourceRank := 1
  outputRank := 3
  inputIndices := #[.add
    (.mul (.add (.mul (.axis 0) (.literal stateCount)) (.axis 1)) (.literal digitBase))
    (.axis 2)]
}

def injectorTargetReindexPayloadIsExact
    (payload : NodePayload) (inputCount stateCount digitBase : Nat) : Prop :=
  match payload with
  | .familyReindex shape map =>
      shape = #[.literal inputCount, .literal stateCount, .literal digitBase] ∧
        map = injectorTargetReindexMap stateCount digitBase
  | _ => False

def injectorTargetGridInputMode : GridInputMode := {
  reindex := true
  map := some {
    sourceRank := 1
    outputRank := 1
    inputIndices := #[.axis 0]
  }
}

def injectorTargetGridPayloadIsExact
    (payload : NodePayload) (inputCount stateCount digitBase : Nat) : Prop :=
  match payload with
  | .parallelGrid grid =>
      grid.shape = #[.literal (inputCount * stateCount * digitBase)] ∧
        grid.indexSlots.size = 1 ∧ grid.bindings = #[] ∧
        grid.inputModes[0]? = some injectorTargetGridInputMode
  | _ => False

def injectorTraceMatches {program : Program} (entries : Array (TaggedInjectorSite program)) : Prop :=
  entries.size = injectorTraceRoles.size ∧
    ∀ index, index < injectorTraceRoles.size →
      ∃ entry, entries[index]? = some entry ∧ entry.role = injectorTraceRoles[index]! ∧
        injectorRoleKindMatches entry ∧ injectorRolePayloadMatches entry

def bggSourceDescriptors (sources : Array BggOperandSource) : Array BggOperandSourceDescriptor :=
  sources.map Mxx.Bgg.OperandSource.descriptor

def bggTraceEntryMatches {program : Program} (index : Nat)
    (entry : TaggedBggSite program) (expected : BggExpectedTraceEntry) : Prop :=
  entry.lane = expected.lane ∧ entry.subrole = expected.subrole ∧ entry.role = expected.role ∧
    entry.operands.size = expected.operands ∧ entry.layer.isSome = expected.hasLayer ∧
    entry.gateSlot.isSome = expected.hasGateSlot ∧ entry.candidate.isNone ∧
    entry.step = expectedBggSteps[index]! ∧
    bggSourceDescriptors entry.operandSources = expectedBggOperandDescriptors[index]!

def findBggEntryWireBefore? {program : Program} :
    Nat → List (TaggedBggSite program) → BggTraceStep → Option WireRef
  | 0, _, _ => none
  | _, [], _ => none
  | limit + 1, entry :: rest, step =>
      if entry.step = step then some entry.site.reference.2.wire
      else findBggEntryWireBefore? limit rest step

def validateBggEntrySources {program : Program} (stage : Stage)
    (entry : TaggedBggSite program)
    (prior : BggTraceStep → Option WireRef) : Prop :=
  Mxx.Bgg.OperandSource.valid stage
    { step := entry.step, operands := entry.operands, sources := entry.operandSources }
    prior

def bggTraceSourcesMatch {program : Program}
    (entries : Array (TaggedBggSite program)) (index : Nat) : Prop :=
    ∃ entry, entries[index]? = some entry ∧
      ∃ stage, program.data.stages[entry.site.stage]? = some stage ∧
        validateBggEntrySources stage entry
          (fun step => findBggEntryWireBefore? index entries.toList step)

def bggTraceTemplateMatches {program : Program} (entries : Array (TaggedBggSite program)) : Prop :=
  entries.size = expectedBggLayerTrace.size + 1 ∧
    (∃ first, entries[0]? = some first ∧
      ∀ index, index < expectedBggLayerTrace.size →
        ∃ expected, expectedBggLayerTrace[index]? = some expected ∧
          ∃ entry, entries[index]? = some entry ∧ entry.layer = first.layer ∧
            bggTraceEntryMatches index entry expected ∧ bggTraceSourcesMatch entries index) ∧
    (∃ entry, entries[expectedBggLayerTrace.size]? = some entry ∧
      bggTraceEntryMatches expectedBggLayerTrace.size entry expectedBggEpilogue ∧
      bggTraceSourcesMatch entries expectedBggLayerTrace.size)

/- The seven retained sites expose the actual transition-target equation
   `target = selector * nextPublic + gaussianError`.  `targetPublic` is the
   next-level public base; it is deliberately distinct from the current
   grouped source carried by `FamilyPreimageOccurrenceFact.source`. -/
structure InjectorTargetTraceSites (program : Program)
    (inputCount batchBits stateCount digitBase errorBound : Nat) where
  targetPublic : StoredNodeRef program
  targetGrid : StoredNodeRef program
  targetReindex : StoredNodeRef program
  entries : Array (TaggedInjectorTargetSite program)
  traceComplete : injectorTargetTraceMatches entries batchBits errorBound
  targetGridPayload : injectorTargetGridPayloadIsExact targetGrid.payload
    inputCount stateCount digitBase
  targetReindexPayload : injectorTargetReindexPayloadIsExact targetReindex.payload
    inputCount stateCount digitBase
  targetGridArgument : targetGrid.arguments[0]? = some targetPublic.reference.2.wire
  entryStages : ∀ entry ∈ entries, entry.site.stage = targetGrid.stage
  targetGridChildInput : ∃ product, entries[1]? = some product ∧
    product.operands[1]? = declaredGridChildInput targetGrid 0
  selectorProductEdges : ∃ selector product, entries[0]? = some selector ∧
    entries[1]? = some product ∧ product.operands[0]? = some selector.site.reference.2.wire
  targetAddEdges : ∃ product error target, entries[1]? = some product ∧
    entries[2]? = some error ∧ entries[3]? = some target ∧
    target.operands = #[product.site.reference.2.wire, error.site.reference.2.wire]
  targetAddChildOutput : ∃ target, entries[3]? = some target ∧
    declaredGridChildOutput targetGrid = some target.site.reference.2.wire
  targetReindexArgument : targetReindex.arguments[0]? = some targetGrid.reference.2.wire

def declaredSequentialChildOutput {program : Program} (site : StoredNodeRef program) :
    Option WireRef := do
  let stage ← program.data.stages[site.stage]?
  let payload ← match site.payload with
    | .sequentialLoop loop => some loop
    | _ => none
  let child ← scopeAt stage payload.child
  child.outputs[0]?

def declaredSequentialChildInput {program : Program} (site : StoredNodeRef program)
    (index : Nat) : Option WireRef := do
  let stage ← program.data.stages[site.stage]?
  let payload ← match site.payload with
    | .sequentialLoop loop => some loop
    | _ => none
  let child ← scopeAt stage payload.child
  child.inputs[index]?

def selectorMagnitudeOperandEdges {program : Program}
    (entry : TaggedSelectorMagnitudeSite program) : Prop :=
  match entry.role with
  | .digitSecretSample | .identity | .bitZero | .bitIdentity | .specialBottom =>
      entry.operands = #[] ∧ entry.site.arguments = #[]
  | .selectedSecret =>
      ∃ family, entry.operands = #[family] ∧ entry.site.arguments[0]? = some family
  | .regularDiagonal | .kDiagonal | .secretTimesBitValue | .specialTop | .specialConcat =>
      entry.site.arguments = entry.operands ∧ entry.operands.size = 2
  | .initialSelect | .bitValueSelect | .carriedVsSpecialSelect =>
      entry.operands.size = 2 ∧ entry.site.arguments[1]? = entry.operands[0]? ∧
        entry.site.arguments[2]? = entry.operands[1]?

/- These sites are a typed proof view of the actual selector construction. The view contains no
   control-arithmetic nodes: zero/identity, concatenation, multiplication, and branch selection
   are sufficient to establish the coefficient bound. -/
structure SelectorMagnitudeTraceSites (program : Program)
    (inputCount batchBits digitBase : Nat) where
  digitSecrets : StoredNodeRef program
  targetGrid : StoredNodeRef program
  selectorLoop : StoredNodeRef program
  entries : Array (TaggedSelectorMagnitudeSite program)
  traceComplete : selectorMagnitudeTraceMatches entries
  digitGridPayload : match digitSecrets.payload with
    | .parallelGrid grid =>
        grid.shape = #[.literal (inputCount * digitBase)] ∧ grid.indexSlots.size = 1 ∧
          grid.bindings = #[] ∧ grid.inputModes = #[]
    | _ => False
  selectorLoopPayload : match selectorLoop.payload with
    | .sequentialLoop loop =>
        loop.count = .literal batchBits ∧ loop.carriedCount = 1 ∧ loop.bindings = #[]
    | _ => False
  entryStages : ∀ entry ∈ entries, entry.site.stage = targetGrid.stage
  operandEdges : ∀ entry ∈ entries, selectorMagnitudeOperandEdges entry
  digitSampleOutput : ∃ sample, entries[0]? = some sample ∧
    declaredGridChildOutput digitSecrets = some sample.site.reference.2.wire
  selectedSecretFamily : ∃ selected, entries[1]? = some selected ∧
    selected.operands[0]? = some digitSecrets.reference.2.wire
  initialCarriedEdge : ∃ initial, entries[5]? = some initial ∧
    selectorLoop.arguments[0]? = some initial.site.reference.2.wire
  selectedSecretInvariantEdge : ∃ selected, entries[1]? = some selected ∧
    selectorLoop.arguments[4]? = some selected.site.reference.2.wire
  secretMultiplyInvariantEdge : ∃ multiply, entries[9]? = some multiply ∧
    multiply.operands[0]? = declaredSequentialChildInput selectorLoop 4
  carriedBranchEdge : ∃ selected, entries[13]? = some selected ∧
    selected.operands[0]? = declaredSequentialChildInput selectorLoop 0
  loopBodyOutput : ∃ body, entries[13]? = some body ∧
    declaredSequentialChildOutput selectorLoop = some body.site.reference.2.wire

structure EncryptionGraphSites (program : Program)
    (inputCount batchBits stateCount digitBase errorBound : Nat) where
  injectorInitial : StoredNodeRef program
  injectorTransitions : StoredNodeRef program
  injectorTargetTrace : InjectorTargetTraceSites program
    inputCount batchBits stateCount digitBase errorBound
  selectorMagnitudeTrace : SelectorMagnitudeTraceSites program inputCount batchBits digitBase
  selectorMagnitudeTargetEdge : selectorMagnitudeTrace.targetGrid.reference.2.wire =
    injectorTargetTrace.targetGrid.reference.2.wire
  selectorMagnitudeLoopEdge : ∃ selector, injectorTargetTrace.entries[0]? = some selector ∧
    selectorMagnitudeTrace.selectorLoop.reference.2.wire = selector.site.reference.2.wire
  injectorTransitionTargetEdge :
    injectorTransitions.arguments[2]? =
      some injectorTargetTrace.targetReindex.reference.2.wire
  injectorFinalTrapdoor : StoredNodeRef program
  onePreimage : StoredNodeRef program
  kPreimage : StoredNodeRef program
  decoderPreimage : StoredNodeRef program
  publicKeys : StoredNodeRef program
  witnessPreimages : StoredNodeRef program
  rDecomposition : StoredNodeRef program

structure DecryptionGraphSites (program : Program)
    (modulus inputCount batchBits stateCount digitBase : Nat) where
  injectorInitial : StoredNodeRef program
  injectorTransitions : StoredNodeRef program
  injectorStates : StoredNodeRef program
  injectorLoopOutput : StoredNodeRef program
  injectorBodyOutput : StoredNodeRef program
  injectorTraceEntries : Array (TaggedInjectorSite program)
  injectorTraceComplete : injectorTraceMatches injectorTraceEntries
  bggOperations : Array (TaggedBggSite program)
  bggArgumentsComplete : ∀ entry, entry ∈ bggOperations → entry.site.arguments = entry.operands
  witnessVectors : StoredNodeRef program
  oneProjection : StoredNodeRef program
  kProjection : StoredNodeRef program
  decoderProjection : StoredNodeRef program
  publicKeys : StoredNodeRef program
  circuitOutput : StoredNodeRef program
  oneMinusCircuit : StoredNodeRef program
  projectedDifference : StoredNodeRef program
  rDecomposition : StoredNodeRef program
  kPlusProjection : StoredNodeRef program
  noisyPlaintext : StoredNodeRef program
  decoded : StoredNodeRef program
  decoderCoefficient : StoredNodeRef program
  decoderLowerComparison : StoredNodeRef program
  decoderUpperComparison : StoredNodeRef program
  decoderLowerBoolToInt : StoredNodeRef program
  decoderUpperBoolToInt : StoredNodeRef program
  decoderSum : StoredNodeRef program
  decoderEqualsTwo : StoredNodeRef program
  decoderQuarter : StoredNodeRef program
  decoderThreeQuarter : StoredNodeRef program
  decoderTwo : StoredNodeRef program
  decoderThree : StoredNodeRef program
  injectorLoopExactPayload : injectorLoopExact injectorLoopOutput.payload inputCount
    injectorStates.outputs.size
  injectorTraceBodyScope : ∀ entry, entry ∈ injectorTraceEntries →
    match injectorLoopOutput.payload with
    | .sequentialLoop loop => entry.site.scope = loop.child
    | _ => False
  injectorStatesLoopEdge : injectorStates.reference.2.wire = injectorLoopOutput.reference.2.wire
  injectorBodyApplyEdge : injectorTraceEntries[4]?.bind (fun entry =>
    some entry.site.reference.2.wire) = some injectorBodyOutput.reference.2.wire
  injectorNextBodyEdge : injectorTraceEntries[6]?.bind (fun entry =>
    some entry.site.reference.2.wire) = some injectorStates.reference.2.wire
  injectorTraceCoordinates : ∀ entry, entry ∈ injectorTraceEntries →
    match injectorLoopOutput.payload with
    | .sequentialLoop loop => entry.coordinate = .structuralSlot loop.indexSlot
    | _ => False
  injectorSourceReindexPayload : ∃ entry, injectorTraceEntries[1]? = some entry ∧
      injectorPayloadIsSourceReindex entry.site.payload batchBits stateCount
  injectorTransitionReindexPayload : ∃ entry, injectorTraceEntries[2]? = some entry ∧
    injectorPayloadIsTransitionReindex entry.site.payload stateCount digitBase
  injectorNextGridPayload : ∃ next body, injectorTraceEntries[6]? = some next ∧
    injectorTraceEntries[4]? = some body ∧
    nextStateGridPayloadIsExact next.site.payload body.site.scope stateCount
  injectorGridChildOutput : ∃ next body, injectorTraceEntries[6]? = some next ∧
    injectorTraceEntries[4]? = some body ∧
    declaredGridChildOutput next.site = some body.site.reference.2.wire
  injectorReindexCarriedEdge : injectorTraceEntries[1]?.bind (fun entry =>
    entry.site.arguments[0]?) = injectorTraceEntries[4]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorSelectionDigitEdge : injectorTraceEntries[3]?.bind (fun entry =>
    entry.site.arguments[1]?) = injectorTraceEntries[0]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorSelectionReindexEdge : injectorTraceEntries[3]?.bind (fun entry =>
    entry.site.arguments[0]?) = injectorTraceEntries[2]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorApplySourceEdge : injectorTraceEntries[4]?.bind (fun entry =>
    entry.site.arguments[0]?) = injectorTraceEntries[1]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorApplyTransitionEdge : injectorTraceEntries[4]?.bind (fun entry =>
    entry.site.arguments[1]?) = injectorTraceEntries[3]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorNextSourceEdge : injectorTraceEntries[6]?.bind (fun entry =>
    entry.site.arguments[0]?) = injectorTraceEntries[1]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  injectorNextTransitionEdge : injectorTraceEntries[6]?.bind (fun entry =>
    entry.site.arguments[1]?) = injectorTraceEntries[3]?.bind (fun entry =>
    some entry.site.reference.2.wire)
  bggLayerOutputEdge : bggOperations[expectedBggLayerTrace.size]?.bind (fun entry =>
    some entry.site.reference.2.wire) = circuitOutput.arguments[0]?
  oneMinusOneEdge : oneMinusCircuit.arguments[0]? = some oneProjection.reference.2.wire
  oneMinusCircuitEdge : oneMinusCircuit.arguments[1]? = some circuitOutput.reference.2.wire
  projectedDifferenceValueEdge :
    projectedDifference.arguments[0]? = some oneMinusCircuit.reference.2.wire
  projectedDifferencePreimageEdge :
    projectedDifference.arguments[1]? = some rDecomposition.reference.2.wire
  kPlusProjectionKEdge : kPlusProjection.arguments[0]? = some kProjection.reference.2.wire
  kPlusProjectionDifferenceEdge :
    kPlusProjection.arguments[1]? = some projectedDifference.reference.2.wire
  noisyPlaintextDecoderEdge :
    noisyPlaintext.arguments[0]? = some decoderProjection.reference.2.wire
  noisyPlaintextProjectionEdge :
    noisyPlaintext.arguments[1]? = some kPlusProjection.reference.2.wire
  decoderCoefficientNoiseEdge : decoderCoefficient.arguments[0]? = some noisyPlaintext.reference.2.wire
  decoderLowerCoefficientEdge : decoderLowerComparison.arguments[1]? = some decoderCoefficient.reference.2.wire
  decoderUpperCoefficientEdge : decoderUpperComparison.arguments[0]? = some decoderCoefficient.reference.2.wire
  decoderLowerIntEdge : decoderLowerBoolToInt.arguments[0]? = some decoderLowerComparison.reference.2.wire
  decoderUpperIntEdge : decoderUpperBoolToInt.arguments[0]? = some decoderUpperComparison.reference.2.wire
  decoderSumLowerEdge : decoderSum.arguments[0]? = some decoderLowerBoolToInt.reference.2.wire
  decoderSumUpperEdge : decoderSum.arguments[1]? = some decoderUpperBoolToInt.reference.2.wire
  decoderEqualsSumEdge : decoderEqualsTwo.arguments[0]? = some decoderSum.reference.2.wire
  decoderEqualsTwoEdge : decoderEqualsTwo.arguments[1]? = some decoderTwo.reference.2.wire
  decoderLowerQuarterEdge : decoderLowerComparison.arguments[0]? = some decoderQuarter.reference.2.wire
  decoderUpperThreeQuarterEdge : decoderUpperComparison.arguments[1]? = some decoderThreeQuarter.reference.2.wire
  decoderThreeQuarterThreeEdge : decoderThreeQuarter.arguments[1]? = some decoderThree.reference.2.wire
  decoderThreeQuarterQuarterEdge : decoderThreeQuarter.arguments[0]? = some decoderQuarter.reference.2.wire
  decodedDecoderEdge : decoded.reference.2.wire = decoderEqualsTwo.reference.2.wire
  injectorLoopIsSequential : injectorLoopOutput.kind = .sequentialLoop
  injectorLoopPayload : loopCarries injectorLoopOutput.payload injectorStates.outputs.size
  decoderCoefficientIsExtraction : decoderCoefficient.kind = .coefficientExtraction
  decoderComparisonsAreInteger : decoderLowerComparison.kind = .other ∧
    decoderUpperComparison.kind = .other
  decoderCoefficientLiteral : isCoefficientAtZero decoderCoefficient.payload
  decoderLowerComparisonPayload : isIntegerComparison decoderLowerComparison.payload
  decoderUpperComparisonPayload : isIntegerComparison decoderUpperComparison.payload
  decoderLowerLessEqualPayload : isLessEqualComparison decoderLowerComparison.payload
  decoderUpperLessEqualPayload : isLessEqualComparison decoderUpperComparison.payload
  decoderLowerBoolToIntPayload : isBoolToInt decoderLowerBoolToInt.payload
  decoderUpperBoolToIntPayload : isBoolToInt decoderUpperBoolToInt.payload
  decoderSumPayload : isIntegerAdd decoderSum.payload
  decoderEqualsPayload : isIntegerComparison decoderEqualsTwo.payload
  decoderEqualsExactPayload : isEqualComparison decoderEqualsTwo.payload
  decoderQuarterPayload : isQuarterPayload decoderQuarter.payload modulus
  decoderThreeQuarterPayload : isIntegerMultiply decoderThreeQuarter.payload
  decoderTwoPayload : decoderTwo.payload = .constantInt 2
  decoderThreePayload : decoderThree.payload = .constantInt 3
  circuitOutputPayload : isFamilyGetDynamicZero circuitOutput.payload
  oneMinusCircuitPayload : oneMinusCircuit.payload = .matrixBinary .subtract
  projectedDifferencePayload : projectedDifference.payload = .applyPreimage
  kPlusProjectionPayload : kPlusProjection.payload = .matrixBinary .add
  noisyPlaintextPayload : noisyPlaintext.payload = .matrixBinary .subtract

def messageNat : Bool → Nat
  | false => 0
  | true => 1

noncomputable def idealPlaintext (parameters : Parameters) (message : Bool) :
    ExactMatrix parameters.modulus parameters.ringDimension 1 1 :=
  fun _ _ =>
    algebraMap Int (ExactPoly parameters.modulus parameters.ringDimension)
      ((parameters.modulus / 2 * messageNat message : Nat) : Int)

/- A typed external reference points to an input node in the stage's root scope, so its evaluator
   occurrence is always the top-level empty path.  Fixing that path here prevents callers from
   substituting a same-typed value stored under an unrelated nested occurrence. -/
def externalInputAt {candidate : Candidate} {wireType : WireType}
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (externalReference : TypedExternalInputRef candidate.program wireType) :
    Except EvalError (RuntimeDynamicValue oracle) :=
  let reference := externalReference.reference
  envInput env reference.stage reference.wire.scope reference.wire.node #[] reference.wire

def typedExternalInputAt {candidate : Candidate} {wireType : WireType}
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (externalReference : TypedExternalInputRef candidate.program wireType) :
    Except EvalError (RuntimeValue oracle wireType) := do
  let value ← externalInputAt oracle env externalReference
  match coerceValue wireType value with
  | some typed => pure typed
  | none =>
      let reference := externalReference.reference
      throw (.wrongType reference.stage reference.wire.scope reference.wire.node)

def oneDimensionalFamily {length : Nat} (values : Family [length] Int) : Fin length → Int :=
  fun index => values (index, ())

def externalCircuitAt {candidate : Candidate}
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (env : RuntimeEvalEnv oracle candidate.program.data)
    (references : BooleanCircuitInputRefs candidate.program candidate.circuitShape) :
    Except EvalError (LayeredBoolCircuit candidate.circuitShape) := do
  let activeGateCounts ← typedExternalInputAt oracle env references.activeGateCountsInput
  let gateKinds ← typedExternalInputAt oracle env references.circuitGateKindsInput
  let leftSources ← typedExternalInputAt oracle env references.circuitLeftSourcesInput
  let rightSources ← typedExternalInputAt oracle env references.circuitRightSourcesInput
  let outputSources ← typedExternalInputAt oracle env references.circuitOutputSourceInput
  pure (.ofFamilies (oneDimensionalFamily activeGateCounts)
    (oneDimensionalFamily gateKinds) (oneDimensionalFamily leftSources)
    (oneDimensionalFamily rightSources) (oneDimensionalFamily outputSources))

def canonicalPaddedBits {maximumWidth : Nat} (logicalWidth : Nat)
    (values : Fin maximumWidth → Int) : Prop :=
  logicalWidth ≤ maximumWidth ∧ ∀ slot,
    if slot.val < logicalWidth then values slot = 0 ∨ values slot = 1
    else values slot = 0

def logicalBits {maximumWidth logicalWidth : Nat} (widthBound : logicalWidth ≤ maximumWidth)
    (values : Fin maximumWidth → Int) : Fin logicalWidth → Bool :=
  fun index => decide (values ⟨index.val, lt_of_lt_of_le index.isLt widthBound⟩ = 1)

theorem LayeredBoolCircuitShape.instanceWidth_le_maxLayerWidth
    {shape : LayeredBoolCircuitShape} (validity : shape.Valid) :
    shape.instanceWidth ≤ shape.maxLayerWidth := by
  exact le_trans (Nat.le_add_right _ _) validity.2.2.2.2.2

theorem LayeredBoolCircuitShape.witnessWidth_le_maxLayerWidth
    {shape : LayeredBoolCircuitShape} (validity : shape.Valid) :
    shape.witnessWidth ≤ shape.maxLayerWidth := by
  exact le_trans (Nat.le_add_left _ _) validity.2.2.2.2.2

def ValidExternalInputs (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (message : Bool) : Prop :=
  ∃ circuit instanceBits witnessBits,
    typedExternalInputAt oracle env candidate.refs.messageInput = .ok message ∧
    externalCircuitAt oracle env candidate.refs.encryptionCircuit = .ok circuit ∧
    externalCircuitAt oracle env candidate.refs.decryptionCircuit = .ok circuit ∧
    typedExternalInputAt oracle env candidate.refs.encryptionInstanceBitsInput =
      .ok instanceBits ∧
    typedExternalInputAt oracle env candidate.refs.decryptionInstanceBitsInput =
      .ok instanceBits ∧
    typedExternalInputAt oracle env candidate.refs.witnessBitsInput = .ok witnessBits ∧
    circuit.Valid ∧
    canonicalPaddedBits candidate.circuitShape.instanceWidth
      (oneDimensionalFamily instanceBits) ∧
    canonicalPaddedBits candidate.circuitShape.witnessWidth
      (oneDimensionalFamily witnessBits)

def BooleanCircuitEvaluatesToOne (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) : Prop :=
  ∃ circuit instanceBits witnessBits, ∃ validity : circuit.Valid,
    externalCircuitAt oracle env candidate.refs.decryptionCircuit = .ok circuit ∧
    typedExternalInputAt oracle env candidate.refs.decryptionInstanceBitsInput =
      .ok instanceBits ∧
    typedExternalInputAt oracle env candidate.refs.witnessBitsInput = .ok witnessBits ∧
    circuit.evaluate validity
      (logicalBits (LayeredBoolCircuitShape.instanceWidth_le_maxLayerWidth validity.1)
        (oneDimensionalFamily instanceBits))
      (logicalBits (LayeredBoolCircuitShape.witnessWidth_le_maxLayerWidth validity.1)
        (oneDimensionalFamily witnessBits)) = some true

/- Both public predicates read the circuit and padded bit families through the same typed
   external-input functions.  Their successful results therefore determine one common circuit,
   instance family, and witness family.  This view records that shared data once, together with
   the canonical-padding facts and the accepting logical evaluation. -/
structure AcceptingExternalInputView
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (message : Bool) where
  circuit : LayeredBoolCircuit candidate.circuitShape
  instanceBits : Family [candidate.circuitShape.maxLayerWidth] Int
  witnessBits : Family [candidate.circuitShape.maxLayerWidth] Int
  validity : circuit.Valid
  messageRead : typedExternalInputAt oracle env candidate.refs.messageInput = .ok message
  encryptionCircuitRead :
    externalCircuitAt oracle env candidate.refs.encryptionCircuit = .ok circuit
  decryptionCircuitRead :
    externalCircuitAt oracle env candidate.refs.decryptionCircuit = .ok circuit
  encryptionInstanceRead :
    typedExternalInputAt oracle env candidate.refs.encryptionInstanceBitsInput = .ok instanceBits
  decryptionInstanceRead :
    typedExternalInputAt oracle env candidate.refs.decryptionInstanceBitsInput = .ok instanceBits
  witnessRead : typedExternalInputAt oracle env candidate.refs.witnessBitsInput = .ok witnessBits
  instanceCanonical : canonicalPaddedBits candidate.circuitShape.instanceWidth
    (oneDimensionalFamily instanceBits)
  witnessCanonical : canonicalPaddedBits candidate.circuitShape.witnessWidth
    (oneDimensionalFamily witnessBits)
  evaluatesToOne : circuit.evaluate validity
    (logicalBits (LayeredBoolCircuitShape.instanceWidth_le_maxLayerWidth validity.1)
      (oneDimensionalFamily instanceBits))
    (logicalBits (LayeredBoolCircuitShape.witnessWidth_le_maxLayerWidth validity.1)
      (oneDimensionalFamily witnessBits)) = some true

/- Successful `Except.ok` results are injective, so independently quantified witnesses returned
   by the two public predicates cannot describe different external values.  No caller equality is
   needed: all three identities are consequences of the actual typed reads. -/
theorem acceptingExternalInputView
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {message : Bool}
    (inputs : ValidExternalInputs oracle candidate env message)
    (accepting : BooleanCircuitEvaluatesToOne oracle candidate env) :
    Nonempty (AcceptingExternalInputView oracle candidate env message) := by
  obtain ⟨circuit, instanceBits, witnessBits, messageRead, encryptionCircuitRead,
    decryptionCircuitRead, encryptionInstanceRead, decryptionInstanceRead, witnessRead,
    validity, instanceCanonical, witnessCanonical⟩ := inputs
  obtain ⟨acceptingCircuit, acceptingInstanceBits, acceptingWitnessBits,
    acceptingValidity, acceptingCircuitRead, acceptingInstanceRead, acceptingWitnessRead,
    evaluatesToOne⟩ := accepting
  have circuitEq : acceptingCircuit = circuit :=
    Except.ok.inj (acceptingCircuitRead.symm.trans decryptionCircuitRead)
  have instanceBitsEq : acceptingInstanceBits = instanceBits :=
    Except.ok.inj (acceptingInstanceRead.symm.trans decryptionInstanceRead)
  have witnessBitsEq : acceptingWitnessBits = witnessBits :=
    Except.ok.inj (acceptingWitnessRead.symm.trans witnessRead)
  subst acceptingCircuit
  subst acceptingInstanceBits
  subst acceptingWitnessBits
  have validityEq : acceptingValidity = validity := Subsingleton.elim _ _
  subst acceptingValidity
  exact ⟨{
    circuit
    instanceBits
    witnessBits
    validity
    messageRead
    encryptionCircuitRead
    decryptionCircuitRead
    encryptionInstanceRead
    decryptionInstanceRead
    witnessRead
    instanceCanonical
    witnessCanonical
    evaluatesToOne
  }⟩

def traceValueAt {backend : SemanticBackend} (trace : Trace backend)
    (occurrence : WireOccurrence) : Option (DynamicValue backend) :=
  Mxx.IR.traceValueAt trace occurrence

def traceTypedValueAt {backend : SemanticBackend} {program : Program} {wireType : WireType}
    (trace : Trace backend) (path : OccurrencePath)
    (reference : TypedWireRef program wireType) : Option (Value backend wireType) :=
  (traceValueAt trace (occurrenceOf reference.stage path reference.wire)).bind
    (coerceValue wireType)

def preimageSourceType (preimageType : MatrixType) (sourceRows : Nat) : MatrixType := {
  modulus := preimageType.modulus
  ringDimension := preimageType.ringDimension
  rows := sourceRows
  columns := preimageType.rows
}

def preimageTargetType (preimageType : MatrixType) (sourceRows : Nat) : MatrixType := {
  modulus := preimageType.modulus
  ringDimension := preimageType.ringDimension
  rows := sourceRows
  columns := preimageType.columns
}

/- Even samplers without a smallness obligation must return the value stored in the actual trace.
   This prevents an environment fact for one dynamic occurrence from being reused for another. -/
structure SampleOutputFact (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle)
    (sample : SampleRef candidate.program.data) where
  value : RuntimeDynamicValue oracle
  environment_output : env.sampleOutput sample = .ok value
  trace_output : traceValueAt trace sample.occurrence = some value

structure CutoffSampleOccurrenceFact (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle)
    (sample : SampleRef candidate.program.data) (matrixType : MatrixType)
    (bound : Nat) where
  actual : RuntimeMatrixValue matrixType
  environment_output : env.sampleOutput sample = .ok ⟨.matrix matrixType, actual⟩
  trace_output :
    traceValueAt trace sample.occurrence = some ⟨.matrix matrixType, actual⟩
  bounded : BoundedLift actual bound

structure PreimageOccurrenceFact (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle)
    (sample : SampleRef candidate.program.data) (preimageType : MatrixType)
    (preimageBound : Nat) where
  sourceRows : Nat
  sourceOccurrence : WireOccurrence
  targetOccurrence : WireOccurrence
  source_argument : sample.argumentOccurrence? 0 = some sourceOccurrence
  target_argument : sample.argumentOccurrence? 2 = some targetOccurrence
  source : RuntimeMatrixValue (preimageSourceType preimageType sourceRows)
  target : RuntimeMatrixValue (preimageTargetType preimageType sourceRows)
  preimage : Mxx.Runtime.PreimageValue preimageType
  source_trace : traceValueAt trace sourceOccurrence =
    some ⟨.matrix (preimageSourceType preimageType sourceRows), source⟩
  target_trace : traceValueAt trace targetOccurrence =
    some ⟨.matrix (preimageTargetType preimageType sourceRows), target⟩
  environment_output : env.sampleOutput sample = .ok ⟨.preimage preimageType, preimage⟩
  trace_output : traceValueAt trace sample.occurrence =
    some ⟨.preimage preimageType, preimage⟩
  relation : RightPreimage source preimage.exactMatrix target
  bounded : PreimageWithin preimage.exactMatrix preimageBound

/- Rank-zero family sources are represented by an ordinary matrix wire; positive-rank sources are
   represented by a family wire.  Both normalize to the same group-indexed mathematical source. -/
inductive MatrixFamilyValue (oracle : Mxx.Runtime.RuntimeGadgetOracle) (matrixType : MatrixType) :
    (shape : List Nat) → RuntimeDynamicValue oracle →
      (Family shape (RuntimeMatrixValue matrixType)) → Prop
  | scalar (matrix : RuntimeMatrixValue matrixType) :
      MatrixFamilyValue oracle matrixType [] ⟨.matrix matrixType, matrix⟩ (fun _ ↦ matrix)
  | family (shape : List Nat) (matrices : Family shape (RuntimeMatrixValue matrixType)) :
      MatrixFamilyValue oracle matrixType shape ⟨.family shape (.matrix matrixType), matrices⟩ matrices

structure FamilyPreimageOccurrenceFact (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle)
    (sample : SampleRef candidate.program.data) (preimageType : MatrixType)
    (preimageBound : Nat) where
  groupShape : List Nat
  branchExtent : Nat
  branch_nonempty : 0 < branchExtent
  sourceRows : Nat
  sourceOccurrence : WireOccurrence
  targetOccurrence : WireOccurrence
  source_argument : sample.argumentOccurrence? 0 = some sourceOccurrence
  target_argument : sample.argumentOccurrence? 2 = some targetOccurrence
  sourceDynamic : RuntimeDynamicValue oracle
  source : Family groupShape
    (RuntimeMatrixValue (preimageSourceType preimageType sourceRows))
  source_value : MatrixFamilyValue oracle (preimageSourceType preimageType sourceRows)
    groupShape sourceDynamic source
  target : Family (groupShape ++ [branchExtent])
    (RuntimeMatrixValue (preimageTargetType preimageType sourceRows))
  preimage : Family (groupShape ++ [branchExtent])
    (Mxx.Runtime.PreimageValue preimageType)
  source_trace : traceValueAt trace sourceOccurrence = some sourceDynamic
  target_trace : traceValueAt trace targetOccurrence = some
    ⟨.family (groupShape ++ [branchExtent])
      (.matrix (preimageTargetType preimageType sourceRows)), target⟩
  environment_output : env.sampleOutput sample = .ok
    ⟨.family (groupShape ++ [branchExtent]) (.preimage preimageType), preimage⟩
  trace_output : traceValueAt trace sample.occurrence = some
    ⟨.family (groupShape ++ [branchExtent]) (.preimage preimageType), preimage⟩
  relation : ∀ group branch,
    RightPreimage (source group)
      (preimage (FamilyIndex.append group branch)).exactMatrix
      (target (FamilyIndex.append group branch))
  bounded : ∀ group branch,
    PreimageWithin (preimage (FamilyIndex.append group branch)).exactMatrix preimageBound

def SamplerOccurrenceFact (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle)
    (sample : SampleRef candidate.program.data) : Prop :=
  match sample.factKind with
  | .cutoff matrixType bound =>
      Nonempty (CutoffSampleOccurrenceFact oracle candidate env trace sample matrixType bound)
  | .preimage matrixType bound =>
      Nonempty (PreimageOccurrenceFact oracle candidate env trace sample matrixType bound)
  | .familyPreimage matrixType bound =>
      Nonempty (FamilyPreimageOccurrenceFact oracle candidate env trace sample matrixType bound)
  | .outputOnly => Nonempty (SampleOutputFact oracle candidate env trace sample)
  | .invalid => False

/- Quantification over every reached `SampleRef` is extensionally exhaustive.  There is no caller
   inventory to omit an occurrence and no empty-plan constructor. -/
structure GoodSamples (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) (trace : RuntimeTrace oracle) : Prop where
  occurrence : ∀ sample, sample.Reached trace →
    SamplerOccurrenceFact oracle candidate env trace sample

/- Correctness is conditional on one successful evaluator execution whose actual sampler
   occurrences all satisfy `GoodSamples`.  This packages liveness and sampling for the same trace;
   it does not claim that arbitrary typed environments evaluate successfully. -/
structure GoodRunPromise (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (env : RuntimeEvalEnv oracle candidate.program.data) where
  trace : RuntimeTrace oracle
  evaluated : Mxx.IR.eval (RuntimeBackend oracle) candidate.program env = .ok trace
  goodSamples : GoodSamples oracle candidate env trace

/- A type-valued view is the place where generated topology and local equations will be attached.
   It contains no alternative evaluator and no graph-wide normal form. -/
structure Candidate.HasDiamondGraphShape (candidate : Candidate) : Type where
  encryptionCircuit : BooleanCircuitInputRefs candidate.program candidate.circuitShape
  decryptionCircuit : BooleanCircuitInputRefs candidate.program candidate.circuitShape
  encryptionCircuitOutput : AnyTypedWireRef candidate.program
  decryptionCircuitOutput : AnyTypedWireRef candidate.program
  refs_encryption_eq : encryptionCircuit = candidate.refs.encryptionCircuit
  refs_decryption_eq : decryptionCircuit = candidate.refs.decryptionCircuit
  refs_encryption_output_eq : encryptionCircuitOutput = candidate.refs.encryptionCircuitOutput
  refs_decryption_output_eq : decryptionCircuitOutput = candidate.refs.decryptionCircuitOutput
  encryptionSites : EncryptionGraphSites candidate.program candidate.parameters.inputCount
    candidate.parameters.batchBits (candidate.parameters.witnessWidth + 1) candidate.parameters.digitBase
    candidate.parameters.errorCutoff
  decryptionSites : DecryptionGraphSites candidate.program candidate.parameters.modulus
    candidate.parameters.inputCount candidate.parameters.batchBits (candidate.parameters.witnessWidth + 1)
    candidate.parameters.digitBase
  decoded_site_eq : decryptionSites.decoded.reference.2.wire = candidate.refs.decodedOutput.wire
  noisy_plaintext_site_eq :
    decryptionSites.noisyPlaintext.reference.2.wire = candidate.refs.noisyPlaintextOutput.wire
  bggTraceTemplate : bggTraceTemplateMatches decryptionSites.bggOperations
  bggTraceCoverage : ∀ entry, entry ∈ decryptionSites.bggOperations →
    entry.site.arguments = entry.operands
  bggOutputSelectorEdge : decryptionSites.circuitOutput.arguments[1]? =
    some decryptionCircuit.circuitOutputSourceInput.reference.wire

/- Parameter bindings are part of a generated stage's concrete identity.  Looking for a name in
   an arbitrary stage is insufficient: a same-typed binding in an unrelated stage (or a legacy
   unprefixed name) must not discharge a Diamond parameter obligation.  The stage index is taken
   from a generated typed input reference, while the stage and binding names below are the
   authoritative names declared by `DiamondGraphParams` and `BooleanCircuitFamilyParams`. -/
def bindingValueAt (stage : Mxx.IR.Stage) (name : String) : Option Int :=
  (stage.bindings.find? (fun binding => binding.1 = name)).bind (fun binding => some binding.2)

def hasConcreteBindingAt (data : Mxx.IR.ProgramData) (stageIndex : Nat)
    (stageName bindingName : String) (value : Nat) : Prop :=
  ∃ stage, data.stages[stageIndex]? = some stage ∧
    stage.name = stageName ∧ bindingValueAt stage bindingName = some (value : Int)

def bindingIdentityRegressionData : Mxx.IR.ProgramData := {
  identity := { irVersion := 0, linkedProgramSha256 := [] }
  stages := #[{
    name := "diamond-we-encryption"
    bindings := #[
      ("diamond_modulus", 17),
      ("diamond_ring_dimension", 4)]
    scopes := #[]
    root := 0
    namedOutputs := #[]
  }]
  artifactLinks := #[]
}

def bindingIdentityWrongValue : Mxx.IR.ProgramData := {
  bindingIdentityRegressionData with
    stages := #[{
      name := "diamond-we-encryption"
      bindings := #[
        ("diamond_modulus", 19),
        ("diamond_ring_dimension", 4)]
      scopes := #[]
      root := 0
      namedOutputs := #[]
    }]
}

def bindingIdentityWrongName : Mxx.IR.ProgramData := {
  bindingIdentityRegressionData with
    stages := #[{
      name := "diamond-we-encryption"
      bindings := #[
        ("modulus", 17),
        ("diamond_ring_dimension", 4)]
      scopes := #[]
      root := 0
      namedOutputs := #[]
    }]
}

def bindingIdentityWrongStage : Mxx.IR.ProgramData := {
  bindingIdentityRegressionData with
    stages := #[{
      name := "unrelated-stage"
      bindings := #[
        ("diamond_modulus", 17),
        ("diamond_ring_dimension", 4)]
      scopes := #[]
      root := 0
      namedOutputs := #[]
    }]
}

example : hasConcreteBindingAt bindingIdentityRegressionData 0
    "diamond-we-encryption" "diamond_modulus" 17 := by
  simp [hasConcreteBindingAt, bindingValueAt, bindingIdentityRegressionData]

example : ¬ hasConcreteBindingAt bindingIdentityWrongValue 0
    "diamond-we-encryption" "diamond_modulus" 17 := by
  simp [hasConcreteBindingAt, bindingValueAt, bindingIdentityWrongValue,
    bindingIdentityRegressionData]

example : ¬ hasConcreteBindingAt bindingIdentityWrongName 0
    "diamond-we-encryption" "diamond_modulus" 17 := by
  simp [hasConcreteBindingAt, bindingValueAt, bindingIdentityWrongName,
    bindingIdentityRegressionData]

example : ¬ hasConcreteBindingAt bindingIdentityWrongStage 0
    "diamond-we-encryption" "diamond_modulus" 17 := by
  simp [hasConcreteBindingAt, bindingValueAt, bindingIdentityWrongStage,
    bindingIdentityRegressionData]

inductive DiamondGraphBinding
  | modulus
  | ringDimension
  | gadgetBase
  | gadgetDigitCount
  | witnessWidth
  | errorCutoff
  | preimageCutoff
  | instanceWidth
  | depth
  | maxLayerWidth
  deriving Repr, DecidableEq

def DiamondGraphBinding.name : DiamondGraphBinding → String
  | .modulus => "diamond_modulus"
  | .ringDimension => "diamond_ring_dimension"
  | .gadgetBase => "diamond_gadget_base"
  | .gadgetDigitCount => "diamond_digit_count"
  | .witnessWidth => "witness_width"
  | .errorCutoff => "diamond_error_max_coefficient_bound"
  | .preimageCutoff => "diamond_preimage_max_coefficient_bound"
  | .instanceWidth => "instance_width"
  | .depth => "depth"
  | .maxLayerWidth => "max_layer_width"

def hasDiamondGraphBinding (candidate : Candidate) (binding : DiamondGraphBinding)
    (value : Nat) : Prop :=
  hasConcreteBindingAt candidate.program.data candidate.refs.messageInput.reference.stage
      "diamond-we-encryption" binding.name value ∧
    hasConcreteBindingAt candidate.program.data
      candidate.refs.decryptionInstanceBitsInput.reference.stage
      "diamond-we-decryption" binding.name value

structure CandidateParametersMatch (candidate : Candidate) : Prop where
  modulus_binding :
    hasDiamondGraphBinding candidate .modulus candidate.parameters.modulus
  ring_dimension_binding :
    hasDiamondGraphBinding candidate .ringDimension candidate.parameters.ringDimension
  gadget_base_binding :
    hasDiamondGraphBinding candidate .gadgetBase candidate.parameters.gadgetBase
  gadget_digit_count_binding :
    hasDiamondGraphBinding candidate .gadgetDigitCount candidate.parameters.gadgetDigitCount
  witness_width_binding :
    hasDiamondGraphBinding candidate .witnessWidth candidate.parameters.witnessWidth
  witness_width_shape :
    candidate.circuitShape.witnessWidth = candidate.parameters.witnessWidth
  error_cutoff_binding :
    hasDiamondGraphBinding candidate .errorCutoff candidate.parameters.errorCutoff
  preimage_cutoff_binding :
    hasDiamondGraphBinding candidate .preimageCutoff candidate.parameters.preimageCutoff
  instance_width_binding :
    hasDiamondGraphBinding candidate .instanceWidth candidate.circuitShape.instanceWidth
  depth_binding :
    hasDiamondGraphBinding candidate .depth candidate.circuitShape.depth
  max_layer_width_binding :
    hasDiamondGraphBinding candidate .maxLayerWidth candidate.circuitShape.maxLayerWidth
  circuit_shape_valid : candidate.circuitShape.Valid
  plaintext_modulus :
    candidate.plaintextMatrixType.modulus = Int.ofNat candidate.parameters.modulus
  plaintext_ring_dimension :
    candidate.plaintextMatrixType.ringDimension = candidate.parameters.ringDimension
  plaintext_rows : candidate.plaintextMatrixType.rows = 1
  plaintext_columns : candidate.plaintextMatrixType.columns = 1
  plaintext_valid : candidate.plaintextMatrixType.Valid

/- The parameter certificate is a view of the generated program.  Keeping the
   alias lets downstream proofs depend on the view name while all fields above
   remain kernel-checked against the concrete IR bindings and matrix type. -/
abbrev ParametersMatchProgram (candidate : Candidate) : Prop := CandidateParametersMatch candidate

structure CertifiedCandidate where
  candidate : Candidate
  parametersMatch : ParametersMatchProgram candidate
  representation : RuntimeDcrtRepresentation
  representationMatches : representation.validFor candidate.parameters

theorem CandidateParametersMatch.plaintext_modulus_toNat
    {candidate : Candidate} (parametersMatch : CandidateParametersMatch candidate) :
    candidate.plaintextMatrixType.modulus.toNat = candidate.parameters.modulus := by
  rw [parametersMatch.plaintext_modulus]
  simp

theorem CandidateParametersMatch.runtime_plaintext_type
    {candidate : Candidate} (parametersMatch : CandidateParametersMatch candidate) :
    RuntimeMatrixValue candidate.plaintextMatrixType =
      ExactMatrix candidate.parameters.modulus candidate.parameters.ringDimension 1 1 := by
  unfold RuntimeMatrixValue
  rw [parametersMatch.plaintext_modulus_toNat]
  rw [parametersMatch.plaintext_ring_dimension]
  rw [parametersMatch.plaintext_rows, parametersMatch.plaintext_columns]

/- A runtime plaintext is transported to the application plaintext type only through the checked
   parameter equalities above.  This is an equality cast, not a matrix conversion: no coefficients
   are changed and a value with a mismatched modulus, ring dimension, or shape cannot be supplied. -/
noncomputable def CandidateParametersMatch.plaintextTransport
    {candidate : Candidate} (parametersMatch : CandidateParametersMatch candidate)
    (value : RuntimeMatrixValue candidate.plaintextMatrixType) :
    ExactMatrix candidate.parameters.modulus candidate.parameters.ringDimension 1 1 :=
  parametersMatch.runtime_plaintext_type ▸ value

theorem CandidateParametersMatch.plaintextTransport_eq
    {candidate : Candidate} (parametersMatch : CandidateParametersMatch candidate)
    (value : RuntimeMatrixValue candidate.plaintextMatrixType) :
    parametersMatch.plaintextTransport value =
      (parametersMatch.runtime_plaintext_type ▸ value) := rfl

end Mxx.We.DiamondWE
