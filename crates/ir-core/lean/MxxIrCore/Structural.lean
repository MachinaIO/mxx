import MxxIrCore.Program

namespace Mxx
namespace IR

/-! Structural execution data.  These types are intentionally independent of
the concrete cryptographic backend; a backend supplies the meaning of matrix,
trapdoor, and preimage wires. -/

structure OccurrenceFrame where
  stage : Nat
  scope : ScopeId
  owner : NodeId
  laneOrIteration : Nat
  deriving Repr, DecidableEq

abbrev OccurrencePath := Array OccurrenceFrame

inductive GadgetMode where
  | regular
  | small
  deriving Repr, DecidableEq

structure GadgetLayout where
  mode : GadgetMode
  base : Nat
  digits : Nat
  sourceRows : Nat
  targetRows : Nat
  sourceColumns : Nat
  targetColumns : Nat
  deriving Repr, DecidableEq

def GadgetLayout.Valid (layout : GadgetLayout) : Prop :=
  1 < layout.base ∧ 0 < layout.digits ∧
    layout.targetRows = layout.sourceRows * layout.digits ∧
    layout.targetColumns = layout.sourceColumns

def gadgetMatrixType (target : MatrixType) (layout : GadgetLayout) : MatrixType :=
  { target with columns := target.rows * layout.digits }

def gadgetPreimageType (target : MatrixType) (layout : GadgetLayout) : MatrixType :=
  { target with rows := target.rows * layout.digits }

inductive GadgetFailure where
  | invalidLayout
  | unsupported
  deriving Repr, DecidableEq

structure SemanticBackend where
  denoteMatrix : MatrixType → Type
  denoteTrapdoor : TrapdoorType → Type
  denotePreimage : MatrixType → Type
  denoteTypedBlob : String → List UInt8 → Type
  matrixZero : (t : MatrixType) → denoteMatrix t
  matrixIdentity : (t : MatrixType) → denoteMatrix t
  matrixAdd : (t : MatrixType) → denoteMatrix t → denoteMatrix t → denoteMatrix t
  matrixSubtract : (t : MatrixType) → denoteMatrix t → denoteMatrix t → denoteMatrix t
  matrixScale : (t : MatrixType) → Int → denoteMatrix t → denoteMatrix t
  matrixMultiply : (left right out : MatrixType) → denoteMatrix left → denoteMatrix right →
    denoteMatrix out
  matrixNegate : (t : MatrixType) → denoteMatrix t → denoteMatrix t
  matrixTranspose : (input output : MatrixType) → denoteMatrix input → denoteMatrix output
  matrixConstant : (t : MatrixType) → MatrixLiteral → StructuralEnv → denoteMatrix t
  matrixSlice : (input output : MatrixType) → Nat → Nat → Nat → Nat →
    denoteMatrix input → denoteMatrix output
  matrixConcat : (output : MatrixType) → ConcatAxis →
    Array (Σ t : MatrixType, denoteMatrix t) → denoteMatrix output
  /-- Certificate family for a decomposition, indexed by the exact gadget constructor and
      exact values observed at the evaluator boundary. -/
  gadgetCertificate : (targetType : MatrixType) → (layout : GadgetLayout) → StructuralEnv →
    denoteMatrix (gadgetMatrixType targetType layout) → denoteMatrix targetType →
    denotePreimage (gadgetPreimageType targetType layout) → Prop
  gadgetDecompose : (targetType : MatrixType) → (layout : GadgetLayout) →
    (structural : StructuralEnv) → (gadget : denoteMatrix (gadgetMatrixType targetType layout)) →
    (target : denoteMatrix targetType) →
    Except GadgetFailure
      (Σ preimage : denotePreimage (gadgetPreimageType targetType layout), PLift
        (gadgetCertificate targetType layout structural gadget target preimage))
  extractCoefficient : (t : MatrixType) → Nat → denoteMatrix t → Int
  bitExtract : Int → Int → Bool
  trapdoorPublic : (t : TrapdoorType) → denoteTrapdoor t → denoteMatrix t.matrix
  materializePreimage : (t : MatrixType) → denotePreimage t → denoteMatrix t
  applyPreimage : (left right out : MatrixType) → denoteMatrix left → denotePreimage right →
    denoteMatrix out

def FamilyIndex : List Nat → Type
  | [] => Unit
  | extent :: rest => Fin extent × FamilyIndex rest

def Family (shape : List Nat) (element : Type) : Type := FamilyIndex shape → element

def Family.shapeProduct : List Nat → Nat
  | [] => 1
  | extent :: rest => extent * Family.shapeProduct rest

def Family.rowMajorOffset : (shape : List Nat) → FamilyIndex shape → Nat
  | [], _ => 0
  | extent :: rest, (head, tail) => head.val * Family.shapeProduct rest +
      Family.rowMajorOffset rest tail

theorem Family.rowMajorOffset_lt (shape : List Nat) (index : FamilyIndex shape) :
    Family.rowMajorOffset shape index < Family.shapeProduct shape := by
  have shapeProduct_pos : ∀ (shape : List Nat), ∀ index : FamilyIndex shape,
      0 < Family.shapeProduct shape := by
    intro current
    induction current with
    | nil => intro; simp [Family.shapeProduct]
    | cons extent rest ih =>
        intro coordinate
        cases coordinate with
        | mk head tail =>
            exact Nat.mul_pos (Nat.zero_lt_of_lt head.isLt) (ih tail)
  induction shape with
  | nil => simp [Family.rowMajorOffset, Family.shapeProduct]
  | cons extent rest ih =>
      cases index with
      | mk head tail =>
          simp only [Family.rowMajorOffset, Family.shapeProduct]
          have hTail := ih tail
          have hPos := shapeProduct_pos rest tail
          have hStep : head.val * Family.shapeProduct rest +
              Family.shapeProduct rest ≤ extent * Family.shapeProduct rest := by
            rw [← Nat.succ_mul]
            exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt head.isLt)
          exact lt_of_lt_of_le (Nat.add_lt_add_left hTail _) hStep

def Family.pack (shape : List Nat) {element : Type} (values : Array element) :
    Option (Family shape element) :=
  if hSize : values.size = Family.shapeProduct shape then
    some (fun index => values[Family.rowMajorOffset shape index]'(by
      simpa [hSize] using Family.rowMajorOffset_lt shape index))
  else none

def Family.get {shape : List Nat} {element : Type} (value : Family shape element)
    (index : FamilyIndex shape) : element := value index

def Value (backend : SemanticBackend) : WireType → Type
  | .constantInt => Int
  | .int => Int
  | .constantReal => Real
  | .real => Real
  | .constantBool => Bool
  | .bool => Bool
  | .bytes length => Fin length → UInt8
  | .typedBlob name hash => backend.denoteTypedBlob name hash
  | .matrix matrixType => backend.denoteMatrix matrixType
  | .trapdoor trapdoorType => backend.denoteTrapdoor trapdoorType
  | .preimage matrixType => backend.denotePreimage matrixType
  | .family shape element => Family shape (Value backend element)

abbrev DynamicValue (backend : SemanticBackend) := Σ wireType, Value backend wireType

structure Binding (backend : SemanticBackend) where
  wire : WireRef
  value : DynamicValue backend

def DynamicValue.wireType {backend : SemanticBackend} (value : DynamicValue backend) : WireType :=
  value.1

def lookup {backend : SemanticBackend} (values : Array (Binding backend)) (wire : WireRef) :
    Option (DynamicValue backend) :=
  (values.find? (fun binding => binding.wire = wire)).map (fun binding => binding.value)

structure ScopeTrace (backend : SemanticBackend) where
  scope : ScopeId
  occurrence : OccurrencePath
  values : Array (Binding backend)

structure StageTrace (backend : SemanticBackend) where
  stage : Nat
  scopes : Array (ScopeTrace backend)

structure Trace (backend : SemanticBackend) where
  stages : Array (StageTrace backend)

structure WireOccurrence where
  stage : Nat
  path : OccurrencePath
  wire : WireRef
  deriving Repr, DecidableEq

def scopeTraceContains {backend : SemanticBackend} (occurrence : WireOccurrence)
    (scopeTrace : ScopeTrace backend) : Bool :=
  scopeTrace.scope = occurrence.wire.scope && scopeTrace.occurrence = occurrence.path &&
    (lookup scopeTrace.values occurrence.wire).isSome

/- Scope snapshots are stored in reverse node order by `evalScope`.  Searching
   the reversed array therefore selects the first snapshot after the producer
   node, while the stage array remains in increasing stage order. -/
def traceValueAt {backend : SemanticBackend} (trace : Trace backend)
    (occurrence : WireOccurrence) : Option (DynamicValue backend) :=
  (trace.stages.find? (fun stage => stage.stage = occurrence.stage)).bind fun stage =>
    (stage.scopes.reverse.find? (scopeTraceContains occurrence)).bind fun scopeTrace =>
      lookup scopeTrace.values occurrence.wire

def WireOccurrence.sameStatic (left right : WireOccurrence) : Bool :=
  left.stage = right.stage && left.wire = right.wire

/-! An explicit edge from a parent value into one input of a parallel-grid body.
    The edge is structural data, not a semantic identity: it can only be
    followed when the parent argument, child scope, child input, and input-mode
    entry all agree exactly. -/
structure ChildInputHop where
  parentScope : ScopeId
  owner : NodeId
  inputIndex : Nat
  deriving Repr, DecidableEq

structure ParallelOutputHop where
  parentScope : ScopeId
  owner : NodeId
  outputIndex : Nat
  deriving Repr, DecidableEq

structure StructuralValueRoute where
  exits : Array ParallelOutputHop
  enters : Array ChildInputHop
  deriving Repr, DecidableEq

def followChildInputHop? (stage : Stage) (current : WireRef) (hop : ChildInputHop) :
    Option WireRef :=
  if current.scope ≠ hop.parentScope then none else
    match scopeAt stage hop.parentScope with
    | none => none
    | some parent =>
      match parent.nodes[hop.owner]?, parent.nodes[hop.owner]?.bind (fun node =>
        node.arguments[hop.inputIndex]?) with
      | some { payload := .parallelGrid grid, .. }, some argument =>
          if argument ≠ current then none else
            match grid.inputModes[hop.inputIndex]?, scopeAt stage grid.child with
            | some _, some child =>
                child.inputs[hop.inputIndex]?.bind (fun input =>
                  if input.scope = child.id then some input else none)
            | _, _ => none
      | _, _ => none

def followChildInputPath? (stage : Stage) (current : WireRef) :
    List ChildInputHop → Option WireRef
  | [] => some current
  | hop :: rest =>
      (followChildInputHop? stage current hop).bind (fun next =>
        followChildInputPath? stage next rest)

/- A body output is exported only through the exact parent parallel-grid node.
   The family shape and element type are checked here; no raw child wire is
   treated as a family value without this boundary. -/
def followParallelOutputHop? (stage : Stage) (current : WireRef) (hop : ParallelOutputHop) :
    Option WireRef :=
  match scopeAt stage hop.parentScope, scopeAt stage current.scope with
  | some parent, some child =>
      match nodeAt parent hop.owner with
      | some { payload := .parallelGrid grid, outputs := parentOutputs, .. } =>
          if grid.child ≠ current.scope then none else
            match child.outputs[hop.outputIndex]?, wireType? child current with
            | some childOutput, some childType =>
                if childOutput ≠ current ∨
                    (child.outputs.toList.filter (fun output => output = current)).length ≠ 1 then none else
                  match shapeExpression? grid.shape, parentOutputs[hop.outputIndex]? with
                  | some shape, some (.family outputShape outputElement) =>
                      if outputShape = shape ∧ outputElement = childType then
                        some { scope := hop.parentScope, node := hop.owner, port := hop.outputIndex }
                      else none
                  | _, _ => none
            | _, _ => none
      | _ => none
  | _, _ => none

def identityIndexMap (rank : Nat) (indices : Array IndexMapExpr) : Prop :=
  indices.toList = (List.range rank).map IndexMapExpr.axis

noncomputable def followRouteChildInputHop? (stage : Stage) (current : WireRef) (hop : ChildInputHop) :
    Option WireRef :=
  by
    classical
    exact match scopeAt stage hop.parentScope with
    | some parent =>
        match nodeAt parent hop.owner with
        | some { payload := .parallelGrid grid, .. } =>
            match grid.inputModes[hop.inputIndex]? with
            | some { reindex := true, map := some map } =>
                if map.sourceRank = grid.shape.size ∧ map.outputRank = grid.shape.size ∧
                    map.inputIndices.size = grid.shape.size ∧
                      identityIndexMap grid.shape.size map.inputIndices then
                  followChildInputHop? stage current hop
                else none
            | _ => none
        | _ => none
    | none => none

noncomputable def followStructuralValueRoute? (stage : Stage) (current : WireRef) :
    List ParallelOutputHop → List ChildInputHop → Option WireRef
  | [], [] => some current
  | exit :: exits, enters =>
      (followParallelOutputHop? stage current exit).bind (fun next =>
        followStructuralValueRoute? stage next exits enters)
  | [], enter :: enters =>
      (followRouteChildInputHop? stage current enter).bind (fun next =>
        followStructuralValueRoute? stage next [] enters)

def followsChildInputPath (stage : Stage) (start target : WireRef)
    (hops : Array ChildInputHop) : Prop :=
  followChildInputPath? stage start hops.toList = some target

def followsStructuralValueRoute (stage : Stage) (start target : WireRef)
    (route : StructuralValueRoute) : Prop :=
  followStructuralValueRoute? stage start route.exits.toList route.enters.toList = some target

end IR
end Mxx
