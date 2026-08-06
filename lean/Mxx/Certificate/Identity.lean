import Mxx.Ir

namespace Mxx.Certificate

abbrev IntExpr := Mxx.Ir.IntExpr
abbrev MatrixTypeExpr := Mxx.Ir.MatrixTypeExpr
abbrev IntBinaryOp := Mxx.Ir.IntBinaryOp
abbrev IntCompareOp := Mxx.Ir.IntCompareOp

structure StageId where
  name : String
  deriving BEq, DecidableEq, Repr

structure StaticScopeId where
  path : List String
  deriving BEq, DecidableEq, Repr

structure NodeId where
  value : Nat
  deriving BEq, DecidableEq, Repr

structure ProtocolInputId where
  name : String
  deriving BEq, DecidableEq, Repr

structure JointFamilyId where
  name : String
  deriving BEq, DecidableEq, Repr

structure IndexVar where
  slot : Nat
  deriving BEq, DecidableEq, Repr

structure CoreNodeRef where
  stage : StageId
  scope : StaticScopeId
  node : NodeId
  deriving BEq, DecidableEq, Repr

structure CoreWireRef extends CoreNodeRef where
  port : Nat
  deriving BEq, DecidableEq, Repr

structure DefinitionRef where
  stage : StageId
  name : String
  deriving BEq, DecidableEq, Repr

/-- A wire inside a reusable subgraph or loop definition. -/
structure TemplateWireRef where
  definition : DefinitionRef
  bodyScope : StaticScopeId
  node : NodeId
  port : Nat
  deriving BEq, DecidableEq, Repr

structure LoopRef where
  site : CoreNodeRef
  deriving BEq, DecidableEq, Repr

/-- Frozen structural identity of a sequential-loop occurrence.  It is the exact loop node, not
a certificate-provided name. -/
structure SequentialRecurrenceRef where
  site : CoreNodeRef
  deriving BEq, DecidableEq, Repr

inductive RuntimeScalarType where
  | integer
  | boolean
  deriving BEq, DecidableEq, Repr

/-- Typed reference into the immutable runtime-expression table.
This breaks the syntax/identity recursion without creating an executable DAG. -/
structure RuntimeExprRef (type : RuntimeScalarType) where
  id : Nat
  deriving BEq, DecidableEq, Repr

/-- Typed reference into the immutable matrix-expression table. -/
structure MatrixExprRef where
  id : Nat
  deriving BEq, DecidableEq, Repr

inductive InstanceFrame where
  | subgraphCall (callSite : CoreNodeRef)
  | parallelLane (loopSite : CoreNodeRef) (index : RuntimeExprRef .integer)
  | sequentialIteration (loopSite : CoreNodeRef) (index : RuntimeExprRef .integer)
  deriving BEq, DecidableEq, Repr

abbrev InstancePathExpr := List InstanceFrame

/-- The normalized dynamic path of an aggregate value.  Static family and recurrence identifiers
name definition sites only; this path distinguishes their concrete invocations. -/
abbrev AggregateInstancePath := InstancePathExpr

/-- Stable identity of a family aggregate.  Nested family-valued elements retain the complete
identity of their parent rather than being flattened into a synthesized string. -/
inductive FamilyAggregateRef where
  | joint
      (joint : JointFamilyId)
      (outputSlot : Nat)
      (path : AggregateInstancePath)
  | carriedInput (carriedSlot : Nat)
  | recurrenceResult
      (recurrence : SequentialRecurrenceRef)
      (path : AggregateInstancePath)
      (slot : Nat)
  | familyElement
      (parent : FamilyAggregateRef)
      (index : RuntimeExprRef .integer)
  deriving BEq, DecidableEq, Repr

/-- A dynamic occurrence of a statically named recurrence. -/
structure SequentialRecurrenceInstanceRef where
  recurrence : SequentialRecurrenceRef
  path : AggregateInstancePath
  deriving BEq, DecidableEq, Repr

private def InstanceFrame.isParallelLane : InstanceFrame → Bool
  | .parallelLane .. => true
  | .subgraphCall _ | .sequentialIteration .. => false

/-- A materialized lane of a uniform parallel-family template may append only `parallelLane`
frames to the analyzer-owned recurrence occurrence.  Frozen sites and the complete base path
remain exact; sequential iterations and subgraph calls are never erased. -/
def SequentialRecurrenceInstanceRef.isLaneUniformInstantiationOf
    (reference base : SequentialRecurrenceInstanceRef) : Bool :=
  if reference.recurrence != base.recurrence then false
  else
    let rec checkPath : AggregateInstancePath → AggregateInstancePath → Bool
      | [], suffix => suffix.all InstanceFrame.isParallelLane
      | baseHead :: baseTail, referenceHead :: referenceTail =>
          baseHead == referenceHead && checkPath baseTail referenceTail
      | _ :: _, [] => false
    checkPath base.path reference.path

/-- Return the unique analyzer-owned entry whose recurrence identity is the lane-uniform base of
`reference`.  Ambiguous prefixes fail closed; callers must not choose a longest prefix. -/
def uniqueLaneUniformRecurrenceMatch?
    {α : Type}
    (reference : SequentialRecurrenceInstanceRef)
    (entries : List α)
    (identity : α → SequentialRecurrenceInstanceRef) : Option α :=
  match entries.filter (fun entry => reference.isLaneUniformInstantiationOf (identity entry)) with
  | [entry] => some entry
  | _ => none

def SequentialRecurrenceInstanceRef.appendPath
    (reference : SequentialRecurrenceInstanceRef)
    (suffix : InstancePathExpr) : SequentialRecurrenceInstanceRef :=
  { reference with path := reference.path ++ suffix }

private def laneMatchSite (node : Nat) : CoreNodeRef := {
  stage := ⟨"lane-match"⟩
  scope := ⟨[]⟩
  node := ⟨node⟩
}

private def laneMatchIndex (id : Nat) : RuntimeExprRef .integer := ⟨id⟩

private def laneMatchBase : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨laneMatchSite 8⟩
  path := []
}

private def laneMatchReference : SequentialRecurrenceInstanceRef :=
  laneMatchBase.appendPath [
    .parallelLane (laneMatchSite 29) (laneMatchIndex 83),
    .parallelLane (laneMatchSite 31) (laneMatchIndex 85),
    .parallelLane (laneMatchSite 32) (laneMatchIndex 86),
    .parallelLane (laneMatchSite 33) (laneMatchIndex 90)
  ]

example : laneMatchReference.isLaneUniformInstantiationOf laneMatchBase = true := by
  rfl

example : uniqueLaneUniformRecurrenceMatch? laneMatchReference [laneMatchBase] id =
    some laneMatchBase := by
  rfl

example :
    let sequential := laneMatchBase.appendPath [
      .sequentialIteration (laneMatchSite 29) (laneMatchIndex 83)
    ]
    uniqueLaneUniformRecurrenceMatch? sequential [laneMatchBase] id = none := by
  rfl

example :
    let wrongSite : SequentialRecurrenceInstanceRef := {
      recurrence := ⟨laneMatchSite 9⟩
      path := laneMatchReference.path
    }
    uniqueLaneUniformRecurrenceMatch? wrongSite [laneMatchBase] id = none := by
  rfl

example :
    let nonPrefix : SequentialRecurrenceInstanceRef := {
      laneMatchBase with
      path := [.parallelLane (laneMatchSite 30) (laneMatchIndex 84)]
    }
    uniqueLaneUniformRecurrenceMatch? laneMatchReference [nonPrefix] id = none := by
  rfl

example :
    let longerBase := laneMatchBase.appendPath [
      .parallelLane (laneMatchSite 29) (laneMatchIndex 83)
    ]
    uniqueLaneUniformRecurrenceMatch? laneMatchReference [laneMatchBase, longerBase] id = none := by
  rfl

def FamilyAggregateRef.appendPath
    (suffix : InstancePathExpr) : FamilyAggregateRef → FamilyAggregateRef
  | .joint familyId outputSlot instancePath =>
      .joint familyId outputSlot (instancePath ++ suffix)
  | .carriedInput slot => .carriedInput slot
  | .recurrenceResult recurrence path slot =>
      .recurrenceResult recurrence (path ++ suffix) slot
  | .familyElement parent index => .familyElement (parent.appendPath suffix) index

inductive ValueInstanceRef where
  | protocolInput (input : ProtocolInputId)
  | concrete (wire : CoreWireRef)
  | template (wire : TemplateWireRef)
  | instantiatedTemplate (wire : TemplateWireRef) (path : InstancePathExpr)
  | familyElement
      (aggregate : FamilyAggregateRef)
      (index : RuntimeExprRef .integer)
  | recurrenceResult (recurrence : SequentialRecurrenceInstanceRef) (slot : Nat)
  deriving BEq, DecidableEq, Repr

/-- Canonical stable identity for a frozen wire. Root wires are concrete; a non-root static scope
names the reusable definition followed by its nested body scope. -/
def ValueInstanceRef.ofCoreWire (wire : CoreWireRef) : ValueInstanceRef :=
  match wire.scope.path with
  | [] => .concrete wire
  | definitionName :: bodyScope => .template {
      definition := { stage := wire.stage, name := definitionName }
      bodyScope := ⟨bodyScope⟩
      node := wire.node
      port := wire.port
    }

structure MatrixInstanceRef where
  value : ValueInstanceRef
  type : MatrixTypeExpr

end Mxx.Certificate
