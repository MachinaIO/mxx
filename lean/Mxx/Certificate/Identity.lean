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

structure FactRecurrenceRef where
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
      (recurrence : FactRecurrenceRef)
      (path : AggregateInstancePath)
      (slot : Nat)
  | familyElement
      (parent : FamilyAggregateRef)
      (index : RuntimeExprRef .integer)
  deriving BEq, DecidableEq, Repr

/-- A dynamic occurrence of a statically named recurrence. -/
structure FactRecurrenceInstanceRef where
  recurrence : FactRecurrenceRef
  path : AggregateInstancePath
  deriving BEq, DecidableEq, Repr

def FamilyAggregateRef.appendPath
    (suffix : InstancePathExpr) : FamilyAggregateRef → FamilyAggregateRef
  | .joint familyId outputSlot instancePath =>
      .joint familyId outputSlot (instancePath ++ suffix)
  | .carriedInput slot => .carriedInput slot
  | .recurrenceResult recurrence path slot =>
      .recurrenceResult recurrence (path ++ suffix) slot
  | .familyElement parent index => .familyElement (parent.appendPath suffix) index

def FactRecurrenceInstanceRef.appendPath
    (reference : FactRecurrenceInstanceRef)
    (suffix : InstancePathExpr) : FactRecurrenceInstanceRef :=
  { reference with path := reference.path ++ suffix }

inductive ValueInstanceRef where
  | protocolInput (input : ProtocolInputId)
  | concrete (wire : CoreWireRef)
  | template (wire : TemplateWireRef)
  | instantiatedTemplate (wire : TemplateWireRef) (path : InstancePathExpr)
  | familyElement
      (aggregate : FamilyAggregateRef)
      (index : RuntimeExprRef .integer)
  | recurrenceResult (recurrence : FactRecurrenceInstanceRef) (slot : Nat)
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
