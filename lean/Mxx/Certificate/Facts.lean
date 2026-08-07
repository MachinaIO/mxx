import Mxx.Certificate.Syntax

namespace Mxx.Certificate

inductive BoundExpr where
  | constant (value : Nat)
  | parameter (value : IntExpr)
  | add (left right : BoundExpr)
  | multiply (left right : BoundExpr)
  | maximum (left right : BoundExpr)
  | absolute (value : IntExpr)
  | floorDivide (value : BoundExpr) (positiveDivisor : Nat)
  | matrixProduct
      (ringDimension innerDimension : IntExpr)
      (left right : BoundExpr)
  | minimum (left right : BoundExpr)
  | recurrenceResult (recurrence : SequentialRecurrenceInstanceRef) (path : BoundFactPath)
  /-- Analyzer-only placeholder for a bound inside the previous carried state. -/
  | carriedInput (path : BoundFactPath)

/-- Signed integer-bound expressions.  Unlike `IntExpr`, these may refer to analyzer-derived
matrix bounds and checked sequential-recurrence results. -/
inductive IntBoundExpr where
  | integer (value : IntExpr)
  | natural (value : BoundExpr)
  | negate (value : IntBoundExpr)
  | add (left right : IntBoundExpr)
  | subtract (left right : IntBoundExpr)
  | multiply (left right : IntBoundExpr)
  | divide (left right : IntBoundExpr)
  | minimum (left right : IntBoundExpr)
  | maximum (left right : IntBoundExpr)
  | carriedInput (path : IntBoundFactPath)
  | recurrenceResult
      (recurrence : SequentialRecurrenceInstanceRef)
      (path : IntBoundFactPath)

structure BoundedMatrixExpr where
  expression : MatrixExpr
  normBound : BoundExpr

inductive SignalProductMode where
  | ordinaryMatrixProduct
  | leftPolynomialScalarBroadcast
  | rightPolynomialScalarBroadcast
  | swappedRowVectorScalarProduct
  deriving BEq, DecidableEq, Repr

structure SignalTerm where
  coefficient : BoundedMatrixExpr
  basis : MatrixExpr
  mode : SignalProductMode

structure AffineForm where
  terms : List SignalTerm
  noiseBound : BoundExpr

inductive MatrixPrimaryForm where
  | exact (expression : MatrixExpr)
  | affine (form : AffineForm)

inductive MatrixRelationKind where
  | preimage
  | gadgetDecomposition
  deriving BEq, DecidableEq, Repr

inductive MatrixRelation where
  | preimage
      (subject : ValueInstanceRef)
      (source : MatrixInstanceRef)
      (target : MatrixInstanceRef)
      (trapdoor : ValueInstanceRef)
  | gadgetDecomposition
      (subject : ValueInstanceRef)
      (target : MatrixInstanceRef)
      (base digitCount : IntExpr)

/-- Change only the matrix value that owns a retained sampler relation. Source, target,
trapdoor, decomposition base, and digit count remain exact provenance from the original node. -/
def MatrixRelation.retargetSubject
    (subject : ValueInstanceRef) : MatrixRelation → MatrixRelation
  | .preimage _ source target trapdoor => .preimage subject source target trapdoor
  | .gadgetDecomposition _ target base digitCount =>
      .gadgetDecomposition subject target base digitCount

/-- Representation of raw stored coefficients.  A centered norm never establishes this field. -/
inductive CoefficientRepresentation where
  | unknown
  | canonicalResidues (modulus : IntExpr)
  deriving BEq, DecidableEq

structure SignalTermSchema where
  coefficientType : MatrixTypeExpr
  basisType : MatrixTypeExpr
  mode : SignalProductMode
  deriving BEq, DecidableEq

inductive MatrixPrimarySchema where
  | exact
  | affine (terms : List SignalTermSchema)
  deriving BEq, DecidableEq

inductive ValueFactSchema where
  | matrix
      (type : MatrixTypeExpr)
      (primary : MatrixPrimarySchema)
      (relations : List MatrixRelationKind)
      (coefficientRepresentation : CoefficientRepresentation)
  | trapdoor
  | integer
  | boolean
  | bytes
  | family (count : IntExpr) (element : ValueFactSchema)
  deriving BEq, DecidableEq

structure MatrixFact where
  subject : ValueInstanceRef
  primary : MatrixPrimaryForm
  relations : List MatrixRelation
  totalNormBound : BoundExpr
  coefficientRepresentation : CoefficientRepresentation := .unknown

/-- A closed, parameter-only range requirement for a dynamic family access under a structural
loop. The checker evaluates it exactly in Phase B. -/
structure LoopFamilyRangeRequirement where
  loopCount : IntExpr
  offset : Nat
  familyCount : IntExpr
  deriving BEq, DecidableEq

/-- Exact alias discovered from a body-local matrix fact whose primary form is exact. It is an
analyzer-owned bridge from a producer-lane local subject to its preserved external provenance. -/
structure MatrixAliasTemplate where
  subject : TemplateWireRef
  subjectType : MatrixTypeExpr
  exactTarget : MatrixExpr

/-- One mechanically checked use of the GGH recurrence rewrite. Every identity in this key is
fixed by frozen IR and analyzer output; a caller cannot construct it as semantic evidence. -/
structure RecurrenceBasisAlignmentKey where
  recurrence : SequentialRecurrenceInstanceRef
  carriedSlot : Nat
  signalTerm : Nat
  multiplication : CoreNodeRef
  rightOperandSubject : ValueInstanceRef
  relationSubject : ValueInstanceRef
  relationSource : MatrixInstanceRef
  relationTarget : MatrixInstanceRef
  sourceOrigin : IndexedMatrixOrigin
  successorOrigin : IndexedMatrixOrigin
  coefficientType : MatrixTypeExpr
  basisType : MatrixTypeExpr
  rightOperandType : MatrixTypeExpr
  productMode : SignalProductMode
  deriving BEq, DecidableEq

/-- Analyzer output needed to instantiate the one-step GGH hard-bound transition. This is
diagnostic data only; soundness is reconstructed from execution and the frozen trace. -/
structure RecurrenceBasisAlignmentSummary where
  key : RecurrenceBasisAlignmentKey
  targetCoefficient : MatrixExpr
  targetCoefficientBound : BoundExpr
  targetNoiseBound : BoundExpr
  rightTotalBound : BoundExpr
  successorBasisBound : BoundExpr
  rangeRequirements : List LoopFamilyRangeRequirement

structure TrapdoorFact where
  privatePort : ValueInstanceRef
  publicPort : ValueInstanceRef
  publicMatrix : MatrixExpr

structure IntegerFact where
  expression : RuntimeExpr .integer
  lower : IntBoundExpr
  upper : IntBoundExpr

structure BooleanFact where
  expression : RuntimeExpr .boolean

/-- An aggregate family carries its recursive schema.  The exact element template is resolved
from its analyzer-owned joint-family or recurrence table using `aggregate`. -/
structure FamilyFact where
  aggregate : FamilyAggregateRef
  count : IntExpr
  elementSchema : ValueFactSchema

inductive ValueFact where
  | matrix (fact : MatrixFact)
  | trapdoor (fact : TrapdoorFact)
  | integer (fact : IntegerFact)
  | boolean (fact : BooleanFact)
  | bytes (wire : ValueInstanceRef)
  | family (fact : FamilyFact)

/-- A normalized body fact whose identities may refer to template wires.

The analyzer constructs `fact` from proved local rules and separately checks that
its shape is exactly `schema`; a certificate cannot provide either field. Keeping
the full fact is necessary to instantiate signal terms, bounds, and relations at
each family lane or sequential-loop iteration without unrolling the loop. -/
structure ValueFactTemplate where
  fact : ValueFact
  schema : ValueFactSchema

structure JointFamilyFact where
  id : JointFamilyId
  count : IntExpr
  indexVariable : IndexVar
  outputFamilies : List CoreWireRef
  outputArity : Nat
  elementTuple : Vector ValueFactTemplate outputArity

/-- One loop-invariant argument and its complete typed fact.  The template is retained rather
than only the raw fact so a nested body trace can reconstruct the same analyzer seed table
without guessing a matrix type. -/
structure InvariantInputFact where
  wire : CoreWireRef
  template : ValueFactTemplate

/-- A dynamic family-index interval discovered while analyzing one sequential-loop body. The
interval is still symbolic here because it may read the previous carried state; construction
translates it to typed recurrence paths before Phase B evaluates each step. -/
structure SequentialBodyRangeRequirement where
  site : CoreNodeRef
  lower : IntBoundExpr
  upper : IntBoundExpr
  familyCount : IntExpr

structure SequentialRecurrenceSource where
  loop : LoopRef
  count : IntExpr
  carriedArity : Nat
  /-- Typed facts for the actual initial carried wires. Keeping the owning schema here is
  necessary for matrix-carried loops; a bare `ValueFact.matrix` does not contain its matrix type. -/
  initial : Vector ValueFactTemplate carriedArity
  bodyInputs : Vector TemplateWireRef carriedArity
  bodyOutputs : Vector ValueFactTemplate carriedArity
  /-- Analyzer-owned element templates for family aggregates reachable from carried facts. -/
  familyElementTemplates : List (FamilyAggregateRef × ValueFactTemplate) := []
  bodyRangeRequirements : List SequentialBodyRangeRequirement := []
  invariantInputs : List InvariantInputFact
  iterationVariable : IndexVar

end Mxx.Certificate
