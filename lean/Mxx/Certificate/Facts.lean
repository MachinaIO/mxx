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
  | recurrenceResult (recurrence : FactRecurrenceInstanceRef) (path : BoundFactPath)
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
      (recurrence : FactRecurrenceInstanceRef)
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

structure FactRecurrence where
  loop : LoopRef
  count : IntExpr
  carriedArity : Nat
  initial : Vector ValueFact carriedArity
  bodyInputs : Vector TemplateWireRef carriedArity
  bodyOutputs : Vector ValueFactTemplate carriedArity
  invariantInputs : List ValueFact
  iterationVariable : IndexVar

end Mxx.Certificate
