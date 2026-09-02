import Mathlib

namespace Mxx
namespace IR

noncomputable section

abbrev ScopeId := Nat
abbrev NodeId := Nat
abbrev Port := Nat

structure Rational where
  numerator : Int
  denominator : Int
  deriving Repr, DecidableEq

inductive StructuralIntExpr where
  | literal (value : Int)
  | structuralSlot (slot : Nat)
  | add (left right : StructuralIntExpr)
  | subtract (left right : StructuralIntExpr)
  | multiply (left right : StructuralIntExpr)
  | exactDivide (left right : StructuralIntExpr)
  | roundDivide (left right : StructuralIntExpr)
  | log2Ceil (value : StructuralIntExpr)
  deriving Repr, DecidableEq

inductive IndexMapExpr where
  | literal (value : Int)
  | axis (index : Nat)
  | structuralSlot (slot : Nat)
  | add (left right : IndexMapExpr)
  | sub (left right : IndexMapExpr)
  | mul (left right : IndexMapExpr)
  | divide (left right : IndexMapExpr)
  | remainder (left right : IndexMapExpr)
  | equal (left right : IndexMapExpr)
  | less (left right : IndexMapExpr)
  | lessEqual (left right : IndexMapExpr)
  | log2Ceil (value : IndexMapExpr)
  | select (selector : IndexMapExpr) (branches : Array IndexMapExpr)
  deriving Repr

noncomputable instance : DecidableEq IndexMapExpr := Classical.decEq _

inductive RealExpr where
  | literal (value : Rational)
  | fromInt (value : StructuralIntExpr)
  | add (left right : RealExpr)
  | subtract (left right : RealExpr)
  | multiply (left right : RealExpr)
  | divide (left right : RealExpr)
  | sqrt (value : RealExpr)
  deriving Repr, DecidableEq

structure MatrixType where
  modulus : Int
  ringDimension : Nat
  rows : Nat
  columns : Nat
  deriving Repr, DecidableEq

def MatrixType.Valid (t : MatrixType) : Prop :=
  1 < t.modulus ∧ 0 < t.ringDimension ∧ 0 < t.rows ∧ 0 < t.columns

structure TrapdoorType where
  matrix : MatrixType
  sigma : RealExpr
  gadgetBase : StructuralIntExpr
  digitCount : StructuralIntExpr
  preimageMaxCoefficientBound : StructuralIntExpr
  deriving Repr, DecidableEq

inductive WireType where
  | constantInt | constantReal | constantBool
  | int | real | bool
  | bytes (length : Nat)
  | typedBlob (typeName : String) (schemaHash : List UInt8)
  | matrix (matrixType : MatrixType)
  | trapdoor (trapdoorType : TrapdoorType)
  | preimage (matrixType : MatrixType)
  | family (shape : List Nat) (element : WireType)
  deriving Repr, DecidableEq

structure WireRef where
  scope : ScopeId
  node : NodeId
  port : Port
  deriving Repr, DecidableEq

structure IntRange where
  start : StructuralIntExpr
  stop : StructuralIntExpr
  deriving Repr, DecidableEq

structure IndexMap where
  sourceRank : Nat
  outputRank : Nat
  inputIndices : Array IndexMapExpr
  deriving Repr, DecidableEq

structure GridInputMode where
  reindex : Bool
  map : Option IndexMap
  deriving Repr, DecidableEq

structure SubgraphPayload where
  child : ScopeId
  definition : String
  bindings : Array (String × StructuralIntExpr)
  canonicalInputExclusiveUppers : Array (Option Nat)
  deriving Repr, DecidableEq

structure LoopPayload where
  child : ScopeId
  count : StructuralIntExpr
  indexSlot : Nat
  bindings : Array (String × StructuralIntExpr)
  carriedCount : Nat
  deriving Repr, DecidableEq

structure GridPayload where
  child : ScopeId
  shape : Array StructuralIntExpr
  indexSlots : Array Nat
  bindings : Array (String × StructuralIntExpr)
  inputModes : Array GridInputMode
  deriving Repr, DecidableEq

inductive MatrixLiteral where
  | zero | identity
  | unitRow (index : StructuralIntExpr)
  | unitColumn (index : StructuralIntExpr)
  | gadget (base : StructuralIntExpr) (small : Bool)
  | powerOfBase (base exponent : StructuralIntExpr)
  | rotation (exponent : StructuralIntExpr)
  | polynomial (coefficients : Array StructuralIntExpr)
  deriving Repr, DecidableEq

inductive Confidentiality where
  | Public
  | Private
  deriving Repr, DecidableEq

structure ArtifactInput where
  index : Nat
  name : String
  confidentiality : Confidentiality
  deriving Repr, DecidableEq

inductive IntBinaryOp where | add | subtract | multiply | divide | remainder
  deriving Repr, DecidableEq
inductive IntCompareOp where | equal | less | lessEqual
  deriving Repr, DecidableEq
inductive RealBinaryOp where | add | subtract | multiply | divide
  deriving Repr, DecidableEq
inductive MatrixBinaryOp where | add | subtract | multiply
  deriving Repr, DecidableEq
inductive PreimageBinaryOp where | add | rightMultiplyExact | composeExactDecomposition
  deriving Repr, DecidableEq
inductive ConcatAxis where | rows | columns | diagonal
  deriving Repr, DecidableEq

inductive NodePayload where
  | input (index : Nat)
  | artifactInput (input : ArtifactInput)
  | constantInt (value : Int)
  | evaluateInt (value : StructuralIntExpr)
  | constantReal (value : RealExpr)
  | constantBool (value : Bool)
  | constantMatrix (matrixType : MatrixType) (literal : MatrixLiteral)
  | gadgetTrapdoor (matrixType : MatrixType) (base : StructuralIntExpr)
  | trapdoorPublic
  | intBinary (op : IntBinaryOp)
  | intCompare (op : IntCompareOp)
  | bitExtract (bit : StructuralIntExpr)
  | intToReal | boolToInt
  | realBinary (op : RealBinaryOp) | realSqrt
  | matrixBinary (op : MatrixBinaryOp)
  | matrixMulAccumulate (coefficients : Array StructuralIntExpr) (hasBias : Bool)
  | matrixNegate | matrixScale (scalar : StructuralIntExpr) | transpose
  | slice (rows columns : Option IntRange) | tensor
  | concat (axis : ConcatAxis)
  | uniformResidueSample (matrixType : MatrixType)
  | uniformIntervalSample (matrixType : MatrixType) (range : IntRange)
  | gaussianSample (matrixType : MatrixType) (sigma : RealExpr) (bound : StructuralIntExpr)
  | hashSample (matrixType : MatrixType) (tagPrefix : List UInt8)
      (tagExpressions tagDecimalExpressions tagU64LEExpressions : Array StructuralIntExpr)
  | trapdoorSample (matrixType : MatrixType) (sigma : RealExpr)
      (base digits bound : StructuralIntExpr)
  | preimageSample (matrixType : MatrixType) (bound : StructuralIntExpr)
  | applyPreimage | materializePreimageExact
  | preimageBinary (op : PreimageBinaryOp) | preimageConcatColumns
  | familyPreimageSample (matrixType : MatrixType) (bound : StructuralIntExpr)
  | gadgetDecompose (base : StructuralIntExpr) (small : Bool) (digits : StructuralIntExpr)
  | decompositionEntry (row column : StructuralIntExpr)
  | extractCoefficient (position : StructuralIntExpr) (canonicalUpper : Option Nat)
  | liftIntegerToConstantPolynomial (matrixType : MatrixType)
  | thresholdDecode (plaintextModulus length : StructuralIntExpr) (outputBool : Bool)
  | crtRecompose (plaintextModuli reconstructionCoefficients : Array StructuralIntExpr)
  | packPolynomialCoefficients (matrixType : MatrixType) (coefficientBits : StructuralIntExpr)
  | subgraphCall (payload : SubgraphPayload)
  | sequentialLoop (payload : LoopPayload)
  | familyPack (shape : Array StructuralIntExpr)
  | familyGetStatic (indices : Array IndexMapExpr)
  | familyGetDynamic (rank : Nat)
  | familySelectAxis (axis : Nat)
  | familyReindex (outputShape : Array StructuralIntExpr) (map : IndexMap)
  | familyGather (outputShape : Array StructuralIntExpr) (inputRank : Nat)
  | parallelGrid (payload : GridPayload)
  | select (count : StructuralIntExpr)
  deriving Repr, DecidableEq

structure Node where
  payload : NodePayload
  arguments : Array WireRef
  outputs : Array WireType
  deriving Repr, DecidableEq

inductive StructuralSlotKind where
  | sequentialIteration
  | gridAxis (axis : Nat)
  deriving Repr, DecidableEq

structure StructuralSlotDecl where
  slot : Nat
  kind : StructuralSlotKind
  upperBound : Int
  deriving Repr, DecidableEq

structure Scope where
  id : ScopeId
  structuralSlots : Array StructuralSlotDecl
  nodes : Array Node
  inputs : Array WireRef
  outputs : Array WireRef
  deriving Repr, DecidableEq

structure NamedOutput where
  name : String
  wire : WireRef
  deriving Repr, DecidableEq

structure Stage where
  name : String
  bindings : Array (String × Int)
  scopes : Array Scope
  root : ScopeId
  namedOutputs : Array NamedOutput
  deriving Repr, DecidableEq

structure ArtifactLink where
  consumerStage : Nat
  consumer : WireRef
  argument : Nat
  consumerArtifact : String
  consumerConfidentiality : Confidentiality
  consumerType : WireType
  producerStage : Nat
  producer : WireRef
  producerArtifact : String
  producerConfidentiality : Confidentiality
  producerType : WireType
  deriving Repr, DecidableEq

structure SemanticIdentity where
  irVersion : Nat
  linkedProgramSha256 : List UInt8
  deriving Repr, DecidableEq

structure ProgramData where
  identity : SemanticIdentity
  stages : Array Stage
  artifactLinks : Array ArtifactLink
  deriving Repr, DecidableEq

end
end IR
end Mxx
