import Lean.Data.Json

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.SchemaV1

open Lean

/-! Typed wire DTOs for the frozen Rust certificate schema version 1. These definitions only
    describe decoded statement data. They do not construct `Core.Cert`, prove validity, or accept
    a certificate. -/

structure ExpressionRef where row : Nat deriving DecidableEq, Repr
structure ProgramRef where row : Nat deriving DecidableEq, Repr
structure SourceRef where row : Nat deriving DecidableEq, Repr
structure EventRef where row : Nat deriving DecidableEq, Repr
structure IndexUseRef where row : Nat deriving DecidableEq, Repr
structure SliceGroupRef where row : Nat deriving DecidableEq, Repr

structure Range where
  minimum : Nat
  maximumExclusive : Nat
deriving DecidableEq, Repr

inductive ValueType where
  | bool
  | int
  | real
  | bytes
  | matrix (modulus : String) (ringDimension rows columns : Nat)
  | trapdoor
deriving DecidableEq, Repr

inductive ConstantValue where
  | bool (value : Bool)
  | int (value : String)
  | real (value : String)
  | bytes (value : List Nat)
deriving DecidableEq, Repr

structure Constant where
  valueType : ValueType
  value : ConstantValue
deriving DecidableEq, Repr

inductive MatrixConstantKind where
  | zero
  | identity
  | unitRow (index : Nat)
  | unitColumn (index : Nat)
  | gadget (base : Nat) (small : Bool)
  | powerOfBase (base exponent : String)
  | rotation (exponent : Nat)
  | polynomial (coefficients : List String)
deriving DecidableEq, Repr

structure Artifact where
  definition : String
  version : Nat
  confidentiality : Nat
  valueType : ValueType
  layout : String
  domain : Option (Nat × Nat)
deriving DecidableEq, Repr

structure SampleDescriptor where
  definition : String
  parameters : List Nat
  outputType : ValueType
  gadgetBase : Option String
  digitCount : Option Nat
  decomposition : Option String
deriving DecidableEq, Repr

structure SourceIdentity where
  definition : String
  sampleEvent : Option EventRef
  outputRole : String
  artifact : Option Artifact
  valueType : ValueType
  coordinates : List Nat
  matrixConstant : Option MatrixConstantKind
deriving DecidableEq, Repr

structure FamilySourceIdentity where
  definition : String
  invocation : String
  elementType : ValueType
  domain : Nat × Nat
  artifact : Option Artifact
deriving DecidableEq, Repr

inductive Scope where
  | root
  | subgraph (canonicalName : String)
  | parallelBody (parent : Scope) (owner : Nat)
  | sequentialBody (parent : Scope) (owner : Nat)
deriving DecidableEq, Repr

structure ObservedWire where
  stage : String
  definition : Scope
  path : Nat
  node : Nat
  port : Nat
deriving DecidableEq, Repr

structure ObservedProducer where
  consumer : ObservedWire
  consumerInput : String
  producerStage : String
  producerOutput : String
  producer : ObservedWire
deriving DecidableEq, Repr

inductive ObservedSourceAccess where
  | declaredProtocolInput (owner : ObservedWire) (input : String)
  | unboundOccurrenceInput (owner : ObservedWire)
  | producerArtifact (producer : ObservedProducer)
deriving DecidableEq, Repr

structure SignedRange where
  minimum : String
  maxExclusive : String
deriving DecidableEq, Repr

inductive RawCoefficientClass where
  | exactZero
  | finite (maximumAbsoluteCoefficient : String)
  | large
deriving DecidableEq, Repr

structure RawValueContract where
  signedRange : Option SignedRange
  coefficientClass : Option RawCoefficientClass
  canonicalCoefficientExclusiveUpper : Option String
  polynomialSupportUpper : Option Nat
deriving DecidableEq, Repr

inductive SourceRow where
  | constant (value : Constant)
  | direct (identity : SourceIdentity) (access : Option ObservedSourceAccess)
      (contract : Option RawValueContract)
  | family (identity : FamilySourceIdentity) (contract : Option RawValueContract)
deriving DecidableEq, Repr

inductive HashVariant where
  | plain
  | decomposed
  | smallDecomposed
deriving DecidableEq, Repr

inductive HashDefinition where
  | mxxPolynomialHash
deriving DecidableEq, Repr

inductive SamplerOperation where
  | uniformResidue (output : ValueType)
  | uniformInterval (output : ValueType) (minimum maximum : String)
  | gaussian (output : ValueType) (sigma maxCoefficientBound : String)
  | hash (output : ValueType) (variant : HashVariant) (tagPrefix : List Nat)
      (tagExpressions tagDecimalExpressions tagU64LeExpressions : List ExpressionRef)
      (base digitCount : Option Nat)
  | trapdoor (output : ValueType) (sigma : String) (gadgetBase digitCount : Nat)
      (preimageMaxCoefficientBound : String)
  | preimage (output : ValueType) (maxCoefficientBound : String)
deriving DecidableEq, Repr

inductive StatementScope where
  | closed (root : ExpressionRef)
  | program (program : ProgramRef)
deriving DecidableEq, Repr

inductive EventRow where
  | sample (owner : ObservedWire) (descriptor : SampleDescriptor)
      (contract : Option RawValueContract)
  | sampler (owner : ObservedWire) (operation : SamplerOperation)
      (contract : Option RawValueContract)
  | gadgetDecompose (scope : StatementScope) (expression : ExpressionRef)
      (output : ValueType) (base : Nat) (small : Bool) (digitCount : Nat)
      (input : ExpressionRef) (contract : Option RawValueContract)
deriving DecidableEq, Repr

inductive ExpressionSource where
  | direct (source : SourceRef)
  | family (source : SourceRef) (selector : ExpressionRef)
deriving DecidableEq, Repr

inductive ExpressionEventOperator where
  | sample (event : EventRef)
  | sampler (event : EventRef)
  | gadgetDecompose (events : List EventRef)
deriving DecidableEq, Repr

inductive ScalarOperation where
  | add | subtract | multiply | divide | remainder | negate
  | equal | less | lessEqual | boolToInt | intToReal
  | realAdd | realSubtract | realMultiply | realDivide | realSqrt
  | thresholdDecode (plaintextModulus : String) (length : Nat) (outputBool : Bool)
  | bit (position : Nat)
  | slice (start endExclusive : Nat)
  | hash (tag : String) (dynamicTags : List Nat)
  | extractCoefficient (row column : Nat)
  | liftConstantPolynomial (output : ValueType) (coefficientBits : Nat)
deriving DecidableEq, Repr

structure Layout where
  name : String
  rowStride : Nat
  columnStride : Nat
deriving DecidableEq, Repr

inductive MatrixOperation where
  | add | subtract | multiply | negate | scale | transpose
  | ringAutomorphism (index : Nat)
  | slice (rowStart rowEndExclusive columnStart columnEndExclusive : Nat) (layout : Layout)
  | indexedSlice (output : ValueType) (layout : Layout)
  | view (output : ValueType) (layout : Layout)
  | concat (axis : Nat) (output : ValueType) (layout : Layout)
  | tensor (output : ValueType) (leftLayout rightLayout outputLayout : Layout)
  | crtRecompose (plaintextModuli reconstructionCoefficients : List String)
      (output : ValueType)
  | extractCoefficient (row column : Nat)
  | liftConstantPolynomial (output : ValueType) (coefficientBits : Nat)
deriving DecidableEq, Repr

inductive TransformOperation where
  | gadgetDecompose (output : ValueType) (base : Nat) (small : Bool) (digitCount : Nat)
  | packPolynomialCoefficients (output : ValueType) (coefficientBits : Nat)
deriving DecidableEq, Repr

inductive TrapdoorOperation where
  | generate (descriptor : String) (parameters : List Nat)
      (pairedPublicEvent : Option EventRef) (pairedPublicOutputRole : String)
  | transform (descriptor : String) (output : ValueType) (parameters : List Nat)
deriving DecidableEq, Repr

inductive StableOperator where
  | argument (position : Nat) (valueType : ValueType)
  | constant (value : Constant)
  | source (identity : SourceIdentity)
  | sample (event : Option EventRef) (descriptor : SampleDescriptor)
  | sampler (event : Option EventRef) (operation : SamplerOperation)
  | deterministicHash (definition : HashDefinition) (version keyByteLength : Nat)
      (output : ValueType) (tagPrefix : List Nat) (binaryTagCount decimalTagCount : Nat)
      (u64LeTagCount dynamicTagCount : Nat)
  | opaqueFamilyElement (identity : FamilySourceIdentity)
  | indexMap (definition : Nat) (parameters : List Nat)
  | explicitElement (domain : Nat × Nat) (elementType : ValueType)
  | programCall
  | transform (operation : TransformOperation)
  | extractCoefficient (position : Nat) (canonicalInputExclusiveUpper : Option String)
  | scalar (operation : ScalarOperation)
  | matrix (operation : MatrixOperation)
  | trapdoor (operation : TrapdoorOperation)
deriving DecidableEq, Repr

inductive ExpressionOperator where
  | stable (operator : StableOperator)
  | event (operator : ExpressionEventOperator)
deriving DecidableEq, Repr

inductive ExpressionDescriptor where
  | source (source : ExpressionSource)
  | event (operator : ExpressionEventOperator)
  | operation (operator : ExpressionOperator) (valueType : ValueType)
deriving DecidableEq, Repr

structure ExpressionRow where
  descriptor : ExpressionDescriptor
  inputs : List ExpressionRef
  program : Option ProgramRef
deriving DecidableEq, Repr

structure ProgramInput where
  valueType : ValueType
  trustedIndexRange : Option Range
deriving DecidableEq, Repr

structure Family where
  domain : Range
  elementType : ValueType
  reducible : Bool
  artifact : Option Artifact
deriving DecidableEq, Repr

structure ProgramRow where
  signature : List ProgramInput
  output : ValueType
  family : Option Family
  root : ExpressionRef
deriving DecidableEq, Repr

inductive PlanRef where
  | expression (row : ExpressionRef)
  | family (row : ProgramRef)
deriving DecidableEq, Repr

structure ObservedOccurrence where
  definition : Scope
  path : Nat
deriving DecidableEq, Repr

inductive FrontierAxis where
  | argument (owner : ObservedOccurrence) (expression : PlanRef) (position : Nat)
      (domain : Nat × Nat)
  | extractedCoefficient (owner : ObservedOccurrence) (expression : PlanRef)
      (domain : Nat × Nat)
deriving DecidableEq, Repr

inductive IndexUseKind where
  | integerExpression | familyGetStatic | familyGetDynamic | select | indexedSlice
deriving DecidableEq, Repr

structure IndexLutRow where
  tuple : List String
  output : String
deriving DecidableEq, Repr

structure IndexUseRow where
  owner : ObservedWire
  result : Option PlanRef
  consumed : Option PlanRef
  kind : IndexUseKind
  index : PlanRef
  outputRange : Option Range
  outputType : ValueType
  frontier : List FrontierAxis
  rows : List IndexLutRow
deriving DecidableEq, Repr

inductive SliceMemberRole where
  | rowStart | rowEndExclusive | columnStart | columnEndExclusive
deriving DecidableEq, Repr

structure SliceMember where
  role : SliceMemberRole
  expression : PlanRef
  range : Range
deriving DecidableEq, Repr

structure SliceLutRow where
  tuple : List String
  rowStart : String
  rowEndExclusive : String
  columnStart : String
  columnEndExclusive : String
deriving DecidableEq, Repr

structure SliceGroupRow where
  owner : ObservedWire
  result : Option PlanRef
  consumed : Option PlanRef
  outputType : ValueType
  frontier : List FrontierAxis
  rowSpan : Option Nat
  columnSpan : Option Nat
  members : List SliceMember
  rows : List SliceLutRow
deriving DecidableEq, Repr

inductive ResidualRoot where
  | closed (expression : ExpressionRef)
  | family (program : ProgramRef) (domain : Range)
deriving DecidableEq, Repr

structure Document where
  schemaId : String
  schemaVersion : Nat
  plaintextModulus : String
  ciphertextModulus : String
  ringDimension : Nat
  expressions : List ExpressionRow
  programs : List ProgramRow
  sources : List SourceRow
  events : List EventRow
  indexUses : List IndexUseRow
  sliceGroups : List SliceGroupRow
  residualRoot : ResidualRoot
deriving DecidableEq, Repr

end Mxx.Certificate.OperationalNoise.SchemaV1
