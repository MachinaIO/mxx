import Mxx.Certificate.Facts
import Mxx.Certificate.ProtocolSyntax

namespace Mxx.Certificate

/-- The complete initial rule universe. Rules outside this type cannot be emitted. -/
inductive Rule where
  | introduceMatrixInput
  | introduceExactConstant
  | introduceGaussian
  | introduceUniform
  | introduceHash
  | introduceTrapdoorSample
  | introducePreimage
  | addAffine
  | subtractAffine
  | negateAffine
  | materializeIdentity
  | multiplyAffineRight
  | multiplyAffineLeft
  | applyPreimageRelation
  | applyGadgetDecompositionRelation
  | decomposeGadget
  | sliceAffine
  | concatAffine
  | reshapeAffine
  | selectAffine
  | inheritArtifact
  | getFamilyStatic
  | getFamilyDynamic
  | evaluateParallelLoop
  | evaluateSequentialLoop
  | decodeThresholdBool
  deriving BEq, DecidableEq, Repr

/-- The single authoritative enabled subset of the closed rule universe.
No node rule is enabled until its verifier branch and local soundness theorem both exist. -/
def enabledInitialRules : List Rule := [
  .introduceExactConstant,
  .introduceGaussian,
  .introduceHash,
  .introduceTrapdoorSample,
  .introducePreimage,
  .addAffine,
  .subtractAffine,
  .negateAffine,
  .materializeIdentity,
  .multiplyAffineRight,
  .multiplyAffineLeft,
  .decomposeGadget
]

def EnabledInitialRule (rule : Rule) : Prop :=
  rule ∈ enabledInitialRules

def isInitialRuleEnabled (rule : Rule) : Bool :=
  enabledInitialRules.contains rule

/-- Reject a rule that has not yet acquired a verified implementation. -/
def requireInitialRule (rule : Rule) : Except Rule Unit :=
  if isInitialRuleEnabled rule then .ok () else .error rule

theorem introduceMatrixInput_disabled :
    ¬EnabledInitialRule .introduceMatrixInput := by
  simp [EnabledInitialRule, enabledInitialRules]

theorem introduceGaussian_enabled :
    EnabledInitialRule .introduceGaussian := by
  simp [EnabledInitialRule, enabledInitialRules]

example : requireInitialRule .multiplyAffineRight = .ok () := by
  decide

inductive RuleInputRef where
  | value (reference : ValueInstanceRef)
  | matrixFact (reference : ValueInstanceRef)
  | trapdoorFact (reference : ValueInstanceRef)
  | family (reference : JointFamilyId)
  | recurrence (reference : SequentialRecurrenceRef)
  deriving BEq, DecidableEq, Repr

/-- An override chooses a closed rule and existing inputs, but cannot state its output fact. -/
structure RuleUse where
  output : SemanticAnchorRef
  rule : Rule
  inputs : List RuleInputRef
  deriving BEq, DecidableEq, Repr

/-- Sparse user guidance. It may choose only a closed rule and existing typed inputs. -/
structure SparseCertificate where
  overrides : List RuleUse

inductive ValueType where
  | matrix (type : MatrixTypeExpr)
  | integer
  | boolean
  | bytes (length : IntExpr)
  | family (count : IntExpr) (element : ValueType)

inductive CarrierSchema where
  | thresholdBoolean
  | diamondBooleanInterval
  deriving BEq, DecidableEq, Repr

/-- Closed structural patterns understood by the initial endpoint verifier. -/
inductive EndpointNodePattern where
  | thresholdDecodeBool
  | diamondCoefficientInterval
  deriving BEq, DecidableEq, Repr

structure EndpointSpec where
  id : EndpointSpecId
  inputType : ValueType
  outputType : ValueType
  carrierSchema : CarrierSchema
  structuralPattern : EndpointNodePattern

inductive SemanticLemmaId where
  | booleanCircuitInterpreterCorrect
  deriving BEq, DecidableEq, Repr

end Mxx.Certificate
