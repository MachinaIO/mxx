import Mxx.Certificate.Rules.CanonicalResidues
import Mxx.Certificate.Normalize
import Mxx.Certificate.Workflow

namespace Mxx.Certificate

inductive DiamondEndpointValidationError where
  | wrongSemanticBinding
  | missingAnchor (anchor : SemanticAnchorRef)
  | invalidAnchorArity (anchor : SemanticAnchorRef)
  | invalidDecoderPattern
  | invalidMessageOrigin (message : ProtocolInputId)
  | missingResidualFact
  | missingCarrierFact
  | invalidResidualShape
  | invalidMessageCoefficient
  | invalidCarrierIdentity
  deriving Repr

/-- The information derived by the closed Diamond endpoint matcher.  In particular, `residual`
comes from the executable decoder chain and `carrier` is the exact expression attached to the
registered frozen carrier wire; neither expression is certificate input. -/
structure CheckedDiamondResidual where
  residual : CoreWireRef
  carrier : MatrixExpr
  message : ProtocolInputId
  ciphertextModulus : IntExpr
  noiseBound : BoundExpr

private def endpointNode (program : Mxx.Ir.Prog) (wire : Mxx.Ir.WireRef) : Option Mxx.Ir.Node :=
  if wire.port = 0 then program.root.nodes[wire.node]? else none

private def oneArgument (node : Mxx.Ir.Node) : Option Mxx.Ir.WireRef :=
  match node.arguments with
  | [argument] => some argument
  | _ => none

private def twoArguments (node : Mxx.Ir.Node) : Option (Mxx.Ir.WireRef × Mxx.Ir.WireRef) :=
  match node.arguments with
  | [left, right] => some (left, right)
  | _ => none

private def expectConstantInt
    (program : Mxx.Ir.Prog)
    (wire : Mxx.Ir.WireRef)
    (expected : Int) : Bool :=
  match endpointNode program wire with
  | some { kind := .constantInt actual, arguments := [], .. } => actual == expected
  | _ => false

private def isIntAdd : Mxx.Ir.NodeKind → Bool
  | .intBinary .add => true
  | _ => false

private def isIntMultiply : Mxx.Ir.NodeKind → Bool
  | .intBinary .multiply => true
  | _ => false

private def isIntEqual : Mxx.Ir.NodeKind → Bool
  | .intCompare .equal => true
  | _ => false

private def isIntLessEqual : Mxx.Ir.NodeKind → Bool
  | .intCompare .lessEqual => true
  | _ => false

private def isBoolToInt : Mxx.Ir.NodeKind → Bool
  | .boolToInt => true
  | _ => false

/-- Match the exact executable scalar chain registered for `diamondBooleanInterval` and recover
the matrix whose coefficient is decoded.  Every edge is checked by frozen wire identity. -/
def matchDiamondDecoderPattern
    (program : Mxx.Ir.Prog)
    (decoded : Mxx.Ir.WireRef) : Option (Mxx.Ir.WireRef × IntExpr) := do
  let resultNode ← endpointNode program decoded
  guard (isIntEqual resultNode.kind)
  let (sumRef, twoRef) ← twoArguments resultNode
  guard (expectConstantInt program twoRef 2)

  let sumNode ← endpointNode program sumRef
  guard (isIntAdd sumNode.kind)
  let (lowerIntRef, upperIntRef) ← twoArguments sumNode

  let lowerIntNode ← endpointNode program lowerIntRef
  guard (isBoolToInt lowerIntNode.kind)
  let lowerBoolRef ← oneArgument lowerIntNode
  let upperIntNode ← endpointNode program upperIntRef
  guard (isBoolToInt upperIntNode.kind)
  let upperBoolRef ← oneArgument upperIntNode

  let lowerNode ← endpointNode program lowerBoolRef
  guard (isIntLessEqual lowerNode.kind)
  let (quarterRef, coefficientRef) ← twoArguments lowerNode
  let upperNode ← endpointNode program upperBoolRef
  guard (isIntLessEqual upperNode.kind)
  let (sameCoefficientRef, upperRef) ← twoArguments upperNode
  guard (coefficientRef == sameCoefficientRef)

  let upperValueNode ← endpointNode program upperRef
  guard (isIntMultiply upperValueNode.kind)
  let (sameQuarterRef, threeRef) ← twoArguments upperValueNode
  guard (quarterRef == sameQuarterRef)
  guard (expectConstantInt program threeRef 3)

  let coefficientNode ← endpointNode program coefficientRef
  let position ← match coefficientNode.kind with
    | .extractCoefficient position => pure position
    | _ => none
  guard (position == .constant 0)
  let residualRef ← oneArgument coefficientNode

  let quarterNode ← endpointNode program quarterRef
  let modulus ← match quarterNode.kind, quarterNode.arguments with
    | .evaluateInt (.roundDivide (.subtract modulus (.constant 2)) (.constant 4)), [] =>
        pure modulus
    | _, _ => none
  return (residualRef, modulus)

private def resolveEndpointAnchor
    (bundle : ClosedProtocolBundle)
    (anchor : SemanticAnchorRef) :
    Except DiamondEndpointValidationError CoreWireRef := do
  let binding ← match bundle.anchorBindings.find? (·.anchor == anchor) with
    | some binding => pure binding
    | none => throw (.missingAnchor anchor)
  match binding.wires with
  | [wire] => return wire
  | _ => throw (.invalidAnchorArity anchor)

private def scopedMatrixFact
    (facts : ScopedWireFactTable)
    (wire : CoreWireRef) : Option (MatrixTypeExpr × MatrixFact) := do
  let fact ← facts.find? (·.wire == wire)
  let matrixType ← fact.matrixType
  let .matrix matrix := fact.fact | none
  return (matrixType, matrix)

private def messageInputIsEndpointIdeal
    (bundle : ClosedProtocolBundle)
    (endpoint : EndpointAnchor)
    (message : ProtocolInputId) : Bool :=
  match bundle.ideal.root.outputs.find? (·.1 == endpoint.idealOutput) with
  | none => false
  | some (_, output) =>
      match endpointNode bundle.ideal output with
      | some { kind := .input inputName, arguments := [], .. } =>
          let declaredBoolean := bundle.inputContract.inputs.any fun (id, name, contract) =>
            id == message && name == inputName && match contract with
              | .boolean => true
              | _ => false
          let boundToIdeal := bundle.inputBindings.any fun binding =>
            binding.input == message && binding.destinations.any fun destination =>
              destination == .ideal inputName
          declaredBoolean && boundToIdeal
      | _ => false

private def isMessageCoefficient
    (message : ProtocolInputId)
    (expectedType : MatrixTypeExpr) : MatrixExpr → Bool
  | .select (.boolToInt (.boolWire (.protocolInput actual)))
      [.zero zeroType, .identity identityType] =>
      actual == message && zeroType == expectedType && identityType == expectedType
  | _ => false

private def sameCarrierIdentity (actual expected : MatrixExpr) : Bool :=
  match actual.sameSupported expected with
  | .equal _ => true
  | .unknown => false

/-- Validate the exact Diamond residual shape
`residual = message * registeredCarrier + noise`.  A merely nonempty affine form is rejected:
there must be exactly one signal term, its coefficient must be the Boolean protocol-input selector,
and its basis must have the exact frozen identity registered as the decoder carrier. -/
def checkDiamondResidual
    (bundle : ClosedProtocolBundle)
    (facts : ScopedWireFactTable)
    (endpoint : EndpointAnchor) :
    Except DiamondEndpointValidationError CheckedDiamondResidual := do
  let (residualAnchor, carrierAnchor, message) ← match endpoint.semantics with
    | .diamondBoolean residual carrier message => pure (residual, carrier, message)
    | _ => throw .wrongSemanticBinding
  unless messageInputIsEndpointIdeal bundle endpoint message do
    throw (.invalidMessageOrigin message)

  let decodedWire ← resolveEndpointAnchor bundle endpoint.semanticAnchor
  unless decodedWire.stage == endpoint.stage && decodedWire.scope.path.isEmpty do
    throw .invalidDecoderPattern
  let stage ← match bundle.workflow.stages.find? (·.id == endpoint.stage.name) with
    | some stage => pure stage
    | none => throw .invalidDecoderPattern
  let (residualRef, ciphertextModulus) ← match
      matchDiamondDecoderPattern stage.program ⟨decodedWire.node.value, decodedWire.port⟩ with
    | some result => pure result
    | none => throw .invalidDecoderPattern
  let residualWire : CoreWireRef := {
    stage := endpoint.stage
    scope := ⟨[]⟩
    node := ⟨residualRef.node⟩
    port := residualRef.port
  }
  let registeredResidual ← resolveEndpointAnchor bundle residualAnchor
  unless registeredResidual == residualWire do throw .invalidDecoderPattern

  let (residualType, residualFact) ← match scopedMatrixFact facts residualWire with
    | some fact => pure fact
    | none => throw .missingResidualFact
  let carrierWire ← resolveEndpointAnchor bundle carrierAnchor
  let (carrierType, carrierFact) ← match scopedMatrixFact facts carrierWire with
    | some fact => pure fact
    | none => throw .missingCarrierFact
  let carrier ← match carrierFact.primary with
    | .exact expression => pure expression
    | .affine _ => throw .invalidCarrierIdentity
  unless carrierType == residualType do throw .invalidCarrierIdentity

  let form ← match residualFact.primary with
    | .affine form => pure form
    | .exact _ => throw .invalidResidualShape
  let term ← match form.terms with
    | [term] => pure term
    | _ => throw .invalidResidualShape
  unless isMessageCoefficient message residualType term.coefficient.expression do
    throw .invalidMessageCoefficient
  unless sameCarrierIdentity term.basis carrier do throw .invalidCarrierIdentity
  return {
    residual := residualWire
    carrier := carrier
    message := message
    ciphertextModulus := ciphertextModulus
    noiseBound := form.noiseBound
  }

/-- Exact composition of the executable scalar nodes used by the Diamond decoder. This is not an
alternative evaluator: every arithmetic and comparison step calls the corresponding IR function. -/
def diamondDecoderPipelineResult (modulus raw : Int) : Option Bool := do
  let quarter := Mxx.Ir.roundDiv (modulus - 2) 4
  let upper ← Mxx.Ir.evaluateIntBinary .multiply quarter 3
  let lowerOk := Mxx.Ir.evaluateIntCompare .lessEqual quarter raw
  let upperOk := Mxx.Ir.evaluateIntCompare .lessEqual raw upper
  let lowerInt := if lowerOk then 1 else 0
  let upperInt := if upperOk then 1 else 0
  let total ← Mxx.Ir.evaluateIntBinary .add lowerInt upperInt
  return Mxx.Ir.evaluateIntCompare .equal total 2

theorem diamondQuarter_eq_ediv_four (modulus : Int) :
    Mxx.Ir.roundDiv (modulus - 2) 4 = modulus / 4 := by
  unfold Mxx.Ir.roundDiv
  omega

theorem diamondDecoderPipelineResult_eq_true_iff (modulus raw : Int) :
    diamondDecoderPipelineResult modulus raw = some true ↔
      modulus / 4 ≤ raw ∧ raw ≤ 3 * (modulus / 4) := by
  rw [← diamondQuarter_eq_ediv_four modulus]
  simp [diamondDecoderPipelineResult, Mxx.Ir.evaluateIntBinary,
    Mxx.Ir.evaluateIntCompare]
  omega

private theorem error_within_nat_bound
    {error : Int} {bound : Nat} (bounded : error.natAbs ≤ bound) :
    -(bound : Int) ≤ error ∧ error ≤ bound := by
  have boundedInt : (error.natAbs : Int) ≤ (bound : Int) := by exact_mod_cast bounded
  rw [Int.natCast_natAbs] at boundedInt
  exact abs_le.mp boundedInt

/-- A false carrier is represented by the canonical residue of its centered error. The strict
quarter bound places that residue outside the decoder's closed middle interval. -/
theorem diamondDecoder_false_of_checkedInterval
    (modulus error raw : Int) (noise : Nat)
    (modulusGe : 4 ≤ modulus)
    (canonical : CanonicalResidue modulus raw)
    (rawRelation : raw = error % modulus)
    (errorBound : error.natAbs ≤ noise)
    (checkedNearZero : noise < (modulus / 4).toNat)
    (checkedWrapped : 3 * (modulus / 4) + noise < modulus) :
    diamondDecoderPipelineResult modulus raw = some false := by
  obtain ⟨errorLower, errorUpper⟩ := error_within_nat_bound errorBound
  have quarterNonnegative : 0 ≤ modulus / 4 := by omega
  have noiseLt : (noise : Int) < modulus / 4 := by
    rw [← Int.toNat_of_nonneg quarterNonnegative]
    exact_mod_cast checkedNearZero
  rcases canonical with ⟨rawNonnegative, rawBelow⟩
  rw [rawRelation]
  by_cases errorNonnegative : 0 ≤ error
  · have reduced : error % modulus = error := Int.emod_eq_of_lt errorNonnegative (by omega)
    simp [diamondDecoderPipelineResult, diamondQuarter_eq_ediv_four, reduced,
      Mxx.Ir.evaluateIntBinary, Mxx.Ir.evaluateIntCompare]
    omega
  · have shifted : (error + modulus) % modulus = error + modulus :=
      Int.emod_eq_of_lt (by omega) (by omega)
    have reduced : error % modulus = error + modulus := by
      simpa [Int.add_emod] using shifted
    simp [diamondDecoderPipelineResult, diamondQuarter_eq_ediv_four, reduced,
      Mxx.Ir.evaluateIntBinary, Mxx.Ir.evaluateIntCompare]
    omega

/-- Exact safe condition for the true carrier. Both inequalities use the runtime quarter; unlike
the previous checker formula, this remains sound for moduli not divisible by four. -/
theorem diamondDecoder_true_of_checkedInterval
    (modulus error raw : Int) (noise : Nat)
    (modulusGe : 4 ≤ modulus)
    (canonical : CanonicalResidue modulus raw)
    (rawRelation : raw = (modulus / 2 + error) % modulus)
    (errorBound : error.natAbs ≤ noise)
    (checkedLower : modulus / 4 + noise ≤ modulus / 2)
    (checkedUpper : modulus / 2 + noise ≤ 3 * (modulus / 4)) :
    diamondDecoderPipelineResult modulus raw = some true := by
  obtain ⟨errorLower, errorUpper⟩ := error_within_nat_bound errorBound
  rcases canonical with ⟨rawNonnegative, rawBelow⟩
  have signalNonnegative : 0 ≤ modulus / 2 + error := by omega
  have signalBelow : modulus / 2 + error < modulus := by omega
  have reduced : (modulus / 2 + error) % modulus = modulus / 2 + error :=
    Int.emod_eq_of_lt signalNonnegative signalBelow
  rw [rawRelation, reduced]
  simp [diamondDecoderPipelineResult, diamondQuarter_eq_ediv_four,
    Mxx.Ir.evaluateIntBinary, Mxx.Ir.evaluateIntCompare]
  omega

/-- The closed Boolean endpoint rule.  The analyzer must supply the exact message-carrier
relation and canonical-residue provenance; this theorem only composes those derived facts with
the two parameter-only interval obligations checked by Phase B. -/
theorem diamondDecoder_of_checkedIntervals
    (message : Bool)
    (modulus error raw : Int)
    (noise : Nat)
    (modulusGe : 4 ≤ modulus)
    (canonical : CanonicalResidue modulus raw)
    (rawRelation :
      raw = ((if message then modulus / 2 else 0) + error) % modulus)
    (errorBound : error.natAbs ≤ noise)
    (checkedNearZero : noise < (modulus / 4).toNat)
    (checkedWrapped : 3 * (modulus / 4) + noise < modulus)
    (checkedLower : modulus / 4 + noise ≤ modulus / 2)
    (checkedUpper : modulus / 2 + noise ≤ 3 * (modulus / 4)) :
    diamondDecoderPipelineResult modulus raw = some message := by
  cases message
  · apply diamondDecoder_false_of_checkedInterval modulus error raw noise modulusGe canonical
      (by simpa using rawRelation) errorBound checkedNearZero checkedWrapped
  · apply diamondDecoder_true_of_checkedInterval modulus error raw noise modulusGe canonical
      (by simpa using rawRelation) errorBound checkedLower checkedUpper

example : diamondDecoderPipelineResult 17 0 = some false := rfl
example : diamondDecoderPipelineResult 17 8 = some true := rfl

/-- Boundary counterexample: the old true obligation accepted this, but the real decoder does not. -/
example :
    (1 + 1 < (6 / 2 : Nat)) ∧
      diamondDecoderPipelineResult 6 (((6 / 2 : Int) + 1) % 6) = some false := by
  decide

end Mxx.Certificate
