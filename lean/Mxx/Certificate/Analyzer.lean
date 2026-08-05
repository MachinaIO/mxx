import Mxx.Certificate.LocalSoundness
import Mxx.Certificate.AffineNormalize
import Mxx.Certificate.Workflow
import Mxx.Certificate.Bounds
import Mxx.Certificate.Typing
import Mxx.Certificate.Rules.Family
import Mxx.Certificate.Rules.DiamondEndpoint
import Mxx.Certificate.Rules.LoopRecurrence
import Mxx.Certificate.Rules.MatrixAffine
import Mxx.Certificate.Rules.MatrixSelect
import Mxx.Certificate.Rules.ScalarControl
import Mxx.Certificate.Rules.Transforms

namespace Mxx.Certificate

inductive VerifyError where
  | disabledRule (rule : Rule)
  | unsupportedNode (stage : StageId) (node : NodeId)
  | unsupportedDefinition (stage : StageId) (name : String)
  | missingInputFact (stage : StageId) (node : NodeId) (input : Mxx.Ir.WireRef)
  | expectedMatrixFact (wire : CoreWireRef)
  | expectedTrapdoorFact (wire : CoreWireRef)
  | trapdoorPublicMismatch (wire : CoreWireRef)
  | missingAnchorBinding (anchor : SemanticAnchorRef)
  | invalidAnchorWire (anchor : SemanticAnchorRef) (wire : CoreWireRef)
  | unsupportedOverride (anchor : SemanticAnchorRef)
  | mismatchedMatrixTypes (left right : MatrixTypeExpr)
  | expectedIntegerFact (wire : CoreWireRef)
  | expectedBooleanFact (wire : CoreWireRef)
  | missingInputContract (name : String)
  | missingProgramInput (stage : StageId) (name : String)
  | missingArtifactOutput (stage : StageId) (name : String)
  | invalidInputCoverage (input : ProtocolInputId)
  | invalidInputDestination (input : ProtocolInputId)
  | invalidEndpointCoverage (endpoint : EndpointSpecId)
  | invalidEndpointConnection (endpoint : EndpointSpecId)
  | invalidPreconditionSpec
  | duplicateInputId (input : ProtocolInputId)
  | duplicateInputName (name : String)
  | duplicateInputDestination (destination : ProtocolInputDestination)
  | unboundProgramInput (stage : StageId) (name : String)
  | duplicateEndpointSpec (endpoint : EndpointSpecId)
  | invalidComparatorPolarity (endpoint : EndpointSpecId)
  | nonBooleanOutput (stage : StageId) (name : String)
  | invalidEndpointAnchorArity (endpoint : EndpointSpecId)
  | missingOrInvalidOutputTypes (stage : StageId) (node : NodeId)
  | inputContractTypeMismatch (input : ProtocolInputId) (stage : StageId) (name : String)
  | duplicateParameter (name : String)
  | missingParameterDeclaration (name : String)
  | parameterKindMismatch (name : String)
  | typing (error : TypingError)
  | exactLeftAffineRightProduct (stage : StageId) (node : NodeId)
  | generalAffineProduct (stage : StageId) (node : NodeId)
  | missingFamily (joint : JointFamilyId)
  | invalidFamilySlot (joint : JointFamilyId) (slot : Nat)
  | invalidLoopDefinition (stage : StageId) (name : String)
  | invalidLoopArity (stage : StageId) (node : NodeId)
  | invalidLoopArityInScope (stage : StageId) (scope : StaticScopeId) (node : NodeId)
  | unsupportedSequentialRecurrence (stage : StageId) (node : NodeId)
  | unsupportedCarriedKind (stage : StageId) (node : NodeId) (slot : Nat)
  | relationBearingCarriedMatrix (stage : StageId) (node : NodeId) (slot : Nat)
  | escapedCarriedInput (stage : StageId) (node : NodeId) (slot : Nat)
  | invalidExpressionReference
  | scalarControl (error : ScalarControlRuleError)
  | matrixAffine (error : MatrixAffineError)
  | matrixSelect (wire : CoreWireRef) (error : MatrixSelectError)
  | transform (error : TransformRuleError)
  | affineNormalize (wire : CoreWireRef) (error : AffineNormalizeError)

private def rootScope : StaticScopeId := ⟨[]⟩

private def scopedWire
    (stage : StageId)
    (scope : StaticScopeId)
    (wire : Mxx.Ir.WireRef) : CoreWireRef where
  stage
  scope
  node := ⟨wire.node⟩
  port := wire.port

private def coreWire (stage : StageId) (wire : Mxx.Ir.WireRef) : CoreWireRef :=
  scopedWire stage rootScope wire

private def scopedOutputWire
    (stage : StageId)
    (scope : StaticScopeId)
    (node : Nat)
    (port : Nat := 0) : CoreWireRef :=
  scopedWire stage scope ⟨node, port⟩

private def outputWire (stage : StageId) (node : Nat) (port : Nat := 0) : CoreWireRef :=
  scopedOutputWire stage rootScope node port

private def matrixInstance (wire : CoreWireRef) (type : MatrixTypeExpr) : MatrixInstanceRef where
  value := .ofCoreWire wire
  type := type

private def lookupScopedFact (wire : CoreWireRef) : ScopedWireFactTable → Option ScopedWireFact :=
  fun entries =>
    match entries with
    | [] => none
    | entry :: tail => if entry.wire = wire then some entry else lookupScopedFact wire tail

private def inputNodeWire
    (stage : StageId)
    (name : String)
    (nodes : List Mxx.Ir.Node) : Option CoreWireRef :=
  let rec visit (index : Nat) : List Mxx.Ir.Node → Option CoreWireRef
    | [] => none
    | node :: tail =>
        match node.kind with
        | .input candidate => if candidate = name then some (outputWire stage index) else
            visit (index + 1) tail
        | _ => visit (index + 1) tail
  visit 0 nodes

private def inputNodeWireInScope
    (stage : StageId)
    (scope : StaticScopeId)
    (name : String)
    (nodes : List Mxx.Ir.Node) : Option CoreWireRef :=
  let rec visit (index : Nat) : List Mxx.Ir.Node → Option CoreWireRef
    | [] => none
    | node :: tail =>
        match node.kind with
        | .input candidate =>
            if candidate = name then some (scopedOutputWire stage scope index) else
              visit (index + 1) tail
        | _ => visit (index + 1) tail
  visit 0 nodes

private def inputContractByName
    (contract : InputContract)
    (name : String) : Option (ProtocolInputId × InputValueContract) :=
  match contract.inputs.find? (fun entry => entry.2.1 = name) with
  | some entry => some (entry.1, entry.2.2)
  | none => none

private def inputContractById
    (contract : InputContract)
    (id : ProtocolInputId) : Option InputValueContract :=
  match contract.inputs.find? (fun entry => entry.1 = id) with
  | some entry => some entry.2.2
  | none => none

def transportFact (wire : CoreWireRef) (source : ScopedWireFact) : ScopedWireFact :=
  match source.fact with
  | .matrix matrix => {
      source with
      wire
      fact := .matrix { matrix with subject := .ofCoreWire wire }
    }
  | _ => { source with wire }

private def requireMatrix
    (facts : ScopedWireFactTable)
    (wire : CoreWireRef) : Except VerifyError (MatrixFact × MatrixTypeExpr) :=
  match lookupScopedFact wire facts with
  | some { fact := .matrix fact, matrixType := some type, .. } => .ok (fact, type)
  | some _ => .error (.expectedMatrixFact wire)
  | none => .error (.expectedMatrixFact wire)

private def requireTrapdoor
    (facts : ScopedWireFactTable)
    (wire : CoreWireRef) : Except VerifyError TrapdoorFact :=
  match lookupScopedFact wire facts with
  | some { fact := .trapdoor fact, .. } => .ok fact
  | some _ => .error (.expectedTrapdoorFact wire)
  | none => .error (.expectedTrapdoorFact wire)

private def requireInteger
    (facts : ScopedWireFactTable)
    (wire : CoreWireRef) : Except VerifyError IntegerFact :=
  match lookupScopedFact wire facts with
  | some { fact := .integer fact, .. } => .ok fact
  | _ => .error (.expectedIntegerFact wire)

private def requireBoolean
    (facts : ScopedWireFactTable)
    (wire : CoreWireRef) : Except VerifyError BooleanFact :=
  match lookupScopedFact wire facts with
  | some { fact := .boolean fact, .. } => .ok fact
  | _ => .error (.expectedBooleanFact wire)

private def scalarFact
    (wire : CoreWireRef)
    (fact : ValueFact) : ScopedWireFact := { wire, matrixType := none, fact }

private def centeredBound (type : MatrixTypeExpr) : BoundExpr :=
  .floorDivide (.absolute type.modulus) 2

private def exactFact
    (wire : CoreWireRef)
    (type : MatrixTypeExpr)
    (expression : MatrixExpr)
    (bound : BoundExpr) : ScopedWireFact := {
  wire
  matrixType := some type
  fact := .matrix {
    subject := .ofCoreWire wire
    primary := .exact expression
    relations := []
    totalNormBound := bound
  }
}

private def boundedFact
    (wire : CoreWireRef)
    (type : MatrixTypeExpr)
    (bound : BoundExpr)
    (relations : List MatrixRelation := []) : ScopedWireFact := {
  wire
  matrixType := some type
  fact := .matrix {
    subject := .ofCoreWire wire
    primary := .affine { terms := [], noiseBound := bound }
    relations
    totalNormBound := bound
  }
}

/-- Apply the proof-producing affine normalizer at every analyzer boundary that can create or
rewrite signal terms.  Exact facts and non-matrix facts are unchanged. -/
private def normalizeScopedFact
    (fact : ScopedWireFact) : Except VerifyError ScopedWireFact := do
  match fact.fact with
  | .matrix matrix =>
      match matrix.primary with
      | .exact _ => return fact
      | .affine form =>
          let normalized ← normalizeAffineForm form |>.mapError
            (.affineNormalize fact.wire)
          return { fact with fact := .matrix {
            matrix with primary := .affine normalized.form
          } }
  | _ => return fact

private def normalizeMatrixFact
    (wire : CoreWireRef)
    (fact : MatrixFact) : Except VerifyError MatrixFact := do
  match fact.primary with
  | .exact _ => return fact
  | .affine form =>
      let normalized ← normalizeAffineForm form |>.mapError (.affineNormalize wire)
      return { fact with primary := .affine normalized.form }

private def InputValueContract.factSchema : InputValueContract → ValueFactSchema
  | .matrixExact type => .matrix type .exact [] .unknown
  | .matrixBounded type _ => .matrix type (.affine []) [] .unknown
  | .integerRange .. => .integer
  | .boolean => .boolean
  | .bytes .. => .bytes
  | .family count element => .family count element.factSchema

private def protocolInputFact
    (wire : CoreWireRef)
    (id : ProtocolInputId) : InputValueContract → ScopedWireFact
  | .matrixExact type => exactFact wire type
      (.wire { value := .protocolInput id, type }) (centeredBound type)
  | .matrixBounded type bound => boundedFact wire type bound.toBoundExpr
  | .integerRange lower upper => scalarFact wire (.integer {
      expression := .intWire (.protocolInput id)
      lower := .integer lower
      upper := .integer upper
    })
  | .boolean => scalarFact wire (.boolean { expression := .boolWire (.protocolInput id) })
  | .bytes _ => scalarFact wire (.bytes (.protocolInput id))
  | .family count element => scalarFact wire (.family {
      aggregate := .joint ⟨id.name⟩ 0 []
      count
      elementSchema := element.factSchema
    })

private def normalizeIntExpr : IntExpr → IntExpr
  | .add left right =>
      match normalizeIntExpr left, normalizeIntExpr right with
      | .constant 0, value | value, .constant 0 => value
      | left, right => .add left right
  | .multiply left right =>
      match normalizeIntExpr left, normalizeIntExpr right with
      | .constant 1, value | value, .constant 1 => value
      | left, right => .multiply left right
  | .subtract left right => .subtract (normalizeIntExpr left) (normalizeIntExpr right)
  | .divide left right => .divide (normalizeIntExpr left) (normalizeIntExpr right)
  | .roundDivide left right => .roundDivide (normalizeIntExpr left) (normalizeIntExpr right)
  | .log2Ceil value => .log2Ceil (normalizeIntExpr value)
  | value => value

private def intExprEqual (left right : IntExpr) : Bool :=
  normalizeIntExpr left == normalizeIntExpr right

private def matrixTypeEqual (left right : MatrixTypeExpr) : Bool :=
  intExprEqual left.modulus right.modulus &&
    intExprEqual left.ringDimension right.ringDimension &&
    intExprEqual left.rows right.rows && intExprEqual left.columns right.columns

private def boundExprEqual : BoundExpr → BoundExpr → Bool
  | .constant left, .constant right => left == right
  | .parameter left, .parameter right
  | .absolute left, .absolute right => intExprEqual left right
  | .add left₁ right₁, .add left₂ right₂
  | .multiply left₁ right₁, .multiply left₂ right₂
  | .maximum left₁ right₁, .maximum left₂ right₂
  | .minimum left₁ right₁, .minimum left₂ right₂ =>
      boundExprEqual left₁ left₂ && boundExprEqual right₁ right₂
  | .floorDivide value₁ divisor₁, .floorDivide value₂ divisor₂ =>
      boundExprEqual value₁ value₂ && divisor₁ == divisor₂
  | .matrixProduct ring₁ inner₁ left₁ right₁,
      .matrixProduct ring₂ inner₂ left₂ right₂ =>
      intExprEqual ring₁ ring₂ && intExprEqual inner₁ inner₂ &&
        boundExprEqual left₁ left₂ && boundExprEqual right₁ right₂
  | .recurrenceResult recurrence₁ path₁, .recurrenceResult recurrence₂ path₂ =>
      recurrence₁ == recurrence₂ && path₁ == path₂
  | .carriedInput path₁, .carriedInput path₂ => path₁ == path₂
  | _, _ => false

private def intBoundExprEqual : IntBoundExpr → IntBoundExpr → Bool
  | .integer left, .integer right => intExprEqual left right
  | .natural left, .natural right => boundExprEqual left right
  | .negate left, .negate right => intBoundExprEqual left right
  | .add left₁ right₁, .add left₂ right₂
  | .subtract left₁ right₁, .subtract left₂ right₂
  | .multiply left₁ right₁, .multiply left₂ right₂
  | .divide left₁ right₁, .divide left₂ right₂
  | .minimum left₁ right₁, .minimum left₂ right₂
  | .maximum left₁ right₁, .maximum left₂ right₂ =>
      intBoundExprEqual left₁ left₂ && intBoundExprEqual right₁ right₂
  | .carriedInput path₁, .carriedInput path₂ => path₁ == path₂
  | .recurrenceResult recurrence₁ path₁, .recurrenceResult recurrence₂ path₂ =>
      recurrence₁ == recurrence₂ && path₁ == path₂
  | _, _ => false

private def wireTypeMatchesContract : Mxx.Ir.WireTypeExpr → InputValueContract → Bool
  | .constantInt, .integerRange _ _ | .integer, .integerRange _ _ => true
  | .constantBool, .boolean | .boolean, .boolean => true
  | .bytes actualLength, .bytes expectedLength => intExprEqual actualLength expectedLength
  | .matrix actual, .matrixExact expected
  | .matrix actual, .matrixBounded expected _
  | .preimage actual, .matrixExact expected
  | .preimage actual, .matrixBounded expected _ => matrixTypeEqual actual expected
  | .indexedFamily actualElement actualCount, .family expectedCount expectedElement =>
      intExprEqual actualCount expectedCount &&
        wireTypeMatchesContract actualElement expectedElement
  | _, _ => false

private def declaredMatrixTypeMatches
    (expected : MatrixTypeExpr) : Mxx.Ir.WireTypeExpr → Bool
  | .matrix actual | .preimage actual => matrixTypeEqual actual expected
  | _ => false

private def nodeIntrinsicOutputTypesMatch (node : Mxx.Ir.Node) : Bool :=
  match node.kind, node.outputTypes with
  | .constantInt _, [.constantInt] | .evaluateInt _, [.constantInt] => true
  | .constantBool _, [.constantBool] => true
  | .boolToInt, [.integer] => true
  | .intBinary _, [.integer] => true
  | .intCompare _, [.boolean] | .bitExtract _, [.boolean] => true
  | .zeroMatrix type, [output]
  | .identityMatrix type, [output]
  | .constantMatrix type _, [output]
  | .gadgetMatrix type _, [output]
  | .gaussianSample type _, [output]
  | .uniformSample type _ _, [output]
  | .hashSample type _ _ _ _ _ _ _, [output]
  | .gadgetDecompose type _ _, [output]
  | .preimageSample type _, [output] => declaredMatrixTypeMatches type output
  | .trapdoorSample type _, [.matrix publicType, .trapdoor privateType _ _ _ _] =>
      matrixTypeEqual type publicType && matrixTypeEqual type privateType
  | .thresholdDecodeBool _ _ _, outputs =>
      outputs.length = node.outputCount && outputs.all (fun output => output = .boolean)
  | _, outputs => outputs.length = node.outputCount

private def requireDeclaredMatrixType
    (stage : StageId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node) : Except VerifyError MatrixTypeExpr :=
  match node.outputTypes with
  | [.matrix type] | [.preimage type] => .ok type
  | _ => .error (.missingOrInvalidOutputTypes stage ⟨nodeId⟩)

private def identityTypeFor (type : MatrixTypeExpr) : MatrixTypeExpr where
  modulus := type.modulus
  ringDimension := type.ringDimension
  rows := type.rows
  columns := type.rows

private def concatOutputType
    (axis : Mxx.Ir.ConcatAxis)
    (types : List MatrixTypeExpr) : Option MatrixTypeExpr := do
  let first ← match types with
    | first :: _ => some first
    | [] => none
  for candidate in types.drop 1 do
    if candidate.modulus != first.modulus ||
        candidate.ringDimension != first.ringDimension then
      none
    match axis with
    | .rows => if candidate.columns != first.columns then none
    | .columns => if candidate.rows != first.rows then none
    | .diagonal => pure ()
  let sumRows := (types.drop 1).foldl
    (fun total type => .add total type.rows) first.rows
  let sumColumns := (types.drop 1).foldl
    (fun total type => .add total type.columns) first.columns
  match axis with
  | .rows => some {
      modulus := first.modulus
      ringDimension := first.ringDimension
      rows := sumRows
      columns := first.columns
    }
  | .columns => some {
      modulus := first.modulus
      ringDimension := first.ringDimension
      rows := first.rows
      columns := sumColumns
    }
  | .diagonal => some {
      modulus := first.modulus
      ringDimension := first.ringDimension
      rows := sumRows
      columns := sumColumns
    }

private def exactSignalTerm
    (type : MatrixTypeExpr)
    (signal : MatrixExpr) : Except TypingError SignalTerm := do
  let coefficientType := identityTypeFor type
  let product ← inferMatrixProductType coefficientType type
  return {
    coefficient := {
      expression := .identity coefficientType
      normBound := .constant 1
    }
    basis := signal
    mode := product.mode
  }

private def signedTerms (subtract : Bool) (terms : List SignalTerm) : List SignalTerm :=
  if subtract then
    terms.map fun term => {
      term with coefficient := {
        term.coefficient with expression := .negate term.coefficient.expression
      }
    }
  else terms

private def addOrSubtractFact
    (subtract : Bool)
    (output : CoreWireRef)
    (type : MatrixTypeExpr)
    (left right : MatrixFact) : Except VerifyError ScopedWireFact :=
  let bound := BoundExpr.add left.totalNormBound right.totalNormBound
  match left.primary, right.primary with
  | .exact leftExpression, .exact rightExpression =>
      let expression := if subtract then
        MatrixExpr.add leftExpression (.negate rightExpression)
      else MatrixExpr.add leftExpression rightExpression
      .ok (exactFact output type expression bound)
  | .affine { terms := [], noiseBound := leftNoise },
      .affine { terms := [], noiseBound := rightNoise } =>
      .ok (boundedFact output type (.add leftNoise rightNoise))
  | .affine leftForm, .affine rightForm =>
      .ok {
        wire := output
        matrixType := some type
        fact := .matrix {
          subject := .ofCoreWire output
          primary := .affine {
            terms := leftForm.terms ++ signedTerms subtract rightForm.terms
            noiseBound := .add leftForm.noiseBound rightForm.noiseBound
          }
          relations := []
          totalNormBound := bound
        }
      }
  | .exact leftExpression, .affine rightForm => do
      let leftTerm ← (exactSignalTerm type leftExpression).mapError .typing
      return {
        wire := output
        matrixType := some type
        fact := .matrix {
          subject := .ofCoreWire output
          primary := .affine {
            terms := leftTerm :: signedTerms subtract rightForm.terms
            noiseBound := rightForm.noiseBound
          }
          relations := []
          totalNormBound := bound
        }
      }
  | .affine leftForm, .exact rightExpression => do
      let rightExpression := if subtract then .negate rightExpression else rightExpression
      let rightTerm ← (exactSignalTerm type rightExpression).mapError .typing
      return {
        wire := output
        matrixType := some type
        fact := .matrix {
          subject := .ofCoreWire output
          primary := .affine {
            terms := leftForm.terms ++ [rightTerm]
            noiseBound := leftForm.noiseBound
          }
          relations := []
          totalNormBound := bound
        }
      }

private def negateFact
    (output : CoreWireRef)
    (type : MatrixTypeExpr)
    (input : MatrixFact) : Except VerifyError ScopedWireFact :=
  match input.primary with
  | .exact expression =>
      .ok (exactFact output type (.negate expression) input.totalNormBound)
  | .affine { terms := [], noiseBound } =>
      .ok (boundedFact output type noiseBound)
  | _ => .error (.unsupportedNode output.stage output.node)

private def matrixProductBound
    (leftType rightType : MatrixTypeExpr)
    (leftBound rightBound : BoundExpr) : Except TypingError (MatrixProductType × BoundExpr) := do
  let product ← inferMatrixProductType leftType rightType
  let innerDimension := match product.mode with
    | .ordinaryMatrixProduct => leftType.columns
    | .leftPolynomialScalarBroadcast | .rightPolynomialScalarBroadcast |
        .swappedRowVectorScalarProduct => .constant 1
  return (product, .matrixProduct leftType.ringDimension innerDimension leftBound rightBound)

private def wholeBoundedView
    (wire : CoreWireRef)
    (type : MatrixTypeExpr)
    (fact : MatrixFact) : BoundedMatrixExpr := {
  expression := .wire (matrixInstance wire type)
  normBound := fact.totalNormBound
}

private def scopedMatrixFact
    (wire : CoreWireRef)
    (type : MatrixTypeExpr)
    (primary : MatrixPrimaryForm)
    (totalNormBound : BoundExpr) : ScopedWireFact := {
  wire
  matrixType := some type
  fact := .matrix {
    subject := .ofCoreWire wire
    primary
    relations := []
    totalNormBound
  }
}

private def multiplyAffineByRight
    (leftType rightType : MatrixTypeExpr)
    (form : AffineForm)
    (rightExpression : MatrixExpr)
    (rightBound : BoundExpr) : Except TypingError AffineForm := do
  let terms ← form.terms.mapM fun term ↦
    mkSignalTerm term.coefficient (.multiply term.basis rightExpression)
  let ⟨_, noiseBound⟩ ←
    matrixProductBound leftType rightType form.noiseBound rightBound
  return { terms, noiseBound }

private def multiplyBoundedByAffine
    (leftType rightType : MatrixTypeExpr)
    (left : BoundedMatrixExpr)
    (form : AffineForm) : Except TypingError AffineForm := do
  let terms ← form.terms.mapM fun term ↦ do
    let coefficientType ← match term.coefficient.expression.inferType with
      | some type => pure type
      | none => throw .unknownExpressionType
    let ⟨_, coefficientBound⟩ ← matrixProductBound leftType coefficientType
      left.normBound term.coefficient.normBound
    mkSignalTerm {
      expression := .multiply left.expression term.coefficient.expression
      normBound := coefficientBound
    } term.basis
  let ⟨_, noiseBound⟩ ← matrixProductBound leftType rightType
    left.normBound form.noiseBound
  return { terms, noiseBound }

private def multiplyFact
    (rule : Rule)
    (output : CoreWireRef)
    (outputType : MatrixTypeExpr)
    (leftWire : CoreWireRef)
    (leftType : MatrixTypeExpr)
    (left : MatrixFact)
    (rightWire : CoreWireRef)
    (rightType : MatrixTypeExpr)
    (right : MatrixFact) : Except VerifyError ScopedWireFact := do
  let ⟨product, totalBound⟩ ←
    (matrixProductBound leftType rightType left.totalNormBound right.totalNormBound).mapError .typing
  if !matrixTypeEqual product.output outputType then
    throw (.mismatchedMatrixTypes product.output outputType)
  let leftView := wholeBoundedView leftWire leftType left
  let rightView := wholeBoundedView rightWire rightType right
  match rule, left.primary, right.primary with
  | .multiplyAffineRight, .exact leftExpression, .exact rightExpression =>
      return scopedMatrixFact output outputType
        (.exact (.multiply leftExpression rightExpression)) totalBound
  | .multiplyAffineRight, .affine leftForm, .exact rightExpression =>
      if leftForm.terms.isEmpty then
        let term ← (mkSignalTerm leftView rightExpression).mapError .typing
        return scopedMatrixFact output outputType
          (.affine { terms := [term], noiseBound := .constant 0 }) totalBound
      let form ← (multiplyAffineByRight leftType rightType leftForm rightExpression
        right.totalNormBound).mapError .typing
      return scopedMatrixFact output outputType (.affine form) totalBound
  | .multiplyAffineRight, .affine leftForm, .affine rightForm =>
      if !rightForm.terms.isEmpty then
        throw (.generalAffineProduct output.stage output.node)
      if leftForm.terms.isEmpty then
        return scopedMatrixFact output outputType
          (.affine { terms := [], noiseBound := totalBound }) totalBound
      let form ← (multiplyAffineByRight leftType rightType leftForm rightView.expression
        rightView.normBound).mapError .typing
      return scopedMatrixFact output outputType (.affine form) totalBound
  | .multiplyAffineRight, .exact _, .affine rightForm =>
      if !rightForm.terms.isEmpty then
        throw (.exactLeftAffineRightProduct output.stage output.node)
      return scopedMatrixFact output outputType
        (.affine { terms := [], noiseBound := totalBound }) totalBound
  | .multiplyAffineLeft, .affine leftForm, .affine rightForm =>
      if !leftForm.terms.isEmpty || rightForm.terms.isEmpty then
        throw (.generalAffineProduct output.stage output.node)
      let form ← (multiplyBoundedByAffine leftType rightType leftView rightForm).mapError .typing
      return scopedMatrixFact output outputType (.affine form) totalBound
  | _, _, _ => throw (.unsupportedNode output.stage output.node)

private def lookupMatrixBySubject
    (subject : ValueInstanceRef) : ScopedWireFactTable → Option (MatrixFact × MatrixTypeExpr)
  | [] => none
  | entry :: tail => match entry.fact, entry.matrixType with
      | .matrix fact, some type =>
          if fact.subject == subject then some (fact, type)
          else lookupMatrixBySubject subject tail
      | _, _ => lookupMatrixBySubject subject tail

private def matchingRelationTarget
    (right : MatrixFact)
    (basis : MatrixExpr)
    (facts : ScopedWireFactTable) : Option (MatrixFact × MatrixTypeExpr) :=
  let rec visit : List MatrixRelation → Option (MatrixFact × MatrixTypeExpr)
    | [] => none
    | .preimage subject source target _ :: tail =>
        if subject != right.subject then visit tail
        else match basis with
          | .wire actual =>
              if actual.value == source.value && matrixTypeEqual actual.type source.type then
                lookupMatrixBySubject target.value facts
              else visit tail
          | _ => visit tail
    | .gadgetDecomposition subject target base digitCount :: tail =>
        if subject != right.subject then visit tail
        else match basis with
          | .gadget actualType actualBase =>
              let expectedType : MatrixTypeExpr := {
                modulus := target.type.modulus
                ringDimension := target.type.ringDimension
                rows := target.type.rows
                columns := .multiply target.type.rows digitCount
              }
              if intExprEqual actualBase base && matrixTypeEqual actualType expectedType then
                lookupMatrixBySubject target.value facts
              else visit tail
          | _ => visit tail
  visit right.relations

private def expandSignalThroughTarget
    (term : SignalTerm)
    (target : MatrixFact)
    (targetType : MatrixTypeExpr) : Except VerifyError (List SignalTerm × BoundExpr) := do
  let coefficientType ← match term.coefficient.expression.inferType with
    | some type => pure type
    | none => throw (.typing .unknownExpressionType)
  match target.primary with
  | .exact expression =>
      let expanded ← mkSignalTerm term.coefficient expression |>.mapError .typing
      return ([expanded], .constant 0)
  | .affine form =>
      let expanded ← form.terms.mapM fun inner ↦ do
        let innerCoefficientType ← match inner.coefficient.expression.inferType with
          | some type => pure type
          | none => throw (.typing .unknownExpressionType)
        let ⟨_, coefficientBound⟩ ← matrixProductBound coefficientType innerCoefficientType
          term.coefficient.normBound inner.coefficient.normBound |>.mapError .typing
        mkSignalTerm {
          expression := .multiply term.coefficient.expression inner.coefficient.expression
          normBound := coefficientBound
        } inner.basis |>.mapError .typing
      let ⟨_, noiseBound⟩ ← matrixProductBound coefficientType targetType
        term.coefficient.normBound form.noiseBound |>.mapError .typing
      return (expanded, noiseBound)

/-- Rewrite every matching `B * K` signal factor through the exact sampler relation
`B * K = target` in `R_q`.  Unmatched terms remain ordinary products.  The original affine noise
is multiplied by the complete right operand; target noise introduced by a rewrite is added once
per rewritten coefficient. -/
private def rewriteAffinePreimageProduct
    (leftType rightType : MatrixTypeExpr)
    (left : AffineForm)
    (right : MatrixFact)
    (rightExpression : MatrixExpr)
    (facts : ScopedWireFactTable) : Except VerifyError AffineForm := do
  let ⟨_, initialNoise⟩ ← matrixProductBound leftType rightType left.noiseBound
    right.totalNormBound |>.mapError .typing
  let ⟨terms, noise⟩ ← left.terms.foldlM (init := ([], initialNoise)) fun state term ↦ do
    match matchingRelationTarget right term.basis facts with
    | some (target, targetType) =>
        let ⟨expanded, addedNoise⟩ ← expandSignalThroughTarget term target targetType
        return (state.1 ++ expanded, .add state.2 addedNoise)
    | none =>
        let unchanged ← mkSignalTerm term.coefficient (.multiply term.basis rightExpression)
          |>.mapError .typing
        return (state.1 ++ [unchanged], state.2)
  return { terms, noiseBound := noise }

private def rewritePreimageProduct?
    (output : CoreWireRef)
    (outputType leftType rightType : MatrixTypeExpr)
    (left right : MatrixFact)
    (facts : ScopedWireFactTable) : Except VerifyError (Option ScopedWireFact) := do
  let rightExpression : MatrixExpr := .wire { value := right.subject, type := rightType }
  match left.primary with
  | .exact expression =>
      match matchingRelationTarget right expression facts with
      | none => return none
      | some (target, targetType) =>
          if !matrixTypeEqual targetType outputType then
            throw (.mismatchedMatrixTypes targetType outputType)
          return some (scopedMatrixFact output outputType target.primary target.totalNormBound)
  | .affine form =>
      let hasRewrite := form.terms.any fun term ↦
        (matchingRelationTarget right term.basis facts).isSome
      if !hasRewrite then return none
      let rewritten ← rewriteAffinePreimageProduct leftType rightType form right rightExpression facts
      let ⟨product, totalBound⟩ ← matrixProductBound leftType rightType
        left.totalNormBound right.totalNormBound |>.mapError .typing
      if !matrixTypeEqual product.output outputType then
        throw (.mismatchedMatrixTypes product.output outputType)
      return some (scopedMatrixFact output outputType (.affine rewritten) totalBound)

private def materializeIdentityFact
    (output : CoreWireRef)
    (type : MatrixTypeExpr)
    (input : MatrixFact) : Except VerifyError ScopedWireFact := do
  let relations := input.relations.map fun
    | .preimage _ source target trapdoor =>
        .preimage (.ofCoreWire output) source target trapdoor
    | .gadgetDecomposition _ target base digitCount =>
        .gadgetDecomposition (.ofCoreWire output) target base digitCount
  match input.primary with
  | .exact expression =>
      return {
        wire := output
        matrixType := some type
        fact := .matrix {
          subject := .ofCoreWire output
          primary := .exact (.scalarMultiply (.constant 1) expression)
          relations
          totalNormBound := input.totalNormBound
        }
      }
  | .affine _ =>
      return boundedFact output type input.totalNormBound relations

structure RuleApplication where
  stage : StageId
  scope : StaticScopeId := rootScope
  nodeId : Nat
  node : Mxx.Ir.Node
  rule : Rule

/-- Apply one inferred leaf rule. Every output fact is reconstructed from the frozen node. -/
def applyRule
    (facts : ScopedWireFactTable)
    (application : RuleApplication) :
    Except VerifyError (ScopedWireFactTable × DerivedObligations × List EndpointFact) := do
  if !isInitialRuleEnabled application.rule then
    throw (.disabledRule application.rule)
  let output := scopedOutputWire application.stage application.scope application.nodeId
  let noObligations : DerivedObligations := { static := [], input := [], semantic := [] }
  match application.rule, application.node.kind with
  | .introduceExactConstant, .zeroMatrix type =>
      return (facts ++ [exactFact output type (.zero type) (.constant 0)], noObligations, [])
  | .introduceExactConstant, .identityMatrix type =>
      let expression := MatrixExpr.wire (matrixInstance output type)
      return (facts ++ [exactFact output type expression (.constant 1)], noObligations, [])
  | .introduceExactConstant, .constantMatrix type _ =>
      let expression := MatrixExpr.wire (matrixInstance output type)
      return (facts ++ [exactFact output type expression (centeredBound type)], noObligations, [])
  | .introduceExactConstant, .gadgetMatrix type base =>
      let expression := MatrixExpr.gadget type base
      return (facts ++ [exactFact output type expression (centeredBound type)], noObligations, [])
  | .introduceGaussian, .gaussianSample type cutoff =>
      return (facts ++ [boundedFact output type (.parameter cutoff)], noObligations, [])
  | .introduceHash, .hashSample type .plain _ _ _ _ none none =>
      let expression := MatrixExpr.wire (matrixInstance output type)
      return (facts ++ [exactFact output type expression (centeredBound type)], noObligations, [])
  | .introduceTrapdoorSample, .trapdoorSample type _ =>
      let privateWire := scopedOutputWire application.stage application.scope application.nodeId 1
      let expression := MatrixExpr.wire (matrixInstance output type)
      let publicFact := exactFact output type expression (centeredBound type)
      let privateFact : ScopedWireFact := {
        wire := privateWire
        matrixType := none
        fact := .trapdoor {
          privatePort := .ofCoreWire privateWire
          publicPort := .ofCoreWire output
          publicMatrix := expression
        }
      }
      return (facts ++ [publicFact, privateFact], noObligations, [])
  | .introducePreimage, .preimageSample type cutoff =>
      match application.node.arguments with
      | [sourceRef, trapdoorRef, targetRef] =>
          let ⟨source, sourceType⟩ ←
            requireMatrix facts (scopedWire application.stage application.scope sourceRef)
          let trapdoor ← requireTrapdoor facts
            (scopedWire application.stage application.scope trapdoorRef)
          if trapdoor.publicPort != source.subject then
            throw (.trapdoorPublicMismatch
              (scopedWire application.stage application.scope trapdoorRef))
          let ⟨_target, targetType⟩ ←
            requireMatrix facts (scopedWire application.stage application.scope targetRef)
          let relation := MatrixRelation.preimage (.ofCoreWire output)
            (matrixInstance (scopedWire application.stage application.scope sourceRef) sourceType)
            (matrixInstance (scopedWire application.stage application.scope targetRef) targetType)
            trapdoor.privatePort
          return (facts ++ [boundedFact output type (.parameter cutoff) [relation]],
            noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .decomposeGadget, .gadgetDecompose type base digitCount =>
      match application.node.arguments with
      | [targetRef] =>
          let ⟨_target, targetType⟩ ←
            requireMatrix facts (scopedWire application.stage application.scope targetRef)
          let bound := BoundExpr.maximum (.floorDivide (.absolute base) 2) (.constant 1)
          let relation := MatrixRelation.gadgetDecomposition (.ofCoreWire output)
            (matrixInstance (scopedWire application.stage application.scope targetRef) targetType)
            base digitCount
          return (facts ++ [boundedFact output type bound [relation]], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .addAffine, .matrixAdd =>
      match application.node.arguments with
      | [leftRef, rightRef] =>
          let ⟨left, leftType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope leftRef)
          let ⟨right, rightType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope rightRef)
          if !matrixTypeEqual leftType rightType then
            throw (.mismatchedMatrixTypes leftType rightType)
          let outputType ← requireDeclaredMatrixType application.stage application.nodeId
            application.node
          if !matrixTypeEqual leftType outputType then
            throw (.mismatchedMatrixTypes leftType outputType)
          let result ← addOrSubtractFact false output leftType left right
          let result ← normalizeScopedFact result
          return (facts ++ [result], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .subtractAffine, .matrixSubtract =>
      match application.node.arguments with
      | [leftRef, rightRef] =>
          let ⟨left, leftType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope leftRef)
          let ⟨right, rightType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope rightRef)
          if !matrixTypeEqual leftType rightType then
            throw (.mismatchedMatrixTypes leftType rightType)
          let outputType ← requireDeclaredMatrixType application.stage application.nodeId
            application.node
          if !matrixTypeEqual leftType outputType then
            throw (.mismatchedMatrixTypes leftType outputType)
          let result ← addOrSubtractFact true output leftType left right
          let result ← normalizeScopedFact result
          return (facts ++ [result], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .negateAffine, .matrixNegate =>
      match application.node.arguments with
      | [inputRef] =>
          let ⟨input, inputType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope inputRef)
          let outputType ← requireDeclaredMatrixType application.stage application.nodeId
            application.node
          if !matrixTypeEqual inputType outputType then
            throw (.mismatchedMatrixTypes inputType outputType)
          let result ← deriveMatrixNegate (.ofCoreWire output) inputType input
            |>.mapError .matrixAffine
          let normalized ← normalizeMatrixFact output result.fact
          let derived : ScopedWireFact := {
            wire := output
            matrixType := some inputType
            fact := .matrix normalized
          }
          return (facts ++ [derived], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .materializeIdentity, .matrixScale (.constant 1) =>
      match application.node.arguments with
      | [inputRef] =>
          let ⟨input, inputType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope inputRef)
          let outputType ← requireDeclaredMatrixType application.stage application.nodeId
            application.node
          if !matrixTypeEqual inputType outputType then
            throw (.mismatchedMatrixTypes inputType outputType)
          let result ← deriveMatrixScaleOne (.ofCoreWire output) inputType (.constant 1) input
            |>.mapError .matrixAffine
          let derived : ScopedWireFact := {
            wire := output
            matrixType := some inputType
            fact := .matrix result.fact
          }
          return (facts ++ [derived], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | .multiplyAffineRight, .matrixMultiply
  | .multiplyAffineLeft, .matrixMultiply =>
      match application.node.arguments with
      | [leftRef, rightRef] =>
          let ⟨left, leftType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope leftRef)
          let ⟨right, rightType⟩ ← requireMatrix facts
            (scopedWire application.stage application.scope rightRef)
          let outputType ← requireDeclaredMatrixType application.stage application.nodeId
            application.node
          match ← rewritePreimageProduct? output outputType leftType rightType left right facts with
          | some rewritten =>
              let rewritten ← normalizeScopedFact rewritten
              return (facts ++ [rewritten], noObligations, [])
          | none => pure ()
          let result ← deriveMatrixMultiply (.ofCoreWire output) leftType rightType left right
            |>.mapError .matrixAffine
          if !matrixTypeEqual result.type outputType then
            throw (.mismatchedMatrixTypes result.type outputType)
          let derived : ScopedWireFact := {
            wire := output
            matrixType := some outputType
            fact := .matrix result.fact
          }
          let derived ← normalizeScopedFact derived
          return (facts ++ [derived], noObligations, [])
      | _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)
  | _, _ => throw (.unsupportedNode application.stage ⟨application.nodeId⟩)

private def inferredRule : Mxx.Ir.NodeKind → Option Rule
  | .zeroMatrix _ | .identityMatrix _ | .constantMatrix _ _ | .gadgetMatrix _ _ =>
      some .introduceExactConstant
  | .gaussianSample _ _ => some .introduceGaussian
  | .hashSample _ .plain _ _ _ _ none none => some .introduceHash
  | .trapdoorSample _ _ => some .introduceTrapdoorSample
  | .preimageSample _ _ => some .introducePreimage
  | .gadgetDecompose _ _ _ => some .decomposeGadget
  | .matrixAdd => some .addAffine
  | .matrixSubtract => some .subtractAffine
  | .matrixNegate => some .negateAffine
  | .matrixScale (.constant 1) => some .materializeIdentity
  | .constantInt _ | .evaluateInt _ | .constantBool _ | .boolToInt | .intBinary _ |
      .intCompare _ | .bitExtract _ | .extractCoefficient _ => none
  | _ => none

private def inferredMatrixMultiplyRule
    (facts : ScopedWireFactTable)
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node) : Except VerifyError Rule := do
  match node.arguments with
  | [leftRef, rightRef] =>
      let ⟨left, _⟩ ← requireMatrix facts (scopedWire stage scope leftRef)
      let ⟨right, _⟩ ← requireMatrix facts (scopedWire stage scope rightRef)
      match left.primary, right.primary with
      | .exact _, .affine rightForm =>
          if rightForm.terms.isEmpty then return .multiplyAffineRight
          return .multiplyAffineLeft
      | .affine leftForm, .affine rightForm =>
          if leftForm.terms.isEmpty && !rightForm.terms.isEmpty then
            return .multiplyAffineLeft
          return .multiplyAffineRight
      | _, _ => return .multiplyAffineRight
  | _ => throw (.unsupportedNode stage ⟨nodeId⟩)

private def maxBounds : List BoundExpr → BoundExpr
  | [] => .constant 0
  | head :: tail => tail.foldl .maximum head

private def inferScalarOrSelect
    (stage : StageId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (facts : ScopedWireFactTable)
    (scope : StaticScopeId := rootScope) : Except VerifyError ScopedWireFactTable := do
  let output := scopedOutputWire stage scope nodeId
  match node.kind, node.arguments with
  | .constantInt value, [] =>
      return facts ++ [scalarFact output (.integer {
        expression := .intConstant value
        lower := .integer (.constant value)
        upper := .integer (.constant value)
      })]
  | .evaluateInt value, [] =>
      return facts ++ [scalarFact output (.integer {
        expression := .parameter value
        lower := .integer value
        upper := .integer value
      })]
  | .constantBool value, [] =>
      return facts ++ [scalarFact output (.boolean { expression := .boolConstant value })]
  | .boolToInt, [inputRef] =>
      let input ← requireBoolean facts (scopedWire stage scope inputRef)
      return facts ++ [scalarFact output (.integer {
        expression := .boolToInt input.expression
        lower := .integer (.constant 0)
        upper := .integer (.constant 1)
      })]
  | .intBinary operation, [leftRef, rightRef] =>
      let left ← requireInteger facts (scopedWire stage scope leftRef)
      let right ← requireInteger facts (scopedWire stage scope rightRef)
      let rule ← inferScalarControlRule (.intBinary operation) |>.mapError .scalarControl
      match ← deriveIntBinaryOutput rule left right |>.mapError .scalarControl with
      | .integer expression lower upper =>
          return facts ++ [scalarFact output (.integer { expression, lower, upper })]
      | .integerPending .. => throw (.unsupportedNode stage ⟨nodeId⟩)
      | .boolean .. => throw (.unsupportedNode stage ⟨nodeId⟩)
  | .intCompare operation, [leftRef, rightRef] =>
      let left ← requireInteger facts (scopedWire stage scope leftRef)
      let right ← requireInteger facts (scopedWire stage scope rightRef)
      let rule ← inferScalarControlRule (.intCompare operation) |>.mapError .scalarControl
      match ← deriveCompareOutput rule left right |>.mapError .scalarControl with
      | .boolean expression =>
          return facts ++ [scalarFact output (.boolean { expression })]
      | _ => throw (.unsupportedNode stage ⟨nodeId⟩)
  | .select, indexRef :: branchRefs =>
      let index ← requireInteger facts (scopedWire stage scope indexRef)
      let branches ← branchRefs.mapM (fun reference ↦
        requireMatrix facts (scopedWire stage scope reference))
      match branches with
      | [] => throw (.unsupportedNode stage ⟨nodeId⟩)
      | (_, type) :: tail =>
          let outputType ← requireDeclaredMatrixType stage nodeId node
          if !matrixTypeEqual type outputType then
            throw (.mismatchedMatrixTypes type outputType)
          match tail.find? (fun branch => !matrixTypeEqual branch.2 type) with
          | some mismatch => throw (.mismatchedMatrixTypes type mismatch.2)
          | none => pure ()
          let result ← deriveMatrixSelect (.ofCoreWire output) outputType index.expression
            (branches.map fun branch => (branch.2, branch.1)) |>.mapError (.matrixSelect output)
          return facts ++ [{
            wire := output
            matrixType := some outputType
            fact := .matrix result
          }]
  | .thresholdDecodeBool ciphertextModulus plaintextModulus _, [inputRef] =>
      let ⟨input, _⟩ ← requireMatrix facts (scopedWire stage scope inputRef)
      let mut outputs : ScopedWireFactTable := []
      for port in [0:node.outputCount] do
        outputs := outputs ++ [scalarFact (scopedOutputWire stage scope nodeId port) (.boolean {
          expression := .thresholdDecodeBool input.subject ciphertextModulus plaintextModulus
            (.constant port)
        })]
      return facts ++ outputs
  | _, _ => return facts

/-- Parameter-only side conditions for the nonnegative scalar interval fragment.  The analyzer
derives these from the already-established input facts; callers cannot supply them.  Phase B
evaluates them exactly, including recurrence-result endpoints. -/
private def scalarRangeObligations
    (stage : StageId)
    (scope : StaticScopeId)
    (_nodeId : Nat)
    (node : Mxx.Ir.Node)
    (facts : ScopedWireFactTable) : Except VerifyError (List StaticObligation) := do
  match node.kind, node.arguments with
  | .intBinary operation, [leftRef, rightRef] =>
      let left ← requireInteger facts (scopedWire stage scope leftRef)
      let right ← requireInteger facts (scopedWire stage scope rightRef)
      let ordered := [
        .intBoundsOrdered left.lower left.upper,
        .intBoundsOrdered right.lower right.upper
      ]
      match operation with
      | .multiply => return ordered ++ [
          .intBoundNonnegative left.lower,
          .intBoundNonnegative right.lower
        ]
      | .divide | .remainder => return ordered ++ [
          .intBoundNonnegative left.lower,
          .intBoundPositive right.lower
        ]
      | .add | .subtract => return ordered
  | _, _ => return []

def inferNodeFacts
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (facts : ScopedWireFactTable) : Except VerifyError ScopedWireFactTable := do
  let inferred : Option Rule ← match node.kind with
    | .matrixMultiply => (inferredMatrixMultiplyRule facts stage scope nodeId node).map some
    | _ => pure (inferredRule node.kind)
  match inferred with
  | some rule =>
      let ⟨nextFacts, _, _⟩ ← applyRule facts { stage, scope, nodeId, node, rule }
      return nextFacts
  | none =>
      match node.kind with
      | .input _ => return facts
      | .constantInt _ | .evaluateInt _ | .constantBool _ | .boolToInt |
          .intBinary _ | .intCompare _ | .select |
          .thresholdDecodeBool _ _ _ => inferScalarOrSelect stage nodeId node facts scope
      | .bitExtract _ | .extractCoefficient _ => return facts
      | .uniformSample _ (.constant (-1)) (.constant 1) =>
          let output := scopedOutputWire stage scope nodeId
          let outputType ← requireDeclaredMatrixType stage nodeId node
          let _ ← inferTransformRule node.kind |>.mapError .transform
          let result := deriveUniformMinusOneOneFact (.ofCoreWire output)
          return facts ++ [{
            wire := output
            matrixType := some outputType
            fact := .matrix result
          }]
      | .reshape _ _ =>
          match node.arguments with
          | [inputRef] =>
              let output := scopedOutputWire stage scope nodeId
              let ⟨input, _inputType⟩ ← requireMatrix facts
                (scopedWire stage scope inputRef)
              let outputType ← requireDeclaredMatrixType stage nodeId node
              let _ ← inferTransformRule node.kind |>.mapError .transform
              let result ← deriveReshapeBoundedFact (.ofCoreWire output) input
                |>.mapError .transform
              return facts ++ [{
                wire := output
                matrixType := some outputType
                fact := .matrix result
              }]
          | _ => throw (.unsupportedNode stage ⟨nodeId⟩)
      | .concat axis =>
          let output := scopedOutputWire stage scope nodeId
          let inputsAndTypes ← node.arguments.mapM fun inputRef ↦
            requireMatrix facts (scopedWire stage scope inputRef)
          let inputs := inputsAndTypes.map (·.1)
          let inputTypes := inputsAndTypes.map (·.2)
          let outputType ← requireDeclaredMatrixType stage nodeId node
          let expectedType ← match concatOutputType axis inputTypes with
            | some type => pure type
            | none => throw (.unsupportedNode stage ⟨nodeId⟩)
          if !matrixTypeEqual expectedType outputType then
            throw (.mismatchedMatrixTypes expectedType outputType)
          let _ ← inferTransformRule node.kind |>.mapError .transform
          let result ← deriveConcatFact axis (.ofCoreWire output) inputs
            |>.mapError .transform
          return facts ++ [{
            wire := output
            matrixType := some outputType
            fact := .matrix result
          }]
      | .slice rows columns =>
          match node.arguments with
          | [inputRef] =>
              let output := scopedOutputWire stage scope nodeId
              let ⟨input, _inputType⟩ ← requireMatrix facts (scopedWire stage scope inputRef)
              let outputType ← requireDeclaredMatrixType stage nodeId node
              let _ ← inferTransformRule node.kind |>.mapError .transform
              let result ← deriveSliceFact (.ofCoreWire output) rows columns input
                |>.mapError .transform
              return facts ++ [{
                wire := output
                matrixType := some outputType
                fact := .matrix result
              }]
          | _ => throw (.unsupportedNode stage ⟨nodeId⟩)
      | _ => throw (.unsupportedNode stage ⟨nodeId⟩)

@[simp] theorem inferNodeFacts_input
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (name : String) (outputCount : Nat) (outputTypes : List Mxx.Ir.WireTypeExpr)
    (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .input name
      arguments := []
      outputCount
      outputTypes
    } facts = .ok facts := by
  rfl

@[simp] theorem inferNodeFacts_constantInt
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (value : Int) (outputCount : Nat) (outputTypes : List Mxx.Ir.WireTypeExpr)
    (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .constantInt value
      arguments := []
      outputCount
      outputTypes
    } facts = .ok (facts ++ [{
      wire := { stage, scope, node := ⟨nodeId⟩, port := 0 }
      matrixType := none
      fact := .integer {
        expression := .intConstant value
        lower := .integer (.constant value)
        upper := .integer (.constant value)
      }
    }]) := by
  rfl

@[simp] theorem inferNodeFacts_evaluateInt
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (expression : IntExpr) (outputCount : Nat) (outputTypes : List Mxx.Ir.WireTypeExpr)
    (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .evaluateInt expression
      arguments := []
      outputCount
      outputTypes
    } facts = .ok (facts ++ [{
      wire := { stage, scope, node := ⟨nodeId⟩, port := 0 }
      matrixType := none
      fact := .integer {
        expression := .parameter expression
        lower := .integer expression
        upper := .integer expression
      }
    }]) := by
  rfl

@[simp] theorem inferNodeFacts_constantBool
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (value : Bool) (outputCount : Nat) (outputTypes : List Mxx.Ir.WireTypeExpr)
    (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .constantBool value
      arguments := []
      outputCount
      outputTypes
    } facts = .ok (facts ++ [{
      wire := { stage, scope, node := ⟨nodeId⟩, port := 0 }
      matrixType := none
      fact := .boolean { expression := .boolConstant value }
    }]) := by
  rfl

theorem inferNodeFacts_zeroMatrix
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (matrixType : MatrixTypeExpr) (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr) (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .zeroMatrix matrixType
      arguments := []
      outputCount
      outputTypes
    } facts = (applyRule facts {
      stage, scope, nodeId
      node := {
        kind := .zeroMatrix matrixType
        arguments := []
        outputCount
        outputTypes
      }
      rule := .introduceExactConstant
    }).map (fun result => result.1) := by
  rfl

theorem inferNodeFacts_identityMatrix
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (matrixType : MatrixTypeExpr) (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr) (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .identityMatrix matrixType
      arguments := []
      outputCount
      outputTypes
    } facts = (applyRule facts {
      stage, scope, nodeId
      node := {
        kind := .identityMatrix matrixType
        arguments := []
        outputCount
        outputTypes
      }
      rule := .introduceExactConstant
    }).map (fun result => result.1) := by
  rfl

theorem inferNodeFacts_constantMatrix
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (matrixType : MatrixTypeExpr) (coefficients : List IntExpr) (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr) (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .constantMatrix matrixType coefficients
      arguments := []
      outputCount
      outputTypes
    } facts = (applyRule facts {
      stage, scope, nodeId
      node := {
        kind := .constantMatrix matrixType coefficients
        arguments := []
        outputCount
        outputTypes
      }
      rule := .introduceExactConstant
    }).map (fun result => result.1) := by
  rfl

theorem inferNodeFacts_gadgetMatrix
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (matrixType : MatrixTypeExpr) (base : IntExpr) (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr) (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .gadgetMatrix matrixType base
      arguments := []
      outputCount
      outputTypes
    } facts = (applyRule facts {
      stage, scope, nodeId
      node := {
        kind := .gadgetMatrix matrixType base
        arguments := []
        outputCount
        outputTypes
      }
      rule := .introduceExactConstant
    }).map (fun result => result.1) := by
  rfl

theorem inferNodeFacts_gaussianSample
    (stage : StageId) (scope : StaticScopeId) (nodeId : Nat)
    (matrixType : MatrixTypeExpr) (cutoff : IntExpr) (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr) (facts : ScopedWireFactTable) :
    inferNodeFacts stage scope nodeId {
      kind := .gaussianSample matrixType cutoff
      arguments := []
      outputCount
      outputTypes
    } facts = (applyRule facts {
      stage, scope, nodeId
      node := {
        kind := .gaussianSample matrixType cutoff
        arguments := []
        outputCount
        outputTypes
      }
      rule := .introduceGaussian
    }).map (fun result => result.1) := by
  rfl

/-- Analyze a suffix while preserving its absolute SSA node identifier. Soundness induction uses
this entrypoint after splitting an executable path at its head. -/
def inferRulesFrom
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (nodes : List Mxx.Ir.Node)
    (facts : ScopedWireFactTable) : Except VerifyError ScopedWireFactTable := do
  match nodes with
  | [] => return facts
  | node :: tail =>
      inferRulesFrom stage scope (nodeId + 1) tail
        (← inferNodeFacts stage scope nodeId node facts)
termination_by nodes.length

/-- Deterministically infer and apply all currently verified leaf rules in one static scope. -/
def inferRules
    (stage : StageId)
    (nodes : List Mxx.Ir.Node)
    (initialFacts : ScopedWireFactTable := [])
    (scope : StaticScopeId := rootScope) : Except VerifyError ScopedWireFactTable :=
  inferRulesFrom stage scope 0 nodes initialFacts

structure AnalysisState where
  facts : ScopedWireFactTable
  families : List (JointFamilyId × JointFamilyFact) := []
  /-- Closed protocol-input family contracts.  These are analyzer-owned data copied from the
  verified bundle, not certificate input.  Unlike executable loop families they do not need a
  synthetic `JointFamilyFact`: an element fact is derived directly from its input contract. -/
  protocolFamilies : List (JointFamilyId × InputValueContract) := []
  recurrences : List (FactRecurrenceInstanceRef × FactRecurrence) := []
  symbolicRecurrences : List SymbolicRecurrenceTransfer := []
  expressionArena : ExpressionArena := { entries := [] }
  symbolicFormArena : SymbolicMatrixFormArena := {}
  boundWitnessArena : BoundWitnessArena := {}
  symbolicMatrixFacts : List MatrixSymbolicFact := []
  staticObligations : List StaticObligation := []

private def lookupJointFamily
    (joint : JointFamilyId) : List (JointFamilyId × JointFamilyFact) → Option JointFamilyFact
  | [] => none
  | entry :: tail => if entry.1 = joint then some entry.2 else lookupJointFamily joint tail

private def lookupProtocolFamily
    (joint : JointFamilyId) : List (JointFamilyId × InputValueContract) →
      Option InputValueContract
  | [] => none
  | entry :: tail =>
      if entry.1 = joint then some entry.2 else lookupProtocolFamily joint tail

/-- Recover the closed input contract represented by an aggregate.  Nested protocol-input
families are resolved structurally from the root contract, so no dynamic family table or lane
enumeration is required. -/
private def protocolFamilyContract
    (families : List (JointFamilyId × InputValueContract)) :
    FamilyAggregateRef → Option InputValueContract
  | .joint joint 0 _ => lookupProtocolFamily joint families
  | .familyElement parent _ => do
      let parentContract ← protocolFamilyContract families parent
      match parentContract with
      | .family _ element => some element
      | _ => none
  | _ => none

private def protocolFamilyElementContract
    (families : List (JointFamilyId × InputValueContract))
    (aggregate : FamilyAggregateRef) : Option InputValueContract := do
  let contract ← protocolFamilyContract families aggregate
  match contract with
  | .family _ element => some element
  | _ => none

private def protocolFamilyContracts
    (contract : InputContract) : List (JointFamilyId × InputValueContract) :=
  contract.inputs.filterMap fun entry ↦
    match entry.2.2 with
    | .family count element => some (⟨entry.1.name⟩, .family count element)
    | _ => none

private def instantiateProtocolFamilyElement
    (aggregate : FamilyAggregateRef)
    (index : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (contract : InputValueContract)
    (output : CoreWireRef) : ScopedWireFact :=
  let provenance := ValueInstanceRef.familyElement aggregate index
  match contract with
  | .matrixExact type => {
      wire := output
      matrixType := some type
      fact := .matrix {
        subject := .ofCoreWire output
        primary := .exact (.wire { value := provenance, type })
        relations := []
        totalNormBound := centeredBound type
      }
    }
  | .matrixBounded type bound => {
      wire := output
      matrixType := some type
      fact := .matrix {
        subject := .ofCoreWire output
        primary := .affine { terms := [], noiseBound := bound.toBoundExpr }
        relations := []
        totalNormBound := bound.toBoundExpr
      }
    }
  | .integerRange lower upper => scalarFact output (.integer {
      expression := .familyElement .integer aggregate index indexExpression
      lower := .integer lower
      upper := .integer upper
    })
  | .boolean => scalarFact output (.boolean {
      expression := .familyElement .boolean aggregate index indexExpression
    })
  | .bytes _ => scalarFact output (.bytes provenance)
  | .family count element => scalarFact output (.family {
      aggregate := .familyElement aggregate index
      count
      elementSchema := element.factSchema
    })

private def resolveJointFamilyPort
    (port : FamilyFact)
    (families : List (JointFamilyId × JointFamilyFact)) :
    Except VerifyError (JointFamilyFact × Nat) :=
  match port.aggregate with
  | .joint joint outputSlot _ =>
      match lookupJointFamily joint families with
      | some family => .ok (family, outputSlot)
      | none => .error (.missingFamily joint)
  | _ => .error (.invalidLoopArity ⟨"family-get"⟩ ⟨0⟩)

private def templateMatrixType : ValueFactTemplate → Option MatrixTypeExpr
  | ⟨_, .matrix type _ _ _⟩ => some type
  | _ => none

private def instantiateFamilyElement
    (family : JointFamilyFact)
    (slot : Nat)
    (index : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (output : CoreWireRef) : Except VerifyError ScopedWireFact := do
  let template ← match family.elementTuple[slot]? with
    | some template => pure template
    | none => throw (.invalidFamilySlot family.id slot)
  let loopSite ← match family.outputFamilies.head? with
    | some wire => pure { stage := wire.stage, scope := wire.scope, node := wire.node }
    | none => throw (.invalidFamilySlot family.id slot)
  let aggregate := FamilyAggregateRef.joint family.id slot []
  let provenance := ValueInstanceRef.familyElement aggregate index
  let instantiated := instantiateParallelTemplate loopSite index family.id
    family.bodyOutputTemplates (.ofCoreWire output) template
  let instantiated := match instantiated with
    | .integer fact => .integer { fact with
        expression := .familyElement .integer aggregate index indexExpression }
    | .boolean fact => .boolean { fact with
        expression := .familyElement .boolean aggregate index indexExpression }
    | .bytes _ => .bytes provenance
    | fact => fact
  return { wire := output, matrixType := templateMatrixType template, fact := instantiated }

private def instantiateAbstractFamilyElement
    (rootSlot : Nat)
    (parent : FamilyAggregateRef)
    (index : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (schema : ValueFactSchema)
    (output : CoreWireRef) : Except VerifyError ScopedWireFact := do
  let aggregate := FamilyAggregateRef.familyElement parent index
  let result : ValueFact × Option MatrixTypeExpr ← match schema with
    | .matrix type primary relations representation => do
        if !relations.isEmpty then
          throw (.relationBearingCarriedMatrix output.stage output.node rootSlot)
        let primary := match primary with
          | .exact => .exact (.carriedInput type
              (.familyElement rootSlot index (.exactExpression rootSlot)))
          | .affine terms => .affine {
              terms := terms.zipIdx.map fun (term, termIndex) => {
                coefficient := {
                  expression := .carriedInput term.coefficientType
                    (.familyElement rootSlot index (.affineCoefficient rootSlot termIndex))
                  normBound := .carriedInput
                    (.familyElement rootSlot index (.affineCoefficientBound rootSlot termIndex))
                }
                basis := .carriedInput term.basisType
                  (.familyElement rootSlot index (.affineBasis rootSlot termIndex))
                mode := term.mode
              }
              noiseBound := .carriedInput
                (.familyElement rootSlot index (.affineNoiseBound rootSlot))
            }
        pure (
          .matrix {
          subject := .ofCoreWire output
          primary
          relations := []
          totalNormBound := .carriedInput
            (.familyElement rootSlot index (.matrixTotalBound rootSlot))
          coefficientRepresentation := representation
        }, some type)
    | .integer => pure (.integer {
        expression := .familyElement .integer parent index indexExpression
        lower := .carriedInput (.familyElement rootSlot index (.lower rootSlot))
        upper := .carriedInput (.familyElement rootSlot index (.upper rootSlot))
      }, none)
    | .boolean => pure (.boolean {
        expression := .familyElement .boolean parent index indexExpression
      }, none)
    | .family count element => pure (.family {
        aggregate
        count
        elementSchema := element
      }, none)
    | _ => throw (.unsupportedCarriedKind output.stage output.node rootSlot)
  return {
    wire := output
    matrixType := result.2
    fact := result.1
  }

private def applyFamilyGet
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (state : AnalysisState) : Except VerifyError AnalysisState := do
  let familyRef ← match node.arguments.head? with
    | some reference => pure reference
    | none => throw (.invalidLoopArity stage ⟨nodeId⟩)
  let familyPort ← match lookupScopedFact (scopedWire stage scope familyRef) state.facts with
    | some { fact := .family family, .. } => pure family
    | _ => throw (.invalidLoopArity stage ⟨nodeId⟩)
  let ⟨indexExpression, arena, index⟩ ← match node.kind with
    | .familyGetStatic value =>
        let expression : RuntimeExpr .integer := .parameter value
        match state.expressionArena.internInteger expression with
        | some (arena, reference) => pure (expression, arena, reference)
        | none => throw .invalidExpressionReference
    | .familyGetDynamic =>
        match node.arguments with
        | [_, indexRef] =>
            let indexFact ← requireInteger state.facts (scopedWire stage scope indexRef)
            match state.expressionArena.internInteger indexFact.expression with
            | some (arena, reference) => pure (indexFact.expression, arena, reference)
            | none => throw .invalidExpressionReference
        | _ => throw (.invalidLoopArity stage ⟨nodeId⟩)
    | _ => throw (.unsupportedNode stage ⟨nodeId⟩)
  let output := scopedOutputWire stage scope nodeId
  let fact ← match familyPort.aggregate with
    | .joint joint outputSlot _ =>
        match lookupJointFamily joint state.families with
        | some family => instantiateFamilyElement family outputSlot index indexExpression output
        | none =>
            let contract ← match protocolFamilyElementContract state.protocolFamilies
                familyPort.aggregate with
              | some contract => pure contract
              | none => throw (.missingFamily joint)
            pure (instantiateProtocolFamilyElement familyPort.aggregate index indexExpression
              contract output)
    | aggregate@(.familyElement ..) =>
        let contract ← match protocolFamilyElementContract state.protocolFamilies aggregate with
          | some contract => pure contract
          | none => throw (.invalidLoopArity stage ⟨nodeId⟩)
        pure (instantiateProtocolFamilyElement aggregate index indexExpression contract output)
    | .carriedInput slot =>
        instantiateAbstractFamilyElement slot familyPort.aggregate index indexExpression
          familyPort.elementSchema output
    | _ => throw (.invalidLoopArity stage ⟨nodeId⟩)
  return { state with facts := state.facts ++ [fact], expressionArena := arena }

private def internIntegerBranches :
    ExpressionArena → List IntegerFact → Option (ExpressionArena × List (RuntimeExprRef .integer))
  | arena, [] => some (arena, [])
  | arena, fact :: tail => do
      let ⟨nextArena, reference⟩ ← arena.internInteger fact.expression
      let ⟨finalArena, references⟩ ← internIntegerBranches nextArena tail
      return (finalArena, reference :: references)

private def internBooleanBranches :
    ExpressionArena → List BooleanFact → Option (ExpressionArena × List (RuntimeExprRef .boolean))
  | arena, [] => some (arena, [])
  | arena, fact :: tail => do
      let ⟨nextArena, reference⟩ ← arena.internBoolean fact.expression
      let ⟨finalArena, references⟩ ← internBooleanBranches nextArena tail
      return (finalArena, reference :: references)

/-- Stateful scalar select. Runtime expressions store analyzer-owned references to branch
expressions. Integer intervals use the pointwise minimum of branch lower bounds and maximum of
branch upper bounds; these constructors are evaluated and proved by the shared bound layer. -/
private def applyScalarSelect
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (state : AnalysisState) : Except VerifyError AnalysisState := do
  let indexRef :: branchRefs := node.arguments
    | throw (.unsupportedNode stage ⟨nodeId⟩)
  let index ← requireInteger state.facts (scopedWire stage scope indexRef)
  let output := scopedOutputWire stage scope nodeId
  match node.outputTypes with
  | [.integer] | [.constantInt] =>
      let branches ← branchRefs.mapM fun reference ↦
        requireInteger state.facts (scopedWire stage scope reference)
      let first :: rest := branches | throw (.unsupportedNode stage ⟨nodeId⟩)
      let ⟨arena, references⟩ ← match internIntegerBranches state.expressionArena branches with
        | some result => pure result
        | none => throw .invalidExpressionReference
      let fact := scalarFact output (.integer {
        expression := .select .integer index.expression references
        lower := rest.foldl (fun lower branch => .minimum lower branch.lower) first.lower
        upper := rest.foldl (fun upper branch => .maximum upper branch.upper) first.upper
      })
      return { state with facts := state.facts ++ [fact], expressionArena := arena }
  | [.boolean] | [.constantBool] =>
      let branches ← branchRefs.mapM fun reference ↦
        requireBoolean state.facts (scopedWire stage scope reference)
      if branches.isEmpty then throw (.unsupportedNode stage ⟨nodeId⟩)
      let ⟨arena, references⟩ ← match internBooleanBranches state.expressionArena branches with
        | some result => pure result
        | none => throw .invalidExpressionReference
      let fact := scalarFact output (.boolean {
        expression := .select .boolean index.expression references
      })
      return { state with facts := state.facts ++ [fact], expressionArena := arena }
  | _ => throw (.unsupportedNode stage ⟨nodeId⟩)

private def bitPositionIsStructurallyNonnegative : IntExpr → Bool
  | .constant value => 0 ≤ value
  | .loopIndex _ => true
  | _ => false

/-- Bit extraction is accepted only where Phase A has direct nonnegativity evidence for the bit
position. Loop indices are natural by executable loop construction; arbitrary parameters remain
unsupported until a checked range obligation is connected. -/
private def applyBitExtract
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (position : IntExpr)
    (state : AnalysisState) : Except VerifyError AnalysisState := do
  let [inputRef] := node.arguments | throw (.unsupportedNode stage ⟨nodeId⟩)
  if !bitPositionIsStructurallyNonnegative position then
    throw (.unsupportedNode stage ⟨nodeId⟩)
  let input ← requireInteger state.facts (scopedWire stage scope inputRef)
  let output := scopedOutputWire stage scope nodeId
  match deriveBitExtractOutput input position with
  | .boolean expression =>
      return { state with
        facts := state.facts ++ [scalarFact output (.boolean { expression })] }
  | _ => throw (.unsupportedNode stage ⟨nodeId⟩)

/-- Coefficient extraction first interns the exact current matrix identity. The executable
operation observes `reduceCoefficient q value`, so a positive modulus is the only condition needed
for the canonical interval `[0,q-1]`; no matrix-provenance metadata is involved. -/
private def applyExtractCoefficient
    (stage : StageId)
    (scope : StaticScopeId)
    (nodeId : Nat)
    (node : Mxx.Ir.Node)
    (position : IntExpr)
    (state : AnalysisState) : Except VerifyError AnalysisState := do
  let [inputRef] := node.arguments | throw (.unsupportedNode stage ⟨nodeId⟩)
  let ⟨input, inputType⟩ ← requireMatrix state.facts (scopedWire stage scope inputRef)
  let expression := MatrixExpr.wire { value := input.subject, type := inputType }
  let ⟨arena, matrixReference⟩ ← match state.expressionArena.internMatrix expression with
    | some result => pure result
    | none => throw .invalidExpressionReference
  let derived := deriveExtractCoefficientOutput matrixReference position inputType.modulus
  let output := scopedOutputWire stage scope nodeId
  match derived with
  | .integer scalarExpression lower upper =>
      return {
        state with
        facts := state.facts ++ [scalarFact output (.integer {
          expression := scalarExpression
          lower
          upper
        })]
        expressionArena := arena
        staticObligations := state.staticObligations ++ [.positiveModulus inputType.modulus]
      }
  | _ => throw (.unsupportedNode stage ⟨nodeId⟩)

private def definitionByName
    (name : String) : List (String × Mxx.Ir.Scope) → Option Mxx.Ir.Scope
  | [] => none
  | definition :: tail => if definition.1 = name then some definition.2 else
      definitionByName name tail

private def seedParallelInputs
    (stage : StageId)
    (bodyScope : StaticScopeId)
    (body : Mxx.Ir.Scope)
    (loopSite : CoreNodeRef)
    (index : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (arguments : List Mxx.Ir.WireRef)
    (modes : List Mxx.Ir.LoopInputMode)
    (outerScope : StaticScopeId)
    (state : AnalysisState) : Except VerifyError ScopedWireFactTable := do
  let rec visit
      (names : List String)
      (arguments : List Mxx.Ir.WireRef)
      (modes : List Mxx.Ir.LoopInputMode)
      (accumulator : ScopedWireFactTable) : Except VerifyError ScopedWireFactTable := do
    match names, arguments, modes with
    | [], [], [] => return accumulator
    | name :: nameTail, argument :: argumentTail, mode :: modeTail =>
        let destination ← match inputNodeWireInScope stage bodyScope name body.nodes with
          | some wire => pure wire
          | none => throw (.missingProgramInput stage name)
        let sourceWire := scopedWire stage outerScope argument
        let source ← match lookupScopedFact sourceWire state.facts with
          | some fact => pure fact
          | none => throw (.missingInputFact stage ⟨loopSite.node.value⟩ argument)
        let seeded ← match mode with
          | .broadcast => pure (transportFact destination source)
          | .zip =>
              match source.fact with
              | .family port =>
                  match port.aggregate with
                  | .joint joint outputSlot _ =>
                      match lookupJointFamily joint state.families with
                      | some family =>
                          instantiateFamilyElement family outputSlot index indexExpression
                            destination
                      | none =>
                          let contract ← match protocolFamilyElementContract
                              state.protocolFamilies port.aggregate with
                            | some contract => pure contract
                            | none => throw (.missingFamily joint)
                          pure (instantiateProtocolFamilyElement port.aggregate index
                            indexExpression contract destination)
                  | aggregate@(.familyElement ..) =>
                      let contract ← match protocolFamilyElementContract state.protocolFamilies
                          aggregate with
                        | some contract => pure contract
                        | none => throw (.invalidLoopArity stage loopSite.node)
                      pure (instantiateProtocolFamilyElement aggregate index indexExpression
                        contract destination)
                  | _ => throw (.invalidLoopArity stage loopSite.node)
              | _ => throw (.invalidLoopArity stage loopSite.node)
          | .zipOffset _ => throw (.unsupportedNode stage loopSite.node)
        visit nameTail argumentTail modeTail (accumulator ++ [seeded])
    | _, _, _ => throw (.invalidLoopArity stage loopSite.node)
  visit body.inputNames arguments modes []

private def wireMatrixType : Mxx.Ir.WireTypeExpr → Option MatrixTypeExpr
  | .matrix type | .preimage type => some type
  | _ => none

def analyzeStateFrom
    (fuel : Nat)
    (stage : StageId)
    (definitions : List (String × Mxx.Ir.Scope))
    (scope : StaticScopeId)
    (instancePath : InstancePathExpr)
    (nodeId : Nat)
    (nodes : List Mxx.Ir.Node)
    (state : AnalysisState) : Except VerifyError AnalysisState := do
  match fuel with
  | 0 => throw (.invalidLoopArity stage ⟨nodeId⟩)
  | fuel + 1 =>
      match nodes with
      | [] => return state
      | node :: tail =>
          let next ← match node.kind with
            | .familyGetStatic _ | .familyGetDynamic =>
                applyFamilyGet stage scope nodeId node state
            | .select =>
                match node.outputTypes with
                | [.integer] | [.constantInt] | [.boolean] | [.constantBool] =>
                    applyScalarSelect stage scope nodeId node state
                | _ =>
                    pure { state with
                      facts := (← inferNodeFacts stage scope nodeId node state.facts) }
            | .bitExtract position =>
                applyBitExtract stage scope nodeId node position state
            | .extractCoefficient position =>
                applyExtractCoefficient stage scope nodeId node position state
            | .parallelLoop definition count indexSlot _bindings modes => do
                let body ← match definitionByName definition definitions with
                  | some body => pure body
                  | none => throw (.invalidLoopDefinition stage definition)
                let loopSite : CoreNodeRef := { stage, scope, node := ⟨nodeId⟩ }
                let indexExpression : RuntimeExpr .integer := .loopIndex { site := loopSite }
                let ⟨arena, index⟩ ← match state.expressionArena.internInteger indexExpression with
                  | some result => pure result
                  | none => throw .invalidExpressionReference
                let childScope : StaticScopeId := ⟨scope.path ++ [definition]⟩
                let initial ← seedParallelInputs stage childScope body loopSite index indexExpression
                  node.arguments modes scope { state with expressionArena := arena }
                let childPath := instancePath ++ [.parallelLane loopSite index]
                let bodyState ← analyzeStateFrom fuel stage definitions childScope childPath 0 body.nodes
                  { state with facts := state.facts ++ initial, expressionArena := arena }
                  |>.mapError fun error => match error with
                    | .invalidLoopArity errorStage errorNode =>
                        .invalidLoopArityInScope errorStage childScope errorNode
                    | error => error
                let templates ← body.outputs.mapM fun output ↦ do
                  let wire := scopedWire stage childScope output.2
                  let fact ← match lookupScopedFact wire bodyState.facts with
                    | some fact => pure fact
                    | none => throw (.missingInputFact stage ⟨nodeId⟩ output.2)
                  match fact.toTemplate with
                  | some template => pure template
                  | none => throw (.invalidLoopArity stage ⟨nodeId⟩)
                if h : templates.length = node.outputCount then
                  let joint := parallelJointFamilyId loopSite
                  let outputs := (List.range node.outputCount).map fun port =>
                    scopedOutputWire stage scope nodeId port
                  let family : JointFamilyFact := {
                    id := joint
                    count
                    indexVariable := ⟨indexSlot⟩
                    outputFamilies := outputs
                    outputArity := node.outputCount
                    elementTuple := ⟨templates.toArray, by simp [h]⟩
                  }
                  let outputFacts ← outputs.zipIdx.zip templates |>.mapM
                    fun ((wire, port), template) => do
                      let insideSequentialTemplate := instancePath.any fun
                        | .sequentialIteration .. => true
                        | _ => false
                      if template.fact.hasCarriedInput && !insideSequentialTemplate then
                        throw (.escapedCarriedInput stage ⟨nodeId⟩ port)
                      pure (scalarFact wire (.family {
                        aggregate := .joint joint port instancePath
                        count
                        elementSchema := template.schema
                      }))
                  pure {
                    bodyState with
                    facts := state.facts ++ outputFacts
                    families := bodyState.families ++ [(joint, family)]
                  }
                else throw (.invalidLoopArity stage ⟨nodeId⟩)
            | .sequentialLoop .. =>
                throw (.unsupportedSequentialRecurrence stage ⟨nodeId⟩)
            | _ =>
                let obligations ← scalarRangeObligations stage scope nodeId node state.facts
                pure { state with
                  facts := (← inferNodeFacts stage scope nodeId node state.facts)
                  staticObligations := state.staticObligations ++ obligations }
          analyzeStateFrom fuel stage definitions scope instancePath (nodeId + 1) tail next
termination_by fuel

def analyzeLeafProgramState
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (initial : AnalysisState)
    (scope : StaticScopeId := rootScope) : Except VerifyError AnalysisState := do
  let facts ← inferRulesFrom stage scope 0 program.root.nodes initial.facts
  return { initial with facts }

theorem analyzeLeafProgramState_facts
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (initial : AnalysisState)
    (scope : StaticScopeId := rootScope) :
    (analyzeLeafProgramState stage program initial scope).map (·.facts) =
      inferRulesFrom stage scope 0 program.root.nodes initial.facts := by
  unfold analyzeLeafProgramState
  generalize inferRulesFrom stage scope 0 program.root.nodes initial.facts = result
  cases result <;> rfl

def analyzeProgramState
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (initial : AnalysisState)
    (scope : StaticScopeId := rootScope) : Except VerifyError AnalysisState :=
  analyzeStateFrom (program.root.nodes.length +
      (program.definitions.map (fun definition => definition.2.nodes.length)).sum +
      program.definitions.length + 1)
    stage program.definitions scope [] 0 program.root.nodes initial

def analyzeProgram
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (initialFacts : ScopedWireFactTable := [])
    (scope : StaticScopeId := rootScope) : Except VerifyError ScopedWireFactTable := do
  match program.definitions with
  | [] => inferRules stage program.root.nodes initialFacts scope
  | _ => return (← analyzeProgramState stage program { facts := initialFacts } scope).facts

private def seedProtocolInput
    (bundle : ClosedProtocolBundle)
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (inputName protocolName : String) : Except VerifyError ScopedWireFact := do
  let wire ← match inputNodeWire stage inputName program.root.nodes with
    | some wire => pure wire
    | none => throw (.missingProgramInput stage inputName)
  let ⟨id, contract⟩ ← match inputContractByName bundle.inputContract protocolName with
    | some entry => pure entry
    | none => throw (.missingInputContract protocolName)
  return protocolInputFact wire id contract

private def producerOutputFact
    (bundle : ClosedProtocolBundle)
    (facts : ScopedWireFactTable)
    (producerStage : String)
    (outputName : String) : Except VerifyError ScopedWireFact := do
  let stage ← match bundle.workflow.stages.find? (fun stage => stage.id = producerStage) with
    | some stage => pure stage
    | none => throw (.missingArtifactOutput ⟨producerStage⟩ outputName)
  let outputRef ← match stage.program.root.outputs.find? (fun output =>
      output.1 = outputName) with
    | some output => pure output.2
    | none => throw (.missingArtifactOutput ⟨producerStage⟩ outputName)
  match lookupScopedFact (coreWire ⟨producerStage⟩ outputRef) facts with
  | some fact => pure fact
  | none => throw (.missingArtifactOutput ⟨producerStage⟩ outputName)

private def workflowStageInitialFacts
    (bundle : ClosedProtocolBundle)
    (stage : Mxx.Ir.Stage)
    (facts : ScopedWireFactTable) : Except VerifyError ScopedWireFactTable := do
  let stageId : StageId := ⟨stage.id⟩
  let mut initial : ScopedWireFactTable := []
  for (inputName, source) in stage.inputs do
    let wire ← match inputNodeWire stageId inputName stage.program.root.nodes with
      | some wire => pure wire
      | none => throw (.missingProgramInput stageId inputName)
    match source with
    | .protocol protocolName =>
        let fact ← seedProtocolInput bundle stageId stage.program inputName protocolName
        initial := initial ++ [fact]
    | .artifact producer output =>
        let source ← producerOutputFact bundle facts producer output
        initial := initial ++ [transportFact wire source]
  return initial

private def idealInitialFacts
    (bundle : ClosedProtocolBundle) : Except VerifyError ScopedWireFactTable := do
  let stage : StageId := ⟨"$ideal"⟩
  let mut initial : ScopedWireFactTable := []
  for binding in bundle.inputBindings do
    let contract ← match inputContractById bundle.inputContract binding.input with
      | some contract => pure contract
      | none => throw (.missingInputContract binding.input.name)
    for destination in binding.destinations do
      match destination with
      | .ideal inputName =>
          let wire ← match inputNodeWire stage inputName bundle.ideal.root.nodes with
            | some wire => pure wire
            | none => throw (.missingProgramInput stage inputName)
          initial := initial ++ [protocolInputFact wire binding.input contract]
      | _ => pure ()
  return initial

private def requirementInitialFacts
    (bundle : ClosedProtocolBundle)
    (index : Nat)
    (program : Mxx.Ir.Prog) : Except VerifyError ScopedWireFactTable := do
  let stage : StageId := ⟨s!"$requirement:{index}"⟩
  let mut initial : ScopedWireFactTable := []
  for binding in bundle.inputBindings do
    let contract ← match inputContractById bundle.inputContract binding.input with
      | some contract => pure contract
      | none => throw (.missingInputContract binding.input.name)
    for destination in binding.destinations do
      match destination with
      | .requirement candidate inputName =>
          if candidate = index then
            let wire ← match inputNodeWire stage inputName program.root.nodes with
              | some wire => pure wire
              | none => throw (.missingProgramInput stage inputName)
            initial := initial ++ [protocolInputFact wire binding.input contract]
      | _ => pure ()
  return initial

private def programOutputFact
    (facts : ScopedWireFactTable)
    (stage : StageId)
    (program : Mxx.Ir.Prog)
    (outputName : String) : Except VerifyError ScopedWireFact := do
  let outputRef ← match program.root.outputs.find? (fun output => output.1 = outputName) with
    | some output => pure output.2
    | none => throw (.missingArtifactOutput stage outputName)
  match lookupScopedFact (coreWire stage outputRef) facts with
  | some fact => pure fact
  | none => throw (.missingArtifactOutput stage outputName)

private def comparatorInitialFacts
    (bundle : ClosedProtocolBundle)
    (program : Mxx.Ir.Prog)
    (bindings : List ComparatorEndpointBinding)
    (facts : ScopedWireFactTable) : Except VerifyError ScopedWireFactTable := do
  let comparatorStage : StageId := ⟨"$comparator"⟩
  let mut initial : ScopedWireFactTable := []
  for binding in bindings do
    let endpoint ← match bundle.endpoints.entries.find? (fun endpoint =>
        endpoint.specification = binding.endpoint) with
      | some endpoint => pure endpoint
      | none => throw (.missingProgramInput comparatorStage binding.actualInput)
    let workflowStage ← match bundle.workflow.stages.find? (fun stage =>
        stage.id = endpoint.stage.name) with
      | some stage => pure stage
      | none => throw (.missingArtifactOutput endpoint.stage endpoint.workflowOutput)
    let actualSource ← programOutputFact facts endpoint.stage workflowStage.program
      endpoint.workflowOutput
    let actualWire ← match inputNodeWire comparatorStage binding.actualInput
        program.root.nodes with
      | some wire => pure wire
      | none => throw (.missingProgramInput comparatorStage binding.actualInput)
    initial := initial ++ [transportFact actualWire actualSource]
    match inputNodeWire comparatorStage binding.idealInput program.root.nodes with
    | none => pure ()
    | some idealWire =>
        let idealSource ← programOutputFact facts ⟨"$ideal"⟩ bundle.ideal endpoint.idealOutput
        initial := initial ++ [transportFact idealWire idealSource]
  return initial

private def matrixNoiseBound (fact : MatrixFact) : BoundExpr :=
  match fact.primary with
  | .exact _ => .constant 0
  | .affine form => form.noiseBound

private def comparatorBindings : ComparatorSpec → List ComparatorEndpointBinding
  | .equality bindings | .equalityAfterMap _ bindings => bindings

private def endpointInstanceRef
    (bundle : ClosedProtocolBundle)
    (endpoint : EndpointAnchor) : Except VerifyError ValueInstanceRef := do
  let binding ← match bundle.anchorBindings.find? (fun binding =>
      binding.anchor = endpoint.semanticAnchor) with
    | some binding => pure binding
    | none => throw (.missingAnchorBinding endpoint.semanticAnchor)
  let wire ← match binding.wires with
    | [wire] => pure wire
    | _ => throw (.invalidEndpointAnchorArity endpoint.specification)
  match wire.scope.path with
  | [] => return .concrete wire
  | definitionName :: nestedScope =>
      return .template {
        definition := { stage := wire.stage, name := definitionName }
        bodyScope := ⟨nestedScope⟩
        node := wire.node
        port := wire.port
      }

private def deriveEndpointObligations
    (bundle : ClosedProtocolBundle)
    (facts : ScopedWireFactTable) :
    Except VerifyError (List StaticObligation × List EndpointFact) := do
  let bindings := comparatorBindings bundle.comparator
  let mut endpointFacts : List EndpointFact := []
  for endpoint in bundle.endpoints.entries do
    let resolvedEndpoint ← endpointInstanceRef bundle endpoint
    let binding ← match bindings.find? (fun binding =>
        binding.endpoint = endpoint.specification) with
      | some binding => pure binding
      | none => throw (.invalidEndpointCoverage endpoint.specification)
    endpointFacts := endpointFacts ++ [{
      anchor := endpoint.semanticAnchor
      specification := endpoint.specification
      resolvedEndpoint
      stage := endpoint.stage
      workflowOutput := endpoint.workflowOutput
      idealOutput := endpoint.idealOutput
      comparatorActualInput := binding.actualInput
      comparatorIdealInput := binding.idealInput
      comparatorResultOutput := binding.resultOutput
      failureValue := binding.failureValue
    }]
  let mut obligations : List StaticObligation := []
  for endpoint in bundle.endpoints.entries do
    if endpoint.specification = .diamondBooleanInterval then
      let checked ← checkDiamondResidual bundle facts endpoint |>.mapError fun _ =>
        .invalidEndpointConnection endpoint.specification
      obligations := obligations ++ [
        .diamondFalseInterval checked.noiseBound checked.ciphertextModulus,
        .diamondTrueInterval checked.noiseBound checked.ciphertextModulus
      ]
    else if endpoint.specification = .toyThresholdDecode then
      let stageProgram ← match bundle.workflow.stages.find? (fun stage =>
          stage.id = endpoint.stage.name) with
        | some stage => pure stage.program
        | none => throw (.invalidEndpointConnection endpoint.specification)
      let anchorBinding ← match bundle.anchorBindings.find? (fun binding =>
          binding.anchor = endpoint.semanticAnchor) with
        | some binding => pure binding
        | none => throw (.missingAnchorBinding endpoint.semanticAnchor)
      let anchorWire ← match anchorBinding.wires with
        | [wire] => pure wire
        | _ => throw (.invalidEndpointAnchorArity endpoint.specification)
      let node ← match stageProgram.root.nodes[anchorWire.node.value]? with
        | some node => pure node
        | none => throw (.invalidEndpointConnection endpoint.specification)
      match node.kind, node.arguments with
      | .thresholdDecodeBool ciphertextModulus plaintextModulus _, [inputRef] =>
          let ⟨input, _⟩ ← requireMatrix facts (coreWire endpoint.stage inputRef)
          obligations := obligations ++ [.thresholdNoise (matrixNoiseBound input)
            ciphertextModulus plaintextModulus]
      | _, _ => throw (.invalidEndpointConnection endpoint.specification)
  return (obligations, endpointFacts)

private def programForStage
    (bundle : ClosedProtocolBundle)
    (stage : StageId) : Option Mxx.Ir.Prog :=
  match bundle.workflow.stages.find? (fun candidate => candidate.id = stage.name) with
  | some workflowStage => some workflowStage.program
  | none =>
      if stage.name = "$ideal" then some bundle.ideal
      else if stage.name = "$comparator" then
        match bundle.comparator with
        | .equality _ => none
        | .equalityAfterMap program _ => some program
      else none

private def programHasInput (program : Mxx.Ir.Prog) (name : String) : Bool :=
  program.root.inputNames.contains name &&
    program.root.nodes.any (fun node => match node.kind with
      | .input candidate => candidate = name
      | _ => false)

private def programInputType
    (program : Mxx.Ir.Prog)
    (name : String) : Option Mxx.Ir.WireTypeExpr :=
  match program.root.nodes.find? (fun node => match node.kind with
      | .input candidate => candidate = name
      | _ => false) with
  | some node =>
      if node.outputCount = 1 then node.outputTypes[0]? else none
  | none => none

private def programTypeTableComplete (program : Mxx.Ir.Prog) : Bool :=
  program.root.nodes.all (fun node =>
      node.outputTypes.length = node.outputCount && nodeIntrinsicOutputTypesMatch node) &&
    program.definitions.all (fun definition =>
      definition.2.nodes.all (fun node =>
        node.outputTypes.length = node.outputCount && nodeIntrinsicOutputTypesMatch node))

private def programHasOutput (program : Mxx.Ir.Prog) (name : String) : Bool :=
  program.root.outputs.any (fun output => output.1 = name)

private def nodeProducesBoolean : Mxx.Ir.NodeKind → Bool
  | .constantBool _ | .intCompare _ | .bitExtract _ | .thresholdDecodeBool _ _ _ => true
  | _ => false

private def programHasBooleanOutput (program : Mxx.Ir.Prog) (name : String) : Bool :=
  match program.root.outputs.find? (fun output => output.1 = name) with
  | none => false
  | some (_, wire) =>
      match program.root.nodes[wire.node]? with
      | some node => nodeProducesBoolean node.kind && wire.port < node.outputCount
      | none => false

private def contractIsBoolean : InputValueContract → Bool
  | .boolean => true
  | _ => false

private def requirementHasBooleanOutput
    (bundle : ClosedProtocolBundle)
    (index : Nat)
    (program : Mxx.Ir.Prog)
    (name : String) : Bool :=
  match program.root.outputs.find? (fun output => output.1 = name) with
  | none => false
  | some (_, wire) =>
      match program.root.nodes[wire.node]? with
      | some { kind := .input inputName, .. } =>
          bundle.inputBindings.any fun binding =>
            binding.destinations.contains (.requirement index inputName) &&
              match inputContractById bundle.inputContract binding.input with
              | some contract => contractIsBoolean contract
              | none => false
      | some node => nodeProducesBoolean node.kind && wire.port < node.outputCount
      | none => false

private def destinationExists
    (bundle : ClosedProtocolBundle) : ProtocolInputDestination → Bool
  | .workflowStage stage inputName =>
      match programForStage bundle stage with
      | some program => programHasInput program inputName
      | none => false
  | .requirement index inputName =>
      match bundle.requirements[index]? with
      | some program => programHasInput program inputName
      | none => false
  | .ideal inputName => programHasInput bundle.ideal inputName

private def destinationTypeMatches
    (bundle : ClosedProtocolBundle)
    (contract : InputValueContract) : ProtocolInputDestination → Bool
  | .workflowStage stage inputName =>
      match programForStage bundle stage >>= fun program => programInputType program inputName with
      | some type => wireTypeMatchesContract type contract
      | none => false
  | .requirement index inputName =>
      match bundle.requirements[index]? >>= fun program => programInputType program inputName with
      | some type => wireTypeMatchesContract type contract
      | none => false
  | .ideal inputName =>
      match programInputType bundle.ideal inputName with
      | some type => wireTypeMatchesContract type contract
      | none => false

private def verifyInputCoverage (bundle : ClosedProtocolBundle) : Except VerifyError Unit := do
  for entry in bundle.inputContract.inputs do
    let id := entry.1
    let name := entry.2.1
    let contract := entry.2.2
    if (bundle.inputContract.inputs.filter (fun candidate => candidate.1 = id)).length != 1 then
      throw (.duplicateInputId id)
    if (bundle.inputContract.inputs.filter (fun candidate => candidate.2.1 = name)).length != 1 then
      throw (.duplicateInputName name)
    let matchingBindings := bundle.inputBindings.filter (fun binding => binding.input = id)
    match matchingBindings with
    | [binding] =>
        if binding.destinations.isEmpty ||
            binding.destinations.any (fun destination => !destinationExists bundle destination) then
          throw (.invalidInputDestination id)
        for destination in binding.destinations do
          if !destinationTypeMatches bundle contract destination then
            match destination with
            | .workflowStage stage inputName =>
                throw (.inputContractTypeMismatch id stage inputName)
            | .requirement index inputName =>
                throw (.inputContractTypeMismatch id ⟨s!"$requirement:{index}"⟩ inputName)
            | .ideal inputName =>
                throw (.inputContractTypeMismatch id ⟨"$ideal"⟩ inputName)
        for destination in binding.destinations do
          let duplicates := binding.destinations.filter (fun candidate =>
            candidate = destination)
          if duplicates.length != 1 then
            throw (.duplicateInputDestination destination)
    | _ => throw (.invalidInputCoverage id)
  for binding in bundle.inputBindings do
    if bundle.inputContract.inputs.all (fun entry => entry.1 != binding.input) then
      throw (.invalidInputCoverage binding.input)
  for stage in bundle.workflow.stages do
    for (inputName, source) in stage.inputs do
      match source with
      | .artifact _ _ => pure ()
      | .protocol protocolName =>
          let ⟨id, _⟩ ← match inputContractByName bundle.inputContract protocolName with
            | some entry => pure entry
            | none => throw (.missingInputContract protocolName)
          let count := bundle.inputBindings.foldl (fun total binding =>
            total + if binding.input = id then
              (binding.destinations.filter (fun destination =>
                destination = .workflowStage ⟨stage.id⟩ inputName)).length
            else 0) 0
          if count != 1 then throw (.unboundProgramInput ⟨stage.id⟩ inputName)
  for inputName in bundle.ideal.root.inputNames do
    let count := bundle.inputBindings.foldl (fun total binding =>
      total + (binding.destinations.filter (fun destination =>
        destination = .ideal inputName)).length) 0
    if count != 1 then throw (.unboundProgramInput ⟨"$ideal"⟩ inputName)
  for program in bundle.requirements, index in [0:bundle.requirements.length] do
    for inputName in program.root.inputNames do
      let count := bundle.inputBindings.foldl (fun total binding =>
        total + (binding.destinations.filter (fun destination =>
          destination = .requirement index inputName)).length) 0
      if count != 1 then throw (.unboundProgramInput ⟨s!"$requirement:{index}"⟩ inputName)

private def verifyEndpointCoverage (bundle : ClosedProtocolBundle) : Except VerifyError Unit := do
  let bindings := comparatorBindings bundle.comparator
  for specification in bundle.endpointSpecs do
    if (bundle.endpointSpecs.filter (fun candidate => candidate = specification)).length != 1 then
      throw (.duplicateEndpointSpec specification)
    let endpoints := bundle.endpoints.entries.filter (fun endpoint =>
      endpoint.specification = specification)
    let endpointBindings := bindings.filter (fun binding => binding.endpoint = specification)
    match endpoints, endpointBindings with
    | [endpoint], [binding] =>
        let stage ← match programForStage bundle endpoint.stage with
          | some stage => pure stage
          | none => throw (.invalidEndpointConnection specification)
        if !programHasOutput stage endpoint.workflowOutput ||
            !programHasOutput bundle.ideal endpoint.idealOutput then
          throw (.invalidEndpointConnection specification)
        let outputRef ← match stage.root.outputs.find? (fun output =>
            output.1 = endpoint.workflowOutput) with
          | some output => pure output.2
          | none => throw (.invalidEndpointConnection specification)
        let anchorBinding ← match bundle.anchorBindings.find? (fun anchorBinding =>
            anchorBinding.anchor = endpoint.semanticAnchor) with
          | some anchorBinding => pure anchorBinding
          | none => throw (.invalidEndpointConnection specification)
        match anchorBinding.wires with
        | [anchorWire] =>
            if anchorWire != coreWire endpoint.stage outputRef then
              throw (.invalidEndpointConnection specification)
        | _ => throw (.invalidEndpointAnchorArity specification)
        match bundle.comparator with
        | .equality _ =>
            if binding.actualInput.isEmpty || binding.idealInput.isEmpty ||
                binding.resultOutput.isEmpty then
              throw (.invalidEndpointConnection specification)
            if binding.actualInput != endpoint.workflowOutput ||
                binding.idealInput != endpoint.idealOutput then
              throw (.invalidEndpointConnection specification)
            if binding.failureValue != true then
              throw (.invalidComparatorPolarity specification)
        | .equalityAfterMap comparator _ =>
            if !programHasInput comparator binding.actualInput ||
                (!binding.idealInput.isEmpty &&
                  programHasInput comparator binding.idealInput = false) ||
                !programHasBooleanOutput comparator binding.resultOutput then
              throw (.invalidEndpointConnection specification)
    | _, _ => throw (.invalidEndpointCoverage specification)
  for endpoint in bundle.endpoints.entries do
    if !bundle.endpointSpecs.contains endpoint.specification then
      throw (.invalidEndpointCoverage endpoint.specification)
  for binding in bindings do
    if !bundle.endpointSpecs.contains binding.endpoint then
      throw (.invalidEndpointCoverage binding.endpoint)

private def verifyPreconditions (bundle : ClosedProtocolBundle) : Except VerifyError Unit := do
  if bundle.preconditionSpec.requirementOutputs.length != bundle.requirements.length then
    throw .invalidPreconditionSpec
  for program in bundle.requirements, output in bundle.preconditionSpec.requirementOutputs,
      index in [0:bundle.requirements.length] do
    if !requirementHasBooleanOutput bundle index program output then
      throw .invalidPreconditionSpec

private def intParameterReferences : IntExpr → List (String × Bool)
  | .constant _ | .loopIndex _ => []
  | .parameter name => [(name, false)]
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intParameterReferences left ++ intParameterReferences right
  | .log2Ceil value => intParameterReferences value

private def realParameterReferences : Mxx.Ir.RealExpr → List (String × Bool)
  | .rational _ => []
  | .parameter name => [(name, true)]
  | .fromInt value => intParameterReferences value
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      realParameterReferences left ++ realParameterReferences right
  | .sqrt value => realParameterReferences value

private def matrixTypeParameterReferences (type : MatrixTypeExpr) : List (String × Bool) :=
  intParameterReferences type.modulus ++ intParameterReferences type.ringDimension ++
    intParameterReferences type.rows ++ intParameterReferences type.columns

private def wireTypeParameterReferences : Mxx.Ir.WireTypeExpr → List (String × Bool)
  | .bytes length => intParameterReferences length
  | .matrix type | .preimage type => matrixTypeParameterReferences type
  | .trapdoor type sigma base digits bound =>
      matrixTypeParameterReferences type ++ realParameterReferences sigma ++
        intParameterReferences base ++ intParameterReferences digits ++ intParameterReferences bound
  | .indexedFamily element count =>
      wireTypeParameterReferences element ++ intParameterReferences count
  | _ => []

private def optionalRangeReferences : Option (IntExpr × IntExpr) → List (String × Bool)
  | none => []
  | some (start, stop) => intParameterReferences start ++ intParameterReferences stop

private def nodeKindParameterReferences : Mxx.Ir.NodeKind → List (String × Bool)
  | .evaluateInt value => intParameterReferences value
  | .zeroMatrix type | .identityMatrix type => matrixTypeParameterReferences type
  | .constantMatrix type coefficients =>
      matrixTypeParameterReferences type ++ coefficients.flatMap intParameterReferences
  | .gadgetMatrix type base =>
      matrixTypeParameterReferences type ++ intParameterReferences base
  | .bitExtract bit | .matrixScale bit | .familyGetStatic bit => intParameterReferences bit
  | .uniformSample type minimum maximum =>
      matrixTypeParameterReferences type ++ intParameterReferences minimum ++
        intParameterReferences maximum
  | .gaussianSample type bound | .trapdoorSample type bound | .preimageSample type bound =>
      matrixTypeParameterReferences type ++ intParameterReferences bound
  | .hashSample type _ _ tags decimals words base digits =>
      matrixTypeParameterReferences type ++ tags.flatMap intParameterReferences ++
        decimals.flatMap intParameterReferences ++ words.flatMap intParameterReferences ++
        base.toList.flatMap intParameterReferences ++ digits.toList.flatMap intParameterReferences
  | .gadgetDecompose type base digits =>
      matrixTypeParameterReferences type ++ intParameterReferences base ++
        intParameterReferences digits
  | .slice rows columns => optionalRangeReferences rows ++ optionalRangeReferences columns
  | .reshape rows columns => intParameterReferences rows ++ intParameterReferences columns
  | .thresholdDecodeBool ciphertext plaintext length =>
      intParameterReferences ciphertext ++ intParameterReferences plaintext ++
        intParameterReferences length
  | .subgraphCall _ bindings => bindings.flatMap fun binding => intParameterReferences binding.2
  | .parallelLoop _ count _ bindings _ =>
      intParameterReferences count ++ bindings.flatMap fun binding =>
        intParameterReferences binding.2
  | .sequentialLoop _ count _ bindings _ =>
      intParameterReferences count ++ bindings.flatMap fun binding =>
        intParameterReferences binding.2
  | _ => []

private def programParameterReferences (program : Mxx.Ir.Prog) : List (String × Bool) :=
  let scopeReferences (scope : Mxx.Ir.Scope) := scope.nodes.flatMap fun node =>
    nodeKindParameterReferences node.kind ++ node.outputTypes.flatMap wireTypeParameterReferences
  scopeReferences program.root ++ program.definitions.flatMap fun definition =>
    scopeReferences definition.2

private def declaredBoundParameterReferences : DeclaredBoundExpr → List (String × Bool)
  | .constant _ => []
  | .parameter value | .absolute value => intParameterReferences value
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      declaredBoundParameterReferences left ++ declaredBoundParameterReferences right
  | .floorDivide value _ => declaredBoundParameterReferences value
  | .matrixProduct ringDimension innerDimension left right =>
      intParameterReferences ringDimension ++ intParameterReferences innerDimension ++
        declaredBoundParameterReferences left ++ declaredBoundParameterReferences right

private def inputContractParameterReferences : InputValueContract → List (String × Bool)
  | .matrixExact type => matrixTypeParameterReferences type
  | .matrixBounded type bound =>
      matrixTypeParameterReferences type ++ declaredBoundParameterReferences bound
  | .integerRange lower upper =>
      intParameterReferences lower ++ intParameterReferences upper
  | .bytes length => intParameterReferences length
  | .family count element =>
      intParameterReferences count ++ inputContractParameterReferences element
  | .boolean => []

private def verifyParameters (protocol : ClosedProtocolDecl) : Except VerifyError Unit := do
  for parameter in protocol.parameters do
    let declarations := protocol.parameters.filter (fun candidate =>
      candidate.name = parameter.name)
    if declarations.length != 1 then
      throw (.duplicateParameter parameter.name)
  let bundle := protocol.bundle
  let mut references := programParameterReferences bundle.ideal
  references := references ++ bundle.inputContract.inputs.flatMap fun entry =>
    inputContractParameterReferences entry.2.2
  for stage in bundle.workflow.stages do
    references := references ++ programParameterReferences stage.program
  for requirement in bundle.requirements do
    references := references ++ programParameterReferences requirement
  match bundle.comparator with
  | .equality _ => pure ()
  | .equalityAfterMap program _ =>
      references := references ++ programParameterReferences program
  for (name, requiresRational) in references do
    let declaration ← match protocol.parameters.find? (fun parameter =>
        parameter.name = name) with
      | some declaration => pure declaration
      | none => throw (.missingParameterDeclaration name)
    if requiresRational then
      if declaration.kind != .rational then throw (.parameterKindMismatch name)
    else if declaration.kind = .rational then throw (.parameterKindMismatch name)

private def verifyBundle (bundle : ClosedProtocolBundle) : Except VerifyError Unit := do
  for stage in bundle.workflow.stages do
    if !programTypeTableComplete stage.program then
      throw (.missingOrInvalidOutputTypes ⟨stage.id⟩ ⟨0⟩)
  for requirement in bundle.requirements, index in [0:bundle.requirements.length] do
    if !programTypeTableComplete requirement then
      throw (.missingOrInvalidOutputTypes ⟨s!"$requirement:{index}"⟩ ⟨0⟩)
  if !programTypeTableComplete bundle.ideal then
    throw (.missingOrInvalidOutputTypes ⟨"$ideal"⟩ ⟨0⟩)
  match bundle.comparator with
  | .equality _ => pure ()
  | .equalityAfterMap program _ =>
      if !programTypeTableComplete program then
        throw (.missingOrInvalidOutputTypes ⟨"$comparator"⟩ ⟨0⟩)
  verifyInputCoverage bundle
  verifyEndpointCoverage bundle
  verifyPreconditions bundle

private def contractObligations
    (id : ProtocolInputId) : InputValueContract → List InputObligation
  | .matrixBounded _ bound => [.matrixNorm id bound.toBoundExpr]
  | .integerRange lower upper => [.integerRange id lower upper]
  | .family _ element => contractObligations id element
  | _ => []

private def inputObligations (contract : InputContract) : List InputObligation :=
  contract.inputs.flatMap fun entry => contractObligations entry.1 entry.2.2

private def resolveAnchorWire
    (bundle : ClosedProtocolBundle)
    (wire : CoreWireRef) : Option ValueInstanceRef :=
  match programForStage bundle wire.stage with
  | none => none
  | some program =>
      match wire.scope.path with
      | [] =>
        match program.root.nodes[wire.node.value]? with
        | none => none
        | some node => if wire.port < node.outputCount then some (.concrete wire) else none
      | definitionName :: nestedScope =>
          match program.definitions.find? (fun definition => definition.1 = definitionName) with
          | none => none
          | some (_, scope) =>
              match scope.nodes[wire.node.value]? with
              | none => none
              | some node =>
                  if wire.port < node.outputCount then
                    some (.template {
                      definition := { stage := wire.stage, name := definitionName }
                      bodyScope := ⟨nestedScope⟩
                      node := wire.node
                      port := wire.port
                    })
                  else none

def resolveSemanticAnchor
    (bundle : ClosedProtocolBundle)
    (anchor : SemanticAnchorRef) : Except VerifyError (List ValueInstanceRef) :=
  match bundle.anchorBindings.find? (fun binding => binding.anchor = anchor) with
  | none => .error (.missingAnchorBinding anchor)
  | some binding =>
      binding.wires.mapM fun wire =>
        match resolveAnchorWire bundle wire with
        | some resolved => pure resolved
        | none => throw (.invalidAnchorWire anchor wire)

private def verifySemanticAnchors (bundle : ClosedProtocolBundle) : Except VerifyError Unit := do
  for endpoint in bundle.endpoints.entries do
    let resolved ← resolveSemanticAnchor bundle endpoint.semanticAnchor
    if resolved.length != 1 then
      throw (.invalidEndpointAnchorArity endpoint.specification)
  for binding in bundle.anchorBindings do
    let _ ← resolveSemanticAnchor bundle binding.anchor

/-- Analyze every closed-bundle program. Sparse overrides are Lean-authored; nonempty overrides
remain fail-closed until semantic-anchor resolution is implemented. -/
def analyzeProtocol
    (protocol : ClosedProtocolDecl)
    (overrides : SparseCertificate) : Except VerifyError AnalysisResult := do
  verifyParameters protocol
  let bundle := protocol.bundle
  verifyBundle bundle
  verifySemanticAnchors bundle
  match overrides.overrides with
  | use :: _ =>
      let _ ← resolveSemanticAnchor bundle use.output
      throw (.unsupportedOverride use.output)
  | [] => pure ()
  let mut state : AnalysisState := {
    facts := []
    protocolFamilies := protocolFamilyContracts bundle.inputContract
  }
  for stage in bundle.workflow.stages do
    let initial ← workflowStageInitialFacts bundle stage state.facts
    state ← analyzeProgramState ⟨stage.id⟩ stage.program
      { state with facts := state.facts ++ initial }
  for requirement in bundle.requirements, index in [0:bundle.requirements.length] do
    let initial ← requirementInitialFacts bundle index requirement
    state ← analyzeProgramState ⟨s!"$requirement:{index}"⟩ requirement
      { state with facts := state.facts ++ initial }
  let idealInitial ← idealInitialFacts bundle
  state ← analyzeProgramState ⟨"$ideal"⟩ bundle.ideal
    { state with facts := state.facts ++ idealInitial }
  match bundle.comparator with
  | .equality _ => pure ()
  | .equalityAfterMap program bindings =>
      let initial ← comparatorInitialFacts bundle program bindings state.facts
      state ← analyzeProgramState ⟨"$comparator"⟩ program
        { state with facts := state.facts ++ initial }
  let ⟨endpointObligations, endpointFacts⟩ ← deriveEndpointObligations bundle state.facts
  return {
    expressionArena := state.expressionArena
    symbolicFormArena := state.symbolicFormArena
    boundWitnessArena := state.boundWitnessArena
    symbolicMatrixFacts := state.symbolicMatrixFacts
    facts := state.facts
    families := state.families
    recurrences := state.recurrences
    symbolicRecurrences := state.symbolicRecurrences
    staticObligations := state.staticObligations ++ endpointObligations
    inputObligations := inputObligations bundle.inputContract
    semanticObligations := []
    endpointFacts
    usedRules := []
  }

private def matrixRuleTestType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def matrixRuleTestStage : StageId := ⟨"matrix-rule-test"⟩

private def matrixRuleTestWire (node : Nat) : CoreWireRef :=
  outputWire matrixRuleTestStage node

private def matrixRuleTestExpression (node : Nat) : MatrixExpr :=
  .wire (matrixInstance (matrixRuleTestWire node) matrixRuleTestType)

private def matrixRuleTestExact (node : Nat) : MatrixFact := {
  subject := .concrete (matrixRuleTestWire node)
  primary := .exact (matrixRuleTestExpression node)
  relations := []
  totalNormBound := .constant 8
}

private def matrixRuleTestBounded (node : Nat) : MatrixFact := {
  subject := .concrete (matrixRuleTestWire node)
  primary := .affine { terms := [], noiseBound := .constant 2 }
  relations := []
  totalNormBound := .constant 2
}

private def matrixRuleTestAffine (node : Nat) : MatrixFact := {
  subject := .concrete (matrixRuleTestWire node)
  primary := .affine {
    terms := [{
      coefficient := {
        expression := matrixRuleTestExpression node
        normBound := .constant 2
      }
      basis := matrixRuleTestExpression node
      mode := .ordinaryMatrixProduct
    }]
    noiseBound := .constant 1
  }
  relations := []
  totalNormBound := .constant 8
}

private def matrixRuleTestScoped (node : Nat) (fact : MatrixFact) : ScopedWireFact := {
  wire := matrixRuleTestWire node
  matrixType := some matrixRuleTestType
  fact := .matrix fact
}

private def matrixRuleTestMultiplyNode : Mxx.Ir.Node := {
  kind := .matrixMultiply
  arguments := [⟨0, 0⟩, ⟨1, 0⟩]
  outputCount := 1
  outputTypes := [.matrix matrixRuleTestType]
}

example : inferredMatrixMultiplyRule
    [matrixRuleTestScoped 0 (matrixRuleTestExact 0),
      matrixRuleTestScoped 1 (matrixRuleTestExact 1)]
    matrixRuleTestStage rootScope 2 matrixRuleTestMultiplyNode = .ok .multiplyAffineRight := rfl

example : inferredMatrixMultiplyRule
    [matrixRuleTestScoped 0 (matrixRuleTestBounded 0),
      matrixRuleTestScoped 1 (matrixRuleTestAffine 1)]
    matrixRuleTestStage rootScope 2 matrixRuleTestMultiplyNode = .ok .multiplyAffineLeft := rfl

example : inferredMatrixMultiplyRule
    [matrixRuleTestScoped 0 (matrixRuleTestExact 0),
      matrixRuleTestScoped 1 (matrixRuleTestAffine 1)]
    matrixRuleTestStage rootScope 2 matrixRuleTestMultiplyNode =
      .ok .multiplyAffineLeft := rfl

example : inferredMatrixMultiplyRule
    [matrixRuleTestScoped 0 (matrixRuleTestAffine 0),
      matrixRuleTestScoped 1 (matrixRuleTestAffine 1)]
    matrixRuleTestStage rootScope 2 matrixRuleTestMultiplyNode =
      .ok .multiplyAffineRight := rfl

private def matrixRuleTestRelation : MatrixRelation :=
  .gadgetDecomposition (.concrete (matrixRuleTestWire 0))
    (matrixInstance (matrixRuleTestWire 1) matrixRuleTestType) (.constant 2) (.constant 4)

example : (materializeIdentityFact (matrixRuleTestWire 2) matrixRuleTestType
    { matrixRuleTestBounded 0 with relations := [matrixRuleTestRelation] }).toOption.isSome = true :=
  rfl

example : (materializeIdentityFact (matrixRuleTestWire 2) matrixRuleTestType
    (matrixRuleTestExact 0)).toOption.isSome = true := rfl

example : (multiplyFact .multiplyAffineRight (matrixRuleTestWire 2) matrixRuleTestType
    (matrixRuleTestWire 0) matrixRuleTestType (matrixRuleTestExact 0)
    (matrixRuleTestWire 1) matrixRuleTestType (matrixRuleTestExact 1)).toOption.isSome = true := by
  decide

private def familyRuleTestBody : Mxx.Ir.Scope := {
  nodes := [{
    kind := .input "x"
    arguments := []
    outputCount := 1
    outputTypes := [.matrix matrixRuleTestType]
  }]
  outputs := [("out", ⟨0, 0⟩)]
  inputNames := ["x"]
}

private def familyRuleTestProgram : Mxx.Ir.Prog := {
  root := {
    nodes := [
      {
        kind := .input "x"
        arguments := []
        outputCount := 1
        outputTypes := [.matrix matrixRuleTestType]
      },
      {
        kind := .parallelLoop "body" (.constant 2) 0 [] [.broadcast]
        arguments := [⟨0, 0⟩]
        outputCount := 1
        outputTypes := [.indexedFamily (.matrix matrixRuleTestType) (.constant 2)]
      },
      {
        kind := .familyGetStatic (.constant 0)
        arguments := [⟨1, 0⟩]
        outputCount := 1
        outputTypes := [.matrix matrixRuleTestType]
      }
    ]
    outputs := [("selected", ⟨2, 0⟩)]
    inputNames := ["x"]
  }
  definitions := [("body", familyRuleTestBody)]
}

end Mxx.Certificate
