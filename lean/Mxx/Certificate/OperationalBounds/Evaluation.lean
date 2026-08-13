import Mxx.Certificate.OperationalBounds.IndexedEngine
import Mxx.Certificate.OperationalBounds.Progress

namespace Mxx.Certificate

open Mxx.Ir

/-- Every delayed descriptor is bound at construction to the complete owner-bearing contexts of
its direct operands.  Graph IR's numeric loop slots are accepted only when this merge identifies
one lexical binder; ambiguous slots never enter the carrier. -/
def directOperationIndexContext (inputs : List OperationalFact) : Except OperationalError IndexContext :=
  match mergeIndexContextsN (inputs.map (·.context)) with
  | some context => pure context
  | none => throw (.unsupportedOperationalExpr 0)

def genericNodeMatrixFactConcrete
    (scopeKey : ScopeTemplateKey)
    (nodeIndex : Nat)
    (node : Node)
    (rule : DerivationRule)
    (outputPort : Nat)
    (outputType : WireTypeExpr)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalMatrixFact := do
  /- `genericNodeMatrixFactConcrete` is only the fixed-assignment transfer boundary.  It may inspect a
  canonical concrete root, but never summarizes an indexed primitive or selection; general wire
  access below uses the later schema-derived `matrixFactAt` instead. -/
  let fixedMatrixFactAt (wire : WireRef) : Except OperationalError OperationalMatrixFact := do
    match ← lookupFact nodeIndex facts wire with
    | { context := { binders := #[] }, payload := .directValue root, .. } =>
        facts.arena.direct.matrixFactAt environment [] root (facts.arena.direct.values.size + 1)
    | _ => throw (.operandNotMatrix nodeIndex wire)
  let matrixType? := match outputType with
    | .matrix matrixType | .preimage matrixType => some matrixType
    | _ => none
  let embeddedMatrixType? := match node.kind with
    | .zeroMatrix matrixType
    | .identityMatrix matrixType
    | .constantMatrix matrixType _
    | .unitRowMatrix matrixType _
    | .unitColumnMatrix matrixType _
    | .gadgetMatrix matrixType _
    | .smallGadgetMatrix matrixType _
    | .powerOfBaseMatrix matrixType _ _
    | .rotationMatrix matrixType _
    | .uniformResidueSample matrixType
    | .uniformIntervalSample matrixType _ _
    | .gaussianSample matrixType _
    | .preimageSample matrixType _
    | .packPolynomialCoefficients matrixType _ => some matrixType
    | .liftIntegerToConstantPolynomial matrixType => some matrixType
    | .trapdoorSample matrixType _ =>
        if outputPort == 0 then some matrixType else none
    | .hashSample matrixType .plain _ _ _ _ _ _ => some matrixType
    | .hashSample _ .decomposed _ _ _ _ _ _
    | .hashSample _ .smallDecomposed _ _ _ _ _ _ => none
    | _ => none
  match embeddedMatrixType?, matrixType? with
  | some embedded, some output =>
      if embedded != output then throw (.outputTypeMismatch nodeIndex)
  | some _, none => throw (.outputTypeMismatch nodeIndex)
  | none, _ => pure ()
  let outputIsInteger := match outputType with
    | .integer | .constantInt => true
    | _ => false
  let outputIsBoolean := match outputType with
    | .boolean | .constantBool => true
    | _ => false
  match node.kind with
  | .constantInt _ | .evaluateInt _ | .boolToInt | .intBinary _ | .extractCoefficient _ =>
      if !outputIsInteger then throw (.outputTypeMismatch nodeIndex)
  | .constantBool _ | .intCompare _ | .bitExtract _ | .thresholdDecodeBool _ _ _ =>
      if !outputIsBoolean then throw (.outputTypeMismatch nodeIndex)
  | _ => pure ()
  match matrixType? with
  | some matrixType =>
      let _ ← evaluateIntInvariant environment loopDomains matrixType.modulus
      let _ ← evaluateIntInvariant environment loopDomains matrixType.ringDimension
      let _ ← evaluateIntInvariant environment loopDomains matrixType.rows
      let _ ← evaluateIntInvariant environment loopDomains matrixType.columns
      pure ()
  | none => pure ()
  match node.kind, matrixType? with
  | .input _, _ =>
      defaultFact nodeIndex outputPort outputType environment
  | .zeroMatrix _, some matrixType =>
      polynomialMatrixFact nodeIndex outputPort matrixType environment [] (.below 1)
  | .identityMatrix _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false
        (.below 2) { isConstantPolynomial := true }
  | .constantMatrix _ coefficients, some matrixType =>
      let values ← coefficients.mapM (evaluateIntInvariant environment loopDomains)
      let ringDimension ← match matrixType.ringDimension.evaluate environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      if ringDimension <= 0 then throw (.invalidMatrixParameters nodeIndex)
      let modulus ← match matrixType.modulus.evaluate environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let canonicalMaximum := values.foldl (fun maximum value =>
        max maximum ((if modulus > 0 then value % modulus else value).toNat)) 0
      classifiedMatrixFact nodeIndex outputPort matrixType environment
        (values.foldl (fun maximum value => max maximum (absolute value)) 0) false
        (.below (canonicalMaximum + 1)) {
          isConstantPolynomial := values.zipIdx.all fun (value, index) =>
            index % ringDimension.toNat = 0 || value = 0
        }
  | .uniformResidueSample _, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
  | .uniformIntervalSample _ minimum maximum, some matrixType =>
      let lower ← evaluateIntMinimum environment loopDomains minimum
      let upper ← evaluateIntMaximum environment loopDomains maximum
      let bound := OperationalBoundExpr.maximum
        (.contextual .maximumAbsolute environment loopDomains minimum)
        (.contextual .maximumAbsolute environment loopDomains maximum)
      classifiedMatrixFactExpr nodeIndex outputPort matrixType environment bound
        false (if lower >= 0 then .below (upper.toNat + 1) else .unknown)
  | .gaussianSample _ maximum, some matrixType =>
      let _ ← validateContextualCutoffNonnegative nodeIndex environment loopDomains maximum
      cappedMatrixFactExpr nodeIndex outputPort matrixType environment
        (.contextual .maximum environment loopDomains maximum)
  | .preimageSample _ maximum, some matrixType =>
      let _ ← validateContextualCutoffNonnegative nodeIndex environment loopDomains maximum
      let bound := OperationalBoundExpr.contextual .maximum environment loopDomains maximum
      let publicWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let trapdoorWire ← match node.arguments[1]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex publicWire)
      let targetWire ← match node.arguments[2]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex publicWire)
      let publicFact ← fixedMatrixFactAt publicWire
      let trapdoor ← trapdoorFactAt nodeIndex facts trapdoorWire
      let target ← fixedMatrixFactAt targetWire
      let publicIdentity ← match publicFact.identity with
        | some identity => pure identity
        | none => throw (.missingPublicIdentity nodeIndex publicWire)
      if publicIdentity != trapdoor.publicIdentity then
        throw (.publicIdentityMismatch nodeIndex)
      let trapdoorCutoff ← trapdoor.preimageCutoff.mapM (requireMaterializedScalarBound nodeIndex)
      let _ ← validatePreimageCutoffAgreement nodeIndex environment loopDomains maximum
        trapdoor.publicIdentity trapdoorCutoff
      let result ← cappedMatrixFactExpr nodeIndex outputPort matrixType environment bound
      let relation : PreimageRelation := {
        producer := result.origin
        publicIdentity
        targetOrigin := target.origin
        targetSummary := matrixTargetSummary target
      }
      pure ({ result with relations := [.preimage relation] }).refreshPrimitivePolynomial
  | .hashSample _ variant tagPrefix tagExpressions tagDecimalExpressions tagU64LeExpressions
      base digitCount, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let keyWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let keyOrigin ← valueOriginAt scopeKey nodeIndex facts keyWire
      let trailingIntegerOrigins ← (node.arguments.drop 1).mapM
        (valueOriginAt scopeKey nodeIndex facts)
      let hashInputs ← node.arguments.mapM (lookupFact nodeIndex facts)
      let context ← directOperationIndexContext hashInputs
      let parameterEnvironment ← match IndexedParamEnvironment.fromIrAtWithDomains context loopDomains environment with
        | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
      let parameterDomains ← match IndexedOperationalParameterDomain.fromIrAt context loopDomains with
        | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
      let tagExpressions ← match tagExpressions.mapM (IndexedParameterExpr.fromIrAt context) with
        | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
      let tagDecimalExpressions ← match tagDecimalExpressions.mapM (IndexedParameterExpr.fromIrAt context) with
        | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
      let tagU64LeExpressions ← match tagU64LeExpressions.mapM (IndexedParameterExpr.fromIrAt context) with
        | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
      let hashIdentity (targetType : MatrixTypeExpr) : Except OperationalError DeterministicHashIdentity := do
        let matrixType ← match IndexedMatrixTypeExpr.fromIrAt context targetType with
          | some value => pure value | none => throw (.unsupportedOperationalExpr nodeIndex)
        pure {
          keyOrigin
          matrixType
          parameterEnvironment
          parameterDomains
          tagPrefix
          tagExpressions
          tagDecimalExpressions
          tagU64LeExpressions
          trailingIntegerOrigins
        }
      match variant with
      | .plain =>
          let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
          pure { result with origin := .deterministicHash (← hashIdentity matrixType) }
      | .decomposed | .smallDecomposed =>
          let base ← match base with
            | some expression => evaluateIntInvariant environment loopDomains expression
            | none => throw (.gadgetLayoutMismatch nodeIndex)
          let digitCount ← match digitCount with
            | some expression => evaluateIntInvariant environment loopDomains expression
            | none => throw (.gadgetLayoutMismatch nodeIndex)
          if base <= 1 || digitCount <= 0 then throw (.gadgetLayoutMismatch nodeIndex)
          let outputParams ← match matrixType.evaluate environment (.constant 0) with
            | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
          if outputParams.rows % digitCount.toNat != 0 then
            throw (.gadgetLayoutMismatch nodeIndex)
          let descriptor ← resolveGadgetLayout nodeIndex layouts outputParams
          let small := variant == Mxx.HashVariant.smallDecomposed
          let expectedCount := if small then descriptor.smallDigitCount else
            descriptor.regularDigitCount
          if descriptor.base != base || expectedCount != digitCount.toNat then
            throw (.gadgetLayoutMismatch nodeIndex)
          let targetRows := outputParams.rows / digitCount.toNat
          let targetType : MatrixTypeExpr := {
            modulus := .constant outputParams.modulus
            ringDimension := .constant (Int.ofNat outputParams.ringDimension)
            rows := .constant (Int.ofNat targetRows)
            columns := .constant (Int.ofNat outputParams.columns)
          }
          let targetParams : Mxx.SamplerParams := {
            maxCoefficientBound := cap.natAbs
            modulus := outputParams.modulus
            ringDimension := outputParams.ringDimension
            rows := targetRows
            columns := outputParams.columns
          }
          let targetOrigin := MatrixOriginIdentity.deterministicHash (← hashIdentity targetType)
          let targetSummary : RelationTargetSummary := {
            origin := targetOrigin
            matrixType := targetType
            matrixParams := targetParams
            totalHardBound := .closedInt (.constant cap)
            canonicalRange := .unknown
            polynomial := relationSnapshotPolynomial (primitiveOperationalPolynomial targetOrigin
              targetType targetParams.rows (.closedInt (.constant cap)) .large none [] {})
          }
          let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId
            outputParams targetRows base small digitCount.toNat
          let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment
            (Int.ofNat (Mxx.gadgetDecompositionBound base small)) false
            (if small then .below base.natAbs else .unknown)
          let relation : DecompositionRelation := {
            producer := result.origin
            publicIdentity
            inputOrigin := targetOrigin
            inputSummary := targetSummary
            base
            small
            digitCount := digitCount.toNat
            status := if small then .smallRangeMissing descriptor.smallestCrtModulus else .available
          }
          pure ({ result with relations := [.decomposition relation] }).refreshPrimitivePolynomial
  | .gadgetDecompose declaredType base small digitCount, some matrixType =>
      let bound ← evaluateIntInvariant environment loopDomains base
      let count ← evaluateIntInvariant environment loopDomains digitCount
      if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch nodeIndex)
      let params ← match declaredType.evaluate environment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let expectedCount := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if count.toNat != expectedCount || bound != descriptor.base then
        throw (.gadgetLayoutMismatch nodeIndex)
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← fixedMatrixFactAt inputWire
      let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId params
        input.matrixParams.rows bound small count.toNat
      let result ← cappedMatrixFact nodeIndex outputPort matrixType environment
        (Int.ofNat (Mxx.gadgetDecompositionBound bound small))
      let status := if !small then ReconstructionStatus.available else
        match input.canonicalRange with
        | .below upper => if upper <= descriptor.smallestCrtModulus then
            .available else .smallRangeMissing descriptor.smallestCrtModulus
        | .unknown => .smallRangeMissing descriptor.smallestCrtModulus
      let relation : DecompositionRelation := {
        producer := result.origin
        publicIdentity
        inputOrigin := input.origin
        inputSummary := matrixTargetSummary input
        base := bound
        small
        digitCount := count.toNat
        status
      }
      pure ({ result with
        canonicalRange := if small then .below bound.natAbs else .unknown
        relations := [.decomposition relation]
      }).refreshPrimitivePolynomial
  | .matrixAdd, some matrixType | .matrixSubtract, some matrixType =>
      if node.arguments.length != 2 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let leftFact ← fixedMatrixFactAt leftWire
      let rightFact ← fixedMatrixFactAt rightWire
      let combinePair
          (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
        let left ← retypeMatrixFact nodeIndex matrixType left environment
        let right ← retypeMatrixFact nodeIndex matrixType right environment
        let polynomial := match node.kind with
          | .matrixAdd => addOperationalPolynomials left.polynomial right.polynomial
          | .matrixSubtract => subtractOperationalPolynomials left.polynomial right.polynomial
          | _ => []
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      combinePair leftFact rightFact
  | .concat axis, some matrixType =>
      let inputs ← node.arguments.mapM fixedMatrixFactAt
      let polynomial ← concatOperationalPolynomials axis matrixType (inputs.map (·.polynomial))
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
        (joinCanonicalRanges (inputs.map (·.canonicalRange)))
  | .select, some _ => throw (.unsupportedNode nodeIndex)
  | .transpose, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← fixedMatrixFactAt inputWire
      let polynomial ← transposeOperationalPolynomial input.polynomial
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .slice rows columns, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← fixedMatrixFactAt inputWire
      let polynomial ← sliceOperationalPolynomial rows columns matrixType input.polynomial
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .liftIntegerToConstantPolynomial _, some matrixType =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let input ← integerFactAt nodeIndex facts inputWire
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters nodeIndex)
      if params.rows != 1 || params.columns != 1 || params.modulus <= 0 ||
          params.ringDimension == 0 then
        throw (.invalidMatrixParameters nodeIndex)
      let lower ← requireMaterializedScalarBound nodeIndex input.lowerExpression
      let upper ← requireMaterializedScalarBound nodeIndex input.upperExpression
      let bound := OperationalBoundExpr.maximum (.negate lower) upper
      classifiedMatrixFactExpr nodeIndex outputPort matrixType environment bound
        false (.below params.modulus.toNat) { isConstantPolynomial := true }
  | .matrixNegate, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← fixedMatrixFactAt inputWire
      let input ← retypeMatrixFact nodeIndex matrixType input environment
      polynomialMatrixFact nodeIndex outputPort matrixType environment
        (scaleOperationalPolynomial (-1) input.polynomial) (negateCanonicalRange input.canonicalRange)
  | .matrixScale scalar, some matrixType =>
      let scalarValues ← evaluateIntOverLoops environment loopDomains scalar
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← fixedMatrixFactAt inputWire
      let input ← retypeMatrixFact nodeIndex matrixType input environment
      match scalarValues with
      | [] => throw (.invalidMatrixParameters nodeIndex)
      | first :: tail =>
          if first == 1 && tail.all (· == 1) then
            pure { input with subject := { node := nodeIndex, port := outputPort } }
          else
            let polynomial ←
              if tail.all (· == first) then
                pure (scaleOperationalPolynomial first input.polynomial)
              else
                multiplyOperationalPolynomials
                  (parameterScalarPolynomial environment loopDomains scalar matrixType)
                  input.polynomial |>.mapError (flatErrorAt nodeIndex)
            polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
              (scaleCanonicalRange scalarValues input.canonicalRange)
  | .matrixMultiply, some matrixType =>
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let leftFact ← fixedMatrixFactAt leftWire
      let rightFact ← fixedMatrixFactAt rightWire
      multiplyConcreteMatrixFacts nodeIndex outputPort matrixType rule rightWire environment
        leftFact rightFact
  | .tensor, some matrixType =>
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let left ← fixedMatrixFactAt leftWire
      let right ← fixedMatrixFactAt rightWire
      let polynomial ← tensorOperationalPolynomials matrixType
        left.polynomial right.polynomial |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
  | .crtRecompose plaintextModuli reconstructionCoefficients, some matrixType =>
      if node.arguments.isEmpty || node.arguments.length != plaintextModuli.length ||
          node.arguments.length != reconstructionCoefficients.length then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let moduli ← plaintextModuli.mapM (evaluateIntInvariant environment loopDomains)
      let coefficients ← reconstructionCoefficients.mapM
        (evaluateIntInvariant environment loopDomains)
      let inputs ← node.arguments.mapM fixedMatrixFactAt
      let modulus ← evaluateIntInvariant environment loopDomains matrixType.modulus
      if modulus <= 0 || moduli.any (fun value => value <= 1 || value >= modulus) ||
          coefficients.any (fun value => value < 0 || value >= modulus) then
        throw (.invalidMatrixParameters nodeIndex)
      let inputs ← inputs.mapM fun input => retypeMatrixFact nodeIndex matrixType input environment
      if inputs.any (·.matrixParams.rows != 1) then
        throw (.invalidMatrixParameters nodeIndex)
      let polynomial := (coefficients.zip inputs).foldl
        (fun result pair ↦ addOperationalPolynomials result
          (scaleOperationalPolynomial pair.1 pair.2.polynomial)) []
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
  | .trapdoorSample _ maximum, some matrixType =>
      let _ ← validateContextualCutoffNonnegative nodeIndex environment loopDomains maximum
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      pure ({ result with identity := some (.sampledTrapdoor scopeKey
        { node := nodeIndex, port := 0 }) }).refreshPrimitivePolynomial
  | .trapdoorPublic, some matrixType =>
      let trapdoorWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let trapdoor ← trapdoorFactAt nodeIndex facts trapdoorWire
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let bound := publicIdentityMaximum cap trapdoor.publicIdentity
      let large := publicIdentityIsLarge trapdoor.publicIdentity
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment bound large
      pure ({ result with identity := some trapdoor.publicIdentity }).refreshPrimitivePolynomial
  | .gadgetTrapdoor _ base, some matrixType =>
      let bound ← evaluateIntInvariant environment loopDomains base
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let count := descriptor.regularDigitCount
      if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
      let identity := PublicMatrixIdentity.gadget descriptor.paramsId params
        params.rows bound false count
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      pure ({ result with identity := some identity }).refreshPrimitivePolynomial
  | .unitRowMatrix _ _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false
        (.below 2) { isConstantPolynomial := true }
  | .unitColumnMatrix _ position, some matrixType =>
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters nodeIndex)
      let positions ← evaluateIntOverLoops environment loopDomains position
      if params.rows = 0 || params.columns != 1 || positions.isEmpty ||
          positions.any fun index => index < 0 || index >= params.rows then
        throw (.invalidMatrixParameters nodeIndex)
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false
        (.below 2) {
          isConstantPolynomial := true
          knownZeroRows := some (.constant (params.rows - 1))
        }
  | .rotationMatrix _ _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false
  | .gadgetMatrix _ base, some matrixType | .smallGadgetMatrix _ base, some matrixType =>
      let bound ← evaluateIntMaximumAbsolute environment loopDomains base
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let small := match node.kind with | .smallGadgetMatrix _ _ => true | _ => false
      let count := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      pure ({ result with
        identity := some (.gadget descriptor.paramsId params params.rows bound small count)
      }).refreshPrimitivePolynomial
  | .powerOfBaseMatrix _ base _, some matrixType =>
      let _ ← evaluateIntMaximumAbsolute environment loopDomains base
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
  | _, some _ => throw (.unsupportedNode nodeIndex)
  | _, none => throw (.unsupportedNode nodeIndex)

/-- Selection-aware scalar endpoint.  Ordinary concrete operations retain the existing transfer;
indexed operands are lifted pointwise and return the extended request-local arena. -/
def genericNodeFact
    (scopeKey : ScopeTemplateKey)
    (nodeIndex : Nat)
    (node : Node)
    (rule : DerivationRule)
    (outputPort : Nat)
    (outputType : WireTypeExpr)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let arguments ← node.arguments.mapM (lookupFact nodeIndex facts)
  if let .input _ := node.kind then
    if let .indexedFamily element countExpression := outputType then
      let count ← match countExpression.evaluate environment with
        | some value => pure value
        | none => throw .nonClosedExpression
      if count <= 0 then throw (.invalidCount nodeIndex count)
      let subject : WireRef := { node := nodeIndex, port := outputPort }
      let binder : FamilyTemplateBinder := {
        owner := scopeKey, producerNode := nodeIndex, binderSlot := outputPort
      }
      let selection := DynamicSelectionIdentity.fromOrigin (.local scopeKey subject) count.toNat
      match element with
      | .matrix matrixType | .preimage matrixType =>
          let base ← unconstrainedMatrixFact nodeIndex outputPort matrixType environment
          let representativeFact :=
            indexMatrixFact binder selection subject base
          let context ← selectionIndexedContext selection nodeIndex
          let (fixed, reference) := facts.arena.direct.fixed.pushMatrix representativeFact
          let direct := { facts.arena.direct with fixed }
          let (direct, root) ← match direct.pushShared context (.matrix matrixType) reference with
            | some result => pure result
            | none => throw (.unsupportedOperationalExpr direct.values.size)
          let value ← match direct.valueAt? root with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef root)
          return ({ facts.arena with direct }, {
            context := value.context, payload := .directValue root, storage := value.storage })
      | _ =>
          let scalar ← defaultScalarFact nodeIndex outputPort element environment loopDomains
          let scalar := indexScalarFact binder selection subject scalar
          let context ← selectionIndexedContext selection nodeIndex
          let (fixed, reference) := facts.arena.direct.fixed.pushScalar scalar
          let direct := { facts.arena.direct with fixed }
          let (direct, root) ← match direct.pushShared context
              (.scalar (operationalScalarSchema scalar)) reference with
            | some result => pure result
            | none => throw (.unsupportedOperationalExpr direct.values.size)
          let value ← match direct.valueAt? root with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef root)
          return ({ facts.arena with direct }, {
            context := value.context, payload := .directValue root, storage := value.storage })
  let scalarOutput := match outputType with
    | .matrix _ | .preimage _ | .indexedFamily _ _ => false
    | _ => true
  match node.kind with
  | .constantInt _ | .evaluateInt _ | .boolToInt | .intBinary _ | .extractCoefficient _ =>
      if outputType != .integer && outputType != .constantInt then
        throw (.outputTypeMismatch nodeIndex)
  | .constantBool _ | .intCompare _ | .bitExtract _ | .thresholdDecodeBool _ _ _ =>
      if outputType != .boolean && outputType != .constantBool then
        throw (.outputTypeMismatch nodeIndex)
  | _ => pure ()
  let indexedPackOutput := match node.kind with
    | .packPolynomialCoefficients _ _ => true
    | _ => false
  /- Scalar primitives remain in the direct carrier, including at parallel-loop boundaries. -/
  if scalarOutput then
    let directOperation (kind : OperationalScalarPrimitiveKind) : DirectScalarOperation := {
      kind, ownerScope := some scopeKey, ownerNode := nodeIndex, outputPort }
    match node.kind with
    | .boolToInt =>
        let argument ← match arguments[0]? with
          | some argument => pure argument
          | none => throw (.unsupportedOutputArity nodeIndex arguments.length)
        let schema ← match argument.payload with
          | .directValue root => match facts.arena.direct.valueAt? root with
            | some value => pure value.payload.schema
            | none => throw (.invalidOperationalExprRef root)
        match schema with
        | .scalar .boolean => return (← facts.arena.pushDirectScalarPointwiseN
            (directOperation .boolToInt) arguments.toArray)
        | _ => throw (.operandNotBoolean nodeIndex
            (node.arguments.headD { node := nodeIndex, port := outputPort }))
    | .intBinary kind => return (← facts.arena.pushDirectScalarPointwiseN
        (directOperation (.intBinary kind)) arguments.toArray)
    | .intCompare kind => return (← facts.arena.pushDirectScalarPointwiseN
        (directOperation (.intCompare kind)) arguments.toArray)
    | .bitExtract position =>
        let position ← evaluateIntInvariant environment loopDomains position
        if position < 0 then throw (.invalidCount nodeIndex position)
        return (← facts.arena.pushDirectScalarPointwiseN
          (directOperation (.bitExtract position)) arguments.toArray)
    | .intToReal => return (← facts.arena.pushDirectScalarPointwiseN
        (directOperation .intToReal) arguments.toArray)
    | .realBinary kind => return (← facts.arena.pushDirectScalarPointwiseN
        (directOperation (.realBinary kind)) arguments.toArray)
    | .realSqrt => return (← facts.arena.pushDirectScalarPointwiseN
        (directOperation .realSqrt) arguments.toArray)
    | _ => pure ()
  /- Matrix outputs with scalar operands remain in the direct carrier. -/
  if !scalarOutput then
    match node.kind, arguments with
    | .trapdoorPublic, [input] =>
        let matrixType ← match outputType with
          | .matrix matrixType => pure matrixType
          | _ => throw (.outputTypeMismatch nodeIndex)
        let operation : DirectValueMatrixOperation := {
          kind := .trapdoorPublic matrixType
          ownerScope := some scopeKey
          ownerNode := nodeIndex
          outputPort
          parameterEnvironment := environment
        }
        return ← facts.arena.pushDirectIntegerLiftPointwise operation input
    | .liftIntegerToConstantPolynomial matrixType, [input] =>
        let outputType ← match outputType with
          | .matrix outputType => pure outputType
          | _ => throw (.outputTypeMismatch nodeIndex)
        if !operationalMatrixTypeEqual matrixType outputType then
          throw (.outputTypeMismatch nodeIndex)
        let operation : DirectValueMatrixOperation := {
          kind := .liftIntegerToConstantPolynomial matrixType
          ownerScope := some scopeKey
          ownerNode := nodeIndex
          outputPort
          parameterEnvironment := environment
        }
        return ← facts.arena.pushDirectIntegerLiftPointwise operation input
    | _, _ => pure ()
  if indexedPackOutput then
    match node.kind, arguments with
    | .packPolynomialCoefficients matrixType coefficientBits, [input] => do
        let root := input.payload.root
        let value ← match facts.arena.direct.valueAt? root with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef root)
        let bits ← evaluateIntMaximum environment loopDomains coefficientBits
        let params ← match matrixType.evaluate environment with
          | some params => pure params
          | none => throw (.invalidMatrixParameters nodeIndex)
        let expectedCount := Int.ofNat params.ringDimension * bits
        if bits <= 0 || params.rows != 1 || params.columns != 1 ||
            (2 : Int) ^ bits.toNat < params.modulus then
          throw (.loopInputModeMismatch nodeIndex 0)
        let (coefficientBinder, coefficients) ← match value with
        | { payload := .explicit (.scalar .boolean) binder coefficients, .. } =>
            pure (binder, coefficients.map (fun reference => some reference))
        | { payload := .explicitValues (.scalar .boolean) binder coefficients, .. } =>
            pure (binder, coefficients.map (fun value =>
              match facts.arena.direct.valueAt? value with
              | some { payload := .shared (.scalar .boolean) reference, .. } => some reference
              | _ => none))
        | _ => throw (.loopInputModeMismatch nodeIndex 0)
        if coefficientBinder.count.evaluate environment != some expectedCount then
          throw (.loopInputModeMismatch nodeIndex 0)
        if !value.context.binders.contains coefficientBinder then
          throw (.loopInputModeMismatch nodeIndex 0)
        if coefficients.size != expectedCount.toNat then
          throw (.loopInputModeMismatch nodeIndex 0)
        for coefficient in coefficients do
          match coefficient with
          | some (.scalar reference) =>
              if facts.arena.direct.fixed.scalars[reference]? != some .boolean then
                throw (.operandNotBoolean nodeIndex
                  (node.arguments.headD { node := nodeIndex, port := 0 }))
          | some (.matrix _) | none =>
              throw (.operandNotBoolean nodeIndex
                (node.arguments.headD { node := nodeIndex, port := 0 }))
        let residualContext : IndexContext := {
          binders := value.context.binders.filter (fun binder => binder != coefficientBinder) }
        let cap ← match matrixCap matrixType environment with
          | some value => pure value
          | none => throw (.invalidMatrixParameters nodeIndex)
        let packed ← residualContext.binders.foldlM (fun indexed binder =>
          let familyBinder : FamilyTemplateBinder := {
            owner := scopeKey
            producerNode := nodeIndex
            binderSlot := binder.slot
          }
          let selection : DynamicSelectionIdentity := {
            index := .local scopeKey { node := nodeIndex, port := outputPort }
            expression := .variable binder
          }
          pure (indexMatrixFact familyBinder selection { node := nodeIndex, port := outputPort }
            indexed))
          (← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
            (.below params.modulus.toNat))
        let (fixed, reference) := facts.arena.direct.fixed.pushMatrix packed
        let direct := { facts.arena.direct with fixed }
        let (direct, packedRoot) ← match direct.pushShared residualContext (.matrix matrixType)
            reference with
          | some result => pure result
          | none => throw (.unsupportedOperationalExpr root)
        let packedValue ← match direct.valueAt? packedRoot with
          | some packedValue => pure packedValue
          | none => throw (.invalidOperationalExprRef packedRoot)
        return ({ facts.arena with direct }, {
          context := packedValue.context
          payload := .directValue packedRoot
          storage := packedValue.storage
        })
    | _, _ => throw (.loopInputModeMismatch nodeIndex 0)
  if !scalarOutput && !indexedPackOutput then
    let output ← genericNodeMatrixFactConcrete scopeKey nodeIndex node rule outputPort outputType facts
      environment loopDomains layouts
    facts.arena.promoteConcreteMatrixFact output
  else
    let subject : WireRef := { node := nodeIndex, port := outputPort }
    match node.kind with
    | .input _ =>
        let scalar ← defaultScalarFact nodeIndex outputPort outputType environment loopDomains
        facts.arena.promoteConcreteScalarFact scalar
    | .constantInt value => do
        if !node.arguments.isEmpty then
          throw (.unsupportedOutputArity nodeIndex node.arguments.length)
        let scalar ← integerFact nodeIndex outputPort value value
        facts.arena.promoteConcreteScalarFact scalar
    | .evaluateInt value => do
        if !node.arguments.isEmpty then
          throw (.unsupportedOutputArity nodeIndex node.arguments.length)
        let scalar ← integerFactWithExpressions nodeIndex outputPort
          (← evaluateIntMinimum environment loopDomains value)
          (← evaluateIntMaximum environment loopDomains value)
          (.contextual .minimum environment loopDomains value)
          (.contextual .maximum environment loopDomains value)
        facts.arena.promoteConcreteScalarFact scalar
    | .constantBool _ =>
        if node.arguments.isEmpty then
          facts.arena.promoteConcreteScalarFact .boolean
        else throw (.unsupportedOutputArity nodeIndex node.arguments.length)
    | .constantReal _ =>
        if node.arguments.isEmpty then
          facts.arena.promoteConcreteScalarFact .real
        else throw (.unsupportedOutputArity nodeIndex node.arguments.length)
    | .trapdoorSample _ maximum => do
        let (matrixType, sigma, gadgetBase, digitCount, cutoff) ← match outputType with
          | .trapdoor matrixType sigma gadgetBase digitCount cutoff =>
              pure (matrixType, sigma, gadgetBase, digitCount, cutoff)
          | _ => throw (.outputTypeMismatch nodeIndex)
        let boundExpr := OperationalBoundExpr.contextual .maximum environment loopDomains maximum
        let bound ← boundExpr.evaluate environment #[]
        let cap ← match matrixCap matrixType environment with
          | some value => pure value
          | none => throw (.invalidMatrixParameters nodeIndex)
        let maximum := min cap bound
        let params ← match matrixType.evaluate environment (.constant maximum) with
          | some params => pure params
          | none => throw (.invalidMatrixParameters nodeIndex)
        let preimageCutoff := some (← validateContextualCutoffNonnegative nodeIndex environment
          loopDomains cutoff)
        let scalar : OperationalScalarFact := .trapdoor {
          subject, matrixType, matrixParams := params
          sigma, gadgetBase, digitCount, preimageMaxCoefficientBound := cutoff
          maximum := .closed (.minimum (.closedInt (.constant cap)) boundExpr)
          preimageCutoff := preimageCutoff.map .closed
          publicIdentity := .sampledTrapdoor scopeKey { node := nodeIndex, port := 0 }
        }
        facts.arena.promoteConcreteScalarFact scalar
    | .gadgetTrapdoor _ base => do
        let (matrixType, sigma, gadgetBase, digitCount, cutoff) ← match outputType with
          | .trapdoor matrixType sigma gadgetBase digitCount cutoff =>
              pure (matrixType, sigma, gadgetBase, digitCount, cutoff)
          | _ => throw (.outputTypeMismatch nodeIndex)
        let bound ← evaluateIntInvariant environment loopDomains base
        let params ← match matrixType.evaluate environment (.constant 0) with
          | some params => pure params
          | none => throw (.invalidMatrixParameters nodeIndex)
        let descriptor ← resolveGadgetLayout nodeIndex layouts params
        if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
        let scalar : OperationalScalarFact := .trapdoor {
          subject, matrixType, matrixParams := params
          sigma, gadgetBase, digitCount, preimageMaxCoefficientBound := cutoff
          maximum := .closed (.closedInt (.constant (absolute bound)))
          preimageCutoff := none
          publicIdentity := .gadget descriptor.paramsId params params.rows bound false
            descriptor.regularDigitCount
        }
        facts.arena.promoteConcreteScalarFact scalar
    | .packPolynomialCoefficients _ _ =>
        throw (.loopInputModeMismatch nodeIndex 0)
    | _ => throw (.unsupportedNode nodeIndex)

def lookupCheckedDefinition
    (name : String)
    (definitions : List (String × Scope))
    (derivations : List (String × ScopeDerivation)) :
    Except OperationalError (Scope × ScopeDerivation) :=
  match definitions, derivations with
  | [], _ => .error (.missingDefinition name)
  | _, [] => .error (.missingDefinition name)
  | (definitionName, scope) :: definitionTail,
      (derivationName, derivation) :: derivationTail =>
      if definitionName != derivationName then .error (.missingDefinition name)
      else if definitionName = name then .ok (scope, derivation)
      else lookupCheckedDefinition name definitionTail derivationTail

def validateScopeInputs (scope : Scope) : Except OperationalError Unit := do
  let nodeNames := scope.nodes.filterMap fun node => match node.kind with
    | .input name => some name
    | _ => none
  for name in scope.inputNames do
    if scope.inputNames.count name != 1 then throw (.duplicateInputName name)
    if nodeNames.count name = 0 then throw (.missingInputNode name)
    if nodeNames.count name != 1 then throw (.duplicateInputName name)
  for name in nodeNames do
    if !scope.inputNames.contains name then throw (.unexpectedInputNode name)

def findDefinitionIndex
    (name : String) : List (String × Scope) → Nat → Option Nat
  | [], _ => none
  | (candidate, _) :: tail, index =>
      if candidate == name then some index else findDefinitionIndex name tail (index + 1)

def prepareOperationalScope
    (definitions : List (String × Scope))
    (scope : Scope)
    (derivation : ScopeDerivation) : Except OperationalError PreparedOperationalScope := do
  validateScopeInputs scope
  let inputIndices := scope.nodes.map fun node => match node.kind with
    | .input name => scope.inputNames.idxOf? name
    | _ => none
  let definitionIndices := scope.nodes.map fun node => match node.kind with
    | .subgraphCall name _ => findDefinitionIndex name definitions 0
    | .parallelLoop name _ _ _ _ => findDefinitionIndex name definitions 0
    | .sequentialLoop name _ _ _ _ => findDefinitionIndex name definitions 0
    | _ => none
  for (node, index) in scope.nodes.zipIdx do
    match node.kind with
    | .subgraphCall name _ | .parallelLoop name _ _ _ _ | .sequentialLoop name _ _ _ _ =>
        match definitionIndices[index]? with
        | some (some _) => pure ()
        | _ => throw (OperationalError.missingDefinition name)
    | _ => pure ()
  let mut attachmentBuckets := Array.replicate scope.nodes.size #[]
  for attachment in derivation.attachments do
    validateDerivationAttachment scope attachment
    let readyNode := attachment.roles.foldl (fun current role => max current role.2.node) 0
    match attachmentBuckets[readyNode]? with
    | some bucket => attachmentBuckets := attachmentBuckets.set! readyNode (bucket.push attachment)
    | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  pure { scope, derivation, inputIndices, definitionIndices, attachmentBuckets }

/-- Checks a frozen program once and resolves every structure-only lookup used by later requests. -/
def prepareProgramOperational
    (program : Prog)
    (derivation : ProgramDerivation) : Except OperationalError PreparedOperationalProgram := do
  match checkProgramDerivation program derivation with
  | .error error => throw (.derivation error)
  | .ok () => pure ()
  let root ← prepareOperationalScope program.definitions program.root derivation.root
  let definitionPairs := program.definitions.zip derivation.definitions
  let definitions ← definitionPairs.mapM fun pair => do
    let ((name, scope), (derivationName, scopeDerivation)) := pair
    if name != derivationName then throw (.missingDefinition name)
    return (name, ← prepareOperationalScope program.definitions scope scopeDerivation)
  pure { root, definitions := definitions.toArray }

def preparedDefinitionAt
    (node : Nat)
    (prepared : PreparedOperationalScope)
    (definitions : Array (String × PreparedOperationalScope)) :
    Except OperationalError PreparedOperationalScope := do
  let definitionIndex ← match prepared.definitionIndices[node]? with
    | some (some index) => pure index
    | _ => throw (OperationalError.missingDefinition s!"node-{node}")
  match definitions[definitionIndex]? with
  | some (_, definition) => pure definition
  | none => throw (OperationalError.missingDefinition s!"node-{node}")

/-- Whether an executable real expression reads one exact lexical Graph-IR loop slot. -/
private def realExprUsesLoopSlot (slot : Nat) : RealExpr → Bool
  | .rational _ | .parameter _ => false
  | .fromInt value => intExprUsesLoop slot value
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      realExprUsesLoopSlot slot left || realExprUsesLoopSlot slot right
  | .sqrt value => realExprUsesLoopSlot slot value

private def matrixTypeUsesLoopSlot (slot : Nat) (matrixType : MatrixTypeExpr) : Bool :=
  intExprUsesLoop slot matrixType.modulus || intExprUsesLoop slot matrixType.ringDimension ||
    intExprUsesLoop slot matrixType.rows || intExprUsesLoop slot matrixType.columns

private def wireTypeUsesLoopSlot (slot : Nat) : WireTypeExpr → Bool
  | .constantInt | .constantReal | .constantBool | .integer | .real | .boolean => false
  | .bytes length => intExprUsesLoop slot length
  | .typedBlob _ _ => false
  | .matrix matrixType | .preimage matrixType => matrixTypeUsesLoopSlot slot matrixType
  | .trapdoor matrixType sigma base digitCount cutoff =>
      matrixTypeUsesLoopSlot slot matrixType || realExprUsesLoopSlot slot sigma ||
        intExprUsesLoop slot base || intExprUsesLoop slot digitCount || intExprUsesLoop slot cutoff
  | .indexedFamily element count => wireTypeUsesLoopSlot slot element || intExprUsesLoop slot count

/-- Scan every Graph-IR expression field evaluated in the current lexical scope.  The caller
scans nested definitions separately, because a nested loop may shadow the numeric slot. -/
private def nodeKindUsesLoopSlot (slot : Nat) : NodeKind → Bool
  | .input _ | .constantInt _ | .constantBool _ | .boolToInt | .intToReal | .intBinary _ |
      .realBinary _ | .realSqrt | .intCompare _ | .select | .trapdoorPublic | .matrixAdd |
      .matrixSubtract | .matrixMultiply | .matrixNegate | .transpose | .tensor | .concat _ |
      .familyPack | .familyGetDynamic => false
  | .evaluateInt value => intExprUsesLoop slot value
  | .constantReal value => realExprUsesLoopSlot slot value
  | .zeroMatrix matrixType | .identityMatrix matrixType | .liftIntegerToConstantPolynomial matrixType |
      .uniformResidueSample matrixType => matrixTypeUsesLoopSlot slot matrixType
  | .constantMatrix matrixType coefficients =>
      matrixTypeUsesLoopSlot slot matrixType || coefficients.any (intExprUsesLoop slot)
  | .unitRowMatrix matrixType index | .unitColumnMatrix matrixType index =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot index
  | .gadgetMatrix matrixType base | .smallGadgetMatrix matrixType base |
      .gadgetTrapdoor matrixType base =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot base
  | .powerOfBaseMatrix matrixType base exponent =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot base || intExprUsesLoop slot exponent
  | .rotationMatrix matrixType exponent =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot exponent
  | .bitExtract exponent | .extractCoefficient exponent | .matrixScale exponent |
      .familyGetStatic exponent => intExprUsesLoop slot exponent
  | .uniformIntervalSample matrixType minimum maximum =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot minimum || intExprUsesLoop slot maximum
  | .gaussianSample matrixType cutoff | .trapdoorSample matrixType cutoff |
      .preimageSample matrixType cutoff =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot cutoff
  | .hashSample matrixType _ _ tags decimalTags u64Tags base digitCount =>
      matrixTypeUsesLoopSlot slot matrixType || tags.any (intExprUsesLoop slot) ||
        decimalTags.any (intExprUsesLoop slot) || u64Tags.any (intExprUsesLoop slot) ||
        base.any (intExprUsesLoop slot) || digitCount.any (intExprUsesLoop slot)
  | .gadgetDecompose matrixType base _ digits =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot base || intExprUsesLoop slot digits
  | .slice rows columns => rows.any (fun (rowCount, columnCount) =>
      intExprUsesLoop slot rowCount || intExprUsesLoop slot columnCount) ||
        columns.any (fun (rowCount, columnCount) =>
          intExprUsesLoop slot rowCount || intExprUsesLoop slot columnCount)
  | .thresholdDecodeBool ciphertext plaintext length | .thresholdDecodeInt ciphertext plaintext length =>
      intExprUsesLoop slot ciphertext || intExprUsesLoop slot plaintext || intExprUsesLoop slot length
  | .crtRecompose plaintextModuli reconstructionCoefficients =>
      plaintextModuli.any (intExprUsesLoop slot) || reconstructionCoefficients.any (intExprUsesLoop slot)
  | .packPolynomialCoefficients matrixType coefficientBits =>
      matrixTypeUsesLoopSlot slot matrixType || intExprUsesLoop slot coefficientBits
  | .subgraphCall _ bindings => bindings.any fun (_, value) => intExprUsesLoop slot value
  | .parallelLoop _ count _ bindings _ | .sequentialLoop _ count _ bindings _ =>
      intExprUsesLoop slot count || bindings.any (fun (_, value) => intExprUsesLoop slot value)

private def nodeUsesLoopSlot (slot : Nat) (node : Node) : Bool :=
  nodeKindUsesLoopSlot slot node.kind || node.outputTypes.any (wireTypeUsesLoopSlot slot)

/-- Determine whether the sequential body needs the exact lexical binder before its evaluation.
Calls inherit their caller's lexical frame, while a nested loop with the same numeric slot shadows
it; scan that loop's own count/bindings but never mistake its body-local binder for the parent. -/
private def preparedScopeUsesLoopSlot
    (definitions : Array (String × PreparedOperationalScope))
    (slot : Nat) : Nat → PreparedOperationalScope → Bool
  | 0, _ => false
  | fuel + 1, prepared =>
      prepared.scope.nodes.zipIdx.any fun (node, nodeIndex) =>
        nodeUsesLoopSlot slot node ||
          match prepared.definitionIndices[nodeIndex]? with
          | some (some definitionIndex) =>
              match definitions[definitionIndex]? with
              | some (_, child) =>
                  match node.kind with
                  | .parallelLoop _ _ nestedSlot _ _ | .sequentialLoop _ _ nestedSlot _ _ =>
                      nestedSlot != slot && preparedScopeUsesLoopSlot definitions slot fuel child
                  | .subgraphCall .. => preparedScopeUsesLoopSlot definitions slot fuel child
                  | _ => false
              | none => false
          | _ => false

/-- Resolve the direct family lane binder from the producer's IR shape and declared family count.
The direct carrier may also contain an independent select-choice binder, which is deliberately
left in its context when a get substitutes only the family lane. -/
partial def directFamilyLaneCarrierTrace
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId) : Nat → String
  | 0 => "fuel_exhausted(root=" ++ toString root ++ ")"
  | fuel + 1 =>
      match arena.valueAt? root with
      | none => "missing(root=" ++ toString root ++ ")"
      | some value =>
          let context := reprStr value.context
          match value.payload with
          | .shared _ _ => "shared(root=" ++ toString root ++ "; context=" ++ context ++ ")"
          | .explicit _ binder _ =>
              "explicit(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
                "; context=" ++ context ++ ")"
          | .explicitValues _ binder _ =>
              "explicit_values(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
                "; context=" ++ context ++ ")"
          | .mapped _ source map =>
              "mapped(root=" ++ toString root ++ "; context=" ++ context ++
                "; assignments=" ++ reprStr map.assignments ++ ") -> " ++
                directFamilyLaneCarrierTrace arena source fuel
          | .rebound _ source subject =>
              "rebound(root=" ++ toString root ++ "; subject=" ++ reprStr subject ++
                "; context=" ++ context ++ ") -> " ++ directFamilyLaneCarrierTrace arena source fuel
          | .indexedOutput _ source binder selection subject =>
              "indexed_output(root=" ++ toString root ++ "; binder=" ++ reprStr binder ++
                "; selection=" ++ reprStr selection ++ "; subject=" ++ reprStr subject ++
                "; context=" ++ context ++ ") -> " ++ directFamilyLaneCarrierTrace arena source fuel
          | .matrixResultBound _ source _ =>
              "matrix_result_bound(root=" ++ toString root ++ "; context=" ++ context ++ ") -> " ++
                directFamilyLaneCarrierTrace arena source fuel
          | .pointwise _ _ inputs =>
              "pointwise(root=" ++ toString root ++ "; context=" ++ context ++
                "; inputs=" ++ reprStr inputs ++ ")"

partial def directFamilyLaneBinderFromCarrier
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId) : Nat → Option IndexVariable
  | 0 => none
  | fuel + 1 => do
      let value ← arena.valueAt? root
      match value.payload with
      /- A workflow artifact may enter the consuming stage as one shared external family: its
      physical lane coordinate lives in the owner-aware fact context rather than an explicit
      table payload.  It is usable for a zip only when that context has one unambiguous lane
      binder; `directFamilyLaneBinderAt` validates the declared family count before transport. -/
      | .shared _ _ =>
          match value.context.binders.toList with
          | [binder] => some binder
          | _ => none
      | .explicit _ binder _ | .explicitValues _ binder _ => some binder
      | .mapped _ source map => do
          let sourceValue ← arena.valueAt? source
          if !map.transportValid || map.source != sourceValue.context || map.destination != value.context then
            none
          else match directFamilyLaneBinderFromCarrier arena source fuel with
          | some sourceBinder =>
              match map.assignmentFor sourceBinder with
              | some assignment => assignment.identityVariable?
              | _ => none
          /- A direct-carrier context lift has no source lane to substitute: a parent loop can
          introduce exactly one owner-bearing destination lane around a shared artifact, then
          lazy rebound views preserve that map.  Recover only that checked singleton destination,
          never a binder reconstructed from the consumer scope or a multi-binder context. -/
          | none =>
              match sourceValue.context.binders.toList, map.destination.binders.toList,
                  value.context.binders.toList with
              | [], [destination], [current] =>
                  if map.isDirectCarrierContextLift && destination == current then some destination
                  else none
              | _, _, _ => none
      | .rebound _ source _ => directFamilyLaneBinderFromCarrier arena source fuel
      /- The overlay is the executable outer family produced by the parallel loop.  Its exact
      lexical selector is therefore the physical lane seen by a downstream zip; any source-family
      coordinate remains an independent nested dimension in the context. -/
      | .indexedOutput _ _ _ selection _ => selection.expression.identityVariable?
      | .matrixResultBound _ source _ => directFamilyLaneBinderFromCarrier arena source fuel
      /- `pushPointwise` constructs this context by merging its direct input contexts and checks
      the payload schemas. Re-establish both invariants here before using a unique owner-aware
      binder: a hand-constructed root must not advertise a lane absent from its inputs, and an
      arbitrary input cannot be selected because it may be a broadcast operand. -/
      | .pointwise schema operation inputs => do
          let inputValues ← inputs.toList.mapM arena.valueAt?
          let (inputContext, _) ← mergeIndexedFactShapeN inputValues
          let inputSchemas := inputValues.toArray.map fun input => input.payload.schema
          if inputContext != value.context || !pointwiseSchemasValid operation inputSchemas schema ||
              !validateContext value.context then
            none
          else
            let candidates := inputs.toList.filterMap fun input =>
              directFamilyLaneBinderFromCarrier arena input fuel
            match candidates.eraseDups with
            | [binder] => if value.context.binders.contains binder then some binder else none
            | _ => none

/-- Validate that one owner-bearing destination binder is actually transported by the carrier.
Unlike `directFamilyLaneBinderFromCarrier`, this does not choose an outermost coordinate: nested
families ask about the binder selected from their declared evaluated domain. -/
partial def directFamilyLaneBinderSupported
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId)
    (requested : IndexVariable) : Nat → Bool
  | 0 => false
  | fuel + 1 => match arena.valueAt? root with
      | none => false
      | some value => match value.payload with
          | .shared _ _ => value.context.binders.contains requested
          | .explicit _ binder _ | .explicitValues _ binder _ => binder == requested
          | .mapped _ source map =>
              match arena.valueAt? source with
              | none => false
              | some sourceValue =>
                  if !map.transportValid || map.source != sourceValue.context ||
                      map.destination != value.context then false
                  else
                    (sourceValue.context.binders.toList.any fun sourceBinder =>
                      (map.assignmentFor sourceBinder).bind IndexExpr.identityVariable? == some requested &&
                        directFamilyLaneBinderSupported arena source sourceBinder fuel) ||
                    (sourceValue.context.binders.isEmpty && map.isDirectCarrierContextLift &&
                      map.destination.binders.toList == [requested])
          | .rebound _ source _ => directFamilyLaneBinderSupported arena source requested fuel
          | .indexedOutput _ source _ selection _ =>
              selection.expression.identityVariable? == some requested ||
                directFamilyLaneBinderSupported arena source requested fuel
          | .matrixResultBound _ source _ =>
              directFamilyLaneBinderSupported arena source requested fuel
          | .pointwise schema operation inputs =>
              match inputs.toList.mapM arena.valueAt? with
              | none => false
              | some inputValues =>
                  let inputSchemas := inputValues.toArray.map fun input => input.payload.schema
                  match mergeIndexedFactShapeN inputValues with
                  | none => false
                  | some (inputContext, _) =>
                      inputContext == value.context && validateContext value.context &&
                        pointwiseSchemasValid operation inputSchemas schema &&
                        inputs.toList.any fun input =>
                          directFamilyLaneBinderSupported arena input requested fuel

private def directFamilyLaneBinderFailureDiagnostic
    (scopeKey : ScopeTemplateKey)
    (familyWire : WireRef)
    (count : IntExpr)
    (family : OperationalFact)
    (arena : DirectOperationalIndexedArena) : Bool :=
  operationalProgress "direct_family_lane_binder" "carrier_unresolved" (reprStr scopeKey)
    family.payload.root arena.values.size
    ("wire=" ++ reprStr familyWire ++ "; declared_count=" ++ reprStr count ++
      "; fact_context=" ++ reprStr family.context ++ "; chain=" ++
      directFamilyLaneCarrierTrace arena family.payload.root (arena.values.size + 1))

def directFamilyLaneBinderAt
    (arena : OperationalExprArena)
    (scopeKey : ScopeTemplateKey)
    (scope : Scope)
    (environment : ParamEnvironment)
    (familyWire : WireRef)
    (family : OperationalFact) : Except OperationalError IndexVariable := do
  let producer ← match scope.nodes[familyWire.node]? with
    | some node => pure node
    | none => throw (.missingOperand familyWire.node familyWire)
  let outputType ← match producer.outputTypes[familyWire.port]? with
    | some outputType => pure outputType
    | none => throw (.missingOperand familyWire.node familyWire)
  let countExpression ← match outputType with
    | .indexedFamily _ count => pure count
    | _ => throw (.loopInputModeMismatch familyWire.node familyWire.port)
  let count ← match countExpression.evaluate environment with
    | some value => pure value
    | none => throw .nonClosedExpression
  if count <= 0 then throw (.invalidCount familyWire.node count)
  match producer.kind with
  | .input _ =>
      /- A nested family may enter a child scope with both its own lane and enclosing loop or
      selection coordinates.  The child input declaration identifies the family lane by its
      positive evaluated domain size; storage shape and outermost carrier wrapper do not.  An
      equal-sized second coordinate is genuinely ambiguous and remains fail-closed. -/
      let root := family.payload.root
      if !validateContext family.context then throw (.unsupportedOperationalExpr root)
      let candidates := family.context.binders.toList.filter fun candidate =>
        match candidate.count.evaluate environment with
        | some candidateCount => candidateCount > 0 && candidateCount == count &&
            directFamilyLaneBinderSupported arena.direct root candidate
              (arena.direct.values.size + 1)
        | none => false
      match candidates with
      | [binder] => pure binder
      | _ =>
          if directFamilyLaneBinderFailureDiagnostic scopeKey familyWire countExpression family arena.direct
          then throw (.loopInputModeMismatch familyWire.node familyWire.port)
          else throw (.unsupportedOperationalExpr root)
  | _ =>
      let binder ← directFamilyLaneBinder scopeKey familyWire.node producer familyWire countExpression count.toNat
      if !family.context.binders.contains binder then
        throw (.loopInputModeMismatch familyWire.node familyWire.port)
      pure binder

/-- Preserve an indexed integer selector as the exact producer-backed gather used by an ordinary
branch select.  A logical selection identity is only safe for a context-free selector; otherwise
the direct carrier determines the unique runtime position and remains registered under the
selector wire's owner. -/
def executableDirectSelectExpression
    (scopeKey : ScopeTemplateKey)
    (node : Nat)
    (selectorWire : WireRef)
    (selection : OperationalIntegerFact)
    (selectionInput : OperationalFact)
    (branchCount : Nat)
    (arena : OperationalExprArena) : Except OperationalError (OperationalExprArena × Option IndexExpr) := do
  if selection.lower == selection.upper then pure (arena, none) else
    match selectionInput with
    | direct@{ payload := .directValue root, .. } =>
        if direct.context.binders.isEmpty then
          /- A context-free interval has no executable runtime coordinate to preserve.  Its
          origin-keyed identity is the declared selection semantics, but pass it explicitly so
          the direct branch selector cannot silently fall back from an indexed producer. -/
          pure (arena, some (DynamicSelectionIdentity.fromOrigin selection.origin branchCount).expression)
        else do
          let position ← match direct.context.binders.toList with
            | [binder] => pure binder
            | _ => match (directFamilyLaneBinderFromCarrier arena.direct root
                (arena.direct.values.size + 1)) with
              | some binder =>
                  if direct.context.binders.contains binder then pure binder
                  else throw (.unsupportedOperationalExpr node)
              | none => throw (.unsupportedOperationalExpr node)
          let owner : GatherLookupOwner := { indices := operationalGatherIndicesWire scopeKey selectorWire }
          let directArena ← match arena.direct.registerGatherIntegerRoot owner root position with
            | some directArena => pure directArena
            | none => throw (.unsupportedOperationalExpr node)
          pure ({ arena with direct := directArena },
            some (.gather owner (IntExpr.constant (Int.ofNat branchCount)) (.variable position)))

def deriveOrdinaryOutputs
    (scopeKey : ScopeTemplateKey)
    (nodeIndex : Nat)
    (node : Node)
    (rule : DerivationRule)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (facts : OperationalScopeFacts) :
    Nat → List WireTypeExpr →
    Except OperationalError (OperationalExprArena × List OperationalFact)
  | _, [] => pure (facts.arena, [])
  | port, outputType :: tail => do
      let (arena, output) ← genericNodeFact scopeKey nodeIndex node rule port outputType facts
        environment loopDomains layouts
      let (arena, tail) ← deriveOrdinaryOutputs scopeKey nodeIndex node rule environment
        loopDomains layouts { facts with arena } (port + 1) tail
      pure (arena, output :: tail)

/-! Evaluate a single unresolved selection by streaming complete concrete alternatives into a
consumer.  A consumer that would combine two selected subexpressions rejects instead of silently
performing Cartesian-time traversal. Selection-aware bound evaluation uses `evaluateCompleteBound`
and does not call this endpoint helper. -/
def evaluatePrimitiveConcrete
    (operation : PrimitiveOperation)
    (arguments : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let binaryArguments : Except OperationalError (OperationalMatrixFact × OperationalMatrixFact) := do
    if arguments.size != 2 then
      throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let left ← match arguments[0]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let right ← match arguments[1]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    pure (left, right)
  match operation.kind with
  | .add subtract =>
      let (left, right) ← binaryArguments
      return ← addConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputSchema
        subtract operation.parameterEnvironment left right
  | .multiply rule rightWire =>
      let (left, right) ← binaryArguments
      return ← multiplyConcreteMatrixFacts operation.ownerNode operation.outputPort
        operation.outputSchema rule rightWire operation.parameterEnvironment left right
  | .tensor =>
      let (left, right) ← binaryArguments
      return ← tensorConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputSchema
        operation.parameterEnvironment left right
  | .concat axis =>
      return ← concatConcreteMatrixFacts operation.ownerNode operation.outputPort axis
        operation.outputSchema operation.parameterEnvironment arguments
  | .transform transform =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      return ← transformConcreteMatrixFact operation.ownerNode operation.outputPort
        operation.outputSchema transform operation.parameterEnvironment value
  | .slice rows columns =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let polynomial ← sliceOperationalPolynomial rows columns operation.outputSchema value.polynomial
        |>.mapError (flatErrorAt operation.ownerNode)
      polynomialMatrixFact operation.ownerNode operation.outputPort operation.outputSchema
        operation.parameterEnvironment polynomial value.canonicalRange
  | .scale scalar loopDomains =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let scalar ← match scalar with
        | .ir scalar => pure scalar | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let loopDomains : List OperationalParameterDomain ← match loopDomains with
        | [] => pure [] | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
      let scalarValue ← evaluateIntInvariant operation.parameterEnvironment loopDomains scalar
      return ← scaleConcreteMatrixFact operation.ownerNode operation.outputPort
        operation.outputSchema scalar [scalarValue] operation.parameterEnvironment loopDomains value
  | .bggGrouping =>
      if arguments.size != 3 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let vector ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let publicKey ← match arguments[1]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let plaintext ← match arguments[2]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      return ← groupBggEncodingSignal vector publicKey plaintext
        |>.mapError (.flat operation.ownerNode)


/-- Recovers the direct carrier's representative matrix fact for a wire. -/
def matrixFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef)
    (environment : ParamEnvironment := []) : Except OperationalError OperationalMatrixFact := do
  let expression ← lookupFact node facts wire
  return ← facts.arena.directValueRepresentativeFactAt environment expression

def matrixMaximum
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError Int := do
  let expression ← lookupFact node facts wire
      let bounds ← (← facts.arena.reducedDirectValueFactsAt environment expression).mapM
        fun entry => match entry.fact.totalHardBound with
          | .closedInt (.constant value) => pure value
          | expression => expression.evaluateWithStates environment []
      match bounds with
      | head :: tail => pure (tail.foldl max head)
      | [] => throw (.invalidCount node 0)

def matrixMaximumExpr
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError OperationalBoundExpr := do
  let expression ← lookupFact node facts wire
      let entries ← facts.arena.reducedDirectValueFactsAt environment expression
      match entries with
      | head :: tail => pure (tail.foldl (fun bound entry =>
          .maximum bound entry.fact.totalHardBound) head.fact.totalHardBound)
      | [] => throw (.invalidCount node 0)

def maximumArgumentExprs
    (node : Nat)
    (arguments : List WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError OperationalBoundExpr := do
  let values ← arguments.mapM (fun wire => matrixMaximumExpr node wire facts environment)
  pure <| values.foldl OperationalBoundExpr.maximum (.closedInt (.constant 0))

def maximumArguments
    (node : Nat)
    (arguments : List WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError Int := do
  let values ← arguments.mapM (fun wire => matrixMaximum node wire facts environment)
  pure <| values.foldl max 0

def factHasRelation
    (arena : OperationalExprArena) (fact : OperationalFact) : Except OperationalError Bool := do
  pure <| matrixFactHasRelation (← arena.directValueFactAt [] fact)

def matrixBoundaryPublicIdentityMatches
    (expected : PublicMatrixIdentity)
    (fact : OperationalMatrixFact) : Bool :=
  match boundaryLastPublicIdentity? fact with
  | some actual => actual == expected || publicIdentityTemplateEqual actual expected
  | none => false

/-- Require the registered public-matrix boundary on every complete alternative.  Exact
selections are checked branch-by-branch.  Compact envelopes are accepted only through their
validated representative and complete shared boundary template; one mismatching branch therefore
rejects the endpoint rather than being hidden by a numerical maximum. -/
partial def requireOperationalBoundaryPublicIdentity
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (node : Nat)
    (expected : PublicMatrixIdentity) : OperationalFact → Except OperationalError Unit
  | expression => do
      let entries ← arena.reducedDirectValueFactsAt environment expression
      if entries.all (fun entry => matrixBoundaryPublicIdentityMatches expected entry.fact) then pure ()
      else throw (.publicIdentityMismatch node)

def sequentialFactHasRelation
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError Bool
  | expression => do
      let value ← match arena.direct.valueAt? expression.payload.root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef expression.payload.root)
      match value.payload.schema with
      | OperationalIndexedPayloadSchema.matrix _ =>
          pure ((← arena.reducedDirectValueFactsAt environment expression).any
            (fun entry => matrixFactHasRelation entry.fact))
      | OperationalIndexedPayloadSchema.scalar _ => pure false

def summarizeSequentialFact
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError OperationalFact
  | expression => do
      if ← sequentialFactHasRelation arena environment expression then
        throw (.relationBearingCarriedValue temporaryScope 0 0)
      pure expression

/-- Substitute one simultaneous previous-state slot through every concrete leaf of an indexed DAG.
The arena mapper also rebuilds Shared envelopes from the mapped conservative fact, so selection
context and storage remain semantic metadata rather than a fact-level fallback. -/
def abstractSequentialFact
    (environment : ParamEnvironment) (slot : Nat)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | { payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      let (direct, mapped) ← match value.payload.schema with
        | OperationalIndexedPayloadSchema.matrix _ =>
            arena.direct.mapMatrixValue root (fun fact => do
              let maximum := OperationalBoundExpr.previous (.matrixMaximum 0 slot)
              let polynomial := fact.polynomial.map fun term => { term with product := {
                term.product with
                factors := term.product.factors.map (replaceOperationalFactorHardBound maximum) }}
              pure { fact with totalHardBound := maximum, polynomial })
        | OperationalIndexedPayloadSchema.scalar _ =>
            arena.direct.mapScalarValue environment root
              (fun fact => pure (abstractCarriedScalarMaximum slot fact))
      let value ← match direct.valueAt? mapped with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef mapped)
      let rebound : OperationalFact := {
        context := value.context
        payload := .directValue mapped
        storage := value.storage
      }
      pure ({ arena with direct }, rebound)

def setSequentialFactRecurrenceState
    (count : Nat)
    (paths : List OperationalBoundPath)
    (initial transition : List OperationalBoundExpr)
    (slot : Nat)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | { payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      let (direct, mapped) ← match value.payload.schema with
        | OperationalIndexedPayloadSchema.matrix _ =>
            let maximum := OperationalBoundExpr.recurrenceState
              count paths initial transition (.matrixMaximum 0 slot)
            match arena.direct.pushMatrixResultBound root maximum with
            | some result => pure result
            | none => throw (.unsupportedOperationalExpr root)
        | OperationalIndexedPayloadSchema.scalar _ =>
            arena.direct.mapScalarValue environment root (fun fact => match fact with
              | .trapdoor fact =>
                  let maximum := OperationalBoundExpr.recurrenceState count paths initial transition
                    (.matrixMaximum 0 slot)
                  pure (.trapdoor { fact with maximum := .closed maximum })
              | .integer fact =>
                  let lowerExpression := OperationalBoundExpr.recurrenceState count paths
                    initial transition (.integerLower 0 slot)
                  let upperExpression := OperationalBoundExpr.recurrenceState count paths
                    initial transition (.integerUpper 0 slot)
                  let fact := { fact with lowerExpression := .closed lowerExpression }
                  pure (.integer { fact with upperExpression := .closed upperExpression })
              | fact => pure fact)
      let (direct, mapped) ← match value.payload.schema with
        | OperationalIndexedPayloadSchema.scalar .integer =>
            direct.mapScalarValue environment mapped (fun fact => match fact with
              | .integer integer => do
                  let lowerExpression ← match integer.lowerExpression.closedOperational? with
                    | some expression => pure expression
                    | none => throw (.unsupportedOperationalExpr mapped)
                  let upperExpression ← match integer.upperExpression.closedOperational? with
                    | some expression => pure expression
                    | none => throw (.unsupportedOperationalExpr mapped)
                  let lower ← lowerExpression.evaluateWithStates environment []
                  let upper ← upperExpression.evaluateWithStates environment []
                  if lower > upper then throw (.invalidBound slot lower)
                  pure (.integer { integer with lower, upper })
              | _ => throw (.unsupportedOperationalExpr mapped))
        | _ => pure (direct, mapped)
      let mappedValue ← match direct.valueAt? mapped with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef mapped)
      let rebound : OperationalFact := {
        context := mappedValue.context
        payload := .directValue mapped
        storage := mappedValue.storage
      }
      pure ({ arena with direct }, rebound)

def evaluatePreparedScope
    (definitions : Array (String × PreparedOperationalScope))
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    ScopeTemplateKey → Nat → PreparedOperationalScope → ParamEnvironment →
      List OperationalParameterDomain →
      OperationalExprArena →
      List OperationalFact →
      Except OperationalError OperationalScopeFacts
  | _, 0, _, _, _, _, _ => .error .definitionFuelExhausted
  | scopeKey, fuel + 1, prepared, environment, loopDomains, initialArena, inputFacts => do
      let scope := prepared.scope
      let derivation := prepared.derivation
      if operationalProgress "evaluate_scope" "scope_start" (reprStr scopeKey) 0 scope.nodes.size
          ("input_facts=" ++ toString inputFacts.length ++ "; fuel=" ++ toString (fuel + 1)) then
        pure ()
      else throw (.unsupportedOperationalExpr 0)
      if !inputFacts.isEmpty && inputFacts.length != scope.inputNames.length then
        throw (.childInputMismatch 0 scope.inputNames.length inputFacts.length)
      let rec collectChildOutputs
          (callerNode port : Nat)
          (outputs : List (String × WireRef))
          (arena : OperationalExprArena)
          (facts : OperationalScopeFacts) :
          Except OperationalError (OperationalExprArena × List OperationalFact) := do
        match outputs with
        | [] => pure (arena, [])
        | (_, wire) :: tail =>
            let fact ← lookupFact callerNode facts wire
            let (arena, rebound) ← rebindOperationalFact { node := callerNode, port } arena fact environment
            let (arena, tail) ← collectChildOutputs callerNode (port + 1) tail arena facts
            pure (arena, rebound :: tail)
      let rec scopeOutputFacts
          (callerNode : Nat)
          (outputs : List (String × WireRef))
          (facts : OperationalScopeFacts) : Except OperationalError (List OperationalFact) := do
        match outputs with
        | [] => pure []
        | (_, wire) :: tail =>
            return (← lookupFact callerNode facts wire) ::
              (← scopeOutputFacts callerNode tail facts)
      let rec prepareParallelInputs
          (nodeIndex indexSlot argumentIndex : Nat)
          (declaredCount : IntExpr)
          (count : Nat)
          (modes : List LoopInputMode)
          (wires : List WireRef)
          (inputs : List OperationalFact)
          (arena : OperationalExprArena) :
          Except OperationalError (OperationalExprArena × List OperationalFact) := do
        match modes, wires, inputs with
        | [], [], [] => pure (arena, [])
        | mode :: modeTail, wire :: wireTail, input :: inputTail =>
            let directLaneBinder ← match mode with
              | .zip | .zipOffset _ => some <$> directFamilyLaneBinderAt arena scopeKey scope environment wire input
              | _ => pure none
            let (arena, head) ←
              loopTemplateArgumentExprWithDirectLaneBinder arena scopeKey nodeIndex indexSlot argumentIndex
                declaredCount count mode directLaneBinder environment input
            let (arena, tail) ← prepareParallelInputs nodeIndex indexSlot (argumentIndex + 1)
              declaredCount count modeTail wireTail inputTail arena
            pure (arena, head :: tail)
        | _, _, _ => throw (.loopInputModeMismatch nodeIndex argumentIndex)
      let mut facts : OperationalScopeFacts := {
        arena := { initialArena with activeScope := some scopeKey, activeNode := none }
      }
      for node in scope.nodes do
            let index := facts.values.size
            if operationalProgressBlock index scope.nodes.size then
              if operationalProgress "evaluate_scope" "node_block_start" (reprStr scopeKey)
                  index scope.nodes.size ("output_count=" ++ toString node.outputCount ++
                    "; arena_values=" ++ toString facts.arena.direct.values.size) then pure () else
                throw (.unsupportedOperationalExpr index)
            else pure ()
            facts := { facts with arena := {
              facts.arena with activeScope := some scopeKey, activeNode := some index
            } }
            if node.outputCount != node.outputTypes.length then
              throw (.unsupportedOutputArity index node.outputCount)
            let step ← match derivation.steps[index]? with
              | some step => pure step
              | none => throw (.derivation (.missingNode index))
            if operationalProgressBlock index scope.nodes.size then
              if operationalProgress "evaluate_scope" "node_rule" (reprStr scopeKey)
                  index scope.nodes.size ("rule=" ++ reprStr step.rule ++
                    "; arguments=" ++ reprStr node.arguments) then pure () else
                throw (.unsupportedOperationalExpr index)
            else pure ()
            let outputs ← try
              match node.kind with
              | .input _ =>
                  if inputFacts.isEmpty then
                    let (arena, outputs) ← deriveOrdinaryOutputs scopeKey index node step.rule
                      environment loopDomains layouts facts 0
                        node.outputTypes
                    facts := { facts with arena }
                    pure outputs
                  else
                    match prepared.inputIndices[index]? with
                    | some (some inputIndex) =>
                        match inputFacts[inputIndex]? with
                        | some input => do
                            let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                              facts.arena input environment
                            facts := { facts with arena }
                            pure [rebound]
                        | none => throw (OperationalError.childInputMismatch index
                            scope.inputNames.length inputFacts.length)
                    | _ => throw (OperationalError.childInputMismatch index
                        scope.inputNames.length inputFacts.length)
              | .subgraphCall _ bindings =>
                  if operationalProgress "evaluate_scope" "subgraph_call_start" (reprStr scopeKey)
                      index scope.nodes.size ("bindings=" ++ toString bindings.length) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let boundParams ← match evaluateBindings environment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ← extendParameterDomains environment loopDomains bindings
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .callBody scopeKey index
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child (boundParams ++ environment)
                    childDomains facts.arena actualInputs).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  let (arena, outputs) ←
                    collectChildOutputs index 0 child.scope.outputs facts.arena childFacts
                  facts := { facts with arena }
                  if operationalProgress "evaluate_scope" "subgraph_call_complete" (reprStr scopeKey)
                      index scope.nodes.size ("outputs=" ++ toString outputs.length) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  pure outputs
              | .familyPack =>
                  if operationalProgress "evaluate_scope" "family_pack_start" (reprStr scopeKey)
                      index scope.nodes.size ("arguments=" ++ toString node.arguments.length) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let elements ← node.arguments.mapM (lookupFact index facts)
                  match node.outputTypes with
                  | [.indexedFamily (.matrix _) familyCount] | [.indexedFamily (.preimage _) familyCount] =>
                      let count ← match familyCount.evaluate environment with
                        | some value => pure value
                        | none => throw .nonClosedExpression
                      if count <= 0 || elements.length != count.toNat then
                        throw (.invalidCount index count)
                      let (arena, family) ← packDirectMatrixFamily scopeKey index environment familyCount
                        facts.arena elements.toArray
                      facts := { facts with arena }
                      pure [family]
                  | [.indexedFamily _ count] =>
                      let count ← match count.evaluate environment with
                        | some value => pure value
                        | none => throw .nonClosedExpression
                      if count <= 0 || elements.length != count.toNat then
                        throw (.invalidCount index count)
                      let (arena, directElements) ← elements.foldlM (fun (arena, packed) element =>
                        pure (arena, packed.push element)) (facts.arena, #[])
                      let (arena, family) ← packDirectScalarFamily scopeKey index environment
                        (match node.outputTypes with
                        | [.indexedFamily _ declaredCount] => declaredCount
                        | _ => .constant count) arena directElements
                      facts := { facts with arena }
                      pure [family]
                  | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
              | .familyGetStatic familyIndex =>
                  if operationalProgress "evaluate_scope" "family_static_get_start" (reprStr scopeKey)
                      index scope.nodes.size ("index=" ++ reprStr familyIndex) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let requested ← match familyIndex.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  let family ← lookupFact index facts familyWire
                  let binder ← directFamilyLaneBinderAt facts.arena scopeKey scope environment familyWire family
                  if requested < 0 then throw (.invalidCount index requested)
                  let staticMap ← match closedStaticIndexMap environment family.context binder requested.toNat with
                    | some map => pure map
                    | none => throw (.loopInputModeMismatch index 0)
                  let (arena, selected) ← facts.arena.reindexDirectFact staticMap family environment
                  let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                    arena selected environment
                  facts := { facts with arena }
                  pure [rebound]
              | .familyGetDynamic =>
                  if operationalProgress "evaluate_scope" "family_dynamic_get_start" (reprStr scopeKey)
                      index scope.nodes.size ("arguments=" ++ toString node.arguments.length) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let indexWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let selectionInput ← lookupFact index facts indexWire
                  let selectionFact : OperationalIntegerFact ← match selectionInput with
                    | { payload := .directValue root, .. } => do
                        let (lower, upper) ← facts.arena.direct.integerInterval index indexWire root
                          (facts.arena.direct.values.size + 1)
                        pure {
                          subject := indexWire
                          origin := .local scopeKey indexWire
                          lower
                          upper
                          lowerExpression := .closed (.closedInt (.constant lower))
                          upperExpression := .closed (.closedInt (.constant upper))
                        }
                  let family ← lookupFact index facts familyWire
                  match selectionFact.lower == selectionFact.upper with
                  | true =>
                      let binder ← directFamilyLaneBinderAt facts.arena scopeKey scope environment familyWire family
                      let requested := selectionFact.lower
                      if requested < 0 then throw (.invalidCount index requested)
                      let staticMap ← match closedStaticIndexMap environment family.context binder requested.toNat with
                        | some map => pure map
                        | none => throw (.loopInputModeMismatch index 0)
                      let (arena, selected) ← facts.arena.reindexDirectFact staticMap family environment
                      let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                        arena selected environment
                      facts := { facts with arena }
                      pure [rebound]
                  | false =>
                      let binder ← directFamilyLaneBinderAt facts.arena scopeKey scope environment familyWire family
                      let selectorCount := match binder.count.evaluate environment with
                        | some count => if count > 0 then count.toNat else 0
                        | none => 0
                      if selectorCount == 0 || selectionFact.lower < 0 ||
                          selectionFact.upper >= Int.ofNat selectorCount then
                        throw (.invalidCount index selectionFact.upper)
                      let selector ← match selectionInput with
                        | direct@{ payload := .directValue root, .. } => do
                            if direct.context.binders.isEmpty then do
                              /- A dynamic family projection introduces its own executable selector
                              coordinate when the scalar is not already indexed. -/
                              let projectionOrigin : OperationalValueOrigin :=
                                .local scopeKey { node := index, port := 1 }
                              let freshSelector := DynamicSelectionIdentity.fromDeclaredCount
                                projectionOrigin binder.count
                              let freshBinder ← match freshSelector.expression with
                                | .variable value => pure value
                                | _ => throw (.loopInputModeMismatch index 0)
                              pure <| match family.context.binders.toList.find? (fun candidate =>
                                  candidate.owner == freshBinder.owner && candidate.slot == freshBinder.slot) with
                                | some candidate => .variable candidate
                                | none => freshSelector.expression
                            else do
                              if operationalProgress "evaluate_scope" "gather_selector_transport" (reprStr scopeKey)
                                  index scope.nodes.size "source=loop_index" then pure () else
                                throw (.unsupportedOperationalExpr index)
                              let position ← match direct.context.binders.toList with
                                | [binder] => pure binder
                                | _ => match (directFamilyLaneBinderFromCarrier facts.arena.direct root
                                    (facts.arena.direct.values.size + 1)) with
                                  | some binder =>
                                      if direct.context.binders.contains binder then pure binder
                                      else
                                        if operationalProgress "evaluate_scope"
                                            "gather_selector_carrier_unresolved" (reprStr scopeKey)
                                            index scope.nodes.size
                                            ("root=" ++ toString root ++
                                              "; context=" ++ reprStr direct.context ++
                                              "; recovered=" ++ reprStr binder ++
                                              "; chain=" ++ directFamilyLaneCarrierTrace
                                                facts.arena.direct root
                                                (facts.arena.direct.values.size + 1)) then
                                          throw (.loopInputModeMismatch index 0)
                                        else throw (.unsupportedOperationalExpr index)
                                  | none =>
                                      if operationalProgress "evaluate_scope"
                                          "gather_selector_carrier_unresolved" (reprStr scopeKey)
                                          index scope.nodes.size
                                          ("root=" ++ toString root ++
                                            "; context=" ++ reprStr direct.context ++
                                            "; recovered=none; chain=" ++ directFamilyLaneCarrierTrace
                                              facts.arena.direct root
                                              (facts.arena.direct.values.size + 1)) then
                                        throw (.loopInputModeMismatch index 0)
                                      else throw (.unsupportedOperationalExpr index)
                              let owner : GatherLookupOwner := {
                                indices := operationalGatherIndicesWire scopeKey indexWire
                              }
                              pure (.gather owner binder.count (.variable position))
                      /- A dependent gather is evaluated from the exact executable integer-family
                      producer, not from a slot-only `IntExpr` projection.  Register before
                      constructing the mapped matrix view so any collision is rejected at the
                      producer boundary rather than depending on later reduction order. -/
                      match selector with
                      | .gather owner _ positionExpression =>
                          let root ← match selectionInput.payload with
                            | .directValue root => pure root
                          let position ← match positionExpression.identityVariable? with
                            | some position => pure position
                            | none => throw (.unsupportedOperationalExpr index)
                          let direct ← match facts.arena.direct.registerGatherIntegerRoot owner root position with
                            | some direct => pure direct
                            | none => throw (.unsupportedOperationalExpr index)
                          facts := { facts with arena := { facts.arena with direct } }
                      | _ => pure ()
                      if selector.freeVariables.isEmpty then throw (.loopInputModeMismatch index 0)
                      let dynamicCandidate := dynamicIndexMap family.context binder selector
                      let closedCandidate := closedDynamicIndexMap environment family.context binder selector
                      let dynamicMap ← match dynamicCandidate with
                        | some map => pure map
                        | none => match closedCandidate with
                          | some map => pure map
                          | none =>
                              if operationalProgress "evaluate_scope" "family_dynamic_get_map_rejected"
                                  (reprStr scopeKey) index scope.nodes.size
                                  ("family_context=" ++ reprStr family.context ++
                                    "; source_binder=" ++ reprStr binder ++
                                    "; selector=" ++ reprStr selector ++
                                    "; selector_free=" ++ reprStr selector.freeVariables ++
                                    "; selection_context=" ++ reprStr selectionInput.context) then
                                throw (.loopInputModeMismatch index 0)
                              else throw (.unsupportedOperationalExpr index)
                      let (arena, selected) ← facts.arena.reindexDirectFact dynamicMap family environment
                      if operationalProgress "evaluate_scope" "family_dynamic_get_reindex_complete"
                          (reprStr scopeKey) index scope.nodes.size "" then pure () else
                        throw (.unsupportedOperationalExpr index)
                      let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                        arena selected environment
                      facts := { facts with arena }
                      pure [rebound]
              | .parallelLoop _ count indexSlot bindings modes =>
                  if operationalProgress "evaluate_scope" "parallel_loop_prepare_start" (reprStr scopeKey)
                      index scope.nodes.size ("index_slot=" ++ toString indexSlot ++
                        "; modes=" ++ toString modes.length ++ "; count=" ++ reprStr count) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let evaluatedCount ← match count.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  if evaluatedCount <= 0 then throw (.invalidCount index evaluatedCount)
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let parentDomains := .loopIndex indexSlot evaluatedCount.toNat ::
                    loopDomains.filter fun domain => match domain with
                      | .loopIndex candidate _ => candidate != indexSlot
                      | .parameter _ _ _ _ => true
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .parallelBody scopeKey index
                  let (arena, templateInputs) ←
                    prepareParallelInputs index indexSlot 0 count evaluatedCount.toNat modes node.arguments
                      actualInputs facts.arena
                  facts := { facts with arena }
                  let iterationEnvironment :=
                    (ParamKey.loopIndex indexSlot, ParamValue.integer 0) :: environment
                  let boundParams ← match evaluateBindings iterationEnvironment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ←
                    extendParameterDomains iterationEnvironment parentDomains bindings
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child (boundParams ++ iterationEnvironment)
                    childDomains facts.arena templateInputs).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  if operationalProgress "evaluate_scope" "parallel_loop_body_complete" (reprStr scopeKey)
                      index scope.nodes.size ("index_slot=" ++ toString indexSlot ++
                        "; child_outputs=" ++ toString childFacts.values.size) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let childOutputs ← scopeOutputFacts index child.scope.outputs childFacts
                  if childOutputs.length != node.outputCount then
                    throw (.childInputMismatch index node.outputCount childOutputs.length)
                  let (nextFacts, outputs) ← childOutputs.zipIdx.foldlM
                    (fun (currentFacts, accumulated) (output, port) => do
                      match output with
                      /- Parallel results are direct indexed values.  Closing installs the one
                      exact loop coordinate shared by every argument and output of this node. -/
                      | direct =>
                          let root := direct.payload.root
                          let value ← match currentFacts.arena.direct.valueAt? root with
                            | some value => pure value
                            | none => throw (.invalidOperationalExprRef root)
                          let schemaTag := match value.payload.schema with
                            | .matrix _ => "matrix"
                            | .scalar _ => "scalar"
                          let detail := "port=" ++ toString port ++ "; root=" ++ toString root ++
                            "; context=" ++ reprStr direct.context ++ "; schema=" ++ schemaTag
                          if port == 0 || port + 1 == childOutputs.length then
                            if operationalProgress "parallel_loop_output_close" "start" (reprStr scopeKey)
                                port childOutputs.length detail then pure () else
                              throw (.unsupportedOperationalExpr root)
                          else pure ()
                          let closedResult : Except OperationalError (OperationalExprArena × OperationalFact) :=
                            match value.payload.schema with
                            | .matrix _ =>
                                parallelLoopIndexedMatrixOutput scopeKey index indexSlot port count
                                  evaluatedCount.toNat environment currentFacts.arena direct
                            | .scalar _ =>
                                closeParallelDirectScalarOutput scopeKey index indexSlot port count
                                  environment currentFacts.arena direct
                          match closedResult with
                          | .ok (arena, family) =>
                              if port == 0 || port + 1 == childOutputs.length then
                                if operationalProgress "parallel_loop_output_close" "complete" (reprStr scopeKey)
                                    port childOutputs.length detail then pure () else
                                  throw (.unsupportedOperationalExpr root)
                              else pure ()
                              pure ({ currentFacts with arena }, accumulated.push family)
                          | .error error =>
                              if operationalProgress "parallel_loop_output_close" "error" (reprStr scopeKey)
                                  port childOutputs.length (detail ++ "; error=" ++ reprStr error) then
                                throw error
                              else throw error
                      )
                    (facts, #[])
                  facts := nextFacts
                  pure outputs.toList
              | .sequentialLoop _ count indexSlot bindings carriedCount =>
                  if operationalProgress "evaluate_scope" "sequential_loop_prepare_start" (reprStr scopeKey)
                      index scope.nodes.size ("index_slot=" ++ toString indexSlot ++
                        "; carried=" ++ toString carriedCount ++ "; count=" ++ reprStr count) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  let evaluatedCount ← match count.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  if evaluatedCount < 0 then throw (.invalidCount index evaluatedCount)
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let carriedFacts := actualInputs.take carriedCount
                  let invariantFacts := actualInputs.drop carriedCount
                  if carriedFacts.length != carriedCount then
                    throw (.childInputMismatch index carriedCount carriedFacts.length)
                  for (fact, slot) in carriedFacts.zipIdx do
                    let relationBearing ← sequentialFactHasRelation facts.arena environment fact
                    if relationBearing then
                      throw (.relationBearingCarriedValue scopeKey index slot)
                  let mut abstractCarried : List OperationalFact := []
                  for (fact, slot) in carriedFacts.zipIdx do
                    let (arena, abstract) ← abstractSequentialFact environment slot facts.arena fact
                    facts := { facts with arena }
                    abstractCarried := abstractCarried ++ [abstract]
                  let mut shiftedInvariantFacts : List OperationalFact := []
                  for fact in invariantFacts do
                    let (arena, shifted) ← shiftFactPreviousDepth environment facts.arena fact
                    facts := { facts with arena }
                    shiftedInvariantFacts := shiftedInvariantFacts ++ [shifted]
                  let child ← preparedDefinitionAt index prepared definitions
                  let needsLexicalBinder := preparedScopeUsesLoopSlot definitions indexSlot fuel child
                  if operationalProgress "evaluate_scope" "sequential_loop_dependency_gate"
                      (reprStr scopeKey) index scope.nodes.size
                      ("index_slot=" ++ toString indexSlot ++ "; lexical_binder=" ++
                        toString needsLexicalBinder) then pure () else
                    throw (.unsupportedOperationalExpr index)
                  /- Introduce the sequential lexical coordinate only when the prepared body
                  actually reads this loop's slot.  Otherwise a synthetic zero-valued binder
                  would leak into invariant carried descriptors and their contextual domains. -/
                  let mut bodyInputs : List OperationalFact := []
                  for fact in abstractCarried ++ shiftedInvariantFacts do
                    if needsLexicalBinder then
                      let (arena, indexed) ← sequentialLoopTemplateArgumentExpr facts.arena scopeKey
                        index indexSlot count environment fact
                      facts := { facts with arena }
                      bodyInputs := bodyInputs ++ [indexed]
                    else
                      bodyInputs := bodyInputs ++ [fact]
                  let iterationEnvironment := if needsLexicalBinder then
                    replaceLoopIndex environment indexSlot 0 else environment
                  let sequentialDomains := if needsLexicalBinder then
                    .loopIndex indexSlot evaluatedCount.toNat ::
                      loopDomains.filter fun domain => match domain with
                        | .loopIndex candidate _ => candidate != indexSlot
                        | .parameter _ _ _ _ => true
                  else loopDomains
                  let boundParams ← match evaluateBindings iterationEnvironment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ← extendParameterDomains iterationEnvironment sequentialDomains bindings
                  let childKey := .sequentialBody scopeKey index
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child
                    (boundParams ++ iterationEnvironment) childDomains
                    facts.arena bodyInputs).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  let rawOutputTemplates ← scopeOutputFacts index child.scope.outputs childFacts
                  if rawOutputTemplates.length != carriedCount then
                    throw (.childInputMismatch index carriedCount rawOutputTemplates.length)
                  let mut initialTemplates : List OperationalFact := []
                  for carried in carriedFacts do
                    initialTemplates := initialTemplates ++
                      [← summarizeSequentialFact facts.arena environment carried]
                  let mut outputTemplates : List OperationalFact := []
                  for output in rawOutputTemplates do
                    outputTemplates := outputTemplates ++
                      [← summarizeSequentialFact facts.arena environment output]
                  for slot in List.range carriedCount do
                    match initialTemplates[slot]?, outputTemplates[slot]? with
                    | some initial, some output =>
                        if !sameSequentialCarriedSchema facts.arena initial output ||
                            (← sequentialFactHasRelation facts.arena environment output) then
                          if ← sequentialFactHasRelation facts.arena environment output then
                            throw (.relationBearingCarriedValue scopeKey index slot)
                          else do
                            let initialCounts ← sequentialCarriedLargeFactorCounts facts.arena initial
                            let outputCounts ← sequentialCarriedLargeFactorCounts facts.arena output
                            throw (.sequentialSchemaMismatch scopeKey index slot initialCounts outputCounts)
                    | _, _ => throw (.childInputMismatch index carriedCount outputTemplates.length)
                  let mut initialComponents : List
                      (OperationalBoundPath × OperationalBoundExpr) := []
                  for (carried, slot) in initialTemplates.zipIdx do
                    initialComponents := initialComponents ++
                      (← sequentialFactNumericExpressions facts.arena slot carried)
                  let mut transitionComponents : List
                      (OperationalBoundPath × OperationalBoundExpr) := []
                  for (output, slot) in outputTemplates.zipIdx do
                    transitionComponents := transitionComponents ++
                      (← sequentialFactNumericExpressions facts.arena slot output)
                  let paths := initialComponents.map (·.1)
                  /- Every carried component has one unique current-depth state address.  This
                  rejects an unsupported fixed carried schema rather than allowing two scalar
                  bounds to overwrite one another in `numericStateFromComponents`. -/
                  if (!paths.isEmpty && (paths.eraseDups.length != paths.length ||
                      paths.any (fun path => !operationalBoundPathAtCurrentDepth path))) ||
                      paths != transitionComponents.map (·.1) then
                    throw (.sequentialSchemaMismatch scopeKey index 0 [] [])
                  let initialExpressions := initialComponents.map (·.2)
                  let transitions := transitionComponents.map (·.2)
                  if evaluatedCount = 0 then
                    let mut arena := facts.arena
                    let mut outputs : List OperationalFact := []
                    for (output, port) in carriedFacts.zipIdx do
                      let (nextArena, rebound) ← rebindOperationalFact
                        { node := index, port } arena output environment
                      arena := nextArena
                      outputs := outputs ++ [rebound]
                    facts := { facts with arena }
                    pure outputs
                  else do
                    let mut outputs : List OperationalFact := []
                    for (output, slot) in rawOutputTemplates.zipIdx do
                      let (arena, output) ← setSequentialFactRecurrenceState evaluatedCount.toNat
                        paths initialExpressions transitions slot environment facts.arena output
                      facts := { facts with arena }
                      let (arena, rebound) ← rebindOperationalFact
                        { node := index, port := slot } facts.arena output environment
                      facts := { facts with arena }
                      outputs := outputs ++ [rebound]
                    pure outputs
              | .thresholdDecodeBool _ _ _ | .thresholdDecodeInt _ _ _ =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  match input with
                  | input@{ payload := .directValue _, .. } =>
                      let mut arena := facts.arena
                      let mut outputs : List OperationalFact := []
                      for (outputType, port) in node.outputTypes.zipIdx do
                        let kind ← match node.kind with
                          | .thresholdDecodeBool ciphertext plaintext length =>
                              pure (.thresholdDecodeBool ciphertext plaintext length)
                          | .thresholdDecodeInt ciphertext plaintext length =>
                              pure (.thresholdDecodeInt ciphertext plaintext length)
                          | _ => throw (OperationalError.unsupportedNode index)
                        let operation : DirectValueScalarOperation := {
                          kind, ownerScope := facts.arena.activeScope, ownerNode := index,
                          outputPort := port, parameterEnvironment := environment }
                        let validOutput := match kind, outputType with
                          | .thresholdDecodeBool .., .boolean => true
                          | .thresholdDecodeInt .., .integer => true
                          | _, _ => false
                        if !validOutput then throw (.unsupportedOutputArity index node.outputTypes.length)
                        let (nextArena, output) ← arena.pushDirectValueScalarPointwise operation input
                        arena := nextArena
                        outputs := outputs ++ [output]
                      facts := { facts with arena }
                      pure outputs
              | .extractCoefficient position =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  let operation : DirectValueScalarOperation := {
                    kind := .extractCoefficient position
                    ownerScope := facts.arena.activeScope
                    ownerNode := index
                    outputPort := 0
                    parameterEnvironment := environment
                  }
                  let (arena, output) ← facts.arena.pushDirectValueScalarPointwise operation input
                  facts := { facts with arena }
                  pure [output]
              | .select =>
                  let indexWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  /- Selectors may be loop-indexed direct scalar values.  Recover their hull
                  from the authoritative direct reducer instead of requiring a context-free
                  representative; the subsequent direct selection retains the executable
                  selector provenance. -/
                  let selectionInput ← lookupFact index facts indexWire
                  let selection : OperationalIntegerFact ← match selectionInput with
                    | { payload := .directValue root, .. } => do
                        let (lower, upper) ← facts.arena.direct.integerInterval index indexWire root
                          (facts.arena.direct.values.size + 1)
                        pure {
                          subject := indexWire
                          origin := .local scopeKey indexWire
                          lower
                          upper
                          lowerExpression := .closed (.closedInt (.constant lower))
                          upperExpression := .closed (.closedInt (.constant upper))
                        }
                  let branchWires := node.arguments.drop 1
                  if branchWires.isEmpty || selection.lower < 0 ||
                      selection.upper >= Int.ofNat branchWires.length then
                    throw (.invalidCount index selection.upper)
                  let (arena, selectionExpression) ← executableDirectSelectExpression scopeKey index indexWire
                    selection selectionInput branchWires.length facts.arena
                  facts := { facts with arena }
                  let branches ← branchWires.mapM (lookupFact index facts)
                  match node.outputTypes with
                  | [.indexedFamily (.matrix matrixType) count]
                  | [.indexedFamily (.preimage matrixType) count] =>
                      let expectedCount ← match count.evaluate environment with
                        | some value => pure value
                        | none => throw .nonClosedExpression
                      if expectedCount <= 0 then throw (.invalidCount index expectedCount)
                      let branchLaneBinders ← branchWires.zip branches |>.mapM fun (wire, branch) =>
                        directFamilyLaneBinderAt facts.arena scopeKey scope environment wire branch
                      let (arena, output) ← selectUniformMatrixFamiliesWithLaneBinders scopeKey index selection selectionExpression
                        matrixType count expectedCount.toNat branches branchLaneBinders environment
                        facts.arena
                      facts := { facts with arena }
                      pure [output]
                  | [.matrix matrixType] | [.preimage matrixType] =>
                      let (arena, output) ← selectDirectMatrixBranches scopeKey index selection
                        { node := index, port := 0 } matrixType environment facts.arena branches.toArray
                        selectionExpression
                      facts := { facts with arena }
                      pure [output]
                  | [outputType] =>
                      let schema ← match operationalScalarWireSchema environment outputType with
                        | some schema => pure schema
                        | none => throw (.outputTypeMismatch index)
                      let (arena, output) ← selectDirectScalarBranches scopeKey index selection
                        { node := index, port := 0 } schema environment
                        facts.arena branches.toArray selectionExpression
                      facts := { facts with arena }
                      pure [output]
                  | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
              | .concat _ =>
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  let axis ← match node.kind with
                    | .concat axis => pure axis
                    | _ => throw (.unsupportedNode index)
                  let context ← directOperationIndexContext inputs.toList
                  let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let operation : PrimitiveOperation := {
                    kind := .concat axis, outputType := outputDescriptor, outputSchema := matrixType,
                    ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                    parameterEnvironment := environment }
                  let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation inputs
                  facts := { facts with arena }
                  pure [output]
              | .crtRecompose _ _ =>
                  let (arena, outputs) ← deriveOrdinaryOutputs scopeKey index node step.rule
                    environment loopDomains layouts facts 0 node.outputTypes
                  facts := { facts with arena }
                  pure outputs
              | .preimageSample .. =>
                  let targetWire ← match node.arguments[2]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let target ← lookupFact index facts targetWire
                  match target with
                  | { payload := .directValue _, .. } =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                      let (arena, inputs) ← inputs.foldlM (fun (arena, promoted) input => do
                        let (arena, input) ← arena.promoteDirectRelationOperand input
                        pure (arena, promoted.push input)) (facts.arena, #[])
                      let context ← directOperationIndexContext inputs.toList
                      let maximum? := match node.kind with
                        | .preimageSample _ maximum => IndexedParameterExpr.fromIrAt context maximum
                        | _ => IndexedParameterExpr.fromIrAt context (.constant 0)
                      let maximum ← match maximum? with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let outputDescriptor? := IndexedMatrixTypeExpr.fromIrAt context matrixType
                      let outputDescriptor ← match outputDescriptor? with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let domains? := IndexedOperationalParameterDomain.fromIrAt context loopDomains
                      let domains ← match domains? with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let operation : DirectRelationOperation := {
                        kind := .preimage maximum domains, outputType := outputDescriptor,
                        outputSchema := matrixType, ownerScope := some scopeKey,
                        ownerNode := index, outputPort := 0, parameterEnvironment := environment }
                      let (arena, output) ← arena.pushDirectRelationPointwise operation inputs
                      facts := { facts with arena }
                      pure [output]
              | .gadgetDecompose _ _ _ _ =>
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  let (arena, inputs) ← inputs.foldlM (fun (arena, promoted) input => do
                    let (arena, input) ← arena.promoteDirectRelationOperand input
                    pure (arena, promoted.push input)) (facts.arena, #[])
                  let context ← directOperationIndexContext inputs.toList
                  let declaredType? := match node.kind with
                    | .gadgetDecompose declaredType _ _ _ => IndexedMatrixTypeExpr.fromIrAt context declaredType
                    | _ => IndexedMatrixTypeExpr.fromIrAt context matrixType
                  let declaredType ← match declaredType? with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let base? := match node.kind with
                    | .gadgetDecompose _ base _ _ => IndexedParameterExpr.fromIrAt context base
                    | _ => IndexedParameterExpr.fromIrAt context (.constant 0)
                  let base ← match base? with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let digitCount? := match node.kind with
                    | .gadgetDecompose _ _ _ digitCount => IndexedParameterExpr.fromIrAt context digitCount
                    | _ => IndexedParameterExpr.fromIrAt context (.constant 0)
                  let digitCount ← match digitCount? with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let outputDescriptor? := IndexedMatrixTypeExpr.fromIrAt context matrixType
                  let outputDescriptor ← match outputDescriptor? with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let domains? := IndexedOperationalParameterDomain.fromIrAt context loopDomains
                  let domains ← match domains? with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let operation : DirectRelationOperation := {
                    kind := .decomposition declaredType base
                      (match node.kind with | .gadgetDecompose _ _ small _ => small | _ => false)
                      digitCount domains layouts, outputType := outputDescriptor,
                      outputSchema := matrixType, ownerScope := some scopeKey
                    ownerNode := index, outputPort := 0, parameterEnvironment := environment }
                  let (arena, output) ← arena.pushDirectRelationPointwise operation inputs
                  facts := { facts with arena }
                  pure [output]
              | .matrixScale _ =>
                  if node.arguments.length != 1 then
                    throw (.unsupportedOutputArity index node.arguments.length)
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  let direct := input
                      let scalar ← match node.kind with
                        | .matrixScale scalar => pure scalar
                        | _ => throw (.unsupportedNode index)
                      let context ← directOperationIndexContext [direct]
                      let scalar ← match IndexedParameterExpr.fromIrAt context scalar with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let domains ← match IndexedOperationalParameterDomain.fromIrAt context loopDomains with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let operation : PrimitiveOperation := {
                        kind := .scale scalar domains, outputType := outputDescriptor, outputSchema := matrixType,
                        ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                        parameterEnvironment := environment }
                      let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation #[direct]
                      facts := { facts with arena }
                  pure [output]
              | .liftIntegerToConstantPolynomial matrixType =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  let direct := input
                      let outputType ← match node.outputTypes with
                        | [.matrix outputType] => pure outputType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      if !operationalMatrixTypeEqual matrixType outputType then
                        throw (.outputTypeMismatch index)
                      let operation : DirectValueMatrixOperation := {
                        kind := .liftIntegerToConstantPolynomial matrixType
                        ownerScope := facts.arena.activeScope
                        ownerNode := index
                        outputPort := 0
                        parameterEnvironment := environment
                      }
                      let (arena, output) ← facts.arena.pushDirectIntegerLiftPointwise operation direct
                      facts := { facts with arena }
                  pure [output]
              | .trapdoorPublic =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  let (inputArena, direct) ← facts.arena.promoteDirectRelationOperand input
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let operation : DirectValueMatrixOperation := {
                        kind := .trapdoorPublic matrixType
                        ownerScope := facts.arena.activeScope
                        ownerNode := index
                        outputPort := 0
                        parameterEnvironment := environment
                      }
                      let (arena, output) ← inputArena.pushDirectIntegerLiftPointwise operation direct
                      facts := { facts with arena }
                  pure [output]
              | .transpose | .matrixNegate | .slice _ _ =>
                  if node.arguments.length != 1 then
                    throw (.unsupportedOutputArity index node.arguments.length)
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  let direct := input
                      let context ← directOperationIndexContext [direct]
                      let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                        | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                      let operationKind : PrimitiveOperationKind ← match node.kind with
                        | .transpose => pure (.transform .transpose)
                        | .matrixNegate => pure (.transform .negate)
                        | .slice rows columns => pure (.slice rows columns)
                        | _ => throw (.unsupportedNode index)
                      let operation : PrimitiveOperation := {
                        kind := operationKind
                        outputType := outputDescriptor
                        outputSchema := matrixType
                        ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                        parameterEnvironment := environment }
                      let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation #[direct]
                      facts := { facts with arena }
                  pure [output]
              | .matrixAdd | .matrixSubtract =>
                  if node.arguments.length != 2 then
                    throw (.unsupportedOutputArity index node.arguments.length)
                  let leftWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let rightWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index leftWire)
                  let left ← lookupFact index facts leftWire
                  let right ← lookupFact index facts rightWire
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let subtract := match node.kind with
                    | .matrixSubtract => true
                    | _ => false
                  let context ← directOperationIndexContext [left, right]
                  let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let operation : PrimitiveOperation := {
                    kind := .add subtract
                    outputType := outputDescriptor
                    outputSchema := matrixType
                    ownerScope := facts.arena.activeScope
                    ownerNode := index
                    outputPort := 0
                    parameterEnvironment := environment
                  }
                  let (arena, output) ← facts.arena.pushDirectMatrixPointwise operation left right
                  facts := { facts with arena }
                  pure [output]
              | .matrixMultiply =>
                  if node.arguments.length != 2 then
                    throw (.unsupportedOutputArity index node.arguments.length)
                  let leftWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let rightWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index leftWire)
                  let left ← lookupFact index facts leftWire
                  let right ← lookupFact index facts rightWire
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let context ← directOperationIndexContext [left, right]
                  let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let operation : PrimitiveOperation := {
                    kind := .multiply step.rule rightWire
                    outputType := outputDescriptor
                    outputSchema := matrixType
                    ownerScope := facts.arena.activeScope
                    ownerNode := index
                    outputPort := 0
                    parameterEnvironment := environment
                  }
                  let (arena, output) ← facts.arena.pushDirectMatrixPointwise operation left right
                  facts := { facts with arena }
                  pure [output]
              | .tensor =>
                  if node.arguments.length != 2 then
                    throw (.unsupportedOutputArity index node.arguments.length)
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  let context ← directOperationIndexContext inputs.toList
                  let outputDescriptor ← match IndexedMatrixTypeExpr.fromIrAt context matrixType with
                    | some value => pure value | none => throw (.unsupportedOperationalExpr index)
                  let operation : PrimitiveOperation := {
                    kind := .tensor, outputType := outputDescriptor, outputSchema := matrixType,
                    ownerScope := facts.arena.activeScope,
                    ownerNode := index, outputPort := 0, parameterEnvironment := environment }
                  let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation inputs
                  facts := { facts with arena }
                  pure [output]
              | _ =>
                  let (arena, outputs) ← deriveOrdinaryOutputs scopeKey index node step.rule
                    environment loopDomains layouts facts 0
                      node.outputTypes
                  facts := { facts with arena }
                  pure outputs
            catch error =>
              if operationalProgress "evaluate_scope" "node_error" (reprStr scopeKey)
                  index scope.nodes.size ("arguments=" ++ reprStr node.arguments ++
                    "; arena_values=" ++ toString facts.arena.direct.values.size ++
                    "; error=" ++ reprStr error) then
                throw error
              else throw error
            let mut namespacedOutputs : Array OperationalFact := #[]
            for (output, port) in outputs.toArray.zipIdx do
                  let expression := output
                  let wire : WireRef := { node := index, port }
                  let (arena, expression) ← try
                    namespaceFreshDirectOutput scopeKey wire facts.arena expression
                  catch error =>
                    let root := expression.payload.root
                    let schemaDetail : OperationalIndexedPayloadSchema → String
                      | .matrix _ => "matrix"
                      | .scalar scalar => match scalar with
                        | .boolean => "scalar_boolean"
                        | .integer => "scalar_integer"
                        | .real => "scalar_real"
                        | .bytes _ => "scalar_bytes"
                        | .trapdoor .. => "scalar_trapdoor"
                        | .typedBlob .. => "scalar_typed_blob"
                        | .unknown _ => "scalar_unknown"
                    let payloadDetail := match facts.arena.direct.valueAt? root with
                      | none => "missing"
                      | some value => match value.payload with
                        | .shared schema (.matrix reference) =>
                            "shared_matrix(schema=" ++ schemaDetail schema ++
                              "; reference=" ++ toString reference ++ ")"
                        | .shared schema (.scalar reference) =>
                            "shared_scalar(schema=" ++ schemaDetail schema ++
                              "; reference=" ++ toString reference ++ ")"
                        | .explicit .. => "explicit"
                        | .explicitValues .. => "explicit_values"
                        | .mapped .. => "mapped"
                        | .rebound .. => "rebound"
                        | .indexedOutput .. => "indexed_output"
                        | .matrixResultBound .. => "matrix_result_bound"
                        | .pointwise .. => "pointwise"
                    if operationalProgress "evaluate_scope" "namespace_error" (reprStr scopeKey)
                        index scope.nodes.size ("wire=" ++ reprStr wire ++ "; root=" ++
                          toString root ++ "; payload=" ++ payloadDetail ++
                          "; error=" ++ reprStr error) then
                      throw error
                    else throw error
                  facts := { facts with arena }
                  namespacedOutputs := namespacedOutputs.push expression
            let outputs := namespacedOutputs
            facts := { facts with values := facts.values.push outputs }
            let attachments := prepared.attachmentBuckets[index]?.getD #[]
            facts := ← try
              applyPreparedDerivationAttachments index attachments facts
            catch error =>
              let attachmentDetail := attachments.toList.map fun attachment =>
                (attachment.ownerNamespace, attachment.ruleName, attachment.roles)
              let roleRoots := attachments.toList.flatMap fun attachment =>
                attachment.roles.filterMap fun (name, wire) =>
                  match facts.values[wire.node]? >>= fun outputs => outputs[wire.port]? with
                  | some { payload := .directValue root, .. } =>
                      some (name ++ "@" ++ reprStr wire ++ " root=" ++ toString root ++
                        " carrier=" ++ directFamilyLaneCarrierTrace facts.arena.direct root
                          (facts.arena.direct.values.size + 1))
                  | none => none
              if operationalProgress "evaluate_scope" "attachment_error" (reprStr scopeKey)
                  index scope.nodes.size ("attachments=" ++ reprStr attachmentDetail ++
                    "; role_roots=" ++ reprStr roleRoots ++ "; error=" ++ reprStr error) then
                throw error
              else throw error
            if operationalProgressBlock index scope.nodes.size then
              if operationalProgress "evaluate_scope" "node_block_complete" (reprStr scopeKey)
                  (index + 1) scope.nodes.size ("outputs=" ++ toString outputs.size ++
                    "; arena_values=" ++ toString facts.arena.direct.values.size) then pure () else
                throw (.unsupportedOperationalExpr index)
            else pure ()
      if operationalProgress "evaluate_scope" "scope_complete" (reprStr scopeKey) scope.nodes.size
          scope.nodes.size ("arena_values=" ++ toString facts.arena.direct.values.size) then pure () else
        throw (.unsupportedOperationalExpr scope.nodes.size)
      pure facts
def evaluatePreparedProgramOperationalWithKey
    (programKey : ProgramInstanceKey)
    (program : PreparedOperationalProgram)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluatePreparedScope program.definitions layouts
    (.root programKey) (program.definitions.size + 1) program.root environment [] {} []

def evaluateProgramOperationalWithKey
    (programKey : ProgramInstanceKey)
    (program : Prog)
    (derivation : ProgramDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts := do
  let prepared ← prepareProgramOperational program derivation
  evaluatePreparedProgramOperationalWithKey programKey prepared environment layouts

def findInputWireType (scope : Scope) (name : String) : Option (WireRef × WireTypeExpr) :=
  scope.nodes.zipIdx.findSome? fun (node, index) =>
    if node.kind == .input name then
      node.outputTypes[0]?.map fun wireType => ({ node := index, port := 0 }, wireType)
    else none

def evaluateDeclaredBound
    (environment : ParamEnvironment) : DeclaredBoundExpr → Except OperationalError Int
  | .constant value => pure (Int.ofNat value)
  | .parameter value =>
      match value.evaluate environment with
      | some result => pure result
      | none => throw .nonClosedExpression
  | .absolute value =>
      match value.evaluate environment with
      | some result => pure (absolute result)
      | none => throw .nonClosedExpression
  | .add left right => return (← evaluateDeclaredBound environment left) +
      (← evaluateDeclaredBound environment right)
  | .multiply left right => return (← evaluateDeclaredBound environment left) *
      (← evaluateDeclaredBound environment right)
  | .maximum left right => do
      let left ← evaluateDeclaredBound environment left
      let right ← evaluateDeclaredBound environment right
      pure (max left right)
  | .minimum left right => do
      let left ← evaluateDeclaredBound environment left
      let right ← evaluateDeclaredBound environment right
      pure (min left right)
  | .floorDivide value divisor => do
      if divisor = 0 then throw .divisionByZero else
        return (← evaluateDeclaredBound environment value) / Int.ofNat divisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← match ringDimension.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let innerDimension ← match innerDimension.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      return ringDimension * innerDimension *
        (← evaluateDeclaredBound environment left) * (← evaluateDeclaredBound environment right)

def contractFact
    (arena : OperationalExprArena)
    (scopeKey : ScopeTemplateKey)
    (subject : WireRef)
    (protocolInput : ProtocolInputId)
    (wireType : WireTypeExpr)
    (contract : InputValueContract)
    (environment : ParamEnvironment) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let origin : OperationalValueOrigin := .protocolInput protocolInput
  let setMatrixOrigin (fact : OperationalMatrixFact) : OperationalMatrixFact :=
    { fact with origin := .protocolInput protocolInput }
  match contract, wireType with
  | .matrixExact contractType canonicalUpper isConstantPolynomial, .matrix wireMatrixType =>
      if contractType != wireMatrixType then throw (.inputContractMismatch "matrix")
      let cap ← match matrixCap wireMatrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters subject.node)
      let canonicalRange ← match canonicalUpper with
        | none => pure CanonicalRange.unknown
        | some upper =>
            let upper ← match upper.evaluate environment with
              | some value => pure value
              | none => throw .nonClosedExpression
            if upper <= 0 then throw (.inputContractMismatch "matrix canonical range")
            pure (.below upper.toNat)
      arena.promoteConcreteMatrixFact (setMatrixOrigin
        (← classifiedMatrixFact subject.node subject.port wireMatrixType environment cap true
          canonicalRange { isConstantPolynomial }))
  | .matrixBounded contractType bound, .matrix wireMatrixType =>
      if contractType != wireMatrixType then throw (.inputContractMismatch "matrix")
      let maximum ← evaluateDeclaredBound environment bound
      arena.promoteConcreteMatrixFact (setMatrixOrigin
        (← cappedMatrixFact subject.node subject.port wireMatrixType environment maximum))
  | .integerRange lower upper, .integer | .integerRange lower upper, .constantInt =>
      let evaluatedLower ← match lower.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let evaluatedUpper ← match upper.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if evaluatedLower > evaluatedUpper then throw (.inputContractMismatch "integer range")
      arena.promoteConcreteScalarFact (.integer {
        subject
        origin
        lower := evaluatedLower
        upper := evaluatedUpper
        lowerExpression := .closed (.closedInt (.constant evaluatedLower))
        upperExpression := .closed (.closedInt (.constant evaluatedUpper)) })
  | .boolean, .boolean | .boolean, .constantBool =>
      arena.promoteConcreteScalarFact .boolean
  | .bytes contractLength, .bytes wireLength =>
      let contractLength ← match contractLength.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let wireLength ← match wireLength.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if contractLength != wireLength then throw (.inputContractMismatch "bytes")
      arena.promoteConcreteScalarFact (.bytes { subject, origin, length := contractLength })
  | .family contractCount elementContract, .indexedFamily elementType wireCount =>
      let contractCount ← match contractCount.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let wireCount ← match wireCount.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if contractCount != wireCount || contractCount < 0 then
        throw (.inputContractMismatch "family count")
      let (arena, element) ← contractFact arena scopeKey subject protocolInput elementType
        elementContract environment
      let binder : FamilyTemplateBinder := {
        owner := scopeKey, producerNode := subject.node, binderSlot := subject.port }
      let selection := DynamicSelectionIdentity.fromOrigin (.protocolInput protocolInput)
        contractCount.toNat
      match element with
      | expression@{ payload := .directValue _, .. } => do
          if !expression.context.binders.isEmpty then
            throw (.unsupportedOperationalExpr expression.payload)
          let context ← selectionIndexedContext selection subject.node
          let root := expression.payload.root
          let value ← match arena.direct.valueAt? root with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef root)
          let (direct, root) ← match value.payload.schema with
            | .matrix _ =>
                let representative ← arena.directValueRepresentativeFactAt environment expression
                let representative := indexMatrixFact binder selection subject representative
                let (fixed, reference) := arena.direct.fixed.pushMatrix representative
                let direct := { arena.direct with fixed }
                match direct.pushShared context (.matrix representative.matrixType) reference with
                | some result => pure result
                | none => throw (.unsupportedOperationalExpr direct.values.size)
            | .scalar schema =>
                let representative ← arena.direct.scalarFactAt environment [] root
                  (arena.direct.values.size + 1)
                let representative := indexScalarFact binder selection subject representative
                let (fixed, reference) := arena.direct.fixed.pushScalar representative
                let direct := { arena.direct with fixed }
                match direct.pushShared context (.scalar schema) reference with
                | some result => pure result
                | none => throw (.unsupportedOperationalExpr direct.values.size)
          let value ← match direct.valueAt? root with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef root)
          pure ({ arena with direct }, {
            context := value.context, payload := .directValue root, storage := value.storage })
  | _, _ => throw (.inputContractMismatch "wire type")

/-- Materialize a matrix-valued external family as one indexed shared template. -/
def contractFactInArena
    (arena : OperationalExprArena)
    (scopeKey : ScopeTemplateKey)
    (subject : WireRef)
    (protocolInput : ProtocolInputId)
    (wireType : WireTypeExpr)
    (contract : InputValueContract)
    (environment : ParamEnvironment) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  contractFact arena scopeKey subject protocolInput wireType contract environment

structure OperationalStageResult where
  stage : String
  outputs : List (String × OperationalFact)
  facts : OperationalScopeFacts

structure OperationalAnalysisDiagnostics where
  expressionNodeCount : Nat := 0
  memoEvaluations : Nat := 0
  memoHits : Nat := 0
  memoMisses : Nat := 0
  peakMemoEntries : Nat := 0
  envelopeLogicalBranchCount : Nat := 0
  envelopeStoredBranchCount : Nat := 0
  /-- Direct indexed rewrite instrumentation is connected in Stage 11.  Until then this
  conservative telemetry value is zero and is never used for acceptance. -/
  relationRewriteCount : Nat := 0
  choiceJoinCount : Nat := 0
  domainComparisonCount : Nat := 0
  exactBranchVisitCount : Nat := 0
  sharedLogicalBranchVisitCount : Nat := 0
  transformCacheHits : Nat := 0
  transformCacheMisses : Nat := 0
  cartesianPairVisits : Nat := 0
  maximumPolynomialTerms : Nat := 0
  deriving BEq, DecidableEq, Repr

def operationalAnalysisDiagnostics
    (arena : OperationalExprArena)
    (environment : ParamEnvironment := [])
    (relationRewriteCount : Nat := 0)
    (directMaximumPolynomialTerms : Nat := 0) : OperationalAnalysisDiagnostics := Id.run do
  let mut logicalBranches := 0
  let mut storedBranches := 0
  let mut maximumPolynomialTerms := directMaximumPolynomialTerms
  /- The direct carrier has one physical payload graph.  Only leaf-bearing storage introduces a
  logical family/physical table count; mapped, result-bound, and pointwise nodes are references
  to those same leaves and must not be charged a second time. -/
  for value in arena.direct.values do
    match value.payload with
    | .shared _ _ =>
        let count := value.context.binders.foldl (fun total binder =>
          match binder.count.evaluate environment with
          | some value => total * value.toNat
          | none => total) 1
        logicalBranches := logicalBranches + count
        storedBranches := storedBranches + 1
    | .explicit _ _ references =>
        logicalBranches := logicalBranches + references.size
        storedBranches := storedBranches + references.size
    | .explicitValues _ _ values =>
        logicalBranches := logicalBranches + values.size
        storedBranches := storedBranches + values.size
    | .mapped .. | .rebound .. | .indexedOutput .. | .matrixResultBound .. | .pointwise .. => pure ()
  for fact in arena.direct.fixed.matrices do
    maximumPolynomialTerms := max maximumPolynomialTerms fact.polynomial.length
  return {
    expressionNodeCount := arena.direct.values.size +
      arena.direct.fixed.matrices.size + arena.direct.fixed.scalars.size
    memoEvaluations := 0
    memoHits := 0
    memoMisses := 0
    peakMemoEntries := 0
    envelopeLogicalBranchCount := logicalBranches
    envelopeStoredBranchCount := storedBranches
    relationRewriteCount
    choiceJoinCount := 0
    domainComparisonCount := 0
    exactBranchVisitCount := 0
    sharedLogicalBranchVisitCount := 0
    transformCacheHits := 0
    transformCacheMisses := 0
    cartesianPairVisits := 0
    maximumPolynomialTerms
  }

/-- A closed, generic parameter obligation derived from operational facts. Applications select
the relevant output fact, but do not implement their own arithmetic acceptance condition. -/
inductive OperationalNoiseObligation where
  | decoderThreshold
      (plaintextModulus ciphertextModulus noiseBound : Int)
  | booleanInterval (ciphertextModulus noiseBound : Int)
  deriving BEq, DecidableEq, Repr

/-- Stable, protocol-independent reasons why a closed operational report is rejected. -/
inductive OperationalNoiseRejection where
  | invalidPlaintextModulus (value : Int)
  | invalidCiphertextModulus (value : Int)
  | invalidNoiseBound (value : Int)
  | decoderThresholdNotMet
      (plaintextModulus ciphertextModulus noiseBound : Int)
  | booleanIntervalNotMet (ciphertextModulus noiseBound : Int)
  deriving BEq, DecidableEq, Repr

/-- Result consumed by parameter search. The evaluated workflow outputs are retained so callers
can inspect the facts that produced each obligation; acceptance is only the conjunction of the
listed closed obligations. Wall-clock timing belongs to the IO caller, not this pure result. -/
structure OperationalNoiseCheckReport where
  outputs : List OperationalStageResult
  obligations : List OperationalNoiseObligation
  accepted : Bool
  rejection : Option OperationalNoiseRejection
  diagnostics : OperationalAnalysisDiagnostics := {}

def checkDecoderThreshold
    (plaintextModulus ciphertextModulus noiseBound : Int) :
    Bool × Option OperationalNoiseRejection :=
  if plaintextModulus <= 1 then
    (false, some (.invalidPlaintextModulus plaintextModulus))
  else if ciphertextModulus <= 0 then
    (false, some (.invalidCiphertextModulus ciphertextModulus))
  else if noiseBound < 0 then
    (false, some (.invalidNoiseBound noiseBound))
  else if 2 * plaintextModulus * noiseBound < ciphertextModulus then
    (true, none)
  else
    (false, some (.decoderThresholdNotMet
      plaintextModulus ciphertextModulus noiseBound))

/-- Check both Boolean decoding intervals used by Diamond's executable decoder.  The lower
interval requires `N < floor((q - 2) / 4)` and the upper interval uses the non-strict endpoint
relations emitted by the decoder graph; neither is approximated by a generic `p = 2` threshold. -/
def checkBooleanInterval
    (ciphertextModulus noiseBound : Int) : Bool × Option OperationalNoiseRejection :=
  if ciphertextModulus < 4 then
    (false, some (.invalidCiphertextModulus ciphertextModulus))
  else if noiseBound < 0 then
    (false, some (.invalidNoiseBound noiseBound))
  else
    let quarter := (ciphertextModulus - 2) / 4
    if noiseBound < quarter &&
        3 * quarter + noiseBound < ciphertextModulus &&
        quarter + noiseBound <= ciphertextModulus / 2 &&
        ciphertextModulus / 2 + noiseBound <= 3 * quarter then
      (true, none)
    else
      (false, some (.booleanIntervalNotMet ciphertextModulus noiseBound))

/-- Builds the generic decoder report used by parameter search. This definition intentionally
uses multiplication rather than an integer division such as `noise < q / 4`, so boundary behavior
is exactly the stated strict inequality for every plaintext modulus. -/
def decoderNoiseCheckReport
    (outputs : List OperationalStageResult)
    (residual : OperationalMatrixFact)
    (environment : ParamEnvironment)
    (plaintextModulus ciphertextModulus : Int) :
    Except OperationalError OperationalNoiseCheckReport := do
  let _ ← residual.rejectResidualLargeTerms
  let noiseBound ← residual.evaluateNoiseHardBound environment
  let obligation := OperationalNoiseObligation.decoderThreshold
    plaintextModulus ciphertextModulus noiseBound
  let (accepted, rejection) :=
    checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
  pure { outputs, obligations := [obligation], accepted, rejection }

def collectDecoderResidualBounds
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError (List Int)
  | expression => do
      let entries ← arena.reducedDirectValueFactsAt environment expression
      let bounds ← entries.mapM fun entry => do
        let _ ← entry.fact.rejectResidualLargeTerms
        entry.fact.evaluateNoiseHardBound environment
      pure bounds

/-- Evaluate every matrix-like port produced at `node` across all workflow stages.  This helper is
used only by the external performance harness to time former hot nodes; it does not affect the
accepted bound or executable graph.  Missing node indices are skipped because stage scopes have
different sizes, while a present unsupported port still fails closed. -/
def operationalNodeNoiseBounds
    (outputs : List OperationalStageResult)
    (node : Nat)
    (environment : ParamEnvironment) : Except OperationalError (List Int) := do
  let mut result : List Int := []
  for stage in outputs do
    match stage.facts.values[node]? with
    | none => pure ()
    | some ports =>
        for fact in ports do
          let root := fact.payload.root
          let value ← match stage.facts.arena.direct.valueAt? root with
                | some value => pure value
                | none => throw (.invalidOperationalExprRef root)
              match value.payload.schema with
              | .matrix _ =>
                  let bounds ← collectDecoderResidualBounds stage.facts.arena environment fact
                  result := result ++ bounds
              | .scalar _ => pure ()
  pure result

/-- Evaluates the graph-derived structural bound for a matrix residual or residual family once.
The result is independent of the decoder threshold and can therefore be reused by compatible
parameter requests. Packed families are checked member-by-member and use their maximum bound. -/
def operationalNoiseBoundForFact
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment) :
    Except OperationalError (Int × OperationalAnalysisDiagnostics) := do
  if operationalProgress "direct_arena_reduction" "start" (reprStr arena.activeScope) 0
      arena.direct.values.size ("root=" ++ toString residual.payload.root) then pure () else
    throw (.unsupportedOperationalExpr residual.payload.root)
  let (bounds, rewriteEvents, directMaximumPolynomialTerms) ← match residual with
    | expression => do
        let (entries, rewriteEvents, maximumPolynomialTerms) ←
          arena.reducedDirectValueFactsAtWithDiagnostics environment expression
        let bounds ← entries.mapM fun entry => do
          let _ ← entry.fact.rejectResidualLargeTerms
          entry.fact.evaluateNoiseHardBound environment
        pure (bounds, rewriteEvents, maximumPolynomialTerms)
  if operationalProgress "direct_arena_reduction" "complete" (reprStr arena.activeScope)
      bounds.length bounds.length ("rewrite_events=" ++ toString rewriteEvents.length ++
        "; maximum_polynomial_terms=" ++ toString directMaximumPolynomialTerms) then pure () else
    throw (.unsupportedOperationalExpr residual.payload.root)
  if operationalProgress "symbolic_bound_evaluation" "start" (reprStr arena.activeScope) 0
      bounds.length "" then pure () else throw (.unsupportedOperationalExpr residual.payload.root)
  let noiseBound ← match bounds with
    | head :: tail => pure (tail.foldl max head)
    | [] => throw (OperationalError.invalidCount 0 0)
  if operationalProgress "symbolic_bound_evaluation" "complete" (reprStr arena.activeScope)
      bounds.length bounds.length ("noise_bound=" ++ toString noiseBound) then pure () else
    throw (.unsupportedOperationalExpr residual.payload.root)
  pure (noiseBound, operationalAnalysisDiagnostics arena environment rewriteEvents.length
    directMaximumPolynomialTerms)

/-- Applies a cheap decoder threshold to an already evaluated structural bound. -/
def decoderNoiseCheckReportFromBound
    (outputs : List OperationalStageResult)
    (noiseBound : Int)
    (diagnostics : OperationalAnalysisDiagnostics)
    (plaintextModulus ciphertextModulus : Int) : OperationalNoiseCheckReport :=
  let obligation := OperationalNoiseObligation.decoderThreshold
    plaintextModulus ciphertextModulus noiseBound
  let (accepted, rejection) :=
    checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
  {
    outputs := outputs
    obligations := [obligation]
    accepted := accepted
    rejection := rejection
    diagnostics }

/-- Builds the closed decoder obligation selected by a protocol-owned target.  The modulus is
derived from the residual operational fact; request data cannot substitute a different q. -/
def operationalTargetNoiseCheckReportFromBound
    (outputs : List OperationalStageResult)
    (target : OperationalDecoderTarget)
    (ciphertextModulus : Int)
    (noiseBound : Int)
    (diagnostics : OperationalAnalysisDiagnostics)
    (environment : ParamEnvironment) : Except OperationalError OperationalNoiseCheckReport := do
  let (obligation, accepted, rejection) ← match target.kind with
    | .thresholdDecode plaintextModulus => do
        let plaintextModulus ← match plaintextModulus.evaluate environment with
          | some value => pure value
          | none => throw .nonClosedExpression
        let (accepted, rejection) :=
          checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
        pure (.decoderThreshold plaintextModulus ciphertextModulus noiseBound, accepted, rejection)
    | .booleanInterval =>
        let (accepted, rejection) := checkBooleanInterval ciphertextModulus noiseBound
        pure (.booleanInterval ciphertextModulus noiseBound, accepted, rejection)
  pure { outputs, obligations := [obligation], accepted, rejection, diagnostics }

/-- Builds one decoder obligation from a residual. Prefer `operationalNoiseBoundForFact` followed
by `decoderNoiseCheckReportFromBound` when several threshold requests share the same residual and
numeric environment. -/
def decoderNoiseCheckReportForFact
    (outputs : List OperationalStageResult)
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment)
    (plaintextModulus ciphertextModulus : Int) :
    Except OperationalError OperationalNoiseCheckReport := do
  let (noiseBound, diagnostics) ← operationalNoiseBoundForFact arena residual environment
  pure (decoderNoiseCheckReportFromBound outputs noiseBound diagnostics
    plaintextModulus ciphertextModulus)

def collectOperationalOutputs
    (scope : Scope)
    (facts : OperationalScopeFacts) : Except OperationalError (List (String × OperationalFact)) :=
  scope.outputs.mapM fun (name, wire) => return (name, ← lookupFact scope.nodes.size facts wire)

def findStageOutput
    (results : List OperationalStageResult)
    (stage output : String) : Except OperationalError OperationalFact := do
  let result ← match results.find? fun result => result.stage == stage with
    | some result => pure result
    | none => throw (.missingStageResult stage output)
  match result.outputs.find? fun candidate => candidate.1 == output with
  | some (_, fact) => pure fact
  | none => throw (.missingStageResult stage output)

structure PreparedOperationalStageInput where
  subject : WireRef
  wireType : WireTypeExpr
  source : InputSource

structure PreparedOperationalStage where
  id : String
  program : PreparedOperationalProgram
  inputs : Array PreparedOperationalStageInput

structure PreparedOperationalWorkflow where
  stages : Array PreparedOperationalStage
  inputContract : InputContract
  operationalDecoderTargets : List OperationalDecoderTarget

/-- A strict structural traversal used only to make generated-document timing truthful.  Its
checksum recursively visits typed executable payloads without using `reprStr` or contributing to
operational acceptance. -/
structure OperationalForceStats where
  entries : Nat := 0
  checksum : Nat := 0

private def forcedIntExprChecksum : IntExpr → Nat
  | .constant value => value.natAbs + 1
  | .parameter name => name.length + 2
  | .loopIndex slot => slot + 3
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => forcedIntExprChecksum left + forcedIntExprChecksum right + 5
  | .log2Ceil value => forcedIntExprChecksum value + 7

private def forcedRealExprChecksum : RealExpr → Nat
  | .rational value => value.num.natAbs + value.den + 1
  | .parameter name => name.length + 2
  | .fromInt value => forcedIntExprChecksum value + 3
  | .add left right | .subtract left right | .multiply left right | .divide left right =>
      forcedRealExprChecksum left + forcedRealExprChecksum right + 5
  | .sqrt value => forcedRealExprChecksum value + 7

private def forcedMatrixTypeChecksum (matrixType : MatrixTypeExpr) : Nat :=
  forcedIntExprChecksum matrixType.modulus + forcedIntExprChecksum matrixType.ringDimension +
    forcedIntExprChecksum matrixType.rows + forcedIntExprChecksum matrixType.columns + 1

private def forcedWireTypeChecksum : WireTypeExpr → Nat
  | .constantInt | .constantReal | .constantBool | .integer | .real | .boolean => 1
  | .bytes length => forcedIntExprChecksum length + 2
  | .typedBlob name schemaHash => name.length + schemaHash.foldl (· + ·) 0 + 3
  | .matrix matrixType | .preimage matrixType => forcedMatrixTypeChecksum matrixType + 5
  | .trapdoor matrixType sigma base digits maximum => forcedMatrixTypeChecksum matrixType +
      forcedRealExprChecksum sigma + forcedIntExprChecksum base + forcedIntExprChecksum digits +
      forcedIntExprChecksum maximum + 7
  | .indexedFamily element count => forcedWireTypeChecksum element + forcedIntExprChecksum count + 11

private def forcedWireChecksum (wire : WireRef) : Nat := wire.node + wire.port + 1

private def forcedNodeChecksum (node : Mxx.Ir.Node) : Nat :=
  let kindChecksum := match node.kind with
    | .input name => name.length + 1
    | .constantInt value => value.natAbs + 2
    | .evaluateInt value | .bitExtract value | .extractCoefficient value | .matrixScale value =>
        forcedIntExprChecksum value + 3
    | .constantReal value => forcedRealExprChecksum value + 4
    | .constantBool value => if value then 5 else 6
    | .zeroMatrix matrixType | .identityMatrix matrixType | .liftIntegerToConstantPolynomial matrixType |
        .uniformResidueSample matrixType => forcedMatrixTypeChecksum matrixType + 7
    | .constantMatrix matrixType coefficients => forcedMatrixTypeChecksum matrixType +
        coefficients.foldl (fun sum value => sum + forcedIntExprChecksum value) 0 + 8
    | .unitRowMatrix matrixType index | .unitColumnMatrix matrixType index |
        .gadgetMatrix matrixType index | .smallGadgetMatrix matrixType index |
        .rotationMatrix matrixType index | .gadgetTrapdoor matrixType index |
        .gaussianSample matrixType index | .preimageSample matrixType index =>
        forcedMatrixTypeChecksum matrixType + forcedIntExprChecksum index + 9
    | .powerOfBaseMatrix matrixType base exponent => forcedMatrixTypeChecksum matrixType +
        forcedIntExprChecksum base + forcedIntExprChecksum exponent + 10
    | .boolToInt | .intToReal | .realSqrt | .select | .trapdoorPublic | .matrixAdd |
        .matrixSubtract | .matrixMultiply | .matrixNegate | .transpose | .tensor | .familyPack |
        .familyGetDynamic => 11
    | .intBinary _ | .realBinary _ | .intCompare _ => 12
    | .uniformIntervalSample matrixType minimum maximum => forcedMatrixTypeChecksum matrixType +
        forcedIntExprChecksum minimum + forcedIntExprChecksum maximum + 13
    | .hashSample matrixType _ tags expressions decimalExpressions u64Expressions base digits =>
        forcedMatrixTypeChecksum matrixType + tags.foldl (· + ·) 0 +
          expressions.foldl (fun sum value => sum + forcedIntExprChecksum value) 0 +
          decimalExpressions.foldl (fun sum value => sum + forcedIntExprChecksum value) 0 +
          u64Expressions.foldl (fun sum value => sum + forcedIntExprChecksum value) 0 +
          (base.map forcedIntExprChecksum).getD 0 + (digits.map forcedIntExprChecksum).getD 0 + 14
    | .gadgetDecompose matrixType base _ digits => forcedMatrixTypeChecksum matrixType +
        forcedIntExprChecksum base + forcedIntExprChecksum digits + 15
    | .trapdoorSample matrixType maximum => forcedMatrixTypeChecksum matrixType +
        forcedIntExprChecksum maximum + 16
    | .slice rows columns => (rows.map (fun bounds => forcedIntExprChecksum bounds.1 +
        forcedIntExprChecksum bounds.2)).getD 0 + (columns.map (fun bounds =>
        forcedIntExprChecksum bounds.1 + forcedIntExprChecksum bounds.2)).getD 0 + 17
    | .concat _ => 18
    | .thresholdDecodeBool ciphertext plaintext length | .thresholdDecodeInt ciphertext plaintext length =>
        forcedIntExprChecksum ciphertext + forcedIntExprChecksum plaintext + forcedIntExprChecksum length + 19
    | .crtRecompose plaintexts coefficients => plaintexts.foldl (fun sum value => sum +
        forcedIntExprChecksum value) 0 + coefficients.foldl (fun sum value => sum + forcedIntExprChecksum value) 0 + 20
    | .packPolynomialCoefficients matrixType bits => forcedMatrixTypeChecksum matrixType +
        forcedIntExprChecksum bits + 21
    | .familyGetStatic index => forcedIntExprChecksum index + 22
    | .subgraphCall name bindings => name.length + bindings.foldl (fun sum binding =>
        sum + binding.1.length + forcedIntExprChecksum binding.2) 0 + 23
    | .parallelLoop name count slot bindings modes => name.length + forcedIntExprChecksum count + slot +
        bindings.foldl (fun sum binding => sum + binding.1.length + forcedIntExprChecksum binding.2) 0 +
        modes.foldl (fun sum mode => sum + match mode with | .broadcast => 1 | .zip => 2 | .zipOffset offset => offset + 3) 0 + 24
    | .sequentialLoop name count slot bindings carried => name.length + forcedIntExprChecksum count + slot + carried +
        bindings.foldl (fun sum binding => sum + binding.1.length + forcedIntExprChecksum binding.2) 0 + 25
  kindChecksum + node.arguments.foldl (fun sum wire => sum + forcedWireChecksum wire) 0 +
    node.outputTypes.foldl (fun sum wireType => sum + forcedWireTypeChecksum wireType) 0 + node.outputCount

private def forcedDerivationStepChecksum (step : NodeDerivation) : Nat :=
  step.sourceNode + step.arguments.foldl (fun sum wire => sum + forcedWireChecksum wire) 0 +
    match step.rule with
    | .matrixMultiplyRelation wire => forcedWireChecksum wire + 3
    | .matrixMultiplyBound => 5
    | _ => 1

private def forcedAttachmentChecksum (attachment : DerivationAttachment) : Nat :=
  attachment.ownerNamespace.length + attachment.ruleName.length + attachment.roles.foldl
    (fun sum role => sum + role.1.length + forcedWireChecksum role.2) 0 + 1

private def emitOperationalForceProgress
    (kind document scope event : String) (processed total checksum : Nat) : IO Unit :=
  IO.eprintln ("operational_progress phase=" ++ kind ++ " event=" ++ event ++
    " document=" ++ document ++ " scope=" ++ scope ++ " processed=" ++ toString processed ++
    " total=" ++ toString total ++ " detail=checksum=" ++ toString checksum)

private def forceOperationalScope
    (kind document scopeName : String) (scope : Scope) : IO OperationalForceStats := do
  let mut checksum := scope.outputs.length + scope.inputNames.length
  let total := scope.nodes.size
  for (node, index) in scope.nodes.zipIdx do
    checksum := checksum + forcedNodeChecksum node
    let position := index + 1
    if operationalProgressBlock (position - 1) total then
      emitOperationalForceProgress kind document scopeName "node_block" position total checksum
  pure { entries := total, checksum }

/-- Force every decoded executable node before a raw-document decode is reported complete. -/
def forceProgOperationalDocument (document : String) (program : Prog) : IO OperationalForceStats := do
  emitOperationalForceProgress "document_decode_force" document "root" "start" 0
    (program.root.nodes.size + program.definitions.foldl (fun count entry => count + entry.2.nodes.size) 0) 0
  let root ← forceOperationalScope "document_decode_force" document "root" program.root
  let mut entries := root.entries
  let mut checksum := root.checksum
  for (name, scope) in program.definitions do
    let forced ← forceOperationalScope "document_decode_force" document name scope
    entries := entries + forced.entries
    checksum := checksum + name.length + forced.checksum
  emitOperationalForceProgress "document_decode_force" document "all_scopes" "complete" entries entries checksum
  pure { entries, checksum }

private def forceOperationalScopeDerivation
    (document scopeName : String) (derivation : ScopeDerivation) : IO OperationalForceStats := do
  let mut checksum := 0
  let total := derivation.steps.size + derivation.attachments.length
  let mut processed := 0
  for step in derivation.steps do
    checksum := checksum + forcedDerivationStepChecksum step
    processed := processed + 1
    if operationalProgressBlock (processed - 1) total then
      emitOperationalForceProgress "document_derivation_force" document scopeName "block" processed total checksum
  for attachment in derivation.attachments do
    checksum := checksum + forcedAttachmentChecksum attachment
    processed := processed + 1
    if operationalProgressBlock (processed - 1) total then
      emitOperationalForceProgress "document_derivation_force" document scopeName "block" processed total checksum
  pure { entries := processed, checksum }

/-- Force each decoded derivation payload before its document decode is reported complete. -/
def forceProgramDerivationOperationalDocument
    (document : String) (derivation : ProgramDerivation) : IO OperationalForceStats := do
  let root ← forceOperationalScopeDerivation document "root" derivation.root
  let mut entries := root.entries
  let mut checksum := root.checksum
  for (name, scope) in derivation.definitions do
    let forced ← forceOperationalScopeDerivation document name scope
    entries := entries + forced.entries
    checksum := checksum + name.length + forced.checksum
  emitOperationalForceProgress "document_derivation_force" document "all_scopes" "complete"
    entries entries checksum
  pure { entries, checksum }

/-- Force all prepared structural tables before preparation is reported complete.  This traverses
the same frozen program and derivation data as the evaluator, but does not construct facts or
change any acceptance result. -/
def forcePreparedOperationalWorkflow
    (prepared : PreparedOperationalWorkflow) : IO OperationalForceStats := do
  let mut entries := 0
  let mut checksum := prepared.inputContract.inputs.length + prepared.operationalDecoderTargets.length
  for stage in prepared.stages do
    let root ← forceOperationalScope "prepared_workflow_force" stage.id "root" stage.program.root.scope
    entries := entries + root.entries
    checksum := checksum + stage.id.length + root.checksum + stage.inputs.size
    for (name, scope) in stage.program.definitions do
      let forced ← forceOperationalScope "prepared_workflow_force" stage.id name scope.scope
      entries := entries + forced.entries
      checksum := checksum + name.length + forced.checksum
  emitOperationalForceProgress "prepared_workflow_force" "workflow" "all_stages" "complete"
    entries entries checksum
  pure { entries, checksum }

def decoderNode
    (stage : PreparedOperationalStage)
    (target : OperationalDecoderTarget) : Except OperationalError Mxx.Ir.Node :=
  match stage.program.root.scope.nodes[target.decoderNode]? with
  | some node => pure node
  | none => throw (.invalidOperationalDecoderTarget target.targetId)

def decoderInputMatrixType
    (stage : PreparedOperationalStage)
    (target : OperationalDecoderTarget) : Except OperationalError MatrixTypeExpr := do
  let node ← decoderNode stage target
  let input ← match node.arguments with
    | [input] => pure input
    | _ => throw (.invalidOperationalDecoderTarget target.targetId)
  let inputNode ← match stage.program.root.scope.nodes[input.node]? with
    | some inputNode => pure inputNode
    | none => throw (.invalidOperationalDecoderTarget target.targetId)
  match inputNode.outputTypes[input.port]? with
  | some (.matrix matrixType) => pure matrixType
  | _ => throw (.invalidOperationalDecoderTarget target.targetId)

def decoderStageForTarget
    (prepared : PreparedOperationalWorkflow)
    (target : OperationalDecoderTarget) : Except OperationalError PreparedOperationalStage :=
  match prepared.stages.toList.find? (fun stage => stage.id == target.decoderStage.name) with
  | some stage => pure stage
  | none => throw (.invalidOperationalDecoderTarget target.targetId)

def validateThresholdDecoderTarget
    (stage : PreparedOperationalStage)
    (target : OperationalDecoderTarget)
    (plaintextModulus : IntExpr)
    (residualModulus : Int)
    (environment : ParamEnvironment) : Except OperationalError Unit := do
  let decoder ← decoderNode stage target
  let ciphertextModulus ← match decoder.kind with
    | .thresholdDecodeBool ciphertextModulus decoderPlaintextModulus _ =>
        if decoderPlaintextModulus == plaintextModulus then pure ciphertextModulus
        else throw (.invalidOperationalDecoderTarget target.targetId)
    | _ => throw (.invalidOperationalDecoderTarget target.targetId)
  let inputMatrixType ← decoderInputMatrixType stage target
  if inputMatrixType.modulus != ciphertextModulus then
    throw (.invalidOperationalDecoderTarget target.targetId)
  let ciphertextModulus ← match ciphertextModulus.evaluate environment with
    | some value => pure value
    | none => throw .nonClosedExpression
  if ciphertextModulus != residualModulus then
    throw (.invalidOperationalDecoderTarget target.targetId)

def validateBooleanIntervalDecoderTarget
    (stage : PreparedOperationalStage)
    (target : OperationalDecoderTarget)
    (residualModulus : Int)
    (environment : ParamEnvironment) : Except OperationalError Unit := do
  if target.residualStage.name != target.decoderStage.name then
    throw (.invalidOperationalDecoderTarget target.targetId)
  let (residualWire, ciphertextModulus) ← match
      matchBooleanIntervalDecoder stage.program.root.scope target.decoderNode with
    | some result => pure result
    | none => throw (.invalidOperationalDecoderTarget target.targetId)
  let declaredResidual ← match stage.program.root.scope.outputs.find?
      (fun output => output.1 == target.residualOutput) with
    | some output => pure output.2
    | none => throw (.invalidOperationalDecoderTarget target.targetId)
  if residualWire != declaredResidual then
    throw (.invalidOperationalDecoderTarget target.targetId)
  let ciphertextModulus ← match ciphertextModulus.evaluate environment with
    | some value => pure value
    | none => throw .nonClosedExpression
  if ciphertextModulus != residualModulus then
    throw (.invalidOperationalDecoderTarget target.targetId)

def PreparedOperationalWorkflow.decoderTarget
    (prepared : PreparedOperationalWorkflow)
    (targetId : String) : Except OperationalError OperationalDecoderTarget := do
  let candidates := prepared.operationalDecoderTargets.filter (·.targetId == targetId)
  match candidates with
  | [target] => pure target
  | [] => throw (.unknownOperationalDecoderTarget targetId)
  | _ => throw (.duplicateOperationalDecoderTarget targetId)

def operationalFactModulus
    (arena : OperationalExprArena)
    (fact : OperationalFact)
    (environment : ParamEnvironment) : Except OperationalError Int := do
  match fact with
  | expression@{ payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      let declaredType ← match value.payload.schema with
        | .matrix matrixType => pure matrixType
        | .scalar _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })
      let physicalFacts ← arena.reducedDirectValueFactsAt environment expression
      if physicalFacts.isEmpty || !physicalFacts.all
          (fun entry => entry.fact.matrixType == declaredType) then
        throw (.operandNotMatrix 0 { node := 0, port := 0 })
      match declaredType.modulus.evaluate environment with
      | some modulus => pure modulus
      | none => throw .nonClosedExpression

/-- Resolves the closed target before selecting any residual fact.  The request supplies only the
target id; stage and output names remain bundle-owned data. -/
def operationalTargetNoiseBound
    (prepared : PreparedOperationalWorkflow)
    (outputs : List OperationalStageResult)
    (targetId : String)
    (environment : ParamEnvironment) :
    Except OperationalError
      (OperationalDecoderTarget × Int × Int × OperationalAnalysisDiagnostics) := do
  if operationalProgress "resolve_target_and_evaluate_bound" "target_lookup_start" "workflow" 0 0
      ("target_id=" ++ targetId) then pure () else throw (.unsupportedOperationalExpr 0)
  let target ← prepared.decoderTarget targetId
  let stage ← match outputs.find? (fun result => result.stage == target.residualStage.name) with
    | some result => pure result
    | none => throw (.missingStageResult target.residualStage.name target.residualOutput)
  let residual ← match stage.outputs.find? (fun output => output.1 == target.residualOutput) with
    | some output => pure output.2
    | none => throw (.missingStageResult target.residualStage.name target.residualOutput)
  let residualModulus ← operationalFactModulus stage.facts.arena residual environment
  if operationalProgress "resolve_target_and_evaluate_bound" "target_lookup_complete" "workflow" 0 0
      ("target_id=" ++ targetId ++ "; residual_stage=" ++ target.residualStage.name ++
        "; residual_output=" ++ target.residualOutput) then pure () else throw (.unsupportedOperationalExpr 0)
  let decoderStage ← decoderStageForTarget prepared target
  match target.kind with
  | .thresholdDecode plaintextModulus =>
      validateThresholdDecoderTarget decoderStage target plaintextModulus residualModulus environment
  | .booleanInterval =>
      validateBooleanIntervalDecoderTarget decoderStage target residualModulus environment
  let (noiseBound, diagnostics) ← operationalNoiseBoundForFact stage.facts.arena residual environment
  if operationalProgress "resolve_target_and_evaluate_bound" "target_bound_complete" "workflow" 0 0
      ("target_id=" ++ targetId ++ "; noise_bound=" ++ toString noiseBound) then pure () else
    throw (.unsupportedOperationalExpr 0)
  pure (target, residualModulus, noiseBound, diagnostics)

def validateOperationalDecoderTargets
    (targets : List OperationalDecoderTarget) : Except OperationalError Unit := do
  if targets.isEmpty then throw .emptyOperationalDecoderTargetRegistry
  let rec validateUnique : List OperationalDecoderTarget → Except OperationalError Unit
    | [] => pure ()
    | target :: rest => do
        if target.targetId.isEmpty || rest.any (·.targetId == target.targetId) then
          throw (.invalidOperationalDecoderTarget target.targetId)
        validateUnique rest
  validateUnique targets

example : validateOperationalDecoderTargets [] =
    .error .emptyOperationalDecoderTargetRegistry := by
  native_decide

/-- Validates every frozen stage and resolves its structural lookups exactly once. -/
def prepareWorkflowOperational
    (bundle : OperationalWorkflowSpec)
    (stageDerivations : List (String × ProgramDerivation)) :
    Except OperationalError PreparedOperationalWorkflow := do
  validateOperationalDecoderTargets bundle.operationalDecoderTargets
  let mut stages := #[]
  for stage in bundle.workflow.stages do
    let derivation ← match stageDerivations.find? fun candidate => candidate.1 == stage.id with
      | some (_, derivation) => pure derivation
      | none => throw (.missingStageDerivation stage.id)
    let program ← prepareProgramOperational stage.program derivation
    let inputs ← stage.inputs.mapM fun (inputName, source) => do
      let (subject, wireType) ← match findInputWireType stage.program.root inputName with
        | some result => pure result
        | none => throw (.missingInputNode inputName)
      pure { subject, wireType, source }
    stages := stages.push { id := stage.id, program, inputs := inputs.toArray }
  pure {
    stages
    inputContract := bundle.inputContract
    operationalDecoderTargets := bundle.operationalDecoderTargets
  }

/-- Evaluates request-dependent bounds using a workflow whose structure is already checked. -/
def evaluatePreparedWorkflowOperational
    (prepared : PreparedOperationalWorkflow)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    Except OperationalError (List OperationalStageResult) := do
  let mut results := []
  let mut arena : OperationalExprArena := {}
  for stage in prepared.stages do
    if operationalProgress "evaluate_workflow" "stage_start" stage.id results.length prepared.stages.size
        ("inputs=" ++ toString stage.inputs.size) then pure () else throw (.unsupportedOperationalExpr 0)
    let scopeKey := ScopeTemplateKey.root (.workflowStage ⟨stage.id⟩)
    let mut inputFacts : List OperationalFact := []
    for input in stage.inputs do
      let fact ← match input.source with
        | .artifact producer output => do
            if operationalProgress "artifact_input_rebinding" "start" stage.id inputFacts.length
                stage.inputs.size ("producer=" ++ producer ++ "; output=" ++ output) then pure () else
              throw (.unsupportedOperationalExpr 0)
            let output ← findStageOutput results producer output
            let (nextArena, rebound) ← rebindOperationalFact input.subject arena output environment
            arena := nextArena
            if operationalProgress "artifact_input_rebinding" "complete" stage.id inputFacts.length
                stage.inputs.size "" then pure () else throw (.unsupportedOperationalExpr 0)
            pure rebound
        | .protocol protocolName => do
            let (protocolInput, contract) ← match prepared.inputContract.inputs.find? fun entry =>
                entry.1.name == protocolName with
              | some (protocolInput, _, contract) => pure (protocolInput, contract)
              | none => throw (.missingProtocolContract protocolName)
            let (nextArena, fact) ← contractFactInArena arena scopeKey input.subject protocolInput
              input.wireType contract environment
            arena := nextArena
            pure fact
      inputFacts := inputFacts ++ [fact]
    let facts ← evaluatePreparedScope stage.program.definitions layouts scopeKey
      (stage.program.definitions.size + 1) stage.program.root environment [] arena inputFacts
    arena := facts.arena
    let outputs ← collectOperationalOutputs stage.program.root.scope facts
    results := results ++ [{ stage := stage.id, outputs, facts }]
    if operationalProgress "evaluate_workflow" "stage_complete" stage.id results.length
        prepared.stages.size ("outputs=" ++ toString outputs.length ++
          "; arena_values=" ++ toString arena.direct.values.size) then pure () else
      throw (.unsupportedOperationalExpr 0)
  pure results

/-- Evaluates the exact frozen workflow in stage order. Protocol inputs are constructed from the
reviewed input contract; artifact inputs are the producer's actual operational output facts, so
relations and identities cross a stage boundary without graph search or user annotations. -/
def evaluateWorkflowOperational
    (bundle : OperationalWorkflowSpec)
    (stageDerivations : List (String × ProgramDerivation))
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    Except OperationalError (List OperationalStageResult) := do
  let prepared ← prepareWorkflowOperational bundle stageDerivations
  evaluatePreparedWorkflowOperational prepared environment layouts

def evaluateProgramOperationalWithLayouts
    (program : Prog)
    (derivation : ProgramDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluateProgramOperationalWithKey (.standalone 0) program derivation environment layouts

def evaluateScopeOperationalWithKey
    (scopeKey : ScopeTemplateKey)
    (scope : Scope)
    (derivation : ScopeDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (inputFacts : List OperationalFact := [])
    (initialArena : OperationalExprArena := {}) : Except OperationalError OperationalScopeFacts := do
  let program : Prog := { root := scope }
  let programDerivation : ProgramDerivation := { root := derivation }
  let prepared ← prepareProgramOperational program programDerivation
  evaluatePreparedScope prepared.definitions layouts scopeKey 1 prepared.root environment [] initialArena
    inputFacts

def evaluateScopeOperationalWithLayouts
    (scope : Scope)
    (derivation : ScopeDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluateScopeOperationalWithKey (.root (.standalone 0))
    scope derivation environment layouts

/-- Future local proof target for ordinary addition.  It intentionally states the runtime
connection without presenting the operational estimate as an established theorem. -/
def MatrixAddOperationalSoundnessClaim : Prop :=
  ∀ (scope : Scope) (derivation : ScopeDerivation) (environment : ParamEnvironment),
    checkScopeDerivation scope derivation = .ok () →
      ∃ facts, evaluateScopeOperationalWithLayouts scope derivation environment [] = .ok facts


end Mxx.Certificate
