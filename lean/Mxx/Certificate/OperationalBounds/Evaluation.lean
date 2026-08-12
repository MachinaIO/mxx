import Mxx.Certificate.OperationalBounds.IndexedEngine

namespace Mxx.Certificate

open Mxx.Ir

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
      let _ ← validatePreimageCutoffAgreement nodeIndex environment loopDomains maximum
        trapdoor.publicIdentity trapdoor.preimageCutoff
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
      let hashIdentity (targetType : MatrixTypeExpr) : DeterministicHashIdentity := {
        keyOrigin
        matrixType := targetType
        parameterEnvironment := environment
        parameterDomains := loopDomains
        tagPrefix
        tagExpressions
        tagDecimalExpressions
        tagU64LeExpressions
        trailingIntegerOrigins
      }
      match variant with
      | .plain =>
          let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
          pure { result with origin := .deterministicHash (hashIdentity matrixType) }
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
          let targetOrigin := MatrixOriginIdentity.deterministicHash (hashIdentity targetType)
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
      let bound := OperationalBoundExpr.maximum (.negate input.lowerExpression) input.upperExpression
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
          let base ← fallbackMatrixFact nodeIndex outputPort matrixType environment
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
  /- Once a scalar crosses the direct carrier (notably at a parallel-loop boundary), every
  ordinary scalar primitive remains there.  This is deliberately before the legacy scalar-DAG
  endpoint below: mixing direct and legacy operands would otherwise select a representative or
  reintroduce old selection transport. -/
  if scalarOutput && arguments.all fun argument => match argument.payload with
      | .directValue _ => true
      | _ => false then
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
          | .matrix root => throw (.invalidOperationalExprRef root)
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
  /- Matrix outputs with one direct scalar operand remain in the direct carrier.  This must run
  before the concrete matrix endpoint, whose legacy trapdoor lookup cannot inspect a direct scalar
  payload. -/
  if !scalarOutput && arguments.all fun argument => match argument.payload with
      | .directValue _ => true
      | _ => false then
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
  if indexedPackOutput && arguments.all fun argument => match argument.payload with
      | .directValue _ => true
      | _ => false then
    match node.kind, arguments with
    | .packPolynomialCoefficients matrixType coefficientBits, [input] => do
        let root ← match input with
          | { payload := .directValue root, .. } => pure root
          | { payload := .matrix root, .. } =>
              throw (.unsupportedOperationalExpr root)
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
          maximum := .minimum (.closedInt (.constant cap)) boundExpr
          preimageCutoff
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
          maximum := .closedInt (.constant (absolute bound))
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

/-- Resolve the direct family lane binder from the producer's IR shape and declared family count.
The direct carrier may also contain an independent select-choice binder, which is deliberately
left in its context when a get substitutes only the family lane. -/
def directFamilyLaneBinderAt
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
      /- A broadcast direct family enters a loop body through an input node, so the child IR does
      not repeat the parent's `familyPack` producer.  This is the only input-shaped case accepted
      here: recover the already-carried lane binder only when the declared count identifies it
      uniquely.  In particular, do not turn a malformed or independently selected input into a
      single-binder family. -/
      let root ← match family.payload with
        | .directValue root => pure root
        | .matrix root => throw (.unsupportedOperationalExpr root)
      if !validateContext family.context then throw (.unsupportedOperationalExpr root)
      match family.context.binders.toList.filter (fun candidate => candidate.count == countExpression) with
      | [binder] => pure binder
      | _ => throw (.loopInputModeMismatch familyWire.node familyWire.port)
  | _ =>
      let binder ← directFamilyLaneBinder scopeKey familyWire.node producer familyWire countExpression count.toNat
      if !family.context.binders.contains binder then
        throw (.loopInputModeMismatch familyWire.node familyWire.port)
      pure binder

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
      let (arena, output) ← namespaceFreshOutput scopeKey { node := nodeIndex, port } arena output
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
      return ← addConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputType
        subtract operation.parameterEnvironment left right
  | .multiply rule rightWire =>
      let (left, right) ← binaryArguments
      return ← multiplyConcreteMatrixFacts operation.ownerNode operation.outputPort
        operation.outputType rule rightWire operation.parameterEnvironment left right
  | .tensor =>
      let (left, right) ← binaryArguments
      return ← tensorConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputType
        operation.parameterEnvironment left right
  | .concat axis =>
      return ← concatConcreteMatrixFacts operation.ownerNode operation.outputPort axis
        operation.outputType operation.parameterEnvironment arguments
  | .transform transform =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      return ← transformConcreteMatrixFact operation.ownerNode operation.outputPort
        operation.outputType transform operation.parameterEnvironment value
  | .slice rows columns =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let polynomial ← sliceOperationalPolynomial rows columns operation.outputType value.polynomial
        |>.mapError (flatErrorAt operation.ownerNode)
      polynomialMatrixFact operation.ownerNode operation.outputPort operation.outputType
        operation.parameterEnvironment polynomial value.canonicalRange
  | .scale scalar values loopDomains =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      return ← scaleConcreteMatrixFact operation.ownerNode operation.outputPort
        operation.outputType scalar values operation.parameterEnvironment loopDomains value
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

partial def foldOperationalExprConcreteFacts
    {α : Type}
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId)
    (state : α)
    (visit : α → OperationalMatrixFact → Except OperationalError α) :
    Except OperationalError α := do
  let expression ← match arena.get? root with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef root)
  match expression.node with
  | .concrete fact => visit state fact
  | .primitive operation arguments => do
      if arguments.countP (fun argument =>
          (arena.get? argument).any (·.containsSelection)) > 1 then
        throw (.unsupportedOperationalExpr root)
      let rec visitArguments
          (remaining : List OperationalExprId)
          (reverseFacts : List OperationalMatrixFact)
          (state : α) : Except OperationalError α := do
        match remaining with
        | [] => visit state (← evaluatePrimitiveConcrete operation reverseFacts.reverse.toArray)
        | argument :: tail =>
            foldOperationalExprConcreteFacts arena environment argument state fun state fact =>
              visitArguments tail (fact :: reverseFacts) state
      visitArguments arguments.toList [] state
  | .select _ (.exact branches) =>
      if branches.isEmpty then throw (.invalidCount 0 0)
      else
        let mut state := state
        for branch in branches do
          state ← foldOperationalExprConcreteFacts arena environment branch state visit
        pure state
  | .select selection (.shared representative summary) =>
      if selection.count = 0 then throw (.invalidCount 0 0)
      else do
        let summary ← arena.validatedSchema summary
        let fact ← validateSelectedMatrixSummary representative summary
        if summary.selectionOrigin.isNone then
          throw (.unsupportedOperationalExpr representative)
        visit state fact

inductive OperationalExprBoundKind where
  | total
  | noise

def OperationalExprEvaluationState.memo
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → Array (Option Int)
  | .total => state.totalMemo
  | .noise => state.noiseMemo

def OperationalExprEvaluationState.recordHit
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → OperationalExprEvaluationState
  | .total => { state with totalStats := {
      state.totalStats with memoHits := state.totalStats.memoHits + 1
    } }
  | .noise => { state with noiseStats := {
      state.noiseStats with memoHits := state.noiseStats.memoHits + 1
    } }

def OperationalExprEvaluationState.recordMiss
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → OperationalExprEvaluationState
  | .total => { state with totalStats := {
      state.totalStats with
      evaluations := state.totalStats.evaluations + 1
      memoMisses := state.totalStats.memoMisses + 1
    } }
  | .noise => { state with noiseStats := {
      state.noiseStats with
      evaluations := state.noiseStats.evaluations + 1
      memoMisses := state.noiseStats.memoMisses + 1
    } }

def OperationalExprEvaluationState.store
    (state : OperationalExprEvaluationState)
    (kind : OperationalExprBoundKind)
    (id : OperationalExprId)
    (value : Int) : OperationalExprEvaluationState :=
  match kind with
  | .total => ({ state with totalMemo := state.totalMemo.set! id (some value) }).recordMemoOccupancy
  | .noise => ({ state with noiseMemo := state.noiseMemo.set! id (some value) }).recordMemoOccupancy

def validateOperationalEnvelope
    (representative : OperationalExprId)
    (summary : SelectedMatrixSummary)
    (fact : OperationalMatrixFact) : Except OperationalError Unit := do
  let conservativeFact ← validateSelectedMatrixSummary representative summary
  if conservativeFact.matrixType != fact.matrixType || summary.selectionOrigin.isNone then
    throw (.unsupportedOperationalExpr representative)

def evaluateOperationalConcreteBound
    (kind : OperationalExprBoundKind)
    (environment : ParamEnvironment)
    (fact : OperationalMatrixFact) : Except OperationalError Int :=
  match kind with
  | .total => match fact.totalHardBound with
      | .closedInt (.constant value) => pure value
      | expression => expression.evaluateWithStates environment []
  | .noise => fact.evaluateNoiseHardBound environment

def eraseOperationalFactBounds
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let zero := OperationalBoundExpr.closedInt (.constant 0)
  { fact with
    matrixParams := { fact.matrixParams with maxCoefficientBound := 0 }
    totalHardBound := zero
    polynomial := mapOperationalPolynomial id id id (fun _ => zero) id
      (fact.polynomial.filter operationalTermIsSignal)
    metadata := {}
    identity := none
    relations := [] }

def sameOperationalSelectionShape
    (left right : OperationalMatrixFact) : Bool :=
  operationalUniformSchema (eraseOperationalFactBounds left) ==
    operationalUniformSchema (eraseOperationalFactBounds right)

def maximumBoundExpr
    (first : OperationalBoundExpr)
    (remaining : List OperationalBoundExpr) : OperationalBoundExpr :=
  remaining.foldl OperationalBoundExpr.maximum first

/-- Join complete mutually-exclusive alternatives into one relation-free fact for use by a parent
operation.  The join happens only after every branch has produced its complete polynomial.  Signal
shape is retained from the checked common schema, while the complete bounded-only remainder is
replaced by one summary whose bound is the maximum of the complete per-branch noise bounds.  This
prevents a later independent selection from creating a Cartesian traversal and, unlike taking a
maximum for each term, cannot combine correlated pieces from different branches. -/
def summarizeOperationalSelectionFacts
    (environment : ParamEnvironment)
    (facts : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let first ← match facts[0]? with
    | some first => pure first
    | none => throw (.invalidCount 0 0)
  if facts.any fun fact => fact.matrixType != first.matrixType ||
      !sameOperationalSelectionShape first fact then
    throw (.unsupportedOperationalExpr 0)
  let firstSignal := first.polynomial.filter operationalTermIsSignal
  if facts.any fun fact =>
      (!firstSignal.isEmpty &&
          fact.matrixParams.maxCoefficientBound != first.matrixParams.maxCoefficientBound) ||
        fact.polynomial.filter operationalTermIsSignal != firstSignal then
    -- Keeping branch zero would under-estimate a later bounded multiplication whenever a Large
    -- alternative has a different magnitude or identity-bearing factor. Such selections require
    -- an operation-specific exact rule rather than the relation-free representative join.
    throw (.unsupportedOperationalExpr 0)
  if facts.any matrixFactHasRelation then
    throw (.unsupportedOperationalExpr 0)
  else
    let noiseSummaries ← facts.mapM fun fact =>
      fact.noiseHardBound.mapError fun _ => .invalidMatrixParameters fact.subject.node
    let noiseBound ← match noiseSummaries[0]? with
      | some firstBound => pure (maximumBoundExpr firstBound noiseSummaries.toList.tail)
      | none => throw (.invalidCount 0 0)
    let branchMetadata := facts.map (·.metadata)
    let metadata : OperationalMatrixMetadata := {
      isConstantPolynomial := branchMetadata.all (·.isConstantPolynomial)
      knownZeroRows := match branchMetadata[0]? with
        | some value =>
            if branchMetadata.all (·.knownZeroRows == value.knownZeroRows) then
              value.knownZeroRows
            else none
        | none => none
    }
    let signal := firstSignal
    let noise := if noiseBound == .closedInt (.constant 0) then [] else
      let tokens := [.sumStart, .summaryBound noiseBound, .summaryMetadata metadata, .sumEnd]
      let summary : OperationalBoundedFactorSummary := {
        matrixType := first.matrixType
        rowCount := first.matrixParams.rows
        hardBound := noiseBound
        metadata
        provenance := tokens
      }
      let origin : OperationalCompressionOrigin := {
        kind := .boundedNoiseSum
        tokens
      }
      let factor : OperationalFactorKey := {
        leaf := .boundedSummary origin summary
        inputType := first.matrixType
        outputType := first.matrixType
        role := .bounded
        boundedSummary := some summary
      }
      [{ coefficient := 1, product := {
          factors := [factor], modes := [], outputType := first.matrixType } }]
    let output ← polynomialMatrixFact first.subject.node first.subject.port first.matrixType
      environment (signal ++ noise) first.canonicalRange
    pure { output with
      subject := first.subject
      origin := first.origin
      identity := if facts.all (·.identity == first.identity) then first.identity else none }

/-! Return a Shared value's semantic pair without evaluating, flattening, or re-joining its
representative DAG. Exact alternatives deliberately have no representative. -/
def tryUniformRepresentative
    (arena : OperationalExprArena)
    (id : OperationalExprId) :
    Except OperationalError (Option (OperationalExprId × ValidatedSchemaId)) := do
  let expression ← match arena.get? id with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef id)
  match expression.node with
  | .select selection (.shared representative schema) =>
      if selection.count = 0 then throw (.invalidCount 0 0)
      else pure (some (representative, schema))
  | .select _ (.exact _) | .concrete _ | .primitive _ _ => pure none

/-- Derive the one fact needed only to validate or transfer a uniform schema. Unlike a value
representative, this operation may close a relation-free Exact choice by summarizing all complete
branches. It is never used for relation rewriting or executable identity checks. -/
def deriveOperationalSchemaFactWithFuel
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match state.schemaFactMemo[id]? with
    | none => throw (.invalidOperationalExprRef id)
    | some (some fact) => pure (fact, { state with schemaStats := {
        state.schemaStats with memoHits := state.schemaStats.memoHits + 1 } })
    | some none => do
        let state := { state with schemaStats := {
          evaluations := state.schemaStats.evaluations + 1
          memoHits := state.schemaStats.memoHits
          memoMisses := state.schemaStats.memoMisses + 1
        }}
        let expression ← match arena.get? id with
          | some expression => pure expression
          | none => throw (.invalidOperationalExprRef id)
        let (fact, state) ← match expression.node with
          | .concrete fact => pure (fact, state)
          | .select _ (.exact branches) => do
              if branches.isEmpty then throw (.invalidCount 0 0)
              let mut state := state
              let mut facts : Array OperationalMatrixFact := #[]
              for branch in branches do
                let (fact, nextState) ← deriveOperationalSchemaFactWithFuel
                  arena environment branch state fuel
                if matrixFactHasRelation fact then
                  throw (.unsupportedOperationalExpr branch)
                facts := facts.push fact
                state := nextState
              pure (← summarizeOperationalSelectionFacts environment facts, state)
          | .select selection (.shared representative summaryId) => do
              if selection.count = 0 then throw (.invalidCount 0 0)
              let summary ← arena.validatedSchema summaryId
              pure (← validateSelectedMatrixSummary representative summary, state)
          | .primitive operation arguments =>
              match compositionalTransferRegistry (primitiveTransferClass operation) with
              | .requiresConcreteStructure => do
                  let unresolved := arguments.any fun argument =>
                    match arena.get? argument with
                    | some expression => expression.containsSelection
                    | none => true
                  if unresolved then
                    throw (.unresolvedConcreteStructure operation.ownerNode id)
                  let mut state := state
                  let mut facts : Array OperationalMatrixFact := #[]
                  for argument in arguments do
                    let (fact, nextState) ← deriveOperationalSchemaFactWithFuel
                      arena environment argument state fuel
                    facts := facts.push fact
                    state := nextState
                  pure (← evaluatePrimitiveConcrete operation facts, state)
              | .supported _ => do
                  let mut state := state
                  let mut facts : Array OperationalMatrixFact := #[]
                  for argument in arguments do
                    let (fact, nextState) ← deriveOperationalSchemaFactWithFuel
                      arena environment argument state fuel
                    facts := facts.push fact
                    state := nextState
                  pure (← evaluatePrimitiveConcrete operation facts, state)
        let schemaFactMemo := state.schemaFactMemo.set! id (some fact)
        pure (fact, ({ state with schemaFactMemo }).recordPolynomialTerms fact |>.recordMemoOccupancy)

def deriveOperationalSchemaFact
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :=
  deriveOperationalSchemaFactWithFuel arena environment id state (arena.nodes.size + 1)

/-- Recover a matrix wire through the arena's checked schema derivation.  Every ordinary matrix
wire is an empty-context indexed expression, so this preserves the producer's relations,
origins, provenance, schema, and bound rather than requiring a raw concrete wrapper. -/
def matrixFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef)
    (environment : ParamEnvironment := []) : Except OperationalError OperationalMatrixFact := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .directValue _, .. } =>
      return ← facts.arena.directValueRepresentativeFactAt environment expression
  | _ => throw (.operandNotMatrix node wire)

def OperationalExprArena.deriveOperationalSchemaFactCached
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId) :
    Except OperationalError (OperationalMatrixFact × OperationalExprArena) := do
  let evaluationState := OperationalExprEvaluationState.forEnvironment
    arena environment arena.evaluationState
  match evaluationState.schemaMemo[id]? with
  | none => throw (.invalidOperationalExprRef id)
  | some (some schemaId) => do
      let schema ← arena.validatedSchema schemaId
      let fact ← validateSelectedMatrixSummary id schema
      pure (fact, { arena with evaluationState := {
        evaluationState with schemaStats := {
          evaluationState.schemaStats with
          memoHits := evaluationState.schemaStats.memoHits + 1 }
      } })
  | some none => do
      let (fact, evaluationState) ←
        deriveOperationalSchemaFact arena environment id evaluationState
      let (arena, schemaId) ← match ← tryUniformRepresentative arena id with
        | some (_, schemaId) => pure (arena, schemaId)
        | none =>
            let summary := selectedMatrixSummary #[fact]
            if summary.uniformSchema.isNone || summary.conservativeFact.isNone then
              throw (.unsupportedOperationalExpr id)
            pure (arena.internValidatedSchema summary)
      let evaluationState := ({ evaluationState with
        schemaMemo := evaluationState.schemaMemo.set! id (some schemaId) }).recordMemoOccupancy
      pure (fact, { arena with evaluationState })

def concatOperationalExprIds
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (roots : Array OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  if roots.isEmpty then throw (.invalidCount nodeIndex 0)
  let operation : PrimitiveOperation := {
    kind := .concat axis
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) :=
    concatConcreteMatrixFacts nodeIndex outputPort axis matrixType environment arguments
  liftPrimitiveOperation operation .concat concreteTransfer deriveOperationalSchemaFact
    arena roots fuel

def concatIndexedOperationalFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (inputs : Array IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  liftIndexedOperationalFacts arena inputs fun arena roots =>
    concatOperationalExprIds nodeIndex outputPort axis matrixType environment arena roots
      (arena.nodes.size + 1)

def concatOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (inputs : Array OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let mut arena := arena
  let mut indexedInputs : Array IndexedOperationalFact := #[]
  for input in inputs do
    match input with
    | expression@{ payload := .matrix _, .. } =>
        arena ← arena.rememberIndexedExpr expression
        indexedInputs := indexedInputs.push expression
    | _ => throw (.operandNotMatrix nodeIndex { node := nodeIndex, port := outputPort })
  let (finalArena, result) ← concatIndexedOperationalFacts nodeIndex outputPort axis matrixType environment
    arena indexedInputs
  let finalArena ← finalArena.rememberIndexedExpr result
  pure (finalArena, result)

def evaluateCompleteBoundWithFuel
    (kind : OperationalExprBoundKind)
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (Int × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match (state.memo kind)[id]? with
  | none => throw (.invalidOperationalExprRef id)
  | some (some value) => pure (value, state.recordHit kind)
  | some none => do
      let expression ← match arena.get? id with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef id)
      let state := state.recordMiss kind
      let evaluateChildren
          (children : Array OperationalExprId)
          (state : OperationalExprEvaluationState) := do
        let mut state := state
        for child in children do
          let (_, nextState) ← evaluateCompleteBoundWithFuel
            kind arena environment child state fuel
          state := nextState
        pure state
      let (value, state) ← match expression.node with
        | .concrete fact => pure (← evaluateOperationalConcreteBound kind environment fact, state)
        | .primitive operation arguments => do
            match compositionalTransferRegistry (primitiveTransferClass operation) with
            | .supported .addSubtract =>
                if arguments.size != 2 then
                  throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let left ← match arguments[0]? with
                  | some value => pure value
                  | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let right ← match arguments[1]? with
                  | some value => pure value
                  | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let (leftBound, state) ← evaluateCompleteBoundWithFuel
                  kind arena environment left state fuel
                let (rightBound, state) ← evaluateCompleteBoundWithFuel
                  kind arena environment right state fuel
                pure (leftBound + rightBound, state)
            | .supported _ | .requiresConcreteStructure =>
                let state := ← evaluateChildren arguments state
                match deriveOperationalSchemaFact arena environment id state with
                | .ok (fact, state) =>
                    pure (← evaluateOperationalConcreteBound kind environment fact, state)
                | .error (.unsupportedOperationalExpr _) |
                  .error (.unresolvedConcreteStructure _ _) =>
                    throw (.unresolvedConcreteStructure operation.ownerNode id)
                | .error error => throw error
        | .select _ (.exact branches) => do
            let first ← match branches[0]? with
              | some first => pure first
              | none => throw (.invalidCount 0 0)
            let (firstBound, state) ← evaluateCompleteBoundWithFuel
              kind arena environment first state fuel
            let mut maximum := firstBound
            let mut state := state
            for branch in branches.extract 1 branches.size do
              let (bound, nextState) ← evaluateCompleteBoundWithFuel
                kind arena environment branch state fuel
              maximum := max maximum bound
              state := nextState
            pure (maximum, state)
        | .select selection (.shared representative summary) => do
            if selection.count = 0 then throw (.invalidCount 0 0)
            let summary ← arena.validatedSchema summary
            let conservativeFact ← validateSelectedMatrixSummary representative summary
            let (representativeBound, state) ← evaluateCompleteBoundWithFuel
              kind arena environment representative state fuel
            let envelopeBound ← evaluateOperationalConcreteBound
              kind environment conservativeFact
            if representativeBound > envelopeBound then
              throw (.unsupportedOperationalExpr representative)
            pure (envelopeBound, state)
      pure (value, state.store kind id value)

def evaluateCompleteBound
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (Int × OperationalExprEvaluationState) :=
  evaluateCompleteBoundWithFuel .total arena environment id state (arena.nodes.size + 1)

def OperationalExprArena.evaluateCompleteBoundCached
    (arena : OperationalExprArena)
    (kind : OperationalExprBoundKind)
    (environment : ParamEnvironment)
    (id : OperationalExprId) : Except OperationalError (Int × OperationalExprArena) := do
  let (bound, evaluationState) ← evaluateCompleteBoundWithFuel kind arena environment id
    (OperationalExprEvaluationState.forEnvironment arena environment arena.evaluationState)
    (arena.nodes.size + 1)
  pure (bound, { arena with evaluationState })

def evaluateOperationalExprNoiseBoundWithState
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (Int × OperationalExprEvaluationState) :=
  evaluateCompleteBoundWithFuel .noise arena environment id state (arena.nodes.size + 1)

def matrixMaximum
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError Int := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .directValue _, .. } =>
      let bounds ← (← facts.arena.reducedDirectValueFactsAt environment expression).mapM
        (fun entry => evaluateOperationalConcreteBound .total environment entry.fact)
      match bounds with
      | head :: tail => pure (tail.foldl max head)
      | [] => throw (.invalidCount node 0)
  | _ => throw (.operandNotMatrix node wire)

def matrixMaximumExpr
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError OperationalBoundExpr := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .directValue _, .. } =>
      let entries ← facts.arena.reducedDirectValueFactsAt environment expression
      match entries with
      | head :: tail => pure (tail.foldl (fun bound entry =>
          .maximum bound entry.fact.totalHardBound) head.fact.totalHardBound)
      | [] => throw (.invalidCount node 0)
  | _ => throw (.operandNotMatrix node wire)

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
  match fact with
  | expression@{ payload := .directValue _, .. } =>
      pure <| matrixFactHasRelation (← arena.directValueFactAt [] expression)
  | _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })

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
  | expression@{ payload := .directValue _, .. } => do
      let entries ← arena.reducedDirectValueFactsAt environment expression
      if entries.all (fun entry => matrixBoundaryPublicIdentityMatches expected entry.fact) then pure ()
      else throw (.publicIdentityMismatch node)
  | _ => throw (.operandNotMatrix node { node, port := 0 })

def sequentialFactHasRelation
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError Bool
  | expression@{ payload := .directValue _, .. } => do
      let value ← match arena.direct.valueAt? expression.payload.root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef expression.payload.root)
      match value.payload.schema with
      | OperationalIndexedPayloadSchema.matrix _ =>
          pure ((← arena.reducedDirectValueFactsAt environment expression).any
            (fun entry => matrixFactHasRelation entry.fact))
      | OperationalIndexedPayloadSchema.scalar _ => pure false
  | _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })

def summarizeSequentialFact
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError OperationalFact
  | expression@{ payload := .directValue _, .. } => do
      if ← sequentialFactHasRelation arena environment expression then
        throw (.relationBearingCarriedValue temporaryScope 0 0)
      pure expression
  | _ => throw (.unsupportedOperationalExpr arena.direct.values.size)

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
  | { payload := .matrix root, .. } =>
      throw (.unsupportedOperationalExpr root)

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
                  pure (.trapdoor { fact with maximum })
              | .integer fact =>
                  let lowerExpression := OperationalBoundExpr.recurrenceState count paths
                    initial transition (.integerLower 0 slot)
                  let upperExpression := OperationalBoundExpr.recurrenceState count paths
                    initial transition (.integerUpper 0 slot)
                  pure (.integer { fact with lowerExpression, upperExpression })
              | fact => pure fact)
      let (direct, mapped) ← match value.payload.schema with
        | OperationalIndexedPayloadSchema.scalar .integer =>
            direct.mapScalarValue environment mapped (fun fact => match fact with
              | .integer integer => do
                  let lower ← integer.lowerExpression.evaluateWithStates environment []
                  let upper ← integer.upperExpression.evaluateWithStates environment []
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
  | { payload := .matrix root, .. } =>
      throw (.unsupportedOperationalExpr root)

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
            let (arena, input) ← match input.payload with
              | .directValue _ => pure (arena, input)
              | _ => throw (.loopInputModeMismatch nodeIndex argumentIndex)
            let directLaneBinder ← match mode, input.payload with
              | .zip, .directValue _ | .zipOffset _, .directValue _ =>
                  some <$> directFamilyLaneBinderAt scopeKey scope environment wire input
              | _, _ => pure none
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
            facts := { facts with arena := {
              facts.arena with activeScope := some scopeKey, activeNode := some index
            } }
            if node.outputCount != node.outputTypes.length then
              throw (.unsupportedOutputArity index node.outputCount)
            let step ← match derivation.steps[index]? with
              | some step => pure step
              | none => throw (.derivation (.missingNode index))
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
                  pure outputs
              | .familyPack =>
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
                      let (arena, directElements) ← elements.foldlM (fun (arena, packed) element => do
                        let (arena, direct) ← match element with
                          | direct@{ payload := .directValue _, .. } => pure (arena, direct)
                          | _ => throw (.loopInputModeMismatch index 0)
                        pure (arena, packed.push direct)) (facts.arena, #[])
                      let (arena, family) ← packDirectScalarFamily scopeKey index environment
                        (match node.outputTypes with
                        | [.indexedFamily _ declaredCount] => declaredCount
                        | _ => .constant count) arena directElements
                      facts := { facts with arena }
                      pure [family]
                  | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
              | .familyGetStatic familyIndex =>
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let requested ← match familyIndex.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  match ← lookupFact index facts familyWire with
                  | family@{ payload := .directValue _, .. } =>
                      let binder ← directFamilyLaneBinderAt scopeKey scope environment familyWire family
                      if requested < 0 then throw (.invalidCount index requested)
                      let staticMap ← match closedStaticIndexMap environment family.context binder requested.toNat with
                        | some map => pure map
                        | none => throw (.loopInputModeMismatch index 0)
                      let (arena, selected) ← facts.arena.reindexDirectMatrixFact staticMap family environment
                      let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                        arena selected environment
                      facts := { facts with arena }
                      pure [rebound]
                  | _ => throw (.loopInputModeMismatch index 0)
              | .familyGetDynamic =>
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let indexWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let selectionInput ← lookupFact index facts indexWire
                  let selectionFact : OperationalIntegerFact ← match selectionInput with
                    | { payload := .directValue root, .. } => do
                        let (lower, upper) ← facts.arena.direct.integerInterval root
                          (facts.arena.direct.values.size + 1)
                        pure {
                          subject := indexWire
                          origin := .local scopeKey indexWire
                          lower
                          upper
                          lowerExpression := .closedInt (.constant lower)
                          upperExpression := .closedInt (.constant upper)
                        }
                    | _ => throw (.loopInputModeMismatch index 1)
                  let selection := selectionFact.origin
                  let family ← lookupFact index facts familyWire
                  match selectionFact.lower == selectionFact.upper, family with
                  | true, family@{ payload := .directValue _, .. } =>
                      let binder ← directFamilyLaneBinderAt scopeKey scope environment familyWire family
                      let requested := selectionFact.lower
                      if requested < 0 then throw (.invalidCount index requested)
                      let staticMap ← match closedStaticIndexMap environment family.context binder requested.toNat with
                        | some map => pure map
                        | none => throw (.loopInputModeMismatch index 0)
                      let (arena, selected) ← facts.arena.reindexDirectMatrixFact staticMap family environment
                      let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                        arena selected environment
                      facts := { facts with arena }
                      pure [rebound]
                  | false, family@{ payload := .directValue _, .. } =>
                      let binder ← directFamilyLaneBinderAt scopeKey scope environment familyWire family
                      let selectorCount := match binder.count.evaluate environment with
                        | some count => if count > 0 then count.toNat else 0
                        | none => 0
                      if selectorCount == 0 || selectionFact.lower < 0 ||
                          selectionFact.upper >= Int.ofNat selectorCount then
                        throw (.invalidCount index selectionFact.upper)
                      let selector ← match selectionInput with
                        | direct@{ context := { binders := #[_] }, payload := .directValue _, .. } => do
                            let position ← directSingleIndexBinder index direct
                            let owner : GatherLookupOwner := {
                              indices := operationalGatherIndicesWire scopeKey indexWire
                            }
                            pure (.gather owner binder.count (.variable position))
                        | { context, payload := .directValue _, .. } =>
                            if context.binders.isEmpty then do
                              let freshSelector := DynamicSelectionIdentity.fromDeclaredCount selection binder.count
                              let freshBinder ← match freshSelector.expression with
                                | .variable value => pure value
                                | _ => throw (.loopInputModeMismatch index 0)
                              pure <| match family.context.binders.toList.find? (fun candidate =>
                                  candidate.owner == freshBinder.owner && candidate.slot == freshBinder.slot) with
                                | some candidate => .variable candidate
                                | none => freshSelector.expression
                            else throw (.loopInputModeMismatch index 0)
                        | _ => do
                            let freshSelector := DynamicSelectionIdentity.fromDeclaredCount selection binder.count
                            let freshBinder ← match freshSelector.expression with
                              | .variable value => pure value
                              | _ => throw (.loopInputModeMismatch index 0)
                            pure <| match family.context.binders.toList.find? (fun candidate =>
                                candidate.owner == freshBinder.owner && candidate.slot == freshBinder.slot) with
                              | some candidate => .variable candidate
                              | none => freshSelector.expression
                      if selector.freeVariables.isEmpty then throw (.loopInputModeMismatch index 0)
                      let dynamicMap ← match dynamicIndexMap family.context binder selector with
                        | some map => pure map
                        | none => match closedDynamicIndexMap environment family.context binder selector with
                          | some map => pure map
                          | none => throw (.loopInputModeMismatch index 0)
                      let (arena, selected) ← facts.arena.reindexDirectMatrixFact dynamicMap family environment
                      let (arena, rebound) ← rebindOperationalFact { node := index, port := 0 }
                        arena selected environment
                      facts := { facts with arena }
                      pure [rebound]
                  | _, _ => throw (.loopInputModeMismatch index 0)
              | .parallelLoop _ count indexSlot bindings modes =>
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
                  let childOutputs ← scopeOutputFacts index child.scope.outputs childFacts
                  if childOutputs.length != node.outputCount then
                    throw (.childInputMismatch index node.outputCount childOutputs.length)
                  let (nextFacts, outputs) ← childOutputs.zipIdx.foldlM
                    (fun (currentFacts, accumulated) (output, port) => do
                      match output with
                      /- Parallel results are direct indexed values.  Closing installs the one
                      exact loop coordinate shared by every argument and output of this node. -/
                      | direct@{ payload := .directValue _, .. } =>
                          let root := direct.payload.root
                          let value ← match currentFacts.arena.direct.valueAt? root with
                            | some value => pure value
                            | none => throw (.invalidOperationalExprRef root)
                          let closed ← match value.payload.schema with
                            | .matrix _ =>
                                parallelLoopIndexedMatrixOutput scopeKey index indexSlot port count
                                  evaluatedCount.toNat environment currentFacts.arena direct
                            | .scalar _ =>
                                closeParallelDirectScalarOutput scopeKey index indexSlot port count
                                  environment currentFacts.arena direct
                          let (arena, family) := closed
                          pure ({ currentFacts with arena }, accumulated.push family)
                      | _ => throw (.loopInputModeMismatch index port)
                      )
                    (facts, #[])
                  facts := nextFacts
                  pure outputs.toList
              | .sequentialLoop _ count indexSlot bindings carriedCount =>
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
                  let iterationEnvironment := replaceLoopIndex environment indexSlot 0
                  let sequentialDomains := .loopIndex indexSlot evaluatedCount.toNat ::
                    loopDomains.filter fun domain => match domain with
                      | .loopIndex candidate _ => candidate != indexSlot
                      | .parameter _ _ _ _ => true
                  let boundParams ← match evaluateBindings iterationEnvironment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ← extendParameterDomains iterationEnvironment sequentialDomains bindings
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .sequentialBody scopeKey index
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child
                    (boundParams ++ iterationEnvironment) childDomains
                    facts.arena (abstractCarried ++ shiftedInvariantFacts)).mapError (.inScope childKey)
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
              | .thresholdDecodeBool ciphertextModulus plaintextModulus length |
                  .thresholdDecodeInt ciphertextModulus plaintextModulus length =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  match ← lookupFact index facts inputWire with
                  | { payload := .matrix root, .. } =>
                      let ciphertext ← evaluateIntInvariant environment loopDomains
                        ciphertextModulus
                      let plaintext ← evaluateIntInvariant environment loopDomains
                        plaintextModulus
                      let count ← evaluateIntInvariant environment loopDomains length
                      let allValid ← foldOperationalExprConcreteFacts facts.arena environment root
                        true fun valid branch => pure (valid &&
                          branch.matrixParams.rows == 1 && branch.matrixParams.columns == 1 &&
                          ciphertext == branch.matrixParams.modulus && plaintext > 1 && count > 0 &&
                          count <= Int.ofNat branch.matrixParams.ringDimension &&
                          node.outputCount == count.toNat)
                      if !allValid then throw (.invalidMatrixParameters index)
                      let mut arena := facts.arena
                      let mut outputs : List OperationalFact := []
                      for (outputType, port) in node.outputTypes.zipIdx do
                        match node.kind, outputType with
                        | .thresholdDecodeBool .., .boolean =>
                            let (nextArena, output) ← arena.promoteConcreteScalarFact .boolean
                            arena := nextArena
                            outputs := outputs ++ [output]
                        | .thresholdDecodeInt .., .integer => do
                            let (nextArena, output) ← arena.promoteConcreteScalarFact
                              (← integerFact index port 0 (plaintext - 1))
                            arena := nextArena
                            outputs := outputs ++ [output]
                        | _, _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      facts := { facts with arena }
                      pure outputs
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
                  match ← lookupFact index facts inputWire with
                  | { payload := .matrix root, .. } =>
                      let minimum ← evaluateIntMinimum environment loopDomains position
                      let maximum ← evaluateIntMaximum environment loopDomains position
                      let exclusiveUpper? ← foldOperationalExprConcreteFacts facts.arena environment
                        root none fun current branch => do
                          if minimum < 0 ||
                              maximum >= Int.ofNat branch.matrixParams.ringDimension then
                            throw (.invalidCount index maximum)
                          let branchUpper : Int := match branch.canonicalRange with
                            | .below upper => Int.ofNat upper
                            | .unknown => branch.matrixParams.modulus
                          if branchUpper <= 0 then throw (.invalidMatrixParameters index)
                          pure (some (match current with
                            | some previous => max previous branchUpper
                            | none => branchUpper))
                      let exclusiveUpper ← match exclusiveUpper? with
                        | some value => pure value
                        | none => throw (.invalidCount index 0)
                      let (arena, output) ← facts.arena.promoteConcreteScalarFact
                        (← integerFact index 0 0 (exclusiveUpper - 1))
                      facts := { facts with arena }
                      pure [output]
                  | input@{ payload := .directValue _, .. } =>
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
                  let selection ← integerFactAt index facts indexWire
                  let branchWires := node.arguments.drop 1
                  if branchWires.isEmpty || selection.lower < 0 ||
                      selection.upper >= Int.ofNat branchWires.length then
                    throw (.invalidCount index selection.upper)
                  let branches ← branchWires.mapM (lookupFact index facts)
                  match node.outputTypes with
                  | [.indexedFamily (.matrix matrixType) count]
                  | [.indexedFamily (.preimage matrixType) count] =>
                      let expectedCount ← match count.evaluate environment with
                        | some value => pure value
                        | none => throw .nonClosedExpression
                      if expectedCount <= 0 then throw (.invalidCount index expectedCount)
                      let branchLaneBinders ← branchWires.zip branches |>.mapM fun (wire, branch) =>
                        directFamilyLaneBinderAt scopeKey scope environment wire branch
                      let (arena, output) ← selectUniformMatrixFamiliesWithLaneBinders scopeKey index selection
                        matrixType count expectedCount.toNat branches branchLaneBinders environment
                        facts.arena
                      facts := { facts with arena }
                      pure [output]
                  | [.matrix matrixType] | [.preimage matrixType] =>
                      let (arena, output) ← selectDirectMatrixBranches scopeKey index selection
                        { node := index, port := 0 } matrixType environment facts.arena branches.toArray
                      facts := { facts with arena }
                      pure [output]
                  | [outputType] =>
                      let schema ← match operationalScalarWireSchema environment outputType with
                        | some schema => pure schema
                        | none => throw (.outputTypeMismatch index)
                      let (arena, output) ← selectDirectScalarBranches scopeKey index selection
                        { node := index, port := 0 } schema environment
                        facts.arena branches.toArray
                      facts := { facts with arena }
                      pure [output]
                  | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
              | .concat _ =>
                  let matrixType ← match node.outputTypes with
                    | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                    | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  if inputs.all fun input => match input.payload with
                      | .directValue _ => true
                      | _ => false then
                    let axis ← match node.kind with
                      | .concat axis => pure axis
                      | _ => throw (.unsupportedNode index)
                    let operation : PrimitiveOperation := {
                      kind := .concat axis, outputType := matrixType,
                      ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                      parameterEnvironment := environment }
                    let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation inputs
                    facts := { facts with arena }
                    pure [output]
                  else
                    throw (.unsupportedOperationalExpr index)
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
                  | { payload := .matrix root, .. } =>
                      throw (.unsupportedOperationalExpr root)
                  | { payload := .directValue _, .. } =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                      let (arena, inputs) ← inputs.foldlM (fun (arena, promoted) input => do
                        let (arena, input) ← arena.promoteDirectRelationOperand input
                        pure (arena, promoted.push input)) (facts.arena, #[])
                      let operation : DirectRelationOperation := {
                        kind := .preimage (match node.kind with | .preimageSample _ maximum => maximum | _ => .constant 0)
                          loopDomains, outputType := matrixType, ownerScope := some scopeKey,
                        ownerNode := index, outputPort := 0, parameterEnvironment := environment }
                      let (arena, output) ← arena.pushDirectRelationPointwise operation inputs
                      facts := { facts with arena }
                      pure [output]
              | .gadgetDecompose _ _ _ _ =>
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  match input with
                  | { payload := .matrix root, .. } =>
                      throw (.unsupportedOperationalExpr root)
                  | { payload := .directValue _, .. } =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                      let (arena, inputs) ← inputs.foldlM (fun (arena, promoted) input => do
                        let (arena, input) ← arena.promoteDirectRelationOperand input
                        pure (arena, promoted.push input)) (facts.arena, #[])
                      let operation : DirectRelationOperation := {
                        kind := .decomposition (match node.kind with | .gadgetDecompose declaredType _ _ _ => declaredType | _ => matrixType)
                          (match node.kind with | .gadgetDecompose _ base _ _ => base | _ => .constant 0)
                          (match node.kind with | .gadgetDecompose _ _ small _ => small | _ => false)
                          (match node.kind with | .gadgetDecompose _ _ _ digitCount => digitCount | _ => .constant 0)
                          loopDomains layouts, outputType := matrixType, ownerScope := some scopeKey
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
                  match input with
                  | direct@{ payload := .directValue _, .. } =>
                      let scalar ← match node.kind with
                        | .matrixScale scalar => pure scalar
                        | _ => throw (.unsupportedNode index)
                      let values ← evaluateIntOverLoops environment loopDomains scalar
                      let operation : PrimitiveOperation := {
                        kind := .scale scalar values loopDomains, outputType := matrixType,
                        ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                        parameterEnvironment := environment }
                      let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation #[direct]
                      facts := { facts with arena }
                      pure [output]
                  | _ =>
                      throw (.unsupportedOperationalExpr index)
              | .liftIntegerToConstantPolynomial matrixType =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  match input with
                  | direct@{ payload := .directValue _, .. } =>
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
                  | _ =>
                      throw (.unsupportedOperationalExpr index)
              | .trapdoorPublic =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  let (inputArena, direct) ← facts.arena.promoteDirectRelationOperand input
                  match direct with
                  | direct@{ payload := .directValue _, .. } =>
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
                  | _ =>
                      throw (.unsupportedOperationalExpr index)
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
                  match input with
                  | direct@{ payload := .directValue _, .. } =>
                      let operationKind : PrimitiveOperationKind ← match node.kind with
                        | .transpose => pure (.transform .transpose)
                        | .matrixNegate => pure (.transform .negate)
                        | .slice rows columns => pure (.slice rows columns)
                        | _ => throw (.unsupportedNode index)
                      let operation : PrimitiveOperation := {
                        kind := operationKind
                        outputType := matrixType,
                        ownerScope := facts.arena.activeScope, ownerNode := index, outputPort := 0,
                        parameterEnvironment := environment }
                      let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation #[direct]
                      facts := { facts with arena }
                      pure [output]
                  | _ =>
                      throw (.unsupportedOperationalExpr index)
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
                  let operation : PrimitiveOperation := {
                    kind := .add subtract
                    outputType := matrixType
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
                  let operation : PrimitiveOperation := {
                    kind := .multiply step.rule rightWire
                    outputType := matrixType
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
                  if inputs.all fun input => match input.payload with
                      | .directValue _ => true
                      | _ => false then
                    let operation : PrimitiveOperation := {
                      kind := .tensor, outputType := matrixType, ownerScope := facts.arena.activeScope,
                      ownerNode := index, outputPort := 0, parameterEnvironment := environment }
                    let (arena, output) ← facts.arena.pushDirectMatrixPointwiseN operation inputs
                    facts := { facts with arena }
                    pure [output]
                  else
                    throw (.unsupportedOperationalExpr index)
              | _ =>
                  let (arena, outputs) ← deriveOrdinaryOutputs scopeKey index node step.rule
                    environment loopDomains layouts facts 0
                      node.outputTypes
                  facts := { facts with arena }
                  pure outputs
            catch error =>
              throw error
            let mut namespacedOutputs : Array OperationalFact := #[]
            for (output, port) in outputs.toArray.zipIdx do
              match output with
              | expression@{ payload := .directValue _, .. } =>
                  let wire : WireRef := { node := index, port }
                  let (arena, expression) ← namespaceFreshDirectOutput scopeKey wire facts.arena
                    expression
                  facts := { facts with arena }
                  namespacedOutputs := namespacedOutputs.push expression
              | _ => throw (.loopInputModeMismatch index port)
            let outputs := namespacedOutputs
            facts := { facts with values := facts.values.push outputs }
            let attachments := prepared.attachmentBuckets[index]?.getD #[]
            facts := ← applyPreparedDerivationAttachments index attachments facts
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
        lowerExpression := .closedInt (.constant evaluatedLower)
        upperExpression := .closedInt (.constant evaluatedUpper) })
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
      | { payload := .matrix root, .. } => throw (.unsupportedOperationalExpr root)
  | _, _ => throw (.inputContractMismatch "wire type")

/-- Materialize a matrix-valued external family directly as one indexed shared template.  The
legacy recursive contract constructor remains only for non-matrix wire values while their indexed
value transport is migrated; it is not used for matrix-family semantics. -/
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
    (state : OperationalExprEvaluationState := arena.evaluationState)
    (relationRewriteCount : Nat := 0)
    (directMaximumPolynomialTerms : Nat := 0) : OperationalAnalysisDiagnostics := Id.run do
  let mut logicalBranches := 0
  let mut storedBranches := 0
  let mut maximumPolynomialTerms := max state.maximumPolynomialTerms directMaximumPolynomialTerms
  for expression in arena.nodes do
    match expression.node with
    | .concrete fact =>
        maximumPolynomialTerms := max maximumPolynomialTerms fact.polynomial.length
    | .select _ (.exact branches) =>
        logicalBranches := logicalBranches + branches.size
        storedBranches := storedBranches + branches.size
    | .select selection (.shared _ _) =>
        logicalBranches := logicalBranches + selection.count
        storedBranches := storedBranches + 1
    | _ => pure ()
  /- The direct carrier has one physical payload graph.  Only leaf-bearing storage introduces a
  logical family/physical table count; mapped, result-bound, and pointwise nodes are references
  to those same leaves and must not be charged a second time. -/
  for value in arena.direct.values do
    match value.payload with
    | .shared _ _ =>
        let environment := state.environment.getD ([] : ParamEnvironment)
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
    | .mapped .. | .matrixResultBound .. | .pointwise .. => pure ()
  for fact in arena.direct.fixed.matrices do
    maximumPolynomialTerms := max maximumPolynomialTerms fact.polynomial.length
  let memoEvaluations := state.totalStats.evaluations + state.noiseStats.evaluations +
    state.schemaStats.evaluations
  let memoHits := state.totalStats.memoHits + state.noiseStats.memoHits + state.schemaStats.memoHits
  let memoMisses := state.totalStats.memoMisses + state.noiseStats.memoMisses +
    state.schemaStats.memoMisses
  return {
    /- This is request-arena inventory telemetry, rather than root reachability telemetry.
    Direct payload nodes and fixed leaves are distinct producer allocations, and unrelated
    allocations intentionally remain visible. Indexed metadata only annotates existing nodes. -/
    expressionNodeCount := arena.nodes.size + arena.direct.values.size +
      arena.direct.fixed.matrices.size + arena.direct.fixed.scalars.size
    memoEvaluations
    memoHits
    memoMisses
    peakMemoEntries := state.peakMemoEntries
    envelopeLogicalBranchCount := logicalBranches
    envelopeStoredBranchCount := storedBranches
    relationRewriteCount
    choiceJoinCount := arena.choiceJoinCount
    domainComparisonCount := arena.domainComparisonCount
    exactBranchVisitCount := arena.exactBranchVisitCount
    sharedLogicalBranchVisitCount := arena.sharedLogicalBranchVisitCount
    transformCacheHits := arena.transformCacheHits
    transformCacheMisses := arena.transformCacheMisses
    cartesianPairVisits := arena.cartesianPairVisitCount
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

/-- Inspect every complete legacy selection alternative before reducing a decoder residual to a
numeric bound.  Exact selections are checked branch-by-branch; Shared selections are checked both
at their validated all-branch envelope and their stored representative.  A primitive is checked
only after its full normalized output is derived, so exact cancellation and relation consumption
can remove Large terms before this boundary. -/
partial def validateResidualExpressionNoLargeTerms
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError Unit := do
  let expression ← match arena.get? root with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef root)
  match expression.node with
  | .concrete fact => fact.rejectResidualLargeTerms
  | .primitive _ _ =>
      let (fact, _) ← deriveOperationalSchemaFact arena environment root
        (OperationalExprEvaluationState.forEnvironment arena environment arena.evaluationState)
      fact.rejectResidualLargeTerms
  | .select _ (.exact branches) =>
      if branches.isEmpty then throw (.invalidCount 0 0)
      branches.forM fun branch => validateResidualExpressionNoLargeTerms arena environment branch
  | .select selection (.shared representative summaryId) => do
      if selection.count = 0 then throw (.invalidCount 0 0)
      let summary ← arena.validatedSchema summaryId
      let envelope ← validateSelectedMatrixSummary representative summary
      let _ ← envelope.rejectResidualLargeTerms
      validateResidualExpressionNoLargeTerms arena environment representative

def evaluateOperationalExprNoiseBoundWithStats
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) :
    Except OperationalError (Int × OperationalExprEvaluationStats) := do
  let _ ← validateResidualExpressionNoLargeTerms arena environment root
  let (maximum, state) ← evaluateOperationalExprNoiseBoundWithState arena environment root
    (OperationalExprEvaluationState.forEnvironment arena environment arena.evaluationState)
  pure (maximum, state.noiseStats)

def evaluateOperationalExprNoiseBound
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError Int := do
  let (maximum, _) ← evaluateOperationalExprNoiseBoundWithStats arena environment root
  pure maximum

partial def collectDecoderResidualBounds
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalExprEvaluationState → OperationalFact →
    Except OperationalError (List Int × OperationalExprEvaluationState)
  | state, expression@{ payload := .directValue _, .. } => do
      let entries ← arena.reducedDirectValueFactsAt environment expression
      let bounds ← entries.mapM fun entry => do
        let _ ← entry.fact.rejectResidualLargeTerms
        entry.fact.evaluateNoiseHardBound environment
      pure (bounds, state)
  | _, _ => throw (.unsupportedOperationalExpr 0)

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
    let mut evaluationState := OperationalExprEvaluationState.forEnvironment
      stage.facts.arena environment stage.facts.arena.evaluationState
    match stage.facts.values[node]? with
    | none => pure ()
    | some ports =>
        for fact in ports do
          match fact with
          | { payload := .directValue root, .. } =>
              let value ← match stage.facts.arena.direct.valueAt? root with
                | some value => pure value
                | none => throw (.invalidOperationalExprRef root)
              match value.payload.schema with
              | .matrix _ =>
                  let (bounds, nextState) ← collectDecoderResidualBounds stage.facts.arena
                    environment evaluationState fact
                  result := result ++ bounds
                  evaluationState := nextState
              | .scalar _ => pure ()
          | _ => throw (.unsupportedOperationalExpr node)
  pure result

/-- Evaluates the graph-derived structural bound for a matrix residual or residual family once.
The result is independent of the decoder threshold and can therefore be reused by compatible
parameter requests. Packed families are checked member-by-member and use their maximum bound. -/
def operationalNoiseBoundForFact
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment) :
    Except OperationalError (Int × OperationalAnalysisDiagnostics) := do
  let initialState := OperationalExprEvaluationState.forEnvironment
    arena environment arena.evaluationState
  let (bounds, evaluationState, rewriteEvents, directMaximumPolynomialTerms) ← match residual with
    | expression@{ payload := .directValue _, .. } => do
        let (entries, rewriteEvents, maximumPolynomialTerms) ←
          arena.reducedDirectValueFactsAtWithDiagnostics environment expression
        let bounds ← entries.mapM fun entry => do
          let _ ← entry.fact.rejectResidualLargeTerms
          entry.fact.evaluateNoiseHardBound environment
        pure (bounds, initialState, rewriteEvents, maximumPolynomialTerms)
    | _ => do
        let (bounds, evaluationState) ←
          collectDecoderResidualBounds arena environment initialState residual
        pure (bounds, evaluationState, [], 0)
  let noiseBound ← match bounds with
    | head :: tail => pure (tail.foldl max head)
    | [] => throw (OperationalError.invalidCount 0 0)
  pure (noiseBound, operationalAnalysisDiagnostics arena evaluationState rewriteEvents.length
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
  | _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })

/-- Resolves the closed target before selecting any residual fact.  The request supplies only the
target id; stage and output names remain bundle-owned data. -/
def operationalTargetNoiseBound
    (prepared : PreparedOperationalWorkflow)
    (outputs : List OperationalStageResult)
    (targetId : String)
    (environment : ParamEnvironment) :
    Except OperationalError
      (OperationalDecoderTarget × Int × Int × OperationalAnalysisDiagnostics) := do
  let target ← prepared.decoderTarget targetId
  let stage ← match outputs.find? (fun result => result.stage == target.residualStage.name) with
    | some result => pure result
    | none => throw (.missingStageResult target.residualStage.name target.residualOutput)
  let residual ← match stage.outputs.find? (fun output => output.1 == target.residualOutput) with
    | some output => pure output.2
    | none => throw (.missingStageResult target.residualStage.name target.residualOutput)
  let residualModulus ← operationalFactModulus stage.facts.arena residual environment
  let decoderStage ← decoderStageForTarget prepared target
  match target.kind with
  | .thresholdDecode plaintextModulus =>
      validateThresholdDecoderTarget decoderStage target plaintextModulus residualModulus environment
  | .booleanInterval =>
      validateBooleanIntervalDecoderTarget decoderStage target residualModulus environment
  let (noiseBound, diagnostics) ← operationalNoiseBoundForFact stage.facts.arena residual environment
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
    let scopeKey := ScopeTemplateKey.root (.workflowStage ⟨stage.id⟩)
    let mut inputFacts : List OperationalFact := []
    for input in stage.inputs do
      let fact ← match input.source with
        | .artifact producer output => do
            let output ← findStageOutput results producer output
            let (nextArena, rebound) ← rebindOperationalFact input.subject arena output environment
            arena := nextArena
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
