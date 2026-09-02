import MxxIrCore.Structural

namespace Mxx
namespace IR

noncomputable section

inductive EvalError where
  | missingScope (stage scope : Nat)
  | missingNode (stage scope node : Nat)
  | missingPort (stage scope node port : Nat)
  | missingInput (stage scope node input : Nat)
  | missingArtifact (stage scope node : Nat)
  | missingSample (stage scope node : Nat)
  | wrongType (stage scope node : Nat)
  | wrongArity (stage scope node : Nat)
  | unsupportedPrimitive (stage scope node : Nat)
  | invalidIndex (stage scope node : Nat)
  | invalidStructuralValue (stage scope node : Nat)
  deriving Repr

def occurrenceOf (stage : Nat) (path : OccurrencePath) (wire : WireRef) : WireOccurrence :=
  { stage := stage, path := path, wire := wire }

def occurrenceValid (data : ProgramData) (occurrence : WireOccurrence) : Prop :=
  ∃ stage scope node,
    (data.stages[occurrence.stage]?) = some stage ∧
    scopeAt stage occurrence.wire.scope = some scope ∧
    nodeAt scope occurrence.wire.node = some node ∧
    occurrence.wire.port < node.outputs.size

def samplerPayload : NodePayload → Bool
  | .uniformResidueSample _ | .uniformIntervalSample _ _ | .gaussianSample _ _ _ |
      .hashSample _ _ _ _ _ | .trapdoorSample _ _ _ _ _ | .preimageSample _ _ |
      .familyPreimageSample _ _ | .gadgetTrapdoor _ _ => true
  | _ => false

/- These payloads currently reach the primitive fallback but always return an unsupported error.
   Keeping them separate from PrimitiveNodePayload prevents generated execution bridges from
   claiming that a successful evaluator step exists. -/
inductive UnsupportedPrimitivePayload : NodePayload → Prop where
  | constantReal (value : RealExpr) : UnsupportedPrimitivePayload (.constantReal value)
  | matrixMulAccumulate (coefficients : Array StructuralIntExpr) (hasBias : Bool) :
      UnsupportedPrimitivePayload (.matrixMulAccumulate coefficients hasBias)
  | constantSmallGadgetMatrix (matrixType : MatrixType) (base : StructuralIntExpr) :
      UnsupportedPrimitivePayload (.constantMatrix matrixType (.gadget base true))
  | smallGadgetDecompose (base digits : StructuralIntExpr) :
      UnsupportedPrimitivePayload (.gadgetDecompose base true digits)
  | tensor : UnsupportedPrimitivePayload .tensor
  | preimageBinary (op : PreimageBinaryOp) :
      UnsupportedPrimitivePayload (.preimageBinary op)
  | preimageConcatColumns : UnsupportedPrimitivePayload .preimageConcatColumns
  | decompositionEntry (row column : StructuralIntExpr) :
      UnsupportedPrimitivePayload (.decompositionEntry row column)
  | liftIntegerToConstantPolynomial (matrixType : MatrixType) :
      UnsupportedPrimitivePayload (.liftIntegerToConstantPolynomial matrixType)
  | thresholdDecode (plaintextModulus length : StructuralIntExpr) (outputBool : Bool) :
      UnsupportedPrimitivePayload (.thresholdDecode plaintextModulus length outputBool)
  | crtRecompose (plaintextModuli reconstructionCoefficients : Array StructuralIntExpr) :
      UnsupportedPrimitivePayload (.crtRecompose plaintextModuli reconstructionCoefficients)
  | packPolynomialCoefficients (matrixType : MatrixType)
      (coefficientBits : StructuralIntExpr) :
      UnsupportedPrimitivePayload (.packPolynomialCoefficients matrixType coefficientBits)

structure SampleRef (data : ProgramData) where
  occurrence : WireOccurrence
  payload : NodePayload
  outputType : WireType
  programValid : data.Valid
  occurrenceValid : Mxx.IR.occurrenceValid data occurrence
  storedPayload : ∃ stage scope node,
    data.stages[occurrence.stage]? = some stage ∧
    scopeAt stage occurrence.wire.scope = some scope ∧
    nodeAt scope occurrence.wire.node = some node ∧ node.payload = payload
  storedOutput : ∃ stage scope node,
    data.stages[occurrence.stage]? = some stage ∧
    scopeAt stage occurrence.wire.scope = some scope ∧
    nodeAt scope occurrence.wire.node = some node ∧
    node.outputs[occurrence.wire.port]? = some outputType
  isSampler : samplerPayload payload = true

noncomputable instance {data : ProgramData} : DecidableEq (SampleRef data) := Classical.decEq _

structure EvalEnv (backend : SemanticBackend) (data : ProgramData) where
  programValid : data.Valid
  externalInput : WireOccurrence → Except EvalError (DynamicValue backend)
  sampleOutput : SampleRef data → Except EvalError (DynamicValue backend)

def valueTypeMatches {backend : SemanticBackend} (value : DynamicValue backend)
    (expected : WireType) : Prop :=
  value.1 = expected

def allTypesMatch {backend : SemanticBackend} (values : Array (DynamicValue backend))
    (expected : WireType) : Prop :=
  ∀ value ∈ values, valueTypeMatches value expected

def integerValue? {backend : SemanticBackend} (value : DynamicValue backend) : Option Int :=
  match value with
  | ⟨.constantInt, integer⟩ | ⟨.int, integer⟩ => some integer
  | _ => none

def booleanValue? {backend : SemanticBackend} (value : DynamicValue backend) : Option Bool :=
  match value with
  | ⟨.constantBool, boolean⟩ | ⟨.bool, boolean⟩ => some boolean
  | _ => none

def expectTwoInt {backend : SemanticBackend} (stage scope node : Nat)
    (arguments : Array (DynamicValue backend)) :
    Except EvalError (Int × Int) :=
  match arguments.toList with
  | [left, right] => match integerValue? left, integerValue? right with
      | some left, some right => pure (left, right)
      | _, _ => throw (.wrongType stage scope node)
  | _ => throw (.wrongArity stage scope node)

def expectTwoReal {backend : SemanticBackend} (stage scope node : Nat)
    (arguments : Array (DynamicValue backend)) :
    Except EvalError (Real × Real) :=
  match arguments.toList with
  | [⟨.real, left⟩, ⟨.real, right⟩] => pure (left, right)
  | _ => throw (.wrongArity stage scope node)

def evalIntBinary (op : IntBinaryOp) (left right : Int) : Int :=
  match op with
  | .add => left + right
  | .subtract => left - right
  | .multiply => left * right
  | .divide => left / right
  | .remainder => left % right

def evalIntCompare (op : IntCompareOp) (left right : Int) : Bool :=
  match op with
  | .equal => left = right
  | .less => left < right
  | .lessEqual => left ≤ right

def evalRealBinary (op : RealBinaryOp) (left right : Real) : Real :=
  match op with
  | .add => left + right
  | .subtract => left - right
  | .multiply => left * right
  | .divide => left / right

def evalNatExpr (structural : StructuralEnv) (stage scope node : Nat)
    (expression : StructuralIntExpr) : Except EvalError Nat := do
  let value ← expression.eval structural |>.mapError (fun _ => EvalError.invalidStructuralValue stage scope node)
  if 0 ≤ value then pure value.toNat else throw (.invalidStructuralValue stage scope node)

def evalRange (structural : StructuralEnv) (stage scope node : Nat)
    (range : Option IntRange) (default : Nat) : Except EvalError (Nat × Nat) :=
  match range with
  | none => pure (0, default)
  | some range => do
      let start ← evalNatExpr structural stage scope node range.start
      let stop ← evalNatExpr structural stage scope node range.stop
      if start ≤ stop then pure (start, stop) else throw (.invalidIndex stage scope node)

/-! The gadget branch is isolated so successful execution can be inverted at the exact backend
    call.  The layout and canonical gadget are derived from the evaluator operands. -/
def evalGadgetDecompose (backend : SemanticBackend) (structural : StructuralEnv)
    (stage scope node : Nat) (baseExpr : StructuralIntExpr) (small : Bool)
    (digitsExpr : StructuralIntExpr) (targetType outputType : MatrixType)
    (target : backend.denoteMatrix targetType) :
    Except EvalError (backend.denotePreimage outputType) :=
  if small then throw (.unsupportedPrimitive stage scope node)
  else do
    let baseInteger ← baseExpr.eval structural |>.mapError
      (fun _ => EvalError.invalidStructuralValue stage scope node)
    let baseValue ← if 0 ≤ baseInteger then pure baseInteger.toNat
      else throw (.invalidStructuralValue stage scope node)
    let digitValue ← evalNatExpr structural stage scope node digitsExpr
    if hValid : 1 < baseValue ∧ 0 < digitValue ∧ targetType.modulus = outputType.modulus ∧
        targetType.ringDimension = outputType.ringDimension ∧
        outputType.rows = targetType.rows * digitValue ∧ outputType.columns = targetType.columns then
      let layout : GadgetLayout := {
        mode := .regular
        base := baseValue
        digits := digitValue
        sourceRows := targetType.rows
        targetRows := outputType.rows
        sourceColumns := targetType.columns
        targetColumns := outputType.columns }
      let gadget := backend.matrixConstant (gadgetMatrixType targetType layout)
        (.gadget (.literal (Int.ofNat baseValue)) false) structural
      if hCapacity : targetType.modulus.toNat ≤ baseValue ^ digitValue then
        let sigma ← backend.gadgetDecompose targetType layout structural gadget target
          |>.mapError (fun _ => EvalError.invalidStructuralValue stage scope node)
        if hOutput : outputType = gadgetPreimageType targetType layout then
          pure (hOutput ▸ sigma.1)
        else throw (.wrongType stage scope node)
      else throw (.invalidStructuralValue stage scope node)
    else throw (.invalidStructuralValue stage scope node)

theorem evalGadgetDecompose_small_ne_ok (backend : SemanticBackend)
    (structural : StructuralEnv) (stage scope node : Nat) (baseExpr digitsExpr : StructuralIntExpr)
    (targetType outputType : MatrixType) (target : backend.denoteMatrix targetType)
    (result : backend.denotePreimage outputType) :
    evalGadgetDecompose backend structural stage scope node baseExpr true digitsExpr targetType
      outputType target ≠ .ok result := by
  simp [evalGadgetDecompose]

def primitive (backend : SemanticBackend) (structural : StructuralEnv) (stage scope node : Nat)
    (payload : NodePayload)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType) :
    Except EvalError (Array (DynamicValue backend)) :=
  match payload with
  | .constantInt value => pure #[⟨.constantInt, value⟩]
  | .evaluateInt value => do
      let evaluated ← value.eval structural |>.mapError (fun _ => EvalError.invalidStructuralValue stage scope node)
      pure #[⟨.constantInt, evaluated⟩]
  | .constantBool value => pure #[⟨.constantBool, value⟩]
  | .constantMatrix matrixType literal =>
      match literal with
      | .gadget _ true => throw (.unsupportedPrimitive stage scope node)
      | _ => pure #[⟨.matrix matrixType, backend.matrixConstant matrixType literal structural⟩]
  | .constantReal _ => throw (.unsupportedPrimitive stage scope node)
  | .intBinary op => do
      let (left, right) ← expectTwoInt stage scope node arguments
      pure #[⟨.int, evalIntBinary op left right⟩]
  | .intCompare op => do
      let (left, right) ← expectTwoInt stage scope node arguments
      pure #[⟨.bool, evalIntCompare op left right⟩]
  | .bitExtract bit =>
      match arguments.toList with
      | [value] => match integerValue? value with
        | some value => do
            let bit ← evalNatExpr structural stage scope node bit
            pure #[⟨.bool, backend.bitExtract value bit⟩]
        | none => throw (.wrongType stage scope node)
      | _ => throw (.wrongArity stage scope node)
  | .realBinary op => do
      let (left, right) ← expectTwoReal stage scope node arguments
      pure #[⟨.real, evalRealBinary op left right⟩]
  | .boolToInt =>
      match arguments.toList with
      | [value] => match booleanValue? value with
        | some bit => pure #[⟨.int, if bit then (1 : Int) else (0 : Int)⟩]
        | none => throw (.wrongType stage scope node)
      | _ => throw (.wrongArity stage scope node)
  | .intToReal =>
      match arguments.toList with
      | [⟨.int, value⟩] =>
          let integer : Int := value
          let realValue : Real := Int.cast integer
          pure #[⟨.real, realValue⟩]
      | _ => throw (.wrongArity stage scope node)
  | .realSqrt =>
      match arguments.toList with
      | [⟨.real, value⟩] => pure #[⟨.real, Real.sqrt value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .matrixNegate =>
      match arguments.toList with
      | [⟨.matrix matrixType, value⟩] => pure #[⟨.matrix matrixType, backend.matrixNegate matrixType value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .matrixScale scalar =>
      match arguments.toList with
      | [⟨.matrix matrixType, value⟩] => do
          let evaluated ← scalar.eval structural |>.mapError (fun _ => EvalError.invalidStructuralValue stage scope node)
          pure #[⟨.matrix matrixType, backend.matrixScale matrixType evaluated value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .transpose =>
      match arguments.toList, outputs.toList with
      | [⟨.matrix inputType, value⟩], [.matrix outputType] =>
          pure #[⟨.matrix outputType, backend.matrixTranspose inputType outputType value⟩]
      | _, _ => throw (.wrongArity stage scope node)
  | .matrixBinary op =>
      match arguments.toList, outputs.toList with
      | [⟨.matrix leftType, left⟩, ⟨.matrix rightType, right⟩], [.matrix outputType] =>
          match op with
          | .multiply =>
              pure #[⟨.matrix outputType,
                backend.matrixMultiply leftType rightType outputType left right⟩]
          | .add | .subtract =>
              if hRight : rightType = leftType then
                if hOutput : leftType = outputType then
                  let right := hRight ▸ right
                  let value := match op with
                    | .add => backend.matrixAdd leftType left right
                    | .subtract => backend.matrixSubtract leftType left right
                    | .multiply => backend.matrixAdd leftType left right
                  pure #[⟨.matrix outputType, hOutput ▸ value⟩]
                else throw (.wrongType stage scope node)
              else throw (.wrongType stage scope node)
      | _, _ => throw (.wrongArity stage scope node)
  | .slice rows columns =>
      match arguments.toList, outputs.toList with
      | [⟨.matrix inputType, value⟩], [.matrix outputType] => do
          let (rowStart, rowStop) ← evalRange structural stage scope node rows inputType.rows
          let (columnStart, columnStop) ← evalRange structural stage scope node columns inputType.columns
          pure #[⟨.matrix outputType,
            backend.matrixSlice inputType outputType rowStart rowStop columnStart columnStop value⟩]
      | _, _ => throw (.wrongArity stage scope node)
  | .concat axis =>
      match outputs.toList with
      | [.matrix outputType] => do
          let matrices ← arguments.mapM (fun argument => match argument with
            | ⟨.matrix matrixType, value⟩ => pure ⟨matrixType, value⟩
            | _ => throw (.wrongType stage scope node))
          if matrices.size = 0 then throw (.wrongArity stage scope node)
          else pure #[⟨.matrix outputType, backend.matrixConcat outputType axis matrices⟩]
      | _ => throw (.wrongArity stage scope node)
  | .gadgetDecompose _ true _ => throw (.unsupportedPrimitive stage scope node)
  | .gadgetDecompose base false digits =>
      match arguments.toList, outputs.toList with
      | [⟨.matrix inputType, value⟩], [.preimage outputType] => do
          let result ← evalGadgetDecompose backend structural stage scope node base false digits
            inputType outputType value
          pure #[⟨.preimage outputType, result⟩]
      | _, _ => throw (.wrongArity stage scope node)
  | .extractCoefficient position _ =>
      match arguments.toList with
      | [⟨.matrix matrixType, value⟩] => do
          let position ← evalNatExpr structural stage scope node position
          pure #[⟨.int, backend.extractCoefficient matrixType position value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .gadgetTrapdoor matrixType _ =>
      throw (.unsupportedPrimitive stage scope node)
  | .trapdoorPublic =>
      match arguments.toList with
      | [⟨.trapdoor trapdoorType, value⟩] =>
          pure #[⟨.matrix trapdoorType.matrix, backend.trapdoorPublic trapdoorType value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .materializePreimageExact =>
      match arguments.toList with
      | [⟨.preimage matrixType, value⟩] =>
          pure #[⟨.matrix matrixType, backend.materializePreimage matrixType value⟩]
      | _ => throw (.wrongArity stage scope node)
  | .applyPreimage =>
      match arguments.toList, outputs.toList with
      | [⟨.matrix leftType, left⟩, ⟨.preimage rightType, right⟩], [.matrix outputType] =>
          pure #[⟨.matrix outputType,
            backend.applyPreimage leftType rightType outputType left right⟩]
      | _, _ => throw (.wrongArity stage scope node)
  | .matrixMulAccumulate _ _ => throw (.unsupportedPrimitive stage scope node)
  | .uniformResidueSample _ | .uniformIntervalSample _ _ | .gaussianSample _ _ _ |
      .hashSample _ _ _ _ _ | .trapdoorSample _ _ _ _ _ | .preimageSample _ _ |
      .familyPreimageSample _ _ => throw (.missingSample stage scope node)
  | _ => throw (.unsupportedPrimitive stage scope node)

def outputTypesMatch {backend : SemanticBackend} : List WireType → List (DynamicValue backend) → Bool
  | [], [] => true
  | expected :: expectedRest, actual :: actualRest =>
      actual.1 == expected && outputTypesMatch expectedRest actualRest
  | _, _ => false

structure ScopeResult (backend : SemanticBackend) where
  values : Array (Binding backend)
  scopes : Array (ScopeTrace backend)

structure NodeResult (backend : SemanticBackend) where
  values : Array (DynamicValue backend)
  scopes : Array (ScopeTrace backend)

def NodeResult.ofValues {backend : SemanticBackend} (values : Array (DynamicValue backend)) :
    NodeResult backend :=
  { values := values, scopes := #[] }

def resolveArguments {backend : SemanticBackend} (stage scope node : Nat)
    (values : Array (Binding backend))
    (arguments : Array WireRef) : Except EvalError (Array (DynamicValue backend)) :=
  arguments.mapM (fun wire => match lookup values wire with
    | some value => pure value
    | none => throw (.missingPort stage scope node wire.port))

def childInputs {backend : SemanticBackend} (child : Scope) (arguments : Array (DynamicValue backend)) :
    Array (Binding backend) :=
  child.inputs.zip arguments |>.map (fun pair => { wire := pair.1, value := pair.2 })

def checkedChildInputs {backend : SemanticBackend} (stage scope node : Nat) (child : Scope)
    (arguments : Array (DynamicValue backend)) : Except EvalError (Array (Binding backend)) :=
  if child.inputs.size = arguments.size then pure (childInputs child arguments)
  else throw (.wrongArity stage scope node)

def envInput {backend : SemanticBackend} {data : ProgramData} (env : EvalEnv backend data)
    (stage scope node : Nat) (path : OccurrencePath)
    (wire : WireRef) : Except EvalError (DynamicValue backend) :=
  env.externalInput (occurrenceOf stage path wire)

def envSample {backend : SemanticBackend} {data : ProgramData} (env : EvalEnv backend data)
    (stageNumber : Nat) (stage : Stage) (scope : Scope) (nodeIndex : Nat) (node : Node)
    (path : OccurrencePath) (wire : WireRef) (outputType : WireType)
    (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage wire.scope = some scope)
    (nodeStored : nodeAt scope wire.node = some node)
    (portStored : wire.port < node.outputs.size)
    (outputStored : node.outputs[wire.port]? = some outputType)
    (isSampler : samplerPayload node.payload = true) :
    Except EvalError (DynamicValue backend) :=
  env.sampleOutput {
    occurrence := occurrenceOf stageNumber path wire
    payload := node.payload
    outputType := outputType
    programValid := env.programValid
    occurrenceValid := ⟨stage, scope, node, stageStored, scopeStored, nodeStored, portStored⟩
    storedPayload := ⟨stage, scope, node, stageStored, scopeStored, nodeStored, rfl⟩
    storedOutput := ⟨stage, scope, node, stageStored, scopeStored, nodeStored, outputStored⟩
    isSampler := isSampler }

def coerceValue {backend : SemanticBackend} (element : WireType)
    (value : DynamicValue backend) : Option (Value backend element) :=
  if h : value.1 = element then some (h ▸ value.2) else none

def coerceValues {backend : SemanticBackend} (element : WireType)
    (values : Array (DynamicValue backend)) : Option (Array (Value backend element)) :=
  values.mapM (coerceValue element)

def packFamily {backend : SemanticBackend} (stage scope node : Nat) (shape : List Nat)
    (element : WireType) (values : Array (DynamicValue backend)) :
    Except EvalError (Array (DynamicValue backend)) :=
  if values.size = Family.shapeProduct shape then
    match coerceValues element values with
    | none => throw (.wrongType stage scope node)
    | some typedValues =>
        match Family.pack shape typedValues with
        | some family => pure #[⟨.family shape element, family⟩]
        | none => throw (.wrongType stage scope node)
  else throw (.wrongArity stage scope node)

def familyIndexFromList : (shape indices : List Nat) → Option (FamilyIndex shape)
  | [], [] => some ()
  | extent :: shape, index :: indices =>
      if h : index < extent then
        match familyIndexFromList shape indices with
        | some tail => some (⟨index, h⟩, tail)
        | none => none
      else none
  | _, _ => none

def familyIndexFromArray (shape indices : Array Nat) : Option (FamilyIndex shape.toList) :=
  familyIndexFromList shape.toList indices.toList

def evalIndexExpr (structural : StructuralEnv) (stage scope node : Nat)
    (expression : IndexMapExpr) : Except EvalError Int :=
  expression.eval structural |>.mapError (fun _ => EvalError.invalidStructuralValue stage scope node)

def familyStaticGet {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) (indices : Array IndexMapExpr)
    (arguments : Array (DynamicValue backend)) : Except EvalError (Array (DynamicValue backend)) :=
  match arguments.toList with
  | [⟨.family shape element, value⟩] => do
      let evaluated ← indices.mapM (evalIndexExpr structural stage scope node)
      match evaluated.mapM (fun coordinate =>
          if 0 ≤ coordinate then some coordinate.toNat else none) with
      | some coordinates =>
          match familyIndexFromList shape coordinates.toList with
          | some index => pure #[⟨element, value index⟩]
          | none => throw (.invalidIndex stage scope node)
      | none => throw (.invalidIndex stage scope node)
  | _ => throw (.wrongType stage scope node)

def familyDynamicGet {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node rank : Nat) (arguments : Array (DynamicValue backend)) :
    Except EvalError (Array (DynamicValue backend)) := do
  if arguments.size ≠ rank + 1 then throw (.wrongArity stage scope node)
  let family ← match arguments[0]? with
    | some value => pure value
    | none => throw (.wrongArity stage scope node)
  let indices ← (arguments.extract 1 arguments.size).mapM (fun value =>
    match value with
    | ⟨.int, index⟩ | ⟨.constantInt, index⟩ => pure (IndexMapExpr.literal index)
    | _ => throw (.wrongType stage scope node))
  familyStaticGet structural stage scope node indices #[family]

def evalShape (structural : StructuralEnv) (stage scope node : Nat)
    (shape : Array StructuralIntExpr) : Except EvalError (Array Nat) :=
  shape.mapM (evalNatExpr structural stage scope node)

def shapeProductArray (shape : Array Nat) : Nat :=
  shape.foldl (fun result extent => result * extent) 1

def coordinatesFromOffset : List Nat → Nat → List Nat
  | [], _ => []
  | extent :: rest, offset =>
      let stride := shapeProductArray rest.toArray
      if stride = 0 then [] else (offset / stride) :: coordinatesFromOffset rest (offset % stride)

def dynamicInt? {backend : SemanticBackend} (value : DynamicValue backend) : Option Int :=
  integerValue? value

def packDeclaredFamily {backend : SemanticBackend} (stage scope node : Nat)
    (declared : WireType) (values : Array (DynamicValue backend)) :
    Except EvalError (Array (DynamicValue backend)) :=
  match declared with
  | .family shape element => packFamily stage scope node shape element values
  | _ => throw (.wrongType stage scope node)

def familyReindex {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) (outputShape : Array StructuralIntExpr) (map : IndexMap)
    (declared : WireType) (arguments : Array (DynamicValue backend)) :
    Except EvalError (Array (DynamicValue backend)) := do
  let concreteShape ← evalShape structural stage scope node outputShape
  match arguments.toList with
  | [⟨.family sourceShape element, sourceValue⟩] => do
      let outputs ← (Array.range (shapeProductArray concreteShape)).mapM (fun offset => do
        let coordinates := coordinatesFromOffset concreteShape.toList offset
        let laneStructural := { structural with axes := coordinates.map Int.ofNat |>.toArray }
        let sourceIndices ← map.inputIndices.mapM (evalIndexExpr laneStructural stage scope node)
        match sourceIndices.mapM (fun index =>
            if 0 ≤ index then some index.toNat else none) with
        | some sourceCoordinates =>
            match familyIndexFromList sourceShape sourceCoordinates.toList with
            | some sourceIndex => pure ⟨element, sourceValue sourceIndex⟩
            | none => throw (.invalidIndex stage scope node)
        | none => throw (.invalidIndex stage scope node))
      packDeclaredFamily stage scope node declared outputs
  | _ => throw (.wrongArity stage scope node)

def familyGatherExact {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) (outputShape : Array StructuralIntExpr) (inputRank : Nat)
    (declared : WireType) (arguments : Array (DynamicValue backend)) :
    Except EvalError (Array (DynamicValue backend)) := do
  if arguments.size ≠ inputRank + 1 then throw (.wrongArity stage scope node) else
  let concreteShape ← evalShape structural stage scope node outputShape
  let source ← match arguments[0]? with
    | some value => match value.1 with
      | .family sourceShape _ => pure (sourceShape, value)
      | _ => throw (.wrongType stage scope node)
    | none => throw (.wrongArity stage scope node)
  let sourceShape := source.1
  let selectorValues := (arguments.extract 1 arguments.size)
  let outputs ← (Array.range (shapeProductArray concreteShape)).mapM (fun offset => do
    let coordinates := coordinatesFromOffset concreteShape.toList offset
    let literalCoordinates := coordinates.map IndexMapExpr.literal |>.toArray
    let sourceCoordinates ← selectorValues.mapM (fun selector => do
      let scalar ← familyStaticGet structural stage scope node literalCoordinates #[selector]
      match scalar[0]? with
      | some value => match dynamicInt? value with
        | some coordinate => pure coordinate
        | none => throw (.wrongType stage scope node)
      | none => throw (.invalidIndex stage scope node))
    let sourceValue := source.2
    match sourceValue.1 with
    | .family _ element =>
      let sourceScalar ← familyStaticGet structural stage scope node
        (sourceCoordinates.map IndexMapExpr.literal) #[sourceValue]
      match sourceScalar[0]? with
      | some value => pure value
      | none => throw (.invalidIndex stage scope node)
    | _ => throw (.wrongType stage scope node))
  packDeclaredFamily stage scope node declared outputs

def familySelectAxisExact {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node axis : Nat) (declared : WireType)
    (arguments : Array (DynamicValue backend)) : Except EvalError (Array (DynamicValue backend)) := do
  match arguments.toList with
  | [family, selector] =>
      match family.1 with
      | .family sourceShape _ =>
          if axis ≥ sourceShape.length then throw (.invalidIndex stage scope node)
          let outputShape := sourceShape.eraseIdx axis
          let outputs ← (Array.range (shapeProductArray outputShape.toArray)).mapM (fun offset => do
            let coordinates := coordinatesFromOffset outputShape offset
            let selected ← match dynamicInt? selector with
              | some value => pure value
              | none => match selector.1 with
                | .family selectorShape _ =>
                    if selectorShape ≠ outputShape then throw (.wrongType stage scope node)
                    else
                    let selectorScalar ← familyStaticGet structural stage scope node
                      (coordinates.map IndexMapExpr.literal |>.toArray) #[selector]
                    match selectorScalar[0]? with
                    | some value => match dynamicInt? value with
                      | some coordinate => pure coordinate
                      | none => throw (.wrongType stage scope node)
                    | none => throw (.invalidIndex stage scope node)
                | _ => throw (.wrongType stage scope node)
            if selected < 0 || selected.toNat ≥ sourceShape[axis]! then
              throw (.invalidIndex stage scope node)
            let sourceCoordinates :=
              (coordinates.take axis) ++ [selected.toNat] ++ coordinates.drop axis
            let values ← familyStaticGet structural stage scope node
              (sourceCoordinates.map IndexMapExpr.literal |>.toArray) #[family]
            match values[0]? with
            | some value => pure value
            | none => throw (.invalidIndex stage scope node))
          if outputShape = [] then
            match outputs[0]? with
            | some output => pure #[output]
            | none => throw (.wrongArity stage scope node)
          else packDeclaredFamily stage scope node declared outputs
      | _ => throw (.wrongType stage scope node)
  | _ => throw (.wrongArity stage scope node)

def gridInputArguments {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) (modes : Array GridInputMode)
    (arguments : Array (DynamicValue backend)) : Except EvalError (Array (DynamicValue backend)) :=
  arguments.mapIdxM (fun argumentIndex value =>
    match modes[argumentIndex]? with
    | none | some { reindex := false, .. } => pure value
    | some { reindex := true, map := some map } => do
        let selected ← familyStaticGet structural stage scope node map.inputIndices #[value]
        match selected[0]? with
        | some result => pure result
        | none => throw (.invalidIndex stage scope node)
    | some { reindex := true, map := none } => throw (.invalidIndex stage scope node))

private theorem except_bind_eq_ok {ε α β : Type} (value : Except ε α)
    (next : α → Except ε β) (result : β)
    (success : value >>= next = .ok result) :
    ∃ input, value = .ok input ∧ next input = .ok result := by
  cases value with
  | error error => cases success
  | ok input => exact ⟨input, rfl, success⟩

private theorem list_mapM_getElem {α : Type u} {β : Type v} {ε : Type w}
    (f : α → Except ε β) {xs : List α} {ys : List β} (success : xs.mapM f = .ok ys) :
    xs.length = ys.length ∧ ∀ (i : Nat) (bound : i < xs.length),
      ∃ outputBound : i < ys.length, f xs[i] = .ok ys[i] := by
  induction xs generalizing ys with
  | nil =>
      simp only [List.mapM_nil] at success
      cases success
      simp
  | cons value rest inductionHypothesis =>
      simp only [List.mapM_cons] at success
      cases valueStored : f value with
      | error error =>
          simp only [valueStored] at success
          cases success
      | ok output =>
          cases restStored : rest.mapM f with
          | error error =>
              simp only [valueStored, restStored] at success
              cases success
          | ok outputs =>
              simp [valueStored, restStored] at success
              cases success
              obtain ⟨lengths, points⟩ := inductionHypothesis restStored
              constructor
              · simp [lengths]
              · intro i bound
                cases i with
                | zero => exact ⟨by simp, valueStored⟩
                | succ i =>
                    have restBound : i < rest.length := by simpa using bound
                    obtain ⟨outputBound, point⟩ := points i restBound
                    exact ⟨by simpa using outputBound, by simpa using point⟩

/- Selects one successful element from an `Array.mapM` equation without unfolding the other
   elements.  Grid occurrence proofs use this to descend into one requested lane. -/
theorem array_mapM_getElem {α : Type u} {β : Type v} {ε : Type w}
    (f : α → Except ε β) {xs : Array α} {ys : Array β}
    (success : xs.mapM f = .ok ys) {i : Nat} (bound : i < xs.size) :
    ∃ outputBound : i < ys.size, f xs[i] = .ok ys[i] := by
  have listSuccess := congrArg (fun result : Except ε (Array β) => Array.toList <$> result) success
  have mappedLists : xs.toList.mapM f = .ok ys.toList := by
    simpa only [Array.toList_mapM] using listSuccess
  obtain ⟨lengths, points⟩ := list_mapM_getElem f mappedLists
  have outputBound : i < ys.toList.length := by
    rw [← lengths]
    simpa using bound
  refine ⟨by simpa using outputBound, ?_⟩
  obtain ⟨listOutputBound, point⟩ := points i (by simpa using bound)
  simpa only [Array.getElem_toList bound, Array.getElem_toList listOutputBound] using point

/- The non-structural fallback is factored without changing its `Except` result.  This is the
   theorem-facing boundary for backend primitive equations; structural nodes continue through the
   existing mutually recursive evaluator below. -/
def evalPrimitiveNode (backend : SemanticBackend) (structural : StructuralEnv)
    (stage scope node : Nat) (payload : NodePayload)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType) :
    Except EvalError (NodeResult backend) := do
  let values ← primitive backend structural stage scope node payload arguments outputs
  pure (NodeResult.ofValues values)

theorem evalPrimitiveNode_success {backend : SemanticBackend} (structural : StructuralEnv)
    (stage scope node : Nat) (payload : NodePayload)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType)
    (result : NodeResult backend)
    (success : evalPrimitiveNode backend structural stage scope node payload arguments outputs =
      .ok result) :
    primitive backend structural stage scope node payload arguments outputs = .ok result.values ∧
      result.scopes = #[] := by
  unfold evalPrimitiveNode at success
  obtain ⟨values, evaluated, resultStored⟩ := except_bind_eq_ok _ _ _ success
  cases resultStored
  exact ⟨evaluated, rfl⟩

structure GadgetDecomposeExecution (backend : SemanticBackend)
    (structural : StructuralEnv) (stage scope node : Nat)
    (baseExpr : StructuralIntExpr) (small : Bool) (digitsExpr : StructuralIntExpr)
    (targetType outputType : MatrixType) (target : backend.denoteMatrix targetType)
    (output : backend.denotePreimage outputType) where
  baseValue : Nat
  digitValue : Nat
  baseEvaluated : baseExpr.eval structural = .ok (Int.ofNat baseValue)
  digitsEvaluated : evalNatExpr structural stage scope node digitsExpr = .ok digitValue
  regularMode : small = false
  baseValid : 1 < baseValue
  digitsValid : 0 < digitValue
  modulusMatches : targetType.modulus = outputType.modulus
  ringDimensionMatches : targetType.ringDimension = outputType.ringDimension
  rowsMatch : outputType.rows = targetType.rows * digitValue
  columnsMatch : outputType.columns = targetType.columns
  capacity : targetType.modulus.toNat ≤ baseValue ^ digitValue
  layout : GadgetLayout
  layoutEq : layout = {
    mode := .regular
    base := baseValue
    digits := digitValue
    sourceRows := targetType.rows
    targetRows := outputType.rows
    sourceColumns := targetType.columns
    targetColumns := outputType.columns }
  layoutValid : layout.Valid
  outputTypeEq : outputType = gadgetPreimageType targetType layout
  sigma : Σ preimage : backend.denotePreimage (gadgetPreimageType targetType layout),
    PLift (backend.gadgetCertificate targetType layout structural
      (backend.matrixConstant (gadgetMatrixType targetType layout)
        (.gadget (.literal (Int.ofNat baseValue)) false) structural) target preimage)
  oracleSuccess : backend.gadgetDecompose targetType layout structural
      (backend.matrixConstant (gadgetMatrixType targetType layout)
        (.gadget (.literal (Int.ofNat baseValue)) false) structural) target = .ok sigma
  outputEq : output = outputTypeEq ▸ sigma.1
  certificate : backend.gadgetCertificate targetType layout structural
    (backend.matrixConstant (gadgetMatrixType targetType layout)
      (.gadget (.literal (Int.ofNat baseValue)) false) structural) target sigma.1

theorem evalGadgetDecompose_success (backend : SemanticBackend) (structural : StructuralEnv)
    (stage scope node : Nat) (baseExpr : StructuralIntExpr) (small : Bool)
    (digitsExpr : StructuralIntExpr) (targetType outputType : MatrixType)
    (target : backend.denoteMatrix targetType) (output : backend.denotePreimage outputType)
    (success : evalGadgetDecompose backend structural stage scope node baseExpr small digitsExpr
      targetType outputType target = .ok output) :
    Nonempty (GadgetDecomposeExecution backend structural stage scope node baseExpr small
      digitsExpr targetType outputType target output) := by
  unfold evalGadgetDecompose at success
  have smallFalse : small ≠ true := by
    intro smallTrue
    simp only [if_pos smallTrue] at success
    cases success
  simp only [if_neg smallFalse] at success
  obtain ⟨baseInteger, baseEvaluated, afterBase⟩ := except_bind_eq_ok _ _ _ success
  have baseNonnegative : 0 ≤ baseInteger := by
    by_contra h
    simp only [if_neg h] at afterBase
    cases afterBase
  simp only [if_pos baseNonnegative] at afterBase
  obtain ⟨baseValue, baseValueEvaluated, afterValue⟩ := except_bind_eq_ok _ _ _ afterBase
  have baseValueEq : baseValue = baseInteger.toNat := by
    exact Except.ok.inj baseValueEvaluated.symm
  subst baseValue
  obtain ⟨digitValue, digitsEvaluated, afterDigits⟩ := except_bind_eq_ok _ _ _ afterValue
  by_cases smallTrue : small = true
  · exact (smallFalse smallTrue).elim
  ·
    by_cases hValid : 1 < baseInteger.toNat ∧ 0 < digitValue ∧
        targetType.modulus = outputType.modulus ∧
        targetType.ringDimension = outputType.ringDimension ∧
        outputType.rows = targetType.rows * digitValue ∧
        outputType.columns = targetType.columns
    · simp only [dif_pos hValid] at afterDigits
      let layout : GadgetLayout := {
        mode := .regular
        base := baseInteger.toNat
        digits := digitValue
        sourceRows := targetType.rows
        targetRows := outputType.rows
        sourceColumns := targetType.columns
        targetColumns := outputType.columns }
      let gadget := backend.matrixConstant (gadgetMatrixType targetType layout)
        (.gadget (.literal (Int.ofNat baseInteger.toNat)) false) structural
      by_cases hCapacity : targetType.modulus.toNat ≤ baseInteger.toNat ^ digitValue
      · simp only [dif_pos hCapacity] at afterDigits
        obtain ⟨sigma, oracleSuccess, afterOracle⟩ := except_bind_eq_ok _ _ _ afterDigits
        split at afterOracle
        · rename_i outputTypeEq
          have baseEvaluated' : baseExpr.eval structural = .ok (Int.ofNat baseInteger.toNat) := by
            cases evaluated : baseExpr.eval structural with
            | error error =>
                exfalso
                rw [evaluated] at baseEvaluated
                cases baseEvaluated
            | ok value =>
                rw [evaluated] at baseEvaluated
                have normalized : (Except.ok value : Except EvalError Int) = .ok baseInteger := by
                  exact baseEvaluated
                have valueEq : value = baseInteger := Except.ok.inj normalized
                have integerEq : value = Int.ofNat baseInteger.toNat :=
                  valueEq.trans (Int.toNat_of_nonneg baseNonnegative).symm
                exact congrArg (fun integer : Int => (Except.ok integer : Except String Int))
                  integerEq
          have regularMode' : small = false := by
            cases small with
            | false => rfl
            | true => contradiction
          have hBase : 1 < baseInteger.toNat := hValid.1
          have hDigits : 0 < digitValue := by exact hValid.2.1
          have hModulus : targetType.modulus = outputType.modulus := hValid.2.2.1
          have hRing : targetType.ringDimension = outputType.ringDimension := hValid.2.2.2.1
          have hRows : outputType.rows = targetType.rows * digitValue := hValid.2.2.2.2.1
          have hColumns : outputType.columns = targetType.columns := hValid.2.2.2.2.2
          have layoutValid : layout.Valid := by
            dsimp [layout, GadgetLayout.Valid]
            exact ⟨hBase, hDigits, hRows, hColumns⟩
          have outputEq : output = outputTypeEq ▸ sigma.1 := by
            exact Except.ok.inj afterOracle |>.symm
          exact ⟨{
            baseValue := baseInteger.toNat
            digitValue := digitValue
            baseEvaluated := baseEvaluated'
            digitsEvaluated := digitsEvaluated
            regularMode := regularMode'
            baseValid := hBase
            digitsValid := hDigits
            modulusMatches := hModulus
            ringDimensionMatches := hRing
            rowsMatch := hRows
            columnsMatch := hColumns
            capacity := hCapacity
            layout := layout
            layoutEq := rfl
            layoutValid := layoutValid
            outputTypeEq := outputTypeEq
            sigma := sigma
            oracleSuccess := by
              cases evaluated : backend.gadgetDecompose targetType layout structural gadget target with
              | error error =>
                  rw [evaluated] at oracleSuccess
                  cases oracleSuccess
              | ok value =>
                  rw [evaluated] at oracleSuccess
                  change (Except.ok value : Except EvalError _) = Except.ok sigma at oracleSuccess
                  have valueEq : value = sigma := Except.ok.inj oracleSuccess
                  exact congrArg (fun result => (Except.ok result : Except GadgetFailure _)) valueEq
            outputEq := outputEq
            certificate := sigma.2.down }⟩
        · cases afterOracle
      · simp only [dif_neg hCapacity] at afterDigits
        cases afterDigits
    · simp only [dif_neg hValid] at afterDigits
      cases afterDigits

/- This predicate names precisely the payloads whose successful execution is delegated to
   `evalPrimitiveNode`.  Samplers, inputs, family operations, and structural control flow have
   separate evaluator branches and therefore cannot inhabit this predicate. -/
inductive PrimitiveNodePayload : NodePayload → Prop where
  | constantInt (value : Int) : PrimitiveNodePayload (.constantInt value)
  | evaluateInt (value : StructuralIntExpr) : PrimitiveNodePayload (.evaluateInt value)
  | constantBool (value : Bool) : PrimitiveNodePayload (.constantBool value)
  | constantMatrix (matrixType : MatrixType) (literal : MatrixLiteral)
      (supported : match literal with | .gadget _ true => False | _ => True) :
      PrimitiveNodePayload (.constantMatrix matrixType literal)
  | intBinary (op : IntBinaryOp) : PrimitiveNodePayload (.intBinary op)
  | intCompare (op : IntCompareOp) : PrimitiveNodePayload (.intCompare op)
  | bitExtract (bit : StructuralIntExpr) : PrimitiveNodePayload (.bitExtract bit)
  | intToReal : PrimitiveNodePayload .intToReal
  | boolToInt : PrimitiveNodePayload .boolToInt
  | realBinary (op : RealBinaryOp) : PrimitiveNodePayload (.realBinary op)
  | realSqrt : PrimitiveNodePayload .realSqrt
  | matrixBinary (op : MatrixBinaryOp) : PrimitiveNodePayload (.matrixBinary op)
  | matrixNegate : PrimitiveNodePayload .matrixNegate
  | matrixScale (scalar : StructuralIntExpr) : PrimitiveNodePayload (.matrixScale scalar)
  | transpose : PrimitiveNodePayload .transpose
  | slice (rows columns : Option IntRange) : PrimitiveNodePayload (.slice rows columns)
  | concat (axis : ConcatAxis) : PrimitiveNodePayload (.concat axis)
  | gadgetDecompose (base digits : StructuralIntExpr) :
      PrimitiveNodePayload (.gadgetDecompose base false digits)
  | extractCoefficient (position : StructuralIntExpr) (upper : Option Nat) :
      PrimitiveNodePayload (.extractCoefficient position upper)
  | trapdoorPublic : PrimitiveNodePayload .trapdoorPublic
  | applyPreimage : PrimitiveNodePayload .applyPreimage
  | materializePreimageExact : PrimitiveNodePayload .materializePreimageExact

theorem evalPrimitiveNode_gadgetDecompose_success {backend : SemanticBackend}
    (structural : StructuralEnv) (stage scope node : Nat) (baseExpr : StructuralIntExpr)
    (small : Bool) (digitsExpr : StructuralIntExpr)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType)
    (result : NodeResult backend)
    (success : evalPrimitiveNode backend structural stage scope node
      (.gadgetDecompose baseExpr small digitsExpr) arguments outputs = .ok result) :
    ∃ (targetType outputType : MatrixType) (target : backend.denoteMatrix targetType)
      (output : backend.denotePreimage outputType),
      arguments = #[⟨.matrix targetType, target⟩] ∧ outputs = #[.preimage outputType] ∧
      result = NodeResult.ofValues #[⟨.preimage outputType, output⟩] ∧
      Nonempty (GadgetDecomposeExecution backend structural stage scope node baseExpr small
        digitsExpr targetType outputType target output) := by
  have smallFalse : small = false := by
    cases small with
    | false => rfl
    | true =>
      exfalso
      simp only [evalPrimitiveNode, primitive] at success
      change (Except.error (.unsupportedPrimitive stage scope node) :
        Except EvalError (NodeResult backend)) = .ok result at success
      cases success
  subst small
  unfold evalPrimitiveNode at success
  obtain ⟨values, primitiveSuccess, resultEq⟩ :=
    except_bind_eq_ok (primitive backend structural stage scope node
      (.gadgetDecompose baseExpr false digitsExpr) arguments outputs) _ _ success
  cases args : arguments.toList with
  | nil => simp [primitive, args] at primitiveSuccess
  | cons argument rest =>
    cases rest with
    | nil =>
      cases outs : outputs.toList with
      | nil => simp [primitive, args, outs] at primitiveSuccess
      | cons outputType rest =>
        cases rest with
        | nil =>
          cases argument with
          | mk argumentType argumentValue =>
            cases argumentType with
            | matrix targetType =>
              cases outputType with
              | preimage outputType =>
                have primitiveSuccess' :
                    (evalGadgetDecompose backend structural stage scope node baseExpr false
                      digitsExpr targetType outputType argumentValue >>= fun result =>
                      pure #[⟨.preimage outputType, result⟩]) = .ok values := by
                  simpa only [primitive, args, outs] using primitiveSuccess
                obtain ⟨output, outputSuccess, arrayEq⟩ :=
                  except_bind_eq_ok _ _ _ primitiveSuccess'
                have execution := evalGadgetDecompose_success backend structural stage scope node
                  baseExpr false digitsExpr targetType outputType argumentValue output outputSuccess
                refine ⟨targetType, outputType, argumentValue, output, ?_, ?_, ?_, execution⟩
                · apply Array.toList_inj.mp
                  simpa [args]
                · apply Array.toList_inj.mp
                  simpa [outs]
                · have valuesEq : values = #[⟨.preimage outputType, output⟩] :=
                    (Except.ok.inj arrayEq).symm
                  rw [valuesEq] at resultEq
                  exact (Except.ok.inj resultEq).symm
              | _ => simp [primitive, args, outs] at primitiveSuccess
            | _ => simp [primitive, args, outs] at primitiveSuccess
        | _ => simp [primitive, args, outs] at primitiveSuccess
    | _ => simp [primitive, args] at primitiveSuccess

def appendNodeBindings {backend : SemanticBackend} (scope index : Nat)
    (values : Array (Binding backend)) (result : Array (DynamicValue backend)) :
    Array (Binding backend) :=
  (List.range result.size).foldl (fun accumulated port =>
    match result[port]? with
    | some value => accumulated.push {
        wire := { scope := scope, node := index, port := port }
        value := value }
    | none => accumulated) values

mutual

def evalScope {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data)
    (structural : StructuralEnv)
    (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId) (scope : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope)
    (inputs : Array (Binding backend)) (path : OccurrencePath) (index : Nat)
    (values : Array (Binding backend)) (fuel : Nat) : Except EvalError (ScopeResult backend) :=
  if fuel = 0 then throw (.invalidStructuralValue stageNumber scope.id index)
  else if hIndex : index < scope.nodes.size then
    match node : scope.nodes[index]? with
    | none => throw (.missingNode stageNumber scope.id index)
    | some nodeValue => do
      let argumentValues ← resolveArguments stageNumber scope.id index values nodeValue.arguments
      let wire : WireRef := { scope := scopeNumber, node := index, port := 0 }
      let result ← match nodeValue.payload with
      | .input inputIndex =>
          match inputs[inputIndex]? with
          | some binding => pure (NodeResult.ofValues #[binding.value])
          | none => do
              let value ← envInput env stageNumber scope.id index path wire
              pure (NodeResult.ofValues #[value])
      | .artifactInput input =>
          match data.artifactLinks[input.index]? with
          | none => throw (.missingArtifact stageNumber scope.id index)
          | some link =>
              if link.consumerStage != stageNumber || link.consumer != wire ||
                  link.argument != wire.port then
                throw (.missingArtifact stageNumber scope.id index)
              else match trace.stages[link.producerStage]? with
              | none => throw (.missingArtifact stageNumber scope.id index)
              | some producerTrace =>
                  match producerTrace.scopes.find? (fun item => item.scope = link.producer.scope) with
                  | none => throw (.missingArtifact stageNumber scope.id index)
                  | some producerScope =>
                      match lookup producerScope.values link.producer with
                      | some value => pure (NodeResult.ofValues #[value])
                      | none => throw (.missingArtifact stageNumber scope.id index)
      | payload =>
          if h : samplerPayload nodeValue.payload = true then do
            let sampled ← nodeValue.outputs.mapIdxM fun port _ =>
              let outputWire : WireRef := { scope := scopeNumber, node := index, port := port }
              if hPort : port < nodeValue.outputs.size then
                let outputType := nodeValue.outputs[port]'hPort
                have hOutput : nodeValue.outputs[port]? = some outputType := by
                  rw [Array.getElem?_eq_getElem]
                envSample env stageNumber stage scope index nodeValue path outputWire outputType
                  stageStored scopeStored node hPort hOutput h
              else throw (.missingPort stageNumber scope.id index port)
            pure (NodeResult.ofValues sampled)
          else
            match payload with
            | .subgraphCall call =>
                -- A call contributes a distinct call frame and preserves every child scope trace.
                match childStored : scopeAt stage call.child with
                | none => throw (.missingScope stageNumber call.child)
                | some child => do
                    let childResult ← evalScope data env structural trace stageNumber stage call.child child
                      stageStored childStored
                      (← checkedChildInputs stageNumber scope.id index child argumentValues) (path.push {
                        stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := 0 })
                      0 #[] (fuel - 1)
                    let childOutputValues ← child.outputs.mapM (fun output =>
                      match lookup childResult.values output with
                      | some value => pure value
                      | none => throw (.missingPort stageNumber child.id output.node output.port))
                    pure { values := childOutputValues, scopes := childResult.scopes }
            | .parallelGrid grid =>
                -- Grid execution is row-major: each lane receives its own axes and declared slots.
                match childStored : scopeAt stage grid.child with
                | none => throw (.missingScope stageNumber grid.child)
                | some child => do
                    let concreteShape ← evalShape structural stageNumber scope.id index grid.shape
                    let lanes := shapeProductArray concreteShape
                    let laneResults ← (Array.range lanes).mapM (fun lane => do
                      let coordinates := coordinatesFromOffset concreteShape.toList lane
                      let laneStructural := { structural with
                        axes := (coordinates.map Int.ofNat).toArray
                        slots := grid.indexSlots.zip coordinates.toArray |>.map
                          (fun item => (item.1, Int.ofNat item.2)) }
                      let lanePath := path.push {
                        stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := lane }
                      let laneArguments ← gridInputArguments laneStructural stageNumber scope.id index
                        grid.inputModes argumentValues
                      let childResult ← evalScope data env laneStructural trace stageNumber stage grid.child child
                        stageStored childStored
                        (← checkedChildInputs stageNumber scope.id index child laneArguments) lanePath 0 #[] (fuel - 1)
                      let outputs ← child.outputs.mapM (fun output =>
                        match lookup childResult.values output with
                        | some value => pure value
                        | none => throw (.missingPort stageNumber child.id output.node output.port))
                      pure (outputs, childResult.scopes))
                    let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
                    let packed ← nodeValue.outputs.mapIdxM (fun outputIndex output => do
                      let laneValues ← laneResults.mapM (fun result =>
                        match result.1[outputIndex]? with
                        | some value => pure value
                        | none => throw (.missingPort stageNumber child.id outputIndex 0))
                      let packedValues ← packDeclaredFamily stageNumber scope.id index output laneValues
                      match packedValues[0]? with
                      | some value => pure value
                      | none => throw (.wrongType stageNumber scope.id index))
                    pure { values := packed, scopes := laneScopes }
            | .sequentialLoop loop =>
                -- Loop iterations carry the first arguments; invariant arguments are appended unchanged.
                match childStored : scopeAt stage loop.child with
                | none => throw (.missingScope stageNumber loop.child)
                | some child =>
                    evalSequentialLoop data env trace stageNumber stage loop.child child stageStored childStored
                      loop index argumentValues structural path 0 (fuel - 1)
            | .familyPack _ =>
                match nodeValue.outputs[0]? with
                | some output => do
                    let values ← packDeclaredFamily stageNumber scope.id index output argumentValues
                    pure (NodeResult.ofValues values)
                | none => throw (.wrongType stageNumber scope.id index)
            | .familyGetStatic indices => do
                let values ← familyStaticGet structural stageNumber scope.id index indices argumentValues
                pure (NodeResult.ofValues values)
            | .familyGetDynamic rank => do
                let values ← familyDynamicGet structural stageNumber scope.id index rank argumentValues
                pure (NodeResult.ofValues values)
            | .familyReindex outputShape map => do
                match nodeValue.outputs[0]? with
                | some declared =>
                    let values ← familyReindex structural stageNumber scope.id index outputShape map declared
                      argumentValues
                    pure (NodeResult.ofValues values)
                | none => throw (.wrongType stageNumber scope.id index)
            | .familyGather outputShape inputRank => do
                match nodeValue.outputs[0]? with
                | some declared => do
                    let values ← familyGatherExact structural stageNumber scope.id index outputShape inputRank
                      declared argumentValues
                    pure (NodeResult.ofValues values)
                | none => throw (.wrongType stageNumber scope.id index)
            | .familySelectAxis axis => do
                match nodeValue.outputs[0]? with
                | some declared => do
                    let values ← familySelectAxisExact structural stageNumber scope.id index axis
                      declared argumentValues
                    pure (NodeResult.ofValues values)
                | none => throw (.wrongType stageNumber scope.id index)
            | .select count => do
                let branchCount ← evalNatExpr structural stageNumber scope.id index count
                if argumentValues.size ≠ branchCount + 1 then throw (.wrongArity stageNumber scope.id index)
                let selector ← match argumentValues[0]? with
                  | some value =>
                      match dynamicInt? value with
                      | some integer => pure integer
                      | none => throw (.wrongType stageNumber scope.id index)
                  | none => throw (.wrongType stageNumber scope.id index)
                if 0 ≤ selector ∧ selector < (branchCount : Int) then
                  match argumentValues[selector.toNat + 1]? with
                  | some value => pure (NodeResult.ofValues #[value])
                  | none => throw (.invalidIndex stageNumber scope.id index)
                else throw (.invalidIndex stageNumber scope.id index)
            | _ => do
                evalPrimitiveNode backend structural stageNumber scope.id index payload
                  argumentValues nodeValue.outputs
      if outputTypesMatch nodeValue.outputs.toList result.values.toList then
        let newValues := appendNodeBindings scope.id index values result.values
        let next ← evalScope data env structural trace stageNumber stage scopeNumber scope
          stageStored scopeStored inputs path
          (index + 1) newValues
          (fuel - 1)
        pure { values := next.values, scopes := result.scopes ++ next.scopes ++ #[{
          scope := scope.id, occurrence := path, values := newValues }] }
      else throw (.wrongType stageNumber scope.id index)
  else pure { values := values, scopes := #[{ scope := scope.id, occurrence := path, values := values }] }
termination_by fuel

def evalSequentialLoop {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (childNumber : ScopeId) (child : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (childStored : scopeAt stage childNumber = some child)
    (loop : LoopPayload) (owner : NodeId) (arguments : Array (DynamicValue backend))
    (structural : StructuralEnv) (path : OccurrencePath) (iteration fuel : Nat) :
    Except EvalError (NodeResult backend) :=
  if fuel = 0 then throw (.invalidStructuralValue stageNumber child.id owner)
  else do
    let count ← evalNatExpr structural stageNumber child.id owner loop.count
    if iteration < count then do
    let iterationPath := path.push {
      stage := stageNumber, scope := child.id, owner := owner, laneOrIteration := iteration }
    let iterationStructural := { structural with
      slots := structural.slots.push (loop.indexSlot, Int.ofNat iteration) }
    let childResult ← evalScope data env iterationStructural trace stageNumber stage childNumber child
      stageStored childStored
      (← checkedChildInputs stageNumber child.id owner child arguments) iterationPath 0 #[] (fuel - 1)
      let childValues ← child.outputs.mapM (fun output =>
        match lookup childResult.values output with
        | some value => pure value
        | none => throw (EvalError.missingPort stageNumber child.id output.node output.port))
      if childValues.size != loop.carriedCount then
        throw (EvalError.wrongArity stageNumber child.id owner)
      else
        let invariants := arguments.extract loop.carriedCount arguments.size
        let nextArguments := childValues ++ invariants
        let rest ← evalSequentialLoop data env trace stageNumber stage childNumber child stageStored childStored loop owner
          nextArguments structural path (iteration + 1) (fuel - 1)
        pure { values := rest.values, scopes := childResult.scopes ++ rest.scopes }
    else pure { values := arguments.extract 0 loop.carriedCount, scopes := #[] }
termination_by fuel

end

/- One successful primitive node step exposes the resolved arguments, the backend primitive
   equation, the exact SSA bindings appended for this node, and the unchanged recursive call for
   the following node.  A generated proof can therefore traverse a concrete scope one node at a
   time without unfolding the entire evaluation. -/
theorem evalScope_success_primitive_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue)
    (payload : NodePayload) (payloadStored : nodeValue.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues result nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalPrimitiveNode backend structural stageNumber scope.id index payload
          argumentValues nodeValue.outputs = .ok result ∧
      outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := result.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values result.values }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  subst payload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode =
        { payload := nodePayload, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ :=
      except_bind_eq_ok _ _ _ success
    cases primitivePayload <;>
      simp only [samplerPayload, Bool.false_eq_true] at afterArguments
    all_goals
      obtain ⟨result, resultStored, afterResult⟩ :=
        except_bind_eq_ok _ _ _ afterArguments
      split at afterResult
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ :=
          except_bind_eq_ok _ _ _ afterResult
        cases finalStored
        exact ⟨argumentValues, result, nextResult, argumentsStored, resultStored,
          typesMatch, nextStored, rfl⟩
      · contradiction

theorem evalScope_success_subgraph_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (call : SubgraphPayload)
    (payloadStored : nodeValue.payload = .subgraphCall call)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues child, ∃ childStored : scopeAt stage call.child = some child,
      ∃ childInputs : Array (Binding backend), ∃ childResult : ScopeResult backend,
      ∃ childOutputs : Array (DynamicValue backend), ∃ nextResult : ScopeResult backend,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      checkedChildInputs stageNumber scope.id index child argumentValues = .ok childInputs ∧
      evalScope data env structural trace stageNumber stage call.child child stageStored childStored
          childInputs (path.push {
            stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := 0 })
          0 #[] (fuel - 1) = .ok childResult ∧
      child.outputs.mapM (fun output =>
          (match lookup childResult.values output with
          | some value => Except.ok value
          | none => Except.error
              (EvalError.missingPort stageNumber child.id output.node output.port) :
            Except EvalError (DynamicValue backend))) = .ok childOutputs ∧
      outputTypesMatch nodeValue.outputs.toList childOutputs.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values childOutputs) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := childResult.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values childOutputs }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = .subgraphCall call at payloadStored
  subst nodePayload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode = {
        payload := .subgraphCall call, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ := except_bind_eq_ok _ _ _ success
    have notSampler : ¬ samplerPayload (.subgraphCall call) = true := by
      simp [samplerPayload]
    simp only [dif_neg notSampler] at afterArguments
    split at afterArguments
    · contradiction
    · rename_i childStored
      obtain ⟨childInputs, childInputsStored, afterChildInputs⟩ :=
        except_bind_eq_ok _ _ _ afterArguments
      obtain ⟨childResult, childResultStored, afterChild⟩ :=
        except_bind_eq_ok _ _ _ afterChildInputs
      obtain ⟨childOutputs, childOutputsStored, afterOutputs⟩ :=
        except_bind_eq_ok _ _ _ afterChild
      obtain ⟨result, resultStored, afterResult⟩ := except_bind_eq_ok _ _ _ afterOutputs
      cases resultStored
      split at afterResult
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ := except_bind_eq_ok _ _ _ afterResult
        cases finalStored
        exact ⟨argumentValues, _, childStored, childInputs, childResult, childOutputs,
          nextResult, argumentsStored, childInputsStored, childResultStored, childOutputsStored,
          typesMatch, nextStored, rfl⟩
      · contradiction

theorem evalScope_success_parallelGrid_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (grid : GridPayload)
    (payloadStored : nodeValue.payload = .parallelGrid grid)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues child, ∃ childStored : scopeAt stage grid.child = some child,
      ∃ concreteShape : Array Nat, ∃ lanes : Nat,
      ∃ laneResults : Array (Array (DynamicValue backend) × Array (ScopeTrace backend)),
      ∃ packed nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalShape structural stageNumber scope.id index grid.shape = .ok concreteShape ∧
      lanes = shapeProductArray concreteShape ∧
      (Array.range lanes).mapM (fun lane => do
        let coordinates := coordinatesFromOffset concreteShape.toList lane
        let laneStructural := { structural with
          axes := (coordinates.map Int.ofNat).toArray
          slots := grid.indexSlots.zip coordinates.toArray |>.map
            (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
        let lanePath := path.push {
          stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := lane }
        let laneArguments ← gridInputArguments laneStructural stageNumber scope.id index
          grid.inputModes argumentValues
        let childInputs ← checkedChildInputs stageNumber scope.id index child laneArguments
        let childResult ← evalScope data env laneStructural trace stageNumber stage grid.child child
          stageStored childStored childInputs lanePath 0 #[] (fuel - 1)
        let outputs ← child.outputs.mapM (fun output =>
          (match lookup childResult.values output with
          | some value => Except.ok value
          | none => Except.error
              (EvalError.missingPort stageNumber child.id output.node output.port) :
            Except EvalError (DynamicValue backend)))
        pure (outputs, childResult.scopes)) = .ok laneResults ∧
      let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
      nodeValue.outputs.mapIdxM (fun outputIndex output => do
        let laneValues ← laneResults.mapM
          (fun result : Array (DynamicValue backend) × Array (ScopeTrace backend) =>
          (match result.1[outputIndex]? with
          | some value => Except.ok value
          | none => Except.error (EvalError.missingPort stageNumber child.id outputIndex 0) :
            Except EvalError (DynamicValue backend)))
        let packedValues ← packDeclaredFamily stageNumber scope.id index output laneValues
        match packedValues[0]? with
        | some value => Except.ok value
        | none => Except.error (EvalError.wrongType stageNumber scope.id index)) = .ok packed ∧
      outputTypesMatch nodeValue.outputs.toList packed.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values packed) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := laneScopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values packed }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = .parallelGrid grid at payloadStored
  subst nodePayload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode = {
        payload := .parallelGrid grid, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ := except_bind_eq_ok _ _ _ success
    have notSampler : ¬ samplerPayload (.parallelGrid grid) = true := by
      simp [samplerPayload]
    simp only [dif_neg notSampler] at afterArguments
    split at afterArguments
    · contradiction
    · rename_i childStored
      obtain ⟨concreteShape, shapeStored, afterShape⟩ := except_bind_eq_ok _ _ _ afterArguments
      obtain ⟨laneResults, lanesStored, afterLanes⟩ := except_bind_eq_ok _ _ _ afterShape
      obtain ⟨packed, packedStored, afterPacked⟩ := except_bind_eq_ok _ _ _ afterLanes
      obtain ⟨result, resultStored, afterResult⟩ := except_bind_eq_ok _ _ _ afterPacked
      cases resultStored
      split at afterResult
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ := except_bind_eq_ok _ _ _ afterResult
        cases finalStored
        exact ⟨argumentValues, _, childStored, concreteShape, shapeProductArray concreteShape,
          laneResults, packed, nextResult, argumentsStored, shapeStored, rfl, lanesStored,
          packedStored, typesMatch, nextStored, rfl⟩
      · contradiction

theorem evalScope_success_sequentialLoop_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (loop : LoopPayload)
    (payloadStored : nodeValue.payload = .sequentialLoop loop)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues child, ∃ childStored : scopeAt stage loop.child = some child,
      ∃ loopResult nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalSequentialLoop data env trace stageNumber stage loop.child child stageStored childStored
          loop index argumentValues structural path 0 (fuel - 1) = .ok loopResult ∧
      outputTypesMatch nodeValue.outputs.toList loopResult.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1)
            (appendNodeBindings scope.id index values loopResult.values) (fuel - 1) = .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := loopResult.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values loopResult.values }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = .sequentialLoop loop at payloadStored
  subst nodePayload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode = {
        payload := .sequentialLoop loop, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ := except_bind_eq_ok _ _ _ success
    have notSampler : ¬ samplerPayload (.sequentialLoop loop) = true := by
      simp [samplerPayload]
    simp only [dif_neg notSampler] at afterArguments
    split at afterArguments
    · contradiction
    · rename_i childStored
      obtain ⟨loopResult, loopStored, afterLoop⟩ := except_bind_eq_ok _ _ _ afterArguments
      split at afterLoop
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ := except_bind_eq_ok _ _ _ afterLoop
        cases finalStored
        exact ⟨argumentValues, _, childStored, loopResult, nextResult, argumentsStored,
          loopStored, typesMatch, nextStored, rfl⟩
      · contradiction


theorem evalSequentialLoop_success_iteration_step {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (childNumber : ScopeId) (child : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (childStored : scopeAt stage childNumber = some child) (loop : LoopPayload) (owner : NodeId)
    (arguments : Array (DynamicValue backend)) (structural : StructuralEnv) (path : OccurrencePath)
    (iteration fuel count : Nat) (fuelPositive : fuel ≠ 0)
    (countStored : evalNatExpr structural stageNumber child.id owner loop.count = .ok count)
    (iterationBound : iteration < count) (finalResult : NodeResult backend)
    (success : evalSequentialLoop data env trace stageNumber stage childNumber child stageStored
      childStored loop owner arguments structural path iteration fuel = .ok finalResult) :
    ∃ childInputs : Array (Binding backend), ∃ childResult : ScopeResult backend,
      ∃ childValues : Array (DynamicValue backend), ∃ rest : NodeResult backend,
      checkedChildInputs stageNumber child.id owner child arguments = .ok childInputs ∧
      evalScope data env { structural with
          slots := structural.slots.push (loop.indexSlot, Int.ofNat iteration) }
          trace stageNumber stage childNumber child stageStored childStored childInputs
          (path.push {
            stage := stageNumber, scope := child.id, owner := owner,
            laneOrIteration := iteration }) 0 #[] (fuel - 1) = .ok childResult ∧
      child.outputs.mapM (fun output =>
          (match lookup childResult.values output with
          | some value => Except.ok value
          | none => Except.error
              (EvalError.missingPort stageNumber child.id output.node output.port) :
            Except EvalError (DynamicValue backend))) = .ok childValues ∧
      childValues.size = loop.carriedCount ∧
      evalSequentialLoop data env trace stageNumber stage childNumber child stageStored childStored
          loop owner (childValues ++ arguments.extract loop.carriedCount arguments.size)
          structural path (iteration + 1) (fuel - 1) = .ok rest ∧
      finalResult = { values := rest.values, scopes := childResult.scopes ++ rest.scopes } := by
  rw [evalSequentialLoop] at success
  simp only [if_neg fuelPositive] at success
  obtain ⟨actualCount, actualCountStored, afterCount⟩ := except_bind_eq_ok _ _ _ success
  have countEq : actualCount = count := by
    exact Except.ok.inj (actualCountStored.symm.trans countStored)
  subst actualCount
  simp only [if_pos iterationBound] at afterCount
  obtain ⟨childInputs, childInputsStored, afterChildInputs⟩ := except_bind_eq_ok _ _ _ afterCount
  obtain ⟨childResult, childResultStored, afterChild⟩ := except_bind_eq_ok _ _ _ afterChildInputs
  obtain ⟨childValues, childValuesStored, afterValues⟩ := except_bind_eq_ok _ _ _ afterChild
  split at afterValues
  · contradiction
  · rename_i carried
    have carriedEq : childValues.size = loop.carriedCount := by
      simpa using carried
    obtain ⟨rest, restStored, afterGuard⟩ := except_bind_eq_ok _ _ _ afterValues
    have finalEq : finalResult = {
        values := rest.values, scopes := childResult.scopes ++ rest.scopes } := by
      exact Except.ok.inj afterGuard |>.symm
    exact ⟨childInputs, childResult, childValues, rest, childInputsStored,
      childResultStored, childValuesStored, carriedEq, restStored, finalEq⟩


def evaluationFuel (data : ProgramData) : Nat :=
  let work := data.stages.foldl (fun stageTotal stage =>
    stageTotal + stage.scopes.foldl (fun scopeTotal scope => scopeTotal + scope.nodes.size + 1) 0) 0
  Nat.pow 2 (work + 1)

def evalStage {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data)
    (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage) (stageStored : data.stages[stageNumber]? = some stage) :
    Except EvalError (StageTrace backend) := do
  match rootStored : scopeAt stage stage.root with
  | none => throw (.missingScope stageNumber stage.root)
  | some root => do
      let result ← evalScope data env {} trace stageNumber stage stage.root root stageStored rootStored
        #[] #[] 0 #[] (evaluationFuel data)
      pure { stage := stageNumber, scopes := result.scopes }

def evalStages {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data) (index : Nat)
    (trace : Trace backend) : Except EvalError (Trace backend) :=
  if h : index < data.stages.size then
    match stageStored : data.stages[index]? with
    | none => throw (.missingScope index 0)
    | some stage => do
        let stageTrace ← evalStage data env trace index stage stageStored
        evalStages data env (index + 1) { stages := trace.stages.push stageTrace }
  else pure trace
termination_by data.stages.size - index

def eval (backend : SemanticBackend) (program : Program) (env : EvalEnv backend program.data) :
    Except EvalError (Trace backend) := evalStages program.data env 0 { stages := #[] }

/- Successful evaluator equations are inverted through the actual `Except.bind`; no relational or
   alternative evaluator is introduced. -/
/- One successful `evalStages` step exposes the exact `evalStage` call and the unchanged recursive
   continuation used by the evaluator. -/
theorem evalStages_success_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (index : Nat)
    (trace finalTrace : Trace backend) (bound : index < data.stages.size)
    (success : evalStages data env index trace = .ok finalTrace) :
    ∃ stage, ∃ stageStored : data.stages[index]? = some stage, ∃ stageTrace,
      evalStage data env trace index stage stageStored = .ok stageTrace ∧
        evalStages data env (index + 1) { stages := trace.stages.push stageTrace } =
          .ok finalTrace := by
  rw [evalStages] at success
  simp only [dif_pos bound] at success
  split at success
  · contradiction
  · rename_i stage stageStored
    obtain ⟨stageTrace, stageTraceStored, restStored⟩ :=
      except_bind_eq_ok _ _ _ success
    exact ⟨stage, stageStored, stageTrace, stageTraceStored, restStored⟩

/- A successful stage exposes the exact root-scope call, including the original fuel and empty
   top-level occurrence path. -/
theorem evalStage_success_root {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage)
    (stageStored : data.stages[stageNumber]? = some stage)
    (stageTrace : StageTrace backend)
    (success : evalStage data env trace stageNumber stage stageStored = .ok stageTrace) :
    ∃ root, ∃ rootStored : scopeAt stage stage.root = some root, ∃ result,
      evalScope data env {} trace stageNumber stage stage.root root stageStored rootStored
        #[] #[] 0 #[] (evaluationFuel data) = .ok result ∧
      stageTrace = { stage := stageNumber, scopes := result.scopes } := by
  rw [evalStage] at success
  split at success
  · contradiction
  · rename_i root rootStored
    obtain ⟨result, resultStored, resultEq⟩ := except_bind_eq_ok _ _ _ success
    have stageTraceEq : stageTrace = { stage := stageNumber, scopes := result.scopes } := by
      cases resultEq
      rfl
    exact ⟨root, rootStored, result, resultStored, stageTraceEq⟩

/- The public evaluator is definitionally the zero-index `evalStages` call.  This bridge keeps
   downstream proofs attached to the exported API before applying `evalStages_success_step`. -/
theorem eval_success_stages {backend : SemanticBackend} (program : Program)
    (env : EvalEnv backend program.data) (trace : Trace backend)
    (success : eval backend program env = .ok trace) :
    evalStages program.data env 0 { stages := #[] } = .ok trace :=
  success

/- A successful backend decomposition exposes the certificate attached to the exact canonical
   gadget, target, and returned preimage.  Eval uses this call directly in its gadget branch. -/
theorem gadgetDecompose_success_certificate {backend : SemanticBackend}
    (targetType : MatrixType) (layout : GadgetLayout) (structural : StructuralEnv)
    (gadget : backend.denoteMatrix (gadgetMatrixType targetType layout))
    (target : backend.denoteMatrix targetType)
    (result : Σ preimage : backend.denotePreimage (gadgetPreimageType targetType layout),
      PLift (backend.gadgetCertificate targetType layout structural gadget target preimage))
    (success : backend.gadgetDecompose targetType layout structural gadget target = .ok result) :
    backend.gadgetCertificate targetType layout structural gadget target result.1 := by
  exact result.2.down

end
end IR
end Mxx
