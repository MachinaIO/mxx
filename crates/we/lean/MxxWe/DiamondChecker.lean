import Mxx.Certificate.Derivation
import Mxx.Certificate.OperationalBounds
import MxxWe.Generated.DiamondWeFamily.Ir

open MxxWe.Generated.DiamondWeFamily

private def describeDerivationError : Mxx.Certificate.DerivationError → String
  | .missingNode _ => "derivation is missing a frozen node"
  | .unexpectedInstruction _ => "derivation contains an extra instruction"
  | .sourceNodeMismatch _ _ => "derivation instruction order does not match the frozen graph"
  | .operandMismatch _ => "derivation operands do not match the frozen graph"
  | .forwardOperand _ _ => "derivation uses a forward operand"
  | .ruleMismatch _ _ => "derivation rule does not match the frozen node"
  | .invalidRelationOperand _ _ => "derivation relation operand is invalid"
  | .definitionMismatch _ _ => "derivation definitions do not match the frozen graph"
  | .missingDefinition _ => "derivation is missing a frozen definition"
  | .unexpectedDefinition _ => "derivation contains an extra definition"

/-- Validates the generated, untrusted derivation skeleton without invoking the
legacy whole-graph analyzer.  This is deliberately the first checker gate. -/
private def checkGeneratedDerivations : Except String Unit := do
  Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_stage_encrypt DiamondWeFamily_stage_encrypt_derivation
    |>.mapError fun error => s!"stage:encrypt: {describeDerivationError error}"
  Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_stage_decrypt DiamondWeFamily_stage_decrypt_derivation
    |>.mapError fun error => s!"stage:decrypt: {describeDerivationError error}"
  Mxx.Certificate.checkProgramDerivation DiamondWeFamily_ideal DiamondWeFamily_ideal_derivation
    |>.mapError fun error => s!"ideal: {describeDerivationError error}"
  Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_requirement_0 DiamondWeFamily_requirement_0_derivation
    |>.mapError fun error => s!"requirement-0: {describeDerivationError error}"
  Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_requirement_1 DiamondWeFamily_requirement_1_derivation
    |>.mapError fun error => s!"requirement-1: {describeDerivationError error}"
  Mxx.Certificate.checkProgramDerivation
    DiamondWeFamily_requirement_2 DiamondWeFamily_requirement_2_derivation
    |>.mapError fun error => s!"requirement-2: {describeDerivationError error}"

private def checkDerivationWithProgress
    (name : String)
    (result : Except Mxx.Certificate.DerivationError Unit) : IO (Except String Unit) := do
  IO.eprintln s!"Diamond derivation check: {name} started"
  let task ← IO.asTask (prio := .dedicated) (pure result)
  let mut elapsedSeconds := 0
  while (← IO.getTaskState task) != .finished do
    IO.sleep 1000
    elapsedSeconds := elapsedSeconds + 1
    if elapsedSeconds % 30 == 0 && (← IO.getTaskState task) != .finished then
      IO.eprintln s!"Diamond derivation check: {name} still running ({elapsedSeconds}s elapsed)"
  match ← IO.wait task with
  | .ok (.ok ()) =>
      IO.eprintln s!"Diamond derivation check: {name} completed after {elapsedSeconds}s"
      pure (.ok ())
  | .ok (.error error) =>
      pure (.error s!"Diamond derivation check failed for {name}: {describeDerivationError error}")
  | .error error =>
      pure (.error s!"Diamond derivation check interrupted for {name}: {error}")

private def checkGeneratedDerivationsWithProgress : IO (Except String Unit) := do
  match ← checkDerivationWithProgress "stage:encrypt"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_stage_encrypt DiamondWeFamily_stage_encrypt_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkDerivationWithProgress "stage:decrypt"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_stage_decrypt DiamondWeFamily_stage_decrypt_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkDerivationWithProgress "ideal"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_ideal DiamondWeFamily_ideal_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkDerivationWithProgress "requirement-0"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_requirement_0 DiamondWeFamily_requirement_0_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkDerivationWithProgress "requirement-1"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_requirement_1 DiamondWeFamily_requirement_1_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  match ← checkDerivationWithProgress "requirement-2"
      (Mxx.Certificate.checkProgramDerivation
        DiamondWeFamily_requirement_2 DiamondWeFamily_requirement_2_derivation) with
  | .error error => return .error error
  | .ok () => pure ()
  pure (.ok ())

/- Obsolete whole-graph analyzer diagnostics.  The active checker reports derivation and
operational-bound errors directly, so this former diagnostic mapping is intentionally inactive.
private def describeVerifyError : Mxx.Certificate.VerifyError → String
  | .disabledRule _ => "disabled rule"
  | .unsupportedNode stage node =>
      s!"unsupported node {stage.name}:{node.value}"
  | .unsupportedNodeInScope stage scope node =>
      s!"unsupported node {stage.name}:{repr scope.path}:{node.value}"
  | .unsupportedDefinition stage name =>
      s!"unsupported definition {stage.name}:{name}"
  | .missingInputFact stage node input =>
      s!"missing input fact {stage.name}:{node.value} <- {input.node}:{input.port}"
  | .expectedMatrixFact wire =>
      s!"expected matrix fact {wire.stage.name}:{String.intercalate "/" wire.scope.path}:" ++
        s!"{wire.node.value}:{wire.port}"
  | .expectedTrapdoorFact wire =>
      s!"expected trapdoor fact {wire.stage.name}:{wire.node.value}:{wire.port}"
  | .trapdoorPublicMismatch wire expected actual sourcePrimary =>
      s!"trapdoor/public mismatch {wire.stage.name}:{wire.node.value}:{wire.port}; " ++
        s!"trapdoor public={repr expected}, source={repr actual}, " ++
          s!"source primary={repr sourcePrimary}"
  | .missingAnchorBinding _ => "missing semantic-anchor binding"
  | .invalidAnchorWire _ _ => "invalid semantic-anchor wire"
  | .unsupportedOverride _ => "unsupported certificate override"
  | .mismatchedMatrixTypes _ _ => "mismatched matrix types"
  | .expectedIntegerFact wire =>
      s!"expected integer fact {wire.stage.name}:{wire.node.value}:{wire.port}"
  | .expectedBooleanFact wire =>
      s!"expected Boolean fact {wire.stage.name}:{wire.node.value}:{wire.port}"
  | .missingInputContract name => s!"missing input contract {name}"
  | .missingProgramInput stage name => s!"missing program input {stage.name}:{name}"
  | .missingArtifactOutput stage name => s!"missing artifact output {stage.name}:{name}"
  | .invalidInputCoverage _ => "invalid input coverage"
  | .invalidInputDestination _ => "invalid input destination"
  | .invalidEndpointCoverage _ => "invalid endpoint coverage"
  | .invalidEndpointConnection _ => "invalid endpoint connection"
  | .diamondEndpoint error => s!"invalid Diamond endpoint: {repr error}"
  | .frozenRecurrenceInterface recurrence error =>
      s!"invalid frozen recurrence {repr recurrence}: {repr error}"
  | .bggRecurrencePrefilter error => s!"invalid BGG recurrence prefilter: {repr error}"
  | .bggCarriedRoleInference error => s!"invalid BGG carried-role inference: {repr error}"
  | .bggThreeTraceInterface error => s!"invalid BGG three-trace interface: {repr error}"
  | .invalidPreconditionSpec => "invalid precondition specification"
  | .duplicateInputId _ => "duplicate input id"
  | .duplicateInputName name => s!"duplicate input name {name}"
  | .duplicateInputDestination _ => "duplicate input destination"
  | .unboundProgramInput stage name => s!"unbound program input {stage.name}:{name}"
  | .duplicateEndpointSpec _ => "duplicate endpoint specification"
  | .invalidComparatorPolarity _ => "invalid comparator polarity"
  | .nonBooleanOutput stage name => s!"non-Boolean output {stage.name}:{name}"
  | .invalidEndpointAnchorArity _ => "invalid endpoint anchor arity"
  | .missingOrInvalidOutputTypes stage node =>
      s!"missing or invalid output types {stage.name}:{node.value}"
  | .inputContractTypeMismatch _ stage name =>
      s!"input-contract type mismatch {stage.name}:{name}"
  | .duplicateParameter name => s!"duplicate parameter {name}"
  | .missingParameterDeclaration name => s!"missing parameter declaration {name}"
  | .parameterKindMismatch name => s!"parameter-kind mismatch {name}"
  | .typing _ => "matrix typing error"
  | .exactLeftAffineRightProduct stage node =>
      s!"exact-left affine-right product {stage.name}:{node.value}"
  | .generalAffineProduct stage node =>
      s!"general affine product {stage.name}:{node.value}"
  | .missingFamily joint => s!"missing family {joint.name}"
  | .invalidFamilySlot _ slot => s!"invalid family slot {slot}"
  | .invalidLoopDefinition stage name =>
      s!"invalid loop definition {stage.name}:{name}"
  | .invalidLoopArity stage node =>
      s!"invalid loop arity {stage.name}:{node.value}"
  | .invalidLoopArityInScope stage scope node =>
      s!"invalid loop arity {stage.name}:{reprStr scope.path}:{node.value}"
  | .unsupportedSequentialRecurrence stage node =>
      s!"unsupported sequential recurrence {stage.name}:{node.value}"
  | .escapedCarriedInput stage node slot =>
      s!"escaped carried input {stage.name}:{node.value}:{slot}"
  | .unsupportedCarriedKind stage node slot =>
      s!"unsupported carried kind {stage.name}:{node.value}:{slot}"
  | .nonUniformNestedRecurrenceInput stage node slot =>
      s!"non-uniform nested recurrence input {stage.name}:{node.value}:{slot}"
  | .relationBearingCarriedMatrix stage node slot =>
      s!"relation-bearing carried matrix {stage.name}:{node.value}:{slot}"
  | .invalidExpressionReference detail => s!"invalid expression reference: {detail}"
  | .scalarControl _ => "unsupported or invalid scalar-control operation"
  | .matrixAffine stage scope node _ =>
      s!"unsupported or invalid affine matrix operation at " ++
        s!"{stage.name}:{repr scope.path}:{node.value}"
  | .matrixSelect wire _ => s!"unsupported or invalid matrix selection at {reprStr wire}"
  | .transform _ => "unsupported or invalid matrix transform"
  | .affineNormalize wire _ =>
      s!"affine normalization failed at {reprStr wire}"
  | .symbolicEvaluation error => s!"symbolic evaluation construction failed: {repr error}"
  | .symbolicRecurrence error => s!"symbolic recurrence construction failed: {repr error}"
-/

private def parseNat (value : String) : IO Nat :=
  match value.toNat? with
  | some result => pure result
  | none => throw <| IO.userError s!"invalid natural number: {value}"

private def parseInt (value : String) : IO Int :=
  match value.toInt? with
  | some result => pure result
  | none => throw <| IO.userError s!"invalid integer: {value}"

private def parseDimension (value : String) : IO Int :=
  return Int.ofNat (← parseNat value)

private def parseRat (numerator denominator : String) : IO Rat := do
  let numerator ← parseInt numerator
  let denominator ← parseNat denominator
  if denominator = 0 then
    throw <| IO.userError "a rational denominator must be positive"
  pure ((numerator : Rat) / (denominator : Rat))

private structure CheckerRequest where
  environment : Mxx.Ir.ParamEnvironment
  layouts : List Mxx.GadgetLayoutDescriptor
  targetId : String
  requestHash : String

private def parseCrtModuli (value : String) : IO (List Nat) :=
  value.splitOn "," |>.mapM parseNat

private def parseRequest (args : List String) : IO CheckerRequest := do
  if args.length != 28 then
    throw <| IO.userError "expected 17 scalar arguments, one 9-field gadget-layout descriptor, a target id, and a request hash"
  match args with
  | [instanceWidth, witnessWidth, depth, maxLayerWidth, ringDimension, inputCount, digitBase,
      batchBits, digitCount, modulus, gadgetBase, errorBound, preimageBound,
      trapdoorNumerator, trapdoorDenominator, errorNumerator, errorDenominator,
      paramsId, layoutRingDimension, crtBits, baseBits, layoutBase, regularDigitCount,
      smallDigitCount, smallestCrtModulus, crtModuli, targetId, requestHash] =>
      return {
        environment := [
        (.parameter "instance_width", .integer (← parseDimension instanceWidth)),
        (.parameter "witness_width", .integer (← parseDimension witnessWidth)),
        (.parameter "depth", .integer (← parseDimension depth)),
        (.parameter "max_layer_width", .integer (← parseDimension maxLayerWidth)),
        (.parameter "diamond_ring_dimension", .integer (← parseDimension ringDimension)),
        (.parameter "diamond_input_count", .integer (← parseDimension inputCount)),
        (.parameter "diamond_digit_base", .integer (← parseDimension digitBase)),
        (.parameter "diamond_batch_bits", .integer (← parseDimension batchBits)),
        (.parameter "diamond_digit_count", .integer (← parseDimension digitCount)),
        (.parameter "diamond_modulus", .integer (← parseInt modulus)),
        (.parameter "diamond_gadget_base", .integer (← parseInt gadgetBase)),
        (.parameter "diamond_error_max_coefficient_bound", .integer (← parseInt errorBound)),
        (.parameter "diamond_preimage_max_coefficient_bound",
          .integer (← parseInt preimageBound)),
        (.parameter "diamond_trapdoor_sigma",
          .rational (← parseRat trapdoorNumerator trapdoorDenominator)),
          (.parameter "diamond_error_sigma",
            .rational (← parseRat errorNumerator errorDenominator))
        ]
        layouts := [{
          paramsId
          ringDimension := ← parseNat layoutRingDimension
          crtModuli := ← parseCrtModuli crtModuli
          crtBits := ← parseNat crtBits
          baseBits := ← parseNat baseBits
          base := ← parseInt layoutBase
          regularDigitCount := ← parseNat regularDigitCount
          smallDigitCount := ← parseNat smallDigitCount
          smallestCrtModulus := ← parseNat smallestCrtModulus
        }]
        targetId
        requestHash
      }
  | _ =>
      throw <| IO.userError "internal argument-count mismatch"

private def describeOperationalError : Mxx.Certificate.OperationalError → String
  | .inScope scope error => s!"in {repr scope}: {describeOperationalError error}"
  | .missingOutputType node port => s!"missing output type {node}:{port}"
  | .missingOperand node operand => s!"missing operand {node}:{operand.node}:{operand.port}"
  | .operandNotMatrix node operand => s!"non-matrix operand {node}:{operand.node}:{operand.port}"
  | .operandNotInteger node operand =>
      s!"non-integer operand {node}:{operand.node}:{operand.port}"
  | .operandNotBoolean node operand =>
      s!"non-Boolean operand {node}:{operand.node}:{operand.port}"
  | .operandNotReal node operand =>
      s!"non-real operand {node}:{operand.node}:{operand.port}"
  | .invalidMatrixParameters node => s!"invalid matrix parameters at {node}"
  | .flat node error => s!"flat operational error at {node}: {repr error}"
  | .invalidBound node bound => s!"invalid bound {bound} at {node}"
  | .missingPreimageCutoff node => s!"missing preimage cutoff at {node}"
  | .preimageCutoffMismatch node => s!"preimage cutoff mismatch at {node}"
  | .invalidCount node count => s!"invalid count {count} at {node}"
  | .missingGadgetLayout node => s!"missing gadget layout at {node}"
  | .ambiguousGadgetLayout node => s!"ambiguous gadget layout at {node}"
  | .invalidGadgetLayout node => s!"invalid gadget layout at {node}"
  | .gadgetLayoutMismatch node => s!"gadget layout mismatch at {node}"
  | .missingPublicIdentity node wire =>
      s!"missing public identity at {node} for {wire.node}:{wire.port}"
  | .publicIdentityMismatch node => s!"public identity mismatch at {node}"
  | .missingRelation node wire => s!"missing relation at {node} for {wire.node}:{wire.port}"
  | .ambiguousRelation node wire =>
      s!"ambiguous relation at {node} for {wire.node}:{wire.port}"
  | .unavailableRelation node wire =>
      s!"unavailable relation at {node} for {wire.node}:{wire.port}"
  | .malformedRelation node => s!"malformed relation at {node}"
  | .missingDefinition name => s!"missing frozen definition {name}"
  | .definitionFuelExhausted => "frozen definition nesting exceeds the checked program budget"
  | .childInputMismatch node expected actual =>
      s!"child input mismatch at {node}: expected {expected}, got {actual}"
  | .duplicateInputName name => s!"duplicate frozen input name {name}"
  | .missingInputNode name => s!"missing frozen input node {name}"
  | .unexpectedInputNode name => s!"unexpected frozen input node {name}"
  | .missingChildOutput node port => s!"missing child output {node}:{port}"
  | .loopInputModeMismatch node argument =>
      s!"loop input mode mismatch at {node}, argument {argument}"
  | .relationBearingCarriedValue scope node slot =>
      s!"relation-bearing sequential carry in {repr scope} at {node}, slot {slot}"
  | .sequentialSchemaMismatch scope node slot initial output =>
      s!"sequential carry schema changed in {repr scope} at {node}, slot {slot}\n" ++
        s!"initial large-factor counts: {initial}\noutput large-factor counts: {output}"
  | .divisionByZero => "division by zero"
  | .negativeDenominator value => s!"negative denominator {value}"
  | .invalidPreviousPath path => s!"invalid recurrence-state path: {repr path}"
  | .nonClosedExpression => "non-closed operational bound expression"
  | .derivation error => s!"invalid derivation: {describeDerivationError error}"
  | .unsupportedOutputArity node actual => s!"invalid output arity {actual} at {node}"
  | .outputTypeMismatch node => s!"output type mismatch at {node}"
  | .missingStageDerivation stage => s!"missing workflow derivation for {stage}"
  | .missingStageResult stage output => s!"missing workflow artifact {stage}.{output}"
  | .invalidOperationalDecoderTarget targetId =>
      s!"invalid operational decoder target {targetId}"
  | .unknownOperationalDecoderTarget targetId =>
      s!"unknown operational decoder target {targetId}"
  | .emptyOperationalDecoderTargetRegistry =>
      "the operational decoder target registry is empty"
  | .duplicateOperationalDecoderTarget targetId =>
      s!"duplicate operational decoder target {targetId}"
  | .missingProtocolContract name => s!"missing protocol input contract for {name}"
  | .inputContractMismatch detail => s!"protocol input contract mismatch: {detail}"
  | .unknownDerivationAttachment ownerNamespace ruleName =>
      s!"unknown derivation attachment {ownerNamespace}.{ruleName}"
  | .missingDerivationAttachmentRole ownerNamespace ruleName roleName =>
      s!"derivation attachment {ownerNamespace}.{ruleName} is missing role {roleName}"
  | .invalidDerivationAttachment ownerNamespace ruleName =>
      s!"invalid derivation attachment {ownerNamespace}.{ruleName}"
  | .invalidOperationalExprRef id => s!"invalid operational expression reference {id}"
  | .operationalExprTypeMismatch left right =>
      s!"operational expression type mismatch between {left} and {right}"
  | .residualContainsLargeTerm node =>
      s!"decoder residual retains a Large term at {node}"
  | .incompatibleRelationDomains node leftDomain rightDomain =>
      s!"incompatible relation domains at {node}: {leftDomain} and {rightDomain}"
  | .unknownRelationRequirement node expression =>
      s!"unknown relation requirement at {node} for expression {expression}"
  | .unresolvedConcreteStructure node expression =>
      s!"unresolved concrete structure at {node} for expression {expression}"
  | .unsupportedOperationalExpr id => s!"unsupported operational expression {id}"
  | .unsupportedNode node => s!"unsupported IR node at {node}"

/-- The operational endpoint used by parameter search while the strict correctness theorem is
unfinished.  It is intentionally named an estimate: it checks generated derivations and derives
all executable-node hard bounds, but does not claim the final runtime theorem. -/
private structure OperationalDiamondEstimate where
  accepted : Bool
  noiseBound : Int
  modulus : Int
  rejection : Option Mxx.Certificate.OperationalNoiseRejection

private def operationalDiamondEstimate
    (request : CheckerRequest) : Except String OperationalDiamondEstimate := do
  let environment := request.environment
  checkGeneratedDerivations
  let operationalWorkflow : Mxx.Certificate.OperationalWorkflowSpec := {
    workflow := DiamondWeFamily_protocol.bundle.workflow
    inputContract := DiamondWeFamily_protocol.bundle.inputContract
    operationalDecoderTargets := DiamondWeFamily_protocol.bundle.operationalDecoderTargets
  }
  let prepared ← Mxx.Certificate.prepareWorkflowOperational operationalWorkflow
      [("encrypt", DiamondWeFamily_stage_encrypt_derivation),
       ("decrypt", DiamondWeFamily_stage_decrypt_derivation)]
    |>.mapError fun error => s!"workflow operational preparation failed: {describeOperationalError error}"
  let workflowResults ← Mxx.Certificate.evaluatePreparedWorkflowOperational prepared environment request.layouts
    |>.mapError fun error => s!"workflow operational bound evaluation failed: {describeOperationalError error}"
  let (target, modulus, noiseBound, diagnostics) ←
    Mxx.Certificate.operationalTargetNoiseBound prepared workflowResults request.targetId environment
      |>.mapError describeOperationalError
  let report ← Mxx.Certificate.operationalTargetNoiseCheckReportFromBound workflowResults
      target modulus noiseBound diagnostics environment |>.mapError describeOperationalError
  let noiseBound ← match report.obligations with
    | [.booleanInterval _ noiseBound] => pure noiseBound
    | _ => throw "operational decoder report did not contain exactly one interval obligation"
  return {
    accepted := report.accepted
    noiseBound
    modulus
    rejection := report.rejection
  }

/-- Runs the pure analysis on a dedicated task while keeping stderr responsive.
The server protocol itself remains strictly stdout-only. -/
private def prepareDiamondOperationalChecker : IO (Except String Unit) := do
  IO.eprintln "Diamond operational checker: started"
  let task ← IO.asTask (prio := .dedicated) (pure checkGeneratedDerivations)
  let mut elapsedSeconds := 0
  while (← IO.getTaskState task) != .finished do
    IO.sleep 1000
    elapsedSeconds := elapsedSeconds + 1
    if elapsedSeconds % 30 == 0 && (← IO.getTaskState task) != .finished then
      IO.eprintln s!"Diamond operational checker: still running ({elapsedSeconds}s elapsed)"
  match ← IO.wait task with
  | .ok result =>
      IO.eprintln s!"Diamond operational checker: completed after {elapsedSeconds}s"
      pure result
  | .error error =>
      let message := s!"Diamond operational checker: interrupted: {error}"
      IO.eprintln message
      pure (.error message)

private def checkArguments (args : List String) : IO (Bool × String) := do
  let started ← IO.monoMsNow
  let request ← parseRequest args
  match operationalDiamondEstimate request with
  | .ok result =>
      let elapsed ← IO.monoMsNow
      IO.eprintln (s!"Diamond operational estimate: accepted={result.accepted}, " ++
        s!"noise_bound={result.noiseBound}, modulus={result.modulus}, " ++
        s!"elapsed_ms={elapsed - started}, rejection={repr result.rejection}")
      return (result.accepted, request.requestHash)
  | .error error =>
      let elapsed ← IO.monoMsNow
      IO.eprintln s!"Diamond operational estimate rejected after {elapsed - started}ms: {error}"
      return (false, request.requestHash)

private def server : IO UInt32 := do
  let preparation ← prepareDiamondOperationalChecker
  match preparation with
    | .ok () => pure ()
    | .error error =>
        IO.eprintln error
        return 1
  let input ← IO.getStdin
  let output ← IO.getStdout
  let mut running := true
  while running do
    let line ← input.getLine
    let tokens := line.trimAscii.toString.splitOn " " |>.filter fun token => !token.isEmpty
    match tokens with
    | ["quit"] => running := false
    | [] => running := false
    | _ =>
        let (accepted, requestHash) ← checkArguments tokens
        output.putStr s!"{if accepted then "true" else "false"} {requestHash}\n"
        output.flush
  return 0

def main (args : List String) : IO UInt32 := do
  IO.eprintln "Diamond checker: process initialized"
  if args = ["--derivation-only"] then
    match ← checkGeneratedDerivationsWithProgress with
    | .ok () =>
        IO.println "true"
        return 0
    | .error error =>
        IO.eprintln error
        return 1
  if args = ["--server"] then
    return ← server
  let preparation ← prepareDiamondOperationalChecker
  match preparation with
    | .ok () => pure ()
    | .error error =>
        IO.eprintln error
        return 1
  let (accepted, _) ← try checkArguments args
    catch error =>
      IO.eprintln error.toString
      return 2
  IO.println (if accepted then "true" else "false")
  return 0
