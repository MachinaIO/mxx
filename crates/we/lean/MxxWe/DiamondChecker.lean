import Mxx.Certificate
import MxxWe.Generated.DiamondWeFamily.Ir

open MxxWe.Generated.DiamondWeFamily

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

private def parseEnvironment (args : List String) : IO Mxx.Ir.ParamEnvironment := do
  if args.length != 17 then
    throw <| IO.userError "expected 13 integer arguments and two exact numerator/denominator pairs"
  match args with
  | [instanceWidth, witnessWidth, depth, maxLayerWidth, ringDimension, inputCount, digitBase,
      batchBits, digitCount, modulus, gadgetBase, errorBound, preimageBound,
      trapdoorNumerator, trapdoorDenominator, errorNumerator, errorDenominator] =>
      return [
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
  | _ =>
      throw <| IO.userError "internal argument-count mismatch"

private def checkEnvironment
    (analysis : Mxx.Certificate.AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment) : Bool :=
  match Mxx.Certificate.checkStaticParameters analysis environment with
  | .ok _ => true
  | .error _ => false

private def analyzeDiamond : Except String Mxx.Certificate.AnalysisResult := do
  let certificate : Mxx.Certificate.SparseCertificate := { overrides := [] }
  Mxx.Certificate.analyzeProtocol DiamondWeFamily_protocol certificate
    |>.mapError fun error => s!"Diamond correctness analysis failed: {describeVerifyError error}"

/-- Runs the pure analysis on a dedicated task while keeping stderr responsive.
The server protocol itself remains strictly stdout-only. -/
private def analyzeDiamondWithProgress : IO (Except String Mxx.Certificate.AnalysisResult) := do
  IO.eprintln "Diamond correctness analysis: started"
  let task ← IO.asTask (prio := .dedicated) (pure analyzeDiamond)
  let mut elapsedSeconds := 0
  while (← IO.getTaskState task) != .finished do
    IO.sleep 1000
    elapsedSeconds := elapsedSeconds + 1
    if elapsedSeconds % 30 == 0 && (← IO.getTaskState task) != .finished then
      IO.eprintln s!"Diamond correctness analysis: still running ({elapsedSeconds}s elapsed)"
  match ← IO.wait task with
  | .ok result =>
      IO.eprintln s!"Diamond correctness analysis: completed after {elapsedSeconds}s"
      pure result
  | .error error =>
      let message := s!"Diamond correctness analysis: interrupted: {error}"
      IO.eprintln message
      pure (.error message)

private def checkArguments
    (analysis : Mxx.Certificate.AnalysisResult)
    (args : List String) : IO Bool := do
  let environment ← parseEnvironment args
  return checkEnvironment analysis environment

private def server : IO UInt32 := do
  let analysisResult ← analyzeDiamondWithProgress
  let analysis ← match analysisResult with
    | .ok analysis => pure analysis
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
        let accepted ← checkArguments analysis tokens
        output.putStr (if accepted then "true\n" else "false\n")
        output.flush
  return 0

def main (args : List String) : IO UInt32 := do
  if args = ["--server"] then
    return ← server
  let analysisResult ← analyzeDiamondWithProgress
  let analysis ← match analysisResult with
    | .ok analysis => pure analysis
    | .error error =>
        IO.eprintln error
        return 1
  let accepted ← try checkArguments analysis args
    catch error =>
      IO.eprintln error.toString
      return 2
  IO.println (if accepted then "true" else "false")
  return 0
