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
  | .trapdoorPublicMismatch wire _ _ =>
      s!"trapdoor/public mismatch {wire.stage.name}:{wire.node.value}:{wire.port}"
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
  | .escapedCarriedInput stage node slot =>
      s!"escaped carried input {stage.name}:{node.value}:{slot}"
  | .unsupportedCarriedKind stage node slot =>
      s!"unsupported carried kind {stage.name}:{node.value}:{slot}"
  | .relationBearingCarriedMatrix stage node slot =>
      s!"relation-bearing carried matrix {stage.name}:{node.value}:{slot}"
  | .invalidExpressionReference => "invalid expression reference"
  | .scalarControl _ => "unsupported or invalid scalar-control operation"
  | .matrixAffine stage scope node _ =>
      s!"unsupported or invalid affine matrix operation at " ++
        s!"{stage.name}:{repr scope.path}:{node.value}"
  | .matrixSelect wire _ => s!"unsupported or invalid matrix selection at {reprStr wire}"
  | .transform _ => "unsupported or invalid matrix transform"
  | .affineNormalize wire _ =>
      s!"affine normalization failed at {reprStr wire}"
  | _ => "unsupported correctness-analysis error"

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

def main (args : List String) : IO UInt32 := do
  if args.length != 17 then
    IO.eprintln "expected 13 integer arguments and two exact numerator/denominator pairs"
    return 2
  match args with
  | [instanceWidth, witnessWidth, depth, maxLayerWidth, ringDimension, inputCount, digitBase,
      batchBits, digitCount, modulus, gadgetBase, errorBound, preimageBound,
      trapdoorNumerator, trapdoorDenominator, errorNumerator, errorDenominator] =>
      let environment : Mxx.Ir.ParamEnvironment := [
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
      let certificate : Mxx.Certificate.SparseCertificate :=
        { overrides := [] }
      let accepted ←
        match Mxx.Certificate.analyzeProtocol DiamondWeFamily_protocol certificate with
        | .error error => do
            IO.eprintln s!"Diamond correctness analysis failed: {describeVerifyError error}"
            pure false
        | .ok analysis =>
            match Mxx.Certificate.checkStaticParameters analysis environment with
            | .error _ => do
                IO.eprintln "Diamond correctness static-parameter checking failed"
                pure false
            | .ok _ => pure true
      IO.println (if accepted then "true" else "false")
      return 0
  | _ =>
      IO.eprintln "internal argument-count mismatch"
      return 2
