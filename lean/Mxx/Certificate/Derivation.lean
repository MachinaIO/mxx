import Mxx.Ir

namespace Mxx.Certificate

/-- A generator-selected, locally checked rule.  The generator chooses a rule only where the
frozen node admits it; Lean checks every premise before accepting the step. -/
inductive DerivationRule where
  | input
  | constantInt
  | evaluateInt
  | constantReal
  | constantBool
  | zeroMatrix
  | identityMatrix
  | constantMatrix
  | unitRowMatrix
  | unitColumnMatrix
  | gadgetMatrix
  | smallGadgetMatrix
  | powerOfBaseMatrix
  | rotationMatrix
  | gadgetTrapdoor
  | intToReal
  | boolToInt
  | intBinary
  | realBinary
  | realSqrt
  | intCompare
  | bitExtract
  | extractCoefficient
  | constantCoefficient
  | select
  | uniformResidueSample
  | uniformIntervalSample
  | gaussianSample
  | hashSample
  | gadgetDecompose
  | trapdoorSample
  | trapdoorPublic
  | preimageSample
  | matrixAdd
  | matrixSubtract
  | matrixMultiplyBound
  | matrixMultiplyPreimage (preimage : Mxx.Ir.WireRef)
  | matrixNegate
  | matrixScale
  | transpose
  | slice
  | tensor
  | reshape
  | concat
  | thresholdDecodeBool
  | thresholdDecodeInt
  | crtRecompose
  | packPolynomialCoefficients
  | familyPack
  | familyGetStatic
  | familyGetDynamic
  | subgraphCall
  | parallelLoop
  | sequentialLoop
  deriving BEq, DecidableEq

/-- One untrusted instruction in the canonical node order of a frozen scope.  Only the selected
rule and operands are repeated.  Output arity and types are read directly from the frozen node,
which avoids duplicating potentially large type expressions in every derivation program. -/
structure NodeDerivation where
  sourceNode : Nat
  rule : DerivationRule
  arguments : List Mxx.Ir.WireRef
  deriving BEq, DecidableEq

structure ScopeDerivation where
  steps : List NodeDerivation
  deriving BEq, DecidableEq

structure ProgramDerivation where
  root : ScopeDerivation
  definitions : List (String × ScopeDerivation) := []
  deriving BEq, DecidableEq

inductive DerivationError where
  | missingNode (expected : Nat)
  | unexpectedInstruction (sourceNode : Nat)
  | sourceNodeMismatch (expected actual : Nat)
  | operandMismatch (node : Nat)
  | forwardOperand (node : Nat) (operand : Mxx.Ir.WireRef)
  | ruleMismatch (node : Nat) (rule : DerivationRule)
  | invalidPreimageRelation (node : Nat) (preimage : Mxx.Ir.WireRef)
  | definitionMismatch (expected actual : String)
  | missingDefinition (expected : String)
  | unexpectedDefinition (actual : String)
  deriving BEq, DecidableEq

private def matchesNodeKind : DerivationRule → Mxx.Ir.NodeKind → Bool
  | .input, .input _ => true
  | .constantInt, .constantInt _ => true
  | .evaluateInt, .evaluateInt _ => true
  | .constantReal, .constantReal _ => true
  | .constantBool, .constantBool _ => true
  | .zeroMatrix, .zeroMatrix _ => true
  | .identityMatrix, .identityMatrix _ => true
  | .constantMatrix, .constantMatrix _ _ => true
  | .unitRowMatrix, .unitRowMatrix _ _ => true
  | .unitColumnMatrix, .unitColumnMatrix _ _ => true
  | .gadgetMatrix, .gadgetMatrix _ _ => true
  | .smallGadgetMatrix, .smallGadgetMatrix _ _ => true
  | .powerOfBaseMatrix, .powerOfBaseMatrix _ _ _ => true
  | .rotationMatrix, .rotationMatrix _ _ => true
  | .gadgetTrapdoor, .gadgetTrapdoor _ _ => true
  | .intToReal, .intToReal => true
  | .boolToInt, .boolToInt => true
  | .intBinary, .intBinary _ => true
  | .realBinary, .realBinary _ => true
  | .realSqrt, .realSqrt => true
  | .intCompare, .intCompare _ => true
  | .bitExtract, .bitExtract _ => true
  | .extractCoefficient, .extractCoefficient _ => true
  | .constantCoefficient, .constantCoefficient _ => true
  | .select, .select => true
  | .uniformResidueSample, .uniformResidueSample _ => true
  | .uniformIntervalSample, .uniformIntervalSample _ _ _ => true
  | .gaussianSample, .gaussianSample _ _ => true
  | .hashSample, .hashSample _ _ _ _ _ _ _ _ => true
  | .gadgetDecompose, .gadgetDecompose _ _ _ => true
  | .trapdoorSample, .trapdoorSample _ _ => true
  | .trapdoorPublic, .trapdoorPublic => true
  | .preimageSample, .preimageSample _ _ => true
  | .matrixAdd, .matrixAdd => true
  | .matrixSubtract, .matrixSubtract => true
  | .matrixMultiplyBound, .matrixMultiply => true
  | .matrixMultiplyPreimage _, .matrixMultiply => true
  | .matrixNegate, .matrixNegate => true
  | .matrixScale, .matrixScale _ => true
  | .transpose, .transpose => true
  | .slice, .slice _ _ => true
  | .tensor, .tensor => true
  | .reshape, .reshape _ _ => true
  | .concat, .concat _ => true
  | .thresholdDecodeBool, .thresholdDecodeBool _ _ _ => true
  | .thresholdDecodeInt, .thresholdDecodeInt _ _ _ => true
  | .crtRecompose, .crtRecompose _ _ => true
  | .packPolynomialCoefficients, .packPolynomialCoefficients _ _ => true
  | .familyPack, .familyPack => true
  | .familyGetStatic, .familyGetStatic _ => true
  | .familyGetDynamic, .familyGetDynamic => true
  | .subgraphCall, .subgraphCall _ _ => true
  | .parallelLoop, .parallelLoop _ _ _ _ _ => true
  | .sequentialLoop, .sequentialLoop _ _ _ _ _ => true
  | _, _ => false

private def validPreimageRelation
    (previous : Array Mxx.Ir.Node)
    (node : Mxx.Ir.Node)
    (preimage : Mxx.Ir.WireRef) : Bool :=
  match previous[preimage.node]? with
  | some source =>
      let sourceIsPreimage := match source.kind with
        | .preimageSample _ _ => true
        | _ => false
      preimage.port < source.outputCount && sourceIsPreimage && node.arguments.contains preimage
  | none => false

private def checkNodeDerivation
    (previous : Array Mxx.Ir.Node)
    (nodeIndex : Nat)
    (node : Mxx.Ir.Node)
    (step : NodeDerivation) : Except DerivationError Unit := do
  if step.sourceNode != nodeIndex then
    throw (.sourceNodeMismatch nodeIndex step.sourceNode)
  if step.arguments != node.arguments then
    throw (.operandMismatch nodeIndex)
  for operand in node.arguments do
    if operand.node >= nodeIndex then
      throw (.forwardOperand nodeIndex operand)
  if !matchesNodeKind step.rule node.kind then
    throw (.ruleMismatch nodeIndex step.rule)
  match step.rule with
  | .matrixMultiplyPreimage preimage =>
      if !validPreimageRelation previous node preimage then
        throw (.invalidPreimageRelation nodeIndex preimage)
  | _ => pure ()

private def checkScopeSteps
    (nodes : List Mxx.Ir.Node)
    (steps : List NodeDerivation)
    (nodeIndex : Nat := 0)
    (previous : Array Mxx.Ir.Node := #[]) : Except DerivationError Unit :=
  match nodes, steps with
  | [], [] => .ok ()
  | [], extra :: _ => .error (.unexpectedInstruction extra.sourceNode)
  | _ :: _, [] => .error (.missingNode nodeIndex)
  | node :: remainingNodes, step :: remainingSteps => do
      checkNodeDerivation previous nodeIndex node step
      checkScopeSteps remainingNodes remainingSteps (nodeIndex + 1) (previous.push node)

def checkScopeDerivation
    (scope : Mxx.Ir.Scope)
    (derivation : ScopeDerivation) : Except DerivationError Unit :=
  checkScopeSteps scope.nodes derivation.steps

private def checkDefinitions :
    List (String × Mxx.Ir.Scope) → List (String × ScopeDerivation) → Except DerivationError Unit
  | [], [] => .ok ()
  | [], (actual, _) :: _ => .error (.unexpectedDefinition actual)
  | (expected, _) :: _, [] => .error (.missingDefinition expected)
  | (expectedName, expectedScope) :: remainingDefinitions,
      (actualName, actualDerivation) :: remainingDerivations => do
      if expectedName != actualName then
        throw (.definitionMismatch expectedName actualName)
      checkScopeDerivation expectedScope actualDerivation
      checkDefinitions remainingDefinitions remainingDerivations

def checkProgramDerivation
    (program : Mxx.Ir.Prog)
    (derivation : ProgramDerivation) : Except DerivationError Unit := do
  checkScopeDerivation program.root derivation.root
  checkDefinitions program.definitions derivation.definitions

/-- The intended semantic connection for the structural checker.  This claim deliberately uses
the same program and derivation types as the operational checker; later local rule proofs must
establish it rather than introducing a second proof-only representation. -/
def StructuralDerivationSoundnessClaim : Prop :=
  ∀ (program : Mxx.Ir.Prog) (derivation : ProgramDerivation),
    checkProgramDerivation program derivation = .ok () →
      derivation.root.steps.length = program.root.nodes.length ∧
        derivation.definitions.length = program.definitions.length

private def fixtureType : Mxx.Ir.MatrixTypeExpr := {
  modulus := .constant 17
  ringDimension := .constant 1
  rows := .constant 1
  columns := .constant 1
}

private def fixtureScope : Mxx.Ir.Scope := {
  nodes := [
    { kind := .input "left", arguments := [], outputCount := 1,
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [], outputCount := 1,
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputCount := 1, outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 2, port := 0 })],
  inputNames := ["left"]
}

private def fixtureDerivation : ScopeDerivation := {
  steps := [
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ]
}

example : checkScopeDerivation fixtureScope fixtureDerivation = .ok () := by
  rfl

private def operandMismatchFixture : ScopeDerivation := {
  steps := [
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .matrixAdd,
      arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }] }
  ]
}

example : checkScopeDerivation fixtureScope operandMismatchFixture =
    .error (.operandMismatch 2) := by
  rfl

example : checkScopeDerivation fixtureScope {
  fixtureDerivation with steps := fixtureDerivation.steps.take 2
} = .error (.missingNode 2) := by
  rfl

private def reorderedFixture : ScopeDerivation := {
  steps := [
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ]
}

example : checkScopeDerivation fixtureScope reorderedFixture =
    .error (.sourceNodeMismatch 0 1) := by
  rfl

private def preimageFixtureScope : Mxx.Ir.Scope := {
  nodes := [
    { kind := .input "source", arguments := [], outputCount := 1,
      outputTypes := [.matrix fixtureType] },
    { kind := .preimageSample fixtureType (.constant 2), arguments := [{ node := 0, port := 0 }],
      outputCount := 1, outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputCount := 1, outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 2, port := 0 })],
  inputNames := ["source"]
}

private def preimageFixtureDerivation : ScopeDerivation := {
  steps := [
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .preimageSample, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .matrixMultiplyPreimage { node := 0, port := 0 },
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ]
}

example : checkScopeDerivation preimageFixtureScope preimageFixtureDerivation =
    .error (.invalidPreimageRelation 2 { node := 0, port := 0 }) := by
  rfl

end Mxx.Certificate
