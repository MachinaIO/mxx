import Mxx.Certificate.Derivation

/-! # Linear operational hard-bound estimator

This module is deliberately separate from the symbolic certificate analyzer.  It evaluates a
frozen scope in node order, after `ProgramDerivation` has checked the generator-selected rule
and operands.  The resulting values are estimates used for parameter search; they are not yet
runtime-bound theorems.

The only recursive bound language is for a sequential-loop numeric state transition.  Ordinary
scope evaluation stores concrete integer bounds, so it neither reconstructs symbolic expressions
nor searches the graph.
-/

namespace Mxx.Certificate

open Mxx.Ir

inductive OperationalBoundPath where
  | matrixMaximum (slot : Nat)
  | integerLower (slot : Nat)
  | integerUpper (slot : Nat)
  deriving BEq, DecidableEq

inductive OperationalBoundExpr where
  | closedInt (value : IntExpr)
  | previous (path : OperationalBoundPath)
  | negate (value : OperationalBoundExpr)
  | add (left right : OperationalBoundExpr)
  | subtract (left right : OperationalBoundExpr)
  | multiply (left right : OperationalBoundExpr)
  | divide (left right : OperationalBoundExpr)
  | minimum (left right : OperationalBoundExpr)
  | maximum (left right : OperationalBoundExpr)
  | centeredCap (modulus value : OperationalBoundExpr)
  | matrixProduct
      (ringDimension innerDimension left right : OperationalBoundExpr)
  deriving BEq, DecidableEq

inductive OperationalFact where
  | matrix (matrixType : MatrixTypeExpr) (maximum : Int)
  | integer (lower upper : Int)
  | boolean
  | real
  | trapdoor (matrixType : MatrixTypeExpr) (maximum : Int)
  | family (element : OperationalFact) (count : Int)
  | bytes (length : Int)
  | typedBlob (typeName : String)
  | unknown (wireType : WireTypeExpr)
  deriving BEq, DecidableEq

abbrev OperationalState := Array OperationalFact

/-- Facts for an ordinary frozen scope are indexed by the exact `(node, port)` wire location. -/
abbrev OperationalScopeFacts := Array (Array OperationalFact)

inductive OperationalError where
  | missingOutputType (node : Nat) (port : Nat)
  | missingOperand (node : Nat) (operand : WireRef)
  | operandNotMatrix (node : Nat) (operand : WireRef)
  | invalidMatrixParameters (node : Nat)
  | invalidBound (node : Nat) (bound : Int)
  | invalidCount (node : Nat) (count : Int)
  | divisionByZero
  | negativeDenominator (value : Int)
  | invalidPreviousPath (path : OperationalBoundPath)
  | nonClosedExpression
  | derivation (error : DerivationError)
  | unsupportedOutputArity (node : Nat) (actual : Nat)
  deriving BEq, DecidableEq

def absolute (value : Int) : Int := if value < 0 then -value else value

def capCentered (modulus value : Int) : Int :=
  if modulus ≤ 0 then 0 else min (modulus / 2) (absolute value)

def matrixCap (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let modulus ← matrixType.modulus.evaluate environment
  if modulus ≤ 0 then none else some (modulus / 2)

def matrixRingDimension (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let value ← matrixType.ringDimension.evaluate environment
  if value < 0 then none else some value

def matrixInnerDimension (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let value ← matrixType.columns.evaluate environment
  if value < 0 then none else some value

def lookupPrevious (state : OperationalState) : OperationalBoundPath → Option Int
  | .matrixMaximum slot =>
      match state[slot]? with
      | some (.matrix _ maximum) => some maximum
      | some (.trapdoor _ maximum) => some maximum
      | _ => none
  | .integerLower slot =>
      match state[slot]? with
      | some (.integer lower _) => some lower
      | _ => none
  | .integerUpper slot =>
      match state[slot]? with
      | some (.integer _ upper) => some upper
      | _ => none

def intExprIsClosed : IntExpr → Bool
  | .constant _ => true
  | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprIsClosed left && intExprIsClosed right
  | .log2Ceil value => intExprIsClosed value

def OperationalBoundExpr.evaluate
    (environment : ParamEnvironment)
    (previousState : OperationalState) : OperationalBoundExpr → Except OperationalError Int
  | .closedInt value => do
      if !intExprIsClosed value then throw .nonClosedExpression
      match value.evaluate environment with
      | some result => pure result
      | none => throw .nonClosedExpression
  | .previous path =>
      match lookupPrevious previousState path with
      | some result => pure result
      | none => throw (.invalidPreviousPath path)
  | .negate value => return -(← value.evaluate environment previousState)
  | .add left right => return (← left.evaluate environment previousState) +
      (← right.evaluate environment previousState)
  | .subtract left right => return (← left.evaluate environment previousState) -
      (← right.evaluate environment previousState)
  | .multiply left right => return (← left.evaluate environment previousState) *
      (← right.evaluate environment previousState)
  | .divide left right => do
      let denominator ← right.evaluate environment previousState
      if denominator = 0 then throw .divisionByZero
      if denominator < 0 then throw (.negativeDenominator denominator)
      return (← left.evaluate environment previousState) / denominator
  | .minimum left right => do
      let left ← left.evaluate environment previousState
      let right ← right.evaluate environment previousState
      return min left right
  | .maximum left right => do
      let left ← left.evaluate environment previousState
      let right ← right.evaluate environment previousState
      return max left right
  | .centeredCap modulus value => do
      let modulus ← modulus.evaluate environment previousState
      let value ← value.evaluate environment previousState
      return capCentered modulus value
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← ringDimension.evaluate environment previousState
      let innerDimension ← innerDimension.evaluate environment previousState
      let left ← left.evaluate environment previousState
      let right ← right.evaluate environment previousState
      return ringDimension * innerDimension * left * right

def evaluateTransition
    (environment : ParamEnvironment)
    (previousState : OperationalState)
    (transition : Array OperationalBoundExpr) : Except OperationalError OperationalState := do
  if transition.size != previousState.size then
    throw (.unsupportedOutputArity transition.size previousState.size)
  let values ← transition.toList.mapM (OperationalBoundExpr.evaluate environment previousState)
  let next := values.zip previousState.toList |>.map fun (value, previous) =>
    match previous with
    | .matrix matrixType _ => .matrix matrixType value
    | .trapdoor matrixType _ => .trapdoor matrixType value
    | .integer lower upper => .integer (min lower value) (max upper value)
    | other => other
  pure next.toArray

def repeatTransition
    (count : Nat)
    (environment : ParamEnvironment)
    (transition : Array OperationalBoundExpr)
    (state : OperationalState) : Except OperationalError OperationalState :=
  match count with
  | 0 => pure state
  | count + 1 => do
      let next ← evaluateTransition environment state transition
      repeatTransition count environment transition next

def defaultFact
    (node : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalFact :=
  match wireType with
  | .matrix matrixType =>
      match matrixCap matrixType environment with
      | some cap => pure (.matrix matrixType cap)
      | none => throw (.invalidMatrixParameters node)
  | .trapdoor matrixType _ _ _ _ =>
      match matrixCap matrixType environment with
      | some cap => pure (.trapdoor matrixType cap)
      | none => throw (.invalidMatrixParameters node)
  | .integer | .constantInt => pure (.integer 0 0)
  | .boolean | .constantBool => pure .boolean
  | .real | .constantReal => pure .real
  | .bytes length =>
      match length.evaluate environment with
      | some value => pure (.bytes value)
      | none => throw (.invalidCount node 0)
  | .typedBlob typeName _ => pure (.typedBlob typeName)
  | .preimage matrixType =>
      match matrixCap matrixType environment with
      | some cap => pure (.matrix matrixType cap)
      | none => throw (.invalidMatrixParameters node)
  | .indexedFamily element count => do
      let element ← defaultFact node element environment
      match count.evaluate environment with
      | some value => pure (.family element value)
      | none => throw (.invalidCount node 0)

def lookupFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalFact :=
  match facts[wire.node]?.bind fun outputs => outputs[wire.port]? with
  | some fact => pure fact
  | none => throw (.missingOperand node wire)

def matrixMaximum
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError Int := do
  match ← lookupFact node facts wire with
  | .matrix _ maximum | .trapdoor _ maximum => pure maximum
  | _ => throw (.operandNotMatrix node wire)

def maximumArguments
    (node : Nat)
    (arguments : List WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError Int := do
  let values ← arguments.mapM (matrixMaximum node · facts)
  pure <| values.foldl max 0

def cappedMatrixFact
    (nodeIndex : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int) : Except OperationalError OperationalFact := do
  let cap ← match matrixCap matrixType environment with
    | some value => pure value
    | none => throw (.invalidMatrixParameters nodeIndex)
  if bound < 0 then throw (.invalidBound nodeIndex bound)
  pure (.matrix matrixType (min cap bound))

def genericNodeFact
    (nodeIndex : Nat)
    (node : Node)
    (outputPort : Nat)
    (outputType : WireTypeExpr)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError OperationalFact := do
  let matrixType? := match outputType with
    | .matrix matrixType | .preimage matrixType => some matrixType
    | _ => none
  match node.kind, matrixType? with
  | .zeroMatrix _, some matrixType => cappedMatrixFact nodeIndex matrixType environment 0
  | .identityMatrix _, some matrixType => cappedMatrixFact nodeIndex matrixType environment 1
  | .constantMatrix _ coefficients, some matrixType =>
      let values ← coefficients.mapM fun coefficient =>
        match coefficient.evaluate environment with
        | some value => pure value
        | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment
        (values.foldl (fun maximum value => max maximum (absolute value)) 0)
  | .uniformResidueSample _, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      cappedMatrixFact nodeIndex matrixType environment cap
  | .uniformIntervalSample _ minimum maximum, some matrixType =>
      let lower ← match minimum.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      let upper ← match maximum.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment (max (absolute lower) (absolute upper))
  | .gaussianSample _ maximum, some matrixType | .preimageSample _ maximum, some matrixType =>
      let bound ← match maximum.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment bound
  | .hashSample _ _ _ _ _ _ _ _, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      cappedMatrixFact nodeIndex matrixType environment cap
  | .gadgetDecompose _ base _, some matrixType =>
      let bound ← match base.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment (absolute bound)
  | .matrixAdd, some matrixType | .matrixSubtract, some matrixType =>
      let bounds ← node.arguments.mapM (matrixMaximum nodeIndex · facts)
      cappedMatrixFact nodeIndex matrixType environment (bounds.foldl (· + ·) 0)
  | .concat _, some matrixType =>
      cappedMatrixFact nodeIndex matrixType environment (← maximumArguments nodeIndex node.arguments facts)
  | .select, some matrixType =>
      -- The first input is the Boolean selector.  The selected matrix is one of the remaining
      -- branches, so their maximum is a sound branch-independent operational bound.
      cappedMatrixFact nodeIndex matrixType environment
        (← maximumArguments nodeIndex (node.arguments.drop 1) facts)
  | .matrixNegate, some matrixType | .transpose, some matrixType | .slice _ _, some matrixType |
      .reshape _ _, some matrixType =>
      cappedMatrixFact nodeIndex matrixType environment (← maximumArguments nodeIndex node.arguments facts)
  | .matrixScale scalar, some matrixType =>
      let scalar ← match scalar.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment
        (absolute scalar * (← maximumArguments nodeIndex node.arguments facts))
  | .matrixMultiply, some matrixType | .tensor, some matrixType =>
      let ringDimension ← match matrixRingDimension matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let innerDimension ← match matrixInnerDimension matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let bounds ← node.arguments.mapM (matrixMaximum nodeIndex · facts)
      let bound := match bounds with
        | left :: right :: _ => ringDimension * innerDimension * left * right
        | _ => 0
      cappedMatrixFact nodeIndex matrixType environment bound
  | .trapdoorSample _ maximum, some matrixType =>
      let bound ← match maximum.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment bound
  | .trapdoorSample _ maximum, none =>
      let bound ← match maximum.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      match outputType with
      | .trapdoor matrixType _ _ _ _ =>
          let cap ← match matrixCap matrixType environment with
            | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
          pure (.trapdoor matrixType (min cap bound))
      | _ => defaultFact nodeIndex outputType environment
  | .trapdoorPublic, some matrixType =>
      cappedMatrixFact nodeIndex matrixType environment (← maximumArguments nodeIndex node.arguments facts)
  | .gadgetTrapdoor _ base, some matrixType =>
      let bound ← match base.evaluate environment with
        | some value => pure value | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment (absolute bound)
  | .unitRowMatrix _ _, some matrixType | .unitColumnMatrix _ _, some matrixType |
      .rotationMatrix _ _, some matrixType =>
      cappedMatrixFact nodeIndex matrixType environment 1
  | .gadgetMatrix _ base, some matrixType | .smallGadgetMatrix _ base, some matrixType |
      .powerOfBaseMatrix _ base _, some matrixType =>
      let bound ← match base.evaluate environment with
        | some value => pure (absolute value)
        | none => throw (.invalidBound nodeIndex 0)
      cappedMatrixFact nodeIndex matrixType environment bound
  | _, _ =>
      -- A generic operation whose output port is not a matrix-specific case still receives a
      -- checked type-derived fact.  Owning crate descriptors later refine relation-bearing ports.
      let _ := outputPort
      defaultFact nodeIndex outputType environment

def evaluateScopeOperational
    (scope : Scope)
    (derivation : ScopeDerivation)
    (environment : ParamEnvironment) : Except OperationalError OperationalScopeFacts := do
  match checkScopeDerivation scope derivation with
  | .error error => throw (.derivation error)
  | .ok () => pure ()
  let rec deriveOutputs
      (nodeIndex : Nat)
      (node : Node)
      (port : Nat)
      (outputTypes : List WireTypeExpr)
      (facts : OperationalScopeFacts) : Except OperationalError (List OperationalFact) := do
    match outputTypes with
    | [] => pure []
    | outputType :: tail =>
        let output ← genericNodeFact nodeIndex node port outputType facts environment
        return output :: (← deriveOutputs nodeIndex node (port + 1) tail facts)
  let rec go (index : Nat) (nodes : List Node) (facts : OperationalScopeFacts) := do
    match nodes with
    | [] => pure facts
    | node :: tail =>
        if node.outputCount != node.outputTypes.length then
          throw (.unsupportedOutputArity index node.outputCount)
        let outputs ← deriveOutputs index node 0 node.outputTypes facts
        go (index + 1) tail (facts.push outputs.toArray)
  go 0 scope.nodes #[]

/-- Future local proof target for ordinary addition.  It intentionally states the runtime
connection without presenting the operational estimate as an established theorem. -/
def MatrixAddOperationalSoundnessClaim : Prop :=
  ∀ (scope : Scope) (derivation : ScopeDerivation) (environment : ParamEnvironment),
    checkScopeDerivation scope derivation = .ok () →
      ∃ facts, evaluateScopeOperational scope derivation environment = .ok facts

private def fixtureType : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 1, columns := .constant 1
}

private def fixtureScope : Scope := {
  nodes := [
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 2, port := 0 })], inputNames := []
}

private def fixtureDerivation : ScopeDerivation := { steps := [
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }] }
] }

example : evaluateScopeOperational fixtureScope fixtureDerivation [] =
    .ok #[#[.matrix fixtureType 0], #[.matrix fixtureType 3], #[.matrix fixtureType 3]] := by
  rfl

example : checkScopeDerivation fixtureScope { steps := [
  { sourceNode := 1, rule := .gaussianSample, arguments := [] }
] } = .error (.sourceNodeMismatch 0 1) := by
  decide

private def selectFixtureScope : Scope := {
  nodes := [
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .select, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
      { node := 2, port := 0 }], outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 3, port := 0 })], inputNames := []
}

private def selectFixtureDerivation : ScopeDerivation := { steps := [
  { sourceNode := 0, rule := .constantBool, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .gaussianSample, arguments := [] },
  { sourceNode := 3, rule := .select, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : evaluateScopeOperational selectFixtureScope selectFixtureDerivation [] =
    .ok #[#[.boolean], #[.matrix fixtureType 3], #[.matrix fixtureType 5],
      #[.matrix fixtureType 5]] := by
  rfl

end Mxx.Certificate
