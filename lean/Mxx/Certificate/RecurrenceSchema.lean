import Mxx.Certificate.SymbolicForm
import Mxx.Certificate.Bounds
import Mxx.Certificate.ValueSemantics

namespace Mxx.Certificate

/-!
Schema-indexed numeric state for sequential-loop bound recurrences.

This module is deliberately independent of executable IR nodes and of protocol-authored
certificates.  A transition can read only typed fields of one immutable previous-state vector;
evaluating the complete transition vector therefore gives simultaneous carried updates.
-/

inductive CarriedValueSchema where
  | matrix
      (matrixType : MatrixTypeExpr)
      (coefficientRepresentation : CoefficientRepresentation)
  | integer
  | boolean
  | bytes
  | family (count : IntExpr) (element : CarriedValueSchema)
  deriving BEq, DecidableEq

/-- Closed runtime meaning of one coarse carried-value schema. Matrix shape, modulus, coefficient
ring, and representation are checked against the actual runtime matrix. Family counts and every
nested element are checked recursively. -/
def CarriedValueSchema.Holds
    (schema : CarriedValueSchema)
    (parameters : Mxx.Ir.ParamEnvironment)
    (value : Mxx.Ir.Value) : Prop :=
  match schema with
  | .matrix matrixType representation =>
      match value with
      | Mxx.Ir.Value.matrix runtimeMatrix =>
          matrixType.Holds parameters runtimeMatrix ∧
            representation.Holds parameters runtimeMatrix
      | _ => False
  | .integer =>
      match value with
      | Mxx.Ir.Value.integer _ => True
      | _ => False
  | .boolean =>
      match value with
      | Mxx.Ir.Value.boolean _ => True
      | _ => False
  | .bytes =>
      match value with
      | Mxx.Ir.Value.bytes _ => True
      | _ => False
  | .family count element =>
      match value with
      | Mxx.Ir.Value.family values =>
          ∃ evaluatedCount,
            evaluateIntExpr parameters count = .ok evaluatedCount ∧
            0 ≤ evaluatedCount ∧
            values.length = evaluatedCount.toNat ∧
            ∀ value ∈ values, element.Holds parameters value
      | _ => False

/-- The actual carried tuple has exactly the analyzer-derived schema, slot by slot. This closed
inductive relation cannot be replaced by a certificate-supplied invariant. -/
inductive CarriedState.Holds
    (parameters : Mxx.Ir.ParamEnvironment) :
    List CarriedValueSchema → List Mxx.Ir.Value → Prop where
  | nil : CarriedState.Holds parameters [] []
  | cons {schema value schemas values}
      (head : schema.Holds parameters value)
      (tail : CarriedState.Holds parameters schemas values) :
      CarriedState.Holds parameters (schema :: schemas) (value :: values)

example :
    (CarriedValueSchema.family (.constant 2) .integer).Holds []
      (.family [.integer 7, .integer (-3)]) := by
  simp [CarriedValueSchema.Holds, evaluateIntExpr]

example :
    ¬ (CarriedValueSchema.family (.constant 2) .integer).Holds []
      (.family [.integer 7]) := by
  simp [CarriedValueSchema.Holds, evaluateIntExpr]

example (matrixType : MatrixTypeExpr) (representation : CoefficientRepresentation) :
    ¬ (CarriedValueSchema.matrix matrixType representation).Holds [] (.integer 0) := by
  simp [CarriedValueSchema.Holds]

def CarriedValueSchema.boundSchema : CarriedValueSchema → CarriedBoundSchema
  | .matrix _ _ => .matrixSummary
  | .integer => .integerInterval
  | .boolean => .boolean
  | .bytes => .bytes
  | .family count element => .family count element.boundSchema

inductive CarriedBoundTemplateState : CarriedBoundSchema → Type where
  | matrix (summary : MatrixBoundSummary) : CarriedBoundTemplateState .matrixSummary
  | integer (lower upper : IntBoundExpr) : CarriedBoundTemplateState .integerInterval
  | boolean : CarriedBoundTemplateState .boolean
  | bytes : CarriedBoundTemplateState .bytes
  | familyEnvelope {count : IntExpr} {elementSchema : CarriedBoundSchema}
      (element : CarriedBoundTemplateState elementSchema) :
      CarriedBoundTemplateState (.family count elementSchema)

/-- Schema-indexed initial bound templates. -/
inductive CarriedBoundTemplateVector : List CarriedBoundSchema → Type where
  | nil : CarriedBoundTemplateVector []
  | cons {schema schemas}
      (head : CarriedBoundTemplateState schema)
      (tail : CarriedBoundTemplateVector schemas) :
      CarriedBoundTemplateVector (schema :: schemas)

inductive RecurrenceSchemaEvalError where
  | natural (error : BoundEvalError)
  | integer (error : IntBoundEvalError)
  | expression (error : IntEvalError)
  | negativeNatural (value : Int)
  | nonPositiveDimension (value : Int)
  | divisionByZero
  deriving BEq, DecidableEq, Repr

private def signalPresenceToBool : SignalPresence → Bool
  | .none => false
  | .present => true

def CarriedBoundTemplateState.evaluate
    {schema : CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceStates : CheckedSymbolicRecurrenceStateTable) :
    CarriedBoundTemplateState schema →
      Except RecurrenceSchemaEvalError (EvaluatedCarriedBoundState schema)
  | .matrix summary => do
      let coefficient ← summary.coefficientL1Bound.evaluateWithSymbolicRecurrences
        environment recurrenceStates |>.mapError .natural
      let noise ← summary.noiseBound.evaluateWithSymbolicRecurrences
        environment recurrenceStates |>.mapError .natural
      let total ← summary.totalBound.evaluateWithSymbolicRecurrences
        environment recurrenceStates |>.mapError .natural
      return .matrix (signalPresenceToBool summary.signal) coefficient noise total
  | .integer lower upper => do
      let lower' ← lower.evaluateWithSymbolicRecurrences
        environment recurrenceStates |>.mapError .integer
      let upper' ← upper.evaluateWithSymbolicRecurrences
        environment recurrenceStates |>.mapError .integer
      return .integer lower' upper'
  | .boolean => .ok .boolean
  | .bytes => .ok .bytes
  | .familyEnvelope element => do
      let element' ← element.evaluate environment recurrenceStates
      return .familyEnvelope element'

def CarriedBoundTemplateVector.evaluate
    {schemas : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceStates : CheckedSymbolicRecurrenceStateTable) :
    CarriedBoundTemplateVector schemas →
      Except RecurrenceSchemaEvalError (CarriedBoundStateVector schemas)
  | .nil => .ok .nil
  | .cons head tail => do
      let head' ← head.evaluate environment recurrenceStates
      let tail' ← tail.evaluate environment recurrenceStates
      return .cons head' tail'

/-! Typed paths through a carried slot and through lane-uniform family envelopes. -/

inductive CarriedBoundNestedPath : CarriedBoundSchema → CarriedBoundSchema → Type where
  | here {schema : CarriedBoundSchema} : CarriedBoundNestedPath schema schema
  | familyElement {count : IntExpr} {element leaf : CarriedBoundSchema}
      (path : CarriedBoundNestedPath element leaf) :
      CarriedBoundNestedPath (.family count element) leaf

inductive CarriedBoundStatePath : List CarriedBoundSchema → CarriedBoundSchema → Type where
  | head {schema leaf : CarriedBoundSchema} {tail : List CarriedBoundSchema}
      (path : CarriedBoundNestedPath schema leaf) :
      CarriedBoundStatePath (schema :: tail) leaf
  | tail {schema leaf : CarriedBoundSchema} {tail : List CarriedBoundSchema}
      (path : CarriedBoundStatePath tail leaf) :
      CarriedBoundStatePath (schema :: tail) leaf

def CarriedBoundNestedPath.read
    {schema leaf : CarriedBoundSchema}
    (path : CarriedBoundNestedPath schema leaf)
    (state : EvaluatedCarriedBoundState schema) : EvaluatedCarriedBoundState leaf :=
  match path with
  | .here => state
  | .familyElement path =>
      match state with
      | .familyEnvelope element => path.read element

def CarriedBoundStatePath.read
    {schemas : List CarriedBoundSchema} {leaf : CarriedBoundSchema}
    (path : CarriedBoundStatePath schemas leaf)
    (state : CarriedBoundStateVector schemas) : EvaluatedCarriedBoundState leaf :=
  match path with
  | .head nested =>
      match state with
      | .cons stateHead _ => nested.read stateHead
  | .tail rest =>
      match state with
      | .cons _ stateTail => rest.read stateTail

inductive MatrixBoundField where
  | coefficientL1
  | noise
  | total
  deriving BEq, DecidableEq, Repr

inductive IntegerBoundField where
  | lower
  | upper
  deriving BEq, DecidableEq, Repr

inductive SignalTransitionExpr (previous : List CarriedBoundSchema) where
  | constant (value : Bool)
  | previousState (path : CarriedBoundStatePath previous .matrixSummary)
  | or (left right : SignalTransitionExpr previous)

inductive NatBoundTransitionExpr (previous : List CarriedBoundSchema) where
  | constant (value : Nat)
  | parameter (value : IntExpr)
  | absolute (value : IntExpr)
  | previousState
      (path : CarriedBoundStatePath previous .matrixSummary)
      (field : MatrixBoundField)
  | add (left right : NatBoundTransitionExpr previous)
  | multiply (left right : NatBoundTransitionExpr previous)
  | minimum (left right : NatBoundTransitionExpr previous)
  | maximum (left right : NatBoundTransitionExpr previous)
  | floorDivide (value : NatBoundTransitionExpr previous) (divisor : Nat)
  | matrixProduct
      (ringDimension innerDimension : IntExpr)
      (left right : NatBoundTransitionExpr previous)
  | centeredModulusCap (modulus : IntExpr) (value : NatBoundTransitionExpr previous)
  /-- A checked bound produced by another analyzer-owned recurrence. This leaf is closed to a
  constant before the current recurrence is iterated. -/
  | externalRecurrence
      (recurrence : SequentialRecurrenceInstanceRef)
      (path : BoundFactPath)

inductive IntBoundTransitionExpr (previous : List CarriedBoundSchema) where
  | constant (value : Int)
  | parameter (value : IntExpr)
  | previousState
      (path : CarriedBoundStatePath previous .integerInterval)
      (field : IntegerBoundField)
  | negate (value : IntBoundTransitionExpr previous)
  | add (left right : IntBoundTransitionExpr previous)
  | subtract (left right : IntBoundTransitionExpr previous)
  | multiply (left right : IntBoundTransitionExpr previous)
  | divide (left right : IntBoundTransitionExpr previous)
  | minimum (left right : IntBoundTransitionExpr previous)
  | maximum (left right : IntBoundTransitionExpr previous)
  /-- Signed counterpart of `NatBoundTransitionExpr.externalRecurrence`. -/
  | externalRecurrence
      (recurrence : SequentialRecurrenceInstanceRef)
      (path : IntBoundFactPath)

def NatBoundTransitionExpr.closeExternalRecurrences
    {previous : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    NatBoundTransitionExpr previous →
      Except RecurrenceSchemaEvalError (NatBoundTransitionExpr previous)
  | .constant value => .ok (.constant value)
  | .parameter value => .ok (.parameter value)
  | .absolute value => .ok (.absolute value)
  | .previousState path field => .ok (.previousState path field)
  | .add left right => return (.add
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .multiply left right => return (.multiply
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .minimum left right => return (.minimum
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .maximum left right => return (.maximum
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .floorDivide value divisor => return (.floorDivide
      (← value.closeExternalRecurrences environment states) divisor)
  | .matrixProduct ringDimension innerDimension left right => return (.matrixProduct
      ringDimension innerDimension
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .centeredModulusCap modulus value => return (.centeredModulusCap modulus
      (← value.closeExternalRecurrences environment states))
  | .externalRecurrence recurrence path => do
      let value ← (BoundExpr.recurrenceResult recurrence path).evaluateWithSymbolicRecurrences
        environment states |>.mapError .natural
      return .constant value

def IntBoundTransitionExpr.closeExternalRecurrences
    {previous : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    IntBoundTransitionExpr previous →
      Except RecurrenceSchemaEvalError (IntBoundTransitionExpr previous)
  | .constant value => .ok (.constant value)
  | .parameter value => .ok (.parameter value)
  | .previousState path field => .ok (.previousState path field)
  | .negate value => return (.negate (← value.closeExternalRecurrences environment states))
  | .add left right => return (.add
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .subtract left right => return (.subtract
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .multiply left right => return (.multiply
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .divide left right => return (.divide
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .minimum left right => return (.minimum
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .maximum left right => return (.maximum
      (← left.closeExternalRecurrences environment states)
      (← right.closeExternalRecurrences environment states))
  | .externalRecurrence recurrence path => do
      let value ← (IntBoundExpr.recurrenceResult recurrence path).evaluateWithSymbolicRecurrences
        environment states |>.mapError .integer
      return .constant value

private def evaluateNaturalParameter
    (environment : Mxx.Ir.ParamEnvironment)
    (expression : IntExpr) : Except RecurrenceSchemaEvalError Nat := do
  let value ← evaluateIntExpr environment expression |>.mapError .expression
  if value < 0 then .error (.negativeNatural value) else .ok value.toNat

private def evaluatePositiveDimensionForTransition
    (environment : Mxx.Ir.ParamEnvironment)
    (expression : IntExpr) : Except RecurrenceSchemaEvalError Nat := do
  let value ← evaluateIntExpr environment expression |>.mapError .expression
  if value ≤ 0 then .error (.nonPositiveDimension value) else .ok value.toNat

def SignalTransitionExpr.evaluate
    {previous : List CarriedBoundSchema}
    (previousState : CarriedBoundStateVector previous) :
    SignalTransitionExpr previous → Bool
  | .constant value => value
  | .previousState path =>
      match path.read previousState with
      | .matrix signal _ _ _ => signal
  | .or left right => left.evaluate previousState || right.evaluate previousState

def NatBoundTransitionExpr.evaluate
    {previous : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (previousState : CarriedBoundStateVector previous) :
    NatBoundTransitionExpr previous → Except RecurrenceSchemaEvalError Nat
  | .constant value => .ok value
  | .parameter value => evaluateNaturalParameter environment value
  | .absolute value =>
      return (← evaluateIntExpr environment value |>.mapError .expression).natAbs
  | .previousState path field =>
      match path.read previousState, field with
      | .matrix _ coefficient _ _, .coefficientL1 => .ok coefficient
      | .matrix _ _ noise _, .noise => .ok noise
      | .matrix _ _ _ total, .total => .ok total
  | .add left right => return (← left.evaluate environment previousState) +
      (← right.evaluate environment previousState)
  | .multiply left right => return (← left.evaluate environment previousState) *
      (← right.evaluate environment previousState)
  | .minimum left right => do
      let left' ← left.evaluate environment previousState
      let right' ← right.evaluate environment previousState
      return Nat.min left' right'
  | .maximum left right => do
      let left' ← left.evaluate environment previousState
      let right' ← right.evaluate environment previousState
      return Nat.max left' right'
  | .floorDivide value divisor =>
      if divisor = 0 then .error .divisionByZero
      else return (← value.evaluate environment previousState) / divisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ring ← evaluatePositiveDimensionForTransition environment ringDimension
      let inner ← evaluatePositiveDimensionForTransition environment innerDimension
      return ring * inner * (← left.evaluate environment previousState) *
        (← right.evaluate environment previousState)
  | .centeredModulusCap modulus value => do
      let modulus ← evaluateIntExpr environment modulus |>.mapError .expression
      return Nat.min (modulus.natAbs / 2) (← value.evaluate environment previousState)
  | .externalRecurrence recurrence path =>
      .error (.natural (.unresolvedRecurrence recurrence path))

def IntBoundTransitionExpr.evaluate
    {previous : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (previousState : CarriedBoundStateVector previous) :
    IntBoundTransitionExpr previous → Except RecurrenceSchemaEvalError Int
  | .constant value => .ok value
  | .parameter value => evaluateIntExpr environment value |>.mapError .expression
  | .previousState path field =>
      match path.read previousState, field with
      | .integer lower _, .lower => .ok lower
      | .integer _ upper, .upper => .ok upper
  | .negate value => return -(← value.evaluate environment previousState)
  | .add left right => return (← left.evaluate environment previousState) +
      (← right.evaluate environment previousState)
  | .subtract left right => return (← left.evaluate environment previousState) -
      (← right.evaluate environment previousState)
  | .multiply left right => return (← left.evaluate environment previousState) *
      (← right.evaluate environment previousState)
  | .divide left right => do
      let denominator ← right.evaluate environment previousState
      if denominator = 0 then .error .divisionByZero
      else return (← left.evaluate environment previousState) / denominator
  | .minimum left right => do
      let left' ← left.evaluate environment previousState
      let right' ← right.evaluate environment previousState
      return min left' right'
  | .maximum left right => do
      let left' ← left.evaluate environment previousState
      let right' ← right.evaluate environment previousState
      return max left' right'
  | .externalRecurrence recurrence path =>
      .error (.integer (.unresolvedRecurrence recurrence path))

structure MatrixSummaryTransition (previous : List CarriedBoundSchema) where
  signal : SignalTransitionExpr previous
  coefficientL1Bound : NatBoundTransitionExpr previous
  noiseBound : NatBoundTransitionExpr previous
  totalBound : NatBoundTransitionExpr previous

inductive CarriedBoundTransition
    (previous : List CarriedBoundSchema) : CarriedBoundSchema → Type where
  | matrix (fields : MatrixSummaryTransition previous) :
      CarriedBoundTransition previous .matrixSummary
  | integer (lower upper : IntBoundTransitionExpr previous) :
      CarriedBoundTransition previous .integerInterval
  | boolean : CarriedBoundTransition previous .boolean
  | bytes : CarriedBoundTransition previous .bytes
  | familyEnvelope {count : IntExpr} {elementSchema : CarriedBoundSchema}
      (element : CarriedBoundTransition previous elementSchema) :
      CarriedBoundTransition previous (.family count elementSchema)

def CarriedBoundTransition.evaluate
    {previous : List CarriedBoundSchema} {schema : CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (previousState : CarriedBoundStateVector previous) :
    CarriedBoundTransition previous schema →
      Except RecurrenceSchemaEvalError (EvaluatedCarriedBoundState schema)
  | .matrix fields => do
      return .matrix
        (fields.signal.evaluate previousState)
        (← fields.coefficientL1Bound.evaluate environment previousState)
        (← fields.noiseBound.evaluate environment previousState)
        (← fields.totalBound.evaluate environment previousState)
  | .integer lower upper => do
      let lower' ← lower.evaluate environment previousState
      let upper' ← upper.evaluate environment previousState
      return .integer lower' upper'
  | .boolean => .ok .boolean
  | .bytes => .ok .bytes
  | .familyEnvelope element => do
      let element' ← element.evaluate environment previousState
      return .familyEnvelope element'

def CarriedBoundTransition.closeExternalRecurrences
    {previous : List CarriedBoundSchema} {schema : CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    CarriedBoundTransition previous schema →
      Except RecurrenceSchemaEvalError (CarriedBoundTransition previous schema)
  | .matrix fields => do
      return .matrix {
        signal := fields.signal
        coefficientL1Bound := ← fields.coefficientL1Bound.closeExternalRecurrences
          environment states
        noiseBound := ← fields.noiseBound.closeExternalRecurrences environment states
        totalBound := ← fields.totalBound.closeExternalRecurrences environment states
      }
  | .integer lower upper => return (.integer
      (← lower.closeExternalRecurrences environment states)
      (← upper.closeExternalRecurrences environment states))
  | .boolean => .ok .boolean
  | .bytes => .ok .bytes
  | .familyEnvelope element => return (.familyEnvelope
      (← element.closeExternalRecurrences environment states))

inductive CarriedBoundTransitionVector
    (previous : List CarriedBoundSchema) : List CarriedBoundSchema → Type where
  | nil : CarriedBoundTransitionVector previous []
  | cons {schema : CarriedBoundSchema} {schemas : List CarriedBoundSchema}
      (head : CarriedBoundTransition previous schema)
      (tail : CarriedBoundTransitionVector previous schemas) :
      CarriedBoundTransitionVector previous (schema :: schemas)

def CarriedBoundTransitionVector.evaluate
    {previous output : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (previousState : CarriedBoundStateVector previous) :
    CarriedBoundTransitionVector previous output →
      Except RecurrenceSchemaEvalError (CarriedBoundStateVector output)
  | .nil => .ok .nil
  | .cons head tail => do
      let head' ← head.evaluate environment previousState
      let tail' ← tail.evaluate environment previousState
      return .cons head' tail'

def CarriedBoundTransitionVector.closeExternalRecurrences
    {previous output : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    CarriedBoundTransitionVector previous output →
      Except RecurrenceSchemaEvalError (CarriedBoundTransitionVector previous output)
  | .nil => .ok .nil
  | .cons head tail => return (.cons
      (← head.closeExternalRecurrences environment states)
      (← tail.closeExternalRecurrences environment states))

def iterateCarriedBoundTransition
    {schema : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (transition : CarriedBoundTransitionVector schema schema) :
    Nat → CarriedBoundStateVector schema →
      Except RecurrenceSchemaEvalError (CarriedBoundStateVector schema)
  | 0, initial => .ok initial
  | count + 1, initial => do
      let next ← transition.evaluate environment initial
      iterateCarriedBoundTransition environment transition count next

theorem iterateCarriedBoundTransition_zero
    {schema : List CarriedBoundSchema}
    (environment : Mxx.Ir.ParamEnvironment)
    (transition : CarriedBoundTransitionVector schema schema)
    (initial : CarriedBoundStateVector schema) :
    iterateCarriedBoundTransition environment transition 0 initial = .ok initial := rfl

/-! A two-slot fixture whose update is observably simultaneous: the intervals are swapped. -/

private abbrev twoIntegerSchemas : List CarriedBoundSchema :=
  [.integerInterval, .integerInterval]

private def firstIntegerPath :
    CarriedBoundStatePath twoIntegerSchemas .integerInterval :=
  .head .here

private def secondIntegerPath :
    CarriedBoundStatePath twoIntegerSchemas .integerInterval :=
  .tail (.head .here)

private def swapIntegerIntervals :
    CarriedBoundTransitionVector twoIntegerSchemas twoIntegerSchemas :=
  .cons (.integer
      (.previousState secondIntegerPath .lower)
      (.previousState secondIntegerPath .upper))
    (.cons (.integer
        (.previousState firstIntegerPath .lower)
        (.previousState firstIntegerPath .upper))
      .nil)

example :
    swapIntegerIntervals.evaluate []
      (.cons (.integer 1 2) (.cons (.integer 10 20) .nil)) =
      .ok (.cons (.integer 10 20) (.cons (.integer 1 2) .nil)) := by
  simp [bind, Except.bind, swapIntegerIntervals,
    CarriedBoundTransitionVector.evaluate,
    CarriedBoundTransition.evaluate, IntBoundTransitionExpr.evaluate,
    firstIntegerPath, secondIntegerPath, CarriedBoundStatePath.read,
    CarriedBoundNestedPath.read]
  rfl

private abbrev uniformMatrixFamilySchema : CarriedBoundSchema :=
  .family (.constant 8) .matrixSummary

private def uniformFamilyMatrixPath :
    CarriedBoundStatePath [uniformMatrixFamilySchema] .matrixSummary :=
  .head (.familyElement .here)

private def incrementUniformFamilyNoise :
    CarriedBoundTransitionVector [uniformMatrixFamilySchema] [uniformMatrixFamilySchema] :=
  .cons (.familyEnvelope (.matrix {
      signal := .previousState uniformFamilyMatrixPath
      coefficientL1Bound := .previousState uniformFamilyMatrixPath .coefficientL1
      noiseBound := .add (.previousState uniformFamilyMatrixPath .noise) (.constant 1)
      totalBound := .add (.previousState uniformFamilyMatrixPath .total) (.constant 1)
    })) .nil

example :
    incrementUniformFamilyNoise.evaluate []
      (.cons (.familyEnvelope (.matrix true 3 5 8)) .nil) =
      .ok (.cons (.familyEnvelope (.matrix true 3 6 9)) .nil) := by
  simp [bind, Except.bind, incrementUniformFamilyNoise,
    CarriedBoundTransitionVector.evaluate,
    CarriedBoundTransition.evaluate, SignalTransitionExpr.evaluate,
    NatBoundTransitionExpr.evaluate, uniformFamilyMatrixPath,
    CarriedBoundStatePath.read,
    CarriedBoundNestedPath.read]
  rfl

end Mxx.Certificate
