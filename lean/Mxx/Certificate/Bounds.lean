import Mxx.Certificate.CheckedRecurrenceState

namespace Mxx.Certificate

inductive IntEvalError where
  | missingParameter (name : String)
  | wrongParameterType (name : String)
  | missingLoopIndex (slot : Nat)
  | divisionByZero
  | roundDivisionByZero
  deriving BEq, DecidableEq, Repr

def evaluateIntExpr (environment : Mxx.Ir.ParamEnvironment) :
    IntExpr → Except IntEvalError Int
  | .constant value => .ok value
  | .parameter name =>
      match Mxx.Ir.lookupParam name environment with
      | none => .error (.missingParameter name)
      | some (.integer value) => .ok value
      | some (.rational _) => .error (.wrongParameterType name)
  | .loopIndex slot =>
      match Mxx.Ir.lookupLoopIndex slot environment with
      | none => .error (.missingLoopIndex slot)
      | some value => .ok value
  | .add left right =>
      return (← evaluateIntExpr environment left) + (← evaluateIntExpr environment right)
  | .subtract left right =>
      return (← evaluateIntExpr environment left) - (← evaluateIntExpr environment right)
  | .multiply left right =>
      return (← evaluateIntExpr environment left) * (← evaluateIntExpr environment right)
  | .divide left right => do
      let denominator ← evaluateIntExpr environment right
      if denominator = 0 then .error .divisionByZero
      else return (← evaluateIntExpr environment left) / denominator
  | .roundDivide left right => do
      let denominator ← evaluateIntExpr environment right
      if denominator = 0 then .error .roundDivisionByZero
      else return Mxx.Ir.roundDiv (← evaluateIntExpr environment left) denominator
  | .log2Ceil value => return Mxx.Ir.log2Ceil (← evaluateIntExpr environment value)

/-- The certificate evaluator is the error-reporting view of the executable IR evaluator.
Successful executable evaluation therefore supplies the exact certificate denotation; callers
never assert a second parameter value. -/
theorem evaluateIntExpr_ok_of_ir_evaluate
    (environment : Mxx.Ir.ParamEnvironment)
    (expression : IntExpr)
    (value : Int)
    (evaluates : expression.evaluate environment = some value) :
    evaluateIntExpr environment expression = .ok value := by
  induction expression generalizing value with
  | constant constant => simpa [Mxx.Ir.IntExpr.evaluate, evaluateIntExpr] using evaluates
  | parameter name =>
      simp only [Mxx.Ir.IntExpr.evaluate] at evaluates
      cases lookup : Mxx.Ir.lookupParam name environment <;> simp [lookup] at evaluates
      rename_i parameter
      cases parameter <;> simp [lookup, evaluateIntExpr] at evaluates ⊢
      exact evaluates
  | loopIndex slot =>
      simp only [Mxx.Ir.IntExpr.evaluate] at evaluates
      cases lookup : Mxx.Ir.lookupLoopIndex slot environment <;> simp [lookup] at evaluates
      subst value
      simp [evaluateIntExpr, lookup]
  | add left right leftInduction rightInduction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases leftEvaluation : left.evaluate environment <;> simp [leftEvaluation] at evaluates
      cases rightEvaluation : right.evaluate environment <;> simp [rightEvaluation] at evaluates
      subst value
      rw [evaluateIntExpr, leftInduction _ leftEvaluation, rightInduction _ rightEvaluation]
      rfl
  | subtract left right leftInduction rightInduction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases leftEvaluation : left.evaluate environment <;> simp [leftEvaluation] at evaluates
      cases rightEvaluation : right.evaluate environment <;> simp [rightEvaluation] at evaluates
      subst value
      rw [evaluateIntExpr, leftInduction _ leftEvaluation, rightInduction _ rightEvaluation]
      rfl
  | multiply left right leftInduction rightInduction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases leftEvaluation : left.evaluate environment <;> simp [leftEvaluation] at evaluates
      cases rightEvaluation : right.evaluate environment <;> simp [rightEvaluation] at evaluates
      subst value
      rw [evaluateIntExpr, leftInduction _ leftEvaluation, rightInduction _ rightEvaluation]
      rfl
  | divide left right leftInduction rightInduction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases rightEvaluation : right.evaluate environment <;> simp [rightEvaluation] at evaluates
      rename_i denominator
      by_cases zero : denominator = 0
      · simp [zero] at evaluates
      · simp [zero] at evaluates
        cases leftEvaluation : left.evaluate environment <;> simp [leftEvaluation] at evaluates
        rename_i numerator
        subst value
        rw [evaluateIntExpr, rightInduction _ rightEvaluation, leftInduction _ leftEvaluation]
        dsimp only [bind, Except.bind]
        rw [if_neg zero]
        rfl
  | roundDivide left right leftInduction rightInduction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases rightEvaluation : right.evaluate environment <;> simp [rightEvaluation] at evaluates
      rename_i denominator
      by_cases zero : denominator = 0
      · simp [zero] at evaluates
      · simp [zero] at evaluates
        cases leftEvaluation : left.evaluate environment <;> simp [leftEvaluation] at evaluates
        rename_i numerator
        subst value
        rw [evaluateIntExpr, rightInduction _ rightEvaluation, leftInduction _ leftEvaluation]
        dsimp only [bind, Except.bind]
        rw [if_neg zero]
        rfl
  | log2Ceil input induction =>
      simp only [Mxx.Ir.IntExpr.evaluate, Option.bind_eq_bind] at evaluates
      cases inputEvaluation : input.evaluate environment <;> simp [inputEvaluation] at evaluates
      subst value
      simp [evaluateIntExpr, induction _ inputEvaluation]

inductive BoundEvalError where
  | integer (error : IntEvalError)
  | negativeParameter (value : Int)
  | nonPositiveDimension (value : Int)
  | nonPositiveDivisor
  | unresolvedRecurrence (recurrence : SequentialRecurrenceInstanceRef) (path : BoundFactPath)
  | escapedCarriedInput (path : BoundFactPath)
  deriving BEq, DecidableEq, Repr

private def evaluateBoundParameter
    (environment : Mxx.Ir.ParamEnvironment)
    (expression : IntExpr) : Except BoundEvalError Nat := do
  let value ← (evaluateIntExpr environment expression).mapError .integer
  if value < 0 then .error (.negativeParameter value) else .ok value.toNat

private def evaluatePositiveDimension
    (environment : Mxx.Ir.ParamEnvironment)
    (expression : IntExpr) : Except BoundEvalError Nat := do
  let value ← (evaluateIntExpr environment expression).mapError .integer
  if value ≤ 0 then .error (.nonPositiveDimension value) else .ok value.toNat

private def boundPathRootSlot : BoundFactPath → Nat
  | .affineCoefficientBound slot _ | .affineNoiseBound slot | .matrixTotalBound slot |
      .familyElement slot _ _ => slot

private def intBoundPathRootSlot : IntBoundFactPath → Nat
  | .lower slot | .upper slot | .familyElement slot _ _ => slot

private def lookupNaturalNested
    (rootSlot : Nat) {schema : CarriedBoundSchema}
    (state : EvaluatedCarriedBoundState schema) : BoundFactPath → Option Nat
  | .affineCoefficientBound slot _ =>
      if slot != rootSlot then none else
        match state with
        | .matrix _ coefficient _ _ => some coefficient
        | _ => none
  | .affineNoiseBound slot =>
      if slot != rootSlot then none else
        match state with
        | .matrix _ _ noise _ => some noise
        | _ => none
  | .matrixTotalBound slot =>
      if slot != rootSlot then none else
        match state with
        | .matrix _ _ _ total => some total
        | _ => none
  | .familyElement slot _ nested =>
      if slot != rootSlot then none else
        match state with
        | .familyEnvelope element => lookupNaturalNested rootSlot element nested
        | _ => none

private def lookupIntegerNested
    (rootSlot : Nat) {schema : CarriedBoundSchema}
    (state : EvaluatedCarriedBoundState schema) : IntBoundFactPath → Option Int
  | .lower slot =>
      if slot != rootSlot then none else
        match state with
        | .integer lower _ => some lower
        | _ => none
  | .upper slot =>
      if slot != rootSlot then none else
        match state with
        | .integer _ upper => some upper
        | _ => none
  | .familyElement slot _ nested =>
      if slot != rootSlot then none else
        match state with
        | .familyEnvelope element => lookupIntegerNested rootSlot element nested
        | _ => none

private def lookupNaturalSlot
    (rootSlot : Nat) :
    (slot : Nat) → {schemas : List CarriedBoundSchema} →
      CarriedBoundStateVector schemas → BoundFactPath → Option Nat
  | _, _, .nil, _ => none
  | 0, _, .cons head _, path => lookupNaturalNested rootSlot head path
  | slot + 1, _, .cons _ tail, path => lookupNaturalSlot rootSlot slot tail path

private def lookupIntegerSlot
    (rootSlot : Nat) :
    (slot : Nat) → {schemas : List CarriedBoundSchema} →
      CarriedBoundStateVector schemas → IntBoundFactPath → Option Int
  | _, _, .nil, _ => none
  | 0, _, .cons head _, path => lookupIntegerNested rootSlot head path
  | slot + 1, _, .cons _ tail, path => lookupIntegerSlot rootSlot slot tail path

private def lookupNaturalSymbolicRecurrence
    (identity : SequentialRecurrenceInstanceRef)
    (path : BoundFactPath)
    (entries : List CheckedSymbolicRecurrenceState) : Option Nat := do
  let entry ← uniqueLaneUniformRecurrenceMatch? identity entries (·.identity)
  lookupNaturalSlot (boundPathRootSlot path) (boundPathRootSlot path) entry.values path

private def lookupIntegerSymbolicRecurrence
    (identity : SequentialRecurrenceInstanceRef)
    (path : IntBoundFactPath)
    (entries : List CheckedSymbolicRecurrenceState) : Option Int := do
  let entry ← uniqueLaneUniformRecurrenceMatch? identity entries (·.identity)
  lookupIntegerSlot (intBoundPathRootSlot path) (intBoundPathRootSlot path) entry.values path

/-- Exact hard-bound evaluation against the dependent table returned by symbolic recurrence
checking. Missing, duplicate, or ill-typed paths fail closed. -/
def BoundExpr.evaluateWithSymbolicRecurrences
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) : BoundExpr → Except BoundEvalError Nat
  | .constant value => .ok value
  | .parameter value => evaluateBoundParameter environment value
  | .add left right => return (← left.evaluateWithSymbolicRecurrences environment states) +
      (← right.evaluateWithSymbolicRecurrences environment states)
  | .multiply left right => return (← left.evaluateWithSymbolicRecurrences environment states) *
      (← right.evaluateWithSymbolicRecurrences environment states)
  | .maximum left right => do
      let left' ← left.evaluateWithSymbolicRecurrences environment states
      let right' ← right.evaluateWithSymbolicRecurrences environment states
      return Nat.max left' right'
  | .absolute value =>
      return (← (evaluateIntExpr environment value).mapError BoundEvalError.integer).natAbs
  | .floorDivide value divisor =>
      if divisor = 0 then .error .nonPositiveDivisor
      else return (← value.evaluateWithSymbolicRecurrences environment states) / divisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ring ← evaluatePositiveDimension environment ringDimension
      let inner ← evaluatePositiveDimension environment innerDimension
      return ring * inner * (← left.evaluateWithSymbolicRecurrences environment states) *
        (← right.evaluateWithSymbolicRecurrences environment states)
  | .minimum left right => do
      let left' ← left.evaluateWithSymbolicRecurrences environment states
      let right' ← right.evaluateWithSymbolicRecurrences environment states
      return Nat.min left' right'
  | .recurrenceResult identity path =>
      match lookupNaturalSymbolicRecurrence identity path states.entries with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence identity path)
  | .carriedInput path => .error (.escapedCarriedInput path)

/-- Evaluate a closed hard-bound expression. Recurrence results and carried placeholders are
rejected directly; this evaluator does not consult the legacy flat recurrence table. -/
def BoundExpr.evaluate (environment : Mxx.Ir.ParamEnvironment) :
    BoundExpr → Except BoundEvalError Nat
  | .constant value => .ok value
  | .parameter value => evaluateBoundParameter environment value
  | .add left right => return (← left.evaluate environment) + (← right.evaluate environment)
  | .multiply left right =>
      return (← left.evaluate environment) * (← right.evaluate environment)
  | .maximum left right =>
      return Nat.max (← left.evaluate environment) (← right.evaluate environment)
  | .absolute value =>
      return (← (evaluateIntExpr environment value).mapError BoundEvalError.integer).natAbs
  | .floorDivide value positiveDivisor =>
      if positiveDivisor = 0 then .error .nonPositiveDivisor
      else return (← value.evaluate environment) / positiveDivisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← evaluatePositiveDimension environment ringDimension
      let innerDimension ← evaluatePositiveDimension environment innerDimension
      return ringDimension * innerDimension * (← left.evaluate environment) *
        (← right.evaluate environment)
  | .minimum left right =>
      return Nat.min (← left.evaluate environment) (← right.evaluate environment)
  | .recurrenceResult recurrence path => .error (.unresolvedRecurrence recurrence path)
  | .carriedInput path => .error (.escapedCarriedInput path)

/-- Successful closed bound evaluation is unchanged by the recurrence-aware evaluator. The two
additional constructors cannot occur on a successful strict closed-evaluation path. -/
theorem BoundExpr.evaluateWithSymbolicRecurrences_of_evaluate_eq_ok
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    ∀ {expression : BoundExpr} {value : Nat}, expression.evaluate environment = .ok value →
      expression.evaluateWithSymbolicRecurrences environment states = .ok value := by
  intro expression
  induction expression <;> intro value evaluates
  case constant => exact evaluates
  case parameter => exact evaluates
  case add left right leftIH rightIH =>
    simp only [BoundExpr.evaluate] at evaluates
    cases leftResult : left.evaluate environment with
    | error error =>
        rw [leftResult] at evaluates
        contradiction
    | ok leftValue =>
        have leftRec := leftIH leftResult
        cases rightResult : right.evaluate environment with
        | error error =>
            rw [leftResult, rightResult] at evaluates
            contradiction
        | ok rightValue =>
            have rightRec := rightIH rightResult
            simp [BoundExpr.evaluateWithSymbolicRecurrences, leftRec, rightRec]
            simpa [leftResult, rightResult] using evaluates
  case multiply left right leftIH rightIH =>
    simp only [BoundExpr.evaluate] at evaluates
    cases leftResult : left.evaluate environment with
    | error error =>
        rw [leftResult] at evaluates
        contradiction
    | ok leftValue =>
        have leftRec := leftIH leftResult
        cases rightResult : right.evaluate environment with
        | error error =>
            rw [leftResult, rightResult] at evaluates
            contradiction
        | ok rightValue =>
            have rightRec := rightIH rightResult
            simp [BoundExpr.evaluateWithSymbolicRecurrences, leftRec, rightRec]
            simpa [leftResult, rightResult] using evaluates
  case maximum left right leftIH rightIH =>
    simp only [BoundExpr.evaluate] at evaluates
    cases leftResult : left.evaluate environment with
    | error error =>
        rw [leftResult] at evaluates
        contradiction
    | ok leftValue =>
        have leftRec := leftIH leftResult
        cases rightResult : right.evaluate environment with
        | error error =>
            rw [leftResult, rightResult] at evaluates
            contradiction
        | ok rightValue =>
            have rightRec := rightIH rightResult
            simp [BoundExpr.evaluateWithSymbolicRecurrences, leftRec, rightRec]
            simpa [leftResult, rightResult] using evaluates
  case absolute => exact evaluates
  case floorDivide expression divisor induction =>
    simp only [BoundExpr.evaluate] at evaluates
    by_cases divisorZero : divisor = 0
    · rw [if_pos divisorZero] at evaluates
      contradiction
    · rw [if_neg divisorZero] at evaluates
      cases result : expression.evaluate environment with
      | error error =>
          rw [result] at evaluates
          contradiction
      | ok resultValue =>
          have recurrenceResult := induction result
          simp [BoundExpr.evaluateWithSymbolicRecurrences, divisorZero, recurrenceResult]
          simpa [result] using evaluates
  case matrixProduct ringDimension innerDimension left right leftIH rightIH =>
    simp only [BoundExpr.evaluate] at evaluates
    cases ringResult : evaluatePositiveDimension environment ringDimension with
    | error error =>
        rw [ringResult] at evaluates
        contradiction
    | ok ring =>
        cases innerResult : evaluatePositiveDimension environment innerDimension with
        | error error =>
            rw [ringResult, innerResult] at evaluates
            contradiction
        | ok inner =>
            cases leftResult : left.evaluate environment with
            | error error =>
                rw [ringResult, innerResult, leftResult] at evaluates
                contradiction
            | ok leftValue =>
                have leftRec := leftIH leftResult
                cases rightResult : right.evaluate environment with
                | error error =>
                    rw [ringResult, innerResult, leftResult, rightResult] at evaluates
                    contradiction
                | ok rightValue =>
                    have rightRec := rightIH rightResult
                    simp [BoundExpr.evaluateWithSymbolicRecurrences, ringResult, innerResult,
                      leftRec, rightRec]
                    simpa [ringResult, innerResult, leftResult, rightResult] using evaluates
  case minimum left right leftIH rightIH =>
    simp only [BoundExpr.evaluate] at evaluates
    cases leftResult : left.evaluate environment with
    | error error =>
        rw [leftResult] at evaluates
        contradiction
    | ok leftValue =>
        have leftRec := leftIH leftResult
        cases rightResult : right.evaluate environment with
        | error error =>
            rw [leftResult, rightResult] at evaluates
            contradiction
        | ok rightValue =>
            have rightRec := rightIH rightResult
            simp [BoundExpr.evaluateWithSymbolicRecurrences, leftRec, rightRec]
            simpa [leftResult, rightResult] using evaluates
  case recurrenceResult => simp [BoundExpr.evaluate] at evaluates
  case carriedInput => simp [BoundExpr.evaluate] at evaluates

inductive IntBoundEvalError where
  | integer (error : IntEvalError)
  | natural (error : BoundEvalError)
  | divisionByZero
  | unresolvedRecurrence (recurrence : SequentialRecurrenceInstanceRef) (path : IntBoundFactPath)
  | escapedCarriedInput (path : IntBoundFactPath)
  deriving BEq, DecidableEq, Repr

def IntBoundExpr.evaluateWithSymbolicRecurrences
    (environment : Mxx.Ir.ParamEnvironment)
    (states : CheckedSymbolicRecurrenceStateTable) :
    IntBoundExpr → Except IntBoundEvalError Int
  | .integer value => evaluateIntExpr environment value |>.mapError .integer
  | .natural value =>
      return (← value.evaluateWithSymbolicRecurrences environment states |>.mapError .natural)
  | .negate value => return -(← value.evaluateWithSymbolicRecurrences environment states)
  | .add left right => return (← left.evaluateWithSymbolicRecurrences environment states) +
      (← right.evaluateWithSymbolicRecurrences environment states)
  | .subtract left right => return (← left.evaluateWithSymbolicRecurrences environment states) -
      (← right.evaluateWithSymbolicRecurrences environment states)
  | .multiply left right => return (← left.evaluateWithSymbolicRecurrences environment states) *
      (← right.evaluateWithSymbolicRecurrences environment states)
  | .divide left right => do
      let denominator ← right.evaluateWithSymbolicRecurrences environment states
      if denominator = 0 then .error .divisionByZero
      else return (← left.evaluateWithSymbolicRecurrences environment states) / denominator
  | .minimum left right => do
      let left' ← left.evaluateWithSymbolicRecurrences environment states
      let right' ← right.evaluateWithSymbolicRecurrences environment states
      return min left' right'
  | .maximum left right => do
      let left' ← left.evaluateWithSymbolicRecurrences environment states
      let right' ← right.evaluateWithSymbolicRecurrences environment states
      return max left' right'
  | .carriedInput path => .error (.escapedCarriedInput path)
  | .recurrenceResult identity path =>
      match lookupIntegerSymbolicRecurrence identity path states.entries with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence identity path)

/-- Evaluate a closed signed bound expression without consulting the legacy flat recurrence
table. Recurrence results and carried placeholders are rejected directly. -/
def IntBoundExpr.evaluateClosed
    (environment : Mxx.Ir.ParamEnvironment) :
    IntBoundExpr → Except IntBoundEvalError Int
  | .integer value => (evaluateIntExpr environment value).mapError .integer
  | .natural value => return (← value.evaluate environment |>.mapError .natural)
  | .negate value => return -(← value.evaluateClosed environment)
  | .add left right =>
      return (← left.evaluateClosed environment) + (← right.evaluateClosed environment)
  | .subtract left right =>
      return (← left.evaluateClosed environment) - (← right.evaluateClosed environment)
  | .multiply left right =>
      return (← left.evaluateClosed environment) * (← right.evaluateClosed environment)
  | .divide left right => do
      let denominator ← right.evaluateClosed environment
      if denominator = 0 then .error .divisionByZero
      else return (← left.evaluateClosed environment) / denominator
  | .minimum left right =>
      return min (← left.evaluateClosed environment) (← right.evaluateClosed environment)
  | .maximum left right =>
      return max (← left.evaluateClosed environment) (← right.evaluateClosed environment)
  | .carriedInput path => .error (.escapedCarriedInput path)
  | .recurrenceResult recurrence path => .error (.unresolvedRecurrence recurrence path)

theorem BoundExpr.evaluate_parameter_nat (name : String) (value : Nat) :
    (BoundExpr.parameter (.parameter name)).evaluate
      [(name, .integer value)] = .ok value := by
  have evaluated : evaluateIntExpr [(name, .integer value)] (.parameter name) = .ok value := by
    simp [evaluateIntExpr, Mxx.Ir.lookupParam]
  unfold BoundExpr.evaluate evaluateBoundParameter
  rw [evaluated]
  change (if (value : Int) < 0 then Except.error (.negativeParameter value)
    else Except.ok (value : Int).toNat) = Except.ok value
  split
  · rename_i negative
    omega
  · rename_i nonnegative
    simp only [Except.ok.injEq]
    rfl

example :
    BoundExpr.evaluate [] (.matrixProduct (.constant 4) (.constant 3) (.constant 2) (.constant 5)) =
      .ok 120 := rfl

example : BoundExpr.evaluate [] (.floorDivide (.constant 9) 0) =
    .error .nonPositiveDivisor := rfl

example : evaluateIntExpr [] (.parameter "missing") =
    .error (.missingParameter "missing") := rfl

example : BoundExpr.evaluate [] (.parameter (.constant (-1))) =
    .error (.negativeParameter (-1)) := rfl

example : evaluateIntExpr [] (.divide (.constant 1) (.constant 0)) =
    .error .divisionByZero := rfl

end Mxx.Certificate
