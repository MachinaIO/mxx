import Mxx.Certificate.Facts

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
  | unresolvedRecurrence (recurrence : FactRecurrenceInstanceRef) (path : BoundFactPath)
  | escapedCarriedInput (path : BoundFactPath)
  deriving BEq, DecidableEq, Repr

structure NaturalRecurrenceBound where
  recurrence : FactRecurrenceInstanceRef
  path : BoundFactPath
  value : Nat

structure IntegerRecurrenceBound where
  recurrence : FactRecurrenceInstanceRef
  path : IntBoundFactPath
  value : Int

/-- Analyzer-produced numeric results of checked sequential recurrences.  Both components are
committed simultaneously after each step by the recurrence resolver. -/
structure CheckedRecurrenceBoundTable where
  natural : List NaturalRecurrenceBound := []
  integer : List IntegerRecurrenceBound := []

structure CarriedNaturalBound where
  path : BoundFactPath
  value : Nat

structure CarriedIntegerBound where
  path : IntBoundFactPath
  value : Int

/-- One immutable numeric snapshot used while evaluating a recurrence body template. -/
structure CarriedBoundTable where
  natural : List CarriedNaturalBound := []
  integer : List CarriedIntegerBound := []

private def lookupCarriedNatural (path : BoundFactPath) : List CarriedNaturalBound → Option Nat
  | [] => none
  | entry :: tail =>
      if entry.path.sameUniformLocation path then some entry.value else lookupCarriedNatural path tail

private def lookupCarriedInteger (path : IntBoundFactPath) : List CarriedIntegerBound → Option Int
  | [] => none
  | entry :: tail =>
      if entry.path.sameUniformLocation path then some entry.value else lookupCarriedInteger path tail

private def lookupNaturalRecurrenceBound
    (recurrence : FactRecurrenceInstanceRef)
    (path : BoundFactPath) : List NaturalRecurrenceBound → Option Nat
  | [] => none
  | entry :: tail =>
      if entry.recurrence = recurrence && entry.path.sameUniformLocation path then some entry.value
      else lookupNaturalRecurrenceBound recurrence path tail

private def lookupIntegerRecurrenceBound
    (recurrence : FactRecurrenceInstanceRef)
    (path : IntBoundFactPath) : List IntegerRecurrenceBound → Option Int
  | [] => none
  | entry :: tail =>
      if entry.recurrence = recurrence && entry.path.sameUniformLocation path then some entry.value
      else lookupIntegerRecurrenceBound recurrence path tail

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

/-- Evaluate a hard-bound expression exactly in `Nat`; no floating-point conversion is used. -/
def BoundExpr.evaluateWithRecurrences
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable) :
    BoundExpr → Except BoundEvalError Nat
  | .constant value => .ok value
  | .parameter value => evaluateBoundParameter environment value
  | .add left right => return (← left.evaluateWithRecurrences environment recurrenceBounds) +
      (← right.evaluateWithRecurrences environment recurrenceBounds)
  | .multiply left right =>
      return (← left.evaluateWithRecurrences environment recurrenceBounds) *
        (← right.evaluateWithRecurrences environment recurrenceBounds)
  | .maximum left right =>
      return Nat.max (← left.evaluateWithRecurrences environment recurrenceBounds)
        (← right.evaluateWithRecurrences environment recurrenceBounds)
  | .absolute value =>
      return (← (evaluateIntExpr environment value).mapError BoundEvalError.integer).natAbs
  | .floorDivide value positiveDivisor =>
      if positiveDivisor = 0 then .error .nonPositiveDivisor
      else return (← value.evaluateWithRecurrences environment recurrenceBounds) / positiveDivisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← evaluatePositiveDimension environment ringDimension
      let innerDimension ← evaluatePositiveDimension environment innerDimension
      return ringDimension * innerDimension *
        (← left.evaluateWithRecurrences environment recurrenceBounds) *
        (← right.evaluateWithRecurrences environment recurrenceBounds)
  | .minimum left right =>
      return Nat.min (← left.evaluateWithRecurrences environment recurrenceBounds)
        (← right.evaluateWithRecurrences environment recurrenceBounds)
  | .recurrenceResult recurrence path =>
      match lookupNaturalRecurrenceBound recurrence path recurrenceBounds.natural with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence recurrence path)
  | .carriedInput path => .error (.escapedCarriedInput path)

def BoundExpr.evaluate (environment : Mxx.Ir.ParamEnvironment) (expression : BoundExpr) :
    Except BoundEvalError Nat :=
  expression.evaluateWithRecurrences environment {}

/-- Evaluate a recurrence body bound against one immutable previous-state snapshot. -/
def BoundExpr.evaluateTemplate
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable)
    (carried : CarriedBoundTable) : BoundExpr → Except BoundEvalError Nat
  | .constant value => .ok value
  | .parameter value => evaluateBoundParameter environment value
  | .add left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return left' + right'
  | .multiply left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return left' * right'
  | .maximum left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return Nat.max left' right'
  | .absolute value =>
      return (← (evaluateIntExpr environment value).mapError BoundEvalError.integer).natAbs
  | .floorDivide value divisor =>
      if divisor = 0 then .error .nonPositiveDivisor
      else return (← value.evaluateTemplate environment recurrenceBounds carried) / divisor
  | .matrixProduct ring inner left right => do
      let ring ← evaluatePositiveDimension environment ring
      let inner ← evaluatePositiveDimension environment inner
      return ring * inner * (← left.evaluateTemplate environment recurrenceBounds carried) *
        (← right.evaluateTemplate environment recurrenceBounds carried)
  | .minimum left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return Nat.min left' right'
  | .recurrenceResult recurrence path =>
      match lookupNaturalRecurrenceBound recurrence path recurrenceBounds.natural with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence recurrence path)
  | .carriedInput path =>
      match lookupCarriedNatural path carried.natural with
      | some value => .ok value
      | none => .error (.escapedCarriedInput path)

inductive IntBoundEvalError where
  | integer (error : IntEvalError)
  | natural (error : BoundEvalError)
  | divisionByZero
  | unresolvedRecurrence (recurrence : FactRecurrenceInstanceRef) (path : IntBoundFactPath)
  | escapedCarriedInput (path : IntBoundFactPath)
  deriving BEq, DecidableEq, Repr

/-- Evaluate a signed integer interval endpoint from parameters and a checked recurrence table. -/
def IntBoundExpr.evaluate
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable) :
    IntBoundExpr → Except IntBoundEvalError Int
  | .integer value => (evaluateIntExpr environment value).mapError .integer
  | .natural value =>
      return (← value.evaluateWithRecurrences environment recurrenceBounds |>.mapError .natural)
  | .negate value => return -(← value.evaluate environment recurrenceBounds)
  | .add left right =>
      return (← left.evaluate environment recurrenceBounds) +
        (← right.evaluate environment recurrenceBounds)
  | .subtract left right =>
      return (← left.evaluate environment recurrenceBounds) -
        (← right.evaluate environment recurrenceBounds)
  | .multiply left right =>
      return (← left.evaluate environment recurrenceBounds) *
        (← right.evaluate environment recurrenceBounds)
  | .divide left right => do
      let denominator ← right.evaluate environment recurrenceBounds
      if denominator = 0 then .error .divisionByZero
      else return (← left.evaluate environment recurrenceBounds) / denominator
  | .minimum left right =>
      return min (← left.evaluate environment recurrenceBounds)
        (← right.evaluate environment recurrenceBounds)
  | .maximum left right =>
      return max (← left.evaluate environment recurrenceBounds)
        (← right.evaluate environment recurrenceBounds)
  | .carriedInput path => .error (.escapedCarriedInput path)
  | .recurrenceResult recurrence path =>
      match lookupIntegerRecurrenceBound recurrence path recurrenceBounds.integer with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence recurrence path)

/-- Signed counterpart of `BoundExpr.evaluateTemplate`, using the same immutable snapshot. -/
def IntBoundExpr.evaluateTemplate
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable)
    (carried : CarriedBoundTable) : IntBoundExpr → Except IntBoundEvalError Int
  | .integer value => (evaluateIntExpr environment value).mapError .integer
  | .natural value =>
      return (← value.evaluateTemplate environment recurrenceBounds carried |>.mapError .natural)
  | .negate value => return -(← value.evaluateTemplate environment recurrenceBounds carried)
  | .add left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return left' + right'
  | .subtract left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return left' - right'
  | .multiply left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return left' * right'
  | .divide left right => do
      let denominator ← right.evaluateTemplate environment recurrenceBounds carried
      if denominator = 0 then .error .divisionByZero
      else return (← left.evaluateTemplate environment recurrenceBounds carried) / denominator
  | .minimum left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return min left' right'
  | .maximum left right => do
      let left' ← left.evaluateTemplate environment recurrenceBounds carried
      let right' ← right.evaluateTemplate environment recurrenceBounds carried
      return max left' right'
  | .carriedInput path =>
      match lookupCarriedInteger path carried.integer with
      | some value => .ok value
      | none => .error (.escapedCarriedInput path)
  | .recurrenceResult recurrence path =>
      match lookupIntegerRecurrenceBound recurrence path recurrenceBounds.integer with
      | some value => .ok value
      | none => .error (.unresolvedRecurrence recurrence path)

theorem BoundExpr.evaluate_parameter_nat (name : String) (value : Nat) :
    (BoundExpr.parameter (.parameter name)).evaluate
      [(name, .integer value)] = .ok value := by
  have evaluated : evaluateIntExpr [(name, .integer value)] (.parameter name) = .ok value := by
    simp [evaluateIntExpr, Mxx.Ir.lookupParam]
  unfold BoundExpr.evaluate BoundExpr.evaluateWithRecurrences evaluateBoundParameter
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
