import Mxx.Certificate.Semantics
import Mxx.Certificate.SymbolicForm

namespace Mxx.Certificate

/-! # Semantics of typed symbolic matrix forms

The relation in this module interprets the analyzer-owned symbolic-form arena against the same
`FactEnvironment` and matrix operations used by ordinary certificate facts.  It introduces no
second matrix evaluator.  Analyzer-only carried and recurrence nodes deliberately have no
constructor here; their denotation must eventually be supplied by trace-derived recurrence
evidence rather than an arbitrary environment callback.
-/

mutual

/-- Denotation of a stable reference into a symbolic-form arena. -/
inductive SymbolicMatrixFormArena.Denotes
    (environment : FactEnvironment)
    (arena : SymbolicMatrixFormArena) : SymbolicMatrixFormRef → Mxx.Matrix → Prop where
  | entry {reference entry value}
      (lookup : arena.lookup reference = some entry)
      (denotes : SymbolicMatrixForm.Denotes environment arena entry.form value) :
      SymbolicMatrixFormArena.Denotes environment arena reference value

/-- Denotation of one symbolic-form node.  Every executable operation delegates to the existing
matrix and scalar denotations.  Relation rewrites preserve the represented matrix value; their
relation-specific decomposition and bound obligations are checked separately by the closed rule
registry.

There are intentionally no constructors for `carriedInput` or `recurrenceResult`. -/
inductive SymbolicMatrixForm.Denotes
    (environment : FactEnvironment)
    (arena : SymbolicMatrixFormArena) : SymbolicMatrixForm → Mxx.Matrix → Prop where
  | signalAtom {expression expressionValue value}
      (lookup : environment.expressionArena.lookupMatrix expression = some expressionValue)
      (denotes : MatrixExpr.Denotes environment expressionValue value) :
      SymbolicMatrixForm.Denotes environment arena (.signalAtom expression) value
  | boundedAtom {expression expressionValue bound value}
      (lookup : environment.expressionArena.lookupMatrix expression = some expressionValue)
      (holds : BoundedMatrixExpr.Holds environment {
        expression := expressionValue
        normBound := bound
      } value) :
      SymbolicMatrixForm.Denotes environment arena (.boundedAtom expression bound) value
  | affineLeaf {form value}
      (holds : AffineForm.Holds environment form value) :
      SymbolicMatrixForm.Denotes environment arena (.affineLeaf form) value
  | add {left right leftValue rightValue}
      (leftDenotes : SymbolicMatrixFormArena.Denotes environment arena left leftValue)
      (rightDenotes : SymbolicMatrixFormArena.Denotes environment arena right rightValue) :
      SymbolicMatrixForm.Denotes environment arena (.add left right)
        (Mxx.matrixAdd leftValue rightValue)
  | negate {expression value}
      (denotes : SymbolicMatrixFormArena.Denotes environment arena expression value) :
      SymbolicMatrixForm.Denotes environment arena (.negate expression)
        (Mxx.matrixNegate value)
  | multiply {left right leftValue rightValue}
      (leftDenotes : SymbolicMatrixFormArena.Denotes environment arena left leftValue)
      (rightDenotes : SymbolicMatrixFormArena.Denotes environment arena right rightValue) :
      SymbolicMatrixForm.Denotes environment arena (.multiply left right)
        (Mxx.matrixMultiply leftValue rightValue)
  | select {index branches indexExpression indexValue branch value}
      (indexLookup : environment.expressionArena.lookupInteger index = some indexExpression)
      (indexDenotes : RuntimeIntExpr.Denotes environment indexExpression indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches.toList[indexValue.toNat]? = some branch)
      (branchDenotes : SymbolicMatrixFormArena.Denotes environment arena branch value) :
      SymbolicMatrixForm.Denotes environment arena (.select index branches) value
  | rowSlice {expression value start stop evaluatedStart evaluatedStop}
      (denotes : SymbolicMatrixFormArena.Denotes environment arena expression value)
      (startEvaluates : evaluateIntExpr environment.parameters start = .ok evaluatedStart)
      (stopEvaluates : evaluateIntExpr environment.parameters stop = .ok evaluatedStop)
      (startNonnegative : 0 ≤ evaluatedStart)
      (ordered : evaluatedStart ≤ evaluatedStop) :
      SymbolicMatrixForm.Denotes environment arena (.rowSlice expression start stop)
        (Mxx.matrixSlice value evaluatedStart.toNat evaluatedStop.toNat 0 value.columns)
  | columnSlice {expression value start stop evaluatedStart evaluatedStop}
      (denotes : SymbolicMatrixFormArena.Denotes environment arena expression value)
      (startEvaluates : evaluateIntExpr environment.parameters start = .ok evaluatedStart)
      (stopEvaluates : evaluateIntExpr environment.parameters stop = .ok evaluatedStop)
      (startNonnegative : 0 ≤ evaluatedStart)
      (ordered : evaluatedStart ≤ evaluatedStop) :
      SymbolicMatrixForm.Denotes environment arena (.columnSlice expression start stop)
        (Mxx.matrixSlice value 0 value.rows evaluatedStart.toNat evaluatedStop.toNat)
  | rowConcat {parts values}
      (denotes : List.Forall₂
        (SymbolicMatrixFormArena.Denotes environment arena) parts.toList values) :
      SymbolicMatrixForm.Denotes environment arena (.rowConcat parts)
        (Mxx.matrixConcatRows values)
  | columnConcat {parts values}
      (denotes : List.Forall₂
        (SymbolicMatrixFormArena.Denotes environment arena) parts.toList values) :
      SymbolicMatrixForm.Denotes environment arena (.columnConcat parts)
        (Mxx.matrixConcatColumns values)
  | diagonalConcat {parts values}
      (denotes : List.Forall₂
        (SymbolicMatrixFormArena.Denotes environment arena) parts.toList values) :
      SymbolicMatrixForm.Denotes environment arena (.diagonalConcat parts)
        (Mxx.matrixConcatDiagonal values)
  | reshape {expression outputType value evaluatedRows evaluatedColumns}
      (denotes : SymbolicMatrixFormArena.Denotes environment arena expression value)
      (rowsEvaluate : evaluateIntExpr environment.parameters outputType.rows = .ok evaluatedRows)
      (columnsEvaluate :
        evaluateIntExpr environment.parameters outputType.columns = .ok evaluatedColumns)
      (rowsNonnegative : 0 ≤ evaluatedRows)
      (columnsNonnegative : 0 ≤ evaluatedColumns) :
      SymbolicMatrixForm.Denotes environment arena (.reshape expression outputType)
        (Mxx.matrixReshape value evaluatedRows.toNat evaluatedColumns.toNat)
  | preimageRewrite {relation input value}
      (inputDenotes : SymbolicMatrixFormArena.Denotes environment arena input value) :
      SymbolicMatrixForm.Denotes environment arena (.preimageRewrite relation input) value
  | gadgetRewrite {relation input value}
      (inputDenotes : SymbolicMatrixFormArena.Denotes environment arena input value) :
      SymbolicMatrixForm.Denotes environment arena (.gadgetRewrite relation input) value

end

/-- A symbolic matrix fact relates one actual matrix to both its immutable exact expression and
its independently normalized decomposition.  Bound-witness soundness is a separate Phase-A
judgment and is therefore not smuggled into this value-denotation predicate. -/
def MatrixSymbolicFact.Holds
    (environment : FactEnvironment)
    (arena : SymbolicMatrixFormArena)
    (fact : MatrixSymbolicFact)
    (value : Mxx.Matrix) : Prop :=
  environment.values fact.subject = some (.matrix value) ∧
    (∃ exactExpression,
      environment.expressionArena.lookupMatrix fact.exactValue = some exactExpression ∧
      MatrixExpr.Denotes environment exactExpression value) ∧
    SymbolicMatrixFormArena.Denotes environment arena fact.decomposition value ∧
    ∀ relation ∈ fact.relations, relation.Holds environment

/-- A matrix-expression leaf can be embedded without changing its denotation. -/
theorem SymbolicMatrixForm.signalAtom_sound
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {expression expressionValue value}
    (lookup : environment.expressionArena.lookupMatrix expression = some expressionValue)
    (denotes : MatrixExpr.Denotes environment expressionValue value) :
    SymbolicMatrixForm.Denotes environment arena (.signalAtom expression) value :=
  .signalAtom lookup denotes

/-- A bounded expression leaf reuses the existing deterministic hard-bound judgment. -/
theorem SymbolicMatrixForm.boundedAtom_sound
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {expression expressionValue bound value}
    (lookup : environment.expressionArena.lookupMatrix expression = some expressionValue)
    (holds : BoundedMatrixExpr.Holds environment {
      expression := expressionValue
      normBound := bound
    } value) :
    SymbolicMatrixForm.Denotes environment arena (.boundedAtom expression bound) value :=
  .boundedAtom lookup holds

/-- An existing affine proof can be embedded as one compact symbolic leaf. -/
theorem SymbolicMatrixForm.affineLeaf_sound
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {form value}
    (holds : AffineForm.Holds environment form value) :
    SymbolicMatrixForm.Denotes environment arena (.affineLeaf form) value :=
  .affineLeaf holds

/-- Symbolic addition uses exactly the runtime matrix-addition operation. -/
theorem SymbolicMatrixForm.add_sound
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {left right leftValue rightValue}
    (leftDenotes : SymbolicMatrixFormArena.Denotes environment arena left leftValue)
    (rightDenotes : SymbolicMatrixFormArena.Denotes environment arena right rightValue) :
    SymbolicMatrixForm.Denotes environment arena (.add left right)
      (Mxx.matrixAdd leftValue rightValue) :=
  .add leftDenotes rightDenotes

/-- Whole-form selection denotes only the branch chosen by the existing integer-expression
semantics; unselected branches neither add values nor bounds. -/
theorem SymbolicMatrixForm.select_sound
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {index branches indexExpression indexValue branch value}
    (indexLookup : environment.expressionArena.lookupInteger index = some indexExpression)
    (indexDenotes : RuntimeIntExpr.Denotes environment indexExpression indexValue)
    (nonnegative : 0 ≤ indexValue)
    (selected : branches.toList[indexValue.toNat]? = some branch)
    (branchDenotes : SymbolicMatrixFormArena.Denotes environment arena branch value) :
    SymbolicMatrixForm.Denotes environment arena (.select index branches) value :=
  .select indexLookup indexDenotes nonnegative selected branchDenotes

/-- Analyzer placeholders cannot acquire a meaning before recurrence evidence is introduced. -/
theorem SymbolicMatrixForm.carriedInput_fail_closed
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {matrixType slot value} :
    ¬ SymbolicMatrixForm.Denotes environment arena (.carriedInput matrixType slot) value := by
  intro denotes
  cases denotes

/-- A recurrence result cannot be interpreted merely by choosing a value in `FactEnvironment`. -/
theorem SymbolicMatrixForm.recurrenceResult_fail_closed
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {matrixType recurrence slot value} :
    ¬ SymbolicMatrixForm.Denotes environment arena
      (.recurrenceResult matrixType recurrence slot) value := by
  intro denotes
  cases denotes

end Mxx.Certificate
