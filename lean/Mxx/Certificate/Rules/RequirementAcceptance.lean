import Mxx.Certificate.Semantics

namespace Mxx.Certificate

/-!
# Closed requirement-acceptance wrappers

This module recognizes the scalar/Boolean wrapper by which a requirement program turns one
selected Boolean recurrence result into its accepted output. The matcher consumes only the
analyzer-produced Boolean expression for the exact frozen output wire. Output names locate that
wire but do not assert the implication.
-/

/-- Structural evidence that one Boolean expression is a direct selection from a recurrence
result family. The expression is an index of the type, so the matcher cannot return evidence for
a different expression. -/
inductive CheckedSelectedRecurrenceBoolean : RuntimeExpr .boolean → Type where
  | familyElement
      (recurrence : SequentialRecurrenceRef)
      (path : AggregateInstancePath)
      (slot : Nat)
      (indexReference : RuntimeExprRef .integer)
      (index : RuntimeExpr .integer) :
      CheckedSelectedRecurrenceBoolean
        (.familyElement .boolean (.recurrenceResult recurrence path slot)
          indexReference index)

def CheckedSelectedRecurrenceBoolean.instance
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression) :
    SequentialRecurrenceInstanceRef :=
  match selected with
  | .familyElement recurrence path .. => { recurrence, path }

def CheckedSelectedRecurrenceBoolean.slot
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression) : Nat :=
  match selected with
  | .familyElement _ _ slot .. => slot

private def matchSelectedRecurrenceBoolean
    (expression : RuntimeExpr .boolean) :
    Option (CheckedSelectedRecurrenceBoolean expression) :=
  match expression with
  | .familyElement .boolean
      (.recurrenceResult recurrence path slot) indexReference index =>
      some (.familyElement recurrence path slot indexReference index)
  | _ => none

/-- The four symmetric syntactic arrangements accepted by the closed wrapper. Exactly one
multiplication operand is the selected recurrence Boolean. -/
inductive CheckedRequirementAcceptanceWrapper : RuntimeExpr .boolean → Type where
  | productFirst
      (validity selected : RuntimeExpr .boolean)
      (selectedRecurrence : CheckedSelectedRecurrenceBoolean selected) :
      CheckedRequirementAcceptanceWrapper
        (.compare .equal
          (.intBinary .multiply (.boolToInt validity) (.boolToInt selected))
          (.intConstant 1))
  | selectedFirst
      (selected validity : RuntimeExpr .boolean)
      (selectedRecurrence : CheckedSelectedRecurrenceBoolean selected) :
      CheckedRequirementAcceptanceWrapper
        (.compare .equal
          (.intBinary .multiply (.boolToInt selected) (.boolToInt validity))
          (.intConstant 1))
  | oneFirst
      (validity selected : RuntimeExpr .boolean)
      (selectedRecurrence : CheckedSelectedRecurrenceBoolean selected) :
      CheckedRequirementAcceptanceWrapper
        (.compare .equal
          (.intConstant 1)
          (.intBinary .multiply (.boolToInt validity) (.boolToInt selected)))
  | oneFirstSelectedFirst
      (selected validity : RuntimeExpr .boolean)
      (selectedRecurrence : CheckedSelectedRecurrenceBoolean selected) :
      CheckedRequirementAcceptanceWrapper
        (.compare .equal
          (.intConstant 1)
          (.intBinary .multiply (.boolToInt selected) (.boolToInt validity)))

def CheckedRequirementAcceptanceWrapper.selectedRecurrence
    {expression : RuntimeExpr .boolean}
    (wrapper : CheckedRequirementAcceptanceWrapper expression) :
    SequentialRecurrenceInstanceRef :=
  match wrapper with
  | .productFirst _ _ selected | .selectedFirst _ _ selected |
      .oneFirst _ _ selected | .oneFirstSelectedFirst _ _ selected => selected.instance

def CheckedRequirementAcceptanceWrapper.selectedExpression
    {expression : RuntimeExpr .boolean}
    (wrapper : CheckedRequirementAcceptanceWrapper expression) : RuntimeExpr .boolean :=
  match wrapper with
  | .productFirst _ selected _ | .selectedFirst selected _ _ |
      .oneFirst _ selected _ | .oneFirstSelectedFirst selected _ _ => selected

def CheckedRequirementAcceptanceWrapper.selectedSlot
    {expression : RuntimeExpr .boolean}
    (wrapper : CheckedRequirementAcceptanceWrapper expression) : Nat :=
  match wrapper with
  | .productFirst _ _ selected | .selectedFirst _ _ selected |
      .oneFirst _ _ selected | .oneFirstSelectedFirst _ _ selected => selected.slot

/-- Analyzer-owned location and checked wrapper for one actual requirement output. The output name
is only the frozen program-output locator; `wrapper` is the evidence that gives it semantics. -/
structure CheckedRequirementAcceptance where
  requirementIndex : Nat
  outputName : String
  outputWire : CoreWireRef
  outputFact : BooleanFact
  wrapper : CheckedRequirementAcceptanceWrapper outputFact.expression

def CheckedRequirementAcceptance.summary
    (acceptance : CheckedRequirementAcceptance) : RequirementAcceptanceSummary := {
  requirementIndex := acceptance.requirementIndex
  outputName := acceptance.outputName
  outputWire := acceptance.outputWire
  selectedRecurrence := acceptance.wrapper.selectedRecurrence
  selectedSlot := acceptance.wrapper.selectedSlot
}

/-- Recognize the exact accepted-output expression. There is no fallback based on a stage,
output, protocol, or semantic-label name. -/
def checkRequirementAcceptanceWrapper
    (outputExpression : RuntimeExpr .boolean) :
    Option (CheckedRequirementAcceptanceWrapper outputExpression) :=
  match outputExpression with
  | .compare .equal
      (.intBinary .multiply (.boolToInt left) (.boolToInt right)) (.intConstant 1) =>
      match matchSelectedRecurrenceBoolean left, matchSelectedRecurrenceBoolean right with
      | some _, some _ | none, none => none
      | none, some selected => some (.productFirst left right selected)
      | some selected, none => some (.selectedFirst left right selected)
  | .compare .equal (.intConstant 1)
      (.intBinary .multiply (.boolToInt left) (.boolToInt right)) =>
      match matchSelectedRecurrenceBoolean left, matchSelectedRecurrenceBoolean right with
      | some _, some _ | none, none => none
      | none, some selected => some (.oneFirst left right selected)
      | some selected, none => some (.oneFirstSelectedFirst left right selected)
  | _ => none

/-- Reconstruct acceptance evidence for one exact frozen requirement output from the analyzer's
fact table.  This is the single matcher used both while constructing `AnalysisResult` and while
connecting that result to a closed execution trace. -/
def checkRequirementAcceptance
    (facts : ScopedWireFactTable)
    (requirementIndex : Nat)
    (program : Mxx.Ir.Prog)
    (outputName : String) : Option CheckedRequirementAcceptance := do
  let outputRef ← (program.root.outputs.find? fun output => output.1 = outputName).map (·.2)
  let outputWire : CoreWireRef := {
    stage := ⟨s!"$requirement:{requirementIndex}"⟩
    scope := ⟨[]⟩
    node := ⟨outputRef.node⟩
    port := outputRef.port
  }
  let scopedFact ← facts.find? fun fact => fact.wire = outputWire
  let outputFact ← match scopedFact.fact with
    | .boolean fact => some fact
    | _ => none
  let wrapper ← checkRequirementAcceptanceWrapper outputFact.expression
  return { requirementIndex, outputName, outputWire, outputFact, wrapper }

private theorem acceptedProduct_selected_true
    (validity selected : Bool)
    (accepted : Mxx.Ir.evaluateIntCompare .equal
      ((if validity then 1 else 0) * (if selected then 1 else 0)) 1 = true) :
    selected = true := by
  cases validity <;> cases selected <;>
    simp [Mxx.Ir.evaluateIntCompare] at accepted ⊢

private theorem compareEqual_denotes
    {environment : FactEnvironment}
    {left right : RuntimeExpr .integer}
    {result : Bool}
    (denotes : RuntimeBoolExpr.Denotes environment (.compare .equal left right) result) :
    ∃ leftValue rightValue,
      RuntimeIntExpr.Denotes environment left leftValue ∧
      RuntimeIntExpr.Denotes environment right rightValue ∧
      result = Mxx.Ir.evaluateIntCompare .equal leftValue rightValue := by
  cases denotes with
  | compare leftDenotes rightDenotes => exact ⟨_, _, leftDenotes, rightDenotes, rfl⟩

private theorem intMultiply_denotes
    {environment : FactEnvironment}
    {left right : RuntimeExpr .integer}
    {value : Int}
    (denotes : RuntimeIntExpr.Denotes environment
      (.intBinary .multiply left right) value) :
    ∃ leftValue rightValue,
      RuntimeIntExpr.Denotes environment left leftValue ∧
      RuntimeIntExpr.Denotes environment right rightValue ∧
      value = leftValue * rightValue := by
  cases denotes with
  | intBinary leftDenotes rightDenotes evaluates =>
      simp [Mxx.Ir.evaluateIntBinary] at evaluates
      exact ⟨_, _, leftDenotes, rightDenotes, evaluates.symm⟩

private theorem boolToInt_denotes
    {environment : FactEnvironment}
    {expression : RuntimeExpr .boolean}
    {value : Int}
    (denotes : RuntimeIntExpr.Denotes environment (.boolToInt expression) value) :
    ∃ boolean,
      RuntimeBoolExpr.Denotes environment expression boolean ∧
      value = if boolean then 1 else 0 := by
  cases denotes with
  | boolToInt input => exact ⟨_, input, rfl⟩

private theorem intConstantOne_denotes
    {environment : FactEnvironment}
    {value : Int}
    (denotes : RuntimeIntExpr.Denotes environment (.intConstant 1) value) : value = 1 := by
  cases denotes
  rfl

private theorem productFirst_selected_true
    {environment : FactEnvironment}
    {validity selected : RuntimeExpr .boolean}
    (outputDenotes : RuntimeBoolExpr.Denotes environment
      (.compare .equal
        (.intBinary .multiply (.boolToInt validity) (.boolToInt selected))
        (.intConstant 1)) true) :
    RuntimeBoolExpr.Denotes environment selected true := by
  obtain ⟨product, one, productDenotes, oneDenotes, accepted⟩ :=
    compareEqual_denotes outputDenotes
  obtain ⟨validInt, selectedInt, validIntDenotes, selectedIntDenotes, productEq⟩ :=
    intMultiply_denotes productDenotes
  obtain ⟨valid, validDenotes, validIntEq⟩ := boolToInt_denotes validIntDenotes
  obtain ⟨selectedValue, selectedDenotes, selectedIntEq⟩ :=
    boolToInt_denotes selectedIntDenotes
  have oneEq := intConstantOne_denotes oneDenotes
  have selectedTrue := acceptedProduct_selected_true valid selectedValue (by
    have accepted' := accepted.symm
    rw [productEq, validIntEq, selectedIntEq, oneEq] at accepted'
    exact accepted')
  simpa [selectedTrue] using selectedDenotes

private theorem intMultiply_swap_denotes
    {environment : FactEnvironment}
    {left right : RuntimeExpr .integer}
    {value : Int}
    (denotes : RuntimeIntExpr.Denotes environment
      (.intBinary .multiply left right) value) :
    RuntimeIntExpr.Denotes environment (.intBinary .multiply right left) value := by
  obtain ⟨leftValue, rightValue, leftDenotes, rightDenotes, valueEq⟩ :=
    intMultiply_denotes denotes
  exact .intBinary rightDenotes leftDenotes (by
    simp [Mxx.Ir.evaluateIntBinary, valueEq, Int.mul_comm])

private theorem compareEqual_swap_denotes
    {environment : FactEnvironment}
    {left right : RuntimeExpr .integer}
    {result : Bool}
    (denotes : RuntimeBoolExpr.Denotes environment (.compare .equal left right) result) :
    RuntimeBoolExpr.Denotes environment (.compare .equal right left) result := by
  obtain ⟨leftValue, rightValue, leftDenotes, rightDenotes, resultEq⟩ :=
    compareEqual_denotes denotes
  have compareEq : Mxx.Ir.evaluateIntCompare .equal rightValue leftValue = result := by
    rw [resultEq]
    simp [Mxx.Ir.evaluateIntCompare, eq_comm]
  have swapped : RuntimeBoolExpr.Denotes environment (.compare .equal right left)
      (Mxx.Ir.evaluateIntCompare .equal rightValue leftValue) :=
    .compare rightDenotes leftDenotes
  simpa [compareEq] using swapped

/-- The accepted wrapper implies truth of the selected recurrence Boolean. This theorem reasons
through the existing exact scalar denotation; it does not assume that the validity factor is one
or evaluate a second copy of the IR. -/
theorem CheckedRequirementAcceptanceWrapper.selected_true
    {environment : FactEnvironment}
    {outputExpression : RuntimeExpr .boolean}
    (wrapper : CheckedRequirementAcceptanceWrapper outputExpression)
    (outputDenotes : RuntimeBoolExpr.Denotes environment outputExpression true) :
    RuntimeBoolExpr.Denotes environment wrapper.selectedExpression true := by
  cases wrapper with
  | productFirst validity selected selectedRecurrence =>
      exact productFirst_selected_true outputDenotes
  | selectedFirst selected validity selectedRecurrence =>
      have swappedProduct : RuntimeBoolExpr.Denotes environment
          (.compare .equal
            (.intBinary .multiply (.boolToInt validity) (.boolToInt selected))
            (.intConstant 1)) true := by
        obtain ⟨product, one, productDenotes, oneDenotes, resultEq⟩ :=
          compareEqual_denotes outputDenotes
        rw [resultEq]
        exact .compare (intMultiply_swap_denotes productDenotes) oneDenotes
      exact productFirst_selected_true swappedProduct
  | oneFirst validity selected selectedRecurrence =>
      exact productFirst_selected_true (compareEqual_swap_denotes outputDenotes)
  | oneFirstSelectedFirst selected validity selectedRecurrence =>
      have productFirst := compareEqual_swap_denotes outputDenotes
      obtain ⟨product, one, productDenotes, oneDenotes, resultEq⟩ :=
        compareEqual_denotes productFirst
      have swappedProduct : RuntimeBoolExpr.Denotes environment
          (.compare .equal
            (.intBinary .multiply (.boolToInt validity) (.boolToInt selected))
            (.intConstant 1)) true := by
        rw [resultEq]
        exact .compare (intMultiply_swap_denotes productDenotes) oneDenotes
      exact productFirst_selected_true swappedProduct

private def fixtureRecurrence : SequentialRecurrenceRef := {
  site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨7⟩ }
}

private def fixtureIndex : RuntimeExpr .integer := .intConstant 3
private def fixtureIndexRef : RuntimeExprRef .integer := ⟨9⟩
private def fixtureSelected : RuntimeExpr .boolean :=
  .familyElement .boolean (.recurrenceResult fixtureRecurrence [] 0)
    fixtureIndexRef fixtureIndex
private def fixtureValidity : RuntimeExpr .boolean := .boolConstant true
private def fixtureOutput : RuntimeExpr .boolean :=
  .compare .equal
    (.intBinary .multiply (.boolToInt fixtureValidity) (.boolToInt fixtureSelected))
    (.intConstant 1)

example : (checkRequirementAcceptanceWrapper fixtureOutput).isSome = true := rfl

example : checkRequirementAcceptanceWrapper
    (.compare .equal (.boolToInt fixtureSelected) (.intConstant 1)) = none := rfl

example : checkRequirementAcceptanceWrapper
    (.compare .equal
      (.intBinary .multiply (.boolToInt fixtureSelected) (.boolToInt fixtureSelected))
      (.intConstant 1)) = none := rfl

end Mxx.Certificate
