import Mxx.Certificate.Semantics

namespace Mxx.Certificate

/-! # Sequential-loop recurrence soundness

This module contains the proof infrastructure shared by closed sequential-loop rules.  It does
not infer an invariant and it does not reinterpret a downstream operation pointwise.  A caller
must supply the one-step preservation theorem obtained by composing the verified local rules of
the actual loop body.  The theorems below only transport that theorem across the executable
`Mxx.Ir.SequentialIterationsTrace`.
-/

/-- A proof about a loop state together with its fixed carried arity. -/
def FactTupleHolds (carriedArity : Nat) (predicate : List Mxx.Ir.Value → Prop)
    (values : List Mxx.Ir.Value) : Prop :=
  values.length = carriedArity ∧ predicate values

/-- The executable child relation for exactly one sequential-loop iteration. -/
def ExecutesLoopBody
    (runChild : Mxx.Ir.ChildRunner)
    (definition : String)
    (params : Mxx.Ir.ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Mxx.Ir.Value)
    (index : Nat)
    (state next : List Mxx.Ir.Value) : Prop :=
  ∃ evaluatedBindings,
    Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
      some evaluatedBindings ∧
    next ∈ runChild definition
      (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
      (state ++ invariantArguments)

/-- A body analyzer derives this theorem by composing the local soundness theorem of each body
node.  This wrapper deliberately accepts no asserted output fact or user-provided invariant. -/
theorem recurrenceStepPreserves
    {runChild : Mxx.Ir.ChildRunner}
    {definition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {indexSlot count carriedArity : Nat}
    {bindings : List (String × IntExpr)}
    {invariantArguments state next : List Mxx.Ir.Value}
    {statePredicate : List Mxx.Ir.Value → Prop}
    (index : Nat)
    (indexRange : index < count)
    (carriedFacts : FactTupleHolds carriedArity statePredicate state)
    (bodyExecution : ExecutesLoopBody runChild definition params indexSlot bindings
      invariantArguments index state next)
    (bodyAnalysisSound : ∀ index state next,
      index < count →
      FactTupleHolds carriedArity statePredicate state →
      ExecutesLoopBody runChild definition params indexSlot bindings
        invariantArguments index state next →
      FactTupleHolds carriedArity statePredicate next) :
    FactTupleHolds carriedArity statePredicate next :=
  bodyAnalysisSound index state next indexRange carriedFacts bodyExecution

/-- A trace whose indices all lie below `count` preserves the derived recurrence predicate.
The induction follows the executable trace and therefore introduces no second evaluator. -/
theorem sequentialIterationsTrace_recurrence
    {runChild : Mxx.Ir.ChildRunner}
    {definition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {indexSlot count carriedArity : Nat}
    {bindings : List (String × IntExpr)}
    {invariantArguments : List Mxx.Ir.Value}
    {statePredicate : List Mxx.Ir.Value → Prop}
    (bodyAnalysisSound : ∀ index state next,
      index < count →
      FactTupleHolds carriedArity statePredicate state →
      ExecutesLoopBody runChild definition params indexSlot bindings
        invariantArguments index state next →
      FactTupleHolds carriedArity statePredicate next) :
    ∀ {indices initial final},
      Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments indices initial final →
      (∀ index ∈ indices, index < count) →
      FactTupleHolds carriedArity statePredicate initial →
      FactTupleHolds carriedArity statePredicate final := by
  intro indices initial final trace indicesInRange initialFacts
  induction trace with
  | nil => exact initialFacts
  | cons index tail state evaluatedBindings next final bindingsEvaluate childMember rest
      induction =>
      apply induction
      · intro tailIndex tailMember
        exact indicesInRange tailIndex (List.mem_cons_of_mem index tailMember)
      · apply recurrenceStepPreserves index (indicesInRange index (by simp)) initialFacts
        exact ⟨evaluatedBindings, bindingsEvaluate, childMember⟩
        exact bodyAnalysisSound

/-- A parameter-count sequential loop preserves the body-derived recurrence property without
unrolling its body graph.  Only the list of runtime iteration indices is traversed. -/
theorem evaluateSequentialIterations_recurrence
    (runChild : Mxx.Ir.ChildRunner)
    (definition : String)
    (params : Mxx.Ir.ParamEnvironment)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (invariantArguments : List Mxx.Ir.Value)
    (count carriedArity : Nat)
    (states : List (List Mxx.Ir.Value))
    (statePredicate : List Mxx.Ir.Value → Prop)
    (initial : ∀ state ∈ states, FactTupleHolds carriedArity statePredicate state)
    (bodyAnalysisSound : ∀ index state next,
      index < count →
      FactTupleHolds carriedArity statePredicate state →
      ExecutesLoopBody runChild definition params indexSlot bindings
        invariantArguments index state next →
      FactTupleHolds carriedArity statePredicate next)
    {final : List Mxx.Ir.Value}
    (finalMember : final ∈
      Mxx.Ir.evaluateSequentialIterations runChild definition params indexSlot bindings
        invariantArguments (List.range count) states) :
    FactTupleHolds carriedArity statePredicate final := by
  obtain ⟨first, firstMember, trace⟩ :=
    (Mxx.Ir.mem_evaluateSequentialIterations_iff_exists_trace runChild definition params indexSlot
      bindings invariantArguments (List.range count) states final).mp finalMember
  apply sequentialIterationsTrace_recurrence bodyAnalysisSound trace
  · intro index indexMember
    exact List.mem_range.mp indexMember
  · exact initial first firstMember

/-- Local soundness interface for an executable `SequentialLoop` node.  The only semantic
reduction is `Mxx.Ir.evaluateNode_sequentialLoop_of_arguments`; recurrence reasoning is delegated
to `evaluateSequentialIterations_recurrence`. -/
theorem evaluateNode_sequentialLoop_recurrence
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (definition : String)
    (countExpression : IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (carriedCount : Nat)
    (argumentRefs : List Mxx.Ir.WireRef)
    (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr)
    (values : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (statePredicate : List Mxx.Ir.Value → Prop)
    (argumentsEvaluate :
      argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) = some values)
    (countEvaluate : countExpression.evaluate params = some evaluatedCount)
    (initial : FactTupleHolds carriedCount statePredicate (values.take carriedCount))
    (bodyAnalysisSound : ∀ index state next,
      index < evaluatedCount.toNat →
      FactTupleHolds carriedCount statePredicate state →
      ExecutesLoopBody runChild definition params indexSlot bindings
        (values.drop carriedCount) index state next →
      FactTupleHolds carriedCount statePredicate next)
    {final : List Mxx.Ir.Value}
    (finalMember : final ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
      arguments := argumentRefs
      outputCount
      outputTypes
    }) :
    FactTupleHolds carriedCount statePredicate final := by
  have nodeEvaluation :
      Mxx.Ir.evaluateNode runChild samplers params inputs wires {
        kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
        arguments := argumentRefs
        outputCount
        outputTypes
      } =
      Mxx.Ir.evaluateSequentialIterations runChild definition params indexSlot bindings
        (values.drop carriedCount) (List.range evaluatedCount.toNat)
        [values.take carriedCount] := by
    let baseNode : Mxx.Ir.Node := {
      kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
      arguments := argumentRefs
      outputCount
    }
    calc
      _ = Mxx.Ir.evaluateNode runChild samplers params inputs wires baseNode :=
        evaluateNode_outputTypes_irrelevant runChild samplers params inputs wires baseNode
          outputTypes
      _ = _ := Mxx.Ir.evaluateNode_sequentialLoop_of_arguments runChild samplers params inputs wires
        definition countExpression indexSlot bindings carriedCount argumentRefs outputCount values
        evaluatedCount argumentsEvaluate countEvaluate
  apply evaluateSequentialIterations_recurrence runChild definition params indexSlot bindings
    (values.drop carriedCount) evaluatedCount.toNat carriedCount [values.take carriedCount]
    statePredicate
  · intro state stateMember
    simp only [List.mem_singleton] at stateMember
    simpa [stateMember] using initial
  · exact bodyAnalysisSound
  · rw [← nodeEvaluation]
    exact finalMember

/-- Exact semantic evidence for one analyzed recurrence occurrence.  The evidence contains a
selected member of the real executable node support and its `SequentialIterationsTrace`; it is
not an asserted invariant or a caller-provided numeric recurrence. -/
structure RecurrenceSemanticResult
    (analysis : AnalysisResult)
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (recurrenceInstance : FactRecurrenceInstanceRef) where
  recurrence : FactRecurrence
  uniqueResolution : analysis.recurrences.filter (fun entry => entry.1 = recurrenceInstance) =
    [(recurrenceInstance, recurrence)]
  definition : String
  countExpression : IntExpr
  indexSlot : Nat
  bindings : List (String × IntExpr)
  carriedCount : Nat
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  countMatches : recurrence.count = countExpression
  carriedArityMatches : recurrence.carriedArity = carriedCount
  iterationSlotMatches : recurrence.iterationVariable.slot = indexSlot
  before : Mxx.Ir.WireEnvironment
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  countNonnegative : 0 ≤ evaluatedCount
  argumentsEvaluate :
    argumentRefs.mapM (fun wire => Mxx.Ir.lookupWire wire before) = some argumentValues
  countEvaluate : countExpression.evaluate params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before {
    kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
    arguments := argumentRefs
    outputCount
    outputTypes
  }
  initial : List Mxx.Ir.Value
  final : List Mxx.Ir.Value
  initialEq : initial = argumentValues.take carriedCount
  finalEq : final = nodeValues
  executionTrace : Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
    (argumentValues.drop carriedCount) (List.range evaluatedCount.toNat) initial final

/-- A recurrence result slot is bound to the exact final state selected by its trace. -/
def RecurrenceSemanticResult.slotValue
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {recurrenceInstance : FactRecurrenceInstanceRef}
    (result : RecurrenceSemanticResult analysis runChild samplers params inputs recurrenceInstance)
    (slot : Nat) : Option Mxx.Ir.Value :=
  result.final[slot]?

/-- Construct recurrence evidence by inverting the selected executable node member.  The trace
is obtained from `mem_evaluateNode_sequentialLoop_iff_trace`; no loop is re-executed. -/
def RecurrenceSemanticResult.ofNodeMember
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {recurrenceInstance : FactRecurrenceInstanceRef}
    (recurrence : FactRecurrence)
    (uniqueResolution : analysis.recurrences.filter
      (fun entry => entry.1 = recurrenceInstance) = [(recurrenceInstance, recurrence)])
    (definition : String)
    (countExpression : IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (carriedCount : Nat)
    (argumentRefs : List Mxx.Ir.WireRef)
    (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr)
    (before : Mxx.Ir.WireEnvironment)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (countNonnegative : 0 ≤ evaluatedCount)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => Mxx.Ir.lookupWire wire before) =
      some argumentValues)
    (countEvaluate : countExpression.evaluate params = some evaluatedCount)
    (countMatches : recurrence.count = countExpression)
    (carriedArityMatches : recurrence.carriedArity = carriedCount)
    (iterationSlotMatches : recurrence.iterationVariable.slot = indexSlot)
    (nodeValues : List Mxx.Ir.Value)
    (nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before {
      kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
      arguments := argumentRefs
      outputCount
      outputTypes
    }) :
    RecurrenceSemanticResult analysis runChild samplers params inputs recurrenceInstance := by
  let baseNode : Mxx.Ir.Node := {
    kind := .sequentialLoop definition countExpression indexSlot bindings carriedCount
    arguments := argumentRefs
    outputCount
  }
  have baseMember : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before
      baseNode := by
    rw [← evaluateNode_outputTypes_irrelevant runChild samplers params inputs before baseNode
      outputTypes]
    exact nodeMember
  have executionTrace :=
    (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace runChild samplers params inputs before
      definition countExpression indexSlot bindings carriedCount argumentRefs outputCount
      argumentValues evaluatedCount argumentsEvaluate countEvaluate nodeValues).mp baseMember
  exact {
    recurrence
    uniqueResolution
    countMatches
    carriedArityMatches
    iterationSlotMatches
    definition
    countExpression
    indexSlot
    bindings
    carriedCount
    argumentRefs
    outputCount
    outputTypes
    before
    argumentValues
    evaluatedCount
    countNonnegative
    argumentsEvaluate
    countEvaluate
    nodeValues
    nodeMember
    initial := argumentValues.take carriedCount
    final := nodeValues
    initialEq := rfl
    finalEq := rfl
    executionTrace
  }

/-! ## Typed result paths

The recurrence result constructors are only safe after their path has been checked against the
fixed body-output schema.  Keeping this validation typed prevents a coefficient, noise, and total
bound from being confused merely because they use the same natural-number slot.
-/

def MatrixFactPath.carriedSlot : MatrixFactPath → Nat
  | .exactExpression slot | .affineCoefficient slot _ | .affineBasis slot _ |
      .familyElement slot _ _ => slot

private def MatrixFactPath.validAt
    (rootSlot : Nat) : MatrixFactPath → ValueFactSchema → Bool
  | .exactExpression slot, .matrix _ .exact _ _ => slot == rootSlot
  | .affineCoefficient slot term, .matrix _ (.affine terms) _ _
  | .affineBasis slot term, .matrix _ (.affine terms) _ _ =>
      slot == rootSlot && term < terms.length
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

private def MatrixFactPath.typeAt
    (rootSlot : Nat) : MatrixFactPath → ValueFactSchema → Option MatrixTypeExpr
  | .exactExpression slot, .matrix type .exact _ _ =>
      if slot == rootSlot then some type else none
  | .affineCoefficient slot term, .matrix _ (.affine terms) _ _ => do
      if slot != rootSlot then none else
      return (← terms[term]?).coefficientType
  | .affineBasis slot term, .matrix _ (.affine terms) _ _ => do
      if slot != rootSlot then none else
      return (← terms[term]?).basisType
  | .familyElement slot _ nested, .family _ element =>
      if slot == rootSlot then nested.typeAt rootSlot element else none
  | _, _ => none

private def MatrixFactPath.checkedType {arity : Nat}
    (path : MatrixFactPath)
    (schemas : Vector ValueFactTemplate arity) : Option MatrixTypeExpr := do
  let slot := path.carriedSlot
  let template ← schemas[slot]?
  path.typeAt slot template.schema

def MatrixFactPath.valid {arity : Nat}
    (path : MatrixFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def BoundFactPath.carriedSlot : BoundFactPath → Nat
  | .affineCoefficientBound slot _ | .affineNoiseBound slot | .matrixTotalBound slot |
      .familyElement slot _ _ => slot

private def BoundFactPath.validAt
    (rootSlot : Nat) : BoundFactPath → ValueFactSchema → Bool
  | .affineCoefficientBound slot term, .matrix _ (.affine terms) _ _ =>
      slot == rootSlot && term < terms.length
  | .affineNoiseBound slot, .matrix _ (.affine _) _ _ => slot == rootSlot
  | .matrixTotalBound slot, .matrix .. => slot == rootSlot
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

def BoundFactPath.valid {arity : Nat}
    (path : BoundFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def RuntimeFactPath.carriedSlot : {type : RuntimeScalarType} → RuntimeFactPath type → Nat
  | _, .integerValue slot | _, .booleanValue slot | _, .familyElement slot _ _ => slot

private def RuntimeFactPath.validAt
    (rootSlot : Nat) : {type : RuntimeScalarType} → RuntimeFactPath type → ValueFactSchema → Bool
  | _, .integerValue slot, .integer => slot == rootSlot
  | _, .booleanValue slot, .boolean => slot == rootSlot
  | _, .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _, _ => false

def RuntimeFactPath.valid {arity : Nat} {type : RuntimeScalarType}
    (path : RuntimeFactPath type)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

def IntBoundFactPath.carriedSlot : IntBoundFactPath → Nat
  | .lower slot | .upper slot | .familyElement slot _ _ => slot

private def IntBoundFactPath.validAt
    (rootSlot : Nat) : IntBoundFactPath → ValueFactSchema → Bool
  | .lower slot, .integer | .upper slot, .integer => slot == rootSlot
  | .familyElement slot _ nested, .family _ element =>
      slot == rootSlot && nested.validAt rootSlot element
  | _, _ => false

def IntBoundFactPath.valid {arity : Nat}
    (path : IntBoundFactPath)
    (schemas : Vector ValueFactTemplate arity) : Bool :=
  let slot := path.carriedSlot
  if slot < arity then
    match schemas[slot]? with
    | some template => path.validAt slot template.schema
    | none => false
  else false

theorem MatrixFactPath.valid_carriedSlot_lt
    {arity : Nat}
    {path : MatrixFactPath}
    {schemas : Vector ValueFactTemplate arity}
    (valid : path.valid schemas) :
    path.carriedSlot < arity := by
  simp only [MatrixFactPath.valid] at valid
  split at valid
  · assumption
  · simp at valid

theorem BoundFactPath.valid_carriedSlot_lt
    {arity : Nat}
    {path : BoundFactPath}
    {schemas : Vector ValueFactTemplate arity}
    (valid : path.valid schemas) :
    path.carriedSlot < arity := by
  simp only [BoundFactPath.valid] at valid
  split at valid
  · assumption
  · simp at valid

private def diamondCarriedFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def diamondCarriedFixtureTemplates : Vector ValueFactTemplate 4 := ⟨#[
  {
    fact := .integer {
      expression := .carriedInput (.integerValue 0)
      lower := .carriedInput (.lower 0)
      upper := .carriedInput (.upper 0)
    }
    schema := .integer
  },
  {
    fact := .boolean { expression := .carriedInput (.booleanValue 1) }
    schema := .boolean
  },
  {
    fact := .family {
      aggregate := .carriedInput 2
      count := .constant 8
      elementSchema := .matrix diamondCarriedFixtureType .exact [] .unknown
    }
    schema := .family (.constant 8)
      (.matrix diamondCarriedFixtureType .exact [] .unknown)
  },
  {
    fact := .matrix {
      subject := .protocolInput ⟨"diamond-matrix-carried"⟩
      primary := .exact (.carriedInput diamondCarriedFixtureType (.exactExpression 3))
      relations := []
      totalNormBound := .carriedInput (.matrixTotalBound 3)
    }
    schema := .matrix diamondCarriedFixtureType .exact [] .unknown
  }
], rfl⟩

/-- The four carried shapes present in the generated Diamond workflow are accepted by the same
recursive schema checker: integer, Boolean, indexed matrix family, and matrix. -/
example : (RuntimeFactPath.integerValue 0).valid diamondCarriedFixtureTemplates = true := rfl
example : (RuntimeFactPath.booleanValue 1).valid diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.familyElement 2 ⟨0⟩ (.exactExpression 2)).valid
    diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.exactExpression 3).valid diamondCarriedFixtureTemplates = true := rfl
example : (MatrixFactPath.exactExpression 3).checkedType diamondCarriedFixtureTemplates =
    some diamondCarriedFixtureType := rfl

/-- A nested path cannot silently switch to a different carried slot. -/
example : (MatrixFactPath.familyElement 2 ⟨0⟩ (.exactExpression 3)).valid
    diamondCarriedFixtureTemplates = false := rfl


/-- Project one recurrence output without unrolling. Every carried path is checked against the
recursive body-output schema before it becomes an instance-qualified recurrence reference. -/
def projectRecurrenceOutput {arity : Nat}
    (recurrence : FactRecurrenceInstanceRef)
    (schemas : Vector ValueFactTemplate arity)
    (slot : Nat)
    (output : ValueInstanceRef) : Option ValueFact := do
  let template ← schemas[slot]?
  match template.fact with
  | .matrix fact => do
      if !fact.relations.isEmpty then none else
      let primary : MatrixPrimaryForm ← match fact.primary with
        | .exact _ =>
            let path := MatrixFactPath.exactExpression slot
            pure (.exact (.loopResult (← path.checkedType schemas) recurrence path))
        | .affine form => do
            let terms ← form.terms.zipIdx.mapM fun (term, termIndex) => do
              let coefficientPath := MatrixFactPath.affineCoefficient slot termIndex
              let basisPath := MatrixFactPath.affineBasis slot termIndex
              return {
                coefficient := {
                  expression := .loopResult (← coefficientPath.checkedType schemas)
                    recurrence coefficientPath
                  normBound := .recurrenceResult recurrence
                    (.affineCoefficientBound slot termIndex)
                }
                basis := .loopResult (← basisPath.checkedType schemas) recurrence basisPath
                mode := term.mode
              }
            pure (.affine {
              terms
              noiseBound := .recurrenceResult recurrence (.affineNoiseBound slot)
            })
      return .matrix {
        fact with
        subject := output
        primary
        relations := []
        totalNormBound := .recurrenceResult recurrence (.matrixTotalBound slot)
      }
  | .integer _ => return .integer {
      expression := .intWire (.recurrenceResult recurrence slot)
      lower := .recurrenceResult recurrence (.lower slot)
      upper := .recurrenceResult recurrence (.upper slot)
    }
  | .boolean _ => return .boolean {
      expression := .boolWire (.recurrenceResult recurrence slot)
    }
  | .family fact => return .family {
      fact with aggregate := .recurrenceResult recurrence.recurrence recurrence.path slot
    }
  | _ => none

private def compactProjectionFixtureRecurrence : FactRecurrenceInstanceRef := {
  recurrence := ⟨"compact-projection"⟩
  path := []
}

/-- Scalar recurrence outputs refer directly to the final carried slot, never to a rewritten
copy of the one-step body expression. -/
example : projectRecurrenceOutput compactProjectionFixtureRecurrence
    diamondCarriedFixtureTemplates 0 (.protocolInput ⟨"integer-output"⟩) = some (.integer {
      expression := .intWire (.recurrenceResult compactProjectionFixtureRecurrence 0)
      lower := .recurrenceResult compactProjectionFixtureRecurrence (.lower 0)
      upper := .recurrenceResult compactProjectionFixtureRecurrence (.upper 0)
    }) := rfl

/-- Matrix recurrence outputs are compact typed paths. In particular, the body-local carried
placeholder is not substituted into a second symbolic expression graph. -/
example : projectRecurrenceOutput compactProjectionFixtureRecurrence
    diamondCarriedFixtureTemplates 3 (.protocolInput ⟨"matrix-output"⟩) = some (.matrix {
      subject := .protocolInput ⟨"matrix-output"⟩
      primary := .exact
        (.loopResult diamondCarriedFixtureType compactProjectionFixtureRecurrence
          (.exactExpression 3))
      relations := []
      totalNormBound := .recurrenceResult compactProjectionFixtureRecurrence
        (.matrixTotalBound 3)
    }) := rfl

def FamilyAggregateRef.hasCarriedInput : FamilyAggregateRef → Bool
  | .carriedInput _ => true
  | .familyElement parent _ => parent.hasCarriedInput
  | _ => false

def ValueInstanceRef.hasCarriedInput : ValueInstanceRef → Bool
  | .familyElement aggregate _ => aggregate.hasCarriedInput
  | _ => false

def RuntimeExpr.hasCarriedInput : {type : RuntimeScalarType} → RuntimeExpr type → Bool
  | _, .intWire wire | _, .boolWire wire => wire.hasCarriedInput
  | _, .intBinary _ left right | _, .compare _ left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | _, .bitExtract value _ | _, .boolToInt value => value.hasCarriedInput
  | _, .thresholdDecodeBool matrix .. => matrix.hasCarriedInput
  | _, .familyElement _ aggregate _ index =>
      aggregate.hasCarriedInput || index.hasCarriedInput
  | _, .select _ index _ => index.hasCarriedInput
  | _, .carriedInput _ => true
  | _, _ => false

mutual
  def MatrixExpr.hasCarriedInput : MatrixExpr → Bool
    | .wire reference => reference.value.hasCarriedInput
    | .gadget _ _ => false
    | .add left right | .multiply left right =>
        left.hasCarriedInput || right.hasCarriedInput
    | .negate value | .scalarMultiply _ value | .rowSlice value _ _ |
        .columnSlice value _ _ | .rowCoefficientEmbed _ _ value |
        .columnBasisEmbed _ _ value | .diagonalCoefficientEmbed _ _ value |
        .diagonalBasisEmbed _ _ value => value.hasCarriedInput
    | .rowConcat parts | .columnConcat parts | .diagonalConcat parts =>
        matrixExprListHasCarriedInput parts
    | .select index branches =>
        index.hasCarriedInput || matrixExprListHasCarriedInput branches
    | .carriedInput _ _ => true
    | _ => false

  def matrixExprListHasCarriedInput : List MatrixExpr → Bool
    | [] => false
    | expression :: tail => expression.hasCarriedInput || matrixExprListHasCarriedInput tail
end

def BoundExpr.hasCarriedInput : BoundExpr → Bool
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | .floorDivide value _ => value.hasCarriedInput
  | .matrixProduct _ _ left right => left.hasCarriedInput || right.hasCarriedInput
  | .carriedInput _ => true
  | _ => false

def IntBoundExpr.hasCarriedInput : IntBoundExpr → Bool
  | .negate value => value.hasCarriedInput
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right =>
      left.hasCarriedInput || right.hasCarriedInput
  | .carriedInput _ => true
  | _ => false

private def MatrixRelation.hasCarriedInput : MatrixRelation → Bool
  | .preimage subject source target trapdoor =>
      subject.hasCarriedInput || source.value.hasCarriedInput ||
        target.value.hasCarriedInput || trapdoor.hasCarriedInput
  | .gadgetDecomposition subject target .. =>
      subject.hasCarriedInput || target.value.hasCarriedInput

def ValueFact.hasCarriedInput : ValueFact → Bool
  | .matrix fact =>
      let primary := match fact.primary with
        | .exact expression => expression.hasCarriedInput
        | .affine form => form.noiseBound.hasCarriedInput || form.terms.any fun term =>
            term.coefficient.expression.hasCarriedInput ||
              term.coefficient.normBound.hasCarriedInput || term.basis.hasCarriedInput
      fact.subject.hasCarriedInput || primary || fact.totalNormBound.hasCarriedInput ||
        fact.relations.any MatrixRelation.hasCarriedInput
  | .trapdoor fact => fact.privatePort.hasCarriedInput || fact.publicPort.hasCarriedInput ||
      fact.publicMatrix.hasCarriedInput
  | .integer fact => fact.expression.hasCarriedInput || fact.lower.hasCarriedInput ||
      fact.upper.hasCarriedInput
  | .boolean fact => fact.expression.hasCarriedInput
  | .bytes wire => wire.hasCarriedInput
  | .family fact => fact.aggregate.hasCarriedInput

inductive ResolveRecurrenceBoundsError where
  | countEvaluation (error : IntEvalError)
  | negativeCount (value : Int)
  | missingRecurrence (recurrence : FactRecurrenceInstanceRef)
  | missingFamily (family : FamilyAggregateRef)
  | familyNestingTooDeep
  | natural (slot : Nat) (error : BoundEvalError)
  | integer (slot : Nat) (error : IntBoundEvalError)
  | duplicateNaturalPath (path : BoundFactPath)
  | duplicateIntegerPath (path : IntBoundFactPath)
  deriving BEq, DecidableEq, Repr

private def insertNaturalBound
    (entry : CarriedNaturalBound)
    (table : CarriedBoundTable) : Except ResolveRecurrenceBoundsError CarriedBoundTable :=
  if table.natural.any (fun current => current.path.sameUniformLocation entry.path) then
    .error (.duplicateNaturalPath entry.path)
  else .ok { table with natural := table.natural ++ [entry] }

private def insertIntegerBound
    (entry : CarriedIntegerBound)
    (table : CarriedBoundTable) : Except ResolveRecurrenceBoundsError CarriedBoundTable :=
  if table.integer.any (fun current => current.path.sameUniformLocation entry.path) then
    .error (.duplicateIntegerPath entry.path)
  else .ok { table with integer := table.integer ++ [entry] }

private def BoundFactPath.withCarriedSlot (slot : Nat) : BoundFactPath → BoundFactPath
  | .affineCoefficientBound _ term => .affineCoefficientBound slot term
  | .affineNoiseBound _ => .affineNoiseBound slot
  | .matrixTotalBound _ => .matrixTotalBound slot
  | .familyElement _ index nested => .familyElement slot index (nested.withCarriedSlot slot)

private def IntBoundFactPath.withCarriedSlot (slot : Nat) : IntBoundFactPath → IntBoundFactPath
  | .lower _ => .lower slot
  | .upper _ => .upper slot
  | .familyElement _ index nested => .familyElement slot index (nested.withCarriedSlot slot)

private def uniformFamilyIndex : RuntimeExprRef .integer := ⟨0⟩

private def CarriedBoundTable.asFamilyElement
    (slot : Nat)
    (table : CarriedBoundTable) : CarriedBoundTable := {
  natural := table.natural.map fun entry => {
    entry with path := .familyElement slot uniformFamilyIndex entry.path
  }
  integer := table.integer.map fun entry => {
    entry with path := .familyElement slot uniformFamilyIndex entry.path
  }
}

private def CarriedBoundTable.copyFamilySlot
    (sourceSlot targetSlot : Nat)
    (table : CarriedBoundTable) : CarriedBoundTable := {
  natural := table.natural.filterMap fun entry => match entry.path with
    | .familyElement slot _ _ =>
        if slot = sourceSlot then some { entry with path := entry.path.withCarriedSlot targetSlot }
        else none
    | _ => none
  integer := table.integer.filterMap fun entry => match entry.path with
    | .familyElement slot _ _ =>
        if slot = sourceSlot then some { entry with path := entry.path.withCarriedSlot targetSlot }
        else none
    | _ => none
}

private def CheckedRecurrenceBoundTable.copyFamilySlot
    (recurrence : FactRecurrenceInstanceRef)
    (sourceSlot targetSlot : Nat)
    (table : CheckedRecurrenceBoundTable) : CarriedBoundTable := {
  natural := table.natural.filterMap fun entry => match entry.path with
    | .familyElement slot _ _ =>
        if entry.recurrence = recurrence && slot = sourceSlot then
          some { path := entry.path.withCarriedSlot targetSlot, value := entry.value }
        else none
    | _ => none
  integer := table.integer.filterMap fun entry => match entry.path with
    | .familyElement slot _ _ =>
        if entry.recurrence = recurrence && slot = sourceSlot then
          some { path := entry.path.withCarriedSlot targetSlot, value := entry.value }
        else none
    | _ => none
}

private def lookupUniqueFamily
    (analysis : AnalysisResult)
    (joint : JointFamilyId) : Option JointFamilyFact :=
  match analysis.families.filter (fun entry => entry.1 = joint) with
  | [entry] => some entry.2
  | _ => none

private def lookupUniqueRecurrence
    (analysis : AnalysisResult)
    (reference : FactRecurrenceInstanceRef) : Option FactRecurrence :=
  match analysis.recurrences.filter (fun entry => entry.1 = reference) with
  | [entry] => some entry.2
  | _ => none

private def resolveFamilyElementTemplate :
    Nat → AnalysisResult → FamilyAggregateRef → Option ValueFactTemplate
  | 0, _, _ => none
  | fuel + 1, analysis, aggregate => do
      match aggregate with
      | .joint joint outputSlot _ =>
          let family ← lookupUniqueFamily analysis joint
          family.elementTuple[outputSlot]?
      | .recurrenceResult recurrence path slot =>
          let recurrence ← lookupUniqueRecurrence analysis { recurrence, path }
          let template ← recurrence.bodyOutputs[slot]?
          match template.fact with
          | .family family => resolveFamilyElementTemplate fuel analysis family.aggregate
          | _ => none
      | .familyElement parent _ =>
          let template ← resolveFamilyElementTemplate fuel analysis parent
          match template.fact with
          | .family family => resolveFamilyElementTemplate fuel analysis family.aggregate
          | _ => none
      | .carriedInput _ => none

private def evaluateFactBoundsWithFuel
    (fuel : Nat)
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment)
    (resolved : CheckedRecurrenceBoundTable)
    (previous : CarriedBoundTable)
    (slot : Nat)
    (fact : ValueFact) : Except ResolveRecurrenceBoundsError CarriedBoundTable := do
  match fuel with
  | 0 => throw .familyNestingTooDeep
  | fuel + 1 => match fact with
  | .matrix matrix =>
      let total ← matrix.totalNormBound.evaluateTemplate environment resolved previous
        |>.mapError (.natural slot)
      let mut table ← insertNaturalBound { path := .matrixTotalBound slot, value := total } {}
      match matrix.primary with
      | .exact _ => return table
      | .affine form =>
          let noise ← form.noiseBound.evaluateTemplate environment resolved previous
            |>.mapError (.natural slot)
          table ← insertNaturalBound { path := .affineNoiseBound slot, value := noise } table
          for (term, termIndex) in form.terms.zipIdx do
            let value ← term.coefficient.normBound.evaluateTemplate environment resolved previous
              |>.mapError (.natural slot)
            table ← insertNaturalBound {
              path := .affineCoefficientBound slot termIndex
              value
            } table
          return table
  | .integer integer =>
      let lower ← integer.lower.evaluateTemplate environment resolved previous
        |>.mapError (.integer slot)
      let upper ← integer.upper.evaluateTemplate environment resolved previous
        |>.mapError (.integer slot)
      let table ← insertIntegerBound { path := .lower slot, value := lower } {}
      insertIntegerBound { path := .upper slot, value := upper } table
    | .family family =>
        match family.aggregate with
        | .carriedInput sourceSlot => return previous.copyFamilySlot sourceSlot slot
        | .recurrenceResult recurrence path sourceSlot =>
            return resolved.copyFamilySlot { recurrence, path } sourceSlot slot
        | aggregate =>
            let template ← match resolveFamilyElementTemplate fuel analysis aggregate with
              | some template => pure template
              | none => throw (.missingFamily aggregate)
            let element ← evaluateFactBoundsWithFuel fuel analysis environment resolved previous
              slot template.fact
            return element.asFamilyElement slot
    | .boolean _ => return {}
    | _ => return {}

private def evaluateFactBounds
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment)
    (resolved : CheckedRecurrenceBoundTable)
    (previous : CarriedBoundTable)
    (slot : Nat)
    (fact : ValueFact) : Except ResolveRecurrenceBoundsError CarriedBoundTable :=
  evaluateFactBoundsWithFuel 64 analysis environment resolved previous slot fact

private def evaluateFactVectorBounds
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment)
    (resolved : CheckedRecurrenceBoundTable)
    (previous : CarriedBoundTable)
    (facts : List ValueFact) : Except ResolveRecurrenceBoundsError CarriedBoundTable := do
  let mut result : CarriedBoundTable := {}
  for (fact, slot) in facts.zipIdx do
    let component ← evaluateFactBounds analysis environment resolved previous slot fact
    for entry in component.natural do
      result ← insertNaturalBound entry result
    for entry in component.integer do
      result ← insertIntegerBound entry result
  return result

private def iterateRecurrenceBounds
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment)
    (resolved : CheckedRecurrenceBoundTable)
    (body : List ValueFact) : Nat → CarriedBoundTable →
      Except ResolveRecurrenceBoundsError CarriedBoundTable
  | 0, current => .ok current
  | count + 1, current => do
      let next ← evaluateFactVectorBounds analysis environment resolved current body
      iterateRecurrenceBounds analysis environment resolved body count next

private def insertResolvedNatural
    (recurrence : FactRecurrenceInstanceRef)
    (entry : CarriedNaturalBound)
    (table : CheckedRecurrenceBoundTable) :
    Except ResolveRecurrenceBoundsError CheckedRecurrenceBoundTable :=
  if table.natural.any (fun current =>
      current.recurrence = recurrence && current.path.sameUniformLocation entry.path) then
    .error (.duplicateNaturalPath entry.path)
  else .ok { table with natural := table.natural ++ [{
    recurrence
    path := entry.path
    value := entry.value
  }] }

private def insertResolvedInteger
    (recurrence : FactRecurrenceInstanceRef)
    (entry : CarriedIntegerBound)
    (table : CheckedRecurrenceBoundTable) :
    Except ResolveRecurrenceBoundsError CheckedRecurrenceBoundTable :=
  if table.integer.any (fun current =>
      current.recurrence = recurrence && current.path.sameUniformLocation entry.path) then
    .error (.duplicateIntegerPath entry.path)
  else .ok { table with integer := table.integer ++ [{
    recurrence
    path := entry.path
    value := entry.value
  }] }

/-- Evaluate one analyzer-owned recurrence numerically. Every body component reads the same
immutable previous table and the complete next table is committed only after all components
succeed. The returned keys are instance-qualified and can be consumed by bound denotation. -/
def resolveRecurrenceBounds
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment)
    (resolved : CheckedRecurrenceBoundTable)
    (recurrenceInstance : FactRecurrenceInstanceRef) :
    Except ResolveRecurrenceBoundsError CheckedRecurrenceBoundTable := do
  let recurrence ← match lookupUniqueRecurrence analysis recurrenceInstance with
    | some recurrence => pure recurrence
    | none => throw (.missingRecurrence recurrenceInstance)
  let countValue ← evaluateIntExpr environment recurrence.count |>.mapError .countEvaluation
  if countValue < 0 then throw (.negativeCount countValue)
  let initial ← evaluateFactVectorBounds analysis environment resolved {} recurrence.initial.toList
  let final ← iterateRecurrenceBounds analysis environment resolved
    (recurrence.bodyOutputs.toList.map (·.fact)) countValue.toNat initial
  let mut result := resolved
  for entry in final.natural do
    result ← insertResolvedNatural recurrenceInstance entry result
  for entry in final.integer do
    result ← insertResolvedInteger recurrenceInstance entry result
  return result

private def zeroIterationFixtureRecurrence : FactRecurrence := {
    loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ } }
    count := .constant 0
    carriedArity := 1
    initial := ⟨#[.integer {
      expression := .intConstant 3
      lower := .integer (.constant 3)
      upper := .integer (.constant 3)
    }], rfl⟩
    bodyInputs := ⟨#[{
      definition := { stage := ⟨"fixture"⟩, name := "body" }
      bodyScope := ⟨[]⟩
      node := ⟨0⟩
      port := 0
    }], rfl⟩
    bodyOutputs := ⟨#[{
      fact := .integer {
        expression := .carriedInput (.integerValue 0)
        lower := .carriedInput (.lower 0)
        upper := .carriedInput (.upper 0)
      }
      schema := .integer
    }], rfl⟩
    invariantInputs := []
    iterationVariable := ⟨0⟩
  }

private def zeroIterationFixtureAnalysis : AnalysisResult where
  facts := []
  families := []
  recurrences := [({ recurrence := ⟨"zero"⟩, path := [] }, zeroIterationFixtureRecurrence)]
  staticObligations := []
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-- Zero iterations return the evaluated initial snapshot, including all carried components. -/
example : resolveRecurrenceBounds zeroIterationFixtureAnalysis [] {}
    { recurrence := ⟨"zero"⟩, path := [] } = .ok {
    natural := []
    integer := [
      { recurrence := { recurrence := ⟨"zero"⟩, path := [] }, path := .lower 0, value := 3 },
      { recurrence := { recurrence := ⟨"zero"⟩, path := [] }, path := .upper 0, value := 3 }
    ]
  } := rfl

private def diamondFamilyFixtureId : JointFamilyId := ⟨"diamond-indexed-matrix"⟩

private def diamondFamilyFixtureElement : ValueFactTemplate := {
  fact := .matrix {
    subject := .protocolInput ⟨"diamond-family-element"⟩
    primary := .exact (.zero diamondCarriedFixtureType)
    relations := []
    totalNormBound := .constant 9
  }
  schema := .matrix diamondCarriedFixtureType .exact [] .unknown
}

private def diamondFamilyFixture : JointFamilyFact := {
  id := diamondFamilyFixtureId
  count := .constant 8
  indexVariable := ⟨0⟩
  outputFamilies := [{
    stage := ⟨"fixture"⟩
    scope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }]
  outputArity := 1
  elementTuple := ⟨#[diamondFamilyFixtureElement], rfl⟩
}

private def diamondFamilyFixtureRecurrence : FactRecurrence := {
  loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨1⟩ } }
  count := .constant 2
  carriedArity := 1
  initial := ⟨#[.family {
    aggregate := .joint diamondFamilyFixtureId 0 []
    count := .constant 8
    elementSchema := .matrix diamondCarriedFixtureType .exact [] .unknown
  }], rfl⟩
  bodyInputs := ⟨#[{
    definition := { stage := ⟨"fixture"⟩, name := "body" }
    bodyScope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }], rfl⟩
  bodyOutputs := ⟨#[{
    fact := .family {
      aggregate := .carriedInput 0
      count := .constant 8
      elementSchema := .matrix diamondCarriedFixtureType .exact [] .unknown
    }
    schema := .family (.constant 8)
      (.matrix diamondCarriedFixtureType .exact [] .unknown)
  }], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩
}

private def diamondFamilyFixtureAnalysis : AnalysisResult where
  facts := []
  families := [(diamondFamilyFixtureId, diamondFamilyFixture)]
  recurrences := [({ recurrence := ⟨"diamond-family"⟩, path := [] },
    diamondFamilyFixtureRecurrence)]
  staticObligations := []
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-- Diamond's indexed matrix carried shape is resolved once, without enumerating its lanes. -/
example : resolveRecurrenceBounds diamondFamilyFixtureAnalysis [] {}
    { recurrence := ⟨"diamond-family"⟩, path := [] } = .ok {
  natural := [{
    recurrence := { recurrence := ⟨"diamond-family"⟩, path := [] }
    path := .familyElement 0 uniformFamilyIndex (.matrixTotalBound 0)
    value := 9
  }]
  integer := []
} := rfl

/-- A later family access may use a different arena index while consuming the uniform bound. -/
example : BoundExpr.evaluateWithRecurrences [] {
    natural := [{
      recurrence := { recurrence := ⟨"diamond-family"⟩, path := [] }
      path := .familyElement 0 uniformFamilyIndex (.matrixTotalBound 0)
      value := 9
    }]
  } (.recurrenceResult { recurrence := ⟨"diamond-family"⟩, path := [] }
    (.familyElement 0 ⟨17⟩ (.matrixTotalBound 0))) = .ok 9 := rfl

/-! ## GGH15 preimage-chain hard-bound recurrence

These definitions are the closed hard-bound rule for the recurrence
`C_(i+1) = (s_i S_i) B_(i+1) + (s_i E_i + e_i K_i)`.  They evaluate only
the three numeric components and never expand the symbolic coefficient product.
-/

structure Ggh15UniformBounds where
  ciphertextModulus : Nat
  ringDimension : Nat
  secretRows : Nat
  publicColumns : Nat
  secretStep : Nat
  relationError : Nat
  preimage : Nat
  publicMatrix : Nat
  deriving BEq, DecidableEq, Repr

structure Ggh15RecurrenceBounds where
  coefficient : Nat
  noise : Nat
  total : Nat
  deriving BEq, DecidableEq, Repr

private def hardMatrixProduct (ringDimension innerDimension left right : Nat) : Nat :=
  ringDimension * innerDimension * left * right

private def centeredCap (ciphertextModulus value : Nat) : Nat :=
  min (ciphertextModulus / 2) value

def Ggh15RecurrenceBounds.step
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) : Ggh15RecurrenceBounds :=
  let coefficient := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      current.coefficient uniform.secretStep)
  let noise := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      current.coefficient uniform.relationError +
    hardMatrixProduct uniform.ringDimension uniform.publicColumns
      current.noise uniform.preimage)
  let total := centeredCap uniform.ciphertextModulus
    (hardMatrixProduct uniform.ringDimension uniform.secretRows
      coefficient uniform.publicMatrix + noise)
  { coefficient, noise, total }

@[simp] theorem Ggh15RecurrenceBounds.step_coefficient
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).coefficient = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        current.coefficient uniform.secretStep) := rfl

@[simp] theorem Ggh15RecurrenceBounds.step_noise
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).noise = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        current.coefficient uniform.relationError +
      hardMatrixProduct uniform.ringDimension uniform.publicColumns
        current.noise uniform.preimage) := rfl

@[simp] theorem Ggh15RecurrenceBounds.step_total
    (uniform : Ggh15UniformBounds)
    (current : Ggh15RecurrenceBounds) :
    (current.step uniform).total = centeredCap uniform.ciphertextModulus
      (hardMatrixProduct uniform.ringDimension uniform.secretRows
        (current.step uniform).coefficient uniform.publicMatrix +
      (current.step uniform).noise) := rfl

/-- Ordered numeric fold for the recurrence components.  The symbolic matrix expression remains
represented by `MatrixExpr.loopResult`; this function does not expand it. -/
def Ggh15RecurrenceBounds.after
    (uniform : Ggh15UniformBounds) : Nat → Ggh15RecurrenceBounds → Ggh15RecurrenceBounds
  | 0, initial => initial
  | count + 1, initial => Ggh15RecurrenceBounds.after uniform count (initial.step uniform)

@[simp] theorem Ggh15RecurrenceBounds.after_zero
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds) :
    initial.after uniform 0 = initial := rfl

@[simp] theorem Ggh15RecurrenceBounds.after_succ
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds)
    (count : Nat) :
    initial.after uniform (count + 1) = (initial.step uniform).after uniform count := rfl

theorem Ggh15RecurrenceBounds.after_add
    (uniform : Ggh15UniformBounds)
    (initial : Ggh15RecurrenceBounds)
    (left right : Nat) :
    initial.after uniform (left + right) = (initial.after uniform left).after uniform right := by
  induction left generalizing initial with
  | zero => simp
  | succ left induction =>
      simp only [Nat.succ_add, after_succ]
      exact induction (initial.step uniform)

def Ggh15RecurrenceBounds.resolvePath
    (bounds : Ggh15RecurrenceBounds)
    (path : BoundFactPath) : Option Nat :=
  match path with
  | .affineCoefficientBound 0 0 => some bounds.coefficient
  | .affineNoiseBound 0 => some bounds.noise
  | .matrixTotalBound 0 => some bounds.total
  | _ => none

example :
    ({ coefficient := 2, noise := 3, total := 0 } : Ggh15RecurrenceBounds).step {
      ciphertextModulus := 100000
      ringDimension := 4
      secretRows := 2
      publicColumns := 3
      secretStep := 5
      relationError := 7
      preimage := 11
      publicMatrix := 13
    } = { coefficient := 80, noise := 508, total := 8828 } := by
  decide

end Mxx.Certificate
