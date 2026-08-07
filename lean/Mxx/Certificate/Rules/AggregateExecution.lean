import Mxx.Certificate.Rules.LoopRecurrence
import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-! # Execution-connected aggregate evidence

These structures retain members of the executable support and the traces obtained by inverting
those members. They do not accept a loop invariant, an asserted output fact, or a replacement
body semantics.
-/

/-- Exact execution evidence for one analyzed parallel-loop family. -/
structure ParallelLoopSemanticResult
    (analysis : AnalysisResult)
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (joint : JointFamilyId) where
  family : JointFamilyFact
  uniqueResolution : analysis.families.filter (fun entry => entry.1 = joint) = [(joint, family)]
  definition : String
  countExpression : IntExpr
  indexSlot : Nat
  bindings : List (String × IntExpr)
  modes : List Mxx.Ir.LoopInputMode
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  countMatches : family.count = countExpression
  arityMatches : family.outputArity = outputCount
  iterationSlotMatches : family.indexVariable.slot = indexSlot
  before : Mxx.Ir.WireEnvironment
  argumentValues : List Mxx.Ir.Value
  evaluatedCount : Int
  countNonnegative : 0 ≤ evaluatedCount
  argumentsEvaluate :
    argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) = some argumentValues
  countEvaluate : countExpression.evaluate params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before {
    kind := .parallelLoop definition countExpression indexSlot bindings modes
    arguments := argumentRefs
    outputCount
    outputTypes
  }
  final : List (List Mxx.Ir.Value)
  executionTrace : Mxx.Ir.ParallelIterationsTrace runChild definition params indexSlot bindings
    modes argumentValues (List.range evaluatedCount.toNat)
    (List.replicate outputCount []) final
  finalEq : nodeValues = final.map Mxx.Ir.Value.family

/-- Invert an actual parallel-loop result once. No child execution or family value is supplied
independently of the executable node member. -/
theorem ParallelLoopSemanticResult.nonempty_ofNodeMember
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {joint : JointFamilyId}
    (family : JointFamilyFact)
    (uniqueResolution : analysis.families.filter (fun entry => entry.1 = joint) =
      [(joint, family)])
    (definition : String)
    (countExpression : IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × IntExpr))
    (modes : List Mxx.Ir.LoopInputMode)
    (argumentRefs : List Mxx.Ir.WireRef)
    (outputCount : Nat)
    (outputTypes : List Mxx.Ir.WireTypeExpr)
    (before : Mxx.Ir.WireEnvironment)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (countNonnegative : 0 ≤ evaluatedCount)
    (argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) =
      some argumentValues)
    (countEvaluate : countExpression.evaluate params = some evaluatedCount)
    (countMatches : family.count = countExpression)
    (arityMatches : family.outputArity = outputCount)
    (iterationSlotMatches : family.indexVariable.slot = indexSlot)
    (nodeValues : List Mxx.Ir.Value)
    (nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs before {
      kind := .parallelLoop definition countExpression indexSlot bindings modes
      arguments := argumentRefs
      outputCount
      outputTypes
    }) : Nonempty (ParallelLoopSemanticResult analysis runChild samplers params inputs joint) := by
  have traced :=
    (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace runChild samplers params inputs before
      definition countExpression indexSlot bindings modes argumentRefs outputCount argumentValues
      evaluatedCount argumentsEvaluate countEvaluate nodeValues).mp nodeMember
  obtain ⟨final, executionTrace, finalEq⟩ := traced
  exact ⟨{
    family
    uniqueResolution
    definition
    countExpression
    indexSlot
    bindings
    modes
    argumentRefs
    outputCount
    outputTypes
    countMatches
    arityMatches
    iterationSlotMatches
    before
    argumentValues
    evaluatedCount
    countNonnegative
    argumentsEvaluate
    countEvaluate
    nodeValues
    nodeMember
    final
    executionTrace
    finalEq
  }⟩

def ParallelLoopSemanticResult.portValues
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {joint : JointFamilyId}
    (result : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    (port : Nat) : Option (List Mxx.Ir.Value) :=
  result.final[port]?

theorem ParallelLoopSemanticResult.nodeValueAt
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {joint : JointFamilyId}
    (result : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    {port : Nat}
    {values : List Mxx.Ir.Value}
    (lookup : result.portValues port = some values) :
    result.nodeValues[port]? = some (.family values) := by
  rw [result.finalEq]
  rw [List.getElem?_map]
  simpa [ParallelLoopSemanticResult.portValues, lookup]

/-- Bind one selected executable parallel-loop port to its concrete output wire. -/
def ParallelLoopSemanticResult.bindPort
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {joint : JointFamilyId}
    (result : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (port : Nat) : Option FactEnvironment := do
  let values ← result.portValues port
  return environment.bind (.ofCoreWire wire) (.family values)

theorem ParallelLoopSemanticResult.bindPort_familyHolds
    {analysis : AnalysisResult}
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {joint : JointFamilyId}
    (result : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (port : Nat)
    (familyFact : FamilyFact)
    {boundEnvironment : FactEnvironment}
    (bound : result.bindPort environment wire port = some boundEnvironment) :
    ScopedWireFact.Holds boundEnvironment {
      wire
      matrixType := none
      fact := .family familyFact
    } := by
  cases lookup : result.portValues port with
  | none => simp [ParallelLoopSemanticResult.bindPort, lookup] at bound
  | some values =>
    simp [ParallelLoopSemanticResult.bindPort, lookup] at bound
    subst boundEnvironment
    exact ⟨values, FactEnvironment.bind_same environment (.ofCoreWire wire) (.family values)⟩

theorem familyGetStatic_selectedValue
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (familyRef : Mxx.Ir.WireRef)
    (family : List Mxx.Ir.Value)
    (index : IntExpr)
    (evaluatedIndex : Int)
    (outputCount : Nat)
    (value : Mxx.Ir.Value)
    (argumentsEvaluate : [familyRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
      some [.family family])
    (indexEvaluate : index.evaluate params = some evaluatedIndex)
    (selected : family[evaluatedIndex.toNat]? = some value)
    {nodeValues : List Mxx.Ir.Value}
    (member : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .familyGetStatic index
      arguments := [familyRef]
      outputCount
    }) : nodeValues = [value] := by
  rw [Mxx.Ir.mem_evaluateNode_familyGetStatic_of_arguments runChild samplers params inputs wires
    familyRef family index evaluatedIndex outputCount argumentsEvaluate indexEvaluate member]
  simp [selected]

theorem familyGetDynamic_selectedValue
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (familyRef indexRef : Mxx.Ir.WireRef)
    (family : List Mxx.Ir.Value)
    (index : Int)
    (outputCount : Nat)
    (value : Mxx.Ir.Value)
    (argumentsEvaluate : [familyRef, indexRef].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire wires) = some [.family family, .integer index])
    (selected : family[index.toNat]? = some value)
    {nodeValues : List Mxx.Ir.Value}
    (member : nodeValues ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .familyGetDynamic
      arguments := [familyRef, indexRef]
      outputCount
    }) : nodeValues = [value] := by
  rw [Mxx.Ir.mem_evaluateNode_familyGetDynamic_of_arguments runChild samplers params inputs wires
    familyRef indexRef family index outputCount argumentsEvaluate member]
  simp [selected]

/-- Bind the executable family-get output and the analyzer-owned element identity to the same
selected runtime value. -/
def FactEnvironment.bindFamilyElement
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (index : RuntimeExprRef .integer)
    (value : Mxx.Ir.Value) : FactEnvironment :=
  (environment.bind (.familyElement aggregate index) value).bind (.ofCoreWire wire) value

theorem FactEnvironment.bindFamilyElement_lookups
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (index : RuntimeExprRef .integer)
    (value : Mxx.Ir.Value)
    (different : ValueInstanceRef.ofCoreWire wire ≠ .familyElement aggregate index) :
    let bound := environment.bindFamilyElement wire aggregate index value
    bound.values (.familyElement aggregate index) = some value ∧
      bound.values (.ofCoreWire wire) = some value := by
  dsimp [FactEnvironment.bindFamilyElement]
  constructor
  · rw [FactEnvironment.bind_other]
    · exact FactEnvironment.bind_same _ _ _
    · exact Ne.symm different
  · exact FactEnvironment.bind_same _ _ _

theorem familyGet_integerHolds
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (indexRef : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (indexValue value lowerValue upperValue : Int)
    (lower upper : IntBoundExpr)
    (different : ValueInstanceRef.ofCoreWire wire ≠ .familyElement aggregate indexRef)
    (arenaLookup : environment.expressionArena.lookupInteger indexRef = some indexExpression)
    (indexDenotes : RuntimeIntExpr.Denotes
      (environment.bindFamilyElement wire aggregate indexRef (.integer value))
      indexExpression indexValue)
    (lowerEvaluates : lower.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates =
      .ok lowerValue)
    (upperEvaluates : upper.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates =
      .ok upperValue)
    (lowerValid : lowerValue ≤ value)
    (upperValid : value ≤ upperValue) :
    ScopedWireFact.Holds
      (environment.bindFamilyElement wire aggregate indexRef (.integer value)) {
        wire
        matrixType := none
        fact := .integer {
          expression := .familyElement .integer aggregate indexRef indexExpression
          lower
          upper
        }
      } := by
  let bound := environment.bindFamilyElement wire aggregate indexRef (.integer value)
  have lookups := environment.bindFamilyElement_lookups wire aggregate indexRef (.integer value)
    different
  exact ⟨value, lowerValue, upperValue, lookups.2,
    .familyElement arenaLookup indexDenotes lookups.1, lowerEvaluates, upperEvaluates,
    lowerValid, upperValid⟩

theorem familyGet_booleanHolds
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (indexRef : RuntimeExprRef .integer)
    (indexExpression : RuntimeExpr .integer)
    (indexValue : Int)
    (value : Bool)
    (different : ValueInstanceRef.ofCoreWire wire ≠ .familyElement aggregate indexRef)
    (arenaLookup : environment.expressionArena.lookupInteger indexRef = some indexExpression)
    (indexDenotes : RuntimeIntExpr.Denotes
      (environment.bindFamilyElement wire aggregate indexRef (.boolean value))
      indexExpression indexValue) :
    ScopedWireFact.Holds
      (environment.bindFamilyElement wire aggregate indexRef (.boolean value)) {
        wire
        matrixType := none
        fact := .boolean {
          expression := .familyElement .boolean aggregate indexRef indexExpression
        }
      } := by
  have lookups := environment.bindFamilyElement_lookups wire aggregate indexRef (.boolean value)
    different
  exact ⟨value, lookups.2, .familyElement arenaLookup indexDenotes lookups.1⟩

theorem familyGet_exactMatrixHolds
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (indexRef : RuntimeExprRef .integer)
    (type : MatrixTypeExpr)
    (value : Mxx.Matrix)
    (bound : BoundExpr)
    (boundValue : Nat)
    (different : ValueInstanceRef.ofCoreWire wire ≠ .familyElement aggregate indexRef)
    (boundEvaluates : bound.evaluate environment.parameters = .ok boundValue)
    (normBound : Mxx.maxCenteredCoefficientNorm value ≤ boundValue) :
    ScopedWireFact.Holds
      (environment.bindFamilyElement wire aggregate indexRef (.matrix value)) {
        wire
        matrixType := some type
        fact := .matrix {
          subject := .ofCoreWire wire
          primary := .exact (.wire { value := .familyElement aggregate indexRef, type })
          relations := []
          totalNormBound := bound
        }
      } := by
  have lookups := environment.bindFamilyElement_lookups wire aggregate indexRef (.matrix value)
    different
  exact exactMatrixFact_holds _ _ _ value bound boundValue lookups.2 (.wire lookups.1)
    boundEvaluates normBound

theorem familyGet_familyHolds
    (environment : FactEnvironment)
    (wire : CoreWireRef)
    (aggregate : FamilyAggregateRef)
    (indexRef : RuntimeExprRef .integer)
    (values : List Mxx.Ir.Value)
    (family : FamilyFact)
    (different : ValueInstanceRef.ofCoreWire wire ≠ .familyElement aggregate indexRef) :
    ScopedWireFact.Holds
      (environment.bindFamilyElement wire aggregate indexRef (.family values)) {
        wire
        matrixType := none
        fact := .family family
      } := by
  have lookups := environment.bindFamilyElement_lookups wire aggregate indexRef (.family values)
    different
  exact ⟨values, lookups.2⟩

end Mxx.Certificate
