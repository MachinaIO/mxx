import MxxIrCore.ScopeInvariant

namespace Mxx.We.DiamondWE

open Mxx.IR

/- The injector executes its transition update inside one sequential-loop iteration and one lane
   of a terminal parallel grid.  This theorem composes those two evaluator inversions.  Every
   scope factor below comes from `evalScope`; callers provide only stored graph facts and the
   generated step callbacks for the concrete scopes. -/
theorem reachedPrimitiveRunInsideLoopTerminalGrid {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (finalTrace evalTrace : Trace backend)
    (stageNumber : Nat) (stage : Stage) (stageStored : data.stages[stageNumber]? = some stage)
    (stageTrace : StageTrace backend)
    (stageFind : finalTrace.stages.find? (fun item => item.stage = stageNumber) = some stageTrace)
    (rootNumber : ScopeId) (root : Scope) (rootStored : scopeAt stage rootNumber = some root)
    (rootInputs : Array (Binding backend)) (rootPath : OccurrencePath) (rootFuel : Nat)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env {} evalTrace stageNumber stage rootNumber root stageStored
      rootStored rootInputs rootPath 0 #[] rootFuel = .ok rootResult)
    (stageRootFactor : stageTrace.scopes = rootResult.scopes)
    (loopIndex : Nat) (loopIndexBound : loopIndex + 1 ≤ root.nodes.size)
    (rootPrefix : ∀ (limit index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (result : ScopeResult backend), limit = loopIndex → index < limit → fuel ≠ 0 →
      evalScope data env {} evalTrace stageNumber stage rootNumber root stageStored rootStored
        rootInputs rootPath index values fuel = .ok result →
      ScopeFreeStep data env {} evalTrace stageNumber stage rootNumber root stageStored rootStored
        rootInputs rootPath index values fuel result)
    (loopNode : Node) (loopNodeStored : root.nodes[loopIndex]? = some loopNode)
    (loop : LoopPayload) (loopPayloadStored : loopNode.payload = .sequentialLoop loop)
    (iteration count : Nat)
    (rootSuffixAvoids : ∀ (selectedPath : OccurrencePath),
      OccurrencePath.Under
        (rootPath.push ⟨stageNumber, loop.child, loopIndex, iteration⟩) selectedPath →
      ∀ (values : Array (Binding backend)) (fuel : Nat) (result : ScopeResult backend),
      evalScope data env {} evalTrace stageNumber stage rootNumber root stageStored rootStored
        rootInputs rootPath (loopIndex + 1) values fuel = .ok result →
      ∀ snapshot ∈ result.scopes, snapshot.occurrence ≠ selectedPath)
    (body : Scope) (bodyStored : scopeAt stage loop.child = some body)
    (bodyId : body.id = loop.child)
    (countStored : evalNatExpr {} stageNumber body.id loopIndex loop.count = .ok count)
    (iterationBound : iteration < count)
    (gridIndex : Nat) (gridTerminal : gridIndex + 1 = body.nodes.size)
    (bodyPrefix : ∀ (inputs : Array (Binding backend)) (limit index : Nat)
      (values : Array (Binding backend)) (fuel : Nat) (result : ScopeResult backend),
      limit = gridIndex → index < limit → fuel ≠ 0 →
      evalScope data env
        { ({} : StructuralEnv) with slots := #[ (loop.indexSlot, Int.ofNat iteration) ] }
        evalTrace stageNumber stage loop.child body stageStored bodyStored
        inputs
        (rootPath.push ⟨stageNumber, body.id, loopIndex, iteration⟩)
        index values fuel = .ok result →
      ScopeFreeStep data env
        { ({} : StructuralEnv) with slots := #[ (loop.indexSlot, Int.ofNat iteration) ] }
        evalTrace stageNumber stage loop.child body stageStored bodyStored
        inputs
        (rootPath.push ⟨stageNumber, body.id, loopIndex, iteration⟩)
        index values fuel result)
    (gridNode : Node) (gridNodeStored : body.nodes[gridIndex]? = some gridNode)
    (grid : GridPayload) (gridPayloadStored : gridNode.payload = .parallelGrid grid)
    (concreteShape : Array Nat)
    (shapeStored : evalShape
      { ({} : StructuralEnv) with slots := #[ (loop.indexSlot, Int.ofNat iteration) ] }
      stageNumber body.id gridIndex grid.shape = .ok concreteShape)
    (lane : Nat) (laneBound : lane < shapeProductArray concreteShape)
    (child : Scope) (childStored : scopeAt stage grid.child = some child)
    (target : Nat) (targetBound : target + 1 ≤ child.nodes.size)
    (childPrefix : ∀ (structural : StructuralEnv) (inputs : Array (Binding backend))
      (path : OccurrencePath) (limit index : Nat) (values : Array (Binding backend))
      (fuel : Nat) (result : ScopeResult backend),
      limit = target → index < limit → fuel ≠ 0 →
      evalScope data env structural evalTrace stageNumber stage grid.child child stageStored
        childStored inputs path index values fuel = .ok result →
      FlatScopeStep data env structural evalTrace stageNumber stage grid.child child stageStored
        childStored inputs path index values fuel result)
    (targetNode : Node) (targetNodeStored : child.nodes[target]? = some targetNode)
    (payload : NodePayload) (payloadStored : targetNode.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload) (port : Nat)
    (portBound : port < targetNode.outputs.size) :
    Nonempty (ReachedPrimitiveRun finalTrace
      { ({} : StructuralEnv) with
        axes := ((coordinatesFromOffset concreteShape.toList lane).map Int.ofNat).toArray
        slots := grid.indexSlots.zip
          (coordinatesFromOffset concreteShape.toList lane).toArray |>.map
            (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
      stageNumber child.id target
      ((rootPath.push ⟨stageNumber, body.id, loopIndex, iteration⟩).push
        ⟨stageNumber, body.id, gridIndex, lane⟩)
      payload targetNode port) := by
  let loopPath := rootPath.push ⟨stageNumber, body.id, loopIndex, iteration⟩
  let gridPath := loopPath.push ⟨stageNumber, body.id, gridIndex, lane⟩
  let bodyStructural : StructuralEnv :=
    { ({} : StructuralEnv) with slots := #[ (loop.indexSlot, Int.ofNat iteration) ] }
  let laneStructural : StructuralEnv :=
    { bodyStructural with
      axes := ((coordinatesFromOffset concreteShape.toList lane).map Int.ofNat).toArray
      slots := grid.indexSlots.zip
        (coordinatesFromOffset concreteShape.toList lane).toArray |>.map
          (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
  obtain ⟨loopValues, loopFuel, rootCurrent, rootPrefixTrailing, loopCurrent,
      loopFuelEq, rootFactor, rootPrefixPaths⟩ :=
    generatedScopeFreePrefixFactor data env {} evalTrace stageNumber stage rootNumber root
      stageStored rootStored rootInputs rootPath #[] rootFuel loopIndex (by omega) rootPrefix
      rootResult rootSuccess
  have loopFuelPositive : loopFuel ≠ 0 := by
    intro fuelZero
    rw [evalScope] at loopCurrent
    simp [fuelZero] at loopCurrent
  obtain ⟨arguments, actualBody, actualBodyStored, loopResult, rootNext, argumentsStored,
      loopStored, typesMatch, rootNextStored, rootCurrentStored⟩ :=
    generatedSequentialLoopNodeEquation data env {} evalTrace stageNumber stage rootNumber root
      stageStored rootStored rootInputs rootPath loopIndex loopValues loopFuel rootCurrent
      loopFuelPositive (by omega) loopNode loopNodeStored loop loopPayloadStored loopCurrent
  have bodyEq : actualBody = body := by
    rw [bodyStored] at actualBodyStored
    exact Option.some.inj actualBodyStored.symm
  subst actualBody
  have gridUnderLoop : OccurrencePath.Under loopPath gridPath := by
    exact OccurrencePath.under_push loopPath _
  obtain ⟨_, reachedFuel, _, bodyInputs, bodyResult, _, bodyInputsStored, bodyEvaluated,
      loopLeading, loopTrailing, loopFactor, loopTrailingMiss⟩ :=
    evalSequentialLoop_success_child_at_with_trailing_miss data env evalTrace stageNumber stage
      loop.child body stageStored bodyStored loop loopIndex arguments {} rootPath gridPath 0
      (loopFuel - 1) count iteration countStored (by omega)
      (by simpa [loopPath] using gridUnderLoop) loopResult loopStored
  obtain ⟨gridValues, gridFuel, gridCurrent, bodyPrefixTrailing, gridCurrentStored,
      gridFuelEq, bodyFactor, bodyPrefixPaths⟩ :=
    generatedScopeFreePrefixFactor data env bodyStructural evalTrace stageNumber stage loop.child
      body stageStored bodyStored bodyInputs loopPath #[] (reachedFuel - 1) gridIndex
      (by omega) (by simpa [bodyStructural, loopPath] using bodyPrefix bodyInputs) bodyResult
      (by simpa [bodyStructural, loopPath] using bodyEvaluated)
  have gridFuelPositive : gridFuel ≠ 0 := by
    intro fuelZero
    rw [evalScope] at gridCurrentStored
    simp [fuelZero] at gridCurrentStored
  obtain ⟨gridArguments, actualChild, actualChildStored, laneResults, packed, gridNext,
      laneArguments, childInputs, childResult, outputs, gridArgumentsStored, laneArgumentsStored,
      childInputsStored, childEvaluated, selectedStored, gridNextStored, gridCurrentFactor,
      laneLeading, laneTrailing, laneFactor, laneTrailingMiss⟩ :=
    evalScope_success_parallelGrid_selected_lane data env evalTrace stageNumber stage stageStored
      bodyStructural loop.child body bodyStored bodyInputs loopPath gridIndex gridValues gridFuel
      gridCurrent gridFuelPositive (by omega) gridNode gridNodeStored grid gridPayloadStored
      concreteShape (by simpa [bodyStructural] using shapeStored) lane laneBound gridPath
      (OccurrencePath.under_refl gridPath) gridCurrentStored
  have childEq : actualChild = child := by
    rw [childStored] at actualChildStored
    exact Option.some.inj actualChildStored.symm
  subst actualChild
  let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
  let parentSnapshot : ScopeTrace backend := {
    scope := body.id
    occurrence := loopPath
    values := appendNodeBindings body.id gridIndex gridValues packed }
  let endSnapshot : ScopeTrace backend := {
    scope := body.id
    occurrence := loopPath
    values := appendNodeBindings body.id gridIndex gridValues packed }
  have gridNextAtEnd : gridNext = {
      values := appendNodeBindings body.id gridIndex gridValues packed
      scopes := #[endSnapshot] } := by
    apply evalScope_success_at_end data env bodyStructural evalTrace stageNumber stage loop.child
      body stageStored bodyStored bodyInputs loopPath (gridIndex + 1)
      (appendNodeBindings body.id gridIndex gridValues packed) (gridFuel - 1) gridNext
    · omega
    · exact gridNextStored
  have bodyCurrentFactor : gridCurrent.scopes =
      laneScopes ++ gridNext.scopes ++ #[parentSnapshot] := by
    have factor := congrArg ScopeResult.scopes gridCurrentFactor
    simpa [laneScopes, parentSnapshot] using factor
  have gridNextFactor : gridNext.scopes = #[endSnapshot] := by
    rw [gridNextAtEnd]
  have parentDifferent : loopPath ≠ gridPath := by
    exact OccurrencePath.ne_parent_of_push_under loopPath gridPath _
      (OccurrencePath.under_refl gridPath)
  obtain ⟨bodyTrailing, bodyGridFactor, bodyTrailingMiss⟩ :=
    terminalGrid_selected_child_factor bodyResult gridCurrent gridNext laneScopes
      childResult.scopes laneLeading laneTrailing bodyPrefixTrailing parentSnapshot endSnapshot
      loopPath gridPath bodyFactor bodyCurrentFactor gridNextFactor laneFactor laneTrailingMiss
      bodyPrefixPaths rfl rfl parentDifferent
  have rootPathDifferent : rootPath ≠ gridPath := by
    exact OccurrencePath.ne_parent_of_push_under rootPath gridPath
      ⟨stageNumber, body.id, loopIndex, iteration⟩
      (by simpa [loopPath, bodyId] using gridUnderLoop)
  have rootNextMiss : ∀ snapshot ∈ rootNext.scopes, snapshot.occurrence ≠ gridPath := by
    simpa [loopPath, gridPath, bodyId] using rootSuffixAvoids gridPath
      (by simpa [loopPath, bodyId] using gridUnderLoop)
      (appendNodeBindings root.id loopIndex loopValues loopResult.values) (loopFuel - 1)
      rootNext rootNextStored
  let rootSnapshot : ScopeTrace backend := {
    scope := root.id
    occurrence := rootPath
    values := appendNodeBindings root.id loopIndex loopValues loopResult.values }
  let outerLeading := loopLeading ++ laneLeading
  let trailing := bodyTrailing ++ loopTrailing ++ rootNext.scopes ++ #[rootSnapshot] ++
    rootPrefixTrailing
  have globalFactor : stageTrace.scopes = outerLeading ++ childResult.scopes ++ trailing := by
    rw [stageRootFactor, rootFactor]
    have rootCurrentScopes := congrArg ScopeResult.scopes rootCurrentStored
    rw [rootCurrentScopes, loopFactor, bodyGridFactor]
    simp [outerLeading, trailing, rootSnapshot, Array.append_assoc]
  have globalTrailingMiss :
      ∀ snapshot ∈ trailing, snapshot.occurrence ≠ gridPath := by
    intro snapshot membership
    simp only [trailing, Array.mem_append, Array.mem_singleton] at membership
    rcases membership with ((((bodyMember | loopMember) | nextMember) | rfl) | prefixMember)
    · exact bodyTrailingMiss snapshot bodyMember
    · exact loopTrailingMiss snapshot loopMember
    · exact rootNextMiss snapshot nextMember
    · simpa [rootSnapshot] using rootPathDifferent
    · simpa [rootPrefixPaths snapshot prefixMember] using rootPathDifferent
  exact reachedPrimitiveRunFromScopeFactor data env finalTrace evalTrace stageNumber stage
    stageStored stageTrace stageFind laneStructural grid.child child childStored childInputs
    gridPath (gridFuel - 1) target targetBound
    (by simpa [laneStructural, gridPath] using childPrefix laneStructural childInputs gridPath)
    childResult (by simpa [laneStructural, gridPath] using childEvaluated) outerLeading trailing
    globalFactor globalTrailingMiss targetNode targetNodeStored payload payloadStored
    primitivePayload port portBound

end Mxx.We.DiamondWE
