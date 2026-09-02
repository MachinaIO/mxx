import MxxIrCore.NodeEquation

namespace Mxx.IR

def OccurrencePath.Under (base path : OccurrencePath) : Prop :=
  List.IsPrefix base.toList path.toList

theorem OccurrencePath.under_refl (path : OccurrencePath) : path.Under path := by
  exact List.prefix_refl _

theorem OccurrencePath.under_push (path : OccurrencePath) (frame : OccurrenceFrame) :
    path.Under (path.push frame) := by
  unfold Under
  rw [Array.toList_push]
  exact List.prefix_append _ _

theorem flatStepScopesUnder {backend : SemanticBackend} (path : OccurrencePath)
    (nodeResult : NodeResult backend) (nextResult finalResult : ScopeResult backend)
    (current : ScopeTrace backend) (nodeEmpty : nodeResult.scopes = #[])
    (nextUnder : ∀ snapshot ∈ nextResult.scopes, path.Under snapshot.occurrence)
    (finalStored : finalResult = {
      values := nextResult.values
      scopes := nodeResult.scopes ++ nextResult.scopes ++ #[current] })
    (currentPath : current.occurrence = path) :
    ∀ snapshot ∈ finalResult.scopes, path.Under snapshot.occurrence := by
  subst finalResult
  rw [nodeEmpty]
  intro snapshot membership
  rw [Array.empty_append, Array.mem_append] at membership
  rcases membership with previous | currentMember
  · exact nextUnder snapshot previous
  · have snapshotEq : snapshot = current := by simpa using currentMember
    subst snapshot
    rw [currentPath]
    exact path.under_refl

theorem nodeStepScopesUnder {backend : SemanticBackend} (path : OccurrencePath)
    (nodeResult : NodeResult backend) (nextResult finalResult : ScopeResult backend)
    (current : ScopeTrace backend)
    (nodeUnder : ∀ snapshot ∈ nodeResult.scopes, path.Under snapshot.occurrence)
    (nextUnder : ∀ snapshot ∈ nextResult.scopes, path.Under snapshot.occurrence)
    (finalStored : finalResult = {
      values := nextResult.values
      scopes := nodeResult.scopes ++ nextResult.scopes ++ #[current] })
    (currentPath : current.occurrence = path) :
    ∀ snapshot ∈ finalResult.scopes, path.Under snapshot.occurrence := by
  subst finalResult
  intro snapshot membership
  rw [Array.mem_append] at membership
  rcases membership with previous | currentMember
  · rw [Array.mem_append] at previous
    rcases previous with nodeMember | nextMember
    · exact nodeUnder snapshot nodeMember
    · exact nextUnder snapshot nextMember
  · have snapshotEq : snapshot = current := by simpa using currentMember
    subst snapshot
    rw [currentPath]
    exact path.under_refl

theorem array_mem_foldl_append_second {α β : Type} (items : Array (α × Array β))
    (initial : Array β) (value : β)
    (membership : value ∈ items.foldl (fun result item => result ++ item.2) initial) :
    value ∈ initial ∨ ∃ index, ∃ bound : index < items.size, value ∈ items[index].2 := by
  have split : value ∈ initial ∨ value ∈ (items.map Prod.snd).flatten := by
    simpa using membership
  rcases split with original | flattened
  · exact Or.inl original
  · rw [Array.mem_flatten] at flattened
    rcases flattened with ⟨itemScopes, itemScopesMember, valueMember⟩
    rw [Array.mem_map] at itemScopesMember
    rcases itemScopesMember with ⟨item, itemMember, itemScopesEq⟩
    subst itemScopes
    rcases Array.mem_iff_getElem.mp itemMember with ⟨index, bound, stored⟩
    exact Or.inr ⟨index, bound, by simpa [stored] using valueMember⟩

/- Folding an array of per-lane scope arrays preserves an existing prefix.  This small algebraic
   fact lets the grid proof separate the selected lane without changing the evaluator's order. -/
theorem array_foldl_append_second_initial {α β : Type}
    (items : Array (α × Array β)) (initial : Array β) :
    items.foldl (fun result item => result ++ item.2) initial =
      initial ++ items.foldl (fun result item => result ++ item.2) #[] := by
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  induction items.toList generalizing initial with
  | nil => simp
  | cons head tail ih =>
      simp only [List.foldl_cons, Array.empty_append]
      rw [ih (initial ++ head.2), ih head.2]
      simp [Array.append_assoc]

/- The grid evaluator concatenates lane scopes in increasing lane order.  Splitting the backing
   lane array at `index` therefore exposes exactly `items[index].2` between the earlier and later
   lane scopes. -/
theorem array_foldl_append_second_factor {α β : Type}
    (items : Array (α × Array β)) (index : Nat) (bound : index < items.size) :
    items.foldl (fun result item => result ++ item.2) #[] =
      (items.extract 0 index).foldl (fun result item => result ++ item.2) #[] ++
      items[index].2 ++
      (items.extract (index + 1) items.size).foldl
        (fun result item => result ++ item.2) #[] := by
  have setSelf : items.set index items[index] = items := Array.set_getElem_self bound
  have decomposition := Array.set_eq_push_extract_append_extract
    (as := items) (i := index) bound (a := items[index])
  rw [setSelf] at decomposition
  conv_lhs => rw [decomposition]
  rw [Array.foldl_append, Array.foldl_push, array_foldl_append_second_initial]

/- A scope-free step is an evaluator-produced node step whose node result contributes no nested
   scopes.  Inputs, samplers, family operations, and supported primitives have this form. -/
def ScopeFreeStep {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) : Prop :=
  ∃ nodeValue, ∃ result : NodeResult backend, ∃ nextResult : ScopeResult backend,
    scope.nodes[index]? = some nodeValue ∧
    result.scopes = #[] ∧
    outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
    evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
      inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
      .ok nextResult ∧
    finalResult = {
      values := nextResult.values
      scopes := result.scopes ++ nextResult.scopes ++ #[{
        scope := scope.id
        occurrence := path
        values := appendNodeBindings scope.id index values result.values }] }

/- A generated suffix callback classifies one concrete node result relative to a selected nested
   occurrence.  The continuation and parent snapshot still come from the evaluator equation. -/
def AvoidingScopeStep {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path selectedPath : OccurrencePath) (index : Nat) (values : Array (Binding backend))
    (fuel : Nat) (finalResult : ScopeResult backend) : Prop :=
  ∃ nodeValue, ∃ result : NodeResult backend, ∃ nextResult : ScopeResult backend,
    scope.nodes[index]? = some nodeValue ∧
    (∀ snapshot ∈ result.scopes, snapshot.occurrence ≠ selectedPath) ∧
    outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
    evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
      inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
      .ok nextResult ∧
    finalResult = {
      values := nextResult.values
      scopes := result.scopes ++ nextResult.scopes ++ #[{
        scope := scope.id
        occurrence := path
        values := appendNodeBindings scope.id index values result.values }] }

theorem ScopeFreeStep.avoiding {backend : SemanticBackend}
    {data : ProgramData} {env : EvalEnv backend data} {structural : StructuralEnv}
    {trace : Trace backend} {stageNumber : Nat} {stage : Stage} {scopeNumber : ScopeId}
    {scope : Scope} {stageStored : data.stages[stageNumber]? = some stage}
    {scopeStored : scopeAt stage scopeNumber = some scope} {inputs : Array (Binding backend)}
    {path selectedPath : OccurrencePath} {index : Nat} {values : Array (Binding backend)}
    {fuel : Nat} {finalResult : ScopeResult backend}
    (step : ScopeFreeStep data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel finalResult) :
    AvoidingScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path selectedPath index values fuel finalResult := by
  obtain ⟨nodeValue, result, nextResult, nodeStored, resultEmpty, typesMatch, nextStored,
      finalStored⟩ := step
  exact ⟨nodeValue, result, nextResult, nodeStored, by simp [resultEmpty], typesMatch,
    nextStored, finalStored⟩

/- Finite generated node cases are enough to prove a whole evaluator suffix misses one selected
   occurrence.  No caller supplies a trailing-scope equation: every recursive continuation is the
   one stored in the successful evaluator step. -/
theorem evalScope_success_suffix_avoids {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path selectedPath : OccurrencePath) (parentDifferent : path ≠ selectedPath)
    (start : Nat)
    (step : ∀ (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (finalResult : ScopeResult backend), start ≤ index → index < scope.nodes.size → fuel ≠ 0 →
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path index values fuel = .ok finalResult →
      AvoidingScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path selectedPath index values fuel finalResult)
    (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend)
    (startBound : start ≤ index)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel = .ok finalResult) :
    ∀ snapshot ∈ finalResult.scopes, snapshot.occurrence ≠ selectedPath := by
  induction fuel generalizing index values finalResult with
  | zero => simp [evalScope] at success
  | succ fuel ih =>
      intro snapshot membership
      by_cases indexBound : index < scope.nodes.size
      · obtain ⟨nodeValue, nodeResult, nextResult, _, nodeMiss, _, nextSuccess, finalStored⟩ :=
          step index values (fuel + 1) finalResult startBound indexBound (by omega) success
        have scopesEq : finalResult.scopes = nodeResult.scopes ++ nextResult.scopes ++ #[{
            scope := scope.id
            occurrence := path
            values := appendNodeBindings scope.id index values nodeResult.values }] :=
          congrArg ScopeResult.scopes finalStored
        rw [scopesEq, Array.mem_append] at membership
        rcases membership with previous | current
        · rw [Array.mem_append] at previous
          rcases previous with nodeMember | nextMember
          · exact nodeMiss snapshot nodeMember
          · exact ih (index := index + 1)
              (values := appendNodeBindings scope.id index values nodeResult.values)
              (finalResult := nextResult) (by omega) nextSuccess snapshot nextMember
        · have snapshotEq : snapshot = {
              scope := scope.id
              occurrence := path
              values := appendNodeBindings scope.id index values nodeResult.values } := by
            simpa using current
          subst snapshot
          exact parentDifferent
      · have atEnd : scope.nodes.size ≤ index := by omega
        rw [evalScope] at success
        simp only [if_neg (by omega : fuel + 1 ≠ 0)] at success
        rw [dif_neg (by omega : ¬index < scope.nodes.size)] at success
        have finalEq : finalResult = {
            values := values
            scopes := #[{ scope := scope.id, occurrence := path, values := values }] } :=
          (Except.ok.inj success).symm
        subst finalResult
        simp at membership
        subst snapshot
        exact parentDifferent

/- Walking only scope-free nodes to `target` leaves the target evaluation as the leading scope
   factor.  Every accumulated snapshot has the unchanged parent path, so a strict nested lane path
   cannot be shadowed by the prefix. -/
theorem generatedScopeFreePrefixFactor {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (rootValues : Array (Binding backend)) (rootFuel target : Nat)
    (targetBound : target ≤ scope.nodes.size)
    (step : ∀ (limit index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (finalResult : ScopeResult backend), limit = target → index < limit → fuel ≠ 0 →
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path index values fuel = .ok finalResult →
      ScopeFreeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 rootValues rootFuel = .ok rootResult) :
    ∃ values fuel currentResult trailing,
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path target values fuel = .ok currentResult ∧
      fuel = rootFuel - target ∧
      rootResult.scopes = currentResult.scopes ++ trailing ∧
      ∀ snapshot ∈ trailing, snapshot.occurrence = path := by
  induction target generalizing rootValues rootFuel rootResult with
  | zero =>
      exact ⟨rootValues, rootFuel, rootResult, #[], rootSuccess, by simp, by simp, by simp⟩
  | succ target ih =>
      have targetIndexBound : target < scope.nodes.size := by omega
      let previousStep := fun (limit index : Nat) (values : Array (Binding backend))
          (fuel : Nat) (currentResult : ScopeResult backend) (limitEq : limit = target)
          (indexBound : index < limit) (fuelPositive : fuel ≠ 0) (currentSuccess :
            evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
              scopeStored inputs path index values fuel = .ok currentResult) =>
        step (limit + 1) index values fuel currentResult (by omega) (by omega) fuelPositive
          currentSuccess
      obtain ⟨values, fuel, currentResult, trailing, currentSuccess, fuelEq, rootFactor,
          trailingPath⟩ :=
        ih previousStep (rootValues := rootValues) (rootFuel := rootFuel)
          (rootResult := rootResult) (targetBound := by omega) rootSuccess
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalScope] at currentSuccess
        simp [fuelZero] at currentSuccess
      obtain ⟨nodeValue, nodeResult, nextResult, _, scopesEmpty, _, nextSuccess, finalStored⟩ :=
        step (target + 1) target values fuel currentResult rfl (by omega) fuelPositive
          currentSuccess
      let currentSnapshot : ScopeTrace backend := {
        scope := scope.id
        occurrence := path
        values := appendNodeBindings scope.id target values nodeResult.values }
      refine ⟨appendNodeBindings scope.id target values nodeResult.values, fuel - 1, nextResult,
        #[currentSnapshot] ++ trailing, nextSuccess, by omega, ?_, ?_⟩
      · rw [rootFactor, finalStored, scopesEmpty]
        simp [currentSnapshot]
      · intro snapshot membership
        simp only [Array.mem_append, Array.mem_singleton] at membership
        rcases membership with rfl | membership
        · rfl
        · exact trailingPath snapshot membership

/- Once the node index reaches the end of a scope, the evaluator emits exactly one parent-scope
   snapshot.  A terminal grid uses this fact to show that no unclassified nested suffix follows
   its lane scopes. -/
theorem evalScope_success_at_end {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (atEnd : scope.nodes.size ≤ index)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel = .ok finalResult) :
    finalResult = {
      values := values
      scopes := #[{ scope := scope.id, occurrence := path, values := values }] } := by
  have fuelPositive : fuel ≠ 0 := by
    intro fuelZero
    rw [evalScope] at success
    simp [fuelZero] at success
  rw [evalScope] at success
  simp only [if_neg fuelPositive] at success
  rw [dif_neg (by omega : ¬index < scope.nodes.size)] at success
  exact (Except.ok.inj success).symm

theorem evalPrimitiveNode_success_scopes_empty {backend : SemanticBackend}
    (structural : StructuralEnv) (stage scope node : Nat) (payload : NodePayload)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType)
    (result : NodeResult backend)
    (success : evalPrimitiveNode backend structural stage scope node payload arguments outputs =
      .ok result) :
    result.scopes = #[] := by
  unfold evalPrimitiveNode at success
  obtain ⟨primitiveValues, _, resultStored⟩ := equation_bind_eq_ok _ _ _ success
  have resultEq : result = NodeResult.ofValues primitiveValues := (Except.ok.inj resultStored).symm
  subst result
  rfl

/- Family indexing is evaluated by dedicated `evalScope` branches rather than by
   `evalPrimitiveNode`.  These five operations nevertheless have the same scope behavior: they
   compute one array of values and introduce no child scope.  Recording that common behavior lets
   generated prefix and suffix proofs remain independent of the particular family expression. -/
inductive ScopeFreeFamilyPayload : NodePayload → Prop where
  | getStatic (indices) : ScopeFreeFamilyPayload (.familyGetStatic indices)
  | getDynamic (rank) : ScopeFreeFamilyPayload (.familyGetDynamic rank)
  | selectAxis (axis) : ScopeFreeFamilyPayload (.familySelectAxis axis)
  | reindex (shape map) : ScopeFreeFamilyPayload (.familyReindex shape map)

theorem evalScope_success_scope_free_family_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue)
    (payload : NodePayload) (payloadStored : nodeValue.payload = payload)
    (familyPayload : ScopeFreeFamilyPayload payload)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel = .ok finalResult) :
    ScopeFreeStep data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel finalResult := by
  have closeValues (operation : Except EvalError (Array (DynamicValue backend)))
      (operationThenRest : (do
        let packed ← operation
        let result := NodeResult.ofValues packed
        if outputTypesMatch nodeValue.outputs.toList result.values.toList then
          let nextResult ← evalScope data env structural trace stageNumber stage scopeNumber scope
            stageStored scopeStored inputs path (index + 1)
              (appendNodeBindings scope.id index values result.values) (fuel - 1)
          pure ({
            values := nextResult.values
            scopes := result.scopes ++ nextResult.scopes ++ #[({
              scope := scope.id
              occurrence := path
              values := appendNodeBindings scope.id index values result.values } :
                ScopeTrace backend)] } : ScopeResult backend)
        else throw (EvalError.wrongType stageNumber scope.id index)) = .ok finalResult) :
      ScopeFreeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult := by
    obtain ⟨packed, _, afterPacked⟩ := equation_bind_eq_ok _ _ _ operationThenRest
    dsimp only at afterPacked
    split at afterPacked
    · rename_i typesMatch
      obtain ⟨nextResult, nextStored, finalStored⟩ := equation_bind_eq_ok _ _ _ afterPacked
      refine ⟨nodeValue, NodeResult.ofValues packed, nextResult, nodeStored, rfl, typesMatch,
        ?_, ?_⟩
      · simpa [NodeResult.ofValues] using nextStored
      · simpa [NodeResult.ofValues] using (Except.ok.inj finalStored).symm
    · contradiction
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  rename_i actualNode actualStored
  have nodeEq : actualNode = nodeValue := Option.some.inj (actualStored.symm.trans nodeStored)
  subst actualNode
  obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
  cases familyPayload with
  | getStatic indices =>
      simp only [payloadStored, samplerPayload, Bool.false_eq_true] at afterArguments
      rw [dif_neg (show ¬False by simp)] at afterArguments
      exact closeValues (familyStaticGet structural stageNumber scope.id index indices
        argumentValues) afterArguments
  | getDynamic rank =>
      simp only [payloadStored, samplerPayload, Bool.false_eq_true] at afterArguments
      rw [dif_neg (show ¬False by simp)] at afterArguments
      exact closeValues (familyDynamicGet structural stageNumber scope.id index rank
        argumentValues) afterArguments
  | selectAxis axis =>
      simp only [payloadStored, samplerPayload, Bool.false_eq_true] at afterArguments
      rw [dif_neg (show ¬False by simp)] at afterArguments
      cases outputStored : nodeValue.outputs[0]? with
      | none =>
          simp only [outputStored] at afterArguments
          obtain ⟨impossible, impossibleStored, _⟩ :=
            equation_bind_eq_ok _ _ _ afterArguments
          cases impossibleStored
      | some declared =>
          simp only [outputStored] at afterArguments
          exact closeValues (familySelectAxisExact structural stageNumber scope.id index axis
            declared argumentValues) afterArguments
  | reindex outputShape map =>
      simp only [payloadStored, samplerPayload, Bool.false_eq_true] at afterArguments
      rw [dif_neg (show ¬False by simp)] at afterArguments
      cases outputStored : nodeValue.outputs[0]? with
      | none =>
          simp only [outputStored] at afterArguments
          obtain ⟨impossible, impossibleStored, _⟩ :=
            equation_bind_eq_ok _ _ _ afterArguments
          cases impossibleStored
      | some declared =>
          simp only [outputStored] at afterArguments
          exact closeValues (familyReindex structural stageNumber scope.id index outputShape map
            declared argumentValues) afterArguments

theorem evalScope_success_scopes_under {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage)
    (stageStored : data.stages[stageNumber]? = some stage)
    (structural : StructuralEnv) (scopeNumber : ScopeId) (scope : Scope)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel = .ok finalResult) :
    ∀ snapshot ∈ finalResult.scopes, path.Under snapshot.occurrence := by
  let motive1 := fun (structural : StructuralEnv) (scopeNumber : ScopeId) (scope : Scope)
      (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
      (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat) =>
    ∀ finalResult,
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path index values fuel = .ok finalResult →
      ∀ snapshot ∈ finalResult.scopes, path.Under snapshot.occurrence
  let motive2 := fun (childNumber : ScopeId) (child : Scope)
      (childStored : scopeAt stage childNumber = some child) (loop : LoopPayload) (owner : NodeId)
      (arguments : Array (DynamicValue backend)) (structural : StructuralEnv)
      (path : OccurrencePath) (iteration fuel : Nat) =>
    ∀ finalResult,
      evalSequentialLoop data env trace stageNumber stage childNumber child stageStored childStored
        loop owner arguments structural path iteration fuel = .ok finalResult →
      ∀ snapshot ∈ finalResult.scopes, path.Under snapshot.occurrence
  have both := evalScope.induct (backend := backend) stageNumber stage motive1 motive2
    (by
      intro structural scopeNumber scope scopeStored inputs path index values finalResult success
      simp [evalScope] at success)
    (by
      intro structural scopeNumber scope scopeStored inputs path index values fuel fuelPositive
        indexBound missing finalResult success
      have outOfBounds : scope.nodes.size ≤ index := Array.getElem?_eq_none_iff.mp missing
      omega)
    (by
      intro structural scopeNumber scope scopeStored inputs path index values fuel fuelPositive
        indexBound nodeValue nodeStored nextIH subgraphIH gridIH loopIH finalResult success
        snapshot membership
      rcases nodeValue with ⟨payload, nodeArguments, nodeOutputs⟩
      have closeFlat (result : NodeResult backend) (nextResult : ScopeResult backend)
          (resultEmpty : result.scopes = #[])
          (nextStored : evalScope data env structural trace stageNumber stage scopeNumber scope
            stageStored scopeStored inputs path (index + 1)
              (appendNodeBindings scope.id index values result.values) (fuel - 1) = .ok nextResult)
          (finalStored : finalResult = {
            values := nextResult.values
            scopes := result.scopes ++ nextResult.scopes ++ #[{
              scope := scope.id
              occurrence := path
              values := appendNodeBindings scope.id index values result.values }] }) :
          path.Under snapshot.occurrence := by
        exact flatStepScopesUnder path result nextResult finalResult _ resultEmpty
          (nextIH result nextResult nextStored) finalStored rfl snapshot membership
      have closeValues (operation : Except EvalError (Array (DynamicValue backend)))
          (operationThenRest : (do
            let packed ← operation
            let result := NodeResult.ofValues packed
            if outputTypesMatch nodeOutputs.toList result.values.toList then
              let nextResult ← evalScope data env structural trace stageNumber stage scopeNumber scope
                stageStored scopeStored inputs path (index + 1)
                  (appendNodeBindings scope.id index values result.values) (fuel - 1)
              pure ({
                values := nextResult.values
                scopes := result.scopes ++ nextResult.scopes ++ #[({
                  scope := scope.id
                  occurrence := path
                  values := appendNodeBindings scope.id index values result.values } :
                    ScopeTrace backend)] } :
                ScopeResult backend)
            else throw (EvalError.wrongType stageNumber scope.id index)) = .ok finalResult) :
          path.Under snapshot.occurrence := by
        obtain ⟨packed, packedStored, afterPacked⟩ :=
          equation_bind_eq_ok _ _ _ operationThenRest
        dsimp only at afterPacked
        split at afterPacked
        · obtain ⟨nextResult, nextStored, finalStored⟩ :=
            equation_bind_eq_ok _ _ _ afterPacked
          exact closeFlat (NodeResult.ofValues packed) nextResult rfl nextStored
            (Except.ok.inj finalStored).symm
        · contradiction
      by_cases primitivePayload : PrimitiveNodePayload payload
      · obtain ⟨_, result, nextResult, _, primitiveStored, _, nextStored, finalStored⟩ :=
          evalScope_success_primitive_step data env structural trace stageNumber stage scopeNumber
            scope stageStored scopeStored inputs path index values fuel finalResult fuelPositive
            indexBound _ nodeStored payload rfl primitivePayload success
        exact closeFlat result nextResult
          (evalPrimitiveNode_success_scopes_empty structural stageNumber scope.id index payload
            _ nodeOutputs result primitiveStored) nextStored finalStored
      by_cases isSampler : samplerPayload payload = true
      · obtain ⟨_, sampled, nextResult, _, _, _, nextStored, finalStored⟩ :=
          evalScope_success_sampler_step data env structural trace stageNumber stage scopeNumber scope
            stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound
            _ nodeStored payload rfl isSampler success
        exact closeFlat (NodeResult.ofValues sampled) nextResult rfl nextStored (by
          simpa [NodeResult.ofValues] using finalStored)
      have unsupportedImpossible (unsupported : UnsupportedPrimitivePayload payload) : False := by
        cases unsupported <;>
          rw [evalScope] at success <;>
          simp only [if_neg fuelPositive, dif_pos indexBound] at success <;>
          split at success
        all_goals try contradiction
        all_goals
          rename_i actualNode actualStored
          have nodeEq := Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ :=
            equation_bind_eq_ok _ _ _ success
          simp only [samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          simp [evalPrimitiveNode, primitive] at afterArguments
          change (Except.error (.unsupportedPrimitive stageNumber scope.id index) :
            Except EvalError (ScopeResult backend)) = .ok finalResult at afterArguments
          cases afterArguments
      cases hPayload : payload
      case neg.input inputIndex =>
        obtain ⟨_, result, nextResult, _, resultStored, _, nextStored, finalStored⟩ :=
          evalScope_success_input_step data env structural trace stageNumber stage scopeNumber scope
            stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound
            _ nodeStored inputIndex hPayload success
        have resultEmpty : result.scopes = #[] := by
          rcases resultStored with ⟨_, _, rfl⟩ | ⟨_, _, _, _, rfl⟩ <;> rfl
        exact closeFlat result nextResult resultEmpty nextStored finalStored
      case neg.artifactInput artifact =>
        obtain ⟨_, _, _, _, value, nextResult, _, _, _, _, _, _, nextStored, finalStored⟩ :=
          evalScope_success_artifact_step data env structural trace stageNumber stage scopeNumber scope
            stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound
            _ nodeStored artifact hPayload success
        exact closeFlat (NodeResult.ofValues #[value]) nextResult rfl nextStored (by
          simpa [NodeResult.ofValues] using finalStored)
      case neg.constantMatrix matrixType literal =>
        cases literal with
        | zero | identity | unitRow | unitColumn | powerOfBase | rotation | polynomial =>
            exact (primitivePayload (hPayload ▸ by constructor; trivial)).elim
        | gadget base small =>
            cases small
            · exact (primitivePayload (hPayload ▸ by constructor; simp)).elim
            · exact False.elim (unsupportedImpossible (hPayload ▸ by constructor))
      case neg.gadgetDecompose base small digits =>
        cases small
        · exact (primitivePayload (hPayload ▸ by constructor)).elim
        · exact False.elim (unsupportedImpossible (hPayload ▸ by constructor))
      case neg.subgraphCall call =>
        obtain ⟨_, child, childStored, childInputs, childResult, childOutputs, nextResult, _, _, childEvaluated,
            _, _, nextStored, finalStored⟩ :=
          evalScope_success_subgraph_step data env structural trace stageNumber stage scopeNumber
            scope stageStored scopeStored inputs path index values fuel finalResult fuelPositive
            indexBound _ nodeStored call hPayload success
        let nodeResult : NodeResult backend := { values := childOutputs, scopes := childResult.scopes }
        apply nodeStepScopesUnder path nodeResult nextResult finalResult _
        · intro childSnapshot childMember
          exact path.under_push _ |>.trans
            (subgraphIH call child childStored childInputs childResult childEvaluated childSnapshot
              childMember)
        · exact nextIH nodeResult nextResult (by simpa [nodeResult] using nextStored)
        · simpa [nodeResult] using finalStored
        · rfl
        · exact membership
      case neg.sequentialLoop loop =>
        obtain ⟨argumentValues, child, childStored, loopResult, nextResult, _, loopEvaluated, _,
            nextStored, finalStored⟩ :=
          evalScope_success_sequentialLoop_step data env structural trace stageNumber stage
            scopeNumber scope stageStored scopeStored inputs path index values fuel finalResult
            fuelPositive indexBound _ nodeStored loop hPayload success
        apply nodeStepScopesUnder path loopResult nextResult finalResult _
        · exact loopIH argumentValues loop child childStored loopResult loopEvaluated
        · exact nextIH loopResult nextResult nextStored
        · exact finalStored
        · rfl
        · exact membership
      case neg.familyPack shape =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, argumentsStored, afterArguments⟩ :=
            equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          cases outputStored : nodeOutputs[0]? with
          | none =>
              simp only [outputStored] at afterArguments
              obtain ⟨impossible, impossibleStored, _⟩ :=
                equation_bind_eq_ok _ _ _ afterArguments
              cases impossibleStored
          | some output =>
              simp only [outputStored] at afterArguments
              obtain ⟨packed, packedStored, afterPacked⟩ :=
                equation_bind_eq_ok _ _ _ afterArguments
              obtain ⟨result, resultStored, afterResult⟩ :=
                equation_bind_eq_ok _ _ _ afterPacked
              cases resultStored
              split at afterResult
              · rename_i typesMatch
                obtain ⟨nextResult, nextStored, finalStored⟩ :=
                  equation_bind_eq_ok _ _ _ afterResult
                have finalEq : finalResult = {
                    values := nextResult.values
                    scopes := nextResult.scopes.push {
                      scope := scope.id
                      occurrence := path
                      values := appendNodeBindings scope.id index values packed } } :=
                  by simpa [NodeResult.ofValues] using (Except.ok.inj finalStored).symm
                exact closeFlat (NodeResult.ofValues packed) nextResult rfl (by
                  simpa [NodeResult.ofValues] using nextStored) (by
                  simpa [NodeResult.ofValues] using finalEq)
              · contradiction
      case neg.familyGetStatic indices =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          exact closeValues
            (familyStaticGet structural stageNumber scope.id index indices argumentValues)
            afterArguments
      case neg.familyGetDynamic rank =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          exact closeValues
            (familyDynamicGet structural stageNumber scope.id index rank argumentValues)
            afterArguments
      case neg.familySelectAxis axis =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          cases outputStored : nodeOutputs[0]? with
          | none =>
              simp only [outputStored] at afterArguments
              obtain ⟨impossible, impossibleStored, _⟩ := equation_bind_eq_ok _ _ _ afterArguments
              cases impossibleStored
          | some declared =>
              simp only [outputStored] at afterArguments
              exact closeValues
                (familySelectAxisExact structural stageNumber scope.id index axis declared
                  argumentValues) afterArguments
      case neg.familyReindex outputShape map =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          cases outputStored : nodeOutputs[0]? with
          | none =>
              simp only [outputStored] at afterArguments
              obtain ⟨impossible, impossibleStored, _⟩ := equation_bind_eq_ok _ _ _ afterArguments
              cases impossibleStored
          | some declared =>
              simp only [outputStored] at afterArguments
              exact closeValues
                (familyReindex structural stageNumber scope.id index outputShape map declared
                  argumentValues) afterArguments
      case neg.familyGather outputShape inputRank =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          cases outputStored : nodeOutputs[0]? with
          | none =>
              simp only [outputStored] at afterArguments
              obtain ⟨impossible, impossibleStored, _⟩ := equation_bind_eq_ok _ _ _ afterArguments
              cases impossibleStored
          | some declared =>
              simp only [outputStored] at afterArguments
              exact closeValues
                (familyGatherExact structural stageNumber scope.id index outputShape inputRank
                  declared argumentValues) afterArguments
      case neg.gadgetTrapdoor => exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.uniformResidueSample =>
        exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.uniformIntervalSample =>
        exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.gaussianSample => exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.hashSample => exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.trapdoorSample => exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.preimageSample => exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.familyPreimageSample =>
        exact (isSampler (by simp [hPayload, samplerPayload])).elim
      case neg.parallelGrid grid =>
        obtain ⟨argumentValues, child, childStored, concreteShape, lanes, laneResults, packed,
            nextResult, _, _, lanesStored, laneResultsStored, _, _, nextStored, finalStored⟩ :=
          evalScope_success_parallelGrid_step data env structural trace stageNumber stage
            scopeNumber scope stageStored scopeStored inputs path index values fuel finalResult
            fuelPositive indexBound _ nodeStored grid hPayload success
        let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
        let nodeResult : NodeResult backend := { values := packed, scopes := laneScopes }
        apply nodeStepScopesUnder path nodeResult nextResult finalResult _
        · intro laneSnapshot laneMember
          have foldedMember : laneSnapshot ∈
              laneResults.foldl (fun result item => result ++ item.2) #[] := by
            simpa [nodeResult, laneScopes] using laneMember
          rcases array_mem_foldl_append_second laneResults #[] laneSnapshot foldedMember with
            impossible | ⟨laneIndex, laneResultBound, laneScopeMember⟩
          · simp at impossible
          · let laneFunction := fun lane => do
              let coordinates := coordinatesFromOffset concreteShape.toList lane
              let laneStructural := { structural with
                axes := (coordinates.map Int.ofNat).toArray
                slots := grid.indexSlots.zip coordinates.toArray |>.map
                  (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
              let lanePath := path.push {
                stage := stageNumber, scope := scope.id, owner := index,
                laneOrIteration := lane }
              let laneArguments ← gridInputArguments laneStructural stageNumber scope.id index
                grid.inputModes argumentValues
              let childInputs ← checkedChildInputs stageNumber scope.id index child laneArguments
              let childResult ← evalScope data env laneStructural trace stageNumber stage grid.child
                child stageStored childStored childInputs lanePath 0 #[] (fuel - 1)
              let outputs ← child.outputs.mapM (fun output =>
                (match lookup childResult.values output with
                | some value => Except.ok value
                | none => Except.error
                    (EvalError.missingPort stageNumber child.id output.node output.port) :
                  Except EvalError (DynamicValue backend)))
              pure (outputs, childResult.scopes)
            have sourceBound : laneIndex < (Array.range lanes).size := by
              have sizeSatisfies := Array.size_mapM laneFunction (Array.range lanes)
              rw [SatisfiesM_Except_eq] at sizeSatisfies
              have sizeEq : laneResults.size = (Array.range lanes).size :=
                sizeSatisfies laneResults (by simpa [laneFunction] using laneResultsStored)
              simpa [sizeEq] using laneResultBound
            obtain ⟨resultBound, laneStored⟩ :=
              array_mapM_getElem laneFunction (by simpa [laneFunction] using laneResultsStored)
                sourceBound
            obtain ⟨laneArguments, laneArgumentsStored, afterLaneArguments⟩ :=
              equation_bind_eq_ok _ _ _ laneStored
            obtain ⟨childInputs, childInputsStored, afterChildInputs⟩ :=
              equation_bind_eq_ok _ _ _ afterLaneArguments
            obtain ⟨childResult, childEvaluated, afterChild⟩ :=
              equation_bind_eq_ok _ _ _ afterChildInputs
            obtain ⟨outputs, outputsStored, pairStored⟩ :=
              equation_bind_eq_ok _ _ _ afterChild
            have scopesEq : laneResults[laneIndex].2 = childResult.scopes := by
              have pairEq : laneResults[laneIndex] = (outputs, childResult.scopes) :=
                (Except.ok.inj pairStored).symm
              exact congrArg Prod.snd pairEq
            have childMember : laneSnapshot ∈ childResult.scopes := by
              simpa [scopesEq] using laneScopeMember
            have childUnder := gridIH grid child childStored concreteShape laneIndex childInputs
              childResult (by simpa [laneFunction] using childEvaluated) laneSnapshot childMember
            exact path.under_push _ |>.trans childUnder
        · exact nextIH nodeResult nextResult (by simpa [nodeResult] using nextStored)
        · simpa [nodeResult, laneScopes] using finalStored
        · rfl
        · exact membership
      case neg.select count =>
        rw [evalScope] at success
        simp only [if_neg fuelPositive, dif_pos indexBound] at success
        split at success
        · contradiction
        · rename_i actualNode actualStored
          have nodeEq : actualNode = {
              payload := payload, arguments := nodeArguments, outputs := nodeOutputs } :=
            Option.some.inj (actualStored.symm.trans nodeStored)
          subst actualNode
          obtain ⟨argumentValues, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ success
          simp only [hPayload, samplerPayload, Bool.false_eq_true] at afterArguments
          rw [dif_neg (show ¬False by simp)] at afterArguments
          obtain ⟨branchCount, branchCountStored, afterBranchCount⟩ :=
            equation_bind_eq_ok _ _ _ afterArguments
          split at afterBranchCount
          · contradiction
          · cases firstStored : argumentValues[0]? with
            | none =>
                simp only [firstStored] at afterBranchCount
                obtain ⟨unitValue, unitStored, afterUnit⟩ :=
                  equation_bind_eq_ok _ _ _ afterBranchCount
                obtain ⟨impossible, impossibleStored, _⟩ :=
                  equation_bind_eq_ok _ _ _ afterUnit
                cases impossibleStored
            | some first =>
                cases selectorStored : dynamicInt? first with
                | none =>
                    simp only [firstStored, selectorStored] at afterBranchCount
                    obtain ⟨unitValue, unitStored, afterUnit⟩ :=
                      equation_bind_eq_ok _ _ _ afterBranchCount
                    obtain ⟨impossible, impossibleStored, _⟩ :=
                      equation_bind_eq_ok _ _ _ afterUnit
                    cases impossibleStored
                | some selector =>
                    simp only [firstStored, selectorStored] at afterBranchCount
                    obtain ⟨unitValue, unitStored, afterUnit⟩ :=
                      equation_bind_eq_ok _ _ _ afterBranchCount
                    have unitEq : unitValue = PUnit.unit := Except.ok.inj unitStored
                    subst unitValue
                    obtain ⟨selectedInteger, selectedIntegerStored, afterSelector⟩ :=
                      equation_bind_eq_ok _ _ _ afterUnit
                    have selectorEq : selectedInteger = selector :=
                      (Except.ok.inj selectedIntegerStored).symm
                    subst selectedInteger
                    split at afterSelector
                    · rename_i selectorBound
                      cases selectedStored : argumentValues[selector.toNat + 1]? with
                      | none =>
                          simp only [selectedStored] at afterSelector
                          obtain ⟨impossible, impossibleStored, _⟩ :=
                            equation_bind_eq_ok _ _ _ afterSelector
                          cases impossibleStored
                      | some selected =>
                          simp only [selectedStored] at afterSelector
                          obtain ⟨result, resultStored, afterResult⟩ :=
                            equation_bind_eq_ok _ _ _ afterSelector
                          cases resultStored
                          split at afterResult
                          · obtain ⟨nextResult, nextStored, finalStored⟩ :=
                              equation_bind_eq_ok _ _ _ afterResult
                            exact closeFlat (NodeResult.ofValues #[selected]) nextResult rfl nextStored
                              (Except.ok.inj finalStored).symm
                          · contradiction
                    · contradiction
      all_goals first
        | exact False.elim (unsupportedImpossible (hPayload ▸ by constructor))
        | exact (primitivePayload (hPayload ▸ by constructor)).elim)
    (by
      intro structural scopeNumber scope scopeStored inputs path index values fuel fuelPositive
        indexBound finalResult success snapshot membership
      rw [evalScope] at success
      simp only [if_neg fuelPositive, dif_neg indexBound] at success
      have resultEq : finalResult = {
          values := values, scopes := #[{ scope := scope.id, occurrence := path, values := values }] } :=
        Except.ok.inj success.symm
      subst finalResult
      simp at membership
      subst snapshot
      exact path.under_refl)
    (by
      intro childNumber child childStored loop owner arguments structural path iteration finalResult success
      simp [evalSequentialLoop] at success)
    (by
      intro childNumber child childStored loop owner arguments structural path iteration fuel
        fuelPositive childIH restIH finalResult success snapshot membership
      rw [evalSequentialLoop] at success
      simp only [if_neg fuelPositive] at success
      obtain ⟨count, countStored, afterCount⟩ := equation_bind_eq_ok _ _ _ success
      split at afterCount
      · rename_i iterationBound
        obtain ⟨childInputs, inputsStored, afterInputs⟩ := equation_bind_eq_ok _ _ _ afterCount
        obtain ⟨childResult, childStoredResult, afterChild⟩ := equation_bind_eq_ok _ _ _ afterInputs
        obtain ⟨childValues, childValuesStored, afterValues⟩ := equation_bind_eq_ok _ _ _ afterChild
        split at afterValues
        · contradiction
        · obtain ⟨rest, restStored, finalStored⟩ := equation_bind_eq_ok _ _ _ afterValues
          have finalEq : finalResult = {
              values := rest.values, scopes := childResult.scopes ++ rest.scopes } :=
            (Except.ok.inj finalStored).symm
          subst finalResult
          simp only [Array.mem_append] at membership
          rcases membership with childMember | restMember
          · have underIteration := childIH count childInputs childResult childStoredResult snapshot
              childMember
            exact path.under_push _ |>.trans underIteration
          · exact restIH count childValues rest restStored snapshot restMember
      · have finalEq : finalResult = {
            values := arguments.extract 0 loop.carriedCount, scopes := #[] } :=
          (Except.ok.inj afterCount).symm
        subst finalResult
        simp at membership)
  exact both structural scopeNumber scope scopeStored inputs path index values fuel finalResult success

/- Distinct loop frames remain distinguishable after either path is extended by arbitrary nested
   scopes. Looking at the first frame after the common path recovers the iteration number. -/
theorem OccurrencePath.ne_of_push_under
    (path : OccurrencePath) (stage scope owner first second : Nat)
    (left right : OccurrencePath) (different : first ≠ second)
    (leftUnder : OccurrencePath.Under
      (path.push ⟨stage, scope, owner, first⟩) left)
    (rightUnder : OccurrencePath.Under
      (path.push ⟨stage, scope, owner, second⟩) right) :
    left ≠ right := by
  intro pathsEqual
  subst right
  have firstAt := leftUnder.getElem (i := path.size) (by simp)
  have secondAt := rightUnder.getElem (i := path.size) (by simp)
  have frameEq :
      OccurrenceFrame.mk stage scope owner first =
        OccurrenceFrame.mk stage scope owner second := by
    simpa [OccurrencePath.Under, Array.toList_push] using firstAt.trans secondAt.symm
  exact different (congrArg OccurrenceFrame.laneOrIteration frameEq)

theorem OccurrencePath.ne_of_distinct_push_under
    (path : OccurrencePath) (firstFrame secondFrame : OccurrenceFrame)
    (left right : OccurrencePath) (different : firstFrame ≠ secondFrame)
    (leftUnder : OccurrencePath.Under (path.push firstFrame) left)
    (rightUnder : OccurrencePath.Under (path.push secondFrame) right) :
    left ≠ right := by
  intro pathsEqual
  subst right
  have firstAt := leftUnder.getElem (i := path.size) (by simp)
  have secondAt := rightUnder.getElem (i := path.size) (by simp)
  have frameEq : firstFrame = secondFrame := by
    simpa [OccurrencePath.Under, Array.toList_push] using firstAt.trans secondAt.symm
  exact different frameEq

/- A path below a pushed frame is strictly deeper than the unextended parent path. -/
theorem OccurrencePath.ne_parent_of_push_under
    (path nested : OccurrencePath) (frame : OccurrenceFrame)
    (nestedUnder : OccurrencePath.Under (path.push frame) nested) :
    path ≠ nested := by
  intro pathsEqual
  subst nested
  have lengthBound := nestedUnder.length_le
  simp [Array.toList_push] at lengthBound

/- A terminal grid has only parent-path snapshots after its lane scopes.  Combining the selected
   lane factor with the scope-free prefix therefore promotes the selected child scopes to a factor
   of the whole loop-body evaluation, while preserving the reverse-lookup miss condition. -/
theorem terminalGrid_selected_child_factor {backend : SemanticBackend}
    (rootResult currentResult nextResult : ScopeResult backend)
    (laneScopes childScopes laneLeading laneTrailing prefixTrailing : Array (ScopeTrace backend))
    (parentSnapshot endSnapshot : ScopeTrace backend) (parentPath selectedPath : OccurrencePath)
    (rootFactor : rootResult.scopes = currentResult.scopes ++ prefixTrailing)
    (currentFactor : currentResult.scopes =
      laneScopes ++ nextResult.scopes ++ #[parentSnapshot])
    (nextFactor : nextResult.scopes = #[endSnapshot])
    (laneFactor : laneScopes = laneLeading ++ childScopes ++ laneTrailing)
    (laneTrailingMiss : ∀ snapshot ∈ laneTrailing, snapshot.occurrence ≠ selectedPath)
    (prefixPath : ∀ snapshot ∈ prefixTrailing, snapshot.occurrence = parentPath)
    (parentSnapshotPath : parentSnapshot.occurrence = parentPath)
    (endSnapshotPath : endSnapshot.occurrence = parentPath)
    (parentDifferent : parentPath ≠ selectedPath) :
    ∃ trailing,
      rootResult.scopes = laneLeading ++ childScopes ++ trailing ∧
      ∀ snapshot ∈ trailing, snapshot.occurrence ≠ selectedPath := by
  let trailing := laneTrailing ++ #[endSnapshot] ++ #[parentSnapshot] ++ prefixTrailing
  refine ⟨trailing, ?_, ?_⟩
  · rw [rootFactor, currentFactor, nextFactor, laneFactor]
    simp [trailing, Array.append_assoc]
  · intro snapshot membership
    simp only [trailing, Array.mem_append, Array.mem_singleton] at membership
    rcases membership with ((laneMember | rfl) | rfl) | prefixMember
    · exact laneTrailingMiss snapshot laneMember
    · simpa [endSnapshotPath] using parentDifferent
    · simpa [parentSnapshotPath] using parentDifferent
    · simpa [prefixPath snapshot prefixMember] using parentDifferent

/- A successful grid step contains one successful child evaluation for every in-range lane.
   This theorem selects one such evaluation and preserves the evaluator's exact scope order.  It
   deliberately stops at `laneScopes`: callers that know a grid is terminal decide how the parent
   scope snapshot is placed around this factor. -/
theorem evalScope_success_parallelGrid_selected_lane
    {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (stageStored : data.stages[stageNumber]? = some stage)
    (structural : StructuralEnv) (scopeNumber : ScopeId) (scope : Scope)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (grid : GridPayload)
    (payloadStored : nodeValue.payload = .parallelGrid grid)
    (concreteShape : Array Nat)
    (shapeStored : evalShape structural stageNumber scope.id index grid.shape = .ok concreteShape)
    (selected : Nat) (selectedBound : selected < shapeProductArray concreteShape)
    (selectedPath : OccurrencePath)
    (selectedUnder : OccurrencePath.Under
      (path.push {
        stage := stageNumber, scope := scope.id, owner := index,
        laneOrIteration := selected }) selectedPath)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues child, ∃ childStored : scopeAt stage grid.child = some child,
      ∃ laneResults : Array (Array (DynamicValue backend) × Array (ScopeTrace backend)),
      ∃ packed : Array (DynamicValue backend),
      ∃ nextResult : ScopeResult backend,
      ∃ laneArguments : Array (DynamicValue backend), ∃ childInputs : Array (Binding backend),
      ∃ childResult : ScopeResult backend, ∃ outputs : Array (DynamicValue backend),
        resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
        gridInputArguments
          { structural with
            axes := ((coordinatesFromOffset concreteShape.toList selected).map Int.ofNat).toArray
            slots := grid.indexSlots.zip
              (coordinatesFromOffset concreteShape.toList selected).toArray |>.map
                (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
          stageNumber scope.id index grid.inputModes argumentValues = .ok laneArguments ∧
        checkedChildInputs stageNumber scope.id index child laneArguments = .ok childInputs ∧
        evalScope data env
          { structural with
            axes := ((coordinatesFromOffset concreteShape.toList selected).map Int.ofNat).toArray
            slots := grid.indexSlots.zip
              (coordinatesFromOffset concreteShape.toList selected).toArray |>.map
                (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
          trace stageNumber stage grid.child child stageStored childStored childInputs
          (path.push {
            stage := stageNumber, scope := scope.id, owner := index,
            laneOrIteration := selected }) 0 #[] (fuel - 1) = .ok childResult ∧
        laneResults[selected]? = some (outputs, childResult.scopes) ∧
        evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
          scopeStored inputs path (index + 1)
          (appendNodeBindings scope.id index values packed) (fuel - 1) = .ok nextResult ∧
        finalResult = {
          values := nextResult.values
          scopes := laneResults.foldl (fun result item => result ++ item.2) #[] ++
            nextResult.scopes ++ #[{
              scope := scope.id
              occurrence := path
              values := appendNodeBindings scope.id index values packed }] } ∧
        ∃ leading trailing,
          laneResults.foldl (fun result item => result ++ item.2) #[] =
            leading ++ childResult.scopes ++ trailing ∧
          ∀ snapshot ∈ trailing, snapshot.occurrence ≠ selectedPath := by
  obtain ⟨argumentValues, child, childStored, actualShape, lanes, laneResults, packed,
      nextResult, argumentsStored, actualShapeStored, lanesStored, laneResultsStored, _, _,
      nextStored, finalStored⟩ :=
    evalScope_success_parallelGrid_step data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound
      nodeValue nodeStored grid payloadStored success
  have shapeEq : actualShape = concreteShape :=
    Except.ok.inj (actualShapeStored.symm.trans shapeStored)
  subst actualShape
  have lanesEq : lanes = shapeProductArray concreteShape := lanesStored
  subst lanes
  let laneFunction := fun lane => do
    let coordinates := coordinatesFromOffset concreteShape.toList lane
    let laneStructural := { structural with
      axes := (coordinates.map Int.ofNat).toArray
      slots := grid.indexSlots.zip coordinates.toArray |>.map
        (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
    let lanePath := path.push {
      stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := lane }
    let laneArguments ← gridInputArguments laneStructural stageNumber scope.id index
      grid.inputModes argumentValues
    let childInputs ← checkedChildInputs stageNumber scope.id index child laneArguments
    let childResult ← evalScope data env laneStructural trace stageNumber stage grid.child child
      stageStored childStored childInputs lanePath 0 #[] (fuel - 1)
    let outputs ← child.outputs.mapM (fun output =>
      (match lookup childResult.values output with
      | some value => Except.ok value
      | none => Except.error
          (EvalError.missingPort stageNumber child.id output.node output.port) :
        Except EvalError (DynamicValue backend)))
    pure (outputs, childResult.scopes)
  have sourceBound : selected < (Array.range (shapeProductArray concreteShape)).size := by
    simpa using selectedBound
  obtain ⟨laneResultBound, selectedStored⟩ :=
    array_mapM_getElem laneFunction (by simpa [laneFunction] using laneResultsStored) sourceBound
  obtain ⟨laneArguments, laneArgumentsStored, afterLaneArguments⟩ :=
    equation_bind_eq_ok _ _ _ selectedStored
  obtain ⟨childInputs, childInputsStored, afterChildInputs⟩ :=
    equation_bind_eq_ok _ _ _ afterLaneArguments
  obtain ⟨childResult, childEvaluated, afterChild⟩ :=
    equation_bind_eq_ok _ _ _ afterChildInputs
  obtain ⟨outputs, _, pairStored⟩ := equation_bind_eq_ok _ _ _ afterChild
  have selectedPair : laneResults[selected] = (outputs, childResult.scopes) :=
    (Except.ok.inj pairStored).symm
  have selectedPairOptional : laneResults[selected]? = some (outputs, childResult.scopes) := by
    rw [Array.getElem?_eq_getElem laneResultBound, selectedPair]
  let leading := (laneResults.extract 0 selected).foldl
    (fun result item => result ++ item.2) #[]
  let trailing := (laneResults.extract (selected + 1) laneResults.size).foldl
    (fun result item => result ++ item.2) #[]
  refine ⟨argumentValues, child, childStored, laneResults, packed, nextResult, laneArguments,
    childInputs, childResult, outputs, argumentsStored, ?_, childInputsStored, ?_,
    selectedPairOptional, nextStored, finalStored,
    leading, trailing, ?_, ?_⟩
  · simpa [laneFunction] using laneArgumentsStored
  · simpa [laneFunction] using childEvaluated
  · have factor := array_foldl_append_second_factor laneResults selected laneResultBound
    have selectedScopes : laneResults[selected].2 = childResult.scopes :=
      congrArg Prod.snd selectedPair
    rw [selectedScopes] at factor
    dsimp [leading, trailing]
    exact factor
  · intro snapshot membership
    have foldedMembership : snapshot ∈
        (laneResults.extract (selected + 1) laneResults.size).foldl
          (fun result item => result ++ item.2) #[] := by
      change snapshot ∈
        (laneResults.extract (selected + 1) laneResults.size).foldl
          (fun result item => result ++ item.2) #[] at membership
      exact membership
    rcases array_mem_foldl_append_second
        (laneResults.extract (selected + 1) laneResults.size) #[] snapshot foldedMembership with
      impossible | ⟨suffixIndex, suffixBound, suffixMember⟩
    · simp at impossible
    · let later := selected + 1 + suffixIndex
      have laterBound : later < laneResults.size := by
        simp only [Array.size_extract] at suffixBound
        dsimp [later]
        omega
      have suffixPoint :
          (laneResults.extract (selected + 1) laneResults.size)[suffixIndex] =
            laneResults[later] := by
        simp [later]
      have laterMember : snapshot ∈ laneResults[later].2 := by
        simpa [suffixPoint] using suffixMember
      have rangeLaterBound : later < (Array.range (shapeProductArray concreteShape)).size := by
        have sizeSatisfies := Array.size_mapM laneFunction
          (Array.range (shapeProductArray concreteShape))
        rw [SatisfiesM_Except_eq] at sizeSatisfies
        have sizeEq : laneResults.size =
            (Array.range (shapeProductArray concreteShape)).size :=
          sizeSatisfies laneResults (by simpa [laneFunction] using laneResultsStored)
        simpa [sizeEq] using laterBound
      obtain ⟨_, laterStored⟩ :=
        array_mapM_getElem laneFunction (by simpa [laneFunction] using laneResultsStored)
          rangeLaterBound
      obtain ⟨laterArguments, _, afterLaterArguments⟩ :=
        equation_bind_eq_ok _ _ _ laterStored
      obtain ⟨laterInputs, _, afterLaterInputs⟩ :=
        equation_bind_eq_ok _ _ _ afterLaterArguments
      obtain ⟨laterResult, laterEvaluated, afterLaterChild⟩ :=
        equation_bind_eq_ok _ _ _ afterLaterInputs
      obtain ⟨laterOutputs, _, laterPairStored⟩ :=
        equation_bind_eq_ok _ _ _ afterLaterChild
      have laterPair : laneResults[later] = (laterOutputs, laterResult.scopes) :=
        (Except.ok.inj laterPairStored).symm
      have laterScopeMember : snapshot ∈ laterResult.scopes := by
        simpa [laterPair] using laterMember
      have laterUnder := evalScope_success_scopes_under data env trace stageNumber stage stageStored
        { structural with
          axes := ((coordinatesFromOffset concreteShape.toList later).map Int.ofNat).toArray
          slots := grid.indexSlots.zip
            (coordinatesFromOffset concreteShape.toList later).toArray |>.map
              (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
        grid.child child childStored laterInputs
        (path.push {
          stage := stageNumber, scope := scope.id, owner := index,
          laneOrIteration := later }) 0 #[] (fuel - 1) laterResult
        (by simpa [laneFunction] using laterEvaluated) snapshot laterScopeMember
      exact (OccurrencePath.ne_of_push_under path stageNumber scope.id index selected later
        selectedPath snapshot.occurrence (by dsimp [later]; omega) selectedUnder laterUnder).symm

/- A concrete grid node avoids a previously selected sibling path when every lane frame differs
   from the selected first frame.  The lane child evaluations and their scopes are recovered from
   the grid equation itself. -/
theorem ParallelGridEquation.avoidingScopeStep {backend : SemanticBackend}
    {data : ProgramData} {env : EvalEnv backend data} {structural : StructuralEnv}
    {trace : Trace backend} {stageNumber : Nat} {stage : Stage} {scopeNumber : ScopeId}
    {scope : Scope} {stageStored : data.stages[stageNumber]? = some stage}
    {scopeStored : scopeAt stage scopeNumber = some scope} {inputs : Array (Binding backend)}
    {path : OccurrencePath} {index : Nat} {values : Array (Binding backend)} {fuel : Nat}
    {nodeValue : Node} {grid : GridPayload} {finalResult : ScopeResult backend}
    (equation : ParallelGridEquation data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel nodeValue grid finalResult)
    (selectedFrame : OccurrenceFrame) (selectedPath : OccurrencePath)
    (selectedUnder : OccurrencePath.Under (path.push selectedFrame) selectedPath)
    (frameDifferent : ∀ lane, OccurrenceFrame.mk stageNumber scope.id index lane ≠ selectedFrame)
    (nodeStored : scope.nodes[index]? = some nodeValue) :
    AvoidingScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path selectedPath index values fuel finalResult := by
  obtain ⟨argumentValues, child, childStored, concreteShape, lanes, laneResults, packed, nextResult,
      _, _, lanesStored, laneResultsStored, _, typesMatch, nextStored, finalStored⟩ := equation
  let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
  let nodeResult : NodeResult backend := { values := packed, scopes := laneScopes }
  refine ⟨nodeValue, nodeResult, nextResult, nodeStored, ?_, typesMatch, ?_, ?_⟩
  · intro snapshot membership
    have foldedMember : snapshot ∈
        laneResults.foldl (fun result item => result ++ item.2) #[] := by
      simpa [nodeResult, laneScopes] using membership
    rcases array_mem_foldl_append_second laneResults #[] snapshot foldedMember with
      impossible | ⟨lane, laneBound, laneMember⟩
    · simp at impossible
    · let laneFunction := fun lane => do
        let coordinates := coordinatesFromOffset concreteShape.toList lane
        let laneStructural := { structural with
          axes := (coordinates.map Int.ofNat).toArray
          slots := grid.indexSlots.zip coordinates.toArray |>.map
            (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
        let lanePath := path.push {
          stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := lane }
        let laneArguments ← gridInputArguments laneStructural stageNumber scope.id index
          grid.inputModes argumentValues
        let childInputs ← checkedChildInputs stageNumber scope.id index child laneArguments
        let childResult ← evalScope data env laneStructural trace stageNumber stage grid.child child
          stageStored childStored childInputs lanePath 0 #[] (fuel - 1)
        let outputs ← child.outputs.mapM (fun output =>
          (match lookup childResult.values output with
          | some value => Except.ok value
          | none => Except.error
              (EvalError.missingPort stageNumber child.id output.node output.port) :
            Except EvalError (DynamicValue backend)))
        pure (outputs, childResult.scopes)
      have sourceBound : lane < (Array.range lanes).size := by
        have sizeSatisfies := Array.size_mapM laneFunction (Array.range lanes)
        rw [SatisfiesM_Except_eq] at sizeSatisfies
        have sizeEq : laneResults.size = (Array.range lanes).size :=
          sizeSatisfies laneResults (by simpa [laneFunction] using laneResultsStored)
        simpa [sizeEq] using laneBound
      obtain ⟨_, laneStored⟩ :=
        array_mapM_getElem laneFunction (by simpa [laneFunction] using laneResultsStored)
          sourceBound
      obtain ⟨laneArguments, _, afterArguments⟩ := equation_bind_eq_ok _ _ _ laneStored
      obtain ⟨childInputs, _, afterInputs⟩ := equation_bind_eq_ok _ _ _ afterArguments
      obtain ⟨childResult, childEvaluated, afterChild⟩ :=
        equation_bind_eq_ok _ _ _ afterInputs
      obtain ⟨outputs, _, pairStored⟩ := equation_bind_eq_ok _ _ _ afterChild
      have pairEq : laneResults[lane] = (outputs, childResult.scopes) :=
        (Except.ok.inj pairStored).symm
      have childMember : snapshot ∈ childResult.scopes := by
        simpa [pairEq] using laneMember
      have childUnder := evalScope_success_scopes_under data env trace stageNumber stage stageStored
        { structural with
          axes := ((coordinatesFromOffset concreteShape.toList lane).map Int.ofNat).toArray
          slots := grid.indexSlots.zip
            (coordinatesFromOffset concreteShape.toList lane).toArray |>.map
              (fun item : Nat × Nat => (item.1, Int.ofNat item.2)) }
        grid.child child childStored childInputs
        (path.push {
          stage := stageNumber, scope := scope.id, owner := index,
          laneOrIteration := lane }) 0 #[] (fuel - 1) childResult
        (by simpa [laneFunction] using childEvaluated) snapshot childMember
      exact (OccurrencePath.ne_of_distinct_push_under path
        (OccurrenceFrame.mk stageNumber scope.id index lane) selectedFrame snapshot.occurrence
        selectedPath (frameDifferent lane) childUnder selectedUnder)
  · simpa [nodeResult] using nextStored
  · simpa [nodeResult, laneScopes] using finalStored

/- Every returned loop scope belongs to an iteration executed at or after the starting index.
   Recording only its iteration-root prefix is enough to rule out trace shadowing by later
   iterations. -/
theorem evalSequentialLoop_success_scopes_from_iterations
    {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (childNumber : ScopeId) (child : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (childStored : scopeAt stage childNumber = some child)
    (loop : LoopPayload) (owner : NodeId)
    (arguments : Array (DynamicValue backend)) (structural : StructuralEnv)
    (path : OccurrencePath) (iteration fuel count : Nat)
    (countStored : evalNatExpr structural stageNumber child.id owner loop.count = .ok count)
    (finalResult : NodeResult backend)
    (success : evalSequentialLoop data env trace stageNumber stage childNumber child stageStored
      childStored loop owner arguments structural path iteration fuel = .ok finalResult) :
    ∀ snapshot ∈ finalResult.scopes,
      ∃ actual, iteration ≤ actual ∧ actual < count ∧
        OccurrencePath.Under
          (path.push {
            stage := stageNumber, scope := child.id, owner := owner,
            laneOrIteration := actual })
          snapshot.occurrence := by
  induction fuel generalizing arguments iteration finalResult with
  | zero =>
      simp [evalSequentialLoop] at success
  | succ fuel ih =>
      intro snapshot membership
      by_cases iterationBound : iteration < count
      · obtain ⟨childInputs, childResult, childValues, rest, _, childEvaluated, _,
            _, restStored, finalStored⟩ :=
          evalSequentialLoop_success_iteration_step data env trace stageNumber stage childNumber
            child stageStored childStored loop owner arguments structural path iteration
            (fuel + 1) count (by omega) countStored iterationBound finalResult success
        have scopesEq :
            finalResult.scopes = childResult.scopes ++ rest.scopes :=
          congrArg NodeResult.scopes finalStored
        rw [scopesEq, Array.mem_append] at membership
        rcases membership with childMember | restMember
        · refine ⟨iteration, Nat.le_refl _, iterationBound, ?_⟩
          exact evalScope_success_scopes_under data env trace stageNumber stage stageStored
            { structural with
              slots := structural.slots.push (loop.indexSlot, Int.ofNat iteration) }
            childNumber child childStored childInputs
            (path.push {
              stage := stageNumber, scope := child.id, owner := owner,
              laneOrIteration := iteration })
            0 #[] fuel childResult (by simpa using childEvaluated) snapshot childMember
        · obtain ⟨actual, later, actualBound, under⟩ :=
            ih (arguments := childValues ++ arguments.extract loop.carriedCount arguments.size)
              (iteration := iteration + 1) (finalResult := rest) (by simpa using restStored)
              snapshot restMember
          exact ⟨actual, by omega, actualBound, under⟩
      · rw [evalSequentialLoop] at success
        simp only [if_neg (by omega : fuel + 1 ≠ 0)] at success
        obtain ⟨actualCount, actualCountStored, afterCount⟩ :=
          equation_bind_eq_ok _ _ _ success
        have actualCountEq : actualCount = count :=
          Except.ok.inj (actualCountStored.symm.trans countStored)
        subst actualCount
        split at afterCount
        · contradiction
        · have resultEq : finalResult = {
              values := arguments.extract 0 loop.carriedCount, scopes := #[] } :=
            (Except.ok.inj afterCount).symm
          subst finalResult
          simp at membership

theorem SequentialLoopEquation.avoidingScopeStep {backend : SemanticBackend}
    {data : ProgramData} {env : EvalEnv backend data} {structural : StructuralEnv}
    {trace : Trace backend} {stageNumber : Nat} {stage : Stage} {scopeNumber : ScopeId}
    {scope : Scope} {stageStored : data.stages[stageNumber]? = some stage}
    {scopeStored : scopeAt stage scopeNumber = some scope} {inputs : Array (Binding backend)}
    {path : OccurrencePath} {index : Nat} {values : Array (Binding backend)} {fuel : Nat}
    {nodeValue : Node} {loop : LoopPayload} {finalResult : ScopeResult backend}
    (equation : SequentialLoopEquation data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel nodeValue loop finalResult)
    (selectedFrame : OccurrenceFrame) (selectedPath : OccurrencePath)
    (selectedUnder : OccurrencePath.Under (path.push selectedFrame) selectedPath)
    (frameDifferent : ∀ childId iteration,
      OccurrenceFrame.mk stageNumber childId index iteration ≠ selectedFrame)
    (nodeStored : scope.nodes[index]? = some nodeValue) :
    AvoidingScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path selectedPath index values fuel finalResult := by
  obtain ⟨argumentValues, child, childStored, loopResult, nextResult, _, loopStored, typesMatch,
      nextStored, finalStored⟩ := equation
  have loopFuelPositive : fuel - 1 ≠ 0 := by
    intro fuelZero
    rw [evalSequentialLoop] at loopStored
    simp [fuelZero] at loopStored
  have loopUnfolded := loopStored
  rw [evalSequentialLoop] at loopUnfolded
  simp only [if_neg loopFuelPositive] at loopUnfolded
  obtain ⟨count, countStored, _⟩ := equation_bind_eq_ok _ _ _ loopUnfolded
  refine ⟨nodeValue, loopResult, nextResult, nodeStored, ?_, typesMatch, nextStored, finalStored⟩
  intro snapshot membership
  obtain ⟨actual, _, _, actualUnder⟩ :=
    evalSequentialLoop_success_scopes_from_iterations data env trace stageNumber stage loop.child
      child stageStored childStored loop index argumentValues structural path 0 (fuel - 1) count
      countStored loopResult loopStored snapshot membership
  exact OccurrencePath.ne_of_distinct_push_under path
    (OccurrenceFrame.mk stageNumber child.id index actual) selectedFrame snapshot.occurrence
    selectedPath (frameDifferent child.id actual) actualUnder selectedUnder

/- Selecting an iteration exposes its exact child evaluation and a factorization of all loop
   scopes.  The final clause is the lookup-critical fact: every trailing snapshot came from a
   strictly later iteration and therefore has a different occurrence path. -/
theorem evalSequentialLoop_success_child_at_with_trailing_miss
    {backend : SemanticBackend} (data : ProgramData) (env : EvalEnv backend data)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (childNumber : ScopeId) (child : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (childStored : scopeAt stage childNumber = some child)
    (loop : LoopPayload) (owner : NodeId)
    (arguments : Array (DynamicValue backend)) (structural : StructuralEnv)
    (path selectedPath : OccurrencePath) (iteration fuel count offset : Nat)
    (countStored : evalNatExpr structural stageNumber child.id owner loop.count = .ok count)
    (requestedBound : iteration + offset < count)
    (selectedUnder : OccurrencePath.Under
      (path.push {
        stage := stageNumber, scope := child.id, owner := owner,
        laneOrIteration := iteration + offset })
      selectedPath)
    (finalResult : NodeResult backend)
    (success : evalSequentialLoop data env trace stageNumber stage childNumber child stageStored
      childStored loop owner arguments structural path iteration fuel = .ok finalResult) :
    ∃ reachedArguments reachedFuel reachedLoopResult childInputs childResult,
      evalSequentialLoop data env trace stageNumber stage childNumber child stageStored childStored
        loop owner reachedArguments structural path (iteration + offset) reachedFuel =
          .ok reachedLoopResult ∧
      checkedChildInputs stageNumber child.id owner child reachedArguments = .ok childInputs ∧
      evalScope data env
        { structural with
          slots := structural.slots.push (loop.indexSlot, Int.ofNat (iteration + offset)) }
        trace stageNumber stage childNumber child stageStored childStored childInputs
        (path.push {
          stage := stageNumber, scope := child.id, owner := owner,
          laneOrIteration := iteration + offset }) 0 #[] (reachedFuel - 1) = .ok childResult ∧
      ∃ leading trailing,
        finalResult.scopes = leading ++ childResult.scopes ++ trailing ∧
        ∀ snapshot ∈ trailing, snapshot.occurrence ≠ selectedPath := by
  induction offset generalizing arguments iteration fuel finalResult with
  | zero =>
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalSequentialLoop] at success
        simp [fuelZero] at success
      obtain ⟨childInputs, childResult, childValues, rest, inputsStored, childStored', _,
          _, restStored, finalStored⟩ :=
        evalSequentialLoop_success_iteration_step data env trace stageNumber stage childNumber child
          stageStored childStored loop owner arguments structural path iteration fuel count
          fuelPositive countStored (by simpa using requestedBound) finalResult success
      refine ⟨arguments, fuel, finalResult, childInputs, childResult, ?_, inputsStored, ?_,
        #[], rest.scopes, ?_, ?_⟩
      · simpa using success
      · simpa using childStored'
      · simpa using congrArg NodeResult.scopes finalStored
      · intro snapshot membership
        obtain ⟨actual, later, actualBound, laterUnder⟩ :=
          evalSequentialLoop_success_scopes_from_iterations data env trace stageNumber stage
            childNumber child stageStored childStored loop owner
            (childValues ++ arguments.extract loop.carriedCount arguments.size) structural path
            (iteration + 1) (fuel - 1) count countStored rest restStored snapshot membership
        exact (OccurrencePath.ne_of_push_under path stageNumber child.id owner iteration actual
          selectedPath snapshot.occurrence (by omega) (by simpa using selectedUnder)
          laterUnder).symm
  | succ offset ih =>
      have currentBound : iteration < count := by omega
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalSequentialLoop] at success
        simp [fuelZero] at success
      obtain ⟨childInputs, childResult, childValues, rest, _, _, _, _, restStored, finalStored⟩ :=
        evalSequentialLoop_success_iteration_step data env trace stageNumber stage childNumber child
          stageStored childStored loop owner arguments structural path iteration fuel count
          fuelPositive countStored currentBound finalResult success
      let nextArguments := childValues ++ arguments.extract loop.carriedCount arguments.size
      have selectedUnder' : OccurrencePath.Under
          (path.push {
            stage := stageNumber, scope := child.id, owner := owner,
            laneOrIteration := (iteration + 1) + offset }) selectedPath := by
        simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using selectedUnder
      obtain ⟨reachedArguments, reachedFuel, reachedLoopResult, reachedInputs, reachedResult,
          reachedLoop, reachedInputsStored, reachedChildStored, leading, trailing, reachedFactor,
          trailingMiss⟩ :=
        ih (arguments := nextArguments) (iteration := iteration + 1) (fuel := fuel - 1)
          (finalResult := rest) (by omega) selectedUnder' restStored
      refine ⟨reachedArguments, reachedFuel, reachedLoopResult, reachedInputs, reachedResult, ?_,
        reachedInputsStored, ?_, childResult.scopes ++ leading, trailing, ?_, trailingMiss⟩
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using reachedLoop
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using reachedChildStored
      · have outerFactor := congrArg NodeResult.scopes finalStored
        rw [outerFactor, reachedFactor]
        simp [Array.append_assoc]

theorem reachedPrimitiveRunFromScopeFactor {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data)
    (finalTrace evalTrace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (stageStored : data.stages[stageNumber]? = some stage)
    (stageTrace : StageTrace backend)
    (stageFind : finalTrace.stages.find? (fun item => item.stage = stageNumber) = some stageTrace)
    (structural : StructuralEnv) (scopeNumber : ScopeId) (scope : Scope)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (rootFuel target : Nat) (targetBound : target + 1 ≤ scope.nodes.size)
    (step : ∀ (limit index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (finalResult : ScopeResult backend), limit = target → index < limit → fuel ≠ 0 →
      evalScope data env structural evalTrace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel = .ok finalResult →
      FlatScopeStep data env structural evalTrace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env structural evalTrace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 #[] rootFuel = .ok rootResult)
    (outerLeading trailing : Array (ScopeTrace backend))
    (stageFactor : stageTrace.scopes = outerLeading ++ rootResult.scopes ++ trailing)
    (trailingMiss : ∀ snapshot ∈ trailing, snapshot.occurrence ≠ path)
    (storedNode : Node) (nodeStored : scope.nodes[target]? = some storedNode)
    (payload : NodePayload) (payloadStored : storedNode.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload) (port : Nat)
    (portBound : port < storedNode.outputs.size) :
    Nonempty (ReachedPrimitiveRun finalTrace structural stageNumber scope.id target path payload
      storedNode port) := by
  obtain ⟨values, fuel, currentResult, nodeResult, nextResult, arguments, _currentSuccess,
      argumentsStored, valuesCovered, primitiveStored, leadingScopes, earlierSnapshots, producer,
      output, scopesFactor, earlierBefore, currentFactor, producerEq, outputStored, lookupStored⟩ :=
    generatedPrimitiveOutputAtWithBindingsBefore data env structural evalTrace stageNumber stage
      scopeNumber scope stageStored scopeStored inputs path #[] rootFuel target targetBound
      (bindingsBefore_empty scope.id 0) step rootResult rootSuccess storedNode nodeStored payload
      payloadStored primitivePayload port portBound
  let outputOccurrence : WireOccurrence :=
    occurrenceOf stageNumber path { scope := scope.id, node := target, port := port }
  have earlierMiss :
      ∀ snapshot ∈ earlierSnapshots, scopeTraceContains outputOccurrence snapshot = false := by
    intro snapshot membership
    have missing := (earlierBefore snapshot membership).lookup_target_none port
    dsimp [outputOccurrence, occurrenceOf]
    simp [scopeTraceContains, missing]
  have producerMatch : scopeTraceContains outputOccurrence producer = true := by
    have concreteLookup := lookupStored
    rw [producerEq] at concreteLookup
    rw [producerEq]
    dsimp [outputOccurrence, occurrenceOf]
    simp [scopeTraceContains, concreteLookup]
  have localOutputFind :
      rootResult.scopes.reverse.find? (scopeTraceContains outputOccurrence) = some producer := by
    exact reverseFindProducerOfFactorization (scopeTraceContains outputOccurrence)
      rootResult.scopes leadingScopes currentResult.scopes earlierSnapshots nodeResult.scopes
      nextResult.scopes producer scopesFactor currentFactor earlierMiss producerMatch
  have trailingOutputMiss :
      ∀ snapshot ∈ trailing, scopeTraceContains outputOccurrence snapshot = false := by
    intro snapshot membership
    exact scopeTraceContains_false_of_path_ne outputOccurrence snapshot
      (trailingMiss snapshot membership)
  have globalOutputFind :
      stageTrace.scopes.reverse.find? (scopeTraceContains outputOccurrence) = some producer :=
    reverseFindOfMiddleFactor (scopeTraceContains outputOccurrence) stageTrace.scopes outerLeading
      rootResult.scopes trailing producer stageFactor trailingOutputMiss localOutputFind
  have valuesTraced : ∀ binding ∈ values,
      traceValueAt finalTrace (occurrenceOf stageNumber path binding.wire) = some binding.value := by
    intro binding membership
    let occurrence := occurrenceOf stageNumber path binding.wire
    have covered := valuesCovered binding membership
    cases localFound : rootResult.scopes.reverse.find? (scopeTraceContains occurrence) with
    | none =>
        have localFoundZero :
            rootResult.scopes.reverse.find?
              (scopeTraceContains { stage := 0, path := path, wire := binding.wire }) = none := by
          simpa [occurrence, occurrenceOf, scopeTraceContains] using localFound
        simp [localFoundZero] at covered
    | some localProducer =>
        have localFoundZero :
            rootResult.scopes.reverse.find?
              (scopeTraceContains { stage := 0, path := path, wire := binding.wire }) =
                some localProducer := by
          simpa [occurrence, occurrenceOf, scopeTraceContains] using localFound
        have trailingValueMiss :
            ∀ snapshot ∈ trailing, scopeTraceContains occurrence snapshot = false := by
          intro snapshot snapshotMember
          exact scopeTraceContains_false_of_path_ne occurrence snapshot
            (trailingMiss snapshot snapshotMember)
        have globalFound :
            stageTrace.scopes.reverse.find? (scopeTraceContains occurrence) = some localProducer :=
          reverseFindOfMiddleFactor (scopeTraceContains occurrence) stageTrace.scopes outerLeading
            rootResult.scopes trailing localProducer stageFactor trailingValueMiss localFound
        have globalFoundConcrete :
            stageTrace.scopes.reverse.find?
              (scopeTraceContains
                { stage := stageNumber, path := path, wire := binding.wire }) =
              some localProducer := by
          simpa [occurrence, occurrenceOf, scopeTraceContains] using globalFound
        dsimp [traceValueAt, occurrence, occurrenceOf]
        rw [stageFind]
        simp only [Option.bind_some]
        rw [globalFoundConcrete]
        simpa [localFoundZero] using covered
  refine ⟨{
    nodeResult := nodeResult
    output := output
    arguments := arguments
    values := values
    argumentsResolved := argumentsStored
    payloadStored := payloadStored
    valuesTraced := valuesTraced
    primitiveEvaluated := primitiveStored
    outputStored := outputStored
    outputTraced := ?_
  }⟩
  have globalOutputFindConcrete :
      stageTrace.scopes.reverse.find?
        (scopeTraceContains {
          stage := stageNumber, path := path,
          wire := { scope := scope.id, node := target, port := port } }) = some producer := by
    simpa [outputOccurrence, occurrenceOf, scopeTraceContains] using globalOutputFind
  dsimp [traceValueAt, outputOccurrence, occurrenceOf]
  rw [stageFind]
  simp only [Option.bind_some]
  rw [globalOutputFindConcrete]
  exact lookupStored

end Mxx.IR
