import MxxIrCore.Eval

namespace Mxx
namespace IR

/- A first-match prefix is the evaluator invariant needed when a scope trace is queried through
   `Array.find?`: the selected binding exists and no earlier binding has the same wire. -/
def FirstMatchPrefix {backend : SemanticBackend} (values : Array (Binding backend))
    (wire : WireRef) (target : Binding backend) : Prop :=
  ∃ index, ∃ indexBound : index < values.size,
    values[index] = target ∧ ∀ earlier, ∀ earlierBound : earlier < values.size,
      earlier < index → values[earlier].wire ≠ wire

theorem lookup_eq_some_of_firstMatchPrefix {backend : SemanticBackend}
    {values : Array (Binding backend)} {wire : WireRef} {value : DynamicValue backend}
    {target : Binding backend} (target_eq : target = { wire := wire, value := value })
    (frontier : FirstMatchPrefix values wire target) : lookup values wire = some value := by
  unfold lookup
  rcases frontier with ⟨index, indexBound, stored, earlier⟩
  have found : values.find? (fun binding => binding.wire = wire) = some target := by
    apply (Array.find?_eq_some_iff_getElem).mpr
    refine ⟨?_, index, indexBound, stored, ?_⟩
    · rw [target_eq]
      simp
    · intro other otherBound
      have hfalse : decide (values[other].wire = wire) = false :=
        decide_eq_false_iff_not.mpr
          (earlier other (lt_trans otherBound indexBound) otherBound)
      simp [hfalse]
  rw [found, target_eq]
  rfl

theorem lookup_eq_some_of_mem_unique {backend : SemanticBackend}
    {values : Array (Binding backend)} {wire : WireRef} {value : DynamicValue backend}
    (target : Binding backend) (target_eq : target = { wire := wire, value := value })
    (mem : target ∈ values)
    (unique : ∀ binding ∈ values, binding.wire = wire → binding = target) :
    lookup values wire = some value := by
  unfold lookup
  have not_none : values.find? (fun binding => binding.wire = wire) ≠ none := by
    intro none
    have no_match := (Array.find?_eq_none).mp none
    have target_match : target.wire = wire := by
      rw [target_eq]
    have contradiction := no_match target mem
    simp [target_match] at contradiction
  cases found : values.find? (fun binding => binding.wire = wire) with
  | none => exact False.elim (not_none found)
  | some binding =>
      have foundData := (Array.find?_eq_some_iff_getElem).mp found
      have foundMem : binding ∈ values := by
        rcases foundData.2 with ⟨index, indexBound, stored, _⟩
        exact Array.mem_iff_getElem.mpr ⟨index, indexBound, stored⟩
      have foundWire : binding.wire = wire := by
        exact of_decide_eq_true foundData.1
      have bindingEq := unique binding foundMem foundWire
      have valueEq : binding.value = value := by
        rw [bindingEq, target_eq]
      simp [valueEq]

theorem appendNodeBindings_mem_output {backend : SemanticBackend}
    {scope index port : Nat} {values : Array (Binding backend)}
    {result : Array (DynamicValue backend)} {value : DynamicValue backend}
    (portBound : port < result.size) (stored : result[port]? = some value) :
    { wire := { scope := scope, node := index, port := port }, value := value } ∈
      appendNodeBindings scope index values result := by
  unfold appendNodeBindings
  let target : Binding backend := {
    wire := { scope := scope, node := index, port := port }, value := value }
  have go : ∀ (entries : List Nat) (acc : Array (Binding backend)),
      target ∈ acc ∨ port ∈ entries → target ∈ entries.foldl (fun accumulated entry =>
        match result[entry]? with
        | some entryValue => accumulated.push {
            wire := { scope := scope, node := index, port := entry }, value := entryValue }
        | none => accumulated) acc := by
    intro entries
    induction entries with
    | nil => simp
    | cons entry rest ih =>
        intro acc membership
        simp only [List.foldl_cons]
        simp only [List.mem_cons] at membership
        by_cases same : entry = port
        · subst entry
          convert ih (acc.push target) (Or.inl Array.mem_push_self) using 1
          simp [target, stored]
        · have carried : target ∈
              (match result[entry]? with
              | some entryValue => acc.push {
                  wire := { scope := scope, node := index, port := entry }, value := entryValue }
              | none => acc) ∨ port ∈ rest := by
            rcases membership with existing | entryOrTail
            · cases entryValue : result[entry]? with
              | none => exact Or.inl existing
              | some sampled =>
                  exact Or.inl (Array.mem_push.mpr (Or.inl existing))
            · exact entryOrTail.elim (fun equality => False.elim (same equality.symm)) Or.inr
          exact ih _ carried
  have rangeMembership : port ∈ List.range result.size := List.mem_range.mpr portBound
  exact go (List.range result.size) values (Or.inr rangeMembership)

/- Bindings already present when node `index` is evaluated belong to this scope and to a strictly
   earlier node.  This is the small SSA fact needed to distinguish an appended output from any
   prior binding with the same port. -/
def BindingsBefore {backend : SemanticBackend} (scope index : Nat)
    (values : Array (Binding backend)) : Prop :=
  ∀ binding ∈ values, binding.wire.scope = scope ∧ binding.wire.node < index

/- Every accumulated SSA binding is recoverable from the evaluator's immutable scope snapshots.
   The stage number is irrelevant inside one stage: `scopeTraceContains` inspects only the path and
   wire, while the public bridge later reinstates the concrete stage selected by `evalStages`. -/
def ScopeTracesCoverBindings {backend : SemanticBackend} (scopes : Array (ScopeTrace backend))
    (path : OccurrencePath) (values : Array (Binding backend)) : Prop :=
  ∀ binding ∈ values,
    (scopes.reverse.find? (scopeTraceContains
      { stage := 0, path := path, wire := binding.wire })).bind
        (fun snapshot => lookup snapshot.values binding.wire) = some binding.value

theorem bindingsBefore_empty {backend : SemanticBackend} (scope index : Nat) :
    BindingsBefore scope index (#[] : Array (Binding backend)) := by
  intro binding membership
  simp at membership

theorem scopeTracesCoverBindings_empty {backend : SemanticBackend}
    (scopes : Array (ScopeTrace backend)) (path : OccurrencePath) :
    ScopeTracesCoverBindings scopes path (#[] : Array (Binding backend)) := by
  intro binding membership
  simp at membership

theorem BindingsBefore.mono {backend : SemanticBackend}
    {scope first later : Nat} {values : Array (Binding backend)}
    (before : BindingsBefore scope first values) (le : first ≤ later) :
    BindingsBefore scope later values := by
  intro binding membership
  obtain ⟨scopeEq, nodeLt⟩ := before binding membership
  exact ⟨scopeEq, lt_of_lt_of_le nodeLt le⟩

theorem BindingsBefore.lookup_target_none {backend : SemanticBackend}
    {scope target : Nat} {values : Array (Binding backend)}
    (before : BindingsBefore scope target values) (port : Nat) :
    lookup values { scope := scope, node := target, port := port } = none := by
  unfold lookup
  apply Option.map_eq_none_iff.mpr
  apply Array.find?_eq_none.mpr
  intro binding membership matching
  have wireEq : binding.wire = { scope := scope, node := target, port := port } :=
    of_decide_eq_true matching
  have nodeLt := (before binding membership).2
  have nodeEq : binding.wire.node = target := congrArg WireRef.node wireEq
  have impossible : target < target := by simpa [nodeEq] using nodeLt
  exact (Nat.lt_irrefl target) impossible

theorem outputTypesMatch_size {backend : SemanticBackend}
    (expected : List WireType) (actual : List (DynamicValue backend))
    (matching : outputTypesMatch expected actual = true) : expected.length = actual.length := by
  induction expected generalizing actual with
  | nil => cases actual <;> simp_all [outputTypesMatch]
  | cons expected rest ih =>
      cases actual with
      | nil => simp [outputTypesMatch] at matching
      | cons actual restActual =>
          simp only [outputTypesMatch, Bool.and_eq_true] at matching
          simp [ih restActual matching.2]

private theorem foldl_bindings_mem_cases {backend : SemanticBackend}
    {scope index : Nat}
    {result : Array (DynamicValue backend)} :
    ∀ (entries : List Nat), (∀ entry ∈ entries, entry < result.size) →
      ∀ (values : Array (Binding backend)) (binding : Binding backend),
      binding ∈ entries.foldl (fun accumulated entry =>
        match result[entry]? with
        | some value => accumulated.push {
            wire := { scope := scope, node := index, port := entry }
            value := value }
        | none => accumulated) values →
      binding ∈ values ∨ ∃ port, port < result.size ∧ result[port]? = some binding.value ∧
        binding.wire = { scope := scope, node := index, port := port } := by
  intro entries
  induction entries with
  | nil =>
      intro _
      intro values binding membership
      exact Or.inl membership
  | cons entry rest ih =>
      intro entriesBound
      intro values binding membership
      simp only [List.foldl_cons] at membership
      by_cases absent : result[entry]? = none
      · exact ih (by
          intro entry' membership'
          exact entriesBound entry' (by simp [membership'])) values binding
          (by simpa [absent] using membership)
      · obtain ⟨entryValue, entryStored⟩ := Option.ne_none_iff_exists'.mp absent
        have restMembership : binding ∈ rest.foldl (fun accumulated entry =>
            match result[entry]? with
            | some value => accumulated.push {
                wire := { scope := scope, node := index, port := entry }
                value := value }
            | none => accumulated) (values.push {
            wire := { scope := scope, node := index, port := entry }
            value := entryValue }) := by
          simpa [entryStored] using membership
        obtain old | ⟨port, portBound, portStored, wireEq⟩ := ih (by
            intro entry' membership'
            exact entriesBound entry' (by simp [membership']))
            (values.push {
              wire := { scope := scope, node := index, port := entry }
              value := entryValue }) binding restMembership
        · simp only [Array.mem_push] at old
          rcases old with old | old
          · exact Or.inl old
          · right
            refine ⟨entry, entriesBound entry (by simp), ?_, ?_⟩
            · subst binding
              exact entryStored
            · simpa [old]
        · exact Or.inr ⟨port, portBound, portStored, wireEq⟩

theorem bindingsBefore_appendNodeBindings {backend : SemanticBackend}
    {scope index : Nat} {values : Array (Binding backend)}
    {result : Array (DynamicValue backend)}
    (before : BindingsBefore scope index values) :
    BindingsBefore scope (index + 1) (appendNodeBindings scope index values result) := by
  intro binding membership
  obtain old | ⟨port, portBound, stored, wireEq⟩ :=
    foldl_bindings_mem_cases (List.range result.size)
      (by intro entry entryMem; exact List.mem_range.mp entryMem)
      values binding (by simpa [appendNodeBindings] using membership)
  · rcases before binding old with ⟨scopeEq, nodeEq⟩
    exact ⟨scopeEq, by
      have successor := Nat.lt_succ_of_lt nodeEq
      simpa [Nat.succ_eq_add_one] using successor⟩
  · rw [wireEq]
    exact ⟨rfl, by simpa [Nat.succ_eq_add_one] using Nat.lt_succ_self index⟩

theorem appendNodeBindings_lookup_output {backend : SemanticBackend}
    {scope index port : Nat} {values : Array (Binding backend)}
    {result : Array (DynamicValue backend)} {value : DynamicValue backend}
    (before : BindingsBefore scope index values) (portBound : port < result.size)
    (stored : result[port]? = some value) :
    lookup (appendNodeBindings scope index values result)
      { scope := scope, node := index, port := port } = some value := by
  apply lookup_eq_some_of_mem_unique
    (target := { wire := { scope := scope, node := index, port := port }, value := value }) rfl
  · exact appendNodeBindings_mem_output portBound stored
  · intro binding membership wireEq
    obtain old | ⟨otherPort, otherBound, otherStored, otherWireEq⟩ :=
      foldl_bindings_mem_cases (List.range result.size)
        (by intro entry entryMem; exact List.mem_range.mp entryMem)
        values binding (by simpa [appendNodeBindings] using membership)
    · have nodeBound := (before binding old).2
      rw [wireEq] at nodeBound
      exact False.elim (Nat.lt_irrefl index nodeBound)
    · have portEq : otherPort = port := by
        simpa [otherWireEq] using wireEq
      subst otherPort
      have valueEq : binding.value = value := by
        exact Option.some.inj (otherStored.symm.trans stored)
      cases binding with
      | mk bindingWire bindingValue =>
          simp only at otherWireEq wireEq ⊢
          subst bindingWire
          simpa [valueEq]

/-! The evaluator records the bindings produced by a node in the current scope trace.  This small
    lemma exposes that trace entry from the final equation without introducing a second evaluator.
    The result is a membership fact rather than a `find?` equality: selecting the first matching
    entry requires the separate, ordinary no-duplicate-binding invariant of the caller. -/
theorem currentScopeTrace_mem_of_finalEquation {backend : SemanticBackend}
    {scope index : Nat} {path : OccurrencePath} {values : Array (Binding backend)}
    {priorScopes : Array (ScopeTrace backend)} {resultValues : Array (DynamicValue backend)}
    {nextResult finalResult : ScopeResult backend}
    (equation : finalResult = {
      values := nextResult.values
      scopes := priorScopes ++ nextResult.scopes ++ #[{
        scope := scope
        occurrence := path
        values := appendNodeBindings scope index values resultValues }] }) :
    { scope := scope
      occurrence := path
      values := appendNodeBindings scope index values resultValues } ∈ finalResult.scopes := by
  subst finalResult
  simp

/- The generated gadget equation hook: once renderer-provided node witnesses identify the
   canonical target/layout and the exact constant gadget, this lemma exposes the certificate on
   the same backend call used by Eval. -/
theorem gadgetDecompose_node_certificate {backend : SemanticBackend}
    (targetType : MatrixType) (layout : GadgetLayout) (structural : StructuralEnv)
    (gadget : backend.denoteMatrix (gadgetMatrixType targetType layout))
    (target : backend.denoteMatrix targetType)
    (result : Σ preimage : backend.denotePreimage (gadgetPreimageType targetType layout),
      PLift (backend.gadgetCertificate targetType layout structural gadget target preimage))
    (success : backend.gadgetDecompose targetType layout structural gadget target = .ok result) :
    backend.gadgetCertificate targetType layout structural gadget target result.1 :=
  gadgetDecompose_success_certificate targetType layout structural gadget target result success

/-! A renderer-facing wrapper retaining the exact values used by the backend call.  The
    evaluator inversion is intentionally kept at the call boundary: generated code supplies the
    evaluated layout, canonical gadget, target, and returned witness, while this lemma exposes the
    non-tautological backend certificate. -/
theorem gadgetDecompose_backend_certificate {backend : SemanticBackend}
    (targetType : MatrixType) (layout : GadgetLayout) (structural : StructuralEnv)
    (gadget : backend.denoteMatrix (gadgetMatrixType targetType layout))
    (target : backend.denoteMatrix targetType)
    (result : Σ preimage : backend.denotePreimage (gadgetPreimageType targetType layout),
      PLift (backend.gadgetCertificate targetType layout structural gadget target preimage))
    (success : backend.gadgetDecompose targetType layout structural gadget target = .ok result) :
    ∃ preimage, result.1 = preimage ∧
      backend.gadgetCertificate targetType layout structural gadget target preimage := by
  exact ⟨result.1, rfl, gadgetDecompose_node_certificate targetType layout structural gadget target result success⟩

theorem equation_bind_eq_ok {ε α β : Type} (value : Except ε α)
    (next : α → Except ε β) (result : β)
    (success : value >>= next = .ok result) :
    ∃ input, value = .ok input ∧ next input = .ok result := by
  cases value with
  | error error => cases success
  | ok input => exact ⟨input, rfl, success⟩

/-! The renderer uses these equations to expose one concrete evaluator step.

    The equations are deliberately just wrappers around the evaluator inversion lemmas.  They do
    not introduce an alternative evaluator or allow a caller to provide an output equation as a
    premise.  A generated theorem supplies the concrete node and its stored proofs; the only
    operational premise is that the actual `evalScope` call returned successfully.
-/

theorem generatedPrimitiveNodeEquation {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (payload : NodePayload)
    (payloadStored : nodeValue.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues result nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalPrimitiveNode backend structural stageNumber scope.id index payload
          argumentValues nodeValue.outputs = .ok result ∧
      outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := result.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values result.values }] } := by
  exact evalScope_success_primitive_step data env structural trace stageNumber stage scopeNumber scope
    stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound nodeValue
    nodeStored payload payloadStored primitivePayload success

/-! This is the direct trace-facing form of `generatedPrimitiveNodeEquation`.  It returns the
    exact node result and proves that the scope trace carrying those output bindings is present in
    the accumulated result.  A caller that needs a `lookup` equality can combine the returned
    membership with `lookup_eq_some_of_mem_unique`; the uniqueness condition remains explicit at
    that boundary because `lookup` intentionally uses first-match semantics. -/
theorem generatedPrimitiveNodeEquation_scopeTrace_mem {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (payload : NodePayload)
    (payloadStored : nodeValue.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues result nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalPrimitiveNode backend structural stageNumber scope.id index payload
          argumentValues nodeValue.outputs = .ok result ∧
      outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
        .ok nextResult ∧
      { scope := scope.id
        occurrence := path
        values := appendNodeBindings scope.id index values result.values } ∈ finalResult.scopes := by
  obtain ⟨argumentValues, result, nextResult, argumentsStored, primitiveStored, typesMatch,
      nextStored, finalStored⟩ := generatedPrimitiveNodeEquation data env structural trace stageNumber
    stage scopeNumber scope stageStored scopeStored inputs path index values fuel finalResult
    fuelPositive indexBound nodeValue nodeStored payload payloadStored primitivePayload success
  exact ⟨argumentValues, result, nextResult, argumentsStored, primitiveStored, typesMatch, nextStored,
    currentScopeTrace_mem_of_finalEquation finalStored⟩

theorem evalScope_success_input_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (inputIndex : Nat)
    (payloadStored : nodeValue.payload = .input inputIndex)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues result nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      ((∃ binding, inputs[inputIndex]? = some binding ∧ result = NodeResult.ofValues #[binding.value]) ∨
        (inputs[inputIndex]? = none ∧
          ∃ value, envInput env stageNumber scope.id index path
            { scope := scopeNumber, node := index, port := 0 } = .ok value ∧
            result = NodeResult.ofValues #[value])) ∧
      outputTypesMatch nodeValue.outputs.toList result.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := result.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values result.values }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = .input inputIndex at payloadStored
  subst nodePayload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode =
        { payload := .input inputIndex, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ :=
      equation_bind_eq_ok _ _ _ success
    simp only [samplerPayload, Bool.false_eq_true] at afterArguments
    split at afterArguments
    · rename_i binding bindingStored
      obtain ⟨result, resultStored, afterResult⟩ := equation_bind_eq_ok _ _ _ afterArguments
      cases resultStored
      split at afterResult
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ := equation_bind_eq_ok _ _ _ afterResult
        cases finalStored
        exact ⟨argumentValues, _, nextResult, argumentsStored,
          Or.inl ⟨binding, bindingStored, rfl⟩, typesMatch, nextStored, rfl⟩
      · contradiction

    · rename_i noBinding
      obtain ⟨value, valueStored, afterValue⟩ := equation_bind_eq_ok _ _ _ afterArguments
      obtain ⟨result, resultStored, afterResult⟩ := equation_bind_eq_ok _ _ _ afterValue
      cases resultStored
      split at afterResult
      · rename_i typesMatch
        obtain ⟨nextResult, nextStored, finalStored⟩ := equation_bind_eq_ok _ _ _ afterResult
        cases finalStored
        exact ⟨argumentValues, _, nextResult, argumentsStored,
          Or.inr ⟨noBinding, value, valueStored, rfl⟩, typesMatch, nextStored, rfl⟩
      · contradiction

theorem evalScope_success_sampler_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (payload : NodePayload)
    (payloadStored : nodeValue.payload = payload) (isSampler : samplerPayload payload = true)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues sampled nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      nodeValue.outputs.mapIdxM (fun port _ =>
        if hPort : port < nodeValue.outputs.size then
          let outputType := nodeValue.outputs[port]'hPort
          have hOutput : nodeValue.outputs[port]? = some outputType := by
            rw [Array.getElem?_eq_getElem]
          envSample env stageNumber stage scope index nodeValue path
            { scope := scopeNumber, node := index, port := port } outputType
            stageStored scopeStored nodeStored hPort hOutput
            (by simpa [payloadStored] using isSampler)
        else throw (.missingPort stageNumber scope.id index port)) = .ok sampled ∧
      outputTypesMatch nodeValue.outputs.toList
        (NodeResult.ofValues sampled).values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values sampled) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values sampled }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = payload at payloadStored
  subst payload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode =
        { payload := nodePayload, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    cases nodePayload <;> simp only [samplerPayload, Bool.false_eq_true] at isSampler
    all_goals
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ :=
      equation_bind_eq_ok _ _ _ success
    obtain ⟨sampled, sampledStored, afterSampled⟩ :=
      equation_bind_eq_ok _ _ _ afterArguments
    obtain ⟨result, resultStored, afterResult⟩ :=
      equation_bind_eq_ok _ _ _ afterSampled
    cases resultStored
    split at afterResult
    · rename_i typesMatch
      obtain ⟨nextResult, nextStored, finalStored⟩ :=
        equation_bind_eq_ok _ _ _ afterResult
      refine ⟨argumentValues, sampled, nextResult, argumentsStored, sampledStored,
        typesMatch, nextStored, ?_⟩
      cases finalStored
      simp [NodeResult.ofValues]
    · contradiction

theorem evalScope_success_artifact_step {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (input : ArtifactInput)
    (payloadStored : nodeValue.payload = .artifactInput input)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues link producerTrace producerScope value nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      data.artifactLinks[input.index]? = some link ∧
      trace.stages[link.producerStage]? = some producerTrace ∧
      producerTrace.scopes.find? (fun item => item.scope = link.producer.scope) = some producerScope ∧
      lookup producerScope.values link.producer = some value ∧
      outputTypesMatch nodeValue.outputs.toList
        (NodeResult.ofValues #[value]).values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values #[value]) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values #[value] }] } := by
  rcases nodeValue with ⟨nodePayload, nodeArguments, nodeOutputs⟩
  change nodePayload = .artifactInput input at payloadStored
  subst nodePayload
  rw [evalScope] at success
  simp only [if_neg fuelPositive, dif_pos indexBound] at success
  split at success
  · contradiction
  · rename_i actualNode actualStored
    have nodeEq : actualNode =
        { payload := .artifactInput input, arguments := nodeArguments, outputs := nodeOutputs } := by
      exact Option.some.inj (actualStored.symm.trans nodeStored)
    subst actualNode
    obtain ⟨argumentValues, argumentsStored, afterArguments⟩ :=
      equation_bind_eq_ok _ _ _ success
    simp at afterArguments
    cases hArtifact : data.artifactLinks[input.index]? with
    | none =>
      simp [hArtifact] at afterArguments
      change Except.error _ = .ok finalResult at afterArguments
      cases afterArguments
    | some link =>
      simp [hArtifact] at afterArguments
      split at afterArguments
      · contradiction
      ·
        cases hTrace : trace.stages[link.producerStage]? with
        | none =>
          simp [hTrace] at afterArguments
          change Except.error _ = .ok finalResult at afterArguments
          cases afterArguments
        | some producerTrace =>
          simp [hTrace] at afterArguments
          cases hScope : producerTrace.scopes.find? (fun item => item.scope = link.producer.scope) with
          | none =>
            simp [hScope] at afterArguments
            change Except.error _ = .ok finalResult at afterArguments
            cases afterArguments
          | some producerScope =>
            simp [hScope] at afterArguments
            cases hValue : lookup producerScope.values link.producer with
            | none =>
              simp [hValue] at afterArguments
              change Except.error _ = .ok finalResult at afterArguments
              cases afterArguments
            | some value =>
              simp [hValue] at afterArguments
              split at afterArguments
              · rename_i typesMatch
                obtain ⟨nextResult, nextStored, finalStored⟩ :=
                  equation_bind_eq_ok _ _ _ afterArguments
                cases finalStored
                exact ⟨argumentValues, link, producerTrace, producerScope, value, nextResult,
                    argumentsStored, (by simpa using hArtifact), (by simpa using hTrace),
                    (by simpa using hScope), (by simpa using hValue), typesMatch,
                    nextStored, by simp [NodeResult.ofValues]⟩
              · contradiction

theorem generatedPrimitivePrefixTwo {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (fuelTwoPositive : fuel - 1 ≠ 0)
    (firstBound : 0 < scope.nodes.size) (secondBound : 1 < scope.nodes.size)
    (firstNode : Node) (firstStored : scope.nodes[0]? = some firstNode)
    (firstPayload : NodePayload) (firstPayloadStored : firstNode.payload = firstPayload)
    (firstPrimitive : PrimitiveNodePayload firstPayload)
    (secondNode : Node) (secondStored : scope.nodes[1]? = some secondNode)
    (secondPayload : NodePayload) (secondPayloadStored : secondNode.payload = secondPayload)
    (secondPrimitive : PrimitiveNodePayload secondPayload)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 values fuel = .ok finalResult) :
    ∃ firstArguments firstResult firstNext secondArguments secondResult secondNext,
      resolveArguments stageNumber scope.id 0 values firstNode.arguments = .ok firstArguments ∧
      evalPrimitiveNode backend structural stageNumber scope.id 0 firstPayload
        firstArguments firstNode.outputs = .ok firstResult ∧
      outputTypesMatch firstNode.outputs.toList firstResult.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path 1 (appendNodeBindings scope.id 0 values firstResult.values) (fuel - 1) =
        .ok firstNext ∧
      resolveArguments stageNumber scope.id 1 (appendNodeBindings scope.id 0 values firstResult.values)
        secondNode.arguments = .ok secondArguments ∧
      evalPrimitiveNode backend structural stageNumber scope.id 1 secondPayload
        secondArguments secondNode.outputs = .ok secondResult ∧
      outputTypesMatch secondNode.outputs.toList secondResult.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path 2 (appendNodeBindings scope.id 1
          (appendNodeBindings scope.id 0 values firstResult.values) secondResult.values) (fuel - 2) =
        .ok secondNext := by
  obtain ⟨firstArguments, firstResult, firstNext, firstResolve, firstEval, firstTypes,
      firstSuccess, _⟩ := generatedPrimitiveNodeEquation data env structural trace stageNumber stage
    scopeNumber scope stageStored scopeStored inputs path 0 values fuel finalResult fuelPositive
    firstBound firstNode firstStored firstPayload firstPayloadStored firstPrimitive success
  obtain ⟨secondArguments, secondResult, secondNext, secondResolve, secondEval, secondTypes,
      secondSuccess, _⟩ := generatedPrimitiveNodeEquation data env structural trace stageNumber stage
    scopeNumber scope stageStored scopeStored inputs path 1
    (appendNodeBindings scope.id 0 values firstResult.values) (fuel - 1) firstNext
    fuelTwoPositive secondBound secondNode secondStored secondPayload secondPayloadStored secondPrimitive
    firstSuccess
  exact ⟨firstArguments, firstResult, firstNext, secondArguments, secondResult, secondNext,
    firstResolve, firstEval, firstTypes, firstSuccess, secondResolve, secondEval,
    secondTypes, secondSuccess⟩

/-! The common continuation shape used by renderer-generated flat-scope prefixes.  The `step`
    argument is supplied by generated code, where each concrete preceding node can use its
    payload-specific equation.  Thus the induction itself is independent of evaluator branches,
    while inputs, artifacts, samplers, primitives, subgraphs, grids, and loops all remain
    supported by the existing concrete equations. -/
def FlatScopeStep {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) : Prop :=
  ∃ nodeValue, ∃ result : NodeResult backend, ∃ nextResult : ScopeResult backend,
    scope.nodes[index]? = some nodeValue ∧
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

theorem generatedFlatScopePrefixAt {backend : SemanticBackend}
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
      FlatScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 rootValues rootFuel = .ok rootResult) :
    ∃ values fuel result,
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path target values fuel = .ok result ∧ fuel = rootFuel - target := by
  induction target generalizing rootValues rootFuel rootResult with
  | zero =>
      exact ⟨rootValues, rootFuel, rootResult, rootSuccess, by simp⟩
  | succ target ih =>
      have targetIndexBound : target < scope.nodes.size := by omega
      let stepPrevious := fun (limit index : Nat) (values : Array (Binding backend))
          (fuel : Nat) (currentResult : ScopeResult backend) (limitEq : limit = target)
          (indexBound : index < limit) (fuelPositive : fuel ≠ 0) (currentSuccess :
            evalScope data env structural trace stageNumber stage scopeNumber scope
              stageStored scopeStored inputs path index values fuel = .ok currentResult) =>
        step (limit + 1) index values fuel currentResult (by omega) (by omega) fuelPositive
          currentSuccess
      obtain ⟨values, fuel, currentResult, currentSuccess, fuelEq⟩ :=
        ih stepPrevious (rootValues := rootValues) (rootFuel := rootFuel) (rootResult := rootResult)
          (targetBound := by omega) (rootSuccess := rootSuccess)
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalScope] at currentSuccess
        simp only [if_pos fuelZero] at currentSuccess
        cases currentSuccess
      obtain ⟨nodeValue, nodeResult, nextResult, nodeStored, typesMatch, nextSuccess, finalStored⟩ :=
        step (target + 1) target values fuel currentResult rfl (by omega) fuelPositive currentSuccess
      exact ⟨appendNodeBindings scope.id target values nodeResult.values, fuel - 1, nextResult,
        nextSuccess, by omega⟩

/- Reversing evaluator scope order puts the locally emitted producer snapshot
   immediately after snapshots from earlier nodes.  If those earlier
   snapshots do not match, `find?` selects the producer before any later-node
   or nested-scope trace. -/
theorem reverseFindProducerOfFactorization {α : Type}
    (predicate : α → Bool) (all leading current earlier nested later : Array α)
    (producer : α)
    (allFactor : all = leading ++ current ++ earlier)
    (currentFactor : current = nested ++ later ++ #[producer])
    (earlierMiss : ∀ item ∈ earlier, predicate item = false)
    (producerMatch : predicate producer = true) :
    all.reverse.find? predicate = some producer := by
  have earlierFind : earlier.reverse.find? predicate = none := by
    apply Array.find?_eq_none.mpr
    intro item membership
    have originalMembership : item ∈ earlier := Array.mem_reverse.mp membership
    simp [earlierMiss item originalMembership]
  rw [allFactor, Array.reverse_append, Array.reverse_append, Array.find?_append, earlierFind]
  rw [currentFactor]
  simp [Array.reverse_append, Array.find?_append, producerMatch]

/- A selected loop iteration is a middle scope factor: reversing first examines later iterations.
   Once those later snapshots are known to have distinct iteration paths, the successful lookup in
   the selected child factor is preserved without inspecting earlier iterations. -/
theorem reverseFindOfMiddleFactor {α : Type}
    (predicate : α → Bool) (all leading middle trailing : Array α) (producer : α)
    (factor : all = leading ++ middle ++ trailing)
    (trailingMiss : ∀ item ∈ trailing, predicate item = false)
    (middleFind : middle.reverse.find? predicate = some producer) :
    all.reverse.find? predicate = some producer := by
  have trailingFind : trailing.reverse.find? predicate = none := by
    apply Array.find?_eq_none.mpr
    intro item membership
    have originalMembership : item ∈ trailing := Array.mem_reverse.mp membership
    simp [trailingMiss item originalMembership]
  rw [factor, Array.reverse_append, Array.reverse_append, Array.find?_append, trailingFind]
  simp [middleFind]

theorem scopeTraceContains_false_of_path_ne {backend : SemanticBackend}
    (occurrence : WireOccurrence) (snapshot : ScopeTrace backend)
    (different : snapshot.occurrence ≠ occurrence.path) :
    scopeTraceContains occurrence snapshot = false := by
  simp [scopeTraceContains, different]

/- `OccurrencePath.push` records the iteration in the last frame.  Distinct iterations therefore
   cannot alias even when the stage, scope, and owner are the same. -/
theorem loopIterationPath_ne (path : OccurrencePath) (stage scope owner first second : Nat)
    (different : first ≠ second) :
    path.push { stage := stage, scope := scope, owner := owner, laneOrIteration := first } ≠
      path.push { stage := stage, scope := scope, owner := owner, laneOrIteration := second } := by
  intro equality
  have lastEq := congrArg (fun value : OccurrencePath => value[path.size]?) equality
  have someFrameEq :
      some (OccurrenceFrame.mk stage scope owner first) =
      some (OccurrenceFrame.mk stage scope owner second) := by
    simpa only [Array.getElem?_push_eq] using lastEq
  have frameEq :
      OccurrenceFrame.mk stage scope owner first =
      OccurrenceFrame.mk stage scope owner second :=
    Option.some.inj someFrameEq
  exact different (congrArg OccurrenceFrame.laneOrIteration frameEq)

/- The same prefix induction, with the SSA frontier made explicit.  The evaluator's root call uses
   an empty array, but the parameter keeps this lemma reusable for a generated sub-scope whose
   incoming bindings have already been checked by its caller. -/
theorem generatedFlatScopePrefixAtWithBindingsBefore {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (rootValues : Array (Binding backend)) (rootFuel target : Nat)
    (targetBound : target ≤ scope.nodes.size)
    (rootBefore : BindingsBefore scope.id 0 rootValues)
    (step : ∀ (limit index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (finalResult : ScopeResult backend), limit = target → index < limit → fuel ≠ 0 →
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path index values fuel = .ok finalResult →
      FlatScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 rootValues rootFuel = .ok rootResult) :
    ∃ values fuel result leadingScopes earlierSnapshots,
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path target values fuel = .ok result ∧ fuel = rootFuel - target ∧
      BindingsBefore scope.id target values ∧
      ScopeTracesCoverBindings rootResult.scopes path values ∧
      rootResult.scopes = leadingScopes ++ result.scopes ++ earlierSnapshots ∧
      ∀ snapshot ∈ earlierSnapshots, BindingsBefore scope.id target snapshot.values := by
  induction target generalizing rootValues rootFuel rootResult with
  | zero =>
      exact ⟨rootValues, rootFuel, rootResult, #[], #[], rootSuccess, by simp, rootBefore,
        by
          intro binding membership
          exact False.elim (Nat.not_lt_zero binding.wire.node (rootBefore binding membership).2),
        by simp, by simp⟩
  | succ target ih =>
      have targetIndexBound : target < scope.nodes.size := by omega
      let stepPrevious := fun (limit index : Nat) (values : Array (Binding backend))
          (fuel : Nat) (currentResult : ScopeResult backend) (limitEq : limit = target)
          (indexBound : index < limit) (fuelPositive : fuel ≠ 0) (currentSuccess :
            evalScope data env structural trace stageNumber stage scopeNumber scope
              stageStored scopeStored inputs path index values fuel = .ok currentResult) =>
        step (limit + 1) index values fuel currentResult (by omega) (by omega) fuelPositive
          currentSuccess
      obtain ⟨values, fuel, currentResult, leadingScopes, earlierSnapshots, currentSuccess,
          fuelEq, valuesBefore, valuesCovered, scopesFactor, earlierBefore⟩ :=
        ih stepPrevious (rootValues := rootValues) (rootFuel := rootFuel) (rootResult := rootResult)
          (targetBound := by omega) (rootBefore := rootBefore) (rootSuccess := rootSuccess)
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalScope] at currentSuccess
        simp only [if_pos fuelZero] at currentSuccess
        cases currentSuccess
      obtain ⟨nodeValue, nodeResult, nextResult, nodeStored, typesMatch, nextSuccess, finalStored⟩ :=
        step (target + 1) target values fuel currentResult rfl (by omega) fuelPositive currentSuccess
      let currentSnapshot : ScopeTrace backend := {
        scope := scope.id
        occurrence := path
        values := appendNodeBindings scope.id target values nodeResult.values }
      refine ⟨appendNodeBindings scope.id target values nodeResult.values, fuel - 1, nextResult,
        leadingScopes ++ nodeResult.scopes, #[currentSnapshot] ++ earlierSnapshots,
        nextSuccess, by omega, bindingsBefore_appendNodeBindings valuesBefore, ?_, ?_, ?_⟩
      · intro binding membership
        obtain old | ⟨port, portBound, portStored, wireEq⟩ :=
          foldl_bindings_mem_cases (List.range nodeResult.values.size)
            (by intro entry entryMem; exact List.mem_range.mp entryMem)
            values binding (by simpa [appendNodeBindings] using membership)
        · exact valuesCovered binding old
        · have currentLookup : lookup currentSnapshot.values binding.wire = some binding.value := by
            rw [wireEq]
            exact appendNodeBindings_lookup_output valuesBefore portBound portStored
          have earlierMiss : ∀ snapshot ∈ earlierSnapshots,
              scopeTraceContains { stage := 0, path := path, wire := binding.wire } snapshot =
                false := by
            intro snapshot snapshotMem
            have missing := (earlierBefore snapshot snapshotMem).lookup_target_none port
            rw [wireEq]
            simp [scopeTraceContains, currentSnapshot, missing]
          have currentMatch :
              scopeTraceContains { stage := 0, path := path, wire := binding.wire }
                currentSnapshot = true := by
            rw [wireEq]
            simp only [scopeTraceContains, currentSnapshot, true_and]
            rw [appendNodeBindings_lookup_output valuesBefore portBound portStored]
            rfl
          have currentScopes :
              currentResult.scopes = nodeResult.scopes ++ nextResult.scopes ++
                #[currentSnapshot] := by
            simpa [currentSnapshot] using congrArg ScopeResult.scopes finalStored
          have found : rootResult.scopes.reverse.find?
              (scopeTraceContains { stage := 0, path := path, wire := binding.wire }) =
                some currentSnapshot := by
            exact reverseFindProducerOfFactorization
              (scopeTraceContains { stage := 0, path := path, wire := binding.wire })
              rootResult.scopes leadingScopes currentResult.scopes earlierSnapshots
              nodeResult.scopes nextResult.scopes currentSnapshot scopesFactor currentScopes
              earlierMiss currentMatch
          rw [found]
          exact currentLookup
      · rw [scopesFactor, finalStored]
        simp [currentSnapshot, Array.append_assoc]
      · intro snapshot membership
        simp only [Array.mem_append, Array.mem_singleton] at membership
        rcases membership with rfl | membership
        · exact bindingsBefore_appendNodeBindings valuesBefore
        · exact (earlierBefore snapshot membership).mono (Nat.le_succ target)

/- Root/subscope clients use this theorem to obtain an output from the actual evaluator trace.  It
   combines the prefix frontier with one primitive step; no caller supplies a trace scope or an
   alternative value. -/
theorem generatedPrimitiveOutputAtWithBindingsBefore {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (rootValues : Array (Binding backend)) (rootFuel target : Nat)
    (targetBound : target + 1 ≤ scope.nodes.size)
    (rootBefore : BindingsBefore scope.id 0 rootValues)
    (step : ∀ (limit index : Nat) (values : Array (Binding backend)) (fuel : Nat)
      (finalResult : ScopeResult backend), limit = target → index < limit → fuel ≠ 0 →
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path index values fuel = .ok finalResult →
      FlatScopeStep data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path index values fuel finalResult)
    (rootResult : ScopeResult backend)
    (rootSuccess : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path 0 rootValues rootFuel = .ok rootResult)
    (nodeValue : Node) (nodeStored : scope.nodes[target]? = some nodeValue)
    (payload : NodePayload) (payloadStored : nodeValue.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload) (port : Nat)
    (portBound : port < nodeValue.outputs.size) :
    ∃ (values : Array (Binding backend)) (fuel : Nat) (currentResult : ScopeResult backend)
      (nodeResult : NodeResult backend) (nextResult : ScopeResult backend)
      (arguments : Array (DynamicValue backend)),
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path target values fuel = .ok currentResult ∧
      resolveArguments stageNumber scope.id target values nodeValue.arguments = .ok arguments ∧
      ScopeTracesCoverBindings rootResult.scopes path values ∧
      evalPrimitiveNode backend structural stageNumber scope.id target payload arguments
        nodeValue.outputs = .ok nodeResult ∧
      ∃ leadingScopes earlierSnapshots scopeTrace output,
        rootResult.scopes = leadingScopes ++ currentResult.scopes ++ earlierSnapshots ∧
        (∀ snapshot ∈ earlierSnapshots,
          BindingsBefore scope.id target snapshot.values) ∧
        currentResult.scopes = nodeResult.scopes ++ nextResult.scopes ++ #[scopeTrace] ∧
        scopeTrace = {
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id target values nodeResult.values } ∧
        nodeResult.values[port]? = some output ∧
        lookup scopeTrace.values { scope := scope.id, node := target, port := port } =
          some output := by
  obtain ⟨values, fuel, currentResult, leadingScopes, earlierSnapshots, currentSuccess,
      _fuelEq, valuesBefore, valuesCovered, scopesFactor, earlierBefore⟩ :=
    generatedFlatScopePrefixAtWithBindingsBefore data env structural trace stageNumber stage
      scopeNumber scope stageStored scopeStored inputs path rootValues rootFuel target
      (by omega) rootBefore step rootResult rootSuccess
  have fuelPositive : fuel ≠ 0 := by
    intro fuelZero
    rw [evalScope] at currentSuccess
    simp only [if_pos fuelZero] at currentSuccess
    cases currentSuccess
  obtain ⟨arguments, nodeResult, nextResult, argumentsStored, resultStored', typesMatch,
      nextSuccess, finalStored⟩ :=
    generatedPrimitiveNodeEquation data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path target values fuel currentResult fuelPositive
      (by omega) nodeValue nodeStored payload payloadStored primitivePayload currentSuccess
  have sizes : nodeValue.outputs.size = nodeResult.values.size := by
    have lengths := outputTypesMatch_size nodeValue.outputs.toList nodeResult.values.toList typesMatch
    simpa using lengths
  have portResultBound : port < nodeResult.values.size := by omega
  obtain ⟨outputValue, outputValueStored⟩ :
      ∃ outputValue, nodeResult.values[port]? = some outputValue := by
    let outputValue := (nodeResult.values[port]'portResultBound)
    have : getElem? nodeResult.values port = some outputValue := by
      simp [outputValue, portResultBound]
    exact ⟨outputValue, this⟩
  let currentScope : ScopeTrace backend := {
    scope := scope.id
    occurrence := path
    values := appendNodeBindings scope.id target values nodeResult.values }
  refine ⟨values, fuel, currentResult, nodeResult, nextResult, arguments, currentSuccess,
    argumentsStored, valuesCovered, resultStored', ?_⟩
  refine ⟨leadingScopes, earlierSnapshots, currentScope, outputValue, scopesFactor,
    earlierBefore, ?_, rfl, outputValueStored, ?_⟩
  · simpa [currentScope] using congrArg ScopeResult.scopes finalStored
  apply appendNodeBindings_lookup_output valuesBefore
  · exact portResultBound
  · exact outputValueStored

/- The public evaluator bridge exposes the first concrete stage/root call.  Later stages are
   obtained by repeating `evalStages_success_step`; keeping that continuation explicit avoids a
   second evaluator and lets generated code choose only the roots it actually consumes. -/
theorem generatedStageSuccessAtFrom {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (finalTrace : Trace backend) (start steps : Nat)
    (targetBound : start + steps < data.stages.size) (trace : Trace backend)
    (success : evalStages data env start trace = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : data.stages[start + steps]? = some stage, ∃ stageTrace,
      evalStages data env (start + steps) { stages := tracePrefix } = .ok finalTrace ∧
      evalStage data env { stages := tracePrefix } (start + steps) stage stageStored = .ok stageTrace := by
  induction steps generalizing start trace with
  | zero =>
      have startBound : start < data.stages.size := by simpa using targetBound
      obtain ⟨stage, stageStored, stageTrace, stageSuccess, _⟩ :=
        evalStages_success_step data env start trace finalTrace startBound success
      exact ⟨trace.stages, stage, stageStored, stageTrace, success, stageSuccess⟩
  | succ steps ih =>
      have startBound : start < data.stages.size := by omega
      obtain ⟨currentStage, currentStored, currentTrace, currentSuccess, restSuccess⟩ :=
        evalStages_success_step data env start trace finalTrace startBound success
      have nextBound : start + 1 + steps < data.stages.size := by omega
      obtain ⟨tracePrefix, stage, stageStored, stageTrace, targetSuccess, stageSuccess⟩ :=
        ih (start := start + 1) (trace := { stages := trace.stages.push currentTrace }) nextBound
          restSuccess
      exact ⟨tracePrefix, stage,
        by simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using stageStored, stageTrace,
        by simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using targetSuccess,
        by simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using stageSuccess⟩

/- Once a stage has been emitted, later stages cannot shadow its stage number. -/
theorem evalStages_preserves_stage_find {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (index target : Nat)
    (trace finalTrace : Trace backend) (stageTrace : StageTrace backend)
    (bound : index ≤ data.stages.size) (targetLt : target < index)
    (find : trace.stages.find? (fun stage => stage.stage = target) = some stageTrace)
    (success : evalStages data env index trace = .ok finalTrace) :
    finalTrace.stages.find? (fun stage => stage.stage = target) = some stageTrace := by
  induction hmeasure : data.stages.size - index using Nat.strong_induction_on
    generalizing index trace finalTrace with
  | h measure ih =>
      by_cases done : index < data.stages.size
      · obtain ⟨stage, stageStored, current, currentSuccess, restSuccess⟩ :=
          evalStages_success_step data env index trace finalTrace done success
        obtain ⟨_, _, _, _, currentEq⟩ :=
          evalStage_success_root data env trace index stage stageStored current currentSuccess
        have currentStage : current.stage = index := by simpa [currentEq]
        have currentNotTarget : current.stage ≠ target := by omega
        have pushedFind :
            (trace.stages.push current).find? (fun item => item.stage = target) =
              some stageTrace := by
          rw [Array.find?_push, find]
          simp [currentNotTarget]
        exact ih (data.stages.size - (index + 1)) (by omega) (index + 1)
          { stages := trace.stages.push current } finalTrace (by omega) (by omega)
          pushedFind restSuccess (by omega)
      · have indexEq : index = data.stages.size := by omega
        subst index
        simp [evalStages, done] at success
        exact (Except.ok.inj success) ▸ find

/- `evalStages` only appends stage traces.  Therefore an artifact producer selected by an
   existing typed link remains the same array entry after the consumer and all later stages run. -/
theorem evalStages_preserves_stage_getElem {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (index target : Nat)
    (trace finalTrace : Trace backend) (stageTrace : StageTrace backend)
    (bound : index ≤ data.stages.size)
    (stored : trace.stages[target]? = some stageTrace)
    (success : evalStages data env index trace = .ok finalTrace) :
    finalTrace.stages[target]? = some stageTrace := by
  induction hmeasure : data.stages.size - index using Nat.strong_induction_on
    generalizing index trace finalTrace with
  | h measure ih =>
      by_cases done : index < data.stages.size
      · obtain ⟨stage, stageStored, current, currentSuccess, restSuccess⟩ :=
          evalStages_success_step data env index trace finalTrace done success
        have targetBound : target < trace.stages.size :=
          (Array.getElem?_eq_some_iff.mp stored).choose
        have pushedStored : (trace.stages.push current)[target]? = some stageTrace := by
          rw [Array.getElem?_push, if_neg (Nat.ne_of_lt targetBound)]
          exact stored
        exact ih (data.stages.size - (index + 1)) (by omega) (index + 1)
          { stages := trace.stages.push current } finalTrace (by omega) pushedStored restSuccess
          (by omega)
      · have indexEq : index = data.stages.size := by omega
        subst index
        simp [evalStages, done] at success
        exact (Except.ok.inj success) ▸ stored

/- Every stage already present before evaluator index `index` was emitted at a strictly smaller
   index.  Starting from the public evaluator's empty trace, this invariant makes stage-number
   lookup deterministic instead of relying on array membership alone. -/
def StageNumbersBefore {backend : SemanticBackend} (trace : Trace backend) (index : Nat) : Prop :=
  ∀ stageTrace ∈ trace.stages, stageTrace.stage < index

theorem stageNumbersBefore_empty {backend : SemanticBackend} (index : Nat) :
    StageNumbersBefore ({ stages := #[] } : Trace backend) index := by
  intro stageTrace membership
  simp at membership

theorem stageNumbersBefore_push {backend : SemanticBackend}
    (trace : Trace backend) (index : Nat) (stageTrace : StageTrace backend)
    (before : StageNumbersBefore trace index) (stageEq : stageTrace.stage = index) :
    StageNumbersBefore { stages := trace.stages.push stageTrace } (index + 1) := by
  intro item membership
  rw [Array.mem_push] at membership
  rcases membership with membership | rfl
  · exact Nat.lt_succ_of_lt (before item membership)
  · omega

theorem stageNumbersBefore_find_none {backend : SemanticBackend}
    (trace : Trace backend) (index : Nat) (before : StageNumbersBefore trace index) :
    trace.stages.find? (fun stage => stage.stage = index) = none := by
  apply Array.find?_eq_none.mpr
  intro stageTrace membership
  have stageLt := before stageTrace membership
  simp [Nat.ne_of_lt stageLt]

/- Follow the public evaluator to one concrete stage, then retain the exact first-match lookup
   through the remaining stages.  The only invariant is derived from the empty initial trace. -/
theorem generatedStageSuccessAtFromWithFind {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (finalTrace : Trace backend)
    (start steps : Nat) (targetBound : start + steps < data.stages.size)
    (trace : Trace backend) (before : StageNumbersBefore trace start)
    (success : evalStages data env start trace = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : data.stages[start + steps]? = some stage, ∃ stageTrace,
      evalStages data env (start + steps) { stages := tracePrefix } = .ok finalTrace ∧
      evalStage data env { stages := tracePrefix } (start + steps) stage stageStored =
        .ok stageTrace ∧
      finalTrace.stages.find? (fun item => item.stage = start + steps) = some stageTrace := by
  induction steps generalizing start trace with
  | zero =>
      have startBound : start < data.stages.size := by simpa using targetBound
      obtain ⟨stage, stageStored, stageTrace, stageSuccess, restSuccess⟩ :=
        evalStages_success_step data env start trace finalTrace startBound success
      obtain ⟨_, _, _, _, stageTraceEq⟩ :=
        evalStage_success_root data env trace start stage stageStored stageTrace stageSuccess
      have stageNumberEq : stageTrace.stage = start := by simpa [stageTraceEq]
      have noEarlier := stageNumbersBefore_find_none trace start before
      have pushedFind :
          (trace.stages.push stageTrace).find? (fun item => item.stage = start) =
            some stageTrace := by
        rw [Array.find?_push, noEarlier]
        simp [stageNumberEq]
      have finalFind := evalStages_preserves_stage_find data env (start + 1) start
        { stages := trace.stages.push stageTrace } finalTrace stageTrace (by omega) (by omega)
        pushedFind restSuccess
      exact ⟨trace.stages, stage, stageStored, stageTrace, success, stageSuccess, by
        simpa using finalFind⟩
  | succ steps ih =>
      have startBound : start < data.stages.size := by omega
      obtain ⟨stage, stageStored, stageTrace, stageSuccess, restSuccess⟩ :=
        evalStages_success_step data env start trace finalTrace startBound success
      obtain ⟨_, _, _, _, stageTraceEq⟩ :=
        evalStage_success_root data env trace start stage stageStored stageTrace stageSuccess
      have stageNumberEq : stageTrace.stage = start := by simpa [stageTraceEq]
      have nextBefore := stageNumbersBefore_push trace start stageTrace before stageNumberEq
      have nextBound : start + 1 + steps < data.stages.size := by omega
      obtain ⟨tracePrefix, targetStage, targetStored, targetTrace, targetSuccess,
          targetStageSuccess, targetFind⟩ :=
        ih (start := start + 1) (trace := { stages := trace.stages.push stageTrace }) nextBound
          nextBefore restSuccess
      exact ⟨tracePrefix, targetStage,
        by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using targetStored,
        targetTrace,
        by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using targetSuccess,
        by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using targetStageSuccess,
        by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using targetFind⟩

theorem generatedStageSuccessAtWithFind {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (finalTrace : Trace backend) (index : Nat)
    (indexBound : index < data.stages.size)
    (success : evalStages data env 0 { stages := #[] } = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : data.stages[index]? = some stage, ∃ stageTrace,
      evalStages data env index { stages := tracePrefix } = .ok finalTrace ∧
      evalStage data env { stages := tracePrefix } index stage stageStored = .ok stageTrace ∧
      finalTrace.stages.find? (fun item => item.stage = index) = some stageTrace := by
  simpa using generatedStageSuccessAtFromWithFind data env finalTrace 0 index
    (by simpa using indexBound) { stages := #[] } (stageNumbersBefore_empty 0) success

theorem generatedStageSuccessAt {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (finalTrace : Trace backend) (index : Nat)
    (indexBound : index < data.stages.size)
    (success : evalStages data env 0 { stages := #[] } = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : data.stages[index]? = some stage, ∃ stageTrace,
      evalStages data env index { stages := tracePrefix } = .ok finalTrace ∧
      evalStage data env { stages := tracePrefix } index stage stageStored = .ok stageTrace := by
  simpa using generatedStageSuccessAtFrom data env finalTrace 0 index (by simpa using indexBound)
    { stages := #[] }
    success

theorem generatedStageRootSuccessAt {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (finalTrace : Trace backend) (index : Nat)
    (indexBound : index < data.stages.size)
    (success : evalStages data env 0 { stages := #[] } = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : data.stages[index]? = some stage, ∃ stageTrace,
      ∃ root, ∃ rootStored : scopeAt stage stage.root = some root, ∃ result,
      evalStages data env index { stages := tracePrefix } = .ok finalTrace ∧
      evalStage data env { stages := tracePrefix } index stage stageStored = .ok stageTrace ∧
      evalScope data env {} { stages := tracePrefix } index stage stage.root root stageStored rootStored
        #[] #[] 0 #[] (evaluationFuel data) = .ok result ∧
      stageTrace = { stage := index, scopes := result.scopes } := by
  obtain ⟨tracePrefix, stage, stageStored, stageTrace, prefixSuccess, stageSuccess⟩ :=
    generatedStageSuccessAt data env finalTrace index indexBound success
  obtain ⟨root, rootStored, result, rootSuccess, stageTraceEq⟩ :=
    evalStage_success_root data env { stages := tracePrefix } index stage stageStored stageTrace
      stageSuccess
  exact ⟨tracePrefix, stage, stageStored, stageTrace, root, rootStored, result, prefixSuccess,
    stageSuccess, rootSuccess, stageTraceEq⟩

theorem generatedFirstStageRootSuccess {backend : SemanticBackend} (program : Program)
    (env : EvalEnv backend program.data) (finalTrace : Trace backend)
    (stageBound : 0 < program.data.stages.size)
    (success : eval backend program env = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : program.data.stages[0]? = some stage, ∃ stageTrace,
      ∃ root, ∃ rootStored : scopeAt stage stage.root = some root, ∃ result,
      evalStages program.data env 0 { stages := tracePrefix } = .ok finalTrace ∧
      evalStage program.data env { stages := tracePrefix } 0 stage stageStored = .ok stageTrace ∧
      evalScope program.data env {} { stages := tracePrefix } 0 stage stage.root root stageStored rootStored
        #[] #[] 0 #[] (evaluationFuel program.data) = .ok result ∧
      stageTrace = { stage := 0, scopes := result.scopes } := by
  exact generatedStageRootSuccessAt program.data env finalTrace 0 stageBound
    (eval_success_stages program env finalTrace success)

/- The public evaluator exposes the exact first-match trace for any concrete stage index. -/
theorem eval_success_stage_lookup {backend : SemanticBackend} (program : Program)
    (env : EvalEnv backend program.data) (finalTrace : Trace backend) (index : Nat)
    (indexBound : index < program.data.stages.size)
    (success : eval backend program env = .ok finalTrace) :
    ∃ tracePrefix : Array (StageTrace backend), ∃ stage,
      ∃ stageStored : program.data.stages[index]? = some stage, ∃ stageTrace,
      ∃ root, ∃ rootStored : scopeAt stage stage.root = some root, ∃ result,
      evalStages program.data env index { stages := tracePrefix } = .ok finalTrace ∧
      evalStage program.data env { stages := tracePrefix } index stage stageStored = .ok stageTrace ∧
      finalTrace.stages.find? (fun item => item.stage = index) = some stageTrace ∧
      evalScope program.data env {} { stages := tracePrefix } index stage stage.root root
        stageStored rootStored #[] #[] 0 #[] (evaluationFuel program.data) = .ok result ∧
      stageTrace = { stage := index, scopes := result.scopes } := by
  obtain ⟨tracePrefix, stage, stageStored, stageTrace, prefixSuccess, stageSuccess,
      stageFind⟩ := generatedStageSuccessAtWithFind program.data env finalTrace index indexBound
    (eval_success_stages program env finalTrace success)
  obtain ⟨root, rootStored, result, rootSuccess, stageTraceEq⟩ :=
    evalStage_success_root program.data env { stages := tracePrefix } index stage stageStored
      stageTrace stageSuccess
  exact ⟨tracePrefix, stage, stageStored, stageTrace, root, rootStored, result, prefixSuccess,
    stageSuccess, stageFind, rootSuccess, stageTraceEq⟩

/- Generated applications supply only stored syntax facts and evaluator inversion
   steps for nodes preceding the target.  The target value and both trace
   lookups are derived from the successful public evaluator run. -/
theorem eval_success_root_primitive_output_at {backend : SemanticBackend}
    (program : Program) (env : EvalEnv backend program.data) (finalTrace : Trace backend)
    (stageIndex : Nat) (stageIndexBound : stageIndex < program.data.stages.size)
    (success : eval backend program env = .ok finalTrace)
    (stage : Stage) (stageStored : program.data.stages[stageIndex]? = some stage)
    (root : Scope) (rootStored : scopeAt stage stage.root = some root)
    (target : Nat) (targetBound : target + 1 ≤ root.nodes.size)
    (prefixSteps : ∀ (tracePrefix : Array (StageTrace backend)) (limit index : Nat)
      (values : Array (Binding backend)) (fuel : Nat) (finalResult : ScopeResult backend),
      limit = target → index < limit → fuel ≠ 0 →
      evalScope program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel = .ok finalResult →
      FlatScopeStep program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel finalResult)
    (nodeValue : Node) (nodeStored : root.nodes[target]? = some nodeValue)
    (payload : NodePayload) (payloadStored : nodeValue.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload) (port : Nat)
    (portBound : port < nodeValue.outputs.size) :
    ∃ nodeResult output arguments values,
      resolveArguments stageIndex root.id target values nodeValue.arguments = .ok arguments ∧
      (∀ binding ∈ values,
        traceValueAt finalTrace (occurrenceOf stageIndex #[] binding.wire) = some binding.value) ∧
      evalPrimitiveNode backend {} stageIndex root.id target payload arguments nodeValue.outputs =
        .ok nodeResult ∧
      nodeResult.values[port]? = some output ∧
      traceValueAt finalTrace
        (occurrenceOf stageIndex #[] { scope := root.id, node := target, port := port }) =
          some output := by
  obtain ⟨tracePrefix, reachedStage, reachedStored, stageTrace, reachedRoot,
      reachedRootStored, rootResult, _prefixSuccess, _stageSuccess, stageFind, rootSuccess,
      stageTraceEq⟩ :=
    eval_success_stage_lookup program env finalTrace stageIndex stageIndexBound success
  have stageEq : reachedStage = stage := Option.some.inj (reachedStored.symm.trans stageStored)
  subst reachedStage
  have rootEq : reachedRoot = root :=
    Option.some.inj (reachedRootStored.symm.trans rootStored)
  subst reachedRoot
  obtain ⟨values, fuel, currentResult, nodeResult, nextResult, arguments, _currentSuccess,
      argumentsStored, valuesCovered, primitiveStored, leadingScopes, earlierSnapshots, producer,
      output, scopesFactor, earlierBefore, currentFactor, producerEq, outputStored, lookupStored⟩ :=
    generatedPrimitiveOutputAtWithBindingsBefore program.data env {} { stages := tracePrefix }
      stageIndex stage stage.root root stageStored rootStored #[] #[] #[]
      (evaluationFuel program.data) target targetBound (bindingsBefore_empty root.id 0)
      (prefixSteps tracePrefix) rootResult rootSuccess nodeValue nodeStored payload payloadStored
      primitivePayload port portBound
  let occurrence : WireOccurrence :=
    occurrenceOf stageIndex #[] { scope := root.id, node := target, port := port }
  have earlierMiss :
      ∀ snapshot ∈ earlierSnapshots, scopeTraceContains occurrence snapshot = false := by
    intro snapshot membership
    have missing := (earlierBefore snapshot membership).lookup_target_none port
    dsimp [occurrence, occurrenceOf]
    simp [scopeTraceContains, missing]
  have producerMatch : scopeTraceContains occurrence producer = true := by
    have concreteLookup := lookupStored
    rw [producerEq] at concreteLookup
    rw [producerEq]
    dsimp [occurrence, occurrenceOf]
    simp [scopeTraceContains, concreteLookup]
  have scopeFind : rootResult.scopes.reverse.find? (scopeTraceContains occurrence) =
      some producer := by
    exact reverseFindProducerOfFactorization (scopeTraceContains occurrence) rootResult.scopes
      leadingScopes currentResult.scopes earlierSnapshots nodeResult.scopes nextResult.scopes producer
      scopesFactor currentFactor earlierMiss producerMatch
  have valuesTraced : ∀ binding ∈ values,
      traceValueAt finalTrace (occurrenceOf stageIndex #[] binding.wire) =
        some binding.value := by
    intro binding membership
    have covered := valuesCovered binding membership
    dsimp [traceValueAt, occurrenceOf]
    rw [stageFind]
    simp only [Option.bind_some]
    rw [stageTraceEq]
    simpa [scopeTraceContains] using covered
  refine ⟨nodeResult, output, arguments, values, argumentsStored, valuesTraced,
    primitiveStored, outputStored, ?_⟩
  dsimp [traceValueAt, occurrenceOf]
  rw [stageFind]
  simp only [Option.bind_some]
  rw [stageTraceEq]
  change (rootResult.scopes.reverse.find? (scopeTraceContains occurrence)).bind
      (fun scopeTrace => lookup scopeTrace.values occurrence.wire) = some output
  rw [scopeFind]
  simp only [Option.bind_some]
  change lookup producer.values { scope := root.id, node := target, port := port } = some output
  exact lookupStored

/- An artifact input does not copy or reinterpret its producer value.  The linked evaluator reads
   one stored `ArtifactLink`, looks up its producer in the already-evaluated stage trace, and
   appends that exact value as port zero of the consumer node.  This theorem exposes both ends of
   that operation from one successful public evaluation, so applications need no provenance
   string or caller-provided equality. -/
theorem eval_success_root_artifact_input_at {backend : SemanticBackend}
    (program : Program) (env : EvalEnv backend program.data) (finalTrace : Trace backend)
    (stageIndex : Nat) (stageIndexBound : stageIndex < program.data.stages.size)
    (success : eval backend program env = .ok finalTrace)
    (stage : Stage) (stageStored : program.data.stages[stageIndex]? = some stage)
    (root : Scope) (rootStored : scopeAt stage stage.root = some root)
    (target : Nat) (targetBound : target + 1 ≤ root.nodes.size)
    (prefixSteps : ∀ (tracePrefix : Array (StageTrace backend)) (limit index : Nat)
      (values : Array (Binding backend)) (fuel : Nat) (finalResult : ScopeResult backend),
      limit = target → index < limit → fuel ≠ 0 →
      evalScope program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel = .ok finalResult →
      FlatScopeStep program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel finalResult)
    (storedNode : Node) (nodeStored : root.nodes[target]? = some storedNode)
    (input : ArtifactInput) (payloadStored : storedNode.payload = .artifactInput input) :
    ∃ link producerTrace producerScope value,
      program.data.artifactLinks[input.index]? = some link ∧
      finalTrace.stages[link.producerStage]? = some producerTrace ∧
      producerTrace.scopes.find? (fun item ↦ item.scope = link.producer.scope) =
        some producerScope ∧
      lookup producerScope.values link.producer = some value ∧
      traceValueAt finalTrace
        (occurrenceOf stageIndex #[] { scope := root.id, node := target, port := 0 }) =
          some value := by
  obtain ⟨tracePrefix, reachedStage, reachedStored, stageTrace, reachedRoot,
      reachedRootStored, rootResult, prefixSuccess, _stageSuccess, stageFind, rootSuccess,
      stageTraceEq⟩ :=
    eval_success_stage_lookup program env finalTrace stageIndex stageIndexBound success
  have stageEq : reachedStage = stage := Option.some.inj (reachedStored.symm.trans stageStored)
  subst reachedStage
  have rootEq : reachedRoot = root :=
    Option.some.inj (reachedRootStored.symm.trans rootStored)
  subst reachedRoot
  obtain ⟨values, fuel, currentResult, leadingScopes, earlierSnapshots, currentSuccess,
      _fuelEq, valuesBefore, _valuesCovered, scopesFactor, earlierBefore⟩ :=
    generatedFlatScopePrefixAtWithBindingsBefore program.data env {} { stages := tracePrefix }
      stageIndex stage stage.root root stageStored rootStored #[] #[] #[]
      (evaluationFuel program.data) target (by omega) (bindingsBefore_empty root.id 0)
      (prefixSteps tracePrefix) rootResult rootSuccess
  have fuelPositive : fuel ≠ 0 := by
    intro fuelZero
    rw [evalScope] at currentSuccess
    simp only [if_pos fuelZero] at currentSuccess
    cases currentSuccess
  obtain ⟨_arguments, link, producerTrace, producerScope, value, nextResult,
      _argumentsStored, linkStored, producerTraceStored, producerScopeStored,
      producerValueStored, _typesMatch, _nextSuccess, finalStored⟩ :=
    evalScope_success_artifact_step program.data env {} { stages := tracePrefix } stageIndex stage
      stage.root root stageStored rootStored #[] #[] target values fuel currentResult fuelPositive
      (by omega) storedNode nodeStored input payloadStored currentSuccess
  let consumerWire : WireRef := { scope := root.id, node := target, port := 0 }
  let consumerScope : ScopeTrace backend := {
    scope := root.id
    occurrence := #[]
    values := appendNodeBindings root.id target values #[value] }
  have consumerLookup : lookup consumerScope.values consumerWire = some value := by
    apply appendNodeBindings_lookup_output valuesBefore
    · simp
    · simp
  have currentFactor : currentResult.scopes = nextResult.scopes ++ #[consumerScope] := by
    simpa [consumerScope] using congrArg ScopeResult.scopes finalStored
  let occurrence : WireOccurrence := occurrenceOf stageIndex #[] consumerWire
  have earlierMiss :
      ∀ snapshot ∈ earlierSnapshots, scopeTraceContains occurrence snapshot = false := by
    intro snapshot membership
    have missing := (earlierBefore snapshot membership).lookup_target_none 0
    dsimp [occurrence, consumerWire, occurrenceOf]
    simp [scopeTraceContains, missing]
  have consumerMatch : scopeTraceContains occurrence consumerScope = true := by
    dsimp [scopeTraceContains, occurrence, consumerWire, occurrenceOf, consumerScope]
    simp only [decide_true]
    change (lookup consumerScope.values consumerWire).isSome = true
    rw [consumerLookup]
    rfl
  have scopeFind : rootResult.scopes.reverse.find? (scopeTraceContains occurrence) =
      some consumerScope := by
    exact reverseFindProducerOfFactorization (scopeTraceContains occurrence) rootResult.scopes
      leadingScopes currentResult.scopes earlierSnapshots #[] nextResult.scopes consumerScope
      scopesFactor (by simpa using currentFactor) earlierMiss consumerMatch
  have finalProducerStored : finalTrace.stages[link.producerStage]? = some producerTrace :=
    evalStages_preserves_stage_getElem program.data env stageIndex link.producerStage
      { stages := tracePrefix } finalTrace producerTrace (by omega) producerTraceStored prefixSuccess
  refine ⟨link, producerTrace, producerScope, value, linkStored, finalProducerStored,
    producerScopeStored, producerValueStored, ?_⟩
  dsimp [traceValueAt, occurrenceOf]
  rw [stageFind]
  simp only [Option.bind_some]
  rw [stageTraceEq]
  change (rootResult.scopes.reverse.find? (scopeTraceContains occurrence)).bind
      (fun scopeTrace ↦ lookup scopeTrace.values occurrence.wire) = some value
  rw [scopeFind]
  simp only [Option.bind_some]
  exact consumerLookup

/- A reached primitive run packages only facts recovered from the public evaluator: the resolved
   operands, backend execution, and producer/output trace equations.  Applications may interpret
   these values, but cannot replace them with caller-supplied output equations. -/
structure ReachedPrimitiveRun {backend : SemanticBackend}
    (trace : Trace backend) (structural : StructuralEnv) (stage scope node : Nat)
    (path : OccurrencePath) (payload : NodePayload) (storedNode : Node) (port : Nat) where
  nodeResult : NodeResult backend
  output : DynamicValue backend
  arguments : Array (DynamicValue backend)
  values : Array (Binding backend)
  argumentsResolved :
    resolveArguments stage scope node values storedNode.arguments = .ok arguments
  payloadStored : storedNode.payload = payload
  valuesTraced : ∀ binding ∈ values,
    traceValueAt trace (occurrenceOf stage path binding.wire) = some binding.value
  primitiveEvaluated :
    evalPrimitiveNode backend structural stage scope node payload arguments storedNode.outputs =
      .ok nodeResult
  outputStored : nodeResult.values[port]? = some output
  outputTraced :
    traceValueAt trace (occurrenceOf stage path { scope := scope, node := node, port := port }) =
      some output

/- Generated site theorems discharge the concrete finite-prefix callback.  Every dynamic field of
   the returned run is then copied from one inversion of the successful public evaluator call. -/
theorem reachedRootPrimitiveRun {backend : SemanticBackend}
    (program : Program) (env : EvalEnv backend program.data) (trace : Trace backend)
    (stageIndex : Nat) (stageIndexBound : stageIndex < program.data.stages.size)
    (success : eval backend program env = .ok trace)
    (stage : Stage) (stageStored : program.data.stages[stageIndex]? = some stage)
    (root : Scope) (rootStored : scopeAt stage stage.root = some root)
    (target : Nat) (targetBound : target + 1 ≤ root.nodes.size)
    (prefixSteps : ∀ (tracePrefix : Array (StageTrace backend)) (limit index : Nat)
      (values : Array (Binding backend)) (fuel : Nat) (finalResult : ScopeResult backend),
      limit = target → index < limit → fuel ≠ 0 →
      evalScope program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel = .ok finalResult →
      FlatScopeStep program.data env {} { stages := tracePrefix } stageIndex stage stage.root root
        stageStored rootStored #[] #[] index values fuel finalResult)
    (storedNode : Node) (nodeStored : root.nodes[target]? = some storedNode)
    (payload : NodePayload) (payloadStored : storedNode.payload = payload)
    (primitivePayload : PrimitiveNodePayload payload) (port : Nat)
    (portBound : port < storedNode.outputs.size) :
    Nonempty (ReachedPrimitiveRun trace {} stageIndex root.id target #[] payload storedNode port) := by
  obtain ⟨nodeResult, output, arguments, values, argumentsResolved, valuesTraced,
      primitiveEvaluated, outputStored, outputTraced⟩ :=
    eval_success_root_primitive_output_at program env trace stageIndex stageIndexBound success
      stage stageStored root rootStored target targetBound prefixSteps storedNode nodeStored payload
      payloadStored primitivePayload port portBound
  exact ⟨{
    nodeResult := nodeResult
    output := output
    arguments := arguments
    values := values
    argumentsResolved := argumentsResolved
    payloadStored := payloadStored
    valuesTraced := valuesTraced
    primitiveEvaluated := primitiveEvaluated
    outputStored := outputStored
    outputTraced := outputTraced
  }⟩

/- A successful argument resolver and the SSA coverage returned by the public root bridge identify
   each operand with the value at its concrete producer occurrence.  This is the generic dataflow
   edge used by application proofs; it does not inspect payloads or invent graph provenance. -/
theorem resolvedArgument_trace {backend : SemanticBackend}
    {stage scope node : Nat} {wires : Array WireRef}
    {values : Array (Binding backend)} {arguments : Array (DynamicValue backend)}
    {trace : Trace backend}
    (resolved : resolveArguments stage scope node values wires = .ok arguments)
    (valuesTraced : ∀ binding ∈ values,
      traceValueAt trace (occurrenceOf stage #[] binding.wire) = some binding.value)
    (index : Nat) (indexBound : index < wires.size) :
    ∃ argumentBound : index < arguments.size,
      traceValueAt trace (occurrenceOf stage #[] wires[index]) = some arguments[index] := by
  unfold resolveArguments at resolved
  let resolveOne : WireRef → Except EvalError (DynamicValue backend) :=
    fun wire => match lookup values wire with
      | some value => Except.ok value
      | none => Except.error (EvalError.missingPort stage scope node wire.port)
  have resolved' : wires.mapM resolveOne = .ok arguments := by
    simpa [resolveOne] using resolved
  obtain ⟨argumentBound, point⟩ := array_mapM_getElem resolveOne resolved' indexBound
  cases lookupStored : lookup values wires[index] with
  | none =>
      simp [resolveOne, lookupStored] at point
  | some value =>
      simp only [resolveOne, lookupStored] at point
      have valueEq : value = arguments[index] := Except.ok.inj point
      subst value
      unfold lookup at lookupStored
      cases found : values.find? (fun binding => binding.wire = wires[index]) with
      | none => simp [found] at lookupStored
      | some binding =>
          have foundData := (Array.find?_eq_some_iff_getElem).mp found
          have membership : binding ∈ values := by
            rcases foundData.2 with ⟨storedIndex, storedBound, stored, _⟩
            exact Array.mem_iff_getElem.mpr ⟨storedIndex, storedBound, stored⟩
          have wireEq : binding.wire = wires[index] := of_decide_eq_true foundData.1
          have valueEq : binding.value = arguments[index] := by
            simpa [found] using lookupStored
          refine ⟨argumentBound, ?_⟩
          rw [← wireEq, ← valueEq]
          exact valuesTraced binding membership

/- The same SSA argument lookup is valid inside a nested loop/grid occurrence.  The occurrence
   path is copied unchanged from the evaluator's `ReachedPrimitiveRun`; argument resolution only
   changes the wire, never the dynamic path that owns its value. -/
theorem resolvedArgument_trace_at_path {backend : SemanticBackend}
    {stage scope node : Nat} {path : OccurrencePath} {wires : Array WireRef}
    {values : Array (Binding backend)} {arguments : Array (DynamicValue backend)}
    {trace : Trace backend}
    (resolved : resolveArguments stage scope node values wires = .ok arguments)
    (valuesTraced : ∀ binding ∈ values,
      traceValueAt trace (occurrenceOf stage path binding.wire) = some binding.value)
    (index : Nat) (indexBound : index < wires.size) :
    ∃ argumentBound : index < arguments.size,
      traceValueAt trace (occurrenceOf stage path wires[index]) = some arguments[index] := by
  unfold resolveArguments at resolved
  let resolveOne : WireRef → Except EvalError (DynamicValue backend) :=
    fun wire => match lookup values wire with
      | some value => Except.ok value
      | none => Except.error (EvalError.missingPort stage scope node wire.port)
  have resolved' : wires.mapM resolveOne = .ok arguments := by
    simpa [resolveOne] using resolved
  obtain ⟨argumentBound, point⟩ := array_mapM_getElem resolveOne resolved' indexBound
  cases lookupStored : lookup values wires[index] with
  | none =>
      simp [resolveOne, lookupStored] at point
  | some value =>
      simp only [resolveOne, lookupStored] at point
      have valueEq : value = arguments[index] := Except.ok.inj point
      subst value
      unfold lookup at lookupStored
      cases found : values.find? (fun binding => binding.wire = wires[index]) with
      | none => simp [found] at lookupStored
      | some binding =>
          have foundData := (Array.find?_eq_some_iff_getElem).mp found
          have membership : binding ∈ values := by
            rcases foundData.2 with ⟨storedIndex, storedBound, stored, _⟩
            exact Array.mem_iff_getElem.mpr ⟨storedIndex, storedBound, stored⟩
          have wireEq : binding.wire = wires[index] := of_decide_eq_true foundData.1
          have valueEq : binding.value = arguments[index] := by
            simpa [found] using lookupStored
          refine ⟨argumentBound, ?_⟩
          rw [← wireEq, ← valueEq]
          exact valuesTraced binding membership

theorem generatedSubgraphNodeEquation {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (call : SubgraphPayload)
    (payloadStored : nodeValue.payload = .subgraphCall call)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues child, ∃ childStored : scopeAt stage call.child = some child,
      ∃ childInputs : Array (Binding backend), ∃ childResult : ScopeResult backend,
      ∃ childOutputs : Array (DynamicValue backend), ∃ nextResult : ScopeResult backend,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      checkedChildInputs stageNumber scope.id index child argumentValues = .ok childInputs ∧
      evalScope data env structural trace stageNumber stage call.child child stageStored childStored
          childInputs (path.push {
            stage := stageNumber, scope := scope.id, owner := index, laneOrIteration := 0 })
          0 #[] (fuel - 1) = .ok childResult ∧
      child.outputs.mapM (fun output =>
          (match lookup childResult.values output with
          | some value => Except.ok value
          | none => Except.error
              (EvalError.missingPort stageNumber child.id output.node output.port) :
            Except EvalError (DynamicValue backend))) = .ok childOutputs ∧
      outputTypesMatch nodeValue.outputs.toList childOutputs.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values childOutputs) (fuel - 1) =
        .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := childResult.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values childOutputs }] } := by
  exact evalScope_success_subgraph_step data env structural trace stageNumber stage scopeNumber scope
    stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound nodeValue
    nodeStored call payloadStored success

def ParallelGridEquation {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (structural : StructuralEnv) (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId) (scope : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope)
    (inputs : Array (Binding backend)) (path : OccurrencePath) (index : Nat)
    (values : Array (Binding backend)) (fuel : Nat) (nodeValue : Node) (grid : GridPayload)
    (finalResult : ScopeResult backend) : Prop :=
  ∃ argumentValues child, ∃ childStored : scopeAt stage grid.child = some child,
    ∃ concreteShape : Array Nat, ∃ lanes : Nat,
    ∃ laneResults : Array (Array (DynamicValue backend) × Array (ScopeTrace backend)),
    ∃ packed nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalShape structural stageNumber scope.id index grid.shape = .ok concreteShape ∧
      lanes = shapeProductArray concreteShape ∧
      (Array.range lanes).mapM (fun lane => do
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
        pure (outputs, childResult.scopes)) = .ok laneResults ∧
      let laneScopes := laneResults.foldl (fun result item => result ++ item.2) #[]
      nodeValue.outputs.mapIdxM (fun outputIndex output => do
        let laneValues ← laneResults.mapM
          (fun result : Array (DynamicValue backend) × Array (ScopeTrace backend) =>
          (match result.1[outputIndex]? with
          | some value => Except.ok value
          | none => Except.error (EvalError.missingPort stageNumber child.id outputIndex 0) :
            Except EvalError (DynamicValue backend)))
        let packedValues ← packDeclaredFamily stageNumber scope.id index output laneValues
        match packedValues[0]? with
        | some value => Except.ok value
        | none => Except.error (EvalError.wrongType stageNumber scope.id index)) = .ok packed ∧
      outputTypesMatch nodeValue.outputs.toList packed.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored
        scopeStored inputs path (index + 1)
        (appendNodeBindings scope.id index values packed) (fuel - 1) = .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := laneScopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values packed }] }

theorem generatedParallelGridNodeEquation {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (grid : GridPayload)
    (payloadStored : nodeValue.payload = .parallelGrid grid)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ParallelGridEquation data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel nodeValue grid finalResult := by
  obtain ⟨argumentValues, child, childStored, concreteShape, lanes, laneResults, packed, nextResult,
      argumentsStored, shapeStored, lanesStored, laneResultsStored, packedStored, typesMatch,
      nextStored, finalStored⟩ :=
    evalScope_success_parallelGrid_step data env structural trace stageNumber stage scopeNumber scope
    stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound nodeValue
      nodeStored grid payloadStored success
  exact ⟨argumentValues, child, childStored, concreteShape, lanes, laneResults, packed,
    nextResult, argumentsStored, shapeStored, lanesStored, laneResultsStored, packedStored,
    typesMatch, nextStored, finalStored⟩

def SequentialLoopEquation {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (structural : StructuralEnv) (trace : Trace backend)
    (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId) (scope : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope)
    (inputs : Array (Binding backend)) (path : OccurrencePath) (index : Nat)
    (values : Array (Binding backend)) (fuel : Nat) (nodeValue : Node) (loop : LoopPayload)
    (finalResult : ScopeResult backend) : Prop :=
  ∃ argumentValues child, ∃ childStored : scopeAt stage loop.child = some child,
    ∃ loopResult nextResult,
      resolveArguments stageNumber scope.id index values nodeValue.arguments = .ok argumentValues ∧
      evalSequentialLoop data env trace stageNumber stage loop.child child stageStored
        childStored loop index argumentValues structural path 0 (fuel - 1) = .ok loopResult ∧
      outputTypesMatch nodeValue.outputs.toList loopResult.values.toList = true ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
        inputs path (index + 1)
          (appendNodeBindings scope.id index values loopResult.values) (fuel - 1) = .ok nextResult ∧
      finalResult = {
        values := nextResult.values
        scopes := loopResult.scopes ++ nextResult.scopes ++ #[{
          scope := scope.id
          occurrence := path
          values := appendNodeBindings scope.id index values loopResult.values }] }

theorem generatedSequentialLoopNodeEquation {backend : SemanticBackend}
    (data : ProgramData) (env : EvalEnv backend data) (structural : StructuralEnv)
    (trace : Trace backend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding backend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding backend)) (fuel : Nat)
    (finalResult : ScopeResult backend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue) (loop : LoopPayload)
    (payloadStored : nodeValue.payload = .sequentialLoop loop)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    SequentialLoopEquation data env structural trace stageNumber stage scopeNumber scope stageStored
      scopeStored inputs path index values fuel nodeValue loop finalResult := by
  obtain ⟨argumentValues, child, childStored, loopResult, nextResult, argumentsStored, loopStored,
      typesMatch, nextStored, finalStored⟩ :=
    evalScope_success_sequentialLoop_step data env structural trace stageNumber stage scopeNumber scope
    stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound nodeValue
      nodeStored loop payloadStored success
  exact ⟨argumentValues, child, childStored, loopResult, nextResult, argumentsStored,
    loopStored, typesMatch, nextStored, finalStored⟩

/- Follow one successful sequential-loop execution to a requested iteration.  The theorem returns
   the exact child inputs, structural index slot, and occurrence path used by `evalScope`; it does
   not summarize or reinterpret the loop body.  This is the narrow nested-scope bridge needed by
   application proofs whose semantic operation occurs inside a loop child. -/
theorem evalSequentialLoop_success_child_at {backend : SemanticBackend} (data : ProgramData)
    (env : EvalEnv backend data) (trace : Trace backend) (stageNumber : Nat) (stage : Stage)
    (childNumber : ScopeId) (child : Scope)
    (stageStored : data.stages[stageNumber]? = some stage)
    (childStored : scopeAt stage childNumber = some child) (loop : LoopPayload) (owner : NodeId)
    (arguments : Array (DynamicValue backend)) (structural : StructuralEnv) (path : OccurrencePath)
    (iteration fuel count offset : Nat)
    (countStored : evalNatExpr structural stageNumber child.id owner loop.count = .ok count)
    (requestedBound : iteration + offset < count) (finalResult : NodeResult backend)
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
        finalResult.scopes = leading ++ childResult.scopes ++ trailing := by
  induction offset generalizing arguments iteration fuel finalResult with
  | zero =>
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalSequentialLoop] at success
        simp [fuelZero] at success
      obtain ⟨childInputs, childResult, childValues, rest, inputsStored, childStored', _, _, _,
          finalStored⟩ :=
        evalSequentialLoop_success_iteration_step data env trace stageNumber stage childNumber child
          stageStored childStored loop owner arguments structural path iteration fuel count
          fuelPositive countStored (by simpa using requestedBound) finalResult success
      refine ⟨arguments, fuel, finalResult, childInputs, childResult, ?_, inputsStored, ?_,
        #[], rest.scopes, ?_⟩
      · simpa using success
      · simpa using childStored'
      · simpa using congrArg NodeResult.scopes finalStored
  | succ offset ih =>
      have currentBound : iteration < count := by omega
      have fuelPositive : fuel ≠ 0 := by
        intro fuelZero
        rw [evalSequentialLoop] at success
        simp [fuelZero] at success
      obtain ⟨childInputs, childResult, childValues, rest, _, _, _, childValuesSize,
          restStored, finalStored⟩ :=
        evalSequentialLoop_success_iteration_step data env trace stageNumber stage childNumber child
          stageStored childStored loop owner arguments structural path iteration fuel count
          fuelPositive countStored currentBound finalResult success
      let nextArguments := childValues ++ arguments.extract loop.carriedCount arguments.size
      obtain ⟨reachedArguments, reachedFuel, reachedLoopResult, reachedInputs, reachedResult,
          reachedLoop,
          reachedInputsStored, reachedChildStored, leading, trailing, reachedFactor⟩ :=
        ih (arguments := nextArguments) (iteration := iteration + 1) (fuel := fuel - 1)
          (finalResult := rest) (by omega) restStored
      refine ⟨reachedArguments, reachedFuel, reachedLoopResult, reachedInputs, reachedResult, ?_,
        reachedInputsStored, ?_, childResult.scopes ++ leading, trailing, ?_⟩
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using reachedLoop
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using reachedChildStored
      · have outerFactor := congrArg NodeResult.scopes finalStored
        rw [outerFactor, reachedFactor]
        simp [Array.append_assoc]

end IR
end Mxx
