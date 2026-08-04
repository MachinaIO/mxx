import MxxWe.Certificate.InputInjectionExecutionBridge

namespace MxxWe.Certificate

/-! Same-path producer-loop semantics used by Diamond input preprocessing. -/

/-- Exact two-port trapdoor sampler node selected by a checked operation reference. -/
structure TrapdoorSampleResolution
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef) where
  matrixType : Mxx.Ir.MatrixTypeExpr
  cutoff : Mxx.Ir.IntExpr
  resolved : resolveNode workflow reference.operation = some {
    kind := .trapdoorSample matrixType cutoff
    arguments := []
    outputCount := 2
  }

theorem trapdoorSampleResolution_of_verified
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    (verified : verifyOperationKind workflow reference (fun kind => match kind with
      | .trapdoorSample _ _ => true
      | _ => false) = true)
    (inputsEmpty : reference.inputs = []) (outputsTwo : reference.outputs.length = 2) :
    Nonempty (TrapdoorSampleResolution workflow reference) := by
  unfold verifyOperationKind at verified
  simp only [Bool.and_eq_true] at verified
  rcases verified with ⟨operationVerified, kindVerified⟩
  unfold verifyOperation at operationVerified
  simp only [Bool.and_eq_true] at operationVerified
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at kindVerified
  | some node =>
      rcases node with ⟨kind, arguments, outputCount⟩
      cases kind <;> simp_all
      rename_i matrixType cutoff
      exact ⟨{ matrixType, cutoff, resolved := by simp_all }⟩

/-- Concrete trapdoor sampler support member.  The public and private ports contain the same
normalized public matrix; the private value carries no independently chosen matrix. -/
structure TrapdoorSampleOutcome
    (workflow : Mxx.Ir.Workflow) (reference : OperationRef)
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) where
  matrixParams : Mxx.SamplerParams
  rawPublicMatrix : Mxx.Matrix
  support : rawPublicMatrix ∈ samplers.trapdoorSample matrixParams
  publicMatrix : Mxx.Matrix
  publicMatrixEq : publicMatrix = rawPublicMatrix.withSamplerParams matrixParams
  valuesEq : execution.values = [.matrix publicMatrix, .trapdoor publicMatrix]
  shape : Mxx.Toolkit.MatrixShape publicMatrix matrixParams.modulus
    matrixParams.ringDimension matrixParams.rows matrixParams.columns

theorem trapdoorSampleOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : OperationRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (resolution : TrapdoorSampleResolution workflow reference)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (matrixParams : Mxx.SamplerParams)
    (matrixTypeEvaluate : resolution.matrixType.evaluate params resolution.cutoff =
      some matrixParams) :
    Nonempty (TrapdoorSampleOutcome workflow reference runChild samplers params inputs
      execution) := by
  have nodeResolved := resolution.resolved
  rw [execution.resolved] at nodeResolved
  have nodeEq := Option.some.inj nodeResolved
  have member : execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs
      execution.before {
        kind := .trapdoorSample resolution.matrixType resolution.cutoff
        arguments := []
        outputCount := 2
      } := by simpa [nodeEq] using execution.member
  obtain ⟨rawPublicMatrix, support, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_trapdoorSample runChild samplers params inputs execution.before
      resolution.matrixType resolution.cutoff matrixParams 2 matrixTypeEvaluate member
  let publicMatrix := rawPublicMatrix.withSamplerParams matrixParams
  exact ⟨{
    matrixParams
    rawPublicMatrix
    support
    publicMatrix
    publicMatrixEq := rfl
    valuesEq := by simpa [publicMatrix] using valuesEq
    shape := Mxx.Toolkit.withSamplerParams_shape rawPublicMatrix matrixParams
  }⟩

/-- A fixed-width port append exposes the old accumulator followed by the new child value. -/
private theorem appendPortValues_getElem?
    {accumulated : List (List Mxx.Ir.Value)} {values : List Mxx.Ir.Value}
    (sameLength : accumulated.length = values.length)
    (port : Nat) (portValid : port < accumulated.length) :
    (Mxx.Ir.appendPortValues accumulated values)[port]? =
      some (accumulated[port] ++ [values[port]]) := by
  induction accumulated generalizing values port with
  | nil => simp at portValid
  | cons head tail induction =>
      cases values with
      | nil => simp at sameLength
      | cons value rest =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          cases port with
          | zero => simp [Mxx.Ir.appendPortValues]
          | succ port =>
              simp only [List.length_cons, Nat.succ_lt_succ_iff] at portValid
              simpa [Mxx.Ir.appendPortValues] using induction sameLength port portValid

private theorem appendPortValues_length
    {accumulated : List (List Mxx.Ir.Value)} {values : List Mxx.Ir.Value}
    (sameLength : accumulated.length = values.length) :
    (Mxx.Ir.appendPortValues accumulated values).length = accumulated.length := by
  induction accumulated generalizing values with
  | nil =>
      have valuesEmpty : values = [] := List.eq_nil_of_length_eq_zero sameLength.symm
      subst values
      rfl
  | cons head tail induction =>
      cases values with
      | nil => simp at sameLength
      | cons value rest =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp [Mxx.Ir.appendPortValues, induction sameLength]

/-- Port-wise append of one fixed-width child result preserves the ordered value sequence. -/
private theorem parallelIterationsTrace_portValues_aux
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {modes : List Mxx.Ir.LoopInputMode}
    {arguments : List Mxx.Ir.Value} {indices : List Nat} {portCount : Nat}
    {initial final : List (List Mxx.Ir.Value)}
    (initialCount : initial.length = portCount)
    (trace : Mxx.Ir.ParallelIterationsTrace runChild definition params indexSlot bindings modes
      arguments indices initial final)
    (valueAt : Nat → List Mxx.Ir.Value)
    (valueCount : ∀ (index : Nat), (valueAt index).length = portCount)
    (childExact : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) => Mxx.Ir.loopArgument mode index value) →
      childValues = valueAt index) :
    final = List.ofFn fun port : Fin portCount =>
      initial[port.val]'(by omega) ++
        indices.map fun index =>
          (valueAt index)[port.val]'(by have := valueCount index; omega) := by
  induction trace with
  | nil =>
      simp [← initialCount]
  | cons index tail state evaluatedBindings childValues final bindingsEvaluate childMember rest
      induction =>
      have valuesEq := childExact index evaluatedBindings childValues bindingsEvaluate childMember
      subst childValues
      have appendedCount :
          (Mxx.Ir.appendPortValues state (valueAt index)).length = portCount := by
        rw [appendPortValues_length (by rw [initialCount, valueCount]), initialCount]
      rw [induction appendedCount]
      apply List.ext_getElem
      · simp
      · intro port leftValid rightValid
        have portLt : port < portCount := by simpa using leftValid
        have statePort : port < state.length := by simpa [initialCount] using portLt
        have valuePort : port < (valueAt index).length := by simpa [valueCount] using portLt
        have appendedPort := appendPortValues_getElem?
          (show state.length = (valueAt index).length by rw [initialCount, valueCount])
          port statePort
        have appendedValid : port < (Mxx.Ir.appendPortValues state (valueAt index)).length := by
          simpa [appendedCount] using portLt
        rw [List.getElem?_eq_getElem appendedValid] at appendedPort
        have appendedEq := Option.some.inj appendedPort
        simp [appendedEq]

/-- One ordered family port stores exactly the values emitted at the corresponding parallel-loop
iteration.  This is the multi-port form needed by two-port trapdoor sampling. -/
theorem parallelIterationsTrace_portValues
    {runChild : Mxx.Ir.ChildRunner} {definition : String}
    {params : Mxx.Ir.ParamEnvironment} {indexSlot : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)} {modes : List Mxx.Ir.LoopInputMode}
    {arguments : List Mxx.Ir.Value} {indices : List Nat} {portCount : Nat}
    {final : List (List Mxx.Ir.Value)}
    (trace : Mxx.Ir.ParallelIterationsTrace runChild definition params indexSlot bindings modes
      arguments indices (List.replicate portCount []) final)
    (valueAt : Nat → List Mxx.Ir.Value)
    (valueCount : ∀ (index : Nat), (valueAt index).length = portCount)
    (childExact : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
          some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) => Mxx.Ir.loopArgument mode index value) →
      childValues = valueAt index) :
    final = List.ofFn fun port : Fin portCount =>
      indices.map fun index =>
        (valueAt index)[port.val]'(by have := valueCount index; omega) := by
  simpa using parallelIterationsTrace_portValues_aux (initialCount := by simp)
    trace valueAt valueCount childExact

end MxxWe.Certificate
