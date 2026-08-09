import Mxx.Ir

namespace Mxx.Ir

/-! Small inversion lemmas for executable IR paths.

The certificate soundness proofs use these facts to recover the concrete value selected at an
SSA node without unfolding an entire generated scope.  No statement in this file assigns
semantics to certificate metadata: every conclusion follows from `evaluateNode` or an
`EvaluatesNodesPath` witness.
-/

theorem lookupWire_append_of_eq_none
    {wire : WireRef} {left right : WireEnvironment}
    (missing : lookupWire wire left = none) :
    lookupWire wire (left ++ right) = lookupWire wire right := by
  induction left with
  | nil => rfl
  | cons head tail induction =>
      rcases head with ⟨candidate, value⟩
      by_cases same : candidate = wire
      · simp [lookupWire, same] at missing
      · simp only [List.cons_append, lookupWire, if_neg same]
        apply induction
        simpa [lookupWire, same] using missing

private theorem lookupWire_zipIdx
    (nodeId start port : Nat) (values : List Value) :
    lookupWire ⟨nodeId, start + port⟩
      (values.zipIdx start |>.map fun (value, index) => (⟨nodeId, index⟩, value)) =
        values[port]? := by
  induction values generalizing start port with
  | nil => simp [lookupWire]
  | cons head tail induction =>
      cases port with
      | zero => simp [lookupWire]
      | succ port =>
          simp only [List.zipIdx, List.map_cons, lookupWire]
          have different :
              (⟨nodeId, start⟩ : WireRef) ≠ ⟨nodeId, start + (port + 1)⟩ := by
            intro same
            have portsEqual := congrArg WireRef.port same
            simp only at portsEqual
            omega
          rw [if_neg different]
          simpa only [List.getElem?_cons_succ, Nat.add_assoc, Nat.add_comm,
            Nat.add_left_comm] using induction (start + 1) port

private theorem lookupWire_zipIdx_node_ne
    (nodeId otherNode start port : Nat)
    (values : List Value)
    (different : otherNode ≠ nodeId) :
    lookupWire ⟨otherNode, port⟩
      (values.zipIdx start |>.map fun (value, index) => (⟨nodeId, index⟩, value)) = none := by
  induction values generalizing start port with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.zipIdx, List.map_cons, lookupWire]
      split
      · rename_i same
        exact (different (congrArg WireRef.node same).symm).elim
      · exact induction (start := start + 1) (port := port)

theorem lookupWire_bindOutputs
    (nodeId port : Nat) (values : List Value) (inBounds : port < values.length) :
    lookupWire ⟨nodeId, port⟩ (bindOutputs nodeId values) = some values[port] := by
  rw [show some values[port] = values[port]? from (List.getElem?_eq_getElem inBounds).symm]
  simpa [bindOutputs] using lookupWire_zipIdx nodeId 0 port values

theorem lookupWire_bindOutputs_eq_none_of_not_lt
    (nodeId port : Nat) (values : List Value) (outOfBounds : ¬port < values.length) :
    lookupWire ⟨nodeId, port⟩ (bindOutputs nodeId values) = none := by
  rw [show lookupWire ⟨nodeId, port⟩ (bindOutputs nodeId values) = values[port]? by
    simpa [bindOutputs] using lookupWire_zipIdx nodeId 0 port values]
  exact List.getElem?_eq_none (by omega)

theorem lookupWire_bindOutputs_of_node_ne
    (nodeId otherNode port : Nat)
    (values : List Value)
    (different : otherNode ≠ nodeId) :
    lookupWire ⟨otherNode, port⟩ (bindOutputs nodeId values) = none := by
  simpa [bindOutputs] using lookupWire_zipIdx_node_ne nodeId otherNode 0 port values different

theorem lookupWire_append_bindOutputs
    {state : WireEnvironment} {nodeId port : Nat} {values : List Value}
    (missing : lookupWire ⟨nodeId, port⟩ state = none)
    (inBounds : port < values.length) :
    lookupWire ⟨nodeId, port⟩ (state ++ bindOutputs nodeId values) = some values[port] := by
  rw [lookupWire_append_of_eq_none missing]
  exact lookupWire_bindOutputs nodeId port values inBounds

/-- A selected path node exposes both its support member and every one of its output ports in the
final SSA environment. -/
theorem EvaluatesNodesPath.outputAtHead
    {runChild : ChildRunner} {samplers : MxxSamplerFamily}
    {params : ParamEnvironment} {inputs : Environment}
    {nodeId : Nat} {node : Node} {nodes : List Node}
    {state output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId
      (node :: nodes) state output)
    (port : Nat)
    (fresh : lookupWire ⟨nodeId, port⟩ state = none) :
    ∃ values,
      values ∈ evaluateNode runChild samplers params inputs state node ∧
      ∀ portValid : port < values.length,
        lookupWire ⟨nodeId, port⟩ output = some values[port] := by
  cases path with
  | cons _ _ _ _ values _ valuesMember tail =>
      refine ⟨values, valuesMember, fun portValid ↦ ?_⟩
      apply tail.lookupWire_preserved
      exact lookupWire_append_bindOutputs fresh portValid

/-- A concrete SSA path cannot bind a wire whose node identifier lies at or after the end of the
path. This discharges output freshness mechanically; callers of certificate soundness never
supply a freshness or per-node soundness callback. -/
theorem EvaluatesNodesPath.lookupWire_after_end
    {runChild : ChildRunner} {samplers : MxxSamplerFamily}
    {params : ParamEnvironment} {inputs : Environment}
    {nodeId : Nat} {nodes : List Node} {state output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId nodes state output)
    (targetNode port : Nat)
    (after : nodeId + nodes.length ≤ targetNode)
    (initialMissing : lookupWire ⟨targetNode, port⟩ state = none) :
    lookupWire ⟨targetNode, port⟩ output = none := by
  induction path with
  | nil => exact initialMissing
  | @cons current node tail currentState values final valuesMember tailPath induction =>
      have tailAfter : current + 1 + tail.length ≤ targetNode := by
        simp only [List.length_cons] at after
        omega
      have different : targetNode ≠ current := by omega
      apply induction tailAfter
      · rw [lookupWire_append_of_eq_none initialMissing]
        exact lookupWire_bindOutputs_of_node_ne current targetNode port values different

/-- A path starting strictly after a wire's node identifier cannot change that wire's lookup.
This is the reverse bridge needed to recover node arguments from the selected final SSA state. -/
theorem EvaluatesNodesPath.lookupWire_before_start
    {runChild : ChildRunner} {samplers : MxxSamplerFamily}
    {params : ParamEnvironment} {inputs : Environment}
    {nodeId : Nat} {nodes : List Node} {state output : WireEnvironment}
    (path : EvaluatesNodesPath runChild samplers params inputs nodeId nodes state output)
    (targetNode port : Nat)
    (before : targetNode < nodeId) :
    lookupWire ⟨targetNode, port⟩ output = lookupWire ⟨targetNode, port⟩ state := by
  induction path with
  | nil => rfl
  | @cons current node tail currentState values final valuesMember tailPath induction =>
      rw [induction (by omega)]
      cases resolved : lookupWire ⟨targetNode, port⟩ currentState with
      | some value =>
          exact lookupWire_append_of_eq_some resolved
      | none =>
          rw [lookupWire_append_of_eq_none resolved]
          exact lookupWire_bindOutputs_of_node_ne current targetNode port values (by omega)

theorem mem_evaluateNode_input
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (name : String) (outputCount : Nat) {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .input name
      arguments := []
      outputCount
    }) :
    values = List.replicate outputCount
      (lookupEnvironment name inputs |>.getD (.invalid s!"missing input {name}")) := by
  simpa [evaluateNode] using member

theorem mem_evaluateNode_matrixAdd_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (leftRef rightRef : WireRef) (left right : Mxx.Matrix) (outputCount : Nat)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
        some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixAdd left right)] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_matrixSubtract_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (leftRef rightRef : WireRef) (left right : Mxx.Matrix) (outputCount : Nat)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
        some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .matrixSubtract
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixSubtract left right)] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_matrixMultiply_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (leftRef rightRef : WireRef) (left right : Mxx.Matrix) (outputCount : Nat)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire => lookupWire wire wires) =
        some [.matrix left, .matrix right])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixMultiply left right)] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_matrixNegate_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (inputRef : WireRef) (input : Mxx.Matrix) (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .matrixNegate
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixNegate input)] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_matrixScale_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (inputRef : WireRef) (input : Mxx.Matrix) (scalar : IntExpr) (evaluatedScalar : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    (scalarEvaluate : scalar.evaluate params = some evaluatedScalar)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .matrixScale scalar
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixScale evaluatedScalar input)] := by
  simpa [evaluateNode, arguments, argumentsEvaluate, scalarEvaluate] using member

theorem mem_evaluateNode_select_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (argumentRefs : List WireRef) (index : Int) (branches : List Value)
    (outputCount : Nat)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) =
      some (.integer index :: branches))
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .select
      arguments := argumentRefs
      outputCount
    }) :
    values = [branches[index.toNat]?.getD (.invalid "Select index out of range")] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_familyPack_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (argumentRefs : List WireRef) (packed : List Value) (outputCount : Nat)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) = some packed)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .familyPack
      arguments := argumentRefs
      outputCount
    }) :
    values = [.family packed] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_familyGetStatic_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (familyRef : WireRef) (family : List Value) (index : IntExpr) (evaluatedIndex : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [familyRef].mapM (fun wire => lookupWire wire wires) =
      some [.family family])
    (indexEvaluate : index.evaluate params = some evaluatedIndex)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .familyGetStatic index
      arguments := [familyRef]
      outputCount
    }) :
    values = [family[evaluatedIndex.toNat]?.getD
      (.invalid "FamilyGetStatic index out of range")] := by
  simpa [evaluateNode, arguments, argumentsEvaluate, indexEvaluate] using member

theorem mem_evaluateNode_familyGetDynamic_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (familyRef indexRef : WireRef) (family : List Value) (index : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [familyRef, indexRef].mapM (fun wire => lookupWire wire wires) =
      some [.family family, .integer index])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .familyGetDynamic
      arguments := [familyRef, indexRef]
      outputCount
    }) :
    values = [family[index.toNat]?.getD (.invalid "FamilyGetDynamic index out of range")] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_uniformResidueSample
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (matrixType : MatrixTypeExpr) (matrixParams : Mxx.SamplerParams) (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .uniformResidueSample matrixType
      arguments := []
      outputCount
    }) :
    ∃ sample ∈ uniformMatrixSupport matrixParams 0 (matrixParams.modulus - 1),
      values = [.matrix sample] := by
  simp only [evaluateNode, matrixTypeEvaluate, List.mem_map] at member
  obtain ⟨sample, sampleMember, rfl⟩ := member
  exact ⟨sample, sampleMember, rfl⟩

theorem mem_evaluateNode_uniformIntervalSample
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (matrixType : MatrixTypeExpr) (minimum maximum : IntExpr)
    (matrixParams : Mxx.SamplerParams) (evaluatedMinimum evaluatedMaximum : Int)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params = some matrixParams)
    (minimumEvaluate : minimum.evaluate params = some evaluatedMinimum)
    (maximumEvaluate : maximum.evaluate params = some evaluatedMaximum)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .uniformIntervalSample matrixType minimum maximum
      arguments := []
      outputCount
    }) :
    ∃ sample ∈ uniformMatrixSupport matrixParams evaluatedMinimum evaluatedMaximum,
      values = [.matrix sample] := by
  simp only [evaluateNode, matrixTypeEvaluate, minimumEvaluate, maximumEvaluate,
    List.mem_map] at member
  obtain ⟨sample, sampleMember, rfl⟩ := member
  exact ⟨sample, sampleMember, rfl⟩

theorem mem_evaluateNode_gaussianSample
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (matrixType : MatrixTypeExpr) (cutoff : IntExpr) (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .gaussianSample matrixType cutoff
      arguments := []
      outputCount
    }) :
    ∃ sample ∈ samplers.gaussianSample matrixParams,
      values = [.matrix (sample.withSamplerParams matrixParams)] := by
  simp only [evaluateNode, matrixTypeEvaluate, List.mem_map] at member
  obtain ⟨sample, sampleMember, rfl⟩ := member
  exact ⟨sample, sampleMember, rfl⟩

theorem mem_evaluateNode_hashSample_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (keyRef : WireRef) (trailingRefs : List WireRef) (key : ByteArray)
    (trailingIntegerValues : List Int) (matrixType : MatrixTypeExpr)
    (variant : Mxx.HashVariant) (tagPrefix : List Nat)
    (tagExpressions tagDecimalExpressions tagU64LeExpressions : List IntExpr)
    (base digitCount : Option IntExpr) (matrixParams : Mxx.SamplerParams)
    (tagValues tagDecimalValues tagU64LeValues : List Int)
    (evaluatedBase evaluatedDigitCount : Option Int) (outputCount : Nat)
    (argumentsEvaluate : (keyRef :: trailingRefs).mapM (fun wire => lookupWire wire wires) =
      some (.bytes key :: trailingIntegerValues.map Value.integer))
    (matrixTypeEvaluate : matrixType.evaluate params (.constant 0) = some matrixParams)
    (tagsEvaluate : tagExpressions.mapM (IntExpr.evaluate params) = some tagValues)
    (decimalTagsEvaluate :
      tagDecimalExpressions.mapM (IntExpr.evaluate params) = some tagDecimalValues)
    (u64TagsEvaluate :
      tagU64LeExpressions.mapM (IntExpr.evaluate params) = some tagU64LeValues)
    (baseEvaluate : evaluateOptionalIntExpr params base = some evaluatedBase)
    (digitCountEvaluate : evaluateOptionalIntExpr params digitCount = some evaluatedDigitCount)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .hashSample matrixType variant tagPrefix tagExpressions tagDecimalExpressions
        tagU64LeExpressions base digitCount
      arguments := keyRef :: trailingRefs
      outputCount
    }) :
    values = [.matrix ((samplers.hashSample {
      params := matrixParams
      key
      variant
      tagPrefix
      tagValues
      tagDecimalValues
      tagU64LeValues
      trailingIntegerTagValues := trailingIntegerValues
      base := evaluatedBase
      digitCount := evaluatedDigitCount
    }).withSamplerParams matrixParams)] := by
  have mapIntegerValues : ∀ integerValues : List Int,
      (integerValues.map Value.integer).mapM (fun value => match value with
        | .integer value => some value
        | _ => none) = some integerValues := by
    intro integerValues
    induction integerValues with
    | nil => rfl
    | cons value tail inductionHypothesis => simp [inductionHypothesis]
  have trailingValuesEvaluate := mapIntegerValues trailingIntegerValues
  have hashArgumentsEvaluate :
      hashArguments (.bytes key :: trailingIntegerValues.map Value.integer) =
        some (key, trailingIntegerValues) := by
    simp only [hashArguments]
    change (trailingIntegerValues.map Value.integer |>.mapM (fun value => match value with
      | Value.integer value => some value
      | _ => none)).bind (fun values => some (key, values)) = some (key, trailingIntegerValues)
    rw [trailingValuesEvaluate]
    rfl
  simpa [evaluateNode, arguments, argumentsEvaluate, hashArgumentsEvaluate, matrixTypeEvaluate,
    tagsEvaluate,
    decimalTagsEvaluate, u64TagsEvaluate, baseEvaluate, digitCountEvaluate,
    trailingValuesEvaluate] using member

/-- Regression fixture: hash operands after the key are retained in their exact order and value. -/
theorem hashArguments_nonempty_integer_tail_fixture (key : ByteArray) :
    hashArguments [.bytes key, .integer 7, .integer (-2)] = some (key, [7, -2]) := by
  rfl

theorem mem_evaluateNode_gadgetDecompose_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (inputRef : WireRef) (input : Mxx.Matrix) (matrixType : MatrixTypeExpr)
    (base digitCount : IntExpr) (small : Bool) (matrixParams : Mxx.SamplerParams)
    (paramsId : Mxx.SamplerParamsId) (evaluatedBase evaluatedDigitCount : Int) (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => lookupWire wire wires) =
      some [.matrix input])
    (matrixTypeEvaluate : matrixType.evaluate params (.constant 0) = some matrixParams)
    (baseEvaluate : base.evaluate params = some evaluatedBase)
    (digitCountEvaluate : digitCount.evaluate params = some evaluatedDigitCount)
    (layoutId : samplers.layoutId matrixParams = some paramsId)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .gadgetDecompose matrixType base small digitCount
      arguments := [inputRef]
      outputCount
    }) :
    values = [.invalid "gadget-decomposition layout is invalid"] ∨
      ∃ output, samplers.gadgetDecompose paramsId matrixParams evaluatedBase small
          evaluatedDigitCount.toNat input = some output ∧
        values = [.matrix (output.withSamplerParams matrixParams)] := by
  simp [evaluateNode, arguments, argumentsEvaluate, matrixTypeEvaluate, baseEvaluate,
    digitCountEvaluate, layoutId] at member
  split at member
  · simp_all
  · split at member <;> simp_all

theorem mem_evaluateNode_trapdoorSample
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (matrixType : MatrixTypeExpr) (cutoff : IntExpr) (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .trapdoorSample matrixType cutoff
      arguments := []
      outputCount
    }) :
    ∃ publicMatrix ∈ samplers.trapdoorSample matrixParams,
      let normalized := publicMatrix.withSamplerParams matrixParams
      values = [.matrix normalized, .trapdoor normalized .sampled] := by
  simp only [evaluateNode, matrixTypeEvaluate, List.mem_map] at member
  obtain ⟨publicMatrix, publicMatrixMember, rfl⟩ := member
  exact ⟨publicMatrix, publicMatrixMember, rfl⟩

theorem mem_evaluateNode_trapdoorPublic_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (trapdoorRef : WireRef) (publicMatrix : Mxx.Matrix) (origin : Mxx.TrapdoorOrigin)
    (outputCount : Nat)
    (argumentsEvaluate : [trapdoorRef].mapM (fun wire => lookupWire wire wires) =
      some [.trapdoor publicMatrix origin])
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .trapdoorPublic
      arguments := [trapdoorRef]
      outputCount
    }) :
    values = [.matrix publicMatrix] := by
  simpa [evaluateNode, arguments, argumentsEvaluate] using member

theorem mem_evaluateNode_preimageSample_of_arguments
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (publicRef trapdoorRef targetRef : WireRef) (publicMatrix target : Mxx.Matrix)
    (matrixType : MatrixTypeExpr) (cutoff : IntExpr) (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (argumentsEvaluate :
      [publicRef, trapdoorRef, targetRef].mapM (fun wire => lookupWire wire wires) =
        some [.matrix publicMatrix, .trapdoor publicMatrix .sampled, .matrix target])
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    {values : List Value}
    (member : values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .preimageSample matrixType cutoff
      arguments := [publicRef, trapdoorRef, targetRef]
      outputCount
    }) :
    ∃ sample ∈ samplers.samplePreimage matrixParams publicMatrix target,
      values = [.matrix (sample.withSamplerParams matrixParams)] := by
  simp [evaluateNode, arguments, argumentsEvaluate, matrixTypeEvaluate] at member
  obtain ⟨sample, sampleMember, rfl⟩ := member
  exact ⟨sample, sampleMember, rfl⟩

/-- Inverting a sequential-loop node produces the exact inductive trace selected by that
execution, with no materialization of the other sampler outcomes. -/
theorem mem_evaluateNode_sequentialLoop_iff_trace
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (definition : String) (count : IntExpr) (indexSlot : Nat)
    (bindings : List (String × IntExpr)) (carriedCount : Nat)
    (argumentRefs : List WireRef) (outputCount : Nat) (argumentValues : List Value)
    (evaluatedCount : Int)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) =
      some argumentValues)
    (countEvaluate : count.evaluate params = some evaluatedCount)
    (final : List Value) :
    final ∈ evaluateNode runChild samplers params inputs wires {
      kind := .sequentialLoop definition count indexSlot bindings carriedCount
      arguments := argumentRefs
      outputCount
    } ↔
      SequentialIterationsTrace runChild definition params indexSlot bindings
        (argumentValues.drop carriedCount) (List.range evaluatedCount.toNat)
        (argumentValues.take carriedCount) final := by
  rw [evaluateNode_sequentialLoop_of_arguments _ _ _ _ _ _ _ _ _ _ _ _ _ _
    argumentsEvaluate countEvaluate]
  rw [mem_evaluateSequentialIterations_iff_exists_trace]
  simp

/-- A parallel-loop result is a family-wrapped accumulator together with the exact iteration
trace that produced the accumulator. -/
theorem mem_evaluateNode_parallelLoop_iff_trace
    (runChild : ChildRunner) (samplers : MxxSamplerFamily)
    (params : ParamEnvironment) (inputs : Environment) (wires : WireEnvironment)
    (definition : String) (count : IntExpr) (indexSlot : Nat)
    (bindings : List (String × IntExpr)) (modes : List LoopInputMode)
    (argumentRefs : List WireRef) (outputCount : Nat) (argumentValues : List Value)
    (evaluatedCount : Int)
    (argumentsEvaluate : argumentRefs.mapM (fun wire => lookupWire wire wires) =
      some argumentValues)
    (countEvaluate : count.evaluate params = some evaluatedCount)
    (values : List Value) :
    values ∈ evaluateNode runChild samplers params inputs wires {
      kind := .parallelLoop definition count indexSlot bindings modes
      arguments := argumentRefs
      outputCount
    } ↔
      ∃ final,
        ParallelIterationsTrace runChild definition params indexSlot bindings modes
          argumentValues (List.range evaluatedCount.toNat)
          (List.replicate outputCount []) final ∧
        values = final.map Value.family := by
  rw [evaluateNode_parallelLoop_of_arguments _ _ _ _ _ _ _ _ _ _ _ _ _ _
    argumentsEvaluate countEvaluate]
  simp only [List.mem_map]
  constructor
  · rintro ⟨final, finalMember, rfl⟩
    obtain ⟨initial, initialMember, trace⟩ :=
      (mem_evaluateParallelIterations_iff_exists_trace _ _ _ _ _ _ _ _ _ _ _).mp finalMember
    simp only [List.mem_singleton] at initialMember
    subst initial
    exact ⟨final, trace, rfl⟩
  · rintro ⟨final, trace, rfl⟩
    refine ⟨final, ?_, rfl⟩
    exact (mem_evaluateParallelIterations_iff_exists_trace _ _ _ _ _ _ _ _ _ _ _).2
      ⟨List.replicate outputCount [], by simp, trace⟩

/-- Every selected child execution in a sequential-loop trace satisfies a step-local predicate.
This is the bridge used by protocol proofs that inspect a nested parallel loop inside each
sequential iteration. -/
theorem SequentialIterationsTrace.everyChild
    {runChild : ChildRunner} {definition : String} {params : ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × IntExpr)} {invariantArguments : List Value}
    (predicate : Nat → ParamEnvironment → List Value → List Value → Prop)
    (holds : ∀ (index : Nat) evaluatedBindings state next,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      next ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments) →
      predicate index
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        (state ++ invariantArguments) next) :
    ∀ {indices initial final},
      SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments indices initial final →
      ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
        childOutputs ∈ runChild definition childParams childInputs ∧
          predicate index childParams childInputs childOutputs := by
  intro indices initial final trace
  induction trace with
  | nil => simp
  | cons iteration _ state evaluatedBindings next _ bindingsEvaluate childMember _ ih =>
      intro queried member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · refine ⟨_, _, next, childMember, ?_⟩
        exact holds _ evaluatedBindings state next bindingsEvaluate childMember
      · exact ih queried member

/-- Every selected child execution in a parallel-loop trace satisfies a step-local predicate.
Together with `SequentialIterationsTrace.everyChild`, this exposes nested loop executions without
expanding the Cartesian product of sampler supports. -/
theorem ParallelIterationsTrace.everyChild
    {runChild : ChildRunner} {definition : String} {params : ParamEnvironment}
    {indexSlot : Nat} {bindings : List (String × IntExpr)} {modes : List LoopInputMode}
    {arguments : List Value}
    (predicate : Nat → ParamEnvironment → List Value → List Value → Prop)
    (holds : ∀ (index : Nat) evaluatedBindings childValues,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) ↦ loopArgument mode index value) →
      predicate index
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) ↦ loopArgument mode index value)
        childValues) :
    ∀ {indices initial final},
      ParallelIterationsTrace runChild definition params indexSlot bindings modes arguments
        indices initial final →
      ∀ index ∈ indices, ∃ childParams childInputs childOutputs,
        childOutputs ∈ runChild definition childParams childInputs ∧
          predicate index childParams childInputs childOutputs := by
  intro indices initial final trace
  induction trace with
  | nil => simp
  | cons iteration _ _ evaluatedBindings childValues _ bindingsEvaluate childMember _ ih =>
      intro queried member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · refine ⟨_, _, childValues, childMember, ?_⟩
        exact holds _ evaluatedBindings childValues bindingsEvaluate childMember
      · exact ih queried member

/-- Port-wise append preserves one selected port when both lists have the same arity. -/
theorem appendPortValues_getElem?_of_length_eq
    {state : List (List Value)}
    {childValues : List Value}
    {port outputCount : Nat}
    {initialValues : List Value}
    {value : Value}
    (stateLength : state.length = outputCount)
    (childLength : childValues.length = outputCount)
    (portInBounds : port < outputCount)
    (statePort : state[port]? = some initialValues)
    (childPort : childValues[port]? = some value) :
    (appendPortValues state childValues)[port]? = some (initialValues ++ [value]) := by
  subst outputCount
  induction state generalizing childValues port initialValues value with
  | nil =>
      simp at portInBounds
  | cons stateHead stateTail induction =>
      cases childValues with
      | nil =>
          simp at childLength
      | cons childHead childTail =>
          cases port with
          | zero =>
              simp only [List.getElem?_cons_zero] at statePort childPort
              have stateEq : stateHead = initialValues := Option.some.inj statePort
              have childEq : childHead = value := Option.some.inj childPort
              subst initialValues
              subst value
              rfl
          | succ port =>
              simp only [List.length_cons] at childLength
              simp only [List.getElem?_cons_succ] at statePort childPort
              simp only [appendPortValues, List.getElem?_cons_succ]
              have tailPortInBounds : port < stateTail.length := by simpa using portInBounds
              exact induction statePort childPort (by omega) tailPortInBounds

/-- Port-wise append preserves the common arity of the accumulator and child output. -/
theorem appendPortValues_length_of_length_eq
    {state : List (List Value)}
    {childValues : List Value}
    {outputCount : Nat}
    (stateLength : state.length = outputCount)
    (childLength : childValues.length = outputCount) :
    (appendPortValues state childValues).length = outputCount := by
  subst outputCount
  induction state generalizing childValues with
  | nil =>
      cases childValues <;> simp_all [appendPortValues]
  | cons stateHead stateTail induction =>
      cases childValues with
      | nil =>
          simp at childLength
      | cons childHead childTail =>
          simp only [List.length_cons] at childLength
          simp only [appendPortValues, List.length_cons]
          have tailLength : childTail.length = stateTail.length := by omega
          exact congrArg Nat.succ (induction tailLength)

/-- With fixed child-output arity, the selected port of a parallel trace contains exactly one
value per iteration, in iteration order.  Each gathered value is tied to the actual child support
member stored by the trace; no replacement runner or child output is accepted. -/
theorem ParallelIterationsTrace.portValues
    {runChild : ChildRunner}
    {definition : String}
    {params : ParamEnvironment}
    {indexSlot : Nat}
    {bindings : List (String × IntExpr)}
    {modes : List LoopInputMode}
    {arguments : List Value}
    {indices : List Nat}
    {initial final : List (List Value)}
    (trace : ParallelIterationsTrace runChild definition params indexSlot bindings modes
      arguments indices initial final)
    (outputCount port : Nat)
    (initialLength : initial.length = outputCount)
    (portInBounds : port < outputCount)
    (childArity : ∀ (index : Nat) evaluatedBindings childValues,
      evaluateBindings ((.loopIndex indexSlot, .integer index) :: params) bindings =
        some evaluatedBindings →
      childValues ∈ runChild definition
        (evaluatedBindings ++ ((.loopIndex indexSlot, .integer index) :: params))
        ((modes.zip arguments).map fun (mode, value) ↦ loopArgument mode index value) →
      childValues.length = outputCount) :
    ∀ {initialValues : List Value}, initial[port]? = some initialValues →
      ∃ gathered,
        final[port]? = some (initialValues ++ gathered) ∧
        gathered.length = indices.length ∧
        ∀ (position : Nat) (positionInBounds : position < indices.length),
          ∃ evaluatedBindings childValues,
            evaluateBindings
                ((.loopIndex indexSlot, .integer indices[position]) :: params) bindings =
              some evaluatedBindings ∧
            childValues ∈ runChild definition
              (evaluatedBindings ++
                ((.loopIndex indexSlot, .integer indices[position]) :: params))
              ((modes.zip arguments).map fun (mode, value) ↦
                loopArgument mode indices[position] value) ∧
            childValues[port]? = gathered[position]? := by
  intro initialValues initialPort
  induction trace generalizing initialValues with
  | nil state =>
      refine ⟨[], ?_, by simp, ?_⟩
      · simpa using initialPort
      · intro position positionInBounds
        simp at positionInBounds
  | @cons index tail state evaluatedBindings childValues traceFinal bindingsEvaluate childMember
      rest induction =>
      have childLength := childArity index evaluatedBindings childValues bindingsEvaluate childMember
      have childPortExists : ∃ value, childValues[port]? = some value := by
        rw [List.getElem?_eq_getElem (by omega)]
        exact ⟨childValues[port], rfl⟩
      obtain ⟨value, childPort⟩ := childPortExists
      have nextLength : (appendPortValues state childValues).length = outputCount :=
        appendPortValues_length_of_length_eq initialLength childLength
      have nextPort : (appendPortValues state childValues)[port]? =
          some (initialValues ++ [value]) :=
        appendPortValues_getElem?_of_length_eq initialLength childLength portInBounds initialPort
          childPort
      obtain ⟨tailValues, finalPort, tailLength, tailEvidence⟩ :=
        induction nextLength nextPort
      refine ⟨value :: tailValues, ?_, by simp [tailLength], ?_⟩
      · simpa [List.append_assoc] using finalPort
      · intro position positionInBounds
        cases position with
        | zero =>
            refine ⟨evaluatedBindings, childValues, ?_, childMember, ?_⟩
            · simpa using bindingsEvaluate
            · simpa using childPort
        | succ position =>
            obtain ⟨tailBindings, tailChildValues, tailBindingsEvaluate, tailChildMember,
              tailPort⟩ := tailEvidence position (by simpa using positionInBounds)
            exact ⟨tailBindings, tailChildValues, by simpa using tailBindingsEvaluate,
              by simpa using tailChildMember, by simpa using tailPort⟩

/-- Membership in a program's root support is exactly one selected SSA execution path.  Keeping
the final wire environment makes certificate-referenced root nodes and outputs available without
unfolding a generated program or relying on generated node numbers. -/
theorem mem_denote_iff_root_path
    (samplers : MxxSamplerFamily) (program : Prog) (params : ParamEnvironment)
    (inputs output : Environment) :
    output ∈ denote samplers program params inputs ↔
      ∃ wires,
        EvaluatesNodesPath
          (childRunnerWithFuel samplers program program.definitions.length)
          samplers params inputs 0 program.root.nodes.toList [] wires ∧
        output = collectOutputs program.root.outputs wires := by
  unfold denote
  rw [denoteScopeWithFuel_succ]
  simp only [List.mem_map]
  constructor
  · rintro ⟨wires, wiresMember, rfl⟩
    obtain ⟨initial, initialMember, path⟩ :=
      (mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).mp wiresMember
    simp only [List.mem_singleton] at initialMember
    subst initial
    exact ⟨wires, path, rfl⟩
  · rintro ⟨wires, path, rfl⟩
    refine ⟨wires, ?_, rfl⟩
    exact (mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).2 ⟨[], by simp, path⟩

/-- Extract the evaluation of an arbitrary certificate-referenced root node.  The caller supplies
the checked node index; no generated node number is embedded in this theorem. -/
theorem rootNodeAt_of_mem_denote
    (samplers : MxxSamplerFamily) (program : Prog) (params : ParamEnvironment)
    (inputs output : Environment) (index : Nat) (inBounds : index < program.root.nodes.size)
    (member : output ∈ denote samplers program params inputs) :
    ∃ before values,
      values ∈ evaluateNode
        (childRunnerWithFuel samplers program program.definitions.length)
        samplers params inputs before program.root.nodes[index] := by
  obtain ⟨wires, path, _⟩ :=
    (mem_denote_iff_root_path samplers program params inputs output).mp member
  obtain ⟨before, values, _, valuesMember, _⟩ :=
    path.atNodeIndex index (by simpa using inBounds)
  exact ⟨before, values, valuesMember⟩

/-- Membership in a named child runner is exactly one execution path through the looked-up child
scope.  This avoids repeatedly unfolding the recursive interpreter in protocol-specific nested
loop proofs. -/
theorem mem_childRunnerWithFuel_succ_iff_path
    (samplers : MxxSamplerFamily) (program : Prog) (fuel : Nat)
    (definition : String) (scope : Scope) (params : ParamEnvironment)
    (inputs values : List Value)
    (definitionFound : lookupDefinition definition program.definitions = some scope) :
    values ∈ childRunnerWithFuel samplers program (fuel + 1) definition params inputs ↔
      ∃ wires,
        EvaluatesNodesPath (childRunnerWithFuel samplers program fuel) samplers params
          (scope.inputNames.zip inputs) 0 scope.nodes.toList [] wires ∧
        values = (collectOutputs scope.outputs wires).map Prod.snd := by
  simp only [childRunnerWithFuel, definitionFound, denoteScopeWithFuel_succ, List.mem_map]
  constructor
  · rintro ⟨environment, environmentMember, rfl⟩
    obtain ⟨wires, wiresMember, rfl⟩ := environmentMember
    obtain ⟨initial, initialMember, path⟩ :=
      (mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).mp wiresMember
    simp only [List.mem_singleton] at initialMember
    subst initial
    exact ⟨wires, path, rfl⟩
  · rintro ⟨wires, path, rfl⟩
    refine ⟨collectOutputs scope.outputs wires, ⟨wires, ?_, rfl⟩, rfl⟩
    exact (mem_evaluateNodes_iff_exists_path _ _ _ _ _ _ _ _).2 ⟨[], by simp, path⟩

end Mxx.Ir
