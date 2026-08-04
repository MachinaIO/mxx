import MxxWe.Certificate.ExecutionBridge

namespace MxxWe.Certificate

private theorem decoder_verifyScopeSsaOrder_argument_lt
    {scope : Mxx.Ir.Scope} (verified : verifyScopeSsaOrder scope = true)
    (index : Nat) (indexLt : index < scope.nodes.length)
    (argument : Mxx.Ir.WireRef) (member : argument ∈ scope.nodes[index].arguments) :
    argument.node < index := by
  unfold verifyScopeSsaOrder at verified
  simp only [Bool.and_eq_true, List.all_eq_true] at verified
  have nodeChecked := (List.forall_mem_zipIdx'.mp verified.1) index indexLt
  exact of_decide_eq_true (nodeChecked.2 argument member).1

private theorem decoder_resolveStage_mem
    {workflow : Mxx.Ir.Workflow} {name : String} {stage : Mxx.Ir.Stage}
    (found : resolveStage workflow name = some stage) :
    stage ∈ workflow.stages := by
  unfold resolveStage at found
  exact List.mem_of_find?_eq_some found

private theorem decoder_list_index_lt_of_getElem?_eq_some
    {α : Type} {values : List α} {index : Nat} {value : α}
    (resolved : values[index]? = some value) :
    index < values.length := by
  by_contra outOfBounds
  rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at resolved
  contradiction

private theorem decoder_list_getElem_eq_of_getElem?_eq_some
    {α : Type} {values : List α} {index : Nat} {value : α}
    (resolved : values[index]? = some value) (indexLt : index < values.length) :
    values[index] = value := by
  rw [List.getElem?_eq_getElem indexLt] at resolved
  exact Option.some.inj resolved

private theorem decoder_lookupWirePair
    {leftRef rightRef : Mxx.Ir.WireRef} {left right : Mxx.Ir.Value}
    {wires : Mxx.Ir.WireEnvironment}
    (leftLookup : Mxx.Ir.lookupWire leftRef wires = some left)
    (rightLookup : Mxx.Ir.lookupWire rightRef wires = some right) :
    [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
      some [left, right] := by
  simp [leftLookup, rightLookup]

/-! # Exact Diamond decoder execution bridge

This module connects the checked decoder construction trace to the executable IR semantics.
It deliberately does not introduce a certificate-side decoder model: every matrix equation below
is obtained from membership in `Mxx.Ir.evaluateNode` at the exact checked references.
-/

/-- The seven matrix operations fixed by an accepted decoder layout. -/
structure VerifiedDecoderMatrixOperations
    (workflow : Mxx.Ir.Workflow) (layout : DecoderLayout) : Prop where
  oneVector : verifyMatrixBinary workflow layout.oneVector .matrixMultiply = true
  kVector : verifyMatrixBinary workflow layout.kVector .matrixMultiply = true
  decoderVector : verifyMatrixBinary workflow layout.decoderVector .matrixMultiply = true
  oneMinusCircuit : verifyMatrixBinary workflow layout.oneMinusCircuit .matrixSubtract = true
  projectedDifference :
    verifyMatrixBinary workflow layout.projectedDifference .matrixMultiply = true
  kPlusProjection : verifyMatrixBinary workflow layout.kPlusProjection .matrixAdd = true
  residual : verifyMatrixBinary workflow layout.residual .matrixSubtract = true

/-- An accepted full certificate exposes all exact matrix-operation checks in its decoder. -/
theorem VerifiedDiamondLayout.decoderMatrixOperations
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    VerifiedDecoderMatrixOperations workflow certificate.decoder := by
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true] at decoderMatches
  exact {
    oneVector := by aesop
    kVector := by aesop
    decoderVector := by aesop
    oneMinusCircuit := by aesop
    projectedDifference := by aesop
    kPlusProjection := by aesop
    residual := by aesop
  }

private theorem matrixBinaryOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (expected : Mxx.Ir.NodeKind)
    (verified : verifyMatrixBinary workflow reference expected = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs execution.before {
      kind := expected
      arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
      outputCount := 1
    } := by
  have exactNode : resolveNode workflow reference.operation = some {
      kind := expected
      arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
      outputCount := 1
    } := by
    cases resolved : resolveNode workflow reference.operation with
    | none =>
        simp [verifyMatrixBinary, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
    | some node =>
        rcases node with ⟨kind, arguments, outputCount⟩
        simp [verifyMatrixBinary, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
        simp_all [wireRef]
  have executionResolved := execution.resolved
  rw [executionResolved] at exactNode
  have nodeEq := Option.some.inj exactNode
  simpa [nodeEq] using execution.member

/-- Exact concrete values of the checked decoder's matrix chain. -/
structure DecoderMatrixOutcome where
  state : Mxx.Matrix
  onePreimage : Mxx.Matrix
  kPreimage : Mxx.Matrix
  decoderPreimage : Mxx.Matrix
  selectedCircuitVector : Mxx.Matrix
  rDecomposed : Mxx.Matrix
  oneVector : Mxx.Matrix
  kVector : Mxx.Matrix
  decoderVector : Mxx.Matrix
  oneMinusCircuit : Mxx.Matrix
  projectedDifference : Mxx.Matrix
  kPlusProjection : Mxx.Matrix
  residual : Mxx.Matrix
  oneVectorEq : oneVector = Mxx.matrixMultiply state onePreimage
  kVectorEq : kVector = Mxx.matrixMultiply state kPreimage
  decoderVectorEq : decoderVector = Mxx.matrixMultiply state decoderPreimage
  oneMinusCircuitEq :
    oneMinusCircuit = Mxx.matrixSubtract oneVector selectedCircuitVector
  projectedDifferenceEq :
    projectedDifference = Mxx.matrixMultiply oneMinusCircuit rDecomposed
  kPlusProjectionEq : kPlusProjection = Mxx.matrixAdd kVector projectedDifference
  residualEq : residual = Mxx.matrixSubtract decoderVector kPlusProjection

/-- Executing the exact seven checked matrix nodes yields the decoder expression emitted by the
DSL.  The argument lookup equalities come from the single SSA execution path; they cannot be
supplied by certificate data because `ReferencedNodeExecution.member` is the executable witness. -/
theorem decoderMatrixOutcome_of_executions
    {workflow : Mxx.Ir.Workflow} {layout : DecoderLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : VerifiedDecoderMatrixOperations workflow layout)
    (oneExecution : ReferencedNodeExecution workflow layout.oneVector.operation runChild
      samplers params inputs)
    (kExecution : ReferencedNodeExecution workflow layout.kVector.operation runChild
      samplers params inputs)
    (decoderExecution : ReferencedNodeExecution workflow layout.decoderVector.operation runChild
      samplers params inputs)
    (differenceExecution : ReferencedNodeExecution workflow layout.oneMinusCircuit.operation
      runChild samplers params inputs)
    (projectionExecution : ReferencedNodeExecution workflow layout.projectedDifference.operation
      runChild samplers params inputs)
    (sumExecution : ReferencedNodeExecution workflow layout.kPlusProjection.operation runChild
      samplers params inputs)
    (residualExecution : ReferencedNodeExecution workflow layout.residual.operation runChild
      samplers params inputs)
    (state onePreimage kPreimage decoderPreimage selectedCircuitVector rDecomposed : Mxx.Matrix)
    (oneVector kVector decoderVector oneMinusCircuit projectedDifference kPlusProjection
      residual : Mxx.Matrix)
    (oneArguments :
      [wireRef layout.oneVector.left.wire, wireRef layout.oneVector.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire oneExecution.before) =
        some [.matrix state, .matrix onePreimage])
    (kArguments :
      [wireRef layout.kVector.left.wire, wireRef layout.kVector.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire kExecution.before) =
        some [.matrix state, .matrix kPreimage])
    (decoderArguments :
      [wireRef layout.decoderVector.left.wire, wireRef layout.decoderVector.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire decoderExecution.before) =
        some [.matrix state, .matrix decoderPreimage])
    (differenceArguments :
      [wireRef layout.oneMinusCircuit.left.wire,
        wireRef layout.oneMinusCircuit.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire differenceExecution.before) =
        some [.matrix oneVector, .matrix selectedCircuitVector])
    (projectionArguments :
      [wireRef layout.projectedDifference.left.wire,
        wireRef layout.projectedDifference.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire projectionExecution.before) =
        some [.matrix oneMinusCircuit, .matrix rDecomposed])
    (sumArguments :
      [wireRef layout.kPlusProjection.left.wire,
        wireRef layout.kPlusProjection.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire sumExecution.before) =
        some [.matrix kVector, .matrix projectedDifference])
    (residualArguments :
      [wireRef layout.residual.left.wire, wireRef layout.residual.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire residualExecution.before) =
        some [.matrix decoderVector, .matrix kPlusProjection])
    (oneValues : oneExecution.values = [.matrix oneVector])
    (kValues : kExecution.values = [.matrix kVector])
    (decoderValues : decoderExecution.values = [.matrix decoderVector])
    (differenceValues : differenceExecution.values = [.matrix oneMinusCircuit])
    (projectionValues : projectionExecution.values = [.matrix projectedDifference])
    (sumValues : sumExecution.values = [.matrix kPlusProjection])
    (residualValues : residualExecution.values = [.matrix residual]) :
    Nonempty DecoderMatrixOutcome := by
  have oneMember := matrixBinaryOutcome_of_execution .matrixMultiply verified.oneVector
    oneExecution
  have kMember := matrixBinaryOutcome_of_execution .matrixMultiply verified.kVector kExecution
  have decoderMember := matrixBinaryOutcome_of_execution .matrixMultiply verified.decoderVector
    decoderExecution
  have differenceMember := matrixBinaryOutcome_of_execution .matrixSubtract
    verified.oneMinusCircuit differenceExecution
  have projectionMember := matrixBinaryOutcome_of_execution .matrixMultiply
    verified.projectedDifference projectionExecution
  have sumMember := matrixBinaryOutcome_of_execution .matrixAdd verified.kPlusProjection
    sumExecution
  have residualMember := matrixBinaryOutcome_of_execution .matrixSubtract verified.residual
    residualExecution
  have oneEq : oneVector = Mxx.matrixMultiply state onePreimage := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, oneArguments, oneValues] using oneMember
  have kEq : kVector = Mxx.matrixMultiply state kPreimage := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, kArguments, kValues] using kMember
  have decoderEq : decoderVector = Mxx.matrixMultiply state decoderPreimage := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, decoderArguments, decoderValues] using
      decoderMember
  have differenceEq :
      oneMinusCircuit = Mxx.matrixSubtract oneVector selectedCircuitVector := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, differenceArguments, differenceValues] using
      differenceMember
  have projectionEq :
      projectedDifference = Mxx.matrixMultiply oneMinusCircuit rDecomposed := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, projectionArguments, projectionValues] using
      projectionMember
  have sumEq : kPlusProjection = Mxx.matrixAdd kVector projectedDifference := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, sumArguments, sumValues] using sumMember
  have residualEq : residual = Mxx.matrixSubtract decoderVector kPlusProjection := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, residualArguments, residualValues] using
      residualMember
  exact ⟨{
    state
    onePreimage
    kPreimage
    decoderPreimage
    selectedCircuitVector
    rDecomposed
    oneVector
    kVector
    decoderVector
    oneMinusCircuit
    projectedDifference
    kPlusProjection
    residual
    oneVectorEq := oneEq
    kVectorEq := kEq
    decoderVectorEq := decoderEq
    oneMinusCircuitEq := differenceEq
    projectedDifferenceEq := projectionEq
    kPlusProjectionEq := sumEq
    residualEq
  }⟩

/-- The residual output is definitionally the complete executable decoder matrix expression. -/
theorem DecoderMatrixOutcome.residual_expression (outcome : DecoderMatrixOutcome) :
    outcome.residual =
      Mxx.matrixSubtract (Mxx.matrixMultiply outcome.state outcome.decoderPreimage)
        (Mxx.matrixAdd (Mxx.matrixMultiply outcome.state outcome.kPreimage)
          (Mxx.matrixMultiply
            (Mxx.matrixSubtract
              (Mxx.matrixMultiply outcome.state outcome.onePreimage)
              outcome.selectedCircuitVector)
            outcome.rDecomposed)) := by
  rw [outcome.residualEq, outcome.kPlusProjectionEq, outcome.decoderVectorEq,
    outcome.kVectorEq, outcome.projectedDifferenceEq, outcome.oneMinusCircuitEq,
    outcome.oneVectorEq]

/-- Shape-refined form of the executable residual.  These are precisely the non-scalar matrix
types emitted by the Diamond graph; the theorem rules out accidentally using either polynomial
scalar-broadcast branch of `matrixMultiply`. -/
theorem DecoderMatrixOutcome.residual_expression_matrixMul
    (outcome : DecoderMatrixOutcome) (q : Int) (ringDimension stateColumns publicColumns : Nat)
    (stateColumnsNotOne : stateColumns ≠ 1)
    (stateShape :
      Mxx.Toolkit.MatrixShape outcome.state q ringDimension 1 stateColumns)
    (onePreimageShape :
      Mxx.Toolkit.MatrixShape outcome.onePreimage q ringDimension stateColumns publicColumns)
    (kPreimageShape :
      Mxx.Toolkit.MatrixShape outcome.kPreimage q ringDimension stateColumns publicColumns)
    (decoderPreimageShape :
      Mxx.Toolkit.MatrixShape outcome.decoderPreimage q ringDimension stateColumns 1)
    (selectedCircuitShape :
      Mxx.Toolkit.MatrixShape outcome.selectedCircuitVector q ringDimension 1 publicColumns)
    (rDecomposedShape :
      Mxx.Toolkit.MatrixShape outcome.rDecomposed q ringDimension publicColumns 1) :
    outcome.residual =
      Mxx.matrixSubtract (Mxx.matrixMul outcome.state outcome.decoderPreimage)
        (Mxx.matrixAdd (Mxx.matrixMul outcome.state outcome.kPreimage)
          (Mxx.matrixMul
            (Mxx.matrixSubtract (Mxx.matrixMul outcome.state outcome.onePreimage)
              outcome.selectedCircuitVector)
            outcome.rDecomposed)) := by
  rw [outcome.residual_expression]
  rw [Mxx.Toolkit.matrixMultiply_nonscalar outcome.state outcome.decoderPreimage stateShape
    decoderPreimageShape (Or.inr stateColumnsNotOne) (Or.inl stateColumnsNotOne)]
  rw [Mxx.Toolkit.matrixMultiply_nonscalar outcome.state outcome.kPreimage stateShape
    kPreimageShape (Or.inr stateColumnsNotOne) (Or.inl stateColumnsNotOne)]
  rw [Mxx.Toolkit.matrixMultiply_nonscalar outcome.state outcome.onePreimage stateShape
    onePreimageShape (Or.inr stateColumnsNotOne) (Or.inl stateColumnsNotOne)]
  have differenceShape : Mxx.Toolkit.MatrixShape
      (Mxx.matrixSubtract (Mxx.matrixMul outcome.state outcome.onePreimage)
        outcome.selectedCircuitVector) q ringDimension 1 publicColumns := by
    exact Mxx.Toolkit.matrixSubtract_shape _ _
      (Mxx.Toolkit.matrixMul_shape _ _ stateShape onePreimageShape) selectedCircuitShape
  by_cases publicColumnsIsOne : publicColumns = 1
  · subst publicColumns
    rw [Mxx.Toolkit.matrixMultiply_leftScalar
      (Mxx.matrixSubtract (Mxx.matrixMul outcome.state outcome.onePreimage)
        outcome.selectedCircuitVector) outcome.rDecomposed differenceShape rDecomposedShape]
  · rw [Mxx.Toolkit.matrixMultiply_nonscalar
      (Mxx.matrixSubtract (Mxx.matrixMul outcome.state outcome.onePreimage)
        outcome.selectedCircuitVector) outcome.rDecomposed differenceShape rDecomposedShape
      (Or.inr publicColumnsIsOne) (Or.inl publicColumnsIsOne)]

/-- Diamond parameters make the state dimension non-scalar independently of the selected
parameter-search candidate. -/
theorem diamondStateColumns_ne_one (p : MxxWe.DiamondWeParameters) : p.stateColumns ≠ 1 := by
  simp [MxxWe.DiamondWeParameters.stateColumns, MxxWe.DiamondWeParameters.stateRows]

/-- The scalar operations fixed by an accepted decoder layout. -/
structure VerifiedDecoderScalarOperations
    (workflow : Mxx.Ir.Workflow) (layout : DecoderLayout) : Prop where
  extractCoefficient :
    verifyUnaryKind workflow layout.extractCoefficient (fun kind ↦ match kind with
      | .extractCoefficient (.constant 0) => true
      | _ => false) = true
  threshold : verifyEvaluateInt workflow layout.threshold = true
  thresholdExpression : layout.threshold.expression =
    .roundDivide (.subtract (.parameter "diamond_modulus") (.constant 2)) (.constant 4)
  lowerCompare : verifyBinaryKind workflow layout.lowerCompare (fun kind ↦ match kind with
    | .intCompare .lessEqual => true
    | _ => false) = true
  upperScale : verifyBinaryKind workflow layout.upperScale (fun kind ↦ match kind with
    | .intBinary .multiply => true
    | _ => false) = true
  upperCompare : verifyBinaryKind workflow layout.upperCompare (fun kind ↦ match kind with
    | .intCompare .lessEqual => true
    | _ => false) = true
  lowerToInt : verifyUnaryKind workflow layout.lowerToInt (fun kind ↦ match kind with
    | .boolToInt => true
    | _ => false) = true
  upperToInt : verifyUnaryKind workflow layout.upperToInt (fun kind ↦ match kind with
    | .boolToInt => true
    | _ => false) = true
  comparisonSum :
    verifyBinaryKind workflow layout.comparisonSum (fun kind ↦ match kind with
      | .intBinary .add => true
      | _ => false) = true
  equalsTwo : verifyBinaryKind workflow layout.equalsTwo (fun kind ↦ match kind with
    | .intCompare .equal => true
    | _ => false) = true

/-- An accepted full certificate exposes the executable scalar decoder checks as well. -/
theorem VerifiedDiamondLayout.decoderScalarOperations
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    VerifiedDecoderScalarOperations workflow certificate.decoder := by
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true] at decoderMatches
  exact {
    extractCoefficient := by aesop
    threshold := by aesop
    thresholdExpression := by aesop
    lowerCompare := by aesop
    upperScale := by aesop
    upperCompare := by aesop
    lowerToInt := by aesop
    upperToInt := by aesop
    comparisonSum := by aesop
    equalsTwo := by aesop
  }

private theorem unaryOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (expected : Mxx.Ir.NodeKind) (accept : Mxx.Ir.NodeKind → Bool)
    (acceptExact : ∀ kind, accept kind = true → kind = expected)
    (verified : verifyUnaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs execution.before {
      kind := expected
      arguments := [wireRef reference.input.wire]
      outputCount := execution.node.outputCount
    } := by
  have exactNode : resolveNode workflow reference.operation = some {
      kind := expected
      arguments := [wireRef reference.input.wire]
      outputCount := execution.node.outputCount
    } := by
    cases resolved : resolveNode workflow reference.operation with
    | none =>
        simp [verifyUnaryKind, verifyUnaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
    | some node =>
        rcases node with ⟨kind, arguments, outputCount⟩
        simp [verifyUnaryKind, verifyUnaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
        have sameNode : execution.node = {
            kind := kind, arguments := arguments, outputCount := outputCount
          } := by
          rw [execution.resolved] at resolved
          exact Option.some.inj resolved
        have kindEq := acceptExact kind verified.2.1
        rw [sameNode]
        simp_all [wireRef]
  rw [execution.resolved] at exactNode
  have nodeEq := Option.some.inj exactNode
  have member := execution.member
  rw [nodeEq] at member
  exact member

private theorem binaryOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (expected : Mxx.Ir.NodeKind) (accept : Mxx.Ir.NodeKind → Bool)
    (acceptExact : ∀ kind, accept kind = true → kind = expected)
    (verified : verifyBinaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs execution.before {
      kind := expected
      arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
      outputCount := execution.node.outputCount
    } := by
  have exactNode : resolveNode workflow reference.operation = some {
      kind := expected
      arguments := [wireRef reference.left.wire, wireRef reference.right.wire]
      outputCount := execution.node.outputCount
    } := by
    cases resolved : resolveNode workflow reference.operation with
    | none =>
        simp [verifyBinaryKind, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
    | some node =>
        rcases node with ⟨kind, arguments, outputCount⟩
        simp [verifyBinaryKind, verifyBinaryNode, verifyOperationOutput, verifyWire, resolved]
          at verified
        have sameNode : execution.node = {
            kind := kind, arguments := arguments, outputCount := outputCount
          } := by
          rw [execution.resolved] at resolved
          exact Option.some.inj resolved
        have kindEq := acceptExact kind verified.2.1
        rw [sameNode]
        simp_all [wireRef]
  rw [execution.resolved] at exactNode
  have nodeEq := Option.some.inj exactNode
  have member := execution.member
  rw [nodeEq] at member
  exact member

/-- Exact coefficient-zero extraction used by the decoder. -/
theorem extractCoefficientZeroOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyUnaryKind workflow reference (fun kind ↦ match kind with
      | .extractCoefficient (.constant 0) => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (matrix : Mxx.Matrix)
    (argumentsEvaluate : [wireRef reference.input.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some [.matrix matrix]) :
    execution.values = [.integer (matrix.coefficients.getD 0 0)] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .extractCoefficient (.constant 0) => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true →
      kind = .extractCoefficient (.constant 0) := by
    intro kind accepted
    grind
  have member := unaryOutcome_of_execution (.extractCoefficient (.constant 0)) accept
    acceptExact verified execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate,
    Mxx.Ir.IntExpr.evaluate, List.mem_singleton] using member

/-- Exact less-than-or-equal comparison selected by a checked binary reference. -/
theorem lessEqualOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyBinaryKind workflow reference (fun kind ↦ match kind with
      | .intCompare .lessEqual => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Int)
    (argumentsEvaluate : [wireRef reference.left.wire, wireRef reference.right.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer left, .integer right]) :
    execution.values = [.boolean (decide (left ≤ right))] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .intCompare .lessEqual => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .intCompare .lessEqual := by
    intro kind accepted
    grind
  have member := binaryOutcome_of_execution (.intCompare .lessEqual) accept acceptExact
    verified execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate,
    Mxx.Ir.evaluateIntCompare] using member

/-- Exact integer multiplication selected by a checked binary reference. -/
theorem intMultiplyOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyBinaryKind workflow reference (fun kind ↦ match kind with
      | .intBinary .multiply => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Int)
    (argumentsEvaluate : [wireRef reference.left.wire, wireRef reference.right.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer left, .integer right]) :
    execution.values = [.integer (left * right)] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .intBinary .multiply => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .intBinary .multiply := by
    intro kind accepted
    grind
  have member := binaryOutcome_of_execution (.intBinary .multiply) accept acceptExact
    verified execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate,
    Mxx.Ir.evaluateIntBinary] using member

/-- Exact Boolean-to-integer conversion selected by a checked unary reference. -/
theorem boolToIntOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyUnaryKind workflow reference (fun kind ↦ match kind with
      | .boolToInt => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (value : Bool)
    (argumentsEvaluate : [wireRef reference.input.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some [.boolean value]) :
    execution.values = [.integer (if value then 1 else 0)] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .boolToInt => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .boolToInt := by
    intro kind accepted
    grind
  have member := unaryOutcome_of_execution .boolToInt accept acceptExact verified execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member

/-- Exact integer addition selected by a checked binary reference. -/
theorem intAddOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyBinaryKind workflow reference (fun kind ↦ match kind with
      | .intBinary .add => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Int)
    (argumentsEvaluate : [wireRef reference.left.wire, wireRef reference.right.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer left, .integer right]) :
    execution.values = [.integer (left + right)] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .intBinary .add => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .intBinary .add := by
    intro kind accepted
    grind
  have member := binaryOutcome_of_execution (.intBinary .add) accept acceptExact verified
    execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate,
    Mxx.Ir.evaluateIntBinary] using member

/-- Exact integer equality selected by the decoder's final checked reference. -/
theorem intEqualOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyBinaryKind workflow reference (fun kind ↦ match kind with
      | .intCompare .equal => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right : Int)
    (argumentsEvaluate : [wireRef reference.left.wire, wireRef reference.right.wire].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.integer left, .integer right]) :
    execution.values = [.boolean (decide (left = right))] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .intCompare .equal => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .intCompare .equal := by
    intro kind accepted
    grind
  have member := binaryOutcome_of_execution (.intCompare .equal) accept acceptExact verified
    execution
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate,
    Mxx.Ir.evaluateIntCompare] using member

/-- The arithmetic identity used by the emitted scalar decoder after the exact node executions
have supplied the two comparison results. -/
theorem decoded_eq_interval (coefficient threshold lowerInt upperInt sum : Int)
    (lower upper decoded : Bool)
    (lowerEq : lower = decide (threshold ≤ coefficient))
    (upperEq : upper = decide (coefficient ≤ threshold * 3))
    (lowerIntEq : lowerInt = if lower then 1 else 0)
    (upperIntEq : upperInt = if upper then 1 else 0)
    (sumEq : sum = lowerInt + upperInt)
    (decodedEq : decoded = decide (sum = 2)) :
    decoded = (decide (threshold ≤ coefficient) && decide (coefficient ≤ threshold * 3)) := by
  rw [decodedEq, sumEq, lowerIntEq, upperIntEq, lowerEq, upperEq]
  by_cases lowerBound : threshold ≤ coefficient <;>
    by_cases upperBound : coefficient ≤ threshold * 3 <;>
    simp [lowerBound, upperBound]

/-- The threshold expression emitted by the DSL evaluates exactly to integer `q / 4`. -/
theorem decoderThresholdExpression_eq (modulus : Int) :
    Mxx.Ir.roundDiv (modulus - 2) 4 = modulus / 4 := by
  unfold Mxx.Ir.roundDiv
  convert Int.mul_ediv_mul_of_pos (a := 2) modulus 4 (by norm_num) using 1
  all_goals ring_nf

/-- The checked threshold node evaluates to `q / 4` for the modulus in the parameter
environment.  No separately asserted threshold equality is needed. -/
theorem decoderThresholdOutcome_of_execution
    {workflow : Mxx.Ir.Workflow} {layout : DecoderLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : VerifiedDecoderScalarOperations workflow layout)
    (execution : ReferencedNodeExecution workflow layout.threshold.operation runChild samplers
      params inputs)
    (modulus : Int)
    (modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
      some modulus) :
    execution.values = [.integer (modulus / 4)] := by
  have checked := verified.threshold
  have exactNode : ∃ arguments outputCount,
      resolveNode workflow layout.threshold.operation = some {
        kind := .evaluateInt layout.threshold.expression
        arguments
        outputCount
      } := by
    cases resolved : resolveNode workflow layout.threshold.operation with
    | none => simp [verifyEvaluateInt, verifyOperationOutput, verifyWire, resolved] at checked
    | some node =>
        rcases node with ⟨kind, arguments, outputCount⟩
        simp [verifyEvaluateInt, verifyOperationOutput, verifyWire, resolved] at checked
        cases kind <;> simp_all
  obtain ⟨arguments, outputCount, resolved⟩ := exactNode
  rw [execution.resolved] at resolved
  have nodeEq := Option.some.inj resolved
  have member := execution.member
  rw [nodeEq] at member
  have expressionEvaluate : layout.threshold.expression.evaluate params = some (modulus / 4) := by
    rw [verified.thresholdExpression]
    cases lookupEq : Mxx.Ir.lookupParam "diamond_modulus" params with
    | none => simp [Mxx.Ir.IntExpr.evaluate, lookupEq] at modulusEvaluate
    | some value =>
        cases value with
        | integer actual =>
            simp [Mxx.Ir.IntExpr.evaluate, lookupEq] at modulusEvaluate
            subst actual
            simp [Mxx.Ir.IntExpr.evaluate, lookupEq, decoderThresholdExpression_eq]
        | rational value => simp [Mxx.Ir.IntExpr.evaluate, lookupEq] at modulusEvaluate
  simpa [Mxx.Ir.evaluateNode, expressionEvaluate, List.mem_singleton] using member

/-- Once the executable threshold is identified with `q / 4`, the scalar IR chain is exactly the
protocol decoder, not merely a pair of similar comparisons. -/
theorem decoded_eq_decodeBooleanInterval (modulus coefficient threshold lowerInt upperInt sum :
    Int) (lower upper decoded : Bool)
    (thresholdEq : threshold = modulus / 4)
    (lowerEq : lower = decide (threshold ≤ coefficient))
    (upperEq : upper = decide (coefficient ≤ threshold * 3))
    (lowerIntEq : lowerInt = if lower then 1 else 0)
    (upperIntEq : upperInt = if upper then 1 else 0)
    (sumEq : sum = lowerInt + upperInt)
    (decodedEq : decoded = decide (sum = 2)) :
    decoded = MxxWe.decodeBooleanInterval modulus coefficient := by
  rw [decoded_eq_interval coefficient threshold lowerInt upperInt sum lower upper decoded
    lowerEq upperEq lowerIntEq upperIntEq sumEq decodedEq, thresholdEq]
  simp [MxxWe.decodeBooleanInterval, Int.mul_comm]

/-- Execution-derived form of `decoded_eq_decodeBooleanInterval`: the threshold equality is
obtained from the checked threshold node itself. -/
theorem decoded_eq_decodeBooleanInterval_of_execution
    {workflow : Mxx.Ir.Workflow} {layout : DecoderLayout}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : VerifiedDecoderScalarOperations workflow layout)
    (thresholdExecution : ReferencedNodeExecution workflow layout.threshold.operation runChild
      samplers params inputs)
    (modulus coefficient threshold lowerInt upperInt sum : Int) (lower upper decoded : Bool)
    (modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
      some modulus)
    (thresholdValues : thresholdExecution.values = [.integer threshold])
    (lowerEq : lower = decide (threshold ≤ coefficient))
    (upperEq : upper = decide (coefficient ≤ threshold * 3))
    (lowerIntEq : lowerInt = if lower then 1 else 0)
    (upperIntEq : upperInt = if upper then 1 else 0)
    (sumEq : sum = lowerInt + upperInt)
    (decodedEq : decoded = decide (sum = 2)) :
    decoded = MxxWe.decodeBooleanInterval modulus coefficient := by
  have thresholdOutcome := decoderThresholdOutcome_of_execution verified thresholdExecution
    modulus modulusEvaluate
  rw [thresholdValues] at thresholdOutcome
  have thresholdEq : threshold = modulus / 4 := by simpa using thresholdOutcome
  exact decoded_eq_decodeBooleanInterval modulus coefficient threshold lowerInt upperInt sum
    lower upper decoded thresholdEq lowerEq upperEq lowerIntEq upperIntEq sumEq decodedEq

/-- Checker-bound-safe conclusion for the exact executable decoder output.  The residual bound
and congruence are supplied by the input-injection and Boolean execution bridges; this theorem
does not assume an external end-to-end correctness predicate. -/
theorem decoded_eq_message_of_checker_bound
    (p : MxxWe.DiamondWeParameters) (message : Bool)
    (accepted : MxxWe.diamondWeChecker p = true)
    (actualBound : Nat) (actualBoundLe : actualBound ≤ p.finalBound)
    (residual noisy : Mxx.Matrix)
    (residualModulus : residual.modulus = p.modulus)
    (residualBound : Mxx.maxCenteredCoefficientNorm residual ≤ actualBound)
    (noisyCanonical : noisy.coefficients.headD 0 =
      Mxx.reduceCoefficient p.modulus (noisy.coefficients.headD 0))
    (congruent : (noisy.coefficients.headD 0 : ZMod p.modulus) =
      (((if message then (p.modulus : Int) / 2 else 0) +
        Mxx.centeredCoefficient p.modulus (residual.coefficients.headD 0) : Int) :
        ZMod p.modulus))
    (decoded : Bool)
    (decodedValue :
      decoded = MxxWe.decodeBooleanInterval p.modulus (noisy.coefficients.headD 0)) :
    decoded = message := by
  rw [decodedValue]
  exact MxxWe.decodeFromCongruence p message accepted actualBound actualBoundLe residual noisy
    residualModulus residualBound noisyCanonical congruent

private theorem verifiedOperand_stage_scope
    {workflow : Mxx.Ir.Workflow} {reference : CoreOperandRef}
    (verified : verifyOperand workflow reference = true) :
    reference.node.stage = reference.wire.node.stage ∧
      reference.node.scope = reference.wire.node.scope := by
  unfold verifyOperand at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  exact verified.1.1

private theorem verifiedBinaryNode_stage_scopes
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    (verified : verifyBinaryNode workflow reference = true) :
    reference.operation.stage = reference.left.wire.node.stage ∧
      reference.operation.scope = reference.left.wire.node.scope ∧
      reference.operation.stage = reference.right.wire.node.stage ∧
      reference.operation.scope = reference.right.wire.node.scope ∧
      reference.output.node = reference.operation := by
  unfold verifyBinaryNode at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have leftNode : reference.left.node = reference.operation := by aesop
  have rightNode : reference.right.node = reference.operation := by aesop
  have leftVerified : verifyOperand workflow reference.left = true := by aesop
  have rightVerified : verifyOperand workflow reference.right = true := by aesop
  have outputVerified :
      verifyOperationOutput workflow reference.operation reference.output = true := by aesop
  have left := verifiedOperand_stage_scope leftVerified
  have right := verifiedOperand_stage_scope rightVerified
  unfold verifyOperationOutput at outputVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at outputVerified
  rw [leftNode] at left
  rw [rightNode] at right
  exact ⟨left.1, left.2, right.1, right.2, outputVerified.1⟩

private theorem verifiedUnaryNode_stage_scope
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    (verified : verifyUnaryNode workflow reference = true) :
    reference.operation.stage = reference.input.wire.node.stage ∧
      reference.operation.scope = reference.input.wire.node.scope ∧
      reference.output.node = reference.operation := by
  unfold verifyUnaryNode at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have inputNode : reference.input.node = reference.operation := by aesop
  have inputVerified : verifyOperand workflow reference.input = true := by aesop
  have outputVerified :
      verifyOperationOutput workflow reference.operation reference.output = true := by aesop
  have input := verifiedOperand_stage_scope inputVerified
  unfold verifyOperationOutput at outputVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at outputVerified
  rw [inputNode] at input
  exact ⟨input.1, input.2, outputVerified.1⟩

private theorem verifiedMatrixBinary_stage_scopes
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef} {kind : Mxx.Ir.NodeKind}
    (verified : verifyMatrixBinary workflow reference kind = true) :
    reference.operation.stage = reference.left.wire.node.stage ∧
      reference.operation.scope = reference.left.wire.node.scope ∧
      reference.operation.stage = reference.right.wire.node.stage ∧
      reference.operation.scope = reference.right.wire.node.scope ∧
      reference.output.node = reference.operation := by
  unfold verifyMatrixBinary at verified
  simp only [Bool.and_eq_true] at verified
  let binary : BinaryNodeRef := {
    operation := reference.operation
    left := reference.left
    right := reference.right
    output := reference.output
  }
  simpa [binary] using verifiedBinaryNode_stage_scopes (reference := binary) verified.1

private theorem verifiedBinaryKind_stage_scopes
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyBinaryKind workflow reference accept = true) :
    reference.operation.stage = reference.left.wire.node.stage ∧
      reference.operation.scope = reference.left.wire.node.scope ∧
      reference.operation.stage = reference.right.wire.node.stage ∧
      reference.operation.scope = reference.right.wire.node.scope ∧
      reference.output.node = reference.operation := by
  apply verifiedBinaryNode_stage_scopes
  unfold verifyBinaryKind at verified
  simp only [Bool.and_eq_true] at verified
  exact verified.1.1

private theorem verifiedUnaryKind_stage_scope
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyUnaryKind workflow reference accept = true) :
    reference.operation.stage = reference.input.wire.node.stage ∧
      reference.operation.scope = reference.input.wire.node.scope ∧
      reference.output.node = reference.operation := by
  apply verifiedUnaryNode_stage_scope
  unfold verifyUnaryKind at verified
  simp only [Bool.and_eq_true] at verified
  exact verified.1.1

private theorem verifiedEvaluateInt_output_node
    {workflow : Mxx.Ir.Workflow} {reference : EvaluateIntRef}
    (verified : verifyEvaluateInt workflow reference = true) :
    reference.evaluated.node = reference.operation := by
  unfold verifyEvaluateInt at verified
  simp only [Bool.and_eq_true] at verified
  have outputVerified := verified.1.1
  unfold verifyOperationOutput at outputVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at outputVerified
  exact outputVerified.1.1

private theorem verifiedEvaluateInt_stage_scope
    {workflow : Mxx.Ir.Workflow} {reference : EvaluateIntRef}
    (verified : verifyEvaluateInt workflow reference = true) :
    reference.operation.stage = reference.output.node.stage ∧
      reference.operation.scope = reference.output.node.scope := by
  have evaluatedNode := verifiedEvaluateInt_output_node verified
  cases materializationEq : reference.materialization with
  | none =>
      unfold verifyEvaluateInt at verified
      rw [materializationEq] at verified
      simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
      rw [verified.2, evaluatedNode]
      exact ⟨rfl, rfl⟩
  | some materialization =>
      unfold verifyEvaluateInt at verified
      rw [materializationEq] at verified
      simp only [Bool.and_eq_true] at verified
      have materializationCheck := verified.2
      simp only [decide_eq_true_eq] at materializationCheck
      have scopes := verifiedBinaryNode_stage_scopes materializationCheck.1.1.1.1
      have leftWire := materializationCheck.1.1.2
      have outputWire := materializationCheck.2
      rw [outputWire, scopes.2.2.2.2]
      constructor
      · calc
          reference.operation.stage = reference.evaluated.node.stage :=
            congrArg CoreNodeRef.stage evaluatedNode |>.symm
          _ = materialization.left.wire.node.stage :=
            congrArg (fun wire : CoreWireRef ↦ wire.node.stage) leftWire |>.symm
          _ = materialization.operation.stage := scopes.1.symm
      · calc
          reference.operation.scope = reference.evaluated.node.scope :=
            congrArg CoreNodeRef.scope evaluatedNode |>.symm
          _ = materialization.left.wire.node.scope :=
            congrArg (fun wire : CoreWireRef ↦ wire.node.scope) leftWire |>.symm
          _ = materialization.operation.scope := scopes.2.1.symm

/-- Stage and root-scope locations of every executable decoder operation. -/
structure VerifiedDecoderOperationLocations
    (layout : DecoderLayout) (stage : String) : Prop where
  oneVector : layout.oneVector.operation.stage = stage ∧
    layout.oneVector.operation.scope = .root
  kVector : layout.kVector.operation.stage = stage ∧ layout.kVector.operation.scope = .root
  decoderVector : layout.decoderVector.operation.stage = stage ∧
    layout.decoderVector.operation.scope = .root
  oneMinusCircuit : layout.oneMinusCircuit.operation.stage = stage ∧
    layout.oneMinusCircuit.operation.scope = .root
  projectedDifference : layout.projectedDifference.operation.stage = stage ∧
    layout.projectedDifference.operation.scope = .root
  kPlusProjection : layout.kPlusProjection.operation.stage = stage ∧
    layout.kPlusProjection.operation.scope = .root
  residual : layout.residual.operation.stage = stage ∧ layout.residual.operation.scope = .root
  extractCoefficient : layout.extractCoefficient.operation.stage = stage ∧
    layout.extractCoefficient.operation.scope = .root
  threshold : layout.threshold.operation.stage = stage ∧ layout.threshold.operation.scope = .root
  lowerCompare : layout.lowerCompare.operation.stage = stage ∧
    layout.lowerCompare.operation.scope = .root
  upperScale : layout.upperScale.operation.stage = stage ∧
    layout.upperScale.operation.scope = .root
  upperCompare : layout.upperCompare.operation.stage = stage ∧
    layout.upperCompare.operation.scope = .root
  lowerToInt : layout.lowerToInt.operation.stage = stage ∧
    layout.lowerToInt.operation.scope = .root
  upperToInt : layout.upperToInt.operation.stage = stage ∧
    layout.upperToInt.operation.scope = .root
  comparisonSum : layout.comparisonSum.operation.stage = stage ∧
    layout.comparisonSum.operation.scope = .root
  equalsTwo : layout.equalsTwo.operation.stage = stage ∧ layout.equalsTwo.operation.scope = .root

private theorem producer_location_of_consumer_input
    (stage : String) (producer consumer : CoreNodeRef) (output input : CoreWireRef)
    (outputNode : output.node = producer) (inputEq : input = output)
    (consumerInputStage : consumer.stage = input.node.stage)
    (consumerInputScope : consumer.scope = input.node.scope)
    (consumerLocation : consumer.stage = stage ∧ consumer.scope = .root) :
    producer.stage = stage ∧ producer.scope = .root := by
  subst input
  rw [outputNode] at consumerInputStage consumerInputScope
  exact ⟨consumerInputStage.symm.trans consumerLocation.1,
    consumerInputScope.symm.trans consumerLocation.2⟩

/-- The checked `diamond-decoded` output belongs to the root decryption stage. -/
theorem VerifiedDiamondLayout.decoderOutputLocation
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    certificate.decoder.decoded.node.stage = certificate.workflow.decryption.stage ∧
      certificate.decoder.decoded.node.scope = .root := by
  have workflowMatches := verified.workflowMatches
  have decoderMatches := verified.decoderMatches
  have interfaceCheck :
      verifyStageInterface workflow certificate.workflow.decryption = true := by
    unfold verifyWorkflow at workflowMatches
    simp only [Bool.and_eq_true] at workflowMatches
    aesop
  have namedCheck : verifyNamedStageOutput certificate.workflow.decryption
      "diamond-decoded" certificate.decoder.decoded = true := by
    unfold verifyDecoder at decoderMatches
    simp only [Bool.and_eq_true] at decoderMatches
    aesop
  unfold verifyNamedStageOutput at namedCheck
  simp only [List.any_eq_true] at namedCheck
  obtain ⟨stageOutput, stageOutputMember, stageOutputCheck⟩ := namedCheck
  simp only [Bool.and_eq_true, decide_eq_true_eq] at stageOutputCheck
  obtain ⟨_, stageOutputEq⟩ := stageOutputCheck
  have outputCheck : verifyStageOutputLayout workflow
      certificate.workflow.decryption.stage stageOutput = true := by
    unfold verifyStageInterface at interfaceCheck
    split at interfaceCheck
    · contradiction
    · simp only [Bool.and_eq_true] at interfaceCheck
      have allOutputs : certificate.workflow.decryption.outputs.all
          (verifyStageOutputLayout workflow certificate.workflow.decryption.stage) = true := by
        aesop
      exact (List.all_eq_true.mp allOutputs) stageOutput stageOutputMember
  unfold verifyStageOutputLayout at outputCheck
  simp only [Bool.and_eq_true, decide_eq_true_eq] at outputCheck
  rw [← stageOutputEq]
  exact ⟨outputCheck.1.1, outputCheck.1.2⟩

/-- All decoder nodes inherit the checked output's root-stage location through exact SSA
wiring. -/
theorem VerifiedDiamondLayout.decoderOperationLocations
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    VerifiedDecoderOperationLocations certificate.decoder
      certificate.workflow.decryption.stage := by
  let layout := certificate.decoder
  let stage := certificate.workflow.decryption.stage
  have matrix := verified.decoderMatrixOperations
  have scalar := verified.decoderScalarOperations
  have oneScopes := verifiedMatrixBinary_stage_scopes matrix.oneVector
  have kScopes := verifiedMatrixBinary_stage_scopes matrix.kVector
  have decoderScopes := verifiedMatrixBinary_stage_scopes matrix.decoderVector
  have differenceScopes := verifiedMatrixBinary_stage_scopes matrix.oneMinusCircuit
  have projectionScopes := verifiedMatrixBinary_stage_scopes matrix.projectedDifference
  have sumScopes := verifiedMatrixBinary_stage_scopes matrix.kPlusProjection
  have residualScopes := verifiedMatrixBinary_stage_scopes matrix.residual
  have extractScopes := verifiedUnaryKind_stage_scope scalar.extractCoefficient
  have thresholdScopes := verifiedEvaluateInt_stage_scope scalar.threshold
  have lowerScopes := verifiedBinaryKind_stage_scopes scalar.lowerCompare
  have scaleScopes := verifiedBinaryKind_stage_scopes scalar.upperScale
  have upperScopes := verifiedBinaryKind_stage_scopes scalar.upperCompare
  have lowerIntScopes := verifiedUnaryKind_stage_scope scalar.lowerToInt
  have upperIntScopes := verifiedUnaryKind_stage_scope scalar.upperToInt
  have comparisonScopes := verifiedBinaryKind_stage_scopes scalar.comparisonSum
  have equalsScopes := verifiedBinaryKind_stage_scopes scalar.equalsTwo
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true, decide_eq_true_eq] at decoderMatches
  have equalsOutput : layout.equalsTwo.output = layout.decoded := by aesop
  have equalsLeft : layout.equalsTwo.left.wire = layout.comparisonSum.output := by aesop
  have comparisonLeft :
      layout.comparisonSum.left.wire = layout.lowerToInt.output := by aesop
  have comparisonRight :
      layout.comparisonSum.right.wire = layout.upperToInt.output := by aesop
  have lowerIntInput : layout.lowerToInt.input.wire = layout.lowerCompare.output := by aesop
  have upperIntInput : layout.upperToInt.input.wire = layout.upperCompare.output := by aesop
  have lowerLeft : layout.lowerCompare.left.wire = layout.threshold.output := by aesop
  have lowerRight :
      layout.lowerCompare.right.wire = layout.extractCoefficient.output := by aesop
  have upperRight : layout.upperCompare.right.wire = layout.upperScale.output := by aesop
  have extractInput : layout.extractCoefficient.input.wire = layout.residual.output := by aesop
  have residualLeft : layout.residual.left.wire = layout.decoderVector.output := by aesop
  have residualRight : layout.residual.right.wire = layout.kPlusProjection.output := by aesop
  have sumLeft : layout.kPlusProjection.left.wire = layout.kVector.output := by aesop
  have sumRight :
      layout.kPlusProjection.right.wire = layout.projectedDifference.output := by aesop
  have projectionLeft :
      layout.projectedDifference.left.wire = layout.oneMinusCircuit.output := by aesop
  have differenceLeft : layout.oneMinusCircuit.left.wire = layout.oneVector.output := by aesop
  have decodedLocation :
      layout.decoded.node.stage = stage ∧ layout.decoded.node.scope = .root :=
    verified.decoderOutputLocation
  have equalsLocation :
      layout.equalsTwo.operation.stage = stage ∧ layout.equalsTwo.operation.scope = .root := by
    rw [← equalsScopes.2.2.2.2, equalsOutput]
    exact decodedLocation
  have comparisonLocation := producer_location_of_consumer_input stage
    layout.comparisonSum.operation layout.equalsTwo.operation layout.comparisonSum.output
    layout.equalsTwo.left.wire comparisonScopes.2.2.2.2 equalsLeft equalsScopes.1
    equalsScopes.2.1 equalsLocation
  have lowerIntLocation := producer_location_of_consumer_input stage
    layout.lowerToInt.operation layout.comparisonSum.operation layout.lowerToInt.output
    layout.comparisonSum.left.wire lowerIntScopes.2.2 comparisonLeft comparisonScopes.1
    comparisonScopes.2.1 comparisonLocation
  have upperIntLocation := producer_location_of_consumer_input stage
    layout.upperToInt.operation layout.comparisonSum.operation layout.upperToInt.output
    layout.comparisonSum.right.wire upperIntScopes.2.2 comparisonRight
    comparisonScopes.2.2.1 comparisonScopes.2.2.2.1 comparisonLocation
  have lowerLocation := producer_location_of_consumer_input stage
    layout.lowerCompare.operation layout.lowerToInt.operation layout.lowerCompare.output
    layout.lowerToInt.input.wire lowerScopes.2.2.2.2 lowerIntInput lowerIntScopes.1
    lowerIntScopes.2.1 lowerIntLocation
  have upperLocation := producer_location_of_consumer_input stage
    layout.upperCompare.operation layout.upperToInt.operation layout.upperCompare.output
    layout.upperToInt.input.wire upperScopes.2.2.2.2 upperIntInput upperIntScopes.1
    upperIntScopes.2.1 upperIntLocation
  have thresholdLocation :
      layout.threshold.operation.stage = stage ∧ layout.threshold.operation.scope = .root := by
    constructor
    · calc
        layout.threshold.operation.stage = layout.threshold.output.node.stage :=
          thresholdScopes.1
        _ = layout.lowerCompare.left.wire.node.stage :=
          congrArg (fun wire : CoreWireRef ↦ wire.node.stage) lowerLeft |>.symm
        _ = layout.lowerCompare.operation.stage := lowerScopes.1.symm
        _ = stage := lowerLocation.1
    · calc
        layout.threshold.operation.scope = layout.threshold.output.node.scope :=
          thresholdScopes.2
        _ = layout.lowerCompare.left.wire.node.scope :=
          congrArg (fun wire : CoreWireRef ↦ wire.node.scope) lowerLeft |>.symm
        _ = layout.lowerCompare.operation.scope := lowerScopes.2.1.symm
        _ = .root := lowerLocation.2
  have extractLocation := producer_location_of_consumer_input stage
    layout.extractCoefficient.operation layout.lowerCompare.operation
    layout.extractCoefficient.output layout.lowerCompare.right.wire extractScopes.2.2 lowerRight
    lowerScopes.2.2.1 lowerScopes.2.2.2.1 lowerLocation
  have scaleLocation := producer_location_of_consumer_input stage
    layout.upperScale.operation layout.upperCompare.operation layout.upperScale.output
    layout.upperCompare.right.wire scaleScopes.2.2.2.2 upperRight upperScopes.2.2.1
    upperScopes.2.2.2.1 upperLocation
  have residualLocation := producer_location_of_consumer_input stage
    layout.residual.operation layout.extractCoefficient.operation layout.residual.output
    layout.extractCoefficient.input.wire residualScopes.2.2.2.2 extractInput extractScopes.1
    extractScopes.2.1 extractLocation
  have decoderLocation := producer_location_of_consumer_input stage
    layout.decoderVector.operation layout.residual.operation layout.decoderVector.output
    layout.residual.left.wire decoderScopes.2.2.2.2 residualLeft residualScopes.1
    residualScopes.2.1 residualLocation
  have sumLocation := producer_location_of_consumer_input stage
    layout.kPlusProjection.operation layout.residual.operation layout.kPlusProjection.output
    layout.residual.right.wire sumScopes.2.2.2.2 residualRight residualScopes.2.2.1
    residualScopes.2.2.2.1 residualLocation
  have kLocation := producer_location_of_consumer_input stage
    layout.kVector.operation layout.kPlusProjection.operation layout.kVector.output
    layout.kPlusProjection.left.wire kScopes.2.2.2.2 sumLeft sumScopes.1 sumScopes.2.1
    sumLocation
  have projectionLocation := producer_location_of_consumer_input stage
    layout.projectedDifference.operation layout.kPlusProjection.operation
    layout.projectedDifference.output layout.kPlusProjection.right.wire
    projectionScopes.2.2.2.2 sumRight sumScopes.2.2.1 sumScopes.2.2.2.1 sumLocation
  have differenceLocation := producer_location_of_consumer_input stage
    layout.oneMinusCircuit.operation layout.projectedDifference.operation
    layout.oneMinusCircuit.output layout.projectedDifference.left.wire
    differenceScopes.2.2.2.2 projectionLeft projectionScopes.1 projectionScopes.2.1
    projectionLocation
  have oneLocation := producer_location_of_consumer_input stage
    layout.oneVector.operation layout.oneMinusCircuit.operation layout.oneVector.output
    layout.oneMinusCircuit.left.wire oneScopes.2.2.2.2 differenceLeft differenceScopes.1
    differenceScopes.2.1 differenceLocation
  exact {
    oneVector := oneLocation
    kVector := kLocation
    decoderVector := decoderLocation
    oneMinusCircuit := differenceLocation
    projectedDifference := projectionLocation
    kPlusProjection := sumLocation
    residual := residualLocation
    extractCoefficient := extractLocation
    threshold := thresholdLocation
    lowerCompare := lowerLocation
    upperScale := scaleLocation
    upperCompare := upperLocation
    lowerToInt := lowerIntLocation
    upperToInt := upperIntLocation
    comparisonSum := comparisonLocation
    equalsTwo := equalsLocation
  }

private theorem verifyWorkflow_decryptionOutputs
    {workflow : Mxx.Ir.Workflow} {layout : DiamondWorkflowLayout}
    (verified : verifyWorkflow workflow layout = true) :
    layout.decryption.outputs.map (fun output ↦ output.name) =
      ["diamond-decoded", "diamond-noisy-plaintext"] := by
  unfold verifyWorkflow at verified
  split at verified <;> simp_all [Bool.and_eq_true, decide_eq_true_eq]

/-- The concrete decryption stage and exact exported decoder wire recovered from verification. -/
structure VerifiedDecoderStage
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate) where
  stage : Mxx.Ir.Stage
  resolved : resolveStage workflow certificate.workflow.decryption.stage = some stage
  outputNamesUnique : (stage.program.root.outputs.map Prod.fst).Nodup
  decodedOutput : ("diamond-decoded", wireRef certificate.decoder.decoded) ∈
    stage.program.root.outputs

/-- Workflow and decoder verification recover all final-stage facts needed by execution. -/
theorem VerifiedDiamondLayout.decoderStage
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    Nonempty (VerifiedDecoderStage workflow certificate) := by
  have workflowMatches := verified.workflowMatches
  have interfaceCheck :
      verifyStageInterface workflow certificate.workflow.decryption = true := by
    unfold verifyWorkflow at workflowMatches
    simp only [Bool.and_eq_true] at workflowMatches
    aesop
  have outputNames := verifyWorkflow_decryptionOutputs verified.workflowMatches
  have decoderMatches := verified.decoderMatches
  have namedCheck : verifyNamedStageOutput certificate.workflow.decryption
      "diamond-decoded" certificate.decoder.decoded = true := by
    unfold verifyDecoder at decoderMatches
    simp only [Bool.and_eq_true] at decoderMatches
    aesop
  unfold verifyNamedStageOutput at namedCheck
  simp only [List.any_eq_true] at namedCheck
  obtain ⟨stageOutput, stageOutputMember, stageOutputCheck⟩ := namedCheck
  simp only [Bool.and_eq_true, decide_eq_true_eq] at stageOutputCheck
  obtain ⟨stageOutputName, stageOutputWire⟩ := stageOutputCheck
  unfold verifyStageInterface at interfaceCheck
  cases resolved : resolveStage workflow certificate.workflow.decryption.stage with
  | none => simp [resolved] at interfaceCheck
  | some stage =>
      rw [resolved] at interfaceCheck
      simp only [Bool.and_eq_true, decide_eq_true_eq] at interfaceCheck
      have outputsEq : certificate.workflow.decryption.outputs.map
          (fun output ↦ (output.name, wireRef output.wire)) = stage.program.root.outputs := by
        aesop
      have decodedOutput :
          ("diamond-decoded", wireRef certificate.decoder.decoded) ∈
            stage.program.root.outputs := by
        rw [← outputsEq]
        apply List.mem_map.mpr
        exact ⟨stageOutput, stageOutputMember, by simp [stageOutputName, stageOutputWire]⟩
      have stageNames : stage.program.root.outputs.map Prod.fst =
          ["diamond-decoded", "diamond-noisy-plaintext"] := by
        rw [← outputsEq, ← outputNames]
        simp
      exact ⟨{
        stage
        resolved
        outputNamesUnique := by rw [stageNames]; simp
        decodedOutput
      }⟩

private theorem rootReference_inBounds
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef} {stage : Mxx.Ir.Stage}
    (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (nodeResolved : ∃ node, resolveNode workflow reference = some node) :
    reference.node < stage.program.root.nodes.length := by
  obtain ⟨node, nodeResolved⟩ := nodeResolved
  rcases reference with ⟨stageName, scope, index⟩
  dsimp at rootScope
  subst scope
  have nodeAt : stage.program.root.nodes[index]? = some node := by
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using
      nodeResolved
  by_contra outOfBounds
  rw [List.getElem?_eq_none (Nat.le_of_not_gt outOfBounds)] at nodeAt
  contradiction

private theorem verifiedMatrixBinary_resolves
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef} {kind : Mxx.Ir.NodeKind}
    (verified : verifyMatrixBinary workflow reference kind = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyMatrixBinary, verifyBinaryNode, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem verifiedBinaryKind_resolves
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyBinaryKind workflow reference accept = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyBinaryKind, verifyBinaryNode, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem verifiedUnaryKind_resolves
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyUnaryKind workflow reference accept = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyUnaryKind, verifyUnaryNode, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem verifiedEvaluateInt_resolves
    {workflow : Mxx.Ir.Workflow} {reference : EvaluateIntRef}
    (verified : verifyEvaluateInt workflow reference = true) :
    ∃ node, resolveNode workflow reference.operation = some node := by
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [verifyEvaluateInt, verifyOperationOutput, verifyWire, resolved] at verified
  | some node => exact ⟨node, rfl⟩

private theorem intEqualExecution_singleton
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (verified : verifyBinaryKind workflow reference (fun kind ↦ match kind with
      | .intCompare .equal => true
      | _ => false) = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    ∃ value, execution.values = [value] := by
  let accept : Mxx.Ir.NodeKind → Bool := fun kind ↦ match kind with
    | .intCompare .equal => true
    | _ => false
  have acceptExact : ∀ kind, accept kind = true → kind = .intCompare .equal := by
    intro kind accepted
    grind
  have member := binaryOutcome_of_execution (.intCompare .equal) accept acceptExact
    verified execution
  simp only [Mxx.Ir.evaluateNode] at member
  split at member <;> simp_all

/-! ## Same-path decoder execution

The following lemmas retain one root execution path.  This is stronger than independently
inverting the same stage outcome at several indices: sampler collisions could otherwise choose
different internal paths with an equal exported environment.
-/

/-- A selected executable stage outcome together with its unique chosen SSA path witness. -/
structure RootStageExecutionPath
    (samplers : Mxx.MxxSamplerFamily) (stage : Mxx.Ir.Stage)
    (params : Mxx.Ir.ParamEnvironment) (inputs output : Mxx.Ir.Environment) where
  finalWires : Mxx.Ir.WireEnvironment
  path : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs 0 stage.program.root.nodes [] finalWires
  outputEq : output = Mxx.Ir.collectOutputs stage.program.root.outputs finalWires

/-- Every concrete stage outcome has one retained root path. -/
theorem rootStageExecutionPath_of_member
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    Nonempty (RootStageExecutionPath samplers stage params inputs output) := by
  obtain ⟨finalWires, path, outputEq⟩ :=
    (Mxx.Ir.mem_denote_iff_root_path samplers stage.program params inputs output).mp member
  exact ⟨{ finalWires, path, outputEq }⟩

/-- An extracted node execution remains tied to the root path that selected it. -/
structure RootedNodeExecution
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs) : Prop where
  beforeFinal : ∀ wire value,
    Mxx.Ir.lookupWire wire execution.before = some value →
      Mxx.Ir.lookupWire wire rootPath.finalWires = some value
  finalBefore : ∀ wire value, wire.node < reference.node →
    Mxx.Ir.lookupWire wire rootPath.finalWires = some value →
      Mxx.Ir.lookupWire wire execution.before = some value
  outputFinal : ∀ port, ∀ portValid : port < execution.values.length,
    Mxx.Ir.lookupWire { node := reference.node, port } rootPath.finalWires =
      some (execution.values.get ⟨port, portValid⟩)

private theorem lookupWire_zipIdx_of_node_ne
    (query : Mxx.Ir.WireRef) (nodeId start : Nat) (values : List Mxx.Ir.Value)
    (different : query.node ≠ nodeId) :
    Mxx.Ir.lookupWire query
      (values.zipIdx start |>.map fun (value, port) ↦ (⟨nodeId, port⟩, value)) = none := by
  induction values generalizing start with
  | nil => simp [Mxx.Ir.lookupWire]
  | cons head tail induction =>
      simp only [List.zipIdx, List.map_cons, Mxx.Ir.lookupWire]
      rw [if_neg]
      · exact induction (start + 1)
      · intro equal
        exact different (congrArg Mxx.Ir.WireRef.node equal).symm

private theorem decoderLookupWire_bindOutputs_of_node_ne
    (query : Mxx.Ir.WireRef) (nodeId : Nat) (values : List Mxx.Ir.Value)
    (different : query.node ≠ nodeId) :
    Mxx.Ir.lookupWire query (Mxx.Ir.bindOutputs nodeId values) = none := by
  exact lookupWire_zipIdx_of_node_ne query nodeId 0 values different

/-- A path starting at `nodeId` cannot create a binding whose node id is at or beyond the first
id after that path. -/
private theorem Mxx.Ir.EvaluatesNodesPath.futureWire_missing
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodeId : Nat} {nodes : List Mxx.Ir.Node}
    {initial output : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (query : Mxx.Ir.WireRef) (future : nodeId + nodes.length ≤ query.node)
    (initialMissing : Mxx.Ir.lookupWire query initial = none) :
    Mxx.Ir.lookupWire query output = none := by
  induction path with
  | nil => exact initialMissing
  | cons current node tail state values final member rest induction =>
      have tailFuture : current + 1 + tail.length ≤ query.node := by
        simp only [List.length_cons] at future
        omega
      have different : query.node ≠ current := by omega
      apply induction tailFuture
      rw [Mxx.Ir.lookupWire_append_of_eq_none initialMissing]
      exact decoderLookupWire_bindOutputs_of_node_ne query current values different

/-- A path containing only later SSA nodes cannot change an earlier wire lookup. -/
private theorem decoderPastWirePreserved
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodeId : Nat} {nodes : List Mxx.Ir.Node}
    {initial output : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId nodes initial output)
    (query : Mxx.Ir.WireRef) (past : query.node < nodeId) :
    Mxx.Ir.lookupWire query output = Mxx.Ir.lookupWire query initial := by
  induction path with
  | nil => rfl
  | cons current node tail state values final member rest induction =>
      rw [induction (by omega)]
      cases lookup : Mxx.Ir.lookupWire query state with
      | none =>
          rw [Mxx.Ir.lookupWire_append_of_eq_none lookup]
          exact decoderLookupWire_bindOutputs_of_node_ne query current values (by omega)
      | some value =>
          exact Mxx.Ir.lookupWire_append_of_eq_some lookup

/-- Invert one checked root reference from the retained stage path and keep the fact that every
selected output port survives in the stage's final SSA environment. -/
theorem RootStageExecutionPath.referencedRootNodeExecution
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (reference : CoreNodeRef) (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (inBounds : reference.node < stage.program.root.nodes.length) :
    ∃ execution : ReferencedNodeExecution workflow reference
        (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
        samplers params inputs,
      RootedNodeExecution rootPath execution := by
  have resolved :
      resolveNode workflow reference = some stage.program.root.nodes[reference.node] := by
    rcases reference with ⟨stageName, scope, node⟩
    dsimp at rootScope
    subst scope
    simp [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved,
      List.getElem?_eq_getElem inBounds]
  obtain ⟨before, values, prefixPath, member, suffixPath⟩ :=
    rootPath.path.atNodeIndex reference.node inBounds
  have fresh : ∀ port,
      Mxx.Ir.lookupWire { node := reference.node, port } before = none := by
    intro port
    apply Mxx.Ir.EvaluatesNodesPath.futureWire_missing (path := prefixPath)
      (query := { node := reference.node, port })
    · simp [List.length_take]
    · simp [Mxx.Ir.lookupWire]
  let execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs := {
    node := stage.program.root.nodes[reference.node]
    before
    values
    resolved
    member
  }
  refine ⟨execution, {
    beforeFinal := ?_
    finalBefore := ?_
    outputFinal := ?_
  }⟩
  · intro wire value lookup
    apply suffixPath.lookupWire_preserved
    exact Mxx.Ir.lookupWire_append_of_eq_some lookup
  · intro wire value earlier finalLookup
    have suffixLookup := decoderPastWirePreserved suffixPath wire (by omega)
    rw [suffixLookup] at finalLookup
    have different : wire.node ≠ reference.node := by omega
    cases beforeLookup : Mxx.Ir.lookupWire wire before with
    | none =>
        rw [Mxx.Ir.lookupWire_append_of_eq_none beforeLookup] at finalLookup
        have bindMissing := decoderLookupWire_bindOutputs_of_node_ne wire
          (0 + reference.node) values (by omega)
        rw [bindMissing] at finalLookup
        contradiction
    | some actual =>
        rw [Mxx.Ir.lookupWire_append_of_eq_some beforeLookup] at finalLookup
        have valueEq := Option.some.inj finalLookup
        subst actual
        rfl
  · intro port portValid
    apply suffixPath.lookupWire_preserved
    simpa [execution] using Mxx.Ir.lookupWire_append_bindOutputs (fresh port) portValid

private theorem RootStageExecutionPath.referencedRootNodeExecution_of_location
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (reference : CoreNodeRef) (stageName : String)
    (location : reference.stage = stageName ∧ reference.scope = .root)
    (stageResolved : resolveStage workflow stageName = some stage)
    (nodeResolved : ∃ node, resolveNode workflow reference = some node) :
    ∃ execution : ReferencedNodeExecution workflow reference
        (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
        samplers params inputs,
      RootedNodeExecution rootPath execution := by
  have referenceStageResolved : resolveStage workflow reference.stage = some stage := by
    rw [location.1]
    exact stageResolved
  apply rootPath.referencedRootNodeExecution reference location.2 referenceStageResolved
  exact rootReference_inBounds location.2 referenceStageResolved nodeResolved

/-- Two checked root references are inverted on the same path.  When `producer` precedes
`consumer`, every producer port is already available in the consumer's exact prefix state. -/
structure SamePathNodePair
    (samplers : Mxx.MxxSamplerFamily) (stage : Mxx.Ir.Stage)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (producer consumer : CoreNodeRef) where
  producerNode : Mxx.Ir.Node
  consumerNode : Mxx.Ir.Node
  producerBefore : Mxx.Ir.WireEnvironment
  consumerBefore : Mxx.Ir.WireEnvironment
  producerValues : List Mxx.Ir.Value
  consumerValues : List Mxx.Ir.Value
  producerResolved : stage.program.root.nodes[producer.node]? = some producerNode
  consumerResolved : stage.program.root.nodes[consumer.node]? = some consumerNode
  producerMember : producerValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs producerBefore producerNode
  consumerMember : consumerValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs consumerBefore consumerNode
  producerAvailable : ∀ port, ∀ portValid : port < producerValues.length,
    Mxx.Ir.lookupWire { node := producer.node, port } consumerBefore =
      some (producerValues.get ⟨port, portValid⟩)

/-- Construct coherent producer/consumer executions from one retained stage path. -/
theorem samePathNodePair_of_path
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (producer consumer : CoreNodeRef)
    (ordered : producer.node < consumer.node)
    (consumerValid : consumer.node < stage.program.root.nodes.length) :
    Nonempty (SamePathNodePair samplers stage params inputs producer consumer) := by
  have producerValid : producer.node < stage.program.root.nodes.length :=
    Nat.lt_trans ordered consumerValid
  obtain ⟨consumerBefore, consumerValues, prefixPath, consumerMember, _⟩ :=
    rootPath.path.atNodeIndex consumer.node consumerValid
  have producerPrefixValid : producer.node < (stage.program.root.nodes.take consumer.node).length :=
    by simpa [List.length_take, Nat.min_eq_left (Nat.le_of_lt consumerValid)] using ordered
  obtain ⟨producerBefore, producerValues, producerPrefixPath, producerMember, producerTail⟩ :=
    prefixPath.atNodeIndex producer.node producerPrefixValid
  have producerFresh : ∀ port,
      Mxx.Ir.lookupWire { node := producer.node, port } producerBefore = none := by
    intro port
    apply Mxx.Ir.EvaluatesNodesPath.futureWire_missing (path := producerPrefixPath)
      (query := { node := producer.node, port })
    · simp [List.length_take]
    · simp [Mxx.Ir.lookupWire]
  have producerAvailable : ∀ port, ∀ portValid : port < producerValues.length,
      Mxx.Ir.lookupWire { node := producer.node, port } consumerBefore =
        some (producerValues.get ⟨port, portValid⟩) := by
    intro port portValid
    apply producerTail.lookupWire_preserved
    simpa using Mxx.Ir.lookupWire_append_bindOutputs (producerFresh port) portValid
  exact ⟨{
    producerNode := stage.program.root.nodes[producer.node]
    consumerNode := stage.program.root.nodes[consumer.node]
    producerBefore
    consumerBefore
    producerValues
    consumerValues
    producerResolved := List.getElem?_eq_getElem producerValid
    consumerResolved := List.getElem?_eq_getElem consumerValid
    producerMember := by
      simpa [List.getElem_take] using producerMember
    consumerMember
    producerAvailable
  }⟩

/-- The producer selected from a coherent root path is an ordinary referenced execution. -/
def SamePathNodePair.producerExecution
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {producer consumer : CoreNodeRef}
    (pair : SamePathNodePair samplers stage params inputs producer consumer)
    (producerRoot : producer.scope = .root)
    (stageResolved : resolveStage workflow producer.stage = some stage) :
    ReferencedNodeExecution workflow producer
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs := by
  have resolved : resolveNode workflow producer = some pair.producerNode := by
    rcases producer with ⟨stageName, scope, node⟩
    dsimp at producerRoot
    subst scope
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using
      pair.producerResolved
  exact {
    node := pair.producerNode
    before := pair.producerBefore
    values := pair.producerValues
    resolved
    member := pair.producerMember
  }

/-- The consumer selected from the same coherent root path is an ordinary referenced execution. -/
def SamePathNodePair.consumerExecution
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {producer consumer : CoreNodeRef}
    (pair : SamePathNodePair samplers stage params inputs producer consumer)
    (consumerRoot : consumer.scope = .root)
    (stageResolved : resolveStage workflow consumer.stage = some stage) :
    ReferencedNodeExecution workflow consumer
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs := by
  have resolved : resolveNode workflow consumer = some pair.consumerNode := by
    rcases consumer with ⟨stageName, scope, node⟩
    dsimp at consumerRoot
    subst scope
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using
      pair.consumerResolved
  exact {
    node := pair.consumerNode
    before := pair.consumerBefore
    values := pair.consumerValues
    resolved
    member := pair.consumerMember
  }

private theorem lookupEnvironment_collectOutputs_of_mem
    (outputs : List (String × Mxx.Ir.WireRef)) (wires : Mxx.Ir.WireEnvironment)
    (name : String) (wire : Mxx.Ir.WireRef)
    (namesUnique : (outputs.map Prod.fst).Nodup)
    (named : (name, wire) ∈ outputs) (value : Mxx.Ir.Value)
    (wireValue : Mxx.Ir.lookupWire wire wires = some value) :
    Mxx.Ir.lookupEnvironment name (Mxx.Ir.collectOutputs outputs wires) = some value := by
  induction outputs with
  | nil => simp at named
  | cons head tail induction =>
      rcases head with ⟨headName, headWire⟩
      simp only [List.map_cons, List.nodup_cons] at namesUnique
      obtain ⟨headFresh, tailUnique⟩ := namesUnique
      simp only [List.mem_cons, Prod.mk.injEq] at named
      rcases named with ⟨rfl, rfl⟩ | named
      · simp [Mxx.Ir.collectOutputs, Mxx.Ir.lookupEnvironment, wireValue]
      · by_cases same : headName = name
        · subst headName
          have nameMember : name ∈ tail.map Prod.fst := by
            exact List.mem_map.mpr ⟨(name, wire), named, rfl⟩
          exact (headFresh nameMember).elim
        · simp [Mxx.Ir.collectOutputs, Mxx.Ir.lookupEnvironment, same]
          exact induction tailUnique named

/-- An exported stage output is the exact final SSA value at its checked root wire. -/
theorem rootStageExecutionPath.lookup_exported_output
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (name : String) (wire : Mxx.Ir.WireRef)
    (namesUnique : (stage.program.root.outputs.map Prod.fst).Nodup)
    (named : (name, wire) ∈ stage.program.root.outputs)
    (value : Mxx.Ir.Value)
    (wireValue : Mxx.Ir.lookupWire wire rootPath.finalWires = some value) :
    Mxx.Ir.lookupEnvironment name output = some value := by
  rw [rootPath.outputEq]
  exact lookupEnvironment_collectOutputs_of_mem stage.program.root.outputs
    rootPath.finalWires name wire namesUnique named value wireValue

/-- All decoder executions selected from one concrete decryption-stage SSA path. -/
structure DecoderStageExecutionBundle
    (workflow : Mxx.Ir.Workflow) (certificate : DiamondCertificate)
    (samplers : Mxx.MxxSamplerFamily) (stage : Mxx.Ir.Stage)
    (params : Mxx.Ir.ParamEnvironment) (inputs output : Mxx.Ir.Environment) where
  rootPath : RootStageExecutionPath samplers stage params inputs output
  oneVector : ReferencedNodeExecution workflow certificate.decoder.oneVector.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  oneVectorRooted : RootedNodeExecution rootPath oneVector
  kVector : ReferencedNodeExecution workflow certificate.decoder.kVector.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  kVectorRooted : RootedNodeExecution rootPath kVector
  decoderVector : ReferencedNodeExecution workflow certificate.decoder.decoderVector.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  decoderVectorRooted : RootedNodeExecution rootPath decoderVector
  oneMinusCircuit : ReferencedNodeExecution
    workflow certificate.decoder.oneMinusCircuit.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  oneMinusCircuitRooted : RootedNodeExecution rootPath oneMinusCircuit
  projectedDifference : ReferencedNodeExecution
    workflow certificate.decoder.projectedDifference.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  projectedDifferenceRooted : RootedNodeExecution rootPath projectedDifference
  kPlusProjection : ReferencedNodeExecution workflow certificate.decoder.kPlusProjection.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  kPlusProjectionRooted : RootedNodeExecution rootPath kPlusProjection
  residual : ReferencedNodeExecution workflow certificate.decoder.residual.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  residualRooted : RootedNodeExecution rootPath residual
  extractCoefficient : ReferencedNodeExecution
    workflow certificate.decoder.extractCoefficient.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  extractCoefficientRooted : RootedNodeExecution rootPath extractCoefficient
  threshold : ReferencedNodeExecution workflow certificate.decoder.threshold.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  thresholdRooted : RootedNodeExecution rootPath threshold
  lowerCompare : ReferencedNodeExecution workflow certificate.decoder.lowerCompare.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  lowerCompareRooted : RootedNodeExecution rootPath lowerCompare
  upperScale : ReferencedNodeExecution workflow certificate.decoder.upperScale.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  upperScaleRooted : RootedNodeExecution rootPath upperScale
  upperCompare : ReferencedNodeExecution workflow certificate.decoder.upperCompare.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  upperCompareRooted : RootedNodeExecution rootPath upperCompare
  lowerToInt : ReferencedNodeExecution workflow certificate.decoder.lowerToInt.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  lowerToIntRooted : RootedNodeExecution rootPath lowerToInt
  upperToInt : ReferencedNodeExecution workflow certificate.decoder.upperToInt.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  upperToIntRooted : RootedNodeExecution rootPath upperToInt
  comparisonSum : ReferencedNodeExecution workflow certificate.decoder.comparisonSum.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  comparisonSumRooted : RootedNodeExecution rootPath comparisonSum
  equalsTwo : ReferencedNodeExecution workflow certificate.decoder.equalsTwo.operation
    (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
    samplers params inputs
  equalsTwoRooted : RootedNodeExecution rootPath equalsTwo
  decodedValue : Mxx.Ir.Value
  equalsTwoValues : equalsTwo.values = [decodedValue]
  exportedDecoded : Mxx.Ir.lookupEnvironment "diamond-decoded" output = some decodedValue

/-- A concrete decryption-stage member selects every decoder node from one retained path. -/
theorem decoderStageExecutionBundle_of_member
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (member : output ∈ Mxx.Ir.denote samplers stage.program params inputs) :
    Nonempty (DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output) := by
  obtain ⟨rootPath⟩ := rootStageExecutionPath_of_member member
  have locations := verified.decoderOperationLocations
  have matrix := verified.decoderMatrixOperations
  have scalar := verified.decoderScalarOperations
  obtain ⟨oneVector, oneVectorRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.oneVector.operation certificate.workflow.decryption.stage
    locations.oneVector stageResolved (verifiedMatrixBinary_resolves matrix.oneVector)
  obtain ⟨kVector, kVectorRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.kVector.operation certificate.workflow.decryption.stage
    locations.kVector stageResolved (verifiedMatrixBinary_resolves matrix.kVector)
  obtain ⟨decoderVector, decoderVectorRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.decoderVector.operation certificate.workflow.decryption.stage
    locations.decoderVector stageResolved (verifiedMatrixBinary_resolves matrix.decoderVector)
  obtain ⟨oneMinusCircuit, oneMinusCircuitRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.oneMinusCircuit.operation certificate.workflow.decryption.stage
    locations.oneMinusCircuit stageResolved
    (verifiedMatrixBinary_resolves matrix.oneMinusCircuit)
  obtain ⟨projectedDifference, projectedDifferenceRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.projectedDifference.operation certificate.workflow.decryption.stage
    locations.projectedDifference stageResolved
    (verifiedMatrixBinary_resolves matrix.projectedDifference)
  obtain ⟨kPlusProjection, kPlusProjectionRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.kPlusProjection.operation certificate.workflow.decryption.stage
    locations.kPlusProjection stageResolved
    (verifiedMatrixBinary_resolves matrix.kPlusProjection)
  obtain ⟨residual, residualRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.residual.operation certificate.workflow.decryption.stage
    locations.residual stageResolved (verifiedMatrixBinary_resolves matrix.residual)
  obtain ⟨extractCoefficient, extractCoefficientRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.extractCoefficient.operation certificate.workflow.decryption.stage
    locations.extractCoefficient stageResolved
    (verifiedUnaryKind_resolves scalar.extractCoefficient)
  obtain ⟨threshold, thresholdRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.threshold.operation certificate.workflow.decryption.stage
    locations.threshold stageResolved (verifiedEvaluateInt_resolves scalar.threshold)
  obtain ⟨lowerCompare, lowerCompareRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.lowerCompare.operation certificate.workflow.decryption.stage
    locations.lowerCompare stageResolved (verifiedBinaryKind_resolves scalar.lowerCompare)
  obtain ⟨upperScale, upperScaleRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.upperScale.operation certificate.workflow.decryption.stage
    locations.upperScale stageResolved (verifiedBinaryKind_resolves scalar.upperScale)
  obtain ⟨upperCompare, upperCompareRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.upperCompare.operation certificate.workflow.decryption.stage
    locations.upperCompare stageResolved (verifiedBinaryKind_resolves scalar.upperCompare)
  obtain ⟨lowerToInt, lowerToIntRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.lowerToInt.operation certificate.workflow.decryption.stage
    locations.lowerToInt stageResolved (verifiedUnaryKind_resolves scalar.lowerToInt)
  obtain ⟨upperToInt, upperToIntRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.upperToInt.operation certificate.workflow.decryption.stage
    locations.upperToInt stageResolved (verifiedUnaryKind_resolves scalar.upperToInt)
  obtain ⟨comparisonSum, comparisonSumRooted⟩ :=
    rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.comparisonSum.operation certificate.workflow.decryption.stage
    locations.comparisonSum stageResolved (verifiedBinaryKind_resolves scalar.comparisonSum)
  obtain ⟨equalsTwo, equalsTwoRooted⟩ := rootPath.referencedRootNodeExecution_of_location
    certificate.decoder.equalsTwo.operation certificate.workflow.decryption.stage
    locations.equalsTwo stageResolved (verifiedBinaryKind_resolves scalar.equalsTwo)
  obtain ⟨decodedValue, equalsTwoValues⟩ := intEqualExecution_singleton scalar.equalsTwo
    equalsTwo
  obtain ⟨stageFacts⟩ := verified.decoderStage
  have stageEq : stage = stageFacts.stage := by
    have stageFactsResolved := stageFacts.resolved
    rw [stageResolved] at stageFactsResolved
    exact Option.some.inj stageFactsResolved
  have outputNamesUnique : (stage.program.root.outputs.map Prod.fst).Nodup := by
    rw [stageEq]
    exact stageFacts.outputNamesUnique
  have decodedOutput :
      ("diamond-decoded", wireRef certificate.decoder.decoded) ∈
        stage.program.root.outputs := by
    rw [stageEq]
    exact stageFacts.decodedOutput
  have equalsPort : certificate.decoder.equalsTwo.output.port = 0 := by
    have equalsVerified := scalar.equalsTwo
    unfold verifyBinaryKind at equalsVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at equalsVerified
    exact equalsVerified.1.2
  have decodedWire : wireRef certificate.decoder.decoded = {
      node := certificate.decoder.equalsTwo.operation.node
      port := 0
    } := by
    have decoderMatches := verified.decoderMatches
    unfold verifyDecoder at decoderMatches
    simp only [Bool.and_eq_true, decide_eq_true_eq] at decoderMatches
    have outputEq :
        certificate.decoder.equalsTwo.output = certificate.decoder.decoded := by aesop
    have scopes := verifiedBinaryKind_stage_scopes scalar.equalsTwo
    rw [← outputEq]
    simp [wireRef, scopes.2.2.2.2, equalsPort]
  have equalsTwoPortValid : 0 < equalsTwo.values.length := by
    rw [equalsTwoValues]
    simp
  have finalDecoded : Mxx.Ir.lookupWire (wireRef certificate.decoder.decoded)
      rootPath.finalWires = some decodedValue := by
    rw [decodedWire]
    simpa [equalsTwoValues] using equalsTwoRooted.outputFinal 0 equalsTwoPortValid
  have exportedDecoded := rootStageExecutionPath.lookup_exported_output rootPath
    "diamond-decoded"
    (wireRef certificate.decoder.decoded) outputNamesUnique decodedOutput decodedValue finalDecoded
  exact ⟨{
    rootPath
    oneVector
    oneVectorRooted
    kVector
    kVectorRooted
    decoderVector
    decoderVectorRooted
    oneMinusCircuit
    oneMinusCircuitRooted
    projectedDifference
    projectedDifferenceRooted
    kPlusProjection
    kPlusProjectionRooted
    residual
    residualRooted
    extractCoefficient
    extractCoefficientRooted
    threshold
    thresholdRooted
    lowerCompare
    lowerCompareRooted
    upperScale
    upperScaleRooted
    upperCompare
    upperCompareRooted
    lowerToInt
    lowerToIntRooted
    upperToInt
    upperToIntRooted
    comparisonSum
    comparisonSumRooted
    equalsTwo
    equalsTwoRooted
    decodedValue
    equalsTwoValues
    exportedDecoded
  }⟩

/-! ## Composed decoder semantics -/

/-- Matrix values produced by upstream execution bridges at the exact decoder source wires. -/
structure DecoderUpstreamValues
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (certificate : DiamondCertificate)
    (rootPath : RootStageExecutionPath samplers stage params inputs output) where
  state : Mxx.Matrix
  onePreimage : Mxx.Matrix
  kPreimage : Mxx.Matrix
  decoderPreimage : Mxx.Matrix
  selectedCircuitVector : Mxx.Matrix
  rDecomposed : Mxx.Matrix
  stateFinal : Mxx.Ir.lookupWire (wireRef certificate.decoder.oneVector.left.wire)
    rootPath.finalWires = some (.matrix state)
  onePreimageFinal : Mxx.Ir.lookupWire (wireRef certificate.decoder.onePreimage)
    rootPath.finalWires = some (.matrix onePreimage)
  kPreimageFinal : Mxx.Ir.lookupWire (wireRef certificate.decoder.kPreimage)
    rootPath.finalWires = some (.matrix kPreimage)
  decoderPreimageFinal : Mxx.Ir.lookupWire (wireRef certificate.decoder.decoderPreimage)
    rootPath.finalWires = some (.matrix decoderPreimage)
  selectedCircuitVectorFinal :
    Mxx.Ir.lookupWire (wireRef certificate.decoder.selectedCircuitVector)
      rootPath.finalWires = some (.matrix selectedCircuitVector)
  rDecomposedFinal : Mxx.Ir.lookupWire (wireRef certificate.decoder.rDecomposed)
    rootPath.finalWires = some (.matrix rDecomposed)

private theorem verified_decryption_root_ssa
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {stage : Mxx.Ir.Stage}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage) :
    verifyScopeSsaOrder stage.program.root = true := by
  have ssaOrderValid := verified.ssaOrderValid
  unfold verifyWorkflowSsaOrder at ssaOrderValid
  simp only [List.all_eq_true, Bool.and_eq_true] at ssaOrderValid
  exact (ssaOrderValid stage (decoder_resolveStage_mem stageResolved)).1

private theorem ReferencedNodeExecution.argument_node_lt
    {workflow : Mxx.Ir.Workflow} {reference : CoreNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (execution : ReferencedNodeExecution workflow reference runChild samplers params inputs)
    (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (ssaOrder : verifyScopeSsaOrder stage.program.root = true)
    (argument : Mxx.Ir.WireRef) (argumentMember : argument ∈ execution.node.arguments) :
    argument.node < reference.node := by
  have nodeAt : stage.program.root.nodes[reference.node]? = some execution.node := by
    rcases reference with ⟨stageName, scope, node⟩
    dsimp at rootScope
    subst scope
    simpa [resolveNode, resolveScope, scopeOwnerMatches, rawScope, stageResolved] using
      execution.resolved
  have inBounds := decoder_list_index_lt_of_getElem?_eq_some nodeAt
  have nodeEq := decoder_list_getElem_eq_of_getElem?_eq_some nodeAt inBounds
  apply decoder_verifyScopeSsaOrder_argument_lt ssaOrder reference.node inBounds argument
  simpa [nodeEq] using argumentMember

private theorem matrixArgumentBefore
    {workflow : Mxx.Ir.Workflow}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {reference : CoreNodeRef}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (execution : ReferencedNodeExecution workflow reference
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (rootScope : reference.scope = .root)
    (stageResolved : resolveStage workflow reference.stage = some stage)
    (ssaOrder : verifyScopeSsaOrder stage.program.root = true)
    (wire : Mxx.Ir.WireRef) (value : Mxx.Ir.Value)
    (argumentMember : wire ∈ execution.node.arguments)
    (finalLookup : Mxx.Ir.lookupWire wire rootPath.finalWires = some value) :
    Mxx.Ir.lookupWire wire execution.before = some value := by
  apply rooted.finalBefore wire value
  · exact execution.argument_node_lt rootScope stageResolved ssaOrder wire argumentMember
  · exact finalLookup

private theorem matrixBinary_execution_arguments
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {expected : Mxx.Ir.NodeKind}
    (verified : verifyMatrixBinary workflow reference expected = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.node.arguments =
      [wireRef reference.left.wire, wireRef reference.right.wire] := by
  unfold verifyMatrixBinary at verified
  simp only [Bool.and_eq_true] at verified
  have binary := verified.1
  unfold verifyBinaryNode at binary
  simp only [Bool.and_eq_true, decide_eq_true_eq] at binary
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at binary
  | some node =>
      have nodeEq : execution.node = node := by
        rw [execution.resolved] at resolved
        exact Option.some.inj resolved
      rw [nodeEq]
      simpa [resolved] using binary.2

private theorem matrixArguments_from_final
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {expected : Mxx.Ir.NodeKind}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyMatrixBinary workflow reference expected = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (rootScope : reference.operation.scope = .root)
    (stageResolved : resolveStage workflow reference.operation.stage = some stage)
    (ssaOrder : verifyScopeSsaOrder stage.program.root = true)
    (left right : Mxx.Matrix)
    (leftFinal : Mxx.Ir.lookupWire (wireRef reference.left.wire) rootPath.finalWires =
      some (.matrix left))
    (rightFinal : Mxx.Ir.lookupWire (wireRef reference.right.wire) rootPath.finalWires =
      some (.matrix right)) :
    [wireRef reference.left.wire, wireRef reference.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
      some [.matrix left, .matrix right] := by
  have argumentsEq := matrixBinary_execution_arguments verified execution
  have leftMember : wireRef reference.left.wire ∈ execution.node.arguments := by
    rw [argumentsEq]
    simp
  have rightMember : wireRef reference.right.wire ∈ execution.node.arguments := by
    rw [argumentsEq]
    simp
  apply decoder_lookupWirePair
  · exact matrixArgumentBefore rootPath execution rooted rootScope stageResolved ssaOrder
      (wireRef reference.left.wire) (.matrix left) leftMember leftFinal
  · exact matrixArgumentBefore rootPath execution rooted rootScope stageResolved ssaOrder
      (wireRef reference.right.wire) (.matrix right) rightMember rightFinal

private theorem matrixExecution_values
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    (expected : Mxx.Ir.NodeKind)
    (verified : verifyMatrixBinary workflow reference expected = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs)
    (left right result : Mxx.Matrix)
    (argumentsEvaluate :
      [wireRef reference.left.wire, wireRef reference.right.wire].mapM
          (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) =
        some [.matrix left, .matrix right])
    (resultEq : match expected with
      | .matrixMultiply => result = Mxx.matrixMultiply left right
      | .matrixAdd => result = Mxx.matrixAdd left right
      | .matrixSubtract => result = Mxx.matrixSubtract left right
      | _ => False) :
    execution.values = [.matrix result] := by
  have member := matrixBinaryOutcome_of_execution expected verified execution
  cases expected <;>
    simp_all [Mxx.Ir.evaluateNode, Mxx.Ir.arguments]

private theorem matrixOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : MatrixBinaryRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {expected : Mxx.Ir.NodeKind}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyMatrixBinary workflow reference expected = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (value : Mxx.Matrix) (valuesEq : execution.values = [.matrix value]) :
    Mxx.Ir.lookupWire (wireRef reference.output) rootPath.finalWires =
      some (.matrix value) := by
  have outputPort : reference.output.port = 0 := by
    have outputVerified :
        verifyOperationOutput workflow reference.operation reference.output = true := by
      unfold verifyMatrixBinary at verified
      simp only [Bool.and_eq_true] at verified
      have binary := verified.1
      unfold verifyBinaryNode at binary
      simp only [Bool.and_eq_true] at binary
      aesop
    unfold verifyOperationOutput at outputVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at outputVerified
    have wireVerified := outputVerified.2
    unfold verifyWire at wireVerified
    rw [outputVerified.1] at wireVerified
    rw [execution.resolved] at wireVerified
    have portLt : reference.output.port < execution.node.outputCount :=
      of_decide_eq_true wireVerified
    have countOne : execution.node.outputCount = 1 := by
      unfold verifyMatrixBinary at verified
      simp only [Bool.and_eq_true] at verified
      rw [execution.resolved] at verified
      simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
      exact verified.2.2
    omega
  have outputNode : reference.output.node = reference.operation := by
    have outputVerified :
        verifyOperationOutput workflow reference.operation reference.output = true := by
      unfold verifyMatrixBinary at verified
      simp only [Bool.and_eq_true] at verified
      have binary := verified.1
      unfold verifyBinaryNode at binary
      simp only [Bool.and_eq_true] at binary
      aesop
    unfold verifyOperationOutput at outputVerified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at outputVerified
    exact outputVerified.1
  rw [wireRef, outputNode, outputPort]
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

/-- Matrix semantics together with the exact final matrix-node runtime value. -/
structure DecoderMatrixExecutionOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output) where
  matrixOutcome : DecoderMatrixOutcome
  residualValues : bundle.residual.values = [.matrix matrixOutcome.residual]

/-- The upstream typed values and one rooted decoder execution determine the entire matrix chain. -/
theorem decoderMatrixExecutionOutcome_of_bundle_upstream
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output)
    (upstream : DecoderUpstreamValues certificate bundle.rootPath) :
    Nonempty (DecoderMatrixExecutionOutcome bundle) := by
  have matrix := verified.decoderMatrixOperations
  have locations := verified.decoderOperationLocations
  have ssaOrder := verified_decryption_root_ssa verified stageResolved
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true, decide_eq_true_eq] at decoderMatches
  have oneRight : certificate.decoder.oneVector.right.wire =
      certificate.decoder.onePreimage := by aesop
  have kRight : certificate.decoder.kVector.right.wire =
      certificate.decoder.kPreimage := by aesop
  have decoderRight : certificate.decoder.decoderVector.right.wire =
      certificate.decoder.decoderPreimage := by aesop
  have kState : certificate.decoder.kVector.left.wire =
      certificate.decoder.oneVector.left.wire := by aesop
  have decoderState : certificate.decoder.decoderVector.left.wire =
      certificate.decoder.oneVector.left.wire := by aesop
  have differenceLeft : certificate.decoder.oneMinusCircuit.left.wire =
      certificate.decoder.oneVector.output := by aesop
  have differenceRight : certificate.decoder.oneMinusCircuit.right.wire =
      certificate.decoder.selectedCircuitVector := by aesop
  have projectionLeft : certificate.decoder.projectedDifference.left.wire =
      certificate.decoder.oneMinusCircuit.output := by aesop
  have projectionRight : certificate.decoder.projectedDifference.right.wire =
      certificate.decoder.rDecomposed := by aesop
  have sumLeft : certificate.decoder.kPlusProjection.left.wire =
      certificate.decoder.kVector.output := by aesop
  have sumRight : certificate.decoder.kPlusProjection.right.wire =
      certificate.decoder.projectedDifference.output := by aesop
  have residualLeft : certificate.decoder.residual.left.wire =
      certificate.decoder.decoderVector.output := by aesop
  have residualRight : certificate.decoder.residual.right.wire =
      certificate.decoder.kPlusProjection.output := by aesop
  have oneArguments := matrixArguments_from_final bundle.rootPath matrix.oneVector
    bundle.oneVector bundle.oneVectorRooted locations.oneVector.2
    (by rw [locations.oneVector.1]; exact stageResolved) ssaOrder upstream.state
    upstream.onePreimage upstream.stateFinal (by simpa [oneRight] using upstream.onePreimageFinal)
  let oneVector := Mxx.matrixMultiply upstream.state upstream.onePreimage
  have oneValues : bundle.oneVector.values = [.matrix oneVector] :=
    matrixExecution_values .matrixMultiply matrix.oneVector bundle.oneVector upstream.state
      upstream.onePreimage oneVector oneArguments rfl
  have oneFinal := matrixOutputFinal bundle.rootPath matrix.oneVector bundle.oneVector
    bundle.oneVectorRooted oneVector oneValues
  have kArguments := matrixArguments_from_final bundle.rootPath matrix.kVector bundle.kVector
    bundle.kVectorRooted locations.kVector.2
    (by rw [locations.kVector.1]; exact stageResolved) ssaOrder upstream.state
    upstream.kPreimage (by simpa [kState] using upstream.stateFinal)
    (by simpa [kRight] using upstream.kPreimageFinal)
  let kVector := Mxx.matrixMultiply upstream.state upstream.kPreimage
  have kValues : bundle.kVector.values = [.matrix kVector] :=
    matrixExecution_values .matrixMultiply matrix.kVector bundle.kVector upstream.state
      upstream.kPreimage kVector kArguments rfl
  have kFinal := matrixOutputFinal bundle.rootPath matrix.kVector bundle.kVector
    bundle.kVectorRooted kVector kValues
  have decoderArguments := matrixArguments_from_final bundle.rootPath matrix.decoderVector
    bundle.decoderVector bundle.decoderVectorRooted locations.decoderVector.2
    (by rw [locations.decoderVector.1]; exact stageResolved) ssaOrder upstream.state
    upstream.decoderPreimage (by simpa [decoderState] using upstream.stateFinal)
    (by simpa [decoderRight] using upstream.decoderPreimageFinal)
  let decoderVector := Mxx.matrixMultiply upstream.state upstream.decoderPreimage
  have decoderValues : bundle.decoderVector.values = [.matrix decoderVector] :=
    matrixExecution_values .matrixMultiply matrix.decoderVector bundle.decoderVector upstream.state
      upstream.decoderPreimage decoderVector decoderArguments rfl
  have decoderFinal := matrixOutputFinal bundle.rootPath matrix.decoderVector
    bundle.decoderVector bundle.decoderVectorRooted decoderVector decoderValues
  have differenceArguments := matrixArguments_from_final bundle.rootPath matrix.oneMinusCircuit
    bundle.oneMinusCircuit bundle.oneMinusCircuitRooted locations.oneMinusCircuit.2
    (by rw [locations.oneMinusCircuit.1]; exact stageResolved) ssaOrder oneVector
    upstream.selectedCircuitVector (by simpa [differenceLeft] using oneFinal)
    (by simpa [differenceRight] using upstream.selectedCircuitVectorFinal)
  let oneMinusCircuit := Mxx.matrixSubtract oneVector upstream.selectedCircuitVector
  have differenceValues : bundle.oneMinusCircuit.values = [.matrix oneMinusCircuit] :=
    matrixExecution_values .matrixSubtract matrix.oneMinusCircuit bundle.oneMinusCircuit oneVector
      upstream.selectedCircuitVector oneMinusCircuit differenceArguments rfl
  have differenceFinal := matrixOutputFinal bundle.rootPath matrix.oneMinusCircuit
    bundle.oneMinusCircuit bundle.oneMinusCircuitRooted oneMinusCircuit differenceValues
  have projectionArguments := matrixArguments_from_final bundle.rootPath
    matrix.projectedDifference bundle.projectedDifference bundle.projectedDifferenceRooted
    locations.projectedDifference.2
    (by rw [locations.projectedDifference.1]; exact stageResolved) ssaOrder oneMinusCircuit
    upstream.rDecomposed (by simpa [projectionLeft] using differenceFinal)
    (by simpa [projectionRight] using upstream.rDecomposedFinal)
  let projectedDifference := Mxx.matrixMultiply oneMinusCircuit upstream.rDecomposed
  have projectionValues : bundle.projectedDifference.values = [.matrix projectedDifference] :=
    matrixExecution_values .matrixMultiply matrix.projectedDifference bundle.projectedDifference
      oneMinusCircuit upstream.rDecomposed projectedDifference projectionArguments rfl
  have projectionFinal := matrixOutputFinal bundle.rootPath matrix.projectedDifference
    bundle.projectedDifference bundle.projectedDifferenceRooted projectedDifference projectionValues
  have sumArguments := matrixArguments_from_final bundle.rootPath matrix.kPlusProjection
    bundle.kPlusProjection bundle.kPlusProjectionRooted locations.kPlusProjection.2
    (by rw [locations.kPlusProjection.1]; exact stageResolved) ssaOrder kVector
    projectedDifference (by simpa [sumLeft] using kFinal)
    (by simpa [sumRight] using projectionFinal)
  let kPlusProjection := Mxx.matrixAdd kVector projectedDifference
  have sumValues : bundle.kPlusProjection.values = [.matrix kPlusProjection] :=
    matrixExecution_values .matrixAdd matrix.kPlusProjection bundle.kPlusProjection kVector
      projectedDifference kPlusProjection sumArguments rfl
  have sumFinal := matrixOutputFinal bundle.rootPath matrix.kPlusProjection bundle.kPlusProjection
    bundle.kPlusProjectionRooted kPlusProjection sumValues
  have residualArguments := matrixArguments_from_final bundle.rootPath matrix.residual
    bundle.residual bundle.residualRooted locations.residual.2
    (by rw [locations.residual.1]; exact stageResolved) ssaOrder decoderVector
    kPlusProjection (by simpa [residualLeft] using decoderFinal)
    (by simpa [residualRight] using sumFinal)
  let residual := Mxx.matrixSubtract decoderVector kPlusProjection
  have residualValues : bundle.residual.values = [.matrix residual] :=
    matrixExecution_values .matrixSubtract matrix.residual bundle.residual decoderVector
      kPlusProjection residual residualArguments rfl
  exact ⟨{
    matrixOutcome := {
      state := upstream.state
      onePreimage := upstream.onePreimage
      kPreimage := upstream.kPreimage
      decoderPreimage := upstream.decoderPreimage
      selectedCircuitVector := upstream.selectedCircuitVector
      rDecomposed := upstream.rDecomposed
      oneVector
      kVector
      decoderVector
      oneMinusCircuit
      projectedDifference
      kPlusProjection
      residual
      oneVectorEq := rfl
      kVectorEq := rfl
      decoderVectorEq := rfl
      oneMinusCircuitEq := rfl
      projectedDifferenceEq := rfl
      kPlusProjectionEq := rfl
      residualEq := rfl
    }
    residualValues
  }⟩

/-- Projection of the complete rooted matrix execution onto its algebraic outcome. -/
theorem decoderMatrixOutcome_of_bundle_upstream
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output)
    (upstream : DecoderUpstreamValues certificate bundle.rootPath) :
    Nonempty DecoderMatrixOutcome := by
  obtain ⟨result⟩ := decoderMatrixExecutionOutcome_of_bundle_upstream verified stageResolved
    bundle upstream
  exact ⟨result.matrixOutcome⟩

private theorem binaryKind_execution_arguments
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyBinaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.node.arguments =
      [wireRef reference.left.wire, wireRef reference.right.wire] := by
  unfold verifyBinaryKind at verified
  simp only [Bool.and_eq_true] at verified
  have binary := verified.1.1
  unfold verifyBinaryNode at binary
  simp only [Bool.and_eq_true, decide_eq_true_eq] at binary
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      have nodeEq : execution.node = node := by
        rw [execution.resolved] at resolved
        exact Option.some.inj resolved
      rw [nodeEq]
      simpa [resolved] using binary.2

private theorem unaryKind_execution_arguments
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (verified : verifyUnaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation runChild samplers params
      inputs) :
    execution.node.arguments = [wireRef reference.input.wire] := by
  unfold verifyUnaryKind at verified
  simp only [Bool.and_eq_true] at verified
  have unary := verified.1.1
  unfold verifyUnaryNode at unary
  simp only [Bool.and_eq_true, decide_eq_true_eq] at unary
  cases resolved : resolveNode workflow reference.operation with
  | none => simp [resolved] at verified
  | some node =>
      have nodeEq : execution.node = node := by
        rw [execution.resolved] at resolved
        exact Option.some.inj resolved
      rw [nodeEq]
      simpa [resolved] using unary.2

private theorem binaryArguments_from_final
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyBinaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (rootScope : reference.operation.scope = .root)
    (stageResolved : resolveStage workflow reference.operation.stage = some stage)
    (ssaOrder : verifyScopeSsaOrder stage.program.root = true)
    (left right : Mxx.Ir.Value)
    (leftFinal : Mxx.Ir.lookupWire (wireRef reference.left.wire) rootPath.finalWires =
      some left)
    (rightFinal : Mxx.Ir.lookupWire (wireRef reference.right.wire) rootPath.finalWires =
      some right) :
    [wireRef reference.left.wire, wireRef reference.right.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some [left, right] := by
  have argumentsEq := binaryKind_execution_arguments verified execution
  apply decoder_lookupWirePair
  · apply rooted.finalBefore _ _
    · apply execution.argument_node_lt rootScope stageResolved ssaOrder
      rw [argumentsEq]
      simp
    · exact leftFinal
  · apply rooted.finalBefore _ _
    · apply execution.argument_node_lt rootScope stageResolved ssaOrder
      rw [argumentsEq]
      simp
    · exact rightFinal

private theorem unaryArguments_from_final
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyUnaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (rootScope : reference.operation.scope = .root)
    (stageResolved : resolveStage workflow reference.operation.stage = some stage)
    (ssaOrder : verifyScopeSsaOrder stage.program.root = true)
    (value : Mxx.Ir.Value)
    (finalLookup : Mxx.Ir.lookupWire (wireRef reference.input.wire)
      rootPath.finalWires = some value) :
    [wireRef reference.input.wire].mapM
        (fun wire ↦ Mxx.Ir.lookupWire wire execution.before) = some [value] := by
  have argumentsEq := unaryKind_execution_arguments verified execution
  have beforeLookup : Mxx.Ir.lookupWire (wireRef reference.input.wire)
      execution.before = some value := by
    apply rooted.finalBefore _ _
    · apply execution.argument_node_lt rootScope stageResolved ssaOrder
      rw [argumentsEq]
      simp
    · exact finalLookup
  simp [beforeLookup]

private theorem binaryOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : BinaryNodeRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyBinaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (value : Mxx.Ir.Value) (valuesEq : execution.values = [value]) :
    Mxx.Ir.lookupWire (wireRef reference.output) rootPath.finalWires = some value := by
  unfold verifyBinaryKind at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputPort := verified.1.2
  have binary := verified.1.1
  have outputNode := (verifiedBinaryNode_stage_scopes binary).2.2.2.2
  rw [wireRef, outputNode, outputPort]
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

private theorem unaryOutputFinal
    {workflow : Mxx.Ir.Workflow} {reference : UnaryNodeRef}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    {accept : Mxx.Ir.NodeKind → Bool}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : verifyUnaryKind workflow reference accept = true)
    (execution : ReferencedNodeExecution workflow reference.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (value : Mxx.Ir.Value) (valuesEq : execution.values = [value]) :
    Mxx.Ir.lookupWire (wireRef reference.output) rootPath.finalWires = some value := by
  unfold verifyUnaryKind at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputPort := verified.1.2
  have unary := verified.1.1
  have outputNode := (verifiedUnaryNode_stage_scope unary).2.2
  rw [wireRef, outputNode, outputPort]
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

private theorem constantIntFinal
    {workflow : Mxx.Ir.Workflow} {samplers : Mxx.MxxSamplerFamily}
    {stage : Mxx.Ir.Stage} {params : Mxx.Ir.ParamEnvironment}
    {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (reference : CoreWireRef) (stageName : String) (value : Int)
    (verified : verifyConstantIntWire workflow reference value = true)
    (location : reference.node.stage = stageName ∧ reference.node.scope = .root)
    (stageResolved : resolveStage workflow stageName = some stage) :
    Mxx.Ir.lookupWire (wireRef reference) rootPath.finalWires = some (.integer value) := by
  have resolved : resolveNode workflow reference.node = some {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    } := by
    unfold verifyConstantIntWire at verified
    simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
    cases nodeResolved : resolveNode workflow reference.node with
    | none => simp [nodeResolved] at verified
    | some node =>
        rcases node with ⟨kind, arguments, outputCount⟩
        simp [nodeResolved] at verified
        cases kind <;> simp_all
  obtain ⟨execution, rooted⟩ := rootPath.referencedRootNodeExecution_of_location
    reference.node stageName location stageResolved ⟨_, resolved⟩
  have nodeEq : execution.node = {
      kind := .constantInt value
      arguments := []
      outputCount := 1
    } := by
    rw [execution.resolved] at resolved
    exact Option.some.inj resolved
  have valuesEq : execution.values = [.integer value] := by
    have member := execution.member
    rw [nodeEq] at member
    simpa [Mxx.Ir.evaluateNode] using member
  have portZero : reference.port = 0 := by
    have checked := verified
    unfold verifyConstantIntWire at checked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
    exact checked.1
  rw [wireRef, portZero]
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

private theorem evaluateIntOutputFinal
    {workflow : Mxx.Ir.Workflow} {layout : DecoderLayout}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (rootPath : RootStageExecutionPath samplers stage params inputs output)
    (verified : VerifiedDecoderScalarOperations workflow layout)
    (execution : ReferencedNodeExecution workflow layout.threshold.operation
      (Mxx.Ir.childRunnerWithFuel samplers stage.program stage.program.definitions.length)
      samplers params inputs)
    (rooted : RootedNodeExecution rootPath execution)
    (modulus : Int)
    (modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
      some modulus)
    (noMaterialization : layout.threshold.materialization = none) :
    Mxx.Ir.lookupWire (wireRef layout.threshold.output) rootPath.finalWires =
      some (.integer (modulus / 4)) := by
  have valuesEq := decoderThresholdOutcome_of_execution verified execution modulus
    modulusEvaluate
  have checked := verified.threshold
  unfold verifyEvaluateInt at checked
  rw [noMaterialization] at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  have outputEq : layout.threshold.output = layout.threshold.evaluated := checked.2
  have evaluatedNode : layout.threshold.evaluated.node = layout.threshold.operation := by
    exact verifiedEvaluateInt_output_node verified.threshold
  have evaluatedPort : layout.threshold.evaluated.port = 0 := checked.1.1.2
  rw [outputEq, wireRef, evaluatedNode, evaluatedPort]
  simpa [valuesEq] using rooted.outputFinal 0 (by simp [valuesEq])

/-- Complete executable decoder semantics.  `residual` is the IR node name; `noisy` is the
protocol name for the same final decoder matrix. -/
structure DecoderSemanticOutcome
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output) where
  matrixExecution : DecoderMatrixExecutionOutcome bundle
  residual : Mxx.Matrix
  noisy : Mxx.Matrix
  residualEq : residual = matrixExecution.matrixOutcome.residual
  noisyEq : noisy = residual
  modulus : Int
  modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
    some modulus
  coefficient : Int
  decoded : Bool
  coefficientEq : coefficient = noisy.coefficients.headD 0
  decodedValue : bundle.decodedValue = .boolean decoded
  decodedEq : decoded = MxxWe.decodeBooleanInterval modulus coefficient
  exportedDecoded : Mxx.Ir.lookupEnvironment "diamond-decoded" output =
    some (.boolean decoded)

/-- Compose the typed upstream values with every checked scalar decoder node. -/
private theorem decoderSemanticOutcome_of_bundle_upstream_noMaterialization
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output)
    (upstream : DecoderUpstreamValues certificate bundle.rootPath)
    (modulus : Int)
    (modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
      some modulus)
    (noMaterialization : certificate.decoder.threshold.materialization = none) :
    Nonempty (DecoderSemanticOutcome bundle) := by
  obtain ⟨matrixExecution⟩ :=
    decoderMatrixExecutionOutcome_of_bundle_upstream verified stageResolved bundle upstream
  have scalar := verified.decoderScalarOperations
  have locations := verified.decoderOperationLocations
  have ssaOrder := verified_decryption_root_ssa verified stageResolved
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true, decide_eq_true_eq] at decoderMatches
  have extractInput : certificate.decoder.extractCoefficient.input.wire =
      certificate.decoder.residual.output := by aesop
  have lowerLeft : certificate.decoder.lowerCompare.left.wire =
      certificate.decoder.threshold.output := by aesop
  have lowerRight : certificate.decoder.lowerCompare.right.wire =
      certificate.decoder.extractCoefficient.output := by aesop
  have upperLeft : certificate.decoder.upperCompare.left.wire =
      certificate.decoder.extractCoefficient.output := by aesop
  have scaleLeft : certificate.decoder.upperScale.left.wire =
      certificate.decoder.lowerCompare.left.wire := by aesop
  have upperRight : certificate.decoder.upperCompare.right.wire =
      certificate.decoder.upperScale.output := by aesop
  have lowerToIntInput : certificate.decoder.lowerToInt.input.wire =
      certificate.decoder.lowerCompare.output := by aesop
  have upperToIntInput : certificate.decoder.upperToInt.input.wire =
      certificate.decoder.upperCompare.output := by aesop
  have sumLeft : certificate.decoder.comparisonSum.left.wire =
      certificate.decoder.lowerToInt.output := by aesop
  have sumRight : certificate.decoder.comparisonSum.right.wire =
      certificate.decoder.upperToInt.output := by aesop
  have equalsLeft : certificate.decoder.equalsTwo.left.wire =
      certificate.decoder.comparisonSum.output := by aesop
  have residualFinal := matrixOutputFinal bundle.rootPath
    (verified.decoderMatrixOperations.residual) bundle.residual bundle.residualRooted
    matrixExecution.matrixOutcome.residual matrixExecution.residualValues
  have extractArguments := unaryArguments_from_final bundle.rootPath scalar.extractCoefficient
    bundle.extractCoefficient bundle.extractCoefficientRooted locations.extractCoefficient.2
    (by rw [locations.extractCoefficient.1]; exact stageResolved) ssaOrder
    (.matrix matrixExecution.matrixOutcome.residual) (by simpa [extractInput] using residualFinal)
  let coefficient := matrixExecution.matrixOutcome.residual.coefficients.headD 0
  have extractValues : bundle.extractCoefficient.values = [.integer coefficient] := by
    simpa only [coefficient, List.headD_eq_getD] using extractCoefficientZeroOutcome_of_execution
      scalar.extractCoefficient bundle.extractCoefficient matrixExecution.matrixOutcome.residual
      extractArguments
  have extractFinal := unaryOutputFinal bundle.rootPath scalar.extractCoefficient
    bundle.extractCoefficient bundle.extractCoefficientRooted (.integer coefficient) extractValues
  have thresholdFinal := evaluateIntOutputFinal bundle.rootPath scalar bundle.threshold
    bundle.thresholdRooted modulus modulusEvaluate noMaterialization
  have lowerArguments := binaryArguments_from_final bundle.rootPath scalar.lowerCompare
    bundle.lowerCompare bundle.lowerCompareRooted locations.lowerCompare.2
    (by rw [locations.lowerCompare.1]; exact stageResolved) ssaOrder
    (.integer (modulus / 4)) (.integer coefficient) (by simpa [lowerLeft] using thresholdFinal)
    (by simpa [lowerRight] using extractFinal)
  let lower := decide (modulus / 4 ≤ coefficient)
  have lowerValues : bundle.lowerCompare.values = [.boolean lower] := by
    simpa [lower] using lessEqualOutcome_of_execution scalar.lowerCompare bundle.lowerCompare
      (modulus / 4) coefficient lowerArguments
  have lowerFinal := binaryOutputFinal bundle.rootPath scalar.lowerCompare bundle.lowerCompare
    bundle.lowerCompareRooted (.boolean lower) lowerValues
  have scaleScopes := verifiedBinaryKind_stage_scopes scalar.upperScale
  have threeFinal := constantIntFinal bundle.rootPath certificate.decoder.upperScale.right.wire
    certificate.workflow.decryption.stage 3 (by aesop)
    ⟨by rw [← scaleScopes.2.2.1, locations.upperScale.1],
      by rw [← scaleScopes.2.2.2.1, locations.upperScale.2]⟩ stageResolved
  have scaleArguments := binaryArguments_from_final bundle.rootPath scalar.upperScale
    bundle.upperScale bundle.upperScaleRooted locations.upperScale.2
    (by rw [locations.upperScale.1]; exact stageResolved) ssaOrder
    (.integer (modulus / 4)) (.integer 3) (by simpa [scaleLeft, lowerLeft] using thresholdFinal)
    threeFinal
  let scaledThreshold := modulus / 4 * 3
  have scaleValues : bundle.upperScale.values = [.integer scaledThreshold] := by
    simpa [scaledThreshold] using intMultiplyOutcome_of_execution scalar.upperScale
      bundle.upperScale (modulus / 4) 3 scaleArguments
  have scaleFinal := binaryOutputFinal bundle.rootPath scalar.upperScale bundle.upperScale
    bundle.upperScaleRooted (.integer scaledThreshold) scaleValues
  have upperArguments := binaryArguments_from_final bundle.rootPath scalar.upperCompare
    bundle.upperCompare bundle.upperCompareRooted locations.upperCompare.2
    (by rw [locations.upperCompare.1]; exact stageResolved) ssaOrder
    (.integer coefficient) (.integer scaledThreshold) (by simpa [upperLeft] using extractFinal)
    (by simpa [upperRight] using scaleFinal)
  let upper := decide (coefficient ≤ scaledThreshold)
  have upperValues : bundle.upperCompare.values = [.boolean upper] := by
    simpa [upper] using lessEqualOutcome_of_execution scalar.upperCompare bundle.upperCompare
      coefficient scaledThreshold upperArguments
  have upperFinal := binaryOutputFinal bundle.rootPath scalar.upperCompare bundle.upperCompare
    bundle.upperCompareRooted (.boolean upper) upperValues
  have lowerIntArguments := unaryArguments_from_final bundle.rootPath scalar.lowerToInt
    bundle.lowerToInt bundle.lowerToIntRooted locations.lowerToInt.2
    (by rw [locations.lowerToInt.1]; exact stageResolved) ssaOrder (.boolean lower)
    (by simpa [lowerToIntInput] using lowerFinal)
  let lowerInt : Int := if lower then 1 else 0
  have lowerIntValues : bundle.lowerToInt.values = [.integer lowerInt] := by
    simpa [lowerInt] using boolToIntOutcome_of_execution scalar.lowerToInt bundle.lowerToInt lower
      lowerIntArguments
  have lowerIntFinal := unaryOutputFinal bundle.rootPath scalar.lowerToInt bundle.lowerToInt
    bundle.lowerToIntRooted (.integer lowerInt) lowerIntValues
  have upperIntArguments := unaryArguments_from_final bundle.rootPath scalar.upperToInt
    bundle.upperToInt bundle.upperToIntRooted locations.upperToInt.2
    (by rw [locations.upperToInt.1]; exact stageResolved) ssaOrder (.boolean upper)
    (by simpa [upperToIntInput] using upperFinal)
  let upperInt : Int := if upper then 1 else 0
  have upperIntValues : bundle.upperToInt.values = [.integer upperInt] := by
    simpa [upperInt] using boolToIntOutcome_of_execution scalar.upperToInt bundle.upperToInt upper
      upperIntArguments
  have upperIntFinal := unaryOutputFinal bundle.rootPath scalar.upperToInt bundle.upperToInt
    bundle.upperToIntRooted (.integer upperInt) upperIntValues
  have sumArguments := binaryArguments_from_final bundle.rootPath scalar.comparisonSum
    bundle.comparisonSum bundle.comparisonSumRooted locations.comparisonSum.2
    (by rw [locations.comparisonSum.1]; exact stageResolved) ssaOrder (.integer lowerInt)
    (.integer upperInt) (by simpa [sumLeft] using lowerIntFinal)
    (by simpa [sumRight] using upperIntFinal)
  let sum : Int := lowerInt + upperInt
  have sumValues : bundle.comparisonSum.values = [.integer sum] := by
    simpa [sum] using intAddOutcome_of_execution scalar.comparisonSum bundle.comparisonSum
      lowerInt upperInt sumArguments
  have sumFinal := binaryOutputFinal bundle.rootPath scalar.comparisonSum bundle.comparisonSum
    bundle.comparisonSumRooted (.integer sum) sumValues
  have equalsScopes := verifiedBinaryKind_stage_scopes scalar.equalsTwo
  have twoFinal := constantIntFinal bundle.rootPath certificate.decoder.equalsTwo.right.wire
    certificate.workflow.decryption.stage 2 (by aesop)
    ⟨by rw [← equalsScopes.2.2.1, locations.equalsTwo.1],
      by rw [← equalsScopes.2.2.2.1, locations.equalsTwo.2]⟩ stageResolved
  have equalsArguments := binaryArguments_from_final bundle.rootPath scalar.equalsTwo
    bundle.equalsTwo bundle.equalsTwoRooted locations.equalsTwo.2
    (by rw [locations.equalsTwo.1]; exact stageResolved) ssaOrder (.integer sum) (.integer 2)
    (by simpa [equalsLeft] using sumFinal) twoFinal
  let decoded := decide (sum = 2)
  have equalsValues : bundle.equalsTwo.values = [.boolean decoded] := by
    simpa [decoded] using intEqualOutcome_of_execution scalar.equalsTwo bundle.equalsTwo sum 2
      equalsArguments
  have decodedValue : bundle.decodedValue = .boolean decoded := by
    have equalsTwoValues := bundle.equalsTwoValues
    rw [equalsValues] at equalsTwoValues
    simpa using equalsTwoValues.symm
  have decodedEq : decoded = MxxWe.decodeBooleanInterval modulus coefficient := by
    apply decoded_eq_decodeBooleanInterval_of_execution scalar bundle.threshold modulus coefficient
      (modulus / 4) lowerInt upperInt sum lower upper decoded modulusEvaluate
    · exact decoderThresholdOutcome_of_execution scalar bundle.threshold modulus modulusEvaluate
    · rfl
    · simp [upper, scaledThreshold]
    · rfl
    · rfl
    · rfl
    · rfl
  have exportedDecoded : Mxx.Ir.lookupEnvironment "diamond-decoded" output =
      some (.boolean decoded) := by
    simpa [decodedValue] using bundle.exportedDecoded
  exact ⟨{
    matrixExecution
    residual := matrixExecution.matrixOutcome.residual
    noisy := matrixExecution.matrixOutcome.residual
    residualEq := rfl
    noisyEq := rfl
    modulus
    modulusEvaluate
    coefficient
    decoded
    coefficientEq := rfl
    decodedValue
    decodedEq
    exportedDecoded
  }⟩

private theorem VerifiedDiamondLayout.decoderThreshold_noMaterialization
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    (verified : VerifiedDiamondLayout workflow certificate) :
    certificate.decoder.threshold.materialization = none := by
  have decoderMatches := verified.decoderMatches
  unfold verifyDecoder at decoderMatches
  simp only [Bool.and_eq_true, decide_eq_true_eq] at decoderMatches
  have materializationNone : certificate.decoder.threshold.materialization.isNone = true := by
    aesop
  cases materializationEq : certificate.decoder.threshold.materialization <;>
    simp_all

/-- The complete decoder semantics require only verified layout, rooted execution, upstream
typed values, and the evaluated modulus.  All wire lookups and singleton values are internal. -/
theorem decoderSemanticOutcome_of_bundle_upstream
    {workflow : Mxx.Ir.Workflow} {certificate : DiamondCertificate}
    {samplers : Mxx.MxxSamplerFamily} {stage : Mxx.Ir.Stage}
    {params : Mxx.Ir.ParamEnvironment} {inputs output : Mxx.Ir.Environment}
    (verified : VerifiedDiamondLayout workflow certificate)
    (stageResolved :
      resolveStage workflow certificate.workflow.decryption.stage = some stage)
    (bundle : DecoderStageExecutionBundle workflow certificate samplers stage params inputs
      output)
    (upstream : DecoderUpstreamValues certificate bundle.rootPath)
    (modulus : Int)
    (modulusEvaluate : (.parameter "diamond_modulus" : Mxx.Ir.IntExpr).evaluate params =
      some modulus) :
    Nonempty (DecoderSemanticOutcome bundle) :=
  decoderSemanticOutcome_of_bundle_upstream_noMaterialization verified stageResolved bundle
    upstream modulus modulusEvaluate verified.decoderThreshold_noMaterialization

end MxxWe.Certificate
