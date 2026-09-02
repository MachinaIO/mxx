import MxxBgg.Invariant
import MxxWe.DiamondWE.Model
import MxxWe.DiamondWE.Operational

namespace Mxx.We.DiamondWE

open Mxx.Primitives
open Mxx.IR

/- A successful root primitive run resolves exactly the values stored on its
   two IR input wires.  In equations, if the trace contains

       trace(leftWire)  = left
       trace(rightWire) = right,

   then the evaluator operand array is exactly `#[left, right]`.  This lemma
   is payload-independent: subtraction, addition, and `applyPreimage` all use
   the same SSA dataflow argument. -/
theorem RootPrimitiveRun.binaryArguments_eq
    {backend : SemanticBackend} {trace : Trace backend}
    {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node} {port : Nat} {leftWire rightWire : WireRef}
    (run : RootPrimitiveRun trace stage scope node payload storedNode port)
    (argumentsStored : storedNode.arguments = #[leftWire, rightWire])
    (left right : DynamicValue backend)
    (leftTraced : traceValueAt trace (occurrenceOf stage #[] leftWire) = some left)
    (rightTraced : traceValueAt trace (occurrenceOf stage #[] rightWire) = some right) :
    run.arguments = #[left, right] := by
  have argumentSize : run.arguments.size = storedNode.arguments.size := by
    have resolved := run.argumentsResolved
    unfold resolveArguments at resolved
    let resolveOne : WireRef → Except EvalError (DynamicValue backend) :=
      fun wire => match lookup run.values wire with
        | some value => pure value
        | none => throw (EvalError.missingPort stage scope node wire.port)
    have sizeFact := Array.size_mapM resolveOne storedNode.arguments
    change storedNode.arguments.mapM resolveOne = .ok run.arguments at resolved
    rw [resolved] at sizeFact
    simpa using sizeFact
  obtain ⟨leftBound, argumentLeftTraced⟩ := run.argumentTraced (index := 0)
    (by simp [argumentsStored])
  obtain ⟨rightBound, argumentRightTraced⟩ := run.argumentTraced (index := 1)
    (by simp [argumentsStored])
  have leftEq : run.arguments[0] = left := by
    apply Option.some.inj
    calc
      some run.arguments[0] =
          traceValueAt trace (occurrenceOf stage #[] storedNode.arguments[0]) :=
        argumentLeftTraced.symm
      _ = traceValueAt trace (occurrenceOf stage #[] leftWire) := by
        simp [argumentsStored]
      _ = some left := leftTraced
  have rightEq : run.arguments[1] = right := by
    apply Option.some.inj
    calc
      some run.arguments[1] =
          traceValueAt trace (occurrenceOf stage #[] storedNode.arguments[1]) :=
        argumentRightTraced.symm
      _ = traceValueAt trace (occurrenceOf stage #[] rightWire) := by
        simp [argumentsStored]
      _ = some right := rightTraced
  apply Array.ext
  · simpa [argumentsStored] using argumentSize
  · intro index leftIndexBound rightIndexBound
    have indexBound : index < 2 := by simpa using rightIndexBound
    interval_cases index
    · simpa using leftEq
    · simpa using rightEq

/- Once evaluator inversion determines the singleton result array, the value
   stored at output port zero is forced.  This is deliberately a conclusion
   from `nodeResult.values` and `outputStored`; it is not a caller equation
   about the reached wire. -/
theorem RootPrimitiveRun.output_eq_of_singleton
    {backend : SemanticBackend} {trace : Trace backend}
    {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node}
    (run : RootPrimitiveRun trace stage scope node payload storedNode 0)
    (expected : DynamicValue backend)
    (valuesEq : run.nodeResult.values = #[expected]) :
    run.output = expected := by
  have stored := run.outputStored
  rw [valuesEq] at stored
  simpa using (Option.some.inj stored).symm

/- The public evaluator bridge returns the resolved argument array together
   with trace coverage for every SSA binding used by that resolution.  These
   two facts are sufficient to recover both producer values of a concrete
   binary Diamond site.  The application supplies only stored wire edges;
   no matrix equation or caller-selected output enters this bridge. -/
theorem resolvedBinarySiteArgumentsTrace
    {backend : SemanticBackend} {stage scope node : Nat}
    {wires : Array WireRef} {values : Array (Binding backend)}
    {arguments : Array (DynamicValue backend)} {trace : Trace backend}
    {leftWire rightWire : WireRef}
    (resolved : resolveArguments stage scope node values wires = .ok arguments)
    (valuesTraced : ∀ binding ∈ values,
      Mxx.IR.traceValueAt trace (occurrenceOf stage #[] binding.wire) = some binding.value)
    (leftEdge : wires[0]? = some leftWire)
    (rightEdge : wires[1]? = some rightWire) :
    ∃ leftBound : 0 < arguments.size, ∃ rightBound : 1 < arguments.size,
      Mxx.IR.traceValueAt trace (occurrenceOf stage #[] leftWire) = some arguments[0] ∧
        Mxx.IR.traceValueAt trace (occurrenceOf stage #[] rightWire) = some arguments[1] := by
  have leftWireBound : 0 < wires.size := by
    exact (Array.getElem?_eq_some_iff.mp leftEdge).1
  have rightWireBound : 1 < wires.size := by
    exact (Array.getElem?_eq_some_iff.mp rightEdge).1
  obtain ⟨leftBound, leftTrace⟩ :=
    resolvedArgument_trace resolved valuesTraced 0 leftWireBound
  obtain ⟨rightBound, rightTrace⟩ :=
    resolvedArgument_trace resolved valuesTraced 1 rightWireBound
  have leftWireEq : wires[0] = leftWire := (Array.getElem?_eq_some_iff.mp leftEdge).2
  have rightWireEq : wires[1] = rightWire := (Array.getElem?_eq_some_iff.mp rightEdge).2
  exact ⟨leftBound, rightBound, by simpa [leftWireEq] using leftTrace,
    by simpa [rightWireEq] using rightTrace⟩

/- Inverting an actually executed subtraction node gives its backend matrix
   equation.  This is the operational counterpart of `left - right`; the
   equation is a conclusion of `evalPrimitiveNode`, not an application
   premise about the node output. -/
theorem runtimeMatrixSubtractExecution
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) {matrixType : MatrixType}
    (valid : matrixType.Valid) (stage scope node : Nat)
    (left right : RuntimeMatrixValue matrixType) (result : NodeResult (RuntimeBackend oracle))
    (execution : evalPrimitiveNode (RuntimeBackend oracle) {} stage scope node
      (.matrixBinary .subtract)
      #[⟨.matrix matrixType, left⟩, ⟨.matrix matrixType, right⟩]
      #[.matrix matrixType] = .ok result) :
    result.values = #[⟨.matrix matrixType, left - right⟩] := by
  have resultEq :
      (NodeResult.ofValues #[⟨.matrix matrixType, left - right⟩] :
        NodeResult (RuntimeBackend oracle)) = result := by
    simp [evalPrimitiveNode, primitive,
      Mxx.Runtime.irBackendWithGadgetOracle,
      Mxx.Runtime.irBackend_matrixSubtract valid] at execution
    change Except.ok _ = Except.ok result at execution
    exact Except.ok.inj execution
  rw [← resultEq]
  rfl

/- The same inversion for the middle fuse addition proves
   `kPlusProjection = kProjection + projectedDifference` from execution. -/
theorem runtimeMatrixAddExecution
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) {matrixType : MatrixType}
    (valid : matrixType.Valid) (stage scope node : Nat)
    (left right : RuntimeMatrixValue matrixType) (result : NodeResult (RuntimeBackend oracle))
    (execution : evalPrimitiveNode (RuntimeBackend oracle) {} stage scope node
      (.matrixBinary .add)
      #[⟨.matrix matrixType, left⟩, ⟨.matrix matrixType, right⟩]
      #[.matrix matrixType] = .ok result) :
    result.values = #[⟨.matrix matrixType, left + right⟩] := by
  have resultEq :
      (NodeResult.ofValues #[⟨.matrix matrixType, left + right⟩] :
        NodeResult (RuntimeBackend oracle)) = result := by
    simp [evalPrimitiveNode, primitive,
      Mxx.Runtime.irBackendWithGadgetOracle,
      Mxx.Runtime.irBackend_matrixAdd valid] at execution
    change Except.ok _ = Except.ok result at execution
    exact Except.ok.inj execution
  rw [← resultEq]
  rfl

/- Applying the retained R decomposition is also obtained by evaluator
   inversion.  `exactMultiply` is the runtime's typed matrix product; a
   concrete generated MatrixType later reduces it to the ordinary product
   used by `bggDifferenceProjectionWithin`. -/
theorem runtimeApplyPreimageExecution
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    {leftType preimageType outputType : MatrixType}
    (valid : leftType.Valid ∧ preimageType.Valid ∧ outputType.Valid ∧
      matrixProductType leftType preimageType outputType)
    (stage scope node : Nat) (left : RuntimeMatrixValue leftType)
    (preimage : Mxx.Runtime.PreimageValue preimageType)
    (result : NodeResult (RuntimeBackend oracle))
    (execution : evalPrimitiveNode (RuntimeBackend oracle) {} stage scope node .applyPreimage
      #[⟨.matrix leftType, left⟩, ⟨.preimage preimageType, preimage⟩]
      #[.matrix outputType] = .ok result) :
    result.values = #[⟨.matrix outputType,
      Mxx.Runtime.exactMultiply leftType preimageType outputType valid
        left preimage.exactMatrix⟩] := by
  have resultEq :
      (NodeResult.ofValues #[⟨.matrix outputType,
        Mxx.Runtime.exactMultiply leftType preimageType outputType valid
          left preimage.exactMatrix⟩] : NodeResult (RuntimeBackend oracle)) = result := by
    simp [evalPrimitiveNode, primitive,
      Mxx.Runtime.irBackendWithGadgetOracle,
      Mxx.Runtime.irBackend_applyPreimage valid] at execution
    change Except.ok _ = Except.ok result at execution
    exact Except.ok.inj execution
  rw [← resultEq]
  rfl

/- The four primitive calls below are exactly the Diamond fuse:

     difference = one - circuit
     projected  = difference * rPreimage
     kPlus      = kProjection + projected
     noisy      = decoder - kPlus.

   Every premise is an `evalPrimitiveNode` result produced by evaluator
   inversion.  The final matrix equation is therefore derived from those
   executions; it is not accepted as a separate protocol contract. -/
theorem runtimeFuseExecution
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    {encodingType preimageType scalarType : MatrixType}
    (encodingValid : encodingType.Valid)
    (productValid : encodingType.Valid ∧ preimageType.Valid ∧ scalarType.Valid ∧
      matrixProductType encodingType preimageType scalarType)
    (one circuit : RuntimeMatrixValue encodingType)
    (rPreimage : Mxx.Runtime.PreimageValue preimageType)
    (kProjection decoder : RuntimeMatrixValue scalarType)
    (differenceResult projectedResult kPlusResult noisyResult :
      NodeResult (RuntimeBackend oracle))
    (differenceStage differenceScope differenceNode : Nat)
    (projectedStage projectedScope projectedNode : Nat)
    (kPlusStage kPlusScope kPlusNode : Nat)
    (noisyStage noisyScope noisyNode : Nat)
    (differenceExecution : evalPrimitiveNode (RuntimeBackend oracle) {}
      differenceStage differenceScope differenceNode (.matrixBinary .subtract)
      #[⟨.matrix encodingType, one⟩, ⟨.matrix encodingType, circuit⟩]
      #[.matrix encodingType] = .ok differenceResult)
    (projectedExecution : evalPrimitiveNode (RuntimeBackend oracle) {}
      projectedStage projectedScope projectedNode .applyPreimage
      #[⟨.matrix encodingType, one - circuit⟩, ⟨.preimage preimageType, rPreimage⟩]
      #[.matrix scalarType] = .ok projectedResult)
    (kPlusExecution : evalPrimitiveNode (RuntimeBackend oracle) {}
      kPlusStage kPlusScope kPlusNode (.matrixBinary .add)
      #[⟨.matrix scalarType, kProjection⟩, ⟨.matrix scalarType,
        Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix⟩]
      #[.matrix scalarType] = .ok kPlusResult)
    (noisyExecution : evalPrimitiveNode (RuntimeBackend oracle) {}
      noisyStage noisyScope noisyNode (.matrixBinary .subtract)
      #[⟨.matrix scalarType, decoder⟩, ⟨.matrix scalarType,
        kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix⟩]
      #[.matrix scalarType] = .ok noisyResult) :
    differenceResult.values = #[⟨.matrix encodingType, one - circuit⟩] ∧
      projectedResult.values = #[⟨.matrix scalarType,
        Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix⟩] ∧
      kPlusResult.values = #[⟨.matrix scalarType,
        kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix⟩] ∧
      noisyResult.values = #[⟨.matrix scalarType, decoder - (kProjection +
        Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix)⟩] := by
  have differenceEq := runtimeMatrixSubtractExecution oracle encodingValid
    differenceStage differenceScope differenceNode one circuit differenceResult differenceExecution
  have projectedEq := runtimeApplyPreimageExecution oracle productValid
    projectedStage projectedScope projectedNode (one - circuit) rPreimage projectedResult
    projectedExecution
  have kPlusEq := runtimeMatrixAddExecution oracle productValid.2.2.1
    kPlusStage kPlusScope kPlusNode kProjection
    (Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
      (one - circuit) rPreimage.exactMatrix) kPlusResult kPlusExecution
  have noisyEq := runtimeMatrixSubtractExecution oracle productValid.2.2.1
    noisyStage noisyScope noisyNode decoder
    (kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
      (one - circuit) rPreimage.exactMatrix) noisyResult noisyExecution
  exact ⟨differenceEq, projectedEq, kPlusEq, noisyEq⟩

/- Four generated `RootPrimitiveRun` witnesses are enough to replay the
   concrete Diamond fuse dataflow.  The equations are derived in this order:

       difference = one - circuit
       projected  = difference * rPreimage
       kPlus      = kProjection + projected
       noisy      = decoder - kPlus.

   Each downstream operand is identified by equating two readings of the
   same producer occurrence in `trace`.  Consequently the final trace fact is
   about the actual output of the executed `noisyPlaintext` node, with no
   caller-supplied equation for any intermediate or final matrix. -/
theorem runtimeFuseTrace
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    {trace : RuntimeTrace oracle} {stage scope : Nat}
    {encodingType preimageType scalarType : MatrixType}
    (encodingValid : encodingType.Valid)
    (productValid : encodingType.Valid ∧ preimageType.Valid ∧ scalarType.Valid ∧
      matrixProductType encodingType preimageType scalarType)
    (one circuit : RuntimeMatrixValue encodingType)
    (rPreimage : Mxx.Runtime.PreimageValue preimageType)
    (kProjection decoder : RuntimeMatrixValue scalarType)
    {oneWire circuitWire preimageWire kWire decoderWire : WireRef}
    {differenceNode projectedNode kPlusNode noisyNode : Nat}
    {differenceStored projectedStored kPlusStored noisyStored : Node}
    (differenceRun : RootPrimitiveRun trace stage scope differenceNode
      (.matrixBinary .subtract) differenceStored 0)
    (projectedRun : RootPrimitiveRun trace stage scope projectedNode
      .applyPreimage projectedStored 0)
    (kPlusRun : RootPrimitiveRun trace stage scope kPlusNode
      (.matrixBinary .add) kPlusStored 0)
    (noisyRun : RootPrimitiveRun trace stage scope noisyNode
      (.matrixBinary .subtract) noisyStored 0)
    (differenceArguments : differenceStored.arguments = #[oneWire, circuitWire])
    (projectedArguments : projectedStored.arguments =
      #[{ scope := scope, node := differenceNode, port := 0 }, preimageWire])
    (kPlusArguments : kPlusStored.arguments =
      #[kWire, { scope := scope, node := projectedNode, port := 0 }])
    (noisyArguments : noisyStored.arguments =
      #[decoderWire, { scope := scope, node := kPlusNode, port := 0 }])
    (differenceOutputs : differenceStored.outputs = #[.matrix encodingType])
    (projectedOutputs : projectedStored.outputs = #[.matrix scalarType])
    (kPlusOutputs : kPlusStored.outputs = #[.matrix scalarType])
    (noisyOutputs : noisyStored.outputs = #[.matrix scalarType])
    (oneTraced : traceValueAt trace (occurrenceOf stage #[] oneWire) =
      some ⟨.matrix encodingType, one⟩)
    (circuitTraced : traceValueAt trace (occurrenceOf stage #[] circuitWire) =
      some ⟨.matrix encodingType, circuit⟩)
    (preimageTraced : traceValueAt trace (occurrenceOf stage #[] preimageWire) =
      some ⟨.preimage preimageType, rPreimage⟩)
    (kTraced : traceValueAt trace (occurrenceOf stage #[] kWire) =
      some ⟨.matrix scalarType, kProjection⟩)
    (decoderTraced : traceValueAt trace (occurrenceOf stage #[] decoderWire) =
      some ⟨.matrix scalarType, decoder⟩) :
    traceValueAt trace
      (occurrenceOf stage #[] { scope := scope, node := noisyNode, port := 0 }) =
        some ⟨.matrix scalarType, decoder - (kProjection +
          Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
            (one - circuit) rPreimage.exactMatrix)⟩ := by
  have differenceArgumentsEq := differenceRun.binaryArguments_eq differenceArguments
    ⟨.matrix encodingType, one⟩ ⟨.matrix encodingType, circuit⟩ oneTraced circuitTraced
  have differenceExecution := differenceRun.primitiveEvaluated
  rw [differenceArgumentsEq, differenceOutputs] at differenceExecution
  have differenceValues := runtimeMatrixSubtractExecution oracle encodingValid
    stage scope differenceNode one circuit differenceRun.nodeResult differenceExecution
  have differenceOutput : differenceRun.output =
      ⟨.matrix encodingType, one - circuit⟩ :=
    differenceRun.output_eq_of_singleton _ differenceValues
  have differenceTraced : traceValueAt trace
      (occurrenceOf stage #[] { scope := scope, node := differenceNode, port := 0 }) =
        some ⟨.matrix encodingType, one - circuit⟩ := by
    simpa [differenceOutput] using differenceRun.outputTraced

  have projectedArgumentsEq := projectedRun.binaryArguments_eq projectedArguments
    ⟨.matrix encodingType, one - circuit⟩ ⟨.preimage preimageType, rPreimage⟩
    differenceTraced preimageTraced
  have projectedExecution := projectedRun.primitiveEvaluated
  rw [projectedArgumentsEq, projectedOutputs] at projectedExecution
  have projectedValues := runtimeApplyPreimageExecution oracle productValid
    stage scope projectedNode (one - circuit) rPreimage projectedRun.nodeResult projectedExecution
  have projectedOutput : projectedRun.output = ⟨.matrix scalarType,
      Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
        (one - circuit) rPreimage.exactMatrix⟩ :=
    projectedRun.output_eq_of_singleton _ projectedValues
  have projectedTraced : traceValueAt trace
      (occurrenceOf stage #[] { scope := scope, node := projectedNode, port := 0 }) =
        some ⟨.matrix scalarType,
          Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
            (one - circuit) rPreimage.exactMatrix⟩ := by
    simpa [projectedOutput] using projectedRun.outputTraced

  have kPlusArgumentsEq := kPlusRun.binaryArguments_eq kPlusArguments
    ⟨.matrix scalarType, kProjection⟩
    ⟨.matrix scalarType,
      Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
        (one - circuit) rPreimage.exactMatrix⟩ kTraced projectedTraced
  have kPlusExecution := kPlusRun.primitiveEvaluated
  rw [kPlusArgumentsEq, kPlusOutputs] at kPlusExecution
  have kPlusValues := runtimeMatrixAddExecution oracle productValid.2.2.1
    stage scope kPlusNode kProjection
    (Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
      (one - circuit) rPreimage.exactMatrix) kPlusRun.nodeResult kPlusExecution
  have kPlusOutput : kPlusRun.output = ⟨.matrix scalarType,
      kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
        (one - circuit) rPreimage.exactMatrix⟩ :=
    kPlusRun.output_eq_of_singleton _ kPlusValues
  have kPlusTraced : traceValueAt trace
      (occurrenceOf stage #[] { scope := scope, node := kPlusNode, port := 0 }) =
        some ⟨.matrix scalarType,
          kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
            (one - circuit) rPreimage.exactMatrix⟩ := by
    simpa [kPlusOutput] using kPlusRun.outputTraced

  have noisyArgumentsEq := noisyRun.binaryArguments_eq noisyArguments
    ⟨.matrix scalarType, decoder⟩
    ⟨.matrix scalarType,
      kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
        (one - circuit) rPreimage.exactMatrix⟩ decoderTraced kPlusTraced
  have noisyExecution := noisyRun.primitiveEvaluated
  rw [noisyArgumentsEq, noisyOutputs] at noisyExecution
  have noisyValues := runtimeMatrixSubtractExecution oracle productValid.2.2.1
    stage scope noisyNode decoder
    (kProjection + Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
      (one - circuit) rPreimage.exactMatrix) noisyRun.nodeResult noisyExecution
  have noisyOutput : noisyRun.output = ⟨.matrix scalarType,
      decoder - (kProjection +
        Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
          (one - circuit) rPreimage.exactMatrix)⟩ :=
    noisyRun.output_eq_of_singleton _ noisyValues
  simpa [noisyOutput] using noisyRun.outputTraced

/- `StoredNodeRef` keeps the three concrete node fields separately so the
   generated topology proof can inspect them.  Reassembling the node here
   gives the exact type expected by the generic reached-primitive hook. -/
def StoredNodeRef.concreteNode {program : Program} (site : StoredNodeRef program) : Node := {
  payload := site.payload
  arguments := site.arguments
  outputs := site.outputs
}

/- The candidate emitter uses this canonical runtime type for matrices whose
   exact semantics live in `ExactMatrix q n rows columns`. -/
abbrev exactRuntimeMatrixType := Mxx.Runtime.naturalMatrixType

def runtimeMatrixValue_exactRuntimeMatrixType (q n rows columns : Nat) :
    RuntimeMatrixValue (exactRuntimeMatrixType q n rows columns) =
      ExactMatrix q n rows columns :=
  Mxx.Runtime.matrixValue_naturalMatrixType q n rows columns

noncomputable def exactToRuntimeMatrix {q n rows columns : Nat}
    (value : ExactMatrix q n rows columns) :
    RuntimeMatrixValue (exactRuntimeMatrixType q n rows columns) :=
  (runtimeMatrixValue_exactRuntimeMatrixType q n rows columns).symm ▸ value

noncomputable def runtimeToExactMatrix {q n rows columns : Nat}
    (value : RuntimeMatrixValue (exactRuntimeMatrixType q n rows columns)) :
    ExactMatrix q n rows columns :=
  Mxx.Runtime.naturalToExact value

/- At Diamond's concrete `1 × g`, `g × 1`, `1 × 1` descriptors,
   runtime multiplication is ordinary exact matrix multiplication.  The
   assumption `1 < g` excludes both scalar shortcuts in `exactMultiply`; each
   same-ring cast is then reflexive. -/
theorem runtimeToExactMatrix_exactMultiply {q n gadgetColumns : Nat}
    (hg : 1 < gadgetColumns)
    (valid :
      (exactRuntimeMatrixType q n 1 gadgetColumns).Valid ∧
      (exactRuntimeMatrixType q n gadgetColumns 1).Valid ∧
      (exactRuntimeMatrixType q n 1 1).Valid ∧
      matrixProductType (exactRuntimeMatrixType q n 1 gadgetColumns)
        (exactRuntimeMatrixType q n gadgetColumns 1)
        (exactRuntimeMatrixType q n 1 1))
    (left : RuntimeMatrixValue (exactRuntimeMatrixType q n 1 gadgetColumns))
    (right : RuntimeMatrixValue (exactRuntimeMatrixType q n gadgetColumns 1)) :
    runtimeToExactMatrix (Mxx.Runtime.exactMultiply
      (exactRuntimeMatrixType q n 1 gadgetColumns)
      (exactRuntimeMatrixType q n gadgetColumns 1)
      (exactRuntimeMatrixType q n 1 1)
      valid left right) =
      runtimeToExactMatrix left * runtimeToExactMatrix right := by
  exact Mxx.Runtime.exactMultiply_natural_row_column hg valid left right

/- Generated Diamond code fills this record with the four theorems named
   `<node>ReachedPrimitiveRunFromPublicEval`.  The record fixes every run to
   the candidate's named stored node, and its finite array equalities are
   concrete graph facts discharged by reduction.  No semantic matrix or
   noise equation is a field. -/
structure CandidateFusePrimitiveRuns
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (shape : candidate.HasDiamondGraphShape) (trace : RuntimeTrace oracle) where
  stage : Nat
  scope : Nat
  oneMinusStage : shape.decryptionSites.oneMinusCircuit.stage = stage
  projectedStage : shape.decryptionSites.projectedDifference.stage = stage
  kPlusStage : shape.decryptionSites.kPlusProjection.stage = stage
  noisyStage : shape.decryptionSites.noisyPlaintext.stage = stage
  oneMinusScope : shape.decryptionSites.oneMinusCircuit.scope = scope
  projectedScope : shape.decryptionSites.projectedDifference.scope = scope
  kPlusScope : shape.decryptionSites.kPlusProjection.scope = scope
  noisyScope : shape.decryptionSites.noisyPlaintext.scope = scope
  oneMinusArguments : shape.decryptionSites.oneMinusCircuit.arguments = #[
    shape.decryptionSites.oneProjection.reference.2.wire,
    shape.decryptionSites.circuitOutput.reference.2.wire]
  projectedArguments : shape.decryptionSites.projectedDifference.arguments = #[
    shape.decryptionSites.oneMinusCircuit.reference.2.wire,
    shape.decryptionSites.rDecomposition.reference.2.wire]
  kPlusArguments : shape.decryptionSites.kPlusProjection.arguments = #[
    shape.decryptionSites.kProjection.reference.2.wire,
    shape.decryptionSites.projectedDifference.reference.2.wire]
  noisyArguments : shape.decryptionSites.noisyPlaintext.arguments = #[
    shape.decryptionSites.decoderProjection.reference.2.wire,
    shape.decryptionSites.kPlusProjection.reference.2.wire]
  oneMinusOutputWire : shape.decryptionSites.oneMinusCircuit.reference.2.wire = {
    scope := scope
    node := shape.decryptionSites.oneMinusCircuit.reference.2.wire.node
    port := 0
  }
  projectedOutputWire : shape.decryptionSites.projectedDifference.reference.2.wire = {
    scope := scope
    node := shape.decryptionSites.projectedDifference.reference.2.wire.node
    port := 0
  }
  kPlusOutputWire : shape.decryptionSites.kPlusProjection.reference.2.wire = {
    scope := scope
    node := shape.decryptionSites.kPlusProjection.reference.2.wire.node
    port := 0
  }
  oneMinus : RootPrimitiveRun trace stage scope
    shape.decryptionSites.oneMinusCircuit.reference.2.wire.node
    (.matrixBinary .subtract) shape.decryptionSites.oneMinusCircuit.concreteNode 0
  projected : RootPrimitiveRun trace stage scope
    shape.decryptionSites.projectedDifference.reference.2.wire.node
    .applyPreimage shape.decryptionSites.projectedDifference.concreteNode 0
  kPlus : RootPrimitiveRun trace stage scope
    shape.decryptionSites.kPlusProjection.reference.2.wire.node
    (.matrixBinary .add) shape.decryptionSites.kPlusProjection.concreteNode 0
  noisy : RootPrimitiveRun trace stage scope
    shape.decryptionSites.noisyPlaintext.reference.2.wire.node
    (.matrixBinary .subtract) shape.decryptionSites.noisyPlaintext.concreteNode 0

/- This candidate-level wrapper consumes only generated run/topology facts and
   typed values already present on the five incoming producer wires.  It then
   delegates all arithmetic and SSA chaining to `runtimeFuseTrace`. -/
theorem CandidateFusePrimitiveRuns.noisyTrace
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {shape : candidate.HasDiamondGraphShape} {trace : RuntimeTrace oracle}
    (runs : CandidateFusePrimitiveRuns oracle candidate shape trace)
    {encodingType preimageType scalarType : MatrixType}
    (encodingValid : encodingType.Valid)
    (productValid : encodingType.Valid ∧ preimageType.Valid ∧ scalarType.Valid ∧
      matrixProductType encodingType preimageType scalarType)
    (one circuit : RuntimeMatrixValue encodingType)
    (rPreimage : Mxx.Runtime.PreimageValue preimageType)
    (kProjection decoder : RuntimeMatrixValue scalarType)
    (oneMinusOutputs : shape.decryptionSites.oneMinusCircuit.outputs =
      #[.matrix encodingType])
    (projectedOutputs : shape.decryptionSites.projectedDifference.outputs =
      #[.matrix scalarType])
    (kPlusOutputs : shape.decryptionSites.kPlusProjection.outputs = #[.matrix scalarType])
    (noisyOutputs : shape.decryptionSites.noisyPlaintext.outputs = #[.matrix scalarType])
    (oneTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.oneProjection.reference.2.wire) =
        some ⟨.matrix encodingType, one⟩)
    (circuitTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.circuitOutput.reference.2.wire) =
        some ⟨.matrix encodingType, circuit⟩)
    (preimageTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.rDecomposition.reference.2.wire) =
        some ⟨.preimage preimageType, rPreimage⟩)
    (kTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.kProjection.reference.2.wire) =
        some ⟨.matrix scalarType, kProjection⟩)
    (decoderTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.decoderProjection.reference.2.wire) =
        some ⟨.matrix scalarType, decoder⟩) :
    traceValueAt trace (occurrenceOf runs.stage #[] {
      scope := runs.scope
      node := shape.decryptionSites.noisyPlaintext.reference.2.wire.node
      port := 0
    }) = some ⟨.matrix scalarType, decoder - (kProjection +
      Mxx.Runtime.exactMultiply encodingType preimageType scalarType productValid
        (one - circuit) rPreimage.exactMatrix)⟩ := by
  apply runtimeFuseTrace oracle encodingValid productValid one circuit rPreimage
    kProjection decoder runs.oneMinus runs.projected runs.kPlus runs.noisy
  · exact runs.oneMinusArguments
  · have stored := runs.projectedArguments
    rw [runs.oneMinusOutputWire] at stored
    exact stored
  · have stored := runs.kPlusArguments
    rw [runs.projectedOutputWire] at stored
    exact stored
  · have stored := runs.noisyArguments
    rw [runs.kPlusOutputWire] at stored
    exact stored
  · exact oneMinusOutputs
  · exact projectedOutputs
  · exact kPlusOutputs
  · exact noisyOutputs
  · exact oneTraced
  · exact circuitTraced
  · exact preimageTraced
  · exact kTraced
  · exact decoderTraced

/- A certified BGG run and the ordinary Boolean evaluator read the same final
   active slot.  Hence an accepting Boolean evaluation proves that the
   selected encoding carries Boolean one; its error bound is inherited from
   the final certified layer rather than supplied at the application boundary. -/
theorem acceptingCertifiedBggOutput
    {shape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit shape}
    {q n secretColumns gadgetColumns : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : Mxx.Bgg.BooleanEncodingValue mask gadget}
    {oneCarries : Mxx.Bgg.EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : Mxx.Bgg.ExactLayerState mask gadget oneMessage shape.inputWidth}
    (instanceBits : Fin shape.instanceWidth → Bool)
    (witnessBits : Fin shape.witnessWidth → Bool)
    (initialRuntime : Array.ofFn initial.bits =
      (Array.ofFn instanceBits).append (Array.ofFn witnessBits))
    (final : Mxx.Bgg.ExactLayerState mask gadget oneMessage
      (circuit.activeWidth (Mxx.Bgg.finalLayer valid)))
    (run : Mxx.Bgg.CertifiedLayeredRun valid one oneCarries oneMessageIdempotent initial
      shape.depth ⟨_, final⟩)
    (accepted : circuit.evaluate valid instanceBits witnessBits = some true) :
    Mxx.Bgg.EncodingCarriesBool
        (final.values (Mxx.Bgg.outputIndex circuit valid)).encoding oneMessage true ∧
      Mxx.Bgg.EncodingErrorWithin
        (final.values (Mxx.Bgg.outputIndex circuit valid)).encoding final.noiseBound := by
  have evaluated := Mxx.Bgg.CertifiedLayeredRun.evaluateUnchecked instanceBits witnessBits
    initialRuntime final run
  have selectedTrue : final.bits (Mxx.Bgg.outputIndex circuit valid) = true := by
    unfold Mxx.Gadgets.LayeredBoolCircuit.evaluate at accepted
    rw [evaluated] at accepted
    exact Option.some.inj accepted
  constructor
  · rw [← selectedTrue]
    exact Mxx.Bgg.selectedOutput_carries valid final
  · exact Mxx.Bgg.selectedOutput_within valid final

/- Subtracting two encodings of the same Boolean message cancels only the
   payload term:

     C_one - C_out = s * (A_one - A_out) + (e_one - e_out).

   Multiplication by the exact decomposition `D` therefore approximates
   `s * ((A_one - A_out) * D)`.  One output coefficient of the negacyclic
   matrix product pays exactly `gadgetColumns * n`, never `n^2`. -/
noncomputable def bggDifferenceProjectionWithin
    {q n secretColumns gadgetColumns projectionColumns : Nat}
    (hn : 0 < n)
    {oneCiphertext circuitCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic circuitPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (oneEncoding : Mxx.Bgg.Encoding oneCiphertext mask payload onePublic gadget message)
    (circuitEncoding : Mxx.Bgg.Encoding
      circuitCiphertext mask payload circuitPublic gadget message)
    {decomposition : ExactMatrix q n gadgetColumns projectionColumns}
    {decompositionBound oneBound circuitBound : Nat}
    (decompositionLift : BoundedLift decomposition decompositionBound)
    (oneWithin : Mxx.Bgg.EncodingErrorWithin oneEncoding oneBound)
    (circuitWithin : Mxx.Bgg.EncodingErrorWithin circuitEncoding circuitBound) :
    ApproxWithin ((oneCiphertext - circuitCiphertext) * decomposition)
      (mask * ((onePublic - circuitPublic) * decomposition))
      (gadgetColumns * n * (oneBound + circuitBound) * decompositionBound) := by
  let difference := Mxx.Bgg.sub oneEncoding circuitEncoding
  refine {
    toApprox := {
      error := difference.error * decompositionLift.witness
      equation := ?_
    }
    norm_le := ?_
  }
  · have differenceEquation := difference.equation
    simp only [sub_self, zero_smul, sub_zero] at differenceEquation
    have reducedProduct :
        reduceMatrix q n 1 gadgetColumns difference.error * decomposition =
          reduceMatrix q n 1 projectionColumns
            (difference.error * decompositionLift.witness) := by
      calc
        reduceMatrix q n 1 gadgetColumns difference.error * decomposition =
            reduceMatrix q n 1 gadgetColumns difference.error *
              reduceMatrix q n gadgetColumns projectionColumns
                decompositionLift.witness :=
          congrArg (fun value ↦
            reduceMatrix q n 1 gadgetColumns difference.error * value)
            decompositionLift.reduce_eq.symm
        _ = reduceMatrix q n 1 projectionColumns
            (difference.error * decompositionLift.witness) :=
          (reduceMatrix_mul q n 1 gadgetColumns projectionColumns _ _).symm
    calc
      (oneCiphertext - circuitCiphertext) * decomposition =
          (mask * (onePublic - circuitPublic) +
            reduceMatrix q n 1 gadgetColumns difference.error) * decomposition :=
        congrArg (fun value ↦ value * decomposition) differenceEquation
      _ = mask * ((onePublic - circuitPublic) * decomposition) +
          reduceMatrix q n 1 projectionColumns
            (difference.error * decompositionLift.witness) := by
        rw [Matrix.add_mul, Matrix.mul_assoc, reducedProduct]
  · apply (matrixNorm_mul_le hn).trans
    calc
      gadgetColumns * n * matrixNorm difference.error *
            matrixNorm decompositionLift.witness ≤
          gadgetColumns * n * (oneBound + circuitBound) * decompositionBound := by
        exact Nat.mul_le_mul
          (Nat.mul_le_mul_left (gadgetColumns * n)
            (Mxx.Bgg.sub_gate_error_within oneEncoding circuitEncoding
              oneWithin circuitWithin))
          decompositionLift.norm_le

/- Two independently typed encodings that both carry Boolean true have the
   same payload message.  This small transport is what lets the certified BGG
   output feed the subtraction lemma without globally normalizing exact
   public terms. -/
noncomputable def bggAcceptedDifferenceProjectionWithin
    {q n secretColumns gadgetColumns projectionColumns : Nat}
    (hn : 0 < n)
    {oneCiphertext circuitCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic circuitPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage circuitMessage acceptedMessage : ExactPoly q n}
    (oneEncoding : Mxx.Bgg.Encoding
      oneCiphertext mask payload onePublic gadget oneMessage)
    (circuitEncoding : Mxx.Bgg.Encoding
      circuitCiphertext mask payload circuitPublic gadget circuitMessage)
    (oneCarries : Mxx.Bgg.EncodingCarriesBool oneEncoding acceptedMessage true)
    (circuitCarries : Mxx.Bgg.EncodingCarriesBool circuitEncoding acceptedMessage true)
    {decomposition : ExactMatrix q n gadgetColumns projectionColumns}
    {decompositionBound oneBound circuitBound : Nat}
    (decompositionLift : BoundedLift decomposition decompositionBound)
    (oneWithin : Mxx.Bgg.EncodingErrorWithin oneEncoding oneBound)
    (circuitWithin : Mxx.Bgg.EncodingErrorWithin circuitEncoding circuitBound) :
    ApproxWithin ((oneCiphertext - circuitCiphertext) * decomposition)
      (mask * ((onePublic - circuitPublic) * decomposition))
      (gadgetColumns * n * (oneBound + circuitBound) * decompositionBound) := by
  have oneMessageEq : oneMessage = acceptedMessage := by
    simpa [Mxx.Bgg.EncodingCarriesBool, Mxx.Bgg.IsBooleanMessage,
      Mxx.Bgg.boolMessage] using oneCarries
  have circuitMessageEq : circuitMessage = acceptedMessage := by
    simpa [Mxx.Bgg.EncodingCarriesBool, Mxx.Bgg.IsBooleanMessage,
      Mxx.Bgg.boolMessage] using circuitCarries
  cases oneMessageEq
  cases circuitMessageEq
  exact bggDifferenceProjectionWithin hn oneEncoding circuitEncoding decompositionLift
    oneWithin circuitWithin

/- Linear fuse composition.  The actual runtime expression is

     decoder - (kProjection + projectedDifference).

   Each input approximation contributes its integer error with the same
   signs, so the triangle inequality gives the sum of the three bounds. -/
noncomputable def fuseApproximations
    {q n rows columns decoderBound kBound projectedBound : Nat}
    {decoder kProjection projectedDifference : ExactMatrix q n rows columns}
    {decoderIdeal kIdeal projectedIdeal : ExactMatrix q n rows columns}
    (decoderWithin : ApproxWithin decoder decoderIdeal decoderBound)
    (kWithin : ApproxWithin kProjection kIdeal kBound)
    (projectedWithin : ApproxWithin projectedDifference projectedIdeal projectedBound) :
    ApproxWithin (decoder - (kProjection + projectedDifference))
      (decoderIdeal - (kIdeal + projectedIdeal))
      (decoderBound + kBound + projectedBound) := by
  refine {
    toApprox := {
      error := decoderWithin.error - (kWithin.error + projectedWithin.error)
      equation := ?_
    }
    norm_le := ?_
  }
  · have decoderEquation := decoderWithin.equation
    have kEquation := kWithin.equation
    have projectedEquation := projectedWithin.equation
    have reduceError :
        reduceMatrix q n rows columns
            (decoderWithin.error - (kWithin.error + projectedWithin.error)) =
          reduceMatrix q n rows columns decoderWithin.error -
            (reduceMatrix q n rows columns kWithin.error +
              reduceMatrix q n rows columns projectedWithin.error) := by
      ext row column
      simp [reduceMatrix_apply, sub_eq_add_neg]
    calc
      decoder - (kProjection + projectedDifference) =
          (decoderIdeal + reduceMatrix q n rows columns decoderWithin.error) -
            ((kIdeal + reduceMatrix q n rows columns kWithin.error) +
              (projectedIdeal + reduceMatrix q n rows columns projectedWithin.error)) := by
        exact congrArg₂ (fun left right ↦ left - right) decoderEquation
          (congrArg₂ (fun left right ↦ left + right) kEquation projectedEquation)
      _ = decoderIdeal - (kIdeal + projectedIdeal) +
          reduceMatrix q n rows columns
            (decoderWithin.error - (kWithin.error + projectedWithin.error)) := by
        rw [reduceError]
        abel
  · apply (Mxx.Bgg.matrixNorm_sub_le _ _).trans
    calc
      matrixNorm decoderWithin.error + matrixNorm (kWithin.error + projectedWithin.error) ≤
          decoderBound + (kBound + projectedBound) := by
        exact Nat.add_le_add decoderWithin.norm_le
          ((matrixNorm_add_le _ _).trans
            (Nat.add_le_add kWithin.norm_le projectedWithin.norm_le))
      _ = decoderBound + kBound + projectedBound := by omega

/- Diamond's accepting fuse specializes the linear composition above.  The
   decoder ideal is `shared + projectedIdeal + center`, whereas the K
   projection ideal is `shared`.  These exact public terms cancel locally;
   the proof never expands or normalizes unrelated public matrices. -/
noncomputable def acceptingFuseWithin
    {q n rows columns payloadBound projectedBound : Nat}
    {decoder kProjection projectedDifference : ExactMatrix q n rows columns}
    {shared projectedIdeal center : ExactMatrix q n rows columns}
    (decoderWithin : ApproxWithin decoder (shared + projectedIdeal + center) payloadBound)
    (kWithin : ApproxWithin kProjection shared payloadBound)
    (projectedWithin : ApproxWithin projectedDifference projectedIdeal projectedBound) :
    ApproxWithin (decoder - (kProjection + projectedDifference)) center
      (2 * payloadBound + projectedBound) := by
  have combined := fuseApproximations decoderWithin kWithin projectedWithin
  convert combined using 1
  · abel
  · omega

/- Complete application-level BGG/fuse invariant.  When the certified circuit
   output carries one, `oneEncoding` and `circuitEncoding` have the same
   message, so their payload terms cancel.  The actual four-node fuse output
   is within

     2 * payloadBound
       + gadgetColumns * n * (oneBound + circuitBound) * decompositionBound

   of the Boolean message center. -/
noncomputable def bggAcceptingFuseWithin
    {q n secretColumns gadgetColumns projectionColumns payloadBound
      decompositionBound oneBound circuitBound : Nat}
    (hn : 0 < n)
    {oneCiphertext circuitCiphertext : ExactMatrix q n 1 gadgetColumns}
    {mask payload : ExactMatrix q n 1 secretColumns}
    {onePublic circuitPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {message : ExactPoly q n}
    (oneEncoding : Mxx.Bgg.Encoding oneCiphertext mask payload onePublic gadget message)
    (circuitEncoding : Mxx.Bgg.Encoding
      circuitCiphertext mask payload circuitPublic gadget message)
    {decomposition : ExactMatrix q n gadgetColumns projectionColumns}
    (decompositionLift : BoundedLift decomposition decompositionBound)
    (oneWithin : Mxx.Bgg.EncodingErrorWithin oneEncoding oneBound)
    (circuitWithin : Mxx.Bgg.EncodingErrorWithin circuitEncoding circuitBound)
    {decoder kProjection shared center : ExactMatrix q n 1 projectionColumns}
    (decoderWithin : ApproxWithin decoder
      (shared + mask * ((onePublic - circuitPublic) * decomposition) + center) payloadBound)
    (kWithin : ApproxWithin kProjection shared payloadBound) :
    ApproxWithin
      (decoder - (kProjection + (oneCiphertext - circuitCiphertext) * decomposition))
      center
      (2 * payloadBound +
        gadgetColumns * n * (oneBound + circuitBound) * decompositionBound) := by
  let projectedWithin := bggDifferenceProjectionWithin hn oneEncoding circuitEncoding
    decompositionLift oneWithin circuitWithin
  exact acceptingFuseWithin decoderWithin kWithin projectedWithin

/- Candidate-level accepting BGG/fuse certificate.  The actual trace output
   comes from the four generated primitive runs, while the semantic bound is
   assembled from the certified accepting BGG layer and the two lower payload
   approximations:

       2 * payloadBound
         + gadgetColumns * n * (payloadBound + final.noiseBound)
             * decompositionBound.

   The local `runtimeToExactMatrix_exactMultiply` bridge identifies the runtime
   `exactMultiply` result with ordinary exact matrix multiplication at these
   concrete types.  There is no premise stating the fused output equation or
   its bound. -/
noncomputable def CandidateFusePrimitiveRuns.acceptingNoisyWithin
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {graphShape : candidate.HasDiamondGraphShape} {trace : RuntimeTrace oracle}
    (runs : CandidateFusePrimitiveRuns oracle candidate graphShape trace)
    {boolShape : Mxx.Gadgets.LayeredBoolCircuitShape}
    {circuit : Mxx.Gadgets.LayeredBoolCircuit boolShape}
    {q n secretColumns gadgetColumns payloadBound
      decompositionBound : Nat}
    {mask : ExactMatrix q n 1 secretColumns}
    {gadget : ExactMatrix q n secretColumns gadgetColumns}
    {oneMessage : ExactPoly q n}
    {valid : circuit.Valid}
    {one : Mxx.Bgg.BooleanEncodingValue mask gadget}
    {oneCarries : Mxx.Bgg.EncodingCarriesBool one.encoding oneMessage true}
    {oneMessageIdempotent : oneMessage * oneMessage = oneMessage}
    {initial : Mxx.Bgg.ExactLayerState mask gadget oneMessage boolShape.inputWidth}
    (instanceBits : Fin boolShape.instanceWidth → Bool)
    (witnessBits : Fin boolShape.witnessWidth → Bool)
    (initialRuntime : Array.ofFn initial.bits =
      (Array.ofFn instanceBits).append (Array.ofFn witnessBits))
    (final : Mxx.Bgg.ExactLayerState mask gadget oneMessage
      (circuit.activeWidth (Mxx.Bgg.finalLayer valid)))
    (certified : Mxx.Bgg.CertifiedLayeredRun valid one oneCarries
      oneMessageIdempotent initial boolShape.depth ⟨_, final⟩)
    (accepted : circuit.evaluate valid instanceBits witnessBits = some true)
    (hn : 0 < n)
    (encodingValid : (exactRuntimeMatrixType q n 1 gadgetColumns).Valid)
    (productValid :
      (exactRuntimeMatrixType q n 1 gadgetColumns).Valid ∧
      (exactRuntimeMatrixType q n gadgetColumns 1).Valid ∧
      (exactRuntimeMatrixType q n 1 1).Valid ∧
      matrixProductType (exactRuntimeMatrixType q n 1 gadgetColumns)
        (exactRuntimeMatrixType q n gadgetColumns 1)
        (exactRuntimeMatrixType q n 1 1))
    (rPreimage : Mxx.Runtime.PreimageValue
      (exactRuntimeMatrixType q n gadgetColumns 1))
    (kProjection decoder : RuntimeMatrixValue (exactRuntimeMatrixType q n 1 1))
    (shared center : ExactMatrix q n 1 1)
    (oneMinusOutputs : graphShape.decryptionSites.oneMinusCircuit.outputs =
      #[.matrix (exactRuntimeMatrixType q n 1 gadgetColumns)])
    (projectedOutputs : graphShape.decryptionSites.projectedDifference.outputs =
      #[.matrix (exactRuntimeMatrixType q n 1 1)])
    (kPlusOutputs : graphShape.decryptionSites.kPlusProjection.outputs =
      #[.matrix (exactRuntimeMatrixType q n 1 1)])
    (noisyOutputs : graphShape.decryptionSites.noisyPlaintext.outputs =
      #[.matrix (exactRuntimeMatrixType q n 1 1)])
    (oneTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      graphShape.decryptionSites.oneProjection.reference.2.wire) =
        some ⟨.matrix (exactRuntimeMatrixType q n 1 gadgetColumns),
          exactToRuntimeMatrix one.ciphertext⟩)
    (circuitTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      graphShape.decryptionSites.circuitOutput.reference.2.wire) = some
        ⟨.matrix (exactRuntimeMatrixType q n 1 gadgetColumns),
          exactToRuntimeMatrix
            (final.values (Mxx.Bgg.outputIndex circuit valid)).ciphertext⟩)
    (preimageTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      graphShape.decryptionSites.rDecomposition.reference.2.wire) = some
        ⟨.preimage (exactRuntimeMatrixType q n gadgetColumns 1), rPreimage⟩)
    (kTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      graphShape.decryptionSites.kProjection.reference.2.wire) = some
        ⟨.matrix (exactRuntimeMatrixType q n 1 1), kProjection⟩)
    (decoderTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      graphShape.decryptionSites.decoderProjection.reference.2.wire) = some
        ⟨.matrix (exactRuntimeMatrixType q n 1 1), decoder⟩)
    (hg : 1 < gadgetColumns)
    (decompositionLift : BoundedLift
      (runtimeToExactMatrix rPreimage.exactMatrix) decompositionBound)
    (oneWithin : Mxx.Bgg.EncodingErrorWithin one.encoding payloadBound)
    (decoderWithin : ApproxWithin (runtimeToExactMatrix decoder)
      (shared + mask * ((one.publicMatrix -
        (final.values (Mxx.Bgg.outputIndex circuit valid)).publicMatrix) *
          runtimeToExactMatrix rPreimage.exactMatrix) + center) payloadBound)
    (kWithin : ApproxWithin (runtimeToExactMatrix kProjection) shared payloadBound) :
    ∃ noisy : ExactMatrix q n 1 1,
      traceValueAt trace (occurrenceOf runs.stage #[] {
        scope := runs.scope
        node := graphShape.decryptionSites.noisyPlaintext.reference.2.wire.node
        port := 0
      }) = some ⟨.matrix (exactRuntimeMatrixType q n 1 1),
        exactToRuntimeMatrix noisy⟩ ∧
      Nonempty (ApproxWithin noisy center
        (2 * payloadBound + gadgetColumns * n *
          (payloadBound + final.noiseBound) * decompositionBound)) := by
  obtain ⟨circuitCarries, circuitWithin⟩ := acceptingCertifiedBggOutput
    instanceBits witnessBits initialRuntime final certified accepted
  have noisyTraced := runs.noisyTrace encodingValid productValid
    (exactToRuntimeMatrix one.ciphertext)
    (exactToRuntimeMatrix
      (final.values (Mxx.Bgg.outputIndex circuit valid)).ciphertext) rPreimage
    kProjection decoder oneMinusOutputs projectedOutputs kPlusOutputs noisyOutputs
    oneTraced circuitTraced preimageTraced kTraced decoderTraced
  have projectedWithin := bggAcceptedDifferenceProjectionWithin hn one.encoding
    (final.values (Mxx.Bgg.outputIndex circuit valid)).encoding oneCarries circuitCarries
    decompositionLift oneWithin circuitWithin
  have runtimeProjectedWithin : ApproxWithin
      (runtimeToExactMatrix (Mxx.Runtime.exactMultiply
        (exactRuntimeMatrixType q n 1 gadgetColumns)
        (exactRuntimeMatrixType q n gadgetColumns 1)
        (exactRuntimeMatrixType q n 1 1) productValid
        (exactToRuntimeMatrix one.ciphertext - exactToRuntimeMatrix
          (final.values (Mxx.Bgg.outputIndex circuit valid)).ciphertext)
        rPreimage.exactMatrix))
      (mask * ((one.publicMatrix -
        (final.values (Mxx.Bgg.outputIndex circuit valid)).publicMatrix) *
          runtimeToExactMatrix rPreimage.exactMatrix))
      (gadgetColumns * n * (payloadBound + final.noiseBound) * decompositionBound) := by
    rw [runtimeToExactMatrix_exactMultiply hg productValid]
    simpa [exactToRuntimeMatrix, runtimeToExactMatrix,
      runtimeMatrixValue_exactRuntimeMatrixType] using projectedWithin
  have fusedWithin := acceptingFuseWithin decoderWithin kWithin runtimeProjectedWithin
  let noisyRuntime := decoder - (kProjection +
    Mxx.Runtime.exactMultiply
      (exactRuntimeMatrixType q n 1 gadgetColumns)
      (exactRuntimeMatrixType q n gadgetColumns 1)
      (exactRuntimeMatrixType q n 1 1) productValid
      (exactToRuntimeMatrix one.ciphertext - exactToRuntimeMatrix
        (final.values (Mxx.Bgg.outputIndex circuit valid)).ciphertext)
      rPreimage.exactMatrix)
  let noisy := runtimeToExactMatrix noisyRuntime
  refine ⟨noisy, ?_, ⟨?_⟩⟩
  · simpa [noisy, noisyRuntime, exactToRuntimeMatrix, runtimeToExactMatrix,
      runtimeMatrixValue_exactRuntimeMatrixType] using noisyTraced
  · simpa [noisy, noisyRuntime, runtimeToExactMatrix,
      runtimeMatrixValue_exactRuntimeMatrixType] using fusedWithin

end Mxx.We.DiamondWE
