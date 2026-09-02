import MxxWe.DiamondWE.BggFuse

namespace Mxx.We.DiamondWE

open Mxx.IR

/- The decoder sites are named by generated typed references.  This constructor changes only the
   output port to zero; it never embeds a generated stage, scope, or raw node number. -/
def decoderOutputWire (scope : Nat) (site : StoredNodeRef program) : WireRef :=
  { scope := scope, node := site.reference.2.wire.node, port := 0 }

/- A unary primitive receives exactly the dynamic value stored on its sole SSA input wire. -/
theorem RootPrimitiveRun.unaryArguments_eq
    {backend : SemanticBackend} {trace : Trace backend}
    {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node} {port : Nat} {inputWire : WireRef}
    (run : RootPrimitiveRun trace stage scope node payload storedNode port)
    (argumentsStored : storedNode.arguments = #[inputWire])
    (input : DynamicValue backend)
    (inputTraced : traceValueAt trace (occurrenceOf stage #[] inputWire) = some input) :
    run.arguments = #[input] := by
  -- Argument resolution preserves the one-element arity recorded by the generated node.
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
  -- The evaluator trace identifies the resolved argument with the named producer value.
  obtain ⟨inputBound, argumentTraced⟩ := run.argumentTraced (index := 0)
    (by simp [argumentsStored])
  have inputEq : run.arguments[0] = input := by
    apply Option.some.inj
    calc
      some run.arguments[0] =
          traceValueAt trace (occurrenceOf stage #[] storedNode.arguments[0]) :=
        argumentTraced.symm
      _ = traceValueAt trace (occurrenceOf stage #[] inputWire) := by
        simp [argumentsStored]
      _ = some input := inputTraced
  -- Equal length and equality at index zero determine the complete unary operand array.
  apply Array.ext
  · simpa [argumentsStored] using argumentSize
  · intro index leftBound rightBound
    simp at rightBound
    have : index = 0 := by omega
    subst index
    simpa using inputEq

/- A nullary primitive resolves the empty operand array recorded by its generated node. -/
theorem RootPrimitiveRun.nullaryArguments_eq
    {backend : SemanticBackend} {trace : Trace backend}
    {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node} {port : Nat}
    (run : RootPrimitiveRun trace stage scope node payload storedNode port)
    (argumentsStored : storedNode.arguments = #[]) : run.arguments = #[] := by
  have resolved := run.argumentsResolved
  rw [argumentsStored] at resolved
  simp [resolveArguments] at resolved
  exact (Except.ok.inj resolved).symm

/- Once generated topology fixes operands and output types, primitive evaluation determines the
   value stored in the actual trace.  No caller equation about the reached output is accepted. -/
theorem RootPrimitiveRun.outputTrace_of_primitive
    {backend : SemanticBackend} {trace : Trace backend}
    {stage scope node : Nat} {payload : NodePayload} {storedNode : Node}
    (run : RootPrimitiveRun trace stage scope node payload storedNode 0)
    (arguments : Array (DynamicValue backend)) (outputs : Array WireType)
    (expected : DynamicValue backend)
    (argumentsEq : run.arguments = arguments) (outputsEq : storedNode.outputs = outputs)
    (evaluated : evalPrimitiveNode backend {} stage scope node payload arguments outputs =
      .ok (NodeResult.ofValues #[expected])) :
    traceValueAt trace (occurrenceOf stage #[] { scope := scope, node := node, port := 0 }) =
      some expected := by
  -- Rewrite the reached execution to the exact operands and result type certified by the graph.
  have actualExecution := run.primitiveEvaluated
  rw [argumentsEq, outputsEq, evaluated] at actualExecution
  have nodeResultEq : run.nodeResult = NodeResult.ofValues #[expected] :=
    (Except.ok.inj actualExecution).symm
  -- Port zero of that singleton result is therefore the primitive's computed value.
  have outputEq : run.output = expected := by
    apply run.output_eq_of_singleton expected
    rw [nodeResultEq]
    rfl
  -- The public evaluator stored the same reached value at the named occurrence.
  simpa [outputEq] using run.outputTraced

/- These are exactly the eleven root primitives of the generated Diamond threshold decoder.
   Every operand array and output type is a concrete generated graph fact; semantic output values
   are intentionally absent from the record. -/
structure CandidateDecoderPrimitiveRuns
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) (candidate : Candidate)
    (shape : candidate.HasDiamondGraphShape) (trace : RuntimeTrace oracle) where
  stage : Nat
  scope : Nat
  coefficientArguments : shape.decryptionSites.decoderCoefficient.arguments =
    #[shape.decryptionSites.noisyPlaintext.reference.2.wire]
  quarterArguments : shape.decryptionSites.decoderQuarter.arguments = #[]
  threeArguments : shape.decryptionSites.decoderThree.arguments = #[]
  threeQuarterArguments : shape.decryptionSites.decoderThreeQuarter.arguments = #[
    decoderOutputWire scope shape.decryptionSites.decoderQuarter,
    decoderOutputWire scope shape.decryptionSites.decoderThree]
  lowerArguments : shape.decryptionSites.decoderLowerComparison.arguments = #[
    decoderOutputWire scope shape.decryptionSites.decoderQuarter,
    decoderOutputWire scope shape.decryptionSites.decoderCoefficient]
  upperArguments : shape.decryptionSites.decoderUpperComparison.arguments = #[
    decoderOutputWire scope shape.decryptionSites.decoderCoefficient,
    decoderOutputWire scope shape.decryptionSites.decoderThreeQuarter]
  lowerIntArguments : shape.decryptionSites.decoderLowerBoolToInt.arguments =
    #[decoderOutputWire scope shape.decryptionSites.decoderLowerComparison]
  upperIntArguments : shape.decryptionSites.decoderUpperBoolToInt.arguments =
    #[decoderOutputWire scope shape.decryptionSites.decoderUpperComparison]
  sumArguments : shape.decryptionSites.decoderSum.arguments = #[
    decoderOutputWire scope shape.decryptionSites.decoderLowerBoolToInt,
    decoderOutputWire scope shape.decryptionSites.decoderUpperBoolToInt]
  twoArguments : shape.decryptionSites.decoderTwo.arguments = #[]
  decodedArguments : shape.decryptionSites.decoderEqualsTwo.arguments = #[
    decoderOutputWire scope shape.decryptionSites.decoderSum,
    decoderOutputWire scope shape.decryptionSites.decoderTwo]
  coefficientOutputs : shape.decryptionSites.decoderCoefficient.outputs = #[.int]
  quarterOutputs : shape.decryptionSites.decoderQuarter.outputs = #[.constantInt]
  threeOutputs : shape.decryptionSites.decoderThree.outputs = #[.constantInt]
  threeQuarterOutputs : shape.decryptionSites.decoderThreeQuarter.outputs = #[.int]
  lowerOutputs : shape.decryptionSites.decoderLowerComparison.outputs = #[.bool]
  upperOutputs : shape.decryptionSites.decoderUpperComparison.outputs = #[.bool]
  lowerIntOutputs : shape.decryptionSites.decoderLowerBoolToInt.outputs = #[.int]
  upperIntOutputs : shape.decryptionSites.decoderUpperBoolToInt.outputs = #[.int]
  sumOutputs : shape.decryptionSites.decoderSum.outputs = #[.int]
  twoOutputs : shape.decryptionSites.decoderTwo.outputs = #[.constantInt]
  decodedOutputs : shape.decryptionSites.decoderEqualsTwo.outputs = #[.bool]
  decodedOutputWire : candidate.refs.decodedOutput.wire =
    decoderOutputWire scope shape.decryptionSites.decoderEqualsTwo
  coefficient : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderCoefficient.reference.2.wire.node
    (.extractCoefficient (.literal 0) none)
    shape.decryptionSites.decoderCoefficient.concreteNode 0
  quarter : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderQuarter.reference.2.wire.node
    (quarterPayload candidate.parameters.modulus)
    shape.decryptionSites.decoderQuarter.concreteNode 0
  three : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderThree.reference.2.wire.node (.constantInt 3)
    shape.decryptionSites.decoderThree.concreteNode 0
  threeQuarter : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderThreeQuarter.reference.2.wire.node (.intBinary .multiply)
    shape.decryptionSites.decoderThreeQuarter.concreteNode 0
  lower : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderLowerComparison.reference.2.wire.node (.intCompare .lessEqual)
    shape.decryptionSites.decoderLowerComparison.concreteNode 0
  upper : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderUpperComparison.reference.2.wire.node (.intCompare .lessEqual)
    shape.decryptionSites.decoderUpperComparison.concreteNode 0
  lowerInt : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderLowerBoolToInt.reference.2.wire.node .boolToInt
    shape.decryptionSites.decoderLowerBoolToInt.concreteNode 0
  upperInt : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderUpperBoolToInt.reference.2.wire.node .boolToInt
    shape.decryptionSites.decoderUpperBoolToInt.concreteNode 0
  sum : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderSum.reference.2.wire.node (.intBinary .add)
    shape.decryptionSites.decoderSum.concreteNode 0
  two : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderTwo.reference.2.wire.node (.constantInt 2)
    shape.decryptionSites.decoderTwo.concreteNode 0
  decoded : RootPrimitiveRun trace stage scope
    shape.decryptionSites.decoderEqualsTwo.reference.2.wire.node (.intCompare .equal)
    shape.decryptionSites.decoderEqualsTwo.concreteNode 0

private theorem decoderQuarter_evaluated
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {q : Nat} (hq : 1 < q) :
    evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0 (quarterPayload q) #[]
      #[.constantInt] =
        .ok (NodeResult.ofValues #[⟨.constantInt, (decoderQuarter q : Int)⟩]) := by
  -- The generated expression `(q - 2 + 2) / 4` agrees with `decoderQuarter` when `q >= 2`.
  simp [evalPrimitiveNode, primitive, quarterPayload, decoderQuarter, StructuralIntExpr.eval]
  simp only [Except.mapError, pure, Functor.map, Except.pure, Except.map]
  change Except.ok (NodeResult.ofValues #[⟨.constantInt, (q : Int) / 4⟩]) =
    Except.ok (NodeResult.ofValues #[⟨.constantInt, (((q - 2 : Nat) : Int) + 2) / 4⟩])
  congr 4
  have castSub : ((q - 2 : Nat) : Int) = (q : Int) - 2 := by omega
  rw [castSub]
  ring_nf

/- Replaying the eleven reached primitives reconstructs the actual decoder pipeline.  The only
   external value fact is the noisy matrix at the decoder input; every later scalar is forced by
   an IR edge and the semantics of the primitive at that edge. -/
theorem CandidateDecoderPrimitiveRuns.decoderTrace
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {shape : candidate.HasDiamondGraphShape} {trace : RuntimeTrace oracle}
    (runs : CandidateDecoderPrimitiveRuns oracle candidate shape trace)
    {matrixType : MatrixType} (actual : RuntimeMatrixValue matrixType)
    (modulusValid : 1 < candidate.parameters.modulus)
    (noisyTraced : traceValueAt trace (occurrenceOf runs.stage #[]
      shape.decryptionSites.noisyPlaintext.reference.2.wire) =
        some ⟨.matrix matrixType, actual⟩) :
    ∃ coefficient decoded,
      Nonempty (DecoderPrimitiveChain oracle trace matrixType actual
        candidate.parameters.modulus coefficient decoded) ∧
      traceValueAt trace (occurrenceOf runs.stage #[] candidate.refs.decodedOutput.wire) =
        some ⟨.bool, decoded⟩ ∧
      decoded = decodeInterval candidate.parameters.modulus coefficient := by
  let q := candidate.parameters.modulus
  let coefficient := Mxx.Runtime.exactExtractCoefficient matrixType 0 actual
  let quarter : Int := decoderQuarter q
  let three : Int := 3
  let threeQuarter : Int := (3 * decoderQuarter q : Nat)
  let lower := decide (quarter ≤ coefficient)
  let upper := decide (coefficient ≤ threeQuarter)
  let lowerInt : Int := if lower then 1 else 0
  let upperInt : Int := if upper then 1 else 0
  let sum := lowerInt + upperInt
  let two : Int := 2
  let decoded := decide (sum = two)
  have coefficientArguments := runs.coefficient.unaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.coefficientArguments)
    ⟨.matrix matrixType, actual⟩ noisyTraced
  have coefficientTrace := runs.coefficient.outputTrace_of_primitive
    #[⟨.matrix matrixType, actual⟩] #[.int] ⟨.int, coefficient⟩ coefficientArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.coefficientOutputs) (by rfl)
  have quarterArguments := runs.quarter.nullaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.quarterArguments)
  have quarterTrace := runs.quarter.outputTrace_of_primitive #[] #[.constantInt]
    ⟨.constantInt, quarter⟩ quarterArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.quarterOutputs)
    (by simpa [q, quarter] using
      (decoderQuarter_evaluated (oracle := oracle) modulusValid))
  have threeArguments := runs.three.nullaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.threeArguments)
  have threeTrace := runs.three.outputTrace_of_primitive #[] #[.constantInt]
    ⟨.constantInt, three⟩ threeArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.threeOutputs) (by rfl)
  have threeQuarterArguments := runs.threeQuarter.binaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.threeQuarterArguments)
    ⟨.constantInt, quarter⟩ ⟨.constantInt, three⟩ quarterTrace threeTrace
  have threeQuarterTrace := runs.threeQuarter.outputTrace_of_primitive
    #[⟨.constantInt, quarter⟩, ⟨.constantInt, three⟩] #[.int]
    ⟨.int, threeQuarter⟩ threeQuarterArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.threeQuarterOutputs)
    (by simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntBinary,
      Except.instMonad, Except.bind, Except.pure, threeQuarter, three, quarter, q,
      Nat.cast_mul, mul_comm])
  have lowerArguments := runs.lower.binaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.lowerArguments)
    ⟨.constantInt, quarter⟩ ⟨.int, coefficient⟩ quarterTrace coefficientTrace
  have lowerTrace := runs.lower.outputTrace_of_primitive
    #[⟨.constantInt, quarter⟩, ⟨.int, coefficient⟩] #[.bool] ⟨.bool, lower⟩
    lowerArguments (by simpa [StoredNodeRef.concreteNode] using runs.lowerOutputs)
    (by simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
      Except.instMonad, Except.bind, Except.pure, lower])
  have upperArguments := runs.upper.binaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.upperArguments)
    ⟨.int, coefficient⟩ ⟨.int, threeQuarter⟩ coefficientTrace threeQuarterTrace
  have upperTrace := runs.upper.outputTrace_of_primitive
    #[⟨.int, coefficient⟩, ⟨.int, threeQuarter⟩] #[.bool] ⟨.bool, upper⟩
    upperArguments (by simpa [StoredNodeRef.concreteNode] using runs.upperOutputs)
    (by simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
      Except.instMonad, Except.bind, Except.pure, upper])
  have lowerIntArguments := runs.lowerInt.unaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.lowerIntArguments)
    ⟨.bool, lower⟩ lowerTrace
  have lowerIntTrace := runs.lowerInt.outputTrace_of_primitive #[⟨.bool, lower⟩] #[.int]
    ⟨.int, lowerInt⟩ lowerIntArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.lowerIntOutputs)
    (by simp [evalPrimitiveNode, primitive, booleanValue?, Except.instMonad, Except.bind,
      Except.pure, lowerInt])
  have upperIntArguments := runs.upperInt.unaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.upperIntArguments)
    ⟨.bool, upper⟩ upperTrace
  have upperIntTrace := runs.upperInt.outputTrace_of_primitive #[⟨.bool, upper⟩] #[.int]
    ⟨.int, upperInt⟩ upperIntArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.upperIntOutputs)
    (by simp [evalPrimitiveNode, primitive, booleanValue?, Except.instMonad, Except.bind,
      Except.pure, upperInt])
  have sumArguments := runs.sum.binaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.sumArguments)
    ⟨.int, lowerInt⟩ ⟨.int, upperInt⟩ lowerIntTrace upperIntTrace
  have sumTrace := runs.sum.outputTrace_of_primitive
    #[⟨.int, lowerInt⟩, ⟨.int, upperInt⟩] #[.int] ⟨.int, sum⟩ sumArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.sumOutputs)
    (by simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntBinary,
      Except.instMonad, Except.bind, Except.pure, sum])
  have twoArguments := runs.two.nullaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.twoArguments)
  have twoTrace := runs.two.outputTrace_of_primitive #[] #[.constantInt]
    ⟨.constantInt, two⟩ twoArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.twoOutputs) (by rfl)
  have decodedArguments := runs.decoded.binaryArguments_eq
    (by simpa [StoredNodeRef.concreteNode] using runs.decodedArguments)
    ⟨.int, sum⟩ ⟨.constantInt, two⟩ sumTrace twoTrace
  have decodedTrace := runs.decoded.outputTrace_of_primitive
    #[⟨.int, sum⟩, ⟨.constantInt, two⟩] #[.bool] ⟨.bool, decoded⟩ decodedArguments
    (by simpa [StoredNodeRef.concreteNode] using runs.decodedOutputs)
    (by simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
      Except.instMonad, Except.bind, Except.pure, decoded])
  let chain : DecoderPrimitiveChain oracle trace matrixType actual q coefficient decoded := {
    coefficientOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderCoefficient)
    quarterOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderQuarter)
    threeOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderThree)
    threeQuarterOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderThreeQuarter)
    lowerOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderLowerComparison)
    upperOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderUpperComparison)
    lowerIntOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderLowerBoolToInt)
    upperIntOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderUpperBoolToInt)
    sumOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderSum)
    twoOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderTwo)
    decodedOccurrence := occurrenceOf runs.stage #[]
      (decoderOutputWire runs.scope shape.decryptionSites.decoderEqualsTwo)
    coefficientResult := NodeResult.ofValues #[⟨.int, coefficient⟩]
    quarterResult := NodeResult.ofValues #[⟨.constantInt, quarter⟩]
    threeResult := NodeResult.ofValues #[⟨.constantInt, three⟩]
    threeQuarterResult := NodeResult.ofValues #[⟨.int, threeQuarter⟩]
    lowerResult := NodeResult.ofValues #[⟨.bool, lower⟩]
    upperResult := NodeResult.ofValues #[⟨.bool, upper⟩]
    lowerIntResult := NodeResult.ofValues #[⟨.int, lowerInt⟩]
    upperIntResult := NodeResult.ofValues #[⟨.int, upperInt⟩]
    sumResult := NodeResult.ofValues #[⟨.int, sum⟩]
    twoResult := NodeResult.ofValues #[⟨.constantInt, two⟩]
    decodedResult := NodeResult.ofValues #[⟨.bool, decoded⟩]
    coefficientExecution := by rfl
    coefficientStored := by rfl
    coefficientTrace := coefficientTrace
    quarterExecution := by simpa [q, quarter] using
      (decoderQuarter_evaluated (oracle := oracle) modulusValid)
    quarterStored := by rfl
    quarterTrace := quarterTrace
    threeExecution := by rfl
    threeStored := by rfl
    threeTrace := threeTrace
    threeQuarterExecution := by
      simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntBinary,
        Except.instMonad, Except.bind, Except.pure, threeQuarter, q, Nat.cast_mul, mul_comm]
    threeQuarterStored := by rfl
    threeQuarterTrace := threeQuarterTrace
    lowerExecution := by
      simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
        Except.instMonad, Except.bind, Except.pure, lower, quarter]
    lowerStored := by rfl
    lowerTrace := lowerTrace
    upperExecution := by
      simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
        Except.instMonad, Except.bind, Except.pure, upper, threeQuarter]
    upperStored := by rfl
    upperTrace := upperTrace
    lowerIntExecution := by
      simp [evalPrimitiveNode, primitive, booleanValue?, Except.instMonad, Except.bind, Except.pure,
        lowerInt, lower, quarter]
    lowerIntStored := by rfl
    lowerIntTrace := lowerIntTrace
    upperIntExecution := by
      simp [evalPrimitiveNode, primitive, booleanValue?, Except.instMonad, Except.bind, Except.pure,
        upperInt, upper, threeQuarter]
    upperIntStored := by rfl
    upperIntTrace := upperIntTrace
    sumExecution := by
      simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntBinary,
        Except.instMonad, Except.bind, Except.pure, sum, lowerInt, upperInt, lower, upper,
        quarter, threeQuarter]
    sumStored := by rfl
    sumTrace := sumTrace
    twoExecution := by rfl
    twoStored := by rfl
    twoTrace := twoTrace
    decodedExecution := by
      simp [evalPrimitiveNode, primitive, expectTwoInt, integerValue?, evalIntCompare,
        Except.instMonad, Except.bind, Except.pure, decoded, sum, two, lowerInt, upperInt,
        lower, upper,
        quarter, threeQuarter]
    decodedStored := by rfl
    decodedTrace := decodedTrace
  }
  have finalTrace : traceValueAt trace
      (occurrenceOf runs.stage #[] candidate.refs.decodedOutput.wire) =
        some ⟨.bool, decoded⟩ := by
    rw [runs.decodedOutputWire]
    exact decodedTrace
  exact ⟨coefficient, decoded, ⟨chain⟩, finalTrace, chain.decoded_eq⟩

end Mxx.We.DiamondWE
