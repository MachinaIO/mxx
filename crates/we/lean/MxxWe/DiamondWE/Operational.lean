import MxxWe.DiamondWE.Exact
import MxxIrCore.ScopeInvariant
import MxxRuntime.Backend

namespace Mxx.We.DiamondWE

open Mxx.IR
open Mxx.Primitives

/- Diamond uses the generic evaluator certificate without adding an application-specific dynamic
   field.  The local name keeps the application theorem signatures compact. -/
abbrev RootPrimitiveRun {backend : SemanticBackend}
    (trace : Trace backend) (stage scope node : Nat) (payload : NodePayload)
    (storedNode : Node) (port : Nat) :=
  Mxx.IR.ReachedPrimitiveRun trace {} stage scope node #[] payload storedNode port

theorem RootPrimitiveRun.argumentTraced {backend : SemanticBackend}
    {trace : Trace backend} {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node} {port index : Nat}
    (run : RootPrimitiveRun trace stage scope node payload storedNode port)
    (indexBound : index < storedNode.arguments.size) :
    ∃ argumentBound : index < run.arguments.size,
      traceValueAt trace (occurrenceOf stage #[] storedNode.arguments[index]) =
        some run.arguments[index] :=
  resolvedArgument_trace run.argumentsResolved run.valuesTraced index indexBound

/- Primitive semantics determine the dynamic value stored at the reached output port.  Generated
   application code proves `primitiveEvaluated` by reducing the concrete payload and resolved
   operands; it never supplies the reached output as an independent equation. -/
theorem RootPrimitiveRun.output_eq_of_primitive_values {backend : SemanticBackend}
    {trace : Trace backend} {stage scope node : Nat} {payload : NodePayload}
    {storedNode : Node} {port : Nat}
    (run : RootPrimitiveRun trace stage scope node payload storedNode port)
    (expected : DynamicValue backend)
    (primitiveEvaluated :
      primitive backend {} stage scope node payload run.arguments storedNode.outputs =
        .ok #[expected])
    (portZero : port = 0) : run.output = expected := by
  have actualPrimitive := (evalPrimitiveNode_success _ _ _ _ _ _ _ _
    run.primitiveEvaluated).1
  have valuesEq : run.nodeResult.values = #[expected] := by
    exact Except.ok.inj (actualPrimitive.symm.trans primitiveEvaluated)
  cases portZero
  have stored := run.outputStored
  rw [valuesEq] at stored
  simpa using (Option.some.inj stored).symm

/- The decoder certificate names every runtime primitive in the generated threshold circuit.
   Each equation is an `evalPrimitiveNode` result on the exact operands produced by the preceding
   node.  Thus the final Boolean is constrained by the executed IR pipeline, not by a caller's
   direct assertion that decoding succeeded. -/
structure DecoderPrimitiveChain
    (oracle : Mxx.Runtime.RuntimeGadgetOracle) (trace : RuntimeTrace oracle)
    (matrixType : MatrixType) (actual : RuntimeMatrixValue matrixType)
    (q : Nat) (coefficient : Int)
    (decoded : Bool) where
  coefficientOccurrence : WireOccurrence
  quarterOccurrence : WireOccurrence
  threeOccurrence : WireOccurrence
  threeQuarterOccurrence : WireOccurrence
  lowerOccurrence : WireOccurrence
  upperOccurrence : WireOccurrence
  lowerIntOccurrence : WireOccurrence
  upperIntOccurrence : WireOccurrence
  sumOccurrence : WireOccurrence
  twoOccurrence : WireOccurrence
  decodedOccurrence : WireOccurrence
  coefficientResult : NodeResult (RuntimeBackend oracle)
  quarterResult : NodeResult (RuntimeBackend oracle)
  threeResult : NodeResult (RuntimeBackend oracle)
  threeQuarterResult : NodeResult (RuntimeBackend oracle)
  lowerResult : NodeResult (RuntimeBackend oracle)
  upperResult : NodeResult (RuntimeBackend oracle)
  lowerIntResult : NodeResult (RuntimeBackend oracle)
  upperIntResult : NodeResult (RuntimeBackend oracle)
  sumResult : NodeResult (RuntimeBackend oracle)
  twoResult : NodeResult (RuntimeBackend oracle)
  decodedResult : NodeResult (RuntimeBackend oracle)
  coefficientExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.extractCoefficient (.literal 0) none) #[⟨.matrix matrixType, actual⟩] #[.int] =
      .ok coefficientResult
  coefficientStored : coefficientResult.values[0]? = some ⟨.int, coefficient⟩
  coefficientTrace : traceValueAt trace coefficientOccurrence = some ⟨.int, coefficient⟩
  quarterExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (quarterPayload q) #[] #[.constantInt] = .ok quarterResult
  quarterStored : quarterResult.values[0]? = some ⟨.constantInt, (decoderQuarter q : Int)⟩
  quarterTrace : traceValueAt trace quarterOccurrence =
    some ⟨.constantInt, (decoderQuarter q : Int)⟩
  threeExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.constantInt 3) #[] #[.constantInt] = .ok threeResult
  threeStored : threeResult.values[0]? = some ⟨.constantInt, (3 : Int)⟩
  threeTrace : traceValueAt trace threeOccurrence = some ⟨.constantInt, (3 : Int)⟩
  threeQuarterExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.intBinary .multiply)
    #[⟨.constantInt, (decoderQuarter q : Int)⟩, ⟨.constantInt, (3 : Int)⟩] #[.int] =
      .ok threeQuarterResult
  threeQuarterStored : threeQuarterResult.values[0]? =
    some ⟨.int, ((3 * decoderQuarter q : Nat) : Int)⟩
  threeQuarterTrace : traceValueAt trace threeQuarterOccurrence =
    some ⟨.int, ((3 * decoderQuarter q : Nat) : Int)⟩
  lowerExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.intCompare .lessEqual)
    #[⟨.constantInt, (decoderQuarter q : Int)⟩, ⟨.int, coefficient⟩] #[.bool] =
      .ok lowerResult
  lowerStored : lowerResult.values[0]? =
    some ⟨.bool, decide ((decoderQuarter q : Int) ≤ coefficient)⟩
  lowerTrace : traceValueAt trace lowerOccurrence =
    some ⟨.bool, decide ((decoderQuarter q : Int) ≤ coefficient)⟩
  upperExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.intCompare .lessEqual)
    #[⟨.int, coefficient⟩, ⟨.int, ((3 * decoderQuarter q : Nat) : Int)⟩] #[.bool] =
      .ok upperResult
  upperStored : upperResult.values[0]? =
    some ⟨.bool, decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int))⟩
  upperTrace : traceValueAt trace upperOccurrence =
    some ⟨.bool, decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int))⟩
  lowerIntExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0 .boolToInt
    #[⟨.bool, decide ((decoderQuarter q : Int) ≤ coefficient)⟩] #[.int] = .ok lowerIntResult
  lowerIntStored : lowerIntResult.values[0]? = some ⟨.int,
    if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else (0 : Int)⟩
  lowerIntTrace : traceValueAt trace lowerIntOccurrence = some ⟨.int,
    if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else (0 : Int)⟩
  upperIntExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0 .boolToInt
    #[⟨.bool, decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int))⟩] #[.int] =
      .ok upperIntResult
  upperIntStored : upperIntResult.values[0]? = some ⟨.int,
    if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then (1 : Int) else (0 : Int)⟩
  upperIntTrace : traceValueAt trace upperIntOccurrence = some ⟨.int,
    if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then (1 : Int) else (0 : Int)⟩
  sumExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0 (.intBinary .add)
    #[⟨.int, if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else (0 : Int)⟩,
      ⟨.int, if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then
        (1 : Int) else (0 : Int)⟩] #[.int] =
      .ok sumResult
  sumStored : sumResult.values[0]? = some ⟨.int,
    (if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else 0) +
      (if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then (1 : Int) else 0)⟩
  sumTrace : traceValueAt trace sumOccurrence = some ⟨.int,
    (if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else 0) +
      (if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then (1 : Int) else 0)⟩
  twoExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.constantInt 2) #[] #[.constantInt] = .ok twoResult
  twoStored : twoResult.values[0]? = some ⟨.constantInt, (2 : Int)⟩
  twoTrace : traceValueAt trace twoOccurrence = some ⟨.constantInt, (2 : Int)⟩
  decodedExecution : evalPrimitiveNode (RuntimeBackend oracle) {} 0 0 0
    (.intCompare .equal)
    #[⟨.int, (if decide ((decoderQuarter q : Int) ≤ coefficient) then (1 : Int) else (0 : Int)) +
        (if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then
          (1 : Int) else (0 : Int))⟩,
      ⟨.constantInt, (2 : Int)⟩] #[.bool] = .ok decodedResult
  decodedStored : decodedResult.values[0]? = some ⟨.bool, decoded⟩
  decodedTrace : traceValueAt trace decodedOccurrence = some ⟨.bool, decoded⟩

theorem DecoderPrimitiveChain.decoded_eq
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {trace : RuntimeTrace oracle}
    {matrixType : MatrixType} {actual : RuntimeMatrixValue matrixType}
    {q : Nat} {coefficient : Int} {decoded : Bool}
    (chain : DecoderPrimitiveChain oracle trace matrixType actual q coefficient decoded) :
    decoded = decodeInterval q coefficient := by
  have evaluated := evalPrimitiveNode_success _ _ _ _ _ _ _ _ chain.decodedExecution
  have outputEq : chain.decodedResult.values =
      #[⟨.bool,
        decide (((if decide ((decoderQuarter q : Int) ≤ coefficient) then 1 else 0) +
          (if decide (coefficient ≤ ((3 * decoderQuarter q : Nat) : Int)) then
            1 else 0) : Int) = 2)⟩] := by
    have primitiveEq := evaluated.1
    simp [primitive, evalIntCompare, expectTwoInt, integerValue?] at primitiveEq
    simpa [Nat.cast_mul] using (Except.ok.inj primitiveEq).symm
  have decodedEq := chain.decodedStored
  rw [outputEq] at decodedEq
  simp at decodedEq
  rw [← decodedEq]
  unfold decodeInterval
  by_cases lower : (decoderQuarter q : Int) ≤ coefficient <;>
    by_cases upper : coefficient ≤ 3 * (decoderQuarter q : Int) <;>
      simp [lower, upper, Nat.cast_mul]

/- Interpreting an executed `applyPreimage` node gives the same matrix product consumed by the
   injector invariant.  The subsequent approximation therefore contains exactly
   `L * E + e * K`, with the two tight `inner * n` product bounds supplied by the generic gadget
   lemma. -/
noncomputable def inputInjectorApplyPreimageWithin
    {q n sourceRows inner targetColumns resultRows : Nat}
    (source : ExactMatrix q n sourceRows inner)
    (preimage : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns)
    (left : ExactMatrix q n resultRows sourceRows)
    (value : ExactMatrix q n resultRows inner)
    (runtimeOutput : ExactMatrix q n resultRows targetColumns)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (runtimeEquation : runtimeOutput = value * preimage)
    (relation : RightPreimage source preimage target)
    (leftMagnitude : MagnitudeFact left)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift preimage preimageBound)
    (valueApprox : ApproxWithin value (left * source) xNoiseBound)
    (targetApprox : ApproxWithin target idealTarget targetNoiseBound)
    (hn : 0 < n) :
    ApproxWithin runtimeOutput (left * idealTarget)
      (sourceRows * n * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) := by
  -- The evaluator output is first identified with `value * preimage`; the generic injector
  -- theorem then expands it as `left * idealTarget + (left * E + e * preimage)`.
  rw [runtimeEquation]
  exact input_injector_step_with_bound hn source preimage target left value idealTarget relation
    leftMagnitude preimageLift valueApprox targetApprox

end Mxx.We.DiamondWE
