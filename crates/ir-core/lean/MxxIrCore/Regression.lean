import MxxIrCore.Eval

namespace Mxx
namespace IR

def regressionLeft : MatrixType :=
  { modulus := 17, ringDimension := 4, rows := 2, columns := 3 }

def regressionRight : MatrixType :=
  { modulus := 17, ringDimension := 4, rows := 3, columns := 5 }

def regressionProduct : MatrixType :=
  { modulus := 17, ringDimension := 4, rows := 2, columns := 5 }

def regressionTranspose : MatrixType :=
  { modulus := 17, ringDimension := 4, rows := 3, columns := 2 }

def unitBackend : SemanticBackend where
  denoteMatrix := fun _ ↦ Unit
  denoteTrapdoor := fun _ ↦ Unit
  denotePreimage := fun _ ↦ Unit
  denoteTypedBlob := fun _ _ ↦ Unit
  matrixZero := fun _ ↦ ()
  matrixIdentity := fun _ ↦ ()
  matrixAdd := fun _ _ _ ↦ ()
  matrixSubtract := fun _ _ _ ↦ ()
  matrixScale := fun _ _ _ ↦ ()
  matrixMultiply := fun _ _ _ _ _ ↦ ()
  matrixNegate := fun _ _ ↦ ()
  matrixTranspose := fun _ _ _ ↦ ()
  matrixConstant := fun _ _ _ ↦ ()
  matrixSlice := fun _ _ _ _ _ _ _ ↦ ()
  matrixConcat := fun _ _ _ ↦ ()
  gadgetCertificate := fun _ layout _ _ _ _ ↦ layout.mode = layout.mode
  gadgetDecompose := fun _ layout _ _ _ ↦ .ok ⟨(), ⟨rfl⟩⟩
  extractCoefficient := fun _ _ _ ↦ 0
  bitExtract := fun _ _ ↦ false
  trapdoorPublic := fun _ _ ↦ ()
  materializePreimage := fun _ _ ↦ ()
  applyPreimage := fun _ _ _ _ _ ↦ ()

example : Family.shapeProduct [2, 3] = 6 := by decide

example : Family.rowMajorOffset [2, 3] (⟨1, ⟨2, ()⟩⟩ : FamilyIndex [2, 3]) = 5 := by decide

example : Family.rowMajorOffset [2, 3] (⟨1, ⟨2, ()⟩⟩ : FamilyIndex [2, 3]) <
    Family.shapeProduct [2, 3] := by
  exact Family.rowMajorOffset_lt _ _

example : OccurrenceFrame.mk 0 0 3 7 ≠ OccurrenceFrame.mk 0 0 3 8 := by decide

example : ¬ operationArityOK (.intBinary .add) 1 1 := by simp [operationArityOK]

example : operationArityOK (.intBinary .add) 2 1 := by simp [operationArityOK]

example : familyIndexFromArray #[2, 3] #[1, 2] =
    some (⟨1, ⟨2, ()⟩⟩ : FamilyIndex [2, 3]) := by rfl

example : familyIndexFromArray #[2, 3] #[2, 0] = none := by rfl

example : StructuralIntExpr.eval {} (.exactDivide (.literal 12) (.literal 3)) = .ok 4 := by rfl

example : StructuralIntExpr.eval {} (.exactDivide (.literal 5) (.literal 2)) = .error "non-exact division" := by
  rfl

example : IndexMapExpr.eval { axes := #[4, 9] } (.add (.axis 0) (.literal 3)) = .ok 7 := by
  rfl

example : IndexMapExpr.eval {} (.divide (.literal 7) (.literal 0)) = .error "index division by zero" := by
  rfl

example : IndexMapExpr.eval {} (.select (.literal 1) #[.literal 10, .literal 20]) = .ok 20 := by
  rfl

example : matrixProductType
    { modulus := 17, ringDimension := 4, rows := 2, columns := 3 }
    { modulus := 17, ringDimension := 4, rows := 3, columns := 5 }
    { modulus := 17, ringDimension := 4, rows := 2, columns := 5 } := by
  simp [matrixProductType, sameRing]

example : matrixProductTypeB regressionLeft regressionRight regressionProduct = true := by decide

def regressionTrapdoor : TrapdoorType :=
  { matrix := regressionLeft, sigma := .literal { numerator := 1, denominator := 1 },
    gadgetBase := .literal 2, digitCount := .literal 1,
    preimageMaxCoefficientBound := .literal 1 }

example : familyPreimageSampleOperationTypesB regressionRight
    [some (.matrix regressionLeft), some (.trapdoor regressionTrapdoor),
      some (.family [7] (.matrix regressionProduct))]
    [.family [7] (.preimage regressionRight)] = true := by decide

example : indexMapCheckedB
    { sourceRank := 1, outputRank := 1, inputIndices := #[.literal 8] } [2] #[] = false := by
  simp [indexMapCheckedB, indexExprFuel, indexExprCheckedFuelB]

example : indexMapCheckedB
    { sourceRank := 1, outputRank := 1,
      inputIndices := #[.divide (.axis 0) (.literal 0)] } [2] #[] = false := by
  simp [indexMapCheckedB, indexExprFuel, indexExprCheckedFuelB]

example : indexMapCheckedB
    { sourceRank := 1, outputRank := 1, inputIndices := #[.log2Ceil (.literal 0)] }
    [2] #[] = false := by simp [indexMapCheckedB, indexExprFuel, indexExprCheckedFuelB]

example : ¬gridInputTypeOK 1 .constantInt .int { reindex := false, map := none } := by
  simp [gridInputTypeOK]

example : indexExprCheckedFuelB 1 #[] 2 (.axis 1) = .invalid := by rfl

example : shapeExpression? #[.add (.literal 1) (.literal 1),
    .multiply (.literal 1) (.literal 1)] = some [2, 1] := by rfl

example : shapeExpressionIs #[.add (.literal 1) (.literal 1),
    .multiply (.literal 1) (.literal 1)] [2, 1] := by rfl

example : structuralExpressionIsNat (.add (.literal 1) (.literal 1)) 2 := by rfl

example : primitive unitBackend {} 0 0 0 (.constantInt 7) #[] #[.constantInt] =
    .ok #[⟨.constantInt, (show Value unitBackend .constantInt from (7 : Int))⟩] := by rfl

example : evalPrimitiveNode unitBackend {} 0 0 0 (.constantInt 7) #[] #[.constantInt] =
    .ok (NodeResult.ofValues
      #[⟨.constantInt, (show Value unitBackend .constantInt from (7 : Int))⟩]) := by
  rfl

example (result : NodeResult unitBackend)
    (success : evalPrimitiveNode unitBackend {} 0 0 0 (.constantInt 7) #[] #[.constantInt] =
      .ok result) :
    primitive unitBackend {} 0 0 0 (.constantInt 7) #[] #[.constantInt] = .ok result.values ∧
      result.scopes = #[] :=
  evalPrimitiveNode_success {} 0 0 0 (.constantInt 7) #[] #[.constantInt] result success

example : PrimitiveNodePayload (.constantInt 7) := .constantInt 7

#check evalScope_success_subgraph_step

#check evalScope_success_parallelGrid_step

#check evalScope_success_sequentialLoop_step

#check evalSequentialLoop_success_iteration_step

example :
    ∃ outputBound : 1 < (#[2, 3] : Array Nat).size,
      (Except.ok (2 + 1) : Except Unit Nat) = .ok (#[2, 3] : Array Nat)[1] := by
  exact array_mapM_getElem (fun value : Nat => Except.ok (value + 1))
    (xs := #[1, 2]) (ys := #[2, 3])
    (by simp [Array.mapM_eq_foldlM_push]; rfl) (by decide)

example (values : Array (Binding unitBackend)) :
    appendNodeBindings 3 4 values #[] = values := by
  rfl

example {data : ProgramData} (env : EvalEnv unitBackend data) (structural : StructuralEnv)
    (trace : Trace unitBackend) (stageNumber : Nat) (stage : Stage) (scopeNumber : ScopeId)
    (scope : Scope) (stageStored : data.stages[stageNumber]? = some stage)
    (scopeStored : scopeAt stage scopeNumber = some scope) (inputs : Array (Binding unitBackend))
    (path : OccurrencePath) (index : Nat) (values : Array (Binding unitBackend)) (fuel : Nat)
    (finalResult : ScopeResult unitBackend) (fuelPositive : fuel ≠ 0)
    (indexBound : index < scope.nodes.size) (nodeValue : Node)
    (nodeStored : scope.nodes[index]? = some nodeValue)
    (payloadStored : nodeValue.payload = .constantInt 7)
    (success : evalScope data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel = .ok finalResult) :
    ∃ argumentValues result nextResult,
      evalPrimitiveNode unitBackend structural stageNumber scope.id index (.constantInt 7)
          argumentValues nodeValue.outputs = .ok result ∧
      evalScope data env structural trace stageNumber stage scopeNumber scope stageStored scopeStored
          inputs path (index + 1) (appendNodeBindings scope.id index values result.values) (fuel - 1) =
        .ok nextResult := by
  obtain ⟨argumentValues, result, nextResult, _, resultStored, _, nextStored, _⟩ :=
    evalScope_success_primitive_step data env structural trace stageNumber stage scopeNumber scope
      stageStored scopeStored inputs path index values fuel finalResult fuelPositive indexBound
      nodeValue nodeStored (.constantInt 7) payloadStored (.constantInt 7) success
  exact ⟨argumentValues, result, nextResult, resultStored, nextStored⟩

example : primitive unitBackend {} 0 0 0 .transpose
    #[⟨.matrix regressionLeft, ()⟩] #[.matrix regressionTranspose] =
    .ok #[⟨.matrix regressionTranspose, ()⟩] := by rfl

/- Every non-sampler primitive reached by the concrete Diamond projection must enter the
   evaluator's backend boundary instead of falling through to `unsupportedPrimitive`. -/
example : primitive unitBackend {} 0 0 0
    (.constantMatrix regressionLeft .zero) #[] #[.matrix regressionLeft] =
    .ok #[⟨.matrix regressionLeft, ()⟩] := by
  simp [primitive, unitBackend, evalNatExpr, Except.instMonad, Except.bind, Except.mapError,
    Except.map, Except.pure]

example : primitive unitBackend {} 0 0 0
    (.constantMatrix regressionLeft (.gadget (.literal 2) true)) #[] #[.matrix regressionLeft] =
    .error (.unsupportedPrimitive 0 0 0) := by
  rfl

example : primitive unitBackend {} 0 0 0 (.bitExtract (.literal 1))
    #[⟨.int, (3 : Int)⟩] #[.bool] = .ok #[⟨.bool, false⟩] := by
  simp [primitive, unitBackend, evalNatExpr, StructuralIntExpr.eval, integerValue?,
    Except.instMonad, Except.bind, Except.mapError, Except.map, Except.pure]

example : primitive unitBackend {} 0 0 0
    (.slice none (some { start := .literal 0, stop := .literal 1 }))
    #[⟨.matrix regressionLeft, ()⟩] #[.matrix regressionTranspose] =
    .ok #[⟨.matrix regressionTranspose, ()⟩] := by
  simp [primitive, unitBackend, evalNatExpr, evalRange, StructuralIntExpr.eval,
    Except.instMonad, Except.bind, Except.mapError, Except.map, Except.pure]

example : primitive unitBackend {} 0 0 0 (.concat .rows)
    #[⟨.matrix regressionLeft, ()⟩, ⟨.matrix regressionLeft, ()⟩]
    #[.matrix { modulus := 17, ringDimension := 4, rows := 4, columns := 3 }] =
    .ok #[⟨.matrix { modulus := 17, ringDimension := 4, rows := 4, columns := 3 }, ()⟩] := by
  simp [primitive, unitBackend, Except.instMonad, Except.bind, Except.mapError, Except.map,
    Except.pure]

example : primitive unitBackend {} 0 0 0 (.gadgetDecompose (.literal 32) false (.literal 1))
    #[⟨.matrix regressionLeft, ()⟩]
    #[.preimage (gadgetPreimageType regressionLeft
      { mode := .regular, base := 32, digits := 1, sourceRows := 2, targetRows := 2,
        sourceColumns := 3, targetColumns := 3 })] =
    .ok #[⟨.preimage (gadgetPreimageType regressionLeft
      { mode := .regular, base := 32, digits := 1, sourceRows := 2, targetRows := 2,
        sourceColumns := 3, targetColumns := 3 }), ()⟩] := by
  rfl

example : primitive unitBackend {} 0 0 0 (.gadgetDecompose (.literal 2) true (.literal 1))
    #[⟨.matrix regressionLeft, ()⟩]
    #[.preimage (gadgetPreimageType regressionLeft
      { mode := .regular, base := 2, digits := 1, sourceRows := 2, targetRows := 2,
        sourceColumns := 3, targetColumns := 3 })] =
    .error (.unsupportedPrimitive 0 0 0) := by
  rfl

example : primitive unitBackend {} 0 0 0 (.extractCoefficient (.literal 0) none)
    #[⟨.matrix regressionLeft, ()⟩] #[.int] = .ok #[⟨.int, (0 : Int)⟩] := by
  simp [primitive, unitBackend, evalNatExpr, StructuralIntExpr.eval, Except.instMonad,
    Except.bind, Except.mapError, Except.map, Except.pure]

example : primitive unitBackend {} 0 0 0 (.matrixBinary .multiply)
    #[⟨.matrix regressionLeft, ()⟩, ⟨.matrix regressionRight, ()⟩]
    #[.matrix regressionProduct] = .ok #[⟨.matrix regressionProduct, ()⟩] := by rfl

example : primitive unitBackend {} 0 0 0 .applyPreimage
    #[⟨.matrix regressionLeft, ()⟩, ⟨.preimage regressionRight, ()⟩]
    #[.matrix regressionProduct] = .ok #[⟨.matrix regressionProduct, ()⟩] := by rfl

example {data : ProgramData} (env : EvalEnv unitBackend data) (index : Nat)
    (trace finalTrace : Trace unitBackend) (bound : index < data.stages.size)
    (success : evalStages data env index trace = .ok finalTrace) :
    ∃ stage, ∃ stageStored : data.stages[index]? = some stage, ∃ stageTrace,
      evalStage data env trace index stage stageStored = .ok stageTrace ∧
        evalStages data env (index + 1) { stages := trace.stages.push stageTrace } =
          .ok finalTrace :=
  evalStages_success_step data env index trace finalTrace bound success

example {data : ProgramData} (env : EvalEnv unitBackend data) (trace : Trace unitBackend)
    (stageNumber : Nat) (stage : Stage)
    (stageStored : data.stages[stageNumber]? = some stage) (stageTrace : StageTrace unitBackend)
    (success : evalStage data env trace stageNumber stage stageStored = .ok stageTrace) :
    ∃ root, ∃ rootStored : scopeAt stage stage.root = some root, ∃ result,
      evalScope data env {} trace stageNumber stage stage.root root stageStored rootStored
        #[] #[] 0 #[] (evaluationFuel data) = .ok result ∧
      stageTrace = { stage := stageNumber, scopes := result.scopes } :=
  evalStage_success_root data env trace stageNumber stage stageStored stageTrace success

example : ¬matrixProductType
    { modulus := 17, ringDimension := 4, rows := 2, columns := 3 }
    { modulus := 17, ringDimension := 4, rows := 4, columns := 5 }
    { modulus := 17, ringDimension := 4, rows := 2, columns := 5 } := by
  simp [matrixProductType, sameRing]

example : removeAt? [2, 3, 5] 1 = some [2, 5] := by rfl

example : removeAt? [2] 0 = some [] := by rfl

example {data : ProgramData} (sample : SampleRef data) :
    occurrenceValid data sample.occurrence := sample.occurrenceValid

example {data : ProgramData} (sample : SampleRef data) :
    ∃ stage scope node,
      data.stages[sample.occurrence.stage]? = some stage ∧
      scopeAt stage sample.occurrence.wire.scope = some scope ∧
      nodeAt scope sample.occurrence.wire.node = some node ∧
      node.payload = sample.payload := sample.storedPayload

example {data : ProgramData} (sample : SampleRef data) :
    ∃ stage scope node,
      data.stages[sample.occurrence.stage]? = some stage ∧
      scopeAt stage sample.occurrence.wire.scope = some scope ∧
      nodeAt scope sample.occurrence.wire.node = some node ∧
      node.outputs[sample.occurrence.wire.port]? = some sample.outputType := sample.storedOutput

/- A generated proof can discharge each node separately, then aggregate those facts
   without unfolding or recomputing any node validator. -/
example {stage : Stage} {scope : Scope}
    (header : scopeHeaderValidB scope = true)
    (eachNode : ∀ index, index < scope.nodes.size →
      nodeAtIndexValidB stage scope index = true) : scopeValidB stage scope = true :=
  scopeValidB_of_components header (scopeNodesValidB_of_each eachNode)

def certificateExampleNode : Node :=
  { payload := .constantInt 7, arguments := #[], outputs := #[.constantInt] }

def certificateExampleScope : Scope :=
  { id := 0, structuralSlots := #[], nodes := #[certificateExampleNode],
    inputs := #[], outputs := #[] }

def certificateExampleStage : Stage :=
  { name := "certificate-example", bindings := #[], scopes := #[certificateExampleScope],
    root := 0, namedOutputs := #[] }

def certificateExampleDirectOperation :
    DirectOperationCert certificateExampleScope (.constantInt 7) [] #[.constantInt] :=
  .constantInt 7 ⟨rfl, rfl⟩

example : operationTypesOK certificateExampleStage certificateExampleScope (.constantInt 7)
    #[] #[.constantInt] := certificateExampleDirectOperation.sound rfl

def certificateExampleNodeLeaf :
    StoredNodeCert certificateExampleStage certificateExampleScope 0 where
  node := certificateExampleNode
  stored := by rfl
  valid := {
    outputsNonempty := by decide
    outputTypes := by
      intro output member
      simp [certificateExampleNode] at member
      subst output
      trivial
    argumentsPrevious := by simp [certificateExampleNode]
    payload := by simp [certificateExampleNode, NodePayload.Valid, validPayload]
    slotsUsed := by simp [certificateExampleScope, certificateExampleNode, payloadSlotsUsed]
    operation := by simp [certificateExampleNode, operationTypesOK, referencedTypes]
  }

def certificateExampleNodeRange :
    NodeRangeCert certificateExampleStage certificateExampleScope 0 1 :=
  .single 0 certificateExampleNodeLeaf

def certificateExampleRank : ScopeId → Nat := fun _ => 0

def certificateExampleScopeLeaf :
    StoredScopeCert certificateExampleStage certificateExampleRank 0 where
  scope := certificateExampleScope
  stored := by rfl
  valid := {
    slots := by
      constructor
      · intro first second left right leftStored rightStored different
        simp [certificateExampleScope] at leftStored
      · simp [certificateExampleScope]
    inputs := by simp [certificateExampleScope]
    outputs := by simp [certificateExampleScope]
    childrenDecrease := by simp [certificateExampleScope, certificateExampleNode,
      structuralChildren, NodePayload.childScope?]
  }
  nodes := certificateExampleNodeRange

/- The diagnostic Boolean can be checked locally, independently of the trusted certificate. -/
example : nodeValidB certificateExampleStage certificateExampleScope 0 certificateExampleNode = true := by
  norm_num [nodeValidB, certificateExampleStage, certificateExampleScope,
    certificateExampleNode, validWireTypeB, payloadSlotsUsedB, operationTypesOKB,
    noInputSingleOutputB, referencedTypes, wireType?, nodeAt]

/- A stored leaf is bound to the exact array payload, so a mutation must prove equality first. -/
example (replacement : Node)
    (stored : certificateExampleScope.nodes[0]? = some replacement) :
    certificateExampleNode = replacement := by
  rw [certificateExampleNodeLeaf.stored] at stored
  exact Option.some.inj stored

/- Balanced aggregation preserves random access without unfolding sibling leaves. -/
noncomputable example : StoredNodeCert certificateExampleStage certificateExampleScope 0 :=
  certificateExampleNodeRange.covers 0 (by omega) (by omega)

end IR
end Mxx
