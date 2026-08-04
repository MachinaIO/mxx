import MxxWe.Certificate.Syntax
import Mxx.Ir.ExecutionFacts

namespace MxxWe.Certificate

deriving instance DecidableEq for Mxx.Ir.MatrixTypeExpr
deriving instance DecidableEq for Mxx.Ir.InputSource
deriving instance DecidableEq for Mxx.Ir.IntBinaryOp
deriving instance DecidableEq for Mxx.Ir.IntCompareOp
deriving instance DecidableEq for Mxx.Ir.LoopInputMode
deriving instance DecidableEq for Mxx.Ir.ConcatAxis
deriving instance DecidableEq for Mxx.Ir.NodeKind
deriving instance DecidableEq for Mxx.Ir.Node

def resolveStage (workflow : Mxx.Ir.Workflow) (name : String) : Option Mxx.Ir.Stage :=
  workflow.stages.find? fun stage => stage.id = name

def ScopeRef.definitionName : ScopeRef → String
  | .root => "__root"
  | .subgraph canonicalName => s!"subgraph:{canonicalName}"
  | .parallelBody parent owner => s!"parallel:{parent.definitionName}:{owner}"
  | .sequentialBody parent owner => s!"sequential:{parent.definitionName}:{owner}"

def rawScope (workflow : Mxx.Ir.Workflow) (stageName : String) : ScopeRef → Option Mxx.Ir.Scope
  | .root => (resolveStage workflow stageName).map (·.program.root)
  | reference => do
      let stage ← resolveStage workflow stageName
      Mxx.Ir.lookupDefinition reference.definitionName stage.program.definitions

def rawNode (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef) : Option Mxx.Ir.Node := do
  let scope ← rawScope workflow reference.stage reference.scope
  scope.nodes[reference.node]?

def scopeOwnerMatches (workflow : Mxx.Ir.Workflow) (stage : String) : ScopeRef → Bool
  | .root => true
  | .subgraph canonicalName => (rawScope workflow stage (.subgraph canonicalName)).isSome
  | reference@(.parallelBody parent owner) =>
      scopeOwnerMatches workflow stage parent &&
        match rawNode workflow { stage, scope := parent, node := owner } with
        | some { kind := .parallelLoop definition _ _ _ _, .. } =>
            decide (definition = reference.definitionName)
        | _ => false
  | reference@(.sequentialBody parent owner) =>
      scopeOwnerMatches workflow stage parent &&
        match rawNode workflow { stage, scope := parent, node := owner } with
        | some { kind := .sequentialLoop definition _ _ _ _, .. } =>
            decide (definition = reference.definitionName)
        | _ => false

def resolveScope (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef) : Option Mxx.Ir.Scope := do
  guard (scopeOwnerMatches workflow reference.stage reference.scope)
  rawScope workflow reference.stage reference.scope

def resolveNode (workflow : Mxx.Ir.Workflow) (reference : CoreNodeRef) : Option Mxx.Ir.Node := do
  let scope ← resolveScope workflow reference
  scope.nodes[reference.node]?

def wireRef (reference : CoreWireRef) : Mxx.Ir.WireRef :=
  { node := reference.node.node, port := reference.port }

def scopeInputWires (scope : Mxx.Ir.Scope) : List Mxx.Ir.WireRef :=
  scope.inputNames.filterMap fun name =>
    (scope.nodes.zipIdx.find? fun entry => match entry.1.kind with
      | .input actual => actual = name
      | _ => false).map fun entry => { node := entry.2, port := 0 }

def scopeOutputWires (scope : Mxx.Ir.Scope) : List Mxx.Ir.WireRef :=
  scope.outputs.map Prod.snd

def verifyWire (workflow : Mxx.Ir.Workflow) (reference : CoreWireRef) : Bool :=
  match resolveNode workflow reference.node with
  | none => false
  | some node => decide (reference.port < node.outputCount)

def verifyInputWire (workflow : Mxx.Ir.Workflow) (reference : CoreWireRef)
    (stage inputName : String) : Bool :=
  decide (reference.node.stage = stage) && decide (reference.node.scope = .root) &&
    decide (reference.port = 0) &&
    match resolveNode workflow reference.node with
    | some { kind := .input actual, arguments, outputCount } =>
        decide (actual = inputName) && decide arguments.isEmpty && decide (outputCount = 1)
    | _ => false

def verifyOperand (workflow : Mxx.Ir.Workflow) (reference : CoreOperandRef) : Bool :=
  decide (reference.node.stage = reference.wire.node.stage) &&
    decide (reference.node.scope = reference.wire.node.scope) &&
    verifyWire workflow reference.wire &&
    match resolveNode workflow reference.node with
    | none => false
    | some node => decide (node.arguments[reference.operand]? = some (wireRef reference.wire))

def verifyOperationOutput (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (output : CoreWireRef) : Bool :=
  decide (output.node = operation) && verifyWire workflow output

def verifyUnaryNode (workflow : Mxx.Ir.Workflow) (reference : UnaryNodeRef) : Bool :=
  decide (reference.input.node = reference.operation) && decide (reference.input.operand = 0) &&
    verifyOperand workflow reference.input &&
    verifyOperationOutput workflow reference.operation reference.output &&
    match resolveNode workflow reference.operation with
    | some node => decide (node.arguments = [wireRef reference.input.wire])
    | none => false

def verifyBinaryNode (workflow : Mxx.Ir.Workflow) (reference : BinaryNodeRef) : Bool :=
  decide (reference.left.node = reference.operation) && decide (reference.left.operand = 0) &&
    decide (reference.right.node = reference.operation) && decide (reference.right.operand = 1) &&
    verifyOperand workflow reference.left && verifyOperand workflow reference.right &&
    verifyOperationOutput workflow reference.operation reference.output &&
    match resolveNode workflow reference.operation with
    | some node =>
        decide (node.arguments = [wireRef reference.left.wire, wireRef reference.right.wire])
    | none => false

def verifyMatrixBinary (workflow : Mxx.Ir.Workflow) (reference : MatrixBinaryRef)
    (expected : Mxx.Ir.NodeKind) : Bool :=
  verifyBinaryNode workflow {
    operation := reference.operation
    left := reference.left
    right := reference.right
    output := reference.output
  } &&
    match resolveNode workflow reference.operation with
    | some node => decide (node.kind = expected) && decide (node.outputCount = 1)
    | none => false

def verifyConstantIntWire (workflow : Mxx.Ir.Workflow) (reference : CoreWireRef)
    (value : Int) : Bool :=
  decide (reference.port = 0) &&
    match resolveNode workflow reference.node with
    | some { kind := .constantInt actual, arguments, outputCount } =>
        decide (actual = value) && decide arguments.isEmpty && decide (outputCount = 1)
    | _ => false

def verifyConstantTwoMatrix (workflow : Mxx.Ir.Workflow) (reference : CoreWireRef) : Bool :=
  decide (reference.port = 0) &&
  match resolveNode workflow reference.node with
  | some { kind := .constantMatrix actualType coefficients, arguments, outputCount } =>
      decide (actualType = {
        modulus := .parameter "diamond_modulus"
        ringDimension := .parameter "diamond_ring_dimension"
        rows := .constant 1
        columns := .constant 1
      }) && decide (coefficients = [.constant 2]) &&
        decide arguments.isEmpty &&
          decide (outputCount = 1)
    | _ => false

def verifyDynamicGet (workflow : Mxx.Ir.Workflow) (reference : DynamicFamilyGetRef)
    (family : CoreWireRef) : Bool :=
  decide (reference.family.node = reference.operation) && decide (reference.family.operand = 0) &&
    decide (reference.family.wire = family) &&
    decide (reference.index.node = reference.operation) &&
    decide (reference.index.operand = 1) && verifyOperand workflow reference.family &&
    verifyOperand workflow reference.index &&
    verifyOperationOutput workflow reference.operation reference.output &&
    decide (reference.output.port = 0) &&
    match resolveNode workflow reference.operation with
    | some { kind := .familyGetDynamic, arguments, outputCount } =>
        decide (arguments = [wireRef reference.family.wire, wireRef reference.index.wire]) &&
          decide (outputCount = 1)
    | _ => false

def verifySelect (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (selector : CoreOperandRef) (branches : List CoreOperandRef)
    (output : CoreWireRef) : Bool :=
  decide (selector.node = operation) && decide (selector.operand = 0) &&
    verifyOperand workflow selector &&
    branches.zipIdx.all (fun branch =>
      decide (branch.1.node = operation) && decide (branch.1.operand = branch.2 + 1) &&
        verifyOperand workflow branch.1) &&
    verifyOperationOutput workflow operation output &&
    match resolveNode workflow operation with
    | some { kind := .select, arguments, outputCount } =>
        decide (arguments.length = branches.length + 1) && decide (outputCount = 1)
    | _ => false

def verifySixWaySelect (workflow : Mxx.Ir.Workflow) (reference : SixWaySelectRef) : Bool :=
  verifySelect workflow reference.operation reference.selector (List.ofFn reference.branches)
    reference.output

def verifyTwoWaySelect (workflow : Mxx.Ir.Workflow) (reference : TwoWaySelectRef) : Bool :=
  verifySelect workflow reference.operation reference.selector (List.ofFn reference.branches)
    reference.output

def CertifiedLoopInputMode.toIr : CertifiedLoopInputMode → Mxx.Ir.LoopInputMode
  | .broadcast => .broadcast
  | .zip => .zip
  | .zipOffset offset => .zipOffset offset

def verifySequentialLoop (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (bodyScope : ScopeRef) (carried invariants : List CoreOperandRef)
    (outputs : List CoreWireRef) : Bool :=
  let operands := carried ++ invariants
  operands.zipIdx.all (fun operand =>
      decide (operand.1.node = operation) && decide (operand.1.operand = operand.2) &&
        verifyOperand workflow operand.1) &&
    outputs.zipIdx.all (fun output =>
      decide (output.1.node = operation) && decide (output.1.port = output.2) &&
        verifyWire workflow output.1) &&
    decide (bodyScope = .sequentialBody operation.scope operation.node) &&
    match resolveNode workflow operation,
        resolveScope workflow { operation with scope := bodyScope } with
    | some node, some body => match node.kind with
      | .sequentialLoop definition _ _ _ carriedCount =>
          decide (definition = bodyScope.definitionName) &&
            decide (carriedCount = carried.length) &&
            decide (node.arguments = operands.map (wireRef ∘ CoreOperandRef.wire)) &&
            decide body.inputNames.Nodup &&
            decide ((scopeInputWires body).length = body.inputNames.length) &&
            decide (operands.length = body.inputNames.length) &&
            decide (node.outputCount = outputs.length) && decide (outputs.length = carried.length)
      | _ => false
    | _, _ => false

def verifyLoopBody (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (bodyScope : ScopeRef) (outer : List CoreOperandRef)
    (inner bodyOutputs outputs : List CoreWireRef) : Bool :=
  decide (outer.length = inner.length) && decide (bodyOutputs.length = outputs.length) &&
    (outer.zip inner).all (fun pair =>
      verifyOperand workflow pair.1 && verifyWire workflow pair.2 &&
        decide (pair.1.node = operation) && decide (pair.2.node.stage = operation.stage) &&
        decide (pair.2.node.scope = bodyScope)) &&
    bodyOutputs.all (fun output =>
      verifyWire workflow output && decide (output.node.stage = operation.stage) &&
        decide (output.node.scope = bodyScope)) &&
    outputs.zipIdx.all (fun output =>
      verifyOperationOutput workflow operation output.1 && decide (output.1.port = output.2)) &&
    match resolveScope workflow { operation with scope := bodyScope } with
    | some body =>
        decide body.inputNames.Nodup &&
          decide ((scopeInputWires body).length = body.inputNames.length) &&
          decide (inner.map wireRef = scopeInputWires body) &&
          decide (bodyOutputs.map wireRef = scopeOutputWires body)
    | none => false

def verifyParallelLoop (workflow : Mxx.Ir.Workflow) (reference : ParallelLoopRef) : Bool :=
  decide (reference.bodyScope =
    .parallelBody reference.operation.scope reference.operation.node) &&
    reference.arguments.zipIdx.all (fun argument =>
      decide (argument.1.node = reference.operation) &&
        decide (argument.1.operand = argument.2) && verifyOperand workflow argument.1) &&
    reference.outputs.zipIdx.all (fun output =>
      verifyOperationOutput workflow reference.operation output.1 &&
        decide (output.1.port = output.2)) &&
    reference.bodyInputs.all (fun input =>
      verifyWire workflow input && decide (input.node.stage = reference.operation.stage) &&
        decide (input.node.scope = reference.bodyScope)) &&
    reference.bodyOutputs.all (fun output =>
      verifyWire workflow output && decide (output.node.stage = reference.operation.stage) &&
        decide (output.node.scope = reference.bodyScope)) &&
    match resolveNode workflow reference.operation,
        resolveScope workflow { reference.operation with scope := reference.bodyScope } with
    | some node, some body => match node.kind with
      | .parallelLoop definition count indexSlot bindings inputModes =>
          decide (definition = reference.bodyScope.definitionName) &&
            decide (count = reference.count) && decide (indexSlot = reference.indexSlot) &&
            decide (bindings = reference.bindings) &&
            decide (inputModes = reference.inputModes.map CertifiedLoopInputMode.toIr) &&
            decide (node.arguments = reference.arguments.map (wireRef ∘ CoreOperandRef.wire)) &&
            decide body.inputNames.Nodup &&
            decide ((scopeInputWires body).length = body.inputNames.length) &&
            decide (reference.arguments.length = body.inputNames.length) &&
            decide (reference.bodyInputs.map wireRef = scopeInputWires body) &&
            decide (reference.bodyOutputs.map wireRef = scopeOutputWires body) &&
            decide (node.outputCount = reference.outputs.length)
      | _ => false
    | _, _ => false

def verifySequentialLoopRef (workflow : Mxx.Ir.Workflow)
    (reference : SequentialLoopRef) : Bool :=
  decide (reference.bodyScope =
    .sequentialBody reference.operation.scope reference.operation.node) &&
    reference.arguments.zipIdx.all (fun argument =>
      decide (argument.1.node = reference.operation) &&
        decide (argument.1.operand = argument.2) && verifyOperand workflow argument.1) &&
    reference.outputs.zipIdx.all (fun output =>
      verifyOperationOutput workflow reference.operation output.1 &&
        decide (output.1.port = output.2)) &&
    reference.bodyInputs.all (fun input =>
      verifyWire workflow input && decide (input.node.stage = reference.operation.stage) &&
        decide (input.node.scope = reference.bodyScope)) &&
    reference.bodyOutputs.all (fun output =>
      verifyWire workflow output && decide (output.node.stage = reference.operation.stage) &&
        decide (output.node.scope = reference.bodyScope)) &&
    match resolveNode workflow reference.operation,
        resolveScope workflow { reference.operation with scope := reference.bodyScope } with
    | some node, some body => match node.kind with
      | .sequentialLoop definition count indexSlot bindings carriedCount =>
          decide (definition = reference.bodyScope.definitionName) &&
            decide (count = reference.count) && decide (indexSlot = reference.indexSlot) &&
            decide (bindings = reference.bindings) &&
            decide (carriedCount = reference.carriedCount) &&
            decide (node.arguments = reference.arguments.map (wireRef ∘ CoreOperandRef.wire)) &&
            decide body.inputNames.Nodup &&
            decide ((scopeInputWires body).length = body.inputNames.length) &&
            decide (reference.arguments.length = body.inputNames.length) &&
            decide (reference.bodyInputs.map wireRef = scopeInputWires body) &&
            decide (reference.bodyOutputs.map wireRef = scopeOutputWires body) &&
            decide (node.outputCount = reference.outputs.length) &&
            decide (reference.outputs.length = reference.carriedCount) &&
            decide (reference.bodyOutputs.length = reference.carriedCount)
      | _ => false
    | _, _ => false

def verifyExactParallelLoopRole (workflow : Mxx.Ir.Workflow)
    (reference : ParallelLoopRef) (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List CertifiedLoopInputMode) : Bool :=
  verifyParallelLoop workflow reference &&
    decide (reference.count = count) && decide (reference.indexSlot = indexSlot) &&
    decide (reference.bindings = []) && decide (reference.inputModes = inputModes) &&
    decide (reference.bodyOutputs.length = 1) && decide (reference.outputs.length = 1)

def verifyExactParallelNodeRole (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (count : Mxx.Ir.IntExpr) (indexSlot : Nat)
    (inputModes : List Mxx.Ir.LoopInputMode) : Bool :=
  let bodyScope := ScopeRef.parallelBody operation.scope operation.node
  match resolveNode workflow operation,
      resolveScope workflow { operation with scope := bodyScope } with
  | some node, some body =>
      match node.kind with
      | .parallelLoop definition actualCount actualIndexSlot bindings actualModes =>
          decide (definition = bodyScope.definitionName) && decide (actualCount = count) &&
            decide (actualIndexSlot = indexSlot) && decide (bindings = []) &&
            decide (actualModes = inputModes) && decide (node.outputCount = 1) &&
            decide ((scopeOutputWires body).length = 1)
      | _ => false
  | _, _ => false

def verifyExactSequentialNodeRole (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (count : Mxx.Ir.IntExpr) (indexSlot : Nat) : Bool :=
  match resolveNode workflow operation with
  | some { kind := .sequentialLoop _ actualCount actualIndexSlot bindings _, .. } =>
      decide (actualCount = count) && decide (actualIndexSlot = indexSlot) &&
        decide (bindings = [])
  | _ => false

def verifyParallelBoundary (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (bodyScope : ScopeRef) (outer : List CoreOperandRef) (inner : List CoreWireRef)
    (bodyOutput output : CoreWireRef) : Bool :=
  decide (bodyScope = .parallelBody operation.scope operation.node) &&
    decide (outer.length = inner.length) &&
    outer.zipIdx.all (fun operand =>
      decide (operand.1.node = operation) && decide (operand.1.operand = operand.2) &&
        verifyOperand workflow operand.1) &&
    inner.all (fun input =>
      verifyWire workflow input && decide (input.node.stage = operation.stage) &&
        decide (input.node.scope = bodyScope)) &&
    verifyWire workflow bodyOutput && decide (bodyOutput.node.stage = operation.stage) &&
    decide (bodyOutput.node.scope = bodyScope) &&
    verifyOperationOutput workflow operation output && decide (output.port = 0) &&
    match resolveNode workflow operation,
        resolveScope workflow { operation with scope := bodyScope } with
    | some node, some body => match node.kind with
      | .parallelLoop definition _ _ _ _ =>
          decide (definition = bodyScope.definitionName) &&
            decide (node.arguments = outer.map (wireRef ∘ CoreOperandRef.wire)) &&
            decide body.inputNames.Nodup &&
            decide ((scopeInputWires body).length = body.inputNames.length) &&
            decide (outer.length = body.inputNames.length) &&
            decide (inner.map wireRef = scopeInputWires body) &&
            decide (scopeOutputWires body = [wireRef bodyOutput]) &&
            decide (node.outputCount = 1)
      | _ => false
    | _, _ => false

def verifyParallelMatrixBinary (workflow : Mxx.Ir.Workflow)
    (reference : ParallelMatrixBinaryRef) (expected : Mxx.Ir.NodeKind) : Bool :=
  verifyParallelBoundary workflow reference.parallelLoop reference.bodyScope
      [reference.leftFamily, reference.rightFamily] [reference.bodyLeft, reference.bodyRight]
      reference.bodyOutput reference.outputFamily &&
    verifyMatrixBinary workflow reference.operation expected &&
    decide (reference.operation.left.wire = reference.bodyLeft) &&
    decide (reference.operation.right.wire = reference.bodyRight) &&
    decide (reference.operation.output = reference.bodyOutput)

def verifyParallelFamilyGet (workflow : Mxx.Ir.Workflow)
    (reference : ParallelFamilyGetRef) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
    verifyDynamicGet workflow reference.get reference.bodySource &&
    decide (reference.parallelLoop.arguments = [reference.indexFamily, reference.sourceFamily]) &&
    decide (reference.parallelLoop.bodyInputs = [reference.bodyIndex, reference.bodySource]) &&
    decide (reference.parallelLoop.bodyOutputs = [reference.get.output]) &&
    decide (reference.parallelLoop.outputs = [reference.outputFamily]) &&
    decide (reference.get.index.wire = reference.bodyIndex)

def verifyParallelSixWaySelect (workflow : Mxx.Ir.Workflow)
    (reference : ParallelSixWaySelectRef) : Bool :=
  let outer := reference.selectorFamily :: List.ofFn reference.branchFamilies
  let inner := reference.bodySelector :: List.ofFn reference.bodyBranches
  verifyParallelBoundary workflow reference.parallelLoop reference.bodyScope outer inner
      reference.bodyOutput reference.outputFamily &&
    verifySixWaySelect workflow reference.select &&
    decide (reference.select.selector.wire = reference.bodySelector) &&
    decide ((List.ofFn reference.select.branches).map (·.wire) =
      List.ofFn reference.bodyBranches) &&
    decide (reference.select.output = reference.bodyOutput)

def verifyParallelTwoWaySelect (workflow : Mxx.Ir.Workflow)
    (reference : ParallelTwoWaySelectRef) : Bool :=
  let outer := reference.selectorFamily :: List.ofFn reference.branchFamilies
  let inner := reference.bodySelector :: List.ofFn reference.bodyBranches
  verifyParallelBoundary workflow reference.parallelLoop reference.bodyScope outer inner
      reference.bodyOutput reference.outputFamily &&
    verifyTwoWaySelect workflow reference.select &&
    decide (reference.select.selector.wire = reference.bodySelector) &&
    decide ((List.ofFn reference.select.branches).map (·.wire) =
      List.ofFn reference.bodyBranches) &&
    decide (reference.select.output = reference.bodyOutput)

def verifyEvaluateInt (workflow : Mxx.Ir.Workflow) (reference : EvaluateIntRef) : Bool :=
  verifyOperationOutput workflow reference.operation reference.evaluated &&
    decide (reference.evaluated.port = 0) &&
    (match resolveNode workflow reference.operation with
    | some { kind := .evaluateInt expression, arguments, outputCount } =>
        decide (expression = reference.expression) && decide arguments.isEmpty &&
          decide (outputCount = 1)
    | _ => false) &&
    match reference.materialization with
    | none => decide (reference.output = reference.evaluated)
    | some materialization =>
        verifyBinaryNode workflow materialization &&
          (match resolveNode workflow materialization.operation with
          | some { kind := .intBinary .add, outputCount, .. } => decide (outputCount = 1)
          | _ => false) &&
          decide (materialization.left.wire = reference.evaluated) &&
          verifyConstantIntWire workflow materialization.right.wire 0 &&
          decide (reference.output = materialization.output)

def verifyStageInputLayout (workflow : Mxx.Ir.Workflow) (stage : String)
    (input : StageInputLayout) : Bool :=
  verifyInputWire workflow { node := input.node, port := 0 } stage input.name

def verifyStageOutputLayout (workflow : Mxx.Ir.Workflow) (stage : String)
    (output : StageOutputLayout) : Bool :=
  decide (output.wire.node.stage = stage) && decide (output.wire.node.scope = .root) &&
    verifyWire workflow output.wire

def verifyStageInterface (workflow : Mxx.Ir.Workflow)
    (interface : StageInterfaceLayout) : Bool :=
  match resolveStage workflow interface.stage with
  | none => false
  | some stage =>
      interface.inputs.all (verifyStageInputLayout workflow interface.stage) &&
        interface.outputs.all (verifyStageOutputLayout workflow interface.stage) &&
        decide (interface.inputs.map (·.node)).Nodup &&
        decide (interface.inputs.map (·.name) = stage.program.root.inputNames) &&
        decide (interface.inputs.map (·.name) = stage.inputs.map Prod.fst) &&
        decide (interface.outputs.map (fun output => (output.name, wireRef output.wire)) =
          stage.program.root.outputs)

structure ArtifactBindingKey where
  consumerStage : String
  consumerInput : String
  producerStage : String
  producerOutput : String
  deriving DecidableEq

def ArtifactProvenance.bindingKey (reference : ArtifactProvenance) : ArtifactBindingKey := {
  consumerStage := reference.consumerStage
  consumerInput := reference.consumerInput.name
  producerStage := reference.producerStage
  producerOutput := reference.producerOutput.name
}

def workflowArtifactBindings (workflow : Mxx.Ir.Workflow) : List ArtifactBindingKey :=
  workflow.stages.flatMap fun stage => stage.inputs.filterMap fun input => match input.2 with
    | .artifact producerStage producerOutput => some {
        consumerStage := stage.id
        consumerInput := input.1
        producerStage
        producerOutput
      }
    | .protocol _ => none

def interfaceForStage (layout : DiamondWorkflowLayout)
    (stage : String) : Option StageInterfaceLayout :=
  if layout.encryption.stage = stage then some layout.encryption
  else if layout.decryption.stage = stage then some layout.decryption
  else none

def verifyArtifactProvenance (workflow : Mxx.Ir.Workflow)
    (reference : ArtifactProvenance) : Bool :=
  match resolveStage workflow reference.producerStage,
      resolveStage workflow reference.consumerStage with
  | some producer, some consumer =>
      decide (reference.producerOutput.wire.node.stage = reference.producerStage) &&
        decide (reference.producerOutput.wire.node.scope = .root) &&
        verifyWire workflow reference.producerOutput.wire &&
        decide (producer.program.root.outputs.any fun output =>
          output = (reference.producerOutput.name, wireRef reference.producerOutput.wire)) &&
        verifyStageInputLayout workflow reference.consumerStage reference.consumerInput &&
        decide (consumer.inputs.any fun input => input =
          (reference.consumerInput.name,
            .artifact reference.producerStage reference.producerOutput.name))
  | _, _ => false

def verifyArtifactInLayout (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout)
    (reference : ArtifactProvenance) : Bool :=
  verifyArtifactProvenance workflow reference &&
    match interfaceForStage layout reference.producerStage,
        interfaceForStage layout reference.consumerStage with
    | some producer, some consumer =>
        decide (reference.producerOutput ∈ producer.outputs) &&
          decide (reference.consumerInput ∈ consumer.inputs)
    | _, _ => false

def definitionsUnique (workflow : Mxx.Ir.Workflow) : Bool :=
  workflow.stages.all fun stage => decide (stage.program.definitions.map Prod.fst).Nodup

def verifyScopeSsaOrder (scope : Mxx.Ir.Scope) : Bool :=
  scope.nodes.zipIdx.all (fun entry =>
    let node := entry.1
    let index := entry.2
    (match node.kind with
      | .input _ => decide node.arguments.isEmpty
      | _ => true) &&
    node.arguments.all (fun argument =>
      decide (argument.node < index) &&
      match scope.nodes[argument.node]? with
      | some producer => decide (argument.port < producer.outputCount)
      | none => false)) &&
  scope.outputs.all (fun output =>
    match scope.nodes[output.2.node]? with
    | some producer => decide (output.2.port < producer.outputCount)
    | none => false)

def verifyWorkflowSsaOrder (workflow : Mxx.Ir.Workflow) : Bool :=
  workflow.stages.all (fun stage =>
    verifyScopeSsaOrder stage.program.root &&
    stage.program.definitions.all (fun definition => verifyScopeSsaOrder definition.2))

def verifyWorkflow (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout) : Bool :=
  let declaredBindings := layout.artifacts.map ArtifactProvenance.bindingKey
  let actualBindings := workflowArtifactBindings workflow
  decide (layout.encryption.stage = "encrypt") && decide (layout.decryption.stage = "decrypt") &&
    decide (workflow.entrypoint = layout.decryption.stage) &&
    verifyStageInterface workflow layout.encryption &&
    verifyStageInterface workflow layout.decryption &&
    layout.artifacts.all (verifyArtifactInLayout workflow layout) &&
    decide declaredBindings.Nodup && decide actualBindings.Nodup &&
    decide (declaredBindings.length = actualBindings.length) &&
    declaredBindings.all (fun binding => decide (binding ∈ actualBindings)) &&
    decide (layout.artifacts.map (fun artifact =>
      (artifact.consumerStage, artifact.consumerInput.node))).Nodup &&
    match workflow.stages with
    | [encrypt, decrypt] =>
        decide (encrypt.id = layout.encryption.stage) &&
          decide (decrypt.id = layout.decryption.stage) &&
          decide (layout.encryption.outputs.map (·.name) = [
            "diamond_decoder_preimage", "diamond_initial_state", "diamond_k_preimage",
            "diamond_one_preimage", "diamond_public_keys", "diamond_r_decomposed",
            "diamond_transitions", "diamond_witness_preimages"]) &&
          decide (layout.decryption.outputs.map (·.name) =
            ["diamond-decoded", "diamond-noisy-plaintext"])
    | _ => false

def verifyParallelIndexFormulaRef (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
    verifyWire workflow reference.bodyOutput &&
    decide (reference.bodyOutput.node.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.parallelLoop.bodyOutputs = [reference.bodyOutput]) &&
    decide (reference.parallelLoop.outputs.length = 1)

private def localWire (node : Nat) : Mxx.Ir.WireRef := { node := node, port := 0 }

private def exactNode (kind : Mxx.Ir.NodeKind)
    (arguments : List Mxx.Ir.WireRef := []) : Mxx.Ir.Node := {
  kind := kind
  arguments := arguments
}

private def exactNodeWithOutputs (kind : Mxx.Ir.NodeKind)
    (arguments : List Mxx.Ir.WireRef) (outputCount : Nat) : Mxx.Ir.Node := {
  kind := kind
  arguments := arguments
  outputCount := outputCount
}

private def inputCount : Mxx.Ir.IntExpr := .parameter "diamond_input_count"
private def batchBits : Mxx.Ir.IntExpr := .parameter "diamond_batch_bits"
private def digitBase : Mxx.Ir.IntExpr := .parameter "diamond_digit_base"
private def digitCount : Mxx.Ir.IntExpr := .parameter "diamond_digit_count"
private def modulus : Mxx.Ir.IntExpr := .parameter "diamond_modulus"
private def ringDimension : Mxx.Ir.IntExpr := .parameter "diamond_ring_dimension"
private def errorBound : Mxx.Ir.IntExpr :=
  .parameter "diamond_error_max_coefficient_bound"
private def preimageBound : Mxx.Ir.IntExpr :=
  .parameter "diamond_preimage_max_coefficient_bound"
private def witnessSize : Mxx.Ir.IntExpr := .multiply batchBits inputCount
private def publicKeyCount : Mxx.Ir.IntExpr := .add witnessSize (.constant 1)
private def stateWidth : Mxx.Ir.IntExpr :=
  .add (.constant 1) (.multiply batchBits inputCount)
private def stateColumns : Mxx.Ir.IntExpr :=
  .add (.constant 4) (.multiply (.constant 2) digitCount)
private def trapdoorBaseCount : Mxx.Ir.IntExpr :=
  .add (.add (.add (.constant 1) witnessSize) (.multiply witnessSize inputCount)) inputCount
private def transitionStride : Mxx.Ir.IntExpr :=
  .add (.multiply (.multiply batchBits digitBase) inputCount) digitBase
private def transitionCount : Mxx.Ir.IntExpr :=
  .add (.multiply (.multiply (.multiply batchBits digitBase) inputCount) inputCount)
    (.multiply digitBase inputCount)

private def matrixType (rows columns : Mxx.Ir.IntExpr) : Mxx.Ir.MatrixTypeExpr := {
  modulus := modulus
  ringDimension := ringDimension
  rows := rows
  columns := columns
}

private def unitMatrixType : Mxx.Ir.MatrixTypeExpr := matrixType (.constant 1) (.constant 1)
private def trapdoorMatrixType : Mxx.Ir.MatrixTypeExpr :=
  matrixType (.constant 2) stateColumns
private def initialErrorMatrixType : Mxx.Ir.MatrixTypeExpr :=
  matrixType (.constant 1) stateColumns
private def transitionErrorMatrixType : Mxx.Ir.MatrixTypeExpr :=
  matrixType (.constant 2) stateColumns
private def preimageMatrixType (columns : Mxx.Ir.IntExpr) : Mxx.Ir.MatrixTypeExpr :=
  matrixType stateColumns columns

private def witnessPublicKeyTag : List Nat :=
  [109, 120, 120, 58, 100, 105, 97, 109, 111, 110, 100, 45, 119, 101, 58, 119, 105,
    116, 110, 101, 115, 115, 95, 112, 117, 98, 108, 105, 99, 95, 107, 101, 121, 115]

private def kPublicKeyTag : List Nat :=
  [109, 120, 120, 58, 100, 105, 97, 109, 111, 110, 100, 45, 119, 101, 58, 107, 95,
    112, 117, 98, 108, 105, 99, 95, 107, 101, 121]

private def rTag : List Nat :=
  [109, 120, 120, 58, 100, 105, 97, 109, 111, 110, 100, 45, 119, 101, 58, 114]

private def verifyExactParallelBody (workflow : Mxx.Ir.Workflow)
    (reference : ParallelLoopRef) (expected : List Mxx.Ir.Node) : Bool :=
  match resolveScope workflow { reference.operation with scope := reference.bodyScope } with
  | some body => decide (body.nodes = expected)
  | none => false

private def sameCoreScopeWire (context : CoreNodeRef)
    (wire : Mxx.Ir.WireRef) : CoreWireRef := {
  node := { context with node := wire.node }
  port := wire.port
}

def verifyOnlineSourceLowerBound (workflow : Mxx.Ir.Workflow)
    (wire : CoreWireRef) : Bool :=
  decide (wire.port = 0) && match resolveNode workflow wire.node with
  | some addNode => match addNode.kind, addNode.arguments, addNode.outputCount with
    | .intBinary .add, [product, one], 1 =>
        let product := sameCoreScopeWire wire.node product
        let one := sameCoreScopeWire wire.node one
        decide (product.port = 0) && verifyConstantIntWire workflow one 1 &&
        match resolveNode workflow product.node with
        | some productNode => match productNode.kind, productNode.arguments,
            productNode.outputCount with
          | .intBinary .multiply, [level, width], 1 =>
              let level := sameCoreScopeWire wire.node level
              let width := sameCoreScopeWire wire.node width
              decide (level.port = 0) && decide (width.port = 0) &&
                match resolveNode workflow level.node, resolveNode workflow width.node with
              | some levelNode, some widthNode =>
                  decide (levelNode.kind = .evaluateInt (.loopIndex 0)) &&
                  decide levelNode.arguments.isEmpty && decide (levelNode.outputCount = 1) &&
                  decide (widthNode.kind = .evaluateInt batchBits) &&
                  decide widthNode.arguments.isEmpty && decide (widthNode.outputCount = 1)
              | _, _ => false
          | _, _, _ => false
        | none => false
    | _, _, _ => false
  | none => false

private def preprocessingSourceIndexNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.evaluateInt transitionStride),
  exactNode (.intBinary .divide) [localWire 0, localWire 1],
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .multiply) [localWire 2, localWire 3],
  exactNode (.evaluateInt batchBits),
  exactNode (.intBinary .multiply) [localWire 2, localWire 5],
  exactNode (.constantInt 1),
  exactNode (.intBinary .add) [localWire 6, localWire 7],
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .remainder) [localWire 0, localWire 9],
  exactNode (.intCompare .lessEqual) [localWire 8, localWire 10],
  exactNode .boolToInt [localWire 11],
  exactNode (.constantInt 0),
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 13, localWire 14],
  exactNode .select [localWire 12, localWire 10, localWire 15],
  exactNode (.intBinary .add) [localWire 4, localWire 16]
]

private def preprocessingTargetIndexNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.evaluateInt transitionStride),
  exactNode (.intBinary .divide) [localWire 0, localWire 1],
  exactNode (.constantInt 1),
  exactNode (.intBinary .add) [localWire 2, localWire 3],
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .multiply) [localWire 4, localWire 5],
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .remainder) [localWire 0, localWire 7],
  exactNode (.intBinary .add) [localWire 6, localWire 8]
]

private def preprocessingDigitSecretIndexNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .divide) [localWire 0, localWire 1]
]

private def onlineSourceIndexNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt (.loopIndex 1)),
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 1, localWire 4],
  exactNode (.constantInt 0),
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 6, localWire 7],
  exactNode .select [localWire 3, localWire 5, localWire 8]
]

private def onlineTransitionIndexNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt transitionStride),
  exactNode (.intBinary .multiply) [localWire 0, localWire 1],
  exactNode (.input "__capture_1"),
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .multiply) [localWire 3, localWire 4],
  exactNode (.intBinary .add) [localWire 2, localWire 5],
  exactNode (.evaluateInt (.loopIndex 1)),
  exactNode (.intBinary .add) [localWire 6, localWire 7]
]

private def initialStateExpansionNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.constantInt 0),
  exactNode (.intCompare .equal) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.zeroMatrix {
    modulus := .parameter "diamond_modulus"
    ringDimension := .parameter "diamond_ring_dimension"
    rows := .constant 1
    columns := .add (.constant 4) (.multiply (.constant 2)
      (.parameter "diamond_digit_count"))
  }),
  exactNode (.input "__capture_0"),
  exactNode .select [localWire 3, localWire 4, localWire 5]
]

private def witnessDigitOuterNodes (definition : String) : List Mxx.Ir.Node := [
  exactNode (.constantInt 0),
  exactNode (.constantInt 1),
  exactNode (.input "pack-bit-source"),
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.evaluateInt batchBits),
  exactNodeWithOutputs (.sequentialLoop definition batchBits 1 [] 2)
    [localWire 0, localWire 1, localWire 2, localWire 3, localWire 4] 2
]

private def witnessDigitScanNodes : List Mxx.Ir.Node := [
  exactNode (.input "arg-0-integer"),
  exactNode (.input "arg-2-family"),
  exactNode (.input "__capture_0"),
  exactNode (.input "__capture_1"),
  exactNode (.intBinary .multiply) [localWire 2, localWire 3],
  exactNode (.evaluateInt (.loopIndex 1)),
  exactNode (.intBinary .add) [localWire 4, localWire 5],
  exactNode .familyGetDynamic [localWire 1, localWire 6],
  exactNode (.input "arg-1-integer"),
  exactNode (.intBinary .multiply) [localWire 7, localWire 8],
  exactNode (.intBinary .add) [localWire 0, localWire 9],
  exactNode (.constantInt 2),
  exactNode (.intBinary .multiply) [localWire 8, localWire 11]
]

private def transitionTargetNodes (definition : String) : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .remainder) [localWire 0, localWire 1],
  exactNode (.constantInt 0),
  exactNode (.intCompare .equal) [localWire 2, localWire 3],
  exactNode .boolToInt [localWire 4],
  exactNode (.input "arg-0-matrix"),
  exactNode (.concat .diagonal) [localWire 6, localWire 6],
  exactNode (.identityMatrix unitMatrixType),
  exactNode (.concat .diagonal) [localWire 6, localWire 8],
  exactNode .select [localWire 5, localWire 7, localWire 9],
  exactNode (.evaluateInt stateWidth),
  exactNode (.intBinary .divide) [localWire 0, localWire 11],
  exactNode (.evaluateInt digitBase),
  exactNode (.intBinary .remainder) [localWire 12, localWire 13],
  exactNode (.evaluateInt transitionStride),
  exactNode (.intBinary .divide) [localWire 0, localWire 15],
  exactNode (.evaluateInt batchBits),
  exactNode (.intBinary .multiply) [localWire 16, localWire 17],
  exactNode (.constantInt 1),
  exactNode (.intBinary .add) [localWire 18, localWire 19],
  exactNode (.sequentialLoop definition batchBits 1 [] 1)
    [localWire 10, localWire 14, localWire 2, localWire 20, localWire 6],
  exactNode (.input "arg-1-matrix"),
  exactNode .matrixMultiply [localWire 21, localWire 22],
  exactNode (.gaussianSample transitionErrorMatrixType errorBound),
  exactNode .matrixAdd [localWire 23, localWire 24]
]

private def transitionSelectorBitNodes : List Mxx.Ir.Node := [
  exactNode (.input "arg-2-integer"),
  exactNode (.input "arg-3-integer"),
  exactNode (.evaluateInt (.loopIndex 1)),
  exactNode (.intBinary .add) [localWire 1, localWire 2],
  exactNode (.intCompare .equal) [localWire 0, localWire 3],
  exactNode .boolToInt [localWire 4],
  exactNode (.input "arg-0-matrix"),
  exactNode (.input "arg-4-matrix"),
  exactNode (.input "arg-1-integer"),
  exactNode (.bitExtract (.loopIndex 1)) [localWire 8],
  exactNode .boolToInt [localWire 9],
  exactNode (.zeroMatrix unitMatrixType),
  exactNode (.identityMatrix unitMatrixType),
  exactNode .select [localWire 10, localWire 11, localWire 12],
  exactNode .matrixMultiply [localWire 7, localWire 13],
  exactNode (.concat .columns) [localWire 7, localWire 14],
  exactNode (.zeroMatrix (matrixType (.constant 1) (.constant 2))),
  exactNode (.concat .rows) [localWire 15, localWire 16],
  exactNode .select [localWire 5, localWire 6, localWire 17]
]

private def encryptionPublicIndexNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.input "__capture_1"),
  exactNode (.intCompare .lessEqual) [localWire 1, localWire 4],
  exactNode .boolToInt [localWire 5],
  exactNode (.intBinary .multiply) [localWire 3, localWire 6],
  exactNode (.constantInt 0),
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 8, localWire 9],
  exactNode (.intBinary .subtract) [localWire 1, localWire 0],
  exactNode (.constantInt 1),
  exactNode (.intBinary .add) [localWire 11, localWire 12],
  exactNode .select [localWire 7, localWire 10, localWire 13]
]

private def encryptionPackedInputNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.input "__capture_1"),
  exactNode (.input "__capture_2"),
  exactNode (.intCompare .lessEqual) [localWire 1, localWire 5],
  exactNode .boolToInt [localWire 6],
  exactNode (.input "item"),
  exactNode .select [localWire 7, localWire 4, localWire 8],
  exactNode .select [localWire 3, localWire 4, localWire 9]
]

private def encryptionCircuitInputNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.input "__capture_0"),
  exactNode (.constantInt 1),
  exactNode (.intBinary .subtract) [localWire 1, localWire 2],
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 3],
  exactNode .boolToInt [localWire 4],
  exactNode (.input "arg-1-matrix"),
  exactNode (.input "arg-0-integer"),
  exactNode (.input "__capture_1"),
  exactNode (.input "__capture_2"),
  exactNode .select [localWire 7, localWire 8, localWire 9],
  exactNode .select [localWire 5, localWire 6, localWire 10]
]

private def decryptionPackedIndexNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.input "__capture_1"),
  exactNode (.intCompare .lessEqual) [localWire 1, localWire 4],
  exactNode .boolToInt [localWire 5],
  exactNode (.intBinary .multiply) [localWire 3, localWire 6],
  exactNode (.constantInt 0),
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 8, localWire 9],
  exactNode (.intBinary .subtract) [localWire 1, localWire 0],
  exactNode .select [localWire 7, localWire 10, localWire 11]
]

private def decryptionWitnessIndexNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.constantInt 0),
  exactNode (.intBinary .add) [localWire 0, localWire 1]
]

private def decryptionActiveWitnessNodes : List Mxx.Ir.Node := [
  exactNode (.input "__capture_0"),
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 1],
  exactNode .boolToInt [localWire 2],
  exactNode (.input "__capture_1"),
  exactNode (.intCompare .lessEqual) [localWire 1, localWire 4],
  exactNode .boolToInt [localWire 5],
  exactNode (.intBinary .multiply) [localWire 3, localWire 6]
]

private def decryptionActiveInstanceNodes : List Mxx.Ir.Node := [
  exactNode (.evaluateInt (.loopIndex 0)),
  exactNode (.input "__capture_0"),
  exactNode (.constantInt 1),
  exactNode (.intBinary .subtract) [localWire 1, localWire 2],
  exactNode (.intCompare .lessEqual) [localWire 0, localWire 3],
  exactNode .boolToInt [localWire 4]
]

def verifyPreprocessingSourceIndexFormula (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelIndexFormulaRef workflow reference &&
    decide (reference.parallelLoop.count = transitionCount) &&
    decide reference.parallelLoop.arguments.isEmpty &&
    decide (reference.parallelLoop.indexSlot = 0) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide reference.parallelLoop.inputModes.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 17) &&
    verifyExactParallelBody workflow reference.parallelLoop preprocessingSourceIndexNodes

def verifyPreprocessingTargetIndexFormula (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelIndexFormulaRef workflow reference &&
    decide (reference.parallelLoop.count = transitionCount) &&
    decide reference.parallelLoop.arguments.isEmpty &&
    decide (reference.parallelLoop.indexSlot = 0) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide reference.parallelLoop.inputModes.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 9) &&
    verifyExactParallelBody workflow reference.parallelLoop preprocessingTargetIndexNodes

def verifyPreprocessingDigitSecretIndexFormula (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelIndexFormulaRef workflow reference &&
    decide (reference.parallelLoop.count = transitionCount) &&
    decide reference.parallelLoop.arguments.isEmpty &&
    decide (reference.parallelLoop.indexSlot = 0) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide reference.parallelLoop.inputModes.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 2) &&
    verifyExactParallelBody workflow reference.parallelLoop preprocessingDigitSecretIndexNodes

def verifyOnlineSourceIndexFormula (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelIndexFormulaRef workflow reference &&
    decide (reference.parallelLoop.count = stateWidth) &&
    decide (reference.parallelLoop.inputModes = [.broadcast]) &&
    (match reference.parallelLoop.arguments with
    | [lowerBound] => verifyOnlineSourceLowerBound workflow lowerBound.wire
    | _ => false) &&
    decide (reference.parallelLoop.indexSlot = 1) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 9) &&
    verifyExactParallelBody workflow reference.parallelLoop onlineSourceIndexNodes

def verifyOnlineTransitionIndexFormula (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : Bool :=
  verifyParallelIndexFormulaRef workflow reference &&
    decide (reference.parallelLoop.count = stateWidth) &&
    decide (reference.parallelLoop.inputModes = [.broadcast, .broadcast]) &&
    decide (reference.parallelLoop.indexSlot = 1) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 8) &&
    verifyExactParallelBody workflow reference.parallelLoop onlineTransitionIndexNodes

def verifyInitialStateExpansionRef (workflow : Mxx.Ir.Workflow)
    (reference : InitialStateExpansionRef) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
    verifyWire workflow reference.bodyOutput &&
    decide (reference.bodyOutput.node.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.parallelLoop.bodyOutputs = [reference.bodyOutput]) &&
    decide (reference.parallelLoop.arguments.length = 1) &&
    decide (reference.parallelLoop.outputs.length = 1) &&
    decide (reference.parallelLoop.count = stateWidth) &&
    decide (reference.parallelLoop.inputModes = [.broadcast]) &&
    decide (reference.parallelLoop.indexSlot = 0) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide (wireRef reference.bodyOutput = localWire 6) &&
    verifyExactParallelBody workflow reference.parallelLoop initialStateExpansionNodes

def verifyWitnessDigitPackingInputNames (workflow : Mxx.Ir.Workflow)
    (reference : WitnessDigitPackingRef) : Bool :=
  (match resolveScope workflow {
      reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } with
  | some outerBody => decide (outerBody.inputNames = ["pack-bit-source"])
  | none => false) &&
  (match resolveScope workflow {
      reference.bitScan.operation with scope := reference.bitScan.bodyScope } with
  | some scanBody => decide (scanBody.inputNames =
      ["arg-0-integer", "arg-1-integer", "arg-2-family", "__capture_0",
        "__capture_1"])
  | none => false)

private def verifyWitnessDigitOuterBody (workflow : Mxx.Ir.Workflow)
    (reference : WitnessDigitPackingRef) : Bool :=
  verifyExactParallelBody workflow reference.parallelLoop
    (witnessDigitOuterNodes reference.bitScan.bodyScope.definitionName)

private def verifyWitnessDigitScanBody (workflow : Mxx.Ir.Workflow)
    (reference : WitnessDigitPackingRef) : Bool :=
  match resolveScope workflow { reference.bitScan.operation with
      scope := reference.bitScan.bodyScope } with
  | some body => decide (body.nodes = witnessDigitScanNodes)
  | none => false

def verifyWitnessDigitPackingRef (workflow : Mxx.Ir.Workflow)
    (reference : WitnessDigitPackingRef) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
  verifySequentialLoopRef workflow reference.bitScan &&
    decide (reference.parallelLoop.count = inputCount) &&
    decide (reference.parallelLoop.indexSlot = 0) &&
    decide reference.parallelLoop.bindings.isEmpty &&
    decide (reference.parallelLoop.inputModes = [.broadcast]) &&
    decide (reference.parallelLoop.bodyOutputs.length = 1) &&
    decide (reference.parallelLoop.outputs.length = 1) &&
    decide (reference.bitScan.count = batchBits) &&
    decide (reference.bitScan.indexSlot = 1) &&
    decide reference.bitScan.bindings.isEmpty &&
    decide (reference.bitScan.carriedCount = 2) &&
    decide (reference.bodyOutput.node.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.parallelLoop.bodyOutputs = [reference.bodyOutput]) &&
    decide (reference.bitScan.operation = reference.bodyOutput.node) &&
    decide (reference.bitScan.outputs[0]? = some reference.bodyOutput) &&
    decide (wireRef reference.bodyOutput = localWire 5) &&
    decide (reference.bitScan.bodyOutputs.map wireRef = [localWire 10, localWire 12]) &&
    verifyWitnessDigitOuterBody workflow reference &&
    verifyWitnessDigitScanBody workflow reference &&
    verifyWitnessDigitPackingInputNames workflow reference

def verifyInputInjectionExternalSources (workflow : Mxx.Ir.Workflow)
    (layout : InputInjectionLayout) : Bool :=
  (match layout.initialStatesExpansion.parallelLoop.arguments[0]? with
  | some initialState =>
      verifyInputWire workflow initialState.wire "decrypt" "diamond_initial_state"
  | none => false) &&
  verifyInputWire workflow layout.transitionFamily.wire "decrypt"
    "artifact:diamond_transitions"

def verifyInputInjection (workflow : Mxx.Ir.Workflow)
    (layout : InputInjectionLayout) : Bool :=
  decide (layout.stateScan.stage = "decrypt") &&
    decide (layout.stateScan.scope = .root) &&
    verifyInitialStateExpansionRef workflow layout.initialStatesExpansion &&
    decide (layout.initialStatesExpansion.parallelLoop.outputs =
      [layout.initialStates.wire]) &&
    verifySequentialLoop workflow layout.stateScan layout.bodyScope [layout.initialStates]
      [layout.packedDigits, layout.transitionFamily] [layout.finalStates] &&
    verifyExactSequentialNodeRole workflow layout.stateScan inputCount 0 &&
    verifyLoopBody workflow layout.stateScan layout.bodyScope
      [layout.initialStates, layout.packedDigits, layout.transitionFamily]
      [layout.bodyInitialStates, layout.bodyPackedDigits, layout.bodyTransitionFamily]
      [layout.bodyFinalStates] [layout.finalStates] &&
    verifyDynamicGet workflow layout.selectedDigit layout.bodyPackedDigits &&
    verifyOnlineSourceIndexFormula workflow layout.sourceIndices &&
    verifyParallelFamilyGet workflow layout.sourceStates &&
    verifyExactParallelLoopRole workflow layout.sourceStates.parallelLoop stateWidth 1
      [.zip, .broadcast] &&
    verifyOnlineTransitionIndexFormula workflow layout.transitionIndices &&
    decide (layout.transitionIndices.parallelLoop.arguments.map (·.wire) =
      [layout.selectedDigit.index.wire, layout.selectedDigit.output]) &&
    verifyParallelFamilyGet workflow layout.selectedTransitions &&
    verifyExactParallelLoopRole workflow layout.selectedTransitions.parallelLoop stateWidth 1
      [.zip, .broadcast] &&
    verifyParallelMatrixBinary workflow layout.stateProduct .matrixMultiply &&
    verifyExactParallelNodeRole workflow layout.stateProduct.parallelLoop stateWidth 1
      [.zip, .zip] &&
    decide (layout.stateProduct.parallelLoop.scope = layout.bodyScope) &&
    decide (layout.bodyFinalStates = layout.stateProduct.outputFamily) &&
    decide (layout.sourceIndices.parallelLoop.outputs.length = 1) &&
    decide (layout.transitionIndices.parallelLoop.outputs.length = 1) &&
    decide (layout.sourceIndices.parallelLoop.outputs[0]? =
      some layout.sourceStates.indexFamily.wire) &&
    decide (layout.sourceStates.sourceFamily.wire = layout.bodyInitialStates) &&
    decide (layout.transitionIndices.parallelLoop.outputs[0]? =
      some layout.selectedTransitions.indexFamily.wire) &&
    decide (layout.selectedTransitions.sourceFamily.wire = layout.bodyTransitionFamily) &&
    decide (layout.stateProduct.leftFamily.wire = layout.sourceStates.outputFamily) &&
    decide (layout.stateProduct.rightFamily.wire = layout.selectedTransitions.outputFamily) &&
    [layout.selectedDigit.operation, layout.sourceIndices.parallelLoop.operation,
      layout.sourceStates.parallelLoop.operation,
      layout.transitionIndices.parallelLoop.operation,
      layout.selectedTransitions.parallelLoop.operation,
      layout.stateProduct.parallelLoop].all
      (fun operation => decide (operation.stage = layout.stateScan.stage)) &&
    decide (layout.selectedDigit.operation.scope = layout.bodyScope) &&
    decide (layout.sourceIndices.parallelLoop.operation.scope = layout.bodyScope) &&
    decide (layout.sourceStates.parallelLoop.operation.scope = layout.bodyScope) &&
    decide (layout.transitionIndices.parallelLoop.operation.scope = layout.bodyScope) &&
    decide (layout.selectedTransitions.parallelLoop.operation.scope = layout.bodyScope) &&
    (match resolveNode workflow layout.stateScan,
        resolveNode workflow layout.selectedDigit.index.wire.node with
    | some { kind := .sequentialLoop _ _ indexSlot _ _, .. },
        some { kind := .evaluateInt (.loopIndex selectedSlot), arguments, outputCount } =>
        decide (selectedSlot = indexSlot) && decide arguments.isEmpty &&
          decide (outputCount = 1) && decide (layout.selectedDigit.index.wire.port = 0)
    | _, _ => false) &&
  verifyInputInjectionExternalSources workflow layout

def verifyOperation (workflow : Mxx.Ir.Workflow) (reference : OperationRef) : Bool :=
  reference.inputs.zipIdx.all (fun input =>
      decide (input.1.node = reference.operation) && decide (input.1.operand = input.2) &&
        verifyOperand workflow input.1) &&
    reference.outputs.zipIdx.all (fun output =>
      decide (output.1.node = reference.operation) && decide (output.1.port = output.2) &&
        verifyWire workflow output.1) &&
    match resolveNode workflow reference.operation with
    | some node =>
        decide (node.arguments = reference.inputs.map (wireRef ∘ CoreOperandRef.wire)) &&
          decide (node.outputCount = reference.outputs.length)
    | none => false

def verifyOperationKind (workflow : Mxx.Ir.Workflow) (reference : OperationRef)
    (accept : Mxx.Ir.NodeKind → Bool) : Bool :=
  verifyOperation workflow reference &&
    match resolveNode workflow reference.operation with
    | some node => accept node.kind
    | none => false

def verifySelectOperation (workflow : Mxx.Ir.Workflow) (reference : OperationRef) : Bool :=
  verifyOperationKind workflow reference (fun kind => match kind with
    | .select => true
    | _ => false) &&
    decide (!reference.inputs.isEmpty) && decide (reference.outputs.length = 1)

def verifyParallelOperation (workflow : Mxx.Ir.Workflow)
    (reference : ParallelOperationRef) (accept : Mxx.Ir.NodeKind → Bool) : Bool :=
    verifyParallelLoop workflow reference.parallelLoop &&
    verifyOperationKind workflow reference.body accept &&
    decide (reference.body.operation.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.body.inputs.map (·.wire) = reference.parallelLoop.bodyInputs) &&
    decide (reference.body.outputs = reference.parallelLoop.bodyOutputs) &&
    decide (reference.parallelLoop.outputs.length = reference.body.outputs.length)

def verifyParallelGather (workflow : Mxx.Ir.Workflow)
    (reference : ParallelGatherRef) : Bool :=
  let outer := reference.indexFamily :: reference.sourceFamilies
  let inner := reference.bodyIndex :: reference.bodySources
  verifyParallelLoop workflow reference.parallelLoop &&
    decide (!reference.sourceFamilies.isEmpty) &&
    decide (reference.parallelLoop.arguments = outer) &&
    decide (reference.parallelLoop.bodyInputs = inner) &&
    decide (reference.gets.length = reference.sourceFamilies.length) &&
    decide (reference.bodySources.length = reference.sourceFamilies.length) &&
    decide (reference.outputFamilies.length = reference.sourceFamilies.length) &&
    decide (reference.parallelLoop.bodyOutputs = reference.gets.map (·.output)) &&
    decide (reference.parallelLoop.outputs = reference.outputFamilies) &&
    (reference.gets.zip reference.bodySources).all (fun pair =>
      verifyDynamicGet workflow pair.1 pair.2 &&
        decide (pair.1.index.wire = reference.bodyIndex))

def verifyPreimage (workflow : Mxx.Ir.Workflow) (reference : PreimageRef)
    (columns : Mxx.Ir.IntExpr) : Bool :=
  verifyOperationKind workflow reference.sample (fun kind => match kind with
    | .preimageSample matrixType cutoff =>
        decide (matrixType = preimageMatrixType columns) && decide (cutoff = preimageBound)
    | _ => false) &&
  verifyOperationKind workflow reference.materialize (fun kind => match kind with
    | .matrixScale (.constant 1) => true
    | _ => false) &&
  decide (reference.sample.inputs.length = 3) &&
  decide (reference.sample.outputs.length = 1) &&
  decide (reference.materialize.inputs.length = 1) &&
  decide (reference.materialize.outputs.length = 1) &&
  decide (reference.materialize.inputs.map (·.wire) = reference.sample.outputs)

def verifyParallelPreimage (workflow : Mxx.Ir.Workflow)
    (reference : ParallelPreimageRef) (columns : Mxx.Ir.IntExpr) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
    verifyPreimage workflow reference.body columns &&
    decide (reference.body.sample.operation.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.body.materialize.operation.scope = reference.parallelLoop.bodyScope) &&
    decide (reference.parallelLoop.bodyInputs = reference.body.sample.inputs.map (·.wire)) &&
    decide (reference.body.materialize.outputs = reference.parallelLoop.bodyOutputs)

def verifyTransitionSelectorBitBodyInputNames (workflow : Mxx.Ir.Workflow)
    (reference : TransitionSelectorLayout) : Bool :=
  match resolveScope workflow {
      reference.bitScan.operation with scope := reference.bitScan.bodyScope } with
  | some bitBody => decide (bitBody.inputNames =
      ["arg-0-matrix", "arg-1-integer", "arg-2-integer", "arg-3-integer",
        "arg-4-matrix"])
  | none => false

private def verifyTransitionSelectorBitBody (workflow : Mxx.Ir.Workflow)
    (reference : TransitionSelectorLayout) : Bool :=
  match resolveScope workflow {
      reference.bitScan.operation with scope := reference.bitScan.bodyScope } with
  | some body => decide (body.nodes = transitionSelectorBitNodes)
  | none => false

def verifyTransitionSelector (workflow : Mxx.Ir.Workflow)
    (reference : TransitionSelectorLayout) : Bool :=
  let body := reference.bitBody
  verifyOperationKind workflow reference.regular (fun kind => match kind with
    | .concat .diagonal => true
    | _ => false) &&
  verifyOperationKind workflow reference.kIdentity (fun kind => match kind with
    | .identityMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifyOperationKind workflow reference.k (fun kind => match kind with
    | .concat .diagonal => true
    | _ => false) &&
  verifySelectOperation workflow reference.initialSelect &&
  verifySequentialLoopRef workflow reference.bitScan &&
  verifyOperationKind workflow body.bitExtract (fun kind => match kind with
    | .bitExtract (.loopIndex 1) => true
    | _ => false) &&
  verifyOperationKind workflow body.bitToInt (fun kind => match kind with
    | .boolToInt => true
    | _ => false) &&
  verifyOperationKind workflow body.bitZero (fun kind => match kind with
    | .zeroMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifyOperationKind workflow body.bitOne (fun kind => match kind with
    | .identityMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifySelectOperation workflow body.bitSelect &&
  verifyOperationKind workflow body.specialProduct (fun kind => match kind with
    | .matrixMultiply => true
    | _ => false) &&
  verifyOperationKind workflow body.specialTop (fun kind => match kind with
    | .concat .columns => true
    | _ => false) &&
  verifyOperationKind workflow body.specialBottom (fun kind => match kind with
    | .zeroMatrix actualType =>
        decide (actualType = matrixType (.constant 1) (.constant 2))
    | _ => false) &&
  verifyOperationKind workflow body.special (fun kind => match kind with
    | .concat .rows => true
    | _ => false) &&
  verifyOperationKind workflow body.stateMatch (fun kind => match kind with
    | .intCompare .equal => true
    | _ => false) &&
  verifyOperationKind workflow body.stateMatchToInt (fun kind => match kind with
    | .boolToInt => true
    | _ => false) &&
  verifySelectOperation workflow body.selector &&
  [body.bitExtract, body.bitToInt, body.bitZero, body.bitOne, body.bitSelect,
    body.specialProduct, body.specialTop, body.specialBottom, body.special,
    body.stateMatch, body.stateMatchToInt, body.selector].all
      (fun operation => decide (operation.operation.scope = reference.bitScan.bodyScope)) &&
  decide (reference.regular.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 6, localWire 6]) &&
  decide (reference.regular.outputs.map wireRef = [localWire 7]) &&
  decide reference.kIdentity.inputs.isEmpty &&
  decide (reference.kIdentity.outputs.map wireRef = [localWire 8]) &&
  decide (reference.k.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 6, localWire 8]) &&
  decide (reference.k.outputs.map wireRef = [localWire 9]) &&
  decide (reference.initialSelect.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 5, localWire 7, localWire 9]) &&
  decide (reference.initialSelect.outputs.map wireRef = [localWire 10]) &&
  decide (reference.bitScan.count = batchBits) &&
  decide (reference.bitScan.indexSlot = 1) &&
  decide reference.bitScan.bindings.isEmpty &&
  decide (reference.bitScan.carriedCount = 1) &&
  decide (reference.bitScan.arguments.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 10, localWire 14, localWire 2, localWire 20, localWire 6]) &&
  decide (reference.bitScan.bodyOutputs.map wireRef = [localWire 18]) &&
  decide (reference.bitScan.outputs.map wireRef = [localWire 21]) &&
  decide (body.bitExtract.inputs.map (wireRef ∘ CoreOperandRef.wire) = [localWire 8]) &&
  decide (body.bitExtract.outputs.map wireRef = [localWire 9]) &&
  decide (body.bitToInt.inputs.map (wireRef ∘ CoreOperandRef.wire) = [localWire 9]) &&
  decide (body.bitToInt.outputs.map wireRef = [localWire 10]) &&
  decide body.bitZero.inputs.isEmpty && decide (body.bitZero.outputs.map wireRef = [localWire 11]) &&
  decide body.bitOne.inputs.isEmpty && decide (body.bitOne.outputs.map wireRef = [localWire 12]) &&
  decide (body.bitSelect.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 10, localWire 11, localWire 12]) &&
  decide (body.bitSelect.outputs.map wireRef = [localWire 13]) &&
  decide (body.specialProduct.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 7, localWire 13]) &&
  decide (body.specialProduct.outputs.map wireRef = [localWire 14]) &&
  decide (body.specialTop.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 7, localWire 14]) &&
  decide (body.specialTop.outputs.map wireRef = [localWire 15]) &&
  decide body.specialBottom.inputs.isEmpty &&
  decide (body.specialBottom.outputs.map wireRef = [localWire 16]) &&
  decide (body.special.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 15, localWire 16]) &&
  decide (body.special.outputs.map wireRef = [localWire 17]) &&
  decide (body.stateMatch.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 0, localWire 3]) &&
  decide (body.stateMatch.outputs.map wireRef = [localWire 4]) &&
  decide (body.stateMatchToInt.inputs.map (wireRef ∘ CoreOperandRef.wire) = [localWire 4]) &&
  decide (body.stateMatchToInt.outputs.map wireRef = [localWire 5]) &&
  decide (body.selector.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 5, localWire 6, localWire 17]) &&
  decide (body.selector.outputs.map wireRef = [localWire 18]) &&
  verifyTransitionSelectorBitBodyInputNames workflow reference &&
  verifyTransitionSelectorBitBody workflow reference

def verifyParallelTransitionTarget (workflow : Mxx.Ir.Workflow)
    (reference : ParallelTransitionTargetRef) : Bool :=
  verifyParallelLoop workflow reference.parallelLoop &&
    verifyWire workflow reference.body.digitSecret &&
    verifyWire workflow reference.body.targetPublic &&
    verifyWire workflow reference.body.selector &&
  verifyTransitionSelector workflow reference.body.selectorConstruction &&
  verifyExactParallelBody workflow reference.parallelLoop
    (transitionTargetNodes reference.body.selectorConstruction.bitScan.bodyScope.definitionName) &&
  verifyOperationKind workflow reference.body.errorSample (fun kind => match kind with
      | .gaussianSample matrixType cutoff =>
          decide (matrixType = transitionErrorMatrixType) && decide (cutoff = errorBound)
      | _ => false) &&
    verifyOperationKind workflow reference.body.selectorProduct (fun kind => match kind with
      | .matrixMultiply => true
      | _ => false) &&
    verifyOperationKind workflow reference.body.targetSum (fun kind => match kind with
      | .matrixAdd => true
      | _ => false) &&
    decide (reference.parallelLoop.bodyInputs.take 2 =
      [reference.body.digitSecret, reference.body.targetPublic]) &&
    decide (reference.body.selectorConstruction.bitScan.outputs =
      [reference.body.selector]) &&
    decide (reference.body.selectorProduct.inputs.map (·.wire) =
      [reference.body.selector, reference.body.targetPublic]) &&
    decide reference.body.errorSample.inputs.isEmpty &&
    decide (reference.body.errorSample.outputs.length = 1) &&
    decide (reference.body.selectorProduct.outputs.length = 1) &&
    decide (reference.body.targetSum.inputs.map (·.wire) =
      reference.body.selectorProduct.outputs ++ reference.body.errorSample.outputs) &&
    decide (reference.body.targetSum.outputs.length = 1) &&
    decide (reference.body.targetSum.outputs = reference.parallelLoop.bodyOutputs) &&
    decide (reference.parallelLoop.outputs.length = 1)

def verifyStaticFamilyGet (workflow : Mxx.Ir.Workflow) (source output : CoreWireRef)
    (index : Int) : Bool :=
  verifyWire workflow source && verifyWire workflow output && decide (output.port = 0) &&
    match resolveNode workflow output.node with
    | some { kind := .familyGetStatic (.constant actual), arguments, outputCount } =>
        decide (actual = index) && decide (arguments = [wireRef source]) &&
          decide (outputCount = 1)
    | _ => false

def verifyInputPreprocessing (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (layout : DiamondInputPreprocessingLayout) : Bool :=
  [layout.trapdoorSamples.parallelLoop.operation, layout.secretSample.operation,
    layout.messageSelector.operation, layout.initialErrorSample.operation,
    layout.initialPublicProduct.operation, layout.initialState.operation,
    layout.transitionSourceIndices.parallelLoop.operation,
    layout.transitionTargetIndices.parallelLoop.operation,
    layout.digitSecretIndices.parallelLoop.operation,
    layout.digitSecretSamples.parallelLoop.operation,
    layout.digitSecrets.parallelLoop.operation, layout.transitionSources.parallelLoop.operation,
    layout.targetPublicMatrices.parallelLoop.operation,
    layout.transitionTargets.parallelLoop.operation,
    layout.transitionPreimages.parallelLoop.operation, layout.finalIndices.operation,
    layout.finalTrapdoors.parallelLoop.operation].all
      (fun operation => decide (operation.stage = workflowLayout.encryption.stage) &&
        decide (operation.scope = .root)) &&
    verifyArtifactInLayout workflow workflowLayout layout.initialStateArtifact &&
    verifyArtifactInLayout workflow workflowLayout layout.transitionsArtifact &&
    decide (layout.initialStateArtifact ∈ workflowLayout.artifacts) &&
    decide (layout.transitionsArtifact ∈ workflowLayout.artifacts) &&
    decide (layout.initialStateArtifact.producerStage = workflowLayout.encryption.stage) &&
    decide (layout.transitionsArtifact.producerStage = workflowLayout.encryption.stage) &&
    verifyParallelOperation workflow layout.trapdoorSamples (fun kind => match kind with
      | .trapdoorSample matrixType cutoff =>
          decide (matrixType = trapdoorMatrixType) && decide (cutoff = preimageBound)
      | _ => false) &&
    verifyOperationKind workflow layout.secretSample (fun kind => match kind with
      | .uniformSample matrixType (.constant (-1)) (.constant 1) =>
          decide (matrixType = unitMatrixType)
      | _ => false) &&
    verifyOperationKind workflow layout.messageSelector (fun kind => match kind with
      | .concat .columns => true
      | _ => false) &&
    verifyOperationKind workflow layout.initialErrorSample (fun kind => match kind with
      | .gaussianSample matrixType cutoff =>
          decide (matrixType = initialErrorMatrixType) && decide (cutoff = errorBound)
      | _ => false) &&
    verifyOperationKind workflow layout.initialPublicProduct (fun kind => match kind with
      | .matrixMultiply => true
      | _ => false) &&
    verifyOperationKind workflow layout.initialState (fun kind => match kind with
      | .matrixAdd => true
      | _ => false) &&
    verifyPreprocessingSourceIndexFormula workflow layout.transitionSourceIndices &&
    verifyPreprocessingTargetIndexFormula workflow layout.transitionTargetIndices &&
    verifyPreprocessingDigitSecretIndexFormula workflow layout.digitSecretIndices &&
    verifyParallelOperation workflow layout.digitSecretSamples (fun kind => match kind with
      | .uniformSample matrixType (.constant (-1)) (.constant 1) =>
          decide (matrixType = unitMatrixType)
      | _ => false) &&
    verifyParallelGather workflow layout.digitSecrets &&
    verifyParallelGather workflow layout.transitionSources &&
    verifyParallelGather workflow layout.targetPublicMatrices &&
    verifyParallelTransitionTarget workflow layout.transitionTargets &&
    verifyParallelPreimage workflow layout.transitionPreimages stateColumns &&
    verifyParallelLoop workflow layout.finalIndices &&
    verifyParallelGather workflow layout.finalTrapdoors &&
    decide (layout.trapdoorSamples.parallelLoop.count = trapdoorBaseCount) &&
    decide (layout.trapdoorSamples.parallelLoop.indexSlot = 0) &&
    decide layout.trapdoorSamples.parallelLoop.bindings.isEmpty &&
    decide layout.trapdoorSamples.parallelLoop.inputModes.isEmpty &&
    decide (layout.trapdoorSamples.parallelLoop.bodyOutputs.length = 2) &&
    decide (layout.trapdoorSamples.parallelLoop.outputs.length = 2) &&
    verifyExactParallelLoopRole workflow layout.digitSecretSamples.parallelLoop
      (.multiply inputCount digitBase) 0 [] &&
    verifyExactParallelLoopRole workflow layout.digitSecrets.parallelLoop
      transitionCount 0 [.zip, .broadcast] &&
    decide (layout.transitionSources.parallelLoop.count = transitionCount) &&
    decide (layout.transitionSources.parallelLoop.indexSlot = 0) &&
    decide layout.transitionSources.parallelLoop.bindings.isEmpty &&
    decide (layout.transitionSources.parallelLoop.inputModes =
      [.zip, .broadcast, .broadcast]) &&
    decide (layout.transitionSources.parallelLoop.bodyOutputs.length = 2) &&
    decide (layout.transitionSources.parallelLoop.outputs.length = 2) &&
    verifyExactParallelLoopRole workflow layout.targetPublicMatrices.parallelLoop
      transitionCount 0 [.zip, .broadcast] &&
    verifyExactParallelNodeRole workflow layout.transitionTargets.parallelLoop.operation
      transitionCount 0 [.zip, .zip] &&
    verifyExactParallelLoopRole workflow layout.transitionPreimages.parallelLoop
      transitionCount 0 [.zip, .zip, .zip] &&
    decide (layout.finalIndices.count = stateWidth) &&
    decide (layout.finalIndices.indexSlot = 0) &&
    decide layout.finalIndices.bindings.isEmpty &&
    decide layout.finalIndices.inputModes.isEmpty &&
    decide (layout.finalIndices.bodyOutputs.length = 1) &&
    decide (layout.finalIndices.outputs.length = 1) &&
    decide (layout.finalTrapdoors.parallelLoop.count = stateWidth) &&
    decide (layout.finalTrapdoors.parallelLoop.indexSlot = 0) &&
    decide layout.finalTrapdoors.parallelLoop.bindings.isEmpty &&
    decide (layout.finalTrapdoors.parallelLoop.inputModes =
      [.zip, .broadcast, .broadcast]) &&
    decide (layout.finalTrapdoors.parallelLoop.bodyOutputs.length = 2) &&
    decide (layout.finalTrapdoors.parallelLoop.outputs.length = 2) &&
    decide (layout.trapdoorSamples.body.inputs.isEmpty) &&
    decide (layout.trapdoorSamples.body.outputs.length = 2) &&
    decide (layout.secretSample.inputs.isEmpty) &&
    decide (layout.secretSample.outputs.length = 1) &&
    decide (layout.messageSelector.inputs.length = 2) &&
    decide ((layout.messageSelector.inputs.take 1).map (·.wire) =
      layout.secretSample.outputs) &&
    decide (layout.messageSelector.outputs.length = 1) &&
    decide (layout.initialErrorSample.inputs.isEmpty) &&
    decide (layout.initialErrorSample.outputs.length = 1) &&
    decide (layout.initialPublicProduct.inputs.length = 2) &&
    decide ((layout.initialPublicProduct.inputs.take 1).map (·.wire) =
      layout.messageSelector.outputs) &&
    (match layout.trapdoorSamples.parallelLoop.outputs,
        layout.initialPublicProduct.inputs.drop 1 with
    | [publicFamily, _], [basePublic] =>
        verifyStaticFamilyGet workflow publicFamily basePublic.wire 0
    | _, _ => false) &&
    decide (layout.initialPublicProduct.outputs.length = 1) &&
    decide (layout.initialState.inputs.map (·.wire) =
      layout.initialPublicProduct.outputs ++ layout.initialErrorSample.outputs) &&
    decide (layout.initialState.outputs.length = 1) &&
    decide ([layout.initialStateArtifact.producerOutput.wire] = layout.initialState.outputs) &&
    decide ([layout.digitSecrets.indexFamily.wire] =
      layout.digitSecretIndices.parallelLoop.outputs) &&
    decide (layout.digitSecrets.sourceFamilies.map (·.wire) =
      layout.digitSecretSamples.parallelLoop.outputs) &&
    decide ([layout.transitionSources.indexFamily.wire] =
      layout.transitionSourceIndices.parallelLoop.outputs) &&
    decide (layout.transitionSources.sourceFamilies.map (·.wire) =
      layout.trapdoorSamples.parallelLoop.outputs) &&
    decide ([layout.targetPublicMatrices.indexFamily.wire] =
      layout.transitionTargetIndices.parallelLoop.outputs) &&
    decide (layout.targetPublicMatrices.sourceFamilies.map (·.wire) =
      layout.trapdoorSamples.parallelLoop.outputs.take 1) &&
    decide ((layout.transitionTargets.parallelLoop.arguments.take 2).map (·.wire) =
      layout.digitSecrets.outputFamilies ++ layout.targetPublicMatrices.outputFamilies) &&
    decide (layout.transitionPreimages.parallelLoop.arguments.map (·.wire) =
      layout.transitionSources.outputFamilies ++ layout.transitionTargets.parallelLoop.outputs) &&
    decide (layout.transitionPreimages.body.sample.inputs.map (·.wire) =
      layout.transitionPreimages.parallelLoop.bodyInputs) &&
    decide ([layout.transitionsArtifact.producerOutput.wire] =
      layout.transitionPreimages.parallelLoop.outputs) &&
    decide ([layout.finalTrapdoors.indexFamily.wire] = layout.finalIndices.outputs) &&
    decide (layout.finalTrapdoors.sourceFamilies.map (·.wire) =
      layout.trapdoorSamples.parallelLoop.outputs)

/-- The checked trapdoor-sampling operation includes a checked enclosing parallel loop. -/
theorem verifyInputPreprocessing_trapdoorSamplesLoop
    {workflow : Mxx.Ir.Workflow} {workflowLayout : DiamondWorkflowLayout}
    {layout : DiamondInputPreprocessingLayout}
    (verified : verifyInputPreprocessing workflow workflowLayout layout = true) :
    verifyParallelLoop workflow layout.trapdoorSamples.parallelLoop = true := by
  unfold verifyInputPreprocessing at verified
  simp only [Bool.and_eq_true] at verified
  have checked : verifyParallelOperation workflow layout.trapdoorSamples (fun kind =>
      match kind with
      | .trapdoorSample matrixType cutoff =>
          decide (matrixType = trapdoorMatrixType) && decide (cutoff = preimageBound)
      | _ => false) = true := by
    aesop
  unfold verifyParallelOperation at checked
  simp only [Bool.and_eq_true] at checked
  exact checked.1.1.1.1.1

/-- The checked digit-secret sampling operation includes a checked enclosing parallel loop. -/
theorem verifyInputPreprocessing_digitSecretSamplesLoop
    {workflow : Mxx.Ir.Workflow} {workflowLayout : DiamondWorkflowLayout}
    {layout : DiamondInputPreprocessingLayout}
    (verified : verifyInputPreprocessing workflow workflowLayout layout = true) :
    verifyParallelLoop workflow layout.digitSecretSamples.parallelLoop = true := by
  unfold verifyInputPreprocessing at verified
  simp only [Bool.and_eq_true] at verified
  have checked : verifyParallelOperation workflow layout.digitSecretSamples (fun kind =>
      match kind with
      | .uniformSample matrixType (.constant (-1)) (.constant 1) =>
          decide (matrixType = unitMatrixType)
      | _ => false) = true := by
    aesop
  unfold verifyParallelOperation at checked
  simp only [Bool.and_eq_true] at checked
  exact checked.1.1.1.1.1

def verifyMessageConstruction (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (layout : MessageConstructionLayout) : Bool :=
  [layout.toInt.operation, layout.zero.operation, layout.one.operation,
    layout.select.operation].all
      (fun operation => decide (operation.stage = workflowLayout.encryption.stage) &&
        decide (operation.scope = .root)) &&
  verifyOperationKind workflow layout.toInt (fun kind => match kind with
    | .boolToInt => true
    | _ => false) &&
  (match layout.toInt.inputs with
  | [message] => verifyInputWire workflow message.wire workflowLayout.encryption.stage
      "diamond-message"
  | _ => false) &&
  verifyOperationKind workflow layout.zero (fun kind => match kind with
    | .zeroMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifyOperationKind workflow layout.one (fun kind => match kind with
    | .identityMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifySelectOperation workflow layout.select &&
  decide (layout.select.inputs[0]?.map (·.wire) = layout.toInt.outputs[0]?) &&
  decide (layout.select.inputs[1]?.map (·.wire) = layout.zero.outputs[0]?) &&
  decide (layout.select.inputs[2]?.map (·.wire) = layout.one.outputs[0]?)

/-- Producer-node checks exposed as one shallow final conjunct for execution lifting. -/
def verifyPublicKeySamplingProducerNodes (workflow : Mxx.Ir.Workflow)
    (layout : BggPublicKeySamplingLayout) : Bool :=
  verifyOperation workflow layout.packedHash &&
    verifyParallelLoop workflow layout.slices.parallelLoop

def verifyPublicKeySampling (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (layout : BggPublicKeySamplingLayout) : Bool :=
  [layout.packedHash.operation, layout.slices.parallelLoop.operation].all (fun operation =>
      decide (operation.stage = workflowLayout.encryption.stage) &&
        decide (operation.scope = .root)) &&
  verifyArtifactInLayout workflow workflowLayout layout.publicKeysArtifact &&
  decide (layout.publicKeysArtifact ∈ workflowLayout.artifacts) &&
  verifyOperationKind workflow layout.packedHash (fun kind => match kind with
    | .hashSample actualType .plain tag [] [] [] none none =>
        decide (actualType =
          matrixType (.constant 1) (.multiply digitCount publicKeyCount)) &&
          decide (tag = witnessPublicKeyTag)
    | _ => false) &&
  verifyParallelOperation workflow layout.slices (fun kind => match kind with
    | .slice none (some (start, stop)) =>
        decide (start = .multiply digitCount (.loopIndex 0)) &&
          decide (stop = .add (.multiply digitCount (.loopIndex 0)) digitCount)
    | _ => false) &&
  verifyExactParallelLoopRole workflow layout.slices.parallelLoop publicKeyCount 0
    [.broadcast] &&
  decide (layout.packedHash.inputs.length = 1) &&
  (match layout.packedHash.inputs with
  | [hashKey] => verifyInputWire workflow hashKey.wire workflowLayout.encryption.stage
      "diamond-hash-key"
  | _ => false) &&
  decide (layout.packedHash.outputs.length = 1) &&
  decide (layout.slices.parallelLoop.arguments.length = 1) &&
  decide (layout.slices.parallelLoop.arguments[0]?.map (·.wire) =
    layout.packedHash.outputs[0]?) &&
  decide (layout.slices.body.inputs.length = 1) &&
  decide (layout.slices.body.inputs[0]?.map (·.wire) =
    layout.slices.parallelLoop.bodyInputs[0]?) &&
  decide (some layout.publicKeysArtifact.producerOutput.wire =
    layout.slices.parallelLoop.outputs[0]?) &&
  verifyPublicKeySamplingProducerNodes workflow layout

def verifyParallelSelectChain (workflow : Mxx.Ir.Workflow)
    (parallelLoop : ParallelLoopRef) (operations : List OperationRef) : Bool :=
  verifyParallelLoop workflow parallelLoop && decide (!operations.isEmpty) &&
    operations.all (fun operation =>
      verifySelectOperation workflow operation &&
        decide (operation.operation.scope = parallelLoop.bodyScope)) &&
    decide (operations.getLast?.map (·.outputs) = some parallelLoop.bodyOutputs) &&
    decide (parallelLoop.outputs.length = 1)

def verifyEncryptionInitialPublicKeys (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (sampling : BggPublicKeySamplingLayout)
    (layout : EncryptionInitialPublicKeysLayout) : Bool :=
  [layout.onePublicKey.operation, layout.zeroPublicKey.operation,
    layout.instanceWidth.operation,
    layout.publicIndices.operation, layout.publicCandidates.parallelLoop.operation,
    layout.packedInputs.parallelLoop.operation,
    layout.circuitInputs.parallelLoop.operation].all
      (fun operation => decide (operation.stage = workflowLayout.encryption.stage)) &&
  verifyOperationKind workflow layout.onePublicKey (fun kind => match kind with
    | .familyGetStatic (.constant 0) => true
    | _ => false) &&
  verifyOperationKind workflow layout.zeroPublicKey (fun kind => match kind with
    | .matrixSubtract => true
    | _ => false) &&
  verifyEvaluateInt workflow layout.instanceWidth &&
  verifyParallelLoop workflow layout.publicIndices &&
  verifyParallelGather workflow layout.publicCandidates &&
  verifyParallelSelectChain workflow layout.packedInputs.parallelLoop
    [layout.packedInputs.inRange, layout.packedInputs.padded] &&
  verifyParallelSelectChain workflow layout.circuitInputs.parallelLoop
    [layout.circuitInputs.selectedInstance, layout.circuitInputs.selectedSource] &&
  verifyExactParallelLoopRole workflow layout.publicIndices
    (.parameter "max_layer_width") 0 [.broadcast, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.publicCandidates.parallelLoop
    (.parameter "max_layer_width") 0 [.zip, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.packedInputs.parallelLoop
    (.parameter "max_layer_width") 0 [.zip, .broadcast, .broadcast, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.circuitInputs.parallelLoop
    (.parameter "max_layer_width") 0 [.zip, .zip, .broadcast, .broadcast, .broadcast] &&
  verifyExactParallelBody workflow layout.publicIndices encryptionPublicIndexNodes &&
  verifyExactParallelBody workflow layout.packedInputs.parallelLoop
    encryptionPackedInputNodes &&
  verifyExactParallelBody workflow layout.circuitInputs.parallelLoop
    encryptionCircuitInputNodes &&
  decide (layout.packedInputs.inRange.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 7, localWire 4, localWire 8]) &&
  decide (layout.packedInputs.inRange.outputs.map wireRef = [localWire 9]) &&
  decide (layout.packedInputs.padded.inputs.map (wireRef ∘ CoreOperandRef.wire) =
    [localWire 3, localWire 4, localWire 9]) &&
  decide (layout.packedInputs.padded.outputs.map wireRef = [localWire 10]) &&
  decide (layout.circuitInputs.selectedInstance.inputs.map
    (wireRef ∘ CoreOperandRef.wire) = [localWire 7, localWire 8, localWire 9]) &&
  decide (layout.circuitInputs.selectedInstance.outputs.map wireRef = [localWire 10]) &&
  decide (layout.circuitInputs.selectedSource.inputs.map
    (wireRef ∘ CoreOperandRef.wire) = [localWire 5, localWire 6, localWire 10]) &&
  decide (layout.circuitInputs.selectedSource.outputs.map wireRef = [localWire 11]) &&
  decide (layout.onePublicKey.inputs.length = 1) &&
  decide (layout.onePublicKey.inputs[0]?.map (·.wire) =
    sampling.slices.parallelLoop.outputs[0]?) &&
  decide (layout.onePublicKey.outputs.length = 1) &&
  decide (layout.zeroPublicKey.inputs.length = 2) &&
  decide (layout.zeroPublicKey.inputs.map (·.wire) =
    layout.onePublicKey.outputs ++ layout.onePublicKey.outputs) &&
  decide (layout.publicCandidates.indexFamily.wire ∈ layout.publicIndices.outputs) &&
  decide (layout.publicCandidates.sourceFamilies.length = 1) &&
  decide (layout.publicCandidates.sourceFamilies[0]?.map (·.wire) =
    sampling.slices.parallelLoop.outputs[0]?) &&
  decide (layout.packedInputs.parallelLoop.arguments[0]?.map (·.wire) =
    layout.publicCandidates.outputFamilies[0]?) &&
  decide (layout.packedInputs.padded.inputs[2]?.map (·.wire) =
    layout.packedInputs.inRange.outputs[0]?) &&
  decide (layout.circuitInputs.selectedSource.inputs[1]?.map (·.wire) =
    layout.circuitInputs.parallelLoop.bodyInputs[1]?) &&
  decide (layout.circuitInputs.selectedSource.inputs[2]?.map (·.wire) =
    layout.circuitInputs.selectedInstance.outputs[0]?)

def verifyStaticTrapdoor (workflow : Mxx.Ir.Workflow)
    (layout : StaticTrapdoorLayout) : Bool :=
  [layout.publicOperation, layout.secret].all (fun operation =>
    verifyOperationKind workflow operation (fun kind => match kind with
      | .familyGetStatic (.constant 0) => true
      | _ => false) && decide (operation.inputs.length = 1) &&
      decide (operation.outputs.length = 1))

def verifyParallelWitnessTarget (workflow : Mxx.Ir.Workflow)
    (gadget : CoreWireRef) (layout : ParallelWitnessTargetLayout) : Bool :=
  verifyParallelLoop workflow layout.parallelLoop &&
  verifyOperationKind workflow layout.negatedGadget (fun kind => match kind with
    | .matrixNegate => true
    | _ => false) &&
  verifyOperationKind workflow layout.target (fun kind => match kind with
    | .concat .rows => true
    | _ => false) &&
  decide (layout.negatedGadget.operation.scope = layout.parallelLoop.bodyScope) &&
  decide (layout.target.operation.scope = layout.parallelLoop.bodyScope) &&
  decide (layout.parallelLoop.arguments.length = 2) &&
  decide (layout.parallelLoop.arguments[1]?.map (·.wire) = some gadget) &&
  decide (layout.parallelLoop.bodyInputs.length = 2) &&
  decide (layout.negatedGadget.inputs.map (·.wire) =
    layout.parallelLoop.bodyInputs.drop 1) &&
  decide (layout.negatedGadget.outputs.length = 1) &&
  decide (layout.target.inputs.map (·.wire) =
    layout.parallelLoop.bodyInputs.take 1 ++ layout.negatedGadget.outputs) &&
  decide (layout.target.inputs[1]?.map (·.wire) = layout.negatedGadget.outputs[0]?) &&
  decide (layout.target.outputs.length = 1) &&
  decide (layout.parallelLoop.bodyOutputs = layout.target.outputs) &&
  decide (layout.parallelLoop.outputs.length = 1)

/-- Artifact producer-node checks exposed as one shallow final conjunct for execution lifting. -/
def verifyArtifactPreprocessingProducerNodes (workflow : Mxx.Ir.Workflow)
    (layout : DiamondArtifactPreprocessingLayout) : Bool :=
  verifyParallelLoop workflow layout.witnessPreimages.parallelLoop &&
    verifyOperation workflow layout.onePreimage.sample &&
    verifyOperation workflow layout.onePreimage.materialize &&
    verifyOperation workflow layout.kPreimage.sample &&
    verifyOperation workflow layout.kPreimage.materialize &&
    verifyOperation workflow layout.rDecomposition &&
    verifyOperation workflow layout.rMaterialization &&
    verifyOperation workflow layout.rReshape &&
    verifyOperation workflow layout.decoderPreimage.sample &&
    verifyOperation workflow layout.decoderPreimage.materialize

def verifyArtifactPreprocessing (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout)
    (inputPreprocessing : DiamondInputPreprocessingLayout)
    (sampling : BggPublicKeySamplingLayout) (booleanLayers : BooleanLayersLayout)
    (layout : DiamondArtifactPreprocessingLayout) : Bool :=
  [layout.onePreimage.sample.operation, layout.onePreimage.materialize.operation,
    layout.witnessPreimages.parallelLoop.operation, layout.kPreimage.sample.operation,
    layout.kPreimage.materialize.operation, layout.rDecomposition.operation,
    layout.rMaterialization.operation, layout.rReshape.operation,
    layout.decoderPreimage.sample.operation, layout.decoderPreimage.materialize.operation].all
      (fun operation => decide (operation.stage = workflowLayout.encryption.stage) &&
        decide (operation.scope = .root)) &&
  [layout.onePreimageArtifact, layout.witnessPreimagesArtifact,
    layout.kPreimageArtifact, layout.rDecomposedArtifact,
    layout.decoderPreimageArtifact].all (fun artifact =>
      verifyArtifactInLayout workflow workflowLayout artifact &&
        decide (artifact ∈ workflowLayout.artifacts) &&
        decide (artifact.producerStage = workflowLayout.encryption.stage)) &&
  verifyStaticTrapdoor workflow layout.projectionTrapdoor &&
  verifyOperationKind workflow layout.oneTarget.gadget (fun kind => match kind with
    | .gadgetMatrix actualType base =>
        decide (actualType = matrixType (.constant 1) digitCount) &&
          decide (base = .parameter "diamond_gadget_base")
    | _ => false) &&
  verifyOperationKind workflow layout.oneTarget.difference (fun kind => match kind with
    | .matrixSubtract => true
    | _ => false) &&
  verifyOperationKind workflow layout.oneTarget.zeroRow (fun kind => match kind with
    | .zeroMatrix actualType => decide (actualType = matrixType (.constant 1) digitCount)
    | _ => false) &&
  verifyOperationKind workflow layout.oneTarget.target (fun kind => match kind with
    | .concat .rows => true
    | _ => false) &&
  verifyPreimage workflow layout.onePreimage digitCount &&
  verifyParallelLoop workflow layout.witnessIndices &&
  verifyParallelGather workflow layout.witnessTrapdoors &&
  verifyParallelGather workflow layout.witnessPublicKeys &&
  (match layout.oneTarget.gadget.outputs with
  | [gadget] => verifyParallelWitnessTarget workflow gadget layout.witnessTargets
  | _ => false) &&
  verifyParallelPreimage workflow layout.witnessPreimages digitCount &&
  verifyExactParallelLoopRole workflow layout.witnessIndices witnessSize 0 [] &&
  decide (layout.witnessTrapdoors.parallelLoop.count = witnessSize) &&
  decide (layout.witnessTrapdoors.parallelLoop.indexSlot = 0) &&
  decide layout.witnessTrapdoors.parallelLoop.bindings.isEmpty &&
  decide (layout.witnessTrapdoors.parallelLoop.inputModes =
    [.zip, .broadcast, .broadcast]) &&
  decide (layout.witnessTrapdoors.parallelLoop.bodyOutputs.length = 2) &&
  decide (layout.witnessTrapdoors.parallelLoop.outputs.length = 2) &&
  verifyExactParallelLoopRole workflow layout.witnessPublicKeys.parallelLoop witnessSize 0
    [.zip, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.witnessTargets.parallelLoop witnessSize 0
    [.zip, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.witnessPreimages.parallelLoop witnessSize 0
    [.zip, .zip, .zip] &&
  verifyOperationKind workflow layout.kTarget.publicKeyHash (fun kind => match kind with
    | .hashSample actualType .plain tag [] [] [] none none =>
        decide (actualType = matrixType (.constant 1) digitCount) &&
          decide (tag = kPublicKeyTag)
    | _ => false) &&
  verifyOperationKind workflow layout.kTarget.firstColumn (fun kind => match kind with
    | .slice none (some ((.constant 0), (.constant 1))) => true
    | _ => false) &&
  verifyOperationKind workflow layout.kTarget.halfModulus (fun kind => match kind with
    | .constantMatrix actualType [coefficient] =>
          decide (actualType = unitMatrixType) &&
          decide (coefficient =
            .roundDivide modulus (.constant 2))
    | _ => false) &&
  verifyOperationKind workflow layout.kTarget.target (fun kind => match kind with
    | .concat .rows => true
    | _ => false) &&
  verifyPreimage workflow layout.kPreimage (.constant 1) &&
  verifyOperationKind workflow layout.rHash (fun kind => match kind with
    | .hashSample actualType .plain tag [] [] [] none none =>
        decide (actualType = matrixType (.constant 1) digitCount) && decide (tag = rTag)
    | _ => false) &&
  verifyOperationKind workflow layout.rSlice (fun kind => match kind with
    | .slice none (some ((.constant 0), (.constant 1))) => true
    | _ => false) &&
  verifyOperationKind workflow layout.rDecomposition (fun kind => match kind with
    | .gadgetDecompose actualType base count =>
        decide (actualType = matrixType (.multiply (.constant 1) digitCount) (.constant 1)) &&
          decide (base = .parameter "diamond_gadget_base") && decide (count = digitCount)
    | _ => false) &&
  verifyOperationKind workflow layout.rMaterialization (fun kind => match kind with
    | .matrixScale (.constant 1) => true
    | _ => false) &&
  verifyOperationKind workflow layout.rReshape (fun kind => match kind with
    | .reshape rows (.constant 1) => decide (rows = digitCount)
    | _ => false) &&
  verifyOperationKind workflow layout.decoderTarget.publicKeyDifference (fun kind => match kind with
    | .matrixSubtract => true
    | _ => false) &&
  verifyOperationKind workflow layout.decoderTarget.projectedDifference (fun kind => match kind with
    | .matrixMultiply => true
    | _ => false) &&
  verifyOperationKind workflow layout.decoderTarget.publicKeySum (fun kind => match kind with
    | .matrixAdd => true
    | _ => false) &&
  verifyOperationKind workflow layout.decoderTarget.zero (fun kind => match kind with
    | .zeroMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  verifyOperationKind workflow layout.decoderTarget.target (fun kind => match kind with
    | .concat .rows => true
    | _ => false) &&
  verifyPreimage workflow layout.decoderPreimage (.constant 1) &&
  [layout.kTarget.publicKeyHash, layout.kTarget.firstColumn, layout.rHash,
    layout.rSlice].all (fun operation =>
      decide (operation.inputs.length = 1) && decide (operation.outputs.length = 1)) &&
  decide layout.oneTarget.gadget.inputs.isEmpty &&
  decide (layout.oneTarget.gadget.outputs.length = 1) &&
  decide (layout.oneTarget.difference.inputs.length = 2) &&
  decide (layout.oneTarget.difference.outputs.length = 1) &&
  decide (layout.oneTarget.difference.inputs[1]?.map (·.wire) =
    layout.oneTarget.gadget.outputs[0]?) &&
  (match sampling.slices.parallelLoop.outputs, layout.oneTarget.difference.inputs with
  | [publicKeys], [onePublicKey, _] =>
      verifyStaticFamilyGet workflow publicKeys onePublicKey.wire 0
  | _, _ => false) &&
  decide (layout.oneTarget.target.inputs.map (·.wire) =
    layout.oneTarget.difference.outputs ++ layout.oneTarget.zeroRow.outputs) &&
  decide (layout.oneTarget.target.outputs.length = 1) &&
  decide (layout.witnessTargets.parallelLoop.arguments.map (·.wire) =
    layout.witnessPublicKeys.outputFamilies ++ layout.oneTarget.gadget.outputs) &&
  decide (layout.kTarget.target.inputs.map (·.wire) =
    layout.kTarget.firstColumn.outputs ++ layout.kTarget.halfModulus.outputs) &&
  decide (layout.kTarget.target.outputs.length = 1) &&
  decide (layout.rDecomposition.inputs.map (·.wire) = layout.rSlice.outputs) &&
  decide (layout.rDecomposition.outputs.length = 1) &&
  decide (layout.rMaterialization.inputs.map (·.wire) = layout.rDecomposition.outputs) &&
  decide (layout.rMaterialization.outputs.length = 1) &&
  decide (layout.rReshape.inputs.map (·.wire) = layout.rMaterialization.outputs) &&
  decide (layout.rReshape.outputs.length = 1) &&
  decide (layout.decoderTarget.publicKeyDifference.inputs.length = 2) &&
  decide (layout.decoderTarget.publicKeyDifference.outputs.length = 1) &&
  (match sampling.slices.parallelLoop.outputs,
      layout.decoderTarget.publicKeyDifference.inputs with
  | [publicKeys], [onePublicKey, selectedCircuit] =>
      verifyStaticFamilyGet workflow publicKeys onePublicKey.wire 0 &&
        decide (selectedCircuit.wire = booleanLayers.encryption.selectedOutput.output)
  | _, _ => false) &&
  decide (layout.decoderTarget.projectedDifference.inputs.map (·.wire) =
    layout.decoderTarget.publicKeyDifference.outputs ++ layout.rReshape.outputs) &&
  decide (layout.decoderTarget.projectedDifference.outputs.length = 1) &&
  decide (layout.decoderTarget.publicKeySum.inputs.map (·.wire) =
    layout.kTarget.firstColumn.outputs ++ layout.decoderTarget.projectedDifference.outputs) &&
  decide (layout.decoderTarget.publicKeySum.outputs.length = 1) &&
  decide (layout.decoderTarget.target.inputs.map (·.wire) =
    layout.decoderTarget.publicKeySum.outputs ++ layout.decoderTarget.zero.outputs) &&
  decide (layout.decoderTarget.target.outputs.length = 1) &&
  [layout.oneTarget.zeroRow, layout.kTarget.halfModulus,
    layout.decoderTarget.zero].all (fun operation =>
      decide operation.inputs.isEmpty && decide (operation.outputs.length = 1)) &&
  decide (layout.rMaterialization.inputs.length = 1) &&
  decide (layout.rMaterialization.outputs.length = 1) &&
  decide (layout.projectionTrapdoor.publicOperation.inputs[0]?.map (·.wire) =
    inputPreprocessing.finalTrapdoors.outputFamilies[0]?) &&
  decide (layout.projectionTrapdoor.secret.inputs[0]?.map (·.wire) =
    inputPreprocessing.finalTrapdoors.outputFamilies[1]?) &&
  [layout.onePreimage.sample, layout.kPreimage.sample,
    layout.decoderPreimage.sample].all (fun sample =>
      decide (sample.inputs[0]?.map (·.wire) =
        layout.projectionTrapdoor.publicOperation.outputs[0]?) &&
      decide (sample.inputs[1]?.map (·.wire) =
        layout.projectionTrapdoor.secret.outputs[0]?)) &&
  decide (layout.oneTarget.target.inputs[0]?.map (·.wire) =
    layout.oneTarget.difference.outputs[0]?) &&
  decide (layout.oneTarget.target.inputs[1]?.map (·.wire) =
    layout.oneTarget.zeroRow.outputs[0]?) &&
  decide (layout.oneTarget.difference.inputs[1]?.map (·.wire) =
    layout.oneTarget.gadget.outputs[0]?) &&
  decide (layout.onePreimage.sample.inputs[2]?.map (·.wire) =
    layout.oneTarget.target.outputs[0]?) &&
  decide (some layout.onePreimageArtifact.producerOutput.wire =
    layout.onePreimage.materialize.outputs[0]?) &&
  decide (layout.witnessTrapdoors.indexFamily.wire ∈ layout.witnessIndices.outputs) &&
  decide (layout.witnessPublicKeys.indexFamily.wire ∈ layout.witnessIndices.outputs) &&
  decide (layout.witnessPublicKeys.sourceFamilies[0]?.map (·.wire) =
    sampling.slices.parallelLoop.outputs[0]?) &&
  decide (layout.witnessTargets.parallelLoop.arguments[0]?.map (·.wire) =
    layout.witnessPublicKeys.outputFamilies[0]?) &&
  decide (some layout.witnessPreimagesArtifact.producerOutput.wire =
    layout.witnessPreimages.parallelLoop.outputs[0]?) &&
  decide (layout.kTarget.publicKeyHash.inputs[0]?.map (·.wire) =
    sampling.packedHash.inputs[0]?.map (·.wire)) &&
  decide (layout.kTarget.firstColumn.inputs[0]?.map (·.wire) =
    layout.kTarget.publicKeyHash.outputs[0]?) &&
  decide (layout.kTarget.target.inputs[0]?.map (·.wire) =
    layout.kTarget.firstColumn.outputs[0]?) &&
  decide (layout.kTarget.target.inputs[1]?.map (·.wire) =
    layout.kTarget.halfModulus.outputs[0]?) &&
  decide (layout.kPreimage.sample.inputs[2]?.map (·.wire) =
    layout.kTarget.target.outputs[0]?) &&
  decide (some layout.kPreimageArtifact.producerOutput.wire =
    layout.kPreimage.materialize.outputs[0]?) &&
  decide (layout.rHash.inputs[0]?.map (·.wire) = sampling.packedHash.inputs[0]?.map (·.wire)) &&
  decide (layout.rSlice.inputs[0]?.map (·.wire) = layout.rHash.outputs[0]?) &&
  decide (layout.rDecomposition.inputs[0]?.map (·.wire) = layout.rSlice.outputs[0]?) &&
  decide (layout.rMaterialization.inputs[0]?.map (·.wire) =
    layout.rDecomposition.outputs[0]?) &&
  decide (layout.rReshape.inputs[0]?.map (·.wire) = layout.rMaterialization.outputs[0]?) &&
  decide (some layout.rDecomposedArtifact.producerOutput.wire = layout.rReshape.outputs[0]?) &&
  decide (layout.decoderTarget.publicKeyDifference.inputs[1]?.map (·.wire) =
    some booleanLayers.encryption.selectedOutput.output) &&
  decide (layout.decoderTarget.projectedDifference.inputs[0]?.map (·.wire) =
    layout.decoderTarget.publicKeyDifference.outputs[0]?) &&
  decide (layout.decoderTarget.projectedDifference.inputs[1]?.map (·.wire) =
    layout.rReshape.outputs[0]?) &&
  decide (layout.decoderTarget.publicKeySum.inputs[0]?.map (·.wire) =
    layout.kTarget.firstColumn.outputs[0]?) &&
  decide (layout.decoderTarget.publicKeySum.inputs[1]?.map (·.wire) =
    layout.decoderTarget.projectedDifference.outputs[0]?) &&
  decide (layout.decoderTarget.target.inputs[0]?.map (·.wire) =
    layout.decoderTarget.publicKeySum.outputs[0]?) &&
  decide (layout.decoderPreimage.sample.inputs[2]?.map (·.wire) =
    layout.decoderTarget.target.outputs[0]?) &&
  decide (some layout.decoderPreimageArtifact.producerOutput.wire =
    layout.decoderPreimage.materialize.outputs[0]?) &&
  verifyArtifactPreprocessingProducerNodes workflow layout

def verifyEncodingComponentOperations (workflow : Mxx.Ir.Workflow)
    (layout : EncodingComponentOperationsLayout) : Bool :=
  [layout.vectors, layout.publicKeys, layout.plaintexts].all (fun operation =>
    verifyParallelOperation workflow operation (fun kind => match kind with
      | .select => true
      | _ => false) && decide (!operation.body.inputs.isEmpty) &&
      decide (operation.body.outputs.length = 1) &&
      verifyExactParallelLoopRole workflow operation.parallelLoop
        (.parameter "max_layer_width") 0 [.zip, .zip, .zip])

private def verifyParallelSelectionSources (reference : ParallelOperationRef)
    (selector branchZero branchOne : Option CoreWireRef) : Bool :=
  match selector, branchZero, branchOne with
  | some selector, some branchZero, some branchOne =>
      decide (reference.parallelLoop.arguments.map (·.wire) =
        [selector, branchZero, branchOne])
  | _, _, _ => false

def verifyDecryptionWitnessIndexFormula (workflow : Mxx.Ir.Workflow)
    (layout : DecryptionInitialEncodingsLayout) : Bool :=
  verifyExactParallelLoopRole workflow layout.witnessIndices witnessSize 0 [] &&
    verifyExactParallelBody workflow layout.witnessIndices decryptionWitnessIndexNodes &&
    decide (layout.witnessIndices.bodyOutputs.map wireRef = [localWire 2]) &&
    decide layout.witnessIndices.arguments.isEmpty

/-- The exact witness-index formula has no outer arguments.  This is verified wiring exposed for
execution lifting; it adds no certificate field or semantic premise. -/
theorem verifyDecryptionWitnessIndexFormula_argumentsEmpty
    {workflow : Mxx.Ir.Workflow} {layout : DecryptionInitialEncodingsLayout}
    (verified : verifyDecryptionWitnessIndexFormula workflow layout = true) :
    layout.witnessIndices.arguments = [] := by
  unfold verifyDecryptionWitnessIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  simpa using verified.2

/-- The witness-index loop count is exactly the checked batch-width product. -/
theorem verifyDecryptionWitnessIndexFormula_count
    {workflow : Mxx.Ir.Workflow} {layout : DecryptionInitialEncodingsLayout}
    (verified : verifyDecryptionWitnessIndexFormula workflow layout = true) :
    layout.witnessIndices.count =
      .multiply (.parameter "diamond_batch_bits") (.parameter "diamond_input_count") := by
  unfold verifyDecryptionWitnessIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have roleVerified := verified.1.1.1
  unfold verifyExactParallelLoopRole at roleVerified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at roleVerified
  have countEq := roleVerified.1.1.1.1.1.2
  change layout.witnessIndices.count =
    .multiply (.parameter "diamond_batch_bits") (.parameter "diamond_input_count") at countEq
  exact countEq

def verifyWitnessGatherParent (workflow : Mxx.Ir.Workflow)
    (layout : DecryptionInitialEncodingsLayout) : Bool :=
  verifyParallelGather workflow layout.witnessBits &&
    verifyExactParallelLoopRole workflow layout.witnessBits.parallelLoop witnessSize 0
      [.zip, .broadcast]

def verifyWitnessSourceLoopLocations (workflowLayout : DiamondWorkflowLayout)
    (layout : DecryptionInitialEncodingsLayout) : Bool :=
  decide (layout.witnessIndices.operation.stage = workflowLayout.decryption.stage) &&
    decide (layout.witnessIndices.operation.scope = .root) &&
    decide (layout.witnessBits.parallelLoop.operation.stage = workflowLayout.decryption.stage) &&
    decide (layout.witnessBits.parallelLoop.operation.scope = .root)

def verifyWitnessDigitPackingSources (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (inputInjection : InputInjectionLayout)
    (layout : DecryptionInitialEncodingsLayout) : Bool :=
  verifyWitnessSourceLoopLocations workflowLayout layout &&
  verifyWitnessGatherParent workflow layout &&
  verifyDecryptionWitnessIndexFormula workflow layout &&
  (match layout.witnessBits.sourceFamilies[0]? with
  | some witness =>
      verifyInputWire workflow witness.wire workflowLayout.decryption.stage "boolean-witness"
  | none => false) &&
  decide (layout.witnessDigits.parallelLoop.outputs[0]? = some inputInjection.packedDigits.wire)

def verifyDecryptionInitialEncodings (workflow : Mxx.Ir.Workflow)
    (workflowLayout : DiamondWorkflowLayout) (inputInjection : InputInjectionLayout)
    (booleanLayers : BooleanLayersLayout) (layout : DecryptionInitialEncodingsLayout) : Bool :=
  [layout.initialStateArtifact, layout.onePreimageArtifact,
    layout.witnessPreimagesArtifact, layout.publicKeysArtifact].all (fun artifact =>
      verifyArtifactInLayout workflow workflowLayout artifact &&
        decide (artifact ∈ workflowLayout.artifacts)) &&
  verifyParallelLoop workflow layout.witnessIndices &&
  verifyParallelGather workflow layout.witnessBits &&
  verifyWitnessDigitPackingRef workflow layout.witnessDigits &&
  verifyOperationKind workflow layout.initialProjectionState (fun kind => match kind with
    | .familyGetStatic (.constant 0) => true
    | _ => false) &&
  verifyOperationKind workflow layout.onePublicKey (fun kind => match kind with
    | .familyGetStatic (.constant 0) => true
    | _ => false) &&
  verifyOperationKind workflow layout.onePlaintext (fun kind => match kind with
    | .identityMatrix actualType => decide (actualType = unitMatrixType)
    | _ => false) &&
  decide (layout.zeroEncoding.length = 3) &&
  layout.zeroEncoding.all (fun operation =>
    verifyOperationKind workflow operation (fun kind => match kind with
      | .matrixSubtract => true
      | _ => false) && decide (operation.inputs.length = 2) &&
      decide (operation.inputs[0]?.map (·.wire) = operation.inputs[1]?.map (·.wire))) &&
  verifyParallelLoop workflow layout.witnessStateIndices &&
  verifyParallelGather workflow layout.witnessStates &&
  verifyParallelMatrixBinary workflow layout.witnessVectors .matrixMultiply &&
  verifyExactParallelNodeRole workflow layout.witnessVectors.parallelLoop witnessSize 0
    [.zip, .zip] &&
  verifyParallelLoop workflow layout.witnessPublicIndices &&
  verifyParallelGather workflow layout.witnessPublicKeys &&
  decide (layout.witnessPlaintextConstants.length = 2) &&
  layout.witnessPlaintextConstants.all (verifyParallelLoop workflow) &&
  verifyParallelOperation workflow layout.witnessPlaintexts (fun kind => match kind with
    | .select => true
    | _ => false) &&
  verifyEvaluateInt workflow layout.instanceWidth &&
  verifyParallelLoop workflow layout.packedIndices &&
  verifyParallelGather workflow layout.packedVectors &&
  verifyParallelGather workflow layout.packedPublicKeys &&
  verifyParallelGather workflow layout.packedPlaintexts &&
  verifyParallelLoop workflow layout.activeWitness &&
  decide (layout.activeWitnessZeroes.length = 3) &&
  layout.activeWitnessZeroes.all (verifyParallelLoop workflow) &&
  verifyEncodingComponentOperations workflow layout.activeWitnessSelection &&
  decide (layout.instanceConstants.length = 3) &&
  layout.instanceConstants.all (fun component =>
    decide (component.length = 2) && component.all (verifyParallelLoop workflow)) &&
  verifyEncodingComponentOperations workflow layout.selectedInstance &&
  verifyParallelLoop workflow layout.activeInstance &&
  verifyEncodingComponentOperations workflow layout.circuitInputs &&
  verifyExactParallelLoopRole workflow layout.witnessIndices witnessSize 0 [] &&
  verifyExactParallelLoopRole workflow layout.witnessBits.parallelLoop witnessSize 0
    [.zip, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.witnessStateIndices witnessSize 0 [] &&
  verifyExactParallelLoopRole workflow layout.witnessStates.parallelLoop witnessSize 0
    [.zip, .broadcast] &&
  verifyExactParallelLoopRole workflow layout.witnessPublicIndices witnessSize 0 [] &&
  verifyExactParallelLoopRole workflow layout.witnessPublicKeys.parallelLoop witnessSize 0
    [.zip, .broadcast] &&
  layout.witnessPlaintextConstants.all (fun loop =>
    verifyExactParallelLoopRole workflow loop witnessSize 0 []) &&
  verifyExactParallelLoopRole workflow layout.witnessPlaintexts.parallelLoop witnessSize 0
    [.zip, .zip, .zip] &&
  verifyExactParallelLoopRole workflow layout.packedIndices
    (.parameter "max_layer_width") 0 [.broadcast, .broadcast] &&
  [layout.packedVectors, layout.packedPublicKeys, layout.packedPlaintexts].all (fun gather =>
    verifyExactParallelLoopRole workflow gather.parallelLoop
      (.parameter "max_layer_width") 0 [.zip, .broadcast]) &&
  verifyExactParallelLoopRole workflow layout.activeWitness
    (.parameter "max_layer_width") 0 [.broadcast, .broadcast] &&
  layout.activeWitnessZeroes.all (fun loop =>
    verifyExactParallelLoopRole workflow loop
      (.parameter "max_layer_width") 0 [.broadcast]) &&
  layout.instanceConstants.all (fun component => component.all (fun loop =>
    verifyExactParallelLoopRole workflow loop
      (.parameter "max_layer_width") 0 [.broadcast])) &&
  verifyExactParallelLoopRole workflow layout.activeInstance
    (.parameter "max_layer_width") 0 [.broadcast] &&
  verifyExactParallelBody workflow layout.packedIndices decryptionPackedIndexNodes &&
  verifyExactParallelBody workflow layout.activeWitness decryptionActiveWitnessNodes &&
  verifyExactParallelBody workflow layout.activeInstance decryptionActiveInstanceNodes &&
  (match layout.activeWitnessZeroes with
  | [zeroVector, zeroPublicKey, zeroPlaintext] =>
      verifyParallelSelectionSources layout.activeWitnessSelection.vectors
        layout.activeWitness.outputs[0]? zeroVector.outputs[0]?
        layout.packedVectors.outputFamilies[0]? &&
      verifyParallelSelectionSources layout.activeWitnessSelection.publicKeys
        layout.activeWitness.outputs[0]? zeroPublicKey.outputs[0]?
        layout.packedPublicKeys.outputFamilies[0]? &&
      verifyParallelSelectionSources layout.activeWitnessSelection.plaintexts
        layout.activeWitness.outputs[0]? zeroPlaintext.outputs[0]?
        layout.packedPlaintexts.outputFamilies[0]?
  | _ => false) &&
  (match layout.instanceConstants,
      layout.selectedInstance.vectors.parallelLoop.arguments[0]? with
  | [[zeroVector, oneVector], [zeroPublicKey, onePublicKey],
      [zeroPlaintext, onePlaintext]], some selector =>
      verifyInputWire workflow selector.wire workflowLayout.decryption.stage
        "boolean-instance" &&
      verifyParallelSelectionSources layout.selectedInstance.vectors (some selector.wire)
        zeroVector.outputs[0]? oneVector.outputs[0]? &&
      verifyParallelSelectionSources layout.selectedInstance.publicKeys (some selector.wire)
        zeroPublicKey.outputs[0]? onePublicKey.outputs[0]? &&
      verifyParallelSelectionSources layout.selectedInstance.plaintexts (some selector.wire)
        zeroPlaintext.outputs[0]? onePlaintext.outputs[0]?
  | _, _ => false) &&
  verifyParallelSelectionSources layout.circuitInputs.vectors
    layout.activeInstance.outputs[0]?
    layout.activeWitnessSelection.vectors.parallelLoop.outputs[0]?
    layout.selectedInstance.vectors.parallelLoop.outputs[0]? &&
  verifyParallelSelectionSources layout.circuitInputs.publicKeys
    layout.activeInstance.outputs[0]?
    layout.activeWitnessSelection.publicKeys.parallelLoop.outputs[0]?
    layout.selectedInstance.publicKeys.parallelLoop.outputs[0]? &&
  verifyParallelSelectionSources layout.circuitInputs.plaintexts
    layout.activeInstance.outputs[0]?
    layout.activeWitnessSelection.plaintexts.parallelLoop.outputs[0]?
    layout.selectedInstance.plaintexts.parallelLoop.outputs[0]? &&
  [layout.witnessIndices.operation, layout.witnessBits.parallelLoop.operation,
    layout.witnessDigits.parallelLoop.operation, layout.initialProjectionState.operation,
    layout.onePublicKey.operation, layout.onePlaintext.operation,
    layout.witnessStateIndices.operation, layout.witnessStates.parallelLoop.operation,
    layout.witnessVectors.parallelLoop, layout.witnessPublicIndices.operation,
    layout.witnessPublicKeys.parallelLoop.operation,
    layout.witnessPlaintexts.parallelLoop.operation, layout.instanceWidth.operation,
    layout.packedIndices.operation, layout.packedVectors.parallelLoop.operation,
    layout.packedPublicKeys.parallelLoop.operation,
    layout.packedPlaintexts.parallelLoop.operation, layout.activeWitness.operation,
    layout.activeInstance.operation].all
      (fun operation => decide (operation.stage = workflowLayout.decryption.stage) &&
        decide (operation.scope = .root)) &&
  decide (layout.initialProjectionState.inputs[0]?.map (·.wire) =
    some inputInjection.finalStates) &&
  decide (layout.onePublicKey.inputs[0]?.map (·.wire) =
    some { node := layout.publicKeysArtifact.consumerInput.node, port := 0 }) &&
  decide (layout.witnessBits.indexFamily.wire ∈ layout.witnessIndices.outputs) &&
  decide (layout.witnessDigits.parallelLoop.arguments[0]?.map (·.wire) =
    layout.witnessBits.outputFamilies[0]?) &&
  decide (layout.witnessStates.indexFamily.wire ∈ layout.witnessStateIndices.outputs) &&
  decide (layout.witnessStates.sourceFamilies[0]?.map (·.wire) =
    some inputInjection.finalStates) &&
  decide (layout.witnessVectors.leftFamily.wire ∈ layout.witnessStates.outputFamilies) &&
  decide (layout.witnessVectors.rightFamily.wire =
    { node := layout.witnessPreimagesArtifact.consumerInput.node, port := 0 }) &&
  decide (layout.witnessPublicKeys.indexFamily.wire ∈ layout.witnessPublicIndices.outputs) &&
  decide (layout.witnessPublicKeys.sourceFamilies[0]?.map (·.wire) =
    some { node := layout.publicKeysArtifact.consumerInput.node, port := 0 }) &&
  decide (layout.witnessPlaintexts.parallelLoop.arguments[0]?.map (·.wire) =
    layout.witnessBits.outputFamilies[0]?) &&
  decide (layout.packedVectors.indexFamily.wire ∈ layout.packedIndices.outputs) &&
  decide (layout.packedPublicKeys.indexFamily.wire ∈ layout.packedIndices.outputs) &&
  decide (layout.packedPlaintexts.indexFamily.wire ∈ layout.packedIndices.outputs) &&
  decide (layout.circuitInputs.vectors.parallelLoop.outputs[0]? =
    some booleanLayers.decryption.initialVectors.wire) &&
  decide (layout.circuitInputs.publicKeys.parallelLoop.outputs[0]? =
    some booleanLayers.decryption.initialPublicKeys.wire) &&
  decide (layout.circuitInputs.plaintexts.parallelLoop.outputs[0]? =
    some booleanLayers.decryption.initialPlaintexts.wire) &&
  verifyWitnessDigitPackingSources workflow workflowLayout inputInjection layout

def verifyRootProtocolInput (workflow : Mxx.Ir.Workflow) (name : String)
    (wire : CoreWireRef) : Bool :=
  verifyInputWire workflow wire wire.node.stage name &&
    match resolveStage workflow wire.node.stage with
    | some stage => match stage.inputs.find? fun input => input.1 = name with
      | some (_, .protocol _) => true
      | _ => false
    | none => false

def verifyExactFamilyGetRole (workflow : Mxx.Ir.Workflow)
    (reference : ParallelFamilyGetRef) : Bool :=
  verifyParallelFamilyGet workflow reference &&
    verifyExactParallelLoopRole workflow reference.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .broadcast]

def verifyScalarMetadata (workflow : Mxx.Ir.Workflow) (sequential : CoreNodeRef)
    (expectedOuter : CoreOperandRef) (expectedInner : CoreWireRef)
    (layout : LayerScalarMetadataRef) : Bool :=
  verifyEvaluateInt workflow layout.layerIndex &&
    verifyDynamicGet workflow layout.selected layout.bodySource &&
    verifyRootProtocolInput workflow layout.sourceInputName layout.rootInput &&
    decide (layout.sequentialOperand = expectedOuter) &&
    decide (layout.bodySource = expectedInner) &&
    decide (layout.rootInput = layout.sequentialOperand.wire) &&
    decide (layout.selected.index.wire = layout.layerIndex.output) &&
    match resolveNode workflow sequential with
    | some { kind := .sequentialLoop _ _ indexSlot _ _, .. } =>
        decide (layout.layerIndex.expression = .loopIndex indexSlot)
    | _ => false

def verifyFamilyMetadata (workflow : Mxx.Ir.Workflow) (sequential : CoreNodeRef)
    (expectedOuter : CoreOperandRef) (expectedInner : CoreWireRef)
    (layout : LayerFamilyMetadataRef) : Bool :=
  verifyExactParallelLoopRole workflow layout.flattenedIndices
      (.parameter "max_layer_width") 1 [] &&
    verifyEvaluateInt workflow layout.flattenedIndex &&
    verifyExactFamilyGetRole workflow layout.gathered &&
    verifyRootProtocolInput workflow layout.sourceInputName layout.rootInput &&
    decide (layout.sequentialOperand = expectedOuter) &&
    decide (layout.bodySource = expectedInner) &&
    decide (layout.rootInput = layout.sequentialOperand.wire) &&
    decide layout.flattenedIndices.arguments.isEmpty &&
    decide (layout.flattenedIndices.bodyOutputs = [layout.flattenedIndex.output]) &&
    decide (layout.flattenedIndices.outputs[0]? = some layout.gathered.indexFamily.wire) &&
    decide (layout.gathered.sourceFamily.wire = layout.bodySource) &&
    match resolveNode workflow sequential with
    | some { kind := .sequentialLoop _ _ layerSlot _ _, .. } =>
        decide (layout.flattenedIndex.expression = .add
          (.multiply (.loopIndex layerSlot) layout.flattenedIndices.count)
          (.loopIndex layout.flattenedIndices.indexSlot))
    | _ => false

def verifyBooleanMetadata (workflow : Mxx.Ir.Workflow) (sequential : CoreNodeRef)
    (layout : BooleanLayerMetadataLayout)
    (expected : List (CoreOperandRef × CoreWireRef)) : Bool :=
  match expected with
  | [active, opcode, leftSource, rightSource] =>
      verifyScalarMetadata workflow sequential active.1 active.2 layout.activeGateCount &&
        verifyFamilyMetadata workflow sequential opcode.1 opcode.2 layout.opcode &&
        verifyFamilyMetadata workflow sequential leftSource.1 leftSource.2 layout.leftSource &&
        verifyFamilyMetadata workflow sequential rightSource.1 rightSource.2 layout.rightSource
  | _ => false

def ScopeRef.isWithin (scope ancestor : ScopeRef) : Bool :=
  if scope = ancestor then true else
    match scope with
    | .parallelBody parent _ | .sequentialBody parent _ => parent.isWithin ancestor
    | .root | .subgraph _ => false

def verifyNodeParameter (operation : CoreNodeRef) (reference : CoreNodeParameterRef)
    (expected : CoreNodeParameter) : Bool :=
  decide (reference.node = operation) && decide (reference.parameter = expected)

def verifyDecompositionMaterialization (workflow : Mxx.Ir.Workflow)
    (decomposition materialized : CoreWireRef) : Bool :=
  if materialized = decomposition then true else
    decide (materialized.node.stage = decomposition.node.stage) &&
      decide (materialized.node.scope = decomposition.node.scope) &&
      decide (materialized.port = 0) &&
      match resolveNode workflow materialized.node with
      | some { kind := .matrixScale (.constant 1), arguments, outputCount } =>
          decide (arguments = [wireRef decomposition]) && decide (outputCount = 1)
      | _ => false

def verifyLocalDecomposition (workflow : Mxx.Ir.Workflow) (expectedScope : ScopeRef)
    (reference : LocalGadgetDecompositionRef) : Bool :=
  decide (reference.decompositionNode.scope = expectedScope) &&
    verifyNodeParameter reference.decompositionNode reference.base .gadgetDecomposeBase &&
    verifyNodeParameter reference.decompositionNode reference.digitCount
      .gadgetDecomposeDigitCount &&
    decide (reference.rightPublicKey.node = reference.decompositionNode) &&
    decide (reference.rightPublicKey.operand = 0) &&
    verifyOperand workflow reference.rightPublicKey &&
    verifyOperationOutput workflow reference.decompositionNode reference.decomposition &&
    verifyDecompositionMaterialization workflow reference.decomposition reference.materialized &&
    match resolveNode workflow reference.decompositionNode with
    | some { kind := .gadgetDecompose actualType base count, arguments, outputCount } =>
        decide (actualType = matrixType (.multiply (.constant 1) digitCount) digitCount) &&
          decide (base = .parameter "diamond_gadget_base") &&
          decide (count = digitCount) &&
          decide (arguments = [wireRef reference.rightPublicKey.wire]) &&
          decide (outputCount = 1)
    | _ => false

def verifyParallelInput (workflow : Mxx.Ir.Workflow) (operation : CoreNodeRef)
    (bodyScope : ScopeRef) (outer : CoreOperandRef) (inner : CoreWireRef) : Bool :=
  decide (bodyScope = .parallelBody operation.scope operation.node) &&
    decide (outer.node = operation) && verifyOperand workflow outer && verifyWire workflow inner &&
    decide (inner.node.stage = operation.stage) && decide (inner.node.scope = bodyScope) &&
    match resolveNode workflow operation,
        resolveScope workflow { operation with scope := bodyScope } with
    | some { kind := .parallelLoop definition _ _ _ _, .. }, some body =>
        decide (definition = bodyScope.definitionName) &&
          decide ((scopeInputWires body)[outer.operand]? = some (wireRef inner))
    | _, _ => false

def verifyMultiplyConsumer (workflow : Mxx.Ir.Workflow) (expected : CoreWireRef)
    (consumer : CoreOperandRef) : Bool :=
  decide (consumer.operand = 1) && decide (consumer.wire = expected) &&
    verifyOperand workflow consumer &&
    match resolveNode workflow consumer.node with
    | some { kind := .matrixMultiply, .. } => true
    | _ => false

def verifyParallelDecompositionConsumer (workflow : Mxx.Ir.Workflow)
    (booleanBody : ScopeRef) (decompositionFamily : CoreWireRef)
    (consumer : ParallelDecompositionConsumer) : Bool :=
  verifyParallelInput workflow consumer.consumerLoop consumer.bodyScope
      consumer.decompositionFamily consumer.bodyDecomposition &&
    verifyExactParallelNodeRole workflow consumer.consumerLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] &&
    consumer.consumerLoop.scope.isWithin booleanBody &&
    decide (consumer.decompositionFamily.wire = decompositionFamily) &&
    verifyMultiplyConsumer workflow consumer.bodyDecomposition consumer.multiplicationConsumer

def verifyEncryptDecomposition (workflow : Mxx.Ir.Workflow) (booleanBody : ScopeRef)
    (layout : EncryptPublicKeyRhsDecomposition) : Bool :=
  verifyExactFamilyGetRole workflow layout.rightSelection &&
    verifyParallelInput workflow layout.enclosingParallelLoop layout.bodyScope
      layout.rightPublicKeyFamily layout.bodyRightPublicKey &&
    verifyExactParallelNodeRole workflow layout.enclosingParallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip, .zip, .broadcast, .broadcast] &&
    layout.enclosingParallelLoop.scope.isWithin booleanBody &&
    decide (layout.rightSelection.outputFamily = layout.rightPublicKeyFamily.wire) &&
    verifyLocalDecomposition workflow layout.bodyScope layout.localDecomposition &&
    decide (layout.localDecomposition.rightPublicKey.wire = layout.bodyRightPublicKey) &&
    verifyMultiplyConsumer workflow layout.localDecomposition.materialized
      layout.multiplicationConsumer

def verifyDecryptDecomposition (workflow : Mxx.Ir.Workflow) (booleanBody : ScopeRef)
    (layout : DecryptEncodingRhsDecomposition) : Bool :=
  verifyExactFamilyGetRole workflow layout.rightSelection &&
    verifyParallelInput workflow layout.decompositionLoop layout.bodyScope
      layout.rightPublicKeyFamily layout.bodyRightPublicKey &&
    verifyExactParallelNodeRole workflow layout.decompositionLoop
      (.parameter "max_layer_width") 1 [.zip] &&
    layout.decompositionLoop.scope.isWithin booleanBody &&
    decide (layout.rightSelection.outputFamily = layout.rightPublicKeyFamily.wire) &&
    verifyLocalDecomposition workflow layout.bodyScope layout.localDecomposition &&
    decide (layout.localDecomposition.rightPublicKey.wire = layout.bodyRightPublicKey) &&
    verifyWire workflow layout.bodyOutput &&
    verifyDecompositionMaterialization workflow layout.localDecomposition.decomposition
      layout.bodyOutput &&
    verifyOperationOutput workflow layout.decompositionLoop layout.decompositionFamily &&
    verifyParallelDecompositionConsumer workflow booleanBody layout.decompositionFamily
      layout.publicKeyConsumer &&
    verifyParallelDecompositionConsumer workflow booleanBody layout.decompositionFamily
      layout.vectorConsumer &&
    decide (layout.publicKeyConsumer.consumerLoop != layout.vectorConsumer.consumerLoop) &&
    match resolveScope workflow { layout.decompositionLoop with scope := layout.bodyScope } with
    | some body => decide (scopeOutputWires body = [wireRef layout.bodyOutput])
    | none => false

def verifyConstantTwoFamily (workflow : Mxx.Ir.Workflow) (reference : CoreWireRef) : Bool :=
  verifyConstantTwoMatrix workflow reference ||
  (decide (reference.port = 0) &&
    verifyExactParallelNodeRole workflow reference.node
      (.parameter "max_layer_width") 1 [] &&
    match resolveNode workflow reference.node with
    | some { kind := .parallelLoop definition _ _ _ _, arguments, outputCount } =>
        let bodyScope := ScopeRef.parallelBody reference.node.scope reference.node.node
        decide arguments.isEmpty && decide (outputCount = 1) &&
          decide (definition = bodyScope.definitionName) &&
          match resolveScope workflow { reference.node with scope := bodyScope } with
          | some body => match scopeOutputWires body with
            | [output] => verifyConstantTwoMatrix workflow {
                node := { reference.node with scope := bodyScope, node := output.node }
                port := output.port
              }
            | _ => false
          | none => false
    | _ => false)

def sameScopeWire (context : CoreNodeRef) (wire : Mxx.Ir.WireRef) : CoreWireRef := {
  node := { context with node := wire.node }
  port := wire.port
}

def verifyActiveMaskFormula (workflow : Mxx.Ir.Workflow) (parallelLoop : ParallelLoopRef)
    (activeCount selector : CoreWireRef) : Bool :=
  decide (selector.node.stage = parallelLoop.operation.stage) &&
  decide (selector.node.scope = parallelLoop.bodyScope) &&
  decide (activeCount.node.stage = selector.node.stage) &&
  decide (activeCount.node.scope = selector.node.scope) &&
  decide (selector.port = 0) &&
  match resolveNode workflow selector.node with
  | some { kind := .boolToInt, arguments := [comparison], outputCount := 1 } =>
      let comparison := sameScopeWire selector.node comparison
      match resolveNode workflow comparison.node with
      | some node => match node.kind, node.arguments, node.outputCount with
        | .intCompare .lessEqual, [slot, upper], 1 =>
            let slot := sameScopeWire selector.node slot
            let upper := sameScopeWire selector.node upper
            match resolveNode workflow slot.node, resolveNode workflow upper.node with
            | some slotNode, some upperNode =>
                match slotNode.kind, slotNode.arguments, slotNode.outputCount,
                    upperNode.kind, upperNode.arguments, upperNode.outputCount with
                | .evaluateInt (.loopIndex indexSlot), [], 1,
                    .intBinary .subtract, [count, one], 1 =>
                    decide (indexSlot = parallelLoop.indexSlot) &&
                    decide (sameScopeWire selector.node count = activeCount) &&
                    verifyConstantIntWire workflow (sameScopeWire selector.node one) 1
                | _, _, _, _, _, _ => false
            | _, _ => false
        | _, _, _ => false
      | _ => false
  | _ => false

def verifyLocalBooleanGate (workflow : Mxx.Ir.Workflow)
    (layout : LocalBooleanGateLayout) : Bool :=
  verifyExactParallelLoopRole workflow layout.parentLoop
      (.parameter "max_layer_width") 1 [.zip, .zip, .zip, .broadcast, .broadcast] &&
    verifyExactFamilyGetRole workflow layout.leftSelection &&
    verifyMatrixBinary workflow layout.zero .matrixSubtract &&
    verifyMatrixBinary workflow layout.not .matrixSubtract &&
    verifyMatrixBinary workflow layout.product .matrixMultiply &&
    verifyMatrixBinary workflow layout.sum .matrixAdd &&
    verifyMatrixBinary workflow layout.twoProduct .matrixMultiply &&
    verifyMatrixBinary workflow layout.xor .matrixSubtract &&
    verifySixWaySelect workflow layout.candidateSelect &&
    verifyTwoWaySelect workflow layout.activeSelect &&
    decide (layout.parentLoop.bodyScope = layout.bodyScope) &&
    decide (layout.parentLoop.arguments.length = 5) &&
    decide (layout.parentLoop.bodyInputs.length = 5) &&
    [(layout.opcodeFamily, layout.bodyOpcode), (layout.leftFamily, layout.bodyLeft),
      (layout.rightFamily, layout.bodyRight),
      (layout.onePublicKey, layout.bodyOnePublicKey),
      (layout.activeGateCount, layout.bodyActiveGateCount)].all (fun pair =>
        decide (layout.parentLoop.arguments[pair.1.operand]? = some pair.1) &&
        decide (layout.parentLoop.bodyInputs[pair.1.operand]? = some pair.2)) &&
    decide (layout.parentLoop.bodyOutputs = [layout.activeSelect.output]) &&
    decide (layout.parentLoop.outputs.length = 1) &&
    decide (layout.leftSelection.outputFamily = layout.leftFamily.wire) &&
    decide (layout.bodyOpcode = layout.candidateSelect.selector.wire) &&
    decide (layout.bodyLeft = layout.copy) &&
    decide (layout.bodyOnePublicKey = layout.one) &&
    verifyActiveMaskFormula workflow layout.parentLoop layout.bodyActiveGateCount
      layout.activeSelect.selector.wire &&
    decide (layout.zero.left.wire = layout.one) &&
    decide (layout.zero.right.wire = layout.one) &&
    decide (layout.not.left.wire = layout.one) &&
    decide (layout.not.right.wire = layout.copy) &&
    decide (layout.product.left.wire = layout.copy) &&
    decide (layout.sum.left.wire = layout.copy) &&
    decide (layout.sum.right.wire = layout.bodyRight) &&
    decide (layout.twoProduct.left.wire = layout.product.output) &&
    verifyConstantTwoMatrix workflow layout.twoProduct.right.wire &&
    decide (layout.xor.left.wire = layout.sum.output) &&
    decide (layout.xor.right.wire = layout.twoProduct.output) &&
    decide ((List.ofFn layout.candidateSelect.branches).map (·.wire) = [
      layout.zero.output, layout.one, layout.copy, layout.not.output,
      layout.product.output, layout.xor.output]) &&
    decide ((List.ofFn layout.activeSelect.branches).map (·.wire) = [
      layout.zero.output, layout.candidateSelect.output]) &&
    [layout.zero.operation, layout.not.operation, layout.product.operation,
      layout.sum.operation, layout.twoProduct.operation, layout.xor.operation,
      layout.candidateSelect.operation, layout.activeSelect.operation].all
      (fun operation => decide (operation.scope = layout.bodyScope))

def verifyFamilyProduct (workflow : Mxx.Ir.Workflow) : FamilyProductRef → Bool
  | .direct operation =>
      verifyParallelMatrixBinary workflow operation .matrixMultiply &&
        verifyExactParallelNodeRole workflow operation.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip]
  | .encodingVector leftTimesRightDecomposition rightTimesLeftPlaintext sum =>
      verifyParallelMatrixBinary workflow leftTimesRightDecomposition .matrixMultiply &&
        verifyParallelMatrixBinary workflow rightTimesLeftPlaintext .matrixMultiply &&
        verifyParallelMatrixBinary workflow sum .matrixAdd &&
        verifyExactParallelNodeRole workflow leftTimesRightDecomposition.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] &&
        verifyExactParallelNodeRole workflow rightTimesLeftPlaintext.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] &&
        verifyExactParallelNodeRole workflow sum.parallelLoop
          (.parameter "max_layer_width") 1 [.zip, .zip] &&
        decide (sum.leftFamily.wire = leftTimesRightDecomposition.outputFamily) &&
        decide (sum.rightFamily.wire = rightTimesLeftPlaintext.outputFamily)

def verifyFamilyBooleanGate (workflow : Mxx.Ir.Workflow)
    (layout : FamilyBooleanGateLayout) : Bool :=
  verifyExactFamilyGetRole workflow layout.leftSelection &&
  verifyExactFamilyGetRole workflow layout.rightSelection &&
    verifyExactParallelLoopRole workflow layout.oneRepetition
      (.parameter "max_layer_width") 1 [.broadcast] &&
    verifyExactParallelLoopRole workflow layout.activeMask
      (.parameter "max_layer_width") 1 [.broadcast] &&
    verifyParallelMatrixBinary workflow layout.zero .matrixSubtract &&
    verifyExactParallelNodeRole workflow layout.zero.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] &&
    verifyParallelMatrixBinary workflow layout.not .matrixSubtract &&
    verifyExactParallelNodeRole workflow layout.not.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] &&
    verifyFamilyProduct workflow layout.product &&
    verifyParallelMatrixBinary workflow layout.sum .matrixAdd &&
    verifyExactParallelNodeRole workflow layout.sum.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] &&
    verifyParallelMatrixBinary workflow layout.twoProduct .matrixMultiply &&
    verifyExactParallelNodeRole workflow layout.twoProduct.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .broadcast] &&
    verifyParallelMatrixBinary workflow layout.xor .matrixSubtract &&
    verifyExactParallelNodeRole workflow layout.xor.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip] &&
    verifyParallelSixWaySelect workflow layout.candidateSelect &&
    verifyExactParallelNodeRole workflow layout.candidateSelect.parallelLoop
      (.parameter "max_layer_width") 1 (List.replicate 7 .zip) &&
    verifyParallelTwoWaySelect workflow layout.activeSelect &&
    verifyExactParallelNodeRole workflow layout.activeSelect.parallelLoop
      (.parameter "max_layer_width") 1 [.zip, .zip, .zip] &&
    decide (layout.activeMask.arguments.length = 1) &&
    (match layout.activeMask.bodyInputs, layout.activeMask.bodyOutputs with
    | [activeCount], [selector] =>
        verifyActiveMaskFormula workflow layout.activeMask activeCount selector
    | _, _ => false) &&
    decide (layout.leftSelection.sourceFamily.wire = layout.stateInput) &&
    decide (layout.rightSelection.sourceFamily.wire = layout.stateInput) &&
    decide (layout.leftSelection.outputFamily = layout.copyFamily) &&
    decide (layout.oneRepetition.outputs = [layout.oneFamily]) &&
    decide (layout.activeMask.outputs = [layout.activeFamily]) &&
    decide (layout.zero.leftFamily.wire = layout.oneFamily) &&
    decide (layout.zero.rightFamily.wire = layout.oneFamily) &&
    decide (layout.not.leftFamily.wire = layout.oneFamily) &&
    decide (layout.not.rightFamily.wire = layout.copyFamily) &&
    decide (layout.sum.leftFamily.wire = layout.copyFamily) &&
    decide (layout.sum.rightFamily.wire = layout.rightSelection.outputFamily) &&
    decide (layout.twoProduct.leftFamily.wire = layout.product.outputFamily) &&
    verifyConstantTwoFamily workflow layout.twoProduct.rightFamily.wire &&
    decide (layout.xor.leftFamily.wire = layout.sum.outputFamily) &&
    decide (layout.xor.rightFamily.wire = layout.twoProduct.outputFamily) &&
    decide (layout.candidateSelect.selectorFamily.wire = layout.opcodeFamily) &&
    decide ((List.ofFn layout.candidateSelect.branchFamilies).map (·.wire) = [
      layout.zero.outputFamily, layout.oneFamily, layout.copyFamily,
      layout.not.outputFamily, layout.product.outputFamily, layout.xor.outputFamily]) &&
    decide (layout.activeSelect.selectorFamily.wire = layout.activeFamily) &&
    decide ((List.ofFn layout.activeSelect.branchFamilies).map (·.wire) = [
      layout.zero.outputFamily, layout.candidateSelect.outputFamily]) &&
    decide (layout.activeSelect.outputFamily = layout.stateOutput)

def verifyCircuitOutputIndex (workflow : Mxx.Ir.Workflow) (wire : CoreWireRef) : Bool :=
  decide (wire.port = 0) &&
  match resolveNode workflow wire.node with
  | some node => match node.kind, node.arguments, node.outputCount with
    | .familyGetDynamic, [source, index], 1 =>
        let source := sameScopeWire wire.node source
        let index := sameScopeWire wire.node index
        verifyConstantIntWire workflow index 0 &&
        match resolveNode workflow source.node with
        | some sourceNode =>
            decide (sourceNode.kind = .input "circuit-output-source") &&
            decide sourceNode.arguments.isEmpty && decide (sourceNode.outputCount = 1)
        | none => false
    | _, _, _ => false
  | none => false

def verifyPublicKeyBooleanLoop (workflow : Mxx.Ir.Workflow)
    (layout : PublicKeyBooleanLoopLayout) : Bool :=
  verifySequentialLoop workflow layout.layerScan layout.bodyScope [layout.initialPublicKeys]
      [layout.activeGateCounts, layout.gateKinds, layout.leftSources, layout.rightSources,
        layout.onePublicKey] [layout.finalPublicKeys] &&
    verifyExactSequentialNodeRole workflow layout.layerScan (.parameter "depth") 0 &&
    verifyLoopBody workflow layout.layerScan layout.bodyScope
      [layout.initialPublicKeys, layout.activeGateCounts, layout.gateKinds,
        layout.leftSources, layout.rightSources, layout.onePublicKey]
      [layout.bodyInitialPublicKeys, layout.bodyActiveGateCounts, layout.bodyGateKinds,
        layout.bodyLeftSources, layout.bodyRightSources, layout.bodyOnePublicKey]
      [layout.bodyFinalPublicKeys] [layout.finalPublicKeys] &&
    verifyDynamicGet workflow layout.selectedOutput layout.finalPublicKeys &&
    verifyCircuitOutputIndex workflow layout.selectedOutput.index.wire &&
    verifyBooleanMetadata workflow layout.layerScan layout.metadata [
      (layout.activeGateCounts, layout.bodyActiveGateCounts),
      (layout.gateKinds, layout.bodyGateKinds),
      (layout.leftSources, layout.bodyLeftSources),
      (layout.rightSources, layout.bodyRightSources)]

def verifyEncodingBooleanLoop (workflow : Mxx.Ir.Workflow)
    (layout : EncodingBooleanLoopLayout) : Bool :=
  verifySequentialLoop workflow layout.layerScan layout.bodyScope
      [layout.initialVectors, layout.initialPublicKeys, layout.initialPlaintexts]
      [layout.activeGateCounts, layout.gateKinds, layout.leftSources, layout.rightSources,
        layout.oneVector, layout.onePublicKey, layout.onePlaintext]
      [layout.finalVectors, layout.finalPublicKeys, layout.finalPlaintexts] &&
    verifyExactSequentialNodeRole workflow layout.layerScan (.parameter "depth") 0 &&
    verifyLoopBody workflow layout.layerScan layout.bodyScope
      [layout.initialVectors, layout.initialPublicKeys, layout.initialPlaintexts,
        layout.activeGateCounts, layout.gateKinds, layout.leftSources, layout.rightSources,
        layout.oneVector, layout.onePublicKey, layout.onePlaintext]
      [layout.bodyInitialVectors, layout.bodyInitialPublicKeys, layout.bodyInitialPlaintexts,
        layout.bodyActiveGateCounts, layout.bodyGateKinds, layout.bodyLeftSources,
        layout.bodyRightSources, layout.bodyOneVector, layout.bodyOnePublicKey,
        layout.bodyOnePlaintext]
      [layout.bodyFinalVectors, layout.bodyFinalPublicKeys, layout.bodyFinalPlaintexts]
      [layout.finalVectors, layout.finalPublicKeys, layout.finalPlaintexts] &&
    verifyDynamicGet workflow layout.selectedVector layout.finalVectors &&
    verifyCircuitOutputIndex workflow layout.selectedVector.index.wire &&
    verifyBooleanMetadata workflow layout.layerScan layout.metadata [
      (layout.activeGateCounts, layout.bodyActiveGateCounts),
      (layout.gateKinds, layout.bodyGateKinds),
      (layout.leftSources, layout.bodyLeftSources),
      (layout.rightSources, layout.bodyRightSources)]

def sameBooleanMetadataInputs (left right : BooleanLayerMetadataLayout) : Bool :=
  decide (left.activeGateCount.sourceInputName = right.activeGateCount.sourceInputName) &&
    decide (left.opcode.sourceInputName = right.opcode.sourceInputName) &&
    decide (left.leftSource.sourceInputName = right.leftSource.sourceInputName) &&
    decide (left.rightSource.sourceInputName = right.rightSource.sourceInputName)

def protocolInputBindingAux (workflow : Mxx.Ir.Workflow) : Nat → CoreWireRef → Option String
  | 0, _ => none
  | fuel + 1, wire => do
      let node ← resolveNode workflow wire.node
      match node.kind with
      | .input name => do
          guard (wire.node.scope = .root ∧ wire.port = 0)
          let stage ← resolveStage workflow wire.node.stage
          let source ← (stage.inputs.find? fun input => input.1 = name).map (·.2)
          match source with
          | .protocol protocolName => some protocolName
          | .artifact _ _ => none
      | .familyGetDynamic => do
          let source ← node.arguments[0]?
          protocolInputBindingAux workflow fuel {
            node := { wire.node with node := source.node }
            port := source.port
          }
      | _ => none

def protocolInputBinding (workflow : Mxx.Ir.Workflow) (wire : CoreWireRef) : Option String :=
  let stageNodes := (resolveStage workflow wire.node.stage).map
    (fun stage => stage.program.root.nodes.length) |>.getD 0
  protocolInputBindingAux workflow (stageNodes + 1) wire

def sameProtocolInputBinding (workflow : Mxx.Ir.Workflow)
    (left right : CoreWireRef) : Bool :=
  decide (protocolInputBinding workflow left = protocolInputBinding workflow right) &&
    decide (protocolInputBinding workflow left).isSome

def familyHasArgument (loop : ParallelLoopRef) (wire : CoreWireRef) : Bool :=
  loop.arguments.any fun argument => decide (argument.wire = wire)

def verifyFamilyGateRole (layout : FamilyBooleanGateLayout)
    (stateInput stateOutput one : CoreWireRef) (metadata : BooleanLayerMetadataLayout) : Bool :=
  decide (layout.stateInput = stateInput) && decide (layout.stateOutput = stateOutput) &&
    decide (layout.leftSelection.indexFamily.wire = metadata.leftSource.gathered.outputFamily) &&
    decide (layout.rightSelection.indexFamily.wire = metadata.rightSource.gathered.outputFamily) &&
    decide (layout.opcodeFamily = metadata.opcode.gathered.outputFamily) &&
    familyHasArgument layout.oneRepetition one &&
    familyHasArgument layout.activeMask metadata.activeGateCount.selected.output

def verifyBooleanLayers (workflow : Mxx.Ir.Workflow) (workflowLayout : DiamondWorkflowLayout)
    (layout : BooleanLayersLayout) : Bool :=
  verifyArtifactInLayout workflow workflowLayout layout.publicKeysArtifact &&
    decide (layout.publicKeysArtifact ∈ workflowLayout.artifacts) &&
    decide (layout.encryption.layerScan.stage = workflowLayout.encryption.stage) &&
    decide (layout.decryption.layerScan.stage = workflowLayout.decryption.stage) &&
    verifyPublicKeyBooleanLoop workflow layout.encryption &&
    verifyEncodingBooleanLoop workflow layout.decryption &&
    sameBooleanMetadataInputs layout.encryption.metadata layout.decryption.metadata &&
    sameProtocolInputBinding workflow layout.encryption.activeGateCounts.wire
      layout.decryption.activeGateCounts.wire &&
    sameProtocolInputBinding workflow layout.encryption.gateKinds.wire
      layout.decryption.gateKinds.wire &&
    sameProtocolInputBinding workflow layout.encryption.leftSources.wire
      layout.decryption.leftSources.wire &&
    sameProtocolInputBinding workflow layout.encryption.rightSources.wire
      layout.decryption.rightSources.wire &&
    sameProtocolInputBinding workflow layout.encryption.selectedOutput.index.wire
      layout.decryption.selectedVector.index.wire &&
    verifyEncryptDecomposition workflow layout.encryption.bodyScope
      layout.encryptPublicKeyRhsDecomposition &&
    verifyDecryptDecomposition workflow layout.decryption.bodyScope
      layout.decryptEncodingRhsDecomposition &&
    verifyLocalBooleanGate workflow layout.encryptionGate &&
    verifyFamilyBooleanGate workflow layout.decryptionVectors &&
    verifyFamilyBooleanGate workflow layout.decryptionPublicKeys &&
    verifyFamilyBooleanGate workflow layout.decryptionPlaintexts &&
    decide (layout.encryptionGate.bodyScope =
      layout.encryptPublicKeyRhsDecomposition.bodyScope) &&
    decide (layout.encryptionGate.parentLoop.operation =
      layout.encryptPublicKeyRhsDecomposition.enclosingParallelLoop) &&
    decide (layout.encryptionGate.rightFamily =
      layout.encryptPublicKeyRhsDecomposition.rightPublicKeyFamily) &&
    decide (layout.encryptionGate.bodyRight =
      layout.encryptPublicKeyRhsDecomposition.bodyRightPublicKey) &&
    decide (layout.encryptionGate.product.right.wire =
      layout.encryptPublicKeyRhsDecomposition.localDecomposition.materialized) &&
    decide (layout.encryptionGate.sum.right.wire = layout.encryptionGate.bodyRight) &&
    decide (layout.encryptionGate.opcodeFamily.wire =
      layout.encryption.metadata.opcode.gathered.outputFamily) &&
    decide (layout.encryptionGate.leftSelection.sourceFamily.wire =
      layout.encryption.bodyInitialPublicKeys) &&
    decide (layout.encryptionGate.leftSelection.indexFamily.wire =
      layout.encryption.metadata.leftSource.gathered.outputFamily) &&
    decide (layout.encryptionGate.leftSelection.outputFamily =
      layout.encryptionGate.leftFamily.wire) &&
    decide (layout.encryptPublicKeyRhsDecomposition.rightSelection.sourceFamily.wire =
      layout.encryption.bodyInitialPublicKeys) &&
    decide (layout.encryptPublicKeyRhsDecomposition.rightSelection.indexFamily.wire =
      layout.encryption.metadata.rightSource.gathered.outputFamily) &&
    decide (layout.encryptPublicKeyRhsDecomposition.rightSelection.outputFamily =
      layout.encryptionGate.rightFamily.wire) &&
    decide (layout.encryptionGate.onePublicKey.wire = layout.encryption.bodyOnePublicKey) &&
    decide (layout.encryptionGate.activeGateCount.wire =
      layout.encryption.metadata.activeGateCount.selected.output) &&
    decide (layout.encryptionGate.parentLoop.outputs =
      [layout.encryption.bodyFinalPublicKeys]) &&
    verifyFamilyGateRole layout.decryptionVectors layout.decryption.bodyInitialVectors
      layout.decryption.bodyFinalVectors layout.decryption.bodyOneVector
      layout.decryption.metadata &&
    verifyFamilyGateRole layout.decryptionPublicKeys layout.decryption.bodyInitialPublicKeys
      layout.decryption.bodyFinalPublicKeys layout.decryption.bodyOnePublicKey
      layout.decryption.metadata &&
    verifyFamilyGateRole layout.decryptionPlaintexts layout.decryption.bodyInitialPlaintexts
      layout.decryption.bodyFinalPlaintexts layout.decryption.bodyOnePlaintext
      layout.decryption.metadata &&
    decide (layout.decryptionPublicKeys.rightSelection =
      layout.decryptEncodingRhsDecomposition.rightSelection) &&
    decide (layout.decryptionPublicKeys.rightSelection.outputFamily =
      layout.decryptEncodingRhsDecomposition.rightPublicKeyFamily.wire) &&
    (match layout.decryptionPublicKeys.product with
    | .direct product =>
        decide (product.leftFamily.wire = layout.decryptionPublicKeys.leftSelection.outputFamily) &&
          decide (product.rightFamily.wire =
            layout.decryptEncodingRhsDecomposition.decompositionFamily)
    | _ => false) &&
    (match layout.decryptionVectors.product with
    | .encodingVector leftTimesRightDecomposition rightTimesLeftPlaintext _ =>
        decide (leftTimesRightDecomposition.rightFamily.wire =
          layout.decryptEncodingRhsDecomposition.decompositionFamily) &&
        decide (leftTimesRightDecomposition.leftFamily.wire =
          layout.decryptionVectors.leftSelection.outputFamily) &&
        decide (rightTimesLeftPlaintext.leftFamily.wire =
          layout.decryptionVectors.rightSelection.outputFamily) &&
        decide (rightTimesLeftPlaintext.rightFamily.wire =
          layout.decryptionPlaintexts.leftSelection.outputFamily)
    | _ => false) &&
    (match layout.decryptionPlaintexts.product with
    | .direct product =>
        decide (product.leftFamily.wire =
          layout.decryptionPlaintexts.leftSelection.outputFamily) &&
        decide (product.rightFamily.wire =
          layout.decryptionPlaintexts.rightSelection.outputFamily)
    | _ => false) &&
    [layout.encryptPublicKeyRhsDecomposition.localDecomposition.materialized,
      layout.decryptEncodingRhsDecomposition.localDecomposition.materialized,
      layout.decryptEncodingRhsDecomposition.decompositionFamily].all fun decomposition =>
        workflowLayout.encryption.outputs.all
            (fun output => decide (output.wire != decomposition)) &&
          workflowLayout.decryption.outputs.all
            (fun output => decide (output.wire != decomposition))

def verifyUnaryKind (workflow : Mxx.Ir.Workflow) (reference : UnaryNodeRef)
    (accept : Mxx.Ir.NodeKind → Bool) : Bool :=
  verifyUnaryNode workflow reference && decide (reference.output.port = 0) &&
    match resolveNode workflow reference.operation with
    | some node => accept node.kind && decide (node.outputCount = 1)
    | none => false

def verifyBinaryKind (workflow : Mxx.Ir.Workflow) (reference : BinaryNodeRef)
    (accept : Mxx.Ir.NodeKind → Bool) : Bool :=
  verifyBinaryNode workflow reference && decide (reference.output.port = 0) &&
    match resolveNode workflow reference.operation with
    | some node => accept node.kind && decide (node.outputCount = 1)
    | none => false

def verifyNamedArtifactInput (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout)
    (producerOutputName : String) (wire : CoreWireRef) : Bool :=
  decide (wire.port = 0) && layout.artifacts.any fun artifact =>
    decide (artifact.producerOutput.name = producerOutputName) &&
      decide (artifact.consumerStage = wire.node.stage) &&
      decide (artifact.consumerInput.node = wire.node) &&
      verifyArtifactInLayout workflow layout artifact

def verifyNamedStageOutput (layout : StageInterfaceLayout) (name : String)
    (wire : CoreWireRef) : Bool :=
  layout.outputs.any fun output => decide (output.name = name) && decide (output.wire = wire)

def operationNodeRefs (reference : OperationRef) : List CoreNodeRef :=
  [reference.operation]

def sequentialLoopNodeRefs (reference : SequentialLoopRef) : List CoreNodeRef :=
  reference.operation :: reference.bodyOutputs.map (·.node)

def parallelLoopNodeRefs (reference : ParallelLoopRef) : List CoreNodeRef :=
  reference.operation :: reference.bodyOutputs.map (·.node)

def parallelOperationNodeRefs (reference : ParallelOperationRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parallelLoop ++ operationNodeRefs reference.body

def directSelectArgumentNode (workflow : Mxx.Ir.Workflow) (output : CoreWireRef)
    (argument : Nat) : List CoreNodeRef :=
  match resolveNode workflow output.node with
  | some node => match node.kind, node.arguments[argument]? with
    | .select, some wire => [{ output.node with node := wire.node }]
    | _, _ => []
  | none => []

def indexFormulaNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : ParallelIndexFormulaRef) : List CoreNodeRef :=
  let directSelects :=
    match resolveNode workflow reference.bodyOutput.node with
    | some node => match node.kind, node.arguments with
      | .intBinary .add, [_, candidate] =>
          let candidateRef := { reference.bodyOutput.node with node := candidate.node }
          match resolveNode workflow candidateRef with
          | some candidateNode => match candidateNode.kind with
            | .select => [candidateRef]
            | _ => []
          | none => []
      | _, _ => []
    | none => []
  parallelLoopNodeRefs reference.parallelLoop ++ directSelects

def initialStateExpansionNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : InitialStateExpansionRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parallelLoop ++
    directSelectArgumentNode workflow reference.bodyOutput 1

def witnessDigitPackingNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : WitnessDigitPackingRef) : List CoreNodeRef :=
  let dynamicGet :=
    match reference.bitScan.bodyOutputs[0]? with
    | some accumulator => match resolveNode workflow accumulator.node with
      | some sum => match sum.kind, sum.arguments with
        | .intBinary .add, [_, productWire] =>
            let productRef := { accumulator.node with node := productWire.node }
            match resolveNode workflow productRef with
            | some product => match product.kind, product.arguments with
              | .intBinary .multiply, [bitWire, _] =>
                  let bitRef := { accumulator.node with node := bitWire.node }
                  match resolveNode workflow bitRef with
                  | some bit => match bit.kind with
                    | .familyGetDynamic => [bitRef]
                    | _ => []
                  | none => []
              | _, _ => []
            | none => []
        | _, _ => []
      | none => []
    | none => []
  parallelLoopNodeRefs reference.parallelLoop ++ sequentialLoopNodeRefs reference.bitScan ++
    dynamicGet

def preimageNodeRefs (reference : PreimageRef) : List CoreNodeRef :=
  operationNodeRefs reference.sample ++ operationNodeRefs reference.materialize

def parallelPreimageNodeRefs (reference : ParallelPreimageRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parallelLoop ++ preimageNodeRefs reference.body

def dynamicGetNodeRefs (reference : DynamicFamilyGetRef) : List CoreNodeRef :=
  [reference.operation]

def evaluateIntNodeRefs (reference : EvaluateIntRef) : List CoreNodeRef :=
  reference.operation :: reference.materialization.toList.map (·.operation)

def parallelFamilyGetNodeRefs (reference : ParallelFamilyGetRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parallelLoop ++ dynamicGetNodeRefs reference.get

def parallelGatherNodeRefs (reference : ParallelGatherRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parallelLoop ++ reference.gets.map (·.operation)

def matrixBinaryNodeRefs (reference : MatrixBinaryRef) : List CoreNodeRef :=
  [reference.operation]

def parallelMatrixBinaryNodeRefs (reference : ParallelMatrixBinaryRef) : List CoreNodeRef :=
  [reference.parallelLoop, reference.operation.operation]

def parallelSixWaySelectNodeRefs (reference : ParallelSixWaySelectRef) : List CoreNodeRef :=
  [reference.parallelLoop, reference.select.operation]

def parallelTwoWaySelectNodeRefs (reference : ParallelTwoWaySelectRef) : List CoreNodeRef :=
  [reference.parallelLoop, reference.select.operation]

def familyProductNodeRefs : FamilyProductRef → List CoreNodeRef
  | .direct operation => parallelMatrixBinaryNodeRefs operation
  | .encodingVector left right sum =>
      parallelMatrixBinaryNodeRefs left ++ parallelMatrixBinaryNodeRefs right ++
        parallelMatrixBinaryNodeRefs sum

def scalarMetadataNodeRefs (reference : LayerScalarMetadataRef) : List CoreNodeRef :=
  evaluateIntNodeRefs reference.layerIndex ++ dynamicGetNodeRefs reference.selected

def familyMetadataNodeRefs (reference : LayerFamilyMetadataRef) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.flattenedIndices ++ evaluateIntNodeRefs reference.flattenedIndex ++
    parallelFamilyGetNodeRefs reference.gathered

def booleanMetadataNodeRefs (reference : BooleanLayerMetadataLayout) : List CoreNodeRef :=
  scalarMetadataNodeRefs reference.activeGateCount ++ familyMetadataNodeRefs reference.opcode ++
    familyMetadataNodeRefs reference.leftSource ++ familyMetadataNodeRefs reference.rightSource

def publicKeyBooleanLoopNodeRefs (reference : PublicKeyBooleanLoopLayout) : List CoreNodeRef :=
  [reference.layerScan] ++ booleanMetadataNodeRefs reference.metadata ++
    dynamicGetNodeRefs reference.selectedOutput ++ [reference.selectedOutput.index.wire.node]

def encodingBooleanLoopNodeRefs (reference : EncodingBooleanLoopLayout) : List CoreNodeRef :=
  [reference.layerScan] ++ booleanMetadataNodeRefs reference.metadata ++
    dynamicGetNodeRefs reference.selectedVector ++ [reference.selectedVector.index.wire.node]

def localDecompositionNodeRefs (reference : LocalGadgetDecompositionRef) : List CoreNodeRef :=
  reference.decompositionNode ::
    (if reference.materialized = reference.decomposition then [] else [reference.materialized.node])

def encryptDecompositionNodeRefs
    (reference : EncryptPublicKeyRhsDecomposition) : List CoreNodeRef :=
  parallelFamilyGetNodeRefs reference.rightSelection ++ [reference.enclosingParallelLoop] ++
    localDecompositionNodeRefs reference.localDecomposition ++
    [reference.multiplicationConsumer.node]

def parallelDecompositionConsumerNodeRefs
    (reference : ParallelDecompositionConsumer) : List CoreNodeRef :=
  [reference.consumerLoop, reference.multiplicationConsumer.node]

def decryptDecompositionNodeRefs
    (reference : DecryptEncodingRhsDecomposition) : List CoreNodeRef :=
  parallelFamilyGetNodeRefs reference.rightSelection ++ [reference.decompositionLoop] ++
    localDecompositionNodeRefs reference.localDecomposition ++
    parallelDecompositionConsumerNodeRefs reference.publicKeyConsumer ++
    parallelDecompositionConsumerNodeRefs reference.vectorConsumer

def localBooleanGateNodeRefs (reference : LocalBooleanGateLayout) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.parentLoop ++ parallelFamilyGetNodeRefs reference.leftSelection ++
    [reference.zero.operation, reference.not.operation, reference.product.operation,
      reference.sum.operation, reference.twoProduct.operation, reference.twoProduct.right.wire.node,
      reference.xor.operation, reference.candidateSelect.operation,
      reference.activeSelect.operation]

def constantTwoFamilyNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : CoreWireRef) : List CoreNodeRef :=
  match resolveNode workflow reference.node with
  | some { kind := .constantMatrix .., .. } => [reference.node]
  | some { kind := .parallelLoop .., .. } =>
      let bodyScope := ScopeRef.parallelBody reference.node.scope reference.node.node
      match resolveScope workflow { reference.node with scope := bodyScope } with
      | some body => (scopeOutputWires body).map fun output =>
          { reference.node with scope := bodyScope, node := output.node }
      | none => []
  | _ => []

def familyBooleanGateNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : FamilyBooleanGateLayout) : List CoreNodeRef :=
  parallelFamilyGetNodeRefs reference.leftSelection ++
    parallelFamilyGetNodeRefs reference.rightSelection ++
    parallelLoopNodeRefs reference.oneRepetition ++ parallelLoopNodeRefs reference.activeMask ++
    parallelMatrixBinaryNodeRefs reference.zero ++ parallelMatrixBinaryNodeRefs reference.not ++
    familyProductNodeRefs reference.product ++ parallelMatrixBinaryNodeRefs reference.sum ++
    parallelMatrixBinaryNodeRefs reference.twoProduct ++
    constantTwoFamilyNodeRefs workflow reference.twoProduct.rightFamily.wire ++
    parallelMatrixBinaryNodeRefs reference.xor ++
    parallelSixWaySelectNodeRefs reference.candidateSelect ++
    parallelTwoWaySelectNodeRefs reference.activeSelect

def transitionSelectorNodeRefs (reference : TransitionSelectorLayout) : List CoreNodeRef :=
  operationNodeRefs reference.regular ++ operationNodeRefs reference.kIdentity ++
    operationNodeRefs reference.k ++ operationNodeRefs reference.initialSelect ++
    sequentialLoopNodeRefs reference.bitScan ++ operationNodeRefs reference.bitBody.bitExtract ++
    operationNodeRefs reference.bitBody.bitToInt ++ operationNodeRefs reference.bitBody.bitZero ++
    operationNodeRefs reference.bitBody.bitOne ++ operationNodeRefs reference.bitBody.bitSelect ++
    operationNodeRefs reference.bitBody.specialProduct ++
    operationNodeRefs reference.bitBody.specialTop ++
    operationNodeRefs reference.bitBody.specialBottom ++
    operationNodeRefs reference.bitBody.special ++ operationNodeRefs reference.bitBody.stateMatch ++
    operationNodeRefs reference.bitBody.stateMatchToInt ++
    operationNodeRefs reference.bitBody.selector

def preprocessingNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : DiamondInputPreprocessingLayout) : List CoreNodeRef :=
  [reference.trapdoorSamples.parallelLoop.operation] ++
    operationNodeRefs reference.trapdoorSamples.body ++
    operationNodeRefs reference.secretSample ++ operationNodeRefs reference.messageSelector ++
    operationNodeRefs reference.initialErrorSample ++
    operationNodeRefs reference.initialPublicProduct ++
    (reference.initialPublicProduct.inputs.drop 1).map (·.wire.node) ++
    operationNodeRefs reference.initialState ++
    indexFormulaNodeRefs workflow reference.transitionSourceIndices ++
    indexFormulaNodeRefs workflow reference.transitionTargetIndices ++
    indexFormulaNodeRefs workflow reference.digitSecretIndices ++
    [reference.digitSecretSamples.parallelLoop.operation] ++
    operationNodeRefs reference.digitSecretSamples.body ++
    parallelGatherNodeRefs reference.digitSecrets ++
    parallelGatherNodeRefs reference.transitionSources ++
    parallelGatherNodeRefs reference.targetPublicMatrices ++
    [reference.transitionTargets.parallelLoop.operation] ++
    transitionSelectorNodeRefs reference.transitionTargets.body.selectorConstruction ++
    operationNodeRefs reference.transitionTargets.body.errorSample ++
    operationNodeRefs reference.transitionTargets.body.selectorProduct ++
    operationNodeRefs reference.transitionTargets.body.targetSum ++
    [reference.transitionPreimages.parallelLoop.operation] ++
    operationNodeRefs reference.transitionPreimages.body.sample ++
    operationNodeRefs reference.transitionPreimages.body.materialize ++
    [reference.finalIndices.operation] ++ parallelGatherNodeRefs reference.finalTrapdoors

def inputInjectionNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : InputInjectionLayout) : List CoreNodeRef :=
  [reference.stateScan] ++
    initialStateExpansionNodeRefs workflow reference.initialStatesExpansion ++
    dynamicGetNodeRefs reference.selectedDigit ++
    indexFormulaNodeRefs workflow reference.sourceIndices ++
    parallelFamilyGetNodeRefs reference.sourceStates ++
    indexFormulaNodeRefs workflow reference.transitionIndices ++
    parallelFamilyGetNodeRefs reference.selectedTransitions ++
    parallelMatrixBinaryNodeRefs reference.stateProduct

def booleanLayersNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : BooleanLayersLayout) : List CoreNodeRef :=
  publicKeyBooleanLoopNodeRefs reference.encryption ++
    encodingBooleanLoopNodeRefs reference.decryption ++
    encryptDecompositionNodeRefs reference.encryptPublicKeyRhsDecomposition ++
    decryptDecompositionNodeRefs reference.decryptEncodingRhsDecomposition ++
    localBooleanGateNodeRefs reference.encryptionGate ++
    familyBooleanGateNodeRefs workflow reference.decryptionVectors ++
    familyBooleanGateNodeRefs workflow reference.decryptionPublicKeys ++
    familyBooleanGateNodeRefs workflow reference.decryptionPlaintexts

def decoderNodeRefs (reference : DecoderLayout) : List CoreNodeRef :=
  [reference.oneVector.operation, reference.kVector.operation, reference.decoderVector.operation,
    reference.oneMinusCircuit.operation, reference.projectedDifference.operation,
    reference.kPlusProjection.operation, reference.residual.operation,
    reference.extractCoefficient.operation] ++ evaluateIntNodeRefs reference.threshold ++
    [reference.lowerCompare.operation,
    reference.upperScale.operation, reference.upperScale.right.wire.node,
    reference.upperCompare.operation, reference.lowerToInt.operation,
    reference.upperToInt.operation, reference.comparisonSum.operation,
    reference.equalsTwo.operation, reference.equalsTwo.right.wire.node]

def messageNodeRefs (reference : MessageConstructionLayout) : List CoreNodeRef :=
  operationNodeRefs reference.toInt ++ operationNodeRefs reference.zero ++
    operationNodeRefs reference.one ++ operationNodeRefs reference.select

def publicKeySamplingNodeRefs (reference : BggPublicKeySamplingLayout) : List CoreNodeRef :=
  operationNodeRefs reference.packedHash ++ parallelOperationNodeRefs reference.slices

def encryptionInitialPublicKeyNodeRefs
    (reference : EncryptionInitialPublicKeysLayout) : List CoreNodeRef :=
  operationNodeRefs reference.onePublicKey ++ operationNodeRefs reference.zeroPublicKey ++
    evaluateIntNodeRefs reference.instanceWidth ++
    parallelLoopNodeRefs reference.publicIndices ++
    parallelGatherNodeRefs reference.publicCandidates ++
    parallelLoopNodeRefs reference.packedInputs.parallelLoop ++
    operationNodeRefs reference.packedInputs.inRange ++
    operationNodeRefs reference.packedInputs.padded ++
    parallelLoopNodeRefs reference.circuitInputs.parallelLoop ++
    operationNodeRefs reference.circuitInputs.selectedInstance ++
    operationNodeRefs reference.circuitInputs.selectedSource

def artifactPreprocessingNodeRefs
    (reference : DiamondArtifactPreprocessingLayout) : List CoreNodeRef :=
  operationNodeRefs reference.projectionTrapdoor.publicOperation ++
    operationNodeRefs reference.projectionTrapdoor.secret ++
    operationNodeRefs reference.oneTarget.gadget ++
    operationNodeRefs reference.oneTarget.difference ++
    operationNodeRefs reference.oneTarget.zeroRow ++ operationNodeRefs reference.oneTarget.target ++
    preimageNodeRefs reference.onePreimage ++ parallelLoopNodeRefs reference.witnessIndices ++
    parallelGatherNodeRefs reference.witnessTrapdoors ++
    parallelGatherNodeRefs reference.witnessPublicKeys ++
    parallelLoopNodeRefs reference.witnessTargets.parallelLoop ++
    operationNodeRefs reference.witnessTargets.negatedGadget ++
    operationNodeRefs reference.witnessTargets.target ++
    parallelPreimageNodeRefs reference.witnessPreimages ++
    operationNodeRefs reference.kTarget.publicKeyHash ++
    operationNodeRefs reference.kTarget.firstColumn ++
    operationNodeRefs reference.kTarget.halfModulus ++
    operationNodeRefs reference.kTarget.target ++ preimageNodeRefs reference.kPreimage ++
    operationNodeRefs reference.rHash ++ operationNodeRefs reference.rSlice ++
    operationNodeRefs reference.rDecomposition ++ operationNodeRefs reference.rMaterialization ++
    operationNodeRefs reference.rReshape ++
    operationNodeRefs reference.decoderTarget.publicKeyDifference ++
    operationNodeRefs reference.decoderTarget.projectedDifference ++
    operationNodeRefs reference.decoderTarget.publicKeySum ++
    operationNodeRefs reference.decoderTarget.zero ++
    operationNodeRefs reference.decoderTarget.target ++ preimageNodeRefs reference.decoderPreimage

def encodingComponentNodeRefs (reference : EncodingComponentOperationsLayout) : List CoreNodeRef :=
  parallelOperationNodeRefs reference.vectors ++
    parallelOperationNodeRefs reference.publicKeys ++
    parallelOperationNodeRefs reference.plaintexts

def decryptionInitialEncodingNodeRefs (workflow : Mxx.Ir.Workflow)
    (reference : DecryptionInitialEncodingsLayout) : List CoreNodeRef :=
  parallelLoopNodeRefs reference.witnessIndices ++ parallelGatherNodeRefs reference.witnessBits ++
    witnessDigitPackingNodeRefs workflow reference.witnessDigits ++
    operationNodeRefs reference.initialProjectionState ++
    operationNodeRefs reference.onePublicKey ++ operationNodeRefs reference.onePlaintext ++
    reference.zeroEncoding.flatMap operationNodeRefs ++
    parallelLoopNodeRefs reference.witnessStateIndices ++
    parallelGatherNodeRefs reference.witnessStates ++
    parallelMatrixBinaryNodeRefs reference.witnessVectors ++
    parallelLoopNodeRefs reference.witnessPublicIndices ++
    parallelGatherNodeRefs reference.witnessPublicKeys ++
    reference.witnessPlaintextConstants.flatMap parallelLoopNodeRefs ++
    parallelOperationNodeRefs reference.witnessPlaintexts ++
    evaluateIntNodeRefs reference.instanceWidth ++ parallelLoopNodeRefs reference.packedIndices ++
    parallelGatherNodeRefs reference.packedVectors ++
    parallelGatherNodeRefs reference.packedPublicKeys ++
    parallelGatherNodeRefs reference.packedPlaintexts ++
    parallelLoopNodeRefs reference.activeWitness ++
    reference.activeWitnessZeroes.flatMap parallelLoopNodeRefs ++
    encodingComponentNodeRefs reference.activeWitnessSelection ++
    reference.instanceConstants.flatMap
      (fun constants => constants.flatMap parallelLoopNodeRefs) ++
    encodingComponentNodeRefs reference.selectedInstance ++
    parallelLoopNodeRefs reference.activeInstance ++
    encodingComponentNodeRefs reference.circuitInputs

def validatedNodeRefs (workflow : Mxx.Ir.Workflow)
    (certificate : DiamondCertificate) : List CoreNodeRef :=
  (messageNodeRefs certificate.message ++
    preprocessingNodeRefs workflow certificate.inputPreprocessing ++
    publicKeySamplingNodeRefs certificate.publicKeySampling ++
    encryptionInitialPublicKeyNodeRefs certificate.encryptionInitialPublicKeys ++
    artifactPreprocessingNodeRefs certificate.artifactPreprocessing ++
    inputInjectionNodeRefs workflow certificate.inputInjection ++
    decryptionInitialEncodingNodeRefs workflow certificate.decryptionInitialEncodings ++
    booleanLayersNodeRefs workflow certificate.booleanLayers ++
    decoderNodeRefs certificate.decoder).eraseDups

def verifyDecoder (workflow : Mxx.Ir.Workflow) (workflowLayout : DiamondWorkflowLayout)
    (booleanLayers : BooleanLayersLayout) (layout : DecoderLayout) : Bool :=
  verifyMatrixBinary workflow layout.oneVector .matrixMultiply &&
    verifyMatrixBinary workflow layout.kVector .matrixMultiply &&
    verifyMatrixBinary workflow layout.decoderVector .matrixMultiply &&
    verifyMatrixBinary workflow layout.oneMinusCircuit .matrixSubtract &&
    verifyMatrixBinary workflow layout.projectedDifference .matrixMultiply &&
    verifyMatrixBinary workflow layout.kPlusProjection .matrixAdd &&
    verifyMatrixBinary workflow layout.residual .matrixSubtract &&
    [layout.oneVector.operation, layout.kVector.operation, layout.decoderVector.operation,
      layout.oneMinusCircuit.operation, layout.projectedDifference.operation,
      layout.kPlusProjection.operation, layout.residual.operation].all
      (fun operation => decide (operation.stage = workflowLayout.decryption.stage)) &&
    verifyNamedArtifactInput workflow workflowLayout "diamond_one_preimage" layout.onePreimage &&
    verifyNamedArtifactInput workflow workflowLayout "diamond_k_preimage" layout.kPreimage &&
    verifyNamedArtifactInput workflow workflowLayout "diamond_decoder_preimage"
      layout.decoderPreimage &&
    verifyNamedArtifactInput workflow workflowLayout "diamond_r_decomposed" layout.rDecomposed &&
    decide (layout.oneVector.right.wire = layout.onePreimage) &&
    decide (layout.kVector.right.wire = layout.kPreimage) &&
    decide (layout.decoderVector.right.wire = layout.decoderPreimage) &&
    decide (layout.oneVector.left.wire = layout.kVector.left.wire) &&
    decide (layout.oneVector.left.wire = layout.decoderVector.left.wire) &&
    decide (layout.selectedCircuitVector = booleanLayers.decryption.selectedVector.output) &&
    decide (layout.oneMinusCircuit.left.wire = layout.oneVector.output) &&
    decide (layout.oneMinusCircuit.right.wire = layout.selectedCircuitVector) &&
    decide (layout.projectedDifference.left.wire = layout.oneMinusCircuit.output) &&
    decide (layout.projectedDifference.right.wire = layout.rDecomposed) &&
    decide (layout.kPlusProjection.left.wire = layout.kVector.output) &&
    decide (layout.kPlusProjection.right.wire = layout.projectedDifference.output) &&
    decide (layout.residual.left.wire = layout.decoderVector.output) &&
    decide (layout.residual.right.wire = layout.kPlusProjection.output) &&
    verifyUnaryKind workflow layout.extractCoefficient (fun kind => match kind with
      | .extractCoefficient (.constant 0) => true
      | _ => false) &&
    verifyEvaluateInt workflow layout.threshold &&
    decide layout.threshold.materialization.isNone &&
    decide (layout.threshold.expression =
      .roundDivide (.subtract (.parameter "diamond_modulus") (.constant 2)) (.constant 4)) &&
    decide (layout.threshold.operation.stage = workflowLayout.decryption.stage) &&
    verifyBinaryKind workflow layout.lowerCompare (fun kind => match kind with
      | .intCompare .lessEqual => true
      | _ => false) &&
    verifyBinaryKind workflow layout.upperScale (fun kind => match kind with
      | .intBinary .multiply => true
      | _ => false) &&
    verifyBinaryKind workflow layout.upperCompare (fun kind => match kind with
      | .intCompare .lessEqual => true
      | _ => false) &&
    verifyUnaryKind workflow layout.lowerToInt (fun kind => match kind with
      | .boolToInt => true
      | _ => false) &&
    verifyUnaryKind workflow layout.upperToInt (fun kind => match kind with
      | .boolToInt => true
      | _ => false) &&
    verifyBinaryKind workflow layout.comparisonSum (fun kind => match kind with
      | .intBinary .add => true
      | _ => false) &&
    verifyBinaryKind workflow layout.equalsTwo (fun kind => match kind with
      | .intCompare .equal => true
      | _ => false) &&
    decide (layout.extractCoefficient.input.wire = layout.residual.output) &&
    decide (layout.lowerCompare.left.wire = layout.threshold.output) &&
    decide (layout.lowerCompare.right.wire = layout.extractCoefficient.output) &&
    decide (layout.upperCompare.left.wire = layout.extractCoefficient.output) &&
    decide (layout.upperScale.left.wire = layout.lowerCompare.left.wire) &&
    verifyConstantIntWire workflow layout.upperScale.right.wire 3 &&
    decide (layout.upperCompare.right.wire = layout.upperScale.output) &&
    decide (layout.lowerToInt.input.wire = layout.lowerCompare.output) &&
    decide (layout.upperToInt.input.wire = layout.upperCompare.output) &&
    decide (layout.comparisonSum.left.wire = layout.lowerToInt.output) &&
    decide (layout.comparisonSum.right.wire = layout.upperToInt.output) &&
    decide (layout.equalsTwo.left.wire = layout.comparisonSum.output) &&
    verifyConstantIntWire workflow layout.equalsTwo.right.wire 2 &&
    decide (layout.equalsTwo.output = layout.decoded) &&
    verifyNamedStageOutput workflowLayout.decryption "diamond-noisy-plaintext"
      layout.residual.output &&
    verifyNamedStageOutput workflowLayout.decryption "diamond-decoded" layout.decoded

structure ClosurePoint where
  wire : CoreWireRef
  subgraphCallers : List (ScopeRef × CoreNodeRef)
  deriving DecidableEq

def findWireIndex (target : Mxx.Ir.WireRef) : List Mxx.Ir.WireRef → Option Nat
  | [] => none
  | head :: tail => if head = target then some 0 else (findWireIndex target tail).map (· + 1)

def sameScopeArguments (reference : CoreNodeRef) (node : Mxx.Ir.Node)
    (callers : List (ScopeRef × CoreNodeRef)) : List ClosurePoint :=
  node.arguments.map fun argument => {
    wire := {
      node := { reference with node := argument.node }
      port := argument.port
    }
    subgraphCallers := callers
  }

def bodyInputDependency (workflow : Mxx.Ir.Workflow) (point : ClosurePoint) :
    Option ClosurePoint := do
  let scope ← resolveScope workflow point.wire.node
  let inputIndex ← findWireIndex (wireRef point.wire) (scopeInputWires scope)
  let caller ← match point.wire.node.scope with
    | .parallelBody parent owner | .sequentialBody parent owner =>
        some { point.wire.node with scope := parent, node := owner }
    | child@(.subgraph _) =>
        (point.subgraphCallers.find? fun entry => entry.1 = child).map (·.2)
    | .root => none
  let callerNode ← resolveNode workflow caller
  let argument ← callerNode.arguments[inputIndex]?
  return {
    wire := {
      node := { caller with node := argument.node }
      port := argument.port
    }
    subgraphCallers := point.subgraphCallers
  }

def rootInputDependencies (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout)
    (point : ClosurePoint) (name : String) : Option (List ClosurePoint) := do
  guard (point.wire.port = 0)
  let interface ← interfaceForStage layout point.wire.node.stage
  guard (interface.inputs.any fun input =>
    input.name = name && input.node = point.wire.node)
  let stage ← resolveStage workflow point.wire.node.stage
  let source ← (stage.inputs.find? fun input => input.1 = name).map (·.2)
  match source with
  | .protocol _ => return []
  | .artifact producerStage producerOutput =>
      let artifact ← layout.artifacts.find? fun artifact =>
        artifact.consumerStage = point.wire.node.stage &&
          artifact.consumerInput.name = name &&
          artifact.consumerInput.node = point.wire.node &&
          artifact.producerStage = producerStage &&
          artifact.producerOutput.name = producerOutput
      return [{ wire := artifact.producerOutput.wire, subgraphCallers := [] }]

def structuralOutputDependencies (workflow : Mxx.Ir.Workflow) (point : ClosurePoint)
    (node : Mxx.Ir.Node) : Option (List ClosurePoint) := do
  let (childScope, initial) ← match node.kind with
    | .parallelLoop definition _ _ _ _ =>
        let child := ScopeRef.parallelBody point.wire.node.scope point.wire.node.node
        guard (definition = child.definitionName)
        some (child, [])
    | .sequentialLoop definition _ _ _ _ =>
        let child := ScopeRef.sequentialBody point.wire.node.scope point.wire.node.node
        guard (definition = child.definitionName)
        let argument ← node.arguments[point.wire.port]?
        some (child, [{
          wire := {
            node := { point.wire.node with node := argument.node }
            port := argument.port
          }
          subgraphCallers := point.subgraphCallers
        }])
    | .subgraphCall definition _ => some (.subgraph definition, [])
    | _ => none
  let child ← resolveScope workflow { point.wire.node with scope := childScope }
  let output ← (scopeOutputWires child)[point.wire.port]?
  let callers := match childScope with
    | .subgraph _ => (childScope, point.wire.node) :: point.subgraphCallers
    | _ => point.subgraphCallers
  return {
    wire := {
      node := { point.wire.node with scope := childScope, node := output.node }
      port := output.port
    }
    subgraphCallers := callers
  } :: initial

def closureDependencies (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout)
    (point : ClosurePoint) : Option (Mxx.Ir.Node × List ClosurePoint) := do
  guard (verifyWire workflow point.wire)
  let node ← resolveNode workflow point.wire.node
  let dependencies ← match node.kind with
    | .input name => match point.wire.node.scope with
      | .root => rootInputDependencies workflow layout point name
      | _ => (bodyInputDependency workflow point).map (· :: [])
    | .parallelLoop .. | .sequentialLoop .. | .subgraphCall .. =>
        structuralOutputDependencies workflow point node
    | _ => some (sameScopeArguments point.wire.node node point.subgraphCallers)
  return (node, dependencies)

def genericScalarOrControl : Mxx.Ir.NodeKind → Bool
  | .constantInt _ | .evaluateInt _ | .constantBool _ | .boolToInt |
      .intBinary _ | .intCompare _ | .bitExtract _ => true
  | _ => false

def closureNodeOwned (validated : List CoreNodeRef) (reference : CoreNodeRef)
    (node : Mxx.Ir.Node) : Bool :=
  decide (reference ∈ validated) || genericScalarOrControl node.kind ||
    match node.kind with
    | .input _ => true
    | _ => false

def scopeBudget (scope : Mxx.Ir.Scope) : Nat :=
  scope.nodes.foldl (fun total node =>
    total + (node.arguments.length + 3) * (node.outputCount + 1)) 1

def stageBudget (stage : Mxx.Ir.Stage) : Nat :=
  scopeBudget stage.program.root +
    stage.program.definitions.foldl (fun total definition => total + scopeBudget definition.2) 0

def workflowClosureBudget (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout) : Nat :=
  let base := workflow.stages.foldl (fun total stage => total + stageBudget stage) 1
  base * base + 2 * layout.artifacts.length

def verifyClosureAux (workflow : Mxx.Ir.Workflow) (layout : DiamondWorkflowLayout)
    (validated : List CoreNodeRef) : Nat → List ClosurePoint → List ClosurePoint → Bool
  | _, [], _ => true
  | 0, _ :: _, _ => false
  | fuel + 1, point :: pending, visited =>
      if point ∈ visited then verifyClosureAux workflow layout validated fuel pending visited
      else match closureDependencies workflow layout point with
      | none => false
      | some (node, dependencies) =>
          closureNodeOwned validated point.wire.node node &&
            verifyClosureAux workflow layout validated fuel (dependencies ++ pending)
              (point :: visited)

def verifyOutputRootedClosure (workflow : Mxx.Ir.Workflow)
    (certificate : DiamondCertificate) : Bool :=
  verifyClosureAux workflow certificate.workflow (validatedNodeRefs workflow certificate)
    (workflowClosureBudget workflow certificate.workflow)
    [{ wire := certificate.decoder.decoded, subgraphCallers := [] }] []

def verifyDiamondCertificate (workflow : Mxx.Ir.Workflow)
    (certificate : DiamondCertificate) : Bool :=
  verifyWorkflow workflow certificate.workflow &&
    verifyMessageConstruction workflow certificate.workflow certificate.message &&
    verifyInputPreprocessing workflow certificate.workflow certificate.inputPreprocessing &&
    verifyPublicKeySampling workflow certificate.workflow certificate.publicKeySampling &&
    verifyEncryptionInitialPublicKeys workflow certificate.workflow
      certificate.publicKeySampling certificate.encryptionInitialPublicKeys &&
    verifyArtifactPreprocessing workflow certificate.workflow certificate.inputPreprocessing
      certificate.publicKeySampling certificate.booleanLayers certificate.artifactPreprocessing &&
    verifyInputInjection workflow certificate.inputInjection &&
    verifyDecryptionInitialEncodings workflow certificate.workflow certificate.inputInjection
      certificate.booleanLayers certificate.decryptionInitialEncodings &&
    verifyBooleanLayers workflow certificate.workflow certificate.booleanLayers &&
    verifyDecoder workflow certificate.workflow certificate.booleanLayers certificate.decoder &&
    verifyOutputRootedClosure workflow certificate &&
    definitionsUnique workflow &&
    verifyWorkflowSsaOrder workflow

/-- The exact checked preprocessing digit-index body computes the advertised quotient on every
selected executable SSA path.  The statement exposes no generated node number or private body
template. -/
theorem verifyPreprocessingDigitSecretIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {index width : Int}
    (verified : verifyPreprocessingDigitSecretIndexFormula workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (widthEvaluate : stateWidth.evaluate params = some width)
    (widthNonzero : width ≠ 0) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.integer (index / width)) := by
  unfold verifyPreprocessingDigitSecretIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 2 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  rw [bodyExact] at path
  rw [outputWire]
  cases path with
  | cons _ _ _ _ firstValues _ firstMember rest =>
      have firstEq : firstValues = [.integer index] := by
        simpa [exactNode, Mxx.Ir.evaluateNode, indexEvaluate] using firstMember
      subst firstValues
      cases rest with
      | cons _ _ _ _ widthValues _ widthMember rest =>
          have widthEq : widthValues = [.integer width] := by
            simpa [exactNode, Mxx.Ir.evaluateNode, widthEvaluate] using widthMember
          subst widthValues
          cases rest with
          | cons _ _ _ _ quotientValues _ quotientMember rest =>
              have divisionEvaluate :
                  Mxx.Ir.evaluateIntBinary .divide index width = some (index / width) := by
                simp [Mxx.Ir.evaluateIntBinary, widthNonzero]
              have quotientEq : quotientValues = [.integer (index / width)] := by
                simpa [exactNode, localWire, Mxx.Ir.evaluateNode, Mxx.Ir.arguments,
                  Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, divisionEvaluate] using quotientMember
              subst quotientValues
              exact rest.lookupWire_preserved
                (by simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs])

/-- The checked preprocessing source-index body masks future state slots and offsets the
remaining local slot into the selected input level. -/
theorem verifyPreprocessingSourceIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {index stride width batch : Int}
    (verified : verifyPreprocessingSourceIndexFormula workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (strideEvaluate : transitionStride.evaluate params = some stride)
    (widthEvaluate : stateWidth.evaluate params = some width)
    (batchEvaluate : batchBits.evaluate params = some batch)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.integer ((index / stride) * width +
        if (index / stride) * batch + 1 ≤ index % width then 0 else index % width)) := by
  unfold verifyPreprocessingSourceIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 17 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [preprocessingSourceIndexNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    indexEvaluate, strideEvaluate, widthEvaluate, batchEvaluate, strideNonzero,
    widthNonzero, Mxx.Ir.evaluateIntBinary, Mxx.Ir.evaluateIntCompare] at finalMember
  by_cases active : index / stride * batch < index % width
  · simp [active] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, show
      index / stride * batch + 1 ≤ index % width by omega]
  · simp [active] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, show
      ¬index / stride * batch + 1 ≤ index % width by omega]

/-- The checked preprocessing target-index body offsets the local state slot into the next input
level. -/
theorem verifyPreprocessingTargetIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {index stride width : Int}
    (verified : verifyPreprocessingTargetIndexFormula workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (strideEvaluate : transitionStride.evaluate params = some stride)
    (widthEvaluate : stateWidth.evaluate params = some width)
    (strideNonzero : stride ≠ 0) (widthNonzero : width ≠ 0) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.integer ((index / stride + 1) * width + index % width)) := by
  unfold verifyPreprocessingTargetIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 9 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [preprocessingTargetIndexNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    indexEvaluate, strideEvaluate, widthEvaluate, strideNonzero, widthNonzero,
    Mxx.Ir.evaluateIntBinary] at finalMember
  subst final
  rw [outputWire]
  simp [localWire, Mxx.Ir.lookupWire]

/-- The checked online source-index body preserves an already available state slot and maps a
future slot to the distinguished zero-state index. -/
theorem verifyOnlineSourceIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {lowerBound index : Int}
    (verified : verifyOnlineSourceIndexFormula workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (lowerBoundInput :
      Mxx.Ir.lookupEnvironment "__capture_0" inputs = some (.integer lowerBound))
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some index) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.integer (if lowerBound ≤ index then 0 else index)) := by
  unfold verifyOnlineSourceIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 9 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [onlineSourceIndexNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    lowerBoundInput, indexEvaluate, Mxx.Ir.evaluateIntBinary,
    Mxx.Ir.evaluateIntCompare] at finalMember
  by_cases future : lowerBound ≤ index
  · simp [future] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, future]
  · simp [future] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, future]

/-- The checked online transition-index body flattens `(level, digit, local-state)` in the same
order used by preprocessing. -/
theorem verifyOnlineTransitionIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : ParallelIndexFormulaRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {level digit stride width index : Int}
    (verified : verifyOnlineTransitionIndexFormula workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (levelInput : Mxx.Ir.lookupEnvironment "__capture_0" inputs = some (.integer level))
    (digitInput : Mxx.Ir.lookupEnvironment "__capture_1" inputs = some (.integer digit))
    (strideEvaluate : transitionStride.evaluate params = some stride)
    (widthEvaluate : stateWidth.evaluate params = some width)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some index) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.integer (level * stride + digit * width + index)) := by
  unfold verifyOnlineTransitionIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 8 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [onlineTransitionIndexNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    levelInput, digitInput, strideEvaluate, widthEvaluate, indexEvaluate,
    Mxx.Ir.evaluateIntBinary] at finalMember
  subst final
  rw [outputWire]
  simp [localWire, Mxx.Ir.lookupWire]

/-- The checked initial-state expansion body places the captured initial matrix at family index
zero and the exact zero matrix at every other index. -/
theorem verifyInitialStateExpansionRef_pathOutput
    {workflow : Mxx.Ir.Workflow} {reference : InitialStateExpansionRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {index : Int} {initial : Mxx.Matrix} {matrixParams : Mxx.SamplerParams}
    (verified : verifyInitialStateExpansionRef workflow reference = true)
    (bodyResolved :
      resolveScope workflow { reference.parallelLoop.operation with
        scope := reference.parallelLoop.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index)
    (initialInput :
      Mxx.Ir.lookupEnvironment "__capture_0" inputs = some (.matrix initial))
    (matrixTypeEvaluate :
      ({ modulus := .parameter "diamond_modulus"
         ringDimension := .parameter "diamond_ring_dimension"
         rows := .constant 1
         columns := .add (.constant 4) (.multiply (.constant 2)
           (.parameter "diamond_digit_count")) } : Mxx.Ir.MatrixTypeExpr).evaluate params =
        some matrixParams) :
    Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
      some (.matrix (if index = 0 then initial else
        Mxx.Matrix.withSamplerParams {
          coefficients := List.replicate
            (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension) 0
        } matrixParams)) := by
  unfold verifyInitialStateExpansionRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outputWire : wireRef reference.bodyOutput = localWire 6 := verified.1.2
  have bodyExact := verified.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [initialStateExpansionNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    indexEvaluate, initialInput, matrixTypeEvaluate, Mxx.Ir.evaluateIntCompare] at finalMember
  by_cases first : index = 0
  · simp [first] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, first]
  · simp [first] at finalMember
    subst final
    rw [outputWire]
    simp [localWire, Mxx.Ir.lookupWire, first]

/-- The checked witness-index body returns its loop index.  Keeping the private body template
behind this theorem lets execution bridges consume the verified semantics without depending on
the verifier's internal node-list representation. -/
theorem verifyDecryptionWitnessIndexFormula_pathOutput
    {workflow : Mxx.Ir.Workflow} {layout : DecryptionInitialEncodingsLayout}
    {bodyOutput : CoreWireRef} {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment} {index : Int}
    (verified : verifyDecryptionWitnessIndexFormula workflow layout = true)
    (bodyOutputs : layout.witnessIndices.bodyOutputs = [bodyOutput])
    (bodyResolved : resolveScope workflow
      { layout.witnessIndices.operation with scope := layout.witnessIndices.bodyScope } =
        some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (indexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some index) :
    Mxx.Ir.lookupWire (wireRef bodyOutput) final = some (.integer index) := by
  unfold verifyDecryptionWitnessIndexFormula at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have bodyExact := verified.1.1.2
  unfold verifyExactParallelBody at bodyExact
  rw [bodyResolved] at bodyExact
  simp only [decide_eq_true_eq] at bodyExact
  have outputWire : wireRef bodyOutput = localWire 2 := by
    rw [bodyOutputs] at verified
    simpa using verified.1.2
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [bodyExact] at finalMember
  simp [decryptionWitnessIndexNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    indexEvaluate, Mxx.Ir.evaluateIntBinary] at finalMember
  subst final
  rw [outputWire]
  simp [localWire, Mxx.Ir.lookupWire]

/-- The checked inner witness-packing scan performs one little-endian accumulator step and
doubles its carried power of two. -/
theorem verifyWitnessDigitPackingRef_scanPathOutputs
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {accumulator outerIndex batchSize bitIndex weight bitValue : Int}
    {bits : List Mxx.Ir.Value}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (accumulatorInput : Mxx.Ir.lookupEnvironment "arg-0-integer" inputs =
      some (.integer accumulator))
    (bitsInput : Mxx.Ir.lookupEnvironment "arg-2-family" inputs = some (.family bits))
    (outerInput : Mxx.Ir.lookupEnvironment "__capture_0" inputs = some (.integer outerIndex))
    (batchInput : Mxx.Ir.lookupEnvironment "__capture_1" inputs = some (.integer batchSize))
    (weightInput : Mxx.Ir.lookupEnvironment "arg-1-integer" inputs = some (.integer weight))
    (bitIndexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some bitIndex)
    (bitAt : bits[(outerIndex * batchSize + bitIndex).toNat]? = some (.integer bitValue)) :
    Mxx.Ir.lookupWire (localWire 10) final =
        some (.integer (accumulator + bitValue * weight)) ∧
      Mxx.Ir.lookupWire (localWire 12) final = some (.integer (weight * 2)) := by
  unfold verifyWitnessDigitPackingRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have scanChecked : verifyWitnessDigitScanBody workflow reference = true := by aesop
  unfold verifyWitnessDigitScanBody at scanChecked
  rw [bodyResolved] at scanChecked
  simp only [decide_eq_true_eq] at scanChecked
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [scanChecked] at finalMember
  simp [witnessDigitScanNodes, exactNode, localWire, Mxx.Ir.evaluateNodes,
    Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs,
    accumulatorInput, bitsInput, outerInput, batchInput, weightInput, bitIndexEvaluate,
    bitAt, Mxx.Ir.evaluateIntBinary] at finalMember
  subst final
  constructor <;> simp [localWire, Mxx.Ir.lookupWire]

/-- Runtime zero matrix used by the transition-selector bit body after its checked matrix type
has been evaluated.  This is a semantic value, not an exposed copy of the private node list. -/
def transitionSelectorZeroValue (matrixParams : Mxx.SamplerParams) : Mxx.Matrix :=
  Mxx.Matrix.withSamplerParams {
    coefficients := List.replicate
      (matrixParams.rows * matrixParams.columns * matrixParams.ringDimension) 0
  } matrixParams

/-- Runtime identity matrix used by the transition-selector bit body. -/
def transitionSelectorIdentityValue (matrixParams : Mxx.SamplerParams) : Mxx.Matrix :=
  Mxx.Matrix.withSamplerParams {
    coefficients := (List.range matrixParams.rows).flatMap fun row =>
      (List.range matrixParams.columns).flatMap fun column =>
        (List.range matrixParams.ringDimension).map fun coefficient =>
          if row = column ∧ coefficient = 0 then 1 else 0
  } matrixParams

/-- Pure semantics of one checked transition-selector scan iteration. -/
def transitionSelectorStepValue
    (unitParams bottomParams : Mxx.SamplerParams)
    (carried secret : Mxx.Matrix)
    (stateIndex specialIndex digit bitIndex : Int) : Mxx.Matrix :=
  let bitSet := ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0
  let bitMatrix := if bitSet then
      transitionSelectorIdentityValue unitParams
    else transitionSelectorZeroValue unitParams
  let second := Mxx.matrixMultiply secret bitMatrix
  let special := Mxx.matrixConcatRows [
    Mxx.matrixConcatColumns [secret, second],
    transitionSelectorZeroValue bottomParams]
  if stateIndex = specialIndex + bitIndex then special else carried

private theorem transitionSelectorPath_uncons
    {runChild : Mxx.Ir.ChildRunner} {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment} {inputs : Mxx.Ir.Environment}
    {nodeId : Nat} {node : Mxx.Ir.Node} {nodes : List Mxx.Ir.Node}
    {state final : Mxx.Ir.WireEnvironment}
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId
      (node :: nodes) state final) :
    ∃ values,
      values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs state node ∧
      Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs (nodeId + 1) nodes
        (state ++ Mxx.Ir.bindOutputs nodeId values) final := by
  cases path with
  | cons _ _ _ _ values _ valuesMember tail => exact ⟨values, valuesMember, tail⟩

/-- The checked private 19-node transition-selector body has the exact matrix semantics exposed
here.  Consumers reason from this theorem and never depend on the verifier's private node-list
representation. -/
theorem verifyTransitionSelector_bitPathOutput
    {workflow : Mxx.Ir.Workflow} {reference : TransitionSelectorLayout}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {unitParams bottomParams : Mxx.SamplerParams}
    {carried secret : Mxx.Matrix} {stateIndex specialIndex digit bitIndex : Int}
    (verified : verifyTransitionSelector workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (stateIndexInput : Mxx.Ir.lookupEnvironment "arg-2-integer" inputs =
      some (.integer stateIndex))
    (specialIndexInput : Mxx.Ir.lookupEnvironment "arg-3-integer" inputs =
      some (.integer specialIndex))
    (carriedInput : Mxx.Ir.lookupEnvironment "arg-0-matrix" inputs = some (.matrix carried))
    (secretInput : Mxx.Ir.lookupEnvironment "arg-4-matrix" inputs = some (.matrix secret))
    (digitInput : Mxx.Ir.lookupEnvironment "arg-1-integer" inputs = some (.integer digit))
    (bitIndexEvaluate : (Mxx.Ir.IntExpr.loopIndex 1).evaluate params = some bitIndex)
    (bitIndexNonnegative : 0 ≤ bitIndex)
    (unitTypeEvaluate : unitMatrixType.evaluate params = some unitParams)
    (bottomTypeEvaluate : (matrixType (.constant 1) (.constant 2)).evaluate params =
      some bottomParams) :
    Mxx.Ir.lookupWire (localWire 18) final = some (.matrix
      (transitionSelectorStepValue unitParams bottomParams carried secret stateIndex
        specialIndex digit bitIndex)) := by
  unfold verifyTransitionSelector at verified
  simp only [Bool.and_eq_true] at verified
  have bodyChecked : verifyTransitionSelectorBitBody workflow reference = true := by aesop
  unfold verifyTransitionSelectorBitBody at bodyChecked
  rw [bodyResolved] at bodyChecked
  simp only [decide_eq_true_eq] at bodyChecked
  rw [bodyChecked] at path
  have bitIndexNotNegative : ¬ bitIndex < 0 := by omega
  obtain ⟨values0, member0, path1⟩ := transitionSelectorPath_uncons path
  have values0Eq : values0 = [.integer stateIndex] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, stateIndexInput] using member0
  subst values0
  obtain ⟨values1, member1, path2⟩ := transitionSelectorPath_uncons path1
  have values1Eq : values1 = [.integer specialIndex] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, specialIndexInput] using member1
  subst values1
  obtain ⟨values2, member2, path3⟩ := transitionSelectorPath_uncons path2
  have values2Eq : values2 = [.integer bitIndex] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, bitIndexEvaluate] using member2
  subst values2
  obtain ⟨values3, member3, path4⟩ := transitionSelectorPath_uncons path3
  have values3Eq : values3 = [.integer (specialIndex + bitIndex)] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, Mxx.Ir.evaluateIntBinary] using member3
  subst values3
  obtain ⟨values4, member4, path5⟩ := transitionSelectorPath_uncons path4
  have values4Eq : values4 =
      [.boolean (Mxx.Ir.evaluateIntCompare .equal stateIndex (specialIndex + bitIndex))] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs] using member4
  subst values4
  obtain ⟨values5, member5, path6⟩ := transitionSelectorPath_uncons path5
  have values5Eq : values5 = [.integer (if
      Mxx.Ir.evaluateIntCompare .equal stateIndex (specialIndex + bitIndex) then 1 else 0)] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs] using member5
  subst values5
  obtain ⟨values6, member6, path7⟩ := transitionSelectorPath_uncons path6
  have values6Eq : values6 = [.matrix carried] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, carriedInput] using member6
  subst values6
  obtain ⟨values7, member7, path8⟩ := transitionSelectorPath_uncons path7
  have values7Eq : values7 = [.matrix secret] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, secretInput] using member7
  subst values7
  obtain ⟨values8, member8, path9⟩ := transitionSelectorPath_uncons path8
  have values8Eq : values8 = [.integer digit] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, digitInput] using member8
  subst values8
  obtain ⟨values9, member9, path10⟩ := transitionSelectorPath_uncons path9
  have values9Eq : values9 =
      [.boolean (((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0)] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, bitIndexEvaluate,
      bitIndexNotNegative] using member9
  subst values9
  obtain ⟨values10, member10, path11⟩ := transitionSelectorPath_uncons path10
  have values10Eq : values10 = [.integer (if
      ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0 then 1 else 0)] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs] using member10
  subst values10
  obtain ⟨values11, member11, path12⟩ := transitionSelectorPath_uncons path11
  have values11Eq : values11 = [.matrix (transitionSelectorZeroValue unitParams)] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, unitTypeEvaluate,
      transitionSelectorZeroValue] using member11
  subst values11
  obtain ⟨values12, member12, path13⟩ := transitionSelectorPath_uncons path12
  have values12Eq : values12 = [.matrix (transitionSelectorIdentityValue unitParams)] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, unitTypeEvaluate,
      transitionSelectorIdentityValue] using member12
  subst values12
  obtain ⟨values13, member13, path14⟩ := transitionSelectorPath_uncons path13
  have bitRemainderNonnegative :
      0 ≤ (digit / (2 ^ bitIndex.toNat)) % 2 := Int.emod_nonneg _ (by norm_num)
  have bitRemainderBelowTwo :
      (digit / (2 ^ bitIndex.toNat)) % 2 < 2 := Int.emod_lt_of_pos _ (by norm_num)
  have bitTestEq :
      ((digit / (2 ^ bitIndex.toNat)) % 2 ≠ 0) ↔
        (digit / (2 ^ bitIndex.toNat)) % 2 = 1 := by omega
  have evenTestEq :
      (2 : Int) ∣ digit / (2 ^ bitIndex.toNat) ↔
        (digit / (2 ^ bitIndex.toNat)) % 2 = 0 :=
    Int.dvd_iff_emod_eq_zero
  let bitMatrix := if ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0 then
    transitionSelectorIdentityValue unitParams else transitionSelectorZeroValue unitParams
  have values13Eq : values13 = [.matrix bitMatrix] := by
    have selected := Mxx.Ir.mem_evaluateNode_select_of_arguments runChild samplers params inputs
      _
      [localWire 10, localWire 11, localWire 12]
      (if ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0 then 1 else 0)
      [.matrix (transitionSelectorZeroValue unitParams),
        .matrix (transitionSelectorIdentityValue unitParams)] 1
      (by simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs]) member13
    by_cases hbit : (digit / (2 ^ bitIndex.toNat)) % 2 = 1
    · simpa [bitMatrix, hbit, bitTestEq, evenTestEq] using selected
    · have hzero : (digit / (2 ^ bitIndex.toNat)) % 2 = 0 := by omega
      simpa [bitMatrix, hbit, hzero, bitTestEq, evenTestEq] using selected
  subst values13
  obtain ⟨values14, member14, path15⟩ := transitionSelectorPath_uncons path14
  have values14Eq : values14 = [.matrix (Mxx.matrixMultiply secret bitMatrix)] := by
    apply Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs
      _
      (localWire 7) (localWire 13) secret bitMatrix 1
      (by simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs]) member14
  subst values14
  obtain ⟨values15, member15, path16⟩ := transitionSelectorPath_uncons path15
  have values15Eq : values15 = [.matrix
      (Mxx.matrixConcatColumns [secret, Mxx.matrixMultiply secret bitMatrix])] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs] using member15
  subst values15
  obtain ⟨values16, member16, path17⟩ := transitionSelectorPath_uncons path16
  have values16Eq : values16 = [.matrix (transitionSelectorZeroValue bottomParams)] := by
    simpa [transitionSelectorBitNodes, exactNode, Mxx.Ir.evaluateNode, bottomTypeEvaluate,
      transitionSelectorZeroValue] using member16
  subst values16
  obtain ⟨values17, member17, path18⟩ := transitionSelectorPath_uncons path17
  let special := Mxx.matrixConcatRows [
    Mxx.matrixConcatColumns [secret, Mxx.matrixMultiply secret bitMatrix],
    transitionSelectorZeroValue bottomParams]
  have values17Eq : values17 = [.matrix special] := by
    simpa [transitionSelectorBitNodes, exactNode, localWire, Mxx.Ir.evaluateNode,
      Mxx.Ir.arguments, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, special] using member17
  subst values17
  obtain ⟨values18, member18, path19⟩ := transitionSelectorPath_uncons path18
  have values18Eq : values18 = [.matrix (if stateIndex = specialIndex + bitIndex then
      special else carried)] := by
    have selected := Mxx.Ir.mem_evaluateNode_select_of_arguments runChild samplers params inputs
      _
      [localWire 5, localWire 6, localWire 17]
      (if Mxx.Ir.evaluateIntCompare .equal stateIndex (specialIndex + bitIndex) then 1 else 0)
      [.matrix carried, .matrix special] 1
      (by simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs]) member18
    by_cases hstate : stateIndex = specialIndex + bitIndex
    · simpa [Mxx.Ir.evaluateIntCompare, hstate] using selected
    · simpa [Mxx.Ir.evaluateIntCompare, hstate] using selected
  subst values18
  cases path19
  by_cases hbit : (digit / (2 ^ bitIndex.toNat)) % 2 = 1
  · simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, transitionSelectorStepValue,
      bitMatrix, special, hbit]
  · have hzero : (digit / (2 ^ bitIndex.toNat)) % 2 = 0 := by omega
    simp [localWire, Mxx.Ir.lookupWire, Mxx.Ir.bindOutputs, transitionSelectorStepValue,
      bitMatrix, special, hzero]

/-- The checked outer witness-packing body exposes the exact inner sequential trace whose first
carried value is emitted as the packed digit. -/
theorem verifyWitnessDigitPackingRef_outerPathTrace
    {workflow : Mxx.Ir.Workflow} {reference : WitnessDigitPackingRef}
    {body : Mxx.Ir.Scope} {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment} {final : Mxx.Ir.WireEnvironment}
    {bits : List Mxx.Ir.Value} {outerIndex batchSize : Int}
    (verified : verifyWitnessDigitPackingRef workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.parallelLoop.operation with scope := reference.parallelLoop.bodyScope } =
        some body)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 body.nodes [] final)
    (bitsInput : Mxx.Ir.lookupEnvironment "pack-bit-source" inputs = some (.family bits))
    (outerIndexEvaluate : (Mxx.Ir.IntExpr.loopIndex 0).evaluate params = some outerIndex)
    (batchEvaluate : batchBits.evaluate params = some batchSize) :
    ∃ finalValues,
      Mxx.Ir.SequentialIterationsTrace runChild reference.bitScan.bodyScope.definitionName
          params 1 [] [.family bits, .integer outerIndex, .integer batchSize]
          (List.range batchSize.toNat) [.integer 0, .integer 1] finalValues ∧
        Mxx.Ir.lookupWire (wireRef reference.bodyOutput) final =
          finalValues[0]? := by
  unfold verifyWitnessDigitPackingRef at verified
  simp only [Bool.and_eq_true, decide_eq_true_eq] at verified
  have outerChecked : verifyWitnessDigitOuterBody workflow reference = true := by aesop
  have outputWire : wireRef reference.bodyOutput = localWire 5 := by aesop
  unfold verifyWitnessDigitOuterBody at outerChecked
  unfold verifyExactParallelBody at outerChecked
  rw [bodyResolved] at outerChecked
  simp only [decide_eq_true_eq] at outerChecked
  have finalMember :
      final ∈ Mxx.Ir.evaluateNodes runChild samplers params inputs body.nodes 0 [[]] :=
    (Mxx.Ir.mem_evaluateNodes_iff_exists_path runChild samplers params inputs body.nodes 0
      [[]] final).2 ⟨[], by simp, path⟩
  rw [outerChecked] at finalMember
  simp [witnessDigitOuterNodes, exactNode, exactNodeWithOutputs, localWire,
    Mxx.Ir.evaluateNodes, Mxx.Ir.evaluateNode, Mxx.Ir.arguments, Mxx.Ir.lookupWire,
    Mxx.Ir.bindOutputs, bitsInput, outerIndexEvaluate, batchEvaluate] at finalMember
  obtain ⟨finalValues, finalValuesMember, rfl⟩ := finalMember
  refine ⟨finalValues, ?_, ?_⟩
  · obtain ⟨initial, initialMember, trace⟩ :=
      (Mxx.Ir.mem_evaluateSequentialIterations_iff_exists_trace runChild
      reference.bitScan.bodyScope.definitionName params 1 []
      [.family bits, .integer outerIndex, .integer batchSize]
      (List.range batchSize.toNat) [[.integer 0, .integer 1]] finalValues).1
      finalValuesMember
    have initialEq : initial = [.integer 0, .integer 1] := by simpa using initialMember
    simpa [initialEq] using trace
  · rw [outputWire]
    cases finalValues <;> simp [localWire, Mxx.Ir.lookupWire]

end MxxWe.Certificate
