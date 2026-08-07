import Mxx.Certificate.Rules.PointwiseFormulaSemantics

namespace Mxx.Certificate

/-!
# Trace-derived pointwise formula boundaries

These constructors connect normalized formula semantics across executable scope boundaries.  Every
boundary is justified by `PointwiseRuntimeResult`, whose constructors retain the selected SSA node,
the exact child execution, and the corresponding input or output lookup.  Consequently none of the
definitions below accepts a caller-provided value resolver.
-/

/-- The selected formal-input wire has the value found in the child execution's real input
environment. -/
theorem inputSubstitutionRuntimeLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (wire : Mxx.Ir.WireRef)
    (name : String)
    (matrix : Mxx.Matrix)
    (nodeInBounds : wire.node < current.execution.scope.nodes.length)
    (outputCount : Nat)
    (nodeEq : current.execution.scope.nodes[wire.node] = {
      kind := .input name
      arguments := []
      outputCount
      outputTypes := current.execution.scope.nodes[wire.node].outputTypes
    })
    (formalInputFound : Mxx.Ir.lookupEnvironment name
      (current.execution.scope.inputNames.zip current.arguments) = some (.matrix matrix))
    (portInBounds : wire.port < outputCount) :
    Mxx.Ir.lookupWire wire current.execution.wires = some (.matrix matrix) := by
  exact current.execution.formalInputMatrixLookup wire.node nodeInBounds name outputCount wire.port
    matrix nodeEq formalInputFound portInBounds

/-- The selected child output is exactly the value returned at the parent subgraph-call port. -/
theorem subgraphCallRuntimeLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parent : ExecutedScope samplers program}
    (edge : ExactSubgraphExecutionEdge parent)
    (outputPort : Nat)
    (outputName : String)
    (outputWire : Mxx.Ir.WireRef)
    (matrix : Mxx.Matrix)
    (outputEntry : edge.child.execution.scope.outputs[outputPort]? =
      some (outputName, outputWire))
    (childOutputFound : Mxx.Ir.lookupWire outputWire edge.child.execution.wires =
      some (.matrix matrix)) :
    Mxx.Ir.lookupWire ⟨edge.nodeIndex, outputPort⟩ parent.execution.wires =
      some (.matrix matrix) := by
  let selected := parent.execution.nodeExecutionAt edge.nodeIndex edge.nodeInBounds
  have returnedValue : selected.nodeValues[outputPort]? = some (.matrix matrix) := by
    rw [← edge.outputsEq, edge.child.execution.outputEq]
    simp [Mxx.Ir.collectOutputs, List.getElem?_map, outputEntry, childOutputFound]
  have outputPortInBounds : outputPort < selected.nodeValues.length :=
    List.getElem?_eq_some_iff.mp returnedValue |>.1
  have outputLookup := selected.outputLookup outputPort outputPortInBounds
  have returnedValue' : selected.nodeValues[outputPort] = .matrix matrix :=
    Option.some.inj ((List.getElem?_eq_getElem outputPortInBounds).symm.trans returnedValue)
  simpa [returnedValue'] using outputLookup

/-- The selected parallel coordinate is the exact matrix returned by that child and stored in the
parent's gathered family. -/
theorem parallelLoopElementRuntimeEqualities
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parent : ExecutedScope samplers program}
    (edge : ExactParallelLaneExecutionEdge parent)
    (outputPort : Nat)
    (portInBounds : outputPort < edge.view.outputCount)
    (gathered : List Mxx.Ir.Value)
    (finalPort : edge.trace.final[outputPort]? = some gathered)
    (outputName : String)
    (outputWire : Mxx.Ir.WireRef)
    (matrix : Mxx.Matrix)
    (outputEntry : edge.child.execution.scope.outputs[outputPort]? =
      some (outputName, outputWire))
    (childOutputFound : Mxx.Ir.lookupWire outputWire edge.child.execution.wires =
      some (.matrix matrix))
    (coordinateOutput : edge.child.outputs[outputPort]? = gathered[edge.position]?) :
    Mxx.Ir.lookupWire ⟨edge.nodeIndex, outputPort⟩ parent.execution.wires =
        some (.family gathered) ∧
      gathered[edge.position]? = some (.matrix matrix) := by
  have parentLookup := edge.trace.parentFamilyLookup outputPort portInBounds gathered finalPort
  have childReturned : edge.child.outputs[outputPort]? = some (.matrix matrix) := by
    rw [edge.child.execution.outputEq]
    simp [Mxx.Ir.collectOutputs, List.getElem?_map, outputEntry, childOutputFound]
  exact ⟨parentLookup, coordinateOutput.symm.trans childReturned⟩

/-- An atomic formula denotes the matrix selected by its exact runtime trace evidence. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.atom
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {matrix : Mxx.Matrix}
    (runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult
      frame (.atom scope wire) matrix)
    (layout : Mxx.Toolkit.MatrixLayout matrix q ringDimension rows columns) :
    (FrozenPointwiseMatrixProgramFormula.atom scope wire).SemanticResultAt
      frame q ringDimension rows columns matrix :=
  SemanticResultAt.refl (.atom runtime) layout

/-- Transport a parent formula through the exact formal input selected by a subgraph call. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.inputSubstitutionSubgraph
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parentScope : ExecutedScope samplers program}
    {parent : FormulaExecutionFrame samplers program parentScope}
    {edge : ExactSubgraphExecutionEdge parentScope}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {slot : Nat}
    {value : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {matrix : Mxx.Matrix}
    (_runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult
      (.subgraph parent edge) (.inputSubstitution scope wire slot value) matrix)
    (parentResult : value.SemanticResultAt parent q ringDimension rows columns matrix) :
    (FrozenPointwiseMatrixProgramFormula.inputSubstitution scope wire slot value).SemanticResultAt
      (.subgraph parent edge) q ringDimension rows columns matrix := {
  normalizedValue := parentResult.normalizedValue
  normalizedDenotes := .inputSubstitutionSubgraph parentResult.normalizedDenotes
  runtimeLayout := parentResult.runtimeLayout
  normalizedLayout := parentResult.normalizedLayout
  runtimeEquation := parentResult.runtimeEquation
}

/-- Transport a parent formula through the exact formal input selected by one parallel lane. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.inputSubstitutionParallel
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parentScope : ExecutedScope samplers program}
    {parent : FormulaExecutionFrame samplers program parentScope}
    {edge : ExactParallelLaneExecutionEdge parentScope}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {slot : Nat}
    {value : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {matrix : Mxx.Matrix}
    (_runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult
      (.parallelLane parent edge) (.inputSubstitution scope wire slot value) matrix)
    (parentResult : value.SemanticResultAt parent q ringDimension rows columns matrix) :
    (FrozenPointwiseMatrixProgramFormula.inputSubstitution scope wire slot value).SemanticResultAt
      (.parallelLane parent edge) q ringDimension rows columns matrix := {
  normalizedValue := parentResult.normalizedValue
  normalizedDenotes := .inputSubstitutionParallel parentResult.normalizedDenotes
  runtimeLayout := parentResult.runtimeLayout
  normalizedLayout := parentResult.normalizedLayout
  runtimeEquation := parentResult.runtimeEquation
}

/-- Return an exact child subgraph output to the selected parent SSA call node. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.subgraphCall
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parentScope : ExecutedScope samplers program}
    {parent : FormulaExecutionFrame samplers program parentScope}
    {edge : ExactSubgraphExecutionEdge parentScope}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {definition : String}
    {outputPort : Nat}
    {arguments : List FrozenPointwiseMatrixProgramFormula}
    {output : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {matrix : Mxx.Matrix}
    (_runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult parent
      (.subgraphCall scope wire definition outputPort arguments output) matrix)
    (childResult : output.SemanticResultAt (.subgraph parent edge)
      q ringDimension rows columns matrix) :
    (FrozenPointwiseMatrixProgramFormula.subgraphCall scope wire definition outputPort arguments
      output).SemanticResultAt parent q ringDimension rows columns matrix := {
  normalizedValue := childResult.normalizedValue
  normalizedDenotes := .subgraphCall edge scope wire definition outputPort arguments output
    childResult.normalizedValue childResult.normalizedDenotes
  runtimeLayout := childResult.runtimeLayout
  normalizedLayout := childResult.normalizedLayout
  runtimeEquation := childResult.runtimeEquation
}

/-- Return one exact child output coordinate as the selected element of its parent parallel family. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.parallelLoopElement
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {parentScope : ExecutedScope samplers program}
    {parent : FormulaExecutionFrame samplers program parentScope}
    {edge : ExactParallelLaneExecutionEdge parentScope}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {definition : String}
    {outputPort : Nat}
    {arguments : List FrozenPointwiseMatrixProgramFormula}
    {output : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {matrix : Mxx.Matrix}
    (_runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult parent
      (.parallelLoop scope wire definition outputPort arguments output) matrix)
    (childResult : output.SemanticResultAt (.parallelLane parent edge)
      q ringDimension rows columns matrix) :
    (FrozenPointwiseMatrixProgramFormula.parallelLoop scope wire definition outputPort arguments
      output).SemanticResultAt parent q ringDimension rows columns matrix := {
  normalizedValue := childResult.normalizedValue
  normalizedDenotes := .parallelLoop edge scope wire definition outputPort arguments output
    childResult.normalizedValue childResult.normalizedDenotes
  runtimeLayout := childResult.runtimeLayout
  normalizedLayout := childResult.normalizedLayout
  runtimeEquation := childResult.runtimeEquation
}

end Mxx.Certificate
