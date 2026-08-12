import Mxx.Certificate.Rules.PointwiseFormulaExecution
import Mxx.Certificate.Rules.MatrixRules

namespace Mxx.Certificate

open scoped Matrix

/-!
# Frame-indexed pointwise matrix semantics

Unlike the earlier erased denotation, this judgment does not carry one parameter environment for
an entire expanded formula.  Every executable constructor reads the parameters of its current
runtime frame.  A subgraph call or parallel lane changes the frame before interpreting its child,
so loop bindings and lane indices cannot accidentally be evaluated in the parent environment.
-/

/-- Arithmetic meaning of a provenance-preserving formula at one exact runtime frame.  Atomic
values are trace-derived.  Boundary constructors change to the exact child frame retained by the
execution edge; arithmetic constructors never change frames. -/
inductive FrozenPointwiseMatrixProgramFormula.DenotesAt
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog} :
    {current : ExecutedScope samplers program} → FormulaExecutionFrame samplers program current →
      FrozenPointwiseMatrixProgramFormula → Mxx.Matrix → Type where
  | atom
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {matrix : Mxx.Matrix}
      (runtime : FrozenPointwiseMatrixProgramFormula.PointwiseRuntimeResult
        frame (.atom scope wire) matrix) :
      DenotesAt frame (.atom scope wire) matrix
  | inputSubstitutionSubgraph
      {parentScope : ExecutedScope samplers program}
      {parent : FormulaExecutionFrame samplers program parentScope}
      {edge : ExactSubgraphExecutionEdge parentScope}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {slot : Nat}
      {value : FrozenPointwiseMatrixProgramFormula}
      {matrix : Mxx.Matrix}
      (parentDenotes : DenotesAt parent value matrix) :
      DenotesAt (.subgraph parent edge) (.inputSubstitution scope wire slot value) matrix
  | inputSubstitutionParallel
      {parentScope : ExecutedScope samplers program}
      {parent : FormulaExecutionFrame samplers program parentScope}
      {edge : ExactParallelLaneExecutionEdge parentScope}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {slot : Nat}
      {value : FrozenPointwiseMatrixProgramFormula}
      {matrix : Mxx.Matrix}
      (parentDenotes : DenotesAt parent value matrix) :
      DenotesAt (.parallelLane parent edge) (.inputSubstitution scope wire slot value) matrix
  | zero
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams) :
      DenotesAt frame (.zero scope wire matrixType) (zeroConstantOutput matrixParams)
  | identity
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams) :
      DenotesAt frame (.identity scope wire matrixType) (identityConstantOutput matrixParams)
  | constant
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (coefficients : List Mxx.Ir.IntExpr)
      (matrixParams : Mxx.SamplerParams)
      (values : List Int)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (coefficientsEvaluate :
        coefficients.mapM (Mxx.Ir.IntExpr.evaluate current.params) = some values) :
      DenotesAt frame (.constant scope wire matrixType coefficients)
        (Mxx.Matrix.withSamplerParams {
          coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus)
        } matrixParams)
  | gadget
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (base : Mxx.Ir.IntExpr)
      (matrixParams : Mxx.SamplerParams)
      (baseValue : Int)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (baseEvaluates : base.evaluate current.params = some baseValue) :
      DenotesAt frame (.gadget scope wire matrixType base)
        (Mxx.gadgetMatrix matrixParams baseValue
          (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows))
  | decompose
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (base digitCount : Mxx.Ir.IntExpr)
      (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input output : Mxx.Matrix)
      (matrixParams : Mxx.SamplerParams)
      (baseValue digitCountValue : Int)
      (inputDenotes : DenotesAt frame inputFormula input)
      (typeEvaluates :
        matrixType.evaluate current.params (.constant 0) = some matrixParams)
      (baseEvaluates : base.evaluate current.params = some baseValue)
      (digitCountEvaluates : digitCount.evaluate current.params = some digitCountValue)
      (decompositionRelation : Mxx.MatrixModEq
        (Mxx.matrixMul
          (Mxx.gadgetMatrix {
            matrixParams with
            rows := input.rows
            columns := input.rows * digitCountValue.toNat
          } baseValue digitCountValue.toNat)
          output)
        input) :
      DenotesAt frame (.decompose scope wire matrixType base digitCount inputFormula) output
  | preimage
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire publicWire trapdoor targetWire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (cutoff : Mxx.Ir.IntExpr)
      (publicMatrix target output : Mxx.Matrix)
      (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate current.params cutoff = some matrixParams)
      (relation : Mxx.MatrixModEq (Mxx.matrixMul publicMatrix output) target)
      (bound : Mxx.maxCenteredCoefficientNorm output ≤ matrixParams.maxCoefficientBound) :
      DenotesAt frame (.preimage scope wire matrixType cutoff publicWire trapdoor targetWire)
        output
  | slice
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (rowStart rowEnd columnStart columnEnd : Mxx.Ir.IntExpr)
      (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (rowStartValue rowEndValue columnStartValue columnEndValue : Int)
      (rowStartEvaluate : rowStart.evaluate current.params = some rowStartValue)
      (rowEndEvaluate : rowEnd.evaluate current.params = some rowEndValue)
      (columnStartEvaluate : columnStart.evaluate current.params = some columnStartValue)
      (columnEndEvaluate : columnEnd.evaluate current.params = some columnEndValue) :
      DenotesAt frame (.slice scope wire (some (rowStart, rowEnd))
        (some (columnStart, columnEnd)) inputFormula)
        (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat columnStartValue.toNat
          columnEndValue.toNat)
  | sliceRows
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (rowStart rowEnd : Mxx.Ir.IntExpr) (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (rowStartValue rowEndValue : Int)
      (rowStartEvaluate : rowStart.evaluate current.params = some rowStartValue)
      (rowEndEvaluate : rowEnd.evaluate current.params = some rowEndValue) :
      DenotesAt frame (.slice scope wire (some (rowStart, rowEnd)) none inputFormula)
        (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat 0 input.columns)
  | sliceColumns
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (columnStart columnEnd : Mxx.Ir.IntExpr)
      (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (columnStartValue columnEndValue : Int)
      (columnStartEvaluate : columnStart.evaluate current.params = some columnStartValue)
      (columnEndEvaluate : columnEnd.evaluate current.params = some columnEndValue) :
      DenotesAt frame (.slice scope wire none (some (columnStart, columnEnd)) inputFormula)
        (Mxx.matrixSlice input 0 input.rows columnStartValue.toNat columnEndValue.toNat)
  | concatRows
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (leftFormula rightFormula : FrozenPointwiseMatrixProgramFormula)
      (left right : Mxx.Matrix) :
      DenotesAt frame (.concatRows scope wire leftFormula rightFormula)
        (Mxx.matrixConcatRows [left, right])
  | add
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {leftFormula rightFormula : FrozenPointwiseMatrixProgramFormula}
      {left right : Mxx.Matrix}
      (leftDenotes : DenotesAt frame leftFormula left)
      (rightDenotes : DenotesAt frame rightFormula right) :
      DenotesAt frame (.add scope wire leftFormula rightFormula) (Mxx.matrixAdd left right)
  | subtract
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {leftFormula rightFormula : FrozenPointwiseMatrixProgramFormula}
      {left right : Mxx.Matrix}
      (leftDenotes : DenotesAt frame leftFormula left)
      (rightDenotes : DenotesAt frame rightFormula right) :
      DenotesAt frame (.subtract scope wire leftFormula rightFormula)
        (Mxx.matrixSubtract left right)
  | multiply
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {leftFormula rightFormula : FrozenPointwiseMatrixProgramFormula}
      {left right : Mxx.Matrix}
      (leftDenotes : DenotesAt frame leftFormula left)
      (rightDenotes : DenotesAt frame rightFormula right) :
      DenotesAt frame (.multiply scope wire leftFormula rightFormula)
        (Mxx.matrixMultiply left right)
  | negate
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {inputFormula : FrozenPointwiseMatrixProgramFormula}
      {input : Mxx.Matrix}
      (inputDenotes : DenotesAt frame inputFormula input) :
      DenotesAt frame (.negate scope wire inputFormula) (Mxx.matrixNegate input)
  | scale
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (scalar : Mxx.Ir.IntExpr)
      (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (scalarValue : Int)
      (input : Mxx.Matrix)
      (scalarEvaluates : scalar.evaluate current.params = some scalarValue)
      (inputDenotes : DenotesAt frame inputFormula input) :
      DenotesAt frame (.scale scope wire scalar inputFormula) (Mxx.matrixScale scalarValue input)
  | scaleOne
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire : Mxx.Ir.WireRef}
      {inputFormula : FrozenPointwiseMatrixProgramFormula}
      {input : Mxx.Matrix}
      (inputDenotes : DenotesAt frame inputFormula input) :
      DenotesAt frame (.scaleOne scope wire inputFormula) input
  | select
      {current : ExecutedScope samplers program}
      {frame : FormulaExecutionFrame samplers program current}
      {scope : StaticScopeId}
      {wire indexWire : Mxx.Ir.WireRef}
      {branches : List FrozenPointwiseMatrixProgramFormula}
      {index : Int}
      {selectedFormula : FrozenPointwiseMatrixProgramFormula}
      {matrix : Mxx.Matrix}
      (indexFound : Mxx.Ir.lookupWire indexWire current.execution.wires =
        some (.integer index))
      (selectedBranch : branches[index.toNat]? = some selectedFormula)
      (selectedDenotes : DenotesAt frame selectedFormula matrix) :
      DenotesAt frame (.select scope wire indexWire branches) matrix
  | subgraphCall
      {parentScope : ExecutedScope samplers program}
      {parent : FormulaExecutionFrame samplers program parentScope}
      (edge : ExactSubgraphExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (definition : String)
      (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (outputDenotes : DenotesAt (.subgraph parent edge) output matrix) :
      DenotesAt parent (.subgraphCall scope wire definition outputPort arguments output) matrix
  | parallelLoop
      {parentScope : ExecutedScope samplers program}
      {parent : FormulaExecutionFrame samplers program parentScope}
      (edge : ExactParallelLaneExecutionEdge parentScope)
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (definition : String)
      (outputPort : Nat)
      (arguments : List FrozenPointwiseMatrixProgramFormula)
      (output : FrozenPointwiseMatrixProgramFormula)
      (matrix : Mxx.Matrix)
      (outputDenotes : DenotesAt (.parallelLane parent edge) output matrix) :
      DenotesAt parent (.parallelLoop scope wire definition outputPort arguments output) matrix

/-- A normalized pointwise formula together with the exact executable representative from the
same runtime frame. -/
structure FrozenPointwiseMatrixProgramFormula.SemanticResultAt
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (formula : FrozenPointwiseMatrixProgramFormula)
    (q ringDimension rows columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (runtimeValue : Mxx.Matrix) : Type where
  normalizedValue : Mxx.Matrix
  normalizedDenotes : formula.DenotesAt frame normalizedValue
  runtimeLayout : Mxx.Toolkit.MatrixLayout runtimeValue q ringDimension rows columns
  normalizedLayout : Mxx.Toolkit.MatrixLayout normalizedValue q ringDimension rows columns
  runtimeEquation : Mxx.MatrixModEq runtimeValue normalizedValue

/-- A frame-indexed semantic result identifies the executable value and its normalized formula
value in the exact negacyclic quotient.  This is the only bridge needed by quotient-level BGG
algebra; it never strengthens modular equality to equality of stored integer representatives. -/
theorem FrozenPointwiseMatrixProgramFormula.SemanticResultAt.matrixValue_eq
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {runtimeValue : Mxx.Matrix}
    (result : formula.SemanticResultAt frame q ringDimension rows columns runtimeValue) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns runtimeValue =
      Mxx.Toolkit.matrixValue q ringDimension rows columns result.normalizedValue :=
  Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension rows columns runtimeValue
    result.normalizedValue result.runtimeLayout result.normalizedLayout result.runtimeEquation

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.refl
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {value : Mxx.Matrix}
    (denotes : formula.DenotesAt frame value)
    (layout : Mxx.Toolkit.MatrixLayout value q ringDimension rows columns) :
    formula.SemanticResultAt frame q ringDimension rows columns value := {
  normalizedValue := value
  normalizedDenotes := denotes
  runtimeLayout := layout
  normalizedLayout := layout
  runtimeEquation := Mxx.MatrixModEq.refl value
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.transport
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {formula : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {sourceValue runtimeValue : Mxx.Matrix}
    (result : formula.SemanticResultAt frame q ringDimension rows columns sourceValue)
    (runtimeLayout : Mxx.Toolkit.MatrixLayout runtimeValue q ringDimension rows columns)
    (runtimeToSource : Mxx.MatrixModEq runtimeValue sourceValue) :
    formula.SemanticResultAt frame q ringDimension rows columns runtimeValue := {
  normalizedValue := result.normalizedValue
  normalizedDenotes := result.normalizedDenotes
  runtimeLayout
  normalizedLayout := result.normalizedLayout
  runtimeEquation := runtimeToSource.trans result.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.zero
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {matrixParams : Mxx.SamplerParams}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = rows)
    (columnsEq : matrixParams.columns = columns) :
    (FrozenPointwiseMatrixProgramFormula.zero scope wire matrixType).SemanticResultAt
      frame q ringDimension rows columns (zeroConstantOutput matrixParams) := by
  apply SemanticResultAt.refl (.zero scope wire matrixType matrixParams typeEvaluates)
  simpa only [zeroConstantOutput, modulusEq, ringDimensionEq, rowsEq, columnsEq] using
    Mxx.Toolkit.withSamplerParams_layout
      { coefficients := List.replicate (matrixParams.rows * matrixParams.columns *
          matrixParams.ringDimension) 0 } matrixParams

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.identity
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {matrixParams : Mxx.SamplerParams}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = rows)
    (columnsEq : matrixParams.columns = columns) :
    (FrozenPointwiseMatrixProgramFormula.identity scope wire matrixType).SemanticResultAt
      frame q ringDimension rows columns (identityConstantOutput matrixParams) := by
  apply SemanticResultAt.refl (.identity scope wire matrixType matrixParams typeEvaluates)
  simpa only [identityConstantOutput, modulusEq, ringDimensionEq, rowsEq, columnsEq] using
    Mxx.Toolkit.withSamplerParams_layout _ matrixParams

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.constant
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {coefficients : List Mxx.Ir.IntExpr}
    {matrixParams : Mxx.SamplerParams}
    {values : List Int}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
    (coefficientsEvaluate :
      coefficients.mapM (Mxx.Ir.IntExpr.evaluate current.params) = some values)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = rows)
    (columnsEq : matrixParams.columns = columns) :
    (FrozenPointwiseMatrixProgramFormula.constant scope wire matrixType coefficients).SemanticResultAt
      frame q ringDimension rows columns
      (Mxx.Matrix.withSamplerParams {
        coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus)
      } matrixParams) := by
  apply SemanticResultAt.refl
    (.constant scope wire matrixType coefficients matrixParams values typeEvaluates
      coefficientsEvaluate)
  simpa [modulusEq, ringDimensionEq, rowsEq, columnsEq] using
    Mxx.Toolkit.withSamplerParams_layout
      { coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus) } matrixParams

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.gadget
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {base : Mxx.Ir.IntExpr}
    {matrixParams : Mxx.SamplerParams}
    {baseValue : Int}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
    (baseEvaluates : base.evaluate current.params = some baseValue)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = rows)
    (columnsEq : matrixParams.rows *
      (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows) = columns) :
    (FrozenPointwiseMatrixProgramFormula.gadget scope wire matrixType base).SemanticResultAt
      frame q ringDimension rows columns
      (Mxx.gadgetMatrix matrixParams baseValue
        (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows)) := by
  apply SemanticResultAt.refl
    (.gadget scope wire matrixType base matrixParams baseValue typeEvaluates baseEvaluates)
  have layout := Mxx.Toolkit.gadgetMatrix_layout matrixParams baseValue
    (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows)
  have columnsEq' : rows *
      (if rows = 0 then 0 else matrixParams.columns / rows) = columns := by
    simpa only [rowsEq] using columnsEq
  rw [modulusEq, ringDimensionEq, rowsEq, columnsEq'] at layout
  simpa only [rowsEq] using layout

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.decompose
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {base digitCount : Mxx.Ir.IntExpr}
    {input : FrozenPointwiseMatrixProgramFormula}
    {inputValue output : Mxx.Matrix}
    {matrixParams : Mxx.SamplerParams}
    {baseValue digitCountValue : Int}
    {q ringDimension inputRows inputColumns outputRows outputColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (inputResult : input.SemanticResultAt frame q ringDimension inputRows inputColumns inputValue)
    (typeEvaluates :
      matrixType.evaluate current.params (.constant 0) = some matrixParams)
    (baseEvaluates : base.evaluate current.params = some baseValue)
    (digitCountEvaluates : digitCount.evaluate current.params = some digitCountValue)
    (outputMember : output ∈ samplers.gadgetDecompose matrixParams baseValue
      digitCountValue.toNat inputValue)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = outputRows)
    (columnsEq : matrixParams.columns = outputColumns) :
    (FrozenPointwiseMatrixProgramFormula.decompose scope wire matrixType base digitCount input).SemanticResultAt
      frame q ringDimension outputRows outputColumns
        (output.withSamplerParams matrixParams) := by
  have actualRelation :=
    (contract.gadgetDecomposeContract matrixParams baseValue digitCountValue.toNat
      inputValue output outputMember).1
  have normalizedRelation : Mxx.MatrixModEq
      (Mxx.matrixMul
        (Mxx.gadgetMatrix {
          matrixParams with
          rows := inputResult.normalizedValue.rows
          columns := inputResult.normalizedValue.rows * digitCountValue.toNat
        } baseValue digitCountValue.toNat)
        (output.withSamplerParams matrixParams))
      inputResult.normalizedValue := by
    have relationToNormalized := actualRelation.trans inputResult.runtimeEquation
    simpa only [inputResult.runtimeEquation.rows] using relationToNormalized
  apply SemanticResultAt.refl
    (.decompose scope wire matrixType base digitCount input inputResult.normalizedValue
      (output.withSamplerParams matrixParams) matrixParams baseValue digitCountValue
      inputResult.normalizedDenotes typeEvaluates baseEvaluates digitCountEvaluates
      normalizedRelation)
  simpa [modulusEq, ringDimensionEq, rowsEq, columnsEq] using
    Mxx.Toolkit.withSamplerParams_layout output matrixParams

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.preimage
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire publicWire trapdoor targetWire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {cutoff : Mxx.Ir.IntExpr}
    {publicMatrix target sample : Mxx.Matrix}
    {matrixParams : Mxx.SamplerParams}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (typeEvaluates : matrixType.evaluate current.params cutoff = some matrixParams)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul publicMatrix
      (sample.withSamplerParams matrixParams)) target)
    (bound : Mxx.maxCenteredCoefficientNorm (sample.withSamplerParams matrixParams) ≤
      matrixParams.maxCoefficientBound)
    (modulusEq : matrixParams.modulus = q)
    (ringDimensionEq : matrixParams.ringDimension = ringDimension)
    (rowsEq : matrixParams.rows = rows)
    (columnsEq : matrixParams.columns = columns) :
    (FrozenPointwiseMatrixProgramFormula.preimage scope wire matrixType cutoff publicWire
      trapdoor targetWire).SemanticResultAt frame q ringDimension rows columns
        (sample.withSamplerParams matrixParams) := by
  apply SemanticResultAt.refl
    (.preimage scope wire publicWire trapdoor targetWire matrixType cutoff publicMatrix target
      (sample.withSamplerParams matrixParams) matrixParams typeEvaluates relation bound)
  simpa [modulusEq, ringDimensionEq, rowsEq, columnsEq] using
    Mxx.Toolkit.withSamplerParams_layout sample matrixParams

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.add
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {leftValue rightValue : Mxx.Matrix}
    (leftResult : left.SemanticResultAt frame q ringDimension rows columns leftValue)
    (rightResult : right.SemanticResultAt frame q ringDimension rows columns rightValue) :
    (FrozenPointwiseMatrixProgramFormula.add scope wire left right).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixAdd leftValue rightValue) := {
  normalizedValue := Mxx.matrixAdd leftResult.normalizedValue rightResult.normalizedValue
  normalizedDenotes := .add leftResult.normalizedDenotes rightResult.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixAdd_layout leftValue rightValue
    leftResult.runtimeLayout rightResult.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixAdd_layout
    leftResult.normalizedValue rightResult.normalizedValue
    leftResult.normalizedLayout rightResult.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.add q ringDimension rows columns
    leftValue leftResult.normalizedValue rightValue rightResult.normalizedValue
    leftResult.runtimeLayout leftResult.normalizedLayout rightResult.runtimeLayout
    rightResult.normalizedLayout leftResult.runtimeEquation rightResult.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.subtract
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {leftValue rightValue : Mxx.Matrix}
    (leftResult : left.SemanticResultAt frame q ringDimension rows columns leftValue)
    (rightResult : right.SemanticResultAt frame q ringDimension rows columns rightValue) :
    (FrozenPointwiseMatrixProgramFormula.subtract scope wire left right).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixSubtract leftValue rightValue) := {
  normalizedValue := Mxx.matrixSubtract leftResult.normalizedValue rightResult.normalizedValue
  normalizedDenotes := .subtract leftResult.normalizedDenotes rightResult.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixSubtract_layout leftValue rightValue
    leftResult.runtimeLayout rightResult.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixSubtract_layout
    leftResult.normalizedValue rightResult.normalizedValue
    leftResult.normalizedLayout rightResult.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.subtract q ringDimension rows columns
    leftValue leftResult.normalizedValue rightValue rightResult.normalizedValue
    leftResult.runtimeLayout leftResult.normalizedLayout rightResult.runtimeLayout
    rightResult.normalizedLayout leftResult.runtimeEquation rightResult.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.multiply
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows inner columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {leftValue rightValue : Mxx.Matrix}
    (leftResult : left.SemanticResultAt frame q ringDimension rows inner leftValue)
    (rightResult : right.SemanticResultAt frame q ringDimension inner columns rightValue) :
    (FrozenPointwiseMatrixProgramFormula.multiply scope wire left right).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixMultiply leftValue rightValue) := {
  normalizedValue := Mxx.matrixMultiply leftResult.normalizedValue rightResult.normalizedValue
  normalizedDenotes := .multiply leftResult.normalizedDenotes rightResult.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixMultiply_layout leftValue rightValue
    leftResult.runtimeLayout rightResult.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixMultiply_layout
    leftResult.normalizedValue rightResult.normalizedValue
    leftResult.normalizedLayout rightResult.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.multiply q ringDimension rows inner columns
    leftValue leftResult.normalizedValue rightValue rightResult.normalizedValue
    leftResult.runtimeLayout leftResult.normalizedLayout rightResult.runtimeLayout
    rightResult.normalizedLayout leftResult.runtimeEquation rightResult.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.multiplyLeftBroadcast
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {leftValue rightValue : Mxx.Matrix}
    (leftResult : left.SemanticResultAt frame q ringDimension 1 1 leftValue)
    (rightResult : right.SemanticResultAt frame q ringDimension rows columns rightValue) :
    (FrozenPointwiseMatrixProgramFormula.multiply scope wire left right).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixMultiply leftValue rightValue) := {
  normalizedValue := Mxx.matrixMultiply leftResult.normalizedValue rightResult.normalizedValue
  normalizedDenotes := .multiply leftResult.normalizedDenotes rightResult.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixMultiply_leftBroadcast_layout leftValue rightValue
    leftResult.runtimeLayout rightResult.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixMultiply_leftBroadcast_layout
    leftResult.normalizedValue rightResult.normalizedValue
    leftResult.normalizedLayout rightResult.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.multiplyLeftBroadcast q ringDimension rows columns
    leftValue leftResult.normalizedValue rightValue rightResult.normalizedValue
    leftResult.runtimeLayout leftResult.normalizedLayout rightResult.runtimeLayout
    rightResult.normalizedLayout leftResult.runtimeEquation rightResult.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.multiplyRightBroadcast
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {leftValue rightValue : Mxx.Matrix}
    (leftResult : left.SemanticResultAt frame q ringDimension rows columns leftValue)
    (rightResult : right.SemanticResultAt frame q ringDimension 1 1 rightValue) :
    (FrozenPointwiseMatrixProgramFormula.multiply scope wire left right).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixMultiply leftValue rightValue) := {
  normalizedValue := Mxx.matrixMultiply leftResult.normalizedValue rightResult.normalizedValue
  normalizedDenotes := .multiply leftResult.normalizedDenotes rightResult.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixMultiply_rightBroadcast_layout leftValue rightValue
    leftResult.runtimeLayout rightResult.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixMultiply_rightBroadcast_layout
    leftResult.normalizedValue rightResult.normalizedValue
    leftResult.normalizedLayout rightResult.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.multiplyRightBroadcast q ringDimension rows columns
    leftValue leftResult.normalizedValue rightValue rightResult.normalizedValue
    leftResult.runtimeLayout leftResult.normalizedLayout rightResult.runtimeLayout
    rightResult.normalizedLayout leftResult.runtimeEquation rightResult.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.negate
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {input : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {inputValue : Mxx.Matrix}
    (result : input.SemanticResultAt frame q ringDimension rows columns inputValue) :
    (FrozenPointwiseMatrixProgramFormula.negate scope wire input).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixNegate inputValue) := {
  normalizedValue := Mxx.matrixNegate result.normalizedValue
  normalizedDenotes := .negate result.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixNegate_layout inputValue result.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixNegate_layout result.normalizedValue
    result.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.negate q ringDimension rows columns
    inputValue result.normalizedValue result.runtimeLayout result.normalizedLayout
    result.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.scale
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {scalar : Mxx.Ir.IntExpr}
    {input : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {scalarValue : Int}
    {inputValue : Mxx.Matrix}
    (scalarEvaluates : scalar.evaluate current.params = some scalarValue)
    (result : input.SemanticResultAt frame q ringDimension rows columns inputValue) :
    (FrozenPointwiseMatrixProgramFormula.scale scope wire scalar input).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixScale scalarValue inputValue) := {
  normalizedValue := Mxx.matrixScale scalarValue result.normalizedValue
  normalizedDenotes := .scale scope wire scalar input scalarValue result.normalizedValue
    scalarEvaluates result.normalizedDenotes
  runtimeLayout := Mxx.Toolkit.matrixScale_layout scalarValue inputValue result.runtimeLayout
  normalizedLayout := Mxx.Toolkit.matrixScale_layout scalarValue result.normalizedValue
    result.normalizedLayout
  runtimeEquation := Mxx.Toolkit.MatrixModEq.scale q ringDimension rows columns scalarValue
    inputValue result.normalizedValue result.runtimeLayout result.normalizedLayout
    result.runtimeEquation
}

def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.scaleOne
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {input : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {inputValue : Mxx.Matrix}
    (result : input.SemanticResultAt frame q ringDimension rows columns inputValue) :
    (FrozenPointwiseMatrixProgramFormula.scaleOne scope wire input).SemanticResultAt
      frame q ringDimension rows columns (Mxx.matrixScale 1 inputValue) := by
  let scaledLayout := Mxx.Toolkit.matrixScale_layout 1 inputValue result.runtimeLayout
  let scaledInputEquation : Mxx.MatrixModEq (Mxx.matrixScale 1 inputValue) inputValue :=
    Mxx.Toolkit.modEq_of_matrixValue_eq q ringDimension rows columns
      (Mxx.matrixScale 1 inputValue) inputValue scaledLayout result.runtimeLayout
      (matrixScaleOne_local_sound q ringDimension rows columns inputValue result.runtimeLayout)
  exact {
    normalizedValue := result.normalizedValue
    normalizedDenotes := .scaleOne result.normalizedDenotes
    runtimeLayout := scaledLayout
    normalizedLayout := result.normalizedLayout
    runtimeEquation := scaledInputEquation.trans result.runtimeEquation
  }

/-- Selection preserves the selected branch's normalized value.  The branch is chosen by the
actual integer wire in the current execution frame, not by certificate-supplied control data. -/
def FrozenPointwiseMatrixProgramFormula.SemanticResultAt.select
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {scope : StaticScopeId}
    {wire indexWire : Mxx.Ir.WireRef}
    {branches : List FrozenPointwiseMatrixProgramFormula}
    {index : Int}
    {selectedFormula : FrozenPointwiseMatrixProgramFormula}
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {selectedValue : Mxx.Matrix}
    (indexFound : Mxx.Ir.lookupWire indexWire current.execution.wires =
      some (.integer index))
    (selectedBranch : branches[index.toNat]? = some selectedFormula)
    (result : selectedFormula.SemanticResultAt frame q ringDimension rows columns selectedValue) :
    (FrozenPointwiseMatrixProgramFormula.select scope wire indexWire branches).SemanticResultAt
      frame q ringDimension rows columns selectedValue := {
  normalizedValue := result.normalizedValue
  normalizedDenotes := .select indexFound selectedBranch result.normalizedDenotes
  runtimeLayout := result.runtimeLayout
  normalizedLayout := result.normalizedLayout
  runtimeEquation := result.runtimeEquation
}

end Mxx.Certificate
