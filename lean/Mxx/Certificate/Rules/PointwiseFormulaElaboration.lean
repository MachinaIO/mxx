import Mxx.Certificate.Rules.PointwiseFormulaValidation
import Mxx.Certificate.Rules.PointwiseFormulaSemantics

namespace Mxx.Certificate

/-!
# Local pointwise-formula elaboration

This module composes closed formula validation with one exact executed scope.  Inputs contain only
kernel-checked parameter evaluations and layouts for opaque leaves.  Executable node selection,
wire values, arithmetic equations, and decomposition support are recovered from the frozen
program and the actual execution trace.
-/

/-- The static scope named by the current frame is exactly the scope traversed by its execution
path.  Boundary elaboration constructs this fact when it constructs a child frame. -/
structure LocalFormulaFrameValid
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program) : Prop where
  scopeFound : scopeAtStaticPath? program current.scopeId = some current.execution.scope

/-- Non-provenance inputs needed to elaborate a local arithmetic formula. -/
inductive FrozenPointwiseMatrixProgramFormula.LocalElaborationInputs
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (q ringDimension : Nat)
    [Fact (1 < q)] [NeZero ringDimension] :
    FrozenPointwiseMatrixProgramFormula → Nat → Nat → Type where
  | atom
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrix : Mxx.Matrix)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (found : Mxx.Ir.lookupWire wire current.execution.wires = some (.matrix matrix))
      (layout : Mxx.Toolkit.MatrixLayout matrix q ringDimension rows columns) :
      LocalElaborationInputs frame q ringDimension (.atom scope wire) rows columns
  | zero
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (matrixParams : Mxx.SamplerParams)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = rows)
      (columnsEq : matrixParams.columns = columns) :
      LocalElaborationInputs frame q ringDimension (.zero scope wire matrixType) rows columns
  | identity
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (matrixParams : Mxx.SamplerParams)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = rows)
      (columnsEq : matrixParams.columns = columns) :
      LocalElaborationInputs frame q ringDimension (.identity scope wire matrixType) rows columns
  | constant
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (coefficients : List Mxx.Ir.IntExpr)
      (matrixParams : Mxx.SamplerParams)
      (values : List Int)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (coefficientsEvaluate :
        coefficients.mapM (Mxx.Ir.IntExpr.evaluate current.params) = some values)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = rows)
      (columnsEq : matrixParams.columns = columns) :
      LocalElaborationInputs frame q ringDimension
        (.constant scope wire matrixType coefficients) rows columns
  | gadget
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (base : Mxx.Ir.IntExpr)
      (matrixParams : Mxx.SamplerParams)
      (baseValue : Int)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (typeEvaluates : matrixType.evaluate current.params = some matrixParams)
      (baseEvaluates : base.evaluate current.params = some baseValue)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = rows)
      (columnsEq : matrixParams.rows *
        (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows) = columns) :
      LocalElaborationInputs frame q ringDimension (.gadget scope wire matrixType base)
        rows columns
  | decompose
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (base digitCount : Mxx.Ir.IntExpr)
      (input : FrozenPointwiseMatrixProgramFormula)
      (matrixParams : Mxx.SamplerParams)
      (baseValue digitCountValue : Int)
      (inputRows inputColumns outputRows outputColumns : Nat)
      (scopeEq : scope = current.scopeId)
      (inputInputs : LocalElaborationInputs frame q ringDimension input inputRows inputColumns)
      (typeEvaluates : matrixType.evaluate current.params (.constant 0) = some matrixParams)
      (baseEvaluates : base.evaluate current.params = some baseValue)
      (digitCountEvaluates : digitCount.evaluate current.params = some digitCountValue)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = outputRows)
      (columnsEq : matrixParams.columns = outputColumns) :
      LocalElaborationInputs frame q ringDimension
        (.decompose scope wire matrixType base digitCount input) outputRows outputColumns
  | preimage
      (scope : StaticScopeId)
      (wire publicWire trapdoor targetWire : Mxx.Ir.WireRef)
      (matrixType : MatrixTypeExpr)
      (cutoff : Mxx.Ir.IntExpr)
      (publicMatrix target : Mxx.Matrix)
      (matrixParams : Mxx.SamplerParams)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (publicEarlier : publicWire.node < wire.node)
      (trapdoorEarlier : trapdoor.node < wire.node)
      (targetEarlier : targetWire.node < wire.node)
      (publicFound : Mxx.Ir.lookupWire publicWire current.execution.wires =
        some (.matrix publicMatrix))
      (trapdoorFound : Mxx.Ir.lookupWire trapdoor current.execution.wires =
        some (.trapdoor publicMatrix))
      (targetFound : Mxx.Ir.lookupWire targetWire current.execution.wires = some (.matrix target))
      (typeEvaluates : matrixType.evaluate current.params cutoff = some matrixParams)
      (modulusEq : matrixParams.modulus = q)
      (ringDimensionEq : matrixParams.ringDimension = ringDimension)
      (rowsEq : matrixParams.rows = rows)
      (columnsEq : matrixParams.columns = columns) :
      LocalElaborationInputs frame q ringDimension
        (.preimage scope wire matrixType cutoff publicWire trapdoor targetWire) rows columns
  | slice
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (rowStart rowEnd columnStart columnEnd : Mxx.Ir.IntExpr)
      (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (rowStartValue rowEndValue columnStartValue columnEndValue : Int)
      (rows columns : Nat) (scopeEq : scope = current.scopeId)
      (inputEarlier : inputFormula.source.2.node < wire.node)
      (inputFound : Mxx.Ir.lookupWire inputFormula.source.2 current.execution.wires =
        some (.matrix input))
      (rowStartEvaluate : rowStart.evaluate current.params = some rowStartValue)
      (rowEndEvaluate : rowEnd.evaluate current.params = some rowEndValue)
      (columnStartEvaluate : columnStart.evaluate current.params = some columnStartValue)
      (columnEndEvaluate : columnEnd.evaluate current.params = some columnEndValue)
      (rowStartNonnegative : 0 ≤ rowStartValue) (rowOrdered : rowStartValue ≤ rowEndValue)
      (columnStartNonnegative : 0 ≤ columnStartValue)
      (columnOrdered : columnStartValue ≤ columnEndValue)
      (layout : Mxx.Toolkit.MatrixLayout
        (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat columnStartValue.toNat
          columnEndValue.toNat) q ringDimension rows columns) :
      LocalElaborationInputs frame q ringDimension
        (.slice scope wire (some (rowStart, rowEnd)) (some (columnStart, columnEnd)) inputFormula)
        rows columns
  | sliceRows
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (rowStart rowEnd : Mxx.Ir.IntExpr) (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (rowStartValue rowEndValue : Int) (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (inputEarlier : inputFormula.source.2.node < wire.node)
      (inputFound : Mxx.Ir.lookupWire inputFormula.source.2 current.execution.wires =
        some (.matrix input))
      (rowStartEvaluate : rowStart.evaluate current.params = some rowStartValue)
      (rowEndEvaluate : rowEnd.evaluate current.params = some rowEndValue)
      (rowStartNonnegative : 0 ≤ rowStartValue) (rowOrdered : rowStartValue ≤ rowEndValue)
      (layout : Mxx.Toolkit.MatrixLayout
        (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat 0 input.columns)
        q ringDimension rows columns) :
      LocalElaborationInputs frame q ringDimension
        (.slice scope wire (some (rowStart, rowEnd)) none inputFormula) rows columns
  | sliceColumns
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (columnStart columnEnd : Mxx.Ir.IntExpr) (inputFormula : FrozenPointwiseMatrixProgramFormula)
      (input : Mxx.Matrix) (columnStartValue columnEndValue : Int) (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (inputEarlier : inputFormula.source.2.node < wire.node)
      (inputFound : Mxx.Ir.lookupWire inputFormula.source.2 current.execution.wires =
        some (.matrix input))
      (columnStartEvaluate : columnStart.evaluate current.params = some columnStartValue)
      (columnEndEvaluate : columnEnd.evaluate current.params = some columnEndValue)
      (columnStartNonnegative : 0 ≤ columnStartValue)
      (columnOrdered : columnStartValue ≤ columnEndValue)
      (layout : Mxx.Toolkit.MatrixLayout
        (Mxx.matrixSlice input 0 input.rows columnStartValue.toNat columnEndValue.toNat)
        q ringDimension rows columns) :
      LocalElaborationInputs frame q ringDimension
        (.slice scope wire none (some (columnStart, columnEnd)) inputFormula) rows columns
  | concatRows
      (scope : StaticScopeId) (wire : Mxx.Ir.WireRef)
      (leftFormula rightFormula : FrozenPointwiseMatrixProgramFormula)
      (left right : Mxx.Matrix) (rows columns : Nat) (scopeEq : scope = current.scopeId)
      (leftEarlier : leftFormula.source.2.node < wire.node)
      (rightEarlier : rightFormula.source.2.node < wire.node)
      (leftFound : Mxx.Ir.lookupWire leftFormula.source.2 current.execution.wires =
        some (.matrix left))
      (rightFound : Mxx.Ir.lookupWire rightFormula.source.2 current.execution.wires =
        some (.matrix right))
      (layout : Mxx.Toolkit.MatrixLayout (Mxx.matrixConcatRows [left, right])
        q ringDimension rows columns) :
      LocalElaborationInputs frame q ringDimension
        (.concatRows scope wire leftFormula rightFormula) rows columns
  | add
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (leftInputs : LocalElaborationInputs frame q ringDimension left rows columns)
      (rightInputs : LocalElaborationInputs frame q ringDimension right rows columns) :
      LocalElaborationInputs frame q ringDimension (.add scope wire left right) rows columns
  | subtract
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (leftInputs : LocalElaborationInputs frame q ringDimension left rows columns)
      (rightInputs : LocalElaborationInputs frame q ringDimension right rows columns) :
      LocalElaborationInputs frame q ringDimension (.subtract scope wire left right) rows columns
  | multiply
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (left right : FrozenPointwiseMatrixProgramFormula)
      (rows inner columns : Nat)
      (scopeEq : scope = current.scopeId)
      (leftInputs : LocalElaborationInputs frame q ringDimension left rows inner)
      (rightInputs : LocalElaborationInputs frame q ringDimension right inner columns) :
      LocalElaborationInputs frame q ringDimension (.multiply scope wire left right) rows columns
  | negate
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (input : FrozenPointwiseMatrixProgramFormula)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (inputInputs : LocalElaborationInputs frame q ringDimension input rows columns) :
      LocalElaborationInputs frame q ringDimension (.negate scope wire input) rows columns
  | scale
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (scalar : Mxx.Ir.IntExpr)
      (input : FrozenPointwiseMatrixProgramFormula)
      (scalarValue : Int)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (scalarEvaluates : scalar.evaluate current.params = some scalarValue)
      (inputInputs : LocalElaborationInputs frame q ringDimension input rows columns) :
      LocalElaborationInputs frame q ringDimension (.scale scope wire scalar input) rows columns
  | scaleOne
      (scope : StaticScopeId)
      (wire : Mxx.Ir.WireRef)
      (input : FrozenPointwiseMatrixProgramFormula)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (inputInputs : LocalElaborationInputs frame q ringDimension input rows columns) :
      LocalElaborationInputs frame q ringDimension (.scaleOne scope wire input) rows columns
  | select
      (scope : StaticScopeId)
      (wire indexWire : Mxx.Ir.WireRef)
      (branches : List FrozenPointwiseMatrixProgramFormula)
      (index : Int)
      (selectedFormula : FrozenPointwiseMatrixProgramFormula)
      (branchValues : List Mxx.Ir.Value)
      (rows columns : Nat)
      (scopeEq : scope = current.scopeId)
      (indexFound : Mxx.Ir.lookupWire indexWire current.execution.wires =
        some (.integer index))
      (branchesFound : (branches.map fun branch => branch.source.2).mapM
        (fun branchWire => Mxx.Ir.lookupWire branchWire current.execution.wires) =
          some branchValues)
      (selectedBranch : branches[index.toNat]? = some selectedFormula)
      (indexEarlier : indexWire.node < wire.node)
      (branchesEarlier : ∀ branch ∈ branches, branch.source.2.node < wire.node)
      (selectedInputs : LocalElaborationInputs frame q ringDimension selectedFormula rows columns) :
      LocalElaborationInputs frame q ringDimension
        (.select scope wire indexWire branches) rows columns

/-- Output of local elaboration: the actual SSA value and its normalized frame-indexed semantics. -/
structure FrozenPointwiseMatrixProgramFormula.LocalElaborationResult
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    (frame : FormulaExecutionFrame samplers program current)
    (q ringDimension : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (formula : FrozenPointwiseMatrixProgramFormula)
    (rows columns : Nat) : Type where
  runtimeValue : Mxx.Matrix
  runtimeFound : Mxx.Ir.lookupWire formula.source.2 current.execution.wires =
    some (.matrix runtimeValue)
  semanticResult : formula.SemanticResultAt frame q ringDimension rows columns runtimeValue

private theorem ValidatedPointwiseNode.atCurrentScope
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {scopeId : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {kind : Mxx.Ir.NodeKind}
    {arguments : List Mxx.Ir.WireRef}
    (frameValid : LocalFormulaFrameValid current)
    (scopeEq : scopeId = current.scopeId)
    (validated : ValidatedPointwiseNode program scopeId wire kind arguments) :
    ∃ nodeInBounds : wire.node < current.execution.scope.nodes.length,
      current.execution.scope.nodes[wire.node] = {
        kind
        arguments
        outputCount := current.execution.scope.nodes[wire.node].outputCount
        outputTypes := current.execution.scope.nodes[wire.node].outputTypes
      } := by
  subst scopeId
  obtain ⟨scope, outputCount, outputTypes, scopeFound, nodeFound⟩ := validated.nodeFound
  rw [frameValid.scopeFound] at scopeFound
  have scopeEq : scope = current.execution.scope := Option.some.inj scopeFound.symm
  subst scope
  obtain ⟨nodeInBounds, nodeEq⟩ := List.getElem?_eq_some_iff.mp nodeFound
  refine ⟨nodeInBounds, ?_⟩
  rw [nodeEq]

private theorem wire_eq_zero_port
    (wire : Mxx.Ir.WireRef)
    (portZero : wire.port = 0) : wire = ⟨wire.node, 0⟩ := by
  cases wire
  simp_all

private theorem FrozenPointwiseMatrixProgramFormula.validDecomposeChild
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {matrixType : MatrixTypeExpr}
    {base digitCount : Mxx.Ir.IntExpr}
    {input : FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.decompose scope wire matrixType base
      digitCount input).validIn program substitutions = true) :
    input.validIn program substitutions = true ∧ input.source.2.node < wire.node := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  unfold pointwiseFormulaNodeValid at valid
  split at valid <;> try simp_all
  split at valid <;> try simp_all
  rename_i node nodeFound
  rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
  cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

set_option maxHeartbeats 800000 in
private theorem FrozenPointwiseMatrixProgramFormula.validBinaryChildren
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {left right : FrozenPointwiseMatrixProgramFormula}
    {kind : Mxx.Ir.NodeKind}
    {formula : FrozenPointwiseMatrixProgramFormula}
    (formulaEq : formula = match kind with
      | .matrixAdd => .add scope wire left right
      | .matrixSubtract => .subtract scope wire left right
      | .matrixMultiply => .multiply scope wire left right
      | _ => formula)
    (kindSupported : kind = .matrixAdd ∨ kind = .matrixSubtract ∨ kind = .matrixMultiply)
    (valid : formula.validIn program substitutions = true) :
    left.validIn program substitutions = true ∧ right.validIn program substitutions = true ∧
      left.source.2.node < wire.node ∧ right.source.2.node < wire.node := by
  rcases kindSupported with rfl | rfl | rfl <;> simp only at formulaEq <;> subst formula
  all_goals
    unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
    unfold pointwiseFormulaNodeValid at valid
    split at valid <;> try simp_all
    split at valid <;> try simp_all
    rename_i node nodeFound
    rcases node with ⟨nodeKind, arguments, outputCount, outputTypes⟩
    cases nodeKind <;> simp_all [pointwiseFormulaArgumentsMatch]

private theorem FrozenPointwiseMatrixProgramFormula.validUnaryChild
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scope : StaticScopeId}
    {wire : Mxx.Ir.WireRef}
    {input : FrozenPointwiseMatrixProgramFormula}
    {formula : FrozenPointwiseMatrixProgramFormula}
    (formulaEq : formula = .negate scope wire input ∨
      (∃ scalar, formula = .scale scope wire scalar input) ∨
      formula = .scaleOne scope wire input)
    (valid : formula.validIn program substitutions = true) :
    input.validIn program substitutions = true ∧ input.source.2.node < wire.node := by
  rcases formulaEq with rfl | ⟨scalar, rfl⟩ | rfl
  all_goals
    unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
    unfold pointwiseFormulaNodeValid at valid
    split at valid <;> try simp_all
    split at valid <;> try simp_all
    rename_i node nodeFound
    rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
    cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

private theorem allPointwiseFormulasValid_getElem
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {formulas : List FrozenPointwiseMatrixProgramFormula}
    {index : Nat}
    {formula : FrozenPointwiseMatrixProgramFormula}
    (allValid : allPointwiseFormulasValid program substitutions formulas = true)
    (found : formulas[index]? = some formula) :
    formula.validIn program substitutions = true := by
  induction formulas generalizing index with
  | nil => simp at found
  | cons head tail induction =>
      rw [allPointwiseFormulasValid] at allValid
      have validHead := (Bool.and_eq_true_iff.mp allValid).1
      have validTail := (Bool.and_eq_true_iff.mp allValid).2
      cases index with
      | zero =>
          have : head = formula := by simpa using Option.some.inj (show some head = some formula by
            simpa using found)
          simpa [this] using validHead
      | succ index =>
          exact induction validTail (by simpa using found)

private theorem FrozenPointwiseMatrixProgramFormula.validSelectBranches
    {program : Mxx.Ir.Prog}
    {substitutions : List FrozenPointwiseMatrixProgramFormula}
    {scope : StaticScopeId}
    {wire indexWire : Mxx.Ir.WireRef}
    {branches : List FrozenPointwiseMatrixProgramFormula}
    (valid : (FrozenPointwiseMatrixProgramFormula.select scope wire indexWire branches).validIn
      program substitutions = true) :
    allPointwiseFormulasValid program substitutions branches = true := by
  unfold FrozenPointwiseMatrixProgramFormula.validIn at valid
  unfold pointwiseFormulaNodeValid at valid
  split at valid
  · contradiction
  · rename_i staticScope staticScopeFound
    split at valid
    · contradiction
    · rename_i node nodeFound
      rcases node with ⟨kind, arguments, outputCount, outputTypes⟩
      cases kind <;> simp_all [pointwiseFormulaArgumentsMatch]

/-- Elaborate every arithmetic constructor in one exact runtime frame.  Boundary constructors
have no `LocalElaborationInputs` constructor and therefore cannot enter this theorem. -/
theorem FrozenPointwiseMatrixProgramFormula.LocalElaborationInputs.elaborate
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {frame : FormulaExecutionFrame samplers program current}
    {q ringDimension : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {formula : FrozenPointwiseMatrixProgramFormula}
    {rows columns : Nat}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (frameValid : LocalFormulaFrameValid current)
    (inputs : formula.LocalElaborationInputs frame q ringDimension rows columns)
    (valid : formula.validIn program = true) :
    Nonempty (formula.LocalElaborationResult frame q ringDimension rows columns) := by
  induction inputs with
  | atom scope wire matrix rows columns scopeEq found layout =>
      exact ⟨{
        runtimeValue := matrix
        runtimeFound := found
        semanticResult := .refl (.atom (.atom frame scope wire matrix scopeEq found)) layout
      }⟩
  | zero scope wire matrixType matrixParams rows columns scopeEq typeEvaluates modulusEq
      ringDimensionEq rowsEq columnsEq =>
      have validated := FrozenPointwiseMatrixProgramFormula.validZeroNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.zeroMatrixLookup wire.node nodeInBounds matrixType
        matrixParams nodeEq typeEvaluates
      exact ⟨{
        runtimeValue := zeroConstantOutput matrixParams
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .zero typeEvaluates modulusEq ringDimensionEq rowsEq columnsEq
      }⟩
  | identity scope wire matrixType matrixParams rows columns scopeEq typeEvaluates modulusEq
      ringDimensionEq rowsEq columnsEq =>
      have validated := FrozenPointwiseMatrixProgramFormula.validIdentityNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.identityMatrixLookup wire.node nodeInBounds matrixType
        matrixParams nodeEq typeEvaluates
      exact ⟨{
        runtimeValue := identityConstantOutput matrixParams
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .identity typeEvaluates modulusEq ringDimensionEq rowsEq columnsEq
      }⟩
  | constant scope wire matrixType coefficients matrixParams values rows columns scopeEq
      typeEvaluates coefficientsEvaluate modulusEq ringDimensionEq rowsEq columnsEq =>
      have validated := FrozenPointwiseMatrixProgramFormula.validConstantNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.constantMatrixLookup wire.node nodeInBounds matrixType
        coefficients matrixParams values nodeEq typeEvaluates coefficientsEvaluate
      exact ⟨{
        runtimeValue := Mxx.Matrix.withSamplerParams {
          coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus)
        } matrixParams
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .constant typeEvaluates coefficientsEvaluate modulusEq ringDimensionEq
          rowsEq columnsEq
      }⟩
  | gadget scope wire matrixType base matrixParams baseValue rows columns scopeEq typeEvaluates
      baseEvaluates modulusEq ringDimensionEq rowsEq columnsEq =>
      have validated := FrozenPointwiseMatrixProgramFormula.validGadgetNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.gadgetMatrixLookup wire.node nodeInBounds matrixType base
        matrixParams baseValue nodeEq typeEvaluates baseEvaluates
      exact ⟨{
        runtimeValue := Mxx.gadgetMatrix matrixParams baseValue
          (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows)
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .gadget typeEvaluates baseEvaluates modulusEq ringDimensionEq rowsEq
          columnsEq
      }⟩
  | decompose scope wire matrixType base digitCount input matrixParams baseValue digitCountValue
      inputRows inputColumns outputRows outputColumns scopeEq inputInputs typeEvaluates
      baseEvaluates digitCountEvaluates modulusEq ringDimensionEq rowsEq columnsEq inputInputs_ih =>
      obtain ⟨inputValid, inputEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validDecomposeChild valid
      obtain ⟨inputResult⟩ := inputInputs_ih inputValid
      have validated := FrozenPointwiseMatrixProgramFormula.validDecomposeNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      obtain ⟨output, outputMember, found⟩ := current.execution.gadgetDecomposeLookup wire.node
        nodeInBounds input.source.2 inputResult.runtimeValue matrixType base digitCount matrixParams
        baseValue digitCountValue inputEarlier nodeEq inputResult.runtimeFound typeEvaluates
        baseEvaluates digitCountEvaluates
      exact ⟨{
        runtimeValue := output.withSamplerParams matrixParams
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .decompose contract inputResult.semanticResult typeEvaluates
          baseEvaluates digitCountEvaluates outputMember modulusEq ringDimensionEq rowsEq columnsEq
      }⟩
  | preimage scope wire publicWire trapdoor targetWire matrixType cutoff publicMatrix target
      matrixParams rows columns scopeEq publicEarlier trapdoorEarlier targetEarlier publicFound
      trapdoorFound targetFound typeEvaluates modulusEq ringDimensionEq rowsEq columnsEq =>
      have validated := FrozenPointwiseMatrixProgramFormula.validPreimageNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      obtain ⟨sample, sampleMember, found, relation, bound⟩ :=
        current.execution.preimageSampleLookup contract
        wire.node nodeInBounds publicWire trapdoor targetWire publicMatrix target matrixType cutoff
        matrixParams publicEarlier trapdoorEarlier targetEarlier nodeEq publicFound trapdoorFound
        targetFound typeEvaluates
      exact ⟨{
        runtimeValue := sample.withSamplerParams matrixParams
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .preimage typeEvaluates relation bound modulusEq ringDimensionEq rowsEq
          columnsEq
      }⟩
  | slice scope wire rowStart rowEnd columnStart columnEnd inputFormula input rowStartValue
      rowEndValue columnStartValue columnEndValue rows columns scopeEq inputEarlier inputFound
      rowStartEvaluate rowEndEvaluate columnStartEvaluate columnEndEvaluate rowStartNonnegative
      rowOrdered columnStartNonnegative columnOrdered layout =>
      have validated := FrozenPointwiseMatrixProgramFormula.validSliceNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixSliceLookup wire.node nodeInBounds
        inputFormula.source.2 input rowStart rowEnd columnStart columnEnd rowStartValue rowEndValue
        columnStartValue columnEndValue inputEarlier nodeEq inputFound rowStartEvaluate
        rowEndEvaluate columnStartEvaluate columnEndEvaluate rowStartNonnegative rowOrdered
        columnStartNonnegative columnOrdered
      exact ⟨{
        runtimeValue := Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat
          columnStartValue.toNat columnEndValue.toNat
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .refl (.slice scope wire rowStart rowEnd columnStart columnEnd
          inputFormula input rowStartValue rowEndValue columnStartValue columnEndValue
          rowStartEvaluate rowEndEvaluate columnStartEvaluate columnEndEvaluate) layout
      }⟩
  | sliceRows scope wire rowStart rowEnd inputFormula input rowStartValue rowEndValue rows columns
      scopeEq inputEarlier inputFound rowStartEvaluate rowEndEvaluate rowStartNonnegative
      rowOrdered layout =>
      have validated := FrozenPointwiseMatrixProgramFormula.validSliceNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixSliceRowsLookup wire.node nodeInBounds
        inputFormula.source.2 input rowStart rowEnd rowStartValue rowEndValue inputEarlier nodeEq
        inputFound rowStartEvaluate rowEndEvaluate rowStartNonnegative rowOrdered
      exact ⟨{
        runtimeValue := Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat 0 input.columns
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .refl (.sliceRows scope wire rowStart rowEnd inputFormula input
          rowStartValue rowEndValue rowStartEvaluate rowEndEvaluate) layout
      }⟩
  | sliceColumns scope wire columnStart columnEnd inputFormula input columnStartValue
      columnEndValue rows columns scopeEq inputEarlier inputFound columnStartEvaluate
      columnEndEvaluate columnStartNonnegative columnOrdered layout =>
      have validated := FrozenPointwiseMatrixProgramFormula.validSliceNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixSliceColumnsLookup wire.node nodeInBounds
        inputFormula.source.2 input columnStart columnEnd columnStartValue columnEndValue
        inputEarlier nodeEq inputFound columnStartEvaluate columnEndEvaluate columnStartNonnegative
        columnOrdered
      exact ⟨{
        runtimeValue := Mxx.matrixSlice input 0 input.rows columnStartValue.toNat
          columnEndValue.toNat
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .refl (.sliceColumns scope wire columnStart columnEnd inputFormula input
          columnStartValue columnEndValue columnStartEvaluate columnEndEvaluate) layout
      }⟩
  | concatRows scope wire leftFormula rightFormula left right rows columns scopeEq leftEarlier
      rightEarlier leftFound rightFound layout =>
      have validated := FrozenPointwiseMatrixProgramFormula.validConcatRowsNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixConcatRowsTwoLookup wire.node nodeInBounds
        leftFormula.source.2 rightFormula.source.2 left right leftEarlier rightEarlier nodeEq
        leftFound rightFound
      exact ⟨{
        runtimeValue := Mxx.matrixConcatRows [left, right]
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .refl (.concatRows scope wire leftFormula rightFormula left right) layout
      }⟩
  | add scope wire left right rows columns scopeEq leftInputs rightInputs leftInputs_ih
      rightInputs_ih =>
      obtain ⟨leftValid, rightValid, leftEarlier, rightEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validBinaryChildren (kind := .matrixAdd) rfl
          (Or.inl rfl) valid
      obtain ⟨leftResult⟩ := leftInputs_ih leftValid
      obtain ⟨rightResult⟩ := rightInputs_ih rightValid
      have validated := FrozenPointwiseMatrixProgramFormula.validAddNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixAddLookup wire.node nodeInBounds left.source.2
        right.source.2 leftResult.runtimeValue rightResult.runtimeValue leftEarlier rightEarlier
        nodeEq leftResult.runtimeFound rightResult.runtimeFound
      exact ⟨{
        runtimeValue := Mxx.matrixAdd leftResult.runtimeValue rightResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .add leftResult.semanticResult rightResult.semanticResult
      }⟩
  | subtract scope wire left right rows columns scopeEq leftInputs rightInputs leftInputs_ih
      rightInputs_ih =>
      obtain ⟨leftValid, rightValid, leftEarlier, rightEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validBinaryChildren (kind := .matrixSubtract) rfl
          (Or.inr (Or.inl rfl)) valid
      obtain ⟨leftResult⟩ := leftInputs_ih leftValid
      obtain ⟨rightResult⟩ := rightInputs_ih rightValid
      have validated := FrozenPointwiseMatrixProgramFormula.validSubtractNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixSubtractLookup wire.node nodeInBounds left.source.2
        right.source.2 leftResult.runtimeValue rightResult.runtimeValue leftEarlier rightEarlier
        nodeEq leftResult.runtimeFound rightResult.runtimeFound
      exact ⟨{
        runtimeValue := Mxx.matrixSubtract leftResult.runtimeValue rightResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .subtract leftResult.semanticResult rightResult.semanticResult
      }⟩
  | multiply scope wire left right rows inner columns scopeEq leftInputs rightInputs leftInputs_ih
      rightInputs_ih =>
      obtain ⟨leftValid, rightValid, leftEarlier, rightEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validBinaryChildren (kind := .matrixMultiply) rfl
          (Or.inr (Or.inr rfl)) valid
      obtain ⟨leftResult⟩ := leftInputs_ih leftValid
      obtain ⟨rightResult⟩ := rightInputs_ih rightValid
      have validated := FrozenPointwiseMatrixProgramFormula.validMultiplyNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixMultiplyLookup wire.node nodeInBounds left.source.2
        right.source.2 leftResult.runtimeValue rightResult.runtimeValue leftEarlier rightEarlier
        nodeEq leftResult.runtimeFound rightResult.runtimeFound
      exact ⟨{
        runtimeValue := Mxx.matrixMultiply leftResult.runtimeValue rightResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .multiply leftResult.semanticResult rightResult.semanticResult
      }⟩
  | negate scope wire input rows columns scopeEq inputInputs inputInputs_ih =>
      obtain ⟨inputValid, inputEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validUnaryChild (Or.inl rfl) valid
      obtain ⟨inputResult⟩ := inputInputs_ih inputValid
      have validated := FrozenPointwiseMatrixProgramFormula.validNegateNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixNegateLookup wire.node nodeInBounds input.source.2
        inputResult.runtimeValue inputEarlier nodeEq inputResult.runtimeFound
      exact ⟨{
        runtimeValue := Mxx.matrixNegate inputResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .negate inputResult.semanticResult
      }⟩
  | scale scope wire scalar input scalarValue rows columns scopeEq scalarEvaluates inputInputs
      inputInputs_ih =>
      obtain ⟨inputValid, inputEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validUnaryChild
          (Or.inr (Or.inl ⟨scalar, rfl⟩)) valid
      obtain ⟨inputResult⟩ := inputInputs_ih inputValid
      have validated := FrozenPointwiseMatrixProgramFormula.validScaleNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixScaleLookup wire.node nodeInBounds input.source.2
        inputResult.runtimeValue scalar scalarValue inputEarlier nodeEq inputResult.runtimeFound
        scalarEvaluates
      exact ⟨{
        runtimeValue := Mxx.matrixScale scalarValue inputResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .scale scalarEvaluates inputResult.semanticResult
      }⟩
  | scaleOne scope wire input rows columns scopeEq inputInputs inputInputs_ih =>
      obtain ⟨inputValid, inputEarlier⟩ :=
        FrozenPointwiseMatrixProgramFormula.validUnaryChild (Or.inr (Or.inr rfl)) valid
      obtain ⟨inputResult⟩ := inputInputs_ih inputValid
      have validated := FrozenPointwiseMatrixProgramFormula.validScaleOneNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have found := current.execution.matrixScaleLookup wire.node nodeInBounds input.source.2
        inputResult.runtimeValue (.constant 1) 1 inputEarlier nodeEq inputResult.runtimeFound (by rfl)
      exact ⟨{
        runtimeValue := Mxx.matrixScale 1 inputResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .scaleOne inputResult.semanticResult
      }⟩
  | select scope wire indexWire branches index selectedFormula branchValues rows columns scopeEq
      indexFound branchesFound selectedBranch indexEarlier branchesEarlier selectedInputs
      selectedInputs_ih =>
      have branchesValid := FrozenPointwiseMatrixProgramFormula.validSelectBranches valid
      have selectedValid := allPointwiseFormulasValid_getElem branchesValid selectedBranch
      obtain ⟨selectedResult⟩ := selectedInputs_ih selectedValid
      have validated := FrozenPointwiseMatrixProgramFormula.validSelectNode valid
      obtain ⟨nodeInBounds, nodeEq⟩ := validated.atCurrentScope frameValid scopeEq
      have selectedRef : (branches.map fun branch => branch.source.2)[index.toNat]? =
          some selectedFormula.source.2 := by
        simpa using congrArg (Option.map fun branch => branch.source.2) selectedBranch
      have found := current.execution.matrixSelectLookup wire.node nodeInBounds indexWire
        selectedFormula.source.2 (branches.map fun branch => branch.source.2) branchValues index
        selectedResult.runtimeValue indexEarlier
        (by
          intro branchRef branchRefMember
          obtain ⟨branch, branchMember, rfl⟩ := List.mem_map.mp branchRefMember
          exact branchesEarlier branch branchMember)
        selectedRef nodeEq indexFound branchesFound selectedResult.runtimeFound
      exact ⟨{
        runtimeValue := selectedResult.runtimeValue
        runtimeFound := by rw [wire_eq_zero_port wire validated.outputPortZero]; exact found
        semanticResult := .select indexFound selectedBranch selectedResult.semanticResult
      }⟩
end Mxx.Certificate
