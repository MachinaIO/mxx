import Mxx.Certificate.LocalSoundness
import Mxx.Certificate.Facts
import Mxx.Certificate.Rules.CanonicalResidues
import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-- Closed transform subset present in the current Diamond workflow. -/
inductive TransformRule where
  | uniformMinusOneOne
  | uniformZeroOne
  | slice
  | concatRows
  | concatColumns
  | concatDiagonal
  deriving BEq, DecidableEq, Repr

/- Norm derivation in this module is intentionally independent from coefficient representation.
Once `MatrixFact` obtains the accepted provenance field, slice and concat may preserve
`canonicalResidues(q)` only by applying the executable lemmas from `CanonicalResidues`; they must
never infer it merely from a centered bound. -/

inductive TransformRuleError where
  | unsupportedNodeKind (kind : Mxx.Ir.NodeKind)
  | relationBearingInput
  | affineSignalInput
  | emptyConcat

def inferTransformRule : Mxx.Ir.NodeKind → Except TransformRuleError TransformRule
  | .uniformIntervalSample _ (.constant (-1)) (.constant 1) => .ok .uniformMinusOneOne
  | .uniformIntervalSample _ (.constant 0) (.constant 1) => .ok .uniformZeroOne
  | .slice _ _ => .ok .slice
  | .concat .rows => .ok .concatRows
  | .concat .columns => .ok .concatColumns
  | .concat .diagonal => .ok .concatDiagonal
  | kind => .error (.unsupportedNodeKind kind)

private def boundedOnly (fact : MatrixFact) : Except TransformRuleError BoundExpr := do
  if !fact.relations.isEmpty then throw .relationBearingInput
  match fact.primary with
  | .affine form =>
      if form.terms.isEmpty then return fact.totalNormBound
      throw .affineSignalInput
  | .exact _ => throw .affineSignalInput

private def sliceExpression
    (expression : MatrixExpr)
    (rows columns : Option (IntExpr × IntExpr)) : MatrixExpr :=
  let rowsSliced := match rows with
    | some (start, stop) => .rowSlice expression start stop
    | none => expression
  match columns with
  | some (start, stop) => .columnSlice rowsSliced start stop
  | none => rowsSliced

/-- Slice preserves an exact symbolic expression by applying the corresponding typed expression
constructor.  A noise-only affine input remains noise-only.  Signal-bearing affine forms and
relations are rejected because slicing does not in general distribute through their products or
transport their relations.  Coefficient representation is preserved solely because the runtime
slice copies coefficients, as proved by `matrixSlice_preservesCanonicalResidues`. -/
def deriveSliceFact
    (output : ValueInstanceRef)
    (rows columns : Option (IntExpr × IntExpr))
    (input : MatrixFact) : Except TransformRuleError MatrixFact := do
  let primary := match input.primary, input.relations.isEmpty with
    | .exact expression, true => .exact (sliceExpression expression rows columns)
    | _, _ => .affine { terms := [], noiseBound := input.totalNormBound }
  return {
    subject := output
    primary
    relations := []
    totalNormBound := input.totalNormBound
    coefficientRepresentation := input.coefficientRepresentation
  }

def deriveUniformBoundOneFact (output : ValueInstanceRef) : MatrixFact := {
  subject := output
  primary := .affine { terms := [], noiseBound := .constant 1 }
  relations := []
  totalNormBound := .constant 1
}

def maximumBounds : List BoundExpr → BoundExpr
  | [] => .constant 0
  | head :: tail => tail.foldl .maximum head

def deriveBoundedConcatFact
    (output : ValueInstanceRef)
    (inputs : List MatrixFact) : Except TransformRuleError MatrixFact := do
  if inputs.isEmpty then throw .emptyConcat
  let bounds ← inputs.mapM boundedOnly
  let bound := maximumBounds bounds
  return {
    subject := output
    primary := .affine { terms := [], noiseBound := bound }
    relations := []
    totalNormBound := bound
  }

private def exactExpressions (inputs : List MatrixFact) : Option (List MatrixExpr) :=
  inputs.mapM fun input => match input.primary, input.relations.isEmpty with
    | .exact expression, true => some expression
    | _, _ => none

/-- Concatenation preserves a symbolic exact expression only when every input is exact and carries
no relation.  Every other accepted input is conservatively materialized as one bounded value.
This is sound for rows, columns, and diagonal concatenation because these operations only copy
coefficients and insert zeros, so the output maximum coefficient norm is the maximum input norm. -/
def deriveConcatFact
    (axis : Mxx.Ir.ConcatAxis)
    (output : ValueInstanceRef)
    (inputs : List MatrixFact) : Except TransformRuleError MatrixFact := do
  if inputs.isEmpty then throw .emptyConcat
  let bound := maximumBounds (inputs.map (·.totalNormBound))
  let primary := match exactExpressions inputs with
    | some expressions => .exact <| match axis with
        | .rows => .rowConcat expressions
        | .columns => .columnConcat expressions
        | .diagonal => .diagonalConcat expressions
    | none => .affine { terms := [], noiseBound := bound }
  return {
    subject := output
    primary
    relations := []
    totalNormBound := bound
  }

theorem uniformMinusOneOneNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params = some matrixParams)
    (modulusGe : 2 ≤ matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .uniformIntervalSample matrixType (.constant (-1)) (.constant 1)
      arguments := []
      outputCount
    }) :
    ∃ matrix,
      values = [.matrix matrix] ∧ Mxx.maxCenteredCoefficientNorm matrix ≤ 1 := by
  obtain ⟨matrix, matrixMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_uniformIntervalSample
    runChild samplers params inputs wires matrixType (.constant (-1)) (.constant 1)
    matrixParams (-1) 1 outputCount matrixTypeEvaluate rfl rfl member
  exact ⟨matrix, rfl,
    Mxx.Toolkit.uniformMatrixSupport_minusOneOne_norm_le matrixParams modulusGe matrix matrixMember⟩

/-- A full-residue sample is intentionally not turned into a bounded-noise fact.  This theorem
records only its universal centered cap; higher-level analysis must retain it as a carrier. -/
theorem uniformResidueNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params = some matrixParams)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .uniformResidueSample matrixType
      arguments := []
      outputCount
    }) :
    ∃ matrix,
      values = [.matrix matrix] ∧
        Mxx.maxCenteredCoefficientNorm matrix ≤ matrixParams.modulus.natAbs / 2 := by
  obtain ⟨matrix, matrixMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_uniformResidueSample
    runChild samplers params inputs wires matrixType matrixParams outputCount matrixTypeEvaluate member
  refine ⟨matrix, rfl, ?_⟩
  have layout := Mxx.Toolkit.uniformMatrixSupport_layout matrixParams 0
    (matrixParams.modulus - 1) matrix matrixMember
  rw [← layout.modulus]
  exact matrix_norm_le_centered_radius matrix (by simpa [layout.modulus] using modulusPositive)

theorem uniformZeroOneNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (matrixType : MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params = some matrixParams)
    (modulusGe : 2 ≤ matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .uniformIntervalSample matrixType (.constant 0) (.constant 1)
      arguments := []
      outputCount
    }) :
    ∃ matrix,
      values = [.matrix matrix] ∧ Mxx.maxCenteredCoefficientNorm matrix ≤ 1 := by
  obtain ⟨matrix, matrixMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_uniformIntervalSample
    runChild samplers params inputs wires matrixType (.constant 0) (.constant 1)
    matrixParams 0 1 outputCount matrixTypeEvaluate rfl rfl member
  rw [Mxx.Ir.uniformMatrixSupport] at matrixMember
  obtain ⟨coefficients, coefficientsMember, rfl⟩ := List.mem_map.mp matrixMember
  refine ⟨_, rfl, ?_⟩
  apply withSamplerParams_zeroOne_norm_le matrixParams coefficients
  · intro coefficient coefficientMember
    have sourceMember : coefficient ∈ coefficients := coefficientMember
    have coefficientRange := Mxx.Toolkit.coefficientVectors_member
      coefficientsMember coefficient sourceMember
    have range : Mxx.Ir.integerRange 0 1 = [0, 1] := by decide
    rw [range] at coefficientRange
    simpa using coefficientRange
  · omega

theorem sliceNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (matrix : Mxx.Matrix)
    (rowStart rowEnd columnStart columnEnd : IntExpr)
    (evaluatedRowStart evaluatedRowEnd evaluatedColumnStart evaluatedColumnEnd : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix matrix])
    (rowStartEvaluate : rowStart.evaluate params = some evaluatedRowStart)
    (rowEndEvaluate : rowEnd.evaluate params = some evaluatedRowEnd)
    (columnStartEvaluate : columnStart.evaluate params = some evaluatedColumnStart)
    (columnEndEvaluate : columnEnd.evaluate params = some evaluatedColumnEnd)
    (rowStartNonnegative : 0 ≤ evaluatedRowStart)
    (rowOrdered : evaluatedRowStart ≤ evaluatedRowEnd)
    (columnStartNonnegative : 0 ≤ evaluatedColumnStart)
    (columnOrdered : evaluatedColumnStart ≤ evaluatedColumnEnd)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .slice (some (rowStart, rowEnd)) (some (columnStart, columnEnd))
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixSlice matrix evaluatedRowStart.toNat evaluatedRowEnd.toNat
      evaluatedColumnStart.toNat evaluatedColumnEnd.toNat)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, rowStartEvaluate,
    rowEndEvaluate, columnStartEvaluate, columnEndEvaluate, not_lt.mpr rowStartNonnegative,
    not_lt.mpr rowOrdered, not_lt.mpr columnStartNonnegative, not_lt.mpr columnOrdered] using member

/-- Executable semantics for a row-only slice.  An omitted column range denotes all columns of
the runtime matrix, exactly as in `Mxx.Ir.evaluateNode`. -/
theorem sliceRowsNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (matrix : Mxx.Matrix)
    (rowStart rowEnd : IntExpr)
    (evaluatedRowStart evaluatedRowEnd : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix matrix])
    (rowStartEvaluate : rowStart.evaluate params = some evaluatedRowStart)
    (rowEndEvaluate : rowEnd.evaluate params = some evaluatedRowEnd)
    (rowStartNonnegative : 0 ≤ evaluatedRowStart)
    (rowOrdered : evaluatedRowStart ≤ evaluatedRowEnd)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .slice (some (rowStart, rowEnd)) none
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixSlice matrix evaluatedRowStart.toNat evaluatedRowEnd.toNat
      0 matrix.columns)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, rowStartEvaluate,
    rowEndEvaluate, not_lt.mpr rowStartNonnegative, not_lt.mpr rowOrdered] using member

/-- Executable semantics for Diamond's column-only slices.  An omitted row range denotes all rows
of the runtime matrix, exactly as in `Mxx.Ir.evaluateNode`. -/
theorem sliceColumnsNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (matrix : Mxx.Matrix)
    (columnStart columnEnd : IntExpr)
    (evaluatedColumnStart evaluatedColumnEnd : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix matrix])
    (columnStartEvaluate : columnStart.evaluate params = some evaluatedColumnStart)
    (columnEndEvaluate : columnEnd.evaluate params = some evaluatedColumnEnd)
    (columnStartNonnegative : 0 ≤ evaluatedColumnStart)
    (columnOrdered : evaluatedColumnStart ≤ evaluatedColumnEnd)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .slice none (some (columnStart, columnEnd))
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixSlice matrix 0 matrix.rows evaluatedColumnStart.toNat
      evaluatedColumnEnd.toNat)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, columnStartEvaluate,
    columnEndEvaluate, not_lt.mpr columnStartNonnegative, not_lt.mpr columnOrdered] using member

theorem concatRowsTwoNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q leftBound rightBound : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .concat .rows
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixConcatRows [left, right])] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatRows [left, right]) ≤
        max leftBound rightBound := by
  constructor
  · simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member
  · exact le_trans
      (Mxx.Toolkit.matrixConcatRows_two_norm_le q left right leftModulus rightModulus)
      (max_le_max leftNorm rightNorm)

theorem concatColumnsTwoNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q leftBound rightBound : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .concat .columns
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixConcatColumns [left, right])] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatColumns [left, right]) ≤
        max leftBound rightBound := by
  constructor
  · simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member
  · exact le_trans
      (Mxx.Toolkit.matrixConcatColumns_two_norm_le q left right leftModulus rightModulus)
      (max_le_max leftNorm rightNorm)

theorem concatDiagonalTwoNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q leftBound rightBound : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix left, .matrix right])
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .concat .diagonal
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixConcatDiagonal [left, right])] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatDiagonal [left, right]) ≤
        max leftBound rightBound := by
  constructor
  · simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member
  · exact le_trans
      (Mxx.Toolkit.matrixConcatDiagonal_two_norm_le q left right leftModulus rightModulus)
      (max_le_max leftNorm rightNorm)

example : inferTransformRule (.uniformIntervalSample {
    modulus := .constant 17
    ringDimension := .constant 4
    rows := .constant 1
    columns := .constant 1
  } (.constant 0) (.constant 1)) = .ok .uniformZeroOne := rfl

private def fixtureValue : ValueInstanceRef := .protocolInput ⟨"matrix"⟩

private def fixtureBoundedMatrix : MatrixFact := {
  subject := fixtureValue
  primary := .affine { terms := [], noiseBound := .constant 7 }
  relations := []
  totalNormBound := .constant 7
}

private def fixtureSignalMatrix : MatrixFact := {
  subject := fixtureValue
  primary := .affine {
    terms := [{
      coefficient := {
        expression := .zero {
          modulus := .constant 17
          ringDimension := .constant 4
          rows := .constant 1
          columns := .constant 1
        }
        normBound := .constant 1
      }
      basis := .zero {
        modulus := .constant 17
        ringDimension := .constant 4
        rows := .constant 1
        columns := .constant 1
      }
      mode := .ordinaryMatrixProduct
    }]
    noiseBound := .constant 0
  }
  relations := []
  totalNormBound := .constant 1
}

private def fixtureExactMatrix : MatrixFact := {
  subject := fixtureValue
  primary := .exact (.wire {
    value := fixtureValue
    type := {
      modulus := .constant 17
      ringDimension := .constant 4
      rows := .constant 2
      columns := .constant 2
    }
  })
  relations := []
  totalNormBound := .constant 8
  coefficientRepresentation := .canonicalResidues (.constant 17)
}

example : deriveSliceFact (.protocolInput ⟨"sliced"⟩) none
    (some (.constant 0, .constant 1)) fixtureExactMatrix = .ok {
      subject := .protocolInput ⟨"sliced"⟩
      primary := .exact (.columnSlice (.wire {
        value := fixtureValue
        type := {
          modulus := .constant 17
          ringDimension := .constant 4
          rows := .constant 2
          columns := .constant 2
        }
      }) (.constant 0) (.constant 1))
      relations := []
      totalNormBound := .constant 8
      coefficientRepresentation := .canonicalResidues (.constant 17)
    } := rfl

example : deriveBoundedConcatFact (.protocolInput ⟨"concatenated"⟩) [] =
    .error .emptyConcat := rfl

example : deriveUniformBoundOneFact (.protocolInput ⟨"uniform"⟩) = {
    subject := .protocolInput ⟨"uniform"⟩
    primary := .affine { terms := [], noiseBound := .constant 1 }
    relations := []
    totalNormBound := .constant 1
  } := rfl

end Mxx.Certificate
