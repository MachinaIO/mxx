import Mxx.Ir.ExecutionFacts
import Mxx.Toolkit.Negacyclic
import Mxx.Toolkit.Norms

namespace Mxx.Certificate

/-- Output-type transport is checked by the certificate analyzer and does not alter execution. -/
theorem evaluateNode_outputTypes_irrelevant
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (node : Mxx.Ir.Node)
    (outputTypes : List Mxx.Ir.WireTypeExpr) :
    Mxx.Ir.evaluateNode runChild samplers params inputs wires
        { node with outputTypes := outputTypes } =
      Mxx.Ir.evaluateNode runChild samplers params inputs wires node := by
  cases node
  rfl

theorem centeredCoefficient_le_radius (q : Nat) [NeZero q] (value : Int) :
    (Mxx.centeredCoefficient q value).natAbs ≤ q / 2 := by
  rw [Mxx.Toolkit.centeredCoefficient_eq_valMinAbs]
  exact ZMod.natAbs_valMinAbs_le _

/-- Every stored coefficient is within the centered radius. No assumption about the coefficient
list's expected shape or length is needed. Positivity is necessary because nonpositive moduli make
`centeredCoefficient` return the input unchanged. -/
theorem matrix_norm_le_centered_radius
    (matrix : Mxx.Matrix)
    (modulusPositive : 0 < matrix.modulus) :
    Mxx.maxCenteredCoefficientNorm matrix ≤ matrix.modulus.natAbs / 2 := by
  let q := matrix.modulus.toNat
  have qPositive : 0 < q := by
    simp [q]
    exact modulusPositive
  letI : NeZero q := ⟨Nat.ne_of_gt qPositive⟩
  have modulusEq : (q : Int) = matrix.modulus := Int.toNat_of_nonneg modulusPositive.le
  unfold Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro coefficient member
  obtain ⟨original, _, rfl⟩ := List.mem_map.mp member
  rw [← modulusEq]
  change (Mxx.centeredCoefficient q original).natAbs ≤ q / 2
  exact centeredCoefficient_le_radius q original

theorem withSamplerParams_zeroOne_norm_le
    (params : Mxx.SamplerParams)
    (coefficients : List Int)
    (coefficientsZeroOne : ∀ coefficient ∈ coefficients, coefficient = 0 ∨ coefficient = 1)
    (modulusPositive : 0 < params.modulus) :
    Mxx.maxCenteredCoefficientNorm
      (Mxx.Matrix.withSamplerParams { coefficients } params) ≤ 1 := by
  have modulusEq : (params.modulus.toNat : Int) = params.modulus :=
    Int.toNat_of_nonneg modulusPositive.le
  letI : NeZero params.modulus.toNat := ⟨by omega⟩
  unfold Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append, List.mem_replicate]
    at coefficientMember
  rcases coefficientMember with coefficientMember | ⟨_, rfl⟩
  · have sourceMember : coefficient ∈ coefficients := List.mem_of_mem_take coefficientMember
    change (Mxx.centeredCoefficient params.modulus coefficient).natAbs ≤ 1
    rw [← modulusEq]
    exact le_trans
      (Mxx.Toolkit.centeredCoefficient_natAbs_le params.modulus.toNat coefficient) (by
        rcases coefficientsZeroOne coefficient sourceMember with rfl | rfl <;> decide)
  · change (Mxx.centeredCoefficient params.modulus 0).natAbs ≤ 1
    rw [← modulusEq]
    exact le_trans
      (Mxx.Toolkit.centeredCoefficient_natAbs_le params.modulus.toNat 0) (by decide)

def zeroConstantOutput (params : Mxx.SamplerParams) : Mxx.Matrix :=
  let count := params.rows * params.columns * params.ringDimension
  Mxx.Matrix.withSamplerParams { coefficients := List.replicate count 0 } params

theorem zeroConstant_norm_eq_zero (params : Mxx.SamplerParams) :
    Mxx.maxCenteredCoefficientNorm (zeroConstantOutput params) = 0 := by
  have centeredZero : Mxx.centeredCoefficient params.modulus 0 = 0 := by
    by_cases nonpositive : params.modulus ≤ 0
    · simp [Mxx.centeredCoefficient, nonpositive]
    · have positive : 0 < params.modulus := lt_of_not_ge nonpositive
      simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, nonpositive, positive.ne',
        positive.le]
  have coefficientNormZeros :
      ∀ count : Nat, Mxx.coefficientNorm (List.replicate count 0) = 0 := by
    intro count
    induction count with
    | zero => rfl
    | succ count induction =>
        rw [List.replicate_succ, Mxx.coefficientNorm, induction]
        rfl
  simp [zeroConstantOutput, Mxx.Matrix.withSamplerParams, Mxx.maxCenteredCoefficientNorm,
    centeredZero, coefficientNormZeros]

def identityConstantOutput (params : Mxx.SamplerParams) : Mxx.Matrix :=
  let coefficients :=
    (List.range params.rows).flatMap fun row ↦
      (List.range params.columns).flatMap fun column ↦
        (List.range params.ringDimension).map fun coefficient ↦
          if row = column ∧ coefficient = 0 then 1 else 0
  Mxx.Matrix.withSamplerParams { coefficients } params

theorem identityConstant_norm_le_one
    (params : Mxx.SamplerParams)
    (modulusPositive : 0 < params.modulus) :
    Mxx.maxCenteredCoefficientNorm (identityConstantOutput params) ≤ 1 := by
  apply withSamplerParams_zeroOne_norm_le params _ _ modulusPositive
  intro value valueMember
  simp only [List.mem_flatMap, List.mem_range, List.mem_map] at valueMember
  obtain ⟨row, _, column, _, coefficient, _, rfl⟩ := valueMember
  by_cases isUnit : row = column ∧ coefficient = 0 <;> simp [isUnit]

/-- The canonical symbolic identity denotes a left identity for every compatible executable
matrix layout. The symbolic constructor itself is not an executable IR node. -/
theorem identityConstant_matrixValue_mul_left
    (q n rows columns bound : Nat)
    [NeZero q] [NeZero n]
    (right : Mxx.Matrix)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q n rows columns) :
    Mxx.Toolkit.matrixValue q n rows columns
        (Mxx.matrixMultiply
          (identityConstantOutput {
            maxCoefficientBound := bound
            modulus := q
            ringDimension := n
            rows
            columns := rows
          }) right) =
      Mxx.Toolkit.matrixValue q n rows columns right := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q n rows rows columns]
  · have identityValue :
        Mxx.Toolkit.matrixValue q n rows rows
            (identityConstantOutput {
              maxCoefficientBound := bound
              modulus := q
              ringDimension := n
              rows
              columns := rows
            }) = 1 := by
        simpa [identityConstantOutput] using
          (Mxx.Toolkit.matrixValue_withSamplerParams_identity q n rows bound)
    rw [identityValue]
    exact Matrix.one_mul _
  · exact Mxx.Toolkit.withSamplerParams_layout _ _
  · exact rightLayout

inductive ExactConstantOutput where
  | zero (params : Mxx.SamplerParams)
  | identity (params : Mxx.SamplerParams)
  | polynomial (output : Mxx.Matrix)
  | gadget (output : Mxx.Matrix)

def ExactConstantOutput.matrix : ExactConstantOutput → Mxx.Matrix
  | .zero params => zeroConstantOutput params
  | .identity params => identityConstantOutput params
  | .polynomial output => output
  | .gadget output => output

def ExactConstantOutput.bound : ExactConstantOutput → Nat
  | .zero _ => 0
  | .identity _ => 1
  | .polynomial output => output.modulus.natAbs / 2
  | .gadget output => output.modulus.natAbs / 2

/-- Closed dispatch for exactly the four constant constructors accepted by the Rust analyzer. -/
theorem exactConstant_constructor_dispatch
    (constant : ExactConstantOutput)
    (modulusPositive : 0 < constant.matrix.modulus) :
    Mxx.maxCenteredCoefficientNorm constant.matrix ≤ constant.bound := by
  cases constant with
  | zero params => exact (zeroConstant_norm_eq_zero params).le
  | identity params => exact identityConstant_norm_le_one params modulusPositive
  | polynomial output => exact matrix_norm_le_centered_radius output modulusPositive
  | gadget output => exact matrix_norm_le_centered_radius output modulusPositive

/-- Compose an executable node's exact matrix-output inversion with the generic centered-radius
bound. Constant-matrix and plain-hash nodes supply `outputExact` directly from `evaluateNode` (the
hash case uses `mem_evaluateNode_hashSample_of_arguments`). -/
theorem exactMatrixOutput_local_sound
    {values : List Mxx.Ir.Value}
    (output : Mxx.Matrix)
    (outputExact : values = [.matrix output])
    (modulusPositive : 0 < output.modulus) :
    values = [.matrix output] ∧
      Mxx.maxCenteredCoefficientNorm output ≤ output.modulus.natAbs / 2 :=
  ⟨outputExact, matrix_norm_le_centered_radius output modulusPositive⟩

/-- `MatrixScale(1)` is not an unconditional identity on the executable `Matrix` representation:
it canonicalizes stored coefficients modulo the matrix modulus. -/
theorem matrixScale_one_not_unconditional :
    Mxx.matrixScale 1 {
      coefficients := [7]
      modulus := 5
      ringDimension := 1
      rows := 1
      columns := 1
    } ≠ {
      coefficients := [7]
      modulus := 5
      ringDimension := 1
      rows := 1
      columns := 1
    } := by
  decide

theorem matrixAddNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q leftBound rightBound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix left, .matrix right])
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixAdd left right)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixAdd left right) ≤ leftBound + rightBound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixAdd_of_arguments runChild samplers params inputs wires
      leftRef rightRef left right outputCount argumentsEvaluate member
  · exact le_trans (Mxx.Toolkit.matrixAdd_norm_le q left right leftModulus rightModulus)
      (Nat.add_le_add leftNorm rightNorm)

theorem matrixSubtractNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q leftBound rightBound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix left, .matrix right])
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixSubtract
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixSubtract left right)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixSubtract left right) ≤
        leftBound + rightBound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixSubtract_of_arguments runChild samplers params inputs wires
      leftRef rightRef left right outputCount argumentsEvaluate member
  · exact le_trans (Mxx.Toolkit.matrixSubtract_norm_le q left right leftModulus rightModulus)
      (Nat.add_le_add leftNorm rightNorm)

theorem matrixNegateNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (input : Mxx.Matrix)
    (outputCount q bound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [inputRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) = some [.matrix input])
    (modulus : input.modulus = q)
    (inputNorm : Mxx.maxCenteredCoefficientNorm input ≤ bound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixNegate
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixNegate input)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixNegate input) ≤ bound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixNegate_of_arguments runChild samplers params inputs wires
      inputRef input outputCount argumentsEvaluate member
  · rw [Mxx.Toolkit.matrixNegate_norm q input modulus]
    exact inputNorm

/-- Gaussian introduction is locally sound for every executable support member. -/
theorem gaussianNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (cutoff : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .gaussianSample matrixType cutoff
      arguments := []
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      Mxx.maxCenteredCoefficientNorm output ≤ matrixParams.maxCoefficientBound := by
  obtain ⟨sample, sampleMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_gaussianSample
    runChild samplers params inputs wires matrixType cutoff matrixParams outputCount
    matrixTypeEvaluate member
  exact ⟨sample.withSamplerParams matrixParams, rfl,
    contract.gaussianHardSupport matrixParams sample sampleMember⟩

/-- Preimage introduction proves both the exact source relation and its hard bound. -/
theorem preimageNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (publicRef trapdoorRef targetRef : Mxx.Ir.WireRef)
    (publicMatrix target : Mxx.Matrix)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (cutoff : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (argumentsEvaluate :
      [publicRef, trapdoorRef, targetRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix publicMatrix, .trapdoor publicMatrix .sampled, .matrix target])
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .preimageSample matrixType cutoff
      arguments := [publicRef, trapdoorRef, targetRef]
      outputCount
    }) :
    ∃ output,
      values = [.matrix output] ∧
      Mxx.MatrixModEq (Mxx.matrixMul publicMatrix output) target ∧
      Mxx.maxCenteredCoefficientNorm output ≤ matrixParams.maxCoefficientBound ∧
      output.WellFormed ∧
      output.modulus = matrixParams.modulus ∧
      output.ringDimension = matrixParams.ringDimension ∧
      output.rows = matrixParams.rows ∧
      output.columns = matrixParams.columns := by
  obtain ⟨sample, sampleMember, rfl⟩ := Mxx.Ir.mem_evaluateNode_preimageSample_of_arguments
    runChild samplers params inputs wires publicRef trapdoorRef targetRef publicMatrix target
    matrixType cutoff matrixParams outputCount argumentsEvaluate matrixTypeEvaluate member
  obtain ⟨relation, bound⟩ :=
    contract.preimageContract matrixParams publicMatrix target sample sampleMember
  exact ⟨sample.withSamplerParams matrixParams, rfl, relation, bound,
    Mxx.Matrix.withSamplerParams_wellFormed _ _, rfl, rfl, rfl, rfl⟩

/-- A successful backend gadget decomposition carries exactly the modular relation and hard
coefficient bound consumed by the operational relation token. This theorem does not reconstruct
the gadget matrix or CRT layout in Lean. -/
theorem gadgetDecompose_local_sound
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (paramsId : Mxx.SamplerParamsId)
    (params : Mxx.SamplerParams)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (input publicMatrix output : Mxx.Matrix)
    (publicResult :
      samplers.gadgetPublicMatrix paramsId params input.rows base small digitCount =
        some publicMatrix)
    (decompositionResult :
      samplers.gadgetDecompose paramsId params base small digitCount input = some output)
    (reconstructionAvailable :
      small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit ∧
        Mxx.maxCanonicalCoefficient input < limit) :
    Mxx.MatrixModEq (Mxx.matrixMul publicMatrix (output.withSamplerParams params)) input ∧
      Mxx.maxCenteredCoefficientNorm (output.withSamplerParams params) ≤
        Mxx.gadgetDecompositionBound base small := by
  exact ⟨contract.gadgetDecomposeRelation paramsId params base small digitCount input publicMatrix
      output publicResult decompositionResult reconstructionAvailable,
    contract.gadgetDecomposeHardBound paramsId params base small digitCount input output
      decompositionResult⟩

/-- The unsigned small-digit range is independent of the centered hard bound. -/
theorem gadgetDecomposeSmall_canonical_range
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (paramsId : Mxx.SamplerParamsId)
    (params : Mxx.SamplerParams)
    (base : Int)
    (digitCount : Nat)
    (input output : Mxx.Matrix)
    (decompositionResult :
      samplers.gadgetDecompose paramsId params base true digitCount input = some output) :
    Mxx.maxCanonicalCoefficient (output.withSamplerParams params) < base.natAbs :=
  contract.gadgetDecomposeSmallCanonicalRange paramsId params base digitCount input output
    decompositionResult

/-- A decomposed-hash leaf is the backend decomposition of the exactly matching plain-hash
query. Consequently it receives the same modular reconstruction relation and hard digit bound as
a direct decomposition; Lean neither reconstructs hashing nor the backend CRT layout. -/
theorem decomposedHash_local_sound
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (paramsId : Mxx.SamplerParamsId)
    (plain decomposed : Mxx.HashQuery)
    (base : Int)
    (small : Bool)
    (digitCount : Nat)
    (publicMatrix : Mxx.Matrix)
    (layoutId : samplers.layoutId decomposed.params = some paramsId)
    (queriesMatch : Mxx.HashQueriesMatchDecomposition plain decomposed base small digitCount)
    (publicResult :
      samplers.gadgetPublicMatrix paramsId decomposed.params
          plain.params.rows base small digitCount = some publicMatrix)
    (reconstructionAvailable :
      small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId decomposed.params = some limit ∧
        Mxx.maxCanonicalCoefficient
          ((samplers.hashSample plain).withSamplerParams plain.params) < limit) :
    Mxx.MatrixModEq
        (Mxx.matrixMul publicMatrix
          ((samplers.hashSample decomposed).withSamplerParams decomposed.params))
        ((samplers.hashSample plain).withSamplerParams plain.params) ∧
      Mxx.maxCenteredCoefficientNorm
          ((samplers.hashSample decomposed).withSamplerParams decomposed.params) ≤
        Mxx.gadgetDecompositionBound base small := by
  have decompositionResult := contract.decomposedHashConsistency paramsId plain decomposed base
    small digitCount layoutId queriesMatch
  exact gadgetDecompose_local_sound samplers contract paramsId decomposed.params base small digitCount
    ((samplers.hashSample plain).withSamplerParams plain.params) publicMatrix
    (samplers.hashSample decomposed) publicResult decompositionResult reconstructionAvailable

/-- The executable trapdoor sampler always pairs the public output with the same matrix stored in
the private trapdoor output. This proves pairing, but not the public matrix's centered-radius
bound. -/
theorem trapdoorNode_pairs_public_and_private
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (cutoff : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (outputCount : Nat)
    (matrixTypeEvaluate : matrixType.evaluate params cutoff = some matrixParams)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .trapdoorSample matrixType cutoff
      arguments := []
      outputCount
    }) :
    ∃ publicMatrix,
      publicMatrix ∈ samplers.trapdoorSample matrixParams ∧
      let normalized := publicMatrix.withSamplerParams matrixParams
      values = [.matrix normalized, .trapdoor normalized .sampled] ∧
        Mxx.maxCenteredCoefficientNorm normalized ≤ normalized.modulus.natAbs / 2 := by
  obtain ⟨publicMatrix, publicMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_trapdoorSample runChild samplers params inputs wires matrixType cutoff
      matrixParams outputCount matrixTypeEvaluate member
  refine ⟨publicMatrix, publicMember, valuesEq, matrix_norm_le_centered_radius _ ?_⟩
  simpa [Mxx.Matrix.withSamplerParams] using modulusPositive

end Mxx.Certificate
