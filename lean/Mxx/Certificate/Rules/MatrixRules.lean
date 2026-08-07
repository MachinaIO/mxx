import Mxx.Certificate.Semantics
import Mxx.Certificate.Typing

namespace Mxx.Certificate

private theorem reduce_sub_eq_add_reduce_neg
    (q left right : Int)
    (qPositive : 0 < q) :
    Mxx.reduceCoefficient q (left - right) =
      Mxx.reduceCoefficient q (left + Mxx.reduceCoefficient q (-right)) := by
  simp [Mxx.reduceCoefficient, not_le.mpr qPositive, sub_eq_add_neg, Int.add_emod]

private theorem subtractCoefficients_reduce_eq_add_negate
    (q : Int)
    (qPositive : 0 < q) :
    ∀ left right : List Int,
      (Mxx.subtractCoefficients left right).map (Mxx.reduceCoefficient q) =
        (Mxx.addCoefficients left (right.map fun coefficient ↦
          Mxx.reduceCoefficient q (-coefficient))).map (Mxx.reduceCoefficient q)
  | [], [] => rfl
  | [], rightHead :: rightTail => by
      simp [Mxx.subtractCoefficients, Mxx.addCoefficients, List.map_map, Function.comp_def,
        Mxx.reduceCoefficient, not_le.mpr qPositive]
  | leftHead :: leftTail, [] => by
      simp [Mxx.subtractCoefficients, Mxx.addCoefficients]
  | leftHead :: leftTail, rightHead :: rightTail => by
      simp only [Mxx.subtractCoefficients, Mxx.addCoefficients, List.map_cons]
      congr 1
      · exact reduce_sub_eq_add_reduce_neg q leftHead rightHead qPositive
      · exact subtractCoefficients_reduce_eq_add_negate q qPositive leftTail rightTail

theorem matrixSubtract_eq_matrixAdd_negate
    (left right : Mxx.Matrix)
    (sameModulus : right.modulus = left.modulus)
    (modulusPositive : 0 < left.modulus) :
    Mxx.matrixSubtract left right = Mxx.matrixAdd left (Mxx.matrixNegate right) := by
  cases left with
  | mk leftCoefficients modulus ringDimension rows columns =>
      cases right with
      | mk rightCoefficients rightModulus rightRingDimension rightRows rightColumns =>
          simp only at sameModulus modulusPositive ⊢
          subst rightModulus
          simp only [Mxx.matrixSubtract, Mxx.matrixAdd, Mxx.matrixNegate]
          congr 1
          exact subtractCoefficients_reduce_eq_add_negate modulus modulusPositive
            leftCoefficients rightCoefficients

theorem exactAdd_primary_local_sound
    {environment : FactEnvironment}
    {leftExpression rightExpression : MatrixExpr}
    {left right : Mxx.Matrix}
    (leftDenotes : leftExpression.Denotes environment left)
    (rightDenotes : rightExpression.Denotes environment right) :
    MatrixPrimaryForm.Holds environment
      (.exact (.add leftExpression rightExpression)) (Mxx.matrixAdd left right) :=
  .add leftDenotes rightDenotes

theorem exactSubtract_primary_local_sound
    {environment : FactEnvironment}
    {leftExpression rightExpression : MatrixExpr}
    {left right : Mxx.Matrix}
    (leftDenotes : leftExpression.Denotes environment left)
    (rightDenotes : rightExpression.Denotes environment right)
    (sameModulus : right.modulus = left.modulus)
    (modulusPositive : 0 < left.modulus) :
    MatrixPrimaryForm.Holds environment
      (.exact (.add leftExpression (.negate rightExpression)))
      (Mxx.matrixSubtract left right) := by
  rw [matrixSubtract_eq_matrixAdd_negate left right sameModulus modulusPositive]
  exact .add leftDenotes (.negate rightDenotes)

theorem exactNegate_primary_local_sound
    {environment : FactEnvironment}
    {expression : MatrixExpr}
    {value : Mxx.Matrix}
    (denotes : expression.Denotes environment value) :
    MatrixPrimaryForm.Holds environment
      (.exact (.negate expression)) (Mxx.matrixNegate value) :=
  .negate denotes

/-- Selecting one coefficient expression while keeping a common basis denotes the product with
the selected coefficient. A branch may be a typed zero expression, which is how affine select
aligns a basis that is absent from that branch. -/
theorem selectCoefficient_signalTerm_local_sound
    {environment : FactEnvironment}
    {index : RuntimeExpr .integer}
    {branches : List MatrixExpr}
    {indexValue : Int}
    {selected : MatrixExpr}
    {coefficientValue basisValue : Mxx.Matrix}
    {coefficientBound : BoundExpr}
    {coefficientBoundValue : Nat}
    {basis : MatrixExpr}
    {mode : SignalProductMode}
    (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
    (nonnegative : 0 ≤ indexValue)
    (branchSelected : branches[indexValue.toNat]? = some selected)
    (coefficientDenotes : MatrixExpr.Denotes environment selected coefficientValue)
    (boundEvaluates :
      coefficientBound.evaluate environment.parameters = .ok coefficientBoundValue)
    (coefficientNorm :
      Mxx.maxCenteredCoefficientNorm coefficientValue ≤ coefficientBoundValue)
    (basisDenotes : MatrixExpr.Denotes environment basis basisValue) :
    SignalTerm.Denotes environment {
      coefficient := {
        expression := .select index branches
        normBound := coefficientBound
      }
      basis
      mode
    } (Mxx.matrixMultiply coefficientValue basisValue) := by
  exact SignalTerm.Denotes.product
    ⟨.select indexDenotes nonnegative branchSelected coefficientDenotes,
      coefficientBoundValue, boundEvaluates, coefficientNorm⟩ basisDenotes

/-- The absent-basis case of affine select: the selected coefficient is a typed zero with exact
zero norm. -/
theorem selectZeroCoefficient_signalTerm_local_sound
    {environment : FactEnvironment}
    {index : RuntimeExpr .integer}
    {branches : List MatrixExpr}
    {indexValue : Int}
    {coefficientType : MatrixTypeExpr}
    {coefficientParams : Mxx.SamplerParams}
    {basis : MatrixExpr}
    {basisValue : Mxx.Matrix}
    {mode : SignalProductMode}
    (indexDenotes : RuntimeIntExpr.Denotes environment index indexValue)
    (nonnegative : 0 ≤ indexValue)
    (branchSelected : branches[indexValue.toNat]? = some (.zero coefficientType))
    (coefficientTypeEvaluates :
      coefficientType.evaluate environment.parameters = some coefficientParams)
    (basisDenotes : MatrixExpr.Denotes environment basis basisValue) :
    SignalTerm.Denotes environment {
      coefficient := {
        expression := .select index branches
        normBound := .constant 0
      }
      basis
      mode
    } (Mxx.matrixMultiply (zeroConstantOutput coefficientParams) basisValue) := by
  apply selectCoefficient_signalTerm_local_sound indexDenotes nonnegative branchSelected
    (.zero coefficientTypeEvaluates) rfl
  · simp [zeroConstant_norm_eq_zero]
  · exact basisDenotes

/-- Lift a selected, basis-aligned affine branch to the output affine form. The caller proves
termwise that each selected coefficient (including typed zeros for absent bases) denotes the same
term value, while the output noise bound conservatively dominates the selected branch bound. -/
theorem alignedAffineSelect_primary_local_sound
    {environment : FactEnvironment}
    {selected output : AffineForm}
    {value : Mxx.Matrix}
    {outputNoiseBound : Nat}
    (selectedHolds : selected.Holds environment value)
    (termsTransport : ∀ termValues,
      List.Forall₂ (SignalTerm.Denotes environment) selected.terms termValues →
        List.Forall₂ (SignalTerm.Denotes environment) output.terms termValues)
    (outputBoundEvaluates :
      output.noiseBound.evaluate environment.parameters = .ok outputNoiseBound)
    (boundDominates : ∀ selectedNoiseBound,
      selected.noiseBound.evaluate environment.parameters = .ok selectedNoiseBound →
        selectedNoiseBound ≤ outputNoiseBound) :
    MatrixPrimaryForm.Holds environment (.affine output) value := by
  obtain ⟨termValues, noise, selectedNoiseBound, selectedTermsDenote,
    selectedBoundEvaluates, noiseNorm, reconstruction⟩ := selectedHolds
  exact ⟨termValues, noise, outputNoiseBound, termsTransport termValues selectedTermsDenote,
    outputBoundEvaluates,
    noiseNorm.trans (boundDominates selectedNoiseBound selectedBoundEvaluates),
    reconstruction⟩

/-- A polynomial-scalar broadcast has the same worst-case negacyclic factor as a matrix product,
but no matrix contraction factor. -/
theorem matrixPolynomialScale_norm_le
    (q ringDimension scalarBound matrixBound : Nat)
    [NeZero q]
    (scalar matrix : Mxx.Matrix)
    (scalarModulus : scalar.modulus = q)
    (matrixModulus : matrix.modulus = q)
    (matrixRing : matrix.ringDimension = ringDimension)
    (scalarNorm : Mxx.maxCenteredCoefficientNorm scalar ≤ scalarBound)
    (matrixNorm : Mxx.maxCenteredCoefficientNorm matrix ≤ matrixBound) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixPolynomialScale scalar matrix) ≤
      ringDimension * scalarBound * matrixBound := by
  unfold Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro outputCoefficient outputMember
  simp only [List.mem_map] at outputMember
  obtain ⟨reduced, reducedMember, rfl⟩ := outputMember
  obtain ⟨linear, rfl⟩ := List.mem_ofFn.mp reducedMember
  let coefficient := linear.val % matrix.ringDimension
  let entry := linear.val / matrix.ringDimension
  let column := entry % matrix.columns
  let row := entry / matrix.columns
  have ringPositive : 0 < matrix.ringDimension := by
    by_contra nonpositive
    have zero : matrix.ringDimension = 0 := Nat.eq_zero_of_not_pos nonpositive
    have linearLt := linear.isLt
    simp [zero] at linearLt
  have coefficientLt : coefficient < matrix.ringDimension :=
    Nat.mod_lt _ ringPositive
  change
    (Mxx.centeredCoefficient matrix.modulus
      (Mxx.reduceCoefficient matrix.modulus
        (Mxx.negacyclicCoefficient matrix.ringDimension
          (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient))).natAbs ≤
      ringDimension * scalarBound * matrixBound
  rw [matrixModulus, Mxx.Toolkit.centeredCoefficient_reduce]
  rw [matrixRing]
  apply le_trans
    (Mxx.Toolkit.negacyclicCoefficient_natAbs_le q ringDimension
      (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient
      scalarBound matrixBound (by simpa [matrixRing] using coefficientLt) _ _)
  · rfl
  · intro index indexLt
    exact le_trans
      (by simpa [scalarModulus] using
        Mxx.Toolkit.centeredEntry_natAbs_le_norm scalar 0 0 index)
      scalarNorm
  · intro index indexLt
    exact le_trans
      (by simpa [matrixModulus, matrixRing] using
        Mxx.Toolkit.centeredEntry_natAbs_le_norm matrix row column index)
      matrixNorm

/-- The ordinary `matrixMultiply` branch is locally sound, including the exact executable output
and the full worst-case ring-by-contraction coefficient bound. The branch equality is discharged
by the typed analyzer from `inferMatrixProductType`. -/
theorem matrixMultiplyNode_ordinary_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q ringDimension inner leftBound rightBound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix left, .matrix right])
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixMul left right)
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftRing : left.ringDimension = ringDimension)
    (rightRing : right.ringDimension = ringDimension)
    (leftColumns : left.columns = inner)
    (rightRows : right.rows = inner)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixMultiply left right)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
        ringDimension * inner * leftBound * rightBound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs wires
      leftRef rightRef left right outputCount argumentsEvaluate member
  · rw [runtimeBranch]
    exact Mxx.Toolkit.matrixMul_norm_le q ringDimension inner leftBound rightBound left right
      leftModulus rightModulus leftRing rightRing leftColumns rightRows leftNorm rightNorm

/-- The row-vector/right-scalar boundary reverses the executable operands before ordinary
`matrixMul`. The hard bound remains symmetric in the operand bounds and has contraction one. -/
theorem matrixMultiplyNode_swapped_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (outputCount q ringDimension leftBound rightBound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix left, .matrix right])
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixMul right left)
    (leftModulus : left.modulus = q)
    (rightModulus : right.modulus = q)
    (leftRing : left.ringDimension = ringDimension)
    (rightRing : right.ringDimension = ringDimension)
    (rightColumns : right.columns = 1)
    (leftRows : left.rows = 1)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixMultiply left right)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
        ringDimension * 1 * leftBound * rightBound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs wires
      leftRef rightRef left right outputCount argumentsEvaluate member
  · rw [runtimeBranch]
    rw [show ringDimension * 1 * leftBound * rightBound =
      ringDimension * 1 * rightBound * leftBound by ac_rfl]
    exact Mxx.Toolkit.matrixMul_norm_le q ringDimension 1 rightBound leftBound right left
      rightModulus leftModulus rightRing leftRing rightColumns leftRows rightNorm leftNorm

/-- The polynomial-scalar broadcast branch is locally sound. Its contraction dimension is one,
so its bound is the same `matrixProduct` formula with the contraction factor specialized to one. -/
theorem matrixMultiplyNode_broadcast_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right scalar matrix : Mxx.Matrix)
    (outputCount q ringDimension leftBound rightBound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) =
        some [.matrix left, .matrix right])
    (runtimeBranch : Mxx.matrixMultiply left right = Mxx.matrixPolynomialScale scalar matrix)
    (matrixIsOtherOperand :
      (scalar = left ∧ matrix = right) ∨ (scalar = right ∧ matrix = left))
    (scalarModulus : scalar.modulus = q)
    (matrixModulus : matrix.modulus = q)
    (matrixRing : matrix.ringDimension = ringDimension)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixMultiply left right)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply left right) ≤
        ringDimension * 1 * leftBound * rightBound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixMultiply_of_arguments runChild samplers params inputs wires
      leftRef rightRef left right outputCount argumentsEvaluate member
  · rw [runtimeBranch]
    rcases matrixIsOtherOperand with ⟨scalarEq, matrixEq⟩ | ⟨scalarEq, matrixEq⟩
    · have scalarNorm' : Mxx.maxCenteredCoefficientNorm scalar ≤ leftBound := by
        simpa [scalarEq] using leftNorm
      have matrixNorm' : Mxx.maxCenteredCoefficientNorm matrix ≤ rightBound := by
        simpa [matrixEq] using rightNorm
      simpa [Nat.mul_assoc] using
        matrixPolynomialScale_norm_le q ringDimension leftBound rightBound scalar matrix
          scalarModulus matrixModulus matrixRing scalarNorm' matrixNorm'
    · have scalarNorm' : Mxx.maxCenteredCoefficientNorm scalar ≤ rightBound := by
        simpa [scalarEq] using rightNorm
      have matrixNorm' : Mxx.maxCenteredCoefficientNorm matrix ≤ leftBound := by
        simpa [matrixEq] using leftNorm
      rw [show ringDimension * 1 * leftBound * rightBound =
        ringDimension * rightBound * leftBound by ac_rfl]
      exact matrixPolynomialScale_norm_le q ringDimension rightBound leftBound scalar matrix
        scalarModulus matrixModulus matrixRing scalarNorm' matrixNorm'

/-- A checked preimage relation can only rewrite the exact source multiplied by that relation's
subject. The sampler's coefficient-wise modular relation is transported to the typed negacyclic
quotient; target coefficient representatives need not be canonical. -/
theorem applyPreimageRelation_local_sound
    (environment : FactEnvironment)
    (subjectRef : ValueInstanceRef)
    (sourceRef targetRef : MatrixInstanceRef)
    (trapdoorRef : ValueInstanceRef)
    (source subject target : Mxx.Matrix)
    (sourceParams targetParams : Mxx.SamplerParams)
    (subjectLookup : environment.values subjectRef = some (.matrix subject))
    (sourceLookup : environment.values sourceRef.value = some (.matrix source))
    (targetLookup : environment.values targetRef.value = some (.matrix target))
    (trapdoorLookup : environment.values trapdoorRef = some (.trapdoor source))
    (sourceTypeEvaluate :
      sourceRef.type.evaluate environment.parameters (.constant 0) = some sourceParams)
    (targetTypeEvaluate :
      targetRef.type.evaluate environment.parameters (.constant 0) = some targetParams)
    (sourceLayout : Mxx.Toolkit.MatrixLayout source sourceParams.modulus
      sourceParams.ringDimension sourceParams.rows sourceParams.columns)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject sourceParams.modulus
      sourceParams.ringDimension sourceParams.columns targetParams.columns)
    (targetLayout : Mxx.Toolkit.MatrixLayout target targetParams.modulus
      targetParams.ringDimension targetParams.rows targetParams.columns)
    (sameModulus : sourceParams.modulus = targetParams.modulus)
    (sameRingDimension : sourceParams.ringDimension = targetParams.ringDimension)
    (sameRows : sourceParams.rows = targetParams.rows)
    (modulusPositive : 0 < sourceParams.modulus)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul source subject) target) :
    MatrixRelation.Holds environment
      (.preimage subjectRef sourceRef targetRef trapdoorRef) := by
  let q := sourceParams.modulus.toNat
  letI : NeZero q := ⟨Nat.ne_of_gt (by simp [q]; exact modulusPositive)⟩
  have modulusEq : (q : Int) = sourceParams.modulus := by
    exact Int.toNat_of_nonneg modulusPositive.le
  have sourceLayoutQ : Mxx.Toolkit.MatrixLayout source q sourceParams.ringDimension
      sourceParams.rows sourceParams.columns := by
    simpa [modulusEq] using sourceLayout
  have subjectLayoutQ : Mxx.Toolkit.MatrixLayout subject q sourceParams.ringDimension
      sourceParams.columns targetParams.columns := by
    simpa [modulusEq] using subjectLayout
  have targetLayoutQ : Mxx.Toolkit.MatrixLayout target q sourceParams.ringDimension
      sourceParams.rows targetParams.columns := by
    rw [← sameModulus, ← sameRingDimension, ← sameRows] at targetLayout
    simpa [modulusEq] using targetLayout
  have productLayout := Mxx.Toolkit.matrixMul_layout source subject sourceLayoutQ subjectLayoutQ
  exact ⟨subject, source, target, sourceParams, targetParams, subjectLookup, sourceLookup,
    targetLookup, trapdoorLookup, sourceTypeEvaluate, targetTypeEvaluate, sourceLayout,
    subjectLayout, targetLayout, sameModulus, sameRingDimension, sameRows,
    Mxx.Toolkit.matrixValue_eq_of_modEq q sourceParams.ringDimension sourceParams.rows
      targetParams.columns _ _ productLayout targetLayoutQ relation⟩

/-- A checked gadget-decomposition relation rewrites only the exact gadget/subject pair from the
sampler contract. -/
theorem applyGadgetDecompositionRelation_local_sound
    (environment : FactEnvironment)
    (subjectRef : ValueInstanceRef)
    (targetRef : MatrixInstanceRef)
    (base digitCount : IntExpr)
    (subject target : Mxx.Matrix)
    (matrixParams : Mxx.SamplerParams)
    (evaluatedBase evaluatedDigitCount : Int)
    (subjectLookup : environment.values subjectRef = some (.matrix subject))
    (targetLookup : environment.values targetRef.value = some (.matrix target))
    (typeEvaluate :
      targetRef.type.evaluate environment.parameters (.constant 0) = some matrixParams)
    (baseEvaluate : evaluateIntExpr environment.parameters base = .ok evaluatedBase)
    (digitCountEvaluate :
      evaluateIntExpr environment.parameters digitCount = .ok evaluatedDigitCount)
    (targetLayout : Mxx.Toolkit.MatrixLayout target matrixParams.modulus
      matrixParams.ringDimension matrixParams.rows matrixParams.columns)
    (subjectLayout : Mxx.Toolkit.MatrixLayout subject matrixParams.modulus
      matrixParams.ringDimension (matrixParams.rows * evaluatedDigitCount.toNat)
      matrixParams.columns)
    (modulusPositive : 0 < matrixParams.modulus)
    (relation : Mxx.MatrixModEq
      (Mxx.matrixMul
        (Mxx.gadgetMatrix {
          matrixParams with
          columns := matrixParams.rows * evaluatedDigitCount.toNat
        } evaluatedBase evaluatedDigitCount.toNat)
        subject) target) :
    MatrixRelation.Holds environment
      (.gadgetDecomposition subjectRef targetRef base digitCount) :=
  by
    let q := matrixParams.modulus.toNat
    letI : NeZero q := ⟨Nat.ne_of_gt (by simp [q]; exact modulusPositive)⟩
    have modulusEq : (q : Int) = matrixParams.modulus :=
      Int.toNat_of_nonneg modulusPositive.le
    have targetLayoutQ : Mxx.Toolkit.MatrixLayout target q matrixParams.ringDimension
        matrixParams.rows matrixParams.columns := by
      simpa [modulusEq] using targetLayout
    have subjectLayoutQ : Mxx.Toolkit.MatrixLayout subject q matrixParams.ringDimension
        (matrixParams.rows * evaluatedDigitCount.toNat) matrixParams.columns := by
      simpa [modulusEq] using subjectLayout
    have gadgetLayoutQ : Mxx.Toolkit.MatrixLayout
        (Mxx.gadgetMatrix {
          matrixParams with
          columns := matrixParams.rows * evaluatedDigitCount.toNat
        } evaluatedBase evaluatedDigitCount.toNat)
        q matrixParams.ringDimension matrixParams.rows
          (matrixParams.rows * evaluatedDigitCount.toNat) := by
      simpa [modulusEq] using Mxx.Toolkit.gadgetMatrix_layout
        { matrixParams with columns := matrixParams.rows * evaluatedDigitCount.toNat }
        evaluatedBase evaluatedDigitCount.toNat
    have productLayout := Mxx.Toolkit.matrixMul_layout _ _ gadgetLayoutQ subjectLayoutQ
    exact ⟨subject, target, matrixParams, evaluatedBase, evaluatedDigitCount, subjectLookup,
      targetLookup, typeEvaluate, baseEvaluate, digitCountEvaluate, targetLayout, subjectLayout,
      Mxx.Toolkit.matrixValue_eq_of_modEq q matrixParams.ringDimension matrixParams.rows
        matrixParams.columns _ _ productLayout targetLayoutQ relation⟩

/-- Symbolic identity elimination is valid at the quotient-matrix denotation used by correctness,
provided the executable matrix has the checked layout. -/
theorem identityMultiplyLeft_local_sound
    (q ringDimension rows columns bound : Nat)
    [NeZero q] [NeZero ringDimension]
    (right : Mxx.Matrix)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension rows columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply
          (identityConstantOutput {
            maxCoefficientBound := bound
            modulus := q
            ringDimension
            rows
            columns := rows
          }) right) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns right :=
  identityConstant_matrixValue_mul_left q ringDimension rows columns bound right rightLayout

/-- `MatrixScale(1)` is an identity at the quotient-matrix denotation used by correctness. This
does not claim raw `Mxx.Matrix` equality. The relation-retargeting theorems below remain entirely
in the quotient and therefore require no canonical representative for the subject or target. -/
theorem matrixScaleOne_local_sound
    (q ringDimension rows columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (matrix : Mxx.Matrix)
    (layout : Mxx.Toolkit.MatrixLayout matrix q ringDimension rows columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns (Mxx.matrixScale 1 matrix) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns matrix := by
  rw [Mxx.Toolkit.matrixValue_scale q ringDimension rows columns 1 matrix
    ⟨layout.modulus, layout.ringDimension, layout.rows, layout.columns⟩]
  simp

/-- Multiplying by a subject after executable `MatrixScale(1)` preserves the product in the
negacyclic quotient. This deliberately does not claim raw equality of either subject or product:
`matrixScale` canonicalizes stored coefficients, while protocol relations are equations in
`R_q`. -/
theorem matrixMul_scaleOne_right_matrixValue
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (source subject : Mxx.Matrix)
    (sourceType : source.modulus = q ∧ source.ringDimension = ringDimension ∧
      source.rows = rows ∧ source.columns = inner)
    (subjectType : subject.modulus = q ∧ subject.ringDimension = ringDimension ∧
      subject.rows = inner ∧ subject.columns = columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMul source (Mxx.matrixScale 1 subject)) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns (Mxx.matrixMul source subject) := by
  have scaledType : (Mxx.matrixScale 1 subject).modulus = q ∧
      (Mxx.matrixScale 1 subject).ringDimension = ringDimension ∧
      (Mxx.matrixScale 1 subject).rows = inner ∧
      (Mxx.matrixScale 1 subject).columns = columns := by
    simpa [Mxx.matrixScale] using subjectType
  rw [Mxx.Toolkit.matrixValue_mul q ringDimension rows inner columns source
      (Mxx.matrixScale 1 subject) sourceType scaledType,
    Mxx.Toolkit.matrixValue_mul q ringDimension rows inner columns source subject sourceType
      subjectType,
    Mxx.Toolkit.matrixValue_scale q ringDimension inner columns 1 subject subjectType]
  simp

/-- A preimage equation in `R_q` remains true after retargeting its subject to the executable
`MatrixScale(1)` output. The target may use any coefficient representative of the same quotient
matrix; no canonical-target premise is required. -/
theorem preimageRelation_scaleOne_subject_matrixValue
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (source subject target : Mxx.Matrix)
    (sourceType : source.modulus = q ∧ source.ringDimension = ringDimension ∧
      source.rows = rows ∧ source.columns = inner)
    (subjectType : subject.modulus = q ∧ subject.ringDimension = ringDimension ∧
      subject.rows = inner ∧ subject.columns = columns)
    (relation : Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMul source subject) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns target) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMul source (Mxx.matrixScale 1 subject)) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns target := by
  rw [matrixMul_scaleOne_right_matrixValue q ringDimension rows inner columns source subject
    sourceType subjectType]
  exact relation

/-- The same quotient-preserving retargeting specialized to a gadget-decomposition equation.
The theorem is intentionally independent of how the gadget matrix was constructed; the analyzer
must still establish its checked layout and exact base/digit-count provenance. -/
theorem gadgetDecompositionRelation_scaleOne_subject_matrixValue
    (q ringDimension rows digitColumns columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (gadget subject target : Mxx.Matrix)
    (gadgetType : gadget.modulus = q ∧ gadget.ringDimension = ringDimension ∧
      gadget.rows = rows ∧ gadget.columns = digitColumns)
    (subjectType : subject.modulus = q ∧ subject.ringDimension = ringDimension ∧
      subject.rows = digitColumns ∧ subject.columns = columns)
    (relation : Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMul gadget subject) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns target) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMul gadget (Mxx.matrixScale 1 subject)) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns target := by
  exact preimageRelation_scaleOne_subject_matrixValue q ringDimension rows digitColumns columns
    gadget subject target gadgetType subjectType relation

end Mxx.Certificate
