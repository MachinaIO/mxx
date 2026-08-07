import Mxx.Certificate.LocalSoundness

namespace Mxx.Certificate

/-- Closed executable inversion for the zero-matrix constructor. -/
theorem zeroMatrixNode_local_sound
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (matrixType : Mxx.Ir.MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams) (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .zeroMatrix matrixType, arguments := [], outputCount }) :
    values = [.matrix (zeroConstantOutput matrixParams)] ∧
      Mxx.maxCenteredCoefficientNorm (zeroConstantOutput matrixParams) = 0 := by
  constructor
  · simpa [Mxx.Ir.evaluateNode, typeEvaluates, zeroConstantOutput] using member
  · exact zeroConstant_norm_eq_zero matrixParams

/-- Closed executable inversion for the identity-matrix constructor. -/
theorem identityMatrixNode_local_sound
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (matrixType : Mxx.Ir.MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams) (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .identityMatrix matrixType, arguments := [], outputCount }) :
    values = [.matrix (identityConstantOutput matrixParams)] ∧
      Mxx.maxCenteredCoefficientNorm (identityConstantOutput matrixParams) ≤ 1 := by
  constructor
  · simpa [Mxx.Ir.evaluateNode, typeEvaluates, identityConstantOutput] using member
  · exact identityConstant_norm_le_one matrixParams modulusPositive

/-- Closed executable inversion for a literal polynomial matrix. -/
theorem constantMatrixNode_local_sound
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (matrixType : Mxx.Ir.MatrixTypeExpr)
    (coefficients : List Mxx.Ir.IntExpr) (matrixParams : Mxx.SamplerParams)
    (evaluated : List Int) (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    (coefficientsEvaluate : coefficients.mapM (Mxx.Ir.IntExpr.evaluate params) = some evaluated)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .constantMatrix matrixType coefficients, arguments := [], outputCount }) :
    let output := Mxx.Matrix.withSamplerParams {
      coefficients := evaluated.map (Mxx.reduceCoefficient matrixParams.modulus)
    } matrixParams
    values = [.matrix output] ∧
      Mxx.maxCenteredCoefficientNorm output ≤ output.modulus.natAbs / 2 := by
  dsimp
  constructor
  · simpa [Mxx.Ir.evaluateNode, typeEvaluates, coefficientsEvaluate] using member
  · exact matrix_norm_le_centered_radius _
      (by simpa [Mxx.Matrix.withSamplerParams] using modulusPositive)

/-- Closed executable inversion for the deterministic gadget matrix. -/
theorem gadgetMatrixNode_local_sound
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (matrixType : Mxx.Ir.MatrixTypeExpr)
    (base : Mxx.Ir.IntExpr) (matrixParams : Mxx.SamplerParams)
    (evaluatedBase : Int) (outputCount : Nat)
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    (baseEvaluates : base.evaluate params = some evaluatedBase)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .gadgetMatrix matrixType base, arguments := [], outputCount }) :
    let digits := if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows
    let output := Mxx.gadgetMatrix matrixParams evaluatedBase digits
    values = [.matrix output] ∧
      Mxx.maxCenteredCoefficientNorm output ≤ output.modulus.natAbs / 2 := by
  dsimp
  constructor
  · simpa [Mxx.Ir.evaluateNode, typeEvaluates, baseEvaluates] using member
  · exact matrix_norm_le_centered_radius _
      (by simp [Mxx.gadgetMatrix, Mxx.Matrix.withSamplerParams, modulusPositive])

/-- Closed executable inversion for the plain deterministic hash leaf used by Diamond. -/
theorem plainHashNode_local_sound
    (runChild : Mxx.Ir.ChildRunner) (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment) (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment) (keyRef : Mxx.Ir.WireRef) (key : ByteArray)
    (matrixType : Mxx.Ir.MatrixTypeExpr) (tagPrefix : List Nat)
    (tagExpressions tagDecimalExpressions tagU64LeExpressions : List Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (tagValues tagDecimalValues tagU64LeValues : List Int) (outputCount : Nat)
    (argumentsEvaluate : [keyRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.bytes key])
    (typeEvaluates : matrixType.evaluate params (.constant 0) = some matrixParams)
    (tagsEvaluate : tagExpressions.mapM (Mxx.Ir.IntExpr.evaluate params) = some tagValues)
    (decimalTagsEvaluate : tagDecimalExpressions.mapM (Mxx.Ir.IntExpr.evaluate params) =
      some tagDecimalValues)
    (u64TagsEvaluate : tagU64LeExpressions.mapM (Mxx.Ir.IntExpr.evaluate params) =
      some tagU64LeValues)
    (modulusPositive : 0 < matrixParams.modulus)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .hashSample matrixType .plain tagPrefix tagExpressions tagDecimalExpressions
        tagU64LeExpressions none none
      arguments := [keyRef]
      outputCount
    }) :
    let output := (samplers.hashSample {
      params := matrixParams
      key
      variant := .plain
      tagPrefix
      tagValues
      tagDecimalValues
      tagU64LeValues
      base := none
      digitCount := none
    }).withSamplerParams matrixParams
    values = [.matrix output] ∧
      Mxx.maxCenteredCoefficientNorm output ≤ output.modulus.natAbs / 2 := by
  dsimp
  constructor
  · exact Mxx.Ir.mem_evaluateNode_hashSample_of_arguments runChild samplers params inputs wires
      keyRef key matrixType .plain tagPrefix tagExpressions tagDecimalExpressions
      tagU64LeExpressions none none matrixParams tagValues tagDecimalValues tagU64LeValues
      none none outputCount argumentsEvaluate typeEvaluates tagsEvaluate decimalTagsEvaluate
      u64TagsEvaluate rfl rfl member
  · exact matrix_norm_le_centered_radius _
      (by simpa [Mxx.Matrix.withSamplerParams] using modulusPositive)

end Mxx.Certificate
