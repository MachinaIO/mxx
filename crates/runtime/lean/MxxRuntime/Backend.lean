import MxxIrCore.Eval
import MxxPrimitives

namespace Mxx.Runtime

open scoped BigOperators

abbrev MatrixValue (matrixType : IR.MatrixType) :=
  Primitives.ExactMatrix matrixType.modulus.toNat matrixType.ringDimension matrixType.rows
    matrixType.columns

/-- Runtime-private trapdoor state. Its representation is intentionally not modeled. -/
opaque TrapdoorPrivateState : Type := Unit

/-- A trapdoor exposes only its exact public matrix to the structural evaluator. -/
structure TrapdoorValue (trapdoorType : IR.TrapdoorType) where
  privateState : TrapdoorPrivateState
  publicMatrix : MatrixValue trapdoorType.matrix

/-- A preimage is exactly the sampled matrix. Its sampling relation is proved separately. -/
structure PreimageValue (matrixType : IR.MatrixType) where
  exactMatrix : MatrixValue matrixType

structure RuntimeGadgetCertificate (targetType : IR.MatrixType) (layout : IR.GadgetLayout)
    (_structural : IR.StructuralEnv)
    (gadget : MatrixValue (IR.gadgetMatrixType targetType layout))
    (target : MatrixValue targetType)
    (preimage : PreimageValue (IR.gadgetPreimageType targetType layout)) : Prop where
  regular : layout.mode = .regular
  valid : layout.Valid
  relation : Primitives.RightPreimage gadget preimage.exactMatrix target
  bounded : Primitives.PreimageWithin preimage.exactMatrix (layout.base - 1)

def RuntimeGadgetOracle : Type :=
  ∀ (targetType : IR.MatrixType) (layout : IR.GadgetLayout)
    (structural : IR.StructuralEnv)
    (gadget : MatrixValue (IR.gadgetMatrixType targetType layout))
    (target : MatrixValue targetType),
    Except IR.GadgetFailure
      (Σ preimage : PreimageValue (IR.gadgetPreimageType targetType layout), PLift
        (RuntimeGadgetCertificate targetType layout structural gadget target preimage))

/-- Typed blobs retain bytes while their type name and schema hash remain in the Lean type. -/
structure TypedBlobValue (_typeName : String) (_schemaHash : List UInt8) where
  bytes : List UInt8
  deriving Repr, DecidableEq

private noncomputable def castPoly {q₁ q₂ n₁ n₂ : Nat} (hq : q₁ = q₂)
    (hn : n₁ = n₂) : Primitives.ExactPoly q₁ n₁ → Primitives.ExactPoly q₂ n₂ := by
  subst q₂
  subst n₂
  exact id

noncomputable def exactIdentity (matrixType : IR.MatrixType) : MatrixValue matrixType :=
  fun row column ↦ if row.val = column.val then 1 else 0

noncomputable def exactScale (matrixType : IR.MatrixType) (scalar : Int)
    (matrix : MatrixValue matrixType) : MatrixValue matrixType :=
  fun row column ↦ scalar * matrix row column

noncomputable def exactConstantPoly (q n : Nat) (value : Int) : Primitives.ExactPoly q n :=
  algebraMap (ZMod q) (Primitives.ExactPoly q n) (value : ZMod q)

noncomputable def exactPolynomial (q n : Nat) (coefficients : Array Int) :
    Primitives.ExactPoly q n :=
  ∑ index : Fin coefficients.size,
    algebraMap (ZMod q) (Primitives.ExactPoly q n)
      (coefficients[index]'index.isLt : ZMod q) *
      (AdjoinRoot.root (Primitives.negacyclicModulus n (ZMod q))) ^ index.val

noncomputable def exactPolynomialResidues {q n : Nat} (coefficients : Fin n → ZMod q) :
    Primitives.ExactPoly q n :=
  ∑ index : Fin n,
    algebraMap (ZMod q) (Primitives.ExactPoly q n) (coefficients index) *
      (AdjoinRoot.root (Primitives.negacyclicModulus n (ZMod q))) ^ index.val

noncomputable def exactSlice (input output : IR.MatrixType)
    (rowStart rowStop columnStart columnStop : Nat) (matrix : MatrixValue input) :
    MatrixValue output := by
  classical
  exact if h : input.modulus = output.modulus ∧ input.ringDimension = output.ringDimension ∧
      output.rows = rowStop - rowStart ∧ output.columns = columnStop - columnStart ∧
      rowStart ≤ rowStop ∧ rowStop ≤ input.rows ∧ columnStart ≤ columnStop ∧
      columnStop ≤ input.columns then
    fun row column ↦
      castPoly (congrArg Int.toNat h.1) h.2.1
        (matrix ⟨rowStart + row.val, by omega⟩ ⟨columnStart + column.val, by omega⟩)
  else 0

noncomputable def concatPolyAt (output : IR.MatrixType) (axis : IR.ConcatAxis)
    (parts : List (Σ t : IR.MatrixType, MatrixValue t)) (row column : Nat) :
    Primitives.ExactPoly output.modulus.toNat output.ringDimension :=
  match parts with
  | [] => 0
  | ⟨matrixType, matrix⟩ :: rest =>
      let castEntry (entry : Primitives.ExactPoly matrixType.modulus.toNat matrixType.ringDimension) :=
        if h : matrixType.modulus.toNat = output.modulus.toNat ∧
            matrixType.ringDimension = output.ringDimension then
          castPoly h.1 h.2 entry
        else 0
      match axis with
      | .rows =>
          if hRow : row < matrixType.rows then
            if hColumn : column < matrixType.columns then
              castEntry (matrix ⟨row, hRow⟩ ⟨column, hColumn⟩)
            else 0
          else concatPolyAt output axis rest (row - matrixType.rows) column
      | .columns =>
          if hColumn : column < matrixType.columns then
            if hRow : row < matrixType.rows then
              castEntry (matrix ⟨row, hRow⟩ ⟨column, hColumn⟩)
            else 0
          else concatPolyAt output axis rest row (column - matrixType.columns)
      | .diagonal =>
          if h : row < matrixType.rows ∧ column < matrixType.columns then
            castEntry (matrix ⟨row, h.1⟩ ⟨column, h.2⟩)
          else if matrixType.rows ≤ row ∧ matrixType.columns ≤ column then
            concatPolyAt output axis rest (row - matrixType.rows) (column - matrixType.columns)
          else 0

noncomputable def exactConcat (output : IR.MatrixType) (axis : IR.ConcatAxis)
    (parts : Array (Σ t : IR.MatrixType, MatrixValue t)) : MatrixValue output :=
  fun row column ↦ concatPolyAt output axis parts.toList row.val column.val

noncomputable def exactDecomposePoly {q n : Nat} (base digits : Nat)
    (value : Primitives.ExactPoly q n) : Array (Primitives.ExactPoly q n) :=
  let radix := if base = 0 then 1 else base
  (Array.range digits).map fun digit => exactPolynomialResidues fun coefficient =>
    let residue := (Primitives.Negacyclic.coeff value coefficient).val
    ((residue / radix ^ digit) % radix : ZMod q)

noncomputable def exactGadgetDecompose (input output : IR.MatrixType)
    (layout : IR.GadgetLayout) (matrix : MatrixValue input) :
    PreimageValue output := by
  classical
  exact {
    exactMatrix := if h : input.modulus = output.modulus ∧
        input.ringDimension = output.ringDimension ∧ output.rows = input.rows * layout.digits ∧
        output.columns = input.columns ∧ layout.mode = .regular ∧ layout.Valid then
      fun row column ↦
        let sourceRow := row.val / layout.digits
        let digit := row.val % layout.digits
        let decomposed := exactDecomposePoly layout.base layout.digits
            (castPoly (congrArg Int.toNat h.1) h.2.1
              (matrix ⟨sourceRow, Nat.div_lt_of_lt_mul (by
                have hRows : output.rows = layout.digits * input.rows :=
                  h.2.2.1.trans (Nat.mul_comm _ _)
                have : row.val < layout.digits * input.rows := by
                  calc
                    row.val < output.rows := row.isLt
                    _ = layout.digits * input.rows := hRows
                exact this)⟩
              ⟨column.val, by omega⟩))
        decomposed[digit]'(by
          have bound : digit < layout.digits := Nat.mod_lt _ h.2.2.2.2.2.2.1
          simpa [decomposed, exactDecomposePoly] using bound)
    else 0 }

noncomputable def exactExtractCoefficient (matrixType : IR.MatrixType) (position : Nat)
    (matrix : MatrixValue matrixType) : Int :=
  if h : matrixType.rows = 1 ∧ matrixType.columns = 1 ∧ position < matrixType.ringDimension then
    Int.ofNat (Primitives.Negacyclic.coeff
      (matrix ⟨0, by omega⟩ ⟨0, by omega⟩) ⟨position, h.2.2⟩).val
  else 0

@[simp] theorem exactExtractCoefficient_valid {matrixType : IR.MatrixType} {position : Nat}
    (rows : matrixType.rows = 1) (columns : matrixType.columns = 1)
    (positionBound : position < matrixType.ringDimension) (matrix : MatrixValue matrixType) :
    exactExtractCoefficient matrixType position matrix =
      Int.ofNat (Primitives.Negacyclic.coeff
        (matrix ⟨0, by omega⟩ ⟨0, by omega⟩) ⟨position, positionBound⟩).val := by
  simp [exactExtractCoefficient, rows, columns, positionBound]

noncomputable def exactConstantMatrix (matrixType : IR.MatrixType) (literal : IR.MatrixLiteral)
    (structural : IR.StructuralEnv) : MatrixValue matrixType := by
  classical
  let q := matrixType.modulus.toNat
  let n := matrixType.ringDimension
  let evaluate (expression : IR.StructuralIntExpr) : Int :=
    match expression.eval structural with
    | .ok value => value
    | .error _ => 0
  exact match literal with
  | .zero => 0
  | .identity => exactIdentity matrixType
  | .unitRow index =>
      fun row column ↦ if row.val = 0 ∧ column.val = (evaluate index).toNat then 1 else 0
  | .unitColumn index =>
      fun row column ↦ if row.val = (evaluate index).toNat ∧ column.val = 0 then 1 else 0
  | .gadget base small =>
      fun row column ↦
        let digits := if matrixType.rows = 0 then 0 else matrixType.columns / matrixType.rows
        let digit := if digits = 0 then 0 else column.val % digits
        let weight := (evaluate base) ^ digit
        if digits > 0 ∧ row.val = column.val / digits then exactConstantPoly q n weight else 0
  | .powerOfBase base exponent =>
      fun _ _ ↦ exactConstantPoly q n ((evaluate base) ^ (evaluate exponent).toNat)
  | .rotation exponent =>
      fun _ _ ↦ (AdjoinRoot.root (Primitives.negacyclicModulus n (ZMod q))) ^ (evaluate exponent).toNat
  | .polynomial coefficients =>
      fun _ _ ↦ exactPolynomial q n (coefficients.map (evaluate ·))

noncomputable def exactMultiply (left right output : IR.MatrixType)
    (valid : left.Valid ∧ right.Valid ∧ output.Valid ∧ IR.matrixProductType left right output)
    (a : MatrixValue left) (b : MatrixValue right) : MatrixValue output := by
  classical
  rcases valid with ⟨leftValid, rightValid, _outputValid, product⟩
  rcases product with ⟨leftRightRing, leftOutputRing, dimensions⟩
  rcases leftRightRing with ⟨leftRightModulus, leftRightDimension⟩
  rcases leftOutputRing with ⟨leftOutputModulus, leftOutputDimension⟩
  by_cases leftScalar : left.rows = 1 ∧ left.columns = 1
  · rw [if_pos leftScalar] at dimensions
    rcases dimensions with ⟨outputRows, outputColumns⟩
    exact fun row column ↦
      castPoly (congrArg Int.toNat leftOutputModulus) leftOutputDimension
          (a ⟨0, by omega⟩ ⟨0, by omega⟩) *
        castPoly
          (congrArg Int.toNat (leftRightModulus.symm.trans leftOutputModulus))
          (leftRightDimension.symm.trans leftOutputDimension)
          (b (Fin.cast outputRows row) (Fin.cast outputColumns column))
  · rw [if_neg leftScalar] at dimensions
    by_cases rightScalar : right.rows = 1 ∧ right.columns = 1
    · rw [if_pos rightScalar] at dimensions
      rcases dimensions with ⟨outputRows, outputColumns⟩
      exact fun row column ↦
        castPoly (congrArg Int.toNat leftOutputModulus) leftOutputDimension
            (a (Fin.cast outputRows row) (Fin.cast outputColumns column)) *
          castPoly
            (congrArg Int.toNat (leftRightModulus.symm.trans leftOutputModulus))
            (leftRightDimension.symm.trans leftOutputDimension)
            (b ⟨0, by omega⟩ ⟨0, by omega⟩)
    · rw [if_neg rightScalar] at dimensions
      rcases dimensions with ⟨inner, outputRows, outputColumns⟩
      exact fun row column ↦ ∑ index : Fin left.columns,
        castPoly (congrArg Int.toNat leftOutputModulus) leftOutputDimension
            (a (Fin.cast outputRows row) index) *
          castPoly
            (congrArg Int.toNat (leftRightModulus.symm.trans leftOutputModulus))
            (leftRightDimension.symm.trans leftOutputDimension)
            (b (Fin.cast inner index) (Fin.cast outputColumns column))

/- Canonical descriptor used when a runtime matrix directly represents an
   exact matrix over a natural modulus. -/
def naturalMatrixType (q n rows columns : Nat) : IR.MatrixType := {
  modulus := Int.ofNat q
  ringDimension := n
  rows := rows
  columns := columns
}

def matrixValue_naturalMatrixType (q n rows columns : Nat) :
    MatrixValue (naturalMatrixType q n rows columns) =
      Primitives.ExactMatrix q n rows columns := by
  rfl

noncomputable def naturalToExact {q n rows columns : Nat}
    (value : MatrixValue (naturalMatrixType q n rows columns)) :
    Primitives.ExactMatrix q n rows columns :=
  matrixValue_naturalMatrixType q n rows columns ▸ value

@[simp] private theorem andRec_pair_const {a b c : Prop} {alpha : Sort u}
    (proof : a ∧ b ∧ c) (value : alpha) :
    And.rec (fun _ rest ↦ And.rec (fun _ _ ↦ value) rest) proof = value := by
  rcases proof with ⟨_, _, _⟩
  rfl

/- For a non-scalar `1 × g` row times a `g × 1` column, the runtime
   implementation is exactly the ordinary inner-product formula.  Keeping
   this lemma beside `castPoly` makes every same-ring cast reduce locally. -/
theorem exactMultiply_natural_row_column {q n gadgetColumns : Nat}
    (hg : 1 < gadgetColumns)
    (valid :
      (naturalMatrixType q n 1 gadgetColumns).Valid ∧
      (naturalMatrixType q n gadgetColumns 1).Valid ∧
      (naturalMatrixType q n 1 1).Valid ∧
      IR.matrixProductType (naturalMatrixType q n 1 gadgetColumns)
        (naturalMatrixType q n gadgetColumns 1) (naturalMatrixType q n 1 1))
    (left : MatrixValue (naturalMatrixType q n 1 gadgetColumns))
    (right : MatrixValue (naturalMatrixType q n gadgetColumns 1)) :
    naturalToExact (exactMultiply (naturalMatrixType q n 1 gadgetColumns)
      (naturalMatrixType q n gadgetColumns 1) (naturalMatrixType q n 1 1)
      valid left right) = naturalToExact left * naturalToExact right := by
  rcases valid with
    ⟨leftValid, rightValid, outputValid,
      ⟨leftRightRing, leftOutputRing, dimensions⟩⟩
  have leftNotScalar : ¬((naturalMatrixType q n 1 gadgetColumns).rows = 1 ∧
      (naturalMatrixType q n 1 gadgetColumns).columns = 1) := by
    simp [naturalMatrixType, Nat.ne_of_gt hg]
  have rightNotScalar : ¬((naturalMatrixType q n gadgetColumns 1).rows = 1 ∧
      (naturalMatrixType q n gadgetColumns 1).columns = 1) := by
    simp [naturalMatrixType, Nat.ne_of_gt hg]
  have leftRightRingEq : leftRightRing = ⟨rfl, rfl⟩ := Subsingleton.elim _ _
  have leftOutputRingEq : leftOutputRing = ⟨rfl, rfl⟩ := Subsingleton.elim _ _
  subst leftRightRing
  subst leftOutputRing
  rw [if_neg leftNotScalar, if_neg rightNotScalar] at dimensions
  have dimensionsEq : dimensions = ⟨rfl, rfl, rfl⟩ := Subsingleton.elim _ _
  subst dimensions
  ext row column
  simp [naturalToExact, exactMultiply,
    naturalMatrixType, castPoly, Nat.ne_of_gt hg, Matrix.mul_apply, andRec_pair_const]

/- Canonical non-scalar rectangular descriptors use the ordinary matrix
   product.  In particular, the injector target equation interprets its
   `sourceRows × sourceRows` selector times the `sourceRows × columns` next
   public base without introducing a runtime-specific multiplication term. -/
theorem exactMultiply_natural_rectangular {q n rows inner columns : Nat}
    (leftNotScalar : ¬(rows = 1 ∧ inner = 1))
    (rightNotScalar : ¬(inner = 1 ∧ columns = 1))
    (valid :
      (naturalMatrixType q n rows inner).Valid ∧
      (naturalMatrixType q n inner columns).Valid ∧
      (naturalMatrixType q n rows columns).Valid ∧
      IR.matrixProductType (naturalMatrixType q n rows inner)
        (naturalMatrixType q n inner columns) (naturalMatrixType q n rows columns))
    (left : MatrixValue (naturalMatrixType q n rows inner))
    (right : MatrixValue (naturalMatrixType q n inner columns)) :
    naturalToExact (exactMultiply (naturalMatrixType q n rows inner)
      (naturalMatrixType q n inner columns) (naturalMatrixType q n rows columns)
      valid left right) = naturalToExact left * naturalToExact right := by
  rcases valid with
    ⟨leftValid, rightValid, outputValid,
      ⟨leftRightRing, leftOutputRing, dimensions⟩⟩
  have leftRightRingEq : leftRightRing = ⟨rfl, rfl⟩ := Subsingleton.elim _ _
  have leftOutputRingEq : leftOutputRing = ⟨rfl, rfl⟩ := Subsingleton.elim _ _
  subst leftRightRing
  subst leftOutputRing
  rw [if_neg (by simpa [naturalMatrixType] using leftNotScalar),
    if_neg (by simpa [naturalMatrixType] using rightNotScalar)] at dimensions
  have dimensionsEq : dimensions = ⟨rfl, rfl, rfl⟩ := Subsingleton.elim _ _
  subst dimensions
  ext row column
  simp [naturalToExact, exactMultiply, naturalMatrixType, castPoly,
    leftNotScalar, rightNotScalar, Matrix.mul_apply, andRec_pair_const]

private noncomputable def multiply (left right output : IR.MatrixType)
    (a : MatrixValue left) (b : MatrixValue right) : MatrixValue output := by
  classical
  exact if valid : left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output then
    exactMultiply left right output valid a b
  else 0

noncomputable def exactTranspose (input output : IR.MatrixType)
    (valid : input.Valid ∧ output.Valid ∧ IR.sameRing input output ∧
      output.rows = input.columns ∧ output.columns = input.rows)
    (matrix : MatrixValue input) : MatrixValue output := by
  rcases valid with ⟨_inputValid, _outputValid, ring, outputRows, outputColumns⟩
  exact fun row column ↦
    castPoly (congrArg Int.toNat ring.1) ring.2
      (matrix (Fin.cast outputColumns column) (Fin.cast outputRows row))

private noncomputable def transpose (input output : IR.MatrixType)
    (matrix : MatrixValue input) : MatrixValue output := by
  classical
  exact if valid : input.Valid ∧ output.Valid ∧ IR.sameRing input output ∧
      output.rows = input.columns ∧ output.columns = input.rows then
    exactTranspose input output valid matrix
  else 0

private noncomputable def applyPreimage (left right output : IR.MatrixType)
    (a : MatrixValue left) (preimage : PreimageValue right) : MatrixValue output := by
  classical
  exact if valid : left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output then
    exactMultiply left right output valid a preimage.exactMatrix
  else 0

noncomputable def irBackend : IR.SemanticBackend := by
  classical
  exact {
    denoteMatrix := MatrixValue
    denoteTrapdoor := TrapdoorValue
    denotePreimage := PreimageValue
    denoteTypedBlob := TypedBlobValue
    matrixZero := fun _ ↦ 0
    matrixIdentity := fun matrixType ↦
      if matrixType.Valid then exactIdentity matrixType else 0
    matrixAdd := fun matrixType left right ↦
      if matrixType.Valid then left + right else 0
    matrixSubtract := fun matrixType left right ↦
      if matrixType.Valid then left - right else 0
    matrixScale := fun matrixType scalar matrix ↦
      if matrixType.Valid then exactScale matrixType scalar matrix else 0
    matrixMultiply := multiply
    matrixNegate := fun matrixType matrix ↦
      if matrixType.Valid then -matrix else 0
    matrixTranspose := transpose
    matrixConstant := exactConstantMatrix
    matrixSlice := exactSlice
    matrixConcat := exactConcat
    gadgetCertificate := fun _ _ _ _ _ _ ↦ True
    gadgetDecompose := fun target layout structural gadget value ↦
      .ok ⟨exactGadgetDecompose target (IR.gadgetPreimageType target layout) layout value, ⟨trivial⟩⟩
    extractCoefficient := exactExtractCoefficient
    bitExtract := fun value bit ↦ (value / (2 ^ bit.toNat)) % 2 = 1
    trapdoorPublic := fun trapdoorType trapdoor ↦
      if trapdoorType.matrix.Valid then trapdoor.publicMatrix else 0
    materializePreimage := fun matrixType preimage ↦
      if matrixType.Valid then preimage.exactMatrix else 0
    applyPreimage := applyPreimage
  }

/-! A certified runtime backend is parameterized by an oracle.  The unsigned
    radix decomposition above is deliberately retained only as an experimental
    backend fixture; it does not inhabit this certificate family. -/
noncomputable def irBackendWithGadgetOracle (oracle : RuntimeGadgetOracle) : IR.SemanticBackend :=
  { irBackend with
    gadgetCertificate := RuntimeGadgetCertificate
    gadgetDecompose := oracle }

/-! The IR evaluator only establishes that the runtime oracle executed successfully.  This bridge
    is the point where that execution is interpreted as the concrete runtime certificate whose
    fields include the exact right-preimage relation and the preimage magnitude bound. -/
theorem runtimeGadgetCertificate_of_execution {oracle : RuntimeGadgetOracle}
    {structural : IR.StructuralEnv} {stage scope node : Nat}
    {baseExpr : IR.StructuralIntExpr} {small : Bool} {digitsExpr : IR.StructuralIntExpr}
    {targetType outputType : IR.MatrixType}
    {target : (irBackendWithGadgetOracle oracle).denoteMatrix targetType}
    {output : (irBackendWithGadgetOracle oracle).denotePreimage outputType}
    (execution : IR.GadgetDecomposeExecution (irBackendWithGadgetOracle oracle) structural
      stage scope node baseExpr small digitsExpr targetType outputType target output) :
    RuntimeGadgetCertificate targetType execution.layout structural
      ((irBackendWithGadgetOracle oracle).matrixConstant
        (IR.gadgetMatrixType targetType execution.layout)
        (.gadget (.literal (Int.ofNat execution.baseValue)) false) structural)
      target execution.sigma.1 := by
  exact execution.certificate

/- A plain `irBackend` execution is intentionally not accepted by this theorem: the theorem's
   backend index is `irBackendWithGadgetOracle oracle`, whose certificate family is the runtime
   relation above rather than the placeholder `True` family of the untrusted fixture backend. -/
theorem no_runtime_certificate_for_small_layout
    (targetType : IR.MatrixType) (base digits sourceRows targetRows sourceColumns targetColumns : Nat)
    (structural : IR.StructuralEnv)
    (gadget : MatrixValue {
      targetType with columns := targetType.rows * digits })
    (target : MatrixValue targetType)
    (preimage : PreimageValue {
      targetType with rows := targetType.rows * digits }) :
    ¬ RuntimeGadgetCertificate targetType
      { mode := .small, base, digits, sourceRows, targetRows, sourceColumns, targetColumns }
      structural gadget target preimage := by
  intro certificate
  cases certificate.regular


@[simp] theorem irBackend_matrixZero (matrixType : IR.MatrixType) :
    irBackend.matrixZero matrixType = (0 : MatrixValue matrixType) := rfl

@[simp] theorem irBackend_matrixIdentity {matrixType : IR.MatrixType}
    (valid : matrixType.Valid) :
    irBackend.matrixIdentity matrixType = exactIdentity matrixType := by
  simp [irBackend, valid]

@[simp] theorem irBackend_matrixAdd {matrixType : IR.MatrixType} (valid : matrixType.Valid)
    (left right : MatrixValue matrixType) :
    irBackend.matrixAdd matrixType left right = left + right := by
  simp [irBackend, valid]

@[simp] theorem irBackend_matrixSubtract {matrixType : IR.MatrixType}
    (valid : matrixType.Valid) (left right : MatrixValue matrixType) :
    irBackend.matrixSubtract matrixType left right = left - right := by
  simp [irBackend, valid]

@[simp] theorem irBackend_matrixScale {matrixType : IR.MatrixType} (valid : matrixType.Valid)
    (scalar : Int) (matrix : MatrixValue matrixType) :
    irBackend.matrixScale matrixType scalar matrix = exactScale matrixType scalar matrix := by
  simp [irBackend, valid]

@[simp] theorem irBackend_matrixMultiply {left right output : IR.MatrixType}
    (valid : left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output)
    (a : MatrixValue left) (b : MatrixValue right) :
    irBackend.matrixMultiply left right output a b =
      exactMultiply left right output valid a b := by
  simp [irBackend, multiply, valid]

@[simp] theorem irBackend_matrixNegate {matrixType : IR.MatrixType}
    (valid : matrixType.Valid) (matrix : MatrixValue matrixType) :
    irBackend.matrixNegate matrixType matrix = -matrix := by
  simp [irBackend, valid]

@[simp] theorem irBackend_matrixTranspose {input output : IR.MatrixType}
    (valid : input.Valid ∧ output.Valid ∧ IR.sameRing input output ∧
      output.rows = input.columns ∧ output.columns = input.rows)
    (matrix : MatrixValue input) :
    irBackend.matrixTranspose input output matrix =
      exactTranspose input output valid matrix := by
  simp [irBackend, transpose, valid]

@[simp] theorem irBackend_trapdoorPublic {trapdoorType : IR.TrapdoorType}
    (valid : trapdoorType.matrix.Valid) (trapdoor : TrapdoorValue trapdoorType) :
    irBackend.trapdoorPublic trapdoorType trapdoor = trapdoor.publicMatrix := by
  simp [irBackend, valid]

@[simp] theorem irBackend_materializePreimage {matrixType : IR.MatrixType}
    (valid : matrixType.Valid) (preimage : PreimageValue matrixType) :
    irBackend.materializePreimage matrixType preimage = preimage.exactMatrix := by
  simp [irBackend, valid]

@[simp] theorem irBackend_applyPreimage {left right output : IR.MatrixType}
    (valid : left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output)
    (a : MatrixValue left) (preimage : PreimageValue right) :
    irBackend.applyPreimage left right output a preimage =
      exactMultiply left right output valid a preimage.exactMatrix := by
  simp [irBackend, applyPreimage, valid]

@[simp] theorem irBackend_invalid_matrixAdd {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) (left right : MatrixValue matrixType) :
    irBackend.matrixAdd matrixType left right = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_matrixIdentity {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) :
    irBackend.matrixIdentity matrixType = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_matrixSubtract {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) (left right : MatrixValue matrixType) :
    irBackend.matrixSubtract matrixType left right = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_matrixScale {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) (scalar : Int) (matrix : MatrixValue matrixType) :
    irBackend.matrixScale matrixType scalar matrix = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_matrixNegate {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) (matrix : MatrixValue matrixType) :
    irBackend.matrixNegate matrixType matrix = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_matrixMultiply {left right output : IR.MatrixType}
    (invalid : ¬(left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output))
    (a : MatrixValue left) (b : MatrixValue right) :
    irBackend.matrixMultiply left right output a b = (0 : MatrixValue output) := by
  simp [irBackend, multiply, invalid]

@[simp] theorem irBackend_invalid_matrixTranspose {input output : IR.MatrixType}
    (invalid : ¬(input.Valid ∧ output.Valid ∧ IR.sameRing input output ∧
      output.rows = input.columns ∧ output.columns = input.rows))
    (matrix : MatrixValue input) :
    irBackend.matrixTranspose input output matrix = (0 : MatrixValue output) := by
  simp [irBackend, transpose, invalid]

@[simp] theorem irBackend_invalid_trapdoorPublic {trapdoorType : IR.TrapdoorType}
    (invalid : ¬trapdoorType.matrix.Valid) (trapdoor : TrapdoorValue trapdoorType) :
    irBackend.trapdoorPublic trapdoorType trapdoor =
      (0 : MatrixValue trapdoorType.matrix) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_materializePreimage {matrixType : IR.MatrixType}
    (invalid : ¬matrixType.Valid) (preimage : PreimageValue matrixType) :
    irBackend.materializePreimage matrixType preimage = (0 : MatrixValue matrixType) := by
  simp [irBackend, invalid]

@[simp] theorem irBackend_invalid_applyPreimage {left right output : IR.MatrixType}
    (invalid : ¬(left.Valid ∧ right.Valid ∧ output.Valid ∧
      IR.matrixProductType left right output))
    (a : MatrixValue left) (preimage : PreimageValue right) :
    irBackend.applyPreimage left right output a preimage = (0 : MatrixValue output) := by
  simp [irBackend, applyPreimage, invalid]

section Regression

private def matrix22 : IR.MatrixType :=
  { modulus := 257, ringDimension := 8, rows := 2, columns := 2 }

private def matrix22Valid : matrix22.Valid := by
  norm_num [matrix22, IR.MatrixType.Valid]

private def scalar11 : IR.MatrixType :=
  { modulus := 257, ringDimension := 8, rows := 1, columns := 1 }

example (left right : MatrixValue matrix22) :
    irBackend.matrixAdd matrix22 left right = left + right := by
  exact irBackend_matrixAdd matrix22Valid left right

example : irBackend.matrixIdentity matrix22 = exactIdentity matrix22 := by
  exact irBackend_matrixIdentity matrix22Valid

example (left right : MatrixValue matrix22) :
    irBackend.matrixSubtract matrix22 left right = left - right := by
  exact irBackend_matrixSubtract matrix22Valid left right

example (scalar : Int) (matrix : MatrixValue matrix22) :
    irBackend.matrixScale matrix22 scalar matrix = exactScale matrix22 scalar matrix := by
  exact irBackend_matrixScale matrix22Valid scalar matrix

example (left right : MatrixValue matrix22) :
    irBackend.matrixMultiply matrix22 matrix22 matrix22 left right =
      exactMultiply matrix22 matrix22 matrix22 (by
        norm_num [matrix22, IR.MatrixType.Valid, IR.matrixProductType, IR.sameRing])
        left right := by
  apply irBackend_matrixMultiply

example (scalar : MatrixValue scalar11) (matrix : MatrixValue matrix22) :
    irBackend.matrixMultiply scalar11 matrix22 matrix22 scalar matrix =
      exactMultiply scalar11 matrix22 matrix22 (by
        norm_num [scalar11, matrix22, IR.MatrixType.Valid, IR.matrixProductType, IR.sameRing])
        scalar matrix := by
  apply irBackend_matrixMultiply

example (matrix : MatrixValue matrix22) :
    irBackend.matrixNegate matrix22 matrix = -matrix := by
  exact irBackend_matrixNegate matrix22Valid matrix

example (matrix : MatrixValue matrix22) :
    irBackend.matrixTranspose matrix22 matrix22 matrix =
      exactTranspose matrix22 matrix22 (by
        norm_num [matrix22, IR.MatrixType.Valid, IR.sameRing]) matrix := by
  apply irBackend_matrixTranspose

example (preimage : PreimageValue matrix22) :
    irBackend.materializePreimage matrix22 preimage = preimage.exactMatrix := by
  exact irBackend_materializePreimage matrix22Valid preimage

example (left : MatrixValue matrix22) (preimage : PreimageValue matrix22) :
    irBackend.applyPreimage matrix22 matrix22 matrix22 left preimage =
      exactMultiply matrix22 matrix22 matrix22 (by
        norm_num [matrix22, IR.MatrixType.Valid, IR.matrixProductType, IR.sameRing])
        left preimage.exactMatrix := by
  apply irBackend_applyPreimage

example (left : MatrixValue scalar11) (preimage : PreimageValue matrix22) :
    irBackend.applyPreimage scalar11 matrix22 matrix22 left preimage =
      exactMultiply scalar11 matrix22 matrix22 (by
        norm_num [scalar11, matrix22, IR.MatrixType.Valid, IR.matrixProductType, IR.sameRing])
        left preimage.exactMatrix := by
  apply irBackend_applyPreimage

example (left : MatrixValue matrix22) (preimage : PreimageValue scalar11) :
    irBackend.applyPreimage matrix22 scalar11 matrix22 left preimage =
      exactMultiply matrix22 scalar11 matrix22 (by
        norm_num [scalar11, matrix22, IR.MatrixType.Valid, IR.matrixProductType, IR.sameRing])
        left preimage.exactMatrix := by
  apply irBackend_applyPreimage

example {trapdoorType : IR.TrapdoorType} (valid : trapdoorType.matrix.Valid)
    (trapdoor : TrapdoorValue trapdoorType) :
    irBackend.trapdoorPublic trapdoorType trapdoor = trapdoor.publicMatrix := by
  exact irBackend_trapdoorPublic valid trapdoor

private def malformed : IR.MatrixType :=
  { modulus := -1, ringDimension := 8, rows := 2, columns := 2 }

example (left right : MatrixValue malformed) :
    irBackend.matrixAdd malformed left right = (0 : MatrixValue malformed) := by
  apply irBackend_invalid_matrixAdd
  norm_num [malformed, IR.MatrixType.Valid]

end Regression

end Mxx.Runtime
