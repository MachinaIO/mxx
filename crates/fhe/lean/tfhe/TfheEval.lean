import TfheKeygen

/-!
# The closed form of the bootstrapped NAND gate

`evalOf` is the ciphertext the `Eval` graph determines: the NAND input `(-(a₀ + a₁), q/8 - (b₀ +
b₁))`, its rounding to `2 N`, the blind rotation `brSeq` of the rotated lookup table by the
bootstrapping keys, and the sample extraction and key switch of the result (the loops of
`TfheEncrypt`).
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

/-! ## Rows -/

/-- Rows `top` then `bottom`. -/
def concatRowsOf {q n lr rr columns : Nat} (top : ExactMatrix q n lr columns)
    (bottom : ExactMatrix q n rr columns) : ExactMatrix q n (lr + rr) columns :=
  fun row column ↦ if h : row.val < lr then top ⟨row.val, h⟩ column
    else bottom ⟨row.val - lr, by omega⟩ column

theorem concatRows_iff {q n lr rr columns : Nat} (top : ExactMatrix q n lr columns)
    (bottom : ExactMatrix q n rr columns) (out : ExactMatrix q n (lr + rr) columns) :
    concatRows top bottom out ↔ out = concatRowsOf top bottom := by
  constructor
  · intro h
    funext row column
    exact h row column
  · rintro rfl row column
    rfl

/-- Row `r` of a two-row column. -/
def rowOf {q n : Nat} (x : ExactMatrix q n 2 1) (r : Fin 2) : ExactMatrix q n 1 1 :=
  fun _ c ↦ x r c

theorem slice_row0_iff {q n : Nat} (x : ExactMatrix q n 2 1) (out : ExactMatrix q n 1 1) :
    sliceMatrix x 0 1 0 1 out ↔ out = rowOf x 0 := by
  constructor
  · rintro ⟨-, -, -, -, -, -, -, -, h⟩
    funext r c
    rw [h r c (by have := r.isLt; cast_omega) (by have := c.isLt; cast_omega)]
    simp only [rowOf]
    congr 1 <;> exact Fin.ext (by have := r.isLt; have := c.isLt; cast_omega)
  · rintro rfl
    refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
      by norm_num, fun r c _ _ ↦ ?_⟩
    simp only [rowOf]
    congr 1 <;> exact Fin.ext (by have := r.isLt; have := c.isLt; cast_omega)

theorem slice_row1_iff {q n : Nat} (x : ExactMatrix q n 2 1) (out : ExactMatrix q n 1 1) :
    sliceMatrix x 1 2 0 1 out ↔ out = rowOf x 1 := by
  constructor
  · rintro ⟨-, -, -, -, -, -, -, -, h⟩
    funext r c
    rw [h r c (by have := r.isLt; cast_omega) (by have := c.isLt; cast_omega)]
    simp only [rowOf]
    congr 1 <;> exact Fin.ext (by have := r.isLt; have := c.isLt; cast_omega)
  · rintro rfl
    refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
      by norm_num, fun r c _ _ ↦ ?_⟩
    simp only [rowOf]
    congr 1 <;> exact Fin.ext (by have := r.isLt; have := c.isLt; cast_omega)

/-! ## The blind rotation -/

/-- The digits of a two-row column under the ring gadget. -/
noncomputable def decompose (diff : ExactMatrix Q N 2 1) : ExactMatrix Q N 12 1 :=
  castMatrixRows (by rw [layout_digitCount]) (regularDecomposeMatrix Backend.layout0 diff)

theorem gadgetDecomposeRuns_eq {diff : ExactMatrix Q N 2 1} {D : ExactMatrix Q N 12 1}
    (hD : gadgetDecomposeRuns Backend.backend 64 6 diff D) : D = decompose diff := by
  obtain ⟨layout, hl, _, _, _, hdeq, _⟩ := hD
  rw [backend_layout] at hl
  cases hl
  rw [hdeq]
  rfl

theorem gadgetDecomposeRuns_holds (diff : ExactMatrix Q N 2 1) :
    gadgetDecomposeRuns Backend.backend 64 6 diff (decompose diff) :=
  ⟨Backend.layout0, backend_layout, by decide, by rw [layout_digitCount]; rfl,
    by rw [layout_digitCount], rfl,
    regularGadgetMatrix_residual_exact _ _ (by decide) (by decide) layout_exact⟩

/-- Rounding from `q` to `2 N`. -/
def roundRing (x : Int) : Int := (x * 2048 + 2147483648) / 4294967296 % 2048

/-- One external product: `acc + [A_i; B_i] G⁻¹(X^{r_i} acc - acc)`. -/
noncomputable def brStep (A B : Fin 720 → ExactMatrix Q N 1 12) (rot : Fin 720 → Int)
    (i : Fin 720) (acc : ExactMatrix Q N 2 1) : ExactMatrix Q N 2 1 :=
  matrixAdd acc (matrixMul (concatRowsOf (A i) (B i))
    (decompose (matrixSub (multiplyMonomial acc (rot i)) acc)))

/-- The accumulator after each external product. -/
noncomputable def brSeq (A B : Fin 720 → ExactMatrix Q N 1 12) (rot : Fin 720 → Int)
    (init : ExactMatrix Q N 2 1) : Nat → ExactMatrix Q N 2 1
  | 0 => init
  | n + 1 => if h : n < 720 then brStep A B rot ⟨n, h⟩ (brSeq A B rot init n)
      else brSeq A B rot init n

theorem br_round_iff (m : Int) (i : Fin 720) (o : Int) :
    Eval.parallel_scope_tfhe_blind_rotation_6 Backend.backend Eval_params i m o ↔
      o = roundRing m := by
  constructor
  · rintro ⟨-, -, h⟩; exact h
  · intro h; exact ⟨by norm_num, by norm_num, h⟩

theorem br_step_iff (A B : Fin 720 → ExactMatrix Q N 1 12) (rot : Fin 720 → Int) (i : Fin 720)
    (acc out : ExactMatrix Q N 2 1) :
    Eval.sequential_scope_tfhe_blind_rotation_7 Backend.backend Eval_params i
      (acc, (A, (B, (rot, ())))) out ↔ out = brStep A B rot i acc := by
  have hi := i.isLt
  have hidx : ((i.val : Nat) : Int) = Int.ofNat i.val := rfl
  constructor
  · rintro ⟨w3, w5, w6, w8, w11, -, -, h3, -, -, h5, h6, -, -, h8, h11, hout⟩
    have e3 : w3 = A i := familyGet_eq' hidx h3
    have e5 : w5 = B i := familyGet_eq' hidx h5
    have e8 : w8 = rot i := familyGet_eq' hidx h8
    subst e3 e5 e8
    have e6 := (concatRows_iff _ _ _).mp h6
    subst e6
    have e11 := gadgetDecomposeRuns_eq h11
    subst e11
    rw [hout]
    rfl
  · rintro rfl
    exact ⟨A i, B i, concatRowsOf (A i) (B i), rot i, _, by cast_omega, by cast_omega,
      ⟨i, hidx, rfl⟩, by cast_omega, by cast_omega, ⟨i, hidx, rfl⟩, (concatRows_iff _ _ _).mpr rfl,
      by cast_omega, by cast_omega, ⟨i, hidx, rfl⟩, gadgetDecomposeRuns_holds _, rfl⟩

/-- A run of every step from a sequence of states. -/
theorem iterRuns_of_seq {State : Type} {body : Nat → State → State → Prop} (seq : Nat → State) :
    ∀ count, (∀ i < count, body i (seq i) (seq (i + 1))) →
      MxxIR.IterRuns body count (seq 0) (seq count)
  | 0, _ => .zero _
  | count + 1, h => .step (iterRuns_of_seq seq count fun i hi ↦ h i (by omega)) (h count (by omega))

/-- The rows the blind rotation returns. -/
noncomputable def brOut (acc0 acc1 : ExactMatrix Q N 1 1) (masks : Fin 720 → Int)
    (A B : Fin 720 → ExactMatrix Q N 1 12) : ExactMatrix Q N 1 1 × ExactMatrix Q N 1 1 × Unit :=
  let final := brSeq A B (fun i ↦ roundRing (masks i)) (concatRowsOf acc0 acc1) 720
  (rowOf final 0, rowOf final 1, ())

theorem blind_rotation_iff (acc0 acc1 : ExactMatrix Q N 1 1) (masks : Fin 720 → Int)
    (A B : Fin 720 → ExactMatrix Q N 1 12) (out : ExactMatrix Q N 1 1 × ExactMatrix Q N 1 1 × Unit) :
    Eval.scope_tfhe_blind_rotation Backend.backend Eval_params (acc0, (acc1, (masks, (A, (B, ())))))
      out ↔ out = brOut acc0 acc1 masks A B := by
  set rot : Fin 720 → Int := fun i ↦ roundRing (masks i)
  set init := concatRowsOf acc0 acc1
  constructor
  · rintro ⟨w, h2, -, h6, -, h7, -, -, -, -, -, -, h8, -, -, -, -, -, -, h9, hout⟩
    obtain ⟨w2, w6, w7, w8, w9⟩ := w
    have e2 : w2 = init := (concatRows_iff _ _ _).mp h2
    have e6 : w6 = rot := funext fun i ↦ (br_round_iff _ i _).mp (h6 i)
    subst e2 e6
    have hfinal := MxxIR.IterRuns.invariant
      (Invariant := fun i (acc : ExactMatrix Q N 2 1) ↦ acc = brSeq A B rot init i) rfl ?_ h7
    · have e8 := (slice_row0_iff _ _).mp h8
      have e9 := (slice_row1_iff _ _).mp h9
      dsimp only at hfinal e8 e9
      rw [hout, e8, e9, hfinal]
      rfl
    · intro i current next hcur hstep
      subst hcur
      have hi : i < 720 := by
        obtain ⟨_, _, _, _, _, -, hlt, -⟩ := hstep
        cast_omega
      rw [(br_step_iff A B rot ⟨i, hi⟩ _ _).mp hstep]
      simp only [brSeq, dif_pos hi]
  · rintro rfl
    refine ⟨⟨init, rot, brSeq A B rot init 720, rowOf (brSeq A B rot init 720) 0,
      rowOf (brSeq A B rot init 720) 1⟩, (concatRows_iff _ _ _).mpr rfl, rfl,
      fun i ↦ (br_round_iff _ i _).mpr rfl, by norm_num, ?_, by norm_num, by norm_num, by norm_num,
      by norm_num, by norm_num, by norm_num, (slice_row0_iff _ _).mpr rfl, by norm_num, by norm_num,
      by norm_num, by norm_num, by norm_num, by norm_num, (slice_row1_iff _ _).mpr rfl, rfl⟩
    refine iterRuns_of_seq (brSeq A B rot init) 720 fun i hi ↦ ?_
    refine (br_step_iff A B rot ⟨i, hi⟩ _ _).mpr ?_
    simp only [brSeq, dif_pos hi]

/-! ## The loops of `Eval` -/

theorem eval_extract_iff (c : Fin 1024 → Int) (i : Fin 1024) (o : Int) :
    Eval.parallel_generatedRoot_38 Backend.backend Eval_params i (c, (4294967296, ())) o ↔
      o = extractAt c i := by
  constructor
  · rintro ⟨w6, _, _, _, hget, _, _, _, hout⟩
    rw [hout, familyGet_eq' (extractIndex_val i) hget]
    simp only [extractAt, rescale, decide_eq_true_eq]
    rfl
  · intro ho
    subst ho
    have hi := i.isLt
    refine ⟨c (extractIndex i), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · norm_num
    · show (0 : Int) ≤ (1024 - (i.val : Int)) % 1024
      omega
    · show (1024 - (i.val : Int)) % 1024 < 1024
      omega
    · exact ⟨extractIndex i, extractIndex_val i, rfl⟩
    · norm_num
    · norm_num
    · norm_num
    · simp only [extractAt, rescale, decide_eq_true_eq]
      rfl

theorem digitIndex_val' (i : Fin 8192) :
    ((⟨i.val / 8, by omega⟩ : Fin 1024).val : Int) = (i.val : Int) / 8 := by
  simp only
  omega

theorem eval_digit_iff (a : Fin 1024 → Int) (i : Fin 8192) (o : Int) :
    Eval.parallel_generatedRoot_47 Backend.backend Eval_params i
      (a, (1, (4, (16, (64, (256, (1024, (4096, (16384, ()))))))))) o ↔ o = digitAt a i := by
  constructor
  · rintro ⟨w4, w37, _, _, _, hget, _, _, _, _, _, _, hsel, _, _, hout⟩
    rw [hout, familyGet_eq' (digitIndex_val' i) hget, select_powers hsel]
    rfl
  · intro ho
    subst ho
    have hi := i.isLt
    refine ⟨a ⟨i.val / 8, by omega⟩, (4 : Int) ^ (((i.val : Int) % 8).toNat), ?_⟩
    refine ⟨by norm_num, ?_, ?_, ⟨_, digitIndex_val' i, rfl⟩, by norm_num, by norm_num, by norm_num,
      ?_, ?_, rfl, ?_, by positivity, by norm_num, rfl⟩
    · show (0 : Int) ≤ (i.val : Int) / 8; omega
    · show (i.val : Int) / 8 < 1024; omega
    · show (0 : Int) ≤ (i.val : Int) % 8; omega
    · show (i.val : Int) % 8 < 8; omega
    · have h := select_powers_holds (k := (i.val : Int) % 8) ⟨by omega, by omega⟩
      dsimp only
      exact h

theorem eval_neg_iff (x : Int) (i : Fin 720) (o : Int) :
    Eval.parallel_generatedRoot_50 Backend.backend Eval_params i (x, (4294967296, ())) o ↔
      o = (0 - x) % 4294967296 := by
  constructor
  · rintro ⟨_, hout⟩
    exact hout
  · intro ho
    exact ⟨by norm_num, ho⟩

theorem eval_sum_iff (x y : Int) (i : Fin 720) (o : Int) :
    Eval.parallel_generatedRoot_31 Backend.backend Eval_params i (x, (y, (4294967296, ()))) o ↔
      o = (x + y) % 4294967296 := by
  constructor
  · rintro ⟨_, h⟩; exact h
  · intro h; exact ⟨by norm_num, h⟩

theorem eval_neg0_iff (x : Int) (i : Fin 720) (o : Int) :
    Eval.parallel_generatedRoot_32 Backend.backend Eval_params i (x, (4294967296, ())) o ↔
      o = (0 - x) % 4294967296 := by
  constructor
  · rintro ⟨_, h⟩; exact h
  · intro h; exact ⟨by norm_num, h⟩

/-! ## The NAND gate -/

/-- The inputs of `Eval`: the key-switching masks, the two bodies and masks, the bootstrapping
keys and the key-switching bodies. -/
abbrev EvalInputs := (Fin 5898240 → Int) × Int × Int × (Fin 720 → Int) × (Fin 720 → Int) ×
  (Fin 720 → ExactMatrix Q N 1 12) × (Fin 720 → ExactMatrix Q N 1 12) × (Fin 8192 → Int) × Unit

/-- The lookup table `⌊Q/8⌋ (1 + X + ⋯ + X^{N-1})`. -/
noncomputable def lut : ExactMatrix Q N 1 1 :=
  (packedPolynomial 30 1024 Eval.generatedRoot.table_2 : ExactMatrix Q N 1 1)

/-- The body of the NAND input `q/8 - (b₀ + b₁)`. -/
def nandBody (b0 b1 : Int) : Int :=
  ((0 - (b0 + b1) % 4294967296) % 4294967296 + 536870912) % 4294967296

/-- The mask of the NAND input `-(a₀ + a₁)`. -/
def nandMask (a0 a1 : Fin 720 → Int) (j : Fin 720) : Int :=
  (0 - (a0 j + a1 j) % 4294967296) % 4294967296

/-- The exponent of the initial rotation of the lookup table. -/
def initialExp (b : Int) : Int := (0 - roundRing b) % 2048

/-- The blind rotation of the NAND gate. -/
noncomputable def nandBr (inputs : EvalInputs) : ExactMatrix Q N 1 1 × ExactMatrix Q N 1 1 × Unit :=
  brOut 0 (multiplyMonomial lut (initialExp (nandBody inputs.2.1 inputs.2.2.1)))
    (nandMask inputs.2.2.2.1 inputs.2.2.2.2.1) inputs.2.2.2.2.2.1 inputs.2.2.2.2.2.2.1

/-- The key-switching digits of the bootstrapped ciphertext. -/
noncomputable def nandDigits (inputs : EvalInputs) : Fin 8192 → Int :=
  digitAt (extractAt (coeffVals (nandBr inputs).1))

/-- The ciphertext `Eval` returns. -/
noncomputable def evalOf (inputs : EvalInputs) : (Fin 720 → Int) × Int × Unit :=
  (ksMask inputs.1 (nandDigits inputs),
    ksBody inputs.2.2.2.2.2.2.2.1 (rescale (coeffVals (nandBr inputs).2.1 0)) (nandDigits inputs), ())

theorem evalOut_eq (inputs : EvalInputs) :
    (determined fun outputs ↦ ∃ witness,
      Eval.generatedRoot.body Backend.backend Eval_params inputs outputs witness) =
      evalOf inputs := by
  set br := nandBr inputs
  set d := nandDigits inputs
  refine determined_eq ⟨⟨fun j ↦ (inputs.2.2.2.1 j + inputs.2.2.2.2.1 j) % 4294967296,
    nandMask inputs.2.2.2.1 inputs.2.2.2.2.1, br.1, br.2.1, coeffVals br.1,
    extractAt (coeffVals br.1), d, ksMask inputs.1 d, coeffVals br.2.1, coeffVals br.2.1 0,
    intMatrixVectorProduct (outer := 1) true inputs.2.2.2.2.2.2.2.1 d 0⟩, ?_⟩ ?_
  · refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, rfl,
      fun i ↦ (eval_sum_iff _ _ i _).mpr rfl, rfl, fun i ↦ (eval_neg0_iff _ i _).mpr rfl,
      (blind_rotation_iff _ _ _ _ _ _).mpr rfl, polynomialValues_coeffVals _, rfl,
      fun i ↦ (eval_extract_iff _ i _).mpr rfl, rfl, fun i ↦ (eval_digit_iff _ i _).mpr rfl, rfl,
      fun i ↦ (eval_neg_iff _ i _).mpr rfl, polynomialValues_coeffVals _, familyGet_holds _ 0,
      by norm_num, by norm_num, by norm_num, familyGet_holds _ 0, by norm_num, rfl⟩
  · rintro other ⟨w, -, -, -, -, -, -, -, h31, -, h32, hbr, h36, -, h38, -, h47, -, h50, h51, h52,
      -, -, -, h64, -, hout⟩
    obtain ⟨w31, w32, w350, w351, w36, w38, w47, w50, w51, w52, w64⟩ := w
    have e31 : w31 = fun j ↦ (inputs.2.2.2.1 j + inputs.2.2.2.2.1 j) % 4294967296 :=
      funext fun i ↦ (eval_sum_iff _ _ i _).mp (h31 i)
    subst e31
    have e32 : w32 = nandMask inputs.2.2.2.1 inputs.2.2.2.2.1 :=
      funext fun i ↦ (eval_neg0_iff _ i _).mp (h32 i)
    subst e32
    have ebr := (blind_rotation_iff _ _ _ _ _ _).mp hbr
    have e350 : w350 = br.1 := congrArg Prod.fst ebr
    have e351 : w351 = br.2.1 := congrArg (fun x ↦ x.2.1) ebr
    subst e350 e351
    have e36 : w36 = coeffVals br.1 := polynomialValues_eq h36
    subst e36
    have e38 : w38 = extractAt (coeffVals br.1) := funext fun i ↦ (eval_extract_iff _ i _).mp (h38 i)
    subst e38
    have e47 : w47 = d := funext fun i ↦ (eval_digit_iff _ i _).mp (h47 i)
    subst e47
    have e50 : w50 = ksMask inputs.1 d := funext fun i ↦ (eval_neg_iff _ i _).mp (h50 i)
    subst e50
    have e51 : w51 = coeffVals br.2.1 := polynomialValues_eq h51
    subst e51
    have e52 : w52 = coeffVals br.2.1 0 := familyGet_eq (i := (0 : Fin 1024)) h52
    have e64 : w64 = intMatrixVectorProduct (outer := 1) true inputs.2.2.2.2.2.2.2.1 d 0 :=
      familyGet_eq (i := (0 : Fin 1)) h64
    subst e52 e64
    rw [hout]
    rfl

end MxxFheTfhe
