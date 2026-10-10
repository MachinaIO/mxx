import PePreprocess
import ExponentErrorForm

/-!
# Encryption

The Dynamic packages of the ciphertext, under the secret `s`, published compactly over `R_{PQ}`:
the encoding of the initial state `X⁰`, the helper of `K`, and the digit companions of the
attribute ciphertext and of the switching ciphertext `C_K`. Each is a pair of single entries
`c̃_j = s̄ Ã_{0j} + Ã_{1j} - P x_j + e_j` under a compact key `Ã`; its expansion
(`expand_rel`) is the full encoding of `x_j g` under the expanded key. Every error is named by its
tape site, so each output is an explicit function of the secret, the hash keys, the attribute
ciphertext, and the tape. The switching ciphertext `C_K` is computed at `Q` from the expanded
helper.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the encryption errors. -/
abbrev encLaw : SamplerLaw := .gaussian (319 / 100) {{CF}}

/-- The law of the helper errors. -/
abbrev helperLaw : SamplerLaw := .gaussian (319 / 100) {{CE}}

/-- An error entry of a compact encoding, sampled at `site`. -/
noncomputable abbrev errAt (tape : SampleTape) (site : List Nat) (law : SamplerLaw) : ErrorMatrix N 1 1 :=
  tapeWitness tape site law

abbrev EncryptOutputs := (Fin {{L}} → ExactMatrix PQ N 1 1) × (Fin {{L}} → ExactMatrix PQ N 1 1) ×
  (Fin 1 → ExactMatrix PQ N 1 1) × (Fin 1 → ExactMatrix PQ N 1 1) × ExactMatrix PQ N 1 1 ×
  ExactMatrix PQ N 1 1 × (Fin {{L}} → ExactMatrix PQ N 1 1) × (Fin {{L}} → ExactMatrix PQ N 1 1) × Unit

/-- The compact target `P x_j` of a payload `x`. -/
noncomputable abbrev ctarget (x : Fin 2 → ErrorPoly N) : Fin 2 → ExactMatrix PQ N 1 1 :=
  fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) * x j)

theorem message_one_zero : (message (n := N) 1 0 : ErrorPoly N) = 1 := by
  simp [message]

/-- A ternary polynomial is below half of `Q`. -/
theorem ternary_small {sm : ErrorPoly N} (hsm : polyNorm sm ≤ 1) : ∀ i, 2 * (sm.coeff i).natAbs < Q := by
  intro i
  have h := (coeff_natAbs_le_polyNorm sm i).trans hsm
  have hq : 4 < Q := by norm_num
  omega

/-- The centered rebase to `PQ` of a polynomial below half of `Q` is its reduction. -/
theorem rebase_small {x : ErrorPoly N} (hx : ∀ i, 2 * (x.coeff i).natAbs < Q) {y : ExactMatrix PQ N 1 1}
    (h : centeredRebaseRuns (fun _ _ ↦ reducePoly Q N x : ExactMatrix Q N 1 1) y) :
    y = fun _ _ ↦ reducePoly PQ N x := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [centeredIntLift_reducePoly hQ hN x hx]

/-- The `1 × 1` identity is the constant one. -/
theorem one_eq_const {q : Nat} : (1 : ExactMatrix q N 1 1) = fun _ _ ↦ reducePoly q N 1 := by
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

theorem one_small : ∀ i, 2 * ((1 : ErrorPoly N).coeff i).natAbs < Q := by
  intro i
  have h1 : (1 : ErrorPoly N) = algebraMap Int (ErrorPoly N) 1 := (map_one _).symm
  rw [h1, MxxPe.coeff_algebraMap' hN]
  have hq : 4 < Q := by norm_num
  split <;> simp <;> omega

set_option maxHeartbeats 16000000 in
/-- The initial state is a compact encoding of `P X⁰` under the compact input key. -/
theorem encrypt_input {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hsm : polyNorm sm ≤ 1) :
    RelE sm (inputKeyC model key) (ctarget fun j ↦ ((1 : Nat) : Int) * 1 * unitPair sm j)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ![errAt tape (path ++ [28]) encLaw, errAt tape (path ++ [39]) encLaw] := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x9, x10, x12, x13, x23, x24, x28, x30, x32, x35, x39, -, -, -,
    -, hout⟩ := h
  subst hsecret
  have hw9 := rebase_small (ternary_small hsm) x9
  have hw24 := rebase_small (ternary_small hsm) x24
  rw [one_eq_const] at x12 x35
  have hw12 := rebase_small one_small x12
  have hw35 := rebase_small one_small x35
  have h22 : (matrixMulScalarLeft (matrixPolynomial [(1 : Int)])
      (multiplyMonomial (liftInteger (1 : Int)) (0 : Int)) : ExactMatrix Q N 1 1) =
      fun _ _ ↦ reducePoly Q N 1 := by
    rw [multiplyMonomial_liftInteger, message_one_zero]
    funext r c
    simp [matrixMulScalarLeft, matrixPolynomial_single]
  rw [h22] at x23
  have hw23 := rebase_small one_small x23
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x28
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x39
  subst hout
  have hkey : ∀ k j, inputKeyC model key k j = ![![w.w_10_0, w.w_30_0], ![w.w_13_0, w.w_32_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x10).symm
    · exact (hashSample_eq x30).symm
    · exact (hashSample_eq x13).symm
    · exact (hashSample_eq x32).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c' = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      matrixMulScalarLeft_eq, hw9, hw12, hw23, hw24, hw35, he0, he1, errAt, unitPair, reduceMatrix,
      ctarget, matrixPolynomial_single]
    simp [map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
/-- The helper of `K`: a compact encoding of `P σ_K(s)` against the compact helper key. -/
theorem encrypt_helper {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hsm : polyNorm sm ≤ 1) :
    RelE sm (helperKeyC model key)
      (ctarget fun j ↦ (((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution K hK sm) j)
      ![outputs.2.2.1 0, outputs.2.2.2.1 0]
      ![errAt tape (path ++ [5, 0, 14]) helperLaw, errAt tape (path ++ [5, 0, 23]) helperLaw] := by
  obtain ⟨w, x3, x4, -, h5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := h
  subst hsecret
  have hw3 := rebase_small (ternary_small hsm) x3
  rw [one_eq_const] at x4
  have hw4 := rebase_small one_small x4
  obtain ⟨w2, w5, w10, w11, w14, w16, w18, w23, -, -, x2, -, -, x5, -, -, x10, x11, x14, -, -, x16, -, -,
    x18, x23, xout⟩ := h5 0
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x14
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x23
  have hs : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [{{K}}] = (K : Int) := by decide
  rw [hs] at x2 x5 x10 x16 x18
  obtain ⟨hodd, hw10⟩ := ringAutomorphismRuns_eq hQ hN x10
  have hσ : substitution (n := N) (R := ZMod Q) (Int.toNat (K : Int)) hodd (reducePoly Q N sm) =
      reducePoly Q N (substitution K hK sm) := by
    rw [reducePoly_substitution]
    rfl
  have hσsm : polyNorm (substitution K hK sm) ≤ 1 :=
    (polyNorm_substitution_le hN hK hKcoprime sm).trans hsm
  have h10 : w10 = fun _ _ ↦ reducePoly Q N (substitution K hK sm) := by
    rw [hw10]
    funext r c
    exact hσ
  rw [h10] at x11
  have hw11 := rebase_small (ternary_small hσsm) x11
  have hkey : ∀ k j, helperKeyC model key k j = ![![w2, w16], ![w5, w18]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x2).symm
    · exact (hashSample_eq x16).symm
    · exact (hashSample_eq x5).symm
    · exact (hashSample_eq x18).symm
  subst hout
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c' = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      matrixMulScalarLeft_eq, hw3, hw4, hw11, he0, he1, errAt, unitPair, reduceMatrix, ctarget,
      matrixPolynomial_single, List.append_assoc, List.cons_append, List.nil_append]
    simp only [map_mul, map_one, map_ofNat, map_natCast, map_intCast, Nat.cast_one, Int.cast_one,
      Nat.cast_ofNat, Int.cast_natCast, one_mul, mul_one, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one, one_smul, smul_eq_mul, Fin.val_zero, Fin.isValue]
    try ring

/-- Digits of a decomposition rebase to themselves at the same modulus. -/
theorem rebase_self {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    {x : ExactMatrix Q N 1 1} {u : ExactMatrix Q N {{D}} 1}
    (hu : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} x u) : compactCenteredRebaseRuns {{HALFBASE}} u u := by
  have hb := (decompose_entry hQ hN Backend.layout0 layoutQ rfl hg hu).2
  exact ⟨Or.inl dvd_rfl, hb, centeredRebaseRuns_self hQ hN u, hb⟩

/-- The column digits of the joined rows run. -/
theorem colDigits_runs {g : ExactMatrix Q N 1 {{D}}} (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    (x y : ExactPoly Q N) :
    ∃ D' : ExactMatrix Q N ({{D}} + {{D}}) 1,
      gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (pairColumn x y) D' ∧
        compactCenteredRebaseRuns {{HALFBASE}} D'
          (columnDigits Q ({{D}} + {{D}}) Backend.backend {{BASE}} {{D}} {{HALFBASE}} x y) :=
  columnDigits_runs hQ hQ hN hg Backend.layout0 layoutQ rfl rfl (by rw [show Backend.layout0.base = {{BASE}} from rfl]; norm_num)
    (by rw [show Backend.layout0.base = {{BASE}} from rfl]; norm_num) x y

theorem lane_indices {jj a : Nat} (hjj : jj < {{M}}) (ha : a < {{M}}) :
    Int.fdiv (Int.ofNat (jj + {{M}} * a)) {{L}} = 0 ∧
      Int.fmod (Int.ofNat (jj + {{M}} * a)) {{M}} = ((jj : Nat) : Int) ∧
      Int.fmod (Int.fdiv (Int.ofNat (jj + {{M}} * a)) {{M}}) {{M}} = ((a : Nat) : Int) ∧
      Int.fmod ((a : Nat) : Int) {{D}} = (((a : Nat) % {{D}} : Nat) : Int) ∧
      Int.fdiv ((a : Nat) : Int) {{D}} = (((a : Nat) / {{D}} : Nat) : Int) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  · rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  · rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  · rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  · rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega

/-- The centered rebase to `PQ` of a `1 × 1` entry is the reduction of its centered lift. -/
theorem rebase_lift {x : ExactMatrix Q N 1 1} {y : ExactMatrix PQ N 1 1} (h : centeredRebaseRuns x y) :
    y = fun _ _ ↦ reducePoly PQ N (centeredIntLift (x 0 0)) := by
  rw [centeredRebaseRuns_reduce h]
  funext r c
  rw [Subsingleton.elim r 0, Subsingleton.elim c 0]

set_option maxHeartbeats 64000000 in
/-- The digit companions of the attribute ciphertext: lane `jj + {{M}} a` is a compact encoding of
`P` times digit `a` of column `jj` of `C = (C_top, C_bot)` (lifted to the integers) under the
compact companion keys of that column and digit. -/
theorem encrypt_rhs_digits {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hsm : polyNorm sm ≤ 1)
    (jj a : Fin {{M}}) :
    RelE sm (fun k r ↦ rhsKeyC model key jj a (2 * k + r))
      (ctarget fun r ↦ centeredIntLift (colDigits (joinRow ![C00 0, C01 0]) (joinRow ![C10 0, C11 0]) jj a 0) *
        unitPair sm r)
      ![outputs.2.2.2.2.2.2.1 ⟨jj + {{M}} * a, by omega⟩, outputs.2.2.2.2.2.2.2.1 ⟨jj + {{M}} * a, by omega⟩]
      ![errAt tape (path ++ [46, (jj : Nat) + {{M}} * (a : Nat), 29]) encLaw,
        errAt tape (path ++ [46, (jj : Nat) + {{M}} * (a : Nat), 39]) encLaw] := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h41, -, hgather, hout⟩ := h
  subst hout hsecret
  obtain ⟨g42, g43, g44, g45, -, -, x42, -, -, x43, -, -, x44, -, -, x45, hlane⟩ :=
    hgather ⟨jj + {{M}} * a, by omega⟩
  obtain ⟨-, h41i⟩ := h41 ⟨jj + {{M}} * a, by omega⟩
  obtain ⟨hmem, hcol, hrow, hdig, hent⟩ := lane_indices jj.isLt a.isLt
  have hw41 : w.w_41_0 ⟨jj + {{M}} * a, by omega⟩ = Int.ofNat 0 := by
    rw [h41i]
    simp only [hmem, add_zero]
    rfl
  rw [hw41] at x42 x43 x44 x45 hlane
  obtain ⟨_, hf00⟩ := familyGet_eq (k := 0) x42
  obtain ⟨_, hf01⟩ := familyGet_eq (k := 0) x43
  obtain ⟨_, hf10⟩ := familyGet_eq (k := 0) x44
  obtain ⟨_, hf11⟩ := familyGet_eq (k := 0) x45
  subst hf00 hf01 hf10 hf11
  obtain ⟨w1, w3, w6, w7, w12, w15, w16, w19, w20, w21, w22, w23, w25, w29, w31, w33, w39, x1, -, -, -,
    -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x12, x15, -, -, -, -, -, -, -,
    -, x16, x19, -, -, -, -, -, -, -, -, x20, x21, -, -, -, -, -, -, -, -, -, -, -, -, x22, x23, x25,
    x29, -, -, -, -, x31, -, -, -, -, x33, x39, xout⟩ := hlane
  rw [hmem, hcol, hrow] at x3 x7 x31 x33
  rw [hcol] at x16 x20
  rw [hrow] at x12 x22
  rw [hdig] at x12
  rw [hent] at x22
  have hw1 := rebase_small (ternary_small hsm) x1
  rw [one_eq_const] at x6
  have hw6 := rebase_small one_small x6
  have hw12 := unitRow_window (c := (((a : Nat) % {{D}} : Nat) : Int)) (by omega) x12
  obtain ⟨_, hw16⟩ := sliceMatrix_column x16
  obtain ⟨_, hw20⟩ := sliceMatrix_column x20
  obtain ⟨_, hw22⟩ := sliceMatrix_column x22
  rw [concatColumns_entry x21, joinedEntry_single] at hw22
  simp only [Int.toNat_natCast, Fin.eta] at hw16 hw20 hw22
  have he : w22 0 0 = pairColumn (joinRow ![C00 0, C01 0] 0 jj) (joinRow ![C10 0, C11 0] 0 jj)
      ⟨(a : Nat) / {{D}}, by omega⟩ 0 := by
    rw [hw22]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw16, hw20]
    rw [← concat_joinRow (r := ![C00 0, C01 0]) x15, ← concat_joinRow (r := ![C10 0, C11 0]) x19]
  have hw23 := columnDigits_entry (q := Q) (K := {{D}}) rfl ⟨a, by omega⟩ he x23
    (rebase_self g0_runs x23) (colDigits_runs g0_runs _ _)
  -- The digit `d = U Dec(·)` of the lane, lifted to `PQ`.
  have hd : (matrixMul w12 w23 : ExactMatrix Q N 1 1) 0 0 =
      colDigits (joinRow ![C00 0, C01 0]) (joinRow ![C10 0, C11 0]) jj a 0 := by
    rw [matrixMul, hw12, unitRow_mul_entry (k := {{D}}) ((a : Nat) % {{D}}) (by omega)]
    exact hw23
  have hw25 := rebase_lift x25
  rw [hd] at hw25
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x29
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x39
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hkey : ∀ k r : Fin 2, rhsKeyC model key jj a (2 * k + r) = ![![w3, w31], ![w7, w33]] k r := by
    intro k r
    fin_cases k <;> fin_cases r
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x31).symm
    · exact (hashSample_eq x7).symm
    · exact (hashSample_eq x33).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c' = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      matrixMulScalarLeft_eq, hw1, hw6, hw25, he0, he1, errAt, unitPair, reduceMatrix, ctarget,
      matrixPolynomial_single, List.append_assoc, List.cons_append, List.nil_append]
    simp only [map_mul, map_one, map_natCast, map_intCast, Nat.cast_one, Int.cast_one,
      Nat.cast_ofNat, Int.cast_natCast, one_mul, mul_one, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one, one_smul, smul_eq_mul, Fin.val_zero, Fin.isValue]
    try ring

/-- The switching ciphertext `C_K = σ_K(H) - [0; σ_K(h)]`, from the helper key and the helper of
`K = K⁻¹`, both expanded. -/
noncomputable def switchTop (model : HashModel) (key : ByteArray) : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  fun j ↦ substituteMatrix K hK (helperKey model key 0 j)

noncomputable def switchBot (model : HashModel) (key : ByteArray) (hrow : Fin 2 → ExactMatrix Q N 1 {{D}}) :
    Fin 2 → ExactMatrix Q N 1 {{D}} :=
  fun j ↦ substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)

set_option maxHeartbeats 16000000 in
theorem encrypt_switch {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs) :
    ∃ w : Stage_encrypt.generatedRoot.Witness,
      Stage_encrypt.generatedRoot.body Backend.backend model tape path GeneratedClaim.stage_4_params
        (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs w ∧
      ![w.w_6_0 0, w.w_6_1 0] = switchTop model key ∧
      ![w.w_6_2 0, w.w_6_3 0] = switchBot model key
        ![expand (outputs.2.2.1 0), expand (outputs.2.2.2.1 0)] := by
  obtain ⟨w, hb⟩ := h
  refine ⟨w, hb, ?_⟩
  obtain ⟨-, -, -, -, -, h6, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hb
  obtain ⟨w1, w2, w3, w5, w6, w7, w8, w10, w11, w12, w13, w15, w16, w19, w20, w21, w23, w24, w26, w27,
    w29, w30, w32, w33, w35, w36, -, -, x1, x2, x3, x5, -, -, x6, -, -, x7, x8, x10, -, -, x11, -, -,
    x12, x13, x15, -, -, x16, -, -, -, -, x19, x20, x21, x23, -, -, x24, -, -, x26, x27, x29, -, -, x30,
    -, -, x32, x33, x35, -, -, x36, xout⟩ := h6 0
  have hs : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [{{K}}] = (K : Int) := by decide
  have hz : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [0] = Int.ofNat 0 := by decide
  rw [hs] at x1 x7 x12 x26
  rw [hz] at x19 x32
  have hkeyC : ∀ k j, helperKeyC model key k j = ![![w1, w7], ![w12, w26]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x1).symm
    · exact (hashSample_eq x7).symm
    · exact (hashSample_eq x12).symm
    · exact (hashSample_eq x26).symm
  have hkey : ∀ k j, helperKey model key k j = ![![w5, w10], ![w15, w29]] k j := by
    have h5 := expand_read x2 x3 x5
    have h10 := expand_read x2 x8 x10
    have h15 := expand_read x2 x13 x15
    have h29 := expand_read x2 x27 x29
    intro k j
    fin_cases k <;> fin_cases j <;> simp [helperKey, expandKey, hkeyC, h5, h10, h15, h29]
  obtain ⟨_, hw19⟩ := familyGet_eq x19
  obtain ⟨_, hw32⟩ := familyGet_eq x32
  have hw23 := expand_read x20 x21 x23
  have hw35 := expand_read x20 x33 x35
  rw [hw19] at hw23
  rw [hw32] at hw35
  have hw6 := routing_K x6
  have hw11 := routing_K x11
  have hw16 := routing_K x16
  have hw24 := routing_K x24
  have hw30 := routing_K x30
  have hw36 := routing_K x36
  subst hout
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := xout
  refine ⟨?_, ?_⟩
  · funext j
    fin_cases j
    · simp [switchTop, ho0, hw6, hkey]
    · simp [switchTop, ho1, hw11, hkey]
  · funext j
    fin_cases j
    · simp [switchBot, ho2, hw16, hw24, hw23, hkey, matrixSub]
    · simp [switchBot, ho3, hw30, hw36, hw35, hkey, matrixSub]

set_option maxHeartbeats 64000000 in
/-- The digit companions of the switching ciphertext: lane `jj + {{M}} a` is a compact encoding,
under `σ_K(s)`, of `P` times digit `a` of column `jj` of `C_K` (lifted to the integers). -/
theorem encrypt_aut_digits {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm) (hsm : polyNorm sm ≤ 1)
    (jj a : Fin {{M}}) :
    RelE (substitution K hK sm) (fun k r ↦ autKeyC model key jj a (2 * k + r))
      (ctarget fun r ↦ centeredIntLift (colDigits (joinRow (switchTop model key))
          (joinRow (switchBot model key ![expand (outputs.2.2.1 0), expand (outputs.2.2.2.1 0)])) jj a 0) *
        unitPair (substitution K hK sm) r)
      ![outputs.1 ⟨jj + {{M}} * a, by omega⟩, outputs.2.1 ⟨jj + {{M}} * a, by omega⟩]
      ![errAt tape (path ++ [8, (jj : Nat) + {{M}} * (a : Nat), 35]) encLaw,
        errAt tape (path ++ [8, (jj : Nat) + {{M}} * (a : Nat), 45]) encLaw] := by
  obtain ⟨w, hb, htop, hbot⟩ := encrypt_switch h
  obtain ⟨-, -, -, -, -, -, -, h7, -, h8, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hb
  subst hout hsecret
  obtain ⟨-, h7i⟩ := h7 ⟨jj + {{M}} * a, by omega⟩
  obtain ⟨hmem, hcol, hrow, hdig, hent⟩ := lane_indices jj.isLt a.isLt
  have hw7 : w.w_7_0 ⟨jj + {{M}} * a, by omega⟩ = Int.ofNat 0 := by
    rw [h7i]
    simp only [hmem, add_zero]
    rfl
  have hlane := h8 ⟨jj + {{M}} * a, by omega⟩
  rw [hw7] at hlane
  obtain ⟨w1, w2, w4, w7, w8, w13, w16, w18, w19, w20, w22, w24, w25, w26, w27, w28, w29, w31, w35,
    w37, w39, w45, -, -, -, x1, x2, -, -, -, -, -, -, x4, x7, -, -, -, -, -, -, x8, -, -, -, -, -, -, -,
    -, -, -, -, -, -, -, x13, -, -, x16, -, -, x18, x19, -, -, -, -, -, -, -, -, x20, -, -, x22, -, -,
    x24, x25, -, -, -, -, -, -, -, -, x26, x27, -, -, -, -, -, -, -, -, -, -, -, -, x28, x29, x31, x35,
    -, -, -, -, -, -, x37, -, -, -, -, -, -, x39, x45, xout⟩ := hlane
  rw [hmem] at x1 x4 x8 x37 x39
  rw [hcol, hrow] at x4 x8 x37 x39
  rw [hcol] at x20 x26
  rw [hrow] at x13 x28
  rw [hdig] at x13
  rw [hent] at x28
  have hs0 : MxxIR.selectInt 0 [{{K}}] = (K : Int) := by decide
  rw [hs0] at x1 x4 x8 x37 x39
  obtain ⟨hodd, hw1⟩ := ringAutomorphismRuns_eq hQ hN x1
  have hσ : substitution (n := N) (R := ZMod Q) (Int.toNat (K : Int)) hodd (reducePoly Q N sm) =
      reducePoly Q N (substitution K hK sm) := by
    rw [reducePoly_substitution]
    rfl
  have hσsm : polyNorm (substitution K hK sm) ≤ 1 :=
    (polyNorm_substitution_le hN hK hKcoprime sm).trans hsm
  have h1 : w1 = fun _ _ ↦ reducePoly Q N (substitution K hK sm) := by
    rw [hw1]
    funext r c
    exact hσ
  rw [h1] at x2
  have hw2 := rebase_small (ternary_small hσsm) x2
  rw [one_eq_const] at x7
  have hw7' := rebase_small one_small x7
  obtain ⟨_, hf00⟩ := familyGet_eq (k := 0) x16
  obtain ⟨_, hf01⟩ := familyGet_eq (k := 0) x18
  obtain ⟨_, hf10⟩ := familyGet_eq (k := 0) x22
  obtain ⟨_, hf11⟩ := familyGet_eq (k := 0) x24
  have hw13 := unitRow_window (c := (((a : Nat) % {{D}} : Nat) : Int)) (by omega) x13
  obtain ⟨_, hw20⟩ := sliceMatrix_column x20
  obtain ⟨_, hw26⟩ := sliceMatrix_column x26
  obtain ⟨_, hw28⟩ := sliceMatrix_column x28
  rw [concatColumns_entry x27, joinedEntry_single] at hw28
  simp only [Int.toNat_natCast, Fin.eta] at hw20 hw26 hw28
  have h19 : w19 = joinRow (switchTop model key) := by
    rw [← htop]
    refine concat_joinRow (r := ![w.w_6_0 0, w.w_6_1 0]) ?_
    rw [hf00, hf01] at x19
    exact x19
  have h25 : w25 = joinRow (switchBot model key ![expand ((w.w_5_0) 0), expand ((w.w_5_1) 0)]) := by
    rw [← hbot]
    refine concat_joinRow (r := ![w.w_6_2 0, w.w_6_3 0]) ?_
    rw [hf10, hf11] at x25
    exact x25
  have he : w28 0 0 = pairColumn (joinRow (switchTop model key) 0 jj)
      (joinRow (switchBot model key ![expand ((w.w_5_0) 0), expand ((w.w_5_1) 0)]) 0 jj)
      ⟨(a : Nat) / {{D}}, by omega⟩ 0 := by
    rw [hw28]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw20, hw26, h19, h25]
  have hw29 := columnDigits_entry (q := Q) (K := {{D}}) rfl ⟨a, by omega⟩ he x29
    (rebase_self g0_runs x29) (colDigits_runs g0_runs _ _)
  have hd : (matrixMul w13 w29 : ExactMatrix Q N 1 1) 0 0 =
      colDigits (joinRow (switchTop model key))
        (joinRow (switchBot model key ![expand ((w.w_5_0) 0), expand ((w.w_5_1) 0)])) jj a 0 := by
    rw [matrixMul, hw13, unitRow_mul_entry (k := {{D}}) ((a : Nat) % {{D}}) (by omega)]
    exact hw29
  have hw31 := rebase_lift x31
  rw [hd] at hw31
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x35
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x45
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hkey : ∀ k r : Fin 2, autKeyC model key jj a (2 * k + r) = ![![w4, w37], ![w8, w39]] k r := by
    intro k r
    fin_cases k <;> fin_cases r
    · exact (hashSample_eq x4).symm
    · exact (hashSample_eq x37).symm
    · exact (hashSample_eq x8).symm
    · exact (hashSample_eq x39).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c' = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      matrixMulScalarLeft_eq, hw2, hw7', hw31, he0, he1, errAt, unitPair, reduceMatrix, ctarget,
      matrixPolynomial_single, List.append_assoc, List.cons_append, List.nil_append]
    simp only [map_mul, map_one, map_natCast, map_intCast, Nat.cast_one, Int.cast_one,
      Nat.cast_ofNat, Int.cast_natCast, one_mul, mul_one, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_fin_one, one_smul, smul_eq_mul, Fin.val_zero, Fin.isValue]
    try ring

/-- A Gaussian tape sample lies within its cutoff. -/
theorem errAt_bound {tape : SampleTape} {site : List Nat} {sigma : Rat} {c : Int} {b : Nat}
    (hc : c.toNat = b) {q : Nat} {x : ExactMatrix q N 1 1} (h : gaussianSampleAt tape site sigma c x) :
    CoeffBound (errAt tape site (.gaussian sigma c)) b :=
  hc ▸ (gaussianSampleAt_explicit hN h).2

set_option maxHeartbeats 64000000 in
/-- The errors of the compact encodings of encryption lie within their cutoffs. -/
theorem encrypt_err_bounds {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 {{D}}}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs) :
    (∀ j, CoeffBound (![errAt tape (path ++ [28]) encLaw, errAt tape (path ++ [39]) encLaw] j) {{CF}}) ∧
    (∀ j, CoeffBound (![errAt tape (path ++ [5, 0, 14]) helperLaw,
      errAt tape (path ++ [5, 0, 23]) helperLaw] j) {{CE}}) ∧
    (∀ (i : Fin {{L}}) j, CoeffBound (![errAt tape (path ++ [46, (i : Nat), 29]) encLaw,
      errAt tape (path ++ [46, (i : Nat), 39]) encLaw] j) {{CF}}) ∧
    (∀ (i : Fin {{L}}) j, CoeffBound (![errAt tape (path ++ [8, (i : Nat), 35]) encLaw,
      errAt tape (path ++ [8, (i : Nat), 45]) encLaw] j) {{CF}}) := by
  obtain ⟨w, -, -, -, h5, -, -, -, -, -, h8, -, -, -, -, -, -, x28, -, -, -, x39, -, -, -, hgather, -⟩ := h
  have hv0 : ∀ {a b : ErrorMatrix N 1 1}, (![a, b] : Fin 2 → ErrorMatrix N 1 1) 0 = a := fun {_ _} ↦ rfl
  have hv1 : ∀ {a b : ErrorMatrix N 1 1}, (![a, b] : Fin 2 → ErrorMatrix N 1 1) 1 = b := fun {_ _} ↦ rfl
  refine ⟨fun j ↦ ?_, fun j ↦ ?_, fun i j ↦ ?_, fun i j ↦ ?_⟩
  · fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x28
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x39
  · obtain ⟨w2, w5, w10, w11, w14, w16, w18, w23, -, -, x2, -, -, x5, -, -, x10, x11, x14, -, -, x16, -,
      -, x18, x23, xout⟩ := h5 0
    rw [List.append_assoc] at x14 x23
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x14
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x23
  · obtain ⟨g42, g43, g44, g45, -, -, x42, -, -, x43, -, -, x44, -, -, x45, hlane⟩ := hgather i
    obtain ⟨w1, w3, w6, w7, w12, w15, w16, w19, w20, w21, w22, w23, w25, w29, w31, w33, w39, x1, -, -, -,
      -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x12, x15, -, -, -, -, -, -, -,
      -, x16, x19, -, -, -, -, -, -, -, -, x20, x21, -, -, -, -, -, -, -, -, -, -, -, -, x22, x23, x25,
      x29, -, -, -, -, x31, -, -, -, -, x33, x39, xout⟩ := hlane
    rw [List.append_assoc] at x29 x39
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x29
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x39
  · obtain ⟨w1, w2, w4, w7, w8, w13, w16, w18, w19, w20, w22, w24, w25, w26, w27, w28, w29, w31, w35,
      w37, w39, w45, -, -, -, x1, x2, -, -, -, -, -, -, x4, x7, -, -, -, -, -, -, x8, -, -, -, -, -, -, -,
      -, -, -, -, -, -, -, x13, -, -, x16, -, -, x18, x19, -, -, -, -, -, -, -, -, x20, -, -, x22, -, -,
      x24, x25, -, -, -, -, -, -, -, -, x26, x27, -, -, -, -, -, -, -, -, -, -, -, -, x28, x29, x31, x35,
      -, -, -, -, -, -, x37, -, -, -, -, -, -, x39, x45, xout⟩ := h8 i
    rw [List.append_assoc] at x35 x45
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x35
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x45

end PeProof
