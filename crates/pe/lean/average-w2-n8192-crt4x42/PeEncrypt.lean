import PePreprocess
import ExponentErrorForm

/-!
# Encryption

The Dynamic packages of the ciphertext, under the secret `s`: the encoding of the initial state
`X⁰`, the helper of `K`, and the digit companions of the attribute ciphertext and of the switching
ciphertext `C_K`. Every error is named by its tape site, so each output is an explicit function
of the secret, the hash keys, the attribute ciphertext, and the tape.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The law of the encryption errors. -/
abbrev encLaw : SamplerLaw := .gaussian (319 / 100) 20

/-- The law of the helper errors. -/
abbrev helperLaw : SamplerLaw := .gaussian (319 / 100) 20

/-- An error component sampled at `site`. -/
noncomputable abbrev errAt (tape : SampleTape) (site : List Nat) (law : SamplerLaw) : ErrorMatrix N 1 8 :=
  tapeWitness tape site law

abbrev EncryptOutputs := (Fin 256 → ExactMatrix Q N 1 8) × (Fin 256 → ExactMatrix Q N 1 8) ×
  (Fin 1 → ExactMatrix Q N 1 8) × (Fin 1 → ExactMatrix Q N 1 8) × ExactMatrix Q N 1 8 ×
  ExactMatrix Q N 1 8 × (Fin 256 → ExactMatrix Q N 1 8) × (Fin 256 → ExactMatrix Q N 1 8) × Unit

theorem message_one_zero : (message (n := N) 1 0 : ErrorPoly N) = 1 := by
  simp [message]

set_option maxHeartbeats 16000000 in
/-- The initial state is an encoding of `X⁰` under the input key. -/
theorem encrypt_input {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g) :
    RelE sm (inputKey model key) (fun j ↦ reducePoly Q N (((1 : Nat) : Int) * 1 * unitPair sm j) • g)
      ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ![errAt tape (path ++ [25]) encLaw, errAt tape (path ++ [35]) encLaw] := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, h9, h11, h14, h15, h25, h27, h29, h35, -, -, -, -, -,
    hout⟩ := h
  have hg15 : w.w_15_0 = g :=
    (centeredRebaseRuns_unique h15 (centeredRebaseRuns_self hQ hN _)).trans (gadgetMatrixRuns_unique h14 hg)
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN h25
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN h35
  subst hout hsecret
  have hkey : ∀ k j, inputKey model key k j = ![![w.w_9_0, w.w_27_0], ![w.w_11_0, w.w_29_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h9).symm
    · exact (hashSample_eq h27).symm
    · exact (hashSample_eq h11).symm
    · exact (hashSample_eq h29).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [matrixAdd, matrixSub, matrixMulScalarRight_eq, multiplyMonomial_liftInteger,
      matrixPolynomial_single, hg15, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      matrixMulScalarLeft_eq, message_one_zero, he0, he1, errAt, unitPair, reduceMatrix]
    simp
    try ring

set_option maxHeartbeats 16000000 in
/-- The helper of `K`: an encoding of `σ_K(s)` against the gadget. -/
theorem encrypt_helper {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g) :
    RelE sm (helperKey model key) (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
        unitPair (substitution K hK sm) j) • g)
      ![outputs.2.2.1 0, outputs.2.2.2.1 0]
      ![errAt tape (path ++ [4, 0, 12]) helperLaw, errAt tape (path ++ [4, 0, 20]) helperLaw] := by
  obtain ⟨w, h3, -, h4, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := h
  obtain ⟨w2, w5, w8, w12, w14, w16, w20, -, -, x2, -, -, x5, -, -, x8, x12, -, -, x14, -, -, x16,
    x20, xout⟩ := h4 0
  have hg3 : w.w_3_0 = g := gadgetMatrixRuns_unique h3 hg
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x12
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x20
  have hs : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [8193] = (K : Int) := by decide
  rw [hs] at x2 x5 x14 x16 x8
  obtain ⟨hodd, hw8⟩ := ringAutomorphismRuns_eq hQ hN x8
  have hkey : ∀ k j, helperKey model key k j = ![![w2, w14], ![w5, w16]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x2).symm
    · exact (hashSample_eq x14).symm
    · exact (hashSample_eq x5).symm
    · exact (hashSample_eq x16).symm
  subst hout hsecret
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hσ : substitution (n := N) (R := ZMod Q) (Int.toNat (K : Int)) hodd (reducePoly Q N sm) =
      reducePoly Q N (substitution K hK sm) := by
    rw [reducePoly_substitution]
    rfl
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, hg3, hw8, hσ, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, matrixMulScalarLeft_eq, he0, he1, errAt, unitPair, reduceMatrix]
    simp only [Matrix.one_apply_eq, one_smul, List.append_assoc, List.cons_append, List.nil_append,
      Fin.val_zero, Matrix.cons_val_zero, Matrix.cons_val_one, Nat.cast_one,
      Int.cast_one, one_mul, map_one]
    try rfl

/-- Digits of a decomposition rebase to themselves at the same modulus. -/
theorem rebase_self {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g)
    {x : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 8 1}
    (hu : gadgetDecomposeRuns Backend.backend 2097152 8 x u) : compactCenteredRebaseRuns 1048576 u u := by
  have hb := (decompose_entry hQ hN Backend.layout0 layoutQ rfl hg hu).2
  exact ⟨Or.inl dvd_rfl, hb, centeredRebaseRuns_self hQ hN u, hb⟩

/-- The column digits of the joined rows run. -/
theorem colDigits_runs {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g)
    (x y : ExactPoly Q N) :
    ∃ D' : ExactMatrix Q N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn x y) D' ∧
        compactCenteredRebaseRuns 1048576 D'
          (columnDigits Q (8 + 8) Backend.backend 2097152 8 1048576 x y) :=
  columnDigits_runs hQ hQ hN hg Backend.layout0 layoutQ rfl rfl (by rw [show Backend.layout0.base = 2097152 from rfl]; norm_num)
    (by rw [show Backend.layout0.base = 2097152 from rfl]; norm_num) x y

theorem lane_indices {jj a : Nat} (hjj : jj < 16) (ha : a < 16) :
    Int.fdiv (Int.ofNat (jj + 16 * a)) 256 = 0 ∧
      Int.fmod (Int.ofNat (jj + 16 * a)) 16 = ((jj : Nat) : Int) ∧
      Int.fmod (Int.fdiv (Int.ofNat (jj + 16 * a)) 16) 16 = ((a : Nat) : Int) ∧
      Int.fmod ((a : Nat) : Int) 8 = (((a : Nat) % 8 : Nat) : Int) ∧
      Int.fdiv ((a : Nat) : Int) 8 = (((a : Nat) / 8 : Nat) : Int) := by
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

set_option maxHeartbeats 64000000 in
/-- The digit companions of the attribute ciphertext: lane `jj + 16 a` encrypts digit `a` of column
`jj` of `C = (C_top, C_bot)` under the companion keys of that column and digit. -/
theorem encrypt_rhs_digits {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g) (jj a : Fin 16) :
    RelE sm (fun k r ↦ rhsKey model key jj a (2 * k + r))
      (fun r ↦ (colDigits (joinRow ![C00 0, C01 0]) (joinRow ![C10 0, C11 0]) jj a 0 *
        reducePoly Q N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.1 ⟨jj + 16 * a, by omega⟩, outputs.2.2.2.2.2.2.2.1 ⟨jj + 16 * a, by omega⟩]
      ![errAt tape (path ++ [43, (jj : Nat) + 16 * (a : Nat), 26]) encLaw,
        errAt tape (path ++ [43, (jj : Nat) + 16 * (a : Nat), 35]) encLaw] := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h37, h42, -, hgather, hout⟩ := h
  subst hout hsecret
  obtain ⟨g38, g39, g40, g41, -, -, x38, -, -, x39, -, -, x40, -, -, x41, hlane⟩ :=
    hgather ⟨jj + 16 * a, by omega⟩
  obtain ⟨-, h37i⟩ := h37 ⟨jj + 16 * a, by omega⟩
  obtain ⟨hmem, hcol, hrow, hdig, hent⟩ := lane_indices jj.isLt a.isLt
  have hw37 : w.w_37_0 ⟨jj + 16 * a, by omega⟩ = Int.ofNat 0 := by
    rw [h37i]
    simp only [hmem, add_zero]
    rfl
  rw [hw37] at x38 x39 x40 x41 hlane
  obtain ⟨_, hf00⟩ := familyGet_eq (k := 0) x38
  obtain ⟨_, hf01⟩ := familyGet_eq (k := 0) x39
  obtain ⟨_, hf10⟩ := familyGet_eq (k := 0) x40
  obtain ⟨_, hf11⟩ := familyGet_eq (k := 0) x41
  subst hf00 hf01 hf10 hf11
  obtain ⟨w2, w5, w9, w12, w13, w16, w17, w18, w19, w20, w23, w26, w28, w30, w35, -, -, -, -, x2, -,
    -, -, -, x5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x9, x12, -, -, -, -, -, -, -, -, x13,
    x16, -, -, -, -, -, -, -, -, x17, x18, -, -, -, -, -, -, -, -, -, -, -, -, x19, x20, x23, x26,
    -, -, -, -, x28, -, -, -, -, x30, x35, xout⟩ := hlane
  rw [hmem, hcol, hrow] at x2 x5 x28 x30
  rw [hcol] at x13 x17
  rw [hrow] at x9 x19
  rw [hdig] at x9
  rw [hent] at x19
  have hw9 := unitRow_window (c := (((a : Nat) % 8 : Nat) : Int)) (by omega) x9
  obtain ⟨_, hw13⟩ := sliceMatrix_column x13
  obtain ⟨_, hw17⟩ := sliceMatrix_column x17
  obtain ⟨_, hw19⟩ := sliceMatrix_column x19
  rw [concatColumns_entry x18, joinedEntry_single] at hw19
  simp only [Int.toNat_natCast, Fin.eta] at hw13 hw17 hw19
  have he : w19 0 0 = pairColumn (joinRow ![C00 0, C01 0] 0 jj) (joinRow ![C10 0, C11 0] 0 jj)
      ⟨(a : Nat) / 8, by omega⟩ 0 := by
    rw [hw19]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw13, hw17]
    rw [← concat_joinRow (r := ![C00 0, C01 0]) x12, ← concat_joinRow (r := ![C10 0, C11 0]) x16]
  have hw20 := columnDigits_entry (q := Q) (K := 8) rfl ⟨a, by omega⟩ he x20
    (rebase_self h42 x20) (colDigits_runs h42 _ _)
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x26
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x35
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 8, w23 0 c = colDigits (joinRow ![C00 0, C01 0]) (joinRow ![C10 0, C11 0]) jj a 0 *
      w.w_42_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x23 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 8) ((a : Nat) % 8) (by omega)] at h1
    exact h1.trans (congrArg₂ (· * ·) hw20 rfl)
  have hg42 : w.w_42_0 = g := gadgetMatrixRuns_unique h42 hg
  have hkey : ∀ k r : Fin 2, rhsKey model key jj a (2 * k + r) = ![![w2, w28], ![w5, w30]] k r := by
    intro k r
    fin_cases k <;> fin_cases r
    · exact (hashSample_eq x2).symm
    · exact (hashSample_eq x28).symm
    · exact (hashSample_eq x5).symm
    · exact (hashSample_eq x30).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply,
      Matrix.smul_apply, matrixMulScalarLeft_eq, he0, he1, errAt, unitPair, reduceMatrix, hw23, hg42]
    simp only [Matrix.one_apply_eq, List.append_assoc, List.cons_append, List.nil_append,
      Matrix.cons_val_zero, Matrix.cons_val_one, smul_eq_mul, mul_one, one_mul, map_one]
    try ring

/-- The switching ciphertext `C_K = σ_K(H) - [0; σ_K(h)]`, from the helper key and the helper of
`K = K⁻¹`. -/
noncomputable def switchTop (model : HashModel) (key : ByteArray) : Fin 2 → ExactMatrix Q N 1 8 :=
  fun j ↦ substituteMatrix K hK (helperKey model key 0 j)

noncomputable def switchBot (model : HashModel) (key : ByteArray) (hrow : Fin 2 → ExactMatrix Q N 1 8) :
    Fin 2 → ExactMatrix Q N 1 8 :=
  fun j ↦ substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)

set_option maxHeartbeats 16000000 in
theorem encrypt_switch {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs) :
    ∃ w : Stage_encrypt.generatedRoot.Witness,
      Stage_encrypt.generatedRoot.body Backend.backend model tape path GeneratedClaim.stage_4_params
        (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs w ∧
      ![w.w_5_0 0, w.w_5_1 0] = switchTop model key ∧
      ![w.w_5_2 0, w.w_5_3 0] = switchBot model key ![outputs.2.2.1 0, outputs.2.2.2.1 0] := by
  obtain ⟨w, hb⟩ := h
  refine ⟨w, hb, ?_⟩
  obtain ⟨-, -, -, -, h5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hb
  obtain ⟨w1, w2, w3, w4, w5, w6, w9, w10, w12, w13, w15, w16, -, -, x1, -, -, x2, -, -, x3, -, -,
    x4, -, -, x5, -, -, x6, -, -, -, -, x9, -, -, x10, -, -, x12, -, -, x13, -, -, x15, -, -, x16,
    xout⟩ := h5 0
  have hs : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [8193] = (K : Int) := by decide
  have hz : MxxIR.selectInt (Int.ofNat ((0 : Fin 1) : Nat)) [0] = Int.ofNat 0 := by decide
  rw [hs] at x1 x3 x5 x12
  rw [hz] at x9 x15
  have hkey : ∀ k j, helperKey model key k j = ![![w1, w3], ![w5, w12]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x1).symm
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x5).symm
    · exact (hashSample_eq x12).symm
  obtain ⟨_, hw9⟩ := familyGet_eq x9
  obtain ⟨_, hw15⟩ := familyGet_eq x15
  have hw2 := routing_K x2
  have hw4 := routing_K x4
  have hw6 := routing_K x6
  have hw10 := routing_K x10
  have hw13 := routing_K x13
  have hw16 := routing_K x16
  subst hout
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := xout
  refine ⟨?_, ?_⟩
  · funext j
    fin_cases j
    · simp [switchTop, ho0, hw2, hkey]
    · simp [switchTop, ho1, hw4, hkey]
  · funext j
    fin_cases j
    · simp [switchBot, ho2, hw6, hw10, hw9, hkey, matrixSub]
    · simp [switchBot, ho3, hw13, hw16, hw15, hkey, matrixSub]

set_option maxHeartbeats 64000000 in
/-- The digit companions of the switching ciphertext: lane `jj + 16 a` encrypts, under `σ_K(s)`,
digit `a` of column `jj` of `C_K`. -/
theorem encrypt_aut_digits {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q N sm)
    {g : ExactMatrix Q N 1 8} (hg : gadgetMatrixRuns Backend.backend 2097152 8 g) (jj a : Fin 16) :
    RelE (substitution K hK sm) (fun k r ↦ autKey model key jj a (2 * k + r))
      (fun r ↦ (colDigits (joinRow (switchTop model key))
          (joinRow (switchBot model key ![outputs.2.2.1 0, outputs.2.2.2.1 0])) jj a 0 *
        reducePoly Q N (unitPair (substitution K hK sm) r)) • g)
      ![outputs.1 ⟨jj + 16 * a, by omega⟩, outputs.2.1 ⟨jj + 16 * a, by omega⟩]
      ![errAt tape (path ++ [8, (jj : Nat) + 16 * (a : Nat), 32]) encLaw,
        errAt tape (path ++ [8, (jj : Nat) + 16 * (a : Nat), 41]) encLaw] := by
  obtain ⟨w, hb, htop, hbot⟩ := encrypt_switch h
  obtain ⟨-, -, -, -, -, -, h6, h7, -, h8, -, -, -, -, -, -, -, -, -, -, -, -, -, hout⟩ := hb
  subst hout hsecret
  obtain ⟨-, h6i⟩ := h6 ⟨jj + 16 * a, by omega⟩
  obtain ⟨hmem, hcol, hrow, hdig, hent⟩ := lane_indices jj.isLt a.isLt
  have hw6 : w.w_6_0 ⟨jj + 16 * a, by omega⟩ = Int.ofNat 0 := by
    rw [h6i]
    simp only [hmem, add_zero]
    rfl
  have hlane := h8 ⟨jj + 16 * a, by omega⟩
  rw [hw6] at hlane
  obtain ⟨w1, w3, w6, w10, w13, w15, w16, w17, w19, w21, w22, w23, w24, w25, w26, w29, w32, w34, w36,
    w41, -, -, -, x1, -, -, -, -, -, -, x3, -, -, -, -, -, -, x6, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, x10, -, -, x13, -, -, x15, x16, -, -, -, -, -, -, -, -, x17, -, -, x19, -, -, x21, x22, -,
    -, -, -, -, -, -, -, x23, x24, -, -, -, -, -, -, -, -, -, -, -, -, x25, x26, x29, x32, -, -, -,
    -, -, -, x34, -, -, -, -, -, -, x36, x41, xout⟩ := hlane
  rw [hmem] at x1 x3 x6 x34 x36
  rw [hcol, hrow] at x3 x6 x34 x36
  rw [hcol] at x17 x23
  rw [hrow] at x10 x25
  rw [hdig] at x10
  rw [hent] at x25
  have hs0 : MxxIR.selectInt 0 [8193] = (K : Int) := by decide
  rw [hs0] at x1 x3 x6 x34 x36
  obtain ⟨hodd, hw1⟩ := ringAutomorphismRuns_eq hQ hN x1
  obtain ⟨_, hf00⟩ := familyGet_eq (k := 0) x13
  obtain ⟨_, hf01⟩ := familyGet_eq (k := 0) x15
  obtain ⟨_, hf10⟩ := familyGet_eq (k := 0) x19
  obtain ⟨_, hf11⟩ := familyGet_eq (k := 0) x21
  have hw10 := unitRow_window (c := (((a : Nat) % 8 : Nat) : Int)) (by omega) x10
  obtain ⟨_, hw17⟩ := sliceMatrix_column x17
  obtain ⟨_, hw23⟩ := sliceMatrix_column x23
  obtain ⟨_, hw25⟩ := sliceMatrix_column x25
  rw [concatColumns_entry x24, joinedEntry_single] at hw25
  simp only [Int.toNat_natCast, Fin.eta] at hw17 hw23 hw25
  have h16 : w16 = joinRow (switchTop model key) := by
    rw [← htop]
    refine concat_joinRow (r := ![w.w_5_0 0, w.w_5_1 0]) ?_
    rw [hf00, hf01] at x16
    exact x16
  have h22 : w22 = joinRow (switchBot model key ![(w.w_4_0) 0, (w.w_4_1) 0]) := by
    rw [← hbot]
    refine concat_joinRow (r := ![w.w_5_2 0, w.w_5_3 0]) ?_
    rw [hf10, hf11] at x22
    exact x22
  have he : w25 0 0 = pairColumn (joinRow (switchTop model key) 0 jj)
      (joinRow (switchBot model key ![(w.w_4_0) 0, (w.w_4_1) 0]) 0 jj)
      ⟨(a : Nat) / 8, by omega⟩ 0 := by
    rw [hw25]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw17, hw23, h16, h22]
  have hw26 := columnDigits_entry (q := Q) (K := 8) rfl ⟨a, by omega⟩ he x26
    (rebase_self h7 x26) (colDigits_runs h7 _ _)
  obtain ⟨he0, -⟩ := gaussianSampleAt_explicit hN x32
  obtain ⟨he1, -⟩ := gaussianSampleAt_explicit hN x41
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw29 : ∀ c : Fin 8, w29 0 c = colDigits (joinRow (switchTop model key))
      (joinRow (switchBot model key ![(w.w_4_0) 0, (w.w_4_1) 0])) jj a 0 * w.w_7_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x29 c.val c.isLt
    rw [matrixMul, hw10, unitRow_mul_entry (k := 8) ((a : Nat) % 8) (by omega)] at h1
    exact h1.trans (congrArg₂ (· * ·) hw26 rfl)
  have hg7 : w.w_7_0 = g := gadgetMatrixRuns_unique h7 hg
  have hσ : substitution (n := N) (R := ZMod Q) (Int.toNat (K : Int)) hodd (reducePoly Q N sm) =
      reducePoly Q N (substitution K hK sm) := by
    rw [reducePoly_substitution]
    rfl
  have hkey : ∀ k r : Fin 2, autKey model key jj a (2 * k + r) = ![![w3, w34], ![w6, w36]] k r := by
    intro k r
    fin_cases k <;> fin_cases r
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x34).symm
    · exact (hashSample_eq x6).symm
    · exact (hashSample_eq x36).symm
  intro j
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _
  all_goals
    simp only [Fin.zero_eta, Fin.mk_one, Matrix.cons_val_zero, Matrix.cons_val_one]
    simp only [ho0, ho1, matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, hw1, hσ,
      Matrix.smul_apply, matrixMulScalarLeft_eq, he0, he1, errAt, unitPair, reduceMatrix, hw29, hg7]
    simp only [Matrix.one_apply_eq, List.append_assoc, List.cons_append, List.nil_append,
      Matrix.cons_val_zero, Matrix.cons_val_one, smul_eq_mul, mul_one, one_mul, map_one]
    try ring

/-- A Gaussian tape sample lies within its cutoff. -/
theorem errAt_bound {tape : SampleTape} {site : List Nat} {sigma : Rat} {c : Int} {b : Nat}
    (hc : c.toNat = b) {q : Nat} {x : ExactMatrix q N 1 8} (h : gaussianSampleAt tape site sigma c x) :
    CoeffBound (errAt tape site (.gaussian sigma c)) b :=
  hc ▸ (gaussianSampleAt_explicit hN h).2

set_option maxHeartbeats 64000000 in
/-- The errors of encryption lie within their cutoffs. -/
theorem encrypt_err_bounds {model : HashModel} {tape : SampleTape} {path : List Nat}
    {secret : ExactMatrix Q N 1 1} {key : ByteArray} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {outputs : EncryptOutputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model tape path GeneratedClaim.stage_4_params
      (secret, key, 1, 0, C00, C01, C10, C11, ()) outputs) :
    (∀ j, CoeffBound (![errAt tape (path ++ [25]) encLaw, errAt tape (path ++ [35]) encLaw] j) 20) ∧
    (∀ j, CoeffBound (![errAt tape (path ++ [4, 0, 12]) helperLaw,
      errAt tape (path ++ [4, 0, 20]) helperLaw] j) 20) ∧
    (∀ (i : Fin 256) j, CoeffBound (![errAt tape (path ++ [43, (i : Nat), 26]) encLaw,
      errAt tape (path ++ [43, (i : Nat), 35]) encLaw] j) 20) ∧
    (∀ (i : Fin 256) j, CoeffBound (![errAt tape (path ++ [8, (i : Nat), 32]) encLaw,
      errAt tape (path ++ [8, (i : Nat), 41]) encLaw] j) 20) := by
  obtain ⟨w, -, -, h4, -, -, -, -, -, -, h8, -, -, -, -, h25, -, -, h35, -, -, -, -, hgather, -⟩ := h
  have hv0 : ∀ {a b : ErrorMatrix N 1 8}, (![a, b] : Fin 2 → ErrorMatrix N 1 8) 0 = a := fun {_ _} ↦ rfl
  have hv1 : ∀ {a b : ErrorMatrix N 1 8}, (![a, b] : Fin 2 → ErrorMatrix N 1 8) 1 = b := fun {_ _} ↦ rfl
  refine ⟨fun j ↦ ?_, fun j ↦ ?_, fun i j ↦ ?_, fun i j ↦ ?_⟩
  · fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl h25
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl h35
  · obtain ⟨w2, w5, w8, w12, w14, w16, w20, -, -, x2, -, -, x5, -, -, x8, x12, -, -, x14, -, -, x16,
      x20, xout⟩ := h4 0
    rw [List.append_assoc] at x12 x20
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x12
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x20
  · obtain ⟨g38, g39, g40, g41, -, -, x38, -, -, x39, -, -, x40, -, -, x41, hlane⟩ := hgather i
    obtain ⟨w2, w5, w9, w12, w13, w16, w17, w18, w19, w20, w23, w26, w28, w30, w35, -, -, -, -, x2, -,
      -, -, -, x5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x9, x12, -, -, -, -, -, -, -, -, x13,
      x16, -, -, -, -, -, -, -, -, x17, x18, -, -, -, -, -, -, -, -, -, -, -, -, x19, x20, x23, x26,
      -, -, -, -, x28, -, -, -, -, x30, x35, xout⟩ := hlane
    rw [List.append_assoc] at x26 x35
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x26
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x35
  · obtain ⟨w1, w3, w6, w10, w13, w15, w16, w17, w19, w21, w22, w23, w24, w25, w26, w29, w32, w34, w36,
      w41, -, -, -, x1, -, -, -, -, -, -, x3, -, -, -, -, -, -, x6, -, -, -, -, -, -, -, -, -, -, -, -,
      -, -, x10, -, -, x13, -, -, x15, x16, -, -, -, -, -, -, -, -, x17, -, -, x19, -, -, x21, x22, -,
      -, -, -, -, -, -, -, x23, x24, -, -, -, -, -, -, -, -, -, -, -, -, x25, x26, x29, x32, -, -, -,
      -, -, -, x34, -, -, -, -, -, -, x36, x41, xout⟩ := h8 i
    rw [List.append_assoc] at x32 x41
    fin_cases j
    · simp only [Fin.zero_eta, hv0]
      exact errAt_bound rfl x32
    · simp only [Fin.mk_one, hv1]
      exact errAt_bound rfl x41

end PeProof
