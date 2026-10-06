import NandSetup

/-!
# Digit companions

Dynamic encryption stores, for every digit `d` of the decomposition of a ciphertext column and every
row of the decomposition of the state key, a companion `z = s B - d (s, 1) G + e` under the digit
key `B`. The RHS companions of a round encrypt the digits of the RHS ciphertext; the LUT
companions of branch `σ` encrypt `σ⁻¹` of the digits of the switching ciphertext of `σ`, so that
`σ(z)` is a companion of the digit itself under the substituted secret. Each lane decomposes only
the ciphertext entry that holds its digit row; `columnDigits_entry` identifies that digit with the
row of the column digits.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`0` RHS: lane `jj + 24 a` encrypts row `a` of the digits of
column `jj` of the RHS ciphertext. -/
theorem encrypt_rhs_digits0 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q7 N 1 14} (hg' : gadgetMatrixRuns Backend.backend 1048576 14 g')
    (hg : centeredRebaseRuns g' g) (jj : Fin 24) (a : Fin 28) :
    Rel sm (digitKey model key Q7 14 (rhsOwner 48 0) ((jj : Nat) : Int) ((a : Nat) : Int))
      (fun r ↦ (columnDigits Q7 (14 + 14) Backend.backend 1048576 14 524288
          (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) a 0 *
        reducePoly Q7 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 24 * a, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 24 * a, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53, h54,
    h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104,
    h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136,
    h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w2, w5, w9, w12, w13, w16, w17, w18, w19, w20, w21, w24, w27, w29, w31, w36, -, -, -, x2,
    -, -, -, x5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x9, x12, -, -, -, -, -, -, -, -, x13,
    x16, -, -, -, -, -, -, -, -, x17, x18, -, -, -, -, -, -, -, -, -, -, -, -, x19, x20, x21, x24,
    x27, -, -, -, x29, -, -, -, x31, x36, xout⟩ := hpar75 ⟨jj + 24 * a, by omega⟩
  have hgg' := gadgetMatrixRuns_unique h74 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ7 hN _)
  subst hgg
  have hcol : Int.fmod (Int.ofNat (jj + 24 * a)) 24 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (jj + 24 * a)) 24) 28 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 14 = (((a : Nat) / 14 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 14 = (((a : Nat) % 14 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hcol] at x2 x5 x29 x31 x13 x17
  rw [hrow] at x2 x5 x29 x31 x9 x19
  rw [hdig] at x9
  rw [hent] at x19
  have hw9 := unitRow_window (c := (((a : Nat) % 14 : Nat) : Int)) (by omega) x9
  obtain ⟨_, hw13⟩ := sliceMatrix_column x13
  obtain ⟨_, hw17⟩ := sliceMatrix_column x17
  obtain ⟨_, hw19⟩ := sliceMatrix_column x19
  rw [concatColumns_entry x18, joinedEntry_single] at hw19
  simp only [Int.toNat_natCast, Fin.eta] at hw13 hw17 hw19
  have he : w19 0 0 = pairColumn (w12 0 jj) (w16 0 jj) ⟨(a : Nat) / 14, by omega⟩ 0 := by
    rw [hw19]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw13, hw17]
  have hw21 := columnDigits_entry (q := Q7) (K := 14) rfl ⟨a, a.isLt⟩ he x20 x21
    (columnDigits_runs (bound := 524288) hQ7 hQ7 hN hg' Backend.layout4 layout7 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x12, concatColumns_entry x16] at hw21
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x27
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x36
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw24 : ∀ c : Fin 14, w24 0 c = w21 ⟨(a : Nat) % 14, by omega⟩ 0 * w.w_78_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x24 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 14) ((a : Nat) % 14) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q7 14 (rhsOwner 48 0) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w2, w29], ![w5, w31]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x5, hashSample_eq x29, hashSample_eq x31]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw24,
      hw21]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`1` RHS: lane `jj + 20 a` encrypts row `a` of the digits of
column `jj` of the RHS ciphertext. -/
theorem encrypt_rhs_digits1 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q6 N 1 12} (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g')
    (hg : centeredRebaseRuns g' g) (jj : Fin 20) (a : Fin 24) :
    Rel sm (digitKey model key Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int) ((a : Nat) : Int))
      (fun r ↦ (columnDigits Q6 (12 + 12) Backend.backend 1048576 12 524288
          (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) a 0 *
        reducePoly Q6 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 20 * a, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 20 * a, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53, h54,
    h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104,
    h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136,
    h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w23, w26, w29, w31, w33, w38, x1,
    -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -,
    -, -, -, x15, x18, -, -, -, -, -, -, -, -, x19, x20, -, -, -, -, -, -, -, -, -, -, -, -, x21,
    x22, x23, x26, x29, -, -, -, x31, -, -, -, x33, x38, xout⟩ := hpar103 ⟨jj + 20 * a, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ6 hN x1
  have hone := modulusReduce_one hQ7 hQ6 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h102 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ6 hN _)
  subst hgg
  have hcol : Int.fmod (Int.ofNat (jj + 20 * a)) 20 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (jj + 20 * a)) 20) 24 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 12 = (((a : Nat) / 12 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 12 = (((a : Nat) % 12 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hcol] at x3 x7 x31 x33 x15 x19
  rw [hrow] at x3 x7 x31 x33 x11 x21
  rw [hdig] at x11
  rw [hent] at x21
  have hw9 := unitRow_window (c := (((a : Nat) % 12 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  obtain ⟨_, hw19⟩ := sliceMatrix_column x21
  rw [concatColumns_entry x20, joinedEntry_single] at hw19
  simp only [Int.toNat_natCast, Fin.eta] at hw13 hw17 hw19
  have he : w21 0 0 = pairColumn (w14 0 jj) (w18 0 jj) ⟨(a : Nat) / 12, by omega⟩ 0 := by
    rw [hw19]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw13, hw17]
  have hw21 := columnDigits_entry (q := Q6) (K := 12) rfl ⟨a, a.isLt⟩ he x22 x23
    (columnDigits_runs (bound := 524288) hQ6 hQ6 hN hg' Backend.layout3 layout6 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x14, concatColumns_entry x18] at hw21
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x29
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x38
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw24 : ∀ c : Fin 12, w26 0 c = w23 ⟨(a : Nat) % 12, by omega⟩ 0 * w.w_106_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x26 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 12) ((a : Nat) % 12) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w31], ![w7, w33]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x31, hashSample_eq x33]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw24,
      hw21]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`2` RHS: lane `jj + 16 a` encrypts row `a` of the digits of
column `jj` of the RHS ciphertext. -/
theorem encrypt_rhs_digits2 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q5 N 1 10} (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g')
    (hg : centeredRebaseRuns g' g) (jj : Fin 16) (a : Fin 20) :
    Rel sm (digitKey model key Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int) ((a : Nat) : Int))
      (fun r ↦ (columnDigits Q5 (10 + 10) Backend.backend 1048576 10 524288
          (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) a 0 *
        reducePoly Q5 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 16 * a, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 16 * a, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53, h54,
    h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104,
    h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136,
    h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w23, w26, w29, w31, w33, w38, x1,
    -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -,
    -, -, -, x15, x18, -, -, -, -, -, -, -, -, x19, x20, -, -, -, -, -, -, -, -, -, -, -, -, x21,
    x22, x23, x26, x29, -, -, -, x31, -, -, -, x33, x38, xout⟩ := hpar131 ⟨jj + 16 * a, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ5 hN x1
  have hone := modulusReduce_one hQ7 hQ5 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h130 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ5 hN _)
  subst hgg
  have hcol : Int.fmod (Int.ofNat (jj + 16 * a)) 16 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (jj + 16 * a)) 16) 20 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 10 = (((a : Nat) / 10 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 10 = (((a : Nat) % 10 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hcol] at x3 x7 x31 x33 x15 x19
  rw [hrow] at x3 x7 x31 x33 x11 x21
  rw [hdig] at x11
  rw [hent] at x21
  have hw9 := unitRow_window (c := (((a : Nat) % 10 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  obtain ⟨_, hw19⟩ := sliceMatrix_column x21
  rw [concatColumns_entry x20, joinedEntry_single] at hw19
  simp only [Int.toNat_natCast, Fin.eta] at hw13 hw17 hw19
  have he : w21 0 0 = pairColumn (w14 0 jj) (w18 0 jj) ⟨(a : Nat) / 10, by omega⟩ 0 := by
    rw [hw19]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw13, hw17]
  have hw21 := columnDigits_entry (q := Q5) (K := 10) rfl ⟨a, a.isLt⟩ he x22 x23
    (columnDigits_runs (bound := 524288) hQ5 hQ5 hN hg' Backend.layout2 layout5 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x14, concatColumns_entry x18] at hw21
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x29
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x38
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw24 : ∀ c : Fin 10, w26 0 c = w23 ⟨(a : Nat) % 10, by omega⟩ 0 * w.w_134_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x26 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 10) ((a : Nat) % 10) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w31], ![w7, w33]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x31, hashSample_eq x33]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw24,
      hw21]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`3` RHS: lane `jj + 12 a` encrypts row `a` of the digits of
column `jj` of the RHS ciphertext. -/
theorem encrypt_rhs_digits3 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q4 N 1 8} (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g')
    (hg : centeredRebaseRuns g' g) (jj : Fin 12) (a : Fin 16) :
    Rel sm (digitKey model key Q4 8 (rhsOwner 51 6) ((jj : Nat) : Int) ((a : Nat) : Int))
      (fun r ↦ (columnDigits Q4 (8 + 8) Backend.backend 1048576 8 524288
          (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] jj) a 0 *
        reducePoly Q4 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 12 * a, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨jj + 12 * a, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53, h54,
    h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104,
    h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136,
    h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w23, w26, w29, w31, w33, w38, x1,
    -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -,
    -, -, -, x15, x18, -, -, -, -, -, -, -, -, x19, x20, -, -, -, -, -, -, -, -, -, -, -, -, x21,
    x22, x23, x26, x29, -, -, -, x31, -, -, -, x33, x38, xout⟩ := hpar159 ⟨jj + 12 * a, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ4 hN x1
  have hone := modulusReduce_one hQ7 hQ4 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h158 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ4 hN _)
  subst hgg
  have hcol : Int.fmod (Int.ofNat (jj + 12 * a)) 12 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (jj + 12 * a)) 12) 16 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 8 = (((a : Nat) / 8 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 8 = (((a : Nat) % 8 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hcol] at x3 x7 x31 x33 x15 x19
  rw [hrow] at x3 x7 x31 x33 x11 x21
  rw [hdig] at x11
  rw [hent] at x21
  have hw9 := unitRow_window (c := (((a : Nat) % 8 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  obtain ⟨_, hw19⟩ := sliceMatrix_column x21
  rw [concatColumns_entry x20, joinedEntry_single] at hw19
  simp only [Int.toNat_natCast, Fin.eta] at hw13 hw17 hw19
  have he : w21 0 0 = pairColumn (w14 0 jj) (w18 0 jj) ⟨(a : Nat) / 8, by omega⟩ 0 := by
    rw [hw19]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw13, hw17]
  have hw21 := columnDigits_entry (q := Q4) (K := 8) rfl ⟨a, a.isLt⟩ he x22 x23
    (columnDigits_runs (bound := 524288) hQ4 hQ4 hN hg' Backend.layout1 layout4 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x14, concatColumns_entry x18] at hw21
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x29
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x38
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw24 : ∀ c : Fin 8, w26 0 c = w23 ⟨(a : Nat) % 8, by omega⟩ 0 * w.w_162_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x26 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 8) ((a : Nat) % 8) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q4 8 (rhsOwner 51 6) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w31], ![w7, w33]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x31, hashSample_eq x33]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw24,
      hw21]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`0` LUT: lane `24 (24 s + a) + jj` encrypts `σ_s⁻¹` of row
`a` of the digits of column `jj` of the switching ciphertext of branch `s`. -/
theorem encrypt_lut_digits0 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q7 N 1 14} (hg' : gadgetMatrixRuns Backend.backend 1048576 14 g')
    (hg : centeredRebaseRuns g' g) (s : Fin 16) (jj : Fin 24) (a : Fin 24) :
    Rel sm (digitKey model key Q7 14 (lutOwner 48 ((s : Nat) : Int) 1) ((jj : Nat) : Int)
        ((a : Nat) : Int))
      (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
          (columnDigits Q7 (12 + 12) Backend.backend 1048576 12 524288
            (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) a 0) *
        reducePoly Q7 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨24 * (24 * s + a) + jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨24 * (24 * s + a) + jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, hmem, h40, -, hpar41, -, hpar42, -, -,
    h43, -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53,
    h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103,
    h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135,
    h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w2, w5, w9, w12, w14, w15, w16, w18, w20, w21, w22, w23, w24, w25, w26, w28, w30, w33,
    w35, w37, w42, -, -, -, -, x2, -, -, -, -, x5, -, -, -, -, -, -, -, -, -, -, -, -, -, -, x9, -,
    -, x12, -, -, x14, x15, -, -, -, -, -, -, -, -, x16, -, -, x18, -, -, x20, x21, -, -, -, -, -,
    -, -, -, x22, x23, -, -, -, -, -, -, -, -, -, -, -, -, x24, x25, x26, -, -, -, x28, x30, x33, -,
    -, -, -, x35, -, -, -, -, x37, x42, xout⟩ := hpar41 ⟨24 * (24 * s + a) + jj, by omega⟩
  obtain ⟨-, hm⟩ := hmem ⟨24 * (24 * s + a) + jj, by omega⟩
  have hgg' := gadgetMatrixRuns_unique h40 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ7 hN _)
  subst hgg
  have hmem' : Int.fdiv (Int.ofNat (24 * (24 * s + a) + jj)) 576 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.ofNat (24 * (24 * s + a) + jj)) 24 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (24 * (24 * s + a) + jj)) 24) 24 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 12 = (((a : Nat) / 12 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 12 = (((a : Nat) % 12 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hmem', add_zero] at hm
  rw [hm] at x12 x14 x18 x20
  rw [hmem', hcol, hrow] at x2 x5 x35 x37
  rw [hmem'] at x28
  rw [hcol] at x16 x22
  rw [hrow] at x9 x24
  rw [hdig] at x9
  rw [hent] at x24
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x12
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x18
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x20
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := (((a : Nat) % 12 : Nat) : Int)) (by omega) x9
  obtain ⟨_, hw16⟩ := sliceMatrix_column x16
  obtain ⟨_, hw22⟩ := sliceMatrix_column x22
  obtain ⟨_, hw24⟩ := sliceMatrix_column x24
  rw [concatColumns_entry x23, joinedEntry_single] at hw24
  simp only [Int.toNat_natCast, Fin.eta] at hw16 hw22 hw24
  have he : w24 0 0 = pairColumn (w15 0 jj) (w21 0 jj) ⟨(a : Nat) / 12, by omega⟩ 0 := by
    rw [hw24]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw16, hw22]
  obtain ⟨g6, hg6⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  have hw26 := columnDigits_entry (q := Q7) (K := 12) rfl ⟨a, a.isLt⟩ he x25 x26
    (columnDigits_runs (bound := 524288) hQ7 hQ6 hN hg6 Backend.layout3 layout6 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x15, concatColumns_entry x21] at hw26
  obtain ⟨_, hw28⟩ := ringAutomorphismRuns_eq hQ7 hN x28
  subst hw28
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x33
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x42
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw30 : ∀ c : Fin 14, w30 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w26 ⟨(a : Nat) % 12, by omega⟩ 0) * w.w_41_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x30 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 12) ((a : Nat) % 12) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q7 14 (lutOwner 48 ((s : Nat) : Int) 1)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w2, w35], ![w5, w37]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x5, hashSample_eq x35, hashSample_eq x37]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw30,
      hw26]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`1` LUT: lane `20 (20 s + a) + jj` encrypts `σ_s⁻¹` of row
`a` of the digits of column `jj` of the switching ciphertext of branch `s`. -/
theorem encrypt_lut_digits1 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q6 N 1 12} (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g')
    (hg : centeredRebaseRuns g' g) (s : Fin 16) (jj : Fin 20) (a : Fin 20) :
    Rel sm (digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3) ((jj : Nat) : Int)
        ((a : Nat) : Int))
      (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
          (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
            (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) a 0) *
        reducePoly Q6 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨20 * (20 * s + a) + jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨20 * (20 * s + a) + jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, hmem,
    h43, -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53,
    h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103,
    h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135,
    h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w28, w30, w32,
    w35, w37, w39, w44, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, x11, -, -, x14, -, -, x16, x17, -, -, -, -, -, -, -, -, x18, -, -, x20, -, -, x22, x23, -,
    -, -, -, -, -, -, -, x24, x25, -, -, -, -, -, -, -, -, -, -, -, -, x26, x27, x28, -, -, -, x30,
    x32, x35, -, -, -, -, x37, -, -, -, -, x39, x44, xout⟩ := hpar44 ⟨20 * (20 * s + a) + jj, by omega⟩
  obtain ⟨-, hm⟩ := hmem ⟨20 * (20 * s + a) + jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ6 hN x1
  have hone := modulusReduce_one hQ7 hQ6 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h43 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ6 hN _)
  subst hgg
  have hmem' : Int.fdiv (Int.ofNat (20 * (20 * s + a) + jj)) 400 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.ofNat (20 * (20 * s + a) + jj)) 20 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (20 * (20 * s + a) + jj)) 20) 20 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 10 = (((a : Nat) / 10 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 10 = (((a : Nat) % 10 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hmem', add_zero] at hm
  rw [hm] at x14 x16 x20 x22
  rw [hmem', hcol, hrow] at x3 x7 x37 x39
  rw [hmem'] at x30
  rw [hcol] at x18 x24
  rw [hrow] at x11 x26
  rw [hdig] at x11
  rw [hent] at x26
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := (((a : Nat) % 10 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw18⟩ := sliceMatrix_column x18
  obtain ⟨_, hw24⟩ := sliceMatrix_column x24
  obtain ⟨_, hw26⟩ := sliceMatrix_column x26
  rw [concatColumns_entry x25, joinedEntry_single] at hw26
  simp only [Int.toNat_natCast, Fin.eta] at hw18 hw24 hw26
  have he : w26 0 0 = pairColumn (w17 0 jj) (w23 0 jj) ⟨(a : Nat) / 10, by omega⟩ 0 := by
    rw [hw26]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw18, hw24]
  obtain ⟨gd, hgd⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  have hw28 := columnDigits_entry (q := Q6) (K := 10) rfl ⟨a, a.isLt⟩ he x27 x28
    (columnDigits_runs (bound := 524288) hQ6 hQ5 hN hgd Backend.layout2 layout5 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x17, concatColumns_entry x23] at hw28
  obtain ⟨_, hw30⟩ := ringAutomorphismRuns_eq hQ6 hN x30
  subst hw30
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x35
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x44
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw32 : ∀ c : Fin 12, w32 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w28 ⟨(a : Nat) % 10, by omega⟩ 0) * w.w_45_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x32 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 10) ((a : Nat) % 10) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w37], ![w7, w39]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x37, hashSample_eq x39]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw32,
      hw28]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`2` LUT: lane `16 (16 s + a) + jj` encrypts `σ_s⁻¹` of row
`a` of the digits of column `jj` of the switching ciphertext of branch `s`. -/
theorem encrypt_lut_digits2 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q5 N 1 10} (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g')
    (hg : centeredRebaseRuns g' g) (s : Fin 16) (jj : Fin 16) (a : Fin 16) :
    Rel sm (digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5) ((jj : Nat) : Int)
        ((a : Nat) : Int))
      (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
          (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
            (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) a 0) *
        reducePoly Q5 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨16 * (16 * s + a) + jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨16 * (16 * s + a) + jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, hmem, h46, -, hpar47, -, hpar48, -, -, h49, -, hpar50, h51, h52, h53,
    h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103,
    h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135,
    h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w28, w30, w32,
    w35, w37, w39, w44, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, x11, -, -, x14, -, -, x16, x17, -, -, -, -, -, -, -, -, x18, -, -, x20, -, -, x22, x23, -,
    -, -, -, -, -, -, -, x24, x25, -, -, -, -, -, -, -, -, -, -, -, -, x26, x27, x28, -, -, -, x30,
    x32, x35, -, -, -, -, x37, -, -, -, -, x39, x44, xout⟩ := hpar47 ⟨16 * (16 * s + a) + jj, by omega⟩
  obtain ⟨-, hm⟩ := hmem ⟨16 * (16 * s + a) + jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ5 hN x1
  have hone := modulusReduce_one hQ7 hQ5 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h46 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ5 hN _)
  subst hgg
  have hmem' : Int.fdiv (Int.ofNat (16 * (16 * s + a) + jj)) 256 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.ofNat (16 * (16 * s + a) + jj)) 16 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (16 * (16 * s + a) + jj)) 16) 16 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 8 = (((a : Nat) / 8 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 8 = (((a : Nat) % 8 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hmem', add_zero] at hm
  rw [hm] at x14 x16 x20 x22
  rw [hmem', hcol, hrow] at x3 x7 x37 x39
  rw [hmem'] at x30
  rw [hcol] at x18 x24
  rw [hrow] at x11 x26
  rw [hdig] at x11
  rw [hent] at x26
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := (((a : Nat) % 8 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw18⟩ := sliceMatrix_column x18
  obtain ⟨_, hw24⟩ := sliceMatrix_column x24
  obtain ⟨_, hw26⟩ := sliceMatrix_column x26
  rw [concatColumns_entry x25, joinedEntry_single] at hw26
  simp only [Int.toNat_natCast, Fin.eta] at hw18 hw24 hw26
  have he : w26 0 0 = pairColumn (w17 0 jj) (w23 0 jj) ⟨(a : Nat) / 8, by omega⟩ 0 := by
    rw [hw26]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw18, hw24]
  obtain ⟨gd, hgd⟩ := gadget_exists (m := 8) layout4 rfl rfl rfl
  have hw28 := columnDigits_entry (q := Q5) (K := 8) rfl ⟨a, a.isLt⟩ he x27 x28
    (columnDigits_runs (bound := 524288) hQ5 hQ4 hN hgd Backend.layout1 layout4 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x17, concatColumns_entry x23] at hw28
  obtain ⟨_, hw30⟩ := ringAutomorphismRuns_eq hQ5 hN x30
  subst hw30
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x35
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x44
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw32 : ∀ c : Fin 10, w32 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w28 ⟨(a : Nat) % 8, by omega⟩ 0) * w.w_49_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x32 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 8) ((a : Nat) % 8) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w37], ![w7, w39]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x37, hashSample_eq x39]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw32,
      hw28]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`3` LUT: lane `2 (12 s + a) + jj` encrypts `σ_s⁻¹` of row
`a` of the digits of column `jj` of the switching ciphertext of branch `s`. -/
theorem encrypt_lut_digits3 {model : HashModel} {key : ByteArray} {secret : ExactMatrix Q7 N 1 1}
    {c e c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_encrypt.generatedRoot Backend.backend model GeneratedClaim.stage_1_params
      (secret, key, c, e, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sm : ErrorPoly N} (hsecret : secret = fun _ _ ↦ reducePoly Q7 N sm)
    {g' g : ExactMatrix Q4 N 1 8} (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g')
    (hg : centeredRebaseRuns g' g) (s : Fin 16) (jj : Fin 2) (a : Fin 12) :
    Rel sm (digitKey model key Q4 8 (lutOwner 51 ((s : Nat) : Int) 7) ((jj : Nat) : Int)
        ((a : Nat) : Int))
      (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
          (columnDigits Q4 (6 + 6) Backend.backend 1048576 6 524288
            (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) (joinedEntry ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 s] jj) a 0) *
        reducePoly Q4 N (unitPair sm r)) • g)
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨2 * (12 * s + a) + jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨2 * (12 * s + a) + jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -,
    hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, -, -, h40, -, hpar41, -, hpar42, -, -, h43,
    -, hpar44, -, hpar45, -, -, h46, -, hpar47, -, hpar48, -, hmem, h49, -, hpar50, h51, h52, h53,
    h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103,
    h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135,
    h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w28, w30, w32,
    w35, w37, w39, w44, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, x11, -, -, x14, -, -, x16, x17, -, -, -, -, -, -, -, -, x18, -, -, x20, -, -, x22, x23, -,
    -, -, -, -, -, -, -, x24, x25, -, -, -, -, -, -, -, -, -, -, -, -, x26, x27, x28, -, -, -, x30,
    x32, x35, -, -, -, -, x37, -, -, -, -, x39, x44, xout⟩ := hpar50 ⟨2 * (12 * s + a) + jj, by omega⟩
  obtain ⟨-, hm⟩ := hmem ⟨2 * (12 * s + a) + jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ4 hN x1
  have hone := modulusReduce_one hQ7 hQ4 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h49 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ4 hN _)
  subst hgg
  have hmem' : Int.fdiv (Int.ofNat (2 * (12 * s + a) + jj)) 24 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.ofNat (2 * (12 * s + a) + jj)) 2 = ((jj : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.fdiv (Int.ofNat (2 * (12 * s + a) + jj)) 2) 12 = ((a : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hent : Int.fdiv ((a : Nat) : Int) 6 = (((a : Nat) / 6 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    omega
  have hdig : Int.fmod ((a : Nat) : Int) 6 = (((a : Nat) % 6 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    omega
  rw [hmem', add_zero] at hm
  rw [hm] at x14 x16 x20 x22
  rw [hmem', hcol, hrow] at x3 x7 x37 x39
  rw [hmem'] at x30
  rw [hcol] at x18 x24
  rw [hrow] at x11 x26
  rw [hdig] at x11
  rw [hent] at x26
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := (((a : Nat) % 6 : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw18⟩ := sliceMatrix_column x18
  obtain ⟨_, hw24⟩ := sliceMatrix_column x24
  obtain ⟨_, hw26⟩ := sliceMatrix_column x26
  rw [concatColumns_entry x25, joinedEntry_single] at hw26
  simp only [Int.toNat_natCast, Fin.eta] at hw18 hw24 hw26
  have he : w26 0 0 = pairColumn (w17 0 jj) (w23 0 jj) ⟨(a : Nat) / 6, by omega⟩ 0 := by
    rw [hw26]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, hw18, hw24]
  obtain ⟨gd, hgd⟩ := gadget_exists (m := 6) layout3 rfl rfl rfl
  have hw28 := columnDigits_entry (q := Q4) (K := 6) rfl ⟨a, a.isLt⟩ he x27 x28
    (columnDigits_runs (bound := 524288) hQ4 hQ3 hN hgd Backend.layout0 layout3 rfl rfl (by norm_num)
      (by norm_num) _ _)
  rw [concatColumns_entry x17, concatColumns_entry x23] at hw28
  obtain ⟨_, hw30⟩ := ringAutomorphismRuns_eq hQ4 hN x30
  subst hw30
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x35
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x44
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw32 : ∀ c : Fin 8, w32 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w28 ⟨(a : Nat) % 6, by omega⟩ 0) * w.w_53_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x32 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 6) ((a : Nat) % 6) (by omega)] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q4 8 (lutOwner 51 ((s : Nat) : Int) 7)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w37], ![w7, w39]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x37, hashSample_eq x39]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw32,
      hw28]; try ring)

end NandProof
