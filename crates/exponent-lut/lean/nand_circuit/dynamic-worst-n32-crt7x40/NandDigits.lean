import NandSetup

/-!
# Digit companions

Dynamic encryption stores, for every digit `d` of the decomposition of a ciphertext column and every
row of the decomposition of the state key, a companion `z = s B - d (s, 1) G + e` under the digit
key `B`. The RHS companions of a round encrypt the digits of the RHS ciphertext; the LUT
companions of branch `σ` encrypt `σ⁻¹` of the digits of the switching ciphertext of `σ`, so that
`σ(z)` is a companion of the digit itself under the substituted secret.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`0` RHS: lane `a + 28 jj` encrypts row `a` of the digits of
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 28 * jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 28 * jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w2, w5, w9, w12, w13, w16, w17, w18, w19, w20, w23, w26, w28, w30, w35, -, -, -, x2, -, -, -, x5, -, -, -, -, -, -, -, -, -, -, x9, x12, -, -, -, -, -, -, -, -, -, -, x13, x16, -, -, -, -, -, -, -, -, -, -, x17, x18, x19, x20, x23, x26, -, -, -, x28, -, -, -, x30, x35, xout⟩ := hpar75 ⟨a + 28 * jj, by omega⟩
  have hgg' := gadgetMatrixRuns_unique h74 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ7 hN _)
  subst hgg
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 28 * jj)) 28) 24 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 28 * jj)) 28 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hcol] at x2 x5 x28 x30 x13 x17
  rw [hrow] at x2 x5 x28 x30 x9
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x9
  obtain ⟨_, hw13⟩ := sliceMatrix_column x13
  obtain ⟨_, hw17⟩ := sliceMatrix_column x17
  rw [concatColumns_entry x12] at hw13
  rw [concatColumns_entry x16] at hw17
  rw [concatRows_pairColumn x18, hw13, hw17] at x19
  simp only [Int.toNat_natCast, Fin.eta] at x19
  have hw20 := columnDigits_eq x19 x20
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x26
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x35
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 14, w23 0 c = w20 a 0 * w.w_74_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x23 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 14 + 14) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q7 14 (rhsOwner 48 0) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w2, w28], ![w5, w30]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x5, hashSample_eq x28, hashSample_eq x30]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`1` RHS: lane `a + 24 jj` encrypts row `a` of the digits of
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 24 * jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 24 * jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w25, w28, w30, w32, w37, x1, -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -, -, -, -, -, -, x15, x18, -, -, -, -, -, -, -, -, -, -, x19, x20, x21, x22, x25, x28, -, -, -, x30, -, -, -, x32, x37, xout⟩ := hpar103 ⟨a + 24 * jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ6 hN x1
  have hone := modulusReduce_one hQ7 hQ6 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h102 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ6 hN _)
  subst hgg
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 24 * jj)) 24) 20 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 24 * jj)) 24 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hcol] at x3 x7 x30 x32 x15 x19
  rw [hrow] at x3 x7 x30 x32 x11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  rw [concatColumns_entry x14] at hw13
  rw [concatColumns_entry x18] at hw17
  rw [concatRows_pairColumn x20, hw13, hw17] at x21
  simp only [Int.toNat_natCast, Fin.eta] at x21
  have hw20 := columnDigits_eq x21 x22
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x28
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x37
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 12, w25 0 c = w22 a 0 * w.w_102_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x25 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 12 + 12) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w30], ![w7, w32]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x30, hashSample_eq x32]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`2` RHS: lane `a + 20 jj` encrypts row `a` of the digits of
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 20 * jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 20 * jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w25, w28, w30, w32, w37, x1, -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -, -, -, -, -, -, x15, x18, -, -, -, -, -, -, -, -, -, -, x19, x20, x21, x22, x25, x28, -, -, -, x30, -, -, -, x32, x37, xout⟩ := hpar131 ⟨a + 20 * jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ5 hN x1
  have hone := modulusReduce_one hQ7 hQ5 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h130 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ5 hN _)
  subst hgg
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 20 * jj)) 20) 16 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 20 * jj)) 20 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hcol] at x3 x7 x30 x32 x15 x19
  rw [hrow] at x3 x7 x30 x32 x11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  rw [concatColumns_entry x14] at hw13
  rw [concatColumns_entry x18] at hw17
  rw [concatRows_pairColumn x20, hw13, hw17] at x21
  simp only [Int.toNat_natCast, Fin.eta] at x21
  have hw20 := columnDigits_eq x21 x22
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x28
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x37
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 10, w25 0 c = w22 a 0 * w.w_130_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x25 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 10 + 10) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w30], ![w7, w32]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x30, hashSample_eq x32]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`3` RHS: lane `a + 16 jj` encrypts row `a` of the digits of
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 16 * jj, by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 16 * jj, by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w15, w18, w19, w20, w21, w22, w25, w28, w30, w32, w37, x1, -, -, -, x3, x6, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, x14, -, -, -, -, -, -, -, -, -, -, x15, x18, -, -, -, -, -, -, -, -, -, -, x19, x20, x21, x22, x25, x28, -, -, -, x30, -, -, -, x32, x37, xout⟩ := hpar159 ⟨a + 16 * jj, by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ4 hN x1
  have hone := modulusReduce_one hQ7 hQ4 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h158 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ4 hN _)
  subst hgg
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 16 * jj)) 16) 12 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 16 * jj)) 16 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hcol] at x3 x7 x30 x32 x15 x19
  rw [hrow] at x3 x7 x30 x32 x11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x15
  obtain ⟨_, hw17⟩ := sliceMatrix_column x19
  rw [concatColumns_entry x14] at hw13
  rw [concatColumns_entry x18] at hw17
  rw [concatRows_pairColumn x20, hw13, hw17] at x21
  simp only [Int.toNat_natCast, Fin.eta] at x21
  have hw20 := columnDigits_eq x21 x22
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x28
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x37
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 8, w25 0 c = w22 a 0 * w.w_158_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x25 c.val c.isLt
    rw [matrixMul, hw9, unitRow_mul_entry (k := 8 + 8) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q4 8 (rhsOwner 51 6) ((jj : Nat) : Int)
      ((a : Nat) : Int) k j = ![![w3, w30], ![w7, w32]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x30, hashSample_eq x32]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`0` LUT: lane `a + 24 (24 s + jj)` encrypts `σ_s⁻¹` of row
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 24 * (24 * s + jj), by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 24 * (24 * s + jj), by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w2, w5, w9, w12, w14, w15, w16, w18, w20, w21, w22, w23, w24, w25, w27, w29, w32, w34, w36, w41, -, -, -, -, x2, -, -, -, -, x5, -, -, -, -, -, -, -, -, -, -, x9, -, -, x11, x12, -, x11'1, x14, x15, -, -, -, -, -, -, -, -, -, -, x16, -, x11'2, x18, -, x11'3, x20, x21, -, -, -, -, -, -, -, -, -, -, x22, x23, x24, x25, -, -, -, x27, x29, x32, -, -, -, -, x34, -, -, -, -, x36, x41, xout⟩ := hpar41 ⟨a + 24 * (24 * s + jj), by omega⟩
  have hgg' := gadgetMatrixRuns_unique h40 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ7 hN _)
  subst hgg
  have hmem : Int.fdiv (Int.ofNat (a + 24 * (24 * s + jj))) 576 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 24 * (24 * s + jj))) 24) 24 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 24 * (24 * s + jj))) 24 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hmem, hcol, hrow] at x2 x5 x34 x36
  rw [hmem] at x12 x14 x18 x20 x27
  rw [hcol] at x16 x22
  rw [hrow] at x9
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x12
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x18
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x20
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x9
  obtain ⟨_, hw13⟩ := sliceMatrix_column x16
  obtain ⟨_, hw17⟩ := sliceMatrix_column x22
  rw [concatColumns_entry x15] at hw13
  rw [concatColumns_entry x21] at hw17
  rw [concatRows_pairColumn x23, hw13, hw17] at x24
  simp only [Int.toNat_natCast, Fin.eta] at x24
  have hw20 := columnDigits_eq x24 x25
  obtain ⟨_, hw27⟩ := ringAutomorphismRuns_eq hQ7 hN x27
  subst hw27
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x32
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x41
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 14, w29 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w25 a 0) * w.w_40_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x29 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 12 + 12) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q7 14 (lutOwner 48 ((s : Nat) : Int) 1)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w2, w34], ![w5, w36]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x5, hashSample_eq x34, hashSample_eq x36]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`1` LUT: lane `a + 20 (20 s + jj)` encrypts `σ_s⁻¹` of row
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 20 * (20 * s + jj), by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 20 * (20 * s + jj), by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w29, w31, w34, w36, w38, w43, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, -, -, x13, x14, -, x13'1, x16, x17, -, -, -, -, -, -, -, -, -, -, x18, -, x13'2, x20, -, x13'3, x22, x23, -, -, -, -, -, -, -, -, -, -, x24, x25, x26, x27, -, -, -, x29, x31, x34, -, -, -, -, x36, -, -, -, -, x38, x43, xout⟩ := hpar44 ⟨a + 20 * (20 * s + jj), by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ6 hN x1
  have hone := modulusReduce_one hQ7 hQ6 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h43 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ6 hN _)
  subst hgg
  have hmem : Int.fdiv (Int.ofNat (a + 20 * (20 * s + jj))) 400 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 20 * (20 * s + jj))) 20) 20 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 20 * (20 * s + jj))) 20 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hmem, hcol, hrow] at x3 x7 x36 x38
  rw [hmem] at x14 x16 x20 x22 x29
  rw [hcol] at x18 x24
  rw [hrow] at x11
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x18
  obtain ⟨_, hw17⟩ := sliceMatrix_column x24
  rw [concatColumns_entry x17] at hw13
  rw [concatColumns_entry x23] at hw17
  rw [concatRows_pairColumn x25, hw13, hw17] at x26
  simp only [Int.toNat_natCast, Fin.eta] at x26
  have hw20 := columnDigits_eq x26 x27
  obtain ⟨_, hw27⟩ := ringAutomorphismRuns_eq hQ6 hN x29
  subst hw27
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x34
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x43
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 12, w31 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w27 a 0) * w.w_43_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x31 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 10 + 10) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w36], ![w7, w38]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x36, hashSample_eq x38]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`2` LUT: lane `a + 16 (16 s + jj)` encrypts `σ_s⁻¹` of row
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 16 * (16 * s + jj), by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 16 * (16 * s + jj), by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w29, w31, w34, w36, w38, w43, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, -, -, x13, x14, -, x13'1, x16, x17, -, -, -, -, -, -, -, -, -, -, x18, -, x13'2, x20, -, x13'3, x22, x23, -, -, -, -, -, -, -, -, -, -, x24, x25, x26, x27, -, -, -, x29, x31, x34, -, -, -, -, x36, -, -, -, -, x38, x43, xout⟩ := hpar47 ⟨a + 16 * (16 * s + jj), by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ5 hN x1
  have hone := modulusReduce_one hQ7 hQ5 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h46 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ5 hN _)
  subst hgg
  have hmem : Int.fdiv (Int.ofNat (a + 16 * (16 * s + jj))) 256 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 16 * (16 * s + jj))) 16) 16 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 16 * (16 * s + jj))) 16 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hmem, hcol, hrow] at x3 x7 x36 x38
  rw [hmem] at x14 x16 x20 x22 x29
  rw [hcol] at x18 x24
  rw [hrow] at x11
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x18
  obtain ⟨_, hw17⟩ := sliceMatrix_column x24
  rw [concatColumns_entry x17] at hw13
  rw [concatColumns_entry x23] at hw17
  rw [concatRows_pairColumn x25, hw13, hw17] at x26
  simp only [Int.toNat_natCast, Fin.eta] at x26
  have hw20 := columnDigits_eq x26 x27
  obtain ⟨_, hw27⟩ := ringAutomorphismRuns_eq hQ5 hN x29
  subst hw27
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x34
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x43
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 10, w31 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w27 a 0) * w.w_46_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x31 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 8 + 8) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w36], ![w7, w38]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x36, hashSample_eq x38]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

set_option maxHeartbeats 16000000 in
/-- The digit companions of the round-`3` LUT: lane `a + 12 (2 s + jj)` encrypts `σ_s⁻¹` of row
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
      ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 12 * (2 * s + jj), by omega⟩, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ⟨a + 12 * (2 * s + jj), by omega⟩] 26 := by
  obtain ⟨w, h2, h5, h8, h9, h19, h21, h23, h29, -, hpar31, -, hpar32, -, hpar33, -, hpar34, -, hpar35, -, hpar36, -, hpar37, -, hpar38, -, hpar39, h40, -, hpar41, -, hpar42, h43, -, hpar44, -, hpar45, h46, -, hpar47, -, hpar48, h49, -, hpar50, h51, h52, h53, h54, h65, h72, h74, -, hpar75, h76, h77, h78, h79, h80, h88, h92, h94, h100, h102, -, hpar103, h104, h105, h106, h107, h108, h116, h120, h122, h128, h130, -, hpar131, h132, h133, h134, h135, h136, h144, h148, h150, h156, h158, -, hpar159, hout⟩ := h
  subst hout hsecret
  obtain ⟨w1, w3, w6, w7, w11, w14, w16, w17, w18, w20, w22, w23, w24, w25, w26, w27, w29, w31, w34, w36, w38, w43, x1, -, -, -, -, x3, x6, -, -, -, -, x7, -, -, -, -, -, -, -, -, -, -, x11, -, -, x13, x14, -, x13'1, x16, x17, -, -, -, -, -, -, -, -, -, -, x18, -, x13'2, x20, -, x13'3, x22, x23, -, -, -, -, -, -, -, -, -, -, x24, x25, x26, x27, -, -, -, x29, x31, x34, -, -, -, -, x36, -, -, -, -, x38, x43, xout⟩ := hpar50 ⟨a + 12 * (2 * s + jj), by omega⟩
  have hsec := modulusReduce_poly hQ7 hQ4 hN x1
  have hone := modulusReduce_one hQ7 hQ4 hN x6
  subst hsec hone
  have hgg' := gadgetMatrixRuns_unique h49 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique hg (centeredRebaseRuns_self hQ4 hN _)
  subst hgg
  have hmem : Int.fdiv (Int.ofNat (a + 12 * (2 * s + jj))) 24 = ((s : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hcol : Int.fmod (Int.fdiv (Int.ofNat (a + 12 * (2 * s + jj))) 12) 2 = ((jj : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat (a + 12 * (2 * s + jj))) 12 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hmem, hcol, hrow] at x3 x7 x36 x38
  rw [hmem] at x14 x16 x20 x22 x29
  rw [hcol] at x18 x24
  rw [hrow] at x11
  obtain ⟨_, hf00⟩ := familyGet_eq (k := s) x14
  obtain ⟨_, hf01⟩ := familyGet_eq (k := s) x16
  obtain ⟨_, hf10⟩ := familyGet_eq (k := s) x20
  obtain ⟨_, hf11⟩ := familyGet_eq (k := s) x22
  simp only [Fin.eta] at hf00 hf01 hf10 hf11
  subst hf00 hf01 hf10 hf11
  have hw9 := unitRow_window (c := ((a : Nat) : Int)) (by omega) x11
  obtain ⟨_, hw13⟩ := sliceMatrix_column x18
  obtain ⟨_, hw17⟩ := sliceMatrix_column x24
  rw [concatColumns_entry x17] at hw13
  rw [concatColumns_entry x23] at hw17
  rw [concatRows_pairColumn x25, hw13, hw17] at x26
  simp only [Int.toNat_natCast, Fin.eta] at x26
  have hw20 := columnDigits_eq x26 x27
  obtain ⟨_, hw27⟩ := ringAutomorphismRuns_eq hQ4 hN x29
  subst hw27
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound x34
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound x43
  simp only [Prod.mk.injEq] at xout
  obtain ⟨ho0, ho1, -⟩ := xout
  have hw23 : ∀ c : Fin 8, w31 0 c = substitution (sigmaInv s).toNat
      (sigmaInv_odd s) (w27 a 0) * w.w_49_0 0 c := by
    intro c
    have h1 := tensorRuns_scalar x31 c.val c.isLt
    dsimp only at h1
    rw [matrixMul, hw9, unitRow_mul_entry (k := 6 + 6) a a.isLt] at h1
    exact h1
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have hkey : ∀ k j, digitKey model key Q4 8 (lutOwner 51 ((s : Nat) : Int) 7)
      ((jj : Nat) : Int) ((a : Nat) : Int) k j =
        ![![w3, w36], ![w7, w38]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x3,
        hashSample_eq x7, hashSample_eq x36, hashSample_eq x38]
  simp only [hkey]
  fin_cases j <;> funext r c' <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    (simp [ho0, ho1, matrixAdd, matrixSub, unitPair, he0, he1, reduceMatrix, Matrix.mul_apply, hw23,
      hw20]; try ring)

end NandProof
