import NandDigits
import NandRound0

/-!
# Online evaluation, round 1

Round `0` multiplies the level-`0` state encoding by the Dynamic RHS of circuit input `0` and
evaluates the first LUT. The generated loops compute, in order, the companion keys and row of the
RHS (`sequential_generatedRoot_128`), the companion keys and rows of the sixteen LUT branches
(`sequential_generatedRoot_135`), the ciphertext action `c Dec(C)` (`sequential_generatedRoot_146`),
and the sixteen branches (`sequential_generatedRoot_150`); the root moves the sum down to `Q5`.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The state of the RHS companion loop of round `1`. -/
abbrev RhsState1 := ExactMatrix Q6 N 1 20 × ExactMatrix Q6 N 1 20 × ExactMatrix Q6 N 1 20 ×
  ExactMatrix Q6 N 1 12 × ExactMatrix Q6 N 1 12 × Unit

set_option maxHeartbeats 8000000 in
/-- One iteration of the RHS companion loop of round `1` adds, at column `i / 24`, the companion
keys and the companion `i` paired with the digits of key entry `i % 24`. -/
theorem rhsLoop1_step {model : HashModel} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 12} {A0 A1 : ExactMatrix Q6 N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 480 → ExactMatrix Q6 N 1 12} {i : Nat} (hi : i < 480) {s t : RhsState1}
    (hs : s.2.2.2.1 = z0 ⟨i, hi⟩ ∧ s.2.2.2.2.1 = z1 ⟨i, hi⟩)
    (h : Stage_online_eval.sequential_generatedRoot_128 Backend.backend model
      GeneratedClaim.stage_2_params i
      (s.1, s.2.1, s.2.2.1, s.2.2.2.1, s.2.2.2.2.1, key, A0, A1, z0, z1, ()) t) :
    let B := digitKey model key Q6 12 (rhsOwner 49 2) ((i / 24 : Nat) : Int) ((i % 24 : Nat) : Int)
    let U := keyDigits 12 A ⟨i % 24, by omega⟩
    t.1 = s.1 + (B 0 0 * U 0 + B 0 1 * U 1) 0 0 • unitRow ((i / 24 : Nat) : Int) ∧
      t.2.1 = s.2.1 + (B 1 0 * U 0 + B 1 1 * U 1) 0 0 • unitRow ((i / 24 : Nat) : Int) ∧
      t.2.2.1 = s.2.2.1 +
        (z0 ⟨i, hi⟩ * U 0 + z1 ⟨i, hi⟩ * U 1) 0 0 • unitRow ((i / 24 : Nat) : Int) ∧
      t.2.2.2.1 = z0 ⟨(i + 1) % 480, Nat.mod_lt _ (by norm_num)⟩ ∧
      t.2.2.2.2.1 = z1 ⟨(i + 1) % 480, Nat.mod_lt _ (by norm_num)⟩ := by
  intro B U
  obtain ⟨w2, w4, w5, w7, w9, w10, w14, w18, w20, w35, w37, -, -, -, x2, -, -, -, -, -, -, -, -,
    x4, x5, -, -, -, x7, -, -, -, -, -, -, -, -, x9, x10, -, -, -, -, -, -, -, -, -, -, x14, -, -, -,
    x18, -, -, -, x20, -, -, -, -, -, -, -, x35, -, -, x37, xout⟩ := h
  have hcol : Int.fmod (Int.fdiv (Int.ofNat i) 24) 20 = ((i / 24 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hdiv : Int.fdiv (Int.ofNat i) 24 = ((i / 24 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 24 = ((i % 24 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hnext : 24 * Int.fmod (Int.fdiv (Int.fmod (1 + Int.ofNat i) 480) 24) 20 +
      Int.fmod (Int.fmod (1 + Int.ofNat i) 480) 24 = Int.ofNat ((i + 1) % 480) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 24 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 24 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 480 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hcol, hrow] at x2 x7 x18 x20
  rw [hrow] at x4 x9
  rw [hdiv] at x14
  rw [hnext] at x35 x37
  obtain ⟨_, hw35⟩ := familyGet_eq x35
  obtain ⟨_, hw37⟩ := familyGet_eq x37
  have hw14 := unitRow_window (c := ((i / 24 : Nat) : Int)) (by omega) x14
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q6 N 1 24} {x : ExactMatrix Q6 N 1 1}
      {u : ExactMatrix Q6 N 12 1}, concatColumns (A r 0) (A r 1) Ar →
      sliceMatrix Ar 0 1 ((i % 24 : Nat) : Int) (1 + ((i % 24 : Nat) : Int)) x →
      gadgetDecomposeRuns Backend.backend 1048576 12 x u → u = U r := by
    intro r Ar x u hAr hx hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    have hxe : x = fun _ _ ↦ joinedEntry (A r) ⟨i % 24, by omega⟩ := by
      funext a b
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0]
      congr 1
    rw [hxe] at hu
    exact entryDigits_eq hu
  have hw5 := hentry hA0 x4 x5
  have hw10 := hentry hA1 x9 x10
  have hB : B = ![![w2, w7], ![w18, w20]] := by
    funext k j
    fin_cases k <;> fin_cases j <;>
      simp [B, digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x7, hashSample_eq x18, hashSample_eq x20]
  subst xout hw5 hw10 hw14 hw35 hw37
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, ?_, ?_, rfl, rfl⟩ <;>
    simp only [hB, hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq, Matrix.cons_val_zero,
      Matrix.cons_val_one]

/-- The digit-row sum of the RHS companion keys `B_k0 u_0 + B_k1 u_1` of round `1`. -/
noncomputable def rhsKeySum1 (model : HashModel) (key : ByteArray)
    (A : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 12) (k : Fin 2) (jj : Fin 20) (a : Fin 24) :
    ExactPoly Q6 N :=
  (digitKey model key Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int) ((a : Nat) : Int) k 0 *
      keyDigits 12 A a 0 +
    digitKey model key Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int) ((a : Nat) : Int) k 1 *
      keyDigits 12 A a 1) 0 0

/-- The digit-row sum of the RHS companions `z_0 u_0 + z_1 u_1` of round `1`. -/
noncomputable def rhsCompanionSum1 (A : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 12)
    (z0 z1 : Fin 480 → ExactMatrix Q6 N 1 12) (jj : Fin 20) (a : Fin 24) : ExactPoly Q6 N :=
  (z0 ⟨a + 24 * jj, by omega⟩ * keyDigits 12 A a 0 + z1 ⟨a + 24 * jj, by omega⟩ * keyDigits 12 A a 1) 0 0

set_option maxHeartbeats 8000000 in
/-- The RHS companion loop of round `1` ends with the companion keys and the companion row, column by
column. -/
theorem rhsLoop1 {model : HashModel} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 12} {A0 A1 : ExactMatrix Q6 N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 480 → ExactMatrix Q6 N 1 12} {final : RhsState1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : RhsState1) ↦
      Stage_online_eval.sequential_generatedRoot_128 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (key, (A0,
          (A1, (z0, (z1, ())))))))))) next) (Int.toNat 480)
      (0, (0, (0, (z0 ⟨0, by norm_num⟩, (z1 ⟨0, by norm_num⟩, ()))))) final) (jj : Fin 20) :
    final.1 0 jj = ∑ a, rhsKeySum1 model key A 0 jj a ∧
      final.2.1 0 jj = ∑ a, rhsKeySum1 model key A 1 jj a ∧
      final.2.2.1 0 jj = ∑ a, rhsCompanionSum1 A z0 z1 jj a := by
  have hcount : Int.toNat 480 = 20 * 24 := rfl
  rw [hcount] at h
  let Q : Nat → RhsState1 → Prop := fun i s ↦
    s.2.2.2.1 = z0 ⟨i % 480, Nat.mod_lt _ (by norm_num)⟩ ∧
      s.2.2.2.2.1 = z1 ⟨i % 480, Nat.mod_lt _ (by norm_num)⟩
  have hQ0 : Q 0 (0, (0, (0, (z0 ⟨0, by norm_num⟩, (z1 ⟨0, by norm_num⟩, ()))))) := ⟨rfl, rfl⟩
  have hstep : ∀ i (s t : RhsState1), i < 20 * 24 → Q i s →
      Stage_online_eval.sequential_generatedRoot_128 Backend.backend model GeneratedClaim.stage_2_params i
        (s.1, (s.2.1, (s.2.2.1, (s.2.2.2.1, (s.2.2.2.2.1, (key, (A0, (A1, (z0, (z1, ())))))))))) t →
      (t.1 = s.1 + (if h : i < 20 * 24 then
          rhsKeySum1 model key A 0 ⟨i / 24, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 24, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 24 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0) ∧
        t.2.1 = s.2.1 + (if h : i < 20 * 24 then
          rhsKeySum1 model key A 1 ⟨i / 24, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 24, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 24 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0) ∧
        t.2.2.1 = s.2.2.1 + (if h : i < 20 * 24 then
          rhsCompanionSum1 A z0 z1 ⟨i / 24, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 24, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 24 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0)) ∧ Q (i + 1) t := by
    intro i s t hi hq hb
    have hi' : i < 480 := hi
    have hmod : i % 480 = i := Nat.mod_eq_of_lt hi'
    have hq' : s.2.2.2.1 = z0 ⟨i, hi'⟩ ∧ s.2.2.2.2.1 = z1 ⟨i, hi'⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext hmod)), h1.trans (congrArg z1 (Fin.ext hmod))⟩
    obtain ⟨h0, h1, h2, h3, h4⟩ := rhsLoop1_step hA0 hA1 hi' hq' hb
    have hz : (⟨i % 24 + 24 * (i / 24), by omega⟩ : Fin 480) = ⟨i, hi'⟩ :=
      Fin.ext (Nat.mod_add_div i 24)
    refine ⟨⟨?_, ?_, ?_⟩, h3, h4⟩
    · rw [h0, dif_pos hi]
      rfl
    · rw [h1, dif_pos hi]
      rfl
    · rw [h2, dif_pos hi]
      simp only [rhsCompanionSum1, hz]
  have hk0 := (IterRuns.accumulate (fun s : RhsState1 ↦ s.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.1, (hstep i s t hi hq hb).2⟩) h).1
  have hk1 := (IterRuns.accumulate (fun s : RhsState1 ↦ s.2.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.1, (hstep i s t hi hq hb).2⟩) h).1
  have hk2 := (IterRuns.accumulate (fun s : RhsState1 ↦ s.2.2.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.2, (hstep i s t hi hq hb).2⟩) h).1
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0, zero_add]
    exact sum_unitRow_entry (rhsKeySum1 model key A 0) jj
  · rw [hk1, zero_add]
    exact sum_unitRow_entry (rhsKeySum1 model key A 1) jj
  · rw [hk2, zero_add]
    exact sum_unitRow_entry (rhsCompanionSum1 A z0 z1) jj

/-- The digits of column `jj` of the joined round-`0` RHS ciphertext `(top, bot)`. -/
noncomputable abbrev rhsDigits1 (top bot : ExactMatrix Q6 N 1 20) (jj : Fin 20) :
    ExactMatrix Q6 N (12 + 12) 1 :=
  columnDigits Q6 (12 + 12) Backend.backend 1048576 12 524288 (top 0 jj) (bot 0 jj)

set_option maxHeartbeats 8000000 in
/-- The column loop of round `1` ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoop1 {model : HashModel} {c : ExactMatrix Q6 N 1 24} {top bot : ExactMatrix Q6 N 1 20}
    {final : ExactMatrix Q6 N 1 20}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q6 N 1 20) ↦
      Stage_online_eval.sequential_generatedRoot_146 Backend.backend model GeneratedClaim.stage_2_params i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 20) 0 final) (jj : Fin 20) :
    final 0 jj = (c * rhsDigits1 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q6 N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 1048576 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (rhsDigits1 top bot jj) := by
  have hcount : Int.toNat 20 = 20 * 1 := rfl
  rw [hcount] at h
  let F : Fin 20 → Fin 1 → ExactPoly Q6 N := fun jj _ ↦ (c * rhsDigits1 top bot jj) 0 0
  let Q : Nat → ExactMatrix Q6 N 1 20 → Prop := fun i _ ↦ ∀ jj : Fin 20, jj.val < i →
    ∃ D' : ExactMatrix Q6 N (12 + 12) 1,
      gadgetDecomposeRuns Backend.backend 1048576 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (rhsDigits1 top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q6 N 1 20 ↦ s)
    (fun i ↦ if h : i < 20 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0) Q
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 20 := by omega
      have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
      obtain ⟨_, hw3⟩ := sliceMatrix_column x3
      obtain ⟨_, hw5⟩ := sliceMatrix_column x5
      simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
      rw [concatRows_pairColumn x6, hw3, hw5] at x7
      have hrun := x7
      have hw8 := columnDigits_eq x7 x8
      subst xout hw8 hw11
      refine ⟨?_, fun jj hjj ↦ ?_⟩
      · simp only [dif_pos hi, matrixAdd, matrixMulScalarLeft_eq, matrixMul, Nat.div_one]
        rfl
      · rcases Nat.lt_succ_iff_lt_or_eq.mp hjj with hlt | heq
        · exact hq jj hlt
        · have : jj = ⟨i, hi'⟩ := Fin.ext heq
          subst this
          exact ⟨w7, hrun, x8⟩) h
  obtain ⟨hsum, hQ⟩ := hacc
  dsimp only at hsum
  refine ⟨?_, hQ jj jj.isLt⟩
  rw [hsum, zero_add]
  rw [sum_unitRow_entry F jj]
  simp [F]

/-- The state of the LUT companion loop of round `1`. -/
abbrev LutState1 := ExactMatrix Q6 N 1 320 × ExactMatrix Q6 N 1 320 × ExactMatrix Q6 N 1 320 ×
  ExactMatrix Q6 N 1 12 × ExactMatrix Q6 N 1 12 × Unit

/-- Companion `k` of the round-`0` LUT, substituted by the automorphism of its branch `k / 400`. -/
noncomputable def lutCarry1 (z : Fin 6400 → ExactMatrix Q6 N 1 12) (k : Fin 6400) :
    ExactMatrix Q6 N 1 12 :=
  substituteMatrix (sigma (k / 20 / 20)).toNat (sigma_odd ⟨k / 20 / 20, by omega⟩) (z k)

set_option maxHeartbeats 16000000 in
/-- One iteration of the LUT companion loop of round `1` adds, at column `i / 20` (branch
`i / 20 / 20`, column `i / 20 % 20`), the substituted companion keys and companion `i` paired with the
digits of the substituted key entry `i % 20`. -/
theorem lutLoop1_step {model : HashModel} {key : ByteArray}
    {M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10} {A0 A1 : ExactMatrix Q6 N 1 20}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0) (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    {z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12} {i : Nat} (hi : i < 6400) {s t : LutState1}
    (hs : s.2.2.2.1 = lutCarry1 z0 ⟨i, hi⟩ ∧ s.2.2.2.2.1 = lutCarry1 z1 ⟨i, hi⟩)
    (h : Stage_online_eval.sequential_generatedRoot_135 Backend.backend model
      GeneratedClaim.stage_2_params i
      (s.1, s.2.1, s.2.2.1, s.2.2.2.1, s.2.2.2.2.1, key, A0, A1, z0, z1, ()) t) :
    let B := fun k r ↦ substituteMatrix (sigma (i / 20 / 20)).toNat (sigma_odd ⟨i / 20 / 20, by omega⟩) (digitKey model key Q6 12 (lutOwner 49 ((i / 20 / 20 : Nat) : Int) 3)
      ((i / 20 % 20 : Nat) : Int) ((i % 20 : Nat) : Int) k r)
    let U := keyDigits 12 (fun r j ↦ substituteMatrix (sigma (i / 20 / 20)).toNat (sigma_odd ⟨i / 20 / 20, by omega⟩) (M r j)) ⟨i % 20, by omega⟩
    t.1 = s.1 + (B 0 0 * U 0 + B 0 1 * U 1) 0 0 • unitRow ((i / 20 : Nat) : Int) ∧
      t.2.1 = s.2.1 + (B 1 0 * U 0 + B 1 1 * U 1) 0 0 • unitRow ((i / 20 : Nat) : Int) ∧
      t.2.2.1 = s.2.2.1 +
        (lutCarry1 z0 ⟨i, hi⟩ * U 0 + lutCarry1 z1 ⟨i, hi⟩ * U 1) 0 0 • unitRow ((i / 20 : Nat) : Int) ∧
      t.2.2.2.1 = lutCarry1 z0 ⟨(i + 1) % 6400, Nat.mod_lt _ (by norm_num)⟩ ∧
      t.2.2.2.2.1 = lutCarry1 z1 ⟨(i + 1) % 6400, Nat.mod_lt _ (by norm_num)⟩ := by
  intro B U
  obtain ⟨w2, w3, w5, w6, w7, w9, w10, w12, w13, w14, w18, w22, w23, w25, w26, w41, w42, w44, w45,
    -, -, -, -, -, x2, -, -, -, -, x3, -, -, -, -, -, -, -, -, x5, -, -, -, -, x6, x7, -, -, -, -, -,
    x9, -, -, -, -, x10, -, -, -, -, -, -, -, -, x12, -, -, -, -, x13, x14, -, -, -, -, -, -, -, -,
    -, -, x18, -, -, -, -, -, x22, -, -, -, -, x23, -, -, -, -, -, x25, -, -, -, -, x26, -, -, -, -,
    -, -, -, -, -, -, x41, -, -, -, -, -, x42, -, -, x44, -, -, -, -, -, x45, xout⟩ := h
  have hmem : Int.fdiv (Int.fdiv (Int.ofNat i) 20) 20 = Int.ofNat (i / 20 / 20) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num), Int.ofNat_eq_natCast]
    omega
  have hmem' : Int.fdiv (Int.fdiv (Int.ofNat i) 20) 20 = ((i / 20 / 20 : Nat) : Int) := hmem
  have hcol : Int.fmod (Int.fdiv (Int.ofNat i) 20) 20 = ((i / 20 % 20 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hdiv : Int.fdiv (Int.ofNat i) 20 = ((i / 20 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 20 = ((i % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hnext : 400 * Int.fdiv (Int.fdiv (Int.fmod (1 + Int.ofNat i) 6400) 20) 20 +
      20 * Int.fmod (Int.fdiv (Int.fmod (1 + Int.ofNat i) 6400) 20) 20 +
      Int.fmod (Int.fmod (1 + Int.ofNat i) 6400) 20 = Int.ofNat ((i + 1) % 6400) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 6400 by norm_num), Int.ofNat_eq_natCast]
    omega
  have hsnext : Int.fdiv (Int.fdiv (Int.fmod (1 + Int.ofNat i) 6400) 20) 20 =
      Int.ofNat ((i + 1) % 6400 / 20 / 20) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 6400 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hmem', hcol, hrow] at x2 x9 x22 x25
  rw [hmem] at x3 x6 x10 x13 x23 x26
  rw [hrow] at x5 x12
  rw [hdiv] at x18
  rw [hnext] at x41 x44
  rw [hsnext] at x42 x45
  obtain ⟨_, hw41⟩ := familyGet_eq x41
  obtain ⟨_, hw44⟩ := familyGet_eq x44
  have hw18 := unitRow_window (c := ((i / 20 : Nat) : Int)) (by omega) x18
  obtain ⟨_, hw3⟩ := ringAutomorphismRuns_eq hQ6 hN x3
  obtain ⟨_, hw10⟩ := ringAutomorphismRuns_eq hQ6 hN x10
  obtain ⟨_, hw23⟩ := ringAutomorphismRuns_eq hQ6 hN x23
  obtain ⟨_, hw26⟩ := ringAutomorphismRuns_eq hQ6 hN x26
  obtain ⟨_, hw42⟩ := ringAutomorphismRuns_eq hQ6 hN x42
  obtain ⟨_, hw45⟩ := ringAutomorphismRuns_eq hQ6 hN x45
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q6 N 1 20} {x y : ExactMatrix Q6 N 1 1}
      {u : ExactMatrix Q6 N 12 1}, concatColumns (M r 0) (M r 1) Ar →
      sliceMatrix Ar 0 1 ((i % 20 : Nat) : Int) (1 + ((i % 20 : Nat) : Int)) x →
      ringAutomorphismRuns (sigma (i / 20 / 20)) x y →
      gadgetDecomposeRuns Backend.backend 1048576 12 y u → u = U r := by
    intro r Ar x y u hAr hx hy hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    obtain ⟨_, hy0⟩ := ringAutomorphismRuns_eq hQ6 hN hy
    have hye : y = fun _ _ ↦ joinedEntry (fun j ↦ substituteMatrix (sigma (i / 20 / 20)).toNat (sigma_odd ⟨i / 20 / 20, by omega⟩) (M r j)) ⟨i % 20, by omega⟩ := by
      funext a b
      rw [hy0]
      dsimp only
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0, joinedEntry_substitute]
      rfl
    rw [hye] at hu
    exact entryDigits_eq hu
  have hw7 := hentry hA0 x5 x6 x7
  have hw14 := hentry hA1 x12 x13 x14
  have hK : ∀ k r, digitKey model key Q6 12 (lutOwner 49 ((i / 20 / 20 : Nat) : Int) 3)
      ((i / 20 % 20 : Nat) : Int) ((i % 20 : Nat) : Int) k r = ![![w2, w9], ![w22, w25]] k r := by
    intro k r
    fin_cases k <;> fin_cases r <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x9, hashSample_eq x22, hashSample_eq x25]
  subst hw3 hw10 hw23 hw26
  subst xout hw7 hw14 hw18 hw41 hw44 hw42 hw45
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, ?_, ?_, rfl, rfl⟩ <;>
    simp only [B, hK, hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq, Matrix.cons_val_zero,
      Matrix.cons_val_one] <;>
    rfl

/-- The digit-row sum of the substituted LUT companion keys `σB_k0 u_0 + σB_k1 u_1` of round `1`
at position `p` (branch `p / 20`, column `p % 20`). -/
noncomputable def lutKeySum1 (model : HashModel) (key : ByteArray)
    (M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10) (k : Fin 2) (p : Fin 320) (a : Fin 20) :
    ExactPoly Q6 N :=
  (substituteMatrix (sigma (p / 20)).toNat (sigma_odd ⟨p / 20, by omega⟩)
      (digitKey model key Q6 12 (lutOwner 49 ((p / 20 : Nat) : Int) 3) ((p % 20 : Nat) : Int)
        ((a : Nat) : Int) k 0) *
      keyDigits 12 (fun r j ↦ substituteMatrix (sigma (p / 20)).toNat
        (sigma_odd ⟨p / 20, by omega⟩) (M r j)) a 0 +
    substituteMatrix (sigma (p / 20)).toNat (sigma_odd ⟨p / 20, by omega⟩)
      (digitKey model key Q6 12 (lutOwner 49 ((p / 20 : Nat) : Int) 3) ((p % 20 : Nat) : Int)
        ((a : Nat) : Int) k 1) *
      keyDigits 12 (fun r j ↦ substituteMatrix (sigma (p / 20)).toNat
        (sigma_odd ⟨p / 20, by omega⟩) (M r j)) a 1) 0 0

/-- The digit-row sum of the substituted LUT companions `σz_0 u_0 + σz_1 u_1` of round `1`. -/
noncomputable def lutCompanionSum1 (M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10)
    (z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12) (p : Fin 320) (a : Fin 20) : ExactPoly Q6 N :=
  (lutCarry1 z0 ⟨a + 20 * p, by omega⟩ *
      keyDigits 12 (fun r j ↦ substituteMatrix (sigma (p / 20)).toNat
        (sigma_odd ⟨p / 20, by omega⟩) (M r j)) a 0 +
    lutCarry1 z1 ⟨a + 20 * p, by omega⟩ *
      keyDigits 12 (fun r j ↦ substituteMatrix (sigma (p / 20)).toNat
        (sigma_odd ⟨p / 20, by omega⟩) (M r j)) a 1) 0 0

set_option maxHeartbeats 16000000 in
/-- The LUT companion loop of round `1` ends with the substituted companion keys and companion rows of
every branch, position by position. -/
theorem lutLoop1 {model : HashModel} {key : ByteArray}
    {M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10} {A0 A1 : ExactMatrix Q6 N 1 20}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0) (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    {z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12} {final : LutState1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : LutState1) ↦
      Stage_online_eval.sequential_generatedRoot_135 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (key, (A0,
          (A1, (z0, (z1, ())))))))))) next) (Int.toNat 6400)
      (0, (0, (0, (lutCarry1 z0 ⟨0, by norm_num⟩, (lutCarry1 z1 ⟨0, by norm_num⟩, ()))))) final)
    (p : Fin 320) :
    final.1 0 p = ∑ a, lutKeySum1 model key M 0 p a ∧
      final.2.1 0 p = ∑ a, lutKeySum1 model key M 1 p a ∧
      final.2.2.1 0 p = ∑ a, lutCompanionSum1 M z0 z1 p a := by
  have hcount : Int.toNat 6400 = 320 * 20 := rfl
  rw [hcount] at h
  let Q : Nat → LutState1 → Prop := fun i s ↦
    s.2.2.2.1 = lutCarry1 z0 ⟨i % 6400, Nat.mod_lt _ (by norm_num)⟩ ∧
      s.2.2.2.2.1 = lutCarry1 z1 ⟨i % 6400, Nat.mod_lt _ (by norm_num)⟩
  have hQ0 : Q 0 (0, (0, (0, (lutCarry1 z0 ⟨0, by norm_num⟩, (lutCarry1 z1 ⟨0, by norm_num⟩, ()))))) :=
    ⟨rfl, rfl⟩
  have hstep : ∀ i (s t : LutState1), i < 320 * 20 → Q i s →
      Stage_online_eval.sequential_generatedRoot_135 Backend.backend model GeneratedClaim.stage_2_params i
        (s.1, (s.2.1, (s.2.2.1, (s.2.2.2.1, (s.2.2.2.2.1, (key, (A0, (A1, (z0, (z1, ())))))))))) t →
      (t.1 = s.1 + (if h : i < 320 * 20 then
          lutKeySum1 model key M 0 ⟨i / 20, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 20, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 20 : Nat) : Int)) : ExactMatrix Q6 N 1 320) else 0) ∧
        t.2.1 = s.2.1 + (if h : i < 320 * 20 then
          lutKeySum1 model key M 1 ⟨i / 20, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 20, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 20 : Nat) : Int)) : ExactMatrix Q6 N 1 320) else 0) ∧
        t.2.2.1 = s.2.2.1 + (if h : i < 320 * 20 then
          lutCompanionSum1 M z0 z1 ⟨i / 20, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
            ⟨i % 20, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
          (unitRow (((i / 20 : Nat) : Int)) : ExactMatrix Q6 N 1 320) else 0)) ∧ Q (i + 1) t := by
    intro i s t hi hq hb
    have hi' : i < 6400 := hi
    have hmod : i % 6400 = i := Nat.mod_eq_of_lt hi'
    have hq' : s.2.2.2.1 = lutCarry1 z0 ⟨i, hi'⟩ ∧ s.2.2.2.2.1 = lutCarry1 z1 ⟨i, hi'⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg (lutCarry1 z0) (Fin.ext hmod)),
        h1.trans (congrArg (lutCarry1 z1) (Fin.ext hmod))⟩
    obtain ⟨h0, h1, h2, h3, h4⟩ := lutLoop1_step hA0 hA1 hi' hq' hb
    have hz : (⟨i % 20 + 20 * (i / 20), by omega⟩ : Fin 6400) = ⟨i, hi'⟩ :=
      Fin.ext (Nat.mod_add_div i 20)
    refine ⟨⟨?_, ?_, ?_⟩, h3, h4⟩
    · rw [h0, dif_pos hi]
      rfl
    · rw [h1, dif_pos hi]
      rfl
    · rw [h2, dif_pos hi]
      simp only [lutCompanionSum1, hz]
  have hk0 := (IterRuns.accumulate (fun s : LutState1 ↦ s.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.1, (hstep i s t hi hq hb).2⟩) h).1
  have hk1 := (IterRuns.accumulate (fun s : LutState1 ↦ s.2.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.1, (hstep i s t hi hq hb).2⟩) h).1
  have hk2 := (IterRuns.accumulate (fun s : LutState1 ↦ s.2.2.1) _ Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.2, (hstep i s t hi hq hb).2⟩) h).1
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0, zero_add]
    exact sum_unitRow_entry (lutKeySum1 model key M 0) p
  · rw [hk1, zero_add]
    exact sum_unitRow_entry (lutKeySum1 model key M 1) p
  · rw [hk2, zero_add]
    exact sum_unitRow_entry (lutCompanionSum1 M z0 z1) p

/-- The digits, re-encoded at `Q6`, of column `jj` of the joined switching data `(top, bot)` at `Q5`. -/
noncomputable abbrev switchDigits1 (top bot : ExactMatrix Q5 N 1 20) (jj : Fin 20) :
    ExactMatrix Q6 N (10 + 10) 1 :=
  columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288 (top 0 jj) (bot 0 jj)

set_option maxHeartbeats 8000000 in
/-- The switching column loop of a round-`0` branch ends with the routed row times the digits of the
switching ciphertext, column by column, and every column decomposition runs. -/
theorem switchLoop1 {model : HashModel} {i0 : Nat} {c : ExactMatrix Q6 N 1 20}
    {top bot : ExactMatrix Q5 N 1 20} {final : ExactMatrix Q6 N 1 20}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q6 N 1 20) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_150_44 Backend.backend model
        GeneratedClaim.stage_2_params i0 i (current, (c, (top, (bot, ())))) next) (Int.toNat 20) 0
      final) (jj : Fin 20) :
    final 0 jj = (c * switchDigits1 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (switchDigits1 top bot jj) := by
  have hcount : Int.toNat 20 = 20 * 1 := rfl
  rw [hcount] at h
  let F : Fin 20 → Fin 1 → ExactPoly Q6 N := fun jj _ ↦ (c * switchDigits1 top bot jj) 0 0
  let Q : Nat → ExactMatrix Q6 N 1 20 → Prop := fun i _ ↦ ∀ jj : Fin 20, jj.val < i →
    ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (switchDigits1 top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q6 N 1 20 ↦ s)
    (fun i ↦ if h : i < 20 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0) Q
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 20 := by omega
      have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
      obtain ⟨_, hw3⟩ := sliceMatrix_column x3
      obtain ⟨_, hw5⟩ := sliceMatrix_column x5
      simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
      rw [concatRows_pairColumn x6, hw3, hw5] at x7
      have hrun := x7
      have hw8 := columnDigits_eq x7 x8
      subst xout hw8 hw11
      refine ⟨?_, fun jj hjj ↦ ?_⟩
      · simp only [dif_pos hi, matrixAdd, matrixMulScalarLeft_eq, matrixMul, Nat.div_one]
        rfl
      · rcases Nat.lt_succ_iff_lt_or_eq.mp hjj with hlt | heq
        · exact hq jj hlt
        · have : jj = ⟨i, hi'⟩ := Fin.ext heq
          subst this
          exact ⟨w7, hrun, x8⟩) h
  obtain ⟨hsum, hQ⟩ := hacc
  dsimp only at hsum
  refine ⟨?_, hQ jj jj.isLt⟩
  rw [hsum, zero_add, sum_unitRow_entry F jj]
  simp [F]

/-- The state of the helper column loop of a round-`0` branch. -/
abbrev HelperState1 := ExactMatrix Q6 N 1 20 × ExactMatrix Q6 N 1 20 × ExactMatrix Q6 N 1 20 × Unit

set_option maxHeartbeats 8000000 in
/-- The helper column loop of a round-`0` branch ends with the helper row and its two key rows times
the digits of the moved-down companion keys, column by column, and every column decomposition
runs. -/
theorem helperLoop1 {model : HashModel} {i0 : Nat} {h0 k0 k1 : ExactMatrix Q6 N 1 20}
    {top bot : ExactMatrix Q5 N 1 20} {final : HelperState1}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : HelperState1) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_150_30 Backend.backend model
        GeneratedClaim.stage_2_params i0 i
        (current.1, (current.2.1, (current.2.2.1, (h0, (top, (bot, (k0, (k1, ())))))))) next)
      (Int.toNat 20) (0, (0, (0, ()))) final) (jj : Fin 20) :
    final.1 0 jj = (h0 * switchDigits1 top bot jj) 0 0 ∧
      final.2.1 0 jj = (k0 * switchDigits1 top bot jj) 0 0 ∧
      final.2.2.1 0 jj = (k1 * switchDigits1 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (switchDigits1 top bot jj) := by
  have hcount : Int.toNat 20 = 20 * 1 := rfl
  rw [hcount] at h
  let F : ExactMatrix Q6 N 1 20 → Fin 20 → Fin 1 → ExactPoly Q6 N := fun x jj _ ↦
    (x * switchDigits1 top bot jj) 0 0
  let f : ExactMatrix Q6 N 1 20 → Nat → ExactMatrix Q6 N 1 20 := fun x i ↦ if h : i < 20 * 1 then
      F x ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q6 N 1 20) else 0
  let Q : Nat → HelperState1 → Prop := fun i _ ↦ ∀ jj : Fin 20, jj.val < i →
    ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (switchDigits1 top bot jj)
  have hstep : ∀ i (s t : HelperState1), i < 20 * 1 → Q i s →
      Stage_online_eval.sequential_sequential_generatedRoot_150_30 Backend.backend model
        GeneratedClaim.stage_2_params i0 i
        (s.1, (s.2.1, (s.2.2.1, (h0, (top, (bot, (k0, (k1, ())))))))) t →
      (t.1 = s.1 + f h0 i ∧ t.2.1 = s.2.1 + f k0 i ∧ t.2.2.1 = s.2.2.1 + f k1 i) ∧ Q (i + 1) t := by
    intro i s t hi hq hb
    obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
      -, -, -, -, -, -, x11, xout⟩ := hb
    have hi' : i < 20 := by omega
    have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
    obtain ⟨_, hw3⟩ := sliceMatrix_column x3
    obtain ⟨_, hw5⟩ := sliceMatrix_column x5
    simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
    rw [concatRows_pairColumn x6, hw3, hw5] at x7
    have hrun := x7
    have hw8 := columnDigits_eq x7 x8
    subst xout hw8 hw11
    refine ⟨⟨?_, ?_, ?_⟩, fun jj hjj ↦ ?_⟩
    · simp only [f, F, dif_pos hi, matrixAdd, matrixMulScalarLeft_eq, matrixMul, Nat.div_one]
    · simp only [f, F, dif_pos hi, matrixAdd, matrixMulScalarLeft_eq, matrixMul, Nat.div_one]
    · simp only [f, F, dif_pos hi, matrixAdd, matrixMulScalarLeft_eq, matrixMul, Nat.div_one]
    · rcases Nat.lt_succ_iff_lt_or_eq.mp hjj with hlt | heq
      · exact hq jj hlt
      · have : jj = ⟨i, hi'⟩ := Fin.ext heq
        subst this
        exact ⟨w7, hrun, x8⟩
  have hQ0 : Q 0 (0, (0, (0, ()))) := fun jj h ↦ absurd h (Nat.not_lt_zero _)
  obtain ⟨hk0, hQ⟩ := IterRuns.accumulate (fun s : HelperState1 ↦ s.1) (f h0) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.1, (hstep i s t hi hq hb).2⟩) h
  obtain ⟨hk1, -⟩ := IterRuns.accumulate (fun s : HelperState1 ↦ s.2.1) (f k0) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.1, (hstep i s t hi hq hb).2⟩) h
  obtain ⟨hk2, -⟩ := IterRuns.accumulate (fun s : HelperState1 ↦ s.2.2.1) (f k1) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.2, (hstep i s t hi hq hb).2⟩) h
  dsimp only at hk0 hk1 hk2
  refine ⟨?_, ?_, ?_, hQ jj jj.isLt⟩
  · rw [hk0, zero_add, sum_unitRow_entry (F h0) jj]
    simp [F]
  · rw [hk1, zero_add, sum_unitRow_entry (F k0) jj]
    simp [F]
  · rw [hk2, zero_add, sum_unitRow_entry (F k1) jj]
    simp [F]

/-- The digits of column `l` of component `j` of the round-`0` switching ciphertext of branch `σ`. -/
noncomputable def lutDigits1 (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10) (σ : Fin 16)
    (j : Fin 2) : ExactMatrix Q6 N (10 + 10) 10 :=
  fun a l ↦ columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288 (fam 0 j σ 0 l)
    (fam 1 j σ 0 l) a 0

theorem lutKeySum1_eq {model : HashModel} {key : ByteArray}
    (M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10) (k : Fin 2) (σ : Fin 16) (p : Fin 320)
    (hp : (p : Nat) / 20 = σ) (a : Fin 20) :
    lutKeySum1 model key M k p a =
      (substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3) ((p % 20 : Nat) : Int)
            ((a : Nat) : Int) k 0) *
        keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
      substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3) ((p % 20 : Nat) : Int)
            ((a : Nat) : Int) k 1) *
        keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1) 0 0 := by
  unfold lutKeySum1
  simp only [sigma_congr (by omega) σ.isLt hp, hp]

theorem lutCompanionSum1_eq (M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10)
    (z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12) (σ : Fin 16) (p : Fin 320) (hp : (p : Nat) / 20 = σ)
    (a : Fin 20) :
    lutCompanionSum1 M z0 z1 p a =
      (substituteMatrix (sigma σ).toNat (sigma_odd σ) (z0 ⟨(a : Nat) + 20 * p, by omega⟩) *
        keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
      substituteMatrix (sigma σ).toNat (sigma_odd σ) (z1 ⟨(a : Nat) + 20 * p, by omega⟩) *
        keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1) 0 0 := by
  have e2 : ((a : Nat) + 20 * p) / 20 / 20 = σ := by omega
  unfold lutCompanionSum1 lutCarry1
  simp only [sigma_congr (by omega) σ.isLt hp, sigma_congr (by omega) σ.isLt e2]

set_option maxHeartbeats 16000000 in
/-- The companion row of branch `σ` of the round-`0` LUT, read from the LUT companion loop, adds
`σ(s) (M_σ0 - σ(A_0) D_σ) + (M_σ1 - σ(A_1) D_σ)` and a small error. -/
theorem lutCompanionRows1 {model : HashModel} {key : ByteArray} {sm : ErrorPoly N}
    {g14' g14 : ExactMatrix Q6 N 1 12} (hg14' : gadgetMatrixRuns Backend.backend 1048576 12 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {M : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10} {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10}
    {z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12}
    (hz : ∀ (s : Fin 16) (jj : Fin 20) (a : Fin 20),
      Rel sm (digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q6 N (unitPair sm r)) • g14)
        ![z0 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩,
          z1 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q6 N 1 320}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum1 model key M 0 p a ∧
      L1 0 p = ∑ a, lutKeySum1 model key M 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum1 M z0 z1 p a)
    (σ : Fin 16) :
    ∃ f : Fin 2 → ErrorMatrix N 1 10,
      (∀ j, CoeffBound (f j) ((10 + 10) * (12 * N * 26 * 524288 + 12 * N * 26 * 524288))) ∧
      ∀ j : Fin 2, rowWindow 10 (20 * σ + 10 * j) (by omega) L2 =
        reducePoly Q6 N (substitution (sigma σ).toNat (sigma_odd σ) sm) •
          (rowWindow 10 (20 * σ + 10 * j) (by omega) L0 -
            pairMul (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M 0 i))
              (lutDigits1 fam σ j)) +
        (rowWindow 10 (20 * σ + 10 * j) (by omega) L1 -
          pairMul (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M 1 i))
            (lutDigits1 fam σ j)) +
        reduceMatrix Q6 N 1 10 (f j) := by
  have hgg : g14 = g14' := centeredRebaseRuns_unique hg14 (centeredRebaseRuns_self hQ6 hN g14')
  subst hgg
  have hcop : N.Coprime (sigma σ).toNat := by
    rw [hpow]
    exact coprime_of_odd_pow_two (sigma_odd σ)
  have hdig := fun a r ↦ decompose_entry hQ6 hN Backend.layout3 layout6 rfl hg14'
    (entryDigits_runs hQ6 hN hg14' Backend.layout3 layout6 rfl
      (fun _ _ ↦ joinedEntry (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a))
  choose u' hu' hβ using fun (a : Fin (10 + 10)) (r : Fin 2) ↦ (hdig a r).2
  have hgad : substituteMatrix (sigma σ).toNat (sigma_odd σ) g14 = g14 :=
    substituteMatrix_rebasedGadget hN hg14' (centeredRebaseRuns_self hQ6 hN _) _
  have hz' : ∀ (j : Fin 2) (l : Fin 10) (a : Fin (10 + 10)),
      Rel (substitution (sigma σ).toNat (sigma_odd σ) sm)
        (fun k r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3) ((10 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) k r))
        (fun r ↦ (lutDigits1 fam σ j a l *
          reducePoly Q6 N (unitPair (substitution (sigma σ).toNat (sigma_odd σ) sm) r)) • g14)
        ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩)] 26 := by
    intro j l a
    have h := (hz σ ⟨10 * (j : Nat) + l, by omega⟩ a).substitute hN (sigma_odd σ) hcop
    have hc : (fun r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (![z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + (10 * (j : Nat) + l)), by omega⟩,
          z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + (10 * (j : Nat) + l)), by omega⟩] r)) =
        ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩)] := by
      have e : (⟨(a : Nat) + 20 * (20 * (σ : Nat) + (10 * (j : Nat) + l)), by omega⟩ : Fin 6400) =
          ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩ :=
        Fin.ext (by dsimp only; omega)
      funext r
      fin_cases r <;> simp [e]
    have ht : (fun r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
        ((substitution (sigmaInv σ).toNat (sigmaInv_odd σ)
            (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
              (joinedEntry ![fam 0 0 σ, fam 0 1 σ] ⟨10 * (j : Nat) + l, by omega⟩)
              (joinedEntry ![fam 1 0 σ, fam 1 1 σ] ⟨10 * (j : Nat) + l, by omega⟩) a 0) *
          reducePoly Q6 N (unitPair sm r)) • g14)) =
        fun r ↦ (lutDigits1 fam σ j a l *
          reducePoly Q6 N (unitPair (substitution (sigma σ).toNat (sigma_odd σ) sm) r)) • g14 := by
      funext r
      rw [substituteMatrix_smul', hgad, map_mul,
        substitution_inverse (sigma_odd σ) (sigmaInv_odd σ) (sigma_mul_sigmaInv σ),
        ← reducePoly_substitution, substitution_unitPair, joinedEntry_pair, joinedEntry_pair,
        pair_eta (fun j ↦ fam 0 j σ), pair_eta (fun j ↦ fam 1 j σ)]
      rfl
    rw [hc, ht] at h
    exact h
  have hErow : ∀ (j : Fin 2) (l : Fin 10),
      rowWindow 10 (20 * σ + 10 * j) (by omega) L2 0 l =
        (∑ a : Fin (10 + 10), (![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩)] 0 *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩)] 1 *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).2.2, Matrix.sum_apply]
    simp only [lutCompanionSum1_eq M z0 z1 σ ⟨20 * (σ : Nat) + 10 * (j : Nat) + l, by omega⟩
      (show (20 * (σ : Nat) + 10 * (j : Nat) + l) / 20 = σ by omega)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  have hMrow0 : ∀ (j : Fin 2) (l : Fin 10),
      rowWindow 10 (20 * σ + 10 * j) (by omega) L0 0 l =
        (∑ a : Fin (10 + 10),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 0 0) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 0 1) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).1, Matrix.sum_apply]
    simp only [lutKeySum1_eq M _ σ ⟨20 * (σ : Nat) + 10 * (j : Nat) + l, by omega⟩
        (show (20 * (σ : Nat) + 10 * (j : Nat) + l) / 20 = σ by omega),
      show (20 * (σ : Nat) + 10 * (j : Nat) + l) % 20 = 10 * (j : Nat) + l by omega]
  have hMrow1 : ∀ (j : Fin 2) (l : Fin 10),
      rowWindow 10 (20 * σ + 10 * j) (by omega) L1 0 l =
        (∑ a : Fin (10 + 10),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 1 0) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 1 1) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).2.1, Matrix.sum_apply]
    simp only [lutKeySum1_eq M _ σ ⟨20 * (σ : Nat) + 10 * (j : Nat) + l, by omega⟩
        (show (20 * (σ : Nat) + 10 * (j : Nat) + l) / 20 = σ by omega),
      show (20 * (σ : Nat) + 10 * (j : Nat) + l) % 20 = 10 * (j : Nat) + l by omega]
  have hMrow : ∀ (k j : Fin 2) (l : Fin 10),
      rowWindow 10 (20 * σ + 10 * j) (by omega) (![L0, L1] k) 0 l =
        (∑ a : Fin (10 + 10),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) k 0) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3)
                ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) k 1) *
            keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro k j l
    fin_cases k
    exacts [hMrow0 j l, hMrow1 j l]
  obtain ⟨f, hf, hE⟩ := companion_rows (q := Q6) (m := 10) (m' := 10) (Dg := 12) hN
    (sm := substitution (sigma σ).toNat (sigma_odd σ) sm) (g := g14)
    (A := fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i))
    (D := lutDigits1 fam σ)
    (B := fun j l a k r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
      (digitKey model key Q6 12 (lutOwner 49 ((σ : Nat) : Int) 3) ((10 * (j : Nat) + l : Nat) : Int)
        ((a : Nat) : Int) k r))
    (z := fun j l a ↦ ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (z0 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩),
      substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (z1 ⟨(a : Nat) + 20 * (20 * (σ : Nat) + 10 * (j : Nat) + l), by omega⟩)])
    (bound := 26) hz'
    (u := fun a r ↦ keyDigits 12 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a r)
    (fun a r ↦ (hdig a r).1) (u' := u') (β := 524288) hu' hβ
    (E := fun j ↦ rowWindow 10 (20 * σ + 10 * j) (by omega) L2)
    (M := fun k j ↦ rowWindow 10 (20 * σ + 10 * j) (by omega) (![L0, L1] k)) hErow hMrow
  exact ⟨f, hf, hE⟩

/-- The state of the branch loop of round `1`: the accumulated row and key, and the carried
switching ciphertext. -/
abbrev BranchState1 := ExactMatrix Q6 N 1 10 × ExactMatrix Q6 N 1 10 × ExactMatrix Q6 N 1 10 ×
  ExactMatrix Q6 N 1 10 × ExactMatrix Q6 N 1 10 × ExactMatrix Q6 N 1 10 × ExactMatrix Q5 N 1 10 ×
    ExactMatrix Q5 N 1 10 × ExactMatrix Q5 N 1 10 × ExactMatrix Q5 N 1 10 × Unit

/-- The accumulated row. -/
def BranchState1.row (s : BranchState1) : Fin 2 → ExactMatrix Q6 N 1 10 := ![s.1, s.2.1]

/-- The accumulated key. -/
def BranchState1.key (s : BranchState1) : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10 :=
  ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]

/-- The carried switching ciphertext. -/
def BranchState1.switch (s : BranchState1) : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10 :=
  ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]]

/-- The target of branch `i` of the round-`0` LUT over the switching families `fam`. -/
noncomputable def branchTarget1 (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10) (i : Fin 16) :
    Fin 2 → ExactMatrix Q6 N 1 10 :=
  fun j ↦ scaleUpMatrix Q6 P6 (reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) μ) •
    (reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i + fam 1 j i))

/-- The error bound added by one branch of the round-`0` LUT. -/
abbrev branchBound1 (B : Nat) : Nat :=
  (10 + 10) * N * 26 * 524288 + ((10 + 10) * N * B * 524288 +
    (10 + 10) * (12 * N * 26 * 524288 + 12 * N * 26 * 524288)) + (N * 1 * (P6 / 2) + P6 / 2)

set_option maxHeartbeats 64000000 in
/-- One iteration of the branch loop of round `1` adds branch `i` to the accumulated relation and
carries the switching ciphertext of branch `i + 1`. -/
theorem branchLoop1_step {model : HashModel} {key : ByteArray} {sm μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    {g14' g14 : ExactMatrix Q6 N 1 12} (hg14' : gadgetMatrixRuns Backend.backend 1048576 12 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {c : Fin 2 → ExactMatrix Q6 N 1 10} {Ap : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10} {B : Nat}
    (henc : Rel sm Ap (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * μ * unitPair sm j) • g) c B)
    {h0 h1 : Fin 16 → ExactMatrix Q6 N 1 10}
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model key Q6 10 3 (sigma i))
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] 26)
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10}
    {z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12}
    (hz : ∀ (s : Fin 16) (jj : Fin 20) (a : Fin 20),
      Rel sm (digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q6 N (unitPair sm r)) • g14)
        ![z0 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩,
          z1 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q6 N 1 320}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum1 model key Ap 0 p a ∧
      L1 0 p = ∑ a, lutKeySum1 model key Ap 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum1 Ap z0 z1 p a)
    {i : Nat} (hi : i < 16) {state next : BranchState1}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, hi⟩)
    {T : Fin 2 → ExactMatrix Q6 N 1 10} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    (hbody : Stage_online_eval.sequential_generatedRoot_150 Backend.backend model
      GeneratedClaim.stage_2_params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, h1, L0, L1, key, c 0, c 1, L2, fam 0 0, fam 0 1, fam 1 0,
        fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget1 μ sm fam ⟨i, hi⟩) next.row (Bs + branchBound1 B) ∧
      next.switch = fun k j ↦ fam k j ⟨(i + 1) % 16, Nat.mod_lt _ (by norm_num)⟩ := by
  obtain ⟨w, -, x3, x4, -, -, x6, x7, -, -, -, -, -, -, x9, -, -, -, -, -, -, x10, x11, -, -, -, -, -, -, x12, x13, x14, -, -, -, -, -, -, x16, -, -, -, -, -, -, x17, x18, -, -, -, -, -, -, x19, x20, x21, -, -, x24, x25, x26, x27, x28, x29, -, xseq30, -, -, -, -, -, -, x31, -, -, x34, -, -, x36, x37, x40, x43, -, xseq44, -, -, -, -, -, -, x45, -, -, -, -, -, -, x47, -, -, -, -, -, -, x48, -, -, -, -, -, -, x53, -, -, -, -, -, -, x54, -, -, -, -, -, -, x55, -, -, -, -, -, -, x60, -, -, -, -, -, -, x63, -, -, -, -, -, -, x66, -, -, -, -, -, -, x69, -, -, -, x73, -, -, x75, -, -, x77, -, -, x79, xout⟩ := hbody
  have hs00 : state.2.2.2.2.2.2.1 = fam 0 0 ⟨i, hi⟩ := congrFun (congrFun hswitch 0) 0
  have hs01 : state.2.2.2.2.2.2.2.1 = fam 0 1 ⟨i, hi⟩ := congrFun (congrFun hswitch 0) 1
  have hs10 : state.2.2.2.2.2.2.2.2.1 = fam 1 0 ⟨i, hi⟩ := congrFun (congrFun hswitch 1) 0
  have hs11 : state.2.2.2.2.2.2.2.2.2.1 = fam 1 1 ⟨i, hi⟩ := congrFun (congrFun hswitch 1) 1
  obtain ⟨_, hw4⟩ := familyGet_eq (k := i) x4
  obtain ⟨_, hw6⟩ := familyGet_eq (k := i) x6
  obtain ⟨_, hw34⟩ := ringAutomorphismRuns_eq hQ6 hN x34
  obtain ⟨_, hw36⟩ := ringAutomorphismRuns_eq hQ6 hN x36
  -- The companion keys and row of branch `i`, read from the LUT companion loop.
  have h24 : (20 * Int.ofNat i : Int) = ((20 * i : Nat) : Int) := by simp [Int.ofNat_eq_natCast]
  have hM00 := sliceMatrix_window₂ (20 * i) (10 * 0) x9 x10 h24 (by simp) (by omega)
  have hM01 := sliceMatrix_window₂ (20 * i) (10 * 1) x9 x12 h24 (by simp) (by omega)
  have hM10 := sliceMatrix_window₂ (20 * i) (10 * 0) x16 x17 h24 (by simp) (by omega)
  have hM11 := sliceMatrix_window₂ (20 * i) (10 * 1) x16 x19 h24 (by simp) (by omega)
  have hE0 := sliceMatrix_window₂ (20 * i) (10 * 0) x47 x48 h24 (by simp) (by omega)
  have hE1 := sliceMatrix_window₂ (20 * i) (10 * 1) x47 x55 h24 (by simp) (by omega)
  obtain ⟨f, hf, hEq⟩ := lutCompanionRows1 hg14' hg14 hz hL ⟨i, hi⟩
  have hEσ : ∀ j, ![w.w_48_0, w.w_55_0] j =
      reducePoly Q6 N (substitution (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) sm) •
        (![![w.w_10_0, w.w_12_0], ![w.w_17_0, w.w_19_0]] 0 j -
          pairMul (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩)
            (Ap 0 i')) (lutDigits1 fam ⟨i, hi⟩ j)) +
      (![![w.w_10_0, w.w_12_0], ![w.w_17_0, w.w_19_0]] 1 j -
        pairMul (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩)
          (Ap 1 i')) (lutDigits1 fam ⟨i, hi⟩ j)) +
      reduceMatrix Q6 N 1 10 (f j) := by
    intro j
    fin_cases j
    · simp only [Fin.zero_eta, Matrix.cons_val_zero]
      rw [hE0, hM00, hM10]
      exact hEq 0
    · simp only [Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero]
      rw [hE1, hM01, hM11]
      exact hEq 1
  -- The switching action on the routed row.
  have hsw := switchLoop1 xseq44
  have hR37 : concatColumns
      (substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c 0))
      (substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c 1)) w.w_37_0 := by
    rw [hw34, hw36] at x37
    exact x37
  have hD40 : ∀ (j : Fin 2) (l : Fin 10) (h : 10 * (j : Nat) + l < 10 + 10),
      w.w_40_0 0 ⟨10 * (j : Nat) + l, h⟩ = fam 0 j ⟨i, hi⟩ 0 l := by
    intro j l h
    rw [concatColumns_entry x40, joinedEntry_pair]
    fin_cases j
    · exact congrFun (congrFun hs00 0) l
    · exact congrFun (congrFun hs01 0) l
  have hD43 : ∀ (j : Fin 2) (l : Fin 10) (h : 10 * (j : Nat) + l < 10 + 10),
      w.w_43_0 0 ⟨10 * (j : Nat) + l, h⟩ = fam 1 j ⟨i, hi⟩ 0 l := by
    intro j l h
    rw [concatColumns_entry x43, joinedEntry_pair]
    fin_cases j
    · exact congrFun (congrFun hs10 0) l
    · exact congrFun (congrFun hs11 0) l
  have hDdig : ∀ j : Fin 2, (fun (a : Fin (10 + 10)) (l : Fin 10) ↦ switchDigits1 w.w_40_0 w.w_43_0
      ⟨10 * (j : Nat) + l, by omega⟩ a 0 : ExactMatrix Q6 N (10 + 10) 10) = lutDigits1 fam ⟨i, hi⟩ j := by
    intro j
    funext a l
    simp only [switchDigits1, lutDigits1, hD40, hD43]
  have hX : ∀ j, ![w.w_45_0, w.w_54_0] j = pairMul
      (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c i'))
      (lutDigits1 fam ⟨i, hi⟩ j) := by
    intro j
    rw [← hDdig j, ← window_mul_columns (acc := w.w_44_0) hR37 (fun p ↦ (hsw p).1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (10 * 0) x45 (by simp) (by omega)
    · exact sliceMatrix_window (10 * 1) x54 (by simp) (by omega)
  have hD : ∀ j l, ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10
        (pairColumn (fam 0 j ⟨i, hi⟩ 0 l) (fam 1 j ⟨i, hi⟩ 0 l)) D' ∧
      compactCenteredRebaseRuns 524288 D' (fun a _ ↦ lutDigits1 fam ⟨i, hi⟩ j a l) := by
    intro j l
    obtain ⟨D', h1, h2⟩ := (hsw ⟨10 * (j : Nat) + l, by omega⟩).2
    rw [hD40, hD43] at h1
    refine ⟨D', h1, ?_⟩
    have hfun : (fun (a : Fin (10 + 10)) (_ : Fin 1) ↦ lutDigits1 fam ⟨i, hi⟩ j a l) =
        switchDigits1 w.w_40_0 w.w_43_0 ⟨10 * (j : Nat) + l, by omega⟩ := by
      funext a b
      rw [Subsingleton.elim b 0]
      simp only [switchDigits1, lutDigits1, hD40, hD43]
    rw [hfun]
    exact h2
  -- The helper action on the moved-down companion keys.
  have hhl := helperLoop1 xseq30
  have hA14 : ∀ (j : Fin 2) (l : Fin 10) (h : 10 * (j : Nat) + l < 10 + 10),
      w.w_14_0 0 ⟨10 * (j : Nat) + l, h⟩ = ![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 0 j 0 l := by
    intro j l h
    rw [concatColumns_entry x14, joinedEntry_pair]
    fin_cases j <;> rfl
  have hA21 : ∀ (j : Fin 2) (l : Fin 10) (h : 10 * (j : Nat) + l < 10 + 10),
      w.w_21_0 0 ⟨10 * (j : Nat) + l, h⟩ = ![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 1 j 0 l := by
    intro j l h
    rw [concatColumns_entry x21, joinedEntry_pair]
    fin_cases j <;> rfl
  have hEdec : ∀ j l, ∃ E' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10
        (pairColumn (![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 0 j 0 l)
          (![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 1 j 0 l)) E' ∧
      compactCenteredRebaseRuns 524288 E' (fun a _ ↦ (fun (j : Fin 2) (a : Fin (10 + 10)) (l : Fin 10) ↦
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) j a l) := by
    intro j l
    obtain ⟨E', h1, h2⟩ := (hhl ⟨10 * (j : Nat) + l, by omega⟩).2.2.2
    rw [hA14, hA21] at h1
    refine ⟨E', h1, ?_⟩
    have hfun : (fun (a : Fin (10 + 10)) (_ : Fin 1) ↦
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) =
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ := by
      funext a b
      rw [Subsingleton.elim b 0]
    rw [hfun]
    exact h2
  have hY : ∀ j, ![w.w_31_0, w.w_53_0] j = pairMul ![w.w_4_0, w.w_6_0]
      ((fun (j : Fin 2) (a : Fin (10 + 10)) (l : Fin 10) ↦
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_4_0, w.w_6_0]) (acc := w.w_30_0) x7 (fun p ↦ (hhl p).1) j
      (by omega)]
    fin_cases j
    · exact sliceMatrix_window (10 * 0) x31 (by simp) (by omega)
    · exact sliceMatrix_window (10 * 1) x53 (by simp) (by omega)
  have hK0 : ∀ j, ![w.w_60_0, w.w_63_0] j = pairMul ![w.w_24_0, w.w_25_0]
      ((fun (j : Fin 2) (a : Fin (10 + 10)) (l : Fin 10) ↦
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_24_0, w.w_25_0]) (acc := w.w_30_1) x26
      (fun p ↦ (hhl p).2.1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (10 * 0) x60 (by simp) (by omega)
    · exact sliceMatrix_window (10 * 1) x63 (by simp) (by omega)
  have hK1 : ∀ j, ![w.w_66_0, w.w_69_0] j = pairMul ![w.w_27_0, w.w_28_0]
      ((fun (j : Fin 2) (a : Fin (10 + 10)) (l : Fin 10) ↦
        switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_27_0, w.w_28_0]) (acc := w.w_30_2) x29
      (fun p ↦ (hhl p).2.2.1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (10 * 0) x66 (by simp) (by omega)
    · exact sliceMatrix_window (10 * 1) x69 (by simp) (by omega)
  have hK : ∀ k j, ![![w.w_60_0, w.w_63_0], ![w.w_66_0, w.w_69_0]] k j =
      pairMul (![![w.w_24_0, w.w_25_0], ![w.w_27_0, w.w_28_0]] k)
        ((fun (j : Fin 2) (a : Fin (10 + 10)) (l : Fin 10) ↦
          switchDigits1 w.w_14_0 w.w_21_0 ⟨10 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro k j
    fin_cases k
    exacts [hK0 j, hK1 j]
  have hAd : lutA model key Q6 10 3 (sigma (⟨i, hi⟩ : Fin 16)) =
      ![![w.w_24_0, w.w_25_0], ![w.w_27_0, w.w_28_0]] := by
    funext k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq x24, hashSample_eq x25,
        hashSample_eq x27, hashSample_eq x28]
  have hhelp' := hhelp ⟨i, hi⟩
  rw [hAd, show h0 ⟨i, hi⟩ = w.w_4_0 from hw4.symm, show h1 ⟨i, hi⟩ = w.w_6_0 from hw6.symm] at hhelp'
  -- The branch.
  have hbranch := lut_branch_dynamic (q := Q6) (q' := Q5) (p := P6) (m := 10) (m' := 10) (log := 5)
    (bound := 524288) hN hpow q6_eq hQ5 Backend.layout2 layout5 rfl hg' hg hsm henc
    (sigma_odd ⟨i, hi⟩) (bsw := fun k j ↦ fam k j ⟨i, hi⟩) hD hX hEσ hf
    (Asw := ![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [x11, x13, x18, x20]) hEdec hhelp' hY hK
  have hsum := hstate.add hbranch
  have hnext : Int.fmod (1 + Int.ofNat i) 16 = Int.ofNat ((i + 1) % 16) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hnext] at x73 x75 x77 x79
  obtain ⟨_, hw73⟩ := familyGet_eq x73
  obtain ⟨_, hw75⟩ := familyGet_eq x75
  obtain ⟨_, hw77⟩ := familyGet_eq x77
  obtain ⟨_, hw79⟩ := familyGet_eq x79
  refine ⟨?_, ?_⟩
  · have hk : next.key = state.key + ![![w.w_60_0, w.w_63_0], ![w.w_66_0, w.w_69_0]] := by
      rw [xout]
      funext k j
      fin_cases k <;> fin_cases j <;> rfl
    have hr : next.row = state.row + fun j ↦ ![w.w_31_0, w.w_53_0] j +
        (![w.w_45_0, w.w_54_0] j + ![w.w_48_0, w.w_55_0] j) := by
      rw [xout]
      funext j
      fin_cases j <;> rfl
    rw [hk, hr]
    exact hsum
  · rw [xout]
    funext k j
    fin_cases k <;> fin_cases j
    exacts [hw73, hw75, hw77, hw79]

/-- Branch `i` of the round-`0` LUT, zero past the sixteen branches. -/
noncomputable def branchTargetN1 (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10) (i : Nat) : Fin 2 → ExactMatrix Q6 N 1 10 :=
  if h : i < 16 then branchTarget1 μ sp fam ⟨i, h⟩ else 0

set_option maxHeartbeats 16000000 in
/-- The branch loop of round `1` accumulates the sixteen branches. -/
theorem branchLoop1 {model : HashModel} {key : ByteArray} {sm μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    {g14' g14 : ExactMatrix Q6 N 1 12} (hg14' : gadgetMatrixRuns Backend.backend 1048576 12 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {c : Fin 2 → ExactMatrix Q6 N 1 10} {Ap : Fin 2 → Fin 2 → ExactMatrix Q6 N 1 10} {B : Nat}
    (henc : Rel sm Ap (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * μ * unitPair sm j) • g) c B)
    {h0 h1 : Fin 16 → ExactMatrix Q6 N 1 10}
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model key Q6 10 3 (sigma i))
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] 26)
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q5 N 1 10}
    {z0 z1 : Fin 6400 → ExactMatrix Q6 N 1 12}
    (hz : ∀ (s : Fin 16) (jj : Fin 20) (a : Fin 20),
      Rel sm (digitKey model key Q6 12 (lutOwner 49 ((s : Nat) : Int) 3) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q6 N (unitPair sm r)) • g14)
        ![z0 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩,
          z1 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q6 N 1 320}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum1 model key Ap 0 p a ∧
      L1 0 p = ∑ a, lutKeySum1 model key Ap 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum1 Ap z0 z1 p a)
    {final : BranchState1}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : BranchState1) ↦
      Stage_online_eval.sequential_generatedRoot_150 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, h1, L0, L1, key, c 0, c 1, L2,
          fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) (Int.toNat 16)
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range 16, branchTargetN1 μ sm fam i) final.row
      (16 * branchBound1 B) := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at hruns
  refine (MxxIR.IterRuns.invariant
    (Invariant := fun k (s : BranchState1) ↦ k ≤ 16 →
      Rel sm s.key (∑ i ∈ Finset.range k, branchTargetN1 μ sm fam i) s.row (k * branchBound1 B) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k % 16, Nat.mod_lt _ (by norm_num)⟩) ?_ ?_ hruns le_rfl).1
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q6 N 1 10) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [BranchState1.key, BranchState1.row, hrow]
      exact Rel.zero
    · funext k j
      fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    have hi : i < 16 := by omega
    have hsw2 : current.switch = fun k' j ↦ fam k' j ⟨i, hi⟩ := by
      rw [hsw]
      funext k' j
      exact congrArg (fam k' j) (Fin.ext (Nat.mod_eq_of_lt hi))
    obtain ⟨hrel', hsw'⟩ := branchLoop1_step hsm hg' hg hg14' hg14 henc hhelp hz hL hi hsw2 hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, branchTargetN1, dif_pos hi, add_mul, one_mul]
    exact hrel'


/-- The error bound after the round-`0` multiplication. -/
abbrev mulBound1 : Nat :=
  (12 + 12) * N * roundBound0 * 524288 + (12 + 12) * (12 * N * 26 * 524288 + 12 * N * 26 * 524288) +
    N * 1 * 26

/-- The error bound of the round-`0` output at level `4`. -/
abbrev roundBound1 : Nat :=
  (16 * branchBound1 mulBound1 + (N + 2) * (P6 / 2)) / P6 + 16 * (N * 1 * 26)

set_option maxHeartbeats 128000000 in
/-- Round `0`: the state encoding of `μ` at level `4` times the Dynamic RHS of `ν` at level `3`, then
the LUT of table `1`, gives an encoding of `X^(T₀ arg)` at level `4` when `μ ν = X^arg`. -/
theorem round1 {model : HashModel}
    {v1 v3 v5 v7 : Fin 16 → ExactMatrix Q3 N 1 1} {v9 v10 : Fin 16 → ExactMatrix Q4 N 1 6}
    {v12 v15 : Fin 384 → ExactMatrix Q4 N 1 8} {v18 : ByteArray}
    {v20 v22 : Fin 192 → ExactMatrix Q4 N 1 8} {v25 v27 v29 v31 : Fin 16 → ExactMatrix Q4 N 1 8}
    {v33 v34 : Fin 16 → ExactMatrix Q5 N 1 8} {v36 v39 : Fin 4096 → ExactMatrix Q5 N 1 10}
    {v43 v45 : Fin 320 → ExactMatrix Q5 N 1 10} {v48 v50 v52 v54 : Fin 16 → ExactMatrix Q5 N 1 10}
    {v56 v57 : Fin 16 → ExactMatrix Q6 N 1 10} {v59 v62 : Fin 6400 → ExactMatrix Q6 N 1 12}
    {v66 v68 : Fin 480 → ExactMatrix Q6 N 1 12} {v71 v73 v75 v77 : Fin 16 → ExactMatrix Q6 N 1 12}
    {v79 v80 : Fin 16 → ExactMatrix Q7 N 1 12} {v82 v85 : Fin 9216 → ExactMatrix Q7 N 1 14}
    {v89 v91 : Fin 672 → ExactMatrix Q7 N 1 14} {v108 v109 : ExactMatrix Q7 N 1 14}
    {v111 v112 v114 v115 : ExactMatrix Q7 N 1 12} {v140 v141 v143 v144 : ExactMatrix Q6 N 1 10}
    {v169 v170 v172 v173 : ExactMatrix Q5 N 1 8} {v198 v199 v201 v202 : ExactMatrix Q4 N 1 6}
    {outputs} {w : Stage_online_eval.generatedRoot.Witness}
    (hb : Stage_online_eval.generatedRoot.body Backend.backend model GeneratedClaim.stage_2_params
      (v1, v3, v5, v7, v9, v10, v12, v15, v18, v20, v22, v25, v27, v29, v31, v33, v34, v36, v39,
        v43, v45, v48, v50, v52, v54, v56, v57, v59, v62, v66, v68, v71, v73, v75, v77, v79, v80,
        v82, v85, v89, v91, v108, v109, v111, v112, v114, v115, v140, v141, v143, v144, v169, v170,
        v172, v173, v198, v199, v201, v202, ())
      outputs w)
    {sm μ ν : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hμ : polyNorm μ ≤ 1)
    {g14' g14 : ExactMatrix Q6 N 1 12} (hg14' : gadgetMatrixRuns Backend.backend 1048576 12 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    (hinput : Rel sm ![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]
      (fun j ↦ reducePoly Q6 N (1 * μ * unitPair sm j) • g14) ![w.w_137_0, w.w_138_0] roundBound0)
    (hrhs : ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N sm • ![v140, v141] j + ![v143, v144] j =
        reducePoly Q6 N (((P6 : Nat) : Int) * ν * unitPair sm j) • g + reduceMatrix Q6 N 1 10 (e j))
    (hrhsz : ∀ (jj : Fin 20) (a : Fin 24),
      Rel sm (digitKey model v18 Q6 12 (rhsOwner 49 2) ((jj : Nat) : Int) ((a : Nat) : Int))
        (fun r ↦ (columnDigits Q6 (12 + 12) Backend.backend 1048576 12 524288
            (joinedEntry ![v140, v141] jj) (joinedEntry ![v143, v144] jj) a 0 *
          reducePoly Q6 N (unitPair sm r)) • g14)
        ![v66 ⟨(a : Nat) + 24 * jj, by omega⟩, v68 ⟨(a : Nat) + 24 * jj, by omega⟩] 26)
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model v18 Q6 10 3 (sigma i))
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![v56 i, v57 i] 26)
    (hswitch : ∀ i : Fin 16, ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) sm) • ![v48 i, v50 i] j +
          ![v52 i, v54 i] j =
        (lutCoefficient Q5 inverse5 (MxxExponentLutNand.table 1) (sigma i).toNat *
          reducePoly Q5 N (1 * unitPair sm j)) • g' + reduceMatrix Q5 N 1 10 (e j))
    (hlutz : ∀ (s : Fin 16) (jj : Fin 20) (a : Fin 20),
      Rel sm (digitKey model v18 Q6 12 (lutOwner 49 ((s : Nat) : Int) 3) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q6 (10 + 10) Backend.backend 1048576 10 524288
              (joinedEntry ![v48 s, v50 s] jj) (joinedEntry ![v52 s, v54 s] jj) a 0) *
          reducePoly Q6 N (unitPair sm r)) • g14)
        ![v59 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩,
          v62 ⟨(a : Nat) + 20 * (20 * (s : Nat) + jj), by omega⟩] 26)
    {arg : Fin 16} (harg : μ * ν = Xe (n := N) arg) :
    Rel sm ![![w.w_151_0, w.w_152_0], ![w.w_154_0, w.w_155_0]]
      (fun j ↦ reducePoly Q5 N (1 * Xe (n := N) (MxxExponentLutNand.table 1 arg) * unitPair sm j) • g')
      ![w.w_166_0, w.w_167_0] roundBound1 := by
  unfold Stage_online_eval.generatedRoot.body at hb
  dsimp only at hb
  have h68 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h69 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h70 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h71 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h77 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h82 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h88 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h93 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h97 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h101 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h199 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h202 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h204 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h211 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h218 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h219 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h226 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h233 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h234 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h236 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h239 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h240 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h241 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h243 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h250 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h257 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h259 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h260 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h261 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h263 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h264 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h300 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h301 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hgg : g14 = g14' := centeredRebaseRuns_unique hg14 (centeredRebaseRuns_self hQ6 hN g14')
  subst hgg
  have hinput' := hinput.target_eq (t' := fun j ↦
    reducePoly Q6 N (((1 : Nat) : Int) * μ * unitPair sm j) • g14) (by funext j; simp)
  -- The key of the state input, moved down by the previous round.
  have hA0 : concatColumns (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]] 0 0) (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]] 0 1) w.w_124_0 := h199
  have hA1 : concatColumns (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]] 1 0) (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]] 1 1) w.w_127_0 := h202
  -- The RHS companion loop.
  obtain ⟨_, hw90⟩ := familyGet_index (k := 0) h97 (by decide)
  obtain ⟨_, hw92⟩ := familyGet_index (k := 0) h101 (by decide)
  rw [hw90, hw92] at h204
  have hrl := rhsLoop1 hA0 hA1 h204
  dsimp only at hrl
  -- The column loop and the digits of the RHS columns.
  have hcl := colLoop1 h243
  let D : Fin 2 → ExactMatrix Q6 N (12 + 12) 10 := fun j a l ↦
    columnDigits Q6 (12 + 12) Backend.backend 1048576 12 524288
      (joinedEntry ![v140, v141] ⟨10 * (j : Nat) + l, by omega⟩)
      (joinedEntry ![v143, v144] ⟨10 * (j : Nat) + l, by omega⟩) a 0
  have hDcol : ∀ (j : Fin 2) (l : Fin 10),
      rhsDigits1 w.w_142_0 w.w_145_0 ⟨10 * (j : Nat) + l, by omega⟩ = fun a _ ↦ D j a l := by
    intro j l
    funext a b
    rw [Subsingleton.elim b 0]
    simp only [rhsDigits1, D, concatColumns_entry h240, concatColumns_entry h241]
  have hD : ∀ j l, ∃ D' : ExactMatrix Q6 N (12 + 12) 1,
      gadgetDecomposeRuns Backend.backend 1048576 12
        (pairColumn (![v140, v141] j 0 l) (![v143, v144] j 0 l)) D' ∧
      compactCenteredRebaseRuns 524288 D' (fun a _ ↦ D j a l) := by
    intro j l
    obtain ⟨D', h1, h2⟩ := (hcl ⟨10 * (j : Nat) + l, by omega⟩).2
    rw [concatColumns_entry h240, concatColumns_entry h241, joinedEntry_pair, joinedEntry_pair] at h1
    rw [hDcol] at h2
    exact ⟨D', h1, h2⟩
  have hcD : ∀ j : Fin 2, rowWindow 10 (10 * j) (by omega) w.w_146_0 = pairMul ![w.w_137_0, w.w_138_0] (D j) := by
    intro j
    rw [window_mul_columns (m := 12) (k := 10) (row := ![w.w_137_0, w.w_138_0]) (acc := w.w_146_0) h239
      (fun p ↦ (hcl p).1) j (by omega)]
    refine congrArg (pairMul ![w.w_137_0, w.w_138_0]) ?_
    funext a l
    rw [hDcol]
  -- The companion row and keys of the RHS.
  have hdig := fun (a : Fin (12 + 12)) (r : Fin 2) ↦ decompose_entry hQ6 hN Backend.layout3 layout6 rfl
    hg14' (entryDigits_runs hQ6 hN hg14' Backend.layout3 layout6 rfl
      (fun _ _ ↦ joinedEntry (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]] r) a))
  choose u' hu' hβ using fun (a : Fin (12 + 12)) (r : Fin 2) ↦ (hdig a r).2
  have hErow : ∀ (j : Fin 2) (l : Fin 10),
      rowWindow 10 (10 * j) (by omega) w.w_128_2 0 l =
        (∑ a : Fin (12 + 12),
          (![v66 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩,
              v68 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩] 0 *
            keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 0 +
          ![v66 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩,
              v68 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩] 1 *
            keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hrl ⟨10 * (j : Nat) + l, by omega⟩).2.2, Matrix.sum_apply]
    simp only [rhsCompanionSum1, Matrix.cons_val_zero, Matrix.cons_val_one]
  have hMw : ∀ j : Fin 2, ![w.w_129_0, w.w_130_0] j = rowWindow 10 (10 * j) (by omega) w.w_128_0 ∧
      ![w.w_132_0, w.w_133_0] j = rowWindow 10 (10 * j) (by omega) w.w_128_1 := by
    intro j
    fin_cases j
    · exact ⟨sliceMatrix_window (10 * 0) h211 (by simp) (by omega),
        sliceMatrix_window (10 * 0) h226 (by simp) (by omega)⟩
    · exact ⟨sliceMatrix_window (10 * 1) h218 (by simp) (by omega),
        sliceMatrix_window (10 * 1) h233 (by simp) (by omega)⟩
  have hMrow0 : ∀ (j : Fin 2) (l : Fin 10), ![w.w_129_0, w.w_130_0] j 0 l =
      (∑ a : Fin (12 + 12),
        (digitKey model v18 Q6 12 (rhsOwner 49 2) ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int)
            0 0 * keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 0 +
          digitKey model v18 Q6 12 (rhsOwner 49 2) ((10 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) 0 1 * keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 1)) 0 0 := by
    intro j l
    rw [(hMw j).1]
    simp only [rowWindow]
    rw [(hrl ⟨10 * (j : Nat) + l, by omega⟩).1, Matrix.sum_apply]
    rfl
  have hMrow1 : ∀ (j : Fin 2) (l : Fin 10), ![w.w_132_0, w.w_133_0] j 0 l =
      (∑ a : Fin (12 + 12),
        (digitKey model v18 Q6 12 (rhsOwner 49 2) ((10 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int)
            1 0 * keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 0 +
          digitKey model v18 Q6 12 (rhsOwner 49 2) ((10 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) 1 1 * keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a 1)) 0 0 := by
    intro j l
    rw [(hMw j).2]
    simp only [rowWindow]
    rw [(hrl ⟨10 * (j : Nat) + l, by omega⟩).2.1, Matrix.sum_apply]
    rfl
  obtain ⟨f, hf, hE⟩ := companion_rows (q := Q6) (m := 12) (m' := 10) (Dg := 12) hN (sm := sm)
    (g := g14) (A := ![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) (D := D)
    (B := fun j l a ↦ digitKey model v18 Q6 12 (rhsOwner 49 2) ((10 * (j : Nat) + l : Nat) : Int)
      ((a : Nat) : Int))
    (z := fun j l a ↦ ![v66 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩,
      v68 ⟨(a : Nat) + 24 * (10 * (j : Nat) + l), by omega⟩])
    (bound := 26) (fun j l a ↦ hrhsz ⟨10 * (j : Nat) + l, by omega⟩ a)
    (u := fun a r ↦ keyDigits 12 (![![w.w_122_0, w.w_123_0], ![w.w_125_0, w.w_126_0]]) a r) (fun a r ↦ (hdig a r).1) (u' := u')
    (β := 524288) hu' hβ (E := fun j ↦ rowWindow 10 (10 * j) (by omega) w.w_128_2)
    (M := ![![w.w_129_0, w.w_130_0], ![w.w_132_0, w.w_133_0]]) hErow
    (fun k j l ↦ by fin_cases k; exacts [hMrow0 j l, hMrow1 j l])
  -- The product.
  obtain ⟨eB, heB, hrhs'⟩ := hrhs
  have hmul := mul_private_dynamic (q := Q6) (m := 12) (m' := 10) (bound := 524288) hN hQ6
    Backend.layout3 layout6 rfl hg14' (centeredRebaseRuns_self hQ6 hN _) hinput'
    (top := ![v140, v141]) (bot := ![v143, v144]) (D := D) hD hrhs' heB
    (smulGain_of_polyNorm hN hμ) hE hf
  have hprod : Rel sm ![![w.w_129_0, w.w_130_0], ![w.w_132_0, w.w_133_0]]
      (fun j ↦ reducePoly Q6 N (((P6 : Nat) : Int) * Xe (n := N) arg * unitPair sm j) • g)
      ![w.w_148_0, w.w_149_0] mulBound1 := by
    have hrows : ![w.w_148_0, w.w_149_0] = fun j ↦ pairMul ![w.w_137_0, w.w_138_0] (D j) +
        rowWindow 10 (10 * j) (by omega) w.w_128_2 := by
      funext j
      rw [← hcD j, ← rowWindow_add]
      fin_cases j
      · exact sliceMatrix_window (10 * 0) h250 (by simp) (by omega)
      · exact sliceMatrix_window (10 * 1) h257 (by simp) (by omega)
    rw [hrows]
    convert hmul using 2 with j
    rw [smul_smul, ← map_mul, ← harg]
    congr 2
    ring
  -- The LUT companion loop.
  obtain ⟨_, hw83⟩ := familyGet_index (k := 0) h77 (by decide)
  obtain ⟨_, hw86⟩ := familyGet_index (k := 0) h88 (by decide)
  obtain ⟨_, hw84⟩ := ringAutomorphismRuns_eq hQ6 hN h82
  obtain ⟨_, hw87⟩ := ringAutomorphismRuns_eq hQ6 hN h93
  have hc0 : w.w_61_0 = lutCarry1 v59 ⟨0, by norm_num⟩ := by
    rw [hw84, hw83]
    rfl
  have hc1 : w.w_64_0 = lutCarry1 v62 ⟨0, by norm_num⟩ := by
    rw [hw87, hw86]
    rfl
  rw [hc0, hc1] at h236
  have hll := lutLoop1 (M := ![![w.w_129_0, w.w_130_0], ![w.w_132_0, w.w_133_0]]) h219 h234 h236
  dsimp only at hll
  -- The sixteen branches.
  obtain ⟨_, hw72⟩ := familyGet_index (k := 0) h68 (by decide)
  obtain ⟨_, hw74⟩ := familyGet_index (k := 0) h69 (by decide)
  obtain ⟨_, hw76⟩ := familyGet_index (k := 0) h70 (by decide)
  obtain ⟨_, hw78⟩ := familyGet_index (k := 0) h71 (by decide)
  rw [hw72, hw74, hw76, hw78] at h259
  have hbl := branchLoop1 (fam := ![![v48, v50], ![v52, v54]]) (c := ![w.w_148_0, w.w_149_0])
    hsm hg' hg hg14' (centeredRebaseRuns_self hQ6 hN _) hprod hhelp hlutz hll h259
  -- The sum of the sixteen branches, moved down to level `4`.
  have hsum := hbl.target_eq (t' := fun j ↦ ∑ i : Fin 16, scaleUpMatrix Q6 P6
      (reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) arg)) •
        (reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) sm) •
          ![![v48, v50], ![v52, v54]] 0 j i + ![![v48, v50], ![v52, v54]] 1 j i))) (by
    funext j
    rw [Finset.sum_apply, Finset.sum_range]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [branchTargetN1, dif_pos i.isLt]
    rfl)
  choose eS heS hS using hswitch
  have hwalsh : ∑ i : Fin 16, reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i)
      (Xe (n := N) arg)) * lutCoefficient Q5 inverse5 (MxxExponentLutNand.table 1) (sigma i).toNat =
      reducePoly Q5 N (Xe (n := N) (MxxExponentLutNand.table 1 arg)) := by
    rw [reducePoly_Xe, ← lut_walsh inverse5_mul (MxxExponentLutNand.table 1) arg]
    apply Finset.sum_congr rfl
    intro i _
    rw [substitution_root_pow, reducePoly_Xe, Nat.mul_comm]
  exact lut_finish (W := 16) (q := Q6) (q' := Q5) (p := P6) (m' := 10) hN q6_eq hQ6 hQ5
    (by norm_num) (sp := sm) (μ := Xe (n := N) arg)
    (ν := Xe (n := N) (MxxExponentLutNand.table 1 arg)) (Δ' := 1) hsm
    (fun i ↦ substitution (sigma i).toNat (sigma_odd i)) (fun i ↦ polyNorm_substitution_Xe i arg)
    (bsw := fun i k j ↦ ![![v48, v50], ![v52, v54]] k j i) hsum (g' := g')
    (C := fun i ↦ lutCoefficient Q5 inverse5 (MxxExponentLutNand.table 1) (sigma i).toNat)
    (fun i j ↦ by fin_cases j; exacts [hS i 0, hS i 1]) (fun i j ↦ heS i j) hwalsh
    (K' := ![![w.w_151_0, w.w_152_0], ![w.w_154_0, w.w_155_0]])
    (R' := ![w.w_166_0, w.w_167_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h260, h261, h263, h264])
    (fun j ↦ by fin_cases j; exacts [h300, h301])




end NandProof

