import NandDigits
import NandRound1

/-!
# Online evaluation, round 2

Round `0` multiplies the level-`0` state encoding by the Dynamic RHS of circuit input `0` and
evaluates the first LUT. The generated loops compute, in order, the companion keys and row of the
RHS (`sequential_generatedRoot_133`), the companion keys and rows of the sixteen LUT branches
(`sequential_generatedRoot_140`), the ciphertext action `c Dec(C)` (`sequential_generatedRoot_151`),
and the sixteen branches (`sequential_generatedRoot_155`); the root moves the sum down to `Q4`.
Both companion loops decompose one key column per iteration and accumulate its companion keys and
row over the output columns in a nested loop, which reads each companion one iteration ahead.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The state of the RHS companion loop of round `2`: the companion keys and the companion row. -/
abbrev RhsState2 := ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × Unit

/-- The state of the inner RHS companion loop of round `2`: the blocks of the companion keys and
the companion row, and the carried companion. -/
abbrev RhsBlock2 := ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 ×
  ExactMatrix Q5 N 1 10 × ExactMatrix Q5 N 1 10 × Unit

/-- The digit-row sum of the RHS companion keys `B_k0 u_0 + B_k1 u_1` of round `2`. -/
noncomputable def rhsKeySum2 (model : HashModel) (key : ByteArray)
    (A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10) (k : Fin 2) (jj : Fin 16) (a : Fin 20) :
    ExactPoly Q5 N :=
  (digitKey model key Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int) ((a : Nat) : Int) k 0 *
      keyDigits 10 A a 0 +
    digitKey model key Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int) ((a : Nat) : Int) k 1 *
      keyDigits 10 A a 1) 0 0

/-- The digit-row sum of the RHS companions `z_0 u_0 + z_1 u_1` of round `2`. -/
noncomputable def rhsCompanionSum2 (A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10)
    (z0 z1 : Fin 320 → ExactMatrix Q5 N 1 10) (jj : Fin 16) (a : Fin 20) : ExactPoly Q5 N :=
  (z0 ⟨jj + 16 * a, by omega⟩ * keyDigits 10 A a 0 + z1 ⟨jj + 16 * a, by omega⟩ * keyDigits 10 A a 1) 0 0

set_option maxHeartbeats 8000000 in
/-- One iteration `j` of the inner RHS companion loop of round `2` at digit row `a` adds, at column
`j`, the companion keys and companion `j + 16 a` paired with the digits `U` of key entry `a`, and
reads the next companion of that digit row. -/
theorem rhsInner2_step {model : HashModel} {key : ByteArray} {U : Fin 2 → ExactMatrix Q5 N 10 1}
    {z0 z1 : Fin 320 → ExactMatrix Q5 N 1 10} {a j : Nat} (ha : a < 20) (hj : j < 16)
    {s t : RhsBlock2}
    (hs : s.2.2.2.1 = z0 ⟨j + 16 * a, by omega⟩ ∧ s.2.2.2.2.1 = z1 ⟨j + 16 * a, by omega⟩)
    (h : Stage_online_eval.sequential_sequential_generatedRoot_133_16 Backend.backend model
      GeneratedClaim.stage_2_params a j
      (s.1, s.2.1, s.2.2.1, s.2.2.2.1, s.2.2.2.2.1, key, U 0, U 1, z0, z1, ()) t) :
    let B := digitKey model key Q5 10 (rhsOwner 50 4) ((j : Nat) : Int) ((a : Nat) : Int)
    t.1 = s.1 + (B 0 0 * U 0 + B 0 1 * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (B 1 0 * U 0 + B 1 1 * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.2.1 = s.2.2.1 + (z0 ⟨j + 16 * a, by omega⟩ * U 0 + z1 ⟨j + 16 * a, by omega⟩ * U 1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.2.2.1 = z0 ⟨(j + 1) % 16 + 16 * a, by omega⟩ ∧
      t.2.2.2.2.1 = z1 ⟨(j + 1) % 16 + 16 * a, by omega⟩ := by
  intro B
  obtain ⟨w2, w5, w10, w14, w16, w31, w33, -, x2, -, x5, -, -, -, -, -, -, -, -, x10, -, x14, -,
    x16, -, -, -, -, x31, -, -, x33, xout⟩ := h
  have hrow : Int.fmod (Int.ofNat a) 20 = ((a : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hnext : 16 * Int.fmod (Int.ofNat a) 20 + Int.fmod (1 + Int.ofNat j) 16 =
      Int.ofNat ((j + 1) % 16 + 16 * a) := by
    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 20 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hrow] at x2 x5 x14 x16
  rw [hnext] at x31 x33
  obtain ⟨_, hw31⟩ := familyGet_eq x31
  obtain ⟨_, hw33⟩ := familyGet_eq x33
  have hw10 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x10
  have hB : B = ![![w2, w5], ![w14, w16]] := by
    funext k i
    fin_cases k <;> fin_cases i <;>
      simp [B, digitKey, digitKeyPrefix, digitKeyHead, rhsOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x5, hashSample_eq x14, hashSample_eq x16]
  subst xout hw10 hw31 hw33
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, ?_, ?_, rfl, rfl⟩ <;>
    simp only [hB, hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq, Matrix.cons_val_zero,
      Matrix.cons_val_one]

set_option maxHeartbeats 8000000 in
/-- The inner RHS companion loop of round `2` at digit row `a` ends with the companion keys and the
companion row of that digit row, column by column. -/
theorem rhsInner2 {model : HashModel} {key : ByteArray} (A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10)
    {z0 z1 : Fin 320 → ExactMatrix Q5 N 1 10} (a : Fin 20) {final : RhsBlock2}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : RhsBlock2) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_133_16 Backend.backend model
        GeneratedClaim.stage_2_params a i
        (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (key,
          (keyDigits 10 A a 0, (keyDigits 10 A a 1, (z0, (z1, ())))))))))) next) (Int.toNat 16)
      (0, (0, (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))))) final) (jj : Fin 16) :
    final.1 0 jj = rhsKeySum2 model key A 0 jj a ∧ final.2.1 0 jj = rhsKeySum2 model key A 1 jj a ∧
      final.2.2.1 0 jj = rhsCompanionSum2 A z0 z1 jj a := by
  let Q : Nat → RhsBlock2 → Prop := fun i s ↦
    s.2.2.2.1 = z0 ⟨i % 16 + 16 * a, by omega⟩ ∧ s.2.2.2.2.1 = z1 ⟨i % 16 + 16 * a, by omega⟩
  have hQ0 : Q 0 (0, (0, (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))))) :=
    ⟨congrArg z0 (Fin.ext (by simp)), congrArg z1 (Fin.ext (by simp))⟩
  have hstep : ∀ i (s t : RhsBlock2) (hi : i < 16), Q i s →
      Stage_online_eval.sequential_sequential_generatedRoot_133_16 Backend.backend model
        GeneratedClaim.stage_2_params a i
        (s.1, (s.2.1, (s.2.2.1, (s.2.2.2.1, (s.2.2.2.2.1, (key, (keyDigits 10 A a 0,
          (keyDigits 10 A a 1, (z0, (z1, ())))))))))) t →
      (∀ w, t.1 0 w = s.1 0 w + rhsKeySum2 model key A 0 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧
        (∀ w, t.2.1 0 w = s.2.1 0 w + rhsKeySum2 model key A 1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧
        (∀ w, t.2.2.1 0 w = s.2.2.1 0 w + rhsCompanionSum2 A z0 z1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧ Q (i + 1) t := by
    intro i s t hi hq hb
    have hq' : s.2.2.2.1 = z0 ⟨i + 16 * a, by omega⟩ ∧ s.2.2.2.2.1 = z1 ⟨i + 16 * a, by omega⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext (by simp [Nat.mod_eq_of_lt hi]))),
        h1.trans (congrArg z1 (Fin.ext (by simp [Nat.mod_eq_of_lt hi])))⟩
    obtain ⟨h0, h1, h2, h3, h4⟩ := rhsInner2_step (U := keyDigits 10 A a) a.isLt hi hq' hb
    refine ⟨fun w ↦ ?_, fun w ↦ ?_, fun w ↦ ?_, h3, h4⟩
    · rw [h0]
      rfl
    · rw [h1]
      rfl
    · rw [h2]
      rfl
  have hk0 := IterRuns.accumulate_entries (fun s : RhsBlock2 ↦ s.1)
    (fun w (j : Fin 16) ↦ rhsKeySum2 model key A 0 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1, (hstep i s t hi hq hb).2.2.2⟩) h jj
  have hk1 := IterRuns.accumulate_entries (fun s : RhsBlock2 ↦ s.2.1)
    (fun w (j : Fin 16) ↦ rhsKeySum2 model key A 1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).2.1, (hstep i s t hi hq hb).2.2.2⟩) h jj
  have hk2 := IterRuns.accumulate_entries (fun s : RhsBlock2 ↦ s.2.2.1)
    (fun w (j : Fin 16) ↦ rhsCompanionSum2 A z0 z1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).2.2.1, (hstep i s t hi hq hb).2.2.2⟩) h jj
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0, sum_unitRow_mul (fun j ↦ rhsKeySum2 model key A 0 j a) jj]
    simp
  · rw [hk1, sum_unitRow_mul (fun j ↦ rhsKeySum2 model key A 1 j a) jj]
    simp
  · rw [hk2, sum_unitRow_mul (fun j ↦ rhsCompanionSum2 A z0 z1 j a) jj]
    simp

set_option maxHeartbeats 16000000 in
/-- One iteration `a` of the RHS companion loop of round `2` decomposes key column `a` once and
adds the companion keys and the companion row of digit row `a`, column by column. -/
theorem rhsLoop2_step {model : HashModel} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10} {A0 A1 : ExactMatrix Q5 N 1 20}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 320 → ExactMatrix Q5 N 1 10} {i : Nat} (hi : i < 20) {s t : RhsState2}
    (h : Stage_online_eval.sequential_generatedRoot_133 Backend.backend model
      GeneratedClaim.stage_2_params i (s.1, s.2.1, s.2.2.1, z0, z1, key, A0, A1, ()) t)
    (jj : Fin 16) :
    t.1 0 jj = s.1 0 jj + rhsKeySum2 model key A 0 jj ⟨i, hi⟩ ∧
      t.2.1 0 jj = s.2.1 0 jj + rhsKeySum2 model key A 1 jj ⟨i, hi⟩ ∧
      t.2.2.1 0 jj = s.2.2.1 0 jj + rhsCompanionSum2 A z0 z1 jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, x6, -, -, x8, -, -, -, -, -, -, -, -, x11,
    x12, -, -, -, -, -, -, -, -, x14, x15, -, hin, x17, x20, x23, xout⟩ := h
  have hdiv : Int.fdiv (Int.ofNat i) 20 = 0 := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 20 = ((i : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 16 * Int.fmod (Int.ofNat i) 20 = Int.ofNat (16 * i) := by
    rw [hrow]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hdiv] at x2
  rw [hrow] at x11 x14
  rw [hfirst] at x6 x8
  have hw2 := unitRow_window (c := 0) (by omega) x2
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨_, hw8⟩ := familyGet_eq x8
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q5 N 1 20} {x : ExactMatrix Q5 N 1 1}
      {u : ExactMatrix Q5 N 10 1}, concatColumns (A r 0) (A r 1) Ar →
      sliceMatrix Ar 0 1 ((i : Nat) : Int) (1 + ((i : Nat) : Int)) x →
      gadgetDecomposeRuns Backend.backend 1048576 10 x u → u = keyDigits 10 A ⟨i, hi⟩ r := by
    intro r Ar x u hAr hx hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    have hxe : x = fun _ _ ↦ joinedEntry (A r) ⟨i, hi⟩ := by
      funext a b
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0]
      congr 1
    rw [hxe] at hu
    exact entryDigits_eq hu
  have hw12 := hentry hA0 x11 x12
  have hw15 := hentry hA1 x14 x15
  rw [hw12, hw15, hw6, hw8] at hin
  have hblock := rhsInner2 (key := key) A ⟨i, hi⟩ hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q5 N 1 16}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  refine ⟨?_, ?_, ?_⟩
  · simp only [matrixAdd, Matrix.add_apply]
    rw [ht x17]
    exact congrArg _ hblock.1
  · simp only [matrixAdd, Matrix.add_apply]
    rw [ht x20]
    exact congrArg _ hblock.2.1
  · simp only [matrixAdd, Matrix.add_apply]
    rw [ht x23]
    exact congrArg _ hblock.2.2

set_option maxHeartbeats 8000000 in
/-- The RHS companion loop of round `2` ends with the companion keys and the companion row, column by
column. -/
theorem rhsLoop2 {model : HashModel} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 10} {A0 A1 : ExactMatrix Q5 N 1 20}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 320 → ExactMatrix Q5 N 1 10} {final : RhsState2}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : RhsState2) ↦
      Stage_online_eval.sequential_generatedRoot_133 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, (current.2.1, (current.2.2.1, (z0, (z1, (key, (A0, (A1, ())))))))) next)
      (Int.toNat 20) (0, (0, (0, ()))) final) (jj : Fin 16) :
    final.1 0 jj = ∑ a, rhsKeySum2 model key A 0 jj a ∧
      final.2.1 0 jj = ∑ a, rhsKeySum2 model key A 1 jj a ∧
      final.2.2.1 0 jj = ∑ a, rhsCompanionSum2 A z0 z1 jj a := by
  have hcount : Int.toNat 20 = 20 := rfl
  rw [hcount] at h
  have hk0 := IterRuns.accumulate_entries (fun s : RhsState2 ↦ s.1)
    (fun w a ↦ rhsKeySum2 model key A 0 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsLoop2_step hA0 hA1 hi hb w).1, trivial⟩) h jj
  have hk1 := IterRuns.accumulate_entries (fun s : RhsState2 ↦ s.2.1)
    (fun w a ↦ rhsKeySum2 model key A 1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsLoop2_step hA0 hA1 hi hb w).2.1, trivial⟩) h jj
  have hk2 := IterRuns.accumulate_entries (fun s : RhsState2 ↦ s.2.2.1)
    (fun w a ↦ rhsCompanionSum2 A z0 z1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsLoop2_step hA0 hA1 hi hb w).2.2, trivial⟩) h jj
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0]
    simp
  · rw [hk1]
    simp
  · rw [hk2]
    simp

/-- The digits of column `jj` of the joined round-`0` RHS ciphertext `(top, bot)`. -/
noncomputable abbrev rhsDigits2 (top bot : ExactMatrix Q5 N 1 16) (jj : Fin 16) :
    ExactMatrix Q5 N (10 + 10) 1 :=
  columnDigits Q5 (10 + 10) Backend.backend 1048576 10 524288 (top 0 jj) (bot 0 jj)

set_option maxHeartbeats 8000000 in
/-- The column loop of round `2` ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoop2 {model : HashModel} {c : ExactMatrix Q5 N 1 20} {top bot : ExactMatrix Q5 N 1 16}
    {final : ExactMatrix Q5 N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 N 1 16) ↦
      Stage_online_eval.sequential_generatedRoot_151 Backend.backend model GeneratedClaim.stage_2_params i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = (c * rhsDigits2 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
        gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (rhsDigits2 top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : Fin 16 → Fin 1 → ExactPoly Q5 N := fun jj _ ↦ (c * rhsDigits2 top bot jj) 0 0
  let Q : Nat → ExactMatrix Q5 N 1 16 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (rhsDigits2 top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q5 N 1 16 ↦ s)
    (fun i ↦ if h : i < 16 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q5 N 1 16) else 0) Q
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 16 := by omega
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

/-- The state of the LUT companion loop of round `2`: the substituted companion keys and the
companion rows of every branch. -/
abbrev LutState2 := ExactMatrix Q5 N 1 256 × ExactMatrix Q5 N 1 256 × ExactMatrix Q5 N 1 256 × Unit

/-- The state of the inner LUT companion loop of round `2`: the blocks of one branch and digit row,
and the carried companion. -/
abbrev LutBlock2 := ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 ×
  ExactMatrix Q5 N 1 10 × ExactMatrix Q5 N 1 10 × Unit

/-- Companion `k` of the round-`0` LUT, substituted by the automorphism of its branch `k / 256`. -/
noncomputable def lutCarry2 (z : Fin 4096 → ExactMatrix Q5 N 1 10) (k : Fin 4096) :
    ExactMatrix Q5 N 1 10 :=
  substituteMatrix (sigma (k / 16 / 16)).toNat (sigma_odd ⟨k / 16 / 16, by omega⟩) (z k)

/-- The digit-row sum of the substituted LUT companion keys `σB_k0 u_0 + σB_k1 u_1` of round `2`
at position `p` (branch `p / 16`, column `p % 16`). -/
noncomputable def lutKeySum2 (model : HashModel) (key : ByteArray)
    (M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8) (k : Fin 2) (p : Fin 256) (a : Fin 16) :
    ExactPoly Q5 N :=
  (substituteMatrix (sigma (p / 16)).toNat (sigma_odd ⟨p / 16, by omega⟩)
      (digitKey model key Q5 10 (lutOwner 50 ((p / 16 : Nat) : Int) 5) ((p % 16 : Nat) : Int)
        ((a : Nat) : Int) k 0) *
      keyDigits 10 (fun r j ↦ substituteMatrix (sigma (p / 16)).toNat
        (sigma_odd ⟨p / 16, by omega⟩) (M r j)) a 0 +
    substituteMatrix (sigma (p / 16)).toNat (sigma_odd ⟨p / 16, by omega⟩)
      (digitKey model key Q5 10 (lutOwner 50 ((p / 16 : Nat) : Int) 5) ((p % 16 : Nat) : Int)
        ((a : Nat) : Int) k 1) *
      keyDigits 10 (fun r j ↦ substituteMatrix (sigma (p / 16)).toNat
        (sigma_odd ⟨p / 16, by omega⟩) (M r j)) a 1) 0 0

/-- The digit-row sum of the substituted LUT companions `σz_0 u_0 + σz_1 u_1` of round `2`: the
companion of branch `p / 16`, digit row `a`, and column `p % 16` is `16 (16 (p / 16) + a) + p % 16`. -/
noncomputable def lutCompanionSum2 (M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8)
    (z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10) (p : Fin 256) (a : Fin 16) : ExactPoly Q5 N :=
  (lutCarry2 z0 ⟨16 * (16 * (p / 16) + a) + p % 16, by omega⟩ *
      keyDigits 10 (fun r j ↦ substituteMatrix (sigma (p / 16)).toNat
        (sigma_odd ⟨p / 16, by omega⟩) (M r j)) a 0 +
    lutCarry2 z1 ⟨16 * (16 * (p / 16) + a) + p % 16, by omega⟩ *
      keyDigits 10 (fun r j ↦ substituteMatrix (sigma (p / 16)).toNat
        (sigma_odd ⟨p / 16, by omega⟩) (M r j)) a 1) 0 0

set_option maxHeartbeats 16000000 in
/-- One iteration `j` of the inner LUT companion loop of round `2` at pair `i` (branch `i / 16`,
digit row `i % 16`) adds, at column `j`, the substituted companion keys and companion `16 i + j`
paired with the digits `U`, and reads the next companion of that pair. -/
theorem lutInner2_step {model : HashModel} {key : ByteArray} {U : Fin 2 → ExactMatrix Q5 N 10 1}
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10} {i j : Nat} (hi : i < 256) (hj : j < 16)
    {s t : LutBlock2}
    (hs : s.2.2.2.1 = lutCarry2 z0 ⟨16 * i + j, by omega⟩ ∧
      s.2.2.2.2.1 = lutCarry2 z1 ⟨16 * i + j, by omega⟩)
    (h : Stage_online_eval.sequential_sequential_generatedRoot_140_20 Backend.backend model
      GeneratedClaim.stage_2_params i j
      (s.1, s.2.1, s.2.2.1, s.2.2.2.1, s.2.2.2.2.1, key, U 0, U 1, z0, z1, ()) t) :
    let B := fun k r ↦ substituteMatrix (sigma (i / 16)).toNat (sigma_odd ⟨i / 16, by omega⟩)
      (digitKey model key Q5 10 (lutOwner 50 ((i / 16 : Nat) : Int) 5) ((j : Nat) : Int)
        ((i % 16 : Nat) : Int) k r)
    t.1 = s.1 + (B 0 0 * U 0 + B 0 1 * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (B 1 0 * U 0 + B 1 1 * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.2.1 = s.2.2.1 + (lutCarry2 z0 ⟨16 * i + j, by omega⟩ * U 0 +
        lutCarry2 z1 ⟨16 * i + j, by omega⟩ * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.2.2.1 = lutCarry2 z0 ⟨16 * i + (j + 1) % 16, by omega⟩ ∧
      t.2.2.2.2.1 = lutCarry2 z1 ⟨16 * i + (j + 1) % 16, by omega⟩ := by
  intro B
  obtain ⟨w2, w3, w6, w7, w12, w16, w17, w19, w20, w35, w36, w38, w39, -, -, x2, -, -, -, x3, -, -,
    x6, -, -, -, x7, -, -, -, -, -, -, -, -, x12, -, -, x16, -, -, -, x17, -, -, x19, -, -, -, x20,
    -, -, -, -, -, x35, -, -, -, x36, -, -, x38, -, -, -, x39, xout⟩ := h
  have hmem : Int.fdiv (Int.ofNat i) 16 = ((i / 16 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 16 = ((i % 16 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hnext : 256 * Int.fdiv (Int.ofNat i) 16 + 16 * Int.fmod (Int.ofNat i) 16 +
      Int.fmod (1 + Int.ofNat j) 16 = Int.ofNat (16 * i + (j + 1) % 16) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num), Int.ofNat_eq_natCast]
    omega
  have hnmem : ((16 * i + (j + 1) % 16) / 16 / 16 : Nat) = i / 16 := by omega
  rw [hnext] at x35 x38
  rw [hmem, hrow] at x2 x6 x16 x19
  rw [hmem] at x3 x7 x17 x20 x36 x39
  obtain ⟨_, hw35⟩ := familyGet_eq x35
  obtain ⟨_, hw38⟩ := familyGet_eq x38
  have hw12 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x12
  obtain ⟨_, hw3⟩ := ringAutomorphismRuns_eq hQ5 hN x3
  obtain ⟨_, hw7⟩ := ringAutomorphismRuns_eq hQ5 hN x7
  obtain ⟨_, hw17⟩ := ringAutomorphismRuns_eq hQ5 hN x17
  obtain ⟨_, hw20⟩ := ringAutomorphismRuns_eq hQ5 hN x20
  obtain ⟨_, hw36⟩ := ringAutomorphismRuns_eq hQ5 hN x36
  obtain ⟨_, hw39⟩ := ringAutomorphismRuns_eq hQ5 hN x39
  have hK : ∀ k r, digitKey model key Q5 10 (lutOwner 50 ((i / 16 : Nat) : Int) 5)
      ((j : Nat) : Int) ((i % 16 : Nat) : Int) k r = ![![w2, w6], ![w16, w19]] k r := by
    intro k r
    fin_cases k <;> fin_cases r <;>
      simp [digitKey, digitKeyPrefix, digitKeyHead, lutOwner, componentBytes, hashSample_eq x2,
        hashSample_eq x6, hashSample_eq x16, hashSample_eq x19]
  subst hw3 hw7 hw17 hw20
  subst xout hw12 hw35 hw38 hw36 hw39
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [B, hK, matrixAdd, matrixMul, matrixMulScalarLeft_eq, Matrix.cons_val_zero,
      Matrix.cons_val_one]
    rfl
  · simp only [B, hK, matrixAdd, matrixMul, matrixMulScalarLeft_eq, Matrix.cons_val_zero,
      Matrix.cons_val_one]
    rfl
  · simp only [hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq]
  · simp only [lutCarry2, sigma_congr (by omega) (by omega) hnmem]
    rfl
  · simp only [lutCarry2, sigma_congr (by omega) (by omega) hnmem]
    rfl

set_option maxHeartbeats 16000000 in
/-- The inner LUT companion loop of round `2` at pair `i` ends with the substituted companion keys
and the companion row of branch `i / 16` and digit row `i % 16`, column by column. -/
theorem lutInner2 {model : HashModel} {key : ByteArray} (M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8)
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10} {i : Nat} (hi : i < 256) {final : LutBlock2}
    (h : MxxIR.IterRuns (fun (j : Nat) (current next : LutBlock2) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_140_20 Backend.backend model
        GeneratedClaim.stage_2_params i j
        (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (key,
          (keyDigits 10 (fun r j ↦ substituteMatrix (sigma (i / 16)).toNat
            (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ 0,
          (keyDigits 10 (fun r j ↦ substituteMatrix (sigma (i / 16)).toNat
            (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ 1, (z0, (z1, ()))))))))))
        next) (Int.toNat 16)
      (0, (0, (0, (lutCarry2 z0 ⟨16 * i, by omega⟩, (lutCarry2 z1 ⟨16 * i, by omega⟩, ()))))) final)
    (jj : Fin 16) :
    final.1 0 jj = lutKeySum2 model key M 0 ⟨16 * (i / 16) + jj, by omega⟩ ⟨i % 16, by omega⟩ ∧
      final.2.1 0 jj = lutKeySum2 model key M 1 ⟨16 * (i / 16) + jj, by omega⟩ ⟨i % 16, by omega⟩ ∧
      final.2.2.1 0 jj =
        lutCompanionSum2 M z0 z1 ⟨16 * (i / 16) + jj, by omega⟩ ⟨i % 16, by omega⟩ := by
  have hp : ∀ j : Fin 16, (16 * (i / 16) + j) / 16 = i / 16 ∧ (16 * (i / 16) + j) % 16 = j :=
    fun j ↦ ⟨by omega, by omega⟩
  let Q : Nat → LutBlock2 → Prop := fun j s ↦
    s.2.2.2.1 = lutCarry2 z0 ⟨16 * i + j % 16, by omega⟩ ∧
      s.2.2.2.2.1 = lutCarry2 z1 ⟨16 * i + j % 16, by omega⟩
  have hQ0 : Q 0 (0, (0, (0, (lutCarry2 z0 ⟨16 * i, by omega⟩, (lutCarry2 z1 ⟨16 * i, by omega⟩,
      ()))))) :=
    ⟨congrArg (lutCarry2 z0) (Fin.ext (by simp)), congrArg (lutCarry2 z1) (Fin.ext (by simp))⟩
  have hstep : ∀ j (s t : LutBlock2) (hj : j < 16), Q j s →
      Stage_online_eval.sequential_sequential_generatedRoot_140_20 Backend.backend model
        GeneratedClaim.stage_2_params i j
        (s.1, (s.2.1, (s.2.2.1, (s.2.2.2.1, (s.2.2.2.2.1, (key,
          (keyDigits 10 (fun r j ↦ substituteMatrix (sigma (i / 16)).toNat
            (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ 0,
          (keyDigits 10 (fun r j ↦ substituteMatrix (sigma (i / 16)).toNat
            (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ 1, (z0, (z1, ()))))))))))
        t →
      (∀ w, t.1 0 w = s.1 0 w +
          lutKeySum2 model key M 0 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
            (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧
        (∀ w, t.2.1 0 w = s.2.1 0 w +
          lutKeySum2 model key M 1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
            (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧
        (∀ w, t.2.2.1 0 w = s.2.2.1 0 w +
          lutCompanionSum2 M z0 z1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
            (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) ∧ Q (j + 1) t := by
    intro j s t hj hq hb
    have hq' : s.2.2.2.1 = lutCarry2 z0 ⟨16 * i + j, by omega⟩ ∧
        s.2.2.2.2.1 = lutCarry2 z1 ⟨16 * i + j, by omega⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg (lutCarry2 z0) (Fin.ext (by simp [Nat.mod_eq_of_lt hj]))),
        h1.trans (congrArg (lutCarry2 z1) (Fin.ext (by simp [Nat.mod_eq_of_lt hj])))⟩
    obtain ⟨h0, h1, h2, h3, h4⟩ := lutInner2_step hi hj hq' hb
    have e1 : (16 * (i / 16) + j) / 16 = i / 16 := by omega
    have e2 : (16 * (i / 16) + j) % 16 = j := by omega
    have e3 : (⟨16 * (16 * ((16 * (i / 16) + j) / 16) + i % 16) + (16 * (i / 16) + j) % 16,
        by omega⟩ : Fin 4096) = ⟨16 * i + j, by omega⟩ := Fin.ext (by simp only; omega)
    refine ⟨fun w ↦ ?_, fun w ↦ ?_, fun w ↦ ?_, h3, h4⟩
    · rw [h0]
      simp only [lutKeySum2, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
        sigma_congr (by omega) (by omega) e1, e1, e2]
    · rw [h1]
      simp only [lutKeySum2, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
        sigma_congr (by omega) (by omega) e1, e1, e2]
    · rw [h2]
      simp only [lutCompanionSum2, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
        sigma_congr (by omega) (by omega) e1, e3]
  have hk0 := IterRuns.accumulate_entries (fun s : LutBlock2 ↦ s.1)
    (fun w (j : Fin 16) ↦ lutKeySum2 model key M 0 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun j s t hj hq hb ↦ ⟨(hstep j s t hj hq hb).1, (hstep j s t hj hq hb).2.2.2⟩) h jj
  have hk1 := IterRuns.accumulate_entries (fun s : LutBlock2 ↦ s.2.1)
    (fun w (j : Fin 16) ↦ lutKeySum2 model key M 1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun j s t hj hq hb ↦ ⟨(hstep j s t hj hq hb).2.1, (hstep j s t hj hq hb).2.2.2⟩) h jj
  have hk2 := IterRuns.accumulate_entries (fun s : LutBlock2 ↦ s.2.2.1)
    (fun w (j : Fin 16) ↦ lutCompanionSum2 M z0 z1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩ *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q5 N 1 16) 0 w) Q hQ0
    (fun j s t hj hq hb ↦ ⟨(hstep j s t hj hq hb).2.2.1, (hstep j s t hj hq hb).2.2.2⟩) h jj
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0, sum_unitRow_mul (fun j : Fin 16 ↦
      lutKeySum2 model key M 0 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩) jj]
    simp
  · rw [hk1, sum_unitRow_mul (fun j : Fin 16 ↦
      lutKeySum2 model key M 1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩) jj]
    simp
  · rw [hk2, sum_unitRow_mul (fun j : Fin 16 ↦
      lutCompanionSum2 M z0 z1 ⟨16 * (i / 16) + j, by omega⟩ ⟨i % 16, by omega⟩) jj]
    simp

set_option maxHeartbeats 16000000 in
/-- One iteration `i` of the LUT companion loop of round `2` decomposes the substituted key column
`i % 16` of branch `i / 16` once and adds, at the columns of that branch, its substituted companion
keys and companion row. -/
theorem lutLoop2_step {model : HashModel} {key : ByteArray}
    {M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {A0 A1 : ExactMatrix Q5 N 1 16}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0) (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10} {i : Nat} (hi : i < 256) {s t : LutState2}
    (h : Stage_online_eval.sequential_generatedRoot_140 Backend.backend model
      GeneratedClaim.stage_2_params i (s.1, s.2.1, s.2.2.1, z0, z1, key, A0, A1, ()) t)
    (p : Fin 256) :
    t.1 0 p = s.1 0 p + (if (p : Nat) / 16 = i / 16 then
        lutKeySum2 model key M 0 p ⟨i % 16, by omega⟩ else 0) ∧
      t.2.1 0 p = s.2.1 0 p + (if (p : Nat) / 16 = i / 16 then
        lutKeySum2 model key M 1 p ⟨i % 16, by omega⟩ else 0) ∧
      t.2.2.1 0 p = s.2.2.1 0 p + (if (p : Nat) / 16 = i / 16 then
        lutCompanionSum2 M z0 z1 p ⟨i % 16, by omega⟩ else 0) := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, -, x6, -, -, -, x7, -, -, x9, -, -, -, x10,
    -, -, -, -, -, -, -, -, x13, -, -, -, x14, x15, -, -, -, -, -, -, -, -, x17, -, -, -, x18, x19, -,
    hin, x21, x24, x27, xout⟩ := h
  have hmem : Int.fdiv (Int.ofNat i) 16 = ((i / 16 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 16 = ((i % 16 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 256 * Int.fdiv (Int.ofNat i) 16 + 16 * Int.fmod (Int.ofNat i) 16 =
      Int.ofNat (16 * i) := by
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num),
      Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hfirst] at x6 x9
  rw [hmem] at x2 x7 x10 x14 x18
  rw [hrow] at x13 x17
  have hw2 := unitRow_window (c := ((i / 16 : Nat) : Int)) (by omega) x2
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨_, hw9⟩ := familyGet_eq x9
  obtain ⟨_, hw7⟩ := ringAutomorphismRuns_eq hQ5 hN x7
  obtain ⟨_, hw10⟩ := ringAutomorphismRuns_eq hQ5 hN x10
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q5 N 1 16} {x y : ExactMatrix Q5 N 1 1}
      {u : ExactMatrix Q5 N 10 1}, concatColumns (M r 0) (M r 1) Ar →
      sliceMatrix Ar 0 1 ((i % 16 : Nat) : Int) (1 + ((i % 16 : Nat) : Int)) x →
      ringAutomorphismRuns (sigma (i / 16)) x y →
      gadgetDecomposeRuns Backend.backend 1048576 10 y u →
      u = keyDigits 10 (fun r j ↦ substituteMatrix (sigma (i / 16)).toNat
        (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ r := by
    intro r Ar x y u hAr hx hy hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    obtain ⟨_, hy0⟩ := ringAutomorphismRuns_eq hQ5 hN hy
    have hye : y = fun _ _ ↦ joinedEntry (fun j ↦ substituteMatrix (sigma (i / 16)).toNat
        (sigma_odd ⟨i / 16, by omega⟩) (M r j)) ⟨i % 16, by omega⟩ := by
      funext a b
      rw [hy0]
      dsimp only
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0, joinedEntry_substitute]
      rfl
    rw [hye] at hu
    exact entryDigits_eq hu
  have hw15 := hentry hA0 x13 x14 x15
  have hw19 := hentry hA1 x17 x18 x19
  have hc0 : w.w_7_0 = lutCarry2 z0 ⟨16 * i, by omega⟩ := by
    rw [hw7, hw6]
    simp only [lutCarry2, sigma_congr (by omega) (by omega)
      (show (16 * i) / 16 / 16 = i / 16 by omega)]
    rfl
  have hc1 : w.w_10_0 = lutCarry2 z1 ⟨16 * i, by omega⟩ := by
    rw [hw10, hw9]
    simp only [lutCarry2, sigma_congr (by omega) (by omega)
      (show (16 * i) / 16 / 16 = i / 16 by omega)]
    rfl
  rw [hw15, hw19, hc0, hc1] at hin
  have hblock := lutInner2 (key := key) M hi hin
  rw [hw2] at x21 x24 x27
  have hp : (p : Nat) / 16 = i / 16 →
      (⟨16 * (i / 16) + (p : Nat) % 16, by omega⟩ : Fin 256) = p := fun h ↦ Fin.ext (by
    simp only
    omega)
  subst xout
  refine ⟨?_, ?_, ?_⟩
  · simp only [matrixAdd, Matrix.add_apply]
    rw [tensorRuns_unitRow x21 p]
    split_ifs with hpi
    · have e := (hblock ⟨(p : Nat) % 16, by omega⟩).1
      dsimp only at e
      rw [e, hp hpi]
    · rfl
  · simp only [matrixAdd, Matrix.add_apply]
    rw [tensorRuns_unitRow x24 p]
    split_ifs with hpi
    · have e := (hblock ⟨(p : Nat) % 16, by omega⟩).2.1
      dsimp only at e
      rw [e, hp hpi]
    · rfl
  · simp only [matrixAdd, Matrix.add_apply]
    rw [tensorRuns_unitRow x27 p]
    split_ifs with hpi
    · have e := (hblock ⟨(p : Nat) % 16, by omega⟩).2.2
      dsimp only at e
      rw [e, hp hpi]
    · rfl

set_option maxHeartbeats 16000000 in
/-- The LUT companion loop of round `2` ends with the substituted companion keys and companion rows
of every branch, position by position. -/
theorem lutLoop2 {model : HashModel} {key : ByteArray}
    {M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {A0 A1 : ExactMatrix Q5 N 1 16}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0) (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10} {final : LutState2}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : LutState2) ↦
      Stage_online_eval.sequential_generatedRoot_140 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, (current.2.1, (current.2.2.1, (z0, (z1, (key, (A0, (A1, ())))))))) next)
      (Int.toNat 256) (0, (0, (0, ()))) final)
    (p : Fin 256) :
    final.1 0 p = ∑ a, lutKeySum2 model key M 0 p a ∧
      final.2.1 0 p = ∑ a, lutKeySum2 model key M 1 p a ∧
      final.2.2.1 0 p = ∑ a, lutCompanionSum2 M z0 z1 p a := by
  have hcount : Int.toNat 256 = 16 * 16 := rfl
  rw [hcount] at h
  have hk0 := IterRuns.accumulate_entries (fun s : LutState2 ↦ s.1)
    (fun (w : Fin 256) (i : Fin (16 * 16)) ↦ if (i : Nat) / 16 = (w : Nat) / 16 then
      lutKeySum2 model key M 0 w ⟨(i : Nat) % 16, by omega⟩ else 0) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by
      dsimp only
      rw [(lutLoop2_step hA0 hA1 hi hb w).1]
      congr 1
      exact if_congr ⟨Eq.symm, Eq.symm⟩ rfl rfl, trivial⟩) h p
  have hk1 := IterRuns.accumulate_entries (fun s : LutState2 ↦ s.2.1)
    (fun (w : Fin 256) (i : Fin (16 * 16)) ↦ if (i : Nat) / 16 = (w : Nat) / 16 then
      lutKeySum2 model key M 1 w ⟨(i : Nat) % 16, by omega⟩ else 0) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by
      dsimp only
      rw [(lutLoop2_step hA0 hA1 hi hb w).2.1]
      congr 1
      exact if_congr ⟨Eq.symm, Eq.symm⟩ rfl rfl, trivial⟩) h p
  have hk2 := IterRuns.accumulate_entries (fun s : LutState2 ↦ s.2.2.1)
    (fun (w : Fin 256) (i : Fin (16 * 16)) ↦ if (i : Nat) / 16 = (w : Nat) / 16 then
      lutCompanionSum2 M z0 z1 w ⟨(i : Nat) % 16, by omega⟩ else 0) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by
      dsimp only
      rw [(lutLoop2_step hA0 hA1 hi hb w).2.2]
      congr 1
      exact if_congr ⟨Eq.symm, Eq.symm⟩ rfl rfl, trivial⟩) h p
  refine ⟨?_, ?_, ?_⟩
  · rw [hk0]
    simpa using sum_branch_entry (B := 16) (A := 16) (fun a ↦ lutKeySum2 model key M 0 p a)
      ⟨(p : Nat) / 16, by omega⟩
  · rw [hk1]
    simpa using sum_branch_entry (B := 16) (A := 16) (fun a ↦ lutKeySum2 model key M 1 p a)
      ⟨(p : Nat) / 16, by omega⟩
  · rw [hk2]
    simpa using sum_branch_entry (B := 16) (A := 16) (fun a ↦ lutCompanionSum2 M z0 z1 p a)
      ⟨(p : Nat) / 16, by omega⟩

/-- The digits, re-encoded at `Q5`, of column `jj` of the joined switching data `(top, bot)` at `Q4`. -/
noncomputable abbrev switchDigits2 (top bot : ExactMatrix Q4 N 1 16) (jj : Fin 16) :
    ExactMatrix Q5 N (8 + 8) 1 :=
  columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288 (top 0 jj) (bot 0 jj)

set_option maxHeartbeats 8000000 in
/-- The switching column loop of a round-`0` branch ends with the routed row times the digits of the
switching ciphertext, column by column, and every column decomposition runs. -/
theorem switchLoop2 {model : HashModel} {i0 : Nat} {c : ExactMatrix Q5 N 1 16}
    {top bot : ExactMatrix Q4 N 1 16} {final : ExactMatrix Q5 N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 N 1 16) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_155_44 Backend.backend model
        GeneratedClaim.stage_2_params i0 i (current, (c, (top, (bot, ())))) next) (Int.toNat 16) 0
      final) (jj : Fin 16) :
    final 0 jj = (c * switchDigits2 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q4 N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 1048576 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (switchDigits2 top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : Fin 16 → Fin 1 → ExactPoly Q5 N := fun jj _ ↦ (c * switchDigits2 top bot jj) 0 0
  let Q : Nat → ExactMatrix Q5 N 1 16 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q4 N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 1048576 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (switchDigits2 top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q5 N 1 16 ↦ s)
    (fun i ↦ if h : i < 16 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q5 N 1 16) else 0) Q
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 16 := by omega
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
abbrev HelperState2 := ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × ExactMatrix Q5 N 1 16 × Unit

set_option maxHeartbeats 8000000 in
/-- The helper column loop of a round-`0` branch ends with the helper row and its two key rows times
the digits of the moved-down companion keys, column by column, and every column decomposition
runs. -/
theorem helperLoop2 {model : HashModel} {i0 : Nat} {h0 k0 k1 : ExactMatrix Q5 N 1 16}
    {top bot : ExactMatrix Q4 N 1 16} {final : HelperState2}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : HelperState2) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_155_30 Backend.backend model
        GeneratedClaim.stage_2_params i0 i
        (current.1, (current.2.1, (current.2.2.1, (h0, (top, (bot, (k0, (k1, ())))))))) next)
      (Int.toNat 16) (0, (0, (0, ()))) final) (jj : Fin 16) :
    final.1 0 jj = (h0 * switchDigits2 top bot jj) 0 0 ∧
      final.2.1 0 jj = (k0 * switchDigits2 top bot jj) 0 0 ∧
      final.2.2.1 0 jj = (k1 * switchDigits2 top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q4 N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 1048576 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 524288 D' (switchDigits2 top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : ExactMatrix Q5 N 1 16 → Fin 16 → Fin 1 → ExactPoly Q5 N := fun x jj _ ↦
    (x * switchDigits2 top bot jj) 0 0
  let f : ExactMatrix Q5 N 1 16 → Nat → ExactMatrix Q5 N 1 16 := fun x i ↦ if h : i < 16 * 1 then
      F x ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q5 N 1 16) else 0
  let Q : Nat → HelperState2 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q4 N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 1048576 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 524288 D' (switchDigits2 top bot jj)
  have hstep : ∀ i (s t : HelperState2), i < 16 * 1 → Q i s →
      Stage_online_eval.sequential_sequential_generatedRoot_155_30 Backend.backend model
        GeneratedClaim.stage_2_params i0 i
        (s.1, (s.2.1, (s.2.2.1, (h0, (top, (bot, (k0, (k1, ())))))))) t →
      (t.1 = s.1 + f h0 i ∧ t.2.1 = s.2.1 + f k0 i ∧ t.2.2.1 = s.2.2.1 + f k1 i) ∧ Q (i + 1) t := by
    intro i s t hi hq hb
    obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
      -, -, -, -, -, -, x11, xout⟩ := hb
    have hi' : i < 16 := by omega
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
  obtain ⟨hk0, hQ⟩ := IterRuns.accumulate (fun s : HelperState2 ↦ s.1) (f h0) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.1, (hstep i s t hi hq hb).2⟩) h
  obtain ⟨hk1, -⟩ := IterRuns.accumulate (fun s : HelperState2 ↦ s.2.1) (f k0) Q hQ0
    (fun i s t hi hq hb ↦ ⟨(hstep i s t hi hq hb).1.2.1, (hstep i s t hi hq hb).2⟩) h
  obtain ⟨hk2, -⟩ := IterRuns.accumulate (fun s : HelperState2 ↦ s.2.2.1) (f k1) Q hQ0
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
noncomputable def lutDigits2 (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8) (σ : Fin 16)
    (j : Fin 2) : ExactMatrix Q5 N (8 + 8) 8 :=
  fun a l ↦ columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288 (fam 0 j σ 0 l)
    (fam 1 j σ 0 l) a 0

theorem lutKeySum2_eq {model : HashModel} {key : ByteArray}
    (M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8) (k : Fin 2) (σ : Fin 16) (p : Fin 256)
    (hp : (p : Nat) / 16 = σ) (a : Fin 16) :
    lutKeySum2 model key M k p a =
      (substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5) ((p % 16 : Nat) : Int)
            ((a : Nat) : Int) k 0) *
        keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
      substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5) ((p % 16 : Nat) : Int)
            ((a : Nat) : Int) k 1) *
        keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1) 0 0 := by
  unfold lutKeySum2
  simp only [sigma_congr (by omega) σ.isLt hp, hp]

theorem lutCompanionSum2_eq (M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8)
    (z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10) (σ : Fin 16) (p : Fin 256) (hp : (p : Nat) / 16 = σ)
    (c : Nat) (hc : (p : Nat) % 16 = c) (a : Fin 16) :
    lutCompanionSum2 M z0 z1 p a =
      (substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (z0 ⟨16 * (16 * (σ : Nat) + a) + c, by omega⟩) *
        keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
      substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (z1 ⟨16 * (16 * (σ : Nat) + a) + c, by omega⟩) *
        keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1) 0 0 := by
  have e2 : (16 * (16 * (σ : Nat) + a) + c) / 16 / 16 = σ := by omega
  have e3 : (⟨16 * (16 * ((p : Nat) / 16) + a) + (p : Nat) % 16, by omega⟩ : Fin 4096) =
      ⟨16 * (16 * (σ : Nat) + a) + c, by omega⟩ := Fin.ext (by simp only; omega)
  unfold lutCompanionSum2
  rw [e3]
  unfold lutCarry2
  simp only [sigma_congr (by omega) σ.isLt hp, sigma_congr (by omega) σ.isLt e2]

set_option maxHeartbeats 16000000 in
/-- The companion row of branch `σ` of the round-`0` LUT, read from the LUT companion loop, adds
`σ(s) (M_σ0 - σ(A_0) D_σ) + (M_σ1 - σ(A_1) D_σ)` and a small error. -/
theorem lutCompanionRows2 {model : HashModel} {key : ByteArray} {sm : ErrorPoly N}
    {g14' g14 : ExactMatrix Q5 N 1 10} (hg14' : gadgetMatrixRuns Backend.backend 1048576 10 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {M : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8}
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10}
    (hz : ∀ (s : Fin 16) (jj : Fin 16) (a : Fin 16),
      Rel sm (digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q5 N (unitPair sm r)) • g14)
        ![z0 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩,
          z1 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q5 N 1 256}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum2 model key M 0 p a ∧
      L1 0 p = ∑ a, lutKeySum2 model key M 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum2 M z0 z1 p a)
    (σ : Fin 16) :
    ∃ f : Fin 2 → ErrorMatrix N 1 8,
      (∀ j, CoeffBound (f j) ((8 + 8) * (10 * N * 26 * 524288 + 10 * N * 26 * 524288))) ∧
      ∀ j : Fin 2, rowWindow 8 (16 * σ + 8 * j) (by omega) L2 =
        reducePoly Q5 N (substitution (sigma σ).toNat (sigma_odd σ) sm) •
          (rowWindow 8 (16 * σ + 8 * j) (by omega) L0 -
            pairMul (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M 0 i))
              (lutDigits2 fam σ j)) +
        (rowWindow 8 (16 * σ + 8 * j) (by omega) L1 -
          pairMul (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M 1 i))
            (lutDigits2 fam σ j)) +
        reduceMatrix Q5 N 1 8 (f j) := by
  have hgg : g14 = g14' := centeredRebaseRuns_unique hg14 (centeredRebaseRuns_self hQ5 hN g14')
  subst hgg
  have hcop : N.Coprime (sigma σ).toNat := by
    rw [hpow]
    exact coprime_of_odd_pow_two (sigma_odd σ)
  have hdig := fun a r ↦ decompose_entry hQ5 hN Backend.layout2 layout5 rfl hg14'
    (entryDigits_runs hQ5 hN hg14' Backend.layout2 layout5 rfl
      (fun _ _ ↦ joinedEntry (fun i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a))
  choose u' hu' hβ using fun (a : Fin (8 + 8)) (r : Fin 2) ↦ (hdig a r).2
  have hgad : substituteMatrix (sigma σ).toNat (sigma_odd σ) g14 = g14 :=
    substituteMatrix_rebasedGadget hN hg14' (centeredRebaseRuns_self hQ5 hN _) _
  have hz' : ∀ (j : Fin 2) (l : Fin 8) (a : Fin (8 + 8)),
      Rel (substitution (sigma σ).toNat (sigma_odd σ) sm)
        (fun k r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
          (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5) ((8 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) k r))
        (fun r ↦ (lutDigits2 fam σ j a l *
          reducePoly Q5 N (unitPair (substitution (sigma σ).toNat (sigma_odd σ) sm) r)) • g14)
        ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩)] 26 := by
    intro j l a
    have h := (hz σ ⟨8 * (j : Nat) + l, by omega⟩ a).substitute hN (sigma_odd σ) hcop
    have hc : (fun r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (![z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩,
          z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩] r)) =
        ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩)] := by
      funext r
      fin_cases r <;> rfl
    have ht : (fun r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
        ((substitution (sigmaInv σ).toNat (sigmaInv_odd σ)
            (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
              (joinedEntry ![fam 0 0 σ, fam 0 1 σ] ⟨8 * (j : Nat) + l, by omega⟩)
              (joinedEntry ![fam 1 0 σ, fam 1 1 σ] ⟨8 * (j : Nat) + l, by omega⟩) a 0) *
          reducePoly Q5 N (unitPair sm r)) • g14)) =
        fun r ↦ (lutDigits2 fam σ j a l *
          reducePoly Q5 N (unitPair (substitution (sigma σ).toNat (sigma_odd σ) sm) r)) • g14 := by
      funext r
      rw [substituteMatrix_smul', hgad, map_mul,
        substitution_inverse (sigma_odd σ) (sigmaInv_odd σ) (sigma_mul_sigmaInv σ),
        ← reducePoly_substitution, substitution_unitPair, joinedEntry_pair, joinedEntry_pair,
        pair_eta (fun j ↦ fam 0 j σ), pair_eta (fun j ↦ fam 1 j σ)]
      rfl
    rw [hc, ht] at h
    exact h
  have hErow : ∀ (j : Fin 2) (l : Fin 8),
      rowWindow 8 (16 * σ + 8 * j) (by omega) L2 0 l =
        (∑ a : Fin (8 + 8), (![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩)] 0 *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩),
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
            (z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩)] 1 *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).2.2, Matrix.sum_apply]
    simp only [lutCompanionSum2_eq M z0 z1 σ ⟨16 * (σ : Nat) + 8 * (j : Nat) + l, by omega⟩
      (show (16 * (σ : Nat) + 8 * (j : Nat) + l) / 16 = σ by omega) (8 * (j : Nat) + l)
      (show (16 * (σ : Nat) + 8 * (j : Nat) + l) % 16 = 8 * (j : Nat) + l by omega)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  have hMrow0 : ∀ (j : Fin 2) (l : Fin 8),
      rowWindow 8 (16 * σ + 8 * j) (by omega) L0 0 l =
        (∑ a : Fin (8 + 8),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 0 0) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 0 1) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).1, Matrix.sum_apply]
    simp only [lutKeySum2_eq M _ σ ⟨16 * (σ : Nat) + 8 * (j : Nat) + l, by omega⟩
        (show (16 * (σ : Nat) + 8 * (j : Nat) + l) / 16 = σ by omega),
      show (16 * (σ : Nat) + 8 * (j : Nat) + l) % 16 = 8 * (j : Nat) + l by omega]
  have hMrow1 : ∀ (j : Fin 2) (l : Fin 8),
      rowWindow 8 (16 * σ + 8 * j) (by omega) L1 0 l =
        (∑ a : Fin (8 + 8),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 1 0) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) 1 1) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hL _).2.1, Matrix.sum_apply]
    simp only [lutKeySum2_eq M _ σ ⟨16 * (σ : Nat) + 8 * (j : Nat) + l, by omega⟩
        (show (16 * (σ : Nat) + 8 * (j : Nat) + l) / 16 = σ by omega),
      show (16 * (σ : Nat) + 8 * (j : Nat) + l) % 16 = 8 * (j : Nat) + l by omega]
  have hMrow : ∀ (k j : Fin 2) (l : Fin 8),
      rowWindow 8 (16 * σ + 8 * j) (by omega) (![L0, L1] k) 0 l =
        (∑ a : Fin (8 + 8),
          (substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) k 0) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 0 +
          substituteMatrix (sigma σ).toNat (sigma_odd σ)
              (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5)
                ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int) k 1) *
            keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a 1)) 0 0 := by
    intro k j l
    fin_cases k
    exacts [hMrow0 j l, hMrow1 j l]
  obtain ⟨f, hf, hE⟩ := companion_rows (q := Q5) (m := 8) (m' := 8) (Dg := 10) hN
    (sm := substitution (sigma σ).toNat (sigma_odd σ) sm) (g := g14)
    (A := fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i))
    (D := lutDigits2 fam σ)
    (B := fun j l a k r ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ)
      (digitKey model key Q5 10 (lutOwner 50 ((σ : Nat) : Int) 5) ((8 * (j : Nat) + l : Nat) : Int)
        ((a : Nat) : Int) k r))
    (z := fun j l a ↦ ![substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (z0 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩),
      substituteMatrix (sigma σ).toNat (sigma_odd σ)
        (z1 ⟨16 * (16 * (σ : Nat) + a) + (8 * (j : Nat) + l), by omega⟩)])
    (bound := 26) hz'
    (u := fun a r ↦ keyDigits 10 (fun r i ↦ substituteMatrix (sigma σ).toNat (sigma_odd σ) (M r i)) a r)
    (fun a r ↦ (hdig a r).1) (u' := u') (β := 524288) hu' hβ
    (E := fun j ↦ rowWindow 8 (16 * σ + 8 * j) (by omega) L2)
    (M := fun k j ↦ rowWindow 8 (16 * σ + 8 * j) (by omega) (![L0, L1] k)) hErow hMrow
  exact ⟨f, hf, hE⟩

/-- The state of the branch loop of round `2`: the accumulated row and key, and the carried
switching ciphertext. -/
abbrev BranchState2 := ExactMatrix Q5 N 1 8 × ExactMatrix Q5 N 1 8 × ExactMatrix Q5 N 1 8 ×
  ExactMatrix Q5 N 1 8 × ExactMatrix Q5 N 1 8 × ExactMatrix Q5 N 1 8 × ExactMatrix Q4 N 1 8 ×
    ExactMatrix Q4 N 1 8 × ExactMatrix Q4 N 1 8 × ExactMatrix Q4 N 1 8 × Unit

/-- The accumulated row. -/
def BranchState2.row (s : BranchState2) : Fin 2 → ExactMatrix Q5 N 1 8 := ![s.1, s.2.1]

/-- The accumulated key. -/
def BranchState2.key (s : BranchState2) : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8 :=
  ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]

/-- The carried switching ciphertext. -/
def BranchState2.switch (s : BranchState2) : Fin 2 → Fin 2 → ExactMatrix Q4 N 1 8 :=
  ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]]

/-- The target of branch `i` of the round-`0` LUT over the switching families `fam`. -/
noncomputable def branchTarget2 (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8) (i : Fin 16) :
    Fin 2 → ExactMatrix Q5 N 1 8 :=
  fun j ↦ scaleUpMatrix Q5 P5 (reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) μ) •
    (reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) sp) • fam 0 j i + fam 1 j i))

/-- The error bound added by one branch of the round-`0` LUT. -/
abbrev branchBound2 (B : Nat) : Nat :=
  (8 + 8) * N * 26 * 524288 + ((8 + 8) * N * B * 524288 +
    (8 + 8) * (10 * N * 26 * 524288 + 10 * N * 26 * 524288)) + (N * 1 * (P5 / 2) + P5 / 2)

set_option maxHeartbeats 64000000 in
/-- One iteration of the branch loop of round `2` adds branch `i` to the accumulated relation and
carries the switching ciphertext of branch `i + 1`. -/
theorem branchLoop2_step {model : HashModel} {key : ByteArray} {sm μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    {g14' g14 : ExactMatrix Q5 N 1 10} (hg14' : gadgetMatrixRuns Backend.backend 1048576 10 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {Ap : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm Ap (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) * μ * unitPair sm j) • g) c B)
    {h0 h1 : Fin 16 → ExactMatrix Q5 N 1 8}
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model key Q5 8 5 (sigma i))
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] 26)
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8}
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10}
    (hz : ∀ (s : Fin 16) (jj : Fin 16) (a : Fin 16),
      Rel sm (digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q5 N (unitPair sm r)) • g14)
        ![z0 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩,
          z1 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q5 N 1 256}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum2 model key Ap 0 p a ∧
      L1 0 p = ∑ a, lutKeySum2 model key Ap 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum2 Ap z0 z1 p a)
    {i : Nat} (hi : i < 16) {state next : BranchState2}
    (hswitch : state.switch = fun k j ↦ fam k j ⟨i, hi⟩)
    {T : Fin 2 → ExactMatrix Q5 N 1 8} {Bs : Nat} (hstate : Rel sm state.key T state.row Bs)
    (hbody : Stage_online_eval.sequential_generatedRoot_155 Backend.backend model
      GeneratedClaim.stage_2_params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, h1, L0, L1, key, c 0, c 1, L2, fam 0 0, fam 0 1, fam 1 0,
        fam 1 1, ()) next) :
    Rel sm next.key (T + branchTarget2 μ sm fam ⟨i, hi⟩) next.row (Bs + branchBound2 B) ∧
      next.switch = fun k j ↦ fam k j ⟨(i + 1) % 16, Nat.mod_lt _ (by norm_num)⟩ := by
  obtain ⟨w, -, x3, x4, -, -, x6, x7, -, -, -, -, -, -, x9, -, -, -, -, -, -, x10, x11, -, -, -, -, -, -, x12, x13, x14, -, -, -, -, -, -, x16, -, -, -, -, -, -, x17, x18, -, -, -, -, -, -, x19, x20, x21, -, -, x24, x25, x26, x27, x28, x29, -, xseq30, -, -, -, -, -, -, x31, -, -, x34, -, -, x36, x37, x40, x43, -, xseq44, -, -, -, -, -, -, x45, -, -, -, -, -, -, x47, -, -, -, -, -, -, x48, -, -, -, -, -, -, x53, -, -, -, -, -, -, x54, -, -, -, -, -, -, x55, -, -, -, -, -, -, x60, -, -, -, -, -, -, x63, -, -, -, -, -, -, x66, -, -, -, -, -, -, x69, -, -, -, x73, -, -, x75, -, -, x77, -, -, x79, xout⟩ := hbody
  have hs00 : state.2.2.2.2.2.2.1 = fam 0 0 ⟨i, hi⟩ := congrFun (congrFun hswitch 0) 0
  have hs01 : state.2.2.2.2.2.2.2.1 = fam 0 1 ⟨i, hi⟩ := congrFun (congrFun hswitch 0) 1
  have hs10 : state.2.2.2.2.2.2.2.2.1 = fam 1 0 ⟨i, hi⟩ := congrFun (congrFun hswitch 1) 0
  have hs11 : state.2.2.2.2.2.2.2.2.2.1 = fam 1 1 ⟨i, hi⟩ := congrFun (congrFun hswitch 1) 1
  obtain ⟨_, hw4⟩ := familyGet_eq (k := i) x4
  obtain ⟨_, hw6⟩ := familyGet_eq (k := i) x6
  obtain ⟨_, hw34⟩ := ringAutomorphismRuns_eq hQ5 hN x34
  obtain ⟨_, hw36⟩ := ringAutomorphismRuns_eq hQ5 hN x36
  -- The companion keys and row of branch `i`, read from the LUT companion loop.
  have h24 : (16 * Int.ofNat i : Int) = ((16 * i : Nat) : Int) := by simp [Int.ofNat_eq_natCast]
  have hM00 := sliceMatrix_window₂ (16 * i) (8 * 0) x9 x10 h24 (by simp) (by omega)
  have hM01 := sliceMatrix_window₂ (16 * i) (8 * 1) x9 x12 h24 (by simp) (by omega)
  have hM10 := sliceMatrix_window₂ (16 * i) (8 * 0) x16 x17 h24 (by simp) (by omega)
  have hM11 := sliceMatrix_window₂ (16 * i) (8 * 1) x16 x19 h24 (by simp) (by omega)
  have hE0 := sliceMatrix_window₂ (16 * i) (8 * 0) x47 x48 h24 (by simp) (by omega)
  have hE1 := sliceMatrix_window₂ (16 * i) (8 * 1) x47 x55 h24 (by simp) (by omega)
  obtain ⟨f, hf, hEq⟩ := lutCompanionRows2 hg14' hg14 hz hL ⟨i, hi⟩
  have hEσ : ∀ j, ![w.w_48_0, w.w_55_0] j =
      reducePoly Q5 N (substitution (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) sm) •
        (![![w.w_10_0, w.w_12_0], ![w.w_17_0, w.w_19_0]] 0 j -
          pairMul (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩)
            (Ap 0 i')) (lutDigits2 fam ⟨i, hi⟩ j)) +
      (![![w.w_10_0, w.w_12_0], ![w.w_17_0, w.w_19_0]] 1 j -
        pairMul (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩)
          (Ap 1 i')) (lutDigits2 fam ⟨i, hi⟩ j)) +
      reduceMatrix Q5 N 1 8 (f j) := by
    intro j
    fin_cases j
    · simp only [Fin.zero_eta, Matrix.cons_val_zero]
      rw [hE0, hM00, hM10]
      exact hEq 0
    · simp only [Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero]
      rw [hE1, hM01, hM11]
      exact hEq 1
  -- The switching action on the routed row.
  have hsw := switchLoop2 xseq44
  have hR37 : concatColumns
      (substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c 0))
      (substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c 1)) w.w_37_0 := by
    rw [hw34, hw36] at x37
    exact x37
  have hD40 : ∀ (j : Fin 2) (l : Fin 8) (h : 8 * (j : Nat) + l < 8 + 8),
      w.w_40_0 0 ⟨8 * (j : Nat) + l, h⟩ = fam 0 j ⟨i, hi⟩ 0 l := by
    intro j l h
    rw [concatColumns_entry x40, joinedEntry_pair]
    fin_cases j
    · exact congrFun (congrFun hs00 0) l
    · exact congrFun (congrFun hs01 0) l
  have hD43 : ∀ (j : Fin 2) (l : Fin 8) (h : 8 * (j : Nat) + l < 8 + 8),
      w.w_43_0 0 ⟨8 * (j : Nat) + l, h⟩ = fam 1 j ⟨i, hi⟩ 0 l := by
    intro j l h
    rw [concatColumns_entry x43, joinedEntry_pair]
    fin_cases j
    · exact congrFun (congrFun hs10 0) l
    · exact congrFun (congrFun hs11 0) l
  have hDdig : ∀ j : Fin 2, (fun (a : Fin (8 + 8)) (l : Fin 8) ↦ switchDigits2 w.w_40_0 w.w_43_0
      ⟨8 * (j : Nat) + l, by omega⟩ a 0 : ExactMatrix Q5 N (8 + 8) 8) = lutDigits2 fam ⟨i, hi⟩ j := by
    intro j
    funext a l
    simp only [switchDigits2, lutDigits2, hD40, hD43]
  have hX : ∀ j, ![w.w_45_0, w.w_54_0] j = pairMul
      (fun i' ↦ substituteMatrix (sigma (⟨i, hi⟩ : Fin 16)).toNat (sigma_odd ⟨i, hi⟩) (c i'))
      (lutDigits2 fam ⟨i, hi⟩ j) := by
    intro j
    rw [← hDdig j, ← window_mul_columns (acc := w.w_44_0) hR37 (fun p ↦ (hsw p).1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (8 * 0) x45 (by simp) (by omega)
    · exact sliceMatrix_window (8 * 1) x54 (by simp) (by omega)
  have hD : ∀ j l, ∃ D' : ExactMatrix Q4 N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 1048576 8
        (pairColumn (fam 0 j ⟨i, hi⟩ 0 l) (fam 1 j ⟨i, hi⟩ 0 l)) D' ∧
      compactCenteredRebaseRuns 524288 D' (fun a _ ↦ lutDigits2 fam ⟨i, hi⟩ j a l) := by
    intro j l
    obtain ⟨D', h1, h2⟩ := (hsw ⟨8 * (j : Nat) + l, by omega⟩).2
    rw [hD40, hD43] at h1
    refine ⟨D', h1, ?_⟩
    have hfun : (fun (a : Fin (8 + 8)) (_ : Fin 1) ↦ lutDigits2 fam ⟨i, hi⟩ j a l) =
        switchDigits2 w.w_40_0 w.w_43_0 ⟨8 * (j : Nat) + l, by omega⟩ := by
      funext a b
      rw [Subsingleton.elim b 0]
      simp only [switchDigits2, lutDigits2, hD40, hD43]
    rw [hfun]
    exact h2
  -- The helper action on the moved-down companion keys.
  have hhl := helperLoop2 xseq30
  have hA14 : ∀ (j : Fin 2) (l : Fin 8) (h : 8 * (j : Nat) + l < 8 + 8),
      w.w_14_0 0 ⟨8 * (j : Nat) + l, h⟩ = ![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 0 j 0 l := by
    intro j l h
    rw [concatColumns_entry x14, joinedEntry_pair]
    fin_cases j <;> rfl
  have hA21 : ∀ (j : Fin 2) (l : Fin 8) (h : 8 * (j : Nat) + l < 8 + 8),
      w.w_21_0 0 ⟨8 * (j : Nat) + l, h⟩ = ![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 1 j 0 l := by
    intro j l h
    rw [concatColumns_entry x21, joinedEntry_pair]
    fin_cases j <;> rfl
  have hEdec : ∀ j l, ∃ E' : ExactMatrix Q4 N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 1048576 8
        (pairColumn (![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 0 j 0 l)
          (![![w.w_11_0, w.w_13_0], ![w.w_18_0, w.w_20_0]] 1 j 0 l)) E' ∧
      compactCenteredRebaseRuns 524288 E' (fun a _ ↦ (fun (j : Fin 2) (a : Fin (8 + 8)) (l : Fin 8) ↦
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) j a l) := by
    intro j l
    obtain ⟨E', h1, h2⟩ := (hhl ⟨8 * (j : Nat) + l, by omega⟩).2.2.2
    rw [hA14, hA21] at h1
    refine ⟨E', h1, ?_⟩
    have hfun : (fun (a : Fin (8 + 8)) (_ : Fin 1) ↦
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) =
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ := by
      funext a b
      rw [Subsingleton.elim b 0]
    rw [hfun]
    exact h2
  have hY : ∀ j, ![w.w_31_0, w.w_53_0] j = pairMul ![w.w_4_0, w.w_6_0]
      ((fun (j : Fin 2) (a : Fin (8 + 8)) (l : Fin 8) ↦
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_4_0, w.w_6_0]) (acc := w.w_30_0) x7 (fun p ↦ (hhl p).1) j
      (by omega)]
    fin_cases j
    · exact sliceMatrix_window (8 * 0) x31 (by simp) (by omega)
    · exact sliceMatrix_window (8 * 1) x53 (by simp) (by omega)
  have hK0 : ∀ j, ![w.w_60_0, w.w_63_0] j = pairMul ![w.w_24_0, w.w_25_0]
      ((fun (j : Fin 2) (a : Fin (8 + 8)) (l : Fin 8) ↦
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_24_0, w.w_25_0]) (acc := w.w_30_1) x26
      (fun p ↦ (hhl p).2.1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (8 * 0) x60 (by simp) (by omega)
    · exact sliceMatrix_window (8 * 1) x63 (by simp) (by omega)
  have hK1 : ∀ j, ![w.w_66_0, w.w_69_0] j = pairMul ![w.w_27_0, w.w_28_0]
      ((fun (j : Fin 2) (a : Fin (8 + 8)) (l : Fin 8) ↦
        switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro j
    rw [← window_mul_columns (row := ![w.w_27_0, w.w_28_0]) (acc := w.w_30_2) x29
      (fun p ↦ (hhl p).2.2.1) j (by omega)]
    fin_cases j
    · exact sliceMatrix_window (8 * 0) x66 (by simp) (by omega)
    · exact sliceMatrix_window (8 * 1) x69 (by simp) (by omega)
  have hK : ∀ k j, ![![w.w_60_0, w.w_63_0], ![w.w_66_0, w.w_69_0]] k j =
      pairMul (![![w.w_24_0, w.w_25_0], ![w.w_27_0, w.w_28_0]] k)
        ((fun (j : Fin 2) (a : Fin (8 + 8)) (l : Fin 8) ↦
          switchDigits2 w.w_14_0 w.w_21_0 ⟨8 * (j : Nat) + l, by omega⟩ a 0) j) := by
    intro k j
    fin_cases k
    exacts [hK0 j, hK1 j]
  have hAd : lutA model key Q5 8 5 (sigma (⟨i, hi⟩ : Fin 16)) =
      ![![w.w_24_0, w.w_25_0], ![w.w_27_0, w.w_28_0]] := by
    funext k j
    fin_cases k <;> fin_cases j <;>
      simp [lutA, lutAPrefix, componentBytes, sigma, hashSample_eq x24, hashSample_eq x25,
        hashSample_eq x27, hashSample_eq x28]
  have hhelp' := hhelp ⟨i, hi⟩
  rw [hAd, show h0 ⟨i, hi⟩ = w.w_4_0 from hw4.symm, show h1 ⟨i, hi⟩ = w.w_6_0 from hw6.symm] at hhelp'
  -- The branch.
  have hbranch := lut_branch_dynamic (q := Q5) (q' := Q4) (p := P5) (m := 8) (m' := 8) (log := 5)
    (bound := 524288) hN hpow q5_eq hQ4 Backend.layout1 layout4 rfl hg' hg hsm henc
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
noncomputable def branchTargetN2 (μ sp : ErrorPoly N)
    (fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8) (i : Nat) : Fin 2 → ExactMatrix Q5 N 1 8 :=
  if h : i < 16 then branchTarget2 μ sp fam ⟨i, h⟩ else 0

set_option maxHeartbeats 16000000 in
/-- The branch loop of round `2` accumulates the sixteen branches. -/
theorem branchLoop2 {model : HashModel} {key : ByteArray} {sm μ : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    {g14' g14 : ExactMatrix Q5 N 1 10} (hg14' : gadgetMatrixRuns Backend.backend 1048576 10 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {c : Fin 2 → ExactMatrix Q5 N 1 8} {Ap : Fin 2 → Fin 2 → ExactMatrix Q5 N 1 8} {B : Nat}
    (henc : Rel sm Ap (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) * μ * unitPair sm j) • g) c B)
    {h0 h1 : Fin 16 → ExactMatrix Q5 N 1 8}
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model key Q5 8 5 (sigma i))
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![h0 i, h1 i] 26)
    {fam : Fin 2 → Fin 2 → Fin 16 → ExactMatrix Q4 N 1 8}
    {z0 z1 : Fin 4096 → ExactMatrix Q5 N 1 10}
    (hz : ∀ (s : Fin 16) (jj : Fin 16) (a : Fin 16),
      Rel sm (digitKey model key Q5 10 (lutOwner 50 ((s : Nat) : Int) 5) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
              (joinedEntry ![fam 0 0 s, fam 0 1 s] jj) (joinedEntry ![fam 1 0 s, fam 1 1 s] jj) a 0) *
          reducePoly Q5 N (unitPair sm r)) • g14)
        ![z0 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩,
          z1 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩] 26)
    {L0 L1 L2 : ExactMatrix Q5 N 1 256}
    (hL : ∀ p, L0 0 p = ∑ a, lutKeySum2 model key Ap 0 p a ∧
      L1 0 p = ∑ a, lutKeySum2 model key Ap 1 p a ∧ L2 0 p = ∑ a, lutCompanionSum2 Ap z0 z1 p a)
    {final : BranchState2}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : BranchState2) ↦
      Stage_online_eval.sequential_generatedRoot_155 Backend.backend model GeneratedClaim.stage_2_params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1,
          current.2.2.2.2.2.1, current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1,
          current.2.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.2.1, h0, h1, L0, L1, key, c 0, c 1, L2,
          fam 0 0, fam 0 1, fam 1 0, fam 1 1, ()) next) (Int.toNat 16)
      (0, 0, 0, 0, 0, 0, fam 0 0 0, fam 0 1 0, fam 1 0 0, fam 1 1 0, ()) final) :
    Rel sm final.key (∑ i ∈ Finset.range 16, branchTargetN2 μ sm fam i) final.row
      (16 * branchBound2 B) := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at hruns
  refine (MxxIR.IterRuns.invariant
    (Invariant := fun k (s : BranchState2) ↦ k ≤ 16 →
      Rel sm s.key (∑ i ∈ Finset.range k, branchTargetN2 μ sm fam i) s.row (k * branchBound2 B) ∧
        s.switch = fun k' j ↦ fam k' j ⟨k % 16, Nat.mod_lt _ (by norm_num)⟩) ?_ ?_ hruns le_rfl).1
  · intro _
    refine ⟨?_, ?_⟩
    · rw [Finset.range_zero, Finset.sum_empty, zero_mul]
      have hrow : (![0, 0] : Fin 2 → ExactMatrix Q5 N 1 8) = 0 := by
        funext j
        fin_cases j <;> rfl
      simp only [BranchState2.key, BranchState2.row, hrow]
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
    obtain ⟨hrel', hsw'⟩ := branchLoop2_step hsm hg' hg hg14' hg14 henc hhelp hz hL hi hsw2 hrel hbody
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, branchTargetN2, dif_pos hi, add_mul, one_mul]
    exact hrel'


/-- The error bound after the round-`0` multiplication. -/
abbrev mulBound2 : Nat :=
  (10 + 10) * N * roundBound1 * 524288 + (10 + 10) * (10 * N * 26 * 524288 + 10 * N * 26 * 524288) +
    N * 1 * 26

/-- The error bound of the round-`0` output at level `6`. -/
abbrev roundBound2 : Nat :=
  (16 * branchBound2 mulBound2 + (N + 2) * (P5 / 2)) / P5 + 16 * (N * 1 * 26)

set_option maxHeartbeats 128000000 in
/-- Round `0`: the state encoding of `μ` at level `4` times the Dynamic RHS of `ν` at level `5`, then
the LUT of table `2`, gives an encoding of `X^(T₀ arg)` at level `6` when `μ ν = X^arg`. -/
theorem round2 {model : HashModel}
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
    {g14' g14 : ExactMatrix Q5 N 1 10} (hg14' : gadgetMatrixRuns Backend.backend 1048576 10 g14')
    (hg14 : centeredRebaseRuns g14' g14)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    (hinput : Rel sm ![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]
      (fun j ↦ reducePoly Q5 N (1 * μ * unitPair sm j) • g14) ![w.w_142_0, w.w_143_0] roundBound1)
    (hrhs : ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N sm • ![v169, v170] j + ![v172, v173] j =
        reducePoly Q5 N (((P5 : Nat) : Int) * ν * unitPair sm j) • g + reduceMatrix Q5 N 1 8 (e j))
    (hrhsz : ∀ (jj : Fin 16) (a : Fin 20),
      Rel sm (digitKey model v18 Q5 10 (rhsOwner 50 4) ((jj : Nat) : Int) ((a : Nat) : Int))
        (fun r ↦ (columnDigits Q5 (10 + 10) Backend.backend 1048576 10 524288
            (joinedEntry ![v169, v170] jj) (joinedEntry ![v172, v173] jj) a 0 *
          reducePoly Q5 N (unitPair sm r)) • g14)
        ![v43 ⟨jj + 16 * a, by omega⟩, v45 ⟨jj + 16 * a, by omega⟩] 26)
    (hhelp : ∀ i : Fin 16, Rel sm (lutA model v18 Q5 8 5 (sigma i))
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![v33 i, v34 i] 26)
    (hswitch : ∀ i : Fin 16, ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) sm) • ![v25 i, v27 i] j +
          ![v29 i, v31 i] j =
        (lutCoefficient Q4 inverse4 (MxxExponentLutNand.table 2) (sigma i).toNat *
          reducePoly Q4 N (1 * unitPair sm j)) • g' + reduceMatrix Q4 N 1 8 (e j))
    (hlutz : ∀ (s : Fin 16) (jj : Fin 16) (a : Fin 16),
      Rel sm (digitKey model v18 Q5 10 (lutOwner 50 ((s : Nat) : Int) 5) ((jj : Nat) : Int)
          ((a : Nat) : Int))
        (fun r ↦ (substitution (sigmaInv s).toNat (sigmaInv_odd s)
            (columnDigits Q5 (8 + 8) Backend.backend 1048576 8 524288
              (joinedEntry ![v25 s, v27 s] jj) (joinedEntry ![v29 s, v31 s] jj) a 0) *
          reducePoly Q5 N (unitPair sm r)) • g14)
        ![v36 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩,
          v39 ⟨16 * (16 * (s : Nat) + a) + jj, by omega⟩] 26)
    {arg : Fin 16} (harg : μ * ν = Xe (n := N) arg) :
    Rel sm ![![w.w_156_0, w.w_157_0], ![w.w_159_0, w.w_160_0]]
      (fun j ↦ reducePoly Q4 N (1 * Xe (n := N) (MxxExponentLutNand.table 2 arg) * unitPair sm j) • g')
      ![w.w_171_0, w.w_172_0] roundBound2 := by
  unfold Stage_online_eval.generatedRoot.body at hb
  dsimp only at hb
  have h34 := hb.2.2.2.2.1
  have h35 := hb.2.2.2.2.2.1
  have h36 := hb.2.2.2.2.2.2.1
  have h37 := hb.2.2.2.2.2.2.2.1
  have h262 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h265 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h267 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h274 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h281 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h282 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h289 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h296 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h297 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h299 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h302 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h303 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h304 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h306 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h313 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h320 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h322 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h323 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h324 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h326 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h327 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h363 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h364 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hgg : g14 = g14' := centeredRebaseRuns_unique hg14 (centeredRebaseRuns_self hQ5 hN g14')
  subst hgg
  have hinput' := hinput.target_eq (t' := fun j ↦
    reducePoly Q5 N (((1 : Nat) : Int) * μ * unitPair sm j) • g14) (by funext j; simp)
  -- The key of the state input, moved down by the previous round.
  have hA0 : concatColumns (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]] 0 0) (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]] 0 1) w.w_129_0 := h262
  have hA1 : concatColumns (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]] 1 0) (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]] 1 1) w.w_132_0 := h265
  -- The RHS companion loop.
  have hrl := rhsLoop2 hA0 hA1 h267
  dsimp only at hrl
  -- The column loop and the digits of the RHS columns.
  have hcl := colLoop2 h306
  let D : Fin 2 → ExactMatrix Q5 N (10 + 10) 8 := fun j a l ↦
    columnDigits Q5 (10 + 10) Backend.backend 1048576 10 524288
      (joinedEntry ![v169, v170] ⟨8 * (j : Nat) + l, by omega⟩)
      (joinedEntry ![v172, v173] ⟨8 * (j : Nat) + l, by omega⟩) a 0
  have hDcol : ∀ (j : Fin 2) (l : Fin 8),
      rhsDigits2 w.w_147_0 w.w_150_0 ⟨8 * (j : Nat) + l, by omega⟩ = fun a _ ↦ D j a l := by
    intro j l
    funext a b
    rw [Subsingleton.elim b 0]
    simp only [rhsDigits2, D, concatColumns_entry h303, concatColumns_entry h304]
  have hD : ∀ j l, ∃ D' : ExactMatrix Q5 N (10 + 10) 1,
      gadgetDecomposeRuns Backend.backend 1048576 10
        (pairColumn (![v169, v170] j 0 l) (![v172, v173] j 0 l)) D' ∧
      compactCenteredRebaseRuns 524288 D' (fun a _ ↦ D j a l) := by
    intro j l
    obtain ⟨D', h1, h2⟩ := (hcl ⟨8 * (j : Nat) + l, by omega⟩).2
    rw [concatColumns_entry h303, concatColumns_entry h304, joinedEntry_pair, joinedEntry_pair] at h1
    rw [hDcol] at h2
    exact ⟨D', h1, h2⟩
  have hcD : ∀ j : Fin 2, rowWindow 8 (8 * j) (by omega) w.w_151_0 = pairMul ![w.w_142_0, w.w_143_0] (D j) := by
    intro j
    rw [window_mul_columns (m := 10) (k := 8) (row := ![w.w_142_0, w.w_143_0]) (acc := w.w_151_0) h302
      (fun p ↦ (hcl p).1) j (by omega)]
    refine congrArg (pairMul ![w.w_142_0, w.w_143_0]) ?_
    funext a l
    rw [hDcol]
  -- The companion row and keys of the RHS.
  have hdig := fun (a : Fin (10 + 10)) (r : Fin 2) ↦ decompose_entry hQ5 hN Backend.layout2 layout5 rfl
    hg14' (entryDigits_runs hQ5 hN hg14' Backend.layout2 layout5 rfl
      (fun _ _ ↦ joinedEntry (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]] r) a))
  choose u' hu' hβ using fun (a : Fin (10 + 10)) (r : Fin 2) ↦ (hdig a r).2
  have hErow : ∀ (j : Fin 2) (l : Fin 8),
      rowWindow 8 (8 * j) (by omega) w.w_133_2 0 l =
        (∑ a : Fin (10 + 10),
          (![v43 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩,
              v45 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩] 0 *
            keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 0 +
          ![v43 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩,
              v45 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩] 1 *
            keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 1)) 0 0 := by
    intro j l
    simp only [rowWindow]
    rw [(hrl ⟨8 * (j : Nat) + l, by omega⟩).2.2, Matrix.sum_apply]
    simp only [rhsCompanionSum2, Matrix.cons_val_zero, Matrix.cons_val_one]
  have hMw : ∀ j : Fin 2, ![w.w_134_0, w.w_135_0] j = rowWindow 8 (8 * j) (by omega) w.w_133_0 ∧
      ![w.w_137_0, w.w_138_0] j = rowWindow 8 (8 * j) (by omega) w.w_133_1 := by
    intro j
    fin_cases j
    · exact ⟨sliceMatrix_window (8 * 0) h274 (by simp) (by omega),
        sliceMatrix_window (8 * 0) h289 (by simp) (by omega)⟩
    · exact ⟨sliceMatrix_window (8 * 1) h281 (by simp) (by omega),
        sliceMatrix_window (8 * 1) h296 (by simp) (by omega)⟩
  have hMrow0 : ∀ (j : Fin 2) (l : Fin 8), ![w.w_134_0, w.w_135_0] j 0 l =
      (∑ a : Fin (10 + 10),
        (digitKey model v18 Q5 10 (rhsOwner 50 4) ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int)
            0 0 * keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 0 +
          digitKey model v18 Q5 10 (rhsOwner 50 4) ((8 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) 0 1 * keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 1)) 0 0 := by
    intro j l
    rw [(hMw j).1]
    simp only [rowWindow]
    rw [(hrl ⟨8 * (j : Nat) + l, by omega⟩).1, Matrix.sum_apply]
    rfl
  have hMrow1 : ∀ (j : Fin 2) (l : Fin 8), ![w.w_137_0, w.w_138_0] j 0 l =
      (∑ a : Fin (10 + 10),
        (digitKey model v18 Q5 10 (rhsOwner 50 4) ((8 * (j : Nat) + l : Nat) : Int) ((a : Nat) : Int)
            1 0 * keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 0 +
          digitKey model v18 Q5 10 (rhsOwner 50 4) ((8 * (j : Nat) + l : Nat) : Int)
            ((a : Nat) : Int) 1 1 * keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a 1)) 0 0 := by
    intro j l
    rw [(hMw j).2]
    simp only [rowWindow]
    rw [(hrl ⟨8 * (j : Nat) + l, by omega⟩).2.1, Matrix.sum_apply]
    rfl
  obtain ⟨f, hf, hE⟩ := companion_rows (q := Q5) (m := 10) (m' := 8) (Dg := 10) hN (sm := sm)
    (g := g14) (A := ![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) (D := D)
    (B := fun j l a ↦ digitKey model v18 Q5 10 (rhsOwner 50 4) ((8 * (j : Nat) + l : Nat) : Int)
      ((a : Nat) : Int))
    (z := fun j l a ↦ ![v43 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩,
      v45 ⟨8 * (j : Nat) + l + 16 * a, by omega⟩])
    (bound := 26) (fun j l a ↦ hrhsz ⟨8 * (j : Nat) + l, by omega⟩ a)
    (u := fun a r ↦ keyDigits 10 (![![w.w_127_0, w.w_128_0], ![w.w_130_0, w.w_131_0]]) a r) (fun a r ↦ (hdig a r).1) (u' := u')
    (β := 524288) hu' hβ (E := fun j ↦ rowWindow 8 (8 * j) (by omega) w.w_133_2)
    (M := ![![w.w_134_0, w.w_135_0], ![w.w_137_0, w.w_138_0]]) hErow
    (fun k j l ↦ by fin_cases k; exacts [hMrow0 j l, hMrow1 j l])
  -- The product.
  obtain ⟨eB, heB, hrhs'⟩ := hrhs
  have hmul := mul_private_dynamic (q := Q5) (m := 10) (m' := 8) (bound := 524288) hN hQ5
    Backend.layout2 layout5 rfl hg14' (centeredRebaseRuns_self hQ5 hN _) hinput'
    (top := ![v169, v170]) (bot := ![v172, v173]) (D := D) hD hrhs' heB
    (smulGain_of_polyNorm hN hμ) hE hf
  have hprod : Rel sm ![![w.w_134_0, w.w_135_0], ![w.w_137_0, w.w_138_0]]
      (fun j ↦ reducePoly Q5 N (((P5 : Nat) : Int) * Xe (n := N) arg * unitPair sm j) • g)
      ![w.w_153_0, w.w_154_0] mulBound2 := by
    have hrows : ![w.w_153_0, w.w_154_0] = fun j ↦ pairMul ![w.w_142_0, w.w_143_0] (D j) +
        rowWindow 8 (8 * j) (by omega) w.w_133_2 := by
      funext j
      rw [← hcD j, ← rowWindow_add]
      fin_cases j
      · exact sliceMatrix_window (8 * 0) h313 (by simp) (by omega)
      · exact sliceMatrix_window (8 * 1) h320 (by simp) (by omega)
    rw [hrows]
    convert hmul using 2 with j
    rw [smul_smul, ← map_mul, ← harg]
    congr 2
    ring
  -- The LUT companion loop.
  have hll := lutLoop2 (M := ![![w.w_134_0, w.w_135_0], ![w.w_137_0, w.w_138_0]]) h282 h297 h299
  dsimp only at hll
  -- The sixteen branches.
  obtain ⟨_, hw72⟩ := familyGet_index (k := 0) h34 (by decide)
  obtain ⟨_, hw74⟩ := familyGet_index (k := 0) h35 (by decide)
  obtain ⟨_, hw76⟩ := familyGet_index (k := 0) h36 (by decide)
  obtain ⟨_, hw78⟩ := familyGet_index (k := 0) h37 (by decide)
  rw [hw72, hw74, hw76, hw78] at h322
  have hbl := branchLoop2 (fam := ![![v25, v27], ![v29, v31]]) (c := ![w.w_153_0, w.w_154_0])
    hsm hg' hg hg14' (centeredRebaseRuns_self hQ5 hN _) hprod hhelp hlutz hll h322
  -- The sum of the sixteen branches, moved down to level `6`.
  have hsum := hbl.target_eq (t' := fun j ↦ ∑ i : Fin 16, scaleUpMatrix Q5 P5
      (reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) (Xe (n := N) arg)) •
        (reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) sm) •
          ![![v25, v27], ![v29, v31]] 0 j i + ![![v25, v27], ![v29, v31]] 1 j i))) (by
    funext j
    rw [Finset.sum_apply, Finset.sum_range]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [branchTargetN2, dif_pos i.isLt]
    rfl)
  choose eS heS hS using hswitch
  have hwalsh : ∑ i : Fin 16, reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i)
      (Xe (n := N) arg)) * lutCoefficient Q4 inverse4 (MxxExponentLutNand.table 2) (sigma i).toNat =
      reducePoly Q4 N (Xe (n := N) (MxxExponentLutNand.table 2 arg)) := by
    rw [reducePoly_Xe, ← lut_walsh inverse4_mul (MxxExponentLutNand.table 2) arg]
    apply Finset.sum_congr rfl
    intro i _
    rw [substitution_root_pow, reducePoly_Xe, Nat.mul_comm]
  exact lut_finish (W := 16) (q := Q5) (q' := Q4) (p := P5) (m' := 8) hN q5_eq hQ5 hQ4
    (by norm_num) (sp := sm) (μ := Xe (n := N) arg)
    (ν := Xe (n := N) (MxxExponentLutNand.table 2 arg)) (Δ' := 1) hsm
    (fun i ↦ substitution (sigma i).toNat (sigma_odd i)) (fun i ↦ polyNorm_substitution_Xe i arg)
    (bsw := fun i k j ↦ ![![v25, v27], ![v29, v31]] k j i) hsum (g' := g')
    (C := fun i ↦ lutCoefficient Q4 inverse4 (MxxExponentLutNand.table 2) (sigma i).toNat)
    (fun i j ↦ by fin_cases j; exacts [hS i 0, hS i 1]) (fun i j ↦ heS i j) hwalsh
    (K' := ![![w.w_156_0, w.w_157_0], ![w.w_159_0, w.w_160_0]])
    (R' := ![w.w_171_0, w.w_172_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h323, h324, h326, h327])
    (fun j ↦ by fin_cases j; exacts [h363, h364])




end NandProof

