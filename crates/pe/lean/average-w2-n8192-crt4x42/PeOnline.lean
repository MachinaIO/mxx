import PeSetup

/-!
# Online evaluation

The Dynamic product of the initial state by the attribute package, then the coefficient LUT of the
predicate: the identity branch and the branch of `K`, each through its public scalar action.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The digits of column `jj` of the joined ciphertext `(top, bot)`. -/
noncomputable abbrev colDigits (top bot : ExactMatrix Q N 1 16) (jj : Fin 16) :
    ExactMatrix Q N (8 + 8) 1 :=
  columnDigits Q (8 + 8) Backend.backend 2097152 8 1048576 (top 0 jj) (bot 0 jj)


set_option maxHeartbeats 8000000 in
/-- The column loop of the product ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoopMul {model : HashModel} {tape : SampleTape} {path : List Nat} {c : ExactMatrix Q N 1 16} {top bot : ExactMatrix Q N 1 16}
    {final : ExactMatrix Q N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 16) ↦
      Stage_online_eval.sequential_generatedRoot_18 Backend.backend model tape (path ++ [18, i]) GeneratedClaim.stage_7_params i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : Fin 16 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 16 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 16 ↦ s)
    (fun i ↦ if h : i < 16 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 16) else 0) Inv
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


set_option maxHeartbeats 8000000 in
/-- The column loop of the helper action ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoopHelper {model : HashModel} {tape : SampleTape} {path : List Nat} {i0 : Nat} {c : ExactMatrix Q N 1 16} {top bot : ExactMatrix Q N 1 16}
    {final : ExactMatrix Q N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 16) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_64_18 Backend.backend model tape (path ++ [18, i]) GeneratedClaim.stage_7_params i0 i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : Fin 16 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 16 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 16 ↦ s)
    (fun i ↦ if h : i < 16 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 16) else 0) Inv
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


set_option maxHeartbeats 8000000 in
/-- The column loop of the automorphism switch ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoopSwitch {model : HashModel} {tape : SampleTape} {path : List Nat} {i0 : Nat} {c : ExactMatrix Q N 1 16} {top bot : ExactMatrix Q N 1 16}
    {final : ExactMatrix Q N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 16) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_64_44 Backend.backend model tape (path ++ [44, i]) GeneratedClaim.stage_7_params i0 i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 16 = 16 * 1 := rfl
  rw [hcount] at h
  let F : Fin 16 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 16 → Prop := fun i _ ↦ ∀ jj : Fin 16, jj.val < i →
    ∃ D' : ExactMatrix Q N (8 + 8) 1,
      gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 16 ↦ s)
    (fun i ↦ if h : i < 16 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 16) else 0) Inv
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


/-! ## The companion row of the product -/

/-- The state of the inner companion loop of the product: the row and the carried companions. -/
abbrev MulBlock := ExactMatrix Q N 1 16 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × Unit

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the product's companions at column `jj`. -/
noncomputable def mulCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8)
    (z0 z1 : Fin 256 → ExactMatrix Q N 1 8) (jj : Fin 16) (a : Fin 16) : ExactPoly Q N :=
  (z0 ⟨jj + 16 * a, by omega⟩ * keyDigits A a 0 + z1 ⟨jj + 16 * a, by omega⟩ * keyDigits A a 1) 0 0

set_option maxHeartbeats 8000000 in
/-- One iteration `j` of the inner companion loop of the product at digit row `a` adds, at column
`j`, companion `j + 16 a` paired with the digits `U`, and reads the next companion. -/
theorem mulInner_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {U : Fin 2 → ExactMatrix Q N 8 1} {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {a j : Nat}
    (ha : a < 16) (hj : j < 16) {s t : MulBlock}
    (hs : s.2.1 = z0 ⟨j + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨j + 16 * a, by omega⟩)
    (h : Stage_online_eval.sequential_sequential_generatedRoot_29_15 Backend.backend model tape path
      GeneratedClaim.stage_7_params a j (s.1, s.2.1, s.2.2.1, U 0, U 1, z0, z1, ()) t) :
    t.1 = s.1 + (z0 ⟨j + 16 * a, by omega⟩ * U 0 + z1 ⟨j + 16 * a, by omega⟩ * U 1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = z0 ⟨(j + 1) % 16 + 16 * a, by omega⟩ ∧
      t.2.2.1 = z1 ⟨(j + 1) % 16 + 16 * a, by omega⟩ := by
  obtain ⟨w9, w14, w16, -, -, -, -, -, -, -, -, x9, -, -, -, -, x14, -, -, x16, xout⟩ := h
  have hnext : 16 * Int.fmod (Int.ofNat a) 16 + Int.fmod (1 + Int.ofNat j) 16 =
      Int.ofNat ((j + 1) % 16 + 16 * a) := by
    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hnext] at x14 x16
  obtain ⟨_, hw14⟩ := familyGet_eq x14
  obtain ⟨_, hw16⟩ := familyGet_eq x16
  have hw9 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x9
  subst xout hw9 hw14 hw16
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, rfl, rfl⟩
  simp only [hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq]

set_option maxHeartbeats 8000000 in
/-- The inner companion loop of the product at digit row `a` ends with that digit row's companion
row, column by column. -/
theorem mulInner {model : HashModel} {tape : SampleTape} {path : List Nat}
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8) {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} (a : Fin 16)
    {final : MulBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : MulBlock) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_29_15 Backend.backend model tape
        (path ++ [15, i]) GeneratedClaim.stage_7_params a i
        (current.1, (current.2.1, (current.2.2.1, (keyDigits A a 0, (keyDigits A a 1, (z0,
          (z1, ()))))))) next) (Int.toNat 16)
      (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))) final) (jj : Fin 16) :
    final.1 0 jj = mulCompanionSum A z0 z1 jj a := by
  let Inv : Nat → MulBlock → Prop := fun i s ↦
    s.2.1 = z0 ⟨i % 16 + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i % 16 + 16 * a, by omega⟩
  have hInv0 : Inv 0 (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))) :=
    ⟨congrArg z0 (Fin.ext (by simp)), congrArg z1 (Fin.ext (by simp))⟩
  have hstep : ∀ i (s t : MulBlock) (hi : i < 16), Inv i s →
      Stage_online_eval.sequential_sequential_generatedRoot_29_15 Backend.backend model tape
        (path ++ [15, i]) GeneratedClaim.stage_7_params a i
        (s.1, (s.2.1, (s.2.2.1, (keyDigits A a 0, (keyDigits A a 1, (z0, (z1, ()))))))) t →
      (∀ w, t.1 0 w = s.1 0 w + mulCompanionSum A z0 z1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) ∧ Inv (i + 1) t := by
    intro i s t hi hq hb
    have hq' : s.2.1 = z0 ⟨i + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i + 16 * a, by omega⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext (by simp [Nat.mod_eq_of_lt hi]))),
        h1.trans (congrArg z1 (Fin.ext (by simp [Nat.mod_eq_of_lt hi])))⟩
    obtain ⟨h0, h1, h2⟩ := mulInner_step (U := keyDigits A a) a.isLt hi hq' hb
    refine ⟨fun w ↦ ?_, h1, h2⟩
    rw [h0]
    rfl
  have hk := IterRuns.accumulate_entries (fun s : MulBlock ↦ s.1)
    (fun w (j : Fin 16) ↦ mulCompanionSum A z0 z1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    Inv hInv0 (fun i s t hi hq hb ↦ hstep i s t hi hq hb) h jj
  rw [hk, sum_unitRow_mul (fun j ↦ mulCompanionSum A z0 z1 j a) jj]
  simp

set_option maxHeartbeats 16000000 in
/-- One iteration `a` of the companion loop of the product decomposes key column `a` once and adds
the companion row of digit row `a`, column by column. -/
theorem mulLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {i : Nat} (hi : i < 16) {s t : ExactMatrix Q N 1 16}
    (h : Stage_online_eval.sequential_generatedRoot_29 Backend.backend model tape path
      GeneratedClaim.stage_7_params i (s, z0, z1, A0, A1, ()) t) (jj : Fin 16) :
    t 0 jj = s 0 jj + mulCompanionSum A z0 z1 jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, x6, -, -, x8, -, -, -, -, -, -, -, -, x10,
    x11, -, -, -, -, -, -, -, -, x13, x14, -, hin, x16, xout⟩ := h
  have hdiv : Int.fdiv (Int.ofNat i) 16 = 0 := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 16 = ((i : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 16 * Int.fmod (Int.ofNat i) 16 = Int.ofNat (16 * i) := by
    rw [hrow]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hdiv] at x2
  rw [hrow] at x10 x13
  rw [hfirst] at x6 x8
  have hw2 := unitRow_window (c := 0) (by omega) x2
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨_, hw8⟩ := familyGet_eq x8
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q N 1 16} {x : ExactMatrix Q N 1 1}
      {u : ExactMatrix Q N 8 1}, concatColumns (A r 0) (A r 1) Ar →
      sliceMatrix Ar 0 1 ((i : Nat) : Int) (1 + ((i : Nat) : Int)) x →
      gadgetDecomposeRuns Backend.backend 2097152 8 x u → u = keyDigits A ⟨i, hi⟩ r := by
    intro r Ar x u hAr hx hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    have hxe : x = fun _ _ ↦ joinedEntry (A r) ⟨i, hi⟩ := by
      funext a b
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0]
      congr 1
    rw [hxe] at hu
    exact entryDigits_eq hu
  have hw11 := hentry hA0 x10 x11
  have hw14 := hentry hA1 x13 x14
  rw [hw11, hw14, hw6, hw8] at hin
  have hblock := mulInner (model := model) (tape := tape) (path := path) A ⟨i, hi⟩ hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 16}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x16]
  exact congrArg _ hblock

set_option maxHeartbeats 8000000 in
/-- The companion loop of the product ends with the companion row, column by column. -/
theorem mulLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {final : ExactMatrix Q N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 16) ↦
      Stage_online_eval.sequential_generatedRoot_29 Backend.backend model tape (path ++ [29, i])
        GeneratedClaim.stage_7_params i (current, (z0, (z1, (A0, (A1, ()))))) next)
      (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = ∑ a, mulCompanionSum A z0 z1 jj a := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have hk := IterRuns.accumulate_entries (fun s : ExactMatrix Q N 1 16 ↦ s)
    (fun w a ↦ mulCompanionSum A z0 z1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ mulLoop_step hA0 hA1 hi hb w, trivial⟩) h jj
  rw [hk]
  simp

/-! ## The companion row of the automorphism -/

/-- The key `σ_K(A)` routed by the automorphism of the LUT branch. -/
noncomputable def routedKey (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 8 :=
  fun r j ↦ substituteMatrix K hK (A r j)

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the automorphism companions at column `jj`, against
the digits `u` of the routed key. -/
noncomputable def lutCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8)
    (z0 z1 : Fin 256 → ExactMatrix Q N 1 8) (jj : Fin 16) (a : Fin 16) : ExactPoly Q N :=
  (z0 ⟨jj + 16 * a, by omega⟩ * keyDigits (routedKey A) a 0 +
    z1 ⟨jj + 16 * a, by omega⟩ * keyDigits (routedKey A) a 1) 0 0

set_option maxHeartbeats 8000000 in
theorem lutInner_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {U : Fin 2 → ExactMatrix Q N 8 1} {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {a j : Nat}
    (ha : a < 16) (hj : j < 16) {s t : MulBlock}
    (hs : s.2.1 = z0 ⟨j + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨j + 16 * a, by omega⟩)
    (h : Stage_online_eval.sequential_sequential_generatedRoot_63_17 Backend.backend model tape path
      GeneratedClaim.stage_7_params a j (s.1, s.2.1, s.2.2.1, U 0, U 1, z0, z1, ()) t) :
    t.1 = s.1 + (z0 ⟨j + 16 * a, by omega⟩ * U 0 + z1 ⟨j + 16 * a, by omega⟩ * U 1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = z0 ⟨(j + 1) % 16 + 16 * a, by omega⟩ ∧
      t.2.2.1 = z1 ⟨(j + 1) % 16 + 16 * a, by omega⟩ := by
  obtain ⟨w9, w14, w16, -, -, -, -, -, -, -, -, x9, -, -, -, -, -, -, -, x14, -, -, x16, xout⟩ := h
  have hnext : 16 * Int.fmod (Int.ofNat a) 16 + Int.fmod (1 + Int.ofNat j) 16 +
      256 * MxxIR.selectInt (Int.fdiv (Int.ofNat a) 16) [0] = Int.ofNat ((j + 1) % 16 + 16 * a) := by
    have hd : Int.fdiv (Int.ofNat a) 16 = 0 := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
      simp only [Int.ofNat_eq_natCast]
      omega
    rw [hd]
    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 16 by norm_num), Int.ofNat_eq_natCast]
    simp [MxxIR.selectInt]
    omega
  rw [hnext] at x14 x16
  obtain ⟨_, hw14⟩ := familyGet_eq x14
  obtain ⟨_, hw16⟩ := familyGet_eq x16
  have hw9 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x9
  subst xout hw9 hw14 hw16
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, rfl, rfl⟩
  simp only [hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq]

set_option maxHeartbeats 8000000 in
theorem lutInner {model : HashModel} {tape : SampleTape} {path : List Nat}
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8) {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} (a : Fin 16)
    {final : MulBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : MulBlock) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_63_17 Backend.backend model tape
        (path ++ [17, i]) GeneratedClaim.stage_7_params a i
        (current.1, (current.2.1, (current.2.2.1, (keyDigits (routedKey A) a 0,
          (keyDigits (routedKey A) a 1, (z0, (z1, ()))))))) next) (Int.toNat 16)
      (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))) final) (jj : Fin 16) :
    final.1 0 jj = lutCompanionSum A z0 z1 jj a := by
  let Inv : Nat → MulBlock → Prop := fun i s ↦
    s.2.1 = z0 ⟨i % 16 + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i % 16 + 16 * a, by omega⟩
  have hInv0 : Inv 0 (0, (z0 ⟨16 * a, by omega⟩, (z1 ⟨16 * a, by omega⟩, ()))) :=
    ⟨congrArg z0 (Fin.ext (by simp)), congrArg z1 (Fin.ext (by simp))⟩
  have hstep : ∀ i (s t : MulBlock) (hi : i < 16), Inv i s →
      Stage_online_eval.sequential_sequential_generatedRoot_63_17 Backend.backend model tape
        (path ++ [17, i]) GeneratedClaim.stage_7_params a i
        (s.1, (s.2.1, (s.2.2.1, (keyDigits (routedKey A) a 0, (keyDigits (routedKey A) a 1,
          (z0, (z1, ()))))))) t →
      (∀ w, t.1 0 w = s.1 0 w + lutCompanionSum A z0 z1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) ∧ Inv (i + 1) t := by
    intro i s t hi hq hb
    have hq' : s.2.1 = z0 ⟨i + 16 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i + 16 * a, by omega⟩ := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext (by simp [Nat.mod_eq_of_lt hi]))),
        h1.trans (congrArg z1 (Fin.ext (by simp [Nat.mod_eq_of_lt hi])))⟩
    obtain ⟨h0, h1, h2⟩ := lutInner_step (U := keyDigits (routedKey A) a) a.isLt hi hq' hb
    refine ⟨fun w ↦ ?_, h1, h2⟩
    rw [h0]
    rfl
  have hk := IterRuns.accumulate_entries (fun s : MulBlock ↦ s.1)
    (fun w (j : Fin 16) ↦ lutCompanionSum A z0 z1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    Inv hInv0 (fun i s t hi hq hb ↦ hstep i s t hi hq hb) h jj
  rw [hk, sum_unitRow_mul (fun j ↦ lutCompanionSum A z0 z1 j a) jj]
  simp

/-- Substitutions by equal indices agree. -/
theorem substitution_congr {R : Type} [CommRing R] {i j : Nat} (h : i = j) (hi : Odd i)
    (hj : Odd j) : (substitution (n := N) (R := R) i hi : Negacyclic N R →+* Negacyclic N R) =
      substitution j hj := by
  subst h
  rfl

/-- The digits of an entry routed by `σ_K`. -/
theorem routed_entry_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {r : Fin 2} {a : Fin 16}
    {Ar : ExactMatrix Q N 1 16} {x y : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 8 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar)
    (hx : sliceMatrix Ar 0 1 ((a : Nat) : Int) (1 + ((a : Nat) : Int)) x)
    (hy : ringAutomorphismRuns (MxxIR.selectInt 0 [8193]) x y)
    (hu : gadgetDecomposeRuns Backend.backend 2097152 8 y u) : u = keyDigits (routedKey A) a r := by
  obtain ⟨_, hx0⟩ := sliceMatrix_column hx
  rw [concatColumns_entry hAr, pair_eta] at hx0
  have hxe : x = fun _ _ ↦ joinedEntry (A r) a := by
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0, hx0]
    congr 1
  obtain ⟨hodd, hye⟩ := ringAutomorphismRuns_eq hQ hN hy
  have hidx : (MxxIR.selectInt 0 [8193]).toNat = K := by decide
  have hyx : y = fun _ _ ↦ joinedEntry (routedKey A r) a := by
    rw [hye, hxe]
    funext i j
    rw [show joinedEntry (routedKey A r) a = substitution K hK (joinedEntry (A r) a) from
      joinedEntry_substitute hK (fun j ↦ A r j) a]
    rw [substitution_congr hidx hodd hK]
  rw [hyx] at hu
  exact entryDigits_eq hu

set_option maxHeartbeats 16000000 in
/-- One iteration `a` of the companion loop of the automorphism decomposes the routed key column
`a` once and adds the companion row of digit row `a`, column by column. -/
theorem lutLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {i : Nat} (hi : i < 16) {s t : ExactMatrix Q N 1 16}
    (h : Stage_online_eval.sequential_generatedRoot_63 Backend.backend model tape path
      GeneratedClaim.stage_7_params i (s, z0, z1, A0, A1, ()) t) (jj : Fin 16) :
    t 0 jj = s 0 jj + lutCompanionSum A z0 z1 jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, -, -, -, x6, -, -, x8, -, -, -, -, -, -, -,
    -, x10, -, -, -, x11, x12, -, -, -, -, -, -, -, -, x14, -, -, -, x15, x16, -, hin, x18,
    xout⟩ := h
  have hdiv : Int.fdiv (Int.ofNat i) 16 = 0 := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 16 = ((i : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 16 * Int.fmod (Int.ofNat i) 16 + 256 * MxxIR.selectInt (Int.fdiv (Int.ofNat i) 16) [0] =
      Int.ofNat (16 * i) := by
    rw [hrow, hdiv]
    simp [MxxIR.selectInt]
  rw [hdiv] at x2 x11 x15
  rw [hrow] at x10 x14
  rw [hfirst] at x6 x8
  have hw2 := unitRow_window (c := 0) (by omega) x2
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨_, hw8⟩ := familyGet_eq x8
  have hw12 := routed_entry_digits (a := ⟨i, hi⟩) hA0 x10 x11 x12
  have hw16 := routed_entry_digits (a := ⟨i, hi⟩) hA1 x14 x15 x16
  rw [hw12, hw16, hw6, hw8] at hin
  have hblock := lutInner (model := model) (tape := tape) (path := path) A ⟨i, hi⟩ hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 16}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x18]
  exact congrArg _ hblock

set_option maxHeartbeats 8000000 in
/-- The companion loop of the automorphism ends with its companion row, column by column. -/
theorem lutLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {final : ExactMatrix Q N 1 16}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 16) ↦
      Stage_online_eval.sequential_generatedRoot_63 Backend.backend model tape (path ++ [63, i])
        GeneratedClaim.stage_7_params i (current, (z0, (z1, (A0, (A1, ()))))) next)
      (Int.toNat 16) 0 final) (jj : Fin 16) :
    final 0 jj = ∑ a, lutCompanionSum A z0 z1 jj a := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have hk := IterRuns.accumulate_entries (fun s : ExactMatrix Q N 1 16 ↦ s)
    (fun w a ↦ lutCompanionSum A z0 z1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ lutLoop_step hA0 hA1 hi hb w, trivial⟩) h jj
  rw [hk]
  simp

/-- The digits of output column `l` of component `j` of the action of `(top, bot)`. -/
noncomputable def actionDigits (top bot : ExactMatrix Q N 1 16) : Fin 2 → ExactMatrix Q N (8 + 8) 8 :=
  fun j a l ↦ colDigits top bot ⟨8 * (j : Nat) + l, by omega⟩ a 0

/-- A column loop of a joined row `c = [row₀ | row₁]` by the joined matrix `(top, bot)` with
components `tk`, `bk`: component `j` of the result is `row` times the digits of the columns of
component `j`, and every column decomposition runs. -/
theorem colAction {c : ExactMatrix Q N 1 16} {row : Fin 2 → ExactMatrix Q N 1 8}
    (hc : concatColumns (row 0) (row 1) c) {top bot : ExactMatrix Q N 1 16}
    {tk bk : Fin 2 → ExactMatrix Q N 1 8} (htop : ∀ p : Fin 16, top 0 p = joinedEntry tk p)
    (hbot : ∀ p : Fin 16, bot 0 p = joinedEntry bk p) {acc : ExactMatrix Q N 1 16}
    (hacc : ∀ jj : Fin 16, acc 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits top bot jj)) :
    (∀ j : Fin 2, rowWindow 8 (8 * j) (by omega) acc = pairMul row (actionDigits top bot j)) ∧
      ∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (tk j 0 l) (bk j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits top bot j a l) := by
  refine ⟨fun j ↦ ?_, fun j l ↦ ?_⟩
  · rw [window_mul_columns (m := 8) (k := 8) hc (fun p ↦ (hacc p).1) j (by omega)]
    rfl
  · obtain ⟨D', h1, h2⟩ := (hacc ⟨8 * (j : Nat) + l, by omega⟩).2
    rw [htop, hbot, joinedEntry_pair, joinedEntry_pair] at h1
    refine ⟨D', h1, ?_⟩
    have hcol : colDigits top bot ⟨8 * (j : Nat) + l, by omega⟩ =
        fun a _ ↦ actionDigits top bot j a l := by
      funext a b
      rw [Subsingleton.elim b 0]
      rfl
    rwa [hcol] at h2

/-! ## The LUT branches -/

/-- `2⁻¹ mod Q`. -/
abbrev inverse2 : Int := 187072069502007624630008562053914270362791364190209

/-- The scalar `D_k = 2⁻¹ (L(0) σ_k(X⁰) + L(1) σ_k(-X^8191))` of the branch of `k` for the table
`L = (0, 1)`, as the graph computes it, with `σ` the routing of `k`. -/
noncomputable def lutScalar (σ : ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1) : ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (matrixPolynomial [inverse2])
    (matrixAdd (matrixMulScalarLeft (matrixPolynomial [0]) (σ (rotationPolynomial 0)))
      (matrixMulScalarLeft (matrixPolynomial [1]) (σ (matrixNeg (rotationPolynomial 8191)))))

/-- The two halves of a row, concatenated, are the row. -/
theorem concat_windows {x y : ExactMatrix Q N 1 16}
    (h : concatColumns (rowWindow 8 0 (by omega) x) (rowWindow 8 8 (by omega) x) y) : y = x := by
  funext r p
  rw [Subsingleton.elim r 0, h 0 p]
  split
  · simp only [rowWindow]
    congr 1
    ext
    simp
  · simp only [rowWindow]
    congr 1
    ext
    simp
    omega

/-- The selector of the branch of `K`. -/
theorem selectK : MxxIR.selectInt (Int.ofNat 0) [8193] = (K : Int) := by decide

/-- A routing by the branch selector is the substitution of `K`. -/
theorem routing_K {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat 0) [8193]) x y) :
    y = substituteMatrix K hK x := by
  rw [selectK] at h
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  funext r c
  exact congrFun (congrArg DFunLike.coe (substitution_congr (by decide) hodd hK)) _

/-- The joined row `[r₀ | r₁]`. -/
def joinRow (r : Fin 2 → ExactMatrix Q N 1 8) : ExactMatrix Q N 1 16 := fun _ p ↦ joinedEntry r p

theorem concat_joinRow {r : Fin 2 → ExactMatrix Q N 1 8} {y : ExactMatrix Q N 1 16}
    (h : concatColumns (r 0) (r 1) y) : y = joinRow r := by
  funext a p
  rw [Subsingleton.elim a 0, concatColumns_entry h, pair_eta]
  rfl

/-- The halves of a joined row. -/
def halves (x : ExactMatrix Q N 1 16) : Fin 2 → ExactMatrix Q N 1 8 :=
  fun j ↦ rowWindow 8 (8 * j) (by omega) x

/-- Entry `p` of a joined row, by the half it lies in. -/
theorem joinedEntry_split (r : Fin 2 → ExactMatrix Q N 1 8) (p : Fin 16) :
    joinedEntry r p = if h : p.val < 8 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 8, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

theorem concatColumns_joinRow (r : Fin 2 → ExactMatrix Q N 1 8) :
    concatColumns (r 0) (r 1) (joinRow r) := by
  intro a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split]

theorem joinRow_halves (x : ExactMatrix Q N 1 16) : joinRow (halves x) = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split, halves, rowWindow]
  split
  · simp
  · congr 1
    ext
    simp
    omega

theorem entry_halves (x : ExactMatrix Q N 1 16) (p : Fin 16) : x 0 p = joinedEntry (halves x) p := by
  conv_lhs => rw [← joinRow_halves x]
  rfl

theorem rowWindow_full (x : ExactMatrix Q N 1 16) : rowWindow 16 0 (by omega) x = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp [rowWindow]

set_option maxHeartbeats 64000000 in
/-- The branch of `K` of the LUT: from the accumulated identity branch `acc`, it adds, per
component `j`, the helper action `h Dec(T)`, the switch `σ_K(c) Dec(C_K)`, and the companion row
`E`, all through the scalar action `V = Dec(D_K g)`. `C_K` has the rows `σ_K(H⁰)` and
`σ_K(H¹) - σ_K(h)` of the helper key `H` of `K` and the helper `h` of `K = K⁻¹`. -/
theorem branch_rows {model : HashModel} {tape : SampleTape} {path : List Nat}
    {acc0 acc1 : ExactMatrix Q N 1 8} {hf0 hf1 : Fin 1 → ExactMatrix Q N 1 8}
    {T0 T1 : ExactMatrix Q N 1 16} {c0 c1 : ExactMatrix Q N 1 8} {key : ByteArray}
    {E : ExactMatrix Q N 1 16} {outputs : ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × Unit}
    (h : Stage_online_eval.sequential_generatedRoot_64 Backend.backend model tape path
      GeneratedClaim.stage_7_params 0 (acc0, acc1, hf0, hf1, T0, T1, c0, c1, key, E, ()) outputs) :
    let hrow : Fin 2 → ExactMatrix Q N 1 8 := ![hf0 0, hf1 0]
    let top : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 8 := ![substituteMatrix K hK c0, substituteMatrix K hK c1]
    ∃ (g : ExactMatrix Q N 1 8) (V : ExactMatrix Q N 8 8),
      gadgetMatrixRuns Backend.backend 2097152 8 g ∧
      gadgetDecomposeRuns Backend.backend 2097152 8
        (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g) V ∧
      (∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (halves T0 j 0 l) (halves T1 j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits T0 T1 j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits (joinRow top) (joinRow bot) j a l)) ∧
      ∀ j : Fin 2, ![outputs.1, outputs.2.1] j = ![acc0, acc1] j +
        (pairMul hrow (actionDigits T0 T1 j) + (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) +
          halves E j)) * V := by
  intro hrow top bot σc
  obtain ⟨w, -, -, -, -, x4, -, -, x6, x7, -, -, -, -, -, -, x9, -, -, -, -, -, -, x10, -, -, -, -,
    -, -, x11, x12, -, -, -, -, -, -, x14, -, -, -, -, -, -, x15, -, -, -, -, -, -, x16, x17, -, x18,
    -, -, -, -, -, -, x19, -, -, x22, -, -, x24, x25, -, -, x27, -, -, x28, -, -, x29, -, -, x30,
    x31, -, -, x32, -, -, x33, -, -, -, -, x35, -, -, x36, -, -, x38, -, -, x39, -, -, x40, -, -,
    x41, x43, -, x44, -, -, -, -, -, -, x45, -, -, -, -, -, -, x47, -, -, -, -, -, -, x48, -, -, -,
    -, x54, -, -, -, -, x59, x63, x65, -, -, -, -, -, -, x69, -, -, -, -, -, -, x70, -, -, -, -,
    -, -, x71, xout⟩ := h
  have hzero : MxxIR.selectInt (Int.ofNat 0) [0] = Int.ofNat 0 := by decide
  -- The helper of `K`.
  rw [hzero] at x4 x6
  obtain ⟨_, hw4⟩ := familyGet_eq x4
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  rw [hw4, hw6] at x7
  have hw7 : w.w_7_0 = joinRow hrow := concat_joinRow (r := hrow) x7
  -- The switched key rows.
  have hw9 := (sliceMatrix_window 0 x9 (by simp) (by omega)).trans (rowWindow_full T0)
  have hw14 := (sliceMatrix_window 0 x14 (by simp) (by omega)).trans (rowWindow_full T1)
  rw [hw9] at x10 x11
  rw [hw14] at x15 x16
  have hw10 := sliceMatrix_window 0 x10 (by simp) (by omega)
  have hw11 := sliceMatrix_window 8 x11 (by simp) (by omega)
  have hw15 := sliceMatrix_window 0 x15 (by simp) (by omega)
  have hw16 := sliceMatrix_window 8 x16 (by simp) (by omega)
  rw [hw10, hw11] at x12
  rw [hw15, hw16] at x17
  have hw12 : w.w_12_0 = T0 := (concat_joinRow (r := halves T0) x12).trans (joinRow_halves T0)
  have hw17 : w.w_17_0 = T1 := (concat_joinRow (r := halves T1) x17).trans (joinRow_halves T1)
  -- The helper action.
  rw [hw7, hw12, hw17] at x18
  have hhelp := colAction (row := hrow) (concatColumns_joinRow hrow) (tk := halves T0) (bk := halves T1)
    (entry_halves T0) (entry_halves T1)
    (colLoopHelper (i0 := 0) x18)
  -- The routed state rows.
  have hw22 := routing_K x22
  have hw24 := routing_K x24
  rw [hw22, hw24] at x25
  have hw25 : w.w_25_0 = joinRow σc := concat_joinRow (r := σc) x25
  -- The switching ciphertext `C_K`.
  have hkey : ∀ k j, helperKey model key k j = ![![w.w_27_0, w.w_29_0], ![w.w_32_0, w.w_38_0]] k j := by
    intro k j
    rw [selectK] at x27 x29 x32 x38
    fin_cases k <;> fin_cases j <;>
      simp [helperKey, dynamicPrefix, schemaBytes, componentBytes, hashSample_eq x27,
        hashSample_eq x29, hashSample_eq x32, hashSample_eq x38]
  rw [hzero] at x35 x40
  obtain ⟨_, hw35⟩ := familyGet_eq x35
  obtain ⟨_, hw40⟩ := familyGet_eq x40
  have hw28 := routing_K x28
  have hw30 := routing_K x30
  have hw33 := routing_K x33
  have hw36 := routing_K x36
  have hw39 := routing_K x39
  have hw41 := routing_K x41
  have htop : ![w.w_28_0, w.w_30_0] = top := by
    funext j
    fin_cases j <;> simp [top, hw28, hw30, hkey]
  have hbot : ![matrixSub w.w_33_0 w.w_36_0, matrixSub w.w_39_0 w.w_41_0] = bot := by
    funext j
    fin_cases j <;> simp [bot, hrow, hw33, hw36, hw39, hw41, hw35, hw40, hkey, matrixSub]
  have hw31 : w.w_31_0 = joinRow top := by
    rw [← htop]
    exact concat_joinRow (r := ![w.w_28_0, w.w_30_0]) x31
  have hw43 : w.w_43_0 = joinRow bot := by
    rw [← hbot]
    exact concat_joinRow (r := ![matrixSub w.w_33_0 w.w_36_0, matrixSub w.w_39_0 w.w_41_0]) x43
  rw [hw25, hw31, hw43] at x44
  have hswitch := colAction (row := σc) (concatColumns_joinRow σc) (tk := top) (bk := bot)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (colLoopSwitch (i0 := 0) x44)
  -- The companion row and the halves of the actions.
  have hw47 := (sliceMatrix_window 0 x47 (by simp) (by omega)).trans (rowWindow_full E)
  rw [hw47] at x48 x71
  have hw48 : w.w_48_0 = halves E 0 := sliceMatrix_window 0 x48 (by simp) (by omega)
  have hw71 : w.w_71_0 = halves E 1 := sliceMatrix_window 8 x71 (by simp) (by omega)
  have hw19 : w.w_19_0 = pairMul hrow (actionDigits T0 T1 0) :=
    (sliceMatrix_window 0 x19 (by simp) (by omega)).trans (hhelp.1 0)
  have hw69 : w.w_69_0 = pairMul hrow (actionDigits T0 T1 1) :=
    (sliceMatrix_window 8 x69 (by simp) (by omega)).trans (hhelp.1 1)
  have hw45 : w.w_45_0 = pairMul σc (actionDigits (joinRow top) (joinRow bot) 0) :=
    (sliceMatrix_window 0 x45 (by simp) (by omega)).trans (hswitch.1 0)
  have hw70 : w.w_70_0 = pairMul σc (actionDigits (joinRow top) (joinRow bot) 1) :=
    (sliceMatrix_window 8 x70 (by simp) (by omega)).trans (hswitch.1 1)
  -- The scalar action.
  have hw54 := routing_K x54
  have hw59 := routing_K x59
  refine ⟨w.w_63_0, w.w_65_0, x63, ?_, hhelp.2, hswitch.2, fun j ↦ ?_⟩
  · convert x65 using 2
    simp only [lutScalar, hw54, hw59]
  · subst xout
    fin_cases j
    · simp [matrixAdd, matrixMul, hw19, hw45, hw48]
    · simp [matrixAdd, matrixMul, hw69, hw70, hw71]

/-- The substitution `X ↦ X` is the identity. -/
theorem substitution_one {R : Type} [CommRing R] (h : Odd 1) (x : Negacyclic N R) :
    substitution 1 h x = x := by
  have hid : (substitution (n := N) (R := R) 1 h) = RingHom.id _ := by
    apply AdjoinRoot.ringHom_ext
    · ext r
      simp [substitution]
    · simp [substitution_root]
  rw [hid]
  rfl

/-- The routing of the identity branch is the identity. -/
theorem routing_one {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns 1 x y) : y = x := by
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  funext r c
  exact substitution_one hodd (x r c)

/-- The scalar `D_1` of the identity branch. -/
noncomputable abbrev lutScalarOne : ExactMatrix Q N 1 1 := lutScalar id

set_option maxHeartbeats 64000000 in
/-- The output rows of online evaluation: the product row `c Dec(C) + E` of the initial state and
the attribute package, then the LUT `D_1 (·) + D_K (Aut_K(·))` through the public scalar actions;
the key ports pass through from preprocessing. -/
theorem online_rows {model : HashModel} {tape : SampleTape} {path : List Nat}
    {v0 v1 v2 v3 c0a c0b : ExactMatrix Q N 1 8} {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 8}
    {z0 z1 : Fin 256 → ExactMatrix Q N 1 8} {key : ByteArray}
    {hf0 hf1 : Fin 1 → ExactMatrix Q N 1 8} {T0 T1 : ExactMatrix Q N 1 16}
    {za0 za1 : Fin 256 → ExactMatrix Q N 1 8} {M00 M01 M10 M11 : ExactMatrix Q N 1 8}
    {outputs : ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 ×
      ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × Unit}
    (h : Stage_online_eval.generatedRoot Backend.backend model tape path
      GeneratedClaim.stage_7_params (v0, v1, v2, v3, c0a, c0b, C00, C01, C10, C11, z0, z1, key, hf0,
        hf1, T0, T1, za0, za1, M00, M01, M10, M11, ()) outputs) :
    let cIn : Fin 2 → ExactMatrix Q N 1 8 := ![c0a, c0b]
    let Ctop : Fin 2 → ExactMatrix Q N 1 8 := ![C00 0, C01 0]
    let Cbot : Fin 2 → ExactMatrix Q N 1 8 := ![C10 0, C11 0]
    let A := inputKey model key
    let Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 8 := ![![M00, M01], ![M10, M11]]
    let Emul : ExactMatrix Q N 1 16 := fun _ jj ↦ ∑ a, mulCompanionSum A z0 z1 jj a
    let mulRow : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦
      pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j
    let Elut : ExactMatrix Q N 1 16 := fun _ jj ↦ ∑ a, lutCompanionSum Mmul za0 za1 jj a
    let hrow : Fin 2 → ExactMatrix Q N 1 8 := ![hf0 0, hf1 0]
    let top : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 8 := fun j ↦ substituteMatrix K hK (mulRow j)
    ∃ (g : ExactMatrix Q N 1 8) (V1 VK : ExactMatrix Q N 8 8),
      gadgetMatrixRuns Backend.backend 2097152 8 g ∧
      gadgetDecomposeRuns Backend.backend 2097152 8 (matrixMulScalarLeft lutScalarOne g) V1 ∧
      gadgetDecomposeRuns Backend.backend 2097152 8
        (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g) VK ∧
      (∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (Ctop j 0 l) (Cbot j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits (joinRow Ctop) (joinRow Cbot) j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (halves T0 j 0 l) (halves T1 j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits T0 T1 j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 8), ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (fun a _ ↦ actionDigits (joinRow top) (joinRow bot) j a l)) ∧
      outputs.1 = v0 ∧ outputs.2.1 = v1 ∧ outputs.2.2.1 = v2 ∧ outputs.2.2.2.1 = v3 ∧
      ∀ j : Fin 2, ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j = mulRow j * V1 +
        (pairMul hrow (actionDigits T0 T1 j) +
          (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) + halves Elut j)) * VK := by
  intro cIn Ctop Cbot A Mmul Emul mulRow Elut hrow top bot σc
  obtain ⟨w, x7, x9, x11, x12, x14, x16, x17, -, x18, x23, x24, x25, x26, x27, x28, -, x29, -, -, -,
    -, -, -, x31, -, -, x35, -, -, x40, x44, x46, -, -, -, -, -, -, x48, x59, x62, -, x63, -, x64,
    xout⟩ := h
  -- The state and the attribute package.
  obtain ⟨_, hw9⟩ := familyGet_eq (k := 0) x9
  obtain ⟨_, hw11⟩ := familyGet_eq (k := 0) x11
  obtain ⟨_, hw14⟩ := familyGet_eq (k := 0) x14
  obtain ⟨_, hw16⟩ := familyGet_eq (k := 0) x16
  have hw7 : w.w_7_0 = joinRow cIn := concat_joinRow (r := cIn) x7
  have hw12 : w.w_12_0 = joinRow Ctop := by
    rw [hw9, hw11] at x12
    exact concat_joinRow (r := Ctop) x12
  have hw17 : w.w_17_0 = joinRow Cbot := by
    rw [hw14, hw16] at x17
    exact concat_joinRow (r := Cbot) x17
  rw [hw7, hw12, hw17] at x18
  have hmul := colAction (row := cIn) (concatColumns_joinRow cIn) (tk := Ctop) (bk := Cbot)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (colLoopMul x18)
  -- The product's companion row.
  have hkey : ∀ k j, A k j = ![![w.w_23_0, w.w_24_0], ![w.w_26_0, w.w_27_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [A, inputKey, dynamicPrefix, schemaBytes, componentBytes, hashSample_eq x23,
        hashSample_eq x24, hashSample_eq x26, hashSample_eq x27]
  have hA0 : concatColumns (A 0 0) (A 0 1) w.w_25_0 := by rw [hkey, hkey]; exact x25
  have hA1 : concatColumns (A 1 0) (A 1 1) w.w_28_0 := by rw [hkey, hkey]; exact x28
  have hw29 := mulLoop hA0 hA1 x29
  have hw30 : ∀ j : Fin 2, halves (matrixAdd w.w_18_0 w.w_29_0) j = mulRow j := by
    intro j
    have h18 := hmul.1 j
    funext a l
    rw [Subsingleton.elim a 0]
    have h18l := congrFun (congrFun h18 0) l
    simp only [halves, rowWindow, matrixAdd, Matrix.add_apply, mulRow] at h18l ⊢
    rw [h18l, hw29]
  have hw31 : w.w_31_0 = mulRow 0 :=
    (sliceMatrix_window 0 x31 (by simp) (by omega)).trans (hw30 0)
  have hw48 : w.w_48_0 = mulRow 1 :=
    (sliceMatrix_window 8 x48 (by simp) (by omega)).trans (hw30 1)
  -- The identity branch.
  have hw35 := routing_one x35
  have hw40 := routing_one x40
  -- The companion row of the automorphism.
  have hw63 := lutLoop (A := Mmul) x59 x62 x63
  -- The branch of `K`.
  have hcount : Int.toNat 1 = 0 + 1 := rfl
  rw [hcount] at x64
  obtain ⟨cur, h0, hbranch⟩ := MxxIR.IterRuns.step_of_succ x64
  cases h0
  rw [hw31, hw48] at hbranch
  obtain ⟨g, VK, hg, hVK, hT, hC, hout⟩ := branch_rows hbranch
  have hg' := gadgetMatrixRuns_unique hg x44
  rw [hg'] at hVK
  have hE : w.w_63_0 = Elut := by
    funext a jj
    rw [Subsingleton.elim a 0, hw63]
  refine ⟨w.w_44_0, w.w_46_0, VK, x44, ?_, hVK, hmul.2, hT, ?_, ?_, ?_, ?_, ?_, fun j ↦ ?_⟩
  · convert x46 using 2
    simp only [lutScalarOne, lutScalar, hw35, hw40, id]
  · exact hC
  all_goals subst xout
  all_goals try rfl
  have hj := hout j
  rw [hE] at hj
  fin_cases j
  · simpa [σc, matrixMul] using hj
  · simpa [σc, matrixMul] using hj

end PeProof
