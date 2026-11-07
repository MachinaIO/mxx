import PeSetup

/-!
# Online evaluation

The Dynamic product of the initial state by the attribute package, then the coefficient LUT of the
predicate: the identity branch and the branch of `K`, each through its public scalar action.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The digits of column `jj` of the joined ciphertext `(top, bot)`. -/
noncomputable abbrev colDigits (top bot : ExactMatrix Q N 1 24) (jj : Fin 24) :
    ExactMatrix Q N (12 + 12) 1 :=
  columnDigits Q (12 + 12) Backend.backend 32768 12 16384 (top 0 jj) (bot 0 jj)


set_option maxHeartbeats 8000000 in
/-- The column loop of the product ends with the ciphertext action `c Dec(C)`, column by column, and
every column decomposition runs. -/
theorem colLoopMul {model : HashModel} {tape : SampleTape} {path : List Nat} {c : ExactMatrix Q N 1 24} {top bot : ExactMatrix Q N 1 24}
    {final : ExactMatrix Q N 1 24}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 24) ↦
      Stage_online_eval.sequential_generatedRoot_28 Backend.backend model tape (path ++ [28, i]) GeneratedClaim.stage_7_params i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 24) 0 final) (jj : Fin 24) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 24 = 24 * 1 := rfl
  rw [hcount] at h
  let F : Fin 24 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 24 → Prop := fun i _ ↦ ∀ jj : Fin 24, jj.val < i →
    ∃ D' : ExactMatrix Q N (12 + 12) 1,
      gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 16384 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 24 ↦ s)
    (fun i ↦ if h : i < 24 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 24) else 0) Inv
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 24 := by omega
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
theorem colLoopHelper {model : HashModel} {tape : SampleTape} {path : List Nat} {i0 : Nat} {c : ExactMatrix Q N 1 24} {top bot : ExactMatrix Q N 1 24}
    {final : ExactMatrix Q N 1 24}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 24) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_91_25 Backend.backend model tape (path ++ [25, i]) GeneratedClaim.stage_7_params i0 i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 24) 0 final) (jj : Fin 24) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 24 = 24 * 1 := rfl
  rw [hcount] at h
  let F : Fin 24 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 24 → Prop := fun i _ ↦ ∀ jj : Fin 24, jj.val < i →
    ∃ D' : ExactMatrix Q N (12 + 12) 1,
      gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 16384 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 24 ↦ s)
    (fun i ↦ if h : i < 24 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 24) else 0) Inv
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 24 := by omega
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
theorem colLoopSwitch {model : HashModel} {tape : SampleTape} {path : List Nat} {i0 : Nat} {c : ExactMatrix Q N 1 24} {top bot : ExactMatrix Q N 1 24}
    {final : ExactMatrix Q N 1 24}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 24) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_91_71 Backend.backend model tape (path ++ [71, i]) GeneratedClaim.stage_7_params i0 i
        (current, (c, (top, (bot, ())))) next) (Int.toNat 24) 0 final) (jj : Fin 24) :
    final 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits top bot jj) := by
  have hcount : Int.toNat 24 = 24 * 1 := rfl
  rw [hcount] at h
  let F : Fin 24 → Fin 1 → ExactPoly Q N := fun jj _ ↦ (c * colDigits top bot jj) 0 0
  let Inv : Nat → ExactMatrix Q N 1 24 → Prop := fun i _ ↦ ∀ jj : Fin 24, jj.val < i →
    ∃ D' : ExactMatrix Q N (12 + 12) 1,
      gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
        compactCenteredRebaseRuns 16384 D' (colDigits top bot jj)
  have hacc := IterRuns.accumulate (fun s : ExactMatrix Q N 1 24 ↦ s)
    (fun i ↦ if h : i < 24 * 1 then
      F ⟨i / 1, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % 1, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / 1 : Nat) : Int)) : ExactMatrix Q N 1 24) else 0) Inv
    (fun jj h ↦ absurd h (Nat.not_lt_zero _)) (fun i s t hi hq hb ↦ by
      obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
        -, -, -, -, -, -, x11, xout⟩ := hb
      have hi' : i < 24 := by omega
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
abbrev MulBlock := ExactMatrix Q N 1 24 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the product's companions at column `jj`. -/
noncomputable def mulCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12)
    (z0 z1 : Fin 576 → ExactMatrix Q N 1 12) (jj : Fin 24) (a : Fin 24) : ExactPoly Q N :=
  (z0 ⟨jj + 24 * a, by omega⟩ * keyDigits A a 0 + z1 ⟨jj + 24 * a, by omega⟩ * keyDigits A a 1) 0 0

/-- The expansions of a compact family. -/
noncomputable abbrev expandFam {k : Nat} (zc : Fin k → ExactMatrix PQ N 1 1) : Fin k → ExactMatrix Q N 1 12 :=
  fun i ↦ expand (zc i)

set_option maxHeartbeats 8000000 in
/-- One iteration `j` of the inner companion loop of the product at digit row `a` adds, at column
`j`, the expansion of companion `j + 24 a` paired with the digits `U`, and reads and expands the
next companion. -/
theorem mulInner_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {U : Fin 2 → ExactMatrix Q N 12 1} {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {a j : Nat}
    {m : Int} (hm : m = 0) (ha : a < 24) (hj : j < 24) {s t : MulBlock}
    (hs : s.2.1 = expand (zc0 ⟨j + 24 * a, by omega⟩) ∧ s.2.2.1 = expand (zc1 ⟨j + 24 * a, by omega⟩))
    (h : Stage_online_eval.sequential_sequential_generatedRoot_53_26 Backend.backend model tape path
      GeneratedClaim.stage_7_params a j (s.1, s.2.1, s.2.2.1, U 0, U 1, zc0, m, zc1, ()) t) :
    t.1 = s.1 + (expand (zc0 ⟨j + 24 * a, by omega⟩) * U 0 +
        expand (zc1 ⟨j + 24 * a, by omega⟩) * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.1 = expand (zc0 ⟨(j + 1) % 24 + 24 * a, by omega⟩) ∧
      t.2.2.1 = expand (zc1 ⟨(j + 1) % 24 + 24 * a, by omega⟩) := by
  obtain ⟨w9, w18, w19, w20, w22, w24, w25, w27, -, -, -, -, -, -, -, -, x9, -, -, -, -, x18, x19, x20,
    x22, -, -, x24, x25, x27, xout⟩ := h
  have hnext : 24 * Int.fmod (Int.ofNat a) 24 + Int.fmod (1 + Int.ofNat j) 24 =
      Int.ofNat ((j + 1) % 24 + 24 * a) := by
    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 24 by norm_num), Int.ofNat_eq_natCast]
    omega
  rw [hm, zero_mul, zero_add, hnext] at x18 x24
  obtain ⟨_, hw18⟩ := familyGet_eq x18
  obtain ⟨_, hw24⟩ := familyGet_eq x24
  have hw22 := expand_read x19 x20 x22
  have hw27 := expand_read x19 x25 x27
  have hw9 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x9
  subst hw18 hw24
  subst xout hw9 hw22 hw27
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, rfl, rfl⟩
  simp only [hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq]

set_option maxHeartbeats 8000000 in
/-- The inner companion loop of the product at digit row `a` ends with that digit row's companion
row, column by column. -/
theorem mulInner {model : HashModel} {tape : SampleTape} {path : List Nat}
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12) {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} (a : Fin 24)
    {m : Int} (hm : m = 0) {final : MulBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : MulBlock) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_53_26 Backend.backend model tape
        (path ++ [26, i]) GeneratedClaim.stage_7_params a i
        (current.1, (current.2.1, (current.2.2.1, (keyDigits A a 0, (keyDigits A a 1, (zc0,
          (m, (zc1, ())))))))) next) (Int.toNat 24)
      (0, (expand (zc0 ⟨24 * a, by omega⟩), (expand (zc1 ⟨24 * a, by omega⟩), ()))) final)
    (jj : Fin 24) :
    final.1 0 jj = mulCompanionSum A (expandFam zc0) (expandFam zc1) jj a := by
  let z0 := expandFam zc0
  let z1 := expandFam zc1
  let Inv : Nat → MulBlock → Prop := fun i s ↦
    s.2.1 = z0 ⟨i % 24 + 24 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i % 24 + 24 * a, by omega⟩
  have hInv0 : Inv 0 (0, (expand (zc0 ⟨24 * a, by omega⟩), (expand (zc1 ⟨24 * a, by omega⟩), ()))) :=
    ⟨congrArg z0 (Fin.ext (by simp)), congrArg z1 (Fin.ext (by simp))⟩
  have hstep : ∀ i (s t : MulBlock) (hi : i < 24), Inv i s →
      Stage_online_eval.sequential_sequential_generatedRoot_53_26 Backend.backend model tape
        (path ++ [26, i]) GeneratedClaim.stage_7_params a i
        (s.1, (s.2.1, (s.2.2.1, (keyDigits A a 0, (keyDigits A a 1, (zc0, (m, (zc1, ())))))))) t →
      (∀ w, t.1 0 w = s.1 0 w + mulCompanionSum A z0 z1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) ∧ Inv (i + 1) t := by
    intro i s t hi hq hb
    have hq' : s.2.1 = expand (zc0 ⟨i + 24 * a, by omega⟩) ∧
        s.2.2.1 = expand (zc1 ⟨i + 24 * a, by omega⟩) := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext (by simp [Nat.mod_eq_of_lt hi]))),
        h1.trans (congrArg z1 (Fin.ext (by simp [Nat.mod_eq_of_lt hi])))⟩
    obtain ⟨h0, h1, h2⟩ := mulInner_step (U := keyDigits A a) hm a.isLt hi hq' hb
    refine ⟨fun w ↦ ?_, h1, h2⟩
    rw [h0]
    rfl
  have hk := IterRuns.accumulate_entries (fun s : MulBlock ↦ s.1)
    (fun w (j : Fin 24) ↦ mulCompanionSum A z0 z1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w)
    Inv hInv0 (fun i s t hi hq hb ↦ hstep i s t hi hq hb) h jj
  rw [hk, sum_unitRow_mul (fun j ↦ mulCompanionSum A z0 z1 j a) jj]
  simp [z0, z1]

set_option maxHeartbeats 16000000 in
/-- One iteration `a` of the companion loop of the product decomposes key column `a` once and adds
the companion row of digit row `a`, column by column. -/
theorem mulLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {m : Int} (hm : m = 0) {i : Nat} (hi : i < 24)
    {s t : ExactMatrix Q N 1 24}
    (h : Stage_online_eval.sequential_generatedRoot_53 Backend.backend model tape path
      GeneratedClaim.stage_7_params i (s, zc0, m, zc1, A0, A1, ()) t) (jj : Fin 24) :
    t 0 jj = s 0 jj + mulCompanionSum A (expandFam zc0) (expandFam zc1) jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, x10, x11, x12, x14, -, -, x16, x17, x19, -, -, -,
    -, -, -, -, -, x21, x22, -, -, -, -, -, -, -, -, x24, x25, -, hin, x27, xout⟩ := h
  have hdiv : Int.fdiv (Int.ofNat i) 24 = 0 := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 24 = ((i : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 24 * Int.fmod (Int.ofNat i) 24 = Int.ofNat (24 * i) := by
    rw [hrow]
    simp only [Int.ofNat_eq_natCast]
    omega
  rw [hdiv] at x2
  rw [hrow] at x21 x24
  rw [hm, zero_mul, zero_add, hfirst] at x10 x16
  have hw2 := unitRow_window (c := 0) (by omega) x2
  obtain ⟨_, hw10⟩ := familyGet_eq x10
  obtain ⟨_, hw16⟩ := familyGet_eq x16
  have hw14 := expand_read x11 x12 x14
  have hw19 := expand_read x11 x17 x19
  rw [hw10] at hw14
  rw [hw16] at hw19
  have hentry : ∀ {r : Fin 2} {Ar : ExactMatrix Q N 1 24} {x : ExactMatrix Q N 1 1}
      {u : ExactMatrix Q N 12 1}, concatColumns (A r 0) (A r 1) Ar →
      sliceMatrix Ar 0 1 ((i : Nat) : Int) (1 + ((i : Nat) : Int)) x →
      gadgetDecomposeRuns Backend.backend 32768 12 x u → u = keyDigits A ⟨i, hi⟩ r := by
    intro r Ar x u hAr hx hu
    obtain ⟨_, hx0⟩ := sliceMatrix_column hx
    rw [concatColumns_entry hAr, pair_eta] at hx0
    have hxe : x = fun _ _ ↦ joinedEntry (A r) ⟨i, hi⟩ := by
      funext a b
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hx0]
      congr 1
    rw [hxe] at hu
    exact entryDigits_eq hu
  have hw22 := hentry hA0 x21 x22
  have hw25 := hentry hA1 x24 x25
  rw [hw22, hw25, hw14, hw19] at hin
  have hblock := mulInner (model := model) (tape := tape) (path := path) A ⟨i, hi⟩ hm hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 24}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x27]
  exact congrArg _ hblock

set_option maxHeartbeats 8000000 in
/-- The companion loop of the product ends with the companion row, column by column. -/
theorem mulLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {m : Int} (hm : m = 0)
    {final : ExactMatrix Q N 1 24}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 24) ↦
      Stage_online_eval.sequential_generatedRoot_53 Backend.backend model tape (path ++ [53, i])
        GeneratedClaim.stage_7_params i (current, (zc0, (m, (zc1, (A0, (A1, ())))))) next)
      (Int.toNat 24) 0 final) (jj : Fin 24) :
    final 0 jj = ∑ a, mulCompanionSum A (expandFam zc0) (expandFam zc1) jj a := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have hk := IterRuns.accumulate_entries (fun s : ExactMatrix Q N 1 24 ↦ s)
    (fun w a ↦ mulCompanionSum A (expandFam zc0) (expandFam zc1) w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ mulLoop_step hA0 hA1 hm hi hb w, trivial⟩) h jj
  rw [hk]
  simp

/-! ## The companion row of the automorphism -/

/-- The key `σ_K(A)` routed by the automorphism of the LUT branch. -/
noncomputable def routedKey (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12) :
    Fin 2 → Fin 2 → ExactMatrix Q N 1 12 :=
  fun r j ↦ substituteMatrix K hK (A r j)

/-- The digit-row sum `z_0 u_0 + z_1 u_1` of the automorphism companions at column `jj`, against
the digits `u` of the routed key. -/
noncomputable def lutCompanionSum (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12)
    (z0 z1 : Fin 576 → ExactMatrix Q N 1 12) (jj : Fin 24) (a : Fin 24) : ExactPoly Q N :=
  (z0 ⟨jj + 24 * a, by omega⟩ * keyDigits (routedKey A) a 0 +
    z1 ⟨jj + 24 * a, by omega⟩ * keyDigits (routedKey A) a 1) 0 0

set_option maxHeartbeats 8000000 in
theorem lutInner_step {model : HashModel} {tape : SampleTape} {path : List Nat}
    {U : Fin 2 → ExactMatrix Q N 12 1} {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {a j : Nat}
    (ha : a < 24) (hj : j < 24) {s t : MulBlock}
    (hs : s.2.1 = expand (zc0 ⟨j + 24 * a, by omega⟩) ∧ s.2.2.1 = expand (zc1 ⟨j + 24 * a, by omega⟩))
    (h : Stage_online_eval.sequential_sequential_generatedRoot_90_24 Backend.backend model tape path
      GeneratedClaim.stage_7_params a j (s.1, s.2.1, s.2.2.1, U 0, U 1, zc0, zc1, ()) t) :
    t.1 = s.1 + (expand (zc0 ⟨j + 24 * a, by omega⟩) * U 0 +
        expand (zc1 ⟨j + 24 * a, by omega⟩) * U 1) 0 0 • unitRow ((j : Nat) : Int) ∧
      t.2.1 = expand (zc0 ⟨(j + 1) % 24 + 24 * a, by omega⟩) ∧
      t.2.2.1 = expand (zc1 ⟨(j + 1) % 24 + 24 * a, by omega⟩) := by
  obtain ⟨w9, w14, w15, w16, w18, w20, w21, w23, -, -, -, -, -, -, -, -, x9, -, -, -, -, -, -, -, x14,
    x15, x16, x18, -, -, x20, x21, x23, xout⟩ := h
  have hnext : 24 * Int.fmod (Int.ofNat a) 24 + Int.fmod (1 + Int.ofNat j) 24 +
      576 * MxxIR.selectInt (Int.fdiv (Int.ofNat a) 24) [0] = Int.ofNat ((j + 1) % 24 + 24 * a) := by
    have hd : Int.fdiv (Int.ofNat a) 24 = 0 := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
      simp only [Int.ofNat_eq_natCast]
      omega
    rw [hd]
    simp only [Int.fmod_eq_emod_of_nonneg _ (show (0 : Int) ≤ 24 by norm_num), Int.ofNat_eq_natCast]
    simp [MxxIR.selectInt]
    omega
  rw [hnext] at x14 x20
  obtain ⟨_, hw14⟩ := familyGet_eq x14
  obtain ⟨_, hw20⟩ := familyGet_eq x20
  have hw18 := expand_read x15 x16 x18
  have hw23 := expand_read x15 x21 x23
  have hw9 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x9
  subst hw14 hw20
  subst xout hw9 hw18 hw23
  obtain ⟨hs0, hs1⟩ := hs
  refine ⟨?_, rfl, rfl⟩
  simp only [hs0, hs1, matrixAdd, matrixMul, matrixMulScalarLeft_eq]

set_option maxHeartbeats 8000000 in
theorem lutInner {model : HashModel} {tape : SampleTape} {path : List Nat}
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12) {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} (a : Fin 24)
    {final : MulBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : MulBlock) ↦
      Stage_online_eval.sequential_sequential_generatedRoot_90_24 Backend.backend model tape
        (path ++ [24, i]) GeneratedClaim.stage_7_params a i
        (current.1, (current.2.1, (current.2.2.1, (keyDigits (routedKey A) a 0,
          (keyDigits (routedKey A) a 1, (zc0, (zc1, ()))))))) next) (Int.toNat 24)
      (0, (expand (zc0 ⟨24 * a, by omega⟩), (expand (zc1 ⟨24 * a, by omega⟩), ()))) final)
    (jj : Fin 24) :
    final.1 0 jj = lutCompanionSum A (expandFam zc0) (expandFam zc1) jj a := by
  let z0 := expandFam zc0
  let z1 := expandFam zc1
  let Inv : Nat → MulBlock → Prop := fun i s ↦
    s.2.1 = z0 ⟨i % 24 + 24 * a, by omega⟩ ∧ s.2.2.1 = z1 ⟨i % 24 + 24 * a, by omega⟩
  have hInv0 : Inv 0 (0, (expand (zc0 ⟨24 * a, by omega⟩), (expand (zc1 ⟨24 * a, by omega⟩), ()))) :=
    ⟨congrArg z0 (Fin.ext (by simp)), congrArg z1 (Fin.ext (by simp))⟩
  have hstep : ∀ i (s t : MulBlock) (hi : i < 24), Inv i s →
      Stage_online_eval.sequential_sequential_generatedRoot_90_24 Backend.backend model tape
        (path ++ [24, i]) GeneratedClaim.stage_7_params a i
        (s.1, (s.2.1, (s.2.2.1, (keyDigits (routedKey A) a 0, (keyDigits (routedKey A) a 1,
          (zc0, (zc1, ()))))))) t →
      (∀ w, t.1 0 w = s.1 0 w + lutCompanionSum A z0 z1 ⟨i, hi⟩ a *
          (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) ∧ Inv (i + 1) t := by
    intro i s t hi hq hb
    have hq' : s.2.1 = expand (zc0 ⟨i + 24 * a, by omega⟩) ∧
        s.2.2.1 = expand (zc1 ⟨i + 24 * a, by omega⟩) := by
      obtain ⟨h0, h1⟩ := hq
      exact ⟨h0.trans (congrArg z0 (Fin.ext (by simp [Nat.mod_eq_of_lt hi]))),
        h1.trans (congrArg z1 (Fin.ext (by simp [Nat.mod_eq_of_lt hi])))⟩
    obtain ⟨h0, h1, h2⟩ := lutInner_step (U := keyDigits (routedKey A) a) a.isLt hi hq' hb
    refine ⟨fun w ↦ ?_, h1, h2⟩
    rw [h0]
    rfl
  have hk := IterRuns.accumulate_entries (fun s : MulBlock ↦ s.1)
    (fun w (j : Fin 24) ↦ lutCompanionSum A z0 z1 j a * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w)
    Inv hInv0 (fun i s t hi hq hb ↦ hstep i s t hi hq hb) h jj
  rw [hk, sum_unitRow_mul (fun j ↦ lutCompanionSum A z0 z1 j a) jj]
  simp [z0, z1]

/-- Substitutions by equal indices agree. -/
theorem substitution_congr {R : Type} [CommRing R] {i j : Nat} (h : i = j) (hi : Odd i)
    (hj : Odd j) : (substitution (n := N) (R := R) i hi : Negacyclic N R →+* Negacyclic N R) =
      substitution j hj := by
  subst h
  rfl

/-- The digits of an entry routed by `σ_K`. -/
theorem routed_entry_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {r : Fin 2} {a : Fin 24}
    {Ar : ExactMatrix Q N 1 24} {x y : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 12 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar)
    (hx : sliceMatrix Ar 0 1 ((a : Nat) : Int) (1 + ((a : Nat) : Int)) x)
    (hy : ringAutomorphismRuns (MxxIR.selectInt 0 [32769]) x y)
    (hu : gadgetDecomposeRuns Backend.backend 32768 12 y u) : u = keyDigits (routedKey A) a r := by
  obtain ⟨_, hx0⟩ := sliceMatrix_column hx
  rw [concatColumns_entry hAr, pair_eta] at hx0
  have hxe : x = fun _ _ ↦ joinedEntry (A r) a := by
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0, hx0]
    congr 1
  obtain ⟨hodd, hye⟩ := ringAutomorphismRuns_eq hQ hN hy
  have hidx : (MxxIR.selectInt 0 [32769]).toNat = K := by decide
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
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {i : Nat} (hi : i < 24) {s t : ExactMatrix Q N 1 24}
    (h : Stage_online_eval.sequential_generatedRoot_90 Backend.backend model tape path
      GeneratedClaim.stage_7_params i (s, zc0, zc1, A0, A1, ()) t) (jj : Fin 24) :
    t 0 jj = s 0 jj + lutCompanionSum A (expandFam zc0) (expandFam zc1) jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, -, -, -, x6, x7, x8, x10, -, -, x12, x13, x15,
    -, -, -, -, -, -, -, -, x17, -, -, -, x18, x19, -, -, -, -, -, -, -, -, x21, -, -, -, x22, x23, -,
    hin, x25, xout⟩ := h
  have hdiv : Int.fdiv (Int.ofNat i) 24 = 0 := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hrow : Int.fmod (Int.ofNat i) 24 = ((i : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
    simp only [Int.ofNat_eq_natCast]
    omega
  have hfirst : 24 * Int.fmod (Int.ofNat i) 24 + 576 * MxxIR.selectInt (Int.fdiv (Int.ofNat i) 24) [0] =
      Int.ofNat (24 * i) := by
    rw [hrow, hdiv]
    simp [MxxIR.selectInt]
  rw [hdiv] at x2 x18 x22
  rw [hrow] at x17 x21
  rw [hfirst] at x6 x12
  have hw2 := unitRow_window (c := 0) (by omega) x2
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨_, hw12⟩ := familyGet_eq x12
  have hw10 := expand_read x7 x8 x10
  have hw15 := expand_read x7 x13 x15
  rw [hw6] at hw10
  rw [hw12] at hw15
  have hw19 := routed_entry_digits (a := ⟨i, hi⟩) hA0 x17 x18 x19
  have hw23 := routed_entry_digits (a := ⟨i, hi⟩) hA1 x21 x22 x23
  rw [hw19, hw23, hw10, hw15] at hin
  have hblock := lutInner (model := model) (tape := tape) (path := path) A ⟨i, hi⟩ hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 24}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x25]
  exact congrArg _ hblock

set_option maxHeartbeats 8000000 in
/-- The companion loop of the automorphism ends with its companion row, column by column. -/
theorem lutLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {zc0 zc1 : Fin 576 → ExactMatrix PQ N 1 1} {final : ExactMatrix Q N 1 24}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q N 1 24) ↦
      Stage_online_eval.sequential_generatedRoot_90 Backend.backend model tape (path ++ [90, i])
        GeneratedClaim.stage_7_params i (current, (zc0, (zc1, (A0, (A1, ()))))) next)
      (Int.toNat 24) 0 final) (jj : Fin 24) :
    final 0 jj = ∑ a, lutCompanionSum A (expandFam zc0) (expandFam zc1) jj a := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have hk := IterRuns.accumulate_entries (fun s : ExactMatrix Q N 1 24 ↦ s)
    (fun w a ↦ lutCompanionSum A (expandFam zc0) (expandFam zc1) w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ lutLoop_step hA0 hA1 hi hb w, trivial⟩) h jj
  rw [hk]
  simp

/-- The digits of output column `l` of component `j` of the action of `(top, bot)`. -/
noncomputable def actionDigits (top bot : ExactMatrix Q N 1 24) : Fin 2 → ExactMatrix Q N (12 + 12) 12 :=
  fun j a l ↦ colDigits top bot ⟨12 * (j : Nat) + l, by omega⟩ a 0

/-- A column loop of a joined row `c = [row₀ | row₁]` by the joined matrix `(top, bot)` with
components `tk`, `bk`: component `j` of the result is `row` times the digits of the columns of
component `j`, and every column decomposition runs. -/
theorem colAction {c : ExactMatrix Q N 1 24} {row : Fin 2 → ExactMatrix Q N 1 12}
    (hc : concatColumns (row 0) (row 1) c) {top bot : ExactMatrix Q N 1 24}
    {tk bk : Fin 2 → ExactMatrix Q N 1 12} (htop : ∀ p : Fin 24, top 0 p = joinedEntry tk p)
    (hbot : ∀ p : Fin 24, bot 0 p = joinedEntry bk p) {acc : ExactMatrix Q N 1 24}
    (hacc : ∀ jj : Fin 24, acc 0 jj = (c * colDigits top bot jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top 0 jj) (bot 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits top bot jj)) :
    (∀ j : Fin 2, rowWindow 12 (12 * j) (by omega) acc = pairMul row (actionDigits top bot j)) ∧
      ∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (tk j 0 l) (bk j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits top bot j a l) := by
  refine ⟨fun j ↦ ?_, fun j l ↦ ?_⟩
  · rw [window_mul_columns (m := 12) (k := 12) hc (fun p ↦ (hacc p).1) j (by omega)]
    rfl
  · obtain ⟨D', h1, h2⟩ := (hacc ⟨12 * (j : Nat) + l, by omega⟩).2
    rw [htop, hbot, joinedEntry_pair, joinedEntry_pair] at h1
    refine ⟨D', h1, ?_⟩
    have hcol : colDigits top bot ⟨12 * (j : Nat) + l, by omega⟩ =
        fun a _ ↦ actionDigits top bot j a l := by
      funext a b
      rw [Subsingleton.elim b 0]
      rfl
    rwa [hcol] at h2

/-! ## The LUT branches -/

/-- `2⁻¹ mod Q`. -/
abbrev inverse2 : Int := 754577806326161481947490779154010711265593984364478465

/-- The scalar `D_k = 2⁻¹ (t₀ σ_k(X⁰) + t₁ σ_k(-X^32767))` of the branch of `k` for the table
`(t₀, t₁)` of the key's program, as the graph computes it, with `σ` the routing of `k`. -/
noncomputable def lutScalar (t0 t1 : Int) (σ : ExactMatrix Q N 1 1 → ExactMatrix Q N 1 1) :
    ExactMatrix Q N 1 1 :=
  matrixMulScalarLeft (matrixPolynomial [inverse2])
    (matrixAdd (matrixMulScalarLeft (liftInteger t0) (σ (rotationPolynomial 0)))
      (matrixMulScalarLeft (liftInteger t1) (σ (matrixNeg (rotationPolynomial 32767)))))

/-- The two halves of a row, concatenated, are the row. -/
theorem concat_windows {x y : ExactMatrix Q N 1 24}
    (h : concatColumns (rowWindow 12 0 (by omega) x) (rowWindow 12 12 (by omega) x) y) : y = x := by
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
theorem selectK : MxxIR.selectInt (Int.ofNat 0) [32769] = (K : Int) := by decide

/-- A routing by the branch selector is the substitution of `K`. -/
theorem routing_K {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat 0) [32769]) x y) :
    y = substituteMatrix K hK x := by
  rw [selectK] at h
  obtain ⟨hodd, hy⟩ := ringAutomorphismRuns_eq hQ hN h
  rw [hy]
  funext r c
  exact congrFun (congrArg DFunLike.coe (substitution_congr (by decide) hodd hK)) _

/-- The joined row `[r₀ | r₁]`. -/
def joinRow (r : Fin 2 → ExactMatrix Q N 1 12) : ExactMatrix Q N 1 24 := fun _ p ↦ joinedEntry r p

theorem concat_joinRow {r : Fin 2 → ExactMatrix Q N 1 12} {y : ExactMatrix Q N 1 24}
    (h : concatColumns (r 0) (r 1) y) : y = joinRow r := by
  funext a p
  rw [Subsingleton.elim a 0, concatColumns_entry h, pair_eta]
  rfl

/-- The halves of a joined row. -/
def halves (x : ExactMatrix Q N 1 24) : Fin 2 → ExactMatrix Q N 1 12 :=
  fun j ↦ rowWindow 12 (12 * j) (by omega) x

/-- Entry `p` of a joined row, by the half it lies in. -/
theorem joinedEntry_split (r : Fin 2 → ExactMatrix Q N 1 12) (p : Fin 24) :
    joinedEntry r p = if h : p.val < 12 then r 0 0 ⟨p.val, h⟩ else r 1 0 ⟨p.val - 12, by omega⟩ := by
  unfold joinedEntry Fin.addCases
  split <;> simp_all <;> rfl

theorem concatColumns_joinRow (r : Fin 2 → ExactMatrix Q N 1 12) :
    concatColumns (r 0) (r 1) (joinRow r) := by
  intro a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split]

theorem joinRow_halves (x : ExactMatrix Q N 1 24) : joinRow (halves x) = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp only [joinRow, joinedEntry_split, halves, rowWindow]
  split
  · simp
  · congr 1
    ext
    simp
    omega

theorem entry_halves (x : ExactMatrix Q N 1 24) (p : Fin 24) : x 0 p = joinedEntry (halves x) p := by
  conv_lhs => rw [← joinRow_halves x]
  rfl

theorem rowWindow_full (x : ExactMatrix Q N 1 24) : rowWindow 24 0 (by omega) x = x := by
  funext a p
  rw [Subsingleton.elim a 0]
  simp [rowWindow]

set_option maxHeartbeats 64000000 in
/-- The branch of `K` of the LUT: from the accumulated identity branch `acc`, it adds, per
component `j`, the helper action `h Dec(T)`, the switch `σ_K(c) Dec(C_K)`, and the companion row
`E`, all through the scalar action `V = Dec(D_K g)`. `C_K` has the rows `σ_K(H⁰)` and
`σ_K(H¹) - σ_K(h)` of the helper key `H` of `K` and the helper `h` of `K = K⁻¹`; the helper and
its key are read compactly and expanded. -/
theorem branch_rows {model : HashModel} {tape : SampleTape} {path : List Nat}
    {acc0 acc1 : ExactMatrix Q N 1 12} {hf0 hf1 : Fin 1 → ExactMatrix PQ N 1 1}
    {T0 T1 : ExactMatrix Q N 1 24} {c0 c1 : ExactMatrix Q N 1 12} {key : ByteArray}
    {E : ExactMatrix Q N 1 24} {t0 t1 : Int}
    {outputs : ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit}
    (h : Stage_online_eval.sequential_generatedRoot_91 Backend.backend model tape path
      GeneratedClaim.stage_7_params 0 (acc0, acc1, hf0, hf1, T0, T1, c0, c1, key, E, t0, t1, ()) outputs) :
    let hrow : Fin 2 → ExactMatrix Q N 1 12 := ![expand (hf0 0), expand (hf1 0)]
    let top : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 12 := ![substituteMatrix K hK c0, substituteMatrix K hK c1]
    ∃ (g : ExactMatrix Q N 1 12) (V : ExactMatrix Q N 12 12),
      gadgetMatrixRuns Backend.backend 32768 12 g ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g) V ∧
      (∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (halves T0 j 0 l) (halves T1 j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits T0 T1 j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits (joinRow top) (joinRow bot) j a l)) ∧
      ∀ j : Fin 2, ![outputs.1, outputs.2.1] j = ![acc0, acc1] j +
        (pairMul hrow (actionDigits T0 T1 j) + (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) +
          halves E j)) * V := by
  intro hrow top bot σc
  obtain ⟨w, -, -, -, -, x4, x5, x6, x8, -, -, x10, x11, x13, x14, -, -, -, -, -, -, x16, -, -, -, -, -,
    -, x17, -, -, -, -, -, -, x18, x19, -, -, -, -, -, -, x21, -, -, -, -, -, -, x22, -, -, -, -, -, -,
    x23, x24, -, x25, -, -, -, -, -, -, x26, -, -, x29, -, -, x31, x32, -, -, x34, x35, x36, x38, -, -,
    x39, -, -, x40, x41, x43, -, -, x44, x45, -, -, x46, x47, x49, -, -, x50, -, -, -, -, x52, x53, x54,
    x56, -, -, x57, -, -, x59, x60, x62, -, -, x63, -, -, x64, x65, x67, -, -, x68, x70, -, x71, -, -,
    -, -, -, -, x72, -, -, -, -, -, -, x74, -, -, -, -, -, -, x75, -, -, -, -, x82, -, -, -, -, x88,
    x92, x94, -, -, -, -, -, -, x98, -, -, -, -, -, -, x99, -, -, -, -, -, -, x100, xout⟩ := h
  have hzero : MxxIR.selectInt (Int.ofNat 0) [0] = Int.ofNat 0 := by decide
  -- The helper of `K`, expanded.
  rw [hzero] at x4 x10
  obtain ⟨_, hw4⟩ := familyGet_eq x4
  obtain ⟨_, hw10⟩ := familyGet_eq x10
  have hw8 := expand_read x5 x6 x8
  have hw13 := expand_read x5 x11 x13
  rw [hw4] at hw8
  rw [hw10] at hw13
  rw [hw8, hw13] at x14
  have hw14 : w.w_14_0 = joinRow hrow := concat_joinRow (r := hrow) x14
  -- The switched key rows.
  have hw16 := (sliceMatrix_window 0 x16 (by simp) (by omega)).trans (rowWindow_full T0)
  have hw21 := (sliceMatrix_window 0 x21 (by simp) (by omega)).trans (rowWindow_full T1)
  rw [hw16] at x17 x18
  rw [hw21] at x22 x23
  have hw17 := sliceMatrix_window 0 x17 (by simp) (by omega)
  have hw18 := sliceMatrix_window 12 x18 (by simp) (by omega)
  have hw22 := sliceMatrix_window 0 x22 (by simp) (by omega)
  have hw23 := sliceMatrix_window 12 x23 (by simp) (by omega)
  rw [hw17, hw18] at x19
  rw [hw22, hw23] at x24
  have hw19 : w.w_19_0 = T0 := (concat_joinRow (r := halves T0) x19).trans (joinRow_halves T0)
  have hw24 : w.w_24_0 = T1 := (concat_joinRow (r := halves T1) x24).trans (joinRow_halves T1)
  -- The helper action.
  rw [hw14, hw19, hw24] at x25
  have hhelp := colAction (row := hrow) (concatColumns_joinRow hrow) (tk := halves T0) (bk := halves T1)
    (entry_halves T0) (entry_halves T1)
    (colLoopHelper (i0 := 0) x25)
  -- The routed state rows.
  have hw29 := routing_K x29
  have hw31 := routing_K x31
  rw [hw29, hw31] at x32
  have hw32 : w.w_32_0 = joinRow σc := concat_joinRow (r := σc) x32
  -- The switching ciphertext `C_K`, from the expanded helper key and helper.
  have hkey : ∀ k j, helperKeyC model key k j = ![![w.w_34_0, w.w_40_0], ![w.w_46_0, w.w_59_0]] k j := by
    intro k j
    rw [selectK] at x34 x40 x46 x59
    fin_cases k <;> fin_cases j <;>
      simp [helperKeyC, dynamicPrefix, schemaBytes, componentBytes, hashSample_eq x34,
        hashSample_eq x40, hashSample_eq x46, hashSample_eq x59]
  have hE : ∀ k j, helperKey model key k j = ![![w.w_38_0, w.w_43_0], ![w.w_49_0, w.w_62_0]] k j := by
    have h38 := expand_read x35 x36 x38
    have h43 := expand_read x35 x41 x43
    have h49 := expand_read x35 x47 x49
    have h62 := expand_read x35 x60 x62
    intro k j
    fin_cases k <;> fin_cases j <;> simp [helperKey, expandKey, hkey, h38, h43, h49, h62]
  rw [hzero] at x52 x64
  obtain ⟨_, hw52⟩ := familyGet_eq x52
  obtain ⟨_, hw64⟩ := familyGet_eq x64
  have hw56 := expand_read x53 x54 x56
  have hw67 := expand_read x53 x65 x67
  rw [hw52] at hw56
  rw [hw64] at hw67
  have hw39 := routing_K x39
  have hw44 := routing_K x44
  have hw50 := routing_K x50
  have hw57 := routing_K x57
  have hw63 := routing_K x63
  have hw68 := routing_K x68
  have htop : ![w.w_39_0, w.w_44_0] = top := by
    funext j
    fin_cases j <;> simp [top, hw39, hw44, hE]
  have hbot : ![matrixSub w.w_50_0 w.w_57_0, matrixSub w.w_63_0 w.w_68_0] = bot := by
    funext j
    fin_cases j <;> simp [bot, hrow, hw50, hw57, hw63, hw68, hw56, hw67, hE, matrixSub]
  have hw45 : w.w_45_0 = joinRow top := by
    rw [← htop]
    exact concat_joinRow (r := ![w.w_39_0, w.w_44_0]) x45
  have hw70 : w.w_70_0 = joinRow bot := by
    rw [← hbot]
    exact concat_joinRow (r := ![matrixSub w.w_50_0 w.w_57_0, matrixSub w.w_63_0 w.w_68_0]) x70
  rw [hw32, hw45, hw70] at x71
  have hswitch := colAction (row := σc) (concatColumns_joinRow σc) (tk := top) (bk := bot)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (colLoopSwitch (i0 := 0) x71)
  -- The companion row and the halves of the actions.
  have hw74 := (sliceMatrix_window 0 x74 (by simp) (by omega)).trans (rowWindow_full E)
  rw [hw74] at x75 x100
  have hw75 : w.w_75_0 = halves E 0 := sliceMatrix_window 0 x75 (by simp) (by omega)
  have hw100 : w.w_100_0 = halves E 1 := sliceMatrix_window 12 x100 (by simp) (by omega)
  have hw26 : w.w_26_0 = pairMul hrow (actionDigits T0 T1 0) :=
    (sliceMatrix_window 0 x26 (by simp) (by omega)).trans (hhelp.1 0)
  have hw98 : w.w_98_0 = pairMul hrow (actionDigits T0 T1 1) :=
    (sliceMatrix_window 12 x98 (by simp) (by omega)).trans (hhelp.1 1)
  have hw72 : w.w_72_0 = pairMul σc (actionDigits (joinRow top) (joinRow bot) 0) :=
    (sliceMatrix_window 0 x72 (by simp) (by omega)).trans (hswitch.1 0)
  have hw99 : w.w_99_0 = pairMul σc (actionDigits (joinRow top) (joinRow bot) 1) :=
    (sliceMatrix_window 12 x99 (by simp) (by omega)).trans (hswitch.1 1)
  -- The scalar action.
  have hw82 := routing_K x82
  have hw88 := routing_K x88
  refine ⟨w.w_92_0, w.w_94_0, x92, ?_, hhelp.2, hswitch.2, fun j ↦ ?_⟩
  · convert x94 using 2
    simp only [lutScalar, hw82, hw88]
  · subst xout
    fin_cases j
    · simp [matrixAdd, matrixMul, hw26, hw72, hw75]
    · simp [matrixAdd, matrixMul, hw98, hw99, hw100]

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

/-- The scalar `D_1` of the identity branch for the table `(t₀, t₁)`. -/
noncomputable abbrev lutScalarOne (t0 t1 : Int) : ExactMatrix Q N 1 1 := lutScalar t0 t1 id

/-- A read from a one-member family is at index 0. -/
theorem familyGet_one {α : Type} {f : Fin 1 → α} {i : Int} {out : α} (h : familyGetDynamic f i out) :
    i = 0 ∧ out = f 0 := by
  obtain ⟨position, hposition, hout⟩ := h
  rw [Subsingleton.elim position 0] at hposition hout
  exact ⟨hposition.symm, hout⟩

set_option maxHeartbeats 64000000 in
/-- The output rows of online evaluation: the product row `c Dec(C) + E` of the initial state and
the attribute package, then the LUT `D_1 (·) + D_K (Aut_K(·))` through the public scalar actions;
the key ports pass through from preprocessing. The initial state, the companions, and the helper
are read compactly and expanded. -/
theorem online_rows {model : HashModel} {tape : SampleTape} {path : List Nat}
    {v0 v1 v2 v3 : ExactMatrix Q N 1 12} {c0a c0b : ExactMatrix PQ N 1 1}
    {C00 C01 C10 C11 : Fin 1 → ExactMatrix Q N 1 12}
    {z0 z1 : Fin 576 → ExactMatrix PQ N 1 1} {key : ByteArray}
    {hf0 hf1 : Fin 1 → ExactMatrix PQ N 1 1} {T0 T1 : ExactMatrix Q N 1 24}
    {za0 za1 : Fin 576 → ExactMatrix PQ N 1 1} {M00 M01 M10 M11 : ExactMatrix Q N 1 12}
    {reads : Fin 1 → Int} {tables : Fin 2 → Int}
    {outputs : ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 ×
      ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit}
    (h : Stage_online_eval.generatedRoot Backend.backend model tape path
      GeneratedClaim.stage_7_params (v0, v1, v2, v3, c0a, c0b, C00, reads, C01, C10, C11, z0, z1, key,
        tables, hf0, hf1, T0, T1, za0, za1, M00, M01, M10, M11, ()) outputs) :
    let cIn : Fin 2 → ExactMatrix Q N 1 12 := ![expand c0a, expand c0b]
    let Ctop : Fin 2 → ExactMatrix Q N 1 12 := ![C00 0, C01 0]
    let Cbot : Fin 2 → ExactMatrix Q N 1 12 := ![C10 0, C11 0]
    let A := inputKey model key
    let Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 12 := ![![M00, M01], ![M10, M11]]
    let Emul : ExactMatrix Q N 1 24 := fun _ jj ↦ ∑ a, mulCompanionSum A (expandFam z0) (expandFam z1) jj a
    let mulRow : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦
      pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j
    let Elut : ExactMatrix Q N 1 24 := fun _ jj ↦
      ∑ a, lutCompanionSum Mmul (expandFam za0) (expandFam za1) jj a
    let hrow : Fin 2 → ExactMatrix Q N 1 12 := ![expand (hf0 0), expand (hf1 0)]
    let top : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 12 := fun j ↦ substituteMatrix K hK (mulRow j)
    ∃ (g : ExactMatrix Q N 1 12) (V1 VK : ExactMatrix Q N 12 12),
      gadgetMatrixRuns Backend.backend 32768 12 g ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalarOne (tables 0) (tables 1)) g) V1 ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalar (tables 0) (tables 1) (substituteMatrix K hK)) g) VK ∧
      (∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (Ctop j 0 l) (Cbot j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits (joinRow Ctop) (joinRow Cbot) j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (halves T0 j 0 l) (halves T1 j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits T0 T1 j a l)) ∧
      (∀ (j : Fin 2) (l : Fin 12), ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
          compactCenteredRebaseRuns 16384 D' (fun a _ ↦ actionDigits (joinRow top) (joinRow bot) j a l)) ∧
      outputs.1 = v0 ∧ outputs.2.1 = v1 ∧ outputs.2.2.1 = v2 ∧ outputs.2.2.2.1 = v3 ∧
      ∀ j : Fin 2, ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j = mulRow j * V1 +
        (pairMul hrow (actionDigits T0 T1 j) +
          (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) + halves Elut j)) * VK := by
  intro cIn Ctop Cbot A Mmul Emul mulRow Elut hrow top bot σc
  obtain ⟨w, x6, x7, x8, x10, x12, x14, x15, x18, -, -, x19, -, -, x21, x22, -, -, x24, -, -, x26, x27, -,
    x28, x33, x34, x35, x36, x38, x39, x40, x42, x43, x44, x45, x47, x48, x49, x51, x52, -, x53, -, -, -,
    -, -, -, x55, x58, -, -, x61, x63, -, -, x67, x71, x73, -, -, -, -, -, -, x75, x86, x89, -, x90, -, x91,
    xout⟩ := h
  -- The state, expanded through the state gadget, and the attribute package.
  have hw10 := expand_read2 x6 x7 x8 x10
  have hw14 := expand_read2 x6 x7 x12 x14
  obtain ⟨hm, hw19⟩ := familyGet_one x19
  obtain ⟨_, hw21⟩ := familyGet_one x21
  obtain ⟨_, hw24⟩ := familyGet_one x24
  obtain ⟨_, hw26⟩ := familyGet_one x26
  rw [hw10, hw14] at x15
  have hw15 : w.w_15_0 = joinRow cIn := concat_joinRow (r := cIn) x15
  have hw22 : w.w_22_0 = joinRow Ctop := by
    rw [hw19, hw21] at x22
    exact concat_joinRow (r := Ctop) x22
  have hw27 : w.w_27_0 = joinRow Cbot := by
    rw [hw24, hw26] at x27
    exact concat_joinRow (r := Cbot) x27
  rw [hw15, hw22, hw27] at x28
  have hmul := colAction (row := cIn) (concatColumns_joinRow cIn) (tk := Ctop) (bk := Cbot)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (colLoopMul x28)
  -- The product's companion row, against the expanded input key.
  have hkeyC : ∀ k j, inputKeyC model key k j = ![![w.w_33_0, w.w_39_0], ![w.w_44_0, w.w_48_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j <;>
      simp [inputKeyC, dynamicPrefix, schemaBytes, componentBytes, hashSample_eq x33,
        hashSample_eq x39, hashSample_eq x44, hashSample_eq x48]
  have hkey : ∀ k j, A k j = ![![w.w_38_0, w.w_42_0], ![w.w_47_0, w.w_51_0]] k j := by
    have h38 := expand_read2 x34 x35 x36 x38
    have h42 := expand_read2 x34 x35 x40 x42
    have h47 := expand_read2 x34 x35 x45 x47
    have h51 := expand_read2 x34 x35 x49 x51
    intro k j
    fin_cases k <;> fin_cases j <;> simp [A, inputKey, expandKey, hkeyC, h38, h42, h47, h51]
  have hA0 : concatColumns (A 0 0) (A 0 1) w.w_43_0 := by rw [hkey, hkey]; exact x43
  have hA1 : concatColumns (A 1 0) (A 1 1) w.w_52_0 := by rw [hkey, hkey]; exact x52
  have hw53 := mulLoop hA0 hA1 hm x53
  have hw54 : ∀ j : Fin 2, halves (matrixAdd w.w_28_0 w.w_53_0) j = mulRow j := by
    intro j
    have h28 := hmul.1 j
    funext a l
    rw [Subsingleton.elim a 0]
    have h26l := congrFun (congrFun h28 0) l
    simp only [halves, rowWindow, matrixAdd, Matrix.add_apply, mulRow] at h26l ⊢
    rw [h26l, hw53]
  have hw55 : w.w_55_0 = mulRow 0 :=
    (sliceMatrix_window 0 x55 (by simp) (by omega)).trans (hw54 0)
  have hw75 : w.w_75_0 = mulRow 1 :=
    (sliceMatrix_window 12 x75 (by simp) (by omega)).trans (hw54 1)
  -- The table entries.
  obtain ⟨_, hw58⟩ := familyGet_eq (k := 0) x58
  obtain ⟨_, hw63⟩ := familyGet_eq (k := 1) x63
  -- The identity branch.
  have hw61 := routing_one x61
  have hw67 := routing_one x67
  -- The companion row of the automorphism.
  have hw90 := lutLoop (A := Mmul) x86 x89 x90
  -- The branch of `K`.
  have hcount : Int.toNat 1 = 0 + 1 := rfl
  rw [hcount] at x91
  obtain ⟨cur, h0, hbranch⟩ := MxxIR.IterRuns.step_of_succ x91
  cases h0
  rw [hw55, hw75] at hbranch
  obtain ⟨g, VK, hg, hVK, hT, hC, hout⟩ := branch_rows hbranch
  have hg' := gadgetMatrixRuns_unique hg x71
  rw [hg', hw58, hw63] at hVK
  have hE : w.w_90_0 = Elut := by
    funext a jj
    rw [Subsingleton.elim a 0, hw90]
  refine ⟨w.w_71_0, w.w_73_0, VK, x71, ?_, hVK, hmul.2, hT, ?_, ?_, ?_, ?_, ?_, fun j ↦ ?_⟩
  · convert x73 using 2
    simp only [lutScalarOne, lutScalar, hw58, hw63, hw61, hw67, id]
    rfl
  · exact hC
  all_goals subst xout
  all_goals try rfl
  have hj := hout j
  rw [hE] at hj
  -- The forms differ only in `σc`; stating it as a vector keeps the kernel from unfolding the
  -- expansions.
  have hσ : σc = ![substituteMatrix K hK (mulRow 0), substituteMatrix K hK (mulRow 1)] := by
    funext j
    fin_cases j <;> rfl
  fin_cases j
  · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, matrixMul_eq_mul] at hj ⊢
    rw [hσ]
    exact hj
  · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_fin_one, matrixMul_eq_mul] at hj ⊢
    rw [hσ]
    exact hj

end PeProof
