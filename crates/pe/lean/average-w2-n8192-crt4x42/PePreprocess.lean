import PeOnline

/-!
# Preprocessing

Key generation computes the public part of the predicate's evaluation once: the companion keys
`M` of the Dynamic product, the switched keys `T` of the branch of `K`, and the output key
`A_F = M V_1 + (H Dec(T)) V_K`. Every key is a sum of hashed companion keys against the digits of
the key it acts on.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- `companion-key` -/
def companionKeyBytes : Blob := [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121]
/-- `column` -/
def columnBytes : Blob := [99, 111, 108, 117, 109, 110]
/-- `digit-row` -/
def digitRowBytes : Blob := [100, 105, 103, 105, 116, 45, 114, 111, 119]

/-- The companion key of the product at column `jj`, digit row `a`, and component `c`. -/
noncomputable def rhsKey (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix Q N 1 8 :=
  model.sample Q N 1 8 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes companionKeyBytes, .bytes [114, 104, 115], .u64Le 0,
      .bytes [109, 101, 109, 98, 101, 114], .u64Le 0, .bytes columnBytes, .u64Le (jj : Int),
      .bytes digitRowBytes, .u64Le (a : Int), .bytes componentBytes, .u64Le (c : Int)])

/-- The companion key of the automorphism `σ_K` at column `jj`, digit row `a`, and component
`c`. -/
noncomputable def autKey (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix Q N 1 8 :=
  model.sample Q N 1 8 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes companionKeyBytes,
      .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (K : Int),
      .bytes columnBytes, .u64Le (jj : Int), .bytes digitRowBytes, .u64Le (a : Int),
      .bytes componentBytes, .u64Le (c : Int)])

abbrev KeyBlock := ExactMatrix Q N 1 16 × ExactMatrix Q N 1 16 × Unit

/-- The digit-row sum of the companion keys `B` of column `jj` against the digits of key column
`a`, for output row `k`. -/
noncomputable def keySum (B : Nat → Nat → Nat → ExactMatrix Q N 1 8)
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8) (k : Fin 2) (jj : Fin 16) (a : Fin 16) : ExactPoly Q N :=
  (B jj a (2 * k) * keyDigits A a 0 + B jj a (2 * k + 1) * keyDigits A a 1) 0 0

theorem fmod20 {a : Nat} (ha : a < 16) : Int.fmod (Int.ofNat a) 16 = ((a : Nat) : Int) := by
  rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
  simp only [Int.ofNat_eq_natCast]
  omega

theorem fdiv20 {a : Nat} (ha : a < 16) : Int.fdiv (Int.ofNat a) 16 = 0 := by
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
  simp only [Int.ofNat_eq_natCast]
  omega

set_option maxHeartbeats 8000000 in
theorem rhsKeyInner_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 8 1} {a j : Nat} (ha : a < 16) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_sequential_generatedRoot_8_11 Backend.backend model tape path
      GeneratedClaim.stage_5_params a j (s.1, s.2.1, key, u0, u1, ()) t) :
    t.1 = s.1 + (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0 •
        unitRow ((j : Nat) : Int) := by
  obtain ⟨w2, w5, w10, w14, w16, -, x2, -, x5, -, -, -, -, -, -, -, -, x10, -, x14, -, x16, xout⟩ := h
  rw [fmod20 ha] at x2 x5 x14 x16
  have hw10 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x10
  subst xout hw10
  have e2 : w2 = rhsKey model key j a 0 := hashSample_eq x2
  have e5 : w5 = rhsKey model key j a 1 := hashSample_eq x5
  have e14 : w14 = rhsKey model key j a 2 := hashSample_eq x14
  have e16 : w16 = rhsKey model key j a 3 := hashSample_eq x16
  subst e2 e5 e14 e16
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 8000000 in
theorem rhsKeyInner {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 8 1} {a : Nat} (ha : a < 16) {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_8_11 Backend.backend model tape
        (path ++ [11, i]) GeneratedClaim.stage_5_params a i
        (current.1, (current.2.1, (key, (u0, (u1, ()))))) next) (Int.toNat 16) (0, (0, ())) final)
    (jj : Fin 16) :
    final.1 0 jj = (rhsKey model key jj a 0 * u0 + rhsKey model key jj a 1 * u1) 0 0 ∧
      final.2.1 0 jj = (rhsKey model key jj a 2 * u0 + rhsKey model key jj a 3 * u1) 0 0 := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w (j : Fin 16) ↦ (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(rhsKeyInner_step ha hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 16) ↦ (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(rhsKeyInner_step ha hb).2]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0)
    jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0)
    jj] at h1
  simpa using And.intro h0 h1

/-- The digits of key column `a` of a joined key row. -/
theorem key_entry_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {r : Fin 2} {a : Fin 16}
    {Ar : ExactMatrix Q N 1 16} {x : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 8 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar)
    (hx : sliceMatrix Ar 0 1 ((a : Nat) : Int) (1 + ((a : Nat) : Int)) x)
    (hu : gadgetDecomposeRuns Backend.backend 2097152 8 x u) : u = keyDigits A a r := by
  obtain ⟨_, hx0⟩ := sliceMatrix_column hx
  rw [concatColumns_entry hAr, pair_eta] at hx0
  have hxe : x = fun _ _ ↦ joinedEntry (A r) a := by
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0, hx0]
    congr 1
  rw [hxe] at hu
  exact entryDigits_eq hu

set_option maxHeartbeats 16000000 in
theorem rhsKeyLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {i : Nat} (hi : i < 16) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_generatedRoot_8 Backend.backend model tape path
      GeneratedClaim.stage_5_params i (s.1, s.2.1, key, A0, A1, ()) t) (jj : Fin 16) :
    t.1 0 jj = s.1 0 jj + keySum (rhsKey model key) A 0 jj ⟨i, hi⟩ ∧
      t.2.1 0 jj = s.2.1 0 jj + keySum (rhsKey model key) A 1 jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, -, -, -, -, -, x6, x7, -, -, -, -, -, -, -,
    -, x9, x10, -, hin, x12, x15, xout⟩ := h
  rw [fdiv20 hi] at x2
  rw [fmod20 hi] at x6 x9
  have hw2 := unitRow_window (c := 0) (by omega) x2
  have hw7 := key_entry_digits (a := ⟨i, hi⟩) hA0 x6 x7
  have hw10 := key_entry_digits (a := ⟨i, hi⟩) hA1 x9 x10
  rw [hw7, hw10] at hin
  have hblock := rhsKeyInner (model := model) (tape := tape) (path := path) hi hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 16}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x12, ht x15]
  exact ⟨congrArg _ hblock.1, congrArg _ hblock.2⟩

set_option maxHeartbeats 8000000 in
/-- The companion keys of the product, column by column. -/
theorem rhsKeyLoop {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_generatedRoot_8 Backend.backend model tape (path ++ [8, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (key, (A0, (A1, ()))))) next)
      (Int.toNat 16) (0, (0, ())) final) (jj : Fin 16) :
    final.1 0 jj = ∑ a, keySum (rhsKey model key) A 0 jj a ∧
      final.2.1 0 jj = ∑ a, keySum (rhsKey model key) A 1 jj a := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w a ↦ keySum (rhsKey model key) A 0 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsKeyLoop_step hA0 hA1 hi hb w).1, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w a ↦ keySum (rhsKey model key) A 1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsKeyLoop_step hA0 hA1 hi hb w).2, trivial⟩) h jj
  simpa using And.intro h0 h1

theorem selectK0 : MxxIR.selectInt 0 [8193] = (K : Int) := by decide

set_option maxHeartbeats 8000000 in
theorem autKeyInner_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 8 1} {a j : Nat} (ha : a < 16) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_sequential_generatedRoot_16_13 Backend.backend model tape path
      GeneratedClaim.stage_5_params a j (s.1, s.2.1, key, u0, u1, ()) t) :
    t.1 = s.1 + (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0 •
        unitRow ((j : Nat) : Int) := by
  obtain ⟨w2, w5, w10, w14, w16, -, -, -, -, x2, -, -, -, -, x5, -, -, -, -, -, -, -, -, x10, -, -,
    -, -, x14, -, -, -, -, x16, xout⟩ := h
  rw [fmod20 ha, fdiv20 ha] at x2 x5 x14 x16
  have hw10 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x10
  subst xout hw10
  have e2 : w2 = autKey model key j a 0 := by rw [hashSample_eq x2, selectK0]; rfl
  have e5 : w5 = autKey model key j a 1 := by rw [hashSample_eq x5, selectK0]; rfl
  have e14 : w14 = autKey model key j a 2 := by rw [hashSample_eq x14, selectK0]; rfl
  have e16 : w16 = autKey model key j a 3 := by rw [hashSample_eq x16, selectK0]; rfl
  subst e2 e5 e14 e16
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 8000000 in
theorem autKeyInner {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 8 1} {a : Nat} (ha : a < 16) {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_16_13 Backend.backend model tape
        (path ++ [13, i]) GeneratedClaim.stage_5_params a i
        (current.1, (current.2.1, (key, (u0, (u1, ()))))) next) (Int.toNat 16) (0, (0, ())) final)
    (jj : Fin 16) :
    final.1 0 jj = (autKey model key jj a 0 * u0 + autKey model key jj a 1 * u1) 0 0 ∧
      final.2.1 0 jj = (autKey model key jj a 2 * u0 + autKey model key jj a 3 * u1) 0 0 := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w (j : Fin 16) ↦ (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(autKeyInner_step ha hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 16) ↦ (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(autKeyInner_step ha hb).2]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0)
    jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0)
    jj] at h1
  simpa using And.intro h0 h1

set_option maxHeartbeats 16000000 in
theorem autKeyLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {i : Nat} (hi : i < 16) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_generatedRoot_16 Backend.backend model tape path
      GeneratedClaim.stage_5_params i (s.1, s.2.1, key, A0, A1, ()) t) (jj : Fin 16) :
    t.1 0 jj = s.1 0 jj + keySum (autKey model key) (routedKey A) 0 jj ⟨i, hi⟩ ∧
      t.2.1 0 jj = s.2.1 0 jj + keySum (autKey model key) (routedKey A) 1 jj ⟨i, hi⟩ := by
  obtain ⟨w, -, -, -, -, -, -, -, -, -, -, x2, -, -, -, -, -, -, -, -, x6, -, -, -, x7, x8, -, -, -,
    -, -, -, -, -, x10, -, -, -, x11, x12, -, hin, x14, x17, xout⟩ := h
  rw [fdiv20 hi] at x2 x7 x11
  rw [fmod20 hi] at x6 x10
  have hw2 := unitRow_window (c := 0) (by omega) x2
  have hw8 := routed_entry_digits (a := ⟨i, hi⟩) hA0 x6 x7 x8
  have hw12 := routed_entry_digits (a := ⟨i, hi⟩) hA1 x10 x11 x12
  rw [hw8, hw12] at hin
  have hblock := autKeyInner (model := model) (tape := tape) (path := path) hi hin jj
  have hone : w.w_2_0 0 0 = 1 := by rw [hw2]; simp [unitRow]
  have ht : ∀ {x y : ExactMatrix Q N 1 16}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
    intro x y hxy
    calc y 0 jj = w.w_2_0 0 0 * x 0 jj := tensorRuns_scalar hxy jj.val jj.isLt
      _ = x 0 jj := by rw [hone, one_mul]
  subst xout
  simp only [matrixAdd, Matrix.add_apply]
  rw [ht x14, ht x17]
  exact ⟨congrArg _ hblock.1, congrArg _ hblock.2⟩

set_option maxHeartbeats 8000000 in
/-- The switched keys `T` of the branch of `K`, column by column. -/
theorem autKeyLoop {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 8} {A0 A1 : ExactMatrix Q N 1 16}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_generatedRoot_16 Backend.backend model tape (path ++ [16, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (key, (A0, (A1, ()))))) next)
      (Int.toNat 16) (0, (0, ())) final) (jj : Fin 16) :
    final.1 0 jj = ∑ a, keySum (autKey model key) (routedKey A) 0 jj a ∧
      final.2.1 0 jj = ∑ a, keySum (autKey model key) (routedKey A) 1 jj a := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w a ↦ keySum (autKey model key) (routedKey A) 0 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (autKeyLoop_step hA0 hA1 hi hb w).1, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w a ↦ keySum (autKey model key) (routedKey A) 1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (autKeyLoop_step hA0 hA1 hi hb w).2, trivial⟩) h jj
  simpa using And.intro h0 h1

set_option maxHeartbeats 8000000 in
/-- The column loop of the helper key action: row `k` of the result is the joined helper key row
`H_k` times the digits of the columns of `T`, and every column decomposition runs. -/
theorem helperKeyLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {H0 H1 T0 T1 : ExactMatrix Q N 1 16} {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_36_19 Backend.backend model tape
        (path ++ [19, i]) GeneratedClaim.stage_5_params 0 i
        (current.1, (current.2.1, (H0, (T0, (T1, (H1, ())))))) next) (Int.toNat 16) (0, (0, ()))
      final) (jj : Fin 16) :
    final.1 0 jj = (H0 * colDigits T0 T1 jj) 0 0 ∧ final.2.1 0 jj = (H1 * colDigits T0 T1 jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (T0 0 jj) (T1 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits T0 T1 jj) := by
  have hcount : Int.toNat 16 = 16 := rfl
  rw [hcount] at h
  have hstep : ∀ i (s t : KeyBlock) (hi : i < 16), True →
      Stage_preprocess.sequential_sequential_generatedRoot_36_19 Backend.backend model tape
        (path ++ [19, i]) GeneratedClaim.stage_5_params 0 i
        (s.1, (s.2.1, (H0, (T0, (T1, (H1, ())))))) t →
      t.1 = s.1 + (H0 * colDigits T0 T1 ⟨i, hi⟩) 0 0 • unitRow ((i : Nat) : Int) ∧
      t.2.1 = s.2.1 + (H1 * colDigits T0 T1 ⟨i, hi⟩) 0 0 • unitRow ((i : Nat) : Int) ∧
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (T0 0 ⟨i, hi⟩) (T1 0 ⟨i, hi⟩)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits T0 T1 ⟨i, hi⟩) := by
    intro i s t hi _ hb
    obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
      -, -, -, -, -, -, x11, xout⟩ := hb
    have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
    obtain ⟨_, hw3⟩ := sliceMatrix_column x3
    obtain ⟨_, hw5⟩ := sliceMatrix_column x5
    simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
    rw [concatRows_pairColumn x6, hw3, hw5] at x7
    have hw8 := columnDigits_eq x7 x8
    subst xout hw8 hw11
    refine ⟨?_, ?_, _, x7, x8⟩
    · simp only [matrixAdd, matrixMulScalarLeft_eq, matrixMul]
    · simp only [matrixAdd, matrixMulScalarLeft_eq, matrixMul]
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w (j : Fin 16) ↦ (H0 * colDigits T0 T1 j) 0 0 * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(hstep i s t hi trivial hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 16) ↦ (H1 * colDigits T0 T1 j) 0 0 * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 16) 0 w)
    (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(hstep i s t hi trivial hb).2.1]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (H0 * colDigits T0 T1 j) 0 0) jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 16 ↦ (H1 * colDigits T0 T1 j) 0 0) jj] at h1
  have hrun := MxxIR.IterRuns.invariant
    (Invariant := fun k (_ : KeyBlock) ↦ ∀ jj : Fin 16, jj.val < k →
      ∃ D' : ExactMatrix Q N (8 + 8) 1,
        gadgetDecomposeRuns Backend.backend 2097152 8 (pairColumn (T0 0 jj) (T1 0 jj)) D' ∧
          compactCenteredRebaseRuns 1048576 D' (colDigits T0 T1 jj))
    (fun jj hjj ↦ absurd hjj (Nat.not_lt_zero _))
    (fun i s t hq hb jj hjj ↦ by
      by_cases hi : i < 16
      · rcases Nat.lt_succ_iff_lt_or_eq.mp hjj with hlt | heq
        · exact hq jj hlt
        · obtain ⟨-, -, hD⟩ := hstep i s t hi trivial hb
          have hjj' : jj = ⟨i, hi⟩ := Fin.ext heq
          subst hjj'
          exact hD
      · exact hq jj (by omega)) h
  simpa using And.intro h0 (And.intro h1 (hrun jj jj.isLt))

/-- The output keys of the predicate: the identity branch `M V_1` plus the branch of `K`,
`(H Dec(T)) V_K`. -/
noncomputable def outputKey (model : HashModel) (key : ByteArray) (Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 8)
    (T0 T1 : ExactMatrix Q N 1 16) (V1 VK : ExactMatrix Q N 8 8) : Fin 2 → Fin 2 → ExactMatrix Q N 1 8 :=
  fun k j ↦ Mmul k j * V1 + pairMul (helperKey model key k) (actionDigits T0 T1 j) * VK

abbrev PortBlock := ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 ×
  ExactMatrix Q N 1 8 × Unit

set_option maxHeartbeats 64000000 in
/-- The loop over the one branch of `K` adds `(H Dec(T)) V_K` to the identity keys. -/
theorem branchKeys {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {T0 T1 : ExactMatrix Q N 1 16} {init final : PortBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : PortBlock) ↦
      Stage_preprocess.sequential_generatedRoot_36 Backend.backend model tape (path ++ [36, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1,
          (key, (T0, (T1, ()))))))) next) (Int.toNat 1) init final) :
    ∃ (g : ExactMatrix Q N 1 8) (VK : ExactMatrix Q N 8 8),
      gadgetMatrixRuns Backend.backend 2097152 8 g ∧
      gadgetDecomposeRuns Backend.backend 2097152 8
        (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g) VK ∧
      ∀ k j : Fin 2, ![![final.1, final.2.1], ![final.2.2.1, final.2.2.2.1]] k j =
        ![![init.1, init.2.1], ![init.2.2.1, init.2.2.2.1]] k j +
          pairMul (helperKey model key k) (actionDigits T0 T1 j) * VK := by
  have hcount : Int.toNat 1 = 0 + 1 := rfl
  rw [hcount] at h
  obtain ⟨cur, h0, hb⟩ := MxxIR.IterRuns.step_of_succ h
  cases h0
  obtain ⟨w, -, -, x3, -, -, x4, x5, -, -, -, -, -, -, x7, -, -, -, -, -, -, x8, -, -, -, -, -, -, x9,
    x10, -, -, -, -, -, -, x12, -, -, -, -, -, -, x13, -, -, -, -, -, -, x14, x15, -, -, x16, -, -,
    x17, x18, -, x19, -, -, -, -, -, -, x20, -, -, -, -, x24, -, -, -, -, x29, x33, x35, -, -, -, -,
    -, -, x39, -, -, -, -, -, -, x43, -, -, -, -, -, -, x47, xout⟩ := hb
  have hz : (16 * Int.ofNat 0 : Int) = 0 := by decide
  rw [hz] at x7 x12
  have hw7 := (sliceMatrix_window 0 x7 (by simp) (by omega)).trans (rowWindow_full T0)
  have hw12 := (sliceMatrix_window 0 x12 (by simp) (by omega)).trans (rowWindow_full T1)
  rw [hw7] at x8 x9
  rw [hw12] at x13 x14
  have hw8 := sliceMatrix_window 0 x8 (by simp) (by omega)
  have hw9 := sliceMatrix_window 8 x9 (by simp) (by omega)
  have hw13 := sliceMatrix_window 0 x13 (by simp) (by omega)
  have hw14 := sliceMatrix_window 8 x14 (by simp) (by omega)
  rw [hw8, hw9] at x10
  rw [hw13, hw14] at x15
  have hw10 : w.w_10_0 = T0 := (concat_joinRow (r := halves T0) x10).trans (joinRow_halves T0)
  have hw15 : w.w_15_0 = T1 := (concat_joinRow (r := halves T1) x15).trans (joinRow_halves T1)
  have hkey : ∀ k j, helperKey model key k j = ![![w.w_3_0, w.w_4_0], ![w.w_16_0, w.w_17_0]] k j := by
    have hs : MxxIR.selectInt (Int.ofNat 0) [8193] = (K : Int) := by decide
    rw [hs] at x3 x4 x16 x17
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x4).symm
    · exact (hashSample_eq x16).symm
    · exact (hashSample_eq x17).symm
  have hH0 : concatColumns (helperKey model key 0 0) (helperKey model key 0 1) w.w_5_0 := by
    rw [hkey, hkey]; exact x5
  have hH1 : concatColumns (helperKey model key 1 0) (helperKey model key 1 1) w.w_18_0 := by
    rw [hkey, hkey]; exact x18
  rw [hw10, hw15] at x19
  have hloop := helperKeyLoop x19
  have hact : ∀ (r : Fin 2) {R acc : ExactMatrix Q N 1 16},
      concatColumns (helperKey model key r 0) (helperKey model key r 1) R →
      (∀ p, acc 0 p = (R * colDigits T0 T1 p) 0 0) →
      ∀ j : Fin 2, rowWindow 8 (8 * j) (by omega) acc =
        pairMul (helperKey model key r) (actionDigits T0 T1 j) := by
    intro r R acc hR hacc j
    rw [window_mul_columns (m := 8) (k := 8) hR hacc j (by omega)]
    rfl
  have h0 := hact 0 hH0 (fun p ↦ (hloop p).1)
  have h1 := hact 1 hH1 (fun p ↦ (hloop p).2.1)
  have hw20 := (sliceMatrix_window 0 x20 (by simp) (by omega)).trans (h0 0)
  have hw39 := (sliceMatrix_window 8 x39 (by simp) (by omega)).trans (h0 1)
  have hw43 := (sliceMatrix_window 0 x43 (by simp) (by omega)).trans (h1 0)
  have hw47 := (sliceMatrix_window 8 x47 (by simp) (by omega)).trans (h1 1)
  have hw24 := routing_K x24
  have hw29 := routing_K x29
  refine ⟨w.w_33_0, w.w_35_0, x33, ?_, fun k j ↦ ?_⟩
  · convert x35 using 2
    simp only [lutScalar, hw24, hw29]
  · subst xout
    fin_cases k <;> fin_cases j
    · simp [matrixAdd, matrixMul, hw20]
    · simp [matrixAdd, matrixMul, hw39]
    · simp [matrixAdd, matrixMul, hw43]
    · simp [matrixAdd, matrixMul, hw47]

abbrev PreprocessOutputs := ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 ×
  ExactMatrix Q N 1 8 × ExactMatrix Q N 1 16 × ExactMatrix Q N 1 16 × ExactMatrix Q N 1 8 ×
  ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 ×
  ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × ExactMatrix Q N 1 8 × Unit

set_option maxHeartbeats 64000000 in
/-- Preprocessing outputs the companion keys `M` of the product, the switched keys `T` of the
branch of `K`, and the output key `A_F = M V_1 + (H Dec(T)) V_K` (twice: for key generation and
for online evaluation). -/
theorem preprocess_keys {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {outputs : PreprocessOutputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_5_params
      key outputs) :
    let Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 8 :=
      ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
    let T0 := outputs.2.2.2.2.1
    let T1 := outputs.2.2.2.2.2.1
    (∀ (k j : Fin 2) (l : Fin 8), Mmul k j 0 l =
      ∑ a, keySum (rhsKey model key) (inputKey model key) k ⟨8 * (j : Nat) + l, by omega⟩ a) ∧
    (∀ (k j : Fin 2) (l : Fin 8), halves (![T0, T1] k) j 0 l =
      ∑ a, keySum (autKey model key) (routedKey Mmul) k ⟨8 * (j : Nat) + l, by omega⟩ a) ∧
    ∃ (g : ExactMatrix Q N 1 8) (V1 VK : ExactMatrix Q N 8 8),
      gadgetMatrixRuns Backend.backend 2097152 8 g ∧
      gadgetDecomposeRuns Backend.backend 2097152 8 (matrixMulScalarLeft lutScalarOne g) V1 ∧
      gadgetDecomposeRuns Backend.backend 2097152 8
        (matrixMulScalarLeft (lutScalar (substituteMatrix K hK)) g) VK ∧
      (∀ k j : Fin 2, ![![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1],
          ![outputs.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.1]] k j =
        outputKey model key Mmul T0 T1 V1 VK k j) ∧
      (∀ k j : Fin 2, ![![outputs.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.1],
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1]] k j =
        outputKey model key Mmul T0 T1 V1 VK k j) := by
  intro Mmul T0 T1
  obtain ⟨w, x2, x3, x4, x5, x6, x7, -, x8, -, -, -, -, -, -, x9, -, -, -, -, -, -, x10, -, -, -, -,
    -, -, x11, -, -, -, -, -, -, x12, x14, x15, -, x16, -, -, x20, -, -, x25, x29, x31, -, x36,
    xout⟩ := h
  subst xout
  -- The companion keys of the product.
  have hkey : ∀ k j, inputKey model key k j = ![![w.w_2_0, w.w_3_0], ![w.w_5_0, w.w_6_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x2).symm
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x5).symm
    · exact (hashSample_eq x6).symm
  have hA0 : concatColumns (inputKey model key 0 0) (inputKey model key 0 1) w.w_4_0 := by
    rw [hkey, hkey]; exact x4
  have hA1 : concatColumns (inputKey model key 1 0) (inputKey model key 1 1) w.w_7_0 := by
    rw [hkey, hkey]; exact x7
  have hM := rhsKeyLoop hA0 hA1 x8
  have hw9 := sliceMatrix_window 0 x9 (by simp) (by omega)
  have hw10 := sliceMatrix_window 8 x10 (by simp) (by omega)
  have hw11 := sliceMatrix_window 0 x11 (by simp) (by omega)
  have hw12 := sliceMatrix_window 8 x12 (by simp) (by omega)
  have hMrow : ∀ (k j : Fin 2) (l : Fin 8), Mmul k j 0 l =
      ∑ a, keySum (rhsKey model key) (inputKey model key) k ⟨8 * (j : Nat) + l, by omega⟩ a := by
    intro k j l
    fin_cases k <;> fin_cases j
    · simp only [Mmul, hw9]
      exact (hM _).1
    · simp only [Mmul, hw10]
      exact (hM _).1
    · simp only [Mmul, hw11]
      exact (hM _).2
    · simp only [Mmul, hw12]
      exact (hM _).2
  -- The switched keys.
  have hM0 : concatColumns (Mmul 0 0) (Mmul 0 1) w.w_14_0 := x14
  have hM1 : concatColumns (Mmul 1 0) (Mmul 1 1) w.w_15_0 := x15
  have hT := autKeyLoop hM0 hM1 x16
  refine ⟨hMrow, fun k j l ↦ ?_, ?_⟩
  · fin_cases k
    · exact (hT _).1
    · exact (hT _).2
  -- The output keys.
  have hw20 := routing_one x20
  have hw25 := routing_one x25
  obtain ⟨g, VK, hg, hVK, hports⟩ := branchKeys x36
  have hgg := gadgetMatrixRuns_unique x29 hg
  subst hgg
  have hV1 : gadgetDecomposeRuns Backend.backend 2097152 8 (matrixMulScalarLeft lutScalarOne w.w_29_0)
      w.w_31_0 := by
    convert x31 using 2
    simp only [lutScalarOne, lutScalar, hw20, hw25, id]
  have hout : ∀ k j : Fin 2, ![![w.w_36_0, w.w_36_1], ![w.w_36_2, w.w_36_3]] k j =
      outputKey model key Mmul T0 T1 w.w_31_0 VK k j := by
    intro k j
    rw [hports k j]
    fin_cases k <;> fin_cases j <;> rfl
  exact ⟨w.w_29_0, w.w_31_0, VK, x29, hV1, hVK, hout, hout⟩

end PeProof
