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

/-- The compact companion key of the product at column `jj`, digit row `a`, and component `c`. -/
noncomputable def rhsKeyC (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix PQ N 1 1 :=
  model.sample PQ N 1 1 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes companionKeyBytes, .bytes [114, 104, 115], .u64Le 0,
      .bytes [109, 101, 109, 98, 101, 114], .u64Le 0, .bytes columnBytes, .u64Le (jj : Int),
      .bytes digitRowBytes, .u64Le (a : Int), .bytes componentBytes, .u64Le (c : Int)])

/-- The companion key of the product: the expansion of its compact key. -/
noncomputable def rhsKey (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix Q N 1 12 :=
  expand (rhsKeyC model key jj a c)

/-- The compact companion key of the automorphism `σ_K` at column `jj`, digit row `a`, and
component `c`. -/
noncomputable def autKeyC (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix PQ N 1 1 :=
  model.sample PQ N 1 1 key (completeHashTag dynamicPrefix
    [.bytes schemaBytes, .bytes companionKeyBytes,
      .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (K : Int),
      .bytes columnBytes, .u64Le (jj : Int), .bytes digitRowBytes, .u64Le (a : Int),
      .bytes componentBytes, .u64Le (c : Int)])

/-- The companion key of the automorphism: the expansion of its compact key. -/
noncomputable def autKey (model : HashModel) (key : ByteArray) (jj a c : Nat) : ExactMatrix Q N 1 12 :=
  expand (autKeyC model key jj a c)

abbrev KeyBlock := ExactMatrix Q N 1 24 × ExactMatrix Q N 1 24 × Unit

/-- The digit-row sum of the companion keys `B` of column `jj` against the digits of key column
`a`, for output row `k`. -/
noncomputable def keySum (B : Nat → Nat → Nat → ExactMatrix Q N 1 12)
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12) (k : Fin 2) (jj : Fin 24) (a : Fin 24) : ExactPoly Q N :=
  (B jj a (2 * k) * keyDigits A a 0 + B jj a (2 * k + 1) * keyDigits A a 1) 0 0

theorem fmod20 {a : Nat} (ha : a < 24) : Int.fmod (Int.ofNat a) 24 = ((a : Nat) : Int) := by
  rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
  simp only [Int.ofNat_eq_natCast]
  omega

theorem fdiv20 {a : Nat} (ha : a < 24) : Int.fdiv (Int.ofNat a) 24 = 0 := by
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
  simp only [Int.ofNat_eq_natCast]
  omega

set_option maxHeartbeats 8000000 in
theorem rhsKeyInner_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 12 1} {a j : Nat} (ha : a < 24) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_sequential_generatedRoot_24_11 Backend.backend model tape path
      GeneratedClaim.stage_5_params a j (s.1, s.2.1, key, u0, u1, ()) t) :
    t.1 = s.1 + (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0 •
        unitRow ((j : Nat) : Int) := by
  obtain ⟨w2, w3, w4, w6, w9, w10, w12, w17, w21, w22, w24, w26, w27, w29, -, x2, x3, x4, x6, -, x9,
    x10, x12, -, -, -, -, -, -, -, -, x17, -, x21, x22, x24, -, x26, x27, x29, xout⟩ := h
  rw [fmod20 ha] at x2 x9 x21 x26
  have hw17 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x17
  subst xout hw17
  have e6 : w6 = rhsKey model key j a 0 :=
    (expand_read x3 x4 x6).trans (congrArg expand (hashSample_eq x2))
  have e12 : w12 = rhsKey model key j a 1 :=
    (expand_read x3 x10 x12).trans (congrArg expand (hashSample_eq x9))
  have e24 : w24 = rhsKey model key j a 2 :=
    (expand_read x3 x22 x24).trans (congrArg expand (hashSample_eq x21))
  have e29 : w29 = rhsKey model key j a 3 :=
    (expand_read x3 x27 x29).trans (congrArg expand (hashSample_eq x26))
  subst e6 e12 e24 e29
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 8000000 in
theorem rhsKeyInner {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 12 1} {a : Nat} (ha : a < 24) {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_24_11 Backend.backend model tape
        (path ++ [11, i]) GeneratedClaim.stage_5_params a i
        (current.1, (current.2.1, (key, (u0, (u1, ()))))) next) (Int.toNat 24) (0, (0, ())) final)
    (jj : Fin 24) :
    final.1 0 jj = (rhsKey model key jj a 0 * u0 + rhsKey model key jj a 1 * u1) 0 0 ∧
      final.2.1 0 jj = (rhsKey model key jj a 2 * u0 + rhsKey model key jj a 3 * u1) 0 0 := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w (j : Fin 24) ↦ (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(rhsKeyInner_step ha hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 24) ↦ (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(rhsKeyInner_step ha hb).2]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (rhsKey model key j a 0 * u0 + rhsKey model key j a 1 * u1) 0 0)
    jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (rhsKey model key j a 2 * u0 + rhsKey model key j a 3 * u1) 0 0)
    jj] at h1
  simpa using And.intro h0 h1

/-- The digits of key column `a` of a joined key row. -/
theorem key_entry_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {r : Fin 2} {a : Fin 24}
    {Ar : ExactMatrix Q N 1 24} {x : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 12 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar)
    (hx : sliceMatrix Ar 0 1 ((a : Nat) : Int) (1 + ((a : Nat) : Int)) x)
    (hu : gadgetDecomposeRuns Backend.backend 32768 12 x u) : u = keyDigits A a r := by
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
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {i : Nat} (hi : i < 24) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_generatedRoot_24 Backend.backend model tape path
      GeneratedClaim.stage_5_params i (s.1, s.2.1, key, A0, A1, ()) t) (jj : Fin 24) :
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
  have ht : ∀ {x y : ExactMatrix Q N 1 24}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
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
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_generatedRoot_24 Backend.backend model tape (path ++ [24, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (key, (A0, (A1, ()))))) next)
      (Int.toNat 24) (0, (0, ())) final) (jj : Fin 24) :
    final.1 0 jj = ∑ a, keySum (rhsKey model key) A 0 jj a ∧
      final.2.1 0 jj = ∑ a, keySum (rhsKey model key) A 1 jj a := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w a ↦ keySum (rhsKey model key) A 0 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsKeyLoop_step hA0 hA1 hi hb w).1, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w a ↦ keySum (rhsKey model key) A 1 w a) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ (rhsKeyLoop_step hA0 hA1 hi hb w).2, trivial⟩) h jj
  simpa using And.intro h0 h1

theorem selectK0 : MxxIR.selectInt 0 [32769] = (K : Int) := by decide

set_option maxHeartbeats 8000000 in
theorem autKeyInner_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 12 1} {a j : Nat} (ha : a < 24) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_sequential_generatedRoot_36_13 Backend.backend model tape path
      GeneratedClaim.stage_5_params a j (s.1, s.2.1, key, u0, u1, ()) t) :
    t.1 = s.1 + (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0 •
        unitRow ((j : Nat) : Int) ∧
      t.2.1 = s.2.1 + (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0 •
        unitRow ((j : Nat) : Int) := by
  obtain ⟨w2, w3, w4, w6, w9, w10, w12, w17, w21, w22, w24, w26, w27, w29, -, -, -, -, x2, x3, x4, x6,
    -, -, -, -, x9, x10, x12, -, -, -, -, -, -, -, -, x17, -, -, -, -, x21, x22, x24, -, -, -, -, x26,
    x27, x29, xout⟩ := h
  rw [fmod20 ha, fdiv20 ha] at x2 x9 x21 x26
  have hw17 := unitRow_window (c := ((j : Nat) : Int)) (by simp only [Int.ofNat_eq_natCast]; omega) x17
  subst xout hw17
  have e6 : w6 = autKey model key j a 0 := by
    rw [expand_read x3 x4 x6, hashSample_eq x2, selectK0]; rfl
  have e12 : w12 = autKey model key j a 1 := by
    rw [expand_read x3 x10 x12, hashSample_eq x9, selectK0]; rfl
  have e24 : w24 = autKey model key j a 2 := by
    rw [expand_read x3 x22 x24, hashSample_eq x21, selectK0]; rfl
  have e29 : w29 = autKey model key j a 3 := by
    rw [expand_read x3 x27 x29, hashSample_eq x26, selectK0]; rfl
  subst e6 e12 e24 e29
  exact ⟨rfl, rfl⟩

set_option maxHeartbeats 8000000 in
theorem autKeyInner {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {u0 u1 : ExactMatrix Q N 12 1} {a : Nat} (ha : a < 24) {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_36_13 Backend.backend model tape
        (path ++ [13, i]) GeneratedClaim.stage_5_params a i
        (current.1, (current.2.1, (key, (u0, (u1, ()))))) next) (Int.toNat 24) (0, (0, ())) final)
    (jj : Fin 24) :
    final.1 0 jj = (autKey model key jj a 0 * u0 + autKey model key jj a 1 * u1) 0 0 ∧
      final.2.1 0 jj = (autKey model key jj a 2 * u0 + autKey model key jj a 3 * u1) 0 0 := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have h0 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.1)
    (fun w (j : Fin 24) ↦ (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(autKeyInner_step ha hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 24) ↦ (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0 *
      (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(autKeyInner_step ha hb).2]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (autKey model key j a 0 * u0 + autKey model key j a 1 * u1) 0 0)
    jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (autKey model key j a 2 * u0 + autKey model key j a 3 * u1) 0 0)
    jj] at h1
  simpa using And.intro h0 h1

set_option maxHeartbeats 16000000 in
theorem autKeyLoop_step {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {i : Nat} (hi : i < 24) {s t : KeyBlock}
    (h : Stage_preprocess.sequential_generatedRoot_36 Backend.backend model tape path
      GeneratedClaim.stage_5_params i (s.1, s.2.1, key, A0, A1, ()) t) (jj : Fin 24) :
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
  have ht : ∀ {x y : ExactMatrix Q N 1 24}, tensorRuns w.w_2_0 x y → y 0 jj = x 0 jj := by
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
    {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 12} {A0 A1 : ExactMatrix Q N 1 24}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_generatedRoot_36 Backend.backend model tape (path ++ [36, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (key, (A0, (A1, ()))))) next)
      (Int.toNat 24) (0, (0, ())) final) (jj : Fin 24) :
    final.1 0 jj = ∑ a, keySum (autKey model key) (routedKey A) 0 jj a ∧
      final.2.1 0 jj = ∑ a, keySum (autKey model key) (routedKey A) 1 jj a := by
  have hcount : Int.toNat 24 = 24 := rfl
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
    {H0 H1 T0 T1 : ExactMatrix Q N 1 24} {final : KeyBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : KeyBlock) ↦
      Stage_preprocess.sequential_sequential_generatedRoot_59_32 Backend.backend model tape
        (path ++ [32, i]) GeneratedClaim.stage_5_params 0 i
        (current.1, (current.2.1, (H0, (T0, (T1, (H1, ())))))) next) (Int.toNat 24) (0, (0, ()))
      final) (jj : Fin 24) :
    final.1 0 jj = (H0 * colDigits T0 T1 jj) 0 0 ∧ final.2.1 0 jj = (H1 * colDigits T0 T1 jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (T0 0 jj) (T1 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits T0 T1 jj) := by
  have hcount : Int.toNat 24 = 24 := rfl
  rw [hcount] at h
  have hstep : ∀ i (s t : KeyBlock) (hi : i < 24), True →
      Stage_preprocess.sequential_sequential_generatedRoot_59_32 Backend.backend model tape
        (path ++ [32, i]) GeneratedClaim.stage_5_params 0 i
        (s.1, (s.2.1, (H0, (T0, (T1, (H1, ())))))) t →
      t.1 = s.1 + (H0 * colDigits T0 T1 ⟨i, hi⟩) 0 0 • unitRow ((i : Nat) : Int) ∧
      t.2.1 = s.2.1 + (H1 * colDigits T0 T1 ⟨i, hi⟩) 0 0 • unitRow ((i : Nat) : Int) ∧
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (T0 0 ⟨i, hi⟩) (T1 0 ⟨i, hi⟩)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits T0 T1 ⟨i, hi⟩) := by
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
    (fun w (j : Fin 24) ↦ (H0 * colDigits T0 T1 j) 0 0 * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w)
    (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(hstep i s t hi trivial hb).1]; rfl, trivial⟩) h jj
  have h1 := IterRuns.accumulate_entries (fun s : KeyBlock ↦ s.2.1)
    (fun w (j : Fin 24) ↦ (H1 * colDigits T0 T1 j) 0 0 * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 24) 0 w)
    (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨fun w ↦ by dsimp only; rw [(hstep i s t hi trivial hb).2.1]; rfl, trivial⟩) h jj
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (H0 * colDigits T0 T1 j) 0 0) jj] at h0
  rw [sum_unitRow_mul (fun j : Fin 24 ↦ (H1 * colDigits T0 T1 j) 0 0) jj] at h1
  have hrun := MxxIR.IterRuns.invariant
    (Invariant := fun k (_ : KeyBlock) ↦ ∀ jj : Fin 24, jj.val < k →
      ∃ D' : ExactMatrix Q N (12 + 12) 1,
        gadgetDecomposeRuns Backend.backend 32768 12 (pairColumn (T0 0 jj) (T1 0 jj)) D' ∧
          compactCenteredRebaseRuns 16384 D' (colDigits T0 T1 jj))
    (fun jj hjj ↦ absurd hjj (Nat.not_lt_zero _))
    (fun i s t hq hb jj hjj ↦ by
      by_cases hi : i < 24
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
noncomputable def outputKey (model : HashModel) (key : ByteArray) (Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 12)
    (T0 T1 : ExactMatrix Q N 1 24) (V1 VK : ExactMatrix Q N 12 12) : Fin 2 → Fin 2 → ExactMatrix Q N 1 12 :=
  fun k j ↦ Mmul k j * V1 + pairMul (helperKey model key k) (actionDigits T0 T1 j) * VK

/-- A selection from a one-element list is that element. -/
theorem select_one {α : Type} {i : Int} {x out : α} (h : MxxRuntime.select i [x] out) : out = x := by
  obtain ⟨⟨p, hp⟩, -, hout⟩ := h
  simp only [List.length_singleton, Nat.lt_one_iff] at hp
  subst hp
  exact hout

abbrev PortBlock := ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 ×
  ExactMatrix Q N 1 12 × Unit

set_option maxHeartbeats 64000000 in
/-- The loop over the one branch of `K` adds `(H Dec(T)) V_K` to the identity keys. -/
theorem branchKeys {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {T0 T1 : ExactMatrix Q N 1 24} {t0 t1 : Int} {init final : PortBlock}
    (h : MxxIR.IterRuns (fun (i : Nat) (current next : PortBlock) ↦
      Stage_preprocess.sequential_generatedRoot_59 Backend.backend model tape (path ++ [59, i])
        GeneratedClaim.stage_5_params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1,
          (key, (T0, (T1, (t0, (t1, ()))))))))) next) (Int.toNat 1) init final) :
    ∃ (g : ExactMatrix Q N 1 12) (VK : ExactMatrix Q N 12 12),
      gadgetMatrixRuns Backend.backend 32768 12 g ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g) VK ∧
      ∀ k j : Fin 2, ![![final.1, final.2.1], ![final.2.2.1, final.2.2.2.1]] k j =
        ![![init.1, init.2.1], ![init.2.2.1, init.2.2.2.1]] k j +
          pairMul (helperKey model key k) (actionDigits T0 T1 j) * VK := by
  have hcount : Int.toNat 1 = 0 + 1 := rfl
  rw [hcount] at h
  obtain ⟨cur, h0, hb⟩ := MxxIR.IterRuns.step_of_succ h
  cases h0
  obtain ⟨w, -, -, x3, x4, x5, x7, -, -, x8, x9, x11, x12, -, -, -, -, -, -, x14, -, -, -, -, -, -, x15,
    -, -, -, -, -, -, x16, x17, -, -, -, -, -, -, x19, -, -, -, -, -, -, x20, -, -, -, -, -, -, x21,
    x22, -, -, x23, x24, x26, -, -, x27, x28, x30, x31, -, x32, -, -, -, -, -, -, x33, -, -, -, -, x38,
    -, -, -, -, x44, x48, x50, -, -, -, -, -, -, x54, -, -, -, -, -, -, x58, -, -, -, -, -, -, x62,
    xout⟩ := hb
  have hz : (24 * Int.ofNat 0 : Int) = 0 := by decide
  rw [hz] at x14 x19
  have hw14 := (sliceMatrix_window 0 x14 (by simp) (by omega)).trans (rowWindow_full T0)
  have hw19 := (sliceMatrix_window 0 x19 (by simp) (by omega)).trans (rowWindow_full T1)
  rw [hw14] at x15 x16
  rw [hw19] at x20 x21
  have hw15 := sliceMatrix_window 0 x15 (by simp) (by omega)
  have hw16 := sliceMatrix_window 12 x16 (by simp) (by omega)
  have hw20 := sliceMatrix_window 0 x20 (by simp) (by omega)
  have hw21 := sliceMatrix_window 12 x21 (by simp) (by omega)
  rw [hw15, hw16] at x17
  rw [hw20, hw21] at x22
  have hw17 : w.w_17_0 = T0 := (concat_joinRow (r := halves T0) x17).trans (joinRow_halves T0)
  have hw22 : w.w_22_0 = T1 := (concat_joinRow (r := halves T1) x22).trans (joinRow_halves T1)
  have hkeyC : ∀ k j, helperKeyC model key k j = ![![w.w_3_0, w.w_8_0], ![w.w_23_0, w.w_27_0]] k j := by
    have hs : MxxIR.selectInt (Int.ofNat 0) [32769] = (K : Int) := by decide
    rw [hs] at x3 x8 x23 x27
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x3).symm
    · exact (hashSample_eq x8).symm
    · exact (hashSample_eq x23).symm
    · exact (hashSample_eq x27).symm
  have hkey : ∀ k j, helperKey model key k j = ![![w.w_7_0, w.w_11_0], ![w.w_26_0, w.w_30_0]] k j := by
    have h7 := expand_read x4 x5 x7
    have h11 := expand_read x4 x9 x11
    have h26 := expand_read x4 x24 x26
    have h30 := expand_read x4 x28 x30
    intro k j
    fin_cases k <;> fin_cases j <;> simp [helperKey, expandKey, hkeyC, h7, h11, h26, h30]
  have hH0 : concatColumns (helperKey model key 0 0) (helperKey model key 0 1) w.w_12_0 := by
    rw [hkey, hkey]; exact x12
  have hH1 : concatColumns (helperKey model key 1 0) (helperKey model key 1 1) w.w_31_0 := by
    rw [hkey, hkey]; exact x31
  rw [hw17, hw22] at x32
  have hloop := helperKeyLoop x32
  have hact : ∀ (r : Fin 2) {R acc : ExactMatrix Q N 1 24},
      concatColumns (helperKey model key r 0) (helperKey model key r 1) R →
      (∀ p, acc 0 p = (R * colDigits T0 T1 p) 0 0) →
      ∀ j : Fin 2, rowWindow 12 (12 * j) (by omega) acc =
        pairMul (helperKey model key r) (actionDigits T0 T1 j) := by
    intro r R acc hR hacc j
    rw [window_mul_columns (m := 12) (k := 12) hR hacc j (by omega)]
    rfl
  have h0 := hact 0 hH0 (fun p ↦ (hloop p).1)
  have h1 := hact 1 hH1 (fun p ↦ (hloop p).2.1)
  have hw33 := (sliceMatrix_window 0 x33 (by simp) (by omega)).trans (h0 0)
  have hw54 := (sliceMatrix_window 12 x54 (by simp) (by omega)).trans (h0 1)
  have hw58 := (sliceMatrix_window 0 x58 (by simp) (by omega)).trans (h1 0)
  have hw62 := (sliceMatrix_window 12 x62 (by simp) (by omega)).trans (h1 1)
  have hw38 := routing_K x38
  have hw44 := routing_K x44
  refine ⟨w.w_48_0, w.w_50_0, x48, ?_, fun k j ↦ ?_⟩
  · convert x50 using 2
    simp only [lutScalar, hw38, hw44]
  · subst xout
    fin_cases k <;> fin_cases j
    · simp [matrixAdd, matrixMul, hw33]
    · simp [matrixAdd, matrixMul, hw54]
    · simp [matrixAdd, matrixMul, hw58]
    · simp [matrixAdd, matrixMul, hw62]

abbrev PreprocessOutputs := ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 ×
  ExactMatrix Q N 1 12 × ExactMatrix Q N 1 24 × ExactMatrix Q N 1 24 × ExactMatrix Q N 1 12 ×
  ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 ×
  ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit

set_option maxHeartbeats 64000000 in
/-- Preprocessing outputs the companion keys `M` of the product, the switched keys `T` of the
branch of `K`, and the output key `A_F = M V_1 + (H Dec(T)) V_K` (twice: for key generation and
for online evaluation). -/
theorem preprocess_keys {model : HashModel} {tape : SampleTape} {path : List Nat} {key : ByteArray}
    {reads : Fin 1 → Int} {tables : Fin 2 → Int} {outputs : PreprocessOutputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_5_params
      (reads, key, tables, ()) outputs) :
    let Mmul : Fin 2 → Fin 2 → ExactMatrix Q N 1 12 :=
      ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
    let T0 := outputs.2.2.2.2.1
    let T1 := outputs.2.2.2.2.2.1
    (∀ (k j : Fin 2) (l : Fin 12), Mmul k j 0 l =
      ∑ a, keySum (rhsKey model key) (inputKey model key) k ⟨12 * (j : Nat) + l, by omega⟩ a) ∧
    (∀ (k j : Fin 2) (l : Fin 12), halves (![T0, T1] k) j 0 l =
      ∑ a, keySum (autKey model key) (routedKey Mmul) k ⟨12 * (j : Nat) + l, by omega⟩ a) ∧
    ∃ (g : ExactMatrix Q N 1 12) (V1 VK : ExactMatrix Q N 12 12),
      gadgetMatrixRuns Backend.backend 32768 12 g ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalarOne (tables 0) (tables 1)) g) V1 ∧
      gadgetDecomposeRuns Backend.backend 32768 12
        (matrixMulScalarLeft (lutScalar (tables 0) (tables 1) (substituteMatrix K hK)) g) VK ∧
      (∀ k j : Fin 2, ![![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1],
          ![outputs.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.1]] k j =
        outputKey model key Mmul T0 T1 V1 VK k j) ∧
      (∀ k j : Fin 2, ![![outputs.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.1],
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1]] k j =
        outputKey model key Mmul T0 T1 V1 VK k j) := by
  intro Mmul T0 T1
  obtain ⟨w, -, x4, x5, x6, x7, x9, x10, x11, x13, x14, x15, x16, x18, x19, x20, x22, x23, -, x24, -, -, -,
    -, -, -, x25, -, -, -, x26, -, -, -, -, -, -, x27, -, -, -, x28, -, -, -, -, -, -, x29, -, -, -, x30, -,
    -, -, -, -, -, x31, -, -, -, x32, x34, x35, -, x36, x39, -, -, x42, x44, -, -, x48, x52, x54, -, x59,
    xout⟩ := h
  subst xout
  -- The companion keys of the product, against the expanded input key.
  have hkeyC : ∀ k j, inputKeyC model key k j = ![![w.w_4_0, w.w_10_0], ![w.w_15_0, w.w_19_0]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq x4).symm
    · exact (hashSample_eq x10).symm
    · exact (hashSample_eq x15).symm
    · exact (hashSample_eq x19).symm
  have hkey : ∀ k j, inputKey model key k j = ![![w.w_9_0, w.w_13_0], ![w.w_18_0, w.w_22_0]] k j := by
    have h9 := expand_read2 x5 x6 x7 x9
    have h13 := expand_read2 x5 x6 x11 x13
    have h18 := expand_read2 x5 x6 x16 x18
    have h22 := expand_read2 x5 x6 x20 x22
    intro k j
    fin_cases k <;> fin_cases j <;> simp [inputKey, expandKey, hkeyC, h9, h13, h18, h22]
  have hA0 : concatColumns (inputKey model key 0 0) (inputKey model key 0 1) w.w_14_0 := by
    rw [hkey, hkey]; exact x14
  have hA1 : concatColumns (inputKey model key 1 0) (inputKey model key 1 1) w.w_23_0 := by
    rw [hkey, hkey]; exact x23
  have hM := rhsKeyLoop hA0 hA1 x24
  -- The keys of the one member, selected at the member read.
  have hm00 := (select_one x26).trans (sliceMatrix_window 0 x25 (by simp) (by omega))
  have hm01 := (select_one x28).trans (sliceMatrix_window 12 x27 (by simp) (by omega))
  have hm10 := (select_one x30).trans (sliceMatrix_window 0 x29 (by simp) (by omega))
  have hm11 := (select_one x32).trans (sliceMatrix_window 12 x31 (by simp) (by omega))
  have hMrow : ∀ (k j : Fin 2) (l : Fin 12), Mmul k j 0 l =
      ∑ a, keySum (rhsKey model key) (inputKey model key) k ⟨12 * (j : Nat) + l, by omega⟩ a := by
    intro k j l
    fin_cases k <;> fin_cases j
    · simp only [Mmul, hm00]
      exact (hM _).1
    · simp only [Mmul, hm01]
      exact (hM _).1
    · simp only [Mmul, hm10]
      exact (hM _).2
    · simp only [Mmul, hm11]
      exact (hM _).2
  -- The switched keys.
  have hM0 : concatColumns (Mmul 0 0) (Mmul 0 1) w.w_34_0 := x34
  have hM1 : concatColumns (Mmul 1 0) (Mmul 1 1) w.w_35_0 := x35
  have hT := autKeyLoop hM0 hM1 x36
  refine ⟨hMrow, fun k j l ↦ ?_, ?_⟩
  · fin_cases k
    · exact (hT _).1
    · exact (hT _).2
  -- The output keys.
  obtain ⟨_, hw39⟩ := familyGet_eq (k := 0) x39
  obtain ⟨_, hw44⟩ := familyGet_eq (k := 1) x44
  have hw42 := routing_one x42
  have hw48 := routing_one x48
  obtain ⟨g, VK, hg, hVK, hports⟩ := branchKeys x59
  have hgg := gadgetMatrixRuns_unique x52 hg
  subst hgg
  rw [hw39, hw44] at hVK
  have hV1 : gadgetDecomposeRuns Backend.backend 32768 12
      (matrixMulScalarLeft (lutScalarOne (tables 0) (tables 1)) w.w_52_0) w.w_54_0 := by
    convert x54 using 2
    simp only [lutScalarOne, lutScalar, hw39, hw44, hw42, hw48, id]
    rfl
  have hout : ∀ k j : Fin 2, ![![w.w_59_0, w.w_59_1], ![w.w_59_2, w.w_59_3]] k j =
      outputKey model key Mmul T0 T1 w.w_54_0 VK k j := by
    intro k j
    rw [hports k j]
    fin_cases k <;> fin_cases j <;> rfl
  exact ⟨w.w_52_0, w.w_54_0, VK, x52, hV1, hVK, hout, hout⟩

end PeProof
