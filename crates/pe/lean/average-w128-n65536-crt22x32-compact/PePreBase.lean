import PeEncrypt
import Stage_preprocess
import Stage_preprocess_Facts

/-!
# Preprocessing: shared definitions

The public evaluation of the keys pairs the expanded companion keys with key digits, and adds the
public top terms of the digits of public rows.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The two key rows a key loop accumulates. -/
abbrev KeyRows := ExactMatrix Q N 1 88 × ExactMatrix Q N 1 88 × Unit

/-- The digit-row sum `B_{k0} u_0 + B_{k1} u_1` of the expanded product companion keys of position
`p` at column `jj` and private digit row `a`, for key row `k`. -/
noncomputable def keyCompanionSum (model : HashModel) (key : ByteArray) (p : Nat)
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44) (k : Fin 2) (jj : Fin 88) (a : Fin 44) : ExactPoly Q N :=
  (expand (rhsCompKey model key p jj (44 + a) k 0) * keyDigits A ⟨44 + a, by omega⟩ 0 +
    expand (rhsCompKey model key p jj (44 + a) k 1) * keyDigits A ⟨44 + a, by omega⟩ 1) 0 0

/-- The digits of a key column read by a slice and a decomposition. -/
theorem key_column_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {r : Fin 2} {Ar : ExactMatrix Q N 1 88}
    {c : Nat} (hc : c < 88) {y : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 44 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar) (hy : sliceMatrix Ar 0 1 (c : Int) (1 + (c : Int)) y)
    (hu : gadgetDecomposeRuns Backend.backend 65536 44 y u) : u = keyDigits A ⟨c, hc⟩ r := by
  obtain ⟨_, hy0⟩ := sliceMatrix_column hy
  rw [concatColumns_entry hAr, pair_eta] at hy0
  have hye : y = fun _ _ ↦ joinedEntry (A r) ⟨c, hc⟩ := by
    funext a b
    rw [Subsingleton.elim a 0, Subsingleton.elim b 0, hy0]
    congr 1
  rw [hye] at hu
  exact entryDigits_eq hu

/-- The digits of a public row, decomposed column by column and rebased. -/
noncomputable def rowDigits (R : ExactMatrix Q N 1 88) : ExactMatrix Q N 44 88 :=
  Classical.epsilon fun D ↦ ∃ D' : ExactMatrix Q N 44 88,
    gadgetDecomposeRuns Backend.backend 65536 44 R D' ∧ compactCenteredRebaseRuns 32768 D' D

theorem rowDigits_eq {R : ExactMatrix Q N 1 88} {D' D : ExactMatrix Q N 44 88}
    (hD' : gadgetDecomposeRuns Backend.backend 65536 44 R D') (hD : compactCenteredRebaseRuns 32768 D' D) :
    D = rowDigits R := by
  obtain ⟨E', hE', hE⟩ := Classical.epsilon_spec (p := fun D ↦ ∃ D' : ExactMatrix Q N 44 88,
    gadgetDecomposeRuns Backend.backend 65536 44 R D' ∧ compactCenteredRebaseRuns 32768 D' D) ⟨D, D', hD', hD⟩
  obtain rfl := gadgetDecomposeRuns_unique hD' hE'
  exact centeredRebaseRuns_unique hD.2.2.1 hE.2.2.1

/-- The digit-row sum of the expanded LUT companion keys of first-depth branch `b` at column `jj`
and private digit row `a`, paired with the digits of the routed key `σ_k(A)`, for key row `r`. -/
noncomputable def lutKeyCompanionSum (model : HashModel) (key : ByteArray) (b : Fin 22)
    (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44) (r : Fin 2) (jj : Fin 88) (a : Fin 44) : ExactPoly Q N :=
  (expand (lutCompKey model key (kSw b) ((jj : Nat) : Int) ((44 + a : Nat) : Int) r 0) *
      keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨44 + a, by omega⟩ 0 +
    expand (lutCompKey model key (kSw b) ((jj : Nat) : Int) ((44 + a : Nat) : Int) r 1) *
      keyDigits (routedKey (kSw b) (kSw_odd b) A) ⟨44 + a, by omega⟩ 1) 0 0

/-- The digits of a routed key column read by a slice, a routing, and a decomposition. -/
theorem routed_column_digits {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {r : Fin 2}
    {Ar : ExactMatrix Q N 1 88} {b : Nat} (hb : b < 22) {c : Nat} (hc : c < 88)
    {x y : ExactMatrix Q N 1 1} {u : ExactMatrix Q N 44 1}
    (hAr : concatColumns (A r 0) (A r 1) Ar) (hx : sliceMatrix Ar 0 1 (c : Int) (1 + (c : Int)) x)
    (hy : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat b) switchList) x y)
    (hu : gadgetDecomposeRuns Backend.backend 65536 44 y u) :
    u = keyDigits (routedKey (kSw b) (kSw_odd ⟨b, hb⟩) A) ⟨c, hc⟩ r := by
  obtain ⟨_, hx0⟩ := sliceMatrix_column hx
  rw [concatColumns_entry hAr, pair_eta] at hx0
  have hye : y = fun _ _ ↦ joinedEntry (routedKey (kSw b) (kSw_odd ⟨b, hb⟩) A r) ⟨c, hc⟩ := by
    rw [routing_kSw hb hy]
    funext a d
    rw [Subsingleton.elim a 0, Subsingleton.elim d 0]
    show substitution _ _ (x 0 0) = _
    rw [hx0]
    show _ = joinedEntry (fun j ↦ substituteMatrix (kSw b) (kSw_odd ⟨b, hb⟩) (A r j)) _
    rw [joinedEntry_substitute]
    rfl
  rw [hye] at hu
  exact entryDigits_eq hu

/-- The public top row of the switching ciphertext of first-depth branch `b`. -/
noncomputable def switchTopRow (model : HashModel) (key : ByteArray) (b : Fin 22) : ExactMatrix Q N 1 88 :=
  joinRow ![substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 0),
    substituteMatrix (kSw b) (kSw_odd b) (helperKey model key (kInv b) 0 1)]

/-- The four key blocks a key sum accumulates. -/
abbrev KeyState := ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 × ExactMatrix Q N 1 44 ×
  ExactMatrix Q N 1 44 × Unit

/-- The blocks of a key state as a key. -/
def keyOf (s : KeyState) : Fin 2 → Fin 2 → ExactMatrix Q N 1 44 := ![![s.1, s.2.1], ![s.2.2.1, s.2.2.2.1]]

/-! ## Generic loop lemmas of the preprocessing lanes -/

theorem concat_left {a b : Nat} {x : ExactMatrix Q N 1 a} {y : ExactMatrix Q N 1 b}
    {T : ExactMatrix Q N 1 (a + b)} (h : concatColumns x y T) (c : Nat) (hc : c < a) :
    T 0 ⟨c, by omega⟩ = x 0 ⟨c, hc⟩ := by
  rw [h 0 ⟨c, by omega⟩, dif_pos hc]

/-- The entry `a < 44` of the joined key row `r` is the key entry. -/
theorem key_row_entry {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {A0 A1 : ExactMatrix Q N 1 88}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    (r : Fin 2) (a : Fin 44) : ![A0, A1] r 0 ⟨a, by omega⟩ = A r 0 0 a := by
  fin_cases r
  · exact concat_left hA0 a a.isLt
  · exact concat_left hA1 a a.isLt

set_option maxHeartbeats 4000000 in
/-- A public top loop over the `44` public digit rows whose iteration `i` adds
`σ_k(X_r[i]) D_i` to row `r` ends with the sum over the digit rows. -/
theorem topAccum {body : Nat → KeyRows → KeyRows → Prop} {k : Nat} (hk : Odd k)
    {X0 X1 : ExactMatrix Q N 1 88} {D : ExactMatrix Q N 44 88} {init final : KeyRows}
    (hstep : ∀ i s t (hi : i < 44), body i s t → ∀ w : Fin 88,
      t.1 0 w = s.1 0 w + substitution k hk (X0 0 ⟨i, by omega⟩) * D ⟨i, hi⟩ w ∧
      t.2.1 0 w = s.2.1 0 w + substitution k hk (X1 0 ⟨i, by omega⟩) * D ⟨i, hi⟩ w)
    (h : MxxIR.IterRuns body 44 init final) (r : Fin 2) (w : Fin 88) :
    ![final.1, final.2.1] r 0 w = ![init.1, init.2.1] r 0 w +
      ∑ a : Fin 44, substitution k hk (![X0, X1] r 0 ⟨a, by omega⟩) * D a w := by
  have hacc := IterRuns.accumulate_entries (fun s : KeyRows ↦ ![s.1, s.2.1] r)
    (fun w (i : Fin 44) ↦ substitution k hk (![X0, X1] r 0 ⟨i, by omega⟩) * D i w)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨fun w ↦ by
      have e := hstep i s t hi hb w
      fin_cases r
      · exact e.1
      · exact e.2, trivial⟩) h w
  exact hacc

set_option maxHeartbeats 4000000 in
/-- A key loop over the `44` private digit rows whose iteration `i` adds `F r w i` to entry `w` of
row `r` ends with the sum over the digit rows. -/
theorem keyAccum {body : Nat → KeyRows → KeyRows → Prop} {F : Fin 2 → Fin 88 → Fin 44 → ExactPoly Q N}
    {init final : KeyRows}
    (hstep : ∀ i s t (hi : i < 44), body i s t → ∀ (r : Fin 2) (w : Fin 88),
      ![t.1, t.2.1] r 0 w = ![s.1, s.2.1] r 0 w + F r w ⟨i, hi⟩)
    (h : MxxIR.IterRuns body 44 init final) (r : Fin 2) (w : Fin 88) :
    ![final.1, final.2.1] r 0 w = ![init.1, init.2.1] r 0 w + ∑ a, F r w a :=
  IterRuns.accumulate_entries (fun s : KeyRows ↦ ![s.1, s.2.1] r) (fun w i ↦ F r w i)
    (fun _ _ ↦ True) trivial (fun i s t hi _ hb ↦ ⟨fun w ↦ hstep i s t hi hb r w, trivial⟩) h w

set_option maxHeartbeats 4000000 in
/-- A column loop over the `88` columns whose iteration `j` adds `c r j` at column `j` of row `r`,
from zero rows, ends with the values `c r jj`. -/
theorem colAccum {body : Nat → KeyRows → KeyRows → Prop} {c : Fin 2 → Fin 88 → ExactPoly Q N}
    {init final : KeyRows} (h0 : init = (0, 0, ()))
    (hstep : ∀ j s t (hj : j < 88), body j s t →
      t.1 = s.1 + c 0 ⟨j, hj⟩ • (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 88) ∧
      t.2.1 = s.2.1 + c 1 ⟨j, hj⟩ • (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 88))
    (h : MxxIR.IterRuns body 88 init final) (r : Fin 2) (jj : Fin 88) :
    ![final.1, final.2.1] r 0 jj = c r jj := by
  subst h0
  have hacc := IterRuns.accumulate_entries (fun s : KeyRows ↦ ![s.1, s.2.1] r)
    (fun w (j : Fin 88) ↦ c r j * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 88) 0 w)
    (fun _ _ ↦ True) trivial (fun j s t hj _ hb ↦ ⟨fun w ↦ by
      have e := hstep j s t hj hb
      fin_cases r
      · show t.1 0 w = _
        rw [e.1]
        rfl
      · show t.2.1 0 w = _
        rw [e.2]
        rfl, trivial⟩) h jj
  rw [hacc, sum_unitRow_mul (fun j ↦ c r j) jj]
  fin_cases r <;> simp

theorem carriedColAccum {State : Type} {body : Nat → State → State → Prop}
    (row : State → ExactMatrix Q N 1 88) (c : Fin 88 → ExactPoly Q N)
    {init final : State}
    (hstep : ∀ j s t (hj : j < 88), body j s t →
      row t = row s + c ⟨j, hj⟩ • unitRow ((j : Nat) : Int))
    (h : MxxIR.IterRuns body 88 init final) (jj : Fin 88) :
    row final 0 jj = row init 0 jj + c jj := by
  have hacc := IterRuns.accumulate_entries row
    (fun w (j : Fin 88) ↦ c j * (unitRow ((j : Nat) : Int) : ExactMatrix Q N 1 88) 0 w)
    (fun _ _ ↦ True) trivial
    (fun j s t hj _ hb ↦ ⟨fun w ↦ by rw [hstep j s t hj hb]; rfl, trivial⟩) h jj
  rw [hacc, sum_unitRow_mul c jj]

set_option maxHeartbeats 4000000 in
/-- A member sum over a family of `n + 2` key terms: the members `0` and `1` are added before the
loop, and iteration `i` adds member `i + 2`. -/
theorem memberAccum {n : Nat} {body : Nat → KeyState → KeyState → Prop}
    {f : Fin (n + 2) → Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {init final : KeyState}
    (hinit : ∀ r j, keyOf init r j = f 0 r j + f 1 r j)
    (hstep : ∀ i s t (hi : i < n), body i s t → ∀ r j,
      keyOf t r j = keyOf s r j + f ⟨i + 2, by omega⟩ r j)
    (h : MxxIR.IterRuns body n init final) (r j : Fin 2) :
    keyOf final r j = ∑ m, f m r j := by
  have hacc := IterRuns.accumulate (fun s : KeyState ↦ keyOf s r j)
    (fun i ↦ if hi : i < n then f ⟨i + 2, by omega⟩ r j else 0) (fun _ _ ↦ True) trivial
    (fun i s t hi _ hb ↦ ⟨by simp only [dif_pos hi]; exact hstep i s t hi hb r j, trivial⟩) h
  obtain ⟨hs, -⟩ := hacc
  dsimp only at hs
  rw [hs, hinit, ← Fin.sum_univ_eq_sum_range (fun i ↦ if hi : i < n then f ⟨i + 2, by omega⟩ r j else 0),
    Fin.sum_univ_succ, Fin.sum_univ_succ, add_assoc]
  congr 2
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [dif_pos i.isLt]
  rfl


/-! ## Second-depth branches -/

theorem leaf_switch {m : Nat} (hm : m < 7) :
    MxxIR.selectInt (Int.ofNat m) [16385, 32769, 49153, 65537, 81921, 98305, 114689] =
      ((kSw (15 + m) : Nat) : Int) := by
  interval_cases m <;> rfl

theorem leaf_inverse {m : Nat} (hm : m < 7) :
    MxxIR.selectInt (Int.ofNat m) [114689, 98305, 81921, 65537, 49153, 32769, 16385] =
      ((kInv (15 + m) : Nat) : Int) := by
  interval_cases m <;> rfl

theorem leaf_position (p : Nat) {m : Nat} (hm : m < 7) :
    MxxIR.selectInt (Int.ofNat m)
      [p, p + 15, p + 30, p + 45, p + 60, p + 75, p + 90] = ((p + 15 * m : Nat) : Int) := by
  interval_cases m <;> simp [MxxIR.selectInt] <;> push_cast <;> ring

theorem leaf_member {m : Nat} (hm : m < 7) :
    MxxIR.selectInt (Int.ofNat m) [15, 16, 17, 18, 19, 20, 21] = ((15 + m : Nat) : Int) := by
  interval_cases m <;> rfl

theorem member_cast {d : Nat} (hd : d < 105) :
    MxxIR.selectInt (Int.ofNat d) switchList2 = ((kSw (15 + d / 15) : Nat) : Int) := by
  rw [switchList2_eq d hd, kSw_cast _ (by omega)]

/-- A routing by the automorphism of second-depth branch `d` is the substitution of its member's. -/
theorem routing2 {d : Nat} (hd : d < 105) {rows columns : Nat} {x y : ExactMatrix Q N rows columns}
    (h : ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat d) switchList2) x y) :
    y = substituteMatrix (kSw (15 + d / 15)) (kSw_odd (member2 d hd)) x := by
  rw [switchList2_eq d hd] at h
  exact routing_kSw (by omega) h

/-- The parent of second-depth branch `d`. -/
theorem parent_read {α : Type} {fam : Fin 15 → α} {d : Nat} (hd : d < 105) {out : α}
    (h : familyGetDynamic fam (MxxIR.selectInt (Int.ofNat d) parentList2) out) :
    out = fam ⟨d % 15, by omega⟩ := by
  rw [parentList2_eq d hd] at h
  obtain ⟨_, e⟩ := familyGet_eq h
  exact e

end PeProof
