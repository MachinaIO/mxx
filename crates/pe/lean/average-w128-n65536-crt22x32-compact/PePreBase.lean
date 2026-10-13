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

end PeProof
