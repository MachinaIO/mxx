import PeLut
import PePublicRows

/-!
# The encoding relation through one LUT branch

A branch of automorphism `k` routes the step's product encoding by `σ_k`, switches it with the
switching ciphertext `C_k` (public top row `σ_k(H_{k⁻¹}^{(0)})`) and the companions of the bottom
digit rows (`aut_switch_public`), moves the mask back with the helper of `k` and the digits of
the imported key `T` (`mask_switch`), and applies the public scalar action `V`
(`Rel.blockAction`).
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem hg0 : centeredRebaseRuns g0 g0 := centeredRebaseRuns_self hQ hN g0

/-- The digits of key column `a` reconstruct it, within `base / 2`. -/
theorem key_digits (A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) (a : Fin (30 + 30)) (r : Fin 2) :
    g0 * keyDigits A a r = (fun _ _ ↦ joinedEntry (A r) a) ∧ PreimageWithin (keyDigits A a r) 2048 :=
  decompose_entry hQ hN Backend.layout0 layoutQ rfl g0_runs
    (entryDigits_runs hQ hN g0_runs Backend.layout0 layoutQ rfl _)

/-- The public scalar action `V = Dec(D g)` on a gadget target: `(x g) V = (x D) g`. -/
theorem gadget_scalar_action {D : ExactMatrix Q N 1 1} {V : ExactMatrix Q N 30 30}
    (hV : gadgetDecomposeRuns Backend.backend 4096 30 (matrixMulScalarLeft D g0) V) (x : ExactPoly Q N) :
    (x • g0) * V = (x * D 0 0) • g0 := by
  rw [Matrix.smul_mul, (decompose_row hQ hN Backend.layout0 layoutQ rfl g0_runs hV).1,
    matrixMulScalarLeft_eq, smul_smul]

/-- The digits of a public scalar action are within `base / 2`. -/
theorem scalar_digits {D : ExactMatrix Q N 1 1} {V : ExactMatrix Q N 30 30}
    (hV : gadgetDecomposeRuns Backend.backend 4096 30 (matrixMulScalarLeft D g0) V) :
    ∃ v : ErrorMatrix N 30 30, V = reduceMatrix Q N 30 30 v ∧ CoeffBound v 2048 :=
  (decompose_row hQ hN Backend.layout0 layoutQ rfl g0_runs hV).2

/-- The error bound of a branch row before its scalar action, from the bounds of the routed
encoding `B`, the helpers `Bh`, the companions `Bz`, and the gain `K` of the routed payload. -/
def switchBound (B Bh Bz K : Nat) : Nat :=
  (30 + 30) * N * Bh * 2048 +
    ((30 + 30) * N * B * 2048 + 30 * (30 * N * Bz * 2048 + 30 * N * Bz * 2048) + K * Bh)

/-- The error bound after a public scalar action. -/
def scalarBound (B : Nat) : Nat := 30 * N * B * 2048

set_option maxHeartbeats 4000000 in
/-- The public scalar action `V = Dec(W⁻¹ α g)` of a branch: an encoding of `μ` becomes an encoding
of `μ (W⁻¹ α)` under the acted key. -/
theorem scalar_rel {s μ : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    {c : Fin 2 → ExactMatrix Q N 1 30} {B : Nat}
    (h : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0) c B)
    {coef : ExactMatrix Q N 1 1} {V : ExactMatrix Q N 30 30}
    (hV : gadgetDecomposeRuns Backend.backend 4096 30
      (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g0) V) :
    Rel s (fun r j ↦ A r j * V)
      (fun j ↦ (reducePoly Q N μ * reducePoly Q N (unitPair s j) *
        (matrixMulScalarLeft winvPoly coef) 0 0) • g0)
      (fun j ↦ c j * V) (scalarBound B) := by
  obtain ⟨v, hv, hb⟩ := scalar_digits hV
  have h' := Rel.blockAction hN h v hb
  rw [← hv] at h'
  refine Rel.mono (Rel.target_eq h' ?_) (le_of_eq (by unfold scalarBound; ring))
  funext j
  rw [gadget_scalar_action hV]
  congr 1
  simp only [map_mul, Nat.cast_one, Int.cast_one, one_mul]

set_option maxHeartbeats 16000000 in
/-- One LUT branch of automorphism `k` before its scalar action: the branch row
`h_k Dec(T) + σ_k(c) Dec(C_k) + E` encodes `σ_k(μ)` under the branch key `H_k Dec(T)`. -/
theorem switch_rel {s μ : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    {c : Fin 2 → ExactMatrix Q N 1 30} {B : Nat}
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0) c B)
    {k kinv : Nat} (hk : Odd k) (hkinv : Odd kinv) (hinv : k * kinv % (2 * N) = 1)
    (hcop : N.Coprime k)
    {Hi : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {hi : Fin 2 → ExactMatrix Q N 1 30} {Bh : Nat}
    (hhi : Rel s Hi (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution kinv hkinv s) j) • g0) hi Bh)
    (hD : ∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
      gadgetDecomposeRuns Backend.backend 4096 30
        (pairColumn (substituteMatrix k hk (Hi 0 j) 0 l)
          ((substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) 0 l)) D' ∧
        compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits
          (joinRow fun j ↦ substituteMatrix k hk (Hi 0 j))
          (joinRow fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) j a l))
    {K : Nat} (hμ : SmulGain (substitution k hk μ) K)
    {Bc : Fin 2 → Fin 30 → Fin 30 → Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    {z : Fin 2 → Fin 30 → Fin 30 → Fin 2 → ExactMatrix Q N 1 30} {Bz : Nat}
    (hz : ∀ j l a, Rel (substitution k hk s) (Bc j l a)
      (fun r ↦ (actionDigits
          (joinRow fun j ↦ substituteMatrix k hk (Hi 0 j))
          (joinRow fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) j
          (Fin.natAdd 30 a) l * reducePoly Q N (unitPair (substitution k hk s) r)) • g0)
      (z j l a) Bz)
    {E : Fin 2 → ExactMatrix Q N 1 30}
    (hErow : ∀ j l, E j 0 l = (∑ a, (z j l a 0 * keyDigits (routedKey k hk A) (Fin.natAdd 30 a) 0 +
      z j l a 1 * keyDigits (routedKey k hk A) (Fin.natAdd 30 a) 1)) 0 0)
    {Mc : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    (hMrow : ∀ r j l, Mc r j 0 l = (∑ a, (Bc j l a r 0 * keyDigits (routedKey k hk A) (Fin.natAdd 30 a) 0 +
      Bc j l a r 1 * keyDigits (routedKey k hk A) (Fin.natAdd 30 a) 1)) 0 0)
    {Hk : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {hrow : Fin 2 → ExactMatrix Q N 1 30}
    (hhk : Rel s Hk (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution k hk s) j) • g0) hrow Bh)
    {Tb0 Tb1 : ExactMatrix Q N 1 60}
    (hT : ∀ r j, halves (![Tb0, Tb1] r) j = Mc r j + pairMulTop (fun i ↦ substituteMatrix k hk (A r i))
      (actionDigits (joinRow fun j ↦ substituteMatrix k hk (Hi 0 j))
        (joinRow fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) j))
    (hDt : ∀ (j : Fin 2) (l : Fin 30), ∃ D' : ExactMatrix Q N (30 + 30) 1,
      gadgetDecomposeRuns Backend.backend 4096 30 (pairColumn (halves Tb0 j 0 l) (halves Tb1 j 0 l)) D' ∧
        compactCenteredRebaseRuns 2048 D' (fun a _ ↦ actionDigits Tb0 Tb1 j a l)) :
    Rel s (fun r j ↦ pairMul (Hk r) (actionDigits Tb0 Tb1 j))
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * substitution k hk μ * unitPair s j) • g0)
      (fun j ↦ pairMul hrow (actionDigits Tb0 Tb1 j) +
        (pairMul (fun i ↦ substituteMatrix k hk (c i))
          (actionDigits (joinRow fun j ↦ substituteMatrix k hk (Hi 0 j))
            (joinRow fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) j) + E j))
      (switchBound B Bh Bz K) := by
  set D := actionDigits (joinRow fun j ↦ substituteMatrix k hk (Hi 0 j))
    (joinRow fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j)) with hDdef
  -- The companion row of the switch.
  obtain ⟨f, hf, hE⟩ := companion_rows_bottom (q := Q) (m := 30) (m' := 30) (Dg := 30) hN
    (sm := substitution k hk s) (g := g0) (A := routedKey k hk A) (D := D) (B := Bc) (z := z)
    (bound := Bz) hz (u := fun a r ↦ keyDigits (routedKey k hk A) (Fin.natAdd 30 a) r)
    (fun a r ↦ by
      rw [(key_digits (routedKey k hk A) (Fin.natAdd 30 a) r).1]
      funext x y
      simp only [joinedEntry, Fin.addCases_right])
    (u' := fun a r ↦ (key_digits (routedKey k hk A) (Fin.natAdd 30 a) r).2.choose)
    (fun a r ↦ (key_digits (routedKey k hk A) (Fin.natAdd 30 a) r).2.choose_spec.1)
    (fun a r ↦ (key_digits (routedKey k hk A) (Fin.natAdd 30 a) r).2.choose_spec.2)
    (E := E) (M := Mc) hErow hMrow
  -- The switch.
  have hsw := aut_switch_public hN hQ Backend.layout0 layoutQ rfl g0_runs hg0 henc hk hkinv hinv hcop
    hhi (top := fun j ↦ substituteMatrix k hk (Hi 0 j))
    (bot := fun j ↦ substituteMatrix k hk (Hi 1 j) - substituteMatrix k hk (hi j))
    (fun _ ↦ rfl) (fun _ ↦ rfl) (D := D) (fun j l ↦ by simpa [joinRow, joinedEntry_pair] using hD j l) hμ
    (E := E) (M := Mc) (fun j ↦ hE j) hf
  -- The mask switch with the imported key.
  have hT' : (fun r j ↦ Mc r j + pairMulTop (fun i ↦ substituteMatrix k hk (A r i)) (D j)) =
      fun r j ↦ halves (![Tb0, Tb1] r) j := by
    funext r j
    rw [hT r j]
  rw [hT'] at hsw
  have hmask := mask_switch hN hQ Backend.layout0 layoutQ rfl g0_runs hg0 hhk hsw
    (Dt := actionDigits Tb0 Tb1) (fun j l ↦ by simpa using hDt j l)
  refine Rel.mono (Rel.target_eq hmask ?_) (le_of_eq (by unfold switchBound; ring))
  funext j
  simp only [map_mul, Nat.cast_one, Int.cast_one, one_mul, smul_smul]

end PeProof
