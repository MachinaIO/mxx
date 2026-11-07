import PeCompanions

/-!
# The product of a step

The state encoding of `μ` times the attribute ciphertext `(R, Y)` of `X^x` (public top row `R`),
plus the companion row of the bottom digit rows, encodes `μ X^x` under the product key
`M_{k,j} = Σ_a B_{jj,a,k} Dec(A_{k,44+a}) + A_{k,0} topRows(Dec(R))`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- The product error bound. -/
def productBound (B Bz : Nat) : Nat :=
  (44 + 44) * N * B * 32768 + 44 * (44 * N * Bz * 32768 + 44 * N * Bz * 32768) + 1 * 31

set_option maxHeartbeats 16000000 in
/-- The product of a step: the state rows times the digits of the attribute ciphertext plus the
companion row encode `μ X^x` under the product key. -/
theorem product_rel {s μ : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    {c : Fin 2 → ExactMatrix Q N 1 44} {B : Nat}
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0) c B)
    (hμ : SmulGain μ 1) {x : Nat} {R Y : Fin 2 → ExactMatrix Q N 1 44} {eB : Fin 2 → ErrorMatrix N 1 44}
    (hrhs : ∀ j, reducePoly Q N s • R j + Y j =
      reducePoly Q N (Xe x * unitPair s j) • g0 + reduceMatrix Q N 1 44 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 31)
    (hD : ∀ (j : Fin 2) (l : Fin 44), ∃ D' : ExactMatrix Q N (44 + 44) 1,
      gadgetDecomposeRuns Backend.backend 65536 44 (pairColumn (R j 0 l) (Y j 0 l)) D' ∧
        compactCenteredRebaseRuns 32768 D' (fun a _ ↦ actionDigits (joinRow R) (joinRow Y) j a l))
    {Bk : Fin 88 → Fin 44 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    {z : Fin 88 → Fin 44 → Fin 2 → ExactMatrix Q N 1 44} {Bz : Nat}
    (hz : ∀ jj a, Rel s (Bk jj a)
      (fun r ↦ (reducePoly Q N (entryDigit (joinedEntry Y jj) a) * reducePoly Q N (unitPair s r)) • g0)
      (z jj a) Bz)
    {E : ExactMatrix Q N 1 88}
    (hErow : ∀ jj, E 0 jj = ∑ a, (z jj a 0 * keyDigits A (Fin.natAdd 44 a) 0 +
      z jj a 1 * keyDigits A (Fin.natAdd 44 a) 1) 0 0)
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    (hM : ∀ k j l, M k j 0 l = (∑ a, (Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 0 * keyDigits A (Fin.natAdd 44 a) 0 +
      Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 1 * keyDigits A (Fin.natAdd 44 a) 1)) 0 0 +
      ∑ a : Fin 44, A k 0 0 a * rowDigits (joinRow R) a ⟨44 * (j : Nat) + l, by omega⟩) :
    Rel s M (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * (μ * Xe x) * unitPair s j) • g0)
      (fun j ↦ pairMul c (actionDigits (joinRow R) (joinRow Y) j) + halves E j) (productBound B Bz) := by
  set D := actionDigits (joinRow R) (joinRow Y) with hDdef
  obtain ⟨f, hf, hE⟩ := companion_rows_bottom (q := Q) (m := 44) (m' := 44) (Dg := 44) hN
    (sm := s) (g := g0) (A := A) (D := D)
    (B := fun j l a ↦ Bk ⟨44 * (j : Nat) + l, by omega⟩ a)
    (z := fun j l a ↦ z ⟨44 * (j : Nat) + l, by omega⟩ a) (bound := Bz)
    (fun j l a ↦ by
      refine Rel.target_eq (hz _ a) ?_
      funext r
      rw [hDdef, actionDigits_bottom]
      rfl)
    (u := fun a r ↦ keyDigits A (Fin.natAdd 44 a) r)
    (fun a r ↦ by
      rw [(key_digits A (Fin.natAdd 44 a) r).1]
      funext x y
      simp only [joinedEntry, Fin.addCases_right])
    (u' := fun a r ↦ (key_digits A (Fin.natAdd 44 a) r).2.choose)
    (fun a r ↦ (key_digits A (Fin.natAdd 44 a) r).2.choose_spec.1)
    (fun a r ↦ (key_digits A (Fin.natAdd 44 a) r).2.choose_spec.2)
    (E := fun j ↦ halves E j)
    (M := fun k j ↦ fun _ l ↦ (∑ a, (Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 0 * keyDigits A (Fin.natAdd 44 a) 0 +
      Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 1 * keyDigits A (Fin.natAdd 44 a) 1)) 0 0)
    (fun j l ↦ by simp only [halves, rowWindow]; exact hErow _)
    (fun k j l ↦ rfl)
  set Mc : Fin 2 → Fin 2 → ExactMatrix Q N 1 44 := fun k j ↦ fun _ l ↦
    (∑ a, (Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 0 * keyDigits A (Fin.natAdd 44 a) 0 +
      Bk ⟨44 * (j : Nat) + l, by omega⟩ a k 1 * keyDigits A (Fin.natAdd 44 a) 1)) 0 0 with hMc
  have h := mul_public_top hN hQ Backend.layout0 layoutQ rfl g0_runs hg0 henc (top := R) (bot := Y) (D := D)
    hD hrhs heB hμ (E := fun j ↦ halves E j) (M := Mc) hE hf
  have hkey : (fun k j ↦ Mc k j + pairMulTop (A k) (D j)) = M := by
    funext k j r l
    rw [Subsingleton.elim r 0, hM]
    simp only [Matrix.add_apply, hMc, pairMulTop, Matrix.mul_apply, topRows, Matrix.submatrix_apply, id]
    refine congrArg₂ (· + ·) rfl (Finset.sum_congr rfl fun a _ ↦ ?_)
    rw [hDdef, actionDigits_top (joinRow R) (joinRow Y) j a l]
  rw [hkey] at h
  refine Rel.mono (Rel.target_eq h ?_) (le_of_eq (by unfold productBound; ring))
  funext j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

end PeProof
