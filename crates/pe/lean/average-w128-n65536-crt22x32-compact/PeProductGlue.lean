import PeStepData

/-!
# The product of a step from the runtime

The online column loop and companion loop of a step, the attribute ciphertext of the position it
reads, the encryption's product companions, and the preprocessed product key give `product_rel`.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem Xe_mul_Xe (a b : Nat) : Xe (n := N) a * Xe (n := N) b = Xe (n := N) (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem mulIndex_div (x : Fin 4) (a : Fin 44) (jj : Fin 88) :
    (mulIndex x a jj : Nat) / 3872 = x ∧ (mulIndex x a jj : Nat) % 88 = jj ∧
      (mulIndex x a jj : Nat) / 88 % 44 = a := by
  simp only [mulIndex]
  omega

set_option maxHeartbeats 64000000 in
/-- The product of a step at the runtime: the halves of the product row encode `X^(v + x)` under the
preprocessed product key. -/
theorem product_glue {model : HashModel} {key : ByteArray} {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {v xv : Nat} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {B : Nat} {c0 c1 : ExactMatrix Q N 1 44}
    {C : ExactMatrix Q N 1 88} (hC : concatColumns c0 c1 C)
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe v * unitPair s j) • g0)
      ![c0, c1] B)
    (p : Fin 4) {Y0 Y1 : ExactMatrix Q N 1 44} {eB : Fin 2 → ErrorMatrix N 1 44}
    (hrhs : ∀ j, reducePoly Q N s • ![rhsTop model key p 0, rhsTop model key p 1] j + ![Y0, Y1] j =
      reducePoly Q N (Xe xv * unitPair s j) • g0 + reduceMatrix Q N 1 44 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 31)
    {P1 : ExactMatrix Q N 1 88}
    (hcol : ∀ jj, P1 0 jj = (C * rhsDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1])
        (joinRow ![Y0, Y1]) jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (44 + 44) 1, gadgetDecomposeRuns Backend.backend 65536 44
        (pairColumn (joinRow ![rhsTop model key p 0, rhsTop model key p 1] 0 jj) (joinRow ![Y0, Y1] 0 jj)) D' ∧
        compactCenteredRebaseRuns 32768 D' (rhsDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1])
          (joinRow ![Y0, Y1]) jj))
    {zc0 zc1 : Fin 15488 → ExactMatrix PQ N 1 1} {ez : Fin 15488 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 15488, (i : Nat) / 3872 = p →
      RelE s (rhsCompKey model key ((i / 3872 : Nat) : Int) ((i % 88 : Nat) : Int) ((44 + (i / 88) % 44 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![Y0, Y1] ⟨i % 88, by omega⟩) ⟨(i / 88) % 44, by omega⟩ * unitPair s j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    {P2 : ExactMatrix Q N 1 88}
    (hmul : ∀ jj, P2 0 jj = ∑ a, mulCompanionSum A p (expandFam zc0) (expandFam zc1) jj a)
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    (hM : ∀ r (j : Fin 2) (l : Fin 44), M r j 0 l = ∑ a, keyCompanionSum model key p A r ⟨44 * (j : Nat) + l, by omega⟩ a +
      ∑ a : Fin 44, A r 0 0 a * rowDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1]) a
        ⟨44 * (j : Nat) + l, by omega⟩) :
    Rel s M (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe (v + xv) * unitPair s j) • g0)
      ![rowWindow 44 0 (by omega) (P1 + P2), rowWindow 44 44 (by omega) (P1 + P2)]
      (productBound B companionBound) := by
  have hcs := colActionSplit (row := ![c0, c1]) hC (top := joinRow ![rhsTop model key p 0, rhsTop model key p 1])
    (bot := joinRow ![Y0, Y1]) (tk := ![rhsTop model key p 0, rhsTop model key p 1]) (bk := ![Y0, Y1])
    (fun _ ↦ rfl) (fun _ ↦ rfl) hcol
  have hμ : SmulGain (Xe (n := N) v) 1 := by
    rw [Xe]
    exact smulGain_root_pow hN1 _
  have hr := product_rel (s := s) (μ := Xe v) henc hμ (x := xv) (R := ![rhsTop model key p 0, rhsTop model key p 1])
    (Y := ![Y0, Y1]) (eB := eB) (fun j ↦ by fin_cases j <;> exact hrhs _) heB hcs.2
    (Bk := fun jj a ↦ expandKey (rhsCompKey model key (p : Nat) jj (44 + a)))
    (z := fun jj a r ↦ expand (![zc0 (mulIndex p a jj), zc1 (mulIndex p a jj)] r)) (Bz := companionBound)
    (fun jj a ↦ by
      obtain ⟨h1, h2, h3⟩ := mulIndex_div p a jj
      obtain ⟨hc, he⟩ := hcomp (mulIndex p a jj) h1
      have e1 : (⟨(mulIndex p a jj : Nat) % 88, by omega⟩ : Fin 88) = jj := Fin.ext h2
      have e2 : (⟨(mulIndex p a jj : Nat) / 88 % 44, by omega⟩ : Fin 44) = a := Fin.ext h3
      rw [e1, e2] at hc
      simp only [h1, h2, h3] at hc
      have := companion_expand hs hc he
      refine Rel.target_eq this ?_
      rfl)
    (E := P2) (fun jj ↦ by rw [hmul]; rfl)
    (M := M) (fun k j l ↦ by rw [hM]; rfl)
  rw [Xe_mul_Xe] at hr
  convert hr using 1
  funext j a l
  have h := congrFun (congrFun (hcs.1 j) a) l
  rw [Matrix.add_apply, ← h]
  fin_cases j <;> rfl

end PeProof
