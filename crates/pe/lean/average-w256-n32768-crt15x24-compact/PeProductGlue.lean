import PeStepData

/-!
# The product of a step from the runtime

The online column loop and companion loop of a step, the attribute ciphertext of the position it
reads, the encryption's product companions, and the preprocessed product key give `product_rel`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem Xe_mul_Xe (a b : Nat) : Xe (n := N) a * Xe (n := N) b = Xe (n := N) (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem mulIndex_div (x : Fin 2) (a : Fin 30) (jj : Fin 60) :
    (mulIndex x a jj : Nat) / 1800 = x ∧ (mulIndex x a jj : Nat) % 60 = jj ∧
      (mulIndex x a jj : Nat) / 60 % 30 = a := by
  simp only [mulIndex]
  omega

set_option maxHeartbeats 64000000 in
/-- The product of a step at the runtime: the halves of the product row encode `X^(v + x)` under the
preprocessed product key. -/
theorem product_glue {model : HashModel} {key : ByteArray} {s : ErrorPoly N} (hs : polyNorm s ≤ 1)
    {v xv : Nat} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {B : Nat} {c0 c1 : ExactMatrix Q N 1 30}
    {C : ExactMatrix Q N 1 60} (hC : concatColumns c0 c1 C)
    (henc : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe v * unitPair s j) • g0)
      ![c0, c1] B)
    (p : Fin 2) {Y0 Y1 : ExactMatrix Q N 1 30} {eB : Fin 2 → ErrorMatrix N 1 30}
    (hrhs : ∀ j, reducePoly Q N s • ![rhsTop model key p 0, rhsTop model key p 1] j + ![Y0, Y1] j =
      reducePoly Q N (Xe xv * unitPair s j) • g0 + reduceMatrix Q N 1 30 (eB j))
    (heB : ∀ j, CoeffBound (eB j) 31)
    {P1 : ExactMatrix Q N 1 60}
    (hcol : ∀ jj, P1 0 jj = (C * rhsDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1])
        (joinRow ![Y0, Y1]) jj) 0 0 ∧
      ∃ D' : ExactMatrix Q N (30 + 30) 1, gadgetDecomposeRuns Backend.backend 4096 30
        (pairColumn (joinRow ![rhsTop model key p 0, rhsTop model key p 1] 0 jj) (joinRow ![Y0, Y1] 0 jj)) D' ∧
        compactCenteredRebaseRuns 2048 D' (rhsDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1])
          (joinRow ![Y0, Y1]) jj))
    {zc0 zc1 : Fin 3600 → ExactMatrix PQ N 1 1} {ez : Fin 3600 → Fin 2 → ErrorMatrix N 1 1}
    (hcomp : ∀ i : Fin 3600, (i : Nat) / 1800 = p →
      RelE s (rhsCompKey model key ((i / 1800 : Nat) : Int) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![Y0, Y1] ⟨i % 60, by omega⟩) ⟨(i / 60) % 30, by omega⟩ * unitPair s j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    {P2 : ExactMatrix Q N 1 60}
    (hmul : ∀ jj, P2 0 jj = ∑ a, mulCompanionSum A p (expandFam zc0) (expandFam zc1) jj a)
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    (hM : ∀ r (j : Fin 2) (l : Fin 30), M r j 0 l = ∑ a, keyCompanionSum model key p A r ⟨30 * (j : Nat) + l, by omega⟩ a +
      ∑ a : Fin 30, A r 0 0 a * rowDigits (joinRow ![rhsTop model key p 0, rhsTop model key p 1]) a
        ⟨30 * (j : Nat) + l, by omega⟩) :
    Rel s M (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe (v + xv) * unitPair s j) • g0)
      ![rowWindow 30 0 (by omega) (P1 + P2), rowWindow 30 30 (by omega) (P1 + P2)]
      (productBound B companionBound) := by
  have hcs := colActionSplit (row := ![c0, c1]) hC (top := joinRow ![rhsTop model key p 0, rhsTop model key p 1])
    (bot := joinRow ![Y0, Y1]) (tk := ![rhsTop model key p 0, rhsTop model key p 1]) (bk := ![Y0, Y1])
    (fun _ ↦ rfl) (fun _ ↦ rfl) hcol
  have hμ : SmulGain (Xe (n := N) v) 1 := by
    rw [Xe]
    exact smulGain_root_pow hN1 _
  have hr := product_rel (s := s) (μ := Xe v) henc hμ (x := xv) (R := ![rhsTop model key p 0, rhsTop model key p 1])
    (Y := ![Y0, Y1]) (eB := eB) (fun j ↦ by fin_cases j <;> exact hrhs _) heB hcs.2
    (Bk := fun jj a ↦ expandKey (rhsCompKey model key (p : Nat) jj (30 + a)))
    (z := fun jj a r ↦ expand (![zc0 (mulIndex p a jj), zc1 (mulIndex p a jj)] r)) (Bz := companionBound)
    (fun jj a ↦ by
      obtain ⟨h1, h2, h3⟩ := mulIndex_div p a jj
      obtain ⟨hc, he⟩ := hcomp (mulIndex p a jj) h1
      have e1 : (⟨(mulIndex p a jj : Nat) % 60, by omega⟩ : Fin 60) = jj := Fin.ext h2
      have e2 : (⟨(mulIndex p a jj : Nat) / 60 % 30, by omega⟩ : Fin 30) = a := Fin.ext h3
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
