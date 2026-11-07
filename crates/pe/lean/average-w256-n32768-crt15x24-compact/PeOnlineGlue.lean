import PeProductGlue

/-!
# One online step from the conclusions of its runtime scopes

The runtime layer regroups the first-depth and second-depth leaves into fifteen member lanes and
sixteen slots per lane. After `memberTerms_sum` restores the branch sums, the internal branches,
leaf decompositions, encryption helpers, LUT companions and preprocessed branch keys give
`step_correct`. The mathematical interface is independent of the runtime artifact layout.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- The key of the summed rows of a step: the product key and the branch keys, each acted on by the
scalar of its coefficient. -/
noncomputable def stepKey (model : HashModel) (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) (d1 : BranchData)
    (d2 : Leaf2Data) (coef : Nat → ExactMatrix Q N 1 1) : Fin 2 → Fin 2 → ExactMatrix Q N 1 30 :=
  fun r j ↦ M r j * scalarV (coef 1) +
    ∑ b : Fin 15, d1.key1 model ⟨b, by omega⟩ r j * scalarV (coef (kSw b)) +
    ∑ i : Fin 15, d1.key1 model ⟨15 + i, by omega⟩ r j * scalarV (coef (kSw (15 + i))) +
    ∑ l : Fin 225, d2.key2 model l r j * scalarV (coef (kLeaf2 l))

set_option maxRecDepth 200000 in
set_option maxHeartbeats 64000000 in
/-- One online step: the product encoding of `X^e` and the online branch data give an encoding of
the table value `T e` in the summed rows. -/
theorem online_step {model : HashModel} {s : ErrorPoly N} (hs : polyNorm s ≤ 1) {e : Nat} (he : e < 256)
    {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {Bp : Nat} (d1 : BranchData) (d2 : Leaf2Data)
    (hP : Rel s M (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe e * unitPair s j) • g0)
      ![d1.c0, d1.c1] Bp)
    {zc0 zc1 : Fin 54000 → ExactMatrix PQ N 1 1} {ez : Fin 54000 → Fin 2 → ErrorMatrix N 1 1}
    (hhelp : ∀ (i : Fin 45) (k : Nat) (hk : Odd k), kHelp i = k → Rel s (helperKey model d1.key k)
      (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * unitPair (substitution k hk s) j) • g0)
      ![d1.hf0 i, d1.hf1 i] 31)
    (hcomp : ∀ i : Fin 54000,
      RelE (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s)
        (lutCompKey model d1.key (kSw (i / 1800)) ((i % 60 : Nat) : Int) ((30 + (i / 60) % 30 : Nat) : Int))
        (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) *
          (entryDigit (joinedEntry ![switchBot model d1.key d1.hf0 d1.hf1 ⟨i / 1800, by omega⟩ 0,
              switchBot model d1.key d1.hf0 d1.hf1 ⟨i / 1800, by omega⟩ 1] ⟨i % 60, by omega⟩)
            ⟨(i / 60) % 30, by omega⟩ *
            unitPair (substitution (kSw (i / 1800)) (kSw_odd ⟨i / 1800, by omega⟩) s) j)))
        ![zc0 i, zc1 i] (ez i) ∧ ∀ j, CoeffBound (ez i j) 31)
    (hE1 : ∀ b : Fin 30, d1.E b = lutCompRow M (expandFam zc0) (expandFam zc1) b)
    (hT1 : ∀ (b : Fin 30) r jj, ![d1.Tb0 b b.isLt, d1.Tb1 b b.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model d1.key b M r jj a +
      ∑ a : Fin 30, substitution (kSw b) (kSw_odd b) (M r 0 0 a) * rowDigits (switchTopRow model d1.key b) a jj)
    {o0 o1 o2 o3 o4 o5 : Fin 15 → ExactMatrix Q N 1 30}
    (hint : ∀ b : Fin 15, d1.Runs model b (by omega) ∧
      (∀ j, ![o0 b, o1 b] j = d1.row model b (by omega) j) ∧
      ∀ r j, ![![o2 b, o3 b], ![o4 b, o5 b]] r j = d1.key1 model ⟨b, by omega⟩ r j)
    (hleaf1 : ∀ i (hi : i < 15), d1.Runs model (15 + i) (by omega))
    (hkey2 : d2.key = d1.key) (hf2 : d2.hf0 = d1.hf0 ∧ d2.hf1 = d1.hf1)
    (hc2 : d2.c0 = o0 ∧ d2.c1 = o1)
    (hE2 : ∀ l : Fin 225, d2.E l = lutCompRow (parentKeys o2 o3 o4 o5 ⟨l % 15, by omega⟩)
      (expandFam zc0) (expandFam zc1) (member2 l l.isLt))
    (hT2 : ∀ (l : Fin 225) r jj, ![d2.Tb0 l l.isLt, d2.Tb1 l l.isLt] r 0 jj =
      ∑ a, lutKeyCompanionSum model d1.key (member2 l l.isLt) (d1.key1 model ⟨l % 15, by omega⟩) r jj a +
      ∑ a : Fin 30, substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
        (d1.key1 model ⟨l % 15, by omega⟩ r 0 0 a) *
        rowDigits (switchTopRow model d1.key (member2 l l.isLt)) a jj)
    (hleaf2 : ∀ i (hi : i < 225), d2.Runs model i hi)
    (table : Fin (2 ^ 8) → ExactPoly Q N) (coef : Nat → ExactMatrix Q N 1 1)
    (hcoef : ∀ k, Odd k → (Winv : ExactPoly Q N) * coef k 0 0 =
      lutFourierCoefficient N (X Q) (Winv : ExactPoly Q N) table k)
    {R : Fin 2 → ExactMatrix Q N 1 30}
    (hR : ∀ j, R j = ![d1.c0, d1.c1] j * scalarV (coef 1) +
        ∑ b : Fin 15, ![o0 b, o1 b] j * scalarV (coef (kSw b)) +
        ∑ i : Fin 15, d1.row model (15 + i) (by omega) j * scalarV (coef (kSw (15 + i))) +
        ∑ l : Fin 225, d2.row model l l.isLt j * scalarV (coef (kLeaf2 l))) :
    Rel s (stepKey model M d1 d2 coef)
      (fun j ↦ (reducePoly Q N (unitPair s j) * table ⟨e, by norm_num; omega⟩) • g0) R
      (stepBound Bp (bound1 Bp) (bound2 Bp)) := by
  have H : BranchHyps model s e M d1 d2 zc0 zc1 ez := by
    refine ⟨hhelp, hcomp, fun b jj ↦ ?_, hT1, fun b ↦ ?_, hkey2, hf2, fun p ↦ ?_, fun l jj ↦ ?_, hT2,
      fun l ↦ hleaf2 l l.isLt⟩
    · show d1.E b 0 jj = _
      rw [hE1]
      rfl
    · obtain ⟨bv, hbv⟩ := b
      by_cases hb : bv < 15
      · exact (hint ⟨bv, hb⟩).1
      · obtain ⟨i, rfl⟩ : ∃ i, bv = 15 + i := ⟨bv - 15, by omega⟩
        exact hleaf1 i (by omega)
    · rw [hc2.1, hc2.2]
      exact ⟨((hint p).2.1 0).trans rfl, ((hint p).2.1 1).trans rfl⟩
    · show d2.E l 0 jj = _
      have hk : parentKeys o2 o3 o4 o5 ⟨l % 15, by omega⟩ = d1.key1 model ⟨l % 15, by omega⟩ := by
        funext r j
        rw [← (hint ⟨l % 15, by omega⟩).2.2 r j]
        fin_cases r <;> fin_cases j <;> rfl
      rw [hE2, hk]
      rfl
  have h := step_correct hs he hP H table coef hcoef
  have hrows : (fun j ↦ ![d1.c0, d1.c1] j * scalarV (coef 1) +
        ∑ b : Fin 15, d1.row model b (by omega) j * scalarV (coef (kSw b)) +
        ∑ i : Fin 15, d1.row model (15 + i) (by omega) j * scalarV (coef (kSw (15 + i))) +
        ∑ l : Fin 225, d2.row model l l.isLt j * scalarV (coef (kLeaf2 l))) = R := by
    funext j
    rw [hR j]
    congr 3
    refine Finset.sum_congr rfl fun b _ ↦ ?_
    rw [(hint b).2.1 j]
  rw [hrows] at h
  exact h

end PeProof
