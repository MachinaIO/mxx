import MxxSecurity.Lemmas

/-!
# Distributions of the samplers

* `evalDist_query`: an oracle query is distributed by its law.
* `query_perm`: relabelling the answer of a query by a map that preserves its law does not
  change the distribution.
-/

namespace MxxSecurity

open MeasureTheory OracleComp MxxRuntime Mxx.Primitives

/-- The raw oracle query of `law`. -/
def queryLaw (law : SamplerLaw) : Comp Int :=
  (OracleSpec.query (spec := samplerSpec) (.law law) : Comp Int)

theorem sample_eq (law : SamplerLaw) :
    sample law = queryLaw law >>= fun x ↦ pure (law.clamp x) := rfl

theorem evalDist_queryLaw (law : SamplerLaw) : 𝒟[queryLaw law] = law.measure := by
  have h : 𝒟[queryLaw law] = _ := evalDist_liftM_query (spec := samplerSpec) (.law law)
  refine h.trans ?_
  exact MeasureTheory.trim_eq_self

/-- A relabelling of query answers that preserves the law preserves the distribution. -/
theorem query_perm {β : Type} (law : SamplerLaw) (φ : Int → Int)
    (hφ : law.measure.map φ = law.measure) (f : Int → Comp β) :
    (queryLaw law >>= f) =ᵈ (queryLaw law >>= fun x ↦ f (φ x)) := by
  letI : MeasurableSpace β := ⊤
  haveI : DiscreteMeasurableSpace β := ⟨fun _ ↦ MeasurableSpace.measurableSet_top⟩
  apply EvalDistEq.of_evalDist_eq
  rw [evalDist_bind_of_discrete, evalDist_bind_of_discrete, evalDist_queryLaw]
  conv_lhs => rw [← hφ]
  exact Measure.bind_map Measurable.of_discrete.aemeasurable Measurable.of_discrete.aemeasurable

end MxxSecurity

namespace MxxSecurity

open MeasureTheory OracleComp MxxRuntime

/-- Continuations that agree on a set of full measure give the same distribution after a query. -/
theorem query_congr_ae {β : Type} (law : SamplerLaw) (support : Set Int)
    (hsupport : law.measure supportᶜ = 0) {f g : Int → Comp β}
    (h : ∀ x ∈ support, f x = g x) : (queryLaw law >>= f) =ᵈ (queryLaw law >>= g) := by
  letI : MeasurableSpace β := ⊤
  haveI : DiscreteMeasurableSpace β := ⟨fun _ ↦ MeasurableSpace.measurableSet_top⟩
  apply EvalDistEq.of_evalDist_eq
  refine evalDist_bind_congr_ae _ f g Measurable.of_discrete Measurable.of_discrete ?_
  rw [evalDist_queryLaw]
  refine measure_mono_null (fun x hx ↦ ?_) hsupport
  intro hmem
  exact hx (show 𝒟[f x] = 𝒟[g x] by rw [h x hmem])

/-- The residues `[0, q)` as integers. -/
abbrev residues (q : Nat) : Finset Int := (Finset.range q).image fun value : Nat ↦ (value : Int)

theorem residues_nonempty {q : Nat} (hq : 0 < q) : (residues q).Nonempty :=
  (Finset.nonempty_range_iff.mpr (by omega)).image _

theorem mem_residues {q : Nat} {x : Int} : x ∈ residues q ↔ 0 ≤ x ∧ x < q := by
  constructor
  · intro hx
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.mp hx
    exact ⟨by omega, by simpa using hv⟩
  · rintro ⟨h0, hq⟩
    exact Finset.mem_image.mpr ⟨x.toNat, Finset.mem_range.mpr (by omega), by omega⟩

theorem residue_measure_eq {q : Nat} (hq : 0 < q) :
    (SamplerLaw.residue q).measure =
      (uniformPMF (residues q) (residues_nonempty hq)).toMeasure := by
  simp only [SamplerLaw.measure, SamplerLaw.pmf, dif_pos hq]

theorem residue_measure_compl {q : Nat} (hq : 0 < q) :
    (SamplerLaw.residue q).measure (↑(residues q))ᶜ = 0 := by
  rw [residue_measure_eq hq, PMF.toMeasure_apply_eq_zero_iff _ MeasurableSet.of_discrete]
  rw [Set.disjoint_compl_right_iff_subset]
  intro x hx
  rw [PMF.mem_support_iff, uniformPMF_apply] at hx
  by_contra hmem
  exact hx (if_neg hmem)

/-- A bijection of the residues preserves the uniform residue law. -/
theorem residue_measure_map {q : Nat} (hq : 0 < q) {φ : Int → Int}
    (hφ : Set.BijOn φ (residues q) (residues q)) :
    (SamplerLaw.residue q).measure.map φ = (SamplerLaw.residue q).measure := by
  rw [residue_measure_eq hq, PMF.toMeasure_map φ _ Measurable.of_discrete]
  congr 1
  ext y
  rw [PMF.map_apply, uniformPMF_apply]
  by_cases hy : y ∈ residues q
  · obtain ⟨x, hx, rfl⟩ := hφ.surjOn hy
    rw [tsum_eq_single x, if_pos rfl, uniformPMF_apply, if_pos (Finset.mem_coe.mp hx), if_pos hy]
    intro z hz
    by_cases hzS : z ∈ residues q
    · rw [if_neg]
      intro heq
      exact hz (hφ.injOn hzS hx heq.symm)
    · rw [uniformPMF_apply, if_neg hzS]
      simp
  · rw [if_neg hy]
    refine ENNReal.tsum_eq_zero.mpr fun z ↦ ?_
    split_ifs with heq
    · rw [uniformPMF_apply, if_neg]
      intro hz
      exact hy (heq ▸ hφ.mapsTo hz)
    · rfl

/-- Relabelling uniform residues by a bijection of the residues preserves the distribution. -/
theorem sampleResidueValue_perm {β : Type} {q : Nat} (hq : 0 < q) {ψ : Int → Int}
    (hψ : Set.BijOn ψ (residues q) (residues q)) (f : Int → Comp β) :
    (sample (.residue q) >>= f) =ᵈ (sample (.residue q) >>= fun x ↦ f (ψ x)) := by
  classical
  let φ : Int → Int := fun x ↦ if x ∈ residues q then ψ x else x
  have hφ : Set.BijOn φ (residues q) (residues q) :=
    hψ.congr fun x hx ↦ by simp only [φ, if_pos (Finset.mem_coe.mp hx)]
  have hclamp : ∀ x ∈ (residues q : Set Int), SamplerLaw.clamp (.residue q) x = x := by
    intro x hx
    have := mem_residues.mp hx
    simp [SamplerLaw.clamp, this.1, this.2]
  simp only [sample_eq, bind_assoc, pure_bind]
  refine (query_perm _ φ (residue_measure_map hq hφ) _).trans ?_
  refine query_congr_ae _ _ (residue_measure_compl hq) fun x hx ↦ ?_
  have hxS : x ∈ residues q := hx
  simp only [φ, if_pos hxS, hclamp x hx, hclamp _ (hψ.mapsTo hxS)]

end MxxSecurity
