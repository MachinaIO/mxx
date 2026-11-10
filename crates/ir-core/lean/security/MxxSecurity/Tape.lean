import MxxSecurity.Distribution
import RuntimeProbability
import Mathlib.Probability.ProductMeasure

/-!
# Sampled computations as functions of a tape

A tape assigns an integer to every coordinate `i : ι`, and the tape measure draws the coordinates
independently, coordinate `i` by `μ i`. `Represents μ c S R` says that the computation `c` is
distributed as `R` of a random tape and that `R` reads only the coordinates `S`. Sampling one
coefficient reads one coordinate, and a computation that runs `c` and then a continuation reading
coordinates disjoint from those of `c` reads the union. A Boolean computation represented by `R`
returns `true` with the tape measure of `{ω | R ω = true}`, so tail bounds over independent
coordinates (`RuntimeProbability`, `RuntimeSubGaussian`) bound its probability.
-/

namespace MxxSecurity

set_option linter.unusedSectionVars false
set_option linter.style.haveILetI false

open MeasureTheory OracleComp MxxRuntime Mxx.Primitives Function

variable {ι : Type} [DecidableEq ι] (μ : ι → Measure ℤ) [∀ i, IsProbabilityMeasure (μ i)]

/-- Splitting a tape at `S`: the coordinates of `S` from the first tape, the others from the
second. -/
theorem measurable_piecewise_pair (S : Finset ι) :
    Measurable fun p : (ι → ℤ) × (ι → ℤ) ↦ S.piecewise p.1 p.2 := by
  refine Measurable.of_eval fun i ↦ ?_
  by_cases h : i ∈ S
  · simp only [Finset.piecewise_eq_of_mem _ _ _ h]
    exact (measurable_pi_apply i).comp measurable_fst
  · simp only [Finset.piecewise_eq_of_notMem _ _ _ h]
    exact (measurable_pi_apply i).comp measurable_snd

/-- Drawing the coordinates of `S` and the others from two independent tapes gives the tape
measure. -/
theorem infinitePi_eq_map_piecewise (S : Finset ι) :
    Measure.infinitePi μ =
      ((Measure.infinitePi μ).prod (Measure.infinitePi μ)).map
        fun p ↦ S.piecewise p.1 p.2 := by
  symm
  apply Measure.eq_infinitePi
  intro s t ht
  rw [Measure.map_apply (measurable_piecewise_pair S)
    (MeasurableSet.pi s.countable_toSet fun i _ ↦ ht i)]
  have hpre : (fun p : (ι → ℤ) × (ι → ℤ) ↦ S.piecewise p.1 p.2) ⁻¹' Set.pi (↑s) t =
      Set.pi (↑(s.filter (· ∈ S))) t ×ˢ Set.pi (↑(s.filter (· ∉ S))) t := by
    ext ⟨x, y⟩
    simp only [Set.mem_preimage, Set.mem_pi, Finset.mem_coe, Set.mem_prod, Finset.mem_filter]
    constructor
    · intro h
      refine ⟨fun i ⟨hi, hS⟩ ↦ ?_, fun i ⟨hi, hS⟩ ↦ ?_⟩
      · simpa [Finset.piecewise_eq_of_mem _ _ _ hS] using h i hi
      · simpa [Finset.piecewise_eq_of_notMem _ _ _ hS] using h i hi
    · rintro ⟨hx, hy⟩ i hi
      by_cases hS : i ∈ S
      · simpa [Finset.piecewise_eq_of_mem _ _ _ hS] using hx i ⟨hi, hS⟩
      · simpa [Finset.piecewise_eq_of_notMem _ _ _ hS] using hy i ⟨hi, hS⟩
  rw [hpre, Measure.prod_prod, Measure.infinitePi_pi _ fun i _ ↦ ht i,
    Measure.infinitePi_pi _ fun i _ ↦ ht i, Finset.prod_filter_mul_prod_filter_not]

/-- `c` is distributed as `R` of a random tape, and `R` reads only the coordinates `S`. -/
def Represents {α : Type} (c : Comp α) (S : Finset ι) (R : (ι → ℤ) → α) : Prop :=
  DependsOn R S ∧
    @evalDist Comp _ α ⊤ c = @Measure.map _ _ _ ⊤ R (Measure.infinitePi μ)

variable {μ}

theorem Represents.dependsOn {α : Type} {c : Comp α} {S : Finset ι} {R : (ι → ℤ) → α}
    (h : Represents μ c S R) : DependsOn R S := h.1

theorem Represents.measurable {α : Type} {c : Comp α} {S : Finset ι} {R : (ι → ℤ) → α}
    (h : Represents μ c S R) : @Measurable _ _ _ ⊤ R :=
  @measurable_of_dependsOn ι _ α ⊤ R S h.1

theorem Represents.mono {α : Type} {c : Comp α} {S T : Finset ι} {R : (ι → ℤ) → α}
    (h : Represents μ c S R) (hST : S ⊆ T) : Represents μ c T R :=
  ⟨h.1.mono (by simpa using hST), h.2⟩

theorem Represents.congr {α : Type} {c c' : Comp α} {S : Finset ι} {R : (ι → ℤ) → α}
    (h : Represents μ c S R) (hc : c =ᵈ c') : Represents μ c' S R := by
  refine ⟨h.1, ?_⟩
  letI : MeasurableSpace α := ⊤
  rw [← EvalDistEq.evalDist_eq hc]
  exact h.2

theorem Represents.congr_fun {α : Type} {c : Comp α} {S : Finset ι} {R R' : (ι → ℤ) → α}
    (h : Represents μ c S R) (hR : ∀ ω, R ω = R' ω) : Represents μ c S R' := by
  have : R = R' := funext hR
  subst this
  exact h

theorem represents_pure {α : Type} (x : α) : Represents μ (pure x : Comp α) ∅ fun _ ↦ x := by
  refine ⟨fun _ _ _ ↦ rfl, ?_⟩
  letI : MeasurableSpace α := ⊤
  rw [evalDist_pure, Measure.map_const, measure_univ, one_smul]

theorem represents_bind {α β : Type} {c : Comp α} {k : α → Comp β} {S T : Finset ι}
    {R : (ι → ℤ) → α} {R' : α → (ι → ℤ) → β} (hc : Represents μ c S R)
    (hk : ∀ x, Represents μ (k x) T (R' x)) (hST : Disjoint S T) :
    Represents μ (c >>= k) (S ∪ T) fun ω ↦ R' (R ω) ω := by
  letI : MeasurableSpace α := ⊤
  letI : MeasurableSpace β := ⊤
  haveI : DiscreteMeasurableSpace α := ⟨fun _ ↦ MeasurableSpace.measurableSet_top⟩
  have hdep : DependsOn (fun ω ↦ R' (R ω) ω) (↑(S ∪ T) : Set ι) := by
    intro ω ω' h
    have hS : R ω = R ω' := hc.1 fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi])
    show R' (R ω) ω = R' (R ω') ω'
    rw [hS]
    exact (hk _).1 fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi])
  refine ⟨hdep, ?_⟩
  have hmeas : Measurable fun ω ↦ R' (R ω) ω := measurable_of_dependsOn (S ∪ T) hdep
  rw [evalDist_bind_of_discrete, hc.2]
  simp_rw [fun x ↦ (hk x).2]
  rw [Measure.bind_map hc.measurable.aemeasurable Measurable.of_discrete.aemeasurable]
  ext A hA
  have hg : AEMeasurable (fun a ↦ Measure.map (R' (R a)) (Measure.infinitePi μ))
      (Measure.infinitePi μ) :=
    ((Measurable.of_discrete (f := fun x : α ↦ Measure.map (R' x) (Measure.infinitePi μ))).comp
      hc.measurable).aemeasurable
  rw [Measure.bind_apply hA hg, Measure.map_apply hmeas hA]
  conv_rhs => rw [infinitePi_eq_map_piecewise μ S]
  have hset : MeasurableSet ((fun p : (ι → ℤ) × (ι → ℤ) ↦ S.piecewise p.1 p.2) ⁻¹'
      ((fun ω ↦ R' (R ω) ω) ⁻¹' A)) :=
    (measurable_piecewise_pair S) (hmeas hA)
  rw [Measure.map_apply (measurable_piecewise_pair S) (hmeas hA), Measure.prod_apply hset]
  refine lintegral_congr fun ω ↦ ?_
  rw [Measure.map_apply (hk _).measurable hA]
  congr 1
  ext ω'
  simp only [Set.mem_preimage]
  have hR : R (S.piecewise ω ω') = R ω :=
    hc.1 fun i hi ↦ Finset.piecewise_eq_of_mem _ _ _ (Finset.mem_coe.mp hi)
  have hR' : R' (R ω) (S.piecewise ω ω') = R' (R ω) ω' :=
    (hk _).1 fun i hi ↦ Finset.piecewise_eq_of_notMem _ _ _
      (Finset.disjoint_right.mp hST (Finset.mem_coe.mp hi))
  rw [hR, hR']

theorem represents_bind_pure {α β : Type} {c : Comp α} {S : Finset ι} {R : (ι → ℤ) → α}
    (hc : Represents μ c S R) (f : α → β) :
    Represents μ (c >>= fun x ↦ pure (f x)) S fun ω ↦ f (R ω) := by
  have h := represents_bind hc (fun x ↦ represents_pure (μ := μ) (f x))
    (Finset.disjoint_empty_right S)
  rw [Finset.union_empty] at h
  exact h

theorem represents_map {α β : Type} {c : Comp α} {S : Finset ι} {R : (ι → ℤ) → α}
    (hc : Represents μ c S R) (f : α → β) : Represents μ (f <$> c) S fun ω ↦ f (R ω) := by
  rw [map_eq_bind_pure_comp]
  exact represents_bind_pure hc f

theorem represents_query (k : ι) (law : SamplerLaw) (hk : μ k = law.measure) :
    Represents μ (queryLaw law) {k} fun ω ↦ ω k := by
  refine ⟨fun ω ω' h ↦ h k (by simp), ?_⟩
  have hq : @evalDist Comp _ Int ⊤ (queryLaw law) = law.measure := evalDist_queryLaw law
  rw [hq, ← hk]
  exact (Measure.infinitePi_map_eval μ k).symm

theorem represents_sample (k : ι) (law : SamplerLaw) (hk : μ k = law.measure) :
    Represents μ (sample law) {k} fun ω ↦ law.clamp (ω k) := by
  rw [sample_eq]
  exact represents_bind_pure (represents_query k law hk) law.clamp

/-- `count` independent draws from `law`, draw `j` reading coordinate `key j`. -/
theorem represents_sampleFamily (law : SamplerLaw) :
    (count : Nat) → (key : Fin count → ι) → Injective key → (∀ j, μ (key j) = law.measure) →
      Represents μ (sampleFamily law count) (Finset.univ.image key)
        fun ω j ↦ law.clamp (ω (key j))
  | 0, key, _, _ => by
    refine (represents_pure (μ := μ) Fin.elim0).mono (Finset.empty_subset _) |>.congr_fun ?_
    intro ω
    funext j
    exact j.elim0
  | count + 1, key, hinj, hlaw => by
    have hhead := represents_sample (μ := μ) (key 0) law (hlaw 0)
    have htail := represents_sampleFamily law count (fun j ↦ key j.succ)
      (hinj.comp (Fin.succ_injective _)) fun j ↦ hlaw j.succ
    have hdisj : Disjoint ({key 0} : Finset ι) (Finset.univ.image fun j : Fin count ↦ key j.succ) := by
      rw [Finset.disjoint_singleton_left, Finset.mem_image]
      rintro ⟨j, -, hj⟩
      exact Fin.succ_ne_zero j (hinj hj)
    have hk := represents_bind hhead
      (fun head ↦ represents_bind_pure htail fun tail ↦ (Fin.cons head tail : Fin (count + 1) → ℤ))
      hdisj
    refine (hk.mono ?_).congr_fun ?_
    · intro i hi
      simp only [Finset.mem_union, Finset.mem_singleton, Finset.mem_image, Finset.mem_univ,
        true_and] at hi ⊢
      rcases hi with rfl | ⟨j, rfl⟩
      · exact ⟨0, rfl⟩
      · exact ⟨j.succ, rfl⟩
    · intro ω
      funext j
      refine Fin.cases ?_ (fun j ↦ ?_) j <;> simp

/-- A sampled matrix, coefficient `j` of the row-major flattening reading coordinate `key j`. -/
theorem represents_sampleMatrix (law : SamplerLaw) (q n rows columns : Nat)
    (key : Fin (rows * columns * n) → ι) (hinj : Injective key)
    (hlaw : ∀ j, μ (key j) = law.measure) :
    Represents μ (sampleMatrix law q n rows columns) (Finset.univ.image key) fun ω row column ↦
      polynomialOfCoefficients fun coefficient ↦
        law.clamp (ω (key (finProdFinEquiv (finProdFinEquiv (row, column), coefficient)))) :=
  represents_bind_pure (represents_sampleFamily law _ key hinj hlaw) _

/-- A Boolean computation returns `true` with the tape measure of the tapes `R` maps to `true`. -/
theorem Represents.probTrue_eq {c : Comp Bool} {S : Finset ι} {R : (ι → ℤ) → Bool}
    (h : Represents μ c S R) : probTrue c = Measure.infinitePi μ {ω | R ω = true} := by
  unfold probTrue
  have h2 : 𝒟[c] = (Measure.infinitePi μ).map R := h.2
  rw [h2, Measure.map_apply h.measurable (measurableSet_singleton true)]
  rfl

end MxxSecurity
