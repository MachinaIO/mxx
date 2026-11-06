import ExponentPublic
import ExponentSampling

/-!
# Runtime relations are functional

Every runtime relation of the exported stages determines its output from its inputs (and a tape
sample from the tape). An average-case proof that names the wires of a run as functions of the
tape uses these lemmas to identify the wires of two runs with the same inputs.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime

variable {q p n rows columns : Nat}

theorem blockModSwitchRuns_unique {basis : List Nat} {plaintext : Int}
    {input : ExactMatrix q n rows columns} {x y : ExactMatrix p n rows columns}
    (hx : blockModSwitchRuns basis plaintext input x)
    (hy : blockModSwitchRuns basis plaintext input y) : x = y := by
  funext row column
  rw [hx.2.2.2.2.2.2.2.2.2.2.2 row column, hy.2.2.2.2.2.2.2.2.2.2.2 row column]

theorem modulusReduceRuns_unique {input : ExactMatrix q n rows columns}
    {x y : ExactMatrix p n rows columns} (hx : modulusReduceRuns input x)
    (hy : modulusReduceRuns input y) : x = y :=
  hx.2.2.trans hy.2.2.symm

theorem compactCenteredRebaseRuns_unique {bound : Nat} {input : ExactMatrix q n rows columns}
    {x y : ExactMatrix p n rows columns} (hx : compactCenteredRebaseRuns bound input x)
    (hy : compactCenteredRebaseRuns bound input y) : x = y :=
  hx.2.2.1.2.trans hy.2.2.1.2.symm

theorem ringAutomorphismRuns_unique {index : Int} {input x y : ExactMatrix q n rows columns}
    (hx : ringAutomorphismRuns index input x) (hy : ringAutomorphismRuns index input y) : x = y :=
  hx.2.2.2.trans hy.2.2.2.symm

theorem sliceMatrix_unique {outRows outColumns : Nat} {input : ExactMatrix q n rows columns}
    {rowStart rowEnd columnStart columnEnd : Int} {x y : ExactMatrix q n outRows outColumns}
    (hx : sliceMatrix input rowStart rowEnd columnStart columnEnd x)
    (hy : sliceMatrix input rowStart rowEnd columnStart columnEnd y) : x = y := by
  obtain ⟨h0, -, h2, h3, -, h5, h6, h7, hxv⟩ := hx
  obtain ⟨-, -, -, -, -, -, -, -, hyv⟩ := hy
  funext row column
  simp only [Int.ofNat_eq_natCast] at h2 h5 h6 h7
  have hr := row.isLt
  have hc := column.isLt
  have hRow : rowStart.toNat + row.val < rows := by omega
  have hColumn : columnStart.toNat + column.val < columns := by omega
  rw [hxv row column hRow hColumn, hyv row column hRow hColumn]

theorem concatRows_unique {leftRows rightRows : Nat} {left : ExactMatrix q n leftRows columns}
    {right : ExactMatrix q n rightRows columns} {x y : ExactMatrix q n (leftRows + rightRows) columns}
    (hx : concatRows left right x) (hy : concatRows left right y) : x = y := by
  funext row column
  rw [hx row column, hy row column]

theorem concatColumns_unique {leftColumns rightColumns : Nat}
    {left : ExactMatrix q n rows leftColumns} {right : ExactMatrix q n rows rightColumns}
    {x y : ExactMatrix q n rows (leftColumns + rightColumns)}
    (hx : concatColumns left right x) (hy : concatColumns left right y) : x = y := by
  funext row column
  rw [hx row column, hy row column]

theorem familyGetDynamic_unique {α : Type} {count : Nat} {family : Fin count → α} {index : Int}
    {x y : α} (hx : familyGetDynamic family index x) (hy : familyGetDynamic family index y) :
    x = y := by
  obtain ⟨i, hi, rfl⟩ := hx
  obtain ⟨j, hj, rfl⟩ := hy
  rw [Fin.ext (by omega : i.val = j.val)]

theorem gaussianSampleAt_unique {tape : SampleTape} {site : List Nat} {sigma : Rat} {cutoff : Int}
    {x y : ExactMatrix q n rows columns} (hx : gaussianSampleAt tape site sigma cutoff x)
    (hy : gaussianSampleAt tape site sigma cutoff y) : x = y :=
  hx.2.2.2.2.trans hy.2.2.2.2.symm

theorem uniformIntervalSampleAt_unique {tape : SampleTape} {site : List Nat} {lo hi : Int}
    {x y : ExactMatrix q n rows columns} (hx : uniformIntervalSampleAt tape site lo hi x)
    (hy : uniformIntervalSampleAt tape site lo hi y) : x = y :=
  hx.2.2.trans hy.2.2.symm

/-! ## Outputs of functional relations -/

/-- The output of a relation at `x`, when one exists. -/
noncomputable def theOutput {α β : Type} [Zero β] (R : α → β → Prop) (x : α) : β :=
  open Classical in if h : ∃ y, R x y then h.choose else 0

/-- A functional relation's output is `theOutput`. -/
theorem theOutput_eq {α β : Type} [Zero β] {R : α → β → Prop} {x : α} {y : β}
    (hunique : ∀ y', R x y' → y' = y) (h : R x y) : theOutput R x = y := by
  have hex : ∃ y, R x y := ⟨y, h⟩
  rw [theOutput, dif_pos hex]
  exact hunique _ hex.choose_spec

/-- Two-argument relations, curried. -/
theorem theOutput_eq₂ {α γ β : Type} [Zero β] {R : α → γ → β → Prop} {x : α} {z : γ} {y : β}
    (hunique : ∀ y', R x z y' → y' = y) (h : R x z y) :
    theOutput (fun p : α × γ ↦ R p.1 p.2) (x, z) = y :=
  theOutput_eq (R := fun p : α × γ ↦ R p.1 p.2) hunique h

/-- The centered integer lift of every entry. -/
noncomputable def liftMatrix (x : ExactMatrix q n rows columns) : ErrorMatrix n rows columns :=
  fun row column ↦ centeredIntLift (x row column)

theorem liftMatrix_reduce (hq : 1 < q) (hn : 0 < n) {d : ErrorMatrix n rows columns} {b : Nat}
    (hd : CoeffBound d b) (hb : 2 * b < q) : liftMatrix (reduceMatrix q n rows columns d) = d := by
  funext row column
  exact centeredIntLift_reducePoly hq hn _ fun i ↦ by have := hd row column i; omega

/-! ## Tape samples as functions of the tape -/

/-- A Gaussian tape sample is the reduction of the tape values at its site, which lie within the
cutoff. -/
theorem gaussianSampleAt_explicit (hn : 0 < n) {tape : SampleTape} {site : List Nat} {sigma : Rat}
    {cutoff : Int} {x : ExactMatrix q n rows columns} (h : gaussianSampleAt tape site sigma cutoff x) :
    x = reduceMatrix q n rows columns (tapeWitness tape site (.gaussian sigma cutoff)) ∧
      CoeffBound (tapeWitness (n := n) (rows := rows) (columns := columns) tape site
        (.gaussian sigma cutoff)) cutoff.toNat := by
  obtain ⟨-, -, hbound, -, hout⟩ := h
  refine ⟨hout.trans (tapeMatrix_eq_reduce _ _ _), fun row column k ↦ ?_⟩
  rw [tapeWitness, intPoly_coeff hn]
  have := hbound row column k
  rw [abs_le] at this
  omega

/-- A ternary tape sample is the reduction of the tape values at its site. -/
theorem ternarySampleAt_explicit (hn : 0 < n) {tape : SampleTape} {site : List Nat}
    {x : ExactMatrix q n 1 1} (h : uniformIntervalSampleAt tape site (-1) 1 x) :
    x = (fun _ _ ↦ reducePoly q n (tapeWitness (n := n) (rows := 1) (columns := 1) tape site
        (.interval (-1) 1) 0 0)) ∧
      ∀ k, |(tapeWitness (n := n) (rows := 1) (columns := 1) tape site (.interval (-1) 1) 0 0).coeff k|
        ≤ 1 := by
  obtain ⟨-, hbound, hout⟩ := h
  refine ⟨?_, fun k ↦ ?_⟩
  · rw [hout, tapeMatrix_eq_reduce]
    funext i j
    rw [Subsingleton.elim i 0, Subsingleton.elim j 0]
    rfl
  · rw [tapeWitness, intPoly_coeff hn, abs_le]
    exact hbound 0 0 k

/-- The tape values at `site` clamped to `[lo, hi]`: a sample of the model, which every run
reads unchanged. -/
noncomputable def clampedWitness {rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (law : SamplerLaw) (lo hi : Int) : ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun k ↦ max lo (min hi (tape ⟨site, row.val, column.val, k.val, law⟩))

theorem clampedWitness_eq_of_gaussian (hn : 0 < n) {tape : SampleTape} {site : List Nat}
    {sigma : Rat} {cutoff : Int} {x : ExactMatrix q n rows columns}
    (h : gaussianSampleAt tape site sigma cutoff x) :
    (clampedWitness tape site (.gaussian sigma cutoff) (-cutoff) cutoff : ErrorMatrix n rows columns) =
      tapeWitness tape site (.gaussian sigma cutoff) := by
  obtain ⟨-, -, hbound, -, -⟩ := h
  funext row column
  apply negacyclic_ext hn
  intro k
  rw [clampedWitness, tapeWitness, intPoly_coeff hn, intPoly_coeff hn]
  have := hbound row column k
  rw [abs_le] at this
  omega

theorem clampedWitness_eq_of_interval (hn : 0 < n) {tape : SampleTape} {site : List Nat}
    {lo hi : Int} {x : ExactMatrix q n rows columns} (h : uniformIntervalSampleAt tape site lo hi x) :
    (clampedWitness tape site (.interval lo hi) lo hi : ErrorMatrix n rows columns) =
      tapeWitness tape site (.interval lo hi) := by
  obtain ⟨-, hbound, -⟩ := h
  funext row column
  apply negacyclic_ext hn
  intro k
  rw [clampedWitness, tapeWitness, intPoly_coeff hn, intPoly_coeff hn]
  have := hbound row column k
  omega

theorem clampedWitness_bound (hn : 0 < n) {rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (law : SamplerLaw) {c : Nat} :
    CoeffBound (clampedWitness tape site law (-(c : Int)) c : ErrorMatrix n rows columns) c := by
  intro row column k
  rw [clampedWitness, intPoly_coeff hn]
  omega

/-- Samples within their bound are unchanged by the clamp. -/
theorem clampedWitness_eq_of_bound (hn : 0 < n) {rows columns : Nat} {tape : SampleTape}
    {site : List Nat} {law : SamplerLaw} {c : Nat}
    (h : CoeffBound (tapeWitness (n := n) (rows := rows) (columns := columns) tape site law) c) :
    (clampedWitness tape site law (-(c : Int)) c : ErrorMatrix n rows columns) =
      tapeWitness tape site law := by
  funext row column
  apply negacyclic_ext hn
  intro k
  have hb := h row column k
  rw [tapeWitness, intPoly_coeff hn] at hb
  rw [clampedWitness, tapeWitness, intPoly_coeff hn, intPoly_coeff hn]
  omega

/-! ## Restricted tapes -/

/-- The tape that keeps the coordinates in `S` and zeroes the others. A function of the restricted
tape depends only on `S`. -/
def restrictTape (S : Finset SampleKey) (tape : SampleTape) : SampleTape :=
  fun key ↦ if key ∈ S then tape key else 0

theorem restrictTape_dependsOn {β : Type} (S : Finset SampleKey) (f : SampleTape → β) :
    DependsOn (fun tape ↦ f (restrictTape S tape)) (S : Set SampleKey) := by
  intro t t' h
  have : restrictTape S t = restrictTape S t' := by
    funext key
    unfold restrictTape
    split
    · exact h key (by simpa)
    · rfl
  simp only [this]

/-- The keys of a matrix sampled at `site` under `law`. -/
def SiteKeysIn (S : Finset SampleKey) (site : List Nat) (law : SamplerLaw) (rows columns n : Nat) :
    Prop :=
  ∀ (row : Fin rows) (column : Fin columns) (k : Fin n), (⟨site, row.val, column.val, k.val, law⟩ :
    SampleKey) ∈ S

theorem gaussianSampleAt_restrict {S : Finset SampleKey} {tape : SampleTape} {site : List Nat}
    {sigma : Rat} {cutoff : Int} {x : ExactMatrix q n rows columns}
    (hS : SiteKeysIn S site (.gaussian sigma cutoff) rows columns n) :
    gaussianSampleAt (restrictTape S tape) site sigma cutoff x ↔
      gaussianSampleAt tape site sigma cutoff x := by
  have hk : ∀ (row : Fin rows) (column : Fin columns) (k : Fin n),
      restrictTape S tape ⟨site, row.val, column.val, k.val, .gaussian sigma cutoff⟩ =
        tape ⟨site, row.val, column.val, k.val, .gaussian sigma cutoff⟩ := fun row column k ↦ by
    simp [restrictTape, hS row column k]
  unfold gaussianSampleAt
  rw [tapeMatrix_congr (q := q) hk]
  simp only [hk]

theorem uniformIntervalSampleAt_restrict {S : Finset SampleKey} {tape : SampleTape}
    {site : List Nat} {lo hi : Int} {x : ExactMatrix q n rows columns}
    (hS : SiteKeysIn S site (.interval lo hi) rows columns n) :
    uniformIntervalSampleAt (restrictTape S tape) site lo hi x ↔
      uniformIntervalSampleAt tape site lo hi x := by
  have hk : ∀ (row : Fin rows) (column : Fin columns) (k : Fin n),
      restrictTape S tape ⟨site, row.val, column.val, k.val, .interval lo hi⟩ =
        tape ⟨site, row.val, column.val, k.val, .interval lo hi⟩ := fun row column k ↦ by
    simp [restrictTape, hS row column k]
  unfold uniformIntervalSampleAt
  rw [tapeMatrix_congr (q := q) hk]
  simp only [hk]

/-- The sites of length at most `L` whose entries are below `M`. -/
def boundedSites (L M : Nat) : Finset (List Nat) :=
  (Finset.range (L + 1)).biUnion fun l ↦
    (Fintype.piFinset fun _ : Fin l ↦ Finset.range M).image List.ofFn

theorem mem_boundedSites {L M : Nat} {site : List Nat} (hL : site.length ≤ L)
    (hM : ∀ x ∈ site, x < M) : site ∈ boundedSites L M := by
  unfold boundedSites
  refine Finset.mem_biUnion.mpr ⟨site.length, Finset.mem_range.mpr (by omega), ?_⟩
  refine Finset.mem_image.mpr ⟨fun i ↦ site.get i, ?_, List.ofFn_get site⟩
  exact Fintype.mem_piFinset.mpr fun i ↦ Finset.mem_range.mpr (hM _ (List.get_mem site i))

/-- The keys whose site starts with one of `heads`, of length at most `L` with entries below `M`,
at rows, columns and coefficients below `R`, `C`, `N`, under one of `laws`. -/
def sampleUniverse (heads : List Nat) (L M R C N : Nat) (laws : List SamplerLaw) :
    Finset SampleKey :=
  ((((boundedSites L M ×ˢ Finset.range R) ×ˢ Finset.range C) ×ˢ Finset.range N) ×ˢ
      laws.toFinset).image (fun x ↦ (⟨x.1.1.1.1, x.1.1.1.2, x.1.1.2, x.1.2, x.2⟩ : SampleKey))
    |>.filter fun key ↦ key.site.head? ∈ heads.map some

theorem siteKeysIn_sampleUniverse {heads : List Nat} {L M R C N : Nat} {laws : List SamplerLaw}
    {site : List Nat} {law : SamplerLaw} {rows columns n : Nat}
    (hhead : site.head? ∈ heads.map some) (hL : site.length ≤ L) (hM : ∀ x ∈ site, x < M)
    (hR : rows ≤ R) (hC : columns ≤ C) (hN : n ≤ N) (hlaw : law ∈ laws) :
    SiteKeysIn (sampleUniverse heads L M R C N laws) site law rows columns n := by
  intro row column k
  unfold sampleUniverse
  refine Finset.mem_filter.mpr ⟨Finset.mem_image.mpr ⟨⟨⟨⟨⟨site, row.val⟩, column.val⟩, k.val⟩, law⟩,
    ?_, rfl⟩, hhead⟩
  simp only [Finset.mem_product, Finset.mem_range, List.mem_toFinset]
  exact ⟨⟨⟨⟨mem_boundedSites hL hM, by omega⟩, by omega⟩, by omega⟩, hlaw⟩

/-- The head of the site of a key of a universe given by its defining equation. Stating the
universe through an equation keeps its concrete value, which `whnf` would enumerate, out of the
elaborated terms. -/
theorem head_of_mem_eq {S : Finset SampleKey} {heads : List Nat} {L M R C N : Nat}
    {laws : List SamplerLaw} (hS : S = sampleUniverse heads L M R C N laws) {key : SampleKey}
    (h : key ∈ S) : key.site.head? ∈ heads.map some := by
  subst hS
  exact (Finset.mem_filter.mp h).2

theorem siteKeysIn_of_eq {S : Finset SampleKey} {heads : List Nat} {L M R C N : Nat}
    {laws : List SamplerLaw} (hS : S = sampleUniverse heads L M R C N laws) {site : List Nat}
    {law : SamplerLaw} {rows columns n : Nat} (hhead : site.head? ∈ heads.map some)
    (hL : site.length ≤ L) (hM : ∀ x ∈ site, x < M) (hR : rows ≤ R) (hC : columns ≤ C) (hN : n ≤ N)
    (hlaw : law ∈ laws) : SiteKeysIn S site law rows columns n := by
  subst hS
  exact siteKeysIn_sampleUniverse hhead hL hM hR hC hN hlaw

/-- A path under one of `heads`, with room for `d` more entries below the length `L`, and entries
below `M`. -/
def GoodPath (heads : List Nat) (L M d : Nat) (path : List Nat) : Prop :=
  path.head? ∈ heads.map some ∧ path.length + d ≤ L ∧ ∀ x ∈ path, x < M

theorem GoodPath.append {heads : List Nat} {L M d d' : Nat} {path suffix : List Nat}
    (hp : GoodPath heads L M d path) (hlen : suffix.length + d' ≤ d)
    (hM : ∀ x ∈ suffix, x < M) : GoodPath heads L M d' (path ++ suffix) := by
  obtain ⟨hh, hl, hm⟩ := hp
  have hne : path ≠ [] := by
    rintro rfl
    simp at hh
  refine ⟨?_, by simp; omega, fun x hx ↦ ?_⟩
  · rw [List.head?_append_of_ne_nil _ hne]
    exact hh
  · rcases List.mem_append.mp hx with h | h
    exacts [hm x h, hM x h]

theorem GoodPath.siteKeysIn {heads : List Nat} {L M R C N : Nat} {laws : List SamplerLaw}
    {d : Nat} {path suffix : List Nat} {law : SamplerLaw} {rows columns n : Nat}
    (hp : GoodPath heads L M d path) (hlen : suffix.length ≤ d) (hM : ∀ x ∈ suffix, x < M)
    (hR : rows ≤ R) (hC : columns ≤ C) (hN : n ≤ N) (hlaw : law ∈ laws) :
    SiteKeysIn (sampleUniverse heads L M R C N laws) (path ++ suffix) law rows columns n := by
  obtain ⟨hh, hl, hm⟩ := hp.append (d' := 0) (by omega) hM
  exact siteKeysIn_sampleUniverse hh (by omega) hm hR hC hN hlaw

end MxxExponentLut
