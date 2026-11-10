import Scheme
import MxxSecurity.Tape

/-!
# The TFHE tape

Every sampler of one run of `evaluationFails` draws its coefficients from its own site of a tape
`ω : Key → ℤ`: coefficient `j` of the row-major flattening of the matrix drawn at site `s` is
`ω (s, j)`. The sites are, in order, those of key generation (`0`–`14`), of the first encryption
(`15`–`17`) and of the second (`18`–`20`). `smp` reads a sampled matrix from the tape, and
`represents_smp_bind` lets a sampled computation draw one more matrix.
-/

namespace MxxFheTfhe

open MeasureTheory Mxx.Primitives MxxRuntime MxxSecurity

/-- A tape coordinate: a sampler site and a coefficient index. -/
abbrev Key := Nat × Nat

abbrev Q : Nat := 5234636801
abbrev N : Nat := 1024
abbrev lweN : Nat := 720
abbrev q : Nat := 4294967296
abbrev Qmask : Nat := 5846006549229237458670166626046031780481016238081

abbrev bitLaw : SamplerLaw := .interval 0 1
abbrev ringLaw : SamplerLaw := .gaussian (156 / 1) 2496
abbrev lweLaw : SamplerLaw := .gaussian (32768 / 1) 524288

/-- The law of each site. -/
def lawOf : Nat → SamplerLaw
  | 0 => .residue Q
  | 1 => bitLaw
  | 2 => bitLaw
  | 3 => ringLaw
  | 4 => .residue Qmask
  | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 => lweLaw
  | 13 => .residue Q
  | 14 => ringLaw
  | 15 => bitLaw
  | 16 | 17 => ringLaw
  | 18 => bitLaw
  | 19 | 20 => ringLaw
  | _ => .residue 1

/-- The tape measure. -/
noncomputable abbrev μ : Key → Measure ℤ := fun k ↦ (lawOf k.1).measure

/-- The coordinates of site `s` with `m` coefficients. -/
irreducible_def siteSet (s m : Nat) : Finset Key := Finset.univ.image fun j : Fin m ↦ (s, j.val)

theorem mem_siteSet {s m : Nat} {p : Key} (h : p ∈ siteSet s m) : p.1 = s := by
  rw [siteSet_def] at h
  obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp h
  rfl

theorem sites_ne_empty {s : Nat} : ∀ p ∈ (∅ : Finset Key), p.1 ≠ s := by simp

theorem sites_ne_union {s : Nat} {A B : Finset Key} (hA : ∀ p ∈ A, p.1 ≠ s)
    (hB : ∀ p ∈ B, p.1 ≠ s) : ∀ p ∈ A ∪ B, p.1 ≠ s := by
  intro p hp
  rcases Finset.mem_union.mp hp with h | h
  · exact hA p h
  · exact hB p h

theorem sites_ne_siteSet {s t m : Nat} (h : t ≠ s) : ∀ p ∈ siteSet t m, p.1 ≠ s := by
  intro p hp
  rw [mem_siteSet hp]
  exact h

/-- Discharge the disjointness side goals of `represents_smp_bind`. -/
macro "sites_ne" : tactic =>
  `(tactic| repeat (first | exact sites_ne_siteSet (by decide) | exact sites_ne_empty |
      refine sites_ne_union ?_ ?_))

/-- A bound on the number of coefficients of one site. -/
abbrev siteBound : Nat := 16777216

/-- The coordinates of the sites `[lo, hi)`. -/
irreducible_def block (lo hi : Nat) : Finset Key :=
  (Finset.range (hi - lo)).biUnion fun t ↦ siteSet (lo + t) siteBound

theorem mem_block {lo hi : Nat} {p : Key} (h : p ∈ block lo hi) : lo ≤ p.1 ∧ p.1 < hi := by
  rw [block_def] at h
  obtain ⟨t, ht, hp⟩ := Finset.mem_biUnion.mp h
  rw [mem_siteSet hp]
  have := Finset.mem_range.mp ht
  omega

theorem siteSet_subset_block {s m lo hi : Nat} (hs : lo ≤ s ∧ s < hi) (hm : m ≤ siteBound) :
    siteSet s m ⊆ block lo hi := by
  intro p hp
  rw [block_def]
  refine Finset.mem_biUnion.mpr ⟨s - lo, Finset.mem_range.mpr (by omega), ?_⟩
  rw [siteSet_def] at hp ⊢
  obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp hp
  exact Finset.mem_image.mpr ⟨⟨j.val, by omega⟩, Finset.mem_univ _, by simp; omega⟩

/-- Discharge `S ⊆ block lo hi` for a union of sites. -/
macro "sites_sub" : tactic =>
  `(tactic| repeat (first | exact Finset.empty_subset _ |
      exact siteSet_subset_block (by decide) (by decide) | refine Finset.union_subset ?_ ?_))

theorem block_disjoint {a b c d : Nat} (h : b ≤ c) : Disjoint (block a b) (block c d) := by
  rw [Finset.disjoint_left]
  intro p h1 h2
  have := mem_block h1
  have := mem_block h2
  omega

/-- The matrix drawn from `law` at site `s`. -/
noncomputable def smp {q n rows columns : Nat} (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) :
    ExactMatrix q n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun coefficient ↦
    law.clamp (ω (s, (finProdFinEquiv (finProdFinEquiv (row, column), coefficient)).val))

theorem represents_smp {q n rows columns : Nat} (s : Nat) {law : SamplerLaw} (hs : lawOf s = law) :
    Represents μ (sampleMatrix law q n rows columns) (siteSet s (rows * columns * n))
      (smp law s) := by
  rw [siteSet_def]
  refine represents_sampleMatrix (μ := μ) law q n rows columns (fun j ↦ (s, j.val)) ?_ ?_
  · intro i j h
    exact Fin.ext (Prod.ext_iff.mp h).2
  · intro j
    show (lawOf s).measure = law.measure
    rw [hs]

/-- Draw one more matrix at site `s`, which the rest of the computation does not read. -/
theorem represents_smp_bind {α : Type} {q n rows columns : Nat} (s : Nat) {law : SamplerLaw}
    (hs : lawOf s = law) {k : ExactMatrix q n rows columns → Comp α} {T : Finset Key}
    {R : ExactMatrix q n rows columns → (Key → ℤ) → α} (hk : ∀ x, Represents μ (k x) T (R x))
    (hT : ∀ p ∈ T, p.1 ≠ s) :
    Represents μ (sampleMatrix law q n rows columns >>= k) (siteSet s (rows * columns * n) ∪ T)
      fun ω ↦ R (smp law s ω) ω := by
  refine represents_bind (represents_smp s hs) hk ?_
  rw [Finset.disjoint_left]
  intro p hp hpT
  exact hT p hpT (mem_siteSet hp)

end MxxFheTfhe
