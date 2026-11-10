import MxxRuntime
import VCVio.OracleComp.EvalDist.AnswerMeasure

/-!
# Sampled computations

The algorithms of a generated scheme, the challengers of the hardness assumptions, and the
adversaries are VCVio oracle computations over `samplerSpec`. A query names a sampler law, and its
answer is one coefficient drawn from that law, distributed by `MxxRuntime.SamplerLaw.measure`, the
same law as the sampling tapes of the generated correctness claims; or it names a preimage
request, and its answer is the coefficients of a preimage drawn by the ideal preimage sampler.
Every query is an independent ideal sample (assumption R1 of `LEAN-SPEC.md`): for the preimage
sampler, the runtime's Gaussian sampler of Micciancio and Peikert is trusted to draw this law.
-/

namespace MxxSecurity

open MeasureTheory MxxRuntime Mxx.Primitives OracleSpec

/-- A request to the ideal preimage sampler: a preimage `x` of `target` under `publicMatrix`
(`publicMatrix * x = target`) whose coefficients lie in `[-bound, bound]`, drawn by the discrete
Gaussian of width `sigma` on those preimages. -/
structure PreimageRequest where
  q : Nat
  n : Nat
  rows : Nat
  inner : Nat
  columns : Nat
  publicMatrix : ExactMatrix q n rows inner
  target : ExactMatrix q n rows columns
  sigma : Rat
  bound : Nat

/-- The coefficients of a preimage, entry by entry. -/
abbrev PreimageRequest.Coefficients (r : PreimageRequest) : Type :=
  Fin r.inner → Fin r.columns → Fin r.n → Int

/-- The matrix of preimage coefficients. -/
noncomputable def PreimageRequest.matrixOf (r : PreimageRequest) (c : r.Coefficients) :
    ExactMatrix r.q r.n r.inner r.columns :=
  fun i j ↦ polynomialOfCoefficients (c i j)

/-- The preimages with coefficients in `[-bound, bound]`. -/
noncomputable def PreimageRequest.preimages (r : PreimageRequest) : Finset r.Coefficients := by
  classical
  exact (Fintype.piFinset fun _ ↦ Fintype.piFinset fun _ ↦ Fintype.piFinset fun _ ↦
    Finset.Icc (-(r.bound : Int)) r.bound).filter
      fun c ↦ r.publicMatrix * r.matrixOf c = r.target

/-- A bounded preimage is a preimage with coefficients in `[-bound, bound]`. -/
theorem PreimageRequest.mem_preimages {r : PreimageRequest} {c : r.Coefficients}
    (h : c ∈ r.preimages) :
    r.publicMatrix * r.matrixOf c = r.target ∧ ∀ i j k, |c i j k| ≤ r.bound := by
  classical
  simp only [PreimageRequest.preimages, Finset.mem_filter, Fintype.mem_piFinset,
    Finset.mem_Icc] at h
  exact ⟨h.2, fun i j k ↦ abs_le.mpr (h.1 i j k)⟩

/-- The unnormalized Gaussian mass `exp(-‖c‖² / (2 sigma²))` of a coefficient array. -/
noncomputable def PreimageRequest.mass (r : PreimageRequest) (c : r.Coefficients) : ENNReal :=
  ENNReal.ofReal (Real.exp (-(∑ i, ∑ j, ∑ k, ((c i j k : Int) : ℝ) ^ 2) / (2 * (r.sigma : ℝ) ^ 2)))

theorem PreimageRequest.tsum_mass (r : PreimageRequest) :
    ∑' c, (if c ∈ r.preimages then r.mass c else 0) = ∑ c ∈ r.preimages, r.mass c := by
  rw [tsum_eq_sum (s := r.preimages) fun c hc ↦ if_neg hc]
  exact Finset.sum_congr rfl fun c hc ↦ if_pos hc

/-- The ideal preimage sampler: the discrete Gaussian of width `sigma` on the bounded preimages,
or the zero array when there is none. -/
noncomputable def PreimageRequest.pmf (r : PreimageRequest) : PMF r.Coefficients :=
  if h : r.preimages.Nonempty then
    PMF.normalize (fun c ↦ if c ∈ r.preimages then r.mass c else 0)
      (by
        rw [r.tsum_mass]
        obtain ⟨c, hc⟩ := h
        refine ne_of_gt (lt_of_lt_of_le ?_ (Finset.single_le_sum (fun _ _ ↦ bot_le) hc))
        exact ENNReal.ofReal_pos.mpr (Real.exp_pos _))
      (by
        rw [r.tsum_mass]
        exact ENNReal.sum_ne_top.mpr fun _ _ ↦ ENNReal.ofReal_ne_top)
  else PMF.pure fun _ _ _ ↦ 0

/-- A query to the samplers: one coefficient of a sampler law, or a preimage. -/
inductive SamplerQuery where
  | law (law : SamplerLaw)
  | preimage (request : PreimageRequest)

/-- The answer of a query. -/
@[reducible] def SamplerQuery.Answer : SamplerQuery → Type
  | .law _ => Int
  | .preimage request => request.Coefficients

/-- The samplers as one oracle family. -/
abbrev samplerSpec : OracleSpec SamplerQuery := SamplerQuery.Answer

instance (law : SamplerLaw) : IsProbabilityMeasure law.measure := by
  unfold SamplerLaw.measure
  infer_instance

/-- The law of the answer of a query, on the discrete σ-algebra. -/
noncomputable def SamplerQuery.measure : (query : SamplerQuery) → @Measure query.Answer ⊤
  | .law l => (l.measure : Measure Int)
  | .preimage request => @PMF.toMeasure _ ⊤ request.pmf

theorem SamplerQuery.isProbabilityMeasure (query : SamplerQuery) :
    @IsProbabilityMeasure query.Answer ⊤ query.measure := by
  cases query with
  | law l =>
    dsimp only [SamplerQuery.measure, SamplerLaw.measure]
    exact @PMF.toMeasure.isProbabilityMeasure _ ⊤ l.pmf
  | preimage request =>
    dsimp only [SamplerQuery.measure]
    exact @PMF.toMeasure.isProbabilityMeasure _ ⊤ request.pmf

/-- Each query is answered by its law. -/
noncomputable instance samplerAnswers : samplerSpec.AnswerMeasure :=
  @PFunctor.AnswerMeasure.mk samplerSpec.toPFunctor (fun _ ↦ ⊤)
    SamplerQuery.measure SamplerQuery.isProbabilityMeasure

/-- A computation that draws ideal samples. -/
abbrev Comp := OracleComp samplerSpec

/-- The support of `law` as a predicate: what the runtime sampler can return. -/
def _root_.MxxRuntime.SamplerLaw.Supports : SamplerLaw → Int → Prop
  | .gaussian sigma cutoff, x => if sigma ≠ 0 ∧ 0 ≤ cutoff then |x| ≤ cutoff else x = 0
  | .interval minimum maximum, x =>
      if minimum ≤ maximum then minimum ≤ x ∧ x ≤ maximum else x = minimum
  | .residue modulus, x => if 0 < modulus then 0 ≤ x ∧ x < modulus else x = 0

/-- A value in the support of `law`: `x` itself when `law` supports it, else a fixed supported
value. `law.measure` gives probability zero to the values this changes. -/
def _root_.MxxRuntime.SamplerLaw.clamp : SamplerLaw → Int → Int
  | .gaussian sigma cutoff, x => if sigma ≠ 0 ∧ 0 ≤ cutoff ∧ |x| ≤ cutoff then x else 0
  | .interval minimum maximum, x =>
      if minimum ≤ x ∧ x ≤ maximum then x else minimum
  | .residue modulus, x => if 0 ≤ x ∧ x < modulus then x else 0

theorem _root_.MxxRuntime.SamplerLaw.clamp_supports (law : SamplerLaw) (x : Int) : law.Supports (law.clamp x) := by
  cases law with
  | gaussian sigma cutoff =>
    simp only [MxxRuntime.SamplerLaw.Supports, MxxRuntime.SamplerLaw.clamp]
    split_ifs <;> simp_all
  | interval minimum maximum =>
    simp only [MxxRuntime.SamplerLaw.Supports, MxxRuntime.SamplerLaw.clamp]
    split_ifs <;> first | omega | (constructor <;> omega) | simp_all
  | residue modulus =>
    simp only [MxxRuntime.SamplerLaw.Supports, MxxRuntime.SamplerLaw.clamp]
    split_ifs <;> first | omega | (constructor <;> omega) | simp_all

/-- One coefficient drawn from `law`. The oracle answers by `law.measure`; `clamp` changes only
answers of probability zero, so every returned value is one the runtime sampler can return. -/
def sample (law : SamplerLaw) : Comp Int := do
  let x ← (OracleSpec.query (spec := samplerSpec) (.law law) : Comp Int)
  pure (law.clamp x)

/-- A family of independent draws from `law`, in index order. -/
def sampleFamily (law : SamplerLaw) : (count : Nat) → Comp (Fin count → Int)
  | 0 => pure Fin.elim0
  | count + 1 => do
      let head ← sample law
      let tail ← sampleFamily law count
      pure (Fin.cons head tail)

/-- A matrix over `Z_q[X]/(X^n + 1)` whose coefficients are independent draws from `law`,
reduced modulo `q`: entry by entry in row-major order, coefficient by coefficient. -/
noncomputable def sampleMatrix (law : SamplerLaw) (q n rows columns : Nat) :
    Comp (ExactMatrix q n rows columns) := do
  let values ← sampleFamily law (rows * columns * n)
  pure fun row column ↦ polynomialOfCoefficients fun coefficient ↦
    values (finProdFinEquiv (finProdFinEquiv (row, column), coefficient))

/-- The uniform residue sampler of the runtime, for the matrix type of its output. -/
noncomputable def sampleResidue {q n rows columns : Nat} : Comp (ExactMatrix q n rows columns) :=
  sampleMatrix (.residue q) q n rows columns

/-- The uniform interval sampler of the runtime on `[minimum, maximum]`. -/
noncomputable def sampleInterval {q n rows columns : Nat} (minimum maximum : Int) :
    Comp (ExactMatrix q n rows columns) :=
  sampleMatrix (.interval minimum maximum) q n rows columns

/-- The truncated Gaussian sampler of the runtime with width `sigma` and bound `cutoff`. -/
noncomputable def sampleGaussian {q n rows columns : Nat} (sigma : Rat) (cutoff : Int) :
    Comp (ExactMatrix q n rows columns) :=
  sampleMatrix (.gaussian sigma cutoff) q n rows columns

/-- A uniformly random bit. -/
def sampleBit : Comp Bool := do
  let value ← sample (.interval 0 1)
  pure (value = 1)

/-- The output a relation determines. Generated computations draw their samplers and return the
outputs their generated relation relates to those draws; for a graph whose remaining operations
are deterministic this is the unique output of the runtime. -/
noncomputable def determined {α : Type} [Nonempty α] (relation : α → Prop) : α :=
  Classical.epsilon relation

theorem determined_eq {α : Type} [Nonempty α] {relation : α → Prop} {value : α}
    (holds : relation value) (unique : ∀ other, relation other → other = value) :
    determined relation = value :=
  unique _ (Classical.epsilon_spec ⟨value, holds⟩)

/-- The probability that a Boolean computation returns `true`. -/
noncomputable def probTrue (game : Comp Bool) : ENNReal :=
  evalDist game {true}

/-- The distinguishing advantage between two Boolean computations. -/
noncomputable def distance (left right : Comp Bool) : ENNReal :=
  max (probTrue left - probTrue right) (probTrue right - probTrue left)

end MxxSecurity
