import RuntimeMatrixOps
import RuntimeHash
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.Distributions.Uniform

/-!
# Sampling tapes

A probabilistic claim reads every sampled coefficient from a sampling tape. A key names one
coefficient: the occurrence's `site` (its stage, enclosing call and loop nodes with their
iterations, and the sampler node), the entry and coefficient within the sampled matrix, and the
law the sampler draws from. `tapeMeasure` makes all keys independent, each distributed by its
law: this is the ideal-sampler assumption (R1) of the generated claims.

The random-oracle law of hash models (R2) is `randomOracle`. A claim that holds for every hash
model also holds averaged over it (`lintegral_randomOracle_le`).
-/

namespace MxxRuntime

open MeasureTheory Mxx.Primitives

/-- The distribution of one sampled coefficient. -/
inductive SamplerLaw where
  /-- The discrete Gaussian `∝ exp(-x² / (2 sigma²))` conditioned on `|x| ≤ cutoff`; the point
  mass at zero when `sigma = 0` or `cutoff < 0`. -/
  | gaussian (sigma : Rat) (cutoff : Int)
  /-- The uniform distribution on `[minimum, maximum]`; the point mass at `minimum` when empty. -/
  | interval (minimum maximum : Int)
  /-- The uniform distribution on `[0, modulus)`; the point mass at zero when empty. -/
  | residue (modulus : Nat)
  deriving DecidableEq

/-- One sampled coefficient: coefficient `coefficient` of entry `(row, column)` at `site`. -/
structure SampleKey where
  site : List Nat
  row : Nat
  column : Nat
  coefficient : Nat
  law : SamplerLaw
  deriving DecidableEq

abbrev SampleTape := SampleKey → Int

/-- The unnormalized truncated Gaussian weight. -/
noncomputable def gaussianWeight (sigma : Rat) (cutoff : Int) (x : Int) : ENNReal :=
  if |x| ≤ cutoff then ENNReal.ofReal (Real.exp (-((x : ℝ) ^ 2) / (2 * (sigma : ℝ) ^ 2))) else 0

theorem gaussianWeight_eq_zero {sigma : Rat} {cutoff x : Int} (hx : x ∉ Finset.Icc (-cutoff) cutoff) :
    gaussianWeight sigma cutoff x = 0 := by
  unfold gaussianWeight
  rw [if_neg]
  intro h
  exact hx (Finset.mem_Icc.mpr ⟨by have := neg_abs_le x; linarith, (le_abs_self x).trans h⟩)

theorem gaussianWeight_tsum_ne_top (sigma : Rat) (cutoff : Int) :
    ∑' x, gaussianWeight sigma cutoff x ≠ ⊤ := by
  rw [tsum_eq_sum (s := Finset.Icc (-cutoff) cutoff) fun x hx ↦ gaussianWeight_eq_zero hx]
  exact ENNReal.sum_ne_top.mpr fun x _ ↦ by unfold gaussianWeight; split_ifs <;> simp

theorem gaussianWeight_tsum_ne_zero {sigma : Rat} {cutoff : Int} (hcutoff : 0 ≤ cutoff) :
    ∑' x, gaussianWeight sigma cutoff x ≠ 0 := by
  refine ne_of_gt (lt_of_lt_of_le ?_ (ENNReal.le_tsum 0))
  simp [gaussianWeight, hcutoff]

/-- The distribution of one coefficient drawn by `law`. -/
noncomputable def SamplerLaw.pmf : SamplerLaw → PMF Int
  | .gaussian sigma cutoff =>
    if h : sigma ≠ 0 ∧ 0 ≤ cutoff then
      PMF.normalize (gaussianWeight sigma cutoff) (gaussianWeight_tsum_ne_zero h.2)
        (gaussianWeight_tsum_ne_top sigma cutoff)
    else PMF.pure 0
  | .interval minimum maximum =>
    if h : minimum ≤ maximum then
      PMF.uniformOfFinset (Finset.Icc minimum maximum) ⟨minimum, by simp [h]⟩
    else PMF.pure minimum
  | .residue modulus =>
    if h : 0 < modulus then
      PMF.uniformOfFinset ((Finset.range modulus).image (fun value : Nat ↦ (value : Int)))
        ⟨0, by simp [h]⟩
    else PMF.pure 0

noncomputable def SamplerLaw.measure (law : SamplerLaw) : Measure Int := law.pmf.toMeasure

instance (law : SamplerLaw) : IsProbabilityMeasure law.measure := by
  unfold SamplerLaw.measure
  infer_instance

/-- Every key independent with its own law (R1). -/
noncomputable def tapeMeasure : Measure SampleTape :=
  Measure.infinitePi fun key ↦ key.law.measure

instance : IsProbabilityMeasure tapeMeasure := by
  unfold tapeMeasure
  infer_instance

/-- The matrix whose coefficients are the tape values at `site` under `law`. -/
noncomputable def tapeMatrix {q n rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (law : SamplerLaw) : ExactMatrix q n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun coefficient : Fin n ↦
    tape ⟨site, row.val, column.val, coefficient.val, law⟩

/-- `uniformResidueSample` reading its coefficients from the tape. -/
def uniformResidueSampleAt {q n rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (output : ExactMatrix q n rows columns) : Prop :=
  (∀ (row : Fin rows) (column : Fin columns) (coefficient : Fin n),
    0 ≤ tape ⟨site, row.val, column.val, coefficient.val, .residue q⟩ ∧
      tape ⟨site, row.val, column.val, coefficient.val, .residue q⟩ < q) ∧
  output = tapeMatrix tape site (.residue q)

/-- `uniformIntervalSample` reading its coefficients from the tape. -/
def uniformIntervalSampleAt {q n rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (minimum maximum : Int) (output : ExactMatrix q n rows columns) : Prop :=
  minimum ≤ maximum ∧
  (∀ (row : Fin rows) (column : Fin columns) (coefficient : Fin n),
    minimum ≤ tape ⟨site, row.val, column.val, coefficient.val, .interval minimum maximum⟩ ∧
      tape ⟨site, row.val, column.val, coefficient.val, .interval minimum maximum⟩ ≤ maximum) ∧
  output = tapeMatrix tape site (.interval minimum maximum)

/-- `gaussianSample` reading its coefficients from the tape. -/
def gaussianSampleAt {q n rows columns : Nat} (tape : SampleTape) (site : List Nat)
    (sigma : Rat) (cutoff : Int) (output : ExactMatrix q n rows columns) : Prop :=
  0 ≤ sigma ∧ 0 ≤ cutoff ∧
  (∀ (row : Fin rows) (column : Fin columns) (coefficient : Fin n),
    |tape ⟨site, row.val, column.val, coefficient.val, .gaussian sigma cutoff⟩| ≤ cutoff) ∧
  (sigma = 0 → output = 0) ∧
  output = tapeMatrix tape site (.gaussian sigma cutoff)

theorem tapeMatrix_coeff {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n) (tape : SampleTape)
    (site : List Nat) (law : SamplerLaw) (row : Fin rows) (column : Fin columns) (k : Fin n) :
    ((tapeMatrix (q := q) (n := n) tape site law : ExactMatrix q n rows columns) row column).coeff k =
      (tape ⟨site, row.val, column.val, k.val, law⟩ : ZMod q) :=
  polynomialOfCoefficients_coeff hq hn _ k

/-- Tape values within `[0, q)` are their own canonical residues. -/
theorem tapeMatrix_coeff_val {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (tape : SampleTape) (site : List Nat) (law : SamplerLaw) (row : Fin rows)
    (column : Fin columns) (k : Fin n) (h0 : 0 ≤ tape ⟨site, row.val, column.val, k.val, law⟩)
    (hlt : tape ⟨site, row.val, column.val, k.val, law⟩ < q) :
    ((((tapeMatrix (q := q) (n := n) tape site law : ExactMatrix q n rows columns) row column).coeff
      k).val : Int) = tape ⟨site, row.val, column.val, k.val, law⟩ := by
  letI : NeZero q := ⟨by omega⟩
  rw [tapeMatrix_coeff hq hn, ZMod.val_intCast, Int.emod_eq_of_lt h0 hlt]

/-- Tape values of a matrix sampled at `site` agree when the tapes agree on its keys. -/
theorem tapeMatrix_congr {q n rows columns : Nat} {tape tape' : SampleTape} {site : List Nat}
    {law : SamplerLaw}
    (h : ∀ (row : Fin rows) (column : Fin columns) (k : Fin n),
      tape ⟨site, row.val, column.val, k.val, law⟩ = tape' ⟨site, row.val, column.val, k.val, law⟩) :
    (tapeMatrix tape site law : ExactMatrix q n rows columns) = tapeMatrix tape' site law := by
  funext row column
  unfold tapeMatrix
  congr 1
  funext k
  exact h row column k

/-! ## Random oracle (R2) -/

/-- One output coefficient of a hash model: a `sample` coefficient or an `integers` entry. -/
inductive HashKey where
  | sample (q n rows columns : Nat) (key : ByteArray) (blob : Blob) (row column coefficient : Nat)
  | integers (count modulus : Nat) (key : ByteArray) (blob : Blob) (index : Nat)

/-- Each coefficient uniform below its modulus. -/
noncomputable def HashKey.measure : HashKey → Measure Int
  | .sample q .. => (SamplerLaw.residue q).measure
  | .integers _ modulus .. => (SamplerLaw.residue modulus).measure

instance (key : HashKey) : IsProbabilityMeasure key.measure := by
  cases key <;> unfold HashKey.measure <;> infer_instance

/-- The hash model whose outputs are the given coefficients. -/
noncomputable def hashModelOf (values : HashKey → Int) : HashModel where
  sample q n rows columns key blob row column := polynomialOfCoefficients fun coefficient ↦
    values (.sample q n rows columns key blob row.val column.val coefficient.val)
  integers count modulus key blob index := (values (.integers count modulus key blob index.val)).toNat

instance : MeasurableSpace HashModel :=
  MeasurableSpace.map hashModelOf MeasurableSpace.pi

/-- Every hash output independent and uniform (R2). -/
noncomputable def randomOracle : Measure HashModel :=
  (Measure.infinitePi HashKey.measure).map hashModelOf

/-- A bound for every hash model is also a bound averaged over the random oracle. -/
theorem lintegral_randomOracle_le {f : HashModel → ENNReal} {bound : ENNReal}
    (h : ∀ model, f model ≤ bound) : ∫⁻ model, f model ∂randomOracle ≤ bound := by
  have : IsProbabilityMeasure randomOracle := by
    unfold randomOracle
    exact Measure.isProbabilityMeasure_map (show Measurable hashModelOf from fun _ hs ↦ hs).aemeasurable
  calc ∫⁻ model, f model ∂randomOracle ≤ ∫⁻ _, bound ∂randomOracle := lintegral_mono h
    _ = bound := by simp

end MxxRuntime
