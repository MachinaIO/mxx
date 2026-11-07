import RuntimeLemmas
import RuntimeSampling

/-!
# Tape samplers

The average-case claim reads every sampler from a tape (`…SampleAt`). Each tape relation records
the cutoff of its law, so a tape sample is also a sample of the deterministic relation used by
the worst-case analysis: `uniformIntervalSampleAt` gives `uniformIntervalSample`, and
`gaussianSampleAt` gives `gaussianSample`. The failure set of an average-case claim whose
deterministic analysis holds on every such tape is therefore empty.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime

variable {q n rows columns : Nat}

/-- The integer matrix of the tape values at `site` under `law`. -/
noncomputable def tapeWitness (tape : SampleTape) (site : List Nat) (law : SamplerLaw) :
    ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun k ↦ tape ⟨site, row.val, column.val, k.val, law⟩

theorem tapeMatrix_eq_reduce (tape : SampleTape) (site : List Nat) (law : SamplerLaw) :
    (tapeMatrix tape site law : ExactMatrix q n rows columns) =
      reduceMatrix q n rows columns (tapeWitness tape site law) := by
  funext row column
  exact polynomialOfCoefficients_eq_reduce _

theorem uniformIntervalSampleAt_sample (hn : 0 < n) {tape : SampleTape} {site : List Nat}
    {minimum maximum : Int} {output : ExactMatrix q n rows columns}
    (h : uniformIntervalSampleAt tape site minimum maximum output) :
    uniformIntervalSample minimum maximum output := by
  obtain ⟨hle, hbound, hout⟩ := h
  refine ⟨hle, tapeWitness tape site (.interval minimum maximum),
    hout.trans (tapeMatrix_eq_reduce _ _ _), fun row column k ↦ ?_⟩
  rw [tapeWitness, intPoly_coeff hn]
  exact hbound row column k

theorem gaussianSampleAt_sample (hn : 0 < n) {tape : SampleTape} {site : List Nat}
    {sigma : Rat} {cutoff : Int} {output : ExactMatrix q n rows columns}
    (h : gaussianSampleAt tape site sigma cutoff output) : gaussianSample sigma cutoff output := by
  obtain ⟨hs, hc, hbound, hzero, hout⟩ := h
  refine ⟨hs, hc, ⟨tapeWitness tape site (.gaussian sigma cutoff),
    hout.trans (tapeMatrix_eq_reduce _ _ _), fun row column k ↦ ?_⟩, hzero⟩
  rw [tapeWitness, intPoly_coeff hn]
  have := hbound row column k
  rw [abs_le] at this
  omega

end MxxExponentLut
