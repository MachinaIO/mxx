import MxxWe.DiamondGeneric

open Mxx

namespace MxxWe

/-! Parameter-independent bridges from executable sampler support to the matrix facts consumed by
the Diamond algebra.  These lemmas only package the common sampler contract; they introduce no
protocol-specific assumption. -/

structure BoundedMatrix (matrix : Mxx.Matrix) (params : Mxx.SamplerParams) : Prop where
  shape : Mxx.Toolkit.MatrixShape matrix params.modulus params.ringDimension
    params.rows params.columns
  norm : Mxx.maxCenteredCoefficientNorm matrix ≤ params.maxCoefficientBound

theorem gaussianSample_bounded
    (samplers : MxxSamplerFamily) (contract : MxxBoundedSamplerContract samplers)
    (params : Mxx.SamplerParams) (raw : Mxx.Matrix)
    (member : raw ∈ samplers.gaussianSample params) :
    BoundedMatrix (raw.withSamplerParams params) params := by
  exact ⟨Mxx.Toolkit.withSamplerParams_shape raw params,
    contract.gaussianHardSupport params raw member⟩

theorem preimageSample_bounded
    (samplers : MxxSamplerFamily) (contract : MxxBoundedSamplerContract samplers)
    (params : Mxx.SamplerParams) (publicMatrix target raw : Mxx.Matrix)
    (member : raw ∈ samplers.samplePreimage params publicMatrix target) :
    Mxx.matrixMul publicMatrix (raw.withSamplerParams params) = target ∧
      BoundedMatrix (raw.withSamplerParams params) params := by
  have sampled := contract.preimageContract params publicMatrix target raw member
  exact ⟨sampled.1, Mxx.Toolkit.withSamplerParams_shape raw params, sampled.2⟩

theorem gadgetDecompose_bounded
    (samplers : MxxSamplerFamily) (contract : MxxBoundedSamplerContract samplers)
    (params : Mxx.SamplerParams) (base : Int) (digitCount : Nat)
    (input raw : Mxx.Matrix)
    (member : raw ∈ samplers.gadgetDecompose params base digitCount input) :
    Mxx.matrixMul
        (Mxx.gadgetMatrix
          { params with rows := input.rows, columns := input.rows * digitCount }
          base digitCount)
        (raw.withSamplerParams params) = input ∧
      Mxx.Toolkit.MatrixShape (raw.withSamplerParams params) params.modulus
        params.ringDimension params.rows params.columns ∧
      Mxx.maxCenteredCoefficientNorm (raw.withSamplerParams params) ≤
        max (base.natAbs / 2) 1 := by
  have decomposed := contract.gadgetDecomposeContract params base digitCount input raw member
  exact ⟨decomposed.1, Mxx.Toolkit.withSamplerParams_shape raw params, decomposed.2⟩

end MxxWe
