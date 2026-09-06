import Stage_encrypt
import DiamondProofParameters
import DiamondSelectorWitness

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

/-- A projection of one actual encryption run. This packages existing values and
    generated-scope proofs, and defines no replacement execution semantics. -/
structure InjectorRootWitness
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initialState : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner) where
  bases : Fin basePoolCount → ExactMatrix q n 2 inner
  trapdoors : Fin basePoolCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit
  secret : ExactMatrix q n 1 1
  messageValue : ExactMatrix q n 1 1
  initialSelector : ExactMatrix q n 1 2
  initialBase : ExactMatrix q n 2 inner
  initialError : ExactMatrix q n 1 inner
  sourceIndices : Fin transitionCount → Int
  digitIndices : Fin transitionCount → Int
  targetIndices : Fin transitionCount → Int
  sourcePublics : Fin transitionCount → ExactMatrix q n 2 inner
  targetPublics : Fin transitionCount → ExactMatrix q n 2 inner
  targets : Fin transitionCount → ExactMatrix q n 2 inner
  sourceTrapdoors : Fin transitionCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit
  digitSamples : Fin sampleCount → ExactMatrix q n 1 1
  digitSecrets : Fin transitionCount → ExactMatrix q n 1 1
  stateCount : 1 + params.diamond_batch_bits * params.diamond_input_count = (DiamondProofParameters.stateCount : Int)
  basesRun : ∀ i : Fin basePoolCount, Stage_encrypt.parallel_generatedRoot_0 backend hashModel params
    i () (bases i, trapdoors i, ())
  secretRun : uniformIntervalSample (-1) 1 secret
  messageRun : select (if message then 1 else 0) [0, 1] messageValue
  initialSelectorRun : concatColumns secret messageValue initialSelector
  initialBaseRun : familyGetStatic bases 0 initialBase
  initialErrorRun : gaussianSample params.diamond_error_sigma
    params.diamond_error_max_coefficient_bound initialError
  initialEquation : initialState = initialSelector * initialBase + initialError
  sourceIndicesRun : ∀ i : Fin transitionCount, ∃ selected : Int,
    select (if decide (((i.val : Int) /
        (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
          params.diamond_digit_base)) * params.diamond_batch_bits + 1 ≤
        (i.val : Int) % (1 + params.diamond_batch_bits * params.diamond_input_count)) then 1 else 0)
      [(i.val : Int) % (1 + params.diamond_batch_bits * params.diamond_input_count), 0] selected ∧
    sourceIndices i = ((i.val : Int) /
      (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base)) * (1 + params.diamond_batch_bits * params.diamond_input_count) + selected
  sourcesRun : ∀ i : Fin transitionCount,
    familyGetDynamic bases (sourceIndices i) (sourcePublics i) ∧
    familyGetDynamic trapdoors (sourceIndices i) (sourceTrapdoors i)
  digitIndicesRun : ∀ i : Fin transitionCount, digitIndices i = (i.val : Int) /
    (1 + params.diamond_batch_bits * params.diamond_input_count)
  samplesRun : ∀ i : Fin sampleCount, Stage_encrypt.parallel_generatedRoot_59 backend hashModel params
    i () (digitSamples i)
  secretsRun : ∀ i : Fin transitionCount,
    familyGetDynamic digitSamples (digitIndices i) (digitSecrets i)
  targetIndicesRun : ∀ i : Fin transitionCount, targetIndices i =
    (((i.val : Int) /
      (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
        params.diamond_digit_base)) + 1) *
      (1 + params.diamond_batch_bits * params.diamond_input_count) +
      (i.val : Int) % (1 + params.diamond_batch_bits * params.diamond_input_count)
  targetPublicsRun : ∀ i : Fin transitionCount,
    familyGetDynamic bases (targetIndices i) (targetPublics i)
  targetsRun : ∀ i : Fin transitionCount,
    Nonempty (InjectorSelectorWitness backend hashModel params i.val
      (digitSecrets i) (targetPublics i) (targets i))
  preimagesRun : ∀ i : Fin transitionCount,
    preimageRunsDispatched backend (sourcePublics i) (sourceTrapdoors i) (targets i)
      params.diamond_preimage_max_coefficient_bound.toNat (transitions i)

end DiamondGeneratedProof
