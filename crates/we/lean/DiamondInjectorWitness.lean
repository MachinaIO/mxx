import Stage_encrypt
import DiamondProofParameters

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
  basesRun : ∀ i : Fin basePoolCount, Stage_encrypt.parallel_generatedRoot_1 backend hashModel params
    i () (bases i, trapdoors i, ())
  secretRun : uniformIntervalSample (-1) 1 secret
  messageRun : select (if message then 1 else 0) [0, 1] messageValue
  initialSelectorRun : concatColumns secret messageValue initialSelector
  initialBaseRun : familyGetStatic bases 0 initialBase
  initialErrorRun : gaussianSample params.diamond_error_sigma
    params.diamond_error_max_coefficient_bound initialError
  initialEquation : initialState = initialSelector * initialBase + initialError
  sourceIndicesRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_65 backend hashModel params
    i () (sourceIndices i)
  sourcesRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_66 backend hashModel params
    i (sourceIndices i, bases, trapdoors, ()) (sourcePublics i, sourceTrapdoors i, ())
  digitIndicesRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_67 backend hashModel params
    i () (digitIndices i)
  samplesRun : ∀ i : Fin sampleCount, Stage_encrypt.parallel_generatedRoot_68 backend hashModel params
    i () (digitSamples i)
  secretsRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_69 backend hashModel params
    i (digitIndices i, digitSamples, ()) (digitSecrets i)
  targetIndicesRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_70 backend hashModel params
    i () (targetIndices i)
  targetPublicsRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_71 backend hashModel params
    i (targetIndices i, bases, ()) (targetPublics i)
  targetsRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_72 backend hashModel params
    i (digitSecrets i, targetPublics i, ()) (targets i)
  preimagesRun : ∀ i : Fin transitionCount, Stage_encrypt.parallel_generatedRoot_73 backend hashModel params
    i (sourcePublics i, sourceTrapdoors i, targets i, ()) (transitions i)

end DiamondGeneratedProof
