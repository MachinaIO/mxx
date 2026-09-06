import DiamondProofParameters
import DiamondLayerProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_injector_loop_common_secret
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (digitBase count : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbaseParams : params.diamond_digit_base = decryptParams.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (packed : Fin inputCount → Int)
    (hpacked : ∀ index : Fin inputCount, 0 ≤ packed index ∧ packed index < digitBase)
    (states outputs : Fin stateCount → ExactMatrix q n 1 inner)
    (hinitial : ∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend decryptParams
      state.val initial (states state))
    (hrun : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_8 backend decryptParams
        layer (current, packed, transitions, ()) next) count states outputs) :
    ∃ commonSecret : ExactPoly q n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat count * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ExactMatrix q n 1 2)
        (error : ExactMatrix q n 1 inner), position.val = count * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧ outputs state = row * producer.bases position + error := by
  let Invariant := fun (layer : Nat) (values : Fin stateCount → ExactMatrix q n 1 inner) ↦
    ∃ commonSecret : ExactPoly q n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ExactMatrix q n 1 2)
        (error : ExactMatrix q n 1 inner), position.val = layer * DiamondProofParameters.stateCount + state.val ∧
        row 0 0 = commonSecret ∧ values state = row * producer.bases position + error
  have hstart : Invariant 0 states := by
    refine ⟨producer.secret 0 0, ?_⟩
    intro state hactive
    have hstate : state = (0 : Fin stateCount) := by
      apply Fin.ext
      dsimp at hactive ⊢
      omega
    subst state
    have hzero := hinitial 0
    dsimp only [Stage_decrypt.parallel_generatedRoot_2] at hzero
    rcases hzero with ⟨value, _, _, _, ⟨position, hposition, hvalue⟩, hout⟩
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hstateValue : states 0 = initial := hout.trans hvalue
    rcases producer.initialBaseRun with ⟨basePosition, hbasePosition, hbaseValue⟩
    refine ⟨basePosition, producer.initialSelector, producer.initialError, ?_, ?_, ?_⟩
    · dsimp at hbasePosition ⊢
      omega
    · simpa [concatColumns] using producer.initialSelectorRun 0 0
    · exact hstateValue.trans (by simpa only [hbaseValue] using producer.initialEquation)
  apply MxxIR.IterRuns.invariant (Invariant := Invariant) hstart _ hrun
  intro layer current next ih hstep
  obtain ⟨secret, hstates⟩ := ih
  obtain ⟨digit, samplePosition, _, _, hnext⟩ := generated_injector_layer backend hashModel
    params decryptParams message initial transitions producer digitBase layer hdigits hbase
      hbaseParams hbatch hbatchNonneg packed hpacked current next secret hstates hstep
  refine ⟨secret * producer.digitSamples samplePosition 0 0, ?_⟩
  intro state _
  exact hnext state

#print axioms generated_injector_loop_common_secret

end DiamondGeneratedProof
