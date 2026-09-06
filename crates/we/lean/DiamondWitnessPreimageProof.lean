import DiamondProofParameters
import DiamondInjectorWitness

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_witness_preimage_link
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (terminalBases : Fin stateCount → ExactMatrix q n 2 inner)
    (terminalTrapdoors : Fin stateCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (hbases : ∀ i : Fin stateCount, Stage_encrypt.parallel_generatedRoot_1 backend hashModel params
      i.val (producer.bases, producer.trapdoors, ()) (terminalBases i, terminalTrapdoors i, ()))
    (publicInputs : Fin stateCount → ExactMatrix q n 1 ell) (gadget : ExactMatrix q n 1 ell)
    (i : Fin witnessSlots) (preimage : ExactMatrix q n inner ell)
    (hrun : Stage_encrypt.parallel_generatedRoot_62 backend hashModel params i.val
      (terminalBases ⟨i.val + 1, by have := i.isLt; dsimp [witnessSlots, stateCount] at *; omega⟩,
       terminalTrapdoors ⟨i.val + 1, by have := i.isLt; dsimp [witnessSlots, stateCount] at *; omega⟩,
       publicInputs ⟨i.val + 1, by have := i.isLt; dsimp [witnessSlots, stateCount] at *; omega⟩,
       gadget, ()) preimage) :
    ∃ (state : Fin stateCount) (position : Fin basePoolCount) (target : ExactMatrix q n 2 ell),
      state.val = i.val + 1 ∧
      (position.val : Int) = params.diamond_input_count *
        (1 + params.diamond_batch_bits * params.diamond_input_count) + (state.val : Int) ∧
      concatRows (publicInputs state) (-gadget) target ∧
      producer.bases position * preimage = target ∧
      PreimageWithin preimage params.diamond_preimage_max_coefficient_bound.toNat := by
  let state : Fin stateCount := ⟨i.val + 1, by have := i.isLt; dsimp [witnessSlots, stateCount] at *; omega⟩
  rcases hbases state with ⟨baseValue, baseTrapdoor, _, _,
    ⟨position, hposition, hbaseValue⟩, _, _,
    ⟨trapdoorPosition, htrapdoorPosition, hbaseTrapdoor⟩, hbaseOutput⟩
  have hpositions : trapdoorPosition = position := by
    apply Fin.ext
    exact_mod_cast htrapdoorPosition.trans hposition.symm
  subst trapdoorPosition
  have hbaseEq : terminalBases state = producer.bases position :=
    (congrArg (fun value ↦ value.1) hbaseOutput).trans hbaseValue
  have htrapdoorPool : terminalTrapdoors state = producer.trapdoors position :=
    (congrArg (fun value ↦ value.2.1) hbaseOutput).trans hbaseTrapdoor
  rcases hrun with ⟨target, sampled, hrows, _, hdispatch, hout⟩
  have hequation : producer.bases position * preimage = target := by
    rw [← hbaseEq, hout]
    exact preimageRunsDispatched_equation (by decide) (by decide) hdispatch
  rcases producer.basesRun position with ⟨sampledTrapdoor, sampledBase, hsample, hsampleOut⟩
  have hsampledTrapdoor : producer.trapdoors position = sampledTrapdoor :=
    congrArg (fun value ↦ value.2.1) hsampleOut
  have hkind : (terminalTrapdoors state).kind = .sampledSecret := by
    rw [htrapdoorPool, hsampledTrapdoor]
    exact trapdoorSample_sampled hsample
  have hbound : PreimageWithin preimage params.diamond_preimage_max_coefficient_bound.toNat := by
    rcases hdispatch.2 with hsampled | hpublic
    · exact hout.symm ▸ preimageRuns_bounded hsampled
    · have hbad := hkind.symm.trans hpublic.1
      cases hbad
  exact ⟨state, position, target, rfl, hposition.trans (by simp only [Int.ofNat_eq_natCast]; ring), hrows, hequation, hbound⟩

#print axioms generated_witness_preimage_link

end DiamondGeneratedProof
