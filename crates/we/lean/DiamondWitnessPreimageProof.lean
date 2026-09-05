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
    (terminalIndices : Fin stateCount → Int)
    (terminalBases : Fin stateCount → ExactMatrix q n 2 inner)
    (terminalTrapdoors : Fin stateCount → TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (hindices : ∀ i : Fin stateCount, Stage_encrypt.parallel_generatedRoot_0 backend hashModel params
      i.val () (terminalIndices i))
    (hbases : ∀ i : Fin stateCount, Stage_encrypt.parallel_generatedRoot_2 backend hashModel params
      i.val (terminalIndices i, producer.bases, producer.trapdoors, ())
      (terminalBases i, terminalTrapdoors i, ()))
    (publicInputs : Fin stateCount → ExactMatrix q n 1 ell) (gadget : ExactMatrix q n 1 ell)
    (i : Fin witnessSlots) (index : Int)
    (source : ExactMatrix q n 2 inner)
    (trapdoor : TrapdoorValue (ExactMatrix q n 2 inner) Unit)
    (publicMatrix : ExactMatrix q n 1 ell) (target : ExactMatrix q n 2 ell)
    (preimage : ExactMatrix q n inner ell)
    (hindex : Stage_encrypt.parallel_generatedRoot_74 backend hashModel params i.val () index)
    (hsource : Stage_encrypt.parallel_generatedRoot_75 backend hashModel params i.val
      (index, terminalBases, terminalTrapdoors, ()) (source, trapdoor, ()))
    (hpublic : Stage_encrypt.parallel_generatedRoot_76 backend hashModel params i.val
      (index, publicInputs, ()) publicMatrix)
    (htarget : Stage_encrypt.parallel_generatedRoot_77 backend hashModel params i.val
      (publicMatrix, gadget, ()) target)
    (hpreimage : Stage_encrypt.parallel_generatedRoot_78 backend hashModel params i.val
      (source, trapdoor, target, ()) preimage) :
    ∃ (state : Fin stateCount) (position : Fin basePoolCount), state.val = i.val + 1 ∧
      (position.val : Int) = params.diamond_input_count *
        (1 + params.diamond_batch_bits * params.diamond_input_count) + (state.val : Int) ∧
      concatRows (publicInputs state) (-gadget) target ∧
      producer.bases position * preimage = target ∧
      PreimageWithin preimage params.diamond_preimage_max_coefficient_bound.toNat := by
  rcases hsource with ⟨sourceValue, trapdoorValue, _, _,
    ⟨state, hstate, hsourceValue⟩, _, _, ⟨trapdoorState, htrapdoorState, htrapdoorValue⟩,
    hsourceOutput⟩
  have hstates : trapdoorState = state := by
    apply Fin.ext
    exact_mod_cast htrapdoorState.trans hstate.symm
  subst trapdoorState
  have hsourceEq : source = terminalBases state :=
    (congrArg (fun value ↦ value.1) hsourceOutput).trans hsourceValue
  have htrapdoorEq : trapdoor = terminalTrapdoors state :=
    (congrArg (fun value ↦ value.2.1) hsourceOutput).trans htrapdoorValue
  rcases hbases state with ⟨baseValue, baseTrapdoor, _, _,
    ⟨position, hposition, hbaseValue⟩, _, _,
    ⟨trapdoorPosition, htrapdoorPosition, hbaseTrapdoor⟩, hbaseOutput⟩
  have hpositions : trapdoorPosition = position := by
    apply Fin.ext
    exact_mod_cast htrapdoorPosition.trans hposition.symm
  subst trapdoorPosition
  have hbaseEq : source = producer.bases position := hsourceEq.trans
    ((congrArg (fun value ↦ value.1) hbaseOutput).trans hbaseValue)
  have htrapdoorPool : trapdoor = producer.trapdoors position := htrapdoorEq.trans
    ((congrArg (fun value ↦ value.2.1) hbaseOutput).trans hbaseTrapdoor)
  rcases hpublic with ⟨publicValue, _, _, ⟨publicState, hpublicState, hpublicValue⟩, hpublicOut⟩
  have hpublicStates : publicState = state := by
    apply Fin.ext
    exact_mod_cast hpublicState.trans hstate.symm
  subst publicState
  have hpublicEq : publicMatrix = publicInputs state := hpublicOut.trans hpublicValue
  rcases htarget with ⟨targetValue, hrows, htargetOut⟩
  have htargetRows : concatRows (publicInputs state) (-gadget) target := by
    simpa only [← hpublicEq, htargetOut] using hrows
  rcases hpreimage with ⟨preimageValue, _, hdispatch, hpreimageOut⟩
  have hequation : producer.bases position * preimage = target := by
    rw [← hbaseEq, hpreimageOut]
    exact preimageRunsDispatched_equation (by decide) (by decide) hdispatch
  rcases producer.basesRun position with ⟨sampledTrapdoor, sampledBase, hsample, hsampleOut⟩
  have hsampledTrapdoor : producer.trapdoors position = sampledTrapdoor :=
    congrArg (fun value ↦ value.2.1) hsampleOut
  have hkind : trapdoor.kind = .sampledSecret := by
    rw [htrapdoorPool, hsampledTrapdoor]
    exact trapdoorSample_sampled hsample
  have hbound : PreimageWithin preimage params.diamond_preimage_max_coefficient_bound.toNat := by
    rcases hdispatch.2 with hsampled | hpublic
    · exact hpreimageOut.symm ▸ preimageRuns_bounded hsampled
    · have hbad := hkind.symm.trans hpublic.1
      cases hbad
  refine ⟨state, position, ?_, ?_, htargetRows, hequation, hbound⟩
  · change index = Int.ofNat i.val + 1 at hindex
    have h := hstate.trans hindex
    change (state.val : Int) = (i.val : Int) + 1 at h
    exact_mod_cast h
  · exact hposition.trans (hindices state)

#print axioms generated_witness_preimage_link

end DiamondGeneratedProof
