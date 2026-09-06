import DiamondProofParameters
import DiamondInjectorWitness
import DiamondFinalPublicProof
import DiamondWitnessPreimageProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192
set_option maxHeartbeats 1000000

/-- Initial state and transitions extracted together from the actual encryption root,
    retaining one sampled base pool, digit-secret family, and Boolean message selection. -/
theorem generated_injector_root
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (inputs : _) (outputs : _) (rootWitness : Stage_encrypt.generatedRoot.Witness)
    (hbody : Stage_encrypt.generatedRoot.body backend hashModel params inputs outputs rootWitness) :
    ∃ producer : InjectorRootWitness backend hashModel params inputs.2.2.2.2.2.2.2.1
      outputs.2.1 outputs.2.2.2.2.2.2.1,
      ∃ finalPublic : FinalPublicWitness backend params outputs.1 outputs.2.2.1 outputs.2.2.2.1
        outputs.2.2.2.2.2.1 (matrixPolynomial [MxxIR.roundDiv params.diamond_modulus 2]) outputs.2.2.2.2.1
        rootWitness.w_30_0,
        ∃ terminal : Fin basePoolCount,
          (terminal.val : Int) = params.diamond_input_count *
            (1 + params.diamond_batch_bits * params.diamond_input_count) ∧
          finalPublic.base = producer.bases terminal ∧
          ∀ i : Fin witnessSlots, ∃ (state : Fin stateCount) (position : Fin basePoolCount) (target : ExactMatrix q n 2 ell),
            state.val = i.val + 1 ∧
            (position.val : Int) = params.diamond_input_count *
              (1 + params.diamond_batch_bits * params.diamond_input_count) + state.val ∧
            concatRows (outputs.2.2.2.2.1 state) (-finalPublic.gadget) target ∧
            producer.bases position * outputs.2.2.2.2.2.2.2.1 i = target ∧
            PreimageWithin (outputs.2.2.2.2.2.2.2.1 i)
              params.diamond_preimage_max_coefficient_bound.toNat := by
  dsimp only [Stage_encrypt.generatedRoot.body, Stage_encrypt.generatedRoot.constraints_0,
    Stage_encrypt.generatedRoot.constraints_1] at hbody
  have hwhole := hbody
  rcases hbody with ⟨_, hbases, hstates, hterminalBases, hterminalBase, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsecret, _, _, _, hmessage, hselector, hinitialBase, herror, _, _, _, _, _, _, _, _, _, hsamples, _, htargets, _, hpreimages, _, hwitnesses, houtputValues⟩
  have houtputs : outputs = (rootWitness.w_39_0,
      rootWitness.w_46_0 * rootWitness.w_47_0 + rootWitness.w_49_0,
      rootWitness.w_53_0, rootWitness.w_58_0, rootWitness.w_8_0, rootWitness.w_34_0,
      rootWitness.w_61_0, rootWitness.w_62_0, ()) := houtputValues
  subst outputs
  have htargetViews := fun i : Fin transitionCount ↦ generated_selector_witness
    backend hashModel params i.val rootWitness.w_59_0 rootWitness.w_0_0
    (rootWitness.w_60_0 i) (htargets i)
  choose secrets targetPublics hsecrets htargetPublics hselectors using htargetViews
  have hsourceViews (i : Fin transitionCount) := hpreimages i
  dsimp only [Stage_encrypt.parallel_generatedRoot_61,
    Stage_encrypt.parallel_generatedRoot_61.constraints_0] at hsourceViews
  choose selected sourcePublics sourceTrapdoors sampled hsourceViews using hsourceViews
  have hsampled (i : Fin transitionCount) : rootWitness.w_61_0 i = sampled i := by
    exact (hsourceViews i).2.2.2.2.2.2.2.2.2.2.2.2.2.2
  let sourceIndices := fun i : Fin transitionCount ↦ (i.val : Int) /
    (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count +
      params.diamond_digit_base) * (1 + params.diamond_batch_bits * params.diamond_input_count) + selected i
  let producer : InjectorRootWitness backend hashModel params inputs.2.2.2.2.2.2.2.1
      (rootWitness.w_46_0 * rootWitness.w_47_0 + rootWitness.w_49_0) rootWitness.w_61_0 := {
    bases := rootWitness.w_0_0
    trapdoors := rootWitness.w_0_1
    secret := rootWitness.w_40_0
    messageValue := rootWitness.w_45_0
    initialSelector := rootWitness.w_46_0
    initialBase := rootWitness.w_47_0
    initialError := rootWitness.w_49_0
    sourceIndices := sourceIndices
    digitIndices := fun i ↦ (i.val : Int) / (1 + params.diamond_batch_bits * params.diamond_input_count)
    targetIndices := fun i ↦ (((i.val : Int) /
      (params.diamond_batch_bits * params.diamond_digit_base * params.diamond_input_count + params.diamond_digit_base)) + 1) *
      (1 + params.diamond_batch_bits * params.diamond_input_count) +
      (i.val : Int) % (1 + params.diamond_batch_bits * params.diamond_input_count)
    sourcePublics := sourcePublics
    targetPublics := targetPublics
    targets := rootWitness.w_60_0
    sourceTrapdoors := sourceTrapdoors
    digitSamples := rootWitness.w_59_0
    digitSecrets := secrets
    stateCount := hstates
    basesRun := hbases
    secretRun := hsecret
    messageRun := hmessage
    initialSelectorRun := hselector
    initialBaseRun := hinitialBase
    initialErrorRun := herror
    initialEquation := rfl
    sourceIndicesRun := by
      intro i
      have h := hsourceViews i
      exact ⟨selected i, by simpa only [add_zero] using h.2.2.2.2.2.1, rfl⟩
    sourcesRun := by
      intro i
      have h := hsourceViews i
      exact ⟨h.2.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.2.2.2.1⟩
    digitIndicesRun := fun _ ↦ rfl
    samplesRun := hsamples
    secretsRun := hsecrets
    targetIndicesRun := fun _ ↦ rfl
    targetPublicsRun := htargetPublics
    targetsRun := hselectors
    preimagesRun := by
      intro i
      rw [hsampled i]
      exact (hsourceViews i).2.2.2.2.2.2.2.2.2.2.2.2.2.1 }
  obtain ⟨terminalFamilyPosition, hterminalFamilyPosition, hterminalValue⟩ := hterminalBase
  have hterminalZero : terminalFamilyPosition = (0 : Fin stateCount) := by
    apply Fin.ext
    dsimp at hterminalFamilyPosition ⊢
    omega
  subst terminalFamilyPosition
  rcases hterminalBases 0 with ⟨baseValue, trapdoorValue, _, _,
    ⟨terminal, hterminalIndex, hbaseValue⟩, _, _, _, hterminalOutput⟩
  have hbase : rootWitness.w_2_0 = rootWitness.w_0_0 terminal := hterminalValue.trans
    ((congrArg (fun value ↦ value.1) hterminalOutput).trans hbaseValue)
  have hterminalAddress : (terminal.val : Int) = params.diamond_input_count *
      (1 + params.diamond_batch_bits * params.diamond_input_count) := by
    apply hterminalIndex.trans
    change params.diamond_batch_bits * params.diamond_input_count * params.diamond_input_count +
      params.diamond_input_count + 0 = _
    ring
  obtain ⟨finalPublic, hfinalBase, hfinalGadget⟩ :=
    generated_final_public_witness backend hashModel params rootWitness hwhole
  refine ⟨producer, finalPublic, terminal, hterminalAddress, hfinalBase.trans hbase, ?_⟩
  intro i
  have hpreimage := hwitnesses i
  obtain ⟨state, position, target, hstate, hposition, hrows, hequation, hbound⟩ :=
    generated_witness_preimage_link backend hashModel params _ _ _ producer
      rootWitness.w_1_0 rootWitness.w_1_1 hterminalBases rootWitness.w_8_0 rootWitness.w_54_0 i
      (rootWitness.w_62_0 i) hpreimage
  exact ⟨state, position, target, hstate, hposition, by simpa only [hfinalGadget] using hrows,
    hequation, hbound⟩

#print axioms generated_injector_root

end DiamondGeneratedProof
