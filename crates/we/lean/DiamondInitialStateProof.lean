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
    (inputs : _) (outputs : _)
    (hrun : Stage_encrypt.generatedRoot backend hashModel params inputs outputs) :
    ∃ producer : InjectorRootWitness backend hashModel params inputs.2.2.2.2.2.2.2.1
      outputs.2.1 outputs.2.2.2.2.2.2.1,
      ∃ finalPublic : FinalPublicWitness backend params outputs.1 outputs.2.2.1 outputs.2.2.2.1
        outputs.2.2.2.2.2.1 outputs.2.2.2.2.2.2.2.2.1 outputs.2.2.2.2.1
        outputs.2.2.2.2.2.2.2.2.2.1,
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
  dsimp only [Stage_encrypt.generatedRoot] at hrun
  rcases hrun with ⟨w_0_0, w_1_0, w_1_1, w_2_0, w_2_1, w_3_0, w_4_0, w_6_0, w_7_0, w_8_0, w_9_0, w_10_0,
    w_19_0, w_20_0, w_26_0, w_27_0, w_32_0, w_35_0, w_36_0, w_38_0, w_39_0, w_40_0,
    w_44_0, w_45_0, w_46_0, w_51_0, w_52_0, w_53_0, w_55_0, w_58_0, w_59_0, w_60_0,
    w_63_0, w_64_0, w_65_0, w_66_0, w_66_1, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0,
    w_72_0, w_73_0, w_74_0, w_75_0, w_75_1, w_76_0, w_77_0, w_78_0, hrelations⟩
  have hwhole := hrelations
  rcases hrelations with ⟨hstateCount, hterminalIndices, _, hbases, _, hterminalBases,
    hterminalBase, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, hsecret, _, _, _, hmessage, hselector, hbase,
    herror, _, _, _, _, _, _, _, _, _, hsourceIndices, _, hsources, _,
    hdigitIndices, _, hsamples, _, hsecrets, _, htargetIndices, _,
    htargetPublics, _, htargets, _, hpreimages, _, _, _, _, _, _, _, _,
    _, _, houtputs⟩
  subst outputs
  let producer : InjectorRootWitness backend hashModel params inputs.2.2.2.2.2.2.2.1
      (w_52_0 * w_53_0 + w_55_0) w_73_0 := {
    bases := w_1_0
    trapdoors := w_1_1
    secret := w_46_0
    messageValue := w_51_0
    initialSelector := w_52_0
    initialBase := w_53_0
    initialError := w_55_0
    sourceIndices := w_65_0
    digitIndices := w_67_0
    targetIndices := w_70_0
    sourcePublics := w_66_0
    targetPublics := w_71_0
    targets := w_72_0
    sourceTrapdoors := w_66_1
    digitSamples := w_68_0
    digitSecrets := w_69_0
    stateCount := hstateCount
    basesRun := hbases
    secretRun := hsecret
    messageRun := hmessage
    initialSelectorRun := hselector
    initialBaseRun := hbase
    initialErrorRun := herror
    initialEquation := rfl
    sourceIndicesRun := hsourceIndices
    sourcesRun := hsources
    digitIndicesRun := hdigitIndices
    samplesRun := hsamples
    secretsRun := hsecrets
    targetIndicesRun := htargetIndices
    targetPublicsRun := htargetPublics
    targetsRun := htargets
    preimagesRun := hpreimages }
  obtain ⟨terminalFamilyPosition, hterminalFamilyPosition, hterminalValue⟩ := hterminalBase
  have hterminalZero : terminalFamilyPosition = (0 : Fin stateCount) := by
    apply Fin.ext
    dsimp at hterminalFamilyPosition ⊢
    omega
  subst terminalFamilyPosition
  rcases hterminalBases 0 with ⟨baseValue, trapdoorValue, _, _,
    ⟨terminal, hterminalIndex, hbaseValue⟩, _, _, _, hterminalOutput⟩
  have hbase : w_3_0 = w_1_0 terminal := hterminalValue.trans
    ((congrArg (fun value ↦ value.1) hterminalOutput).trans hbaseValue)
  have hterminalAddress : (terminal.val : Int) = params.diamond_input_count *
      (1 + params.diamond_batch_bits * params.diamond_input_count) := by
    have hindex := hterminalIndices 0
    change w_0_0 0 = params.diamond_input_count *
      (1 + params.diamond_batch_bits * params.diamond_input_count) + Int.ofNat 0 at hindex
    simpa only [Int.ofNat_eq_natCast, Nat.cast_zero, add_zero] using hterminalIndex.trans hindex
  have hpublicOne : w_10_0 = w_9_0 0 := by
    obtain ⟨position, hposition, hvalue⟩ : familyGetStatic w_9_0 0 w_10_0 := by tauto
    have hz : position = 0 := Fin.ext (by change position.val = 0; omega)
    simpa only [hz] using hvalue
  let finalPublic : FinalPublicWitness backend params w_45_0 w_59_0 w_64_0 w_40_0
      (matrixPolynomial [MxxIR.roundDiv params.diamond_modulus 2]) w_9_0 w_36_0 := {
    base := w_3_0
    uniform := w_7_0
    target := w_39_0
    gadget := w_60_0
    decoderTarget := w_44_0
    keyTarget := w_58_0
    oneTarget := w_63_0
    decompositionRun := by tauto
    gadgetRun := by tauto
    decoderRows := by rw [← hpublicOne]; tauto
    keyRows := by tauto
    oneRows := by rw [← hpublicOne]; tauto
    decoderEquation := preimageRunsDispatched_equation (by decide) (by decide) (by tauto)
    keyEquation := preimageRunsDispatched_equation (by decide) (by decide) (by tauto)
    oneEquation := preimageRunsDispatched_equation (by decide) (by decide) (by tauto)
    halfEquation := rfl }
  refine ⟨producer, finalPublic, terminal, hterminalAddress, hbase, ?_⟩
  intro i
  obtain ⟨state, position, hstate, hposition, hrows, hequation, hbound⟩ :=
    generated_witness_preimage_link backend hashModel params _ _ _ producer w_0_0 w_2_0 w_2_1
      hterminalIndices hterminalBases w_9_0 w_60_0 i (w_74_0 i) (w_75_0 i) (w_75_1 i)
      (w_76_0 i) (w_77_0 i) (w_78_0 i)
      (by tauto) (by tauto) (by tauto) (by tauto) (by tauto)
  exact ⟨state, position, w_77_0 i, hstate, hposition, hrows, hequation, hbound⟩

#print axioms generated_injector_root

end DiamondGeneratedProof
