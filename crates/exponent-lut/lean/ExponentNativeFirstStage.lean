-- Concrete application proof for fixture 677b51307ba5ca4e9adc09ee98fb65b4d57c3310e41140dcdc210d2a30f9a436.
import ExecutionWitnesses
import RootProgramContext
import RootPrivate
import RootCacheAlignment
import RootProgramKeys
import RootProgramSetup
import RootReduction
import ExponentNativeProgram

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000
namespace ExponentGeneratedNativeProof
open Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution
open scoped BigOperators

theorem first_stage {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state (6 * (denseRegularGain RefreshBackend.layout0 32 2 * 6 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)) ∧
      state.ciphertext = (cw.w_121_0) ∧ state.publicKey = (pw.w_120_0) lane ∧
      StridedMonomial 2 16 state.message ∧ state.mask = vw.w_2_0 ∧ state.payload = pw.w_101_0 := by
  classical
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  let cw := programWitness runs lane
  let pin := preprocessingInputs runs.2.1
  let vin := verificationInputs runs.2.2.1
  let ci : ExponentGeneratedProgramRoot.RootInput := (vw.w_40_0, vin.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vin.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, vw.w_17_0, ())
  have pf : RefreshPreprocessing.generatedRoot.body RefreshBackend.backend hashModel preprocessing_params pin execution.preprocessing pw := preprocessingFacts runs
  have vf : RefreshVerification.generatedRoot.body RefreshBackend.backend hashModel verification_params vin execution.verification vw := verificationFacts runs
  have cf : RefreshVerification.parallel_generatedRoot_56.body RefreshBackend.backend hashModel verification_params lane.val ci _ cw := programFacts runs lane
  have pre6 : execution.preprocessing.2.2.2.2.2.2.1 = pw.w_104_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.1 = pw.w_107_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre28 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_712_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre29 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_101_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre30 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_93_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  let g := (vw.w_11_0)
  let layout := RefreshBackend.layout0
  let gain := denseRegularGain layout 32 2
  have lookup : RefreshBackend.backend.regularLayout 1329227985420794846003957960673218177 32 = some layout := by rfl
  have hg := vf.2.2.2.2.1
  have initial := ExponentGeneratedInputsRoot.initial_program_state_0 vw vf
  let st0 := initial.state
  have initialMessage : st0.message = 1 := by simp [st0,ProgramDerivation.state,MxxRuntime.matrixPolynomial]
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (show SignedMonomial st0.message from initialMessage ▸ signedMonomial_one (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_0 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_95 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 6) : Fin (7680 * 6) := ⟨lane.val * 6 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 6, ∃ action : ErrorMatrix 32 60 60, (pw.w_107_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 6, (cw.w_101_0) i = (pw.w_107_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_99] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_101_0 i = (execution.preprocessing.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have initialKey : st0.publicKey = (pw.w_94_0) lane := by
    have repeated := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 lane
    change execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_94_0 lane
    exact pre30.trans repeated.symm
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_101_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨0+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := pre29
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    simpa only [st0,ProgramDerivation.state] using publicPayload i
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_0 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    rcases valid with ⟨selectors,bytes,monomials⟩
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_0 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨0+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift 6 6 gain gain 6
    initial publicDerived (by rfl) publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; simpa only [st0,ProgramDerivation.state] using publicMask i
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * 6 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.1
  have initialStrided : StridedMonomial 2 16 st0.message := by
    refine ⟨⟨0, by decide⟩, ?_⟩
    simpa using initialMessage
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨0 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,rfl,?_⟩
  · simpa only [gain,layout] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_0 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 0 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_94,
        RefreshVerification.parallel_parallel_generatedRoot_56_94.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 0 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
        apply Fin.ext
        dsimp only
        clear * - lane i localFact globalFact li gi
        omega
      rw [publicKey i]
      have aligned := ExponentGeneratedProgramSetup.public_inputs_cache pw pf g hg (positions i)
      rw [positionEq] at aligned
      change g * (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (positions i) = _
      rw [pre28, positionEq]
      exact aligned
    simp_rw [selectedKeys]
    exact (ExponentGeneratedProgramKeys.key_after_0 pw pf lane).symm

  · exact pre29


/-- The first reduced online ciphertext satisfies the native integer error bound. -/
theorem first_stage_approx {hashModel : MxxRuntime.HashModel} {external : ExternalInputs}
    {execution : Execution} (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ message : ExactPoly 1329227985420794846003957960673218177 32,
      StridedMonomial 2 16 message ∧
      Approx cw.w_121_0 (vw.w_2_0 * (pw.w_120_0 lane) - message • (pw.w_101_0 * vw.w_11_0))
        (6 * (denseRegularGain RefreshBackend.layout0 32 2 * 6 + 6 +
          denseRegularGain RefreshBackend.layout0 32 2 * 6)) := by
  obtain ⟨state, derived, ciphertext, key, message, mask, payload⟩ := first_stage runs lane
  refine ⟨state.message, message, ?_⟩
  have invariant := derived.sound (by decide : 0 < 32)
  simpa only [ProgramState.Invariant, ciphertext, key, mask, payload] using invariant

end ExponentGeneratedNativeProof

#print axioms ExponentGeneratedNativeProof.first_stage

#print axioms ExponentGeneratedNativeProof.first_stage_approx
