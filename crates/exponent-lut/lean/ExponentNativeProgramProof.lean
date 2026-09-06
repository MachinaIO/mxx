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
namespace ExponentGeneratedNativeProgramProof
open Mxx.Primitives MxxExponentLut GeneratedClaim ExponentGeneratedExecution
open scoped BigOperators

abbrev nativeBound0 : Nat := 6
abbrev nativeBound1 : Nat := 6 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound0 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_0 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound1 ∧
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
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_0 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨0+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound0 6 gain gain 6
    initial publicDerived (by rfl) publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; simpa only [st0,ProgramDerivation.state] using publicMask i
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound0 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
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
  · simpa only [gain,layout,nativeBound1] using combinedDerivation
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


abbrev nativeBound2 : Nat := 5 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound1 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_1 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound2 ∧
      state.ciphertext = (cw.w_137_0) ∧ state.publicKey = (pw.w_141_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.1 = pw.w_125_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.1 = pw.w_128_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_0 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_1 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_81 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 5) : Fin (7680 * 5) := ⟨lane.val * 5 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 5, ∃ action : ErrorMatrix 32 60 60, (pw.w_128_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 5, (cw.w_87_0) i = (pw.w_128_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_85] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_87_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_87_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨6+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_1 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_1 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨6+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound1 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound1 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.1
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨6 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound2] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_1 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 6 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_80,
        RefreshVerification.parallel_parallel_generatedRoot_56_80.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 6 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_1 pw pf lane).symm

  · exact initialPayload


abbrev nativeBound3 : Nat := 5 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound2 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_2 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound3 ∧
      state.ciphertext = (cw.w_153_0) ∧ state.publicKey = (pw.w_162_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.1 = pw.w_146_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_149_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_1 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_2 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_67 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 5) : Fin (7680 * 5) := ⟨lane.val * 5 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 5, ∃ action : ErrorMatrix 32 60 60, (pw.w_149_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 5, (cw.w_73_0) i = (pw.w_149_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_71] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_73_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_73_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨11+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_2 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_2 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨11+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound2 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound2 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.2.1
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨11 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound3] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_2 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 11 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_66,
        RefreshVerification.parallel_parallel_generatedRoot_56_66.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 11 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_2 pw pf lane).symm

  · exact initialPayload


abbrev nativeBound4 : Nat := 7 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound3 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_3 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound4 ∧
      state.ciphertext = (cw.w_169_0) ∧ state.publicKey = (pw.w_183_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_167_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_170_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_2 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_3 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_53 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 7) : Fin (7680 * 7) := ⟨lane.val * 7 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 7, ∃ action : ErrorMatrix 32 60 60, (pw.w_170_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 7, (cw.w_59_0) i = (pw.w_170_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_57] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_59_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_59_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨16+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_3 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_3 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨16+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound3 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound3 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.2.2.1
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨16 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound4] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_3 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 16 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_52,
        RefreshVerification.parallel_parallel_generatedRoot_56_52.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 16 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_3 pw pf lane).symm

  · exact initialPayload


abbrev nativeBound5 : Nat := 5 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound4 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_4 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound5 ∧
      state.ciphertext = (cw.w_185_0) ∧ state.publicKey = (pw.w_204_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_188_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_191_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_3 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_4 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_39 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 5) : Fin (7680 * 5) := ⟨lane.val * 5 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 5, ∃ action : ErrorMatrix 32 60 60, (pw.w_191_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 5, (cw.w_45_0) i = (pw.w_191_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_43] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_45_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_45_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨23+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_4 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_4 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨23+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound4 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound4 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.2.2.2.1
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨23 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound5] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_4 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 23 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_38,
        RefreshVerification.parallel_parallel_generatedRoot_56_38.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 23 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_4 pw pf lane).symm

  · exact initialPayload


abbrev nativeBound6 : Nat := 5 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound5 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_5 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound6 ∧
      state.ciphertext = (cw.w_201_0) ∧ state.publicKey = (pw.w_225_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_209_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_212_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_4 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_5 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_25 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 5) : Fin (7680 * 5) := ⟨lane.val * 5 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 5, ∃ action : ErrorMatrix 32 60 60, (pw.w_212_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 5, (cw.w_31_0) i = (pw.w_212_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_29] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_31_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_31_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨28+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_5 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_5 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨28+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound5 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound5 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.2.2.2.2.1
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨28 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound6] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_5 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 28 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_24,
        RefreshVerification.parallel_parallel_generatedRoot_56_24.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 28 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_5 pw pf lane).symm

  · exact initialPayload


abbrev nativeBound7 : Nat := 4 * (denseRegularGain RefreshBackend.layout0 32 2 * nativeBound6 + 6 + denseRegularGain RefreshBackend.layout0 32 2 * 6)
theorem stage_6 {hashModel : MxxRuntime.HashModel} {external : ExternalInputs} {execution : Execution}
    (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    let cw := programWitness runs lane
    ∃ state : ProgramState 1329227985420794846003957960673218177 32 2 60,
      ProgramDerivation (vw.w_11_0) state nativeBound7 ∧
      state.ciphertext = (cw.w_213_0) ∧ state.publicKey = (pw.w_242_0) lane ∧
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
  have pre6 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_230_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
  have pre7 : execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = pw.w_233_0 :=
    congrArg (fun output : ExponentGeneratedPrivateRoot.RootOutput => output.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (preprocessingOutput runs)
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
  obtain ⟨st0, initial, initialCipher, initialKey, initialStrided, initialMask, initialPayload⟩ := stage_5 runs lane
  obtain ⟨initialLift, initialEquation, initialNorm⟩ := signedMonomial_integer_lift
    (by decide : 0 < 32) (initialStrided.signed (by decide) (by decide))
  have privateFacts :=  ExponentGeneratedPrivateRoot.private_bucket_6 pw pf layout lookup g hg
  choose privateError privateAction privateEq privateGain privateCache privateBound using privateFacts
  have publicFacts :=  ExponentGeneratedProgramContext.public_branches_9 vw vf lane.val cw _ cf
  choose positions localPositions publicState publicDerived publicCipher publicMessage publicMask publicPayload publicKey globalIndex localIndex using publicFacts
  let flat (i : Fin 4) : Fin (7680 * 4) := ⟨lane.val * 4 + i.val, by clear * - lane i; omega⟩
  have actions : ∀ i : Fin 4, ∃ action : ErrorMatrix 32 60 60, (pw.w_233_0) (flat i) = reduceMatrix 1329227985420794846003957960673218177 32 60 60 action ∧ ActionGain action gain := by
    intro i
    obtain ⟨decomposition,hd,hout⟩ := pf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (flat i)
    obtain ⟨action,ha,hgain⟩ := gadgetDecomposeRuns_dense_gain layout lookup (by decide) (by decide) hd
    exact ⟨action,hout.trans ha,hgain⟩
  choose publicAction publicEq publicGain using actions
  have onlineCache : ∀ i : Fin 4, (cw.w_15_0) i = (pw.w_233_0) (flat i) := by
    intro i
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    have indexEquation := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_13] at indexEquation
    have same : position = flat i := by apply Fin.ext; dsimp [flat]; clear * - lane i hindex indexEquation; omega
    rw [same] at hselected
    have result := hout.trans hselected
    change cw.w_15_0 i = (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) (flat i) at result
    exact result.trans (congrFun pre7 (flat i))
  have privateOnline : ∀ i, (ci.2.2.2.2.1) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    change (execution.preprocessing.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) i = _
    exact (congrFun pre6 i).trans (privateEq i)
  have publicOnline : ∀ i, (cw.w_15_0) i = reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) := by
    intro i
    exact (onlineCache i).trans (publicEq i)
  have cache : ∀ i, (st0.payload * g) * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) = (external.input_0 ⟨33+i.val,by clear * - lane i; omega⟩ 0 0) • (st0.payload * g) + reduceMatrix 1329227985420794846003957960673218177 32 1 60 (privateError i) := by
    intro i
    have payload : st0.payload = pw.w_101_0 := initialPayload
    rw [payload]
    simpa only [privateEq i] using privateCache i
  have shared : ∀ i, (publicState i).payload = st0.mask := by
    intro i
    exact (publicPayload i).trans initialMask.symm
  have decomposition : ∀ i, g * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (publicAction i) = st0.publicKey * reduceMatrix 1329227985420794846003957960673218177 32 60 60 (privateAction i) := by
    intro i
    rw [← publicEq i,initialKey,← privateEq i]
    exact ExponentGeneratedCacheAlignment.public_cache_aligned_6 pw pf g hg lane i
  have publicStrided : ∀ i, StridedMonomial 2 16 (publicState i).message := by
    intro i
    rw [publicMessage i]
    obtain ⟨selected,hlo,hhi,hget,hout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i
    obtain ⟨position,hindex,hselected⟩ := hget
    obtain ⟨raw,rlo,rhi,rget,rout⟩ := cf.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 position
    obtain ⟨rawPosition,rawIndex,rawSelected⟩ := rget
    rw [hout,hselected]
    change StridedMonomial 2 16 (cw.w_18_0 position 0 0)
    rw [rout,rawSelected]
    change StridedMonomial 2 16 (external.input_2 rawPosition 0 0)
    have valid := runs.1
    simp only [ValidExternals] at valid
    have monomials := valid.2.2
    obtain ⟨exponent,he⟩ := monomials rawPosition
    exact ⟨exponent, he⟩
  have publicSigned := fun i => (publicStrided i).signed (by decide : 0 < 32) (by decide : 32 = 2 * 16)
  have lifts := fun i => signedMonomial_integer_lift (by decide : 0 < 32) (publicSigned i)
  choose publicLift liftEquation liftNorm using lifts
  have branches := ExponentGeneratedProgramRoot.program_family_6 lane.val cw cf g st0 publicState privateAction publicAction
    (fun i => external.input_0 ⟨33+i.val,by clear * - lane i; omega⟩ 0 0) privateError initialLift publicLift nativeBound6 6 gain gain 6
    initial publicDerived initialCipher publicCipher publicMessage privateOnline publicOnline cache initialEquation.symm initialNorm privateBound privateGain shared decomposition
    (fun i => (liftEquation i).symm) liftNorm publicGain
  choose result derived ciphertext resultKey resultMessage resultMask resultPayload using branches
  let combined := ProgramState.familyOutput result st0.mask st0.payload
  have masks : ∀ i, (result i).mask = st0.mask := by intro i; rw [resultMask i]; exact (publicMask i).trans initialMask.symm
  have payloads : ∀ i, (result i).payload = st0.payload := resultPayload
  have combinedDerivation := ProgramDerivation.familyReduce result (fun _ => gain * nativeBound6 + 6 + gain * 6) st0.mask st0.payload derived masks payloads
  obtain ⟨selected, selectorEquation⟩ := runs.1.1.2.2.2.2.2.2.2
  have monomial : StridedMonomial 2 16 combined.message := by
    change StridedMonomial 2 16 (∑ i, (result i).message)
    simp_rw [resultMessage]
    exact native_family_message (by decide) (by decide) st0.message
      (fun i => external.input_0 ⟨33 + i.val, by clear * - i; omega⟩ 0 0)
      (fun i => (publicState i).message) selected initialStrided (publicStrided selected)
      (by simpa using selectorEquation selected)
      (by intro i h; simpa [h] using selectorEquation i)
  refine ⟨combined,?_,?_,?_,monomial,initialMask,?_⟩
  · simpa only [gain,layout,nativeBound7] using combinedDerivation
  · change (∑ i, (result i).ciphertext) = _
    simp_rw [ciphertext]
    exact (ExponentGeneratedReduction.reduction_6 lane.val cw cf).symm
  · change (∑ i, (result i).publicKey) = _
    simp_rw [resultKey,← publicEq]
    -- Both global and bucket indices below come from the same actual gathers.
    have selectedKeys : ∀ i, (publicState i).publicKey = (pw.w_42_0) ⟨lane.val * 37 + 33 + i.val,by clear * - lane i; omega⟩ := by
      intro i
      have localFact := localIndex i
      have globalFact := globalIndex i
      have li := cf.2.2.2.2.2.1 i
      have gi := cf.2.2.2.2.2.2.2.2.2.2.2.1 (localPositions i)
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_2,
        RefreshVerification.parallel_parallel_generatedRoot_56_2.constraints_0] at li
      norm_num [RefreshVerification.parallel_parallel_generatedRoot_56_6,
        RefreshVerification.parallel_parallel_generatedRoot_56_6.constraints_0] at gi
      norm_num [MxxIR.exactDiv] at gi
      have positionEq : positions i = (⟨lane.val * 37 + 33 + i.val, by clear * - lane i; omega⟩ : Fin 284160) := by
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
    exact (ExponentGeneratedProgramKeys.key_after_6 pw pf lane).symm

  · exact initialPayload



theorem native_output_bound : nativeBound7 = 254170737014643844390606300252323864 := by decide

/-- The actual final program output satisfies the existing worst-case report. -/
theorem programOutputApprox {hashModel : MxxRuntime.HashModel} {external : ExternalInputs}
    {execution : Execution} (runs : Runs hashModel external execution) (lane : Fin 7680) :
    let pw := preprocessingWitness runs
    let vw := verificationWitness runs
    ∃ message : ExactPoly 1329227985420794846003957960673218177 32,
      StridedMonomial 2 16 message ∧
      Approx ((vw.w_56_0) lane) (vw.w_2_0 * ((pw.w_242_0) lane) - message • (pw.w_101_0 * vw.w_11_0)) 254170737014643844390606300252323864 := by
  let pw := preprocessingWitness runs
  let vw := verificationWitness runs
  let cw := programWitness runs lane
  obtain ⟨state, derived, ciphertext, key, message, mask, payload⟩ := stage_6 runs lane
  refine ⟨state.message, message, ?_⟩
  have invariant := derived.sound (by decide : 0 < 32)
  have output := (programFacts runs lane).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  change ((vw.w_56_0) lane) = (cw.w_213_0) at output
  change Approx ((vw.w_56_0) lane) _ _
  rw [output]
  simpa only [ProgramState.Invariant, ciphertext, key, mask, payload, native_output_bound] using invariant

end ExponentGeneratedNativeProgramProof
#print axioms ExponentGeneratedNativeProgramProof.stage_0
#print axioms ExponentGeneratedNativeProgramProof.stage_1
#print axioms ExponentGeneratedNativeProgramProof.stage_2
#print axioms ExponentGeneratedNativeProgramProof.stage_3
#print axioms ExponentGeneratedNativeProgramProof.stage_4
#print axioms ExponentGeneratedNativeProgramProof.stage_5
#print axioms ExponentGeneratedNativeProgramProof.stage_6
#print axioms ExponentGeneratedNativeProgramProof.programOutputApprox
