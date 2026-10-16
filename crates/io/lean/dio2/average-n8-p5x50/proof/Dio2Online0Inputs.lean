import Dio2Online0Assembly0
import Dio2Import
import Dio2PreprocessSwitch
import Dio2PreprocessRhs

/-!
# Claim-level inputs of the first online stage

The adapters in this module use the actual initial, import, preprocessing, and online runs.
In particular, no hypothesis asserts a relation on an intermediate ciphertext. The remaining
plaintext conditions concern the external label payloads and selector tables, or bounds and
table identities on their plaintext products. The refresh residue and `Noise0` are separate.

The generated claim connects sigma readout to preprocessing `readout`, sigma sampling to
`sampler`, fresh readout to `sigmaReadout`, and smudge readout to `smudge`. These connections
are obtained from the generated input/output tuples, not assumed as additional link premises.
-/

set_option maxHeartbeats 64000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof RefreshTopProof.Gen
  RefreshTopProof.GenIface RefreshTopProof.GenInst

namespace Online0Claim

open W_Stage_online_0

/-- The preprocessing input tuple in the composed claim, including all external selector tables. -/
noncomputable def preprocessInputs (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) :=
  (external.input_1, execution.«stage_0», external.input_2, external.input_33,
    external.input_34, external.input_25, external.input_24, external.input_16,
    external.input_40, external.input_17, external.input_40, external.input_27,
    external.input_26, external.input_18, external.input_39, external.input_19,
    external.input_39, external.input_29, external.input_28, external.input_20,
    external.input_38, external.input_21, external.input_38, ())

/-- The composed claim runs preprocessing on exactly `preprocessInputs`. -/
theorem preprocess_runs {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    W_Stage_preprocess_0_0.generatedRoot Backend.backend model W_GeneratedClaim.stage_6_params
      (preprocessInputs external execution) execution.«stage_6» :=
  h.2.2.2.2.2.2.2.1

/-- The four contexts and their actual claim wiring, with no additional execution assumptions. -/
structure Context (model : HashModel) (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) where
  initial : W_Stage_initial.generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_4_params
  imported : W_Stage_import_0.generatedRoot.Ctx W_GeneratedClaim.stage_10_params
  preprocess : W_Stage_preprocess_0_0.generatedRoot.Ctx Backend.backend model
    W_GeneratedClaim.stage_6_params
  online : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params
  initial_inputs : initial.inputs = importClaimInitialInputs external execution
  imported_inputs : imported.inputs = importClaim0Inputs external execution
  preprocess_inputs : preprocess.inputs = preprocessInputs external execution
  online_inputs : online.inputs = importClaimOnline0Inputs external execution
  initial_body : W_Stage_initial.generatedRoot.body Backend.backend model
    W_GeneratedClaim.stage_4_params initial.inputs execution.«stage_4» initial.witness
  imported_facts : W_Stage_import_0.generatedRoot.Facts imported execution.«stage_10»
  preprocess_facts : W_Stage_preprocess_0_0.generatedRoot.Facts preprocess execution.«stage_6»
  online_facts : generatedRoot.Facts online execution.«stage_11»

/-- Every claim run supplies all four contexts and their facts simultaneously. -/
theorem contexts {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    Nonempty (Context model external execution) := by
  obtain ⟨wi, hi⟩ := importClaimInitialInputs_runs h
  obtain ⟨wm, hm⟩ := importClaim0Inputs_runs h
  obtain ⟨wp, hp⟩ := preprocess_runs h
  obtain ⟨wo, ho⟩ := importClaimOnline0Inputs_runs h
  exact ⟨⟨⟨_, wi⟩, ⟨_, wm⟩, ⟨_, wp⟩, ⟨_, wo⟩, rfl, rfl, rfl, rfl, hi,
    W_Stage_import_0.generatedRoot.facts ⟨_, wm⟩ hm,
    W_Stage_preprocess_0_0.generatedRoot.facts ⟨_, wp⟩ hp, generatedRoot.facts ⟨_, wo⟩ ho⟩⟩

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution} (C : Context model external execution)

/-- Every imported row has its initial plaintext relation with error bound one. -/
theorem Context.rel {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (port : Import0Port) (member : Fin port.members) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25
      port.root.prefix (port.root.tags member))
      (fun j ↦ reducePoly modulus_0 8 (initialImportValue C.initial port.root member *
        unitPair (initialImportMask sm tp port.root member) j) • generatedRoot.w_57_0 C.online)
      (import0OnlineRows C.online port member) 1 := by
  apply import0_claim_online_rel C.initial C.imported C.online external execution C.initial_body
    C.imported_facts C.initial_inputs C.imported_inputs C.online_inputs
    (by simpa only [C.initial_inputs, importClaimInitialInputs] using hmask)
    (by simpa only [C.initial_inputs, importClaimInitialInputs] using hpay)
    C.online_facts.c1.h57_0 port member

/-- Every imported payload is the message specified by its external coefficient and exponent. -/
theorem Context.payload (label : Import0Label) (member : Fin label.members) :
    import0OnlinePayload C.online label member = fun _ _ ↦ reducePoly modulus_0 8
      (message (import0Coefficient C.imported label member)
        (import0Exponent C.imported label member)) := by
  rw [import0_claim_payload_wiring C.imported C.online external execution C.imported_facts
    C.online_inputs label member]
  exact import0_payload_eq C.imported C.imported_facts label member

/-- The preprocessing secret is the obfuscator's plaintext secret, not an online premise. -/
theorem Context.secret {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    W_Stage_preprocess_0_0.generatedRoot.w_1_0 C.preprocess =
      fun _ _ ↦ reducePoly modulus_0 8 tp := by
  funext row column
  fin_cases row
  fin_cases column
  simpa only [W_Stage_preprocess_0_0.generatedRoot.w_1_0, C.preprocess_inputs,
    preprocessInputs] using hpay


/-- Preprocessing discharges `σswitch` at scale one and zero switching error. -/
theorem Context.σswitch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) (0)) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (tp)) •
        ![![generatedRoot.w_9_0 C.online, generatedRoot.w_10_0 C.online],
          ![generatedRoot.w_11_0 C.online, generatedRoot.w_12_0 C.online]] 0 j i +
      ![![generatedRoot.w_9_0 C.online, generatedRoot.w_10_0 C.online],
        ![generatedRoot.w_11_0 C.online, generatedRoot.w_12_0 C.online]] 1 j i =
    ((PreprocessSwitch.readoutCoefficients) i * reducePoly modulus_0 8 ((1) * unitPair (tp) j)) • generatedRoot.w_57_0 C.online +
      reduceMatrix modulus_0 8 1 25 (e j) := by
  have hs := PreprocessSwitch.preprocess0_readout_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_9_0, generatedRoot.w_10_0, generatedRoot.w_11_0, generatedRoot.w_12_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (PreprocessSwitch.switching_errors hs)

/-- Preprocessing discharges `σswitchS` at scale one and zero switching error. -/
theorem Context.σswitchS {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) (0)) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (tp)) •
        ![![generatedRoot.w_2_0 C.online, generatedRoot.w_3_0 C.online],
          ![generatedRoot.w_4_0 C.online, generatedRoot.w_5_0 C.online]] 0 j i +
      ![![generatedRoot.w_2_0 C.online, generatedRoot.w_3_0 C.online],
        ![generatedRoot.w_4_0 C.online, generatedRoot.w_5_0 C.online]] 1 j i =
    ((PreprocessSwitch.samplerCoefficients) i * reducePoly modulus_0 8 ((1) * unitPair (tp) j)) • generatedRoot.w_57_0 C.online +
      reduceMatrix modulus_0 8 1 25 (e j) := by
  have hs := PreprocessSwitch.preprocess0_sampler_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_2_0, generatedRoot.w_3_0, generatedRoot.w_4_0, generatedRoot.w_5_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (PreprocessSwitch.switching_errors hs)

/-- Preprocessing discharges `fSwitch` at scale one and zero switching error. -/
theorem Context.fSwitch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) (0)) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (tp)) •
        ![![generatedRoot.w_1164_0 C.online, generatedRoot.w_1165_0 C.online],
          ![generatedRoot.w_1166_0 C.online, generatedRoot.w_1167_0 C.online]] 0 j i +
      ![![generatedRoot.w_1164_0 C.online, generatedRoot.w_1165_0 C.online],
        ![generatedRoot.w_1166_0 C.online, generatedRoot.w_1167_0 C.online]] 1 j i =
    ((PreprocessSwitch.sigmaReadoutCoefficients) i * reducePoly modulus_0 8 ((1) * unitPair (tp) j)) • generatedRoot.w_57_0 C.online +
      reduceMatrix modulus_0 8 1 25 (e j) := by
  have hs := PreprocessSwitch.preprocess0_sigmaReadout_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_1164_0, generatedRoot.w_1165_0, generatedRoot.w_1166_0, generatedRoot.w_1167_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (PreprocessSwitch.switching_errors hs)

/-- Preprocessing discharges `sSwitch` at scale one and zero switching error. -/
theorem Context.sSwitch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, ∃ e : Fin 2 → ErrorMatrix 8 1 25, (∀ j, CoeffBound (e j) (0)) ∧ ∀ j,
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) (tp)) •
        ![![generatedRoot.w_1221_0 C.online, generatedRoot.w_1222_0 C.online],
          ![generatedRoot.w_1223_0 C.online, generatedRoot.w_1224_0 C.online]] 0 j i +
      ![![generatedRoot.w_1221_0 C.online, generatedRoot.w_1222_0 C.online],
        ![generatedRoot.w_1223_0 C.online, generatedRoot.w_1224_0 C.online]] 1 j i =
    ((PreprocessSwitch.smudgeCoefficients) i * reducePoly modulus_0 8 ((1) * unitPair (tp) j)) • generatedRoot.w_57_0 C.online +
      reduceMatrix modulus_0 8 1 25 (e j) := by
  have hs := PreprocessSwitch.preprocess0_smudge_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_1221_0, generatedRoot.w_1222_0, generatedRoot.w_1223_0, generatedRoot.w_1224_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (PreprocessSwitch.switching_errors hs)

/-- Preprocessing discharges `stSwitch` at scale one and zero switching error. -/
theorem Context.stSwitch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (jc : Fin 2) (i : Fin 4), ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) (tp)) •
        ![![generatedRoot.w_247_0 C.online, generatedRoot.w_249_0 C.online],
          ![generatedRoot.w_251_0 C.online, generatedRoot.w_253_0 C.online]] 0 jc i +
      ![![generatedRoot.w_247_0 C.online, generatedRoot.w_249_0 C.online],
        ![generatedRoot.w_251_0 C.online, generatedRoot.w_253_0 C.online]] 1 jc i =
      ((PreprocessSwitch.stateCoefficients) i * reducePoly modulus_0 8 ((1) * unitPair (tp) jc)) • generatedRoot.w_229_0 C.online +
        reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_state_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  rw [← gadgetMatrixRuns_unique C.online_facts.c1.h57_0 C.online_facts.c6.h229_0]
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_247_0, generatedRoot.w_249_0, generatedRoot.w_251_0, generatedRoot.w_253_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (fun column branch ↦ hs column branch)

/-- Preprocessing discharges `a3switch` at scale one and zero switching error. -/
theorem Context.a3switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 3 (⟨1, rfl⟩ : Odd 3) (tp)) • generatedRoot.w_758_0 C.online 0 +
      generatedRoot.w_762_0 C.online 0 = ((PreprocessSwitch.aut3Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut3_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_758_0, generatedRoot.w_762_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a5switch` at scale one and zero switching error. -/
theorem Context.a5switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 5 (⟨2, rfl⟩ : Odd 5) (tp)) • generatedRoot.w_816_0 C.online 0 +
      generatedRoot.w_820_0 C.online 0 = ((PreprocessSwitch.aut5Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut5_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_816_0, generatedRoot.w_820_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a7switch` at scale one and zero switching error. -/
theorem Context.a7switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 7 (⟨3, rfl⟩ : Odd 7) (tp)) • generatedRoot.w_874_0 C.online 0 +
      generatedRoot.w_878_0 C.online 0 = ((PreprocessSwitch.aut7Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut7_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_874_0, generatedRoot.w_878_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a9switch` at scale one and zero switching error. -/
theorem Context.a9switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 9 (⟨4, rfl⟩ : Odd 9) (tp)) • generatedRoot.w_932_0 C.online 0 +
      generatedRoot.w_936_0 C.online 0 = ((PreprocessSwitch.aut9Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut9_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_932_0, generatedRoot.w_936_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a11switch` at scale one and zero switching error. -/
theorem Context.a11switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 11 (⟨5, rfl⟩ : Odd 11) (tp)) • generatedRoot.w_990_0 C.online 0 +
      generatedRoot.w_994_0 C.online 0 = ((PreprocessSwitch.aut11Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut11_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_990_0, generatedRoot.w_994_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a13switch` at scale one and zero switching error. -/
theorem Context.a13switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 13 (⟨6, rfl⟩ : Odd 13) (tp)) • generatedRoot.w_1048_0 C.online 0 +
      generatedRoot.w_1052_0 C.online 0 = ((PreprocessSwitch.aut13Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut13_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_1048_0, generatedRoot.w_1052_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- Preprocessing discharges `a15switch` at scale one and zero switching error. -/
theorem Context.a15switch {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e (0) ∧
    reducePoly modulus_0 8 (substitution 15 (⟨7, rfl⟩ : Odd 15) (tp)) • generatedRoot.w_1106_0 C.online 0 +
      generatedRoot.w_1110_0 C.online 0 = ((PreprocessSwitch.aut15Coefficients 0) * reducePoly modulus_0 8 ((1) * unitPair (tp) 1)) •
        generatedRoot.w_57_0 C.online + reduceMatrix modulus_0 8 1 25 e := by
  have hs := PreprocessSwitch.preprocess0_aut15_switching C.preprocess C.preprocess_facts
    (by norm_num [modulus_0]) (C.secret hpay) C.online_facts.c1.h57_0
  simpa only [Dio2Proof.LutKey.fam4, generatedRoot.w_1106_0, generatedRoot.w_1110_0, C.online_inputs,
    importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs] using (hs 1 0)

/-- The actual claim wiring discharges selector RHS `0`, with explicit zero error. -/
theorem Context.rhs0 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 72000) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPre1
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1233_0 C.online i, generatedRoot.w_1234_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_16 i) (external.input_40 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs0 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_1233_0, generatedRoot.w_1234_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_77_0,
    W_Stage_preprocess_0_0.generatedRoot.w_78_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `1`, with explicit zero error. -/
theorem Context.rhs1 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 72000) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPre1b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1237_0 C.online i, generatedRoot.w_1238_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_17 i) (external.input_40 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs1 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_1237_0, generatedRoot.w_1238_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_81_0,
    W_Stage_preprocess_0_0.generatedRoot.w_82_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `2`, with explicit zero error. -/
theorem Context.rhs2 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 1) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPreY
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_298_0 C.online i, generatedRoot.w_308_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_33 i) (external.input_34 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs2 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_298_0, generatedRoot.w_308_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_34_0,
    W_Stage_preprocess_0_0.generatedRoot.w_35_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `3`, with explicit zero error. -/
theorem Context.rhs3 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPre0
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_27_0 C.online i, generatedRoot.w_28_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_18 i) (external.input_39 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs3 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_27_0, generatedRoot.w_28_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_114_0,
    W_Stage_preprocess_0_0.generatedRoot.w_115_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `4`, with explicit zero error. -/
theorem Context.rhs4 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPre0b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_31_0 C.online i, generatedRoot.w_32_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_19 i) (external.input_39 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs4 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_31_0, generatedRoot.w_32_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_118_0,
    W_Stage_preprocess_0_0.generatedRoot.w_119_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `5`, with explicit zero error. -/
theorem Context.rhs5 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 14400) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPreF0
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1179_0 C.online i, generatedRoot.w_1180_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_20 i) (external.input_38 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs5 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_1179_0, generatedRoot.w_1180_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_159_0,
    W_Stage_preprocess_0_0.generatedRoot.w_160_0, C.preprocess_inputs, preprocessInputs] using hs

/-- The actual claim wiring discharges selector RHS `6`, with explicit zero error. -/
theorem Context.rhs6 {tp : ErrorPoly 8}
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ (i : Fin 14400) (j : Fin 2), reducePoly modulus_0 8 tp •
        topPairG model (generatedRoot.w_26_0 C.online) modulus_0 25 rhsPreF0b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_1183_0 C.online i, generatedRoot.w_1184_0 C.online i] j =
      reducePoly modulus_0 8
        (message (external.input_21 i) (external.input_38 i) * unitPair tp j) •
          generatedRoot.w_57_0 C.online +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  have hs := Preprocess0.rhs6 C.preprocess C.preprocess_facts (C.secret hpay)
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self (by norm_num [modulus_0]) _)
  simpa only [generatedRoot.w_26_0, generatedRoot.w_1183_0, generatedRoot.w_1184_0,
    C.online_inputs, importClaimOnline0Inputs, C.preprocess_facts.c17.hout,
    W_Stage_preprocess_0_0.generatedRoot.outs, W_Stage_preprocess_0_0.generatedRoot.w_0_0,
    W_Stage_preprocess_0_0.generatedRoot.w_163_0,
    W_Stage_preprocess_0_0.generatedRoot.w_164_0, C.preprocess_inputs, preprocessInputs] using hs


/-- The two root-label plaintexts are determined solely by the obfuscator's label exponents. -/
noncomputable def labelValue (external : GeneratedClaim.ExternalInputs) (i : Fin 2) : ErrorPoly 8 :=
  if i.val = 0 then message 1 (external.input_0 0) else 1

/-- The four input-label plaintexts are determined solely by external label exponents. -/
noncomputable def inputValue (external : GeneratedClaim.ExternalInputs) (i : Fin 4) : ErrorPoly 8 :=
  message 1 (external.input_0 ⟨i.val + 1, by omega⟩)

/-- Initial-root correctness and import wiring discharge `Online0Inputs.help` with bound one. -/
theorem Context.help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas8 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_13_0 C.online i, generatedRoot.w_14_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .helperLabel i
  fin_cases i
  all_goals
    simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.helpS` with bound one. -/
theorem Context.helpS {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 8, Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas8 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_6_0 C.online i, generatedRoot.w_8_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .helperSigma i
  fin_cases i
  all_goals
    simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.stHelp` with bound one. -/
theorem Context.stHelp {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 4, Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre
      [.integer (sigmaOf sigmas4 i)])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_255_0 C.online i, generatedRoot.w_256_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .helperState i
  fin_cases i
  all_goals
    simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.one` with bound one. -/
theorem Context.one {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelPre 1)
    (fun j ↦ reducePoly modulus_0 8 (1 * unitPair sm j) • generatedRoot.w_57_0 C.online)
    ![(generatedRoot.w_16_0 C.online 1), generatedRoot.w_17_0 C.online 1] 1 := by
  have h := C.rel hmask hpay .labelSigma (1 : Fin 2)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.oneS` with bound one. -/
theorem Context.oneS {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelPre
      ((1 : Fin 2) : Nat))
    (fun j ↦ reducePoly modulus_0 8 (1 * unitPair sm j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_257_0 C.online 1, generatedRoot.w_262_0 C.online 1] 1 := by
  have h := C.rel hmask hpay .labelState (1 : Fin 2)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.lab` with bound one. -/
theorem Context.lab {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 4, Rel sm
    (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelInPre (i : Nat))
    (fun j ↦ reducePoly modulus_0 8 ((inputValue external) i * unitPair sm j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_372_0 C.online i, generatedRoot.w_377_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .labelInput i
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.priv` with bound one. -/
theorem Context.priv {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (rootKey (n := 8) model (generatedRoot.w_7_0 C.online) privPre0 [] modulus_0 25)
    (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_229_0 C.online)
    ![generatedRoot.w_21_0 C.online, generatedRoot.w_25_0 C.online] 1 := by
  have h := C.rel hmask hpay .privateSigma (0 : Fin 1)
  rw [gadgetMatrixRuns_unique C.online_facts.c1.h57_0 C.online_facts.c6.h229_0] at h
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a3help` with bound one. -/
theorem Context.a3help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 3])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 3 (⟨1, rfl⟩ : Odd 3) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_764_0 C.online 0, generatedRoot.w_765_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut3 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a5help` with bound one. -/
theorem Context.a5help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 5])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 5 (⟨2, rfl⟩ : Odd 5) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_822_0 C.online 0, generatedRoot.w_823_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut5 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a7help` with bound one. -/
theorem Context.a7help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 7])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 7 (⟨3, rfl⟩ : Odd 7) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_880_0 C.online 0, generatedRoot.w_881_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut7 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a9help` with bound one. -/
theorem Context.a9help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 9])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 9 (⟨4, rfl⟩ : Odd 9) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_938_0 C.online 0, generatedRoot.w_939_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut9 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a11help` with bound one. -/
theorem Context.a11help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 11])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 11 (⟨5, rfl⟩ : Odd 11) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_996_0 C.online 0, generatedRoot.w_997_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut11 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a13help` with bound one. -/
theorem Context.a13help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 13])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 13 (⟨6, rfl⟩ : Odd 13) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_1054_0 C.online 0, generatedRoot.w_1055_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut13 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Initial-root correctness and import wiring discharge `Online0Inputs.a15help` with bound one. -/
theorem Context.a15help {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 helperPre [.integer 15])
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
      unitPair (substitution 15 (⟨7, rfl⟩ : Odd 15) sm) j) • generatedRoot.w_57_0 C.online)
    ![generatedRoot.w_1112_0 C.online 0, generatedRoot.w_1113_0 C.online 0] 1 := by
  have h := C.rel hmask hpay .aut15 (0 : Fin 1)
  simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
      ImportRoot.tags, ImportRoot.sigma, import0OnlineRows, inputValue,
      C.initial_inputs, importClaimInitialInputs, labelKeysG, import_rootKey_eq, sigmas8, sigmas4,
      sigmaOf, MxxIR.selectInt, ImportRoot.members, privPre0, privatePrefix] using h

/-- Imported `labelSigma` rows encode the externally determined root-label plaintexts. -/
theorem Context.labelSigma {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelPre i)
      (fun j ↦ reducePoly modulus_0 8 (labelValue external i * unitPair sm j) •
        generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_16_0 C.online i, generatedRoot.w_17_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .labelSigma i
  fin_cases i <;> simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, labelValue, C.initial_inputs,
    importClaimInitialInputs, labelKeysG, ImportRoot.members] using h

/-- Imported `labelFresh` rows encode the externally determined root-label plaintexts. -/
theorem Context.labelFresh {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelPre i)
      (fun j ↦ reducePoly modulus_0 8 (labelValue external i * unitPair sm j) •
        generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_1169_0 C.online i, generatedRoot.w_1170_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .labelFresh i
  fin_cases i <;> simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, labelValue, C.initial_inputs,
    importClaimInitialInputs, labelKeysG, ImportRoot.members] using h

/-- Imported `labelState` rows encode the externally determined root-label plaintexts. -/
theorem Context.labelState {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    ∀ i : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_7_0 C.online) modulus_0 25 labelPre i)
      (fun j ↦ reducePoly modulus_0 8 (labelValue external i * unitPair sm j) •
        generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_257_0 C.online i, generatedRoot.w_262_0 C.online i] 1 := by
  intro i
  have h := C.rel hmask hpay .labelState i
  fin_cases i <;> simpa [initialImportValue, initialImportMask, Import0Port.root, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, labelValue, C.initial_inputs,
    importClaimInitialInputs, labelKeysG, ImportRoot.members] using h

/-- External payload agreement discharges the `sigma` label-payload premise. -/
theorem Context.pay_sigma
    (hag : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i) :
    ∀ i : Fin 2, generatedRoot.w_22_0 C.online i =
      fun _ _ ↦ reducePoly modulus_0 8 (labelValue external i) := by
  intro i
  have h := C.payload .sigma i
  simpa only [import0OnlinePayload, import0Coefficient, import0Exponent,
    W_Stage_import_0.generatedRoot.w_9_0,
    W_Stage_import_0.generatedRoot.w_10_0,
    C.imported_inputs, importClaim0Inputs, hag] using h

/-- External payload agreement discharges the `fresh` label-payload premise. -/
theorem Context.pay_fresh
    (hag : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i) :
    ∀ i : Fin 2, generatedRoot.w_1175_0 C.online i =
      fun _ _ ↦ reducePoly modulus_0 8 (labelValue external i) := by
  intro i
  have h := C.payload .fresh i
  simpa only [import0OnlinePayload, import0Coefficient, import0Exponent,
    W_Stage_import_0.generatedRoot.w_12_0,
    W_Stage_import_0.generatedRoot.w_13_0,
    C.imported_inputs, importClaim0Inputs, hag] using h

/-- External payload agreement discharges the `state` label-payload premise. -/
theorem Context.pay_state
    (hag : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i) :
    ∀ i : Fin 2, generatedRoot.w_276_0 C.online i =
      fun _ _ ↦ reducePoly modulus_0 8 (labelValue external i) := by
  intro i
  have h := C.payload .state i
  simpa only [import0OnlinePayload, import0Coefficient, import0Exponent,
    W_Stage_import_0.generatedRoot.w_4_0,
    W_Stage_import_0.generatedRoot.w_5_0,
    C.imported_inputs, importClaim0Inputs, hag] using h

/-- External input-label payload agreement discharges `Online0Inputs.pay`. -/
theorem Context.pay
    (hag : ∀ i : Fin 4, message (external.input_10 i) (external.input_11 i) =
      inputValue external i) :
    ∀ i : Fin 4, generatedRoot.w_391_0 C.online i =
      fun _ _ ↦ reducePoly modulus_0 8 (inputValue external i) := by
  intro i
  have h := C.payload .input i
  simpa only [import0OnlinePayload, import0Coefficient, import0Exponent,
    W_Stage_import_0.generatedRoot.w_17_0, W_Stage_import_0.generatedRoot.w_18_0,
    C.imported_inputs, importClaim0Inputs, hag] using h


/-- The first private seed is the imported private unit; its keys are sampled by the online run. -/
theorem Context.sigma_seed {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm ![![generatedRoot.w_19_0 C.online, generatedRoot.w_23_0 C.online],
      ![generatedRoot.w_20_0 C.online, generatedRoot.w_24_0 C.online]]
      (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_21_0 C.online, generatedRoot.w_25_0 C.online] 1 := by
  have h := C.rel hmask hpay .privateSigma (0 : Fin 1)
  have hk : keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 privatePrefix [] =
      ![![generatedRoot.w_19_0 C.online, generatedRoot.w_23_0 C.online],
        ![generatedRoot.w_20_0 C.online, generatedRoot.w_24_0 C.online]] := by
    funext row column
    fin_cases row <;> fin_cases column
    · exact (hashSample_eq C.online_facts.c0.h19_0).symm
    · exact (hashSample_eq C.online_facts.c0.h23_0).symm
    · exact (hashSample_eq C.online_facts.c0.h20_0).symm
    · exact (hashSample_eq C.online_facts.c0.h24_0).symm
  simpa only [Import0Port.root, initialImportValue, initialImportMask, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, hk] using h

/-- The imported private unit supplies this bucket seed, including the actual sampled keys. -/
theorem Context.fresh_seed {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm ![![generatedRoot.w_1172_0 C.online, generatedRoot.w_1176_0 C.online],
      ![generatedRoot.w_1173_0 C.online, generatedRoot.w_1177_0 C.online]]
      (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_1174_0 C.online, generatedRoot.w_1178_0 C.online] 1 := by
  have h := C.rel hmask hpay .privateFresh (0 : Fin 1)
  have hk : keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 privatePrefix [] =
      ![![generatedRoot.w_1172_0 C.online, generatedRoot.w_1176_0 C.online],
        ![generatedRoot.w_1173_0 C.online, generatedRoot.w_1177_0 C.online]] := by
    funext row column
    fin_cases row <;> fin_cases column
    · exact (hashSample_eq C.online_facts.c45.h1172_0).symm
    · exact (hashSample_eq C.online_facts.c45.h1176_0).symm
    · exact (hashSample_eq C.online_facts.c45.h1173_0).symm
    · exact (hashSample_eq C.online_facts.c45.h1177_0).symm
  simpa only [Import0Port.root, initialImportValue, initialImportMask, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, hk] using h

/-- The imported private unit supplies this bucket seed, including the actual sampled keys. -/
theorem Context.smudge_seed {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm ![![generatedRoot.w_1227_0 C.online, generatedRoot.w_1230_0 C.online],
      ![generatedRoot.w_1228_0 C.online, generatedRoot.w_1231_0 C.online]]
      (fun j ↦ reducePoly modulus_0 8 (1 * unitPair tp j) • generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_1229_0 C.online, generatedRoot.w_1232_0 C.online] 1 := by
  have h := C.rel hmask hpay .privateRoot (0 : Fin 1)
  have hk : keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 privatePrefix [] =
      ![![generatedRoot.w_1227_0 C.online, generatedRoot.w_1230_0 C.online],
        ![generatedRoot.w_1228_0 C.online, generatedRoot.w_1231_0 C.online]] := by
    funext row column
    fin_cases row <;> fin_cases column
    · exact (hashSample_eq C.online_facts.c46.h1227_0).symm
    · exact (hashSample_eq C.online_facts.c46.h1230_0).symm
    · exact (hashSample_eq C.online_facts.c46.h1228_0).symm
    · exact (hashSample_eq C.online_facts.c46.h1231_0).symm
  simpa only [Import0Port.root, initialImportValue, initialImportMask, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, hk] using h

/-- The bound for one selector cell, with imported label and zero-error selector RHS. -/
def selectorCellBound (seedBound : Nat) : Nat :=
  (25 + 25) * 8 * ((25 + 25) * 8 * 1 * 2048 + 8 * 1 * seedBound) * 2048 + 8 * 1 * 0

/-- The bound for one binary selector bucket. -/
def selectorBucketBound (seedBound : Nat) : Nat := 2 * selectorCellBound seedBound

/-- The two `sigma` PRF buckets discharge the consuming row's `hin`; all residual
hypotheses refer only to external label plaintexts and selector messages. -/
theorem Context.sigma_inputs {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hlabel : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i)
    (hnorm : ∀ k : Fin 2, polyNorm (labelValue external k) ≤ 1)
    (hproduct : ∀ k l : Fin 2,
      polyNorm (labelValue external k * labelValue external l) ≤ 1)
    (sel0 sel1 : Fin 8 → Fin 2)
    (hsel0 : ∀ o j, message (external.input_18 (cellAt o j))
      (external.input_39 (cellAt o j)) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (hsel1 : ∀ o j, message (external.input_19 (cellAt o j))
      (external.input_39 (cellAt o j)) = if j = sel1 o then (1 : ErrorPoly 8) else 0) :
    ∃ key0 key1 : Fin 8 → Fin 2, ∀ o : Fin 8,
      EncRel sm (generatedRoot.w_34_0 C.online o, (generatedRoot.w_34_1 C.online o, (generatedRoot.w_34_2 C.online o, (generatedRoot.w_34_3 C.online o, (generatedRoot.w_34_4 C.online o, (generatedRoot.w_34_5 C.online o, ()))))))
        (fun j ↦ reducePoly modulus_0 8
          (((1 : Nat) : Int) * (labelValue external (key1 o) * labelValue external (key0 o)) *
            unitPair tp j) • generatedRoot.w_57_0 C.online)
        (selectorBucketBound (selectorBucketBound 1)) := by
  have hq : 1 < modulus_0 := by norm_num [modulus_0]
  have hG := C.online_facts.c1.h57_0
  have hself := Dio2Proof.centeredRebaseRuns_self hq (generatedRoot.w_57_0 C.online)
  have hr0 := C.rhs3 hpay
  have hr1 := C.rhs4 hpay
  have hp := C.pay_sigma hlabel
  have ha := C.labelSigma hmask hpay
  choose key0 hkey0 hrow0 using fun o : Fin 8 ↦
    online0_bucket0 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := 1) (fun k ↦ by simpa only [mul_one] using hnorm k)
      hG ha hp hG hself (C.sigma_seed hmask hpay)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr0)
      (PreprocessRhs.selector_zero_bound) sel0 hsel0 o
  choose key1 hkey1 hrow1 using fun o : Fin 8 ↦
    online0_bucket1 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := fun o ↦ labelValue external (key0 o))
      (fun k o ↦ hproduct k (key0 o)) hG ha hp hG hself
      (fun o ↦ by simpa only [mul_one] using hrow0 o)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr1)
      (PreprocessRhs.selector_zero_bound) sel1 hsel1 o
  exact ⟨key0, key1, hrow1⟩

/-- The two `fresh` PRF buckets discharge the consuming row's `hin`; all residual
hypotheses refer only to external label plaintexts and selector messages. -/
theorem Context.fresh_inputs {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hlabel : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i)
    (hnorm : ∀ k : Fin 2, polyNorm (labelValue external k) ≤ 1)
    (hproduct : ∀ k l : Fin 2,
      polyNorm (labelValue external k * labelValue external l) ≤ 1)
    (sel0 sel1 : Fin 7200 → Fin 2)
    (hsel0 : ∀ o j, message (external.input_20 (cellAt o j))
      (external.input_38 (cellAt o j)) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (hsel1 : ∀ o j, message (external.input_21 (cellAt o j))
      (external.input_38 (cellAt o j)) = if j = sel1 o then (1 : ErrorPoly 8) else 0) :
    ∃ key0 key1 : Fin 7200 → Fin 2, ∀ o : Fin 7200,
      EncRel sm (generatedRoot.w_1186_0 C.online o, (generatedRoot.w_1186_1 C.online o, (generatedRoot.w_1186_2 C.online o, (generatedRoot.w_1186_3 C.online o, (generatedRoot.w_1186_4 C.online o, (generatedRoot.w_1186_5 C.online o, ()))))))
        (fun j ↦ reducePoly modulus_0 8
          (((1 : Nat) : Int) * (labelValue external (key1 o) * labelValue external (key0 o)) *
            unitPair tp j) • generatedRoot.w_57_0 C.online)
        (selectorBucketBound (selectorBucketBound 1)) := by
  have hq : 1 < modulus_0 := by norm_num [modulus_0]
  have hG := C.online_facts.c1.h57_0
  have hself := Dio2Proof.centeredRebaseRuns_self hq (generatedRoot.w_57_0 C.online)
  have hr0 := C.rhs5 hpay
  have hr1 := C.rhs6 hpay
  have hp := C.pay_fresh hlabel
  have ha := C.labelFresh hmask hpay
  choose key0 hkey0 hrow0 using fun o : Fin 7200 ↦
    online0_fresh_bucket0 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := 1) (fun k ↦ by simpa only [mul_one] using hnorm k)
      hG ha hp hG hself (C.fresh_seed hmask hpay)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr0)
      (PreprocessRhs.selector_zero_bound) sel0 hsel0 o
  choose key1 hkey1 hrow1 using fun o : Fin 7200 ↦
    online0_fresh_bucket1 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := fun o ↦ labelValue external (key0 o))
      (fun k o ↦ hproduct k (key0 o)) hG ha hp hG hself
      (fun o ↦ by simpa only [mul_one] using hrow0 o)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr1)
      (PreprocessRhs.selector_zero_bound) sel1 hsel1 o
  exact ⟨key0, key1, hrow1⟩

/-- The two `smudge` PRF buckets discharge the consuming row's `hin`; all residual
hypotheses refer only to external label plaintexts and selector messages. -/
theorem Context.smudge_inputs {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hlabel : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i)
    (hnorm : ∀ k : Fin 2, polyNorm (labelValue external k) ≤ 1)
    (hproduct : ∀ k l : Fin 2,
      polyNorm (labelValue external k * labelValue external l) ≤ 1)
    (sel0 sel1 : Fin 36000 → Fin 2)
    (hsel0 : ∀ o j, message (external.input_16 (cellAt o j))
      (external.input_40 (cellAt o j)) = if j = sel0 o then (1 : ErrorPoly 8) else 0)
    (hsel1 : ∀ o j, message (external.input_17 (cellAt o j))
      (external.input_40 (cellAt o j)) = if j = sel1 o then (1 : ErrorPoly 8) else 0) :
    ∃ key0 key1 : Fin 36000 → Fin 2, ∀ o : Fin 36000,
      EncRel sm (generatedRoot.w_1240_0 C.online o, (generatedRoot.w_1240_1 C.online o, (generatedRoot.w_1240_2 C.online o, (generatedRoot.w_1240_3 C.online o, (generatedRoot.w_1240_4 C.online o, (generatedRoot.w_1240_5 C.online o, ()))))))
        (fun j ↦ reducePoly modulus_0 8
          (((1 : Nat) : Int) * (labelValue external (key1 o) * labelValue external (key0 o)) *
            unitPair tp j) • generatedRoot.w_57_0 C.online)
        (selectorBucketBound (selectorBucketBound 1)) := by
  have hq : 1 < modulus_0 := by norm_num [modulus_0]
  have hG := C.online_facts.c1.h57_0
  have hself := Dio2Proof.centeredRebaseRuns_self hq (generatedRoot.w_57_0 C.online)
  have hr0 := C.rhs0 hpay
  have hr1 := C.rhs1 hpay
  have hp := C.pay_state hlabel
  have ha := C.labelState hmask hpay
  choose key0 hkey0 hrow0 using fun o : Fin 36000 ↦
    online0_smudge_bucket0 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := 1) (fun k ↦ by simpa only [mul_one] using hnorm k)
      hG ha hp hG hself (C.smudge_seed hmask hpay)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr0)
      (PreprocessRhs.selector_zero_bound) sel0 hsel0 o
  choose key1 hkey1 hrow1 using fun o : Fin 36000 ↦
    online0_smudge_bucket1 C.online C.online_facts hq layout hlayout hexact
      hnorm (μ := fun o ↦ labelValue external (key0 o))
      (fun k o ↦ hproduct k (key0 o)) hG ha hp hG hself
      (fun o ↦ by simpa only [mul_one] using hrow0 o)
      (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr1)
      (PreprocessRhs.selector_zero_bound) sel1 hsel1 o
  exact ⟨key0, key1, hrow1⟩

/-- The initial state plaintext depends only on the genuine external state bit. -/
noncomputable def stateSeed (external : GeneratedClaim.ExternalInputs) : ErrorPoly 8 :=
  message 1 (external.input_37 * 2)

/-- The state-LUT input plaintext is the initial state times the external state selector. -/
noncomputable def statePlain (external : GeneratedClaim.ExternalInputs) : ErrorPoly 8 :=
  stateSeed external * message (external.input_33 0) (external.input_34 0)

/-- Import wiring supplies the state seed at bound one. -/
theorem Context.state_seed {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp) :
    Rel sm (keyPairG model (generatedRoot.w_7_0 C.online) modulus_0 25 statePre [])
      (fun j ↦ reducePoly modulus_0 8 (stateSeed external * unitPair tp j) •
        generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_275_0 C.online, generatedRoot.w_290_0 C.online] 1 := by
  simpa only [Import0Port.root, initialImportValue, initialImportMask, ImportRoot.prefix,
    ImportRoot.tags, import0OnlineRows, C.initial_inputs, importClaimInitialInputs,
    stateSeed] using C.rel hmask hpay .state (0 : Fin 1)

/-- The composed state cell discharges `Online0Inputs.stEnc`; its only payload hypothesis
is equality of the external public label message with the initial label plaintext. -/
theorem Context.stEnc {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hlabel : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
      labelValue external i)
    (hnorm : polyNorm (stateSeed external) ≤ 1) :
    Rel sm ![![generatedRoot.w_305_0 C.online, generatedRoot.w_315_0 C.online],
      ![generatedRoot.w_306_0 C.online, generatedRoot.w_316_0 C.online]]
      (fun j ↦ reducePoly modulus_0 8
        (((1 : Nat) : Int) * statePlain external * unitPair tp j) •
          generatedRoot.w_57_0 C.online)
      ![generatedRoot.w_304_0 C.online, generatedRoot.w_314_0 C.online]
      (selectorCellBound 1) := by
  have hq : 1 < modulus_0 := by norm_num [modulus_0]
  have hone : polyNorm (1 : ErrorPoly 8) ≤ 1 := by
    simpa only [pow_zero] using polyNorm_Xe (n := 8) (by decide) 0
  have hp := C.pay_state hlabel 1
  have hp' : generatedRoot.w_276_0 C.online 1 =
      fun _ _ ↦ reducePoly modulus_0 8 (1 : ErrorPoly 8) := by
    simpa [labelValue] using hp
  have h := online0_lut_input C.online C.online_facts hq layout hlayout hexact
    (α := 1) hone (by simpa only [one_mul] using hnorm)
    C.online_facts.c1.h57_0 (C.oneS hmask hpay) hp'
    C.online_facts.c1.h57_0
    (Dio2Proof.centeredRebaseRuns_self hq (generatedRoot.w_57_0 C.online))
    (C.state_seed hmask hpay) (C.rhs2 hpay (0 : Fin 1))
    (PreprocessRhs.selector_zero_bound (0 : Fin 1))
  simpa only [selectorCellBound, statePlain, Nat.cast_one, Int.cast_one, one_mul,
    smul_smul, ← map_mul, mul_assoc] using h



/-- Products of integer monomials are integer monomials, without any exponent range premise. -/
theorem monomial_mul {left right : ErrorPoly 8}
    (hleft : ∃ exponent : Nat, left = Xe exponent)
    (hright : ∃ exponent : Nat, right = Xe exponent) :
    ∃ exponent : Nat, left * right = Xe exponent := by
  obtain ⟨leftExp, rfl⟩ := hleft
  obtain ⟨rightExp, rfl⟩ := hright
  exact ⟨leftExp + rightExp, (pow_add _ _ _).symm⟩

/-- An integer monomial satisfies the online cell's norm bound. -/
theorem monomial_small {value : ErrorPoly 8}
    (hvalue : ∃ exponent : Nat, value = Xe exponent) : polyNorm value ≤ 1 := by
  obtain ⟨exponent, rfl⟩ := hvalue
  exact polyNorm_Xe (by decide) exponent

/-- Each root label is a monomial determined only by the external label exponent. -/
theorem label_monomial (external : GeneratedClaim.ExternalInputs) (i : Fin 2) :
    ∃ exponent : Nat, labelValue external i = Xe exponent := by
  by_cases hi : i.val = 0
  · exact ⟨(external.input_0 0 % 16).toNat, by simp [labelValue, hi, message]⟩
  · exact ⟨0, by simp [labelValue, hi]⟩

/-- Each input label is a monomial determined only by its external label exponent. -/
theorem input_monomial (external : GeneratedClaim.ExternalInputs) (i : Fin 4) :
    ∃ exponent : Nat, inputValue external i = Xe exponent := by
  exact ⟨(external.input_0 ⟨i.val + 1, by omega⟩ % 16).toNat,
    by simp [inputValue, message]⟩

/-- The initial state monomial satisfies the online cell's norm bound. -/
theorem state_seed_small (external : GeneratedClaim.ExternalInputs) :
    polyNorm (stateSeed external) ≤ 1 := by
  apply monomial_small
  exact ⟨((external.input_37 * 2) % 16).toNat, by simp [stateSeed, message]⟩

/-- The width-eight helper automorphisms preserve the required integer norm bound. -/
theorem small_sigma8 (value : ErrorPoly 8) (hvalue : polyNorm value ≤ 1) (i : Fin 8) :
    polyNorm (substitution (sigmaOf sigmas8 i).toNat (sigmas8_odd i) value) ≤ 1 := by
  have hcoprime : ∀ i : Fin 8, Nat.Coprime 8 (sigmaOf sigmas8 i).toNat := by decide
  exact (polyNorm_substitution_le (by decide) (sigmas8_odd i) (hcoprime i) value).trans hvalue

/-- The stride-two state helper automorphisms preserve the integer norm bound. -/
theorem small_sigma4 (value : ErrorPoly 8) (hvalue : polyNorm value ≤ 1) (i : Fin 4) :
    polyNorm (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) value) ≤ 1 := by
  have hcoprime : ∀ i : Fin 4, Nat.Coprime 8 (sigmaOf sigmas4 i).toNat := by decide
  exact (polyNorm_substitution_le (by decide) (sigmas4_odd i) (hcoprime i) value).trans hvalue

/-- A signed monomial LUT remains a monomial in the integer negacyclic ring. -/
theorem signed_monomial (table : Fin 8 → ErrorPoly 8)
    (htable : ∀ i, ∃ exponent : Nat, table i = Xe exponent) (exponent : Nat) :
    ∃ power : Nat, PreprocessSwitch.signedValue (by decide) table exponent = Xe power := by
  obtain ⟨power, hpower⟩ := htable ⟨exponent % 8, Nat.mod_lt _ (by decide)⟩
  refine ⟨8 * (exponent / 8) + power, ?_⟩
  simp only [PreprocessSwitch.signedValue, hpower, Xe, pow_add, pow_mul,
    Negacyclic.root_pow_n, Int.cast_pow, Int.cast_neg, Int.cast_one]

/-- Every signed sigma-readout table value is an integer monomial. -/
theorem readout_monomial (exponent : Nat) :
    ∃ power : Nat, PreprocessSwitch.readoutValue exponent = Xe power := by
  apply signed_monomial
  intro i
  exact ⟨_, rfl⟩

/-- The scalar selected by the coefficient sampler, including its negacyclic sign. -/
def samplerScalar (exponent : Nat) : Int :=
  (-1) ^ (exponent / 8) *
    MxxIR.selectInt (Int.ofNat (exponent % 8)) [0, 1, 0, 1, 0, 1, 0, 1]

/-- The coefficient sampler is the constant polynomial of `samplerScalar`. -/
theorem sampler_value_cast (exponent : Nat) :
    PreprocessSwitch.samplerValue exponent = (samplerScalar exponent : ErrorPoly 8) := by
  simp [PreprocessSwitch.samplerValue, PreprocessSwitch.signedValue, samplerScalar]

/-- Every signed scalar in the coefficient sampler has absolute value at most one. -/
theorem sampler_scalar_small (exponent : Nat) : (samplerScalar exponent).natAbs ≤ 1 := by
  have htable : ∀ i : Fin 8,
      (MxxIR.selectInt (Int.ofNat i) [0, 1, 0, 1, 0, 1, 0, 1]).natAbs ≤ 1 := by decide
  simpa only [samplerScalar, Int.natAbs_mul, Int.natAbs_pow, Int.natAbs_neg,
    Int.natAbs_one, one_pow, one_mul] using
    htable ⟨exponent % 8, Nat.mod_lt _ (by decide)⟩

/-- Distinct coefficient placements keep the entire sampled sigma polynomial small. -/
theorem sampler_packed_small (exponents : Fin 8 → Nat) :
    polyNorm (∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (exponents o)) ≤ 1 := by
  have heq : (∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (exponents o)) =
      intPoly (fun o ↦ samplerScalar (exponents o)) := by
    simp only [sampler_value_cast, intPoly, Xe, mul_comm]
  rw [heq]
  apply polyNorm_le_of_coeff
  intro i
  rw [intPoly_coeff (by decide)]
  exact sampler_scalar_small _

/-- Genuine external payload agreements and binary-selector choices.
No field refers to an intermediate ciphertext or a generated witness. -/
structure ExternalConditions (external : GeneratedClaim.ExternalInputs) where
  /-- Root-label payloads agree with the initial root-label plaintexts. -/
  label_payload : ∀ i : Fin 2, message (external.input_12 i) (external.input_13 i) =
    labelValue external i
  /-- Input-label payloads agree with the initial input-label plaintexts. -/
  input_payload : ∀ i : Fin 4, message (external.input_10 i) (external.input_11 i) =
    inputValue external i
  /-- Selected member of external selector table `σ0`. -/
  σsel0 : Fin 8 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  σselector0 : ∀ o j, message (external.input_18 (cellAt o j))
    (external.input_39 (cellAt o j)) = if j = σsel0 o then (1 : ErrorPoly 8) else 0
  /-- Selected member of external selector table `σ1`. -/
  σsel1 : Fin 8 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  σselector1 : ∀ o j, message (external.input_19 (cellAt o j))
    (external.input_39 (cellAt o j)) = if j = σsel1 o then (1 : ErrorPoly 8) else 0
  /-- Selected member of external selector table `f0`. -/
  fsel0 : Fin 7200 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  fselector0 : ∀ o j, message (external.input_20 (cellAt o j))
    (external.input_38 (cellAt o j)) = if j = fsel0 o then (1 : ErrorPoly 8) else 0
  /-- Selected member of external selector table `f1`. -/
  fsel1 : Fin 7200 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  fselector1 : ∀ o j, message (external.input_21 (cellAt o j))
    (external.input_38 (cellAt o j)) = if j = fsel1 o then (1 : ErrorPoly 8) else 0
  /-- Selected member of external selector table `s0`. -/
  ssel0 : Fin 36000 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  sselector0 : ∀ o j, message (external.input_16 (cellAt o j))
    (external.input_40 (cellAt o j)) = if j = ssel0 o then (1 : ErrorPoly 8) else 0
  /-- Selected member of external selector table `s1`. -/
  ssel1 : Fin 36000 → Fin 2
  /-- The external selector table is exactly one-hot as an integer plaintext. -/
  sselector1 : ∀ o j, message (external.input_17 (cellAt o j))
    (external.input_40 (cellAt o j)) = if j = ssel1 o then (1 : ErrorPoly 8) else 0

/-- The three PRF member families, constructed from both buckets rather than assumed. -/
structure Members (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    (sm tp : ErrorPoly 8) where
  /-- The sigma sampler's selected plaintexts. -/
  σμ : Fin 8 → ErrorPoly 8
  /-- The fresh sampler's selected plaintexts. -/
  fμ : Fin 7200 → ErrorPoly 8
  /-- The smudge sampler's selected plaintexts. -/
  sμ : Fin 36000 → ErrorPoly 8
  /-- Both selector buckets establish this online member relation. -/
  σin : ∀ o : Fin 8, EncRel sm (generatedRoot.w_34_0 Con o, (generatedRoot.w_34_1 Con o,
      (generatedRoot.w_34_2 Con o, (generatedRoot.w_34_3 Con o, (generatedRoot.w_34_4 Con o,
        (generatedRoot.w_34_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * σμ o * unitPair tp j) • generatedRoot.w_57_0 Con) (selectorBucketBound (selectorBucketBound 1))
  /-- Both selector buckets establish this online member relation. -/
  fIn : ∀ o : Fin 7200, EncRel sm (generatedRoot.w_1186_0 Con o, (generatedRoot.w_1186_1 Con o,
      (generatedRoot.w_1186_2 Con o, (generatedRoot.w_1186_3 Con o, (generatedRoot.w_1186_4 Con o,
        (generatedRoot.w_1186_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * fμ o * unitPair tp j) • generatedRoot.w_57_0 Con) (selectorBucketBound (selectorBucketBound 1))
  /-- Both selector buckets establish this online member relation. -/
  sIn : ∀ o : Fin 36000, EncRel sm (generatedRoot.w_1240_0 Con o, (generatedRoot.w_1240_1 Con o,
      (generatedRoot.w_1240_2 Con o, (generatedRoot.w_1240_3 Con o, (generatedRoot.w_1240_4 Con o,
        (generatedRoot.w_1240_5 Con o, ()))))))
    (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * sμ o * unitPair tp j) • generatedRoot.w_57_0 Con) (selectorBucketBound (selectorBucketBound 1))

  /-- The selected external label products are small integer plaintexts. -/
  σsmall : ∀ o, polyNorm (σμ o) ≤ 1
  /-- The selected external label products are small integer plaintexts. -/
  fsmall : ∀ o, polyNorm (fμ o) ≤ 1
  /-- The selected external label products are small integer plaintexts. -/
  ssmall : ∀ o, polyNorm (sμ o) ≤ 1
  /-- The sigma inputs are products of genuine external label monomials. -/
  σmonomial : ∀ o, ∃ exponent : Nat, σμ o = Xe exponent
  /-- The fresh inputs are products of genuine external label monomials. -/
  fmonomial : ∀ o, ∃ exponent : Nat, fμ o = Xe exponent
  /-- The smudge inputs are products of genuine external label monomials. -/
  smonomial : ∀ o, ∃ exponent : Nat, sμ o = Xe exponent

/-- Both PRF buckets for all three families are consequences of the composed claim. -/
theorem Context.members {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) (E : ExternalConditions external) :
    Nonempty (Members C.online sm tp) := by
  obtain ⟨σ0, σ1, hσ⟩ := C.sigma_inputs hmask hpay layout hlayout hexact
    E.label_payload (fun k ↦ monomial_small (label_monomial external k))
    (fun k l ↦ monomial_small (monomial_mul (label_monomial external k)
      (label_monomial external l))) E.σsel0 E.σsel1 E.σselector0 E.σselector1
  obtain ⟨f0, f1, hf⟩ := C.fresh_inputs hmask hpay layout hlayout hexact
    E.label_payload (fun k ↦ monomial_small (label_monomial external k))
    (fun k l ↦ monomial_small (monomial_mul (label_monomial external k)
      (label_monomial external l))) E.fsel0 E.fsel1 E.fselector0 E.fselector1
  obtain ⟨s0, s1, hs⟩ := C.smudge_inputs hmask hpay layout hlayout hexact
    E.label_payload (fun k ↦ monomial_small (label_monomial external k))
    (fun k l ↦ monomial_small (monomial_mul (label_monomial external k)
      (label_monomial external l))) E.ssel0 E.ssel1 E.sselector0 E.sselector1
  exact ⟨⟨(fun o ↦ labelValue external (σ1 o) * labelValue external (σ0 o)),
    (fun o ↦ labelValue external (f1 o) * labelValue external (f0 o)),
    (fun o ↦ labelValue external (s1 o) * labelValue external (s0 o)), hσ, hf, hs,
    (fun o ↦ monomial_small (monomial_mul (label_monomial external (σ1 o))
      (label_monomial external (σ0 o)))),
    (fun o ↦ monomial_small (monomial_mul (label_monomial external (f1 o))
      (label_monomial external (f0 o)))),
    (fun o ↦ monomial_small (monomial_mul (label_monomial external (s1 o))
      (label_monomial external (s0 o)))),
    (fun o ↦ monomial_mul (label_monomial external (σ1 o))
      (label_monomial external (σ0 o))),
    (fun o ↦ monomial_mul (label_monomial external (f1 o))
      (label_monomial external (f0 o))),
    (fun o ↦ monomial_mul (label_monomial external (s1 o))
      (label_monomial external (s0 o)))⟩⟩

/-- Only scalar plaintext/table conditions remain after the generated execution is used.
The monomial forms discharge Walsh identities through the actual preprocessing tables;
the integer norm premises are explicit and do not follow from modular equalities alone. -/
structure Plaintexts (external : GeneratedClaim.ExternalInputs)
    (σμ : Fin 8 → ErrorPoly 8) (fμ : Fin 7200 → ErrorPoly 8)
    (sμ : Fin 36000 → ErrorPoly 8) where
  /-- Exponents at the sigma readout input. -/
  σreadoutExp : Fin 8 → Nat
  /-- Exponents at the sigma coefficient-sampler input. -/
  σsamplerExp : Fin 8 → Nat
  /-- Exponents at the fresh-noise readout input. -/
  freshExp : Fin 7200 → Nat
  /-- Exponents at the smudge readout input. -/
  smudgeExp : Fin 36000 → Nat
  /-- Exponent at the stride-two state LUT input. -/
  stateExp : Nat
  /-- The sigma readout receives the stated monomial modulo the ciphertext modulus. -/
  σreadoutForm : ∀ o, reducePoly modulus_0 8 (σμ o) =
    reducePoly modulus_0 8 (Xe (n := 8) (1 * σreadoutExp o))
  /-- The coefficient sampler receives the readout's stated monomial. -/
  σsamplerForm : ∀ o, reducePoly modulus_0 8 (PreprocessSwitch.readoutValue (σreadoutExp o)) =
    reducePoly modulus_0 8 (Xe (n := 8) (1 * σsamplerExp o))
  /-- The fresh-noise readout receives the stated monomial. -/
  freshForm : ∀ o, reducePoly modulus_0 8 (fμ o) =
    reducePoly modulus_0 8 (Xe (n := 8) (1 * freshExp o))
  /-- The smudge readout receives the stated monomial. -/
  smudgeForm : ∀ o, reducePoly modulus_0 8 (sμ o) =
    reducePoly modulus_0 8 (Xe (n := 8) (1 * smudgeExp o))
  /-- The state LUT receives the stated stride-two monomial. -/
  stateForm : reducePoly modulus_0 8 (statePlain external) =
    reducePoly modulus_0 8 (Xe (n := 8) (2 * stateExp))
  /-- The integer lift selected for the sigma readout remains small. -/
  readoutSmall : ∀ o, polyNorm (PreprocessSwitch.readoutValue (σreadoutExp o)) ≤ 1
  /-- The genuine external state-selector product remains small. -/
  stateSmall : polyNorm (statePlain external) ≤ 1
  /-- The coefficient-sampler outputs form a small packed sigma plaintext. -/
  sigmaSmall : polyNorm (∑ o : Fin 8, Xe (n := 8) (o : Nat) *
    PreprocessSwitch.samplerValue (σsamplerExp o)) ≤ 1

/-- All LUT forms and integer norm bounds follow from the selected label products and
one genuine external state-selector plaintext equality. -/
noncomputable def Plaintexts.of_state {sm tp : ErrorPoly 8}
    (M : Members C.online sm tp) (stateExp : Nat)
    (hstate : statePlain external = Xe (2 * stateExp)) :
    Plaintexts external M.σμ M.fμ M.sμ := by
  classical
  let readout : Fin 8 → Nat := fun o ↦ Classical.choose (M.σmonomial o)
  let sampler : Fin 8 → Nat := fun o ↦ Classical.choose (readout_monomial (readout o))
  let fresh : Fin 7200 → Nat := fun o ↦ Classical.choose (M.fmonomial o)
  let smudge : Fin 36000 → Nat := fun o ↦ Classical.choose (M.smonomial o)
  exact {
    σreadoutExp := readout
    σsamplerExp := sampler
    freshExp := fresh
    smudgeExp := smudge
    stateExp := stateExp
    σreadoutForm := fun o ↦ by
      simpa only [one_mul] using
        congrArg (reducePoly modulus_0 8) (Classical.choose_spec (M.σmonomial o))
    σsamplerForm := fun o ↦ by
      simpa only [one_mul] using
        congrArg (reducePoly modulus_0 8) (Classical.choose_spec (readout_monomial (readout o)))
    freshForm := fun o ↦ by
      simpa only [one_mul] using
        congrArg (reducePoly modulus_0 8) (Classical.choose_spec (M.fmonomial o))
    smudgeForm := fun o ↦ by
      simpa only [one_mul] using
        congrArg (reducePoly modulus_0 8) (Classical.choose_spec (M.smonomial o))
    stateForm := congrArg (reducePoly modulus_0 8) hstate
    readoutSmall := fun o ↦ monomial_small (readout_monomial (readout o))
    stateSmall := monomial_small ⟨2 * stateExp, hstate⟩
    sigmaSmall := sampler_packed_small sampler }

/-- The independent smudge-residue obligation, deliberately not derived in this module. -/
def Residue (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    {σμ : Fin 8 → ErrorPoly 8} {fμ : Fin 7200 → ErrorPoly 8} {sμ : Fin 36000 → ErrorPoly 8}
    (P : Plaintexts external σμ fμ sμ) (BM : Nat) : Prop :=
  ∀ i : Fin 36, ∃ M : ErrorMatrix 8 1 25, CoeffBound M BM ∧
    Matrix.of (fun _ l ↦ reducePoly modulus_0 8
        (cellVal (C := 4500) (by norm_num) 1 (fun o ↦ PreprocessSwitch.smudgeValue (P.smudgeExp o)) ⟨125 * i + l.val, by have := i.isLt; have := l.isLt; omega⟩) * (generatedRoot.w_1274_0 Con 0 0)) -
      generatedRoot.w_1276_0 Con = reduceMatrix modulus_0 8 1 25 M

/-- Assemble every online input field from the claim contexts and scalar plaintext conditions.
Only the independently owned smudge residue is passed separately; `Noise0` is not assumed. -/
noncomputable def Context.inputs {sm tp : ErrorPoly 8}
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (hsmall : polyNorm sm ≤ 1) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) (E : ExternalConditions external)
    (M : Members C.online sm tp) (P : Plaintexts external M.σμ M.fμ M.sμ)
    (BM : Nat) (hresidue : Residue C.online P BM) : Online0Inputs C.online sm where
  sp := tp
  hsm := hsmall
  σμ := M.σμ
  σB := selectorBucketBound (selectorBucketBound 1)
  σin := M.σin
  Bh := 1
  help := C.help hmask hpay
  σΔr := 1
  σνr := fun o ↦ PreprocessSwitch.readoutValue (P.σreadoutExp o)
  σCr := PreprocessSwitch.readoutCoefficients
  σBBr := 0
  σswitch := C.σswitch hpay
  σhμ := fun o i ↦ small_sigma8 _ (M.σsmall o) i
  σwalsh := fun o ↦ PreprocessSwitch.readout_walsh (P.σreadoutExp o) (P.σreadoutForm o)
  BhS := 1
  helpS := C.helpS hmask hpay
  σΔ := 1
  σν := fun o ↦ PreprocessSwitch.samplerValue (P.σsamplerExp o)
  σCS := PreprocessSwitch.samplerCoefficients
  σBBS := 0
  σswitchS := C.σswitchS hpay
  σhμS := by
    intro o i
    simpa only [one_mul] using small_sigma8 _ (P.readoutSmall o) i
  σwalshS := by
    intro o
    simpa only [one_mul] using
      PreprocessSwitch.sampler_walsh (P.σsamplerExp o) (P.σsamplerForm o)
  oneC0 := generatedRoot.w_16_0 C.online 1
  oneB := 1
  one := C.one hmask hpay
  tp := tp
  privB := 1
  priv := C.priv hmask hpay
  stμ := statePlain external
  stB := selectorCellBound 1
  stEnc := C.stEnc hmask hpay layout hlayout hexact E.label_payload (state_seed_small external)
  stBh := 1
  stHelp := C.stHelp hmask hpay
  stΔ := 1
  stν := PreprocessSwitch.stateValue P.stateExp
  stC := PreprocessSwitch.stateCoefficients
  stBB := 0
  stSwitch := C.stSwitch hpay
  stHμ := small_sigma4 _ P.stateSmall
  stWalsh := PreprocessSwitch.state_walsh P.stateExp P.stateForm
  lv := inputValue external
  hα := fun i ↦ monomial_small (input_monomial external i)
  Ba := 1
  lab := C.lab hmask hpay
  pay := C.pay E.input_payload
  hα13 := monomial_small (monomial_mul (input_monomial external 1) (input_monomial external 3))
  hα01 := monomial_small (monomial_mul (input_monomial external 0) (input_monomial external 1))
  hα0123 := monomial_small (monomial_mul (monomial_mul (input_monomial external 0) (input_monomial external 1)) (monomial_mul (input_monomial external 2) (input_monomial external 3)))
  hα02 := monomial_small (monomial_mul (input_monomial external 0) (input_monomial external 2))
  hα023 := monomial_small (monomial_mul (monomial_mul (input_monomial external 0) (input_monomial external 2)) (input_monomial external 3))
  hα12 := monomial_small (monomial_mul (input_monomial external 1) (input_monomial external 2))
  βo := 1
  Bo := 1
  oneS := C.oneS hmask hpay
  pay0 := by
    simpa [labelValue] using C.pay_state E.label_payload 1
  hα1β := by
    simpa only [mul_one] using monomial_small (input_monomial external 1)
  a3Bh := 1
  a3help := C.a3help hmask hpay
  a3Δ := 1
  a3ν := substitution 3 (⟨1, rfl⟩ : Odd 3) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a3C := PreprocessSwitch.aut3Coefficients 0
  a3BB := 0
  a3switch := C.a3switch hpay
  a3hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨1, by decide⟩ : Fin 8)
  a3walsh := by
    simpa [PreprocessSwitch.aut3Coefficients, PreprocessSwitch.aut3Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 3)
        (⟨1, rfl⟩ : Odd 3) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a5Bh := 1
  a5help := C.a5help hmask hpay
  a5Δ := 1
  a5ν := substitution 5 (⟨2, rfl⟩ : Odd 5) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a5C := PreprocessSwitch.aut5Coefficients 0
  a5BB := 0
  a5switch := C.a5switch hpay
  a5hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨2, by decide⟩ : Fin 8)
  a5walsh := by
    simpa [PreprocessSwitch.aut5Coefficients, PreprocessSwitch.aut5Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 5)
        (⟨2, rfl⟩ : Odd 5) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a7Bh := 1
  a7help := C.a7help hmask hpay
  a7Δ := 1
  a7ν := substitution 7 (⟨3, rfl⟩ : Odd 7) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a7C := PreprocessSwitch.aut7Coefficients 0
  a7BB := 0
  a7switch := C.a7switch hpay
  a7hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨3, by decide⟩ : Fin 8)
  a7walsh := by
    simpa [PreprocessSwitch.aut7Coefficients, PreprocessSwitch.aut7Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 7)
        (⟨3, rfl⟩ : Odd 7) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a9Bh := 1
  a9help := C.a9help hmask hpay
  a9Δ := 1
  a9ν := substitution 9 (⟨4, rfl⟩ : Odd 9) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a9C := PreprocessSwitch.aut9Coefficients 0
  a9BB := 0
  a9switch := C.a9switch hpay
  a9hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨4, by decide⟩ : Fin 8)
  a9walsh := by
    simpa [PreprocessSwitch.aut9Coefficients, PreprocessSwitch.aut9Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 9)
        (⟨4, rfl⟩ : Odd 9) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a11Bh := 1
  a11help := C.a11help hmask hpay
  a11Δ := 1
  a11ν := substitution 11 (⟨5, rfl⟩ : Odd 11) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a11C := PreprocessSwitch.aut11Coefficients 0
  a11BB := 0
  a11switch := C.a11switch hpay
  a11hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨5, by decide⟩ : Fin 8)
  a11walsh := by
    simpa [PreprocessSwitch.aut11Coefficients, PreprocessSwitch.aut11Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 11)
        (⟨5, rfl⟩ : Odd 11) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a13Bh := 1
  a13help := C.a13help hmask hpay
  a13Δ := 1
  a13ν := substitution 13 (⟨6, rfl⟩ : Odd 13) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a13C := PreprocessSwitch.aut13Coefficients 0
  a13BB := 0
  a13switch := C.a13switch hpay
  a13hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨6, by decide⟩ : Fin 8)
  a13walsh := by
    simpa [PreprocessSwitch.aut13Coefficients, PreprocessSwitch.aut13Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 13)
        (⟨6, rfl⟩ : Odd 13) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  a15Bh := 1
  a15help := C.a15help hmask hpay
  a15Δ := 1
  a15ν := substitution 15 (⟨7, rfl⟩ : Odd 15) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o))
  a15C := PreprocessSwitch.aut15Coefficients 0
  a15BB := 0
  a15switch := C.a15switch hpay
  a15hμ := by
    simpa [sigmas8, sigmaOf, MxxIR.selectInt, one_mul] using
      small_sigma8 _ P.sigmaSmall (⟨7, by decide⟩ : Fin 8)
  a15walsh := by
    simpa [PreprocessSwitch.aut15Coefficients, PreprocessSwitch.aut15Sigmas,
      sigmaOf, MxxIR.selectInt, one_mul] using
      (PreprocessSwitch.aut_table_walsh (n := 8) (q := modulus_0)
        (by norm_num [modulus_0]) (by decide) (sigma := 15)
        (⟨7, rfl⟩ : Odd 15) (1 * ∑ o : Fin 8, Xe (n := 8) (o : Nat) *
      PreprocessSwitch.samplerValue (P.σsamplerExp o)))
  fμ := M.fμ
  fB := selectorBucketBound (selectorBucketBound 1)
  fIn := M.fIn
  fΔ := 1
  fν := fun o ↦ PreprocessSwitch.sigmaReadoutValue (P.freshExp o)
  fC := PreprocessSwitch.sigmaReadoutCoefficients
  fBB := 0
  fSwitch := C.fSwitch hpay
  fHμ := fun o i ↦ small_sigma8 _ (M.fsmall o) i
  fWalsh := fun o ↦ PreprocessSwitch.sigmaReadout_walsh (P.freshExp o) (P.freshForm o)
  sμ := M.sμ
  sB := selectorBucketBound (selectorBucketBound 1)
  sIn := M.sIn
  sΔ := 1
  sν := fun o ↦ PreprocessSwitch.smudgeValue (P.smudgeExp o)
  sC := PreprocessSwitch.smudgeCoefficients
  sBB := 0
  sSwitch := C.sSwitch hpay
  sHμ := fun o i ↦ small_sigma8 _ (M.ssmall o) i
  sWalsh := fun o ↦ PreprocessSwitch.smudge_walsh (P.smudgeExp o) (P.smudgeForm o)
  hconst := generatedRoot.w_1274_0 C.online 0 0
  hconst_eq := by
    funext row column
    fin_cases row
    fin_cases column
    rfl
  BM := BM
  hTH := hresidue

/-- At the claim level, all ciphertext premises of `Online0Inputs` are discharged.
The remaining universally quantified inputs are scalar plaintext/table conditions, followed
by the independent smudge residue; no online encoding relation or switching equation remains. -/
theorem claim_online0_inputs {sm tp : ErrorPoly 8}
    (h : W_GeneratedClaim.Runs model external execution)
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (hsmall : polyNorm sm ≤ 1) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) (E : ExternalConditions external) :
    ∃ C : Context model external execution, ∃ M : Members C.online sm tp,
      ∀ (P : Plaintexts external M.σμ M.fμ M.sμ) (BM : Nat),
        Residue C.online P BM → Nonempty (Online0Inputs C.online sm) := by
  obtain ⟨C⟩ := contexts h
  obtain ⟨M⟩ := C.members hmask hpay layout hlayout hexact E
  exact ⟨C, M, fun P BM hresidue ↦
    ⟨C.inputs hmask hpay hsmall layout hlayout hexact E M P BM hresidue⟩⟩

/-- A claim run supplies every field of `Online0Inputs` except the separately supplied
smudge residue. Only genuine external payload agreements, one-hot selector messages,
the state-selector plaintext equality, and the secret/layout conditions are assumed. -/
theorem claim_online0_inputs_of_state {sm tp : ErrorPoly 8}
    (h : W_GeneratedClaim.Runs model external execution)
    (hmask : execution.«stage_3» 0 0 = reducePoly modulus_0 8 sm)
    (hpay : execution.«stage_0» 0 0 = reducePoly modulus_0 8 tp)
    (hsmall : polyNorm sm ≤ 1) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) (E : ExternalConditions external)
    (stateExp : Nat) (hstate : statePlain external = Xe (2 * stateExp)) :
    ∃ C : Context model external execution, ∃ M : Members C.online sm tp,
      ∀ BM : Nat, Residue C.online (Plaintexts.of_state C M stateExp hstate) BM →
        Nonempty (Online0Inputs C.online sm) := by
  obtain ⟨C⟩ := contexts h
  obtain ⟨M⟩ := C.members hmask hpay layout hlayout hexact E
  exact ⟨C, M, fun BM hresidue ↦
    ⟨C.inputs hmask hpay hsmall layout hlayout hexact E M
      (Plaintexts.of_state C M stateExp hstate) BM hresidue⟩⟩

end Online0Claim

end Dio2Proof
