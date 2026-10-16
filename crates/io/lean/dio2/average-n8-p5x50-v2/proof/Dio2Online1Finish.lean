import Dio2Online1Closure

set_option maxHeartbeats 16000000
set_option maxRecDepth 16384

namespace Dio2Proof.Online1Closure
open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
  RefreshTopProof.GenIface RefreshTopProof.GenInst RefreshTopProof.GenFinal
  Mxx.ExponentLut.Refresh
open scoped BigOperators

open W_Stage_online_1 Dio2Proof.LutKey

theorem normalized_exponent (exponent : Int) (hfactor : exponent % (1 : Int) = 0) :
    (exponent % (2 * 8 : Nat)).toNat =
      1 * ((exponent % (2 * 8 : Nat)).toNat / 1) := by
  have hmod := Int.emod_nonneg exponent (by norm_num : (2 * 8 : Int) ≠ 0)
  have hlt := Int.emod_lt_of_pos exponent (by norm_num : 0 < (2 * 8 : Int))
  omega

noncomputable def labelExponent (external : GeneratedClaim.ExternalInputs) (member : Fin 2) : Nat :=
  (Complete.label1Exponent external member % (2 * 8 : Nat)).toNat / 1

theorem label_monomial {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hrun : W_GeneratedClaim.Runs model external execution) (member : Fin 2) :
    online1Labels external member = Xe (n := 8) (1 * labelExponent external member) := by
  change message _ _ = _
  rw [Complete.label1_payload hrun member]
  unfold message
  rw [Int.cast_one, one_mul]
  congr 1
  apply normalized_exponent
  exact Int.emod_one _

theorem smudge_component {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (C : Claim1.Context model external execution)
    (hrun : W_GeneratedClaim.Runs model external execution)
    {sm sp current : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    (hpayload : execution.«stage_0» = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns Backend.backend 4096 25 gadget)
    (roots : Online1Roots C.online execution.«stage_1».2.2.1 sm sp current
      (online1Labels external) gadget) :
    ∃ (target : ExactMatrix modulus_0 8 1 1) (error residual : ErrorMatrix 8 1 1),
      CoeffBound error Dio2NoiseCertificate.b10_smudge ∧
      CoeffBound residual Dio2NoiseCertificate.b10_residue ∧
      generatedRoot.w_178_0 C.online = maskPart (reducePoly modulus_0 8 sm)
        (smudgeKeyS1 C.online) - target + reduceMatrix modulus_0 8 1 1 error ∧
      target - generatedRoot.w_181_0 C.online = reduceMatrix modulus_0 8 1 1 residual := by
  classical
  have hq : 1 < modulus_0 := by norm_num [modulus_0]
  have hg := centeredRebaseRuns_self hq (by decide) gadget
  have hseed : generatedRoot.w_70_0 C.online = W_Stage_preprocess_1_1.generatedRoot.w_0_0 C.preprocess := by
    change C.online.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 = C.preprocess.inputs.1
    rw [C.online_inputs, C.preprocess_inputs]
    rfl

  obtain ⟨sel0, hsel0⟩ := Complete.selector_loop_86 hrun
  let selected0 (member : Fin 8) : Fin 2 :=
    ⟨(external.input_28 (cellAt member (sel0 member))).toNat, by
      have := (Complete.valid hrun).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (cellAt member (sel0 member))
      omega⟩
  have hselected0 (member : Fin 8) :
      ((selected0 member).val : Int) = generatedRoot.w_140_0 C.online
        (cellAt member (sel0 member)) := by
    change ((external.input_28 (cellAt member (sel0 member))).toNat : Int) =
      C.online.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
        (cellAt member (sel0 member))
    rw [C.online_inputs]
    change ((external.input_28 (cellAt member (sel0 member))).toNat : Int) =
      external.input_28 (cellAt member (sel0 member))
    have := (Complete.valid hrun).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (cellAt member (sel0 member))
    omega
  let ν0 (member : Fin 16) : ErrorPoly 8 :=
    message (external.input_20 member) (external.input_37 member)
  choose rhsError0 hrhsBound0 hrhs0 using
    rhs_parallel_generatedRoot_75 C.preprocess C.preprocess_facts
      (Claim1.preprocess_payload C hpayload) hG
  have hrhs'0 (member : Fin 16) (port : Fin 2) := hrhs0 member port
  simp only [W_Stage_preprocess_1_1.generatedRoot.w_73_0,
    W_Stage_preprocess_1_1.generatedRoot.w_74_0, C.preprocess_inputs,
    Claim1.preprocessInputs] at hrhs'0
  let e0 (member : Fin 8) : Nat :=
    labelExponent external (selected0 member) + 0
  let μ0 (member : Fin 8) : ErrorPoly 8 :=
    online1Labels external (selected0 member) * (1 : ErrorPoly 8)
  have hμ0 (member : Fin 8) : μ0 member = Xe (n := 8) (1 * e0 member) := by
    dsimp only [μ0, e0]
    rw [label_monomial hrun]
    simp only [mul_one, Xe, ← pow_add, Nat.mul_add, Nat.mul_zero, add_zero]
  have hrhsProof0 (member : Fin 16) (port : Fin 2) :
      reducePoly modulus_0 8 sp •
        topPairG model (generatedRoot.w_70_0 C.online) modulus_0 25
          online1_bucket0_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 member) port +
        ![generatedRoot.w_147_0 C.online member, generatedRoot.w_148_0 C.online member] port =
        reducePoly modulus_0 8 (((1 : Nat) : Int) * ν0 member * unitPair sp port) • gadget +
          reduceMatrix modulus_0 8 1 25 (rhsError0 member port) := by
    simpa only [Claim1.preprocess_route_w_147_0 C, Claim1.preprocess_route_w_148_0 C,
      hseed, Nat.cast_one, Int.cast_one, one_mul] using hrhs'0 member port

  obtain ⟨sel1, hsel1⟩ := Complete.selector_loop_88 hrun
  let selected1 (member : Fin 8) : Fin 2 :=
    ⟨(external.input_29 (cellAt member (sel1 member))).toNat, by
      have := (Complete.valid hrun).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (cellAt member (sel1 member))
      omega⟩
  have hselected1 (member : Fin 8) :
      ((selected1 member).val : Int) = generatedRoot.w_139_0 C.online
        (cellAt member (sel1 member)) := by
    change ((external.input_29 (cellAt member (sel1 member))).toNat : Int) =
      C.online.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
        (cellAt member (sel1 member))
    rw [C.online_inputs]
    change ((external.input_29 (cellAt member (sel1 member))).toNat : Int) =
      external.input_29 (cellAt member (sel1 member))
    have := (Complete.valid hrun).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 (cellAt member (sel1 member))
    omega
  let ν1 (member : Fin 16) : ErrorPoly 8 :=
    message (external.input_21 member) (external.input_37 member)
  choose rhsError1 hrhsBound1 hrhs1 using
    rhs_parallel_generatedRoot_79 C.preprocess C.preprocess_facts
      (Claim1.preprocess_payload C hpayload) hG
  have hrhs'1 (member : Fin 16) (port : Fin 2) := hrhs1 member port
  simp only [W_Stage_preprocess_1_1.generatedRoot.w_77_0,
    W_Stage_preprocess_1_1.generatedRoot.w_78_0, C.preprocess_inputs,
    Claim1.preprocessInputs] at hrhs'1
  let e1 (member : Fin 8) : Nat :=
    labelExponent external (selected1 member) + e0 member
  let μ1 (member : Fin 8) : ErrorPoly 8 :=
    online1Labels external (selected1 member) * μ0 member
  have hμ1 (member : Fin 8) : μ1 member = Xe (n := 8) (1 * e1 member) := by
    dsimp only [μ1, e1]
    rw [label_monomial hrun]
    rw [hμ0]
    simp only [mul_one, Xe, ← pow_add, Nat.mul_add, Nat.mul_zero, add_zero]
  have hnorm1 (key : Fin 2) (member : Fin 8) :
      polyNorm (online1Labels external key * μ0 member) ≤ 1 := by
    rw [label_monomial hrun, hμ0]
    simpa only [Xe, ← pow_add] using
      polyNorm_Xe (n := 8) (by decide) (1 * labelExponent external key + 1 * e0 member)
  have hrhsProof1 (member : Fin 16) (port : Fin 2) :
      reducePoly modulus_0 8 sp •
        topPairG model (generatedRoot.w_70_0 C.online) modulus_0 25
          online1_bucket1_rhsPre (prfTopTags 36550556566445887949135852474979514307313305075533710697673106717124183540354 84715014674746765542951856319120207364894094043218421459706511318570933778708 member) port +
        ![generatedRoot.w_151_0 C.online member, generatedRoot.w_152_0 C.online member] port =
        reducePoly modulus_0 8 (((1 : Nat) : Int) * ν1 member * unitPair sp port) • gadget +
          reduceMatrix modulus_0 8 1 25 (rhsError1 member port) := by
    simpa only [Claim1.preprocess_route_w_151_0 C, Claim1.preprocess_route_w_152_0 C,
      hseed, Nat.cast_one, Int.cast_one, one_mul] using hrhs'1 member port

  let eLast := e1
  have hnorm (key : Fin 2) : polyNorm (online1Labels external key * (1 : ErrorPoly 8)) ≤ 1 := by
    rw [mul_one, label_monomial hrun]
    exact polyNorm_Xe (by decide) _
  have hswitch := PreprocessSwitch.switching_errors
    (Online1Preprocess.claim_smudge_switching C hpayload hG)
  obtain ⟨error, residual, herror, hresidual, heq, htarget⟩ := online1_smudge_certified
    C.online C.online_facts hq Backend.layout0 Complete.layout_some Complete.layout_exact
    (μ := 1) (α := online1Labels external) (Complete.label1_gain hrun) hnorm hG
    roots.labels roots.payload hG hg
    (by simpa only [one_mul] using roots.privateUnit)
    hrhsProof0 hrhsBound0 sel0 hsel0 selected0 hselected0 μ0 (fun _ ↦ rfl) hnorm1 hrhsProof1 hrhsBound1 sel1 hsel1 selected1 hselected1 μ1 (fun _ ↦ rfl) hsm
    (by simpa only [Nat.cast_one, Int.cast_one, one_mul] using roots.smudgeHelpers)
    (Δ' := 1) (ρ := fun member ↦ (smudgeDigit (eLast member) : ErrorPoly 8))
    (C := Online1Preprocess.smudgeCoefficients)
    (by
      intro branch
      obtain ⟨error, hbound, heq⟩ := hswitch branch
      exact ⟨error, fun port ↦ coeffBound_mono (hbound port) (Nat.zero_le _), heq⟩)
    (fun member branch ↦ by
      rw [hμ1]
      exact polyNorm_substitution_Xe_gen (by decide) (online1Sigmas_odd branch) _)
    (fun member ↦ smudge_walsh (eLast member) (hμ1 member)) hG
    (fun digit _ coefficient ↦ smudgeDigit (eLast
      (online1PlacementSource ⟨8 * digit.val + coefficient.val,
        by have := digit.isLt; have := coefficient.isLt; omega⟩)))
    (fun digit column ↦ congrArg (reducePoly modulus_0 8) (packed_readout eLast digit column))
    (fun digit column coefficient ↦ (smudge_digit_range _).1)
    (fun digit column coefficient ↦ (smudge_digit_range _).2)
  exact ⟨(∑ r, reducePoly modulus_0 8 (online1PackedReadout 1 (fun member ↦ (smudgeDigit (eLast member) : ErrorPoly 8)) r 0) • generatedRoot.w_177_0 C.online r), error, residual, herror, hresidual, heq, htarget⟩

theorem step1 {model external execution} (hrun : W_GeneratedClaim.Runs model external execution)
    (state bit : Nat) (hstate : state ≤ 1) (hbit : bit ≤ 1)
    (hlabel : Label model external execution 1 (Xe (n := 8) (2 * bit)))
    (hinput : State1 model external execution (Xe (n := 8) (2 * state))) :
    FinalOutput execution (Xe (n := 8)
      (2 * (entry table1 ((state + 1 + bit : Nat) : Int)).toNat)) := by
  obtain ⟨C⟩ := Claim1.contexts hrun
  obtain ⟨sm, sp, gadget, hsm, hsp, hG, hpayload, roots⟩ := Claim1.imported_roots C hinput
  obtain ⟨stateError, hstateError, hstateEq⟩ := state_component C hrun hsm hpayload hG
    state bit hstate hbit hlabel roots
  obtain ⟨target, smudgeError, residual, hsmudgeError, hresidual, hsmudgeEq, htarget⟩ :=
    smudge_component C hrun hsm hpayload hG roots
  exact online1_output_from_components C hrun roots hstateEq hstateError
    hsmudgeEq hsmudgeError htarget hresidual

end Dio2Proof.Online1Closure
