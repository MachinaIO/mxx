import Dio2Correct
import Dio2Online0Inputs
import Dio2Online1Assembly
import Dio2Online0AssemblyP
import Dio2SmudgeHalf
import Dio2Requirement

/-!
# The complete correctness of dio2

`Dio2Correct.certificate` turns a `CorrectInputs` value into the average-case claim. This module
constructs that value (`correctInputs`) from the proven pieces; the final theorem
`dio2_correctness_claim : GeneratedClaim.CorrectnessClaim` carries no assumption:

* the obfuscator's plaintext tables are checked by the run itself: the label payloads, the two
  state selectors `message (input_33 0) (input_34 0) = X^(2 y₀)` and
  `message (input_35 0) (input_36 0) = X^(2 y₁)`, the step-one labels and all eight one-hot
  selector tables follow from `W_Requirement_0` (`Dio2Requirement.requirement_facts`) and the
  validity contract (`external_conditions`, `state_selector0`, `state_selector1`, `onehot`);
* the mask and payload secrets are the samples of `W_Stage_mask`/`W_Stage_payload`
  (`mask_lift`, `payload_lift`), and the layout is the backend's (`layout_some`, `layout_exact`);
* step zero's `Online0Inputs` is `Online0Claim.Context.inputs` with the bound-four smudge
  residues of `Dio2SmudgeHalf` (`residue0`, `Step0.hTHP`) and the decoding inequalities of every
  part and prime (`noise0_of_bounds`, as `Dio2Noise`);
* the step transition: refreshed targets (`Step0.targets`), fresh bound eight
  (`Step0.freshBound`), new-root key agreement (`root_keys`) and the refreshed hint
  (`hint_of_rows`, bound eight) are proven here from the generated online-0 facts;
* step one is `Online1Assembly` with the state selector left general (`Online1InputsR`,
  `claim_online1_outputR`): `Online1Inputs.stateRhs` fixes the step-one selector plaintext to
  `1`, i.e. `y₁ = 0`, so `Online1Inputs` itself cannot be built for `y₁ = 1`.

`MissingLinks` has one field, the full CRT rows of step zero; `online0Rows_holds` proves it from
`Online0InputsP.part_term0 … part_term4` (`Dio2Online0AssemblyP`), the preimages, the hint and
the keys stage, and `dio2_correctness_claim` applies `dio2_correct` to it.
-/

set_option maxHeartbeats 64000000
set_option maxRecDepth 16384
set_option linter.unusedSimpArgs false

namespace Dio2Proof

namespace Complete

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof RefreshTopProof.Gen
  RefreshTopProof.GenIface

/-! ## Facts of every run that are external in the stage proofs -/

section RunFacts

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- The backend's regular layout at the ciphertext modulus. -/
theorem layout_some :
    Backend.backend.regularLayout W_Stage_online_0.modulus_0 8 = some Backend.layout0 := by
  unfold Backend.backend
  simp

/-- The backend layout drops no modulus. -/
theorem layout_exact : Backend.layout0.droppedModuli = 0 := rfl

/-- The mask stage samples a `{0, 1}` polynomial. -/
theorem mask_lift (h : W_GeneratedClaim.Runs model external execution) :
    ∃ sm : ErrorPoly 8, execution.«stage_3» 0 0 = reducePoly W_Stage_online_0.modulus_0 8 sm ∧
      polyNorm sm ≤ 1 := by
  obtain ⟨-, -, -, -, ⟨w, ⟨-, witness, hw, hb⟩, hout⟩, -⟩ := h
  refine ⟨witness 0 0, ?_, polyNorm_le_of_coeff fun i ↦ ?_⟩
  · rw [hout, hw]
    rfl
  · have := hb 0 0 i
    omega

/-- The payload stage samples a ternary polynomial. -/
theorem payload_lift (h : W_GeneratedClaim.Runs model external execution) :
    ∃ tp : ErrorPoly 8, execution.«stage_0» 0 0 = reducePoly W_Stage_online_0.modulus_0 8 tp ∧
      polyNorm tp ≤ 1 := by
  obtain ⟨-, ⟨w, ⟨-, witness, hw, hb⟩, hout⟩, -⟩ := h
  refine ⟨witness 0 0, ?_, polyNorm_le_of_coeff fun i ↦ ?_⟩
  · rw [hout, hw]
    rfl
  · have := hb 0 0 i
    omega

/-- The requirement checked by every run. -/
theorem requirement (h : W_GeneratedClaim.Runs model external execution) :
    Dio2Requirement.Facts external.input_12 external.input_13 external.input_0 external.input_10
      external.input_11 external.input_33 external.input_34 external.input_32 external.input_14
      external.input_15 external.input_35 external.input_36 external.input_16 external.input_17
      external.input_18 external.input_19 external.input_20 external.input_21 external.input_22
      external.input_23 := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hr, -, ho⟩ := h
  exact Dio2Requirement.requirement_facts hr ho

/-- Both summands of a true two-way conjunction are true. -/
theorem both {p q : Prop} [Decidable p] [Decidable q]
    (h : ((if p then (1 : Int) else 0) + (if q then (1 : Int) else 0)) = 2) : p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp_all

/-- `1 · X^a` for `a ∈ [0, 15]` is the monomial `X^a`. -/
theorem message_one_nat {a : Int} (ha : 0 ≤ a ∧ a ≤ 15) :
    message (n := 8) 1 a = Xe (n := 8) a.toNat :=
  message_one rfl (Int.toNat_of_nonneg ha.1).symm (by omega)

/-- A signed plaintext `c X^e` checked by the requirement against `X^a`. -/
theorem message_signed {c e a : Int} (hc : c * c = 1) (he : 0 ≤ e ∧ e ≤ 7)
    (ha : 0 ≤ a ∧ a ≤ 15) (h : e * 2 + (1 - c) * 8 = a * 1 * 2) :
    message (n := 8) c e = message (n := 8) 1 a := by
  have hc' : c = 1 ∨ c = -1 := by
    rcases Int.mul_eq_one_iff_eq_one_or_neg_one.mp hc with h | h <;> simp [h.1]
  rcases hc' with rfl | rfl
  · have : e = a := by omega
    rw [this]
  · rw [message_one_nat ha, message, Int.emod_eq_of_lt (by omega) (by omega)]
    have hk : a.toNat = 8 + e.toNat := by omega
    rw [hk]
    show _ = AdjoinRoot.root (negacyclicModulus 8 Int) ^ (8 + e.toNat)
    rw [pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

/-- A unit-checked plaintext `c X^e` whose requirement target is `0` is `1`. -/
theorem message_unit {c e : Int} (hc : c * c = 1) (he : 0 ≤ e ∧ e ≤ 7)
    (h : e * 2 + (1 - c) * 8 = 0) : message (n := 8) c e = 1 := by
  rw [message_signed (a := 0) hc he (by omega) (by omega), message_one_nat (by omega)]
  rfl

/-- A one-hot pair table with zero exponents is a binary selector. -/
theorem onehot {M : Nat} (c e : Fin (M * 2) → Int) (hc : ∀ k, 0 ≤ c k ∧ c k ≤ 1)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 0)
    (hsum : ∀ o < M, (0 + ∑ j ∈ Finset.range 2,
      famAt c ((Int.ofNat o) * 2 + (Int.ofNat j)).toNat) = 1) :
    ∃ sel : Fin M → Fin 2, ∀ o j, message (n := 8) (c (cellAt o j)) (e (cellAt o j)) =
      if j = sel o then (1 : ErrorPoly 8) else 0 := by
  refine ⟨fun o ↦ if c (cellAt o 0) = 1 then 0 else 1, fun o j ↦ ?_⟩
  have hs := hsum o.val o.isLt
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at hs
  simp only [Int.ofNat_eq_natCast] at hs ⊢
  have h0 : famAt c (((o.val : Int)) * 2 + ((0 : Nat) : Int)).toNat = c (cellAt o 0) := by
    rw [famAt_of_lt]
    · congr 1
      apply Fin.ext
      simp [cellAt]
      omega
    · have := o.isLt
      omega
  have h1 : famAt c (((o.val : Int)) * 2 + ((1 : Nat) : Int)).toNat = c (cellAt o 1) := by
    rw [famAt_of_lt]
    · congr 1
      apply Fin.ext
      simp [cellAt]
      omega
    · have := o.isLt
      omega
  rw [h0, h1] at hs
  have hz : e (cellAt o j) = 0 := by have := he (cellAt o j); omega
  have hm : message (n := 8) (c (cellAt o j)) (e (cellAt o j)) = ((c (cellAt o j) : Int) : ErrorPoly 8) := by
    rw [hz, message]
    simp
  rw [hm]
  have hc0 := hc (cellAt o 0)
  have hc1 := hc (cellAt o 1)
  fin_cases j
  · by_cases h : c (cellAt o 0) = 1
    · simp [h]
    · have : c (cellAt o 0) = 0 := by omega
      simp [this]
  · by_cases h : c (cellAt o 0) = 1
    · have : c (cellAt o 1) = 0 := by omega
      simp [h, this]
    · have : c (cellAt o 1) = 1 := by omega
      simp [h, this]

/-- The validity contract of every run. -/
theorem valid (h : W_GeneratedClaim.Runs model external execution) :
    GeneratedClaim.ValidExternals external := h.1

/-- The requirement's label checks: the root-label payloads are the initial label plaintexts. -/
theorem label_payload (h : W_GeneratedClaim.Runs model external execution) (i : Fin 2) :
    message (external.input_12 i) (external.input_13 i) = Online0Claim.labelValue external i := by
  obtain ⟨v0, -, -, -, -, -, -, -, -, -, -, -, v12, v13, -⟩ := valid h
  have R := requirement h
  fin_cases i
  · obtain ⟨hc, he⟩ := both R.cond_0
    simp only [famAt, Nat.ofNat_pos, ↓reduceDIte, Fin.zero_eta] at hc he
    simp only [decide_eq_true_eq] at hc he
    simpa [Online0Claim.labelValue] using
      message_signed hc (v13 0) (v0 0) he
  · obtain ⟨hc, he⟩ := both R.cond_1
    simp only [famAt, Nat.one_lt_ofNat, ↓reduceDIte, Fin.mk_one] at hc he
    simp only [decide_eq_true_eq] at hc he
    simpa [Online0Claim.labelValue] using message_unit hc (v13 1) he

/-- The requirement's input-label checks. -/
theorem input_payload (h : W_GeneratedClaim.Runs model external execution) (i : Fin 4) :
    message (external.input_10 i) (external.input_11 i) = Online0Claim.inputValue external i := by
  obtain ⟨v0, -, -, -, -, -, -, -, -, -, v10, v11, -⟩ := valid h
  have R := requirement h
  fin_cases i
  · obtain ⟨hc, he⟩ := both R.cond_2
    simp only [famAt, Fin.isValue, Nat.ofNat_pos, ↓reduceDIte, Fin.zero_eta, Nat.one_lt_ofNat,
      Fin.mk_one] at hc he
    simp only [decide_eq_true_eq] at hc he
    exact message_signed hc (v11 0) (v0 1) he
  · obtain ⟨hc, he⟩ := both R.cond_3
    simp only [famAt, Fin.isValue, Nat.one_lt_ofNat, ↓reduceDIte, Fin.mk_one,
      Nat.reduceLT] at hc he
    simp only [decide_eq_true_eq] at hc he
    exact message_signed hc (v11 1) (v0 2) he
  · obtain ⟨hc, he⟩ := both R.cond_4
    simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.reduceLT] at hc he
    simp only [decide_eq_true_eq] at hc he
    exact message_signed hc (v11 2) (v0 3) he
  · obtain ⟨hc, he⟩ := both R.cond_5
    simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.reduceLT] at hc he
    simp only [decide_eq_true_eq] at hc he
    exact message_signed hc (v11 3) (v0 4) he

/-- The requirement's step-zero state-selector check: `X^(2 y₀)`. -/
theorem state_selector0 (h : W_GeneratedClaim.Runs model external execution) :
    message (n := 8) (external.input_33 0) (external.input_34 0) =
      Xe (n := 8) (2 * (external.input_32 0).toNat) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, -, v32, -, v34, -⟩ := valid h
  have R := requirement h
  obtain ⟨hc, he⟩ := both R.cond_6
  simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.reduceLT, Nat.lt_irrefl, Nat.zero_lt_one,
    Nat.ofNat_pos, Fin.zero_eta] at hc he
  simp only [decide_eq_true_eq] at hc he
  have hy := v32 0
  rw [message_signed (a := external.input_32 0 * 2) hc (v34 0) (by omega) (by linarith),
    message_one_nat (by omega)]
  congr 1
  omega

/-- The requirement's step-one state-selector check: `X^(2 y₁)`. -/
theorem state_selector1 (h : W_GeneratedClaim.Runs model external execution) :
    message (n := 8) (external.input_35 0) (external.input_36 0) =
      Xe (n := 8) (2 * (external.input_32 1).toNat) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -,
    -, -, -, v32, -, -, -, v36, -⟩ := valid h
  have R := requirement h
  obtain ⟨hc, he⟩ := both R.cond_9
  simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.reduceLT, Nat.lt_irrefl, Nat.zero_lt_one,
    Nat.ofNat_pos, Fin.zero_eta, Nat.one_lt_ofNat, Fin.mk_one] at hc he
  simp only [decide_eq_true_eq] at hc he
  have hy := v32 1
  rw [message_signed (a := external.input_32 1 * 2) hc (v36 0) (by omega) (by linarith),
    message_one_nat (by omega)]
  congr 1
  omega

/-- The requirement's step-one label checks: label `0` is `X^((a₂ + a₄) mod 16)`. -/
theorem step1_label0 (h : W_GeneratedClaim.Runs model external execution) :
    message (n := 8) (external.input_14 0) (external.input_15 0) =
      message (n := 8) 1 ((external.input_0 2 + external.input_0 4) % 16) := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, v15, -⟩ := valid h
  have R := requirement h
  obtain ⟨hc, he⟩ := both R.cond_7
  simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.reduceLT, Nat.ofNat_pos, Fin.zero_eta,
    zero_add] at hc he
  simp only [decide_eq_true_eq] at hc he
  exact message_signed hc (v15 0) ⟨Int.emod_nonneg _ (by norm_num),
    by have := Int.emod_lt_of_pos (external.input_0 2 + external.input_0 4) (by norm_num : (0 : Int) < 16); omega⟩ he

/-- The requirement's step-one label checks: label `1` is the unit. -/
theorem step1_label1 (h : W_GeneratedClaim.Runs model external execution) :
    message (n := 8) (external.input_14 1) (external.input_15 1) = 1 := by
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, v15, -⟩ := valid h
  have R := requirement h
  obtain ⟨hc, he⟩ := both R.cond_8
  simp only [famAt, Fin.isValue, ↓reduceDIte, Nat.one_lt_ofNat, Fin.mk_one] at hc he
  simp only [decide_eq_true_eq] at hc he
  exact message_unit hc (v15 1) he

/-- Every external condition of `Online0Inputs` follows from the validity contract and the
requirement. -/
noncomputable def external_conditions (h : W_GeneratedClaim.Runs model external execution) :
    Online0Claim.ExternalConditions external := by
  have v := valid h
  have R := requirement h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, v16, v17, v18, v19, v20, v21, -, -, -,
    -, -, -, -, -, -, -, -, -, -, -, -, -, v38, v39, v40⟩ := v
  exact {
    label_payload := label_payload h
    input_payload := input_payload h
    σsel0 := Classical.choose (onehot (M := 8) external.input_18 external.input_39 v18 v39 R.loop_302)
    σselector0 := Classical.choose_spec
      (onehot (M := 8) external.input_18 external.input_39 v18 v39 R.loop_302)
    σsel1 := Classical.choose (onehot (M := 8) external.input_19 external.input_39 v19 v39 R.loop_304)
    σselector1 := Classical.choose_spec
      (onehot (M := 8) external.input_19 external.input_39 v19 v39 R.loop_304)
    fsel0 := Classical.choose
      (onehot (M := 7200) external.input_20 external.input_38 v20 v38 R.loop_306)
    fselector0 := Classical.choose_spec
      (onehot (M := 7200) external.input_20 external.input_38 v20 v38 R.loop_306)
    fsel1 := Classical.choose
      (onehot (M := 7200) external.input_21 external.input_38 v21 v38 R.loop_308)
    fselector1 := Classical.choose_spec
      (onehot (M := 7200) external.input_21 external.input_38 v21 v38 R.loop_308)
    ssel0 := Classical.choose
      (onehot (M := 36000) external.input_16 external.input_40 v16 v40 R.loop_298)
    sselector0 := Classical.choose_spec
      (onehot (M := 36000) external.input_16 external.input_40 v16 v40 R.loop_298)
    ssel1 := Classical.choose
      (onehot (M := 36000) external.input_17 external.input_40 v17 v40 R.loop_300)
    sselector1 := Classical.choose_spec
      (onehot (M := 36000) external.input_17 external.input_40 v17 v40 R.loop_300) }

end RunFacts

/-! ## Step zero: the online inputs of the actual run -/

section StepZero

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- The step-zero state LUT input is `X^(2 (z₀ + y₀))`. -/
theorem state_plain (h : W_GeneratedClaim.Runs model external execution) :
    Online0Claim.statePlain external =
      Xe (n := 8) (2 * (external.input_37.toNat + (external.input_32 0).toNat)) := by
  have hz := (valid h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  rw [Online0Claim.statePlain, Online0Claim.stateSeed, state_selector0 h,
    message_one_nat (a := external.input_37 * 2) (by omega)]
  rw [show (external.input_37 * 2).toNat = 2 * external.input_37.toNat by omega, mul_add]
  exact (pow_add _ _ _).symm

/-- The smudge table value is the signed odd readout digit. -/
theorem smudgeValue_digit (e : Nat) :
    PreprocessSwitch.smudgeValue e = ((smudgeReadoutDigit 8 8 e : Int) : ErrorPoly 8) := by
  unfold PreprocessSwitch.smudgeValue PreprocessSwitch.signedValue smudgeReadoutDigit
  rw [Int.cast_mul]
  congr 1
  have hk : e % 8 < 8 := Nat.mod_lt _ (by decide)
  have key : ∀ k < 8, ((MxxIR.selectInt (Int.ofNat k) [1, 1, 3, 3, 5, 5, 7, 7] : Int) : ErrorPoly 8) =
      ((readoutVal (8 / 2) 8 k : Nat) : Int) := by
    intro k hk
    interval_cases k <;> simp [MxxIR.selectInt, readoutVal_eq]
  simpa using key _ hk

/-- The step-zero smudge offset constant is the modular half. -/
theorem half0 {backend : BackendContext} {params : W_Stage_online_0.Params}
    (ctx : W_Stage_online_0.generatedRoot.Ctx backend model params) :
    W_Stage_online_0.generatedRoot.w_1274_0 ctx 0 0 =
      ((smudgeHalf0 : Int) : ExactPoly W_Stage_online_0.modulus_0 8) := by
  simp [W_Stage_online_0.generatedRoot.w_1274_0, MxxRuntime.matrixPolynomial, smudgeHalf0]

end StepZero

/-! ## Step one with a general state selector -/

section Online1R

open Mxx.ExponentLut.Refresh RefreshTopProof.GenInst RefreshTopProof.GenFinal StepTransition
  W_Stage_online_1
open scoped BigOperators


/-- `Online1Inputs` with the state selector's plaintext `rho` left general: the obfuscator's
step-one bit enters the state LUT through the preprocessed state RHS
`message (input_35 0) (input_36 0) = X^(2 y₁)`, which `Online1Inputs.stateRhs` fixes to `1`
(that is, to `y₁ = 0`). Every other field is that of `Online1Inputs`. -/
structure Online1InputsR {backend : BackendContext} {model : HashModel}
    {oldParams : W_Stage_online_0.Params} {params : Params}
    {importParams : W_Stage_import_1.Params} {preParams : W_Stage_preprocess_1_1.Params}
    (old : W_Stage_online_0.generatedRoot.Ctx backend model oldParams)
    (input : W_Stage_import_1.generatedRoot.Ctx importParams)
    (pre : W_Stage_preprocess_1_1.generatedRoot.Ctx backend model preParams)
    (online : generatedRoot.Ctx backend model params) (sm : ErrorPoly 8)
    (anchor : ExactMatrix modulus_0 8 2 54) where
  /-- The retained private payload. -/
  sp : ErrorPoly 8
  /-- The refreshed state value. -/
  state : ErrorPoly 8
  /-- The refreshed label values. -/
  labels : Fin 2 → ErrorPoly 8
  /-- The remaining refreshed labels, not consumed by the last program. -/
  otherLabels : Fin 4 → ErrorPoly 8
  /-- The common gadget of the root, selector, and switching relations. -/
  gadget : ExactMatrix modulus_0 8 1 25
  /-- The gadget follows the runtime layout. -/
  hgadget : gadgetMatrixRuns backend 4096 25 gadget
  /-- Fresh error budget in the step-zero refresh parts. -/
  bound : Nat
  /-- The fresh errors retained by step zero. -/
  fresh : Fin 36 → ErrorMatrix 8 1 25
  /-- The target of every step-zero part. -/
  target : Fin 36 → ExactPoly modulus_0 8
  /-- Step zero's decoder-success premise, including every fresh error. -/
  parts : ∀ index, W_Stage_online_0.generatedRoot.w_1281_0 old index =
    reducePoly modulus_0 8 sm • MxxExponentLut.rowOf
        (W_Stage_online_0.generatedRoot.w_227_0 old index) 0 +
      MxxExponentLut.rowOf (W_Stage_online_0.generatedRoot.w_227_0 old index) 1 -
      target index • gadget + reduceMatrix modulus_0 8 1 25 (fresh index)
  /-- The step-zero fresh errors satisfy the advertised common budget. -/
  freshBound : ∀ index, CoeffBound (fresh index) bound
  /-- Step zero's new root keys agree with the next step's hash seed. -/
  keys : ∀ (root : ImportRoot) (member : Fin root.members),
    nextKeys old (rootTarget root member) =
      keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 root.prefix (root.tags member)
  /-- Step zero's targets are the values required by the last program. -/
  targets : ∀ (root : ImportRoot) (member : Fin root.members) (component : Fin 2),
    target (partIndex (rootTarget root member) component) = reducePoly modulus_0 8
      (Online1Assembly.values labels otherLabels state root member *
        unitPair (rootPayload sm sp root member) component)
  /-- The preprocessing payload is the retained private payload. -/
  preprocessSecret : W_Stage_preprocess_1_1.generatedRoot.w_1_0 pre =
    fun _ _ ↦ reducePoly modulus_0 8 sp
  /-- External import payloads agree with the refreshed label plaintexts. -/
  payload : ∀ member, message (import1Coefficient input member)
    (import1Exponent input member) = labels member
  /-- The common root mask is small. -/
  smallMask : polyNorm sm ≤ 1
  /-- Public labels are small. -/
  smallLabels : ∀ member, polyNorm (labels member) ≤ 1
  /-- The state cell's public monomial multiplier is small. -/
  smallStateLabel : polyNorm (Xe (n := 8) 2 * labels 1) ≤ 1
  /-- The product passed to the state cell is small. -/
  smallStateProduct : polyNorm (Xe (n := 8) 2 * labels 1 * state) ≤ 1
  /-- The selected coordinate in each first-bucket member. -/
  select0 : Fin 8 → Fin 2
  /-- The selected coordinate in each second-bucket member. -/
  select1 : Fin 8 → Fin 2
  /-- The label selected by each first-bucket member. -/
  key0 : Fin 8 → Fin 2
  /-- The label selected by each second-bucket member. -/
  key1 : Fin 8 → Fin 2
  /-- First-bucket gather indices agree with the advertised labels. -/
  gather0 : ∀ member, generatedRoot.w_139_0 online (cellAt member (select0 member)) =
    ((key0 member).val : Int)
  /-- Second-bucket gather indices agree with the advertised labels. -/
  gather1 : ∀ member, generatedRoot.w_138_0 online (cellAt member (select1 member)) =
    ((key1 member).val : Int)
  /-- The first selector RHS is the advertised one-hot selector. -/
  selector0 : ∀ member component,
    message (W_Stage_preprocess_1_1.generatedRoot.w_73_0 pre (cellAt member component))
      (W_Stage_preprocess_1_1.generatedRoot.w_74_0 pre (cellAt member component)) =
        if component = select0 member then (1 : ErrorPoly 8) else 0
  /-- The second selector RHS is the advertised one-hot selector. -/
  selector1 : ∀ member component,
    message (W_Stage_preprocess_1_1.generatedRoot.w_77_0 pre (cellAt member component))
      (W_Stage_preprocess_1_1.generatedRoot.w_78_0 pre (cellAt member component)) =
        if component = select1 member then (1 : ErrorPoly 8) else 0
  /-- The state RHS multiplies by the retained private unit. -/
  rho : ErrorPoly 8
  /-- The state RHS multiplies by the selector plaintext `rho` and the retained private unit. -/
  stateRhs : message (W_Stage_preprocess_1_1.generatedRoot.w_31_0 pre 0)
    (W_Stage_preprocess_1_1.generatedRoot.w_32_0 pre 0) = rho
  /-- All second-bucket products are small, including unselected labels. -/
  smallBucket : ∀ key member, polyNorm (labels key * labels (key0 member)) ≤ 1
  /-- The state LUT value. -/
  ν : ErrorPoly 8
  /-- The state LUT inputs are small under every branch automorphism. -/
  smallState : ∀ branch : Fin 4, polyNorm (substitution (sigmaOf sigmas4 branch).toNat
    (sigmas4_odd branch) (Xe (n := 8) 2 * labels 1 * state * rho)) ≤ 1
  /-- The state value identity, with the actual encrypted table coefficients. -/
  stateWalsh : ∑ branch, reducePoly modulus_0 8
      (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch)
        (Xe (n := 8) 2 * labels 1 * state * rho)) * PreprocessSwitch.state1Coefficients branch =
    reducePoly modulus_0 8 ν
  /-- The eight smudge LUT values. -/
  smudgeValues : Fin 8 → ErrorPoly 8
  /-- The smudge LUT inputs are small under every branch automorphism. -/
  smallSmudge : ∀ member branch, polyNorm (substitution (sigmaOf sigmas8 branch).toNat
    (sigmas8_odd branch) (labels (key1 member) * labels (key0 member))) ≤ 1
  /-- Smudge values are the actual readout table values, including negacyclic signs. -/
  smudgeWalsh : ∀ member, ∑ branch, reducePoly modulus_0 8
      (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch)
        (labels (key1 member) * labels (key0 member))) *
        PreprocessSwitch.smudgeCoefficients branch = reducePoly modulus_0 8 (smudgeValues member)
  /-- The step-zero refreshed hint's integer error. -/
  hintError : ErrorMatrix 8 1 54
  /-- The refreshed hint's error budget. -/
  hintBound : Nat
  /-- The hint belongs to the sampled next anchor; this is a step-zero transition premise. -/
  hint : W_Stage_online_0.generatedRoot.w_1287_0 old =
    maskPart (reducePoly modulus_0 8 sm) (MxxExponentLut.rowOf anchor) +
      reduceMatrix modulus_0 8 1 54 hintError
  /-- The refreshed hint error satisfies its budget. -/
  hintErrorBound : CoeffBound hintError hintBound
  /-- The smudge residual after subtracting a half from every coefficient. -/
  residual : ErrorMatrix 8 1 1
  /-- The smudge residual budget. -/
  residualBound : Nat
  /-- The explicit smudge obligation `hTH` of `online1_output`. -/
  hTH : Online1Assembly.smudgeTarget online gadget smudgeValues - generatedRoot.w_186_0 online =
    reduceMatrix modulus_0 8 1 1 residual
  /-- Every smudge residual coefficient is small. -/
  residualSmall : CoeffBound residual residualBound
  /-- The complete strict decoder inequality, with the derived state and smudge budgets. -/
  noise : 2 * (54 * 8 * hintBound * 22011706 +
    25 * 8 * Online1Assembly.stateBound bound * 2048 +
    25 * 8 * Online1Assembly.smudgeBound bound * 2048 + residualBound) <
      1606938044254428791916078459060479968194539524630727034176641

namespace Online1InputsR

variable {backend : BackendContext} {model : HashModel}
  {oldParams : W_Stage_online_0.Params} {params : Params}
  {importParams : W_Stage_import_1.Params} {preParams : W_Stage_preprocess_1_1.Params}
  {old : W_Stage_online_0.generatedRoot.Ctx backend model oldParams}
  {input : W_Stage_import_1.generatedRoot.Ctx importParams}
  {pre : W_Stage_preprocess_1_1.generatedRoot.Ctx backend model preParams}
  {online : generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
  {anchor : ExactMatrix modulus_0 8 2 54}
  (I : Online1InputsR old input pre online sm anchor)

/-- The exact remaining decoder inequality, with no synthetic or assumed encoding budgets. -/
def Noise1 : Prop :=
  2 * (54 * 8 * I.hintBound * 22011706 +
    25 * 8 * Online1Assembly.stateBound I.bound * 2048 +
    25 * 8 * Online1Assembly.smudgeBound I.bound * 2048 + I.residualBound) <
      1606938044254428791916078459060479968194539524630727034176641

/-- The step-zero part forms supply every imported step-one root relation. -/
theorem root_rel {outputs : _} (F : W_Stage_online_0.generatedRoot.Facts old outputs) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25
        root.prefix (root.tags member))
        (fun component ↦ reducePoly modulus_0 8
          (Online1Assembly.values I.labels I.otherLabels I.state root member *
            unitPair (rootPayload sm I.sp root member) component) • I.gadget)
        (rootRows old root member) I.bound :=
  online0_root_rel old F I.freshBound I.parts
    (Online1Assembly.values I.labels I.otherLabels I.state)
    I.keys I.targets

/-- The exact selectors and the refreshed private root assemble both smudge PRF buckets. -/
theorem buckets {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (ha : ∀ member : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre member)
      (fun component ↦ reducePoly modulus_0 8 (I.labels member * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online member, generatedRoot.w_27_0 online member] I.bound)
    (hpay : ∀ member, generatedRoot.w_47_0 online member =
      fun _ _ ↦ reducePoly modulus_0 8 (I.labels member))
    (hpriv : Rel sm ![![generatedRoot.w_140_0 online, generatedRoot.w_143_0 online],
        ![generatedRoot.w_141_0 online, generatedRoot.w_144_0 online]]
      (fun component ↦ reducePoly modulus_0 8 (1 * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_142_0 online, generatedRoot.w_145_0 online] I.bound) :
    ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_153_0 online member, (generatedRoot.w_153_1 online member,
        (generatedRoot.w_153_2 online member, (generatedRoot.w_153_3 online member,
          (generatedRoot.w_153_4 online member, (generatedRoot.w_153_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * (I.labels (I.key1 member) * I.labels (I.key0 member)) * unitPair I.sp component) •
          I.gadget) (Online1Assembly.bucket1Bound I.bound) := by
  have hg := MxxExponentLut.centeredRebaseRuns_self hq (by decide : 0 < 8) I.gadget
  have hr0 := Preprocess1.rhs0 pre Fp I.preprocessSecret I.hgadget hg
  have hr1 := Preprocess1.rhs1 pre Fp I.preprocessSecret I.hgadget hg
  have hb0 := online1_bucket0 online F hq layout hlayout hexact I.smallLabels
    (fun member ↦ by simpa using I.smallLabels member) I.hgadget ha hpay I.hgadget hg hpriv
    (ν := fun index ↦ message (W_Stage_preprocess_1_1.generatedRoot.w_73_0 pre index)
      (W_Stage_preprocess_1_1.generatedRoot.w_74_0 pre index))
    (eB := fun _ _ ↦ 0) (BB := 0)
    (fun index component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 index, rows 1 index]) L.rhs0
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr0 index component)
    (fun _ _ _ _ _ ↦ by simp) I.select0 I.selector0
  have hfirst : ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_149_0 online member, (generatedRoot.w_149_1 online member,
        (generatedRoot.w_149_2 online member, (generatedRoot.w_149_3 online member,
          (generatedRoot.w_149_4 online member, (generatedRoot.w_149_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * I.labels (I.key0 member) * unitPair I.sp component) • I.gadget)
        (Online1Assembly.bucket0Bound I.bound) := by
    intro member
    obtain ⟨key, hkey, hrel⟩ := hb0 member
    have heq : key = I.key0 member := Fin.ext (by have := I.gather0 member; omega)
    simpa only [heq, Nat.cast_one, Int.cast_one, mul_one, one_mul, Online1Assembly.bucket0Bound,
      Online1Assembly.cellBound] using hrel
  intro member
  obtain ⟨key, hkey, hrel⟩ := online1_bucket1 online F hq layout hlayout hexact
    I.smallLabels I.smallBucket I.hgadget ha hpay I.hgadget hg hfirst
    (ν := fun index ↦ message (W_Stage_preprocess_1_1.generatedRoot.w_77_0 pre index)
      (W_Stage_preprocess_1_1.generatedRoot.w_78_0 pre index))
    (eB := fun _ _ ↦ 0) (BB := 0)
    (fun index component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 index, rows 1 index]) L.rhs1
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [Nat.cast_one, Int.cast_one, one_mul] using hr1 index component)
    (fun _ _ _ _ _ ↦ by simp) I.select1 I.selector1 member
  have heq : key = I.key1 member := Fin.ext (by have := I.gather1 member; omega)
  simpa only [heq, Online1Assembly.bucket1Bound, Online1Assembly.cellBound] using hrel

/-- Exact preprocessing and the imported state relation discharge the state LUT equation. -/
theorem state_lut {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (ha : Rel sm (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre 1)
      (fun component ↦ reducePoly modulus_0 8 (I.labels 1 * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online 1, generatedRoot.w_27_0 online 1] I.bound)
    (hpay : generatedRoot.w_47_0 online 1 = fun _ _ ↦ reducePoly modulus_0 8 (I.labels 1))
    (hstate : Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 statePre [])
      (fun component ↦ reducePoly modulus_0 8 (I.state * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_46_0 online, generatedRoot.w_63_0 online] I.bound)
    (hhelp : ∀ branch : Fin 4, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_13_0 online branch, generatedRoot.w_15_0 online branch] I.bound) :
    ∃ error : ErrorMatrix 8 1 25, CoeffBound error (Online1Assembly.stateBound I.bound) ∧
      generatedRoot.w_124_0 online =
        maskPart (reducePoly modulus_0 8 sm) (stateKey1 online) -
          reducePoly modulus_0 8 (1 * I.ν) • I.gadget + reduceMatrix modulus_0 8 1 25 error := by
  have hg := MxxExponentLut.centeredRebaseRuns_self hq (by decide : 0 < 8) I.gadget
  have hr := Preprocess1.rhs2 pre Fp I.preprocessSecret I.hgadget hg
  have hi := online1_lut_input online F hq layout hlayout hexact
    I.smallStateLabel I.smallStateProduct I.hgadget ha hpay I.hgadget hg hstate
    (T := fun component ↦ reducePoly modulus_0 8 (I.rho * unitPair I.sp component) • I.gadget)
    (eB := fun _ ↦ 0) (BB := 0)
    (fun component ↦ by
      have hrhs := congrArg (fun rows ↦ ![rows 0 0, rows 1 0]) L.rhs2
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one] at hrhs
      rw [L.seed, congrFun hrhs component]
      simpa only [I.stateRhs] using hr 0 component)
    (fun _ _ _ _ ↦ by simp)
  have hs := PreprocessSwitch.preprocess1_state_switching pre Fp hq
    I.preprocessSecret I.hgadget
  rw [← L.state] at hs
  have hsw : ∀ branch : Fin 4, ∃ error : ErrorMatrix 8 1 25, CoeffBound error 0 ∧
      reducePoly modulus_0 8
          (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) I.sp) •
          generatedRoot.w_7_0 online branch + generatedRoot.w_11_0 online branch =
        (PreprocessSwitch.state1Coefficients branch * reducePoly modulus_0 8 1) • I.gadget +
          reduceMatrix modulus_0 8 1 25 error := by
    intro branch
    simpa [unitPair] using hs 1 branch
  have he := online1_state_lut online F hq layout hlayout hexact I.smallMask I.hgadget hg
    (by
      have e : ∀ j, Xe (n := 8) 2 * I.labels 1 * I.state * (I.rho * unitPair I.sp j) =
          (((1 : Nat) : Int) : ErrorPoly 8) * (Xe (n := 8) 2 * I.labels 1 * I.state * I.rho) *
            unitPair I.sp j := fun j ↦ by push_cast; ring
      simpa only [smul_smul, ← map_mul, e] using hi)
    hhelp hsw
    I.smallState I.stateWalsh
  obtain ⟨error, hbound, heq⟩ := he
  exact ⟨error, hbound, by simpa [stateKey1, maskPart] using heq⟩

/-- The smudge readouts discharge the decoder's smudge equation with bounded digit noise. -/
theorem smudge {outputs : _} (F : generatedRoot.Facts online outputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hin : ∀ member : Fin 8, EncRel sm
      (generatedRoot.w_153_0 online member, (generatedRoot.w_153_1 online member,
        (generatedRoot.w_153_2 online member, (generatedRoot.w_153_3 online member,
          (generatedRoot.w_153_4 online member, (generatedRoot.w_153_5 online member, ()))))))
      (fun component ↦ reducePoly modulus_0 8
        (1 * (I.labels (I.key1 member) * I.labels (I.key0 member)) * unitPair I.sp component) •
          I.gadget) (Online1Assembly.bucket1Bound I.bound))
    (hhelp : ∀ branch : Fin 8, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_136_0 online branch, generatedRoot.w_137_0 online branch] I.bound) :
    ∃ error : ErrorMatrix 8 1 1,
      CoeffBound error (25 * 8 * Online1Assembly.smudgeBound I.bound * 2048) ∧
      generatedRoot.w_183_0 online = maskPart (reducePoly modulus_0 8 sm) (smudgeKeyS1 online) -
        Online1Assembly.smudgeTarget online I.gadget I.smudgeValues +
          reduceMatrix modulus_0 8 1 1 error := by
  have hs := PreprocessSwitch.preprocess1_smudge_switching pre Fp hq
    I.preprocessSecret I.hgadget
  rw [← L.smudge] at hs
  have ht := online1_smudge_total online F hq layout hlayout hexact I.smallMask I.hgadget
    (MxxExponentLut.centeredRebaseRuns_self hq (by decide) I.gadget) hin hhelp
    (PreprocessSwitch.switching_errors hs) I.smallSmudge I.smudgeWalsh
  obtain ⟨error, hbound, heq⟩ := Rel.comp ht 1
  obtain ⟨digits, hdigits, hdb⟩ := F.c7.h182_0.2.2.2
  refine ⟨error * digits, coeffBound_mul (by decide) hbound hdb, ?_⟩
  have htarget : (∑ member : Fin 8, fun component ↦
      reducePoly modulus_0 8 (Xe (n := 8) member) •
        (reducePoly modulus_0 8 (1 * I.smudgeValues member * unitPair I.sp component) •
          I.gadget)) 1 =
      reducePoly modulus_0 8 (Online1Assembly.smudgeValue I.smudgeValues) • I.gadget := by
    simp only [Finset.sum_apply, unitPair, Matrix.cons_val_one, Matrix.cons_val_zero,
      one_mul, mul_one, smul_smul, ← map_mul, ← Finset.sum_smul, ← map_sum,
      Online1Assembly.smudgeValue]
  have hrow : generatedRoot.w_175_0 online = generatedRoot.w_174_5 online :=
    key1_slice_full F.c7.h175_0
  change generatedRoot.w_175_0 online * generatedRoot.w_182_0 online = _
  rw [hrow]
  change generatedRoot.w_174_5 online = _ at heq
  rw [htarget] at heq
  rw [heq, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, hdigits, ← reduceMatrix_mul, ← hdigits]
  rfl

/-- Refreshed roots, exact preprocessing, the solved preimage, and the explicit residual
discharge every semantic input to `online1_output`. -/
theorem output {oldOutputs : _} (F0 : W_Stage_online_0.generatedRoot.Facts old oldOutputs)
    {importOutputs : _} (Fi : W_Stage_import_1.generatedRoot.Facts input importOutputs)
    {preOutputs : _} (Fp : W_Stage_preprocess_1_1.generatedRoot.Facts pre preOutputs)
    {outputs : _} (F : generatedRoot.Facts online outputs)
    (L : Online1Assembly.PreprocessLink online pre)
    (hinput : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows old port.root member)
    (honline : ∀ (port : Import1Port) (member : Fin port.members),
      import1OnlineRows online port member = import1OutputRows input port member)
    (hpayload : ∀ member, import1OnlinePayload online member = import1Payload input member)
    (hhint : generatedRoot.w_0_0 online = W_Stage_online_0.generatedRoot.w_1287_0 old)
    (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    {Pd : ErrorMatrix 8 54 1}
    (hP : generatedRoot.w_1_0 online = reduceMatrix modulus_0 8 54 1 Pd)
    (hPd : CoeffBound Pd 22011706)
    (hpre : ∀ component, MxxExponentLut.rowOf anchor component * generatedRoot.w_1_0 online =
      stateKey1 online component * generatedRoot.w_130_0 online + smudgeKeyS1 online component)
    (column : Fin 1) (coefficient : Fin 8) :
    (generatedRoot.w_188_0 online 0 column).coeff coefficient =
      (((reducePoly modulus_0 8 (1 * I.ν)).coeff coefficient).val : ZMod modulus_1) := by
  have hq : 1 < modulus_0 := by decide
  have hrel := online0_online1_rel old input Fi online hinput honline
    (Online1Assembly.values I.labels I.otherLabels I.state) (I.root_rel F0)
  have ha : ∀ member : Fin 2, Rel sm
      (labelKeysG model (generatedRoot.w_14_0 online) modulus_0 25 labelPre member)
      (fun component ↦ reducePoly modulus_0 8 (I.labels member * unitPair sm component) • I.gadget)
      ![generatedRoot.w_16_0 online member, generatedRoot.w_27_0 online member] I.bound := by
    intro member
    simpa [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      rootPayload, Online1Assembly.values, labelKeysG] using hrel .labelState member
  have hp := import1_online_payload input Fi online hpayload I.payload
  have hs : Rel sm (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 statePre [])
      (fun component ↦ reducePoly modulus_0 8 (I.state * unitPair I.sp component) • I.gadget)
      ![generatedRoot.w_46_0 online, generatedRoot.w_63_0 online] I.bound := by
    simpa [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      rootPayload, Online1Assembly.values] using hrel .state ⟨0, by decide⟩
  have h4 : ∀ branch : Fin 4, (ImportRoot.exp4.sigma branch : Int) = sigmaOf sigmas4 branch :=
    by decide
  have h8 : ∀ branch : Fin 8, (ImportRoot.exp8.sigma branch : Int) = sigmaOf sigmas8 branch :=
    by decide
  have hh4 : ∀ branch : Fin 4, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_13_0 online branch, generatedRoot.w_15_0 online branch] I.bound := by
    intro branch
    simpa only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      Online1Assembly.values, rootPayload, ← h4 branch, Int.toNat_natCast] using
        hrel .helperState branch
  have hh8 : ∀ branch : Fin 8, Rel sm
      (keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas8 branch)])
      (fun component ↦ reducePoly modulus_0 8
        (1 * unitPair (substitution (sigmaOf sigmas8 branch).toNat (sigmas8_odd branch) sm)
          component) • I.gadget)
      ![generatedRoot.w_136_0 online branch, generatedRoot.w_137_0 online branch] I.bound := by
    intro branch
    simpa only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
      Online1Assembly.values, rootPayload, ← h8 branch, Int.toNat_natCast] using
        hrel .helperSmudge branch
  have hk : keyPairG model (generatedRoot.w_14_0 online) modulus_0 25 privatePrefix [] =
      ![![generatedRoot.w_140_0 online, generatedRoot.w_143_0 online],
        ![generatedRoot.w_141_0 online, generatedRoot.w_144_0 online]] := by
    funext side component
    fin_cases side <;> fin_cases component
    · exact (keyPairG_of_hash 0 0 (c := 0) (by norm_num) F.c6.h140_0).symm
    · exact (keyPairG_of_hash 0 1 (c := 1) (by norm_num) F.c6.h143_0).symm
    · exact (keyPairG_of_hash 1 0 (c := 2) (by norm_num) F.c6.h141_0).symm
    · exact (keyPairG_of_hash 1 1 (c := 3) (by norm_num) F.c6.h144_0).symm
  have hpriv := hrel .privateRoot ⟨0, by decide⟩
  simp only [Import1Port.root, ImportRoot.prefix, ImportRoot.tags, import1OnlineRows,
    rootPayload, Online1Assembly.values, hk] at hpriv
  have hbucket := I.buckets F Fp L hq layout hlayout hexact ha hp hpriv
  obtain ⟨el, hel, hlut⟩ := I.state_lut F Fp L hq layout hlayout hexact (ha 1) (hp 1) hs hh4
  obtain ⟨es, hes, hsmudge⟩ := I.smudge F Fp L hq layout hlayout hexact hbucket hh8
  obtain ⟨digits, hdigits, hdb⟩ := F.c6.h130_0.2.2.2
  apply online1_output online F layout hlayout hexact (by decide) I.hgadget hel
    (K0 := stateKey1 online 0) (K1 := stateKey1 online 1)
    (by simpa [stateKey1, maskPart] using hlut)
    (by rw [hhint]; exact I.hint) I.hintErrorBound hP hPd
    (by intro component; simpa [stateKey1] using hpre component)
    hsmudge hes I.hTH I.residualSmall hdigits hdb I.noise column coefficient

end Online1InputsR


/-- `claim_online1_output` for `Online1InputsR`, at any step-zero context `old` of the run, with the
three step-one contexts and their input tuples exposed. -/
theorem claim_online1_outputR {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution)
    (sm : ErrorPoly 8) (layout : RegularLayout modulus_0)
    (hlayout : Backend.backend.regularLayout modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (old : W_Stage_online_0.generatedRoot.Ctx Backend.backend model
        W_GeneratedClaim.stage_11_params)
    (F0 : W_Stage_online_0.generatedRoot.Facts old execution.«stage_11») :
    ∃ (input : W_Stage_import_1.generatedRoot.Ctx W_GeneratedClaim.stage_16_params)
      (pre : W_Stage_preprocess_1_1.generatedRoot.Ctx Backend.backend model
        W_GeneratedClaim.stage_12_params)
      (online : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_17_params),
      input.inputs = importClaim1Inputs external execution ∧
      pre.inputs = (external.input_1, execution.stage_0,
        external.input_3, external.input_35, external.input_36, external.input_31,
        external.input_30, external.input_22, external.input_39, external.input_23,
        external.input_39, ()) ∧
      online.inputs = importClaimOnline1Inputs external execution ∧
      generatedRoot.Facts online execution.«stage_17» ∧
      ∀ I : Online1InputsR old input pre online sm execution.«stage_1».2.2.1,
        ∀ (column : Fin 1) (coefficient : Fin 8),
          (execution.«stage_17».1 0 column).coeff coefficient =
            (((reducePoly modulus_0 8 (1 * I.ν)).coeff coefficient).val : ZMod modulus_1) := by
  have hp := (claim_preimages h).2
  obtain ⟨wi, hwi⟩ := importClaim1Inputs_runs h
  obtain ⟨wo, hwo⟩ := importClaimOnline1Inputs_runs h
  have runs := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, h12, -, h14, -⟩ := runs
  obtain ⟨wp, hwp⟩ := h12
  obtain ⟨wk, hwk⟩ := h14
  let input : W_Stage_import_1.generatedRoot.Ctx W_GeneratedClaim.stage_16_params :=
    ⟨importClaim1Inputs external execution, wi⟩
  let pre : W_Stage_preprocess_1_1.generatedRoot.Ctx Backend.backend model
      W_GeneratedClaim.stage_12_params := ⟨(external.input_1, execution.stage_0,
        external.input_3, external.input_35, external.input_36, external.input_31,
        external.input_30, external.input_22, external.input_39, external.input_23,
        external.input_39, ()), wp⟩
  let online : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_17_params :=
    ⟨importClaimOnline1Inputs external execution, wo⟩
  let keys := (fun inputs (_ : W_Stage_keys_online_1_1.generatedRoot.body Backend.backend
      model W_GeneratedClaim.stage_14_params inputs execution.stage_14 wk) ↦
        (⟨inputs, wk⟩ : W_Stage_keys_online_1_1.generatedRoot.Ctx Backend.backend model
          W_GeneratedClaim.stage_14_params)) _ hwk
  have Fi := W_Stage_import_1.generatedRoot.facts input hwi
  have Fp := W_Stage_preprocess_1_1.generatedRoot.facts pre hwp
  have Fo := generatedRoot.facts online hwo
  have Fk := W_Stage_keys_online_1_1.generatedRoot.facts keys hwk
  have L : Online1Assembly.PreprocessLink online pre := by
    constructor <;>
      simp only [generatedRoot.w_70_0, generatedRoot.w_5_0, generatedRoot.w_7_0,
        generatedRoot.w_9_0, generatedRoot.w_11_0, generatedRoot.w_132_0,
        generatedRoot.w_133_0, generatedRoot.w_134_0, generatedRoot.w_135_0,
        generatedRoot.w_146_0, generatedRoot.w_147_0, generatedRoot.w_150_0,
        generatedRoot.w_151_0, generatedRoot.w_72_0, generatedRoot.w_82_0,
        online, importClaimOnline1Inputs, Fp.c3.hout,
        W_Stage_preprocess_1_1.generatedRoot.outs, LutKey.fam4,
        W_Stage_preprocess_1_1.generatedRoot.w_0_0, pre]
  have hinput : ∀ (port : Import1Port) (member : Fin port.members),
      import1InputRows input port member = rootRows old port.root member := by
    intro port member
    cases port <;>
      simp only [import1InputRows, Import1Port.root, rootRows, input, importClaim1Inputs,
        W_Stage_import_1.generatedRoot.w_0_0, W_Stage_import_1.generatedRoot.w_1_0,
        W_Stage_import_1.generatedRoot.w_2_0, W_Stage_import_1.generatedRoot.w_3_0,
        W_Stage_import_1.generatedRoot.w_7_0, W_Stage_import_1.generatedRoot.w_8_0,
        W_Stage_import_1.generatedRoot.w_9_0, W_Stage_import_1.generatedRoot.w_10_0,
        W_Stage_import_1.generatedRoot.w_12_0, W_Stage_import_1.generatedRoot.w_13_0,
        F0.c70.hout, W_Stage_online_0.generatedRoot.outs]
  have honline := import1_claim_wiring input online external execution Fi rfl
  have hpay := import1_claim_payload_wiring input online external execution Fi rfl
  have hhint : generatedRoot.w_0_0 online = W_Stage_online_0.generatedRoot.w_1287_0 old := by
    simp only [generatedRoot.w_0_0, online, importClaimOnline1Inputs,
      F0.c70.hout, W_Stage_online_0.generatedRoot.outs]
  have hsolved : execution.«stage_1».2.2.1 * generatedRoot.w_1_0 online =
      W_Stage_keys_online_1_1.generatedRoot.w_185_0 keys := by
    have heq := hp.1
    rw [Fk.c9.hout] at heq
    exact heq
  have hpre := online1_hpre Fo Fk (by constructor <;> rfl) (by decide)
    layout hlayout hexact hsolved
  obtain ⟨Pd, hPd, hbound⟩ := hp.2
  refine ⟨input, pre, online, rfl, rfl, rfl, Fo, fun I column coefficient ↦ ?_⟩
  rw [Fo.c9.hout]
  exact I.output F0 Fi Fp Fo L hinput honline hpay hhint layout hlayout hexact
    hPd hbound hpre column coefficient

end Online1R

/-! ## Step zero: the inputs of the actual run and their noise -/

section Build0

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- The smudge residue of step zero, prime zero, with bound four (`online0_smudge_hTH`). -/
theorem residue0 {C : Online0Claim.Context model external execution} {sm tp : ErrorPoly 8}
    {M : Online0Claim.Members C.online sm tp}
    (P : Online0Claim.Plaintexts external M.σμ M.fμ M.sμ) :
    Online0Claim.Residue C.online P 4 := by
  intro i
  exact online0_smudge_hTH C.online C.online_facts _
    (fun o ↦ smudgeReadoutDigit 8 8 (P.smudgeExp o)) (fun o ↦ by rw [smudgeValue_digit])
    (fun o ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).1)
    (fun o ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).2) rfl
    (half0 C.online) i

/-- The step-zero data of a run: the secret lifts, the claim contexts and the PRF members. -/
structure Step0 (model : HashModel) (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) where
  /-- The mask (the hint's and every root's mask). -/
  sm : ErrorPoly 8
  /-- The LUT payload secret. -/
  tp : ErrorPoly 8
  hmask : execution.«stage_3» 0 0 = reducePoly W_Stage_online_0.modulus_0 8 sm
  hpay : execution.«stage_0» 0 0 = reducePoly W_Stage_online_0.modulus_0 8 tp
  hsm : polyNorm sm ≤ 1
  htp : polyNorm tp ≤ 1
  /-- The initial, import, preprocessing and online contexts of step zero. -/
  C : Online0Claim.Context model external execution
  /-- The three PRF member families. -/
  M : Online0Claim.Members C.online sm tp

/-- Every run has step-zero data. -/
noncomputable def step0 (h : W_GeneratedClaim.Runs model external execution) :
    Step0 model external execution :=
  let hm := Classical.choose_spec (mask_lift h)
  let hp := Classical.choose_spec (payload_lift h)
  let C := Classical.choice (Online0Claim.contexts h)
  { sm := Classical.choose (mask_lift h)
    tp := Classical.choose (payload_lift h)
    hmask := hm.1
    hpay := hp.1
    hsm := hm.2
    htp := hp.2
    C := C
    M := Classical.choice (C.members hm.1 hp.1 Backend.layout0 layout_some layout_exact
      (external_conditions h)) }

/-- The step-zero plaintext conditions, with state exponent `z₀ + y₀`. -/
noncomputable def Step0.P (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) :
    Online0Claim.Plaintexts external S.M.σμ S.M.fμ S.M.sμ :=
  Online0Claim.Plaintexts.of_state S.C S.M _ (state_plain h)

/-- The complete `Online0Inputs` of the run (`Online0Claim.Context.inputs`), with residue bound
four. -/
noncomputable def Step0.I (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) : Online0Inputs S.C.online S.sm :=
  S.C.inputs S.hmask S.hpay S.hsm Backend.layout0 layout_some layout_exact
    (external_conditions h) S.M (S.P h) 4 (residue0 (S.P h))

/-- The quotients `Q₀ / q_p` of the strict decoding cutoffs of step zero
(`Online0AssemblyP.online0PrimeQuotient`). -/
def quotient0 : Fin 5 → Nat :=
  ![1606938044254428791916078459060479968194539524630727034176641,
    1606938044254565807694577896670997561277593686540426466090209,
    1606938044254771331362327096896778694705812889646799470123121,
    1606938044255228050623992174508164557277769211170177223088561,
    1606938044255707605848740785410594412685698358681526360823809]

/-- The decoding noise condition of part `i` at prime `p` (`Online0InputsP.Noise`). -/
def noise0 {backend : BackendContext} {params : W_Stage_online_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
    (I : Online0Inputs Con sm) (p : Fin 5) (i : Fin 36) : Prop :=
  2 * (54 * 8 * 1 * 22011706 +
      (25 * 8 * I.gBound * 2048 + 25 * 8 * I.c1Bound * 2048 + 25 * 8 * I.cxBound i * 2048) +
      25 * 8 * ((25 * 25) * 8 * I.freshBound * 2048) * 2048 +
      (25 * 25) * 8 * I.smudgeBound * 2048 + I.BM) <
    quotient0 p

/-- The bound equalities of `Dio2Noise.Noise.Inputs0`. -/
structure Bounds0 {backend : BackendContext} {params : W_Stage_online_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
    (I : Online0Inputs Con sm) : Prop where
  sigma : I.σB = Online0Claim.selectorBucketBound (Online0Claim.selectorBucketBound 1)
  helper : I.Bh = 1
  samplerHelper : I.BhS = 1
  sigmaReadout : I.σBBr = 0
  sigmaSampler : I.σBBS = 0
  one : I.oneB = 1
  privateUnit : I.privB = 1
  state : I.stB = Online0Claim.selectorCellBound 1
  stateHelper : I.stBh = 1
  stateSwitch : I.stBB = 0
  label : I.Ba = 1
  stateLabel : I.Bo = 1
  autHelpers : I.a3Bh = 1 ∧ I.a5Bh = 1 ∧ I.a7Bh = 1 ∧
    I.a9Bh = 1 ∧ I.a11Bh = 1 ∧ I.a13Bh = 1 ∧ I.a15Bh = 1
  autSwitches : I.a3BB = 0 ∧ I.a5BB = 0 ∧ I.a7BB = 0 ∧
    I.a9BB = 0 ∧ I.a11BB = 0 ∧ I.a13BB = 0 ∧ I.a15BB = 0
  fresh : I.fB = Online0Claim.selectorBucketBound (Online0Claim.selectorBucketBound 1)
  freshSwitch : I.fBB = 0
  smudge : I.sB = Online0Claim.selectorBucketBound (Online0Claim.selectorBucketBound 1)
  smudgeSwitch : I.sBB = 0
  residue : I.BM = 4

/-- The actual inputs have the bounds of `Bounds0`. -/
theorem Step0.bounds (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) : Bounds0 (S.I h) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩, ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩,
    rfl, rfl, rfl, rfl, rfl⟩

/-- Every step-zero decoding inequality holds under `Bounds0` (as `Noise.online0_all_primes_noise`). -/
theorem noise0_of_bounds {backend : BackendContext} {params : W_Stage_online_0.Params}
    {Con : W_Stage_online_0.generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
    {I : Online0Inputs Con sm} (B : Bounds0 I) (p : Fin 5) (i : Fin 36) : noise0 I p i := by
  have hσ : I.σBound = 6044688588822783587178439036108800 := by
    norm_num [Online0Inputs.σBound, B.sigma, B.helper, B.samplerHelper, B.sigmaReadout,
      B.sigmaSampler, Online0Claim.selectorBucketBound, Online0Claim.selectorCellBound]
  have hg : I.gBound = 2475904445981812157308288629190164480000 := by
    rw [Online0Inputs.gBound, hσ]
  have hc : I.c1Bound = 409600 := by norm_num [Online0Inputs.c1Bound, B.one]
  have hs : I.stBound = 2199044730391756800 := by
    norm_num [Online0Inputs.stBound, B.state, B.stateHelper, B.stateSwitch,
      Online0Claim.selectorCellBound]
  have hp : I.pubBound = 1014130461074150259633475022516291371008000000 := by
    norm_num [Online0Inputs.pubBound, hg, hc]
  have hf : I.freshBound = 922346281253476499508428800 := by
    norm_num [Online0Inputs.freshBound, B.helper, B.fresh, B.freshSwitch,
      Online0Claim.selectorBucketBound, Online0Claim.selectorCellBound]
  have hsm : I.smudgeBound = 922346281253476499508428800 := by
    norm_num [Online0Inputs.smudgeBound, B.helper, B.smudge, B.smudgeSwitch,
      Online0Claim.selectorBucketBound, Online0Claim.selectorCellBound]
  obtain ⟨ha3, ha5, ha7, ha9, ha11, ha13, ha15⟩ := B.autHelpers
  obtain ⟨hz3, hz5, hz7, hz9, hz11, hz13, hz15⟩ := B.autSwitches
  have htarget : ∀ part, I.cxBound part ≤ 830775673711943892691742738445346028568707072000000 := by
    norm_num [Online0Inputs.cxBound, Online0Inputs.lab425Bound, Online0Inputs.lab543Bound,
      Online0Inputs.lab626Bound, Online0Inputs.lab680Bound, Online0Inputs.lab734Bound,
      Online0Inputs.sigBound, Online0Inputs.aut3Bound, Online0Inputs.aut5Bound,
      Online0Inputs.aut7Bound, Online0Inputs.aut9Bound, Online0Inputs.aut11Bound,
      Online0Inputs.aut13Bound, Online0Inputs.aut15Bound, Fin.forall_fin_succ,
      hσ, hg, hc, hs, hp, B.privateUnit, B.label, B.stateLabel,
      ha3, ha5, ha7, ha9, ha11, ha13, ha15, hz3, hz5, hz7, hz9, hz11, hz13, hz15]
  have hi := htarget i
  have hq : 680571431906852705552444235198697479435629248211202433992 < quotient0 p := by
    have all : ∀ p, 680571431906852705552444235198697479435629248211202433992 < quotient0 p := by
      norm_num [quotient0, Fin.forall_fin_succ]
    exact all p
  unfold noise0
  rw [hg, hc, hf, hsm, B.residue]
  omega

end Build0

/-! ## The remaining links between step zero and step one -/

section Links

open W_Stage_online_0 StepTransition

/-- `Online0InputsP.hTHP`: the smudge residues of primes `1 … 4`. -/
def HTHP {backend : BackendContext} {model : HashModel} {params : Params}
    {Con : generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
    (I : Online0Inputs Con sm) : Prop :=
  ∀ (p : Fin 4) (i : Fin 36), ∃ residue : ErrorMatrix 8 1 25,
    CoeffBound residue I.BM ∧
      Matrix.of (fun _ col ↦ reducePoly modulus_0 8
        (cellVal (C := 4500) (by norm_num) I.sΔ I.sν
          ⟨125 * i + 25 * (p.val + 1) + col.val,
            by have := i.isLt; have := p.isLt; have := col.isLt; omega⟩) * I.hconst) -
        generatedRoot.w_1276_0 Con = reduceMatrix modulus_0 8 1 25 residue

/-- `Online0Inputs.partRow`: the full intended row `β A'₀ + A'₁ - T_i G' + E_i` of part `i`. -/
noncomputable def partRow0 {backend : BackendContext} {model : HashModel} {params : Params}
    {Con : generatedRoot.Ctx backend model params} {sm : ErrorPoly 8}
    (I : Online0Inputs Con sm) (i : Fin 36) : ExactMatrix modulus_0 8 1 25 :=
  reducePoly modulus_0 8 I.xσ • MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 0 +
    MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 1 -
    reducePoly modulus_0 8 (I.cxPlain i) • generatedRoot.w_57_0 Con +
    reduceMatrix modulus_0 8 1 25 (I.E0 i)

/-- **The proof obligation not discharged in this module.** It is a statement about the
generated stages alone (no external plaintext appears), quantified over every run, and is proven
from `Dio2Online0AssemblyP` once that module can be imported here (see the end of this file). -/
structure MissingLinks : Prop where
  /-- Every step-zero refresh part returns its full CRT row. This is `claim_online0_rows`
  (`Dio2Online0AssemblyP`) for a given online context: its stage-level lemmas
  `Online0InputsP.part_term0 … part_term4` apply to any context with the claim's inputs, and
  `Online0InputsP` is `I` together with `HTHP I`. It cannot be imported here: `Dio2KeyAgree0P1`
  and `Dio2KeyAgree1` both declare `Dio2Proof.smudgeKeyS1` and `Dio2Proof.claim_hpreS1`. -/
  online0Rows : ∀ {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}, W_GeneratedClaim.Runs model external execution →
    ∀ {sm : ErrorPoly 8}, execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm →
    ∀ Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params,
      Con.inputs = importClaimOnline0Inputs external execution →
      generatedRoot.Facts Con execution.«stage_11» →
      ∀ I : Online0Inputs Con sm, HTHP I → (∀ p i, noise0 I p i) →
      ∀ i, generatedRoot.w_1281_0 Con i = partRow0 I i

end Links

/-! ## The refreshed hint -/

section Hint

open W_Stage_online_0

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- A static family lookup at a natural index. -/
theorem get_static {α : Type} {size : Nat} {f : Fin size → α} {x : α} {k : Int} (i : Fin size)
    (hk : (i.val : Int) = k) (h : familyGetStatic f k x) : x = f i := by
  obtain ⟨position, hposition, rfl⟩ := h
  congr 1
  apply Fin.ext
  omega

/-- The key columns of the anchor parts `32 + j` are the `25`-column blocks of the padded next
anchor `w_218 = (stage_2.2.1 ‖ padding)`. -/
theorem anchor_keys (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    {outputs : _} (F : generatedRoot.Facts Con outputs) (j : Fin 3) (k : Fin 2) (col : Fin 25) :
    generatedRoot.w_227_0 Con ⟨32 + j.val, by omega⟩ k col =
      generatedRoot.w_218_0 Con k ⟨25 * j.val + col.val, by omega⟩ := by
  have hpack := import_familyPack_get F.c6.h227_0
  have slice : ∀ {x y : ExactMatrix modulus_0 8 2 25} (a : Nat) {b : Int} (_ : a + 25 ≤ 100),
      sliceMatrix (generatedRoot.w_218_0 Con) 0 2 (a : Int) b x → modulusReduceRuns x y →
      y k col = generatedRoot.w_218_0 Con k ⟨a + col.val, by have := col.isLt; omega⟩ := by
    intro x y a b ha hs hr
    rw [modulusReduce_same (by decide) hr]
    obtain ⟨-, -, -, -, -, -, -, -, hs⟩ := hs
    rw [hs k col (by simp) (by have := col.isLt; simp; omega)]
    congr 1
    apply Fin.ext
    simp
  fin_cases j
  · rw [hpack]
    exact slice 0 (by norm_num) F.c5.h219_0 F.c5.h220_0
  · rw [hpack]
    exact slice 25 (by norm_num) F.c5.h221_0 F.c5.h222_0
  · rw [hpack]
    exact slice 50 (by norm_num) F.c6.h223_0 F.c6.h224_0

/-- The next anchor's columns are the first 54 columns of the padded anchor `w_218`. -/
theorem anchor_columns (h : W_GeneratedClaim.Runs model external execution)
    (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    (hin : Con.inputs = importClaimOnline0Inputs external execution)
    {outputs : _} (F : generatedRoot.Facts Con outputs) (k : Fin 2) (c : Nat) (hc : c < 54) :
    generatedRoot.w_218_0 Con k ⟨c, by omega⟩ = execution.«stage_1».2.2.1 k ⟨c, hc⟩ := by
  rw [F.c5.h218_0 k ⟨c, by omega⟩, dif_pos hc, ← claim_anchor1 h]
  rw [generatedRoot.w_216_0, hin]
  rfl

/-- **The refreshed hint**: the first 54 columns of the anchor parts `32 … 35` are the next
anchor masked by the next mask `β`, with their fresh errors. -/
theorem hint_of_rows (h : W_GeneratedClaim.Runs model external execution)
    (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    (hin : Con.inputs = importClaimOnline0Inputs external execution)
    (F : generatedRoot.Facts Con execution.«stage_11»)
    (β : ErrorPoly 8) (T : Fin 36 → ExactPoly modulus_0 8) (E : Fin 36 → ErrorMatrix 8 1 25)
    (rows : ∀ i, generatedRoot.w_1281_0 Con i =
      reducePoly modulus_0 8 β • MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 0 +
        MxxExponentLut.rowOf (generatedRoot.w_227_0 Con i) 1 -
        T i • generatedRoot.w_57_0 Con + reduceMatrix modulus_0 8 1 25 (E i))
    (hE : ∀ i, CoeffBound (E i) 8) (hT : ∀ i : Fin 36, 32 ≤ i.val → T i = 0) :
    ∃ e : ErrorMatrix 8 1 54, CoeffBound e 8 ∧
      generatedRoot.w_1287_0 Con = Mxx.ExponentLut.Refresh.maskPart (reducePoly modulus_0 8 β)
        (MxxExponentLut.rowOf execution.«stage_1».2.2.1) + reduceMatrix modulus_0 8 1 54 e := by
  -- One anchor part, entrywise.
  have hpart : ∀ (i : Fin 36) (j : Fin 3) (col : Fin 25) (c : Fin 54), i.val = 32 + j.val →
      c.val = 25 * j.val + col.val →
      generatedRoot.w_1281_0 Con i 0 col =
        reducePoly modulus_0 8 β * execution.«stage_1».2.2.1 0 c +
          execution.«stage_1».2.2.1 1 c + reducePoly modulus_0 8 (E i 0 col) := by
    intro i j col c hi hc
    have hi' : i = ⟨32 + j.val, by omega⟩ := Fin.ext hi
    have hc' : c = ⟨25 * j.val + col.val, by omega⟩ := Fin.ext hc
    rw [rows, hT _ (by omega)]
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, MxxExponentLut.rowOf,
      smul_eq_mul, zero_smul, Matrix.zero_apply, sub_zero, reduceMatrix]
    rw [hi', hc', anchor_keys Con F j 0 col, anchor_keys Con F j 1 col,
      anchor_columns h Con hin F 0 _ (by omega), anchor_columns h Con hin F 1 _ (by omega)]
  let e : ErrorMatrix 8 1 54 := fun _ c ↦
    if h1 : c.val < 25 then E 32 0 ⟨c.val, h1⟩
    else if h2 : c.val < 50 then E 33 0 ⟨c.val - 25, by omega⟩
    else E 34 0 ⟨c.val - 50, by omega⟩
  refine ⟨e, fun r c k ↦ ?_, ?_⟩
  · simp only [e]
    split_ifs
    · exact hE _ _ _ _
    · exact hE _ _ _ _
    · exact hE _ _ _ _
  have p32 := get_static (f := generatedRoot.w_1281_0 Con) 32 rfl F.c49.h1282_0
  have p33 := get_static (f := generatedRoot.w_1281_0 Con) 33 rfl F.c49.h1283_0
  have p34 := get_static (f := generatedRoot.w_1281_0 Con) 34 rfl F.c49.h1284_0
  funext r c
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  have hc := c.isLt
  have h1287 : generatedRoot.w_1287_0 Con 0 c =
      generatedRoot.w_1286_0 Con 0 ⟨c.val, by omega⟩ := by
    obtain ⟨-, -, -, -, -, -, -, -, hs⟩ := F.c49.h1287_0
    rw [hs 0 c (by simp) (by simp; omega)]
    congr 1
    apply Fin.ext
    simp
  rw [h1287, F.c49.h1286_0, dif_pos (show c.val < 75 by omega), F.c49.h1284_0_1]
  simp only [Matrix.add_apply, Mxx.ExponentLut.Refresh.maskPart, Matrix.smul_apply,
    MxxExponentLut.rowOf, smul_eq_mul, reduceMatrix, e]
  by_cases h1 : c.val < 25
  · rw [dif_pos (show c.val < 50 by omega), F.c49.h1283_0_1, dif_pos h1, p32, dif_pos h1]
    exact hpart 32 0 ⟨c.val, h1⟩ c rfl (by simp)
  · by_cases h2 : c.val < 50
    · rw [dif_pos h2, F.c49.h1283_0_1, dif_neg h1, p33, dif_neg h1, dif_pos h2]
      exact hpart 33 1 ⟨c.val - 25, by omega⟩ c rfl (by simp; omega)
    · rw [dif_neg h2, p34, dif_neg h1, dif_neg h2]
      exact hpart 34 2 ⟨c.val - 50, by omega⟩ c rfl (by simp; omega)

/-- One refresh part's key columns are the hashed key pair of its root (component `j`). -/
theorem part_keys {q : Nat} {model : HashModel} {key : ByteArray} {pre : Blob}
    {tags : List HashTagComponent} (j : Fin 2) {c0 c1 : Int}
    (hc0 : c0 = 2 * ((0 : Fin 2) : Int) + (j : Int)) (hc1 : c1 = 2 * ((1 : Fin 2) : Int) + (j : Int))
    {A B : ExactMatrix q 8 1 25} {C D : ExactMatrix q 8 2 25}
    (hA : hashSample model pre (tags ++ [.bytes componentTagG, .u64Le c0]) key A)
    (hB : hashSample model pre (tags ++ [.bytes componentTagG, .u64Le c1]) key B)
    (hC : concatRows A B C) (hD : modulusReduceRuns C D) (side : Fin 2) :
    MxxExponentLut.rowOf D side = keyPairG model key q 25 pre tags side j := by
  rw [modulusReduce_same (by decide) hD]
  funext r col
  have hr : r = 0 := Subsingleton.elim _ _
  subst hr
  simp only [MxxExponentLut.rowOf]
  rw [hC side col]
  fin_cases side
  · simp only [Fin.isValue, Fin.val_zero, Nat.ofNat_pos, ↓reduceDIte]
    rw [keyPairG_of_hash 0 j hc0 hA]
    rfl
  · simp only [Fin.isValue, Fin.val_one, lt_self_iff_false, ↓reduceDIte]
    rw [keyPairG_of_hash 1 j hc1 hB]
    rfl

set_option maxHeartbeats 0 in
/-- **New-root key agreement**: the key columns of every exported next root are the keys hashed
from the shared seed `input_3`. -/
theorem root_keys (Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params)
    (hin : Con.inputs = importClaimOnline0Inputs external execution)
    {outputs : _} (F : generatedRoot.Facts Con outputs) :
    ∀ (root : ImportRoot) (member : Fin root.members),
      StepTransition.nextKeys Con (StepTransition.rootTarget root member) =
        keyPairG model external.input_3 modulus_0 25 root.prefix (root.tags member) := by
  have hk : generatedRoot.w_71_0 Con = external.input_3 := by
    rw [generatedRoot.w_71_0, hin]
    rfl
  rw [← hk]
  have hpack := import_familyPack_get F.c6.h227_0
  intro root member
  funext side comp
  simp only [StepTransition.nextKeys, hpack]
  cases root <;> fin_cases member <;> fin_cases comp
  all_goals simp only [StepTransition.rootTarget, StepTransition.partIndex, Fin.isValue,
    ImportRoot.prefix, ImportRoot.tags, ImportRoot.sigma, List.getElem_cons_zero,
    List.getElem_cons_succ, Fin.val_zero, Fin.val_one, Nat.reduceMul, Nat.reduceAdd,
    Matrix.cons_val_zero, Matrix.cons_val_one, ↓reduceIte, Fin.mk_one, Fin.zero_eta]
  all_goals first
      | exact part_keys 0 rfl rfl F.c1.h72_0 F.c1.h73_0 F.c1.h74_0 F.c1.h75_0 side
      | exact part_keys 1 rfl rfl F.c1.h76_0 F.c1.h77_0 F.c1.h78_0 F.c1.h79_0 side
      | exact part_keys 0 rfl rfl F.c1.h80_0 F.c1.h81_0 F.c1.h82_0 F.c1.h83_0 side
      | exact part_keys 1 rfl rfl F.c1.h84_0 F.c1.h85_0 F.c1.h86_0 F.c1.h87_0 side
      | exact part_keys 0 rfl rfl F.c1.h88_0 F.c1.h89_0 F.c1.h90_0 F.c1.h91_0 side
      | exact part_keys 1 rfl rfl F.c1.h92_0 F.c1.h93_0 F.c1.h94_0 F.c1.h95_0 side
      | exact part_keys 0 rfl rfl F.c2.h96_0 F.c2.h97_0 F.c2.h98_0 F.c2.h99_0 side
      | exact part_keys 1 rfl rfl F.c2.h100_0 F.c2.h101_0 F.c2.h102_0 F.c2.h103_0 side
      | exact part_keys 0 rfl rfl F.c2.h104_0 F.c2.h105_0 F.c2.h106_0 F.c2.h107_0 side
      | exact part_keys 1 rfl rfl F.c2.h108_0 F.c2.h109_0 F.c2.h110_0 F.c2.h111_0 side
      | exact part_keys 0 rfl rfl F.c2.h112_0 F.c2.h113_0 F.c2.h114_0 F.c2.h115_0 side
      | exact part_keys 1 rfl rfl F.c2.h116_0 F.c2.h117_0 F.c2.h118_0 F.c2.h119_0 side
      | exact part_keys 0 rfl rfl F.c2.h120_0 F.c2.h121_0 F.c2.h122_0 F.c2.h123_0 side
      | exact part_keys 1 rfl rfl F.c2.h124_0 F.c2.h125_0 F.c2.h126_0 F.c2.h127_0 side
      | exact part_keys 0 rfl rfl F.c3.h128_0 F.c3.h129_0 F.c3.h130_0 F.c3.h131_0 side
      | exact part_keys 1 rfl rfl F.c3.h132_0 F.c3.h133_0 F.c3.h134_0 F.c3.h135_0 side
      | exact part_keys 0 rfl rfl F.c3.h137_0 F.c3.h138_0 F.c3.h139_0 F.c3.h140_0 side
      | exact part_keys 1 rfl rfl F.c3.h142_0 F.c3.h143_0 F.c3.h144_0 F.c3.h145_0 side
      | exact part_keys 0 rfl rfl F.c3.h147_0 F.c3.h148_0 F.c3.h149_0 F.c3.h150_0 side
      | exact part_keys 1 rfl rfl F.c3.h152_0 F.c3.h153_0 F.c3.h154_0 F.c3.h155_0 side
      | exact part_keys 0 rfl rfl F.c3.h157_0 F.c3.h158_0 F.c3.h159_0 F.c3.h160_0 side
      | exact part_keys 1 rfl rfl F.c3.h162_0 F.c3.h163_0 F.c3.h164_0 F.c3.h165_0 side
      | exact part_keys 0 rfl rfl F.c4.h167_0 F.c4.h168_0 F.c4.h169_0 F.c4.h170_0 side
      | exact part_keys 1 rfl rfl F.c4.h172_0 F.c4.h173_0 F.c4.h174_0 F.c4.h175_0 side
      | exact part_keys 0 rfl rfl F.c4.h177_0 F.c4.h178_0 F.c4.h179_0 F.c4.h180_0 side
      | exact part_keys 1 rfl rfl F.c4.h182_0 F.c4.h183_0 F.c4.h184_0 F.c4.h185_0 side
      | exact part_keys 0 rfl rfl F.c4.h187_0 F.c4.h188_0 F.c4.h189_0 F.c4.h190_0 side
      | exact part_keys 1 rfl rfl F.c4.h192_0 F.c4.h193_0 F.c4.h194_0 F.c4.h195_0 side
      | exact part_keys 0 rfl rfl F.c4.h197_0 F.c4.h198_0 F.c4.h199_0 F.c4.h200_0 side
      | exact part_keys 1 rfl rfl F.c4.h202_0 F.c4.h203_0 F.c4.h204_0 F.c4.h205_0 side
      | exact part_keys 0 rfl rfl F.c5.h207_0 F.c5.h208_0 F.c5.h209_0 F.c5.h210_0 side
      | exact part_keys 1 rfl rfl F.c5.h212_0 F.c5.h213_0 F.c5.h214_0 F.c5.h215_0 side

end Hint

/-! ## Step zero: the remaining facts of the actual inputs -/

section Facts0

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- The smudge residues of primes `1 … 4` (`online0_part_smudge_half`). -/
theorem Step0.hTHP (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) : HTHP (S.I h) := by
  intro p i
  have := online0_part_smudge_half S.C.online S.C.online_facts
    (fun o ↦ PreprocessSwitch.smudgeValue ((S.P h).smudgeExp o))
    (fun o ↦ smudgeReadoutDigit 8 8 ((S.P h).smudgeExp o))
    (fun o ↦ congrArg _ (smudgeValue_digit _))
    (fun o ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).1)
    (fun o ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).2) rfl i p.succ
  simpa only [smudgePartColumn, Fin.val_succ, ← half0 S.C.online] using this

/-- A signed table value with monomial entries is a monomial. -/
theorem signed_monomial4 (table : Fin 4 → ErrorPoly 8)
    (htable : ∀ i, ∃ exponent : Nat, table i = Xe exponent) (exponent : Nat) :
    ∃ power : Nat, PreprocessSwitch.signedValue (by decide) table exponent = Xe power := by
  obtain ⟨power, hpower⟩ := htable ⟨exponent % 4, Nat.mod_lt _ (by decide)⟩
  refine ⟨8 * (exponent / 4) + power, ?_⟩
  simp only [PreprocessSwitch.signedValue, hpower, Xe, pow_add, pow_mul,
    Negacyclic.root_pow_n, Int.cast_pow, Int.cast_neg, Int.cast_one]

/-- The state table values are monomials. -/
theorem state_monomial (exponent : Nat) :
    ∃ power : Nat, PreprocessSwitch.stateValue exponent = Xe power :=
  signed_monomial4 _ (fun _ ↦ ⟨_, rfl⟩) exponent

/-- The fresh readout table values are monomials. -/
theorem sigmaReadout_monomial (exponent : Nat) :
    ∃ power : Nat, PreprocessSwitch.sigmaReadoutValue exponent = Xe power :=
  Online0Claim.signed_monomial _ (fun _ ↦ ⟨_, rfl⟩) exponent

/-- The fresh errors of every refresh part have coefficient bound `8`: each coefficient
cell is a sum of eight monomials. -/
theorem Step0.freshBound (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) (i : Fin 36) :
    CoeffBound ((S.I h).E0 i) 8 := by
  intro row column coefficient
  refine (coeff_natAbs_le_polyNorm _ coefficient).trans ?_
  show polyNorm (cellVal (C := 900) (by norm_num) 1
    (fun o ↦ PreprocessSwitch.sigmaReadoutValue ((S.P h).freshExp o)) _) ≤ 8
  unfold cellVal
  refine (polyNorm_sum_le _ _).trans ?_
  have hterm : ∀ j : Fin 8, ∀ o : Fin 7200, polyNorm (Xe (n := 8) (j : Nat) *
      (1 * PreprocessSwitch.sigmaReadoutValue ((S.P h).freshExp o))) ≤ 1 := by
    intro j o
    obtain ⟨power, hpower⟩ := sigmaReadout_monomial ((S.P h).freshExp o)
    rw [hpower, one_mul]
    exact Online0Claim.monomial_small (Online0Claim.monomial_mul ⟨_, rfl⟩ ⟨_, rfl⟩)
  calc ∑ j : Fin 8, _ ≤ ∑ _j : Fin 8, 1 := Finset.sum_le_sum fun j _ ↦ hterm j _
    _ = 8 := by simp

end Facts0

/-! ## Step one: the inputs of the actual run -/

section Build1

open W_Stage_online_1 StepTransition

variable {model : HashModel} {external : GeneratedClaim.ExternalInputs}
  {execution : GeneratedClaim.Execution}

/-- Two label monomials multiply to the label of the summed exponent modulo `16`. -/
theorem message_mul {a b : Int} (ha : 0 ≤ a ∧ a ≤ 15) (hb : 0 ≤ b ∧ b ≤ 15) :
    message (n := 8) 1 a * message (n := 8) 1 b = message (n := 8) 1 ((a + b) % 16) := by
  have hab : 0 ≤ (a + b) % 16 ∧ (a + b) % 16 ≤ 15 :=
    ⟨Int.emod_nonneg _ (by norm_num),
      by have := Int.emod_lt_of_pos (a + b) (by norm_num : (0 : Int) < 16); omega⟩
  rw [message_one_nat ha, message_one_nat hb, message_one_nat hab]
  have h16 : AdjoinRoot.root (negacyclicModulus 8 Int) ^ 16 = 1 := by
    rw [show 16 = 8 * 2 by rfl, pow_mul, Negacyclic.root_pow_n]
    norm_num
  show AdjoinRoot.root _ ^ _ * AdjoinRoot.root _ ^ _ = AdjoinRoot.root _ ^ _
  rw [← pow_add]
  have hsplit : a.toNat + b.toNat = 16 * ((a + b) / 16).toNat + ((a + b) % 16).toNat := by
    have := Int.emod_add_mul_ediv (a + b) 16
    have : 0 ≤ (a + b) / 16 := Int.ediv_nonneg (by omega) (by norm_num)
    omega
  rw [hsplit, pow_add, pow_mul, h16, one_pow, one_mul]

/-- The smudge target of the last decoder is the placed readout times the modular half: the
gadget times the decomposition of the half constant is that constant. -/
theorem smudge_target {params : Params} (online : generatedRoot.Ctx Backend.backend model params)
    {outputs : _} (F : generatedRoot.Facts online outputs) {G : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns Backend.backend 4096 25 G) (values : Fin 8 → ErrorPoly 8) :
    Online1Assembly.smudgeTarget online G values = fun _ _ ↦ reducePoly modulus_0 8
      (∑ member : Fin 8, Xe (n := 8) member.val * (1 * values member)) *
        ((smudgeHalf0 : Int) : ExactPoly modulus_0 8) := by
  have hq : 1 < modulus_0 := by decide
  have h182 := RefreshTopProof.GenFinal.compactRebase_self_gen (by decide) hq F.c7.h182_0
  have hD := gadget_mul_decompose hq (by decide) Backend.layout0 layout_some layout_exact hG
    F.c7.h181_0
  have h178 := centeredRebaseRuns_unique F.c7.h178_0
    (MxxExponentLut.centeredRebaseRuns_self hq (by decide) _)
  unfold Online1Assembly.smudgeTarget
  rw [Matrix.smul_mul, h182, hD]
  funext i j
  have hi : i = 0 := Subsingleton.elim _ _
  have hj : j = 0 := Subsingleton.elim _ _
  subst hi hj
  simp only [Matrix.smul_apply, smul_eq_mul, generatedRoot.w_180_0, MxxRuntime.matrixMulScalarLeft,
    h178, generatedRoot.w_177_0, generatedRoot.w_176_0, generatedRoot.w_179_0,
    MxxRuntime.matrixPolynomial, Online1Assembly.smudgeValue, one_mul, smudgeHalf0]
  simp

/-- The step-one labels: label `0` is `X^(a₂ + a₄)`, label `1` is the public unit. -/
noncomputable def labels1 (external : GeneratedClaim.ExternalInputs) : Fin 2 → ErrorPoly 8 :=
  ![Online0Claim.inputValue external 1 * Online0Claim.inputValue external 3, 1]

/-- The step-one labels not consumed by the last program. -/
noncomputable def otherLabels1 (external : GeneratedClaim.ExternalInputs) : Fin 4 → ErrorPoly 8 :=
  ![Online0Claim.inputValue external 0 * Online0Claim.inputValue external 1 *
      (Online0Claim.inputValue external 2 * Online0Claim.inputValue external 3),
    Online0Claim.inputValue external 0 * Online0Claim.inputValue external 2 *
      Online0Claim.inputValue external 3,
    Online0Claim.inputValue external 1 * 1,
    Online0Claim.inputValue external 1 * Online0Claim.inputValue external 2]

/-- The step-one labels are monomials. -/
theorem labels1_monomial (external : GeneratedClaim.ExternalInputs) (k : Fin 2) :
    ∃ e : Nat, labels1 external k = Xe e := by
  fin_cases k
  · exact Online0Claim.monomial_mul (Online0Claim.input_monomial external 1)
      (Online0Claim.input_monomial external 3)
  · exact ⟨0, rfl⟩

/-- Substitution by `1` is the identity. -/
theorem substitution_one' (hodd : Odd 1) (x : ErrorPoly 8) : substitution 1 hodd x = x := by
  have hid : substitution (n := 8) (R := Int) 1 hodd = RingHom.id _ := by
    apply AdjoinRoot.ringHom_ext
    · ext a
      simp [substitution]
    · simp
  rw [hid]
  rfl

set_option maxHeartbeats 0 in
/-- The refreshed targets of step zero are the values required by step one
(`Online1Inputs.targets`). -/
theorem Step0.targets (S : Step0 model external execution)
    (h : W_GeneratedClaim.Runs model external execution) :
    ∀ (root : ImportRoot) (member : Fin root.members) (component : Fin 2),
      reducePoly modulus_0 8 ((S.I h).cxPlain (partIndex (rootTarget root member) component)) =
        reducePoly modulus_0 8 (Online1Assembly.values (labels1 external) (otherLabels1 external)
          (PreprocessSwitch.stateValue (external.input_37.toNat + (external.input_32 0).toNat))
            root member * unitPair (rootPayload (S.I h).xσ S.tp root member) component) := by
  intro root member component
  congr 1
  cases root <;> fin_cases member <;> fin_cases component
  all_goals simp only [rootTarget, partIndex, Online1Assembly.values, rootPayload, unitPair,
    ImportRoot.sigma, labels1, otherLabels1, Fin.isValue, Matrix.cons_val_zero,
    Matrix.cons_val_one, Fin.mk_one, Fin.zero_eta, Fin.val_zero, Fin.val_one, Nat.reduceMul,
    Nat.reduceAdd, mul_one, one_mul, Online0Inputs.cxPlain]
  all_goals first
    | rfl
    | exact one_mul _
    | exact mul_one _
    | exact congrArg (· * _) (one_mul _)
    | exact congrArg (· * _) (mul_one _)
    | exact (substitution_one' _ _).symm

/-- **The step-one pipeline.** For every run, the decoded output coefficients are the residues of
any value `ν` of the step-one state LUT on its input
`X² · 1 · stateValue (z₀ + y₀) · X^(2 y₁)`. -/
theorem pipeline (L : MissingLinks) (h : W_GeneratedClaim.Runs model external execution)
    {ν : ErrorPoly 8}
    (hν : ∑ branch, reducePoly modulus_0 8
      (substitution (sigmaOf sigmas4 branch).toNat (sigmas4_odd branch)
        (Xe (n := 8) 2 * 1 *
          PreprocessSwitch.stateValue (external.input_37.toNat + (external.input_32 0).toNat) *
          Xe (n := 8) (2 * (external.input_32 1).toNat))) *
        PreprocessSwitch.state1Coefficients branch = reducePoly modulus_0 8 ν)
    (i : Fin 8) :
    (execution.«stage_17».1 0 0).coeff i =
      (((reducePoly modulus_0 8 (1 * ν)).coeff i).val : ZMod modulus_1) := by
  let S := step0 h
  let I0 := S.I h
  have hq : 1 < modulus_0 := by decide
  have rows := L.online0Rows h S.hmask S.C.online S.C.online_inputs S.C.online_facts I0
    (S.hTHP h) (noise0_of_bounds (S.bounds h))
  have hzero : ∀ index : Fin 36, 32 ≤ index.val →
      reducePoly modulus_0 8 (I0.cxPlain index) = 0 := by
    intro index hindex
    obtain ⟨v, hv⟩ := index
    simp only at hindex
    interval_cases v <;> exact map_zero _
  obtain ⟨hintError, hintErrorBound, hint⟩ := hint_of_rows h S.C.online S.C.online_inputs
    S.C.online_facts I0.xσ (fun index ↦ reducePoly modulus_0 8 (I0.cxPlain index)) I0.E0 rows
    (S.freshBound h) hzero
  obtain ⟨input, pre, online, hin, hpre, hon, Fo, out⟩ := claim_online1_outputR h I0.xσ
    Backend.layout0 layout_some layout_exact S.C.online S.C.online_facts
  have v := valid h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, v22, v23, -, -, -,
    -, -, -, v30, v31, -, -, -, -, -, -, -, v39, -⟩ := v
  have R := requirement h
  obtain ⟨sel0, hsel0⟩ := onehot (M := 8) external.input_22 external.input_39 v22 v39 R.loop_310
  obtain ⟨sel1, hsel1⟩ := onehot (M := 8) external.input_23 external.input_39 v23 v39 R.loop_312
  let key0 : Fin 8 → Fin 2 := fun member ↦
    ⟨(external.input_30 (cellAt member (sel0 member))).toNat,
      by have := v30 (cellAt member (sel0 member)); omega⟩
  let key1 : Fin 8 → Fin 2 := fun member ↦
    ⟨(external.input_31 (cellAt member (sel1 member))).toNat,
      by have := v31 (cellAt member (sel1 member)); omega⟩
  have hprod : ∀ member, ∃ e : Nat,
      labels1 external (key1 member) * labels1 external (key0 member) = Xe e :=
    fun member ↦ Online0Claim.monomial_mul (labels1_monomial external _)
      (labels1_monomial external _)
  let sexp : Fin 8 → Nat := fun member ↦ Classical.choose (hprod member)
  have hsexp : ∀ member, labels1 external (key1 member) * labels1 external (key0 member) =
      Xe (sexp member) := fun member ↦ Classical.choose_spec (hprod member)
  let smudgeValues : Fin 8 → ErrorPoly 8 := fun member ↦
    PreprocessSwitch.smudgeValue (sexp member)
  have hG : gadgetMatrixRuns Backend.backend 4096 25
      (W_Stage_online_0.generatedRoot.w_57_0 S.C.online) := S.C.online_facts.c1.h57_0
  obtain ⟨residual, residualSmall, hTH⟩ := online1_smudge_half online smudgeValues
    (fun member ↦ smudgeReadoutDigit 8 8 (sexp member))
    (fun member ↦ congrArg _ (smudgeValue_digit _))
    (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).1)
    (fun member ↦ (smudgeReadoutDigit_range (by decide) (by decide) (by decide) _).2) rfl
    (smudge_target online Fo hG smudgeValues)
  have hstate := state_monomial (external.input_37.toNat + (external.input_32 0).toNat)
  exact out {
    sp := S.tp
    state := PreprocessSwitch.stateValue (external.input_37.toNat + (external.input_32 0).toNat)
    labels := labels1 external
    otherLabels := otherLabels1 external
    gadget := W_Stage_online_0.generatedRoot.w_57_0 S.C.online
    hgadget := hG
    bound := 8
    fresh := I0.E0
    target := fun index ↦ reducePoly modulus_0 8 (I0.cxPlain index)
    parts := rows
    freshBound := S.freshBound h
    keys := by
      have hseed : generatedRoot.w_14_0 online = external.input_3 := by
        rw [generatedRoot.w_14_0, hon]
        rfl
      rw [hseed]
      exact root_keys S.C.online S.C.online_inputs S.C.online_facts
    targets := S.targets h
    preprocessSecret := by
      rw [W_Stage_preprocess_1_1.generatedRoot.w_1_0, hpre]
      funext a b
      rw [Subsingleton.elim a 0, Subsingleton.elim b 0]
      exact S.hpay
    payload := by
      intro member
      rw [import1Coefficient, import1Exponent, W_Stage_import_1.generatedRoot.w_4_0,
        W_Stage_import_1.generatedRoot.w_5_0, hin]
      fin_cases member
      · show message (external.input_14 0) (external.input_15 0) = _
        rw [step1_label0 h, labels1, Online0Claim.inputValue, Online0Claim.inputValue]
        have v0 := (valid h).1
        exact (message_mul (v0 _) (v0 _)).symm
      · show message (external.input_14 1) (external.input_15 1) = _
        rw [step1_label1 h]
        rfl
    smallMask := by
      show polyNorm (1 * _) ≤ 1
      rw [one_mul]
      exact (S.P h).sigmaSmall
    smallLabels := fun member ↦ Online0Claim.monomial_small (labels1_monomial external member)
    smallStateLabel := Online0Claim.monomial_small
      (Online0Claim.monomial_mul ⟨2, rfl⟩ (labels1_monomial external 1))
    smallStateProduct := Online0Claim.monomial_small (Online0Claim.monomial_mul
      (Online0Claim.monomial_mul ⟨2, rfl⟩ (labels1_monomial external 1)) hstate)
    select0 := sel0
    select1 := sel1
    key0 := key0
    key1 := key1
    gather0 := by
      intro member
      rw [generatedRoot.w_139_0, hon]
      have := v30 (cellAt member (sel0 member))
      show external.input_30 _ = ((external.input_30 (cellAt member (sel0 member))).toNat : Int)
      omega
    gather1 := by
      intro member
      rw [generatedRoot.w_138_0, hon]
      have := v31 (cellAt member (sel1 member))
      show external.input_31 _ = ((external.input_31 (cellAt member (sel1 member))).toNat : Int)
      omega
    selector0 := by
      rw [W_Stage_preprocess_1_1.generatedRoot.w_73_0, W_Stage_preprocess_1_1.generatedRoot.w_74_0,
        hpre]
      exact hsel0
    selector1 := by
      rw [W_Stage_preprocess_1_1.generatedRoot.w_77_0, W_Stage_preprocess_1_1.generatedRoot.w_78_0,
        hpre]
      exact hsel1
    rho := Xe (n := 8) (2 * (external.input_32 1).toNat)
    stateRhs := by
      rw [W_Stage_preprocess_1_1.generatedRoot.w_31_0, W_Stage_preprocess_1_1.generatedRoot.w_32_0,
        hpre]
      exact state_selector1 h
    smallBucket := fun key member ↦ Online0Claim.monomial_small
      (Online0Claim.monomial_mul (labels1_monomial external key) (labels1_monomial external _))
    ν := ν
    smallState := fun branch ↦ Online0Claim.small_sigma4 _ (Online0Claim.monomial_small
      (Online0Claim.monomial_mul (Online0Claim.monomial_mul
        (Online0Claim.monomial_mul ⟨2, rfl⟩ (labels1_monomial external 1)) hstate)
        ⟨_, rfl⟩)) branch
    stateWalsh := hν
    smudgeValues := smudgeValues
    smallSmudge := fun member branch ↦ Online0Claim.small_sigma8 _
      (Online0Claim.monomial_small (Online0Claim.monomial_mul (labels1_monomial external _)
        (labels1_monomial external _))) branch
    smudgeWalsh := fun member ↦ PreprocessSwitch.smudge_walsh (sexp member)
      (by rw [hsexp, one_mul])
    hintError := hintError
    hintBound := 8
    hint := hint
    hintErrorBound := hintErrorBound
    residual := residual
    residualBound := 4
    hTH := hTH
    residualSmall := residualSmall
    noise := by
      norm_num [Online1Assembly.stateBound, Online1Assembly.smudgeBound,
        Online1Assembly.cellBound, Online1Assembly.bucket0Bound, Online1Assembly.bucket1Bound] } 0 i

end Build1

/-! ## The correctness of dio2 -/

section Final

open W_Stage_online_1

/-- **The stage-level facts of `Dio2Correct`, from the proven pieces and `MissingLinks`.** The
predicates: the step-zero state is `X^(2 z₀)`, the label read by step `k` is `X^(2 y_k)`, and the
step-one state is, modulo `Q`, the state LUT value `stateValue (z₀ + y₀)`. -/
noncomputable def correctInputs (L : MissingLinks) : CorrectInputs where
  State0 _ external _ μ := μ = Xe (n := 8) (2 * external.input_37.toNat)
  State1 _ external _ ν := reducePoly W_Stage_online_0.modulus_0 8 ν =
    reducePoly W_Stage_online_0.modulus_0 8
      (PreprocessSwitch.stateValue (external.input_37.toNat + (external.input_32 0).toNat))
  Label _ external _ k α := α = Xe (n := 8) (2 * (external.input_32 k).toNat)
  initial _ := rfl
  labels _ _ := rfl
  step0 := by
    intro model external execution _ α μ ν hα hμ hwalsh
    subst hα hμ
    rw [← hwalsh]
    refine (PreprocessSwitch.state_walsh
      (message := Xe (n := 8) (2 * (external.input_32 0).toNat) *
        Xe (n := 8) (2 * external.input_37.toNat)) _ ?_)
    congr 1
    show AdjoinRoot.root _ ^ _ * AdjoinRoot.root _ ^ _ = AdjoinRoot.root _ ^ _
    rw [← pow_add]
    congr 1
    ring
  step1 := by
    intro model external execution h α μ ν hα hμ hwalsh i
    subst hα
    refine pipeline L h ?_ i
    rw [← hwalsh]
    refine Finset.sum_congr rfl fun branch _ ↦ ?_
    congr 1
    rw [reducePoly_substitution, reducePoly_substitution]
    congr 1
    simp only [map_mul, map_one]
    rw [hμ]
    ring

/-- **dio2 is correct**: the average-case correctness claim holds with failure measure `0`,
assuming only the stage-level `MissingLinks`. No external or semantic assumption remains: the
obfuscator's plaintext tables, label payloads and selector tables are checked by the run's
requirement and validity contract, the secrets by the mask and payload samplers, and the
layout by the backend. -/
theorem dio2_correct (L : MissingLinks) : GeneratedClaim.CorrectnessClaim :=
  certificate (correctInputs L)

/-- info: 'Dio2Proof.Complete.dio2_correct' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dio2_correct

end Final

section Rows

open W_Stage_online_0

/-- **The full CRT rows of step zero** (`MissingLinks.online0Rows`): the five prime terms of every
part (`Dio2Online0AssemblyP`), from the run's preimages, hint and keys stage. -/
theorem online0Rows_holds :
  ∀ {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}, W_GeneratedClaim.Runs model external execution →
    ∀ {sm : ErrorPoly 8}, execution.«stage_3» 0 0 = reducePoly W_Stage_initial.modulus_0 8 sm →
    ∀ Con : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params,
      Con.inputs = importClaimOnline0Inputs external execution →
      generatedRoot.Facts Con execution.«stage_11» →
      ∀ I : Online0Inputs Con sm, HTHP I → (∀ p i, noise0 I p i) →
      ∀ i, generatedRoot.w_1281_0 Con i = partRow0 I i
 := by
  intro model external execution h sm hmask Con hin F I hthp hnoise i
  let IP : Online0InputsP Con sm := { toOnline0Inputs := I, hTHP := hthp }
  obtain ⟨inputs, w⟩ := Con
  simp only at hin
  subst hin
  have hp := (claim_preimages h).1
  obtain ⟨eb, heb, hb⟩ := claim_hint0 h hmask
  obtain ⟨-, -, -, -, -, -, -, -, -, h8, -⟩ := h
  obtain ⟨wk, hk⟩ := h8
  let Ck : W_Stage_keys_online_0_0.generatedRoot.Ctx Backend.backend model
      W_GeneratedClaim.stage_8_params :=
    (fun inputs (_ : W_Stage_keys_online_0_0.generatedRoot.body Backend.backend model
      W_GeneratedClaim.stage_8_params inputs execution.«stage_8» wk) ↦ ⟨inputs, wk⟩) _ hk
  have Fk := W_Stage_keys_online_0_0.generatedRoot.facts Ck hk
  have hsolve : ∀ (p : Fin 5) (j : Fin 36),
      execution.«stage_1».1 * online0PrimePreimage (⟨importClaimOnline0Inputs external execution, w⟩ : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params) p j =
        online0PrimeKeyImage Ck p j := by
    intro p j
    fin_cases p
    · have equation := (hp.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.2.1 j).1
      rw [Fk.c67.hout] at equation
      exact equation
    · have equation := (hp.2.2.2.2 j).1
      rw [Fk.c67.hout] at equation
      exact equation
  have hP : ∀ (p : Fin 5) (j : Fin 36), ∃ Pd : ErrorMatrix 8 54 25,
      online0PrimePreimage (⟨importClaimOnline0Inputs external execution, w⟩ : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params) p j =
        reduceMatrix modulus_0 8 54 25 Pd ∧ CoeffBound Pd 22011706 := by
    intro p j
    fin_cases p
    · exact (hp.1 j).2
    · exact (hp.2.1 j).2
    · exact (hp.2.2.1 j).2
    · exact (hp.2.2.2.1 j).2
    · exact (hp.2.2.2.2 j).2
  have hq1 : 1 < modulus_0 := by decide
  have hn : ∀ p i, IP.Noise p i := hnoise
  obtain ⟨witness, facts⟩ := generatedRoot.h1281_0_at F i
  let c : parallel_generatedRoot_1281.Ctx Backend.backend model W_GeneratedClaim.stage_11_params i :=
    ⟨Online0Inputs.PartInputs0 (⟨importClaimOnline0Inputs external execution, w⟩ : generatedRoot.Ctx Backend.backend model W_GeneratedClaim.stage_11_params) i, witness⟩
  have t0 := IP.part_term0 F Fk (by constructor <;> rfl) hq1 Backend.layout0
    (by unfold Backend.backend; simp) rfl (hsolve 0) (hP 0) heb hb (hn 0) i c rfl facts
  have t1 := IP.part_term1 F Fk (by constructor <;> rfl) hq1 Backend.layout0
    (by unfold Backend.backend; simp) rfl (hsolve 1) (hP 1) heb hb (hn 1) i c rfl facts
  have t2 := IP.part_term2 F Fk (by constructor <;> rfl) hq1 Backend.layout0
    (by unfold Backend.backend; simp) rfl (hsolve 2) (hP 2) heb hb (hn 2) i c rfl facts
  have t3 := IP.part_term3 F Fk (by constructor <;> rfl) hq1 Backend.layout0
    (by unfold Backend.backend; simp) rfl (hsolve 3) (hP 3) heb hb (hn 3) i c rfl facts
  have t4 := IP.part_term4 F Fk (by constructor <;> rfl) hq1 Backend.layout0
    (by unfold Backend.backend; simp) rfl (hsolve 4) (hP 4) heb hb (hn 4) i c rfl facts
  exact facts.c14.hout.trans (online0_part_row c t0 t1 t2 t3 t4)

end Rows

/-- **The Diamond iO 2 correctness claim.** The exported average-case claim holds: every run decodes
the ideal output, with no assumption beyond the generated claim itself. -/
theorem dio2_correctness_claim : GeneratedClaim.CorrectnessClaim :=
  dio2_correct ⟨online0Rows_holds⟩

/-- info: 'Dio2Proof.Complete.dio2_correctness_claim' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dio2_correctness_claim

end Complete

end Dio2Proof

