import BridgeRuns
import W_Claim
import Dio2Decode
import Dio2Ideal
import Dio2PreprocessSwitch

/-!
# The correctness of dio2 and the average-case certificate

The top of the proof. Every worst-case run (`W_GeneratedClaim.Runs`) decodes the ideal output
(`correct`), and every average-case run is a worst-case run (`GeneratedClaim.runs_bridge`), so the
set of tapes on which some run fails is empty and the average-case claim holds with measure `0`
(`certificate`), as for refresh_top (`proof/RefreshTopCertificate.lean`).

Both theorems take the stage-level facts of the run as an explicit structure `CorrectInputs`, one
field per fact, each discharged by the module named in its docstring. The structure fixes two
encoding predicates (`State0`, `State1`: the state row an online step reads encodes a plaintext
message) and one label predicate (`Label`: the payload label of an online step), chosen by the
modules that discharge the fields; the conclusion only depends on the decoded output, so any choice
is sound.

The value chain proven here, on the evaluator's chunks `x = (0, 1)`, the obfuscator's bits
`y = input_32` and the initial state `z₀ = input_37`:

* step 0's LUT input is `α₀ μ₀ = X^(2 y₀) X^(2 z₀) = X^(2 (y₀ + z₀))`; the state LUT's Walsh
  identity (`PreprocessSwitch.state_walsh`) gives its output `X^(2 z₁)`, `z₁ = [1, 0, 1, 0][z₀ + y₀]`
  (`stateValue_lt`), which is step 1's input state;
* step 1's LUT input is `X² α₁ μ₁ = X^(2 (1 + y₁ + z₁))`; the step-1 state LUT
  (`PreprocessSwitch.state1_walsh`) gives `ν = X^(2 z₂)`, `z₂ = [0, 1, 1, 0][z₁ + 1 + y₁]`
  (`state1Value_lt`);
* the output decoder holds the residues of `Δ' ν` (`Δ' = 1`, the switching tables carry no
  noise), whose coefficients are `0`/`1` (`coeff_val_Xe`), so the decode stage outputs `z₂`
  (`decode_monomial`), the ideal output (`ideal_value`).
-/

set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
  Dio2Proof.PreprocessSwitch

/-- The stage-level facts of a worst-case run, from which `correct` derives the claim. The
predicates are chosen by the modules that discharge the fields. -/
structure CorrectInputs where
  /-- The state row read by step 0's online stage (`W_Stage_online_0`) encodes the message `μ`
  (the step input invariant of `Dio2Online0State`/`Dio2Online0Inputs`). -/
  State0 : HashModel → GeneratedClaim.ExternalInputs → GeneratedClaim.Execution → ErrorPoly 8 → Prop
  /-- The state row read by step 1's online stage (`W_Stage_online_1`, after `W_Stage_import_1`)
  encodes the message `μ` (the step input invariant of `Dio2Online1State`). -/
  State1 : HashModel → GeneratedClaim.ExternalInputs → GeneratedClaim.Execution → ErrorPoly 8 → Prop
  /-- The payload label of the obfuscator's bit read by online step `k` is `α` (`hlab`/`hpay` of
  `online0_lut_input`/`online1_lut_input`). -/
  Label : HashModel → GeneratedClaim.ExternalInputs → GeneratedClaim.Execution → Fin 2 →
    ErrorPoly 8 → Prop
  /-- **The initial state** encodes `X^(2 z₀)`, `z₀ = input_37`. Discharged by
  `Dio2InitialTargets`/`Dio2Initial` (the initial stage's state row) through `Dio2Import`
  (`W_Stage_import_0`). -/
  initial : ∀ {model external execution}, W_GeneratedClaim.Runs model external execution →
    State0 model external execution (Xe (n := 8) (2 * external.input_37.toNat))
  /-- **The labels**: online step `k` reads the label `X^(2 y_k)` of the obfuscator's bit
  `y_k = input_32 k`. Discharged by `Dio2InitialTargets`/`Dio2Import` (the payload and the label
  rows of `W_Stage_initial` routed through `W_Stage_import_0`/`W_Stage_import_1`). -/
  labels : ∀ {model external execution}, W_GeneratedClaim.Runs model external execution →
    ∀ k : Fin 2, Label model external execution k (Xe (n := 8) (2 * (external.input_32 k).toNat))
  /-- **Step 0** (`x₀ = 0`): from the input state `μ` and the label `α`, the state LUT whose
  Walsh sum on the LUT input `α μ` is `ν` (the `hwalsh` of `Dio2Online0State`, with the
  preprocessed coefficients `stateCoefficients` and `Δ' = 1` from
  `PreprocessSwitch.preprocess0_state_switching`) produces a state row that, after
  `W_Stage_import_1`, encodes `ν` for step 1. Discharged by `Dio2Online0Assembly0`/
  `Dio2Online0AssemblyP`/`Dio2Online0Inputs` (step 0's refresh of the state row) and `Dio2Import`
  (`W_Stage_import_1`). -/
  step0 : ∀ {model external execution}, W_GeneratedClaim.Runs model external execution →
    ∀ {α μ ν : ErrorPoly 8}, Label model external execution 0 α → State0 model external execution μ →
    (∑ branch, reducePoly W_Stage_preprocess_0_0.modulus_0 8
        (substitution (sigmaOf stateSigmas branch).toNat (stateOdd branch) (α * μ)) *
        stateCoefficients branch = reducePoly W_Stage_preprocess_0_0.modulus_0 8 ν) →
    State1 model external execution ν
  /-- **Step 1 and its output decoder** (`x₁ = 1`, `XB = X²`): from the input state `μ` and the
  label `α`, the state LUT whose Walsh sum on the LUT input `X² α μ` is `ν` (the `hwalsh` of
  `online1_state_lut`, with `state1Coefficients` and `Δ' = 1` from
  `PreprocessSwitch.preprocess1_state_switching`) gives the output decoder (`online1_output`)
  whose every coefficient is the residue of `Δ' ν` modulo the first prime. Discharged by
  `Dio2Online1Assembly` (`online1_lut_input`, `online1_state_lut`, `online1_output`, the smudge
  of `Dio2SmudgeHalf`). -/
  step1 : ∀ {model external execution}, W_GeneratedClaim.Runs model external execution →
    ∀ {α μ ν : ErrorPoly 8}, Label model external execution 1 α → State1 model external execution μ →
    (∑ branch, reducePoly W_Stage_preprocess_1_1.modulus_0 8
        (substitution (sigmaOf stateSigmas branch).toNat (stateOdd branch) (Xe (n := 8) 2 * α * μ)) *
        state1Coefficients branch = reducePoly W_Stage_preprocess_1_1.modulus_0 8 ν) →
    ∀ i : Fin 8, (execution.«stage_17».1 0 0).coeff i =
      (((reducePoly W_Stage_online_1.modulus_0 8 ((1 : ErrorPoly 8) * ν)).coeff i).val :
        ZMod W_Stage_online_1.modulus_1)

theorem Xe_mul_Xe' (a b : Nat) : Xe (n := 8) a * Xe (n := 8) b = Xe (n := 8) (a + b) :=
  (pow_add _ a b).symm

theorem Xe_congr' {a b : Nat} (h : a = b) : Xe (n := 8) a = Xe (n := 8) b := h ▸ rfl

/-- The step-0 state table on an in-range input: no negacyclic sign. -/
theorem stateValue_lt {e : Nat} (he : e < 4) :
    stateValue e = Xe (n := 8) (2 * (entry table0 (e : Int)).toNat) := by
  unfold stateValue signedValue
  rw [Nat.div_eq_of_lt he, pow_zero, Int.cast_one, one_mul]
  apply Xe_congr'
  interval_cases e <;> rfl

/-- The step-1 state table on an in-range input: no negacyclic sign. -/
theorem state1Value_lt {e : Nat} (he : e < 4) :
    state1Value e = Xe (n := 8) (2 * (entry table1 (e : Int)).toNat) := by
  unfold state1Value signedValue
  rw [Nat.div_eq_of_lt he, pow_zero, Int.cast_one, one_mul]
  apply Xe_congr'
  interval_cases e <;> rfl

/-- The output residues of the monomial `X^(2 t)` are its `0`/`1` coefficients. -/
theorem coeff_val_Xe {t : Nat} (ht : 2 * t < 8) (i : Fin 8) :
    ((((reducePoly W_Stage_online_1.modulus_0 8 ((1 : ErrorPoly 8) * Xe (n := 8) (2 * t))).coeff
        i).val : ZMod W_Stage_online_1.modulus_1).val : Int) = if i.val = 2 * t then 1 else 0 := by
  haveI : Fact (1 < W_Stage_online_1.modulus_0) := ⟨by norm_num⟩
  haveI : Fact (1 < W_Stage_online_1.modulus_1) := ⟨by norm_num⟩
  have hc : (Xe (n := 8) (2 * t)).coeff i = if (⟨2 * t, ht⟩ : Fin 8) = i then 1 else 0 :=
    Negacyclic.coeff_root_pow (R := Int) (n := 8) (by norm_num) ⟨2 * t, ht⟩ i
  rw [one_mul, reducePoly_coeff (by norm_num) (by norm_num), hc]
  by_cases hi : i.val = 2 * t
  · have : (⟨2 * t, ht⟩ : Fin 8) = i := Fin.ext hi.symm
    simp [this, hi, ZMod.val_one]
  · have : ¬ (⟨2 * t, ht⟩ : Fin 8) = i := fun h ↦ hi (by rw [← h])
    simp [this, hi]

/-- A value in `[0, 1]` is the cast of a natural number at most `1`. -/
theorem bit_nat {v : Int} (h0 : 0 ≤ v) (h1 : v ≤ 1) : v = (v.toNat : Int) ∧ v.toNat ≤ 1 :=
  ⟨(Int.toNat_of_nonneg h0).symm, by omega⟩

/-- **Every worst-case run decodes the ideal output**, given the stage-level facts. -/
theorem correct (I : CorrectInputs) {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    execution.«stage_18» = execution.«ideal» := by
  have hrun := h
  obtain ⟨hv, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, r_decode, -, r_ideal, -⟩ := h
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -,
    -, h32, -, -, -, -, h37, -, -, -⟩ := hv
  -- The bits as natural numbers.
  obtain ⟨hz, ha⟩ := bit_nat (h37.1) (h37.2)
  obtain ⟨hy0, hb⟩ := bit_nat (h32 0).1 (h32 0).2
  obtain ⟨hy1, hc⟩ := bit_nat (h32 1).1 (h32 1).2
  set a := external.input_37.toNat
  set b := (external.input_32 0).toNat
  set c := (external.input_32 1).toNat
  -- Step 0: `X^(2 b) X^(2 a) = X^(2 (b + a))`, through the state table.
  have hμ1 : I.State1 model external execution (stateValue (b + a)) := by
    refine I.step0 hrun (I.labels hrun 0) (I.initial hrun) ?_
    refine state_walsh (b + a) ?_
    rw [Xe_mul_Xe', mul_add]
  set t0 := (entry table0 ((b + a : Nat) : Int)).toNat
  have ht0 : t0 ≤ 1 := by
    simp only [t0]
    interval_cases a <;> interval_cases b <;> decide
  -- Step 1: `X² X^(2 c) X^(2 t0) = X^(2 (1 + c + t0))`, through the step-1 state table.
  have hout := I.step1 hrun (I.labels hrun 1) hμ1 (ν := state1Value (1 + c + t0)) (by
    refine state1_walsh (1 + c + t0) ?_
    rw [stateValue_lt (by omega), Xe_mul_Xe', Xe_mul_Xe']
    congr 2
    ring)
  set t1 := (entry table1 ((1 + c + t0 : Nat) : Int)).toNat
  have ht1 : t1 ≤ 1 := by
    simp only [t1]
    interval_cases c <;> interval_cases t0 <;> decide
  rw [state1Value_lt (by omega)] at hout
  -- The decode stage reads `X^(2 t1)` and outputs `t1`.
  have hdec := decode_monomial W_GeneratedClaim.stage_18_params _ t1 (by omega)
    (fun i ↦ by rw [hout i]; exact coeff_val_Xe (by omega) i) _ r_decode
  rw [hdec, ideal_value W_GeneratedClaim.ideal_params _ _ _ r_ideal, hz, hy0, hy1]
  simp only [t1, t0]
  interval_cases a <;> interval_cases b <;> interval_cases c <;> decide

/-- **The average-case claim.** Every average-case run is a worst-case run
(`GeneratedClaim.runs_bridge`), and every worst-case run decodes the ideal output (`correct`): the
set of tapes on which some run fails is empty, so its measure is `0`. -/
theorem certificate (I : CorrectInputs) : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_18» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (correct I (GeneratedClaim.runs_bridge hruns))
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'Dio2Proof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end Dio2Proof
