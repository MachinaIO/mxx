import W_Stage_import_0_Facts
import W_Stage_import_1_Facts
import Gen_W_Stage_import_0_Scopes
import Gen_W_Stage_import_1_Scopes
import W_Claim
import Dio2InitialTargets

/-! Generated import identities and relation transport.
Copy outputs are aliases in v2, not calls to retained unused scopes.
The import-1 source is the refreshed online-0 output, not S_0.
Plaintext agreement with external exponents remains a separate premise. -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

attribute [local irreducible] Mxx.Primitives.negacyclicModulus

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

open GeneratedClaim W_GeneratedClaim

/-- All imported roots and helpers, with label publics evaluated symbolically. -/
noncomputable def import0_expected (inputs : ((W_Stage_import_0.generatedRoot.InputType))) : ((W_Stage_import_0.generatedRoot.OutputType)) :=
  let w_0_0 := inputs.1
  let w_1_0 := inputs.2.1
  let w_2_0 := inputs.2.2.1
  let w_3_0 := inputs.2.2.2.1
  let w_4_0 := inputs.2.2.2.2.1
  let w_5_0 := inputs.2.2.2.2.2.1
  let w_7_0 := inputs.2.2.2.2.2.2.1
  let w_8_0 := inputs.2.2.2.2.2.2.2.1
  let w_9_0 := inputs.2.2.2.2.2.2.2.2.1
  let w_10_0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_12_0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_13_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_18_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_21_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_27_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_28_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_31_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_33_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_34_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_36_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_37_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_40_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_6_0 := fun member : Fin 2 ↦ w_2_0 member
  let w_6_1 := fun member : Fin 2 ↦ w_3_0 member
  let w_6_2 := fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger (w_4_0 member)) (w_5_0 member)
  let w_11_0 := fun member : Fin 2 ↦ w_2_0 member
  let w_11_1 := fun member : Fin 2 ↦ w_3_0 member
  let w_11_2 := fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger (w_9_0 member)) (w_10_0 member)
  let w_14_0 := fun member : Fin 2 ↦ w_2_0 member
  let w_14_1 := fun member : Fin 2 ↦ w_3_0 member
  let w_14_2 := fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger (w_12_0 member)) (w_13_0 member)
  let w_17_0 := w_15_0
  let w_17_1 := w_16_0
  let w_20_0 := w_18_0
  let w_20_1 := w_19_0
  let w_23_0 := w_21_0
  let w_23_1 := w_22_0
  let w_26_0 := w_24_0
  let w_26_1 := w_25_0
  let w_29_0 := w_27_0
  let w_29_1 := w_28_0
  let w_32_0 := w_30_0
  let w_32_1 := w_31_0
  let w_35_0 := w_33_0
  let w_35_1 := w_34_0
  let w_38_0 := w_36_0
  let w_38_1 := w_37_0
  let w_41_0 := w_39_0
  let w_41_1 := w_40_0
  let w_42_0 := w_15_0
  let w_42_1 := w_16_0
  let w_43_0 := w_18_0
  let w_43_1 := w_19_0
  let w_44_0 := w_21_0
  let w_44_1 := w_22_0
  let w_45_0 := w_24_0
  let w_45_1 := w_25_0
  let w_46_0 := w_27_0
  let w_46_1 := w_28_0
  let w_47_0 := w_30_0
  let w_47_1 := w_31_0
  let w_48_0 := w_33_0
  let w_48_1 := w_34_0
  let w_49_0 := w_36_0
  let w_49_1 := w_37_0
  let w_50_0 := w_39_0
  let w_50_1 := w_40_0
  let w_51_0 := w_15_0
  let w_51_1 := w_16_0
  let w_52_0 := w_18_0
  let w_52_1 := w_19_0
  let w_53_0 := w_21_0
  let w_53_1 := w_22_0
  let w_54_0 := w_24_0
  let w_54_1 := w_25_0
  let w_55_0 := w_27_0
  let w_55_1 := w_28_0
  let w_56_0 := w_30_0
  let w_56_1 := w_31_0
  let w_57_0 := w_33_0
  let w_57_1 := w_34_0
  let w_58_0 := w_36_0
  let w_58_1 := w_37_0
  let w_59_0 := w_39_0
  let w_59_1 := w_40_0
  let w_60_0 := w_15_0
  let w_60_1 := w_16_0
  let w_61_0 := w_18_0
  let w_61_1 := w_19_0
  let w_62_0 := w_21_0
  let w_62_1 := w_22_0
  let w_63_0 := w_24_0
  let w_63_1 := w_25_0
  let w_64_0 := w_27_0
  let w_64_1 := w_28_0
  let w_65_0 := w_30_0
  let w_65_1 := w_31_0
  let w_66_0 := w_33_0
  let w_66_1 := w_34_0
  let w_67_0 := w_36_0
  let w_67_1 := w_37_0
  let w_68_0 := w_39_0
  let w_68_1 := w_40_0
  let w_69_0 := w_15_0
  let w_69_1 := w_16_0
  let w_70_0 := w_18_0
  let w_70_1 := w_19_0
  let w_71_0 := w_21_0
  let w_71_1 := w_22_0
  let w_72_0 := w_24_0
  let w_72_1 := w_25_0
  let w_73_0 := w_27_0
  let w_73_1 := w_28_0
  let w_74_0 := w_30_0
  let w_74_1 := w_31_0
  let w_75_0 := w_33_0
  let w_75_1 := w_34_0
  let w_76_0 := w_36_0
  let w_76_1 := w_37_0
  let w_77_0 := w_39_0
  let w_77_1 := w_40_0
  (w_0_0, (w_1_0, (w_6_0, (w_6_1, (w_6_2, (w_7_0, (w_8_0, (w_0_0, (w_1_0, (w_11_0, (w_11_1, (w_11_2, (w_0_0, (w_1_0, (w_14_0, (w_14_1, (w_14_2, (w_17_0, (w_17_1, (w_20_0, (w_20_1, (w_23_0, (w_23_1, (w_26_0, (w_26_1, (w_29_0, (w_29_1, (w_32_0, (w_32_1, (w_35_0, (w_35_1, (w_38_0, (w_38_1, (w_41_0, (w_41_1, (w_42_0, (w_42_1, (w_43_0, (w_43_1, (w_44_0, (w_44_1, (w_45_0, (w_45_1, (w_46_0, (w_46_1, (w_47_0, (w_47_1, (w_48_0, (w_48_1, (w_49_0, (w_49_1, (w_50_0, (w_50_1, (w_51_0, (w_51_1, (w_52_0, (w_52_1, (w_53_0, (w_53_1, (w_54_0, (w_54_1, (w_55_0, (w_55_1, (w_56_0, (w_56_1, (w_57_0, (w_57_1, (w_58_0, (w_58_1, (w_59_0, (w_59_1, (w_60_0, (w_60_1, (w_61_0, (w_61_1, (w_62_0, (w_62_1, (w_63_0, (w_63_1, (w_64_0, (w_64_1, (w_65_0, (w_65_1, (w_66_0, (w_66_1, (w_67_0, (w_67_1, (w_68_0, (w_68_1, (w_69_0, (w_69_1, (w_70_0, (w_70_1, (w_71_0, (w_71_1, (w_72_0, (w_72_1, (w_73_0, (w_73_1, (w_74_0, (w_74_1, (w_75_0, (w_75_1, (w_76_0, (w_76_1, (w_77_0, (w_77_1, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

/-- Exact import semantics for every output at once, without expanding family members. -/
theorem import0_spec {params : W_Stage_import_0.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_0.generatedRoot params inputs outputs) : outputs = import0_expected inputs := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_0.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_0.generatedRoot.facts ctx hbody
  rw [F.c1.hout]
  have eq_w_6_0 : W_Stage_import_0.generatedRoot.w_6_0 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_2_0 ctx) member := by
    funext member
    exact congrArg Prod.fst (W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  have eq_w_6_1 : W_Stage_import_0.generatedRoot.w_6_1 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_3_0 ctx) member := by
    funext member
    exact congrArg (fun result ↦ result.2.1) (W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  have eq_w_6_2 : W_Stage_import_0.generatedRoot.w_6_2 ctx =
      fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger ((W_Stage_import_0.generatedRoot.w_4_0 ctx) member)) ((W_Stage_import_0.generatedRoot.w_5_0 ctx) member) := by
    funext member
    exact congrArg (fun result ↦ result.2.2.1) (W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  have eq_w_11_0 : W_Stage_import_0.generatedRoot.w_11_0 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_2_0 ctx) member := by
    funext member
    exact congrArg Prod.fst (W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member))
  have eq_w_11_1 : W_Stage_import_0.generatedRoot.w_11_1 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_3_0 ctx) member := by
    funext member
    exact congrArg (fun result ↦ result.2.1) (W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member))
  have eq_w_11_2 : W_Stage_import_0.generatedRoot.w_11_2 ctx =
      fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger ((W_Stage_import_0.generatedRoot.w_9_0 ctx) member)) ((W_Stage_import_0.generatedRoot.w_10_0 ctx) member) := by
    funext member
    exact congrArg (fun result ↦ result.2.2.1) (W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member))
  have eq_w_14_0 : W_Stage_import_0.generatedRoot.w_14_0 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_2_0 ctx) member := by
    funext member
    exact congrArg Prod.fst (W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member))
  have eq_w_14_1 : W_Stage_import_0.generatedRoot.w_14_1 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_0.generatedRoot.w_3_0 ctx) member := by
    funext member
    exact congrArg (fun result ↦ result.2.1) (W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member))
  have eq_w_14_2 : W_Stage_import_0.generatedRoot.w_14_2 ctx =
      fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger ((W_Stage_import_0.generatedRoot.w_12_0 ctx) member)) ((W_Stage_import_0.generatedRoot.w_13_0 ctx) member) := by
    funext member
    exact congrArg (fun result ↦ result.2.2.1) (W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member))
  unfold W_Stage_import_0.generatedRoot.outs import0_expected
  simp only [eq_w_6_0, eq_w_6_1, eq_w_6_2, eq_w_11_0, eq_w_11_1, eq_w_11_2, eq_w_14_0, eq_w_14_1, eq_w_14_2] <;> rfl

/-- Any root-family/online-input projection preserves an established encoding relation. -/
theorem import0_online_from_rows {params : W_Stage_import_0.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_0.generatedRoot params inputs outputs)
    (rows : ((W_Stage_import_0.generatedRoot.OutputType)) → Fin 2 → ExactMatrix W_Stage_import_0.modulus_0 8 1 25)
    {sm : ErrorPoly 8} {key : Fin 2 → Fin 2 → ExactMatrix W_Stage_import_0.modulus_0 8 1 25}
    {target : Fin 2 → ExactMatrix W_Stage_import_0.modulus_0 8 1 25} {bound : Nat}
    (hrel : Rel sm key target (rows (import0_expected inputs)) bound) :
    Rel sm key target (rows outputs) bound := by
  rw [import0_spec h]
  exact hrel

theorem import0_payload_parallel_generatedRoot_6
    {params : W_Stage_import_0.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_0.generatedRoot params inputs outputs) (member : Fin 2) :
    outputs.2.2.2.2.1 member = fun _ _ ↦ reducePoly W_Stage_import_0.modulus_0 8
      (message ((inputs.2.2.2.2.1) member) ((inputs.2.2.2.2.2.1) member)) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_0.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_0.generatedRoot.facts ctx hbody
  rw [F.c1.hout]
  have hscope := W_Stage_import_0.parallel_generatedRoot_6.sem (F.c0.h6_2 member)
  exact (congrArg (fun result ↦ result.2.2.1) hscope).trans
    (multiplyMonomial_liftInteger _ _)

theorem import0_payload_parallel_generatedRoot_11
    {params : W_Stage_import_0.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_0.generatedRoot params inputs outputs) (member : Fin 2) :
    outputs.2.2.2.2.2.2.2.2.2.2.2.1 member = fun _ _ ↦ reducePoly W_Stage_import_0.modulus_0 8
      (message ((inputs.2.2.2.2.2.2.2.2.1) member) ((inputs.2.2.2.2.2.2.2.2.2.1) member)) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_0.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_0.generatedRoot.facts ctx hbody
  rw [F.c1.hout]
  have hscope := W_Stage_import_0.parallel_generatedRoot_11.sem (F.c0.h11_2 member)
  exact (congrArg (fun result ↦ result.2.2.1) hscope).trans
    (multiplyMonomial_liftInteger _ _)

theorem import0_payload_parallel_generatedRoot_14
    {params : W_Stage_import_0.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_0.generatedRoot params inputs outputs) (member : Fin 2) :
    outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 member = fun _ _ ↦ reducePoly W_Stage_import_0.modulus_0 8
      (message ((inputs.2.2.2.2.2.2.2.2.2.2.1) member) ((inputs.2.2.2.2.2.2.2.2.2.2.2.1) member)) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_0.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_0.generatedRoot.facts ctx hbody
  rw [F.c1.hout]
  have hscope := W_Stage_import_0.parallel_generatedRoot_14.sem (F.c0.h14_2 member)
  exact (congrArg (fun result ↦ result.2.2.1) hscope).trans
    (multiplyMonomial_liftInteger _ _)

/-- The actual claim import run, with source connections resolved by `W_Claim`. -/
theorem import0_claim_runs {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs hashModel external execution) :
    W_Stage_import_0.generatedRoot stage_10_params ((execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_9, external.input_10, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_9, external.input_10, external.input_9, external.input_10, execution.«stage_4».2.1, execution.«stage_4».2.2.1, execution.«stage_4».2.2.2.1, execution.«stage_4».2.2.2.2.1, execution.«stage_4».2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ())) execution.«stage_10» :=
  h.2.2.2.2.2.2.2.2.2.2.2.1

/-- The matching online run consumes the import outputs wired by `W_Claim`. -/
theorem import0_online_claim_runs {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs hashModel external execution) :
    W_Stage_online_0.generatedRoot Backend.backend hashModel stage_11_params ((execution.«stage_4».1, execution.«stage_9».1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_2, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_25, execution.«stage_10».2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.1, external.input_24, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».1, execution.«stage_6».2.1, execution.«stage_6».2.2.1, execution.«stage_6».2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.1, execution.«stage_10».2.2.2.1, execution.«stage_10».2.2.2.2.2.1, execution.«stage_10».2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_13, external.input_3, execution.«stage_2».2.1, execution.«stage_6».2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.1, external.input_27, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_26, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.1, execution.«stage_6».2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.1, external.input_23, external.input_22, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_10».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_9».2.1, execution.«stage_9».2.2.1, execution.«stage_9».2.2.2.1, execution.«stage_9».2.2.2.2.1, ())) execution.«stage_11» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.1

/-- All imported roots and helpers, with label publics evaluated symbolically. -/
noncomputable def import1_expected (inputs : ((W_Stage_import_1.generatedRoot.InputType))) : ((W_Stage_import_1.generatedRoot.OutputType)) :=
  let w_0_0 := inputs.1
  let w_1_0 := inputs.2.1
  let w_2_0 := inputs.2.2.1
  let w_3_0 := inputs.2.2.2.1
  let w_4_0 := inputs.2.2.2.2.1
  let w_5_0 := inputs.2.2.2.2.2.1
  let w_7_0 := inputs.2.2.2.2.2.2.1
  let w_8_0 := inputs.2.2.2.2.2.2.2.1
  let w_9_0 := inputs.2.2.2.2.2.2.2.2.1
  let w_10_0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_12_0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_13_0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_6_0 := fun member : Fin 2 ↦ w_2_0 member
  let w_6_1 := fun member : Fin 2 ↦ w_3_0 member
  let w_6_2 := fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger (w_4_0 member)) (w_5_0 member)
  let w_11_0 := w_9_0
  let w_11_1 := w_10_0
  let w_14_0 := w_12_0
  let w_14_1 := w_13_0
  let w_15_0 := w_9_0
  let w_15_1 := w_10_0
  let w_16_0 := w_12_0
  let w_16_1 := w_13_0
  let w_17_0 := w_9_0
  let w_17_1 := w_10_0
  let w_18_0 := w_12_0
  let w_18_1 := w_13_0
  let w_19_0 := w_9_0
  let w_19_1 := w_10_0
  let w_20_0 := w_12_0
  let w_20_1 := w_13_0
  let w_21_0 := w_9_0
  let w_21_1 := w_10_0
  let w_22_0 := w_12_0
  let w_22_1 := w_13_0
  (w_0_0, (w_1_0, (w_6_0, (w_6_1, (w_6_2, (w_7_0, (w_8_0, (w_11_0, (w_11_1, (w_14_0, (w_14_1, (w_15_0, (w_15_1, (w_16_0, (w_16_1, (w_17_0, (w_17_1, (w_18_0, (w_18_1, (w_19_0, (w_19_1, (w_20_0, (w_20_1, (w_21_0, (w_21_1, (w_22_0, (w_22_1, ())))))))))))))))))))))))))))

/-- Exact import semantics for every output at once, without expanding family members. -/
theorem import1_spec {params : W_Stage_import_1.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_1.generatedRoot params inputs outputs) : outputs = import1_expected inputs := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_1.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_1.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  have eq_w_6_0 : W_Stage_import_1.generatedRoot.w_6_0 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_1.generatedRoot.w_2_0 ctx) member := by
    funext member
    exact congrArg Prod.fst (W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  have eq_w_6_1 : W_Stage_import_1.generatedRoot.w_6_1 ctx =
      fun member : Fin 2 ↦ (W_Stage_import_1.generatedRoot.w_3_0 ctx) member := by
    funext member
    exact congrArg (fun result ↦ result.2.1) (W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  have eq_w_6_2 : W_Stage_import_1.generatedRoot.w_6_2 ctx =
      fun member : Fin 2 ↦ MxxRuntime.multiplyMonomial (MxxRuntime.liftInteger ((W_Stage_import_1.generatedRoot.w_4_0 ctx) member)) ((W_Stage_import_1.generatedRoot.w_5_0 ctx) member) := by
    funext member
    exact congrArg (fun result ↦ result.2.2.1) (W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member))
  unfold W_Stage_import_1.generatedRoot.outs import1_expected
  simp only [eq_w_6_0, eq_w_6_1, eq_w_6_2] <;> rfl

/-- Any root-family/online-input projection preserves an established encoding relation. -/
theorem import1_online_from_rows {params : W_Stage_import_1.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_1.generatedRoot params inputs outputs)
    (rows : ((W_Stage_import_1.generatedRoot.OutputType)) → Fin 2 → ExactMatrix W_Stage_import_1.modulus_0 8 1 25)
    {sm : ErrorPoly 8} {key : Fin 2 → Fin 2 → ExactMatrix W_Stage_import_1.modulus_0 8 1 25}
    {target : Fin 2 → ExactMatrix W_Stage_import_1.modulus_0 8 1 25} {bound : Nat}
    (hrel : Rel sm key target (rows (import1_expected inputs)) bound) :
    Rel sm key target (rows outputs) bound := by
  rw [import1_spec h]
  exact hrel

theorem import1_payload_parallel_generatedRoot_6
    {params : W_Stage_import_1.Params} {inputs : _} {outputs : _}
    (h : W_Stage_import_1.generatedRoot params inputs outputs) (member : Fin 2) :
    outputs.2.2.2.2.1 member = fun _ _ ↦ reducePoly W_Stage_import_1.modulus_0 8
      (message ((inputs.2.2.2.2.1) member) ((inputs.2.2.2.2.2.1) member)) := by
  obtain ⟨witness, hbody⟩ := h
  let ctx : W_Stage_import_1.generatedRoot.Ctx params := ⟨inputs, witness⟩
  have F := W_Stage_import_1.generatedRoot.facts ctx hbody
  rw [F.c0.hout]
  have hscope := W_Stage_import_1.parallel_generatedRoot_6.sem (F.c0.h6_2 member)
  exact (congrArg (fun result ↦ result.2.2.1) hscope).trans
    (multiplyMonomial_liftInteger _ _)

/-- The actual claim import run, with source connections resolved by `W_Claim`. -/
theorem import1_claim_runs {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs hashModel external execution) :
    W_Stage_import_1.generatedRoot stage_16_params ((execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_11, external.input_12, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_11».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ())) execution.«stage_16» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

/-- The matching online run consumes the import outputs wired by `W_Claim`. -/
theorem import1_online_claim_runs {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs hashModel external execution) :
    W_Stage_online_1.generatedRoot Backend.backend hashModel stage_17_params ((execution.«stage_11».1, execution.«stage_15», execution.«stage_12».1, execution.«stage_12».2.1, execution.«stage_12».2.2.1, execution.«stage_12».2.2.2.1, execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_3, execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_16».2.2.1, execution.«stage_16».2.2.2.1, execution.«stage_16».2.2.2.2.2.1, execution.«stage_16».2.2.2.2.1, execution.«stage_16».2.2.2.2.2.2.1, external.input_1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.1, execution.«stage_12».2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.2.1, execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_16».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, external.input_29, external.input_28, execution.«stage_16».1, execution.«stage_16».2.1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_12».2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, ())) execution.«stage_17» :=
  h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

end Dio2Proof
