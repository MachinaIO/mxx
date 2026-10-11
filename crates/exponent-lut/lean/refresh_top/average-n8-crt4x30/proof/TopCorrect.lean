import RequirementRoot
import TopSecrets
import TopDecodeRun
import TopGadgets
import TopFinal
import Part_0_0_Assembly
import Part_0_1_Assembly

/-! Every run of the worst-case relation decodes the ideal output (written by
`generator/gen_correct.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    execution.«stage_8» = execution.«ideal» := by
  obtain ⟨hv, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12⟩ := h
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62⟩ := hv
  obtain ⟨r, hrE, hr⟩ := payload_ternary r1
  obtain ⟨sm, hsmE, hsm⟩ := mask_ternary r2
  obtain ⟨rq0, rq1, rq2, rq3, rq4, rq5, rq6, rq7, rq8, rq9, rq10, rq11, rq12, rq13, rq14, rq15, rq16, rq17, rq18, rq19, rq20, rq21, rq22, rq23, rq24, rq25, rq26, rq27, rq28, rq29, rq30, rq31, rq32⟩ := requirementFacts r10 r12
  obtain ⟨G, hG⟩ := gadgetQ1
  have P0_0 := Part_0_0.partAssembly hr hsm hrE hsmE (by have := v23; omega) (by have := v24; omega) (by have := v58; omega) (by have := v59; omega) (fun k ↦ (rq0 k).2.1) (fun k ↦ by rw [← (rq0 k).1]; exact (rq0 k).2.2) (fun i ↦ by have := v26 i; omega) (fun i ↦ by have := v32 i; omega) (fun i ↦ by have := v33 i; omega) (fun i ↦ by have := v60 i; omega) rq5 rq6 (fun i ↦ by have := v28 i; omega) (fun i ↦ by have := v29 i; omega) (fun i ↦ by have := v61 i; omega) rq1 rq2 (fun i ↦ by have := v34 i; omega) (fun i ↦ by have := v35 i; omega) (fun i ↦ by have := v62 i; omega) rq7 rq8 (fun i ↦ by have := v36 i; omega) (fun i ↦ by have := v37 i; omega) rq9 rq10 (fun i ↦ by have := v38 i; omega) (fun i ↦ by have := v39 i; omega) rq11 rq12 (fun i ↦ by have := v40 i; omega) (fun i ↦ by have := v41 i; omega) rq13 rq14 (fun i ↦ by have := v42 i; omega) (fun i ↦ by have := v43 i; omega) rq15 rq16 (fun i ↦ by have := v44 i; omega) (fun i ↦ by have := v45 i; omega) rq17 rq18 (by have := v18; omega) (by have := v19; omega) (by have := v20; omega) rq31 r3 r4 r5 hG
  have P0_1 := Part_0_1.partAssembly hr hsm hrE hsmE (by have := v23; omega) (by have := v24; omega) (by have := v58; omega) (by have := v59; omega) (fun k ↦ (rq0 k).2.1) (fun k ↦ by rw [← (rq0 k).1]; exact (rq0 k).2.2) (fun i ↦ by have := v26 i; omega) (fun i ↦ by have := v32 i; omega) (fun i ↦ by have := v33 i; omega) (fun i ↦ by have := v60 i; omega) rq5 rq6 (fun i ↦ by have := v30 i; omega) (fun i ↦ by have := v31 i; omega) (fun i ↦ by have := v61 i; omega) rq3 rq4 (fun i ↦ by have := v46 i; omega) (fun i ↦ by have := v47 i; omega) (fun i ↦ by have := v62 i; omega) rq19 rq20 (fun i ↦ by have := v48 i; omega) (fun i ↦ by have := v49 i; omega) rq21 rq22 (fun i ↦ by have := v50 i; omega) (fun i ↦ by have := v51 i; omega) rq23 rq24 (fun i ↦ by have := v52 i; omega) (fun i ↦ by have := v53 i; omega) rq25 rq26 (fun i ↦ by have := v54 i; omega) (fun i ↦ by have := v55 i; omega) rq27 rq28 (fun i ↦ by have := v56 i; omega) (fun i ↦ by have := v57 i; omega) rq29 rq30 (by have := v18; omega) (by have := v19; omega) (by have := v20; omega) rq31 r6 r7 r8 hG
  have hσ : external.input_22 = sigmaOf external.input_32 external.input_33 := by
    have hl : ∀ k, external.input_27 k = (labelExp k : Int) := fun k ↦ (rq0 k).1
    funext o
    rw [rq32 o, sigmaOf, zOf]
    simp only [hl]
  rw [ideal_zero r11]
  refine decode_zero r9 hG (fun j ↦ ![freshE external.input_28 external.input_29,
    freshE external.input_30 external.input_31] j) (fun j c i ↦ by fin_cases j <;> exact freshE_bound _ _ c i)
    ?_ ?_
  · rw [hσ, hrE]
    exact P0_0
  · rw [hσ]
    exact P0_1

end RefreshTopProof
