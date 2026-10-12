import RequirementRoot
import TopSecrets
import TopGadgets
import TopFinal
import TopDecodeCrt
import TopUnpack
import StageWrappers
import Modular_Part_I_Assembly

/-! Every run of the worst-case relation decodes the ideal output (written by
`generator/gen_correct_merged.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- A fact about every member of a packed input holds at every instance. -/
theorem get_all8 {α : Type} {P : α → Prop} {a0 a1 a2 a3 a4 a5 a6 a7 : α} (h0 : P a0) (h1 : P a1)
    (h2 : P a2) (h3 : P a3) (h4 : P a4) (h5 : P a5) (h6 : P a6) (h7 : P a7) (i : Fin 8) :
    P ([a0, a1, a2, a3, a4, a5, a6, a7].get i) := by
  fin_cases i <;> assumption

@[simp] theorem get_const8 {α : Type} (a : α) (i : Fin 8) : [a, a, a, a, a, a, a, a].get i = a := by
  fin_cases i <;> rfl

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (h : W_GeneratedClaim.Runs model external execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hv, r1, r2, r3, r4, r5, r6, r7, r8, r9⟩ := h
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152, v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167, v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181, v182, v183, v184, v185⟩ := hv
  obtain ⟨r, hrE, hr⟩ := payload_ternary r1
  obtain ⟨sm, hsmE, hsm⟩ := mask_ternary r2
  obtain ⟨rq0, rq1, rq2, rq3, rq4, rq5, rq6, rq7, rq8, rq9, rq10, rq11, rq12, rq13, rq14, rq15, rq16, rq17, rq18, rq19, rq20, rq21, rq22, rq23, rq24, rq25, rq26, rq27, rq28, rq29, rq30, rq31, rq32, rq33, rq34, rq35, rq36, rq37, rq38, rq39, rq40, rq41, rq42, rq43, rq44, rq45, rq46, rq47, rq48, rq49, rq50, rq51, rq52, rq53, rq54, rq55, rq56, rq57, rq58, rq59, rq60, rq61, rq62, rq63, rq64, rq65, rq66, rq67, rq68, rq69, rq70, rq71, rq72, rq73, rq74, rq75, rq76, rq77, rq78, rq79, rq80, rq81, rq82, rq83, rq84, rq85, rq86, rq87, rq88, rq89, rq90, rq91, rq92, rq93, rq94, rq95, rq96, rq97, rq98, rq99, rq100, rq101, rq102, rq103, rq104⟩ := requirementFacts r7 r9
  obtain ⟨G, hG⟩ := gadgetQ1
  -- the instance theorem at every instance
  have P := fun (i0 : Nat) (hi : i0 < 8) ↦ haveI : Fact (i0 < 8) := ⟨hi⟩
    Part_I.Modular.partAssembly (i0 := i0) hr hsm hrE hsmE (by simp only [get_const8]; exact (by have := v74; omega)) (by simp only [get_const8]; exact (by have := v75; omega)) (by simp only [get_const8]; exact (by have := v181; omega)) (by simp only [get_const8]; exact (by have := v182; omega)) (by simp only [get_const8]; exact (fun k ↦ (rq0 k).2.1)) (by simp only [get_const8]; exact (fun k ↦ by rw [← (rq0 k).1]; exact (rq0 k).2.2)) (by simp only [get_const8]; exact (fun i ↦ by have := v77 i; omega)) (by simp only [get_const8]; exact (fun i ↦ by have := v83 i; omega)) (by simp only [get_const8]; exact (fun i ↦ by have := v84 i; omega)) (by simp only [get_const8]; exact (fun i ↦ by have := v183 i; omega)) (by simp only [get_const8]; exact rq5) (by simp only [get_const8]; exact rq6) (get_all8 (P := fun (x : Fin 240 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v79 i; omega) (fun i ↦ by have := v81 i; omega) (fun i ↦ by have := v79 i; omega) (fun i ↦ by have := v81 i; omega) (fun i ↦ by have := v79 i; omega) (fun i ↦ by have := v81 i; omega) (fun i ↦ by have := v79 i; omega) (fun i ↦ by have := v81 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 240 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v80 i; omega) (fun i ↦ by have := v82 i; omega) (fun i ↦ by have := v80 i; omega) (fun i ↦ by have := v82 i; omega) (fun i ↦ by have := v80 i; omega) (fun i ↦ by have := v82 i; omega) (fun i ↦ by have := v80 i; omega) (fun i ↦ by have := v82 i; omega) ⟨i0, hi0⟩) (by simp only [get_const8]; exact (fun i ↦ by have := v184 i; omega)) (get_all8 (P := fun (x : Fin 240 → Int) ↦ ∀ o : Fin 48, ∑ k : Fin 5, x (cellIndex o k) = 1) rq1 rq3 rq1 rq3 rq1 rq3 rq1 rq3 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 240 → Int) ↦ ∀ o : Fin 48, ∑ k : Fin 5, x (cellIndex o k) = 1) rq2 rq4 rq2 rq4 rq2 rq4 rq2 rq4 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v85 i; omega) (fun i ↦ by have := v97 i; omega) (fun i ↦ by have := v109 i; omega) (fun i ↦ by have := v121 i; omega) (fun i ↦ by have := v133 i; omega) (fun i ↦ by have := v145 i; omega) (fun i ↦ by have := v157 i; omega) (fun i ↦ by have := v169 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v86 i; omega) (fun i ↦ by have := v98 i; omega) (fun i ↦ by have := v110 i; omega) (fun i ↦ by have := v122 i; omega) (fun i ↦ by have := v134 i; omega) (fun i ↦ by have := v146 i; omega) (fun i ↦ by have := v158 i; omega) (fun i ↦ by have := v170 i; omega) ⟨i0, hi0⟩) (by simp only [get_const8]; exact (fun i ↦ by have := v185 i; omega)) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq7 rq19 rq31 rq43 rq55 rq67 rq79 rq91 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq8 rq20 rq32 rq44 rq56 rq68 rq80 rq92 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v87 i; omega) (fun i ↦ by have := v99 i; omega) (fun i ↦ by have := v111 i; omega) (fun i ↦ by have := v123 i; omega) (fun i ↦ by have := v135 i; omega) (fun i ↦ by have := v147 i; omega) (fun i ↦ by have := v159 i; omega) (fun i ↦ by have := v171 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v88 i; omega) (fun i ↦ by have := v100 i; omega) (fun i ↦ by have := v112 i; omega) (fun i ↦ by have := v124 i; omega) (fun i ↦ by have := v136 i; omega) (fun i ↦ by have := v148 i; omega) (fun i ↦ by have := v160 i; omega) (fun i ↦ by have := v172 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq9 rq21 rq33 rq45 rq57 rq69 rq81 rq93 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq10 rq22 rq34 rq46 rq58 rq70 rq82 rq94 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v89 i; omega) (fun i ↦ by have := v101 i; omega) (fun i ↦ by have := v113 i; omega) (fun i ↦ by have := v125 i; omega) (fun i ↦ by have := v137 i; omega) (fun i ↦ by have := v149 i; omega) (fun i ↦ by have := v161 i; omega) (fun i ↦ by have := v173 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v90 i; omega) (fun i ↦ by have := v102 i; omega) (fun i ↦ by have := v114 i; omega) (fun i ↦ by have := v126 i; omega) (fun i ↦ by have := v138 i; omega) (fun i ↦ by have := v150 i; omega) (fun i ↦ by have := v162 i; omega) (fun i ↦ by have := v174 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq11 rq23 rq35 rq47 rq59 rq71 rq83 rq95 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq12 rq24 rq36 rq48 rq60 rq72 rq84 rq96 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v91 i; omega) (fun i ↦ by have := v103 i; omega) (fun i ↦ by have := v115 i; omega) (fun i ↦ by have := v127 i; omega) (fun i ↦ by have := v139 i; omega) (fun i ↦ by have := v151 i; omega) (fun i ↦ by have := v163 i; omega) (fun i ↦ by have := v175 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v92 i; omega) (fun i ↦ by have := v104 i; omega) (fun i ↦ by have := v116 i; omega) (fun i ↦ by have := v128 i; omega) (fun i ↦ by have := v140 i; omega) (fun i ↦ by have := v152 i; omega) (fun i ↦ by have := v164 i; omega) (fun i ↦ by have := v176 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq13 rq25 rq37 rq49 rq61 rq73 rq85 rq97 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq14 rq26 rq38 rq50 rq62 rq74 rq86 rq98 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v93 i; omega) (fun i ↦ by have := v105 i; omega) (fun i ↦ by have := v117 i; omega) (fun i ↦ by have := v129 i; omega) (fun i ↦ by have := v141 i; omega) (fun i ↦ by have := v153 i; omega) (fun i ↦ by have := v165 i; omega) (fun i ↦ by have := v177 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v94 i; omega) (fun i ↦ by have := v106 i; omega) (fun i ↦ by have := v118 i; omega) (fun i ↦ by have := v130 i; omega) (fun i ↦ by have := v142 i; omega) (fun i ↦ by have := v154 i; omega) (fun i ↦ by have := v166 i; omega) (fun i ↦ by have := v178 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq15 rq27 rq39 rq51 rq63 rq75 rq87 rq99 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq16 rq28 rq40 rq52 rq64 rq76 rq88 rq100 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v95 i; omega) (fun i ↦ by have := v107 i; omega) (fun i ↦ by have := v119 i; omega) (fun i ↦ by have := v131 i; omega) (fun i ↦ by have := v143 i; omega) (fun i ↦ by have := v155 i; omega) (fun i ↦ by have := v167 i; omega) (fun i ↦ by have := v179 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ i, 0 ≤ x i ∧ x i ≤ 1) (fun i ↦ by have := v96 i; omega) (fun i ↦ by have := v108 i; omega) (fun i ↦ by have := v120 i; omega) (fun i ↦ by have := v132 i; omega) (fun i ↦ by have := v144 i; omega) (fun i ↦ by have := v156 i; omega) (fun i ↦ by have := v168 i; omega) (fun i ↦ by have := v180 i; omega) ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq17 rq29 rq41 rq53 rq65 rq77 rq89 rq101 ⟨i0, hi0⟩) (get_all8 (P := fun (x : Fin 800 → Int) ↦ ∀ o : Fin 160, ∑ k : Fin 5, x (cellIndex o k) = 1) rq18 rq30 rq42 rq54 rq66 rq78 rq90 rq102 ⟨i0, hi0⟩) (by simp only [get_const8]; exact (by have := v69; omega)) (by simp only [get_const8]; exact (by have := v70; omega)) (by have := v71; omega) (by simp only [get_const8]; exact rq103) (preprocess_at r3 ⟨i0, hi0⟩) (encrypt_at r4 ⟨i0, hi0⟩) (online_eval_at r5 ⟨i0, hi0⟩) hG
  have U := online_eval_unpack r5
  have hσ : external.input_73 = sigmaOf external.input_83 external.input_84 := by
    have hl : ∀ k, external.input_78 k = (labelExp k : Int) := fun k ↦ (rq0 k).1
    funext o
    rw [rq104 o, sigmaOf, zOf]
    simp only [hl]
  rw [ideal_zero r8]
  refine decode_zero r6 hG (fun j ↦ ![freshE external.input_79 external.input_80,
    freshE external.input_81 external.input_82] j)
    (fun j c i ↦ by fin_cases j <;> exact lt_of_lt_of_le (freshE_bound _ _ c i) (by decide))
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · rw [hσ, hrE, U.1]
    exact P 0 (by decide)
  · rw [hσ, hrE, U.2.2.1]
    exact P 2 (by decide)
  · rw [hσ, hrE, U.2.2.2.2.1]
    exact P 4 (by decide)
  · rw [hσ, hrE, U.2.2.2.2.2.2.1]
    exact P 6 (by decide)
  · rw [hσ, U.2.1]
    exact P 1 (by decide)
  · rw [hσ, U.2.2.2.1]
    exact P 3 (by decide)
  · rw [hσ, U.2.2.2.2.2.1]
    exact P 5 (by decide)
  · rw [hσ, U.2.2.2.2.2.2.2]
    exact P 7 (by decide)

end RefreshTopProof
