import TopKeyForms
import TopLut2
import W_Stage_preprocess_0_0
import W_Stage_online_eval_0_0
import Part_0_0_Smudge

/-! The keys of every LUT (written by `generator/gen_keys_luts.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_42_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q5 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2 layoutQ0
    rfl rfl (digits := 8) (basis := [q0, q1, q2, q3, p4]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_42_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_42_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q5 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q5 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_42_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q5 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_42_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_generatedRoot_146 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat}
    (hi : i < 4) {state next : Keys Q5 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_generatedRoot_146 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_generatedRoot_146 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_generatedRoot_146.constraints_0, W_Stage_preprocess_0_0.sequential_generatedRoot_146.constraints_1, W_Stage_preprocess_0_0.sequential_generatedRoot_146.constraints_2, W_Stage_preprocess_0_0.sequential_generatedRoot_146.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2 layoutQ0
    rfl rfl (digits := 8) (basis := [q0, q1, q2, q3, p4]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_generatedRoot_146` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_generatedRoot_146 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q5 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q5 6) ↦
      W_Stage_preprocess_0_0.sequential_generatedRoot_146 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q5 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_generatedRoot_146 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_generatedRoot_151 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_generatedRoot_151 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_generatedRoot_151 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_generatedRoot_151.constraints_0, W_Stage_preprocess_0_0.sequential_generatedRoot_151.constraints_1, W_Stage_preprocess_0_0.sequential_generatedRoot_151.constraints_2, W_Stage_preprocess_0_0.sequential_generatedRoot_151.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_generatedRoot_151` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_generatedRoot_151 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_generatedRoot_151 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_generatedRoot_151 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_152_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_152_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_152_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_152_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_152_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_generatedRoot_185 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat}
    (hi : i < 2) {state next : Keys Q0 4} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 2 → ExactMatrix Q1 N 1 4}
    (h : W_Stage_preprocess_0_0.sequential_generatedRoot_185 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_generatedRoot_185 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_generatedRoot_185.constraints_0, W_Stage_preprocess_0_0.sequential_generatedRoot_185.constraints_1, W_Stage_preprocess_0_0.sequential_generatedRoot_185.constraints_2, W_Stage_preprocess_0_0.sequential_generatedRoot_185.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 0 (sigma2 i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 4) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma2 i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_generatedRoot_185` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_generatedRoot_185 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 2 → ExactMatrix Q1 N 1 4}
    {count : Nat} (hcount : count ≤ 2) {init final : Keys Q0 4}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 4) ↦
      W_Stage_preprocess_0_0.sequential_generatedRoot_185 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2 ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 4) ↦ k ≤ 2 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2 ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_generatedRoot_185 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_217_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q5 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2 layoutQ0
    rfl rfl (digits := 8) (basis := [q0, q1, q2, q3, p4]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_217_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_217_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q5 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q5 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_217_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q5 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_217_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_219_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_219_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_219_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_219_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_219_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_323_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_323_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_323_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_323_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_323_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_388_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_388_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_388_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_388_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_388_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_432_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_432_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_432_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_432_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_432_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_476_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_476_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_476_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_476_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_476_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_520_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_520_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_520_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_520_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_520_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem preKeyStep_sequential_parallel_generatedRoot_564_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 4) {state next : Keys Q0 6} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (h : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, key, A00, A10, f00, A01, A11, f10, f01, f11,
        ()) next) :
    next = state + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma
      ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  unfold W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10 at h
  extract_lets -merge _params _backend w_0_0 w_44_0 w_69_0 w_72_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_45_0 w_50_0 w_2_0 w_12_0 at h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_10_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 at h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  extract_lets -merge w_32_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  obtain ⟨w_37_0, h⟩ := h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 w_41_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  extract_lets -merge w_54_0 w_55_0 at h
  obtain ⟨w_56_0, h⟩ := h
  obtain ⟨w_57_0, h⟩ := h
  obtain ⟨w_58_0, h⟩ := h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 at h
  obtain ⟨w_61_0, h⟩ := h
  obtain ⟨w_62_0, h⟩ := h
  obtain ⟨w_63_0, h⟩ := h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 at h
  obtain ⟨w_70_0, h⟩ := h
  extract_lets -merge w_71_0 at h
  obtain ⟨w_73_0, h⟩ := h
  extract_lets -merge w_74_0 at h
  simp only [W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10.constraints_0, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10.constraints_1, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10.constraints_2, W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10.constraints_3] at h
  obtain ⟨-, -, h3_0, h4_0, h5_0, -, -, h7_0, -, -, h9_0, h10_0, h12_0, h12_0_1, h13_0, h14_0, h15_0, -, -, h18_0, -, -, h20_0, h21_0, h12_0_2, h12_0_3, h23_0, h24_0, h25_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h36_0, h37_0, h38_0, h39_0, -, -, -, -, -, -, h42_0, h12_0_4, h12_0_5, h46_0, h47_0, h48_0, h12_0_6, h12_0_7, h51_0, h52_0, h53_0, -, -, -, -, -, -, h56_0, h57_0, h58_0, h59_0, -, -, -, -, -, -, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h67_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h73_0, hout⟩ := h
  have hidx : w_12_0 = Int.ofNat i := rfl
  rw [hidx] at h13_0 h23_0 h46_0 h51_0
  obtain ⟨_, e13⟩ := familyGet_eq (k := i) h13_0
  obtain ⟨_, e23⟩ := familyGet_eq (k := i) h23_0
  obtain ⟨_, e46⟩ := familyGet_eq (k := i) h46_0
  obtain ⟨_, e51⟩ := familyGet_eq (k := i) h51_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_3_0, w_33_0], ![w_4_0, w_34_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h33_0).symm
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h34_0).symm
  have hsw : (fun k j ↦ ![![f00, f01], ![f10, f11]] k j ⟨i, hi⟩) = ![![w_13_0, w_46_0], ![w_23_0, w_51_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact e13.symm
    · exact e46.symm
    · exact e23.symm
    · exact e51.symm
  have hK := branchKey_of_pre (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1 layoutQ1
    rfl rfl (digits := 6) (basis := [q0, q1, q2, q3]) (index := sigma i)
    (A := ![![A00, A01], ![A10, A11]]) (Ad := ![![w_3_0, w_33_0], ![w_4_0, w_34_0]])
    h7_0 h9_0 h18_0 h20_0 h10_0 h21_0 h5_0 h35_0
    (sw := ![![w_13_0, w_46_0], ![w_23_0, w_51_0]])
    (dA := ![w_14_0, w_47_0]) (rA := ![w_15_0, w_48_0]) (dB := ![w_24_0, w_52_0]) (rB := ![w_25_0, w_53_0])
    (dC := ![w_30_0, w_58_0]) (rC := ![w_31_0, w_59_0]) (dE := ![w_38_0, w_63_0]) (rE := ![w_39_0, w_64_0])
    (fun j ↦ by fin_cases j; exacts [h14_0, h47_0]) (fun j ↦ by fin_cases j; exacts [h15_0, h48_0])
    (fun j ↦ by fin_cases j; exacts [h24_0, h52_0]) (fun j ↦ by fin_cases j; exacts [h25_0, h53_0])
    (M := ![![w_28_0, w_56_0], ![w_36_0, w_61_0]])
    (fun j ↦ by fin_cases j; exacts [h28_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h36_0, h61_0])
    (Asw := ![![w_29_0, w_57_0], ![w_37_0, w_62_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h29_0, h57_0, h37_0, h62_0])
    (fun j ↦ by fin_cases j; exacts [h30_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h31_0, h59_0])
    (fun j ↦ by fin_cases j; exacts [h38_0, h63_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h64_0])
    (K := ![![w_42_0, w_67_0], ![w_70_0, w_73_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h67_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h73_0])
  rw [← hAd, ← hsw] at hK
  rw [lutKeys, dif_pos hi, ← hK, hout]
  rfl

/-- The loop `sequential_parallel_generatedRoot_564_10` accumulates the keys of its first `count` branches. -/
theorem preKeyLoop_sequential_parallel_generatedRoot_564_10 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {o : Nat} {key : ByteArray}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {count : Nat} (hcount : count ≤ 4) {init final : Keys Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : Keys Q0 6) ↦
      W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_564_10 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, key, A00, A10, f00, A01, A11, f10,
          f01, f11, ()) next) count init final) :
    final = init + ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : Keys Q0 6) ↦ k ≤ 4 → s = init + ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i)
    (fun _ ↦ by simp) (fun i current next hinv hbody hk ↦ ?_) hruns hcount
  rw [preKeyStep_sequential_parallel_generatedRoot_564_10 (by omega) hbody, hinv (by omega), Finset.sum_range_succ, add_assoc]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_generatedRoot_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {i : Nat}
    (hi : i < 1) {state next : LoopState Q0 Q1 4} {key : ByteArray} {h0 h1 : Fin 2 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 2 → ExactMatrix Q1 N 1 4}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_generatedRoot_18 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2 ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_generatedRoot_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_generatedRoot_18.constraints_0, W_Stage_online_eval_0_0.sequential_generatedRoot_18.constraints_1, W_Stage_online_eval_0_0.sequential_generatedRoot_18.constraints_2, W_Stage_online_eval_0_0.sequential_generatedRoot_18.constraints_3, W_Stage_online_eval_0_0.sequential_generatedRoot_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 0 (sigma2 i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 4) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma2 i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_generatedRoot_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_generatedRoot_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {key : ByteArray}
    {h0 h1 : Fin 2 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 2 → ExactMatrix Q1 N 1 4} {count : Nat} (hcount : count ≤ 1)
    {final : LoopState Q0 Q1 4}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 4) ↦
      W_Stage_online_eval_0_0.sequential_generatedRoot_18 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2 ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 4) ↦ ∀ hk : k ≤ 1,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model key 0 sigma2 ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_generatedRoot_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_121_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q5 Q0 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2
    layoutQ0 rfl rfl (digits := 8) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_121_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_121_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q5 Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q5 Q0 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_121_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q5 Q0 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_121_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_generatedRoot_186 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {i : Nat}
    (hi : i < 3) {state next : LoopState Q5 Q0 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_generatedRoot_186 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_generatedRoot_186 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_generatedRoot_186.constraints_0, W_Stage_online_eval_0_0.sequential_generatedRoot_186.constraints_1, W_Stage_online_eval_0_0.sequential_generatedRoot_186.constraints_2, W_Stage_online_eval_0_0.sequential_generatedRoot_186.constraints_3, W_Stage_online_eval_0_0.sequential_generatedRoot_186.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2
    layoutQ0 rfl rfl (digits := 8) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_generatedRoot_186` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_generatedRoot_186 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q5 Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q5 Q0 6) ↦
      W_Stage_online_eval_0_0.sequential_generatedRoot_186 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q5 Q0 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_generatedRoot_186 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_generatedRoot_252 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_generatedRoot_252 Backend.backend model params i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_generatedRoot_252 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_generatedRoot_252.constraints_0, W_Stage_online_eval_0_0.sequential_generatedRoot_252.constraints_1, W_Stage_online_eval_0_0.sequential_generatedRoot_252.constraints_2, W_Stage_online_eval_0_0.sequential_generatedRoot_252.constraints_3, W_Stage_online_eval_0_0.sequential_generatedRoot_252.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_generatedRoot_252` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_generatedRoot_252 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_generatedRoot_252 Backend.backend model params i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_generatedRoot_252 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_312_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_312_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_312_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_312_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_312_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_392_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q5 Q0 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q5 8 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2
    layoutQ0 rfl rfl (digits := 8) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_392_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_392_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q5 Q0 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q5 Q0 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_392_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q5 Q0 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_392_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_393_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 3 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_393_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_393_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_393_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_393_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_560_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_560_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_560_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_560_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_560_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_615_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_615_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_615_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_615_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_615_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_669_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_669_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_669_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_669_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_669_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_723_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_723_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_723_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_723_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_723_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_777_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_777_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_777_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_777_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_777_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem keyStep_sequential_parallel_generatedRoot_831_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {i : Nat}
    (hi : i < 3) {state next : LoopState Q0 Q1 6} {key : ByteArray} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6}
    (hswitch : state.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i, by omega⟩)
    (hbody : W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18 Backend.backend model params o i
      (state.1, state.2.1, state.2.2.1, state.2.2.2.1, state.2.2.2.2.1, state.2.2.2.2.2.1,
        state.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.1, state.2.2.2.2.2.2.2.2.1,
        state.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) :
    keyTuple next.key = keyTuple state.key + lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      next.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨i + 1, by omega⟩ := by
  unfold W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18 at hbody
  extract_lets -merge _params _backend w_0_0 w_48_0 w_67_0 w_70_0 w_73_0 w_76_0 w_30_0 w_49_0 w_31_0 w_50_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_24_0 w_26_0 w_79_0 w_83_0 w_85_0 w_87_0 w_2_0 w_5_0 w_80_0 w_81_0 at hbody
  obtain ⟨w_3_0, hbody⟩ := hbody
  obtain ⟨w_6_0, hbody⟩ := hbody
  obtain ⟨w_7_0, hbody⟩ := hbody
  obtain ⟨w_8_0, hbody⟩ := hbody
  obtain ⟨w_8_concat_1, hbody⟩ := hbody
  obtain ⟨w_10_0, hbody⟩ := hbody
  obtain ⟨w_11_0, hbody⟩ := hbody
  obtain ⟨w_12_0, hbody⟩ := hbody
  obtain ⟨w_13_0, hbody⟩ := hbody
  obtain ⟨w_13_concat_1, hbody⟩ := hbody
  obtain ⟨w_14_0, hbody⟩ := hbody
  obtain ⟨w_16_0, hbody⟩ := hbody
  obtain ⟨w_18_0, hbody⟩ := hbody
  obtain ⟨w_20_0, hbody⟩ := hbody
  obtain ⟨w_21_0, hbody⟩ := hbody
  obtain ⟨w_21_concat_1, hbody⟩ := hbody
  obtain ⟨w_23_0, hbody⟩ := hbody
  obtain ⟨w_25_0, hbody⟩ := hbody
  obtain ⟨w_27_0, hbody⟩ := hbody
  obtain ⟨w_28_0, hbody⟩ := hbody
  obtain ⟨w_28_concat_1, hbody⟩ := hbody
  obtain ⟨w_29_0, hbody⟩ := hbody
  obtain ⟨w_32_0, hbody⟩ := hbody
  obtain ⟨w_33_0, hbody⟩ := hbody
  obtain ⟨w_34_0, hbody⟩ := hbody
  extract_lets -merge w_35_0 at hbody
  obtain ⟨w_36_0, hbody⟩ := hbody
  obtain ⟨w_37_0, hbody⟩ := hbody
  obtain ⟨w_38_0, hbody⟩ := hbody
  obtain ⟨w_39_0, hbody⟩ := hbody
  obtain ⟨w_40_0, hbody⟩ := hbody
  obtain ⟨w_41_0, hbody⟩ := hbody
  obtain ⟨w_42_0, hbody⟩ := hbody
  extract_lets -merge w_43_0 at hbody
  obtain ⟨w_44_0, hbody⟩ := hbody
  obtain ⟨w_45_0, hbody⟩ := hbody
  extract_lets -merge w_46_0 w_47_0 at hbody
  obtain ⟨w_51_0, hbody⟩ := hbody
  obtain ⟨w_52_0, hbody⟩ := hbody
  obtain ⟨w_53_0, hbody⟩ := hbody
  extract_lets -merge w_54_0 at hbody
  obtain ⟨w_55_0, hbody⟩ := hbody
  obtain ⟨w_56_0, hbody⟩ := hbody
  obtain ⟨w_57_0, hbody⟩ := hbody
  obtain ⟨w_58_0, hbody⟩ := hbody
  obtain ⟨w_59_0, hbody⟩ := hbody
  obtain ⟨w_60_0, hbody⟩ := hbody
  obtain ⟨w_61_0, hbody⟩ := hbody
  extract_lets -merge w_62_0 at hbody
  obtain ⟨w_63_0, hbody⟩ := hbody
  obtain ⟨w_64_0, hbody⟩ := hbody
  extract_lets -merge w_65_0 w_66_0 at hbody
  obtain ⟨w_68_0, hbody⟩ := hbody
  extract_lets -merge w_69_0 at hbody
  obtain ⟨w_71_0, hbody⟩ := hbody
  extract_lets -merge w_72_0 at hbody
  obtain ⟨w_74_0, hbody⟩ := hbody
  extract_lets -merge w_75_0 at hbody
  obtain ⟨w_77_0, hbody⟩ := hbody
  extract_lets -merge w_78_0 at hbody
  obtain ⟨w_82_0, hbody⟩ := hbody
  obtain ⟨w_84_0, hbody⟩ := hbody
  obtain ⟨w_86_0, hbody⟩ := hbody
  obtain ⟨w_88_0, hbody⟩ := hbody
  simp only [W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18.constraints_0, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18.constraints_1, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18.constraints_2, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18.constraints_3, W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18.constraints_4] at hbody
  obtain ⟨h2_0, h2_0_1, h3_0, -, -, h6_0, h7_0, h6_0_1, h8_0, h2_0_2, h2_0_3, h10_0, h11_0, h12_0, h11_0_1, h13_0, h14_0, -, -, h16_0, -, -, h18_0, -, -, h20_0, h18_0_1, h21_0, -, -, h23_0, -, -, h25_0, -, -, h27_0, h25_0_1, h28_0, h29_0, h32_0, h33_0, h34_0, -, -, -, -, -, -, h36_0, h37_0, -, -, -, -, -, -, h38_0, h39_0, h40_0, h41_0, h42_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h45_0, h51_0, h52_0, h53_0, -, -, -, -, -, -, h55_0, h56_0, -, -, -, -, -, -, h57_0, h58_0, h59_0, h60_0, h61_0, -, -, -, -, -, -, h63_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h77_0, h81_0, h81_0_1, h82_0, h81_0_2, h81_0_3, h84_0, h81_0_4, h81_0_5, h86_0, h81_0_6, h81_0_7, h88_0, hout⟩ := hbody
  have hidx : w_2_0 = Int.ofNat i := rfl
  have hnext : w_81_0 = Int.ofNat (i + 1) := by
    simp only [w_81_0, w_80_0, w_2_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hnext] at h82_0 h84_0 h86_0 h88_0
  obtain ⟨_, hs0⟩ := familyGet_eq h82_0
  obtain ⟨_, hs1⟩ := familyGet_eq h84_0
  obtain ⟨_, hs2⟩ := familyGet_eq h86_0
  obtain ⟨_, hs3⟩ := familyGet_eq h88_0
  have hAd : lutA model key Q0 6 2 (sigma i) = ![![w_6_0, w_11_0], ![w_7_0, w_12_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h11_0).symm
    · exact (hashSample_eq h7_0).symm
    · exact (hashSample_eq h12_0).symm
  have hK := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma i)
    h16_0 h18_0 h20_0 h23_0 h25_0 h27_0 h18_0_1 h21_0 h25_0_1 h28_0 h29_0
    (sw := state.switch) (Xb := ![w_32_0, w_51_0])
    (fun j ↦ by fin_cases j; exacts [h32_0, h51_0])
    (D' := ![w_33_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h33_0, h52_0])
    (D := ![w_34_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h34_0, h53_0])
    (M := ![![w_36_0, w_55_0], ![w_38_0, w_57_0]])
    (fun j ↦ by fin_cases j; exacts [h36_0, h55_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h57_0])
    (Asw := ![![w_37_0, w_56_0], ![w_39_0, w_58_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h37_0, h56_0, h39_0, h58_0])
    (Z := ![w_40_0, w_59_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h59_0])
    (E' := ![w_41_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h41_0, h60_0])
    (E := ![w_42_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h42_0, h61_0])
    (h := ![w_3_0, w_10_0]) (Ad := ![![w_6_0, w_11_0], ![w_7_0, w_12_0]])
    h6_0_1 h8_0 h11_0_1 h13_0 h14_0 (K := ![![w_68_0, w_71_0], ![w_74_0, w_77_0]])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h77_0])
  rw [← hAd, hswitch] at hK
  subst hout
  refine ⟨?_, ?_⟩
  · rw [lutKeys, dif_pos (by omega), ← hK]
    rfl
  · funext k j
    fin_cases k <;> fin_cases j
    exacts [hs0, hs1, hs2, hs3]

/-- The loop `sequential_parallel_generatedRoot_831_18` accumulates the keys of its first `count` branches. -/
theorem keyLoop_sequential_parallel_generatedRoot_831_18 {model : HashModel} {params : W_Stage_online_eval_0_0.Params} {o : Nat} {key : ByteArray}
    {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6} {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {count : Nat} (hcount : count ≤ 3)
    {final : LoopState Q0 Q1 6}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : LoopState Q0 Q1 6) ↦
      W_Stage_online_eval_0_0.sequential_parallel_generatedRoot_831_18 Backend.backend model params o i
        (current.1, current.2.1, current.2.2.1, current.2.2.2.1, current.2.2.2.2.1, current.2.2.2.2.2.1,
        current.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.1, current.2.2.2.2.2.2.2.2.1,
        current.2.2.2.2.2.2.2.2.2.1, h0, key, h1, c0, A00, A10, c1, A01, A11, fam00, fam01,
        fam10, fam11, ()) next) count
      (0, 0, 0, 0, 0, 0, fam00 0, fam01 0, fam10 0, fam11 0, ()) final) :
    keyTuple final.key = ∑ i ∈ Finset.range count, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
      final.switch = fun k j ↦ ![![fam00, fam01], ![fam10, fam11]] k j ⟨count, by omega⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : LoopState Q0 Q1 6) ↦ ∀ hk : k ≤ 3,
      keyTuple s.key = ∑ i ∈ Finset.range k, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i ∧
        s.switch = fun k' j ↦ ![![fam00, fam01], ![fam10, fam11]] k' j ⟨k, by omega⟩) ?_ ?_ hruns hcount
  · intro _
    refine ⟨rfl, ?_⟩
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := keyStep_sequential_parallel_generatedRoot_831_18 (by omega) hsw hbody
    exact ⟨by rw [hrel', hrel, Finset.sum_range_succ], hsw'⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound121 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_121 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (blockModSwitch (p := Q0) [q0, q1, q2, q3, p4] 1) (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_121.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_121.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_121.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_121.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_121.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_121.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, h55_0, -, -, -, -, -, -, h57_0, h58_0, -, -, -, -, -, -, h59_0, h60_0, h61_0, h62_0, h63_0, -, -, -, -, -, -, h65_0, h67_0, -, -, -, -, -, -, h68_0, h70_0, -, -, -, -, -, -, h71_0, h73_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h80_0, h83_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_121_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2
    layoutQ0 rfl rfl (digits := 8) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h53_0])
    (D' := ![w_39_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h54_0])
    (D := ![w_40_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h55_0])
    (M := ![![w_42_0, w_57_0], ![w_44_0, w_59_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h57_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h59_0])
    (Asw := ![![w_43_0, w_58_0], ![w_45_0, w_60_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h58_0, h45_0, h60_0])
    (Z := ![w_46_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h61_0])
    (E' := ![w_47_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h62_0])
    (E := ![w_48_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h63_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q5 8 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h65_0]) (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_66_0, w_69_0, w_72_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_65_0], ![w_68_0, w_71_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)
  · exact blockModSwitchRuns_unique h67_0 (blockModSwitchRuns_fn h67_0 _)
  · exact blockModSwitchRuns_unique h70_0 (blockModSwitchRuns_fn h70_0 _)
  · exact blockModSwitchRuns_unique h73_0 (blockModSwitchRuns_fn h73_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound392 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q0 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q5 N 1 8}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q5 N 1 8} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_392 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (blockModSwitch (p := Q0) [q0, q1, q2, q3, p4] 1) (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_392.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_392.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_392.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_392.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_392.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_392.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, h55_0, -, -, -, -, -, -, h57_0, h58_0, -, -, -, -, -, -, h59_0, h60_0, h61_0, h62_0, h63_0, -, -, -, -, -, -, h65_0, h67_0, -, -, -, -, -, -, h68_0, h70_0, -, -, -, -, -, -, h71_0, h73_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h80_0, h83_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_392_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q5 8 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by norm_num) Backend.layout2
    layoutQ0 rfl rfl (digits := 8) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h53_0])
    (D' := ![w_39_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h54_0])
    (D := ![w_40_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h55_0])
    (M := ![![w_42_0, w_57_0], ![w_44_0, w_59_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h57_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h59_0])
    (Asw := ![![w_43_0, w_58_0], ![w_45_0, w_60_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h58_0, h45_0, h60_0])
    (Z := ![w_46_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h61_0])
    (E' := ![w_47_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h62_0])
    (E := ![w_48_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h63_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q5 8 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h65_0]) (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_66_0, w_69_0, w_72_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_65_0], ![w_68_0, w_71_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)
  · exact blockModSwitchRuns_unique h67_0 (blockModSwitchRuns_fn h67_0 _)
  · exact blockModSwitchRuns_unique h70_0 (blockModSwitchRuns_fn h70_0 _)
  · exact blockModSwitchRuns_unique h73_0 (blockModSwitchRuns_fn h73_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound393 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_393 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_393.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_393.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_393.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_393.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_393.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_393.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h53_0, h54_0, -, -, -, -, -, -, h56_0, h57_0, -, -, -, -, -, -, h58_0, h59_0, h60_0, h61_0, h62_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h66_0, -, -, -, -, -, -, h68_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h75_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_393_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_52_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h52_0])
    (D' := ![w_39_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h53_0])
    (D := ![w_40_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h54_0])
    (M := ![![w_42_0, w_56_0], ![w_44_0, w_58_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h56_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h58_0])
    (Asw := ![![w_43_0, w_57_0], ![w_45_0, w_59_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h57_0, h45_0, h59_0])
    (Z := ![w_46_0, w_60_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h60_0])
    (E' := ![w_47_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h61_0])
    (E := ![w_48_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h62_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 3 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_64_0], ![w_66_0, w_68_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h64_0]) (fun j ↦ by fin_cases j; exacts [h66_0, h68_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_65_0, w_67_0, w_69_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_64_0], ![w_66_0, w_68_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS, hout]
  try rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound312 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {off : Blocks Q0 6} {outputs : Blocks Q0 6}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_312 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, off.1, off.2.1, off.2.2.1, off.2.2.2.1, off.2.2.2.2.1, off.2.2.2.2.2.1, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) + (off.1, off.2.1, off.2.2.1, off.2.2.2.1, ()) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_312.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_52_0 w_68_0 w_72_0 w_76_0 w_82_0 w_88_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_89_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_312.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_312.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_312.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_312.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_312.constraints_4] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h54_0, h55_0, h56_0, -, -, -, -, -, -, h58_0, h59_0, -, -, -, -, -, -, h60_0, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h74_0, -, -, -, -, -, -, h78_0, -, -, -, -, -, -, h79_0, -, -, -, -, -, -, h84_0, -, -, -, -, -, -, h85_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_312_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 3 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h54_0])
    (D' := ![w_39_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h55_0])
    (D := ![w_40_0, w_56_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h56_0])
    (M := ![![w_42_0, w_58_0], ![w_44_0, w_60_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h58_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h60_0])
    (Asw := ![![w_43_0, w_59_0], ![w_45_0, w_61_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h59_0, h45_0, h61_0])
    (Z := ![w_46_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h62_0])
    (E' := ![w_47_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h63_0])
    (E := ![w_48_0, w_64_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h64_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 3 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_66_0], ![w_70_0, w_74_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h66_0]) (fun j ↦ by fin_cases j; exacts [h70_0, h74_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_67_0, w_71_0, w_75_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_66_0], ![w_70_0, w_74_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS, hout]
  try rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound42 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_42 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (blockModSwitch (p := Q0) [q0, q1, q2, q3, p4] 1) (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_42.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_42.constraints_0] at h
  obtain ⟨-, h10_3, h11_0, h12_0, h13_0, h14_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_42_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6),
    (0 : ExactMatrix Q5 N 1 6), ()) = (0 : Keys Q5 6) from rfl, zero_add] at hloop
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _)
  · exact blockModSwitchRuns_unique h12_0 (blockModSwitchRuns_fn h12_0 _)
  · exact blockModSwitchRuns_unique h13_0 (blockModSwitchRuns_fn h13_0 _)
  · exact blockModSwitchRuns_unique h14_0 (blockModSwitchRuns_fn h14_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound217 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q5 N 1 8} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q0 N 1 6}
    {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_217 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (blockModSwitch (p := Q0) [q0, q1, q2, q3, p4] 1) (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_217.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_217.constraints_0] at h
  obtain ⟨-, h10_3, h11_0, h12_0, h13_0, h14_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_217_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6),
    (0 : ExactMatrix Q5 N 1 6), ()) = (0 : Keys Q5 6) from rfl, zero_add] at hloop
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _)
  · exact blockModSwitchRuns_unique h12_0 (blockModSwitchRuns_fn h12_0 _)
  · exact blockModSwitchRuns_unique h13_0 (blockModSwitchRuns_fn h13_0 _)
  · exact blockModSwitchRuns_unique h14_0 (blockModSwitchRuns_fn h14_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound219 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_219 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_219.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_219.constraints_0] at h
  obtain ⟨-, h10_3, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_219_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  rw [hout, ← hloop]

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound152 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6} {off : Keys Q0 6}
    {outputs : Keys Q0 6}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_152 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, off.1, off.2.1, off.2.2.1, off.2.2.2.1, ()) outputs) :
    outputs = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 3 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) + off := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_152.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_12_0 w_14_0 w_16_0 w_18_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_152.constraints_0] at h
  obtain ⟨-, h10_3, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_152_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  rw [hout, ← hloop]
  rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound323 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_323 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_323.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_323.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_323_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound388 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_388 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_388.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_388.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_388.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_388_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound432 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_432 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_432.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_432.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_432_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound476 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_476 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_476.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_476.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_476_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound520 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_520 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_520.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_520.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_520_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem preKeyRound564 {model : HashModel} {key : ByteArray} {params : W_Stage_preprocess_0_0.Params} {o : Nat}
    {A00 A10 A01 A11 : ExactMatrix Q0 N 1 6} {f00 f10 f01 f11 : Fin 4 → ExactMatrix Q1 N 1 6}
    {outputs : Keys Q1 1}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_564 Backend.backend model params o
      (key, A00, A10, f00, A01, A11, f10, f01, f11, ()) outputs) :
    outputs = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![f00, f01], ![f10, f11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_564.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_32_concat_2 w_32_concat_3 w_32_concat_4 w_32_concat_5 w_32_concat_6 w_32_concat_7 w_32_concat_8 w_32_concat_9 w_32_concat_10 w_32_concat_11 w_32_concat_12 w_32_concat_13 w_32_concat_14 w_32_concat_15 w_32_concat_16 w_32_concat_17 w_32_concat_18 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_564.constraints_1] at h
  obtain ⟨-, h10_3, h11_0, h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, -, -, -, -, -, -, -, -, h33_0, h34_0, h35_0, -, h39_0, h42_0, h45_0, hout⟩ := h
  rw [show Int.toNat 4 = 4 from rfl] at h10_3
  have hloop := preKeyLoop_sequential_parallel_generatedRoot_564_10 (count := 4) (by decide) h10_3
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add] at hloop
  simp only [concatColumns] at h13_0 h14_0 h15_0 h16_0 h17_0 h18_0 h19_0 h20_0 h21_0 h22_0 h23_0 h24_0 h25_0 h26_0 h27_0 h28_0 h29_0 h30_0 h32_0
  have hrow : ∀ c : Fin 20, w_32_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h13_0, h14_0, h15_0, h16_0, h17_0, h18_0, h19_0, h20_0, h21_0, h22_0, h23_0, h24_0, h25_0, h26_0, h27_0, h28_0, h29_0, h30_0, h32_0, w_12_0, w_13_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_23_0, w_24_0, w_25_0, w_26_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h33_0
  have hK : w_33_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h33_0 (by simp; omega)]
    exact hrow _
  have hW : w_35_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h35_0, ← hK]
    exact dec_eq h34_0
  rw [hout, ← hloop]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · rw [← blockModSwitchRuns_unique h11_0 (blockModSwitchRuns_fn h11_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h39_0 (blockModSwitchRuns_fn h39_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h42_0 (blockModSwitchRuns_fn h42_0 _), ← hW]
    rfl
  · rw [← blockModSwitchRuns_unique h45_0 (blockModSwitchRuns_fn h45_0 _), ← hW]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound560 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_560 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_560.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_560.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_560_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight560 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight560]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight560 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound615 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_615 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight615 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_615.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_615.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_615_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight615 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight615]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight615 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight615 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound669 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_669 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight669 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_669.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_669.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_669_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight669 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight669]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight669 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight669 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound723 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_723 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight723 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_723.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_723.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_723_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight723 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight723]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight723 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight723 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound777 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_777 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight777 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_777.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_777.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_777_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight777 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight777]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight777 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight777 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem keyRound831 {model : HashModel} {key : ByteArray} {params : W_Stage_online_eval_0_0.Params} {o : Nat}
    {fam00 fam01 fam10 fam11 : Fin 4 → ExactMatrix Q1 N 1 6} {h0 h1 : Fin 4 → ExactMatrix Q0 N 1 6}
    {c0 A00 A10 c1 A01 A11 : ExactMatrix Q0 N 1 6} {outputs : Blocks Q1 1}
    (h : W_Stage_online_eval_0_0.parallel_generatedRoot_831 Backend.backend model params o
      (fam00, fam01, fam10, fam11, h0, key, h1, c0, A00, A10, c1, A01, A11, ()) outputs) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) = keyMap (fun x ↦ multiplyMonomial (blockModSwitch (p := Q1) [q0, q1, q2, q3] 1 x * (dec (inner := 6) 6 (matrixPolynomial [smudgeWeight831 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] : ExactMatrix Q1 N 1 1))) (Int.fdiv (Int.ofNat o) 20)) (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
  obtain ⟨w, h⟩ := h
  unfold W_Stage_online_eval_0_0.parallel_generatedRoot_831.body at h
  extract_lets -merge _params _backend w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_concat_1 w_73_concat_2 w_73_concat_3 w_73_concat_4 w_73_concat_5 w_73_concat_6 w_73_concat_7 w_73_concat_8 w_73_concat_9 w_73_concat_10 w_73_concat_11 w_73_concat_12 w_73_concat_13 w_73_concat_14 w_73_concat_15 w_73_concat_16 w_73_concat_17 w_73_concat_18 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 at h
  simp only [W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_0, W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_1, W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_2, W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_3, W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_4, W_Stage_online_eval_0_0.parallel_generatedRoot_831.constraints_5] at h
  obtain ⟨h2_0, h4_0, h6_0, h8_0, -, h18_9, h19_0, -, -, h21_0, h22_0, h21_0_1, h23_0, h24_0, h25_0, h26_0, h25_0_1, h27_0, h28_0, -, -, h29_0, -, -, h30_0, -, -, h31_0, h30_0_1, h32_0, -, -, h33_0, -, -, h34_0, -, -, h35_0, h34_0_1, h36_0, h37_0, h38_0, h39_0, h40_0, -, -, -, -, -, -, h42_0, h43_0, -, -, -, -, -, -, h44_0, h45_0, h46_0, h47_0, h48_0, -, -, -, -, -, -, h50_0, h52_0, h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, -, -, -, -, -, -, -, -, h74_0, h75_0, h76_0, -, h80_0, h81_0, h82_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, h88_0, h89_0, h90_0, -, -, -, -, -, -, h92_0, h94_0, -, -, -, -, -, -, h97_0, h99_0, -, -, -, -, -, -, h102_0, h104_0, -, -, -, -, -, -, h107_0, -, -, -, -, -, -, h108_0, h111_0, -, -, -, -, -, -, h114_0, -, -, -, -, -, -, h115_0, h118_0, hout⟩ := h
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  rw [hwa, hwb, hwc, hwd] at h18_9
  rw [show Int.toNat 3 = 3 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_parallel_generatedRoot_831_18 (count := 3) (by decide) h18_9
  have hA0 : w_21_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model key Q0 6 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := Q1) (m := 6) (m' := 6) (by norm_num) Backend.layout1
    layoutQ1 rfl rfl (digits := 6) (c := ![c0, c1]) (A := ![![A00, A01], ![A10, A11]]) (index := sigma 3)
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (sw := LoopState.switch (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) (Xb := ![w_38_0, w_80_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h80_0])
    (D' := ![w_39_0, w_81_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h81_0])
    (D := ![w_40_0, w_82_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h82_0])
    (M := ![![w_42_0, w_84_0], ![w_44_0, w_86_0]]) (fun j ↦ by fin_cases j; exacts [h42_0, h84_0]) (fun j ↦ by fin_cases j; exacts [h44_0, h86_0])
    (Asw := ![![w_43_0, w_85_0], ![w_45_0, w_87_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h85_0, h45_0, h87_0])
    (Z := ![w_46_0, w_88_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h88_0])
    (E' := ![w_47_0, w_89_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h89_0])
    (E := ![w_48_0, w_90_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h90_0])
    (h := ![w_19_0, w_24_0]) (Ad := lutA model key Q0 6 2 (sigma (3 : Fin 4)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_92_0], ![w_97_0, w_102_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h92_0]) (fun j ↦ by fin_cases j; exacts [h97_0, h102_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_93_0, w_98_0, w_103_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := Q1) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model key 2 sigma ![![A00, A01], ![A10, A11]] ![![fam00, fam01], ![fam10, fam11]] i) := by
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by norm_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_92_0], ![w_97_0, w_102_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  simp only [concatColumns] at h54_0 h55_0 h56_0 h57_0 h58_0 h59_0 h60_0 h61_0 h62_0 h63_0 h64_0 h65_0 h66_0 h67_0 h68_0 h69_0 h70_0 h71_0 h73_0
  have hrow : ∀ c : Fin 20, w_73_0 0 c = (matrixPolynomial [smudgeWeight831 c] : ExactMatrix Q1 N 1 1) 0 0 := by
    intro c
    fin_cases c <;> simp [h54_0, h55_0, h56_0, h57_0, h58_0, h59_0, h60_0, h61_0, h62_0, h63_0, h64_0, h65_0, h66_0, h67_0, h68_0, h69_0, h70_0, h71_0, h73_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_58_0, w_59_0, w_60_0, w_61_0, w_62_0, w_63_0, w_64_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_70_0, w_71_0, w_72_0, smudgeWeight831]
  have hd : Int.fmod (Int.ofNat o) 20 = ((o % 20 : Nat) : Int) := by
    rw [Int.fmod_eq_emod_of_nonneg _ (by norm_num)]; rfl
  rw [hd] at h74_0
  have hK : w_74_0 = matrixPolynomial [smudgeWeight831 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] := by
    funext a b
    obtain rfl : a = 0 := Subsingleton.elim _ _
    obtain rfl : b = 0 := Subsingleton.elim _ _
    rw [slice_entry h74_0 (by simp; omega)]
    exact hrow _
  have hW : w_76_0 = dec (inner := 6) 6 (matrixPolynomial [smudgeWeight831 ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩] :
      ExactMatrix Q1 N 1 1) := by
    rw [rebase_self (by norm_num) h76_0, ← hK]
    exact dec_eq h75_0
  rw [← hS, hout]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · show multiplyMonomial (matrixMul w_52_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_94_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h94_0 (blockModSwitchRuns_fn h94_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_99_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h99_0 (blockModSwitchRuns_fn h99_0 _)]
    rfl
  · show multiplyMonomial (matrixMul w_104_0 w_76_0) w_78_0 = _
    rw [hW, blockModSwitchRuns_unique h104_0 (blockModSwitchRuns_fn h104_0 _)]
    rfl

end RefreshTopProof.Part_0_0
