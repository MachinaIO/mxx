import TopInst
import Part_I_KeysLuts

/-! The keys of the online root's LUTs (written by `generator/gen_keys_root.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the root LUT `18`. -/
theorem keyRootLut18 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params}
    {w_1_0 : Fin 2 → ExactMatrix (R i0) 8 1 4}
    {w_3_0 : Fin 2 → ExactMatrix (R i0) 8 1 4}
    {w_5_0 : Fin 2 → ExactMatrix (R i0) 8 1 4}
    {w_7_0 : Fin 2 → ExactMatrix (R i0) 8 1 4}
    {w_9_0 : Fin 2 → ExactMatrix Q0 8 1 6}
    {w_10_0 : ByteArray}
    {w_11_0 : Fin 2 → ExactMatrix Q0 8 1 6}
    {w_12_0 : ExactMatrix Q0 8 1 6}
    {w_15_0 : ExactMatrix Q0 8 1 6}
    {w_0_0 : ExactMatrix Q0 8 1 4}
    {w_2_0 : ExactMatrix (R i0) 8 1 4}
    {w_4_0 : ExactMatrix (R i0) 8 1 4}
    {w_6_0 : ExactMatrix (R i0) 8 1 4}
    {w_8_0 : ExactMatrix (R i0) 8 1 4}
    {w_13_0 : ExactMatrix Q0 8 1 6}
    {w_14_0 : ExactMatrix Q0 8 1 6}
    {w_16_0 : ExactMatrix Q0 8 1 6}
    {w_17_0 : ExactMatrix Q0 8 1 6}
    {w_18_0 : ExactMatrix Q0 8 1 4}
    {w_18_1 : ExactMatrix Q0 8 1 4}
    {w_18_2 : ExactMatrix Q0 8 1 4}
    {w_18_3 : ExactMatrix Q0 8 1 4}
    {w_18_4 : ExactMatrix Q0 8 1 4}
    {w_18_5 : ExactMatrix Q0 8 1 4}
    {w_18_6 : ExactMatrix (R i0) 8 1 4}
    {w_18_7 : ExactMatrix (R i0) 8 1 4}
    {w_18_8 : ExactMatrix (R i0) 8 1 4}
    {w_18_9 : ExactMatrix (R i0) 8 1 4}
    {w_19_0 : ExactMatrix Q0 8 1 6}
    {w_20_0 : Int}
    {w_21_0 : ExactMatrix Q0 8 1 6}
    {w_22_0 : ExactMatrix Q0 8 1 6}
    {w_23_0 : ExactMatrix Q0 8 3 6}
    {w_23_concat_1 : ExactMatrix Q0 8 2 6}
    {w_24_0 : ExactMatrix Q0 8 1 6}
    {w_25_0 : ExactMatrix Q0 8 1 6}
    {w_26_0 : ExactMatrix Q0 8 1 6}
    {w_27_0 : ExactMatrix Q0 8 3 6}
    {w_27_concat_1 : ExactMatrix Q0 8 2 6}
    {w_28_0 : ExactMatrix Q0 8 3 12}
    {w_29_0 : ExactMatrix Q0 8 1 6}
    {w_30_0 : ExactMatrix Q0 8 1 6}
    {w_31_0 : ExactMatrix Q0 8 1 6}
    {w_32_0 : ExactMatrix Q0 8 3 6}
    {w_32_concat_1 : ExactMatrix Q0 8 2 6}
    {w_33_0 : ExactMatrix Q0 8 1 6}
    {w_34_0 : ExactMatrix Q0 8 1 6}
    {w_35_0 : ExactMatrix Q0 8 1 6}
    {w_36_0 : ExactMatrix Q0 8 3 6}
    {w_36_concat_1 : ExactMatrix Q0 8 2 6}
    {w_37_0 : ExactMatrix Q0 8 3 12}
    {w_38_0 : ExactMatrix (R i0) 8 2 4}
    {w_39_0 : ExactMatrix (R i0) 8 12 4}
    {w_40_0 : ExactMatrix Q0 8 12 4}
    {w_41_0 : ExactMatrix Q0 8 3 4}
    {w_42_0 : ExactMatrix Q0 8 1 4}
    {w_43_0 : ExactMatrix (R i0) 8 1 4}
    {w_44_0 : ExactMatrix Q0 8 1 4}
    {w_45_0 : ExactMatrix (R i0) 8 1 4}
    {w_46_0 : ExactMatrix (R i0) 8 2 4}
    {w_47_0 : ExactMatrix (R i0) 8 12 4}
    {w_48_0 : ExactMatrix Q0 8 12 4}
    {w_49_0 : ExactMatrix Q0 8 3 4}
    {w_50_0 : ExactMatrix Q0 8 1 4}
    {w_51_0 : ExactMatrix Q0 8 1 4}
    {w_52_0 : ExactMatrix (R i0) 8 1 4}
    {w_53_0 : ExactMatrix (R i0) 8 2 4}
    {w_54_0 : ExactMatrix (R i0) 8 12 4}
    {w_55_0 : ExactMatrix Q0 8 12 4}
    {w_56_0 : ExactMatrix Q0 8 3 4}
    {w_57_0 : ExactMatrix Q0 8 1 4}
    {w_58_0 : ExactMatrix (R i0) 8 1 4}
    {w_59_0 : ExactMatrix Q0 8 1 4}
    {w_60_0 : ExactMatrix (R i0) 8 1 4}
    {w_61_0 : ExactMatrix (R i0) 8 2 4}
    {w_62_0 : ExactMatrix (R i0) 8 12 4}
    {w_63_0 : ExactMatrix Q0 8 12 4}
    {w_64_0 : ExactMatrix Q0 8 3 4}
    {w_65_0 : ExactMatrix Q0 8 1 4}
    {w_66_0 : ExactMatrix Q0 8 1 4}
    {w_67_0 : ExactMatrix (R i0) 8 1 4}
    {w_68_0 : ExactMatrix Q0 8 1 4}
    {w_69_0 : ExactMatrix Q0 8 1 4}
    {w_70_0 : ExactMatrix (R i0) 8 1 4}
    {w_71_0 : ExactMatrix Q0 8 1 4}
    {w_72_0 : ExactMatrix Q0 8 1 4}
    {w_73_0 : ExactMatrix (R i0) 8 1 4}
    {w_74_0 : ExactMatrix Q0 8 1 4}
    {w_75_0 : ExactMatrix Q0 8 1 4}
    {w_76_0 : ExactMatrix Q0 8 1 4}
    {w_77_0 : ExactMatrix Q0 8 1 4}
    {w_78_0 : ExactMatrix (R i0) 8 1 4}
    {w_79_0 : ExactMatrix Q0 8 1 4}
    {w_80_0 : ExactMatrix Q0 8 1 4}
    {w_81_0 : ExactMatrix Q0 8 1 4}
    {w_82_0 : ExactMatrix Q0 8 1 4}
    {w_83_0 : ExactMatrix (R i0) 8 1 4}
    (e_0_0 : w_0_0 = (0 : ExactMatrix Q0 8 1 4))
    (e_20_0 : w_20_0 = (MxxIR.selectInt (1) [(1), (9)]))
    (e_41_0 : w_41_0 = matrixMul w_37_0 w_40_0)
    (e_49_0 : w_49_0 = matrixMul w_28_0 w_48_0)
    (e_51_0 : w_51_0 = matrixAdd w_18_2 w_50_0)
    (e_56_0 : w_56_0 = matrixMul w_37_0 w_55_0)
    (e_64_0 : w_64_0 = matrixMul w_28_0 w_63_0)
    (e_66_0 : w_66_0 = matrixAdd w_18_3 w_65_0)
    (e_69_0 : w_69_0 = matrixAdd w_18_4 w_68_0)
    (e_72_0 : w_72_0 = matrixAdd w_18_5 w_71_0)
    (e_76_0 : w_76_0 = matrixAdd w_74_0 w_75_0)
    (e_77_0 : w_77_0 = matrixAdd w_18_0 w_76_0)
    (e_81_0 : w_81_0 = matrixAdd w_79_0 w_80_0)
    (e_82_0 : w_82_0 = matrixAdd w_18_1 w_81_0)
    (h2_0 : familyGetStatic w_1_0 (0) w_2_0)
    (h4_0 : familyGetStatic w_3_0 (0) w_4_0)
    (h6_0 : familyGetStatic w_5_0 (0) w_6_0)
    (h8_0 : familyGetStatic w_7_0 (0) w_8_0)
    (h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix (R i0) 8 1 4 × ExactMatrix (R i0) 8 1 4 × ExactMatrix (R i0) 8 1 4 × ExactMatrix (R i0) 8 1 4 × Unit) => Stage_online_eval.sequential_parallel_generatedRoot_290_18 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (1)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))))
    (h19_0 : familyGetStatic w_9_0 (1) w_19_0)
    (h21_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_21_0)
    (h22_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_22_0)
    (h21_0_1 : concatRows w_19_0 w_21_0 w_23_concat_1)
    (h23_0 : concatRows w_23_concat_1 w_22_0 w_23_0)
    (h24_0 : familyGetStatic w_11_0 (1) w_24_0)
    (h25_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_25_0)
    (h26_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (0), .integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_26_0)
    (h25_0_1 : concatRows w_24_0 w_25_0 w_27_concat_1)
    (h27_0 : concatRows w_27_concat_1 w_26_0 w_27_0)
    (h28_0 : concatColumns w_23_0 w_27_0 w_28_0)
    (h29_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_12_0 w_29_0)
    (h30_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_13_0 w_30_0)
    (h31_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_14_0 w_31_0)
    (h30_0_1 : concatRows w_29_0 w_30_0 w_32_concat_1)
    (h32_0 : concatRows w_32_concat_1 w_31_0 w_32_0)
    (h33_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_15_0 w_33_0)
    (h34_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_16_0 w_34_0)
    (h35_0 : ringAutomorphismRuns ((MxxIR.selectInt (1) [(1), (9)])) w_17_0 w_35_0)
    (h34_0_1 : concatRows w_33_0 w_34_0 w_36_concat_1)
    (h36_0 : concatRows w_36_concat_1 w_35_0 w_36_0)
    (h37_0 : concatColumns w_32_0 w_36_0 w_37_0)
    (h38_0 : concatRows w_18_6 w_18_8 w_38_0)
    (h39_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_38_0 w_39_0)
    (h40_0 : compactCenteredRebaseRuns 16384 w_39_0 w_40_0)
    (h42_0 : sliceMatrix w_41_0 1 2 0 4 w_42_0)
    (h43_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_42_0 w_43_0)
    (h44_0 : sliceMatrix w_41_0 2 3 0 4 w_44_0)
    (h45_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_44_0 w_45_0)
    (h46_0 : concatRows w_43_0 w_45_0 w_46_0)
    (h47_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_46_0 w_47_0)
    (h48_0 : compactCenteredRebaseRuns 16384 w_47_0 w_48_0)
    (h50_0 : sliceMatrix w_49_0 1 2 0 4 w_50_0)
    (h52_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_51_0 w_52_0)
    (h53_0 : concatRows w_18_7 w_18_9 w_53_0)
    (h54_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_53_0 w_54_0)
    (h55_0 : compactCenteredRebaseRuns 16384 w_54_0 w_55_0)
    (h57_0 : sliceMatrix w_56_0 1 2 0 4 w_57_0)
    (h58_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_57_0 w_58_0)
    (h59_0 : sliceMatrix w_56_0 2 3 0 4 w_59_0)
    (h60_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_59_0 w_60_0)
    (h61_0 : concatRows w_58_0 w_60_0 w_61_0)
    (h62_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_61_0 w_62_0)
    (h63_0 : compactCenteredRebaseRuns 16384 w_62_0 w_63_0)
    (h65_0 : sliceMatrix w_64_0 1 2 0 4 w_65_0)
    (h67_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_66_0 w_67_0)
    (h68_0 : sliceMatrix w_49_0 2 3 0 4 w_68_0)
    (h70_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_69_0 w_70_0)
    (h71_0 : sliceMatrix w_64_0 2 3 0 4 w_71_0)
    (h73_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_72_0 w_73_0)
    (h74_0 : sliceMatrix w_49_0 0 1 0 4 w_74_0)
    (h75_0 : sliceMatrix w_41_0 0 1 0 4 w_75_0)
    (h78_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_77_0 w_78_0)
    (h79_0 : sliceMatrix w_64_0 0 1 0 4 w_79_0)
    (h80_0 : sliceMatrix w_56_0 0 1 0 4 w_80_0)
    (h83_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_82_0 w_83_0) :
    (w_52_0, w_67_0, w_70_0, w_73_0, ()) = keyMap (blockModSwitch (p := (R i0)) [q0, q1, q2, q3] 1) (∑ i ∈ Finset.range 2, lutKeys (q := Q0) (q' := (R i0)) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model w_10_0 0 sigma2 ![![w_13_0, w_16_0], ![w_14_0, w_17_0]] ![![w_1_0, w_3_0], ![w_5_0, w_7_0]] i) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  subst e_0_0 e_20_0 e_41_0 e_49_0 e_56_0 e_64_0 e_76_0 e_77_0 e_81_0 e_82_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 1 = 1 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_generatedRoot_18 (count := 1) (by decide) h18_9
  have hA0 : w_21_0 = lutA model w_10_0 Q0 6 0 (sigma2 (1 : Fin 2)) 0 0 :=
    hashSample_eq h21_0
  have hA1 : w_25_0 = lutA model w_10_0 Q0 6 0 (sigma2 (1 : Fin 2)) 0 1 :=
    hashSample_eq h25_0
  have hA2 : w_22_0 = lutA model w_10_0 Q0 6 0 (sigma2 (1 : Fin 2)) 1 0 :=
    hashSample_eq h22_0
  have hA3 : w_26_0 = lutA model w_10_0 Q0 6 0 (sigma2 (1 : Fin 2)) 1 1 :=
    hashSample_eq h26_0
  rw [hA0] at h21_0_1
  rw [hA2] at h23_0
  rw [hA1] at h25_0_1
  rw [hA3] at h27_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := (R i0)) (m := 6) (m' := 4) (by inst_num) L hL hLx (by rw [hLd]) (digits := 6) (c := ![w_12_0, w_15_0]) (A := ![![w_13_0, w_16_0], ![w_14_0, w_17_0]]) (index := sigma2 1)
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
    (h := ![w_19_0, w_24_0]) (Ad := lutA model w_10_0 Q0 6 0 (sigma2 (1 : Fin 2)))
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]]) (fun j ↦ by fin_cases j; exacts [h50_0, h65_0]) (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
  rw [hloopSwitch] at hK4
  have hS : (w_51_0, w_66_0, w_69_0, w_72_0, ()) = (∑ i ∈ Finset.range 2, lutKeys (q := Q0) (q' := (R i0)) (m := 6) (m' := 4) 6 [q0, q1, q2, q3] model w_10_0 0 sigma2 ![![w_13_0, w_16_0], ![w_14_0, w_17_0]] ![![w_1_0, w_3_0], ![w_5_0, w_7_0]] i) := by
    rw [e_51_0, e_66_0, e_69_0, e_72_0]
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by inst_num)]
    refine (show _ = keyTuple (LoopState.key (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) +
      keyTuple ![![w_50_0, w_65_0], ![w_68_0, w_71_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h52_0 (blockModSwitchRuns_fn h52_0 _)
  · exact blockModSwitchRuns_unique h67_0 (blockModSwitchRuns_fn h67_0 _)
  · exact blockModSwitchRuns_unique h70_0 (blockModSwitchRuns_fn h70_0 _)
  · exact blockModSwitchRuns_unique h73_0 (blockModSwitchRuns_fn h73_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the root LUT `186`. -/
theorem keyRootLut186 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params}
    {w_92_0 : ByteArray}
    {w_94_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_95_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_96_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_97_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_98_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_99_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_127_0 : ExactMatrix Q5 8 1 6}
    {w_128_0 : ExactMatrix Q0 8 1 6}
    {w_129_0 : ExactMatrix Q0 8 1 6}
    {w_130_0 : ExactMatrix Q0 8 1 6}
    {w_131_0 : ExactMatrix Q0 8 1 6}
    {w_169_0 : ExactMatrix Q5 8 1 8}
    {w_171_0 : ExactMatrix Q5 8 1 8}
    {w_173_0 : ExactMatrix Q5 8 1 8}
    {w_181_0 : ExactMatrix Q5 8 1 8}
    {w_183_0 : ExactMatrix Q5 8 1 8}
    {w_185_0 : ExactMatrix Q5 8 1 8}
    {w_186_0 : ExactMatrix Q5 8 1 6}
    {w_186_1 : ExactMatrix Q5 8 1 6}
    {w_186_2 : ExactMatrix Q5 8 1 6}
    {w_186_3 : ExactMatrix Q5 8 1 6}
    {w_186_4 : ExactMatrix Q5 8 1 6}
    {w_186_5 : ExactMatrix Q5 8 1 6}
    {w_186_6 : ExactMatrix Q0 8 1 6}
    {w_186_7 : ExactMatrix Q0 8 1 6}
    {w_186_8 : ExactMatrix Q0 8 1 6}
    {w_186_9 : ExactMatrix Q0 8 1 6}
    {w_187_0 : ExactMatrix Q5 8 1 8}
    {w_188_0 : Int}
    {w_189_0 : ExactMatrix Q5 8 1 8}
    {w_190_0 : ExactMatrix Q5 8 1 8}
    {w_191_0 : ExactMatrix Q5 8 3 8}
    {w_191_concat_1 : ExactMatrix Q5 8 2 8}
    {w_192_0 : ExactMatrix Q5 8 1 8}
    {w_193_0 : ExactMatrix Q5 8 1 8}
    {w_194_0 : ExactMatrix Q5 8 1 8}
    {w_195_0 : ExactMatrix Q5 8 3 8}
    {w_195_concat_1 : ExactMatrix Q5 8 2 8}
    {w_196_0 : ExactMatrix Q5 8 3 16}
    {w_197_0 : ExactMatrix Q5 8 1 8}
    {w_198_0 : ExactMatrix Q5 8 1 8}
    {w_199_0 : ExactMatrix Q5 8 1 8}
    {w_200_0 : ExactMatrix Q5 8 3 8}
    {w_200_concat_1 : ExactMatrix Q5 8 2 8}
    {w_201_0 : ExactMatrix Q5 8 1 8}
    {w_202_0 : ExactMatrix Q5 8 1 8}
    {w_203_0 : ExactMatrix Q5 8 1 8}
    {w_204_0 : ExactMatrix Q5 8 3 8}
    {w_204_concat_1 : ExactMatrix Q5 8 2 8}
    {w_205_0 : ExactMatrix Q5 8 3 16}
    {w_206_0 : ExactMatrix Q0 8 2 6}
    {w_207_0 : ExactMatrix Q0 8 16 6}
    {w_208_0 : ExactMatrix Q5 8 16 6}
    {w_209_0 : ExactMatrix Q5 8 3 6}
    {w_210_0 : ExactMatrix Q5 8 1 6}
    {w_211_0 : ExactMatrix Q0 8 1 6}
    {w_212_0 : ExactMatrix Q5 8 1 6}
    {w_213_0 : ExactMatrix Q0 8 1 6}
    {w_214_0 : ExactMatrix Q0 8 2 6}
    {w_215_0 : ExactMatrix Q0 8 16 6}
    {w_216_0 : ExactMatrix Q5 8 16 6}
    {w_217_0 : ExactMatrix Q5 8 3 6}
    {w_218_0 : ExactMatrix Q5 8 1 6}
    {w_219_0 : ExactMatrix Q5 8 1 6}
    {w_220_0 : ExactMatrix Q5 8 1 6}
    {w_221_0 : ExactMatrix Q5 8 1 6}
    {w_222_0 : ExactMatrix Q0 8 1 6}
    {w_223_0 : ExactMatrix Q5 8 1 6}
    {w_224_0 : ExactMatrix Q5 8 1 6}
    {w_225_0 : ExactMatrix Q0 8 1 6}
    {w_226_0 : ExactMatrix Q5 8 1 6}
    {w_227_0 : ExactMatrix Q5 8 1 6}
    {w_228_0 : ExactMatrix Q0 8 1 6}
    {w_229_0 : ExactMatrix Q0 8 2 6}
    {w_230_0 : ExactMatrix Q0 8 16 6}
    {w_231_0 : ExactMatrix Q5 8 16 6}
    {w_232_0 : ExactMatrix Q5 8 3 6}
    {w_233_0 : ExactMatrix Q5 8 1 6}
    {w_234_0 : ExactMatrix Q0 8 1 6}
    {w_235_0 : ExactMatrix Q5 8 1 6}
    {w_236_0 : ExactMatrix Q0 8 1 6}
    {w_237_0 : ExactMatrix Q0 8 2 6}
    {w_238_0 : ExactMatrix Q0 8 16 6}
    {w_239_0 : ExactMatrix Q5 8 16 6}
    {w_240_0 : ExactMatrix Q5 8 3 6}
    {w_241_0 : ExactMatrix Q5 8 1 6}
    {w_242_0 : ExactMatrix Q5 8 1 6}
    {w_243_0 : ExactMatrix Q5 8 1 6}
    {w_244_0 : ExactMatrix Q5 8 1 6}
    {w_245_0 : ExactMatrix Q0 8 1 6}
    {w_246_0 : ExactMatrix Q5 8 1 6}
    {w_247_0 : ExactMatrix Q5 8 1 6}
    {w_248_0 : ExactMatrix Q0 8 1 6}
    {w_249_0 : ExactMatrix Q5 8 1 6}
    {w_250_0 : ExactMatrix Q5 8 1 6}
    {w_251_0 : ExactMatrix Q0 8 1 6}
    (e_127_0 : w_127_0 = (0 : ExactMatrix Q5 8 1 6))
    (e_188_0 : w_188_0 = (MxxIR.selectInt (3) [(1), (3), (5), (15)]))
    (e_209_0 : w_209_0 = matrixMul w_205_0 w_208_0)
    (e_217_0 : w_217_0 = matrixMul w_196_0 w_216_0)
    (e_220_0 : w_220_0 = matrixAdd w_218_0 w_219_0)
    (e_221_0 : w_221_0 = matrixAdd w_186_0 w_220_0)
    (e_224_0 : w_224_0 = matrixAdd w_186_2 w_223_0)
    (e_227_0 : w_227_0 = matrixAdd w_186_4 w_226_0)
    (e_232_0 : w_232_0 = matrixMul w_205_0 w_231_0)
    (e_240_0 : w_240_0 = matrixMul w_196_0 w_239_0)
    (e_243_0 : w_243_0 = matrixAdd w_241_0 w_242_0)
    (e_244_0 : w_244_0 = matrixAdd w_186_1 w_243_0)
    (e_247_0 : w_247_0 = matrixAdd w_186_3 w_246_0)
    (e_250_0 : w_250_0 = matrixAdd w_186_5 w_249_0)
    (h128_0 : familyGetStatic w_94_0 (0) w_128_0)
    (h129_0 : familyGetStatic w_95_0 (0) w_129_0)
    (h130_0 : familyGetStatic w_96_0 (0) w_130_0)
    (h131_0 : familyGetStatic w_97_0 (0) w_131_0)
    (h186_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval.sequential_parallel_generatedRoot_290_186 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_98_0, (w_92_0, (w_99_0, (w_169_0, (w_171_0, (w_173_0, (w_181_0, (w_183_0, (w_185_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, ())))))))))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_186_4, (w_186_5, (w_186_6, (w_186_7, (w_186_8, (w_186_9, ())))))))))))
    (h187_0 : familyGetStatic w_98_0 (3) w_187_0)
    (h189_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_189_0)
    (h190_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_190_0)
    (h189_0_1 : concatRows w_187_0 w_189_0 w_191_concat_1)
    (h191_0 : concatRows w_191_concat_1 w_190_0 w_191_0)
    (h192_0 : familyGetStatic w_99_0 (3) w_192_0)
    (h193_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_193_0)
    (h194_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_194_0)
    (h193_0_1 : concatRows w_192_0 w_193_0 w_195_concat_1)
    (h195_0 : concatRows w_195_concat_1 w_194_0 w_195_0)
    (h196_0 : concatColumns w_191_0 w_195_0 w_196_0)
    (h197_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_169_0 w_197_0)
    (h198_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_171_0 w_198_0)
    (h199_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_173_0 w_199_0)
    (h198_0_1 : concatRows w_197_0 w_198_0 w_200_concat_1)
    (h200_0 : concatRows w_200_concat_1 w_199_0 w_200_0)
    (h201_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_181_0 w_201_0)
    (h202_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_183_0 w_202_0)
    (h203_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_185_0 w_203_0)
    (h202_0_1 : concatRows w_201_0 w_202_0 w_204_concat_1)
    (h204_0 : concatRows w_204_concat_1 w_203_0 w_204_0)
    (h205_0 : concatColumns w_200_0 w_204_0 w_205_0)
    (h206_0 : concatRows w_186_6 w_186_8 w_206_0)
    (h207_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_206_0 w_207_0)
    (h208_0 : compactCenteredRebaseRuns 16384 w_207_0 w_208_0)
    (h210_0 : sliceMatrix w_209_0 1 2 0 6 w_210_0)
    (h211_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_210_0 w_211_0)
    (h212_0 : sliceMatrix w_209_0 2 3 0 6 w_212_0)
    (h213_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_212_0 w_213_0)
    (h214_0 : concatRows w_211_0 w_213_0 w_214_0)
    (h215_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_214_0 w_215_0)
    (h216_0 : compactCenteredRebaseRuns 16384 w_215_0 w_216_0)
    (h218_0 : sliceMatrix w_217_0 0 1 0 6 w_218_0)
    (h219_0 : sliceMatrix w_209_0 0 1 0 6 w_219_0)
    (h222_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_221_0 w_222_0)
    (h223_0 : sliceMatrix w_217_0 1 2 0 6 w_223_0)
    (h225_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_224_0 w_225_0)
    (h226_0 : sliceMatrix w_217_0 2 3 0 6 w_226_0)
    (h228_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_227_0 w_228_0)
    (h229_0 : concatRows w_186_7 w_186_9 w_229_0)
    (h230_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_229_0 w_230_0)
    (h231_0 : compactCenteredRebaseRuns 16384 w_230_0 w_231_0)
    (h233_0 : sliceMatrix w_232_0 1 2 0 6 w_233_0)
    (h234_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_233_0 w_234_0)
    (h235_0 : sliceMatrix w_232_0 2 3 0 6 w_235_0)
    (h236_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_235_0 w_236_0)
    (h237_0 : concatRows w_234_0 w_236_0 w_237_0)
    (h238_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_237_0 w_238_0)
    (h239_0 : compactCenteredRebaseRuns 16384 w_238_0 w_239_0)
    (h241_0 : sliceMatrix w_240_0 0 1 0 6 w_241_0)
    (h242_0 : sliceMatrix w_232_0 0 1 0 6 w_242_0)
    (h245_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_244_0 w_245_0)
    (h246_0 : sliceMatrix w_240_0 1 2 0 6 w_246_0)
    (h248_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_247_0 w_248_0)
    (h249_0 : sliceMatrix w_240_0 2 3 0 6 w_249_0)
    (h251_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_250_0 w_251_0) :
    (w_225_0, w_248_0, w_228_0, w_251_0, ()) = keyMap (blockModSwitch (p := Q0) [q0, q1, q2, q3, p4] 1) (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model w_92_0 2 sigma ![![w_171_0, w_183_0], ![w_173_0, w_185_0]] ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] i) := by
  subst e_127_0 e_188_0 e_209_0 e_217_0 e_220_0 e_221_0 e_232_0 e_240_0 e_243_0 e_244_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h128_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h129_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h130_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h131_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 3 = 3 from rfl] at h186_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_generatedRoot_186 (count := 3) (by decide) h186_9
  have hA0 : w_189_0 = lutA model w_92_0 Q5 8 2 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h189_0
  have hA1 : w_193_0 = lutA model w_92_0 Q5 8 2 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h193_0
  have hA2 : w_190_0 = lutA model w_92_0 Q5 8 2 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h190_0
  have hA3 : w_194_0 = lutA model w_92_0 Q5 8 2 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h194_0
  rw [hA0] at h189_0_1
  rw [hA2] at h191_0
  rw [hA1] at h193_0_1
  rw [hA3] at h195_0
  have hK4 := branchKey_of_stacks (q := Q5) (q' := Q0) (m := 8) (m' := 6) (by inst_num) Backend.layout5
    layoutQ0 rfl rfl (digits := 8) (c := ![w_169_0, w_181_0]) (A := ![![w_171_0, w_183_0], ![w_173_0, w_185_0]]) (index := sigma 3)
    h197_0 h198_0 h199_0 h201_0 h202_0 h203_0 h198_0_1 h200_0 h202_0_1 h204_0 h205_0
    (sw := LoopState.switch (w_186_0, w_186_1, w_186_2, w_186_3, w_186_4, w_186_5, w_186_6, w_186_7, w_186_8, w_186_9, ())) (Xb := ![w_206_0, w_229_0]) (fun j ↦ by fin_cases j; exacts [h206_0, h229_0])
    (D' := ![w_207_0, w_230_0]) (fun j ↦ by fin_cases j; exacts [h207_0, h230_0])
    (D := ![w_208_0, w_231_0]) (fun j ↦ by fin_cases j; exacts [h208_0, h231_0])
    (M := ![![w_210_0, w_233_0], ![w_212_0, w_235_0]]) (fun j ↦ by fin_cases j; exacts [h210_0, h233_0]) (fun j ↦ by fin_cases j; exacts [h212_0, h235_0])
    (Asw := ![![w_211_0, w_234_0], ![w_213_0, w_236_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h211_0, h234_0, h213_0, h236_0])
    (Z := ![w_214_0, w_237_0]) (fun j ↦ by fin_cases j; exacts [h214_0, h237_0])
    (E' := ![w_215_0, w_238_0]) (fun j ↦ by fin_cases j; exacts [h215_0, h238_0])
    (E := ![w_216_0, w_239_0]) (fun j ↦ by fin_cases j; exacts [h216_0, h239_0])
    (h := ![w_187_0, w_192_0]) (Ad := lutA model w_92_0 Q5 8 2 (sigma (3 : Fin 4)))
    h189_0_1 h191_0 h193_0_1 h195_0 h196_0
    (K := ![![w_223_0, w_246_0], ![w_226_0, w_249_0]]) (fun j ↦ by fin_cases j; exacts [h223_0, h246_0]) (fun j ↦ by fin_cases j; exacts [h226_0, h249_0])
  rw [hloopSwitch] at hK4
  have hS : (w_224_0, w_247_0, w_227_0, w_250_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q5) (q' := Q0) (m := 8) (m' := 6) 8 [q0, q1, q2, q3, p4] model w_92_0 2 sigma ![![w_171_0, w_183_0], ![w_173_0, w_185_0]] ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] i) := by
    rw [e_224_0, e_247_0, e_227_0, e_250_0]
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by inst_num)]
    refine (show _ = keyTuple (LoopState.key (w_186_0, w_186_1, w_186_2, w_186_3, w_186_4, w_186_5, w_186_6, w_186_7, w_186_8, w_186_9, ())) +
      keyTuple ![![w_223_0, w_246_0], ![w_226_0, w_249_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  rw [← hS]
  simp only [keyMap]
  refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl)))
  · exact blockModSwitchRuns_unique h225_0 (blockModSwitchRuns_fn h225_0 _)
  · exact blockModSwitchRuns_unique h248_0 (blockModSwitchRuns_fn h248_0 _)
  · exact blockModSwitchRuns_unique h228_0 (blockModSwitchRuns_fn h228_0 _)
  · exact blockModSwitchRuns_unique h251_0 (blockModSwitchRuns_fn h251_0 _)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the root LUT `252`. -/
theorem keyRootLut252 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params}
    {w_87_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_88_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_89_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_90_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_91_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_92_0 : ByteArray}
    {w_93_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_122_0 : ExactMatrix Q0 8 1 6}
    {w_123_0 : ExactMatrix (R i0) 8 1 6}
    {w_124_0 : ExactMatrix (R i0) 8 1 6}
    {w_125_0 : ExactMatrix (R i0) 8 1 6}
    {w_126_0 : ExactMatrix (R i0) 8 1 6}
    {w_222_0 : ExactMatrix Q0 8 1 6}
    {w_225_0 : ExactMatrix Q0 8 1 6}
    {w_228_0 : ExactMatrix Q0 8 1 6}
    {w_245_0 : ExactMatrix Q0 8 1 6}
    {w_248_0 : ExactMatrix Q0 8 1 6}
    {w_251_0 : ExactMatrix Q0 8 1 6}
    {w_252_0 : ExactMatrix Q0 8 1 6}
    {w_252_1 : ExactMatrix Q0 8 1 6}
    {w_252_2 : ExactMatrix Q0 8 1 6}
    {w_252_3 : ExactMatrix Q0 8 1 6}
    {w_252_4 : ExactMatrix Q0 8 1 6}
    {w_252_5 : ExactMatrix Q0 8 1 6}
    {w_252_6 : ExactMatrix (R i0) 8 1 6}
    {w_252_7 : ExactMatrix (R i0) 8 1 6}
    {w_252_8 : ExactMatrix (R i0) 8 1 6}
    {w_252_9 : ExactMatrix (R i0) 8 1 6}
    {w_253_0 : ExactMatrix Q0 8 1 6}
    {w_254_0 : Int}
    {w_255_0 : ExactMatrix Q0 8 1 6}
    {w_256_0 : ExactMatrix Q0 8 1 6}
    {w_257_0 : ExactMatrix Q0 8 3 6}
    {w_257_concat_1 : ExactMatrix Q0 8 2 6}
    {w_258_0 : ExactMatrix Q0 8 1 6}
    {w_259_0 : ExactMatrix Q0 8 1 6}
    {w_260_0 : ExactMatrix Q0 8 1 6}
    {w_261_0 : ExactMatrix Q0 8 3 6}
    {w_261_concat_1 : ExactMatrix Q0 8 2 6}
    {w_262_0 : ExactMatrix Q0 8 3 12}
    {w_263_0 : ExactMatrix Q0 8 1 6}
    {w_264_0 : ExactMatrix Q0 8 1 6}
    {w_265_0 : ExactMatrix Q0 8 1 6}
    {w_266_0 : ExactMatrix Q0 8 3 6}
    {w_266_concat_1 : ExactMatrix Q0 8 2 6}
    {w_267_0 : ExactMatrix Q0 8 1 6}
    {w_268_0 : ExactMatrix Q0 8 1 6}
    {w_269_0 : ExactMatrix Q0 8 1 6}
    {w_270_0 : ExactMatrix Q0 8 3 6}
    {w_270_concat_1 : ExactMatrix Q0 8 2 6}
    {w_271_0 : ExactMatrix Q0 8 3 12}
    {w_272_0 : ExactMatrix (R i0) 8 2 6}
    {w_273_0 : ExactMatrix (R i0) 8 12 6}
    {w_274_0 : ExactMatrix Q0 8 12 6}
    {w_275_0 : ExactMatrix Q0 8 3 6}
    {w_276_0 : ExactMatrix Q0 8 1 6}
    {w_277_0 : ExactMatrix (R i0) 8 1 6}
    {w_278_0 : ExactMatrix Q0 8 1 6}
    {w_279_0 : ExactMatrix (R i0) 8 1 6}
    {w_280_0 : ExactMatrix (R i0) 8 2 6}
    {w_281_0 : ExactMatrix (R i0) 8 12 6}
    {w_282_0 : ExactMatrix Q0 8 12 6}
    {w_283_0 : ExactMatrix Q0 8 3 6}
    {w_284_0 : ExactMatrix Q0 8 1 6}
    {w_285_0 : ExactMatrix Q0 8 1 6}
    {w_286_0 : ExactMatrix (R i0) 8 2 6}
    {w_287_0 : ExactMatrix (R i0) 8 12 6}
    {w_288_0 : ExactMatrix Q0 8 12 6}
    {w_289_0 : ExactMatrix Q0 8 3 6}
    {w_290_0 : ExactMatrix Q0 8 1 6}
    {w_291_0 : ExactMatrix (R i0) 8 1 6}
    {w_292_0 : ExactMatrix Q0 8 1 6}
    {w_293_0 : ExactMatrix (R i0) 8 1 6}
    {w_294_0 : ExactMatrix (R i0) 8 2 6}
    {w_295_0 : ExactMatrix (R i0) 8 12 6}
    {w_296_0 : ExactMatrix Q0 8 12 6}
    {w_297_0 : ExactMatrix Q0 8 3 6}
    {w_298_0 : ExactMatrix Q0 8 1 6}
    {w_299_0 : ExactMatrix Q0 8 1 6}
    {w_300_0 : ExactMatrix Q0 8 1 6}
    {w_301_0 : ExactMatrix Q0 8 1 6}
    {w_302_0 : ExactMatrix Q0 8 1 6}
    {w_303_0 : ExactMatrix Q0 8 1 6}
    {w_304_0 : ExactMatrix Q0 8 1 6}
    {w_305_0 : ExactMatrix Q0 8 1 6}
    {w_308_0 : ExactMatrix Q0 8 1 6}
    {w_309_0 : ExactMatrix Q0 8 1 6}
    (e_122_0 : w_122_0 = (0 : ExactMatrix Q0 8 1 6))
    (e_254_0 : w_254_0 = (MxxIR.selectInt (3) [(1), (3), (5), (15)]))
    (e_275_0 : w_275_0 = matrixMul w_271_0 w_274_0)
    (e_283_0 : w_283_0 = matrixMul w_262_0 w_282_0)
    (e_285_0 : w_285_0 = matrixAdd w_252_2 w_284_0)
    (e_289_0 : w_289_0 = matrixMul w_271_0 w_288_0)
    (e_297_0 : w_297_0 = matrixMul w_262_0 w_296_0)
    (e_299_0 : w_299_0 = matrixAdd w_252_3 w_298_0)
    (e_301_0 : w_301_0 = matrixAdd w_252_4 w_300_0)
    (e_303_0 : w_303_0 = matrixAdd w_252_5 w_302_0)
    (h123_0 : familyGetStatic w_87_0 (0) w_123_0)
    (h124_0 : familyGetStatic w_88_0 (0) w_124_0)
    (h125_0 : familyGetStatic w_89_0 (0) w_125_0)
    (h126_0 : familyGetStatic w_90_0 (0) w_126_0)
    (h252_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix (R i0) 8 1 6 × ExactMatrix (R i0) 8 1 6 × ExactMatrix (R i0) 8 1 6 × ExactMatrix (R i0) 8 1 6 × Unit) => Stage_online_eval.sequential_parallel_generatedRoot_290_252 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_91_0, (w_92_0, (w_93_0, (w_222_0, (w_225_0, (w_228_0, (w_245_0, (w_248_0, (w_251_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, ())))))))))) (w_252_0, (w_252_1, (w_252_2, (w_252_3, (w_252_4, (w_252_5, (w_252_6, (w_252_7, (w_252_8, (w_252_9, ())))))))))))
    (h253_0 : familyGetStatic w_91_0 (3) w_253_0)
    (h255_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_255_0)
    (h256_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_256_0)
    (h255_0_1 : concatRows w_253_0 w_255_0 w_257_concat_1)
    (h257_0 : concatRows w_257_concat_1 w_256_0 w_257_0)
    (h258_0 : familyGetStatic w_93_0 (3) w_258_0)
    (h259_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_259_0)
    (h260_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_260_0)
    (h259_0_1 : concatRows w_258_0 w_259_0 w_261_concat_1)
    (h261_0 : concatRows w_261_concat_1 w_260_0 w_261_0)
    (h262_0 : concatColumns w_257_0 w_261_0 w_262_0)
    (h263_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_222_0 w_263_0)
    (h264_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_225_0 w_264_0)
    (h265_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_228_0 w_265_0)
    (h264_0_1 : concatRows w_263_0 w_264_0 w_266_concat_1)
    (h266_0 : concatRows w_266_concat_1 w_265_0 w_266_0)
    (h267_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_245_0 w_267_0)
    (h268_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_248_0 w_268_0)
    (h269_0 : ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_251_0 w_269_0)
    (h268_0_1 : concatRows w_267_0 w_268_0 w_270_concat_1)
    (h270_0 : concatRows w_270_concat_1 w_269_0 w_270_0)
    (h271_0 : concatColumns w_266_0 w_270_0 w_271_0)
    (h272_0 : concatRows w_252_6 w_252_8 w_272_0)
    (h273_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_272_0 w_273_0)
    (h274_0 : compactCenteredRebaseRuns 16384 w_273_0 w_274_0)
    (h276_0 : sliceMatrix w_275_0 1 2 0 6 w_276_0)
    (h277_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_276_0 w_277_0)
    (h278_0 : sliceMatrix w_275_0 2 3 0 6 w_278_0)
    (h279_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_278_0 w_279_0)
    (h280_0 : concatRows w_277_0 w_279_0 w_280_0)
    (h281_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_280_0 w_281_0)
    (h282_0 : compactCenteredRebaseRuns 16384 w_281_0 w_282_0)
    (h284_0 : sliceMatrix w_283_0 1 2 0 6 w_284_0)
    (h286_0 : concatRows w_252_7 w_252_9 w_286_0)
    (h287_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_286_0 w_287_0)
    (h288_0 : compactCenteredRebaseRuns 16384 w_287_0 w_288_0)
    (h290_0 : sliceMatrix w_289_0 1 2 0 6 w_290_0)
    (h291_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_290_0 w_291_0)
    (h292_0 : sliceMatrix w_289_0 2 3 0 6 w_292_0)
    (h293_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_292_0 w_293_0)
    (h294_0 : concatRows w_291_0 w_293_0 w_294_0)
    (h295_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_294_0 w_295_0)
    (h296_0 : compactCenteredRebaseRuns 16384 w_295_0 w_296_0)
    (h298_0 : sliceMatrix w_297_0 1 2 0 6 w_298_0)
    (h300_0 : sliceMatrix w_283_0 2 3 0 6 w_300_0)
    (h302_0 : sliceMatrix w_297_0 2 3 0 6 w_302_0)
    (h304_0 : sliceMatrix w_283_0 0 1 0 6 w_304_0)
    (h305_0 : sliceMatrix w_275_0 0 1 0 6 w_305_0)
    (h308_0 : sliceMatrix w_297_0 0 1 0 6 w_308_0)
    (h309_0 : sliceMatrix w_289_0 0 1 0 6 w_309_0) :
    (w_285_0, w_299_0, w_301_0, w_303_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := (R i0)) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model w_92_0 3 sigma ![![w_225_0, w_248_0], ![w_228_0, w_251_0]] ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] i) := by
  obtain ⟨L, hL, hLx, -, hLd⟩ := layoutR i0 hi0
  subst e_122_0 e_254_0 e_275_0 e_283_0 e_289_0 e_297_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h123_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h124_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h125_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h126_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 3 = 3 from rfl] at h252_9
  obtain ⟨hloop, hloopSwitch⟩ := keyLoop_sequential_generatedRoot_252 (count := 3) (by decide) h252_9
  have hA0 : w_255_0 = lutA model w_92_0 Q0 6 3 (sigma (3 : Fin 4)) 0 0 :=
    hashSample_eq h255_0
  have hA1 : w_259_0 = lutA model w_92_0 Q0 6 3 (sigma (3 : Fin 4)) 0 1 :=
    hashSample_eq h259_0
  have hA2 : w_256_0 = lutA model w_92_0 Q0 6 3 (sigma (3 : Fin 4)) 1 0 :=
    hashSample_eq h256_0
  have hA3 : w_260_0 = lutA model w_92_0 Q0 6 3 (sigma (3 : Fin 4)) 1 1 :=
    hashSample_eq h260_0
  rw [hA0] at h255_0_1
  rw [hA2] at h257_0
  rw [hA1] at h259_0_1
  rw [hA3] at h261_0
  have hK4 := branchKey_of_stacks (q := Q0) (q' := (R i0)) (m := 6) (m' := 6) (by inst_num) L hL hLx (by rw [hLd]) (digits := 6) (c := ![w_222_0, w_245_0]) (A := ![![w_225_0, w_248_0], ![w_228_0, w_251_0]]) (index := sigma 3)
    h263_0 h264_0 h265_0 h267_0 h268_0 h269_0 h264_0_1 h266_0 h268_0_1 h270_0 h271_0
    (sw := LoopState.switch (w_252_0, w_252_1, w_252_2, w_252_3, w_252_4, w_252_5, w_252_6, w_252_7, w_252_8, w_252_9, ())) (Xb := ![w_272_0, w_286_0]) (fun j ↦ by fin_cases j; exacts [h272_0, h286_0])
    (D' := ![w_273_0, w_287_0]) (fun j ↦ by fin_cases j; exacts [h273_0, h287_0])
    (D := ![w_274_0, w_288_0]) (fun j ↦ by fin_cases j; exacts [h274_0, h288_0])
    (M := ![![w_276_0, w_290_0], ![w_278_0, w_292_0]]) (fun j ↦ by fin_cases j; exacts [h276_0, h290_0]) (fun j ↦ by fin_cases j; exacts [h278_0, h292_0])
    (Asw := ![![w_277_0, w_291_0], ![w_279_0, w_293_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h277_0, h291_0, h279_0, h293_0])
    (Z := ![w_280_0, w_294_0]) (fun j ↦ by fin_cases j; exacts [h280_0, h294_0])
    (E' := ![w_281_0, w_295_0]) (fun j ↦ by fin_cases j; exacts [h281_0, h295_0])
    (E := ![w_282_0, w_296_0]) (fun j ↦ by fin_cases j; exacts [h282_0, h296_0])
    (h := ![w_253_0, w_258_0]) (Ad := lutA model w_92_0 Q0 6 3 (sigma (3 : Fin 4)))
    h255_0_1 h257_0 h259_0_1 h261_0 h262_0
    (K := ![![w_284_0, w_298_0], ![w_300_0, w_302_0]]) (fun j ↦ by fin_cases j; exacts [h284_0, h298_0]) (fun j ↦ by fin_cases j; exacts [h300_0, h302_0])
  rw [hloopSwitch] at hK4
  have hS : (w_285_0, w_299_0, w_301_0, w_303_0, ()) = (∑ i ∈ Finset.range 4, lutKeys (q := Q0) (q' := (R i0)) (m := 6) (m' := 6) 6 [q0, q1, q2, q3] model w_92_0 3 sigma ![![w_225_0, w_248_0], ![w_228_0, w_251_0]] ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] i) := by
    rw [e_285_0, e_299_0, e_301_0, e_303_0]
    rw [Finset.sum_range_succ, ← hloop, lutKeys, dif_pos (by inst_num)]
    refine (show _ = keyTuple (LoopState.key (w_252_0, w_252_1, w_252_2, w_252_3, w_252_4, w_252_5, w_252_6, w_252_7, w_252_8, w_252_9, ())) +
      keyTuple ![![w_284_0, w_298_0], ![w_300_0, w_302_0]] from rfl).trans ?_
    rw [hK4]
    rfl
  exact hS

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys of the encoding of one after the two private multiplications by the unit offset RHS. -/
theorem keyRootOffset {model : HashModel}
    {w_106_0 : ExactMatrix Q7 8 1 12}
    {w_107_0 : ExactMatrix Q7 8 1 12}
    {w_108_0 : ExactMatrix Q7 8 1 12}
    {w_110_0 : ExactMatrix Q7 8 1 12}
    {w_111_0 : ExactMatrix Q7 8 1 12}
    {w_112_0 : ExactMatrix Q7 8 1 12}
    {w_136_0 : ExactMatrix Q6 8 1 10}
    {w_149_0 : ExactMatrix Q6 8 1 10}
    {w_163_0 : ExactMatrix Q5 8 1 8}
    {w_175_0 : ExactMatrix Q5 8 1 8}
    {w_132_0 : ExactMatrix Q7 8 3 12}
    {w_132_concat_1 : ExactMatrix Q7 8 2 12}
    {w_133_0 : ExactMatrix Q7 8 3 12}
    {w_133_concat_1 : ExactMatrix Q7 8 2 12}
    {w_134_0 : ExactMatrix Q7 8 3 24}
    {w_135_0 : ExactMatrix Q6 8 1 10}
    {w_137_0 : ExactMatrix Q6 8 2 10}
    {w_138_0 : ExactMatrix Q6 8 24 10}
    {w_139_0 : ExactMatrix Q7 8 24 10}
    {w_140_0 : ExactMatrix Q7 8 3 10}
    {w_142_0 : ExactMatrix Q6 8 1 10}
    {w_143_0 : ExactMatrix Q7 8 1 10}
    {w_144_0 : ExactMatrix Q6 8 1 10}
    {w_145_0 : ExactMatrix Q7 8 1 10}
    {w_146_0 : ExactMatrix Q6 8 1 10}
    {w_147_0 : ExactMatrix Q6 8 3 10}
    {w_147_concat_1 : ExactMatrix Q6 8 2 10}
    {w_148_0 : ExactMatrix Q6 8 1 10}
    {w_150_0 : ExactMatrix Q6 8 2 10}
    {w_151_0 : ExactMatrix Q6 8 24 10}
    {w_152_0 : ExactMatrix Q7 8 24 10}
    {w_153_0 : ExactMatrix Q7 8 3 10}
    {w_155_0 : ExactMatrix Q6 8 1 10}
    {w_156_0 : ExactMatrix Q7 8 1 10}
    {w_157_0 : ExactMatrix Q6 8 1 10}
    {w_158_0 : ExactMatrix Q7 8 1 10}
    {w_159_0 : ExactMatrix Q6 8 1 10}
    {w_160_0 : ExactMatrix Q6 8 3 10}
    {w_160_concat_1 : ExactMatrix Q6 8 2 10}
    {w_161_0 : ExactMatrix Q6 8 3 20}
    {w_162_0 : ExactMatrix Q5 8 1 8}
    {w_164_0 : ExactMatrix Q5 8 2 8}
    {w_165_0 : ExactMatrix Q5 8 20 8}
    {w_166_0 : ExactMatrix Q6 8 20 8}
    {w_167_0 : ExactMatrix Q6 8 3 8}
    {w_170_0 : ExactMatrix Q6 8 1 8}
    {w_171_0 : ExactMatrix Q5 8 1 8}
    {w_172_0 : ExactMatrix Q6 8 1 8}
    {w_173_0 : ExactMatrix Q5 8 1 8}
    {w_174_0 : ExactMatrix Q5 8 1 8}
    {w_176_0 : ExactMatrix Q5 8 2 8}
    {w_177_0 : ExactMatrix Q5 8 20 8}
    {w_178_0 : ExactMatrix Q6 8 20 8}
    {w_179_0 : ExactMatrix Q6 8 3 8}
    {w_182_0 : ExactMatrix Q6 8 1 8}
    {w_183_0 : ExactMatrix Q5 8 1 8}
    {w_184_0 : ExactMatrix Q6 8 1 8}
    {w_185_0 : ExactMatrix Q5 8 1 8}
    (e_140_0 : w_140_0 = matrixMul w_134_0 w_139_0)
    (e_153_0 : w_153_0 = matrixMul w_134_0 w_152_0)
    (e_167_0 : w_167_0 = matrixMul w_161_0 w_166_0)
    (e_179_0 : w_179_0 = matrixMul w_161_0 w_178_0)
    (h106_0 : concatRows w_108_0 w_106_0 w_132_concat_1)
    (h132_0 : concatRows w_132_concat_1 w_107_0 w_132_0)
    (h110_0 : concatRows w_112_0 w_110_0 w_133_concat_1)
    (h133_0 : concatRows w_133_concat_1 w_111_0 w_133_0)
    (h134_0 : concatColumns w_132_0 w_133_0 w_134_0)
    (h137_0 : concatRows w_135_0 w_136_0 w_137_0)
    (h138_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_137_0 w_138_0)
    (h139_0 : compactCenteredRebaseRuns 16384 w_138_0 w_139_0)
    (h143_0 : sliceMatrix w_140_0 1 2 0 10 w_143_0)
    (h144_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_143_0 w_144_0)
    (h145_0 : sliceMatrix w_140_0 2 3 0 10 w_145_0)
    (h146_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_145_0 w_146_0)
    (h144_0_1 : concatRows w_142_0 w_144_0 w_147_concat_1)
    (h147_0 : concatRows w_147_concat_1 w_146_0 w_147_0)
    (h150_0 : concatRows w_148_0 w_149_0 w_150_0)
    (h151_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_150_0 w_151_0)
    (h152_0 : compactCenteredRebaseRuns 16384 w_151_0 w_152_0)
    (h156_0 : sliceMatrix w_153_0 1 2 0 10 w_156_0)
    (h157_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_156_0 w_157_0)
    (h158_0 : sliceMatrix w_153_0 2 3 0 10 w_158_0)
    (h159_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_158_0 w_159_0)
    (h157_0_1 : concatRows w_155_0 w_157_0 w_160_concat_1)
    (h160_0 : concatRows w_160_concat_1 w_159_0 w_160_0)
    (h161_0 : concatColumns w_147_0 w_160_0 w_161_0)
    (h164_0 : concatRows w_162_0 w_163_0 w_164_0)
    (h165_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_164_0 w_165_0)
    (h166_0 : compactCenteredRebaseRuns 16384 w_165_0 w_166_0)
    (h170_0 : sliceMatrix w_167_0 1 2 0 8 w_170_0)
    (h171_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_170_0 w_171_0)
    (h172_0 : sliceMatrix w_167_0 2 3 0 8 w_172_0)
    (h173_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_172_0 w_173_0)
    (h176_0 : concatRows w_174_0 w_175_0 w_176_0)
    (h177_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_176_0 w_177_0)
    (h178_0 : compactCenteredRebaseRuns 16384 w_177_0 w_178_0)
    (h182_0 : sliceMatrix w_179_0 1 2 0 8 w_182_0)
    (h183_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_182_0 w_183_0)
    (h184_0 : sliceMatrix w_179_0 2 3 0 8 w_184_0)
    (h185_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_184_0 w_185_0) :
    ![![w_171_0, w_183_0], ![w_173_0, w_185_0]] = privKey (q := Q6) (q' := Q5) 10 [q0, q1, q2, q3, p4, p5]
      (privKey (q := Q7) (q' := Q6) 12 [q0, q1, q2, q3, p4, p5, p6] ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
        ![w_135_0, w_148_0] ![w_136_0, w_149_0])
      ![w_162_0, w_174_0] ![w_163_0, w_175_0] := by
  subst e_140_0 e_153_0 e_167_0 e_179_0
  have h1 := privKey_of_stacks (q := Q7) (q' := Q6) (m := 12) (m' := 10) (by norm_num) Backend.layout7 layoutQ6 rfl rfl
    (digits := 12) (c := ![w_108_0, w_112_0]) (A := ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]) h106_0 h132_0 h110_0 h133_0 h134_0
    (top := ![w_135_0, w_148_0]) (bot := ![w_136_0, w_149_0])
    (Xb := ![w_137_0, w_150_0]) (fun j ↦ by fin_cases j; exacts [h137_0, h150_0]) (D' := ![w_138_0, w_151_0]) (fun j ↦ by fin_cases j; exacts [h138_0, h151_0]) (D := ![w_139_0, w_152_0]) (fun j ↦ by fin_cases j; exacts [h139_0, h152_0])
    (M := ![![w_143_0, w_156_0], ![w_145_0, w_158_0]])
    (fun j ↦ by fin_cases j; exacts [h143_0, h156_0]) (fun j ↦ by fin_cases j; exacts [h145_0, h158_0]) (A' := ![![w_144_0, w_157_0], ![w_146_0, w_159_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h144_0, h157_0, h146_0, h159_0])
  have h2 := privKey_of_stacks (q := Q6) (q' := Q5) (m := 10) (m' := 8) (by norm_num) Backend.layout6 layoutQ5 rfl rfl
    (digits := 10) (c := ![w_142_0, w_155_0]) (A := ![![w_144_0, w_157_0], ![w_146_0, w_159_0]]) h144_0_1 h147_0 h157_0_1 h160_0 h161_0
    (top := ![w_162_0, w_174_0]) (bot := ![w_163_0, w_175_0])
    (Xb := ![w_164_0, w_176_0]) (fun j ↦ by fin_cases j; exacts [h164_0, h176_0]) (D' := ![w_165_0, w_177_0]) (fun j ↦ by fin_cases j; exacts [h165_0, h177_0]) (D := ![w_166_0, w_178_0]) (fun j ↦ by fin_cases j; exacts [h166_0, h178_0])
    (M := ![![w_170_0, w_182_0], ![w_172_0, w_184_0]])
    (fun j ↦ by fin_cases j; exacts [h170_0, h182_0]) (fun j ↦ by fin_cases j; exacts [h172_0, h184_0]) (A' := ![![w_171_0, w_183_0], ![w_173_0, w_185_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h171_0, h183_0, h173_0, h185_0])
  rw [h2, h1]

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The keys the preprocessing computes for the encoding of one after the two private
multiplications. -/
theorem preKeyOffset {model : HashModel}
    {w_27_0 : ExactMatrix Q7 8 1 12}
    {w_28_0 : ExactMatrix Q7 8 1 12}
    {w_29_0 : ExactMatrix Q7 8 1 12}
    {w_30_0 : ExactMatrix Q7 8 1 12}
    {w_46_0 : ExactMatrix Q7 8 2 12}
    {w_47_0 : ExactMatrix Q6 8 1 10}
    {w_48_0 : ExactMatrix Q6 8 12 10}
    {w_49_0 : ExactMatrix Q7 8 12 10}
    {w_50_0 : ExactMatrix Q7 8 2 10}
    {w_51_0 : ExactMatrix Q7 8 2 12}
    {w_52_0 : ExactMatrix Q6 8 1 1}
    {w_54_0 : ExactMatrix Q6 8 1 10}
    {w_55_0 : ExactMatrix Q6 8 1 10}
    {w_56_0 : ExactMatrix Q6 8 1 1}
    {w_57_0 : ExactMatrix Q6 8 1 10}
    {w_62_0 : ExactMatrix Q6 8 1 1}
    {w_63_0 : ExactMatrix Q6 8 1 10}
    {w_64_0 : ExactMatrix Q6 8 1 10}
    {w_65_0 : ExactMatrix Q6 8 1 10}
    {w_66_0 : ExactMatrix Q6 8 1 10}
    {w_67_0 : ExactMatrix Q6 8 1 10}
    {w_68_0 : ExactMatrix Q6 8 12 10}
    {w_69_0 : ExactMatrix Q7 8 12 10}
    {w_70_0 : ExactMatrix Q7 8 2 10}
    {w_71_0 : ExactMatrix Q7 8 2 10}
    {w_72_0 : ExactMatrix Q7 8 1 10}
    {w_73_0 : ExactMatrix Q6 8 1 10}
    {w_74_0 : ExactMatrix Q7 8 1 10}
    {w_75_0 : ExactMatrix Q6 8 1 10}
    {w_76_0 : ExactMatrix Q6 8 2 10}
    {w_77_0 : ExactMatrix Q5 8 1 8}
    {w_78_0 : ExactMatrix Q5 8 10 8}
    {w_79_0 : ExactMatrix Q6 8 10 8}
    {w_80_0 : ExactMatrix Q6 8 2 8}
    {w_81_0 : ExactMatrix Q6 8 1 10}
    {w_82_0 : ExactMatrix Q6 8 12 10}
    {w_83_0 : ExactMatrix Q7 8 12 10}
    {w_84_0 : ExactMatrix Q7 8 2 10}
    {w_85_0 : ExactMatrix Q6 8 1 1}
    {w_86_0 : ExactMatrix Q6 8 1 10}
    {w_87_0 : ExactMatrix Q6 8 1 10}
    {w_88_0 : ExactMatrix Q6 8 1 10}
    {w_89_0 : ExactMatrix Q6 8 1 10}
    {w_90_0 : ExactMatrix Q6 8 1 10}
    {w_91_0 : ExactMatrix Q6 8 1 10}
    {w_92_0 : ExactMatrix Q6 8 1 10}
    {w_93_0 : ExactMatrix Q6 8 12 10}
    {w_94_0 : ExactMatrix Q7 8 12 10}
    {w_95_0 : ExactMatrix Q7 8 2 10}
    {w_96_0 : ExactMatrix Q7 8 2 10}
    {w_97_0 : ExactMatrix Q7 8 1 10}
    {w_98_0 : ExactMatrix Q6 8 1 10}
    {w_99_0 : ExactMatrix Q7 8 1 10}
    {w_100_0 : ExactMatrix Q6 8 1 10}
    {w_101_0 : ExactMatrix Q6 8 2 10}
    {w_102_0 : ExactMatrix Q5 8 1 1}
    {w_104_0 : ExactMatrix Q5 8 1 8}
    {w_105_0 : ExactMatrix Q5 8 1 8}
    {w_106_0 : ExactMatrix Q5 8 1 1}
    {w_107_0 : ExactMatrix Q5 8 1 8}
    {w_112_0 : ExactMatrix Q5 8 1 1}
    {w_113_0 : ExactMatrix Q5 8 1 8}
    {w_114_0 : ExactMatrix Q5 8 1 8}
    {w_115_0 : ExactMatrix Q5 8 1 8}
    {w_116_0 : ExactMatrix Q5 8 1 8}
    {w_117_0 : ExactMatrix Q5 8 1 8}
    {w_118_0 : ExactMatrix Q5 8 10 8}
    {w_119_0 : ExactMatrix Q6 8 10 8}
    {w_120_0 : ExactMatrix Q6 8 2 8}
    {w_121_0 : ExactMatrix Q6 8 2 8}
    {w_122_0 : ExactMatrix Q6 8 1 8}
    {w_123_0 : ExactMatrix Q5 8 1 8}
    {w_124_0 : ExactMatrix Q6 8 1 8}
    {w_125_0 : ExactMatrix Q5 8 1 8}
    {w_126_0 : ExactMatrix Q5 8 1 8}
    {w_127_0 : ExactMatrix Q5 8 10 8}
    {w_128_0 : ExactMatrix Q6 8 10 8}
    {w_129_0 : ExactMatrix Q6 8 2 8}
    {w_130_0 : ExactMatrix Q5 8 1 1}
    {w_131_0 : ExactMatrix Q5 8 1 8}
    {w_132_0 : ExactMatrix Q5 8 1 8}
    {w_133_0 : ExactMatrix Q5 8 1 8}
    {w_134_0 : ExactMatrix Q5 8 1 8}
    {w_135_0 : ExactMatrix Q5 8 1 8}
    {w_136_0 : ExactMatrix Q5 8 1 8}
    {w_137_0 : ExactMatrix Q5 8 1 8}
    {w_138_0 : ExactMatrix Q5 8 10 8}
    {w_139_0 : ExactMatrix Q6 8 10 8}
    {w_140_0 : ExactMatrix Q6 8 2 8}
    {w_141_0 : ExactMatrix Q6 8 2 8}
    {w_142_0 : ExactMatrix Q6 8 1 8}
    {w_143_0 : ExactMatrix Q5 8 1 8}
    {w_144_0 : ExactMatrix Q6 8 1 8}
    {w_145_0 : ExactMatrix Q5 8 1 8}
    (e_50_0 : w_50_0 = matrixMul w_46_0 w_49_0)
    (e_55_0 : w_55_0 = matrixMulScalarLeft w_52_0 w_54_0)
    (e_56_0 : w_56_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_57_0 : w_57_0 = matrixMulScalarRight w_55_0 w_56_0)
    (e_63_0 : w_63_0 = matrixMulScalarRight w_57_0 w_62_0)
    (e_64_0 : w_64_0 = matrixMulScalarLeft w_52_0 w_47_0)
    (e_65_0 : w_65_0 = matrixSub w_63_0 w_64_0)
    (e_67_0 : w_67_0 = matrixAdd w_65_0 w_66_0)
    (e_70_0 : w_70_0 = matrixMul w_51_0 w_69_0)
    (e_71_0 : w_71_0 = matrixAdd w_50_0 w_70_0)
    (e_80_0 : w_80_0 = matrixMul w_76_0 w_79_0)
    (e_84_0 : w_84_0 = matrixMul w_46_0 w_83_0)
    (e_86_0 : w_86_0 = matrixMulScalarLeft w_85_0 w_54_0)
    (e_87_0 : w_87_0 = matrixMulScalarRight w_86_0 w_56_0)
    (e_88_0 : w_88_0 = matrixMulScalarRight w_87_0 w_62_0)
    (e_89_0 : w_89_0 = matrixMulScalarLeft w_52_0 w_81_0)
    (e_90_0 : w_90_0 = matrixSub w_88_0 w_89_0)
    (e_92_0 : w_92_0 = matrixAdd w_90_0 w_91_0)
    (e_95_0 : w_95_0 = matrixMul w_51_0 w_94_0)
    (e_96_0 : w_96_0 = matrixAdd w_84_0 w_95_0)
    (e_105_0 : w_105_0 = matrixMulScalarLeft w_102_0 w_104_0)
    (e_106_0 : w_106_0 = (matrixPolynomial [(p4)] : ExactMatrix Q5 8 1 1))
    (e_107_0 : w_107_0 = matrixMulScalarRight w_105_0 w_106_0)
    (e_113_0 : w_113_0 = matrixMulScalarRight w_107_0 w_112_0)
    (e_114_0 : w_114_0 = matrixMulScalarLeft w_102_0 w_77_0)
    (e_115_0 : w_115_0 = matrixSub w_113_0 w_114_0)
    (e_117_0 : w_117_0 = matrixAdd w_115_0 w_116_0)
    (e_120_0 : w_120_0 = matrixMul w_101_0 w_119_0)
    (e_121_0 : w_121_0 = matrixAdd w_80_0 w_120_0)
    (e_129_0 : w_129_0 = matrixMul w_76_0 w_128_0)
    (e_131_0 : w_131_0 = matrixMulScalarLeft w_130_0 w_104_0)
    (e_132_0 : w_132_0 = matrixMulScalarRight w_131_0 w_106_0)
    (e_133_0 : w_133_0 = matrixMulScalarRight w_132_0 w_112_0)
    (e_134_0 : w_134_0 = matrixMulScalarLeft w_102_0 w_126_0)
    (e_135_0 : w_135_0 = matrixSub w_133_0 w_134_0)
    (e_137_0 : w_137_0 = matrixAdd w_135_0 w_136_0)
    (e_140_0 : w_140_0 = matrixMul w_101_0 w_139_0)
    (e_141_0 : w_141_0 = matrixAdd w_129_0 w_140_0)
    (h46_0 : concatRows w_27_0 w_28_0 w_46_0)
    (h48_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_47_0 w_48_0)
    (h49_0 : compactCenteredRebaseRuns 16384 w_48_0 w_49_0)
    (h51_0 : concatRows w_29_0 w_30_0 w_51_0)
    (h68_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_67_0 w_68_0)
    (h69_0 : compactCenteredRebaseRuns 16384 w_68_0 w_69_0)
    (h72_0 : sliceMatrix w_71_0 0 1 0 10 w_72_0)
    (h73_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_72_0 w_73_0)
    (h74_0 : sliceMatrix w_71_0 1 2 0 10 w_74_0)
    (h75_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_74_0 w_75_0)
    (h76_0 : concatRows w_73_0 w_75_0 w_76_0)
    (h78_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_77_0 w_78_0)
    (h79_0 : compactCenteredRebaseRuns 16384 w_78_0 w_79_0)
    (h82_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_81_0 w_82_0)
    (h83_0 : compactCenteredRebaseRuns 16384 w_82_0 w_83_0)
    (h93_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_92_0 w_93_0)
    (h94_0 : compactCenteredRebaseRuns 16384 w_93_0 w_94_0)
    (h97_0 : sliceMatrix w_96_0 0 1 0 10 w_97_0)
    (h98_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_97_0 w_98_0)
    (h99_0 : sliceMatrix w_96_0 1 2 0 10 w_99_0)
    (h100_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_99_0 w_100_0)
    (h101_0 : concatRows w_98_0 w_100_0 w_101_0)
    (h118_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_117_0 w_118_0)
    (h119_0 : compactCenteredRebaseRuns 16384 w_118_0 w_119_0)
    (h122_0 : sliceMatrix w_121_0 0 1 0 8 w_122_0)
    (h123_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_122_0 w_123_0)
    (h124_0 : sliceMatrix w_121_0 1 2 0 8 w_124_0)
    (h125_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_124_0 w_125_0)
    (h127_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_126_0 w_127_0)
    (h128_0 : compactCenteredRebaseRuns 16384 w_127_0 w_128_0)
    (h138_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_137_0 w_138_0)
    (h139_0 : compactCenteredRebaseRuns 16384 w_138_0 w_139_0)
    (h142_0 : sliceMatrix w_141_0 0 1 0 8 w_142_0)
    (h143_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_142_0 w_143_0)
    (h144_0 : sliceMatrix w_141_0 1 2 0 8 w_144_0)
    (h145_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_144_0 w_145_0) :
    ![![w_123_0, w_143_0], ![w_125_0, w_145_0]] = privKey (q := Q6) (q' := Q5) 10 [q0, q1, q2, q3, p4, p5]
      (privKey (q := Q7) (q' := Q6) 12 [q0, q1, q2, q3, p4, p5, p6] ![![w_27_0, w_29_0], ![w_28_0, w_30_0]]
        ![w_47_0, w_81_0] ![w_67_0, w_92_0])
      ![w_77_0, w_126_0] ![w_117_0, w_137_0] := by
  subst e_50_0 e_55_0 e_56_0 e_57_0 e_63_0 e_64_0 e_65_0 e_70_0 e_71_0 e_80_0 e_84_0 e_86_0 e_87_0 e_88_0 e_89_0 e_90_0 e_95_0 e_96_0 e_105_0 e_106_0 e_107_0 e_113_0 e_114_0 e_115_0 e_120_0 e_121_0 e_129_0 e_131_0 e_132_0 e_133_0 e_134_0 e_135_0 e_140_0 e_141_0
  have h1 := privKey_of_pre (q := Q7) (q' := Q6) (m := 12) (m' := 10) (digits := 12)
    (A := ![![w_27_0, w_29_0], ![w_28_0, w_30_0]]) h46_0 h51_0 (top := ![w_47_0, w_81_0]) (bot := ![w_67_0, w_92_0])
    (dT := ![w_48_0, w_82_0]) (rT := ![w_49_0, w_83_0]) (dB := ![w_68_0, w_93_0]) (rB := ![w_69_0, w_94_0])
    (fun j ↦ by fin_cases j; exacts [h48_0, h82_0]) (fun j ↦ by fin_cases j; exacts [h49_0, h83_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h93_0]) (fun j ↦ by fin_cases j; exacts [h69_0, h94_0])
    (M := ![![w_72_0, w_97_0], ![w_74_0, w_99_0]])
    (fun j ↦ by fin_cases j; exacts [h72_0, h97_0]) (fun j ↦ by fin_cases j; exacts [h74_0, h99_0])
    (A' := ![![w_73_0, w_98_0], ![w_75_0, w_100_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h73_0, h98_0, h75_0, h100_0])
  have h2 := privKey_of_pre (q := Q6) (q' := Q5) (m := 10) (m' := 8) (digits := 10)
    (A := ![![w_73_0, w_98_0], ![w_75_0, w_100_0]]) h76_0 h101_0 (top := ![w_77_0, w_126_0]) (bot := ![w_117_0, w_137_0])
    (dT := ![w_78_0, w_127_0]) (rT := ![w_79_0, w_128_0]) (dB := ![w_118_0, w_138_0]) (rB := ![w_119_0, w_139_0])
    (fun j ↦ by fin_cases j; exacts [h78_0, h127_0]) (fun j ↦ by fin_cases j; exacts [h79_0, h128_0])
    (fun j ↦ by fin_cases j; exacts [h118_0, h138_0]) (fun j ↦ by fin_cases j; exacts [h119_0, h139_0])
    (M := ![![w_122_0, w_142_0], ![w_124_0, w_144_0]])
    (fun j ↦ by fin_cases j; exacts [h122_0, h142_0]) (fun j ↦ by fin_cases j; exacts [h124_0, h144_0])
    (A' := ![![w_123_0, w_143_0], ![w_125_0, w_145_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h123_0, h143_0, h125_0, h145_0])
  rw [h2, h1]

end RefreshTopProof.Part_I
