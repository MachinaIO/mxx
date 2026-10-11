import Part_0_1_KeysRoot
import Part_0_1_KeysLuts

/-! The source LUT's keys of the online root of part `0_1` are those the preprocessing computes
(written by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The source LUT's keys of row `1`: the online root's are the preprocessing's. -/
theorem sourceKeyLink {model : HashModel} {params : Stage_online_eval_0_1.Params} {params' : Stage_preprocess_0_1.Params}
    {w_1_0 : Fin 2 → ExactMatrix Q1 8 1 4}
    {w_3_0 : Fin 2 → ExactMatrix Q1 8 1 4}
    {w_5_0 : Fin 2 → ExactMatrix Q1 8 1 4}
    {w_7_0 : Fin 2 → ExactMatrix Q1 8 1 4}
    {w_9_0 : Fin 2 → ExactMatrix Q0 8 1 6}
    {w_10_0 : ByteArray}
    {w_11_0 : Fin 2 → ExactMatrix Q0 8 1 6}
    {w_12_0 : ExactMatrix Q0 8 1 6}
    {w_15_0 : ExactMatrix Q0 8 1 6}
    {w_0_0 : ExactMatrix Q0 8 1 4}
    {w_2_0 : ExactMatrix Q1 8 1 4}
    {w_4_0 : ExactMatrix Q1 8 1 4}
    {w_6_0 : ExactMatrix Q1 8 1 4}
    {w_8_0 : ExactMatrix Q1 8 1 4}
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
    {w_18_6 : ExactMatrix Q1 8 1 4}
    {w_18_7 : ExactMatrix Q1 8 1 4}
    {w_18_8 : ExactMatrix Q1 8 1 4}
    {w_18_9 : ExactMatrix Q1 8 1 4}
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
    {w_38_0 : ExactMatrix Q1 8 2 4}
    {w_39_0 : ExactMatrix Q1 8 12 4}
    {w_40_0 : ExactMatrix Q0 8 12 4}
    {w_41_0 : ExactMatrix Q0 8 3 4}
    {w_42_0 : ExactMatrix Q0 8 1 4}
    {w_43_0 : ExactMatrix Q1 8 1 4}
    {w_44_0 : ExactMatrix Q0 8 1 4}
    {w_45_0 : ExactMatrix Q1 8 1 4}
    {w_46_0 : ExactMatrix Q1 8 2 4}
    {w_47_0 : ExactMatrix Q1 8 12 4}
    {w_48_0 : ExactMatrix Q0 8 12 4}
    {w_49_0 : ExactMatrix Q0 8 3 4}
    {w_50_0 : ExactMatrix Q0 8 1 4}
    {w_51_0 : ExactMatrix Q0 8 1 4}
    {w_52_0 : ExactMatrix Q1 8 1 4}
    {w_53_0 : ExactMatrix Q1 8 2 4}
    {w_54_0 : ExactMatrix Q1 8 12 4}
    {w_55_0 : ExactMatrix Q0 8 12 4}
    {w_56_0 : ExactMatrix Q0 8 3 4}
    {w_57_0 : ExactMatrix Q0 8 1 4}
    {w_58_0 : ExactMatrix Q1 8 1 4}
    {w_59_0 : ExactMatrix Q0 8 1 4}
    {w_60_0 : ExactMatrix Q1 8 1 4}
    {w_61_0 : ExactMatrix Q1 8 2 4}
    {w_62_0 : ExactMatrix Q1 8 12 4}
    {w_63_0 : ExactMatrix Q0 8 12 4}
    {w_64_0 : ExactMatrix Q0 8 3 4}
    {w_65_0 : ExactMatrix Q0 8 1 4}
    {w_66_0 : ExactMatrix Q0 8 1 4}
    {w_67_0 : ExactMatrix Q1 8 1 4}
    {w_68_0 : ExactMatrix Q0 8 1 4}
    {w_69_0 : ExactMatrix Q0 8 1 4}
    {w_70_0 : ExactMatrix Q1 8 1 4}
    {w_71_0 : ExactMatrix Q0 8 1 4}
    {w_72_0 : ExactMatrix Q0 8 1 4}
    {w_73_0 : ExactMatrix Q1 8 1 4}
    {w_74_0 : ExactMatrix Q0 8 1 4}
    {w_75_0 : ExactMatrix Q0 8 1 4}
    {w_76_0 : ExactMatrix Q0 8 1 4}
    {w_77_0 : ExactMatrix Q0 8 1 4}
    {w_78_0 : ExactMatrix Q1 8 1 4}
    {w_79_0 : ExactMatrix Q0 8 1 4}
    {w_80_0 : ExactMatrix Q0 8 1 4}
    {w_81_0 : ExactMatrix Q0 8 1 4}
    {w_82_0 : ExactMatrix Q0 8 1 4}
    {w_83_0 : ExactMatrix Q1 8 1 4}
    (e_0_0 : w_0_0 = (0 : ExactMatrix Q0 8 1 4))
    (e_20_0 : w_20_0 = MxxIR.selectInt (1) [(1), (9)])
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
    (h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × Unit) => Stage_online_eval_0_1.sequential_generatedRoot_18 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (1)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))))
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
    (h29_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_12_0 w_29_0)
    (h30_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_13_0 w_30_0)
    (h31_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_14_0 w_31_0)
    (h30_0_1 : concatRows w_29_0 w_30_0 w_32_concat_1)
    (h32_0 : concatRows w_32_concat_1 w_31_0 w_32_0)
    (h33_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_15_0 w_33_0)
    (h34_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_16_0 w_34_0)
    (h35_0 : ringAutomorphismRuns (MxxIR.selectInt (1) [(1), (9)]) w_17_0 w_35_0)
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
    (h83_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_82_0 w_83_0)
    {pw_168_0 : ByteArray}
    {pw_179_0 : ExactMatrix Q0 8 1 4}
    {pw_180_0 : ExactMatrix Q0 8 1 6}
    {pw_181_0 : ExactMatrix Q0 8 1 6}
    {pw_182_0 : Fin 2 → ExactMatrix Q1 8 1 4}
    {pw_182_1 : Fin 2 → ExactMatrix Q1 8 1 4}
    {pw_182_2 : Fin 2 → ExactMatrix Q1 8 1 4}
    {pw_182_3 : Fin 2 → ExactMatrix Q1 8 1 4}
    {pw_183_0 : ExactMatrix Q0 8 1 6}
    {pw_184_0 : ExactMatrix Q0 8 1 6}
    {pw_185_0 : ExactMatrix Q0 8 1 4}
    {pw_185_1 : ExactMatrix Q0 8 1 4}
    {pw_185_2 : ExactMatrix Q0 8 1 4}
    {pw_185_3 : ExactMatrix Q0 8 1 4}
    {pw_186_0 : ExactMatrix Q1 8 1 4}
    {pw_342_0 : ExactMatrix Q1 8 1 4}
    (pe_179_0 : pw_179_0 = (0 : ExactMatrix Q0 8 1 4))
    (ph185_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × Unit) => Stage_preprocess_0_1.sequential_generatedRoot_185 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_168_0, (pw_180_0, (pw_181_0, (pw_182_0, (pw_183_0, (pw_184_0, (pw_182_2, (pw_182_1, (pw_182_3, ()))))))))))))) next) (Int.toNat (2)) (pw_179_0, (pw_179_0, (pw_179_0, (pw_179_0, ())))) (pw_185_0, (pw_185_1, (pw_185_2, (pw_185_3, ())))))
    (ph186_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) pw_185_1 pw_186_0)
    (ph342_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) pw_185_3 pw_342_0)
    (hk : w_10_0 = pw_168_0)
    (ha0 : w_13_0 = pw_180_0) (ha1 : w_14_0 = pw_181_0) (ha2 : w_16_0 = pw_183_0) (ha3 : w_17_0 = pw_184_0)
    (hf0 : w_1_0 = pw_182_0) (hf1 : w_3_0 = pw_182_1) (hf2 : w_5_0 = pw_182_2) (hf3 : w_7_0 = pw_182_3) :
    w_67_0 = pw_186_0 ∧ w_73_0 = pw_342_0 := by
  have a := keyRootLut18 (model := model) (e_0_0 := e_0_0) (e_20_0 := e_20_0) (e_41_0 := e_41_0) (e_49_0 := e_49_0) (e_51_0 := e_51_0) (e_56_0 := e_56_0) (e_64_0 := e_64_0) (e_66_0 := e_66_0) (e_69_0 := e_69_0) (e_72_0 := e_72_0) (e_76_0 := e_76_0) (e_77_0 := e_77_0) (e_81_0 := e_81_0) (e_82_0 := e_82_0) (h2_0 := h2_0) (h4_0 := h4_0) (h6_0 := h6_0) (h8_0 := h8_0) (h18_9 := h18_9) (h19_0 := h19_0) (h21_0 := h21_0) (h22_0 := h22_0) (h21_0_1 := h21_0_1) (h23_0 := h23_0) (h24_0 := h24_0) (h25_0 := h25_0) (h26_0 := h26_0) (h25_0_1 := h25_0_1) (h27_0 := h27_0) (h28_0 := h28_0) (h29_0 := h29_0) (h30_0 := h30_0) (h31_0 := h31_0) (h30_0_1 := h30_0_1) (h32_0 := h32_0) (h33_0 := h33_0) (h34_0 := h34_0) (h35_0 := h35_0) (h34_0_1 := h34_0_1) (h36_0 := h36_0) (h37_0 := h37_0) (h38_0 := h38_0) (h39_0 := h39_0) (h40_0 := h40_0) (h42_0 := h42_0) (h43_0 := h43_0) (h44_0 := h44_0) (h45_0 := h45_0) (h46_0 := h46_0) (h47_0 := h47_0) (h48_0 := h48_0) (h50_0 := h50_0) (h52_0 := h52_0) (h53_0 := h53_0) (h54_0 := h54_0) (h55_0 := h55_0) (h57_0 := h57_0) (h58_0 := h58_0) (h59_0 := h59_0) (h60_0 := h60_0) (h61_0 := h61_0) (h62_0 := h62_0) (h63_0 := h63_0) (h65_0 := h65_0) (h67_0 := h67_0) (h68_0 := h68_0) (h70_0 := h70_0) (h71_0 := h71_0) (h73_0 := h73_0) (h74_0 := h74_0) (h75_0 := h75_0) (h78_0 := h78_0) (h79_0 := h79_0) (h80_0 := h80_0) (h83_0 := h83_0)
  have b := preKeyLoop_sequential_generatedRoot_185 (by decide) ph185_3
  subst hk ha0 ha1 ha2 ha3 hf0 hf1 hf2 hf3 pe_179_0
  rw [show ((0 : ExactMatrix Q0 N 1 4), (0 : ExactMatrix Q0 N 1 4), (0 : ExactMatrix Q0 N 1 4),
    (0 : ExactMatrix Q0 N 1 4), ()) = (0 : Keys Q0 4) from rfl, zero_add,
    show Int.toNat 2 = 2 from rfl] at b
  rw [← b] at a
  simp only [keyMap, Prod.mk.injEq] at a
  rw [blockModSwitchRuns_eq ph186_0, blockModSwitchRuns_eq ph342_0]
  exact ⟨a.2.1, a.2.2.2.1⟩

end RefreshTopProof.Part_0_1
