import Part_0_0_RootLuts
import Part_0_0_Readout

/-! The segments of the online root (written by `generator/gen_online_root.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The table `[X, 1]` of the source LUT. -/
noncomputable def sourceTable (k : Fin 2) : ErrorPoly N := ![Xe 1, 1] k

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The root LUT `18`. -/
theorem rootLut18 {model : HashModel} {params : Stage_online_eval_0_0.Params}
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
    (h18_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q0 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × ExactMatrix Q1 8 1 4 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_18 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (1)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))))
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
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![w_13_0, w_16_0], ![w_14_0, w_17_0]] (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * Xe (n := N) z * unitPair sp j) • g) ![w_12_0, w_15_0] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 2, Rel sm (lutA model w_10_0 Q0 6 0 (sigma2 i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma2 i).toNat (sigma2_odd i) sm) j) • g) ![w_9_0 i, w_11_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 4} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((2 : Nat) : ExactPoly Q1 N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 2, ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma2 i).toNat (sigma2_odd i) sp) •
          ![![w_1_0, w_3_0], ![w_5_0, w_7_0]] 0 j i + ![![w_1_0, w_3_0], ![w_5_0, w_7_0]] 1 j i =
        (lutCoefficient2 Q1 inverse (fun k ↦ reducePoly Q1 N (sourceTable k)) (sigma2 i).toNat *
          reducePoly Q1 N ((((q2 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 4 (e j)) :
    Rel sm ![![w_52_0, w_67_0], ![w_70_0, w_73_0]]
      (fun j ↦ reducePoly Q1 N ((((q2 : Nat) : Int) : ErrorPoly N) * (Xe (z - z % 2) * sourceTable ⟨z % 2, Nat.mod_lt _ (by norm_num)⟩) * unitPair sp j) • gq)
      ![w_78_0, w_83_0]
      (((1 * lutBranchBound18 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * (q3 / 2) + q3 / 2))) + (N + 2) * (q3 / 2)) / q3 + 2 * (N * 1 * BB)) := by
  subst e_0_0 e_20_0 e_41_0 e_49_0 e_51_0 e_56_0 e_64_0 e_66_0 e_69_0 e_72_0 e_76_0 e_77_0 e_81_0 e_82_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h2_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h4_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h6_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h8_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 1 = 1 from rfl] at h18_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop18 (fam := fun k j ↦ ![![w_1_0, w_3_0], ![w_5_0, w_7_0]] k j)
    (c := ![w_12_0, w_15_0]) (A := ![![w_13_0, w_16_0], ![w_14_0, w_17_0]])
    (final := (w_18_0, w_18_1, w_18_2, w_18_3, w_18_4, w_18_5, w_18_6, w_18_7, w_18_8, w_18_9, ())) hsm hg' hg hinput hhelp (count := 1) (by decide) h18_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 1) h19_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 1) h24_0
  rw [hw19] at h21_0_1
  rw [hw24] at h25_0_1
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
  have hhelp3 := hhelp 1
  have hfin := lut_round2 (q := Q0) (q' := Q1) (p := q3) (m := 6) (m' := 4) q0_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout1 layoutQ1 rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    h29_0 h30_0 h31_0 h33_0 h34_0 h35_0 h30_0_1 h32_0 h34_0_1 h36_0 h37_0
    (Xb := ![w_38_0, w_53_0]) (fun j ↦ by fin_cases j; exacts [h38_0, h53_0])
    (D' := ![w_39_0, w_54_0]) (fun j ↦ by fin_cases j; exacts [h39_0, h54_0])
    (D := ![w_40_0, w_55_0]) (fun j ↦ by fin_cases j; exacts [h40_0, h55_0])
    (M := ![![w_42_0, w_57_0], ![w_44_0, w_59_0]])
    (fun j ↦ by fin_cases j; exacts [h42_0, h57_0])
    (fun j ↦ by fin_cases j; exacts [h44_0, h59_0])
    (Asw := ![![w_43_0, w_58_0], ![w_45_0, w_60_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h43_0, h58_0, h45_0, h60_0])
    (Z := ![w_46_0, w_61_0]) (fun j ↦ by fin_cases j; exacts [h46_0, h61_0])
    (E' := ![w_47_0, w_62_0]) (fun j ↦ by fin_cases j; exacts [h47_0, h62_0])
    (E := ![w_48_0, w_63_0]) (fun j ↦ by fin_cases j; exacts [h48_0, h63_0])
    h21_0_1 h23_0 h25_0_1 h27_0 h28_0
    (Y := ![w_74_0, w_79_0]) (X := ![w_75_0, w_80_0])
    (K := ![![w_50_0, w_65_0], ![w_68_0, w_71_0]])
    (fun j ↦ by fin_cases j; exacts [h74_0, h79_0])
    (fun j ↦ by fin_cases j; exacts [h75_0, h80_0])
    (fun j ↦ by fin_cases j; exacts [h50_0, h65_0])
    (fun j ↦ by fin_cases j; exacts [h68_0, h71_0])
    (T := sourceTable) (Δ' := (((q2 : Nat) : Int) : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_52_0, w_67_0], ![w_70_0, w_73_0]])
    (R' := ![w_78_0, w_83_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h52_0, h67_0, h70_0, h73_0])
    (fun j ↦ by fin_cases j; exacts [h78_0, h83_0])
  exact hfin

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The root LUT `186`. -/
theorem rootLut186 {model : HashModel} {params : Stage_online_eval_0_0.Params}
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
    (e_188_0 : w_188_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
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
    (h186_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_186 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_98_0, (w_92_0, (w_99_0, (w_169_0, (w_171_0, (w_173_0, (w_181_0, (w_183_0, (w_185_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, ())))))))))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_186_4, (w_186_5, (w_186_6, (w_186_7, (w_186_8, (w_186_9, ())))))))))))
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
    (h197_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_169_0 w_197_0)
    (h198_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_171_0 w_198_0)
    (h199_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_173_0 w_199_0)
    (h198_0_1 : concatRows w_197_0 w_198_0 w_200_concat_1)
    (h200_0 : concatRows w_200_concat_1 w_199_0 w_200_0)
    (h201_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_181_0 w_201_0)
    (h202_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_183_0 w_202_0)
    (h203_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_185_0 w_203_0)
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
    (h251_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_250_0 w_251_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![w_171_0, w_183_0], ![w_173_0, w_185_0]] (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g) ![w_169_0, w_181_0] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![w_98_0 i, w_99_0 i] Bh)
    {gq : ExactMatrix Q0 N 1 6} {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 0 j i + ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q0 N 1 6 (e j)) :
    Rel sm ![![w_225_0, w_248_0], ![w_228_0, w_251_0]]
      (fun j ↦ reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * ((((-1 : Int) ^ (z / 4) : Int) : ErrorPoly N) * readoutTable ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩) * unitPair sp j) • gq)
      ![w_222_0, w_245_0]
      (((3 * lutBranchBound186 B Bh + ((8 + 8) * N * Bh * 16384 + (8 + 8) * N * B * 16384 +
        (N * 1 * (p4 / 2) + p4 / 2))) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BB)) := by
  subst e_127_0 e_188_0 e_209_0 e_217_0 e_220_0 e_221_0 e_224_0 e_227_0 e_232_0 e_240_0 e_243_0 e_244_0 e_247_0 e_250_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h128_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h129_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h130_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h131_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 3 = 3 from rfl] at h186_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop186 (fam := fun k j ↦ ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] k j)
    (c := ![w_169_0, w_181_0]) (A := ![![w_171_0, w_183_0], ![w_173_0, w_185_0]])
    (final := (w_186_0, w_186_1, w_186_2, w_186_3, w_186_4, w_186_5, w_186_6, w_186_7, w_186_8, w_186_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h186_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h187_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h192_0
  rw [hw19] at h189_0_1
  rw [hw24] at h193_0_1
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
  have hhelp3 := hhelp 3
  have hfin := lut_round (q := Q5) (q' := Q0) (p := p4) (m := 8) (m' := 6) q5_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout2 layoutQ0 rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    h197_0 h198_0 h199_0 h201_0 h202_0 h203_0 h198_0_1 h200_0 h202_0_1 h204_0 h205_0
    (Xb := ![w_206_0, w_229_0]) (fun j ↦ by fin_cases j; exacts [h206_0, h229_0])
    (D' := ![w_207_0, w_230_0]) (fun j ↦ by fin_cases j; exacts [h207_0, h230_0])
    (D := ![w_208_0, w_231_0]) (fun j ↦ by fin_cases j; exacts [h208_0, h231_0])
    (M := ![![w_210_0, w_233_0], ![w_212_0, w_235_0]])
    (fun j ↦ by fin_cases j; exacts [h210_0, h233_0])
    (fun j ↦ by fin_cases j; exacts [h212_0, h235_0])
    (Asw := ![![w_211_0, w_234_0], ![w_213_0, w_236_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h211_0, h234_0, h213_0, h236_0])
    (Z := ![w_214_0, w_237_0]) (fun j ↦ by fin_cases j; exacts [h214_0, h237_0])
    (E' := ![w_215_0, w_238_0]) (fun j ↦ by fin_cases j; exacts [h215_0, h238_0])
    (E := ![w_216_0, w_239_0]) (fun j ↦ by fin_cases j; exacts [h216_0, h239_0])
    h189_0_1 h191_0 h193_0_1 h195_0 h196_0
    (Y := ![w_218_0, w_241_0]) (X := ![w_219_0, w_242_0])
    (K := ![![w_223_0, w_246_0], ![w_226_0, w_249_0]])
    (fun j ↦ by fin_cases j; exacts [h218_0, h241_0])
    (fun j ↦ by fin_cases j; exacts [h219_0, h242_0])
    (fun j ↦ by fin_cases j; exacts [h223_0, h246_0])
    (fun j ↦ by fin_cases j; exacts [h226_0, h249_0])
    (T := readoutTable) (Δ' := (((q3 : Nat) : Int) : ErrorPoly N)) hinverse
    (fun i ↦ hswitch i) (K' := ![![w_225_0, w_248_0], ![w_228_0, w_251_0]])
    (R' := ![w_222_0, w_245_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h225_0, h248_0, h228_0, h251_0])
    (fun j ↦ by fin_cases j; exacts [h222_0, h245_0])
  exact hfin

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The root LUT `252`. -/
theorem rootLut252 {model : HashModel} {params : Stage_online_eval_0_0.Params}
    {w_87_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_88_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_89_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_90_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_91_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_92_0 : ByteArray}
    {w_93_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_122_0 : ExactMatrix Q0 8 1 6}
    {w_123_0 : ExactMatrix Q1 8 1 6}
    {w_124_0 : ExactMatrix Q1 8 1 6}
    {w_125_0 : ExactMatrix Q1 8 1 6}
    {w_126_0 : ExactMatrix Q1 8 1 6}
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
    {w_252_6 : ExactMatrix Q1 8 1 6}
    {w_252_7 : ExactMatrix Q1 8 1 6}
    {w_252_8 : ExactMatrix Q1 8 1 6}
    {w_252_9 : ExactMatrix Q1 8 1 6}
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
    {w_272_0 : ExactMatrix Q1 8 2 6}
    {w_273_0 : ExactMatrix Q1 8 12 6}
    {w_274_0 : ExactMatrix Q0 8 12 6}
    {w_275_0 : ExactMatrix Q0 8 3 6}
    {w_276_0 : ExactMatrix Q0 8 1 6}
    {w_277_0 : ExactMatrix Q1 8 1 6}
    {w_278_0 : ExactMatrix Q0 8 1 6}
    {w_279_0 : ExactMatrix Q1 8 1 6}
    {w_280_0 : ExactMatrix Q1 8 2 6}
    {w_281_0 : ExactMatrix Q1 8 12 6}
    {w_282_0 : ExactMatrix Q0 8 12 6}
    {w_283_0 : ExactMatrix Q0 8 3 6}
    {w_284_0 : ExactMatrix Q0 8 1 6}
    {w_285_0 : ExactMatrix Q0 8 1 6}
    {w_286_0 : ExactMatrix Q1 8 2 6}
    {w_287_0 : ExactMatrix Q1 8 12 6}
    {w_288_0 : ExactMatrix Q0 8 12 6}
    {w_289_0 : ExactMatrix Q0 8 3 6}
    {w_290_0 : ExactMatrix Q0 8 1 6}
    {w_291_0 : ExactMatrix Q1 8 1 6}
    {w_292_0 : ExactMatrix Q0 8 1 6}
    {w_293_0 : ExactMatrix Q1 8 1 6}
    {w_294_0 : ExactMatrix Q1 8 2 6}
    {w_295_0 : ExactMatrix Q1 8 12 6}
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
    {w_306_0 : ExactMatrix Q0 8 1 6}
    {w_307_0 : ExactMatrix Q0 8 1 6}
    {w_308_0 : ExactMatrix Q0 8 1 6}
    {w_309_0 : ExactMatrix Q0 8 1 6}
    {w_310_0 : ExactMatrix Q0 8 1 6}
    {w_311_0 : ExactMatrix Q0 8 1 6}
    (e_122_0 : w_122_0 = (0 : ExactMatrix Q0 8 1 6))
    (e_254_0 : w_254_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
    (e_275_0 : w_275_0 = matrixMul w_271_0 w_274_0)
    (e_283_0 : w_283_0 = matrixMul w_262_0 w_282_0)
    (e_285_0 : w_285_0 = matrixAdd w_252_2 w_284_0)
    (e_289_0 : w_289_0 = matrixMul w_271_0 w_288_0)
    (e_297_0 : w_297_0 = matrixMul w_262_0 w_296_0)
    (e_299_0 : w_299_0 = matrixAdd w_252_3 w_298_0)
    (e_301_0 : w_301_0 = matrixAdd w_252_4 w_300_0)
    (e_303_0 : w_303_0 = matrixAdd w_252_5 w_302_0)
    (e_306_0 : w_306_0 = matrixAdd w_304_0 w_305_0)
    (e_307_0 : w_307_0 = matrixAdd w_252_0 w_306_0)
    (e_310_0 : w_310_0 = matrixAdd w_308_0 w_309_0)
    (e_311_0 : w_311_0 = matrixAdd w_252_1 w_310_0)
    (h123_0 : familyGetStatic w_87_0 (0) w_123_0)
    (h124_0 : familyGetStatic w_88_0 (0) w_124_0)
    (h125_0 : familyGetStatic w_89_0 (0) w_125_0)
    (h126_0 : familyGetStatic w_90_0 (0) w_126_0)
    (h252_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_252 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_91_0, (w_92_0, (w_93_0, (w_222_0, (w_225_0, (w_228_0, (w_245_0, (w_248_0, (w_251_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, ())))))))))) (w_252_0, (w_252_1, (w_252_2, (w_252_3, (w_252_4, (w_252_5, (w_252_6, (w_252_7, (w_252_8, (w_252_9, ())))))))))))
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
    (h263_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_222_0 w_263_0)
    (h264_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_225_0 w_264_0)
    (h265_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_228_0 w_265_0)
    (h264_0_1 : concatRows w_263_0 w_264_0 w_266_concat_1)
    (h266_0 : concatRows w_266_concat_1 w_265_0 w_266_0)
    (h267_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_245_0 w_267_0)
    (h268_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_248_0 w_268_0)
    (h269_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_251_0 w_269_0)
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
    (h309_0 : sliceMatrix w_289_0 0 1 0 6 w_309_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g)
    {B : Nat}
    (hinput : Rel sm ![![w_225_0, w_248_0], ![w_228_0, w_251_0]] (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * Xe (n := N) (2 * z) * unitPair sp j) • g) ![w_222_0, w_245_0] B)
    {Bh : Nat} (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g) ![w_91_0 i, w_93_0 i] Bh) :
    Rel sm ![![w_285_0, w_299_0], ![w_301_0, w_303_0]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * z))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 0 j i + ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 1 j i)))
      ![w_307_0, w_311_0] ((3 * lutBranchBound252 B Bh + ((6 + 6) * N * Bh * 16384 + (6 + 6) * N * B * 16384 +
        (N * 1 * (q3 / 2) + q3 / 2)))) := by
  subst e_122_0 e_254_0 e_275_0 e_283_0 e_285_0 e_289_0 e_297_0 e_299_0 e_301_0 e_303_0 e_306_0 e_307_0 e_310_0 e_311_0
  obtain ⟨_, hwa⟩ := familyGet_eq (k := 0) h123_0
  obtain ⟨_, hwb⟩ := familyGet_eq (k := 0) h124_0
  obtain ⟨_, hwc⟩ := familyGet_eq (k := 0) h125_0
  obtain ⟨_, hwd⟩ := familyGet_eq (k := 0) h126_0
  subst hwa hwb hwc hwd
  rw [show Int.toNat 3 = 3 from rfl] at h252_9
  obtain ⟨hloop, hloopSwitch⟩ := lutLoop252 (fam := fun k j ↦ ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] k j)
    (c := ![w_222_0, w_245_0]) (A := ![![w_225_0, w_248_0], ![w_228_0, w_251_0]])
    (final := (w_252_0, w_252_1, w_252_2, w_252_3, w_252_4, w_252_5, w_252_6, w_252_7, w_252_8, w_252_9, ())) hsm hg' hg hinput hhelp (count := 3) (by decide) h252_9
  obtain ⟨_, hw19⟩ := familyGet_eq (k := 3) h253_0
  obtain ⟨_, hw24⟩ := familyGet_eq (k := 3) h258_0
  rw [hw19] at h255_0_1
  rw [hw24] at h259_0_1
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
  have hhelp3 := hhelp 3
  have hsum := lut_branches (q := Q0) (q' := Q1) (p := q3) (m := 6) (m' := 6) q0_eq (by norm_num)
    (by norm_num) (by norm_num) Backend.layout1 layoutQ1 rfl hsm hg' hg hinput hloop hloopSwitch hhelp3
    h263_0 h264_0 h265_0 h267_0 h268_0 h269_0 h264_0_1 h266_0 h268_0_1 h270_0 h271_0
    (Xb := ![w_272_0, w_286_0]) (fun j ↦ by fin_cases j; exacts [h272_0, h286_0])
    (D' := ![w_273_0, w_287_0]) (fun j ↦ by fin_cases j; exacts [h273_0, h287_0])
    (D := ![w_274_0, w_288_0]) (fun j ↦ by fin_cases j; exacts [h274_0, h288_0])
    (M := ![![w_276_0, w_290_0], ![w_278_0, w_292_0]])
    (fun j ↦ by fin_cases j; exacts [h276_0, h290_0])
    (fun j ↦ by fin_cases j; exacts [h278_0, h292_0])
    (Asw := ![![w_277_0, w_291_0], ![w_279_0, w_293_0]])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h277_0, h291_0, h279_0, h293_0])
    (Z := ![w_280_0, w_294_0]) (fun j ↦ by fin_cases j; exacts [h280_0, h294_0])
    (E' := ![w_281_0, w_295_0]) (fun j ↦ by fin_cases j; exacts [h281_0, h295_0])
    (E := ![w_282_0, w_296_0]) (fun j ↦ by fin_cases j; exacts [h282_0, h296_0])
    h255_0_1 h257_0 h259_0_1 h261_0 h262_0
    (Y := ![w_304_0, w_308_0]) (X := ![w_305_0, w_309_0])
    (K := ![![w_284_0, w_298_0], ![w_300_0, w_302_0]])
    (fun j ↦ by fin_cases j; exacts [h304_0, h308_0])
    (fun j ↦ by fin_cases j; exacts [h305_0, h309_0])
    (fun j ↦ by fin_cases j; exacts [h284_0, h298_0])
    (fun j ↦ by fin_cases j; exacts [h300_0, h302_0])
  convert hsum using 1
  all_goals first
    | rfl
    | (funext a j; fin_cases a <;> fin_cases j <;> rfl)
    | (funext j; fin_cases j <;> rfl)

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encoding of one moved to `Q5` by two private multiplications with the unit offset RHS:
the input of the binary offset's readout LUT. -/
theorem rootOffset {model : HashModel}
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
    {w_141_0 : ExactMatrix Q7 8 1 10}
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
    {w_154_0 : ExactMatrix Q7 8 1 10}
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
    {w_168_0 : ExactMatrix Q6 8 1 8}
    {w_169_0 : ExactMatrix Q5 8 1 8}
    {w_170_0 : ExactMatrix Q6 8 1 8}
    {w_171_0 : ExactMatrix Q5 8 1 8}
    {w_172_0 : ExactMatrix Q6 8 1 8}
    {w_173_0 : ExactMatrix Q5 8 1 8}
    {w_174_0 : ExactMatrix Q5 8 1 8}
    {w_176_0 : ExactMatrix Q5 8 2 8}
    {w_177_0 : ExactMatrix Q5 8 20 8}
    {w_178_0 : ExactMatrix Q6 8 20 8}
    {w_179_0 : ExactMatrix Q6 8 3 8}
    {w_180_0 : ExactMatrix Q6 8 1 8}
    {w_181_0 : ExactMatrix Q5 8 1 8}
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
    (h141_0 : sliceMatrix w_140_0 0 1 0 10 w_141_0)
    (h142_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_141_0 w_142_0)
    (h143_0 : sliceMatrix w_140_0 1 2 0 10 w_143_0)
    (h144_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_143_0 w_144_0)
    (h145_0 : sliceMatrix w_140_0 2 3 0 10 w_145_0)
    (h146_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_145_0 w_146_0)
    (h144_0_1 : concatRows w_142_0 w_144_0 w_147_concat_1)
    (h147_0 : concatRows w_147_concat_1 w_146_0 w_147_0)
    (h150_0 : concatRows w_148_0 w_149_0 w_150_0)
    (h151_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_150_0 w_151_0)
    (h152_0 : compactCenteredRebaseRuns 16384 w_151_0 w_152_0)
    (h154_0 : sliceMatrix w_153_0 0 1 0 10 w_154_0)
    (h155_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_154_0 w_155_0)
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
    (h168_0 : sliceMatrix w_167_0 0 1 0 8 w_168_0)
    (h169_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_168_0 w_169_0)
    (h170_0 : sliceMatrix w_167_0 1 2 0 8 w_170_0)
    (h171_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_170_0 w_171_0)
    (h172_0 : sliceMatrix w_167_0 2 3 0 8 w_172_0)
    (h173_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_172_0 w_173_0)
    (h176_0 : concatRows w_174_0 w_175_0 w_176_0)
    (h177_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_176_0 w_177_0)
    (h178_0 : compactCenteredRebaseRuns 16384 w_177_0 w_178_0)
    (h180_0 : sliceMatrix w_179_0 0 1 0 8 w_180_0)
    (h181_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_180_0 w_181_0)
    (h182_0 : sliceMatrix w_179_0 1 2 0 8 w_182_0)
    (h183_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_182_0 w_183_0)
    (h184_0 : sliceMatrix w_179_0 2 3 0 8 w_184_0)
    (h185_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_184_0 w_185_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {B : Nat}
    (hinput : Rel sm ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_108_0, w_112_0] B)
    {g5' : ExactMatrix Q5 N 1 10} {g6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {eB : Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![w_135_0, w_148_0] j +
        ![w_136_0, w_149_0] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {h5 : ExactMatrix Q5 N 1 8}
    {eB' : Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![w_162_0, w_174_0] j +
        ![w_163_0, w_175_0] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB') :
    Rel sm ![![w_171_0, w_183_0], ![w_173_0, w_185_0]]
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5)
      ![w_169_0, w_181_0]
      ((((10 + 10) * N * ((((12 + 12) * N * B * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) *
        16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB')) := by
  subst e_140_0 e_153_0 e_167_0 e_179_0
  have hone : polyNorm (1 : ErrorPoly N) ≤ 1 := by simpa using polyNorm_Xe hN1 0
  have h1 := mul_private_down (q := Q7) (q' := Q6) (p := p6) (m := 12) (m' := 10) (bound := 16384)
    hN q7_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout4 layoutQ6 rfl hg6' hg7 hsm
    hinput
    h106_0 h132_0 h110_0 h133_0 h134_0
    (top := ![w_135_0, w_148_0]) (bot := ![w_136_0, w_149_0])
    (Xb := ![w_137_0, w_150_0]) (fun j ↦ by fin_cases j; exacts [h137_0, h150_0])
    (D' := ![w_138_0, w_151_0]) (fun j ↦ by fin_cases j; exacts [h138_0, h151_0])
    (D := ![w_139_0, w_152_0]) (fun j ↦ by fin_cases j; exacts [h139_0, h152_0])
    hrhs heB hone
    (X := ![w_141_0, w_154_0]) (M := ![![w_143_0, w_156_0], ![w_145_0, w_158_0]])
    (fun j ↦ by fin_cases j; exacts [h141_0, h154_0]) (fun j ↦ by fin_cases j; exacts [h143_0, h156_0]) (fun j ↦ by fin_cases j; exacts [h145_0, h158_0])
    (c' := ![w_142_0, w_155_0]) (A' := ![![w_144_0, w_157_0], ![w_146_0, w_159_0]])
    (fun j ↦ by fin_cases j; exacts [h142_0, h155_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h144_0, h157_0, h146_0, h159_0])
  have h1' := h1.target_eq (t' := fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6) (by
    funext j
    rw [map_one, one_smul])
  have h2 := mul_private_down (q := Q6) (q' := Q5) (p := p5) (m := 10) (m' := 8) (bound := 16384)
    hN q6_eq (by norm_num) (by norm_num) (by norm_num) Backend.layout3 layoutQ5 rfl hg5' hg6 hsm h1'
    h144_0_1 h147_0 h157_0_1 h160_0 h161_0
    (top := ![w_162_0, w_174_0]) (bot := ![w_163_0, w_175_0])
    (Xb := ![w_164_0, w_176_0]) (fun j ↦ by fin_cases j; exacts [h164_0, h176_0])
    (D' := ![w_165_0, w_177_0]) (fun j ↦ by fin_cases j; exacts [h165_0, h177_0])
    (D := ![w_166_0, w_178_0]) (fun j ↦ by fin_cases j; exacts [h166_0, h178_0])
    hrhs' heB' hone
    (X := ![w_168_0, w_180_0]) (M := ![![w_170_0, w_182_0], ![w_172_0, w_184_0]])
    (fun j ↦ by fin_cases j; exacts [h168_0, h180_0]) (fun j ↦ by fin_cases j; exacts [h170_0, h182_0]) (fun j ↦ by fin_cases j; exacts [h172_0, h184_0])
    (c' := ![w_169_0, w_181_0]) (A' := ![![w_171_0, w_183_0], ![w_173_0, w_185_0]])
    (fun j ↦ by fin_cases j; exacts [h169_0, h181_0])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [h171_0, h183_0, h173_0, h185_0])
  convert h2 using 2 with j
  rw [map_one, one_smul]

end RefreshTopProof.Part_0_0
